"""The feature qualification gate — specification section 7.

Runs on **target-training images only**.  No test image, no learned metric, no
held-out quality signal.

**G7 and G8 run first.**  Both are cheap, and between them they test the two
assumptions the entire branch rests on: that the trunk is sensitive to the
bands the foundation is weakest in, and that the chosen levels carry distinct
information.  If either fails, the arm is cancelled before a bank is built or a
coefficient calibrated.
"""

from __future__ import annotations

import torch

from ..stage_cap.diagnostics import haar_transform
from .calibration import _token_pc1_share
from .config import FeatureConfig, QualificationConfig
from .features import encode

Gate = tuple[str, bool, str, dict]

BANDS = ("LL", "LH", "HL", "HH")


def _band_mask(size: int, band: str, device) -> torch.Tensor:
    half = size // 2
    mask = torch.zeros(size, size, device=device)
    rows = slice(0, half) if band[0] == "L" else slice(half, size)
    columns = slice(0, half) if band[1] == "L" else slice(half, size)
    mask[rows, columns] = 1.0
    return mask


def _inverse_haar(coefficients: torch.Tensor) -> torch.Tensor:
    """Map Haar coefficients back to pixels.

    Rather than write a second transform to disagree with the first, the
    analysis matrix is recovered by pushing the standard basis through
    ``haar_transform`` once.  Row ``i`` of that result is ``H e_i``, i.e. column
    ``i`` of ``H``, so the assembled matrix is ``H^T``.  Haar is orthonormal, so
    ``H^-1 = H^T`` and the synthesis of a coefficient row vector ``c`` is
    ``c @ H`` -- which is ``c @ transformed.T``, **not** ``c @ transformed``.

    Getting that transpose wrong is not a small error.  It leaves the operation
    applying the forward transform a second time: measured round-trip error 4.31
    instead of 5e-7, and band-masked perturbations leaking 71-74% of their
    energy outside the band they were supposed to isolate -- which would have
    made G7, the gate that decides whether this arm proceeds at all, report four
    numbers that look like band sensitivities and are not.
    """
    size = coefficients.shape[-1]
    basis = torch.eye(size * size, device=coefficients.device).reshape(
        size * size, 1, size, size
    )
    analysis = haar_transform(basis).reshape(size * size, size * size)
    flat = coefficients.reshape(len(coefficients), -1)
    return (flat @ analysis.T).reshape(coefficients.shape)


def band_sensitivity(
    trunk,
    images: torch.Tensor,
    t_f: float,
    config: FeatureConfig,
    normalization,
    *,
    energy: float = 0.5,
    seed: int = 41,
) -> dict:
    """G7: feature response per Haar band, relative to a raw-pixel control.

    The upper bound rejects the hypersensitivity of the Phases 17-18 pretrained
    ResNet.  **The lower bound is the point**: the trunk allocates capacity
    under MSE, which underweights exactly the bands an MSE-trained generator is
    weakest in, and a map with zero high-frequency sensitivity passes a
    one-sided check trivially.
    """
    generator = torch.Generator().manual_seed(seed)
    size = images.shape[-1]
    result: dict[str, float] = {}
    base = encode(trunk, images, t_f, config, normalization, generator=generator)
    for band in BANDS:
        mask = _band_mask(size, band, images.device)
        raw = torch.randn(images.shape, generator=generator).to(images.device)
        coefficients = haar_transform(raw) * mask
        perturbation = _inverse_haar(coefficients)
        scale = energy / perturbation.reshape(len(perturbation), -1).norm(
            dim=1, keepdim=True
        ).clamp_min(1e-12)
        perturbation = perturbation * scale.reshape(-1, 1, 1, 1)

        probe_generator = torch.Generator().manual_seed(seed)
        shifted = encode(
            trunk,
            images + perturbation,
            t_f,
            config,
            normalization,
            generator=probe_generator,
        )
        feature_delta = sum(
            float((shifted[name] - base[name]).norm()) for name in base
        ) / len(base)
        raw_delta = float(perturbation.norm())
        result[f"rho_{band}"] = feature_delta / max(raw_delta, 1e-12)
    smallest = min(result.values())
    if smallest > 0:
        for band in BANDS:
            result[f"relative_{band}"] = result[f"rho_{band}"] / smallest
    return result


def gate_band_sensitivity(profile: dict, config: QualificationConfig) -> Gate:
    values = {band: profile[f"rho_{band}"] for band in BANDS}
    reference = sum(values.values()) / len(values)
    ratios = {band: value / reference for band, value in values.items()}
    ok = all(
        config.band_sensitivity_low <= ratio <= config.band_sensitivity_high
        for ratio in ratios.values()
    )
    detail = ", ".join(f"{band}={ratios[band]:.2f}" for band in BANDS)
    return (
        "G7 per-band sensitivity",
        ok,
        f"band/mean response {detail} (need "
        f"[{config.band_sensitivity_low}, {config.band_sensitivity_high}])",
        {"rho": values, "band_over_mean": ratios},
    )


def linear_cka(left: torch.Tensor, right: torch.Tensor) -> float:
    """Linear CKA between two descriptor sets, flattened over locations."""
    a = left.reshape(len(left), -1).double()
    b = right.reshape(len(right), -1).double()
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    cross = float((a.T @ b).square().sum())
    left_norm = float((a.T @ a).square().sum())
    right_norm = float((b.T @ b).square().sum())
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return cross / (left_norm * right_norm) ** 0.5


def gate_inter_level(
    descriptor_map: dict[str, torch.Tensor], config: QualificationConfig
) -> Gate:
    """G8: levels must carry distinct information or the branch is a no-op.

    Under the original two-adjacent-level design this was the gap most likely
    to make the whole semantic branch invisible: blocks 6 and 7 of a U-ViT with
    uniform width are separated by one block and one skip fusion.
    """
    names = sorted(descriptor_map)
    pairs: dict[str, float] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            pairs[f"{left}_vs_{right}"] = linear_cka(
                descriptor_map[left], descriptor_map[right]
            )
    worst = max(pairs.values()) if pairs else 0.0
    ok = worst < config.inter_level_cka_ceiling
    return (
        "G8 inter-level non-redundancy",
        ok,
        f"max pairwise CKA {worst:.3f} (need < {config.inter_level_cka_ceiling})",
        {"cka": pairs},
    )


def gate_local_token_rank(
    descriptor_map: dict[str, torch.Tensor], config: QualificationConfig
) -> Gate:
    """G9: rank and concentration on the **local-token** descriptor.

    The global mean/std pair is 2 of 66 vectors; checking only those would miss
    outlier-channel domination in the part that carries 97% of the energy.
    """
    report: dict[str, dict] = {}
    ok = True
    for name, values in descriptor_map.items():
        local = values[:, :-2]
        flat = local.reshape(-1, local.shape[-1]).double()
        centred = flat - flat.mean(dim=0, keepdim=True)
        spectrum = torch.linalg.svdvals(centred).square()
        total = float(spectrum.sum())
        rank = 0.0 if total <= 0 else total**2 / float(spectrum.square().sum())
        pc1 = _token_pc1_share(local)
        report[name] = {"effective_rank": rank, "pc1_share": pc1}
        ok = ok and rank >= config.rank_floor and pc1 <= config.pc1_ceiling
    return (
        "G9 local-token rank and concentration",
        ok,
        ", ".join(
            f"{name}: rank {value['effective_rank']:.1f}, PC1 {value['pc1_share']:.2f}"
            for name, value in report.items()
        ),
        report,
    )


def gate_distance_spread(
    descriptor_map: dict[str, torch.Tensor], config: QualificationConfig
) -> Gate:
    """G4: a flat kernel cannot discriminate anything."""
    report = {}
    ok = True
    for name, values in descriptor_map.items():
        flat = values.reshape(len(values), -1).double()
        distances = torch.cdist(flat, flat)
        off = distances[~torch.eye(len(flat), dtype=torch.bool)]
        cv = float(off.std() / off.mean().clamp_min(1e-12))
        report[name] = cv
        ok = ok and cv >= config.distance_cv_floor
    return (
        "G4 pairwise distance spread",
        ok,
        ", ".join(f"{name}: CV {value:.3f}" for name, value in report.items()),
        report,
    )


def gate_scramble(
    trunk,
    images: torch.Tensor,
    t_f: float,
    features: FeatureConfig,
    normalization,
    config: QualificationConfig,
    *,
    seed: int = 43,
) -> Gate:
    """G2: a patch-shuffled image must read as farther than a benign variant.

    The benign variant is a horizontal flip, which preserves content exactly.
    The shuffle destroys spatial arrangement while preserving every pixel value,
    so a map sensitive only to colour statistics fails this and a map sensitive
    to structure passes.
    """
    generator = torch.Generator().manual_seed(seed)
    base = encode(trunk, images, t_f, features, normalization, generator=generator)

    flipped = torch.flip(images, dims=(-1,))
    benign = encode(
        trunk,
        flipped,
        t_f,
        features,
        normalization,
        generator=torch.Generator().manual_seed(seed),
    )

    side = images.shape[-1] // 4
    patches = images.reshape(len(images), 3, 4, side, 4, side)
    patches = patches.permute(0, 1, 2, 4, 3, 5).reshape(len(images), 3, 16, side, side)
    order = torch.randperm(16, generator=generator)
    shuffled = patches[:, :, order].reshape(len(images), 3, 4, 4, side, side)
    shuffled = shuffled.permute(0, 1, 2, 4, 3, 5).reshape(images.shape)
    scrambled = encode(
        trunk,
        shuffled,
        t_f,
        features,
        normalization,
        generator=torch.Generator().manual_seed(seed),
    )

    wins = []
    for name in base:
        benign_distance = (benign[name] - base[name]).flatten(1).norm(dim=1)
        scramble_distance = (scrambled[name] - base[name]).flatten(1).norm(dim=1)
        wins.append(float((scramble_distance > benign_distance).double().mean()))
    fraction = sum(wins) / len(wins)
    ok = fraction >= config.scramble_fraction_floor
    return (
        "G2 structure over colour statistics",
        ok,
        f"scramble farther than benign in {fraction:.1%} of pairs "
        f"(need >= {config.scramble_fraction_floor:.0%})",
        {"per_level_fraction": dict(zip(sorted(base), wins))},
    )


def run_gate(
    trunk,
    images: torch.Tensor,
    t_f: float,
    features: FeatureConfig,
    normalization,
    config: QualificationConfig,
) -> dict:
    """G7 and G8 first, then the rest.  Fail-fast is the design, not a habit."""
    config.validate()
    descriptor_map = encode(
        trunk,
        images,
        t_f,
        features,
        normalization,
        generator=torch.Generator().manual_seed(97),
    )
    descriptor_map = {name: value.detach() for name, value in descriptor_map.items()}

    profile = band_sensitivity(trunk, images, t_f, features, normalization)
    gates: list[Gate] = [
        gate_band_sensitivity(profile, config),
        gate_inter_level(descriptor_map, config),
    ]
    if all(ok for _, ok, _, _ in gates):
        gates.extend(
            [
                gate_local_token_rank(descriptor_map, config),
                gate_distance_spread(descriptor_map, config),
                gate_scramble(
                    trunk, images, t_f, features, normalization, config
                ),
            ]
        )
    failed = [name for name, ok, _, _ in gates if not ok]
    return {
        "t_f": t_f,
        "band_profile": profile,
        "gates": [
            {"name": name, "passed": ok, "detail": detail, "data": data}
            for name, ok, detail, data in gates
        ],
        "failed": failed,
        "passed": not failed,
        "fail_fast": len(gates) == 2 and bool(failed),
    }
