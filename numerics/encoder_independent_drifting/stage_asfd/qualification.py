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
from .field import descriptor_energy, multi_radius_energy

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
    # Flatten to one plane per row, not one *sample* per row. The transform is
    # two-dimensional and per-channel, so a colour image contributes three
    # independent planes; collapsing them into a single row multiplies a
    # [N, C*size*size] tensor by a [size*size, size*size] matrix, which is a
    # shape error for C > 1 and would silently mix channels if the dimensions
    # ever happened to line up. Every existing caller passed C = 1, so this
    # only surfaced on real three-channel images.
    flat = coefficients.reshape(-1, size * size)
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
) -> dict[str, dict[str, float]]:
    """G7: feature response per Haar band, relative to a raw-pixel control.

    The upper bound rejects the hypersensitivity of the Phases 17-18 pretrained
    ResNet.  **The lower bound is the point**: the trunk allocates capacity
    under MSE, which underweights exactly the bands an MSE-trained generator is
    weakest in, and a map with zero high-frequency sensitivity passes a
    one-sided check trivially.
    """
    generator = torch.Generator().manual_seed(seed)
    size = images.shape[-1]
    result: dict[str, dict[str, float]] = {name: {} for name in config.levels}
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
        raw_delta = float(perturbation.square().mean().sqrt())
        for name in base:
            feature_delta = float((shifted[name] - base[name]).square().mean().sqrt())
            result[name][f"rho_{band}"] = feature_delta / max(raw_delta, 1e-12)
    return result


def gate_band_sensitivity(profile: dict, config: QualificationConfig) -> Gate:
    values = {
        level: {band: record[f"rho_{band}"] for band in BANDS}
        for level, record in profile.items()
    }
    ok = bool(values) and all(
        config.band_sensitivity_low <= value <= config.band_sensitivity_high
        for per_level in values.values()
        for value in per_level.values()
    )
    detail = "; ".join(
        f"{level}: " + ", ".join(f"{band}={value:.2f}" for band, value in row.items())
        for level, row in values.items()
    )
    return (
        "G7 per-band sensitivity",
        ok,
        (
            f"absolute feature/raw response {detail} (each needs "
            f"[{config.band_sensitivity_low}, {config.band_sensitivity_high}])"
        ),
        {"rho": values},
    )


def _effective_rank_and_pc1(values: torch.Tensor) -> tuple[float, float]:
    flat = values.reshape(len(values), -1).double()
    centred = flat - flat.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.svdvals(centred).square()
    total = float(spectrum.sum())
    if total <= 0:
        return 0.0, 1.0
    return total**2 / float(spectrum.square().sum()), float(spectrum[0]) / total


def gate_global_rank(
    descriptor_map: dict[str, torch.Tensor], config: QualificationConfig
) -> Gate:
    """G3: rank/concentration on the invariant global mean/std descriptor."""
    report = {}
    ok = True
    for name, values in descriptor_map.items():
        rank, pc1 = _effective_rank_and_pc1(values[:, -2:])
        report[name] = {"effective_rank": rank, "pc1_share": pc1}
        ok = ok and rank >= config.rank_floor and pc1 <= config.pc1_ceiling
    return (
        "G3 global rank and concentration",
        ok,
        ", ".join(
            f"{name}: rank {row['effective_rank']:.1f}, PC1 {row['pc1_share']:.2f}"
            for name, row in report.items()
        ),
        report,
    )


def _local_inverse_transform(
    values: torch.Tensor, flips: torch.Tensor, shifts: torch.Tensor
) -> torch.Tensor:
    """Undo known image flips/translations on the 8x8 local descriptor grid."""
    local = values[:, :-2]
    side = round(local.shape[1] ** 0.5)
    if side * side != local.shape[1]:
        raise ValueError("local descriptor is not a square grid")
    grid = local.reshape(len(local), side, side, local.shape[-1])
    restored = []
    for index, row in enumerate(grid):
        row = torch.roll(row, shifts=(0, -int(shifts[index])), dims=(0, 1))
        if bool(flips[index]):
            row = torch.flip(row, dims=(1,))
        restored.append(row)
    return torch.stack(restored).reshape_as(local)


def gate_benign_auc(
    trunk,
    images: torch.Tensor,
    t_f: float,
    features: FeatureConfig,
    normalization,
    config: QualificationConfig,
    *,
    seed: int = 39,
) -> Gate:
    """G1 on globals and registered local tokens, never unregistered tokens."""
    if len(images) < 8:
        raise ValueError("benign AUC needs at least eight images")
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype).to(
        images.device
    )
    base_noised = (1.0 - t_f) * images + t_f * noise
    base = encode(trunk, images, t_f, features, normalization, noised=base_noised)
    flips = torch.arange(len(images), device=images.device) % 2 == 0
    shifts = torch.where(flips, 1, -1)
    benign_noised = torch.where(
        flips[:, None, None, None],
        torch.flip(base_noised, dims=(-1,)),
        base_noised,
    )
    # Four pixels correspond to one cell of the 8x8 pooled descriptor grid.
    benign_noised = torch.stack(
        [
            torch.roll(row, shifts=int(shift) * 4, dims=-1)
            for row, shift in zip(benign_noised, shifts)
        ]
    )
    benign = encode(trunk, images, t_f, features, normalization, noised=benign_noised)
    permutation = torch.roll(torch.arange(len(images), device=images.device), 1)
    report = {}
    ok = True
    for name in base:
        global_benign = (
            (benign[name][:, -2:] - base[name][:, -2:]).flatten(1).norm(dim=1)
        )
        global_random = (
            (base[name][permutation, -2:] - base[name][:, -2:]).flatten(1).norm(dim=1)
        )
        local_registered = _local_inverse_transform(benign[name], flips, shifts)
        local_benign = (local_registered - base[name][:, :-2]).flatten(1).norm(dim=1)
        local_random = (
            (base[name][permutation, :-2] - base[name][:, :-2]).flatten(1).norm(dim=1)
        )
        global_auc = float((global_benign < global_random).double().mean())
        local_auc = float((local_benign < local_random).double().mean())
        report[name] = {"global_auc": global_auc, "registered_local_auc": local_auc}
        ok = ok and min(global_auc, local_auc) >= config.benign_auc_floor
    return (
        "G1 benign below random distance",
        ok,
        "; ".join(
            f"{name}: global {row['global_auc']:.2f}, local {row['registered_local_auc']:.2f}"
            for name, row in report.items()
        ),
        report,
    )


def gate_induced_gradient_distinctness(
    raw_roles: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    feature_roles: tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]
    ],
    raw_taus: dict[float, float],
    feature_taus: dict[str, dict[float, float]],
    config: QualificationConfig,
) -> Gate:
    """G6 in a common space: gradients induced on the negative images.

    Raw and feature fields themselves live in R^3072 and R^C respectively, so
    their cosine is undefined.  Their gradients with respect to the same
    generated/negative image tensor are the operational directions actually
    delivered to the online model and are therefore the valid comparison.
    """
    raw_probes, raw_positives, raw_negatives = raw_roles
    feature_probes, feature_positives, feature_negatives = feature_roles
    if not raw_negatives.requires_grad:
        raise ValueError("G6 negatives must retain their image-space graph")
    raw_value, _ = multi_radius_energy(
        raw_probes,
        raw_positives,
        raw_negatives.flatten(1),
        raw_taus,
        diagnostics=False,
    )
    feature_value, _ = descriptor_energy(
        feature_probes,
        feature_positives,
        feature_negatives,
        feature_taus,
        diagnostics=False,
    )
    raw_gradient = torch.autograd.grad(raw_value, raw_negatives, retain_graph=True)[0]
    feature_gradient = torch.autograd.grad(feature_value, raw_negatives)[0]
    left, right = raw_gradient.flatten(), feature_gradient.flatten()
    cosine = float(
        (left.double() @ right.double())
        / (left.double().norm() * right.double().norm()).clamp_min(1e-30)
    )
    ok = abs(cosine) < config.raw_field_cosine_ceiling
    return (
        "G6 raw/self induced-gradient distinctness",
        ok,
        f"absolute image-gradient cosine {abs(cosine):.3f} (need < {config.raw_field_cosine_ceiling})",
        {"cosine": cosine, "comparison_space": "negative-image input gradient"},
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


def gate_inter_level_field_cosine(
    fields: dict[str, torch.Tensor], config: QualificationConfig
) -> Gate:
    """G8's operational half: feature fields must not all point the same way."""
    names = sorted(fields)
    pairs = {}
    ok = len(names) >= 2
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left = fields[left_name].detach().reshape(-1).double()
            right = fields[right_name].detach().reshape(-1).double()
            cosine = float(
                (left @ right) / (left.norm() * right.norm()).clamp_min(1e-30)
            )
            pairs[f"{left_name}_vs_{right_name}"] = cosine
            ok = ok and abs(cosine) < config.inter_level_cosine_ceiling
    worst = max((abs(value) for value in pairs.values()), default=1.0)
    return (
        "G8 inter-level field non-redundancy",
        ok,
        f"max absolute field cosine {worst:.3f} (need < {config.inter_level_cosine_ceiling})",
        {"cosine": pairs},
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
        (
            f"scramble farther than benign in {fraction:.1%} of pairs "
            f"(need >= {config.scramble_fraction_floor:.0%})"
        ),
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
                gate_benign_auc(trunk, images, t_f, features, normalization, config),
                gate_global_rank(descriptor_map, config),
                gate_local_token_rank(descriptor_map, config),
                gate_distance_spread(descriptor_map, config),
                gate_scramble(trunk, images, t_f, features, normalization, config),
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
