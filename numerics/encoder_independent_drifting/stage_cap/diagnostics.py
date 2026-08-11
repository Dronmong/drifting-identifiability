"""Train-only endpoint health for CAP-EMF-1 (protocol section 6).

Nothing here touches the sealed test split, an Inception feature, or a learned
metric.  Two lessons from S3R are built in:

* **Amplitude and rank are always reported together.**  A high effective rank
  at second-moment ratio 0.006 is not diversity, and a nonzero constant image
  clears a second-moment threshold with no diversity at all — hence centered
  variance alongside the raw moment.
* **The rank rule is ``final >= 0.8 * min(best, 1)``**, never ``final / max``.
  S3R's rule called EMF's move from an over-dispersed ratio of 4.056 down
  toward the target ratio of 1.0 a "0.410 retention" failure, which is
  directionally wrong: approaching the target is not collapse.
"""

from __future__ import annotations

import math

import torch


def effective_rank(images: torch.Tensor) -> float:
    """Participation ratio of the covariance spectrum: ``(sum l)^2 / sum l^2``."""
    flat = images.reshape(len(images), -1).double()
    centered = flat - flat.mean(dim=0, keepdim=True)
    total = float(centered.square().sum())
    if total <= 0:
        return 0.0
    # If ``s`` are the singular values, the desired denominator is sum(s^4).
    # It equals ||X Xᵀ||_F² and ||Xᵀ X||_F², so no expensive full SVD is
    # needed.  Form the smaller Gram matrix to keep the 2,048-sample audit
    # practical without changing the exact participation-ratio definition.
    gram = (
        centered @ centered.T
        if len(centered) <= centered.shape[1]
        else centered.T @ centered
    )
    denominator = float(gram.square().sum())
    return 0.0 if denominator <= 0 else total**2 / denominator


def haar_transform(images: torch.Tensor) -> torch.Tensor:
    """Full orthonormal 2-D Haar transform, shape preserving.

    Orthonormal, so Euclidean distances are exactly pixel distances and total
    energy is conserved; the preflight asserts the conservation.
    """
    x = images.clone()
    h, w = x.shape[-2:]
    size = min(h, w)
    root = 1.0 / math.sqrt(2.0)
    while size >= 2 and size % 2 == 0:
        block = x[..., :size, :size]
        low = (block[..., 0::2, :] + block[..., 1::2, :]) * root
        high = (block[..., 0::2, :] - block[..., 1::2, :]) * root
        block = torch.cat([low, high], dim=-2)
        low = (block[..., :, 0::2] + block[..., :, 1::2]) * root
        high = (block[..., :, 0::2] - block[..., :, 1::2]) * root
        block = torch.cat([low, high], dim=-1)
        x = x.clone()
        x[..., :size, :size] = block
        size //= 2
    return x


def haar_band_variances(images: torch.Tensor) -> dict[str, float]:
    """Per-band variance of one level of the Haar decomposition."""
    coefficients = haar_transform(images.double())
    half = images.shape[-1] // 2
    bands = {
        "LL": coefficients[..., :half, :half],
        "LH": coefficients[..., :half, half:],
        "HL": coefficients[..., half:, :half],
        "HH": coefficients[..., half:, half:],
    }
    return {name: float(value.var(unbiased=False)) for name, value in bands.items()}


def patch_phase_report(images: torch.Tensor) -> dict[str, object]:
    """Period-two phase imbalance and Nyquist checkerboard projection.

    Patch-2 synthesis can hide a phase-locked checkerboard even when aggregate
    Haar energy looks plausible.  This diagnostic is encoder-free and is
    therefore safe to record at every train-only health checkpoint.
    """
    if images.ndim != 4 or images.shape[-1] % 2 or images.shape[-2] % 2:
        raise ValueError("patch-phase audit needs even-sized image batches")
    values = images.detach().double()
    phases = torch.stack(
        [
            values[..., row::2, column::2].mean(dim=(-2, -1))
            for row in range(2)
            for column in range(2)
        ],
        dim=-1,
    )
    phase_centered = phases - phases.mean(dim=-1, keepdim=True)
    centered = values - values.mean(dim=(-2, -1), keepdim=True)
    denominator = centered.square().mean().clamp_min(1e-30)
    yy = torch.arange(values.shape[-2], dtype=torch.float64, device=values.device)
    xx = torch.arange(values.shape[-1], dtype=torch.float64, device=values.device)
    checker = (1 - 2 * ((yy[:, None] + xx[None, :]) % 2)).view(1, 1, *values.shape[-2:])
    checker_projection = (values * checker).mean(dim=(-2, -1))
    return {
        "phase_means": phases.mean(dim=(0, 1)).tolist(),
        "phase_imbalance_ratio": float(phase_centered.square().mean() / denominator),
        "checkerboard_projection_ratio": float(
            checker_projection.square().mean() / denominator
        ),
        "raw_rms": float(values.square().mean().sqrt()),
    }


def endpoint_reference(target: torch.Tensor) -> dict[str, object]:
    """Cache target-only quantities reused across component health reports."""
    if target.ndim != 4:
        raise ValueError("endpoint reference needs an image batch")
    target = target.double()
    result: dict[str, object] = {
        "moment": float(target.square().mean()),
        "variance": float(target.var(unbiased=False)),
        "rank": effective_rank(target),
        "bands": haar_band_variances(target),
    }
    if min(float(result[name]) for name in ("moment", "variance", "rank")) <= 0:
        raise ValueError("degenerate target cloud")
    return result


def endpoint_health(
    generated: torch.Tensor,
    target: torch.Tensor,
    reference: dict[str, object] | None = None,
) -> dict[str, float]:
    """Amplitude, dispersion, rank and multiscale energy against a fixed target."""
    if generated.ndim != 4 or target.ndim != 4:
        raise ValueError("endpoint health needs image batches")
    gen = generated.double()
    cached = reference or endpoint_reference(target)
    target_moment = float(cached["moment"])
    target_variance = float(cached["variance"])
    target_rank = float(cached["rank"])

    generated_bands = haar_band_variances(gen)
    target_bands = cached["bands"]
    phase = patch_phase_report(gen)

    result = {
        "second_moment": float(gen.square().mean()),
        "second_moment_ratio": float(gen.square().mean()) / target_moment,
        "centered_variance": float(gen.var(unbiased=False)),
        "centered_variance_ratio": float(gen.var(unbiased=False)) / target_variance,
        "effective_rank": effective_rank(gen),
        "target_effective_rank": target_rank,
        "samples": len(gen),
        "raw_saturation_fraction": float((gen.abs() > 1.0).double().mean()),
        "raw_absolute_max": float(gen.abs().max()),
        "phase_imbalance_ratio": float(phase["phase_imbalance_ratio"]),
        "checkerboard_projection_ratio": float(phase["checkerboard_projection_ratio"]),
    }
    result["effective_rank_ratio"] = result["effective_rank"] / target_rank
    for name, value in generated_bands.items():
        reference = target_bands[name]
        result[f"haar_{name}_variance"] = value
        result[f"haar_{name}_ratio"] = value / reference if reference > 0 else 0.0
    return result


def component_health(
    components: dict[str, torch.Tensor],
    target: torch.Tensor,
    reference: dict[str, object] | None = None,
) -> dict[str, dict[str, float] | float]:
    """Health of the base head, residual, and final output.

    The residual is compared with a zero target only through its own amplitude
    and Haar bands; the base and final outputs use the ordinary target-relative
    endpoint report.  Cross terms make fragile cancellation explicit.
    """
    required = {"base", "refiner_residual", "final"}
    if set(components) != required:
        raise ValueError(f"components must be exactly {sorted(required)}")
    base = components["base"].double()
    residual = components["refiner_residual"].double()
    final = components["final"].double()
    cached = reference or endpoint_reference(target)
    base_flat = base.flatten(1)
    residual_flat = residual.flatten(1)
    denominator = base_flat.norm(dim=1) * residual_flat.norm(dim=1)
    cosine = torch.where(
        denominator > 0,
        (base_flat * residual_flat).sum(dim=1) / denominator,
        torch.zeros_like(denominator),
    )
    residual_bands = haar_band_variances(residual)
    residual_phase = patch_phase_report(residual)
    return {
        "base": endpoint_health(base, target, cached),
        "final": endpoint_health(final, target, cached),
        "refiner_residual": {
            "rms": float(residual.square().mean().sqrt()),
            "rms_over_base": float(residual.square().mean().sqrt())
            / max(float(base.square().mean().sqrt()), 1e-30),
            "cosine_with_base": float(cosine.mean()),
            "phase_imbalance_ratio": float(residual_phase["phase_imbalance_ratio"]),
            "checkerboard_projection_ratio": float(
                residual_phase["checkerboard_projection_ratio"]
            ),
            **{
                f"haar_{name}_variance": value for name, value in residual_bands.items()
            },
        },
    }


def rank_noncollapse(final_ratio: float, best_ratio: float, floor: float) -> bool:
    """``final >= floor * min(best, 1)``.

    Capping ``best`` at one is the whole correction: a run that starts
    over-dispersed and converges toward the target ratio of 1.0 is improving,
    not collapsing, and must not be failed for it.
    """
    return final_ratio >= floor * min(best_ratio, 1.0)


def capability_gate(
    final: dict[str, float],
    best_rank_ratio: float,
    clip_fraction: float | None,
    nonfinite_updates: int,
    inference_forwards: int,
    gate,
) -> dict:
    """Protocol section 7.1.  Train-only; no test image is involved."""
    gate.validate()
    checks: dict[str, bool | None] = {
        "H1_second_moment": final["second_moment_ratio"] >= gate.second_moment_ratio,
        "H1b_second_moment_upper": (
            final["second_moment_ratio"] <= gate.maximum_second_moment_ratio
        ),
        "H2_centered_variance": (
            final["centered_variance_ratio"] >= gate.centered_variance_ratio
        ),
        "H2b_centered_variance_upper": (
            final["centered_variance_ratio"] <= gate.maximum_centered_variance_ratio
        ),
        "H3_rank_floor": (
            final["effective_rank_ratio"] >= gate.minimum_effective_rank_ratio
        ),
        "H3b_rank_noncollapse": rank_noncollapse(
            final["effective_rank_ratio"], best_rank_ratio, gate.rank_retention
        ),
        "H3c_rank_upper": (
            final["effective_rank_ratio"] <= gate.maximum_effective_rank_ratio
        ),
        "H4a_haar_LL_lower": final["haar_LL_ratio"] >= gate.minimum_haar_LL_ratio,
        "H4b_haar_LL_upper": final["haar_LL_ratio"] <= gate.maximum_haar_LL_ratio,
        "H4c_haar_LH_lower": final["haar_LH_ratio"] >= gate.minimum_haar_LH_ratio,
        "H4d_haar_LH_upper": final["haar_LH_ratio"] <= gate.maximum_haar_LH_ratio,
        "H4e_haar_HL_lower": final["haar_HL_ratio"] >= gate.minimum_haar_HL_ratio,
        "H4f_haar_HL_upper": final["haar_HL_ratio"] <= gate.maximum_haar_HL_ratio,
        "H4g_haar_HH_lower": final["haar_HH_ratio"] >= gate.minimum_haar_HH_ratio,
        "H4h_haar_HH_upper": final["haar_HH_ratio"] <= gate.maximum_haar_HH_ratio,
        "H5c_saturation": final.get("raw_saturation_fraction", 0.0)
        <= gate.maximum_saturation_fraction,
        "H6_finite": nonfinite_updates == 0,
        "H7_clip_fraction": (
            None
            if clip_fraction is None
            else clip_fraction < gate.maximum_clip_fraction
        ),
        "H8_one_call": inference_forwards == 1,
    }
    failed = sorted(name for name, ok in checks.items() if ok is False)
    unknown = sorted(name for name, ok in checks.items() if ok is None)
    only_h4 = failed == ["H4g_haar_HH_lower"]
    if unknown:
        verdict = "INCOMPLETE"
    elif not failed:
        verdict = "PASS"
    elif only_h4:
        verdict = "PASS_DETAIL_POOR"
    else:
        verdict = "FAIL"
    return {
        "checks": checks,
        "failed": failed,
        "unknown": unknown,
        "verdict": verdict,
        "thresholds": {
            "second_moment_ratio": gate.second_moment_ratio,
            "centered_variance_ratio": gate.centered_variance_ratio,
            "minimum_effective_rank_ratio": gate.minimum_effective_rank_ratio,
            "rank_retention": gate.rank_retention,
            "minimum_haar_LL_ratio": gate.minimum_haar_LL_ratio,
            "minimum_haar_LH_ratio": gate.minimum_haar_LH_ratio,
            "minimum_haar_HL_ratio": gate.minimum_haar_HL_ratio,
            "minimum_haar_HH_ratio": gate.minimum_haar_HH_ratio,
            "maximum_clip_fraction": gate.maximum_clip_fraction,
            "clip_window_updates": gate.clip_window_updates,
            "maximum_second_moment_ratio": gate.maximum_second_moment_ratio,
            "maximum_centered_variance_ratio": gate.maximum_centered_variance_ratio,
            "maximum_effective_rank_ratio": gate.maximum_effective_rank_ratio,
            "maximum_haar_LL_ratio": gate.maximum_haar_LL_ratio,
            "maximum_haar_LH_ratio": gate.maximum_haar_LH_ratio,
            "maximum_haar_HL_ratio": gate.maximum_haar_HL_ratio,
            "maximum_haar_HH_ratio": gate.maximum_haar_HH_ratio,
            "maximum_saturation_fraction": gate.maximum_saturation_fraction,
        },
        "asfd_note": (
            "H4 is the leading indicator for the ASFD feature gate G7, which "
            "requires per-band sensitivity in every band. A trunk trained to a "
            "model that cannot render diagonal detail is unlikely to encode it."
        ),
    }
