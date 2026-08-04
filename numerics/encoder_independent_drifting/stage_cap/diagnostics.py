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
    values = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True)).square()
    total = float(values.sum())
    return 0.0 if total <= 0 else total**2 / float(values.square().sum())


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


def endpoint_health(
    generated: torch.Tensor, target: torch.Tensor
) -> dict[str, float]:
    """Amplitude, dispersion, rank and multiscale energy against a fixed target."""
    if generated.ndim != 4 or target.ndim != 4:
        raise ValueError("endpoint health needs image batches")
    gen = generated.double()
    tgt = target.double()

    target_moment = float(tgt.square().mean())
    target_variance = float(tgt.var(unbiased=False))
    target_rank = effective_rank(tgt)
    if min(target_moment, target_variance, target_rank) <= 0:
        raise ValueError("degenerate target cloud")

    generated_bands = haar_band_variances(gen)
    target_bands = haar_band_variances(tgt)

    result = {
        "second_moment": float(gen.square().mean()),
        "second_moment_ratio": float(gen.square().mean()) / target_moment,
        "centered_variance": float(gen.var(unbiased=False)),
        "centered_variance_ratio": float(gen.var(unbiased=False)) / target_variance,
        "effective_rank": effective_rank(gen),
        "target_effective_rank": target_rank,
        "samples": int(len(gen)),
    }
    result["effective_rank_ratio"] = result["effective_rank"] / target_rank
    for name, value in generated_bands.items():
        reference = target_bands[name]
        result[f"haar_{name}_variance"] = value
        result[f"haar_{name}_ratio"] = value / reference if reference > 0 else 0.0
    return result


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
    clip_fraction: float,
    nonfinite_updates: int,
    inference_forwards: int,
    gate,
) -> dict:
    """Protocol section 7.1.  Train-only; no test image is involved."""
    gate.validate()
    checks = {
        "H1_second_moment": final["second_moment_ratio"] >= gate.second_moment_ratio,
        "H2_centered_variance": (
            final["centered_variance_ratio"] >= gate.centered_variance_ratio
        ),
        "H3_rank_noncollapse": rank_noncollapse(
            final["effective_rank_ratio"], best_rank_ratio, gate.rank_retention
        ),
        "H4_haar_hh": final["haar_HH_ratio"] >= gate.haar_hh_ratio,
        "H5_haar_detail": min(final["haar_LH_ratio"], final["haar_HL_ratio"])
        >= gate.haar_detail_ratio,
        "H6_finite": nonfinite_updates == 0,
        "H7_clip_fraction": clip_fraction < gate.maximum_clip_fraction,
        "H8_one_call": inference_forwards == 1,
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    only_h4 = failed == ["H4_haar_hh"]
    if not failed:
        verdict = "PASS"
    elif only_h4:
        verdict = "PASS_DETAIL_POOR"
    else:
        verdict = "FAIL"
    return {
        "checks": checks,
        "failed": failed,
        "verdict": verdict,
        "thresholds": {
            "second_moment_ratio": gate.second_moment_ratio,
            "centered_variance_ratio": gate.centered_variance_ratio,
            "rank_retention": gate.rank_retention,
            "haar_hh_ratio": gate.haar_hh_ratio,
            "haar_detail_ratio": gate.haar_detail_ratio,
            "maximum_clip_fraction": gate.maximum_clip_fraction,
        },
        "asfd_note": (
            "H4 is the leading indicator for the ASFD feature gate G7, which "
            "requires per-band sensitivity in every band. A trunk trained to a "
            "model that cannot render diagonal detail is unlikely to encode it."
        ),
    }
