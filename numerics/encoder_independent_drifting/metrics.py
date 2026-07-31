"""Evaluation metrics and the pre-registered normalized geometry score.

Every metric is *internal*: no pretrained evaluator chooses a bandwidth, a
kernel weight, a stopping time or a checkpoint anywhere in this package
(plan section 9, P2.1).

Raw metric values are not comparable across target families, so the Phase-1
gate uses a normalized score.  Each metric is divided by its own
target-vs-target null level at the same sample size -- the value a *fresh
independent target sample* would score.  A normalized value of 1.0 therefore
means "indistinguishable from a real sample under this metric at this sample
size", and values are comparable across metrics and targets.  The composite
is the geometric mean of the normalized components, so no single loosely
scaled metric dominates.  This definition is frozen before any arm is run.
"""

from __future__ import annotations

import numpy as np
import torch

# Version 1: the composite frozen for the Phase-1 screen.  Retained so that
# `phase1_screen.json` stays reproducible; superseded for future screens.
GEOMETRY_SCORE_COMPONENTS = (
    "ed2", "sw1", "patch_ed2", "spectral_l1", "off_support",
)

# Version 2 (reform R4).  Two changes, both from measured defects:
#
#   * `spectral_l1` is DROPPED from scoring.  Its target-vs-target null is
#     ~6e-4, three orders of magnitude below every other component, so its
#     normalized ratio ran to 48-346 and dominated the geometric mean.  It is
#     still reported, just not scored.
#   * `off_support` is REPLACED by `nearest_real`, the median distance from a
#     generated sample to the nearest real one in units of the target's own
#     nearest-neighbour scale.  `off_support` saturates at 1/null once an arm
#     leaves the support and then stops discriminating; the graded statistic
#     keeps ranking arms that are all off support, which is the regime the
#     Phase-1 screen actually operated in.
GEOMETRY_SCORE_COMPONENTS_V2 = ("ed2", "sw1", "patch_ed2", "nearest_real")
REPORTED_NOT_SCORED = ("spectral_l1", "off_support")

# A null below this is treated as unusable for normalization rather than
# silently producing an enormous ratio.
NULL_FLOOR = 1e-6
MIN_TRUSTWORTHY_NULL = 1e-3


def _flat(x: torch.Tensor) -> np.ndarray:
    return x.detach().reshape(len(x), -1).to(torch.float64).numpy()


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    sq = ((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :]
          - 2.0 * a @ b.T)
    return np.sqrt(np.maximum(sq, 0.0))


def energy_distance2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Squared energy distance between two image samples."""
    fa, fb = _flat(a), _flat(b)
    return float(2 * _pairwise(fa, fb).mean()
                 - _pairwise(fa, fa).mean() - _pairwise(fb, fb).mean())


def sliced_w1(a: torch.Tensor, b: torch.Tensor, projections: int,
              rng: np.random.Generator) -> float:
    fa, fb = _flat(a), _flat(b)
    dirs = rng.normal(size=(projections, fa.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    grid = np.linspace(0.01, 0.99, 99)
    total = 0.0
    for u in dirs:
        total += np.abs(np.quantile(fa @ u, grid)
                        - np.quantile(fb @ u, grid)).mean()
    return float(total / projections)


def calibrated_support(gen: torch.Tensor, target_eval: torch.Tensor,
                       cal_a: torch.Tensor, cal_b: torch.Tensor,
                       quantile: float = 0.95) -> dict:
    """Precision/coverage under a target-only calibrated support radius.

    The radius comes from two independent TARGET pools, never from candidate
    output, so no arm can widen its own notion of "on support".
    """
    nearest = _pairwise(_flat(cal_a), _flat(cal_b)).min(axis=1)
    radius = float(np.quantile(nearest, quantile))
    distance = _pairwise(_flat(gen), _flat(target_eval))
    precision = float((distance.min(axis=1) <= radius).mean())
    coverage = float((distance.min(axis=0) <= radius).mean())
    return {"support_radius": radius, "precision": precision,
            "coverage": coverage, "off_support": 1.0 - precision}


def random_patches(images: torch.Tensor, patch: int, count: int,
                   rng: np.random.Generator) -> torch.Tensor:
    """Sample ``count`` random patches from random images."""
    n, c, h, w = images.shape
    if patch > min(h, w):
        raise ValueError("patch larger than the image")
    which = rng.integers(0, n, count)
    rows = rng.integers(0, h - patch + 1, count)
    cols = rng.integers(0, w - patch + 1, count)
    out = torch.empty(count, c, patch, patch, dtype=images.dtype)
    for index, (i, r, cc) in enumerate(zip(which, rows, cols)):
        out[index] = images[i, :, r:r + patch, cc:cc + patch]
    return out


def patch_discrepancy(a: torch.Tensor, b: torch.Tensor, patch: int,
                      count: int, rng: np.random.Generator) -> float:
    """Energy distance between the two patch distributions."""
    return energy_distance2(random_patches(a, patch, count, rng),
                            random_patches(b, patch, count, rng))


def radial_power_spectrum(images: torch.Tensor, bands: int = 6) -> np.ndarray:
    """Mean radially binned power spectrum, normalized to sum one."""
    spectrum = torch.fft.rfft2(images.to(torch.float32))
    power = (spectrum.real ** 2 + spectrum.imag ** 2).mean(dim=(0, 1))
    h, w = power.shape
    fy = torch.fft.fftfreq(images.shape[-2]).abs()[:h]
    fx = torch.fft.rfftfreq(images.shape[-1])[:w]
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    radius = radius / radius.max().clamp_min(1e-30)
    out = np.zeros(bands)
    for band in range(bands):
        low, high = band / bands, (band + 1) / bands
        selector = (radius >= low) & (
            radius <= high if band == bands - 1 else radius < high)
        out[band] = float(power[selector].sum())
    total = out.sum()
    return out / total if total > 0 else out


def spectral_error(a: torch.Tensor, b: torch.Tensor, bands: int = 6) -> float:
    """L1 distance between normalized multiscale power spectra."""
    return float(np.abs(radial_power_spectrum(a, bands)
                        - radial_power_spectrum(b, bands)).sum())


def component_occupancy(images: torch.Tensor, target) -> np.ndarray:
    labels = target.component_label(images)
    counts = np.bincount(labels, minlength=target.n_components)
    return counts / max(len(images), 1)


def occupancy_error(images: torch.Tensor, target) -> float:
    """L1 error of empirical component mass.  Oracle diagnostic only."""
    if target.prototypes is None or target.component_weights is None:
        return float("nan")
    return float(np.abs(component_occupancy(images, target)
                        - target.component_weights).sum())


def effective_dimension(samples: torch.Tensor) -> float:
    """Participation ratio of the covariance spectrum.

    Reform R14.  ``(sum s)^2 / sum s^2`` over the eigenvalues of the centred
    second-moment matrix: the number of directions the cloud actually
    occupies, insensitive to overall scale.

    This is the statistic that made the Phase-2 defect visible after four
    phases of not looking for it -- the drifting generator sat at 2.34
    against CIFAR-16's 8.32 while every other diagnostic looked healthy.  It
    costs one SVD and is reported on every run.
    """
    flat = samples.detach().reshape(len(samples), -1).to(torch.float64)
    centred = flat - flat.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.svdvals(centred) ** 2
    total = float(spectrum.sum())
    if total <= 0:
        return 0.0
    return float(total ** 2 / float((spectrum ** 2).sum()))


def nearest_real_distance(gen: torch.Tensor, target_eval: torch.Tensor,
                          target_null: torch.Tensor) -> float:
    """Median distance to the nearest real sample, in target-NN units.

    Reform R4.  1.0 means "as close to the data as a fresh real sample
    typically is".  Unlike `off_support` this keeps discriminating after an
    arm has left the support entirely, and it is the statistic that exposed
    the off-manifold drift of the kernel-gradient rule (diagnosis section 3).
    """
    reference = _flat(target_eval)
    scale = float(np.median(_pairwise(_flat(target_null), reference)
                            .min(axis=1)))
    nearest = float(np.median(_pairwise(_flat(gen), reference).min(axis=1)))
    return nearest / max(scale, 1e-12)


def raw_metrics(gen: torch.Tensor, target_eval: torch.Tensor,
                cal_a: torch.Tensor, cal_b: torch.Tensor,
                rng: np.random.Generator, target=None,
                target_null: torch.Tensor | None = None) -> dict:
    """Every metric the screen records, before normalization."""
    out = {
        "ed2": energy_distance2(gen, target_eval),
        "sw1": sliced_w1(gen, target_eval, 128, rng),
        "patch_ed2": patch_discrepancy(gen, target_eval, 6, 512, rng),
        "spectral_l1": spectral_error(gen, target_eval),
    }
    if target_null is not None:
        out["nearest_real"] = nearest_real_distance(
            gen, target_eval, target_null)
    # Reform R14: reported always, and as a ratio against the real data so a
    # collapse is legible without a second lookup.
    out["effective_dimension"] = effective_dimension(gen)
    reference_dimension = effective_dimension(target_eval)
    out["reference_effective_dimension"] = reference_dimension
    out["effective_dimension_ratio"] = (
        out["effective_dimension"] / reference_dimension
        if reference_dimension > 0 else float("nan"))
    out.update(calibrated_support(gen, target_eval, cal_a, cal_b))
    if target is not None and target.prototypes is not None:
        out["occupancy_l1"] = occupancy_error(gen, target)
    return out


def null_metrics(target_a: torch.Tensor, target_eval: torch.Tensor,
                 cal_a: torch.Tensor, cal_b: torch.Tensor,
                 rng: np.random.Generator, target=None,
                 target_null: torch.Tensor | None = None) -> dict:
    """Target-vs-target null levels: what a fresh real sample scores."""
    return raw_metrics(target_a, target_eval, cal_a, cal_b, rng, target,
                       target_null=target_null)


def _score(raw: dict, null: dict, components) -> dict:
    ratios: dict[str, float] = {}
    untrustworthy: list[str] = []
    for name in components:
        if name not in raw or name not in null:
            continue
        value, base = float(raw[name]), float(null[name])
        if not np.isfinite(value) or not np.isfinite(base):
            continue
        if base < MIN_TRUSTWORTHY_NULL:
            # A null this small makes the ratio meaningless; record the fact
            # rather than emitting a large number that looks like a result.
            untrustworthy.append(name)
            continue
        ratios[name] = max(value, 0.0) / max(base, NULL_FLOOR)
    if not ratios:
        return {"geometry_score": float("nan"), "geometry_ratios": {},
                "untrustworthy_nulls": untrustworthy}
    logs = [np.log(max(v, NULL_FLOOR)) for v in ratios.values()]
    return {"geometry_score": float(np.exp(np.mean(logs))),
            "geometry_ratios": ratios,
            "untrustworthy_nulls": untrustworthy}


def normalized_geometry_score(raw: dict, null: dict) -> dict:
    """Version-1 composite, frozen for the Phase-1 screen.

    Retained unchanged so `phase1_screen.json` stays reproducible.  New work
    should use :func:`normalized_geometry_score_v2`.
    """
    ratios: dict[str, float] = {}
    for name in GEOMETRY_SCORE_COMPONENTS:
        if name not in raw or name not in null:
            continue
        value, base = float(raw[name]), float(null[name])
        if not np.isfinite(value) or not np.isfinite(base):
            continue
        ratios[name] = max(value, 0.0) / max(base, NULL_FLOOR)
    if not ratios:
        return {"geometry_score": float("nan"), "geometry_ratios": {}}
    logs = [np.log(max(v, NULL_FLOOR)) for v in ratios.values()]
    return {"geometry_score": float(np.exp(np.mean(logs))),
            "geometry_ratios": ratios}


def normalized_geometry_score_v2(raw: dict, null: dict) -> dict:
    """Reform R4 composite: drops `spectral_l1`, grades off-manifold distance.

    Also returns the per-component verdict, so a gate can require the sign
    to hold on a majority of components individually rather than trusting a
    single aggregate (diagnosis section 6).
    """
    result = _score(raw, null, GEOMETRY_SCORE_COMPONENTS_V2)
    result["reported_not_scored"] = {
        name: float(raw[name]) for name in REPORTED_NOT_SCORED
        if name in raw and np.isfinite(float(raw[name]))
    }
    return result


def component_verdicts(candidate_ratios: dict, baseline_ratios: dict) -> dict:
    """Per-component win/loss between two arms' normalized ratios."""
    shared = sorted(set(candidate_ratios) & set(baseline_ratios))
    wins = {name: bool(candidate_ratios[name] < baseline_ratios[name])
            for name in shared}
    return {
        "per_component": wins,
        "components": len(shared),
        "components_won": int(sum(wins.values())),
        "majority": bool(sum(wins.values()) > len(shared) / 2) if shared
        else False,
    }
