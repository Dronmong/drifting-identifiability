"""The normalized Laplace mean-shift field, in whatever geometry it is given.

**One implementation serves both branches.**  The raw-pixel field and the
self-feature field are the same object computed over different tensors, which is
exactly the claim the mechanism rests on: nothing changes except where the
similarity judgement is made.

    V(z) = sum_a w_a(z) y+_a  -  sum_b w_b(z) y-_b ,   w ∝ exp(-||z - y|| / tau)

Those softmax weights *are* the method's entire notion of similarity.  With raw
pixels that is Euclidean distance in R^3072, dominated by low-frequency layout
and mean colour.  With features it is Euclidean distance in the frozen trunk's
hidden state.

Locations vectorize as a leading batch dimension, so a level's 66 descriptor
positions are evaluated in one batched ``cdist`` rather than a Python loop.
"""

from __future__ import annotations

import math

import torch


def _validate(probes: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor):
    if probes.ndim != positives.ndim or probes.ndim != negatives.ndim:
        raise ValueError("field roles must share a rank")
    if probes.ndim < 2:
        raise ValueError("field roles need at least [count, dimension]")
    if (
        probes.shape[:-2] != positives.shape[:-2]
        or probes.shape[:-2] != negatives.shape[:-2]
    ):
        raise ValueError("field roles must share their leading location dimensions")
    if (
        probes.shape[-1] != positives.shape[-1]
        or probes.shape[-1] != negatives.shape[-1]
    ):
        raise ValueError("field roles must share a feature dimension")
    if min(probes.shape[-2], positives.shape[-2], negatives.shape[-2]) < 1:
        raise ValueError("field roles must be nonempty")


def laplace_weights(
    left: torch.Tensor, right: torch.Tensor, tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable exact-Laplace row weights and their Euclidean distances.

    ``softmax(-distance/tau)`` analytically equals the normalized Laplace
    weights while subtracting the row maximum internally, so a remote but finite
    row cannot become the all-zero row that broke the raw Algorithm-2 objective.
    """
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Laplace bandwidth must be positive and finite")
    distances = torch.cdist(left, right, p=2)
    return torch.softmax(-distances / tau, dim=-1), distances


def weight_health(weights: torch.Tensor, distances: torch.Tensor) -> dict:
    """Both sides are reported.  B2 logged this and gated only the target side."""
    with torch.no_grad():
        count = weights.shape[-1]
        ess = weights.square().sum(dim=-1).reciprocal() / count
        flat = ess.reshape(-1)
        maximum = weights.max(dim=-1).values.reshape(-1)
        return {
            "ess_fraction_median": float(flat.median()),
            "ess_fraction_p05": float(flat.quantile(0.05)),
            "ess_fraction_minimum": float(flat.min()),
            "maximum_weight_median": float(maximum.median()),
            "maximum_weight_p95": float(maximum.quantile(0.95)),
            "distance_median": float(distances.median()),
            "row_sum_error_maximum": float((weights.sum(dim=-1) - 1.0).abs().max()),
        }


def laplace_field(
    probes: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    tau: float,
    *,
    diagnostics: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Differentiable sample-split normalized Laplace mean-shift difference.

    Shapes are ``[..., count, dimension]``; leading dimensions are locations.
    Positives and probes are expected detached by the caller; **negatives must
    not be**, or the correction cannot reach the generator.
    """
    _validate(probes, positives, negatives)
    weights_positive, distances_positive = laplace_weights(probes, positives, tau)
    weights_negative, distances_negative = laplace_weights(probes, negatives, tau)
    field = weights_positive @ positives - weights_negative @ negatives
    if not bool(torch.isfinite(field).all()):
        raise FloatingPointError("non-finite normalized Laplace field")
    stats: dict = {}
    if diagnostics:
        stats = {
            "kernel": "exact_laplace",
            "normalization": "row_normalized_mean_shift_difference",
            "tau": float(tau),
            "positive": weight_health(weights_positive, distances_positive),
            "negative": weight_health(weights_negative, distances_negative),
            "field_rms": float(field.detach().square().mean().sqrt()),
        }
    return field, stats


def field_energy(
    probes: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    tau: float,
    *,
    diagnostics: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Mean squared field norm at one bandwidth, normalized by dimension.

    **Never square the mean field.**  The zero set of a sum of squares is the
    intersection of the zero sets; the zero set of a squared sum is not.
    """
    field, stats = laplace_field(
        probes, positives, negatives, tau, diagnostics=diagnostics
    )
    dimension = field.shape[-1]
    value = field.square().sum(dim=-1).mean() / dimension
    if not bool(torch.isfinite(value)):
        raise FloatingPointError("non-finite field energy")
    if diagnostics:
        stats["energy"] = float(value.detach())
    return value, stats


def multi_radius_energy(
    probes: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    taus: dict[float, float],
    *,
    diagnostics: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Average of separately squared per-radius energies.

    Averaging the *energies* rather than squaring an averaged field is the whole
    point: adding three fields and squaring the sum would let two wrong fields
    cancel.  A regression test builds exactly that counterexample.
    """
    if not taus:
        raise ValueError("at least one radius is required")
    total = None
    stats: dict = {}
    for radius, tau in sorted(taus.items()):
        value, health = field_energy(
            probes, positives, negatives, tau, diagnostics=diagnostics
        )
        total = value if total is None else total + value
        if diagnostics:
            stats[f"radius_{radius:g}"] = health
    energy = total / len(taus)
    if diagnostics:
        stats["energy"] = float(energy.detach())
        stats["radii"] = sorted(taus)
    return energy, stats


def descriptor_energy(
    probes: dict[str, torch.Tensor],
    positives: dict[str, torch.Tensor],
    negatives: dict[str, torch.Tensor],
    taus: dict[str, dict[float, float]],
    *,
    diagnostics: bool = True,
) -> tuple[torch.Tensor, dict]:
    """The self-feature energy: separate squares over levels, radii, locations.

    Each argument maps a level name to ``[locations, count, channels]``.  Levels
    are averaged after squaring, exactly as radii are, and locations are inside
    the squared norm's mean -- so cancellation is blocked on every index.
    """
    if not probes:
        raise ValueError("at least one feature level is required")
    if set(probes) != set(positives) or set(probes) != set(negatives):
        raise ValueError("feature levels differ between roles")
    if set(probes) != set(taus):
        raise ValueError("a bandwidth set is required for every level")
    total = None
    stats: dict = {}
    for level in sorted(probes):
        value, health = multi_radius_energy(
            probes[level],
            positives[level],
            negatives[level],
            taus[level],
            diagnostics=diagnostics,
        )
        total = value if total is None else total + value
        if diagnostics:
            stats[level] = health
    energy = total / len(probes)
    if diagnostics:
        stats["energy"] = float(energy.detach())
        stats["levels"] = sorted(probes)
        stats["location_split"] = _location_split(probes, positives, negatives, taus)
    return energy, stats


def _location_split(
    probes: dict[str, torch.Tensor],
    positives: dict[str, torch.Tensor],
    negatives: dict[str, torch.Tensor],
    taus: dict[str, dict[float, float]],
) -> dict:
    """Energy carried by the 2 global vectors versus the 64 local ones.

    97% of the descriptor is position-locked -- location l of a generated image
    is only ever compared with location l of a target image -- so under the
    recorded horizontal flip a flipped image reads as far at every local
    location.  A large share of the energy may be measuring pose rather than
    semantics, and this split is the only way to see it.
    """
    with torch.no_grad():
        local_total, global_total = 0.0, 0.0
        for level in sorted(probes):
            tau = next(iter(sorted(taus[level].values())))
            field, _ = laplace_field(
                probes[level],
                positives[level],
                negatives[level],
                tau,
                diagnostics=False,
            )
            per_location = field.square().sum(dim=-1).mean(dim=-1)
            local_total += float(per_location[:-2].sum())
            global_total += float(per_location[-2:].sum())
        total = local_total + global_total
        if total <= 0:
            return {"local_share": float("nan"), "global_share": float("nan")}
        return {
            "local_share": local_total / total,
            "global_share": global_total / total,
            "local_vectors": int(next(iter(probes.values())).shape[0]) - 2,
        }
