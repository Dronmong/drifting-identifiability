"""Target-only calibration: normalization scales, bandwidths, coefficients.

Everything here is derived from target data before any correction event, and
then frozen.  Nothing is inherited: B1's and B2's constants were calibrated
against a different architecture, a different loss scale and a different data
subset, so only the *procedure* transfers.

The metric must not move in response to model failure, which is why scales come
from targets rather than from a running batch statistic.
"""

from __future__ import annotations

import torch

from .config import FeatureConfig, FieldConfig
from .features import LevelNormalization


def _token_pc1_share(values: torch.Tensor) -> float:
    """Variance share of the leading principal component at token level.

    Transformer hidden states routinely have a few outlier channels dominating
    the L2 norm, in which case a Laplace distance silently reduces to a one- or
    two-dimensional statistic.  Checking this on the *global* descriptor alone
    would miss it, since 64 of 66 vectors are local tokens.
    """
    flat = values.reshape(-1, values.shape[-1]).double()
    centred = flat - flat.mean(dim=0, keepdim=True)
    spectrum = torch.linalg.svdvals(centred).square()
    total = float(spectrum.sum())
    return 0.0 if total <= 0 else float(spectrum[0]) / total


def calibrate_normalization(
    descriptor_map: dict[str, torch.Tensor], config: FeatureConfig
) -> dict[str, LevelNormalization]:
    """Conditional per-channel scaling, then a scalar level scale.

    Stage one fires only above the declared PC1 trigger, and its floor bounds
    amplification of low-variance noise channels at ``1/floor_fraction``.
    Stage two makes distances comparable across levels of different width.
    """
    config.validate()
    result: dict[str, LevelNormalization] = {}
    for name, values in descriptor_map.items():
        if values.ndim != 3:
            raise ValueError("descriptors must be [batch, locations, channels]")
        if len(values) < 3:
            raise ValueError("normalization needs at least three target images")
        work = values.double()
        share = _token_pc1_share(work)

        channel_scale = None
        if share > config.per_channel_pc1_trigger:
            sigma = work.reshape(-1, work.shape[-1]).std(dim=0, unbiased=False)
            floor = config.per_channel_floor_fraction * float(sigma.median())
            channel_scale = sigma.clamp_min(max(floor, 1e-12))
            work = work / channel_scale

        channels = work.shape[-1]
        # Mean pairwise distance **at matched locations**, over distinct images.
        #
        # Computed per location with cdist, not by broadcasting. The naive
        # [N, N, L, C] difference is 12.4 GB in float64 at the production shape
        # (256 images, 66 locations, 384 channels) and simply will not run; the
        # per-location form is [L, N, N], 33 MB.
        per_location = torch.cdist(work.transpose(0, 1), work.transpose(0, 1), p=2)
        count = work.shape[0]
        mask = ~torch.eye(count, dtype=torch.bool, device=work.device)
        level_scale = float(per_location[:, mask].mean()) / (channels**0.5)
        if not (level_scale > 0) or not torch.isfinite(
            torch.tensor(level_scale)
        ):
            raise ValueError(f"degenerate level scale for {name}: {level_scale}")

        result[name] = LevelNormalization(
            channel_scale=None if channel_scale is None else channel_scale.float(),
            level_scale=level_scale,
            pc1_share=share,
            per_channel_applied=channel_scale is not None,
        )
    return result


def _masked_weights(distances: torch.Tensor, tau: float) -> torch.Tensor:
    """Row softmax with the self-match removed.

    The self-match matters: an earlier calibration in this program hit its
    nominal 0.05 target through the zero-distance diagonal while the realized
    off-diagonal neighbourhood was ~0.60, so the label described a regime the
    field was never in.
    """
    count = distances.shape[-1]
    if count < 3:
        raise ValueError("ESS calibration needs at least three targets")
    eye = torch.eye(count, dtype=torch.bool, device=distances.device)
    logits = (-distances / tau).masked_fill(eye.expand_as(distances), float("-inf"))
    return torch.softmax(logits, dim=-1)


def _off_diagonal_ess(distances: torch.Tensor, tau: float) -> torch.Tensor:
    weights = _masked_weights(distances, tau)
    return weights.square().sum(dim=-1).reciprocal() / (distances.shape[-1] - 1)


def _matched_location_distances(
    values: torch.Tensor, samples: int
) -> tuple[torch.Tensor, float]:
    """Pairwise distances **within each location**, pooled across locations.

    The field only ever compares location l of one image with location l of
    another, so the distance distribution that sets the bandwidth is the
    within-location one.  Flattening ``[N, L, C]`` to ``[N*L, C]`` and taking
    the first rows -- as an earlier draft did -- calibrates on distances
    *between* locations of the same few images, which is a different
    distribution with a different scale, and it silently drops most images.
    """
    work = values.double()
    if work.ndim == 2:
        work = work.unsqueeze(1)
    if work.ndim != 3:
        raise ValueError("expected [images, locations, channels]")
    sample = work[:samples].transpose(0, 1).contiguous()
    distances = torch.cdist(sample, sample, p=2)
    positive = distances[distances > 0]
    if positive.numel() == 0:
        raise ValueError("degenerate target cloud: all distances are zero")
    return distances, float(positive.median())


def calibrate_bandwidth(
    values: torch.Tensor, target: float, config: FieldConfig
) -> dict:
    """Bisect tau to a target median off-diagonal ESS fraction.

    ESS is scale-free, so a radius means the same neighbourhood size at every
    level regardless of that level's distance scale.  A raw tau would not.
    """
    if not 0 < target < 1:
        raise ValueError("the ESS target is a fraction in (0,1)")
    distances, base = _matched_location_distances(values, config.ess_samples)

    low, high = 1e-4, 1e4
    for _ in range(config.ess_iterations):
        middle = (low * high) ** 0.5
        achieved = float(_off_diagonal_ess(distances, base * middle).median())
        if achieved < target:
            low = middle
        else:
            high = middle
    tau = base * (low * high) ** 0.5
    ess = _off_diagonal_ess(distances, tau)
    maximum = _masked_weights(distances, tau).max(dim=-1).values
    return {
        "tau": tau,
        "target_ess": target,
        "achieved_ess_median": float(ess.median()),
        "achieved_ess_p05": float(ess.quantile(0.05)),
        "maximum_weight_p95": float(maximum.quantile(0.95)),
        "distance_median": base,
        "tau_over_distance_median": tau / base,
        "images": int(min(len(values), config.ess_samples)),
        "locations": int(distances.shape[0]),
    }


def bandwidth_is_healthy(record: dict, config: FieldConfig) -> bool:
    """The tail floor scales with the requested radius.

    An absolute floor cannot serve a radius set spanning an order of magnitude:
    at a median target of 0.10 it would demand the 5th percentile reach the
    median, rejecting every local radius and silently collapsing the set back
    into three broad fields -- the exact failure the multi-radius design exists
    to avoid.
    """
    floor = config.ess_p05_fraction * record["target_ess"]
    return (
        record["achieved_ess_p05"] >= floor
        and record["maximum_weight_p95"] <= config.max_weight_p95_ceiling
    )


def calibrate_level_bandwidths(
    values: torch.Tensor, config: FieldConfig
) -> dict[float, dict]:
    """One bandwidth per declared radius, with the fallback ladder recorded.

    A genuinely local radius is the point of the multi-radius set, and it is
    also the one that can fail the health floors on a small cloud.  Stepping it
    along the declared ladder is permitted; widening it silently is not.
    """
    config.validate()
    result: dict[float, dict] = {}
    for radius in config.radii:
        candidates = (
            [r for r in config.radius_ladder if r >= radius]
            if radius == min(config.radii)
            else [radius]
        )
        chosen = None
        attempts = []
        for candidate in candidates:
            record = calibrate_bandwidth(values, candidate, config)
            record["requested_radius"] = radius
            record["used_radius"] = candidate
            attempts.append(
                {
                    "radius": candidate,
                    "healthy": bandwidth_is_healthy(record, config),
                    "ess_p05": record["achieved_ess_p05"],
                    "max_weight_p95": record["maximum_weight_p95"],
                }
            )
            if bandwidth_is_healthy(record, config):
                chosen = record
                break
        if chosen is None:
            raise ValueError(
                f"no radius on the ladder cleared the health floors for the "
                f"requested radius {radius}: {attempts}"
            )
        chosen["ladder_attempts"] = attempts
        result[radius] = chosen
    return result


def taus_from_records(records: dict[str, dict[float, dict]]) -> dict[str, dict[float, float]]:
    return {
        level: {radius: record["tau"] for radius, record in per_level.items()}
        for level, per_level in records.items()
    }


def calibrate_coefficient(
    primary_norm: float, component_norm: float, cap: float
) -> float:
    """The weight that puts a component exactly at its declared cap.

    Caps are per component and never shared: a shared total would make the arm
    carrying a new pressure the arm with weakened protection.
    """
    if primary_norm <= 0:
        raise ValueError("the primary gradient norm must be positive")
    if component_norm <= 0:
        raise ValueError("a component with zero gradient cannot be calibrated")
    return cap * primary_norm / component_norm
