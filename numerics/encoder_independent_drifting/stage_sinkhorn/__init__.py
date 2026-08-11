"""Balanced two-batch Sinkhorn correction for the validated B0 bridge."""

from .core import (
    SinkhornConfig,
    SinkhornPlan,
    empirical_cross_self_energy,
    log_sinkhorn_plan,
    quadratic_cost,
    sinkhorn_drifted_target_loss,
    sinkhorn_velocity,
    target_cost_scale,
)
from .training import (
    SinkhornStreams,
    SinkhornTrainResult,
    calibrated_event_lambda,
    paired_seed_manifest,
    sinkhorn_correction_term,
    sinkhorn_streams,
    train_sinkhorn_bridge,
)

__all__ = [
    "SinkhornConfig",
    "SinkhornPlan",
    "SinkhornStreams",
    "SinkhornTrainResult",
    "calibrated_event_lambda",
    "empirical_cross_self_energy",
    "log_sinkhorn_plan",
    "paired_seed_manifest",
    "quadratic_cost",
    "sinkhorn_correction_term",
    "sinkhorn_drifted_target_loss",
    "sinkhorn_streams",
    "sinkhorn_velocity",
    "target_cost_scale",
    "train_sinkhorn_bridge",
]
