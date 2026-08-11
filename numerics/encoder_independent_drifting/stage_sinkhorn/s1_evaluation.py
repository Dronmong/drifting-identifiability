"""Paired held-out identity-Sinkhorn field audit for Stage S1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..config import MASTER_SEED, derive_seed
from ..f3b import F3BModelConfig, TimeConditionedUNet, sample_model
from ..stage_b2.evaluation import B2AuditAllocation
from .core import SinkhornConfig, sinkhorn_velocity

S1_SEED_OFFSET = 131_000


def s1_seed(unit: int, role: str, replicate: int | None = None) -> int:
    labels: tuple[object, ...] = ("sinkhorn-s1", unit, role)
    if replicate is not None:
        labels += (replicate,)
    return derive_seed(MASTER_SEED + S1_SEED_OFFSET, *labels)


def field_energy(
    primary: torch.Tensor,
    real_support: torch.Tensor,
    self_support: torch.Tensor,
    cost_scale: float,
    config: SinkhornConfig,
) -> tuple[float, dict]:
    velocity, health = sinkhorn_velocity(
        primary,
        real_support,
        self_support,
        cost_scale,
        config,
    )
    values = velocity.flatten(1).square().sum(dim=1)
    return float(values.mean()), health


@dataclass(frozen=True)
class S1AuditConfig:
    batch: int = 128
    replicates: int = 4
    nfe: int = 32
    paired_wins_required: int = 3
    reduction_fraction: float = 0.25
    recall_noninferiority_margin: float = 0.025
    effective_rank_retention: float = 0.80

    def validate(self) -> None:
        if self.batch < 2 or self.replicates < 2 or self.nfe <= 0:
            raise ValueError("invalid S1 audit budget")
        if not 1 <= self.paired_wins_required <= self.replicates:
            raise ValueError("invalid S1 paired-win requirement")
        if not 0 < self.reduction_fraction < 1:
            raise ValueError("invalid S1 reduction fraction")
        if not 0 < self.recall_noninferiority_margin < 1:
            raise ValueError("invalid S1 recall margin")
        if not 0 < self.effective_rank_retention <= 1:
            raise ValueError("invalid S1 rank retention")


def audit_model(
    model: TimeConditionedUNet,
    model_config: F3BModelConfig,
    fresh_pool: torch.Tensor,
    allocation: B2AuditAllocation,
    unit: int,
    device: torch.device | str,
    cost_scale: float,
    sinkhorn_config: SinkhornConfig,
    audit_config: S1AuditConfig,
) -> list[dict]:
    """Audit one model with priors shared across the paired B0/candidate call."""
    audit_config.validate()
    if audit_config.replicates > len(allocation.positives):
        raise ValueError("S1 requests more audit roles than B2 allocated")
    device = torch.device(device)
    rows: list[dict] = []
    for replicate in range(audit_config.replicates):
        primary_seed = s1_seed(unit, "audit-primary", replicate)
        self_seed = s1_seed(unit, "audit-self", replicate)
        primary = sample_model(
            model,
            audit_config.batch,
            model_config,
            audit_config.nfe,
            primary_seed,
            device,
        ).to(device)
        self_support = sample_model(
            model,
            audit_config.batch,
            model_config,
            audit_config.nfe,
            self_seed,
            device,
        ).to(device)
        real_support = fresh_pool[torch.as_tensor(allocation.positives[replicate])].to(
            device
        )
        floor_primary = fresh_pool[
            torch.as_tensor(allocation.probe_centres[replicate])
        ].to(device)
        floor_self = fresh_pool[
            torch.as_tensor(allocation.floor_negatives[replicate])
        ].to(device)
        if not (
            len(primary)
            == len(self_support)
            == len(real_support)
            == len(floor_primary)
            == len(floor_self)
            == audit_config.batch
        ):
            raise ValueError("S1 audit allocation and generated batches differ")
        energy, health = field_energy(
            primary, real_support, self_support, cost_scale, sinkhorn_config
        )
        floor, floor_health = field_energy(
            floor_primary, real_support, floor_self, cost_scale, sinkhorn_config
        )
        rows.append(
            {
                "replicate": replicate,
                "primary_seed": primary_seed,
                "self_seed": self_seed,
                "energy": energy,
                "real_real_floor": floor,
                "excess_over_real": energy - floor,
                "health": health,
                "floor_health": floor_health,
            }
        )
        del primary, self_support, real_support, floor_primary, floor_self
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def summarize_audit(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty S1 audit")
    return {
        "median_energy": float(np.median([row["energy"] for row in rows])),
        "median_real_real_floor": float(
            np.median([row["real_real_floor"] for row in rows])
        ),
        "median_excess_over_real": float(
            np.median([row["excess_over_real"] for row in rows])
        ),
    }


def compare_audits(
    candidate: list[dict], baseline: list[dict], config: S1AuditConfig
) -> dict:
    config.validate()
    if len(candidate) != config.replicates or len(baseline) != config.replicates:
        raise ValueError("S1 paired audits have the wrong replicate count")
    for left, right in zip(candidate, baseline, strict=True):
        if (
            left["replicate"] != right["replicate"]
            or left["primary_seed"] != right["primary_seed"]
            or left["self_seed"] != right["self_seed"]
        ):
            raise ValueError("S1 audit roles are not paired")
    candidate_summary = summarize_audit(candidate)
    baseline_summary = summarize_audit(baseline)
    paired_wins = sum(
        float(left["energy"]) < float(right["energy"])
        for left, right in zip(candidate, baseline, strict=True)
    )
    threshold = (1.0 - config.reduction_fraction) * baseline_summary["median_energy"]
    reduction_passes = candidate_summary["median_energy"] <= threshold
    wins_pass = paired_wins >= config.paired_wins_required
    return {
        "candidate": candidate_summary,
        "baseline": baseline_summary,
        "paired_wins": paired_wins,
        "paired_wins_required": config.paired_wins_required,
        "median_energy_threshold": threshold,
        "median_reduction_passes": bool(reduction_passes),
        "paired_wins_passes": bool(wins_pass),
        "passes": bool(reduction_passes and wins_pass),
    }


def all_plans_converged(rows: list[dict]) -> bool:
    return all(
        record[health_key][role]["converged"]
        and not record[health_key][role]["iteration_cap_hit"]
        for record in rows
        for health_key in ("health", "floor_health")
        for role in ("cross", "self")
    )
