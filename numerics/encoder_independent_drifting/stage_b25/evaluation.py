"""Paired B2.5 evaluation and a development-only decision summary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from ..appearance import NEIGHBOURS, _knn_radii, precision_recall
from ..config import MASTER_SEED, derive_seed
from ..fid import frechet_from_features, kid_from_features
from ..stage_b2.core import B2Config, laplace_drift_energy
from ..stage_b2.metrics import effective_rank
from .core import B25_ARMS, B25Config


def _digest(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()[:16]


@dataclass(frozen=True)
class B25EvaluationAllocation:
    """One source partition, shared by all arms and checkpoints."""

    source_id: str
    reference: np.ndarray
    controls: tuple[np.ndarray, ...]
    probe_centres: tuple[np.ndarray, ...]
    positives: tuple[np.ndarray, ...]
    floor_negatives: tuple[np.ndarray, ...]
    unused: np.ndarray

    @property
    def digests(self) -> dict[str, str]:
        arrays: dict[str, np.ndarray] = {
            "reference": self.reference,
            "unused": self.unused,
        }
        arrays |= {
            f"control_{index}": values for index, values in enumerate(self.controls)
        }
        arrays |= {
            f"audit_{index}_probe_centres": values
            for index, values in enumerate(self.probe_centres)
        }
        arrays |= {
            f"audit_{index}_positives": values
            for index, values in enumerate(self.positives)
        }
        arrays |= {
            f"audit_{index}_floor_negatives": values
            for index, values in enumerate(self.floor_negatives)
        }
        return {
            "source_id": hashlib.sha256(self.source_id.encode()).hexdigest()[:16],
            **{name: _digest(values) for name, values in arrays.items()},
        }

    def assert_disjoint(self) -> None:
        combined = np.concatenate(
            [
                self.reference,
                *self.controls,
                *self.probe_centres,
                *self.positives,
                *self.floor_negatives,
                self.unused,
            ]
        )
        if len(combined) != len(np.unique(combined)):
            raise AssertionError("B2.5 evaluation roles overlap")


def evaluation_allocation(
    pool_size: int,
    source_id: str,
    *,
    units: int,
    generated_samples: int,
    reference_samples: int,
    audit_replicates: int,
    audit_batch: int,
) -> B25EvaluationAllocation:
    if not source_id.strip():
        raise ValueError("B2.5 source ID must be stable and nonempty")
    required = (
        reference_samples
        + units * generated_samples
        + 3 * audit_replicates * audit_batch
    )
    if pool_size < required:
        raise ValueError(
            f"B2.5 source has {pool_size} rows but declared roles need {required}"
        )
    order = np.random.default_rng(
        derive_seed(MASTER_SEED + 125_000, "b25-allocation", source_id)
    ).permutation(pool_size)
    cursor = 0

    def take(count: int) -> np.ndarray:
        nonlocal cursor
        values = np.sort(order[cursor : cursor + count])
        cursor += count
        return values

    result = B25EvaluationAllocation(
        source_id=source_id,
        reference=take(reference_samples),
        controls=tuple(take(generated_samples) for _ in range(units)),
        probe_centres=tuple(take(audit_batch) for _ in range(audit_replicates)),
        positives=tuple(take(audit_batch) for _ in range(audit_replicates)),
        floor_negatives=tuple(take(audit_batch) for _ in range(audit_replicates)),
        unused=np.sort(order[cursor:]),
    )
    result.assert_disjoint()
    return result


def _squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    result = (
        (left * left).sum(axis=1)[:, None]
        + (right * right).sum(axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.maximum(result, 0.0)


def precision_recall_membership(
    generated: np.ndarray, real: np.ndarray, k: int = NEIGHBOURS
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-manifold Bernoulli indicators underlying precision and recall."""
    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    real_radii = _knn_radii(real, k)
    generated_radii = _knn_radii(generated, k)
    distances = np.sqrt(_squared_distances(generated, real))
    precision = (distances <= real_radii[None, :]).any(axis=1)
    recall = (distances.T <= generated_radii[None, :]).any(axis=1)
    return precision, recall


def density_coverage(
    generated: np.ndarray, real: np.ndarray, k: int = 5
) -> dict[str, float | int]:
    """Naeem et al. feature-space density and coverage."""
    if k < 1 or len(real) <= k:
        raise ValueError("density/coverage needs more real samples than k")
    generated = np.asarray(generated, dtype=np.float64)
    real = np.asarray(real, dtype=np.float64)
    real_radii = _knn_radii(real, k)
    distances = np.sqrt(_squared_distances(generated, real))
    density = float((distances <= real_radii[None, :]).sum() / (k * len(generated)))
    coverage = float((distances.min(axis=0) <= real_radii).mean())
    return {
        "density": density,
        "coverage": coverage,
        "k": int(k),
        "samples_generated": len(generated),
        "samples_real": len(real),
    }


def metrics_from_features(
    images: torch.Tensor,
    features: np.ndarray,
    reference_features: np.ndarray,
    *,
    density_k: int,
) -> dict:
    pr = precision_recall(features, reference_features)
    return {
        "precision": float(pr["precision"]),
        "recall": float(pr["recall"]),
        "pr_f1": float(pr["f1"]),
        "kid": kid_from_features(features, reference_features),
        "fid": frechet_from_features(features, reference_features),
        "fid_is_small_sample_indicative": True,
        "effective_rank": effective_rank(images.cpu()),
        **density_coverage(features, reference_features, density_k),
        "samples_generated": len(features),
        "samples_reference": len(reference_features),
    }


def _paired_mean_interval(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict:
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    if candidate.shape != baseline.shape or candidate.ndim != 1:
        raise ValueError("paired indicator arrays must be matching vectors")
    rng = np.random.default_rng(seed)
    difference = candidate - baseline
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        chosen = rng.integers(0, len(difference), len(difference))
        draws[index] = difference[chosen].mean()
    return {
        "difference": float(difference.mean()),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
        "scope": "conditional on the trained models and full-set kNN manifolds",
    }


def paired_values_interval(
    candidate: np.ndarray | list[float],
    baseline: np.ndarray | list[float],
    *,
    seed: int,
    replicates: int,
) -> dict:
    """Public paired mean-difference bootstrap for replicate-level values."""
    return _paired_mean_interval(
        np.asarray(candidate, dtype=float),
        np.asarray(baseline, dtype=float),
        seed=seed,
        replicates=replicates,
    )


def paired_precision_recall_intervals(
    candidate_features: np.ndarray,
    baseline_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    seed: int,
    replicates: int,
    k: int = NEIGHBOURS,
) -> dict:
    """Paired membership bootstrap without duplicate-corrupted kNN refits."""
    candidate_precision, candidate_recall = precision_recall_membership(
        candidate_features, reference_features, k
    )
    baseline_precision, baseline_recall = precision_recall_membership(
        baseline_features, reference_features, k
    )
    return {
        "precision": _paired_mean_interval(
            candidate_precision,
            baseline_precision,
            seed=seed,
            replicates=replicates,
        ),
        "recall": _paired_mean_interval(
            candidate_recall,
            baseline_recall,
            seed=seed + 1,
            replicates=replicates,
        ),
        "k": int(k),
    }


def paired_kid_subsample_interval(
    candidate_features: np.ndarray,
    baseline_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    seed: int,
    replicates: int,
    generated_fraction: float = 0.80,
) -> dict:
    """Paired without-replacement KID subsampling interval.

    This is deliberately not called a bootstrap: duplicated observations in a
    naive resample interact badly with neighborhood metrics.  Common generated
    indices and common reference subsets retain the variance reduction from the
    paired design.
    """
    candidate = np.asarray(candidate_features, dtype=np.float64)
    baseline = np.asarray(baseline_features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    if candidate.shape != baseline.shape:
        raise ValueError("paired KID candidates must have matching shapes")
    generated_count = max(2, int(len(candidate) * generated_fraction))
    reference_count = min(len(reference), max(generated_count, 512))
    dimension = candidate.shape[1]

    def kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (left @ right.T / dimension + 1.0) ** 3

    candidate_self = kernel(candidate, candidate)
    baseline_self = kernel(baseline, baseline)
    reference_self = kernel(reference, reference)
    candidate_cross = kernel(candidate, reference)
    baseline_cross = kernel(baseline, reference)

    def mmd(
        own: np.ndarray,
        cross: np.ndarray,
        generated_indices: np.ndarray,
        reference_indices: np.ndarray,
    ) -> float:
        generated_kernel = own[np.ix_(generated_indices, generated_indices)]
        reference_kernel = reference_self[np.ix_(reference_indices, reference_indices)]
        cross_kernel = cross[np.ix_(generated_indices, reference_indices)]
        m = len(generated_indices)
        n = len(reference_indices)
        return float(
            (generated_kernel.sum() - np.trace(generated_kernel)) / (m * (m - 1))
            + (reference_kernel.sum() - np.trace(reference_kernel)) / (n * (n - 1))
            - 2.0 * cross_kernel.mean()
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        generated_indices = rng.choice(
            len(candidate), size=generated_count, replace=False
        )
        reference_indices = rng.choice(
            len(reference), size=reference_count, replace=False
        )
        draws[index] = mmd(
            candidate_self,
            candidate_cross,
            generated_indices,
            reference_indices,
        ) - mmd(
            baseline_self,
            baseline_cross,
            generated_indices,
            reference_indices,
        )
    point = kid_from_features(candidate, reference) - kid_from_features(
        baseline, reference
    )
    return {
        "difference": float(point),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
        "generated_subsample": int(generated_count),
        "reference_subsample": int(reference_count),
        "method": "paired-without-replacement-subsampling",
    }


def drift_energy_audit_suite(
    generated_batches: list[torch.Tensor],
    target_pool: torch.Tensor,
    allocation: B25EvaluationAllocation,
    *,
    tau: float,
    unit: int,
    step: int,
    config: B2Config,
    device: torch.device | str,
) -> list[dict]:
    """Paired raw energy and real-real floor on one declared source."""
    if len(generated_batches) != config.audit_replicates:
        raise ValueError("B2.5 drift audit has the wrong replicate count")
    rows = []
    for replicate, generated in enumerate(generated_batches):
        if len(generated) != config.audit_batch:
            raise ValueError("B2.5 generated audit batch has the wrong size")
        centres = target_pool[torch.as_tensor(allocation.probe_centres[replicate])]
        positive = target_pool[torch.as_tensor(allocation.positives[replicate])]
        floor_negative = target_pool[
            torch.as_tensor(allocation.floor_negatives[replicate])
        ]
        noise_seed = derive_seed(
            MASTER_SEED + 125_000,
            "b25-audit-noise",
            allocation.source_id,
            unit,
            replicate,
        )
        generator = torch.Generator(device="cpu").manual_seed(noise_seed % (2**63 - 1))
        noise = torch.randn(centres.shape, generator=generator, dtype=centres.dtype)
        probes = (centres + config.probe_noise_std * noise).to(device)
        positive_device = positive.to(device)
        floor_device = floor_negative.to(device)
        with torch.no_grad():
            raw, raw_health = laplace_drift_energy(
                probes, positive_device, generated.to(device), tau
            )
            floor, floor_health = laplace_drift_energy(
                probes, positive_device, floor_device, tau
            )
        rows.append(
            {
                "replicate": replicate,
                "unit": int(unit),
                "step": int(step),
                "source_id": allocation.source_id,
                "probe_noise_seed": int(noise_seed),
                "raw_energy": float(raw),
                "real_real_floor": float(floor),
                "floor_relative_excess": float(raw - floor),
                "raw_kernel_health": raw_health,
                "floor_kernel_health": floor_health,
                "allocation_digests": {
                    "probe_centres": _digest(allocation.probe_centres[replicate]),
                    "positives": _digest(allocation.positives[replicate]),
                    "floor_negatives": _digest(allocation.floor_negatives[replicate]),
                },
            }
        )
    return rows


def summarize_drift_audits(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("cannot summarize an empty B2.5 drift audit")
    raw = np.asarray([row["raw_energy"] for row in rows], dtype=float)
    floor = np.asarray([row["real_real_floor"] for row in rows], dtype=float)
    excess = raw - floor
    return {
        "raw_energy_mean": float(raw.mean()),
        "raw_energy_sd": float(raw.std(ddof=1)) if len(raw) > 1 else 0.0,
        "real_real_floor_mean": float(floor.mean()),
        "floor_relative_excess_mean": float(excess.mean()),
        "replicates": len(rows),
    }


def factorial_interaction(rows: dict[str, float]) -> float:
    if set(rows) != set(B25_ARMS):
        raise ValueError("factorial interaction needs all four cells")
    return rows["B1B2"] - rows["B1"] - rows["B2"] + rows["B0"]


def adjudicate_development(final_rows: list[dict], config: B25Config) -> dict:
    """Exact final-checkpoint heuristic; not a confirmation or p-value."""
    config.validate()
    indexed = {(int(row["unit"]), row["arm"]): row for row in final_rows}
    expected = {(unit, arm) for unit in config.units for arm in config.arms}
    if set(indexed) != expected:
        raise ValueError("B2.5 final rows do not form the declared factorial")

    unit_rows = []
    for unit in config.units:
        cells = {arm: indexed[(unit, arm)] for arm in config.arms}
        energy = {
            arm: float(cells[arm]["drift_summary"]["raw_energy_mean"])
            for arm in config.arms
        }
        rank = {
            arm: float(cells[arm]["metrics"]["effective_rank"]) for arm in config.arms
        }
        precision = {
            arm: float(cells[arm]["metrics"]["precision"]) for arm in config.arms
        }
        recall = {arm: float(cells[arm]["metrics"]["recall"]) for arm in config.arms}
        b2_effect = max(energy["B0"] - energy["B2"], 0.0)
        combined_effect = energy["B0"] - energy["B1B2"]
        incumbent_precision = max(precision["B0"], precision["B1"])
        incumbent_recall = max(recall["B0"], recall["B1"])
        record = {
            "unit": unit,
            "drift_effect_retained": bool(
                combined_effect >= config.drift_effect_retention * b2_effect
                and combined_effect >= 0
            ),
            "rank_retention": rank["B1B2"] / max(rank["B0"], 1e-12),
            "rank_restored": bool(
                rank["B1B2"] / max(rank["B0"], 1e-12) >= config.rank_retention_floor
                and rank["B1B2"] > rank["B2"]
            ),
            "precision_retained": bool(
                precision["B1B2"]
                >= config.quality_retention_fraction * incumbent_precision
            ),
            "recall_retained": bool(
                recall["B1B2"] >= config.quality_retention_fraction * incumbent_recall
            ),
            "factorial_interactions": {
                "raw_drift_energy": factorial_interaction(energy),
                "effective_rank": factorial_interaction(rank),
                "precision": factorial_interaction(precision),
                "recall": factorial_interaction(recall),
            },
        }
        record["unit_promising"] = bool(
            record["drift_effect_retained"]
            and record["rank_restored"]
            and record["precision_retained"]
            and record["recall_retained"]
        )
        unit_rows.append(record)
    wins = sum(bool(row["unit_promising"]) for row in unit_rows)
    return {
        "development_only": True,
        "primary_checkpoint": config.final_step,
        "unit_rows": unit_rows,
        "promising_units": wins,
        "required_units": config.unit_wins_required,
        "promising": bool(wins >= config.unit_wins_required),
        "limits": [
            "Three training units support coarse consistency, not precise inference.",
            "Membership intervals condition on fitted models and estimated manifolds.",
            "A promising result requires a separately frozen confirmation.",
        ],
    }
