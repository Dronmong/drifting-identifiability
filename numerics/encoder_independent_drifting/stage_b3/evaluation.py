"""Matched B3 metrics with honest within- and cross-architecture resampling."""

from __future__ import annotations

import numpy as np

from ..appearance import NEIGHBOURS
from ..stage_b25.evaluation import (
    B25EvaluationAllocation,
    density_coverage,
    drift_energy_audit_suite,
    evaluation_allocation,
    metrics_from_features,
    paired_kid_subsample_interval,
    paired_precision_recall_intervals,
    precision_recall_membership,
    summarize_drift_audits,
)

B3EvaluationAllocation = B25EvaluationAllocation


def _interval(draws: np.ndarray, point: float, replicates: int, method: str) -> dict:
    return {
        "difference": float(point),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
        "method": method,
    }


def cross_architecture_membership_intervals(
    candidate_features: np.ndarray,
    baseline_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    seed: int,
    replicates: int,
    k: int = NEIGHBOURS,
) -> dict:
    """No fabricated pairing of unrelated generated indices.

    Precision indicators live on independently generated samples and are
    resampled independently. Recall indicators live on the *same reference
    points*, so their difference may legitimately retain reference pairing.
    The kNN manifolds stay fixed; intervals are conditional on fitted models.
    """
    cp, cr = precision_recall_membership(candidate_features, reference_features, k)
    bp, br = precision_recall_membership(baseline_features, reference_features, k)
    rng = np.random.default_rng(seed)
    precision_draws = np.empty(replicates, dtype=float)
    recall_draws = np.empty(replicates, dtype=float)
    recall_difference = cr.astype(float) - br.astype(float)
    for index in range(replicates):
        ci = rng.integers(0, len(cp), len(cp))
        bi = rng.integers(0, len(bp), len(bp))
        ri = rng.integers(0, len(cr), len(cr))
        precision_draws[index] = cp[ci].mean() - bp[bi].mean()
        recall_draws[index] = recall_difference[ri].mean()
    return {
        "precision": _interval(
            precision_draws,
            float(cp.mean() - bp.mean()),
            replicates,
            "independent-generated-membership-bootstrap",
        ),
        "recall": _interval(
            recall_draws,
            float(cr.mean() - br.mean()),
            replicates,
            "shared-reference-membership-bootstrap",
        ),
        "k": int(k),
        "scope": "conditional on trained models and full-set kNN manifolds",
    }


def cross_architecture_kid_interval(
    candidate_features: np.ndarray,
    baseline_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    seed: int,
    replicates: int,
    generated_fraction: float = 0.80,
) -> dict:
    """Independent generated subsets and a common reference subset."""
    candidate = np.asarray(candidate_features, dtype=np.float64)
    baseline = np.asarray(baseline_features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    if candidate.ndim != 2 or baseline.ndim != 2 or reference.ndim != 2:
        raise ValueError("KID features must be matrices")
    if not (candidate.shape[1] == baseline.shape[1] == reference.shape[1]):
        raise ValueError("KID feature dimensions differ")
    candidate_count = max(2, int(len(candidate) * generated_fraction))
    baseline_count = max(2, int(len(baseline) * generated_fraction))
    reference_count = min(len(reference), max(candidate_count, baseline_count, 512))
    dimension = candidate.shape[1]

    def polynomial_gram(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return (first @ second.T / dimension + 1.0) ** 3

    candidate_self = polynomial_gram(candidate, candidate)
    baseline_self = polynomial_gram(baseline, baseline)
    reference_self = polynomial_gram(reference, reference)
    candidate_reference = polynomial_gram(candidate, reference)
    baseline_reference = polynomial_gram(baseline, reference)

    def indexed_kid(
        generated_self: np.ndarray,
        generated_reference: np.ndarray,
        generated_indices: np.ndarray,
        reference_indices: np.ndarray,
    ) -> float:
        generated_count = len(generated_indices)
        real_count = len(reference_indices)
        generated_block = generated_self[np.ix_(generated_indices, generated_indices)]
        reference_block = reference_self[np.ix_(reference_indices, reference_indices)]
        cross_block = generated_reference[np.ix_(generated_indices, reference_indices)]
        return float(
            (generated_block.sum() - np.trace(generated_block))
            / (generated_count * (generated_count - 1))
            + (reference_block.sum() - np.trace(reference_block))
            / (real_count * (real_count - 1))
            - 2.0 * cross_block.mean()
        )

    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for index in range(replicates):
        ci = rng.choice(len(candidate), candidate_count, replace=False)
        bi = rng.choice(len(baseline), baseline_count, replace=False)
        ri = rng.choice(len(reference), reference_count, replace=False)
        draws[index] = indexed_kid(
            candidate_self, candidate_reference, ci, ri
        ) - indexed_kid(baseline_self, baseline_reference, bi, ri)
    all_reference = np.arange(len(reference))
    point = indexed_kid(
        candidate_self,
        candidate_reference,
        np.arange(len(candidate)),
        all_reference,
    ) - indexed_kid(
        baseline_self,
        baseline_reference,
        np.arange(len(baseline)),
        all_reference,
    )
    return _interval(
        draws,
        point,
        replicates,
        "independent-generated/common-reference-without-replacement-subsampling",
    ) | {
        "candidate_generated_subsample": int(candidate_count),
        "baseline_generated_subsample": int(baseline_count),
        "reference_subsample": int(reference_count),
    }


def compare_within_b3(
    candidate_features: np.ndarray,
    baseline_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict:
    """The two B3 arms share evaluation latents, so sample pairing is real."""
    if candidate_features.shape != baseline_features.shape:
        raise ValueError("paired B3 arms need matching feature arrays")
    return {
        "pairing": "same one-step latent draws",
        "precision_recall": paired_precision_recall_intervals(
            candidate_features,
            baseline_features,
            reference_features,
            seed=seed,
            replicates=replicates,
        ),
        "kid": paired_kid_subsample_interval(
            candidate_features,
            baseline_features,
            reference_features,
            seed=seed + 10,
            replicates=replicates,
        ),
    }


def compare_cross_architecture(
    candidate_features: np.ndarray,
    baseline_features: np.ndarray,
    reference_features: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict:
    return {
        "pairing": "independent generated samples; shared reference",
        "precision_recall": cross_architecture_membership_intervals(
            candidate_features,
            baseline_features,
            reference_features,
            seed=seed,
            replicates=replicates,
        ),
        "kid": cross_architecture_kid_interval(
            candidate_features,
            baseline_features,
            reference_features,
            seed=seed + 10,
            replicates=replicates,
        ),
    }


__all__ = (
    "B3EvaluationAllocation",
    "compare_cross_architecture",
    "compare_within_b3",
    "cross_architecture_kid_interval",
    "cross_architecture_membership_intervals",
    "density_coverage",
    "drift_energy_audit_suite",
    "evaluation_allocation",
    "metrics_from_features",
    "summarize_drift_audits",
)
