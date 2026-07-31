"""Stage-F3B evaluation allocation, metrics, and calibrated veto helpers.

Training never imports this module.  Inception is therefore report-only and
cannot become the authority defining the learned velocity or its loss.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from . import cifar
from .appearance import precision_recall, spectrum_slope
from .config import MASTER_SEED, derive_seed
from .f1 import _self_distances, duplicate_rate, effective_rank, nn_diversity
from .f1_k200 import nearest_reference_distances
from .fid import frechet_from_features, inception_features, kid_from_features


@dataclass(frozen=True)
class F3BEvaluationAllocation:
    development_reference: np.ndarray
    confirmation_reference: np.ndarray
    development_controls: tuple[np.ndarray, ...]
    confirmation_controls: tuple[np.ndarray, ...]
    unused: np.ndarray

    @property
    def digests(self) -> dict[str, str]:
        arrays = {
            "development_reference": self.development_reference,
            "confirmation_reference": self.confirmation_reference,
            "unused": self.unused,
        }
        arrays |= {
            f"development_control_{index}": values
            for index, values in enumerate(self.development_controls)
        }
        arrays |= {
            f"confirmation_control_{index}": values
            for index, values in enumerate(self.confirmation_controls)
        }
        return {
            name: hashlib.sha256(
                np.asarray(values, dtype=np.int64).tobytes()
            ).hexdigest()[:16]
            for name, values in arrays.items()
        }

    def assert_disjoint(self) -> None:
        arrays = [
            self.development_reference,
            self.confirmation_reference,
            *self.development_controls,
            *self.confirmation_controls,
            self.unused,
        ]
        joined = np.concatenate(arrays)
        if len(joined) != len(np.unique(joined)):
            raise AssertionError("F3B evaluation roles are not disjoint")


def evaluation_allocation(
    pool_size: int = 10_000,
    reference_samples: int = 2_048,
    control_samples: int = 512,
    control_groups: int = 3,
) -> F3BEvaluationAllocation:
    """One stage-specific permutation, partitioned before any F3B outcome."""
    required = 2 * reference_samples + 2 * control_groups * control_samples
    if pool_size < required:
        raise ValueError(f"evaluation pool needs {required} entries, got {pool_size}")
    order = np.random.default_rng(
        derive_seed(MASTER_SEED + 73_000, "f3b-evaluation-allocation")
    ).permutation(pool_size)
    cursor = 0

    def take(count: int) -> np.ndarray:
        nonlocal cursor
        result = np.sort(order[cursor : cursor + count])
        cursor += count
        return result

    allocation = F3BEvaluationAllocation(
        development_reference=take(reference_samples),
        confirmation_reference=take(reference_samples),
        development_controls=tuple(
            take(control_samples) for _ in range(control_groups)
        ),
        confirmation_controls=tuple(
            take(control_samples) for _ in range(control_groups)
        ),
        unused=np.sort(order[cursor:]),
    )
    allocation.assert_disjoint()
    return allocation


def allocated_images(
    allocation: F3BEvaluationAllocation,
    phase: str,
    resolution: int,
    root: str | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    pool = cifar.cifar_pool(resolution, "eval", root)
    if phase == "development":
        reference = allocation.development_reference
        controls = allocation.development_controls
    elif phase == "confirmation":
        reference = allocation.confirmation_reference
        controls = allocation.confirmation_controls
    else:
        raise ValueError("phase must be development or confirmation")
    return (
        pool[torch.as_tensor(reference)],
        tuple(pool[torch.as_tensor(indices)] for indices in controls),
    )


def matched_real_metrics(
    control: torch.Tensor, reference_features: np.ndarray, device
) -> dict:
    features = inception_features(control.cpu(), device).double().numpy()
    result = precision_recall(features, reference_features)
    result["kid"] = kid_from_features(features, reference_features)
    return result


def generated_metrics(
    images: torch.Tensor,
    reference_features: np.ndarray,
    real_for_scale: torch.Tensor,
    device,
    include_fid: bool = False,
) -> dict:
    """Report quality/coverage plus raw collapse and range diagnostics."""
    features = inception_features(images.cpu(), device).double().numpy()
    pr = precision_recall(features, reference_features)
    scale = float(_self_distances(real_for_scale.cpu()).min(dim=1).values.median())
    try:
        spectrum_alpha = spectrum_slope(images.cpu())["alpha"]
    except ValueError:
        # The 8x8 mechanics profile has no valid mid-frequency regression
        # band. Scientific 32x32 profiles must produce the measurement.
        if images.shape[-1] >= 16:
            raise
        spectrum_alpha = None
    result = {
        "recall": pr["recall"],
        "precision": pr["precision"],
        "pr_f1": pr["f1"],
        "kid": kid_from_features(features, reference_features),
        "fid": (
            frechet_from_features(features, reference_features) if include_fid else None
        ),
        "fid_is_small_sample_indicative": bool(include_fid),
        "effective_rank": effective_rank(images.cpu()),
        "duplicate_rate": duplicate_rate(images.cpu(), scale),
        "nn_diversity": nn_diversity(images.cpu()),
        "spectrum_alpha": spectrum_alpha,
        "second_moment": float(
            images.cpu().flatten(1).var(0).mean()
            / real_for_scale.cpu().flatten(1).var(0).mean().clamp_min(1e-12)
        ),
        "outside_fraction": float(((images < -1.0) | (images > 1.0)).float().mean()),
        "sample_outside_fraction": float(
            ((images < -1.0) | (images > 1.0)).flatten(1).any(dim=1).float().mean()
        ),
        "minimum": float(images.min()),
        "maximum": float(images.max()),
        "rms": float(images.square().mean().sqrt()),
        "samples_generated": len(images),
        "samples_reference": len(reference_features),
    }
    return result


def memorization_statistics(
    images: torch.Tensor, train: torch.Tensor, normalizer: float, device
) -> dict:
    distances, claimed = nearest_reference_distances(images, train, device)
    return {
        "nearest_train_normalized": float(distances.median()) / normalizer,
        "nearest_train_p05_normalized": float(np.percentile(distances.numpy(), 5))
        / normalizer,
        "distinct_train_claimed": int(torch.unique(claimed).numel()),
        "claimed_train_fraction": float(torch.unique(claimed).numel()) / len(images),
    }


def apply_vetoes(metrics: dict, memorization: dict, thresholds: dict) -> dict:
    comparisons = {
        "effective_rank": {
            "value": float(metrics["effective_rank"]),
            "threshold": float(thresholds["effective_rank"]),
        },
        "one_minus_duplicate_rate": {
            "value": 1.0 - float(metrics["duplicate_rate"]),
            "threshold": float(thresholds["one_minus_duplicate_rate"]),
        },
        "nn_diversity": {
            "value": float(metrics["nn_diversity"]),
            "threshold": float(thresholds["nn_diversity"]),
        },
        "nearest_train_normalized": {
            "value": float(memorization["nearest_train_normalized"]),
            "threshold": float(thresholds["nearest_train_normalized"]),
        },
    }
    for item in comparisons.values():
        item["passes"] = bool(item["value"] >= item["threshold"])
    return {
        "comparisons": comparisons,
        "passes": bool(all(item["passes"] for item in comparisons.values())),
    }


def real_health_statistics(images: torch.Tensor) -> dict:
    scale = float(_self_distances(images).min(dim=1).values.median())
    return {
        "effective_rank": effective_rank(images),
        "one_minus_duplicate_rate": 1.0 - duplicate_rate(images, scale),
        "nn_diversity": nn_diversity(images),
        "real_nn_scale": scale,
    }
