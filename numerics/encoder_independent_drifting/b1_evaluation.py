"""Fresh allocations, paired audits, and conservative B1 adjudication."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from . import cifar
from .b1 import (
    B1Config,
    b1_seed,
    build_bank_for_dimension,
)
from .f1 import _self_distances, duplicate_rate, effective_rank, nn_diversity
from .f1_k200 import nearest_reference_distances
from .f3b import F3BModelConfig, sample_model
from .f3b_evaluation import evaluation_allocation as b0_evaluation_allocation
from .spectral_anchor import anchor_loss, anchor_loss_by_band


@dataclass(frozen=True)
class B1EvaluationAllocation:
    """B0-unused calibration data plus the untouched official test set."""

    calibration_reference: np.ndarray
    calibration_controls: tuple[np.ndarray, ...]
    confirmation_reference: np.ndarray
    confirmation_controls: tuple[np.ndarray, ...]
    audit_target_pairs: tuple[tuple[np.ndarray, np.ndarray], ...]
    calibration_unused: np.ndarray
    confirmation_unused: np.ndarray

    @property
    def digests(self) -> dict[str, str]:
        arrays: dict[str, np.ndarray] = {
            "calibration_reference": self.calibration_reference,
            "confirmation_reference": self.confirmation_reference,
            "calibration_unused": self.calibration_unused,
            "confirmation_unused": self.confirmation_unused,
        }
        arrays |= {
            f"calibration_control_{index}": values
            for index, values in enumerate(self.calibration_controls)
        }
        arrays |= {
            f"confirmation_control_{index}": values
            for index, values in enumerate(self.confirmation_controls)
        }
        for index, (left, right) in enumerate(self.audit_target_pairs):
            arrays[f"audit_target_{index}_left"] = left
            arrays[f"audit_target_{index}_right"] = right
        return {
            name: hashlib.sha256(
                np.asarray(values, dtype=np.int64).tobytes()
            ).hexdigest()[:16]
            for name, values in arrays.items()
        }

    def assert_disjoint(self) -> None:
        calibration = np.concatenate(
            [
                self.calibration_reference,
                *self.calibration_controls,
                self.calibration_unused,
            ]
        )
        if len(calibration) != len(np.unique(calibration)):
            raise AssertionError("B1 calibration roles overlap")
        confirmation = np.concatenate(
            [
                self.confirmation_reference,
                *self.confirmation_controls,
                *[side for pair in self.audit_target_pairs for side in pair],
                self.confirmation_unused,
            ]
        )
        if len(confirmation) != len(np.unique(confirmation)):
            raise AssertionError("B1 confirmation roles overlap")


def evaluation_allocation(
    config: B1Config,
    generated_samples: int = 512,
    reference_samples: int = 2_048,
    groups: int = 3,
) -> B1EvaluationAllocation:
    """Allocate B1 roles before any B1 outcome is observed."""
    config.validate()
    legacy = b0_evaluation_allocation(
        10_000, reference_samples, generated_samples, groups
    )
    # ``legacy.unused`` is sorted for stable storage.  Re-permute it before
    # assigning B1 roles so calibration is not accidentally the lowest-index
    # portion of the old evaluation split.
    available = np.random.default_rng(
        b1_seed("allocation", 0, "b0-unused-calibration")
    ).permutation(legacy.unused)
    needed_calibration = generated_samples * (groups + 1)
    if len(available) < needed_calibration:
        raise ValueError("B0 left too few calibration images for B1")
    calibration_reference = np.sort(available[:generated_samples])
    cursor = generated_samples
    calibration_controls = []
    for _ in range(groups):
        calibration_controls.append(
            np.sort(available[cursor : cursor + generated_samples])
        )
        cursor += generated_samples
    calibration_unused = np.sort(available[cursor:])

    rng = np.random.default_rng(b1_seed("allocation", 0, "official-test"))
    order = rng.permutation(10_000)
    cursor = 0

    def take(count: int) -> np.ndarray:
        nonlocal cursor
        result = np.sort(order[cursor : cursor + count])
        cursor += count
        return result

    confirmation_reference = take(reference_samples)
    confirmation_controls = tuple(take(generated_samples) for _ in range(groups))
    audit_pairs = tuple(
        (take(generated_samples), take(generated_samples))
        for _ in range(config.audit_replicates)
    )
    result = B1EvaluationAllocation(
        calibration_reference=calibration_reference,
        calibration_controls=tuple(calibration_controls),
        confirmation_reference=confirmation_reference,
        confirmation_controls=confirmation_controls,
        audit_target_pairs=audit_pairs,
        calibration_unused=calibration_unused,
        confirmation_unused=np.sort(order[cursor:]),
    )
    result.assert_disjoint()
    return result


def calibration_images(
    allocation: B1EvaluationAllocation,
    resolution: int,
    root: str | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    pool = cifar.cifar_pool(resolution, "eval", root)
    return (
        pool[torch.as_tensor(allocation.calibration_reference)],
        tuple(
            pool[torch.as_tensor(indices)]
            for indices in allocation.calibration_controls
        ),
    )


def confirmation_images(
    allocation: B1EvaluationAllocation,
    resolution: int,
    root: str | None = None,
) -> tuple[
    torch.Tensor,
    tuple[torch.Tensor, ...],
    tuple[tuple[torch.Tensor, torch.Tensor], ...],
]:
    pool = cifar.cifar_pool(resolution, "test", root)
    reference = pool[torch.as_tensor(allocation.confirmation_reference)]
    controls = tuple(
        pool[torch.as_tensor(indices)] for indices in allocation.confirmation_controls
    )
    targets = tuple(
        (
            pool[torch.as_tensor(left)],
            pool[torch.as_tensor(right)],
        )
        for left, right in allocation.audit_target_pairs
    )
    return reference, controls, targets


def memorization_statistics_augmented(
    images: torch.Tensor,
    train: torch.Tensor,
    normalizer: float,
    device: torch.device | str,
) -> dict:
    """Nearest training image over the actual identity/flip augmentation orbit."""
    if normalizer <= 0:
        raise ValueError("memorization normalizer must be positive")
    direct_distance, direct_claim = nearest_reference_distances(images, train, device)
    flipped_distance, flipped_claim = nearest_reference_distances(
        images, torch.flip(train, dims=(-1,)), device
    )
    use_flip = flipped_distance < direct_distance
    distances = torch.where(use_flip, flipped_distance, direct_distance)
    claimed = torch.where(use_flip, flipped_claim + len(train), direct_claim)
    unique = torch.unique(claimed)
    return {
        "nearest_train_or_flip_normalized": float(distances.median()) / normalizer,
        "nearest_train_or_flip_p05_normalized": float(
            np.percentile(distances.numpy(), 5)
        )
        / normalizer,
        "distinct_augmented_train_claimed": int(unique.numel()),
        "claimed_augmented_train_fraction": float(unique.numel()) / len(images),
        "flip_claim_fraction": float(use_flip.float().mean()),
    }


def augmented_real_health(
    images: torch.Tensor,
    train: torch.Tensor,
    device: torch.device | str,
) -> dict:
    scale = float(_self_distances(images).min(dim=1).values.median())
    direct, _ = nearest_reference_distances(images, train, device)
    flipped, _ = nearest_reference_distances(
        images, torch.flip(train, dims=(-1,)), device
    )
    nearest = torch.minimum(direct, flipped)
    return {
        "effective_rank": effective_rank(images),
        "one_minus_duplicate_rate": 1.0 - duplicate_rate(images, scale),
        "nn_diversity": nn_diversity(images),
        "real_nn_scale": scale,
        "nearest_augmented_train": float(nearest.median()),
    }


def apply_b1_vetoes(metrics: dict, memorization: dict, thresholds: dict) -> dict:
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
        "nearest_train_or_flip_normalized": {
            "value": float(memorization["nearest_train_or_flip_normalized"]),
            "threshold": float(thresholds["nearest_train_or_flip_normalized"]),
        },
    }
    for item in comparisons.values():
        item["passes"] = bool(item["value"] >= item["threshold"])
    return {
        "comparisons": comparisons,
        "passes": bool(all(item["passes"] for item in comparisons.values())),
    }


def anchor_audit_suite(
    model: torch.nn.Module,
    model_config: F3BModelConfig,
    nfe: int,
    unit: int,
    scale: float,
    target_pairs: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    device: torch.device | str,
    config: B1Config,
) -> list[dict]:
    """Common-random B0/B1 audits over independent banks and batches."""
    if len(target_pairs) != config.audit_replicates:
        raise ValueError("audit target allocation differs from B1 config")
    dimension = model_config.channels * model_config.image_size**2
    rows = []
    for replicate, (target, real_control) in enumerate(target_pairs):
        bank = build_bank_for_dimension(
            scale,
            dimension,
            "confirmation-audit",
            replicate,
            "held-out-bank",
            config.audit_features,
            config,
        )
        prior_seed = b1_seed("confirmation-audit", unit, "prior", replicate)
        generated = sample_model(
            model,
            len(target),
            model_config,
            nfe,
            prior_seed,
            device,
        )
        with torch.no_grad():
            biased = float(anchor_loss(bank, generated, target, estimator="biased"))
            unbiased = float(anchor_loss(bank, generated, target, estimator="unbiased"))
            real_biased = float(
                anchor_loss(bank, real_control, target, estimator="biased")
            )
            real_unbiased = float(
                anchor_loss(bank, real_control, target, estimator="unbiased")
            )
            bands = anchor_loss_by_band(bank, generated, target, estimator="unbiased")
        rows.append(
            {
                "replicate": replicate,
                "bank_seed": bank.seed,
                "bank_summary": bank.summary(),
                "prior_seed": prior_seed,
                "biased": biased,
                "unbiased": unbiased,
                "real_real_biased": real_biased,
                "real_real_unbiased": real_unbiased,
                "biased_excess_over_real": biased - real_biased,
                "unbiased_excess_over_real": unbiased - real_unbiased,
                "unbiased_by_band": bands,
            }
        )
    return rows


def summarize_anchor_audit(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("anchor audit cannot be empty")

    def median(name: str) -> float:
        return float(np.median([float(row[name]) for row in rows]))

    return {
        "replicates": len(rows),
        "median_biased": median("biased"),
        "median_unbiased": median("unbiased"),
        "median_real_real_biased": median("real_real_biased"),
        "median_real_real_unbiased": median("real_real_unbiased"),
        "median_biased_excess_over_real": median("biased_excess_over_real"),
        "median_unbiased_excess_over_real": median("unbiased_excess_over_real"),
        "replicates_above_real_floor": int(
            sum(float(row["biased_excess_over_real"]) > 0 for row in rows)
        ),
    }


def compare_anchor_audits(
    candidate: list[dict], baseline: list[dict], config: B1Config
) -> dict:
    """Gate on nonnegative-floor-adjusted, common-random paired estimates."""
    if len(candidate) != len(baseline) or len(candidate) != config.audit_replicates:
        raise ValueError("candidate and baseline audit replicates differ")
    candidate_by = {int(row["replicate"]): row for row in candidate}
    baseline_by = {int(row["replicate"]): row for row in baseline}
    if set(candidate_by) != set(baseline_by):
        raise ValueError("candidate and baseline audit IDs differ")
    paired = []
    for replicate in sorted(candidate_by):
        if int(candidate_by[replicate]["prior_seed"]) != int(
            baseline_by[replicate]["prior_seed"]
        ):
            raise ValueError("candidate and baseline used different audit priors")
        if int(candidate_by[replicate]["bank_seed"]) != int(
            baseline_by[replicate]["bank_seed"]
        ):
            raise ValueError("candidate and baseline used different audit banks")
        if not np.isclose(
            float(candidate_by[replicate]["real_real_biased"]),
            float(baseline_by[replicate]["real_real_biased"]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("candidate and baseline used different audit floors")
        new = float(candidate_by[replicate]["biased_excess_over_real"])
        old = float(baseline_by[replicate]["biased_excess_over_real"])
        paired.append(
            {
                "replicate": replicate,
                "candidate_excess": new,
                "baseline_excess": old,
                "difference": new - old,
                "improves": bool(new < old),
            }
        )
    candidate_median = float(np.median([row["candidate_excess"] for row in paired]))
    baseline_median = float(np.median([row["baseline_excess"] for row in paired]))
    paired_wins = int(sum(row["improves"] for row in paired))
    resolvable = bool(baseline_median > 0)
    fraction_pass = bool(
        resolvable
        and candidate_median <= config.anchor_reduction_fraction * baseline_median
    )
    wins_pass = bool(paired_wins >= config.anchor_paired_wins_required)
    return {
        "paired": paired,
        "candidate_median_excess": candidate_median,
        "baseline_median_excess": baseline_median,
        "baseline_resolvable": resolvable,
        "reduction_fraction": config.anchor_reduction_fraction,
        "fraction_passes": fraction_pass,
        "paired_wins": paired_wins,
        "paired_wins_required": config.anchor_paired_wins_required,
        "paired_wins_pass": wins_pass,
        "passes": bool(fraction_pass and wins_pass),
    }


def adjudicate_b1(
    rows: list[dict],
    baseline_rows: list[dict],
    control_rows: list[dict],
    config: B1Config,
) -> dict:
    """Per-unit paired gate with no median-control shortcut."""
    baseline = {int(row["unit"]): row for row in baseline_rows}
    controls = {int(row["group"]): row for row in control_rows}
    units = {}
    for row in rows:
        unit = int(row["unit"])
        control_group = int(row["assigned_control_group"])
        if unit not in baseline or control_group not in controls:
            raise ValueError("B1 unit lacks its paired baseline/control")
        recall = float(row["metrics"]["recall"])
        baseline_recall = float(baseline[unit]["metrics"]["recall"])
        recall_floor = baseline_recall - config.recall_noninferiority_margin
        recall_pass = bool(recall >= recall_floor)
        control_pass = bool(
            float(controls[control_group]["recall"]) > config.metric_control_floor
        )
        anchor_pass = bool(row["anchor_comparison"]["passes"])
        veto_pass = bool(row["veto"]["passes"])
        units[str(unit)] = {
            "recall": recall,
            "baseline_recall": baseline_recall,
            "recall_floor": recall_floor,
            "recall_noninferior": recall_pass,
            "anchor_reduction_passes": anchor_pass,
            "veto_passes": veto_pass,
            "metric_control_group": control_group,
            "metric_control_recall": float(controls[control_group]["recall"]),
            "metric_control_passes": control_pass,
            "unit_passes": bool(
                recall_pass and anchor_pass and veto_pass and control_pass
            ),
        }
    if len(units) != 3:
        raise ValueError("B1 confirmation requires exactly three paired units")
    passes = int(sum(value["unit_passes"] for value in units.values()))
    invalid_controls = [
        unit for unit, value in units.items() if not value["metric_control_passes"]
    ]
    decision = "VOID" if invalid_controls else ("PASS" if passes >= 2 else "FAIL")
    return {
        "decision": decision,
        "units_passing": passes,
        "units": units,
        "invalid_control_units": invalid_controls,
        "reading": {
            "PASS": (
                "PASS: in at least two paired units B1 retained B0 recall within "
                "the frozen margin, reduced held-out source-anchor excess, and "
                "passed collapse, memorization, and metric controls"
            ),
            "FAIL": (
                "FAIL: this frozen anchor configuration did not establish the "
                "joint retention-and-anchor-reduction gate"
            ),
            "VOID": "VOID: at least one assigned matched-real control was invalid",
        }[decision],
    }
