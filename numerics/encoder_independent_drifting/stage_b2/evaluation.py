"""Fresh allocations and paired drift-energy audits for Stage B2."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch

from .core import B2Config, b2_seed, laplace_drift_energy


def _digest(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[
        :16
    ]


@dataclass(frozen=True)
class B2AuditAllocation:
    """Disjoint roles from a genuinely fresh confirmation population."""

    source_id: str
    reference: np.ndarray
    controls: tuple[np.ndarray, ...]
    probe_centres: tuple[np.ndarray, ...]
    positives: tuple[np.ndarray, ...]
    floor_negatives: tuple[np.ndarray, ...]
    unused: np.ndarray

    @property
    def digests(self) -> dict[str, str]:
        result = {
            "source_id": hashlib.sha256(self.source_id.encode("utf-8")).hexdigest()[
                :16
            ],
            "reference": _digest(self.reference),
            "unused": _digest(self.unused),
        }
        for index, values in enumerate(self.controls):
            result[f"control_{index}"] = _digest(values)
        for index, values in enumerate(self.probe_centres):
            result[f"audit_{index}_probe_centres"] = _digest(values)
        for index, values in enumerate(self.positives):
            result[f"audit_{index}_positives"] = _digest(values)
        for index, values in enumerate(self.floor_negatives):
            result[f"audit_{index}_floor_negatives"] = _digest(values)
        return result

    def assert_disjoint(self) -> None:
        roles = [
            self.reference,
            *self.controls,
            *self.probe_centres,
            *self.positives,
            *self.floor_negatives,
            self.unused,
        ]
        combined = np.concatenate(roles)
        if len(combined) != len(np.unique(combined)):
            raise AssertionError("B2 fresh-confirmation roles overlap")


_EXHAUSTED_SOURCE_IDS = {
    "cifar10-test",
    "cifar-10-test",
    "official-cifar10-test",
    "official-cifar-10-test",
}


def fresh_evaluation_allocation(
    pool_size: int,
    source_id: str,
    config: B2Config,
    *,
    generated_samples: int = 512,
    reference_samples: int = 2_048,
    control_groups: int = 3,
) -> B2AuditAllocation:
    """Allocate B2 confirmation roles, refusing B1's exhausted test source."""
    config.validate()
    normalized_source = source_id.strip().lower()
    if not normalized_source:
        raise ValueError("B2 confirmation requires a stable fresh source ID")
    if normalized_source in _EXHAUSTED_SOURCE_IDS:
        raise ValueError(
            "the official CIFAR-10 test set was adaptively consumed by B1; "
            "B2 requires a new confirmation source"
        )
    if min(pool_size, generated_samples, reference_samples, control_groups) <= 0:
        raise ValueError("invalid B2 confirmation allocation size")
    required = (
        reference_samples
        + control_groups * generated_samples
        + 3 * config.audit_replicates * config.audit_batch
    )
    if pool_size < required:
        raise ValueError(
            f"fresh B2 source has {pool_size} samples but the frozen roles need {required}"
        )
    order = np.random.default_rng(
        b2_seed("fresh-allocation", source_id, "permutation")
    ).permutation(pool_size)
    cursor = 0

    def take(count: int) -> np.ndarray:
        nonlocal cursor
        result = np.sort(order[cursor : cursor + count])
        cursor += count
        return result

    result = B2AuditAllocation(
        source_id=source_id,
        reference=take(reference_samples),
        controls=tuple(take(generated_samples) for _ in range(control_groups)),
        probe_centres=tuple(
            take(config.audit_batch) for _ in range(config.audit_replicates)
        ),
        positives=tuple(
            take(config.audit_batch) for _ in range(config.audit_replicates)
        ),
        floor_negatives=tuple(
            take(config.audit_batch) for _ in range(config.audit_replicates)
        ),
        unused=np.sort(order[cursor:]),
    )
    result.assert_disjoint()
    return result


def audit_role_tensors(
    pool: torch.Tensor,
    allocation: B2AuditAllocation,
    replicate: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 0 <= replicate < len(allocation.probe_centres):
        raise ValueError("unknown B2 audit replicate")
    indices = (
        allocation.probe_centres[replicate],
        allocation.positives[replicate],
        allocation.floor_negatives[replicate],
    )
    return tuple(pool[torch.as_tensor(item)].to(device) for item in indices)  # type: ignore[return-value]


def drift_energy_audit_suite(
    generated_batches: list[torch.Tensor],
    fresh_pool: torch.Tensor,
    allocation: B2AuditAllocation,
    tau: float,
    unit: int,
    config: B2Config,
    device: torch.device | str,
) -> list[dict]:
    """Evaluate model-vs-target energy minus an independent real-real floor."""
    config.validate()
    if len(generated_batches) != config.audit_replicates:
        raise ValueError("B2 audit needs one generated batch per replicate")
    rows: list[dict] = []
    for replicate, generated in enumerate(generated_batches):
        if len(generated) != config.audit_batch:
            raise ValueError("generated B2 audit batch has the wrong size")
        centres, positive, floor_negative = audit_role_tensors(
            fresh_pool, allocation, replicate, device
        )
        noise_seed = b2_seed("confirmation-audit", unit, "probe-noise", replicate)
        generator = torch.Generator(device="cpu").manual_seed(noise_seed % (2**63 - 1))
        noise = torch.randn(centres.shape, generator=generator, dtype=centres.dtype).to(
            device
        )
        probes = centres + config.probe_noise_std * noise
        with torch.no_grad():
            model_energy, model_health = laplace_drift_energy(
                probes,
                positive,
                generated.to(device),
                tau,
                diagnostics=True,
            )
            floor_energy, floor_health = laplace_drift_energy(
                probes,
                positive,
                floor_negative,
                tau,
                diagnostics=True,
            )
        model_value = float(model_energy)
        floor_value = float(floor_energy)
        rows.append(
            {
                "replicate": replicate,
                "probe_noise_seed": noise_seed,
                "tau": float(tau),
                "energy": model_value,
                "real_real_floor": floor_value,
                "excess_over_real": model_value - floor_value,
                "model_health": model_health,
                "floor_health": floor_health,
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
        raise ValueError("cannot summarize an empty B2 audit")
    return {
        "median_energy": float(np.median([row["energy"] for row in rows])),
        "median_real_real_floor": float(
            np.median([row["real_real_floor"] for row in rows])
        ),
        "median_excess_over_real": float(
            np.median([row["excess_over_real"] for row in rows])
        ),
    }


def compare_drift_audits(
    candidate: list[dict], baseline: list[dict], config: B2Config
) -> dict:
    if len(candidate) != len(baseline) or len(candidate) != config.audit_replicates:
        raise ValueError("B2 candidate and baseline audit counts differ")
    paired = []
    for candidate_row, baseline_row in zip(candidate, baseline, strict=True):
        for key in ("replicate", "probe_noise_seed", "tau", "allocation_digests"):
            if candidate_row[key] != baseline_row[key]:
                raise ValueError(f"B2 paired audit differs in {key}")
        if not math_isclose(
            float(candidate_row["real_real_floor"]),
            float(baseline_row["real_real_floor"]),
        ):
            raise ValueError("B2 paired audit used a different real-real floor")
        difference = float(candidate_row["excess_over_real"]) - float(
            baseline_row["excess_over_real"]
        )
        paired.append(
            {
                "replicate": int(candidate_row["replicate"]),
                "candidate_excess": float(candidate_row["excess_over_real"]),
                "baseline_excess": float(baseline_row["excess_over_real"]),
                "difference": difference,
                "improves": bool(difference < 0),
            }
        )
    candidate_median = float(np.median([row["candidate_excess"] for row in paired]))
    baseline_median = float(np.median([row["baseline_excess"] for row in paired]))
    wins = sum(row["improves"] for row in paired)
    resolvable = baseline_median > 0
    fraction_passes = bool(
        resolvable
        and candidate_median <= config.drift_reduction_fraction * baseline_median
    )
    return {
        "candidate_median_excess": candidate_median,
        "baseline_median_excess": baseline_median,
        "baseline_resolvable": resolvable,
        "paired_wins": int(wins),
        "fraction_passes": fraction_passes,
        "win_count_passes": bool(wins >= config.drift_paired_wins_required),
        "passes": bool(fraction_passes and wins >= config.drift_paired_wins_required),
        "paired": paired,
    }


def math_isclose(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=1e-10, atol=1e-12))


def adjudicate_b2(
    rows: list[dict],
    baseline_rows: list[dict],
    matched_real_controls: list[dict],
    config: B2Config,
) -> dict:
    """Per-unit mechanism gate; it does not adjudicate identifiability."""
    baseline_by_unit = {int(row["unit"]): row for row in baseline_rows}
    control_by_group = {int(row["group"]): row for row in matched_real_controls}
    units = {}
    invalid_controls = []
    for row in rows:
        unit = int(row["unit"])
        baseline = baseline_by_unit[unit]
        group = int(row["assigned_control_group"])
        control = control_by_group[group]
        control_passes = float(control["recall"]) > config.metric_control_floor
        if not control_passes:
            invalid_controls.append(unit)
        baseline_recall = float(baseline["metrics"]["recall"])
        recall = float(row["metrics"]["recall"])
        recall_floor = baseline_recall - config.recall_noninferiority_margin
        recall_passes = recall >= recall_floor
        drift_passes = bool(row["drift_comparison"]["passes"])
        veto_passes = bool(row["veto"]["passes"])
        units[str(unit)] = {
            "recall": recall,
            "baseline_recall": baseline_recall,
            "recall_floor": recall_floor,
            "recall_noninferior": recall_passes,
            "drift_reduction_passes": drift_passes,
            "veto_passes": veto_passes,
            "metric_control_group": group,
            "metric_control_recall": float(control["recall"]),
            "metric_control_passes": control_passes,
            "unit_passes": bool(
                recall_passes and drift_passes and veto_passes and control_passes
            ),
        }
    if len(units) != 3:
        raise ValueError("B2 confirmation requires exactly three paired units")
    passes = sum(item["unit_passes"] for item in units.values())
    if invalid_controls:
        decision = "VOID"
    else:
        decision = "PASS" if passes >= 2 else "FAIL"
    return {
        "decision": decision,
        "units_passing": int(passes),
        "invalid_control_units": invalid_controls,
        "units": units,
        "reading": {
            "PASS": (
                "PASS: in at least two paired units the differentiable "
                "normalized-Laplace correction retained B0 recall and reduced "
                "the frozen held-out drift-energy excess"
            ),
            "FAIL": (
                "FAIL: this correction did not jointly retain coverage and "
                "reduce its frozen held-out discrepancy"
            ),
            "VOID": "VOID: a matched-real metric control was invalid",
        }[decision],
    }
