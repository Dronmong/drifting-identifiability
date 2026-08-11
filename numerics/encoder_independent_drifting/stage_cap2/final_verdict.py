"""Immutable developmental verdict for completed CAP2 300k continuations.

This is the terminal *local development* certificate for the paid CAP2 screen.
It does not inspect CIFAR-10 test data and does not turn a one-seed screen into
a confirmation claim.  Every decision is recomputed from the selected arms'
immutable 300k result, final raw/EMA checkpoints, fresh raw-checkpoint
numerical admission, strict rolling recovery, durable commits, and fixed 50k
train-reference evaluation with retained feature/PNG evidence.

The 2,048-sample repository feature metrics are retained as diagnostics and
absolute collapse vetoes only.  A paired ordered-over-legacy claim is decided
solely by 50k CleanFID and 50k CleanKID, each beyond the direct real/real
calibration discrepancy.  Both arms must retain their own 150k quality, and
the ordered arm must retain its absolute improvement over the historical
baseline.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from ..stage_cap.config import CAPGateConfig
from ..stage_cap.diagnostics import capability_gate
from ..stage_cap.training import load_recovery_payload, validate_recovery_counters
from .artifacts import (
    assert_unused,
    file_sha256,
    load_checkpoint,
    load_preflight,
    profile_payload,
    verify_file,
    verify_json,
    write_json_atomic,
)
from .config import SAMPLER_ARMS, apply_calibrated_gate, screen_profile
from .durable_mirror import DurableMirror
from .metric_calibration import revalidate_metric_calibration_evidence
from .numerical_admission import admission_matrix_complete
from .preflight import _clean_metrics_valid, _same_hardware, _same_kid_reference
from .promotion import (
    CALIBRATION_MARGIN_KIND,
    DEVELOPMENT_GENERATION_SEED,
    DEVELOPMENT_KID_SEED,
    DEVELOPMENT_SAMPLES,
    FID_KEY,
    FIXED_INFERENCE_CORNER_SAMPLES,
    KID_KEY,
    MAX_EXACT_COPY_FRACTION,
    MAX_GENERATED_DUPLICATE_FRACTION,
    MAX_LATE_INFERENCE_CORNER_ERROR_GROWTH,
    MIN_PR_F1,
    MIN_PRECISION,
    MIN_RECALL,
    _all_true,
    _calibration_tolerance,
    _checkpoint_previews_valid,
    _development_evaluation_schema,
    _evaluation_environment,
    _grid_integrity,
    _reference_binding,
    revalidate_promotion,
)
from .selection import LEGACY_ARM, ORDERED_ARMS, revalidate_selection
from .standard_metrics import revalidate_clean_evaluation_evidence

FINAL_VERDICT_STATUS = "cap-emf2-300k-final-verdict"
PROMOTION_STEP = 150_000
FINAL_STEP = 300_000
FINAL_CORNER_FROM_STEP = 250_000
POLICY_VERSION = "cap2-final-paired-retention-continuity-v2"


@dataclass(frozen=True)
class FinalArmArtifacts:
    """All immutable artifacts required for one selected 300k arm."""

    result: Path
    raw_checkpoint: Path
    ema_checkpoint: Path
    raw_readmission: Path
    evaluation: Path
    mirror_root: Path


def _reference(path: Path, anchor: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()
    except ValueError:
        # Windows cannot express a relative path across drive letters.  The
        # explicit durable trust root is therefore allowed to remain absolute.
        return str(path.resolve())


def _resolve(reference: object, anchor: Path) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("final verdict contains an empty artifact reference")
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (anchor / path).resolve()


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metric(payload: object, key: str) -> float:
    if not isinstance(payload, dict):
        return math.nan
    value = payload.get(key)
    return float(value) if _finite(value) else math.nan


def _hash64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_recovery_authorization(promotion: dict, selection: dict) -> dict:
    return {
        "status": "cap-emf2-300k-recovery-authorization",
        "promotion_sha256": promotion["artifact_sha256"],
        "preflight_sha256": promotion["preflight_sha256"],
        "result_150k_sha256": promotion["result_sha256"],
        "checkpoint_150k_ema_sha256": promotion["checkpoint_sha256"],
        "checkpoint_150k_raw_sha256": promotion["raw_checkpoint_sha256"],
        "readmission_sha256": promotion["readmission_sha256"],
        "development_evaluation_sha256": promotion["development_evaluation_sha256"],
        "selection_sha256": selection["artifact_sha256"],
        "arm": promotion["arm"],
        "candidate": promotion["candidate"],
        "from_updates": 150_000,
        "to_updates": FINAL_STEP,
    }


def _expected_profiles(
    preflight: dict, arm: str, candidate: str, *, updates: int = FINAL_STEP
) -> tuple[dict, dict]:
    declared = profile_payload(
        apply_calibrated_gate(
            screen_profile(arm, candidate, updates=updates),
            preflight["inputs"]["gate_calibration"],
        )
    )
    realized = copy.deepcopy(declared)
    benchmark = preflight["inputs"]["benchmark"]
    realized["train"]["micro_batch"] = int(benchmark["micro_batch"])
    realized["train"]["accumulation_steps"] = int(benchmark["accumulation_steps"])
    realized["train"]["effective_batch"] = int(benchmark["effective_batch"])
    return declared, realized


def _recorded_checkpoint_path(result_path: Path, record: object) -> Path | None:
    if not isinstance(record, dict):
        return None
    value = record.get("path")
    if not isinstance(value, str) or not value.strip():
        return None
    return _resolve(value, result_path.parent)


def _checkpoint_ladder_valid(result: dict, declared: dict) -> bool:
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, dict):
        return False
    expected_steps = {
        str(int(step))
        for step in declared.get("train", {}).get("checkpoint_updates", [])
    }
    if set(checkpoints) != expected_steps or str(FINAL_STEP) not in checkpoints:
        return False
    return all(
        isinstance(record, dict)
        and set(record) == {"raw", "ema"}
        and all(
            isinstance(record[kind], dict)
            and set(record[kind]) == {"path", "sha256", "durable_mirror"}
            and isinstance(record[kind].get("path"), str)
            and bool(record[kind]["path"].strip())
            and _hash64(record[kind].get("sha256"))
            and isinstance(record[kind].get("durable_mirror"), dict)
            for kind in ("raw", "ema")
        )
        for record in checkpoints.values()
    )


def _state_dict_exact(left: object, right: object) -> bool:
    """Return whether two state dictionaries contain bit-identical tensors."""
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if set(left) != set(right):
        return False
    for name in left:
        first = left[name]
        second = right[name]
        if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
            return False
        if first.dtype != second.dtype or first.shape != second.shape:
            return False
        if not torch.equal(first.detach().cpu(), second.detach().cpu()):
            return False
    return True


def _ema_recovery_state(recovery: dict) -> dict | None:
    ema = recovery.get("ema")
    if not isinstance(ema, dict):
        return None
    shadow = ema.get("shadow")
    buffers = ema.get("buffers")
    if not isinstance(shadow, dict) or not isinstance(buffers, dict):
        return None
    if set(shadow) & set(buffers):
        return None
    return {**shadow, **buffers}


def _horizon_neutral_profile(profile: dict) -> dict:
    result = copy.deepcopy(profile)
    train = result.get("train")
    if not isinstance(train, dict):
        return {}
    train.pop("updates", None)
    train.pop("checkpoint_updates", None)
    return result


def _recomputed_capability_gate(result: dict, declared: dict) -> dict | None:
    """Re-run the terminal gate from immutable health and counter leaves."""
    training = result.get("training")
    if not isinstance(training, dict):
        return None
    health = training.get("health")
    if not isinstance(health, list) or not health:
        return None
    final_record = health[-1]
    if not isinstance(final_record, dict) or final_record.get("step") != FINAL_STEP:
        return None
    final = final_record.get("ema", final_record)
    ema_records = [
        record.get("ema")
        for record in health
        if isinstance(record, dict) and isinstance(record.get("ema"), dict)
    ]
    try:
        best_rank = max(float(record["effective_rank_ratio"]) for record in ema_records)
        final_window = int(training["final_window_updates"])
        clipped = int(training["clipped_updates_final_window"])
        clip_fraction = clipped / final_window if final_window else 0.0
        gate = CAPGateConfig(**declared["gate"])
        return capability_gate(
            final,
            best_rank,
            clip_fraction,
            int(training["nonfinite_updates"]),
            int(training["inference_forward_calls"]),
            gate,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _training_cadence_valid(result: dict, declared: dict) -> bool:
    """Prove that gate recomputation sees every scheduled immutable record."""
    training = result.get("training")
    train = declared.get("train")
    if not isinstance(training, dict) or not isinstance(train, dict):
        return False
    history = training.get("history")
    health = training.get("health")
    if not isinstance(history, list) or not isinstance(health, list):
        return False
    if any(
        not isinstance(record, dict)
        or isinstance(record.get("step"), bool)
        or not isinstance(record.get("step"), int)
        for record in [*history, *health]
    ):
        return False
    try:
        updates = int(train["updates"])
        log_every = int(train["log_every"])
        health_every = int(train["health_every"])
        checkpoints = {int(step) for step in train["checkpoint_updates"]}
        history_steps = [record["step"] for record in history]
        health_steps = [record["step"] for record in health]
        ema_steps = {
            int(record["step"])
            for record in health
            if isinstance(record, dict) and isinstance(record.get("ema"), dict)
        }
    except (KeyError, TypeError, ValueError):
        return False
    if min(log_every, health_every, updates) <= 0:
        return False
    expected_history = list(range(log_every, updates + 1, log_every))
    if not expected_history or expected_history[-1] != updates:
        expected_history.append(updates)
    expected_health = sorted(
        set(range(health_every, updates + 1, health_every)) | checkpoints | {updates}
    )
    return (
        history_steps == expected_history
        and health_steps == expected_health
        and ema_steps == checkpoints
    )


def _fixed_exact_corner_trajectory(
    result: dict,
    *,
    from_step: int = FINAL_CORNER_FROM_STEP,
    to_step: int = FINAL_STEP,
) -> dict[str, object]:
    """Recompute the fixed exact-corner trajectory over the final 50k."""
    declared = result.get("declared_profile", {})
    objective = declared.get("objective", {}) if isinstance(declared, dict) else {}
    train = declared.get("train", {}) if isinstance(declared, dict) else {}
    health = result.get("training", {}).get("health", [])
    try:
        samples = int(train["audit_samples"])
    except (KeyError, TypeError, ValueError):
        samples = -1
    expected_numerics = {
        "stopped_evaluation": objective.get("stopped_evaluation"),
        "emf_delta": objective.get("emf_delta"),
        "emf_denominator_floor": objective.get("emf_denominator_floor"),
    }
    points: list[dict[str, object]] = []
    for wanted in (from_step, to_step):
        matches = [
            record
            for record in health
            if isinstance(record, dict) and record.get("step") == wanted
        ]
        if len(matches) != 1:
            points.append({"step": wanted, "valid": False})
            continue
        probe = matches[0].get("fixed_exact_inference_corner")
        if not isinstance(probe, dict):
            points.append({"step": wanted, "valid": False})
            continue
        kinds: dict[str, dict[str, object]] = {}
        for kind in ("raw", "ema"):
            summary = probe.get(kind)
            coefficient = (
                summary.get("coefficient") if isinstance(summary, dict) else None
            )
            values = (
                (
                    summary.get("mean_raw_mse"),
                    summary.get("mean_target_rms"),
                    summary.get("mean_quotient_rms"),
                    coefficient.get("minimum")
                    if isinstance(coefficient, dict)
                    else None,
                    coefficient.get("mean") if isinstance(coefficient, dict) else None,
                    coefficient.get("maximum")
                    if isinstance(coefficient, dict)
                    else None,
                )
                if isinstance(summary, dict)
                else ()
            )
            valid = (
                isinstance(summary, dict)
                and summary.get("count") == samples
                and summary.get("nonfinite_rows") == 0
                and len(values) == 6
                and all(_finite(value) and float(value) >= 0.0 for value in values)
            )
            kinds[kind] = {
                "count": summary.get("count", -1) if isinstance(summary, dict) else -1,
                "mean_raw_mse": float(summary["mean_raw_mse"]) if valid else None,
                "valid": valid,
            }
        valid_point = (
            probe.get("condition") == {"t": 1.0, "r": 0.0, "h": 1.0}
            and probe.get("sealed_train_only") is True
            and probe.get("sample_count") == samples
            and samples == FIXED_INFERENCE_CORNER_SAMPLES
            and probe.get("objective_numerics") == expected_numerics
            and all(record["valid"] is True for record in kinds.values())
        )
        points.append(
            {
                "step": wanted,
                "condition": probe.get("condition"),
                "sample_count": probe.get("sample_count"),
                "raw": kinds["raw"],
                "ema": kinds["ema"],
                "valid": valid_point,
            }
        )
    growth: dict[str, float] = {}
    if len(points) == 2 and all(point.get("valid") is True for point in points):
        for kind in ("raw", "ema"):
            first = float(points[0][kind]["mean_raw_mse"])
            second = float(points[1][kind]["mean_raw_mse"])
            growth[kind] = second / max(first, 1e-12)
    stable = len(growth) == 2 and all(
        math.isfinite(value) and value <= MAX_LATE_INFERENCE_CORNER_ERROR_GROWTH
        for value in growth.values()
    )
    return {
        "definition": "fixed exact t=1, r=0, h=1 on sealed train-only rows",
        "from_step": from_step,
        "to_step": to_step,
        "fixed_probe_points": points,
        "fixed_samples": FIXED_INFERENCE_CORNER_SAMPLES,
        "maximum_late_error_growth": MAX_LATE_INFERENCE_CORNER_ERROR_GROWTH,
        "late_error_growth": growth,
        "stable": stable,
    }


def _checkpoint_ladder_artifacts_valid(
    result_path: Path, result: dict, mirror: DurableMirror
) -> bool:
    checkpoints = result.get("checkpoints")
    if not isinstance(checkpoints, dict):
        return False
    for record in checkpoints.values():
        if not isinstance(record, dict):
            return False
        for kind in ("raw", "ema"):
            artifact = record.get(kind)
            path = _recorded_checkpoint_path(result_path, artifact)
            if path is None or not isinstance(artifact, dict):
                return False
            verify_file(path, artifact.get("sha256"))
            if mirror.verify(path) != artifact.get("durable_mirror"):
                return False
    return True


def _snapshot_steps(result: dict) -> list[int] | None:
    records = result.get("raw_snapshots")
    if not isinstance(records, list):
        return None
    steps: list[int] = []
    for record in records:
        if not isinstance(record, dict):
            return None
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            return None
        steps.append(step)
    return steps


def _recovery_result_checks(
    recovery: dict,
    result: dict,
    *,
    expected_updates: int,
    expected_authorization: dict | None,
) -> dict[str, bool]:
    training = result.get("training", {})
    health = training.get("health", []) if isinstance(training, dict) else []
    ranks = [
        float(record["effective_rank_ratio"])
        for record in health
        if isinstance(record, dict) and _finite(record.get("effective_rank_ratio"))
    ]
    snapshots = _snapshot_steps(result)
    counter_pairs = {
        "optimizer_updates": "optimizer_updates",
        "examples_seen": "examples_seen",
        "model_forwards": "objective_sample_evaluations",
        "objective_forward_calls": "objective_forward_calls",
        "clipped_updates": "clipped_updates",
        "clipped_updates_final_window": "clipped_updates_final_window",
        "final_window_updates": "final_window_updates",
        "nonfinite_updates": "nonfinite_updates",
    }
    return {
        "horizon_and_completion": (
            recovery.get("planned_updates") == expected_updates
            and recovery.get("completed_updates") == expected_updates
            and recovery.get("optimizer_updates") == expected_updates
        ),
        "profile_name": recovery.get("profile_name")
        == result.get("realized_profile", {}).get("name"),
        "continuation_authorization": recovery.get("continuation_authorization")
        == expected_authorization,
        "training_counters": all(
            recovery.get(recovery_key) == training.get(result_key)
            for recovery_key, result_key in counter_pairs.items()
        ),
        "history": recovery.get("history") == training.get("history"),
        "health": recovery.get("health") == training.get("health"),
        "checkpoint_ledger": recovery.get("checkpoints") == result.get("checkpoints"),
        "snapshot_ledger": snapshots is not None
        and recovery.get("snapshots") == snapshots,
        "best_raw_rank": bool(ranks)
        and float(recovery.get("best_rank_ratio", math.nan)) == max(ranks),
    }


def _prefix_continuity_checks(result_150k: dict, result_300k: dict) -> dict[str, bool]:
    training_150k = result_150k.get("training", {})
    training_300k = result_300k.get("training", {})
    history_150k = training_150k.get("history", [])
    history_300k = training_300k.get("history", [])
    health_150k = training_150k.get("health", [])
    health_300k = training_300k.get("health", [])
    checkpoints_300k = result_300k.get("checkpoints", {})
    try:
        prefix_checkpoints = {
            step: record
            for step, record in checkpoints_300k.items()
            if int(step) <= PROMOTION_STEP
        }
    except (AttributeError, TypeError, ValueError):
        prefix_checkpoints = None
    snapshots_150k = result_150k.get("raw_snapshots")
    snapshots_300k = result_300k.get("raw_snapshots")
    if isinstance(snapshots_300k, list):
        try:
            prefix_snapshots = [
                record
                for record in snapshots_300k
                if isinstance(record, dict)
                and int(record.get("step", -1)) <= PROMOTION_STEP
            ]
        except (TypeError, ValueError):
            prefix_snapshots = None
    else:
        prefix_snapshots = None
    return {
        "run_identity": result_150k.get("run_identity_sha256")
        == result_300k.get("run_identity_sha256"),
        "unit_identity": (
            result_150k.get("arm") == result_300k.get("arm")
            and result_150k.get("numerical_candidate")
            == result_300k.get("numerical_candidate")
            and result_150k.get("unit_seed") == result_300k.get("unit_seed")
            and result_150k.get("preflight_sha256")
            == result_300k.get("preflight_sha256")
        ),
        "history_prefix": isinstance(history_150k, list)
        and isinstance(history_300k, list)
        and history_300k[: len(history_150k)] == history_150k
        and len(history_300k) > len(history_150k),
        "health_prefix": isinstance(health_150k, list)
        and isinstance(health_300k, list)
        and health_300k[: len(health_150k)] == health_150k
        and len(health_300k) > len(health_150k),
        "checkpoint_prefix": prefix_checkpoints == result_150k.get("checkpoints"),
        "snapshot_prefix": prefix_snapshots == snapshots_150k,
        "early_admission": result_150k.get("early_admission")
        == result_300k.get("early_admission"),
    }


def _safe_mirror_path(mirror_root: Path, reference: object) -> Path | None:
    if not isinstance(reference, str) or not reference.strip():
        return None
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = (mirror_root.resolve() / relative).resolve()
    try:
        candidate.relative_to(mirror_root.resolve())
    except ValueError:
        return None
    return candidate


def _durable_recovery_record_valid(
    record: object,
    *,
    mirror_root: Path,
    relative_path: str,
    recovery_step: int,
    expected_sha: str,
) -> bool:
    required = {
        "relative_path",
        "sha256",
        "bytes",
        "recovery_step",
        "version_relative_path",
        "commit_relative_path",
    }
    if not isinstance(record, dict) or set(record) != required:
        return False
    version = _safe_mirror_path(mirror_root, record.get("version_relative_path"))
    commit = _safe_mirror_path(mirror_root, record.get("commit_relative_path"))
    if version is None or commit is None:
        return False
    if (
        record.get("relative_path") != relative_path
        or record.get("sha256") != expected_sha
        or record.get("recovery_step") != recovery_step
        or isinstance(record.get("bytes"), bool)
        or not isinstance(record.get("bytes"), int)
        or record["bytes"] < 0
    ):
        return False
    verify_file(version, expected_sha)
    commit_payload = verify_json(commit, "cap-emf2-durable-recovery-commit")
    commit_version = (
        commit.parent.parent / str(commit_payload.get("version"))
    ).resolve()
    return (
        version.stat().st_size == record["bytes"]
        and commit_payload.get("schema_version") == 1
        and commit_payload.get("relative_path") == relative_path
        and commit_payload.get("recovery_step") == recovery_step
        and commit_payload.get("sha256") == expected_sha
        and commit_payload.get("bytes") == record["bytes"]
        and commit_version == version
    )


def _load_mirrored_recovery(path: Path, expected_sha: str) -> dict:
    verify_file(path, expected_sha)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("stage") != "cap-emf-1-recovery":
        raise RuntimeError(f"not a CAP recovery payload: {path}")
    validate_recovery_counters(payload, strict=True)
    return payload


def _quality_retention(
    *,
    final_metrics: dict,
    promotion: dict,
    preflight: dict,
    margin: dict,
    require_historical_improvement: bool,
) -> dict:
    metrics_150k = promotion.get("comparison", {}).get("candidate", {})
    baseline = preflight.get("inputs", {}).get("baseline_standard", {})
    baseline_metrics = baseline.get("metrics", {})
    final_fid = _metric(final_metrics, FID_KEY)
    final_kid = _metric(final_metrics, KID_KEY)
    promotion_fid = _metric(metrics_150k, "clean_fid")
    promotion_kid = _metric(metrics_150k, "clean_kid")
    baseline_fid = _metric(baseline_metrics, FID_KEY)
    baseline_kid = _metric(baseline_metrics, KID_KEY)
    fid_margin = _metric(margin, "clean_fid")
    kid_margin = _metric(margin, "clean_kid")
    checks = {
        "fid_retains_150k_quality_within_margin": final_fid
        <= promotion_fid + fid_margin,
        "kid_retains_150k_quality_within_margin": final_kid
        <= promotion_kid + kid_margin,
    }
    if require_historical_improvement:
        checks.update(
            {
                "fid_beats_historical_baseline_beyond_margin": final_fid
                < baseline_fid - fid_margin,
                "kid_beats_historical_baseline_beyond_margin": final_kid
                < baseline_kid - kid_margin,
            }
        )
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "historical_baseline": {"clean_fid": baseline_fid, "clean_kid": baseline_kid},
        "promotion_150k": {"clean_fid": promotion_fid, "clean_kid": promotion_kid},
        "final_300k": {"clean_fid": final_fid, "clean_kid": final_kid},
        "direct_real_real_margin": copy.deepcopy(margin),
    }


def _absolute_auxiliary_checks(evaluation: dict) -> dict[str, bool]:
    feature = evaluation.get("repository_feature_metrics", {})
    precision = _metric(feature, "precision")
    recall = _metric(feature, "recall")
    pr_f1 = _metric(feature, "pr_f1")
    expected_f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision > 0.0 and recall > 0.0
        else 0.0
    )
    memorization = evaluation.get("memorization", {})
    exact_copy = _metric(memorization, "exact_pixel_copy_fraction")
    duplicates = _metric(memorization, "exact_generated_duplicate_fraction")
    return {
        "precision_noncollapse": MIN_PRECISION <= precision <= 1.0,
        "recall_noncollapse": MIN_RECALL <= recall <= 1.0,
        "pr_f1_noncollapse": MIN_PR_F1 <= pr_f1 <= 1.0,
        "pr_f1_consistent": (
            math.isfinite(expected_f1)
            and math.isfinite(pr_f1)
            and abs(pr_f1 - expected_f1) <= 1e-6
        ),
        "no_exact_training_copy": (
            math.isfinite(exact_copy) and 0.0 <= exact_copy <= MAX_EXACT_COPY_FRACTION
        ),
        "generated_duplicates_bounded": (
            math.isfinite(duplicates)
            and 0.0 <= duplicates <= MAX_GENERATED_DUPLICATE_FRACTION
        ),
    }


def _verify_arm(
    *,
    arm: str,
    paths: FinalArmArtifacts,
    selection: dict,
    promotion: dict,
    promotion_path: Path,
    preflight: dict,
    calibration_margin: dict,
) -> dict:
    """Verify one selected arm and return policy-ready immutable facts."""
    result = verify_json(paths.result, "cap-emf2-screen-unit")
    admission = verify_json(paths.raw_readmission, "cap-emf2-numerical-admission")
    evaluation = verify_json(paths.evaluation, "cap-emf2-development-evaluation")
    candidate = selection["candidate"]
    declared, realized = _expected_profiles(preflight, arm, candidate)
    declared_150k, realized_150k = _expected_profiles(
        preflight, arm, candidate, updates=PROMOTION_STEP
    )

    promotion_references = promotion.get("references", {})
    result_150k_path = _resolve(
        promotion_references.get("result_150k", ""), promotion_path.parent
    )
    raw_150k_path = _resolve(
        promotion_references.get("checkpoint_150k_raw", ""), promotion_path.parent
    )
    ema_150k_path = _resolve(
        promotion_references.get("checkpoint_150k_ema", ""), promotion_path.parent
    )
    result_150k = verify_json(result_150k_path, "cap-emf2-screen-unit")
    if result_150k["artifact_sha256"] != promotion.get("result_sha256"):
        raise RuntimeError("promotion no longer binds its immutable 150k result")

    final_record = result.get("checkpoints", {}).get(str(FINAL_STEP), {})
    raw_record = final_record.get("raw", {}) if isinstance(final_record, dict) else {}
    ema_record = final_record.get("ema", {}) if isinstance(final_record, dict) else {}
    run_identity = result.get("run_identity_sha256")
    unit_seed = result.get("unit_seed")

    # Loading both checkpoint payloads verifies sidecars, exact kind/step,
    # profile, preflight, run identity, arm, and unit seed.
    raw_checkpoint = load_checkpoint(
        paths.raw_checkpoint,
        expected_sha=raw_record.get("sha256"),
        step=FINAL_STEP,
        kind="raw",
        arm=arm,
        declared_profile=declared,
        realized_profile=realized,
        preflight_sha256=preflight["artifact_sha256"],
        run_identity_sha256=run_identity,
        unit_seed=unit_seed,
    )
    ema_checkpoint = load_checkpoint(
        paths.ema_checkpoint,
        expected_sha=ema_record.get("sha256"),
        step=FINAL_STEP,
        kind="ema",
        arm=arm,
        declared_profile=declared,
        realized_profile=realized,
        preflight_sha256=preflight["artifact_sha256"],
        run_identity_sha256=run_identity,
        unit_seed=unit_seed,
    )
    raw_sha = raw_checkpoint["artifact_sha256"]
    ema_sha = ema_checkpoint["artifact_sha256"]
    raw_checkpoint_150k = load_checkpoint(
        raw_150k_path,
        expected_sha=promotion.get("raw_checkpoint_sha256"),
        step=PROMOTION_STEP,
        kind="raw",
        arm=arm,
        declared_profile=declared_150k,
        realized_profile=realized_150k,
        preflight_sha256=preflight["artifact_sha256"],
        run_identity_sha256=run_identity,
        unit_seed=unit_seed,
    )
    ema_checkpoint_150k = load_checkpoint(
        ema_150k_path,
        expected_sha=promotion.get("checkpoint_sha256"),
        step=PROMOTION_STEP,
        kind="ema",
        arm=arm,
        declared_profile=declared_150k,
        realized_profile=realized_150k,
        preflight_sha256=preflight["artifact_sha256"],
        run_identity_sha256=run_identity,
        unit_seed=unit_seed,
    )

    baseline = preflight["inputs"]["baseline_standard"]
    baseline_metrics = baseline.get("metrics", {})
    metrics = evaluation.get("standard_train_reference_metrics", {})
    feature = evaluation.get("repository_feature_metrics", {})
    benchmark = preflight["inputs"]["benchmark"]
    initial_admission = preflight["inputs"]["numerical_admission"]
    recovery = result.get("recovery", {})
    expected_authorization = _expected_recovery_authorization(promotion, selection)
    recovery_path = _resolve(recovery.get("path", ""), paths.result.parent)
    try:
        recovery_relative = recovery_path.relative_to(
            paths.result.parent.resolve()
        ).as_posix()
    except ValueError as error:
        raise RuntimeError("final recovery lies outside its run root") from error
    final_recovery, final_recovery_sha = load_recovery_payload(
        recovery_path, require_sidecar=True, validate_counters=True
    )

    mirror = DurableMirror(paths.result.parent, paths.mirror_root)
    recovery_steps = mirror.recovery_steps(recovery_path)
    final_recovery_mirror = mirror.verify_recovery(
        recovery_path, recovery_step=FINAL_STEP
    )
    identity_path = paths.result.parent / "run_identity.json"
    identity_record = verify_json(identity_path, "cap-emf2-run-identity")
    identity_sha = identity_record.pop("artifact_sha256")
    identity_mirror = mirror.verify(identity_path)
    mirror.verify(paths.result)
    mirror.verify(result_150k_path)
    ladder_artifacts_valid = _checkpoint_ladder_artifacts_valid(
        paths.result, result, mirror
    )

    recovery_150k_record = result_150k.get("recovery", {})
    recovery_150k_path = _resolve(
        recovery_150k_record.get("path", ""), result_150k_path.parent
    )
    recovery_150k_durable = recovery_150k_record.get("durable_mirror")
    durable_150k_valid = _durable_recovery_record_valid(
        recovery_150k_durable,
        mirror_root=paths.mirror_root,
        relative_path=recovery_relative,
        recovery_step=PROMOTION_STEP,
        expected_sha=recovery_150k_record.get("sha256", ""),
    )
    recovery_150k_version = _safe_mirror_path(
        paths.mirror_root,
        recovery_150k_durable.get("version_relative_path", "")
        if isinstance(recovery_150k_durable, dict)
        else "",
    )
    if recovery_150k_version is None:
        raise RuntimeError("150k durable recovery version path is invalid")
    recovery_payload_150k = _load_mirrored_recovery(
        recovery_150k_version, recovery_150k_record.get("sha256", "")
    )

    expected_identity = {
        "status": "cap-emf2-run-identity",
        "arm": arm,
        "candidate": candidate,
        "preflight_sha256": preflight["artifact_sha256"],
        "profile_150k_realized": realized_150k,
        "unit_seed": unit_seed,
        "deterministic_algorithms": result.get("deterministic_algorithms"),
        "precision": result.get("precision"),
        "device": result.get("device"),
        "hardware": result.get("hardware"),
        "benchmark_sha256": benchmark.get("artifact_sha256"),
    }
    expected_recovery_identity_payload = {
        "profile_horizon_neutral": _horizon_neutral_profile(realized),
        "unit_seed": unit_seed,
        "external": expected_identity,
    }
    recovery_checks_150k = _recovery_result_checks(
        recovery_payload_150k,
        result_150k,
        expected_updates=PROMOTION_STEP,
        expected_authorization=None,
    )
    final_recovery_checks = _recovery_result_checks(
        final_recovery,
        result,
        expected_updates=FINAL_STEP,
        expected_authorization=expected_authorization,
    )
    continuity_checks = _prefix_continuity_checks(result_150k, result)
    recomputed_gate = _recomputed_capability_gate(result, declared)
    training_cadence = _training_cadence_valid(result, declared)
    corner_trajectory = _fixed_exact_corner_trajectory(result)
    evaluation_evidence = revalidate_clean_evaluation_evidence(
        evaluation, anchor=paths.evaluation.parent
    )
    recomputed_metrics = evaluation_evidence.get("recomputed", {})
    if not isinstance(recomputed_metrics, dict):
        recomputed_metrics = {}
    leaf_checks = evaluation_evidence.get("checks", {})
    if not isinstance(leaf_checks, dict):
        leaf_checks = {}
    quality_retention = _quality_retention(
        final_metrics=recomputed_metrics,
        promotion=promotion,
        preflight=preflight,
        margin=calibration_margin,
        require_historical_improvement=arm in ORDERED_ARMS,
    )
    evaluation_unit_path = _resolve(
        evaluation.get("unit", {}).get("path", ""), paths.evaluation.parent
    )
    evaluation_checkpoint_path = _resolve(
        evaluation.get("checkpoint", {}).get("path", ""), paths.evaluation.parent
    )

    checkpoint_150k = result.get("checkpoints", {}).get(str(PROMOTION_STEP), {})
    raw_record_150k = (
        checkpoint_150k.get("raw", {}) if isinstance(checkpoint_150k, dict) else {}
    )
    ema_record_150k = (
        checkpoint_150k.get("ema", {}) if isinstance(checkpoint_150k, dict) else {}
    )
    durability = result.get("durability", {})
    probe = durability.get("live_roundtrip_probe", {})
    attestation = mirror.attestation

    checks = {
        "result_is_exact_300k_development_unit": (
            result.get("development_only") is True
            and result.get("arm") == arm
            and result.get("numerical_candidate") == candidate
            and int(result.get("training", {}).get("optimizer_updates", -1))
            == FINAL_STEP
        ),
        "result_preflight_and_identity": (
            result.get("preflight_sha256") == preflight["artifact_sha256"]
            and _hash64(run_identity)
            and isinstance(unit_seed, int)
            and not isinstance(unit_seed, bool)
            and unit_seed >= 0
            and unit_seed == int(benchmark.get("unit_seed", -1))
        ),
        "result_exact_declared_profile": result.get("declared_profile") == declared,
        "result_exact_realized_profile": result.get("realized_profile") == realized,
        "result_realized_batch_split": result.get("realized_batch_split")
        == {
            "micro_batch": int(benchmark["micro_batch"]),
            "accumulation_steps": int(benchmark["accumulation_steps"]),
            "effective_batch": int(benchmark["effective_batch"]),
            "overridden": declared != realized,
        },
        "checkpoint_ladder_complete": _checkpoint_ladder_valid(result, declared),
        "checkpoint_ladder_files_and_durable_copies": ladder_artifacts_valid,
        "checkpoint_previews_complete": _checkpoint_previews_valid(
            result, anchor=paths.result.parent
        ),
        "checkpoint_150k_exact_promotion_boundary": (
            _recorded_checkpoint_path(paths.result, raw_record_150k)
            == raw_150k_path.resolve()
            and _recorded_checkpoint_path(paths.result, ema_record_150k)
            == ema_150k_path.resolve()
            and raw_record_150k.get("sha256") == promotion.get("raw_checkpoint_sha256")
            and ema_record_150k.get("sha256") == promotion.get("checkpoint_sha256")
        ),
        "final_raw_record_exact": (
            _recorded_checkpoint_path(paths.result, raw_record)
            == paths.raw_checkpoint.resolve()
            and raw_record.get("sha256") == raw_sha
        ),
        "final_ema_record_exact": (
            _recorded_checkpoint_path(paths.result, ema_record)
            == paths.ema_checkpoint.resolve()
            and ema_record.get("sha256") == ema_sha
        ),
        "training_gate_recomputed_exact_and_passed": (
            recomputed_gate is not None
            and result.get("train_only_gate") == recomputed_gate
            and recomputed_gate.get("verdict") == "PASS"
            and _all_true(recomputed_gate.get("checks"))
        ),
        "training_history_and_health_cadence_complete": training_cadence,
        "training_clip_fraction_recomputed_exact": (
            int(result.get("training", {}).get("final_window_updates", -1)) > 0
            and result.get("training", {}).get("clip_fraction_final_window")
            == int(result.get("training", {}).get("clipped_updates_final_window", -1))
            / int(result.get("training", {}).get("final_window_updates", -1))
        ),
        "final_exact_corner_trajectory_stable": corner_trajectory["stable"] is True,
        "training_has_no_nonfinite_updates": int(
            result.get("training", {}).get("nonfinite_updates", -1)
        )
        == 0,
        "training_exact_example_accounting": (
            result.get("training", {}).get("examples_seen_target")
            == FINAL_STEP * int(realized.get("train", {}).get("effective_batch", -1))
            == result.get("training", {}).get("examples_seen")
        ),
        "result_recovery_is_authorized_300k": (
            int(recovery.get("planned_updates", -1)) == FINAL_STEP
            and int(recovery.get("completed_updates", -1)) == FINAL_STEP
            and recovery.get("sha256") == final_recovery_sha
            and recovery.get("continuation_authorization") == expected_authorization
        ),
        "final_recovery_result_counters_and_ledgers": all(
            final_recovery_checks.values()
        ),
        "final_recovery_raw_state_exact": _state_dict_exact(
            final_recovery.get("model"), raw_checkpoint.get("state_dict")
        ),
        "final_recovery_ema_state_exact": _state_dict_exact(
            _ema_recovery_state(final_recovery), ema_checkpoint.get("state_dict")
        ),
        "recovery_150k_result_counters_and_ledgers": all(recovery_checks_150k.values()),
        "recovery_150k_raw_state_exact": _state_dict_exact(
            recovery_payload_150k.get("model"),
            raw_checkpoint_150k.get("state_dict"),
        ),
        "recovery_150k_ema_state_exact": _state_dict_exact(
            _ema_recovery_state(recovery_payload_150k),
            ema_checkpoint_150k.get("state_dict"),
        ),
        "recovery_identity_continuous_and_exact": (
            recovery_payload_150k.get("recovery_identity")
            == final_recovery.get("recovery_identity")
            and final_recovery.get("recovery_identity", {}).get("payload")
            == expected_recovery_identity_payload
        ),
        "result_150k_to_300k_prefix_continuity": all(continuity_checks.values()),
        "durable_recovery_commits_150k_and_latest_300k": (
            recovery_150k_path.resolve() == recovery_path.resolve()
            and durable_150k_valid
            and PROMOTION_STEP in recovery_steps
            and bool(recovery_steps)
            and recovery_steps[-1] == FINAL_STEP
            and recovery.get("durable_mirror") == final_recovery_mirror
        ),
        "durable_root_identity_and_probe": (
            durability.get("required") is True
            and durability.get("synchronous") is True
            and isinstance(durability.get("mirror_root"), str)
            and bool(durability.get("mirror_root", "").strip())
            and durability.get("run_identity") == identity_mirror
            and probe.get("status") == "cap-emf2-durable-root-probe"
            and probe.get("storage_id") == attestation.get("storage_id")
            and probe.get("attestation_sha256") == attestation.get("artifact_sha256")
            and isinstance(probe.get("bytes"), int)
            and not isinstance(probe.get("bytes"), bool)
            and probe.get("bytes", 0) > 0
            and _hash64(probe.get("sha256"))
            and probe.get("roundtrip_verified") is True
            and probe.get("probe_removed") is True
        ),
        "run_identity_exact_and_durable": (
            identity_sha == run_identity and identity_record == expected_identity
        ),
        "training_numerical_execution_exact_benchmark": (
            result.get("deterministic_algorithms")
            == benchmark.get("deterministic_algorithms")
            and result.get("precision") == benchmark.get("precision")
            and result.get("device") == benchmark.get("device")
        ),
        "raw_readmission_exact_checkpoint": (
            admission.get("checkpoint_sha256") == raw_sha
            and int(admission.get("checkpoint_step", -1)) == FINAL_STEP
            and admission.get("checkpoint_identity", {}).get("valid") is True
            and admission.get("checkpoint_identity", {}).get("stage")
            == "cap-emf-2-screen"
            and admission.get("checkpoint_identity", {}).get("kind") == "raw"
            and admission.get("checkpoint_identity", {}).get("arm") == arm
        ),
        "raw_readmission_full_matrix_go": (
            admission.get("decision") == "GO"
            and admission_matrix_complete(admission)
            and _all_true(admission.get("protocol_checks"))
        ),
        "raw_readmission_candidate_and_sources": (
            admission.get("candidate", {}).get("name") == candidate
            and admission.get("source_sha256") == preflight.get("source_sha256")
        ),
        "raw_readmission_exact_production_numerical_mode": (
            admission.get("production_numerical_mode")
            == initial_admission.get("production_numerical_mode")
        ),
        "raw_readmission_same_training_environment": (
            _same_hardware(admission.get("hardware"), result.get("hardware"))
            and _same_hardware(admission.get("hardware"), benchmark.get("hardware"))
            and _same_hardware(
                admission.get("hardware"), initial_admission.get("hardware")
            )
        ),
        "evaluation_exact_unit_and_ema": (
            evaluation.get("development_only") is True
            and evaluation.get("arm") == arm
            and int(evaluation.get("step", -1)) == FINAL_STEP
            and evaluation.get("unit", {}).get("sha256") == result["artifact_sha256"]
            and evaluation.get("unit", {}).get("preflight_sha256")
            == preflight["artifact_sha256"]
            and evaluation_unit_path == paths.result.resolve()
            and evaluation.get("checkpoint", {}).get("sha256") == ema_sha
            and evaluation.get("checkpoint", {}).get("kind") == "ema"
            and int(evaluation.get("checkpoint", {}).get("step", -1)) == FINAL_STEP
            and evaluation_checkpoint_path == paths.ema_checkpoint.resolve()
        ),
        "evaluation_fixed_50k_one_call_protocol": (
            _development_evaluation_schema(evaluation)
            and int(evaluation.get("samples", {}).get("count", -1))
            == DEVELOPMENT_SAMPLES
            and evaluation.get("samples", {}).get("one_model_call_per_batch") is True
            and int(evaluation.get("samples", {}).get("seed", -1))
            == DEVELOPMENT_GENERATION_SEED
            and int(evaluation.get("fixed_protocol", {}).get("generated_samples", -1))
            == DEVELOPMENT_SAMPLES
            and int(evaluation.get("fixed_protocol", {}).get("generation_seed", -1))
            == DEVELOPMENT_GENERATION_SEED
            and int(evaluation.get("fixed_protocol", {}).get("clean_kid_seed", -1))
            == DEVELOPMENT_KID_SEED
            and evaluation.get("fixed_protocol", {}).get("numerical_settings")
            == evaluation.get("provenance", {}).get("numerical_settings")
        ),
        "evaluation_standard_metrics_valid": _clean_metrics_valid(metrics),
        "evaluation_leaf_evidence_recomputed": evaluation_evidence.get("valid") is True,
        "evaluation_exact_kid_reference": _same_kid_reference(
            metrics.get("kid_reference"), baseline_metrics.get("kid_reference")
        ),
        "evaluation_exact_auxiliary_reference": (
            _reference_binding(evaluation) == _reference_binding(baseline)
        ),
        "evaluation_exact_local_environment": (
            _evaluation_environment(evaluation) == _evaluation_environment(baseline)
        ),
        "evaluation_uncurated_grid": _grid_integrity(
            evaluation, anchor=paths.evaluation.parent
        ),
        "evaluation_sources": (
            evaluation.get("source_sha256") == preflight.get("source_sha256")
        ),
        **{f"evaluation_leaf_{name}": passed for name, passed in leaf_checks.items()},
        **_absolute_auxiliary_checks(evaluation),
    }
    checks.update(
        {
            f"quality_{name}": passed
            for name, passed in quality_retention["checks"].items()
        }
    )
    return {
        "arm": arm,
        "valid": all(checks.values()),
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
        "metrics": {
            "clean_fid": _metric(recomputed_metrics, FID_KEY),
            "clean_kid": _metric(recomputed_metrics, KID_KEY),
            "repository_kid_reported_only": _metric(feature, "unbiased_kid"),
            "precision_reported_with_absolute_floor": _metric(feature, "precision"),
            "recall_reported_with_absolute_floor": _metric(feature, "recall"),
            "pr_f1_reported_with_absolute_floor": _metric(feature, "pr_f1"),
        },
        "quality_retention": quality_retention,
        "clean_evaluation_evidence": {
            "valid": evaluation_evidence.get("valid"),
            "checks": leaf_checks,
            "recomputed": recomputed_metrics,
            "limit": evaluation_evidence.get("limit"),
        },
        "final_exact_corner_trajectory": corner_trajectory,
        "recovery_continuity": {
            "promotion_boundary": recovery_checks_150k,
            "final": final_recovery_checks,
            "prefix": continuity_checks,
            "committed_steps": recovery_steps,
        },
        "artifact_sha256": {
            "result": result["artifact_sha256"],
            "result_150k": result_150k["artifact_sha256"],
            "raw_checkpoint": raw_sha,
            "ema_checkpoint": ema_sha,
            "raw_readmission": admission["artifact_sha256"],
            "evaluation": evaluation["artifact_sha256"],
            "recovery_150k": recovery_150k_record.get("sha256"),
            "recovery_300k": final_recovery_sha,
            "run_identity": identity_sha,
        },
        "run_identity_sha256": run_identity,
        "unit_seed": unit_seed,
    }


def _selected_set(selection: dict) -> tuple[list[str], str]:
    selected = selection.get("selected_arms")
    if not isinstance(selected, list) or len(selected) != len(set(selected)):
        raise RuntimeError("final verdict selection has malformed selected arms")
    if any(arm not in SAMPLER_ARMS for arm in selected):
        raise RuntimeError("final verdict selection names an unsupported arm")
    ordered = [arm for arm in selected if arm in ORDERED_ARMS]
    if (
        len(selected) != 2
        or len(ordered) != 1
        or selected != [LEGACY_ARM, ordered[0]]
        or selection.get("ordered_winner") != ordered[0]
        or selection.get("decision") != "GO"
    ):
        raise RuntimeError(
            "final verdict requires a concurrent legacy/ordered paired selection"
        )
    return selected, ordered[0]


def _final_policy(
    *, selection: dict, arms: dict[str, dict], calibration_margin: dict
) -> dict:
    selected, ordered = _selected_set(selection)
    if set(arms) != set(selected):
        raise RuntimeError("final verdict arm artifacts do not match the selection")
    fid_margin = _metric(calibration_margin, "clean_fid")
    kid_margin = _metric(calibration_margin, "clean_kid")
    if (
        calibration_margin.get("kind") != CALIBRATION_MARGIN_KIND
        or not math.isfinite(fid_margin)
        or not math.isfinite(kid_margin)
        or fid_margin < 0.0
        or kid_margin < 0.0
    ):
        raise RuntimeError("final verdict lacks valid direct real/real margins")

    ordered_metrics = arms[ordered]["metrics"]
    ordered_fid = _metric(ordered_metrics, "clean_fid")
    ordered_kid = _metric(ordered_metrics, "clean_kid")
    all_arms_valid = all(record.get("valid") is True for record in arms.values())
    legacy_metrics = arms[LEGACY_ARM]["metrics"]
    legacy_fid = _metric(legacy_metrics, "clean_fid")
    legacy_kid = _metric(legacy_metrics, "clean_kid")
    fid_win = ordered_fid < legacy_fid - fid_margin
    kid_win = ordered_kid < legacy_kid - kid_margin
    paired_win = fid_win and kid_win
    quality_retention = arms[ordered].get("quality_retention", {})
    all_arms_retain_150k_quality = all(
        record.get("quality_retention", {}).get("valid") is True
        for record in arms.values()
    )

    checks = {
        "selection_revalidated_go": (
            selection.get("revalidated") is True and selection.get("decision") == "GO"
        ),
        "all_selected_300k_arms_valid": all_arms_valid,
        "all_selected_arms_retain_150k_quality": all_arms_retain_150k_quality,
        "ordered_absolute_quality_retained": quality_retention.get("valid") is True,
        "paired_standard_metric_win": paired_win,
    }
    decision = "GO" if all(checks.values()) else "NO_GO"
    return {
        "decision": decision,
        "claim": {
            "scope": "paired-300k-developmental",
            "ordered_arm": ordered,
            "ordered_beats_concurrent_legacy": paired_win,
            "selection_metrics": ["50k CleanFID", "50k CleanKID"],
            "auxiliary_metrics_used_for_selection": False,
            "absolute_quality_retention_required": True,
        },
        "comparison": {
            "ordered": {
                "arm": ordered,
                "clean_fid": ordered_fid,
                "clean_kid": ordered_kid,
            },
            "legacy": {
                "arm": LEGACY_ARM,
                "clean_fid": legacy_fid,
                "clean_kid": legacy_kid,
            },
            "direct_real_real_margin": calibration_margin,
            "ordered_fid_wins_beyond_margin": fid_win,
            "ordered_kid_wins_beyond_margin": kid_win,
            "ordered_absolute_quality_retention": quality_retention,
            "quality_retention_by_arm": {
                arm: arms[arm].get("quality_retention") for arm in selected
            },
        },
        "final_arm": ordered if decision == "GO" else None,
        "checks": checks,
        "failed": sorted(name for name, passed in checks.items() if not passed),
    }


def _load_context(
    selection_path: Path, arm_paths: dict[str, FinalArmArtifacts]
) -> dict:
    selection = revalidate_selection(selection_path)
    selected, _ordered = _selected_set(selection)
    if set(arm_paths) != set(selected):
        raise RuntimeError("final verdict paths do not match selected 300k arms")

    promotions: dict[str, dict] = {}
    promotion_paths: dict[str, Path] = {}
    preflights: dict[str, dict] = {}
    for arm in selected:
        promotion_path = _resolve(
            selection.get("references", {}).get(arm), selection_path.parent
        )
        promotion = revalidate_promotion(promotion_path, require_go=arm != LEGACY_ARM)
        if (
            arm == LEGACY_ARM
            and promotion.get("control_continuation", {}).get("decision") != "GO"
        ):
            raise RuntimeError("selected legacy arm is not a valid concurrent control")
        if (
            selection.get("promotion_sha256", {}).get(arm)
            != promotion["artifact_sha256"]
        ):
            raise RuntimeError("selection does not bind a selected arm promotion")
        preflight_path = _resolve(
            promotion.get("references", {}).get("preflight"), promotion_path.parent
        )
        preflight = load_preflight(preflight_path)
        if preflight["artifact_sha256"] != promotion.get("preflight_sha256"):
            raise RuntimeError("selected promotion does not bind its preflight")
        promotions[arm] = promotion
        promotion_paths[arm] = promotion_path
        preflights[arm] = preflight

    preflight_hashes = {record["artifact_sha256"] for record in preflights.values()}
    if len(preflight_hashes) != 1:
        raise RuntimeError("selected 300k arms do not share one preflight")
    preflight = preflights[selected[0]]
    if (
        selection.get("candidate") != preflight.get("candidate")
        or selection.get("preflight_sha256") != preflight["artifact_sha256"]
    ):
        raise RuntimeError("selection candidate/preflight binding changed")

    baseline_evidence = revalidate_clean_evaluation_evidence(
        preflight["inputs"]["baseline_standard"], anchor=Path.cwd()
    )
    if baseline_evidence.get("valid") is not True:
        failed = sorted(
            name
            for name, passed in baseline_evidence.get("checks", {}).items()
            if not passed
        )
        raise RuntimeError(f"historical baseline leaf evidence failed: {failed}")
    # Downstream retention gates consume the independently recomputed values,
    # never the JSON's recorded scalar copies.
    preflight = copy.deepcopy(preflight)
    preflight["inputs"]["baseline_standard"]["metrics"].update(
        baseline_evidence["recomputed"]
    )

    calibration = preflight["inputs"]["metric_calibration"]
    calibration_evidence = revalidate_metric_calibration_evidence(
        calibration, anchor=Path.cwd()
    )
    if calibration_evidence.get("valid") is not True:
        failed = sorted(
            name
            for name, passed in calibration_evidence.get("checks", {}).items()
            if not passed
        )
        raise RuntimeError(f"real/real calibration leaf evidence failed: {failed}")
    calibration = copy.deepcopy(calibration)
    calibration["metrics"].update(calibration_evidence["recomputed"])
    margin = {
        "kind": CALIBRATION_MARGIN_KIND,
        "clean_fid": _calibration_tolerance(calibration, FID_KEY),
        "clean_kid": _calibration_tolerance(calibration, KID_KEY),
        "statistical_scope": (
            "deterministic finite-sample margin; not a confidence interval"
        ),
    }
    arms = {
        arm: _verify_arm(
            arm=arm,
            paths=arm_paths[arm],
            selection=selection,
            promotion=promotions[arm],
            promotion_path=promotion_paths[arm],
            preflight=preflight,
            calibration_margin=margin,
        )
        for arm in selected
    }
    unit_seeds = {record["unit_seed"] for record in arms.values()}
    if len(unit_seeds) != 1:
        raise RuntimeError("selected 300k arms do not share the matched unit seed")
    return {
        "selection": selection,
        "preflight": preflight,
        "promotions": promotions,
        "arms": arms,
        "calibration_margin": margin,
        "baseline_evidence": baseline_evidence,
        "calibration_evidence": calibration_evidence,
        "unit_seed": next(iter(unit_seeds)),
    }


def _artifact_references(
    *, selection_path: Path, arm_paths: dict[str, FinalArmArtifacts], anchor: Path
) -> dict:
    return {
        "selection": _reference(selection_path, anchor),
        "arms": {
            arm: {
                "result": _reference(paths.result, anchor),
                "raw_checkpoint": _reference(paths.raw_checkpoint, anchor),
                "ema_checkpoint": _reference(paths.ema_checkpoint, anchor),
                "raw_readmission": _reference(paths.raw_readmission, anchor),
                "evaluation": _reference(paths.evaluation, anchor),
                "mirror_root": _reference(paths.mirror_root, anchor),
            }
            for arm, paths in sorted(arm_paths.items())
        },
    }


def _render_payload(context: dict, references: dict) -> dict:
    selection = context["selection"]
    policy = _final_policy(
        selection=selection,
        arms=context["arms"],
        calibration_margin=context["calibration_margin"],
    )
    return {
        "status": FINAL_VERDICT_STATUS,
        "policy_version": POLICY_VERSION,
        **policy,
        "candidate": selection["candidate"],
        "selected_arms": selection["selected_arms"],
        "unit_seed": context["unit_seed"],
        "selection_sha256": selection["artifact_sha256"],
        "preflight_sha256": context["preflight"]["artifact_sha256"],
        "historical_baseline_evidence": context.get("baseline_evidence"),
        "real_real_calibration_evidence": context.get("calibration_evidence"),
        "promotion_sha256": {
            arm: record["artifact_sha256"]
            for arm, record in sorted(context["promotions"].items())
        },
        "arm_validation": {
            arm: {
                "valid": record["valid"],
                "checks": record["checks"],
                "failed": record["failed"],
                "metrics": record["metrics"],
                "clean_evaluation_evidence": record["clean_evaluation_evidence"],
                "quality_retention": record["quality_retention"],
                "final_exact_corner_trajectory": record[
                    "final_exact_corner_trajectory"
                ],
                "recovery_continuity": record["recovery_continuity"],
                "run_identity_sha256": record["run_identity_sha256"],
                "artifact_sha256": record["artifact_sha256"],
            }
            for arm, record in sorted(context["arms"].items())
        },
        "references": references,
        "implementation_sha256": file_sha256(Path(__file__)),
        "limits": [
            "CIFAR-10 train-reference development verdict; the sealed test split remains closed.",
            "One matched unit seed; this is not a confirmation or general performance claim.",
            "Repository 2,048-sample KID and precision/recall never select an arm; only absolute collapse floors apply.",
            "Durability relies on the operator's instance-independent storage attestation plus verified committed copies.",
            "Generated clean-Inception archives and PNG manifests are hash-bound independently; this verifier does not re-extract features from PNG pixels.",
        ],
    }


def build_final_verdict(
    *,
    selection_path: Path,
    arm_paths: dict[str, FinalArmArtifacts],
    out: Path,
) -> dict:
    assert_unused(out)
    context = _load_context(selection_path, arm_paths)
    references = _artifact_references(
        selection_path=selection_path, arm_paths=arm_paths, anchor=out.parent
    )
    payload = _render_payload(context, references)
    payload["artifact_sha256"] = write_json_atomic(out, payload)
    return payload


def _paths_from_references(references: object, anchor: Path) -> tuple[Path, dict]:
    if not isinstance(references, dict) or set(references) != {"selection", "arms"}:
        raise RuntimeError("final verdict has an incomplete reference ledger")
    arm_refs = references.get("arms")
    if not isinstance(arm_refs, dict) or not arm_refs:
        raise RuntimeError("final verdict has no selected arm references")
    required = {
        "result",
        "raw_checkpoint",
        "ema_checkpoint",
        "raw_readmission",
        "evaluation",
        "mirror_root",
    }
    arm_paths: dict[str, FinalArmArtifacts] = {}
    for arm, record in arm_refs.items():
        if (
            arm not in SAMPLER_ARMS
            or not isinstance(record, dict)
            or set(record) != required
        ):
            raise RuntimeError(
                "final verdict contains a malformed arm reference ledger"
            )
        arm_paths[arm] = FinalArmArtifacts(
            result=_resolve(record["result"], anchor),
            raw_checkpoint=_resolve(record["raw_checkpoint"], anchor),
            ema_checkpoint=_resolve(record["ema_checkpoint"], anchor),
            raw_readmission=_resolve(record["raw_readmission"], anchor),
            evaluation=_resolve(record["evaluation"], anchor),
            mirror_root=_resolve(record["mirror_root"], anchor),
        )
    return _resolve(references["selection"], anchor), arm_paths


def revalidate_final_verdict(path: Path, *, require_go: bool = False) -> dict:
    verdict = verify_json(path, FINAL_VERDICT_STATUS)
    selection_path, arm_paths = _paths_from_references(
        verdict.get("references"), path.parent
    )
    context = _load_context(selection_path, arm_paths)
    expected = _render_payload(
        context,
        _artifact_references(
            selection_path=selection_path, arm_paths=arm_paths, anchor=path.parent
        ),
    )
    actual = {key: value for key, value in verdict.items() if key != "artifact_sha256"}
    if actual != expected:
        changed = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise RuntimeError(f"CAP2 final verdict revalidation failed: {changed}")
    if require_go and verdict.get("decision") != "GO":
        raise RuntimeError("CAP2 final verdict did not return GO")
    verdict["revalidated"] = True
    return verdict


def _parse_arm_artifacts(rows: list[list[str]]) -> dict[str, FinalArmArtifacts]:
    parsed: dict[str, FinalArmArtifacts] = {}
    for arm, result, raw, ema, admission, evaluation, mirror_root in rows:
        if arm in parsed:
            raise ValueError(f"duplicate final arm bundle {arm!r}")
        if arm not in SAMPLER_ARMS:
            raise ValueError(f"unknown final arm {arm!r}")
        parsed[arm] = FinalArmArtifacts(
            result=Path(result),
            raw_checkpoint=Path(raw),
            ema_checkpoint=Path(ema),
            raw_readmission=Path(admission),
            evaluation=Path(evaluation),
            mirror_root=Path(mirror_root),
        )
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revalidate",
        type=Path,
        help="independently reload and recompute an existing final verdict",
    )
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--arm-artifacts",
        action="append",
        nargs=7,
        metavar=(
            "ARM",
            "RESULT",
            "RAW",
            "EMA",
            "READMISSION",
            "EVALUATION",
            "MIRROR_ROOT",
        ),
        help="repeat once for every arm selected for the 300k continuation",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    build_arguments = (args.selection, args.arm_artifacts, args.out)
    if args.revalidate is not None:
        if any(value is not None for value in build_arguments):
            parser.error(
                "--revalidate is mutually exclusive with --selection, "
                "--arm-artifacts, and --out"
            )
        result = revalidate_final_verdict(args.revalidate)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "final_arm": result["final_arm"],
                    "claim": result["claim"],
                    "failed": result["failed"],
                    "revalidated": result["revalidated"],
                },
                indent=2,
            )
        )
        print(f"revalidated {args.revalidate} sha256={result['artifact_sha256']}")
        return 0 if result["decision"] == "GO" else 1

    missing = [
        flag
        for flag, value in (
            ("--selection", args.selection),
            ("--arm-artifacts", args.arm_artifacts),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        parser.error(f"build mode requires {', '.join(missing)}")
    assert args.selection is not None
    assert args.arm_artifacts is not None
    assert args.out is not None
    result = build_final_verdict(
        selection_path=args.selection,
        arm_paths=_parse_arm_artifacts(args.arm_artifacts),
        out=args.out,
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "final_arm": result["final_arm"],
                "claim": result["claim"],
                "failed": result["failed"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out} sha256={result['artifact_sha256']}")
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
