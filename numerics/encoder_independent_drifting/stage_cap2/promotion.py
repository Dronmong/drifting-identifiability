"""Immutable 150k -> 300k promotion certificate for CAP-EMF-2.

The expensive runner must not interpret a command-line boolean as scientific
approval.  A promotion instead binds the completed 150k unit, its declared EMA
checkpoint, a fresh numerical re-admission on that exact checkpoint, and the
development-only standard evaluation.  The builder and loader both recompute
the bindings; the JSON is a portable index, not a trusted assertion by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from .artifacts import (
    assert_unused,
    file_sha256,
    load_checkpoint,
    load_preflight,
    verify_file,
    verify_json,
    write_json_atomic,
)
from .early_admission import revalidate_early_admission
from .numerical_admission import (
    AUDIT_BATCH,
    AUDIT_SOURCES,
    AUDIT_STRATA,
    MINIMUM_REPEATS,
    PRODUCTION_MICROBATCH,
    admission_matrix_complete,
)
from .preflight import _same_kid_reference
from .standard_metrics import (
    REPOSITORY_AUXILIARY_PROTOCOL,
    REPOSITORY_FEATURE_SAMPLES,
    REPOSITORY_MEMORIZATION_SAMPLES,
    REPOSITORY_REFERENCE_SEED,
    revalidate_clean_evaluation_evidence,
)

PROMOTION_STATUS = "cap-emf2-promotion"
FROM_UPDATES = 150_000
TO_UPDATES = 300_000
DEVELOPMENT_SAMPLES = 50_000
DEVELOPMENT_GENERATION_SEED = 20_260_804
DEVELOPMENT_KID_SEED = 20_260_831
CLEANFID_VERSION = "0.1.35"
MIN_PRECISION = 0.05
MIN_RECALL = 0.05
MIN_PR_F1 = 0.05
MAX_EXACT_COPY_FRACTION = 0.0
MAX_GENERATED_DUPLICATE_FRACTION = 0.05
MAX_LATE_HH_MULTIPLICATIVE_CHANGE = 4.0
INFERENCE_CORNER_WINDOW_START = 100_000
INFERENCE_CORNER_WINDOW_MIDPOINT = 125_000
FIXED_INFERENCE_CORNER_SAMPLES = 2_048
MAX_LATE_INFERENCE_CORNER_ERROR_GROWTH = 4.0

FID_KEY = "clean_fid_cifar10_train"
KID_KEY = "clean_kid_cifar10_train"
DIRECT_CALIBRATION_KEYS = {
    FID_KEY: "clean_fid",
    KID_KEY: "clean_kid",
}
CALIBRATION_MARGIN_KIND = "absolute direct disjoint real/real discrepancy"
CONTROL_EXEMPT_SCIENTIFIC_CHECKS = frozenset(
    {
        "candidate_fid_improves_beyond_calibration",
        "candidate_kid_improves_beyond_calibration",
    }
)


def _control_continuation(checks: dict[str, bool]) -> dict[str, object]:
    """Separate control validity from success against a historical model.

    A freshly trained legacy arm is the concurrent baseline.  Requiring that
    baseline to beat a 650k historical checkpoint at 150k made the causal
    comparison contingent on the control itself being a scientific success.
    It may continue when every integrity, numerical, health, noncollapse and
    evaluation check passes; only its two historical-improvement checks are
    irrelevant to its role as a valid control.
    """

    required = {
        name: value
        for name, value in checks.items()
        if name not in CONTROL_EXEMPT_SCIENTIFIC_CHECKS
    }
    failed = sorted(name for name, ok in required.items() if ok is not True)
    return {
        "decision": "GO" if not failed else "NO_GO",
        "checks": required,
        "failed": failed,
        "exempt_scientific_checks": sorted(CONTROL_EXEMPT_SCIENTIFIC_CHECKS),
        "role": "concurrent legacy control validity; not a quality claim",
    }


def _reference(path: Path, anchor: Path) -> str:
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def _resolve(reference: str, anchor: Path) -> Path:
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("promotion contains an empty artifact reference")
    path = Path(reference)
    return path if path.is_absolute() else (anchor / path).resolve()


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _metric(payload: dict, key: str) -> float:
    value = payload.get(key)
    return float(value) if _finite(value) else math.nan


def _all_true(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and bool(payload)
        and all(value is True for value in payload.values())
    )


def _same_hardware(left: object, right: object) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    fields = (
        "actual_gpu_name",
        "compute_capability",
        "torch_version",
        "cuda_runtime",
        "cudnn_version",
        "cublas_workspace_config",
    )
    return (
        left.get("matches") is True
        and right.get("matches") is True
        and all(left.get(field) == right.get(field) for field in fields)
    )


def _hash64(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _development_evaluation_schema(evaluation: dict) -> bool:
    samples = evaluation.get("samples", {})
    metrics = evaluation.get("standard_train_reference_metrics", {})
    generated_features = metrics.get("generated_features", {})
    kid_reference = metrics.get("kid_reference", {})
    auxiliary = evaluation.get("repository_auxiliary", {})
    feature = auxiliary.get("repository_feature_metrics", {})
    memorization = auxiliary.get("memorization", {})
    generated_subset = auxiliary.get("generated_subset", {})
    reference_subset = auxiliary.get("reference_subset", {})
    provenance = evaluation.get("provenance", {})
    return (
        int(samples.get("count", -1)) == DEVELOPMENT_SAMPLES
        and isinstance(samples.get("directory"), str)
        and bool(samples["directory"].strip())
        and list(samples.get("image_size", [])) == [32, 32]
        and samples.get("mode") == "RGB"
        and _hash64(samples.get("png_manifest_sha256"))
        and metrics.get("backend") == "clean-fid"
        and metrics.get("cleanfid_version") == CLEANFID_VERSION
        and metrics.get("reference") == "cifar10/train/32"
        and metrics.get("mode") == "clean"
        and int(metrics.get("feature_count", -1)) == DEVELOPMENT_SAMPLES
        and isinstance(generated_features.get("path"), str)
        and bool(generated_features["path"].strip())
        and _hash64(generated_features.get("sha256"))
        and _hash64(generated_features.get("feature_sha256"))
        and int(generated_features.get("count", -1)) == DEVELOPMENT_SAMPLES
        and int(generated_features.get("dimension", -1)) == 2_048
        and generated_features.get("dtype") in {"float32", "float64"}
        and generated_features.get("source_png_manifest_sha256")
        == samples.get("png_manifest_sha256")
        and int(generated_features.get("generation_seed", -1))
        == DEVELOPMENT_GENERATION_SEED
        and isinstance(kid_reference.get("path"), str)
        and bool(kid_reference["path"].strip())
        and _hash64(kid_reference.get("sha256"))
        and _hash64(kid_reference.get("feature_sha256"))
        and int(kid_reference.get("count", -1)) == DEVELOPMENT_SAMPLES
        and int(kid_reference.get("dimension", -1)) == 2_048
        and kid_reference.get("dtype") in {"float32", "float64"}
        and auxiliary.get("protocol") == REPOSITORY_AUXILIARY_PROTOCOL
        and auxiliary.get("png_manifest_sha256") == samples.get("png_manifest_sha256")
        and int(generated_subset.get("count", -1)) == REPOSITORY_FEATURE_SAMPLES
        and int(reference_subset.get("count", -1)) == REPOSITORY_FEATURE_SAMPLES
        and int(reference_subset.get("seed", -1)) == REPOSITORY_REFERENCE_SEED
        and all(
            _hash64(value)
            for value in (
                auxiliary.get("cifar10_train_tensor_sha256"),
                generated_subset.get("tensor_sha256"),
                reference_subset.get("indices_sha256"),
                reference_subset.get("tensor_sha256"),
                feature.get("generated_feature_sha256"),
                feature.get("reference_feature_sha256"),
            )
        )
        and int(feature.get("samples_generated", -1)) == REPOSITORY_FEATURE_SAMPLES
        and int(feature.get("samples_reference", -1)) == REPOSITORY_FEATURE_SAMPLES
        and int(memorization.get("samples_generated", -1))
        == REPOSITORY_MEMORIZATION_SAMPLES
        and int(memorization.get("reference_train_images", -1)) == 50_000
        and evaluation.get("repository_feature_metrics") == feature
        and evaluation.get("memorization") == memorization
        and provenance.get("deterministic_algorithms") is True
        and provenance.get("packages", {}).get("clean-fid") == CLEANFID_VERSION
        and provenance.get("numerical_settings", {}).get("allow_tf32") is False
    )


def _evaluation_environment(payload: dict) -> dict:
    provenance = payload.get("provenance", {})
    metrics = payload.get("metrics") or payload.get(
        "standard_train_reference_metrics", {}
    )
    auxiliary = payload.get("repository_auxiliary", {})
    return {
        "packages": {
            name: provenance.get("packages", {}).get(name)
            for name in ("clean-fid", "numpy", "Pillow", "torch", "torchvision")
        },
        "device": provenance.get("device"),
        "gpu_name": provenance.get("gpu_name"),
        "torch_cuda_version": provenance.get("torch_cuda_version"),
        "cudnn_version": provenance.get("cudnn_version"),
        "image_quantization": provenance.get("image_quantization"),
        "numerical_settings": provenance.get("numerical_settings"),
        "deterministic_algorithms": provenance.get("deterministic_algorithms"),
        "generation_batch": payload.get("samples", {}).get("batch"),
        "metric_batch": metrics.get("metric_batch"),
        "metric_workers": metrics.get("metric_workers"),
        "feature_batch": auxiliary.get("repository_feature_metrics", {}).get(
            "feature_batch"
        ),
    }


def _reference_binding(payload: dict) -> dict:
    auxiliary = payload.get("repository_auxiliary", {})
    reference = auxiliary.get("reference_subset", {})
    features = auxiliary.get("repository_feature_metrics", {})
    return {
        "train_tensor": auxiliary.get("cifar10_train_tensor_sha256"),
        "seed": reference.get("seed"),
        "indices": reference.get("indices_sha256"),
        "images": reference.get("tensor_sha256"),
        "features": features.get("reference_feature_sha256"),
    }


def _calibration_tolerance(calibration: dict, key: str) -> float:
    """Return the observed direct real-vs-real discrepancy for ``key``.

    This is a deterministic promotion margin, not a confidence interval.  The
    calibration artifact itself explicitly disclaims variance estimation, so
    the certificate does not give this finite-sample discrepancy a stronger
    statistical interpretation.  In particular, this must not be replaced by
    the difference between two scores against the shared full-train reference:
    those highly correlated scores are only sanity checks and can nearly
    cancel even when the direct real/real discrepancy is material.
    """
    direct_key = DIRECT_CALIBRATION_KEYS.get(key)
    if direct_key is None:
        return math.nan
    direct = calibration.get("metrics", {}).get("direct_disjoint_pair", {})
    observed = _metric(direct, direct_key)
    # CleanKID is an unbiased finite-sample estimate and can be negative.  A
    # margin is a magnitude, so preserve the observation through abs rather
    # than allowing a negative estimate to weaken the gate.
    return abs(observed) if math.isfinite(observed) else math.nan


def _grid_integrity(evaluation: dict, *, anchor: Path) -> bool:
    grid = evaluation.get("uncurated_grid", {})
    path_value = grid.get("path")
    expected = grid.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str):
        return False
    path = _resolve(path_value, anchor)
    return (
        path.is_file()
        and file_sha256(path) == expected
        and int(grid.get("rows", 0)) == 8
        and int(grid.get("columns", 0)) == 16
        and "no curation" in str(grid.get("selection", "")).lower()
    )


def _checkpoint_previews_valid(result: dict, *, anchor: Path) -> bool:
    """Verify every fixed checkpoint preview and its durable-mirror ledger."""

    try:
        expected_steps = {
            int(step)
            for step in result["declared_profile"]["train"]["checkpoint_updates"]
        }
    except (KeyError, TypeError, ValueError):
        return False
    found: set[int] = set()
    for health in result.get("training", {}).get("health", []):
        if not isinstance(health, dict):
            return False
        observation = health.get("checkpoint_health_observation")
        if observation is None:
            continue
        try:
            step = int(health["step"])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            step not in expected_steps
            or step in found
            or observation.get("status") != "cap-emf2-fixed-checkpoint-previews"
            or int(observation.get("step", -1)) != step
            or observation.get("quantitative_role")
            != "report/veto only; never rescues a failed gate"
        ):
            return False
        records = observation.get("raw_and_ema")
        if not isinstance(records, dict) or set(records) != {"raw", "ema"}:
            return False
        for record in records.values():
            if not isinstance(record, dict):
                return False
            path_value = record.get("path")
            expected_sha = record.get("sha256")
            mirror = record.get("durable_mirror")
            if (
                not isinstance(path_value, str)
                or not path_value.strip()
                or not _hash64(expected_sha)
                or int(record.get("rows", -1)) != 8
                or int(record.get("columns", -1)) != 16
                or int(record.get("samples", -1)) != 128
                or "no curation" not in str(record.get("selection", "")).lower()
                or not isinstance(mirror, dict)
                or mirror.get("relative_path") != Path(path_value).as_posix()
                or mirror.get("sha256") != expected_sha
                or int(mirror.get("bytes", -1)) <= 0
            ):
                return False
            path = _resolve(path_value, anchor)
            try:
                if verify_file(path, expected_sha) != expected_sha:
                    return False
            except (OSError, RuntimeError, ValueError):
                return False
        found.add(step)
    return found == expected_steps


def _late_ema_hh_trajectory(result: dict) -> dict[str, object]:
    """Extract the last two checkpoint-EMA HH reports and gross stability.

    The factor-four threshold is deliberately a coarse explosion/collapse
    guard, not evidence that the trajectory has converged.  Promotion records
    the raw values and the named threshold so the interpretation is auditable.
    """
    points: list[dict[str, float | int]] = []
    for record in result.get("training", {}).get("health", []):
        components = record.get("ema_components")
        if not isinstance(components, dict):
            continue
        try:
            point = {
                "step": int(record["step"]),
                "base_haar_HH_ratio": float(components["base"]["haar_HH_ratio"]),
                "final_haar_HH_ratio": float(components["final"]["haar_HH_ratio"]),
                "residual_haar_HH_variance": float(
                    components["refiner_residual"]["haar_HH_variance"]
                ),
            }
        except (KeyError, TypeError, ValueError):
            continue
        points.append(point)
    points.sort(key=lambda point: int(point["step"]))
    late = points[-2:]
    fields = (
        "base_haar_HH_ratio",
        "final_haar_HH_ratio",
        "residual_haar_HH_variance",
    )
    finite_positive = len(late) == 2 and all(
        _finite(point[field]) and float(point[field]) > 0
        for point in late
        for field in fields
    )
    ratios = (
        {
            field: (
                max(float(late[0][field]), float(late[1][field]))
                / min(float(late[0][field]), float(late[1][field]))
            )
            for field in fields
        }
        if finite_positive
        else {}
    )
    stable = (
        finite_positive
        and [int(point["step"]) for point in late] == [100_000, FROM_UPDATES]
        and all(value <= MAX_LATE_HH_MULTIPLICATIVE_CHANGE for value in ratios.values())
    )
    return {
        "points": late,
        "multiplicative_changes": ratios,
        "maximum_multiplicative_change": MAX_LATE_HH_MULTIPLICATIVE_CHANGE,
        "stable": stable,
    }


def _late_inference_corner_trajectory(result: dict) -> dict[str, object]:
    """Audit every arm on one common, literal one-step endpoint probe.

    Natural occupancy near ``(t,h)=(1,1)`` is scientifically useful but varies
    by sampler construction: making a shared minimum row count a performance
    gate made the legacy and ordered-logitnormal arms nearly impossible to
    promote.  Checkpoint health now evaluates the same 2,048 sealed rows at
    exactly ``(t,r,h)=(1,0,1)`` for raw and EMA weights.  The sampled windows
    remain below as support diagnostics only.
    """

    windows = (
        (INFERENCE_CORNER_WINDOW_START, INFERENCE_CORNER_WINDOW_MIDPOINT),
        (INFERENCE_CORNER_WINDOW_MIDPOINT, FROM_UPDATES),
    )
    sampled_summaries: list[dict[str, float | int | None | bool]] = []
    history = result.get("training", {}).get("history", [])
    try:
        log_every = int(result["declared_profile"]["train"]["log_every"])
    except (KeyError, TypeError, ValueError):
        log_every = 0
    for low, high in windows:
        count = 0
        weighted_error = 0.0
        records = 0
        malformed = False
        for record in history:
            try:
                step = int(record["step"])
            except (KeyError, TypeError, ValueError):
                malformed = True
                continue
            if not (low < step <= high):
                continue
            corner = (
                record.get("objective_ledger", {})
                .get("named_regions", {})
                .get("inference_corner", {})
            )
            try:
                row_count = int(corner["count"])
                mean_error = float(corner["mean_raw_mse"])
            except (KeyError, TypeError, ValueError):
                malformed = True
                continue
            if row_count < 0 or (row_count and not math.isfinite(mean_error)):
                malformed = True
                continue
            count += row_count
            weighted_error += row_count * mean_error
            records += 1
        mean_error = weighted_error / count if count else None
        sampled_summaries.append(
            {
                "start_exclusive": low,
                "end_inclusive": high,
                "records": records,
                "expected_records": (
                    (high - low) // log_every if log_every > 0 else -1
                ),
                "rows": count,
                "mean_raw_mse": mean_error,
                "malformed": malformed,
            }
        )

    sampled_complete = all(
        int(summary["records"]) == int(summary["expected_records"])
        and int(summary["expected_records"]) > 0
        and summary["malformed"] is False
        for summary in sampled_summaries
    )

    try:
        declared_samples = int(result["declared_profile"]["train"]["audit_samples"])
        declared_objective = result["declared_profile"]["objective"]
    except (KeyError, TypeError, ValueError):
        declared_samples = -1
        declared_objective = {}
    if not isinstance(declared_objective, dict):
        declared_objective = {}
    health = result.get("training", {}).get("health", [])
    points: list[dict[str, object]] = []
    for wanted_step in (INFERENCE_CORNER_WINDOW_START, FROM_UPDATES):
        matches = [
            record
            for record in health
            if isinstance(record, dict) and record.get("step") == wanted_step
        ]
        if len(matches) != 1:
            points.append({"step": wanted_step, "valid": False})
            continue
        probe = matches[0].get("fixed_exact_inference_corner")
        if not isinstance(probe, dict):
            points.append({"step": wanted_step, "valid": False})
            continue
        kinds: dict[str, dict[str, float | int | bool]] = {}
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
                and summary.get("count") == declared_samples
                and summary.get("nonfinite_rows") == 0
                and len(values) == 6
                and all(_finite(value) and float(value) >= 0.0 for value in values)
            )
            kinds[kind] = {
                "count": summary.get("count", -1) if isinstance(summary, dict) else -1,
                "mean_raw_mse": (float(summary["mean_raw_mse"]) if valid else math.nan),
                "valid": valid,
            }
        point_valid = (
            probe.get("condition") == {"t": 1.0, "r": 0.0, "h": 1.0}
            and probe.get("sealed_train_only") is True
            and probe.get("sample_count") == declared_samples
            and declared_samples == FIXED_INFERENCE_CORNER_SAMPLES
            and probe.get("objective_numerics")
            == {
                "stopped_evaluation": declared_objective.get("stopped_evaluation"),
                "emf_delta": declared_objective.get("emf_delta"),
                "emf_denominator_floor": declared_objective.get(
                    "emf_denominator_floor"
                ),
            }
            and all(record["valid"] is True for record in kinds.values())
        )
        points.append(
            {
                "step": wanted_step,
                "condition": probe.get("condition"),
                "sample_count": probe.get("sample_count"),
                "raw": kinds["raw"],
                "ema": kinds["ema"],
                "valid": point_valid,
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
        "fixed_probe_points": points,
        "fixed_samples": FIXED_INFERENCE_CORNER_SAMPLES,
        "maximum_late_error_growth": MAX_LATE_INFERENCE_CORNER_ERROR_GROWTH,
        "late_error_growth": growth,
        "natural_sampled_support": {
            "definition": "t > 0.95 and h = t - r > 0.90",
            "windows": sampled_summaries,
            "complete": sampled_complete,
            "role": "sampler-support diagnostic only; no common minimum occupancy",
        },
        "stable": stable,
    }


def _promotion_inputs(
    *,
    preflight_path: Path,
    result_path: Path,
    raw_checkpoint_path: Path,
    checkpoint_path: Path,
    admission_path: Path,
    evaluation_path: Path,
    arm: str | None = None,
    candidate: str | None = None,
) -> dict:
    preflight = load_preflight(preflight_path)
    result = verify_json(result_path, "cap-emf2-screen-unit")
    admission = verify_json(admission_path, "cap-emf2-numerical-admission")
    evaluation = verify_json(evaluation_path, "cap-emf2-development-evaluation")
    evaluation_evidence = revalidate_clean_evaluation_evidence(
        evaluation, anchor=evaluation_path.parent
    )

    selected_arm = arm or result.get("arm")
    selected_candidate = candidate or result.get("numerical_candidate")
    if selected_arm not in preflight.get("profiles_150k", {}):
        raise RuntimeError(f"promotion arm is not preflighted: {selected_arm!r}")
    preflight_inputs = preflight.get("inputs", {})
    baseline = preflight_inputs.get("baseline_standard")
    positive_control = preflight_inputs.get("positive_control_standard")
    metric_calibration = preflight_inputs.get("metric_calibration")
    if not isinstance(baseline, dict):
        raise TypeError("preflight does not bind the standard CAP-EMF-1 baseline")
    if not isinstance(positive_control, dict):
        raise TypeError("preflight does not bind a standard positive control")
    if not isinstance(metric_calibration, dict):
        raise TypeError("preflight does not bind the real/real metric calibration")
    declared = result.get("declared_profile")
    realized = result.get("realized_profile")
    checkpoint_record = (
        result.get("checkpoints", {}).get(str(FROM_UPDATES), {}).get("ema", {})
    )
    raw_checkpoint_record = (
        result.get("checkpoints", {}).get(str(FROM_UPDATES), {}).get("raw", {})
    )
    checkpoint_sha = checkpoint_record.get("sha256")
    raw_checkpoint_sha = raw_checkpoint_record.get("sha256")
    early_record = result.get("early_admission")
    if not isinstance(early_record, dict):
        raise TypeError("150k result lacks its 50k raw-state admission")
    early_path = _resolve(early_record.get("path", ""), result_path.parent)
    early_admission = revalidate_early_admission(early_path)
    checkpoint = load_checkpoint(
        checkpoint_path,
        expected_sha=checkpoint_sha,
        step=FROM_UPDATES,
        kind="ema",
        arm=selected_arm,
        declared_profile=declared,
        realized_profile=realized,
        preflight_sha256=preflight["artifact_sha256"],
        run_identity_sha256=result.get("run_identity_sha256"),
        unit_seed=int(result.get("unit_seed", -1)),
    )
    raw_checkpoint = load_checkpoint(
        raw_checkpoint_path,
        expected_sha=raw_checkpoint_sha,
        step=FROM_UPDATES,
        kind="raw",
        arm=selected_arm,
        declared_profile=declared,
        realized_profile=realized,
        preflight_sha256=preflight["artifact_sha256"],
        run_identity_sha256=result.get("run_identity_sha256"),
        unit_seed=int(result.get("unit_seed", -1)),
    )

    # The recorded values must agree with this reconstruction, but promotion
    # is decided from the independently recomputed archive values themselves.
    candidate_metrics = {
        **evaluation.get("standard_train_reference_metrics", {}),
        **evaluation_evidence["recomputed"],
    }
    baseline_metrics = baseline.get("metrics", {})
    control_metrics = positive_control.get("metrics", {})
    candidate_fid = _metric(candidate_metrics, FID_KEY)
    candidate_kid = _metric(candidate_metrics, KID_KEY)
    baseline_fid = _metric(baseline_metrics, FID_KEY)
    baseline_kid = _metric(baseline_metrics, KID_KEY)
    control_fid = _metric(control_metrics, FID_KEY)
    control_kid = _metric(control_metrics, KID_KEY)
    fid_tolerance = _calibration_tolerance(metric_calibration, FID_KEY)
    kid_tolerance = _calibration_tolerance(metric_calibration, KID_KEY)

    feature_metrics = evaluation.get("repository_feature_metrics", {})
    baseline_feature_metrics = baseline.get("repository_feature_metrics", {})
    precision = _metric(feature_metrics, "precision")
    recall = _metric(feature_metrics, "recall")
    pr_f1 = _metric(feature_metrics, "pr_f1")
    repository_kid = _metric(feature_metrics, "unbiased_kid")
    baseline_precision = _metric(baseline_feature_metrics, "precision")
    baseline_recall = _metric(baseline_feature_metrics, "recall")
    baseline_pr_f1 = _metric(baseline_feature_metrics, "pr_f1")
    baseline_repository_kid = _metric(baseline_feature_metrics, "unbiased_kid")
    expected_f1 = (
        2 * precision * recall / (precision + recall)
        if precision > 0 and recall > 0
        else 0.0
    )
    memorization = evaluation.get("memorization", {})
    evaluation_provenance = evaluation.get("provenance", {})
    exact_copy = _metric(memorization, "exact_pixel_copy_fraction")
    duplicate_fraction = _metric(memorization, "exact_generated_duplicate_fraction")
    late_hh = _late_ema_hh_trajectory(result)
    late_corner = _late_inference_corner_trajectory(result)
    realized_train = realized.get("train", {}) if isinstance(realized, dict) else {}
    benchmark = preflight_inputs.get("benchmark", {})
    initial_admission = preflight_inputs.get("numerical_admission", {})
    readmission_rows = admission.get("strata", [])
    expected_readmission_keys = {
        (repeat, source, name)
        for repeat in range(int(admission.get("repeats", 0)))
        for source in AUDIT_SOURCES
        for name, _t, _r in AUDIT_STRATA
    }
    actual_readmission_keys = {
        (
            int(row.get("repeat", -1)),
            row.get("input_batch", {}).get("source"),
            row.get("stratum"),
        )
        for row in readmission_rows
    }
    expected_mixed_keys = {
        (repeat, source)
        for repeat in range(int(admission.get("repeats", 0)))
        for source in AUDIT_SOURCES
    }
    readmission_mixed = admission.get("mixed_gradient", [])
    readmission_production_shape = admission.get("production_shape", [])

    def mixed_readmission_valid(
        rows: object, *, expected_batch: int, gradient: bool
    ) -> bool:
        return (
            isinstance(rows, list)
            and len(rows) == len(expected_mixed_keys)
            and {
                (
                    int(row.get("repeat", -1)),
                    row.get("input_batch", {}).get("source"),
                )
                for row in rows
            }
            == expected_mixed_keys
            and all(
                row.get("verdict") == "PASS"
                and _all_true(row.get("admission_checks"))
                and int(row.get("input_batch", {}).get("batch", -1)) == expected_batch
                and (isinstance(row.get("gradient"), dict)) == gradient
                for row in rows
            )
        )

    expected_checkpoint_steps = {
        int(step) for step in declared.get("train", {}).get("checkpoint_updates", [])
    }
    recorded_checkpoint_steps = {int(step) for step in result.get("checkpoints", {})}
    gate_record = result.get("train_only_gate", {})

    comparison = {
        "candidate": {
            "clean_fid": candidate_fid,
            "clean_kid": candidate_kid,
            "repository_kid": repository_kid,
            "precision": precision,
            "recall": recall,
            "pr_f1": pr_f1,
        },
        "baseline": {
            "clean_fid": baseline_fid,
            "clean_kid": baseline_kid,
            "repository_kid": baseline_repository_kid,
            "precision": baseline_precision,
            "recall": baseline_recall,
            "pr_f1": baseline_pr_f1,
        },
        "positive_control": {
            "clean_fid": control_fid,
            "clean_kid": control_kid,
        },
        "calibration_margin": {
            "kind": CALIBRATION_MARGIN_KIND,
            "clean_fid": fid_tolerance,
            "clean_kid": kid_tolerance,
            "statistical_scope": (
                "deterministic finite-sample margin; not a confidence interval"
            ),
        },
        "noncollapse_floors": {
            "precision": MIN_PRECISION,
            "recall": MIN_RECALL,
            "pr_f1": MIN_PR_F1,
        },
        "auxiliary_relative_diagnostics": {
            "policy_role": "reported only; not a promotion requirement",
            "repository_kid_lower_than_baseline": (
                math.isfinite(repository_kid)
                and math.isfinite(baseline_repository_kid)
                and repository_kid < baseline_repository_kid
            ),
            "does_not_lose_both_precision_and_recall": (
                all(
                    math.isfinite(value)
                    for value in (
                        precision,
                        recall,
                        baseline_precision,
                        baseline_recall,
                    )
                )
                and not (precision < baseline_precision and recall < baseline_recall)
            ),
        },
        "late_ema_hh": late_hh,
        "late_inference_corner": late_corner,
    }

    checks = {
        "result_is_150k": (
            int(result.get("training", {}).get("optimizer_updates", -1)) == FROM_UPDATES
            and int(
                result.get("declared_profile", {}).get("train", {}).get("updates", -1)
            )
            == FROM_UPDATES
        ),
        "result_development_only": result.get("development_only") is True,
        "result_arm": result.get("arm") == selected_arm,
        "result_candidate": result.get("numerical_candidate") == selected_candidate,
        "result_preflight": (
            result.get("preflight_sha256") == preflight["artifact_sha256"]
        ),
        "result_profile": declared == preflight["profiles_150k"][selected_arm],
        "result_has_realized_profile": (
            isinstance(realized, dict)
            and int(realized_train.get("updates", -1)) == FROM_UPDATES
            and int(realized_train.get("micro_batch", -1))
            == int(benchmark.get("micro_batch", -2))
            and int(realized_train.get("accumulation_steps", -1))
            == int(benchmark.get("accumulation_steps", -2))
            and int(realized_train.get("micro_batch", -1))
            * int(realized_train.get("accumulation_steps", -1))
            == int(benchmark.get("effective_batch", -2))
        ),
        "result_has_run_identity": (
            isinstance(result.get("run_identity_sha256"), str)
            and len(result["run_identity_sha256"]) == 64
        ),
        "result_has_unit_seed": (
            int(result.get("unit_seed", -1)) >= 0
            and int(result.get("unit_seed", -1)) == int(benchmark.get("unit_seed", -2))
        ),
        "train_only_gate": (
            gate_record.get("verdict") == "PASS"
            and _all_true(gate_record.get("checks"))
            and gate_record.get("thresholds") == declared.get("gate")
        ),
        "checkpoint_ladder_complete": (
            expected_checkpoint_steps == recorded_checkpoint_steps
            and expected_checkpoint_steps
            and all(
                set(result.get("checkpoints", {}).get(str(step), {})) == {"raw", "ema"}
                and all(
                    _hash64(record.get("sha256"))
                    and isinstance(record.get("path"), str)
                    and bool(record["path"])
                    for record in result["checkpoints"][str(step)].values()
                )
                for step in expected_checkpoint_steps
            )
        ),
        "checkpoint_previews_complete": _checkpoint_previews_valid(
            result, anchor=result_path.parent
        ),
        "checkpoint_record": checkpoint_sha == checkpoint["artifact_sha256"],
        "raw_checkpoint_record": (
            raw_checkpoint_sha == raw_checkpoint["artifact_sha256"]
        ),
        "early_admission_bound": (
            early_record.get("sha256") == early_admission["artifact_sha256"]
            and early_admission.get("decision") == "GO"
            and early_admission.get("arm") == selected_arm
            and early_admission.get("candidate") == selected_candidate
            and early_admission.get("preflight_sha256") == preflight["artifact_sha256"]
        ),
        "readmission_go": admission.get("decision") == "GO",
        "readmission_checkpoint": (
            admission.get("checkpoint_sha256") == raw_checkpoint["artifact_sha256"]
            and int(admission.get("checkpoint_step", -1)) == FROM_UPDATES
        ),
        "readmission_candidate": (
            admission.get("candidate", {}).get("name") == selected_candidate
        ),
        "readmission_checkpoint_identity": (
            admission.get("checkpoint_identity", {}).get("valid") is True
            and admission.get("checkpoint_identity", {}).get("stage")
            == "cap-emf-2-screen"
            and admission.get("checkpoint_identity", {}).get("kind") == "raw"
            and admission.get("checkpoint_identity", {}).get("arm") == selected_arm
        ),
        "readmission_full_matrix": (
            admission_matrix_complete(admission)
            and admission.get("decision") == "GO"
            and admission.get("gradient_checked") is True
            and admission.get("cuda_admission") is True
            and int(admission.get("batch_per_stratum", -1)) == AUDIT_BATCH
            and int(admission.get("repeats", -1)) >= MINIMUM_REPEATS
            and admission.get("design", {}).get("sources") == list(AUDIT_SOURCES)
            and admission.get("design", {}).get("strata")
            == [
                {"name": name, "t": t, "r": r, "h": t - r}
                for name, t, r in AUDIT_STRATA
            ]
            and len(readmission_rows) == len(expected_readmission_keys)
            and actual_readmission_keys == expected_readmission_keys
            and all(
                row.get("verdict") == "PASS" and _all_true(row.get("admission_checks"))
                for row in readmission_rows
            )
            and mixed_readmission_valid(
                readmission_mixed,
                expected_batch=AUDIT_BATCH,
                gradient=True,
            )
            and mixed_readmission_valid(
                readmission_production_shape,
                expected_batch=PRODUCTION_MICROBATCH,
                gradient=True,
            )
            and _all_true(admission.get("protocol_checks"))
        ),
        "readmission_same_environment": (
            _same_hardware(admission.get("hardware"), initial_admission.get("hardware"))
            and _same_hardware(admission.get("hardware"), benchmark.get("hardware"))
            and _same_hardware(admission.get("hardware"), result.get("hardware"))
            and admission.get("production_numerical_mode")
            == initial_admission.get("production_numerical_mode")
        ),
        "readmission_sources": (
            admission.get("source_sha256") == preflight.get("source_sha256")
        ),
        "baseline_artifact": (
            baseline.get("status") == "cap-emf-standard-evaluation"
            and baseline.get("checkpoint_kind") == "ema"
            and int(baseline.get("samples", {}).get("count", 0)) == DEVELOPMENT_SAMPLES
            and baseline_metrics.get("backend") == "clean-fid"
            and baseline_metrics.get("cleanfid_version") == CLEANFID_VERSION
            and baseline.get("source_sha256") == preflight.get("source_sha256")
        ),
        "positive_control_artifact": (
            positive_control.get("status") == "cap-emf-standard-evaluation"
            and positive_control.get("checkpoint") is None
            and positive_control.get("checkpoint_kind") == "external-positive-control"
            and int(positive_control.get("samples", {}).get("count", 0))
            == DEVELOPMENT_SAMPLES
            and control_metrics.get("backend") == "clean-fid"
            and control_metrics.get("cleanfid_version") == CLEANFID_VERSION
            and positive_control.get("source_sha256") == preflight.get("source_sha256")
        ),
        "metric_calibration_artifact": (
            metric_calibration.get("status") == "cap-emf2-real-real-calibration"
            and metric_calibration.get("decision") == "COMPLETE"
            and int(metric_calibration.get("samples_per_side", 0)) >= 10_000
            and metric_calibration.get("metrics", {}).get("backend") == "clean-fid"
            and metric_calibration.get("metrics", {}).get("cleanfid_version")
            == CLEANFID_VERSION
            and metric_calibration.get("source_sha256")
            == preflight.get("source_sha256")
        ),
        "comparative_metrics_finite": all(
            math.isfinite(value)
            for value in (
                candidate_fid,
                candidate_kid,
                baseline_fid,
                baseline_kid,
                control_fid,
                control_kid,
                fid_tolerance,
                kid_tolerance,
            )
        )
        and 0.0 <= candidate_fid
        and 0.0 <= baseline_fid
        and 0.0 <= control_fid
        and 0.0 <= fid_tolerance
        and 0.0 <= kid_tolerance,
        "positive_control_better_than_baseline": (
            control_fid < baseline_fid and control_kid < baseline_kid
        ),
        "candidate_fid_improves_beyond_calibration": (
            candidate_fid < baseline_fid - fid_tolerance
        ),
        "candidate_kid_improves_beyond_calibration": (
            candidate_kid < baseline_kid - kid_tolerance
        ),
        "same_backend_feature_protocol": (
            feature_metrics.get("backend")
            == baseline_feature_metrics.get("backend")
            == "torchvision Inception-v3 ImageNet1K_V1 pool features"
            and int(feature_metrics.get("samples_generated", -1)) == 2_048
            and int(feature_metrics.get("samples_reference", -1)) == 2_048
            and int(baseline_feature_metrics.get("samples_generated", -1)) == 2_048
            and int(baseline_feature_metrics.get("samples_reference", -1)) == 2_048
        ),
        "evaluation_auxiliary_metrics_finite": all(
            math.isfinite(value)
            for value in (
                repository_kid,
                baseline_repository_kid,
                precision,
                recall,
                baseline_precision,
                baseline_recall,
            )
        ),
        "same_auxiliary_reference": (
            _reference_binding(evaluation) == _reference_binding(baseline)
            and _reference_binding(evaluation).get("seed") == REPOSITORY_REFERENCE_SEED
            and all(
                _hash64(value)
                for key, value in _reference_binding(evaluation).items()
                if key != "seed"
            )
        ),
        "same_evaluation_environment": (
            _evaluation_environment(evaluation) == _evaluation_environment(baseline)
            and all(
                value is not None
                for value in _evaluation_environment(evaluation).values()
            )
        ),
        "evaluation_development_only": evaluation.get("development_only") is True,
        "evaluation_arm": evaluation.get("arm") == selected_arm,
        "evaluation_unit": (
            evaluation.get("unit", {}).get("sha256") == result["artifact_sha256"]
            and evaluation.get("unit", {}).get("preflight_sha256")
            == preflight["artifact_sha256"]
        ),
        "evaluation_checkpoint": (
            evaluation.get("checkpoint", {}).get("sha256")
            == checkpoint["artifact_sha256"]
            and int(evaluation.get("checkpoint", {}).get("step", -1)) == FROM_UPDATES
            and evaluation.get("checkpoint", {}).get("kind") == "ema"
            and int(evaluation.get("step", -1)) == FROM_UPDATES
        ),
        "evaluation_50k": (
            int(evaluation.get("samples", {}).get("count", 0)) == DEVELOPMENT_SAMPLES
            and int(evaluation.get("fixed_protocol", {}).get("generated_samples", -1))
            == DEVELOPMENT_SAMPLES
        ),
        "evaluation_fixed_seeds": (
            int(evaluation.get("samples", {}).get("seed", -1))
            == DEVELOPMENT_GENERATION_SEED
            and int(evaluation.get("fixed_protocol", {}).get("generation_seed", -1))
            == DEVELOPMENT_GENERATION_SEED
            and int(evaluation.get("fixed_protocol", {}).get("clean_kid_seed", -1))
            == DEVELOPMENT_KID_SEED
        ),
        "evaluation_one_step": (
            evaluation.get("samples", {}).get("one_model_call_per_batch") is True
        ),
        "evaluation_cleanfid": (
            evaluation.get("standard_train_reference_metrics", {}).get("backend")
            == "clean-fid"
            and evaluation.get("standard_train_reference_metrics", {}).get(
                "cleanfid_version"
            )
            == CLEANFID_VERSION
            and evaluation.get("standard_train_reference_metrics", {}).get("reference")
            == "cifar10/train/32"
            and evaluation.get("standard_train_reference_metrics", {}).get("mode")
            == "clean"
            and int(
                evaluation.get("standard_train_reference_metrics", {}).get(
                    "feature_count", -1
                )
            )
            == DEVELOPMENT_SAMPLES
            and int(
                evaluation.get("standard_train_reference_metrics", {}).get(
                    "kid_seed", -1
                )
            )
            == DEVELOPMENT_KID_SEED
        ),
        "evaluation_exact_kid_reference": _same_kid_reference(
            evaluation.get("standard_train_reference_metrics", {}).get("kid_reference"),
            baseline_metrics.get("kid_reference"),
        ),
        "evaluation_shared_auxiliary": _development_evaluation_schema(evaluation),
        "evaluation_leaf_evidence_recomputed": (
            evaluation_evidence.get("valid") is True
        ),
        "evaluation_provenance": (
            evaluation_provenance.get("deterministic_algorithms") is True
            and evaluation_provenance.get("packages", {}).get("clean-fid")
            == CLEANFID_VERSION
            and evaluation_provenance.get("numerical_settings", {}).get("allow_tf32")
            is False
        ),
        "evaluation_precision_recall_noncollapse": (
            all(math.isfinite(value) for value in (precision, recall, pr_f1))
            and feature_metrics.get("backend")
            == "torchvision Inception-v3 ImageNet1K_V1 pool features"
            and int(feature_metrics.get("samples_generated", 0)) == 2_048
            and int(feature_metrics.get("samples_reference", 0)) == 2_048
            and MIN_PRECISION <= precision <= 1.0
            and MIN_RECALL <= recall <= 1.0
            and MIN_PR_F1 <= pr_f1 <= 1.0
            and abs(pr_f1 - expected_f1) <= 1e-6
        ),
        "evaluation_memorization": (
            int(memorization.get("samples_generated", 0)) == 256
            and math.isfinite(exact_copy)
            and math.isfinite(duplicate_fraction)
            and 0.0 <= exact_copy <= MAX_EXACT_COPY_FRACTION
            and 0.0 <= duplicate_fraction <= MAX_GENERATED_DUPLICATE_FRACTION
        ),
        "evaluation_uncurated_grid": _grid_integrity(
            evaluation, anchor=evaluation_path.parent
        ),
        "evaluation_sources": (
            evaluation.get("source_sha256") == preflight.get("source_sha256")
        ),
        "late_ema_hh_trajectory": late_hh["stable"] is True,
        "late_inference_corner_trajectory": late_corner["stable"] is True,
    }
    return {
        "preflight": preflight,
        "result": result,
        "checkpoint": checkpoint,
        "raw_checkpoint": raw_checkpoint,
        "early_admission": early_admission,
        "admission": admission,
        "evaluation": evaluation,
        "arm": selected_arm,
        "candidate": selected_candidate,
        "comparison": comparison,
        "checks": checks,
    }


def build_promotion(
    *,
    preflight_path: Path,
    result_path: Path,
    raw_checkpoint_path: Path,
    checkpoint_path: Path,
    admission_path: Path,
    evaluation_path: Path,
    out: Path,
) -> dict:
    assert_unused(out)
    inputs = _promotion_inputs(
        preflight_path=preflight_path,
        result_path=result_path,
        raw_checkpoint_path=raw_checkpoint_path,
        checkpoint_path=checkpoint_path,
        admission_path=admission_path,
        evaluation_path=evaluation_path,
    )
    failed = sorted(name for name, ok in inputs["checks"].items() if not ok)
    control_continuation = _control_continuation(inputs["checks"])
    anchor = out.parent
    payload = {
        "status": PROMOTION_STATUS,
        "decision": "GO" if not failed else "NO_GO",
        "from_updates": FROM_UPDATES,
        "to_updates": TO_UPDATES,
        "arm": inputs["arm"],
        "candidate": inputs["candidate"],
        "preflight_sha256": inputs["preflight"]["artifact_sha256"],
        "result_sha256": inputs["result"]["artifact_sha256"],
        "checkpoint_sha256": inputs["checkpoint"]["artifact_sha256"],
        "raw_checkpoint_sha256": inputs["raw_checkpoint"]["artifact_sha256"],
        "readmission_sha256": inputs["admission"]["artifact_sha256"],
        "development_evaluation_sha256": inputs["evaluation"]["artifact_sha256"],
        "comparison": inputs["comparison"],
        "checks": inputs["checks"],
        "failed": failed,
        "control_continuation": control_continuation,
        "references": {
            "preflight": _reference(preflight_path, anchor),
            "result_150k": _reference(result_path, anchor),
            "checkpoint_150k_raw": _reference(raw_checkpoint_path, anchor),
            "checkpoint_150k_ema": _reference(checkpoint_path, anchor),
            "readmission": _reference(admission_path, anchor),
            "development_evaluation": _reference(evaluation_path, anchor),
        },
        "limits": [
            "This certificate proves individual 150k eligibility but is not sufficient without the concurrent cross-arm selection.",
            "It does not authorize a fresh 300k run, a confirmation, ASFD, or test-set selection.",
        ],
    }
    payload["artifact_sha256"] = write_json_atomic(out, payload)
    return payload


def load_promotion(
    path: Path,
    *,
    preflight_path: Path,
    result_path: Path,
    raw_checkpoint_path: Path,
    checkpoint_path: Path,
    arm: str,
    candidate: str,
    require_go: bool = True,
) -> dict:
    promotion = verify_json(path, PROMOTION_STATUS)
    references = promotion.get("references", {})
    admission_path = _resolve(references.get("readmission", ""), path.parent)
    evaluation_path = _resolve(
        references.get("development_evaluation", ""), path.parent
    )
    inputs = _promotion_inputs(
        preflight_path=preflight_path,
        result_path=result_path,
        raw_checkpoint_path=raw_checkpoint_path,
        checkpoint_path=checkpoint_path,
        admission_path=admission_path,
        evaluation_path=evaluation_path,
        arm=arm,
        candidate=candidate,
    )
    policy_failed = sorted(name for name, ok in inputs["checks"].items() if not ok)
    decision = "GO" if not policy_failed else "NO_GO"
    control_continuation = _control_continuation(inputs["checks"])
    bindings = {
        "decision": promotion.get("decision") == decision,
        "failed": promotion.get("failed") == policy_failed,
        "from_updates": promotion.get("from_updates") == FROM_UPDATES,
        "to_updates": promotion.get("to_updates") == TO_UPDATES,
        "arm": promotion.get("arm") == arm,
        "candidate": promotion.get("candidate") == candidate,
        "preflight": (
            promotion.get("preflight_sha256") == inputs["preflight"]["artifact_sha256"]
        ),
        "result": promotion.get("result_sha256") == inputs["result"]["artifact_sha256"],
        "checkpoint": (
            promotion.get("checkpoint_sha256")
            == inputs["checkpoint"]["artifact_sha256"]
        ),
        "raw_checkpoint": (
            promotion.get("raw_checkpoint_sha256")
            == inputs["raw_checkpoint"]["artifact_sha256"]
        ),
        "readmission": (
            promotion.get("readmission_sha256")
            == inputs["admission"]["artifact_sha256"]
        ),
        "evaluation": (
            promotion.get("development_evaluation_sha256")
            == inputs["evaluation"]["artifact_sha256"]
        ),
        "comparison_unchanged": promotion.get("comparison") == inputs["comparison"],
        "checks_unchanged": promotion.get("checks") == inputs["checks"],
        "control_continuation_unchanged": (
            promotion.get("control_continuation") == control_continuation
        ),
    }
    binding_failed = sorted(name for name, ok in bindings.items() if not ok)
    if binding_failed:
        raise RuntimeError(f"CAP2 promotion binding failed: {binding_failed}")
    if require_go and decision != "GO":
        raise RuntimeError("CAP2 promotion did not return GO")
    promotion["revalidated"] = True
    return promotion


def revalidate_promotion(path: Path, *, require_go: bool = False) -> dict:
    """Recompute a promotion record using only its immutable references.

    Cross-arm selection must inspect valid ``NO_GO`` records as well as valid
    ``GO`` records.  ``load_promotion`` intentionally rejects the former for
    the expensive runner, whereas this function verifies that either decision
    is the one implied by the bound result, checkpoint, re-admission, and
    development evaluation.
    """
    promotion = verify_json(path, PROMOTION_STATUS)
    references = promotion.get("references")
    required_references = {
        "preflight",
        "result_150k",
        "checkpoint_150k_raw",
        "checkpoint_150k_ema",
        "readmission",
        "development_evaluation",
    }
    if not isinstance(references, dict) or set(references) != required_references:
        raise RuntimeError("CAP2 promotion has an incomplete reference ledger")
    inputs = _promotion_inputs(
        preflight_path=_resolve(references["preflight"], path.parent),
        result_path=_resolve(references["result_150k"], path.parent),
        raw_checkpoint_path=_resolve(references["checkpoint_150k_raw"], path.parent),
        checkpoint_path=_resolve(references["checkpoint_150k_ema"], path.parent),
        admission_path=_resolve(references["readmission"], path.parent),
        evaluation_path=_resolve(references["development_evaluation"], path.parent),
        arm=promotion.get("arm"),
        candidate=promotion.get("candidate"),
    )
    failed = sorted(name for name, ok in inputs["checks"].items() if not ok)
    decision = "GO" if not failed else "NO_GO"
    control_continuation = _control_continuation(inputs["checks"])
    bindings = {
        "decision": promotion.get("decision") == decision,
        "failed": promotion.get("failed") == failed,
        "from_updates": promotion.get("from_updates") == FROM_UPDATES,
        "to_updates": promotion.get("to_updates") == TO_UPDATES,
        "arm": promotion.get("arm") == inputs["arm"],
        "candidate": promotion.get("candidate") == inputs["candidate"],
        "preflight": (
            promotion.get("preflight_sha256") == inputs["preflight"]["artifact_sha256"]
        ),
        "result": (
            promotion.get("result_sha256") == inputs["result"]["artifact_sha256"]
        ),
        "checkpoint": (
            promotion.get("checkpoint_sha256")
            == inputs["checkpoint"]["artifact_sha256"]
        ),
        "raw_checkpoint": (
            promotion.get("raw_checkpoint_sha256")
            == inputs["raw_checkpoint"]["artifact_sha256"]
        ),
        "readmission": (
            promotion.get("readmission_sha256")
            == inputs["admission"]["artifact_sha256"]
        ),
        "evaluation": (
            promotion.get("development_evaluation_sha256")
            == inputs["evaluation"]["artifact_sha256"]
        ),
        "comparison": promotion.get("comparison") == inputs["comparison"],
        "checks": promotion.get("checks") == inputs["checks"],
        "control_continuation": (
            promotion.get("control_continuation") == control_continuation
        ),
    }
    invalid = sorted(name for name, ok in bindings.items() if not ok)
    if invalid:
        raise RuntimeError(f"CAP2 promotion revalidation failed: {invalid}")
    if require_go and decision != "GO":
        raise RuntimeError("CAP2 promotion did not return GO")
    promotion["revalidated"] = True
    return promotion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--result-150k", type=Path, required=True)
    parser.add_argument("--checkpoint-150k-raw", type=Path, required=True)
    parser.add_argument("--checkpoint-150k-ema", type=Path, required=True)
    parser.add_argument("--readmission", type=Path, required=True)
    parser.add_argument("--development-evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--allow-valid-legacy-control",
        action="store_true",
        help=(
            "return success for a legacy quality NO_GO only when its separately "
            "recomputed control-continuation certificate is GO"
        ),
    )
    args = parser.parse_args()
    result = build_promotion(
        preflight_path=args.preflight,
        result_path=args.result_150k,
        raw_checkpoint_path=args.checkpoint_150k_raw,
        checkpoint_path=args.checkpoint_150k_ema,
        admission_path=args.readmission,
        evaluation_path=args.development_evaluation,
        out=args.out,
    )
    print(json.dumps({key: result[key] for key in ("decision", "failed")}, indent=2))
    print(f"wrote {args.out} sha256={result['artifact_sha256']}")
    allowed_control = (
        args.allow_valid_legacy_control
        and result.get("arm") == "legacy"
        and result.get("control_continuation", {}).get("decision") == "GO"
    )
    if args.allow_valid_legacy_control and result.get("arm") != "legacy":
        raise RuntimeError("--allow-valid-legacy-control is valid only for legacy")
    return 0 if result["decision"] == "GO" or allowed_control else 1


if __name__ == "__main__":
    raise SystemExit(main())
