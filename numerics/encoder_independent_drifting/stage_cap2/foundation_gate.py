"""Fail-closed capability gate for the ordered 750k CAP2 foundation.

This certificate is the only CAP2 artifact ASFD may consume.  It binds the
single-campaign preflight, complete 750k result/recovery/checkpoint ladder,
checkpoint-specific raw numerical readmission, fixed 650k/750k 50k-sample
train-reference evaluations, retained metric populations, and the human review
of the fixed uncurated 750k grid.  It does not open CIFAR-10 test data.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from ..stage_cap.training import load_recovery_payload, validate_recovery_counters
from .artifacts import (
    assert_unused,
    load_checkpoint,
    load_preflight,
    profile_payload,
    verify_json,
    write_json_atomic,
)
from .config import apply_calibrated_gate, screen_profile
from .development_evaluation import verify_final_ema
from .durable_mirror import DurableMirror
from .numerical_admission import admission_matrix_complete
from .promotion import (
    FID_KEY,
    KID_KEY,
    MAX_EXACT_COPY_FRACTION,
    MAX_GENERATED_DUPLICATE_FRACTION,
    MIN_PR_F1,
    MIN_PRECISION,
    MIN_RECALL,
    _calibration_tolerance,
    _checkpoint_previews_valid,
    _development_evaluation_schema,
    _evaluation_environment,
    _grid_integrity,
    _reference_binding,
)
from .standard_metrics import revalidate_clean_evaluation_evidence

STATUS = "cap-emf2-750k-foundation-gate"
CAMPAIGN = "ordered_750_foundation"
ARM = "ordered_uniform"
FINAL_STEP = 750_000
COMPARISON_STEP = 650_000


def _reference(path: Path, anchor: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metric(payload: dict, key: str) -> float:
    value = payload.get(key)
    return float(value) if _finite(value) else math.nan


def _auxiliary_noncollapse(evaluation: dict) -> bool:
    features = evaluation.get("repository_feature_metrics", {})
    memorization = evaluation.get("memorization", {})
    precision = _metric(features, "precision")
    recall = _metric(features, "recall")
    pr_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return (
        precision >= MIN_PRECISION
        and recall >= MIN_RECALL
        and pr_f1 >= MIN_PR_F1
        and _metric(memorization, "exact_copy_fraction") <= MAX_EXACT_COPY_FRACTION
        and _metric(memorization, "generated_duplicate_fraction")
        <= MAX_GENERATED_DUPLICATE_FRACTION
    )


def _state_dict_exact(left: object, right: object) -> bool:
    import torch

    if (
        not isinstance(left, dict)
        or not isinstance(right, dict)
        or set(left) != set(right)
    ):
        return False
    return all(
        isinstance(left[name], torch.Tensor)
        and isinstance(right[name], torch.Tensor)
        and left[name].dtype == right[name].dtype
        and left[name].shape == right[name].shape
        and torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
        for name in left
    )


def _evaluation_checks(
    evaluation: dict,
    *,
    evaluation_path: Path,
    result: dict,
    result_path: Path,
    step: int,
) -> dict[str, bool]:
    checkpoint = result.get("checkpoints", {}).get(str(step), {}).get("ema", {})
    retained = revalidate_clean_evaluation_evidence(
        evaluation, anchor=evaluation_path.parent
    )
    return {
        "status_and_scope": (
            evaluation.get("status") == "cap-emf2-development-evaluation"
            and evaluation.get("development_only") is True
            and evaluation.get("arm") == ARM
            and int(evaluation.get("step", -1)) == step
        ),
        "unit_binding": (
            evaluation.get("unit", {}).get("sha256") == result["artifact_sha256"]
            and evaluation.get("unit", {}).get("preflight_sha256")
            == result.get("preflight_sha256")
            and evaluation.get("checkpoint", {}).get("sha256")
            == checkpoint.get("sha256")
            and evaluation.get("checkpoint", {}).get("kind") == "ema"
            and int(evaluation.get("checkpoint", {}).get("step", -1)) == step
        ),
        "schema": _development_evaluation_schema(evaluation),
        "retained_population_recomputed": retained.get("valid") is True,
        "uncurated_grid": _grid_integrity(evaluation, anchor=evaluation_path.parent),
        "recorded_ema_reloads": verify_final_ema(result_path, step=step).step == step,
    }


def build_gate(
    *,
    preflight_path: Path,
    result_path: Path,
    recovery_path: Path,
    raw_readmission_path: Path,
    evaluation_650k_path: Path,
    evaluation_750k_path: Path,
    visual_review_path: Path,
    mirror_root: Path,
    out: Path,
) -> dict:
    preflight = load_preflight(preflight_path)
    result = verify_json(result_path, "cap-emf2-screen-unit")
    readmission = verify_json(raw_readmission_path, "cap-emf2-numerical-admission")
    evaluation_650 = verify_json(
        evaluation_650k_path, "cap-emf2-development-evaluation"
    )
    evaluation_750 = verify_json(
        evaluation_750k_path, "cap-emf2-development-evaluation"
    )
    visual = verify_json(visual_review_path, "cap-emf2-foundation-visual-review")

    candidate = preflight["candidate"]
    declared = profile_payload(
        apply_calibrated_gate(
            screen_profile(ARM, candidate, updates=FINAL_STEP),
            preflight["inputs"]["gate_calibration"],
        )
    )
    benchmark = preflight["inputs"]["benchmark"]
    realized = profile_payload(
        apply_calibrated_gate(
            screen_profile(ARM, candidate, updates=FINAL_STEP),
            preflight["inputs"]["gate_calibration"],
        )
    )
    realized["train"]["micro_batch"] = int(benchmark["micro_batch"])
    realized["train"]["accumulation_steps"] = int(benchmark["accumulation_steps"])
    realized["train"]["effective_batch"] = int(benchmark["effective_batch"])
    checkpoints = result.get("checkpoints", {})
    expected_steps = {str(step) for step in declared["train"]["checkpoint_updates"]}
    raw_record = checkpoints.get(str(FINAL_STEP), {}).get("raw", {})
    ema_record = checkpoints.get(str(FINAL_STEP), {}).get("ema", {})
    raw_path = (
        Path(raw_record.get("path", ""))
        if Path(raw_record.get("path", "")).is_absolute()
        else (result_path.parent / raw_record.get("path", "")).resolve()
    )
    ema_path = (
        Path(ema_record.get("path", ""))
        if Path(ema_record.get("path", "")).is_absolute()
        else (result_path.parent / ema_record.get("path", "")).resolve()
    )
    raw_checkpoint = load_checkpoint(
        raw_path,
        expected_sha=raw_record.get("sha256"),
        step=FINAL_STEP,
        kind="raw",
        arm=ARM,
        declared_profile=declared,
        realized_profile=realized,
        preflight_sha256=preflight["artifact_sha256"],
        run_identity_sha256=result.get("run_identity_sha256"),
        unit_seed=int(result.get("unit_seed", -1)),
    )
    ema_checkpoint = load_checkpoint(
        ema_path,
        expected_sha=ema_record.get("sha256"),
        step=FINAL_STEP,
        kind="ema",
        arm=ARM,
        declared_profile=declared,
        realized_profile=realized,
        preflight_sha256=preflight["artifact_sha256"],
        run_identity_sha256=result.get("run_identity_sha256"),
        unit_seed=int(result.get("unit_seed", -1)),
    )
    recovery, recovery_sha = load_recovery_payload(
        recovery_path, require_sidecar=True, validate_counters=True
    )
    validate_recovery_counters(recovery, strict=True)
    ema_recovery = recovery.get("ema", {})
    merged_ema = {
        **ema_recovery.get("shadow", {}),
        **ema_recovery.get("buffers", {}),
    }
    mirror = DurableMirror(result_path.parent, mirror_root)
    ladder_mirrored = True
    for record in checkpoints.values() if isinstance(checkpoints, dict) else ():
        for kind in ("raw", "ema"):
            artifact = record.get(kind, {}) if isinstance(record, dict) else {}
            value = artifact.get("path")
            if not isinstance(value, str):
                ladder_mirrored = False
                continue
            path = Path(value)
            path = path if path.is_absolute() else (result_path.parent / path).resolve()
            try:
                ladder_mirrored = ladder_mirrored and mirror.verify(
                    path
                ) == artifact.get("durable_mirror")
            except (OSError, RuntimeError, ValueError):
                ladder_mirrored = False

    eval_650_checks = _evaluation_checks(
        evaluation_650,
        evaluation_path=evaluation_650k_path,
        result=result,
        result_path=result_path,
        step=COMPARISON_STEP,
    )
    eval_750_checks = _evaluation_checks(
        evaluation_750,
        evaluation_path=evaluation_750k_path,
        result=result,
        result_path=result_path,
        step=FINAL_STEP,
    )
    baseline = preflight["inputs"]["baseline_standard"]["metrics"]
    metric_calibration = preflight["inputs"]["metric_calibration"]
    metrics_650 = evaluation_650["standard_train_reference_metrics"]
    metrics_750 = evaluation_750["standard_train_reference_metrics"]
    margins = {
        key: _calibration_tolerance(metric_calibration, key)
        for key in (FID_KEY, KID_KEY)
    }
    quality_checks = {
        "650k_beats_historical_fid_beyond_margin": (
            _metric(metrics_650, FID_KEY) + margins[FID_KEY]
            < _metric(baseline, FID_KEY)
        ),
        "650k_beats_historical_kid_beyond_margin": (
            _metric(metrics_650, KID_KEY) + margins[KID_KEY]
            < _metric(baseline, KID_KEY)
        ),
        "750k_retains_650k_fid_within_margin": (
            _metric(metrics_750, FID_KEY)
            <= _metric(metrics_650, FID_KEY) + margins[FID_KEY]
        ),
        "750k_retains_650k_kid_within_margin": (
            _metric(metrics_750, KID_KEY)
            <= _metric(metrics_650, KID_KEY) + margins[KID_KEY]
        ),
        "750k_auxiliary_noncollapse": _auxiliary_noncollapse(evaluation_750),
    }
    checks = {
        "foundation_campaign": preflight.get("budget", {}).get("campaign") == CAMPAIGN,
        "result_identity": (
            result.get("arm") == ARM
            and result.get("numerical_candidate") == candidate
            and result.get("preflight_sha256") == preflight["artifact_sha256"]
            and result.get("declared_profile") == declared
            and result.get("realized_profile") == realized
        ),
        "training_complete": (
            int(result.get("training", {}).get("optimizer_updates", -1)) == FINAL_STEP
            and int(result.get("training", {}).get("inference_forward_calls", -1)) == 1
            and result.get("train_only_gate", {}).get("verdict") == "PASS"
        ),
        "complete_checkpoint_ladder": (
            isinstance(checkpoints, dict)
            and set(checkpoints) == expected_steps
            and all(
                isinstance(record, dict) and set(record) == {"raw", "ema"}
                for record in checkpoints.values()
            )
        ),
        "checkpoint_ladder_durable": ladder_mirrored,
        "terminal_recovery": (
            int(recovery.get("planned_updates", -1)) == FINAL_STEP
            and int(recovery.get("completed_updates", -1)) == FINAL_STEP
            and result.get("recovery", {}).get("sha256") == recovery_sha
            and result.get("recovery", {}).get("continuation_authorization") is None
        ),
        "raw_recovery_matches_checkpoint": _state_dict_exact(
            recovery.get("model"), raw_checkpoint.get("state_dict")
        ),
        "ema_recovery_matches_checkpoint": _state_dict_exact(
            merged_ema, ema_checkpoint.get("state_dict")
        ),
        "checkpoint_previews": _checkpoint_previews_valid(
            result, anchor=result_path.parent
        ),
        "raw_numerical_readmission": (
            readmission.get("decision") == "GO"
            and admission_matrix_complete(readmission)
            and readmission.get("checkpoint_sha256") == raw_record.get("sha256")
            and int(readmission.get("checkpoint_step", -1)) == FINAL_STEP
            and readmission.get("candidate", {}).get("name") == candidate
        ),
        "evaluation_650k": all(eval_650_checks.values()),
        "evaluation_750k": all(eval_750_checks.values()),
        "evaluation_environment_matched": _evaluation_environment(evaluation_650)
        == _evaluation_environment(evaluation_750),
        "evaluation_reference_matched": _reference_binding(evaluation_650)
        == _reference_binding(evaluation_750),
        **quality_checks,
        "fixed_grid_human_review": (
            visual.get("decision") == "PASS"
            and visual.get("evaluation", {}).get("sha256")
            == evaluation_750["artifact_sha256"]
            and visual.get("grid", {}).get("sha256")
            == evaluation_750.get("uncurated_grid", {}).get("sha256")
        ),
    }
    failed = sorted(name for name, passed in checks.items() if passed is not True)
    decision = "GO" if not failed else "NO_GO"
    return {
        "status": STATUS,
        "decision": decision,
        "failed": failed,
        "checks": checks,
        "quality": {
            "historical_baseline": {
                key: _metric(baseline, key) for key in (FID_KEY, KID_KEY)
            },
            "checkpoint_650k": {
                key: _metric(metrics_650, key) for key in (FID_KEY, KID_KEY)
            },
            "checkpoint_750k": {
                key: _metric(metrics_750, key) for key in (FID_KEY, KID_KEY)
            },
            "absolute_real_real_margins": margins,
        },
        "foundation": {
            "arm": ARM,
            "step": FINAL_STEP,
            "ema_checkpoint": _reference(ema_path, out.parent),
            "ema_checkpoint_sha256": ema_record.get("sha256"),
            "raw_checkpoint_sha256": raw_record.get("sha256"),
            "recovery": _reference(recovery_path, out.parent),
            "recovery_sha256": recovery_sha,
            "result_sha256": result["artifact_sha256"],
            "preflight_sha256": preflight["artifact_sha256"],
        },
        "inputs": {
            "preflight": _reference(preflight_path, out.parent),
            "result": _reference(result_path, out.parent),
            "readmission": _reference(raw_readmission_path, out.parent),
            "evaluation_650k": _reference(evaluation_650k_path, out.parent),
            "evaluation_750k": _reference(evaluation_750k_path, out.parent),
            "visual_review": _reference(visual_review_path, out.parent),
            "mirror_root": str(mirror_root.resolve()),
        },
        "scope": (
            "train-reference capability gate for one all-class CIFAR-10 foundation; "
            "authorizes feature qualification only, never ASFD training or a claim"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--raw-readmission", type=Path, required=True)
    parser.add_argument("--evaluation-650k", type=Path, required=True)
    parser.add_argument("--evaluation-750k", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--mirror-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_unused(args.out)
    result = build_gate(
        preflight_path=args.preflight,
        result_path=args.result,
        recovery_path=args.recovery,
        raw_readmission_path=args.raw_readmission,
        evaluation_650k_path=args.evaluation_650k,
        evaluation_750k_path=args.evaluation_750k,
        visual_review_path=args.visual_review,
        mirror_root=args.mirror_root,
        out=args.out,
    )
    digest = write_json_atomic(args.out, result)
    print(f"wrote {args.out} sha256={digest} decision={result['decision']}")
    if result["failed"]:
        print("failed checks: " + ", ".join(result["failed"]))
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
