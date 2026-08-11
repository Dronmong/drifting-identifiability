"""Revalidate and compare the frozen 750k foundation and 800k ASFD evidence."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from ..stage_cap2.artifacts import assert_unused, verify_json, write_json_atomic
from ..stage_cap2.promotion import (
    FID_KEY,
    KID_KEY,
    MAX_EXACT_COPY_FRACTION,
    MAX_GENERATED_DUPLICATE_FRACTION,
    MIN_PR_F1,
    MIN_PRECISION,
    MIN_RECALL,
    _evaluation_environment,
    _grid_integrity,
    _reference_binding,
)
from ..stage_cap2.standard_metrics import revalidate_clean_evaluation_evidence
from .artifacts import source_manifest
from .evaluation import verify_final_ema

STATUS = "asfd-final-report"


def _portable(path: Path, anchor: Path) -> str:
    return Path(os.path.relpath(path.resolve(), anchor.resolve())).as_posix()


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value) if math.isfinite(float(value)) else math.nan


def _auxiliary_metrics(evaluation: dict) -> dict[str, float]:
    features = evaluation.get("repository_feature_metrics", {})
    memorization = evaluation.get("memorization", {})
    precision = _finite(features.get("precision"))
    recall = _finite(features.get("recall"))
    return {
        "precision": precision,
        "recall": recall,
        "pr_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "exact_copy_fraction": _finite(memorization.get("exact_copy_fraction")),
        "generated_duplicate_fraction": _finite(
            memorization.get("generated_duplicate_fraction")
        ),
    }


def _auxiliary_noncollapse(values: dict[str, float]) -> bool:
    return (
        values["precision"] >= MIN_PRECISION
        and values["recall"] >= MIN_RECALL
        and values["pr_f1"] >= MIN_PR_F1
        and values["exact_copy_fraction"] <= MAX_EXACT_COPY_FRACTION
        and values["generated_duplicate_fraction"] <= MAX_GENERATED_DUPLICATE_FRACTION
    )


def _metric_comparison(before: object, after: object, margin: object) -> dict:
    before_value = _finite(before)
    after_value = _finite(after)
    margin_value = _finite(margin)
    return {
        "foundation_750k": before_value,
        "asfd_800k": after_value,
        "delta_after_minus_before": after_value - before_value,
        "lower_is_better": True,
        "asfd_lower": after_value < before_value,
        "real_real_absolute_margin": margin_value,
        "noninferior_within_margin": after_value <= before_value + margin_value,
        "improved_beyond_margin": after_value + margin_value < before_value,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foundation-gate", type=Path, required=True)
    parser.add_argument("--continuation", type=Path, required=True)
    parser.add_argument("--asfd-evaluation", type=Path, required=True)
    parser.add_argument("--visual-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert_unused(args.out)
    foundation = verify_json(args.foundation_gate, "cap-emf2-750k-foundation-gate")
    asfd = verify_json(args.asfd_evaluation, "asfd-final-evaluation")
    visual = verify_json(args.visual_review, "asfd-final-visual-review")
    continuation = verify_final_ema(args.continuation)
    foundation_eval_path = Path(foundation["inputs"]["evaluation_750k"])
    if not foundation_eval_path.is_absolute():
        foundation_eval_path = args.foundation_gate.parent / foundation_eval_path
    baseline = verify_json(foundation_eval_path, "cap-emf2-development-evaluation")
    baseline_revalidation = revalidate_clean_evaluation_evidence(
        baseline, anchor=foundation_eval_path.parent
    )
    asfd_revalidation = revalidate_clean_evaluation_evidence(
        asfd, anchor=args.asfd_evaluation.parent
    )
    live_sources = source_manifest()
    checks = {
        "foundation_gate_go": foundation.get("decision") == "GO",
        "continuation_go": continuation.result.get("decision") == "GO",
        "evaluation_binds_continuation": asfd.get("continuation", {}).get("sha256")
        == continuation.result["artifact_sha256"],
        "foundation_evidence_recomputed": baseline_revalidation.get("valid") is True,
        "asfd_evidence_recomputed": asfd_revalidation.get("valid") is True,
        "asfd_evaluation_source_current": asfd.get("source_sha256") == live_sources,
        "visual_review_source_current": visual.get("source_sha256") == live_sources,
        "foundation_grid_intact": _grid_integrity(
            baseline, anchor=foundation_eval_path.parent
        ),
        "asfd_grid_intact": _grid_integrity(asfd, anchor=args.asfd_evaluation.parent),
        "metric_environment_matched": _evaluation_environment(baseline)
        == _evaluation_environment(asfd),
        "metric_reference_matched": _reference_binding(baseline)
        == _reference_binding(asfd),
        "one_call_inference": continuation.result.get("checks", {}).get(
            "one_call_inference"
        )
        is True,
        "visual_review_binds_fixed_evidence": (
            visual.get("foundation_evaluation", {}).get("sha256")
            == baseline["artifact_sha256"]
            and visual.get("asfd_evaluation", {}).get("sha256")
            == asfd["artifact_sha256"]
            and visual.get("foundation_grid", {}).get("sha256")
            == baseline.get("uncurated_grid", {}).get("sha256")
            and visual.get("asfd_grid", {}).get("sha256")
            == asfd.get("uncurated_grid", {}).get("sha256")
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    baseline_metrics = baseline["standard_train_reference_metrics"]
    asfd_metrics = asfd["standard_train_reference_metrics"]
    recorded_margins = foundation.get("quality", {}).get(
        "absolute_real_real_margins", {}
    )
    comparison = {}
    for key in (FID_KEY, KID_KEY):
        comparison[key] = _metric_comparison(
            baseline_metrics.get(key),
            asfd_metrics.get(key),
            recorded_margins.get(key),
        )
    baseline_auxiliary = _auxiliary_metrics(baseline)
    asfd_auxiliary = _auxiliary_metrics(asfd)
    quality_checks = {
        "fid_noninferior_within_real_real_margin": comparison[FID_KEY][
            "noninferior_within_margin"
        ],
        "kid_noninferior_within_real_real_margin": comparison[KID_KEY][
            "noninferior_within_margin"
        ],
        "at_least_one_primary_metric_improves_beyond_margin": (
            comparison[FID_KEY]["improved_beyond_margin"]
            or comparison[KID_KEY]["improved_beyond_margin"]
        ),
        "final_auxiliary_noncollapse": _auxiliary_noncollapse(asfd_auxiliary),
        "fixed_grid_visual_veto_passed": visual.get("decision") == "PASS",
    }
    quality_pass = not failed and all(quality_checks.values())
    result = {
        "status": STATUS,
        "integrity_decision": "PASS" if not failed else "FAIL",
        "failed": failed,
        "checks": checks,
        "foundation_gate": {
            "path": _portable(args.foundation_gate, args.out.parent),
            "sha256": foundation["artifact_sha256"],
        },
        "foundation_evaluation": {
            "path": _portable(foundation_eval_path, args.out.parent),
            "sha256": baseline["artifact_sha256"],
        },
        "continuation": {
            "path": _portable(args.continuation, args.out.parent),
            "sha256": continuation.result["artifact_sha256"],
        },
        "asfd_evaluation": {
            "path": _portable(args.asfd_evaluation, args.out.parent),
            "sha256": asfd["artifact_sha256"],
        },
        "visual_review": {
            "path": _portable(args.visual_review, args.out.parent),
            "sha256": visual["artifact_sha256"],
        },
        "comparison": comparison,
        "auxiliary_comparison": {
            "foundation_750k": baseline_auxiliary,
            "asfd_800k": asfd_auxiliary,
        },
        "quality_checks": quality_checks,
        "quality_decision": "PROMISING_IMPROVEMENT" if quality_pass else "NO_CLAIM",
        "scientific_interpretation": (
            "A promising improvement requires a change beyond the frozen real-real "
            "calibration margin on at least one primary metric, noninferiority on both, "
            "the auxiliary noncollapse/memorization vetoes, and review of both fixed "
            "uncurated grids. "
            "It does not attribute the change uniquely to ASFD because the concentrated-budget design omits a matched 50k raw-control continuation."
        ),
        "source_sha256": live_sources,
    }
    digest = write_json_atomic(args.out, result)
    print(f"wrote {args.out} sha256={digest} integrity={result['integrity_decision']}")
    return 0 if result["integrity_decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
