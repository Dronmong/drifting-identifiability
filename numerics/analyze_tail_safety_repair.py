"""Analyze the consumed-registry projected-tail/safety factorial screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


BASELINE = "cta-exact-adaptive-rollout"
ARMS = (
    BASELINE,
    "cta-exact-adaptive-rollout-safe",
    "cta-exact-adaptive-rollout-rank-balanced",
    "cta-exact-adaptive-rollout-safe-rank-balanced",
)
QUALITY_METRICS = ("ed2", "heldout_sw1", "training_quantile_rmse")
COST_METRICS = (
    "wall_seconds",
    "projection_scalar_products",
    "sort_work",
    "paper_kernel_pairs",
    "generator_forward_calls_training",
)


def coerce(value: str) -> Any:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and all(mark not in value.lower() for mark in (".", "e")):
        return int(number)
    return number


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.exp(np.mean(np.log(array))))


def grouped(
    ratios: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = {}
    for target, ratio in ratios.items():
        groups.setdefault(str(metadata[target][BASELINE][field]), []).append(ratio)
    return {
        key: {
            "geometric_mean_ratio": geometric_mean(values),
            "wins": int(np.sum(np.asarray(values) < 1.0)),
            "targets": len(values),
        }
        for key, values in sorted(groups.items())
    }


def ratio_result(
    rows: dict[str, dict[str, Any]], arm: str, metric: str
) -> dict[str, Any]:
    ratios = {
        target: float(pair[arm][metric]) / float(pair[BASELINE][metric])
        for target, pair in rows.items()
    }
    values = list(ratios.values())
    return {
        "geometric_mean_ratio": geometric_mean(values),
        "median_ratio": float(np.median(values)),
        "wins": int(np.sum(np.asarray(values) < 1.0)),
        "ties": int(
            np.sum(np.isclose(np.asarray(values), 1.0, rtol=0.0, atol=1e-14))
        ),
        "targets": len(values),
        "by_dimension": grouped(ratios, rows, "dimension"),
        "by_family": grouped(ratios, rows, "family"),
        "per_target": ratios,
    }


def rare_result(
    rows: dict[str, dict[str, Any]], arm: str
) -> dict[str, Any]:
    rare = [
        pair for pair in rows.values() if pair[BASELINE]["family"] == "rare-gmm"
    ]
    metrics = (
        "rare_core_mass",
        "rare_mass_error",
        "maximum_teacher_rare_core_count",
        "maximum_post_student_rare_core_count",
        "final_teacher_rare_core_count",
        "final_post_student_rare_core_count",
        "steps_retaining_teacher_rare_core",
        "mean_teacher_to_post_student_rare_core_ratio",
    )
    result = {}
    for metric in metrics:
        candidate = np.asarray([float(pair[arm][metric]) for pair in rare])
        baseline = np.asarray([float(pair[BASELINE][metric]) for pair in rare])
        result[metric] = {
            "arm_mean": float(np.mean(candidate)),
            "baseline_mean": float(np.mean(baseline)),
            "mean_difference": float(np.mean(candidate - baseline)),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("tail_safety_repair_analysis.json"),
    )
    args = parser.parse_args()
    audit = json.loads((args.artifact / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not audit.get("deep_metrics"):
        raise RuntimeError("artifact lacks a passing deep audit")
    manifest = json.loads(
        (args.artifact / "manifest.json").read_text(encoding="utf-8")
    )
    if tuple(manifest["registered_arms"]) != ARMS:
        raise RuntimeError("artifact has the wrong factorial arms")
    with (args.artifact / "rows.csv").open(newline="", encoding="utf-8") as stream:
        raw = [
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    rows: dict[str, dict[str, Any]] = {}
    for row in raw:
        rows.setdefault(str(row["target"]), {})[str(row["arm"])] = row
    if not rows or any(set(pair) != set(ARMS) for pair in rows.values()):
        raise RuntimeError("artifact lacks complete factorial cells")

    quality = {
        arm: {
            metric: ratio_result(rows, arm, metric)
            for metric in QUALITY_METRICS
        }
        for arm in ARMS[1:]
    }
    cost = {
        arm: {
            metric: ratio_result(rows, arm, metric)
            for metric in COST_METRICS
        }
        for arm in ARMS[1:]
    }
    mechanism = {}
    for arm in ARMS:
        arm_rows = [pair[arm] for pair in rows.values()]
        tail_losses = [
            float(row["mean_tail_student_loss"])
            for row in arm_rows
            if float(row["mean_tail_balanced_count"]) > 0
        ]
        bulk_losses = [
            float(row["mean_bulk_student_loss"])
            for row in arm_rows
            if float(row["mean_tail_balanced_count"]) > 0
        ]
        mechanism[arm] = {
            "mean_tail_student_loss": (
                float(np.mean(tail_losses)) if tail_losses else 0.0
            ),
            "mean_bulk_student_loss": (
                float(np.mean(bulk_losses)) if bulk_losses else 0.0
            ),
            "tail_to_bulk_student_loss_ratio": (
                float(np.mean(np.asarray(tail_losses) / np.asarray(bulk_losses)))
                if tail_losses
                else 0.0
            ),
            "mean_tail_balanced_count": float(
                np.mean(
                    [float(row["mean_tail_balanced_count"]) for row in arm_rows]
                )
            ),
            "rare": rare_result(rows, arm),
        }
    worth_confirmation = {
        arm: (
            quality[arm]["ed2"]["geometric_mean_ratio"] < 1.0
            and quality[arm]["heldout_sw1"]["geometric_mean_ratio"] < 1.0
            and mechanism[arm]["rare"]["rare_core_mass"]["mean_difference"] >= 0.0
        )
        for arm in ARMS[1:]
    }
    report = {
        "schema": "tail-safety-repair-development-v1",
        "status": "consumed-registry development evidence",
        "artifact": str(args.artifact),
        "artifact_audit_sha256": manifest["audit_sha256"],
        "target_count": len(rows),
        "baseline": BASELINE,
        "quality": quality,
        "cost": cost,
        "mechanism": mechanism,
        "worth_fresh_confirmation": worth_confirmation,
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    for arm in ARMS[1:]:
        print(arm)
        for metric in QUALITY_METRICS:
            item = quality[arm][metric]
            print(
                f"  {metric}: {item['geometric_mean_ratio']:.4f}, "
                f"wins={item['wins']}/{item['targets']}"
            )
        print(f"  worth confirmation: {worth_confirmation[arm]}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

