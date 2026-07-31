"""Analyze the frozen projection/kernel confirmation without retuning it."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_ARM = "cta-exact-hybrid"
PAPER_ARM = "paper-neural-optimized"
BOOTSTRAP_SEED = 2026081709
BOOTSTRAP_RESAMPLES = 20_000


def load_artifact(
    directory: Path,
    *,
    expected_representatives: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not audit.get("deep_metrics"):
        raise RuntimeError(f"{directory} has no passing deep audit")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader((directory / "rows.csv").open(encoding="utf-8")))
    by_target: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if raw["arm"] not in (PRIMARY_ARM, PAPER_ARM):
            continue
        target = raw["target"]
        by_target.setdefault(target, {})[raw["arm"]] = {
            key: coerce(value) for key, value in raw.items()
        }
    if not by_target or any(
        set(pair) != {PRIMARY_ARM, PAPER_ARM} for pair in by_target.values()
    ):
        raise RuntimeError(f"{directory} lacks a complete paired primary/paper table")
    primary_rows = [pair[PRIMARY_ARM] for pair in by_target.values()]
    if any(int(row["registered_active_direction_count"]) != 32 for row in primary_rows):
        raise RuntimeError(f"{directory} was not run with active-directions=32")
    if any(int(row["registered_local_field_calls"]) != -1 for row in primary_rows):
        raise RuntimeError(f"{directory} did not retain every local-field call")
    if any(
        int(row["registered_local_representative_count"]) != expected_representatives
        for row in primary_rows
    ):
        raise RuntimeError(f"{directory} has the wrong representative setting")
    return by_target, manifest


def coerce(value: str) -> Any:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and all(mark not in value.lower() for mark in (".", "e")):
        return int(number)
    return number


def ratio_summary(
    paired: dict[str, dict[str, Any]],
    metric: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    names = sorted(paired)
    ratios = np.asarray(
        [
            float(paired[name][PRIMARY_ARM][metric])
            / float(paired[name][PAPER_ARM][metric])
            for name in names
        ]
    )
    if not np.all(np.isfinite(ratios)) or np.any(ratios <= 0.0):
        raise RuntimeError(f"{metric} ratios must be finite and positive")
    log_ratios = np.log(ratios)
    samples = rng.integers(0, len(ratios), size=(BOOTSTRAP_RESAMPLES, len(ratios)))
    bootstrap = np.exp(np.mean(log_ratios[samples], axis=1))
    interval = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "geometric_mean_ratio": float(np.exp(np.mean(log_ratios))),
        "bootstrap_95_interval": [float(interval[0]), float(interval[1])],
        "paired_median_ratio": float(np.median(ratios)),
        "wins": int(np.sum(ratios < 1.0)),
        "ties": int(np.sum(ratios == 1.0)),
        "targets": len(ratios),
    }


def cost_summary(paired: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in (
        "online_training_wall_seconds",
        "setup_plus_training_wall_seconds",
        "paper_kernel_pairs",
        "target_example_accesses",
        "generator_example_evals_training",
    ):
        ratios = [
            float(pair[PRIMARY_ARM][metric]) / float(pair[PAPER_ARM][metric])
            for pair in paired.values()
        ]
        result[f"median_{metric}_ratio"] = float(np.median(ratios))
    return result


def support_summary(paired: dict[str, dict[str, Any]]) -> dict[str, Any]:
    all_primary = [pair[PRIMARY_ARM] for pair in paired.values()]
    all_paper = [pair[PAPER_ARM] for pair in paired.values()]
    primary_coverage = [
        row["mode_coverage"]
        for row in all_primary
        if math.isfinite(float(row["mode_coverage"]))
    ]
    paper_coverage = [
        row["mode_coverage"]
        for row in all_paper
        if math.isfinite(float(row["mode_coverage"]))
    ]
    rare_primary = [row for row in all_primary if row["family"] == "rare-gmm"]
    rare_paper = [row for row in all_paper if row["family"] == "rare-gmm"]
    return {
        "primary_minimum_mode_coverage": float(min(primary_coverage)),
        "paper_minimum_mode_coverage": float(min(paper_coverage)),
        "primary_rare_median_mass_error": float(
            np.median([row["rare_mass_error"] for row in rare_primary])
        ),
        "paper_rare_median_mass_error": float(
            np.median([row["rare_mass_error"] for row in rare_paper])
        ),
        "primary_rare_minimum_mode_coverage": float(
            min(row["mode_coverage"] for row in rare_primary)
        ),
        "paper_rare_minimum_mode_coverage": float(
            min(row["mode_coverage"] for row in rare_paper)
        ),
    }


def field_audit_summary(paired: dict[str, dict[str, Any]]) -> dict[str, float]:
    rows = [pair[PRIMARY_ARM] for pair in paired.values()]
    return {
        "median_field_relative_l2_error": float(
            np.median(
                [row["mean_representative_field_relative_l2_error"] for row in rows]
            )
        ),
        "minimum_field_cosine": float(
            min(row["minimum_representative_field_cosine"] for row in rows)
        ),
        "median_row_mass_relative_l2_error": float(
            np.median(
                [row["mean_representative_row_mass_relative_l2_error"] for row in rows]
            )
        ),
        "maximum_mean_row_mass_relative_l2_error": float(
            max(row["mean_representative_row_mass_relative_l2_error"] for row in rows)
        ),
        "median_column_mass_relative_l2_error": float(
            np.median(
                [
                    row["mean_representative_column_mass_relative_l2_error"]
                    for row in rows
                ]
            )
        ),
        "maximum_mean_column_mass_relative_l2_error": float(
            max(
                row["mean_representative_column_mass_relative_l2_error"] for row in rows
            )
        ),
    }


def paired_arm_ratio(
    numerator: dict[str, dict[str, Any]],
    denominator: dict[str, dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    names = sorted(set(numerator) & set(denominator))
    ratios = np.asarray(
        [
            float(numerator[name][PRIMARY_ARM][metric])
            / float(denominator[name][PRIMARY_ARM][metric])
            for name in names
        ]
    )
    return {
        "geometric_mean_ratio": float(np.exp(np.mean(np.log(ratios)))),
        "paired_median_ratio": float(np.median(ratios)),
        "wins": int(np.sum(ratios < 1.0)),
        "targets": len(ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--aggressive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary, primary_manifest = load_artifact(
        args.primary, expected_representatives=128
    )
    dense, dense_manifest = load_artifact(args.dense, expected_representatives=0)
    aggressive, aggressive_manifest = load_artifact(
        args.aggressive, expected_representatives=64
    )
    hashes = {
        manifest["registry_sha256"]
        for manifest in (primary_manifest, dense_manifest, aggressive_manifest)
    }
    if len(hashes) != 1:
        raise RuntimeError("confirmation artifacts do not share one frozen registry")
    if not (set(primary) == set(dense) == set(aggressive)):
        raise RuntimeError("confirmation artifacts do not share the same targets")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    quality = {
        metric: ratio_summary(primary, metric, rng) for metric in ("ed2", "heldout_sw1")
    }
    success = all(
        result["geometric_mean_ratio"] < 1.0
        and result["bootstrap_95_interval"][1] < 1.0
        and result["wins"] >= 12
        for result in quality.values()
    )
    result = {
        "schema": "projection-kernel-confirmation-analysis-v1",
        "status": "pass",
        "registry_sha256": next(iter(hashes)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "predeclared_primary_success": success,
        "primary_vs_matched_paper": {
            "quality": quality,
            "cost": cost_summary(primary),
            "support": support_summary(primary),
            "field_audit": field_audit_summary(primary),
        },
        "primary_m128_vs_dense_m512": {
            metric: paired_arm_ratio(primary, dense, metric)
            for metric in (
                "ed2",
                "heldout_sw1",
                "online_training_wall_seconds",
                "setup_plus_training_wall_seconds",
                "paper_kernel_pairs",
            )
        },
        "aggressive_m64_vs_dense_m512": {
            metric: paired_arm_ratio(aggressive, dense, metric)
            for metric in (
                "ed2",
                "heldout_sw1",
                "online_training_wall_seconds",
                "setup_plus_training_wall_seconds",
                "paper_kernel_pairs",
            )
        },
        "artifacts": {
            "primary_m128": str(args.primary),
            "dense_m512": str(args.dense),
            "aggressive_m64": str(args.aggressive),
        },
        "limitations": [
            "synthetic dimensions 2, 4, 8, and 16 only",
            "CPU wall time is implementation- and hardware-specific",
            "peak memory was not measured",
            "the balanced tree has no semantic rare-mode oracle",
        ],
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
