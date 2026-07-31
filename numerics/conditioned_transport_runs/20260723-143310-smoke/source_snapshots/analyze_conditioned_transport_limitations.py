"""Reproduce the post-hoc limitation audit from frozen transport artifacts.

This script is diagnostic only. It does not select or evaluate a new model.
Its support metrics were not preregistered in the original confirmations and
must therefore remain labelled exploratory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HYBRID_ARM = "cta-exact-hybrid"
GLOBAL_ARM = "cta-exact-global"
PAPER_ARM = "paper-neural-optimized"


def safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def coerce(value: str) -> Any:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and all(mark not in value.lower() for mark in (".", "e")):
        return int(number)
    return number


def load_rows(directory: Path) -> dict[tuple[str, str], dict[str, Any]]:
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not audit.get("deep_metrics"):
        raise RuntimeError(f"{directory} lacks a passing deep audit")
    with (directory / "rows.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{key: coerce(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    result = {(str(row["target"]), str(row["arm"])): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"{directory} contains duplicate target/arm rows")
    return result


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if len(array) < 1 or np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(array))))


def ratio_record(
    numerator: list[dict[str, Any]],
    denominator: list[dict[str, Any]],
    metric: str,
) -> dict[str, float | int]:
    denominator_by_target = {str(row["target"]): row for row in denominator}
    ratios = [
        float(row[metric]) / float(denominator_by_target[str(row["target"])][metric])
        for row in numerator
    ]
    return {
        "geometric_mean_ratio": geometric_mean(ratios),
        "median_ratio": float(np.median(ratios)),
        "wins": int(np.sum(np.asarray(ratios) < 1.0)),
        "targets": len(ratios),
    }


def grouped_ratios(
    numerator: list[dict[str, Any]],
    denominator: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name, field in (("by_dimension", "dimension"), ("by_family", "family")):
        groups: dict[str, Any] = {}
        values = sorted({str(row[field]) for row in numerator})
        for value in values:
            selected_num = [row for row in numerator if str(row[field]) == value]
            selected_den = [row for row in denominator if str(row[field]) == value]
            groups[value] = {
                metric: ratio_record(selected_num, selected_den, metric)
                for metric in ("ed2", "heldout_sw1")
            }
        result[group_name] = groups
    return result


def kth_nearest_distances(
    query: np.ndarray,
    support: np.ndarray,
    *,
    k: int,
    chunk_size: int = 256,
) -> np.ndarray:
    if not 1 <= k <= len(support):
        raise ValueError("nearest-neighbor k is outside the support")
    result = np.empty(len(query), dtype=float)
    support_norm = np.sum(support * support, axis=1)
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        block = query[start:stop]
        squared = (
            np.sum(block * block, axis=1)[:, None]
            + support_norm[None, :]
            - 2.0 * block @ support.T
        )
        np.maximum(squared, 0.0, out=squared)
        result[start:stop] = np.sqrt(np.partition(squared, k - 1, axis=1)[:, k - 1])
    return result


def exploratory_support_metrics(
    output: np.ndarray,
    reference: np.ndarray,
    *,
    k: int = 5,
    radius_quantile: float = 0.95,
) -> dict[str, float]:
    calibration = np.asarray(reference[::2], dtype=float)
    target_evaluation = np.asarray(reference[1::2], dtype=float)
    output = np.asarray(output, dtype=float)
    target_radii = kth_nearest_distances(target_evaluation, calibration, k=k)
    radius = float(np.quantile(target_radii, radius_quantile))
    output_to_target = kth_nearest_distances(output, calibration, k=1)
    target_to_output = kth_nearest_distances(target_evaluation, output, k=1)
    mean_scale = math.sqrt(
        max(float(np.mean(np.sum(target_evaluation * target_evaluation, axis=1))), 1e-15)
    )
    mean_error = float(
        np.linalg.norm(np.mean(output, axis=0) - np.mean(target_evaluation, axis=0))
        / mean_scale
    )
    output_covariance = np.cov(output, rowvar=False)
    target_covariance = np.cov(target_evaluation, rowvar=False)
    covariance_error = float(
        np.linalg.norm(output_covariance - target_covariance)
        / max(float(np.linalg.norm(target_covariance)), 1e-15)
    )
    precision = float(np.mean(output_to_target <= radius))
    recall = float(np.mean(target_to_output <= radius))
    return {
        "target_calibrated_radius": radius,
        "support_precision": precision,
        "support_recall": recall,
        "off_support_occupancy": 1.0 - precision,
        "normalized_mean_error": mean_error,
        "relative_covariance_frobenius_error": covariance_error,
    }


def summarize_support(
    directory: Path,
    rows: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    target_names = sorted({target for target, _ in rows})
    per_target: dict[str, Any] = {}
    with (
        np.load(directory / "outputs.npz") as outputs,
        np.load(directory / "references.npz") as references,
    ):
        for target in target_names:
            target_result: dict[str, Any] = {}
            reference = references[safe_key(target)]
            for arm in (HYBRID_ARM, GLOBAL_ARM, PAPER_ARM):
                key = safe_key(f"{arm}__{target}")
                if key in outputs.files:
                    target_result[arm] = exploratory_support_metrics(
                        outputs[key], reference
                    )
            per_target[target] = target_result

    aggregate: dict[str, Any] = {}
    for arm in (HYBRID_ARM, GLOBAL_ARM, PAPER_ARM):
        records = [
            arms[arm] for arms in per_target.values() if arm in arms
        ]
        aggregate[arm] = {
            metric: float(np.median([record[metric] for record in records]))
            for metric in (
                "support_precision",
                "support_recall",
                "off_support_occupancy",
                "normalized_mean_error",
                "relative_covariance_frobenius_error",
            )
        }
    return {
        "status": "exploratory_post_hoc",
        "definition": {
            "reference_split": "even indices calibrate; odd indices evaluate recall",
            "radius": "95th percentile target 5-NN distance to calibration split",
            "precision": "output fraction within target-calibrated radius",
            "recall": "target-evaluation fraction within radius of an output",
        },
        "aggregate_medians": aggregate,
        "per_target": per_target,
    }


def dimension_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in sorted({int(row["dimension"]) for row in rows}):
        selected = [row for row in rows if int(row["dimension"]) == dimension]
        result[str(dimension)] = {
            "targets": len(selected),
            "median_field_relative_l2_error": float(
                np.median(
                    [
                        row["mean_representative_field_relative_l2_error"]
                        for row in selected
                    ]
                )
            ),
            "minimum_field_cosine": float(
                min(row["minimum_representative_field_cosine"] for row in selected)
            ),
            "median_row_mass_relative_l2_error": float(
                np.median(
                    [
                        row["mean_representative_row_mass_relative_l2_error"]
                        for row in selected
                    ]
                )
            ),
            "median_column_mass_relative_l2_error": float(
                np.median(
                    [
                        row["mean_representative_column_mass_relative_l2_error"]
                        for row in selected
                    ]
                )
            ),
            "median_positive_rms_radius": float(
                np.median(
                    [row["mean_positive_representative_rms_radius"] for row in selected]
                )
            ),
            "maximum_positive_radius": float(
                max(row["maximum_positive_representative_radius"] for row in selected)
            ),
            "median_negative_rms_radius": float(
                np.median(
                    [row["mean_negative_representative_rms_radius"] for row in selected]
                )
            ),
            "maximum_negative_radius": float(
                max(row["maximum_negative_representative_radius"] for row in selected)
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    primary = load_rows(args.primary)
    dense = load_rows(args.dense)
    targets = sorted({target for target, _ in primary})
    if set(targets) != {target for target, _ in dense}:
        raise RuntimeError("primary and dense artifacts do not share targets")

    def arm_rows(
        rows: dict[tuple[str, str], dict[str, Any]], arm: str
    ) -> list[dict[str, Any]]:
        if any((target, arm) not in rows for target in targets):
            raise RuntimeError(f"artifact lacks arm {arm}")
        return [rows[(target, arm)] for target in targets]

    primary_hybrid = arm_rows(primary, HYBRID_ARM)
    primary_global = arm_rows(primary, GLOBAL_ARM)
    dense_hybrid = arm_rows(dense, HYBRID_ARM)
    result = {
        "schema": "conditioned-transport-limitation-audit-v1",
        "status": "post_hoc_diagnostic",
        "artifacts": {
            "primary": str(args.primary),
            "dense": str(args.dense),
            "primary_rows_sha256": sha256_file(args.primary / "rows.csv"),
            "primary_outputs_sha256": sha256_file(args.primary / "outputs.npz"),
            "dense_rows_sha256": sha256_file(args.dense / "rows.csv"),
        },
        "fixed_local_hybrid_vs_global": grouped_ratios(
            primary_hybrid, primary_global
        ),
        "m128_vs_dense": grouped_ratios(primary_hybrid, dense_hybrid),
        "m128_field_diagnostics_by_dimension": dimension_diagnostics(primary_hybrid),
        "exploratory_support_metrics": summarize_support(args.primary, primary),
        "interpretation_guardrails": [
            "all outputs were consumed before this diagnostic was specified",
            "support metrics are explanatory and not original registered endpoints",
            "the audit may design a new experiment but cannot confirm a repair",
        ],
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
