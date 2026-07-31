"""Compare isolated representative strategies on one consumed registry.

The output is development evidence only. It ranks approximation mechanisms but
does not promote a new primary or consume a fresh confirmation registry.
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

from analyze_conditioned_transport_limitations import (
    exploratory_support_metrics,
    safe_key,
    sha256_file,
)


ARM = "cta-exact-hybrid"
MECHANISM_METRICS = (
    "mean_representative_field_relative_l2_error",
    "minimum_representative_field_cosine",
    "mean_representative_row_mass_relative_l2_error",
    "mean_representative_column_mass_relative_l2_error",
    "mean_positive_representative_rms_radius",
    "maximum_positive_representative_radius",
    "mean_negative_representative_rms_radius",
    "maximum_negative_representative_radius",
)


def coerce(value: str) -> Any:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and all(mark not in value.lower() for mark in (".", "e")):
        return int(number)
    return number


def load_artifact(directory: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not audit.get("deep_metrics"):
        raise RuntimeError(f"{directory} lacks a passing deep audit")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    with (directory / "rows.csv").open(newline="", encoding="utf-8") as stream:
        raw_rows = list(csv.DictReader(stream))
    rows = {
        row["target"]: {key: coerce(value) for key, value in row.items()}
        for row in raw_rows
        if row["arm"] == ARM
    }
    if not rows or len(rows) != len(raw_rows):
        raise RuntimeError(
            f"{directory} must be an isolated one-arm {ARM} artifact"
        )
    return rows, manifest


def geometric_mean(values: list[float]) -> float:
    values_array = np.asarray(values, dtype=float)
    if np.any(values_array <= 0.0) or not np.all(np.isfinite(values_array)):
        raise ValueError("geometric mean requires positive finite values")
    return float(np.exp(np.mean(np.log(values_array))))


def ratio_summary(
    numerator: dict[str, dict[str, Any]],
    denominator: dict[str, dict[str, Any]],
    metric: str,
    names: list[str],
) -> dict[str, float | int]:
    ratios = np.asarray(
        [
            float(numerator[name][metric]) / float(denominator[name][metric])
            for name in names
        ]
    )
    return {
        "geometric_mean_ratio": geometric_mean(list(ratios)),
        "median_ratio": float(np.median(ratios)),
        "wins": int(np.sum(ratios < 1.0)),
        "targets": len(ratios),
    }


def grouped_summary(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name, field in (("by_dimension", "dimension"), ("by_family", "family")):
        groups: dict[str, Any] = {}
        for value in sorted({str(row[field]) for row in rows.values()}):
            selected = [row for row in rows.values() if str(row[field]) == value]
            record: dict[str, Any] = {
                "targets": len(selected),
                "median_ed2": float(np.median([row["ed2"] for row in selected])),
                "median_heldout_sw1": float(
                    np.median([row["heldout_sw1"] for row in selected])
                ),
                "minimum_mode_coverage": (
                    float(
                        min(
                            row["mode_coverage"]
                            for row in selected
                            if math.isfinite(float(row["mode_coverage"]))
                        )
                    )
                    if any(
                        math.isfinite(float(row["mode_coverage"])) for row in selected
                    )
                    else None
                ),
            }
            for metric in MECHANISM_METRICS:
                values = [float(row[metric]) for row in selected]
                aggregator = min if metric == "minimum_representative_field_cosine" else np.median
                record[metric] = float(aggregator(values))
            groups[value] = record
        result[group_name] = groups
    return result


def support_summary(
    directory: Path,
    rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    metrics: list[dict[str, float]] = []
    with (
        np.load(directory / "outputs.npz") as outputs,
        np.load(directory / "references.npz") as references,
    ):
        for target in sorted(rows):
            metrics.append(
                exploratory_support_metrics(
                    outputs[safe_key(f"{ARM}__{target}")],
                    references[safe_key(target)],
                )
            )
    return {
        metric: float(np.median([row[metric] for row in metrics]))
        for metric in (
            "support_precision",
            "support_recall",
            "off_support_occupancy",
            "normalized_mean_error",
            "relative_covariance_frobenius_error",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="label=artifact-directory; repeat for every strategy",
    )
    parser.add_argument("--fixed-label", default="fixed-m128")
    parser.add_argument("--dense-label", default="dense-m512")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifacts: dict[str, Path] = {}
    for item in args.artifact:
        label, separator, raw_path = item.partition("=")
        if not separator or not label or label in artifacts:
            parser.error("every --artifact must be one unique label=path")
        artifacts[label] = Path(raw_path)
    if args.fixed_label not in artifacts or args.dense_label not in artifacts:
        parser.error("fixed and dense labels must both be supplied")

    loaded = {label: load_artifact(path) for label, path in artifacts.items()}
    registry_hashes = {
        manifest["registry_sha256"] for _, manifest in loaded.values()
    }
    if len(registry_hashes) != 1:
        raise RuntimeError("strategy artifacts do not share one registry")
    target_sets = {tuple(sorted(rows)) for rows, _ in loaded.values()}
    if len(target_sets) != 1:
        raise RuntimeError("strategy artifacts do not share one target set")
    names = list(next(iter(target_sets)))
    fixed = loaded[args.fixed_label][0]
    dense = loaded[args.dense_label][0]

    configurations: dict[str, Any] = {}
    for label, directory in artifacts.items():
        rows, _ = loaded[label]
        first = rows[names[0]]
        representative_counts = sorted(
            {
                int(row["registered_local_representative_count"])
                for row in rows.values()
            }
        )
        high_dimensions = [
            name for name in names if int(rows[name]["dimension"]) in (8, 16)
        ]
        field_errors = [
            float(row["mean_representative_field_relative_l2_error"])
            for row in rows.values()
            if float(row["mean_representative_field_relative_l2_error"]) > 0.0
        ]
        configurations[label] = {
            "artifact": str(directory),
            "artifact_rows_sha256": sha256_file(directory / "rows.csv"),
            "strategy": first.get("registered_representative_strategy", "legacy"),
            "representative_count": (
                representative_counts[0]
                if len(representative_counts) == 1
                else representative_counts
            ),
            "tail_reserve_fraction": float(
                first.get("registered_representative_tail_reserve_fraction", 0.0)
            ),
            "grouped": grouped_summary(rows),
            "exploratory_support_medians": support_summary(directory, rows),
            "vs_fixed": {
                metric: ratio_summary(rows, fixed, metric, names)
                for metric in ("ed2", "heldout_sw1")
            },
            "vs_dense": {
                metric: ratio_summary(rows, dense, metric, names)
                for metric in ("ed2", "heldout_sw1")
            },
            "high_dimension_vs_fixed": {
                metric: ratio_summary(rows, fixed, metric, high_dimensions)
                for metric in ("ed2", "heldout_sw1")
            },
            "mechanism": {
                "geometric_mean_positive_field_error": (
                    geometric_mean(field_errors) if field_errors else 0.0
                ),
                "minimum_field_cosine": float(
                    min(
                        row["minimum_representative_field_cosine"]
                        for row in rows.values()
                    )
                ),
                "median_row_mass_relative_l2_error": float(
                    np.median(
                        [
                            row["mean_representative_row_mass_relative_l2_error"]
                            for row in rows.values()
                        ]
                    )
                ),
                "median_column_mass_relative_l2_error": float(
                    np.median(
                        [
                            row["mean_representative_column_mass_relative_l2_error"]
                            for row in rows.values()
                        ]
                    )
                ),
                "median_online_training_wall_seconds": float(
                    np.median(
                        [row["online_training_wall_seconds"] for row in rows.values()]
                    )
                ),
                "median_partition_projection_scalar_products": float(
                    np.median(
                        [
                            row[
                                "representative_partition_projection_scalar_products"
                            ]
                            for row in rows.values()
                        ]
                    )
                ),
            },
        }

    fixed_error = configurations[args.fixed_label]["mechanism"][
        "geometric_mean_positive_field_error"
    ]
    development_order = sorted(
        configurations,
        key=lambda label: (
            configurations[label]["mechanism"][
                "geometric_mean_positive_field_error"
            ]
            if configurations[label]["mechanism"][
                "geometric_mean_positive_field_error"
            ]
            > 0.0
            else math.inf,
            configurations[label]["vs_fixed"]["ed2"]["geometric_mean_ratio"],
        ),
    )
    result = {
        "schema": "representative-strategy-development-analysis-v1",
        "status": "consumed_registry_development_only",
        "registry_sha256": next(iter(registry_hashes)),
        "fixed_label": args.fixed_label,
        "dense_label": args.dense_label,
        "configurations": configurations,
        "mechanism_ranking": development_order,
        "mechanism_gate": {
            label: {
                "field_error_below_fixed": (
                    configurations[label]["mechanism"][
                        "geometric_mean_positive_field_error"
                    ]
                    < fixed_error
                ),
                "field_cosine_at_least_fixed": (
                    configurations[label]["mechanism"]["minimum_field_cosine"]
                    >= configurations[args.fixed_label]["mechanism"][
                        "minimum_field_cosine"
                    ]
                ),
            }
            for label in configurations
            if label != args.dense_label
        },
        "guardrails": [
            "strategy selection uses a consumed development registry",
            "support metrics are exploratory",
            "no configuration is confirmatory until frozen on a new registry",
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
