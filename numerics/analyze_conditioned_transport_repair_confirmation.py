"""Analyze the frozen representative-repair confirmation."""

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


CANDIDATE = "cta-exact-hybrid"
CONTROL = "cta-exact-fixed-control"
PAPER = "paper-neural-optimized"
ARMS = (CANDIDATE, CONTROL, PAPER)
BOOTSTRAP_SEED = 2026092307
BOOTSTRAP_RESAMPLES = 20_000


def coerce(value: str) -> Any:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and all(mark not in value.lower() for mark in (".", "e")):
        return int(number)
    return number


def load_artifact(
    directory: Path,
    initialization: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not audit.get("deep_metrics"):
        raise RuntimeError(f"{directory} lacks a passing deep audit")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["registered_arms"]) != ARMS:
        raise RuntimeError(f"{directory} has the wrong registered arms")
    with (directory / "rows.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{key: coerce(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["initialization"] != initialization:
            raise RuntimeError(f"{directory} has the wrong initialization")
        result.setdefault(str(row["target"]), {})[str(row["arm"])] = row
    if not result or any(set(pair) != set(ARMS) for pair in result.values()):
        raise RuntimeError(f"{directory} lacks complete target triplets")
    for pair in result.values():
        candidate = pair[CANDIDATE]
        control = pair[CONTROL]
        dimension = int(candidate["dimension"])
        expected_count = 128 if dimension < 8 else 256
        if (
            candidate["registered_representative_strategy"] != "variance-per-node"
            or int(candidate["registered_local_representative_count"])
            != expected_count
            or control["registered_representative_strategy"] != "fixed-level"
            or int(control["registered_local_representative_count"]) != 128
        ):
            raise RuntimeError("frozen representative configuration changed")
    return result, manifest


def target_reduced_ratios(
    artifacts: dict[str, dict[str, dict[str, Any]]],
    numerator: str,
    denominator: str,
    metric: str,
) -> dict[str, float]:
    names = sorted(next(iter(artifacts.values())))
    return {
        name: math.exp(
            np.mean(
                [
                    math.log(
                        float(rows[name][numerator][metric])
                        / float(rows[name][denominator][metric])
                    )
                    for rows in artifacts.values()
                ]
            )
        )
        for name in names
    }


def stratified_interval(
    ratios: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> list[float]:
    strata: dict[tuple[int, str], list[float]] = {}
    for name, value in ratios.items():
        row = metadata[name][CANDIDATE]
        strata.setdefault((int(row["dimension"]), str(row["family"])), []).append(
            math.log(value)
        )
    if len(strata) != 16 or any(len(values) != 2 for values in strata.values()):
        raise RuntimeError("bootstrap expected two targets in each of 16 strata")
    samples = []
    for values in strata.values():
        array = np.asarray(values)
        indices = rng.integers(
            0, len(array), size=(BOOTSTRAP_RESAMPLES, len(array))
        )
        samples.append(np.mean(array[indices], axis=1))
    bootstrap = np.exp(np.mean(np.stack(samples, axis=1), axis=1))
    interval = np.quantile(bootstrap, [0.025, 0.975])
    return [float(interval[0]), float(interval[1])]


def ratio_result(
    artifacts: dict[str, dict[str, dict[str, Any]]],
    metadata: dict[str, dict[str, Any]],
    numerator: str,
    denominator: str,
    metric: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ratios = target_reduced_ratios(artifacts, numerator, denominator, metric)
    values = np.asarray(list(ratios.values()))
    return {
        "geometric_mean_ratio": float(np.exp(np.mean(np.log(values)))),
        "bootstrap_95_interval": stratified_interval(ratios, metadata, rng),
        "median_ratio": float(np.median(values)),
        "wins": int(np.sum(values < 1.0)),
        "targets": len(values),
        "per_target": ratios,
    }


def mechanism_result(
    artifacts: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    high_cells: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for rows in artifacts.values():
        for pair in rows.values():
            if int(pair[CANDIDATE]["dimension"]) in (8, 16):
                high_cells.append((pair[CANDIDATE], pair[CONTROL]))
    field_ratios = [
        float(candidate["mean_representative_field_relative_l2_error"])
        / float(control["mean_representative_field_relative_l2_error"])
        for candidate, control in high_cells
    ]
    candidate_cosine = min(
        float(candidate["minimum_representative_field_cosine"])
        for candidate, _ in high_cells
    )
    control_cosine = min(
        float(control["minimum_representative_field_cosine"])
        for _, control in high_cells
    )

    def median_error(metric: str, arm_index: int) -> float:
        return float(np.median([float(pair[arm_index][metric]) for pair in high_cells]))

    candidate_row = median_error(
        "mean_representative_row_mass_relative_l2_error", 0
    )
    control_row = median_error(
        "mean_representative_row_mass_relative_l2_error", 1
    )
    candidate_column = median_error(
        "mean_representative_column_mass_relative_l2_error", 0
    )
    control_column = median_error(
        "mean_representative_column_mass_relative_l2_error", 1
    )
    checks = {
        "field_error_ratio_le_0_75": float(
            np.exp(np.mean(np.log(field_ratios)))
        )
        <= 0.75,
        "minimum_cosine_noninferior": candidate_cosine >= control_cosine - 0.01,
        "row_mass_error_noninferior": candidate_row <= 1.10 * control_row,
        "column_mass_error_noninferior": (
            candidate_column <= 1.10 * control_column
        ),
    }
    return {
        "high_dimension_geometric_field_error_ratio": float(
            np.exp(np.mean(np.log(field_ratios)))
        ),
        "candidate_minimum_field_cosine": candidate_cosine,
        "control_minimum_field_cosine": control_cosine,
        "candidate_median_row_mass_error": candidate_row,
        "control_median_row_mass_error": control_row,
        "candidate_median_column_mass_error": candidate_column,
        "control_median_column_mass_error": control_column,
        "checks": checks,
        "passed": all(checks.values()),
    }


def support_result(
    directories: dict[str, Path],
    artifacts: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    records: dict[str, list[dict[str, float]]] = {
        CANDIDATE: [],
        CONTROL: [],
    }
    coverages: dict[str, list[float]] = {CANDIDATE: [], CONTROL: []}
    rare_errors: dict[str, list[float]] = {CANDIDATE: [], CONTROL: []}
    for initialization, directory in directories.items():
        with (
            np.load(directory / "outputs.npz") as outputs,
            np.load(directory / "references.npz") as references,
        ):
            for target, pair in artifacts[initialization].items():
                reference = references[safe_key(target)]
                for arm in (CANDIDATE, CONTROL):
                    records[arm].append(
                        exploratory_support_metrics(
                            outputs[safe_key(f"{arm}__{target}")],
                            reference,
                        )
                    )
                    coverage = float(pair[arm]["mode_coverage"])
                    if math.isfinite(coverage):
                        coverages[arm].append(coverage)
                    rare_error = float(pair[arm]["rare_mass_error"])
                    if math.isfinite(rare_error):
                        rare_errors[arm].append(rare_error)

    medians = {
        arm: {
            metric: float(np.median([record[metric] for record in records[arm]]))
            for metric in (
                "support_precision",
                "support_recall",
                "off_support_occupancy",
                "normalized_mean_error",
                "relative_covariance_frobenius_error",
            )
        }
        for arm in (CANDIDATE, CONTROL)
    }
    candidate_rare = float(np.median(rare_errors[CANDIDATE]))
    control_rare = float(np.median(rare_errors[CONTROL]))
    checks = {
        "precision_noninferior": (
            medians[CANDIDATE]["support_precision"]
            >= medians[CONTROL]["support_precision"] - 0.02
        ),
        "recall_noninferior": (
            medians[CANDIDATE]["support_recall"]
            >= medians[CONTROL]["support_recall"] - 0.01
        ),
        "mode_coverage_noninferior": (
            min(coverages[CANDIDATE]) >= min(coverages[CONTROL])
        ),
        "rare_mass_error_noninferior": candidate_rare <= 1.20 * control_rare,
    }
    return {
        "status": "frozen_secondary_endpoint",
        "medians": medians,
        "candidate_minimum_mode_coverage": min(coverages[CANDIDATE]),
        "control_minimum_mode_coverage": min(coverages[CONTROL]),
        "candidate_median_rare_mass_error": candidate_rare,
        "control_median_rare_mass_error": control_rare,
        "checks": checks,
        "passed": all(checks.values()),
    }


def cost_result(
    artifacts: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for comparator in (CONTROL, PAPER):
        comparator_result: dict[str, float] = {}
        for metric in (
            "generator_example_evals_training",
            "paper_kernel_pairs",
            "target_example_accesses",
            "projection_scalar_products",
            "sort_work",
            "online_training_wall_seconds",
            "setup_plus_training_wall_seconds",
        ):
            candidate_values = [
                float(pair[CANDIDATE][metric])
                for rows in artifacts.values()
                for pair in rows.values()
            ]
            comparator_values = [
                float(pair[comparator][metric])
                for rows in artifacts.values()
                for pair in rows.values()
            ]
            comparator_result[f"median_candidate_{metric}"] = float(
                np.median(candidate_values)
            )
            comparator_result[f"median_comparator_{metric}"] = float(
                np.median(comparator_values)
            )
            comparator_result[f"median_{metric}_ratio"] = (
                float(
                    np.median(
                        [
                            candidate / denominator
                            for candidate, denominator in zip(
                                candidate_values,
                                comparator_values,
                                strict=True,
                            )
                        ]
                    )
                )
                if all(value > 0.0 for value in comparator_values)
                else None
            )
        result[f"candidate_vs_{comparator}"] = comparator_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concentrated", type=Path, required=True)
    parser.add_argument("--broad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = {"concentrated": args.concentrated, "broad": args.broad}
    loaded = {
        initialization: load_artifact(directory, initialization)
        for initialization, directory in directories.items()
    }
    registry_hashes = {manifest["registry_sha256"] for _, manifest in loaded.values()}
    if len(registry_hashes) != 1:
        raise RuntimeError("confirmation halves do not share one registry")
    artifacts = {key: value[0] for key, value in loaded.items()}
    if set(artifacts["concentrated"]) != set(artifacts["broad"]):
        raise RuntimeError("confirmation halves do not share target names")
    metadata = artifacts["concentrated"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    candidate_vs_control = {
        metric: ratio_result(
            artifacts, metadata, CANDIDATE, CONTROL, metric, rng
        )
        for metric in ("ed2", "heldout_sw1")
    }
    candidate_vs_paper = {
        metric: ratio_result(
            artifacts, metadata, CANDIDATE, PAPER, metric, rng
        )
        for metric in ("ed2", "heldout_sw1")
    }
    gate_a = all(
        result["geometric_mean_ratio"] < 1.0
        and result["bootstrap_95_interval"][1] < 1.0
        and result["wins"] >= 20
        for result in candidate_vs_control.values()
    )
    gate_c = all(
        result["geometric_mean_ratio"] < 1.0
        and result["bootstrap_95_interval"][1] < 1.0
        and result["wins"] >= 24
        for result in candidate_vs_paper.values()
    )
    mechanism = mechanism_result(artifacts)
    support = support_result(directories, artifacts)
    gates = {
        "A_quality_vs_current": gate_a,
        "B_mechanism": mechanism["passed"],
        "C_retain_paper_advantage": gate_c,
        "D_support_noninferiority": support["passed"],
    }
    result = {
        "schema": "conditioned-transport-repair-confirmation-analysis-v1",
        "status": "pass" if all(gates.values()) else "fail",
        "registry_sha256": next(iter(registry_hashes)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "artifacts": {
            initialization: {
                "directory": str(directory),
                "rows_sha256": sha256_file(directory / "rows.csv"),
                "outputs_sha256": sha256_file(directory / "outputs.npz"),
            }
            for initialization, directory in directories.items()
        },
        "candidate_vs_current": candidate_vs_control,
        "candidate_vs_paper": candidate_vs_paper,
        "mechanism": mechanism,
        "support": support,
        "cost": cost_result(artifacts),
        "gates": gates,
        "decision": (
            "promote variance/adaptive representative repair"
            if all(gates.values())
            else "do not promote; retain current confirmed representative system"
        ),
        "scope": (
            "32 fresh synthetic targets, two initializations, dimensions 2--16; "
            "not an image-scale claim"
        ),
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
