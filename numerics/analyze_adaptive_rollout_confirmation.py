"""Analyze the frozen dimension-adaptive rollout confirmation.

The independent unit is a target.  Concentrated and broad initializations are
geometrically reduced inside target before the stratified target bootstrap.
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


CANDIDATE = "cta-exact-adaptive-rollout"
CONTROL = "cta-exact-fixed-control"
GATED = "cta-exact-gated-hybrid"
PAPER = "paper-neural-optimized"
ARMS = (CANDIDATE, CONTROL, GATED, PAPER)
BOOTSTRAP_SEED = 2026110709
BOOTSTRAP_RESAMPLES = 20_000
QUALITY_METRICS = ("ed2", "heldout_sw1", "training_quantile_rmse")
COST_METRICS = (
    "wall_seconds",
    "online_training_wall_seconds",
    "projection_scalar_products",
    "sort_work",
    "paper_kernel_pairs",
    "generator_forward_calls_training",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    registry_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not audit.get("deep_metrics"):
        raise RuntimeError(f"{directory} lacks a passing deep audit")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if tuple(manifest["registered_arms"]) != ARMS:
        raise RuntimeError(f"{directory} has the wrong registered arms")
    if manifest["registry_sha256"] != registry_sha256:
        raise RuntimeError(f"{directory} used the wrong registry")

    with (directory / "rows.csv").open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["initialization"] != initialization:
            raise RuntimeError(f"{directory} has the wrong initialization")
        result.setdefault(str(row["target"]), {})[str(row["arm"])] = row
    if not result or any(set(pair) != set(ARMS) for pair in result.values()):
        raise RuntimeError(f"{directory} lacks complete target quadruplets")

    expected_steps = {2: 1, 4: 1, 8: 2, 16: 4}
    for pair in result.values():
        candidate = pair[CANDIDATE]
        control = pair[CONTROL]
        dimension = int(candidate["dimension"])
        steps = expected_steps[dimension]
        if (
            int(candidate["registered_global_rollout_steps"]) != steps
            or int(candidate["rollout_local_after_global"]) != int(steps > 1)
            or int(candidate["per_particle_local_safety"]) != 0
            or int(candidate["tail_balanced_amortization"]) != 0
        ):
            raise RuntimeError("the frozen adaptive rollout schedule changed")
        expected_count = 128 if dimension < 8 else 256
        expected_strategy = "fixed-level" if dimension < 8 else "variance-per-node"
        if (
            int(candidate["registered_local_representative_count"])
            != expected_count
            or candidate["registered_representative_strategy"] != expected_strategy
            or int(control["registered_global_rollout_steps"]) != 1
            or int(control["registered_local_representative_count"]) != 128
            or control["registered_representative_strategy"] != "fixed-level"
        ):
            raise RuntimeError("the frozen representative configuration changed")
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


def grouped_ratios(
    ratios: dict[str, float],
    metadata: dict[str, dict[str, Any]],
    field: str,
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = {}
    for name, value in ratios.items():
        key = str(metadata[name][CANDIDATE][field])
        groups.setdefault(key, []).append(value)
    return {
        key: {
            "geometric_mean_ratio": float(
                np.exp(np.mean(np.log(np.asarray(values))))
            ),
            "wins": int(np.sum(np.asarray(values) < 1.0)),
            "targets": len(values),
        }
        for key, values in sorted(groups.items())
    }


def ratio_result(
    artifacts: dict[str, dict[str, dict[str, Any]]],
    metadata: dict[str, dict[str, Any]],
    numerator: str,
    denominator: str,
    metric: str,
    rng: np.random.Generator,
    *,
    bootstrap: bool = True,
) -> dict[str, Any]:
    ratios = target_reduced_ratios(
        artifacts, numerator, denominator, metric
    )
    values = np.asarray(list(ratios.values()))
    result: dict[str, Any] = {
        "geometric_mean_ratio": float(np.exp(np.mean(np.log(values)))),
        "median_ratio": float(np.median(values)),
        "wins": int(np.sum(values < 1.0)),
        "ties": int(np.sum(np.isclose(values, 1.0, rtol=0.0, atol=1e-14))),
        "targets": len(values),
        "by_dimension": grouped_ratios(ratios, metadata, "dimension"),
        "by_family": grouped_ratios(ratios, metadata, "family"),
        "per_target": ratios,
    }
    if bootstrap:
        result["bootstrap_95_interval"] = stratified_interval(
            ratios, metadata, rng
        )
    return result


def rare_diagnostics(
    artifacts: dict[str, dict[str, dict[str, Any]]],
    comparator: str,
) -> dict[str, Any]:
    higher_is_better = {
        "rare_core_mass",
        "component_core_coverage",
        "maximum_teacher_rare_core_count",
        "final_teacher_rare_core_count",
        "maximum_run_rare_core_count",
    }
    metrics = (
        "rare_mass_error",
        "rare_core_mass",
        "rare_bridge_mass",
        "component_core_coverage",
        "maximum_teacher_rare_core_count",
        "final_teacher_rare_core_count",
        "maximum_run_rare_core_count",
    )
    names = [
        name
        for name, pair in next(iter(artifacts.values())).items()
        if pair[CANDIDATE]["family"] == "rare-gmm"
    ]
    result: dict[str, Any] = {}
    for metric in metrics:
        candidate_values = []
        comparator_values = []
        dimensions = []
        wins = 0
        for name in names:
            c = float(
                np.mean(
                    [
                        float(rows[name][CANDIDATE][metric])
                        for rows in artifacts.values()
                    ]
                )
            )
            b = float(
                np.mean(
                    [
                        float(rows[name][comparator][metric])
                        for rows in artifacts.values()
                    ]
                )
            )
            candidate_values.append(c)
            comparator_values.append(b)
            dimensions.append(
                int(next(iter(artifacts.values()))[name][CANDIDATE]["dimension"])
            )
            wins += int(c > b if metric in higher_is_better else c < b)
        by_dimension = {}
        for dimension in sorted(set(dimensions)):
            selected = [
                index
                for index, value in enumerate(dimensions)
                if value == dimension
            ]
            c_values = np.asarray(candidate_values)[selected]
            b_values = np.asarray(comparator_values)[selected]
            by_dimension[str(dimension)] = {
                "candidate_mean": float(np.mean(c_values)),
                "comparator_mean": float(np.mean(b_values)),
                "mean_difference": float(np.mean(c_values - b_values)),
            }
        result[metric] = {
            "candidate_mean": float(np.mean(candidate_values)),
            "comparator_mean": float(np.mean(comparator_values)),
            "mean_difference": float(
                np.mean(np.asarray(candidate_values) - np.asarray(comparator_values))
            ),
            "wins": wins,
            "targets": len(names),
            "by_dimension": by_dimension,
        }
    return result


def low_dimension_identity(
    artifacts: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, float | bool]:
    maximum = 0.0
    rows_checked = 0
    for rows in artifacts.values():
        for pair in rows.values():
            if int(pair[CANDIDATE]["dimension"]) >= 8:
                continue
            rows_checked += 1
            for metric in QUALITY_METRICS:
                maximum = max(
                    maximum,
                    abs(
                        float(pair[CANDIDATE][metric])
                        - float(pair[CONTROL][metric])
                    ),
                )
    return {
        "rows_checked": rows_checked,
        "maximum_quality_metric_difference": maximum,
        "bitwise_metric_identity": maximum == 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concentrated", type=Path, required=True)
    parser.add_argument("--broad", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(
            "adaptive_rollout_confirmation_analysis.json"
        ),
    )
    args = parser.parse_args()

    registry_sha256 = sha256_file(args.registry)
    artifacts: dict[str, dict[str, dict[str, Any]]] = {}
    manifests = {}
    for initialization, directory in (
        ("concentrated", args.concentrated),
        ("broad", args.broad),
    ):
        artifacts[initialization], manifests[initialization] = load_artifact(
            directory, initialization, registry_sha256
        )
    if set(artifacts["concentrated"]) != set(artifacts["broad"]):
        raise RuntimeError("initializations do not contain the same targets")

    metadata = artifacts["concentrated"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    quality: dict[str, Any] = {}
    for comparator in (CONTROL, GATED, PAPER):
        quality[comparator] = {
            metric: ratio_result(
                artifacts,
                metadata,
                CANDIDATE,
                comparator,
                metric,
                rng,
            )
            for metric in QUALITY_METRICS
        }
    costs = {
        comparator: {
            metric: ratio_result(
                artifacts,
                metadata,
                CANDIDATE,
                comparator,
                metric,
                rng,
                bootstrap=False,
            )
            for metric in COST_METRICS
        }
        for comparator in (CONTROL, GATED)
    }
    primary = {
        f"candidate_vs_{comparator}_{metric}_upper_below_one": (
            quality[comparator][metric]["bootstrap_95_interval"][1] < 1.0
        )
        for comparator in (CONTROL, PAPER)
        for metric in ("ed2", "heldout_sw1")
    }
    primary["all_primary_gates_pass"] = all(primary.values())

    result = {
        "schema": "adaptive-rollout-confirmation-analysis-v1",
        "candidate": CANDIDATE,
        "comparators": [CONTROL, GATED, PAPER],
        "registry": str(args.registry),
        "registry_sha256": registry_sha256,
        "artifact_directories": {
            key: str(value)
            for key, value in (
                ("concentrated", args.concentrated),
                ("broad", args.broad),
            )
        },
        "artifact_audit_sha256": {
            key: manifest["audit_sha256"] for key, manifest in manifests.items()
        },
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "target_count": len(metadata),
        "quality": quality,
        "cost": costs,
        "rare_diagnostics_vs_fixed": rare_diagnostics(artifacts, CONTROL),
        "rare_diagnostics_vs_gated": rare_diagnostics(artifacts, GATED),
        "low_dimension_identity": low_dimension_identity(artifacts),
        "gates": primary,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(
        f"{sha256_file(args.output)}  {args.output.name}\n",
        encoding="utf-8",
    )

    print(f"targets: {result['target_count']}")
    for comparator in (CONTROL, GATED, PAPER):
        print(f"candidate / {comparator}")
        for metric in QUALITY_METRICS:
            item = quality[comparator][metric]
            interval = item["bootstrap_95_interval"]
            print(
                f"  {metric}: {item['geometric_mean_ratio']:.4f} "
                f"[{interval[0]:.4f}, {interval[1]:.4f}], "
                f"wins={item['wins']}/{item['targets']}"
            )
    print(f"all primary gates pass: {primary['all_primary_gates_pass']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
