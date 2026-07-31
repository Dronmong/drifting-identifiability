"""Post-hoc diagnosis of the conditioned-transport representative repair.

This analysis consumes only the already-frozen repair-confirmation artifacts.
It is explanatory, not a new confirmation and not a source of promotable
performance claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


CANDIDATE = "cta-exact-hybrid"
CONTROL = "cta-exact-fixed-control"
PAPER = "paper-neural-optimized"


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
        rows = [
            {key: coerce(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    result = {(str(row["target"]), str(row["arm"])): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError(f"{directory} contains duplicate target/arm rows")
    return result


def geometric_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires finite positive values")
    return float(np.exp(np.mean(np.log(array))))


def ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic average ranks, including exact ties."""
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return result


def spearman(left: list[float], right: list[float]) -> float:
    x = ranks(np.asarray(left, dtype=float))
    y = ranks(np.asarray(right, dtype=float))
    return float(np.corrcoef(x, y)[0, 1])


def mixed_seed(seed: int, replication: int, purpose: int) -> int:
    sequence = np.random.SeedSequence([seed, replication, purpose])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def normalized_centers(target: dict[str, Any], pool_count: int) -> np.ndarray:
    parameters = target["parameters"]
    centers = np.asarray(parameters["centers"], dtype=float)
    weights = np.asarray(parameters["weights"], dtype=float)
    sigmas = np.asarray(parameters["sigmas"], dtype=float)
    rng = np.random.default_rng(
        mixed_seed(int(target["seeds"]["target_pool"]), 0, 1)
    )
    components = rng.choice(len(weights), size=pool_count, p=weights)
    pool = centers[components] + (
        rng.normal(size=(pool_count, int(target["dimension"])))
        * sigmas[components, None]
    )
    center = pool.mean(axis=0)
    scale = float(np.sqrt(np.mean((pool - center) ** 2)))
    return (centers - center) / scale


def nearest_center_labels(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        values[:, None, :] - centers[None, :, :],
        axis=2,
    )
    return np.argmin(distances, axis=1)


def paired_records(
    artifacts: dict[str, dict[tuple[str, str], dict[str, Any]]],
) -> list[dict[str, Any]]:
    names = sorted(
        {
            target
            for rows in artifacts.values()
            for target, arm in rows
            if arm == CANDIDATE
        }
    )
    records: list[dict[str, Any]] = []
    for initialization, rows in artifacts.items():
        for target in names:
            candidate = rows[(target, CANDIDATE)]
            control = rows[(target, CONTROL)]
            record: dict[str, Any] = {
                "initialization": initialization,
                "target": target,
                "dimension": int(candidate["dimension"]),
                "family": str(candidate["family"]),
            }
            for metric in ("ed2", "heldout_sw1"):
                record[f"{metric}_ratio"] = (
                    float(candidate[metric]) / float(control[metric])
                )
            record.update(
                {
                    "field_error_ratio": (
                        float(
                            candidate[
                                "mean_representative_field_relative_l2_error"
                            ]
                        )
                        / float(
                            control[
                                "mean_representative_field_relative_l2_error"
                            ]
                        )
                    ),
                    "field_cosine_gain": (
                        float(candidate["minimum_representative_field_cosine"])
                        - float(control["minimum_representative_field_cosine"])
                    ),
                    "row_mass_error_ratio": (
                        float(
                            candidate[
                                "mean_representative_row_mass_relative_l2_error"
                            ]
                        )
                        / float(
                            control[
                                "mean_representative_row_mass_relative_l2_error"
                            ]
                        )
                    ),
                    "column_mass_error_ratio": (
                        float(
                            candidate[
                                "mean_representative_column_mass_relative_l2_error"
                            ]
                        )
                        / float(
                            control[
                                "mean_representative_column_mass_relative_l2_error"
                            ]
                        )
                    ),
                    "local_scale_ratio": (
                        float(candidate["mean_local_scale"])
                        / float(control["mean_local_scale"])
                    ),
                    "teacher_displacement_ratio": (
                        float(candidate["mean_teacher_displacement_rms"])
                        / float(control["mean_teacher_displacement_rms"])
                    ),
                    "output_rms_ratio": (
                        float(candidate["output_rms"]) / float(control["output_rms"])
                    ),
                }
            )
            records.append(record)
    return records


def grouped_summary(
    records: list[dict[str, Any]], key: str
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for value in sorted({record[key] for record in records}, key=str):
        selected = [record for record in records if record[key] == value]
        result[str(value)] = {
            "rows": len(selected),
            "ed2_geometric_ratio": geometric_mean(
                [record["ed2_ratio"] for record in selected]
            ),
            "sw1_geometric_ratio": geometric_mean(
                [record["heldout_sw1_ratio"] for record in selected]
            ),
            "ed2_win_fraction": float(
                np.mean([record["ed2_ratio"] < 1.0 for record in selected])
            ),
            "sw1_win_fraction": float(
                np.mean([record["heldout_sw1_ratio"] < 1.0 for record in selected])
            ),
            "field_error_geometric_ratio": geometric_mean(
                [record["field_error_ratio"] for record in selected]
            ),
        }
    return result


def target_reduced(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for target in sorted({record["target"] for record in records}):
        selected = [record for record in records if record["target"] == target]
        result.append(
            {
                "target": target,
                "dimension": selected[0]["dimension"],
                "family": selected[0]["family"],
                "ed2_ratio": geometric_mean(
                    [record["ed2_ratio"] for record in selected]
                ),
                "heldout_sw1_ratio": geometric_mean(
                    [record["heldout_sw1_ratio"] for record in selected]
                ),
            }
        )
    return result


def switched_posthoc(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Describe, but do not validate, a dimension-gated reconstruction."""
    reduced = target_reduced(records)
    ed = [
        row["ed2_ratio"] if int(row["dimension"]) >= 8 else 1.0
        for row in reduced
    ]
    sw = [
        row["heldout_sw1_ratio"] if int(row["dimension"]) >= 8 else 1.0
        for row in reduced
    ]
    return {
        "rule": "use repaired arm only at dimension >= 8; otherwise control",
        "status": "post-hoc design evidence only",
        "target_count": len(reduced),
        "ed2_geometric_ratio": geometric_mean(ed),
        "heldout_sw1_geometric_ratio": geometric_mean(sw),
        "ed2_wins": sum(value < 1.0 for value in ed),
        "heldout_sw1_wins": sum(value < 1.0 for value in sw),
    }


def mode_details(
    values: np.ndarray,
    reference: np.ndarray,
    centers: np.ndarray,
) -> dict[str, Any]:
    labels = nearest_center_labels(values, centers)
    reference_labels = nearest_center_labels(reference, centers)
    counts = np.bincount(labels, minlength=len(centers))
    component_details: list[dict[str, float | int | None]] = []
    for component in range(len(centers)):
        selected = values[labels == component]
        selected_reference = reference[reference_labels == component]
        if len(selected_reference) < 2:
            component_details.append(
                {
                    "component": component,
                    "count": len(selected),
                    "target_calibrated_95pct_core_radius": None,
                    "core_count": None,
                    "core_mass": None,
                    "mean_error": None,
                    "relative_covariance_error": None,
                }
            )
            continue
        core_radius = float(
            np.quantile(
                np.linalg.norm(
                    selected_reference - centers[component],
                    axis=1,
                ),
                0.95,
            )
        )
        core_count = int(
            np.sum(
                np.linalg.norm(values - centers[component], axis=1)
                <= core_radius
            )
        )
        if len(selected) < 2:
            component_details.append(
                {
                    "component": component,
                    "count": len(selected),
                    "target_calibrated_95pct_core_radius": core_radius,
                    "core_count": core_count,
                    "core_mass": core_count / len(values),
                    "mean_error": None,
                    "relative_covariance_error": None,
                }
            )
            continue
        reference_covariance = np.cov(selected_reference, rowvar=False)
        component_details.append(
            {
                "component": component,
                "count": len(selected),
                "target_calibrated_95pct_core_radius": core_radius,
                "core_count": core_count,
                "core_mass": core_count / len(values),
                "mean_error": float(
                    np.linalg.norm(selected.mean(axis=0) - selected_reference.mean(axis=0))
                ),
                "relative_covariance_error": float(
                    np.linalg.norm(
                        np.cov(selected, rowvar=False) - reference_covariance,
                        ord="fro",
                    )
                    / max(np.linalg.norm(reference_covariance, ord="fro"), 1e-15)
                ),
            }
        )
    return {
        "counts": counts.tolist(),
        "masses": (counts / len(values)).tolist(),
        "components": component_details,
    }


def plot_d2_rare(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    reference: np.ndarray,
    centers: np.ndarray,
) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(13.5, 3.3), sharex=True, sharey=True)
    panels = [
        ("Target reference", reference),
        ("Repaired variance tree", arrays[CANDIDATE]),
        ("Current fixed tree", arrays[CONTROL]),
        ("Paper port", arrays[PAPER]),
    ]
    for axis, (title, values) in zip(axes, panels, strict=True):
        labels = nearest_center_labels(values, centers)
        axis.scatter(
            values[labels == 0, 0],
            values[labels == 0, 1],
            s=4,
            alpha=0.28,
            color="#3266a8",
            linewidths=0,
        )
        axis.scatter(
            values[labels == 1, 0],
            values[labels == 1, 1],
            s=7,
            alpha=0.55,
            color="#d95f02",
            linewidths=0,
        )
        axis.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="x",
            s=45,
            color="black",
            linewidths=1.5,
        )
        axis.set_title(title, fontsize=9)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.15)
    axes[0].set_ylabel("normalized feature 2")
    for axis in axes:
        axis.set_xlabel("normalized feature 1")
    figure.suptitle(
        "Broad-init 2D rare mixture: nearest-center color is diagnostic, not a training label",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_d16_projection(
    output_path: Path,
    arrays: dict[str, np.ndarray],
    reference: np.ndarray,
    centers: np.ndarray,
) -> None:
    direction = centers[1] - centers[0]
    direction /= np.linalg.norm(direction)
    origin = centers[0]
    panels = [
        ("Target reference", reference),
        ("Repaired variance tree", arrays[CANDIDATE]),
        ("Current fixed tree", arrays[CONTROL]),
        ("Paper port", arrays[PAPER]),
    ]
    projected = [
        ((values - origin) @ direction, title) for title, values in panels
    ]
    lower = min(float(np.min(values)) for values, _ in projected)
    upper = max(float(np.max(values)) for values, _ in projected)
    bins = np.linspace(lower, upper, 75)
    figure, axis = plt.subplots(figsize=(8.4, 4.2))
    colors = ("black", "#d95f02", "#3266a8", "#6a3d9a")
    for (values, title), color in zip(projected, colors, strict=True):
        axis.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.6,
            label=title,
            color=color,
        )
    axis.axvline(0.0, color="#3266a8", linestyle=":", linewidth=1.0)
    axis.axvline(
        float((centers[1] - origin) @ direction),
        color="#d95f02",
        linestyle=":",
        linewidth=1.0,
    )
    axis.set_yscale("log")
    axis.set_xlabel("projection from common center toward rare center")
    axis.set_ylabel("density (log scale)")
    axis.set_title("Broad-init 16D rare mixture: mode-axis projection")
    axis.grid(alpha=0.15)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concentrated",
        type=Path,
        default=Path(
            "numerics/conditioned_transport_runs/20260723-144129-consumed"
        ),
    )
    parser.add_argument(
        "--broad",
        type=Path,
        default=Path(
            "numerics/conditioned_transport_runs/20260723-144337-consumed"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "numerics/conditioned_transport_repair_failure_diagnosis.json"
        ),
    )
    parser.add_argument(
        "--figure-directory",
        type=Path,
        default=Path("numerics/conditioned_transport_repair_failure_figures"),
    )
    args = parser.parse_args()

    directories = {
        "concentrated": args.concentrated.resolve(),
        "broad": args.broad.resolve(),
    }
    artifacts = {
        initialization: load_rows(directory)
        for initialization, directory in directories.items()
    }
    records = paired_records(artifacts)
    reduced = target_reduced(records)

    correlation_metrics = (
        "field_error_ratio",
        "field_cosine_gain",
        "row_mass_error_ratio",
        "column_mass_error_ratio",
        "local_scale_ratio",
        "teacher_displacement_ratio",
        "output_rms_ratio",
    )
    correlations = {
        endpoint: {
            metric: spearman(
                [record[endpoint] for record in records],
                [record[metric] for record in records],
            )
            for metric in correlation_metrics
        }
        for endpoint in ("ed2_ratio", "heldout_sw1_ratio")
    }

    broad_manifest = json.loads(
        (directories["broad"] / "manifest.json").read_text(encoding="utf-8")
    )
    registry = json.loads(
        (
            directories["broad"]
            / "conditioned_transport_repair_confirmation_registry.json"
        ).read_text(encoding="utf-8")
    )
    target_map = {target["name"]: target for target in registry["targets"]}
    broad_outputs = np.load(directories["broad"] / "outputs.npz")
    broad_references = np.load(directories["broad"] / "references.npz")
    args.figure_directory.mkdir(parents=True, exist_ok=True)

    rare_details: dict[str, Any] = {}
    for target_name in ("NPR-d2-rare-gmm-v00", "NPR-d16-rare-gmm-v00"):
        target = target_map[target_name]
        centers = normalized_centers(
            target, int(broad_manifest["profile"]["target_pool"])
        )
        key = safe_key(target_name)
        reference = broad_references[key]
        arrays = {
            arm: broad_outputs[safe_key(f"{arm}__{target_name}")]
            for arm in (CANDIDATE, CONTROL, PAPER)
        }
        rare_details[target_name] = {
            "target_weights": target["parameters"]["weights"],
            "half_weight_coverage_threshold_count": (
                0.5
                * float(target["parameters"]["weights"][-1])
                * len(reference)
            ),
            "reference": mode_details(reference, reference, centers),
            **{
                arm: mode_details(values, reference, centers)
                for arm, values in arrays.items()
            },
        }
        if int(target["dimension"]) == 2:
            plot_d2_rare(
                args.figure_directory / "d2_rare_v00_scatter.png",
                arrays,
                reference,
                centers,
            )
        else:
            plot_d16_projection(
                args.figure_directory / "d16_rare_v00_mode_axis.png",
                arrays,
                reference,
                centers,
            )

    payload = {
        "schema": "conditioned-transport-repair-failure-diagnosis-v1",
        "status": "post-hoc explanatory analysis of consumed artifacts",
        "inputs": {
            initialization: {
                "directory": str(directory),
                "rows_sha256": sha256_file(directory / "rows.csv"),
                "outputs_sha256": sha256_file(directory / "outputs.npz"),
                "references_sha256": sha256_file(directory / "references.npz"),
            }
            for initialization, directory in directories.items()
        },
        "row_count": len(records),
        "target_count": len(reduced),
        "by_dimension": grouped_summary(records, "dimension"),
        "by_family": grouped_summary(records, "family"),
        "spearman_correlations": correlations,
        "worst_target_reduced_ed2": sorted(
            reduced, key=lambda row: row["ed2_ratio"], reverse=True
        )[:8],
        "posthoc_dimension_switch": switched_posthoc(records),
        "rare_mode_details": rare_details,
        "figures": [
            str(args.figure_directory / "d2_rare_v00_scatter.png"),
            str(args.figure_directory / "d16_rare_v00_mode_axis.png"),
        ],
        "interpretation_guardrail": (
            "No subgroup, switch rule, or diagnostic in this file is a fresh "
            "performance confirmation."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(args.output)
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {args.output.name}\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
