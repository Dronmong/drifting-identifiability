"""Post-hoc component-conditioned moment audit for frozen GMM outputs.

Component parameters are used only after training.  Each target and output
sample is assigned to its nearest normalized component center; the script then
compares component-conditioned means and covariances.  This is a mechanism
diagnostic, not a confirmation endpoint.
"""

from __future__ import annotations

import argparse
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


def safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


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


def component_errors(
    values: np.ndarray, reference: np.ndarray, centers: np.ndarray
) -> dict[str, Any]:
    reference_labels = np.argmin(
        np.linalg.norm(reference[:, None, :] - centers[None, :, :], axis=2),
        axis=1,
    )
    output_labels = np.argmin(
        np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2),
        axis=1,
    )
    weights = []
    mean_errors = []
    covariance_errors = []
    output_counts = []
    for component in range(len(centers)):
        target_values = reference[reference_labels == component]
        output_values = values[output_labels == component]
        weights.append(len(target_values) / len(reference))
        output_counts.append(int(len(output_values)))
        if len(target_values) < 2 or len(output_values) < 2:
            mean_errors.append(None)
            covariance_errors.append(None)
            continue
        mean_errors.append(
            float(
                np.linalg.norm(output_values.mean(axis=0) - target_values.mean(axis=0))
                / math.sqrt(values.shape[1])
            )
        )
        target_covariance = np.cov(target_values, rowvar=False, bias=True)
        output_covariance = np.cov(output_values, rowvar=False, bias=True)
        covariance_errors.append(
            float(
                np.linalg.norm(output_covariance - target_covariance)
                / max(np.linalg.norm(target_covariance), 1e-12)
            )
        )
    complete = all(value is not None for value in mean_errors)
    return {
        "all_components_represented": complete,
        "component_mean_rmse": (
            float(np.dot(weights, mean_errors)) if complete else None
        ),
        "component_covariance_relative_frobenius": (
            float(np.dot(weights, covariance_errors)) if complete else None
        ),
        "rare_component_mean_error": mean_errors[-1],
        "rare_component_covariance_relative_frobenius": covariance_errors[-1],
        "output_component_counts": output_counts,
    }


def finite_summary(records: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    values = [
        float(record[metric])
        for record in records
        if record[metric] is not None
    ]
    return {
        "median_finite": float(np.median(values)) if values else None,
        "mean_finite": float(np.mean(values)) if values else None,
        "finite_rows": len(values),
        "missing_rows": len(records) - len(values),
        "rows": len(records),
    }


def paired_summary(
    records: list[dict[str, Any]], comparator: str, metric: str
) -> dict[str, Any]:
    indexed = {
        (record["initialization"], record["target"], record["arm"]): record
        for record in records
    }
    ratios = []
    wins = 0
    candidate_only_finite = 0
    comparator_only_finite = 0
    dimensions: dict[int, list[float]] = {}
    pairs = sorted(
        {
            (record["initialization"], record["target"])
            for record in records
            if record["arm"] == CANDIDATE
        }
    )
    for initialization, target in pairs:
        candidate = indexed[(initialization, target, CANDIDATE)]
        baseline = indexed[(initialization, target, comparator)]
        c_value = candidate[metric]
        b_value = baseline[metric]
        if c_value is not None and b_value is None:
            candidate_only_finite += 1
        if c_value is None and b_value is not None:
            comparator_only_finite += 1
        if c_value is None or b_value is None:
            continue
        ratio = float(c_value) / float(b_value)
        ratios.append(ratio)
        dimensions.setdefault(int(candidate["dimension"]), []).append(ratio)
        wins += int(ratio < 1.0)
    return {
        "geometric_mean_ratio_on_jointly_finite_rows": (
            float(np.exp(np.mean(np.log(ratios)))) if ratios else None
        ),
        "wins": wins,
        "jointly_finite_rows": len(ratios),
        "candidate_only_finite_rows": candidate_only_finite,
        "comparator_only_finite_rows": comparator_only_finite,
        "by_dimension": {
            str(dimension): {
                "geometric_mean_ratio": float(
                    np.exp(np.mean(np.log(values)))
                ),
                "wins": int(np.sum(np.asarray(values) < 1.0)),
                "rows": len(values),
            }
            for dimension, values in sorted(dimensions.items())
        },
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
            "adaptive_rollout_component_moments.json"
        ),
    )
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    targets = {
        target["name"]: target
        for target in registry["targets"]
        if target["family"] in ("balanced-gmm", "rare-gmm")
    }
    records: list[dict[str, Any]] = []
    for initialization, directory in (
        ("concentrated", args.concentrated),
        ("broad", args.broad),
    ):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        pool_count = int(manifest["profile"]["target_pool"])
        with (
            np.load(directory / "outputs.npz") as outputs,
            np.load(directory / "references.npz") as references,
        ):
            for target_name, target in targets.items():
                centers = normalized_centers(target, pool_count)
                reference = references[safe_key(target_name)]
                for arm in ARMS:
                    diagnostics = component_errors(
                        outputs[safe_key(f"{arm}__{target_name}")],
                        reference,
                        centers,
                    )
                    records.append(
                        {
                            "initialization": initialization,
                            "target": target_name,
                            "family": target["family"],
                            "dimension": int(target["dimension"]),
                            "arm": arm,
                            **diagnostics,
                        }
                    )

    metrics = (
        "component_mean_rmse",
        "component_covariance_relative_frobenius",
        "rare_component_mean_error",
        "rare_component_covariance_relative_frobenius",
    )
    by_arm = {
        arm: {
            metric: finite_summary(
                [record for record in records if record["arm"] == arm],
                metric,
            )
            for metric in metrics
        }
        for arm in ARMS
    }
    paired = {
        comparator: {
            metric: paired_summary(records, comparator, metric)
            for metric in metrics
        }
        for comparator in (CONTROL, GATED, PAPER)
    }
    result = {
        "schema": "adaptive-rollout-component-moments-v1",
        "status": "post-hoc mechanism diagnostic; not a confirmation gate",
        "assignment": "nearest target component center after target-pool normalization",
        "target_count": len(targets),
        "rows": len(records),
        "by_arm": by_arm,
        "candidate_paired": paired,
        "records": records,
    }
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="utf-8"
    )

    print(f"GMM targets: {len(targets)}; rows: {len(records)}")
    for comparator in (CONTROL, GATED, PAPER):
        print(f"candidate / {comparator}")
        for metric in metrics:
            item = paired[comparator][metric]
            print(
                f"  {metric}: "
                f"{item['geometric_mean_ratio_on_jointly_finite_rows']}, "
                f"wins={item['wins']}/{item['jointly_finite_rows']}, "
                f"candidate-only-finite={item['candidate_only_finite_rows']}"
            )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

