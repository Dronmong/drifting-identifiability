"""Audit the cross-fitted local controller on a consumed development artifact."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analyze_conditioned_transport_limitations import sha256_file


CROSSFIT = "cta-exact-crossfit"
FIXED = "cta-exact-hybrid"
GLOBAL = "cta-exact-global"


def coerce(value: str) -> Any:
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer() and all(mark not in value.lower() for mark in (".", "e")):
        return int(number)
    return number


def ratio(
    pairs: dict[str, dict[str, dict[str, Any]]],
    names: list[str],
    numerator: str,
    denominator: str,
    metric: str,
) -> dict[str, float | int]:
    values = np.asarray(
        [
            float(pairs[name][numerator][metric])
            / float(pairs[name][denominator][metric])
            for name in names
        ]
    )
    return {
        "geometric_mean_ratio": float(np.exp(np.mean(np.log(values)))),
        "median_ratio": float(np.median(values)),
        "wins": int(np.sum(values < 1.0)),
        "targets": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads((args.artifact / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "pass" or not audit.get("deep_metrics"):
        raise RuntimeError("crossfit artifact lacks a passing deep audit")
    with (args.artifact / "rows.csv").open(newline="", encoding="utf-8") as stream:
        rows = [{key: coerce(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    pairs: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        pairs.setdefault(str(row["target"]), {})[str(row["arm"])] = row
    expected = {CROSSFIT, FIXED, GLOBAL}
    if any(set(pair) != expected for pair in pairs.values()):
        raise RuntimeError("crossfit artifact does not contain the registered triplet")

    group_specs: dict[str, dict[str, list[str]]] = {
        "all": {"all": sorted(pairs)}
    }
    for field in ("dimension", "family"):
        groups: dict[str, list[str]] = {}
        for name, pair in pairs.items():
            value = str(pair[CROSSFIT][field])
            groups.setdefault(value, []).append(name)
        group_specs[f"by_{field}"] = groups

    groups_result: dict[str, Any] = {}
    for group_kind, groups in group_specs.items():
        groups_result[group_kind] = {}
        for group, names in groups.items():
            groups_result[group_kind][group] = {
                "crossfit_vs_fixed": {
                    metric: ratio(pairs, names, CROSSFIT, FIXED, metric)
                    for metric in ("ed2", "heldout_sw1")
                },
                "crossfit_vs_global": {
                    metric: ratio(pairs, names, CROSSFIT, GLOBAL, metric)
                    for metric in ("ed2", "heldout_sw1")
                },
                "mean_selected_weight": float(
                    np.mean(
                        [
                            pairs[name][CROSSFIT]["mean_selected_local_weight"]
                            for name in names
                        ]
                    )
                ),
                "mean_positive_weight_rate": float(
                    np.mean(
                        [
                            pairs[name][CROSSFIT][
                                "controller_positive_weight_rate"
                            ]
                            for name in names
                        ]
                    )
                ),
                "mean_controller_reported_relative_improvement": float(
                    np.mean(
                        [
                            pairs[name][CROSSFIT][
                                "mean_controller_relative_improvement"
                            ]
                            for name in names
                        ]
                    )
                ),
            }

    aggregate = groups_result["all"]["all"]["crossfit_vs_fixed"]
    passed = all(
        aggregate[metric]["geometric_mean_ratio"] < 1.0
        for metric in ("ed2", "heldout_sw1")
    )
    result = {
        "schema": "crossfit-controller-development-analysis-v1",
        "status": "pass" if passed else "fail",
        "evidence_scope": "consumed_registry_development_only",
        "artifact": str(args.artifact),
        "artifact_rows_sha256": sha256_file(args.artifact / "rows.csv"),
        "groups": groups_result,
        "development_gate": {
            "requires_both_aggregate_ratios_below_one": True,
            "passed": passed,
        },
        "interpretation": (
            "The one-step controller criterion did not predict the later "
            "amortized neural endpoint reliably; retain as an ablation."
            if not passed
            else "The controller cleared the consumed-registry development gate."
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
