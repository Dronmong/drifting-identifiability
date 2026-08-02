"""Aggregate the three B3 units without inventing a pass/fail gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..diagnostics import write_json
from ..stage_b2.artifacts import verify_sidecar
from .artifacts import DEFAULT_PREFLIGHT, DEFAULT_RESULT, assert_unused, load_preflight
from .core import B3_ARMS, B3Config

HERE = Path(__file__).resolve().parent


def _median(values) -> float:
    return float(np.median(np.asarray(list(values), dtype=float)))


def _metric_summary(rows: list[dict]) -> dict:
    names = (
        "recall",
        "precision",
        "kid",
        "fid",
        "density",
        "coverage",
        "effective_rank",
    )
    return {
        name: {
            "raw_units": [float(row["metrics"][name]) for row in rows],
            "median": _median(row["metrics"][name] for row in rows),
        }
        for name in names
    } | {
        "drift_raw_energy": {
            "raw_units": [
                float(row["drift_summary"]["raw_energy_mean"]) for row in rows
            ],
            "median": _median(row["drift_summary"]["raw_energy_mean"] for row in rows),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument(
        "--units",
        type=Path,
        nargs="*",
        default=[HERE / f"b3_unit_{unit}.json" for unit in B3Config().units],
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    assert_unused(args.out)
    preflight = load_preflight(args.preflight)
    config = B3Config()
    payloads = []
    for path in args.units:
        digest = verify_sidecar(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "b3-matched-reference-unit":
            raise RuntimeError(f"not a B3 unit artifact: {path}")
        if payload.get("preflight_sha256") != preflight["artifact_sha256"]:
            raise RuntimeError("B3 units do not share the frozen preflight")
        payload["artifact_sha256"] = digest
        payload["artifact_path"] = str(path.resolve())
        payloads.append(payload)
    if {int(row["unit"]) for row in payloads} != set(config.units):
        raise RuntimeError("B3 aggregation requires exactly units 600, 601, 602")
    payloads.sort(key=lambda row: int(row["unit"]))

    sources = ("in_domain_development_reused", "shifted_disjoint")
    b3_summary = {}
    for arm in B3_ARMS:
        b3_summary[arm.name] = {}
        for source in sources:
            rows = [
                unit["cells"][arm.name]["evaluations"][str(config.steps)][source]
                for unit in payloads
            ]
            b3_summary[arm.name][source] = _metric_summary(rows)
        b3_summary[arm.name]["training"] = {
            "parameters": int(payloads[0]["cells"][arm.name]["parameters"]),
            "wall_seconds_raw_units": [
                float(unit["cells"][arm.name]["training"]["wall_seconds"])
                for unit in payloads
            ],
            "wall_seconds_median": _median(
                unit["cells"][arm.name]["training"]["wall_seconds"] for unit in payloads
            ),
            "peak_memory_reserved_bytes_raw_units": [
                unit["cells"][arm.name]["training"]["peak_memory_reserved_bytes"]
                for unit in payloads
            ],
        }

    bridge_summary = {}
    for name in ("B0", "B1", "B2"):
        bridge_summary[name] = {}
        for source in sources:
            rows = [
                unit["frozen_bridge_references"][name]["evaluations"][source]
                for unit in payloads
            ]
            bridge_summary[name][source] = _metric_summary(rows)

    output = {
        "status": "b3-matched-reference",
        "reference_measurement_only": True,
        "verdict": None,
        "reading": "descriptive matched rows; B3 has no pass/fail category",
        "preflight_path": str(args.preflight.resolve()),
        "preflight_sha256": preflight["artifact_sha256"],
        "unit_artifacts": [
            {
                "unit": int(row["unit"]),
                "path": row["artifact_path"],
                "sha256": row["artifact_sha256"],
            }
            for row in payloads
        ],
        "b3": b3_summary,
        "frozen_bridge_references": bridge_summary,
        "raw_unit_comparisons": [row["comparisons"] for row in payloads],
        "limits": [
            "Three units provide coarse consistency, not high-powered inference.",
            "No source was pooled and no intermediate checkpoint selected the result.",
            "Cross-architecture intervals use independent generated resampling.",
            "This is not an evaluation of the complete published paper model.",
        ],
    }
    digest = write_json(args.out, output)
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
