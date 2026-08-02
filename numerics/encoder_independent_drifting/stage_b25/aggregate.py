"""Aggregate the three paired B2.5 development units exactly once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..diagnostics import write_json
from ..stage_b2.artifacts import verify_sidecar
from .artifacts import DEFAULT_RESULT, assert_result_path_unused, load_preflight
from .core import B25_ARMS, B25Config
from .evaluation import adjudicate_development


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--units",
        type=Path,
        nargs="+",
        default=[
            Path(__file__).resolve().parent / f"b25_unit_{unit}.json"
            for unit in B25Config().units
        ],
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    assert_result_path_unused(args.out)
    config = B25Config()
    preflight = load_preflight()
    payloads = []
    for path in args.units:
        digest = verify_sidecar(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "b25-development-unit":
            raise RuntimeError(f"not a B2.5 unit artifact: {path}")
        if payload.get("preflight_sha256") != preflight["artifact_sha256"]:
            raise RuntimeError(f"B2.5 unit used another preflight: {path}")
        payload["artifact_path"] = str(path.resolve())
        payload["artifact_sha256"] = digest
        payloads.append(payload)
    if {int(payload["unit"]) for payload in payloads} != set(config.units):
        raise RuntimeError("aggregate does not contain the three declared units")

    final_rows = []
    for payload in payloads:
        unit = int(payload["unit"])
        if set(payload["cells"]) != set(B25_ARMS):
            raise RuntimeError(f"unit {unit} is missing a factorial cell")
        for arm in B25_ARMS:
            source = payload["cells"][arm][str(config.final_step)]["sources"][
                "in_domain_development_reused"
            ]
            final_rows.append(
                {
                    "unit": unit,
                    "arm": arm,
                    "metrics": source["metrics"],
                    "drift_summary": source["drift_summary"],
                    "training": payload["cells"][arm]["training"],
                }
            )
    verdict = adjudicate_development(final_rows, config)
    result = {
        "status": "b25-development",
        "development_only": True,
        "preflight_sha256": preflight["artifact_sha256"],
        "unit_artifacts": [
            {
                "unit": int(payload["unit"]),
                "path": payload["artifact_path"],
                "sha256": payload["artifact_sha256"],
            }
            for payload in sorted(payloads, key=lambda item: int(item["unit"]))
        ],
        "final_rows": final_rows,
        "verdict": verdict,
        "total_recorded_wall_seconds": sum(
            float(row["training"]["wall_seconds"]) for row in final_rows
        ),
        "claim_scope": (
            "prospective paired development factorial; any selected arm needs "
            "a new, untouched confirmation"
        ),
    }
    digest = write_json(args.out, result)
    print(f"B2.5 promising={verdict['promising']}")
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
