"""Create a train-only, claim-bounded summary of one matched S3R unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..diagnostics import write_json
from .audit import HERE
from .config import S3R_ARMS


def _load(path: Path, arm: str, unit: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("arm") != arm or int(payload.get("unit", -1)) != unit:
        raise ValueError(f"artifact identity mismatch: {path}")
    return payload


def _endpoint(payload: dict, state: str) -> dict:
    history = payload.get("endpoint_history", [])
    if not history:
        raise ValueError(f"{payload['arm']} has no endpoint-health history")
    return history[-1][state]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=int, default=800)
    parser.add_argument("--run-dir", type=Path, default=HERE / "runs")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    rows = {}
    source = None
    examples = None
    for arm in S3R_ARMS:
        payload = _load(args.run_dir / f"s3r_{arm}_unit_{args.unit}.json", arm, args.unit)
        if source is None:
            source = payload["source_sha256"]
            examples = payload["examples_seen"]
        if payload["source_sha256"] != source or payload["examples_seen"] != examples:
            raise ValueError("arms are not source- and budget-matched")
        raw, ema = _endpoint(payload, "raw"), _endpoint(payload, "ema")
        rows[arm] = {
            "gate_passes": bool(payload["developmental_gate"]["passes"]),
            "raw_second_moment_ratio": raw["second_moment_ratio"],
            "raw_variance_ratio": raw["variance_ratio"],
            "raw_effective_rank_ratio": raw["effective_rank_ratio"],
            "ema_second_moment_ratio": ema["second_moment_ratio"],
            "ema_variance_ratio": ema["variance_ratio"],
            "ema_effective_rank_ratio": ema["effective_rank_ratio"],
            "clipping_fraction": payload["clipping_fraction"],
            "wall_seconds": payload["wall_seconds"],
            "peak_memory_bytes": payload["peak_memory_bytes"],
        }
    passing = [arm for arm, row in rows.items() if row["gate_passes"]]
    report = {
        "status": "s3r-one-unit-train-only-summary",
        "claim_boundary": (
            "one developmental unit can detect whether collapse repairs worked; "
            "it cannot establish image quality, reproducibility, or superiority"
        ),
        "unit": args.unit,
        "source_sha256": source,
        "examples_per_arm": examples,
        "arms": rows,
        "passing_arms": passing,
        "mechanical_conclusion": (
            "at least one repair cleared the predeclared train-only collapse gate"
            if passing
            else "no repair cleared the predeclared train-only collapse gate"
        ),
    }
    destination = args.out or args.run_dir / f"s3r_unit_{args.unit}_summary.json"
    digest = write_json(destination, report)
    print(json.dumps(report, indent=2))
    print(f"summary sha256={digest}")


if __name__ == "__main__":
    main()
