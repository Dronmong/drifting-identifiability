"""Assemble a unit artifact from a recovery file after a budget stop.

``run_unit`` writes its artifact only when training reaches the declared
horizon.  A budget stop -- the predeclared rule that the last completed
checkpoint becomes the result -- therefore leaves the checkpoints on disk with
nothing describing them, and the sealed evaluation needs that description.

This reconstructs it from the recovery file, which already carries the
checkpoint records, the health history and the counters.  It is **not** in
``artifacts.py:_DEPENDENCIES``: it runs after training, changes no training
behaviour, and binding it would invalidate the preflight the run is bound to.

The artifact it writes is explicitly marked as a budget stop, and records the
declared horizon alongside the step actually reached, so no reader can mistake
it for a completed run.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ..diagnostics import write_json
from . import CAP_PHASE, CAP_UNIT
from .artifacts import DEFAULT_RESULT, HERE, assert_result_path_unused, load_preflight
from .config import examples_seen, profile
from .diagnostics import capability_gate
from .model import CAPPixelTransformer
from .training import TrainOutcome, clip_fraction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recovery", type=Path, default=HERE / "checkpoints" / "cap_recovery.pt"
    )
    parser.add_argument("--preflight", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULT)
    parser.add_argument(
        "--reason",
        default="budget exhausted before the declared horizon",
        help="recorded verbatim in the artifact",
    )
    args = parser.parse_args()

    assert_result_path_unused(args.out)
    preflight = load_preflight(args.preflight) if args.preflight else None
    frozen = profile("capability")
    payload = torch.load(args.recovery, map_location="cpu", weights_only=False)
    if payload.get("stage") != "cap-emf-1-recovery":
        raise RuntimeError("not a CAP-EMF-1 recovery file")

    completed = int(payload["completed_updates"])
    checkpoints = payload["checkpoints"]
    if not checkpoints:
        raise RuntimeError("the recovery file records no completed checkpoint")
    reached = max(int(step) for step in checkpoints)

    outcome = TrainOutcome(
        history=payload["history"],
        health=payload["health"],
        checkpoints=checkpoints,
        snapshots=payload.get("snapshots", []),
        wall_seconds=float(payload["wall_seconds"]),
        optimizer_updates=int(payload["optimizer_updates"]),
        examples_seen=int(payload["examples_seen"]),
        model_forwards=int(payload["model_forwards"]),
        clipped_updates=int(payload["clipped_updates"]),
        clipped_updates_final_window=int(
            payload.get("clipped_updates_final_window", 0)
        ),
        final_window_updates=int(payload.get("final_window_updates", 0)),
        nonfinite_updates=int(payload["nonfinite_updates"]),
        best_rank_ratio=float(payload["best_rank_ratio"]),
    )
    counters_present = {
        "clipped_updates_final_window",
        "final_window_updates",
    }.issubset(payload)
    final_clip_fraction = clip_fraction(outcome) if counters_present else None
    final = outcome.health[-1]
    gated_final = final.get("ema", final)
    gated_records = (
        [record["ema"] for record in outcome.health if "ema" in record]
        if "ema" in final
        else outcome.health
    )
    best_gated_rank = max(
        float(record["effective_rank_ratio"]) for record in gated_records
    )
    gate = capability_gate(
        gated_final,
        best_gated_rank,
        final_clip_fraction,
        outcome.nonfinite_updates,
        1,
        frozen.gate,
    )

    result = {
        "status": "cap-emf1-unit",
        "phase": CAP_PHASE,
        "unit": CAP_UNIT,
        "development_only": True,
        "correction": "none",
        "budget_stop": {
            "stopped": True,
            "reason": args.reason,
            "declared_horizon": frozen.train.updates,
            "updates_completed": completed,
            "result_checkpoint": reached,
            "fraction_of_declared": reached / frozen.train.updates,
            "examples_seen": outcome.examples_seen,
            "declared_examples": examples_seen(frozen.train),
            "rule": (
                "predeclared: when wall clock or budget is exhausted the last "
                "completed checkpoint is the result and the fact is recorded. "
                "This is a budget stop, not a metric selection -- no checkpoint "
                "was chosen by looking at its numbers."
            ),
        },
        "profile": None if preflight is None else preflight["profile"],
        "preflight_sha256": None if preflight is None else preflight["artifact_sha256"],
        "parameter_count": CAPPixelTransformer(frozen.model, 0).parameter_count(),
        "training": {
            "history": outcome.history,
            "health": outcome.health,
            "wall_seconds": outcome.wall_seconds,
            "optimizer_updates": outcome.optimizer_updates,
            "examples_seen": outcome.examples_seen,
            "model_forwards": outcome.model_forwards,
            "clipped_updates": outcome.clipped_updates,
            "clipped_updates_final_window": (
                outcome.clipped_updates_final_window if counters_present else None
            ),
            "final_window_updates": (
                outcome.final_window_updates if counters_present else None
            ),
            "clip_fraction_final_window": final_clip_fraction,
            "nonfinite_updates": outcome.nonfinite_updates,
            "best_rank_ratio": outcome.best_rank_ratio,
            "best_gated_rank_ratio": best_gated_rank,
        },
        "checkpoints": checkpoints,
        "train_only_gate": gate,
        "assembled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limits": [
            (
                "Assembled from a recovery file after a budget stop, not written by "
                "a completed run."
            ),
            (
                "Clip-window counters are present and audited."
                if counters_present
                else "Historical recovery lacks clip-window counters; H7 is unknown."
            ),
            "One developmental unit; no replication claim.",
        ],
    }
    digest = write_json(args.out, result)
    print(
        f"result checkpoint : {reached} ({100 * reached / frozen.train.updates:.0f}%)"
    )
    print(f"updates completed : {completed}")
    print(f"examples seen     : {outcome.examples_seen:,}")
    print(f"train-only verdict: {gate['verdict']}  failed={gate['failed']}")
    print(f"wrote {args.out} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
