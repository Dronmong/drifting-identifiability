"""Run the single CAP-EMF-1 capability unit.

Blocked unless a source-matched preflight returning GO is present **and** the
operator explicitly opts in.  This is the expensive rented-GPU run; it must not
be startable by accident.

No test image is instantiated here.  The sealed evaluation is a separate step
that runs only after the final checkpoint is frozen and hashed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..diagnostics import write_json
from . import CAP_PHASE, CAP_UNIT
from .artifacts import (
    CHECKPOINTS,
    DEFAULT_PREFLIGHT,
    DEFAULT_RESULT,
    assert_result_path_unused,
    checkpoint_path,
    load_preflight,
    profile_payload,
    save_checkpoint,
)
from .config import enable_tf32, examples_seen, profile
from .data import cifar10_train_pool
from .diagnostics import capability_gate
from .training import clip_fraction, train_cap_unit


def main() -> int:
    parser = argparse.ArgumentParser(description="CAP-EMF-1 capability unit")
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINTS)
    parser.add_argument("--recovery", type=Path, default=None)
    parser.add_argument(
        "--nondeterministic",
        action="store_true",
        help=(
            "disable deterministic CUDA kernels. Faster, and therefore cheaper "
            "on a rented GPU, but exact replay is then only guaranteed through "
            "the recovery file. The choice is recorded in the artifact."
        ),
    )
    parser.add_argument(
        "--i-have-authorized-the-budget-run",
        action="store_true",
        help="explicit opt-in; the run is refused without it",
    )
    args = parser.parse_args()

    if not args.i_have_authorized_the_budget_run:
        raise SystemExit(
            "CAP-EMF-1 is the budget run. Re-invoke with "
            "--i-have-authorized-the-budget-run once the cloud benchmark has "
            "projected the cost and the operator has approved it."
        )

    assert_result_path_unused(args.out)
    preflight = load_preflight(args.preflight)
    frozen = profile("capability")
    payload = profile_payload(frozen)
    if preflight["profile"] != payload:
        raise RuntimeError("CAP-EMF-1 profile differs from the preflight")

    planned = [
        checkpoint_path(step, kind)
        for step in frozen.train.checkpoint_updates
        for kind in ("raw", "ema")
    ]
    if any(path.exists() for path in planned):
        raise RuntimeError(
            "a planned CAP checkpoint already exists; inspect before resuming"
        )

    torch.set_num_threads(4)
    precision = enable_tf32()
    if not args.nondeterministic:
        # Requires CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment.
        torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    pool = cifar10_train_pool(args.data_root)
    recovery = args.recovery or (args.checkpoint_dir / "cap_recovery.pt")

    def checkpoint(step: int, raw_state: dict, ema_state: dict) -> dict:
        # The count comes from the state being written, not from a variable
        # assigned after training returns -- an earlier draft read a closure
        # that was still zero at every checkpoint, so every checkpoint recorded
        # a parameter count of 0.
        count = sum(value.numel() for value in raw_state.values())
        entry = {}
        for kind, state in (("raw", raw_state), ("ema", ema_state)):
            path = checkpoint_path(step, kind)
            entry[kind] = {
                "path": str(path.resolve()),
                "sha256": save_checkpoint(
                    path,
                    state,
                    step=step,
                    kind=kind,
                    profile=payload,
                    preflight_sha=preflight["artifact_sha256"],
                    parameter_count=count,
                ),
            }
        # Returned, not stored locally: the training loop keeps it on the
        # outcome, which the recovery file restores. A local dict would be
        # empty after a resume and the artifact would lose every checkpoint
        # written before the interruption.
        return entry

    snapshot_dir = args.checkpoint_dir / "posthoc_ema_snapshots"

    def snapshot(step: int, state: dict) -> None:
        """Raw weights for post-hoc EMA synthesis (Karras et al.).

        Secondary by construction: the declared 0.9999 EMA remains the primary
        result, so synthesizing other horizons afterwards cannot become
        checkpoint selection on a metric.
        """
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / f"cap_snapshot_step{step}.pt"
        if path.exists():
            return
        torch.save(
            {
                "stage": "cap-emf-1-snapshot",
                "step": step,
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in state.items()
                    if value.is_floating_point()
                },
            },
            path,
        )

    started = time.time()
    outcome = train_cap_unit(
        pool,
        frozen,
        device,
        recovery_path=recovery,
        checkpoint=checkpoint,
        snapshot=snapshot,
        progress=lambda message: print(message, flush=True),
    )

    final = outcome.health[-1]
    gate = capability_gate(
        final,
        outcome.best_rank_ratio,
        clip_fraction(outcome),
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
        "deterministic_algorithms": not args.nondeterministic,
        "precision": precision,
        "examples_seen_target": examples_seen(frozen.train),
        "device": settings,
        "preflight_sha256": preflight["artifact_sha256"],
        "profile": payload,
        "parameter_count": outcome.parameter_count,
        "training": {
            "history": outcome.history,
            "health": outcome.health,
            "wall_seconds": outcome.wall_seconds,
            "peak_memory_bytes": outcome.peak_memory_bytes,
            "peak_memory_reserved_bytes": outcome.peak_memory_reserved_bytes,
            "optimizer_updates": outcome.optimizer_updates,
            "examples_seen": outcome.examples_seen,
            "model_forwards": outcome.model_forwards,
            "clipped_updates": outcome.clipped_updates,
            "clip_fraction_final_window": clip_fraction(outcome),
            "nonfinite_updates": outcome.nonfinite_updates,
            "best_rank_ratio": outcome.best_rank_ratio,
        },
        "checkpoints": outcome.checkpoints,
        "posthoc_ema_snapshots": {
            "steps": outcome.snapshots,
            "directory": str(snapshot_dir.resolve()),
            "scope": (
                "secondary and exploratory; the declared 0.9999 EMA is the "
                "primary result and no EMA horizon may be selected on a metric "
                "computed from the sealed test split"
            ),
        },
        "train_only_gate": gate,
        "elapsed_seconds": time.time() - started,
        "limits": [
            "One developmental capability unit; no replication claim.",
            "The train-only gate uses no test image; the sealed evaluation is "
            "a separate step run after this artifact is hashed.",
            "Step 160000 is the result. No intermediate checkpoint may be "
            "selected on any metric.",
        ],
    }
    digest = write_json(args.out, result)
    print(f"CAP-EMF-1 train-only verdict={gate['verdict']}")
    print(f"wrote {args.out} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
