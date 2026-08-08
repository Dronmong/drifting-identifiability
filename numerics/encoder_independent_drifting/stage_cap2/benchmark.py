"""Benchmark the complete CAP2 loop, including recovery and checkpoint I/O."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..stage_cap.config import PARAMETER_CEILING, enable_tf32
from ..stage_cap.data import cifar10_train_pool
from ..stage_cap.training import train_cap_unit
from .artifacts import (
    assert_unused,
    profile_payload,
    save_checkpoint,
    save_snapshot,
    source_manifest,
    write_json_atomic,
)
from .config import screen_profile
from .hardware import hardware_binding


def project_runtime(
    *,
    updates: int,
    recovery_every: int,
    snapshot_every: int,
    health_every: int,
    non_io_seconds_per_update: float,
    raw_upper_seconds_per_update: float,
    recovery_event_seconds: float,
    snapshot_event_seconds: float,
    checkpoint_pair_seconds: float,
    ordinary_health_event_seconds: float,
    checkpoint_health_event_seconds: float,
    hourly_rate: float,
) -> dict:
    """Cadence-adjust one measured event of every production path."""
    if updates <= 0 or min(recovery_every, snapshot_every, health_every) <= 0:
        raise ValueError("runtime projection cadences must be positive")
    recovery_events = math.ceil(updates / recovery_every)
    snapshot_events = updates // snapshot_every
    checkpoint_events = updates // 50_000
    all_health_events = updates // health_every
    ordinary_health_events = all_health_events - checkpoint_events
    if ordinary_health_events < 0:
        raise RuntimeError("checkpoint health cadence exceeds all health events")
    estimated_seconds = (
        non_io_seconds_per_update * updates
        + recovery_events * recovery_event_seconds
        + snapshot_events * snapshot_event_seconds
        + checkpoint_events * checkpoint_pair_seconds
        + ordinary_health_events * ordinary_health_event_seconds
        + checkpoint_events * checkpoint_health_event_seconds
    )
    hours = estimated_seconds / 3600
    conservative_upper_hours = raw_upper_seconds_per_update * updates / 3600
    return {
        "hours": hours,
        "cost_at_declared_rate": hours * hourly_rate,
        "conservative_raw_loop_upper_hours": conservative_upper_hours,
        "conservative_raw_loop_upper_cost": conservative_upper_hours * hourly_rate,
        "event_counts": {
            "recovery": recovery_events,
            "snapshot": snapshot_events,
            "checkpoint_pair": checkpoint_events,
            "ordinary_health": ordinary_health_events,
            "checkpoint_health": checkpoint_events,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=("legacy", "ordered_logitnormal", "ordered_uniform"),
        default="ordered_uniform",
    )
    parser.add_argument("--numerical", default="local_1000_d0002_fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--unit-seed", type=int, default=0)
    parser.add_argument("--nondeterministic", action="store_true")
    parser.add_argument("--hourly-rate", type=float, default=0.75)
    parser.add_argument(
        "--expected-gpu-name",
        required=True,
        help="case-insensitive substring of the intended training GPU model",
    )
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).with_name("benchmark.json")
    )
    args = parser.parse_args()
    assert_unused(args.out)
    if args.steps < 2_000:
        raise ValueError(
            "full-loop benchmark must contain at least 2,000 updates so its "
            "logging and separately timed health paths are both exercised"
        )
    if args.steps % 2:
        raise ValueError("full-loop benchmark steps must be even")
    if args.unit_seed < 0:
        raise ValueError("unit seed must be nonnegative")
    if args.hourly_rate <= 0:
        raise ValueError("hourly GPU rate must be positive")

    production_profile = screen_profile(args.arm, args.numerical, updates=50_000)
    production_train = production_profile.train
    frozen = production_profile
    effective = production_train.effective_batch
    if effective % args.micro_batch:
        raise ValueError("microbatch must divide the effective batch")
    frozen = replace(
        frozen,
        name=f"{frozen.name}-benchmark-{args.steps}",
        train=replace(
            frozen.train,
            updates=args.steps,
            micro_batch=args.micro_batch,
            accumulation_steps=effective // args.micro_batch,
            warmup_updates=min(args.steps, production_train.warmup_updates),
            log_every=production_train.log_every,
            # One ordinary event at 1k and one checkpoint event at 2k let the
            # benchmark measure both production-sized health paths, then add
            # them back at their true (2k / 50k) production cadences.
            health_every=args.steps // 2,
            health_samples=production_train.health_samples,
            audit_samples=production_train.audit_samples,
            checkpoint_updates=(args.steps,),
            snapshot_every=args.steps,
            recovery_every=args.steps,
        ),
    )
    frozen.validate()
    device = resolve_device(args.device)
    hardware = hardware_binding(device, args.expected_gpu_name)
    settings = configure(device, allow_tf32=device.type == "cuda")
    precision = enable_tf32()
    torch.set_num_threads(4)
    deterministic = not args.nondeterministic
    torch.use_deterministic_algorithms(deterministic)
    started = time.time()
    pool = cifar10_train_pool(args.data_root)
    data_seconds = time.time() - started

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        recovery = root / "recovery.pt"
        declared_profile = profile_payload(frozen)
        preflight_sha256 = "b" * 64
        run_identity_sha256 = "e" * 64
        checkpoint_seconds = 0.0
        snapshot_seconds = 0.0
        snapshot_record: dict[str, object] = {}

        def checkpoint(step, raw, ema):
            nonlocal checkpoint_seconds
            event_started = time.time()
            records = {}
            for kind, state in (("raw", raw), ("ema", ema)):
                path = root / f"{kind}.pt"
                digest = save_checkpoint(
                    path,
                    state,
                    step=step,
                    kind=kind,
                    arm=args.arm,
                    declared_profile=declared_profile,
                    realized_profile=declared_profile,
                    preflight_sha256=preflight_sha256,
                    run_identity_sha256=run_identity_sha256,
                    unit_seed=args.unit_seed,
                )
                records[kind] = {
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
            checkpoint_seconds += time.time() - event_started
            return records

        def snapshot(step, state):
            nonlocal snapshot_seconds, snapshot_record
            event_started = time.time()
            path = root / "raw_snapshot.pt"
            digest = save_snapshot(
                path,
                state,
                step=step,
                arm=args.arm,
                declared_profile=declared_profile,
                realized_profile=declared_profile,
                preflight_sha256=preflight_sha256,
                run_identity_sha256=run_identity_sha256,
                unit_seed=args.unit_seed,
            )
            snapshot_record = {
                "step": int(step),
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
            snapshot_seconds += time.time() - event_started

        train_started = time.time()
        outcome = train_cap_unit(
            pool,
            frozen,
            device,
            recovery_path=recovery,
            checkpoint=checkpoint,
            snapshot=snapshot,
            recovery_identity={
                "benchmark": True,
                "hardware": hardware,
                "device": settings,
                "precision": precision,
                "deterministic_algorithms": deterministic,
            },
            unit_seed=args.unit_seed,
        )
        train_total = time.time() - train_started
        recovery_bytes = recovery.stat().st_size

    checks = {
        "hardware_bound": hardware["matches"],
        "completed": outcome.optimizer_updates == args.steps,
        "finite": outcome.nonfinite_updates == 0,
        "checkpoint_written": str(args.steps) in outcome.checkpoints,
        "snapshot_written": snapshot_record.get("step") == args.steps,
        "recovery_written": recovery_bytes > 0,
        "parameter_ceiling": outcome.parameter_count <= PARAMETER_CEILING,
    }

    measured_event_seconds = (
        checkpoint_seconds
        + snapshot_seconds
        + outcome.recovery_io_seconds
        + outcome.ordinary_health_seconds
        + outcome.checkpoint_health_seconds
    )
    # The benchmark deliberately exercises every artifact path once, ordinary
    # logging at production cadence, and both production-sized health paths.
    # Treating those rare events as if they happened every 300 updates would
    # overstate a real run by one to two orders of magnitude.  Remove the
    # measured I/O and health events, then add them back at their declared
    # production cadences. Ordinary logging stays in the per-update term
    # because the minimum 2,000-step benchmark contains exactly that cadence.
    non_io_seconds = max(0.0, train_total - measured_event_seconds)
    non_io_seconds_per_update = non_io_seconds / args.steps
    raw_upper_seconds_per_update = train_total / args.steps
    recovery_event_seconds = outcome.recovery_io_seconds
    if outcome.ordinary_health_events != 1 or outcome.checkpoint_health_events != 1:
        raise RuntimeError("benchmark did not measure both declared health events")
    ordinary_health_event_seconds = (
        outcome.ordinary_health_seconds / outcome.ordinary_health_events
    )
    checkpoint_health_event_seconds = (
        outcome.checkpoint_health_seconds / outcome.checkpoint_health_events
    )
    projections = {
        str(updates): project_runtime(
            updates=updates,
            recovery_every=production_train.recovery_every,
            snapshot_every=production_train.snapshot_every,
            health_every=production_train.health_every,
            non_io_seconds_per_update=non_io_seconds_per_update,
            raw_upper_seconds_per_update=raw_upper_seconds_per_update,
            recovery_event_seconds=recovery_event_seconds,
            snapshot_event_seconds=snapshot_seconds,
            checkpoint_pair_seconds=checkpoint_seconds,
            ordinary_health_event_seconds=ordinary_health_event_seconds,
            checkpoint_health_event_seconds=checkpoint_health_event_seconds,
            hourly_rate=args.hourly_rate,
        )
        for updates in (50_000, 150_000, 300_000)
    }
    result = {
        "status": "cap-emf2-full-loop-benchmark",
        "arm": args.arm,
        "numerical": args.numerical,
        "profile": frozen.name,
        "device": settings,
        "hardware": hardware,
        "precision": precision,
        "deterministic_algorithms": deterministic,
        "unit_seed": args.unit_seed,
        "hourly_rate": args.hourly_rate,
        "steps": args.steps,
        "micro_batch": frozen.train.micro_batch,
        "accumulation_steps": frozen.train.accumulation_steps,
        "effective_batch": frozen.train.effective_batch,
        "data_load_seconds": data_seconds,
        "training_and_io_seconds": train_total,
        "projection_method": (
            "measured non-I/O loop time plus measured artifact events at "
            "declared production cadences; residual loop term conservatively "
            "contains monitoring at its production cadence"
        ),
        "non_io_seconds_per_update": non_io_seconds_per_update,
        "raw_full_loop_seconds_per_update": raw_upper_seconds_per_update,
        "recovery_bytes": recovery_bytes,
        "recovery_io_seconds": outcome.recovery_io_seconds,
        "checkpoint_io_seconds": checkpoint_seconds,
        "snapshot_io_seconds": snapshot_seconds,
        "ordinary_health_seconds": ordinary_health_event_seconds,
        "checkpoint_health_seconds": checkpoint_health_event_seconds,
        "ordinary_health_samples": production_train.health_samples,
        "checkpoint_health_samples": production_train.audit_samples,
        "snapshot": snapshot_record,
        "objective_sample_evaluations": outcome.model_forwards,
        "objective_forward_calls": outcome.objective_forward_calls,
        "parameter_count": outcome.parameter_count,
        "peak_memory_bytes": outcome.peak_memory_bytes,
        "peak_memory_reserved_bytes": outcome.peak_memory_reserved_bytes,
        "projections": projections,
        "checks": checks,
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "limits": [
            "Projection is tied to this exact device, numerical mode, and batch split.",
            "The cadence-adjusted estimate is not a confidence interval; the raw-loop projection is retained as a conservative upper bound.",
            "A benchmark on a GPU unlike the declared training GPU cannot emit GO.",
            "Provider startup, sample evaluation, and artifact upload are not included.",
        ],
        "source_sha256": source_manifest(),
    }
    digest = write_json_atomic(args.out, result)
    print(json.dumps(result, indent=2))
    print(f"wrote {args.out} sha256={digest}")
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
