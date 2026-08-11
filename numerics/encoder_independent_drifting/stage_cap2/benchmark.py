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
from ..stage_cap.training import load_recovery_payload, train_cap_unit
from .artifacts import (
    assert_unused,
    profile_payload,
    save_checkpoint,
    save_snapshot,
    source_manifest,
    write_json_atomic,
    write_sha256_sidecar_atomic,
)
from .config import screen_profile
from .durable_mirror import DurableMirror
from .hardware import hardware_binding
from .preview import save_fixed_grid


def _optimizer_step_summary(optimizer: object) -> dict[str, int]:
    """Summarize Adam's serialized per-parameter update counters."""
    if not isinstance(optimizer, dict) or not isinstance(optimizer.get("state"), dict):
        raise TypeError("recovery optimizer state is malformed")
    steps: list[int] = []
    for state in optimizer["state"].values():
        if not isinstance(state, dict) or "step" not in state:
            continue
        value = state["step"]
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError("optimizer step tensor is not scalar")
            value = value.item()
        step = int(value)
        if float(value) != step or step < 0:
            raise ValueError("optimizer step counter is invalid")
        steps.append(step)
    if not steps:
        raise RuntimeError("recovery contains no Adam step counters")
    return {"count": len(steps), "minimum": min(steps), "maximum": max(steps)}


def _hash64(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def resume_rehearsal_consistent(
    payload: object, *, expected_steps: int | None = None
) -> bool:
    """Recompute the benchmark's stop/reload/continue certificate.

    Merely observing a recovery file is insufficient: the old GPU bug appeared
    only on the first EMA update after loading.  This certificate therefore
    requires both Adam's serialized per-parameter counters and EMA's update
    count to advance *after* the on-device reload.
    """
    if not isinstance(payload, dict):
        return False
    before = payload.get("before_resume")
    after = payload.get("after_resume")
    if not isinstance(before, dict) or not isinstance(after, dict):
        return False
    before_optimizer = before.get("optimizer_steps")
    after_optimizer = after.get("optimizer_steps")
    if not isinstance(before_optimizer, dict) or not isinstance(after_optimizer, dict):
        return False
    try:
        split = int(payload["split_step"])
        final = int(payload["final_step"])
        resumed = int(payload["resumed_updates"])
        before_completed = int(before["completed_updates"])
        after_completed = int(after["completed_updates"])
        before_updates = int(before["optimizer_updates"])
        after_updates = int(after["optimizer_updates"])
        before_ema = int(before["ema_updates"])
        after_ema = int(after["ema_updates"])
        before_count = int(before_optimizer["count"])
        after_count = int(after_optimizer["count"])
        before_min = int(before_optimizer["minimum"])
        before_max = int(before_optimizer["maximum"])
        after_min = int(after_optimizer["minimum"])
        after_max = int(after_optimizer["maximum"])
        before_nonfinite = int(before["nonfinite_updates"])
        after_nonfinite = int(after["nonfinite_updates"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    return (
        0 < split < final
        and (expected_steps is None or final == expected_steps)
        and resumed == final - split
        and before_completed == before_updates == before_ema == split
        and after_completed == after_updates == after_ema == final
        and before_nonfinite == after_nonfinite == 0
        and before_count == after_count
        and before_count > 0
        and before_min == before_max == split
        and after_min == after_max == final
        and after_min - before_min == resumed
        and after_max - before_max == resumed
        and payload.get("first_device") == payload.get("second_device")
        and isinstance(payload.get("first_device"), str)
        and bool(payload["first_device"])
        and payload.get("resume_message") == f"resumed CAP-EMF-1 from update {split}"
        and _hash64(payload.get("before_recovery_sha256"))
        and _hash64(payload.get("after_recovery_sha256"))
        and payload.get("before_recovery_sha256")
        != payload.get("after_recovery_sha256")
    )


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
    parser.add_argument("--durable-mirror-dir", type=Path, required=True)
    parser.add_argument("--i-confirm-durable-mirror", action="store_true")
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
    if not args.i_confirm_durable_mirror:
        raise RuntimeError(
            "benchmark requires confirmation that its provisioned mirror is off-instance"
        )

    production_profile = screen_profile(args.arm, args.numerical, updates=50_000)
    production_train = production_profile.train
    frozen = production_profile
    split_step = args.steps - 1
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
            # Keep the horizon-neutral identity identical across the rehearsal
            # while leaving one real optimizer/EMA update after reload.
            warmup_updates=min(split_step, production_train.warmup_updates),
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
    first_profile = replace(
        frozen,
        train=replace(
            frozen.train,
            updates=split_step,
            checkpoint_updates=(split_step,),
        ),
    )
    first_profile.validate()
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
        mirror = DurableMirror(root, args.durable_mirror_dir)
        mirror_probe = mirror.probe()
        recovery = root / "recovery.pt"
        declared_profile = profile_payload(frozen)
        preflight_sha256 = "b" * 64
        run_identity_sha256 = "e" * 64
        checkpoint_seconds = 0.0
        snapshot_seconds = 0.0
        snapshot_record: dict[str, object] = {}
        recovery_mirror_records: list[dict[str, object]] = []
        recovery_identity = {
            "benchmark": True,
            "hardware": hardware,
            "device": settings,
            "precision": precision,
            "deterministic_algorithms": deterministic,
        }

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
                    "durable_mirror": mirror.mirror(path),
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
                "durable_mirror": mirror.mirror(path),
            }
            snapshot_seconds += time.time() - event_started

        def checkpoint_health_observer(step, raw_components, ema_components):
            records = {}
            for kind, components in (("raw", raw_components), ("ema", ema_components)):
                path = root / f"preview_step{step}_{kind}.png"
                digest = save_fixed_grid(components["final"], path, rows=8, columns=16)
                write_sha256_sidecar_atomic(path, digest)
                records[kind] = {
                    "sha256": digest,
                    "durable_mirror": mirror.mirror(path),
                }
            return {"step": step, "fixed_uncurated_previews": records}

        def recovery_saved(step, path):
            recovery_mirror_records.append(
                mirror.mirror(path, mutable=True, recovery_step=step)
            )

        first_messages: list[str] = []
        first_started = time.time()
        first_outcome = train_cap_unit(
            pool,
            first_profile,
            device,
            recovery_path=recovery,
            checkpoint_health_observer=checkpoint_health_observer,
            recovery_saved=recovery_saved,
            # The first horizon exists only to force a real serialized reload;
            # primary benchmark artifacts are emitted at the final horizon.
            progress=first_messages.append,
            recovery_identity=recovery_identity,
            unit_seed=args.unit_seed,
        )
        first_train_seconds = time.time() - first_started
        before_resume, before_recovery_sha = load_recovery_payload(
            recovery,
            require_sidecar=True,
            validate_counters=True,
        )

        second_messages: list[str] = []
        second_started = time.time()
        outcome = train_cap_unit(
            pool,
            frozen,
            device,
            recovery_path=recovery,
            checkpoint=checkpoint,
            snapshot=snapshot,
            checkpoint_health_observer=checkpoint_health_observer,
            recovery_saved=recovery_saved,
            progress=second_messages.append,
            recovery_identity=recovery_identity,
            unit_seed=args.unit_seed,
        )
        second_train_seconds = time.time() - second_started
        after_resume, after_recovery_sha = load_recovery_payload(
            recovery,
            require_sidecar=True,
            validate_counters=True,
        )
        train_total = first_train_seconds + second_train_seconds
        recovery_bytes = recovery.stat().st_size

        resume_message = next(
            (
                message
                for message in second_messages
                if message.startswith("resumed CAP-EMF-1 from update ")
            ),
            "",
        )
        resume_rehearsal = {
            "split_step": split_step,
            "final_step": args.steps,
            "resumed_updates": args.steps - split_step,
            "first_device": str(device),
            "second_device": str(device),
            "resume_message": resume_message,
            "before_recovery_sha256": before_recovery_sha,
            "after_recovery_sha256": after_recovery_sha,
            "before_resume": {
                "completed_updates": int(before_resume["completed_updates"]),
                "optimizer_updates": int(before_resume["optimizer_updates"]),
                "ema_updates": int(before_resume["ema"]["updates"]),
                "nonfinite_updates": int(before_resume["nonfinite_updates"]),
                "optimizer_steps": _optimizer_step_summary(before_resume["optimizer"]),
            },
            "after_resume": {
                "completed_updates": int(after_resume["completed_updates"]),
                "optimizer_updates": int(after_resume["optimizer_updates"]),
                "ema_updates": int(after_resume["ema"]["updates"]),
                "nonfinite_updates": int(after_resume["nonfinite_updates"]),
                "optimizer_steps": _optimizer_step_summary(after_resume["optimizer"]),
            },
            "first_process_seconds": first_train_seconds,
            "second_process_seconds": second_train_seconds,
        }

    recovery_io_total_seconds = (
        first_outcome.recovery_io_seconds + outcome.recovery_io_seconds
    )
    recovery_events_measured = 2
    ordinary_health_total_seconds = (
        first_outcome.ordinary_health_seconds + outcome.ordinary_health_seconds
    )
    ordinary_health_events_measured = (
        first_outcome.ordinary_health_events + outcome.ordinary_health_events
    )
    checkpoint_health_total_seconds = (
        first_outcome.checkpoint_health_seconds + outcome.checkpoint_health_seconds
    )
    checkpoint_health_events_measured = (
        first_outcome.checkpoint_health_events + outcome.checkpoint_health_events
    )

    checks = {
        "hardware_bound": hardware["matches"],
        "completed": outcome.optimizer_updates == args.steps,
        "finite": outcome.nonfinite_updates == 0,
        "checkpoint_written": str(args.steps) in outcome.checkpoints,
        "snapshot_written": snapshot_record.get("step") == args.steps,
        "recovery_written": recovery_bytes > 0,
        "resume_rehearsed": resume_rehearsal_consistent(
            resume_rehearsal, expected_steps=args.steps
        ),
        "durable_mirror_roundtrip": mirror_probe.get("roundtrip_verified") is True,
        "durable_recoveries_committed": (
            [int(record["recovery_step"]) for record in recovery_mirror_records]
            == [split_step, args.steps]
            and mirror.verify_recovery(recovery, recovery_step=args.steps).get("sha256")
            == after_recovery_sha
        ),
        "parameter_ceiling": outcome.parameter_count <= PARAMETER_CEILING,
    }

    measured_event_seconds = (
        checkpoint_seconds
        + snapshot_seconds
        + recovery_io_total_seconds
        + ordinary_health_total_seconds
        + checkpoint_health_total_seconds
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
    recovery_event_seconds = recovery_io_total_seconds / recovery_events_measured
    if ordinary_health_events_measured < 1 or checkpoint_health_events_measured < 1:
        raise RuntimeError("benchmark did not measure both declared health events")
    ordinary_health_event_seconds = (
        ordinary_health_total_seconds / ordinary_health_events_measured
    )
    checkpoint_health_event_seconds = (
        checkpoint_health_total_seconds / checkpoint_health_events_measured
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
        for updates in (50_000, 150_000, 300_000, 500_000, 650_000, 750_000)
    }
    final_checkpoint = outcome.checkpoints.get(str(args.steps), {})
    checkpoint_artifact_bytes = {
        kind: int(final_checkpoint.get(kind, {}).get("bytes", 0))
        for kind in ("raw", "ema")
    }
    if min(checkpoint_artifact_bytes.values()) <= 0:
        raise RuntimeError("benchmark did not retain measured checkpoint sizes")
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
        "checkpoint_artifact_bytes": checkpoint_artifact_bytes,
        "recovery_io_seconds": recovery_event_seconds,
        "recovery_io_total_seconds": recovery_io_total_seconds,
        "recovery_events_measured": recovery_events_measured,
        "checkpoint_io_seconds": checkpoint_seconds,
        "snapshot_io_seconds": snapshot_seconds,
        "ordinary_health_seconds": ordinary_health_event_seconds,
        "checkpoint_health_seconds": checkpoint_health_event_seconds,
        "ordinary_health_events_measured": ordinary_health_events_measured,
        "checkpoint_health_events_measured": checkpoint_health_events_measured,
        "ordinary_health_samples": production_train.health_samples,
        "checkpoint_health_samples": production_train.audit_samples,
        "snapshot": snapshot_record,
        "resume_rehearsal": resume_rehearsal,
        "durable_mirror": {
            "attestation": mirror.attestation,
            "live_roundtrip_probe": mirror_probe,
            "recovery_commits": recovery_mirror_records,
            "synchronous_event_costs_included": True,
        },
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
