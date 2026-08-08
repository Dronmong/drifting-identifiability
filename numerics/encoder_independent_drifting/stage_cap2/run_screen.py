"""Run one guarded CAP-EMF-2 sampler arm, never a full confirmation."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..stage_cap.config import enable_tf32, examples_seen
from ..stage_cap.data import cifar10_train_pool
from ..stage_cap.diagnostics import capability_gate
from ..stage_cap.training import (
    clip_fraction,
    load_recovery_payload,
    recovery_sidecar,
    train_cap_unit,
)
from .artifacts import (
    assert_unused,
    load_checkpoint,
    load_preflight,
    load_snapshot,
    profile_payload,
    save_checkpoint,
    save_snapshot,
    verify_file,
    verify_json,
    write_json_atomic,
)
from .config import SAMPLER_ARMS, apply_calibrated_gate, screen_profile
from .early_admission import load_early_admission
from .hardware import require_same_hardware
from .promotion import load_promotion
from .selection import load_selection


def _with_batch_split(profile, micro_batch: int | None):
    if micro_batch is None:
        return profile
    effective = profile.train.effective_batch
    if micro_batch <= 0 or effective % micro_batch:
        raise ValueError("microbatch must be positive and divide the effective batch")
    result = replace(
        profile,
        train=replace(
            profile.train,
            micro_batch=micro_batch,
            accumulation_steps=effective // micro_batch,
        ),
    )
    result.validate()
    return result


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _load_recovery(path: Path) -> dict:
    payload, digest = load_recovery_payload(
        path, require_sidecar=True, validate_counters=True
    )
    required = {
        "profile_name",
        "planned_updates",
        "completed_updates",
        "checkpoints",
        "snapshots",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise RuntimeError(f"CAP2 recovery lacks required metadata {missing}")
    planned = int(payload["planned_updates"])
    completed = int(payload["completed_updates"])
    if planned not in {50_000, 100_000, 150_000, 300_000}:
        raise RuntimeError(f"invalid CAP2 recovery horizon {planned}")
    if not 0 <= completed <= planned:
        raise RuntimeError("CAP2 recovery completed_updates lies outside its plan")
    payload["artifact_sha256"] = digest
    return payload


def _assert_state_dict_exact(
    recovery_state: dict, checkpoint_state: dict, *, label: str
) -> None:
    """Require bit-exact tensor state at the 150k promotion boundary."""
    if not isinstance(recovery_state, dict) or not isinstance(checkpoint_state, dict):
        raise TypeError(f"{label} state is not a state dictionary")
    if set(recovery_state) != set(checkpoint_state):
        raise RuntimeError(f"{label} recovery/checkpoint keys differ")
    for name in sorted(recovery_state):
        recovered = recovery_state[name]
        recorded = checkpoint_state[name]
        if not isinstance(recovered, torch.Tensor) or not isinstance(
            recorded, torch.Tensor
        ):
            raise TypeError(f"{label} state entry {name!r} is not a tensor")
        if recovered.dtype != recorded.dtype or recovered.shape != recorded.shape:
            raise RuntimeError(f"{label} state entry {name!r} metadata differs")
        if not torch.equal(recovered.cpu(), recorded.cpu()):
            raise RuntimeError(f"{label} state entry {name!r} differs")


def _assert_recovery_matches_150k_checkpoints(
    recovery: dict, *, raw_checkpoint: dict, ema_checkpoint: dict
) -> None:
    """Prove that the promoted process resumes the exact recorded 150k state."""
    if (
        int(recovery.get("planned_updates", -1)) != 150_000
        or int(recovery.get("completed_updates", -1)) != 150_000
    ):
        raise RuntimeError("state reconciliation is only valid at completed 150k")
    ema = recovery.get("ema")
    if not isinstance(ema, dict):
        raise TypeError("150k recovery lacks EMA state")
    shadow = ema.get("shadow")
    buffers = ema.get("buffers")
    if not isinstance(shadow, dict) or not isinstance(buffers, dict):
        raise TypeError("150k recovery EMA state is malformed")
    if set(shadow) & set(buffers):
        raise RuntimeError("150k recovery EMA shadow/buffer keys overlap")
    merged_ema = {**shadow, **buffers}
    _assert_state_dict_exact(
        recovery.get("model"), raw_checkpoint.get("state_dict"), label="150k raw"
    )
    _assert_state_dict_exact(
        merged_ema, ema_checkpoint.get("state_dict"), label="150k EMA"
    )


def _promotion_recovery_authorization(promotion: dict, selection: dict) -> dict:
    """Small immutable binding copied into every promoted rolling recovery."""
    required = {
        "artifact_sha256",
        "preflight_sha256",
        "result_sha256",
        "checkpoint_sha256",
        "raw_checkpoint_sha256",
        "readmission_sha256",
        "development_evaluation_sha256",
        "arm",
        "candidate",
    }
    missing = sorted(required - set(promotion))
    if missing:
        raise RuntimeError(f"CAP2 promotion lacks recovery bindings {missing}")
    if not isinstance(selection.get("artifact_sha256"), str):
        raise TypeError("CAP2 selection lacks its artifact binding")
    return {
        "status": "cap-emf2-300k-recovery-authorization",
        "promotion_sha256": promotion["artifact_sha256"],
        "preflight_sha256": promotion["preflight_sha256"],
        "result_150k_sha256": promotion["result_sha256"],
        "checkpoint_150k_ema_sha256": promotion["checkpoint_sha256"],
        "checkpoint_150k_raw_sha256": promotion["raw_checkpoint_sha256"],
        "readmission_sha256": promotion["readmission_sha256"],
        "development_evaluation_sha256": promotion["development_evaluation_sha256"],
        "selection_sha256": selection["artifact_sha256"],
        "arm": promotion["arm"],
        "candidate": promotion["candidate"],
        "from_updates": 150_000,
        "to_updates": 300_000,
    }


def _assert_recovery_authorization(recovery: dict, expected: dict) -> None:
    if recovery.get("continuation_authorization") != expected:
        raise RuntimeError("interrupted 300k recovery is not bound to this promotion")


def _assert_result_binds_recovery(
    result: dict, recovery: dict, *, recovery_path: Path, root: Path
) -> None:
    """Bind optimizer/RNG/counters, not merely weights, into the 150k result."""
    record = result.get("recovery")
    expected = {
        "path": _relative(recovery_path, root),
        "sha256": recovery.get("artifact_sha256"),
        "planned_updates": 150_000,
        "completed_updates": 150_000,
        "continuation_authorization": None,
    }
    if record != expected:
        raise RuntimeError("immutable 150k result does not bind the promoted recovery")


def _validate_recovery_request(
    recovery: dict | None, *, requested: int, expected_profile_name: str
) -> None:
    if recovery is None:
        if requested == 300_000:
            raise RuntimeError("CAP2 forbids a fresh 300k run")
        if requested > 50_000:
            raise RuntimeError(
                "CAP2 must start at 50k so the trained raw state can receive "
                "checkpoint-specific numerical admission"
            )
        return
    if recovery.get("profile_name") != expected_profile_name:
        raise RuntimeError("CAP2 recovery belongs to another arm/candidate")
    planned = int(recovery["planned_updates"])
    completed = int(recovery["completed_updates"])
    if requested == 300_000:
        if planned == 150_000 and completed == 150_000:
            return
        if planned == 300_000 and 150_000 <= completed < 300_000:
            return
        raise RuntimeError(
            "300k requires a completed 150k recovery or an interrupted promoted 300k recovery"
        )
    if planned > requested:
        raise RuntimeError("CAP2 recovery cannot move to a shorter horizon")
    if planned < requested and completed != planned:
        raise RuntimeError("finish the existing CAP2 horizon before extending it")


def _ensure_run_identity(path: Path, payload: dict, *, dirty: bool) -> str:
    if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
        recorded = verify_json(path, "cap-emf2-run-identity")
        digest = recorded.pop("artifact_sha256")
        if recorded != payload:
            raise RuntimeError("CAP2 run identity changed across resume/promotion")
        return digest
    if dirty:
        raise RuntimeError("CAP2 run files exist without their immutable run identity")
    return write_json_atomic(path, payload)


def _validate_existing_artifacts(
    *,
    root: Path,
    recovery: dict | None,
    arm: str,
    candidate: str,
    preflight: dict,
    calibration: dict,
    realized_micro_batch: int,
    run_identity_sha256: str,
    unit_seed: int,
) -> None:
    checkpoints = root / "checkpoints"
    snapshots = root / "raw_snapshots"
    all_checkpoint_files = sorted(checkpoints.glob("cap2_*_step*_*.pt"))
    all_snapshot_files = sorted(snapshots.glob("cap2_*_snapshot_step*.pt"))
    unexpected_checkpoints = sorted(
        set(checkpoints.glob("*.pt"))
        - set(all_checkpoint_files)
        - {checkpoints / "recovery.pt"}
    )
    unexpected_snapshots = sorted(set(snapshots.glob("*.pt")) - set(all_snapshot_files))
    if unexpected_checkpoints or unexpected_snapshots:
        unexpected = (unexpected_checkpoints + unexpected_snapshots)[0]
        raise RuntimeError(f"unrecognized CAP2 torch artifact: {unexpected}")
    checkpoint_files = sorted(checkpoints.glob(f"cap2_{arm}_step*_*.pt"))
    snapshot_files = sorted(snapshots.glob(f"cap2_{arm}_snapshot_step*.pt"))
    if set(all_checkpoint_files) != set(checkpoint_files):
        raise RuntimeError("CAP2 output directory contains another arm's checkpoint")
    if set(all_snapshot_files) != set(snapshot_files):
        raise RuntimeError("CAP2 output directory contains another arm's snapshot")
    orphan_sidecars = [
        sidecar
        for directory in (checkpoints, snapshots)
        if directory.exists()
        for sidecar in directory.glob("*.pt.sha256")
        if not Path(str(sidecar)[: -len(".sha256")]).is_file()
    ]
    if orphan_sidecars:
        raise RuntimeError(f"orphan CAP2 SHA sidecar: {orphan_sidecars[0]}")
    if recovery is None:
        if checkpoint_files or snapshot_files:
            raise RuntimeError(
                "fresh CAP2 run points at stale checkpoint/snapshot files"
            )
        return

    completed = int(recovery["completed_updates"])
    recorded_checkpoints = recovery.get("checkpoints", {})
    seen: set[tuple[int, str]] = set()
    for path in checkpoint_files:
        payload = load_checkpoint(
            path,
            arm=arm,
            preflight_sha256=preflight["artifact_sha256"],
            run_identity_sha256=run_identity_sha256,
            unit_seed=unit_seed,
        )
        step = int(payload["step"])
        kind = payload["kind"]
        if step > completed:
            raise RuntimeError(f"stale checkpoint is newer than recovery: {path}")
        record = recorded_checkpoints.get(str(step), {}).get(kind)
        if (
            not isinstance(record, dict)
            or record.get("sha256") != payload["artifact_sha256"]
            or record.get("path") != _relative(path, root)
        ):
            raise RuntimeError(f"checkpoint is not recorded by recovery: {path}")
        horizon = int(payload["declared_profile"]["train"]["updates"])
        if horizon not in {50_000, 100_000, 150_000, 300_000} or step > horizon:
            raise RuntimeError(
                f"checkpoint carries an invalid declared horizon: {path}"
            )
        declared_profile = apply_calibrated_gate(
            screen_profile(arm, candidate, updates=horizon), calibration
        )
        realized_profile = _with_batch_split(declared_profile, realized_micro_batch)
        if payload["declared_profile"] != profile_payload(declared_profile):
            raise RuntimeError(f"checkpoint declared profile mismatch: {path}")
        if payload["realized_profile"] != profile_payload(realized_profile):
            raise RuntimeError(f"checkpoint realized profile mismatch: {path}")
        seen.add((step, kind))
    for step, kinds in recorded_checkpoints.items():
        for kind in kinds:
            if (int(step), kind) not in seen:
                raise RuntimeError(
                    f"recovery-recorded checkpoint is missing: {step}/{kind}"
                )

    recorded_snapshots = {int(step) for step in recovery.get("snapshots", [])}
    seen_snapshots: set[int] = set()
    for path in snapshot_files:
        payload = load_snapshot(
            path,
            arm=arm,
            preflight_sha256=preflight["artifact_sha256"],
            run_identity_sha256=run_identity_sha256,
            unit_seed=unit_seed,
        )
        step = int(payload["step"])
        if step > completed or step not in recorded_snapshots:
            raise RuntimeError(f"snapshot is stale or unrecorded by recovery: {path}")
        horizon = int(payload["declared_profile"]["train"]["updates"])
        if horizon not in {50_000, 100_000, 150_000, 300_000} or step > horizon:
            raise RuntimeError(f"snapshot carries an invalid declared horizon: {path}")
        declared_profile = apply_calibrated_gate(
            screen_profile(arm, candidate, updates=horizon), calibration
        )
        realized_profile = _with_batch_split(declared_profile, realized_micro_batch)
        if payload["declared_profile"] != profile_payload(declared_profile):
            raise RuntimeError(f"snapshot declared profile mismatch: {path}")
        if payload["realized_profile"] != profile_payload(realized_profile):
            raise RuntimeError(f"snapshot realized profile mismatch: {path}")
        seen_snapshots.add(step)
    if recorded_snapshots != seen_snapshots:
        raise RuntimeError("one or more recovery-recorded snapshots are missing")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(SAMPLER_ARMS), required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument(
        "--updates",
        type=int,
        choices=(50_000, 100_000, 150_000, 300_000),
        default=150_000,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--promotion",
        type=Path,
        default=None,
        help="immutable 150k promotion certificate; required only for 300k",
    )
    parser.add_argument(
        "--early-admission",
        type=Path,
        default=None,
        help="immutable raw-state 50k admission; required for 100k/150k",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
        help="immutable concurrent three-arm selection; required only for 300k",
    )
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument(
        "--unit-seed",
        type=int,
        default=0,
        help="nonnegative matched-unit seed; part of recovery and result identity",
    )
    parser.add_argument("--nondeterministic", action="store_true")
    parser.add_argument("--i-have-authorized-the-screen-run", action="store_true")
    parser.add_argument("--i-have-authorized-the-300k-promotion", action="store_true")
    args = parser.parse_args()
    if args.unit_seed < 0:
        raise SystemExit("--unit-seed must be nonnegative")
    if not args.i_have_authorized_the_screen_run:
        raise SystemExit("CAP2 refuses to train without explicit screen authorization")
    if args.updates == 300_000:
        if not args.i_have_authorized_the_300k_promotion:
            raise SystemExit(
                "300k is a promoted continuation and needs its explicit flag"
            )
        if args.promotion is None or args.selection is None:
            raise SystemExit(
                "300k requires immutable --promotion and --selection artifacts"
            )
    elif args.promotion is not None or args.selection is not None:
        raise SystemExit(
            "--promotion and --selection are valid only for the 300k continuation"
        )
    if args.updates in {100_000, 150_000} and args.early_admission is None:
        raise SystemExit("100k/150k continuation requires --early-admission")
    if args.updates in {50_000, 300_000} and args.early_admission is not None:
        raise SystemExit("--early-admission is valid only for 100k/150k continuation")

    preflight = load_preflight(args.preflight)
    candidate = preflight["candidate"]
    calibration = preflight["inputs"]["gate_calibration"]
    frozen_150k = apply_calibrated_gate(
        screen_profile(args.arm, candidate, updates=150_000), calibration
    )
    if profile_payload(frozen_150k) != preflight["profiles_150k"][args.arm]:
        raise RuntimeError("live arm differs from the source-bound preflight profile")
    declared = apply_calibrated_gate(
        screen_profile(args.arm, candidate, updates=args.updates), calibration
    )
    declared_payload = profile_payload(declared)
    frozen = _with_batch_split(declared, args.micro_batch)
    realized_payload = profile_payload(frozen)

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(not args.nondeterministic)
    device = resolve_device(args.device)
    live_hardware = require_same_hardware(
        device, preflight["inputs"]["numerical_admission"]["hardware"]
    )
    device_settings = configure(device, allow_tf32=device.type == "cuda")
    precision = enable_tf32()
    benchmark = preflight["inputs"]["benchmark"]
    benchmark_checks = {
        "micro_batch": int(benchmark.get("micro_batch", -1))
        == frozen.train.micro_batch,
        "accumulation_steps": int(benchmark.get("accumulation_steps", -1))
        == frozen.train.accumulation_steps,
        "effective_batch": int(benchmark.get("effective_batch", -1))
        == frozen.train.effective_batch,
        "unit_seed": int(benchmark.get("unit_seed", -1)) == args.unit_seed,
        "deterministic_algorithms": benchmark.get("deterministic_algorithms")
        == (not args.nondeterministic),
        "precision": benchmark.get("precision") == precision,
        "torch_version": benchmark.get("device", {}).get("torch_version")
        == device_settings.get("torch_version"),
        "cuda_version": benchmark.get("device", {}).get("cuda_version")
        == device_settings.get("cuda_version"),
        "allow_tf32": benchmark.get("device", {}).get("allow_tf32")
        == device_settings.get("allow_tf32"),
        "gpu_name": benchmark.get("device", {}).get("gpu_name")
        == device_settings.get("gpu_name"),
    }
    failed_benchmark = sorted(name for name, ok in benchmark_checks.items() if not ok)
    if failed_benchmark:
        raise RuntimeError(
            f"live CAP2 execution differs from its cost/numerical benchmark: {failed_benchmark}"
        )

    root = args.output_dir
    result_path = root / f"result_{args.updates}.json"
    assert_unused(result_path)
    checkpoints = root / "checkpoints"
    snapshots = root / "raw_snapshots"
    recovery = checkpoints / "recovery.pt"
    if recovery.exists() != recovery_sidecar(recovery).exists():
        raise RuntimeError("CAP2 recovery and its SHA sidecar must exist together")
    recovery_payload = _load_recovery(recovery) if recovery.exists() else None
    _validate_recovery_request(
        recovery_payload,
        requested=args.updates,
        expected_profile_name=frozen.name,
    )

    identity_profile = _with_batch_split(frozen_150k, args.micro_batch)
    identity_payload = {
        "status": "cap-emf2-run-identity",
        "arm": args.arm,
        "candidate": candidate,
        "preflight_sha256": preflight["artifact_sha256"],
        # Stable across 50/100/150 and the certified 150->300 extension.
        "profile_150k_realized": profile_payload(identity_profile),
        "unit_seed": args.unit_seed,
        "deterministic_algorithms": not args.nondeterministic,
        "precision": precision,
        "device": device_settings,
        "hardware": live_hardware,
        "benchmark_sha256": benchmark.get("artifact_sha256"),
    }
    any_run_files = bool(
        recovery.exists()
        or recovery_sidecar(recovery).exists()
        or (checkpoints.exists() and any(checkpoints.glob("cap2_*.pt")))
        or (snapshots.exists() and any(snapshots.glob("*.pt")))
    )
    identity_sha = _ensure_run_identity(
        root / "run_identity.json", identity_payload, dirty=any_run_files
    )
    _validate_existing_artifacts(
        root=root,
        recovery=recovery_payload,
        arm=args.arm,
        candidate=candidate,
        preflight=preflight,
        calibration=calibration,
        realized_micro_batch=frozen.train.micro_batch,
        run_identity_sha256=identity_sha,
        unit_seed=args.unit_seed,
    )

    early_admission_record = None
    if args.updates in {100_000, 150_000}:
        result_50k = root / "result_50000.json"
        immutable_50k = verify_json(result_50k, "cap-emf2-screen-unit")
        early_admission_record = load_early_admission(
            args.early_admission, arm=args.arm, candidate=candidate
        )
        if (
            early_admission_record.get("preflight_sha256")
            != preflight["artifact_sha256"]
            or early_admission_record.get("result_sha256")
            != immutable_50k["artifact_sha256"]
            or immutable_50k.get("run_identity_sha256") != identity_sha
            or int(immutable_50k.get("unit_seed", -1)) != args.unit_seed
        ):
            raise RuntimeError(
                "CAP2 early admission is not bound to this exact 50k run"
            )

    recovery_authorization = None
    if args.updates == 300_000:
        result_150k = root / "result_150000.json"
        immutable_150k = verify_json(result_150k, "cap-emf2-screen-unit")
        early_admission_record = immutable_150k.get("early_admission")
        checkpoint_150k_raw = checkpoints / f"cap2_{args.arm}_step150000_raw.pt"
        checkpoint_150k_ema = checkpoints / f"cap2_{args.arm}_step150000_ema.pt"
        promotion = load_promotion(
            args.promotion,
            preflight_path=args.preflight,
            result_path=result_150k,
            raw_checkpoint_path=checkpoint_150k_raw,
            checkpoint_path=checkpoint_150k_ema,
            arm=args.arm,
            candidate=candidate,
        )
        selection = load_selection(
            args.selection,
            promotion_path=args.promotion,
            arm=args.arm,
            candidate=candidate,
        )
        recovery_authorization = _promotion_recovery_authorization(
            promotion, selection
        )
        if int(recovery_payload["planned_updates"]) == 150_000:
            _assert_result_binds_recovery(
                immutable_150k,
                recovery_payload,
                recovery_path=recovery,
                root=root,
            )
            records = recovery_payload["checkpoints"].get("150000", {})
            raw_record = records.get("raw", {})
            ema_record = records.get("ema", {})
            raw_payload = load_checkpoint(
                checkpoint_150k_raw,
                expected_sha=raw_record.get("sha256"),
                step=150_000,
                kind="raw",
                arm=args.arm,
                preflight_sha256=preflight["artifact_sha256"],
                run_identity_sha256=identity_sha,
                unit_seed=args.unit_seed,
            )
            ema_payload = load_checkpoint(
                checkpoint_150k_ema,
                expected_sha=ema_record.get("sha256"),
                step=150_000,
                kind="ema",
                arm=args.arm,
                preflight_sha256=preflight["artifact_sha256"],
                run_identity_sha256=identity_sha,
                unit_seed=args.unit_seed,
            )
            _assert_recovery_matches_150k_checkpoints(
                recovery_payload,
                raw_checkpoint=raw_payload,
                ema_checkpoint=ema_payload,
            )
            if recovery_payload.get("continuation_authorization") is not None:
                raise RuntimeError("150k recovery already carries promotion authority")
        else:
            _assert_recovery_authorization(
                recovery_payload, recovery_authorization
            )

    pool = cifar10_train_pool(args.data_root)

    def checkpoint(step: int, raw: dict, ema: dict) -> dict:
        entry = {}
        for kind, state in (("raw", raw), ("ema", ema)):
            path = checkpoints / f"cap2_{args.arm}_step{step}_{kind}.pt"
            entry[kind] = {
                "path": _relative(path, root),
                "sha256": save_checkpoint(
                    path,
                    state,
                    step=step,
                    kind=kind,
                    arm=args.arm,
                    declared_profile=declared_payload,
                    realized_profile=realized_payload,
                    preflight_sha256=preflight["artifact_sha256"],
                    run_identity_sha256=identity_sha,
                    unit_seed=args.unit_seed,
                ),
            }
        return entry

    def snapshot(step: int, state: dict) -> None:
        snapshots.mkdir(parents=True, exist_ok=True)
        path = snapshots / f"cap2_{args.arm}_snapshot_step{step}.pt"
        save_snapshot(
            path,
            state,
            step=step,
            arm=args.arm,
            declared_profile=declared_payload,
            realized_profile=realized_payload,
            preflight_sha256=preflight["artifact_sha256"],
            run_identity_sha256=identity_sha,
            unit_seed=args.unit_seed,
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
        recovery_identity=identity_payload,
        recovery_authorization=recovery_authorization,
        unit_seed=args.unit_seed,
    )
    final_recovery, final_recovery_sha = load_recovery_payload(
        recovery, require_sidecar=True, validate_counters=True
    )
    if (
        int(final_recovery["planned_updates"]) != args.updates
        or int(final_recovery["completed_updates"]) != outcome.optimizer_updates
    ):
        raise RuntimeError("final CAP2 recovery disagrees with the completed outcome")
    if final_recovery.get("continuation_authorization") != recovery_authorization:
        raise RuntimeError("final CAP2 recovery lost its continuation authorization")
    final_record = outcome.health[-1]
    final = final_record.get("ema", final_record)
    ema_records = [record["ema"] for record in outcome.health if "ema" in record]
    if not ema_records:
        raise RuntimeError("no EMA checkpoint health was recorded")
    best_ema_rank = max(float(record["effective_rank_ratio"]) for record in ema_records)
    gate = capability_gate(
        final,
        best_ema_rank,
        clip_fraction(outcome),
        outcome.nonfinite_updates,
        1,
        frozen.gate,
    )
    snapshot_records = [
        {
            "step": int(path.stem.rsplit("step", 1)[1]),
            "path": _relative(path, root),
            "sha256": verify_file(path),
        }
        for path in sorted(
            snapshots.glob("*.pt"),
            key=lambda candidate_path: int(candidate_path.stem.rsplit("step", 1)[1]),
        )
    ]
    result = {
        "status": "cap-emf2-screen-unit",
        "development_only": True,
        "arm": args.arm,
        "numerical_candidate": candidate,
        "preflight_sha256": preflight["artifact_sha256"],
        "run_identity_sha256": identity_sha,
        "unit_seed": args.unit_seed,
        "declared_profile": declared_payload,
        "realized_profile": realized_payload,
        "realized_batch_split": {
            "micro_batch": frozen.train.micro_batch,
            "accumulation_steps": frozen.train.accumulation_steps,
            "effective_batch": frozen.train.effective_batch,
            "overridden": args.micro_batch is not None,
        },
        "precision": precision,
        "deterministic_algorithms": not args.nondeterministic,
        "device": device_settings,
        "hardware": live_hardware,
        "training": {
            "history": outcome.history,
            "health": outcome.health,
            "optimizer_updates": outcome.optimizer_updates,
            "examples_seen": outcome.examples_seen,
            "examples_seen_target": examples_seen(frozen.train),
            "objective_sample_evaluations": outcome.model_forwards,
            "objective_forward_calls": outcome.objective_forward_calls,
            "clipped_updates": outcome.clipped_updates,
            "clipped_updates_final_window": outcome.clipped_updates_final_window,
            "final_window_updates": outcome.final_window_updates,
            "clip_fraction_final_window": clip_fraction(outcome),
            "nonfinite_updates": outcome.nonfinite_updates,
            "wall_seconds": outcome.wall_seconds,
            "peak_memory_bytes": outcome.peak_memory_bytes,
            "peak_memory_reserved_bytes": outcome.peak_memory_reserved_bytes,
        },
        "checkpoints": outcome.checkpoints,
        "raw_snapshots": snapshot_records,
        "recovery": {
            "path": _relative(recovery, root),
            "sha256": final_recovery_sha,
            "planned_updates": int(final_recovery["planned_updates"]),
            "completed_updates": int(final_recovery["completed_updates"]),
            "continuation_authorization": final_recovery.get(
                "continuation_authorization"
            ),
        },
        "early_admission": (
            {
                "path": _relative(args.early_admission, root),
                "sha256": early_admission_record["artifact_sha256"],
            }
            if args.updates in {100_000, 150_000}
            else early_admission_record
        ),
        "train_only_gate": gate,
        "elapsed_seconds": time.time() - started,
        "next_step": (
            "eligible for fixed 300k promotion review"
            if args.updates == 150_000 and gate["verdict"] == "PASS"
            else "requires comparative review; never auto-promote to confirmation"
        ),
        "limits": [
            "No CIFAR-10 test image is opened by this runner.",
            "One developmental arm and seed; no general performance claim.",
            "This runner cannot launch a 600k-scale confirmation or ASFD.",
        ],
    }
    digest = write_json_atomic(result_path, result)
    print(json.dumps({"arm": args.arm, "gate": gate}, indent=2))
    print(f"wrote {result_path} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
