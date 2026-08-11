"""Run the single 750k -> 800k ASFD continuation with durable recovery."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..stage_cap.data import cifar10_train_pool
from ..stage_cap.training import load_recovery_payload, train_cap_unit
from ..stage_cap2.artifacts import (
    load_checkpoint,
    load_snapshot,
    profile_payload,
    save_checkpoint,
    save_snapshot,
    verify_json,
    write_json_atomic,
)
from ..stage_cap2.durable_mirror import DurableMirror
from ..stage_cap2.run_screen import (
    _require_durable_workspace,
    _require_live_storage_capacity,
)
from .artifacts import assert_no_inherited_freeze, source_manifest
from .config import asfd_config
from .correction import ASFDCorrection
from .preflight import continuation_profile
from .recovery import fork_foundation_recovery

STATUS = "asfd-continuation"
WALL_STOP_STATUS = "asfd-continuation-wall-stop"
FINAL_STEP = 800_000
WALL_CONTINGENCY = 0.15


def _require_launch_authorization(
    *, paid: bool, durable_mirror: bool, durable_workspace: bool
) -> None:
    if not paid:
        raise RuntimeError(
            "ASFD refuses paid continuation without explicit operator authorization"
        )
    if not durable_mirror:
        raise RuntimeError("ASFD requires an explicitly confirmed durable mirror")
    if not durable_workspace:
        raise RuntimeError("ASFD requires an explicitly confirmed durable workspace")


def _require_exact_preflight_runtime(preflight: dict, live: dict) -> dict:
    """Bind paid continuation to the runtime that measured its cost and memory.

    A successful 500-update smoke on one CUDA/PyTorch/GPU stack is not evidence
    that a different stack will fit or reproduce it.  Exact equality is
    intentional: changing even the GPU memory class requires a new measured
    preflight, not an operator assertion at launch time.
    """

    recorded = preflight.get("device")
    if not isinstance(recorded, dict) or recorded != live:
        recorded_keys = set(recorded) if isinstance(recorded, dict) else set()
        changed = sorted(
            key
            for key in recorded_keys | set(live)
            if not isinstance(recorded, dict) or recorded.get(key) != live.get(key)
        )
        raise RuntimeError(
            "ASFD continuation runtime differs from its measured preflight: "
            + ", ".join(changed)
        )
    return dict(live)


def _asfd_wall_policy(
    preflight: dict, *, recovery_every: int
) -> dict[str, float | int]:
    """Turn the measured smoke into a recovery-bound continuation wall stop."""

    smoke = preflight.get("measured_smoke")
    if not isinstance(smoke, dict):
        raise TypeError("ASFD preflight lacks its measured smoke record")
    measured_updates = int(smoke.get("updates", -1))
    wall_seconds = float(smoke.get("wall_seconds", math.nan))
    continuation_updates = FINAL_STEP - 750_000
    if measured_updates != 500 or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise RuntimeError("ASFD preflight has an invalid 500-update wall measurement")
    if recovery_every <= 0 or continuation_updates % recovery_every:
        raise ValueError("ASFD recovery cadence does not divide its continuation")
    measured_seconds_per_update = wall_seconds / measured_updates
    hard_seconds = (
        measured_seconds_per_update * continuation_updates * (1.0 + WALL_CONTINGENCY)
    )
    return {
        "measured_updates": measured_updates,
        "measured_wall_seconds": wall_seconds,
        "continuation_updates": continuation_updates,
        "contingency_fraction": WALL_CONTINGENCY,
        "hard_cumulative_continuation_wall_seconds": hard_seconds,
        "recovery_interval_updates": recovery_every,
        "maximum_detection_overshoot_updates": recovery_every,
        "projected_maximum_detection_overshoot_seconds": (
            measured_seconds_per_update * recovery_every
        ),
    }


def _require_declared_storage_root(storage_plan: dict, requested: Path) -> Path:
    """Reject a launch whose mounted volume differs from CAP2 admission."""

    recorded = Path(str(storage_plan.get("storage_root", ""))).resolve()
    live = requested.resolve()
    if recorded != live:
        raise RuntimeError(
            f"ASFD durable storage root differs from preflight: {live} != {recorded}"
        )
    return live


def _resolve(reference: object, anchor: Path) -> Path:
    if not isinstance(reference, str) or not reference:
        raise RuntimeError("ASFD continuation input contains an empty path")
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (anchor / path).resolve()


def _relative(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()


def _identity(payload: dict) -> tuple[dict, str]:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return payload, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _revalidate_terminal_result(
    completed: dict,
    *,
    result_path: Path,
    preflight: dict,
    mirror: DurableMirror,
) -> None:
    """Re-open every terminal scientific artifact before returning success."""
    if (
        completed.get("decision") != "GO"
        or completed.get("source_sha256") != source_manifest()
        or completed.get("preflight", {}).get("sha256") != preflight["artifact_sha256"]
        or int(completed.get("final_step", -1)) != FINAL_STEP
    ):
        raise RuntimeError("existing ASFD result failed terminal revalidation")

    profile = completed.get("profile")
    run_identity = completed.get("run_identity_sha256")
    checkpoint_records = completed.get("checkpoints", {}).get(str(FINAL_STEP))
    if not isinstance(profile, dict) or not isinstance(checkpoint_records, dict):
        raise TypeError("existing ASFD result lacks its terminal checkpoint pair")
    for kind in ("raw", "ema"):
        record = checkpoint_records.get(kind)
        if not isinstance(record, dict):
            raise TypeError(f"existing ASFD result lacks its final {kind} checkpoint")
        path = _resolve(record.get("path"), result_path.parent)
        load_checkpoint(
            path,
            expected_sha=record.get("sha256"),
            step=FINAL_STEP,
            kind=kind,
            arm="EMF-ASFD",
            declared_profile=profile,
            realized_profile=profile,
            preflight_sha256=preflight["artifact_sha256"],
            run_identity_sha256=run_identity,
            unit_seed=0,
        )
        if mirror.verify(path) != record.get("durable_mirror"):
            raise RuntimeError(f"final ASFD {kind} checkpoint mirror binding changed")

    recovery_record = completed.get("recovery")
    if not isinstance(recovery_record, dict):
        raise TypeError("existing ASFD result lacks its terminal recovery")
    recovery_path = _resolve(recovery_record.get("path"), result_path.parent)
    recovery, digest = load_recovery_payload(
        recovery_path,
        require_sidecar=True,
        validate_counters=True,
    )
    if (
        digest != recovery_record.get("sha256")
        or int(recovery.get("planned_updates", -1)) != FINAL_STEP
        or int(recovery.get("completed_updates", -1)) != FINAL_STEP
    ):
        raise RuntimeError("existing ASFD terminal recovery changed or is incomplete")
    if mirror.verify_recovery(
        recovery_path,
        recovery_step=FINAL_STEP,
    ) != recovery_record.get("durable_mirror"):
        raise RuntimeError("final ASFD recovery mirror binding changed")


def _revalidate_wall_stop(
    stopped: dict,
    *,
    stop_path: Path,
    preflight: dict,
    mirror: DurableMirror,
) -> None:
    """Authenticate a durable wall stop so a paid resume cannot bypass it."""

    step = int(stopped.get("recovery_step", -1))
    recovery_record = stopped.get("recovery")
    if (
        stopped.get("decision") != "HALT"
        or stopped.get("reason") != "measured-continuation-wall-exhausted"
        or stopped.get("source_sha256") != source_manifest()
        or stopped.get("preflight_sha256") != preflight.get("artifact_sha256")
        or not 750_000 < step < FINAL_STEP
        or not isinstance(recovery_record, dict)
    ):
        raise RuntimeError("existing ASFD wall stop failed revalidation")
    recovery_path = _resolve(recovery_record.get("path"), stop_path.parent)
    recovery, digest = load_recovery_payload(
        recovery_path,
        require_sidecar=True,
        validate_counters=True,
    )
    if (
        digest != recovery_record.get("sha256")
        or int(recovery.get("planned_updates", -1)) != FINAL_STEP
        or int(recovery.get("completed_updates", -1)) != step
    ):
        raise RuntimeError("ASFD wall-stop recovery changed or is incomplete")
    if mirror.verify_recovery(recovery_path, recovery_step=step) != recovery_record.get(
        "durable_mirror"
    ):
        raise RuntimeError("ASFD wall-stop recovery mirror binding changed")
    # ``mirror`` repairs a crash-left incomplete immutable pair only when the
    # surviving bytes match this locally authenticated stop. A complete but
    # different remote stop remains fail-closed.
    mirror.mirror(stop_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--durable-mirror-dir", type=Path, required=True)
    parser.add_argument("--durable-workspace-dir", type=Path, required=True)
    parser.add_argument("--durable-storage-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--i-confirm-durable-mirror", action="store_true")
    parser.add_argument("--i-confirm-durable-workspace", action="store_true")
    parser.add_argument("--i-have-authorized-asfd-continuation", action="store_true")
    args = parser.parse_args()
    _require_launch_authorization(
        paid=args.i_have_authorized_asfd_continuation,
        durable_mirror=args.i_confirm_durable_mirror,
        durable_workspace=args.i_confirm_durable_workspace,
    )
    assert_no_inherited_freeze()
    preflight = verify_json(args.preflight, "asfd-preflight")
    if preflight.get("decision") != "GO":
        raise RuntimeError("ASFD continuation requires a GO measured preflight")
    if preflight.get("source_sha256") != source_manifest():
        raise RuntimeError("ASFD source changed after preflight")
    foundation_gate_path = _resolve(
        preflight["foundation_gate"]["path"], args.preflight.parent
    )
    qualification_path = _resolve(
        preflight["qualification"]["path"], args.preflight.parent
    )
    bank_path = _resolve(preflight["feature_bank"]["path"], args.preflight.parent)
    teacher = _resolve(preflight["teacher_checkpoint"]["path"], args.preflight.parent)
    foundation_recovery = _resolve(
        preflight["foundation_recovery"]["path"], args.preflight.parent
    )
    foundation = verify_json(foundation_gate_path, "cap-emf2-750k-foundation-gate")
    qualification = verify_json(qualification_path, "asfd-target-only-qualification")
    bank_payload = verify_json(bank_path, "asfd-feature-bank")
    if (
        foundation.get("artifact_sha256") != preflight["foundation_gate"]["sha256"]
        or qualification.get("artifact_sha256") != preflight["qualification"]["sha256"]
        or bank_payload.get("artifact_sha256") != preflight["feature_bank"]["sha256"]
        or bank_payload.get("qualification", {}).get("sha256")
        != qualification.get("artifact_sha256")
    ):
        raise RuntimeError("ASFD preflight upstream artifact bindings differ")
    cap2_preflight_path = _resolve(
        foundation["inputs"]["preflight"], foundation_gate_path.parent
    )
    from ..stage_cap2.artifacts import load_preflight as load_cap2_preflight

    cap2_preflight = load_cap2_preflight(cap2_preflight_path)
    _require_declared_storage_root(cap2_preflight["storage"], args.durable_storage_root)
    workspace_record = _require_durable_workspace(
        args.durable_workspace_dir,
        required_paths={
            "ASFD output directory": args.output_dir,
            "ASFD preflight": args.preflight,
            "foundation gate": foundation_gate_path,
            "feature qualification": qualification_path,
            "feature bank": bank_path,
            "teacher checkpoint": teacher,
            "foundation recovery": foundation_recovery,
        },
    )
    live_storage = _require_live_storage_capacity(
        cap2_preflight["storage"],
        workspace_root=args.durable_workspace_dir,
        mirror_root=args.durable_mirror_dir,
    )
    device = resolve_device(args.device)
    settings = configure(device, allow_tf32=False)
    _require_exact_preflight_runtime(preflight, settings)
    torch.use_deterministic_algorithms(True)

    if args.output_dir.exists() and not args.output_dir.is_dir():
        raise RuntimeError("ASFD output path is not a directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "asfd_result.json"
    checkpoints = args.output_dir / "checkpoints"
    snapshots = args.output_dir / "snapshots"
    checkpoints.mkdir(exist_ok=True)
    snapshots.mkdir(exist_ok=True)
    mirror = DurableMirror(args.output_dir, args.durable_mirror_dir)
    mirror_probe = mirror.probe()
    wall_stop_path = args.output_dir / "asfd_wall_stop.json"
    wall_stop_sidecar = wall_stop_path.with_suffix(wall_stop_path.suffix + ".sha256")
    remote_wall_stop = mirror.mirror_root / _relative(wall_stop_path, args.output_dir)
    remote_wall_stop_sidecar = remote_wall_stop.with_suffix(
        remote_wall_stop.suffix + ".sha256"
    )
    wall_stop_present = any(
        path.exists()
        for path in (
            wall_stop_path,
            wall_stop_sidecar,
            remote_wall_stop,
            remote_wall_stop_sidecar,
        )
    )
    if result_path.exists():
        if wall_stop_present:
            raise RuntimeError(
                "ASFD cannot contain both a terminal result and a wall-stop marker"
            )
        completed = verify_json(result_path, STATUS)
        _revalidate_terminal_result(
            completed,
            result_path=result_path,
            preflight=preflight,
            mirror=mirror,
        )
        mirror.mirror(result_path)
        print(f"revalidated and mirrored existing {result_path}")
        return 0
    if wall_stop_present:
        if not wall_stop_path.is_file() or not wall_stop_sidecar.is_file():
            raise RuntimeError(
                "ASFD has a durable or partial wall-stop marker; restore the "
                "complete run tree and inspect it instead of resuming paid work"
            )
        stopped = verify_json(wall_stop_path, WALL_STOP_STATUS)
        _revalidate_wall_stop(
            stopped,
            stop_path=wall_stop_path,
            preflight=preflight,
            mirror=mirror,
        )
        raise RuntimeError(
            "ASFD measured continuation wall was exhausted at durable recovery "
            f"step {stopped['recovery_step']}; this campaign may not resume"
        )

    profile = continuation_profile(cap2_preflight, FINAL_STEP)
    wall_policy = _asfd_wall_policy(
        preflight, recovery_every=profile.train.recovery_every
    )
    foundation_payload, foundation_digest = load_recovery_payload(
        foundation_recovery, require_sidecar=True, validate_counters=True
    )
    if foundation_digest != preflight["foundation_recovery"]["sha256"]:
        raise RuntimeError("ASFD foundation recovery changed after measured preflight")
    foundation_wall_seconds = float(foundation_payload.get("wall_seconds", math.nan))
    if not math.isfinite(foundation_wall_seconds) or foundation_wall_seconds < 0:
        raise RuntimeError("ASFD foundation recovery has invalid cumulative wall time")
    declared = profile_payload(profile)
    extension = ASFDCorrection(
        teacher_checkpoint=teacher,
        bank_metadata=bank_path,
        qualification=qualification,
        coefficients=preflight["coefficients"],
        spectral_scale=float(preflight["spectral_scale"]),
        device=device,
        data_root=args.data_root,
        identity_binding={
            "preflight_sha256": preflight["artifact_sha256"],
            "foundation_gate_sha256": foundation["artifact_sha256"],
        },
    )
    external, identity_sha = _identity(
        {
            "status": "asfd-continuation-run-identity",
            "preflight_sha256": preflight["artifact_sha256"],
            "foundation_gate_sha256": foundation["artifact_sha256"],
            "feature_bank_sha256": preflight["feature_bank"]["sha256"],
            "teacher_checkpoint_sha256": preflight["teacher_checkpoint"]["sha256"],
            "source_sha256": source_manifest(),
            "final_step": FINAL_STEP,
        }
    )
    recovery = args.output_dir / "asfd_recovery.pt"
    if not recovery.exists():
        fork_foundation_recovery(
            foundation_recovery,
            recovery,
            profile=profile,
            external_identity=external,
            extension=extension,
            expected_sha256=preflight["foundation_recovery"]["sha256"],
        )
        mirror.mirror(recovery, mutable=True, recovery_step=750_000)
        recovered, _ = load_recovery_payload(
            recovery, require_sidecar=True, validate_counters=True
        )
    else:
        recovered, _ = load_recovery_payload(
            recovery, require_sidecar=True, validate_counters=True
        )
        completed_step = int(recovered["completed_updates"])
        committed = mirror.recovery_steps(recovery)
        if not committed or max(committed) != completed_step:
            raise RuntimeError(
                "local ASFD recovery is not the latest committed durable step"
            )
        mirror.verify_recovery(recovery, recovery_step=completed_step)

    prior_continuation_wall_seconds = (
        float(recovered.get("wall_seconds", math.nan)) - foundation_wall_seconds
    )
    if (
        not math.isfinite(prior_continuation_wall_seconds)
        or prior_continuation_wall_seconds < -1e-6
    ):
        raise RuntimeError("ASFD recovery has invalid continuation wall time")
    prior_continuation_wall_seconds = max(0.0, prior_continuation_wall_seconds)
    hard_wall_seconds = float(wall_policy["hard_cumulative_continuation_wall_seconds"])
    if (
        int(recovered.get("completed_updates", -1)) < FINAL_STEP
        and prior_continuation_wall_seconds >= hard_wall_seconds
    ):
        raise RuntimeError(
            "ASFD measured hard wall was already reached before continuation resume"
        )

    def same_state(left: dict, right: dict) -> bool:
        return set(left) == set(right) and all(
            left[name].dtype == right[name].dtype
            and left[name].shape == right[name].shape
            and torch.equal(left[name].detach().cpu(), right[name].detach().cpu())
            for name in left
        )

    def checkpoint(step: int, raw: dict, ema: dict) -> dict:
        entry = {}
        for kind, state in (("raw", raw), ("ema", ema)):
            path = checkpoints / f"asfd_step{step}_{kind}.pt"
            if path.exists():
                existing = load_checkpoint(
                    path,
                    step=step,
                    kind=kind,
                    arm="EMF-ASFD",
                    declared_profile=declared,
                    realized_profile=declared,
                    preflight_sha256=preflight["artifact_sha256"],
                    run_identity_sha256=identity_sha,
                    unit_seed=0,
                )
                if not same_state(existing["state_dict"], state):
                    raise RuntimeError("replayed ASFD checkpoint state changed")
                digest = existing["artifact_sha256"]
            else:
                digest = save_checkpoint(
                    path,
                    state,
                    step=step,
                    kind=kind,
                    arm="EMF-ASFD",
                    declared_profile=declared,
                    realized_profile=declared,
                    preflight_sha256=preflight["artifact_sha256"],
                    run_identity_sha256=identity_sha,
                    unit_seed=0,
                )
            entry[kind] = {
                "path": _relative(path, args.output_dir),
                "sha256": digest,
                "durable_mirror": mirror.mirror(path),
            }
        return entry

    def snapshot_state(step: int, state: dict) -> None:
        path = snapshots / f"asfd_snapshot_step{step}.pt"
        if path.exists():
            existing = load_snapshot(
                path,
                step=step,
                arm="EMF-ASFD",
                preflight_sha256=preflight["artifact_sha256"],
                run_identity_sha256=identity_sha,
                unit_seed=0,
            )
            floating = {
                name: value
                for name, value in state.items()
                if value.is_floating_point()
            }
            if not same_state(existing["state_dict"], floating):
                raise RuntimeError("replayed ASFD snapshot state changed")
        else:
            save_snapshot(
                path,
                state,
                step=step,
                arm="EMF-ASFD",
                declared_profile=declared,
                realized_profile=declared,
                preflight_sha256=preflight["artifact_sha256"],
                run_identity_sha256=identity_sha,
                unit_seed=0,
            )
        mirror.mirror(path)

    continuation_started = time.monotonic()

    def recovery_saved(step: int, path: Path) -> None:
        recovery_mirror = mirror.mirror(path, mutable=True, recovery_step=step)
        cumulative = prior_continuation_wall_seconds + (
            time.monotonic() - continuation_started
        )
        if step < FINAL_STEP and cumulative >= hard_wall_seconds:
            stop = {
                "status": WALL_STOP_STATUS,
                "decision": "HALT",
                "reason": "measured-continuation-wall-exhausted",
                "recovery_step": step,
                "observed_cumulative_continuation_wall_seconds": cumulative,
                "hard_cumulative_continuation_wall_seconds": hard_wall_seconds,
                "wall_policy": wall_policy,
                "preflight_sha256": preflight["artifact_sha256"],
                "source_sha256": source_manifest(),
                "recovery": {
                    "path": _relative(path, args.output_dir),
                    "sha256": recovery_mirror["sha256"],
                    "durable_mirror": recovery_mirror,
                },
            }
            write_json_atomic(wall_stop_path, stop)
            mirror.mirror(wall_stop_path)
            raise RuntimeError(
                "ASFD measured hard wall reached after a verified durable recovery "
                f"at step {step}; cumulative continuation wall={cumulative:.1f}s, "
                f"limit={hard_wall_seconds:.1f}s"
            )

    pool = cifar10_train_pool(args.data_root)
    outcome = train_cap_unit(
        pool,
        profile,
        device,
        recovery_path=recovery,
        checkpoint=checkpoint,
        snapshot=snapshot_state,
        recovery_saved=recovery_saved,
        progress=lambda message: print(message, flush=True),
        recovery_identity=external,
        training_extension=extension,
    )
    final_recovery, final_digest = load_recovery_payload(
        recovery, require_sidecar=True, validate_counters=True
    )
    final_continuation_wall_seconds = max(
        0.0, float(final_recovery["wall_seconds"]) - foundation_wall_seconds
    )
    checks = {
        "completed_800k": int(final_recovery.get("completed_updates", -1))
        == FINAL_STEP,
        "planned_800k": int(final_recovery.get("planned_updates", -1)) == FINAL_STEP,
        "correction_events": extension.event_count
        == asfd_config().continuation_updates // asfd_config().gradients.cadence,
        "no_nonfinite_updates": outcome.nonfinite_updates == 0,
        "one_call_inference": outcome.inference_forward_calls == 1,
        "final_checkpoint_present": str(FINAL_STEP) in outcome.checkpoints,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    result = {
        "status": STATUS,
        "decision": "GO" if not failed else "NO_GO",
        "failed": failed,
        "checks": checks,
        "preflight": {
            "path": _relative(args.preflight, args.output_dir),
            "sha256": preflight["artifact_sha256"],
        },
        "foundation_gate_sha256": foundation["artifact_sha256"],
        "run_identity_sha256": identity_sha,
        "profile": declared,
        "final_step": FINAL_STEP,
        "recovery": {
            "path": _relative(recovery, args.output_dir),
            "sha256": final_digest,
            "durable_mirror": mirror.verify_recovery(
                recovery, recovery_step=FINAL_STEP
            ),
        },
        "checkpoints": outcome.checkpoints,
        "snapshots": outcome.snapshots,
        "correction": {
            "events": extension.event_count,
            "generated_forwards": extension.generated_forwards,
            "abort_reasons": extension.monitor.reasons,
            "last_record": outcome.auxiliary_history[-1]
            if outcome.auxiliary_history
            else None,
        },
        "training": {
            "optimizer_updates": outcome.optimizer_updates,
            "examples_seen": outcome.examples_seen,
            "cumulative_foundation_and_continuation_wall_seconds": outcome.wall_seconds,
            "continuation_wall_seconds": final_continuation_wall_seconds,
            "peak_memory_bytes": outcome.peak_memory_bytes,
            "device": settings,
            "wall_policy": wall_policy,
        },
        "durability": {
            "workspace": workspace_record,
            "live_storage_capacity": live_storage,
            "mirror_attestation": mirror.attestation,
            "mirror_live_probe": mirror_probe,
        },
        "source_sha256": source_manifest(),
        "scope": "single encoder-free CAP foundation followed by one ASFD correction continuation; no matched 50k raw-control continuation",
    }
    digest = write_json_atomic(result_path, result)
    mirror_record = mirror.mirror(result_path)
    print(f"wrote {result_path} sha256={digest} decision={result['decision']}")
    print(f"durable result: {mirror_record['relative_path']}")
    return 0 if result["decision"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
