"""Run one guarded CAP-EMF-2 sampler arm, never a full confirmation."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import torch

from ..device import configure, resolve_device
from ..stage_cap.config import enable_tf32
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
    file_sha256,
    load_checkpoint,
    load_preflight,
    load_snapshot,
    profile_payload,
    save_checkpoint,
    save_snapshot,
    verify_file,
    verify_json,
    write_json_atomic,
    write_sha256_sidecar_atomic,
)
from .budget import revalidate_budget_plan
from .config import SAMPLER_ARMS, apply_calibrated_gate, screen_profile
from .durable_mirror import DurableMirror, load_root_attestation, probe_root
from .early_admission import load_early_admission
from .hardware import require_same_hardware
from .preview import save_fixed_grid
from .promotion import _checkpoint_previews_valid, load_promotion
from .selection import LEGACY_ARM, load_selection


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


def _sync_existing_sealed_artifacts(
    root: Path,
    mirror: DurableMirror,
    *,
    mutable_steps: dict[Path, int],
    excluded: set[Path] | None = None,
) -> list[dict[str, str | int]]:
    """Verify/synchronize every complete hash pair already in a run tree."""

    exclusions = {path.resolve() for path in (excluded or set())}
    records: list[dict[str, str | int]] = []
    for sidecar in sorted(root.rglob("*.sha256")):
        payload = Path(str(sidecar)[: -len(".sha256")])
        if not payload.is_file():
            raise RuntimeError(f"orphan CAP2 SHA sidecar: {sidecar}")
        resolved = payload.resolve()
        if resolved in exclusions:
            continue
        if resolved in mutable_steps:
            records.append(
                mirror.mirror(
                    payload,
                    mutable=True,
                    recovery_step=mutable_steps[resolved],
                )
            )
        else:
            records.append(mirror.mirror(payload))
    return records


def _hard_wall_policy(preflight: dict, updates: int) -> dict[str, float | int]:
    """Derive the arm-local stop and its one-interval detection envelope."""

    plan = revalidate_budget_plan(
        preflight.get("budget"), preflight["inputs"]["benchmark"]
    )
    projection = (
        preflight["inputs"]["benchmark"].get("projections", {}).get(str(updates))
    )
    if not isinstance(projection, dict):
        raise TypeError(f"CAP2 preflight lacks the {updates} wall-time projection")
    conservative = projection.get("conservative_raw_loop_upper_hours")
    if (
        isinstance(conservative, bool)
        or not isinstance(conservative, (int, float))
        or not 0 < float(conservative) < float("inf")
    ):
        raise RuntimeError("CAP2 conservative wall-time projection is invalid")
    benchmark = preflight["inputs"]["benchmark"]
    frozen = screen_profile(
        str(benchmark["arm"]), str(benchmark["numerical"]), updates=updates
    ).train
    recovery_interval = int(frozen.recovery_every)
    hard_limit = float(conservative) * (1.0 + float(plan["contingency_fraction"]))
    # The stop is checked only after the next recovery has been made durable.
    # Cadence validation below guarantees at most this many additional updates.
    # The hours figure is a conservative projection, not a provider billing
    # guarantee; external provider-side spend limits remain mandatory.
    projected_interval_hours = float(conservative) / float(updates) * recovery_interval
    return {
        "hard_cumulative_wall_hours": hard_limit,
        "recovery_interval_updates": recovery_interval,
        "maximum_detection_overshoot_updates": recovery_interval,
        "conservative_projected_maximum_detection_overshoot_hours": (
            projected_interval_hours
        ),
        "conservative_projected_maximum_detected_wall_hours": (
            hard_limit + projected_interval_hours
        ),
    }


def _hard_cumulative_wall_hours(preflight: dict, updates: int) -> float:
    """Backward-compatible scalar view of :func:`_hard_wall_policy`."""

    return float(_hard_wall_policy(preflight, updates)["hard_cumulative_wall_hours"])


def _require_durable_workspace(
    workspace_root: Path, *, required_paths: dict[str, Path]
) -> dict[str, object]:
    """Verify the layout-preserving authorization workspace before training."""

    root = workspace_root.resolve()
    attestation = load_root_attestation(root)
    probe = probe_root(root)
    if probe.get("roundtrip_verified") is not True:
        raise RuntimeError("durable authorization workspace failed its live probe")
    for label, path in required_paths.items():
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"CAP2 {label} must live inside the durable authorization "
                f"workspace: {resolved} is outside {root}"
            ) from error
    return {
        "root": str(root),
        "attestation": attestation,
        "live_roundtrip_probe": probe,
        "layout_requirement": (
            "preserve the common evidence/gates/runs relative layout across "
            "instance replacement"
        ),
    }


def _require_live_storage_capacity(
    storage_plan: dict, *, workspace_root: Path, mirror_root: Path
) -> dict[str, int | str]:
    """Recheck the shared durable filesystem before each paid continuation."""

    root = Path(str(storage_plan.get("storage_root", ""))).resolve()
    for label, path in (
        ("workspace", workspace_root.resolve()),
        ("per-arm mirror", mirror_root.resolve()),
    ):
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError(
                f"CAP2 {label} is outside the preflighted durable storage root"
            ) from error
    projected = storage_plan.get("projected_bytes")
    units = storage_plan.get("measured_unit_bytes")
    if not isinstance(projected, dict) or not isinstance(units, dict):
        raise TypeError("CAP2 preflight storage plan is malformed")
    required = int(projected.get("required_with_contingency", -1))
    # Enough free space for the next recovery plus a worst-case checkpoint and
    # snapshot transaction, each with its mirrored copy.  The full campaign
    # capacity was already checked at preflight; this catches unrelated usage
    # or a stale/mis-mounted volume before the next paid continuation.
    transaction_headroom = 2 * (
        int(units.get("recovery", 0))
        + int(units.get("checkpoint_raw", 0))
        + int(units.get("checkpoint_ema", 0))
        + int(units.get("snapshot", 0))
    )
    usage = shutil.disk_usage(root)
    if int(usage.total) < required:
        raise RuntimeError("durable storage capacity fell below the preflight plan")
    if int(usage.free) < transaction_headroom:
        raise RuntimeError("durable storage lacks one-transaction safety headroom")
    return {
        "storage_root": str(root),
        "total_bytes": int(usage.total),
        "free_bytes": int(usage.free),
        "required_campaign_bytes": required,
        "minimum_transaction_headroom_bytes": transaction_headroom,
    }


def _require_wall_budget_before_training(
    policy: dict[str, float | int], prior_wall_seconds: float
) -> float:
    if not 0.0 <= prior_wall_seconds < float("inf"):
        raise RuntimeError("CAP2 prior recovery wall time is invalid")
    prior_hours = prior_wall_seconds / 3600.0
    limit = float(policy["hard_cumulative_wall_hours"])
    if prior_hours >= limit:
        raise RuntimeError(
            "CAP2 hard wall-time limit was already reached before training: "
            f"{prior_hours:.3f}h >= {limit:.3f}h"
        )
    return prior_hours


def _require_latest_committed_recovery(
    mirror: DurableMirror, recovery_path: Path, completed_step: int | None
) -> list[int]:
    """Reject a missing/stale local recovery relative to durable history."""

    steps = mirror.recovery_steps(recovery_path)
    if completed_step is None:
        if steps:
            raise RuntimeError(
                "durable mirror has committed recovery history but the local "
                "run has no recovery; restore the latest committed step"
            )
        return steps
    if not steps or max(steps) != completed_step:
        latest = max(steps) if steps else None
        raise RuntimeError(
            "local CAP2 recovery is not the latest durable commit: "
            f"local={completed_step}, durable={latest}"
        )
    return steps


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
    if planned not in {
        50_000,
        100_000,
        150_000,
        300_000,
        500_000,
        650_000,
        750_000,
    }:
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
    result: dict,
    recovery: dict,
    *,
    recovery_path: Path,
    root: Path,
    mirror: DurableMirror,
) -> None:
    """Bind optimizer/RNG/counters, not merely weights, into the 150k result."""
    record = result.get("recovery")
    expected = {
        "path": _relative(recovery_path, root),
        "sha256": recovery.get("artifact_sha256"),
        "planned_updates": 150_000,
        "completed_updates": 150_000,
        "continuation_authorization": None,
        "durable_mirror": mirror.verify_recovery(recovery_path, recovery_step=150_000),
    }
    if record != expected:
        raise RuntimeError("immutable 150k result does not bind the promoted recovery")


def _quarantine_incomplete_terminal_result(path: Path) -> dict[str, str] | None:
    """Move a crash-left result half aside so finalization can be retried.

    A JSON result is committed only when both its payload and SHA sidecar exist.
    A machine loss can occur between those two atomic renames.  Preserve the
    surviving bytes for diagnosis, but do not let an uncommitted half strand a
    completed, durably recovered training run.
    """

    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() == sidecar.exists():
        return None
    quarantine = path.parent / ".uncommitted-terminal-results"
    quarantine.mkdir(parents=True, exist_ok=True)
    records: dict[str, str] = {}
    for label, source in (("payload", path), ("sidecar", sidecar)):
        if not source.is_file():
            continue
        digest = file_sha256(source)
        destination = quarantine / f"{path.name}.{label}-{digest}.bin"
        if destination.exists():
            if file_sha256(destination) != digest:
                raise RuntimeError(
                    f"terminal-result quarantine is corrupt: {destination}"
                )
            source.unlink()
        else:
            os.replace(source, destination)
        records[label] = str(destination.resolve())
    return records


def _verify_completed_terminal_result(
    path: Path,
    *,
    arm: str,
    candidate: str,
    updates: int,
    planned_updates: int,
    preflight_sha256: str,
    run_identity_sha256: str,
    unit_seed: int,
    declared_profile: dict,
    realized_profile: dict,
    expected_examples: int,
    gate_config,
    deterministic_algorithms: bool,
    precision: dict,
    device: dict,
    hardware: dict,
    recovery: dict,
    recovery_path: Path,
    root: Path,
    mirror: DurableMirror,
    workspace_record: dict[str, object],
    mirror_root: Path,
    run_identity_mirror: dict[str, str | int],
    hard_wall_policy: dict[str, float | int],
    storage_plan: dict,
) -> dict:
    """Revalidate an already-published result before idempotent mirroring.

    ``updates`` is the committed execution boundary represented by the result;
    ``planned_updates`` may be larger only for the ordered foundation's 50k
    admission pause.
    """

    planned = int(planned_updates)
    if not updates <= planned:
        raise RuntimeError("completed CAP2 result exceeds its planned horizon")

    result = verify_json(path, "cap-emf2-screen-unit")
    exact = {
        "arm": arm,
        "numerical_candidate": candidate,
        "preflight_sha256": preflight_sha256,
        "run_identity_sha256": run_identity_sha256,
        "unit_seed": unit_seed,
        "declared_profile": declared_profile,
        "realized_profile": realized_profile,
    }
    mismatches = sorted(key for key, value in exact.items() if result.get(key) != value)
    if mismatches:
        raise RuntimeError(
            f"completed CAP2 result changed across finalization: {mismatches}"
        )
    training = result.get("training")
    if not isinstance(training, dict):
        raise TypeError("completed CAP2 result lacks its training ledger")
    if (
        int(training.get("optimizer_updates", -1)) != updates
        or int(training.get("examples_seen_target", -1)) != expected_examples
        or int(training.get("examples_seen", -1)) != expected_examples
    ):
        raise RuntimeError("completed CAP2 result has inconsistent training counters")
    recovery_to_training = {
        "history": "history",
        "health": "health",
        "optimizer_updates": "optimizer_updates",
        "examples_seen": "examples_seen",
        "model_forwards": "objective_sample_evaluations",
        "objective_forward_calls": "objective_forward_calls",
        "clipped_updates": "clipped_updates",
        "clipped_updates_final_window": "clipped_updates_final_window",
        "final_window_updates": "final_window_updates",
        "nonfinite_updates": "nonfinite_updates",
    }
    ledger_mismatches = sorted(
        result_key
        for recovery_key, result_key in recovery_to_training.items()
        if training.get(result_key) != recovery.get(recovery_key)
    )
    final_window = int(recovery.get("final_window_updates", -1))
    clipped_final = int(recovery.get("clipped_updates_final_window", -1))
    expected_clip_fraction = clipped_final / final_window if final_window else 0.0
    if training.get("clip_fraction_final_window") != expected_clip_fraction:
        ledger_mismatches.append("clip_fraction_final_window")
    if int(training.get("inference_forward_calls", -1)) != 1:
        ledger_mismatches.append("inference_forward_calls")
    if ledger_mismatches:
        raise RuntimeError(
            "completed CAP2 result differs from its recovery ledger: "
            f"{sorted(set(ledger_mismatches))}"
        )
    for key in ("wall_seconds", "peak_memory_bytes", "peak_memory_reserved_bytes"):
        observed = training.get(key)
        recovered = recovery.get(key)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < float(recovered)
        ):
            raise RuntimeError(f"completed CAP2 result has invalid post-recovery {key}")
    if result.get("checkpoints") != recovery.get("checkpoints"):
        raise RuntimeError("completed CAP2 result changed its checkpoint ledger")
    snapshot_steps = recovery.get("snapshots")
    if not isinstance(snapshot_steps, list):
        raise TypeError("completed CAP2 recovery has a malformed snapshot ledger")
    snapshot_root = root / "raw_snapshots"
    expected_snapshots = []
    for step in sorted(int(value) for value in snapshot_steps):
        snapshot_path = snapshot_root / f"cap2_{arm}_snapshot_step{step}.pt"
        expected_snapshots.append(
            {
                "step": step,
                "path": _relative(snapshot_path, root),
                "sha256": verify_file(snapshot_path),
                "durable_mirror": mirror.verify(snapshot_path),
            }
        )
    if result.get("raw_snapshots") != expected_snapshots:
        raise RuntimeError("completed CAP2 result changed its raw-snapshot ledger")
    health = recovery.get("health")
    if not isinstance(health, list) or not health:
        raise RuntimeError("completed CAP2 recovery has no health ledger")
    ema_health = [record["ema"] for record in health if "ema" in record]
    if not ema_health:
        raise RuntimeError("completed CAP2 recovery has no EMA checkpoint health")
    recomputed_gate = capability_gate(
        health[-1].get("ema", health[-1]),
        max(float(record["effective_rank_ratio"]) for record in ema_health),
        expected_clip_fraction,
        int(recovery["nonfinite_updates"]),
        1,
        gate_config,
    )
    if result.get("train_only_gate") != recomputed_gate:
        raise RuntimeError(
            "completed CAP2 result changed its recomputed train-only gate"
        )
    preview_result = result
    if updates < planned:
        preview_result = deepcopy(result)
        checkpoint_updates = preview_result["declared_profile"]["train"][
            "checkpoint_updates"
        ]
        preview_result["declared_profile"]["train"]["checkpoint_updates"] = [
            step for step in checkpoint_updates if int(step) <= updates
        ]
    if not _checkpoint_previews_valid(preview_result, anchor=path.parent):
        raise RuntimeError("completed CAP2 result has invalid checkpoint previews")
    expected_batch_split = {
        "micro_batch": realized_profile["train"]["micro_batch"],
        "accumulation_steps": realized_profile["train"]["accumulation_steps"],
        "effective_batch": realized_profile["train"]["effective_batch"],
        "overridden": realized_profile != declared_profile,
    }
    runtime_exact = {
        "realized_batch_split": expected_batch_split,
        "precision": precision,
        "deterministic_algorithms": deterministic_algorithms,
        "device": device,
        "hardware": hardware,
    }
    runtime_mismatches = sorted(
        key for key, value in runtime_exact.items() if result.get(key) != value
    )
    if runtime_mismatches:
        raise RuntimeError(
            f"completed CAP2 result changed its runtime bindings: {runtime_mismatches}"
        )
    authorization = recovery.get("continuation_authorization")
    expected_recovery = {
        "path": _relative(recovery_path, root),
        "sha256": recovery.get("artifact_sha256"),
        "planned_updates": planned,
        "completed_updates": updates,
        "continuation_authorization": authorization,
        "durable_mirror": mirror.verify_recovery(recovery_path, recovery_step=updates),
    }
    if result.get("recovery") != expected_recovery:
        raise RuntimeError("completed CAP2 result does not bind the final recovery")
    durability = result.get("durability")
    if not isinstance(durability, dict):
        raise TypeError("completed CAP2 result lacks its durability ledger")
    recorded_workspace = durability.get("authorization_workspace")
    if not isinstance(recorded_workspace, dict):
        raise TypeError("completed CAP2 result lacks its workspace attestation")
    if (
        durability.get("required") is not True
        or durability.get("synchronous") is not True
        or recorded_workspace.get("root") != workspace_record.get("root")
        or recorded_workspace.get("attestation") != workspace_record.get("attestation")
        or durability.get("mirror_root") != str(mirror_root.resolve())
        or durability.get("hard_cumulative_wall_hours")
        != hard_wall_policy.get("hard_cumulative_wall_hours")
    ):
        raise RuntimeError("completed CAP2 result has inconsistent durability bindings")
    result_policy = durability.get("hard_wall_policy")
    if not isinstance(result_policy, dict):
        raise TypeError("completed CAP2 result lacks its hard-wall ledger")
    for key, expected_value in hard_wall_policy.items():
        if result_policy.get(key) != expected_value:
            raise RuntimeError(
                f"completed CAP2 result changed hard-wall policy field {key!r}"
            )
    prior_hours = result_policy.get("prior_committed_wall_hours")
    if (
        isinstance(prior_hours, bool)
        or not isinstance(prior_hours, (int, float))
        or not 0.0 <= float(prior_hours) <= float(recovery["wall_seconds"]) / 3600.0
        or int(result_policy.get("last_verified_recovery_step", -1)) != updates
        or durability.get("run_identity") != run_identity_mirror
        or not isinstance(durability.get("preexisting_sealed_artifacts_synced"), int)
        or int(durability["preexisting_sealed_artifacts_synced"]) < 0
    ):
        raise RuntimeError("completed CAP2 result has invalid final durability state")
    for probe_name, probe in (
        ("workspace", recorded_workspace.get("live_roundtrip_probe")),
        ("mirror", durability.get("live_roundtrip_probe")),
    ):
        if not isinstance(probe, dict) or probe.get("roundtrip_verified") is not True:
            raise RuntimeError(
                f"completed CAP2 result has invalid {probe_name} live probe"
            )
    recorded_storage = durability.get("live_storage_capacity")
    if not isinstance(recorded_storage, dict) or any(
        isinstance(recorded_storage.get(key), bool)
        or not isinstance(recorded_storage.get(key), int)
        or int(recorded_storage[key]) < 0
        for key in (
            "total_bytes",
            "free_bytes",
            "required_campaign_bytes",
            "minimum_transaction_headroom_bytes",
        )
    ):
        raise RuntimeError("completed CAP2 result has invalid live storage ledger")
    planned_storage = storage_plan.get("projected_bytes", {})
    measured_units = storage_plan.get("measured_unit_bytes", {})
    expected_headroom = 2 * sum(
        int(measured_units.get(key, 0))
        for key in ("recovery", "checkpoint_raw", "checkpoint_ema", "snapshot")
    )
    if (
        recorded_storage.get("storage_root")
        != str(Path(str(storage_plan.get("storage_root", ""))).resolve())
        or int(recorded_storage["required_campaign_bytes"])
        != int(planned_storage.get("required_with_contingency", -1))
        or int(recorded_storage["minimum_transaction_headroom_bytes"])
        != expected_headroom
    ):
        raise RuntimeError("completed CAP2 result names a different storage root")
    if int(recorded_storage["total_bytes"]) < int(
        recorded_storage["required_campaign_bytes"]
    ) or int(recorded_storage["free_bytes"]) < int(
        recorded_storage["minimum_transaction_headroom_bytes"]
    ):
        raise RuntimeError("completed CAP2 result recorded inadequate live storage")
    elapsed = result.get("elapsed_seconds")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise RuntimeError("completed CAP2 result has invalid elapsed time")
    return result


def _validate_recovery_request(
    recovery: dict | None,
    *,
    requested: int,
    expected_profile_name: str,
    campaign: str,
) -> None:
    if campaign == "ordered_750_foundation":
        if requested != 750_000:
            raise RuntimeError("the ordered foundation campaign has one 750k horizon")
        if recovery is None:
            return
        if recovery.get("profile_name") != expected_profile_name:
            raise RuntimeError("CAP2 recovery belongs to another arm/candidate")
        if int(recovery["planned_updates"]) != 750_000:
            raise RuntimeError("foundation recovery was not born with a 750k plan")
        completed = int(recovery["completed_updates"])
        if not 0 <= completed <= 750_000:
            raise RuntimeError("foundation recovery step lies outside its plan")
        return
    if campaign != "matched_screen":
        raise RuntimeError(f"unknown CAP2 campaign {campaign!r}")
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
        if planned == 300_000 and 150_000 <= completed <= 300_000:
            return
        raise RuntimeError(
            "300k requires a completed 150k recovery or a promoted 300k recovery"
        )
    if planned > requested:
        raise RuntimeError("CAP2 recovery cannot move to a shorter horizon")
    if planned < requested and completed != planned:
        raise RuntimeError("finish the existing CAP2 horizon before extending it")


def _foundation_execution_stop(
    recovery: dict | None,
    *,
    pause_for_early_admission: bool,
    has_early_admission: bool,
) -> int:
    """Choose the execution boundary for the single ordered 750k model.

    The 50k boundary is a pause in one 750k recovery identity, not a shorter
    training horizon or a second experimental arm.  A fresh (or interrupted
    pre-50k) process must stop there.  Crossing it is fail-closed on a bound
    raw-state numerical GO artifact.
    """

    completed = 0 if recovery is None else int(recovery["completed_updates"])
    if completed < 50_000:
        if not pause_for_early_admission:
            raise RuntimeError(
                "ordered foundation must first pause durably at 50k for raw-state "
                "numerical admission"
            )
        if has_early_admission:
            raise RuntimeError(
                "ordered foundation cannot consume early admission before its 50k pause"
            )
        return 50_000
    if pause_for_early_admission:
        if completed != 50_000:
            raise RuntimeError(
                "the ordered foundation has already crossed its 50k pause"
            )
        if has_early_admission:
            raise RuntimeError("50k pause finalization does not consume admission")
        return 50_000
    if not has_early_admission:
        raise RuntimeError(
            "ordered foundation continuation past 50k requires the bound GO "
            "early-admission artifact"
        )
    return 750_000


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


def _ensure_foundation_continuation_authority(path: Path, payload: dict) -> str:
    """Freeze the first accepted 50k GO for every later foundation restart."""

    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() != sidecar.exists():
        raise RuntimeError("foundation continuation authority is only half-published")
    if path.exists():
        recorded = verify_json(path, "cap-emf2-foundation-continuation-authority")
        digest = recorded.pop("artifact_sha256")
        if recorded != payload:
            raise RuntimeError(
                "ordered foundation supplied a different 50k continuation authority"
            )
        return digest
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
        if (
            horizon
            not in {
                50_000,
                100_000,
                150_000,
                300_000,
                500_000,
                650_000,
                750_000,
            }
            or step > horizon
        ):
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
        if (
            horizon
            not in {
                50_000,
                100_000,
                150_000,
                300_000,
                500_000,
                650_000,
                750_000,
            }
            or step > horizon
        ):
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
        choices=(50_000, 100_000, 150_000, 300_000, 750_000),
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
        help=(
            "immutable raw-state 50k admission; required for matched 100k/150k "
            "or ordered-foundation continuation past the planned 50k pause"
        ),
    )
    parser.add_argument(
        "--pause-for-early-admission",
        action="store_true",
        help=(
            "ordered-foundation first phase only: keep the 750k plan but stop "
            "durably at 50k so its raw checkpoint can receive admission"
        ),
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
    parser.add_argument(
        "--durable-mirror-dir",
        type=Path,
        required=True,
        help=(
            "second, non-nested storage root backed by a persistent volume or "
            "object-store mount"
        ),
    )
    parser.add_argument(
        "--i-confirm-durable-mirror",
        action="store_true",
        help="confirm that --durable-mirror-dir survives deletion of the GPU instance",
    )
    parser.add_argument(
        "--durable-workspace-dir",
        type=Path,
        required=True,
        help=(
            "attested off-instance root containing evidence, gates, selection, "
            "and every arm run in one stable relative layout"
        ),
    )
    parser.add_argument(
        "--i-confirm-durable-workspace",
        action="store_true",
        help="confirm that the complete authorization workspace survives instance loss",
    )
    parser.add_argument("--i-have-authorized-the-screen-run", action="store_true")
    parser.add_argument("--i-have-authorized-the-300k-promotion", action="store_true")
    args = parser.parse_args()
    if args.unit_seed < 0:
        raise SystemExit("--unit-seed must be nonnegative")
    if not args.i_have_authorized_the_screen_run:
        raise SystemExit("CAP2 refuses to train without explicit screen authorization")
    if not args.i_confirm_durable_mirror:
        raise SystemExit(
            "CAP2 refuses paid training without confirmation that the mirror is durable"
        )
    if not args.i_confirm_durable_workspace:
        raise SystemExit(
            "CAP2 refuses paid training without a durable authorization workspace"
        )
    workspace_paths = {
        "output directory": args.output_dir,
        "preflight": args.preflight,
        **(
            {"early admission": args.early_admission}
            if args.early_admission is not None
            else {}
        ),
        **({"promotion": args.promotion} if args.promotion is not None else {}),
        **({"selection": args.selection} if args.selection is not None else {}),
    }
    workspace_record = _require_durable_workspace(
        args.durable_workspace_dir, required_paths=workspace_paths
    )

    preflight = load_preflight(args.preflight)
    campaign = preflight.get("budget", {}).get("campaign")
    if campaign == "ordered_750_foundation":
        if args.arm != "ordered_uniform" or args.updates != 750_000:
            raise SystemExit(
                "the ordered foundation campaign authorizes only ordered_uniform at 750k"
            )
        if (
            any(value is not None for value in (args.promotion, args.selection))
            or args.i_have_authorized_the_300k_promotion
        ):
            raise SystemExit(
                "foundation training does not consume screen promotion/selection flags"
            )
    elif campaign == "matched_screen":
        if args.pause_for_early_admission:
            raise SystemExit(
                "--pause-for-early-admission is valid only for ordered foundation"
            )
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
            raise SystemExit(
                "--early-admission is valid only for 100k/150k continuation"
            )
    else:
        raise RuntimeError(f"preflight records unknown campaign {campaign!r}")
    live_storage = _require_live_storage_capacity(
        preflight["storage"],
        workspace_root=args.durable_workspace_dir,
        mirror_root=args.durable_mirror_dir,
    )
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
    if campaign == "ordered_750_foundation" and declared_payload != preflight.get(
        "foundation_profile_750k"
    ):
        raise RuntimeError("live foundation differs from its source-bound profile")
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
    mirror = DurableMirror(root, args.durable_mirror_dir)
    mirror_probe = mirror.probe()
    hard_wall_policy = _hard_wall_policy(preflight, args.updates)
    hard_wall_hours = float(hard_wall_policy["hard_cumulative_wall_hours"])
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
        campaign=campaign,
    )
    execution_stop = args.updates
    if campaign == "ordered_750_foundation":
        execution_stop = _foundation_execution_stop(
            recovery_payload,
            pause_for_early_admission=args.pause_for_early_admission,
            has_early_admission=args.early_admission is not None,
        )
    result_path = root / f"result_{execution_stop}.json"
    finalization_only = bool(
        recovery_payload is not None
        and int(recovery_payload["planned_updates"]) == args.updates
        and int(recovery_payload["completed_updates"]) == execution_stop
    )
    result_sidecar = result_path.with_suffix(result_path.suffix + ".sha256")
    if result_path.exists() != result_sidecar.exists():
        if not finalization_only:
            raise RuntimeError(
                "an incomplete terminal result exists without a completed recovery"
            )
        _quarantine_incomplete_terminal_result(result_path)
    completed_result_exists = result_path.is_file() and result_sidecar.is_file()
    if completed_result_exists and not finalization_only:
        raise RuntimeError(
            "a committed terminal result exists without its completed recovery"
        )

    identity_profile = _with_batch_split(
        declared if campaign == "ordered_750_foundation" else frozen_150k,
        args.micro_batch,
    )
    identity_payload = {
        "status": "cap-emf2-run-identity",
        "campaign": campaign,
        "arm": args.arm,
        "candidate": candidate,
        "preflight_sha256": preflight["artifact_sha256"],
        # Stable across the entire authorized campaign.
        (
            "profile_750k_realized"
            if campaign == "ordered_750_foundation"
            else "profile_150k_realized"
        ): profile_payload(identity_profile),
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
    identity_mirror = mirror.mirror(root / "run_identity.json")
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
    preexisting_mirror_records = _sync_existing_sealed_artifacts(
        root,
        mirror,
        mutable_steps=(
            {recovery.resolve(): int(recovery_payload["completed_updates"])}
            if recovery_payload is not None
            else {}
        ),
        # A terminal result is mirrored only after its run identity, counters,
        # final recovery, and durability bindings have been revalidated below.
        excluded={result_path.resolve()},
    )
    _require_latest_committed_recovery(
        mirror,
        recovery,
        (
            int(recovery_payload["completed_updates"])
            if recovery_payload is not None
            else None
        ),
    )
    prior_wall_seconds = (
        float(recovery_payload.get("wall_seconds", 0.0))
        if recovery_payload is not None
        else 0.0
    )
    prior_wall_hours = prior_wall_seconds / 3600.0
    if not finalization_only:
        prior_wall_hours = _require_wall_budget_before_training(
            hard_wall_policy, prior_wall_seconds
        )

    early_admission_record = None
    foundation_continuation_authority = None
    consumes_early_admission = (
        campaign == "ordered_750_foundation" and execution_stop == 750_000
    ) or (campaign == "matched_screen" and args.updates in {100_000, 150_000})
    if consumes_early_admission:
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
        if campaign == "ordered_750_foundation":
            bound_recovery = immutable_50k.get("recovery")
            if (
                not isinstance(bound_recovery, dict)
                or int(bound_recovery.get("planned_updates", -1)) != 750_000
                or int(bound_recovery.get("completed_updates", -1)) != 50_000
                or immutable_50k.get("foundation_pause")
                != {
                    "planned_updates": 750_000,
                    "paused_at": 50_000,
                    "purpose": "raw-state numerical admission before continuation",
                }
            ):
                raise RuntimeError(
                    "ordered foundation 50k result is not a same-horizon admission pause"
                )
            if int(recovery_payload["completed_updates"]) == 50_000 and (
                bound_recovery.get("sha256") != recovery_payload.get("artifact_sha256")
            ):
                raise RuntimeError(
                    "ordered foundation continuation is not the exact optimizer/RNG "
                    "recovery admitted at 50k"
                )
            authority_path = root / "foundation_continuation_authority.json"
            authority_payload = {
                "status": "cap-emf2-foundation-continuation-authority",
                "campaign": campaign,
                "arm": args.arm,
                "candidate": candidate,
                "planned_updates": 750_000,
                "admitted_at": 50_000,
                "preflight_sha256": preflight["artifact_sha256"],
                "run_identity_sha256": identity_sha,
                "early_admission_sha256": early_admission_record["artifact_sha256"],
                "result_50k_sha256": immutable_50k["artifact_sha256"],
                "raw_checkpoint_50k_sha256": early_admission_record[
                    "raw_checkpoint_sha256"
                ],
                "recovery_50k_sha256": bound_recovery["sha256"],
            }
            authority_sha = _ensure_foundation_continuation_authority(
                authority_path, authority_payload
            )
            foundation_continuation_authority = {
                "path": _relative(authority_path, root),
                "sha256": authority_sha,
                "durable_mirror": mirror.mirror(authority_path),
            }

    recovery_authorization = None
    if args.updates == 300_000:
        result_150k = root / "result_150000.json"
        immutable_150k = verify_json(result_150k, "cap-emf2-screen-unit")
        early_admission_record = immutable_150k.get("early_admission")
        checkpoint_150k_raw = checkpoints / f"cap2_{args.arm}_step150000_raw.pt"
        checkpoint_150k_ema = checkpoints / f"cap2_{args.arm}_step150000_ema.pt"
        resuming_promoted = int(recovery_payload["planned_updates"]) == 300_000
        if resuming_promoted:
            # The first promoted start fully revalidates every promotion leaf
            # and the cross-arm selection before writing this authorization
            # into a durable recovery commit.  On a later instance-loss resume,
            # that committed authorization is the trust root: require the
            # supplied compact promotion/selection files to match its hashes,
            # without requiring every losing-arm PNG/evidence leaf to be
            # re-uploaded merely to continue the already-authorized state.
            promotion = verify_json(args.promotion, "cap-emf2-promotion")
            selection = verify_json(args.selection, "cap-emf2-cross-arm-selection")
        else:
            promotion = load_promotion(
                args.promotion,
                preflight_path=args.preflight,
                result_path=result_150k,
                raw_checkpoint_path=checkpoint_150k_raw,
                checkpoint_path=checkpoint_150k_ema,
                arm=args.arm,
                candidate=candidate,
                require_go=args.arm != LEGACY_ARM,
            )
            if (
                args.arm == LEGACY_ARM
                and promotion.get("control_continuation", {}).get("decision") != "GO"
            ):
                raise RuntimeError("legacy arm is not a valid concurrent control")
            selection = load_selection(
                args.selection,
                promotion_path=args.promotion,
                arm=args.arm,
                candidate=candidate,
            )
        recovery_authorization = _promotion_recovery_authorization(promotion, selection)
        if int(recovery_payload["planned_updates"]) == 150_000:
            _assert_result_binds_recovery(
                immutable_150k,
                recovery_payload,
                recovery_path=recovery,
                root=root,
                mirror=mirror,
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
            _assert_recovery_authorization(recovery_payload, recovery_authorization)

    if completed_result_exists:
        completed_result = _verify_completed_terminal_result(
            result_path,
            arm=args.arm,
            candidate=candidate,
            updates=execution_stop,
            planned_updates=args.updates,
            preflight_sha256=preflight["artifact_sha256"],
            run_identity_sha256=identity_sha,
            unit_seed=args.unit_seed,
            declared_profile=declared_payload,
            realized_profile=realized_payload,
            expected_examples=execution_stop * frozen.train.effective_batch,
            gate_config=frozen.gate,
            deterministic_algorithms=not args.nondeterministic,
            precision=precision,
            device=device_settings,
            hardware=live_hardware,
            recovery=recovery_payload,
            recovery_path=recovery,
            root=root,
            mirror=mirror,
            workspace_record=workspace_record,
            mirror_root=args.durable_mirror_dir,
            run_identity_mirror=identity_mirror,
            hard_wall_policy=hard_wall_policy,
            storage_plan=preflight["storage"],
        )
        result_mirror = mirror.mirror(result_path)
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "gate": completed_result.get("train_only_gate"),
                    "finalization": "existing result revalidated and mirrored",
                },
                indent=2,
            )
        )
        print(
            f"verified {result_path}; durable mirror at "
            f"{result_mirror['relative_path']}"
        )
        return 0

    assert_unused(result_path)

    pool = cifar10_train_pool(args.data_root)

    def checkpoint(step: int, raw: dict, ema: dict) -> dict:
        entry = {}
        for kind, state in (("raw", raw), ("ema", ema)):
            path = checkpoints / f"cap2_{args.arm}_step{step}_{kind}.pt"
            digest = save_checkpoint(
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
            )
            entry[kind] = {
                "path": _relative(path, root),
                "sha256": digest,
                "durable_mirror": mirror.mirror(path),
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
        mirror.mirror(path)

    previews = root / "previews"

    def checkpoint_health_observer(
        step: int,
        raw_components: dict[str, torch.Tensor],
        ema_components: dict[str, torch.Tensor],
    ) -> dict:
        """Publish the same fixed, uncurated one-step rows at each checkpoint."""

        records: dict[str, dict] = {}
        for kind, components in (("raw", raw_components), ("ema", ema_components)):
            path = previews / f"cap2_{args.arm}_step{step}_{kind}.png"
            if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
                raise RuntimeError(f"refusing to overwrite checkpoint preview {path}")
            digest = save_fixed_grid(components["final"], path, rows=8, columns=16)
            write_sha256_sidecar_atomic(path, digest)
            records[kind] = {
                "path": _relative(path, root),
                "sha256": digest,
                "rows": 8,
                "columns": 16,
                "samples": 128,
                "selection": (
                    "first fixed sealed train-only health-noise rows; no curation"
                ),
                "durable_mirror": mirror.mirror(path),
            }
        return {
            "status": "cap-emf2-fixed-checkpoint-previews",
            "step": step,
            "raw_and_ema": records,
            "quantitative_role": "report/veto only; never rescues a failed gate",
        }

    started = time.time()
    last_verified_recovery_step = (
        int(recovery_payload["completed_updates"])
        if recovery_payload is not None
        else 0
    )

    def recovery_saved(step: int, path: Path) -> None:
        nonlocal last_verified_recovery_step
        mirror.mirror(path, mutable=True, recovery_step=step)
        advance = int(step) - last_verified_recovery_step
        maximum_advance = int(hard_wall_policy["recovery_interval_updates"])
        if not 0 < advance <= maximum_advance:
            raise RuntimeError(
                "CAP2 recovery cadence exceeded its declared hard-wall detection "
                f"interval: previous={last_verified_recovery_step}, current={step}, "
                f"maximum_advance={maximum_advance}"
            )
        last_verified_recovery_step = int(step)
        cumulative_hours = (prior_wall_seconds + time.time() - started) / 3600.0
        if cumulative_hours > hard_wall_hours:
            raise RuntimeError(
                "CAP2 hard wall-time stop reached after a verified recovery at "
                f"step {step}: {cumulative_hours:.3f}h > {hard_wall_hours:.3f}h; "
                "detection cadence was bounded to one recovery interval"
            )

    outcome = train_cap_unit(
        pool,
        frozen,
        device,
        recovery_path=recovery,
        checkpoint=checkpoint,
        snapshot=snapshot,
        checkpoint_health_observer=checkpoint_health_observer,
        recovery_saved=recovery_saved,
        progress=lambda message: print(message, flush=True),
        recovery_identity=identity_payload,
        recovery_authorization=recovery_authorization,
        unit_seed=args.unit_seed,
        stop_after_updates=execution_stop,
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
        outcome.inference_forward_calls,
        frozen.gate,
    )
    snapshot_records = [
        {
            "step": int(path.stem.rsplit("step", 1)[1]),
            "path": _relative(path, root),
            "sha256": verify_file(path),
            "durable_mirror": mirror.verify(path),
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
            "overridden": realized_payload != declared_payload,
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
            "examples_seen_target": execution_stop * frozen.train.effective_batch,
            "objective_sample_evaluations": outcome.model_forwards,
            "objective_forward_calls": outcome.objective_forward_calls,
            "inference_forward_calls": outcome.inference_forward_calls,
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
            "durable_mirror": mirror.verify_recovery(
                recovery, recovery_step=execution_stop
            ),
        },
        "foundation_pause": (
            {
                "planned_updates": 750_000,
                "paused_at": 50_000,
                "purpose": "raw-state numerical admission before continuation",
            }
            if campaign == "ordered_750_foundation" and execution_stop == 50_000
            else None
        ),
        "durability": {
            "required": True,
            "synchronous": True,
            "authorization_workspace": workspace_record,
            "live_storage_capacity": live_storage,
            "mirror_root": str(args.durable_mirror_dir.resolve()),
            "run_identity": identity_mirror,
            "live_roundtrip_probe": mirror_probe,
            "preexisting_sealed_artifacts_synced": len(preexisting_mirror_records),
            "hard_cumulative_wall_hours": hard_wall_hours,
            "hard_wall_policy": {
                **hard_wall_policy,
                "prior_committed_wall_hours": prior_wall_hours,
                "last_verified_recovery_step": last_verified_recovery_step,
                "semantics": (
                    "process-wall stop checked only after a durable recovery; "
                    "provider-side dollar controls remain independently required"
                ),
            },
            "note": (
                "operator confirmed the mirror survives deletion of the GPU instance"
            ),
        },
        "early_admission": (
            {
                "path": _relative(args.early_admission, root),
                "sha256": early_admission_record["artifact_sha256"],
            }
            if consumes_early_admission
            else early_admission_record
        ),
        "foundation_continuation_authority": foundation_continuation_authority,
        "train_only_gate": gate,
        "elapsed_seconds": time.time() - started,
        "next_step": (
            "run raw numerical admission on the bound 50k checkpoint; do not continue yet"
            if campaign == "ordered_750_foundation" and execution_stop == 50_000
            else "eligible for foundation capability evaluation and ASFD qualification"
            if campaign == "ordered_750_foundation" and gate["verdict"] == "PASS"
            else "eligible for fixed 300k promotion review"
            if args.updates == 150_000 and gate["verdict"] == "PASS"
            else "requires comparative review; never auto-promote to confirmation"
        ),
        "limits": [
            "No CIFAR-10 test image is opened by this runner.",
            "One developmental arm and seed; no general performance claim.",
            "This result never auto-authorizes ASFD or sealed-test evaluation.",
        ],
    }
    digest = write_json_atomic(result_path, result)
    result_mirror = mirror.mirror(result_path)
    print(json.dumps({"arm": args.arm, "gate": gate}, indent=2))
    print(
        f"wrote {result_path} sha256={digest}; "
        f"durable mirror verified at {result_mirror['relative_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
