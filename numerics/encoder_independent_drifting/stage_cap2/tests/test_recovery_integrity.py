"""Disjoint P0 tests for rolling CAP2 recovery integrity and promotion state."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from ...stage_cap.diagnostics import capability_gate
from ...stage_cap.training import (
    _atomic_save,
    _recovery_identity,
    load_recovery_payload,
    recovery_sidecar,
    validate_recovery_counters,
)
from ..artifacts import write_json_atomic
from ..config import screen_profile
from ..run_screen import (
    _assert_recovery_authorization,
    _assert_recovery_matches_150k_checkpoints,
    _assert_result_binds_recovery,
    _promotion_recovery_authorization,
    _quarantine_incomplete_terminal_result,
    _verify_completed_terminal_result,
)


def _raises(fragment: str, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - deliberate boundary testing
        assert fragment in str(error), (fragment, type(error).__name__, str(error))
    else:  # pragma: no cover - assertion branch
        raise AssertionError(f"expected failure containing {fragment!r}")


def _authorization(character: str = "a") -> dict:
    digest = character * 64
    return _promotion_recovery_authorization(
        {
            "artifact_sha256": digest,
            "preflight_sha256": digest,
            "result_sha256": digest,
            "checkpoint_sha256": digest,
            "raw_checkpoint_sha256": digest,
            "readmission_sha256": digest,
            "development_evaluation_sha256": digest,
            "arm": "ordered_uniform",
            "candidate": "local_1000_d0002_fp32",
        },
        {"artifact_sha256": digest},
    )


class _TerminalMirror:
    def __init__(self, record: dict) -> None:
        self.record = record

    def verify_recovery(self, _path: Path, *, recovery_step: int) -> dict:
        assert recovery_step == 300_000
        return self.record


def test_completed_terminal_result_is_idempotently_revalidated() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        result_path = root / "result_300000.json"
        recovery_path = root / "checkpoints" / "recovery.pt"
        mirror_record = {
            "relative_path": "checkpoints/recovery.pt",
            "sha256": "r" * 64,
            "bytes": 123,
            "recovery_step": 300_000,
            "version_relative_path": "versions/recovery.pt",
            "commit_relative_path": "commits/recovery.json",
        }
        authorization = _authorization()
        health_final = {
            "second_moment_ratio": 1.0,
            "centered_variance_ratio": 1.0,
            "effective_rank_ratio": 1.0,
            "haar_LL_ratio": 1.0,
            "haar_LH_ratio": 1.0,
            "haar_HL_ratio": 1.0,
            "haar_HH_ratio": 1.0,
            "raw_saturation_fraction": 0.0,
        }
        health = [{"step": 300_000, "ema": health_final}]
        recovery = {
            "artifact_sha256": "r" * 64,
            "planned_updates": 300_000,
            "completed_updates": 300_000,
            "continuation_authorization": authorization,
            "history": [{"step": 300_000}],
            "health": health,
            "checkpoints": {},
            "snapshots": [],
            "wall_seconds": 10.0,
            "peak_memory_bytes": 100,
            "peak_memory_reserved_bytes": 200,
            "optimizer_updates": 300_000,
            "examples_seen": 9_600_000,
            "model_forwards": 28_800_000,
            "objective_forward_calls": 900_000,
            "clipped_updates": 2,
            "clipped_updates_final_window": 1,
            "final_window_updates": 100,
            "nonfinite_updates": 0,
        }
        workspace = {
            "root": str(root.resolve()),
            "attestation": {"status": "durable-root-attestation"},
            "live_roundtrip_probe": {"roundtrip_verified": True},
        }
        profile = screen_profile("ordered_uniform", "local_1000_d0002_fp32")
        declared_profile = {
            "name": "declared",
            "train": {
                "micro_batch": 4,
                "accumulation_steps": 8,
                "effective_batch": 32,
                "checkpoint_updates": [],
            },
        }
        realized_profile = deepcopy(declared_profile)
        gate = capability_gate(health_final, 1.0, 0.01, 0, 1, profile.gate)
        run_identity_mirror = {
            "relative_path": "run_identity.json",
            "sha256": "i" * 64,
            "bytes": 10,
        }
        hard_wall_policy = {
            "hard_cumulative_wall_hours": 12.0,
            "recovery_interval_updates": 1_000,
            "maximum_detection_overshoot_updates": 1_000,
            "conservative_projected_maximum_detection_overshoot_hours": 0.1,
            "conservative_projected_maximum_detected_wall_hours": 12.1,
        }
        storage_plan = {
            "storage_root": str(root.resolve()),
            "measured_unit_bytes": {
                "recovery": 10,
                "checkpoint_raw": 10,
                "checkpoint_ema": 10,
                "snapshot": 10,
            },
            "projected_bytes": {"required_with_contingency": 1_000},
        }
        payload = {
            "status": "cap-emf2-screen-unit",
            "arm": "ordered_uniform",
            "numerical_candidate": "local_1000_d0002_fp32",
            "preflight_sha256": "p" * 64,
            "run_identity_sha256": "i" * 64,
            "unit_seed": 0,
            "declared_profile": declared_profile,
            "realized_profile": realized_profile,
            "realized_batch_split": {
                "micro_batch": 4,
                "accumulation_steps": 8,
                "effective_batch": 32,
                "overridden": False,
            },
            "precision": {"allow_tf32": False},
            "deterministic_algorithms": True,
            "device": {"gpu_name": "test"},
            "hardware": {"status": "test-hardware"},
            "training": {
                "history": recovery["history"],
                "health": health,
                "optimizer_updates": 300_000,
                "examples_seen": 9_600_000,
                "examples_seen_target": 9_600_000,
                "objective_sample_evaluations": 28_800_000,
                "objective_forward_calls": 900_000,
                "inference_forward_calls": 1,
                "clipped_updates": 2,
                "clipped_updates_final_window": 1,
                "final_window_updates": 100,
                "clip_fraction_final_window": 0.01,
                "nonfinite_updates": 0,
                "wall_seconds": 10.1,
                "peak_memory_bytes": 100,
                "peak_memory_reserved_bytes": 200,
            },
            "checkpoints": {},
            "raw_snapshots": [],
            "recovery": {
                "path": "checkpoints/recovery.pt",
                "sha256": "r" * 64,
                "planned_updates": 300_000,
                "completed_updates": 300_000,
                "continuation_authorization": authorization,
                "durable_mirror": mirror_record,
            },
            "durability": {
                "required": True,
                "synchronous": True,
                "authorization_workspace": workspace,
                "mirror_root": str((root / "mirror").resolve()),
                "run_identity": run_identity_mirror,
                "live_roundtrip_probe": {"roundtrip_verified": True},
                "preexisting_sealed_artifacts_synced": 0,
                "live_storage_capacity": {
                    "storage_root": str(root.resolve()),
                    "total_bytes": 2_000,
                    "free_bytes": 1_500,
                    "required_campaign_bytes": 1_000,
                    "minimum_transaction_headroom_bytes": 80,
                },
                "hard_cumulative_wall_hours": 12.0,
                "hard_wall_policy": {
                    **hard_wall_policy,
                    "prior_committed_wall_hours": 0.0,
                    "last_verified_recovery_step": 300_000,
                    "semantics": "test",
                },
            },
            "train_only_gate": deepcopy(gate),
            "elapsed_seconds": 10.2,
        }
        write_json_atomic(result_path, payload)
        verified = _verify_completed_terminal_result(
            result_path,
            arm="ordered_uniform",
            candidate="local_1000_d0002_fp32",
            updates=300_000,
            planned_updates=300_000,
            preflight_sha256="p" * 64,
            run_identity_sha256="i" * 64,
            unit_seed=0,
            declared_profile=declared_profile,
            realized_profile=realized_profile,
            expected_examples=9_600_000,
            gate_config=profile.gate,
            deterministic_algorithms=True,
            precision={"allow_tf32": False},
            device={"gpu_name": "test"},
            hardware={"status": "test-hardware"},
            recovery=recovery,
            recovery_path=recovery_path,
            root=root,
            mirror=_TerminalMirror(mirror_record),
            workspace_record=workspace,
            mirror_root=root / "mirror",
            run_identity_mirror=run_identity_mirror,
            hard_wall_policy=hard_wall_policy,
            storage_plan=storage_plan,
        )
        assert verified["artifact_sha256"]

        def republish() -> None:
            result_path.unlink()
            result_path.with_suffix(result_path.suffix + ".sha256").unlink()
            write_json_atomic(result_path, payload)

        common = {
            "arm": "ordered_uniform",
            "candidate": "local_1000_d0002_fp32",
            "updates": 300_000,
            "planned_updates": 300_000,
            "preflight_sha256": "p" * 64,
            "run_identity_sha256": "i" * 64,
            "unit_seed": 0,
            "declared_profile": declared_profile,
            "realized_profile": realized_profile,
            "expected_examples": 9_600_000,
            "gate_config": profile.gate,
            "deterministic_algorithms": True,
            "precision": {"allow_tf32": False},
            "device": {"gpu_name": "test"},
            "hardware": {"status": "test-hardware"},
            "recovery": recovery,
            "recovery_path": recovery_path,
            "root": root,
            "mirror": _TerminalMirror(mirror_record),
            "workspace_record": workspace,
            "mirror_root": root / "mirror",
            "run_identity_mirror": run_identity_mirror,
            "hard_wall_policy": hard_wall_policy,
            "storage_plan": storage_plan,
        }
        payload["training"]["optimizer_updates"] = 299_999
        republish()
        _raises(
            "inconsistent training counters",
            _verify_completed_terminal_result,
            result_path,
            **common,
        )
        payload["training"]["optimizer_updates"] = 300_000

        payload["train_only_gate"]["verdict"] = "FORGED_PASS"
        republish()
        _raises(
            "recomputed train-only gate",
            _verify_completed_terminal_result,
            result_path,
            **common,
        )
        payload["train_only_gate"] = deepcopy(gate)

        payload["training"]["history"] = [{"step": 299_999}]
        republish()
        _raises(
            "recovery ledger",
            _verify_completed_terminal_result,
            result_path,
            **common,
        )
        payload["training"]["history"] = recovery["history"]

        payload["durability"]["hard_cumulative_wall_hours"] = 99.0
        republish()
        _raises(
            "durability bindings",
            _verify_completed_terminal_result,
            result_path,
            **common,
        )


def test_incomplete_terminal_result_is_quarantined() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        result_path = root / "result_300000.json"
        result_path.write_text("partial", encoding="utf-8")
        record = _quarantine_incomplete_terminal_result(result_path)
        assert record is not None and "payload" in record
        assert not result_path.exists()
        assert Path(record["payload"]).read_text(encoding="utf-8") == "partial"


def _strict_payload(
    *,
    planned: int = 4,
    completed: int = 2,
    external: dict | None = None,
    authorization: dict | None = None,
    nonfinite: int = 0,
) -> dict:
    profile = screen_profile("ordered_uniform", "local_1000_d0002_fp32", smoke=True)
    identity = _recovery_identity(profile, external or {"test": True}, 0)
    train = profile.train
    examples = completed * train.effective_batch
    calls = completed * train.accumulation_steps
    mode = profile.objective.stopped_evaluation
    multiplier = 4 if mode == "fp32_dense" else 3
    origin = max(0, planned - profile.gate.clip_window_updates)
    return {
        "stage": "cap-emf-1-recovery",
        "profile_name": profile.name,
        "recovery_identity": identity,
        "continuation_authorization": authorization,
        "planned_updates": planned,
        "completed_updates": completed,
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "optimizer": {"state": {}, "param_groups": []},
        "ema": {
            "decay": train.ema_decay,
            "updates": completed - nonfinite,
            "shadow": {"weight": torch.tensor([0.5, 1.5])},
            "buffers": {"counter": torch.tensor(3, dtype=torch.int64)},
        },
        "generators": {
            "data": torch.tensor([1], dtype=torch.uint8),
            "noise": torch.tensor([2], dtype=torch.uint8),
            "time": torch.tensor([3], dtype=torch.uint8),
            "diagonal": torch.tensor([4], dtype=torch.uint8),
            "flip": torch.tensor([5], dtype=torch.uint8),
        },
        "history": [{"step": completed}],
        "health": [{"step": completed}],
        "checkpoints": {},
        "snapshots": [],
        "wall_seconds": 1.0,
        "peak_memory_bytes": 0,
        "peak_memory_reserved_bytes": 0,
        "optimizer_updates": completed,
        "examples_seen": examples,
        "model_forwards": multiplier * examples,
        "objective_forward_calls": multiplier * calls,
        "clipped_updates": 0,
        "clipped_updates_final_window": 0,
        "final_window_updates": max(0, completed - origin),
        "final_window_origin": origin,
        "nonfinite_updates": nonfinite,
        "best_rank_ratio": 1.0,
        "objective_ledger": {},
    }


def test_atomic_recovery_roundtrip_and_rolling_replacement() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "recovery.pt"
        first = _strict_payload()
        first_sha = _atomic_save(first, path)
        assert path.is_file() and recovery_sidecar(path).is_file()
        loaded, loaded_sha = load_recovery_payload(path, require_sidecar=True)
        assert loaded_sha == first_sha
        assert loaded["best_rank_ratio"] == 1.0

        second = deepcopy(first)
        second["best_rank_ratio"] = 2.0
        second_sha = _atomic_save(second, path)
        loaded, loaded_sha = load_recovery_payload(path, require_sidecar=True)
        assert loaded_sha == second_sha != first_sha
        assert loaded["best_rank_ratio"] == 2.0


def test_strict_recovery_rejects_payload_and_sidecar_tampering() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "recovery.pt"
        payload = _strict_payload()
        _atomic_save(payload, path)
        with path.open("ab") as handle:
            handle.write(b"tamper")
        _raises("SHA mismatch", load_recovery_payload, path, require_sidecar=True)

        _atomic_save(payload, path)
        recovery_sidecar(path).write_text("not-a-digest\n", encoding="utf-8")
        _raises("malformed", load_recovery_payload, path, require_sidecar=True)


def test_legacy_missing_sidecar_is_allowed_but_cap2_fails_closed() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "recovery.pt"
        legacy = _strict_payload()
        legacy["recovery_identity"] = None
        _atomic_save(legacy, path)
        recovery_sidecar(path).unlink()
        loaded, _ = load_recovery_payload(path, require_sidecar=False)
        assert loaded["completed_updates"] == 2
        _raises("sidecar is missing", load_recovery_payload, path, require_sidecar=True)


def test_strict_recovery_rejects_self_consistently_hashed_bad_counters() -> None:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "recovery.pt"
        payload = _strict_payload(nonfinite=1)
        validate_recovery_counters(payload, strict=True)
        payload["examples_seen"] += 1
        _atomic_save(payload, path)
        _raises("example counter", load_recovery_payload, path, require_sidecar=True)

        payload = _strict_payload(nonfinite=1)
        payload["ema"]["updates"] += 1
        _atomic_save(payload, path)
        _raises("EMA maturity", load_recovery_payload, path, require_sidecar=True)


def test_150k_recovery_must_equal_both_recorded_checkpoint_states() -> None:
    recovery = {
        "planned_updates": 150_000,
        "completed_updates": 150_000,
        "model": {
            "weight": torch.tensor([1.0, 2.0]),
            "counter": torch.tensor(7, dtype=torch.int64),
        },
        "ema": {
            "shadow": {"weight": torch.tensor([0.5, 1.5])},
            "buffers": {"counter": torch.tensor(7, dtype=torch.int64)},
        },
    }
    raw = {"state_dict": deepcopy(recovery["model"])}
    ema = {
        "state_dict": {
            **deepcopy(recovery["ema"]["shadow"]),
            **deepcopy(recovery["ema"]["buffers"]),
        }
    }
    _assert_recovery_matches_150k_checkpoints(
        recovery, raw_checkpoint=raw, ema_checkpoint=ema
    )
    bad_raw = deepcopy(raw)
    bad_raw["state_dict"]["weight"][0] += 1
    _raises(
        "150k raw",
        _assert_recovery_matches_150k_checkpoints,
        recovery,
        raw_checkpoint=bad_raw,
        ema_checkpoint=ema,
    )
    bad_ema = deepcopy(ema)
    bad_ema["state_dict"].pop("counter")
    _raises(
        "150k EMA",
        _assert_recovery_matches_150k_checkpoints,
        recovery,
        raw_checkpoint=raw,
        ema_checkpoint=bad_ema,
    )


def test_promoted_recovery_requires_and_preserves_exact_authorization() -> None:
    authorization = _authorization("a")
    payload = _strict_payload(
        planned=300_000,
        completed=285_000,
        external={"status": "cap-emf2-run-identity"},
        authorization=authorization,
    )
    validate_recovery_counters(payload, strict=True)
    _assert_recovery_authorization(payload, authorization)

    missing = deepcopy(payload)
    missing["continuation_authorization"] = None
    _raises("lacks continuation", validate_recovery_counters, missing, strict=True)
    _raises(
        "not bound",
        _assert_recovery_authorization,
        payload,
        _authorization("b"),
    )


def test_immutable_150k_result_binds_full_recovery_sha() -> None:
    root = Path("run")
    path = root / "checkpoints" / "recovery.pt"
    recovery = {
        "artifact_sha256": "c" * 64,
        "planned_updates": 150_000,
        "completed_updates": 150_000,
    }
    durable = {
        "relative_path": "checkpoints/recovery.pt",
        "sha256": "c" * 64,
        "bytes": 123,
        "recovery_step": 150_000,
        "version_relative_path": "versions/recovery.pt",
        "commit_relative_path": "commits/150000.json",
    }
    mirror = SimpleNamespace(
        verify_recovery=lambda _path, *, recovery_step: (
            durable if recovery_step == 150_000 else None
        )
    )
    result = {
        "recovery": {
            "path": "checkpoints/recovery.pt",
            "sha256": "c" * 64,
            "planned_updates": 150_000,
            "completed_updates": 150_000,
            "continuation_authorization": None,
            "durable_mirror": durable,
        }
    }
    _assert_result_binds_recovery(
        result, recovery, recovery_path=path, root=root, mirror=mirror
    )
    bad = deepcopy(result)
    bad["recovery"]["sha256"] = "d" * 64
    _raises(
        "does not bind",
        _assert_result_binds_recovery,
        bad,
        recovery,
        recovery_path=path,
        root=root,
        mirror=mirror,
    )
