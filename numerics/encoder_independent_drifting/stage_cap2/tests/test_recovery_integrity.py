"""Disjoint P0 tests for rolling CAP2 recovery integrity and promotion state."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ...stage_cap.training import (
    _atomic_save,
    _recovery_identity,
    load_recovery_payload,
    recovery_sidecar,
    validate_recovery_counters,
)
from ..config import screen_profile
from ..run_screen import (
    _assert_recovery_authorization,
    _assert_recovery_matches_150k_checkpoints,
    _assert_result_binds_recovery,
    _promotion_recovery_authorization,
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


def _strict_payload(
    *,
    planned: int = 4,
    completed: int = 2,
    external: dict | None = None,
    authorization: dict | None = None,
    nonfinite: int = 0,
) -> dict:
    profile = screen_profile(
        "ordered_uniform", "local_1000_d0002_fp32", smoke=True
    )
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
    result = {
        "recovery": {
            "path": "checkpoints/recovery.pt",
            "sha256": "c" * 64,
            "planned_updates": 150_000,
            "completed_updates": 150_000,
            "continuation_authorization": None,
        }
    }
    _assert_result_binds_recovery(result, recovery, recovery_path=path, root=root)
    bad = deepcopy(result)
    bad["recovery"]["sha256"] = "d" * 64
    _raises(
        "does not bind",
        _assert_result_binds_recovery,
        bad,
        recovery,
        recovery_path=path,
        root=root,
    )
