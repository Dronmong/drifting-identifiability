"""Explicit trust boundary for CAP-EMF-2 developmental screens."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

from ..diagnostics import ROOT

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
PROTOCOL = ROOT / "numerics" / "EncoderIndependentCAPEMF2ASFDRunPodProtocol.md"
DEFAULT_PREFLIGHT = HERE / "cap2_preflight.json"

_DEPENDENCIES = (
    "stage_cap/__init__.py",
    "stage_cap/config.py",
    "stage_cap/data.py",
    "stage_cap/diagnostics.py",
    "stage_cap/model.py",
    "stage_cap/monitoring.py",
    "stage_cap/objective.py",
    "stage_cap/training.py",
    "stage_cap2/__init__.py",
    "stage_cap2/artifacts.py",
    "stage_cap2/benchmark.py",
    "stage_cap2/budget.py",
    "stage_cap2/checkpoint_forensics.py",
    "stage_cap2/config.py",
    "stage_cap2/development_evaluation.py",
    "stage_cap2/durable_mirror.py",
    "stage_cap2/early_admission.py",
    "stage_cap2/final_verdict.py",
    "stage_cap2/foundation_gate.py",
    "stage_cap2/foundation_visual_review.py",
    "stage_cap2/gate_calibration.py",
    "stage_cap2/hardware.py",
    "stage_cap2/metric_calibration.py",
    "stage_cap2/numerical_admission.py",
    "stage_cap2/preflight.py",
    "stage_cap2/positive_control.py",
    "stage_cap2/preview.py",
    "stage_cap2/promotion.py",
    "stage_cap2/production_readiness.py",
    "stage_cap2/requirements-eval.txt",
    "stage_cap2/requirements-positive-control.txt",
    "stage_cap2/requirements-production-cu126.txt",
    "stage_cap2/runpod_bootstrap.sh",
    "stage_cap2/runpod_pipeline.sh",
    "stage_cap2/run_screen.py",
    "stage_cap2/sampler_audit.py",
    "stage_cap2/selection.py",
    "stage_cap2/standard_metrics.py",
    # The paid campaign authorizes one foundation *and* its prospectively
    # frozen ASFD continuation.  Binding only CAP2 would allow the correction
    # mechanism to be edited after seeing the 750k foundation.
    "stage_asfd/__init__.py",
    "stage_asfd/artifacts.py",
    "stage_asfd/calibration.py",
    "stage_asfd/config.py",
    "stage_asfd/continuation.py",
    "stage_asfd/correction.py",
    "stage_asfd/evaluation.py",
    "stage_asfd/feature_bank.py",
    "stage_asfd/features.py",
    "stage_asfd/field.py",
    "stage_asfd/final_report.py",
    "stage_asfd/final_visual_review.py",
    "stage_asfd/gradients.py",
    "stage_asfd/preflight.py",
    "stage_asfd/qualification.py",
    "stage_asfd/qualify.py",
    "stage_asfd/recovery.py",
    "stage_b2/metrics.py",
    "spectral_anchor.py",
    "appearance.py",
    "config.py",
    "device.py",
    "diagnostics.py",
    "fid.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash64(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".sha256")


def _atomic_replace_bytes(path: Path, raw: bytes) -> None:
    """Write bytes beside ``path`` and publish them with one atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    # NumPy scalar values occur in diagnostics, but importing NumPy at the
    # artifact boundary would make this low-level module needlessly heavy.
    if type(value).__module__.startswith("numpy") and hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def write_json_atomic(path: Path, payload: dict) -> str:
    """Publish canonical JSON and its SHA sidecar without partial final files."""
    raw = (
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = _sidecar(path)
    if path.exists() or sidecar.exists():
        raise RuntimeError(f"refusing to overwrite consumed CAP2 artifact {path}")
    _atomic_replace_bytes(path, raw)
    try:
        _atomic_replace_bytes(sidecar, f"{digest}  {path.name}\n".encode())
    except Exception:
        # A final file without its authenticity companion is unusable and would
        # otherwise block a clean retry through ``assert_unused``.
        path.unlink(missing_ok=True)
        raise
    return digest


def write_npz_atomic(path: Path, **arrays) -> str:
    """Publish an uncompressed NumPy archive and SHA sidecar atomically."""

    import numpy as np

    sidecar = _sidecar(path)
    if path.exists() or sidecar.exists():
        raise RuntimeError(f"refusing to overwrite consumed CAP2 artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        digest = file_sha256(temporary)
        os.replace(temporary, path)
        try:
            _atomic_replace_bytes(sidecar, f"{digest}  {path.name}\n".encode())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def write_sha256_sidecar_atomic(path: Path, expected_sha: str | None = None) -> str:
    """Seal an already-published immutable file with the standard sidecar.

    Image writers cannot use :func:`write_json_atomic`, but preview grids need
    the same hash-verification and durable-mirror path as every other paid-run
    artifact.  The payload must already exist and the sidecar must not; callers
    therefore cannot silently bless an overwritten file.
    """

    if not path.is_file():
        raise RuntimeError(f"cannot seal missing artifact: {path}")
    sidecar = _sidecar(path)
    if sidecar.exists():
        raise RuntimeError(f"refusing to overwrite consumed SHA sidecar {sidecar}")
    digest = file_sha256(path)
    if expected_sha is not None and digest != expected_sha:
        raise RuntimeError(f"unexpected SHA for {path}: {digest} != {expected_sha}")
    _atomic_replace_bytes(sidecar, f"{digest}  {path.name}\n".encode())
    return digest


def verify_file(path: Path, expected_sha: str | None = None) -> str:
    sidecar = _sidecar(path)
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"artifact or SHA sidecar missing: {path}")
    recorded = sidecar.read_text(encoding="utf-8").split()[0]
    actual = file_sha256(path)
    if recorded != actual:
        raise RuntimeError(f"SHA mismatch for {path}")
    if expected_sha is not None and actual != expected_sha:
        raise RuntimeError(f"unexpected SHA for {path}: {actual} != {expected_sha}")
    return actual


def source_manifest() -> dict[str, str]:
    paths = [PACKAGE / name for name in _DEPENDENCIES]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"declared CAP2 dependency missing: {missing}")
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): file_sha256(path)
        for path in sorted(set(paths))
    }


def verify_json(path: Path, status: str | None = None) -> dict:
    actual = verify_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if status is not None and payload.get("status") != status:
        raise RuntimeError(f"{path} has wrong status {payload.get('status')!r}")
    payload["artifact_sha256"] = actual
    return payload


def profile_payload(profile) -> dict:
    profile.validate()
    return json.loads(json.dumps(asdict(profile), sort_keys=True))


def load_preflight(path: Path = DEFAULT_PREFLIGHT) -> dict:
    """Load and independently reconstruct a CAP2 authorization.

    A matching SHA sidecar proves only that the file has not changed since it
    was written.  It does not make a fabricated ``decision: GO`` trustworthy.
    Recompute the pure admission predicate, rerun the tiny all-arm smoke, and
    reconstruct every frozen profile before an expensive command may consume
    the artifact.
    """
    payload = verify_json(path, "cap-emf2-preflight")
    if payload.get("protocol_sha256") != file_sha256(PROTOCOL):
        raise RuntimeError("CAP2 protocol changed after preflight")
    live = source_manifest()
    if payload.get("source_sha256") != live:
        changed = sorted(
            name
            for name in set(live) & set(payload.get("source_sha256", {}))
            if live[name] != payload["source_sha256"][name]
        )
        raise RuntimeError(f"CAP2 sources changed after preflight: {changed}")
    from .budget import revalidate_budget_plan, revalidate_storage_plan
    from .config import SAMPLER_ARMS, apply_calibrated_gate, screen_profile
    from .preflight import _smoke_all_arms, validate_preflight_inputs

    required_inputs = {
        "numerical_admission",
        "sampler_audit",
        "gate_calibration",
        "benchmark",
        "baseline_standard",
        "positive_control_standard",
        "metric_calibration",
        "checkpoint_forensics",
    }
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != required_inputs:
        raise RuntimeError("CAP2 preflight has an incomplete input ledger")
    decision, checks = validate_preflight_inputs(
        numerical=inputs["numerical_admission"],
        samplers=inputs["sampler_audit"],
        calibration=inputs["gate_calibration"],
        benchmark=inputs["benchmark"],
        baseline=inputs["baseline_standard"],
        positive_control=inputs["positive_control_standard"],
        metric_calibration=inputs["metric_calibration"],
        forensics=inputs["checkpoint_forensics"],
        live_sources=live,
        # The evaluation environment may be separate from training.  The
        # original preflight already bound the installed version; reconstruct
        # that recorded claim rather than requiring CleanFID on every runner.
        installed_cleanfid_version=payload.get("cleanfid_version"),
    )
    candidate = inputs["numerical_admission"].get("candidate", {}).get("name")
    if not isinstance(candidate, str) or payload.get("candidate") != candidate:
        raise RuntimeError("CAP2 preflight candidate is inconsistent")
    smoke = _smoke_all_arms(candidate, inputs["gate_calibration"])
    smoke_ok = all(record.get("verdict") == "PASS" for record in smoke.values())
    checks["all_arm_smoke"] = smoke_ok
    budget = revalidate_budget_plan(payload.get("budget"), inputs["benchmark"])
    checks["aggregate_budget_within_ceiling"] = budget["within_ceiling"] is True
    storage = revalidate_storage_plan(payload.get("storage"), inputs["benchmark"])
    checks["durable_storage_capacity"] = storage["decision"] == "GO"
    retained = payload.get("retained_metric_evidence")
    checks["retained_metric_leaves_recomputed"] = (
        isinstance(retained, dict)
        and set(retained) == {"baseline", "positive_control", "metric_calibration"}
        and all(
            isinstance(record, dict) and record.get("valid") is True
            for record in retained.values()
        )
    )
    decision = (
        "GO"
        if decision == "GO"
        and smoke_ok
        and checks["aggregate_budget_within_ceiling"]
        and checks["durable_storage_capacity"]
        and checks["retained_metric_leaves_recomputed"]
        else "NO_GO"
    )
    expected_profiles = {
        arm: profile_payload(
            apply_calibrated_gate(
                screen_profile(arm, candidate, updates=150_000),
                inputs["gate_calibration"],
            )
        )
        for arm in SAMPLER_ARMS
    }
    expected_foundation = profile_payload(
        apply_calibrated_gate(
            screen_profile("ordered_uniform", candidate, updates=750_000),
            inputs["gate_calibration"],
        )
    )
    failures = []
    if payload.get("decision") != decision or decision != "GO":
        failures.append("decision")
    if payload.get("checks") != checks:
        failures.append("checks")
    if payload.get("smoke") != smoke:
        failures.append("smoke")
    if payload.get("profiles_150k") != expected_profiles:
        failures.append("profiles_150k")
    if payload.get("foundation_profile_750k") != expected_foundation:
        failures.append("foundation_profile_750k")
    if payload.get("storage") != storage:
        failures.append("storage")
    if failures:
        raise RuntimeError(f"CAP2 preflight failed revalidation: {failures}")
    payload["revalidated"] = True
    return payload


def assert_unused(path: Path) -> None:
    if path.exists() or _sidecar(path).exists():
        raise RuntimeError(f"refusing to overwrite consumed CAP2 artifact {path}")


def _atomic_torch_save(payload: dict, path: Path) -> str:
    """Atomically publish a torch payload and a digest sidecar."""
    import torch

    sidecar = _sidecar(path)
    if path.exists() or sidecar.exists():
        raise RuntimeError(f"refusing to overwrite CAP2 torch artifact {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.partial"
    if temporary.exists():
        raise RuntimeError(f"stale temporary CAP2 artifact exists: {temporary}")
    try:
        torch.save(payload, temporary)
        # Re-open before publication.  This catches a truncated serialization,
        # not merely a path that happened to be created.
        try:
            torch.load(temporary, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - older torch
            torch.load(temporary, map_location="cpu")
        os.replace(temporary, path)
        digest = file_sha256(path)
        _atomic_replace_bytes(sidecar, f"{digest}  {path.name}\n".encode())
        return digest
    except Exception:
        temporary.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise


def _load_torch(path: Path) -> dict:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"CAP2 torch artifact is not a mapping: {path}")
    return payload


def _validate_state_dict(payload: dict, path: Path) -> None:
    import torch

    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"CAP2 artifact has no state_dict: {path}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise RuntimeError(f"CAP2 state_dict contains non-tensors: {path}")
    state_count = sum(value.numel() for value in state.values())
    fixed_buffers = {
        "time_embed.frequencies",
        "interval_embed.frequencies",
    }
    unknown_frequency_buffers = {
        name for name in state if name.endswith(".frequencies")
    } - fixed_buffers
    if unknown_frequency_buffers:
        raise RuntimeError(
            f"CAP2 state_dict has an unclassified frequency buffer in {path}"
        )
    trainable_count = sum(
        value.numel() for name, value in state.items() if name not in fixed_buffers
    )
    if int(payload.get("state_value_count", -1)) != state_count:
        raise RuntimeError(f"CAP2 state-value-count mismatch in {path}")
    if int(payload.get("parameter_count", -1)) != trainable_count:
        raise RuntimeError(f"CAP2 trainable-parameter-count mismatch in {path}")


def _state_counts(state: dict) -> tuple[int, int]:
    """Return trainable and serialized counts for the frozen CAP architecture."""
    fixed_buffers = {
        "time_embed.frequencies",
        "interval_embed.frequencies",
    }
    unknown = {name for name in state if name.endswith(".frequencies")} - fixed_buffers
    if unknown:
        raise RuntimeError(f"unclassified CAP2 frequency buffers: {sorted(unknown)}")
    serialized = sum(value.numel() for value in state.values())
    trainable = sum(
        value.numel() for name, value in state.items() if name not in fixed_buffers
    )
    return trainable, serialized


def save_checkpoint(
    path: Path,
    state: dict,
    *,
    step: int,
    kind: str,
    arm: str,
    declared_profile: dict,
    realized_profile: dict,
    preflight_sha256: str,
    run_identity_sha256: str,
    unit_seed: int,
) -> str:
    if kind not in {"raw", "ema"}:
        raise ValueError("CAP2 checkpoint kind must be raw or ema")
    if step <= 0 or unit_seed < 0 or not isinstance(arm, str) or not arm:
        raise ValueError("CAP2 checkpoint identity is malformed")
    if not _hash64(preflight_sha256) or not _hash64(run_identity_sha256):
        raise ValueError("CAP2 checkpoint authorization hash is malformed")
    if not isinstance(declared_profile, dict) or not isinstance(realized_profile, dict):
        raise TypeError("CAP2 checkpoint profiles must be mappings")
    parameter_count, state_value_count = _state_counts(state)
    payload = {
        "stage": "cap-emf-2-screen",
        "step": int(step),
        "kind": kind,
        "arm": arm,
        # ``profile`` remains as a compatibility alias for the evaluators.
        "profile": declared_profile,
        "declared_profile": declared_profile,
        "realized_profile": realized_profile,
        "preflight_sha256": preflight_sha256,
        "run_identity_sha256": run_identity_sha256,
        "unit_seed": int(unit_seed),
        "parameter_count": parameter_count,
        "state_value_count": state_value_count,
        "state_dict": {name: value.detach().cpu() for name, value in state.items()},
    }
    return _atomic_torch_save(payload, path)


def load_checkpoint(
    path: Path,
    *,
    expected_sha: str | None = None,
    step: int | None = None,
    kind: str | None = None,
    arm: str | None = None,
    declared_profile: dict | None = None,
    realized_profile: dict | None = None,
    preflight_sha256: str | None = None,
    run_identity_sha256: str | None = None,
    unit_seed: int | None = None,
) -> dict:
    digest = verify_file(path, expected_sha)
    payload = _load_torch(path)
    checks = {
        "stage": payload.get("stage") == "cap-emf-2-screen",
        "step": step is None or int(payload.get("step", -1)) == int(step),
        "kind": kind is None or payload.get("kind") == kind,
        "arm": arm is None or payload.get("arm") == arm,
        "declared_profile": (
            declared_profile is None
            or payload.get("declared_profile") == declared_profile
        ),
        "realized_profile": (
            realized_profile is None
            or payload.get("realized_profile") == realized_profile
        ),
        "preflight": (
            preflight_sha256 is None
            or payload.get("preflight_sha256") == preflight_sha256
        ),
        "run_identity": (
            run_identity_sha256 is None
            or payload.get("run_identity_sha256") == run_identity_sha256
        ),
        "stored_preflight_hash": _hash64(payload.get("preflight_sha256")),
        "stored_run_identity_hash": _hash64(payload.get("run_identity_sha256")),
        "stored_unit_seed": isinstance(payload.get("unit_seed"), int)
        and not isinstance(payload.get("unit_seed"), bool)
        and payload["unit_seed"] >= 0,
        "unit_seed": (
            unit_seed is None or int(payload.get("unit_seed", -1)) == int(unit_seed)
        ),
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise RuntimeError(f"invalid CAP2 checkpoint metadata {failed}: {path}")
    _validate_state_dict(payload, path)
    payload["artifact_sha256"] = digest
    return payload


def save_snapshot(
    path: Path,
    state: dict,
    *,
    step: int,
    arm: str,
    declared_profile: dict,
    realized_profile: dict,
    preflight_sha256: str,
    run_identity_sha256: str,
    unit_seed: int,
) -> str:
    if step <= 0 or unit_seed < 0 or not isinstance(arm, str) or not arm:
        raise ValueError("CAP2 snapshot identity is malformed")
    if not _hash64(preflight_sha256) or not _hash64(run_identity_sha256):
        raise ValueError("CAP2 snapshot authorization hash is malformed")
    if not isinstance(declared_profile, dict) or not isinstance(realized_profile, dict):
        raise TypeError("CAP2 snapshot profiles must be mappings")
    floating = {
        name: value.detach().cpu()
        for name, value in state.items()
        if value.is_floating_point()
    }
    parameter_count, state_value_count = _state_counts(floating)
    payload = {
        "stage": "cap-emf-2-raw-snapshot",
        "step": int(step),
        "kind": "raw-snapshot",
        "arm": arm,
        "profile": declared_profile,
        "declared_profile": declared_profile,
        "realized_profile": realized_profile,
        "preflight_sha256": preflight_sha256,
        "run_identity_sha256": run_identity_sha256,
        "unit_seed": int(unit_seed),
        "parameter_count": parameter_count,
        "state_value_count": state_value_count,
        "state_dict": floating,
    }
    return _atomic_torch_save(payload, path)


def load_snapshot(
    path: Path,
    *,
    expected_sha: str | None = None,
    step: int | None = None,
    arm: str | None = None,
    preflight_sha256: str | None = None,
    run_identity_sha256: str | None = None,
    unit_seed: int | None = None,
) -> dict:
    digest = verify_file(path, expected_sha)
    payload = _load_torch(path)
    checks = {
        "stage": payload.get("stage") == "cap-emf-2-raw-snapshot",
        "step": step is None or int(payload.get("step", -1)) == int(step),
        "arm": arm is None or payload.get("arm") == arm,
        "preflight": (
            preflight_sha256 is None
            or payload.get("preflight_sha256") == preflight_sha256
        ),
        "run_identity": (
            run_identity_sha256 is None
            or payload.get("run_identity_sha256") == run_identity_sha256
        ),
        "stored_preflight_hash": _hash64(payload.get("preflight_sha256")),
        "stored_run_identity_hash": _hash64(payload.get("run_identity_sha256")),
        "stored_unit_seed": isinstance(payload.get("unit_seed"), int)
        and not isinstance(payload.get("unit_seed"), bool)
        and payload["unit_seed"] >= 0,
        "unit_seed": (
            unit_seed is None or int(payload.get("unit_seed", -1)) == int(unit_seed)
        ),
    }
    failed = sorted(name for name, ok in checks.items() if not ok)
    if failed:
        raise RuntimeError(f"invalid CAP2 snapshot metadata {failed}: {path}")
    _validate_state_dict(payload, path)
    payload["artifact_sha256"] = digest
    return payload
