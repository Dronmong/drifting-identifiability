"""CAP-EMF-1 training loop: EMA, checkpoints, health, and real recovery.

**This loop can resume mid-unit.**  B2.5 could not: it saved EMA weights at
three declared steps and nothing else, so an interrupted unit lost every
completed hour *and* the surviving checkpoints then blocked its own restart.
A 160,000-update rented run must not repeat that, so optimizer, model, EMA and
every RNG stream are written atomically on a fixed cadence and a restart
continues from the last verified point.

No correction of any kind appears here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Protocol

import torch
from torch import nn

from ..config import MASTER_SEED, derive_seed
from . import CAP_PHASE, CAP_UNIT
from .config import CAPObjectiveConfig, CAPProfile
from .data import flip_batch
from .diagnostics import component_health
from .model import CAPPixelTransformer, one_step_components, one_step_sample
from .monitoring import ObjectiveLedger
from .objective import TriangleSample, emf_loss, sample_time_triangle


def cap_seed(role: str, index: int = 0) -> int:
    return derive_seed(MASTER_SEED + 141_000, CAP_PHASE, CAP_UNIT, role, index)


def replicated_cap_seed(role: str, unit_seed: int = 0, index: int = 0) -> int:
    """Preserve unit zero exactly while making later matched units possible."""
    if unit_seed < 0:
        raise ValueError("unit seed must be nonnegative")
    if unit_seed == 0:
        return cap_seed(role, index)
    return derive_seed(
        MASTER_SEED + 141_000,
        CAP_PHASE,
        CAP_UNIT,
        "replicate",
        unit_seed,
        role,
        index,
    )


def _recovery_identity(
    profile: CAPProfile,
    external: dict | None,
    unit_seed: int,
) -> dict:
    """Canonical run identity, allowing only an explicit horizon extension.

    ``updates`` and the checkpoint ladder are plans rather than scientific
    state, so they are the only profile fields omitted.  Batch splitting,
    objective/gate settings, seeds and all caller-supplied environment binding
    remain part of the identity and therefore cannot change on resume.
    """
    profile_data = asdict(profile)
    profile_data["train"].pop("updates")
    profile_data["train"].pop("checkpoint_updates")
    payload = {
        "profile_horizon_neutral": profile_data,
        "unit_seed": int(unit_seed),
        "external": external or {},
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "payload": payload,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


@dataclass
class TrainOutcome:
    history: list[dict] = field(default_factory=list)
    health: list[dict] = field(default_factory=list)
    checkpoints: dict[str, dict] = field(default_factory=dict)
    snapshots: list[int] = field(default_factory=list)
    wall_seconds: float = 0.0
    peak_memory_bytes: int = 0
    peak_memory_reserved_bytes: int = 0
    optimizer_updates: int = 0
    examples_seen: int = 0
    # Historical name: this is a count of per-row sample evaluations, not
    # Python/module forward invocations.  Retained for recovery compatibility.
    model_forwards: int = 0
    objective_forward_calls: int = 0
    # Structural inference audit, measured with a real module hook after the
    # final optimizer update.  This is deliberately runtime-only: it proves
    # the sampler path actually invokes the network once instead of letting a
    # caller satisfy H8 by passing the literal integer ``1``.
    inference_forward_calls: int = 0
    # Wall time spent publishing recovery payloads during this process.  It is
    # deliberately a runtime timing (not scientific state) and is used by the
    # CAP2 benchmark to amortize rare recovery I/O at its real cadence.
    recovery_io_seconds: float = 0.0
    # Runtime-only instrumentation for CAP2's cadence-adjusted benchmark.
    # These values are not scientific/recovery state and intentionally cover
    # only health events executed in the current process.
    ordinary_health_seconds: float = 0.0
    checkpoint_health_seconds: float = 0.0
    ordinary_health_events: int = 0
    checkpoint_health_events: int = 0
    clipped_updates: int = 0
    clipped_updates_final_window: int = 0
    # Pre-clip global gradient norms, one per optimizer update.  Runtime-only
    # instrumentation, deliberately excluded from recovery state: it exists so
    # the clip threshold can be set from the observed distribution rather than
    # assumed, and so H7 can be evaluated on a measurement.
    gradient_norms: list[float] = field(default_factory=list)
    final_window_updates: int = 0
    nonfinite_updates: int = 0
    best_rank_ratio: float = 0.0
    parameter_count: int = 0
    # Optional source-bound correction records.  The base CAP path leaves this
    # empty; a continuation extension owns the schema of each record.
    auxiliary_history: list[dict] = field(default_factory=list)


class TrainingExtension(Protocol):
    """Audited hook for an objective continuation.

    The hook runs after the complete primary gradient has been accumulated and
    before the foundation's ordinary global clip.  It may add conservative
    objective gradients, but it may not step the optimizer or touch primary
    RNG streams.  Its identity and full replay state are persisted in every
    recovery so a resumed run cannot silently change the correction.
    """

    def identity(self) -> dict: ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, payload: dict) -> None: ...

    def apply(self, step: int, model: nn.Module) -> dict | None: ...


class EMAState:
    """Exponential moving average with an explicit maturity accounting."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must lie in [0,1)")
        self.decay = float(decay)
        self.updates = 0
        self.shadow = {
            name: value.detach().clone().float()
            for name, value in model.state_dict().items()
            if value.is_floating_point()
        }
        self.buffers = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if not value.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    value.detach().float(), alpha=1.0 - self.decay
                )
            else:
                self.buffers[name] = value.detach().clone()
        self.updates += 1

    def state_dict(self) -> dict:
        merged = {name: value.clone() for name, value in self.shadow.items()}
        merged.update({name: value.clone() for name, value in self.buffers.items()})
        return merged

    def initialization_weight(self) -> float:
        """Share of the average still contributed by the starting network."""
        return self.decay**self.updates

    def recovery_state(self) -> dict:
        return {
            "decay": self.decay,
            "updates": self.updates,
            "shadow": self.shadow,
            "buffers": self.buffers,
        }

    def load_recovery_state(self, payload: dict, device=None) -> None:
        """Restore the average, placing it on ``device``.

        The device argument is not optional in practice.  Recovery files are
        loaded with ``map_location="cpu"``, so on a GPU run the restored shadow
        arrives on the CPU while the model is on CUDA, and the first
        ``update`` then fails with a device mismatch.  ``model.load_state_dict``
        and ``Optimizer.load_state_dict`` both re-place tensors themselves, so
        the EMA was the only piece that needed this and the only piece that
        broke -- on the first real resume.
        """
        self.decay = float(payload["decay"])
        self.updates = int(payload["updates"])
        move = (
            (lambda t: t.clone().to(device))
            if device is not None
            else (lambda t: t.clone())
        )
        self.shadow = {name: move(value) for name, value in payload["shadow"].items()}
        self.buffers = {name: move(value) for name, value in payload["buffers"].items()}


#: Endpoint-weighted buckets.  The one-call sampler evaluates at t=1, and
#: logit-normal(0.8, 0.8) puts only ~4% of rows above 0.9, so the top bucket is
#: the one that decides whether the sampled configuration is trained at all.
T_BUCKETS: tuple[float, ...] = (0.0, 0.3, 0.6, 0.8, 0.9, 0.95, 1.0)


def bucket_by_time(
    t: torch.Tensor, per_sample: torch.Tensor
) -> dict[str, dict[str, float]]:
    """Mean error and share of rows per time bucket."""
    result: dict[str, dict[str, float]] = {}
    total = max(len(t), 1)
    for low, high in pairwise(T_BUCKETS):
        mask = (t >= low) & (t < high) if high < 1.0 else (t >= low) & (t <= high)
        count = int(mask.sum())
        result[f"{low:g}-{high:g}"] = {
            "share": count / total,
            "mean_raw_mse": float(per_sample[mask].mean()) if count else float("nan"),
            "count": count,
        }
    return result


def learning_rate_at(update: int, train) -> float:
    """Linear warmup, then constant.  Standard for CIFAR-10 at this scale."""
    if train.warmup_updates <= 0 or update >= train.warmup_updates:
        return train.learning_rate
    return train.learning_rate * (update + 1) / train.warmup_updates


def recovery_sidecar(path: Path) -> Path:
    """Return the authenticity companion for a rolling recovery file."""
    return path.with_suffix(path.suffix + ".sha256")


def _hash_handle(handle) -> str:
    digest = hashlib.sha256()
    while block := handle.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def _recorded_recovery_sha(path: Path, *, required: bool) -> str | None:
    sidecar = recovery_sidecar(path)
    if not sidecar.is_file():
        if required:
            raise RuntimeError(f"CAP recovery SHA sidecar is missing: {sidecar}")
        return None
    fields = sidecar.read_text(encoding="utf-8").split()
    if not fields or len(fields[0]) != 64:
        raise RuntimeError(f"CAP recovery SHA sidecar is malformed: {sidecar}")
    if len(fields) != 2 or fields[1] != path.name:
        raise RuntimeError(f"CAP recovery SHA sidecar names another file: {sidecar}")
    try:
        int(fields[0], 16)
    except ValueError as error:
        raise RuntimeError(
            f"CAP recovery SHA sidecar is malformed: {sidecar}"
        ) from error
    return fields[0].lower()


def verify_recovery_file(path: Path, *, require_sidecar: bool = False) -> str:
    """Verify a recovery and return its SHA-256 digest.

    CAP1 recoveries created before the sidecar protocol remain readable when
    ``require_sidecar`` is false.  Every CAP2 caller sets it to true and thus
    fails closed on a missing, stale, or malformed companion.
    """
    if not path.is_file():
        raise RuntimeError(f"CAP recovery file is missing: {path}")
    recorded = _recorded_recovery_sha(path, required=require_sidecar)
    with path.open("rb") as handle:
        actual = _hash_handle(handle)
    if recorded is not None and actual != recorded:
        raise RuntimeError(f"CAP recovery SHA mismatch: {path}")
    return actual


def _torch_load_handle(handle):
    handle.seek(0)
    try:
        return torch.load(handle, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        handle.seek(0)
        return torch.load(handle, map_location="cpu")


def load_recovery_payload(
    path: Path, *, require_sidecar: bool = False, validate_counters: bool = True
) -> tuple[dict, str]:
    """Hash and deserialize exactly the same open recovery file.

    Loading through the verified file handle avoids a verify-then-open race if
    another process atomically publishes a newer rolling recovery.
    """
    if not path.is_file():
        raise RuntimeError(f"CAP recovery file is missing: {path}")
    recorded = _recorded_recovery_sha(path, required=require_sidecar)
    with path.open("rb") as handle:
        actual = _hash_handle(handle)
        if recorded is not None and actual != recorded:
            raise RuntimeError(f"CAP recovery SHA mismatch: {path}")
        payload = _torch_load_handle(handle)
    if not isinstance(payload, dict) or payload.get("stage") != "cap-emf-1-recovery":
        raise RuntimeError(f"not a CAP-EMF-1 recovery file: {path}")
    if validate_counters:
        validate_recovery_counters(payload, strict=require_sidecar)
    return payload, actual


def _integer(payload: dict, key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"CAP recovery counter {key!r} is not an integer")
    return int(value)


def _validate_step_sequence(payload: dict, key: str, completed: int) -> None:
    records = payload.get(key)
    if not isinstance(records, list):
        raise TypeError(f"CAP recovery {key!r} is not a list")
    previous = -1
    for record in records:
        if not isinstance(record, dict):
            raise TypeError(f"CAP recovery {key!r} contains a non-record")
        step = record.get("step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(f"CAP recovery {key!r} contains an invalid step")
        if not previous < step <= completed:
            raise RuntimeError(f"CAP recovery {key!r} steps are not strictly ordered")
        previous = step


def validate_recovery_counters(payload: dict, *, strict: bool = False) -> None:
    """Validate the redundant accounting carried by a recovery payload.

    The checks are intentionally algebraic.  They catch a self-consistently
    hashed but truncated, spliced, or manually edited recovery before model or
    optimizer state is trusted.  Strict mode is used for every CAP2 load.
    """
    if not strict and payload.get("recovery_identity") is None:
        # The earliest CAP1 recovery schema predates horizon/identity/final-
        # window metadata.  Retain only invariants that can be established
        # from fields it actually carried; CAP2 never enters this branch.
        completed = _integer(payload, "completed_updates")
        updates = _integer(payload, "optimizer_updates")
        examples = _integer(payload, "examples_seen")
        model_evaluations = _integer(payload, "model_forwards")
        nonfinite = _integer(payload, "nonfinite_updates")
        clipped = _integer(payload, "clipped_updates")
        if completed < 0 or updates != completed:
            raise RuntimeError("legacy CAP recovery update counters disagree")
        if examples < 0 or model_evaluations < examples:
            raise RuntimeError("legacy CAP recovery sample counters are impossible")
        if not 0 <= nonfinite <= completed or not 0 <= clipped <= completed:
            raise RuntimeError("legacy CAP recovery health counters are impossible")
        ema = payload.get("ema")
        if not isinstance(ema, dict) or _integer(ema, "updates") != (
            completed - nonfinite
        ):
            raise RuntimeError("legacy CAP recovery EMA maturity is inconsistent")
        required = {"model", "optimizer", "generators"}
        if not required.issubset(payload):
            raise RuntimeError("legacy CAP recovery lacks resumable state")
        return

    planned = _integer(payload, "planned_updates")
    completed = _integer(payload, "completed_updates")
    updates = _integer(payload, "optimizer_updates")
    nonfinite = _integer(payload, "nonfinite_updates")
    clipped = _integer(payload, "clipped_updates")
    clipped_final = _integer(payload, "clipped_updates_final_window")
    final_window = _integer(payload, "final_window_updates")
    if planned <= 0 or not 0 <= completed <= planned:
        raise RuntimeError("CAP recovery completion lies outside its plan")
    if updates != completed:
        raise RuntimeError("CAP recovery optimizer/update counters disagree")
    if not 0 <= nonfinite <= completed:
        raise RuntimeError("CAP recovery nonfinite counter is impossible")
    if not 0 <= clipped <= completed:
        raise RuntimeError("CAP recovery clipping counter is impossible")
    if not 0 <= clipped_final <= final_window <= completed:
        raise RuntimeError("CAP recovery final-window counters are impossible")
    if clipped_final > clipped:
        raise RuntimeError("CAP recovery final-window clips exceed total clips")

    ema = payload.get("ema")
    if not isinstance(ema, dict):
        raise TypeError("CAP recovery lacks EMA state")
    ema_updates = _integer(ema, "updates")
    if ema_updates != completed - nonfinite:
        raise RuntimeError("CAP recovery EMA maturity disagrees with finite updates")

    checkpoints = payload.get("checkpoints")
    snapshots = payload.get("snapshots")
    if not isinstance(checkpoints, dict) or not isinstance(snapshots, list):
        raise TypeError("CAP recovery artifact ledger is malformed")
    for raw_step, kinds in checkpoints.items():
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as error:
            raise RuntimeError("CAP recovery checkpoint has an invalid step") from error
        if str(step) != str(raw_step) or not 0 < step <= completed:
            raise RuntimeError("CAP recovery checkpoint lies beyond completion")
        if not isinstance(kinds, dict) or not set(kinds).issubset({"raw", "ema"}):
            raise RuntimeError("CAP recovery checkpoint kinds are malformed")
    if any(
        isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= completed
        for step in snapshots
    ) or snapshots != sorted(set(snapshots)):
        raise RuntimeError("CAP recovery snapshot ledger is malformed")

    _validate_step_sequence(payload, "history", completed)
    _validate_step_sequence(payload, "health", completed)

    identity = payload.get("recovery_identity")
    if not strict and identity is None:
        return
    if not isinstance(identity, dict) or set(identity) != {"payload", "sha256"}:
        raise RuntimeError("CAP recovery has no canonical strict identity")
    canonical = json.dumps(
        identity["payload"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != identity["sha256"]:
        raise RuntimeError("CAP recovery strict identity hash is inconsistent")
    if not strict:
        # Legacy CAP1 files may predate newer redundant accounting fields.
        # Their existing identity is still checked, but only CAP2/replicated
        # callers opt into the complete fail-closed schema below.
        return
    profile = identity["payload"].get("profile_horizon_neutral")
    if not isinstance(profile, dict):
        raise TypeError("CAP recovery strict identity lacks its profile")
    train = profile.get("train")
    objective = profile.get("objective")
    gate = profile.get("gate")
    if not all(isinstance(value, dict) for value in (train, objective, gate)):
        raise RuntimeError("CAP recovery strict profile is malformed")
    micro_batch = int(train.get("micro_batch", 0))
    accumulation = int(train.get("accumulation_steps", 0))
    if micro_batch <= 0 or accumulation <= 0:
        raise RuntimeError("CAP recovery strict batch split is invalid")
    examples = _integer(payload, "examples_seen")
    expected_examples = completed * micro_batch * accumulation
    if examples != expected_examples:
        raise RuntimeError("CAP recovery example counter disagrees with its profile")
    if not math.isclose(
        float(ema.get("decay", math.nan)),
        float(train.get("ema_decay", math.nan)),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise RuntimeError("CAP recovery EMA decay disagrees with its profile")

    model_evaluations = _integer(payload, "model_forwards")
    forward_calls = _integer(payload, "objective_forward_calls")
    mode = objective.get("stopped_evaluation")
    if mode == "fp32_dense":
        evaluations_per_row, calls_per_batch = 4, 4
    elif mode == "dense":
        evaluations_per_row, calls_per_batch = 3, 3
    elif mode == "legacy_sparse":
        evaluations_per_row = calls_per_batch = None
        if not examples <= model_evaluations <= 3 * examples:
            raise RuntimeError(
                "CAP recovery sparse model-evaluation count is impossible"
            )
        batches = completed * accumulation
        if not batches <= forward_calls <= 3 * batches:
            raise RuntimeError("CAP recovery sparse forward-call count is impossible")
        if (model_evaluations - examples) % 2 or (forward_calls - batches) % 2:
            raise RuntimeError(
                "CAP recovery sparse evaluation counters have wrong parity"
            )
    else:
        raise RuntimeError("CAP recovery has an unknown stopped-evaluation mode")
    if evaluations_per_row is not None:
        if model_evaluations != evaluations_per_row * examples:
            raise RuntimeError("CAP recovery model-evaluation counter is inconsistent")
        if forward_calls != calls_per_batch * completed * accumulation:
            raise RuntimeError("CAP recovery forward-call counter is inconsistent")

    clip_window = int(gate.get("clip_window_updates", 0))
    if clip_window <= 0:
        raise RuntimeError("CAP recovery strict clip window is invalid")
    window_origin = _integer(payload, "final_window_origin")
    if not 0 <= window_origin <= planned:
        raise RuntimeError("CAP recovery final-window origin is impossible")
    expected_window = max(0, completed - window_origin)
    if final_window != expected_window:
        raise RuntimeError("CAP recovery final-window length disagrees with its plan")

    required = {"model", "optimizer", "generators", "objective_ledger"}
    if missing := sorted(required - set(payload)):
        raise RuntimeError(f"CAP recovery lacks resumable state {missing}")
    if not isinstance(payload["model"], dict) or not isinstance(
        payload["optimizer"], dict
    ):
        raise TypeError("CAP recovery model/optimizer state is malformed")
    generators = payload["generators"]
    expected_generators = {"data", "noise", "time", "flip"}
    if objective.get("diagonal_sampling") == "fixed_count_first_draw":
        expected_generators.add("diagonal")
    if not isinstance(generators, dict) or not expected_generators.issubset(generators):
        raise RuntimeError("CAP recovery RNG ledger is incomplete")

    for key in ("wall_seconds", "best_rank_ratio"):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise RuntimeError(f"CAP recovery scalar {key!r} is invalid")
    for key in ("peak_memory_bytes", "peak_memory_reserved_bytes"):
        if _integer(payload, key) < 0:
            raise RuntimeError(f"CAP recovery counter {key!r} is negative")

    authorization = payload.get("continuation_authorization")
    external = identity["payload"].get("external")
    cap2_bound = isinstance(external, dict) and external.get("status") == (
        "cap-emf2-run-identity"
    )
    if cap2_bound and planned < 300_000 and authorization is not None:
        raise RuntimeError(
            "a pre-promotion CAP2 recovery carries continuation authority"
        )
    if cap2_bound and planned == 300_000 and not isinstance(authorization, dict):
        raise RuntimeError("a promoted CAP2 recovery lacks continuation authority")
    if authorization is not None:
        if not isinstance(authorization, dict) or authorization.get("status") != (
            "cap-emf2-300k-recovery-authorization"
        ):
            raise RuntimeError("CAP recovery continuation authority is malformed")
        for key in (
            "promotion_sha256",
            "preflight_sha256",
            "result_150k_sha256",
            "checkpoint_150k_ema_sha256",
            "checkpoint_150k_raw_sha256",
            "readmission_sha256",
            "development_evaluation_sha256",
            "selection_sha256",
        ):
            value = authorization.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError("CAP recovery continuation hash is malformed")
            try:
                int(value, 16)
            except ValueError as error:
                raise RuntimeError(
                    "CAP recovery continuation hash is malformed"
                ) from error


def _fsync_directory(path: Path) -> None:
    """Durably publish renames where the host permits directory fsync."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:  # Windows commonly does not expose directory handles here.
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_save(payload: dict, path: Path) -> str:
    """Durably replace a rolling recovery and its SHA companion.

    The payload is serialized and read back before publication.  Publishing
    the data file before the sidecar makes every interrupted two-file update
    fail closed: readers see either the old matching pair, the new matching
    pair, or a mismatch they refuse to consume.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    sidecar = recovery_sidecar(path)
    side_descriptor, side_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".partial", dir=path.parent
    )
    side_temporary = Path(side_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb+") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("rb") as handle:
            check = _torch_load_handle(handle)
        if not isinstance(check, dict) or check.get("stage") != "cap-emf-1-recovery":
            raise RuntimeError("refusing to publish an invalid CAP recovery payload")
        with temporary.open("rb") as handle:
            digest = _hash_handle(handle)
        with os.fdopen(side_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            side_descriptor = -1
            handle.write(f"{digest}  {path.name}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.replace(side_temporary, sidecar)
        _fsync_directory(path.parent)
        return digest
    finally:
        if side_descriptor >= 0:
            os.close(side_descriptor)
        temporary.unlink(missing_ok=True)
        side_temporary.unlink(missing_ok=True)


def _health_component_batch(
    model: CAPPixelTransformer, noise: torch.Tensor, batch: int
) -> dict[str, torch.Tensor]:
    chunks: dict[str, list[torch.Tensor]] = {
        "base": [],
        "refiner_residual": [],
        "final": [],
    }
    with torch.no_grad():
        for start in range(0, len(noise), batch):
            block = one_step_components(model, noise[start : start + batch])
            for name, value in block.items():
                chunks[name].append(value.cpu())
    return {name: torch.cat(values) for name, values in chunks.items()}


def _fixed_exact_inference_corner_summary(
    model: CAPPixelTransformer,
    clean: torch.Tensor,
    noise: torch.Tensor,
    objective: CAPObjectiveConfig,
    batch: int,
) -> dict[str, float | int | dict[str, float]]:
    """Evaluate the sealed audit rows at the literal one-step condition.

    The training sampler's probability of visiting ``(t, r, h) = (1, 0, 1)``
    differs by arm, so sampled-row counts cannot be a common performance gate.
    This diagnostic instead evaluates every arm on the same fixed train-only
    clean/noise pairs.  It deliberately calls :func:`emf_loss`, rather than a
    simplified endpoint surrogate, so the live stopped-difference mode,
    denominator floor, delta and target construction are all exercised.

    The work is chunked, gradient-free and RNG-free.  The caller-visible model
    mode, parameters, buffers and existing gradients are therefore untouched.
    """
    if batch <= 0:
        raise ValueError("exact-corner diagnostic batch must be positive")
    if clean.shape != noise.shape or clean.ndim != 4 or not len(clean):
        raise ValueError("exact-corner clean/noise rows must be nonempty and matched")

    device = next(model.parameters()).device
    was_training = model.training
    raw_mse: list[torch.Tensor] = []
    target_rms: list[torch.Tensor] = []
    quotient_rms: list[torch.Tensor] = []
    coefficient: list[torch.Tensor] = []
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(clean), batch):
                stop = min(start + batch, len(clean))
                noise_block = noise[start:stop].to(device=device)
                clean_block = clean[start:stop].to(
                    device=device, dtype=noise_block.dtype
                )
                rows = stop - start
                triangle = TriangleSample(
                    t=torch.ones(rows, device=device, dtype=noise_block.dtype),
                    r=torch.zeros(rows, device=device, dtype=noise_block.dtype),
                    diagonal=torch.zeros(rows, device=device, dtype=torch.bool),
                )
                result = emf_loss(model, clean_block, noise_block, triangle, objective)
                raw_mse.append(result.per_sample_raw_mse.float().cpu())
                target_rms.append(result.per_sample_target_rms.float().cpu())
                quotient_rms.append(result.per_sample_quotient_rms.float().cpu())
                coefficient.append(result.coefficient.float().cpu())
    finally:
        model.train(was_training)

    raw = torch.cat(raw_mse)
    target = torch.cat(target_rms)
    quotient = torch.cat(quotient_rms)
    coeff = torch.cat(coefficient)
    finite = (
        torch.isfinite(raw)
        & torch.isfinite(target)
        & torch.isfinite(quotient)
        & torch.isfinite(coeff)
    )
    return {
        "count": len(raw),
        "mean_raw_mse": float(raw.mean()),
        "mean_target_rms": float(target.mean()),
        "mean_quotient_rms": float(quotient.mean()),
        "coefficient": {
            "minimum": float(coeff.min()),
            "mean": float(coeff.mean()),
            "maximum": float(coeff.max()),
        },
        "nonfinite_rows": int((~finite).sum()),
    }


def _parameter_gradient_norm(
    model: nn.Module,
    per_sample_objective: torch.Tensor,
    index: int,
    effective_batch: int,
) -> float:
    """Exact norm of one row's contribution to the effective-batch gradient."""
    if effective_batch <= 0:
        raise ValueError("effective batch must be positive")
    parameters = [value for value in model.parameters() if value.requires_grad]
    gradients = torch.autograd.grad(
        per_sample_objective[index] / effective_batch,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = torch.zeros((), device=per_sample_objective.device)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().float().square().sum()
    return float(squared.sqrt())


def _sample_parameter_gradient_categories(
    model: nn.Module,
    result,
    ledger: ObjectiveLedger,
    effective_batch: int,
    accumulation_steps: int,
) -> None:
    """Capture a sparse, stratified parameter-gradient audit.

    At most one example per declared category is retained between log events.
    The diagnostic is therefore inexpensive (a handful of extra backwards per
    500 updates) while measuring a genuine parameter contribution rather than
    inferring it from output residuals.
    """
    seen = {str(record["category"]) for record in ledger.parameter_gradient_samples}
    corner = (result.t > 0.95) & (result.interval > 0.90)
    masks = {
        "diagonal": result.diagonal,
        "ordinary_active": result.active & (result.coefficient <= 3) & ~corner,
        "coefficient_tail": result.coefficient > 7,
        "inference_corner": corner,
    }
    for category, mask in masks.items():
        if category in seen:
            continue
        indices = mask.nonzero(as_tuple=True)[0]
        if not indices.numel():
            continue
        index = int(indices[0])
        norm = _parameter_gradient_norm(
            model, result.per_sample_objective, index, effective_batch
        )
        ledger.add_parameter_gradient_sample(
            {
                "category": category,
                "norm": norm,
                "t": float(result.t[index]),
                "r": float(result.r[index]),
                "h": float(result.interval[index]),
                "coefficient": float(result.coefficient[index]),
                "target_rms": float(result.per_sample_target_rms[index]),
                "output_gradient_norm": float(
                    result.per_sample_output_gradient_norm[index] / accumulation_steps
                ),
            }
        )


def train_cap_unit(
    pool: torch.Tensor,
    profile: CAPProfile,
    device: torch.device | str,
    *,
    recovery_path: Path | None = None,
    checkpoint: Callable[[int, dict, dict], None] | None = None,
    snapshot: Callable[[int, dict], None] | None = None,
    checkpoint_health_observer: (
        Callable[[int, dict[str, torch.Tensor], dict[str, torch.Tensor]], dict | None]
        | None
    ) = None,
    recovery_saved: Callable[[int, Path], None] | None = None,
    progress: Callable[[str], None] | None = None,
    recovery_identity: dict | None = None,
    recovery_authorization: dict | None = None,
    unit_seed: int = 0,
    training_extension: TrainingExtension | None = None,
    stop_after_updates: int | None = None,
) -> TrainOutcome:
    profile.validate()
    train = profile.train
    device = torch.device(device)
    announce = progress or (lambda message: None)

    identity = _recovery_identity(profile, recovery_identity, unit_seed)
    execution_stop = (
        train.updates if stop_after_updates is None else int(stop_after_updates)
    )
    if isinstance(stop_after_updates, bool) or not 0 < execution_stop <= train.updates:
        raise ValueError("stop_after_updates must lie inside the planned horizon")
    if (
        execution_stop != train.updates
        and execution_stop not in train.checkpoint_updates
    ):
        raise ValueError("an intermediate execution stop must be a declared checkpoint")
    model = CAPPixelTransformer(
        profile.model, replicated_cap_seed("model-init", unit_seed)
    ).to(device)
    outcome = TrainOutcome(parameter_count=model.parameter_count())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train.learning_rate,
        betas=(train.beta1, train.beta2),
        weight_decay=train.weight_decay,
    )
    ema = EMAState(model, train.ema_decay)

    data_generator = torch.Generator().manual_seed(
        replicated_cap_seed("data-order", unit_seed)
    )
    noise_generator = torch.Generator().manual_seed(
        replicated_cap_seed("endpoint-noise", unit_seed)
    )
    time_generator = torch.Generator().manual_seed(
        replicated_cap_seed("time-triangle", unit_seed)
    )
    diagonal_generator = (
        torch.Generator().manual_seed(replicated_cap_seed("time-diagonal", unit_seed))
        if profile.objective.diagonal_sampling == "fixed_count_first_draw"
        else None
    )
    flip_generator = torch.Generator().manual_seed(
        replicated_cap_seed("horizontal-flip", unit_seed)
    )

    # Sealed train-only health noise: the same latents at every checkpoint, so
    # health movement is the model moving, not the sample moving.
    health_generator = torch.Generator().manual_seed(
        replicated_cap_seed("health-noise", unit_seed)
    )
    shape = (train.audit_samples, profile.model.channels) + (
        profile.model.image_size,
    ) * 2
    health_noise = torch.randn(shape, generator=health_generator).to(device)
    health_indices = torch.randperm(len(pool), generator=health_generator)
    if len(health_indices) < train.audit_samples:
        # Smoke tests use a deliberately tiny pool.  Repeat its one sealed
        # permutation so checkpoint diagnostics still exercise the declared
        # sample count; production pools are larger and retain the original
        # without-replacement behavior exactly.
        copies = math.ceil(train.audit_samples / len(health_indices))
        health_indices = health_indices.repeat(copies)
    health_target = pool[health_indices[: train.audit_samples]]

    objective_ledger = ObjectiveLedger()
    start_update = 0
    planned_window_start = max(0, train.updates - profile.gate.clip_window_updates)
    final_window_origin = planned_window_start
    active_recovery_authorization: dict | None = None
    strict_recovery = recovery_identity is not None or unit_seed != 0
    if recovery_path is not None:
        data_exists = recovery_path.exists()
        sidecar_exists = recovery_sidecar(recovery_path).exists()
        if sidecar_exists and not data_exists:
            raise RuntimeError("CAP recovery has an orphan SHA sidecar")
        if strict_recovery and data_exists != sidecar_exists:
            raise RuntimeError("strict CAP recovery requires a matching SHA sidecar")
    if recovery_path is not None and recovery_path.exists():
        payload, _ = load_recovery_payload(
            recovery_path,
            require_sidecar=strict_recovery,
            validate_counters=True,
        )
        recorded_identity = payload.get("recovery_identity")
        if recorded_identity is None:
            if recovery_identity is not None or unit_seed != 0:
                raise RuntimeError(
                    "recovery predates strict run identity and cannot resume a "
                    "bound CAP2/replicated run"
                )
            announce("legacy recovery has no strict run identity")
        elif recorded_identity != identity:
            raise RuntimeError(
                "recovery run identity differs from the requested configuration"
            )
        recorded_profile = payload.get("profile_name")
        if recorded_profile is not None and recorded_profile != profile.name:
            raise RuntimeError(
                f"recovery belongs to profile {recorded_profile!r}, not {profile.name!r}"
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        ema.load_recovery_state(payload["ema"], device)
        data_generator.set_state(payload["generators"]["data"])
        noise_generator.set_state(payload["generators"]["noise"])
        time_generator.set_state(payload["generators"]["time"])
        if diagonal_generator is not None:
            if "diagonal" not in payload["generators"]:
                raise RuntimeError("fixed-count recovery lacks diagonal RNG state")
            diagonal_generator.set_state(payload["generators"]["diagonal"])
        flip_generator.set_state(payload["generators"]["flip"])
        outcome.history = payload["history"]
        outcome.health = payload["health"]
        outcome.checkpoints = payload["checkpoints"]
        outcome.snapshots = payload.get("snapshots", [])
        outcome.wall_seconds = float(payload["wall_seconds"])
        outcome.peak_memory_bytes = int(payload.get("peak_memory_bytes", 0))
        outcome.peak_memory_reserved_bytes = int(
            payload.get("peak_memory_reserved_bytes", 0)
        )
        outcome.optimizer_updates = int(payload["optimizer_updates"])
        outcome.examples_seen = int(payload["examples_seen"])
        outcome.model_forwards = int(payload["model_forwards"])
        outcome.objective_forward_calls = int(payload.get("objective_forward_calls", 0))
        outcome.clipped_updates = int(payload["clipped_updates"])
        completed_updates = int(payload["completed_updates"])
        previous_horizon = int(payload.get("planned_updates", train.updates))
        promoted = previous_horizon != train.updates
        recorded_authorization = payload.get("continuation_authorization")
        cap2_bound = (
            isinstance(recovery_identity, dict)
            and recovery_identity.get("status") == "cap-emf2-run-identity"
        )
        if cap2_bound and train.updates == 300_000 and recovery_authorization is None:
            raise RuntimeError("a promoted CAP2 recovery requires its authorization")
        if promoted:
            if previous_horizon >= train.updates or completed_updates >= train.updates:
                raise RuntimeError(
                    "recovery horizon may only be extended to a larger unfinished plan"
                )
            if recorded_authorization is not None:
                raise RuntimeError(
                    "a pre-promotion recovery unexpectedly carries continuation authority"
                )
            active_recovery_authorization = recovery_authorization
            final_window_origin = max(completed_updates, planned_window_start)
            # The promoted run has a new final-window interval.  Historical
            # totals remain valid, but counters for the former horizon cannot
            # be reused as if they belonged to the new final window.
            outcome.clipped_updates_final_window = 0
            outcome.final_window_updates = 0
        else:
            if recovery_authorization is not None:
                if recorded_authorization != recovery_authorization:
                    raise RuntimeError(
                        "recovery is not bound to the requested continuation authority"
                    )
                active_recovery_authorization = recovery_authorization
            elif recorded_authorization is not None:
                # Non-CAP2 callers may resume an already-authorized recovery,
                # but may not silently erase its provenance on the next save.
                active_recovery_authorization = recorded_authorization
            final_window_origin = int(
                payload.get(
                    "final_window_origin",
                    max(0, previous_horizon - profile.gate.clip_window_updates),
                )
            )
            outcome.clipped_updates_final_window = int(
                payload.get("clipped_updates_final_window", 0)
            )
            outcome.final_window_updates = int(payload.get("final_window_updates", 0))
        outcome.nonfinite_updates = int(payload["nonfinite_updates"])
        outcome.best_rank_ratio = float(payload["best_rank_ratio"])
        outcome.auxiliary_history = payload.get("auxiliary_history", [])
        recorded_extension = payload.get("training_extension")
        if training_extension is None:
            if recorded_extension is not None:
                raise RuntimeError(
                    "recovery carries a training extension but none was requested"
                )
        else:
            if not isinstance(recorded_extension, dict):
                raise RuntimeError("extended recovery lacks its replay state")
            if recorded_extension.get("identity") != training_extension.identity():
                raise RuntimeError("training extension identity changed on resume")
            state = recorded_extension.get("state")
            if not isinstance(state, dict):
                raise RuntimeError("training extension recovery state is malformed")
            training_extension.load_state_dict(state)
        objective_ledger.load_state_dict(payload.get("objective_ledger"))
        start_update = completed_updates
        if start_update > execution_stop:
            raise RuntimeError(
                "CAP recovery is newer than the requested execution stop"
            )
        announce(f"resumed CAP-EMF-1 from update {start_update}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    window_start = planned_window_start
    for update in range(start_update, execution_stop):
        current_lr = learning_rate_at(update, train)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(train.accumulation_steps):
            order = torch.randint(
                0, len(pool), (train.micro_batch,), generator=data_generator
            )
            clean = pool[order].to(device)
            if train.horizontal_flip:
                flips = torch.rand(train.micro_batch, generator=flip_generator) < 0.5
                clean = flip_batch(clean, flips)
            noise = torch.randn(
                clean.shape, generator=noise_generator, dtype=clean.dtype
            ).to(device)
            triangle = sample_time_triangle(
                train.micro_batch,
                profile.objective,
                time_generator,
                device,
                diagonal_generator=diagonal_generator,
            )
            result = emf_loss(model, clean, noise, triangle, profile.objective)
            _sample_parameter_gradient_categories(
                model,
                result,
                objective_ledger,
                train.effective_batch,
                train.accumulation_steps,
            )
            (result.loss / train.accumulation_steps).backward()
            # Not a constant: the two stopped evaluations run on active rows
            # only, so this is batch + 2*active rather than 3*batch.
            outcome.model_forwards += result.model_evaluations
            outcome.objective_forward_calls += result.model_forward_calls
            objective_ledger.add(result, accumulation_steps=train.accumulation_steps)
            losses.append(
                (
                    float(result.loss.detach()),
                    float(result.raw_mse.detach()),
                    float(result.diagonal_raw_mse.detach()),
                    float(result.interior_raw_mse.detach()),
                )
            )
        if training_extension is not None:
            extension_record = training_extension.apply(update + 1, model)
            if extension_record is not None:
                try:
                    json.dumps(extension_record, sort_keys=True, allow_nan=False)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        "training extension returned a non-finite JSON record"
                    ) from error
                outcome.auxiliary_history.append(extension_record)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train.gradient_clip)
        pre_clip = float(norm)
        if not torch.isfinite(norm):
            outcome.nonfinite_updates += 1
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()
            ema.update(model)
        # The pre-clip norm was computed and thrown away, leaving only a
        # boolean.  That is the data H7 needs: CAP-EMF-1's recovery file lacked
        # the windowed clip counters, ``finalize.py`` substituted 0.0, and
        # "0.0 < 0.05" passed a gate whose real value was 15.3%.  Retaining the
        # norms makes the clip threshold a measurable quantity instead of an
        # assumed one.  Runtime-only: not scientific state, not persisted to
        # recovery, so a resumed run simply restarts the sample.
        if math.isfinite(pre_clip):
            outcome.gradient_norms.append(pre_clip)
        if pre_clip > train.gradient_clip:
            outcome.clipped_updates += 1
            if update >= window_start:
                outcome.clipped_updates_final_window += 1
        if update >= window_start:
            outcome.final_window_updates += 1
        outcome.optimizer_updates += 1
        outcome.examples_seen += train.effective_batch
        step = update + 1

        if step % train.log_every == 0 or step == execution_stop:
            mean = [sum(values) / len(values) for values in zip(*losses)]
            outcome.history.append(
                {
                    "step": step,
                    "loss": mean[0],
                    "raw_mse": mean[1],
                    "diagonal_raw_mse": mean[2],
                    "interior_raw_mse": mean[3],
                    "gradient_norm_before_clip": pre_clip,
                    "learning_rate": current_lr,
                    "examples_seen": outcome.examples_seen,
                    "ema_updates": ema.updates,
                    # Covers every row since the previous log, not one
                    # arbitrarily selected microbatch.
                    "objective_ledger": objective_ledger.summary(),
                    "wall_seconds": time.time() - started + outcome.wall_seconds,
                }
            )

        if step % train.health_every == 0 or step in train.checkpoint_updates:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            health_started = time.time()
            checkpoint_health = step in train.checkpoint_updates
            samples = (
                train.audit_samples if checkpoint_health else (train.health_samples)
            )
            model.eval()
            raw_components = _health_component_batch(
                model, health_noise[:samples], train.micro_batch * 2
            )
            raw_exact_corner = (
                _fixed_exact_inference_corner_summary(
                    model,
                    health_target[:samples],
                    health_noise[:samples],
                    profile.objective,
                    train.micro_batch * 2,
                )
                if checkpoint_health
                else None
            )
            model.train()
            raw_report = component_health(raw_components, health_target[:samples])
            # Preserve the original flat final-output fields for compatibility
            # while retaining the full source-attribution decomposition.
            record = dict(raw_report["final"])
            record["components"] = raw_report
            if step in train.checkpoint_updates:
                ema_model = deepcopy(model).to(device)
                ema_model.load_state_dict(ema.state_dict())
                ema_model.eval()
                ema_components = _health_component_batch(
                    ema_model, health_noise[:samples], train.micro_batch * 2
                )
                ema_exact_corner = _fixed_exact_inference_corner_summary(
                    ema_model,
                    health_target[:samples],
                    health_noise[:samples],
                    profile.objective,
                    train.micro_batch * 2,
                )
                ema_report = component_health(ema_components, health_target[:samples])
                record["ema"] = ema_report["final"]
                record["ema_components"] = ema_report
                record["fixed_exact_inference_corner"] = {
                    "condition": {"t": 1.0, "r": 0.0, "h": 1.0},
                    "sealed_train_only": True,
                    "sample_count": samples,
                    "objective_numerics": {
                        "stopped_evaluation": profile.objective.stopped_evaluation,
                        "emf_delta": profile.objective.emf_delta,
                        "emf_denominator_floor": (
                            profile.objective.emf_denominator_floor
                        ),
                    },
                    "raw": raw_exact_corner,
                    "ema": ema_exact_corner,
                }
                if checkpoint_health_observer is not None:
                    observation = checkpoint_health_observer(
                        step, raw_components, ema_components
                    )
                    if observation is not None:
                        if not isinstance(observation, dict):
                            raise TypeError(
                                "checkpoint health observer must return a dict or None"
                            )
                        try:
                            json.dumps(observation, sort_keys=True, allow_nan=False)
                        except (TypeError, ValueError) as error:
                            raise TypeError(
                                "checkpoint health observation must be finite JSON"
                            ) from error
                        record["checkpoint_health_observation"] = deepcopy(observation)
                del ema_model
            record["step"] = step
            record["ema_updates"] = ema.updates
            record["ema_initialization_weight"] = ema.initialization_weight()
            record["ema_mature"] = step >= train.ema_mature_at()
            outcome.health.append(record)
            outcome.best_rank_ratio = max(
                outcome.best_rank_ratio, record["effective_rank_ratio"]
            )
            announce(
                f"step {step}: moment {record['second_moment_ratio']:.3f} "
                f"var {record['centered_variance_ratio']:.3f} "
                f"rank {record['effective_rank_ratio']:.3f} "
                f"HH {record['haar_HH_ratio']:.3f}"
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            health_seconds = time.time() - health_started
            if checkpoint_health:
                outcome.checkpoint_health_seconds += health_seconds
                outcome.checkpoint_health_events += 1
            else:
                outcome.ordinary_health_seconds += health_seconds
                outcome.ordinary_health_events += 1

        if step in train.checkpoint_updates and checkpoint is not None:
            record = checkpoint(step, model.state_dict(), ema.state_dict())
            if record is not None:
                # Held on the outcome so the recovery file carries it: a caller
                # keeping its own dict would lose every pre-resume checkpoint.
                outcome.checkpoints[str(step)] = record

        if snapshot is not None and step % train.snapshot_every == 0:
            # Raw weights for post-hoc EMA synthesis. Secondary by
            # construction: the declared 0.9999 EMA remains the primary result,
            # so this cannot become checkpoint selection on a metric.
            snapshot(step, model.state_dict())
            outcome.snapshots.append(step)

        if recovery_path is not None and (
            step % train.recovery_every == 0 or step == execution_stop
        ):
            recovery_started = time.time()
            _atomic_save(
                {
                    "stage": "cap-emf-1-recovery",
                    "profile_name": profile.name,
                    "recovery_identity": identity,
                    "continuation_authorization": active_recovery_authorization,
                    "planned_updates": train.updates,
                    "completed_updates": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema": ema.recovery_state(),
                    "generators": {
                        "data": data_generator.get_state(),
                        "noise": noise_generator.get_state(),
                        "time": time_generator.get_state(),
                        **(
                            {"diagonal": diagonal_generator.get_state()}
                            if diagonal_generator is not None
                            else {}
                        ),
                        "flip": flip_generator.get_state(),
                    },
                    "history": outcome.history,
                    "health": outcome.health,
                    "checkpoints": outcome.checkpoints,
                    "snapshots": outcome.snapshots,
                    "wall_seconds": time.time() - started + outcome.wall_seconds,
                    "peak_memory_bytes": (
                        max(
                            outcome.peak_memory_bytes,
                            int(torch.cuda.max_memory_allocated(device)),
                        )
                        if device.type == "cuda"
                        else outcome.peak_memory_bytes
                    ),
                    "peak_memory_reserved_bytes": (
                        max(
                            outcome.peak_memory_reserved_bytes,
                            int(torch.cuda.max_memory_reserved(device)),
                        )
                        if device.type == "cuda"
                        else outcome.peak_memory_reserved_bytes
                    ),
                    "optimizer_updates": outcome.optimizer_updates,
                    "examples_seen": outcome.examples_seen,
                    "model_forwards": outcome.model_forwards,
                    "objective_forward_calls": outcome.objective_forward_calls,
                    "clipped_updates": outcome.clipped_updates,
                    "clipped_updates_final_window": (
                        outcome.clipped_updates_final_window
                    ),
                    "final_window_updates": outcome.final_window_updates,
                    "final_window_origin": final_window_origin,
                    "nonfinite_updates": outcome.nonfinite_updates,
                    "best_rank_ratio": outcome.best_rank_ratio,
                    "objective_ledger": objective_ledger.state_dict(),
                    "auxiliary_history": outcome.auxiliary_history,
                    "training_extension": (
                        {
                            "identity": training_extension.identity(),
                            "state": training_extension.state_dict(),
                        }
                        if training_extension is not None
                        else None
                    ),
                },
                recovery_path,
            )
            if recovery_saved is not None:
                recovery_saved(step, recovery_path)
            outcome.recovery_io_seconds += time.time() - recovery_started

    outcome.wall_seconds += time.time() - started
    if device.type == "cuda":
        outcome.peak_memory_bytes = max(
            outcome.peak_memory_bytes,
            int(torch.cuda.max_memory_allocated(device)),
        )
        outcome.peak_memory_reserved_bytes = max(
            outcome.peak_memory_reserved_bytes,
            int(torch.cuda.max_memory_reserved(device)),
        )

    # One final, read-only architectural check.  Keep it outside the recovery
    # state because it is an audit of the executable inference path rather
    # than scientific training state.  The sealed health noise makes this
    # deterministic and avoids consuming any RNG stream.
    calls = {"count": 0}
    handle = model.register_forward_pre_hook(
        lambda module, args: calls.__setitem__("count", calls["count"] + 1)
    )
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            one_step_sample(model, health_noise[:1])
    finally:
        handle.remove()
        model.train(was_training)
    outcome.inference_forward_calls = calls["count"]
    return outcome


def clip_fraction(outcome: TrainOutcome) -> float:
    if not outcome.final_window_updates:
        return 0.0
    return outcome.clipped_updates_final_window / outcome.final_window_updates


def history_to_json(outcome: TrainOutcome) -> str:
    return json.dumps({"history": outcome.history, "health": outcome.health})
