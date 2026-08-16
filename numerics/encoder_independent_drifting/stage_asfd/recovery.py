"""Create an immutable ASFD continuation fork from the exact foundation state."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from ..stage_cap.config import CAPProfile
from ..stage_cap.training import (
    TrainingExtension,
    _atomic_save,
    _recovery_identity,
    load_recovery_payload,
    validate_recovery_counters,
)


def _exactly_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _exactly_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _exactly_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def fork_foundation_recovery(
    foundation_recovery: Path,
    output: Path,
    *,
    profile: CAPProfile,
    external_identity: dict,
    extension: TrainingExtension,
    expected_sha256: str,
    unit_seed: int = 0,
    foundation_step: int = 750_000,
) -> str:
    """Copy all scientific state while changing only continuation identity.

    The source recovery is opened read-only.  The fork retains online weights,
    Adam moments, EMA, primary streams, counters and ledgers exactly; the new
    extension begins from its own freshly declared role-separated state.
    """
    if output.exists() or output.with_suffix(output.suffix + ".sha256").exists():
        raise RuntimeError("refusing to overwrite an ASFD continuation recovery")
    payload, digest = load_recovery_payload(
        foundation_recovery,
        require_sidecar=True,
        validate_counters=True,
    )
    if digest != expected_sha256:
        raise RuntimeError("foundation recovery hash differs from its gate")
    validate_recovery_counters(payload, strict=True)
    # The foundation must be *complete* -- planned and completed agree -- which
    # is the property that matters. The step itself was written as a literal
    # 750_000, which pinned the fork to one horizon and made forking any other
    # finished foundation impossible.
    if (
        int(payload.get("planned_updates", -1)) != int(foundation_step)
        or int(payload.get("completed_updates", -1)) != int(foundation_step)
    ):
        raise RuntimeError(
            f"ASFD may fork only a foundation completed at step "
            f"{int(foundation_step)}; this recovery reports planned="
            f"{payload.get('planned_updates')} completed="
            f"{payload.get('completed_updates')}"
        )
    result = deepcopy(payload)
    result["profile_name"] = profile.name
    result["recovery_identity"] = _recovery_identity(
        profile, external_identity, unit_seed
    )
    result["continuation_authorization"] = None
    result["auxiliary_history"] = []
    result["training_extension"] = {
        "identity": extension.identity(),
        "state": extension.state_dict(),
    }
    # Keep planned_updates=750k deliberately.  train_cap_unit recognizes an
    # honest promotion only when the source horizon is smaller than the new
    # 800k plan and resets final-window counters at that boundary.
    _atomic_save(result, output)
    fork, fork_digest = load_recovery_payload(
        output, require_sidecar=True, validate_counters=True
    )
    preserved = (
        "model",
        "optimizer",
        "ema",
        "generators",
        "history",
        "health",
        "checkpoints",
        "snapshots",
        "objective_ledger",
        "optimizer_updates",
        "examples_seen",
        "model_forwards",
        "objective_forward_calls",
        "completed_updates",
    )
    changed = [
        name for name in preserved if not _exactly_equal(fork[name], payload[name])
    ]
    if changed:
        raise RuntimeError(f"ASFD recovery fork changed foundation state: {changed}")
    return fork_digest
