"""Matched fresh-optimizer continuations from immutable B0 EMA states."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..b1 import load_b0_checkpoint_model
from ..f3b import (
    EMAState,
    F3BModelConfig,
    F3BTrainConfig,
    TimeConditionedUNet,
    bridge_streams,
    independent_bridge_batch,
)
from ..stage_b2.core import (
    B2Config,
    b2_streams,
)
from ..stage_b2.core import (
    correction_term as laplace_correction_term,
)
from ..stage_b2.core import (
    should_apply as laplace_should_apply,
)
from .core import SinkhornConfig
from .training import (
    should_apply as sinkhorn_should_apply,
)
from .training import (
    sinkhorn_correction_term,
    sinkhorn_streams,
)

CONTINUATION_PHASE = "sinkhorn-s1-continuation-v2"
CONTINUATION_ARMS = ("control", "laplace", "sinkhorn")


@dataclass(frozen=True)
class ContinuationConfig:
    steps: int = 5_000
    log_every: int = 100

    def validate(self) -> None:
        if self.steps <= 0 or self.log_every <= 0:
            raise ValueError("continuation steps and log cadence must be positive")


@dataclass
class ContinuationResult:
    arm: str
    model: TimeConditionedUNet
    ema: EMAState
    history: list[dict]
    wall_seconds: float
    peak_memory_bytes: int | None
    optimizer_updates: int
    examples_seen: int
    correction_events: int
    correction_model_forwards: int
    start_state_sha256: str
    first_flow_batch_sha256: str
    correction_summary: dict


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def batch_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def continuation_flow_seed_manifest(unit: int) -> dict[str, int]:
    from ..f3b import f3b_seed

    return {
        role: f3b_seed(CONTINUATION_PHASE, unit, role)
        for role in (
            "data-order",
            "endpoint-noise",
            "bridge-time",
            "augmentation",
        )
    }


def _summarize_sinkhorn(events: list[dict]) -> dict:
    if not events:
        return {"events": 0, "cap_hits": 0}
    plans = [event[role] for event in events for role in ("cross", "self")]
    return {
        "events": len(events),
        "plans": len(plans),
        "cap_hits": sum(int(plan["iteration_cap_hit"]) for plan in plans),
        "maximum_relative_error": max(
            float(plan["maximum_relative_error"]) for plan in plans
        ),
        "maximum_iterations": max(int(plan["iterations"]) for plan in plans),
        "cross_entropy_mean": float(
            np.mean([event["cross"]["conditional_entropy_mean"] for event in events])
        ),
        "self_entropy_mean": float(
            np.mean([event["self"]["conditional_entropy_mean"] for event in events])
        ),
        "update_to_sample_rms_mean": float(
            np.mean([event["update_to_sample_rms"] for event in events])
        ),
    }


def _summarize_laplace(events: list[dict]) -> dict:
    if not events:
        return {"events": 0}
    return {
        "events": len(events),
        "energy_mean": float(np.mean([event["drift_energy"] for event in events])),
        "positive_ess_fraction_median_mean": float(
            np.mean([event["positive"]["ess_fraction_median"] for event in events])
        ),
        "negative_ess_fraction_median_mean": float(
            np.mean([event["negative"]["ess_fraction_median"] for event in events])
        ),
        "positive_row_sum_error_maximum": max(
            float(event["positive"]["row_sum_error_maximum"]) for event in events
        ),
        "negative_row_sum_error_maximum": max(
            float(event["negative"]["row_sum_error_maximum"]) for event in events
        ),
    }


def train_continuation_arm(
    *,
    arm: str,
    pool: torch.Tensor,
    checkpoint_record: dict,
    frozen_profile: dict,
    model_config: F3BModelConfig,
    base_train_config: F3BTrainConfig,
    continuation_config: ContinuationConfig,
    unit: int,
    device: torch.device | str,
    sinkhorn_cost_scale: float,
    sinkhorn_lambda: float,
    sinkhorn_config: SinkhornConfig,
    laplace_tau: float,
    laplace_lambda: float,
    laplace_config: B2Config,
) -> ContinuationResult:
    """Continue one arm from the exact same B0 EMA state and fresh optimizer."""
    if arm not in CONTINUATION_ARMS:
        raise ValueError(f"unknown continuation arm {arm}")
    continuation_config.validate()
    model_config.validate()
    base_train_config.validate()
    sinkhorn_config.validate()
    laplace_config.validate()
    if not math.isfinite(sinkhorn_lambda) or sinkhorn_lambda <= 0:
        raise ValueError("invalid frozen Sinkhorn continuation weight")
    if not math.isfinite(laplace_lambda) or laplace_lambda <= 0:
        raise ValueError("invalid frozen Laplace continuation weight")
    if not math.isfinite(laplace_tau) or laplace_tau <= 0:
        raise ValueError("invalid frozen Laplace bandwidth")
    device = torch.device(device)
    model = load_b0_checkpoint_model(
        checkpoint_record,
        frozen_profile,
        model_config,
        unit,
        device,
    )
    model.train()
    starting_sha = state_sha256(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_train_config.learning_rate,
        betas=(base_train_config.beta1, base_train_config.beta2),
        weight_decay=base_train_config.weight_decay,
    )
    ema = EMAState(model, base_train_config.ema_decay)
    flow_streams = bridge_streams(CONTINUATION_PHASE, unit)
    sinkhorn_roles = sinkhorn_streams(CONTINUATION_PHASE, unit)
    laplace_roles = b2_streams(CONTINUATION_PHASE, unit)
    history: list[dict] = []
    correction_health: list[dict] = []
    correction_events = 0
    first_batch_sha = ""
    started = time.time()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, continuation_config.steps + 1):
        mixed, target, _, _, time_value = independent_bridge_batch(
            pool,
            base_train_config.batch,
            flow_streams,
            device,
            base_train_config.horizontal_flip,
        )
        if step == 1:
            first_batch_sha = batch_sha256(mixed, target, time_value)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(mixed, time_value)
        flow_loss = F.mse_loss(prediction, target)
        correction = None
        health: dict = {}
        event_weight = 0.0
        if arm == "sinkhorn" and sinkhorn_should_apply(step, sinkhorn_config):
            correction_events += 1
            correction, health = sinkhorn_correction_term(
                model,
                pool,
                model_config,
                sinkhorn_roles,
                device,
                sinkhorn_cost_scale,
                sinkhorn_config,
                horizontal_flip=base_train_config.horizontal_flip,
            )
            event_weight = sinkhorn_lambda
            correction_health.append(health)
        elif arm == "laplace" and laplace_should_apply(step, laplace_config):
            correction_events += 1
            correction, health = laplace_correction_term(
                model,
                pool,
                model_config,
                laplace_roles,
                device,
                laplace_tau,
                laplace_config,
                horizontal_flip=base_train_config.horizontal_flip,
            )
            event_weight = laplace_lambda
            correction_health.append(health)
        loss = flow_loss
        if correction is not None:
            loss = loss + event_weight * correction.to(flow_loss.dtype)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"non-finite {arm} continuation loss at step {step}"
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), base_train_config.gradient_clip
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise FloatingPointError(
                f"non-finite {arm} continuation gradient at step {step}"
            )
        optimizer.step()
        ema.update(model)
        if (
            step == 1
            or step % continuation_config.log_every == 0
            or step == continuation_config.steps
        ):
            record = {
                "step": step,
                "arm": arm,
                "loss": float(loss.detach()),
                "flow_loss": float(flow_loss.detach()),
                "correction_loss": (
                    None if correction is None else float(correction.detach())
                ),
                "correction_event_count": correction_events,
                "gradient_norm_before_clip": float(gradient_norm),
                "examples_seen": step * base_train_config.batch,
                "wall_seconds": time.time() - started,
            }
            if health:
                record["correction_health"] = health
            history.append(record)

    peak = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    if arm == "sinkhorn":
        correction_summary = _summarize_sinkhorn(correction_health)
        forwards = correction_events * 2 * sinkhorn_config.correction_nfe
    elif arm == "laplace":
        correction_summary = _summarize_laplace(correction_health)
        forwards = correction_events * laplace_config.correction_nfe
    else:
        correction_summary = {"events": 0}
        forwards = 0
    return ContinuationResult(
        arm=arm,
        model=model,
        ema=ema,
        history=history,
        wall_seconds=time.time() - started,
        peak_memory_bytes=peak,
        optimizer_updates=continuation_config.steps,
        examples_seen=continuation_config.steps * base_train_config.batch,
        correction_events=correction_events,
        correction_model_forwards=forwards,
        start_state_sha256=starting_sha,
        first_flow_batch_sha256=first_batch_sha,
        correction_summary=correction_summary,
    )
