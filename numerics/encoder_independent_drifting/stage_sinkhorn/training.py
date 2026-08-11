"""B0-compatible training integration for the balanced Sinkhorn correction."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import MASTER_SEED, derive_seed
from ..f3b import (
    BridgeTrainResult,
    EMAState,
    F3BModelConfig,
    F3BTrainConfig,
    TimeConditionedUNet,
    bridge_streams,
    euler_integrate,
    f3b_seed,
    independent_bridge_batch,
)
from .core import SinkhornConfig, sinkhorn_drifted_target_loss

SINKHORN_SEED_OFFSET = 149_000


def sinkhorn_seed(phase: str, unit: int | str, role: str) -> int:
    return derive_seed(
        MASTER_SEED + SINKHORN_SEED_OFFSET,
        "balanced-sinkhorn",
        phase,
        unit,
        role,
    )


@dataclass
class SinkhornStreams:
    real_data: np.random.Generator
    real_augmentation: torch.Generator
    primary_prior: torch.Generator
    self_prior: torch.Generator


def sinkhorn_streams(phase: str, unit: int) -> SinkhornStreams:
    def torch_stream(role: str) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(
            sinkhorn_seed(phase, unit, role) % (2**63 - 1)
        )

    return SinkhornStreams(
        real_data=np.random.default_rng(sinkhorn_seed(phase, unit, "real-data")),
        real_augmentation=torch_stream("real-augmentation"),
        primary_prior=torch_stream("primary-prior"),
        self_prior=torch_stream("self-prior"),
    )


def paired_seed_manifest(phase: str, unit: int) -> dict[str, int]:
    """Record shared B0 streams and intervention-only streams separately."""

    result = {
        f"shared_{role}": f3b_seed(phase, unit, role)
        for role in (
            "model-init",
            "data-order",
            "endpoint-noise",
            "bridge-time",
            "augmentation",
        )
    }
    result.update(
        {
            f"sinkhorn_{role}": sinkhorn_seed(phase, unit, role)
            for role in (
                "real-data",
                "real-augmentation",
                "primary-prior",
                "self-prior",
            )
        }
    )
    return result


def _augmented_real_batch(
    pool: Tensor,
    streams: SinkhornStreams,
    batch: int,
    horizontal_flip: bool,
) -> Tensor:
    indices = streams.real_data.integers(0, len(pool), size=batch)
    result = pool[torch.as_tensor(indices)].clone()
    if horizontal_flip:
        flip = torch.rand(batch, generator=streams.real_augmentation) < 0.5
        result[flip] = torch.flip(result[flip], dims=(-1,))
    return result


def _initial_noise(
    batch: int,
    model_config: F3BModelConfig,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    return torch.randn(
        batch,
        model_config.channels,
        model_config.image_size,
        model_config.image_size,
        generator=generator,
        dtype=dtype,
    ).to(device)


def sinkhorn_correction_term(
    model: nn.Module,
    target_pool: Tensor,
    model_config: F3BModelConfig,
    streams: SinkhornStreams,
    device: torch.device | str,
    cost_scale: float,
    config: SinkhornConfig,
    *,
    horizontal_flip: bool = True,
) -> tuple[Tensor, dict]:
    """Construct one differentiable two-batch correction event."""

    config.validate()
    model_config.validate()
    device = torch.device(device)
    real = _augmented_real_batch(
        target_pool,
        streams,
        config.real_batch,
        horizontal_flip,
    ).to(device)
    dtype = real.dtype
    primary_initial = _initial_noise(
        config.primary_batch,
        model_config,
        streams.primary_prior,
        device,
        dtype,
    )
    self_initial = _initial_noise(
        config.self_batch,
        model_config,
        streams.self_prior,
        device,
        dtype,
    )
    if torch.equal(primary_initial, self_initial):
        raise RuntimeError("independent Sinkhorn priors unexpectedly coincide")

    # The self support is the current model law but is never part of the
    # parameter gradient. The primary trajectory is constructed separately and
    # retains its full Euler graph.
    with torch.no_grad():
        self_support = euler_integrate(model, self_initial, config.correction_nfe)
    primary = euler_integrate(model, primary_initial, config.correction_nfe)
    value, health = sinkhorn_drifted_target_loss(
        primary,
        real.detach(),
        self_support.detach(),
        cost_scale,
        config,
    )
    health.update(
        {
            "correction_nfe": config.correction_nfe,
            "model_forwards": 2 * config.correction_nfe,
            "primary_prior_and_self_prior_distinct": True,
            "geometry": "raw_identity_pixels",
            "cost": "normalized_half_squared_l2",
            "diagonal_mask": False,
        }
    )
    return value, health


def should_apply(step: int, config: SinkhornConfig) -> bool:
    if step <= 0:
        raise ValueError("training steps are one-indexed")
    return step % config.correction_every == 0


def calibrated_event_lambda(
    flow_gradient_norm: float,
    correction_gradient_norm: float,
    config: SinkhornConfig,
) -> float:
    config.validate()
    if (
        not math.isfinite(flow_gradient_norm)
        or not math.isfinite(correction_gradient_norm)
        or flow_gradient_norm <= 0
        or correction_gradient_norm <= 0
    ):
        raise ValueError("gradient calibration requires positive finite norms")
    return config.event_gradient_ratio * flow_gradient_norm / correction_gradient_norm


def minimax_log_event_lambda(unit_lambdas: list[float]) -> float:
    """Choose the common weight with the smallest worst multiplicative error.

    Each entry is the event weight that would hit the requested gradient ratio
    exactly on one outcome-blind calibration unit.  In log coordinates the
    Chebyshev center of these points is the midpoint of their extrema, hence
    the geometric mean below.  This is preferable to a median when one common
    weight must remain robust across every frozen unit.
    """
    if not unit_lambdas or any(
        not math.isfinite(value) or value <= 0 for value in unit_lambdas
    ):
        raise ValueError("event weights must be a nonempty positive finite list")
    return math.sqrt(min(unit_lambdas) * max(unit_lambdas))


def parameter_gradient_norm(loss: Tensor, model: nn.Module) -> float:
    gradients = torch.autograd.grad(
        loss,
        tuple(model.parameters()),
        retain_graph=True,
        allow_unused=True,
    )
    squared = torch.zeros((), dtype=torch.float64, device=loss.device)
    for gradient in gradients:
        if gradient is not None:
            squared = squared + gradient.detach().double().square().sum()
    return float(squared.sqrt())


def parameter_gradient_geometry(
    first_loss: Tensor,
    second_loss: Tensor,
    model: nn.Module,
) -> dict[str, float]:
    """Return norms and cosine for two parameter gradients without mutation."""
    parameters = tuple(model.parameters())
    first = torch.autograd.grad(
        first_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    second = torch.autograd.grad(
        second_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    first_squared = torch.zeros((), dtype=torch.float64, device=first_loss.device)
    second_squared = torch.zeros((), dtype=torch.float64, device=first_loss.device)
    dot = torch.zeros((), dtype=torch.float64, device=first_loss.device)
    for left, right in zip(first, second, strict=True):
        if left is not None:
            first_squared = first_squared + left.detach().double().square().sum()
        if right is not None:
            second_squared = second_squared + right.detach().double().square().sum()
        if left is not None and right is not None:
            dot = dot + (left.detach().double() * right.detach().double()).sum()
    first_norm = first_squared.sqrt()
    second_norm = second_squared.sqrt()
    if float(first_norm) <= 0 or float(second_norm) <= 0:
        raise ValueError("gradient geometry requires two nonzero gradients")
    return {
        "first_norm": float(first_norm),
        "second_norm": float(second_norm),
        "dot": float(dot),
        "cosine": float(dot / (first_norm * second_norm)),
    }


def _solver_summary(events: list[dict]) -> dict:
    if not events:
        return {"events": 0, "cap_hits": 0}
    cross_errors = [event["cross"]["maximum_relative_error"] for event in events]
    self_errors = [event["self"]["maximum_relative_error"] for event in events]
    cross_iterations = [event["cross"]["iterations"] for event in events]
    self_iterations = [event["self"]["iterations"] for event in events]
    cap_hits = sum(
        int(event[role]["iteration_cap_hit"])
        for event in events
        for role in ("cross", "self")
    )
    return {
        "events": len(events),
        "plans": 2 * len(events),
        "cap_hits": cap_hits,
        "maximum_relative_error": max(cross_errors + self_errors),
        "cross_iterations_mean": float(np.mean(cross_iterations)),
        "self_iterations_mean": float(np.mean(self_iterations)),
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


@dataclass
class SinkhornTrainResult(BridgeTrainResult):
    correction_events: int
    correction_model_forwards: int
    lambda_event: float
    cost_scale: float
    solver_summary: dict


def train_sinkhorn_bridge(
    pool: Tensor,
    model_config: F3BModelConfig,
    train_config: F3BTrainConfig,
    phase: str,
    unit: int,
    device: torch.device | str,
    *,
    cost_scale: float,
    lambda_event: float,
    config: SinkhornConfig,
    checkpoint: Callable[[int, TimeConditionedUNet, dict], None] | None = None,
) -> SinkhornTrainResult:
    """Train the paired B0 + identity-Sinkhorn arm."""

    model_config.validate()
    train_config.validate()
    config.validate()
    if not math.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("invalid frozen Sinkhorn cost scale")
    if not math.isfinite(lambda_event) or lambda_event <= 0:
        raise ValueError("invalid frozen Sinkhorn event weight")
    device = torch.device(device)
    model = TimeConditionedUNet(model_config, f3b_seed(phase, unit, "model-init")).to(
        device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    ema = EMAState(model, train_config.ema_decay)
    flow_streams = bridge_streams(phase, unit)
    correction_streams = sinkhorn_streams(phase, unit)
    history: list[dict] = []
    solver_events: list[dict] = []
    correction_events = 0
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, train_config.steps + 1):
        mixed, target, _, _, time_value = independent_bridge_batch(
            pool,
            train_config.batch,
            flow_streams,
            device,
            train_config.horizontal_flip,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(mixed, time_value)
        flow_value = F.mse_loss(prediction, target)
        if not bool(torch.isfinite(flow_value)):
            raise FloatingPointError(f"non-finite flow loss at step {step}")

        correction: Tensor | None = None
        health: dict = {}
        if should_apply(step, config):
            correction_events += 1
            correction, health = sinkhorn_correction_term(
                model,
                pool,
                model_config,
                correction_streams,
                device,
                cost_scale,
                config,
                horizontal_flip=train_config.horizontal_flip,
            )
            solver_events.append(health)
        loss = flow_value
        if correction is not None:
            loss = loss + lambda_event * correction.to(flow_value.dtype)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite Sinkhorn objective at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.gradient_clip
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        ema.update(model)

        record = {
            "step": step,
            "loss": float(loss.detach()),
            "flow_loss": float(flow_value.detach()),
            "sinkhorn_loss": None if correction is None else float(correction.detach()),
            "correction_event_count": correction_events,
            "gradient_norm_before_clip": float(gradient_norm),
            "examples_seen": step * train_config.batch,
            "wall_seconds": time.time() - started,
        }
        if health:
            record["sinkhorn_health"] = health
        if (
            step == 1
            or step % train_config.log_every == 0
            or step in train_config.checkpoint_steps
        ):
            history.append(record)
        if checkpoint is not None and step in train_config.checkpoint_steps:
            with ema.average_parameters(model):
                model.eval()
                checkpoint(step, model, record)

    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return SinkhornTrainResult(
        model=model,
        ema=ema,
        history=history,
        wall_seconds=time.time() - started,
        peak_memory_bytes=peak,
        examples_seen=train_config.steps * train_config.batch,
        optimizer_updates=train_config.steps,
        correction_events=correction_events,
        correction_model_forwards=correction_events * 2 * config.correction_nfe,
        lambda_event=lambda_event,
        cost_scale=cost_scale,
        solver_summary=_solver_summary(solver_events),
    )
