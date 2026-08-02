"""True full-dose Stage-B2.5 factorial training.

The primary combined cell is a genuine factorial cell: at every scheduled
correction step it receives the complete frozen B1 anchor and the complete
frozen B2 normalized-Laplace correction.  The losses are backpropagated
sequentially before one shared clip/optimizer step.  This computes the same
sum of component gradients while avoiding simultaneous retention of both
trajectory graphs on the 6-GiB development GPU.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..b1 import (
    B1Config,
    anchor_target_batch,
    anchor_term,
    b1_streams,
    build_training_bank,
    should_refresh,
)
from ..f3b import (
    BridgeTrainResult,
    EMAState,
    F3BModelConfig,
    F3BTrainConfig,
    TimeConditionedUNet,
    bridge_streams,
    f3b_seed,
    independent_bridge_batch,
)
from ..spectral_anchor import refresh_bank
from ..stage_b2.core import B2Config, b2_seed, b2_streams, correction_term

B25_PHASE = "b25-development"
B25_UNITS = (500, 501, 502)
B25_ARMS = ("B0", "B1", "B2", "B1B2")
B25Arm = Literal["B0", "B1", "B2", "B1B2"]


@dataclass(frozen=True)
class B25Config:
    """Stage-local decisions, excluding the hash-bound B1/B2 constants."""

    units: tuple[int, ...] = B25_UNITS
    arms: tuple[str, ...] = B25_ARMS
    checkpoint_steps: tuple[int, ...] = (10_000, 20_000, 30_000)
    diagnostic_steps: tuple[int, ...] = (10_000, 20_000, 30_000)
    final_step: int = 30_000
    bootstrap_replicates: int = 500
    density_coverage_neighbors: int = 5
    drift_effect_retention: float = 0.80
    rank_retention_floor: float = 0.85
    quality_retention_fraction: float = 0.90
    unit_wins_required: int = 2

    def validate(self) -> None:
        if not self.units or len(set(self.units)) != len(self.units):
            raise ValueError("B2.5 units must be nonempty and distinct")
        if tuple(self.arms) != B25_ARMS:
            raise ValueError("B2.5 is the declared four-cell factorial")
        if self.final_step <= 0:
            raise ValueError("B2.5 final step must be positive")
        if self.checkpoint_steps != tuple(sorted(set(self.checkpoint_steps))):
            raise ValueError("B2.5 checkpoints must be sorted and unique")
        if not self.checkpoint_steps or self.checkpoint_steps[-1] != self.final_step:
            raise ValueError("B2.5 final step must be the final checkpoint")
        if any(step <= 0 or step > self.final_step for step in self.checkpoint_steps):
            raise ValueError("B2.5 checkpoints lie outside the training budget")
        if any(step not in self.checkpoint_steps for step in self.diagnostic_steps):
            raise ValueError("gradient diagnostics must occur at checkpoints")
        if self.bootstrap_replicates < 100:
            raise ValueError("B2.5 bootstrap budget is too small")
        if self.density_coverage_neighbors < 1:
            raise ValueError("density/coverage k must be positive")
        for value in (
            self.drift_effect_retention,
            self.rank_retention_floor,
            self.quality_retention_fraction,
        ):
            if not 0 < value <= 1:
                raise ValueError("B2.5 retention thresholds lie in (0,1]")
        if not 1 <= self.unit_wins_required <= len(self.units):
            raise ValueError("invalid B2.5 unit consistency requirement")


def b25_config() -> B25Config:
    result = B25Config()
    result.validate()
    return result


def _validate_arm(arm: str) -> B25Arm:
    if arm not in B25_ARMS:
        raise ValueError(f"unknown B2.5 arm {arm!r}; expected {B25_ARMS}")
    return arm  # type: ignore[return-value]


def arm_has_b1(arm: str) -> bool:
    return _validate_arm(arm) in ("B1", "B1B2")


def arm_has_b2(arm: str) -> bool:
    return _validate_arm(arm) in ("B2", "B1B2")


def b25_seed(unit: int, role: str, *, arm: str | None = None) -> int:
    """Stage-local seeds; shared roles deliberately omit the arm label."""
    labels: tuple[object, ...] = (B25_PHASE, unit, role)
    if arm is not None:
        labels += (_validate_arm(arm),)
    return b2_seed("b25", unit, ":".join(map(str, labels)))


def paired_seed_manifest(unit: int, arm: str) -> dict[str, int]:
    """Make common and intervention-specific streams auditable."""
    _validate_arm(arm)
    result = {
        f"shared_{role}": f3b_seed(B25_PHASE, unit, role)
        for role in (
            "model-init",
            "data-order",
            "endpoint-noise",
            "bridge-time",
            "augmentation",
        )
    }
    if arm_has_b1(arm):
        from ..b1 import b1_seed  # local import keeps the manifest explicit

        result |= {
            f"b1_{role}": b1_seed(B25_PHASE, unit, role)
            for role in (
                "anchor-data",
                "anchor-prior",
                "anchor-augmentation",
                "anchor-train-bank",
            )
        }
    if arm_has_b2(arm):
        result |= {
            f"b2_{role}": b2_seed(B25_PHASE, unit, role)
            for role in (
                "positive-data",
                "positive-augmentation",
                "probe-data",
                "probe-augmentation",
                "probe-noise",
                "negative-prior",
            )
        }
    return result


def _gradient_snapshot(model: nn.Module) -> list[torch.Tensor | None]:
    return [
        None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    ]


def _gradient_difference(
    after: Sequence[torch.Tensor | None], before: Sequence[torch.Tensor | None]
) -> list[torch.Tensor | None]:
    if len(after) != len(before):
        raise ValueError("gradient snapshots have different lengths")
    result: list[torch.Tensor | None] = []
    for upper, lower in zip(after, before, strict=True):
        if upper is None and lower is None:
            result.append(None)
        elif upper is None:
            result.append(-lower)  # type: ignore[operator]
        elif lower is None:
            result.append(upper.clone())
        else:
            result.append(upper - lower)
    return result


def _gradient_inner(
    left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]
) -> float:
    if len(left) != len(right):
        raise ValueError("gradient vectors have different lengths")
    total = torch.zeros((), dtype=torch.float64)
    for first, second in zip(left, right, strict=True):
        if first is not None and second is not None:
            total = (
                total + (first.detach().double() * second.detach().double()).sum().cpu()
            )
    return float(total)


def _gradient_norm(values: Sequence[torch.Tensor | None]) -> float:
    return math.sqrt(max(_gradient_inner(values, values), 0.0))


def _gradient_cosine(
    left: Sequence[torch.Tensor | None], right: Sequence[torch.Tensor | None]
) -> float | None:
    denominator = _gradient_norm(left) * _gradient_norm(right)
    return None if denominator <= 0 else _gradient_inner(left, right) / denominator


def _component_diagnostics(
    snapshots: dict[str, Sequence[torch.Tensor | None]], clip: float
) -> dict:
    total = snapshots["total"]
    total_norm = _gradient_norm(total)
    clip_factor = min(1.0, clip / max(total_norm, 1e-30))
    components = {name: values for name, values in snapshots.items() if name != "total"}
    norms = {name: _gradient_norm(values) for name, values in components.items()}
    cosines: dict[str, float | None] = {}
    names = tuple(components)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            cosines[f"{left}_vs_{right}"] = _gradient_cosine(
                components[left], components[right]
            )
    return {
        "weighted_component_norms_pre_clip": norms,
        "weighted_component_norms_post_clip": {
            name: value * clip_factor for name, value in norms.items()
        },
        "pairwise_cosines": cosines,
        "combined_norm_pre_clip": total_norm,
        "global_clip_factor": clip_factor,
    }


def _health_summary(events: list[dict]) -> dict:
    if not events:
        return {"events": 0, "quantiles": {}}
    keys = (
        ("positive_ess_minimum", "positive", "ess_fraction_minimum"),
        ("positive_ess_median", "positive", "ess_fraction_median"),
        ("positive_max_weight_median", "positive", "maximum_weight_median"),
        ("negative_ess_minimum", "negative", "ess_fraction_minimum"),
        ("negative_ess_median", "negative", "ess_fraction_median"),
        ("negative_max_weight_median", "negative", "maximum_weight_median"),
    )
    quantiles = {}
    for output, role, field in keys:
        values = np.asarray([event[role][field] for event in events], dtype=float)
        quantiles[output] = {
            f"q{int(q * 100):02d}": float(np.quantile(values, q))
            for q in (0.01, 0.05, 0.10, 0.50, 0.90, 0.99)
        }
    return {"events": len(events), "quantiles": quantiles}


@contextlib.contextmanager
def _preserve_training_mode(model: nn.Module) -> Iterator[None]:
    was_training = model.training
    try:
        yield
    finally:
        model.train(was_training)


@dataclass
class B25TrainResult(BridgeTrainResult):
    arm: str
    anchor_events: int
    anchor_refreshes: int
    correction_events: int
    anchor_model_forwards: int
    correction_model_forwards: int
    lambda_b1: float | None
    lambda_b2: float | None
    kernel_health_summary: dict
    component_gradient_diagnostics: list[dict]
    peak_memory_reserved_bytes: int | None


def train_b25_arm(
    pool: torch.Tensor,
    model_config: F3BModelConfig,
    train_config: F3BTrainConfig,
    unit: int,
    arm: str,
    device: torch.device | str,
    *,
    b1_scale: float,
    lambda_b1: float,
    tau_b2: float,
    lambda_b2: float,
    b1_config: B1Config,
    b2_config: B2Config,
    stage_config: B25Config,
    checkpoint: Callable[[int, TimeConditionedUNet, dict], None] | None = None,
) -> B25TrainResult:
    """Train one paired factorial cell without confirmation-unit restrictions."""
    arm = _validate_arm(arm)
    model_config.validate()
    train_config.validate()
    b1_config.validate()
    b2_config.validate()
    stage_config.validate()
    if unit not in stage_config.units:
        raise ValueError(f"B2.5 unit must lie in {stage_config.units}")
    if tuple(train_config.checkpoint_steps) != stage_config.checkpoint_steps:
        raise ValueError("training checkpoints differ from the B2.5 protocol")
    if train_config.steps != stage_config.final_step:
        raise ValueError("training budget differs from the B2.5 protocol")
    for value, name in (
        (b1_scale, "B1 scale"),
        (lambda_b1, "B1 lambda"),
        (tau_b2, "B2 tau"),
        (lambda_b2, "B2 lambda"),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid frozen {name}")

    device = torch.device(device)
    model = TimeConditionedUNet(
        model_config, f3b_seed(B25_PHASE, unit, "model-init")
    ).to(device)
    flow_streams = bridge_streams(B25_PHASE, unit)
    anchor_streams = b1_streams(B25_PHASE, unit) if arm_has_b1(arm) else None
    correction_streams = b2_streams(B25_PHASE, unit) if arm_has_b2(arm) else None
    dimension = model_config.channels * model_config.image_size**2
    training_bank = (
        build_training_bank(b1_scale, dimension, B25_PHASE, unit, b1_config)
        if arm_has_b1(arm)
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    ema = EMAState(model, train_config.ema_decay)
    history: list[dict] = []
    gradient_diagnostics: list[dict] = []
    kernel_events: list[dict] = []
    anchor_events = anchor_refreshes = correction_events = 0
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
        scheduled = step % b1_config.anchor_every == 0
        if b1_config.anchor_every != b2_config.correction_every:
            raise ValueError("B2.5 requires the frozen B1/B2 cadences to agree")
        diagnose = step in stage_config.diagnostic_steps

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(mixed, time_value)
        flow_value = F.mse_loss(prediction, target)
        if not bool(torch.isfinite(flow_value)):
            raise FloatingPointError(f"non-finite B2.5 flow loss at step {step}")
        flow_value.backward()
        before_anchor = _gradient_snapshot(model) if diagnose else None

        anchor_value: torch.Tensor | None = None
        anchor_record: dict = {}
        after_anchor: list[torch.Tensor | None] | None = None
        if scheduled and arm_has_b1(arm):
            assert anchor_streams is not None and training_bank is not None
            anchor_events += 1
            real_batch = anchor_target_batch(
                pool,
                anchor_streams,
                b1_config.anchor_batch,
                train_config.horizontal_flip,
            )
            anchor_value, anchor_record = anchor_term(
                model,
                training_bank,
                real_batch,
                model_config,
                anchor_streams.anchor_prior,
                device,
                step / train_config.steps,
                b1_config,
            )
            (lambda_b1 * anchor_value.to(flow_value.dtype)).backward()
            after_anchor = _gradient_snapshot(model) if diagnose else None

        correction_value: torch.Tensor | None = None
        kernel_health: dict = {}
        before_correction = (
            _gradient_snapshot(model)
            if diagnose and scheduled and arm_has_b2(arm)
            else None
        )
        if scheduled and arm_has_b2(arm):
            assert correction_streams is not None
            correction_events += 1
            correction_value, kernel_health = correction_term(
                model,
                pool,
                model_config,
                correction_streams,
                device,
                tau_b2,
                b2_config,
                horizontal_flip=train_config.horizontal_flip,
            )
            (lambda_b2 * correction_value.to(flow_value.dtype)).backward()
            kernel_events.append(kernel_health)

        total_scalar = float(flow_value.detach())
        if anchor_value is not None:
            total_scalar += lambda_b1 * float(anchor_value.detach())
        if correction_value is not None:
            total_scalar += lambda_b2 * float(correction_value.detach())
        if not math.isfinite(total_scalar):
            raise FloatingPointError(f"non-finite B2.5 objective at step {step}")

        if diagnose:
            total_snapshot = _gradient_snapshot(model)
            snapshots: dict[str, Sequence[torch.Tensor | None]] = {
                "flow": before_anchor or before_correction or total_snapshot,
                "total": total_snapshot,
            }
            if after_anchor is not None and before_anchor is not None:
                snapshots["b1_weighted"] = _gradient_difference(
                    after_anchor, before_anchor
                )
            if before_correction is not None:
                snapshots["b2_weighted"] = _gradient_difference(
                    total_snapshot, before_correction
                )
            diagnostics = _component_diagnostics(snapshots, train_config.gradient_clip)
            diagnostics.update({"step": step, "arm": arm})
            gradient_diagnostics.append(diagnostics)

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.gradient_clip
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise FloatingPointError(f"non-finite B2.5 gradient at step {step}")
        optimizer.step()
        ema.update(model)

        if anchor_value is not None and should_refresh(anchor_events, b1_config):
            assert training_bank is not None
            from ..b1 import b1_seed

            training_bank = refresh_bank(
                training_bank,
                b1_config.refresh_fraction,
                b1_seed(B25_PHASE, unit, "anchor-refresh", anchor_events),
            )
            anchor_refreshes += 1

        record = {
            "step": step,
            "arm": arm,
            "loss": total_scalar,
            "flow_loss": float(flow_value.detach()),
            "anchor_loss": None
            if anchor_value is None
            else float(anchor_value.detach()),
            "correction_loss": None
            if correction_value is None
            else float(correction_value.detach()),
            "anchor_event_count": anchor_events,
            "correction_event_count": correction_events,
            "anchor_refreshes": anchor_refreshes,
            "gradient_norm_before_clip": float(gradient_norm),
            "examples_seen": step * train_config.batch,
            "wall_seconds": time.time() - started,
        }
        record.update(anchor_record)
        if kernel_health:
            record["kernel_health"] = kernel_health
        if (
            step == 1
            or step % train_config.log_every == 0
            or step in train_config.checkpoint_steps
        ):
            history.append(record)
        if checkpoint is not None and step in train_config.checkpoint_steps:
            with ema.average_parameters(model), _preserve_training_mode(model):
                model.eval()
                checkpoint(step, model, record)

    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
    )
    return B25TrainResult(
        model=model,
        ema=ema,
        history=history,
        wall_seconds=time.time() - started,
        peak_memory_bytes=peak,
        examples_seen=train_config.steps * train_config.batch,
        optimizer_updates=train_config.steps,
        arm=arm,
        anchor_events=anchor_events,
        anchor_refreshes=anchor_refreshes,
        correction_events=correction_events,
        anchor_model_forwards=anchor_events * b1_config.anchor_nfe,
        correction_model_forwards=correction_events * b2_config.correction_nfe,
        lambda_b1=lambda_b1 if arm_has_b1(arm) else None,
        lambda_b2=lambda_b2 if arm_has_b2(arm) else None,
        kernel_health_summary=_health_summary(kernel_events),
        component_gradient_diagnostics=gradient_diagnostics,
        peak_memory_reserved_bytes=peak_reserved,
    )
