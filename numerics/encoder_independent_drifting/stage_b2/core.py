"""Stage B2: differentiable, data-space Laplace drift-energy correction.

The population object mirrored here is the normalized mean-shift difference

    V[p,q](x) = sum_y k(x,y)y / sum_y k(x,y)
                 - sum_z k(x,z)z / sum_z k(x,z),

for ``k(x,y) = exp(-||x-y|| / tau)``.  The displacement ``-x`` cancels
between the two normalized means.  Unlike :func:`kernel_gradient.field`, this
module deliberately retains the graph through the generated ``q`` samples.

Three sample roles are independent:

* probes are target samples plus non-degenerate Gaussian noise;
* positives are fresh target samples;
* negatives are a separate differentiable model trajectory.

At population level the noised probe law has full Euclidean support.  This is
the support premise needed to turn zero integrated energy of the continuous
field into pointwise zero drift.  Finite batches remain only a stochastic
optimization surrogate and are never described as an identifiability proof.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..config import MASTER_SEED, derive_seed
from ..f3b import (
    CONFIRMATION_UNITS,
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

B2_SEED_OFFSET = 107_000
B2_CONFIRMATION_UNITS = CONFIRMATION_UNITS
B2_CALIBRATION_UNITS = (420, 421, 422)


@dataclass(frozen=True)
class B2Config:
    """Every B2-only decision that must cross the artifact boundary."""

    correction_every: int = 10
    probe_batch: int = 64
    positive_batch: int = 64
    negative_batch: int = 64
    correction_nfe: int = 8
    probe_noise_std: float = 0.05
    bandwidth_quantile: float = 0.5
    target_ess_fraction: float = 0.60
    ess_samples: int = 128
    ess_iterations: int = 24
    event_gradient_ratio: float = 0.25
    calibration_ratio_tolerance_factor: float = 2.0
    maximum_preflight_memory_fraction: float = 0.95
    audit_batch: int = 128
    audit_replicates: int = 6
    baseline_recall_floor: float = 0.05
    recall_noninferiority_margin: float = 0.025
    drift_reduction_fraction: float = 0.5
    drift_paired_wins_required: int = 5
    metric_control_floor: float = 0.5

    def validate(self) -> None:
        if self.correction_every <= 0 or self.correction_nfe <= 0:
            raise ValueError("B2 cadence and trajectory length must be positive")
        if min(self.probe_batch, self.positive_batch, self.negative_batch) < 2:
            raise ValueError("each B2 stochastic role needs at least two samples")
        if self.probe_noise_std <= 0 or not math.isfinite(self.probe_noise_std):
            raise ValueError("the B2 probe law needs positive finite noise")
        if not 0 < self.bandwidth_quantile <= 1:
            raise ValueError("bandwidth quantile must lie in (0,1]")
        if not 0 < self.target_ess_fraction < 1:
            raise ValueError("target ESS fraction must lie in (0,1)")
        if self.ess_samples < 3 or self.ess_iterations <= 0:
            raise ValueError("invalid B2 bandwidth-calibration budget")
        if self.event_gradient_ratio <= 0:
            raise ValueError("B2 event gradient ratio must be positive")
        if self.calibration_ratio_tolerance_factor < 1:
            raise ValueError("B2 gradient tolerance factor must be >= 1")
        if not 0 < self.maximum_preflight_memory_fraction < 1:
            raise ValueError("B2 memory fraction must lie in (0,1)")
        if self.audit_batch < 2 or self.audit_replicates < 2:
            raise ValueError("B2 audit sizes must be at least two")
        if not 0 < self.baseline_recall_floor < 1:
            raise ValueError("invalid B2 baseline recall floor")
        if not 0 < self.recall_noninferiority_margin < 1:
            raise ValueError("invalid B2 recall margin")
        if not 0 < self.drift_reduction_fraction < 1:
            raise ValueError("B2 drift reduction fraction must lie in (0,1)")
        if not 1 <= self.drift_paired_wins_required <= self.audit_replicates:
            raise ValueError("invalid required number of B2 paired wins")

    @property
    def effective_gradient_ratio(self) -> float:
        return self.event_gradient_ratio / self.correction_every


def b2_config() -> B2Config:
    result = B2Config()
    result.validate()
    return result


def config_payload(config: B2Config) -> dict:
    config.validate()
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def b2_seed(phase: str, unit: int | str, role: str, event: int | None = None) -> int:
    labels: tuple[object, ...] = ("b2", phase, unit, role)
    if event is not None:
        labels += (event,)
    return derive_seed(MASTER_SEED + B2_SEED_OFFSET, *labels)


def _flatten(images: torch.Tensor) -> torch.Tensor:
    if images.ndim < 2:
        raise ValueError("B2 samples must have a batch and data dimension")
    return images.reshape(len(images), -1)


def _validate_field_inputs(
    probes: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, tau: float
) -> None:
    if min(len(probes), len(positive), len(negative)) < 1:
        raise ValueError("B2 field roles must be nonempty")
    if probes.shape[1:] != positive.shape[1:] or probes.shape[1:] != negative.shape[1:]:
        raise ValueError("B2 probes, positives, and negatives must share a data shape")
    if probes.device != positive.device or probes.device != negative.device:
        raise ValueError("B2 field roles must live on one device")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Laplace bandwidth must be positive and finite")


def _laplace_weights(
    left: torch.Tensor, right: torch.Tensor, tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stable exact-Laplace row weights and their Euclidean distances.

    ``softmax(-distance/tau)`` analytically equals normalized Laplace weights,
    while subtracting the row maximum internally.  It therefore cannot turn a
    remote but finite row into the all-zero row that broke the raw Algorithm-2
    objective.  PyTorch uses the zero subgradient of ``cdist`` at coincidence;
    independent roles make exact coincidences a probability-zero event in the
    continuous idealization.
    """
    distances = torch.cdist(left, right, p=2)
    weights = torch.softmax(-distances / tau, dim=1)
    return weights, distances


def _weight_health(weights: torch.Tensor, distances: torch.Tensor) -> dict:
    with torch.no_grad():
        ess = weights.square().sum(dim=1).reciprocal()
        normalized_ess = ess / weights.shape[1]
        return {
            "ess_fraction_median": float(normalized_ess.median()),
            "ess_fraction_minimum": float(normalized_ess.min()),
            "maximum_weight_median": float(weights.max(dim=1).values.median()),
            "distance_median": float(distances.median()),
            "distance_minimum": float(distances.min()),
            "distance_maximum": float(distances.max()),
            "row_sum_error_maximum": float((weights.sum(dim=1) - 1.0).abs().max()),
        }


def laplace_mean_shift_field(
    probes: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    tau: float,
    *,
    diagnostics: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Differentiable sample-split normalized Laplace mean-shift difference."""
    _validate_field_inputs(probes, positive, negative, tau)
    probe_flat = _flatten(probes)
    positive_flat = _flatten(positive)
    negative_flat = _flatten(negative)
    weights_positive, distances_positive = _laplace_weights(
        probe_flat, positive_flat, tau
    )
    weights_negative, distances_negative = _laplace_weights(
        probe_flat, negative_flat, tau
    )
    field_flat = weights_positive @ positive_flat - weights_negative @ negative_flat
    if not bool(torch.isfinite(field_flat).all()):
        raise FloatingPointError("non-finite normalized Laplace drift field")
    stats: dict = {}
    if diagnostics:
        stats = {
            "kernel": "exact_laplace",
            "normalization": "row_normalized_mean_shift_difference",
            "tau": float(tau),
            "positive": _weight_health(weights_positive, distances_positive),
            "negative": _weight_health(weights_negative, distances_negative),
            "field_rms": float(field_flat.detach().square().mean().sqrt()),
            "field_l2_per_probe": float(
                field_flat.detach().square().sum(dim=1).mean().sqrt()
            ),
        }
    return field_flat.reshape(probes.shape), stats


def laplace_drift_energy(
    probes: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    tau: float,
    *,
    diagnostics: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Mean per-probe squared L2 field norm; never square the mean field."""
    field, stats = laplace_mean_shift_field(
        probes, positive, negative, tau, diagnostics=diagnostics
    )
    value = field.reshape(len(field), -1).square().sum(dim=1).mean()
    if not bool(torch.isfinite(value)) or float(value.detach()) < -1e-12:
        raise FloatingPointError("invalid B2 drift energy")
    if diagnostics:
        stats["drift_energy"] = float(value.detach())
    return value, stats


def _off_diagonal_ess_fraction(distances: torch.Tensor, tau: float) -> float:
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("ESS calibration requires a square distance matrix")
    n = distances.shape[0]
    if n < 3:
        raise ValueError("ESS calibration requires at least three targets")
    logits = -distances / tau
    logits = logits.masked_fill(
        torch.eye(n, dtype=torch.bool, device=distances.device), -torch.inf
    )
    weights = torch.softmax(logits, dim=1)
    ess = weights.square().sum(dim=1).reciprocal() / (n - 1)
    return float(ess.median())


def calibrate_laplace_bandwidth(
    target: torch.Tensor, config: B2Config
) -> tuple[float, dict]:
    """Target-only exact-Laplace calibration using off-diagonal realized ESS."""
    config.validate()
    if len(target) < config.ess_samples:
        raise ValueError("target pool is too small for B2 ESS calibration")
    with torch.no_grad():
        samples = _flatten(target[: config.ess_samples]).double()
        distances = torch.cdist(samples, samples, p=2)
        mask = ~torch.eye(len(samples), dtype=torch.bool, device=samples.device)
        off_diagonal = distances[mask]
        base = float(torch.quantile(off_diagonal, config.bandwidth_quantile))
        if not math.isfinite(base) or base <= 0:
            raise ValueError("B2 target distances do not define a bandwidth")
        low, high = 1e-4, 1e4
        for _ in range(config.ess_iterations):
            middle = math.sqrt(low * high)
            achieved = _off_diagonal_ess_fraction(distances, base * middle)
            if achieved < config.target_ess_fraction:
                low = middle
            else:
                high = middle
        multiplier = math.sqrt(low * high)
        tau = base * multiplier
        achieved = _off_diagonal_ess_fraction(distances, tau)
    return float(tau), {
        "kernel": "exact_laplace",
        "quantile": config.bandwidth_quantile,
        "base_distance": base,
        "global_multiplier": multiplier,
        "tau": float(tau),
        "target_ess_fraction": config.target_ess_fraction,
        "achieved_off_diagonal_ess_fraction": achieved,
        "exclude_self": True,
        "ess_samples": config.ess_samples,
        "ess_iterations": config.ess_iterations,
    }


@dataclass
class B2Streams:
    positive_data: np.random.Generator
    positive_augmentation: torch.Generator
    probe_data: np.random.Generator
    probe_augmentation: torch.Generator
    probe_noise: torch.Generator
    negative_prior: torch.Generator


def b2_streams(phase: str, unit: int) -> B2Streams:
    def torch_stream(role: str) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(
            b2_seed(phase, unit, role) % (2**63 - 1)
        )

    return B2Streams(
        positive_data=np.random.default_rng(b2_seed(phase, unit, "positive-data")),
        positive_augmentation=torch_stream("positive-augmentation"),
        probe_data=np.random.default_rng(b2_seed(phase, unit, "probe-data")),
        probe_augmentation=torch_stream("probe-augmentation"),
        probe_noise=torch_stream("probe-noise"),
        negative_prior=torch_stream("negative-prior"),
    )


def _augmented_target_batch(
    pool: torch.Tensor,
    data_stream: np.random.Generator,
    augmentation_stream: torch.Generator,
    batch: int,
    horizontal_flip: bool,
) -> torch.Tensor:
    indices = data_stream.integers(0, len(pool), size=batch)
    result = pool[torch.as_tensor(indices)].clone()
    if horizontal_flip:
        flip = torch.rand(batch, generator=augmentation_stream) < 0.5
        result[flip] = torch.flip(result[flip], dims=(-1,))
    return result


def correction_term(
    model: nn.Module,
    target_pool: torch.Tensor,
    model_config: F3BModelConfig,
    streams: B2Streams,
    device: torch.device | str,
    tau: float,
    config: B2Config,
    horizontal_flip: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Construct one differentiable, sample-split B2 correction event."""
    config.validate()
    device = torch.device(device)
    positive = _augmented_target_batch(
        target_pool,
        streams.positive_data,
        streams.positive_augmentation,
        config.positive_batch,
        horizontal_flip=horizontal_flip,
    ).to(device)
    probe_centres = _augmented_target_batch(
        target_pool,
        streams.probe_data,
        streams.probe_augmentation,
        config.probe_batch,
        horizontal_flip=horizontal_flip,
    )
    probe_noise = torch.randn(
        probe_centres.shape,
        generator=streams.probe_noise,
        dtype=probe_centres.dtype,
    )
    probes = (probe_centres + config.probe_noise_std * probe_noise).to(device)
    initial = torch.randn(
        config.negative_batch,
        model_config.channels,
        model_config.image_size,
        model_config.image_size,
        generator=streams.negative_prior,
    ).to(device)
    negative = euler_integrate(model, initial, config.correction_nfe)
    value, health = laplace_drift_energy(
        probes.detach(), positive.detach(), negative, tau, diagnostics=True
    )
    health.update(
        {
            "probe_batch": config.probe_batch,
            "positive_batch": config.positive_batch,
            "negative_batch": config.negative_batch,
            "correction_nfe": config.correction_nfe,
            "probe_noise_std": config.probe_noise_std,
            "gradient_roles": ["negative_model_samples"],
            "detached_roles": ["full_support_probes", "target_positives"],
        }
    )
    return value, health


def total_loss(
    flow_loss: torch.Tensor, correction: torch.Tensor | None, lambda_event: float
) -> torch.Tensor:
    if not math.isfinite(lambda_event) or lambda_event <= 0:
        raise ValueError("B2 event weight must be positive and finite")
    if correction is None:
        return flow_loss
    return flow_loss + lambda_event * correction.to(flow_loss.dtype)


def should_apply(step: int, config: B2Config) -> bool:
    if step <= 0:
        raise ValueError("training steps are one-indexed")
    return step % config.correction_every == 0


def calibrated_event_lambda(
    flow_gradient_norm: float, correction_gradient_norm: float, config: B2Config
) -> float:
    if (
        not math.isfinite(flow_gradient_norm)
        or not math.isfinite(correction_gradient_norm)
        or flow_gradient_norm <= 0
        or correction_gradient_norm <= 0
    ):
        raise ValueError("B2 calibration requires positive finite gradient norms")
    return config.event_gradient_ratio * flow_gradient_norm / correction_gradient_norm


def parameter_gradient_norm(loss: torch.Tensor, model: nn.Module) -> float:
    """L2 norm of a loss gradient without mutating parameter ``.grad`` fields."""
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


@contextlib.contextmanager
def _preserve_training_mode(model: nn.Module) -> Iterator[None]:
    was_training = model.training
    try:
        yield
    finally:
        model.train(was_training)


@dataclass
class B2TrainResult(BridgeTrainResult):
    correction_events: int
    correction_model_forwards: int
    lambda_event: float


def train_b2(
    pool: torch.Tensor,
    model_config: F3BModelConfig,
    train_config: F3BTrainConfig,
    unit: int,
    device: torch.device | str,
    tau: float,
    lambda_event: float,
    config: B2Config,
    checkpoint: Callable[[int, TimeConditionedUNet, dict], None] | None = None,
) -> B2TrainResult:
    """Train B2 with the exact B0 initialization and flow-data streams."""
    config.validate()
    model_config.validate()
    train_config.validate()
    if unit not in B2_CONFIRMATION_UNITS:
        raise ValueError(f"paired B2 unit must lie in {B2_CONFIRMATION_UNITS}")
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("invalid frozen B2 bandwidth")
    if not math.isfinite(lambda_event) or lambda_event <= 0:
        raise ValueError("invalid frozen B2 event weight")
    device = torch.device(device)

    model = TimeConditionedUNet(
        model_config, f3b_seed("confirmation", unit, "model-init")
    ).to(device)
    flow_streams = bridge_streams("confirmation", unit)
    correction_streams = b2_streams("confirmation", unit)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    ema = EMAState(model, train_config.ema_decay)
    history: list[dict] = []
    started = time.time()
    correction_events = 0
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
        correction = None
        health: dict = {}
        if should_apply(step, config):
            correction_events += 1
            correction, health = correction_term(
                model,
                pool,
                model_config,
                correction_streams,
                device,
                tau,
                config,
                horizontal_flip=train_config.horizontal_flip,
            )
        loss = total_loss(flow_value, correction, lambda_event)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite B2 loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.gradient_clip
        )
        if not bool(torch.isfinite(torch.as_tensor(gradient_norm))):
            raise FloatingPointError(f"non-finite B2 gradient at step {step}")
        optimizer.step()
        ema.update(model)

        record = {
            "step": step,
            "loss": float(loss.detach()),
            "flow_loss": float(flow_value.detach()),
            "correction_loss": None
            if correction is None
            else float(correction.detach()),
            "lambda_event": float(lambda_event),
            "correction_event": correction is not None,
            "correction_event_count": correction_events,
            "gradient_norm_before_clip": float(gradient_norm),
            "examples_seen": step * train_config.batch,
            "wall_seconds": time.time() - started,
        }
        if health:
            record["kernel_health"] = health
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
    return B2TrainResult(
        model=model,
        ema=ema,
        history=history,
        wall_seconds=time.time() - started,
        peak_memory_bytes=peak,
        examples_seen=train_config.steps * train_config.batch,
        optimizer_updates=train_config.steps,
        correction_events=correction_events,
        correction_model_forwards=correction_events * config.correction_nfe,
        lambda_event=float(lambda_event),
    )


def paired_seed_manifest(unit: int) -> dict[str, int]:
    return {
        **{
            f"shared_{role}": f3b_seed("confirmation", unit, role)
            for role in (
                "model-init",
                "data-order",
                "endpoint-noise",
                "bridge-time",
                "augmentation",
            )
        },
        **{
            f"b2_{role}": b2_seed("confirmation", unit, role)
            for role in (
                "positive-data",
                "positive-augmentation",
                "probe-data",
                "probe-augmentation",
                "probe-noise",
                "negative-prior",
            )
        },
    }


def evaluation_prior_seed(unit: int, replicate: int = 0) -> int:
    if unit not in B2_CONFIRMATION_UNITS:
        raise ValueError(f"unknown B2 unit {unit}")
    return b2_seed("confirmation-evaluation", unit, "model-negative", replicate)


def gradient_calibration_row(
    model: nn.Module,
    flow_loss: torch.Tensor,
    correction: torch.Tensor,
    config: B2Config,
) -> dict:
    """Pure calibration arithmetic shared by the preflight and tests."""
    flow_norm = parameter_gradient_norm(flow_loss, model)
    correction_norm = parameter_gradient_norm(correction, model)
    event_weight = calibrated_event_lambda(flow_norm, correction_norm, config)
    return {
        "flow_gradient_norm": flow_norm,
        "correction_gradient_norm": correction_norm,
        "lambda_event": event_weight,
        "weighted_event_gradient_ratio": event_weight * correction_norm / flow_norm,
    }
