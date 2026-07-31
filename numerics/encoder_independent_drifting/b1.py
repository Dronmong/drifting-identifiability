"""Paired Stage-B1 flow matching plus an encoder-free spectral anchor.

The ideal population anchor is

    A(p, q) = E_w |phi_p(w) - phi_q(w)|^2.

For the full-support spectral law used by :mod:`spectral_anchor`, ``A = 0``
identifies the source law by characteristic-function uniqueness.  The finite
random-feature V-statistic used here is only a nonnegative stochastic
optimization surrogate; it is not itself measure determining.

B1 deliberately reuses each successful B0 unit's model initialization and
flow-matching streams.  Anchor data, priors, banks, and refreshes use separate
streams.  Thus a paired B0/B1 comparison changes the training objective but
does not confound it with a new flow batch order or initialization.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import MASTER_SEED, AnchorConfig, derive_seed
from .f3b import (
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
from .spectral_anchor import (
    SpectralBank,
    anchor_loss,
    build_bank,
    projected_scale,
    refresh_bank,
)

B1_SEED_OFFSET = 91_000
B1_CONFIRMATION_UNITS = CONFIRMATION_UNITS
B1_CALIBRATION_UNITS = (410, 411, 412)


@dataclass(frozen=True)
class B1Config:
    """Every B1-only knob serialized into calibration and freeze artifacts."""

    anchor_every: int = 10
    anchor_batch: int = 64
    anchor_nfe: int = 8
    scale_probe_samples: int = 256
    audit_features: int = 512
    audit_replicates: int = 6
    refresh_every_events: int = 25
    refresh_fraction: float = 0.25
    event_gradient_ratio: float = 0.25
    calibration_ratio_tolerance_factor: float = 2.0
    recall_noninferiority_margin: float = 0.05
    anchor_reduction_fraction: float = 0.5
    anchor_paired_wins_required: int = 5
    metric_control_floor: float = 0.5
    anchor: AnchorConfig = field(
        default_factory=lambda: AnchorConfig(
            features=256,
            audit_features=512,
            refresh_every=25,
            refresh_fraction=0.25,
            band_schedule="coarse_to_fine",
        )
    )

    def validate(self) -> None:
        if self.anchor_every <= 0 or self.anchor_batch <= 1 or self.anchor_nfe <= 0:
            raise ValueError("anchor cadence, batch, and NFE must be positive")
        if self.scale_probe_samples < 2:
            raise ValueError("projected-scale calibration needs at least two samples")
        if self.audit_features <= 0 or self.audit_replicates < 2:
            raise ValueError("audit bank and replicate counts must be positive")
        if not 0 <= self.refresh_fraction <= 1 or self.refresh_every_events <= 0:
            raise ValueError("invalid training-bank refresh rule")
        if self.event_gradient_ratio <= 0:
            raise ValueError("event gradient ratio must be positive")
        if self.calibration_ratio_tolerance_factor < 1:
            raise ValueError("gradient-ratio tolerance factor must be >= 1")
        if not 0 < self.recall_noninferiority_margin < 1:
            raise ValueError("invalid recall non-inferiority margin")
        if not 0 < self.anchor_reduction_fraction < 1:
            raise ValueError("anchor reduction fraction must lie in (0,1)")
        if not 1 <= self.anchor_paired_wins_required <= self.audit_replicates:
            raise ValueError("invalid required number of paired audit wins")
        if self.anchor.training_estimator != "biased":
            raise ValueError("B1 training requires the nonnegative biased estimator")
        if self.anchor.features <= 0:
            raise ValueError("the training bank must be nonempty")
        if self.anchor.audit_features != self.audit_features:
            raise ValueError("B1 and AnchorConfig audit widths differ")
        if self.anchor.refresh_every != self.refresh_every_events:
            raise ValueError("B1 and AnchorConfig refresh cadences differ")
        if self.anchor.refresh_fraction != self.refresh_fraction:
            raise ValueError("B1 and AnchorConfig refresh fractions differ")

    @property
    def effective_gradient_ratio(self) -> float:
        """Average-event interpretation of the calibrated gradient ratio."""
        return self.event_gradient_ratio / self.anchor_every


def b1_config() -> B1Config:
    result = B1Config()
    result.validate()
    return result


def config_payload(config: B1Config) -> dict:
    """Canonical JSON-compatible form used at every artifact boundary."""
    config.validate()
    return json.loads(json.dumps(asdict(config), sort_keys=True))


def b1_seed(phase: str, unit: int | str, role: str, event: int | None = None) -> int:
    labels: tuple[object, ...] = ("b1", phase, unit, role)
    if event is not None:
        labels += (event,)
    return derive_seed(MASTER_SEED + B1_SEED_OFFSET, *labels)


def scale_calibration_indices(pool_size: int, config: B1Config) -> np.ndarray:
    """A fixed target-only subset, independent of every model outcome."""
    config.validate()
    if pool_size < config.scale_probe_samples:
        raise ValueError("training pool is too small for B1 scale calibration")
    rng = np.random.default_rng(b1_seed("calibration", 0, "scale-indices"))
    return np.sort(
        rng.choice(pool_size, size=config.scale_probe_samples, replace=False)
    )


def calibrate_projected_scale(
    target_pool: torch.Tensor, config: B1Config
) -> tuple[float, np.ndarray]:
    indices = scale_calibration_indices(len(target_pool), config)
    generator = torch.Generator(device="cpu").manual_seed(
        b1_seed("calibration", 0, "scale-directions") % (2**63 - 1)
    )
    samples = target_pool[torch.as_tensor(indices)].detach().cpu()
    value = projected_scale(samples.reshape(len(samples), -1), config.anchor, generator)
    return float(value), indices


def build_training_bank(
    scale: float,
    dimension: int,
    phase: str,
    unit: int | str,
    config: B1Config,
) -> SpectralBank:
    """Construct the mutable training bank; audit banks live elsewhere."""
    config.validate()
    if dimension <= 0:
        raise ValueError("anchor dimension must be positive")
    return build_bank(
        config.anchor,
        dimension,
        scale,
        b1_seed(phase, unit, "anchor-train-bank"),
        features=config.anchor.features,
    )


def build_bank_for_dimension(
    scale: float,
    dimension: int,
    phase: str,
    unit: int | str,
    role: str,
    features: int,
    config: B1Config,
) -> SpectralBank:
    """Dimension-generic helper used by smoke tests and audit replicates."""
    config.validate()
    return build_bank(
        config.anchor,
        dimension,
        scale,
        b1_seed(phase, unit, role),
        features=features,
    )


@dataclass
class B1Streams:
    anchor_data: np.random.Generator
    anchor_prior: torch.Generator
    anchor_augmentation: torch.Generator


def b1_streams(phase: str, unit: int) -> B1Streams:
    return B1Streams(
        anchor_data=np.random.default_rng(b1_seed(phase, unit, "anchor-data")),
        anchor_prior=torch.Generator(device="cpu").manual_seed(
            b1_seed(phase, unit, "anchor-prior") % (2**63 - 1)
        ),
        anchor_augmentation=torch.Generator(device="cpu").manual_seed(
            b1_seed(phase, unit, "anchor-augmentation") % (2**63 - 1)
        ),
    )


def anchor_target_batch(
    pool: torch.Tensor,
    streams: B1Streams,
    batch: int,
    horizontal_flip: bool,
) -> torch.Tensor:
    """Draw a B1 target batch from the same augmented endpoint law as B0."""
    if batch <= 0:
        raise ValueError("anchor target batch must be positive")
    indices = streams.anchor_data.integers(0, len(pool), size=batch)
    target = pool[torch.as_tensor(indices)].clone()
    if horizontal_flip:
        flip = torch.rand(batch, generator=streams.anchor_augmentation) < 0.5
        target[flip] = torch.flip(target[flip], dims=(-1,))
    return target


def anchor_term(
    model: nn.Module,
    bank: SpectralBank,
    real_batch: torch.Tensor,
    model_config: F3BModelConfig,
    prior_generator: torch.Generator,
    device: torch.device | str,
    progress: float,
    config: B1Config,
) -> tuple[torch.Tensor, dict]:
    """Differentiate the finite nonnegative anchor through an Euler trajectory."""
    config.validate()
    if len(real_batch) != config.anchor_batch:
        raise ValueError("B1 anchor target batch has the wrong size")
    initial = torch.randn(
        config.anchor_batch,
        model_config.channels,
        model_config.image_size,
        model_config.image_size,
        generator=prior_generator,
    ).to(device)
    generated = euler_integrate(model, initial, config.anchor_nfe)
    value = anchor_loss(
        bank,
        generated,
        real_batch.to(device),
        estimator="biased",
        progress=progress,
    )
    scalar = float(value.detach())
    if not math.isfinite(scalar) or scalar < -1e-9:
        raise FloatingPointError(f"invalid nonnegative B1 anchor loss: {scalar}")
    return value, {
        "anchor_loss": scalar,
        "anchor_batch": config.anchor_batch,
        "anchor_nfe": config.anchor_nfe,
        "progress": float(progress),
    }


def total_loss(
    flow_loss: torch.Tensor, anchor_value: torch.Tensor | None, lambda_event: float
) -> torch.Tensor:
    """The implemented stochastic surrogate; not a zero-attainability claim."""
    if lambda_event <= 0 or not math.isfinite(lambda_event):
        raise ValueError("B1 event weight must be finite and positive")
    if anchor_value is None:
        return flow_loss
    return flow_loss + lambda_event * anchor_value.to(flow_loss.dtype)


def should_apply(step: int, config: B1Config) -> bool:
    if step <= 0:
        raise ValueError("training steps are one-indexed")
    return step % config.anchor_every == 0


def should_refresh(anchor_event: int, config: B1Config) -> bool:
    if anchor_event <= 0:
        raise ValueError("anchor events are one-indexed")
    return anchor_event % config.refresh_every_events == 0


def parameter_gradient_norm(loss: torch.Tensor, model: nn.Module) -> float:
    """L2 norm of a loss gradient without modifying ``.grad`` fields."""
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


def calibrated_event_lambda(
    flow_gradient_norm: float, anchor_gradient_norm: float, config: B1Config
) -> float:
    """Outcome-blind scale making an anchor event's gradient ratio declared."""
    if (
        not math.isfinite(flow_gradient_norm)
        or not math.isfinite(anchor_gradient_norm)
        or flow_gradient_norm <= 0
        or anchor_gradient_norm <= 0
    ):
        raise ValueError("gradient calibration requires positive finite norms")
    return config.event_gradient_ratio * flow_gradient_norm / anchor_gradient_norm


@contextlib.contextmanager
def _preserve_training_mode(model: nn.Module) -> Iterator[None]:
    was_training = model.training
    try:
        yield
    finally:
        model.train(was_training)


@dataclass
class B1TrainResult(BridgeTrainResult):
    anchor_events: int
    anchor_refreshes: int
    anchor_model_forwards: int
    lambda_event: float


def train_b1(
    pool: torch.Tensor,
    model_config: F3BModelConfig,
    train_config: F3BTrainConfig,
    unit: int,
    device: torch.device | str,
    scale: float,
    lambda_event: float,
    config: B1Config,
    checkpoint: Callable[[int, TimeConditionedUNet, dict], None] | None = None,
) -> B1TrainResult:
    """Train the paired B1 arm with B0-identical flow streams."""
    config.validate()
    model_config.validate()
    train_config.validate()
    if unit not in B1_CONFIRMATION_UNITS:
        raise ValueError(f"paired B1 units must be drawn from {B1_CONFIRMATION_UNITS}")
    if lambda_event <= 0 or not math.isfinite(lambda_event):
        raise ValueError("invalid calibrated B1 event weight")
    device = torch.device(device)

    # These are intentionally the exact B0 confirmation streams.
    model = TimeConditionedUNet(
        model_config, f3b_seed("confirmation", unit, "model-init")
    ).to(device)
    flow_streams = bridge_streams("confirmation", unit)
    anchor_streams = b1_streams("confirmation", unit)
    dimension = model_config.channels * model_config.image_size**2
    training_bank = build_training_bank(scale, dimension, "confirmation", unit, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    ema = EMAState(model, train_config.ema_decay)
    history: list[dict] = []
    started = time.time()
    anchor_events = 0
    anchor_refreshes = 0
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
        anchor_value = None
        anchor_record: dict = {}
        if should_apply(step, config):
            anchor_events += 1
            real_batch = anchor_target_batch(
                pool,
                anchor_streams,
                config.anchor_batch,
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
                config,
            )
        loss = total_loss(flow_value, anchor_value, lambda_event)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite B1 loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.gradient_clip
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError(f"non-finite B1 gradient at step {step}")
        optimizer.step()
        ema.update(model)

        if anchor_value is not None and should_refresh(anchor_events, config):
            training_bank = refresh_bank(
                training_bank,
                config.refresh_fraction,
                b1_seed(
                    "confirmation",
                    unit,
                    "anchor-refresh",
                    anchor_events,
                ),
            )
            anchor_refreshes += 1

        record = {
            "step": step,
            "loss": float(loss.detach()),
            "flow_loss": float(flow_value.detach()),
            "anchor_loss": (
                None if anchor_value is None else float(anchor_value.detach())
            ),
            "lambda_event": float(lambda_event),
            "anchor_event": anchor_value is not None,
            "anchor_event_count": anchor_events,
            "anchor_refreshes": anchor_refreshes,
            "gradient_norm_before_clip": float(gradient_norm),
            "examples_seen": step * train_config.batch,
            "wall_seconds": time.time() - started,
        }
        record.update(anchor_record)
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
    return B1TrainResult(
        model=model,
        ema=ema,
        history=history,
        wall_seconds=time.time() - started,
        peak_memory_bytes=peak,
        examples_seen=train_config.steps * train_config.batch,
        optimizer_updates=train_config.steps,
        anchor_events=anchor_events,
        anchor_refreshes=anchor_refreshes,
        anchor_model_forwards=anchor_events * config.anchor_nfe,
        lambda_event=float(lambda_event),
    )


def paired_seed_manifest(unit: int) -> dict[str, int]:
    """Record common B0 streams and B1-only streams separately."""
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
            f"b1_{role}": b1_seed("confirmation", unit, role)
            for role in (
                "anchor-data",
                "anchor-prior",
                "anchor-augmentation",
                "anchor-train-bank",
            )
        },
    }


def evaluation_prior_seed(unit: int) -> int:
    """Common B0/B1 inference prior used by the paired metric comparison."""
    if unit not in B1_CONFIRMATION_UNITS:
        raise ValueError(f"unknown paired B1 unit {unit}")
    return b1_seed("confirmation-evaluation", unit, "metric-prior")


def load_b0_checkpoint_model(
    checkpoint_record: dict,
    frozen_profile: dict,
    model_config: F3BModelConfig,
    unit: int,
    device: torch.device | str,
) -> TimeConditionedUNet:
    """Load and validate one immutable B0 EMA checkpoint for paired auditing."""
    path = Path(checkpoint_record["path"])
    expected_sha = str(checkpoint_record["sha256"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
        raise RuntimeError(f"B0 checkpoint {unit} changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before the ``weights_only`` keyword.
        payload = torch.load(path, map_location="cpu")
    if int(payload.get("unit", -1)) != unit:
        raise RuntimeError(f"B0 checkpoint payload has the wrong unit for {unit}")
    if payload.get("profile") != frozen_profile:
        raise RuntimeError(f"B0 checkpoint {unit} used a different frozen profile")
    model = TimeConditionedUNet(
        model_config, f3b_seed("confirmation", unit, "model-init")
    ).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model
