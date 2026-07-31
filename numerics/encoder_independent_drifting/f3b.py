"""Prescribed independent-pair flow matching for Stage F3B.

The old one-step generator is deliberately not reused here: it maps a latent
vector to an image and therefore cannot represent ``u(x_t, t)``.  This module
contains an image-to-image, time-conditioned U-Net, the independent linear
bridge, deterministic training streams, and explicit-Euler sampling.

Nothing in this module uses a feature encoder for training.  Inception features
are permitted only in the runners' report-only evaluation layer.
"""

from __future__ import annotations

import contextlib
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import MASTER_SEED, derive_seed

F3B_SEED_OFFSET = 73_000
DEVELOPMENT_UNITS = (200, 201, 202)
CONFIRMATION_UNITS = (300, 301, 302)
RECALL_GATE = 0.05
METRIC_CONTROL_FLOOR = 0.5


@dataclass(frozen=True)
class F3BModelConfig:
    image_size: int = 32
    channels: int = 3
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 2, 2)
    residual_blocks: int = 1
    time_embedding_dim: int = 256
    attention_resolutions: tuple[int, ...] = (8,)
    dropout: float = 0.0

    def validate(self) -> None:
        if self.image_size < 4 or self.image_size & (self.image_size - 1):
            raise ValueError("image_size must be a power of two >= 4")
        if self.channels <= 0 or self.base_channels <= 0:
            raise ValueError("channel counts must be positive")
        if not self.channel_multipliers or any(
            value <= 0 for value in self.channel_multipliers
        ):
            raise ValueError("channel multipliers must be nonempty and positive")
        if self.residual_blocks <= 0:
            raise ValueError("residual_blocks must be positive")
        if self.time_embedding_dim < 4 or self.time_embedding_dim % 2:
            raise ValueError("time_embedding_dim must be even and >= 4")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1)")
        coarsest = self.image_size // (2 ** (len(self.channel_multipliers) - 1))
        if coarsest < 2:
            raise ValueError("too many U-Net levels for the image resolution")
        reachable = {
            self.image_size // (2**level)
            for level in range(len(self.channel_multipliers))
        }
        if any(
            resolution not in reachable for resolution in self.attention_resolutions
        ):
            raise ValueError("attention resolutions must coincide with a U-Net level")


@dataclass(frozen=True)
class F3BTrainConfig:
    steps: int = 30_000
    batch: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    ema_decay: float = 0.9999
    gradient_clip: float = 1.0
    horizontal_flip: bool = True
    log_every: int = 100
    checkpoint_steps: tuple[int, ...] = (1_000, 5_000, 10_000, 30_000)

    def validate(self) -> None:
        if self.steps <= 0 or self.batch <= 0:
            raise ValueError("steps and batch must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer scale")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must lie in [0,1)")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("EMA decay must lie in [0,1)")
        if self.gradient_clip <= 0 or self.log_every <= 0:
            raise ValueError("gradient_clip and log_every must be positive")
        if any(step <= 0 or step > self.steps for step in self.checkpoint_steps):
            raise ValueError("checkpoint steps must lie in [1, steps]")
        if tuple(sorted(set(self.checkpoint_steps))) != self.checkpoint_steps:
            raise ValueError("checkpoint steps must be sorted and unique")


@dataclass(frozen=True)
class F3BEvalConfig:
    generated_samples: int = 512
    reference_samples: int = 2_048
    nfe_ladder: tuple[int, ...] = (8, 32)

    def validate(self) -> None:
        if self.generated_samples < 4 or self.reference_samples < 4:
            raise ValueError("evaluation needs at least four samples per side")
        if not self.nfe_ladder or any(nfe <= 0 for nfe in self.nfe_ladder):
            raise ValueError("NFE values must be positive")
        if tuple(sorted(set(self.nfe_ladder))) != self.nfe_ladder:
            raise ValueError("NFE ladder must be sorted and unique")


@dataclass(frozen=True)
class F3BProfile:
    name: str
    purpose: str
    model: F3BModelConfig
    train: F3BTrainConfig
    evaluation: F3BEvalConfig

    def validate(self) -> None:
        self.model.validate()
        self.train.validate()
        self.evaluation.validate()


def profile(name: str) -> F3BProfile:
    """Declared development profiles; none is silently called published-FID.

    ``smoke`` only validates mechanics. ``compact`` is the resource-realistic
    research arm. ``reference_scale`` uses the conventional 128-channel image
    U-Net scale and a substantially larger budget, but remains a local recipe.
    """
    if name == "smoke":
        result = F3BProfile(
            name="smoke",
            purpose="mechanical smoke test only",
            model=F3BModelConfig(
                image_size=8,
                base_channels=16,
                channel_multipliers=(1, 2),
                residual_blocks=1,
                time_embedding_dim=64,
                attention_resolutions=(),
            ),
            train=F3BTrainConfig(
                steps=4,
                batch=4,
                learning_rate=1e-3,
                ema_decay=0.9,
                log_every=1,
                checkpoint_steps=(2, 4),
            ),
            evaluation=F3BEvalConfig(
                generated_samples=8, reference_samples=8, nfe_ladder=(1, 2)
            ),
        )
    elif name == "compact":
        result = F3BProfile(
            name="compact",
            purpose="resource-realistic B0 reachability development",
            model=F3BModelConfig(),
            train=F3BTrainConfig(),
            evaluation=F3BEvalConfig(),
        )
    elif name == "reference_scale":
        result = F3BProfile(
            name="reference_scale",
            purpose="stronger CIFAR-scale positive-control attempt",
            model=F3BModelConfig(
                base_channels=128,
                channel_multipliers=(1, 2, 2, 2),
                residual_blocks=2,
                time_embedding_dim=512,
                attention_resolutions=(8,),
            ),
            train=F3BTrainConfig(
                steps=150_000,
                batch=64,
                learning_rate=2e-4,
                ema_decay=0.9999,
                log_every=250,
                checkpoint_steps=(10_000, 30_000, 75_000, 150_000),
            ),
            evaluation=F3BEvalConfig(nfe_ladder=(8, 32, 100)),
        )
    else:
        raise ValueError(
            f"unknown F3B profile {name!r}; expected smoke, compact, or reference_scale"
        )
    result.validate()
    return result


def confirmation_profile(development: F3BProfile, steps: int, nfe: int) -> F3BProfile:
    """Remove development ladders and retain exactly one frozen choice."""
    if steps not in development.train.checkpoint_steps:
        raise ValueError("confirmation steps were not measured in development")
    if nfe not in development.evaluation.nfe_ladder:
        raise ValueError("confirmation NFE was not measured in development")
    result = replace(
        development,
        train=replace(development.train, steps=steps, checkpoint_steps=(steps,)),
        evaluation=replace(development.evaluation, nfe_ladder=(nfe,)),
    )
    result.validate()
    return result


def f3b_seed(phase: str, unit: int | str, role: str) -> int:
    """A distinct deterministic stream for every stochastic role."""
    return derive_seed(MASTER_SEED + F3B_SEED_OFFSET, "f3b", phase, unit, role)


@dataclass
class BridgeStreams:
    data: np.random.Generator
    noise: torch.Generator
    time: torch.Generator
    augmentation: torch.Generator


def bridge_streams(phase: str, unit: int) -> BridgeStreams:
    return BridgeStreams(
        data=np.random.default_rng(f3b_seed(phase, unit, "data-order")),
        noise=torch.Generator(device="cpu").manual_seed(
            f3b_seed(phase, unit, "endpoint-noise")
        ),
        time=torch.Generator(device="cpu").manual_seed(
            f3b_seed(phase, unit, "bridge-time")
        ),
        augmentation=torch.Generator(device="cpu").manual_seed(
            f3b_seed(phase, unit, "augmentation")
        ),
    )


def _groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int, max_period: float = 10_000.0) -> None:
        super().__init__()
        self.dimension = int(dimension)
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half) / max(half - 1, 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)

    def forward(self, time_value: torch.Tensor) -> torch.Tensor:
        if time_value.ndim == 2 and time_value.shape[1] == 1:
            time_value = time_value[:, 0]
        if time_value.ndim != 1:
            raise ValueError("time input must have shape [batch] or [batch,1]")
        # Scaling [0,1] to diffusion-style [0,1000] avoids an almost constant
        # low-frequency embedding while retaining the declared physical time.
        angles = time_value.float()[:, None] * 1_000.0 * self.frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


class TimeResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, time_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(
        self, images: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(images)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return self.skip(images) + hidden


class SpatialAttention(nn.Module):
    """Low-resolution self-attention; never instantiated at 32×32."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = int(channels)
        self.heads = max(1, channels // 64)
        while channels % self.heads:
            self.heads -= 1
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, 1)
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = images.shape
        tokens = height * width
        head_dim = channels // self.heads
        qkv = self.qkv(self.norm(images)).reshape(
            batch, 3, self.heads, head_dim, tokens
        )
        query, key, value = qkv.unbind(dim=1)
        weights = torch.softmax(
            torch.einsum("bhdt,bhds->bhts", query, key) / math.sqrt(head_dim), dim=-1
        )
        attended = torch.einsum("bhts,bhds->bhdt", weights, value).reshape(
            batch, channels, height, width
        )
        return images + self.projection(attended)


class ConditionedLevel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        time_dim: int,
        dropout: float,
        attention: bool,
    ) -> None:
        super().__init__()
        residuals = []
        attentions = []
        current = in_channels
        for _ in range(blocks):
            residuals.append(
                TimeResidualBlock(current, out_channels, time_dim, dropout)
            )
            attentions.append(
                SpatialAttention(out_channels) if attention else nn.Identity()
            )
            current = out_channels
        self.residuals = nn.ModuleList(residuals)
        self.attentions = nn.ModuleList(attentions)

    def forward(
        self, images: torch.Tensor, time_embedding: torch.Tensor
    ) -> torch.Tensor:
        hidden = images
        for residual, attention in zip(self.residuals, self.attentions):
            hidden = attention(residual(hidden, time_embedding))
        return hidden


class TimeConditionedUNet(nn.Module):
    """Image + scalar time -> image velocity, with spatial skip connections."""

    def __init__(self, config: F3BModelConfig, seed: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        widths = [config.base_channels * value for value in config.channel_multipliers]
        self.time_fourier = SinusoidalTimeEmbedding(config.time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim),
        )
        self.stem = nn.Conv2d(config.channels, widths[0], 3, padding=1)

        encoder = []
        downsamples = []
        current = widths[0]
        resolution = config.image_size
        for index, width in enumerate(widths):
            encoder.append(
                ConditionedLevel(
                    current,
                    width,
                    config.residual_blocks,
                    config.time_embedding_dim,
                    config.dropout,
                    resolution in config.attention_resolutions,
                )
            )
            current = width
            if index + 1 < len(widths):
                downsamples.append(
                    nn.Conv2d(width, widths[index + 1], 3, stride=2, padding=1)
                )
                current = widths[index + 1]
                resolution //= 2
        self.encoder = nn.ModuleList(encoder)
        self.downsamples = nn.ModuleList(downsamples)

        self.middle1 = TimeResidualBlock(
            current, current, config.time_embedding_dim, config.dropout
        )
        self.middle_attention = (
            SpatialAttention(current)
            if resolution in config.attention_resolutions
            else nn.Identity()
        )
        self.middle2 = TimeResidualBlock(
            current, current, config.time_embedding_dim, config.dropout
        )

        upsamples = []
        decoder = []
        for index in range(len(widths) - 2, -1, -1):
            width = widths[index]
            upsamples.append(nn.Conv2d(current, width, 3, padding=1))
            resolution *= 2
            decoder.append(
                ConditionedLevel(
                    2 * width,
                    width,
                    config.residual_blocks,
                    config.time_embedding_dim,
                    config.dropout,
                    resolution in config.attention_resolutions,
                )
            )
            current = width
        self.upsamples = nn.ModuleList(upsamples)
        self.decoder = nn.ModuleList(decoder)
        self.head_norm = nn.GroupNorm(_groups(current), current)
        self.head = nn.Conv2d(current, config.channels, 3, padding=1)
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) % (2**63 - 1))
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    fan_in = max(1, module.weight[0].numel())
                    bound = fan_in**-0.5
                    module.weight.copy_(
                        (
                            torch.rand(module.weight.shape, generator=generator) * 2.0
                            - 1.0
                        )
                        * bound
                    )
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, images: torch.Tensor, time_value: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.config.channels:
            raise ValueError("images must have shape [batch, channels, H, W]")
        if images.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("image resolution differs from model config")
        if len(time_value) != len(images):
            raise ValueError("one time value is required per image")
        embedding = self.time_mlp(self.time_fourier(time_value))
        hidden = self.stem(images)
        skips = []
        for index, level in enumerate(self.encoder):
            hidden = level(hidden, embedding)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        hidden = self.middle1(hidden, embedding)
        hidden = self.middle_attention(hidden)
        hidden = self.middle2(hidden, embedding)
        for offset, (upsample, level) in enumerate(zip(self.upsamples, self.decoder)):
            hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
            hidden = upsample(hidden)
            skip = skips[-2 - offset]
            if hidden.shape[-2:] != skip.shape[-2:]:
                raise RuntimeError("internal U-Net skip resolution mismatch")
            hidden = level(torch.cat((hidden, skip), dim=1), embedding)
        return self.head(F.silu(self.head_norm(hidden)))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def independent_bridge_batch(
    pool: torch.Tensor,
    batch: int,
    streams: BridgeStreams,
    device: torch.device | str,
    horizontal_flip: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw endpoints and return ``(Xt, target, X0, X1, t)``."""
    if batch <= 0:
        raise ValueError("batch must be positive")
    indices = streams.data.integers(0, len(pool), size=batch)
    endpoint = pool[torch.as_tensor(indices)].clone()
    if horizontal_flip:
        flip = torch.rand(batch, generator=streams.augmentation) < 0.5
        endpoint[flip] = torch.flip(endpoint[flip], dims=(-1,))
    noise = torch.randn(endpoint.shape, generator=streams.noise, dtype=endpoint.dtype)
    time_value = torch.rand(batch, generator=streams.time, dtype=endpoint.dtype)
    mixed = (1.0 - time_value[:, None, None, None]) * noise + time_value[
        :, None, None, None
    ] * endpoint
    target = endpoint - noise
    return (
        mixed.to(device),
        target.to(device),
        noise.to(device),
        endpoint.to(device),
        time_value.to(device),
    )


def oracle_endpoint(
    noise: torch.Tensor, endpoint: torch.Tensor, nfe: int
) -> torch.Tensor:
    """Euler with the fixed pair's true constant velocity (sign sanity)."""
    if nfe <= 0:
        raise ValueError("nfe must be positive")
    state = noise.clone()
    velocity = endpoint - noise
    for _ in range(nfe):
        state = state + velocity / nfe
    return state


def euler_integrate(model: nn.Module, initial: torch.Tensor, nfe: int) -> torch.Tensor:
    """Integrate from t=0 to t=1 without clipping the evolving state."""
    if nfe <= 0:
        raise ValueError("nfe must be positive")
    state = initial
    step = 1.0 / nfe
    for index in range(nfe):
        time_value = torch.full(
            (len(state),), index * step, device=state.device, dtype=state.dtype
        )
        state = state + step * model(state, time_value)
    return state


def sample_model(
    model: nn.Module,
    count: int,
    model_config: F3BModelConfig,
    nfe: int,
    seed: int,
    device: torch.device | str,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("count must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn(
        count,
        model_config.channels,
        model_config.image_size,
        model_config.image_size,
        generator=generator,
    ).to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        result = euler_integrate(model, initial, nfe)
    model.train(was_training)
    return result.detach()


class EMAState:
    """Explicit EMA whose use can be audited and temporarily applied."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[name].copy_(value)

    @contextlib.contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        original = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(original, strict=True)


@dataclass
class BridgeTrainResult:
    model: TimeConditionedUNet
    ema: EMAState
    history: list[dict]
    wall_seconds: float
    peak_memory_bytes: int | None
    examples_seen: int
    optimizer_updates: int


def train_bridge(
    pool: torch.Tensor,
    model_config: F3BModelConfig,
    train_config: F3BTrainConfig,
    phase: str,
    unit: int,
    device: torch.device | str,
    checkpoint: Callable[[int, TimeConditionedUNet, dict], None] | None = None,
) -> BridgeTrainResult:
    """Train one independently seeded bridge unit.

    A callback sees EMA parameters at declared development checkpoints. It may
    evaluate but cannot modify the training stream or optimizer state.
    """
    model_config.validate()
    train_config.validate()
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
    streams = bridge_streams(phase, unit)
    history: list[dict] = []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, train_config.steps + 1):
        mixed, target, _, _, time_value = independent_bridge_batch(
            pool, train_config.batch, streams, device, train_config.horizontal_flip
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(mixed, time_value)
        loss = F.mse_loss(prediction, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite F3B loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), train_config.gradient_clip
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError(f"non-finite F3B gradient at step {step}")
        optimizer.step()
        ema.update(model)

        record = {
            "step": step,
            "loss": float(loss.detach()),
            "gradient_norm_before_clip": float(gradient_norm),
            "examples_seen": step * train_config.batch,
            "wall_seconds": time.time() - started,
        }
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
    return BridgeTrainResult(
        model=model,
        ema=ema,
        history=history,
        wall_seconds=time.time() - started,
        peak_memory_bytes=peak,
        examples_seen=train_config.steps * train_config.batch,
        optimizer_updates=train_config.steps,
    )


def range_report(images: torch.Tensor) -> dict:
    outside = (images < -1.0) | (images > 1.0)
    return {
        "outside_fraction": float(outside.float().mean()),
        "sample_outside_fraction": float(outside.flatten(1).any(dim=1).float().mean()),
        "minimum": float(images.min()),
        "maximum": float(images.max()),
        "rms": float(images.square().mean().sqrt()),
    }


def adjudicate_b0(
    rows: list[dict],
    matched_real_recall: float,
    recall_gate: float = RECALL_GATE,
    control_floor: float = METRIC_CONTROL_FLOOR,
) -> dict:
    """Narrow 2-of-3 reachability decision; never claims general generation."""
    units = {}
    for row in rows:
        unit = str(row["unit"])
        recall_pass = bool(float(row["metrics"]["recall"]) > recall_gate)
        veto_pass = bool(row["veto"]["passes"])
        units[unit] = {
            "recall": float(row["metrics"]["recall"]),
            "recall_passes": recall_pass,
            "veto_passes": veto_pass,
            "unit_passes": bool(recall_pass and veto_pass),
        }
    if len(units) != 3:
        raise ValueError("F3B confirmation requires exactly three units")
    control_valid = bool(matched_real_recall > control_floor)
    passes = sum(item["unit_passes"] for item in units.values())
    decision = "VOID" if not control_valid else ("PASS" if passes >= 2 else "FAIL")
    reading = {
        "PASS": (
            "PASS: the frozen bridge has detected fresh-sample coverage "
            "above the calibrated null in at least two units"
        ),
        "FAIL": (
            "FAIL: this frozen bridge configuration did not establish "
            "detected coverage; flow matching in general is not rejected"
        ),
        "VOID": "VOID: the matched real-vs-real metric control was invalid",
    }[decision]
    return {
        "decision": decision,
        "units_passing": int(passes),
        "units": units,
        "recall_gate": float(recall_gate),
        "matched_real_recall": float(matched_real_recall),
        "metric_control_floor": float(control_floor),
        "metric_control_valid": control_valid,
        "reading": reading,
    }
