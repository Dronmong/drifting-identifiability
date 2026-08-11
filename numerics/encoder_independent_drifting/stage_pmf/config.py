"""Frozen-shaped, audit-pending configuration for local pixel MeanFlow S3."""

from __future__ import annotations

from dataclasses import dataclass

INITIAL_UNITS = (700, 701)
AUTOMOBILE_LABEL = 1
T_MIN_DIVISOR = 0.05


@dataclass(frozen=True)
class PMFModelConfig:
    image_size: int = 32
    channels: int = 3
    patch_size: int = 4
    width: int = 384
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    time_embedding_dim: int = 256

    def validate(self) -> None:
        if self.image_size <= 0 or self.patch_size <= 0:
            raise ValueError("image and patch sizes must be positive")
        if self.image_size % self.patch_size:
            raise ValueError("patch size must divide image size")
        if self.channels <= 0 or self.width <= 0:
            raise ValueError("channel count and width must be positive")
        if self.depth < 4 or self.depth % 2:
            raise ValueError("depth must be even and at least four")
        if self.heads <= 0 or self.width % self.heads:
            raise ValueError("heads must divide width")
        if self.mlp_ratio <= 1:
            raise ValueError("mlp ratio must exceed one")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if self.time_embedding_dim < 4 or self.time_embedding_dim % 2:
            raise ValueError("time embedding dimension must be even and >= 4")


@dataclass(frozen=True)
class PMFObjectiveConfig:
    logit_mean: float = 0.8
    logit_std: float = 0.8
    diagonal_fraction: float = 0.5
    denominator_floor: float = T_MIN_DIVISOR
    adaptive_power: float = 1.0
    adaptive_epsilon: float = 0.01

    def validate(self) -> None:
        if self.logit_std <= 0:
            raise ValueError("logit standard deviation must be positive")
        if not 0 <= self.diagonal_fraction <= 1:
            raise ValueError("diagonal fraction must lie in [0,1]")
        if not 0 < self.denominator_floor <= 1:
            raise ValueError("denominator floor must lie in (0,1]")
        if self.adaptive_power < 0 or self.adaptive_epsilon <= 0:
            raise ValueError("invalid adaptive weighting")


@dataclass(frozen=True)
class PMFTrainConfig:
    updates: int = 60_000
    micro_batch: int = 16
    accumulation_steps: int = 1
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    ema_decay: float = 0.9999
    horizontal_flip: bool = True
    log_every: int = 100
    checkpoint_updates: tuple[int, ...] = (2_000, 10_000, 30_000, 60_000)

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.accumulation_steps

    def validate(self) -> None:
        if self.updates <= 0 or self.micro_batch <= 0 or self.accumulation_steps <= 0:
            raise ValueError("training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer scale")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("optimizer betas must lie in [0,1)")
        if self.gradient_clip <= 0 or not 0 <= self.ema_decay < 1:
            raise ValueError("invalid clipping or EMA value")
        if self.log_every <= 0:
            raise ValueError("log cadence must be positive")
        if tuple(sorted(set(self.checkpoint_updates))) != self.checkpoint_updates:
            raise ValueError("checkpoint updates must be sorted and unique")
        if any(step <= 0 or step > self.updates for step in self.checkpoint_updates):
            raise ValueError("checkpoint update outside training budget")


@dataclass(frozen=True)
class PMFEvalConfig:
    generated_samples: int = 1_000
    grid_samples: int = 64
    inference_batch: int = 64
    fixed_noise_seed_label: str = "sealed-one-step-grid"

    def validate(self) -> None:
        if self.generated_samples <= 1 or self.grid_samples <= 0:
            raise ValueError("evaluation sample counts must be positive")
        if self.grid_samples > self.generated_samples or self.inference_batch <= 0:
            raise ValueError("invalid grid or inference batch")


@dataclass(frozen=True)
class PMFProfile:
    name: str
    purpose: str
    model: PMFModelConfig
    objective: PMFObjectiveConfig
    train: PMFTrainConfig
    evaluation: PMFEvalConfig

    def validate(self) -> None:
        self.model.validate()
        self.objective.validate()
        self.train.validate()
        self.evaluation.validate()


def profile(name: str) -> PMFProfile:
    if name == "smoke":
        result = PMFProfile(
            name="smoke",
            purpose="mechanical correctness only; never a quality result",
            model=PMFModelConfig(
                image_size=8,
                patch_size=2,
                width=64,
                depth=4,
                heads=4,
                mlp_ratio=2.0,
                time_embedding_dim=32,
            ),
            objective=PMFObjectiveConfig(),
            train=PMFTrainConfig(
                updates=2,
                micro_batch=2,
                accumulation_steps=1,
                learning_rate=1e-3,
                ema_decay=0.9,
                log_every=1,
                checkpoint_updates=(1, 2),
            ),
            evaluation=PMFEvalConfig(
                generated_samples=8, grid_samples=8, inference_batch=4
            ),
        )
    elif name == "local_s3":
        result = PMFProfile(
            name="local_s3",
            purpose=(
                "audit-pending two-unit one-class pixel MeanFlow foundation; "
                "not authorized for launch"
            ),
            model=PMFModelConfig(),
            objective=PMFObjectiveConfig(),
            train=PMFTrainConfig(),
            evaluation=PMFEvalConfig(),
        )
    else:
        raise ValueError(f"unknown pMF profile {name!r}; expected smoke or local_s3")
    result.validate()
    return result
