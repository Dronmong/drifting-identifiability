"""Configurations for the outcome-blind S3R mechanism screen."""

from __future__ import annotations

from dataclasses import dataclass

from ..stage_pmf.config import (
    PMFEvalConfig,
    PMFModelConfig,
    PMFObjectiveConfig,
    PMFTrainConfig,
)

S3R_ARMS = ("pmf", "alpha", "emf")


@dataclass(frozen=True)
class S3RModelConfig(PMFModelConfig):
    """The S3 inference trunk plus a genuinely deep training-only branch."""

    auxiliary_depth: int = 4
    condition_on_absolute_time: bool = True

    def validate(self) -> None:
        super().validate()
        if self.auxiliary_depth < 1:
            raise ValueError("the repaired auxiliary branch must contain a block")


@dataclass(frozen=True)
class S3RObjectiveConfig(PMFObjectiveConfig):
    alpha_schedule_start_fraction: float = 0.10
    alpha_schedule_end_fraction: float = 0.90
    alpha_temperature: float = 25.0
    alpha_floor: float = 0.005
    alpha_adaptive_epsilon: float = 0.001
    emf_delta: float = 0.01
    emf_denominator_floor: float = 0.02

    def validate(self) -> None:
        super().validate()
        if not (
            0
            <= self.alpha_schedule_start_fraction
            < self.alpha_schedule_end_fraction
            <= 1
        ):
            raise ValueError("invalid AlphaFlow transition interval")
        if self.alpha_temperature <= 0:
            raise ValueError("AlphaFlow temperature must be positive")
        if not 0 < self.alpha_floor < 0.5:
            raise ValueError("AlphaFlow floor must lie in (0, 0.5)")
        if self.alpha_adaptive_epsilon <= 0:
            raise ValueError("AlphaFlow adaptive epsilon must be positive")
        if not 0 < self.emf_delta < 0.5:
            raise ValueError("EMF local step must lie in (0, 0.5)")
        if not 0 < self.emf_denominator_floor <= 1:
            raise ValueError("EMF denominator floor must lie in (0, 1]")


@dataclass(frozen=True)
class S3RTrainConfig(PMFTrainConfig):
    updates: int = 12_500
    accumulation_steps: int = 4
    gradient_clip: float = 10.0
    log_every: int = 50
    checkpoint_updates: tuple[int, ...] = (2_000, 6_250, 12_500)
    health_every: int = 500
    health_samples: int = 64
    gradient_cosine_every: int = 2_000

    def validate(self) -> None:
        super().validate()
        if self.health_every <= 0 or self.health_samples < 4:
            raise ValueError("invalid endpoint-health cadence")
        if self.gradient_cosine_every <= 0:
            raise ValueError("invalid gradient-cosine cadence")


@dataclass(frozen=True)
class S3RProfile:
    name: str
    purpose: str
    model: S3RModelConfig
    objective: S3RObjectiveConfig
    train: S3RTrainConfig
    evaluation: PMFEvalConfig

    def validate(self) -> None:
        self.model.validate()
        self.objective.validate()
        self.train.validate()
        self.evaluation.validate()


def profile(name: str) -> S3RProfile:
    if name == "smoke":
        result = S3RProfile(
            name="smoke",
            purpose="mechanical S3R checks only",
            model=S3RModelConfig(
                image_size=8,
                patch_size=2,
                width=64,
                depth=4,
                heads=4,
                mlp_ratio=2.0,
                time_embedding_dim=32,
                auxiliary_depth=1,
            ),
            objective=S3RObjectiveConfig(adaptive_power=0.0),
            train=S3RTrainConfig(
                updates=2,
                micro_batch=16,
                accumulation_steps=4,
                learning_rate=1e-3,
                ema_decay=0.9,
                log_every=1,
                checkpoint_updates=(1, 2),
                health_every=1,
                health_samples=8,
                gradient_cosine_every=1,
            ),
            evaluation=PMFEvalConfig(
                generated_samples=8, grid_samples=8, inference_batch=4
            ),
        )
    elif name == "developmental":
        result = S3RProfile(
            name="developmental",
            purpose=(
                "train-only matched pMF/AlphaFlow/EMF screen; no test-set access "
                "and no launch authorization"
            ),
            model=S3RModelConfig(),
            objective=S3RObjectiveConfig(),
            train=S3RTrainConfig(),
            evaluation=PMFEvalConfig(),
        )
    else:
        raise ValueError(f"unknown S3R profile {name!r}")
    result.validate()
    if name == "developmental":
        if result.train.effective_batch != 64:
            raise RuntimeError("developmental S3R effective batch drifted from 64")
        if result.train.updates * result.train.effective_batch != 800_000:
            raise RuntimeError("developmental S3R budget drifted from 800,000 examples")
    return result
