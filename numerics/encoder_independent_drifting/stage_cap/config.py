"""Frozen CAP-EMF-1 configuration.

Every value here is frozen before the cloud benchmark.  The benchmark may
change only the microbatch/accumulation split, fused-kernel settings, and the
selected GPU — never a scientific knob.
"""

from __future__ import annotations

from dataclasses import dataclass

# Unconditional CIFAR-10, all ten classes.  The earlier single-class draft used
# 5,000 automobile images, which at this horizon is 4,096 epochs: memorization
# becomes the likely route to coherent-looking samples, and FID against a
# 1,000-image reference is small-sample biased beyond rescue.  The full training
# set gives 410 epochs and makes the standard 50k-sample FID protocol available,
# so the headline number is comparable to published CIFAR-10 results.
TRAIN_POOL_SIZE = 50_000
SEALED_TEST_POOL_SIZE = 10_000

# Protocol section 5.4.  The port must report the exact parameter count and it
# is frozen at that value; this is the ceiling the count must clear.
PARAMETER_CEILING = 65_000_000


@dataclass(frozen=True)
class CAPModelConfig:
    """Patch-2 U-ViT with AdaLN-Zero conditioning and a local pixel head."""

    image_size: int = 32
    channels: int = 3
    patch_size: int = 2
    width: int = 512
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    time_embedding_dim: int = 256
    # Protocol 5.4: per-block Linear(width, 6*width) modulation would cost
    # 18.9M parameters and push the model past the 65M ceiling.  A shared
    # conditioning trunk projected to this width, then per-block
    # Linear(condition_dim, 6*width), removes most of that without touching
    # model capacity.
    condition_dim: int = 256
    refiner_width: int = 64
    refiner_depth: int = 2

    @property
    def tokens(self) -> int:
        side = self.image_size // self.patch_size
        return side * side

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
        if self.condition_dim < 4:
            raise ValueError("conditioning width must be at least four")
        if self.refiner_width <= 0 or self.refiner_depth < 1:
            raise ValueError("the local pixel refiner needs positive shape")


# Protocol section 10.  Feature levels are declared here, not in ASFD, so the
# trunk's source hash already covers them and no retrofit is needed later.
FEATURE_LEVELS: tuple[tuple[str, str, int], ...] = (
    ("enc_mid", "encoder", 2),
    ("enc_final", "encoder", 5),
    ("dec_mid", "decoder", 2),
    ("dec_final", "decoder", 5),
)


@dataclass(frozen=True)
class CAPObjectiveConfig:
    """Direct-`x` Euler Mean Flow, JVP-free.  Inherited from the audited S3R."""

    logit_mean: float = 0.8
    logit_std: float = 0.8
    diagonal_fraction: float = 0.5
    adaptive_power: float = 1.0
    adaptive_epsilon: float = 0.01
    emf_delta: float = 0.01
    # The EMF paper clamps its (1-t_paper) and (1-r_paper) denominators at
    # 0.02.  Under this repository's data-at-zero clock those are t and r.
    emf_denominator_floor: float = 0.02

    def validate(self) -> None:
        if self.logit_std <= 0:
            raise ValueError("logit standard deviation must be positive")
        if not 0 <= self.diagonal_fraction <= 1:
            raise ValueError("diagonal fraction must lie in [0,1]")
        if self.adaptive_power < 0 or self.adaptive_epsilon <= 0:
            raise ValueError("invalid adaptive weighting")
        if not 0 < self.emf_delta < 0.5:
            raise ValueError("EMF local step must lie in (0, 0.5)")
        if not 0 < self.emf_denominator_floor <= 1:
            raise ValueError("EMF denominator floor must lie in (0, 1]")


@dataclass(frozen=True)
class CAPTrainConfig:
    # 320k x 64 = 20.48M examples = 410 epochs over CIFAR-10.  160k was chosen
    # by analogy to nothing in particular; EMF's own pixel experiment used
    # 600k, and coherent samples are the point of the run, so the budget is set
    # from the outset rather than extended mid-flight.
    updates: int = 320_000
    micro_batch: int = 16
    accumulation_steps: int = 4
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    gradient_clip: float = 10.0
    ema_decay: float = 0.9999
    horizontal_flip: bool = True
    log_every: int = 200
    checkpoint_updates: tuple[int, ...] = (40_000, 80_000, 160_000, 240_000, 320_000)
    health_every: int = 2_000
    health_samples: int = 512
    audit_samples: int = 2_048
    # B2.5 had no within-unit recovery, so an interrupted unit lost every
    # completed hour and then blocked its own restart.  A 160k-update rented
    # run must not repeat that.
    recovery_every: int = 1_000

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.accumulation_steps

    @property
    def ema_half_life(self) -> float:
        import math

        return math.log(0.5) / math.log(self.ema_decay)

    def ema_mature_at(self, half_lives: float = 5.0) -> int:
        return int(round(half_lives * self.ema_half_life))

    def validate(self) -> None:
        if self.updates <= 0 or self.micro_batch <= 0 or self.accumulation_steps <= 0:
            raise ValueError("training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer scale")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("optimizer betas must lie in [0,1)")
        if self.gradient_clip <= 0 or not 0 <= self.ema_decay < 1:
            raise ValueError("invalid clipping or EMA value")
        if self.log_every <= 0 or self.health_every <= 0:
            raise ValueError("logging cadences must be positive")
        if self.health_samples < 64 or self.audit_samples < self.health_samples:
            raise ValueError("health cloud sizes are too small to decide anything")
        if self.recovery_every <= 0:
            raise ValueError("recovery cadence must be positive")
        if not self.checkpoint_updates:
            raise ValueError("at least one checkpoint step is required")
        if sorted(self.checkpoint_updates) != list(self.checkpoint_updates):
            raise ValueError("checkpoint steps must be increasing")
        if self.checkpoint_updates[-1] != self.updates:
            raise ValueError("the final checkpoint must be the primary step")


@dataclass(frozen=True)
class CAPGateConfig:
    """Protocol section 7.1 train-only capability thresholds."""

    second_moment_ratio: float = 0.80
    centered_variance_ratio: float = 0.80
    rank_retention: float = 0.80
    haar_hh_ratio: float = 0.50
    haar_detail_ratio: float = 0.60
    maximum_clip_fraction: float = 0.05
    clip_window_updates: int = 20_000

    def validate(self) -> None:
        values = (
            self.second_moment_ratio,
            self.centered_variance_ratio,
            self.rank_retention,
            self.haar_hh_ratio,
            self.haar_detail_ratio,
        )
        if any(not 0 < value <= 1 for value in values):
            raise ValueError("capability thresholds must lie in (0,1]")
        if not 0 < self.maximum_clip_fraction < 1:
            raise ValueError("clip fraction limit must lie in (0,1)")
        if self.clip_window_updates <= 0:
            raise ValueError("clip window must be positive")


@dataclass(frozen=True)
class CAPProfile:
    name: str
    purpose: str
    model: CAPModelConfig
    objective: CAPObjectiveConfig
    train: CAPTrainConfig
    gate: CAPGateConfig

    def validate(self) -> None:
        self.model.validate()
        self.objective.validate()
        self.train.validate()
        self.gate.validate()


def profile(name: str) -> CAPProfile:
    """``smoke`` is for mechanics only; ``capability`` is the frozen run card."""
    if name == "smoke":
        result = CAPProfile(
            name="smoke",
            purpose="mechanical CAP-EMF-1 checks only; never a quality result",
            model=CAPModelConfig(
                image_size=8,
                patch_size=2,
                width=64,
                depth=4,
                heads=4,
                mlp_ratio=2.0,
                time_embedding_dim=32,
                condition_dim=32,
                refiner_width=8,
                refiner_depth=1,
            ),
            objective=CAPObjectiveConfig(adaptive_power=0.0),
            train=CAPTrainConfig(
                updates=4,
                micro_batch=4,
                accumulation_steps=2,
                log_every=1,
                checkpoint_updates=(2, 4),
                health_every=1,
                health_samples=64,
                audit_samples=64,
                recovery_every=1,
                ema_decay=0.9,
            ),
            gate=CAPGateConfig(),
        )
    elif name == "capability":
        result = CAPProfile(
            name="capability",
            purpose=(
                "one-call raw-pixel EMF capability foundation on unconditional "
                "CIFAR-10; no correction, no test access during training"
            ),
            model=CAPModelConfig(),
            objective=CAPObjectiveConfig(),
            train=CAPTrainConfig(),
            gate=CAPGateConfig(),
        )
    else:
        raise ValueError(f"unknown CAP profile {name!r}")
    result.validate()
    if name == "capability":
        if result.train.effective_batch != 64:
            raise RuntimeError("CAP-EMF-1 effective batch drifted from 64")
        if result.train.updates != 320_000:
            raise RuntimeError("CAP-EMF-1 horizon drifted from 320,000 updates")
        if result.model.tokens != 256:
            raise RuntimeError("CAP-EMF-1 token count drifted from 256")
    return result


def examples_seen(train: CAPTrainConfig) -> int:
    """Total training examples, the quantity a matched drifting arm must equal.

    EMF sees 64 examples per update; a drifting field needs a cloud of order
    256.  Matching *updates* would hand drifting four times the data exposure,
    so the comparison is matched on this number instead.
    """
    return train.updates * train.effective_batch
