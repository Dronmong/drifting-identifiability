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
# set gives 410 epochs and makes a 50k-sample evaluation possible.  CAP-EMF-1's
# historical in-repo feature metric is not a published-comparable CleanFID;
# stage_cap2.standard_metrics provides that separately.
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
    # 384 gives 37.7M parameters, essentially DDPM's 35.7M and inside the
    # 35-56M band every strong CIFAR-10 model sits in.  Width 512 measured 1.43x
    # slower for no capacity argument at 32x32: at a fixed GPU budget the extra
    # parameters cost 30% of the images seen, and images seen is what this run
    # is short of.
    width: int = 384
    depth: int = 12
    heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    time_embedding_dim: int = 256
    # CAP-EMF-1 used 1000.0.  This is now explicit because its production
    # finite-difference step of 0.01 advances the highest sinusoid by ten
    # radians, so a successor must audit the pair (scale, delta) rather than
    # treating either value as an isolated knob.
    scalar_embedding_scale: float = 1_000.0
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
        if self.dropout != 0.0:
            # DDPM uses 0.1 on CIFAR-10 and EDM 0.13, so this looks like an
            # obvious win and is not one: dropout is **incompatible with the
            # EMF local difference**.  The quotient is
            # ``(future - current) / delta`` with delta = 0.01, and the three
            # evaluations would draw independent masks, so the difference would
            # carry mask noise rather than a derivative -- amplified a
            # hundredfold by 1/delta.  Enabling it requires sharing one mask
            # across all three evaluations, which is not implemented.
            raise ValueError(
                "dropout is incompatible with the EMF local difference; "
                "independent masks across the three evaluations inject noise "
                "amplified 100x by 1/delta. Share a mask first."
            )
        if self.time_embedding_dim < 4 or self.time_embedding_dim % 2:
            raise ValueError("time embedding dimension must be even and >= 4")
        if self.scalar_embedding_scale <= 0:
            raise ValueError("scalar embedding scale must be positive")
        if self.condition_dim < 4:
            raise ValueError("conditioning width must be at least four")
        if self.refiner_width <= 0 or self.refiner_depth < 1:
            raise ValueError("the local pixel refiner needs positive shape")
        # A tap index past the end of its stack is silently dropped by the
        # trunk -- the forward still works and the level simply never appears,
        # which downstream code would only discover as a missing key. Catch it
        # where the shape is declared instead.
        half = self.depth // 2
        for label, stack, index in FEATURE_LEVELS:
            if index >= half:
                raise ValueError(
                    f"feature level {label!r} taps {stack} block {index}, but "
                    f"depth {self.depth} gives only {half} blocks per stack"
                )


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
    # The EMF paper uses 0.02 for singular divisions. Under this repository's
    # data-at-zero clock, t is a divisor in the state update and loss weight,
    # while r is the divisor in the correction coefficient. The multiplicative
    # t in that coefficient is a numerator and must not be clamped.
    emf_denominator_floor: float = 0.02
    # Separate floor for the *coefficient* denominator only, i.e. the r in
    # (t-r-delta)*t/r.  ``None`` resolves to ``emf_denominator_floor``, which
    # preserves CAP-EMF-1 exactly.
    #
    # This exists because one constant was serving three unrelated roles: the
    # Euler state divisor (clamps t), the 1/t^2 loss weight (clamps t), and the
    # coefficient denominator (clamps r).  Only the last one needs raising.
    # Measured over 2,000,000 ordered-uniform draws, raising the shared
    # constant to 0.10 would also have quartered the loss weight on the ~1% of
    # rows with t < 0.10 -- an unintended change to the objective.
    #
    # r -> 0 is simultaneously the inference corner the sampler must train and
    # the singularity of Equation 18, so the two cannot be separated in the
    # time law; they separate here instead.
    coefficient_denominator_floor: float | None = None
    # Historical CAP-EMF-1 default.  Successor arms use one of the two ordered
    # iid modes below.  Keeping the legacy value as the default preserves the
    # original run's exact scientific configuration as a matched control.
    sampler_mode: str = "cap_conditional_logitnormal"
    # CAP-EMF-1 clamped the sampled endpoint itself to 0.01.  Ordered successor
    # arms set this to zero and clamp only the numerical denominator.
    sampled_r_floor: float = 0.01
    # CAP-EMF-1 used an independent Bernoulli mask after constructing its
    # conditional endpoint.  CAP2 uses the released MeanFlow convention:
    # choose an exact rounded number of diagonal rows from a separate RNG
    # stream and set both endpoints to the *first unsorted draw*.  The latter
    # matters for ordered samplers -- setting r=t after sorting would put
    # diagonal rows at max(draw1, draw2) and bias half the batch toward noise.
    diagonal_sampling: str = "legacy_bernoulli"
    # How stopped evaluations are formed.  ``legacy_sparse`` is the historical
    # active-row gather.  ``dense`` keeps current/future batch shapes matched;
    # ``fp32_dense`` additionally disables TF32 for the stopped path and
    # evaluates a separate stopped current so subtraction uses one precision.
    stopped_evaluation: str = "legacy_sparse"

    @property
    def resolved_coefficient_floor(self) -> float:
        """The r-clamp actually applied in the correction coefficient."""
        if self.coefficient_denominator_floor is None:
            return self.emf_denominator_floor
        return self.coefficient_denominator_floor

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
        if self.coefficient_denominator_floor is not None and not (
            0 < self.coefficient_denominator_floor <= 1
        ):
            raise ValueError("coefficient denominator floor must lie in (0, 1]")
        allowed_samplers = {
            "cap_conditional_logitnormal",
            "ordered_logitnormal",
            "ordered_uniform",
        }
        if self.sampler_mode not in allowed_samplers:
            raise ValueError(f"unknown time sampler {self.sampler_mode!r}")
        if not 0 <= self.sampled_r_floor < 1:
            raise ValueError("sampled r floor must lie in [0, 1)")
        if self.diagonal_sampling not in {
            "legacy_bernoulli",
            "fixed_count_first_draw",
        }:
            raise ValueError(f"unknown diagonal sampler {self.diagonal_sampling!r}")
        if self.stopped_evaluation not in {
            "legacy_sparse",
            "dense",
            "fp32_dense",
        }:
            raise ValueError(
                f"unknown stopped evaluation mode {self.stopped_evaluation!r}"
            )


@dataclass(frozen=True)
class CAPTrainConfig:
    # 750k x 64 = 48M examples = 960 epochs.  DDPM used 800k x 128 = 102M
    # images (2048 epochs) on this dataset, so this is roughly half its budget
    # at its parameter count -- the closest a 48 h / $25 envelope gets to a
    # known-good CIFAR-10 configuration.  Images seen, not parameters, is what
    # this run is short of.
    updates: int = 750_000
    micro_batch: int = 16
    accumulation_steps: int = 4
    learning_rate: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.0
    gradient_clip: float = 10.0
    ema_decay: float = 0.9999
    horizontal_flip: bool = True
    # Linear warmup. S3R clipped on 4.99% of updates, essentially on its 5%
    # limit, and the first updates are where a zero-initialized output path is
    # moving fastest. Standard for this dataset: DDPM warms up over 5k.
    warmup_updates: int = 5_000
    # Raw parameter snapshots for post-hoc EMA (Karras et al.). EMA 0.9999 over
    # 750k updates is a ~10k-update window, 1.3% of training, and the right
    # horizon is not knowable in advance. Storing snapshots lets the profile be
    # synthesized afterwards without retraining. 30 snapshots x 151 MB = 4.5 GB.
    # Post-hoc variants are SECONDARY: the primary result is the declared
    # 0.9999 EMA, so this cannot become checkpoint selection on a metric.
    snapshot_every: int = 25_000
    log_every: int = 500
    # Every 50k. The primary result is still the final checkpoint; the density
    # is insurance. Under the budget-stop rule the last completed checkpoint
    # becomes the result, and a sparse ladder (100k/200k/400k/600k/750k) would
    # throw away up to 200k updates of finished work on a shortfall. Fifteen
    # checkpoints cost ~4.5 GB and about eight minutes of audit time across a
    # forty-hour run.
    checkpoint_updates: tuple[int, ...] = tuple(range(50_000, 750_001, 50_000))
    health_every: int = 2_000
    health_samples: int = 512
    audit_samples: int = 2_048
    # B2.5 had no within-unit recovery, so an interrupted unit lost every
    # completed hour and then blocked its own restart.  A rented run must not
    # repeat that.
    #
    # 5,000 rather than 1,000, from measurement on the 4090: the recovery
    # payload is 576 MB (model + Adam moments + EMA + four RNG streams +
    # history) and takes ~56 s to serialize, so a 1,000-update cadence cost
    # ~25% of throughput -- 0.2295 s/update measured against 0.173 benchmarked,
    # which would have overrun both the wall clock and the budget. At 5,000 the
    # overhead is ~0.011 s/update and a crash still loses under half an hour.
    recovery_every: int = 5_000

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.accumulation_steps

    @property
    def ema_half_life(self) -> float:
        import math

        return math.log(0.5) / math.log(self.ema_decay)

    def ema_mature_at(self, half_lives: float = 5.0) -> int:
        return round(half_lives * self.ema_half_life)

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
        if self.warmup_updates < 0 or self.warmup_updates > self.updates:
            raise ValueError("warmup must be nonnegative and no longer than the run")
        if self.snapshot_every <= 0:
            raise ValueError("snapshot cadence must be positive")
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
    # Absolute real-data-calibrated lower rank.  This is distinct from the
    # trajectory-retention rule below: a model that stays at rank ratio .5
    # must not pass merely because it never deteriorated from .5.
    minimum_effective_rank_ratio: float = 0.80
    rank_retention: float = 0.80
    minimum_haar_LL_ratio: float = 0.60
    minimum_haar_LH_ratio: float = 0.60
    minimum_haar_HL_ratio: float = 0.60
    minimum_haar_HH_ratio: float = 0.50
    maximum_clip_fraction: float = 0.05
    clip_window_updates: int = 20_000
    # CAP-EMF-1's original gate only had lower bounds, so a cloud with 8.4x
    # target rank and 6.4x target HH energy passed.  Successor gates are
    # explicitly two-sided.  These broad defaults are mechanical safety
    # limits; CAP2 calibrates tighter intervals from independent real subsets.
    maximum_second_moment_ratio: float = 1.25
    maximum_centered_variance_ratio: float = 1.25
    maximum_effective_rank_ratio: float = 1.50
    maximum_haar_LL_ratio: float = 1.75
    maximum_haar_LH_ratio: float = 1.75
    maximum_haar_HL_ratio: float = 1.75
    maximum_haar_HH_ratio: float = 1.75
    maximum_saturation_fraction: float = 0.02

    def validate(self) -> None:
        values = (
            self.second_moment_ratio,
            self.centered_variance_ratio,
            self.minimum_effective_rank_ratio,
            self.rank_retention,
            self.minimum_haar_LL_ratio,
            self.minimum_haar_LH_ratio,
            self.minimum_haar_HL_ratio,
            self.minimum_haar_HH_ratio,
        )
        if any(not 0 < value <= 1 for value in values):
            raise ValueError("capability thresholds must lie in (0,1]")
        if not 0 < self.maximum_clip_fraction < 1:
            raise ValueError("clip fraction limit must lie in (0,1)")
        if self.clip_window_updates <= 0:
            raise ValueError("clip window must be positive")
        upper_values = (
            self.maximum_second_moment_ratio,
            self.maximum_centered_variance_ratio,
            self.maximum_effective_rank_ratio,
            self.maximum_haar_LL_ratio,
            self.maximum_haar_LH_ratio,
            self.maximum_haar_HL_ratio,
            self.maximum_haar_HH_ratio,
        )
        if any(value < 1 for value in upper_values):
            raise ValueError("two-sided upper capability limits must be >= 1")
        lower_upper = (
            (self.second_moment_ratio, self.maximum_second_moment_ratio),
            (self.centered_variance_ratio, self.maximum_centered_variance_ratio),
            (
                self.minimum_effective_rank_ratio,
                self.maximum_effective_rank_ratio,
            ),
            (self.minimum_haar_LL_ratio, self.maximum_haar_LL_ratio),
            (self.minimum_haar_LH_ratio, self.maximum_haar_LH_ratio),
            (self.minimum_haar_HL_ratio, self.maximum_haar_HL_ratio),
            (self.minimum_haar_HH_ratio, self.maximum_haar_HH_ratio),
        )
        if any(lower > upper for lower, upper in lower_upper):
            raise ValueError("a capability lower bound exceeds its upper bound")
        if not 0 <= self.maximum_saturation_fraction < 1:
            raise ValueError("saturation limit must lie in [0,1)")


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
                # Depth 12, not 4: the declared feature taps reach block 5 of
                # each stack, and a shallower smoke trunk would simply not have
                # them -- so the smoke profile would exercise a different
                # architecture from the one being audited.
                depth=12,
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
                # Short enough to exercise both branches of the schedule and
                # the snapshot path within four updates.
                warmup_updates=2,
                snapshot_every=2,
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
        if result.train.updates != 750_000:
            raise RuntimeError("CAP-EMF-1 horizon drifted from 750,000 updates")
        if result.model.width != 384:
            raise RuntimeError("CAP-EMF-1 width drifted from 384")
        if result.model.tokens != 256:
            raise RuntimeError("CAP-EMF-1 token count drifted from 256")
    return result


def enable_tf32() -> dict:
    """Enable TF32 for ordinary model work and record the exact boundary.

    CAP-EMF-1 incorrectly claimed FP32 output storage made its stopped quotient
    immune to TF32 input rounding.  It does not.  Successor objectives may
    disable TF32 inside a matched stopped path while retaining it for the
    graded prediction; the selected objective mode records that distinction.
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return {
        "matmul_tf32": True,
        "cudnn_tf32": True,
        "storage_dtype": "fp32",
        "rationale": (
            "TF32 is enabled globally for throughput, but its safety is not "
            "assumed for a finite-difference subtraction; fp32_dense objective "
            "mode disables it around all stopped evaluations"
        ),
    }


def examples_seen(train: CAPTrainConfig) -> int:
    """Total training examples, the quantity a matched drifting arm must equal.

    EMF sees 64 examples per update; a drifting field needs a cloud of order
    256.  Matching *updates* would hand drifting four times the data exposure,
    so the comparison is matched on this number instead.
    """
    return train.updates * train.effective_batch
