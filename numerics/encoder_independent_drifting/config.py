"""Frozen configuration objects and deterministic seed derivation.

Every experimental knob named in the research plan lives in a frozen
dataclass so that a run can be reproduced from its recorded manifest alone
(plan sections 6.1, 6.2, 6.4, 6.5 and 9).  Configuration objects hash to a
stable digest via :func:`config_digest`; the digest is written into every
artifact.

Seeds are derived, never chosen ad hoc: :func:`derive_seed` maps a master
seed plus a label path to a 63-bit integer.  Two different labels give
independent streams and the same label always gives the same stream.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

# Master seed for the encoder-independent program.  Distinct from the
# low-dimensional program's MASTER (20260719) so no stream is shared.
MASTER_SEED = 20260724

# The audit bank must never be reachable from the training seed path.
AUDIT_SEED = 777_000_001


def derive_seed(master: int, *labels: Any) -> int:
    """Deterministic 63-bit seed from a master seed and a label path."""
    payload = json.dumps([master, [str(x) for x in labels]], sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def config_digest(obj: Any) -> str:
    """SHA-256 of a canonical JSON rendering of a (nested) dataclass."""
    return hashlib.sha256(
        json.dumps(to_jsonable(obj), sort_keys=True).encode("utf-8")
    ).hexdigest()


def to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    if isinstance(obj, float):
        # JSON cannot carry non-finite floats; artifacts must stay valid.
        return obj if math.isfinite(obj) else None
    return str(obj)


# ---------------------------------------------------------------------------
# Branch A: spectral source-law anchor (plan section 6.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BandSpec:
    """One radial frequency band.

    ``length_multiplier`` is applied to the *target-only projected scale*
    (plan section 6.1: calibrate from projected scales, not the ambient
    pixel norm).  Small multipliers give high frequencies.
    """

    name: str
    length_multiplier: float
    weight: float = 1.0
    law: str = "student"          # "half_normal" | "student" | "heavy_cauchy"
    nu: float = 3.0               # degrees of freedom for "student"


DEFAULT_BANDS: tuple[BandSpec, ...] = (
    BandSpec("low", 4.0),
    BandSpec("mid", 1.0),
    BandSpec("high", 0.25),
)


@dataclass(frozen=True)
class AnchorConfig:
    """Finite random-feature approximation of the spectral anchor.

    The exact-zero implication in the plan is a statement about the ideal
    expectation over the spectral measure.  A finite bank is an unbiased
    estimator of that expectation, not itself a characteristic family.
    """

    features: int = 256
    bands: tuple[BandSpec, ...] = DEFAULT_BANDS
    direction_mode: str = "orthogonal"    # "iid" | "orthogonal"
    # The TRAINING estimator is manifestly nonnegative, which makes it a
    # stable penalty and prevents cancellation against the flow term.  This
    # does *not* make a fixed finite bank measure determining, and the raw
    # flow-matching MSE need not attain zero.  The unbiased U-statistic can be
    # negative and is therefore computed for diagnostics only.
    training_estimator: str = "biased"    # "biased" only
    diagnostic_estimator: str = "unbiased"
    refresh_fraction: float = 0.25
    refresh_every: int = 25               # macro steps; 0 disables refreshing
    audit_features: int = 512
    # Reform R6: coarse-to-fine band weighting.  The anchor gradient scales
    # with |omega|, so a fixed bank lets the high band supply the largest
    # gradient while its phase decorrelates across samples -- noise that
    # drowns the informative coarse band.  With a schedule the TRAINING bank
    # starts coarse and admits higher bands as training proceeds; the AUDIT
    # bank always stays at full declared width, so reported detection power
    # is never narrowed.  "fixed" reproduces the Phase-1 behaviour exactly.
    band_schedule: str = "coarse_to_fine"   # "fixed" | "coarse_to_fine"
    schedule_floor: float = 0.05          # min relative weight of any band
    schedule_warmup: float = 0.5          # fraction of training to reach full
    projected_scale_quantile: float = 0.5
    projected_scale_probes: int = 64
    projected_scale_pairs: int = 512


# ---------------------------------------------------------------------------
# Branch B: fixed compositional image geometry (plan section 6.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometryConfig:
    family: str = "wavelet"
    # "raw" | "pyramid" | "wavelet" | "scattering" | "randconv"
    # | "dictionary" | "haar_control" | "reference_encoder"
    scales: int = 3
    orientations: int = 4
    pool: int = 4                 # coarse spatial pooling grid (locations)
    modulus_eps: float = 1e-3     # smooth modulus |u|_eps = sqrt(u^2+eps^2)
    kernel_eps: float = 1e-3      # smooth Laplace kernel epsilon
    second_order: bool = False
    covariance_terms: bool = False
    randconv_channels: int = 16
    randconv_layers: int = 2
    randconv_seed_label: str = "randconv"
    include_global_pyramid: bool = True
    # "auto" reproduces the Phase-1 rule (the paper's non-smooth `laplace`
    # for raw+standard, `smooth_laplace` elsewhere), which confounded
    # geometry with kernel smoothness.  Phase 2 sets this explicitly so every
    # arm shares one base kernel.
    base_kernel: str = "auto"         # "auto" | "smooth_laplace" | ...
    bandwidth_quantile: float = 0.5   # median heuristic, target-only
    bandwidth_multiplier: float = 1.0
    # The plan's declared conservative form.  Phase 0 measured both: "sum"
    # keeps the best-matching block's discrimination when the cloud is far
    # from the target, while "product" accumulates distance over all blocks
    # and flattens there.  See phase0_gate.json / G0.3.
    combine: str = "sum"              # "sum" | "product"; both are PD
    # Declared selectivity: solve target-only for the global bandwidth
    # factor whose median row ESS is this fraction of the batch.  None keeps
    # the bare median heuristic, which is nearly flat in high dimension.
    target_ess_fraction: float | None = 0.5
    # The paper's operating point (README Table 8 / A.6) states temperature
    # in NORMALIZED units, where the mean pairwise feature distance is 1.
    # Setting this pins tau = bandwidth_tau * (median pairwise distance) per
    # block and overrides the ESS calibration, so the paper's declared grid
    # {0.02, 0.05, 0.2} can be tested directly.
    bandwidth_tau: float | None = None


# ---------------------------------------------------------------------------
# Movement (plan section 6.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldConfig:
    direction_mode: str = "kernel_gradient"   # | "standard"
    normalization: str = "rms"                # "none" | "rms"
    denominator_floor: float = 1e-30
    self_mask: bool = False


# ---------------------------------------------------------------------------
# Adaptive mixture and cross-fitting (plan section 6.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MixtureConfig:
    adaptive: bool = False
    floor: float = 0.02
    ema: float = 0.1              # beta
    temperature: float = 1.0      # T
    epsilon: float = 1e-12
    controller_every: int = 5     # macro steps between controller updates


# ---------------------------------------------------------------------------
# Objective (plan section 6.5)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveConfig:
    lambda_anchor: float = 1.0
    lambda_geometry: float = 1.0
    lambda_regularization: float = 0.0
    # Reform R24.  **Inert under an adaptive optimizer.**  The stop-gradient
    # loss has gradient `-2 eta V / n`, so eta enters only as a constant
    # multiplier -- and Adam is invariant to constant gradient rescaling.
    # Measured: eta in {0.05, 0.5, 5.0} gives byte-identical results under
    # Adam and a large effect under SGD.  It still sets the *reported* loss
    # scale and it is the real step for free particles, which have no
    # optimizer, so it is kept rather than removed.  The generator's actual
    # step is `TrainConfig.learning_rate` under `TrainConfig.optimizer`.
    step_eta: float = 0.5         # eta_j in the stop-gradient regression
    # Reform R11, generalized by R26.  Correct the stop-gradient teacher so it
    # carries the real batch's second moment.  Confirmed empirically (Phase 3:
    # 18/18; Phase 5: 9/9); mechanism unknown, four explanations refuted.  It
    # *matches* a second moment -- it is not a "variance collapse repair", see
    # `objectives.corrected_teacher`.  True selects the scalar form for
    # backward compatibility; `teacher_correction` selects among the richer
    # forms and takes precedence when it is not "none".
    teacher_variance_match: bool = False
    # Reform R26 (Phase 6B).  Under an adaptive optimizer only interventions
    # that change the gradient's DIRECTION can act; every scale-based one is
    # invariant.  The scalar match is the crudest direction change, so these
    # are its natural refinements.
    # "none" | "scalar" | "per_coordinate" | "eigendirection"
    teacher_correction: str = "none"
    # Declared, never searched: tests whether the scalar match's undershoot
    # (it reaches 0.81-0.89 of the target, not 1.0) is a systematic gain
    # deficit.
    teacher_correction_gain: float = 1.0
    # Numerical guard for the per-direction forms: a direction carrying almost
    # no teacher variance would otherwise receive an unbounded gain.  How
    # often it binds is reported, not assumed negligible.
    teacher_correction_ratio_cap: float = 10.0
    # Reform R21.  The realized change in OUTPUT space is currently whatever
    # the optimizer produces through the Jacobian: measured at 42x the
    # requested displacement at initialization, and leaving 87% of the gap
    # open later.  This caps the ratio of realized to requested change, so
    # `step_eta` means what the plan says it means.  None disables the cap.
    output_step_cap: float | None = None
    # Reform R16.  The teacher map loses 2-8% of effective dimension per
    # application at constant variance, worst mid-trajectory.  Free particles
    # -- which do not collapse -- apply a decaying fractional step; the
    # regression applies its map at full strength every step.  "linear_decay"
    # mirrors the free-particle schedule exactly.
    eta_schedule: str = "constant"     # "constant" | "linear_decay"


# ---------------------------------------------------------------------------
# Training / arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 300
    # Reform R22.  The recipe silently assumed one optimizer step per teacher.
    # That choice sets which attractor the coupled iteration lands on -- at 1
    # step the generator closes 13% of the gap to its own teacher, at 32 it
    # closes 68% and the effective dimension moves from .27 to 1.4 of the
    # data's -- so it is a declared axis, not an implicit constant.
    steps_per_teacher: int = 1
    batch: int = 64
    # Reform R27.  How many GENERATOR samples form the field's cloud, which
    # until now was silently tied to `batch`.  The Phase-6 follow-up measured
    # this as a first-order axis: at every bandwidth, a 64-particle cloud
    # costs about an order of magnitude of ED2 against 512, while the number
    # of *target* examples barely matters.  The generator has been training
    # in the bad part of that axis because its cloud was its batch.
    # None keeps the previous behaviour exactly (cloud = batch).
    field_cloud: int | None = None
    controller_batch: int = 64
    audit_batch: int = 64
    inner_steps: int = 1
    # Reform R25.  These two are the generator's REAL step control -- the
    # objective's `step_eta` is inert under an adaptive optimizer (R24) -- so
    # they are reported in every artifact summary rather than left implicit.
    # `learning_rate` was fixed at 2e-3 by the Phase-1 pre-registration on a
    # three-point check and inherited unexamined until Phase 6A swept it.
    optimizer: str = "adam"       # "adam" | "sgd" | "sgd_momentum"
    momentum: float = 0.9         # used by "sgd_momentum" only
    learning_rate: float = 2e-3
    latent_dim: int = 32
    width: int = 64
    image_size: int = 16
    channels: int = 3
    eval_samples: int = 512


@dataclass(frozen=True)
class ArmConfig:
    """One row of the plan's Phase-1 arm table (section 9, Phase 1)."""

    arm_id: str
    use_anchor: bool
    geometry: GeometryConfig | None
    field: FieldConfig
    mixture: MixtureConfig = field(default_factory=MixtureConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    anchor: AnchorConfig = field(default_factory=AnchorConfig)
    note: str = ""

    def __post_init__(self) -> None:
        if not self.use_anchor and self.geometry is None:
            raise ValueError(f"arm {self.arm_id} has no loss branch at all")
        if self.use_anchor and self.objective.lambda_anchor <= 0:
            raise ValueError(
                f"arm {self.arm_id} enables the anchor with a non-positive "
                "weight; the exact-zero implication needs lambda_A > 0")
        if self.geometry is not None and self.objective.lambda_geometry < 0:
            raise ValueError(f"arm {self.arm_id} has lambda_G < 0")
        if self.objective.lambda_regularization < 0:
            raise ValueError(f"arm {self.arm_id} has lambda_R < 0")
