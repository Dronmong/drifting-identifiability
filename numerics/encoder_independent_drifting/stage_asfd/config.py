"""Frozen ASFD configuration.

**No constant here is inherited from B1 or B2.**  Their frozen values were
calibrated against a different architecture, a different loss scale and a
different data subset:

===========================  ====================  ==============================
constant                     frozen B1/B2 value    why it cannot transfer
===========================  ====================  ==============================
B2 bandwidth tau             7.085388360479058     all-class CIFAR raw pixels,
                                                   median pairwise distance 37.49
B2 event weight lambda       1.9294302093274076e-4 calibrated against the F3B
                                                   bridge's flow loss
B1 event weight              0.9310125645774651    same
B1 projected scale           0.4299860893300136    bridge-scale activations
===========================  ====================  ==============================

What transfers is the *form* of each term and the *calibration procedure*.  A
regression test asserts that no B1 or B2 freeze artifact is loadable from this
package.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..stage_cap.config import FEATURE_LEVELS

#: Level labels, in trunk order.  Declared by the frozen trunk, not here, so
#: that the taps are covered by CAP-EMF-1's source hash rather than retrofitted.
LEVEL_NAMES: tuple[str, ...] = tuple(name for name, _, _ in FEATURE_LEVELS)

ARMS: tuple[str, ...] = ("EMF-control", "EMF-raw", "EMF-ASFD")


@dataclass(frozen=True)
class FeatureConfig:
    """How the frozen trunk is read."""

    levels: tuple[str, ...] = LEVEL_NAMES
    #: 16x16 tokens -> 8x8 local vectors, plus a global mean and a global std.
    pool: int = 2
    local_vectors: int = 64
    global_vectors: int = 2
    #: Selected on the target-only qualification audit, then frozen. Not
    #: assumed: TFD ablated sigma_tf in {0.02, 0.1, 0.5} and chose 0.1, but its
    #: teacher is a multi-step diffusion model and ours is a one-call direct-x
    #: EMF trunk, so the transfer is an assumption the audit can settle cheaply.
    t_f: float = 0.10
    t_f_grid: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50)
    #: Four rather than two: the positive barycenter built from a small fixed
    #: view set is a fixed random function whose deviation from the population
    #: barycenter is a bias the generator can partly learn to match.
    views_per_image: int = 4
    #: Per-channel standardization fires only above this token-level PC1 share.
    per_channel_pc1_trigger: float = 0.35
    #: Bounds amplification of low-variance noise channels at 10x.
    per_channel_floor_fraction: float = 0.10

    @property
    def vectors_per_level(self) -> int:
        return self.local_vectors + self.global_vectors

    def validate(self) -> None:
        if not self.levels:
            raise ValueError("at least one feature level is required")
        if set(self.levels) - set(LEVEL_NAMES):
            raise ValueError(f"unknown feature level in {self.levels}")
        if self.pool < 1:
            raise ValueError("pool must be positive")
        if self.vectors_per_level != 66:
            raise ValueError("descriptor shape drifted from 66 vectors per level")
        if not 0 < self.t_f < 1:
            raise ValueError("feature noise level must lie in (0,1)")
        if self.t_f not in self.t_f_grid:
            raise ValueError("the frozen t_f must be a member of the declared grid")
        if self.views_per_image < 2:
            raise ValueError("a single cached view cannot debias the bank")
        if not 0 < self.per_channel_pc1_trigger < 1:
            raise ValueError("PC1 trigger must lie in (0,1)")
        if not 0 < self.per_channel_floor_fraction < 1:
            raise ValueError("per-channel floor fraction must lie in (0,1)")


@dataclass(frozen=True)
class FieldConfig:
    """Roles, bandwidths, and the health limits that gate them.

    The radii span local to global on purpose.  A set like {0.35, 0.60, 0.85}
    spans under 2x in tau and contains no local regime, so all three fields are
    blind to sub-bandwidth structure in the same way and averaging them does not
    repair it -- which is the measured mechanism behind B2's 38-40% rank
    collapse.  TFD's ablated set spans 10x.
    """

    radii: tuple[float, ...] = (0.10, 0.35, 0.85)
    #: If the smallest radius fails the health floors, step along this ladder
    #: and record which rung was used. Never widen it silently.
    radius_ladder: tuple[float, ...] = (0.10, 0.15, 0.20)
    #: Asymmetric deliberately: positives are cached so a larger side is nearly
    #: free, while each negative costs a generator forward. At 64-vs-64 the
    #: real-versus-real floor was 54-67% of B2's total measured energy, so most
    #: of what the correction differentiated was sampling noise.
    positives: int = 256
    probes: int = 64
    negatives: int = 64
    probe_noise_std: float = 0.05
    ess_samples: int = 256
    ess_iterations: int = 32
    #: Calibration-time rejection, target side. The tail floor is a **fraction
    #: of the requested radius**, not an absolute value. An absolute floor of
    #: 0.10 was written when the radii were {0.35, 0.60, 0.85}; against a set
    #: that starts at 0.10 it demands the 5th percentile reach the median,
    #: which no distribution satisfies, and every local radius would be
    #: rejected -- silently collapsing "multi-radius" back into broad fields.
    ess_p05_fraction: float = 0.25
    max_weight_p95_ceiling: float = 0.50
    #: Runtime gate, generated side. If this approaches one the negative
    #: barycenter approaches a plain batch mean and the energy has silently
    #: degenerated to first-moment matching. B2 logged this statistic and never
    #: gated it; it ran 0.666 -> 0.702 with nothing watching.
    negative_ess_ceiling: float = 0.90

    def validate(self) -> None:
        if len(self.radii) < 2:
            raise ValueError("multi-radius requires at least two radii")
        if sorted(self.radii) != list(self.radii):
            raise ValueError("radii must be increasing")
        if any(not 0 < value < 1 for value in self.radii):
            raise ValueError("radii are ESS fractions in (0,1)")
        if max(self.radii) / min(self.radii) < 4.0:
            raise ValueError("the radius set does not span a local-to-global range")
        if self.positives < self.negatives:
            raise ValueError("the cached positive side should not be the smaller one")
        if min(self.probes, self.negatives) < 8:
            raise ValueError("field roles are too small to estimate anything")
        if self.probe_noise_std <= 0:
            raise ValueError("the probe law needs positive noise for full support")
        if self.ess_samples < 16 or self.ess_iterations < 4:
            raise ValueError("invalid bandwidth calibration budget")
        if not 0 < self.ess_p05_fraction < 1:
            raise ValueError("the ESS tail fraction must lie in (0,1)")
        if not 0 < self.max_weight_p95_ceiling < 1:
            raise ValueError("maximum-weight ceiling must lie in (0,1)")
        if not 0 < self.negative_ess_ceiling < 1:
            raise ValueError("negative ESS ceiling must lie in (0,1)")


@dataclass(frozen=True)
class GradientConfig:
    """Per-component caps and outcome-based aborts.

    Caps are **per component**, not a shared total.  A shared cap would give the
    ASFD arm ~29% less raw anchor and B1 than the arm it is compared against,
    confounding "added a semantic term" with "cut the protection by a third" --
    and the confound would point the wrong way, since the arm carrying the new
    pressure would be the one with weakened protection.

    There is no projection.  A projected update is non-conservative, so the
    identifiability implication would apply to a loss the optimizer never
    descends; and projection deletes precisely the component of the raw anchor
    that opposes the primary gradient, which is the only component that does
    any work.
    """

    cadence: int = 10
    cap_b1: float = 0.15
    cap_raw: float = 0.10
    cap_self: float = 0.10
    #: A correction that never opposes the primary gradient is useless, so mild
    #: opposition is the working regime. Only the anti-parallel pathology --
    #: the term simply negating training -- is an abort.
    anti_parallel_cosine: float = -0.8
    anti_parallel_window: int = 200
    #: Rank collapse against the arm's own step-zero value, sustained over two
    #: logged checkpoints.
    rank_abort_fraction: float = 0.70

    def validate(self) -> None:
        if self.cadence < 1:
            raise ValueError("correction cadence must be positive")
        if min(self.cap_b1, self.cap_raw, self.cap_self) <= 0:
            raise ValueError("every component cap must be positive")
        if max(self.cap_b1, self.cap_raw, self.cap_self) >= 1.0:
            raise ValueError("a component cap at or above the primary gradient")
        if not -1.0 < self.anti_parallel_cosine < 0.0:
            raise ValueError("the anti-parallel threshold must lie in (-1,0)")
        if self.anti_parallel_window < 1:
            raise ValueError("the cosine window must be positive")
        if not 0 < self.rank_abort_fraction < 1:
            raise ValueError("rank abort fraction must lie in (0,1)")


@dataclass(frozen=True)
class QualificationConfig:
    """Protocol section 7 thresholds.  G7 and G8 run first, on purpose."""

    #: Two-sided, per Haar band. The upper bound rejects the hypersensitivity
    #: seen in the Phases 17-18 pretrained-ResNet harness; the LOWER bound is
    #: the point, because a trunk allocating capacity under MSE underweights
    #: exactly the bands an MSE-trained generator is weakest in, and a map with
    #: zero HF sensitivity passes a one-sided check trivially.
    band_sensitivity_low: float = 0.25
    band_sensitivity_high: float = 4.0
    #: Levels must carry distinct information or the branch is a silent no-op.
    inter_level_cosine_ceiling: float = 0.90
    inter_level_cka_ceiling: float = 0.95
    #: Tightened from 0.995, which permits ~99% shared variance. A branch that
    #: is 90% aligned with the raw branch adds weight, not geometry.
    raw_field_cosine_ceiling: float = 0.90
    benign_auc_floor: float = 0.80
    scramble_fraction_floor: float = 0.80
    #: Recalibrated against measurement. The original 16.0 was never validated
    #: against a trained model -- ``_inverse_haar`` crashed on three-channel
    #: images, so the qualification had never completed on real data at all --
    #: and it rejects **both** trained models available, including the base
    #: foundation the mechanism was designed to continue.
    #:
    #: Effective rank at t_f = 0.10, 512 CIFAR-10 images, per level:
    #:
    #:   level      untrained   base CAP-EMF-1   repaired foundation
    #:   enc_mid         2.36             7.59                 11.66
    #:   enc_final       2.37            18.90                 17.05
    #:   dec_mid         2.41            30.41                 25.93
    #:   dec_final       2.45             7.52                  3.70
    #:
    #: An untrained trunk sits at 2.1-2.45 uniformly, which is what collapse
    #: actually looks like; the working range spans 3.7 to 88. A floor of 16
    #: sits above most of the working range rather than above collapse.
    #:
    #: 3.0 is placed between the two: it rejects every untrained level and
    #: admits every trained one. Collapse protection does not rest on it alone
    #: -- G8 inter-level non-redundancy independently rejects the untrained
    #: control at every t_f, before this gate is even reached, while both
    #: trained models pass G8.
    #:
    #: This is a threshold set after seeing data and should be read as such.
    #: The thinnest margin is the repaired foundation's dec_final at 3.70
    #: against a 2.45 collapse baseline, so that level contributes little
    #: descriptor geometry and the anchor is effectively carried by dec_mid and
    #: enc_final.
    rank_floor: float = 3.0
    pc1_ceiling: float = 0.50
    distance_cv_floor: float = 0.05

    def validate(self) -> None:
        if not 0 < self.band_sensitivity_low < 1 < self.band_sensitivity_high:
            raise ValueError("band sensitivity bounds must straddle one")
        for value in (
            self.inter_level_cosine_ceiling,
            self.inter_level_cka_ceiling,
            self.raw_field_cosine_ceiling,
            self.benign_auc_floor,
            self.scramble_fraction_floor,
            self.pc1_ceiling,
        ):
            if not 0 < value <= 1:
                raise ValueError("qualification thresholds must lie in (0,1]")
        if self.rank_floor < 2:
            raise ValueError("the rank floor is too permissive to reject collapse")
        if self.distance_cv_floor <= 0:
            raise ValueError("distance CV floor must be positive")


@dataclass(frozen=True)
class ASFDConfig:
    features: FeatureConfig = field(default_factory=FeatureConfig)
    field_config: FieldConfig = field(default_factory=FieldConfig)
    gradients: GradientConfig = field(default_factory=GradientConfig)
    qualification: QualificationConfig = field(default_factory=QualificationConfig)
    #: Stage D continuation length per arm, forked from the frozen foundation.
    continuation_updates: int = 50_000
    checkpoint_every: int = 10_000
    development_units: tuple[int, ...] = (910,)

    def validate(self) -> None:
        self.features.validate()
        self.field_config.validate()
        self.gradients.validate()
        self.qualification.validate()
        if self.continuation_updates <= 0:
            raise ValueError("the continuation must be positive")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint cadence must be positive")
        if self.continuation_updates % self.checkpoint_every:
            raise ValueError("the continuation must end on a checkpoint")


def asfd_config() -> ASFDConfig:
    result = ASFDConfig()
    result.validate()
    return result


def smoke_config() -> ASFDConfig:
    """Mechanics only.  Never a result."""
    result = ASFDConfig(
        features=FeatureConfig(views_per_image=2),
        field_config=FieldConfig(
            positives=128,
            probes=8,
            negatives=8,
            # Not smaller: the smallest admissible radius is bounded by the
            # cloud size. At 16 samples an ESS fraction of 0.10 is 1.6
            # neighbours, so one weight dominates and the max-weight ceiling
            # correctly refuses it -- the health floors and the radius set are
            # coupled, and a tiny smoke cloud cannot exercise a local radius.
            ess_samples=128,
            ess_iterations=16,
        ),
        continuation_updates=4,
        checkpoint_every=2,
    )
    result.validate()
    return result
