"""Frozen candidates and matched sampler arms for the CAP-EMF-2 screen."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..stage_cap.config import CAPGateConfig, CAPProfile
from ..stage_cap.config import profile as cap1_profile


@dataclass(frozen=True)
class NumericalCandidate:
    name: str
    embedding_scale: float
    delta: float
    stopped_evaluation: str
    rationale: str

    @property
    def maximum_phase_step(self) -> float:
        return self.embedding_scale * self.delta

    def validate(self) -> None:
        if self.embedding_scale <= 0 or self.delta <= 0:
            raise ValueError("numerical scale and delta must be positive")
        if self.stopped_evaluation not in {"dense", "fp32_dense"}:
            raise ValueError("CAP2 candidates must use a matched dense stopped path")


NUMERICAL_CANDIDATES: dict[str, NumericalCandidate] = {
    # Historical control for the admission audit only.  Its ten-radian maximum
    # phase step is known to fail on the preserved checkpoint and it may never
    # be promoted merely because the name contains "legacy".
    "legacy_1000_d01": NumericalCandidate(
        name="legacy_1000_d01",
        embedding_scale=1_000.0,
        delta=0.01,
        stopped_evaluation="dense",
        rationale="historical numerical control; expected to fail admission",
    ),
    # Predeclared candidate, not an admitted value.  The original narrow audit
    # produced mixed results at delta=.0002 (5/9 old checks passed), so only the
    # expanded real/synthetic, target-and-gradient CUDA matrix may accept it.
    # Full-FP32 dense stopped evaluation is mandatory because a small difference
    # is exposed to subtractive roundoff.
    "local_1000_d0002_fp32": NumericalCandidate(
        name="local_1000_d0002_fp32",
        embedding_scale=1_000.0,
        delta=0.0002,
        stopped_evaluation="fp32_dense",
        rationale=(
            "predeclared small-step repair on the trained CAP checkpoint; "
            "unadmitted until the full production-GPU matrix passes"
        ),
    ),
    # Architecture candidate, not pre-approved.  It bounds the maximum phase
    # move at 0.1 radian while retaining a larger finite difference.  Because
    # the embedding function changes, it needs a short trained-model admission
    # and cannot be certified from the historical checkpoint alone.
    "smooth_100_d001_fp32": NumericalCandidate(
        name="smooth_100_d001_fp32",
        embedding_scale=100.0,
        delta=0.001,
        stopped_evaluation="fp32_dense",
        rationale=(
            "smoother conditioning candidate; requires a short trained-model "
            "audit before sampler comparison"
        ),
    ),
}


SAMPLER_ARMS = {
    "legacy": ("cap_conditional_logitnormal", 0.01),
    "ordered_logitnormal": ("ordered_logitnormal", 0.0),
    "ordered_uniform": ("ordered_uniform", 0.0),
}


def numerical_candidate(name: str) -> NumericalCandidate:
    try:
        result = NUMERICAL_CANDIDATES[name]
    except KeyError as error:
        raise ValueError(f"unknown CAP2 numerical candidate {name!r}") from error
    result.validate()
    return result


def screen_profile(
    arm: str,
    numerical: str,
    *,
    updates: int = 150_000,
    smoke: bool = False,
) -> CAPProfile:
    """Build one matched arm; no scientific knob depends on the arm name."""
    if arm not in SAMPLER_ARMS:
        raise ValueError(f"unknown CAP2 sampler arm {arm!r}")
    candidate = numerical_candidate(numerical)
    base = cap1_profile("smoke" if smoke else "capability")
    sampler_mode, sampled_r_floor = SAMPLER_ARMS[arm]

    if smoke:
        updates = 4
        checkpoints = (2, 4)
    else:
        if updates not in {
            50_000,
            100_000,
            150_000,
            300_000,
            500_000,
            650_000,
            750_000,
        }:
            raise ValueError(
                "CAP2 updates must be a declared 50k/100k/150k/300k/500k/"
                "650k/750k foundation checkpoint"
            )
        checkpoints = tuple(range(50_000, updates + 1, 50_000))

    result = replace(
        base,
        # The scientific identity is stable across 50k -> 150k -> 300k
        # promotion so one recovery stream can continue.  The planned horizon
        # is already recorded separately in ``train.updates``.
        name=f"cap2-{arm}-{candidate.name}{'-smoke' if smoke else ''}",
        purpose=(
            "ordered-uniform CAP-EMF-2 one-call foundation for a separately "
            "gated ASFD continuation; no sealed-test selection"
            if not smoke and updates > 300_000
            else "matched CAP-EMF-2 developmental sampler screen; no ASFD and "
            "no sealed-test selection"
        ),
        model=replace(base.model, scalar_embedding_scale=candidate.embedding_scale),
        objective=replace(
            base.objective,
            sampler_mode=sampler_mode,
            sampled_r_floor=sampled_r_floor,
            diagonal_sampling="fixed_count_first_draw",
            emf_delta=candidate.delta,
            stopped_evaluation=candidate.stopped_evaluation,
        ),
        train=replace(
            base.train,
            updates=updates,
            checkpoint_updates=checkpoints,
            snapshot_every=2 if smoke else 25_000,
        ),
    )
    result.validate()
    return result


def apply_calibrated_gate(profile: CAPProfile, calibration: dict) -> CAPProfile:
    if calibration.get("status") != "cap-emf2-gate-calibration":
        raise ValueError("not a CAP2 gate calibration")
    gate = CAPGateConfig(**calibration["gate"])
    gate.validate()
    result = replace(profile, gate=gate)
    result.validate()
    return result
