"""Optimizer-space Quantile-Guarded Drifting (QGD).

This module implements the mechanism described in
``QuantileGuardedDriftingResearchPlan.md``.  It is development infrastructure,
not a confirmatory result.  The historical QLD, LB-QCD, and paper paths remain
separate and are used as bitwise regression references.

The key operation constructs two objective-specific Adam proposals from a
shared generator state:

* an exact one-dimensional rank-transport proposal;
* a local paper-drift or sharp/log-KDE proposal.

The applied displacement is the closest point to the local proposal, in a
declared diagonal metric, satisfying first-order quantile progress and local
non-ascent constraints whenever they are compatible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal

import numpy as np

from conservative_finishers import conservative_field
from identifiability_drift import compute_field
from lbqcd import (
    AdamProposal,
    AdamState,
    adam_proposal,
    adam_step,
    apply_parameter_delta,
    capture_adam_state,
    copy_adam_state,
    exact_rank_field,
    paper_step,
    restore_adam_state,
    stopgrad_grads,
)
from run_identifiability_generator import TanhMLP


LocalKind = Literal["paper", "sharp"]
MetricKind = Literal["adam", "identity"]
PARAMETER_ORDER = tuple(TanhMLP.names)


@dataclass(frozen=True)
class DualAdamState:
    """Independent optimizer histories for rank and local objectives."""

    quantile: AdamState
    local: AdamState


@dataclass(frozen=True)
class GuardConfig:
    rho: float = 0.10
    metric: MetricKind = "adam"
    trust_factor: float = 2.0
    enforce_local_nonascent: bool = True
    feasibility_tolerance: float = 1e-10
    singular_tolerance: float = 1e-14
    metric_floor: float = 1e-12
    metric_ceiling: float = 1e8

    def validate(self) -> None:
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must lie in [0, 1]")
        if self.metric not in ("adam", "identity"):
            raise ValueError("unknown projection metric")
        if not math.isfinite(self.trust_factor) or self.trust_factor < 1.0:
            raise ValueError("trust_factor must be finite and at least one")
        if self.feasibility_tolerance <= 0.0:
            raise ValueError("feasibility_tolerance must be positive")
        if self.singular_tolerance <= 0.0:
            raise ValueError("singular_tolerance must be positive")
        if (self.metric_floor <= 0.0 or
                self.metric_ceiling < self.metric_floor):
            raise ValueError("invalid metric bounds")


@dataclass(frozen=True)
class SafetyGuardConfig:
    """Configuration for the current-gradient quantile safety guard.

    Unlike QGD-v1, this guard has no quantile optimizer state and imposes no
    positive quantile-progress floor.  It intervenes only when two independent
    empirical quantile gradients agree that the ordinary local Adam proposal
    is an ascent direction, and only when the exact non-ascent correction is a
    small fraction of that local proposal.
    """

    minimum_gradient_agreement: float = 0.25
    minimum_ascent_cosine: float = 0.05
    minimum_signal_to_null: float = 1.50
    maximum_correction_ratio: float = 0.25
    feasibility_tolerance: float = 1e-10
    singular_tolerance: float = 1e-14
    metric_floor: float = 1e-12
    metric_ceiling: float = 1e8

    def validate(self) -> None:
        if not -1.0 <= self.minimum_gradient_agreement <= 1.0:
            raise ValueError("gradient agreement must lie in [-1, 1]")
        if not 0.0 <= self.minimum_ascent_cosine <= 1.0:
            raise ValueError("ascent cosine must lie in [0, 1]")
        if (not math.isfinite(self.minimum_signal_to_null) or
                self.minimum_signal_to_null < 0.0):
            raise ValueError("signal-to-null threshold must be nonnegative")
        if (not math.isfinite(self.maximum_correction_ratio) or
                not 0.0 <= self.maximum_correction_ratio <= 1.0):
            raise ValueError("correction cap must lie in [0, 1]")
        if self.feasibility_tolerance <= 0.0:
            raise ValueError("feasibility tolerance must be positive")
        if self.singular_tolerance <= 0.0:
            raise ValueError("singular tolerance must be positive")
        if (self.metric_floor <= 0.0 or
                self.metric_ceiling < self.metric_floor):
            raise ValueError("invalid metric bounds")


@dataclass(frozen=True)
class SafetyGuardDiagnostics:
    action: str
    robust_ascent_detected: bool
    intervention_applied: bool
    cap_rejected: bool
    local_safety_rejected: bool
    active_set: str
    gradient_agreement: float
    train_ascent_cosine: float
    probe_ascent_cosine: float
    signal_to_null: float
    correction_metric_norm: float
    local_proposal_metric_norm: float
    correction_ratio: float
    train_directional_before: float
    train_directional_after: float
    probe_directional_before: float
    probe_directional_after: float
    local_directional_before: float
    local_directional_after: float
    train_quantile_field_rms: float
    probe_quantile_field_rms: float
    target_null_field_rms: float
    local_field_rms: float
    singular_candidates: int
    kernel_pairs: int
    denominator_min: float
    floor_activations: int
    optimizer_updates: int = 1
    generator_forward_calls: int = 2
    generator_example_evals: int = 0
    target_samples: int = 0
    backward_examples: int = 0
    sort_work: float = 0.0

    def serializable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectionDiagnostics:
    active_set: str
    quantile_constraint_before: float
    quantile_constraint_after: float
    quantile_threshold: float
    local_constraint_before: float
    local_constraint_after: float
    local_threshold: float
    correction_metric_norm: float
    applied_metric_norm: float
    quantile_proposal_metric_norm: float
    local_proposal_metric_norm: float
    gradient_cosine: float
    proposal_cosine: float
    trust_cap_active: bool
    safe_quantile_fallback: bool
    incompatible: bool
    singular_candidates: int
    quantile_feasible: bool
    local_feasible: bool


@dataclass(frozen=True)
class ProjectionResult:
    delta: np.ndarray
    diagnostics: ProjectionDiagnostics


@dataclass(frozen=True)
class GuardedStepDiagnostics:
    projection: ProjectionDiagnostics
    local_kind: str
    tau: float
    quantile_gradient_norm: float
    local_gradient_norm: float
    quantile_field_rms: float
    local_field_rms: float
    kernel_pairs: int
    denominator_min: float
    floor_activations: int
    optimizer_updates: int = 1
    generator_forward_calls: int = 1
    generator_example_evals: int = 0
    target_samples: int = 0
    backward_examples: int = 0
    sort_work: float = 0.0

    def serializable(self) -> dict[str, Any]:
        result = asdict(self)
        projection = result.pop("projection")
        result.update({f"projection_{key}": value
                       for key, value in projection.items()})
        return result


@dataclass(frozen=True)
class GuardedStepResult:
    state: DualAdamState
    diagnostics: GuardedStepDiagnostics


@dataclass(frozen=True)
class CheckpointMetric:
    ed2: float
    sw1: float
    coverage: float = 1.0
    mass_l1: float = 0.0


@dataclass(frozen=True)
class CheckpointSelection:
    selected_step: int
    bank_a_mean_scores: dict[int, float]
    bank_b_mean_scores: dict[int, float]
    bank_b_standard_errors: dict[int, float]
    eligible_steps: tuple[int, ...]
    bank_a_leaders: tuple[int, ...]
    threshold: float


def initialize_dual_state(model: TanhMLP) -> DualAdamState:
    """Copy the carried handoff state into two independent streams."""
    carried = capture_adam_state(model)
    return DualAdamState(
        quantile=copy_adam_state(carried),
        local=copy_adam_state(carried))


def flatten_named(values: dict[str, np.ndarray]) -> np.ndarray:
    if set(values) != set(PARAMETER_ORDER):
        raise ValueError("named arrays do not match the generator parameters")
    return np.concatenate([np.ravel(values[name])
                           for name in PARAMETER_ORDER])


def unflatten_named(vector: np.ndarray,
                    template: dict[str, np.ndarray]) \
        -> dict[str, np.ndarray]:
    flat = np.asarray(vector, dtype=float).ravel()
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name in PARAMETER_ORDER:
        shape = template[name].shape
        size = int(np.prod(shape))
        if offset + size > len(flat):
            raise ValueError("flat vector is shorter than the template")
        result[name] = flat[offset:offset + size].reshape(shape).copy()
        offset += size
    if offset != len(flat):
        raise ValueError("flat vector is longer than the template")
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _metric_norm(vector: np.ndarray, metric: np.ndarray) -> float:
    return float(np.sqrt(np.sum(vector * vector / metric)))


def _feasible(value: float, threshold: float, tolerance: float) -> bool:
    return value <= threshold + tolerance * (
        1.0 + abs(value) + abs(threshold))


def _candidate_for_active_set(
        base: np.ndarray, metric: np.ndarray, gradients: np.ndarray,
        thresholds: np.ndarray, active: tuple[int, ...],
        singular_tolerance: float) \
        -> tuple[np.ndarray, np.ndarray] | None:
    if not active:
        return base.copy(), np.zeros(0)
    selected = gradients[np.asarray(active)]
    rhs = selected @ base - thresholds[np.asarray(active)]
    gram = (selected * metric[None, :]) @ selected.T
    if len(active) == 1:
        denominator = float(gram[0, 0])
        scale = max(1.0, float(np.linalg.norm(selected[0])) ** 2)
        if not math.isfinite(denominator) or \
                denominator <= singular_tolerance * scale:
            return None
        multipliers = np.asarray([rhs[0] / denominator])
    else:
        determinant = float(np.linalg.det(gram))
        scale = max(1.0, float(np.linalg.norm(gram, ord=2)) ** 2)
        if not math.isfinite(determinant) or \
                abs(determinant) <= singular_tolerance * scale:
            return None
        try:
            multipliers = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return None
    candidate = base - metric * (selected.T @ multipliers)
    return candidate, multipliers


def project_guarded_delta(
        quantile_gradient: np.ndarray, local_gradient: np.ndarray,
        quantile_delta: np.ndarray, local_delta: np.ndarray,
        metric: np.ndarray, config: GuardConfig) -> ProjectionResult:
    """Solve the audited two-constraint QGD projection problem."""
    config.validate()
    gq = np.asarray(quantile_gradient, dtype=float).ravel()
    gd = np.asarray(local_gradient, dtype=float).ravel()
    dq = np.asarray(quantile_delta, dtype=float).ravel()
    dd = np.asarray(local_delta, dtype=float).ravel()
    diagonal = np.asarray(metric, dtype=float).ravel()
    if not (gq.shape == gd.shape == dq.shape == dd.shape == diagonal.shape):
        raise ValueError("projection vectors must have the same shape")
    if not all(np.all(np.isfinite(value)) for value in
               (gq, gd, dq, dd, diagonal)):
        raise FloatingPointError("projection input is non-finite")
    diagonal = np.clip(diagonal, config.metric_floor,
                       config.metric_ceiling)
    tolerance = config.feasibility_tolerance

    q_directional = float(np.dot(gq, dq))
    safe_fallback = False
    q_reference = dq.copy()
    if np.linalg.norm(gq) > config.singular_tolerance and \
            q_directional >= -tolerance:
        q_reference = -diagonal * gq
        q_directional = float(np.dot(gq, q_reference))
        safe_fallback = True
    q_threshold = config.rho * min(0.0, q_directional)
    local_threshold = 0.0

    gradients = np.stack([gq, gd])
    thresholds = np.asarray([q_threshold, local_threshold])
    active_sets: list[tuple[int, ...]] = [(), (0,)]
    if config.enforce_local_nonascent:
        active_sets.extend([(1,), (0, 1)])

    feasible_candidates: list[tuple[float, tuple[int, ...], np.ndarray]] = []
    singular = 0
    for active in active_sets:
        result = _candidate_for_active_set(
            dd, diagonal, gradients, thresholds, active,
            config.singular_tolerance)
        if result is None:
            singular += 1
            continue
        candidate, multipliers = result
        if len(multipliers) and np.any(multipliers < -tolerance):
            continue
        values = gradients @ candidate
        if not _feasible(float(values[0]), q_threshold, tolerance):
            continue
        if config.enforce_local_nonascent and not _feasible(
                float(values[1]), local_threshold, tolerance):
            continue
        objective = 0.5 * _metric_norm(candidate - dd, diagonal) ** 2
        feasible_candidates.append((objective, active, candidate))

    incompatible = False
    if feasible_candidates:
        _, active, applied = min(feasible_candidates, key=lambda item: item[0])
        active_label = ("none" if not active else
                        "+".join("quantile" if index == 0 else "local"
                                 for index in active))
    else:
        # The quantile reference is feasible for its own threshold by
        # construction.  The local constraint is deliberately relaxed here.
        applied = q_reference.copy()
        active_label = "quantile-fallback"
        incompatible = True

    q_norm = _metric_norm(q_reference, diagonal)
    d_norm = _metric_norm(dd, diagonal)
    trust_limit = config.trust_factor * max(q_norm, d_norm,
                                             config.singular_tolerance)
    trust_active = _metric_norm(applied, diagonal) > \
        trust_limit * (1.0 + tolerance)
    if trust_active:
        # Scaling the projected point toward zero could violate a negative
        # progress threshold.  The reference quantile proposal is inside the
        # declared radius and satisfies the primary constraint, so it is the
        # only safe capped fallback without solving a third nonlinear
        # constraint.
        applied = q_reference.copy()
        active_label = "trust-quantile-fallback"
        incompatible = incompatible or not _feasible(
            float(np.dot(gd, applied)), 0.0, tolerance)

    q_before = float(np.dot(gq, dd))
    q_after = float(np.dot(gq, applied))
    d_before = float(np.dot(gd, dd))
    d_after = float(np.dot(gd, applied))
    q_ok = _feasible(q_after, q_threshold, tolerance)
    d_ok = (not config.enforce_local_nonascent or
            _feasible(d_after, local_threshold, tolerance))
    if not q_ok:
        raise AssertionError("guarded projection violated its primary constraint")

    diagnostics = ProjectionDiagnostics(
        active_set=active_label,
        quantile_constraint_before=q_before,
        quantile_constraint_after=q_after,
        quantile_threshold=q_threshold,
        local_constraint_before=d_before,
        local_constraint_after=d_after,
        local_threshold=local_threshold,
        correction_metric_norm=_metric_norm(applied - dd, diagonal),
        applied_metric_norm=_metric_norm(applied, diagonal),
        quantile_proposal_metric_norm=q_norm,
        local_proposal_metric_norm=d_norm,
        gradient_cosine=_cosine(gq, gd),
        proposal_cosine=_cosine(q_reference, dd),
        trust_cap_active=trust_active,
        safe_quantile_fallback=safe_fallback,
        incompatible=incompatible,
        singular_candidates=singular,
        quantile_feasible=q_ok,
        local_feasible=d_ok)
    return ProjectionResult(applied, diagnostics)


def _local_field(kind: LocalKind, x: np.ndarray, positive: np.ndarray,
                 tau: float) \
        -> tuple[np.ndarray, int, float, int]:
    if kind == "paper":
        result = compute_field(
            x, positive, tau=tau, gain="paper", mask=True,
            on_degenerate="zero")
        return result.V, int(result.kernel_pairs), float("nan"), 0
    if kind == "sharp":
        result = conservative_field(
            "sharp", x, positive, tau=tau,
            reference_mode="reused_deleted")
        diagnostics = result.diagnostics
        return (
            result.field, int(diagnostics.kernel_pairs),
            float(min(diagnostics.positive_denominator_min,
                      diagnostics.negative_denominator_min)),
            int(diagnostics.floor_activations))
    raise ValueError(f"unknown local field kind: {kind}")


def _objective_data(
        model: TanhMLP, state: DualAdamState, latent: np.ndarray,
        positive: np.ndarray, local_kind: LocalKind, tau: float) \
        -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray],
                 dict[str, np.ndarray], AdamProposal, AdamProposal,
                 int, float, int]:
    x, cache = model.forward(latent, want_cache=True)
    quantile_field = exact_rank_field(x, positive, latent)
    local_field, kernel_pairs, denominator_min, floor_activations = \
        _local_field(local_kind, x, positive, tau)
    quantile_gradient = stopgrad_grads(
        model, cache, quantile_field, len(x))
    local_gradient = stopgrad_grads(model, cache, local_field, len(x))
    quantile_proposal = adam_proposal(
        model, state.quantile, quantile_gradient)
    local_proposal = adam_proposal(model, state.local, local_gradient)
    return (
        quantile_field, local_field, quantile_gradient, local_gradient,
        quantile_proposal, local_proposal, kernel_pairs, denominator_min,
        floor_activations)


def _install_guarded_result(
        model: TanhMLP, delta: np.ndarray,
        quantile_proposal: AdamProposal,
        local_proposal: AdamProposal) -> DualAdamState:
    named_delta = unflatten_named(delta, model.params)
    apply_parameter_delta(model, named_delta)
    # The model carries the local state for interoperable snapshots and a
    # possible historical paper continuation.  The independent quantile state
    # remains in the returned dual state.
    restore_adam_state(model, local_proposal.next_state)
    return DualAdamState(
        quantile=copy_adam_state(quantile_proposal.next_state),
        local=copy_adam_state(local_proposal.next_state))


def guarded_step(
        model: TanhMLP, state: DualAdamState, latent: np.ndarray,
        positive: np.ndarray, *, local_kind: LocalKind = "paper",
        tau: float = 0.5, config: GuardConfig = GuardConfig(),
        guard_enabled: bool = True) -> GuardedStepResult:
    """Apply one QGD update and return the independent next optimizer states."""
    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    config.validate()
    data = _objective_data(
        model, state, latent, positive, local_kind, tau)
    (q_field, d_field, q_gradient, d_gradient, q_proposal, d_proposal,
     kernel_pairs, denominator_min, floor_activations) = data
    flat_gq = flatten_named(q_gradient)
    flat_gd = flatten_named(d_gradient)
    flat_dq = flatten_named(q_proposal.delta)
    flat_dd = flatten_named(d_proposal.delta)

    if not guard_enabled:
        # Exact historical operation order for the compatibility path.  The
        # local state installed in the model must be the source state.
        restore_adam_state(model, state.local)
        before = {name: model.params[name].copy() for name in model.names}
        adam_step(model, d_gradient)
        applied = flatten_named({
            name: model.params[name] - before[name] for name in model.names})
        local_next = capture_adam_state(model)
        next_state = DualAdamState(
            quantile=copy_adam_state(state.quantile),
            local=copy_adam_state(local_next))
        metric = np.ones_like(applied)
        projection = ProjectionDiagnostics(
            active_set="guard-disabled",
            quantile_constraint_before=float(np.dot(flat_gq, applied)),
            quantile_constraint_after=float(np.dot(flat_gq, applied)),
            quantile_threshold=float("nan"),
            local_constraint_before=float(np.dot(flat_gd, applied)),
            local_constraint_after=float(np.dot(flat_gd, applied)),
            local_threshold=float("nan"),
            correction_metric_norm=0.0,
            applied_metric_norm=_metric_norm(applied, metric),
            quantile_proposal_metric_norm=_metric_norm(flat_dq, metric),
            local_proposal_metric_norm=_metric_norm(flat_dd, metric),
            gradient_cosine=_cosine(flat_gq, flat_gd),
            proposal_cosine=_cosine(flat_dq, flat_dd),
            trust_cap_active=False,
            safe_quantile_fallback=False,
            incompatible=False,
            singular_candidates=0,
            quantile_feasible=True,
            local_feasible=True)
    else:
        metric = (flatten_named(q_proposal.preconditioner)
                  if config.metric == "adam" else np.ones_like(flat_dq))
        projected = project_guarded_delta(
            flat_gq, flat_gd, flat_dq, flat_dd, metric, config)
        projection = projected.diagnostics
        next_state = _install_guarded_result(
            model, projected.delta, q_proposal, d_proposal)

    n = len(latent)
    diagnostics = GuardedStepDiagnostics(
        projection=projection,
        local_kind=local_kind,
        tau=float(tau),
        quantile_gradient_norm=float(np.linalg.norm(flat_gq)),
        local_gradient_norm=float(np.linalg.norm(flat_gd)),
        quantile_field_rms=float(np.sqrt(np.mean(q_field * q_field))),
        local_field_rms=float(np.sqrt(np.mean(d_field * d_field))),
        kernel_pairs=kernel_pairs,
        denominator_min=denominator_min,
        floor_activations=floor_activations,
        generator_example_evals=n,
        target_samples=n,
        backward_examples=2 * n,
        sort_work=2.0 * n * math.log2(max(n, 2)))
    return GuardedStepResult(next_state, diagnostics)


def _dual_metric_norm(vector: np.ndarray, metric: np.ndarray) -> float:
    """Dual norm to ``_metric_norm`` for a diagonal Adam metric."""
    return float(np.sqrt(np.sum(vector * vector * metric)))


def _metric_gradient_cosine(left: np.ndarray, right: np.ndarray,
                            metric: np.ndarray) -> float:
    denominator = (_dual_metric_norm(left, metric) *
                   _dual_metric_norm(right, metric))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(left * right * metric) / denominator)


def _metric_directional_cosine(gradient: np.ndarray, delta: np.ndarray,
                               metric: np.ndarray) -> float:
    denominator = (_dual_metric_norm(gradient, metric) *
                   _metric_norm(delta, metric))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(gradient, delta) / denominator)


def project_two_batch_nonascent(
        train_gradient: np.ndarray, probe_gradient: np.ndarray,
        local_delta: np.ndarray, metric: np.ndarray,
        config: SafetyGuardConfig = SafetyGuardConfig()) \
        -> tuple[np.ndarray, str, int]:
    """Project a local proposal onto two empirical non-ascent halfspaces.

    The objective is the squared distance to ``local_delta`` in the local
    Adam metric.  Both thresholds are zero: this is a safety constraint, not
    a forced quantile-progress objective.  The zero displacement is retained
    as a mathematically safe fallback for a singular pair of constraints; the
    caller's correction cap normally rejects that drastic fallback.
    """
    config.validate()
    first = np.asarray(train_gradient, dtype=float).ravel()
    second = np.asarray(probe_gradient, dtype=float).ravel()
    base = np.asarray(local_delta, dtype=float).ravel()
    diagonal = np.asarray(metric, dtype=float).ravel()
    if not (first.shape == second.shape == base.shape == diagonal.shape):
        raise ValueError("safety projection vectors must have the same shape")
    if not all(np.all(np.isfinite(value)) for value in
               (first, second, base, diagonal)):
        raise FloatingPointError("safety projection input is non-finite")
    diagonal = np.clip(
        diagonal, config.metric_floor, config.metric_ceiling)
    gradients = np.stack([first, second])
    thresholds = np.zeros(2)
    labels = {(): "none", (0,): "train", (1,): "probe",
              (0, 1): "train+probe"}
    candidates: list[tuple[float, tuple[int, ...], np.ndarray]] = []
    singular = 0
    for active in ((), (0,), (1,), (0, 1)):
        result = _candidate_for_active_set(
            base, diagonal, gradients, thresholds, active,
            config.singular_tolerance)
        if result is None:
            singular += 1
            continue
        candidate, multipliers = result
        if len(multipliers) and np.any(
                multipliers < -config.feasibility_tolerance):
            continue
        values = gradients @ candidate
        if any(not _feasible(float(value), 0.0,
                             config.feasibility_tolerance)
               for value in values):
            continue
        objective = 0.5 * _metric_norm(
            candidate - base, diagonal) ** 2
        candidates.append((objective, active, candidate))
    if not candidates:
        return np.zeros_like(base), "zero-fallback", singular
    _, active, candidate = min(candidates, key=lambda item: item[0])
    return candidate, labels[active], singular


def quantile_safety_step(
        model: TanhMLP, latent: np.ndarray, positive: np.ndarray,
        probe_latent: np.ndarray, probe_positive: np.ndarray, *,
        local_kind: LocalKind = "paper", tau: float = 0.5,
        config: SafetyGuardConfig = SafetyGuardConfig()) \
        -> SafetyGuardDiagnostics:
    """Apply one local Adam step with a conservative quantile safety check.

    Two independent batches provide current (non-momentum) rank-surrogate
    gradients.  The guard acts only when those gradients agree, both diagnose
    ascent of the ordinary local Adam proposal, and the empirical generator
    mismatch exceeds a target-vs-target sampling-noise floor.  A candidate is
    accepted only if its exact two-batch non-ascent correction is small and it
    does not turn a locally descending proposal into local ascent.

    When the guard is inactive or rejects a correction, the historical Adam
    operation is executed directly, preserving the ordinary paper suffix.
    The model retains exactly one optimizer state: the local objective's.
    """
    config.validate()
    if tau <= 0.0 or not math.isfinite(tau):
        raise ValueError("tau must be finite and positive")
    if (latent.shape != probe_latent.shape or
            positive.shape != probe_positive.shape or
            len(latent) != len(positive) or
            len(probe_latent) != len(probe_positive)):
        raise ValueError("safety batches must have paired equal shapes")

    x, cache = model.forward(latent, want_cache=True)
    local_field, kernel_pairs, denominator_min, floor_activations = \
        _local_field(local_kind, x, positive, tau)
    train_quantile_field = exact_rank_field(x, positive, latent)
    local_gradient = stopgrad_grads(
        model, cache, local_field, len(x))
    train_gradient = stopgrad_grads(
        model, cache, train_quantile_field, len(x))

    probe_x, probe_cache = model.forward(probe_latent, want_cache=True)
    probe_quantile_field = exact_rank_field(
        probe_x, probe_positive, probe_latent)
    probe_gradient = stopgrad_grads(
        model, probe_cache, probe_quantile_field, len(probe_x))
    target_null_field = exact_rank_field(
        positive, probe_positive, latent)

    local_proposal = adam_proposal(
        model, capture_adam_state(model), local_gradient)
    train_flat = flatten_named(train_gradient)
    probe_flat = flatten_named(probe_gradient)
    local_gradient_flat = flatten_named(local_gradient)
    local_delta = flatten_named(local_proposal.delta)
    metric = np.clip(
        flatten_named(local_proposal.preconditioner),
        config.metric_floor, config.metric_ceiling)

    agreement = _metric_gradient_cosine(train_flat, probe_flat, metric)
    train_ascent = _metric_directional_cosine(
        train_flat, local_delta, metric)
    probe_ascent = _metric_directional_cosine(
        probe_flat, local_delta, metric)
    train_rms = float(np.sqrt(np.mean(
        train_quantile_field * train_quantile_field)))
    probe_rms = float(np.sqrt(np.mean(
        probe_quantile_field * probe_quantile_field)))
    null_rms = float(np.sqrt(np.mean(
        target_null_field * target_null_field)))
    signal = math.sqrt(0.5 * (train_rms * train_rms +
                              probe_rms * probe_rms))
    signal_to_null = signal / max(null_rms, config.singular_tolerance)
    robust = bool(
        math.isfinite(agreement) and math.isfinite(train_ascent) and
        math.isfinite(probe_ascent) and
        agreement >= config.minimum_gradient_agreement and
        train_ascent >= config.minimum_ascent_cosine and
        probe_ascent >= config.minimum_ascent_cosine and
        signal_to_null >= config.minimum_signal_to_null)

    applied = local_delta
    active_set = "none"
    singular = 0
    correction_norm = 0.0
    local_norm = _metric_norm(local_delta, metric)
    correction_ratio = 0.0
    cap_rejected = False
    local_rejected = False
    intervention = False
    if robust:
        candidate, active_set, singular = project_two_batch_nonascent(
            train_flat, probe_flat, local_delta, metric, config)
        correction_norm = _metric_norm(candidate - local_delta, metric)
        correction_ratio = correction_norm / max(
            local_norm, config.singular_tolerance)
        cap_rejected = bool(
            correction_ratio > config.maximum_correction_ratio +
            config.feasibility_tolerance)
        local_before = float(np.dot(local_gradient_flat, local_delta))
        local_candidate = float(np.dot(local_gradient_flat, candidate))
        # Do not introduce local ascent when the baseline was descending, and
        # never make an already-ascending momentum step worse.
        local_limit = max(0.0, local_before)
        local_rejected = bool(
            local_candidate > local_limit + config.feasibility_tolerance *
            (1.0 + abs(local_candidate) + abs(local_limit)))
        if not cap_rejected and not local_rejected:
            applied = candidate
            intervention = True

    if intervention:
        apply_parameter_delta(
            model, unflatten_named(applied, model.params))
        restore_adam_state(model, local_proposal.next_state)
        action = "intervened"
    else:
        # Preserve the exact historical local-Adam operation whenever the
        # safety check does not make a small certified intervention.
        adam_step(model, local_gradient)
        if not robust:
            action = "inactive"
        elif cap_rejected:
            action = "cap-rejected"
        elif local_rejected:
            action = "local-rejected"
        else:
            action = "projection-rejected"
        applied = local_delta
        correction_norm = 0.0
        correction_ratio = 0.0

    train_before = float(np.dot(train_flat, local_delta))
    probe_before = float(np.dot(probe_flat, local_delta))
    local_before = float(np.dot(local_gradient_flat, local_delta))
    n = len(latent)
    return SafetyGuardDiagnostics(
        action=action,
        robust_ascent_detected=robust,
        intervention_applied=intervention,
        cap_rejected=cap_rejected,
        local_safety_rejected=local_rejected,
        active_set=active_set,
        gradient_agreement=agreement,
        train_ascent_cosine=train_ascent,
        probe_ascent_cosine=probe_ascent,
        signal_to_null=signal_to_null,
        correction_metric_norm=correction_norm,
        local_proposal_metric_norm=local_norm,
        correction_ratio=correction_ratio,
        train_directional_before=train_before,
        train_directional_after=float(np.dot(train_flat, applied)),
        probe_directional_before=probe_before,
        probe_directional_after=float(np.dot(probe_flat, applied)),
        local_directional_before=local_before,
        local_directional_after=float(np.dot(local_gradient_flat, applied)),
        train_quantile_field_rms=train_rms,
        probe_quantile_field_rms=probe_rms,
        target_null_field_rms=null_rms,
        local_field_rms=float(np.sqrt(np.mean(local_field * local_field))),
        singular_candidates=singular,
        kernel_pairs=kernel_pairs,
        denominator_min=denominator_min,
        floor_activations=floor_activations,
        generator_example_evals=2 * n,
        target_samples=2 * n,
        backward_examples=3 * n,
        sort_work=6.0 * n * math.log2(max(n, 2)))


def fixed_mix_step(
        model: TanhMLP, state: DualAdamState, latent: np.ndarray,
        positive: np.ndarray, *, quantile_weight: float = 0.10,
        local_kind: LocalKind = "paper", tau: float = 0.5,
        metric_kind: MetricKind = "adam") -> GuardedStepResult:
    """Apply a normalized fixed proposal mixture used as a causal ablation."""
    if not 0.0 <= quantile_weight <= 1.0:
        raise ValueError("quantile_weight must lie in [0, 1]")
    data = _objective_data(
        model, state, latent, positive, local_kind, tau)
    (q_field, d_field, q_gradient, d_gradient, q_proposal, d_proposal,
     kernel_pairs, denominator_min, floor_activations) = data
    gq = flatten_named(q_gradient)
    gd = flatten_named(d_gradient)
    dq = flatten_named(q_proposal.delta)
    dd = flatten_named(d_proposal.delta)
    metric = (flatten_named(q_proposal.preconditioner)
              if metric_kind == "adam" else np.ones_like(dq))
    metric = np.clip(metric, 1e-12, 1e8)
    q_norm = _metric_norm(dq, metric)
    d_norm = _metric_norm(dd, metric)
    scale = d_norm / max(q_norm, 1e-14)
    mixed = ((1.0 - quantile_weight) * dd +
             quantile_weight * scale * dq)
    next_state = _install_guarded_result(
        model, mixed, q_proposal, d_proposal)
    projection = ProjectionDiagnostics(
        active_set="fixed-mix",
        quantile_constraint_before=float(np.dot(gq, dd)),
        quantile_constraint_after=float(np.dot(gq, mixed)),
        quantile_threshold=float("nan"),
        local_constraint_before=float(np.dot(gd, dd)),
        local_constraint_after=float(np.dot(gd, mixed)),
        local_threshold=float("nan"),
        correction_metric_norm=_metric_norm(mixed - dd, metric),
        applied_metric_norm=_metric_norm(mixed, metric),
        quantile_proposal_metric_norm=q_norm,
        local_proposal_metric_norm=d_norm,
        gradient_cosine=_cosine(gq, gd),
        proposal_cosine=_cosine(dq, dd),
        trust_cap_active=False,
        safe_quantile_fallback=False,
        incompatible=False,
        singular_candidates=0,
        quantile_feasible=True,
        local_feasible=True)
    n = len(latent)
    diagnostics = GuardedStepDiagnostics(
        projection=projection,
        local_kind=local_kind,
        tau=float(tau),
        quantile_gradient_norm=float(np.linalg.norm(gq)),
        local_gradient_norm=float(np.linalg.norm(gd)),
        quantile_field_rms=float(np.sqrt(np.mean(q_field * q_field))),
        local_field_rms=float(np.sqrt(np.mean(d_field * d_field))),
        kernel_pairs=kernel_pairs,
        denominator_min=denominator_min,
        floor_activations=floor_activations,
        generator_example_evals=n,
        target_samples=n,
        backward_examples=2 * n,
        sort_work=2.0 * n * math.log2(max(n, 2)))
    return GuardedStepResult(next_state, diagnostics)


def checkpoint_score(metric: CheckpointMetric,
                     handoff: CheckpointMetric,
                     epsilon: float = 1e-12) -> float:
    """Equal-log-weight ED2/SW1 score normalized to the handoff."""
    values = (metric.ed2, metric.sw1, handoff.ed2, handoff.sw1)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        return float("inf")
    return float(
        0.5 * math.log((metric.ed2 + epsilon) /
                       (handoff.ed2 + epsilon)) +
        0.5 * math.log((metric.sw1 + epsilon) /
                       (handoff.sw1 + epsilon)))


def select_checkpoint(
        bank_a: dict[int, list[CheckpointMetric]],
        bank_b: dict[int, list[CheckpointMetric]],
        handoff: CheckpointMetric, *, top_k: int = 3,
        minimum_coverage: float = 0.0,
        maximum_mass_l1: float = float("inf")) -> CheckpointSelection:
    """Two-bank earliest-within-one-standard-error checkpoint selection."""
    if not bank_a:
        raise ValueError("Bank A cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    def eligible(replicates: list[CheckpointMetric]) -> bool:
        if not replicates or not all(
                math.isfinite(checkpoint_score(metric, handoff))
                for metric in replicates):
            return False
        mean_coverage = float(np.mean([
            metric.coverage for metric in replicates]))
        mean_mass_l1 = float(np.mean([
            metric.mass_l1 for metric in replicates]))
        return (mean_coverage >= minimum_coverage and
                mean_mass_l1 <= maximum_mass_l1)

    eligible_a = {
        int(step): replicates for step, replicates in bank_a.items()
        if eligible(replicates)
    }
    if not eligible_a:
        raise RuntimeError("no Bank-A checkpoint passes the frozen guards")
    a_means = {
        step: float(np.mean([checkpoint_score(item, handoff)
                             for item in replicates]))
        for step, replicates in eligible_a.items()
    }
    leaders = tuple(sorted(a_means, key=lambda step: (a_means[step], step))
                    [:min(top_k, len(a_means))])
    missing = [step for step in leaders if step not in bank_b]
    if missing:
        raise ValueError(f"Bank B is missing selected leaders: {missing}")
    # Bank A owns checkpoint eligibility.  Bank B is an independent score
    # confirmation; reapplying noisy coverage/mass thresholds to it can reject
    # every already-eligible leader, including the handoff.  It must still be
    # finite.
    eligible_b = {
        step: bank_b[step] for step in leaders
        if bank_b[step] and all(
            math.isfinite(checkpoint_score(metric, handoff))
            for metric in bank_b[step])
    }
    if not eligible_b:
        raise RuntimeError("no Bank-B leader passes the frozen guards")
    b_means: dict[int, float] = {}
    b_errors: dict[int, float] = {}
    for step, replicates in eligible_b.items():
        scores = np.asarray([checkpoint_score(item, handoff)
                             for item in replicates], dtype=float)
        b_means[step] = float(np.mean(scores))
        b_errors[step] = (float(np.std(scores, ddof=1) /
                                math.sqrt(len(scores)))
                          if len(scores) > 1 else 0.0)
    best = min(b_means, key=lambda step: (b_means[step], step))
    threshold = b_means[best] + b_errors[best]
    selected = min(step for step in b_means
                   if b_means[step] <= threshold + 1e-15)
    return CheckpointSelection(
        selected_step=int(selected),
        bank_a_mean_scores=a_means,
        bank_b_mean_scores=b_means,
        bank_b_standard_errors=b_errors,
        eligible_steps=tuple(sorted(eligible_a)),
        bank_a_leaders=leaders,
        threshold=float(threshold))


def invariant_tests(log=print) -> dict[str, Any]:
    """Exercise proposal, projection, compatibility, and selector contracts."""
    checks: list[str] = []

    # The frozen rank surrogate gradient must agree with finite differences.
    # The field is deliberately held fixed: differentiating through sorting is
    # not part of the implemented stop-gradient objective.
    class _Target:
        d = 1
        scale = 2.0
        means = np.asarray([[-1.0], [1.0]])

    rng = np.random.default_rng(20260721)
    model = TanhMLP(_Target(), "concentrated", 17)
    latent = rng.normal(size=(13, model.latent_dim))
    positive = rng.normal(size=(13, 1))
    x, cache = model.forward(latent, want_cache=True)
    field = exact_rank_field(x, positive, latent)
    gradient = stopgrad_grads(model, cache, field, len(field))

    def frozen_surrogate() -> float:
        return -float(np.sum(model.forward(latent) * field)) / len(field)

    epsilon = 1e-6
    for name, index in (("b3", (0,)), ("W3", (0, 0)),
                        ("b2", (0,)), ("W1", (0, 0))):
        original = float(model.params[name][index])
        model.params[name][index] = original + epsilon
        upper = frozen_surrogate()
        model.params[name][index] = original - epsilon
        lower = frozen_surrogate()
        model.params[name][index] = original
        numerical = (upper - lower) / (2.0 * epsilon)
        np.testing.assert_allclose(
            gradient[name][index], numerical, rtol=2e-5, atol=2e-8)
    checks.append("rank_stopgrad_finite_difference")

    # Pure projection active sets.
    config = GuardConfig(rho=0.2, metric="identity", trust_factor=10.0)
    gq = np.asarray([1.0, 0.0])
    gd = np.asarray([0.0, 1.0])
    dq = np.asarray([-1.0, 0.0])
    dd = np.asarray([-0.5, -1.0])
    inactive = project_guarded_delta(gq, gd, dq, dd,
                                     np.ones(2), config)
    np.testing.assert_array_equal(inactive.delta, dd)
    if inactive.diagnostics.active_set != "none":
        raise AssertionError("inactive projection chose an active constraint")
    checks.append("projection_inactive")

    dd_conflict = np.asarray([0.5, -1.0])
    q_active = project_guarded_delta(gq, gd, dq, dd_conflict,
                                    np.ones(2), config)
    np.testing.assert_allclose(q_active.delta, [-0.2, -1.0], atol=1e-12)
    if q_active.diagnostics.active_set != "quantile":
        raise AssertionError("quantile-only active set was not selected")
    checks.append("projection_quantile_active")

    gq2 = np.asarray([1.0, 0.0])
    gd2 = np.asarray([0.0, 1.0])
    dd_local = np.asarray([-1.0, 1.0])
    local_active = project_guarded_delta(
        gq2, gd2, dq, dd_local, np.ones(2),
        GuardConfig(rho=0.0, metric="identity", trust_factor=10.0))
    np.testing.assert_allclose(local_active.delta, [-1.0, 0.0], atol=1e-12)
    if local_active.diagnostics.active_set != "local":
        raise AssertionError("local-only active set was not selected")
    checks.append("projection_local_active")

    gq3 = np.asarray([1.0, 0.0])
    gd3 = np.asarray([1.0, 1.0])
    both = project_guarded_delta(
        gq3, gd3, np.asarray([-1.0, 0.0]), np.asarray([1.0, 1.0]),
        np.ones(2), config)
    if both.diagnostics.active_set != "quantile+local":
        raise AssertionError("two-active projection was not selected")
    if not both.diagnostics.quantile_feasible or \
            not both.diagnostics.local_feasible:
        raise AssertionError("two-active solution is infeasible")
    checks.append("projection_both_active")

    opposing = project_guarded_delta(
        np.asarray([1.0]), np.asarray([-1.0]), np.asarray([-1.0]),
        np.asarray([1.0]), np.ones(1), config)
    if not opposing.diagnostics.incompatible or \
            not opposing.diagnostics.quantile_feasible:
        raise AssertionError("opposing-gradient fallback contract failed")
    checks.append("projection_opposing_fallback")

    zero = project_guarded_delta(
        np.zeros(2), np.zeros(2), np.zeros(2), np.asarray([0.2, -0.1]),
        np.ones(2), config)
    np.testing.assert_array_equal(zero.delta, np.asarray([0.2, -0.1]))
    checks.append("projection_zero_gradients")

    collinear = project_guarded_delta(
        np.asarray([1.0, 0.0]), np.asarray([2.0, 0.0]),
        np.asarray([-1.0, 0.0]), np.asarray([1.0, 0.0]),
        np.ones(2), config)
    if not collinear.diagnostics.quantile_feasible:
        raise AssertionError("collinear projection lost primary feasibility")
    checks.append("projection_collinear_safe")

    trusted = project_guarded_delta(
        np.asarray([1.0, 0.0]), np.asarray([-0.999, 0.01]),
        np.asarray([-1.0, 0.0]), np.asarray([1.0, 1.0]),
        np.ones(2), GuardConfig(
            rho=0.2, metric="identity", trust_factor=1.0))
    if not trusted.diagnostics.trust_cap_active or \
            not trusted.diagnostics.quantile_feasible:
        raise AssertionError("trust fallback contract failed")
    checks.append("projection_trust_fallback")

    # Adam proposal must match a real cloned step numerically; historical
    # bitwise behavior is separately guarded by lbqcd.invariant_tests.
    proposal = adam_proposal(model, capture_adam_state(model), gradient)
    reference = __import__("copy").deepcopy(model)
    adam_step(reference, gradient)
    predicted = {name: model.params[name] + proposal.delta[name]
                 for name in model.names}
    for name in model.names:
        np.testing.assert_allclose(
            predicted[name], reference.params[name], rtol=0.0, atol=5e-17)
    checks.append("adam_proposal_matches_step")

    dual = initialize_dual_state(model)
    local_before = copy_adam_state(dual.local)
    _ = adam_proposal(model, dual.quantile, gradient)
    if dual.local.step_index != local_before.step_index:
        raise AssertionError("quantile proposal mutated local step index")
    for name in model.names:
        np.testing.assert_array_equal(dual.local.m[name], local_before.m[name])
        np.testing.assert_array_equal(dual.local.v[name], local_before.v[name])
    checks.append("dual_optimizer_state_independence")

    # Explicit disabled guard reproduces the historical carried paper step.
    guarded = __import__("copy").deepcopy(model)
    historical = __import__("copy").deepcopy(model)
    state = initialize_dual_state(guarded)
    local = compute_field(
        historical.forward(latent, want_cache=True)[0], positive,
        tau=0.5, gain="paper", mask=True, on_degenerate="zero").V
    # Recompute the historical cache because the field call above intentionally
    # did not retain it.
    _, historical_cache = historical.forward(latent, want_cache=True)
    historical.stopgrad_step(historical_cache, local)
    guarded_step(
        guarded, state, latent, positive, tau=0.5,
        guard_enabled=False)
    for name in model.names:
        np.testing.assert_array_equal(
            guarded.params[name], historical.params[name])
        np.testing.assert_array_equal(guarded.m[name], historical.m[name])
        np.testing.assert_array_equal(guarded.v[name], historical.v[name])
    if guarded.step_index != historical.step_index:
        raise AssertionError("guard-disabled step index differs")
    checks.append("guard_disabled_paper_bitwise")

    # QGD-v2 projects only the ordinary local proposal, using two current
    # quantile gradients and zero (non-ascent) thresholds.
    safety_config = SafetyGuardConfig(
        minimum_gradient_agreement=0.0,
        minimum_ascent_cosine=0.0,
        minimum_signal_to_null=0.0,
        maximum_correction_ratio=1.0)
    safety_both, safety_label, safety_singular = \
        project_two_batch_nonascent(
            np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0]),
            np.asarray([1.0, 1.0]), np.ones(2), safety_config)
    np.testing.assert_allclose(safety_both, np.zeros(2), atol=1e-12)
    if safety_label != "train+probe" or safety_singular != 0:
        raise AssertionError("two-batch safety projection active set failed")
    checks.append("safety_projection_two_batch")

    safety_inactive, safety_label, _ = project_two_batch_nonascent(
        np.asarray([1.0, 0.0]), np.asarray([0.5, 0.5]),
        np.asarray([-1.0, -1.0]), np.ones(2), safety_config)
    np.testing.assert_array_equal(safety_inactive, np.asarray([-1.0, -1.0]))
    if safety_label != "none":
        raise AssertionError("safety projection changed a feasible proposal")
    checks.append("safety_projection_inactive")

    # An inactive safety guard must preserve the historical paper update
    # bitwise even though it performs two additional read-only diagnostics.
    safety_model = __import__("copy").deepcopy(model)
    safety_reference = __import__("copy").deepcopy(model)
    probe_latent = rng.normal(size=latent.shape)
    probe_positive = rng.normal(size=positive.shape)
    paper_step(safety_reference, latent, positive, 0.5)
    safety_diagnostics = quantile_safety_step(
        safety_model, latent, positive, probe_latent, probe_positive,
        tau=0.5, config=SafetyGuardConfig(
            minimum_signal_to_null=1e100))
    if safety_diagnostics.intervention_applied:
        raise AssertionError("disabled-by-signal safety guard intervened")
    for name in model.names:
        np.testing.assert_array_equal(
            safety_model.params[name], safety_reference.params[name])
        np.testing.assert_array_equal(
            safety_model.m[name], safety_reference.m[name])
        np.testing.assert_array_equal(
            safety_model.v[name], safety_reference.v[name])
    if safety_model.step_index != safety_reference.step_index:
        raise AssertionError("inactive safety step index differs")
    checks.append("safety_inactive_paper_bitwise")

    # Selector is deterministic and chooses the earliest checkpoint within one
    # standard error of the Bank-B best.
    handoff = CheckpointMetric(1.0, 1.0)
    bank_a = {
        0: [CheckpointMetric(1.0, 1.0)] * 2,
        10: [CheckpointMetric(0.8, 0.8)] * 2,
        20: [CheckpointMetric(0.79, 0.79)] * 2,
    }
    bank_b = {
        10: [CheckpointMetric(0.80, 0.80),
             CheckpointMetric(0.82, 0.82)],
        20: [CheckpointMetric(0.79, 0.79),
             CheckpointMetric(0.81, 0.81)],
        0: [CheckpointMetric(1.0, 1.0)] * 2,
    }
    selection = select_checkpoint(bank_a, bank_b, handoff, top_k=3)
    if selection.selected_step != 10:
        raise AssertionError("earliest-within-one-SE selector failed")
    checks.append("checkpoint_selector_earliest_one_se")

    constant = {
        step: [CheckpointMetric(1.0, 1.0)] * 2
        for step in (0, 10, 20)
    }
    constant_selection = select_checkpoint(
        constant, constant, handoff, top_k=3)
    if constant_selection.selected_step != 0:
        raise AssertionError("constant checkpoint trajectory was not earliest")
    checks.append("checkpoint_selector_constant_earliest")

    report = {"status": "pass", "check_count": len(checks),
              "checks": checks}
    log(f"QGD invariants: PASS ({len(checks)} checks)")
    return report


if __name__ == "__main__":
    invariant_tests()
