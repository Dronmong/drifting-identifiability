"""Optimizer and occupancy primitives for the calibrated sharp bridge.

This module does not alter the historical ``TanhMLP`` or baseline Adam
implementation.  Experimental arms call ``adam_step_configurable`` explicitly
and can therefore use an objective-specific learning rate and a global
parameter-step trust cap while retaining auditable Adam state semantics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Callable

import numpy as np

from conservative_finishers import (
    FieldResult,
    pairwise_displacements,
    sharp_laplace_kernel_from_radius,
    sharp_laplace_field,
)
from lbqcd import _adam_step, _stopgrad_grads
from run_identifiability_generator import (
    ADAM_EPS,
    ADAM_LR,
    BETA1,
    BETA2,
    TanhMLP,
)


PARAMETER_ORDER = tuple(TanhMLP.names)


@dataclass(frozen=True)
class OptimizerState:
    step_index: int
    m: dict[str, np.ndarray]
    v: dict[str, np.ndarray]


@dataclass(frozen=True)
class AdamStepDiagnostics:
    learning_rate: float
    pre_cap_step_norm: float
    committed_step_norm: float
    maximum_parameter_step: float | None
    cap_activated: bool
    descent_step_cosine: float
    parameter_deltas: dict[str, np.ndarray]


@dataclass(frozen=True)
class SharpBridgeStep:
    field_result: FieldResult
    gradients: dict[str, np.ndarray]
    adam: AdamStepDiagnostics
    occupancy: dict[str, float]


def clone_model(model: TanhMLP) -> TanhMLP:
    clone = copy.deepcopy(model)
    for name in PARAMETER_ORDER:
        for section in ("params", "m", "v"):
            left = getattr(model, section)[name]
            right = getattr(clone, section)[name]
            if np.shares_memory(left, right):
                raise AssertionError(
                    f"cloned {section}.{name} shares mutable storage")
    return clone


def capture_optimizer(model: TanhMLP) -> OptimizerState:
    return OptimizerState(
        step_index=int(model.step_index),
        m={name: model.m[name].copy() for name in PARAMETER_ORDER},
        v={name: model.v[name].copy() for name in PARAMETER_ORDER})


def restore_optimizer(model: TanhMLP, state: OptimizerState) -> None:
    model.step_index = int(state.step_index)
    model.m = {name: state.m[name].copy() for name in PARAMETER_ORDER}
    model.v = {name: state.v[name].copy() for name in PARAMETER_ORDER}


def optimizer_matches(model: TanhMLP, state: OptimizerState) -> bool:
    """Return whether a model's Adam buffers exactly equal a snapshot."""
    return (model.step_index == state.step_index and all(
        np.array_equal(model.m[name], state.m[name]) and
        np.array_equal(model.v[name], state.v[name])
        for name in PARAMETER_ORDER))


def reset_optimizer(model: TanhMLP) -> None:
    model.step_index = 0
    model.m = {name: np.zeros_like(model.params[name])
               for name in PARAMETER_ORDER}
    model.v = {name: np.zeros_like(model.params[name])
               for name in PARAMETER_ORDER}


def flatten_named(values: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.ravel(values[name]) for name in PARAMETER_ORDER])


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.ravel(left)
    right = np.ravel(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def adam_step_configurable(
        model: TanhMLP, gradients: dict[str, np.ndarray], *,
        learning_rate: float,
        maximum_parameter_step: float | None = None) -> AdamStepDiagnostics:
    """Apply Adam with an optional cap on the flattened parameter delta.

    With the repository learning rate and no cap, arithmetic follows
    ``lbqcd._adam_step`` exactly.  The cap is applied as one global positive
    scalar to the bias-corrected Adam displacement, preserving its direction.
    """
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    if maximum_parameter_step is not None:
        maximum_parameter_step = float(maximum_parameter_step)
        if (not math.isfinite(maximum_parameter_step) or
                maximum_parameter_step <= 0.0):
            raise ValueError(
                "maximum_parameter_step must be finite and positive")
    if not all(np.all(np.isfinite(gradients[name]))
               for name in PARAMETER_ORDER):
        raise FloatingPointError("non-finite generator gradient")

    model.step_index += 1
    step_index = model.step_index
    proposed: dict[str, np.ndarray] = {}
    for name in PARAMETER_ORDER:
        gradient = gradients[name]
        model.m[name] = BETA1 * model.m[name] + (1.0 - BETA1) * gradient
        model.v[name] = (BETA2 * model.v[name] +
                         (1.0 - BETA2) * gradient * gradient)
        mhat = model.m[name] / (1.0 - BETA1 ** step_index)
        vhat = model.v[name] / (1.0 - BETA2 ** step_index)
        proposed[name] = (-learning_rate * mhat /
                          (np.sqrt(vhat) + ADAM_EPS))

    flat_proposed = flatten_named(proposed)
    pre_cap_norm = float(np.linalg.norm(flat_proposed))
    cap_activated = (maximum_parameter_step is not None and
                     pre_cap_norm > maximum_parameter_step)
    scale = (maximum_parameter_step / pre_cap_norm
             if cap_activated else 1.0)
    committed: dict[str, np.ndarray] = {}
    for name in PARAMETER_ORDER:
        if scale == 1.0:
            # Keep this algebra identical to the historical subtraction.
            update = -proposed[name]
            model.params[name] -= update
            committed[name] = proposed[name]
        else:
            committed[name] = proposed[name] * scale
            model.params[name] += committed[name]

    flat_committed = flatten_named(committed)
    gradient_flat = flatten_named(gradients)
    return AdamStepDiagnostics(
        learning_rate=learning_rate,
        pre_cap_step_norm=pre_cap_norm,
        committed_step_norm=float(np.linalg.norm(flat_committed)),
        maximum_parameter_step=maximum_parameter_step,
        cap_activated=bool(cap_activated),
        descent_step_cosine=cosine(-gradient_flat, flat_committed),
        parameter_deltas=committed)


def _row_weight_diagnostics(weights: np.ndarray,
                            mask: np.ndarray) -> dict[str, np.ndarray]:
    retained = np.where(mask, weights, 0.0)
    mass = retained.sum(axis=1)
    square_mass = (retained * retained).sum(axis=1)
    ess = np.divide(
        mass * mass, square_mass,
        out=np.zeros_like(mass), where=square_mass > 0.0)
    maximum = retained.max(axis=1)
    max_share = np.divide(
        maximum, mass, out=np.ones_like(mass), where=mass > 0.0)
    threshold = math.exp(-3.0) * maximum
    neighbor_count = np.sum(
        mask & (weights >= threshold[:, None]), axis=1).astype(float)
    return {
        "mass": mass,
        "ess": ess,
        "max_share": max_share,
        "neighbor_count": neighbor_count,
    }


def _summarize_rows(prefix: str,
                    diagnostics: dict[str, np.ndarray]) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, values in diagnostics.items():
        values = np.asarray(values, dtype=float)
        out[f"{prefix}_{name}_min"] = float(np.min(values))
        for label, quantile in (("p01", 0.01), ("p10", 0.10),
                                ("p50", 0.50), ("p90", 0.90)):
            out[f"{prefix}_{name}_{label}"] = float(
                np.quantile(values, quantile))
    return out


def sharp_deleted_occupancy(
        queries: np.ndarray, positives: np.ndarray, tau: float) \
        -> dict[str, float]:
    """Return local-mass/ESS diagnostics for the sharp deleted estimator."""
    _, positive_radius = pairwise_displacements(queries, positives)
    _, negative_radius = pairwise_displacements(queries, queries)
    positive_weights = sharp_laplace_kernel_from_radius(
        positive_radius, tau)
    negative_weights = sharp_laplace_kernel_from_radius(
        negative_radius, tau)
    positive_mask = np.ones_like(positive_weights, dtype=bool)
    negative_mask = np.ones_like(negative_weights, dtype=bool)
    if len(queries) < 2:
        raise ValueError("deleted occupancy requires at least two queries")
    np.fill_diagonal(negative_mask, False)
    positive = _row_weight_diagnostics(positive_weights, positive_mask)
    negative = _row_weight_diagnostics(negative_weights, negative_mask)
    return {
        **_summarize_rows("positive", positive),
        **_summarize_rows("negative", negative),
    }


def sharp_deleted_bridge_step(
        model: TanhMLP, latent: np.ndarray, positives: np.ndarray, *,
        tau: float, learning_rate: float,
        maximum_parameter_step: float | None) -> SharpBridgeStep:
    queries, cache = model.forward(latent, want_cache=True)
    result = sharp_laplace_field(
        queries, positives, tau=tau, reference_mode="reused_deleted")
    gradients = _stopgrad_grads(model, cache, result.field, len(result.field))
    occupancy = sharp_deleted_occupancy(queries, positives, tau)
    adam = adam_step_configurable(
        model, gradients, learning_rate=learning_rate,
        maximum_parameter_step=maximum_parameter_step)
    return SharpBridgeStep(result, gradients, adam, occupancy)


def _models_equal(left: TanhMLP, right: TanhMLP) -> bool:
    return (left.step_index == right.step_index and all(
        np.array_equal(left.params[name], right.params[name]) and
        np.array_equal(left.m[name], right.m[name]) and
        np.array_equal(left.v[name], right.v[name])
        for name in PARAMETER_ORDER))


def invariant_tests(log: Callable[[str], None] = print) -> dict[str, object]:
    checks: list[str] = []
    target = SimpleNamespace(
        d=1, scale=2.0, means=np.asarray([[-1.0], [1.0]]))
    rng = np.random.default_rng(20260722)
    gradients: dict[str, np.ndarray]

    baseline = TanhMLP(target, "missing", 11)
    exact = clone_model(baseline)
    configured = clone_model(baseline)
    gradients = {name: rng.normal(size=baseline.params[name].shape)
                 for name in PARAMETER_ORDER}
    _adam_step(exact, {name: value.copy()
                       for name, value in gradients.items()})
    diagnostics = adam_step_configurable(
        configured, {name: value.copy()
                     for name, value in gradients.items()},
        learning_rate=ADAM_LR)
    if not _models_equal(exact, configured):
        raise AssertionError("configurable Adam changed the base no-cap step")
    checks.append("base_no_cap_bitwise")

    unit = clone_model(baseline)
    quarter = clone_model(baseline)
    unit_diag = adam_step_configurable(
        unit, gradients, learning_rate=ADAM_LR)
    quarter_diag = adam_step_configurable(
        quarter, gradients, learning_rate=0.25 * ADAM_LR)
    np.testing.assert_allclose(
        quarter_diag.pre_cap_step_norm,
        0.25 * unit_diag.pre_cap_step_norm, rtol=2e-15, atol=2e-15)
    checks.append("learning_rate_scales_first_step")

    capped = clone_model(baseline)
    requested = 0.1 * unit_diag.pre_cap_step_norm
    capped_diag = adam_step_configurable(
        capped, gradients, learning_rate=ADAM_LR,
        maximum_parameter_step=requested)
    if not capped_diag.cap_activated:
        raise AssertionError("expected trust cap did not activate")
    np.testing.assert_allclose(
        capped_diag.committed_step_norm, requested,
        rtol=2e-14, atol=2e-14)
    checks.append("active_cap_has_requested_norm")

    inactive = clone_model(baseline)
    inactive_diag = adam_step_configurable(
        inactive, gradients, learning_rate=ADAM_LR,
        maximum_parameter_step=10.0 * unit_diag.pre_cap_step_norm)
    if inactive_diag.cap_activated or not _models_equal(unit, inactive):
        raise AssertionError("inactive cap changed the Adam update")
    checks.append("inactive_cap_is_bitwise_noop")

    if not diagnostics.descent_step_cosine > 0.0:
        raise AssertionError("first configurable step is not a descent step")
    checks.append("first_step_descent_alignment")

    state_source = clone_model(exact)
    saved = capture_optimizer(state_source)
    state_target = clone_model(state_source)
    reset_optimizer(state_target)
    adam_step_configurable(
        state_target, gradients, learning_rate=0.25 * ADAM_LR)
    restore_optimizer(state_target, saved)
    if not optimizer_matches(state_target, saved):
        raise AssertionError("optimizer save/restore is not lossless")
    checks.append("optimizer_state_roundtrip")

    isolated = clone_model(state_source)
    saved_isolated = capture_optimizer(isolated)
    working = clone_model(isolated)
    reset_optimizer(working)
    adam_step_configurable(
        working, gradients, learning_rate=0.25 * ADAM_LR)
    if any(not np.array_equal(saved_isolated.m[name], isolated.m[name]) or
           not np.array_equal(saved_isolated.v[name], isolated.v[name])
           for name in PARAMETER_ORDER):
        raise AssertionError("bridge update mutated saved paper state")
    checks.append("saved_state_isolated")

    queries = np.asarray([[-1.0], [0.0], [1.0], [2.0]])
    positives = queries.copy()
    occupancy = sharp_deleted_occupancy(queries, positives, tau=1e6)
    np.testing.assert_allclose(
        occupancy["positive_ess_p50"], len(positives),
        rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        occupancy["negative_ess_p50"], len(queries) - 1,
        rtol=2e-12, atol=2e-12)
    checks.append("uniform_weight_ess")

    bridge_model = TanhMLP(target, "concentrated", 13)
    latent = rng.normal(size=(8, bridge_model.latent_dim))
    positive = rng.normal(size=(9, 1))
    bridge = sharp_deleted_bridge_step(
        bridge_model, latent, positive, tau=0.5,
        learning_rate=0.25 * ADAM_LR,
        maximum_parameter_step=1e-3)
    if (bridge.field_result.diagnostics.floor_activations != 0 or
            not bridge.adam.cap_activated or
            not math.isfinite(bridge.occupancy["negative_ess_min"])):
        raise AssertionError("integrated bridge-step invariant failed")
    checks.append("integrated_bridge_step")

    report: dict[str, object] = {
        "status": "pass",
        "check_count": len(checks),
        "checks": checks,
    }
    log(f"calibrated bridge invariants: PASS ({len(checks)} checks)")
    return report


if __name__ == "__main__":
    invariant_tests()
