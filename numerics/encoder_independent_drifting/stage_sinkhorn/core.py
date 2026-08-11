"""Auditable two-batch Sinkhorn drift primitives.

The primary generated support is the only differentiable role.  The real and
self supports, transport plans, barycentric velocity, and drifted target are
detached.  This matches the frozen-field regression used by Wasserstein
gradient-flow drifting and deliberately does not differentiate through the
Sinkhorn iterations.

All costs in this module are dimensionless: callers divide the raw quadratic
pixel cost by a positive, target-only frozen ``cost_scale``.  ``epsilon`` is
therefore meaningful only together with that scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SinkhornConfig:
    """Numerical and estimator choices subject to outcome-blind preflight."""

    epsilon: float = 0.10
    relative_tolerance: float = 1e-3
    max_iterations: int = 100
    min_iterations: int = 2
    check_every: int = 1
    eta: float = 0.05
    correction_every: int = 10
    primary_batch: int = 64
    self_batch: int = 64
    real_batch: int = 128
    correction_nfe: int = 8
    event_gradient_ratio: float = 0.25

    def validate(self) -> None:
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("Sinkhorn epsilon must be positive and finite")
        if not math.isfinite(self.relative_tolerance) or self.relative_tolerance <= 0:
            raise ValueError("Sinkhorn tolerance must be positive and finite")
        if self.max_iterations <= 0:
            raise ValueError("Sinkhorn iteration cap must be positive")
        if not 1 <= self.min_iterations <= self.max_iterations:
            raise ValueError("invalid Sinkhorn minimum iteration count")
        if self.check_every <= 0:
            raise ValueError("Sinkhorn residual cadence must be positive")
        if not math.isfinite(self.eta) or self.eta <= 0:
            raise ValueError("Sinkhorn velocity step must be positive and finite")
        if self.correction_every <= 0 or self.correction_nfe <= 0:
            raise ValueError("correction cadence and NFE must be positive")
        if min(self.primary_batch, self.self_batch, self.real_batch) < 2:
            raise ValueError("every Sinkhorn empirical role needs at least two samples")
        if (
            not math.isfinite(self.event_gradient_ratio)
            or self.event_gradient_ratio <= 0
        ):
            raise ValueError("event gradient ratio must be positive and finite")


@dataclass(frozen=True)
class SinkhornPlan:
    """A balanced empirical plan and convergence diagnostics."""

    plan: Tensor
    row_marginal: Tensor
    column_marginal: Tensor
    iterations: int
    converged: bool
    max_relative_row_error: float
    max_relative_column_error: float

    @property
    def maximum_relative_error(self) -> float:
        return max(self.max_relative_row_error, self.max_relative_column_error)

    def diagnostics(self) -> dict:
        return {
            "iterations": self.iterations,
            "converged": self.converged,
            "max_relative_row_error": self.max_relative_row_error,
            "max_relative_column_error": self.max_relative_column_error,
            "maximum_relative_error": self.maximum_relative_error,
            "iteration_cap_hit": bool(not self.converged),
        }


def _solver_dtype(value: Tensor) -> torch.dtype:
    if value.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if value.dtype in (torch.float32, torch.float64):
        return value.dtype
    raise TypeError("Sinkhorn costs must use a floating dtype")


def _positive_marginal(
    value: Tensor | None,
    size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    if value is None:
        result = torch.full((size,), 1.0 / size, device=device, dtype=dtype)
    else:
        if value.shape != (size,):
            raise ValueError(f"{name} marginal has the wrong shape")
        result = value.detach().to(device=device, dtype=dtype)
    if not bool(torch.isfinite(result).all()) or not bool((result > 0).all()):
        raise ValueError(f"{name} marginal must be positive and finite")
    total = result.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0:
        raise ValueError(f"{name} marginal has invalid total mass")
    result = result / total
    return result


def _relative_marginal_errors(
    log_plan: Tensor, row: Tensor, column: Tensor
) -> tuple[float, float]:
    # Compute masses in log space before exponentiating. This makes the
    # convergence check reliable even when individual plan entries underflow.
    actual_row = torch.exp(torch.logsumexp(log_plan, dim=1))
    actual_column = torch.exp(torch.logsumexp(log_plan, dim=0))
    row_error = ((actual_row - row).abs() / row).max()
    column_error = ((actual_column - column).abs() / column).max()
    return float(row_error), float(column_error)


def log_sinkhorn_plan(
    cost: Tensor,
    config: SinkhornConfig,
    *,
    row_marginal: Tensor | None = None,
    column_marginal: Tensor | None = None,
    require_convergence: bool = True,
) -> SinkhornPlan:
    """Solve a balanced entropic OT problem in the log domain.

    The convention is ``<pi,C> + epsilon * KL(pi || a tensor b)``.  The
    additive constants in KL do not affect the optimizer.  This function is a
    plan solver, not a differentiable Sinkhorn layer: the cost and returned
    plan are detached by design.
    """

    config.validate()
    if cost.ndim != 2 or min(cost.shape) <= 0:
        raise ValueError("Sinkhorn cost must be a nonempty matrix")
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("Sinkhorn cost must be finite")
    dtype = _solver_dtype(cost)
    detached_cost = cost.detach().to(dtype=dtype)
    rows, columns = detached_cost.shape
    row = _positive_marginal(
        row_marginal,
        rows,
        device=detached_cost.device,
        dtype=dtype,
        name="row",
    )
    column = _positive_marginal(
        column_marginal,
        columns,
        device=detached_cost.device,
        dtype=dtype,
        name="column",
    )
    log_row = row.log()
    log_column = column.log()
    log_kernel = -detached_cost / config.epsilon
    log_u = torch.zeros_like(row)
    log_v = torch.zeros_like(column)
    converged = False
    row_error = column_error = float("inf")
    iterations = 0

    with torch.no_grad():
        for iteration in range(1, config.max_iterations + 1):
            log_u = log_row - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
            log_v = log_column - torch.logsumexp(log_kernel + log_u.unsqueeze(1), dim=0)

            # The dual potentials have a one-dimensional gauge freedom. Keep
            # them centered to avoid needless growth without changing pi.
            shift = log_u.mean()
            log_u = log_u - shift
            log_v = log_v + shift
            iterations = iteration
            should_check = iteration >= config.min_iterations and (
                iteration % config.check_every == 0
                or iteration == config.max_iterations
            )
            if should_check:
                log_plan = log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0)
                row_error, column_error = _relative_marginal_errors(
                    log_plan, row, column
                )
                if max(row_error, column_error) <= config.relative_tolerance:
                    converged = True
                    break

        log_plan = log_kernel + log_u.unsqueeze(1) + log_v.unsqueeze(0)
        if not math.isfinite(row_error) or not math.isfinite(column_error):
            row_error, column_error = _relative_marginal_errors(log_plan, row, column)
        plan = torch.exp(log_plan)

    if not bool(torch.isfinite(plan).all()):
        raise FloatingPointError("Sinkhorn plan is non-finite")
    result = SinkhornPlan(
        plan=plan,
        row_marginal=row,
        column_marginal=column,
        iterations=iterations,
        converged=converged,
        max_relative_row_error=row_error,
        max_relative_column_error=column_error,
    )
    if require_convergence and not result.converged:
        raise RuntimeError(
            "Sinkhorn plan did not meet the marginal tolerance: "
            f"row={row_error:.3e}, column={column_error:.3e}, "
            f"iterations={iterations}"
        )
    return result


def _flat_samples(value: Tensor, name: str) -> Tensor:
    if value.ndim < 2 or len(value) == 0:
        raise ValueError(f"{name} samples need a batch and data dimension")
    if not value.dtype.is_floating_point:
        raise TypeError(f"{name} samples must use a floating dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} samples must be finite")
    return value.reshape(len(value), -1)


def quadratic_cost(left: Tensor, right: Tensor, cost_scale: float) -> Tensor:
    """Return ``||left-right||^2 / (2*cost_scale)``."""

    if not math.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("quadratic cost scale must be positive and finite")
    left_flat = _flat_samples(left, "left")
    right_flat = _flat_samples(right, "right")
    if left_flat.shape[1] != right_flat.shape[1]:
        raise ValueError("quadratic cost supports have different dimensions")
    if left.device != right.device:
        raise ValueError("quadratic cost supports must share a device")
    return torch.cdist(left_flat, right_flat).square() / (2.0 * cost_scale)


def target_cost_scale(samples: Tensor) -> float:
    """Median off-diagonal target quadratic cost.

    Callers are responsible for selecting the target-only calibration sample
    before any candidate outcome is available.
    """

    flat = _flat_samples(samples, "target calibration")
    if len(flat) < 2:
        raise ValueError("cost-scale calibration needs at least two targets")
    distances = torch.pdist(flat).square() / 2.0
    if not bool(torch.isfinite(distances).all()):
        raise ValueError("target pairwise costs are non-finite")
    scale = float(distances.median())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("target samples do not define a positive cost scale")
    return scale


def _conditional_weights(result: SinkhornPlan) -> Tensor:
    weights = result.plan / result.row_marginal.unsqueeze(1)
    if not bool(torch.isfinite(weights).all()):
        raise FloatingPointError("conditional Sinkhorn weights are non-finite")
    return weights


def _barycenter(result: SinkhornPlan, support: Tensor) -> Tensor:
    flat = _flat_samples(support, "barycenter support")
    if result.plan.shape[1] != len(flat):
        raise ValueError("plan and barycenter support have incompatible sizes")
    weights = _conditional_weights(result).to(dtype=flat.dtype)
    return (weights @ flat).reshape((result.plan.shape[0],) + support.shape[1:])


def _plan_health(result: SinkhornPlan) -> dict:
    weights = _conditional_weights(result)
    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    entropy = -(weights * safe.log()).sum(dim=1)
    if weights.shape[1] > 1:
        entropy = entropy / math.log(weights.shape[1])
    return {
        **result.diagnostics(),
        "conditional_entropy_mean": float(entropy.mean()),
        "conditional_entropy_minimum": float(entropy.min()),
        "conditional_max_weight_mean": float(weights.max(dim=1).values.mean()),
        "conditional_max_weight_maximum": float(weights.max()),
        "plan_minimum": float(result.plan.min()),
        "plan_maximum": float(result.plan.max()),
    }


def sinkhorn_velocity(
    primary: Tensor,
    real_support: Tensor,
    self_support: Tensor,
    cost_scale: float,
    config: SinkhornConfig,
) -> tuple[Tensor, dict]:
    """Compute a detached cross-minus-self barycentric velocity."""

    config.validate()
    if (
        primary.shape[1:] != real_support.shape[1:]
        or primary.shape[1:] != self_support.shape[1:]
    ):
        raise ValueError("Sinkhorn empirical roles have different data shapes")
    if not (primary.device == real_support.device == self_support.device):
        raise ValueError("Sinkhorn empirical roles must share a device")
    if primary.data_ptr() == self_support.data_ptr():
        raise ValueError("self support must be a distinct generated batch")

    with torch.no_grad():
        primary_detached = primary.detach()
        real_detached = real_support.detach()
        self_detached = self_support.detach()
        cross_cost = quadratic_cost(primary_detached, real_detached, cost_scale)
        self_cost = quadratic_cost(primary_detached, self_detached, cost_scale)
        cross = log_sinkhorn_plan(cross_cost, config)
        self_plan = log_sinkhorn_plan(self_cost, config)
        target_barycenter = _barycenter(cross, real_detached)
        self_barycenter = _barycenter(self_plan, self_detached)
        velocity = (target_barycenter - self_barycenter).detach()
        velocity_norm = velocity.flatten(1).norm(dim=1)
        sample_norm = primary_detached.flatten(1).norm(dim=1)
        health = {
            "cross": _plan_health(cross),
            "self": _plan_health(self_plan),
            "cost_scale": float(cost_scale),
            "epsilon": config.epsilon,
            "eta": config.eta,
            "primary_batch": len(primary),
            "real_batch": len(real_support),
            "self_batch": len(self_support),
            "velocity_l2_mean": float(velocity_norm.mean()),
            "velocity_l2_rms": float(velocity_norm.square().mean().sqrt()),
            "sample_l2_rms": float(sample_norm.square().mean().sqrt()),
            "update_to_sample_rms": float(
                config.eta
                * velocity_norm.square().mean().sqrt()
                / sample_norm.square().mean().sqrt().clamp_min(1e-30)
            ),
            "gradient_roles": ["primary_generated_endpoints"],
            "detached_roles": [
                "real_support",
                "independent_self_support",
                "transport_plans",
                "barycentric_velocity",
                "drifted_target",
            ],
        }
    return velocity, health


def sinkhorn_drifted_target_loss(
    primary: Tensor,
    real_support: Tensor,
    self_support: Tensor,
    cost_scale: float,
    config: SinkhornConfig,
) -> tuple[Tensor, dict]:
    """Mean squared L2 regression toward a detached Sinkhorn drift target."""

    velocity, health = sinkhorn_velocity(
        primary, real_support, self_support, cost_scale, config
    )
    target = (primary.detach() + config.eta * velocity).detach()
    residual = (primary - target).flatten(1)
    loss = residual.square().sum(dim=1).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Sinkhorn drifted-target loss is non-finite")
    health = {
        **health,
        "loss": float(loss.detach()),
        "target_requires_grad": bool(target.requires_grad),
    }
    return loss, health


def _regularized_ot_value(cost: Tensor, result: SinkhornPlan, epsilon: float) -> Tensor:
    """Evaluate the declared entropic objective with a frozen optimal plan."""

    plan = result.plan.detach().to(dtype=cost.dtype)
    row = result.row_marginal.detach().to(dtype=cost.dtype)
    column = result.column_marginal.detach().to(dtype=cost.dtype)
    reference = row.unsqueeze(1) * column.unsqueeze(0)
    tiny = torch.finfo(plan.dtype).tiny
    kl = (plan * (plan.clamp_min(tiny).log() - reference.clamp_min(tiny).log())).sum()
    return (plan * cost).sum() + epsilon * kl


def empirical_cross_self_energy(
    primary: Tensor,
    real_support: Tensor,
    self_support: Tensor,
    cost_scale: float,
    config: SinkhornConfig,
) -> Tensor:
    """Two-batch finite energy whose envelope gradient yields the velocity.

    This is ``OT_epsilon(primary, real) - OT_epsilon(primary, self)``.  The
    target-only self term of the full Sinkhorn divergence is constant with
    respect to ``primary`` and is intentionally omitted.
    """

    if primary.data_ptr() == self_support.data_ptr():
        raise ValueError("self support must be distinct in the empirical energy")
    cross_cost = quadratic_cost(primary, real_support.detach(), cost_scale)
    self_cost = quadratic_cost(primary, self_support.detach(), cost_scale)
    cross = log_sinkhorn_plan(cross_cost.detach(), config)
    self_plan = log_sinkhorn_plan(self_cost.detach(), config)
    return _regularized_ot_value(
        cross_cost, cross, config.epsilon
    ) - _regularized_ot_value(self_cost, self_plan, config.epsilon)
