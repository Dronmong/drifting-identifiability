"""Adaptive kernel mixture with cross-fitting (plan section 6.4).

Weights live on the simplex with a hard floor,

    alpha_j >= alpha_min,   sum_j alpha_j = 1,

so no branch can be silently switched off and the collapse-to-one-branch
stop condition (plan section 10.4) is checkable rather than rhetorical.

The utility statistic is the plan's first adaptive rule,

    u_j = |V_j|^2 / (Var(V_j) + eps),

i.e. the cross-fit drift signal-to-noise ratio computed in
``kernel_gradient.field_with_snr``.  Weights follow an EMA of a softmax of
the utilities and are then projected back onto the floored simplex.

Cross-fitting is enforced structurally, not by convention.  :class:`BatchRoles`
carries three disjoint index sets -- controller, field, audit -- and
:func:`assert_disjoint` refuses overlapping roles.  The controller may only
see controller examples; the generator target is computed on field examples;
diagnostics are logged on audit examples.  Estimating alpha and evaluating
the resulting drift on the same examples is exactly how an adaptive rule
overfits minibatch noise into a false development gain.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import torch

from .config import MixtureConfig


def project_simplex_with_floor(values: torch.Tensor,
                               floor: float) -> torch.Tensor:
    """Euclidean projection onto ``{a : a >= floor, sum a = 1}``."""
    n = len(values)
    if n == 0:
        raise ValueError("cannot project an empty weight vector")
    if floor < 0:
        raise ValueError("the floor must be nonnegative")
    if floor * n > 1.0 + 1e-12:
        raise ValueError(
            f"floor {floor} is infeasible for {n} branches (n*floor > 1)")
    mass = 1.0 - floor * n
    shifted = (values - floor).to(torch.float64)
    if mass <= 0:
        return torch.full_like(values, 1.0 / n)
    # Projection onto {w >= 0, sum w = mass} (Duchi et al., 2008).
    sorted_values, _ = torch.sort(shifted, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - mass
    indices = torch.arange(1, n + 1, dtype=torch.float64)
    condition = sorted_values - cumulative / indices > 0
    rho = int(torch.nonzero(condition).max()) if bool(condition.any()) else 0
    theta = cumulative[rho] / (rho + 1)
    projected = (shifted - theta).clamp_min(0.0)
    return (projected + floor).to(values.dtype)


@dataclass
class BatchRoles:
    """Disjoint example index sets for the three cross-fitting roles."""

    controller: np.ndarray
    field: np.ndarray
    audit: np.ndarray

    def assert_disjoint(self) -> None:
        pairs = (("controller", "field"), ("controller", "audit"),
                 ("field", "audit"))
        for left, right in pairs:
            shared = np.intersect1d(getattr(self, left), getattr(self, right))
            if len(shared):
                raise ValueError(
                    f"cross-fitting violated: {left} and {right} share "
                    f"{len(shared)} examples")

    def sizes(self) -> dict[str, int]:
        return {"controller": len(self.controller), "field": len(self.field),
                "audit": len(self.audit)}


def split_roles(total: int, controller: int, field: int, audit: int,
                rng: np.random.Generator) -> BatchRoles:
    """Partition ``range(total)`` into three disjoint role index sets."""
    needed = controller + field + audit
    if needed > total:
        raise ValueError(
            f"cross-fitting needs {needed} disjoint examples but only "
            f"{total} are available")
    order = rng.permutation(total)
    roles = BatchRoles(
        controller=np.sort(order[:controller]),
        field=np.sort(order[controller:controller + field]),
        audit=np.sort(order[controller + field:needed]),
    )
    roles.assert_disjoint()
    return roles


@dataclass
class MixtureController:
    """EMA softmax controller over branch utilities.

    With ``config.adaptive == False`` the weights never move, so an
    adaptive run with adaptation disabled reproduces the fixed-weight run
    exactly -- asserted by ``tests/test_crossfit_controller.py``.
    """

    branch_names: tuple[str, ...]
    config: MixtureConfig
    weights: torch.Tensor = dataclass_field(default=None)  # type: ignore
    history: list[dict] = dataclass_field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.branch_names) != len(set(self.branch_names)):
            raise ValueError(
                "duplicate branch names would silently double a branch's "
                f"effective weight: {self.branch_names}")
        if self.weights is None:
            n = len(self.branch_names)
            self.weights = torch.full((n,), 1.0 / n, dtype=torch.float32)
            if self.config.floor * n > 1.0:
                raise ValueError("mixture floor infeasible for this branch set")

    def as_dict(self) -> dict[str, float]:
        return {name: float(w)
                for name, w in zip(self.branch_names, self.weights)}

    def update(self, utilities: dict[str, float], step: int) -> dict:
        """One controller update from CONTROLLER-batch utilities only."""
        missing = set(self.branch_names) - set(utilities)
        if missing:
            raise ValueError(f"controller is missing utilities for {missing}")
        raw = torch.tensor([float(utilities[n]) for n in self.branch_names],
                           dtype=torch.float64)
        if not torch.isfinite(raw).all():
            raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        record = {"step": step, "utilities": dict(utilities),
                  "before": self.as_dict()}
        if not self.config.adaptive:
            record["after"] = self.as_dict()
            record["changed"] = False
            self.history.append(record)
            return record
        if self.config.temperature <= 0:
            raise ValueError("controller temperature must be positive")
        scaled = raw / self.config.temperature
        target = torch.softmax(scaled - scaled.max(), dim=0)
        beta = self.config.ema
        if not 0.0 < beta <= 1.0:
            raise ValueError("controller EMA beta must lie in (0, 1]")
        blended = (1.0 - beta) * self.weights.to(torch.float64) + beta * target
        self.weights = project_simplex_with_floor(
            blended.to(torch.float32), self.config.floor)
        record["after"] = self.as_dict()
        record["changed"] = True
        self.history.append(record)
        return record

    def diagnostics(self) -> dict:
        """Plan section 10.3 / 10.4: is the mixture collapsing?"""
        weights = self.weights.to(torch.float64)
        entropy = float(-(weights * torch.log(weights.clamp_min(1e-30))).sum())
        return {
            "mixture_weights": self.as_dict(),
            "mixture_entropy": entropy,
            "mixture_entropy_fraction": entropy / float(
                np.log(len(self.branch_names))) if len(
                    self.branch_names) > 1 else 1.0,
            "mixture_max_weight": float(weights.max()),
            "mixture_min_weight": float(weights.min()),
            "mixture_floor_active": bool(
                float(weights.min()) <= self.config.floor + 1e-6),
            "mixture_updates": len(self.history),
        }
