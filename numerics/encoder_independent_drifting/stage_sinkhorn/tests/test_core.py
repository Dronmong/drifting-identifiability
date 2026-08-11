"""Mathematical and gradient-path tests for Stage S0.1--S0.2."""

from __future__ import annotations

from dataclasses import replace

import torch

from ...tests.harness import main
from ..core import (
    SinkhornConfig,
    empirical_cross_self_energy,
    log_sinkhorn_plan,
    sinkhorn_drifted_target_loss,
    sinkhorn_velocity,
    target_cost_scale,
)


def _strict_config(**changes) -> SinkhornConfig:
    base = SinkhornConfig(
        epsilon=0.7,
        relative_tolerance=1e-10,
        max_iterations=2_000,
        min_iterations=2,
        check_every=1,
        eta=0.2,
        correction_every=1,
        primary_batch=2,
        self_batch=2,
        real_batch=3,
        correction_nfe=1,
    )
    result = replace(base, **changes)
    result.validate()
    return result


def test_rectangular_plan_meets_both_uniform_marginals() -> None:
    generator = torch.Generator().manual_seed(17)
    cost = torch.rand(4, 7, generator=generator, dtype=torch.float64) * 3
    result = log_sinkhorn_plan(cost, _strict_config())
    assert result.converged
    assert result.plan.shape == (4, 7)
    assert torch.allclose(
        result.plan.sum(dim=1),
        torch.full((4,), 0.25, dtype=torch.float64),
        rtol=1e-9,
        atol=1e-11,
    )
    assert torch.allclose(
        result.plan.sum(dim=0),
        torch.full((7,), 1 / 7, dtype=torch.float64),
        rtol=1e-9,
        atol=1e-11,
    )


def test_constant_cost_has_uniform_product_plan() -> None:
    cost = torch.full((3, 5), 100_000.0, dtype=torch.float64)
    result = log_sinkhorn_plan(cost, _strict_config())
    expected = torch.full((3, 5), 1 / 15, dtype=torch.float64)
    assert torch.allclose(result.plan, expected, rtol=1e-9, atol=1e-11)


def test_identical_cross_and_self_problems_have_zero_velocity() -> None:
    primary = torch.tensor([[[[-1.0]]], [[[1.0]]]], dtype=torch.float64)
    support = primary.detach().clone()
    velocity, health = sinkhorn_velocity(
        primary,
        support.clone(),
        support.clone(),
        cost_scale=1.0,
        config=_strict_config(real_batch=2),
    )
    assert torch.equal(velocity, torch.zeros_like(velocity))
    assert health["cross"]["converged"]
    assert health["self"]["converged"]


def test_gradient_flows_only_through_primary_support() -> None:
    primary = torch.tensor(
        [[[[-0.8]]], [[[0.4]]]], dtype=torch.float64, requires_grad=True
    )
    real = torch.tensor(
        [[[[-1.2]]], [[[0.2]]], [[[1.1]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    self_support = torch.tensor(
        [[[[-0.5]]], [[[0.9]]]], dtype=torch.float64, requires_grad=True
    )
    loss, health = sinkhorn_drifted_target_loss(
        primary, real, self_support, 1.0, _strict_config()
    )
    loss.backward()
    assert primary.grad is not None
    assert float(primary.grad.norm()) > 0
    assert real.grad is None
    assert self_support.grad is None
    assert not health["target_requires_grad"]
    assert health["gradient_roles"] == ["primary_generated_endpoints"]


def test_detached_target_gradient_matches_cross_self_energy_gradient() -> None:
    config = _strict_config()
    primary = torch.tensor(
        [[[[-0.8]]], [[[0.4]]]], dtype=torch.float64, requires_grad=True
    )
    real = torch.tensor([[[[-1.2]]], [[[0.2]]], [[[1.1]]]], dtype=torch.float64)
    self_support = torch.tensor([[[[-0.5]]], [[[0.9]]]], dtype=torch.float64)
    energy = empirical_cross_self_energy(primary, real, self_support, 1.0, config)
    energy_gradient = torch.autograd.grad(energy, primary)[0]
    loss, _ = sinkhorn_drifted_target_loss(primary, real, self_support, 1.0, config)
    loss_gradient = torch.autograd.grad(loss, primary)[0]
    assert torch.allclose(
        loss_gradient,
        2.0 * config.eta * energy_gradient,
        rtol=2e-7,
        atol=2e-9,
    )

    # Re-solve the plans at both perturbations: this checks the envelope
    # gradient against the actual finite entropic objective, not a frozen-plan
    # linearization.
    delta = 1e-5
    upper = primary.detach().clone()
    lower = primary.detach().clone()
    upper[0, 0, 0, 0] += delta
    lower[0, 0, 0, 0] -= delta
    upper_value = empirical_cross_self_energy(upper, real, self_support, 1.0, config)
    lower_value = empirical_cross_self_energy(lower, real, self_support, 1.0, config)
    finite_difference = float((upper_value - lower_value) / (2 * delta))
    assert abs(finite_difference - float(energy_gradient[0, 0, 0, 0])) < 2e-6


def test_target_cost_scale_is_median_off_diagonal_quadratic_cost() -> None:
    target = torch.tensor([[[[-1.0]]], [[[1.0]]], [[[3.0]]]], dtype=torch.float64)
    # Pair costs are 2, 8, and 2; the median is 2.
    assert target_cost_scale(target) == 2.0


def test_same_tensor_self_support_is_rejected() -> None:
    primary = torch.tensor([[[[-1.0]]], [[[1.0]]]], dtype=torch.float64)
    try:
        sinkhorn_velocity(
            primary,
            primary.detach().clone(),
            primary,
            1.0,
            _strict_config(real_batch=2),
        )
    except ValueError as error:
        assert "distinct generated batch" in str(error)
    else:
        raise AssertionError("same-tensor Sinkhorn self support was accepted")


def test_iteration_cap_is_reported_or_rejected() -> None:
    cost = torch.tensor([[0.0, 10.0, 20.0], [5.0, 0.0, 10.0]], dtype=torch.float64)
    config = _strict_config(
        relative_tolerance=1e-14,
        max_iterations=1,
        min_iterations=1,
    )
    report = log_sinkhorn_plan(cost, config, require_convergence=False)
    assert not report.converged
    assert report.diagnostics()["iteration_cap_hit"]
    try:
        log_sinkhorn_plan(cost, config, require_convergence=True)
    except RuntimeError as error:
        assert "did not meet" in str(error)
    else:
        raise AssertionError("nonconverged Sinkhorn plan was accepted")


if __name__ == "__main__":
    main(__name__, globals())
