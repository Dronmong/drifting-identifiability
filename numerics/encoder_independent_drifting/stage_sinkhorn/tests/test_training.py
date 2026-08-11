"""B0 integration tests for the balanced Sinkhorn correction."""

from __future__ import annotations

from dataclasses import replace

import torch

from ...f3b import profile, train_bridge
from ...tests.harness import main
from ..core import SinkhornConfig, target_cost_scale
from ..freeze import select_candidate
from ..training import (
    minimax_log_event_lambda,
    paired_seed_manifest,
    parameter_gradient_geometry,
    sinkhorn_correction_term,
    sinkhorn_streams,
    train_sinkhorn_bridge,
)


def _pool() -> torch.Tensor:
    generator = torch.Generator().manual_seed(812)
    return torch.rand(24, 3, 8, 8, generator=generator) * 2 - 1


def _config() -> SinkhornConfig:
    result = SinkhornConfig(
        epsilon=0.2,
        relative_tolerance=1e-4,
        max_iterations=300,
        min_iterations=2,
        check_every=1,
        eta=0.05,
        correction_every=1,
        primary_batch=3,
        self_batch=3,
        real_batch=4,
        correction_nfe=1,
        event_gradient_ratio=0.25,
    )
    result.validate()
    return result


def test_correction_reaches_model_parameters_and_replays() -> None:
    selected = profile("smoke")
    pool = _pool()
    scale = target_cost_scale(pool[:12])
    model_a = train_bridge(
        pool,
        selected.model,
        replace(selected.train, steps=1, checkpoint_steps=(1,)),
        "sinkhorn-model-source",
        41,
        "cpu",
    ).model
    model_b = train_bridge(
        pool,
        selected.model,
        replace(selected.train, steps=1, checkpoint_steps=(1,)),
        "sinkhorn-model-source",
        41,
        "cpu",
    ).model
    loss_a, health_a = sinkhorn_correction_term(
        model_a,
        pool,
        selected.model,
        sinkhorn_streams("sinkhorn-replay", 41),
        "cpu",
        scale,
        _config(),
    )
    loss_b, health_b = sinkhorn_correction_term(
        model_b,
        pool,
        selected.model,
        sinkhorn_streams("sinkhorn-replay", 41),
        "cpu",
        scale,
        _config(),
    )
    assert torch.equal(loss_a.detach(), loss_b.detach())
    assert health_a == health_b
    loss_a.backward()
    gradients = [parameter.grad for parameter in model_a.parameters()]
    assert any(value is not None and float(value.norm()) > 0 for value in gradients)
    assert health_a["primary_prior_and_self_prior_distinct"]
    assert health_a["diagonal_mask"] is False


def test_paired_training_keeps_first_flow_batch_and_initialization() -> None:
    selected = profile("smoke")
    pool = _pool()
    scale = target_cost_scale(pool[:12])
    phase = "sinkhorn-paired-smoke"
    unit = 42
    control = train_bridge(
        pool,
        selected.model,
        selected.train,
        phase,
        unit,
        "cpu",
    )
    candidate = train_sinkhorn_bridge(
        pool,
        selected.model,
        selected.train,
        phase,
        unit,
        "cpu",
        cost_scale=scale,
        lambda_event=1e-3,
        config=_config(),
    )
    assert control.history[0]["loss"] == candidate.history[0]["flow_loss"]
    assert candidate.correction_events == selected.train.steps
    assert candidate.correction_model_forwards == 2 * selected.train.steps
    assert candidate.solver_summary["cap_hits"] == 0
    assert candidate.solver_summary["maximum_relative_error"] <= 1e-4


def test_seed_manifest_separates_primary_and_self_roles() -> None:
    manifest = paired_seed_manifest("sinkhorn-seed-test", 43)
    assert manifest["sinkhorn_primary-prior"] != manifest["sinkhorn_self-prior"]
    assert len(set(manifest.values())) == len(manifest)


def test_minimax_log_event_lambda_centers_multiplicative_extrema() -> None:
    selected = minimax_log_event_lambda([1.0, 2.0, 9.0])
    assert selected == 3.0
    assert max(selected / 1.0, 9.0 / selected) == 3.0


def test_parameter_gradient_geometry_reports_cosine() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    first = model.weight[0, 0] + model.weight[0, 1]
    second = model.weight[0, 0] - model.weight[0, 1]
    geometry = parameter_gradient_geometry(first, second, model)
    assert abs(geometry["first_norm"] - 2**0.5) < 1e-12
    assert abs(geometry["second_norm"] - 2**0.5) < 1e-12
    assert abs(geometry["cosine"]) < 1e-12


def test_freeze_selection_uses_only_mechanical_concentration_order() -> None:
    diffuse = {
        "epsilon": 0.1,
        "summary": {
            "all_mechanical_gates_pass": True,
            "conditional_max_weight_maximum": 0.2,
            "maximum_iterations": 10,
            "mean_event_seconds": 2.0,
        },
    }
    sharp = {
        "epsilon": 0.05,
        "summary": {
            "all_mechanical_gates_pass": True,
            "conditional_max_weight_maximum": 0.6,
            "maximum_iterations": 5,
            "mean_event_seconds": 1.0,
        },
    }
    assert select_candidate([sharp, diffuse]) is diffuse


if __name__ == "__main__":
    main(__name__, globals())
