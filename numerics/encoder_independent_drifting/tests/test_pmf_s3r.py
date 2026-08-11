"""Regression tests for the post-failure S3R mechanisms and health gates."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from ..stage_pmf.objective import TriangleSample
from ..stage_pmf_r.audit import require_developmental_preflight
from ..stage_pmf_r.config import profile
from ..stage_pmf_r.data import automobile_train_pool
from ..stage_pmf_r.diagnostics import (
    developmental_series_gate,
    endpoint_health,
    haar_subbands,
)
from ..stage_pmf_r.model import RepairedPixelMeanFlowTransformer
from ..stage_pmf_r.objectives import (
    alpha_flow_loss,
    alpha_schedule,
    emf_local_difference,
    emf_x1_loss,
    one_step_sample,
    pmf_loss,
    velocity_from_pixels,
)
from ..stage_pmf_r.training import train_arm
from .harness import main


def _images(count: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(501)
    clean = torch.rand(count, 3, 8, 8, generator=generator) * 2 - 1
    noise = torch.randn(clean.shape, generator=generator)
    return clean, noise


def _triangle() -> TriangleSample:
    return TriangleSample(
        t=torch.tensor([0.8, 0.7, 0.6, 0.9]),
        r=torch.tensor([0.2, 0.7, 0.1, 0.4]),
        diagonal=torch.tensor([False, True, False, False]),
    )


def test_developmental_budget_is_matched_and_bounded() -> None:
    selected = profile("developmental")
    assert selected.train.effective_batch == 64
    assert selected.train.updates * selected.train.effective_batch == 800_000
    assert selected.train.gradient_clip == 10


def test_developmental_loader_never_opens_official_test_split() -> None:
    calls = []

    class FakeCIFAR:
        def __init__(self, *, train, **_kwargs):
            calls.append(train)
            self.targets = np.ones(5_000, dtype=np.int64)
            self.data = np.zeros((5_000, 32, 32, 3), dtype=np.uint8)

    automobile_train_pool.cache_clear()
    try:
        with patch("torchvision.datasets.CIFAR10", FakeCIFAR):
            pool = automobile_train_pool("audit-only")
        assert pool.shape == (5_000, 3, 32, 32)
        assert calls == [True]
    finally:
        automobile_train_pool.cache_clear()


def test_alpha_schedule_is_monotone_and_never_enters_jvp_phase() -> None:
    selected = profile("developmental")
    values = [
        alpha_schedule(update, selected.train.updates, selected.objective)
        for update in range(0, selected.train.updates + 1, 125)
    ]
    assert all(left >= right for left, right in pairwise(values))
    assert values[0] == 1
    assert min(values) == selected.objective.alpha_floor


def test_alpha_one_is_exactly_trajectory_flow_matching() -> None:
    selected = profile("smoke")
    model = RepairedPixelMeanFlowTransformer(selected.model, seed=502)
    clean, noise = _images()
    triangle = _triangle()
    outcome = alpha_flow_loss(
        model, clean, noise, triangle, selected.objective, alpha=1.0
    )
    t, r = triangle.t, triangle.r
    state = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    predicted = velocity_from_pixels(
        model, state, t, r, selected.objective.denominator_floor
    )
    divisor = t.clamp_min(selected.objective.denominator_floor)[:, None, None, None]
    target = (state - clean) / divisor
    assert torch.allclose(outcome.raw_mse, (predicted - target).square().mean())
    assert outcome.jvp_per_sample_rms.count_nonzero() == 0


def test_alpha_diagonal_rows_stay_flow_matching_below_one() -> None:
    selected = profile("smoke")
    model = RepairedPixelMeanFlowTransformer(selected.model, seed=5021)
    clean, noise = _images()
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    triangle = TriangleSample(t=t, r=t, diagonal=torch.ones(4, dtype=torch.bool))
    alpha = 0.125
    outcome = alpha_flow_loss(
        model, clean, noise, triangle, selected.objective, alpha=alpha
    )
    state = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    prediction = velocity_from_pixels(
        model, state, t, t, selected.objective.denominator_floor
    )
    target = (state - clean) / t[:, None, None, None].clamp_min(
        selected.objective.denominator_floor
    )
    per_sample = (prediction - target).square().flatten(1).mean(dim=1)
    expected = (
        per_sample
        / (per_sample.detach() + selected.objective.alpha_adaptive_epsilon)
    ).mean()
    assert torch.allclose(outcome.loss, expected)
    assert torch.allclose(outcome.raw_mse, per_sample.mean())


class SmoothDirectField(nn.Module):
    def forward(self, state, t, interval):
        return (
            0.3 * state
            + 0.2 * t[:, None, None, None]
            + 0.1 * interval[:, None, None, None]
        )


def test_emf_local_quotient_matches_its_directional_derivative() -> None:
    model = SmoothDirectField().double()
    clean, noise = (value.double() for value in _images())
    triangle = _triangle()
    t, r = triangle.t.double(), triangle.r.double()
    state = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    with torch.no_grad():
        boundary = model(state, t, torch.zeros_like(t))
    direction = (boundary - state) / t[:, None, None, None]

    def field(z, tv, h):
        return model(z, tv, h)

    _, exact = torch.func.jvp(
        field,
        (state, t, t - r),
        (direction, -torch.ones_like(t), -torch.ones_like(t)),
    )
    _, _, finite = emf_local_difference(model, state, t, r, 1e-5)
    exact = exact * ((t - r) > 1e-5)[:, None, None, None]
    assert torch.allclose(finite, exact, atol=2e-10, rtol=2e-10)


def test_emf_diagonal_reduces_to_direct_clean_regression() -> None:
    selected = profile("smoke")
    model = RepairedPixelMeanFlowTransformer(selected.model, seed=503)
    clean, noise = _images()
    t = torch.tensor([0.2, 0.4, 0.6, 0.8])
    triangle = TriangleSample(t=t, r=t, diagonal=torch.ones(4, dtype=torch.bool))
    outcome = emf_x1_loss(model, clean, noise, triangle, selected.objective)
    state = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    prediction = model(state, t, torch.zeros_like(t))
    assert torch.allclose(outcome.raw_mse, (prediction - clean).square().mean())
    per_sample_sum = (prediction - clean).square().flatten(1).sum(dim=1)
    expected = (
        t.clamp_min(selected.objective.emf_denominator_floor).pow(-2)
        * per_sample_sum
        / (
            per_sample_sum.detach() + selected.objective.adaptive_epsilon
        ).pow(selected.objective.adaptive_power)
    ).mean()
    assert torch.allclose(outcome.loss, expected)
    assert outcome.interior_raw_mse == 0


def test_deep_auxiliary_branch_trains_but_is_absent_at_inference() -> None:
    selected = profile("smoke")
    model = RepairedPixelMeanFlowTransformer(selected.model, seed=504)
    with torch.no_grad():
        model.auxiliary_pixel_head.weight.fill_(1e-3)
        for block in model.auxiliary_blocks:
            block.attention_scale.fill_(0.1)
            block.mlp_scale.fill_(0.1)
    clean, noise = _images()
    outcome = pmf_loss(model, clean, noise, _triangle(), selected.objective)
    outcome.loss.backward()
    assert any(
        parameter.grad is not None and parameter.grad.count_nonzero() > 0
        for parameter in model.auxiliary_blocks.parameters()
    )
    calls = 0

    def counted(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = model.auxiliary_pixel_head.register_forward_hook(counted)
    try:
        one_step_sample(model, noise)
    finally:
        handle.remove()
    assert calls == 0
    assert model.inference_parameter_count() < model.parameter_count()


def test_haar_is_energy_preserving_and_health_is_amplitude_aware() -> None:
    target, _ = _images(16)
    generated = target * 0.01
    bands = haar_subbands(target.double())
    assert torch.allclose(
        target.double().square().sum(),
        sum(value.square().sum() for value in bands.values()),
        atol=1e-10,
        rtol=1e-10,
    )
    report = endpoint_health(generated, target)
    assert report["effective_rank"] > 1
    assert report["second_moment_ratio"] < 0.001
    assert not report["rank_interpretable"]
    assert set(report["haar"]) == {"ll", "lh", "hl", "hh"}


def test_health_rejects_nonzero_constant_output_as_rank_evidence() -> None:
    target, _ = _images(16)
    generated = torch.full_like(target, 0.75)
    report = endpoint_health(generated, target)
    assert report["second_moment_ratio"] > 0.5
    assert report["variance_ratio"] == 0
    assert not report["rank_interpretable"]


def test_series_gate_rejects_late_rank_collapse() -> None:
    healthy = {
        "second_moment_ratio": 0.7,
        "variance_ratio": 0.7,
        "rank_interpretable": True,
        "effective_rank_ratio": 0.8,
    }
    collapsed = {
        "second_moment_ratio": 0.7,
        "variance_ratio": 0.7,
        "rank_interpretable": True,
        "effective_rank_ratio": 0.61,
    }
    history = [
        {"raw": dict(healthy), "ema": dict(healthy)},
        {"raw": dict(collapsed), "ema": dict(collapsed)},
    ]
    verdict = developmental_series_gate(history, clipping_fraction=0.0)
    assert verdict["final_snapshot"]["passes_current_snapshot"]
    assert not verdict["rank_retention_ok"]
    assert not verdict["passes"]


def test_long_runner_is_blocked_without_a_matching_preflight() -> None:
    try:
        require_developmental_preflight(Path("intentionally-missing-s3r.json"))
    except RuntimeError as error:
        assert "S3R RUN BLOCKED" in str(error)
    else:
        raise AssertionError("developmental runner accepted a missing preflight")


def test_s3r_checkpoint_resume_replays_exactly() -> None:
    selected = profile("smoke")
    clean, _ = _images(16)
    captured = {}

    def checkpoint(payload):
        captured["payload"] = payload

    train_arm(
        clean,
        selected,
        "alpha",
        "resume-test",
        42,
        "cpu",
        checkpoint=checkpoint,
        stop_after_update=1,
    )
    resumed = train_arm(
        clean,
        selected,
        "alpha",
        "resume-test",
        42,
        "cpu",
        resume_payload=captured["payload"],
    )
    uninterrupted = train_arm(clean, selected, "alpha", "resume-test", 42, "cpu")
    assert all(
        torch.equal(value, uninterrupted.model.state_dict()[name])
        for name, value in resumed.model.state_dict().items()
    )
    assert resumed.examples_seen == uninterrupted.examples_seen


def test_smoke_preflight_cannot_unlock_developmental_runner(tmp_path=None) -> None:
    import json
    import tempfile

    from ..stage_pmf_r.audit import source_digest

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.json"
        path.write_text(
            json.dumps(
                {
                    "status": "s3r-preflight-passed",
                    "profile": {"name": "smoke"},
                    "source_sha256": source_digest(),
                    "launch_authorized": False,
                }
            ),
            encoding="utf-8",
        )
        try:
            require_developmental_preflight(path)
        except RuntimeError as error:
            assert "smoke preflight" in str(error)
        else:
            raise AssertionError("smoke preflight unlocked the developmental runner")


if __name__ == "__main__":
    main(__name__, globals())
