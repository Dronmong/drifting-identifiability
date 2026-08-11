"""Regression tests for the local one-step pixel MeanFlow S3 foundation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

from ..stage_pmf.audit import require_launch_authorization, source_manifest
from ..stage_pmf.config import PMFObjectiveConfig, profile
from ..stage_pmf.model import PixelMeanFlowTransformer
from ..stage_pmf.objective import (
    TriangleSample,
    average_velocity,
    meanflow_loss,
    one_step_sample,
    sample_time_triangle,
)
from ..stage_pmf.training import (
    checkpoint_payload,
    pmf_evaluation_seed,
    pmf_seed,
    pmf_streams,
    train_pmf,
    training_batch,
)
from .harness import main


def _pool(count: int = 16, size: int = 8) -> torch.Tensor:
    return (
        torch.rand(count, 3, size, size, generator=torch.Generator().manual_seed(41))
        * 2
        - 1
    )


def test_direct_pixel_transformer_shape_seed_and_conditions() -> None:
    selected = profile("smoke")
    first = PixelMeanFlowTransformer(selected.model, seed=8)
    replay = PixelMeanFlowTransformer(selected.model, seed=8)
    other = PixelMeanFlowTransformer(selected.model, seed=9)
    images = _pool(2)
    t = torch.tensor([0.4, 0.8])
    h = torch.tensor([0.1, 0.5])
    with torch.no_grad():
        left = first(images, t, h)
        assert left.shape == images.shape
        assert torch.equal(left, replay(images, t, h))
        # Heads are deliberately zero at initialization, but complete model
        # states and internal conditioned representations must still differ.
        assert not all(
            torch.equal(value, other.state_dict()[name])
            for name, value in first.state_dict().items()
        )
        for block in (*first.encoder, *first.decoder):
            block.attention_scale.fill_(0.1)
            block.mlp_scale.fill_(0.1)
        first.pixel_head.weight.copy_(
            torch.randn(
                first.pixel_head.weight.shape,
                generator=torch.Generator().manual_seed(82),
            )
            * 1e-4
        )
        assert not torch.equal(first(images, t, h), first(images, t, h * 0))


def test_transformer_residual_branches_are_zero_gated_at_initialization() -> None:
    model = PixelMeanFlowTransformer(profile("smoke").model, seed=81)
    for block in (*model.encoder, *model.decoder):
        assert torch.count_nonzero(block.attention_scale) == 0
        assert torch.count_nonzero(block.mlp_scale) == 0


def test_auxiliary_head_is_trained_but_absent_from_one_step_inference() -> None:
    selected = profile("smoke")
    model = PixelMeanFlowTransformer(selected.model, seed=83)
    clean = _pool(2)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(84))
    triangle = TriangleSample(
        t=torch.tensor([0.7, 0.8]),
        r=torch.tensor([0.2, 0.8]),
        diagonal=torch.tensor([False, True]),
    )
    outcome = meanflow_loss(model, clean, noise, triangle, selected.objective)
    outcome.loss.backward()
    assert outcome.auxiliary_velocity is not None
    assert model.auxiliary_pixel_head.weight.grad is not None
    assert torch.count_nonzero(model.auxiliary_pixel_head.weight.grad) > 0

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


def test_triangle_is_reproducible_complete_and_half_diagonal() -> None:
    config = PMFObjectiveConfig(diagonal_fraction=0.5)
    values_a = torch.Generator().manual_seed(1)
    masks_a = torch.Generator().manual_seed(2)
    values_b = torch.Generator().manual_seed(1)
    masks_b = torch.Generator().manual_seed(2)
    first = sample_time_triangle(20, config, values_a, masks_a)
    replay = sample_time_triangle(20, config, values_b, masks_b)
    assert torch.equal(first.t, replay.t)
    assert torch.equal(first.r, replay.r)
    assert torch.equal(first.diagonal, replay.diagonal)
    assert int(first.diagonal.sum()) == 10
    assert bool((0 < first.r).all() and (first.r <= first.t).all())
    assert bool((first.t < 1).all())
    assert bool((first.r[~first.diagonal] < first.t[~first.diagonal]).all())


def test_diagonal_time_uses_first_draw_not_pair_maximum() -> None:
    config = PMFObjectiveConfig(diagonal_fraction=0.5)
    raw_generator = torch.Generator().manual_seed(31)
    mask_generator = torch.Generator().manual_seed(32)
    expected_values = torch.sigmoid(
        torch.randn(12, 2, generator=raw_generator) * config.logit_std
        + config.logit_mean
    )
    expected_mask = torch.zeros(12, dtype=torch.bool)
    expected_mask[torch.randperm(12, generator=mask_generator)[:6]] = True
    actual = sample_time_triangle(
        12,
        config,
        torch.Generator().manual_seed(31),
        torch.Generator().manual_seed(32),
    )
    assert torch.equal(actual.diagonal, expected_mask)
    assert torch.equal(actual.t[expected_mask], expected_values[expected_mask, 0])
    assert torch.equal(actual.r[expected_mask], expected_values[expected_mask, 0])


class ConstantVelocityPixels(nn.Module):
    def __init__(self, velocity: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("velocity", velocity)

    def forward(self, state, t, interval):
        del interval
        return state - t[:, None, None, None] * self.velocity


def test_constant_velocity_has_zero_meanflow_jvp_correction() -> None:
    state = torch.randn(2, 1, 2, 2, generator=torch.Generator().manual_seed(3))
    velocity = torch.randn(2, 1, 2, 2, generator=torch.Generator().manual_seed(4))
    model = ConstantVelocityPixels(velocity)
    t = torch.tensor([0.6, 0.8])
    r = torch.tensor([0.1, 0.3])

    def fn(z, tv, rv):
        return average_velocity(model, z, tv, rv, 0.05)

    average, derivative = torch.func.jvp(
        fn, (state, t, r), (velocity, torch.ones_like(t), torch.zeros_like(r))
    )
    assert torch.allclose(average, velocity, atol=2e-6, rtol=2e-6)
    assert torch.allclose(derivative, torch.zeros_like(derivative), atol=2e-6)


def test_x_u_conversion_round_trip() -> None:
    state = torch.randn(3, 2, 2, 2, generator=torch.Generator().manual_seed(5))
    pixels = torch.randn(3, 2, 2, 2, generator=torch.Generator().manual_seed(6))
    t = torch.tensor([0.2, 0.5, 1.0])

    class Fixed(nn.Module):
        def forward(self, z, _t, _h):
            return pixels + 0 * z

    velocity = average_velocity(Fixed(), state, t, t * 0, 0.05)
    recovered = state - t[:, None, None, None] * velocity
    assert torch.allclose(recovered, pixels, atol=1e-6, rtol=1e-6)


def test_jvp_matches_central_finite_difference() -> None:
    selected = profile("smoke")
    model = PixelMeanFlowTransformer(selected.model, seed=19)
    with torch.no_grad():
        model.pixel_head.weight.copy_(
            torch.randn(
                model.pixel_head.weight.shape,
                generator=torch.Generator().manual_seed(85),
            )
            * 1e-3
        )
        for block in (*model.encoder, *model.decoder):
            block.attention_scale.fill_(0.1)
            block.mlp_scale.fill_(0.1)
    z = _pool(2)
    t = torch.full((2,), 0.7)
    r = torch.full((2,), 0.2)

    def fn(zv, tv, rv):
        return average_velocity(model, zv, tv, rv, 0.05)

    with torch.no_grad():
        tangent = fn(z, t, t)
    _, actual = torch.func.jvp(
        fn, (z, t, r), (tangent, torch.ones_like(t), torch.zeros_like(r))
    )
    # Time is embedded on a diffusion-style 0--1000 scale; use a genuinely
    # local difference rather than crossing several high-frequency phases.
    eps = 3e-4
    finite = (fn(z + eps * tangent, t + eps, r) - fn(z - eps * tangent, t - eps, r)) / (
        2 * eps
    )
    relative = (actual - finite).double().norm() / finite.double().norm().clamp_min(
        1e-12
    )
    assert float(relative) < 0.02


class ParametricPixels(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.4))

    def forward(self, state, t, interval):
        del interval
        return state - t[:, None, None, None] * self.scale * state


def test_directional_term_is_stopped_but_primal_gradient_remains() -> None:
    model = ParametricPixels()
    clean = torch.ones(1, 1, 1, 2)
    noise = clean.clone()
    triangle = TriangleSample(
        t=torch.tensor([0.8]), r=torch.tensor([0.2]), diagonal=torch.tensor([False])
    )
    config = PMFObjectiveConfig(adaptive_power=0.0)
    outcome = meanflow_loss(model, clean, noise, triangle, config)
    outcome.loss.backward()
    compound = outcome.compound.detach()
    expected = 2 * (compound * clean).sum()
    assert model.scale.grad is not None
    assert torch.allclose(model.scale.grad, expected, atol=2e-6, rtol=2e-6)
    assert model.scale.grad.abs() > 0


def test_diagonal_samples_remove_derivative_correction() -> None:
    model = ParametricPixels()
    clean = torch.ones(2, 1, 1, 1)
    noise = torch.zeros_like(clean)
    t = torch.tensor([0.3, 0.9])
    triangle = TriangleSample(
        t=t, r=t.clone(), diagonal=torch.ones(2, dtype=torch.bool)
    )
    outcome = meanflow_loss(model, clean, noise, triangle, PMFObjectiveConfig())
    assert torch.equal(outcome.compound, outcome.average_velocity)


def test_velocity_target_uses_the_same_small_time_clamp_as_prediction() -> None:
    model = ParametricPixels()
    clean = torch.ones(1, 1, 1, 1)
    noise = torch.zeros_like(clean)
    triangle = TriangleSample(
        t=torch.tensor([0.01]),
        r=torch.tensor([0.01]),
        diagonal=torch.tensor([True]),
    )
    outcome = meanflow_loss(model, clean, noise, triangle, PMFObjectiveConfig())
    # z_t - x = -0.01 while the stabilized denominator is 0.05.
    assert torch.allclose(outcome.target_velocity, torch.full_like(clean, -0.2))


def test_endpoint_sampler_invokes_network_once() -> None:
    class Counter(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, state, t, h):
            self.calls += 1
            assert torch.equal(t, torch.ones_like(t))
            assert torch.equal(h, torch.ones_like(h))
            return state * 0.25

    model = Counter()
    noise = _pool(5)
    result = one_step_sample(model, noise)
    assert model.calls == 1
    assert torch.equal(result, noise * 0.25)


def test_streams_and_training_batch_replay() -> None:
    left = pmf_streams("test", 7)
    right = pmf_streams("test", 7)
    a = training_batch(_pool(), 5, left, "cpu", True)
    b = training_batch(_pool(), 5, right, "cpu", True)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    ta = sample_time_triangle(
        5, profile("smoke").objective, left.time_values, left.diagonal_mask
    )
    tb = sample_time_triangle(
        5, profile("smoke").objective, right.time_values, right.diagonal_mask
    )
    assert torch.equal(ta.t, tb.t) and torch.equal(ta.r, tb.r)


def test_training_units_are_independent_but_evaluation_noise_is_paired() -> None:
    assert pmf_seed("local-s3-full", 700, "path-noise") != pmf_seed(
        "local-s3-full", 701, "path-noise"
    )
    shared = pmf_evaluation_seed("local-s3-full", "sealed-one-step-grid")
    left = torch.randn(8, generator=torch.Generator().manual_seed(shared))
    right = torch.randn(8, generator=torch.Generator().manual_seed(shared))
    assert torch.equal(left, right)


def test_tiny_training_is_finite_and_reproducible() -> None:
    selected = profile("smoke")
    first = train_pmf(_pool(), selected, "test", 12, "cpu")
    replay = train_pmf(_pool(), selected, "test", 12, "cpu")
    assert all(
        torch.equal(value, replay.model.state_dict()[name])
        for name, value in first.model.state_dict().items()
    )
    assert all(
        torch.isfinite(value).all() for value in first.model.state_dict().values()
    )
    assert first.examples_seen == (
        selected.train.updates
        * selected.train.micro_batch
        * selected.train.accumulation_steps
    )


def test_checkpoint_resume_matches_uninterrupted_training() -> None:
    selected = profile("smoke")
    one_update = replace(
        selected,
        train=replace(selected.train, updates=1, checkpoint_updates=(1,)),
    )
    captured = {}

    def checkpoint(update, model, ema, optimizer, streams, _row, history):
        captured["payload"] = checkpoint_payload(
            update,
            model,
            ema,
            optimizer,
            streams,
            {"test": "profile"},
            14,
            history,
            "test-source",
        )

    train_pmf(_pool(), one_update, "resume-test", 14, "cpu", checkpoint)
    resumed = train_pmf(
        _pool(),
        selected,
        "resume-test",
        14,
        "cpu",
        resume_payload=captured["payload"],
    )
    uninterrupted = train_pmf(_pool(), selected, "resume-test", 14, "cpu")
    assert all(
        torch.equal(value, uninterrupted.model.state_dict()[name])
        for name, value in resumed.model.state_dict().items()
    )
    assert all(
        torch.equal(value, uninterrupted.ema.shadow[name])
        for name, value in resumed.ema.shadow.items()
    )
    assert resumed.wall_seconds >= captured["payload"]["history"][-1]["wall_seconds"]


def test_source_manifest_covers_transitive_evaluation_and_launch_code() -> None:
    paths = set(source_manifest())
    expected = {
        "numerics/encoder_independent_drifting/config.py",
        "numerics/encoder_independent_drifting/device.py",
        "numerics/encoder_independent_drifting/diagnostics.py",
        "numerics/encoder_independent_drifting/fid.py",
        "numerics/encoder_independent_drifting/appearance.py",
        "numerics/encoder_independent_drifting/diagnose_phase20.py",
        "numerics/encoder_independent_drifting/stage_b2/metrics.py",
        "numerics/encoder_independent_drifting/stage_pmf/sanity.py",
        "numerics/encoder_independent_drifting/stage_pmf/launch_s3.ps1",
    }
    assert expected <= paths


def test_full_runner_is_blocked_without_post_audit_authorization() -> None:
    missing = Path("this-file-intentionally-does-not-exist.json")
    try:
        require_launch_authorization(profile("local_s3"), missing)
    except RuntimeError as error:
        assert "FULL S3 RUN BLOCKED" in str(error)
    else:
        raise AssertionError("full runner accepted a missing audit authorization")


if __name__ == "__main__":
    main(__name__, globals())
