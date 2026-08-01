"""Regression tests for the repaired, theory-aligned Stage B2."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ...f3b import TimeConditionedUNet, profile, train_bridge
from ...tests.harness import main
from ..core import (
    B2Config,
    b2_config,
    b2_streams,
    calibrate_laplace_bandwidth,
    correction_term,
    laplace_drift_energy,
    laplace_mean_shift_field,
    parameter_gradient_norm,
    train_b2,
)
from ..evaluation import (
    compare_drift_audits,
    drift_energy_audit_suite,
    fresh_evaluation_allocation,
)
from ..fresh_data import load_fresh_pool


def _small_config() -> B2Config:
    result = B2Config(
        correction_every=1,
        probe_batch=4,
        positive_batch=4,
        negative_batch=4,
        correction_nfe=1,
        probe_noise_std=0.05,
        ess_samples=8,
        ess_iterations=16,
        audit_batch=4,
        audit_replicates=2,
        drift_paired_wins_required=1,
    )
    result.validate()
    return result


def _pool(count: int = 24) -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    return torch.rand(count, 3, 8, 8, generator=generator) * 2 - 1


def test_field_is_zero_for_identical_empirical_laws_and_antisymmetric() -> None:
    probes = torch.tensor([[-0.7], [0.2], [1.4]], dtype=torch.float64)
    positive = torch.tensor([[-1.0], [0.5], [2.0]], dtype=torch.float64)
    zero, _ = laplace_mean_shift_field(probes, positive, positive, 0.8)
    forward, _ = laplace_mean_shift_field(probes, positive, positive + 0.4, 0.8)
    reverse, _ = laplace_mean_shift_field(probes, positive + 0.4, positive, 0.8)
    assert torch.equal(zero, torch.zeros_like(zero))
    assert torch.allclose(forward, -reverse, atol=1e-12, rtol=1e-12)


def test_drift_energy_gradient_matches_finite_difference() -> None:
    probes = torch.tensor([[-0.4], [0.8]], dtype=torch.float64)
    positive = torch.tensor([[-1.0], [0.3], [1.5]], dtype=torch.float64)
    negative = torch.tensor(
        [[-0.7], [0.1], [1.0]], dtype=torch.float64, requires_grad=True
    )
    value, _ = laplace_drift_energy(probes, positive, negative, 0.9)
    (analytic,) = torch.autograd.grad(value, negative)
    step = 1e-6
    with torch.no_grad():
        plus = negative.detach().clone()
        minus = negative.detach().clone()
        plus[1, 0] += step
        minus[1, 0] -= step
        upper, _ = laplace_drift_energy(probes, positive, plus, 0.9)
        lower, _ = laplace_drift_energy(probes, positive, minus, 0.9)
    numeric = (upper - lower) / (2 * step)
    assert value.requires_grad
    assert abs(float(analytic[1, 0] - numeric)) < 2e-7


def test_remote_target_cannot_win_by_affinity_underflow() -> None:
    probes = torch.zeros(4, 2, dtype=torch.float32)
    negative = torch.tensor(
        [[-0.2, 0.0], [0.2, 0.0], [0.0, -0.2], [0.0, 0.2]],
        dtype=torch.float32,
        requires_grad=True,
    )
    near_positive = negative.detach() + 0.1
    far_positive = near_positive + 100.0
    near, _ = laplace_drift_energy(probes, near_positive, negative, 1.0)
    far, health = laplace_drift_energy(probes, far_positive, negative, 1.0)
    far.backward()
    assert torch.isfinite(far)
    assert float(far.detach()) > 1_000.0
    assert float(far.detach()) > float(near.detach())
    assert negative.grad is not None and bool(torch.isfinite(negative.grad).all())
    assert health["positive"]["row_sum_error_maximum"] < 1e-6


def test_cpu_and_cuda_field_agree_when_cuda_exists() -> None:
    if not torch.cuda.is_available():
        return
    generator = torch.Generator().manual_seed(71)
    probes = torch.randn(5, 3, generator=generator)
    positive = torch.randn(7, 3, generator=generator)
    negative = torch.randn(6, 3, generator=generator)
    cpu, _ = laplace_mean_shift_field(probes, positive, negative, 1.7)
    cuda, _ = laplace_mean_shift_field(
        probes.cuda(), positive.cuda(), negative.cuda(), 1.7
    )
    assert torch.allclose(cpu, cuda.cpu(), atol=2e-6, rtol=2e-6)


def test_off_diagonal_bandwidth_calibration_hits_declared_ess() -> None:
    config = _small_config()
    target = torch.linspace(-2, 2, config.ess_samples)[:, None]
    tau, record = calibrate_laplace_bandwidth(target, config)
    assert tau > 0
    assert record["exclude_self"] is True
    assert (
        abs(record["achieved_off_diagonal_ess_fraction"] - config.target_ess_fraction)
        < 2e-3
    )


def test_correction_reaches_model_parameters_through_negative_law() -> None:
    config = _small_config()
    selected = profile("smoke")
    model = TimeConditionedUNet(selected.model, seed=11)
    value, health = correction_term(
        model,
        _pool(),
        selected.model,
        b2_streams("test", 1),
        "cpu",
        tau=4.0,
        config=config,
        horizontal_flip=False,
    )
    assert value.requires_grad
    assert parameter_gradient_norm(value, model) > 0
    assert health["gradient_roles"] == ["negative_model_samples"]


def test_b2_replays_streams_and_preserves_b0_flow_pairing() -> None:
    left = b2_streams("test", 2)
    right = b2_streams("test", 2)
    assert torch.equal(
        torch.randn(5, generator=left.negative_prior),
        torch.randn(5, generator=right.negative_prior),
    )
    assert left.negative_prior.initial_seed() != left.probe_noise.initial_seed()

    config = _small_config()
    selected = profile("smoke")
    train_config = replace(selected.train, steps=1, checkpoint_steps=(1,), log_every=1)
    pool = _pool()
    baseline = train_bridge(
        pool, selected.model, train_config, "confirmation", 300, "cpu"
    )
    candidate = train_b2(
        pool,
        selected.model,
        train_config,
        300,
        "cpu",
        tau=4.0,
        lambda_event=1e-3,
        config=config,
    )
    assert baseline.history[0]["loss"] == candidate.history[0]["flow_loss"]
    assert candidate.history[0]["correction_loss"] is not None
    assert candidate.history[0]["loss"] != candidate.history[0]["flow_loss"]


def test_confirmation_allocation_requires_fresh_source_and_is_disjoint() -> None:
    config = _small_config()
    try:
        fresh_evaluation_allocation(
            100,
            "cifar10-test",
            config,
            generated_samples=8,
            reference_samples=16,
            control_groups=3,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("B2 accepted B1's exhausted official test set")
    allocation = fresh_evaluation_allocation(
        80,
        "synthetic-fresh-test",
        config,
        generated_samples=8,
        reference_samples=16,
        control_groups=3,
    )
    allocation.assert_disjoint()
    assert len(allocation.unused) == 16


def test_paired_drift_audit_uses_identical_roles_and_floor() -> None:
    config = _small_config()
    target = torch.linspace(0.5, 1.5, 80 * 2).reshape(80, 2)
    allocation = fresh_evaluation_allocation(
        len(target),
        "synthetic-fresh-audit",
        config,
        generated_samples=8,
        reference_samples=16,
        control_groups=3,
    )
    candidate_batches = [
        torch.ones(config.audit_batch, 2) for _ in range(config.audit_replicates)
    ]
    baseline_batches = [
        torch.full((config.audit_batch, 2), -3.0)
        for _ in range(config.audit_replicates)
    ]
    candidate = drift_energy_audit_suite(
        candidate_batches, target, allocation, 1.0, 300, config, "cpu"
    )
    baseline = drift_energy_audit_suite(
        baseline_batches, target, allocation, 1.0, 300, config, "cpu"
    )
    comparison = compare_drift_audits(candidate, baseline, config)
    assert comparison["passes"]
    assert comparison["paired_wins"] == config.audit_replicates


def test_default_b2_configuration_is_theory_aligned() -> None:
    config = b2_config()
    assert config.target_ess_fraction == 0.60
    assert config.probe_noise_std > 0
    assert config.recall_noninferiority_margin == 0.025
    assert config.effective_gradient_ratio == 0.025


def test_fresh_loader_normalizes_and_hashes_external_bytes() -> None:
    images = np.arange(12 * 8 * 8 * 3, dtype=np.uint8).reshape(12, 8, 8, 3)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "external.npz"
        np.savez(path, images=images)
        pool, record = load_fresh_pool(path, "external-test-v1", 8)
    assert pool.shape == (12, 3, 8, 8)
    assert pool.dtype == torch.float32
    assert -1 <= float(pool.min()) <= float(pool.max()) <= 1
    assert record["source_id"] == "external-test-v1"
    assert len(record["sha256"]) == 64

    floating = images.astype(np.float32) / 255.0
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "external_float.npy"
        np.save(path, floating)
        try:
            load_fresh_pool(path, "external-float-v1", 8)
        except ValueError:
            pass
        else:
            raise AssertionError("B2 silently guessed a floating-point encoding")
        converted, converted_record = load_fresh_pool(
            path, "external-float-v1", 8, "zero-one"
        )
    assert -1 <= float(converted.min()) <= float(converted.max()) <= 1
    assert converted_record["applied_encoding"] == ("float-zero-one-to-minus-one-one")


if __name__ == "__main__":
    main(__name__, globals())
