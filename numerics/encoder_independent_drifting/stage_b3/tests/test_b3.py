"""Mechanical and methodological tests for the corrected B3 design."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import torch

from ...config import GeometryConfig
from ...fid import kid_from_features
from ...fixed_features import build_family
from ...kernels import BlockKernel
from ...models import OneStepGenerator
from ..artifacts import load_reference_artifacts, phase30_source_comparison
from ..core import (
    B3_ARMS,
    B3Config,
    assert_samplewise_generator,
    b3_seed,
    build_generator,
    construct_full_teacher,
    regression_backward,
    train_b3_arm,
)
from ..evaluation import (
    cross_architecture_kid_interval,
    cross_architecture_membership_intervals,
    evaluation_allocation,
)


def _operator():
    branch = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace"), 3
    ).branches[0]
    kernel = BlockKernel(
        base="smooth_laplace",
        taus=torch.tensor([2.0]),
        weights=torch.tensor([1.0]),
        eps=1e-3,
        combine="sum",
    )
    return branch, kernel


def _models():
    first = OneStepGenerator(8, 3, 8, 16, 123)
    second = OneStepGenerator(8, 3, 8, 16, 123)
    second.load_state_dict(copy.deepcopy(first.state_dict()))
    return first, second


def test_capacity_arm_matches_bridge_without_changing_cloud() -> None:
    config = B3Config()
    native, capacity = B3_ARMS
    native_model = build_generator(native, config, 600, "cpu")
    capacity_model = build_generator(capacity, config, 600, "cpu")
    assert native.field_cloud == capacity.field_cloud == 256
    assert native.backward_microbatch == capacity.backward_microbatch == 256
    assert native_model.parameter_count() == 146_691
    assert capacity_model.parameter_count() == 3_864_003
    gap = abs(capacity_model.parameter_count() - config.bridge_parameter_count)
    assert (
        gap / config.bridge_parameter_count
        < config.capacity_parameter_tolerance_fraction
    )
    assert_samplewise_generator(capacity_model)


def test_microbatch_matches_full_loss_gradients_and_adam_update() -> None:
    torch.manual_seed(5)
    full, chunked = _models()
    latent = torch.randn(8, 8)
    positive = torch.randn(5, 3, 8, 8)
    branch, kernel = _operator()
    teacher, _, _ = construct_full_teacher(full, latent, positive, branch, kernel, 0.5)

    with torch.no_grad():
        whole = full(latent)
        pieces = torch.cat([full(latent[i : i + 2]) for i in range(0, 8, 2)])
    torch.testing.assert_close(whole, pieces, rtol=1e-6, atol=2e-7)

    full.zero_grad(set_to_none=True)
    full_output = full(latent)
    full_loss = (full_output - teacher).square().flatten(1).sum(1).mean()
    full_loss.backward()
    chunked.zero_grad(set_to_none=True)
    chunk_loss = regression_backward(chunked, latent, teacher, 2)
    assert abs(float(full_loss.detach()) - chunk_loss) < 1e-5
    for first, second in zip(full.parameters(), chunked.parameters(), strict=True):
        torch.testing.assert_close(first.grad, second.grad, rtol=2e-5, atol=2e-6)

    first_optimizer = torch.optim.Adam(full.parameters(), lr=2e-3)
    second_optimizer = torch.optim.Adam(chunked.parameters(), lr=2e-3)
    first_optimizer.step()
    second_optimizer.step()
    for first, second in zip(full.parameters(), chunked.parameters(), strict=True):
        torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-6)


def test_recovery_replays_optimizer_rng_and_remaining_updates_exactly() -> None:
    branch, kernel = _operator()
    config = replace(
        B3Config(),
        steps=4,
        checkpoint_steps=(4,),
        positives=4,
        latent_dim=8,
        image_size=8,
        log_every=4,
        recovery_every=2,
    )
    arm = replace(B3_ARMS[0], width=16, field_cloud=8, backward_microbatch=8)
    pool = torch.randn(20, 3, 8, 8, generator=torch.Generator().manual_seed(51))
    captured = {}

    def capture(step, model, optimizer, target_rng, _record, history, wall_seconds):
        if step == 2:
            captured.update(
                {
                    "step": step,
                    "state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "target_rng_state": copy.deepcopy(target_rng.bit_generator.state),
                    "history": copy.deepcopy(history),
                    "wall_seconds": wall_seconds,
                }
            )

    uninterrupted = train_b3_arm(
        pool,
        600,
        arm,
        config,
        branch,
        kernel,
        "cpu",
        recovery=capture,
    )
    resumed = train_b3_arm(
        pool,
        600,
        arm,
        config,
        branch,
        kernel,
        "cpu",
        resume=captured,
    )
    assert resumed.resumed_from_step == 2
    for expected, actual in zip(
        uninterrupted.model.parameters(), resumed.model.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert [row["loss"] for row in uninterrupted.history] == [
        row["loss"] for row in resumed.history
    ]


def test_shared_streams_replay_but_roles_do_not_alias() -> None:
    assert b3_seed(600, "training-latent", 17) == b3_seed(600, "training-latent", 17)
    values = {
        b3_seed(600, "training-latent", 17),
        b3_seed(600, "target-training"),
        b3_seed(600, "calibration-indices"),
        b3_seed(600, "evaluation-latent"),
        b3_seed(601, "training-latent", 17),
    }
    assert len(values) == 5


def test_allocation_is_disjoint_and_cross_architecture_methods_are_unpaired() -> None:
    allocation = evaluation_allocation(
        200,
        "test-b3-source",
        units=3,
        generated_samples=16,
        reference_samples=32,
        audit_replicates=2,
        audit_batch=8,
    )
    allocation.assert_disjoint()
    rng = np.random.default_rng(4)
    candidate = rng.normal(size=(32, 6))
    baseline = rng.normal(size=(32, 6))
    reference = rng.normal(size=(64, 6))
    pr = cross_architecture_membership_intervals(
        candidate, baseline, reference, seed=9, replicates=100, k=3
    )
    kid = cross_architecture_kid_interval(
        candidate, baseline, reference, seed=10, replicates=100
    )
    assert pr["precision"]["method"] == "independent-generated-membership-bootstrap"
    assert pr["recall"]["method"] == "shared-reference-membership-bootstrap"
    assert kid["method"].startswith("independent-generated")
    expected = kid_from_features(candidate, reference) - kid_from_features(
        baseline, reference
    )
    assert abs(kid["difference"] - expected) < 1e-12


def test_frozen_references_and_phase30_provenance_are_auditable() -> None:
    references = load_reference_artifacts()
    assert set(references) == {"B0", "B1", "B2"}
    assert all(
        set(item["checkpoints"]) == {"300", "301", "302"}
        for item in references.values()
    )
    comparison = phase30_source_comparison()
    assert comparison["files"]
    assert all(row["historical_sha256"] for row in comparison["files"].values())
