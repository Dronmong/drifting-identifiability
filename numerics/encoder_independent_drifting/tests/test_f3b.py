"""Regression tests for the corrected Stage-F3B bridge implementation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch import nn

from ..diagnostics import write_json
from ..f3b import (
    TimeConditionedUNet,
    adjudicate_b0,
    bridge_streams,
    confirmation_profile,
    euler_integrate,
    f3b_seed,
    independent_bridge_batch,
    oracle_endpoint,
    profile,
    train_bridge,
)
from ..f3b_evaluation import evaluation_allocation
from ..f3b_freeze import frozen_payload, profile_payload, source_manifest
from .harness import main


def _pool(count: int = 16, size: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.rand(count, 3, size, size, generator=generator) * 2 - 1


def test_velocity_model_is_image_to_image_and_time_conditioned() -> None:
    """1. the replacement is not the old latent-to-image generator"""
    config = profile("smoke").model
    first = TimeConditionedUNet(config, seed=9)
    replay = TimeConditionedUNet(config, seed=9)
    other = TimeConditionedUNet(config, seed=10)
    images = _pool(2)
    early = torch.zeros(2)
    late = torch.ones(2)
    with torch.no_grad():
        output = first(images, early)
        assert output.shape == images.shape
        assert torch.equal(output, replay(images, early))
        assert not torch.equal(output, other(images, early))
        assert not torch.equal(output, first(images + 0.1, early))
        assert not torch.equal(output, first(images, late))


def test_bridge_batch_satisfies_exact_interpolation_identity() -> None:
    """2. Xt and the sampled velocity use the same independent endpoint pair"""
    pool = _pool()
    left = bridge_streams("test", 1)
    right = bridge_streams("test", 1)
    batch = independent_bridge_batch(pool, 5, left, "cpu", False)
    replay = independent_bridge_batch(pool, 5, right, "cpu", False)
    assert all(torch.equal(a, b) for a, b in zip(batch, replay))
    mixed, target, noise, endpoint, time_value = batch
    expected = (1 - time_value[:, None, None, None]) * noise + time_value[
        :, None, None, None
    ] * endpoint
    assert torch.equal(mixed, expected)
    assert torch.equal(target, endpoint - noise)


def test_bridge_seed_roles_are_distinct() -> None:
    """3. data, pairing, time, augmentation, and inference do not share streams"""
    roles = (
        "model-init",
        "data-order",
        "endpoint-noise",
        "bridge-time",
        "augmentation",
        "evaluation-prior",
    )
    values = [f3b_seed("confirmation", 300, role) for role in roles]
    assert len(set(values)) == len(values)
    assert values == [f3b_seed("confirmation", 300, role) for role in roles]
    assert values != [f3b_seed("confirmation", 301, role) for role in roles]


def test_oracle_euler_reconstructs_endpoint_at_every_nfe() -> None:
    """4. the declared time direction and velocity sign are correct"""
    noise = torch.randn(
        3, 2, generator=torch.Generator().manual_seed(1), dtype=torch.float64
    )
    endpoint = torch.randn(
        3, 2, generator=torch.Generator().manual_seed(2), dtype=torch.float64
    )
    for nfe in (1, 2, 7, 32):
        result = oracle_endpoint(noise, endpoint, nfe)
        assert torch.allclose(result, endpoint, atol=2e-15, rtol=2e-15)


def test_generic_euler_calls_the_declared_times() -> None:
    """5. NFE accounting and left-endpoint Euler times are exact"""

    class Constant(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.times = []

        def forward(self, state, time_value):
            self.times.extend(time_value.tolist())
            return torch.ones_like(state) * 2

    model = Constant()
    result = euler_integrate(model, torch.zeros(1, 1), 4)
    assert torch.allclose(result, torch.tensor([[2.0]]))
    assert model.times == [0.0, 0.25, 0.5, 0.75]


def test_smoke_training_is_finite_reproducible_and_checkpointed() -> None:
    """6. a complete bridge optimization replays and invokes only frozen steps"""
    selected = profile("smoke")
    seen_left = []
    seen_right = []
    left = train_bridge(
        _pool(),
        selected.model,
        selected.train,
        "test-train",
        5,
        "cpu",
        checkpoint=lambda step, _model, row: seen_left.append((step, row["loss"])),
    )
    right = train_bridge(
        _pool(),
        selected.model,
        selected.train,
        "test-train",
        5,
        "cpu",
        checkpoint=lambda step, _model, row: seen_right.append((step, row["loss"])),
    )
    assert seen_left == seen_right
    assert [item[0] for item in seen_left] == [2, 4]
    assert all(
        torch.isfinite(value).all() for value in left.model.state_dict().values()
    )
    assert all(
        torch.equal(left.model.state_dict()[name], value)
        for name, value in right.model.state_dict().items()
    )


def test_confirmation_profile_must_come_from_measured_ladder() -> None:
    """7. confirmation cannot introduce a post-development budget or NFE"""
    developed = profile("compact")
    frozen = confirmation_profile(developed, 10_000, 32)
    assert frozen.train.checkpoint_steps == (10_000,)
    assert frozen.evaluation.nfe_ladder == (32,)
    for steps, nfe in ((12_345, 32), (10_000, 17)):
        try:
            confirmation_profile(developed, steps, nfe)
        except ValueError:
            continue
        raise AssertionError("accepted an unmeasured confirmation choice")


def test_evaluation_roles_are_disjoint() -> None:
    """8. development cannot consume the confirmation reference or controls"""
    allocation = evaluation_allocation()
    allocation.assert_disjoint()
    assert len(allocation.development_reference) == 2_048
    assert len(allocation.confirmation_reference) == 2_048
    assert len(allocation.development_controls) == 3
    assert len(allocation.confirmation_controls) == 3


def test_gate_claim_is_narrow_and_requires_vetoes() -> None:
    """9. the 2-of-3 rule cannot turn a failed veto into a reachability pass"""
    rows = [
        {"unit": unit, "metrics": {"recall": recall}, "veto": {"passes": veto}}
        for unit, recall, veto in (
            (300, 0.10, True),
            (301, 0.08, True),
            (302, 0.20, False),
        )
    ]
    verdict = adjudicate_b0(rows, matched_real_recall=0.7)
    assert verdict["decision"] == "PASS"
    assert verdict["units_passing"] == 2
    assert "frozen bridge" in verdict["reading"]
    assert adjudicate_b0(rows, 0.2)["decision"] == "VOID"


def test_freeze_accepts_only_a_measured_non_smoke_selection() -> None:
    """10. mechanics artifacts and unmeasured post-hoc choices are refused"""
    developed = profile("compact")
    base = {
        "status": "f3b-b0-development",
        "mechanics_only": False,
        "profile": profile_payload(developed),
        "provenance": {"source_sha256": source_manifest()},
        "units": [200, 201, 202],
        "evaluations": [
            {"unit": unit, "step": 10_000, "nfe": 32} for unit in (200, 201, 202)
        ],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "development.json"
        write_json(path, base)
        frozen = frozen_payload("compact", 10_000, 32, path)
        assert frozen["selection"] == {"steps": 10_000, "nfe": 32}
        try:
            frozen_payload("compact", 30_000, 32, path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("freeze accepted an unmeasured budget")
        write_json(path, base | {"mechanics_only": True})
        try:
            frozen_payload("compact", 10_000, 32, path)
        except RuntimeError:
            pass
        else:
            raise AssertionError("freeze accepted mechanics-only development")


if __name__ == "__main__":
    main(__name__, globals())
