"""Plan section 9, P0.4 unit tests for cross-fitting during training.

The point of these tests is that cross-fitting is *structural*: a controller
example can never reach the field batch, and the guarantee is enforced by
the code rather than by the runner remembering to do the right thing.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import datasets as D
from ..adaptive_mixture import BatchRoles, split_roles
from ..config import MixtureConfig, TrainConfig
from ..train import arm_by_id, train_arm
from .harness import main

SMALL = TrainConfig(steps=6, batch=16, controller_batch=12, audit_batch=12,
                    latent_dim=8, width=16, image_size=16, channels=3,
                    eval_samples=32)


def test_roles_are_disjoint() -> None:
    """1. the three role sets never share an example"""
    rng = np.random.default_rng(0)
    for _ in range(20):
        roles = split_roles(64, 16, 24, 8, rng)
        roles.assert_disjoint()
        combined = np.concatenate(
            [roles.controller, roles.field, roles.audit])
        assert len(np.unique(combined)) == len(combined)
        assert combined.max() < 64


def test_overlapping_roles_are_rejected() -> None:
    """2. an overlapping role assignment raises rather than proceeding"""
    roles = BatchRoles(controller=np.array([1, 2]), field=np.array([2, 3]),
                       audit=np.array([4]))
    try:
        roles.assert_disjoint()
    except ValueError:
        pass
    else:
        raise AssertionError("accepted overlapping cross-fitting roles")


def test_insufficient_examples_are_rejected() -> None:
    """3. asking for more disjoint examples than exist raises"""
    try:
        split_roles(10, 5, 5, 5, np.random.default_rng(1))
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an impossible role split")


def test_controller_examples_never_enter_the_field_batch() -> None:
    """4. over a whole run, no controller index reaches the field batch"""
    rng = np.random.default_rng(7)
    for _ in range(200):
        roles = split_roles(96, 24, 48, 24, rng)
        assert not set(roles.controller.tolist()) & set(
            roles.field.tolist())


def test_disabling_adaptation_reproduces_the_fixed_run_exactly() -> None:
    """5. an adaptive arm with adaptation off equals the fixed-weight arm"""
    target = D.pinwheel()
    adaptive_arm = arm_by_id("A7")
    fixed_arm = type(adaptive_arm)(
        adaptive_arm.arm_id, adaptive_arm.use_anchor, adaptive_arm.geometry,
        adaptive_arm.field, MixtureConfig(adaptive=False),
        adaptive_arm.objective, adaptive_arm.anchor, adaptive_arm.note)
    left = train_arm(adaptive_arm, target, SMALL, seed=3, log_every=100)
    right = train_arm(fixed_arm, target, SMALL, seed=3, log_every=100)
    again = train_arm(fixed_arm, target, SMALL, seed=3, log_every=100)
    latent = torch.randn(4, SMALL.latent_dim,
                         generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        same = float((right.model(latent) - again.model(latent))
                     .abs().max())
    assert same == 0.0, same

    # The fixed arm's weights stay uniform throughout; the adaptive arm's
    # must actually have moved, otherwise "adaptation off reproduces fixed
    # weights" would hold only because adaptation does nothing.  Weights are
    # stored in float32, so the tolerance is float32-scale.
    fixed_weights = right.parts.controller.as_dict()
    adaptive_weights = left.parts.controller.as_dict()
    uniform = 1.0 / len(fixed_weights)
    assert max(abs(w - uniform) for w in fixed_weights.values()) < 1e-6, (
        fixed_weights)
    assert max(abs(w - uniform) for w in adaptive_weights.values()) > 1e-3, (
        adaptive_weights)


def test_controller_decisions_are_reproducible() -> None:
    """6. the controller history replays exactly from the recorded seed"""
    target = D.checkerboard()
    arm = arm_by_id("A7")
    left = train_arm(arm, target, SMALL, seed=11, log_every=100)
    right = train_arm(arm, target, SMALL, seed=11, log_every=100)
    assert len(left.controller_history) == len(right.controller_history)
    for a, b in zip(left.controller_history, right.controller_history):
        assert a["step"] == b["step"]
        assert a["after"] == b["after"], (a["after"], b["after"])


def test_ledger_separates_examples_by_role() -> None:
    """7. the cost ledger counts controller and audit examples separately"""
    outcome = train_arm(arm_by_id("A7"), D.rings_islands(), SMALL, seed=5,
                        log_every=2)
    ledger = outcome.ledger
    assert ledger.target_examples_field == SMALL.steps * SMALL.batch
    assert ledger.target_examples_controller == (
        SMALL.steps * SMALL.controller_batch)
    assert ledger.target_examples_audit == SMALL.steps * SMALL.audit_batch
    assert ledger.kernel_pairs > 0 and ledger.generator_forwards > 0


if __name__ == "__main__":
    main("cross-fit controller (P0.4)", dict(globals()))
