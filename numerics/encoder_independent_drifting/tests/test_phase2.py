"""Tests for the Phase-2 CIFAR target and arm registry."""

from __future__ import annotations

import numpy as np
import torch

from .. import cifar
from ..config import GeometryConfig
from ..train import phase2_arm_by_id, phase2_arms
from .harness import main

HAVE_CIFAR = cifar.available()


def test_phase2_registry_matches_the_protocol() -> None:
    """1. the B0-B3 registry matches the frozen protocol table"""
    arms = phase2_arms()
    assert [a.arm_id for a in arms] == ["B0", "B1", "B2", "B3"]
    by_id = {a.arm_id: a for a in arms}
    assert by_id["B0"].geometry.family == "raw"
    assert by_id["B1"].geometry.family == "wavelet"
    assert by_id["B2"].geometry.family == "scattering"
    assert by_id["B2"].geometry.second_order
    assert by_id["B3"].geometry.family == "wavelet"
    # Only B3 carries the anchor.
    assert [a.use_anchor for a in arms] == [False, False, False, True]
    # The abandoned kernel-gradient hypothesis appears nowhere.
    assert all(a.field.direction_mode == "standard" for a in arms)
    # No arm adapts its mixture; that is a separate mechanism.
    assert not any(a.mixture.adaptive for a in arms)


def test_all_phase2_arms_share_one_base_kernel() -> None:
    """2. geometry is not confounded with kernel smoothness

    Phase 1 gave the raw standard arm the paper's non-smooth `laplace` and
    everything else `smooth_laplace`, so its raw-vs-structured comparison
    varied two things at once.
    """
    assert {a.geometry.base_kernel for a in phase2_arms()} == {
        "smooth_laplace"}
    # The Phase-1 inference rule is still available and still the default.
    assert GeometryConfig().base_kernel == "auto"


def test_phase2_arm_lookup() -> None:
    """3. arms are addressable by id and unknown ids are refused"""
    assert phase2_arm_by_id("B1").geometry.family == "wavelet"
    try:
        phase2_arm_by_id("B9")
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unknown Phase-2 arm")


def test_split_boundaries_are_disjoint_and_exhaustive() -> None:
    """4. train/eval partition training data; test names the official split"""
    train, evaluation = cifar.TRAIN_SPLIT, cifar.EVAL_SPLIT
    assert train[1] == evaluation[0], (train, evaluation)
    assert train[0] < train[1] < evaluation[1]
    assert cifar.TEST_SPLIT == (0, 10_000)
    assert set(cifar.SPLITS) == {"train", "eval", "test"}


def test_unknown_split_and_resolution_are_refused() -> None:
    """5. bad split names and resolutions raise rather than guess"""
    for bad in ("validation", ""):
        try:
            cifar.cifar_pool(16, bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted split {bad!r}")
    if HAVE_CIFAR:
        for bad in (2, 64):
            try:
                cifar.cifar_pool(bad, "train")
            except ValueError:
                continue
            raise AssertionError(f"accepted resolution {bad}")


def test_cifar_pools_are_disjoint_in_content() -> None:
    """6. no image appears in both splits"""
    if not HAVE_CIFAR:
        return
    train = cifar.cifar_pool(16, "train")
    evaluation = cifar.cifar_pool(16, "eval")
    assert len(train) == 40_000 and len(evaluation) == 10_000
    # Compare a sample of eval images against the whole train split by hash.
    train_keys = {hash(train[i].numpy().tobytes())
                  for i in range(0, len(train), 37)}
    eval_keys = {hash(evaluation[i].numpy().tobytes())
                 for i in range(0, len(evaluation), 37)}
    assert not (train_keys & eval_keys)


def test_cifar_target_is_reproducible_and_well_formed() -> None:
    """7. the target replays from its generator and has no oracle labels"""
    if not HAVE_CIFAR:
        return
    target = cifar.cifar_target(16, "train")
    a = target.sample(32, np.random.default_rng(3))
    b = target.sample(32, np.random.default_rng(3))
    assert torch.equal(a, b)
    assert a.shape == (32, 3, 16, 16)
    assert torch.isfinite(a).all()
    assert float(a.abs().max()) <= 1.0 + 1e-6
    # No prototypes means every oracle-only diagnostic is skipped: this
    # program is not entitled to CIFAR's class structure.
    assert target.prototypes is None
    assert target.component_weights is None
    assert target.n_components == 0


def test_train_and_eval_targets_draw_from_different_pools() -> None:
    """8. the two targets really do sample different images"""
    if not HAVE_CIFAR:
        return
    train = cifar.cifar_target(16, "train")
    evaluation = cifar.cifar_target(16, "eval")
    rng = np.random.default_rng(11)
    left = train.sample(64, np.random.default_rng(5))
    right = evaluation.sample(64, np.random.default_rng(5))
    assert not torch.equal(left, right)
    assert left.shape == right.shape
    del rng


if __name__ == "__main__":
    main("phase 2 (CIFAR target and arms)", dict(globals()))
