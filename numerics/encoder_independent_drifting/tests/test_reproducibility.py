"""Plan section 9 reproducibility tests.

Everything a report cites must be reconstructible from a recorded seed and
configuration digest.  These tests check that the seed derivation is
deterministic and well separated, that a whole arm replays bit-for-bit, and
that the audit bank and the target samplers are stable.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import datasets as D
from .. import spectral_anchor as SA
from ..config import (
    AUDIT_SEED, AnchorConfig, GeometryConfig, MASTER_SEED, TrainConfig,
    config_digest, derive_seed,
)
from ..fixed_features import build_family
from ..models import OneStepGenerator, sample_latent
from ..train import arm_by_id, phase1_arms, train_arm
from .harness import main

SMALL = TrainConfig(steps=5, batch=16, controller_batch=8, audit_batch=8,
                    latent_dim=8, width=16, image_size=16, channels=3,
                    eval_samples=32)


def test_seed_derivation_is_deterministic_and_separated() -> None:
    """1. derived seeds are stable, label-sensitive and non-colliding"""
    assert derive_seed(MASTER_SEED, "a", 1) == derive_seed(MASTER_SEED, "a", 1)
    assert derive_seed(MASTER_SEED, "a", 1) != derive_seed(MASTER_SEED, "a", 2)
    assert derive_seed(MASTER_SEED, "a") != derive_seed(MASTER_SEED, "b")
    assert derive_seed(MASTER_SEED, "a") != derive_seed(AUDIT_SEED, "a")
    labels = [derive_seed(MASTER_SEED, "arm", i) for i in range(500)]
    assert len(set(labels)) == 500
    assert all(0 <= s < 2 ** 63 for s in labels)


def test_config_digest_is_canonical() -> None:
    """2. the configuration digest ignores ordering but not content"""
    left = GeometryConfig(family="wavelet", scales=3)
    right = GeometryConfig(family="wavelet", scales=3)
    assert config_digest(left) == config_digest(right)
    assert config_digest(left) != config_digest(
        GeometryConfig(family="wavelet", scales=2))
    assert len(config_digest(left)) == 64


def test_targets_are_reproducible() -> None:
    """3. every target sampler replays from its generator seed"""
    for target in D.suite():
        a = target.sample(32, np.random.default_rng(4))
        b = target.sample(32, np.random.default_rng(4))
        assert torch.equal(a, b), target.name
        assert torch.isfinite(a).all(), target.name
        c = target.sample(32, np.random.default_rng(5))
        assert not torch.equal(a, c), target.name


def test_target_suite_covers_the_declared_probes() -> None:
    """4. the Phase-1 target list matches the plan's probe list"""
    names = {t.name for t in D.suite()}
    expected = {"checkerboard", "pinwheel", "rings_islands", "texture_blocks",
                "patch_layout", "color_layout", "phase_structured",
                "rare_object", "deformed"}
    assert names == expected, names
    rare = D.named("rare_object")
    assert abs(float(rare.component_weights[1]) - D.RARE_WEIGHT) < 1e-12


def test_generator_is_seed_reproducible_and_one_step() -> None:
    """5. generator init and sampling replay; inference is a single pass"""
    a = OneStepGenerator(8, 3, 16, 16, 99)
    b = OneStepGenerator(8, 3, 16, 16, 99)
    c = OneStepGenerator(8, 3, 16, 16, 100)
    latent = sample_latent(4, 8, 7)
    assert torch.equal(sample_latent(4, 8, 7), latent)
    with torch.no_grad():
        assert torch.equal(a(latent), b(latent))
        assert not torch.equal(a(latent), c(latent))
        assert a(latent).shape == (4, 3, 16, 16)


def test_generator_refuses_unreachable_image_sizes() -> None:
    """5b. an image size the upsampling cannot reach is refused

    Regression: 24 passed the old divisibility check and the generator
    silently emitted 32x32, which only surfaced as a matrix-shape error deep
    inside a kernel during a Phase-3 run.
    """
    for size in (4, 8, 16, 32):
        model = OneStepGenerator(8, 3, size, 16, 5)
        with torch.no_grad():
            assert model(sample_latent(2, 8, 1)).shape[-1] == size, size
    for size in (6, 12, 20, 24, 48):
        try:
            OneStepGenerator(8, 3, size, 16, 5)
        except ValueError:
            continue
        raise AssertionError(f"accepted unreachable image size {size}")


def test_feature_families_are_reproducible() -> None:
    """6. fixed families rebuild identically, including random-conv weights"""
    images = torch.randn(4, 3, 16, 16,
                         generator=torch.Generator().manual_seed(2))
    for name in ("wavelet", "randconv", "dictionary"):
        config = GeometryConfig(family=name)
        left = build_family(config, 3, seed_label="repro")
        right = build_family(config, 3, seed_label="repro")
        other = build_family(config, 3, seed_label="different")
        assert torch.equal(left.branches[0].flat(images),
                           right.branches[0].flat(images)), name
        if name in ("randconv", "dictionary"):
            same = torch.equal(left.branches[-2].flat(images),
                               other.branches[-2].flat(images))
            assert not same, f"{name} ignored its seed label"


def test_audit_bank_is_stable_and_independent() -> None:
    """7. the audit bank replays and differs from every training bank"""
    config = AnchorConfig(features=64, audit_features=128)
    audit_a = SA.build_bank(config, 48, 1.0, derive_seed(AUDIT_SEED, "audit"),
                            features=config.audit_features)
    audit_b = SA.build_bank(config, 48, 1.0, derive_seed(AUDIT_SEED, "audit"),
                            features=config.audit_features)
    assert torch.equal(audit_a.frequencies, audit_b.frequencies)
    for seed in range(20):
        training = SA.build_bank(config, 48, 1.0,
                                 derive_seed(seed, "anchor-bank"))
        assert not torch.equal(training.frequencies[:64],
                               audit_a.frequencies[:64])


def test_whole_arm_replays_bit_for_bit() -> None:
    """8. a full arm run is bit-identical from the same seed"""
    target = D.texture_blocks()
    for arm_id in ("A1", "A5"):
        arm = arm_by_id(arm_id)
        left = train_arm(arm, target, SMALL, seed=21, log_every=2)
        right = train_arm(arm, target, SMALL, seed=21, log_every=2)
        latent = sample_latent(8, SMALL.latent_dim, 3)
        with torch.no_grad():
            assert torch.equal(left.model(latent), right.model(latent)), arm_id
        assert left.ledger.as_dict() == right.ledger.as_dict(), arm_id
        assert left.log.series.keys() == right.log.series.keys(), arm_id


def test_different_seeds_give_different_runs() -> None:
    """9. the seed actually drives the run"""
    arm = arm_by_id("A4")
    left = train_arm(arm, D.pinwheel(), SMALL, seed=31, log_every=100)
    right = train_arm(arm, D.pinwheel(), SMALL, seed=32, log_every=100)
    latent = sample_latent(8, SMALL.latent_dim, 3)
    with torch.no_grad():
        assert not torch.equal(left.model(latent), right.model(latent))


def test_arm_registry_is_well_formed() -> None:
    """10. the A0-A8 registry matches the plan's table"""
    arms = phase1_arms()
    assert [a.arm_id for a in arms] == [f"A{i}" for i in range(9)]
    by_id = {a.arm_id: a for a in arms}
    assert not by_id["A0"].use_anchor and by_id["A0"].geometry.family == "raw"
    assert by_id["A0"].field.direction_mode == "standard"
    assert by_id["A1"].field.direction_mode == "kernel_gradient"
    assert by_id["A2"].use_anchor and by_id["A2"].geometry is None
    assert by_id["A3"].field.direction_mode == "standard"
    assert by_id["A4"].geometry.family == "wavelet"
    assert by_id["A5"].use_anchor and by_id["A5"].geometry.family == "wavelet"
    assert by_id["A6"].geometry.family == "randconv"
    assert by_id["A7"].mixture.adaptive
    assert by_id["A8"].geometry.family == "reference_encoder"
    assert not any(a.mixture.adaptive for a in arms if a.arm_id != "A7")


if __name__ == "__main__":
    main("reproducibility", dict(globals()))
