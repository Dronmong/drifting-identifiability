"""Plan section 9, P0.4 unit tests for the weight simplex and the objective.

These cover the two claims that carry the plan's exact-zero argument: the
mixture stays a nonnegative combination of positive-definite kernels, and
the total loss is a sum of nonnegative terms so that zero total loss forces
zero anchor loss.
"""

from __future__ import annotations

import torch

from .. import fixed_features as FF
from .. import kernels as K
from .. import spectral_anchor as SA
from ..adaptive_mixture import MixtureController, project_simplex_with_floor
from ..config import AnchorConfig, GeometryConfig, MixtureConfig
from ..objectives import (
    branch_gradient_report, geometry_loss, range_regularization,
    reported_geometry_loss, total_objective,
)
from .harness import main

CHANNELS = 3


def _images(n: int, seed: int, shift: float = 0.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, CHANNELS, 8, 8, generator=g) + shift


def test_projection_stays_on_the_simplex() -> None:
    """1. the projection always returns a simplex point"""
    g = torch.Generator().manual_seed(1)
    for size in (2, 5, 11):
        for _ in range(20):
            raw = torch.randn(size, generator=g) * 3
            projected = project_simplex_with_floor(raw, 0.05)
            assert abs(float(projected.sum()) - 1.0) < 1e-6
            assert float(projected.min()) >= 0.05 - 1e-6


def test_floor_is_never_violated() -> None:
    """2. the declared floor survives an extreme utility"""
    controller = MixtureController(
        ("a", "b", "c"), MixtureConfig(adaptive=True, floor=0.1, ema=1.0))
    for _ in range(50):
        controller.update({"a": 1e6, "b": 0.0, "c": 0.0}, 1)
    weights = controller.weights
    assert float(weights.min()) >= 0.1 - 1e-6, weights
    assert abs(float(weights.sum()) - 1.0) < 1e-6


def test_infeasible_floor_is_rejected() -> None:
    """3. a floor that cannot fit on the simplex is refused"""
    try:
        project_simplex_with_floor(torch.ones(5), 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an infeasible floor")


def test_duplicate_branches_are_rejected() -> None:
    """4. duplicate branch names cannot silently double a weight"""
    try:
        MixtureController(("a", "a"), MixtureConfig())
    except ValueError:
        pass
    else:
        raise AssertionError("accepted duplicate branch names")


def test_disabling_adaptation_reproduces_fixed_weights() -> None:
    """5. adaptation off leaves the weights exactly unchanged"""
    controller = MixtureController(
        ("a", "b", "c"), MixtureConfig(adaptive=False, floor=0.05))
    before = controller.as_dict()
    for step in range(10):
        controller.update({"a": 5.0, "b": 0.1, "c": 99.0}, step)
    assert controller.as_dict() == before


def test_adaptation_follows_utility() -> None:
    """6. an adaptive controller moves weight toward the useful branch"""
    controller = MixtureController(
        ("a", "b"), MixtureConfig(adaptive=True, floor=0.02, ema=0.5))
    for step in range(20):
        controller.update({"a": 10.0, "b": 0.0}, step)
    weights = controller.as_dict()
    assert weights["a"] > weights["b"], weights
    assert controller.diagnostics()["mixture_floor_active"]


def test_geometry_loss_is_nonnegative_and_zero_at_zero_drift() -> None:
    """7. the stop-gradient regression is >= 0 and vanishes with the field"""
    output = _images(6, 2)
    assert float(geometry_loss(output, torch.zeros_like(output), 0.5)) == 0.0
    assert float(geometry_loss(output, _images(6, 3), 0.5)) > 0.0


def test_geometry_loss_gradient_moves_along_the_field() -> None:
    """8. its gradient is exactly -2 eta V / n w.r.t. the output"""
    output = _images(5, 4).requires_grad_(True)
    drift = _images(5, 5)
    eta = 0.3
    geometry_loss(output, drift, eta).backward()
    expected = -2.0 * eta * drift / len(drift)
    assert float((output.grad - expected).abs().max()) < 1e-6


def test_trained_geometry_loss_is_pinned_under_rms_normalization() -> None:
    """8b. R2: the RMS-normalized loss is the constant eta^2 -- the defect

    This is the Phase-1 defect made explicit, so it cannot silently return.
    """
    g = torch.Generator().manual_seed(21)
    for eta in (0.25, 0.5, 1.0):
        for _ in range(3):
            drift = torch.randn(32, CHANNELS, 8, 8, generator=g)
            rms = float(torch.sqrt(
                (drift.reshape(32, -1) ** 2).sum(dim=1).mean()))
            value = float(geometry_loss(
                _images(32, 22), drift / rms, eta))
            assert abs(value - eta ** 2) < 1e-4, (eta, value)


def test_reported_geometry_loss_tracks_the_real_field() -> None:
    """8c. R2: the reported loss moves with the field and vanishes with it"""
    assert reported_geometry_loss(0.0, 0.5) == 0.0
    small = reported_geometry_loss(0.1, 0.5)
    large = reported_geometry_loss(2.0, 0.5)
    assert 0.0 < small < large, (small, large)
    # It agrees with the unnormalized stop-gradient loss it stands in for.
    g = torch.Generator().manual_seed(23)
    drift = torch.randn(16, CHANNELS, 8, 8, generator=g)
    rms = float(torch.sqrt((drift.reshape(16, -1) ** 2).sum(dim=1).mean()))
    direct = float(geometry_loss(_images(16, 24), drift, 0.7))
    assert abs(reported_geometry_loss(rms, 0.7) - direct) < 1e-3 * direct


def test_reported_loss_is_non_constant_along_a_real_trajectory() -> None:
    """8d. R2: along an actual run the reported loss varies and decreases

    The old test only checked ``geometry_loss(output, zeros) == 0``, which a
    literal zero drift satisfies -- so it passed against a pipeline in which
    the loss was constant.  This one reads the training log.
    """
    from .. import datasets as D
    from ..config import TrainConfig
    from ..train import arm_by_id, train_arm
    config = TrainConfig(steps=40, batch=32, controller_batch=16,
                         audit_batch=16, latent_dim=8, width=16,
                         eval_samples=32)
    outcome = train_arm(arm_by_id("A4"), D.checkerboard(), config, seed=5,
                        log_every=4)
    pinned = outcome.log.series.get("loss_loss_geometry", [])
    honest = outcome.log.series.get("loss_geometry_unnormalized_total", [])
    assert pinned and len(set(round(v, 6) for v in pinned)) == 1, (
        "the trained loss should still be the constant eta^2")
    assert honest, "no unnormalized geometry loss was recorded"
    assert len(set(round(v, 6) for v in honest)) > 1, honest
    assert min(honest) < max(honest)


def test_total_loss_is_nonnegative_and_forces_the_anchor() -> None:
    """9. zero total loss forces zero anchor loss when lambda_A > 0"""
    output = _images(8, 6)
    target = _images(8, 7, shift=0.4)
    bank = SA.build_bank(AnchorConfig(features=64), output[0].numel(),
                         1.0, 3)
    drifts = {"g": _images(8, 8)}
    total, parts = total_objective(
        output, bank=bank, target=target, drifts=drifts,
        weights={"g": 1.0}, lambda_anchor=1.0, lambda_geometry=1.0,
        lambda_regularization=0.5, eta=0.4)
    assert float(total) >= 0.0
    assert all(v >= 0.0 for k, v in parts.items() if k.startswith("loss_"))
    # Zero total loss is only reachable when every nonnegative term is zero.
    zero, zero_parts = total_objective(
        output, bank=bank, target=output, drifts={"g": torch.zeros_like(
            output)}, weights={"g": 1.0}, lambda_anchor=1.0,
        lambda_geometry=1.0, lambda_regularization=0.0, eta=0.4)
    assert float(zero) == 0.0 and zero_parts["loss_anchor"] == 0.0


def test_unbiased_estimator_is_refused_by_the_objective() -> None:
    """10. the objective refuses a possibly-negative anchor estimator"""
    output = _images(4, 9)
    bank = SA.build_bank(AnchorConfig(features=32), output[0].numel(), 1.0, 4)
    try:
        total_objective(output, bank=bank, target=_images(4, 10), drifts={},
                        weights={}, lambda_anchor=1.0, lambda_geometry=1.0,
                        lambda_regularization=0.0, eta=0.5,
                        anchor_estimator="unbiased")
    except ValueError:
        pass
    else:
        raise AssertionError("the objective accepted the U-statistic")


def test_regularization_is_nonnegative_and_inactive_in_range() -> None:
    """11. the regularizer is zero inside the declared range"""
    inside = torch.full((3, CHANNELS, 8, 8), 0.5)
    outside = torch.full((3, CHANNELS, 8, 8), 9.0)
    assert float(range_regularization(inside, 4.0)) == 0.0
    assert float(range_regularization(outside, 4.0)) > 0.0


def test_gradient_report_detects_cancellation() -> None:
    """12. opposed branches are reported as a negative cosine"""
    output = _images(6, 11)
    drift = _images(6, 12)
    report = branch_gradient_report(
        output, bank=None, target=None,
        drifts={"a": drift, "b": -drift}, weights={"a": 0.5, "b": 0.5},
        lambda_anchor=0.0, lambda_geometry=1.0, eta=0.5)
    assert report["branch_cosine_min"] < -0.99, report["branch_cosine_min"]


def test_mixture_weights_keep_the_kernel_positive_definite() -> None:
    """13. any simplex weighting leaves the block kernel PSD"""
    branch = FF.build_family(
        GeometryConfig(family="wavelet", scales=2, pool=2),
        CHANNELS).branches[0]
    samples = _images(20, 13)
    base = K.calibrate_block_kernel(branch, samples, "smooth_laplace",
                                    0.5, 1.0, 1e-3)
    g = torch.Generator().manual_seed(5)
    for combine in ("sum", "product"):
        for _ in range(3):
            weights = project_simplex_with_floor(
                torch.rand(base.n_blocks, generator=g), 0.01)
            kernel = K.BlockKernel(base.base, base.taus, weights, base.eps,
                                   combine)
            assert K.min_eigenvalue(kernel, branch, samples) > -1e-8, combine


def test_geometric_multipliers_are_centred_and_spanning() -> None:
    """14. the declared bandwidth ladder is geometric and centred on 1"""
    assert K.geometric_multipliers(1, 4.0) == (1.0,)
    assert K.geometric_multipliers(5, 4.0) == (0.25, 0.5, 1.0, 2.0, 4.0)
    ladder = K.geometric_multipliers(3, 2.0)
    assert ladder == (0.5, 1.0, 2.0), ladder
    for levels in (2, 3, 5, 7):
        values = K.geometric_multipliers(levels, 4.0)
        ratios = [b / a for a, b in zip(values, values[1:])]
        assert max(ratios) - min(ratios) < 1e-9, ratios


def test_bandwidth_mixture_repeats_blocks_without_copying() -> None:
    """15. the mixture branch presents each block once per level"""
    branch = FF.build_family(
        GeometryConfig(family="wavelet", scales=2, pool=2),
        CHANNELS).branches[0]
    samples = _images(12, 21)
    base_count = len(branch.blocks(samples))
    for levels in (1, 3, 5):
        mixed = FF.bandwidth_mixture(branch, levels)
        blocks = mixed.blocks(samples)
        assert len(blocks) == base_count * levels
        # Within one extraction the repeats must be the SAME tensor: the
        # mixture costs Gram time, never feature time or memory.  (Across
        # calls they differ -- `extract` re-runs -- so identity is only
        # meaningful inside a single call.)
        for start in range(0, len(blocks), levels):
            group = blocks[start:start + levels]
            assert all(block is group[0] for block in group)
        assert all(
            torch.equal(blocks[index * levels], original)
            for index, original in enumerate(branch.blocks(samples)))


def test_tau_multipliers_scale_bandwidths_and_keep_the_ess_target() -> None:
    """16. a bandwidth mixture stays PSD and still hits the declared ESS"""
    branch = FF.build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace"),
        CHANNELS).branches[0]
    samples = _images(64, 22)
    for levels in (3, 5):
        mixed = FF.bandwidth_mixture(branch, levels)
        multipliers = K.geometric_multipliers(levels, 4.0)
        kernel = K.calibrate_block_kernel(
            mixed, samples, "smooth_laplace", 0.5, 1.0, 1e-3, combine="sum",
            target_ess_fraction=0.5, tau_multipliers=multipliers)
        assert kernel.n_blocks == levels
        # The ESS solve must account for the WHOLE mixture, not one member.
        realized = K.median_ess_fraction(kernel, mixed.blocks(samples))
        assert abs(realized - 0.5) < 0.02, realized
        # Relative spacing must survive the global calibration factor.
        ratios = kernel.taus / kernel.taus[levels // 2]
        for got, want in zip(ratios.tolist(), multipliers):
            assert abs(got - want) < 1e-4, (got, want)
        assert K.min_eigenvalue(kernel, mixed, samples[:32]) > -1e-8


def test_tau_multipliers_reject_a_bad_pattern() -> None:
    """17. a mismatched or non-positive multiplier pattern is refused"""
    branch = FF.build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace"),
        CHANNELS).branches[0]
    samples = _images(32, 23)
    for bad in ((1.0, 2.0), (0.0,), (-1.0,)):
        try:
            K.calibrate_block_kernel(branch, samples, "smooth_laplace", 0.5,
                                     1.0, 1e-3, tau_multipliers=bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted a bad multiplier pattern {bad}")


def test_ess_self_match_changes_the_number_by_an_order_of_magnitude() -> None:
    """18. the Phase-25 correction: a self-paired Gram's diagonal dominates

    The calibration solves on a target-vs-target Gram, every row of which has a
    zero-distance entry for the sample itself.  With the diagonal counted, the
    declared target is met by that self-match rather than by selectivity among
    distinct samples -- so the number describes nothing the field does, since a
    generated cloud point never has a self-match against the real positives.

    Measured on real CIFAR at the bandwidth the legacy solve picks: 0.05
    declared, 0.60 among actual neighbours, 0.71 in the field.  This pins the
    two apart so the distinction cannot silently regress.
    """
    branch = FF.build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace"),
        CHANNELS).branches[0]
    samples = _images(64, 71)
    blocks = branch.blocks(samples)

    legacy = K.calibrate_block_kernel(
        branch, samples, "smooth_laplace", 0.5, 1.0, 1e-3, combine="sum",
        target_ess_fraction=0.05, exclude_self=False)
    # The legacy solve hits its target only when the diagonal is counted.
    with_diagonal = K.median_ess_fraction(legacy, blocks, exclude_self=False)
    without = K.median_ess_fraction(legacy, blocks, exclude_self=True)
    assert abs(with_diagonal - 0.05) < 0.02, with_diagonal
    assert without > 4.0 * with_diagonal, (without, with_diagonal)

    # The corrected solve hits the target among distinct samples, which is the
    # quantity the field actually operates on.
    corrected = K.calibrate_block_kernel(
        branch, samples, "smooth_laplace", 0.5, 1.0, 1e-3, combine="sum",
        target_ess_fraction=0.05, exclude_self=True)
    assert abs(K.median_ess_fraction(corrected, blocks,
                                     exclude_self=True) - 0.05) < 0.02
    # Correcting it must SHRINK the bandwidth: reaching the same fraction
    # without a free self-match requires a sharper kernel.
    assert float(corrected.taus.median()) < float(legacy.taus.median())

    # exclude_self is meaningless off the diagonal and must be refused rather
    # than silently mis-normalized on a rectangular Gram.
    rectangular = legacy.gram_from_blocks(blocks, branch.blocks(_images(16, 7)))
    try:
        K._row_ess_fraction(rectangular, exclude_self=True)
    except ValueError:
        return
    raise AssertionError("accepted exclude_self on a rectangular Gram")


if __name__ == "__main__":
    main("positive kernel mixture (P0.4)", dict(globals()))
