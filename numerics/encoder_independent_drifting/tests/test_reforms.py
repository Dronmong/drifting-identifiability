"""Tests for the Phase-1 diagnosis reforms R1 and R4.

R2, R3 and R6 are tested inside the modules they change
(`test_positive_kernel_mixture`, `test_kernel_gradients`,
`test_spectral_anchor`).  This module covers the two reforms that are about
*how a screen is run* rather than about the field itself.
"""

from __future__ import annotations

import numpy as np
import torch

from .. import datasets as D
from .. import metrics as M
from .. import oracle as O
from ..config import MASTER_SEED, TrainConfig
from ..evaluate import evaluation_pools, null_reference
from .harness import main

SMALL = TrainConfig(steps=60, batch=32, controller_batch=16, audit_batch=16,
                    latent_dim=8, width=16, eval_samples=128)


# ---------------------------------------------------------------------------
# R4: the re-specified composite
# ---------------------------------------------------------------------------


def test_v2_drops_spectral_l1_and_scores_nearest_real() -> None:
    """1. R4: scored components are exactly the declared v2 set"""
    assert M.GEOMETRY_SCORE_COMPONENTS_V2 == (
        "ed2", "sw1", "patch_ed2", "nearest_real")
    assert "spectral_l1" not in M.GEOMETRY_SCORE_COMPONENTS_V2
    assert "off_support" not in M.GEOMETRY_SCORE_COMPONENTS_V2
    assert "spectral_l1" in M.GEOMETRY_SCORE_COMPONENTS      # v1 unchanged


def test_v2_refuses_an_untrustworthy_null() -> None:
    """2. R4: a near-zero null is excluded, not turned into a huge ratio"""
    raw = {"ed2": 1.0, "sw1": 1.0, "patch_ed2": 1.0, "nearest_real": 1.0}
    good = {"ed2": 0.02, "sw1": 0.02, "patch_ed2": 0.02, "nearest_real": 1.0}
    bad = dict(good, ed2=1e-9)
    healthy = M.normalized_geometry_score_v2(raw, good)
    degenerate = M.normalized_geometry_score_v2(raw, bad)
    assert healthy["untrustworthy_nulls"] == []
    assert degenerate["untrustworthy_nulls"] == ["ed2"]
    assert "ed2" not in degenerate["geometry_ratios"]
    assert np.isfinite(degenerate["geometry_score"])


def test_nearest_real_is_one_for_a_fresh_real_sample() -> None:
    """3. R4: the graded off-manifold statistic is calibrated at 1.0"""
    target = D.checkerboard()
    rng = np.random.default_rng(4)
    a, b, c = (target.sample(256, rng) for _ in range(3))
    value = M.nearest_real_distance(a, b, c)
    assert 0.7 < value < 1.4, value


def test_nearest_real_does_not_saturate_off_support() -> None:
    """4. R4: it keeps ranking clouds that are all fully off support

    This is the defect it replaces: `off_support` pins at 1.0 for every arm
    that has left the support, so it cannot separate them.
    """
    target = D.checkerboard()
    rng = np.random.default_rng(5)
    reference, null = target.sample(256, rng), target.sample(256, rng)
    near = target.sample(256, rng) + 6.0
    far = target.sample(256, rng) + 24.0
    support_near = M.calibrated_support(
        near, reference, target.sample(128, rng), target.sample(128, rng))
    support_far = M.calibrated_support(
        far, reference, target.sample(128, rng), target.sample(128, rng))
    assert support_near["off_support"] == support_far["off_support"] == 1.0
    graded_near = M.nearest_real_distance(near, reference, null)
    graded_far = M.nearest_real_distance(far, reference, null)
    assert graded_far > 2 * graded_near, (graded_near, graded_far)


def test_component_verdicts_require_a_majority() -> None:
    """5. R4: per-component verdicts are reported, not just the aggregate"""
    candidate = {"ed2": 1.0, "sw1": 1.0, "patch_ed2": 3.0,
                 "nearest_real": 3.0}
    baseline = {"ed2": 2.0, "sw1": 2.0, "patch_ed2": 2.0,
                "nearest_real": 2.0}
    verdict = M.component_verdicts(candidate, baseline)
    assert verdict["components"] == 4
    assert verdict["components_won"] == 2
    assert verdict["majority"] is False
    better = M.component_verdicts(
        {k: v / 4 for k, v in candidate.items()}, baseline)
    assert better["majority"] is True


# ---------------------------------------------------------------------------
# R1: the skyline arm and admissibility
# ---------------------------------------------------------------------------


def test_skyline_uses_the_same_generator_and_no_kernel() -> None:
    """6. R1: the skyline shares the arms' generator and has no kernel"""
    target = D.checkerboard()
    pools = evaluation_pools(target, SMALL, MASTER_SEED)
    null = null_reference(target, pools, MASTER_SEED)
    result = O.train_skyline(target, SMALL, MASTER_SEED, pools, null)
    from ..models import OneStepGenerator
    assert isinstance(result.model, OneStepGenerator)
    assert result.model.image_size == SMALL.image_size
    assert 0.0 <= result.precision <= 1.0
    assert np.isfinite(result.score["geometry_score"])


def test_sliced_w2_is_zero_for_identical_batches() -> None:
    """7. R1: the skyline objective is a genuine discrepancy"""
    g = torch.Generator().manual_seed(6)
    x = torch.randn(32, 3, 16, 16, generator=g)
    dirs = torch.randn(16, 3 * 16 * 16, generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    assert float(O.sliced_w2_loss(x, x, dirs)) < 1e-10
    assert float(O.sliced_w2_loss(x, x + 1.0, dirs)) > 0.1
    try:
        O.sliced_w2_loss(x, x[:16], dirs)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted mismatched batch sizes")


def test_admissibility_verdict_is_precision_based() -> None:
    """8. R1: a target is admissible only if the skyline nears the null

    Coverage saturates at 1.0 long before a generator is good, which is
    exactly how the Phase-1 budget came to be frozen too low.  The bar is
    stated on precision instead.
    """
    target = D.checkerboard()
    pools = evaluation_pools(target, SMALL, MASTER_SEED)
    null = null_reference(target, pools, MASTER_SEED)
    row = O.admissibility(target, SMALL, MASTER_SEED, pools, null)
    assert row["required_precision"] == (
        O.ADMISSIBLE_PRECISION_FRACTION * row["null_precision"])
    assert row["admissible"] == (
        row["skyline_precision"] >= row["required_precision"])
    # At this deliberately tiny budget nothing should qualify.
    assert not row["admissible"], row


def test_admissible_targets_filters() -> None:
    """9. R1: inadmissible targets are dropped, not silently kept"""
    rows = [{"target": "a", "admissible": True},
            {"target": "b", "admissible": False},
            {"target": "c", "admissible": True}]
    assert O.admissible_targets(rows) == ["a", "c"]


# ---------------------------------------------------------------------------
# R8 / R9: audit reforms
# ---------------------------------------------------------------------------


def test_null_is_averaged_over_independent_draws() -> None:
    """10. R8: the null reference is a median over several draws"""
    from ..evaluate import NULL_REPEATS, evaluation_pools, null_reference
    target = D.checkerboard()
    pools = evaluation_pools(target, SMALL, MASTER_SEED)
    assert NULL_REPEATS >= 3
    repeats = [k for k in pools if k.startswith("null_")]
    assert len(repeats) == NULL_REPEATS
    # The repeat pools are genuinely independent draws.
    assert not torch.equal(pools["null_0"], pools["null_1"])
    null = null_reference(target, pools, MASTER_SEED)
    assert "null_spread" in null and null["null_spread"]["ed2"] > 0
    for key in ("ed2", "sw1", "patch_ed2", "nearest_real", "precision"):
        assert key in null and np.isfinite(null[key]), key


def test_averaged_null_tightens_the_reference() -> None:
    """11. R8: averaging moves a fresh real sample closer to score 1.0

    The reference is still not exactly 1.0 -- energy distance between two
    samples of the same law is a near-zero quantity with real spread -- so
    the achievable floor is reported, not assumed.
    """
    from ..config import TrainConfig
    from ..evaluate import evaluation_pools, null_reference
    config = TrainConfig(steps=1, batch=32, controller_batch=16,
                         audit_batch=16, eval_samples=384)
    target = D.pinwheel()
    pools = evaluation_pools(target, config, MASTER_SEED)
    null = null_reference(target, pools, MASTER_SEED)
    rng = np.random.default_rng(7)
    fresh = target.sample(config.eval_samples, rng)
    raw = M.raw_metrics(fresh, pools["eval"], pools["cal_a"], pools["cal_b"],
                        rng, target, target_null=pools["null"])
    score = M.normalized_geometry_score_v2(raw, null)["geometry_score"]
    assert 0.5 < score < 3.0, score


def test_cached_projection_matches_the_uncached_path() -> None:
    """12. R9: the shared factorization changes no value"""
    from .. import fixed_features as FF
    from .. import kernel_gradient as KG
    from .. import kernels as K
    from ..config import GeometryConfig
    g = torch.Generator().manual_seed(51)
    generated = torch.randn(6, 3, 8, 8, generator=g)
    positive = torch.randn(10, 3, 8, 8, generator=g) + 0.3
    branch = FF.build_family(
        GeometryConfig(family="wavelet", scales=2, pool=2), 3).branches[0]
    kernel = K.BlockKernel("smooth_laplace", torch.full((4,), 2.0),
                           torch.full((4,), 0.25), 1e-3)
    anchors = torch.cat([positive, generated], dim=0)
    cached = KG.data_span_basis(generated, anchors)
    a, _ = KG.field(generated, positive, generated, branch, kernel,
                    direction_mode="projected_kernel_gradient",
                    normalization="none", diagnostics=False)
    b, _ = KG.field(generated, positive, generated, branch, kernel,
                    direction_mode="projected_kernel_gradient",
                    normalization="none", diagnostics=False,
                    projection=cached)
    assert float((a - b).abs().max()) < 1e-5, float((a - b).abs().max())


def test_snr_halves_do_not_reuse_the_full_batch_basis() -> None:
    """13. R9: each SNR half rebuilds its own data span"""
    from .. import fixed_features as FF
    from .. import kernel_gradient as KG
    from .. import kernels as K
    from ..config import GeometryConfig
    g = torch.Generator().manual_seed(52)
    generated = torch.randn(6, 3, 8, 8, generator=g)
    positive = torch.randn(12, 3, 8, 8, generator=g) + 0.3
    branch = FF.build_family(
        GeometryConfig(family="wavelet", scales=2, pool=2), 3).branches[0]
    kernel = K.BlockKernel("smooth_laplace", torch.full((4,), 2.0),
                           torch.full((4,), 0.25), 1e-3)
    wrong = KG.data_span_basis(
        generated, torch.cat([positive, generated], dim=0))
    _, with_cache = KG.field_with_snr(
        generated, positive, generated, branch, kernel,
        direction_mode="projected_kernel_gradient", normalization="rms",
        projection=wrong)
    _, without = KG.field_with_snr(
        generated, positive, generated, branch, kernel,
        direction_mode="projected_kernel_gradient", normalization="rms")
    # The SNR is built from half-batch fields, which must be projected onto
    # their own spans; passing a full-batch basis must not change them.
    assert abs(with_cache["drift_snr"] - without["drift_snr"]) < 1e-6, (
        with_cache["drift_snr"], without["drift_snr"])


# ---------------------------------------------------------------------------
# R11: the teacher variance match
# ---------------------------------------------------------------------------


def test_variance_matched_teacher_restores_spread() -> None:
    """14. R11: the correction gives the teacher the reference's spread"""
    from ..objectives import variance_matched_teacher
    g = torch.Generator().manual_seed(81)
    reference = torch.randn(64, 3, 8, 8, generator=g) * 1.4
    contracted = torch.randn(64, 3, 8, 8, generator=g) * 0.3 + 5.0
    fixed = variance_matched_teacher(contracted, reference)

    def spread(x: torch.Tensor) -> float:
        flat = x.reshape(len(x), -1)
        return float((flat - flat.mean(0, keepdim=True)).pow(2).mean().sqrt())

    assert spread(contracted) < 0.5 * spread(reference)
    assert abs(spread(fixed) - spread(reference)) < 1e-4
    # The mean is untouched, so the field's decision about *where* the cloud
    # goes is preserved; only its allowed contraction changes.
    assert float((fixed.mean(0) - contracted.mean(0)).abs().max()) < 1e-4


def test_variance_match_preserves_direction() -> None:
    """15. R11: it is a scalar rescale, so no sample changes direction"""
    from ..objectives import variance_matched_teacher
    g = torch.Generator().manual_seed(82)
    reference = torch.randn(32, 3, 8, 8, generator=g)
    teacher = torch.randn(32, 3, 8, 8, generator=g) * 0.4
    fixed = variance_matched_teacher(teacher, reference)
    centre = teacher.reshape(32, -1).mean(0, keepdim=True)
    before = teacher.reshape(32, -1) - centre
    after = fixed.reshape(32, -1) - centre
    cosine = torch.nn.functional.cosine_similarity(before, after, dim=1)
    assert float(cosine.min()) > 1.0 - 1e-5, float(cosine.min())


def test_geometry_loss_without_reference_is_unchanged() -> None:
    """16. R11 is opt-in: omitting the reference reproduces the paper form"""
    from ..objectives import geometry_loss
    g = torch.Generator().manual_seed(83)
    output = torch.randn(16, 3, 8, 8, generator=g)
    drift = torch.randn(16, 3, 8, 8, generator=g)
    plain = float(geometry_loss(output, drift, 0.5))
    assert abs(plain - float(geometry_loss(output, drift, 0.5, None))) < 1e-9
    from ..config import ObjectiveConfig
    assert ObjectiveConfig().teacher_variance_match is False


# ---------------------------------------------------------------------------
# R15 / R16 / R18: Phase-5 reforms
# ---------------------------------------------------------------------------


def test_r15_excludes_collapsed_kernels() -> None:
    """17. R15: a dead kernel is inadmissible and says why"""
    from ..diagnostics import MIN_EFFECTIVE_NEIGHBOURS, kernel_admissible
    healthy = kernel_admissible(
        {"collapsed_row_fraction": 0.0, "ess_fraction": 0.5}, 64)
    assert healthy["admissible"] and healthy["reasons"] == []
    collapsed = kernel_admissible(
        {"collapsed_row_fraction": 0.94, "ess_fraction": 0.02}, 64)
    assert not collapsed["admissible"]
    assert any("collapsed_row_fraction" in r for r in collapsed["reasons"])
    # A kernel that is technically alive but sees fewer than the declared
    # number of effective neighbours is also refused.
    starved = kernel_admissible(
        {"collapsed_row_fraction": 0.0,
         "ess_fraction": (MIN_EFFECTIVE_NEIGHBOURS - 0.5) / 64}, 64)
    assert not starved["admissible"]
    assert any("ess_fraction" in r for r in starved["reasons"])
    # NaN ESS (fully collapsed) must not slip through a comparison.
    nan_case = kernel_admissible(
        {"collapsed_row_fraction": 0.0, "ess_fraction": float("nan")}, 64)
    assert not nan_case["admissible"]


def test_r15_reports_health_even_when_excluding() -> None:
    """18. R15: excluded cells carry their numbers, never silent drops"""
    from ..diagnostics import kernel_admissible
    verdict = kernel_admissible(
        {"collapsed_row_fraction": 0.5, "ess_fraction": 0.01,
         "affinity_median": 4e-20}, 64)
    assert verdict["collapsed_row_fraction"] == 0.5
    assert verdict["ess_fraction"] == 0.01
    assert verdict["affinity_median"] == 4e-20
    assert verdict["required_ess_fraction"] > 0


def test_r16_eta_schedule() -> None:
    """19. R16: linear_decay mirrors the free-particle step, constant does not"""
    from ..objectives import scheduled_eta
    assert scheduled_eta(0.5, "constant", 0.0) == 0.5
    assert scheduled_eta(0.5, "constant", 0.9) == 0.5
    assert scheduled_eta(0.5, "constant", None) == 0.5
    # progress None means "no schedule known", so it must not decay.
    assert scheduled_eta(0.5, "linear_decay", None) == 0.5
    assert abs(scheduled_eta(0.5, "linear_decay", 0.0) - 0.5) < 1e-12
    assert abs(scheduled_eta(0.5, "linear_decay", 0.5) - 0.25) < 1e-12
    # Floored positive at the end so the objective stays well defined.
    assert 0.0 < scheduled_eta(0.5, "linear_decay", 1.0) < 1e-3
    for progress in (0.0, 0.3, 0.7, 1.0):
        assert scheduled_eta(0.5, "linear_decay", progress) <= 0.5
    try:
        scheduled_eta(0.5, "cosine", 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted an unknown eta schedule")
    from ..config import ObjectiveConfig
    assert ObjectiveConfig().eta_schedule == "constant"


def test_r18_logs_the_trajectory_contraction() -> None:
    """20. R18: the teacher's dimension ratio is logged during real training

    The two refuted mechanism hypotheses both probed the fixed point, where
    the field is zero by construction.  This reform measures the same thing
    where training actually spends its time.
    """
    from .. import datasets as D
    from ..config import TrainConfig
    from ..train import phase2_arm_by_id, train_arm
    config = TrainConfig(steps=40, batch=32, controller_batch=16,
                         audit_batch=16, latent_dim=8, width=16,
                         eval_samples=32)
    outcome = train_arm(phase2_arm_by_id("B0"), D.checkerboard(), config,
                        seed=9, log_every=4)
    series = outcome.log.series
    for key in ("trajectory_teacher_dimension_ratio",
                "trajectory_output_effective_dimension",
                "trajectory_eta_effective"):
        values = [v for v in series.get(key, []) if np.isfinite(v)]
        assert values, f"{key} was not logged"
    ratios = [v for v in series["trajectory_teacher_dimension_ratio"]
              if np.isfinite(v)]
    assert all(v > 0 for v in ratios)


def test_r16_changes_the_logged_eta() -> None:
    """21. R16: a decaying schedule is visible in the training log"""
    from .. import datasets as D
    from ..config import (
        ArmConfig, FieldConfig, GeometryConfig, MixtureConfig,
        ObjectiveConfig, TrainConfig,
    )
    from ..train import train_arm
    config = TrainConfig(steps=40, batch=32, controller_batch=16,
                         audit_batch=16, latent_dim=8, width=16,
                         eval_samples=32)

    def arm(schedule: str) -> ArmConfig:
        return ArmConfig(
            "t", False, GeometryConfig(family="raw",
                                       base_kernel="smooth_laplace"),
            FieldConfig(direction_mode="paper"), MixtureConfig(),
            ObjectiveConfig(lambda_anchor=0.0, lambda_geometry=1.0,
                            eta_schedule=schedule))

    flat = train_arm(arm("constant"), D.checkerboard(), config, seed=9,
                     log_every=4).log.series["trajectory_eta_effective"]
    decayed = train_arm(arm("linear_decay"), D.checkerboard(), config, seed=9,
                        log_every=4).log.series["trajectory_eta_effective"]
    assert len(set(round(v, 9) for v in flat)) == 1, flat
    assert len(set(round(v, 9) for v in decayed)) > 1, decayed
    assert decayed[-1] < decayed[0]


# ---------------------------------------------------------------------------
# R20 / R21 / R22: root-cause reforms
# ---------------------------------------------------------------------------


def test_r20_dimension_band_is_two_sided() -> None:
    """22. R20: effective dimension is judged with an optimum, not a floor

    Measured on CIFAR-16: ratio .27 scores 7.5, .90 scores 1.8, 1.40 scores
    6.8, 3.35 scores 3.0 with coverage .66.  A one-sided floor would pass the
    last two.
    """
    from ..diagnostics import DIMENSION_BAND, dimension_verdict
    low, high = DIMENSION_BAND
    assert 0 < low < 1 < high, DIMENSION_BAND
    assert dimension_verdict(0.271)["direction"] == "collapsed"
    assert dimension_verdict(0.902)["direction"] == "matched"
    assert dimension_verdict(1.396)["direction"] == "over_dispersed"
    assert dimension_verdict(3.348)["direction"] == "over_dispersed"
    assert dimension_verdict(0.902)["matched"]
    assert not dimension_verdict(1.396)["matched"]
    # The two failures that score alike must be distinguishable.
    assert (dimension_verdict(0.271)["direction"]
            != dimension_verdict(3.348)["direction"])
    unmeasured = dimension_verdict(float("nan"))
    assert not unmeasured["matched"]
    assert unmeasured["direction"] == "unmeasured"


def test_r21_cap_limits_the_realized_output_step() -> None:
    """23. R21: the realized change respects the declared cap"""
    from ..config import ArmConfig, ObjectiveConfig, TrainConfig
    from ..train import phase2_arm_by_id, train_arm
    from .. import datasets as D
    base = phase2_arm_by_id("B0")
    config = TrainConfig(steps=25, batch=32, controller_batch=16,
                         audit_batch=16, latent_dim=8, width=16,
                         eval_samples=32)
    latent = torch.randn(4, 8, generator=torch.Generator().manual_seed(2))

    def run(cap):
        arm = ArmConfig("t", False, base.geometry, base.field, base.mixture,
                        ObjectiveConfig(lambda_anchor=0.0,
                                        lambda_geometry=1.0,
                                        output_step_cap=cap))
        return train_arm(arm, D.checkerboard(), config, seed=4,
                         log_every=100)

    uncapped, capped = run(None), run(0.5)
    with torch.no_grad():
        assert not torch.equal(uncapped.model(latent), capped.model(latent))
    # A cap must reduce, not increase, the distance travelled from init.
    from ..models import OneStepGenerator
    from ..config import derive_seed
    start = OneStepGenerator(8, 3, 16, 16, derive_seed(4, "generator"))
    with torch.no_grad():
        origin = start(latent)
        moved_uncapped = float((uncapped.model(latent) - origin).norm())
        moved_capped = float((capped.model(latent) - origin).norm())
    assert moved_capped < moved_uncapped, (moved_capped, moved_uncapped)
    for bad in (0.0, -1.0):
        try:
            run(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted output_step_cap={bad}")


def test_r22_steps_per_teacher_is_a_declared_axis() -> None:
    """24. R22: one step per teacher is the default, not an assumption"""
    from ..config import TrainConfig
    from ..train import phase2_arm_by_id, train_arm
    from .. import datasets as D
    assert TrainConfig().steps_per_teacher == 1
    arm = phase2_arm_by_id("B0")

    def run(per_teacher):
        config = TrainConfig(steps=20, batch=32, controller_batch=16,
                             audit_batch=16, latent_dim=8, width=16,
                             eval_samples=32,
                             steps_per_teacher=per_teacher)
        return train_arm(arm, D.checkerboard(), config, seed=5,
                         log_every=100)

    one, eight = run(1), run(8)
    # The field is computed once per teacher, so kernel work is unchanged
    # while optimizer updates scale.
    assert one.ledger.optimizer_updates == 20
    assert eight.ledger.optimizer_updates == 160
    assert one.ledger.kernel_pairs == eight.ledger.kernel_pairs
    latent = torch.randn(4, 8, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        assert not torch.equal(one.model(latent), eight.model(latent))


def test_r24_step_eta_is_inert_under_an_adaptive_optimizer() -> None:
    """25. R24: eta cannot act under Adam, and does act under SGD"""
    from ..config import ArmConfig, ObjectiveConfig, TrainConfig
    from ..train import phase2_arm_by_id, train_arm
    from .. import datasets as D
    base = phase2_arm_by_id("B0")
    latent = torch.randn(4, 8, generator=torch.Generator().manual_seed(7))

    def run(eta, optimizer, lr):
        arm = ArmConfig("t", False, base.geometry, base.field, base.mixture,
                        ObjectiveConfig(lambda_anchor=0.0,
                                        lambda_geometry=1.0, step_eta=eta))
        config = TrainConfig(steps=15, batch=32, controller_batch=16,
                             audit_batch=16, latent_dim=8, width=16,
                             eval_samples=32, optimizer=optimizer,
                             learning_rate=lr)
        outcome = train_arm(arm, D.checkerboard(), config, seed=6,
                            log_every=100)
        with torch.no_grad():
            return outcome.model(latent)

    def effect(optimizer, lr=2e-3):
        """Relative change in the model's output from a 100x change in eta."""
        small, large = run(0.05, optimizer, lr), run(5.0, optimizer, lr)
        return float((small - large).norm() / small.norm().clamp_min(1e-12))

    # Adam: the loss gradient is `-2 eta V / n`, a constant rescale, and
    # `lr * m / (sqrt(v) + eps)` is invariant to that up to `eps`.  The
    # invariance is therefore near-exact rather than bit-exact -- the
    # residual below is Adam's epsilon, not a real dependence on eta.
    adam = effect("adam")
    assert adam < 1e-3, adam
    # SGD has no such invariance, so the same change must move the model by
    # orders of magnitude more.
    for optimizer in ("sgd", "sgd_momentum"):
        plain = effect(optimizer)
        assert plain > 0.1, (optimizer, plain)
        assert plain > 1000 * adam, (optimizer, plain, adam)


def test_r25_the_optimizer_is_configurable_and_reported() -> None:
    """26. R25: the real step control is selectable and appears in reports"""
    from ..config import TrainConfig
    from ..models import OneStepGenerator
    from ..train import build_optimizer, optimizer_report
    model = OneStepGenerator(8, 3, 16, 16, seed=1)
    kinds = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD,
             "sgd_momentum": torch.optim.SGD}
    for name, expected in kinds.items():
        optimizer = build_optimizer(model, TrainConfig(optimizer=name))
        assert isinstance(optimizer, expected), name
    assert build_optimizer(
        model, TrainConfig(optimizer="sgd_momentum")
    ).param_groups[0]["momentum"] == 0.9
    for bad in ({"optimizer": "rmsprop"}, {"learning_rate": 0.0}):
        try:
            build_optimizer(model, TrainConfig(**bad))
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    # The report must name the step control and flag eta's inertness.
    report = optimizer_report(TrainConfig(optimizer="adam"))
    assert report["optimizer"] == "adam" and report["step_eta_is_inert"]
    assert optimizer_report(
        TrainConfig(optimizer="sgd"))["step_eta_is_inert"] is False


def test_r26_teacher_corrections_match_the_reference_moment() -> None:
    """27. R26: every correction matches its declared moment, mean fixed"""
    from ..objectives import TEACHER_CORRECTIONS, corrected_teacher
    generator = torch.Generator().manual_seed(11)
    # An anisotropic teacher, so a scalar and a per-direction match differ.
    # The spread range is kept inside the declared ratio cap, so this test
    # measures the match itself; the guard is tested separately below.
    teacher = torch.randn(64, 12, generator=generator) * torch.linspace(
        0.5, 3.0, 12)
    reference = torch.randn(96, 12, generator=generator) * 1.5
    target = (reference - reference.mean(0)).pow(2).mean().sqrt()

    for mode in TEACHER_CORRECTIONS:
        report: dict = {}
        out = corrected_teacher(teacher, reference, mode=mode, report=report)
        assert out.shape == teacher.shape, mode
        if mode == "none":
            assert torch.equal(out, teacher)
            continue
        # No correction may move the cloud's centre.
        assert torch.allclose(out.mean(0), teacher.mean(0), atol=1e-5), mode
        spread = (out - out.mean(0)).pow(2).mean().sqrt()
        assert abs(float(spread) - float(target)) < 0.15 * float(target), mode
        assert report["correction_mode"] == mode
        assert 0.0 <= report["correction_ratio_cap_fraction"] <= 1.0

    # Per-coordinate matches each coordinate, which one scalar cannot.
    per_coordinate = corrected_teacher(teacher, reference,
                                       mode="per_coordinate")
    errors = ((per_coordinate - per_coordinate.mean(0)).pow(2).mean(0).sqrt()
              / (reference - reference.mean(0)).pow(2).mean(0).sqrt())
    assert float((errors - 1).abs().max()) < 0.05, float(errors.max())
    scalar = corrected_teacher(teacher, reference, mode="scalar")
    assert not torch.allclose(scalar, per_coordinate)

    # The gain multiplies the matched spread, exactly.
    gained = corrected_teacher(teacher, reference, mode="scalar", gain=1.2)
    plain_spread = (scalar - scalar.mean(0)).pow(2).mean().sqrt()
    gain_spread = (gained - gained.mean(0)).pow(2).mean().sqrt()
    assert abs(float(gain_spread / plain_spread) - 1.2) < 1e-4

    for bad in ({"gain": 0.0}, {"ratio_cap": 0.5}, {"mode": "whiten"}):
        try:
            corrected_teacher(teacher, reference, **bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_r26_the_ratio_cap_binds_and_is_reported() -> None:
    """29. R26: the per-direction guard is measured, not assumed harmless"""
    from ..objectives import corrected_teacher
    generator = torch.Generator().manual_seed(13)
    # Coordinate 0 needs a factor of 30 to match, which the cap forbids.
    teacher = torch.randn(64, 4, generator=generator) * torch.tensor(
        [0.05, 1.0, 1.0, 1.0])
    reference = torch.randn(96, 4, generator=generator) * 1.5
    report: dict = {}
    out = corrected_teacher(teacher, reference, mode="per_coordinate",
                            ratio_cap=10.0, report=report)
    assert report["correction_ratio_cap_fraction"] == 0.25, report
    # The capped coordinate is expanded as far as allowed and no further.
    spread = (out - out.mean(0)).pow(2).mean(0).sqrt()
    before = (teacher - teacher.mean(0)).pow(2).mean(0).sqrt()
    assert abs(float(spread[0] / before[0]) - 10.0) < 1e-4, float(spread[0])
    # A cap wide enough to be inactive must be reported as never binding.
    corrected_teacher(teacher, reference, mode="per_coordinate",
                      ratio_cap=100.0, report=report)
    assert report["correction_ratio_cap_fraction"] == 0.0, report


def test_r26_eigendirection_match_survives_rank_deficiency() -> None:
    """28. R26: the per-direction match is well posed when n << d"""
    from ..objectives import corrected_teacher
    generator = torch.Generator().manual_seed(12)
    # 16 samples in 200 dimensions: the covariance is rank <= 15, so any
    # formulation that inverted it would be ill posed here.
    teacher = torch.randn(16, 200, generator=generator) * 0.2
    reference = torch.randn(64, 200, generator=generator) * 2.0
    report: dict = {}
    out = corrected_teacher(teacher, reference, mode="eigendirection",
                            report=report)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.mean(0), teacher.mean(0), atol=1e-5)
    # It must expand a shrunken teacher toward the reference, not collapse it.
    before = float((teacher - teacher.mean(0)).pow(2).mean().sqrt())
    after = float((out - out.mean(0)).pow(2).mean().sqrt())
    assert after > before, (before, after)
    assert 0.0 <= report["correction_ratio_cap_fraction"] <= 1.0
    # A teacher with no spread at all must not produce NaNs.
    flat = torch.zeros(8, 20)
    assert torch.isfinite(
        corrected_teacher(flat, reference[:, :20], mode="eigendirection")
    ).all()


def test_r27_field_cloud_is_a_declared_size() -> None:
    """31. R27: the field's cloud is declared, not tied to the target batch"""
    from ..config import TrainConfig
    from ..train import phase2_arm_by_id, train_arm
    from .. import datasets as D
    assert TrainConfig().field_cloud is None      # default = old behaviour
    arm = phase2_arm_by_id("B0")

    def run(cloud):
        config = TrainConfig(steps=12, batch=32, field_cloud=cloud,
                             controller_batch=16, audit_batch=16,
                             latent_dim=8, width=16, eval_samples=32)
        return train_arm(arm, D.checkerboard(), config, seed=8,
                         log_every=100)

    default, tied, larger = run(None), run(32), run(128)
    # None must reproduce `batch` exactly, not merely closely.
    latent = torch.randn(4, 8, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        assert torch.equal(default.model(latent), tied.model(latent))
        assert not torch.equal(default.model(latent), larger.model(latent))
    # A bigger cloud costs more generator work but the SAME target data:
    # the axis must not smuggle in extra target examples.
    assert larger.ledger.generator_examples > default.ledger.generator_examples
    assert (larger.ledger.target_examples_field
            == default.ledger.target_examples_field)
    assert larger.ledger.kernel_pairs > default.ledger.kernel_pairs
    for bad in (0, -8):
        try:
            run(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted field_cloud={bad}")


def test_balancing_depth_zero_is_the_audited_paper_field() -> None:
    """30. the depth axis is anchored: depth 0 IS `drift_paper`"""
    import sys
    from pathlib import Path
    import numpy as np
    root = Path(__file__).resolve().parents[3] / "numerics"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import lowdim_drift as LD                              # noqa: PLC0415
    from ..diagnose_phase7 import drift_balanced
    rng = np.random.default_rng(0)
    # The self-mask is the case that matters: unmasked, every row's maximum
    # is the zero self-distance, so a single row-wise shift happens to be
    # correct for the column softmax too and a wrong implementation still
    # agrees.  Masked, it does not -- that is where the check has teeth.
    for mask in (True, False):
        for tau in (0.05, 0.2, 1.0):
            for dim in (4, 32):
                q = rng.normal(size=(40, dim))
                data = rng.normal(size=(24, dim))
                paper = LD.drift_paper(q, data, tau, mask)
                ours = drift_balanced(q, data, tau, mask, 0)
                scale = max(float(np.abs(paper).max()), 1e-30)
                error = float(np.abs(paper - ours).max()) / scale
                assert error < 1e-9, (mask, tau, dim, error)

    # Balancing must converge, and must not reproduce depth 0.
    q = rng.normal(size=(64, 8))
    data = rng.normal(size=(64, 8))
    fields = {k: drift_balanced(q, data, 0.2, True, k)
              for k in (0, 1, 16, 64)}
    assert not np.allclose(fields[0], fields[16])
    deep = np.abs(fields[64] - fields[16]).max()
    shallow = np.abs(fields[16] - fields[1]).max()
    assert deep < shallow, (deep, shallow)
    try:
        drift_balanced(q, data, 0.2, True, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a negative balancing depth")


if __name__ == "__main__":
    main("diagnosis reforms (R1, R4, R8, R9, R11, R15, R16, R18, R20-R26)",
         dict(globals()))
