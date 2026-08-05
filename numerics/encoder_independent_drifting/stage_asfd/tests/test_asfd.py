"""ASFD regression suite."""

from __future__ import annotations

from pathlib import Path

import torch

from ...stage_cap.config import FEATURE_LEVELS, CAPModelConfig
from ...stage_cap.model import CAPPixelTransformer
from ..artifacts import (
    _DEPENDENCIES,
    assert_no_inherited_freeze,
    manifest_difference,
    source_manifest,
)
from ..calibration import (
    calibrate_bandwidth,
    calibrate_coefficient,
    calibrate_level_bandwidths,
    calibrate_normalization,
    taus_from_records,
)
from ..config import ARMS, LEVEL_NAMES, asfd_config, smoke_config
from ..features import (
    descriptors,
    encode,
    freeze_trunk,
    input_jacobian_is_alive,
    noise_images,
    to_locations,
)
from ..field import (
    descriptor_energy,
    field_energy,
    laplace_field,
    multi_radius_energy,
)
from ..gradients import (
    AbortMonitor,
    capped_weight,
    combine,
    gradient_cosine,
    gradient_norm,
    snapshot,
)
from ..qualification import gate_inter_level, linear_cka


def _trunk(seed: int = 5, width: int = 64) -> CAPPixelTransformer:
    # Depth 12: the declared taps reach block 5 of each stack, so a shallower
    # trunk would not have them and the audit would test a different
    # architecture from the one that ships.
    config = CAPModelConfig(
        image_size=32, patch_size=2, width=width, depth=12, heads=4,
        mlp_ratio=2.0, time_embedding_dim=32, condition_dim=32,
        refiner_width=8, refiner_depth=1,
    )
    return freeze_trunk(CAPPixelTransformer(config, seed))


def test_an_unreachable_feature_tap_is_refused_at_config_time():
    """A tap past the end of its stack is silently dropped by the forward.

    The trunk would work, the level would simply never appear, and the caller
    would meet it as a missing key far downstream. Catch it where the shape is
    declared.
    """
    try:
        CAPModelConfig(depth=4).validate()
    except ValueError as error:
        assert "blocks per stack" in str(error)
    else:
        raise AssertionError("a depth-4 trunk accepted taps at block 5")


def _wake(model: CAPPixelTransformer, seed: int = 31) -> CAPPixelTransformer:
    """A fresh trunk is the zero function; a feature audit on it proves nothing."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for block in list(model.encoder) + list(model.decoder):
            block.modulation.weight.normal_(0.0, 0.05, generator=generator)
            block.modulation.bias.normal_(0.0, 0.05, generator=generator)
    return model


# ---------------------------------------------------------------- configuration


def test_config_validates_and_spans_a_real_radius_range():
    config = asfd_config()
    radii = config.field_config.radii
    assert max(radii) / min(radii) >= 4.0, radii
    assert config.field_config.positives > config.field_config.negatives
    assert config.gradients.cadence == 10


def test_a_narrow_radius_set_is_refused():
    """{0.35, 0.60, 0.85} spans under 2x and has no local regime."""
    from ..config import FieldConfig

    try:
        FieldConfig(radii=(0.35, 0.60, 0.85)).validate()
    except ValueError as error:
        assert "local-to-global" in str(error)
    else:
        raise AssertionError("a narrow radius set was accepted")


def test_caps_are_per_component_not_shared():
    config = asfd_config().gradients
    total = config.cap_b1 + config.cap_raw + config.cap_self
    assert total > max(config.cap_b1, config.cap_raw, config.cap_self)
    assert abs(total - 0.35) < 1e-12


def test_levels_come_from_the_frozen_trunk():
    """Declared by the trunk, so the taps are covered by its source hash."""
    assert LEVEL_NAMES == tuple(name for name, _, _ in FEATURE_LEVELS)
    assert len(ARMS) == 3


# --------------------------------------------------------------------- features


def test_descriptor_shape_is_66_vectors():
    tokens = torch.randn(3, 256, 64)
    out = descriptors(tokens, pool=2)
    assert out.shape == (3, 66, 64)


def test_descriptor_globals_are_permutation_invariant():
    """The last two vectors survive a token shuffle; the first 64 do not."""
    tokens = torch.randn(2, 256, 32)
    order = torch.randperm(256)
    base = descriptors(tokens, 2)
    shuffled = descriptors(tokens[:, order], 2)
    assert torch.allclose(base[:, -2:], shuffled[:, -2:], atol=1e-5)
    assert not torch.allclose(base[:, :-2], shuffled[:, :-2], atol=1e-3)


def test_frozen_trunk_passes_gradient_to_its_input():
    """The single most expensive mistake available: a no_grad frozen forward.

    Everything would still run, the loss would still fall, and nothing would be
    learned through the semantic branch.
    """
    trunk = _wake(_trunk())
    report = input_jacobian_is_alive(trunk, smoke_config().features, t_f=0.1)
    assert report["input_gradient_alive"], report
    assert report["frozen_parameters_received_no_gradient"]


def test_frozen_parameters_do_not_require_grad():
    trunk = _trunk()
    assert all(not p.requires_grad for p in trunk.parameters())


def test_noising_uses_the_native_corruption_path():
    images = torch.zeros(4, 3, 32, 32)
    generator = torch.Generator().manual_seed(3)
    noised = noise_images(images, 0.25, generator)
    # With x = 0 the result is t_f * xi, so its scale identifies t_f.
    assert abs(float(noised.std()) - 0.25) < 0.03


def test_roles_must_not_share_a_noise_draw():
    """Two roles built from one generator must differ, or the barycenters
    correlate and the field measures nothing."""
    images = torch.randn(8, 3, 32, 32)
    generator = torch.Generator().manual_seed(7)
    first = noise_images(images, 0.1, generator)
    second = noise_images(images, 0.1, generator)
    assert not torch.allclose(first, second)


# ------------------------------------------------------------------------ field


def test_field_is_zero_when_the_clouds_coincide():
    values = torch.randn(6, 12, 8)
    field, _ = laplace_field(values, values, values, tau=1.0, diagnostics=False)
    assert float(field.abs().max()) < 1e-6


def test_field_detects_a_shifted_negative_cloud():
    probes = torch.randn(4, 10, 8)
    positives = torch.randn(4, 16, 8)
    field, _ = laplace_field(probes, positives, positives + 3.0, 1.0, diagnostics=False)
    assert float(field.abs().max()) > 0.5


def test_energy_never_squares_the_mean_field():
    """Two opposite fields must not cancel: the counterexample.

    Averaging the fields and squaring the sum would give zero here; averaging
    separately squared energies cannot.
    """
    probes = torch.randn(2, 8, 4)
    positives = probes + 1.0
    negatives = probes - 1.0
    up, _ = field_energy(probes, positives, negatives, 1.0, diagnostics=False)
    down, _ = field_energy(probes, negatives, positives, 1.0, diagnostics=False)
    assert float(up) > 1e-3 and float(down) > 1e-3
    averaged_energy = (float(up) + float(down)) / 2
    field_up, _ = laplace_field(probes, positives, negatives, 1.0, diagnostics=False)
    field_down, _ = laplace_field(probes, negatives, positives, 1.0, diagnostics=False)
    squared_mean = float(((field_up + field_down) / 2).square().sum(-1).mean())
    assert squared_mean < 1e-8, "the aggregate-then-square path did not cancel"
    assert averaged_energy > 1e-3, "separate squares must not cancel"


def test_multi_radius_averages_separately_squared_energies():
    probes = torch.randn(3, 8, 5)
    positives = torch.randn(3, 12, 5)
    negatives = torch.randn(3, 12, 5) + 0.5
    taus = {0.1: 0.5, 0.5: 1.0, 0.9: 2.0}
    total, stats = multi_radius_energy(probes, positives, negatives, taus)
    manual = sum(
        float(field_energy(probes, positives, negatives, tau, diagnostics=False)[0])
        for tau in taus.values()
    ) / len(taus)
    # Relative, not absolute: the implementation accumulates in float32 while
    # the reference sums in Python float64.
    assert abs(float(total) - manual) / manual < 1e-5
    assert stats["radii"] == sorted(taus)


def test_field_reports_both_sides_of_the_weight_health():
    probes = torch.randn(2, 8, 4)
    _, stats = laplace_field(probes, torch.randn(2, 16, 4), torch.randn(2, 16, 4), 1.0)
    for side in ("positive", "negative"):
        assert "ess_fraction_median" in stats[side]
        assert "maximum_weight_p95" in stats[side]


def test_descriptor_energy_reports_the_location_split():
    levels = {"a": torch.randn(66, 8, 6), "b": torch.randn(66, 8, 6)}
    positives = {name: torch.randn(66, 12, 6) for name in levels}
    negatives = {name: torch.randn(66, 12, 6) + 0.3 for name in levels}
    taus = {name: {0.35: 1.0} for name in levels}
    energy, stats = descriptor_energy(levels, positives, negatives, taus)
    assert float(energy) > 0
    split = stats["location_split"]
    assert abs(split["local_share"] + split["global_share"] - 1.0) < 1e-9
    assert split["local_vectors"] == 64


def test_negatives_keep_their_graph():
    probes = torch.randn(2, 6, 4)
    positives = torch.randn(2, 8, 4)
    negatives = torch.randn(2, 8, 4, requires_grad=True)
    energy, _ = field_energy(probes, positives, negatives, 1.0, diagnostics=False)
    energy.backward()
    assert negatives.grad is not None
    assert float(negatives.grad.abs().max()) > 0


# ------------------------------------------------------------------ calibration


def test_bandwidth_bisection_hits_its_ess_target():
    values = torch.randn(64, 12)
    config = smoke_config().field_config
    for target in (0.1, 0.35, 0.85):
        record = calibrate_bandwidth(values, target, config)
        assert abs(record["achieved_ess_median"] - target) < 0.05, record


def test_ess_excludes_the_self_match():
    """A self-match would let a nominal target be met through a zero distance
    while the realized off-diagonal neighbourhood is entirely different."""
    values = torch.randn(48, 8)
    record = calibrate_bandwidth(values, 0.5, smoke_config().field_config)
    assert 0.0 < record["achieved_ess_median"] < 1.0
    assert record["achieved_ess_p05"] <= record["achieved_ess_median"]


def test_normalization_fires_per_channel_only_above_the_trigger():
    features = smoke_config().features
    balanced = {"a": torch.randn(16, 66, 12)}
    calibrated = calibrate_normalization(balanced, features)
    assert not calibrated["a"].per_channel_applied

    dominated = torch.randn(16, 66, 12) * 0.01
    dominated[..., 0] = torch.randn(16, 66) * 50.0
    calibrated = calibrate_normalization({"a": dominated}, features)
    assert calibrated["a"].per_channel_applied
    assert calibrated["a"].pc1_share > features.per_channel_pc1_trigger


def test_normalization_scale_is_frozen_and_positive():
    calibrated = calibrate_normalization(
        {"a": torch.randn(12, 66, 8)}, smoke_config().features
    )
    assert calibrated["a"].level_scale > 0
    scaled = calibrated["a"].apply(torch.randn(4, 66, 8))
    assert torch.isfinite(scaled).all()


def test_level_bandwidths_record_the_ladder():
    values = torch.randn(128, 10)
    records = calibrate_level_bandwidths(values, smoke_config().field_config)
    assert set(records) == set(smoke_config().field_config.radii)
    for record in records.values():
        assert "ladder_attempts" in record
        assert record["tau"] > 0
    taus = taus_from_records({"a": records})
    assert set(taus["a"]) == set(records)
    # tau must increase with the radius: a broader neighbourhood is a wider
    # kernel. If this inverted, the radius labels would be meaningless.
    ordered = [records[r]["tau"] for r in sorted(records)]
    assert ordered == sorted(ordered), ordered


def test_the_tail_floor_scales_with_the_radius():
    """An absolute floor would reject every local radius.

    At a median ESS target of 0.10 an absolute 5th-percentile floor of 0.10
    demands the tail reach the median, which no distribution satisfies -- so
    the whole local end of the set would be silently unreachable.
    """
    from ..calibration import bandwidth_is_healthy
    from ..config import FieldConfig

    config = FieldConfig()
    local = {"target_ess": 0.10, "achieved_ess_p05": 0.03, "maximum_weight_p95": 0.4}
    broad = {"target_ess": 0.85, "achieved_ess_p05": 0.03, "maximum_weight_p95": 0.4}
    assert bandwidth_is_healthy(local, config), "a local radius was rejected"
    assert not bandwidth_is_healthy(broad, config), "a broad radius was excused"


def test_a_small_cloud_cannot_support_a_local_radius():
    """The health floors and the radius set are coupled, and that is by design.

    At an ESS fraction of 0.10 the effective neighbourhood is 0.1n, so a small
    cloud puts almost all the weight on one neighbour. The max-weight ceiling
    refuses it and the ladder reports every rung it tried rather than silently
    widening -- which is the failure mode that would quietly turn a
    "multi-radius" set back into three broad fields.
    """
    from ..config import FieldConfig

    cramped = FieldConfig(positives=16, probes=8, negatives=8, ess_samples=16,
                          ess_iterations=8)
    try:
        calibrate_level_bandwidths(torch.randn(16, 8), cramped)
    except ValueError as error:
        message = str(error)
        assert "ladder" in message
        # Every rung must be reported, not just the failing one.
        for rung in cramped.radius_ladder:
            assert f"'radius': {rung}" in message, message
    else:
        raise AssertionError("a 16-sample cloud admitted a local radius")


def test_coefficient_calibration_lands_on_the_cap():
    weight = calibrate_coefficient(primary_norm=2.0, component_norm=8.0, cap=0.1)
    assert abs(weight * 8.0 / 2.0 - 0.1) < 1e-12


# -------------------------------------------------------------------- gradients


def test_capped_weight_leaves_a_small_component_untouched():
    assert capped_weight(1.0, 0.05, 0.10) == 1.0
    assert abs(capped_weight(1.0, 0.5, 0.10) - 0.2) < 1e-12


def test_combine_is_the_plain_sum_and_never_projects():
    model = torch.nn.Linear(4, 3)
    parameters = list(model.parameters())
    primary = [torch.ones_like(p) for p in parameters]
    # Deliberately opposed: a projection would delete this, a sum keeps it.
    opposed = [-0.05 * torch.ones_like(p) for p in parameters]
    outcome = combine(
        parameters,
        primary,
        {"raw": (opposed, 0.10)},
        asfd_config().gradients,
    )
    assert outcome.cosines["primary_vs_raw"] < -0.99
    for parameter in parameters:
        assert torch.allclose(
            parameter.grad, torch.full_like(parameter.grad, 0.95), atol=1e-6
        )


def test_combine_respects_each_cap_independently():
    model = torch.nn.Linear(6, 6)
    parameters = list(model.parameters())
    primary = [torch.ones_like(p) for p in parameters]
    big = [10.0 * torch.ones_like(p) for p in parameters]
    config = asfd_config().gradients
    outcome = combine(
        parameters,
        primary,
        {"raw": (big, config.cap_raw), "self": (big, config.cap_self)},
        config,
    )
    assert abs(outcome.post_cap_ratio["raw"] - config.cap_raw) < 1e-9
    assert abs(outcome.post_cap_ratio["self"] - config.cap_self) < 1e-9
    # The total is allowed to exceed a single cap: that is the joint treatment.
    assert outcome.total_auxiliary_ratio > config.cap_raw


def test_abort_fires_on_sub_floor_energy():
    monitor = AbortMonitor(asfd_config().gradients)
    monitor.observe_energy_floor(13.933, 14.103)
    assert monitor.should_abort
    assert "estimator exploitation" in monitor.reasons[0]


def test_abort_fires_on_sustained_rank_collapse_not_a_single_dip():
    monitor = AbortMonitor(asfd_config().gradients)
    monitor.observe_rank("feature", 0.5)
    assert not monitor.should_abort, "a single dip must not abort"
    monitor.observe_rank("feature", 0.5)
    assert monitor.should_abort


def test_abort_tolerates_mild_opposition_but_not_negation():
    config = asfd_config().gradients
    mild = AbortMonitor(config)
    for _ in range(config.anti_parallel_window):
        mild.observe_cosines({"primary_vs_raw": -0.3})
    assert not mild.should_abort, "mild opposition is the working regime"

    negating = AbortMonitor(config)
    for _ in range(config.anti_parallel_window):
        negating.observe_cosines({"primary_vs_raw": -0.95})
    assert negating.should_abort


def test_abort_fires_on_degenerate_negative_ess():
    monitor = AbortMonitor(asfd_config().gradients)
    ceiling = asfd_config().field_config.negative_ess_ceiling
    monitor.observe_negative_ess("enc_mid", 0.95, ceiling)
    assert monitor.should_abort


def test_gradient_helpers_agree_with_torch():
    a = [torch.randn(4, 4), torch.randn(3)]
    b = [x.clone() for x in a]
    assert abs(gradient_cosine(a, b) - 1.0) < 1e-9
    flat = torch.cat([x.flatten() for x in a])
    assert abs(gradient_norm(a) - float(flat.norm())) < 1e-6


def test_snapshot_captures_and_detaches():
    model = torch.nn.Linear(3, 2)
    model(torch.randn(2, 3)).sum().backward()
    captured = snapshot(model)
    assert all(g is None or not g.requires_grad for g in captured)


# ---------------------------------------------------------------- qualification


def test_cka_is_one_for_identical_and_low_for_independent():
    left = torch.randn(64, 20)
    assert abs(linear_cka(left, left) - 1.0) < 1e-9
    assert linear_cka(left, torch.randn(64, 20)) < 0.6


def test_inter_level_gate_rejects_duplicate_levels():
    values = torch.randn(32, 66, 8)
    config = asfd_config().qualification
    name, ok, _, _ = gate_inter_level({"a": values, "b": values.clone()}, config)
    assert not ok, "identical levels must fail G8"
    name, ok, _, _ = gate_inter_level(
        {"a": values, "b": torch.randn(32, 66, 8)}, config
    )
    assert ok


# ------------------------------------------------------------------- artifacts


def test_manifest_is_explicit_and_excludes_tests():
    manifest = source_manifest()
    assert len(manifest) == len(_DEPENDENCIES)
    assert not any("tests/" in name for name in manifest)


def test_a_new_module_beside_the_stage_does_not_invalidate_the_manifest():
    """The B2.5 lesson, implemented rather than described."""
    manifest = source_manifest()
    here = Path(__file__).resolve().parents[1]
    scratch = here / "_manifest_probe.py"
    scratch.write_text("# transient probe\n", encoding="utf-8")
    try:
        assert manifest_difference(manifest) == {
            "added": [], "removed": [], "changed": []
        }
    finally:
        scratch.unlink()


def test_no_inherited_freeze_is_reachable():
    assert_no_inherited_freeze()


def test_frozen_trunk_dependencies_are_hashed():
    """A different trunk is a different feature geometry."""
    manifest = source_manifest()
    assert any(name.endswith("stage_cap/model.py") for name in manifest)
    assert any(name.endswith("stage_cap/config.py") for name in manifest)


def _run_all() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    print(f"=== stage_asfd ({len(tests)} tests) ===")
    failures = 0
    for name, function in tests:
        try:
            function()
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            failures += 1
            print(f"  [FAIL] {name}: {type(error).__name__}: {error}")
        else:
            print(f"  [PASS] {name}")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
