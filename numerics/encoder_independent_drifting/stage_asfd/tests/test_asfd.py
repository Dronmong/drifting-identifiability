"""ASFD regression suite."""

from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path

import torch

from ...stage_cap.config import FEATURE_LEVELS, CAPGateConfig, CAPModelConfig
from ...stage_cap.model import CAPPixelTransformer
from .. import continuation as asfd_continuation
from .. import feature_bank as asfd_feature_bank
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
from ..continuation import (
    _asfd_wall_policy,
    _require_declared_storage_root,
    _require_exact_preflight_runtime,
    _require_launch_authorization,
    _revalidate_wall_stop,
)
from ..feature_bank import (
    VIEWS_PER_ROLE,
    dataset_binding,
    stratified_partitions,
    verify_dataset_binding,
)
from ..features import (
    descriptors,
    freeze_trunk,
    input_jacobian_is_alive,
    noise_images,
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
from ..preflight import _require_foundation_runtime_and_rate, continuation_profile
from ..qualification import gate_inter_level, linear_cka


def _trunk(seed: int = 5, width: int = 64) -> CAPPixelTransformer:
    # Depth 12: the declared taps reach block 5 of each stack, so a shallower
    # trunk would not have them and the audit would test a different
    # architecture from the one that ships.
    config = CAPModelConfig(
        image_size=32,
        patch_size=2,
        width=width,
        depth=12,
        heads=4,
        mlp_ratio=2.0,
        time_embedding_dim=32,
        condition_dim=32,
        refiner_width=8,
        refiner_depth=1,
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


def test_feature_bank_populations_are_balanced_disjoint_and_four_view() -> None:
    labels = torch.arange(10).repeat_interleave(1_100)
    excluded = {0, 1, 1_100, 1_101}
    train, fresh = stratified_partitions(labels, excluded)
    assert len(train) == len(fresh) == 5_000
    assert not (set(train.tolist()) & set(fresh.tolist()))
    assert not (set(train.tolist()) | set(fresh.tolist())) & excluded
    assert torch.bincount(labels[train], minlength=10).tolist() == [500] * 10
    assert torch.bincount(labels[fresh], minlength=10).tolist() == [500] * 10
    assert VIEWS_PER_ROLE == asfd_config().features.views_per_image == 4


def test_feature_bank_dataset_binding_rejects_different_raw_population(
    monkeypatch,
) -> None:
    # Meta tensors exercise the exact production schema without allocating a
    # 600 MiB CIFAR float tensor; hashing is isolated from this identity test.
    pool = torch.empty((50_000, 3, 32, 32), dtype=torch.float32, device="meta")
    labels = torch.empty((50_000,), dtype=torch.int64, device="meta")
    monkeypatch.setattr(
        asfd_feature_bank,
        "tensor_content_sha256",
        lambda value: f"{value.dtype}:{tuple(value.shape)}",
    )
    recorded = dataset_binding(pool, labels)
    assert verify_dataset_binding(recorded, pool, labels) == recorded
    changed = dict(recorded)
    changed["train_tensor_sha256"] = "different"
    try:
        verify_dataset_binding(changed, pool, labels)
    except RuntimeError as error:
        assert "different CIFAR-10 population" in str(error)
    else:
        raise AssertionError("changed raw CIFAR content was accepted by a feature bank")


def test_continuation_profile_preserves_the_foundation_and_extends_only_horizon():
    calibration = {
        "status": "cap-emf2-gate-calibration",
        "gate": asdict(CAPGateConfig()),
    }
    preflight = {
        "candidate": "local_1000_d0002_fp32",
        "inputs": {"gate_calibration": calibration},
    }
    continuation = continuation_profile(preflight, 800_000)
    assert continuation.train.updates == 800_000
    assert continuation.train.checkpoint_updates[-2:] == (750_000, 800_000)
    assert continuation.train.snapshot_every == 10_000
    assert continuation.objective.sampler_mode == "ordered_uniform"


def test_paid_continuation_requires_all_three_operator_confirmations() -> None:
    _require_launch_authorization(
        paid=True, durable_mirror=True, durable_workspace=True
    )
    cases = (
        {"paid": False, "durable_mirror": True, "durable_workspace": True},
        {"paid": True, "durable_mirror": False, "durable_workspace": True},
        {"paid": True, "durable_mirror": True, "durable_workspace": False},
    )
    for values in cases:
        try:
            _require_launch_authorization(**values)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"ASFD accepted incomplete authorization {values}")


def test_continuation_runtime_must_exactly_match_measured_preflight() -> None:
    runtime = {
        "device": "cuda",
        "allow_tf32": False,
        "torch_version": "2.7.1+cu126",
        "cuda_version": "12.6",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "gpu_memory_gib": 39.38,
        "capability": "sm_80",
    }
    assert _require_exact_preflight_runtime({"device": runtime}, runtime) == runtime
    changed = dict(runtime, gpu_name="NVIDIA L40S")
    try:
        _require_exact_preflight_runtime({"device": runtime}, changed)
    except RuntimeError as error:
        assert "gpu_name" in str(error)
    else:
        raise AssertionError("ASFD accepted a GPU different from its preflight")


def test_asfd_preflight_uses_foundation_hardware_and_provider_rate() -> None:
    foundation_runtime = {
        "device": "cuda",
        "allow_tf32": True,
        "torch_version": "2.7.1+cu126",
        "cuda_version": "12.6",
        "gpu_name": "NVIDIA A100-SXM4-40GB",
        "gpu_memory_gib": 39.38,
        "capability": "sm_80",
    }
    live = dict(foundation_runtime, allow_tf32=False)
    cap2 = {
        "inputs": {
            "benchmark": {
                "device": foundation_runtime,
                "hourly_rate": 1.75,
            }
        },
        "budget": {"hourly_rate": 1.75},
    }
    binding = _require_foundation_runtime_and_rate(cap2, live)
    assert binding["hourly_rate"] == 1.75
    assert binding["foundation_allow_tf32"] is True
    assert binding["asfd_allow_tf32"] is False

    changed_gpu = dict(live, gpu_name="NVIDIA L40S")
    try:
        _require_foundation_runtime_and_rate(cap2, changed_gpu)
    except RuntimeError as error:
        assert "gpu_name" in str(error)
    else:
        raise AssertionError("ASFD smoke accepted hardware unlike its foundation")

    changed_rate = dict(cap2, budget={"hourly_rate": 2.0})
    try:
        _require_foundation_runtime_and_rate(changed_rate, live)
    except RuntimeError as error:
        assert "provider rate" in str(error)
    else:
        raise AssertionError("ASFD smoke accepted a mismatched provider rate")


def test_asfd_wall_policy_is_the_measured_50k_projection_plus_15_percent() -> None:
    policy = _asfd_wall_policy(
        {"measured_smoke": {"updates": 500, "wall_seconds": 100.0}},
        recovery_every=5_000,
    )
    assert policy["continuation_updates"] == 50_000
    assert policy["recovery_interval_updates"] == 5_000
    assert abs(policy["hard_cumulative_continuation_wall_seconds"] - 11_500.0) < 1e-9
    assert abs(policy["projected_maximum_detection_overshoot_seconds"] - 1_000.0) < 1e-9


def test_asfd_storage_root_must_be_the_preflighted_volume(tmp_path: Path) -> None:
    root = tmp_path / "volume"
    other = tmp_path / "other"
    plan = {"storage_root": str(root.resolve())}
    assert _require_declared_storage_root(plan, root) == root.resolve()
    try:
        _require_declared_storage_root(plan, other)
    except RuntimeError as error:
        assert "differs from preflight" in str(error)
    else:
        raise AssertionError("ASFD accepted an unmeasured storage volume")


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

    cramped = FieldConfig(
        positives=16, probes=8, negatives=8, ess_samples=16, ess_iterations=8
    )
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


def test_abort_reasons_are_deduplicated():
    """A sustained condition fires on every event; the artifact must not fill
    with tens of thousands of copies of one sentence."""
    monitor = AbortMonitor(asfd_config().gradients)
    for _ in range(50):
        monitor.observe_energy_floor(13.933, 14.103)
        monitor.observe_rank("feature", 0.4)
    assert len(monitor.reasons) == 2, monitor.reasons


def test_abort_monitor_recovery_roundtrip_preserves_windows_and_reasons():
    config = asfd_config().gradients
    first = AbortMonitor(config)
    for _ in range(7):
        first.observe_cosines({"primary_vs_raw": -0.25})
    first.observe_rank("raw", 0.5)
    payload = first.state_dict()
    resumed = AbortMonitor(config)
    resumed.load_state_dict(payload)
    assert resumed.state_dict() == payload
    resumed.observe_rank("raw", 0.5)
    assert resumed.should_abort


def test_no_source_file_lost_its_encoding():
    """A scripted rewrite once mangled this package's UTF-8."""
    from .. import artifacts as asfd_artifacts

    for name in asfd_artifacts._DEPENDENCIES:
        if not name.startswith("stage_asfd/"):
            continue
        text = (asfd_artifacts.PACKAGE / name).read_text(encoding="utf-8")
        assert "Ã" not in text, f"{name} shows mojibake"


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


def test_inverse_haar_actually_inverts():
    """A transposed synthesis matrix silently applies the forward transform.

    Measured before the fix: round-trip error 4.31 instead of 5e-7.
    """
    from ...stage_cap.diagnostics import haar_transform
    from ..qualification import _inverse_haar

    x = torch.randn(3, 1, 8, 8)
    recovered = _inverse_haar(haar_transform(x))
    assert float((recovered - x).abs().max()) < 1e-5


def test_inverse_haar_handles_colour_images():
    """G7 perturbs real three-channel images, not single planes.

    Both round-trip tests used C = 1, where flattening a sample and flattening
    a plane coincide. On CIFAR they do not: the synthesis matrix is
    [size*size, size*size] and a sample carries C planes, so the multiply is a
    shape error. G7 is the gate that decides whether this arm proceeds at all
    and it had never been run on a colour image.
    """
    from ...stage_cap.diagnostics import haar_transform
    from ..qualification import _inverse_haar

    x = torch.randn(4, 3, 8, 8)
    recovered = _inverse_haar(haar_transform(x))
    assert recovered.shape == x.shape
    assert float((recovered - x).abs().max()) < 1e-5


def test_inverse_haar_does_not_mix_channels():
    """Each colour plane must synthesise from its own coefficients."""
    from ...stage_cap.diagnostics import haar_transform
    from ..qualification import _inverse_haar

    x = torch.randn(2, 3, 8, 8)
    coefficients = haar_transform(x)
    isolated = torch.zeros_like(coefficients)
    isolated[:, 1] = coefficients[:, 1]
    synthesised = _inverse_haar(isolated)
    assert float(synthesised[:, 0].abs().max()) == 0.0
    assert float(synthesised[:, 2].abs().max()) == 0.0
    assert float((synthesised[:, 1] - x[:, 1]).abs().max()) < 1e-5


def test_band_probes_stay_inside_their_band():
    """G7 is worthless if its perturbations are not band-limited.

    Measured before the fix: 71-74% of the energy landed outside the band,
    so the gate that decides whether this arm proceeds would have reported
    four numbers that look like band sensitivities and are not.
    """
    from ...stage_cap.diagnostics import haar_transform
    from ..qualification import BANDS, _band_mask, _inverse_haar

    size = 8
    # Both channel counts: G7 runs on colour images, and band-limiting has to
    # hold there rather than only on the single planes the test used to build.
    for channels in (1, 3):
        for band in BANDS:
            mask = _band_mask(size, band, torch.device("cpu"))
            coefficients = (
                haar_transform(torch.randn(4, channels, size, size)) * mask
            )
            back = haar_transform(_inverse_haar(coefficients))
            inside = float((back * mask).square().sum())
            outside = float((back * (1 - mask)).square().sum())
            leak = outside / max(inside + outside, 1e-30)
            assert leak < 0.01, (
                f"{band} leaked {leak:.1%} outside its band at C={channels}"
            )


def test_normalization_does_not_allocate_a_quadratic_tensor():
    """The naive [N, N, L, C] difference is 12.4 GB at the production shape.

    Exercised at a shape whose broadcast form would be ~1.7 GB in float64 --
    large enough that the old path would be obvious, small enough to run.
    """
    features = smoke_config().features
    values = {"a": torch.randn(96, 66, 96)}
    calibrated = calibrate_normalization(values, features)
    assert calibrated["a"].level_scale > 0


def test_bandwidth_uses_matched_location_distances():
    """The field compares location l to location l; calibration must too.

    Flattening [N, L, C] to [N*L, C] and slicing calibrates on distances
    *between* locations of a few images -- a different distribution -- and
    silently drops most of the images.
    """
    config = smoke_config().field_config
    torch.manual_seed(0)
    # Locations deliberately given very different offsets: pooling across them
    # would inflate the distance scale far above the within-location one.
    values = torch.randn(64, 8, 6)
    values = values + torch.arange(8).reshape(1, 8, 1) * 25.0
    record = calibrate_bandwidth(values, 0.5, config)
    assert record["locations"] == 8
    assert record["images"] == 64
    within = float(
        torch.cdist(values[:, 0], values[:, 0])[
            ~torch.eye(64, dtype=torch.bool)
        ].median()
    )
    # The calibrated scale must track the within-location spread, not the
    # between-location offsets.
    assert record["distance_median"] < 3.0 * within, (record["distance_median"], within)


def test_cka_is_one_for_identical_and_low_for_independent():
    left = torch.randn(64, 20)
    assert abs(linear_cka(left, left) - 1.0) < 1e-9
    assert linear_cka(left, torch.randn(64, 20)) < 0.6


def test_inter_level_gate_rejects_duplicate_levels():
    values = torch.randn(32, 66, 8)
    config = asfd_config().qualification
    _name, ok, _, _ = gate_inter_level({"a": values, "b": values.clone()}, config)
    assert not ok, "identical levels must fail G8"
    _name, ok, _, _ = gate_inter_level(
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
            "added": [],
            "removed": [],
            "changed": [],
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


def test_transitive_training_and_final_evaluation_dependencies_are_hashed():
    manifest = source_manifest()
    required = (
        "stage_cap/monitoring.py",
        "stage_cap2/benchmark.py",
        "stage_cap2/budget.py",
        "stage_cap2/early_admission.py",
        "stage_cap2/gate_calibration.py",
        "stage_cap2/hardware.py",
        "stage_cap2/metric_calibration.py",
        "stage_cap2/numerical_admission.py",
        "stage_cap2/preflight.py",
        "stage_cap2/preview.py",
        "stage_cap2/selection.py",
        "stage_b2/metrics.py",
        "appearance.py",
        "fid.py",
        "stage_asfd/final_visual_review.py",
    )
    for suffix in required:
        assert any(name.endswith(suffix) for name in manifest), suffix


def test_asfd_manifest_closes_over_local_python_imports() -> None:
    """Every local module imported by source-bound ASFD code is also bound."""
    package_root = Path(__file__).resolve().parents[2]
    manifest = {
        name.split("encoder_independent_drifting/")[-1] for name in source_manifest()
    }
    missing: list[str] = []
    for relative in sorted(manifest):
        if not relative.endswith(".py"):
            continue
        module = package_root / relative
        current_package = list(Path(relative).parent.parts)
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            keep = len(current_package) - (node.level - 1)
            if keep < 0:
                continue
            target_parts = current_package[:keep]
            if node.module:
                target_parts.extend(node.module.split("."))
            candidate = "/".join(target_parts) + ".py"
            init_candidate = "/".join(target_parts + ["__init__"]) + ".py"
            if (package_root / candidate).is_file() and candidate not in manifest:
                missing.append(f"{relative} imports {candidate}")
            elif (
                not (package_root / candidate).is_file()
                and (package_root / init_candidate).is_file()
                and init_candidate not in manifest
            ):
                missing.append(f"{relative} imports {init_candidate}")
    assert not missing, missing


def test_existing_terminal_result_reopens_checkpoints_recovery_and_mirrors(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, object]] = []
    source = {"source.py": "a" * 64}
    preflight = {"artifact_sha256": "p" * 64}
    checkpoint_mirrors = {
        kind: {"relative_path": f"checkpoints/final_{kind}.pt", "sha256": kind}
        for kind in ("raw", "ema")
    }
    recovery_mirror = {
        "relative_path": "asfd_recovery.pt",
        "sha256": "r" * 64,
        "recovery_step": 800_000,
    }
    completed = {
        "decision": "GO",
        "source_sha256": source,
        "preflight": {"sha256": preflight["artifact_sha256"]},
        "final_step": 800_000,
        "profile": {"name": "profile"},
        "run_identity_sha256": "i" * 64,
        "checkpoints": {
            "800000": {
                kind: {
                    "path": f"checkpoints/final_{kind}.pt",
                    "sha256": kind * 32,
                    "durable_mirror": checkpoint_mirrors[kind],
                }
                for kind in ("raw", "ema")
            }
        },
        "recovery": {
            "path": "asfd_recovery.pt",
            "sha256": "r" * 64,
            "durable_mirror": recovery_mirror,
        },
    }

    monkeypatch.setattr(asfd_continuation, "source_manifest", lambda: source)

    def fake_checkpoint(path, **kwargs):
        calls.append(("checkpoint", kwargs["kind"]))
        return {"state_dict": {}}

    monkeypatch.setattr(asfd_continuation, "load_checkpoint", fake_checkpoint)
    monkeypatch.setattr(
        asfd_continuation,
        "load_recovery_payload",
        lambda *args, **kwargs: (
            {"planned_updates": 800_000, "completed_updates": 800_000},
            "r" * 64,
        ),
    )

    class Mirror:
        def verify(self, path):
            kind = "raw" if "raw" in path.name else "ema"
            calls.append(("mirror", kind))
            return checkpoint_mirrors[kind]

        def verify_recovery(self, path, *, recovery_step):
            calls.append(("recovery", recovery_step))
            return recovery_mirror

    asfd_continuation._revalidate_terminal_result(
        completed,
        result_path=tmp_path / "asfd_result.json",
        preflight=preflight,
        mirror=Mirror(),
    )
    assert calls == [
        ("checkpoint", "raw"),
        ("mirror", "raw"),
        ("checkpoint", "ema"),
        ("mirror", "ema"),
        ("recovery", 800_000),
    ]


def test_wall_stop_is_bound_to_a_durable_recovery_and_cannot_disappear(
    tmp_path,
    monkeypatch,
) -> None:
    stop_path = tmp_path / "asfd_wall_stop.json"
    recovery_path = tmp_path / "asfd_recovery.pt"
    recovery_mirror = {
        "relative_path": "asfd_recovery.pt",
        "sha256": "r" * 64,
        "recovery_step": 790_000,
    }
    stopped = {
        "decision": "HALT",
        "reason": "measured-continuation-wall-exhausted",
        "recovery_step": 790_000,
        "preflight_sha256": "p" * 64,
        "source_sha256": {"source.py": "s" * 64},
        "recovery": {
            "path": recovery_path.name,
            "sha256": "r" * 64,
            "durable_mirror": recovery_mirror,
        },
    }
    monkeypatch.setattr(
        asfd_continuation, "source_manifest", lambda: {"source.py": "s" * 64}
    )
    monkeypatch.setattr(
        asfd_continuation,
        "load_recovery_payload",
        lambda *args, **kwargs: (
            {"planned_updates": 800_000, "completed_updates": 790_000},
            "r" * 64,
        ),
    )
    calls = []

    class Mirror:
        def verify_recovery(self, path, *, recovery_step):
            calls.append(("recovery", path, recovery_step))
            return recovery_mirror

        def mirror(self, path):
            calls.append(("stop", path))
            return {"relative_path": path.name}

    _revalidate_wall_stop(
        stopped,
        stop_path=stop_path,
        preflight={"artifact_sha256": "p" * 64},
        mirror=Mirror(),
    )
    assert calls == [
        ("recovery", recovery_path, 790_000),
        ("stop", stop_path),
    ]

    changed = dict(stopped, recovery_step=790_001)
    try:
        _revalidate_wall_stop(
            changed,
            stop_path=stop_path,
            preflight={"artifact_sha256": "p" * 64},
            mirror=Mirror(),
        )
    except RuntimeError as error:
        assert "recovery changed" in str(error)
    else:
        raise AssertionError("ASFD accepted a wall stop detached from its recovery")


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
