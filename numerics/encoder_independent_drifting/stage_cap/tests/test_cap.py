"""CAP-EMF-1 regression suite."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from ..artifacts import _DEPENDENCIES, manifest_difference, source_manifest
from ..config import (
    FEATURE_LEVELS,
    PARAMETER_CEILING,
    TRAIN_POOL_SIZE,
    examples_seen,
    profile,
)
from ..data import sealed_test_pool
from ..diagnostics import (
    capability_gate,
    effective_rank,
    endpoint_health,
    haar_band_variances,
    haar_transform,
    rank_noncollapse,
)
from ..model import CAPPixelTransformer, one_step_sample
from ..objective import (
    directional_jvp_reference,
    emf_local_difference,
    emf_loss,
    sample_time_triangle,
)
from ..preflight import wake_output_path
from ..training import EMAState, clip_fraction, train_cap_unit


def _tiny(seed: int = 3):
    small = profile("smoke")
    return CAPPixelTransformer(small.model, seed), small


def test_capability_profile_is_frozen():
    frozen = profile("capability")
    assert frozen.model.patch_size == 2
    assert frozen.model.tokens == 256
    assert frozen.model.width == 512
    assert frozen.model.depth == 12
    assert frozen.train.updates == 320_000
    assert frozen.train.effective_batch == 64
    assert frozen.train.checkpoint_updates[-1] == 320_000
    assert frozen.train.ema_decay == 0.9999


def test_epoch_count_is_a_healthy_regime():
    """410 epochs, not the 4,096 a 5,000-image class would have given.

    The single-class draft made memorization the likely route to coherent
    samples; unconditional CIFAR-10 at this horizon does not.
    """
    frozen = profile("capability")
    epochs = examples_seen(frozen.train) / TRAIN_POOL_SIZE
    assert 300 < epochs < 600, epochs
    assert examples_seen(frozen.train) == 20_480_000


def test_matched_drifting_budget_is_stated_in_examples():
    """A drifting arm with a 256-cloud matches on examples, not updates."""
    frozen = profile("capability")
    total = examples_seen(frozen.train)
    assert total // 256 == 80_000
    # Matching updates instead would give drifting 4x the data exposure.
    assert frozen.train.updates * 256 == 4 * total


def test_parameter_count_clears_the_ceiling():
    frozen = profile("capability")
    model = CAPPixelTransformer(frozen.model, 1)
    count = model.parameter_count()
    assert count <= PARAMETER_CEILING, f"{count} exceeds {PARAMETER_CEILING}"


def test_no_conditioning_token_slice():
    """The image grid is the whole sequence: 256 tokens, reshaping to 16x16."""
    frozen = profile("capability")
    model = CAPPixelTransformer(frozen.model, 2).eval()
    images = torch.randn(2, 3, 32, 32)
    t = torch.full((2,), 0.5)
    h = torch.full((2,), 0.2)
    with torch.no_grad():
        _, features = model.forward_with_features(images, t, h)
    assert sorted(features) == sorted(name for name, _, _ in FEATURE_LEVELS)
    for value in features.values():
        assert value.shape == (2, 256, frozen.model.width)
        value.reshape(2, 16, 16, frozen.model.width)


def test_feature_taps_do_not_change_the_forward():
    model, _ = _tiny()
    model.eval()
    images = torch.randn(3, 3, 8, 8)
    t = torch.full((3,), 0.6)
    h = torch.full((3,), 0.1)
    with torch.no_grad():
        plain = model(images, t, h)
        tapped, _ = model.forward_with_features(images, t, h)
    assert torch.equal(plain, tapped)


def test_adaln_zero_starts_as_identity():
    """Zero modulation means zero residual gates, so blocks pass tokens through."""
    model, _ = _tiny()
    for block in list(model.encoder) + list(model.decoder):
        assert torch.equal(block.modulation.weight, torch.zeros_like(block.modulation.weight))
        assert torch.equal(block.modulation.bias, torch.zeros_like(block.modulation.bias))
        tokens = torch.randn(2, 5, model.config.width)
        conditioning = torch.randn(2, model.config.condition_dim)
        assert torch.allclose(block(tokens, conditioning), tokens, atol=1e-6)


def test_refiner_starts_as_identity():
    model, _ = _tiny()
    images = torch.randn(2, 3, 8, 8)
    with torch.no_grad():
        assert torch.allclose(model.refiner(images), images, atol=1e-6)


def test_refiner_is_dead_for_exactly_one_step_then_trains():
    """The refiner sees an all-zero input at init; it must not stay dead.

    ``pixel_head`` is zero, so ``base`` is zero; every refiner bias is zero too,
    so every intermediate activation is zero and the final convolution has no
    gradient on the first update.  That is benign only because ``pixel_head``
    moves immediately and wakes it.  A permanently dead refiner would be a
    silent capacity loss of exactly the kind the HH gate is meant to fix.
    """
    small = profile("smoke")
    pool = torch.randn(32, 3, 8, 8)
    model = CAPPixelTransformer(small.model, 4)
    final_conv = model.refiner.body[-1]
    before = final_conv.weight.detach().clone()
    train_cap_unit(pool, small, "cpu")

    # Reproduce the loop's first two updates directly so the gradient at each
    # step is observable rather than inferred.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(7)
    grads = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        clean = pool[:4]
        noise = torch.randn(clean.shape, generator=generator)
        triangle = sample_time_triangle(4, small.objective, generator)
        emf_loss(model, clean, noise, triangle, small.objective).loss.backward()
        grad = final_conv.weight.grad
        grads.append(0.0 if grad is None else float(grad.abs().max()))
        optimizer.step()
    assert grads[0] == 0.0, "the refiner is expected to be dead on update one"
    assert grads[1] > 0.0, "the refiner never woke up"
    assert not torch.equal(final_conv.weight.detach(), before)


def test_one_call_inference():
    model, _ = _tiny()
    model.eval()
    calls = {"n": 0}
    handle = model.register_forward_pre_hook(
        lambda module, args: calls.__setitem__("n", calls["n"] + 1)
    )
    with torch.no_grad():
        one_step_sample(model, torch.randn(2, 3, 8, 8))
    handle.remove()
    assert calls["n"] == 1


def test_a_fresh_model_is_the_zero_function():
    """Zero-initialized pixel head, refiner and modulation compose to zero.

    Intended, and the reason ``wake_output_path`` exists: a derivative audit on
    an untouched model compares zero against zero and proves nothing.
    """
    model, _ = _tiny()
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 8, 8), torch.full((2,), 0.7), torch.full((2,), 0.3))
    assert float(out.abs().max()) == 0.0


def test_emf_difference_converges_to_the_jvp_at_first_order():
    """A sign, clock or velocity error would break the convergence *rate*."""
    model, small = _tiny()
    model = wake_output_path(model.double().eval())
    state = torch.randn(3, 3, 8, 8, dtype=torch.float64)
    t = torch.tensor([0.9, 0.7, 0.5], dtype=torch.float64)
    r = torch.tensor([0.1, 0.2, 0.1], dtype=torch.float64)
    exact = directional_jvp_reference(
        model, state, t, r, small.objective.emf_denominator_floor
    ).detach()
    magnitude = float(exact.abs().max())
    assert magnitude > 1e-6, "the woken model is still degenerate"
    errors = []
    for delta in (1e-3, 1e-4, 1e-5):
        _, quotient = emf_local_difference(
            model, state, t, r, delta, small.objective.emf_denominator_floor
        )
        errors.append(float((quotient - exact).abs().max()) / magnitude)
    for before, after in zip(errors, errors[1:]):
        assert 4.0 <= before / after <= 20.0, f"not first order: {errors}"
    assert errors[-1] < 1e-2


def test_diagonal_rows_reduce_to_endpoint_regression():
    """When r == t the quotient is zero and the target is the clean image."""
    model, small = _tiny()
    state = torch.randn(4, 3, 8, 8)
    t = torch.full((4,), 0.5)
    _, quotient = emf_local_difference(
        model, state, t, t.clone(), small.objective.emf_delta,
        small.objective.emf_denominator_floor,
    )
    assert torch.equal(quotient, torch.zeros_like(quotient))


def test_triangle_respects_its_invariants():
    small = profile("smoke")
    generator = torch.Generator().manual_seed(5)
    triangle = sample_time_triangle(512, small.objective, generator)
    assert bool((triangle.r > 0).all())
    assert bool((triangle.r <= triangle.t + 1e-9).all())
    assert bool((triangle.t > 0).all())
    diagonal_share = float(triangle.diagonal.double().mean())
    assert 0.35 < diagonal_share < 0.65
    assert torch.allclose(
        triangle.r[triangle.diagonal], triangle.t[triangle.diagonal]
    )


def test_haar_is_orthonormal():
    images = torch.randn(4, 3, 32, 32, dtype=torch.float64)
    coefficients = haar_transform(images)
    assert abs(float(coefficients.square().sum()) - float(images.square().sum())) < 1e-8


def test_haar_bands_detect_a_missing_high_frequency():
    """A blurred batch must lose HH energy relative to the original."""
    images = torch.randn(8, 3, 32, 32, dtype=torch.float64)
    blurred = torch.nn.functional.avg_pool2d(images, 2)
    blurred = torch.nn.functional.interpolate(blurred, scale_factor=2, mode="nearest")
    assert haar_band_variances(blurred)["HH"] < 0.25 * haar_band_variances(images)["HH"]


def test_corrected_rank_rule_matches_the_s3r_case():
    # S3R EMF: best 4.056, final 1.661.  final/max = 0.410 failed it wrongly.
    assert rank_noncollapse(1.661, 4.056, 0.8)
    # pMF's 0.349 must still be rejected.
    assert not rank_noncollapse(0.349, 4.056, 0.8)
    # And a genuine late collapse from a healthy peak is still caught.
    assert not rank_noncollapse(0.5, 0.95, 0.8)


def test_endpoint_health_reports_amplitude_with_rank():
    target = torch.randn(64, 3, 32, 32)
    constant = torch.ones(64, 3, 32, 32) * float(target.square().mean().sqrt())
    record = endpoint_health(constant, target)
    # A constant image can clear a second-moment threshold; centered variance
    # is what exposes it, which is why both are reported.
    assert record["second_moment_ratio"] > 0.9
    assert record["centered_variance_ratio"] < 1e-6


def test_effective_rank_is_a_participation_ratio():
    one_direction = torch.randn(1, 8) * torch.ones(32, 1)
    assert effective_rank(one_direction.reshape(32, 8, 1, 1)) < 1.5


def test_capability_gate_verdicts():
    frozen = profile("capability")
    healthy = {
        "second_moment_ratio": 0.9,
        "centered_variance_ratio": 0.9,
        "effective_rank_ratio": 0.95,
        "haar_HH_ratio": 0.7,
        "haar_LH_ratio": 0.8,
        "haar_HL_ratio": 0.8,
    }
    assert capability_gate(healthy, 1.0, 0.01, 0, 1, frozen.gate)["verdict"] == "PASS"
    detail_poor = dict(healthy, haar_HH_ratio=0.2)
    assert (
        capability_gate(detail_poor, 1.0, 0.01, 0, 1, frozen.gate)["verdict"]
        == "PASS_DETAIL_POOR"
    )
    # S3R's measured HH of 0.159 with weak amplitude is a plain FAIL.
    weak = dict(healthy, haar_HH_ratio=0.159, second_moment_ratio=0.495)
    assert capability_gate(weak, 1.0, 0.01, 0, 1, frozen.gate)["verdict"] == "FAIL"
    # Two calls at inference is never acceptable.
    assert capability_gate(healthy, 1.0, 0.01, 0, 2, frozen.gate)["verdict"] == "FAIL"


def test_ema_arithmetic_and_recovery_round_trip():
    model, _ = _tiny()
    ema = EMAState(model, 0.9)
    name = next(iter(ema.shadow))
    before = ema.shadow[name].clone()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    ema.update(model)
    expected = before * 0.9 + model.state_dict()[name].float() * 0.1
    assert torch.allclose(ema.shadow[name], expected, atol=1e-6)
    assert abs(ema.initialization_weight() - 0.9) < 1e-12
    restored = EMAState(model, 0.9)
    restored.load_recovery_state(ema.recovery_state())
    assert torch.allclose(restored.shadow[name], ema.shadow[name])


def test_ema_is_mature_at_the_frozen_horizon():
    train = profile("capability").train
    assert train.ema_half_life > 0
    assert train.ema_mature_at() < train.updates
    # S3R's failure mode: 28.7% of the average still came from initialization.
    assert train.ema_decay**train.updates < 1e-6


def test_training_runs_and_records_health():
    small = profile("smoke")
    pool = torch.randn(32, 3, 8, 8)
    outcome = train_cap_unit(pool, small, "cpu")
    assert outcome.optimizer_updates == small.train.updates
    assert outcome.examples_seen == small.train.updates * small.train.effective_batch
    assert outcome.health, "no health record was produced"
    assert outcome.parameter_count > 0
    assert outcome.nonfinite_updates == 0
    assert 0.0 <= clip_fraction(outcome) <= 1.0
    # Three model evaluations per microbatch example: one graded, two stopped.
    expected = (
        3
        * small.train.updates
        * small.train.accumulation_steps
        * small.train.micro_batch
    )
    assert outcome.model_forwards == expected


def test_restart_reproduces_an_uninterrupted_run():
    small = profile("smoke")
    pool = torch.randn(32, 3, 8, 8)
    reference = train_cap_unit(pool, small, "cpu")
    with tempfile.TemporaryDirectory() as directory:
        recovery = Path(directory) / "recovery.pt"
        half = replace(
            small, train=replace(small.train, updates=2, checkpoint_updates=(2,))
        )
        train_cap_unit(pool, half, "cpu", recovery_path=recovery)
        resumed = train_cap_unit(pool, small, "cpu", recovery_path=recovery)
    assert reference.optimizer_updates == resumed.optimizer_updates
    assert abs(reference.history[-1]["raw_mse"] - resumed.history[-1]["raw_mse"]) < 1e-9


def test_source_manifest_is_an_explicit_list_not_a_glob():
    """A new module beside the stage must not invalidate a frozen preflight.

    This is the B2.5 lesson: b1_freeze globs its package and its tests
    directory, so two unrelated test files added later permanently broke a
    completed confirmation.
    """
    manifest = source_manifest()
    assert len(manifest) == len(_DEPENDENCIES)
    assert not any("tests/" in name for name in manifest)
    here = Path(__file__).resolve().parents[1]
    scratch = here / "_manifest_probe.py"
    scratch.write_text("# transient probe\n", encoding="utf-8")
    try:
        difference = manifest_difference(manifest)
        assert difference == {"added": [], "removed": [], "changed": []}
    finally:
        scratch.unlink()


def test_manifest_difference_reports_a_changed_dependency():
    manifest = dict(source_manifest())
    victim = next(iter(manifest))
    manifest[victim] = "0" * 64
    assert manifest_difference(manifest)["changed"] == [victim]


def test_test_split_is_sealed_by_construction():
    try:
        sealed_test_pool()
    except RuntimeError as error:
        assert "sealed" in str(error)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("the sealed test pool opened without acknowledgement")


def test_no_correction_term_is_importable_from_this_stage():
    """CAP-EMF-1 trains the foundation alone; a correction here is a bug."""
    import numerics.encoder_independent_drifting.stage_cap.training as training

    text = Path(training.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("laplace", "sinkhorn", "spectral_anchor", "drift_energy"):
        assert forbidden not in text, f"{forbidden} leaked into the foundation"


def _run_all() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    print(f"=== stage_cap ({len(tests)} tests) ===")
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
