"""Focused integrity tests for CAP2 numerical admission.

These tests intentionally use tiny CPU models and explicit in-memory image
pools.  They verify the methodology and provenance machinery; only the full
CUDA matrix on a production checkpoint can authorize a paid run.
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from ...stage_cap.model import CAPPixelTransformer
from ...stage_cap.preflight import wake_output_path
from ..config import screen_profile
from ..numerical_admission import (
    AUDIT_BATCH,
    AUDIT_SOURCES,
    AUDIT_STRATA,
    MINIMUM_REPEATS,
    _audit_inputs,
    _checkpoint_identity,
    _strict_admission_checks,
    audit_stratum,
    production_numerical_mode,
)


def _tiny_model(seed: int = 1) -> tuple[CAPPixelTransformer, object]:
    frozen = screen_profile("ordered_uniform", "local_1000_d0002_fp32", smoke=True)
    model = wake_output_path(CAPPixelTransformer(frozen.model, seed).eval())
    return model, frozen


def test_design_covers_real_stress_endpoint_and_weighted_tail():
    assert AUDIT_SOURCES == (
        "cifar10_train",
        "synthetic_gaussian",
        "synthetic_checkerboard",
    )
    assert ("exact_inference", 1.0, 0.0) in AUDIT_STRATA
    assert any(name == "low_t_weighted" and t <= 0.03 for name, t, _ in AUDIT_STRATA)
    assert any(
        name == "below_floor_weighted" and t < 0.02 for name, t, _ in AUDIT_STRATA
    )
    assert AUDIT_BATCH == 4
    assert MINIMUM_REPEATS >= 3


def test_real_and_synthetic_batches_are_replayable_and_identified():
    model, _ = _tiny_model()
    pool = torch.linspace(
        -1,
        1,
        steps=24 * 3 * 8 * 8,
        dtype=torch.float32,
    ).view(24, 3, 8, 8)
    first = _audit_inputs(
        model,
        source="cifar10_train",
        batch=4,
        seed=91,
        device=torch.device("cpu"),
        real_pool=pool,
    )
    second = _audit_inputs(
        model,
        source="cifar10_train",
        batch=4,
        seed=91,
        device=torch.device("cpu"),
        real_pool=pool,
    )
    assert first[2] == second[2]
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert len(first[2]["sample_ids"]) == 4
    checker, _, metadata = _audit_inputs(
        model,
        source="synthetic_checkerboard",
        batch=4,
        seed=92,
        device=torch.device("cpu"),
        real_pool=None,
    )
    assert set(checker.unique().tolist()) == {-1.0, 1.0}
    assert metadata["sample_ids"] is None


def test_scale_blind_gradient_cannot_pass_strict_admission():
    quotient = {"relative_rms_max": 0.01, "cosine_min": 0.999}
    target = {"relative_rms_max": 0.01, "cosine_min": 0.9999}
    # This would pass the historical cosine-only test but is 30% too large.
    gradient = {
        "cosine": 0.999,
        "relative_l2": 0.30,
        "norm_ratio": 1.30,
        "finite": True,
    }
    checks = _strict_admission_checks(quotient, target, gradient)
    assert checks["gradient_cosine"]
    assert not checks["gradient_relative_l2"]
    assert not checks["gradient_norm_ratio"]
    assert not all(checks.values())


def test_audit_reports_assembled_target_gradient_and_batch_provenance():
    model, frozen = _tiny_model(3)
    pool = torch.randn(12, 3, 8, 8)
    result = audit_stratum(
        model,
        frozen.objective,
        t_value=0.85,
        r_value=0.10,
        batch=4,
        seed=17,
        delta=frozen.objective.emf_delta,
        evaluation_mode=frozen.objective.stopped_evaluation,
        device=torch.device("cpu"),
        include_gradient=True,
        source="cifar10_train",
        real_pool=pool,
        stratum_name="integrity_test",
    )
    assert result["stratum"] == "integrity_test"
    assert result["input_batch"]["batch"] == 4
    assert len(result["input_batch"]["sample_ids"]) == 4
    assert result["quotient"] == result["target"]
    assert result["assembled_target"]["relative_rms_max"] >= 0
    assert result["gradient"]["finite"]
    assert {"cosine", "norm_ratio", "relative_l2", "by_group"} <= set(
        result["gradient"]
    )
    assert result["verdict"] in {"PASS", "FAIL"}


def test_production_numerical_context_is_explicit_and_restores_state():
    before = torch.are_deterministic_algorithms_enabled()
    before_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    before_benchmark = torch.backends.cudnn.benchmark
    with production_numerical_mode(torch.device("cpu")) as report:
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert report["deterministic_algorithms"]
        assert not report["graded_forward_tf32"]
        assert not report["graded_backward_tf32"]
    assert torch.are_deterministic_algorithms_enabled() == before
    assert torch.is_deterministic_algorithms_warn_only_enabled() == before_warn
    assert torch.backends.cudnn.benchmark == before_benchmark


def test_checkpoint_identity_binds_profile_kind_step_and_parameter_count():
    model, frozen = _tiny_model()
    payload = {
        "stage": "cap-emf-1",
        "step": 650_000,
        "kind": "ema",
        "arm": None,
        "profile": asdict(frozen),
        "parameter_count": model.parameter_count(),
    }
    valid = _checkpoint_identity(payload, model)
    assert valid["valid"]
    invalid = _checkpoint_identity({**payload, "parameter_count": 1}, model)
    assert not invalid["valid"]
    assert not invalid["checks"]["parameter_count_recognized"]


def test_checkpoint_identity_names_the_historical_buffer_count():
    model, frozen = _tiny_model()
    state = model.state_dict()
    payload = {
        "stage": "cap-emf-1",
        "step": 650_000,
        "kind": "ema",
        "profile": asdict(frozen),
        "parameter_count": sum(value.numel() for value in state.values()),
        "state_dict": state,
    }
    identity = _checkpoint_identity(payload, model)
    assert identity["valid"]
    assert identity["recorded_parameter_count_semantics"] == (
        "legacy_state_dict_values_including_buffers"
    )
    assert identity["loaded_trainable_parameter_count"] == model.parameter_count()


def _run_all() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    print(f"=== numerical-admission-integrity ({len(tests)} tests) ===")
    failures = 0
    for name, function in tests:
        try:
            function()
        except Exception as error:  # noqa: BLE001
            failures += 1
            print(f"  [FAIL] {name}: {type(error).__name__}: {error}")
        else:
            print(f"  [PASS] {name}")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
