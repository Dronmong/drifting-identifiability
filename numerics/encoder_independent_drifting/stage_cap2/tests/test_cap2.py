"""Regression tests for CAP-EMF-2 repairs and experiment guards."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
import torch

from ...stage_cap.diagnostics import capability_gate
from ...stage_cap.model import CAPPixelTransformer
from ...stage_cap.monitoring import ObjectiveLedger
from ...stage_cap.objective import sample_time_triangle
from ...stage_cap.preflight import wake_output_path
from ...stage_cap.training import load_recovery_payload, train_cap_unit
from ..artifacts import source_manifest
from ..checkpoint_forensics import patch_phase_report
from ..config import SAMPLER_ARMS, numerical_candidate, screen_profile
from ..hardware import hardware_binding
from ..numerical_admission import audit_stratum
from ..positive_control import (
    NETWORK_SHA256,
    UPSTREAM_COMMIT,
    balanced_class_indices,
    source_citation,
)
from ..preview import save_fixed_grid
from ..production_readiness import production_commands
from ..promotion import _late_inference_corner_trajectory
from ..sampler_audit import audit_sampler


def test_positive_control_allocation_and_citation_are_frozen():
    classes = balanced_class_indices(20)
    assert classes.tolist() == list(range(10)) * 2
    citation = source_citation()
    assert UPSTREAM_COMMIT in citation
    assert NETWORK_SHA256 in citation
    assert "class=seed mod 10" in citation


def test_fixed_preview_grid_is_deterministic_and_uncurated():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "grid.png"
        images = torch.linspace(-1.0, 1.0, 12 * 3 * 4 * 4).reshape(12, 3, 4, 4)
        digest = save_fixed_grid(images, path, rows=3, columns=4)
        assert path.is_file()
        assert len(digest) == 64
        first = path.read_bytes()
        second_path = Path(directory) / "grid-copy.png"
        assert save_fixed_grid(images, second_path, rows=3, columns=4) == digest
        assert second_path.read_bytes() == first
        with pytest.raises(ValueError, match="not enough images"):
            save_fixed_grid(
                images, Path(directory) / "too-large.png", rows=4, columns=4
            )


def test_production_readiness_can_only_build_the_four_pretraining_gates():
    commands = production_commands(
        python="python",
        checkpoint=Path("checkpoint.pt"),
        expected_gpu_name="RTX 4090",
        hourly_rate=0.75,
        candidate="local_1000_d0002_fp32",
        micro_batch=16,
        data_root=None,
        output_dir=Path("evidence"),
        sampler_audit=Path("sampler.json"),
        gate_calibration=Path("gate.json"),
        baseline_standard=Path("baseline.json"),
        positive_control_standard=Path("control.json"),
        metric_calibration=Path("calibration.json"),
        max_total_cost=25.0,
        nontraining_reserve=3.0,
        contingency_fraction=0.15,
        durable_mirror_dir=Path("durable-benchmark"),
        durable_storage_root=Path("durable-root"),
        artifact_storage_reserve_gib=20.0,
        storage_contingency_fraction=0.20,
    )
    modules = [command[2] for command in commands]
    assert modules == [
        "numerics.encoder_independent_drifting.stage_cap2.numerical_admission",
        "numerics.encoder_independent_drifting.stage_cap2.checkpoint_forensics",
        "numerics.encoder_independent_drifting.stage_cap2.benchmark",
        "numerics.encoder_independent_drifting.stage_cap2.preflight",
    ]
    assert all("run_screen" not in token for command in commands for token in command)
    assert "--hourly-rate" in commands[2]
    assert commands[2][commands[2].index("--hourly-rate") + 1] == "0.75"
    assert commands[3][commands[3].index("--max-total-cost") + 1] == "25.0"
    assert commands[3][commands[3].index("--nontraining-reserve") + 1] == "3.0"
    assert (
        commands[3][commands[3].index("--post-foundation-training-reserve") + 1]
        == "10.0"
    )
    assert "--durable-mirror-dir" in commands[2]
    assert "--i-confirm-durable-mirror" in commands[2]
    assert commands[3][commands[3].index("--durable-storage-root") + 1] == (
        "durable-root"
    )


def test_historical_numerical_control_is_explicitly_nonlocal():
    legacy = numerical_candidate("legacy_1000_d01")
    repaired = numerical_candidate("local_1000_d0002_fp32")
    assert legacy.maximum_phase_step == 10.0
    assert repaired.maximum_phase_step == 0.2
    assert repaired.stopped_evaluation == "fp32_dense"


def test_all_sampler_arms_preserve_the_scientific_configuration():
    profiles = [
        screen_profile(arm, "local_1000_d0002_fp32", smoke=True) for arm in SAMPLER_ARMS
    ]
    reference = profiles[0]
    for frozen in profiles[1:]:
        assert frozen.model == reference.model
        assert frozen.train == reference.train
        assert frozen.gate == reference.gate
        assert frozen.objective.emf_delta == reference.objective.emf_delta
        assert (
            frozen.objective.stopped_evaluation
            == reference.objective.stopped_evaluation
        )


def test_promoted_horizons_keep_one_recovery_identity():
    short = screen_profile("ordered_uniform", "local_1000_d0002_fp32", updates=150_000)
    promoted = screen_profile(
        "ordered_uniform", "local_1000_d0002_fp32", updates=300_000
    )
    assert short.name == promoted.name
    assert short.train.updates == 150_000
    assert promoted.train.updates == 300_000


def test_promotion_resumes_and_restarts_the_new_final_window():
    first = screen_profile("ordered_uniform", "local_1000_d0002_fp32", smoke=True)
    promoted = replace(
        first,
        train=replace(first.train, updates=6, checkpoint_updates=(2, 4, 6)),
    )
    pool = torch.randn(32, 3, 8, 8)
    with TemporaryDirectory() as directory:
        recovery = Path(directory) / "recovery.pt"
        initial = train_cap_unit(pool, first, "cpu", recovery_path=recovery)
        resumed = train_cap_unit(pool, promoted, "cpu", recovery_path=recovery)
    assert initial.optimizer_updates == 4
    assert resumed.optimizer_updates == 6
    assert resumed.final_window_updates == 2


def test_same_horizon_pause_resumes_exact_recovery_without_a_second_model():
    frozen = screen_profile("ordered_uniform", "local_1000_d0002_fp32", smoke=True)
    pool = torch.randn(32, 3, 8, 8)
    with TemporaryDirectory() as directory:
        recovery = Path(directory) / "recovery.pt"
        paused = train_cap_unit(
            pool,
            frozen,
            "cpu",
            recovery_path=recovery,
            stop_after_updates=2,
            unit_seed=7,
        )
        pause_payload, _ = load_recovery_payload(recovery)
        resumed = train_cap_unit(
            pool,
            frozen,
            "cpu",
            recovery_path=recovery,
            stop_after_updates=4,
            unit_seed=7,
        )
        final_payload, _ = load_recovery_payload(recovery)
    assert paused.optimizer_updates == 2
    assert pause_payload["planned_updates"] == 4
    assert pause_payload["completed_updates"] == 2
    assert resumed.optimizer_updates == 4
    assert final_payload["planned_updates"] == 4
    assert final_payload["completed_updates"] == 4
    assert final_payload["recovery_identity"] == pause_payload["recovery_identity"]


def test_ordered_arms_sort_two_endpoints_and_do_not_floor_the_condition():
    for arm in ("ordered_logitnormal", "ordered_uniform"):
        frozen = screen_profile(arm, "local_1000_d0002_fp32", smoke=True)
        triangle = sample_time_triangle(
            100_000,
            frozen.objective,
            torch.Generator().manual_seed(7),
            diagonal_generator=torch.Generator().manual_seed(8),
        )
        assert frozen.objective.sampled_r_floor == 0.0
        assert bool((triangle.r >= 0).all())
        assert bool((triangle.r <= triangle.t).all())
        # No artificial point mass at CAP's old .01 sampled-endpoint floor.
        assert not bool((triangle.r[~triangle.diagonal] == 0.01).any())
        if arm == "ordered_uniform":
            assert float(triangle.r[~triangle.diagonal].min()) < 0.005
            # Diagonal rows use the first iid draw, not the sorted maximum.
            assert abs(float(triangle.t[triangle.diagonal].mean()) - 0.5) < 0.01
        assert torch.equal(triangle.r[triangle.diagonal], triangle.t[triangle.diagonal])


def test_sampler_audit_separates_coefficient_control_and_corner_coverage():
    legacy = audit_sampler("legacy", count=100_000, seed=11)
    ordered_logit = audit_sampler("ordered_logitnormal", count=100_000, seed=11)
    ordered_uniform = audit_sampler("ordered_uniform", count=100_000, seed=11)
    assert legacy["verdict"] == "PASS"
    assert ordered_logit["verdict"] == "PASS"
    assert ordered_uniform["verdict"] == "PASS"
    assert (
        legacy["coefficient_tail_fraction"]["gt_7"]
        > 100 * ordered_logit["coefficient_tail_fraction"]["gt_7"]
    )
    assert (
        ordered_uniform["inference_corner_fraction_t95_h90"]
        > ordered_logit["inference_corner_fraction_t95_h90"]
    )


def test_objective_ledger_accounts_for_every_row_and_joint_corner():
    count = 8
    outcome = SimpleNamespace(
        t=torch.tensor([0.2, 0.5, 0.7, 0.85, 0.96, 0.981, 0.991, 0.999]),
        r=torch.tensor([0.1, 0.2, 0.3, 0.1, 0.01, 0.01, 0.001, 0.0]),
        interval=torch.tensor([0.1, 0.3, 0.4, 0.75, 0.95, 0.971, 0.99, 0.999]),
        diagonal=torch.zeros(count, dtype=torch.bool),
        active=torch.ones(count, dtype=torch.bool),
        coefficient=torch.arange(count, dtype=torch.float32),
        adaptive_weight=torch.ones(count) * 0.5,
        per_sample_raw_mse=torch.ones(count),
        per_sample_weighted_loss=torch.ones(count) * 2,
        per_sample_output_gradient_norm=torch.ones(count) * 3,
        per_sample_target_rms=torch.ones(count) * 4,
        per_sample_quotient_rms=torch.ones(count) * 5,
    )
    ledger = ObjectiveLedger()
    ledger.add(outcome)
    result = ledger.summary()
    assert result["rows"] == count
    assert result["endpoint_fraction"]["t_gt_0.99"] == 2 / count
    assert result["endpoint_fraction"]["t_gt_0.95_h_gt_0.90"] == 4 / count
    assert result["named_regions"]["inference_corner"]["count"] == 4
    assert result["named_regions"]["inference_corner"]["mean_raw_mse"] == 1.0
    assert sum(result["joint_t_h_counts"].values()) == count


def test_objective_ledger_survives_recovery_before_a_log_boundary():
    count = 3
    outcome = SimpleNamespace(
        t=torch.tensor([0.4, 0.8, 0.99]),
        r=torch.tensor([0.2, 0.1, 0.01]),
        interval=torch.tensor([0.2, 0.7, 0.98]),
        diagonal=torch.zeros(count, dtype=torch.bool),
        active=torch.ones(count, dtype=torch.bool),
        coefficient=torch.arange(count, dtype=torch.float32),
        adaptive_weight=torch.ones(count),
        per_sample_raw_mse=torch.ones(count),
        per_sample_weighted_loss=torch.ones(count),
        per_sample_output_gradient_norm=torch.ones(count),
        per_sample_target_rms=torch.ones(count),
        per_sample_quotient_rms=torch.ones(count),
    )
    before = ObjectiveLedger()
    before.add(outcome)
    after = ObjectiveLedger()
    after.load_state_dict(before.state_dict())
    assert after.summary()["rows"] == count


def test_late_inference_corner_gate_is_all_row_one_sided_and_complete():
    objective = screen_profile(
        "ordered_uniform", "local_1000_d0002_fp32", updates=150_000
    ).objective

    def probe(error: float) -> dict:
        summary = {
            "count": 2_048,
            "mean_raw_mse": error,
            "mean_target_rms": 1.0,
            "mean_quotient_rms": 1.0,
            "coefficient": {"minimum": 0.0, "mean": 1.0, "maximum": 99.0},
            "nonfinite_rows": 0,
        }
        return {
            "condition": {"t": 1.0, "r": 0.0, "h": 1.0},
            "sealed_train_only": True,
            "sample_count": 2_048,
            "objective_numerics": {
                "stopped_evaluation": objective.stopped_evaluation,
                "emf_delta": objective.emf_delta,
                "emf_denominator_floor": objective.emf_denominator_floor,
            },
            "raw": dict(summary),
            "ema": dict(summary),
        }

    history = [
        {
            "step": step,
            "objective_ledger": {
                "named_regions": {
                    "inference_corner": {
                        # Natural occupancy may legitimately be zero for an
                        # arm; it describes sampler support, not endpoint
                        # performance.
                        "count": 0,
                        # Improvement in the second window must not fail a
                        # one-sided explosion guard.
                        "mean_raw_mse": 2.0 if step <= 125_000 else 0.5,
                    }
                }
            },
        }
        for step in range(100_500, 150_001, 500)
    ]
    result = {
        "declared_profile": {
            "train": {"log_every": 500, "audit_samples": 2_048},
            "objective": {
                "stopped_evaluation": objective.stopped_evaluation,
                "emf_delta": objective.emf_delta,
                "emf_denominator_floor": objective.emf_denominator_floor,
            },
        },
        "training": {
            "history": history,
            "health": [
                {"step": 100_000, "fixed_exact_inference_corner": probe(2.0)},
                {"step": 150_000, "fixed_exact_inference_corner": probe(0.5)},
            ],
        },
    }
    report = _late_inference_corner_trajectory(result)
    assert report["stable"] is True
    assert [
        window["rows"] for window in report["natural_sampled_support"]["windows"]
    ] == [0, 0]
    assert report["late_error_growth"] == {"raw": 0.25, "ema": 0.25}

    result["training"]["health"][-1]["fixed_exact_inference_corner"] = probe(10.0)
    assert _late_inference_corner_trajectory(result)["stable"] is False
    result["training"]["health"].pop()
    assert _late_inference_corner_trajectory(result)["stable"] is False


def test_two_sided_gate_rejects_cap1_failure_direction():
    frozen = screen_profile("ordered_uniform", "local_1000_d0002_fp32", smoke=True)
    excessive = {
        "second_moment_ratio": 1.0,
        "centered_variance_ratio": 1.0,
        "effective_rank_ratio": 8.44,
        "haar_LL_ratio": 2.50,
        "haar_HH_ratio": 6.37,
        "haar_LH_ratio": 3.85,
        "haar_HL_ratio": 3.50,
        "raw_saturation_fraction": 0.0,
    }
    result = capability_gate(excessive, 8.44, 0.01, 0, 1, frozen.gate)
    assert result["verdict"] == "FAIL"
    assert "H3c_rank_upper" in result["failed"]
    assert "H4h_haar_HH_upper" in result["failed"]


def test_model_exposes_exact_refiner_decomposition():
    frozen = screen_profile("ordered_uniform", "local_1000_d0002_fp32", smoke=True)
    model = CAPPixelTransformer(frozen.model, 1).eval()
    noise = torch.randn(3, 3, 8, 8)
    ones = torch.ones(3)
    parts = model.forward_components(noise, ones, ones)
    assert torch.allclose(parts["base"] + parts["refiner_residual"], parts["final"])
    assert torch.equal(model(noise, ones, ones), parts["final"])


def test_dense_local_difference_runs_on_a_woken_model():
    frozen = screen_profile("ordered_uniform", "local_1000_d0002_fp32", smoke=True)
    model = wake_output_path(CAPPixelTransformer(frozen.model, 2).double().eval())
    state = torch.randn(4, 3, 8, 8, dtype=torch.float64)
    t = torch.full((4,), 0.8, dtype=torch.float64)
    r = torch.full((4,), 0.2, dtype=torch.float64)
    from ...stage_cap.objective import emf_local_difference

    current, quotient = emf_local_difference(
        model,
        state,
        t,
        r,
        frozen.objective.emf_delta,
        frozen.objective.emf_denominator_floor,
        "fp32_dense",
    )
    assert current.shape == quotient.shape == state.shape
    assert bool(torch.isfinite(quotient).all())


def test_numerical_admission_compares_target_and_parameter_gradient():
    frozen = screen_profile("ordered_uniform", "local_1000_d0002_fp32", smoke=True)
    model = wake_output_path(CAPPixelTransformer(frozen.model, 3).eval())
    result = audit_stratum(
        model,
        frozen.objective,
        t_value=0.85,
        r_value=0.10,
        batch=2,
        seed=17,
        delta=frozen.objective.emf_delta,
        evaluation_mode=frozen.objective.stopped_evaluation,
        device=torch.device("cpu"),
        include_gradient=True,
    )
    assert torch.isfinite(torch.tensor(result["target"]["cosine_mean"]))
    assert torch.isfinite(torch.tensor(result["gradient_cosine"]))
    assert set(result["checks"]) == {
        "target_relative_rms",
        "target_cosine",
        "gradient_cosine",
    }


def test_cap2_manifest_includes_every_executable_entry_point():
    manifest = source_manifest()
    for suffix in (
        "stage_cap2/benchmark.py",
        "stage_cap2/budget.py",
        "stage_cap2/checkpoint_forensics.py",
        "stage_cap2/development_evaluation.py",
        "stage_cap2/durable_mirror.py",
        "stage_cap2/final_verdict.py",
        "stage_cap2/metric_calibration.py",
        "stage_cap2/numerical_admission.py",
        "stage_cap2/preflight.py",
        "stage_cap2/preview.py",
        "stage_cap2/production_readiness.py",
        "stage_cap2/promotion.py",
        "stage_cap2/run_screen.py",
        "stage_cap2/selection.py",
        "stage_cap2/standard_metrics.py",
    ):
        assert any(name.endswith(suffix) for name in manifest), suffix


def test_cap2_manifest_closes_over_local_python_imports():
    """No unhashed local module may affect a source-bound CAP2 artifact."""

    import ast

    package_root = Path(__file__).resolve().parents[2]
    manifest = {
        name.split("encoder_independent_drifting/")[-1] for name in source_manifest()
    }
    missing: list[str] = []
    for relative in sorted(manifest):
        if not relative.endswith(".py"):
            continue
        module = package_root / relative
        if not module.is_file():
            continue
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


def test_patch_phase_report_detects_a_checkerboard():
    yy = torch.arange(8)[:, None]
    xx = torch.arange(8)[None, :]
    checker = (1 - 2 * ((yy + xx) % 2)).float().view(1, 1, 8, 8)
    report = patch_phase_report(checker)
    assert report["checkerboard_projection_ratio"] > 0.99
    assert report["phase_imbalance_ratio"] > 0.99


def test_cpu_cannot_satisfy_a_production_gpu_binding():
    report = hardware_binding(torch.device("cpu"), "RTX 4090")
    assert report["matches"] is False
    assert report["actual_gpu_name"] is None


def _run_all() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    print(f"=== stage_cap2 ({len(tests)} tests) ===")
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


def test_legacy_arm_reproduces_cap1_coefficient_floor():
    """The matched control must stay byte-identical to CAP-EMF-1."""
    legacy = screen_profile("legacy", "local_1000_d0002_fp32", smoke=True)
    assert legacy.objective.coefficient_denominator_floor is None
    assert legacy.objective.resolved_coefficient_floor == (
        legacy.objective.emf_denominator_floor
    )
    assert legacy.objective.sampled_r_floor == 0.01


def test_ordered_arms_raise_only_the_coefficient_floor():
    """The Euler divisor and the 1/t^2 weight must not move with it.

    One constant served three roles. Raising the shared value to 0.10 would
    also have quartered the loss weight on rows with t < 0.10.
    """
    ordered = screen_profile("ordered_uniform", "local_1000_d0002_fp32", smoke=True)
    assert ordered.objective.resolved_coefficient_floor == 0.10
    assert ordered.objective.emf_denominator_floor == 0.02
    assert ordered.objective.sampled_r_floor == 0.0


def test_production_arm_coefficient_tail_is_gated_and_passes():
    """ordered_uniform is the production arm; its tail must be enforced.

    At the inherited 0.02 floor it carried CAP-EMF-1's ill-conditioning
    (P(coefficient > 7) = 4.05%, q99.9 = 44.9) and the gate skipped it.
    """
    from ..sampler_audit import audit_sampler

    report = audit_sampler(
        arm="ordered_uniform", numerical="local_1000_d0002_fp32", count=200_000
    )
    assert "coefficient_tail_control" in report["checks"]
    assert "coefficient_extreme_tail_control" in report["checks"]
    assert report["failed"] == []
    assert report["coefficient_tail_fraction"]["gt_15"] == 0.0
    # The corner the sampler exists to train must survive the clamp.
    assert abs(report["inference_corner_fraction_t95_h90"] - 0.00375) < 0.0006


def test_calibrated_gate_rejects_the_cap1_divergence():
    """The decisive regression: the gate must catch what CAP-EMF-1 hid.

    CAP-EMF-1's gate returned PASS with zero failures on a model carrying 8.4x
    the target effective rank and 6.4x the target diagonal high-frequency
    energy, because every threshold was a floor -- and H7 passed on a
    fabricated 0.0 because the recovery file did not carry the windowed clip
    counters. Both are fixed; this pins them so a refactor cannot revert the
    two-sidedness silently.
    """
    import json
    from pathlib import Path

    from ...stage_cap.config import CAPGateConfig
    from ...stage_cap.diagnostics import capability_gate

    root = Path(__file__).resolve().parents[1]
    calibration = json.loads((root / "gate_calibration.json").read_text())
    gate = CAPGateConfig(**calibration["gate"])
    gate.validate()

    # Measured CAP-EMF-1 endpoint health at its sealed 650k EMA checkpoint.
    cap1 = {
        "second_moment_ratio": 1.1230,
        "centered_variance_ratio": 1.1342,
        "effective_rank_ratio": 8.4417,
        "haar_LL_ratio": 0.9314,
        "haar_LH_ratio": 3.8456,
        "haar_HL_ratio": 3.4975,
        "haar_HH_ratio": 6.3709,
    }
    verdict = capability_gate(
        cap1,
        best_rank_ratio=cap1["effective_rank_ratio"],
        clip_fraction=0.153,  # the real run-wide rate, not the fabricated 0.0
        nonfinite_updates=0,
        inference_forwards=1,
        gate=gate,
    )
    assert verdict["verdict"] == "FAIL", verdict
    for name in (
        "H3c_rank_upper",
        "H4h_haar_HH_upper",
        "H7_clip_fraction",
    ):
        assert name in verdict["failed"], (name, verdict["failed"])
