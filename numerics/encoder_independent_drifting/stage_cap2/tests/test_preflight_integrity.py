"""Fail-closed schema tests for the CAP2 authorization boundary."""

from __future__ import annotations

import copy
from dataclasses import asdict

from ..benchmark import project_runtime
from ..config import SAMPLER_ARMS, numerical_candidate
from ..gate_calibration import _derive_gate, gate_calibration_consistent
from ..numerical_admission import (
    ASSEMBLED_TARGET_COSINE_MIN,
    ASSEMBLED_TARGET_RELATIVE_RMS_MAX,
    AUDIT_SOURCES,
    AUDIT_STRATA,
    GRADIENT_COSINE_MIN,
    GRADIENT_NORM_RATIO_MAX,
    GRADIENT_NORM_RATIO_MIN,
    GRADIENT_RELATIVE_L2_MAX,
    TARGET_COSINE_MIN,
    TARGET_RELATIVE_RMS_MAX,
    _expected_mixed_pairs,
    admission_matrix_complete,
)
from ..preflight import validate_preflight_inputs

CANDIDATE = "local_1000_d0002_fp32"
SOURCE = {"source.py": "a" * 64}
CHECKPOINT = "b" * 64
KID_REFERENCE = {
    "path": "/fixed/cifar10_train_clean_features.npz",
    "sha256": "f" * 64,
    "feature_sha256": "0" * 64,
    "count": 50_000,
    "dimension": 2_048,
    "dtype": "float32",
    "population": "all 50,000 CIFAR-10 train images in dataset-index order",
    "preprocessing": "clean-fid 0.1.35 clean Inception preprocessing",
}
METRIC_PROVENANCE = {
    "packages": {
        "clean-fid": "0.1.35",
        "numpy": "1.26.4",
        "Pillow": "12.0.0",
        "torch": "2.7.1",
        "torchvision": "0.22.1",
    },
    "device": "cuda",
    "gpu_name": "Test GPU",
    "torch_cuda_version": "12.6",
    "cudnn_version": 90501,
    "image_quantization": "fixed",
    "numerical_settings": {"allow_tf32": False},
    "deterministic_algorithms": True,
}


def _hardware() -> dict:
    return {
        "matches": True,
        "actual_gpu_name": "Test GPU",
        "compute_capability": "sm_89",
        "torch_version": "2.7.1+cu126",
        "cuda_runtime": "12.6",
        "cudnn_version": 90501,
        "cublas_workspace_config": ":4096:8",
    }


def _auxiliary() -> dict:
    feature_metrics = {
        "backend": "torchvision Inception-v3 ImageNet1K_V1 pool features",
        "samples_generated": 2_048,
        "samples_reference": 2_048,
        "precision": 0.6,
        "recall": 0.5,
        "pr_f1": 6 / 11,
        "unbiased_kid": 0.02,
        "generated_feature_sha256": "4" * 64,
        "reference_feature_sha256": "5" * 64,
    }
    memorization = {
        "samples_generated": 256,
        "reference_train_images": 50_000,
        "exact_pixel_copy_fraction": 0.0,
        "exact_generated_duplicate_fraction": 0.0,
    }
    return {
        "protocol": "train-only-repository-auxiliary-v1",
        "png_manifest_sha256": "c" * 64,
        "generated_subset": {"count": 2_048, "tensor_sha256": "6" * 64},
        "reference_subset": {
            "count": 2_048,
            "seed": 20_260_832,
            "indices_sha256": "7" * 64,
            "tensor_sha256": "8" * 64,
        },
        "cifar10_train_tensor_sha256": "9" * 64,
        "repository_feature_metrics": feature_metrics,
        "memorization": memorization,
    }


def _standard(*, external: bool) -> dict:
    auxiliary = _auxiliary()
    samples = {
        "count": 50_000,
        "image_size": [32, 32],
        "mode": "RGB",
        "png_manifest_sha256": "c" * 64,
    }
    if external:
        samples["source_citation"] = "public fixed positive-control samples"
        samples["source_provenance"] = "/fixed/positive_control_source.json"
        samples["source_provenance_sha256"] = "9" * 64
    else:
        samples |= {
            "seed": 20_260_804,
            "one_model_call_per_batch": True,
        }
    return {
        "status": "cap-emf-standard-evaluation",
        "checkpoint": None if external else "baseline.pt",
        "checkpoint_sha256": None if external else CHECKPOINT,
        "checkpoint_step": None if external else 650_000,
        "checkpoint_kind": "external-positive-control" if external else "ema",
        "samples": samples,
        "metrics": {
            "backend": "clean-fid",
            "cleanfid_version": "0.1.35",
            "reference": "cifar10/train/32",
            "mode": "clean",
            "feature_count": 50_000,
            "kid_seed": 20_260_831,
            "clean_fid_cifar10_train": 2.0 if external else 100.0,
            "clean_kid_cifar10_train": 0.01 if external else 0.10,
            "metric_batch": 128,
            "metric_workers": 0,
            "generated_features": {
                "path": "/fixed/generated_features.npz",
                "sha256": "d" * 64,
                "feature_sha256": "e" * 64,
                "count": 50_000,
                "dimension": 2_048,
                "dtype": "float32",
                "source_png_manifest_sha256": samples["png_manifest_sha256"],
                "generation_seed": samples.get("seed"),
            },
            "kid_reference": copy.deepcopy(KID_REFERENCE),
        },
        "repository_auxiliary": auxiliary,
        "repository_feature_metrics": auxiliary["repository_feature_metrics"],
        "memorization": auxiliary["memorization"],
        "provenance": copy.deepcopy(METRIC_PROVENANCE),
        "source_sha256": SOURCE,
    }


def _fixture() -> dict:
    repeats = 3
    candidate = numerical_candidate(CANDIDATE)
    quotient = {
        "relative_rms_max": 0.01,
        "cosine_min": 0.999,
    }
    assembled_target = {
        "relative_rms_max": 0.01,
        "cosine_min": 0.9999,
    }
    gradient = {
        "cosine": 0.999,
        "relative_l2": 0.01,
        "norm_ratio": 1.0,
        "finite": True,
    }
    admission_checks = {
        "quotient_relative_rms": True,
        "quotient_cosine": True,
        "assembled_target_relative_rms": True,
        "assembled_target_cosine": True,
        "gradient_cosine": True,
        "gradient_relative_l2": True,
        "gradient_norm_ratio": True,
        "gradient_finite": True,
    }

    def input_batch(source: str, batch: int, seed: int) -> dict:
        return {
            "source": source,
            "batch": batch,
            "seed": seed,
            "sample_ids": list(range(batch)) if source == "cifar10_train" else None,
            "clean_sha256": "1" * 64,
            "noise_sha256": "2" * 64,
            "state_sha256": "3" * 64,
        }

    strata = []
    for repeat in range(repeats):
        for source in AUDIT_SOURCES:
            for name, t, r in AUDIT_STRATA:
                strata.append(
                    {
                        "repeat": repeat,
                        "stratum": name,
                        "t": t,
                        "r": r,
                        "h": t - r,
                        "delta": candidate.delta,
                        "evaluation_mode": candidate.stopped_evaluation,
                        "input_batch": input_batch(source, 4, repeat),
                        "quotient": copy.deepcopy(quotient),
                        "assembled_target": copy.deepcopy(assembled_target),
                        "gradient": copy.deepcopy(gradient),
                        "admission_checks": copy.deepcopy(admission_checks),
                        "verdict": "PASS",
                    }
                )
    protocol_checks = {
        "checkpoint_identity": True,
        "cuda": True,
        "hardware_matches": True,
        "parameter_gradient_checked": True,
        "audit_batch_exactly_four": True,
        "minimum_three_repeats": True,
        "deterministic_algorithms": True,
        "all_strata_pass": True,
        "all_mixed_gradient_batches_pass": True,
        "all_production_shape_batches_pass": True,
    }
    mixed_gradient = [
        {
            "repeat": repeat,
            "input_batch": {
                **input_batch(source, 4, repeat),
                "time_pairs": _expected_mixed_pairs(4),
            },
            "delta": candidate.delta,
            "evaluation_mode": candidate.stopped_evaluation,
            "quotient": copy.deepcopy(quotient),
            "assembled_target": copy.deepcopy(assembled_target),
            "gradient": copy.deepcopy(gradient),
            "diagonal_rows": 2,
            "active_rows": 2,
            "admission_checks": copy.deepcopy(admission_checks),
            "verdict": "PASS",
        }
        for repeat in range(repeats)
        for source in AUDIT_SOURCES
    ]
    production_shape = [
        {
            "repeat": repeat,
            "input_batch": {
                **input_batch(source, 16, repeat),
                "time_pairs": _expected_mixed_pairs(16),
            },
            "delta": candidate.delta,
            "evaluation_mode": candidate.stopped_evaluation,
            "quotient": copy.deepcopy(quotient),
            "assembled_target": copy.deepcopy(assembled_target),
            "gradient": copy.deepcopy(gradient),
            "diagonal_rows": 8,
            "active_rows": 8,
            "admission_checks": copy.deepcopy(admission_checks),
            "verdict": "PASS",
        }
        for repeat in range(repeats)
        for source in AUDIT_SOURCES
    ]
    numerical = {
        "decision": "GO",
        "candidate": asdict(candidate),
        "checkpoint_sha256": CHECKPOINT,
        "checkpoint_step": 650_000,
        "checkpoint_identity": {
            "valid": True,
            "stage": "cap-emf-1",
            "kind": "ema",
        },
        "cuda_admission": True,
        "gradient_checked": True,
        "hardware": _hardware(),
        "batch_per_stratum": 4,
        "repeats": repeats,
        "design": {
            "sources": list(AUDIT_SOURCES),
            "strata": [
                {"name": name, "t": t, "r": r, "h": t - r}
                for name, t, r in AUDIT_STRATA
            ],
            "total_stratum_batches": len(strata),
        },
        "thresholds": {
            "quotient_cosine_min": TARGET_COSINE_MIN,
            "quotient_relative_rms_max": TARGET_RELATIVE_RMS_MAX,
            "assembled_target_cosine_min": ASSEMBLED_TARGET_COSINE_MIN,
            "assembled_target_relative_rms_max": ASSEMBLED_TARGET_RELATIVE_RMS_MAX,
            "gradient_cosine_min": GRADIENT_COSINE_MIN,
            "gradient_relative_l2_max": GRADIENT_RELATIVE_L2_MAX,
            "gradient_norm_ratio": [
                GRADIENT_NORM_RATIO_MIN,
                GRADIENT_NORM_RATIO_MAX,
            ],
        },
        "strata": strata,
        "mixed_gradient": mixed_gradient,
        "production_shape": production_shape,
        "protocol_checks": protocol_checks,
        "production_numerical_mode": {
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "graded_forward_tf32": True,
            "graded_backward_tf32": True,
            "exact_jvp_tf32": False,
            "stopped_path_tf32": False,
        },
        "source_sha256": SOURCE,
    }
    sampler_checks = {
        "finite": True,
        "nonnegative_r": True,
        "ordered": True,
        "interval_valid": True,
        "diagonal_fraction": True,
        "diagonal_exact": True,
        "diagonal_base_law": True,
    }
    samplers = {
        "decision": "GO",
        "numerical_candidate": CANDIDATE,
        "count_per_arm": 2_000_000,
        "arms": {
            arm: {
                "verdict": "PASS",
                "numerical_candidate": CANDIDATE,
                "count": 2_000_000,
                "diagonal_sampling": "fixed_count_first_draw",
                "checks": sampler_checks,
            }
            for arm in SAMPLER_ARMS
        },
        "source_sha256": SOURCE,
    }
    gate_names = (
        "second_moment_ratio",
        "centered_variance_ratio",
        "effective_rank_ratio",
        "haar_LL_ratio",
        "haar_LH_ratio",
        "haar_HL_ratio",
        "haar_HH_ratio",
    )
    gate_records = [
        {
            "samples": 2_048,
            **{name: 0.9 + 0.2 * index / 11 for name in gate_names},
            "raw_saturation_fraction": 0.001 * index,
        }
        for index in range(12)
    ]
    gate_lower, gate_upper, gate = _derive_gate(gate_records)
    calibration = {
        "status": "cap-emf2-gate-calibration",
        "decision": "GO",
        "samples_per_subset": 2_048,
        "repeats": 12,
        "globally_disjoint": True,
        "allocation_sha256": [
            {
                "left_sha256": f"{2 * index:064x}",
                "right_sha256": f"{2 * index + 1:064x}",
            }
            for index in range(12)
        ],
        "records": gate_records,
        "empirical_lower": gate_lower,
        "empirical_upper": gate_upper,
        "gate": asdict(gate),
        "source_sha256": SOURCE,
    }
    benchmark_checks = {
        "hardware_bound": True,
        "completed": True,
        "finite": True,
        "checkpoint_written": True,
        "snapshot_written": True,
        "recovery_written": True,
        "resume_rehearsed": True,
        "durable_mirror_roundtrip": True,
        "durable_recoveries_committed": True,
        "parameter_ceiling": True,
    }
    benchmark_timing = {
        "non_io_seconds_per_update": 0.5,
        "raw_full_loop_seconds_per_update": 0.7,
        "recovery_io_seconds": 0.1,
        "checkpoint_io_seconds": 0.1,
        "snapshot_io_seconds": 0.1,
        "ordinary_health_seconds": 0.1,
        "checkpoint_health_seconds": 0.2,
    }
    resume_rehearsal = {
        "split_step": 1_999,
        "final_step": 2_000,
        "resumed_updates": 1,
        "first_device": "cuda:0",
        "second_device": "cuda:0",
        "resume_message": "resumed CAP-EMF-1 from update 1999",
        "before_recovery_sha256": "1" * 64,
        "after_recovery_sha256": "2" * 64,
        "before_resume": {
            "completed_updates": 1_999,
            "optimizer_updates": 1_999,
            "ema_updates": 1_999,
            "nonfinite_updates": 0,
            "optimizer_steps": {"count": 10, "minimum": 1_999, "maximum": 1_999},
        },
        "after_resume": {
            "completed_updates": 2_000,
            "optimizer_updates": 2_000,
            "ema_updates": 2_000,
            "nonfinite_updates": 0,
            "optimizer_steps": {"count": 10, "minimum": 2_000, "maximum": 2_000},
        },
    }
    benchmark = {
        "decision": "GO",
        "arm": "ordered_uniform",
        "numerical": CANDIDATE,
        "steps": 2_000,
        "deterministic_algorithms": True,
        "unit_seed": 0,
        "micro_batch": 16,
        "accumulation_steps": 4,
        "effective_batch": 64,
        "hourly_rate": 1.0,
        "device": {"allow_tf32": True},
        "precision": {"matmul_tf32": True, "cudnn_tf32": True},
        "checks": benchmark_checks,
        "parameter_count": 37_000_000,
        "peak_memory_bytes": 1,
        "recovery_bytes": 600 * 1024**2,
        "checkpoint_artifact_bytes": {
            "raw": 150 * 1024**2,
            "ema": 150 * 1024**2,
        },
        "snapshot": {"bytes": 150 * 1024**2},
        "objective_sample_evaluations": 1,
        "objective_forward_calls": 1,
        "projection_method": "cadence-adjusted measured event costs",
        **benchmark_timing,
        "ordinary_health_samples": 512,
        "checkpoint_health_samples": 2_048,
        "resume_rehearsal": resume_rehearsal,
        "durable_mirror": {
            "attestation": {
                "instance_independent": True,
                "storage_id": "test-persistent-volume",
                "artifact_sha256": "3" * 64,
            },
            "live_roundtrip_probe": {
                "roundtrip_verified": True,
                "probe_removed": True,
                "storage_id": "test-persistent-volume",
                "attestation_sha256": "3" * 64,
            },
            "recovery_commits": [
                {
                    "recovery_step": 1_999,
                    "sha256": resume_rehearsal["before_recovery_sha256"],
                },
                {
                    "recovery_step": 2_000,
                    "sha256": resume_rehearsal["after_recovery_sha256"],
                },
            ],
            "synchronous_event_costs_included": True,
        },
        "projections": {
            str(updates): project_runtime(
                updates=updates,
                recovery_every=5_000,
                snapshot_every=25_000,
                health_every=2_000,
                recovery_event_seconds=benchmark_timing["recovery_io_seconds"],
                snapshot_event_seconds=benchmark_timing["snapshot_io_seconds"],
                checkpoint_pair_seconds=benchmark_timing["checkpoint_io_seconds"],
                ordinary_health_event_seconds=benchmark_timing[
                    "ordinary_health_seconds"
                ],
                checkpoint_health_event_seconds=benchmark_timing[
                    "checkpoint_health_seconds"
                ],
                hourly_rate=1.0,
                non_io_seconds_per_update=benchmark_timing["non_io_seconds_per_update"],
                raw_upper_seconds_per_update=benchmark_timing[
                    "raw_full_loop_seconds_per_update"
                ],
            )
            for updates in (
                50_000,
                150_000,
                300_000,
                500_000,
                650_000,
                750_000,
            )
        },
        "hardware": _hardware(),
        "source_sha256": SOURCE,
    }
    metric_calibration = {
        "decision": "COMPLETE",
        "seed": 20_260_806,
        "samples_per_side": 25_000,
        "left_indices_sha256": "d" * 64,
        "right_indices_sha256": "e" * 64,
        "left": {"count": 25_000},
        "right": {"count": 25_000},
        "metrics": {
            "backend": "clean-fid",
            "cleanfid_version": "0.1.35",
            "kid_seed": 20_260_831,
            "metric_batch": 128,
            "metric_workers": 0,
            "kid_reference": copy.deepcopy(KID_REFERENCE),
            "direct_disjoint_pair": {"clean_fid": 1.0, "clean_kid": 0.001},
            "matched_published_train_reference": {
                "left": {
                    "clean_fid_cifar10_train": 1.0,
                    "clean_kid_cifar10_train": 0.001,
                },
                "right": {
                    "clean_fid_cifar10_train": 1.1,
                    "clean_kid_cifar10_train": 0.002,
                },
            },
        },
        "provenance": copy.deepcopy(METRIC_PROVENANCE),
        "source_sha256": SOURCE,
    }
    forensics = {
        "decision": "COMPLETE",
        "samples": 2_048,
        "grid_samples": 256,
        "checkpoint_kind": "ema",
        "checkpoint_step": 650_000,
        "checkpoint_sha256": CHECKPOINT,
        "device": {"allow_tf32": False},
        "checks": {"complete": True},
        "hardware": _hardware(),
        "source_sha256": SOURCE,
    }
    return {
        "numerical": numerical,
        "samplers": samplers,
        "calibration": calibration,
        "benchmark": benchmark,
        "baseline": _standard(external=False),
        "positive_control": _standard(external=True),
        "metric_calibration": metric_calibration,
        "forensics": forensics,
        "live_sources": SOURCE,
        "installed_cleanfid_version": "0.1.35",
    }


def test_admission_matrix_recomputes_saved_threshold_decisions():
    fixture = _fixture()["numerical"]
    assert admission_matrix_complete(fixture)
    fixture["strata"][0]["quotient"]["cosine_min"] = 0.0
    # Editing the metrics while leaving the serialized PASS bits untouched
    # must not survive the authorization boundary.
    assert not admission_matrix_complete(fixture)


def test_gate_calibration_recomputes_bounds_from_saved_records():
    calibration = _fixture()["calibration"]
    assert gate_calibration_consistent(calibration)
    calibration["gate"]["maximum_haar_HH_ratio"] += 0.1
    assert not gate_calibration_consistent(calibration)


def _validate(fixture: dict) -> tuple[str, dict[str, bool]]:
    return validate_preflight_inputs(**fixture)


def test_complete_schema_can_pass() -> None:
    decision, checks = _validate(_fixture())
    assert decision == "GO", [name for name, passed in checks.items() if not passed]


def _as_off_scale(fixture: dict, *, audited_scale: float = 100.0) -> dict:
    """Point the fixture at a scale-100 candidate audited on its own checkpoint."""
    numerical = fixture["numerical"]
    numerical["candidate"] = dict(numerical["candidate"], embedding_scale=100.0)
    numerical["checkpoint_embedding_scale"] = audited_scale
    numerical["checkpoint_step"] = 50_000
    numerical["checkpoint_identity"] = {
        "valid": True,
        "stage": "cap-emf-2-candidate-audit",
        "kind": "raw",
    }
    return fixture


def test_off_scale_candidate_is_admissible_against_its_own_audit() -> None:
    """The baseline weights cannot audit a candidate at another scale.

    ``run_admission`` refuses a scale mismatch, so demanding cap-emf-1/ema/650000
    unconditionally left an off-scale candidate admissible nowhere while the
    candidate registry simultaneously required a short trained-model audit for
    exactly that case.
    """
    # Scoped to the identity rule. The fixture's strata are generated for the
    # original candidate, so rewriting the candidate's scale in place leaves the
    # full-matrix check inconsistent; regenerating the whole matrix would test
    # the fixture builder rather than this rule.
    _decision, checks = _validate(_as_off_scale(_fixture()))
    assert checks["numerical_checkpoint_identity"] is True


def test_off_scale_candidate_must_be_audited_at_its_own_scale() -> None:
    """The replacement requirement has to actually bind.

    Dropping the fixed provenance is only safe because the property it stood in
    for -- that the audited weights carry the candidate's embedding scale -- is
    checked directly. An audit at the wrong scale must still be refused.
    """
    decision, checks = _validate(_as_off_scale(_fixture(), audited_scale=1_000.0))
    assert checks["numerical_checkpoint_identity"] is False
    assert decision == "NO_GO"


def test_off_scale_candidate_cannot_present_a_screen_checkpoint() -> None:
    """A screen unit is not a substitute for the prescribed audit."""
    fixture = _as_off_scale(_fixture())
    fixture["numerical"]["checkpoint_identity"]["stage"] = "cap-emf-2-screen"
    decision, checks = _validate(fixture)
    assert checks["numerical_checkpoint_identity"] is False
    assert decision == "NO_GO"


def test_baseline_scale_candidate_still_requires_the_preserved_checkpoint() -> None:
    """The amendment must not loosen the ordinary path.

    A scale-1000 candidate is still admissible only against cap-emf-1 EMA
    weights at the baseline step.
    """
    fixture = _fixture()
    fixture["numerical"]["checkpoint_identity"] = {
        "valid": True,
        "stage": "cap-emf-2-candidate-audit",
        "kind": "raw",
    }
    decision, checks = _validate(fixture)
    assert checks["numerical_checkpoint_identity"] is False
    assert decision == "NO_GO"


def test_anonymous_positive_control_is_rejected() -> None:
    fixture = _fixture()
    del fixture["positive_control"]["samples"]["source_citation"]
    decision, checks = _validate(fixture)
    assert decision == "NO_GO"
    assert checks["positive_control_standard_valid"] is False


def test_truncated_numerical_matrix_is_rejected() -> None:
    fixture = _fixture()
    fixture["numerical"]["strata"].pop()
    decision, checks = _validate(fixture)
    assert decision == "NO_GO"
    assert checks["numerical_full_matrix"] is False


def test_legacy_biased_diagonal_semantics_are_rejected() -> None:
    fixture = _fixture()
    fixture["samplers"]["arms"]["legacy"]["diagonal_sampling"] = "legacy_bernoulli"
    decision, checks = _validate(fixture)
    assert decision == "NO_GO"
    assert checks["samplers_go_and_match"] is False


def test_environment_or_dependency_drift_is_rejected() -> None:
    fixture = _fixture()
    fixture["benchmark"]["hardware"]["cuda_runtime"] = "12.7"
    fixture["installed_cleanfid_version"] = "0.1.36"
    decision, checks = _validate(fixture)
    assert decision == "NO_GO"
    assert checks["admission_benchmark_same_environment"] is False
    assert checks["installed_cleanfid_pinned"] is False


def test_missing_post_reload_ema_update_is_rejected() -> None:
    fixture = _fixture()
    fixture["benchmark"]["resume_rehearsal"]["after_resume"]["ema_updates"] = 1_999
    decision, checks = _validate(fixture)
    assert decision == "NO_GO"
    assert checks["benchmark_go_and_complete"] is False


def test_missing_or_unattested_durable_mirror_is_rejected() -> None:
    fixture = _fixture()
    fixture["benchmark"]["durable_mirror"]["attestation"]["instance_independent"] = (
        False
    )
    decision, checks = _validate(fixture)
    assert decision == "NO_GO"
    assert checks["benchmark_go_and_complete"] is False


def test_mismatched_kid_reference_population_is_rejected() -> None:
    fixture = _fixture()
    fixture["positive_control"]["metrics"]["kid_reference"]["sha256"] = "1" * 64
    decision, checks = _validate(fixture)
    assert decision == "NO_GO"
    assert checks["shared_kid_reference_population"] is False


def test_mismatched_metric_environment_is_rejected() -> None:
    fixture = _fixture()
    fixture["positive_control"]["provenance"]["packages"]["torch"] = "2.8.0"
    decision, checks = _validate(fixture)
    assert decision == "NO_GO"
    assert checks["shared_metric_environment"] is False

    fixture = _fixture()
    fixture["metric_calibration"]["metrics"]["metric_workers"] = 4
    decision, checks = _validate(fixture)
    assert decision == "NO_GO"
    assert checks["shared_metric_environment"] is False


def test_inputs_are_not_mutated_by_validation() -> None:
    fixture = _fixture()
    before = copy.deepcopy(fixture)
    _validate(fixture)
    assert fixture == before


def _run_all() -> int:
    tests = [
        (name, value)
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    failures = 0
    for name, function in tests:
        try:
            function()
        except Exception as error:  # noqa: BLE001 - compact standalone runner
            failures += 1
            print(f"[FAIL] {name}: {type(error).__name__}: {error}")
        else:
            print(f"[PASS] {name}")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
