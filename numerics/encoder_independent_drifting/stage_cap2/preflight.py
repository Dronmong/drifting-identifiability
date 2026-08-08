"""Freeze CAP2 sources after numerical, sampler, gate, and cost admission."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path

import torch

from ..stage_cap.training import train_cap_unit
from .artifacts import (
    DEFAULT_PREFLIGHT,
    PROTOCOL,
    assert_unused,
    file_sha256,
    profile_payload,
    source_manifest,
    verify_json,
    write_json_atomic,
)
from .benchmark import project_runtime
from .config import SAMPLER_ARMS, apply_calibrated_gate, screen_profile
from .gate_calibration import gate_calibration_consistent
from .numerical_admission import (
    AUDIT_BATCH,
    AUDIT_SOURCES,
    AUDIT_STRATA,
    MINIMUM_REPEATS,
    PRODUCTION_MICROBATCH,
    admission_matrix_complete,
)
from .standard_metrics import (
    DEFAULT_GENERATION_SEED,
    DEFAULT_KID_SEED,
    DEFAULT_SAMPLE_COUNT,
    REPOSITORY_AUXILIARY_PROTOCOL,
    REPOSITORY_FEATURE_SAMPLES,
    REPOSITORY_MEMORIZATION_SAMPLES,
    REPOSITORY_REFERENCE_SEED,
)

CLEANFID_VERSION = "0.1.35"
CALIBRATION_SAMPLES_PER_SIDE = 25_000
GATE_SAMPLES_PER_SUBSET = 2_048
GATE_REPEATS = 12
SAMPLER_DRAWS_PER_ARM = 2_000_000
BASELINE_CHECKPOINT_STEP = 650_000
KID_REFERENCE_POPULATION = (
    "all 50,000 CIFAR-10 train images in dataset-index order"
)
KID_REFERENCE_PREPROCESSING = "clean-fid 0.1.35 clean Inception preprocessing"


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _hash64(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _all_true(mapping: object, required: tuple[str, ...] = ()) -> bool:
    return (
        isinstance(mapping, dict)
        and bool(mapping)
        and set(required).issubset(mapping)
        and all(value is True for value in mapping.values())
    )


def _same_hardware(left: object, right: object) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    fields = (
        "actual_gpu_name",
        "compute_capability",
        "torch_version",
        "cuda_runtime",
        "cudnn_version",
        "cublas_workspace_config",
    )
    return (
        left.get("matches") is True
        and right.get("matches") is True
        and all(left.get(field) == right.get(field) for field in fields)
    )


def _clean_metrics_valid(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    kid_reference = payload.get("kid_reference")
    return (
        payload.get("backend") == "clean-fid"
        and payload.get("cleanfid_version") == CLEANFID_VERSION
        and payload.get("reference") == "cifar10/train/32"
        and payload.get("mode") == "clean"
        and int(payload.get("feature_count", -1)) == DEFAULT_SAMPLE_COUNT
        and int(payload.get("kid_seed", -1)) == DEFAULT_KID_SEED
        and _finite(payload.get("clean_fid_cifar10_train"))
        and float(payload["clean_fid_cifar10_train"]) >= 0.0
        and _finite(payload.get("clean_kid_cifar10_train"))
        and _kid_reference_valid(kid_reference)
    )


def _kid_reference_valid(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        isinstance(payload.get("path"), str)
        and bool(payload["path"].strip())
        and _hash64(payload.get("sha256"))
        and _hash64(payload.get("feature_sha256"))
        and int(payload.get("count", -1)) == DEFAULT_SAMPLE_COUNT
        and int(payload.get("dimension", -1)) == 2_048
        and payload.get("dtype") in {"float32", "float64"}
        and payload.get("population") == KID_REFERENCE_POPULATION
        and payload.get("preprocessing") == KID_REFERENCE_PREPROCESSING
    )


def _same_kid_reference(*payloads: object) -> bool:
    if not all(_kid_reference_valid(payload) for payload in payloads):
        return False
    fields = (
        "sha256",
        "feature_sha256",
        "count",
        "dimension",
        "dtype",
        "population",
        "preprocessing",
    )
    first = payloads[0]
    assert isinstance(first, dict)
    return all(
        isinstance(payload, dict)
        and all(payload.get(field) == first.get(field) for field in fields)
        for payload in payloads[1:]
    )


def _same_metric_environment(*artifacts: object) -> bool:
    """Require every pre-run metric artifact to share one numerical stack."""

    records = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            return False
        provenance = artifact.get("provenance")
        if not isinstance(provenance, dict):
            return False
        packages = provenance.get("packages")
        if not isinstance(packages, dict) or any(
            not isinstance(packages.get(name), str)
            for name in ("clean-fid", "numpy", "Pillow", "torch", "torchvision")
        ):
            return False
        record = {
            "packages": {
                name: packages[name]
                for name in ("clean-fid", "numpy", "Pillow", "torch", "torchvision")
            },
            "device": provenance.get("device"),
            "gpu_name": provenance.get("gpu_name"),
            "torch_cuda_version": provenance.get("torch_cuda_version"),
            "cudnn_version": provenance.get("cudnn_version"),
            "image_quantization": provenance.get("image_quantization"),
            "numerical_settings": provenance.get("numerical_settings"),
            "deterministic_algorithms": provenance.get("deterministic_algorithms"),
        }
        if (
            record["deterministic_algorithms"] is not True
            or not isinstance(record["image_quantization"], str)
            or not record["image_quantization"]
        ):
            return False
        records.append(record)
    return bool(records) and all(record == records[0] for record in records[1:])


def _auxiliary_valid(payload: object, samples: object) -> bool:
    if not isinstance(payload, dict) or not isinstance(samples, dict):
        return False
    metrics = payload.get("repository_feature_metrics", {})
    memorization = payload.get("memorization", {})
    generated_subset = payload.get("generated_subset", {})
    reference_subset = payload.get("reference_subset", {})
    finite_metrics = (
        "precision",
        "recall",
        "pr_f1",
        "unbiased_kid",
    )
    return (
        payload.get("protocol") == REPOSITORY_AUXILIARY_PROTOCOL
        and payload.get("png_manifest_sha256") == samples.get("png_manifest_sha256")
        and int(payload.get("generated_subset", {}).get("count", -1))
        == REPOSITORY_FEATURE_SAMPLES
        and int(payload.get("reference_subset", {}).get("count", -1))
        == REPOSITORY_FEATURE_SAMPLES
        and metrics.get("backend")
        == "torchvision Inception-v3 ImageNet1K_V1 pool features"
        and int(metrics.get("samples_generated", -1)) == REPOSITORY_FEATURE_SAMPLES
        and int(metrics.get("samples_reference", -1)) == REPOSITORY_FEATURE_SAMPLES
        and all(
            _hash64(value)
            for value in (
                payload.get("cifar10_train_tensor_sha256"),
                generated_subset.get("tensor_sha256"),
                reference_subset.get("indices_sha256"),
                reference_subset.get("tensor_sha256"),
                metrics.get("generated_feature_sha256"),
                metrics.get("reference_feature_sha256"),
            )
        )
        and int(reference_subset.get("seed", -1)) == REPOSITORY_REFERENCE_SEED
        and all(_finite(metrics.get(name)) for name in finite_metrics)
        and 0.0 <= float(metrics["precision"]) <= 1.0
        and 0.0 <= float(metrics["recall"]) <= 1.0
        and 0.0 <= float(metrics["pr_f1"]) <= 1.0
        and int(memorization.get("samples_generated", -1))
        == REPOSITORY_MEMORIZATION_SAMPLES
        and int(memorization.get("reference_train_images", -1)) == 50_000
        and _finite(memorization.get("exact_pixel_copy_fraction"))
        and _finite(memorization.get("exact_generated_duplicate_fraction"))
    )


def _standard_evaluation_valid(payload: object, *, external: bool) -> bool:
    if not isinstance(payload, dict):
        return False
    samples = payload.get("samples", {})
    auxiliary = payload.get("repository_auxiliary", {})
    checkpoint_ok = (
        (
            payload.get("checkpoint") is None
            and payload.get("checkpoint_sha256") is None
            and payload.get("checkpoint_kind") == "external-positive-control"
            and isinstance(samples.get("source_citation"), str)
            and bool(samples["source_citation"].strip())
            and isinstance(samples.get("source_provenance"), str)
            and bool(samples["source_provenance"].strip())
            and _hash64(samples.get("source_provenance_sha256"))
        )
        if external
        else (
            isinstance(payload.get("checkpoint"), str)
            and _hash64(payload.get("checkpoint_sha256"))
            and payload.get("checkpoint_kind") == "ema"
            and int(payload.get("checkpoint_step", -1)) == BASELINE_CHECKPOINT_STEP
            and int(samples.get("seed", -1)) == DEFAULT_GENERATION_SEED
            and samples.get("one_model_call_per_batch") is True
        )
    )
    provenance = payload.get("provenance", {})
    return (
        payload.get("status") == "cap-emf-standard-evaluation"
        and checkpoint_ok
        and int(samples.get("count", -1)) == DEFAULT_SAMPLE_COUNT
        and list(samples.get("image_size", [])) == [32, 32]
        and samples.get("mode") == "RGB"
        and _hash64(samples.get("png_manifest_sha256"))
        and _clean_metrics_valid(payload.get("metrics"))
        and _auxiliary_valid(auxiliary, samples)
        and payload.get("repository_feature_metrics")
        == auxiliary.get("repository_feature_metrics")
        and payload.get("memorization") == auxiliary.get("memorization")
        and provenance.get("deterministic_algorithms") is True
        and provenance.get("packages", {}).get("clean-fid") == CLEANFID_VERSION
    )


def _benchmark_projection_consistent(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    try:
        frozen = screen_profile(
            payload["arm"], payload["numerical"], updates=50_000
        ).train
        expected = {
            str(updates): project_runtime(
                updates=updates,
                recovery_every=frozen.recovery_every,
                snapshot_every=frozen.snapshot_every,
                health_every=frozen.health_every,
                non_io_seconds_per_update=float(payload["non_io_seconds_per_update"]),
                raw_upper_seconds_per_update=float(
                    payload["raw_full_loop_seconds_per_update"]
                ),
                recovery_event_seconds=float(payload["recovery_io_seconds"]),
                snapshot_event_seconds=float(payload["snapshot_io_seconds"]),
                checkpoint_pair_seconds=float(payload["checkpoint_io_seconds"]),
                ordinary_health_event_seconds=float(
                    payload["ordinary_health_seconds"]
                ),
                checkpoint_health_event_seconds=float(
                    payload["checkpoint_health_seconds"]
                ),
                hourly_rate=float(payload["hourly_rate"]),
            )
            for updates in (50_000, 150_000, 300_000)
        }
    except (KeyError, TypeError, ValueError, RuntimeError):
        return False
    return payload.get("projections") == expected


def validate_preflight_inputs(
    *,
    numerical: dict,
    samplers: dict,
    calibration: dict,
    benchmark: dict,
    baseline: dict,
    positive_control: dict,
    metric_calibration: dict,
    forensics: dict,
    live_sources: dict[str, str],
    installed_cleanfid_version: str | None,
) -> tuple[str, dict[str, bool]]:
    """Validate complete artifact schemas; status strings alone are insufficient."""
    candidate = numerical.get("candidate", {}).get("name")
    expected_strata = [
        {"name": name, "t": t, "r": r, "h": t - r} for name, t, r in AUDIT_STRATA
    ]
    expected_batches = (
        len(AUDIT_SOURCES) * len(AUDIT_STRATA) * int(numerical.get("repeats", 0))
    )
    expected_result_keys = {
        (repeat, source, name)
        for repeat in range(int(numerical.get("repeats", 0)))
        for source in AUDIT_SOURCES
        for name, _t, _r in AUDIT_STRATA
    }
    numerical_results = numerical.get("strata", [])
    mixed_results = numerical.get("mixed_gradient", [])
    production_shape_results = numerical.get("production_shape", [])
    numerical_mode = numerical.get("production_numerical_mode", {})
    numerical_identity = numerical.get("checkpoint_identity", {})
    sampler_arms = samplers.get("arms", {})
    required_sampler_checks = (
        "finite",
        "nonnegative_r",
        "ordered",
        "interval_valid",
        "diagonal_fraction",
        "diagonal_exact",
        "diagonal_base_law",
    )
    gate_fields = {
        "second_moment_ratio",
        "centered_variance_ratio",
        "effective_rank_ratio",
        "haar_LL_ratio",
        "haar_LH_ratio",
        "haar_HL_ratio",
        "haar_HH_ratio",
    }
    benchmark_checks = benchmark.get("checks", {})
    calibration_metrics = metric_calibration.get("metrics", {})
    direct = calibration_metrics.get("direct_disjoint_pair", {})
    matched = calibration_metrics.get("matched_published_train_reference", {})
    standard_metric_names = (
        "clean_fid_cifar10_train",
        "clean_kid_cifar10_train",
    )
    baseline_metrics = baseline.get("metrics", {})
    control_metrics = positive_control.get("metrics", {})
    baseline_kid_reference = baseline_metrics.get("kid_reference")
    control_kid_reference = control_metrics.get("kid_reference")
    calibration_kid_reference = calibration_metrics.get("kid_reference")
    input_artifacts = (
        numerical,
        samplers,
        calibration,
        benchmark,
        baseline,
        positive_control,
        metric_calibration,
        forensics,
    )
    actual_result_keys = {
        (
            int(result.get("repeat", -1)),
            result.get("input_batch", {}).get("source"),
            result.get("stratum"),
        )
        for result in numerical_results
    }
    numerical_rows_valid = all(
        result.get("verdict") == "PASS"
        and _all_true(result.get("admission_checks"))
        and int(result.get("input_batch", {}).get("batch", -1)) == AUDIT_BATCH
        and isinstance(result.get("input_batch", {}).get("seed"), int)
        and all(
            isinstance(result.get("input_batch", {}).get(name), str)
            and len(result["input_batch"][name]) == 64
            for name in ("clean_sha256", "noise_sha256", "state_sha256")
        )
        for result in numerical_results
    )
    expected_mixed_keys = {
        (repeat, source)
        for repeat in range(int(numerical.get("repeats", 0)))
        for source in AUDIT_SOURCES
    }

    def mixed_valid(rows: object, *, expected_batch: int, gradient: bool) -> bool:
        return (
            isinstance(rows, list)
            and len(rows) == len(expected_mixed_keys)
            and {
                (
                    int(row.get("repeat", -1)),
                    row.get("input_batch", {}).get("source"),
                )
                for row in rows
            }
            == expected_mixed_keys
            and all(
                row.get("verdict") == "PASS"
                and _all_true(row.get("admission_checks"))
                and int(row.get("input_batch", {}).get("batch", -1)) == expected_batch
                and len(row.get("input_batch", {}).get("time_pairs", []))
                == expected_batch
                and (isinstance(row.get("gradient"), dict)) == gradient
                for row in rows
            )
        )

    allocation_hashes = [
        allocation.get(side)
        for allocation in calibration.get("allocation_sha256", [])
        if isinstance(allocation, dict)
        for side in ("left_sha256", "right_sha256")
    ]
    checks = {
        "numerical_go": numerical.get("decision") == "GO",
        "numerical_candidate_named": isinstance(candidate, str) and bool(candidate),
        "numerical_checkpoint_identity": (
            numerical_identity.get("valid") is True
            and numerical_identity.get("stage") == "cap-emf-1"
            and numerical_identity.get("kind") == "ema"
            and int(numerical.get("checkpoint_step", -1)) == BASELINE_CHECKPOINT_STEP
        ),
        "numerical_cuda_and_hardware": (
            numerical.get("cuda_admission") is True
            and numerical.get("gradient_checked") is True
            and numerical.get("hardware", {}).get("matches") is True
        ),
        "numerical_full_matrix": (
            admission_matrix_complete(numerical)
            and
            int(numerical.get("batch_per_stratum", -1)) == AUDIT_BATCH
            and int(numerical.get("repeats", -1)) >= MINIMUM_REPEATS
            and numerical.get("design", {}).get("sources") == list(AUDIT_SOURCES)
            and numerical.get("design", {}).get("strata") == expected_strata
            and int(numerical.get("design", {}).get("total_stratum_batches", -1))
            == expected_batches
            and isinstance(numerical_results, list)
            and len(numerical_results) == expected_batches
            and actual_result_keys == expected_result_keys
            and numerical_rows_valid
            and mixed_valid(
                mixed_results, expected_batch=AUDIT_BATCH, gradient=True
            )
            and mixed_valid(
                production_shape_results,
                expected_batch=PRODUCTION_MICROBATCH,
                gradient=True,
            )
        ),
        "numerical_protocol_checks": _all_true(
            numerical.get("protocol_checks"),
            (
                "checkpoint_identity",
                "cuda",
                "hardware_matches",
                "parameter_gradient_checked",
                "audit_batch_exactly_four",
                "minimum_three_repeats",
                "deterministic_algorithms",
                "all_strata_pass",
                "all_mixed_gradient_batches_pass",
                "all_production_shape_batches_pass",
            ),
        ),
        "numerical_precision_boundary": (
            numerical_mode.get("deterministic_algorithms") is True
            and numerical_mode.get("deterministic_warn_only") is False
            and numerical_mode.get("cudnn_benchmark") is False
            and numerical_mode.get("graded_forward_tf32") is True
            and numerical_mode.get("graded_backward_tf32") is True
            and numerical_mode.get("exact_jvp_tf32") is False
            and numerical_mode.get("stopped_path_tf32") is False
        ),
        "samplers_go_and_match": (
            samplers.get("decision") == "GO"
            and samplers.get("numerical_candidate") == candidate
            and int(samplers.get("count_per_arm", -1)) == SAMPLER_DRAWS_PER_ARM
            and set(sampler_arms) == set(SAMPLER_ARMS)
            and all(
                result.get("verdict") == "PASS"
                and result.get("numerical_candidate") == candidate
                and int(result.get("count", -1)) == SAMPLER_DRAWS_PER_ARM
                and result.get("diagonal_sampling") == "fixed_count_first_draw"
                and _all_true(result.get("checks"), required_sampler_checks)
                for result in sampler_arms.values()
            )
        ),
        "gate_calibration_complete": (
            gate_calibration_consistent(calibration)
            and
            calibration.get("decision") == "GO"
            and int(calibration.get("samples_per_subset", -1))
            == GATE_SAMPLES_PER_SUBSET
            and int(calibration.get("repeats", -1)) == GATE_REPEATS
            and calibration.get("globally_disjoint") is True
            and len(calibration.get("allocation_sha256", [])) == GATE_REPEATS
            and len(allocation_hashes) == 2 * GATE_REPEATS
            and len(set(allocation_hashes)) == 2 * GATE_REPEATS
            and all(_hash64(value) for value in allocation_hashes)
            and len(calibration.get("records", [])) == GATE_REPEATS
            and all(
                int(record.get("samples", -1)) == GATE_SAMPLES_PER_SUBSET
                for record in calibration.get("records", [])
            )
            and set(calibration.get("empirical_lower", {})) == gate_fields
            and set(calibration.get("empirical_upper", {})) == gate_fields
        ),
        "benchmark_go_and_complete": (
            _benchmark_projection_consistent(benchmark)
            and
            benchmark.get("decision") == "GO"
            and benchmark.get("numerical") == candidate
            and int(benchmark.get("steps", 0)) >= 2_000
            and benchmark.get("deterministic_algorithms") is True
            and int(benchmark.get("unit_seed", -1)) >= 0
            and int(benchmark.get("micro_batch", 0)) == PRODUCTION_MICROBATCH
            and int(benchmark.get("accumulation_steps", 0)) > 0
            and int(benchmark.get("micro_batch", 0))
            * int(benchmark.get("accumulation_steps", 0))
            == int(benchmark.get("effective_batch", -1))
            and benchmark.get("device", {}).get("allow_tf32") is True
            and benchmark.get("precision", {}).get("matmul_tf32") is True
            and benchmark.get("precision", {}).get("cudnn_tf32") is True
            and _all_true(
                benchmark_checks,
                (
                    "hardware_bound",
                    "completed",
                    "finite",
                    "checkpoint_written",
                    "snapshot_written",
                    "recovery_written",
                    "parameter_ceiling",
                ),
            )
            and int(benchmark.get("parameter_count", -1)) > 0
            and int(benchmark.get("peak_memory_bytes", -1)) > 0
            and int(benchmark.get("objective_sample_evaluations", -1)) > 0
            and int(benchmark.get("objective_forward_calls", -1)) > 0
            and isinstance(benchmark.get("projection_method"), str)
            and float(benchmark.get("non_io_seconds_per_update", -1.0)) > 0.0
            and float(benchmark.get("raw_full_loop_seconds_per_update", -1.0))
            >= float(benchmark.get("non_io_seconds_per_update", math.inf))
            and float(benchmark.get("recovery_io_seconds", -1.0)) > 0.0
            and float(benchmark.get("checkpoint_io_seconds", -1.0)) > 0.0
            and float(benchmark.get("snapshot_io_seconds", -1.0)) > 0.0
            and float(benchmark.get("ordinary_health_seconds", -1.0)) > 0.0
            and float(benchmark.get("checkpoint_health_seconds", -1.0)) > 0.0
            and int(benchmark.get("ordinary_health_samples", -1)) == 512
            and int(benchmark.get("checkpoint_health_samples", -1)) == 2_048
            and all(
                _finite(projection.get("hours"))
                and float(projection["hours"]) > 0.0
                and _finite(projection.get("cost_at_declared_rate"))
                and float(projection["cost_at_declared_rate"]) > 0.0
                and _finite(projection.get("conservative_raw_loop_upper_hours"))
                and float(projection["conservative_raw_loop_upper_hours"])
                >= float(projection["hours"])
                for projection in benchmark.get("projections", {}).values()
            )
            and set(benchmark.get("projections", {})) == {"50000", "150000", "300000"}
        ),
        "admission_benchmark_same_environment": _same_hardware(
            numerical.get("hardware"), benchmark.get("hardware")
        ),
        "baseline_standard_valid": _standard_evaluation_valid(baseline, external=False),
        "baseline_checkpoint_matches_admission": (
            baseline.get("checkpoint_sha256") == numerical.get("checkpoint_sha256")
        ),
        "positive_control_standard_valid": _standard_evaluation_valid(
            positive_control, external=True
        ),
        "positive_control_sanity": all(
            _finite(baseline_metrics.get(name))
            and _finite(control_metrics.get(name))
            and float(control_metrics[name]) < float(baseline_metrics[name])
            for name in standard_metric_names
        ),
        "shared_kid_reference_population": _same_kid_reference(
            baseline_kid_reference,
            control_kid_reference,
            calibration_kid_reference,
        ),
        "shared_metric_environment": _same_metric_environment(
            baseline,
            positive_control,
            metric_calibration,
        ),
        "metric_calibration_complete": (
            metric_calibration.get("decision") == "COMPLETE"
            and int(metric_calibration.get("seed", -1)) == 20_260_806
            and int(metric_calibration.get("samples_per_side", -1))
            == CALIBRATION_SAMPLES_PER_SIDE
            and metric_calibration.get("left_indices_sha256")
            != metric_calibration.get("right_indices_sha256")
            and all(
                _hash64(metric_calibration.get(name))
                for name in ("left_indices_sha256", "right_indices_sha256")
            )
            and int(metric_calibration.get("left", {}).get("count", -1))
            == CALIBRATION_SAMPLES_PER_SIDE
            and int(metric_calibration.get("right", {}).get("count", -1))
            == CALIBRATION_SAMPLES_PER_SIDE
            and calibration_metrics.get("backend") == "clean-fid"
            and calibration_metrics.get("cleanfid_version") == CLEANFID_VERSION
            and int(calibration_metrics.get("kid_seed", -1)) == DEFAULT_KID_SEED
            and _kid_reference_valid(calibration_kid_reference)
            and _finite(direct.get("clean_fid"))
            and float(direct["clean_fid"]) >= 0.0
            and _finite(direct.get("clean_kid"))
            and all(
                _finite(side.get(name))
                for side in (matched.get("left", {}), matched.get("right", {}))
                for name in standard_metric_names
            )
        ),
        "forensics_complete": (
            forensics.get("decision") == "COMPLETE"
            and int(forensics.get("samples", -1)) >= 2_048
            and int(forensics.get("grid_samples", -1)) >= 256
            and forensics.get("checkpoint_kind") == "ema"
            and int(forensics.get("checkpoint_step", -1)) == BASELINE_CHECKPOINT_STEP
            and forensics.get("checkpoint_sha256") == numerical.get("checkpoint_sha256")
            and forensics.get("device", {}).get("allow_tf32") is False
            and _all_true(forensics.get("checks"))
            and _same_hardware(numerical.get("hardware"), forensics.get("hardware"))
        ),
        "installed_cleanfid_pinned": installed_cleanfid_version == CLEANFID_VERSION,
        "all_inputs_match_live_sources": all(
            payload.get("source_sha256") == live_sources for payload in input_artifacts
        ),
    }
    return ("GO" if all(checks.values()) else "NO_GO"), checks


def _smoke_all_arms(numerical: str, calibration: dict) -> dict:
    pool = torch.randn(32, 3, 8, 8)
    results = {}
    for arm in SAMPLER_ARMS:
        frozen = apply_calibrated_gate(
            screen_profile(arm, numerical, smoke=True), calibration
        )
        outcome = train_cap_unit(pool, frozen, "cpu")
        last = outcome.history[-1]["objective_ledger"]
        checks = {
            "completed": outcome.optimizer_updates == frozen.train.updates,
            "finite": outcome.nonfinite_updates == 0,
            "all_rows_logged": last["rows"] == frozen.train.effective_batch,
            "components_logged": "components" in outcome.health[-1],
            "ema_logged_at_checkpoint": "ema_components" in outcome.health[-1],
            "clip_window_counted": outcome.final_window_updates > 0,
        }
        results[arm] = {
            "checks": checks,
            "verdict": "PASS" if all(checks.values()) else "FAIL",
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numerical-admission", type=Path, required=True)
    parser.add_argument("--sampler-audit", type=Path, required=True)
    parser.add_argument("--gate-calibration", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--baseline-standard", type=Path, required=True)
    parser.add_argument("--positive-control-standard", type=Path, required=True)
    parser.add_argument("--metric-calibration", type=Path, required=True)
    parser.add_argument("--checkpoint-forensics", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_PREFLIGHT)
    args = parser.parse_args()
    assert_unused(args.out)

    numerical = verify_json(args.numerical_admission, "cap-emf2-numerical-admission")
    samplers = verify_json(args.sampler_audit, "cap-emf2-sampler-audit")
    calibration = verify_json(args.gate_calibration, "cap-emf2-gate-calibration")
    benchmark = verify_json(args.benchmark, "cap-emf2-full-loop-benchmark")
    baseline = verify_json(args.baseline_standard, "cap-emf-standard-evaluation")
    positive_control = verify_json(
        args.positive_control_standard, "cap-emf-standard-evaluation"
    )
    metric_calibration = verify_json(
        args.metric_calibration, "cap-emf2-real-real-calibration"
    )
    forensics = verify_json(args.checkpoint_forensics, "cap-emf2-checkpoint-forensics")
    candidate = numerical["candidate"]["name"]
    live_sources = source_manifest()
    try:
        cleanfid_version = importlib.metadata.version("clean-fid")
    except importlib.metadata.PackageNotFoundError:
        cleanfid_version = None
    decision, admission_checks = validate_preflight_inputs(
        numerical=numerical,
        samplers=samplers,
        calibration=calibration,
        benchmark=benchmark,
        baseline=baseline,
        positive_control=positive_control,
        metric_calibration=metric_calibration,
        forensics=forensics,
        live_sources=live_sources,
        installed_cleanfid_version=cleanfid_version,
    )
    smoke = _smoke_all_arms(candidate, calibration)
    smoke_ok = all(result["verdict"] == "PASS" for result in smoke.values())
    admission_checks["all_arm_smoke"] = smoke_ok
    decision = "GO" if decision == "GO" and smoke_ok else "NO_GO"

    profiles = {
        arm: profile_payload(
            apply_calibrated_gate(
                screen_profile(arm, candidate, updates=150_000), calibration
            )
        )
        for arm in SAMPLER_ARMS
    }
    result = {
        "status": "cap-emf2-preflight",
        "decision": decision,
        "candidate": candidate,
        "checks": admission_checks,
        "smoke": smoke,
        "profiles_150k": profiles,
        "inputs": {
            "numerical_admission": numerical,
            "sampler_audit": samplers,
            "gate_calibration": calibration,
            "benchmark": benchmark,
            "baseline_standard": baseline,
            "positive_control_standard": positive_control,
            "metric_calibration": metric_calibration,
            "checkpoint_forensics": forensics,
        },
        "cleanfid_version": cleanfid_version,
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": live_sources,
        "limits": [
            "This preflight authorizes developmental screens up to 150k only.",
            "A 300k continuation needs an explicit promotion flag and fixed interim report.",
            "It never authorizes a 600k-scale confirmation or ASFD training.",
            "CleanFID may execute in a separate evaluation environment, but its version is recorded.",
        ],
    }
    digest = write_json_atomic(args.out, result)
    print(json.dumps({"decision": decision, "checks": admission_checks}, indent=2))
    print(f"wrote {args.out} sha256={digest}")
    return 0 if decision == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
