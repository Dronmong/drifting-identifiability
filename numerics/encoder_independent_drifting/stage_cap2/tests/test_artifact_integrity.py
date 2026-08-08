"""Focused CAP2 artifact, recovery, and promotion tamper tests."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ...stage_cap.config import CAPGateConfig
from ...stage_cap.model import CAPPixelTransformer
from ..artifacts import (
    PROTOCOL,
    _state_counts,
    file_sha256,
    load_checkpoint,
    profile_payload,
    save_checkpoint,
    source_manifest,
    verify_file,
    write_json_atomic,
    write_npz_atomic,
)
from ..config import SAMPLER_ARMS, apply_calibrated_gate, screen_profile
from ..early_admission import build_early_admission
from ..preflight import _smoke_all_arms, validate_preflight_inputs
from ..promotion import build_promotion, load_promotion
from ..run_screen import (
    _validate_existing_artifacts,
    _validate_recovery_request,
)
from .test_preflight_integrity import _fixture as _preflight_fixture

ARM = "ordered_uniform"
CANDIDATE = "local_1000_d0002_fp32"


def test_frozen_state_count_separates_trainable_parameters_from_buffers():
    frozen = screen_profile(ARM, CANDIDATE, smoke=True)
    model = CAPPixelTransformer(frozen.model, 1)
    trainable, serialized = _state_counts(model.state_dict())
    assert trainable == model.parameter_count()
    assert serialized - trainable == frozen.model.time_embedding_dim


def _raises(fragment: str, function, *args, **kwargs) -> None:
    try:
        function(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - testing the guarded boundary
        assert fragment in str(error), (fragment, type(error).__name__, str(error))
    else:  # pragma: no cover - assertion branch
        raise AssertionError(f"expected failure containing {fragment!r}")


def _calibration() -> dict:
    return {"status": "cap-emf2-gate-calibration", "gate": asdict(CAPGateConfig())}


def _profile(updates: int):
    return apply_calibrated_gate(
        screen_profile(ARM, CANDIDATE, updates=updates), _calibration()
    )


def test_checkpoint_is_atomic_sidecar_bound_and_metadata_checked():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        declared = profile_payload(_profile(50_000))
        digest = save_checkpoint(
            path,
            {"weight": torch.arange(4, dtype=torch.float32)},
            step=50_000,
            kind="ema",
            arm=ARM,
            declared_profile=declared,
            realized_profile=declared,
            preflight_sha256="a" * 64,
            run_identity_sha256="c" * 64,
            unit_seed=0,
        )
        assert path.is_file()
        assert path.with_suffix(".pt.sha256").is_file()
        payload = load_checkpoint(
            path,
            expected_sha=digest,
            step=50_000,
            kind="ema",
            arm=ARM,
            declared_profile=declared,
            realized_profile=declared,
            preflight_sha256="a" * 64,
            run_identity_sha256="c" * 64,
            unit_seed=0,
        )
        assert payload["parameter_count"] == 4
        assert payload["state_value_count"] == 4
        _raises("metadata", load_checkpoint, path, arm="legacy")


def test_npz_artifact_is_atomic_sidecar_bound_and_non_overwriting():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "features.npz"
        features = np.arange(12, dtype=np.float32).reshape(3, 4)
        digest = write_npz_atomic(path, features=features)
        assert verify_file(path) == digest
        with np.load(path, allow_pickle=False) as archive:
            assert archive.files == ["features"]
            assert np.array_equal(archive["features"], features)
        _raises("overwrite", write_npz_atomic, path, features=features)


def test_unrecorded_checkpoint_newer_than_recovery_is_rejected():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "checkpoints" / f"cap2_{ARM}_step50000_ema.pt"
        declared = profile_payload(_profile(50_000))
        save_checkpoint(
            checkpoint,
            {"weight": torch.ones(2)},
            step=50_000,
            kind="ema",
            arm=ARM,
            declared_profile=declared,
            realized_profile=declared,
            preflight_sha256="b" * 64,
            run_identity_sha256="d" * 64,
            unit_seed=0,
        )
        recovery = {
            "completed_updates": 45_000,
            "checkpoints": {},
            "snapshots": [],
        }
        _raises(
            "newer than recovery",
            _validate_existing_artifacts,
            root=root,
            recovery=recovery,
            arm=ARM,
            candidate=CANDIDATE,
            preflight={"artifact_sha256": "b" * 64},
            calibration=_calibration(),
            realized_micro_batch=16,
            run_identity_sha256="d" * 64,
            unit_seed=0,
        )


def test_300k_requires_completed_150k_or_promoted_recovery():
    profile_name = _profile(300_000).name
    _raises(
        "fresh 300k",
        _validate_recovery_request,
        None,
        requested=300_000,
        expected_profile_name=profile_name,
    )
    jump = {
        "profile_name": profile_name,
        "planned_updates": 100_000,
        "completed_updates": 100_000,
    }
    _raises(
        "completed 150k",
        _validate_recovery_request,
        jump,
        requested=300_000,
        expected_profile_name=profile_name,
    )
    valid = {
        "profile_name": profile_name,
        "planned_updates": 150_000,
        "completed_updates": 150_000,
    }
    _validate_recovery_request(
        valid, requested=300_000, expected_profile_name=profile_name
    )


def test_promotion_roundtrip_revalidates_every_bound_artifact():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        preflight_path = root / "preflight.json"
        result_50k_path = root / "result_50000.json"
        raw_checkpoint_50k_path = (
            root / "checkpoints" / f"cap2_{ARM}_step50000_raw.pt"
        )
        readmission_50k_path = root / "readmission_50000_raw.json"
        early_admission_path = root / "early_admission_50000.json"
        result_path = root / "result_150000.json"
        raw_checkpoint_path = (
            root / "checkpoints" / f"cap2_{ARM}_step150000_raw.pt"
        )
        checkpoint_path = root / "checkpoints" / f"cap2_{ARM}_step150000_ema.pt"
        admission_path = root / "readmission.json"
        evaluation_path = root / "development_evaluation.json"
        promotion_path = root / "promotion.json"
        grid_path = root / "uncurated_grid.png"
        grid_path.write_bytes(b"fixed-test-grid")

        sources = source_manifest()
        fixture = _preflight_fixture()
        declared = profile_payload(
            apply_calibrated_gate(
                screen_profile(ARM, CANDIDATE, updates=150_000),
                fixture["calibration"],
            )
        )
        inputs = {
            "numerical_admission": fixture["numerical"],
            "sampler_audit": fixture["samplers"],
            "gate_calibration": fixture["calibration"],
            "benchmark": fixture["benchmark"],
            "baseline_standard": fixture["baseline"],
            "positive_control_standard": fixture["positive_control"],
            "metric_calibration": fixture["metric_calibration"],
            "checkpoint_forensics": fixture["forensics"],
        }
        for artifact in inputs.values():
            artifact["source_sha256"] = sources
        baseline = inputs["baseline_standard"]
        positive_control = inputs["positive_control_standard"]
        metric_calibration = inputs["metric_calibration"]
        metric_calibration["status"] = "cap-emf2-real-real-calibration"
        baseline_feature = baseline["repository_feature_metrics"]
        baseline_feature.update(
            {
                "precision": 0.45,
                "recall": 0.35,
                "pr_f1": 2 * 0.45 * 0.35 / (0.45 + 0.35),
                "unbiased_kid": 0.03,
            }
        )
        baseline["repository_auxiliary"]["repository_feature_metrics"] = (
            baseline_feature
        )
        environment = {
            "deterministic_algorithms": True,
            "packages": {
                "clean-fid": "0.1.35",
                "numpy": "2.2.0",
                "Pillow": "11.0.0",
                "torch": "2.7.1",
                "torchvision": "0.22.1",
            },
            "device": "cuda",
            "gpu_name": "Test GPU",
            "torch_cuda_version": "12.6",
            "cudnn_version": 90501,
            "image_quantization": "fixed",
            "numerical_settings": {"allow_tf32": False},
        }
        baseline["provenance"] = copy.deepcopy(environment)
        positive_control["provenance"] = copy.deepcopy(environment)
        metric_calibration["provenance"] = copy.deepcopy(environment)
        baseline["samples"]["batch"] = 500
        baseline["metrics"].update({"metric_batch": 128, "metric_workers": 4})
        baseline_feature["feature_batch"] = 128
        decision, preflight_checks = validate_preflight_inputs(
            numerical=inputs["numerical_admission"],
            samplers=inputs["sampler_audit"],
            calibration=inputs["gate_calibration"],
            benchmark=inputs["benchmark"],
            baseline=baseline,
            positive_control=positive_control,
            metric_calibration=metric_calibration,
            forensics=inputs["checkpoint_forensics"],
            live_sources=sources,
            installed_cleanfid_version="0.1.35",
        )
        assert decision == "GO"
        smoke = _smoke_all_arms(CANDIDATE, inputs["gate_calibration"])
        preflight_checks["all_arm_smoke"] = all(
            record["verdict"] == "PASS" for record in smoke.values()
        )
        preflight_payload = {
            "status": "cap-emf2-preflight",
            "decision": "GO",
            "candidate": CANDIDATE,
            "checks": preflight_checks,
            "smoke": smoke,
            "profiles_150k": {
                arm: profile_payload(
                    apply_calibrated_gate(
                        screen_profile(arm, CANDIDATE, updates=150_000),
                        inputs["gate_calibration"],
                    )
                )
                for arm in SAMPLER_ARMS
            },
            "protocol_sha256": file_sha256(PROTOCOL),
            "source_sha256": sources,
            "inputs": inputs,
            "cleanfid_version": "0.1.35",
        }
        preflight_sha = write_json_atomic(preflight_path, preflight_payload)
        declared_50k = profile_payload(
            apply_calibrated_gate(
                screen_profile(ARM, CANDIDATE, updates=50_000),
                inputs["gate_calibration"],
            )
        )
        raw_checkpoint_50k_sha = save_checkpoint(
            raw_checkpoint_50k_path,
            {"weight": torch.ones(3)},
            step=50_000,
            kind="raw",
            arm=ARM,
            declared_profile=declared_50k,
            realized_profile=declared_50k,
            preflight_sha256=preflight_sha,
            run_identity_sha256="a" * 64,
            unit_seed=0,
        )
        write_json_atomic(
            result_50k_path,
            {
                "status": "cap-emf2-screen-unit",
                "development_only": True,
                "arm": ARM,
                "numerical_candidate": CANDIDATE,
                "preflight_sha256": preflight_sha,
                "run_identity_sha256": "a" * 64,
                "unit_seed": 0,
                "declared_profile": declared_50k,
                "realized_profile": declared_50k,
                "hardware": copy.deepcopy(inputs["benchmark"]["hardware"]),
                "training": {
                    "optimizer_updates": 50_000,
                    "nonfinite_updates": 0,
                },
                "checkpoints": {
                    "50000": {
                        "raw": {
                            "path": "checkpoints/cap2_ordered_uniform_step50000_raw.pt",
                            "sha256": raw_checkpoint_50k_sha,
                        },
                        "ema": {"path": "unused-ema50.pt", "sha256": "9" * 64},
                    }
                },
            },
        )
        readmission_50k = copy.deepcopy(inputs["numerical_admission"])
        readmission_50k.update(
            {
                "status": "cap-emf2-numerical-admission",
                "checkpoint_sha256": raw_checkpoint_50k_sha,
                "checkpoint_step": 50_000,
                "checkpoint_identity": {
                    "valid": True,
                    "stage": "cap-emf-2-screen",
                    "kind": "raw",
                    "arm": ARM,
                },
                "source_sha256": sources,
            }
        )
        write_json_atomic(readmission_50k_path, readmission_50k)
        early_admission = build_early_admission(
            preflight_path=preflight_path,
            result_path=result_50k_path,
            raw_checkpoint_path=raw_checkpoint_50k_path,
            readmission_path=readmission_50k_path,
            out=early_admission_path,
        )
        assert early_admission["decision"] == "GO", early_admission["failed"]
        checkpoint_sha = save_checkpoint(
            checkpoint_path,
            {"weight": torch.ones(3)},
            step=150_000,
            kind="ema",
            arm=ARM,
            declared_profile=declared,
            realized_profile=declared,
            preflight_sha256=preflight_sha,
            run_identity_sha256="a" * 64,
            unit_seed=0,
        )
        raw_checkpoint_sha = save_checkpoint(
            raw_checkpoint_path,
            {"weight": torch.ones(3)},
            step=150_000,
            kind="raw",
            arm=ARM,
            declared_profile=declared,
            realized_profile=declared,
            preflight_sha256=preflight_sha,
            run_identity_sha256="a" * 64,
            unit_seed=0,
        )
        write_json_atomic(
            result_path,
            {
                "status": "cap-emf2-screen-unit",
                "development_only": True,
                "arm": ARM,
                "numerical_candidate": CANDIDATE,
                "preflight_sha256": preflight_sha,
                "run_identity_sha256": "a" * 64,
                "unit_seed": 0,
                "declared_profile": declared,
                "realized_profile": declared,
                "hardware": copy.deepcopy(inputs["benchmark"]["hardware"]),
                "early_admission": {
                    "path": early_admission_path.name,
                    "sha256": early_admission["artifact_sha256"],
                },
                "training": {
                    "optimizer_updates": 150_000,
                    "history": [
                        {
                            "step": step,
                            "objective_ledger": {
                                "named_regions": {
                                    "inference_corner": {
                                        "count": 2_048,
                                        "mean_raw_mse": (
                                            1.0 if step <= 125_000 else 1.2
                                        ),
                                    }
                                }
                            },
                        }
                        for step in range(100_500, 150_001, 500)
                    ],
                    "health": [
                        {
                            "step": 100_000,
                            "ema_components": {
                                "base": {"haar_HH_ratio": 0.4},
                                "final": {"haar_HH_ratio": 0.5},
                                "refiner_residual": {"haar_HH_variance": 0.1},
                            },
                        },
                        {
                            "step": 150_000,
                            "ema_components": {
                                "base": {"haar_HH_ratio": 0.5},
                                "final": {"haar_HH_ratio": 0.6},
                                "refiner_residual": {"haar_HH_variance": 0.12},
                            },
                        },
                    ],
                },
                "train_only_gate": {
                    "verdict": "PASS",
                    "checks": {"all": True},
                    "thresholds": declared["gate"],
                },
                "checkpoints": {
                    "50000": {
                        "raw": {"path": "raw50.pt", "sha256": "1" * 64},
                        "ema": {"path": "ema50.pt", "sha256": "2" * 64},
                    },
                    "100000": {
                        "raw": {"path": "raw100.pt", "sha256": "3" * 64},
                        "ema": {"path": "ema100.pt", "sha256": "4" * 64},
                    },
                    "150000": {
                        "raw": {
                            "path": "checkpoints/unused-portable-raw-name.pt",
                            "sha256": raw_checkpoint_sha,
                        },
                        "ema": {
                            "path": "checkpoints/unused-portable-name.pt",
                            "sha256": checkpoint_sha,
                        },
                    },
                },
            },
        )
        readmission = copy.deepcopy(inputs["numerical_admission"])
        readmission.update(
            {
                "status": "cap-emf2-numerical-admission",
                "checkpoint_sha256": raw_checkpoint_sha,
                "checkpoint_step": 150_000,
                "checkpoint_identity": {
                    "valid": True,
                    "stage": "cap-emf-2-screen",
                    "kind": "raw",
                    "arm": ARM,
                },
                "source_sha256": sources,
            }
        )
        write_json_atomic(admission_path, readmission)
        candidate_feature = copy.deepcopy(baseline["repository_feature_metrics"])
        candidate_feature.update(
            {
                "precision": 0.50,
                "recall": 0.40,
                "pr_f1": 2 * 0.5 * 0.4 / (0.5 + 0.4),
                "unbiased_kid": 0.02,
                "generated_feature_sha256": "a" * 64,
            }
        )
        candidate_memorization = copy.deepcopy(baseline["memorization"])
        candidate_memorization.update(
            {
                "exact_pixel_copy_fraction": 0.0,
                "exact_generated_duplicate_fraction": 0.0,
            }
        )
        candidate_auxiliary = copy.deepcopy(baseline["repository_auxiliary"])
        candidate_auxiliary["png_manifest_sha256"] = "f" * 64
        candidate_auxiliary["generated_subset"].update({"tensor_sha256": "b" * 64})
        candidate_auxiliary["repository_feature_metrics"] = candidate_feature
        candidate_auxiliary["memorization"] = candidate_memorization
        write_json_atomic(
            evaluation_path,
            {
                "status": "cap-emf2-development-evaluation",
                "development_only": True,
                "arm": ARM,
                "step": 150_000,
                "unit": {
                    "sha256": file_sha256(result_path),
                    "preflight_sha256": preflight_sha,
                },
                "checkpoint": {
                    "sha256": checkpoint_sha,
                    "step": 150_000,
                    "kind": "ema",
                },
                "fixed_protocol": {
                    "generated_samples": 50_000,
                    "generation_seed": 20_260_804,
                    "clean_kid_seed": 20_260_831,
                },
                "samples": {
                    "count": 50_000,
                    "seed": 20_260_804,
                    "batch": 500,
                    "one_model_call_per_batch": True,
                    "png_manifest_sha256": "f" * 64,
                    "image_size": [32, 32],
                    "mode": "RGB",
                },
                "standard_train_reference_metrics": {
                    "backend": "clean-fid",
                    "cleanfid_version": "0.1.35",
                    "reference": "cifar10/train/32",
                    "mode": "clean",
                    "feature_count": 50_000,
                    "kid_seed": 20_260_831,
                    "clean_fid_cifar10_train": 8.0,
                    "clean_kid_cifar10_train": 0.01,
                    "metric_batch": 128,
                    "metric_workers": 4,
                },
                "repository_feature_metrics": candidate_feature,
                "memorization": candidate_memorization,
                "repository_auxiliary": candidate_auxiliary,
                "provenance": copy.deepcopy(environment),
                "uncurated_grid": {
                    "path": str(grid_path.resolve()),
                    "sha256": file_sha256(grid_path),
                    "rows": 8,
                    "columns": 16,
                    "selection": "fixed first 128 samples; no curation",
                },
                "source_sha256": sources,
            },
        )
        promotion = build_promotion(
            preflight_path=preflight_path,
            result_path=result_path,
            raw_checkpoint_path=raw_checkpoint_path,
            checkpoint_path=checkpoint_path,
            admission_path=admission_path,
            evaluation_path=evaluation_path,
            out=promotion_path,
        )
        assert promotion["decision"] == "GO", promotion["failed"]
        loaded = load_promotion(
            promotion_path,
            preflight_path=preflight_path,
            result_path=result_path,
            raw_checkpoint_path=raw_checkpoint_path,
            checkpoint_path=checkpoint_path,
            arm=ARM,
            candidate=CANDIDATE,
        )
        assert loaded["revalidated"] is True

        weaker_evaluation_path = root / "weaker_development_evaluation.json"
        weaker_promotion_path = root / "weaker_promotion.json"
        weaker_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        weaker_evaluation["standard_train_reference_metrics"][
            "clean_fid_cifar10_train"
        ] = (
            baseline["metrics"]["clean_fid_cifar10_train"]
            - abs(
                metric_calibration["metrics"]["matched_published_train_reference"][
                    "left"
                ]["clean_fid_cifar10_train"]
                - metric_calibration["metrics"][
                    "matched_published_train_reference"
                ]["right"]["clean_fid_cifar10_train"]
            )
        )
        write_json_atomic(weaker_evaluation_path, weaker_evaluation)
        rejected = build_promotion(
            preflight_path=preflight_path,
            result_path=result_path,
            raw_checkpoint_path=raw_checkpoint_path,
            checkpoint_path=checkpoint_path,
            admission_path=admission_path,
            evaluation_path=weaker_evaluation_path,
            out=weaker_promotion_path,
        )
        assert rejected["decision"] == "NO_GO"
        assert "candidate_fid_improves_beyond_calibration" in rejected["failed"]

        worse_repository_path = root / "worse_repository_evaluation.json"
        worse_repository_promotion = root / "worse_repository_promotion.json"
        worse_repository = json.loads(evaluation_path.read_text(encoding="utf-8"))
        worse_repository["repository_feature_metrics"]["unbiased_kid"] = 0.04
        write_json_atomic(worse_repository_path, worse_repository)
        rejected = build_promotion(
            preflight_path=preflight_path,
            result_path=result_path,
            raw_checkpoint_path=raw_checkpoint_path,
            checkpoint_path=checkpoint_path,
            admission_path=admission_path,
            evaluation_path=worse_repository_path,
            out=worse_repository_promotion,
        )
        assert rejected["decision"] == "NO_GO"
        assert "candidate_repository_kid_improves" in rejected["failed"]


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
