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
from .. import early_admission as early_admission_module
from .. import promotion as promotion_module
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
    write_sha256_sidecar_atomic,
)
from ..budget import GIB, build_budget_plan, build_storage_plan
from ..config import SAMPLER_ARMS, apply_calibrated_gate, screen_profile
from ..early_admission import build_early_admission
from ..preflight import _smoke_all_arms, validate_preflight_inputs
from ..promotion import build_promotion, load_promotion
from ..run_screen import (
    _ensure_foundation_continuation_authority,
    _foundation_execution_stop,
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
        campaign="matched_screen",
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
        campaign="matched_screen",
    )
    valid = {
        "profile_name": profile_name,
        "planned_updates": 150_000,
        "completed_updates": 150_000,
    }
    _validate_recovery_request(
        valid,
        requested=300_000,
        expected_profile_name=profile_name,
        campaign="matched_screen",
    )
    completed = {
        "profile_name": profile_name,
        "planned_updates": 300_000,
        "completed_updates": 300_000,
    }
    # A crash after the terminal recovery commit but before result publication
    # must be able to re-enter for finalization without another optimizer step.
    _validate_recovery_request(
        completed,
        requested=300_000,
        expected_profile_name=profile_name,
        campaign="matched_screen",
    )


def test_ordered_foundation_is_one_fresh_750k_plan_or_exact_resume():
    profile_name = _profile(750_000).name
    _validate_recovery_request(
        None,
        requested=750_000,
        expected_profile_name=profile_name,
        campaign="ordered_750_foundation",
    )
    _raises(
        "one 750k horizon",
        _validate_recovery_request,
        None,
        requested=300_000,
        expected_profile_name=profile_name,
        campaign="ordered_750_foundation",
    )
    recovery = {
        "profile_name": profile_name,
        "planned_updates": 750_000,
        "completed_updates": 425_000,
    }
    _validate_recovery_request(
        recovery,
        requested=750_000,
        expected_profile_name=profile_name,
        campaign="ordered_750_foundation",
    )
    bad = {**recovery, "planned_updates": 300_000}
    _raises(
        "750k plan",
        _validate_recovery_request,
        bad,
        requested=750_000,
        expected_profile_name=profile_name,
        campaign="ordered_750_foundation",
    )


def test_ordered_foundation_cannot_cross_50k_without_bound_go():
    _raises(
        "must first pause",
        _foundation_execution_stop,
        None,
        pause_for_early_admission=False,
        has_early_admission=False,
    )
    assert (
        _foundation_execution_stop(
            None,
            pause_for_early_admission=True,
            has_early_admission=False,
        )
        == 50_000
    )
    paused = {
        "planned_updates": 750_000,
        "completed_updates": 50_000,
    }
    _raises(
        "requires the bound GO",
        _foundation_execution_stop,
        paused,
        pause_for_early_admission=False,
        has_early_admission=False,
    )
    assert (
        _foundation_execution_stop(
            paused,
            pause_for_early_admission=False,
            has_early_admission=True,
        )
        == 750_000
    )
    resumed = {**paused, "completed_updates": 425_000}
    assert (
        _foundation_execution_stop(
            resumed,
            pause_for_early_admission=False,
            has_early_admission=True,
        )
        == 750_000
    )


def test_foundation_early_admission_binds_the_750k_profile_and_50k_pause(
    monkeypatch,
):
    calibration = _preflight_fixture()["calibration"]
    declared = profile_payload(
        apply_calibrated_gate(
            screen_profile(ARM, CANDIDATE, updates=750_000), calibration
        )
    )
    hardware = {
        "matches": True,
        "actual_gpu_name": "Test GPU",
        "compute_capability": "8.9",
        "torch_version": "2.7.1",
        "cuda_runtime": "12.6",
        "cudnn_version": 90501,
        "cublas_workspace_config": ":4096:8",
    }
    preflight = {
        "artifact_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "candidate": CANDIDATE,
        "budget": {"campaign": "ordered_750_foundation"},
        "inputs": {
            "gate_calibration": calibration,
            "numerical_admission": {
                "hardware": hardware,
                "production_numerical_mode": {"mode": "test"},
            },
            "benchmark": {
                "micro_batch": declared["train"]["micro_batch"],
                "accumulation_steps": declared["train"]["accumulation_steps"],
                "hardware": hardware,
            },
        },
    }
    result = {
        "artifact_sha256": "c" * 64,
        "arm": ARM,
        "numerical_candidate": CANDIDATE,
        "preflight_sha256": "a" * 64,
        "development_only": True,
        "declared_profile": declared,
        "realized_profile": declared,
        "hardware": hardware,
        "training": {"optimizer_updates": 50_000, "nonfinite_updates": 0},
        "checkpoints": {
            "50000": {
                "raw": {"sha256": "d" * 64},
                "ema": {"sha256": "e" * 64},
            }
        },
        "recovery": {
            "planned_updates": 750_000,
            "completed_updates": 50_000,
            "sha256": "f" * 64,
        },
        "foundation_pause": {
            "planned_updates": 750_000,
            "paused_at": 50_000,
            "purpose": "raw-state numerical admission before continuation",
        },
    }
    readmission = {
        "artifact_sha256": "1" * 64,
        "checkpoint_sha256": "d" * 64,
        "checkpoint_step": 50_000,
        "checkpoint_identity": {
            "valid": True,
            "stage": "cap-emf-2-screen",
            "kind": "raw",
            "arm": ARM,
        },
        "candidate": {"name": CANDIDATE},
        "source_sha256": "b" * 64,
        "hardware": hardware,
        "production_numerical_mode": {"mode": "test"},
    }
    monkeypatch.setattr(early_admission_module, "load_preflight", lambda _: preflight)
    monkeypatch.setattr(
        early_admission_module,
        "verify_json",
        lambda path, status: result if "result" in str(path) else readmission,
    )
    monkeypatch.setattr(
        early_admission_module,
        "load_checkpoint",
        lambda *args, **kwargs: {"artifact_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        early_admission_module, "admission_matrix_complete", lambda _: True
    )
    inputs = early_admission_module._inputs(
        preflight_path=Path("preflight.json"),
        result_path=Path("result_50000.json"),
        raw_checkpoint_path=Path("raw.pt"),
        readmission_path=Path("readmission.json"),
    )
    assert inputs["campaign"] == "ordered_750_foundation"
    assert inputs["planned_horizon"] == 750_000
    assert all(inputs["checks"].values()), inputs["checks"]

    result["foundation_pause"] = None
    failed = early_admission_module._inputs(
        preflight_path=Path("preflight.json"),
        result_path=Path("result_50000.json"),
        raw_checkpoint_path=Path("raw.pt"),
        readmission_path=Path("readmission.json"),
    )
    assert failed["checks"]["foundation_same_horizon_pause"] is False


def test_foundation_continuation_authority_is_immutable():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "foundation_continuation_authority.json"
        payload = {
            "status": "cap-emf2-foundation-continuation-authority",
            "early_admission_sha256": "a" * 64,
        }
        first = _ensure_foundation_continuation_authority(path, payload)
        assert _ensure_foundation_continuation_authority(path, payload) == first
        _raises(
            "different 50k continuation authority",
            _ensure_foundation_continuation_authority,
            path,
            {**payload, "early_admission_sha256": "b" * 64},
        )


def test_promotion_roundtrip_revalidates_every_bound_artifact(monkeypatch):
    # The production boundary decodes 50k PNGs and loads two 50k x 2048
    # archives.  Keep this artifact-wiring test small; the evidence helper has
    # focused tests below and is not bypassable by the production CLI.
    monkeypatch.setattr(
        promotion_module,
        "revalidate_clean_evaluation_evidence",
        lambda evaluation, *, anchor: {
            "valid": True,
            "recomputed": {
                key: evaluation["standard_train_reference_metrics"][key]
                for key in (
                    "clean_fid_cifar10_train",
                    "clean_kid_cifar10_train",
                )
            },
        },
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        preflight_path = root / "preflight.json"
        result_50k_path = root / "result_50000.json"
        raw_checkpoint_50k_path = root / "checkpoints" / f"cap2_{ARM}_step50000_raw.pt"
        readmission_50k_path = root / "readmission_50000_raw.json"
        early_admission_path = root / "early_admission_50000.json"
        result_path = root / "result_150000.json"
        raw_checkpoint_path = root / "checkpoints" / f"cap2_{ARM}_step150000_raw.pt"
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

        def fixed_corner(error: float) -> dict:
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
                    "stopped_evaluation": declared["objective"]["stopped_evaluation"],
                    "emf_delta": declared["objective"]["emf_delta"],
                    "emf_denominator_floor": declared["objective"][
                        "emf_denominator_floor"
                    ],
                },
                "raw": copy.deepcopy(summary),
                "ema": copy.deepcopy(summary),
            }

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
        baseline["samples"]["batch"] = 128
        baseline["metrics"].update({"metric_batch": 128, "metric_workers": 0})
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
        budget = build_budget_plan(
            inputs["benchmark"],
            max_total_cost=1_000.0,
            nontraining_reserve=1.0,
            contingency_fraction=0.15,
        )
        assert budget["within_ceiling"] is True
        preflight_checks["aggregate_budget_within_ceiling"] = True
        storage = build_storage_plan(
            inputs["benchmark"],
            storage_root="X:/",
            total_bytes=200 * GIB,
            free_bytes=190 * GIB,
            artifact_reserve_gib=20.0,
            contingency_fraction=0.20,
        )
        assert storage["decision"] == "GO"
        preflight_checks["durable_storage_capacity"] = True
        retained_metric_evidence = {
            "baseline": {"valid": True},
            "positive_control": {"valid": True},
            "metric_calibration": {"valid": True},
        }
        preflight_checks["retained_metric_leaves_recomputed"] = True
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
            "foundation_profile_750k": profile_payload(
                apply_calibrated_gate(
                    screen_profile(ARM, CANDIDATE, updates=750_000),
                    inputs["gate_calibration"],
                )
            ),
            "budget": budget,
            "storage": storage,
            "retained_metric_evidence": retained_metric_evidence,
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

        def checkpoint_preview(step: int) -> dict:
            records = {}
            for kind in ("raw", "ema"):
                path = root / "previews" / f"cap2_{ARM}_step{step}_{kind}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"fixed-{step}-{kind}".encode())
                digest = write_sha256_sidecar_atomic(path)
                relative = path.relative_to(root).as_posix()
                records[kind] = {
                    "path": relative,
                    "sha256": digest,
                    "rows": 8,
                    "columns": 16,
                    "samples": 128,
                    "selection": (
                        "first fixed sealed train-only health-noise rows; no curation"
                    ),
                    "durable_mirror": {
                        "relative_path": relative,
                        "sha256": digest,
                        "bytes": path.stat().st_size,
                    },
                }
            return {
                "status": "cap-emf2-fixed-checkpoint-previews",
                "step": step,
                "raw_and_ema": records,
                "quantitative_role": ("report/veto only; never rescues a failed gate"),
            }

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
                            "step": 50_000,
                            "checkpoint_health_observation": checkpoint_preview(50_000),
                            "ema_components": {
                                "base": {"haar_HH_ratio": 0.35},
                                "final": {"haar_HH_ratio": 0.45},
                                "refiner_residual": {"haar_HH_variance": 0.09},
                            },
                        },
                        {
                            "step": 100_000,
                            "fixed_exact_inference_corner": fixed_corner(1.0),
                            "checkpoint_health_observation": checkpoint_preview(
                                100_000
                            ),
                            "ema_components": {
                                "base": {"haar_HH_ratio": 0.4},
                                "final": {"haar_HH_ratio": 0.5},
                                "refiner_residual": {"haar_HH_variance": 0.1},
                            },
                        },
                        {
                            "step": 150_000,
                            "fixed_exact_inference_corner": fixed_corner(1.2),
                            "checkpoint_health_observation": checkpoint_preview(
                                150_000
                            ),
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
                    "path": result_path.name,
                    "sha256": file_sha256(result_path),
                    "preflight_sha256": preflight_sha,
                },
                "checkpoint": {
                    "path": checkpoint_path.relative_to(root).as_posix(),
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
                    "directory": "eval_pngs",
                    "count": 50_000,
                    "seed": 20_260_804,
                    "batch": 128,
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
                    "metric_workers": 0,
                    "kid_reference": copy.deepcopy(
                        baseline["metrics"]["kid_reference"]
                    ),
                    "generated_features": {
                        "path": "development_evaluation_clean_features.npz",
                        "sha256": "d" * 64,
                        "feature_sha256": "e" * 64,
                        "count": 50_000,
                        "dimension": 2_048,
                        "dtype": "float32",
                        "source_png_manifest_sha256": "f" * 64,
                        "generation_seed": 20_260_804,
                        "population": (
                            "all fixed-seed generated images in sequential PNG order"
                        ),
                        "preprocessing": (
                            "clean-fid 0.1.35 clean Inception preprocessing"
                        ),
                    },
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
        ] = baseline["metrics"]["clean_fid_cifar10_train"] - abs(
            metric_calibration["metrics"]["direct_disjoint_pair"]["clean_fid"]
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
        # A concurrent legacy control may intentionally fail only the two
        # historical-quality checks.  The loader must still revalidate and
        # return that immutable NO_GO record when the runner explicitly asks
        # for control eligibility rather than individual scientific success.
        loaded_control = load_promotion(
            weaker_promotion_path,
            preflight_path=preflight_path,
            result_path=result_path,
            raw_checkpoint_path=raw_checkpoint_path,
            checkpoint_path=checkpoint_path,
            arm=ARM,
            candidate=CANDIDATE,
            require_go=False,
        )
        assert loaded_control["revalidated"] is True
        assert loaded_control["control_continuation"]["decision"] == "GO"

        auxiliary_regression_path = root / "auxiliary_regression_evaluation.json"
        auxiliary_regression_promotion = root / "auxiliary_regression_promotion.json"
        auxiliary_regression = json.loads(evaluation_path.read_text(encoding="utf-8"))
        regressed_feature = auxiliary_regression["repository_feature_metrics"]
        regressed_feature.update(
            {
                "unbiased_kid": 0.04,
                "precision": 0.44,
                "recall": 0.34,
                "pr_f1": 2 * 0.44 * 0.34 / (0.44 + 0.34),
            }
        )
        auxiliary_regression["repository_auxiliary"]["repository_feature_metrics"] = (
            copy.deepcopy(regressed_feature)
        )
        write_json_atomic(auxiliary_regression_path, auxiliary_regression)
        accepted = build_promotion(
            preflight_path=preflight_path,
            result_path=result_path,
            raw_checkpoint_path=raw_checkpoint_path,
            checkpoint_path=checkpoint_path,
            admission_path=admission_path,
            evaluation_path=auxiliary_regression_path,
            out=auxiliary_regression_promotion,
        )
        assert accepted["decision"] == "GO", accepted["failed"]
        diagnostics = accepted["comparison"]["auxiliary_relative_diagnostics"]
        assert diagnostics["repository_kid_lower_than_baseline"] is False
        assert diagnostics["does_not_lose_both_precision_and_recall"] is False

        invalid_auxiliary_path = root / "invalid_auxiliary_evaluation.json"
        invalid_auxiliary_promotion = root / "invalid_auxiliary_promotion.json"
        invalid_auxiliary = json.loads(evaluation_path.read_text(encoding="utf-8"))
        invalid_auxiliary["repository_feature_metrics"]["unbiased_kid"] = float("nan")
        invalid_auxiliary["repository_auxiliary"]["repository_feature_metrics"] = (
            copy.deepcopy(invalid_auxiliary["repository_feature_metrics"])
        )
        write_json_atomic(invalid_auxiliary_path, invalid_auxiliary)
        rejected = build_promotion(
            preflight_path=preflight_path,
            result_path=result_path,
            raw_checkpoint_path=raw_checkpoint_path,
            checkpoint_path=checkpoint_path,
            admission_path=admission_path,
            evaluation_path=invalid_auxiliary_path,
            out=invalid_auxiliary_promotion,
        )
        assert rejected["decision"] == "NO_GO"
        assert "evaluation_auxiliary_metrics_finite" in rejected["failed"]

        collapsed_path = root / "collapsed_evaluation.json"
        collapsed_promotion = root / "collapsed_promotion.json"
        collapsed = json.loads(evaluation_path.read_text(encoding="utf-8"))
        collapsed_feature = collapsed["repository_feature_metrics"]
        collapsed_feature.update(
            {
                "precision": 0.04,
                "recall": 0.40,
                "pr_f1": 2 * 0.04 * 0.40 / (0.04 + 0.40),
            }
        )
        collapsed["repository_auxiliary"]["repository_feature_metrics"] = copy.deepcopy(
            collapsed_feature
        )
        write_json_atomic(collapsed_path, collapsed)
        rejected = build_promotion(
            preflight_path=preflight_path,
            result_path=result_path,
            raw_checkpoint_path=raw_checkpoint_path,
            checkpoint_path=checkpoint_path,
            admission_path=admission_path,
            evaluation_path=collapsed_path,
            out=collapsed_promotion,
        )
        assert rejected["decision"] == "NO_GO"
        assert "evaluation_precision_recall_noncollapse" in rejected["failed"]


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
