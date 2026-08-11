"""Policy and immutable-binding tests for the CAP2 300k verdict."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import asdict

import pytest
import torch

from .. import final_verdict as module
from ..artifacts import write_json_atomic
from ..durable_mirror import DurableMirror, provision_root
from ..final_verdict import (
    FinalArmArtifacts,
    _final_policy,
    build_final_verdict,
    revalidate_final_verdict,
)

MARGIN = {
    "kind": "absolute direct disjoint real/real discrepancy",
    "clean_fid": 2.0,
    "clean_kid": 0.002,
    "statistical_scope": "deterministic finite-sample margin; not a confidence interval",
}


def _selection() -> dict:
    ordered = "ordered_uniform"
    return {
        "artifact_sha256": "a" * 64,
        "decision": "GO",
        "revalidated": True,
        "candidate": "local_1000_d0002_fp32",
        "ordered_winner": ordered,
        "selected_arms": ["legacy", ordered],
    }


def _arm(*, fid: float, kid: float, valid: bool = True) -> dict:
    return {
        "valid": valid,
        "checks": {"absolute_noncollapse": valid},
        "failed": [] if valid else ["absolute_noncollapse"],
        "metrics": {
            "clean_fid": fid,
            "clean_kid": kid,
            # Deliberately awful auxiliary values: these are never selectors.
            "repository_kid_reported_only": 999.0,
            "precision_reported_with_absolute_floor": 0.05,
            "recall_reported_with_absolute_floor": 0.05,
            "pr_f1_reported_with_absolute_floor": 0.05,
        },
        "quality_retention": {
            "valid": True,
            "checks": {"fid_retains_150k_quality_within_margin": True},
            "failed": [],
        },
        "clean_evaluation_evidence": {
            "valid": True,
            "checks": {"clean_fid_recomputed": True},
            "recomputed": {
                module.FID_KEY: fid,
                module.KID_KEY: kid,
            },
            "limit": "test fixture",
        },
        "final_exact_corner_trajectory": {"stable": True},
        "recovery_continuity": {
            "promotion_boundary": {"valid": True},
            "final": {"valid": True},
            "prefix": {"valid": True},
            "committed_steps": [150_000, 300_000],
        },
        "artifact_sha256": {
            name: character * 64
            for name, character in zip(
                (
                    "result",
                    "raw_checkpoint",
                    "ema_checkpoint",
                    "raw_readmission",
                    "evaluation",
                ),
                "bcdef",
                strict=True,
            )
        },
        "run_identity_sha256": "9" * 64,
        "unit_seed": 0,
    }


def test_paired_go_requires_both_50k_standard_metrics_beyond_margin() -> None:
    arms = {
        "legacy": _arm(fid=20.0, kid=0.020),
        "ordered_uniform": _arm(fid=17.0, kid=0.017),
    }
    result = _final_policy(selection=_selection(), arms=arms, calibration_margin=MARGIN)
    assert result["decision"] == "GO"
    assert result["claim"]["ordered_beats_concurrent_legacy"] is True
    assert result["claim"]["auxiliary_metrics_used_for_selection"] is False

    arms["ordered_uniform"]["metrics"]["clean_kid"] = 0.019
    result = _final_policy(selection=_selection(), arms=arms, calibration_margin=MARGIN)
    assert result["decision"] == "NO_GO"
    assert result["comparison"]["ordered_fid_wins_beyond_margin"] is True
    assert result["comparison"]["ordered_kid_wins_beyond_margin"] is False


@pytest.mark.parametrize(
    ("selected", "winner"),
    [
        (["ordered_uniform"], "ordered_uniform"),
        (["ordered_uniform", "legacy"], "ordered_uniform"),
        (["legacy", "legacy"], "ordered_uniform"),
        (["legacy", "ordered_uniform"], "ordered_logitnormal"),
    ],
)
def test_final_policy_rejects_noncanonical_or_unpaired_selection(
    selected: list[str], winner: str
) -> None:
    selection = _selection()
    selection["selected_arms"] = selected
    selection["ordered_winner"] = winner
    with pytest.raises(RuntimeError, match="paired selection|malformed"):
        _final_policy(
            selection=selection,
            arms={
                "legacy": _arm(fid=20.0, kid=0.020),
                "ordered_uniform": _arm(fid=17.0, kid=0.017),
            },
            calibration_margin=MARGIN,
        )


def test_absolute_collapse_veto_still_blocks_go() -> None:
    arms = {
        "legacy": _arm(fid=20.0, kid=0.020),
        "ordered_uniform": _arm(fid=17.0, kid=0.017, valid=False),
    }
    result = _final_policy(selection=_selection(), arms=arms, calibration_margin=MARGIN)
    assert result["decision"] == "NO_GO"
    assert result["checks"]["all_selected_300k_arms_valid"] is False


def test_ordered_absolute_quality_retention_is_a_policy_gate() -> None:
    arms = {
        "legacy": _arm(fid=20.0, kid=0.020),
        "ordered_uniform": _arm(fid=17.0, kid=0.017),
    }
    arms["ordered_uniform"]["quality_retention"]["valid"] = False
    result = _final_policy(selection=_selection(), arms=arms, calibration_margin=MARGIN)
    assert result["decision"] == "NO_GO"
    assert result["checks"]["ordered_absolute_quality_retained"] is False

    arms["ordered_uniform"]["quality_retention"]["valid"] = True
    arms["legacy"]["quality_retention"]["valid"] = False
    result = _final_policy(selection=_selection(), arms=arms, calibration_margin=MARGIN)
    assert result["decision"] == "NO_GO"
    assert result["checks"]["all_selected_arms_retain_150k_quality"] is False


def test_state_dict_exact_checks_keys_dtype_shape_and_bits() -> None:
    reference = {"weight": torch.tensor([1.0, 2.0]), "count": torch.tensor(3)}
    assert module._state_dict_exact(reference, copy.deepcopy(reference))
    assert not module._state_dict_exact(reference, {"weight": torch.tensor([1.0, 2.0])})
    assert not module._state_dict_exact(
        reference,
        {"weight": torch.tensor([1.0, 2.0]), "count": torch.tensor(3.0)},
    )
    changed = copy.deepcopy(reference)
    changed["weight"][1] = 3.0
    assert not module._state_dict_exact(reference, changed)


def _corner_summary(mean_raw_mse: float) -> dict:
    return {
        "count": 2_048,
        "nonfinite_rows": 0,
        "mean_raw_mse": mean_raw_mse,
        "mean_target_rms": 1.0,
        "mean_quotient_rms": 1.0,
        "coefficient": {"minimum": 1.0, "mean": 1.0, "maximum": 1.0},
    }


def _corner_result(final_error: float) -> dict:
    objective = {
        "stopped_evaluation": "fp32_dense",
        "emf_delta": 0.002,
        "emf_denominator_floor": 0.001,
    }

    def record(step: int, error: float) -> dict:
        return {
            "step": step,
            "fixed_exact_inference_corner": {
                "condition": {"t": 1.0, "r": 0.0, "h": 1.0},
                "sealed_train_only": True,
                "sample_count": 2_048,
                "objective_numerics": objective,
                "raw": _corner_summary(error),
                "ema": _corner_summary(error),
            },
        }

    return {
        "declared_profile": {
            "train": {"audit_samples": 2_048},
            "objective": objective,
        },
        "training": {"health": [record(250_000, 1.0), record(300_000, final_error)]},
    }


def test_final_exact_corner_growth_boundary_is_closed() -> None:
    accepted = module._fixed_exact_corner_trajectory(_corner_result(4.0))
    rejected = module._fixed_exact_corner_trajectory(_corner_result(4.0001))
    assert accepted["stable"] is True
    assert rejected["stable"] is False


def test_capability_gate_is_recomputed_from_final_ema_and_counters() -> None:
    final = {
        "second_moment_ratio": 1.0,
        "centered_variance_ratio": 1.0,
        "effective_rank_ratio": 1.0,
        "haar_LL_ratio": 1.0,
        "haar_LH_ratio": 1.0,
        "haar_HL_ratio": 1.0,
        "haar_HH_ratio": 1.0,
        "raw_saturation_fraction": 0.0,
    }
    result = {
        "training": {
            "health": [
                {"step": 250_000, "ema": {**final, "effective_rank_ratio": 0.9}},
                {"step": 300_000, "ema": final},
            ],
            "final_window_updates": 20_000,
            "clipped_updates_final_window": 0,
            "nonfinite_updates": 0,
            "inference_forward_calls": 1,
        }
    }
    declared = {"gate": asdict(module.CAPGateConfig())}
    gate = module._recomputed_capability_gate(result, declared)
    assert gate is not None
    assert gate["verdict"] == "PASS"
    assert gate["thresholds"]["clip_window_updates"] == 20_000


def test_training_cadence_makes_deleted_gate_evidence_detectable() -> None:
    declared = {
        "train": {
            "updates": 6,
            "log_every": 2,
            "health_every": 3,
            "checkpoint_updates": [2, 6],
        }
    }
    result = {
        "training": {
            "history": [{"step": 2}, {"step": 4}, {"step": 6}],
            "health": [
                {"step": 2, "ema": {}},
                {"step": 3},
                {"step": 6, "ema": {}},
            ],
        }
    }
    assert module._training_cadence_valid(result, declared)
    del result["training"]["health"][1]
    assert not module._training_cadence_valid(result, declared)


def test_recovery_result_reconciliation_maps_every_persisted_counter() -> None:
    training = {
        "optimizer_updates": 300_000,
        "examples_seen": 600_000,
        "objective_sample_evaluations": 2_400_000,
        "objective_forward_calls": 1_200_000,
        "clipped_updates": 2,
        "clipped_updates_final_window": 1,
        "final_window_updates": 20_000,
        "nonfinite_updates": 0,
        "history": [{"step": 300_000}],
        "health": [{"step": 300_000, "effective_rank_ratio": 0.9}],
    }
    result = {
        "realized_profile": {"name": "unit"},
        "training": training,
        "checkpoints": {"300000": {"raw": {}, "ema": {}}},
        "raw_snapshots": [{"step": 300_000}],
    }
    recovery = {
        "planned_updates": 300_000,
        "completed_updates": 300_000,
        "optimizer_updates": 300_000,
        "profile_name": "unit",
        "continuation_authorization": {"status": "authorization"},
        "examples_seen": 600_000,
        "model_forwards": 2_400_000,
        "objective_forward_calls": 1_200_000,
        "clipped_updates": 2,
        "clipped_updates_final_window": 1,
        "final_window_updates": 20_000,
        "nonfinite_updates": 0,
        "history": training["history"],
        "health": training["health"],
        "checkpoints": result["checkpoints"],
        "snapshots": [300_000],
        "best_rank_ratio": 0.9,
    }
    checks = module._recovery_result_checks(
        recovery,
        result,
        expected_updates=300_000,
        expected_authorization={"status": "authorization"},
    )
    assert all(checks.values())
    recovery["model_forwards"] += 1
    assert not module._recovery_result_checks(
        recovery,
        result,
        expected_updates=300_000,
        expected_authorization={"status": "authorization"},
    )["training_counters"]


def test_durable_recovery_record_is_bound_to_commit_and_version(tmp_path) -> None:
    # Keep paths short enough for the digest-named Windows recovery version.
    source = tmp_path / "s"
    mirror_root = tmp_path / "m"
    source.mkdir()
    mirror_root.mkdir()
    provision_root(
        mirror_root,
        storage_id="test-volume",
        attest_instance_independent=True,
    )
    recovery = source / "r.pt"
    digest = write_json_atomic(recovery, {"status": "test-recovery-bytes"})
    mirror = DurableMirror(source, mirror_root)
    record = mirror.mirror(recovery, mutable=True, recovery_step=150_000)
    assert module._durable_recovery_record_valid(
        record,
        mirror_root=mirror_root,
        relative_path="r.pt",
        recovery_step=150_000,
        expected_sha=digest,
    )

    changed = copy.deepcopy(record)
    changed["recovery_step"] = 149_999
    assert not module._durable_recovery_record_valid(
        changed,
        mirror_root=mirror_root,
        relative_path="r.pt",
        recovery_step=150_000,
        expected_sha=digest,
    )


def test_quality_retention_uses_closed_150k_and_strict_historical_boundaries() -> None:
    final_metrics = {
        module.FID_KEY: 12.0,
        module.KID_KEY: 0.022,
    }
    promotion = {"comparison": {"candidate": {"clean_fid": 10.0, "clean_kid": 0.020}}}
    preflight = {
        "inputs": {
            "baseline_standard": {
                "metrics": {module.FID_KEY: 14.0, module.KID_KEY: 0.024}
            }
        }
    }
    result = module._quality_retention(
        final_metrics=final_metrics,
        promotion=promotion,
        preflight=preflight,
        margin=MARGIN,
        require_historical_improvement=True,
    )
    assert result["checks"]["fid_retains_150k_quality_within_margin"] is True
    assert result["checks"]["kid_retains_150k_quality_within_margin"] is True
    assert result["checks"]["fid_beats_historical_baseline_beyond_margin"] is False
    assert result["checks"]["kid_beats_historical_baseline_beyond_margin"] is False


def test_final_artifact_revalidation_rejects_self_consistent_policy_tamper(
    tmp_path, monkeypatch
) -> None:
    selection = _selection()
    arms = {
        "legacy": _arm(fid=20.0, kid=0.020),
        "ordered_uniform": _arm(fid=17.0, kid=0.017),
    }
    context = {
        "selection": selection,
        "preflight": {"artifact_sha256": "8" * 64},
        "promotions": {
            "legacy": {"artifact_sha256": "6" * 64},
            "ordered_uniform": {"artifact_sha256": "7" * 64},
        },
        "arms": arms,
        "calibration_margin": copy.deepcopy(MARGIN),
        "unit_seed": 0,
    }
    monkeypatch.setattr(module, "_load_context", lambda *_args, **_kwargs: context)
    arm_paths = {
        arm: FinalArmArtifacts(
            result=tmp_path / f"{arm}-result.json",
            raw_checkpoint=tmp_path / f"{arm}-raw.pt",
            ema_checkpoint=tmp_path / f"{arm}-ema.pt",
            raw_readmission=tmp_path / f"{arm}-admission.json",
            evaluation=tmp_path / f"{arm}-evaluation.json",
            mirror_root=tmp_path / f"{arm}-mirror",
        )
        for arm in selection["selected_arms"]
    }
    path = tmp_path / "final.json"
    built = build_final_verdict(
        selection_path=tmp_path / "selection.json", arm_paths=arm_paths, out=path
    )
    assert built["decision"] == "GO"
    assert revalidate_final_verdict(path, require_go=True)["revalidated"] is True

    tampered_payload = json.loads(path.read_text(encoding="utf-8"))
    tampered_payload["decision"] = "NO_GO"
    tampered = tmp_path / "tampered.json"
    write_json_atomic(tampered, tampered_payload)
    with pytest.raises(RuntimeError, match="revalidation failed"):
        revalidate_final_verdict(tampered)


def test_cli_supports_independent_revalidation(monkeypatch, capsys, tmp_path) -> None:
    path = tmp_path / "final.json"
    monkeypatch.setattr(
        module,
        "revalidate_final_verdict",
        lambda candidate: (
            {
                "decision": "GO",
                "final_arm": "ordered_uniform",
                "claim": {"scope": "paired-300k-developmental"},
                "failed": [],
                "revalidated": True,
                "artifact_sha256": "a" * 64,
            }
            if candidate == path
            else None
        ),
    )
    monkeypatch.setattr(sys, "argv", ["final_verdict", "--revalidate", str(path)])
    assert module.main() == 0
    assert '"revalidated": true' in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["final_verdict", "--revalidate", str(path), "--out", "new.json"],
    )
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
