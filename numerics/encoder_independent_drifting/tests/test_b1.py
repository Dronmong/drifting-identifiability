"""Regression tests for the repaired paired Stage-B1 protocol."""

from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import torch

from ..b1 import (
    B1Config,
    anchor_target_batch,
    anchor_term,
    b1_config,
    b1_streams,
    build_bank_for_dimension,
    calibrated_event_lambda,
    config_payload,
    evaluation_prior_seed,
    parameter_gradient_norm,
    train_b1,
)
from ..b1_evaluation import (
    adjudicate_b1,
    compare_anchor_audits,
    evaluation_allocation,
)
from ..b1_freeze import (
    DEFAULT_B0_RESULT,
    PROTOCOL,
    file_sha256,
    frozen_payload,
    load_adopted_b0,
    load_freeze,
    source_manifest,
)
from ..config import AnchorConfig
from ..diagnostics import write_json
from ..f3b import TimeConditionedUNet, profile, train_bridge
from ..spectral_anchor import anchor_gradient, anchor_loss, refresh_bank
from .harness import main


def _small_config() -> B1Config:
    anchor = AnchorConfig(
        features=6,
        audit_features=6,
        refresh_every=1,
        refresh_fraction=0.5,
        band_schedule="coarse_to_fine",
    )
    result = B1Config(
        anchor_every=1,
        anchor_batch=4,
        anchor_nfe=1,
        scale_probe_samples=8,
        audit_features=6,
        audit_replicates=2,
        refresh_every_events=1,
        refresh_fraction=0.5,
        event_gradient_ratio=0.25,
        anchor_paired_wins_required=1,
        anchor=anchor,
    )
    result.validate()
    return result


def _pool(count: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(123)
    return torch.rand(count, 3, 8, 8, generator=generator) * 2 - 1


def test_anchor_moves_immutable_cpu_bank_to_sample_device() -> None:
    """1. CPU-owned banks evaluate and differentiate on the sample device."""
    config = _small_config()
    bank = build_bank_for_dimension(1.0, 5, "test", 0, "bank", 6, config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    left = torch.randn(8, 5, generator=torch.Generator().manual_seed(1)).to(device)
    right = torch.randn(9, 5, generator=torch.Generator().manual_seed(2)).to(device)
    left.requires_grad_(True)
    value = anchor_loss(bank, left, right, "biased")
    value.backward()
    analytic = anchor_gradient(bank, left.detach(), right)
    assert value.device.type == device.type
    assert analytic.device.type == device.type
    assert torch.isfinite(value)
    assert float((left.grad - analytic).abs().max()) < 1e-7
    refreshed = refresh_bank(bank, 0.5, 77)
    assert refreshed.frequencies.device.type == "cpu"


def test_cpu_and_cuda_anchor_agree_when_cuda_exists() -> None:
    """2. CPU/CUDA use identical banks and stochastic arrays."""
    if not torch.cuda.is_available():
        return
    config = _small_config()
    bank = build_bank_for_dimension(1.0, 7, "test", 1, "bank", 6, config)
    left = torch.randn(10, 7, generator=torch.Generator().manual_seed(3))
    right = torch.randn(11, 7, generator=torch.Generator().manual_seed(4))
    cpu = anchor_loss(bank, left, right, "biased")
    gpu = anchor_loss(bank, left.cuda(), right.cuda(), "biased").cpu()
    assert torch.allclose(cpu, gpu, atol=1e-10, rtol=1e-10), (cpu, gpu)


def test_gradient_calibration_hits_declared_event_ratio() -> None:
    """3. lambda is calibrated by gradient norms, not fixed by intuition."""
    config = b1_config()
    value = calibrated_event_lambda(4.0, 10.0, config)
    assert abs(value * 10.0 / 4.0 - config.event_gradient_ratio) < 1e-15
    assert config.effective_gradient_ratio == 0.025


def test_anchor_term_reaches_model_parameters_through_euler() -> None:
    """4. the anchor is not detached from the integrated model output."""
    config = _small_config()
    selected = profile("smoke")
    model = TimeConditionedUNet(selected.model, seed=11)
    dimension = selected.model.channels * selected.model.image_size**2
    bank = build_bank_for_dimension(1.0, dimension, "test", 2, "bank", 6, config)
    real = _pool()[: config.anchor_batch]
    prior = torch.Generator().manual_seed(19)
    value, _ = anchor_term(
        model,
        bank,
        real,
        selected.model,
        prior,
        "cpu",
        0.5,
        config,
    )
    assert parameter_gradient_norm(value, model) > 0


def test_b1_reuses_b0_initialization_and_first_flow_batch() -> None:
    """5. paired B0/B1 differ first at the declared anchor objective."""
    config = _small_config()
    selected = profile("smoke")
    train_config = replace(
        selected.train,
        steps=1,
        checkpoint_steps=(1,),
        log_every=1,
    )
    pool = _pool()
    b0 = train_bridge(pool, selected.model, train_config, "confirmation", 300, "cpu")
    b1 = train_b1(
        pool,
        selected.model,
        train_config,
        300,
        "cpu",
        1.0,
        0.1,
        config,
    )
    assert b0.history[0]["loss"] == b1.history[0]["flow_loss"]
    assert b1.anchor_events == 1
    assert b1.anchor_refreshes == 1
    assert b1.history[0]["loss"] != b1.history[0]["flow_loss"]


def test_b1_allocation_is_fresh_and_disjoint() -> None:
    """6. calibration uses B0-unused data and confirmation uses test data."""
    allocation = evaluation_allocation(b1_config())
    allocation.assert_disjoint()
    assert len(allocation.calibration_reference) == 512
    assert len(allocation.confirmation_reference) == 2_048
    assert len(allocation.audit_target_pairs) == 6
    assert len(allocation.confirmation_unused) == 272


def _audit_rows(excesses: list[float], floor: float = 0.2) -> list[dict]:
    return [
        {
            "replicate": index,
            "bank_seed": 200 + index,
            "prior_seed": 100 + index,
            "real_real_biased": floor,
            "biased_excess_over_real": excess,
        }
        for index, excess in enumerate(excesses)
    ]


def test_anchor_gate_uses_biased_excess_and_common_randomness() -> None:
    """7. negative U-statistics cannot corrupt the frozen reduction gate."""
    config = _small_config()
    baseline = _audit_rows([1.0, 0.8])
    candidate = _audit_rows([0.3, 0.2])
    result = compare_anchor_audits(candidate, baseline, config)
    assert result["passes"]
    altered = _audit_rows([0.3, 0.2], floor=0.21)
    try:
        compare_anchor_audits(altered, baseline, config)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a different matched-real audit floor")


def test_b1_gate_is_per_unit_and_requires_controls() -> None:
    """8. medians cannot hide a failed paired unit or invalid control."""
    config = b1_config()
    baseline = [{"unit": unit, "metrics": {"recall": 0.20}} for unit in (300, 301, 302)]
    rows = [
        {
            "unit": unit,
            "assigned_control_group": index,
            "metrics": {"recall": recall},
            "anchor_comparison": {"passes": anchor},
            "veto": {"passes": True},
        }
        for index, (unit, recall, anchor) in enumerate(
            ((300, 0.16, True), (301, 0.16, True), (302, 0.10, False))
        )
    ]
    controls = [{"group": index, "recall": 0.7} for index in range(3)]
    assert adjudicate_b1(rows, baseline, controls, config)["decision"] == "PASS"
    controls[1]["recall"] = 0.4
    assert adjudicate_b1(rows, baseline, controls, config)["decision"] == "VOID"


def test_metric_prior_is_common_but_distinct_across_units() -> None:
    """9. each paired arm shares a prior without sharing it across units."""
    values = [evaluation_prior_seed(unit) for unit in (300, 301, 302)]
    assert len(set(values)) == 3
    assert values == [evaluation_prior_seed(unit) for unit in (300, 301, 302)]


def test_b1_config_survives_json_artifact_roundtrip() -> None:
    """10. nested band tuples cannot invalidate a freshly written freeze."""
    payload = config_payload(b1_config())
    assert json.loads(json.dumps(payload)) == payload
    assert isinstance(payload["anchor"]["bands"], list)


def test_anchor_target_augmentation_has_its_own_replayable_stream() -> None:
    """11. anchor flips neither perturb nor borrow B0 flow randomness."""
    left = b1_streams("confirmation", 300)
    right = b1_streams("confirmation", 300)
    first = anchor_target_batch(_pool(), left, 8, True)
    replay = anchor_target_batch(_pool(), right, 8, True)
    assert torch.equal(first, replay)
    assert left.anchor_augmentation.initial_seed() != left.anchor_prior.initial_seed()


def test_b1_freeze_roundtrip_binds_b0_calibration_and_baseline() -> None:
    """12. source-bound prerequisites produce a reloadable immutable freeze."""
    if not DEFAULT_B0_RESULT.exists():
        return
    b0 = load_adopted_b0(DEFAULT_B0_RESULT)
    common = {
        "protocol_sha256": file_sha256(PROTOCOL),
        "source_sha256": source_manifest(),
        "b0_artifact_sha256": b0["artifact_sha256"],
        "allocation_digests": {"role": "fixed"},
        "b1_config": config_payload(b1_config()),
        "verdict": {"decision": "GO"},
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        calibration_path = root / "calibration.json"
        calibration_sha = write_json(
            calibration_path,
            common
            | {
                "status": "b1-calibration",
                "lambda_event": 0.1,
                "scale": 1.0,
                "scale_indices_digest": "fixed",
            },
        )
        baseline_path = root / "baseline.json"
        write_json(
            baseline_path,
            common
            | {
                "status": "b1-b0-paired-baseline",
                "calibration_sha256": calibration_sha,
            },
        )
        payload = frozen_payload(calibration_path, baseline_path, DEFAULT_B0_RESULT)
        freeze_path = root / "freeze.json"
        write_json(freeze_path, payload)
        replay = load_freeze(freeze_path)
        assert replay["lambda_event"] == 0.1
        assert replay["b1_config"] == config_payload(b1_config())


if __name__ == "__main__":
    main(__name__, globals())
