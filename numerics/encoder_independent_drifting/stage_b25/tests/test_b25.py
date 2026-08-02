"""Regression tests for the prospective B2.5 factorial."""

from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from ...b1 import b1_config
from ...f3b import profile
from ...stage_b2.core import b2_config
from ...tests.harness import main
from ..core import B25_ARMS, B25Config, train_b25_arm
from ..evaluation import (
    adjudicate_development,
    density_coverage,
    evaluation_allocation,
    factorial_interaction,
    paired_precision_recall_intervals,
)
from ..source_pool import _load_exclusions


def _stage() -> B25Config:
    result = B25Config(
        units=(500,),
        checkpoint_steps=(2, 4),
        diagnostic_steps=(2, 4),
        final_step=4,
        bootstrap_replicates=100,
        unit_wins_required=1,
    )
    result.validate()
    return result


def _b1():
    base = b1_config()
    anchor = replace(
        base.anchor,
        features=8,
        audit_features=8,
        refresh_every=2,
        refresh_fraction=0.5,
    )
    result = replace(
        base,
        anchor_every=1,
        anchor_batch=4,
        anchor_nfe=1,
        audit_features=8,
        audit_replicates=2,
        refresh_every_events=2,
        refresh_fraction=0.5,
        anchor_paired_wins_required=1,
        anchor=anchor,
    )
    result.validate()
    return result


def _b2():
    result = replace(
        b2_config(),
        correction_every=1,
        probe_batch=4,
        positive_batch=4,
        negative_batch=4,
        correction_nfe=1,
        ess_samples=8,
        ess_iterations=8,
        audit_batch=4,
        audit_replicates=2,
        drift_paired_wins_required=1,
    )
    result.validate()
    return result


def _pool() -> torch.Tensor:
    return torch.rand(24, 3, 8, 8, generator=torch.Generator().manual_seed(77)) * 2 - 1


def test_true_factorial_is_full_dose_and_flow_paired() -> None:
    selected = profile("smoke")
    rows = {}
    for arm in B25_ARMS:
        outcome = train_b25_arm(
            _pool(),
            selected.model,
            selected.train,
            500,
            arm,
            "cpu",
            b1_scale=0.43,
            lambda_b1=0.93,
            tau_b2=7.08,
            lambda_b2=1.93e-4,
            b1_config=_b1(),
            b2_config=_b2(),
            stage_config=_stage(),
        )
        rows[arm] = outcome
    assert len({row.history[0]["flow_loss"] for row in rows.values()}) == 1
    assert rows["B1"].anchor_events == rows["B1B2"].anchor_events == 4
    assert rows["B2"].correction_events == rows["B1B2"].correction_events == 4
    assert rows["B0"].anchor_events == rows["B0"].correction_events == 0


def test_combined_gradient_diagnostics_contain_every_component() -> None:
    selected = profile("smoke")
    outcome = train_b25_arm(
        _pool(),
        selected.model,
        selected.train,
        500,
        "B1B2",
        "cpu",
        b1_scale=0.43,
        lambda_b1=0.93,
        tau_b2=7.08,
        lambda_b2=1.93e-4,
        b1_config=_b1(),
        b2_config=_b2(),
        stage_config=_stage(),
    )
    for row in outcome.component_gradient_diagnostics:
        assert set(row["weighted_component_norms_pre_clip"]) == {
            "flow",
            "b1_weighted",
            "b2_weighted",
        }
        assert set(row["pairwise_cosines"]) == {
            "flow_vs_b1_weighted",
            "flow_vs_b2_weighted",
            "b1_weighted_vs_b2_weighted",
        }


def test_evaluation_allocation_is_disjoint_and_checks_capacity() -> None:
    allocation = evaluation_allocation(
        100,
        "synthetic-b25",
        units=1,
        generated_samples=8,
        reference_samples=16,
        audit_replicates=2,
        audit_batch=4,
    )
    allocation.assert_disjoint()
    assert len(allocation.unused) == 52
    try:
        evaluation_allocation(
            47,
            "too-small",
            units=1,
            generated_samples=8,
            reference_samples=16,
            audit_replicates=2,
            audit_batch=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("B2.5 accepted an undersized evaluation source")


def test_density_coverage_and_paired_membership_interval_sanity() -> None:
    real = np.arange(60, dtype=float).reshape(20, 3) / 10
    generated = real[:10] + 1e-5
    report = density_coverage(generated, real, k=3)
    assert report["density"] > 0
    assert report["coverage"] > 0
    interval = paired_precision_recall_intervals(
        generated,
        generated.copy(),
        real,
        seed=11,
        replicates=100,
        k=3,
    )
    assert interval["precision"]["difference"] == 0
    assert interval["recall"]["difference"] == 0


def test_factorial_interaction_and_development_adjudication() -> None:
    assert factorial_interaction({"B0": 4.0, "B1": 3.0, "B2": 2.0, "B1B2": 1.0}) == 0
    config = _stage()
    rows = []
    values = {
        "B0": (10.0, 10.0, 0.50, 0.50),
        "B1": (9.0, 9.0, 0.55, 0.55),
        "B2": (5.0, 6.0, 0.50, 0.50),
        "B1B2": (5.5, 9.0, 0.54, 0.54),
    }
    for arm, (energy, rank, precision, recall) in values.items():
        rows.append(
            {
                "unit": 500,
                "arm": arm,
                "drift_summary": {"raw_energy_mean": energy},
                "metrics": {
                    "effective_rank": rank,
                    "precision": precision,
                    "recall": recall,
                },
            }
        )
    verdict = adjudicate_development(rows, config)
    assert verdict["promising"]


def test_exclusion_loader_binds_paths_and_pixels() -> None:
    images = np.random.default_rng(19).integers(
        0, 256, size=(4, 32, 32, 3), dtype=np.uint8
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "prior.npz"
        np.savez(
            path,
            images=images,
            source_paths=np.asarray([f"source/{index}.png" for index in range(4)]),
        )
        paths, pixels, records = _load_exclusions([path])
    assert len(paths) == len(pixels) == records[0]["samples"] == 4


if __name__ == "__main__":
    main(__name__, globals())
