"""Mechanical tests for the expedited Stage S1 audit and staging rule."""

from __future__ import annotations

import copy

import torch

from ...tests.harness import main
from ..continuation import (
    CONTINUATION_PHASE,
    batch_sha256,
    continuation_flow_seed_manifest,
)
from ..core import SinkhornConfig
from ..s1 import _unit_gate, staged_verdict
from ..s1_evaluation import S1AuditConfig, compare_audits, field_energy
from ..s1_freeze import deterministic_unit_order


def _sinkhorn_config() -> SinkhornConfig:
    return SinkhornConfig(
        epsilon=0.2,
        relative_tolerance=1e-5,
        max_iterations=200,
        min_iterations=2,
        check_every=1,
        primary_batch=3,
        self_batch=3,
        real_batch=3,
    )


def _audit(energies: list[float]) -> list[dict]:
    return [
        {
            "replicate": index,
            "primary_seed": 100 + index,
            "self_seed": 200 + index,
            "energy": value,
            "real_real_floor": 0.1,
            "excess_over_real": value - 0.1,
        }
        for index, value in enumerate(energies)
    ]


def test_field_energy_zero_for_equal_distinct_supports() -> None:
    generator = torch.Generator().manual_seed(18)
    primary = torch.randn(3, 2, generator=generator)
    support = torch.randn(3, 2, generator=generator)
    value, health = field_energy(
        primary,
        support,
        support.clone(),
        1.0,
        _sinkhorn_config(),
    )
    assert value < 1e-12
    assert health["cross"]["converged"]
    assert health["self"]["converged"]


def test_paired_audit_gate_requires_wins_and_median_reduction() -> None:
    config = S1AuditConfig(
        batch=3,
        replicates=4,
        nfe=1,
        paired_wins_required=3,
        reduction_fraction=0.25,
    )
    baseline = _audit([4.0, 4.0, 4.0, 4.0])
    candidate = _audit([2.0, 2.5, 3.0, 5.0])
    comparison = compare_audits(candidate, baseline, config)
    assert comparison["paired_wins"] == 3
    assert comparison["median_reduction_passes"]
    assert comparison["passes"]

    unpaired = copy.deepcopy(candidate)
    unpaired[0]["primary_seed"] += 1
    try:
        compare_audits(unpaired, baseline, config)
    except ValueError:
        pass
    else:
        raise AssertionError("unpaired S1 audits must be rejected")


def test_two_unit_staging_rule_is_fixed() -> None:
    freeze = {"initial_units": [300, 301], "tiebreaker_unit": 302}
    both = [
        {"unit": 300, "gate": {"passes": True}},
        {"unit": 301, "gate": {"passes": True}},
    ]
    assert staged_verdict(both, freeze)["decision"] == "PROVISIONAL-GO"
    mixed = copy.deepcopy(both)
    mixed[1]["gate"]["passes"] = False
    assert staged_verdict(mixed, freeze)["decision"] == "RUN-TIEBREAKER-302"
    neither = copy.deepcopy(mixed)
    neither[0]["gate"]["passes"] = False
    assert staged_verdict(neither, freeze)["decision"] == "STOP"


def test_continuation_seed_and_batch_digests_are_stable() -> None:
    assert CONTINUATION_PHASE == "sinkhorn-s1-continuation-v2"
    first = continuation_flow_seed_manifest(300)
    assert first == continuation_flow_seed_manifest(300)
    assert first != continuation_flow_seed_manifest(301)
    assert len(set(first.values())) == len(first)

    generator = torch.Generator().manual_seed(99)
    tensors = tuple(torch.randn(3, 2, generator=generator) for _ in range(3))
    digest = batch_sha256(*tensors)
    assert digest == batch_sha256(*(value.clone() for value in tensors))
    changed = [value.clone() for value in tensors]
    changed[0][0, 0] += 1
    assert digest != batch_sha256(*changed)


def test_unit_order_is_deterministic_and_complete() -> None:
    first = deterministic_unit_order("frozen-s0")
    second = deterministic_unit_order("frozen-s0")
    assert first == second
    assert sorted(first) == [300, 301, 302]
    assert len(set(first)) == 3


def _gate_arm(
    *,
    pr_f1: float,
    recall: float,
    rank: float,
    correction_summary: dict | None = None,
) -> dict:
    return {
        "metrics": {
            "pr_f1": pr_f1,
            "recall": recall,
            "effective_rank": rank,
        },
        "veto": {"passes": True},
        "audit_solver_converged": True,
        "training": {"correction_summary": correction_summary or {}},
    }


def test_unit_gate_requires_quality_and_laplace_competitiveness() -> None:
    config = S1AuditConfig()
    arms = {
        "control": _gate_arm(pr_f1=0.20, recall=0.30, rank=10.0),
        "laplace": _gate_arm(pr_f1=0.24, recall=0.31, rank=9.0),
        "sinkhorn": _gate_arm(
            pr_f1=0.23,
            recall=0.29,
            rank=9.0,
            correction_summary={"cap_hits": 0, "maximum_relative_error": 1e-4},
        ),
    }
    arms["laplace"]["training"]["correction_summary"] = {
        "positive_row_sum_error_maximum": 1e-7,
        "negative_row_sum_error_maximum": 1e-7,
    }
    comparison = {"passes": True}
    assert _unit_gate(arms, comparison, config)["passes"]

    no_quality = copy.deepcopy(arms)
    no_quality["sinkhorn"]["metrics"]["pr_f1"] = 0.20
    failed = _unit_gate(no_quality, comparison, config)
    assert not failed["passes"]
    assert not failed["checks"]["pr_f1_improves_control"]

    not_competitive = copy.deepcopy(arms)
    not_competitive["sinkhorn"]["metrics"]["pr_f1"] = 0.21
    failed = _unit_gate(not_competitive, comparison, config)
    assert not failed["passes"]
    assert not failed["checks"]["laplace_pr_f1_competitive"]


if __name__ == "__main__":
    main(__name__, globals())
