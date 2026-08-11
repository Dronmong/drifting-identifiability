from __future__ import annotations

import copy

import pytest

from ..budget import GIB, build_budget_plan, build_storage_plan, revalidate_budget_plan


def _benchmark() -> dict:
    return {
        "hourly_rate": 0.5,
        "recovery_bytes": 600 * 1024**2,
        "checkpoint_artifact_bytes": {
            "raw": 150 * 1024**2,
            "ema": 150 * 1024**2,
        },
        "snapshot": {"bytes": 150 * 1024**2},
        "projections": {
            "150000": {
                "cost_at_declared_rate": 3.0,
                "conservative_raw_loop_upper_cost": 4.0,
                "event_counts": {
                    "recovery": 30,
                    "checkpoint_pair": 3,
                    "snapshot": 6,
                },
            },
            "300000": {
                "cost_at_declared_rate": 6.0,
                "conservative_raw_loop_upper_cost": 8.0,
                "event_counts": {
                    "recovery": 60,
                    "checkpoint_pair": 6,
                    "snapshot": 12,
                },
            },
            "750000": {
                "cost_at_declared_rate": 15.0,
                "conservative_raw_loop_upper_cost": 20.0,
                "event_counts": {
                    "recovery": 150,
                    "checkpoint_pair": 15,
                    "snapshot": 30,
                },
            },
        },
    }


def test_budget_prices_the_complete_worst_case_schedule() -> None:
    plan = build_budget_plan(
        _benchmark(),
        max_total_cost=25.0,
        nontraining_reserve=2.0,
        contingency_fraction=0.1,
    )
    assert plan["estimated_training_cost"] == 15.0
    assert plan["conservative_training_cost"] == 20.0
    assert plan["authorized_upper_cost"] == 24.0
    assert plan["within_ceiling"] is True
    assert revalidate_budget_plan(plan, _benchmark()) == plan


def test_budget_fails_closed_and_detects_tampering() -> None:
    plan = build_budget_plan(
        _benchmark(),
        max_total_cost=23.0,
        nontraining_reserve=2.0,
        contingency_fraction=0.1,
    )
    assert plan["within_ceiling"] is False
    tampered = copy.deepcopy(plan)
    tampered["authorized_upper_cost"] = 1.0
    with pytest.raises(RuntimeError, match="does not match"):
        revalidate_budget_plan(tampered, _benchmark())


def test_budget_prices_one_ordered_foundation_without_hidden_arms() -> None:
    plan = build_budget_plan(
        _benchmark(),
        max_total_cost=50.0,
        nontraining_reserve=5.0,
        contingency_fraction=0.15,
        campaign="ordered_750_foundation",
    )
    assert plan["campaign"] == "ordered_750_foundation"
    assert plan["schedule"]["formula"] == "C750"
    assert plan["schedule"]["total_update_equivalents"] == 750_000
    assert plan["estimated_training_cost"] == 15.0
    assert plan["conservative_training_cost"] == 20.0
    assert plan["post_foundation_training_reserve"] == 10.0
    assert plan["authorized_upper_cost"] == 38.0
    assert plan["within_ceiling"] is True


def test_ordered_foundation_reserve_is_explicit_and_revalidated() -> None:
    plan = build_budget_plan(
        _benchmark(),
        max_total_cost=75.0,
        nontraining_reserve=5.0,
        contingency_fraction=0.15,
        campaign="ordered_750_foundation",
        post_foundation_training_reserve=25.0,
    )
    assert plan["post_foundation_training_reserve"] == 25.0
    assert plan["authorized_upper_cost"] == 53.0
    assert revalidate_budget_plan(plan, _benchmark()) == plan

    tampered = copy.deepcopy(plan)
    tampered["post_foundation_training_reserve"] = 5.0
    with pytest.raises(RuntimeError, match="does not match"):
        revalidate_budget_plan(tampered, _benchmark())


def test_storage_plan_prices_immutable_history_and_fails_closed() -> None:
    plan = build_storage_plan(
        _benchmark(),
        storage_root="X:/",
        total_bytes=200 * GIB,
        free_bytes=190 * GIB,
        artifact_reserve_gib=20.0,
        contingency_fraction=0.20,
    )
    assert plan["schedule"]["recovery_commits"] == 150
    assert plan["schedule"]["checkpoint_events"] == 15
    assert plan["schedule"]["snapshot_events"] == 30
    assert plan["schedule"]["asfd_continuation_included"] is False
    assert plan["projected_bytes"]["asfd_feature_banks"] == 0
    assert plan["projected_bytes"]["recovery_history"] == 150 * 600 * 1024**2
    assert plan["decision"] == "GO"

    too_small = build_storage_plan(
        _benchmark(),
        storage_root="X:/",
        total_bytes=100 * GIB,
        free_bytes=90 * GIB,
        artifact_reserve_gib=20.0,
        contingency_fraction=0.20,
    )
    assert too_small["decision"] == "NO_GO"
    assert too_small["within_total_capacity"] is False


def test_foundation_storage_uses_the_single_750k_event_ledger() -> None:
    plan = build_storage_plan(
        _benchmark(),
        campaign="ordered_750_foundation",
        storage_root="X:/",
        total_bytes=200 * GIB,
        free_bytes=190 * GIB,
        artifact_reserve_gib=20.0,
        contingency_fraction=0.20,
    )
    assert plan["campaign"] == "ordered_750_foundation"
    assert plan["schedule"]["recovery_commits"] == 160
    assert plan["schedule"]["checkpoint_events"] == 16
    assert plan["schedule"]["snapshot_events"] == 35
    assert plan["schedule"]["asfd_continuation_included"] is True
    assert plan["projected_bytes"]["asfd_feature_banks"] == 20 * GIB
    assert plan["projected_bytes"]["recovery_history"] == 160 * 600 * 1024**2
