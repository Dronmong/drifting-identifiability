"""Regression tests for the corrected K=200 confirmation gate."""

from __future__ import annotations

from ..f1_k200 import (
    REGIMES,
    UNIT_IDS,
    decide,
    evaluate_veto,
    frozen_config,
    reference_side_interval,
)
from .harness import main


THRESHOLDS = {
    "nearest_bank_normalized": 0.4,
    "distinct_bank": 100.0,
    "effective_rank": 6.0,
    "one_minus_duplicate_rate": 0.9,
    "nn_diversity": 0.1,
}


def _terminal(recall: float) -> dict:
    return {"recall": recall, "effective_rank": 8.0,
            "duplicate_rate": 0.0, "nn_diversity": 0.5}


def _rows(control: float = 0.7, primary: float = 0.1,
          veto: bool = True) -> list[dict]:
    rows = []
    for unit in UNIT_IDS:
        for regime in REGIMES:
            rows.append({"unit": unit, "arm": "real_data", "regime": regime,
                         "terminal": _terminal(control), "veto": None})
            rows.append({
                "unit": unit, "arm": "random_generator", "regime": regime,
                "terminal": _terminal(primary), "veto": {"passes": veto}})
    return rows


def test_gate_requires_the_full_per_unit_conjunction() -> None:
    """1. recall, veto, and paired control must pass in the same unit"""
    assert all(item["decision"] == "PASS"
               for item in decide(_rows())["per_regime"].values())
    assert all(item["decision"] == "FAIL"
               for item in decide(_rows(primary=0.01))["per_regime"].values())
    assert all(item["decision"] == "FAIL"
               for item in decide(_rows(veto=False))["per_regime"].values())


def test_two_failed_controls_void_each_regime() -> None:
    """2. invalid controls void rather than count as primary failures"""
    rows = _rows()
    for row in rows:
        if row["arm"] == "real_data" and row["unit"] in UNIT_IDS[:2]:
            row["terminal"]["recall"] = 0.2
    result = decide(rows)
    assert all(item["decision"] == "VOID"
               for item in result["per_regime"].values())


def test_veto_values_are_compared_not_merely_present() -> None:
    """3. a below-threshold statistic makes the veto fail"""
    terminal = _terminal(0.2)
    replay = {"nearest_bank_normalized": 0.8, "distinct_bank": 200}
    passed = evaluate_veto("replay", terminal, replay, THRESHOLDS, 0.3)
    assert passed["passes"]
    terminal["effective_rank"] = 2.0
    failed = evaluate_veto("replay", terminal, replay, THRESHOLDS, 0.3)
    assert not failed["passes"]
    assert not failed["comparisons"]["effective_rank"]["passes"]


def test_stochastic_veto_uses_its_own_training_reference_threshold() -> None:
    """4. stochastic does not reuse replay distinct-bank semantics"""
    measured = {"nearest_train_normalized": 0.5}
    result = evaluate_veto(
        "stochastic", _terminal(0.2), measured, THRESHOLDS, 0.6)
    assert not result["passes"]
    assert "nearest_train_normalized" in result["comparisons"]
    assert "distinct_bank" not in result["comparisons"]


def test_reference_interval_is_labelled_and_contains_the_point() -> None:
    """5. the replacement interval is reference-side only and coherent"""
    result = reference_side_interval(10 / 2048)
    low, high = result["interval"]
    assert low <= 10 / 2048 <= high
    assert result["covered"] == 10
    assert "reference-side" in result["scope"]


def test_frozen_design_is_exactly_k200_and_new_seeded() -> None:
    """6. the confirmation cannot silently inherit K=20,000"""
    config = frozen_config()
    assert config["checkpoints"][-1] == 200
    assert config["unit_ids"] == [100, 101, 102]
    assert config["arms"] == ["real_data", "random_generator"]


if __name__ == "__main__":
    main(__name__, globals())

