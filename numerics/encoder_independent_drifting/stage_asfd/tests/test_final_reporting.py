"""Regression tests for the final metric/geometry claim boundary."""

from __future__ import annotations

from ..final_report import (
    _auxiliary_noncollapse,
    _metric_comparison,
)


def test_tiny_metric_decrease_is_not_a_meaningful_improvement() -> None:
    result = _metric_comparison(10.0, 9.99, 0.05)
    assert result["asfd_lower"]
    assert result["noninferior_within_margin"]
    assert not result["improved_beyond_margin"]


def test_change_beyond_real_real_margin_is_detected() -> None:
    result = _metric_comparison(10.0, 9.8, 0.05)
    assert result["improved_beyond_margin"]


def test_metric_win_cannot_override_auxiliary_collapse() -> None:
    collapsed = {
        "precision": 0.99,
        "recall": 0.0,
        "pr_f1": 0.0,
        "exact_copy_fraction": 0.0,
        "generated_duplicate_fraction": 0.0,
    }
    assert not _auxiliary_noncollapse(collapsed)
