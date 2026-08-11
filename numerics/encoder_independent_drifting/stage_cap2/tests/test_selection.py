"""Policy tests for concurrent CAP2 sampler selection."""

from __future__ import annotations

import json
import sys
from copy import deepcopy

import pytest

from .. import selection as selection_module
from ..selection import _policy

CANDIDATE = "local_1000_d0002_fp32"
PREFLIGHT = "a" * 64
MARGIN = {
    "kind": "absolute direct disjoint real/real discrepancy",
    "clean_fid": 0.2,
    "clean_kid": 0.001,
    "statistical_scope": "deterministic finite-sample margin; not a confidence interval",
}


def _record(
    arm: str,
    *,
    fid: float,
    kid: float,
    repository_kid: float,
    precision: float = 0.5,
    recall: float = 0.5,
    decision: str = "GO",
) -> dict:
    return {
        "arm": arm,
        "candidate": CANDIDATE,
        "preflight_sha256": PREFLIGHT,
        "decision": decision,
        "control_continuation": {
            "decision": "GO",
            "checks": {"integrity_and_health": True},
            "failed": [],
        },
        "revalidated": True,
        "comparison": {
            "candidate": {
                "clean_fid": fid,
                "clean_kid": kid,
                "repository_kid": repository_kid,
                "precision": precision,
                "recall": recall,
                "pr_f1": 2 * precision * recall / (precision + recall),
            },
            "calibration_margin": deepcopy(MARGIN),
        },
    }


def _fixture() -> dict[str, dict]:
    return {
        "legacy": _record("legacy", fid=10.0, kid=0.020, repository_kid=0.030),
        "ordered_logitnormal": _record(
            "ordered_logitnormal", fid=9.5, kid=0.017, repository_kid=0.025
        ),
        "ordered_uniform": _record(
            "ordered_uniform", fid=9.0, kid=0.015, repository_kid=0.020
        ),
    }


def test_selection_retains_legacy_and_margin_separated_ordered_winner() -> None:
    result = _policy(_fixture())
    assert result["decision"] == "GO"
    assert result["selected_arms"] == ["legacy", "ordered_uniform"]
    assert result["ordered_winner"] == "ordered_uniform"


def test_selection_does_not_break_a_calibrated_tie_on_auxiliary_kid() -> None:
    fixture = _fixture()
    fixture["ordered_logitnormal"]["comparison"]["candidate"].update(
        {"clean_fid": 9.05, "clean_kid": 0.0155, "repository_kid": 0.001}
    )
    result = _policy(fixture)
    assert result["decision"] == "NO_GO"
    assert result["selected_arms"] == []
    assert "indistinguishable" in result["tie_reason"]


def test_selection_auxiliary_regressions_neither_gate_nor_break_ties() -> None:
    fixture = _fixture()
    # These diagnostics remain present in promotion records, but selection is
    # deliberately a 50k CleanFID/CleanKID policy.  Make the standard-metric
    # winner arbitrarily poor on every auxiliary relative comparison.
    fixture["ordered_uniform"]["comparison"]["candidate"].update(
        {"repository_kid": 99.0, "precision": 0.1, "recall": 0.1}
    )
    result = _policy(fixture)
    assert result["decision"] == "GO"
    assert result["ordered_winner"] == "ordered_uniform"
    assert set(result["eligibility"]["ordered_uniform"]) == {
        "individual_promotion_go",
        "fid_beats_concurrent_legacy_beyond_margin",
        "kid_beats_concurrent_legacy_beyond_margin",
    }


def test_selection_rejects_an_ordered_arm_that_does_not_beat_legacy() -> None:
    fixture = _fixture()
    for arm in ("ordered_logitnormal", "ordered_uniform"):
        fixture[arm]["comparison"]["candidate"].update(
            {"clean_fid": 9.9, "clean_kid": 0.020, "repository_kid": 0.031}
        )
    result = _policy(fixture)
    assert result["decision"] == "NO_GO"
    assert result["eligible_ordered_arms"] == []


def test_selection_does_not_require_the_control_to_be_a_scientific_success() -> None:
    fixture = _fixture()
    fixture["legacy"]["decision"] = "NO_GO"
    result = _policy(fixture)
    assert result["decision"] == "GO"
    assert result["selected_arms"] == ["legacy", "ordered_uniform"]


def test_selection_requires_a_mechanically_viable_concurrent_control() -> None:
    fixture = _fixture()
    fixture["legacy"]["decision"] = "NO_GO"
    fixture["legacy"]["control_continuation"] = {
        "decision": "NO_GO",
        "checks": {"readmission": False},
        "failed": ["readmission"],
    }
    result = _policy(fixture)
    assert result["decision"] == "NO_GO"
    assert result["checks"]["legacy_control_continuation_go"] is False


def test_selection_rejects_stale_or_arm_specific_calibration_margins() -> None:
    stale = _fixture()
    stale["legacy"]["comparison"]["calibration_margin"]["kind"] = (
        "absolute left/right matched-reference discrepancy"
    )
    try:
        _policy(stale)
    except RuntimeError as error:
        assert "direct real/real" in str(error)
    else:  # pragma: no cover - guarded policy boundary
        raise AssertionError("stale matched-reference calibration was accepted")

    mismatched = _fixture()
    mismatched["ordered_uniform"]["comparison"]["calibration_margin"]["clean_fid"] += (
        0.01
    )
    try:
        _policy(mismatched)
    except RuntimeError as error:
        assert "different calibration margins" in str(error)
    else:  # pragma: no cover - guarded policy boundary
        raise AssertionError("arm-specific calibration tampering was accepted")


def test_selection_cli_revalidates_before_printing_winner(
    monkeypatch, capsys, tmp_path
) -> None:
    path = tmp_path / "selection.json"
    monkeypatch.setattr(
        selection_module,
        "revalidate_selection",
        lambda candidate: {
            "decision": "GO",
            "ordered_winner": "ordered_uniform",
            "selected_arms": ["legacy", "ordered_uniform"],
            "revalidated": candidate == path,
        },
    )
    monkeypatch.setattr(sys, "argv", ["selection", "--revalidate", str(path)])
    assert selection_module.main() == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["revalidated"] is True
    assert printed["ordered_winner"] == "ordered_uniform"

    monkeypatch.setattr(
        sys,
        "argv",
        ["selection", "--revalidate", str(path), "--out", str(tmp_path / "x")],
    )
    with pytest.raises(SystemExit) as error:
        selection_module.main()
    assert error.value.code == 2
