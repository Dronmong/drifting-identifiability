"""Policy tests for concurrent CAP2 sampler selection."""

from __future__ import annotations

from copy import deepcopy

from ..selection import _policy

CANDIDATE = "local_1000_d0002_fp32"
PREFLIGHT = "a" * 64
MARGIN = {
    "kind": "absolute left/right matched-reference discrepancy",
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


def test_selection_rejects_an_ordered_arm_that_does_not_beat_legacy() -> None:
    fixture = _fixture()
    for arm in ("ordered_logitnormal", "ordered_uniform"):
        fixture[arm]["comparison"]["candidate"].update(
            {"clean_fid": 9.9, "clean_kid": 0.020, "repository_kid": 0.031}
        )
    result = _policy(fixture)
    assert result["decision"] == "NO_GO"
    assert result["eligible_ordered_arms"] == []


def test_selection_requires_a_viable_concurrent_legacy_control() -> None:
    fixture = _fixture()
    fixture["legacy"]["decision"] = "NO_GO"
    result = _policy(fixture)
    assert result["decision"] == "NO_GO"
    assert result["checks"]["legacy_individual_promotion_go"] is False
