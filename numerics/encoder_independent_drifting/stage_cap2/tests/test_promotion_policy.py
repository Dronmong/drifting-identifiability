"""Focused tests for CAP2 metric-margin and auxiliary-gate policy."""

from __future__ import annotations

import math

from ..promotion import FID_KEY, KID_KEY, _calibration_tolerance, _control_continuation


def _calibration() -> dict:
    return {
        "metrics": {
            "direct_disjoint_pair": {
                "clean_fid": 2.25,
                # Unbiased KID estimates can be negative; the policy margin is
                # the magnitude of the serialized direct observation.
                "clean_kid": -0.003,
            },
            "matched_published_train_reference": {
                "left": {FID_KEY: 0.50, KID_KEY: 0.100},
                "right": {FID_KEY: 0.51, KID_KEY: 0.101},
            },
        }
    }


def test_promotion_margin_is_direct_real_real_not_correlated_score_delta() -> None:
    calibration = _calibration()
    assert _calibration_tolerance(calibration, FID_KEY) == 2.25
    assert _calibration_tolerance(calibration, KID_KEY) == 0.003

    # Tampering with the matched/full-reference sanity scores cannot weaken or
    # strengthen the policy margin.
    calibration["metrics"]["matched_published_train_reference"]["right"].update(
        {FID_KEY: 500.0, KID_KEY: -500.0}
    )
    assert _calibration_tolerance(calibration, FID_KEY) == 2.25
    assert _calibration_tolerance(calibration, KID_KEY) == 0.003


def test_promotion_margin_fails_closed_without_direct_observation() -> None:
    calibration = _calibration()
    del calibration["metrics"]["direct_disjoint_pair"]["clean_fid"]
    assert math.isnan(_calibration_tolerance(calibration, FID_KEY))
    assert math.isnan(_calibration_tolerance(calibration, "unknown_metric"))


def test_legacy_control_exempts_only_historical_quality_improvement() -> None:
    checks = {
        "readmission_go": True,
        "train_only_gate": True,
        "candidate_fid_improves_beyond_calibration": False,
        "candidate_kid_improves_beyond_calibration": False,
    }
    assert _control_continuation(checks)["decision"] == "GO"
    checks["readmission_go"] = False
    report = _control_continuation(checks)
    assert report["decision"] == "NO_GO"
    assert report["failed"] == ["readmission_go"]
