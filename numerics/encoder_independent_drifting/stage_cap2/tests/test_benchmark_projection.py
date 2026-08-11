"""Unit checks for the cadence-adjusted CAP2 cost projection."""

from __future__ import annotations

from ..benchmark import project_runtime, resume_rehearsal_consistent


def test_projection_counts_health_and_artifact_events_at_real_cadence():
    result = project_runtime(
        updates=150_000,
        recovery_every=5_000,
        snapshot_every=25_000,
        health_every=2_000,
        non_io_seconds_per_update=1.0,
        raw_upper_seconds_per_update=2.0,
        recovery_event_seconds=10.0,
        snapshot_event_seconds=20.0,
        checkpoint_pair_seconds=30.0,
        ordinary_health_event_seconds=40.0,
        checkpoint_health_event_seconds=50.0,
        hourly_rate=1.0,
    )
    assert result["event_counts"] == {
        "recovery": 30,
        "snapshot": 6,
        "checkpoint_pair": 3,
        "ordinary_health": 72,
        "checkpoint_health": 3,
    }
    expected_seconds = 150_000 + 30 * 10 + 6 * 20 + 3 * 30 + 72 * 40 + 3 * 50
    assert result["hours"] == expected_seconds / 3_600
    assert result["conservative_raw_loop_upper_hours"] == 300_000 / 3_600


def _resume_record() -> dict:
    return {
        "split_step": 1_999,
        "final_step": 2_000,
        "resumed_updates": 1,
        "first_device": "cuda:0",
        "second_device": "cuda:0",
        "resume_message": "resumed CAP-EMF-1 from update 1999",
        "before_recovery_sha256": "a" * 64,
        "after_recovery_sha256": "b" * 64,
        "before_resume": {
            "completed_updates": 1_999,
            "optimizer_updates": 1_999,
            "ema_updates": 1_999,
            "nonfinite_updates": 0,
            "optimizer_steps": {"count": 10, "minimum": 1_999, "maximum": 1_999},
        },
        "after_resume": {
            "completed_updates": 2_000,
            "optimizer_updates": 2_000,
            "ema_updates": 2_000,
            "nonfinite_updates": 0,
            "optimizer_steps": {"count": 10, "minimum": 2_000, "maximum": 2_000},
        },
    }


def test_resume_rehearsal_requires_post_reload_optimizer_and_ema_updates():
    record = _resume_record()
    assert resume_rehearsal_consistent(record, expected_steps=2_000)

    record["after_resume"]["ema_updates"] = 1_999
    assert not resume_rehearsal_consistent(record, expected_steps=2_000)


def test_resume_rehearsal_rejects_fresh_optimizer_disguised_as_resume():
    record = _resume_record()
    record["after_resume"]["optimizer_steps"] = {
        "count": 10,
        "minimum": 1,
        "maximum": 1,
    }
    assert not resume_rehearsal_consistent(record, expected_steps=2_000)
