"""Unit checks for the cadence-adjusted CAP2 cost projection."""

from __future__ import annotations

from ..benchmark import project_runtime


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
