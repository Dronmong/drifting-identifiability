"""Predeclared aggregate cost envelope for the CAP2 paid screen."""

from __future__ import annotations

import math

GIB = 1024**3
ROLLING_RECOVERIES = 3
CAMPAIGNS = ("matched_screen", "ordered_750_foundation")
DEFAULT_POST_FOUNDATION_TRAINING_RESERVE = 10.0
ASFD_BANK_RESERVE_GIB = 20.0


def _finite_nonnegative(value: object, *, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def build_budget_plan(
    benchmark: dict,
    *,
    max_total_cost: float,
    nontraining_reserve: float,
    contingency_fraction: float,
    campaign: str = "matched_screen",
    post_foundation_training_reserve: float = DEFAULT_POST_FOUNDATION_TRAINING_RESERVE,
) -> dict[str, object]:
    """Price the frozen worst-case successive-halving schedule.

    The legacy matched screen sends three arms to 150k and two to 300k, for
    ``C_150 + 2*C_300``.  The capability-first campaign spends the training
    allocation on one ordered-uniform foundation through 750k, for ``C_750``.
    Both gate on the benchmark's conservative raw-loop projection, then add an
    explicit multiplicative contingency and a dollar reserve for provider
    startup, admission, transfer, and evaluation work that the loop benchmark
    cannot measure.
    """

    maximum = _finite_nonnegative(max_total_cost, name="maximum total cost")
    reserve = _finite_nonnegative(nontraining_reserve, name="nontraining reserve")
    contingency = _finite_nonnegative(contingency_fraction, name="contingency fraction")
    requested_post_foundation_reserve = _finite_nonnegative(
        post_foundation_training_reserve,
        name="post-foundation training reserve",
    )
    if maximum <= 0:
        raise ValueError("maximum total cost must be positive")
    if contingency > 1:
        raise ValueError("contingency fraction may not exceed one")
    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown CAP2 campaign {campaign!r}; expected {CAMPAIGNS}")
    projections = benchmark.get("projections")
    if not isinstance(projections, dict):
        raise TypeError("benchmark lacks runtime projections")

    def cost(horizon: str, key: str) -> float:
        projection = projections.get(horizon)
        if not isinstance(projection, dict):
            raise TypeError(f"benchmark lacks {horizon}-update projection")
        return _finite_nonnegative(projection.get(key), name=f"{horizon} {key}")

    if campaign == "matched_screen":
        estimated_training = cost("150000", "cost_at_declared_rate") + 2 * cost(
            "300000", "cost_at_declared_rate"
        )
        conservative_training = cost(
            "150000", "conservative_raw_loop_upper_cost"
        ) + 2 * cost("300000", "conservative_raw_loop_upper_cost")
        schedule = {
            "campaign": campaign,
            "arms_to_150k": 3,
            "arms_promoted_150k_to_300k": 2,
            "ordered_foundations_to_750k": 0,
            "total_update_equivalents": 750_000,
            "formula": "C150 + 2*C300",
        }
        scope = (
            "developmental three-arm 150k screen plus concurrent legacy and "
            "one ordered arm through 300k; no confirmation"
        )
    else:
        estimated_training = cost("750000", "cost_at_declared_rate")
        conservative_training = cost("750000", "conservative_raw_loop_upper_cost")
        schedule = {
            "campaign": campaign,
            "arms_to_150k": 0,
            "arms_promoted_150k_to_300k": 0,
            "ordered_foundations_to_750k": 1,
            "total_update_equivalents": 750_000,
            "formula": "C750",
        }
        scope = (
            "one ordered-uniform encoder-free foundation through 750k, with "
            "650k matched-horizon evidence; ASFD requires a separate gate"
        )
    contingency_cost = conservative_training * contingency
    post_foundation_reserve = (
        requested_post_foundation_reserve
        if campaign == "ordered_750_foundation"
        else 0.0
    )
    authorized_upper = (
        conservative_training + contingency_cost + reserve + post_foundation_reserve
    )
    return {
        "campaign": campaign,
        "schedule": schedule,
        "hourly_rate": float(benchmark.get("hourly_rate", math.nan)),
        "estimated_training_cost": estimated_training,
        "conservative_training_cost": conservative_training,
        "contingency_fraction": contingency,
        "contingency_cost": contingency_cost,
        "nontraining_reserve": reserve,
        "post_foundation_training_reserve": post_foundation_reserve,
        "authorized_upper_cost": authorized_upper,
        "max_total_cost": maximum,
        "within_ceiling": authorized_upper <= maximum,
        "scope": scope,
    }


def revalidate_budget_plan(plan: object, benchmark: dict) -> dict[str, object]:
    if not isinstance(plan, dict):
        raise TypeError("preflight lacks an aggregate budget plan")
    expected = build_budget_plan(
        benchmark,
        max_total_cost=plan.get("max_total_cost"),
        nontraining_reserve=plan.get("nontraining_reserve"),
        contingency_fraction=plan.get("contingency_fraction"),
        campaign=plan.get("campaign", "matched_screen"),
        post_foundation_training_reserve=plan.get(
            "post_foundation_training_reserve",
            DEFAULT_POST_FOUNDATION_TRAINING_RESERVE,
        ),
    )
    if plan != expected:
        raise RuntimeError("aggregate budget plan does not match the benchmark")
    return expected


def build_storage_plan(
    benchmark: dict,
    *,
    campaign: str = "matched_screen",
    storage_root: str,
    total_bytes: int,
    free_bytes: int,
    artifact_reserve_gib: float,
    contingency_fraction: float,
) -> dict[str, object]:
    """Project the full immutable-history footprint before paid training.

    Recovery versions dominate the footprint: every 5k commit is immutable in
    the second mirror.  Checkpoints and snapshots exist both in the common
    workspace and in that mirror.  The explicit artifact reserve covers PNGs,
    retained feature populations, JSON ledgers, previews, filesystem overhead,
    and the evaluation workspace rather than pretending those are free.
    """

    if (
        not isinstance(total_bytes, int)
        or isinstance(total_bytes, bool)
        or total_bytes <= 0
    ):
        raise ValueError("durable total bytes must be a positive integer")
    if (
        not isinstance(free_bytes, int)
        or isinstance(free_bytes, bool)
        or free_bytes < 0
    ):
        raise ValueError("durable free bytes must be a nonnegative integer")
    if free_bytes > total_bytes:
        raise ValueError("durable free bytes exceed total capacity")
    reserve_gib = _finite_nonnegative(
        artifact_reserve_gib, name="artifact storage reserve GiB"
    )
    contingency = _finite_nonnegative(
        contingency_fraction, name="storage contingency fraction"
    )
    if contingency > 1:
        raise ValueError("storage contingency fraction may not exceed one")
    recovery_bytes = int(benchmark.get("recovery_bytes", 0))
    snapshot = benchmark.get("snapshot")
    checkpoint = benchmark.get("checkpoint_artifact_bytes")
    if (
        recovery_bytes <= 0
        or not isinstance(snapshot, dict)
        or not isinstance(checkpoint, dict)
    ):
        raise RuntimeError("benchmark lacks measured artifact sizes")
    snapshot_bytes = int(snapshot.get("bytes", 0))
    checkpoint_raw_bytes = int(checkpoint.get("raw", 0))
    checkpoint_ema_bytes = int(checkpoint.get("ema", 0))
    if min(snapshot_bytes, checkpoint_raw_bytes, checkpoint_ema_bytes) <= 0:
        raise RuntimeError("benchmark artifact sizes must be positive")

    if campaign not in CAMPAIGNS:
        raise ValueError(f"unknown CAP2 campaign {campaign!r}; expected {CAMPAIGNS}")
    projections = benchmark.get("projections")
    if not isinstance(projections, dict):
        raise TypeError("benchmark lacks storage event projections")

    def event_count(horizon: str, name: str) -> int:
        projection = projections.get(horizon)
        counts = (
            projection.get("event_counts") if isinstance(projection, dict) else None
        )
        value = counts.get(name) if isinstance(counts, dict) else None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"benchmark lacks a valid {horizon} {name} event count")
        return value

    def campaign_events(name: str) -> int:
        if campaign == "ordered_750_foundation":
            return event_count("750000", name)
        values = []
        for horizon in ("150000", "300000"):
            values.append(event_count(horizon, name))
        # Three arms to 150k and two continuing to 300k equals one cumulative
        # 150k schedule plus two cumulative 300k schedules.
        return values[0] + 2 * values[1]

    recovery_commits = campaign_events("recovery")
    checkpoint_events = campaign_events("checkpoint_pair")
    snapshot_events = campaign_events("snapshot")
    asfd_bank_reserve_bytes = 0
    if campaign == "ordered_750_foundation":
        # The 50k ASFD continuation publishes ten 5k recovery commits, one
        # final checkpoint pair and five 10k raw snapshots.  Its four-view,
        # four-level train/fresh positive/probe banks are reserved separately.
        recovery_commits += 10
        checkpoint_events += 1
        snapshot_events += 5
        asfd_bank_reserve_bytes = math.ceil(ASFD_BANK_RESERVE_GIB * GIB)

    recovery_history_bytes = recovery_commits * recovery_bytes
    rolling_recovery_bytes = ROLLING_RECOVERIES * recovery_bytes
    checkpoint_bytes = (
        checkpoint_events * (checkpoint_raw_bytes + checkpoint_ema_bytes) * 2
    )
    snapshot_total_bytes = snapshot_events * snapshot_bytes * 2
    artifact_reserve_bytes = math.ceil(reserve_gib * GIB)
    measured_core_bytes = (
        recovery_history_bytes
        + rolling_recovery_bytes
        + checkpoint_bytes
        + snapshot_total_bytes
    )
    subtotal_bytes = (
        measured_core_bytes + artifact_reserve_bytes + asfd_bank_reserve_bytes
    )
    required_bytes = math.ceil(subtotal_bytes * (1.0 + contingency))
    return {
        "campaign": campaign,
        "storage_root": storage_root,
        "schedule": {
            "recovery_commits": recovery_commits,
            "rolling_recoveries": ROLLING_RECOVERIES,
            "checkpoint_events": checkpoint_events,
            "snapshot_events": snapshot_events,
            "asfd_continuation_included": campaign == "ordered_750_foundation",
            "immutable_checkpoint_snapshot_copies": 2,
        },
        "measured_unit_bytes": {
            "recovery": recovery_bytes,
            "checkpoint_raw": checkpoint_raw_bytes,
            "checkpoint_ema": checkpoint_ema_bytes,
            "snapshot": snapshot_bytes,
        },
        "projected_bytes": {
            "recovery_history": recovery_history_bytes,
            "rolling_recoveries": rolling_recovery_bytes,
            "checkpoints": checkpoint_bytes,
            "snapshots": snapshot_total_bytes,
            "measured_core": measured_core_bytes,
            "artifact_reserve": artifact_reserve_bytes,
            "asfd_feature_banks": asfd_bank_reserve_bytes,
            "subtotal": subtotal_bytes,
            "required_with_contingency": required_bytes,
        },
        "artifact_reserve_gib": reserve_gib,
        "contingency_fraction": contingency,
        "filesystem": {
            "total_bytes_at_preflight": total_bytes,
            "free_bytes_at_preflight": free_bytes,
        },
        "required_gib": required_bytes / GIB,
        "within_total_capacity": required_bytes <= total_bytes,
        "within_current_free_space": required_bytes <= free_bytes,
        "decision": (
            "GO"
            if required_bytes <= total_bytes and required_bytes <= free_bytes
            else "NO_GO"
        ),
        "scope": (
            "shared filesystem containing the common workspace and every per-arm "
            "mirror; storage and egress dollar charges remain part of the explicit "
            "nontraining reserve"
        ),
    }


def revalidate_storage_plan(plan: object, benchmark: dict) -> dict[str, object]:
    """Recompute a frozen storage projection from its measured leaves."""

    if not isinstance(plan, dict):
        raise TypeError("preflight lacks a durable storage plan")
    filesystem = plan.get("filesystem")
    if not isinstance(filesystem, dict):
        raise TypeError("durable storage plan lacks its filesystem observation")
    expected = build_storage_plan(
        benchmark,
        campaign=plan.get("campaign", "matched_screen"),
        storage_root=plan.get("storage_root"),
        total_bytes=filesystem.get("total_bytes_at_preflight"),
        free_bytes=filesystem.get("free_bytes_at_preflight"),
        artifact_reserve_gib=plan.get("artifact_reserve_gib"),
        contingency_fraction=plan.get("contingency_fraction"),
    )
    if plan != expected:
        raise RuntimeError(
            "durable storage plan does not match measured benchmark bytes"
        )
    return expected
