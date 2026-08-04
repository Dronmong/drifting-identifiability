"""Fail-fast pre-run verification for a B2.5 resume.

Unit 500 completed on 2026-08-01/02; units 501/502 are pending.  Those two units
cost roughly eleven GPU-hours and are only comparable with unit 500 if every
hashed input is byte-identical to what unit 500 ran against.  This checks that
*before* the run rather than discovering drift afterwards.

**This file deliberately lives at the top of ``numerics/`` rather than inside
the stage package.**  ``stage_b25/artifacts.py:source_manifest`` builds its
manifest with ``HERE.rglob("*.py")``, and ``diagnostics.py``/``b1_freeze.py``/
``f3b_freeze.py`` glob ``PACKAGE.glob("*.py")``, so adding *any* module to
``stage_b25/`` or to ``encoder_independent_drifting/`` silently invalidates a
hash-bound preflight — the recorded files all still match, but the computed
manifest gains an entry and the dict comparison fails.  An earlier draft of
this tool lived in ``stage_b25/`` and did exactly that, aborting the resume at
``load_preflight``.  Nothing trained, but the guard is the only reason.

That episode is also why :func:`check_manifest_equality` exists.  Verifying
that every *recorded* path still hashes correctly is necessary but **not
sufficient**: it cannot see a file that was added.  The check that matters is
whole-dict equality against the live ``source_manifest()`` — the same
comparison ``load_preflight`` makes.

Read-only: trains nothing, writes nothing, touches no evaluation source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from numerics.encoder_independent_drifting.stage_b25.artifacts import (
    DEFAULT_PREFLIGHT,
    HERE,
    config_payload,
    source_manifest,
)
from numerics.encoder_independent_drifting.stage_b25.core import (
    B25_ARMS,
    B25Config,
    b25_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = REPOSITORY_ROOT / "numerics" / "EncoderIndependentB25Protocol.md"
CHECKPOINTS = HERE / "checkpoints"

# Unit 500 measured 5.49 h wall including evaluation, and its heaviest arm
# (B1B2) peaked at 4.97 GiB reserved.  Measurements, not promises.
MEASURED_UNIT_HOURS = 5.49
MEASURED_PEAK_RESERVED_GIB = 4.97
MINIMUM_FREE_DISK_GIB = 5.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit_artifacts(config: B25Config) -> dict[int, tuple[Path, Path]]:
    return {
        unit: (HERE / f"b25_unit_{unit}.json", HERE / f"b25_unit_{unit}.json.sha256")
        for unit in config.units
    }


def orphaned_checkpoints(config: B25Config) -> list[Path]:
    """Checkpoints belonging to a unit that never produced a sealed artifact."""
    if not CHECKPOINTS.is_dir():
        return []
    orphans: list[Path] = []
    for unit, (result, sidecar) in unit_artifacts(config).items():
        if result.exists() and sidecar.exists():
            continue
        for arm in B25_ARMS:
            for step in config.checkpoint_steps:
                path = CHECKPOINTS / f"b25_u{unit}_{arm.lower()}_step{step}_ema.pt"
                if path.exists():
                    orphans.append(path)
    return orphans


def check_manifest_equality(recorded: dict[str, str]) -> tuple[bool, str]:
    """The exact comparison ``load_preflight`` makes: whole-dict equality.

    Reports added and removed paths separately from content drift, because the
    three have different causes and different fixes.
    """
    live = source_manifest()
    if live == recorded:
        return True, f"{len(live)} files, manifest identical"
    added = sorted(set(live) - set(recorded))
    removed = sorted(set(recorded) - set(live))
    changed = sorted(
        name for name in set(live) & set(recorded) if live[name] != recorded[name]
    )
    parts = []
    if added:
        parts.append(f"ADDED {len(added)}: {', '.join(added)}")
    if removed:
        parts.append(f"REMOVED {len(removed)}: {', '.join(removed)}")
    if changed:
        parts.append(f"CHANGED {len(changed)}: {', '.join(changed)}")
    return False, "; ".join(parts)


def _check_sidecar(path: Path) -> tuple[bool, str]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists():
        return False, f"missing {path.name}"
    if not sidecar.exists():
        return False, f"missing sidecar for {path.name}"
    recorded = sidecar.read_text(encoding="utf-8").split()[0].strip()
    actual = _sha256(path)
    if recorded != actual:
        return False, f"{path.name} sha256 {actual[:16]} != recorded {recorded[:16]}"
    return True, f"{path.name} {actual[:16]}"


def run_checks(config: B25Config, preflight_path: Path) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append((name, bool(ok), detail))

    ok, detail = _check_sidecar(preflight_path)
    record("preflight artifact", ok, detail)
    if not ok:
        return results
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    # ``load_preflight`` injects this after verifying the sidecar, and unit
    # artifacts record that injected value; recompute rather than importing the
    # raising loader, so every check below still reports itemized.
    preflight_digest = _sha256(preflight_path)

    record(
        "preflight verdict",
        preflight.get("verdict", {}).get("decision") == "GO",
        str(preflight.get("verdict", {}).get("decision")),
    )

    # The check that actually gates the run.  Whole-dict equality, not
    # per-recorded-path verification: only this can see an added file.
    ok, detail = check_manifest_equality(preflight["source_sha256"])
    record("source manifest equality", ok, detail)

    record(
        "stage configuration",
        preflight.get("b25_config") == config_payload(b25_config()),
        "B25Config matches preflight",
    )

    record(
        "protocol document",
        PROTOCOL.exists() and _sha256(PROTOCOL) == preflight["protocol_sha256"],
        PROTOCOL.name,
    )

    for prefix in ("b1", "b2"):
        freeze = Path(preflight[f"{prefix}_freeze_path"])
        record(
            f"{prefix.upper()} freeze",
            freeze.exists() and _sha256(freeze) == preflight[f"{prefix}_freeze_sha256"],
            freeze.name,
        )

    external = preflight["external_data"]
    pool = HERE / "data" / "cinic10_imagenet_only_b25_disjoint.npz"
    record(
        "shifted data pool",
        pool.exists() and _sha256(pool) == external["artifact_sha256"],
        external["source_id"],
    )

    completed, pending = [], []
    for unit, (result, sidecar) in unit_artifacts(config).items():
        if result.exists() and sidecar.exists():
            ok, detail = _check_sidecar(result)
            record(f"unit {unit} artifact", ok, detail)
            if ok:
                payload = json.loads(result.read_text(encoding="utf-8"))
                record(
                    f"unit {unit} preflight binding",
                    payload.get("preflight_sha256") == preflight_digest,
                    "bound to this preflight",
                )
                record(
                    f"unit {unit} factorial",
                    set(payload.get("cells", {})) == set(B25_ARMS),
                    "/".join(sorted(payload.get("cells", {}))),
                )
            completed.append(unit)
        elif result.exists() or sidecar.exists():
            record(f"unit {unit} artifact", False, "half-written; inspect before resume")
        else:
            pending.append(unit)
    record(
        "units pending",
        bool(pending),
        f"completed {completed or 'none'}; pending {pending or 'none'}",
    )

    orphans = orphaned_checkpoints(config)
    record(
        "orphaned checkpoints",
        not orphans,
        "none"
        if not orphans
        else f"{len(orphans)} from an interrupted unit: "
        + ", ".join(path.name for path in orphans),
    )

    aggregate = HERE / "b25_development.json"
    record(
        "aggregate not yet written",
        not aggregate.exists() and not Path(str(aggregate) + ".sha256").exists(),
        aggregate.name,
    )

    free_gib = shutil.disk_usage(HERE).free / 2**30
    record(
        "disk headroom",
        free_gib >= MINIMUM_FREE_DISK_GIB,
        f"{free_gib:.1f} GiB free (need >= {MINIMUM_FREE_DISK_GIB:.0f})",
    )

    try:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            total = properties.total_memory / 2**30
            record(
                "cuda device",
                total >= MEASURED_PEAK_RESERVED_GIB,
                f"{properties.name}, {total:.2f} GiB total, "
                f"unit 500 peaked at {MEASURED_PEAK_RESERVED_GIB:.2f} GiB",
            )
        else:
            record("cuda device", False, "torch reports no CUDA device")
    except Exception as error:  # pragma: no cover - environment probe only
        record("cuda device", False, f"probe failed: {error}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="B2.5 resume verification")
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    args = parser.parse_args()
    config = B25Config()
    config.validate()

    results = run_checks(config, args.preflight)
    width = max(len(name) for name, _, _ in results)
    print("=== B2.5 resume verification ===")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<{width}}  {detail}")

    failures = [name for name, ok, _ in results if not ok]
    pending = [
        unit
        for unit, (result, sidecar) in unit_artifacts(config).items()
        if not (result.exists() and sidecar.exists())
    ]
    print()
    print(
        f"  planned work: units {pending or 'none'} "
        f"(~{MEASURED_UNIT_HOURS * len(pending):.1f} h at unit 500's measured rate), "
        "then aggregate"
    )

    orphans = orphaned_checkpoints(config)
    if orphans:
        print()
        print("  An interrupted unit left checkpoints that will block the restart.")
        print("  B2.5 has no within-unit recovery, so the unit restarts from zero.")
        print("  Remove them deliberately, then re-run:")
        for path in orphans:
            print(f"    rm {path}")

    if failures:
        print()
        print(f"  BLOCKED: {len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print()
    print("  GO: every hashed input matches what unit 500 ran against.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
