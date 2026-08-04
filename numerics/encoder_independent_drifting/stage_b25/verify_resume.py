"""Fail-fast pre-run verification for a B2.5 resume.

Unit 500 completed on 2026-08-01/02 and units 501/502 are pending.  Those two
units cost roughly eleven GPU-hours and are only comparable with unit 500 if
every hashed input is byte-identical to what unit 500 ran against.  This module
checks that *before* the run rather than discovering drift afterwards.

It is deliberately read-only: it trains nothing, writes nothing, and touches no
evaluation source.  ``run_all.sh`` invokes it first and refuses to start on a
nonzero exit.

The one failure mode that is not a hash problem is an interrupted unit.  B2.5
has **no within-unit recovery** — ``core.py`` saves a checkpoint only at the
three declared steps and keeps no optimizer/RNG recovery state — so a crash at
update 25 000 loses the unit.  Worse, the surviving checkpoints then trip
``run_unit.py``'s "a planned B2.5 checkpoint path already exists" guard and
block the restart.  ``orphaned_checkpoints`` finds exactly those files and the
caller prints the removal command, so a 3 a.m. recovery is mechanical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .artifacts import DEFAULT_PREFLIGHT, HERE
from .core import B25_ARMS, B25Config

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = REPOSITORY_ROOT / "numerics" / "EncoderIndependentB25Protocol.md"
CHECKPOINTS = HERE / "checkpoints"

# Unit 500 measured 5.49 h wall including evaluation.  Two units is the
# projection quoted to the operator; it is a measurement, not a promise.
MEASURED_UNIT_HOURS = 5.49
# Peak reserved bytes recorded by unit 500's heaviest arm (B1B2).
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
    artifacts = unit_artifacts(config)
    orphans: list[Path] = []
    for unit, (result, sidecar) in artifacts.items():
        if result.exists() and sidecar.exists():
            continue
        for arm in B25_ARMS:
            for step in config.checkpoint_steps:
                path = CHECKPOINTS / f"b25_u{unit}_{arm.lower()}_step{step}_ema.pt"
                if path.exists():
                    orphans.append(path)
    return orphans


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
    # artifacts record that injected value; recompute it rather than importing
    # the raising loader, so every check below still reports itemized.
    preflight_digest = _sha256(preflight_path)

    record(
        "preflight verdict",
        preflight.get("verdict", {}).get("decision") == "GO",
        str(preflight.get("verdict", {}).get("decision")),
    )

    drifted = [
        name
        for name, digest in sorted(preflight["source_sha256"].items())
        if not (REPOSITORY_ROOT / name).exists()
        or _sha256(REPOSITORY_ROOT / name) != digest
    ]
    record(
        "hashed executable sources",
        not drifted,
        f"{len(preflight['source_sha256'])} files match"
        if not drifted
        else "DRIFTED: " + ", ".join(drifted),
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

    artifacts = unit_artifacts(config)
    completed, pending = [], []
    for unit, (result, sidecar) in artifacts.items():
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
    parser = argparse.ArgumentParser(description=__doc__)
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
