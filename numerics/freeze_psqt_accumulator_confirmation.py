"""Create the immutable source/registry hash manifest for confirmation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATASKETCHES_VERSION = "5.2.0"

FROZEN_SOURCES = (
    "numerics/run_psqt_accumulator_confirmatory.py",
    "numerics/freeze_psqt_accumulator_confirmation.py",
    "numerics/standard_projected_kll.py",
    "numerics/audit_projected_kll.py",
    "numerics/projected_quantile_accumulators.py",
    "numerics/persistent_sliced_quantile_transport.py",
    "numerics/persistent_quantile_transport.py",
    "numerics/lowdim_drift.py",
    "numerics/psqt_confirmatory_targets.py",
    "numerics/generate_psqt_accumulator_registry.py",
    "numerics/PSQTAccumulatorConfirmatoryProtocol.md",
    "numerics/PSQTAccumulatorConfirmatoryRoadmap.md",
    "numerics/kll_audit_runs/20260722-004356-k128/summary.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = args.registry.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    payload = json.loads(registry.read_text(encoding="utf-8"))
    if payload.get("status") != "sealed-fresh" or \
            payload.get("target_count") != 64:
        raise ValueError("freeze requires the 64-target sealed registry")
    installed = importlib.metadata.version("datasketches")
    if installed != DATASKETCHES_VERSION:
        raise RuntimeError(
            f"expected datasketches {DATASKETCHES_VERSION}, found {installed}")
    source_hashes = {}
    for relative in FROZEN_SOURCES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[relative] = sha256(path)
    freeze = {
        "status": "frozen-before-confirmatory-execution",
        "registry": {
            "path_at_freeze": str(registry),
            "sha256": sha256(registry),
            "target_count": payload["target_count"],
            "master_seed": payload["master_seed"],
        },
        "python_required": "3.12",
        "datasketches": DATASKETCHES_VERSION,
        "source_sha256": source_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"freeze: {args.output}")
    print(f"registry sha256: {freeze['registry']['sha256']}")
    print(f"sources: {len(source_hashes)}")


if __name__ == "__main__":
    main()
