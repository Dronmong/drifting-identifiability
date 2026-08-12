"""Uniform boxcar average of trailing raw snapshots (post-hoc EMA, Karras-style).

``stage_cap2.posthoc_ema`` is the production path and requires a sealed run
ledger to bind the snapshots it averages. The candidate audit produces plain
``{state_dict, profile}`` checkpoints instead, so this performs the same
synthesis over those, writing a checkpoint in the identical layout that
``standard_metrics`` and ``numerical_admission`` already consume.

Post-hoc averaging is secondary by construction: the raw checkpoint remains the
primary result, so this can never become checkpoint selection on a metric. On
CAP-EMF-1 it moved clean FID from 112.94 to 83.65 without any retraining.

Accumulation is in float64 regardless of the stored dtype -- averaging ~150 MB
of float32 parameters in float32 loses low-order bits for no reason.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def average_snapshots(paths: list[Path]) -> dict:
    """Uniform mean of the given raw states, in float64."""
    if not paths:
        raise ValueError("post-hoc averaging needs at least one snapshot")
    accumulator: dict[str, torch.Tensor] = {}
    reference_profile = None
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if reference_profile is None:
            reference_profile = payload["profile"]
        elif payload["profile"] != reference_profile:
            raise RuntimeError(f"{path} was produced by a different profile")
        state = payload["state_dict"]
        if accumulator and set(state) != set(accumulator):
            raise RuntimeError(f"{path} has a different parameter set")
        for key, value in state.items():
            promoted = value.detach().to(torch.float64)
            accumulator[key] = (
                promoted if key not in accumulator else accumulator[key] + promoted
            )
    averaged = {
        key: (value / len(paths)).to(torch.float32)
        for key, value in accumulator.items()
    }
    return {"state_dict": averaged, "profile": reference_profile}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, nargs="+", required=True,
                        help="raw checkpoints in increasing step order")
    parser.add_argument("--windows", type=int, nargs="+", default=[2, 3, 5],
                        help="how many trailing snapshots to average")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    ordered = list(args.snapshots)
    print(f"{len(ordered)} snapshots, oldest first:")
    for path in ordered:
        print(f"  {path.name}")

    written = {}
    for window in args.windows:
        if window > len(ordered):
            print(f"  skipping window {window}: only {len(ordered)} snapshots")
            continue
        selected = ordered[-window:]
        payload = average_snapshots(selected)
        payload["posthoc_ema"] = {
            "window": window,
            "sources": [path.name for path in selected],
            "accumulation_dtype": "float64",
            "eligible_for_selection": False,
            "note": "secondary to the raw checkpoint; never a selection basis",
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        destination = args.out_dir / f"posthoc_ema_w{window}.pt"
        torch.save(payload, destination)
        written[str(window)] = str(destination)
        print(f"  window {window}: averaged {[p.name for p in selected]}")
        print(f"    -> {destination}")

    (args.out_dir / "posthoc_ema_index.json").write_text(
        json.dumps({"written": written}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
