"""Calibrate F1's stochastic full-teacher-pool memorization veto.

The replay-bank threshold cannot honestly be applied to a stochastic teacher
whose support is the eligible training pool.  This preflight measures exact
pixel-space nearest-training distances for four held-out 512-image groups and
separates them mechanically from known exact training copies.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from .device import configure, resolve_device
from .diagnostics import provenance, write_json
from .f1 import PARTICLES, _self_distances
from .f1_calibration import eval_references
from .f1_k200 import (
    HERE,
    confirmation_allocation,
    load_freeze,
    nearest_reference_distances,
)


GROUPS = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--freeze", type=Path,
                        default=HERE / "f1_k200_freeze.json")
    parser.add_argument("--out", type=Path,
                        default=HERE / "f1_k200_stochastic_veto.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    freeze = load_freeze(args.freeze)

    started = time.time()
    allocation = confirmation_allocation(args.resolution, args.data_root)
    train = cifar.cifar_pool(args.resolution, "train", args.data_root)
    reference = train[torch.as_tensor(allocation.stochastic_pool)]
    _, held_out = eval_references(args.resolution, args.data_root)
    held_out = held_out[:GROUPS * PARTICLES]

    print(f"held-out queries={len(held_out)} eligible train={len(reference)}",
          flush=True)
    distances, _ = nearest_reference_distances(held_out, reference, device)
    normalizers = []
    for group in range(GROUPS):
        start, stop = group * PARTICLES, (group + 1) * PARTICLES
        normalizers.append(float(
            _self_distances(held_out[start:stop]).min(dim=1).values.median()))
    normalizer = float(np.median(normalizers))

    healthy = []
    groups = []
    for group in range(GROUPS):
        start, stop = group * PARTICLES, (group + 1) * PARTICLES
        values = distances[start:stop].numpy() / normalizer
        statistic = float(np.median(values))
        healthy.append(statistic)
        groups.append({
            "group": group,
            "group_self_distance_normalizer": normalizers[group],
            "nearest_train_normalized_median": statistic,
            "nearest_train_normalized_p05": float(np.percentile(values, 5)),
            "nearest_train_normalized_p95": float(np.percentile(values, 95)),
        })

    # Exact eligible-training copies have nearest distance exactly zero.  Half
    # the weakest held-out median is fixed mechanically, accepts every measured
    # healthy group, and rejects every exact-copy group.
    healthy_floor = float(min(healthy))
    threshold = 0.5 * healthy_floor
    healthy_accepted = float(np.mean(np.asarray(healthy) >= threshold))
    exact_copy_rejected = bool(0.0 < threshold)
    decision = "GO" if healthy_accepted >= 0.95 and exact_copy_rejected else "NO-GO"

    payload = {
        "status": "f1-k200-stochastic-veto-calibration",
        "protocol": "numerics/EncoderIndependentF1K200ConfirmationProtocol.md",
        "provenance": provenance(),
        "device": settings,
        "config": {
            "resolution": args.resolution,
            "threads": args.threads,
            "data_root": args.data_root,
            "groups": GROUPS,
            "group_size": PARTICLES,
            "freeze": str(args.freeze),
            "out": str(args.out),
        },
        "freeze_sha256": args.freeze.with_suffix(
            args.freeze.suffix + ".sha256").read_text(
                encoding="utf-8").split()[0],
        "allocation_digests": allocation.digests,
        "elapsed_seconds": time.time() - started,
        "healthy_groups": groups,
        "normalizer": normalizer,
        "known_exact_copy_statistic": 0.0,
        "threshold": threshold,
        "verdict": {
            "decision": decision,
            "healthy_accepted": healthy_accepted,
            "exact_copy_rejected": exact_copy_rejected,
            "reading": (
                f"{decision}: threshold {threshold:.6f} accepts "
                f"{healthy_accepted:.1%} of held-out groups and rejects "
                "known exact training copies"),
        },
    }
    digest = write_json(args.out, payload)
    print("\n=== STOCHASTIC MEMORIZATION VETO ===")
    for group in groups:
        print(f"group {group['group']}: median="
              f"{group['nearest_train_normalized_median']:.6f}")
    print(f"threshold={threshold:.6f} decision={decision}")
    print(f"wrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
