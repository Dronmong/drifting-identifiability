"""Fresh, freeze-compatible null and veto calibration for F3B B0."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from .appearance import precision_recall
from .config import MASTER_SEED, derive_seed
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnostics import write_json
from .f1 import clopper_pearson_upper
from .f3b import METRIC_CONTROL_FLOOR, RECALL_GATE
from .f3b_evaluation import (
    allocated_images,
    evaluation_allocation,
    matched_real_metrics,
    memorization_statistics,
    real_health_statistics,
)
from .f3b_freeze import HERE, load_freeze, source_manifest, verify_sidecar
from .fid import inception_features

NULL_REPLICATES = 200
NULL_TOLERANCE = 0.025
NULL_STATES = ("identical_images", "gaussian_moment_match")


def _null_draw(
    state: str, replicate: int, train: torch.Tensor, reference: torch.Tensor, count: int
) -> tuple[torch.Tensor, bool]:
    seed = derive_seed(MASTER_SEED + 73_000, "f3b-calibration", state, replicate)
    rng = np.random.default_rng(seed)
    if state == "identical_images":
        index = int(rng.integers(0, len(train)))
        return train[index : index + 1].repeat(count, 1, 1, 1), True
    if state == "gaussian_moment_match":
        return gaussian_moment_match(reference, count, rng), False
    raise ValueError(f"unknown F3B null state {state!r}")


def _recall(
    images: torch.Tensor, reference_features: np.ndarray, device, tiled: bool
) -> float:
    if tiled:
        features = inception_features(images[:1], device).double().numpy()
        features = np.repeat(features, len(images), axis=0)
    else:
        features = inception_features(images, device).double().numpy()
    return precision_recall(features, reference_features)["recall"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, default=HERE / "f3b_freeze.json")
    parser.add_argument("--replicates", type=int, default=NULL_REPLICATES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", type=Path, default=HERE / "f3b_calibration.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = resolve_device(args.device)
    settings = configure(device)
    freeze = load_freeze(args.freeze)
    selected = freeze["profile"]
    resolution = int(selected["model"]["image_size"])
    generated = int(selected["evaluation"]["generated_samples"])
    reference_count = int(selected["evaluation"]["reference_samples"])

    train = cifar.cifar_pool(resolution, "train", args.data_root)
    eval_pool = cifar.cifar_pool(resolution, "eval", args.data_root)
    allocation = evaluation_allocation(len(eval_pool), reference_count, generated, 3)
    reference, _ = allocated_images(
        allocation, "confirmation", resolution, args.data_root
    )
    _, development_controls = allocated_images(
        allocation, "development", resolution, args.data_root
    )
    reference_features = inception_features(reference, device).double().numpy()
    started = time.time()

    null_rows = {}
    for state in NULL_STATES:
        values = []
        for replicate in range(args.replicates):
            images, tiled = _null_draw(state, replicate, train, reference, generated)
            values.append(_recall(images, reference_features, device, tiled))
        values_array = np.asarray(values, dtype=float)
        exceedances = int((values_array > RECALL_GATE).sum())
        null_rows[state] = {
            "replicates": args.replicates,
            "mean": float(values_array.mean()),
            "maximum": float(values_array.max()),
            "q99": float(np.percentile(values_array, 99)),
            "exceedances": exceedances,
            "p_null_upper": clopper_pearson_upper(exceedances, args.replicates),
            "values": values,
        }
        print(
            f"{state}: E={exceedances}/{args.replicates} "
            f"p_upper={null_rows[state]['p_null_upper']:.6f}",
            flush=True,
        )

    health_rows = [
        {"group": index, **real_health_statistics(control)}
        for index, control in enumerate(development_controls)
    ]
    normalizer = float(np.median([row["real_nn_scale"] for row in health_rows]))
    memorization_rows = []
    control_rows = []
    for index, control in enumerate(development_controls):
        memory = memorization_statistics(control, train, normalizer, device)
        control_metric = matched_real_metrics(control, reference_features, device)
        memorization_rows.append({"group": index, **memory})
        control_rows.append({"group": index, **control_metric})

    # These are permissive *failure vetoes*, not quality targets. Half the
    # weakest held-out-real value accepts every measured healthy group while
    # exact copies fail the strictly positive memorization threshold.
    thresholds = {
        "effective_rank": 0.5 * min(row["effective_rank"] for row in health_rows),
        "one_minus_duplicate_rate": 0.5
        * min(row["one_minus_duplicate_rate"] for row in health_rows),
        "nn_diversity": 0.5 * min(row["nn_diversity"] for row in health_rows),
        "nearest_train_normalized": 0.5
        * min(row["nearest_train_normalized"] for row in memorization_rows),
    }
    p_upper = max(row["p_null_upper"] for row in null_rows.values())
    full_power = args.replicates >= NULL_REPLICATES
    controls_valid = all(row["recall"] > METRIC_CONTROL_FLOOR for row in control_rows)
    thresholds_valid = all(
        np.isfinite(value) and value > 0 for value in thresholds.values()
    )
    decision = (
        "GO"
        if full_power
        and p_upper < NULL_TOLERANCE
        and controls_valid
        and thresholds_valid
        else "NO-GO"
    )
    verdict = {
        "decision": decision,
        "full_null_power": full_power,
        "p_null_upper": p_upper,
        "null_tolerance": NULL_TOLERANCE,
        "recall_gate": RECALL_GATE,
        "controls_valid": controls_valid,
        "thresholds_valid": thresholds_valid,
        "reading": (
            f"{decision}: p_null_upper={p_upper:.6f}; "
            f"full_power={full_power}; controls_valid={controls_valid}; "
            f"thresholds_valid={thresholds_valid}"
        ),
    }
    payload = {
        "status": "f3b-b0-calibration",
        "protocol": "numerics/EncoderIndependentF3BProtocol.md",
        "freeze_sha256": verify_sidecar(args.freeze),
        "source_sha256": source_manifest(),
        "device": settings,
        "allocation_digests": allocation.digests,
        "config": {
            "resolution": resolution,
            "generated_samples": generated,
            "reference_samples": reference_count,
            "replicates": args.replicates,
            "data_root": args.data_root,
        },
        "null_states": null_rows,
        "health_controls": health_rows,
        "memorization_controls": memorization_rows,
        "matched_real_controls": control_rows,
        "normalizer": normalizer,
        "thresholds": thresholds,
        "verdict": verdict,
        "elapsed_seconds": time.time() - started,
    }
    digest = write_json(args.out, payload)
    print("\n=== F3B CALIBRATION ===")
    print(verdict["reading"])
    print(f"thresholds={thresholds}")
    print(f"wrote {args.out} sha256={digest}")


if __name__ == "__main__":
    main()
