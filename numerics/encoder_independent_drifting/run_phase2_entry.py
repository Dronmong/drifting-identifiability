"""Phase-2A entry gate on CIFAR-16 (protocol section 3).

Four conditions, all recorded whether they pass or fail:

E1  the testbed is solvable        skyline precision >= 0.5 x null precision
E2  pixel geometry is the bottleneck   structured k-NN >= 1.25 x pixel k-NN
E3  each arm's field can reach the target   G0.5 residual/floor <= 2.0
E4  no kernel is collapsed         collapsed_row_fraction == 0

E3 is the condition Phase 1 lacked.  An arm whose field provably plateaus
above the `q = p` floor is not run, and reachability is a property of the
data as much as of the kernel, so it is re-measured here rather than carried
over from the synthetic gate.

The training budget is derived from the skyline, never from a baseline arm:
freezing on a baseline's coverage -- which saturates early -- is what
under-budgeted Phase 1.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase2_entry
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from . import oracle as O
from .audit_phase2 import knn_content_accuracy
from .config import MASTER_SEED, GeometryConfig, TrainConfig, derive_seed
from .diagnostics import provenance, write_json
from .evaluate import evaluation_pools, null_reference
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .train import phase2_arms

HERE = Path(__file__).resolve().parent

# Frozen thresholds (protocol section 3).
SOLVABLE_FRACTION = O.ADMISSIBLE_PRECISION_FRACTION      # 0.5
GEOMETRY_ADVANTAGE = 1.25
ZERO_SET_THRESHOLD = 2.0
BUDGET_LADDER = (300, 600, 1200)


def _targets(resolution: int, root: str | None):
    return (cifar.cifar_target(resolution, "train", root),
            cifar.cifar_target(resolution, "eval", root))


def e1_solvable(resolution: int, root: str | None) -> dict:
    """Skyline admissibility, and the budget it implies."""
    train, evaluation = _targets(resolution, root)
    sweep = []
    chosen = None
    for steps in BUDGET_LADDER:
        config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                             audit_batch=32, eval_samples=512,
                             image_size=resolution)
        pools = evaluation_pools(evaluation, config, MASTER_SEED)
        null = null_reference(evaluation, pools, MASTER_SEED)
        result = O.train_skyline(train, config, MASTER_SEED, pools, null)
        required = SOLVABLE_FRACTION * float(null["precision"])
        row = {
            "steps": steps,
            "skyline_precision": result.precision,
            "skyline_coverage": result.coverage,
            "null_precision": float(null["precision"]),
            "required_precision": required,
            "skyline_score": result.score["geometry_score"],
            "admissible": bool(result.precision >= required),
        }
        sweep.append(row)
        print(f"    E1 steps={steps:5} skyline precision="
              f"{result.precision:.3f} required={required:.3f} "
              f"score={row['skyline_score']:7.3f} "
              f"admissible={row['admissible']}", flush=True)
        if row["admissible"] and chosen is None:
            chosen = steps
    # One step up the ladder for headroom, per protocol section 3.
    budget = None
    if chosen is not None:
        index = BUDGET_LADDER.index(chosen)
        budget = BUDGET_LADDER[min(index + 1, len(BUDGET_LADDER) - 1)]
    return {"sweep": sweep, "smallest_admissible": chosen,
            "budget": budget, "pass": chosen is not None}


def e2_geometry_advantage(resolution: int, count: int,
                          root: str | None) -> dict:
    """Is pixel geometry actually the bottleneck on this data?"""
    pool = cifar.cifar_pool(resolution, "train", root)
    rng = np.random.default_rng(derive_seed(MASTER_SEED, "e2"))
    index = rng.choice(len(pool), size=min(count, len(pool)), replace=False)
    images = pool[index]
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root=root or cifar.DEFAULT_ROOT, download=False)
    start, _ = cifar.SPLITS["train"]
    labels = np.asarray(dataset.targets)[start:start + len(pool)][index]

    accuracy = {"pixel": knn_content_accuracy(images, labels)}
    for family in ("wavelet", "scattering", "randconv"):
        config = GeometryConfig(family=family,
                                second_order=(family == "scattering"))
        built = build_family(config, 3)
        features = torch.cat([b.flat(images) for b in built.branches], dim=1)
        accuracy[family] = knn_content_accuracy(features, labels)
    best = max(accuracy[k] for k in ("wavelet", "scattering"))
    ratio = best / max(accuracy["pixel"], 1e-12)
    print(f"    E2 k-NN content accuracy (chance {1/10:.3f}): "
          + " ".join(f"{k}={v:.3f}" for k, v in accuracy.items())
          + f" | best/pixel={ratio:.3f}", flush=True)
    return {"knn_accuracy": accuracy, "best_structured": best,
            "ratio_over_pixel": ratio, "threshold": GEOMETRY_ADVANTAGE,
            "samples": int(len(images)),
            "pass": bool(ratio >= GEOMETRY_ADVANTAGE)}


def e3_zero_set(resolution: int, seeds: int, batch: int, steps: int,
                step_size: float, root: str | None) -> dict:
    """G0.5 on CIFAR, for every (geometry, direction) pair an arm uses."""
    train, _ = _targets(resolution, root)
    families = {}
    for arm in phase2_arms():
        families.setdefault(arm.geometry.family, arm.geometry)
    rows: list[dict] = []
    for seed in range(seeds):
        rng = np.random.default_rng(
            derive_seed(MASTER_SEED, "e3", resolution, seed))
        calibration = train.sample(256, rng)
        holdout = train.sample(batch, rng)
        matched = train.sample(batch, rng)
        for name, config in families.items():
            branch = build_family(config, 3).branches[0]
            kernel = calibrate_block_kernel(
                branch, calibration,
                "smooth_laplace" if config.base_kernel == "auto"
                else config.base_kernel,
                config.bandwidth_quantile, config.bandwidth_multiplier,
                config.kernel_eps, combine=config.combine,
                target_ess_fraction=config.target_ess_fraction)
            for mode in ("standard",):
                _, floor_stats = KG.field(
                    matched, holdout, matched, branch, kernel,
                    direction_mode=mode, normalization="none")
                floor = floor_stats["drift_rms_raw"]
                cloud = torch.tensor(
                    rng.normal(scale=0.5,
                               size=(batch, 3, resolution, resolution)),
                    dtype=torch.float32)
                start = None
                for index in range(steps):
                    fresh = train.sample(batch, rng)
                    drift, stats = KG.field(
                        cloud, fresh, cloud, branch, kernel,
                        direction_mode=mode, normalization="rms")
                    if start is None:
                        start = stats["drift_rms_raw"]
                    cloud = cloud + step_size * (1.0 - index / steps) * drift
                _, final = KG.field(
                    cloud, holdout, cloud, branch, kernel,
                    direction_mode=mode, normalization="none")
                rows.append({
                    "seed": seed, "family": name, "direction_mode": mode,
                    "floor": floor, "start": start,
                    "residual": final["drift_rms_raw"],
                    "residual_over_floor": (
                        final["drift_rms_raw"] / max(floor, 1e-12)),
                    "descent_fraction": (
                        1.0 - final["drift_rms_raw"] / max(start, 1e-12)),
                    "collapsed_row_fraction": final["collapsed_row_fraction"],
                    "ess_fraction": final["ess_fraction"],
                })

    summary = {}
    for row in rows:
        summary.setdefault(f"{row['family']}::{row['direction_mode']}",
                           []).append(row)
    verdicts = {}
    for key, group in sorted(summary.items()):
        ratio = float(np.median([r["residual_over_floor"] for r in group]))
        descent = float(np.median([r["descent_fraction"] for r in group]))
        collapsed = float(np.max([r["collapsed_row_fraction"]
                                  for r in group]))
        verdicts[key] = {
            "residual_over_floor": ratio, "descent_fraction": descent,
            "max_collapsed_row_fraction": collapsed,
            "reaches": bool(ratio <= ZERO_SET_THRESHOLD),
        }
        mark = "reaches" if verdicts[key]["reaches"] else "PLATEAUS"
        print(f"    E3 {key:34} {ratio:6.2f} "
              f"(descended {descent * 100:5.1f}%) {mark}", flush=True)
    return {
        "threshold": ZERO_SET_THRESHOLD, "verdicts": verdicts,
        "rows": rows,
        "pass": bool(verdicts) and all(v["reaches"] for v in
                                       verdicts.values()),
    }


def e4_kernel_health(zero_set: dict) -> dict:
    """No admitted branch may be operating on a collapsed kernel."""
    worst = {key: value["max_collapsed_row_fraction"]
             for key, value in zero_set.get("verdicts", {}).items()}
    return {"max_collapsed_row_fraction": worst,
            "pass": bool(worst) and all(v == 0.0 for v in worst.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--zero-set-steps", type=int, default=300)
    parser.add_argument("--zero-set-step", type=float, default=0.2)
    parser.add_argument("--knn-samples", type=int, default=2048)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase2_entry.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    if not cifar.available(args.root):
        raise SystemExit(
            "CIFAR-10 is not present locally; Phase 2 refuses to guess. "
            "Fetch it into ~/.cache/cifar first.")

    started = time.time()
    print("--- E1 skyline admissibility and budget ---", flush=True)
    e1 = e1_solvable(args.resolution, args.root)
    print("--- E2 geometry advantage ---", flush=True)
    e2 = e2_geometry_advantage(args.resolution, args.knn_samples, args.root)
    print("--- E3 zero-set reachability ---", flush=True)
    e3 = e3_zero_set(args.resolution, args.seeds, args.batch,
                     args.zero_set_steps, args.zero_set_step, args.root)
    e4 = e4_kernel_health(e3)

    results = {"E1_solvable": e1, "E2_geometry_advantage": e2,
               "E3_zero_set": e3, "E4_kernel_health": e4}
    verdicts = {k: v["pass"] for k, v in results.items()}
    payload = {
        "status": "phase2a-entry-gate",
        "scope": f"CIFAR-10 at {args.resolution}x{args.resolution}, "
                 "disjoint train/eval splits, no pretrained encoder, no "
                 "class labels in any objective",
        "config": vars(args) | {"out": str(args.out)},
        "provenance": provenance(),
        "elapsed_seconds": time.time() - started,
        "verdicts": verdicts,
        "gate_pass": all(verdicts.values()),
        "recommended_budget": e1.get("budget"),
        "results": results,
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 2A ENTRY GATE ===")
    for name, verdict in verdicts.items():
        print(f"  [{'PASS' if verdict else 'FAIL'}] {name}")
    print(f"\n  recommended budget: {e1.get('budget')} steps")
    print(f"  overall: {'PASS' if all(verdicts.values()) else 'FAIL'}")
    print(f"  wrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
