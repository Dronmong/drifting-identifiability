"""Phase-4 design investigation (post-hoc; feeds no gate).

Every anchor measurement this program has ever made was taken on a generator
that we now know was collapsed to ~2.3 effective dimensions.  The anchor
pulls output back toward the data manifold, so on a collapsed cloud its
benefit may be nothing more than an indirect variance correction -- doing
badly what reform R11 now does directly.  If so, the anchor thread's entire
evidence base is confounded and the next phase should not be an anchor
confirmation.

That is a testable question, and it decides the next phase.  This module
answers it before the design is written, rather than assuming either way.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import oracle as O
from .config import (
    ArmConfig, FieldConfig, GeometryConfig, MASTER_SEED, MixtureConfig,
    ObjectiveConfig, TrainConfig,
)
from .diagnostics import paired_log_ratio, provenance, write_json
from .evaluate import evaluate_arm, evaluation_pools, null_reference
from .train import train_arm

HERE = Path(__file__).resolve().parent

# Development seeds; this is a design investigation, not a confirmation.
DESIGN_SEED_OFFSET = 2000


def _arm(arm_id: str, family: str, anchor: bool, variance_match: bool,
         second_order: bool = False) -> ArmConfig:
    return ArmConfig(
        arm_id, anchor,
        GeometryConfig(family=family, base_kernel="smooth_laplace",
                       second_order=second_order),
        FieldConfig(direction_mode="paper"), MixtureConfig(adaptive=False),
        ObjectiveConfig(lambda_anchor=1.0 if anchor else 0.0,
                        lambda_geometry=1.0,
                        teacher_variance_match=variance_match),
        note=f"{family} anchor={anchor} R11={variance_match}")


def g1_anchor_confound(resolution: int, seeds: int, steps: int,
                       root: str | None) -> dict:
    """Does the anchor still help once the variance collapse is repaired?

    A 2x2 factorial: anchor on/off crossed with R11 on/off.  If the anchor's
    benefit is large without R11 and vanishes with it, the benefit was
    variance-mediated and the anchor's prior evidence does not transfer.
    """
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                         audit_batch=32, eval_samples=512,
                         image_size=resolution)
    arms = [
        _arm("plain", "raw", False, False),
        _arm("anchor", "raw", True, False),
        _arm("r11", "raw", False, True),
        _arm("anchor_r11", "raw", True, True),
    ]
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + DESIGN_SEED_OFFSET + index
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        for arm in arms:
            outcome = train_arm(arm, train, config, seed)
            row = {"arm": arm.arm_id, "seed": seed, "note": arm.note}
            row.update(evaluate_arm(outcome, evaluation, config, pools,
                                    null, seed))
            row["anchor_share"] = row.get("anchor_share_median")
            rows.append(row)
            print(f"      {arm.arm_id:12} score="
                  f"{row.get('geometry_score_v2', float('nan')):7.3f} "
                  f"ed2={row.get('ed2', float('nan')):7.4f} "
                  f"cover={row.get('coverage', float('nan')):5.3f} "
                  f"eff_dim="
                  f"{row.get('effective_dimension_ratio', float('nan')):5.3f} "
                  f"{outcome.wall_seconds:5.1f}s", flush=True)
        skyline = O.train_skyline(train, config, seed, pools, null)
        rows.append({
            "arm": "SKY", "seed": seed,
            "geometry_score_v2": skyline.score["geometry_score"],
            "ed2": skyline.metrics["ed2"],
            "coverage": skyline.coverage,
            "effective_dimension_ratio": skyline.metrics.get(
                "effective_dimension_ratio")})
    return {"rows": rows}


def g2_contraction_scope(resolution: int, seeds: int, steps: int,
                         root: str | None) -> dict:
    """Where else does the contraction appear?

    R11 was derived and confirmed on raw pixel geometry at batch 64.  If the
    contraction is a general property of stop-gradient regression onto a
    mean-shift teacher, it should appear for a structured kernel and at
    other batch sizes too -- which decides whether R11 is a local fix or a
    reportable general claim.
    """
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for batch in (32, 128):
        config = TrainConfig(steps=steps, batch=batch, controller_batch=32,
                             audit_batch=32, eval_samples=512,
                             image_size=resolution)
        for index in range(seeds):
            seed = MASTER_SEED + DESIGN_SEED_OFFSET + index
            pools = evaluation_pools(evaluation, config, seed)
            null = null_reference(evaluation, pools, seed)
            for variance_match in (False, True):
                arm = _arm(f"raw_b{batch}_r11={variance_match}", "raw",
                           False, variance_match)
                outcome = train_arm(arm, train, config, seed)
                row = {"arm": arm.arm_id, "seed": seed, "batch": batch,
                       "family": "raw", "r11": variance_match}
                row.update(evaluate_arm(outcome, evaluation, config, pools,
                                        null, seed))
                rows.append(row)
                print(f"      raw batch={batch:4} R11={variance_match!s:5} "
                      f"score={row.get('geometry_score_v2', float('nan')):7.3f} "
                      f"eff_dim="
                      f"{row.get('effective_dimension_ratio', float('nan')):5.3f}",
                      flush=True)
    # A structured kernel at the declared batch.
    config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                         audit_batch=32, eval_samples=512,
                         image_size=resolution)
    for index in range(seeds):
        seed = MASTER_SEED + DESIGN_SEED_OFFSET + index
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        for variance_match in (False, True):
            arm = _arm(f"wavelet_r11={variance_match}", "wavelet", False,
                       variance_match)
            outcome = train_arm(arm, train, config, seed)
            row = {"arm": arm.arm_id, "seed": seed, "batch": 64,
                   "family": "wavelet", "r11": variance_match}
            row.update(evaluate_arm(outcome, evaluation, config, pools,
                                    null, seed))
            rows.append(row)
            print(f"      wavelet batch=  64 R11={variance_match!s:5} "
                  f"score={row.get('geometry_score_v2', float('nan')):7.3f} "
                  f"eff_dim="
                  f"{row.get('effective_dimension_ratio', float('nan')):5.3f}",
                  flush=True)
    return {"rows": rows}


def paired(rows: list[dict], candidate: str, baseline: str) -> dict:
    scores: dict[str, dict[int, float]] = {}
    for row in rows:
        value = row.get("geometry_score_v2")
        if value is not None and np.isfinite(value):
            scores.setdefault(row["arm"], {})[row["seed"]] = value
    if candidate not in scores or baseline not in scores:
        return {"ratio": float("nan"), "pairs": 0}
    keys = sorted(set(scores[candidate]) & set(scores[baseline]))
    if not keys:
        return {"ratio": float("nan"), "pairs": 0}
    return paired_log_ratio([scores[candidate][k] for k in keys],
                            [scores[baseline][k] for k in keys])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--only", type=str, default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase4_design.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if not cifar.available(args.root):
        raise SystemExit("CIFAR-10 is not present locally.")

    started = time.time()
    stages = {
        "G1_anchor_confound": lambda: g1_anchor_confound(
            args.resolution, args.seeds, args.steps, args.root),
        "G2_contraction_scope": lambda: g2_contraction_scope(
            args.resolution, args.seeds, args.steps, args.root),
    }
    wanted = set(stages) if args.only == "all" else set(args.only.split(","))
    results = {}
    for name, function in stages.items():
        if name in wanted:
            print(f"--- {name} ---", flush=True)
            results[name] = function()

    if "G1_anchor_confound" in results:
        rows = results["G1_anchor_confound"]["rows"]
        results["G1_anchor_confound"]["comparisons"] = {
            "anchor_without_r11": paired(rows, "anchor", "plain"),
            "anchor_with_r11": paired(rows, "anchor_r11", "r11"),
            "r11_without_anchor": paired(rows, "r11", "plain"),
            "r11_with_anchor": paired(rows, "anchor_r11", "anchor"),
            "best_vs_skyline": paired(rows, "anchor_r11", "SKY"),
        }

    payload = {
        "status": "phase4-design-investigation-feeds-no-gate",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "elapsed_seconds": time.time() - started,
        "results": results,
    }
    digest = write_json(args.out, payload)
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
