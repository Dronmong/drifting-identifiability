"""Phase 5: reforms R15-R19 (protocol `EncoderIndependentPhase5Protocol.md`).

G5.1  R16 (decaying teacher step) alone beats the uncorrected baseline;
G5.2  R16 repairs the effective-dimension collapse;
G5.3  the mechanism is confirmed on the REAL trajectory (R18);
G5.4  R15 excludes cells whose kernel is numerically dead;
G5.5  the batch x tau interaction is neighbour starvation (R19).

R17 (latent dimension) and R19 are reported, not gated.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .config import (
    ArmConfig, FieldConfig, GeometryConfig, MASTER_SEED, MixtureConfig,
    ObjectiveConfig, TrainConfig, config_digest, derive_seed,
)
from .diagnostics import (
    kernel_admissible, paired_log_ratio, provenance, write_json,
)
from .evaluate import evaluate_arm, evaluation_pools, null_reference
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .train import train_arm

HERE = Path(__file__).resolve().parent

# Frozen (protocol section 2): disjoint from every earlier phase.
SEED_OFFSET = 5000

# Frozen thresholds (protocol section 3).
R16_RATIO = 0.60
DIMENSION_FLOOR = 0.60
CONTRACTION_CEILING = 0.98
BASELINE = "D0"
R16_ARM = "D2"


def phase5_arms() -> list[ArmConfig]:
    """D0-D3: R11 crossed with R16."""
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace")
    field = FieldConfig(direction_mode="paper")
    fixed = MixtureConfig(adaptive=False)

    def objective(variance_match: bool, schedule: str) -> ObjectiveConfig:
        return ObjectiveConfig(lambda_anchor=0.0, lambda_geometry=1.0,
                               teacher_variance_match=variance_match,
                               eta_schedule=schedule)

    return [
        ArmConfig("D0", False, geometry, field, fixed,
                  objective(False, "constant"),
                  note="uncorrected baseline"),
        ArmConfig("D1", False, geometry, field, fixed,
                  objective(True, "constant"),
                  note="R11 teacher variance match (Phase-3 confirmed)"),
        ArmConfig("D2", False, geometry, field, fixed,
                  objective(False, "linear_decay"),
                  note="R16 decaying teacher step, alone"),
        ArmConfig("D3", False, geometry, field, fixed,
                  objective(True, "linear_decay"),
                  note="R11 + R16"),
    ]


def _trajectory(outcome) -> dict:
    series = outcome.log.series
    out = {}
    for key in ("trajectory_teacher_dimension_ratio",
                "trajectory_output_effective_dimension",
                "trajectory_eta_effective"):
        values = [v for v in series.get(key, []) if np.isfinite(v)]
        if values:
            out[f"median_{key}"] = float(np.median(values))
            out[f"first_{key}"] = float(values[0])
            out[f"last_{key}"] = float(values[-1])
    return out


def d_arms_comparison(resolution: int, seeds: int, steps: int,
                      root: str | None, latent_dims=(32,)) -> dict:
    """The R16-versus-R11 comparison, plus the R17 latent sweep."""
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    arms = phase5_arms()
    rows = []
    for latent_dim in latent_dims:
        config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                             audit_batch=32, eval_samples=512,
                             image_size=resolution, latent_dim=latent_dim)
        for index in range(seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            pools = evaluation_pools(evaluation, config, seed)
            null = null_reference(evaluation, pools, seed)
            for arm in arms:
                outcome = train_arm(arm, train, config, seed)
                row = {"arm": arm.arm_id, "seed": seed,
                       "latent_dim": latent_dim, "note": arm.note}
                row.update(evaluate_arm(outcome, evaluation, config, pools,
                                        null, seed))
                row.update(_trajectory(outcome))
                rows.append(row)
                print(f"    latent={latent_dim:4} seed{index} {arm.arm_id} "
                      f"score={row.get('geometry_score_v2', float('nan')):7.3f} "
                      f"eff_dim="
                      f"{row.get('effective_dimension_ratio', float('nan')):5.3f} "
                      f"teacher_ratio="
                      f"{row.get('median_trajectory_teacher_dimension_ratio', float('nan')):6.4f}"
                      f" {outcome.wall_seconds:5.1f}s", flush=True)
    return {"rows": rows}


def r19_batch_temperature(resolution: int, root: str | None,
                          taus=(0.05, 0.2, None),
                          batches=(64, 256, 1024)) -> dict:
    """R19: is the temperature failure neighbour starvation?

    Health is measured first and every cell is reported; cells failing the
    R15 precondition are marked inadmissible rather than scored.
    """
    train = cifar.cifar_target(resolution, "train", root)
    rows = []
    for tau in taus:
        for batch in batches:
            seed = MASTER_SEED + SEED_OFFSET
            rng = np.random.default_rng(
                derive_seed(seed, "r19", str(tau), batch))
            geometry = GeometryConfig(family="raw",
                                      base_kernel="smooth_laplace",
                                      bandwidth_tau=tau)
            branch = build_family(geometry, 3).branches[0]
            calibration = train.sample(256, rng)
            kernel = calibrate_block_kernel(
                branch, calibration, "smooth_laplace",
                geometry.bandwidth_quantile,
                tau if tau is not None else geometry.bandwidth_multiplier,
                geometry.kernel_eps, combine=geometry.combine,
                target_ess_fraction=(None if tau is not None
                                     else geometry.target_ess_fraction))
            model = OneStepGenerator(32, 3, resolution, 64,
                                     derive_seed(seed, "generator"))
            with torch.no_grad():
                cloud = model(sample_latent(batch, 32,
                                            derive_seed(seed, "r19-l")))
            positives = train.sample(batch, rng)
            _, stats = KG.field(cloud, positives, cloud, branch, kernel,
                                direction_mode="paper", normalization="none")
            verdict = kernel_admissible(stats, batch)
            rows.append({
                "tau": tau, "batch": batch,
                "bandwidth": float(kernel.taus.median()),
                "ess_fraction": stats["ess_fraction"],
                "effective_neighbours": (
                    stats["ess_fraction"] * batch
                    if np.isfinite(stats["ess_fraction"]) else float("nan")),
                "affinity_median": stats["affinity_median"],
                "collapsed_row_fraction": stats["collapsed_row_fraction"],
                **{f"admission_{k}": v for k, v in verdict.items()
                   if k in ("admissible", "reasons")},
            })
            print(f"    R19 tau={str(tau):6} batch={batch:5} "
                  f"ESS={stats['ess_fraction']:8.5f} "
                  f"neighbours={rows[-1]['effective_neighbours']:8.2f} "
                  f"collapsed={stats['collapsed_row_fraction']:5.3f} "
                  f"admissible={verdict['admissible']}", flush=True)

    # G5.5: at fixed tau = 0.05, do effective neighbours rise with batch?
    starved = [r for r in rows if r["tau"] == 0.05]
    starved.sort(key=lambda r: r["batch"])
    neighbours = [r["effective_neighbours"] for r in starved]
    monotone = all(
        np.isfinite(neighbours[i]) and np.isfinite(neighbours[i + 1])
        and neighbours[i] < neighbours[i + 1]
        for i in range(len(neighbours) - 1))
    excluded = [r for r in rows if not r["admission_admissible"]]
    return {
        "rows": rows,
        "neighbours_rise_with_batch": bool(monotone),
        "excluded_cells": len(excluded),
        "pass_G5_5": bool(monotone),
        "pass_G5_4": bool(excluded),
    }


def evaluate_gate(comparison: dict, r19: dict) -> dict:
    rows = comparison["rows"]

    def scores(arm: str) -> dict:
        return {(r["seed"], r["latent_dim"]): r.get("geometry_score_v2",
                                                    float("inf"))
                for r in rows if r["arm"] == arm}

    def paired(candidate: str, baseline: str) -> dict:
        left, right = scores(candidate), scores(baseline)
        keys = sorted(set(left) & set(right))
        if not keys:
            return {"ratio": float("nan"), "pairs": 0}
        return paired_log_ratio([left[k] for k in keys],
                               [right[k] for k in keys])

    def median(arm: str, key: str) -> float:
        values = [r.get(key) for r in rows if r["arm"] == arm]
        values = [float(v) for v in values
                  if isinstance(v, (int, float)) and np.isfinite(v)]
        return float(np.median(values)) if values else float("nan")

    r16 = paired(R16_ARM, BASELINE)
    dimension = median(R16_ARM, "effective_dimension_ratio")
    key = "median_trajectory_teacher_dimension_ratio"
    baseline_contraction = median(BASELINE, key)
    r16_contraction = median(R16_ARM, key)
    r11_contraction = median("D1", key)

    conditions = {
        "G5.1_R16_beats_baseline": {
            **r16, "threshold": R16_RATIO,
            "pass": bool(np.isfinite(r16.get("ratio", np.nan))
                         and r16["ratio"] <= R16_RATIO
                         and r16["high"] < 1.0)},
        "G5.2_R16_repairs_collapse": {
            "effective_dimension_ratio": dimension, "floor": DIMENSION_FLOOR,
            "pass": bool(np.isfinite(dimension)
                         and dimension >= DIMENSION_FLOOR)},
        "G5.3_mechanism_on_real_trajectory": {
            "baseline_teacher_dimension_ratio": baseline_contraction,
            "R16_teacher_dimension_ratio": r16_contraction,
            "R11_teacher_dimension_ratio": r11_contraction,
            "ceiling": CONTRACTION_CEILING,
            "pass": bool(
                np.isfinite(baseline_contraction)
                and baseline_contraction < CONTRACTION_CEILING
                and max(
                    x for x in (r16_contraction, r11_contraction)
                    if np.isfinite(x)) > baseline_contraction)},
        "G5.4_R15_excludes_dead_kernels": {
            "excluded_cells": r19["excluded_cells"],
            "pass": r19["pass_G5_4"]},
        "G5.5_batch_temperature_interaction": {
            "neighbours_rise_with_batch": r19["neighbours_rise_with_batch"],
            "pass": r19["pass_G5_5"]},
    }
    return {
        "conditions": conditions,
        "gate_pass": all(bool(v["pass"]) for v in conditions.values()),
        "arm_median_score": {
            arm: median(arm, "geometry_score_v2")
            for arm in sorted({r["arm"] for r in rows})},
        "arm_median_dimension": {
            arm: median(arm, "effective_dimension_ratio")
            for arm in sorted({r["arm"] for r in rows})},
        "comparisons": {
            "D2/D0 (R16 alone)": paired("D2", "D0"),
            "D1/D0 (R11 alone)": paired("D1", "D0"),
            "D3/D0 (both)": paired("D3", "D0"),
            "D2/D1 (R16 vs R11)": paired("D2", "D1"),
            "D3/D1 (adding R16 to R11)": paired("D3", "D1"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--latent-dims", type=str, default="32,64,128")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase5.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if not cifar.available(args.root):
        raise SystemExit("CIFAR-10 is not present locally.")

    latent_dims = tuple(int(x) for x in args.latent_dims.split(","))
    started = time.time()
    print("--- R19 batch x temperature (with R15 admission) ---", flush=True)
    r19 = r19_batch_temperature(args.resolution, args.root)
    print("--- D0-D3 comparison (R16 vs R11) + R17 latent sweep ---",
          flush=True)
    comparison = d_arms_comparison(args.resolution, args.seeds, args.steps,
                                   args.root, latent_dims)
    gate = evaluate_gate(comparison, r19)

    payload = {
        "status": "phase5-reforms-R15-R19",
        "scope": f"CIFAR-10 at {args.resolution}x{args.resolution}, raw pixel "
                 "geometry, disjoint train/eval splits, no pretrained "
                 "encoder, no class labels; fresh seeds disjoint from every "
                 "earlier phase",
        "config": vars(args) | {"out": str(args.out)},
        "seed_offset": SEED_OFFSET,
        "arm_digests": {a.arm_id: config_digest(a) for a in phase5_arms()},
        "provenance": provenance(),
        "elapsed_seconds": time.time() - started,
        "gate": gate,
        "results": {"D_comparison": comparison, "R19": r19},
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 5 ===")
    for arm in sorted(gate["arm_median_score"]):
        print(f"  {arm} median score "
              f"{gate['arm_median_score'][arm]:7.3f}  eff_dim "
              f"{gate['arm_median_dimension'][arm]:5.3f}")
    print()
    for name, result in gate["comparisons"].items():
        if np.isfinite(result.get("ratio", np.nan)):
            print(f"  {name:28} {result['ratio']:6.4f} "
                  f"[{result['low']:6.4f},{result['high']:6.4f}] "
                  f"wins={result['wins']}/{result['pairs']}")
    print()
    for name, condition in gate["conditions"].items():
        print(f"  [{'PASS' if condition['pass'] else 'FAIL'}] {name}")
    print(f"\n  Phase-5 gate: {'PASS' if gate['gate_pass'] else 'FAIL'}")
    print(f"  wrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
