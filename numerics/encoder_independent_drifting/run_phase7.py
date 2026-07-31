"""Phase 7 (protocol `EncoderIndependentPhase7Protocol.md`).

Ask of the kernel what Phase 6A asked of the optimizer.

6A cleared the optimizer: 0 of 12 cells reached the second-moment band across
a 60x learning-rate range.  The Phase-6 follow-up then found a much
better-supported candidate -- across every admissible bandwidth the
free-particle fixed point and its quality are monotone in the kernel's
realized neighbour count, the program has used the bad end since Phase 2, and
free particles at tau = 1 already beat the R11-corrected generator.

  7A  sweep bandwidth x field cloud x R11; **this stage can retire R11**
  7B  particle vs generator at matched bandwidth and cloud -- reported
  7C  is there a target-only bandwidth RULE, or only a lucky value?

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase7
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from . import metrics as M
from .config import (
    ArmConfig, FieldConfig, GeometryConfig, MASTER_SEED, MixtureConfig,
    ObjectiveConfig, TrainConfig, config_digest, derive_seed,
)
from .diagnostics import kernel_admissible, provenance, write_json
from .evaluate import evaluate_arm, evaluation_pools, null_reference
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .train import optimizer_report, train_arm

HERE = Path(__file__).resolve().parent

# Frozen (protocol section 2): disjoint from every earlier phase.
SEED_OFFSET = 12000

# Frozen thresholds (protocol sections 2 and 4).
MOMENT_BAND = (0.7, 1.3)
SUPERSEDE_TOLERANCE = 1.25

# (label, normalized tau, target ESS fraction).  A tau pins the paper's rule
# and overrides the ESS calibration; None with an ESS fraction is the
# repository's own rule.
BANDWIDTHS = (("ess=0.50", None, 0.50), ("ess=0.90", None, 0.90),
              ("tau=1.00", 1.00, None), ("tau=2.00", 2.00, None))
CLOUDS = (64, 256, 512)


def _geometry(tau: float | None, ess: float | None) -> GeometryConfig:
    return GeometryConfig(family="raw", base_kernel="smooth_laplace",
                          bandwidth_tau=tau, target_ess_fraction=ess,
                          bandwidth_multiplier=1.0)


def _arm(arm_id: str, geometry: GeometryConfig, correction: str) -> ArmConfig:
    return ArmConfig(
        arm_id, False, geometry, FieldConfig(direction_mode="paper"),
        MixtureConfig(adaptive=False),
        ObjectiveConfig(lambda_anchor=0.0, lambda_geometry=1.0,
                        teacher_correction=correction),
        note=f"correction={correction}")


def _moment_ok(value: float) -> bool:
    low, high = MOMENT_BAND
    return bool(np.isfinite(value) and low <= value <= high)


def _median(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows
              if r.get(key) is not None and np.isfinite(r.get(key, np.nan))]
    return float(np.median(values)) if values else float("nan")


def _admissible(row: dict, batch: int) -> dict:
    """Reform R15 read off a trained row's logged kernel health."""
    return kernel_admissible(
        {"collapsed_row_fraction": row.get(
            "median_branch_raw_collapsed_row_fraction", 0.0),
         "ess_fraction": row.get("median_branch_raw_ess_fraction",
                                 float("nan"))},
        batch)


def _kernel_for(train, geometry: GeometryConfig, rng):
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile,
        geometry.bandwidth_tau if geometry.bandwidth_tau is not None
        else geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=geometry.target_ess_fraction)
    return branch, kernel


def _target_only_ess(train, branch, kernel, rng, batch: int = 64) -> float:
    """The kernel's effective neighbour count on TARGET data alone.

    The quantity a calibration rule is allowed to see (plan section 6.2:
    calibrate from target data, never from generated output or a metric).
    """
    sample = train.sample(batch, rng)
    _, stats = KG.field(sample, train.sample(batch, rng), sample, branch,
                        kernel, direction_mode="paper", normalization="rms",
                        diagnostics=True)
    return float(stats.get("ess_fraction", float("nan")))


# ---------------------------------------------------------------------------
# 7A: sweep the kernel for the generator
# ---------------------------------------------------------------------------


def stage_7a(resolution: int, seeds: int, steps: int, root: str | None,
             ) -> dict:
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for label, tau, ess in BANDWIDTHS:
        geometry = _geometry(tau, ess)
        for cloud in CLOUDS:
            config = TrainConfig(steps=steps, batch=64, field_cloud=cloud,
                                 controller_batch=32, audit_batch=32,
                                 eval_samples=512, image_size=resolution)
            for index in range(seeds):
                seed = MASTER_SEED + SEED_OFFSET + index
                pools = evaluation_pools(evaluation, config, seed)
                null = null_reference(evaluation, pools, seed)
                for correction in ("none", "scalar"):
                    arm = _arm("G", geometry, correction)
                    outcome = train_arm(arm, train, config, seed)
                    row = {"stage": "7a", "bandwidth": label, "tau": tau,
                           "ess_target": ess, "cloud": cloud, "seed": seed,
                           "r11": correction == "scalar"}
                    row.update(evaluate_arm(outcome, evaluation, config,
                                            pools, null, seed))
                    row["admissible"] = _admissible(row, cloud)["admissible"]
                    rows.append(row)
                    print(f"    7A {label:9} cloud={cloud:4} seed{index} "
                          f"R11={str(row['r11']):5} "
                          f"ed2={row.get('ed2', float('nan')):8.4f} "
                          f"2nd={row.get('second_moment_ratio', float('nan')):6.3f} "
                          f"{'OK ' if row['admissible'] else 'DEAD'} "
                          f"{outcome.wall_seconds:6.1f}s", flush=True)
    return {"rows": rows, "cells": _cells_7a(rows)}


def _cells_7a(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        key = f"{row['bandwidth']}_cloud={row['cloud']}_R11={row['r11']}"
        groups.setdefault(key, []).append(row)
    return {
        key: {
            "bandwidth": group[0]["bandwidth"], "cloud": group[0]["cloud"],
            "r11": group[0]["r11"],
            "median_ed2": _median(group, "ed2"),
            "median_score_v2": _median(group, "geometry_score_v2"),
            "median_second_moment_ratio": _median(group,
                                                  "second_moment_ratio"),
            "median_ess_fraction": _median(
                group, "median_branch_raw_ess_fraction"),
            "admissible": all(r["admissible"] for r in group),
            "moment_in_band": _moment_ok(
                _median(group, "second_moment_ratio")),
            "seeds": len(group),
        }
        for key, group in groups.items()
    }


def gate_7a(cells: dict) -> dict:
    """Can setting the kernel correctly supersede R11?

    Declared before the run.  Inadmissible cells cannot qualify -- a dead
    kernel is not evidence about kernel geometry -- but they are reported.
    """
    admissible = [c for c in cells.values() if c["admissible"]
                  and np.isfinite(c["median_ed2"])]
    r11 = [c for c in admissible if c["r11"]]
    plain = [c for c in admissible if not c["r11"]]
    if not r11 or not plain:
        return {"passed": False, "reason": "no admissible measurable cells"}
    best_r11 = min(r11, key=lambda c: c["median_ed2"])
    ceiling = best_r11["median_ed2"] * SUPERSEDE_TOLERANCE
    qualifying = [c for c in plain
                  if c["moment_in_band"] and c["median_ed2"] <= ceiling]
    return {
        "passed": bool(qualifying),
        "meaning": ("R11 is superseded by setting the kernel correctly"
                    if qualifying else
                    "no admissible kernel reproduces R11 without it"),
        "best_r11_cell": best_r11,
        "best_plain_cell": min(plain, key=lambda c: c["median_ed2"]),
        "ed2_ceiling": ceiling,
        "moment_band": list(MOMENT_BAND),
        "qualifying_cells": qualifying,
        "plain_cells_in_moment_band": [c for c in plain
                                       if c["moment_in_band"]],
        "inadmissible_cells": [k for k, c in cells.items()
                               if not c["admissible"]],
    }


# ---------------------------------------------------------------------------
# 7B: particle versus generator at matched bandwidth and cloud
# ---------------------------------------------------------------------------


def stage_7b(resolution: int, seeds: int, steps: int, root: str | None,
             cloud: int = 512, eta: float = 0.2) -> dict:
    """Does the ordering really invert once the kernel is set properly?"""
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    config = TrainConfig(steps=steps, batch=64, field_cloud=cloud,
                         controller_batch=32, audit_batch=32,
                         eval_samples=cloud, image_size=resolution)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        rng = np.random.default_rng(derive_seed(seed, "7b"))
        for label, tau, ess in BANDWIDTHS:
            geometry = _geometry(tau, ess)
            branch, kernel = _kernel_for(train, geometry, rng)
            particles = torch.tensor(
                rng.normal(scale=0.5,
                           size=(cloud, 3, resolution, resolution)),
                dtype=torch.float32)
            for _ in range(steps):
                drift, _ = KG.field(particles, train.sample(64, rng),
                                    particles, branch, kernel,
                                    direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                particles = particles + eta * drift
            measured = M.raw_metrics(
                particles, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "7b-m")), None,
                target_null=pools["null"])
            rows.append({
                "stage": "7b", "regime": "particles", "bandwidth": label,
                "cloud": cloud, "seed": seed,
                "ed2": measured["ed2"],
                "geometry_score_v2": M.normalized_geometry_score_v2(
                    measured, null)["geometry_score"],
                "second_moment_ratio": float(
                    particles.flatten(1).var(0).mean()) / reference_moment})
            print(f"    7B {label:9} particles           seed{index} "
                  f"ed2={rows[-1]['ed2']:8.4f} "
                  f"2nd={rows[-1]['second_moment_ratio']:6.3f}", flush=True)
            for correction in ("none", "scalar"):
                arm = _arm("G", geometry, correction)
                outcome = train_arm(arm, train, config, seed)
                row = {"stage": "7b", "regime": f"generator_{correction}",
                       "bandwidth": label, "cloud": cloud, "seed": seed}
                row.update(evaluate_arm(outcome, evaluation, config, pools,
                                        null, seed))
                rows.append(row)
                print(f"    7B {label:9} generator_{correction:9} "
                      f"seed{index} "
                      f"ed2={row.get('ed2', float('nan')):8.4f} "
                      f"2nd={row.get('second_moment_ratio', float('nan')):6.3f}",
                      flush=True)
    return {"rows": rows, "summary": _summarize_7b(rows)}


def _summarize_7b(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(f"{row['bandwidth']}_{row['regime']}",
                          []).append(row)
    summary = {
        key: {"median_ed2": _median(group, "ed2"),
              "median_score_v2": _median(group, "geometry_score_v2"),
              "median_second_moment_ratio": _median(group,
                                                    "second_moment_ratio"),
              "moment_in_band": _moment_ok(
                  _median(group, "second_moment_ratio"))}
        for key, group in sorted(groups.items())}
    # The amortization gap: what the generator costs against the particles
    # it is meant to stand in for, at each bandwidth.
    for label, _, _ in BANDWIDTHS:
        particle = summary.get(f"{label}_particles", {})
        for correction in ("none", "scalar"):
            generator = summary.get(f"{label}_generator_{correction}", {})
            if particle.get("median_ed2") and generator.get("median_ed2"):
                summary[f"{label}_generator_{correction}"][
                    "ed2_over_particles"] = (generator["median_ed2"]
                                             / particle["median_ed2"])
    return summary


# ---------------------------------------------------------------------------
# 7C: is there a target-only bandwidth rule?
# ---------------------------------------------------------------------------


def stage_7c(resolution: int, seeds: int, steps: int, root: str | None,
             cloud: int = 512, eta: float = 0.2) -> dict:
    """Locate the turn, and ask whether target data alone predicts it."""
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    config = TrainConfig(steps=steps, batch=64, eval_samples=cloud,
                         image_size=resolution)
    arms = ([(f"tau={t:g}", t, None) for t in (1.0, 2.0, 4.0, 8.0)]
            + [(f"ess={e:g}", None, e) for e in (0.5, 0.7, 0.9, 0.95, 0.99)])
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())
        rng = np.random.default_rng(derive_seed(seed, "7c"))
        for label, tau, ess in arms:
            geometry = _geometry(tau, ess)
            branch, kernel = _kernel_for(train, geometry, rng)
            target_ess = _target_only_ess(train, branch, kernel, rng)
            particles = torch.tensor(
                rng.normal(scale=0.5,
                           size=(cloud, 3, resolution, resolution)),
                dtype=torch.float32)
            health = {}
            for step in range(steps):
                drift, stats = KG.field(
                    particles, train.sample(64, rng), particles, branch,
                    kernel, direction_mode="paper", normalization="rms",
                    diagnostics=(step == 0))
                if step == 0:
                    health = stats
                particles = particles + eta * drift
            measured = M.raw_metrics(
                particles, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "7c-m")), None,
                target_null=pools["null"])
            admissible = kernel_admissible(health, cloud)
            rows.append({
                "stage": "7c", "arm": label, "tau": tau, "ess_target": ess,
                "seed": seed,
                "target_only_ess": target_ess,
                "generated_ess": float(health.get("ess_fraction",
                                                  float("nan"))),
                "admissible": admissible["admissible"],
                "ed2": measured["ed2"],
                "geometry_score_v2": M.normalized_geometry_score_v2(
                    measured, null)["geometry_score"],
                "second_moment_ratio": float(
                    particles.flatten(1).var(0).mean()) / reference_moment})
            print(f"    7C {label:10} seed{index} "
                  f"target_ess={target_ess:6.4f} "
                  f"ed2={rows[-1]['ed2']:8.4f} "
                  f"2nd={rows[-1]['second_moment_ratio']:6.3f} "
                  f"{'OK ' if admissible['admissible'] else 'DEAD'}",
                  flush=True)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["arm"], []).append(row)
    summary = {
        key: {"median_target_only_ess": _median(group, "target_only_ess"),
              "median_ed2": _median(group, "ed2"),
              "median_score_v2": _median(group, "geometry_score_v2"),
              "median_second_moment_ratio": _median(group,
                                                    "second_moment_ratio"),
              "admissible": all(r["admissible"] for r in group),
              "moment_in_band": _moment_ok(
                  _median(group, "second_moment_ratio"))}
        for key, group in sorted(groups.items())}
    return {"rows": rows, "summary": summary, "rule": _rule_7c(summary)}


def _rule_7c(summary: dict) -> dict:
    """Does one target-only ESS value identify the good arms?"""
    usable = {k: v for k, v in summary.items()
              if v["admissible"] and np.isfinite(v["median_ed2"])
              and np.isfinite(v["median_target_only_ess"])}
    if len(usable) < 3:
        return {"available": False, "reason": "too few admissible arms"}
    best = min(usable, key=lambda k: usable[k]["median_ed2"])
    quality = np.array([usable[k]["median_ed2"] for k in usable])
    ess = np.array([usable[k]["median_target_only_ess"] for k in usable])
    # Rank correlation: does more target-only ESS mean better quality?
    order_e = np.argsort(np.argsort(ess))
    order_q = np.argsort(np.argsort(quality))
    n = len(usable)
    spearman = float(1 - 6 * ((order_e - order_q) ** 2).sum()
                     / (n * (n ** 2 - 1))) if n > 1 else float("nan")
    return {
        "available": True,
        "best_arm": best,
        "best_arm_target_only_ess": usable[best]["median_target_only_ess"],
        "best_arm_ed2": usable[best]["median_ed2"],
        "spearman_target_ess_vs_ed2": spearman,
        "monotone": bool(spearman <= -0.9),
        "meaning": ("target-only ESS orders quality, so it is a usable "
                    "calibration statistic" if spearman <= -0.9 else
                    "target-only ESS does not order quality on its own"),
        "arms": usable,
    }


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=("all", "7a", "7b", "7c"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase7.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    payload: dict = {
        "status": "phase7-frozen-protocol",
        "protocol": "numerics/EncoderIndependentPhase7Protocol.md",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "config_digest": config_digest(
            TrainConfig(steps=args.steps, image_size=args.resolution)),
        "optimizer": optimizer_report(TrainConfig()),
        "frozen_thresholds": {
            "moment_band": list(MOMENT_BAND),
            "supersede_tolerance": SUPERSEDE_TOLERANCE,
            "bandwidths": [b[0] for b in BANDWIDTHS],
            "clouds": list(CLOUDS),
        },
    }

    if args.stage in ("all", "7a"):
        print("=== 7A: sweep the kernel for the generator ===", flush=True)
        stage = stage_7a(args.resolution, args.seeds, args.steps,
                         args.data_root)
        gate = gate_7a(stage["cells"])
        payload["stage_7a"] = stage | {"gate": gate}
        print(f"\n  7A gate: {gate['meaning']}", flush=True)

    if args.stage in ("all", "7b"):
        print("\n=== 7B: particle versus generator, matched ===", flush=True)
        payload["stage_7b"] = stage_7b(args.resolution, args.seeds,
                                       args.steps, args.data_root)

    if args.stage in ("all", "7c"):
        print("\n=== 7C: is there a target-only bandwidth rule? ===",
              flush=True)
        payload["stage_7c"] = stage_7c(args.resolution, args.seeds,
                                       args.steps, args.data_root)

    payload["elapsed_seconds"] = time.time() - started
    digest = write_json(args.out, payload)

    print("\n=== PHASE 7 ===")
    if "stage_7a" in payload:
        cells = payload["stage_7a"]["cells"]
        print(f"{'cell':38}{'ed2':>9}{'score':>9}{'2nd_mom':>9}"
              f"{'ESS':>8}  band  health")
        for key in sorted(cells):
            c = cells[key]
            print(f"{key:38}{c['median_ed2']:9.4f}"
                  f"{c['median_score_v2']:9.3f}"
                  f"{c['median_second_moment_ratio']:9.3f}"
                  f"{c['median_ess_fraction']:8.3f}"
                  f"  {'in ' if c['moment_in_band'] else 'out'}"
                  f"   {'OK' if c['admissible'] else 'DEAD'}")
        gate = payload["stage_7a"]["gate"]
        print(f"\n  7A gate passed={gate['passed']}  -- {gate['meaning']}")
    if "stage_7b" in payload:
        print(f"\n{'regime':34}{'ed2':>9}{'2nd_mom':>9}{'vs particles':>14}")
        for key, entry in payload["stage_7b"]["summary"].items():
            ratio = entry.get("ed2_over_particles")
            print(f"{key:34}{entry['median_ed2']:9.4f}"
                  f"{entry['median_second_moment_ratio']:9.3f}"
                  f"{(f'{ratio:.2f}x' if ratio else '--'):>14}")
    if "stage_7c" in payload:
        print(f"\n{'arm':12}{'target_ESS':>12}{'ed2':>9}{'2nd_mom':>9}"
              f"  band  health")
        for key, entry in payload["stage_7c"]["summary"].items():
            print(f"{key:12}{entry['median_target_only_ess']:12.4f}"
                  f"{entry['median_ed2']:9.4f}"
                  f"{entry['median_second_moment_ratio']:9.3f}"
                  f"  {'in ' if entry['moment_in_band'] else 'out'}"
                  f"   {'OK' if entry['admissible'] else 'DEAD'}")
        rule = payload["stage_7c"]["rule"]
        if rule.get("available"):
            print(f"\n  rule: {rule['meaning']}  "
                  f"(spearman {rule['spearman_target_ess_vs_ed2']:+.3f}, "
                  f"best {rule['best_arm']} at target ESS "
                  f"{rule['best_arm_target_only_ess']:.4f})")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
