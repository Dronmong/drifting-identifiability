"""Phase 6 (protocol `EncoderIndependentPhase6Protocol.md`).

Is R11 a repair, or a workaround for an unswept learning rate?

`step_eta` is inert under Adam -- it enters the stop-gradient loss only as a
constant gradient multiplier, and Adam normalizes that away -- so the real
step control is the optimizer and its learning rate, which were fixed at Adam
2e-3 by the Phase-1 pre-registration and inherited unexamined for six phases.
The program's one confirmed positive result has therefore never been compared
against the plainest available explanation.

  6A  sweep the real control variable; **this stage can retire R11**
  6B  richer direction changes (R26), run only if R11 survives 6A
  6C  particle algorithm versus generator port, matched -- scoping, not gated

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase6
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
from .diagnostics import (
    DIMENSION_BAND, dimension_verdict, paired_log_ratio, provenance,
    write_json,
)
from .evaluate import evaluate_arm, evaluation_pools, null_reference
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import sample_latent
from .train import optimizer_report, train_arm

HERE = Path(__file__).resolve().parent

# Frozen (protocol section 2): disjoint from every earlier phase.
SEED_OFFSET = 10000

# Frozen thresholds (protocol sections 2 and 3).
MOMENT_BAND = (0.7, 1.3)          # the second-moment ratio must land here
SUPERSEDE_TOLERANCE = 1.25        # 6A: "within 25% of the best R11 cell"
IMPROVEMENT_RATIO = 0.90          # 6B: "beats E1 by >= 10%"

OPTIMIZERS = ("adam", "sgd", "sgd_momentum")
LEARNING_RATES = (5e-4, 2e-3, 8e-3, 3e-2)


def _geometry() -> GeometryConfig:
    return GeometryConfig(family="raw", base_kernel="smooth_laplace")


def _arm(arm_id: str, *, correction: str = "none", gain: float = 1.0,
         note: str = "") -> ArmConfig:
    return ArmConfig(
        arm_id, False, _geometry(), FieldConfig(direction_mode="paper"),
        MixtureConfig(adaptive=False),
        ObjectiveConfig(lambda_anchor=0.0, lambda_geometry=1.0,
                        teacher_correction=correction,
                        teacher_correction_gain=gain),
        note=note)


def _moment_ok(value: float) -> bool:
    low, high = MOMENT_BAND
    return bool(np.isfinite(value) and low <= value <= high)


def _median(rows: list[dict], key: str) -> float:
    values = [r[key] for r in rows
              if np.isfinite(r.get(key, float("nan")))]
    return float(np.median(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# 6A: sweep the real control variable
# ---------------------------------------------------------------------------


def stage_6a(resolution: int, seeds: int, steps: int, root: str | None,
             optimizers=OPTIMIZERS, rates=LEARNING_RATES) -> dict:
    """The decisive stage: does setting the step correctly supersede R11?

    Every cell is reported in full.  The grid is declared in the protocol and
    is a sweep, not a search for the best cell to headline.
    """
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    baseline = _arm("F0", note="uncorrected")
    matched = _arm("F1", correction="scalar", note="R11 scalar match")
    rows = []
    for optimizer in optimizers:
        for rate in rates:
            config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                                 audit_batch=32, eval_samples=512,
                                 image_size=resolution, optimizer=optimizer,
                                 learning_rate=rate)
            for index in range(seeds):
                seed = MASTER_SEED + SEED_OFFSET + index
                pools = evaluation_pools(evaluation, config, seed)
                null = null_reference(evaluation, pools, seed)
                for arm in (baseline, matched):
                    outcome = train_arm(arm, train, config, seed)
                    row = {"stage": "6a", "arm": arm.arm_id, "seed": seed,
                           "r11": arm.arm_id == "F1"}
                    row.update(evaluate_arm(outcome, evaluation, config,
                                            pools, null, seed))
                    rows.append(row)
                    print(f"    6A {optimizer:12} lr={rate:<8g} seed{index} "
                          f"{arm.arm_id} "
                          f"score={row.get('geometry_score_v2', float('nan')):8.3f} "
                          f"ed2={row.get('ed2', float('nan')):8.4f} "
                          f"2nd={row.get('second_moment_ratio', float('nan')):6.3f} "
                          f"{outcome.wall_seconds:5.1f}s", flush=True)
    return {"rows": rows, "cells": _cells_6a(rows)}


def _cells_6a(rows: list[dict]) -> dict:
    cells: dict[str, dict] = {}
    for row in rows:
        key = (f"{row['optimizer']}_lr={row['learning_rate']:g}"
               f"_R11={row['r11']}")
        cells.setdefault(key, []).append(row)
    return {
        key: {
            "optimizer": group[0]["optimizer"],
            "learning_rate": group[0]["learning_rate"],
            "r11": group[0]["r11"],
            "median_ed2": _median(group, "ed2"),
            "median_score_v2": _median(group, "geometry_score_v2"),
            "median_second_moment_ratio": _median(group,
                                                  "second_moment_ratio"),
            "median_dimension_ratio": _median(group,
                                              "effective_dimension_ratio"),
            "dimension_verdict": dimension_verdict(
                _median(group, "effective_dimension_ratio"))["direction"],
            "moment_in_band": _moment_ok(
                _median(group, "second_moment_ratio")),
            "seeds": len(group),
        }
        for key, group in cells.items()
    }


def gate_6a(cells: dict) -> dict:
    """Does any no-R11 cell match the best R11 cell with a healthy scale?

    Declared before the run: if one does, R11 is superseded by setting the
    step correctly, Phases 3 and 5 are re-scoped, and 6B is not run.
    """
    r11 = [c for c in cells.values() if c["r11"]
           and np.isfinite(c["median_ed2"])]
    plain = [c for c in cells.values() if not c["r11"]
             and np.isfinite(c["median_ed2"])]
    if not r11 or not plain:
        return {"passed": False, "reason": "no measurable cells"}
    best_r11 = min(r11, key=lambda c: c["median_ed2"])
    ceiling = best_r11["median_ed2"] * SUPERSEDE_TOLERANCE
    qualifying = [c for c in plain
                  if c["moment_in_band"] and c["median_ed2"] <= ceiling]
    best_plain = min(plain, key=lambda c: c["median_ed2"])
    return {
        "passed": bool(qualifying),
        "meaning": ("R11 is superseded by setting the step correctly"
                    if qualifying else
                    "no learning rate reproduces R11 without it"),
        "best_r11_cell": best_r11,
        "best_plain_cell": best_plain,
        "ed2_ceiling": ceiling,
        "moment_band": list(MOMENT_BAND),
        "qualifying_cells": qualifying,
        "plain_cells_in_moment_band": [
            c for c in plain if c["moment_in_band"]],
    }


# ---------------------------------------------------------------------------
# 6B: richer direction changes (R26)
# ---------------------------------------------------------------------------


def stage_6b(resolution: int, seeds: int, steps: int, root: str | None,
             optimizer: str = "adam", rate: float = 2e-3) -> dict:
    """Does resolving more directions beat the single scalar?"""
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    arms = [
        _arm("E0", note="no correction"),
        _arm("E1", correction="scalar", note="R11 scalar match (incumbent)"),
        _arm("E2", correction="per_coordinate",
             note="per-coordinate second-moment match"),
        _arm("E3", correction="eigendirection",
             note="per-principal-direction match"),
        _arm("E4", correction="scalar", gain=1.2,
             note="scalar match, declared gain 1.2"),
    ]
    config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                         audit_batch=32, eval_samples=512,
                         image_size=resolution, optimizer=optimizer,
                         learning_rate=rate)
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        for arm in arms:
            outcome = train_arm(arm, train, config, seed)
            row = {"stage": "6b", "arm": arm.arm_id, "seed": seed,
                   "note": arm.note,
                   "correction": arm.objective.teacher_correction,
                   "gain": arm.objective.teacher_correction_gain}
            row.update(evaluate_arm(outcome, evaluation, config, pools, null,
                                    seed))
            rows.append(row)
            print(f"    6B seed{index} {arm.arm_id} "
                  f"score={row.get('geometry_score_v2', float('nan')):8.3f} "
                  f"ed2={row.get('ed2', float('nan')):8.4f} "
                  f"2nd={row.get('second_moment_ratio', float('nan')):6.3f} "
                  f"{outcome.wall_seconds:5.1f}s", flush=True)
    return {"rows": rows, "comparisons": _compare_6b(rows)}


def _compare_6b(rows: list[dict], incumbent: str = "E1") -> dict:
    by_arm: dict[str, list[dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    for group in by_arm.values():
        group.sort(key=lambda r: r["seed"])
    if incumbent not in by_arm:
        return {}
    base = [r.get("geometry_score_v2", float("nan"))
            for r in by_arm[incumbent]]
    out = {}
    for arm, group in sorted(by_arm.items()):
        scores = [r.get("geometry_score_v2", float("nan")) for r in group]
        comparison = paired_log_ratio(scores, base)
        moment = _median(group, "second_moment_ratio")
        out[arm] = {
            "note": group[0].get("note", ""),
            "median_score_v2": _median(group, "geometry_score_v2"),
            "median_ed2": _median(group, "ed2"),
            "median_second_moment_ratio": moment,
            "moment_in_band": _moment_ok(moment),
            "median_correction_ratio_cap_fraction": _median(
                group, "median_correction_ratio_cap_fraction"),
            f"vs_{incumbent}": comparison,
        }
    return out


def gate_6b(comparisons: dict, incumbent: str = "E1") -> dict:
    """A richer correction must beat the scalar on every seed, in band."""
    winners = []
    for arm, entry in comparisons.items():
        if arm == incumbent:
            continue
        comparison = entry[f"vs_{incumbent}"]
        pairs = comparison["pairs"]
        if (pairs > 0 and comparison["ratio"] <= IMPROVEMENT_RATIO
                and comparison["high"] < 1.0
                and comparison["wins"] == pairs
                and entry["moment_in_band"]):
            winners.append(arm)
    return {
        "passed": bool(winners),
        "winners": winners,
        "meaning": ("a richer direction change beats the scalar match"
                    if winners else
                    "resolving more directions does not beat one scalar"),
        "requirement": {
            "ratio_at_most": IMPROVEMENT_RATIO,
            "bootstrap_high_below": 1.0,
            "wins_on_every_seed": True,
            "second_moment_band": list(MOMENT_BAND),
        },
    }


# ---------------------------------------------------------------------------
# 6C: particle algorithm versus generator port, matched
# ---------------------------------------------------------------------------


def stage_6c(resolution: int, seeds: int, steps: int, root: str | None,
             eta: float = 0.5) -> dict:
    """Is the second-moment deficit a property of the PORT, not the method?

    Identical field, batch, budget and target; the only difference is whether
    a parametric generator plus optimizer sits in the loop.  The particle arm
    is also run with the decaying step the free-particle runs have always
    used, since for particles -- which have no optimizer -- eta is the real
    step and the schedule is therefore not inert.
    """
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                         audit_batch=32, eval_samples=512,
                         image_size=resolution)
    geometry = _geometry()
    rows = []
    for index in range(seeds):
        seed = MASTER_SEED + SEED_OFFSET + index
        pools = evaluation_pools(evaluation, config, seed)
        null = null_reference(evaluation, pools, seed)
        rng = np.random.default_rng(derive_seed(seed, "6c"))
        branch = build_family(geometry, config.channels).branches[0]
        kernel = calibrate_block_kernel(
            branch, train.sample(256, rng), "smooth_laplace",
            geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
            geometry.kernel_eps, combine=geometry.combine,
            target_ess_fraction=geometry.target_ess_fraction)
        reference_moment = float(pools["eval"].flatten(1).var(0).mean())

        def record(regime: str, cloud: torch.Tensor, note: str) -> None:
            measured = M.raw_metrics(
                cloud, pools["eval"], pools["cal_a"], pools["cal_b"],
                np.random.default_rng(derive_seed(seed, "6c-metrics")), None,
                target_null=pools["null"])
            row = {"stage": "6c", "regime": regime, "seed": seed,
                   "note": note,
                   "geometry_score_v2": M.normalized_geometry_score_v2(
                       measured, null)["geometry_score"],
                   "ed2": measured["ed2"],
                   "effective_dimension_ratio": measured[
                       "effective_dimension_ratio"],
                   "second_moment_ratio": float(
                       cloud.flatten(1).var(0).mean()) / reference_moment}
            row["moment_in_band"] = _moment_ok(row["second_moment_ratio"])
            rows.append(row)
            print(f"    6C seed{index} {regime:22} "
                  f"score={row['geometry_score_v2']:8.3f} "
                  f"ed2={row['ed2']:8.4f} "
                  f"2nd={row['second_moment_ratio']:6.3f}", flush=True)

        # Particles: eta is the real step here, so both the matched constant
        # step and the decaying one the earlier runs used are reported.
        for schedule in ("constant", "linear_decay"):
            cloud = torch.tensor(
                rng.normal(scale=0.5,
                           size=(config.eval_samples, config.channels,
                                 resolution, resolution)),
                dtype=torch.float32)
            for step in range(steps):
                drift, _ = KG.field(cloud, train.sample(config.batch, rng),
                                    cloud, branch, kernel,
                                    direction_mode="paper",
                                    normalization="rms", diagnostics=False)
                factor = (1.0 - step / steps if schedule == "linear_decay"
                          else 1.0)
                cloud = cloud + eta * factor * drift
            record(f"particles_{schedule}", cloud,
                   f"paper Algorithm 2, no optimizer, eta={eta} {schedule}")

        # Generator port: the same field, the same budget, through a model.
        for correction in ("none", "scalar"):
            arm = _arm("G", correction=correction)
            outcome = train_arm(arm, train, config, seed)
            latent = sample_latent(config.eval_samples, config.latent_dim,
                                   derive_seed(seed, "6c-latent"))
            with torch.no_grad():
                record(f"generator_{correction}", outcome.model(latent),
                       f"stop-gradient port, Adam lr={config.learning_rate}, "
                       f"correction={correction}")
    return {"rows": rows, "summary": _summarize_6c(rows),
            "optimizer": optimizer_report(config)}


def _summarize_6c(rows: list[dict]) -> dict:
    by_regime: dict[str, list[dict]] = {}
    for row in rows:
        by_regime.setdefault(row["regime"], []).append(row)
    return {
        regime: {
            "note": group[0]["note"],
            "median_score_v2": _median(group, "geometry_score_v2"),
            "median_ed2": _median(group, "ed2"),
            "median_second_moment_ratio": _median(group,
                                                  "second_moment_ratio"),
            "moment_in_band": _moment_ok(
                _median(group, "second_moment_ratio")),
            "median_dimension_ratio": _median(group,
                                              "effective_dimension_ratio"),
        }
        for regime, group in sorted(by_regime.items())
    }


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=("all", "6a", "6b", "6c"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase6.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    started = time.time()
    payload: dict = {
        "status": "phase6-frozen-protocol",
        "protocol": "numerics/EncoderIndependentPhase6Protocol.md",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "config_digest": config_digest(
            TrainConfig(steps=args.steps, image_size=args.resolution)),
        "frozen_thresholds": {
            "moment_band": list(MOMENT_BAND),
            "supersede_tolerance": SUPERSEDE_TOLERANCE,
            "improvement_ratio": IMPROVEMENT_RATIO,
            "dimension_band": list(DIMENSION_BAND),
            "optimizers": list(OPTIMIZERS),
            "learning_rates": list(LEARNING_RATES),
        },
    }

    run_6b = args.stage in ("all", "6b")
    if args.stage in ("all", "6a"):
        print("=== 6A: sweep the real control variable ===", flush=True)
        stage = stage_6a(args.resolution, args.seeds, args.steps,
                         args.data_root)
        gate = gate_6a(stage["cells"])
        payload["stage_6a"] = stage | {"gate": gate}
        print(f"\n  6A gate: {gate['meaning']}", flush=True)
        if args.stage == "all" and gate["passed"]:
            # Declared branch: R11 superseded, so the successor stage is moot.
            run_6b = False
            payload["stage_6b_skipped"] = (
                "6A superseded R11; the protocol declares 6B is not run")
            print("  -> 6B skipped by the protocol's declared branch",
                  flush=True)

    if run_6b:
        print("\n=== 6B: richer direction changes ===", flush=True)
        stage = stage_6b(args.resolution, args.seeds, args.steps,
                         args.data_root)
        gate = gate_6b(stage["comparisons"])
        payload["stage_6b"] = stage | {"gate": gate}
        print(f"\n  6B gate: {gate['meaning']}", flush=True)

    if args.stage in ("all", "6c"):
        print("\n=== 6C: particle versus generator, matched ===", flush=True)
        payload["stage_6c"] = stage_6c(args.resolution, args.seeds,
                                       args.steps, args.data_root)

    payload["elapsed_seconds"] = time.time() - started
    digest = write_json(args.out, payload)

    print("\n=== PHASE 6 ===")
    if "stage_6a" in payload:
        cells = payload["stage_6a"]["cells"]
        print(f"{'cell':34}{'ed2':>9}{'score':>9}{'2nd_mom':>9}  band")
        for key in sorted(cells):
            c = cells[key]
            print(f"{key:34}{c['median_ed2']:9.4f}"
                  f"{c['median_score_v2']:9.3f}"
                  f"{c['median_second_moment_ratio']:9.3f}"
                  f"  {'in' if c['moment_in_band'] else 'out'}")
        print(f"\n  6A gate passed={payload['stage_6a']['gate']['passed']}"
              f"  -- {payload['stage_6a']['gate']['meaning']}")
    if "stage_6b" in payload:
        print(f"\n{'arm':6}{'ed2':>9}{'score':>9}{'2nd_mom':>9}"
              f"{'vs E1':>9}{'boot_hi':>9}")
        for arm, entry in sorted(payload["stage_6b"]["comparisons"].items()):
            comparison = entry["vs_E1"]
            print(f"{arm:6}{entry['median_ed2']:9.4f}"
                  f"{entry['median_score_v2']:9.3f}"
                  f"{entry['median_second_moment_ratio']:9.3f}"
                  f"{comparison['ratio']:9.3f}{comparison['high']:9.3f}")
        print(f"\n  6B gate passed={payload['stage_6b']['gate']['passed']}"
              f"  -- {payload['stage_6b']['gate']['meaning']}")
    if "stage_6c" in payload:
        print(f"\n{'regime':24}{'ed2':>9}{'score':>9}{'2nd_mom':>9}  band")
        for regime, entry in payload["stage_6c"]["summary"].items():
            print(f"{regime:24}{entry['median_ed2']:9.4f}"
                  f"{entry['median_score_v2']:9.3f}"
                  f"{entry['median_second_moment_ratio']:9.3f}"
                  f"  {'in' if entry['moment_in_band'] else 'out'}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
