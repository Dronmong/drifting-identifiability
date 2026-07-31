"""Phase 8 (protocol `EncoderIndependentPhase8Protocol.md`).

Are capacity and R11 substitutes?

The generator-contraction pass identified the deficit as least-squares
shrinkage: a free dilation parameter declines to grow (gain -> 0.829), the
field's radial demand at the optimum is +0.0004, and fitting the particle
cloud gives second moment 0.933 -> 0.602 as parameters-per-value falls
2.23 -> 0.28.  If that mechanism operates in the real recipe then capacity
and R11 address the same thing, and the conv width -- pinned at 64 since
Phase 1 -- is the axis that has never been varied.

  8A  width x R11 at the good bandwidth; **this stage can retire R11**
  8B  how much of its own teacher does the generator realize, by width?

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase8
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
from .train import optimizer_report, train_arm

HERE = Path(__file__).resolve().parent

# Frozen (protocol section 2): disjoint from every earlier phase.
SEED_OFFSET = 14000

# Frozen thresholds (protocol section 2).
MOMENT_BAND = (0.7, 1.3)
SUPERSEDE_TOLERANCE = 1.25
WIDTHS = (32, 64, 128, 256)
GOOD_ESS = 0.9
FIELD_CLOUD = 256


def _geometry() -> GeometryConfig:
    return GeometryConfig(family="raw", base_kernel="smooth_laplace",
                          target_ess_fraction=GOOD_ESS)


def _arm(correction: str) -> ArmConfig:
    return ArmConfig(
        "W", False, _geometry(), FieldConfig(direction_mode="paper"),
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


def _tail_fraction(samples: torch.Tensor, keep: int = 32) -> float:
    """Variance beyond the top ``keep`` directions -- the shape readout."""
    flat = samples.reshape(len(samples), -1)
    centred = flat - flat.mean(dim=0, keepdim=True)
    power = torch.linalg.svdvals(centred) ** 2
    total = float(power.sum())
    return float(power[keep:].sum()) / total if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# 8A: the width sweep
# ---------------------------------------------------------------------------


def stage_8a(resolution: int, seeds: int, steps: int, root: str | None,
             widths=WIDTHS) -> dict:
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    rows = []
    for width in widths:
        config = TrainConfig(steps=steps, batch=64, field_cloud=FIELD_CLOUD,
                             controller_batch=32, audit_batch=32,
                             eval_samples=512, image_size=resolution,
                             width=width)
        for index in range(seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            pools = evaluation_pools(evaluation, config, seed)
            null = null_reference(evaluation, pools, seed)
            for correction in ("none", "scalar"):
                arm = _arm(correction)
                outcome = train_arm(arm, train, config, seed)
                latent = sample_latent(config.eval_samples, config.latent_dim,
                                       derive_seed(seed, "eval-latent"))
                with torch.no_grad():
                    generated = outcome.model(latent)
                row = {"stage": "8a", "width": width, "seed": seed,
                       "r11": correction == "scalar",
                       "parameters": outcome.model.parameter_count(),
                       "tail_fraction": _tail_fraction(generated),
                       "reference_tail_fraction": _tail_fraction(
                           pools["eval"])}
                row.update(evaluate_arm(outcome, evaluation, config, pools,
                                        null, seed))
                row["admissible"] = kernel_admissible(
                    {"collapsed_row_fraction": row.get(
                        "median_branch_raw_collapsed_row_fraction", 0.0),
                     "ess_fraction": row.get(
                         "median_branch_raw_ess_fraction", float("nan"))},
                    FIELD_CLOUD)["admissible"]
                rows.append(row)
                print(f"    8A width={width:4} seed{index} "
                      f"R11={str(row['r11']):5} "
                      f"params={row['parameters']:8} "
                      f"ed2={row.get('ed2', float('nan')):8.4f} "
                      f"2nd={row.get('second_moment_ratio', float('nan')):6.3f} "
                      f"tail={row['tail_fraction']:6.4f} "
                      f"{'OK ' if row['admissible'] else 'DEAD'} "
                      f"{outcome.wall_seconds:6.1f}s", flush=True)
    return {"rows": rows, "cells": _cells_8a(rows, widths),
            "trend": _trend_8a(rows, widths)}


def _cells_8a(rows: list[dict], widths) -> dict:
    cells = {}
    for width in widths:
        for r11 in (False, True):
            group = [r for r in rows if r["width"] == width
                     and r["r11"] is r11]
            if not group:
                continue
            cells[f"width={width}_R11={r11}"] = {
                "width": width, "r11": r11,
                "parameters": group[0]["parameters"],
                "median_ed2": _median(group, "ed2"),
                "median_score_v2": _median(group, "geometry_score_v2"),
                "median_second_moment_ratio": _median(
                    group, "second_moment_ratio"),
                "median_tail_fraction": _median(group, "tail_fraction"),
                "admissible": all(r["admissible"] for r in group),
                "moment_in_band": _moment_ok(
                    _median(group, "second_moment_ratio")),
                "seeds": len(group)}
    return cells


def _trend_8a(rows: list[dict], widths) -> dict:
    """The declared secondary prediction: R11's edge shrinks with width."""
    trend = {}
    for width in widths:
        plain = sorted([r for r in rows if r["width"] == width
                        and not r["r11"]], key=lambda r: r["seed"])
        matched = sorted([r for r in rows if r["width"] == width
                          and r["r11"]], key=lambda r: r["seed"])
        if not plain or len(plain) != len(matched):
            continue
        comparison = paired_log_ratio(
            [r.get("ed2", float("nan")) for r in matched],
            [r.get("ed2", float("nan")) for r in plain])
        trend[f"width={width}"] = {
            "r11_over_plain_ed2": comparison["ratio"],
            "low": comparison["low"], "high": comparison["high"],
            "plain_second_moment": _median(plain, "second_moment_ratio"),
            "r11_second_moment": _median(matched, "second_moment_ratio")}
    ratios = [v["r11_over_plain_ed2"] for v in trend.values()
              if np.isfinite(v["r11_over_plain_ed2"])]
    moments = [v["plain_second_moment"] for v in trend.values()
               if np.isfinite(v["plain_second_moment"])]
    # "Shrinking advantage" means the R11/plain ratio RISES toward 1.
    rising = bool(len(ratios) >= 2 and ratios[-1] > ratios[0])
    return {
        "per_width": trend,
        "r11_advantage_shrinks_with_width": rising,
        "plain_second_moment_rises_with_width": bool(
            len(moments) >= 2 and moments[-1] > moments[0]),
        "plain_second_moment_span": ([min(moments), max(moments)]
                                     if moments else None),
        "meaning": ("capacity and R11 behave as substitutes"
                    if rising else
                    "R11's advantage does not shrink with capacity, so "
                    "least-squares shrinkage is not the operative term in "
                    "the real recipe"),
    }


def gate_8a(cells: dict) -> dict:
    """Can capacity supersede R11?  Declared before the run."""
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
        "meaning": ("capacity supersedes R11"
                    if qualifying else
                    "no width reproduces R11 without it"),
        "best_r11_cell": best_r11,
        "best_plain_cell": min(plain, key=lambda c: c["median_ed2"]),
        "ed2_ceiling": ceiling,
        "moment_band": list(MOMENT_BAND),
        "qualifying_cells": qualifying,
        "plain_cells_in_moment_band": [c for c in plain
                                       if c["moment_in_band"]],
    }


# ---------------------------------------------------------------------------
# 8B: how much of its own teacher does the generator realize?
# ---------------------------------------------------------------------------


def stage_8b(resolution: int, seeds: int, steps: int, root: str | None,
             widths=WIDTHS) -> dict:
    """Closes the gap the contraction pass flagged: fixed cloud vs recipe.

    The shrinkage account was demonstrated on a *fixed* cloud.  The recipe
    regresses onto a moving distribution, where N4 found latent dimension
    flat.  If the account transfers, the realized fraction of the teacher
    displacement must rise toward 1 with width.
    """
    train = cifar.cifar_target(resolution, "train", root)
    geometry = _geometry()
    rows = []
    for width in widths:
        config = TrainConfig(steps=steps, batch=64, field_cloud=FIELD_CLOUD,
                             eval_samples=512, image_size=resolution,
                             width=width)
        for index in range(seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            rng = np.random.default_rng(derive_seed(seed, "8b"))
            branch = build_family(geometry, 3).branches[0]
            kernel = calibrate_block_kernel(
                branch, train.sample(256, rng), "smooth_laplace",
                geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
                geometry.kernel_eps, combine=geometry.combine,
                target_ess_fraction=GOOD_ESS)
            model = OneStepGenerator(config.latent_dim, 3, resolution, width,
                                     derive_seed(seed, "generator"))
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=config.learning_rate)
            generator = torch.Generator().manual_seed(
                derive_seed(seed, "8b-latent") % (2 ** 31))
            realized = []
            for step in range(steps):
                positives = train.sample(config.batch, rng)
                latent = torch.randn(FIELD_CLOUD, config.latent_dim,
                                     generator=generator)
                before = model(latent)
                with torch.no_grad():
                    drift, _ = KG.field(before.detach(), positives,
                                        before.detach(), branch, kernel,
                                        direction_mode="paper",
                                        normalization="rms",
                                        diagnostics=False)
                    teacher = before.detach() + 0.5 * drift
                loss = ((before - teacher) ** 2).flatten(1).sum(1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % max(steps // 40, 1) == 0:
                    with torch.no_grad():
                        after = model(latent)
                        want = (teacher - before.detach()).flatten(1)
                        got = (after - before.detach()).flatten(1)
                        denominator = float((want * want).sum())
                        if denominator > 0:
                            realized.append(
                                float((got * want).sum()) / denominator)
            rows.append({"width": width, "seed": seed,
                         "parameters": model.parameter_count(),
                         "median_realized": float(np.median(realized))
                         if realized else float("nan"),
                         "final_realized": realized[-1] if realized
                         else float("nan")})
            print(f"    8B width={width:4} seed{index} "
                  f"realized_median={rows[-1]['median_realized']:8.5f} "
                  f"final={rows[-1]['final_realized']:8.5f}", flush=True)
    summary = {}
    for width in widths:
        group = [r for r in rows if r["width"] == width]
        if group:
            summary[f"width={width}"] = {
                "parameters": group[0]["parameters"],
                "median_realized": _median(group, "median_realized"),
                "median_final_realized": _median(group, "final_realized")}
    values = [v["median_realized"] for v in summary.values()
              if np.isfinite(v["median_realized"])]
    summary_note = {
        "realized_rises_with_width": bool(
            len(values) >= 2 and values[-1] > values[0]),
        "realized_span": [min(values), max(values)] if values else None}
    return {"rows": rows, "summary": summary, "trend": summary_note}


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=("all", "8a", "8b"))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--widths", type=str, default="32,64,128,256")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase8.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    widths = tuple(int(x) for x in args.widths.split(","))

    started = time.time()
    payload: dict = {
        "status": "phase8-frozen-protocol",
        "protocol": "numerics/EncoderIndependentPhase8Protocol.md",
        "provenance": provenance(),
        "config": vars(args) | {"out": str(args.out)},
        "config_digest": config_digest(
            TrainConfig(steps=args.steps, image_size=args.resolution)),
        "optimizer": optimizer_report(TrainConfig()),
        "frozen_thresholds": {
            "moment_band": list(MOMENT_BAND),
            "supersede_tolerance": SUPERSEDE_TOLERANCE,
            "widths": list(widths),
            "target_ess": GOOD_ESS,
            "field_cloud": FIELD_CLOUD,
        },
    }

    if args.stage in ("all", "8a"):
        print("=== 8A: width x R11 at the good bandwidth ===", flush=True)
        stage = stage_8a(args.resolution, args.seeds, args.steps,
                         args.data_root, widths)
        gate = gate_8a(stage["cells"])
        payload["stage_8a"] = stage | {"gate": gate}
        print(f"\n  8A gate: {gate['meaning']}", flush=True)
        print(f"  8A trend: {stage['trend']['meaning']}", flush=True)

    if args.stage in ("all", "8b"):
        print("\n=== 8B: realized fraction of the teacher ===", flush=True)
        payload["stage_8b"] = stage_8b(args.resolution, args.seeds,
                                       args.steps, args.data_root, widths)

    payload["elapsed_seconds"] = time.time() - started
    digest = write_json(args.out, payload)

    print("\n=== PHASE 8 ===")
    if "stage_8a" in payload:
        cells = payload["stage_8a"]["cells"]
        print(f"{'cell':24}{'params':>10}{'ed2':>9}{'score':>9}"
              f"{'2nd_mom':>9}{'tail':>8}  band")
        for key in sorted(cells, key=lambda k: (cells[k]["width"],
                                                cells[k]["r11"])):
            c = cells[key]
            print(f"{key:24}{c['parameters']:10}{c['median_ed2']:9.4f}"
                  f"{c['median_score_v2']:9.3f}"
                  f"{c['median_second_moment_ratio']:9.3f}"
                  f"{c['median_tail_fraction']:8.4f}"
                  f"  {'in ' if c['moment_in_band'] else 'out'}")
        trend = payload["stage_8a"]["trend"]
        print(f"\n{'width':10}{'R11/plain ed2':>16}{'95% interval':>24}"
              f"{'plain 2nd':>12}")
        for key, entry in trend["per_width"].items():
            interval = f"[{entry['low']:.3f}, {entry['high']:.3f}]"
            print(f"{key:10}{entry['r11_over_plain_ed2']:16.3f}"
                  f"{interval:>24}{entry['plain_second_moment']:12.3f}")
        gate = payload["stage_8a"]["gate"]
        print(f"\n  gate passed={gate['passed']}  -- {gate['meaning']}")
        print(f"  trend: {trend['meaning']}")
    if "stage_8b" in payload:
        print(f"\n{'width':12}{'params':>10}{'realized':>12}{'final':>12}")
        for key, entry in payload["stage_8b"]["summary"].items():
            print(f"{key:12}{entry['parameters']:10}"
                  f"{entry['median_realized']:12.5f}"
                  f"{entry['median_final_realized']:12.5f}")
        print(f"  realized rises with width: "
              f"{payload['stage_8b']['trend']['realized_rises_with_width']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
