"""Phase-3 corrected-baseline confirmation (protocol sections 4-5).

Fresh frozen test of reform R11 (teacher variance match) and R12 (the
paper's real Algorithm-2 field), on seeds never used in development.

The R11 finding was development evidence: two to three seeds, one
resolution, one budget, and the correction was adopted after seeing it work.
This runner uses `MASTER_SEED + 1000..` so no development seed is reused, and
every threshold was frozen in `EncoderIndependentPhase3Protocol.md` before
the first run.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase3_confirmation
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
    ObjectiveConfig, TrainConfig, config_digest, derive_seed,
)
from .diagnostics import paired_log_ratio, provenance, write_json
from .evaluate import evaluate_arm, evaluation_pools, null_reference
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .train import train_arm

HERE = Path(__file__).resolve().parent

# Frozen (protocol section 4): development used MASTER_SEED + 0,1,2.
CONFIRMATION_SEED_OFFSET = 1000

# Frozen thresholds (protocol section 5).
MATERIAL = 0.50
DIMENSION_FLOOR = 0.60
SKYLINE_TOLERANCE = 1.25
BASELINE = "C0"
CANDIDATE = "C1"


def phase3_arms() -> list[ArmConfig]:
    """C0-C2, frozen.  Raw pixel geometry only; the geometry thread is closed."""
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace")
    fixed = MixtureConfig(adaptive=False)

    def objective(variance_match: bool) -> ObjectiveConfig:
        return ObjectiveConfig(lambda_anchor=0.0, lambda_geometry=1.0,
                               teacher_variance_match=variance_match)

    return [
        ArmConfig("C0", False, geometry,
                  FieldConfig(direction_mode="paper"), fixed,
                  objective(False),
                  note="paper Algorithm-2 field, no variance match "
                       "(the Phase-2 baseline)"),
        ArmConfig("C1", False, geometry,
                  FieldConfig(direction_mode="paper"), fixed,
                  objective(True),
                  note="paper Algorithm-2 field + R11 teacher variance "
                       "match (the candidate)"),
        ArmConfig("C2", False, geometry,
                  FieldConfig(direction_mode="standard"), fixed,
                  objective(True),
                  note="SNIS field + R11; does R12 still matter?"),
    ]


def run_cell(resolution: int, steps: int, seed: int, root: str | None,
             arms) -> list[dict]:
    train_target = cifar.cifar_target(resolution, "train", root)
    eval_target = cifar.cifar_target(resolution, "eval", root)
    config = TrainConfig(steps=steps, batch=64, controller_batch=32,
                         audit_batch=32, eval_samples=512,
                         image_size=resolution)
    pools = evaluation_pools(eval_target, config, seed)
    null = null_reference(eval_target, pools, seed)

    # R13 needs a field to probe; raw geometry is the only one in this phase.
    rng = np.random.default_rng(derive_seed(seed, "r13-calibration"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace")
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train_target.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=geometry.target_ess_fraction)

    # R13: the same field without a generator, once per direction rule.
    free = {}
    for mode in sorted({a.field.direction_mode for a in arms}):
        free[mode] = O.free_particle_reference(
            train_target, config, seed, branch, kernel, mode, pools, null)
        print(f"      free-particle ({mode}) score="
              f"{free[mode]['free_particle_score']:7.3f}", flush=True)

    rows = []
    for arm in arms:
        outcome = train_arm(arm, train_target, config, seed)
        row = {"arm": arm.arm_id, "seed": seed, "resolution": resolution,
               "steps": steps, "note": arm.note}
        row.update(evaluate_arm(outcome, eval_target, config, pools, null,
                                seed))
        reference = free[arm.field.direction_mode]
        row["r13"] = dict(reference)
        score = row.get("geometry_score_v2")
        base = reference["free_particle_score"]
        row["r13"]["parametric_gap"] = (
            float(score) / base
            if score is not None and np.isfinite(score)
            and base and np.isfinite(base) and base > 0 else float("nan"))
        rows.append(row)
        print(f"      {arm.arm_id:3} score={row.get('geometry_score_v2', float('nan')):7.3f} "
              f"ed2={row.get('ed2', float('nan')):7.4f} "
              f"eff_dim={row.get('effective_dimension_ratio', float('nan')):5.3f} "
              f"gap={row['r13'].get('parametric_gap', float('nan')):6.2f} "
              f"{outcome.wall_seconds:5.1f}s", flush=True)

    skyline = O.train_skyline(train_target, config, seed, pools, null)
    rows.append({
        "arm": "SKY", "seed": seed, "resolution": resolution, "steps": steps,
        "note": "skyline (sliced Wasserstein); never a candidate",
        "geometry_score_v2": skyline.score["geometry_score"],
        "geometry_ratios_v2": skyline.score["geometry_ratios"],
        "ed2": skyline.metrics["ed2"],
        "coverage": skyline.coverage, "precision": skyline.precision,
        "effective_dimension_ratio": skyline.metrics.get(
            "effective_dimension_ratio"),
    })
    print(f"      SKY score={skyline.score['geometry_score']:7.3f} "
          f"ed2={skyline.metrics['ed2']:7.4f} "
          f"eff_dim="
          f"{skyline.metrics.get('effective_dimension_ratio', float('nan')):5.3f}",
          flush=True)
    return rows


def _scores(rows: list[dict], subset=None) -> dict:
    out: dict[str, dict] = {}
    for row in rows:
        if subset is not None and not subset(row):
            continue
        value = row.get("geometry_score_v2")
        if value is None or not np.isfinite(value):
            value = float("inf")
        key = (row["resolution"], row["steps"], row["seed"])
        out.setdefault(row["arm"], {})[key] = value
    return out


def _paired(rows: list[dict], candidate: str, baseline: str,
            subset=None) -> dict:
    scores = _scores(rows, subset)
    if candidate not in scores or baseline not in scores:
        return {"ratio": float("nan"), "pairs": 0}
    keys = sorted(set(scores[candidate]) & set(scores[baseline]))
    if not keys:
        return {"ratio": float("nan"), "pairs": 0}
    return paired_log_ratio([scores[candidate][k] for k in keys],
                            [scores[baseline][k] for k in keys])


def evaluate_gate(rows: list[dict]) -> dict:
    overall = _paired(rows, CANDIDATE, BASELINE)
    per_seed = {s: _paired(rows, CANDIDATE, BASELINE,
                           lambda r, s=s: r["seed"] == s).get("ratio")
                for s in sorted({r["seed"] for r in rows})}
    per_resolution = {
        res: _paired(rows, CANDIDATE, BASELINE,
                     lambda r, res=res: r["resolution"] == res).get("ratio")
        for res in sorted({r["resolution"] for r in rows})}
    per_budget = {
        steps: _paired(rows, CANDIDATE, BASELINE,
                       lambda r, steps=steps: r["steps"] == steps
                       ).get("ratio")
        for steps in sorted({r["steps"] for r in rows})}

    def median(arm: str, key: str) -> float:
        values = [r.get(key) for r in rows if r["arm"] == arm]
        values = [float(v) for v in values
                  if isinstance(v, (int, float)) and np.isfinite(v)]
        return float(np.median(values)) if values else float("nan")

    def median_gap(arm: str) -> float:
        values = [r.get("r13", {}).get("parametric_gap") for r in rows
                  if r["arm"] == arm]
        values = [float(v) for v in values
                  if isinstance(v, (int, float)) and np.isfinite(v)]
        return float(np.median(values)) if values else float("nan")

    candidate_dimension = median(CANDIDATE, "effective_dimension_ratio")
    baseline_dimension = median(BASELINE, "effective_dimension_ratio")
    candidate_gap, baseline_gap = median_gap(CANDIDATE), median_gap(BASELINE)
    skyline = _paired(rows, CANDIDATE, "SKY")

    conditions = {
        "C.1_material_improvement": {
            **overall, "threshold": MATERIAL,
            "pass": bool(np.isfinite(overall.get("ratio", np.nan))
                         and overall["ratio"] <= MATERIAL
                         and overall["high"] < 1.0)},
        "C.2_every_seed": {
            "per_seed_ratio": per_seed,
            "pass": bool(per_seed and all(
                v is not None and np.isfinite(v) and v < 1.0
                for v in per_seed.values()))},
        "C.3_both_resolutions": {
            "per_resolution_ratio": per_resolution,
            "pass": bool(per_resolution and all(
                v is not None and np.isfinite(v) and v < 1.0
                for v in per_resolution.values()))},
        "C.4_every_budget": {
            "per_budget_ratio": per_budget,
            "pass": bool(per_budget and all(
                v is not None and np.isfinite(v) and v < 1.0
                for v in per_budget.values()))},
        "C.5_variance_collapse_repaired": {
            "candidate_effective_dimension_ratio": candidate_dimension,
            "baseline_effective_dimension_ratio": baseline_dimension,
            "floor": DIMENSION_FLOOR,
            "pass": bool(np.isfinite(candidate_dimension)
                         and candidate_dimension >= DIMENSION_FLOOR
                         and candidate_dimension > baseline_dimension)},
        "C.6_reaches_the_skyline": {
            **skyline, "tolerance": SKYLINE_TOLERANCE,
            "pass": bool(np.isfinite(skyline.get("ratio", np.nan))
                         and skyline["ratio"] <= SKYLINE_TOLERANCE)},
        "C.7_parametric_gap_narrows": {
            "candidate_gap": candidate_gap, "baseline_gap": baseline_gap,
            "pass": bool(np.isfinite(candidate_gap)
                         and np.isfinite(baseline_gap)
                         and candidate_gap < baseline_gap)},
    }
    return {
        "conditions": conditions,
        "gate_pass": all(bool(v["pass"]) for v in conditions.values()),
        "arm_median_score": {
            arm: float(np.median([v for v in cells.values()
                                  if np.isfinite(v)]))
            if any(np.isfinite(v) for v in cells.values()) else None
            for arm, cells in _scores(rows).items()},
        "c2_versus_c1": _paired(rows, "C2", CANDIDATE),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolutions", type=str, default="16,32")
    parser.add_argument("--budgets", type=str, default="300,600,1200")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase3_confirmation.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if not cifar.available(args.root):
        raise SystemExit("CIFAR-10 is not present locally.")

    resolutions = [int(x) for x in args.resolutions.split(",")]
    budgets = [int(x) for x in args.budgets.split(",")]
    arms = phase3_arms()

    started = time.time()
    rows: list[dict] = []
    for resolution in resolutions:
        for steps in budgets:
            for index in range(args.seeds):
                seed = MASTER_SEED + CONFIRMATION_SEED_OFFSET + index
                print(f"  res={resolution} steps={steps} seed={seed}",
                      flush=True)
                rows.extend(run_cell(resolution, steps, seed, args.root,
                                     arms))
    gate = evaluate_gate(rows)

    payload = {
        "status": "phase3-confirmation-fresh-seeds",
        "scope": "CIFAR-10 at 16x16 and 24x24, raw pixel geometry, disjoint "
                 "train/eval splits, no pretrained encoder, no class labels; "
                 "confirmation seeds disjoint from development",
        "config": vars(args) | {"out": str(args.out)},
        "seed_offset": CONFIRMATION_SEED_OFFSET,
        "arm_digests": {a.arm_id: config_digest(a) for a in arms},
        "provenance": provenance(),
        "elapsed_seconds": time.time() - started,
        "gate": gate,
        "rows": rows,
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 3 CONFIRMATION ===")
    for arm in sorted(gate["arm_median_score"]):
        score = gate["arm_median_score"][arm]
        shown = "n/a" if score is None else f"{score:7.3f}"
        label = "skyline (not a candidate)" if arm == "SKY" else ""
        print(f"  {arm:4} median score {shown}   {label}")
    print()
    for name, condition in gate["conditions"].items():
        status = "PASS" if condition["pass"] else "FAIL"
        detail = ""
        if np.isfinite(condition.get("ratio", np.nan)):
            detail = f" ratio={condition['ratio']:.4f}"
            if "high" in condition:
                detail += (f" [{condition['low']:.4f},"
                           f"{condition['high']:.4f}]"
                           f" wins={condition['wins']}/{condition['pairs']}")
        print(f"  [{status}] {name}{detail}")
    print(f"\n  Phase-3 gate: {'PASS' if gate['gate_pass'] else 'FAIL'}")
    print(f"  wrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
