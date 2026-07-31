"""Phase-2B screen on CIFAR-16 (protocol section 4-6).

Four arms and a skyline, on real images, with no pretrained encoder in any
training objective and no class label read by anything.

Splits are disjoint by construction: arms, calibration and the skyline draw
from the CIFAR `train` split; every evaluation, null and support-calibration
pool draws from `eval`.  With a finite image pool this matters -- sampling
both from the same pool would let an arm memorize images it is later scored
against.

Exit gate (frozen before the run):

P2.1  best of B1/B2 vs B0: paired v2 ratio <= 0.90 with bootstrap upper
      bound < 1;
P2.2  that arm wins a majority of score components individually;
P2.3  ratio < 1 on every seed;
P2.4  B3 vs B1 <= 1.25 (the anchor is not destructive);
P2.5  unnormalized geometry loss falls >= 25% in every arm;
P2.6  winner's score <= 2.5 x the skyline's.

    uv run --python 3.12 --with torch==2.7.1 --with torchvision \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase2_screen
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import metrics as M
from . import oracle as O
from .config import MASTER_SEED, TrainConfig, config_digest
from .diagnostics import paired_log_ratio, provenance, write_json
from .evaluate import evaluate_arm, evaluation_pools, null_reference
from .train import phase2_arms, train_arm

HERE = Path(__file__).resolve().parent

# Frozen thresholds (protocol section 6).
MATERIAL = 0.90
ANCHOR_TOLERANCE = 1.25
LOSS_DESCENT = 0.25
SKYLINE_TOLERANCE = 2.5
CANDIDATES = ("B1", "B2")
BASELINE = "B0"


def run_cell(train_target, eval_target, seed: int, config: TrainConfig,
             arms) -> list[dict]:
    pools = evaluation_pools(eval_target, config, seed)
    null = null_reference(eval_target, pools, seed)
    rows = []
    for arm in arms:
        outcome = train_arm(arm, train_target, config, seed)
        row = {"arm": arm.arm_id, "seed": seed, "note": arm.note,
               "target": train_target.name}
        row.update(evaluate_arm(outcome, eval_target, config, pools, null,
                                seed))
        row["null_reference"] = null
        first, last = _loss_descent(outcome)
        row["geometry_loss_first"] = first
        row["geometry_loss_last"] = last
        row["geometry_loss_descent"] = (
            1.0 - last / first if first and np.isfinite(first) and first > 0
            else float("nan"))
        rows.append(row)
        print(f"    {arm.arm_id:3} score_v2="
              f"{row.get('geometry_score_v2', float('nan')):8.3f} "
              f"ed2={row.get('ed2', float('nan')):8.4f} "
              f"prec={row.get('precision', float('nan')):5.3f} "
              f"loss_desc={row['geometry_loss_descent']:6.3f} "
              f"{outcome.wall_seconds:6.1f}s", flush=True)

    skyline = O.train_skyline(train_target, config, seed, pools, null)
    rows.append({
        "arm": "SKY", "seed": seed, "target": train_target.name,
        "note": "skyline (sliced Wasserstein); never in a gate",
        "geometry_score_v2": skyline.score["geometry_score"],
        "geometry_ratios_v2": skyline.score["geometry_ratios"],
        "precision": skyline.precision, "coverage": skyline.coverage,
        "ed2": skyline.metrics["ed2"],
    })
    print(f"    SKY score_v2={skyline.score['geometry_score']:8.3f} "
          f"prec={skyline.precision:5.3f}", flush=True)
    return rows


def _loss_descent(outcome) -> tuple[float, float]:
    series = outcome.log.series.get("loss_geometry_unnormalized_total", [])
    finite = [v for v in series if np.isfinite(v)]
    if not finite:
        return float("nan"), float("nan")
    return float(finite[0]), float(finite[-1])


def _scores(rows: list[dict]) -> dict[str, dict[int, float]]:
    out: dict[str, dict[int, float]] = {}
    for row in rows:
        value = row.get("geometry_score_v2")
        if value is None or not np.isfinite(value):
            value = float("inf")
        out.setdefault(row["arm"], {})[row["seed"]] = value
    return out


def _paired(scores: dict, candidate: str, baseline: str) -> dict:
    if candidate not in scores or baseline not in scores:
        return {"ratio": float("nan"), "pairs": 0}
    keys = sorted(set(scores[candidate]) & set(scores[baseline]))
    return paired_log_ratio([scores[candidate][k] for k in keys],
                            [scores[baseline][k] for k in keys])


def evaluate_gate(rows: list[dict]) -> dict:
    scores = _scores(rows)
    comparisons = {c: _paired(scores, c, BASELINE) for c in CANDIDATES
                   if c in scores}
    if not comparisons:
        return {"conditions": {}, "gate_pass": False,
                "reason": "no candidate arm produced a score"}
    winner = min(comparisons, key=lambda c: comparisons[c].get(
        "ratio", float("inf")))
    best = comparisons[winner]

    # P2.2 per-component verdicts, averaged over seeds.
    component_wins: dict[str, list[bool]] = {}
    for seed in sorted({r["seed"] for r in rows}):
        cand = next((r for r in rows
                     if r["arm"] == winner and r["seed"] == seed), None)
        base = next((r for r in rows
                     if r["arm"] == BASELINE and r["seed"] == seed), None)
        if not cand or not base:
            continue
        verdict = M.component_verdicts(
            cand.get("geometry_ratios_v2", {}),
            base.get("geometry_ratios_v2", {}))
        for name, won in verdict["per_component"].items():
            component_wins.setdefault(name, []).append(won)
    majority = {name: float(np.mean(values))
                for name, values in component_wins.items()}
    components_won = sum(1 for v in majority.values() if v > 0.5)

    # P2.3 per-seed.
    per_seed = {}
    for seed in sorted({r["seed"] for r in rows}):
        subset = [r for r in rows if r["seed"] == seed]
        per_seed[seed] = _paired(_scores(subset), winner, BASELINE).get(
            "ratio", float("nan"))

    # P2.5 loss descent, every arm.
    descents = {}
    for row in rows:
        if row["arm"] == "SKY":
            continue
        value = row.get("geometry_loss_descent")
        if value is not None and np.isfinite(value):
            descents.setdefault(row["arm"], []).append(float(value))
    median_descent = {k: float(np.median(v)) for k, v in descents.items()}

    # P2.6 against the skyline.
    winner_score = float(np.median([r["geometry_score_v2"] for r in rows
                                    if r["arm"] == winner
                                    and np.isfinite(
                                        r.get("geometry_score_v2", np.nan))]))
    sky = [r["geometry_score_v2"] for r in rows if r["arm"] == "SKY"
           and np.isfinite(r.get("geometry_score_v2", np.nan))]
    sky_score = float(np.median(sky)) if sky else float("nan")

    anchor = _paired(scores, "B3", "B1")
    conditions = {
        "P2.1_beats_raw_pixels": {
            "winner": winner, **best, "threshold": MATERIAL,
            "pass": bool(np.isfinite(best.get("ratio", np.nan))
                         and best["ratio"] <= MATERIAL
                         and best["high"] < 1.0)},
        "P2.2_majority_of_components": {
            "component_win_rate": majority,
            "components_won": components_won,
            "components": len(majority),
            "pass": bool(majority and components_won > len(majority) / 2)},
        "P2.3_every_seed": {
            "per_seed_ratio": per_seed,
            "pass": bool(per_seed and all(
                np.isfinite(v) and v < 1.0 for v in per_seed.values()))},
        "P2.4_anchor_not_destructive": {
            **anchor, "tolerance": ANCHOR_TOLERANCE,
            "pass": bool(np.isfinite(anchor.get("ratio", np.nan))
                         and anchor["ratio"] <= ANCHOR_TOLERANCE)},
        "P2.5_objective_not_vacuous": {
            "median_descent": median_descent, "threshold": LOSS_DESCENT,
            "pass": bool(median_descent and all(
                v >= LOSS_DESCENT for v in median_descent.values()))},
        "P2.6_within_skyline_reach": {
            "winner_score": winner_score, "skyline_score": sky_score,
            "ratio": (winner_score / sky_score
                      if sky_score and np.isfinite(sky_score) else
                      float("nan")),
            "tolerance": SKYLINE_TOLERANCE,
            "pass": bool(np.isfinite(sky_score) and sky_score > 0
                         and winner_score <= SKYLINE_TOLERANCE * sky_score)},
    }
    return {
        "winner": winner,
        "comparisons": comparisons,
        "conditions": conditions,
        "gate_pass": all(bool(v["pass"]) for v in conditions.values()),
        "arm_median_score": {
            arm: float(np.median([v for v in cells.values()
                                  if np.isfinite(v)]))
            if any(np.isfinite(v) for v in cells.values()) else None
            for arm, cells in scores.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--controller-batch", type=int, default=32)
    parser.add_argument("--audit-batch", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--arms", type=str, default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase2_screen.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    if not cifar.available(args.root):
        raise SystemExit("CIFAR-10 is not present locally.")

    config = TrainConfig(
        steps=args.steps, batch=args.batch,
        controller_batch=args.controller_batch,
        audit_batch=args.audit_batch, eval_samples=args.eval_samples,
        image_size=args.resolution)
    arms = phase2_arms()
    if args.arms != "all":
        wanted = set(args.arms.split(","))
        arms = [a for a in arms if a.arm_id in wanted]
        missing = wanted - {a.arm_id for a in arms}
        if missing:
            raise ValueError(f"unknown arms: {sorted(missing)}")

    train_target = cifar.cifar_target(args.resolution, "train", args.root)
    eval_target = cifar.cifar_target(args.resolution, "eval", args.root)

    started = time.time()
    rows: list[dict] = []
    for seed in range(args.seeds):
        print(f"  seed {seed}", flush=True)
        rows.extend(run_cell(train_target, eval_target,
                             MASTER_SEED + seed, config, arms))
    gate = evaluate_gate(rows)

    payload = {
        "status": "phase2-screen-development-not-confirmation",
        "scope": f"CIFAR-10 at {args.resolution}x{args.resolution}, disjoint "
                 "train/eval splits, no pretrained encoder in any training "
                 "objective, no class label read by any objective, "
                 "controller or metric",
        "config": vars(args) | {"out": str(args.out)},
        "train_config_digest": config_digest(config),
        "arm_digests": {a.arm_id: config_digest(a) for a in arms},
        "provenance": provenance(),
        "elapsed_seconds": time.time() - started,
        "gate": gate,
        "rows": rows,
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 2 SCREEN (CIFAR-16) ===")
    for arm in sorted(gate["arm_median_score"]):
        score = gate["arm_median_score"][arm]
        label = "skyline (not an arm)" if arm == "SKY" else ""
        shown = "n/a" if score is None else f"{score:8.3f}"
        print(f"  {arm:4} median v2 score {shown}   {label}")
    print()
    for name, condition in gate.get("conditions", {}).items():
        status = "PASS" if condition["pass"] else "FAIL"
        detail = ""
        if np.isfinite(condition.get("ratio", np.nan)):
            detail = f" ratio={condition['ratio']:.4f}"
            if "high" in condition:
                detail += (f" [{condition['low']:.4f},"
                           f"{condition['high']:.4f}]"
                           f" wins={condition['wins']}/{condition['pairs']}")
        print(f"  [{status}] {name}{detail}")
    print(f"\n  winner: {gate.get('winner')}")
    print(f"  Phase-2 gate: {'PASS' if gate['gate_pass'] else 'FAIL'}")
    print(f"  wrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
