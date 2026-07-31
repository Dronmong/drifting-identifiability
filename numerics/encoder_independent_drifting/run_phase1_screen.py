"""Phase-1 structured-image mechanism screen (plan section 9, Phase 1).

Arms A0-A8 are run on the nine structured targets across several seeds.  The
exit gate is stated before any arm is run and is evaluated verbatim:

G1.1  A4 materially beats A1 on structured geometry;
G1.2  A5 stays within 10% of A4 on the pre-registered normalized geometry
      score (metrics.normalized_geometry_score, frozen in that module);
G1.3  A5 passes collision cases on which A4 fails;
G1.4  the anchor contributes at least the frozen minimum gradient share for a
      meaningful portion of training
      (diagnostics.ANCHOR_SHARE_THRESHOLD / ANCHOR_PRESENCE_FRACTION);
G1.5  gains hold over multiple seeds and target families.

"Materially" is fixed as a >= 10% reduction in the geometric-mean paired
geometry-score ratio with a bootstrap upper bound below 1, matching the
repository's existing paired-ratio convention.

A8 is a LOCALLY TRAINED encoder stand-in, not the paper's pretrained
encoder (see reference_encoder.py).  It is reported as context and is
excluded from every gate condition.

Run:
    uv run --python 3.12 --with torch==2.7.1 --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase1_screen
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import datasets as D
from .config import MASTER_SEED, TrainConfig, config_digest, derive_seed
from .diagnostics import (
    ANCHOR_PRESENCE_FRACTION, ANCHOR_SHARE_THRESHOLD, paired_log_ratio,
    provenance, write_json,
)
from .evaluate import collision_report, evaluate_arm, evaluation_pools, \
    null_reference
from .reference_encoder import encoder_family, train_reference_encoder
from .train import phase1_arms, train_arm

HERE = Path(__file__).resolve().parent

# Frozen before any arm is run.
MATERIAL_IMPROVEMENT = 0.90        # G1.1: >= 10% better paired ratio
PARITY_TOLERANCE = 1.10            # G1.2: A5 within 10% of A4
GATE_ARMS = ("A1", "A4", "A5")
CONTEXT_ARMS = ("A8",)


# G1.3 needs the geometry blindness of A4 and the anchor coverage of A5;
# every other permutation test would duplicate the Phase-0 blindness table.
COLLISION_SOURCES = {"A4": ("geometry",), "A5": ("anchor",)}


def run_cell(target: D.ImageTarget, seed: int, config: TrainConfig,
             arms, collision_samples: int, permutations: int,
             collide_arms: tuple[str, ...]) -> list[dict]:
    pools = evaluation_pools(target, config, seed)
    null = null_reference(target, pools, seed)
    reference_family = None
    if any(a.arm_id in ("A8",) for a in arms):
        rng = np.random.default_rng(derive_seed(seed, "reference-encoder"))
        model = train_reference_encoder(
            target.sample, config.channels, 32,
            derive_seed(seed, "reference-encoder-init"),
            steps=200, batch=32, learning_rate=2e-3, rng=rng)
        reference_family = encoder_family(
            model, next(a for a in arms if a.arm_id == "A8").geometry)

    rows = []
    for arm in arms:
        outcome = train_arm(arm, target, config, seed,
                            reference_family=reference_family)
        row = {"target": target.name, "target_kind": target.kind,
               "seed": seed, "arm": arm.arm_id, "note": arm.note}
        row.update(evaluate_arm(outcome, target, config, pools, null, seed))
        row["null_reference"] = null
        if arm.arm_id in collide_arms:
            row["collisions"] = collision_report(
                outcome, collision_samples, permutations, seed,
                sources=COLLISION_SOURCES.get(arm.arm_id,
                                              ("anchor", "geometry")))
        rows.append(row)
        print(f"    {arm.arm_id:3} score="
              f"{row.get('geometry_score', float('nan')):8.3f} "
              f"ed2={row.get('ed2', float('nan')):8.4f} "
              f"cover={row.get('coverage', float('nan')):5.3f} "
              f"{outcome.wall_seconds:6.1f}s", flush=True)
    return rows


def _score_matrix(rows: list[dict]) -> dict[str, dict[tuple, float]]:
    out: dict[str, dict[tuple, float]] = defaultdict(dict)
    for row in rows:
        score = row.get("geometry_score")
        if score is None or not np.isfinite(score):
            score = float("inf")
        out[row["arm"]][(row["target"], row["seed"])] = score
    return out


def paired(scores: dict, candidate: str, baseline: str) -> dict:
    if candidate not in scores or baseline not in scores:
        return {"ratio": float("nan"), "pairs": 0}
    keys = sorted(set(scores[candidate]) & set(scores[baseline]))
    return paired_log_ratio([scores[candidate][k] for k in keys],
                            [scores[baseline][k] for k in keys])


def evaluate_gate(rows: list[dict]) -> dict:
    scores = _score_matrix(rows)
    g1_1 = paired(scores, "A4", "A1")
    g1_2 = paired(scores, "A5", "A4")

    # G1.3: collisions A5 sees and A4 misses.
    def missed(arm: str) -> set[str]:
        out: set[str] = set()
        for row in rows:
            if row["arm"] != arm or "collisions" not in row:
                continue
            for source, report in row["collisions"].items():
                for entry in report["rows"]:
                    if not entry["detected"]:
                        out.add(f"{source}::{entry['collision']}")
        return out

    def anchor_detects(arm: str) -> dict:
        detected, total = 0, 0
        for row in rows:
            report = row.get("collisions", {}).get("anchor")
            if row["arm"] != arm or report is None:
                continue
            detected += report["detected"]
            total += report["total"]
        return {"detected": detected, "total": total}

    a4_missed = missed("A4")
    a5_anchor = anchor_detects("A5")
    geometry_only_failures = {m for m in a4_missed
                              if m.startswith("geometry_")}
    g1_3 = {
        "a4_missed_collisions": sorted(a4_missed),
        "a5_anchor_detected": a5_anchor,
        "a4_geometry_blind_cases": sorted(geometry_only_failures),
        "pass": bool(geometry_only_failures)
        and a5_anchor["total"] > 0
        and a5_anchor["detected"] == a5_anchor["total"],
    }

    # G1.4: anchor presence.
    shares = [row.get("anchor_share_median") for row in rows
              if row["arm"] == "A5"]
    present = [bool(row.get("anchor_present")) for row in rows
               if row["arm"] == "A5"]
    g1_4 = {
        "threshold": ANCHOR_SHARE_THRESHOLD,
        "required_fraction": ANCHOR_PRESENCE_FRACTION,
        "median_share": float(np.median([s for s in shares
                                         if s is not None]))
        if any(s is not None for s in shares) else None,
        "cells_present": int(sum(present)),
        "cells": len(present),
        "pass": bool(present) and sum(present) >= 0.5 * len(present),
    }

    # G1.5: robustness across seeds and target families.
    per_target: dict[str, float] = {}
    for target in sorted({row["target"] for row in rows}):
        subset = [row for row in rows if row["target"] == target]
        per_target[target] = paired(_score_matrix(subset), "A4", "A1").get(
            "ratio", float("nan"))
    seeds = sorted({row["seed"] for row in rows})
    per_seed = {}
    for seed in seeds:
        subset = [row for row in rows if row["seed"] == seed]
        per_seed[seed] = paired(_score_matrix(subset), "A4", "A1").get(
            "ratio", float("nan"))
    wins_target = sum(1 for v in per_target.values()
                      if np.isfinite(v) and v < 1.0)
    wins_seed = sum(1 for v in per_seed.values()
                    if np.isfinite(v) and v < 1.0)
    g1_5 = {
        "per_target_ratio": per_target, "per_seed_ratio": per_seed,
        "targets_won": wins_target, "targets": len(per_target),
        "seeds_won": wins_seed, "seeds": len(per_seed),
        "pass": bool(len(per_target) and len(per_seed)
                     and wins_target > len(per_target) / 2
                     and wins_seed == len(per_seed)),
    }

    conditions = {
        "G1.1_A4_beats_A1": {
            **g1_1, "threshold": MATERIAL_IMPROVEMENT,
            "pass": bool(np.isfinite(g1_1.get("ratio", float("nan")))
                         and g1_1["ratio"] <= MATERIAL_IMPROVEMENT
                         and g1_1["high"] < 1.0)},
        "G1.2_A5_within_10pc_of_A4": {
            **g1_2, "tolerance": PARITY_TOLERANCE,
            "pass": bool(np.isfinite(g1_2.get("ratio", float("nan")))
                         and g1_2["ratio"] <= PARITY_TOLERANCE)},
        "G1.3_A5_passes_A4_collisions": g1_3,
        "G1.4_anchor_present": g1_4,
        "G1.5_robust_across_seeds_and_targets": g1_5,
    }
    return {
        "conditions": conditions,
        "gate_pass": all(bool(v["pass"]) for v in conditions.values()),
        "arm_median_score": {
            arm: float(np.median([v for v in cells.values()
                                  if np.isfinite(v)]))
            if any(np.isfinite(v) for v in cells.values()) else None
            for arm, cells in _score_matrix(rows).items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--controller-batch", type=int, default=32)
    parser.add_argument("--audit-batch", type=int, default=32)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--collision-samples", type=int, default=192)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--targets", type=str, default="all")
    parser.add_argument("--arms", type=str, default="all")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase1_screen.json")
    args = parser.parse_args()
    # Recorded in the manifest: a replay must use the same thread count for
    # bit-identical reduction order.
    torch.set_num_threads(args.threads)

    config = TrainConfig(
        steps=args.steps, batch=args.batch,
        controller_batch=args.controller_batch,
        audit_batch=args.audit_batch, eval_samples=args.eval_samples)
    arms = phase1_arms()
    if args.arms != "all":
        wanted = set(args.arms.split(","))
        arms = [a for a in arms if a.arm_id in wanted]
        missing = wanted - {a.arm_id for a in arms}
        if missing:
            raise ValueError(f"unknown arms: {sorted(missing)}")
    targets = D.suite()
    if args.targets != "all":
        wanted = set(args.targets.split(","))
        targets = [t for t in targets if t.name in wanted]
        missing = wanted - {t.name for t in targets}
        if missing:
            raise ValueError(f"unknown targets: {sorted(missing)}")
    collide = tuple(a.arm_id for a in arms if a.arm_id in ("A4", "A5"))

    started = time.time()
    rows: list[dict] = []
    for index, target in enumerate(targets):
        for seed in range(args.seeds):
            print(f"  {target.name} seed {seed}", flush=True)
            # The collision suite carries its own p/q pairs, so the only
            # per-cell variation is the calibrated kernel and bank.  Running
            # it on the first target for every seed establishes blindness
            # without paying for eight redundant repeats.
            cell_collide = collide if index == 0 else ()
            rows.extend(run_cell(target, MASTER_SEED + seed, config, arms,
                                 args.collision_samples, args.permutations,
                                 cell_collide))
    gate = evaluate_gate(rows)

    payload = {
        "status": "phase1-mechanism-screen-development-not-confirmation",
        "scope": "synthetic 16x16 structured images; no pretrained encoder "
                 "in any training objective; A8 is a locally trained "
                 "stand-in, not the paper's pretrained encoder; the anchor "
                 "is a finite random-feature approximation, not an exact "
                 "characteristic family",
        "config": vars(args) | {"out": str(args.out)},
        "train_config_digest": config_digest(config),
        "arm_digests": {a.arm_id: config_digest(a) for a in arms},
        "provenance": provenance(),
        "elapsed_seconds": time.time() - started,
        "gate": gate,
        "rows": rows,
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 1 MECHANISM SCREEN ===")
    print(f"{'arm':4} {'median geometry score':>22}   note")
    for arm in sorted(gate["arm_median_score"]):
        score = gate["arm_median_score"][arm]
        note = "CONTEXT (local encoder stand-in)" if arm in CONTEXT_ARMS \
            else ("gate arm" if arm in GATE_ARMS else "")
        shown = "diverged/nan" if score is None else f"{score:22.4f}"
        print(f"{arm:4} {shown}   {note}")
    print()
    for name, condition in gate["conditions"].items():
        status = "PASS" if condition["pass"] else "FAIL"
        detail = ""
        if "ratio" in condition and np.isfinite(condition.get("ratio", np.nan)):
            detail = (f" ratio={condition['ratio']:.4f} "
                      f"[{condition['low']:.4f},{condition['high']:.4f}] "
                      f"wins={condition['wins']}/{condition['pairs']}")
        print(f"  [{status}] {name}{detail}")
    print(f"\n  Phase-1 gate: {'PASS' if gate['gate_pass'] else 'FAIL'}")
    print(f"  wrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
