"""F1 §7 veto calibration — thresholds from reference states, not intuition.

v1 set `0.25` normalized nearest-bank distance and `64 distinct` from intuition.
That is the failure mode this program has hit five times (Phase 19's sign rule,
Phase 23's precision expectations, Phase 27's basin veto, Phase 28's alpha < 4.0,
and these), so §7 now fixes a **mechanical selection rule** applied before any F1
outcome is exposed:

  1. orient every statistic so larger means healthier;
  2. accept >= 95% of the held-out-real reference AND reject >= 95% of the
     declared collapse references;
  3. among qualifying thresholds take the one maximizing margin to the healthy
     5th percentile;
  4. if no threshold satisfies (2), the statistic is **not a valid categorical
     veto** -- drop it from the gate, recording that it was dropped.

Collapse references are **synthetic known-answer controls** rather than
regenerated Phase-29 states.  §15.7 permits either; synthetic is preferable here
because the Phase-29 artifacts exist only as JSON summaries and image grids, the
regeneration was costed at ~15 min, and a synthetic control has an *exactly*
known answer instead of an approximately reproduced one.  All five statistics are
pixel-space, so no Inception pass is needed and this runs in seconds.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.f1_vetoes
"""

from __future__ import annotations

import argparse
import time
from itertools import chain
from pathlib import Path

import numpy as np
import torch

from .config import MASTER_SEED, derive_seed
from .device import configure, resolve_device
from .diagnostics import provenance, write_json
from .f1 import (
    PARTICLES,
    allocate,
    bank_statistics,
    duplicate_rate,
    effective_rank,
    nn_diversity,
    real_nn_scale,
    source_cloud,
    take,
)

HERE = Path(__file__).resolve().parent

REPLICATES = 20
HEALTHY_ACCEPT = 0.95      # threshold must accept >= this share of healthy
COLLAPSE_REJECT = 0.95     # ...and reject >= this share of collapse
# Oriented so larger = healthier.  `duplicate_rate` is inverted at measurement.
STATISTICS = ("nearest_bank_normalized", "distinct_bank", "effective_rank",
              "one_minus_duplicate_rate", "nn_diversity")
HEALTHY_STATES = ("held_out_real",)
COLLAPSE_STATES = ("exact_bank_copies", "single_image", "eight_modes")
SENSITIVITY_STATES = ("perturbed_bank", "random_generator")

# **Each statistic is calibrated against the failure mode it MEASURES.**
#
# The first version pooled all three collapse states and required every
# statistic to reject 95% of the pool.  That is incoherent, and running it
# showed why: `exact_bank_copies` is memorization but its effective rank is
# 9.18, legitimately similar to healthy real data's 8.4-9.3, because sampling
# 512 of 4096 with replacement is genuinely diverse.  No single statistic can
# reject both memorization and mode collapse, so a pooled reference guarantees
# every statistic is dropped.
#
# The assignment below is by construction -- what each statistic measures -- not
# by which assignment yields a passing threshold.  A distance-to-bank statistic
# detects memorization; diversity and rank statistics detect mode collapse.
VETO_TARGETS = {
    "nearest_bank_normalized": ("exact_bank_copies",),
    "distinct_bank": ("single_image", "eight_modes"),
    "effective_rank": ("single_image", "eight_modes"),
    "one_minus_duplicate_rate": ("exact_bank_copies", "single_image",
                                 "eight_modes"),
    "nn_diversity": ("single_image", "eight_modes"),
}


def build_state(state: str, replicate: int, unit: int, resolution: int,
                root: str | None, device, bank: torch.Tensor,
                pool_free: np.ndarray) -> torch.Tensor:
    """One realization of a reference state, all reproducible from the seed."""
    rng = np.random.default_rng(
        derive_seed(MASTER_SEED + 51000, "veto", state, replicate))
    if state == "held_out_real":
        # Disjoint from the bank by construction: drawn from the unallocated
        # remainder of the train pool.
        pick = rng.choice(len(pool_free), size=PARTICLES, replace=False)
        return take(resolution, "train", root, pool_free[pick]).to(device)
    if state == "exact_bank_copies":
        # WITH replacement, so the set contains genuine duplicates: exact
        # memorization of the bank.
        pick = rng.integers(0, len(bank), PARTICLES)
        return bank[torch.as_tensor(pick, dtype=torch.long)].clone()
    if state == "single_image":
        index = int(rng.integers(0, len(bank)))
        return bank[index:index + 1].repeat(PARTICLES, 1, 1, 1).clone()
    if state == "eight_modes":
        picks = rng.choice(len(bank), size=8, replace=False)
        modes = bank[torch.as_tensor(picks, dtype=torch.long)]
        return modes.repeat_interleave(PARTICLES // 8, dim=0).clone()
    if state == "perturbed_bank":
        pick = rng.choice(len(bank), size=PARTICLES, replace=False)
        base = bank[torch.as_tensor(pick, dtype=torch.long)].clone()
        generator = torch.Generator().manual_seed(replicate + 1)
        noise = torch.randn(base.shape, generator=generator).to(base.device)
        return (base + 0.05 * noise).clamp(-1.0, 1.0)
    if state == "random_generator":
        return source_cloud("random_generator", unit, resolution, root, device)
    raise ValueError(f"unknown reference state {state!r}")


def measure(images: torch.Tensor, bank: torch.Tensor, scale: float) -> dict:
    """The five §7 statistics, all oriented so larger = healthier."""
    stats = bank_statistics(images.cpu(), bank.cpu(), scale)
    return {
        "nearest_bank_normalized": stats["nearest_bank_normalized"],
        "distinct_bank": float(stats["distinct_bank"]),
        "effective_rank": effective_rank(images.cpu()),
        "one_minus_duplicate_rate": 1.0 - duplicate_rate(images.cpu(), scale),
        "nn_diversity": nn_diversity(images.cpu()),
    }


def select_threshold(healthy: np.ndarray, collapse: np.ndarray) -> dict:
    """The §7 mechanical rule. Returns the threshold or an explicit rejection."""
    healthy_p5 = float(np.percentile(healthy, 100 * (1 - HEALTHY_ACCEPT)))
    collapse_p95 = float(np.percentile(collapse, 100 * COLLAPSE_REJECT))
    separable = collapse_p95 < healthy_p5
    threshold = (0.5 * (collapse_p95 + healthy_p5)) if separable else None
    out = {
        "healthy_p5": healthy_p5, "collapse_p95": collapse_p95,
        "healthy_min": float(healthy.min()), "healthy_median": float(
            np.median(healthy)),
        "collapse_max": float(collapse.max()), "collapse_median": float(
            np.median(collapse)),
        "separable": bool(separable), "threshold": threshold,
    }
    if separable:
        out["healthy_accepted"] = float((healthy >= threshold).mean())
        out["collapse_rejected"] = float((collapse < threshold).mean())
        out["margin_to_healthy_p5"] = healthy_p5 - threshold
        out["valid_veto"] = bool(out["healthy_accepted"] >= HEALTHY_ACCEPT
                                 and out["collapse_rejected"] >= COLLAPSE_REJECT)
    else:
        out["valid_veto"] = False
        out["dropped_reason"] = (
            "no threshold accepts >=95% healthy while rejecting >=95% collapse; "
            "the distributions overlap, so this statistic is not a valid "
            "categorical veto")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=REPLICATES)
    parser.add_argument("--unit", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "f1_vetoes.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    started = time.time()
    allocation = allocate(args.unit, args.resolution, args.data_root)
    bank = take(args.resolution, "train", args.data_root,
                allocation.bank).to(device)
    scale = real_nn_scale(args.resolution, args.data_root,
                          allocation.real_data)
    pool_size = allocation.pool_size
    allocated = np.concatenate([allocation.bank, allocation.real_data,
                               allocation.ae_inputs, allocation.calibration,
                               np.array([allocation.identical])])
    pool_free = np.setdiff1d(np.arange(pool_size), allocated)
    print(f"    bank {len(bank)}  free pool {len(pool_free)}  "
          f"real NN scale {scale:.4f}\n", flush=True)

    every = list(HEALTHY_STATES) + list(COLLAPSE_STATES) + list(
        SENSITIVITY_STATES)
    samples: dict[str, dict[str, list]] = {}
    for state in every:
        rows = {name: [] for name in STATISTICS}
        for replicate in range(args.replicates):
            images = build_state(state, replicate, args.unit, args.resolution,
                                 args.data_root, device, bank, pool_free)
            for name, value in measure(images, bank, scale).items():
                rows[name].append(value)
        samples[state] = rows
        print(f"    {state:20} " + "  ".join(
            f"{n.split('_')[0]}={np.median(v):.3f}" for n, v in rows.items()),
            flush=True)

    def pooled(states, name) -> np.ndarray:
        return np.array(list(chain.from_iterable(
            samples[s][name] for s in states)))

    healthy = {name: pooled(HEALTHY_STATES, name) for name in STATISTICS}
    collapse = {name: pooled(VETO_TARGETS[name], name) for name in STATISTICS}

    decisions = {}
    for name in STATISTICS:
        decisions[name] = select_threshold(healthy[name], collapse[name])
        decisions[name]["targets"] = list(VETO_TARGETS[name])
        # Every collapse state this statistic is NOT calibrated against is
        # still reported, so a reader can see what it does and does not catch.
        decisions[name]["untargeted_collapse"] = {
            state: float(np.median(samples[state][name]))
            for state in COLLAPSE_STATES if state not in VETO_TARGETS[name]}
    for name, decision in decisions.items():
        if decision["separable"]:
            decision["sensitivity"] = {
                state: float(np.mean(
                    np.array(samples[state][name]) >= decision["threshold"]))
                for state in SENSITIVITY_STATES}

    valid = {n: d for n, d in decisions.items() if d["valid_veto"]}
    dropped = sorted(set(STATISTICS) - set(valid))
    verdict = {
        "healthy_states": list(HEALTHY_STATES),
        "collapse_states": list(COLLAPSE_STATES),
        "veto_targets": {k: list(v) for k, v in VETO_TARGETS.items()},
        "collapse_references_are_synthetic": True,
        "valid_vetoes": sorted(valid),
        "dropped_statistics": dropped,
        "any_valid_veto": bool(valid),
    }
    verdict["reading"] = (
        f"{len(valid)} of {len(STATISTICS)} statistics separate healthy from "
        f"collapse and become categorical vetoes; {len(dropped)} dropped as "
        f"non-separating"
        if valid else
        "NO statistic separates healthy from collapse -- per §7 the vetoes "
        "cannot be constructed and Stage F1 returns NO-GO")

    payload = {"status": "f1-veto-calibration",
               "protocol": "numerics/EncoderIndependentF1Protocol.md",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "reference_samples": {s: {n: [float(x) for x in v]
                                         for n, v in rows.items()}
                                     for s, rows in samples.items()},
               "thresholds": {n: d["threshold"] for n, d in valid.items()},
               "decisions": decisions, "verdict": verdict}
    digest = write_json(args.out, payload)

    print("\n=== F1 §7 VETO CALIBRATION ===")
    print(f"{'statistic':28}{'healthy p5':>12}{'collapse p95':>14}"
          f"{'threshold':>12}{'valid':>7}")
    for name, d in decisions.items():
        threshold = "--" if d["threshold"] is None else f"{d['threshold']:.4f}"
        print(f"{name:28}{d['healthy_p5']:12.4f}{d['collapse_p95']:14.4f}"
              f"{threshold:>12}{d['valid_veto']!s:>7}")
    print(f"\n    valid vetoes:  {sorted(valid)}")
    print(f"    dropped:       {dropped}")
    for name in sorted(valid):
        print(f"    {name}: sensitivity {decisions[name]['sensitivity']}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
