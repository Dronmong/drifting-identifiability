"""F1 §2.1 null calibration — the pre-flight that can veto Stage F1.

The 0.05 recall gate is calibrated in *scale* (anchors 0.000 x6, 0.224, 0.496,
0.737) but its behaviour near zero has never been measured, which is why
Phase 30's flickers (0.009, 0.044, 0.001) cannot currently be read as signal or
as noise.  This measurement decides both.

**`RECALL_GATE` is held fixed at 0.05.**  What is estimated is the *null
probability of exceeding it*, bounded from above:

  1. per null state, count `E = #{r : recall_r > 0.05}` over 200 independent runs;
  2. one-sided exact Clopper-Pearson 95% upper bound `p_null_upper`;
  3. take the maximum over `identical_images` and `gaussian_mm`;
  4. GO only if `p_null_upper < 0.025`.

An earlier draft decided on a point 99th percentile with a bootstrap CI reported
beside it; at 200 replicates that quantile rests on two or three observations, so
the rule could proceed while its own interval did not support the decision.  A
still earlier draft used `max(0.05, 5*SD)`, which presumes a zero-mean Gaussian
null for a bounded, discrete, manifold-dependent statistic -- and rested on an
impossible construction (20 x 512 = 10 240 disjoint draws from an 8 192 pool).
Both are withdrawn.

`identical_images` uses a *different* random image per replicate, so the control
retains cross-replicate variability, and its 512 Inception features are computed
once and tiled -- exactly equivalent, and 512x cheaper.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.f1_calibration
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from .appearance import precision_recall
from .config import MASTER_SEED, derive_seed
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnostics import provenance, write_json
from .f1 import (
    EVAL_REFERENCE,
    NULL_TOLERANCE,
    PARTICLES,
    RECALL_GATE,
    clopper_pearson_upper,
)
from .fid import inception_features

HERE = Path(__file__).resolve().parent

# Frozen: pilot grid, band, and replicate counts (§3, §2.1).
#
# **Grid widened after a 3-replicate smoke, recorded for transparency.**  The
# protocol's grid was {0.5 ... 0.9}, but the smoke showed lambda = 0.5 already
# yields median recall 0.0000 -- the real->noise transition lies *below* the
# grid's floor, so no grid point could ever have landed in the [0.02, 0.10]
# band.  That is a mis-specified construction, not an outcome ranking, and the
# smoke's declared purpose is to validate feasibility.  The selection RULE
# (smallest qualifying lambda, ties to smaller, report unavailable rather than
# re-tune) is unchanged, and `blend_near_gate` never enters the GO/NO-GO
# decision -- that rests only on `identical_images` and `gaussian_mm`.
PILOT_LAMBDAS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
PILOT_BAND = (0.02, 0.10)
PILOT_REPLICATES = 20
NULL_REPLICATES = 200
NULL_STATES = ("identical_images", "gaussian_mm")


def eval_references(resolution: int, root: str | None) -> tuple:
    """F1's evaluation reference and a DISJOINT pilot-only reference.

    The pilot that selects `blend_near_gate`'s lambda must not touch the
    reference the null decision is made against, or the sensitivity control
    would be tuned on the same data it is later measured with.
    """
    pool = cifar.cifar_pool(resolution, "eval", root)
    order = np.random.default_rng(
        derive_seed(MASTER_SEED + 51000, "eval-ref")).permutation(len(pool))
    primary = np.sort(order[:EVAL_REFERENCE])
    pilot = np.sort(order[EVAL_REFERENCE:2 * EVAL_REFERENCE])
    if np.intersect1d(primary, pilot).size:
        raise AssertionError("pilot and primary references overlap")
    return pool[torch.as_tensor(primary)], pool[torch.as_tensor(pilot)]


def draw(state: str, replicate: int, resolution: int, root: str | None,
         reference: torch.Tensor, lam: float | None = None) -> torch.Tensor:
    """One 512-sample realization of a calibration reference state."""
    seed = derive_seed(MASTER_SEED + 51000, "calib", state, replicate,
                       lam if lam is not None else -1)
    rng = np.random.default_rng(seed)
    train = cifar.cifar_pool(resolution, "train", root)
    if state == "identical_images":
        index = int(rng.integers(0, len(train)))
        return train[index:index + 1].repeat(PARTICLES, 1, 1, 1)
    if state == "gaussian_mm":
        # Per-coordinate mean/sd from the evaluation reference, no clipping --
        # matching every prior use in Phases 15-30 so the 0.867/0.000 anchors
        # stay comparable.
        return gaussian_moment_match(reference, PARTICLES, rng)
    if state == "blend_near_gate":
        real = train[torch.as_tensor(
            rng.choice(len(train), size=PARTICLES, replace=False))]
        generator = torch.Generator().manual_seed(seed % (2 ** 31))
        noise = (torch.randn(real.shape, generator=generator) * 0.5).clamp(
            -1.0, 1.0)
        return (1.0 - lam) * real + lam * noise
    raise ValueError(f"unknown calibration state {state!r}")


def recall_of(images: torch.Tensor, features_reference: np.ndarray, device,
              tile: bool = False) -> float:
    """Recall against a cached reference. `tile` exploits identical inputs."""
    if tile:
        one = inception_features(images[:1].cpu(), device).double().numpy()
        features = np.repeat(one, len(images), axis=0)
    else:
        features = inception_features(images.cpu(), device).double().numpy()
    return precision_recall(features, features_reference)["recall"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=NULL_REPLICATES)
    parser.add_argument("--pilot-replicates", type=int,
                        default=PILOT_REPLICATES)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "f1_calibration.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    started = time.time()
    primary, pilot = eval_references(args.resolution, args.data_root)
    features_primary = inception_features(
        primary, device).double().numpy()
    features_pilot = inception_features(pilot, device).double().numpy()
    print(f"    references: primary {tuple(primary.shape)} "
          f"pilot {tuple(pilot.shape)} (disjoint)\n", flush=True)

    print("=== blend_near_gate pilot: select lambda ON THE PILOT REFERENCE ===",
          flush=True)
    pilot_rows = []
    for lam in PILOT_LAMBDAS:
        values = [recall_of(draw("blend_near_gate", r, args.resolution,
                                 args.data_root, pilot, lam),
                            features_pilot, device)
                  for r in range(args.pilot_replicates)]
        median = float(np.median(values))
        qualifies = PILOT_BAND[0] <= median <= PILOT_BAND[1]
        pilot_rows.append({"lambda": lam, "median_recall": median,
                           "in_band": qualifies,
                           "values": [float(v) for v in values]})
        print(f"    lambda={lam:.1f} median recall={median:.4f} "
              f"{'IN BAND' if qualifies else ''}", flush=True)
    qualifying = [r["lambda"] for r in pilot_rows if r["in_band"]]
    chosen = min(qualifying) if qualifying else None
    print(f"    chosen lambda: {chosen}"
          f"{'  (control unavailable, NOT re-tuned)' if chosen is None else ''}\n",
          flush=True)

    states = list(NULL_STATES) + ([] if chosen is None
                                  else ["blend_near_gate"])
    print(f"=== null calibration: {args.replicates} independent 512-sample "
          f"sets per state ===", flush=True)
    rows = {}
    for state in states:
        lam = chosen if state == "blend_near_gate" else None
        values = []
        for replicate in range(args.replicates):
            images = draw(state, replicate, args.resolution, args.data_root,
                          primary, lam)
            values.append(recall_of(images, features_primary, device,
                                    tile=(state == "identical_images")))
            if replicate and replicate % 50 == 0:
                print(f"      {state:18} {replicate}/{args.replicates} "
                      f"running max={max(values):.4f}", flush=True)
        values = np.asarray(values, dtype=float)
        exceed = int((values > RECALL_GATE).sum())
        entry = {
            "state": state, "lambda": lam, "replicates": len(values),
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "max": float(values.max()),
            "q95": float(np.percentile(values, 95)),
            "q99": float(np.percentile(values, 99)),
            "exceedances_above_gate": exceed,
            "p_null_upper": clopper_pearson_upper(exceed, len(values)),
            "values": values.tolist(),
        }
        rows[state] = entry
        print(f"    {state:18} mean={entry['mean']:.5f} sd={entry['sd']:.5f} "
              f"max={entry['max']:.4f} q99={entry['q99']:.4f} "
              f"E={exceed}/{len(values)} "
              f"p_upper={entry['p_null_upper']:.5f}", flush=True)

    p_upper = max(rows[s]["p_null_upper"] for s in NULL_STATES)
    decision = ("GO" if p_upper < NULL_TOLERANCE else "NO-GO")
    false_pass = 3 * p_upper ** 2 * (1 - p_upper) + p_upper ** 3
    verdict = {
        "recall_gate": RECALL_GATE, "null_tolerance": NULL_TOLERANCE,
        "p_null_upper": p_upper,
        "p_null_upper_by_state": {s: rows[s]["p_null_upper"]
                                  for s in NULL_STATES},
        "two_of_three_false_pass_bound": float(false_pass),
        "chosen_blend_lambda": chosen,
        "pilot": pilot_rows,
        "decision": decision,
        "uncertainty_is_conditional_on_reference": True,
    }
    verdict["reading"] = (
        f"GO -- the null probability of exceeding recall {RECALL_GATE} is "
        f"bounded above by {p_upper:.5f} < {NULL_TOLERANCE}; the gate is usable "
        f"and the 2-of-3 rule has false-pass bound {false_pass:.5f}"
        if decision == "GO" else
        f"NO-GO -- the null can exceed recall {RECALL_GATE} with probability up "
        f"to {p_upper:.5f}, at or above the {NULL_TOLERANCE} tolerance. Increase "
        f"the evaluation sample count and repeat; do NOT move the threshold.")

    payload = {"status": "f1-null-calibration",
               "protocol": "numerics/EncoderIndependentF1Protocol.md",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "states": rows, "verdict": verdict}
    digest = write_json(args.out, payload)

    print("\n=== F1 §2.1 NULL CALIBRATION ===")
    print(f"{'state':20}{'mean':>9}{'sd':>9}{'max':>9}{'q99':>9}"
          f"{'E>gate':>8}{'p_upper':>10}")
    for state in states:
        e = rows[state]
        print(f"{state:20}{e['mean']:9.5f}{e['sd']:9.5f}{e['max']:9.4f}"
              f"{e['q99']:9.4f}{e['exceedances_above_gate']:8}"
              f"{e['p_null_upper']:10.5f}")
    print(f"\n    RECALL_GATE {RECALL_GATE} held fixed; "
          f"p_null_upper = {p_upper:.5f} against tolerance {NULL_TOLERANCE}")
    print(f"    2-of-3 false-pass bound {false_pass:.5f}")
    print(f"    blend_near_gate lambda: {chosen}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
