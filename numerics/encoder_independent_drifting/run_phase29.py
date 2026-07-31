"""Phase 29: is the deficit the assignment DIRECTION?

Phase 29's research pass (`EncoderIndependentPhase29Research.md`) unified Phase
13 and Phase 28 into one mechanism: with fresh latents the target is a random
variable, squared-error regression converges to its conditional mean, and
coverage collapses however sharp each individual target was.  Averaging was
never the operative property -- an exact Hungarian bijection with fresh latents
gave the worst spectral tail ever measured here (0.0006) despite perfect
non-collision, while nearest-neighbour collapsed to `distinct = 1`.

Every scheme tried in 28 phases assigns **cloud -> real**: for each generated
point, find its target.  That direction optimizes *precision* and permits mode
collapse.  Precision was never the deficit -- drifting sits at precision
0.52-0.65 with **recall 0.000**, and even a structureless Gaussian scores
precision 0.842.  **Recall is the entire deficit.**

The opposite direction has never been tried here.  For each *real* sample, draw
k candidates and pull the nearest toward it -- IMLE (Li & Malik 2018).  It
cannot drop a mode by construction, and the nearest generation to a fixed real
point moves slowly under parameter updates, so the correspondence is stable
without being frozen.

Arms, all with FRESH latents except where marked, so these are genuine
generative objectives and not memorization:

  `nearest_fresh`     cloud -> real nearest neighbour.  The collapse control;
                      Phase 13's B6 gave distinct = 1 and second moment 0.000.
  `hungarian_fresh`   cloud <-> real exact bijection.  Non-colliding and
                      non-averaging, and Phase 13 says it still collapses.
  `imle_k8`           real -> cloud, k = 8 candidates.  The untested direction.
  `hungarian_fixed`   frozen latents AND frozen real set, assignment relearned
                      each step.  The memorization ceiling with an *optimal*
                      pairing rather than Phase 28's random one.  Scored on both
                      fresh and training latents, so the amortization gap is
                      visible rather than assumed.

Anchors, already measured: memorization with a random fixed pairing reaches
KID 0.0611 / recall 0.224 with recognizable objects (Phase 28 Stage A); the
drifting objective reaches KID 0.15 / recall 0.000 (Phase 28 Stage B).

**None of these is drifting.**  This is a test of the mechanism above and a
measurement of what this harness can do at all.  No result here licenses any
claim about the paper's method.

Declared before running:

  IMLE reaches recall > 0.1 with fresh latents
      -> the mechanism is escapable and the deficit was the assignment
         DIRECTION; drifting's failure becomes a specific explained negative
  IMLE also collapses to recall ~ 0
      -> the obstruction lies outside the objective family, every result in 28
         phases is bounded by something else, and the memorization ceiling is
         the only meaningful number this harness has produced

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase29
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from . import cifar
from .config import MASTER_SEED, derive_seed
from .device import configure, resolve_device
from .diagnose_phase20 import save_grid
from .diagnose_phase26 import score
from .diagnostics import provenance, write_json
from .fid import inception_features
from .models import OneStepGenerator, sample_latent

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 47000
LATENT = 32
WIDTH = 64
STEPS = 6000
RECORD_EVERY = 1500
EVAL = 512
BATCH = 256           # cloud and real batch, equal so a bijection exists
IMLE_REAL = 64        # real samples per IMLE step, matching drifting's positives
IMLE_K = 8            # candidate generations per real sample
FIXED_BANK = 512

# Already measured, carried here so every row is read against them.
ANCHORS = {"memorization_random_pairing": {"kid": 0.06110, "recall": 0.224,
                                           "precision": 0.791, "alpha": 4.417},
           "drifting_cold": {"kid": 0.14959, "recall": 0.000,
                             "precision": 0.523, "alpha": 4.503}}


def costs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    flat_a, flat_b = a.flatten(1), b.flatten(1)
    return (flat_a.pow(2).sum(1, keepdim=True)
            + flat_b.pow(2).sum(1, keepdim=True).T
            - 2.0 * flat_a @ flat_b.T).clamp_min(0.0)


def step_loss(arm: str, model, train, rng, seed, step, device, bank) -> tuple:
    """One optimization step's loss, plus the distinct-target count."""
    if arm == "imle_k8":
        real = train.sample(IMLE_REAL, rng)
        pool_z = sample_latent(IMLE_REAL * IMLE_K, LATENT,
                               derive_seed(seed, "z", arm, step), device)
        pool = model(pool_z)
        with torch.no_grad():
            # For each REAL sample, its nearest candidate.  This is the
            # direction that cannot drop a mode.
            choice = costs(real, pool.detach()).argmin(dim=1)
        selected = pool[choice]
        loss = ((selected - real) ** 2).flatten(1).sum(1).mean()
        return loss, int(torch.unique(choice).numel()), len(real) * IMLE_K

    if arm == "hungarian_fixed":
        latents, targets = bank
        pick = torch.as_tensor(rng.choice(len(latents), size=BATCH,
                                          replace=False), device=device)
        output = model(latents[pick])
        real = targets[pick]
    else:
        z = sample_latent(BATCH, LATENT, derive_seed(seed, "z", arm, step),
                          device)
        output = model(z)
        real = train.sample(BATCH, rng)

    with torch.no_grad():
        cost = costs(output.detach(), real)
        if arm == "nearest_fresh":
            column = cost.argmin(dim=1)
        else:
            _, column = linear_sum_assignment(cost.cpu().numpy())
            column = torch.as_tensor(column, device=device)
    loss = ((output - real[column]) ** 2).flatten(1).sum(1).mean()
    return loss, int(torch.unique(column).numel()), BATCH


def train_arm(arm: str, train, seed, device, resolution, reference, real_eval,
              bank) -> dict:
    model = OneStepGenerator(LATENT, 3, resolution, WIDTH,
                             derive_seed(seed, "generator")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(derive_seed(seed, "p29", arm))
    probe = sample_latent(EVAL, LATENT, derive_seed(seed, "probe"), device)
    history, distinct, forwards = [], [], 0
    for step in range(STEPS + 1):
        if step % RECORD_EVERY == 0:
            with torch.no_grad():
                emitted = model(probe).detach()
            entry = {"step": step,
                     "median_distinct": float(np.median(distinct))
                                        if distinct else float("nan"),
                     **score(emitted, reference, real_eval, device)}
            history.append(entry)
            print(f"      {arm:17} step {step:5} KID={entry['kid']:+.5f} "
                  f"P={entry['precision']:.3f} R={entry['recall']:.3f} "
                  f"alpha={entry['alpha']:.3f} "
                  f"distinct={entry['median_distinct']:.0f}", flush=True)
            distinct = []
        if step == STEPS:
            break
        for group in optimizer.param_groups:
            group["lr"] = 2e-3 * 0.5 * (1.0 + np.cos(np.pi * step / STEPS))
        loss, unique, used = step_loss(arm, model, train, rng, seed, step,
                                       device, bank)
        distinct.append(unique)
        forwards += used
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        fresh = model(probe).detach()
    out = {"arm": arm, "history": history, "generator_forwards": forwards,
           "fresh": history[-1]}
    if arm == "hungarian_fixed":
        latents, _ = bank
        with torch.no_grad():
            trained = model(latents[:EVAL]).detach()
        out["on_training_latents"] = score(trained, reference, real_eval,
                                           device)
        out["amortization_gap"] = (history[-1]["kid"]
                                   - out["on_training_latents"]["kid"])
    out["sample"] = fresh
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--arms", type=str,
                        default="nearest_fresh,hungarian_fresh,imle_k8,"
                                "hungarian_fixed")
    parser.add_argument("--out", type=Path, default=HERE / "phase29.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    seed = MASTER_SEED + SEED_OFFSET

    started = time.time()
    train = cifar.cifar_target(args.resolution, "train", args.data_root)
    train.device = device
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(seed, "p29"))
    real_eval = evaluation.sample(args.samples, rng)
    reference = inception_features(real_eval, device).double().numpy()
    print(f"    real reference: precision/recall anchors from Phase 28 -- "
          f"memorization KID {ANCHORS['memorization_random_pairing']['kid']:+.5f}"
          f" R {ANCHORS['memorization_random_pairing']['recall']:.3f}; "
          f"drifting KID {ANCHORS['drifting_cold']['kid']:+.5f} R 0.000\n",
          flush=True)

    bank_latents = sample_latent(FIXED_BANK, LATENT,
                                 derive_seed(seed, "bank"), device)
    bank_targets = train.sample(FIXED_BANK, np.random.default_rng(
        derive_seed(seed, "banktargets")))
    bank = (bank_latents, bank_targets)

    rows = []
    for arm in args.arms.split(","):
        print(f"=== {arm} ===", flush=True)
        out = train_arm(arm, train, seed, device, args.resolution, reference,
                        real_eval, bank)
        save_grid(out.pop("sample")[:64].cpu(),
                  HERE / f"phase29_samples_{arm}.png")
        rows.append(out)

    by_arm = {r["arm"]: r for r in rows}
    imle = by_arm.get("imle_k8", {}).get("fresh", {})
    verdict = {
        "anchors": ANCHORS,
        "final": {r["arm"]: r["fresh"] for r in rows},
        "distinct": {r["arm"]: r["fresh"]["median_distinct"] for r in rows},
        "generator_forwards": {r["arm"]: r["generator_forwards"] for r in rows},
        "imle_recall": imle.get("recall"),
        "imle_beats_collapse": bool(imle.get("recall", 0.0) > 0.1),
        "imle_beats_drifting_kid": bool(
            imle.get("kid", 1.0) < ANCHORS["drifting_cold"]["kid"]),
        "best_recall_arm": max(rows, key=lambda r: r["fresh"]["recall"])["arm"],
    }
    if "hungarian_fixed" in by_arm:
        verdict["hungarian_fixed_amortization_gap"] = by_arm[
            "hungarian_fixed"].get("amortization_gap")
    verdict["reading"] = (
        "the real->cloud direction escapes the collapse -- the deficit was the "
        "assignment DIRECTION, and this harness can do generative modelling"
        if verdict["imle_beats_collapse"] else
        "every fresh-latent objective collapses to recall ~ 0, so the "
        "obstruction lies outside the objective family and the memorization "
        "ceiling is the only meaningful number this harness has produced")

    payload = {"status": "phase29-assignment-direction",
               "research": "numerics/EncoderIndependentPhase29Research.md",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "verdict": verdict, "rows": rows}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 29: DOES THE ASSIGNMENT DIRECTION MATTER? ===")
    print(f"{'arm':18}{'KID':>10}{'prec':>8}{'recall':>8}{'alpha':>8}"
          f"{'2nd':>7}{'distinct':>10}{'fwd/step':>10}")
    for row in rows:
        e = row["fresh"]
        print(f"{row['arm']:18}{e['kid']:+10.5f}{e['precision']:8.3f}"
              f"{e['recall']:8.3f}{e['alpha']:8.3f}{e['second_moment']:7.3f}"
              f"{e['median_distinct']:10.0f}"
              f"{row['generator_forwards'] // STEPS:10}")
    for name, a in ANCHORS.items():
        print(f"{'(' + name[:16] + ')':18}{a['kid']:+10.5f}{a['precision']:8.3f}"
              f"{a['recall']:8.3f}{a['alpha']:8.3f}")
    if "hungarian_fixed" in by_arm:
        t = by_arm["hungarian_fixed"]["on_training_latents"]
        print(f"{'hung_fixed(train)':18}{t['kid']:+10.5f}{t['precision']:8.3f}"
              f"{t['recall']:8.3f}{t['alpha']:8.3f}")
        print(f"\n    hungarian_fixed amortization gap (fresh - train KID) = "
              f"{by_arm['hungarian_fixed']['amortization_gap']:+.5f}")
    print(f"\n    best recall: {verdict['best_recall_arm']}"
          f"   IMLE recall = {verdict['imle_recall']}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
