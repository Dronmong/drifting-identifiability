"""Can the generator hold a sharp state, and does a warm start survive drifting?

Phase 26: the data distribution is a stable attractor of the training map.
Phase 27: the map recovers from off-manifold displacement (blend lambda 0.6
returns, improving recall 0.283 -> 0.335) but not from box blur (lambda 0.2
degrades, 0.288 -> 0.271).

**Two methodological problems with the test Phase 27 proposed, fixed here.**

1. "Regress the generator onto real images" is vacuous without a *fixed*
   latent-to-image pairing.  With a fresh random target each step the target is
   independent of z, so squared error is minimized by outputting E[x] -- the mean
   image.  It would read "blurry" for a trivial reason and falsely exonerate the
   field.  Stage A therefore fixes the pairing and is honestly a *memorization
   capacity* test: can this architecture emit sharp images at all?

   Note what a sharp result would and would not license.  It shows a sharp
   solution EXISTS in parameter space, not that the drifting dynamics can reach
   one.  The conclusion available is "the blur is not a representational limit".

2. Phase 27 framed the unrecoverable direction as loss of high-frequency
   content, measured by spectrum alpha.  That contradicts Phase 24:

       real                 alpha 3.61   recall 0.767   KID ~0
       generated cloud      alpha 4.43   recall 0.000   KID 0.15
       AE d=512 recon       alpha 4.70   recall 0.496   KID 0.031

   The autoencoder is spectrally BLURRIER than the drifting output and far
   better on both recall and KID, with recognizable objects.  So high alpha is
   not the failure.  Re-reading Phase 27's matched pair -- blur lambda 0.2
   (alpha 4.76, recall 0.288) fails while blend lambda 0.6 (alpha 3.84, recall
   0.283) recovers -- alpha discriminates only *at matched recall*.  The failing
   region is low recall AND high alpha together, and the AE occupies a region
   (high alpha, high recall) that Phase 27 never tested.  Stage 0 tests it.

Stages:

  0  free-particle iteration from AE reconstructions -- the untested
     high-alpha/high-recall region, using Phase 27's exact method
  A  fixed-pairing MSE regression at two set sizes: can the generator be sharp?
  B  drifting FROM that warm start, against a cold-start control, tracking
     whether sharpness and recall decay

Declared before running:

  Stage 0 returns to the data attractor
      -> high alpha is survivable when recall is intact; an AE warm start is the
         route, and Phase 27's framing needs correcting
  Stage 0 collapses
      -> the AE region is also a trap and warm starting from reconstructions
         cannot work
  Stage A sharp, Stage B holds sharpness
      -> drifting preserves a good initialization; the program has a working
         recipe for the first time
  Stage A sharp, Stage B decays
      -> the dynamics actively destroy sharpness; the field is the problem and
         no initialization saves it
  Stage A blurry
      -> the blur is the generator's own; the field is exonerated and 27 phases
         tuned the wrong component

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase28
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .autoencoder import train_autoencoder
from .config import MASTER_SEED, GeometryConfig, derive_seed
from .device import configure, resolve_device
from .diagnose_phase20 import save_grid
from .diagnose_phase26 import ETA, POSITIVES, score
from .diagnose_phase27 import COLLAPSED_RECALL, DATA_RECALL, classify
from .diagnostics import provenance, write_json
from .fid import inception_features
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 46000
CLOUD = 512
ITERATIONS = 40
LATENT = 32
WIDTH = 64
MEMORY_SIZES = (512, 2048)
DRIFT_STEPS = 6000
DRIFT_RECORD = 1500


def free_particle_iterate(start, source, rng, branch, kernel, reference, real,
                          device, label) -> list[dict]:
    """Phase 27's exact method, so Stage 0 is directly comparable to its rows."""
    state = start.clone()
    history = []
    for step in range(ITERATIONS + 1):
        if step % 20 == 0:
            history.append({"step": step, **score(state, reference, real,
                                                  device)})
            e = history[-1]
            print(f"      {label:12} step {step:3} KID={e['kid']:+.5f} "
                  f"P={e['precision']:.3f} R={e['recall']:.3f} "
                  f"alpha={e['alpha']:.3f}", flush=True)
        if step == ITERATIONS:
            break
        positives = source.sample(POSITIVES, rng)
        with torch.no_grad():
            drift, _ = KG.field(state, positives, state, branch, kernel,
                                direction_mode="paper", normalization="rms",
                                diagnostics=False)
            state = corrected_teacher(state + ETA * drift, positives,
                                      mode="scalar")
    return history


def memorize(images: torch.Tensor, steps: int, seed: int, device,
             resolution: int, batch: int = 128) -> dict:
    """Fixed-pairing MSE regression: latent i is permanently bound to image i.

    Without the fixed pairing this reduces to predicting E[x] and the whole
    stage is uninformative -- see the module docstring.
    """
    model = OneStepGenerator(LATENT, 3, resolution, WIDTH,
                             derive_seed(seed, "generator")).to(device)
    latents = sample_latent(len(images), LATENT, derive_seed(seed, "pairing"),
                            device)
    targets = images.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(derive_seed(seed, "memorize", len(images)))
    history = []
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = 2e-3 * 0.5 * (1.0 + np.cos(np.pi * step
                                                     / max(steps, 1)))
        pick = torch.as_tensor(rng.choice(len(images), size=batch,
                                          replace=False), device=device)
        loss = ((model(latents[pick]) - targets[pick]) ** 2).flatten(1).mean(
            1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(steps // 4, 1) == 0 or step == steps - 1:
            history.append({"step": step, "mse": float(loss)})
    model.eval()
    with torch.no_grad():
        emitted = model(latents[:CLOUD]).detach()
    return {"model": model, "latents": latents, "sample": emitted,
            "history": history}


def drift_from(model, latents, train, branch, kernel, seed, device,
               resolution, reference, real, label) -> list[dict]:
    """Run the drifting objective starting from a given generator state."""
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    rng = np.random.default_rng(derive_seed(seed, "driftfrom", label))
    history = []
    for step in range(DRIFT_STEPS + 1):
        if step % DRIFT_RECORD == 0:
            with torch.no_grad():
                emitted = model(latents[:CLOUD]).detach()
            history.append({"step": step, **score(emitted, reference, real,
                                                  device)})
            e = history[-1]
            print(f"      {label:12} step {step:5} KID={e['kid']:+.5f} "
                  f"P={e['precision']:.3f} R={e['recall']:.3f} "
                  f"alpha={e['alpha']:.3f} 2nd={e['second_moment']:.3f}",
                  flush=True)
        if step == DRIFT_STEPS:
            break
        for group in optimizer.param_groups:
            group["lr"] = 2e-3 * 0.5 * (1.0 + np.cos(
                np.pi * step / max(DRIFT_STEPS, 1)))
        z = sample_latent(256, LATENT, derive_seed(seed, "z", label, step),
                          device)
        output = model(z)
        with torch.no_grad():
            positives = train.sample(POSITIVES, rng)
            field, _ = KG.field(output.detach(), positives, output.detach(),
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            teacher = corrected_teacher(output.detach() + ETA * field,
                                        positives, mode="scalar")
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memorize-steps", type=int, default=12000)
    parser.add_argument("--ae-steps", type=int, default=8000)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase28_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    seed = MASTER_SEED + SEED_OFFSET

    started = time.time()
    train = cifar.cifar_target(args.resolution, "train", args.data_root)
    train.device = device
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(seed, "p28"))
    real = evaluation.sample(args.samples, rng)
    reference = inception_features(real, device).double().numpy()
    branch = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace",
                       target_ess_fraction=0.05), 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=0.05)
    real_cloud = train.sample(CLOUD, rng)

    print("=== Stage 0: is the AE region (high alpha, HIGH recall) a trap? ===",
          flush=True)
    fit = train_autoencoder(train, args.ae_steps, seed, device,
                            latent_channels=32)
    with torch.no_grad():
        recon = fit["model"](real_cloud).detach()
    stage0 = free_particle_iterate(
        recon, train, np.random.default_rng(derive_seed(seed, "s0")), branch,
        kernel, reference, real, device, "ae_recon")
    save_grid(recon[:64].cpu(), HERE / "phase28_stage0_start.png")

    print("\n=== Stage A: can the generator emit sharp images at all? ===",
          flush=True)
    stage_a = {}
    for count in MEMORY_SIZES:
        images = train.sample(count, np.random.default_rng(
            derive_seed(seed, "mem", count)))
        out = memorize(images, args.memorize_steps, seed, device,
                       args.resolution)
        entry = {"count": count, "final_mse": out["history"][-1]["mse"],
                 "history": out["history"],
                 **score(out["sample"], reference, real, device)}
        stage_a[count] = entry | {"_model": out["model"],
                                  "_latents": out["latents"]}
        save_grid(out["sample"][:64].cpu(),
                  HERE / f"phase28_stageA_n{count}.png")
        print(f"    n={count:5} mse={entry['final_mse']:.5f} "
              f"KID={entry['kid']:+.5f} P={entry['precision']:.3f} "
              f"R={entry['recall']:.3f} alpha={entry['alpha']:.3f}", flush=True)

    warm = max(MEMORY_SIZES)
    sharp = bool(stage_a[warm]["alpha"] < 4.0
                 and stage_a[warm]["recall"] > DATA_RECALL)

    print("\n=== Stage B: drifting from the warm start, vs cold ===", flush=True)
    warm_history = drift_from(
        stage_a[warm]["_model"], stage_a[warm]["_latents"], train, branch,
        kernel, seed, device, args.resolution, reference, real, "warm")
    cold_model = OneStepGenerator(LATENT, 3, args.resolution, WIDTH,
                                  derive_seed(seed, "cold")).to(device)
    cold_history = drift_from(
        cold_model, stage_a[warm]["_latents"], train, branch, kernel, seed,
        device, args.resolution, reference, real, "cold")

    for entry in stage_a.values():
        entry.pop("_model", None)
        entry.pop("_latents", None)

    s0_first, s0_last = stage0[0], stage0[-1]
    warm_first, warm_last = warm_history[0], warm_history[-1]
    verdict = {
        "stage0_attractor": classify(s0_last),
        "stage0_recall_change": s0_last["recall"] - s0_first["recall"],
        "stage0_kid_change": s0_last["kid"] - s0_first["kid"],
        "ae_region_survivable": bool(classify(s0_last) == "data"),
        "generator_can_be_sharp": sharp,
        "stage_a": {str(k): {kk: vv for kk, vv in v.items()
                             if kk != "history"}
                    for k, v in stage_a.items()},
        "warm_start": warm_first, "warm_end": warm_last,
        "cold_end": cold_history[-1],
        "warm_holds_sharpness": bool(warm_last["recall"] > DATA_RECALL),
        "warm_beats_cold_kid": float(cold_history[-1]["kid"]
                                     - warm_last["kid"]),
        "warm_recall_lost": warm_first["recall"] - warm_last["recall"],
        "thresholds": {"data_recall": DATA_RECALL,
                       "collapsed_recall": COLLAPSED_RECALL},
    }
    if not sharp:
        verdict["reading"] = (
            "the generator cannot emit sharp images even with a fixed pairing "
            "and a sharp target -- the blur is the GENERATOR's, the field is "
            "exonerated, and 27 phases tuned the wrong component")
    elif verdict["warm_holds_sharpness"]:
        verdict["reading"] = (
            "the generator can be sharp AND drifting preserves it -- the "
            "program has a working recipe; a warm start is the missing piece")
    else:
        verdict["reading"] = (
            "the generator can be sharp but drifting DESTROYS it -- the "
            "dynamics actively remove coverage and no initialization saves it")

    payload = {"status": "phase28-capacity-and-warm-start-probe",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "config_digest": None,
               "elapsed_seconds": time.time() - started,
               "verdict": verdict, "stage0": stage0,
               "stage_a": {str(k): v for k, v in stage_a.items()},
               "warm_history": warm_history, "cold_history": cold_history}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 28 ===")
    print(f"{'stage':22}{'KID':>10}{'prec':>8}{'recall':>8}{'alpha':>8}")
    for label, entry in (("0 ae_recon start", s0_first),
                         ("0 ae_recon +40", s0_last)):
        print(f"{label:22}{entry['kid']:+10.5f}{entry['precision']:8.3f}"
              f"{entry['recall']:8.3f}{entry['alpha']:8.3f}")
    for count, entry in stage_a.items():
        print(f"{'A memorize n=' + str(count):22}{entry['kid']:+10.5f}"
              f"{entry['precision']:8.3f}{entry['recall']:8.3f}"
              f"{entry['alpha']:8.3f}")
    for label, entry in (("B warm start", warm_first),
                         ("B warm +6000", warm_last),
                         ("B cold +6000", cold_history[-1])):
        print(f"{label:22}{entry['kid']:+10.5f}{entry['precision']:8.3f}"
              f"{entry['recall']:8.3f}{entry['alpha']:8.3f}")
    print(f"\n    AE region attractor: {verdict['stage0_attractor']}"
          f"   (recall change {verdict['stage0_recall_change']:+.3f})")
    print(f"    generator can be sharp: {sharp}")
    print(f"    warm holds sharpness:   {verdict['warm_holds_sharpness']}"
          f"   (recall lost {verdict['warm_recall_lost']:+.3f})")
    print(f"    warm beats cold on KID by "
          f"{verdict['warm_beats_cold_kid']:+.5f}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
