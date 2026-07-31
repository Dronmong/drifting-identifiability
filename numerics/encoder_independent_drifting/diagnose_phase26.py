"""Is the data distribution a fixed point of the drifting map at all?

**This probe exists because Phase 25's conclusion may be wrong.**  Phase 25
measured teacher recall at 0.000 across a 7.5x bandwidth range and I wrote that
this "bounds what the generator can learn".  That inference does not follow.
The generator does not sample from the teacher distribution -- it follows a
flow, and what bounds a flow is its **fixed point**, not the quality of one
step's target evaluated at a bad cloud.  A blurry local target is exactly what
a correct mean-shift flow looks like far from convergence.

So the question Phase 25 should have asked, and which 25 phases never asked, is:

    put the cloud AT the real data and see what the field does.

  If the drift nearly vanishes there, the data distribution is (approximately)
  stationary, the method's target is right, and every failure so far is one of
  optimization or dynamics -- which would REOPEN the pixel-space line that
  Phase 25 declared closed.

  If the drift is large and systematic there, the data distribution is not a
  fixed point, no optimizer can reach it, and the line is closed for a much
  stronger reason than Phase 25 gave.

Three things are measured.

`stationarity`  raw (unnormalized) drift magnitude with the cloud at real data,
    against the same quantity with the cloud at a trained generator's output.
    Unnormalized is essential: the RMS normalization used in training pins the
    magnitude to 1 by construction, so it cannot answer this question at all.

`R11 at the truth`  the correction gain applied when the cloud is real data.
    The second moment is already correct there, so an unbiased correction must
    return ~1.0.  Anything else means R11 pushes the cloud AWAY from the truth,
    which would make the program's headline reform an active bias.

`the attractor`  iterate the actual training map -- RMS-normalized drift, eta
    0.5, R11, fresh positives each step -- starting FROM real data, and watch
    where it goes.  If real data degrades into the same hyper-typical blur that
    Phase 25 measured, then that blur is the map's attractor, which explains
    every result in the program and closes the line definitively.

Declared before running:

  drift at truth << drift at generated, and the iteration stays near real
      -> the fixed point is right; the failure is optimization; line REOPENS
  drift at truth comparable to generated, or the iteration walks away
      -> the data distribution is not stationary; the target itself is wrong
  the iteration converges to the Phase-25 blur
      -> that blur is the attractor; strongest possible closure

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase26
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .appearance import precision_recall, spectrum_slope
from .config import MASTER_SEED, GeometryConfig, derive_seed
from .device import configure, resolve_device
from .diagnose_phase20 import save_grid
from .diagnose_phase25 import rectangular_ess, train_cloud
from .diagnostics import provenance, write_json
from .fid import inception_features, kid_from_features
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .objectives import corrected_teacher
from .run_phase16 import _tail

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 44000
POSITIVES = 64
CLOUD = 512
ITERATIONS = 40
RECORD_EVERY = 5
ETA = 0.5


def score(images: torch.Tensor, reference: np.ndarray, real: torch.Tensor,
          device) -> dict:
    features = inception_features(images.cpu(), device).double().numpy()
    pr = precision_recall(features, reference)
    return {"kid": kid_from_features(features, reference),
            "precision": pr["precision"], "recall": pr["recall"],
            "alpha": spectrum_slope(images.cpu())["alpha"],
            "tail": _tail(images.cpu()),
            "second_moment": float(
                images.flatten(1).var(0).mean().cpu()
                / real.flatten(1).var(0).mean().cpu())}


def stationarity(cloud: torch.Tensor, positives: torch.Tensor, branch,
                 kernel) -> dict:
    """Raw drift at a given cloud, plus what R11 does there.

    The RAW field is the only thing that can answer whether a configuration is
    stationary: `normalization="rms"` divides the field by its own magnitude, so
    the normalized drift has RMS 1 whether the cloud is perfect or hopeless.
    """
    with torch.no_grad():
        raw, _ = KG.field(cloud, positives, cloud, branch, kernel,
                          direction_mode="paper", normalization="none",
                          diagnostics=False)
        normalized, _ = KG.field(cloud, positives, cloud, branch, kernel,
                                 direction_mode="paper", normalization="rms",
                                 diagnostics=False)
        report: dict = {}
        corrected_teacher(cloud + ETA * normalized, positives, mode="scalar",
                          report=report)
        flat_raw = raw.flatten(1)
        centre = cloud.flatten(1).mean(dim=0, keepdim=True)
        radial = cloud.flatten(1) - centre
        radial = radial / radial.norm(dim=1, keepdim=True).clamp_min(1e-30)
        # Positive = the field pushes outward from the cloud centre.
        outward = float((flat_raw * radial).sum(dim=1).mean())
        return {
            "raw_drift_per_sample": float(flat_raw.norm(dim=1).mean()),
            "cloud_spread": float(cloud.flatten(1).std(dim=0).mean()),
            "raw_drift_relative": float(
                flat_raw.norm(dim=1).mean()
                / cloud.flatten(1).norm(dim=1).mean()),
            "radial_outward_component": outward,
            "r11_gain": float(report.get("correction_ratio_median", float("nan"))),
        }


def iterate(start: torch.Tensor, positives_source, rng, branch, kernel,
            reference, real, device, label: str) -> list[dict]:
    """Apply the TRAINING map repeatedly and record where it goes.

    Free particles: no generator, no optimizer, no Jacobian in the way, so this
    isolates the map's own attractor from anything the optimizer contributes.
    """
    state = start.clone()
    history = []
    # A map that barely moves anything would make "real data survives" vacuous,
    # so the per-step displacement is recorded rather than assumed nonzero.
    displacement = 0.0
    for step in range(ITERATIONS + 1):
        if step % RECORD_EVERY == 0:
            history.append({"step": step,
                            "step_displacement_relative": displacement,
                            "drift_from_start_relative": float(
                                (state - start).flatten(1).norm(dim=1).mean()
                                / start.flatten(1).norm(dim=1).mean()),
                            **score(state, reference, real, device)})
            entry = history[-1]
            print(f"      {label:14} step {step:3} KID={entry['kid']:+.5f} "
                  f"P={entry['precision']:.3f} R={entry['recall']:.3f} "
                  f"alpha={entry['alpha']:.3f} 2nd={entry['second_moment']:.3f} "
                  f"moved/step={displacement:.4f} "
                  f"from_start={entry['drift_from_start_relative']:.4f}",
                  flush=True)
        if step == ITERATIONS:
            break
        positives = positives_source.sample(POSITIVES, rng)
        with torch.no_grad():
            drift, _ = KG.field(state, positives, state, branch, kernel,
                                direction_mode="paper", normalization="rms",
                                diagnostics=False)
            updated = corrected_teacher(state + ETA * drift, positives,
                                        mode="scalar")
            displacement = float(
                (updated - state).flatten(1).norm(dim=1).mean()
                / state.flatten(1).norm(dim=1).mean().clamp_min(1e-30))
            state = updated
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase26_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    seed = MASTER_SEED + SEED_OFFSET

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(seed, "p26"))
    real = evaluation.sample(args.samples, rng)
    reference = inception_features(real, device).double().numpy()

    print(f"=== a trained cloud for comparison ({args.steps} steps) ===",
          flush=True)
    generated, train = train_cloud(args.steps, seed, device, args.resolution,
                                   args.data_root)
    branch = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace",
                       target_ess_fraction=0.05), 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=0.05)

    real_cloud = train.sample(CLOUD, rng)
    positives = train.sample(POSITIVES, rng)

    print("\n=== stationarity: what does the field do at each cloud? ===",
          flush=True)
    clouds = {"real_data": real_cloud, "generated": generated}
    stationary = {}
    for name, cloud in clouds.items():
        entry = stationarity(cloud, positives, branch, kernel)
        entry["realized_ess"], entry["dead_rows"] = rectangular_ess(
            kernel, branch, cloud, positives)
        stationary[name] = entry
        print(f"    {name:11} raw|V|={entry['raw_drift_per_sample']:8.3f} "
              f"rel={entry['raw_drift_relative']:.4f} "
              f"outward={entry['radial_outward_component']:+8.3f} "
              f"R11 gain={entry['r11_gain']:.4f} "
              f"ESS={entry['realized_ess']:.3f}", flush=True)

    drift_ratio = (stationary["real_data"]["raw_drift_per_sample"]
                   / max(stationary["generated"]["raw_drift_per_sample"], 1e-12))

    print("\n=== the attractor: iterate the TRAINING map from each start ===",
          flush=True)
    histories = {}
    for name, cloud in clouds.items():
        histories[name] = iterate(
            cloud, train, np.random.default_rng(derive_seed(seed, "it", name)),
            branch, kernel, reference, real, device, name)
        save_grid(cloud[:64].cpu(), HERE / f"phase26_start_{name}.png")

    real_start, real_end = histories["real_data"][0], histories["real_data"][-1]
    gen_end = histories["generated"][-1]
    # Does iterating from the truth land where Phase 25's teacher sat?
    converged_to_blur = bool(real_end["recall"] < 0.05
                             and real_end["precision"] < 0.75
                             and real_end["alpha"] > real_start["alpha"] + 0.3)
    verdict = {
        "stationarity": stationary,
        "drift_ratio_real_over_generated": float(drift_ratio),
        "real_start": real_start, "real_end": real_end,
        "generated_end": gen_end,
        "recall_lost_from_truth": real_start["recall"] - real_end["recall"],
        "kid_gained_from_truth": real_end["kid"] - real_start["kid"],
        "data_is_approximately_stationary": bool(drift_ratio < 0.5),
        "iteration_from_truth_degrades": bool(
            real_end["kid"] > real_start["kid"] + 0.02),
        "converged_to_the_phase25_blur": converged_to_blur,
        "r11_gain_at_truth": stationary["real_data"]["r11_gain"],
    }
    if verdict["converged_to_the_phase25_blur"]:
        verdict["reading"] = (
            "iterating the training map FROM REAL DATA destroys it and lands on "
            "the same hyper-typical blur -- that blur is the map's attractor, "
            "which explains every result in the program and closes the "
            "pixel-space line definitively")
    elif verdict["iteration_from_truth_degrades"]:
        verdict["reading"] = (
            "real data is not stable under the training map -- the target "
            "itself is wrong, so no optimizer could have reached it")
    elif verdict["data_is_approximately_stationary"]:
        verdict["reading"] = (
            "the data distribution is approximately stationary and survives "
            "iteration -- the fixed point is RIGHT, Phase 25's inference was "
            "wrong, and the failure is optimization; the line REOPENS")
    else:
        verdict["reading"] = (
            "mixed: the drift does not vanish at the truth but the iteration "
            "does not destroy it either; stationarity is partial")

    payload = {"status": "phase26-fixed-point-probe",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "verdict": verdict, "histories": histories}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 26: IS THE TRUTH A FIXED POINT? ===")
    print(f"{'cloud':12}{'raw |V|':>10}{'relative':>10}{'outward':>10}"
          f"{'R11 gain':>10}{'ESS':>8}")
    for name, entry in stationary.items():
        print(f"{name:12}{entry['raw_drift_per_sample']:10.3f}"
              f"{entry['raw_drift_relative']:10.4f}"
              f"{entry['radial_outward_component']:+10.3f}"
              f"{entry['r11_gain']:10.4f}{entry['realized_ess']:8.3f}")
    print(f"\n    raw drift at truth / at generated = {drift_ratio:.4f}"
          f"   (<<1 would mean the truth is stationary)")
    print(f"    R11 gain at the truth = "
          f"{verdict['r11_gain_at_truth']:.4f}   (1.0 = unbiased there)")
    print(f"\n{'iterating from':16}{'KID':>10}{'prec':>8}{'recall':>8}"
          f"{'alpha':>8}{'2nd':>7}")
    for name, history in histories.items():
        for entry in (history[0], history[-1]):
            print(f"{name + ' step' + str(entry['step']):16}"
                  f"{entry['kid']:+10.5f}{entry['precision']:8.3f}"
                  f"{entry['recall']:8.3f}{entry['alpha']:8.3f}"
                  f"{entry['second_moment']:7.3f}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
