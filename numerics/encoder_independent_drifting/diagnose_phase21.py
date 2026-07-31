"""Is the recipe averaging the data into a blob, and does a sharper kernel fix it?

The Phase-20 render shows the generator producing structureless low-frequency
colour fields -- no edges, no objects, every sample the same kind of blob.
That is the signature of **mode averaging**, and the mechanism predicts it
exactly.

`kernel_gradient._paper_side` builds the drift as a bi-softmax weighted
combination of the positives, so each training target is a weighted AVERAGE
of real images.  How many get averaged is set by the row effective sample
size.  This program calibrates ESS to 0.5 of the batch, so with 64 positives
**every target is an average of ~32 CIFAR images** -- which is a brown-green
blob.  Phase 7 pushed the other way (0.9, i.e. ~58 images) because it scored
better on ED2, and the metric audit later showed ED2 is saturated by matching
two moments: averaging preserves moments exactly while destroying structure.
Phase 19 then found 0.5 beats 0.9 on FID, which is this axis pointing the
same way.  **Nobody has ever looked at the sharp side.**

Three measurements per rung of an ESS ladder, none of which costs more than
the baseline -- bandwidth is free:

  S1  **Realized ESS at the cloud.**  Calibration is target-only by design,
      so the ESS the field actually sees is unmeasured.  If a cloud sitting
      off the data manifold is roughly equidistant from every positive, the
      weights go uniform, the target becomes the global mean, and the blob
      is self-reinforcing.  `kernel_health` already reports this.

  S2  **Teacher spread.**  Mean pairwise distance between teacher targets,
      relative to the same distance between real positives.  Near 0 means
      every particle is being sent to the same place -- mode averaging,
      measured directly rather than inferred.

  S3  **Look at it.**  A sample grid per rung.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase21
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .config import GeometryConfig, MASTER_SEED, TrainConfig, derive_seed
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnose_phase20 import save_grid
from .diagnostics import provenance, write_json
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher
from .run_phase16 import EVAL_SAMPLES, score
from .run_phase19 import CLOUD, POSITIVES, WIDTH, learning_rate

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 34000

# Phase 19's winner sits at 0.5; 0.9 is what Phases 7-18 used.  Everything
# below 0.5 is unexplored territory.  1/64 = 0.0156 is the sharpest a
# 64-positive batch can be -- one image per target.
ESS_LADDER = (0.9, 0.5, 0.2, 0.05, 0.02)


def spread(points: torch.Tensor, sample: int = 64) -> float:
    """Mean pairwise distance within a batch."""
    flat = points.detach().reshape(len(points), -1)[:sample]
    return float(torch.cdist(flat, flat).mean())


def run_rung(ess: float, steps: int, seed: int, device, resolution: int,
             root: str | None, probe_every: int) -> dict:
    train = cifar.cifar_target(resolution, "train", root)
    train.device = device
    rng = np.random.default_rng(derive_seed(seed, "p21"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=ess)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=ess)
    config = TrainConfig(steps=steps, batch=POSITIVES, image_size=resolution,
                         width=WIDTH)
    model = OneStepGenerator(config.latent_dim, 3, resolution, WIDTH,
                             derive_seed(seed, "generator")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    trace = []
    began = time.time()
    for step in range(steps):
        lr = learning_rate(config.learning_rate, "cosine", step, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        latent = sample_latent(CLOUD, config.latent_dim,
                               derive_seed(seed, "latent", step), device)
        output = model(latent)
        want = (step % probe_every == 0) or (step == steps - 1)
        with torch.no_grad():
            positives = train.sample(POSITIVES, rng)
            drift, stats = KG.field(output.detach(), positives,
                                    output.detach(), branch, kernel,
                                    direction_mode="paper",
                                    normalization="rms", diagnostics=want)
            teacher = corrected_teacher(output.detach() + 0.5 * drift,
                                        positives, mode="scalar")
            if want:
                reference = spread(positives)
                trace.append({
                    "step": step,
                    # S1: how many positives the field actually averages.
                    "realized_ess_fraction": stats["ess_fraction"],
                    "realized_ess_count": stats["ess_mean"],
                    "collapsed_row_fraction": stats["collapsed_row_fraction"],
                    # S2: are all the particles being sent to one place?
                    "teacher_spread": spread(teacher),
                    "positive_spread": reference,
                    "teacher_spread_ratio": spread(teacher) / max(reference,
                                                                  1e-12),
                    "output_spread_ratio": spread(output) / max(reference,
                                                               1e-12),
                })
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    probe = sample_latent(EVAL_SAMPLES, config.latent_dim,
                          derive_seed(seed, "probe"), device)
    with torch.no_grad():
        generated = model(probe)
    return {"trace": trace, "sample": generated,
            "wall_seconds": time.time() - began,
            "bandwidth_median": float(kernel.taus.median())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--probe-every", type=int, default=250)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase21_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    seed = MASTER_SEED + SEED_OFFSET

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(seed, "p21-eval"))
    real = evaluation.sample(EVAL_SAMPLES, rng)
    bar = score(gaussian_moment_match(real, EVAL_SAMPLES, rng), real,
                device)["fid"]
    print(f"    BAR {bar:.2f}\n", flush=True)

    rows = []
    for ess in ESS_LADDER:
        out = run_rung(ess, args.steps, seed, device, args.resolution,
                       args.data_root, args.probe_every)
        scored = score(out["sample"], real, device)
        last = out["trace"][-1]
        tag = f"{ess:.2f}".replace(".", "p")
        save_grid(out["sample"], HERE / f"phase21_samples_ess{tag}.png")
        rows.append({"target_ess": ess, "bandwidth_median":
                     out["bandwidth_median"],
                     "wall_seconds": out["wall_seconds"],
                     "final": last, "trace": out["trace"], **scored})
        print(f"    ess={ess:<5} FID={scored['fid']:7.2f} "
              f"realized_ess={last['realized_ess_fraction']:.3f} "
              f"({last['realized_ess_count']:5.1f} of {POSITIVES} images) "
              f"teacher_spread={last['teacher_spread_ratio']:.3f} "
              f"tail={scored['tail']:.4f} {out['wall_seconds']:5.0f}s",
              flush=True)

    best = min(rows, key=lambda r: r["fid"])
    verdict = {
        "bar": bar,
        "best_target_ess": best["target_ess"], "best_fid": best["fid"],
        "sharper_is_better": bool(best["target_ess"] < 0.5),
        "realized_vs_calibrated": {
            f"{r['target_ess']}": {
                "calibrated": r["target_ess"],
                "realized": r["final"]["realized_ess_fraction"]}
            for r in rows},
    }
    verdict["reading"] = (
        "sharper kernels beat the operating point every phase has used -- "
        "mode averaging was limiting the recipe"
        if verdict["sharper_is_better"] else
        "sharpening does not help; mode averaging is not the binding "
        "constraint and the blob has another cause")

    payload = {"status": "phase21-probe-feeds-no-gate",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "bar": bar, "rows": rows, "verdict": verdict}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 21: the mode-averaging ladder ===")
    print(f"{'target ESS':>11}{'FID':>9}{'realized ESS':>14}"
          f"{'images averaged':>17}{'teacher spread':>16}{'tail':>8}")
    for r in rows:
        f = r["final"]
        print(f"{r['target_ess']:11.2f}{r['fid']:9.2f}"
              f"{f['realized_ess_fraction']:14.3f}"
              f"{f['realized_ess_count']:17.1f}"
              f"{f['teacher_spread_ratio']:16.3f}{r['tail']:8.4f}")
    print(f"\n    BAR {bar:.2f}   (real data tail ~= 0.13)")
    print(f"    best target ESS: {verdict['best_target_ess']} "
          f"at FID {verdict['best_fid']:.2f}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
