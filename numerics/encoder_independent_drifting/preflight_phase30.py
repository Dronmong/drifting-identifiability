"""Pre-flight verification for the Phase 30 capacity experiment.

Nothing about Phase 30 gets scheduled until this passes.  The run is the largest
this program has attempted and there is a **known, recorded failure at exactly
this design point**: Phase 16's `S2_wide` (width 256, cloud 256) exhausted this
6 GiB card at 5844/6141 MiB and killed the run before it wrote its artifact --
which is why `phase16.json` does not exist and why cloud size and capacity have
never been measured on FID at all.  Repeating that overnight would waste a
night and produce nothing.

Also verified here: a logic correction to the hypothesis.  I wrote that "the
collapse is variance-driven and variance falls with scale", which conflates two
different mechanisms:

    batch size  reduces target variance  -> attacks the Phase 29 mechanism
    capacity    raises expressiveness    -> does NOT reduce target variance

Both are real levers -- Phase 28 shows capacity is exhausted somewhere between
memorizing 512 images (recall 0.224) and 2048 (recall 0.000) -- but a single
"scaled" arm cannot separate them.  Phase 30 is therefore a 2x2 factorial over
{width 64, 256} x {positives 64, 256}, and this pre-flight measures every cell.

Checks, all of which must pass:

  M  peak GPU memory per cell, against the card's real limit
  T  seconds per step per cell -> projected wall time for the declared budget
  N  numerical health at 200 steps: finite decreasing loss, no NaN, sane second
     moment, sane realized ESS, sane R11 gain
  S  metric sanity IN THIS SCRIPT: real-vs-real recall must be high and a
     moment-matched Gaussian's must be ~0, so a recall of 0.000 in the real run
     cannot be a broken measurement

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.preflight_phase30
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .appearance import precision_recall
from .config import MASTER_SEED, GeometryConfig, TrainConfig, derive_seed
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnose_phase25 import rectangular_ess
from .diagnostics import provenance, write_json
from .fid import inception_features
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 49000

# The declared Phase 30 design.  Widths and positive counts are the factorial;
# 512 is probed for memory only, to see whether headroom exists at all.
CELLS = ((64, 64), (256, 64), (64, 256), (256, 256))
MEMORY_ONLY = ((512, 256),)
CLOUD = 256
PROBE_STEPS = 200
BUDGET_STEPS = 30000
SEEDS = 2
ETA = 0.5
TARGET_ESS = 0.05     # legacy calibration, the operating point Phase 22 measured best

# Thresholds calibrated against MEASURED anchors, not intuition -- the mistake
# made four times already in this program (Phase 19's sign rule, Phase 23's
# precision expectations, Phase 27's basin veto, Phase 28's alpha < 4.0).
#
#   recall 0.000        every fresh-latent objective, six independent measurements
#   recall 0.224        memorization ceiling (Phase 28 Stage A)
#   recall 0.496        autoencoder d=512 reconstruction (Phase 24)
#   recall 0.72-0.77    real data
#
# So "nonzero recall" is declared at 0.05: an order of magnitude above the zero
# that six objectives produced, and far below the memorization anchor.
RECALL_SUCCESS = 0.05
# Card limit, from device_check.  Anything above this is a hard NO-GO.
CARD_MIB = 6141
MEMORY_HEADROOM = 0.80


def build(width: int, positives: int, seed: int, device, resolution: int,
          root: str | None):
    train = cifar.cifar_target(resolution, "train", root)
    train.device = device
    rng = np.random.default_rng(derive_seed(seed, "p30", width, positives))
    branch = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace",
                       target_ess_fraction=TARGET_ESS), 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=TARGET_ESS)
    config = TrainConfig(image_size=resolution, width=width)
    model = OneStepGenerator(config.latent_dim, 3, resolution, width,
                             derive_seed(seed, "generator")).to(device)
    return train, rng, branch, kernel, config, model


def probe_cell(width: int, positives: int, seed: int, device, resolution: int,
               root: str | None, steps: int) -> dict:
    """Memory, speed and numerical health for one factorial cell."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    train, rng, branch, kernel, config, model = build(
        width, positives, seed, device, resolution, root)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    losses, gains, moments = [], [], []
    started = time.time()
    for step in range(steps):
        latent = sample_latent(CLOUD, config.latent_dim,
                               derive_seed(seed, "latent", step), device)
        output = model(latent)
        with torch.no_grad():
            real = train.sample(positives, rng)
            drift, _ = KG.field(output.detach(), real, output.detach(), branch,
                                kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            report: dict = {}
            teacher = corrected_teacher(output.detach() + ETA * drift, real,
                                        mode="scalar", report=report)
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        gains.append(float(report.get("correction_ratio_median", float("nan"))))
        if step % 50 == 0:
            moments.append(float(output.detach().flatten(1).var(0).mean()))
    elapsed = time.time() - started

    with torch.no_grad():
        emitted = model(sample_latent(CLOUD, config.latent_dim,
                                      derive_seed(seed, "check"), device))
        realized, dead = rectangular_ess(kernel, branch, emitted,
                                         train.sample(positives, rng))
    peak = (torch.cuda.max_memory_allocated() / 2 ** 20
            if device.type == "cuda" else float("nan"))
    reserved = (torch.cuda.max_memory_reserved() / 2 ** 20
                if device.type == "cuda" else float("nan"))
    half = len(losses) // 2
    return {
        "width": width, "positives": positives,
        "parameters": model.parameter_count(),
        "peak_allocated_mib": peak, "peak_reserved_mib": reserved,
        "seconds_per_step": elapsed / max(steps, 1),
        "projected_hours_per_run": elapsed / max(steps, 1) * BUDGET_STEPS / 3600,
        "loss_first_half": float(np.mean(losses[:half])),
        "loss_second_half": float(np.mean(losses[half:])),
        "loss_finite": bool(np.all(np.isfinite(losses))),
        "loss_decreasing": bool(np.mean(losses[half:])
                                < np.mean(losses[:half])),
        "r11_gain_median": float(np.nanmedian(gains)),
        "output_variance_last": moments[-1] if moments else float("nan"),
        "realized_ess": realized, "dead_row_fraction": dead,
        "output_finite": bool(torch.isfinite(emitted).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--steps", type=int, default=PROBE_STEPS)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase30_preflight.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    seed = MASTER_SEED + SEED_OFFSET

    started = time.time()

    print("=== S: does the recall measurement work IN THIS SCRIPT? ===",
          flush=True)
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(seed, "p30-eval"))
    real_a = evaluation.sample(args.samples, rng)
    real_b = evaluation.sample(args.samples, rng)
    reference = inception_features(real_a, device).double().numpy()
    sanity = {
        "real": precision_recall(
            inception_features(real_b, device).double().numpy(), reference),
        "gaussian": precision_recall(
            inception_features(gaussian_moment_match(real_a, args.samples, rng),
                               device).double().numpy(), reference),
    }
    for name, entry in sanity.items():
        print(f"    {name:10} precision={entry['precision']:.3f} "
              f"recall={entry['recall']:.3f}", flush=True)
    metric_ok = (sanity["real"]["recall"] > 0.5
                 and sanity["gaussian"]["recall"] < 0.05)
    print(f"    metric sane: {metric_ok}"
          f"   (real recall must be > 0.5, gaussian < 0.05)\n", flush=True)

    print(f"=== M/T/N: per-cell memory, speed and health ({args.steps} steps) "
          f"===", flush=True)
    rows = []
    for width, positives in CELLS:
        entry = probe_cell(width, positives, seed, device, args.resolution,
                           args.data_root, args.steps)
        rows.append(entry)
        print(f"    w{width:<4} p{positives:<4} "
              f"params={entry['parameters']:>9,} "
              f"mem={entry['peak_reserved_mib']:7.0f}MiB "
              f"{entry['seconds_per_step']*1000:6.1f}ms/step "
              f"-> {entry['projected_hours_per_run']:5.2f}h/run  "
              f"loss {entry['loss_first_half']:.1f}->"
              f"{entry['loss_second_half']:.1f} "
              f"ESS={entry['realized_ess']:.3f} "
              f"R11={entry['r11_gain_median']:.3f}", flush=True)

    memory_only = []
    for width, positives in MEMORY_ONLY:
        try:
            entry = probe_cell(width, positives, seed, device, args.resolution,
                               args.data_root, 20)
            memory_only.append(entry)
            print(f"    w{width:<4} p{positives:<4} (headroom probe) "
                  f"mem={entry['peak_reserved_mib']:7.0f}MiB "
                  f"{entry['seconds_per_step']*1000:6.1f}ms/step", flush=True)
        except torch.cuda.OutOfMemoryError:
            memory_only.append({"width": width, "positives": positives,
                                "oom": True})
            print(f"    w{width:<4} p{positives:<4} (headroom probe) OOM",
                  flush=True)

    limit = CARD_MIB * MEMORY_HEADROOM
    worst = max(rows, key=lambda r: r["peak_reserved_mib"])
    total_hours = sum(r["projected_hours_per_run"] for r in rows) * SEEDS
    checks = {
        "metric_sane": bool(metric_ok),
        "all_cells_fit": bool(worst["peak_reserved_mib"] < limit),
        "all_losses_finite": all(r["loss_finite"] for r in rows),
        "all_outputs_finite": all(r["output_finite"] for r in rows),
        "all_losses_decreasing": all(r["loss_decreasing"] for r in rows),
        "no_dead_kernel_rows": all(r["dead_row_fraction"] == 0.0
                                   for r in rows),
        "r11_gain_sane": all(0.5 < r["r11_gain_median"] < 2.0 for r in rows),
        "ess_sane": all(0.05 < r["realized_ess"] < 0.99 for r in rows),
    }
    verdict = {
        "checks": checks, "all_passed": all(checks.values()),
        "memory_limit_mib": limit, "worst_cell_mib": worst["peak_reserved_mib"],
        "worst_cell": f"w{worst['width']}_p{worst['positives']}",
        "projected_total_hours": total_hours,
        "budget_steps": BUDGET_STEPS, "seeds": SEEDS,
        "recall_success_threshold": RECALL_SUCCESS,
        "recall_anchors": {"fresh_latent_objectives": 0.000,
                           "memorization_ceiling": 0.224,
                           "autoencoder_d512": 0.496,
                           "real_data": sanity["real"]["recall"]},
    }
    verdict["reading"] = (
        f"GO -- all checks pass; the 2x2 factorial at {BUDGET_STEPS} steps x "
        f"{SEEDS} seeds projects to {total_hours:.1f} h"
        if verdict["all_passed"] else
        "NO-GO -- at least one check failed; fix before scheduling anything")

    payload = {"status": "phase30-preflight",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "metric_sanity": sanity, "rows": rows,
               "memory_only": memory_only, "verdict": verdict}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 30 PRE-FLIGHT ===")
    for name, passed in checks.items():
        print(f"    {'PASS' if passed else 'FAIL'}  {name}")
    print(f"\n    worst cell {verdict['worst_cell']} at "
          f"{worst['peak_reserved_mib']:.0f} MiB against a {limit:.0f} MiB "
          f"budget ({CARD_MIB} MiB card x {MEMORY_HEADROOM:.0%})")
    print(f"    projected total: {total_hours:.1f} h "
          f"({len(rows)} cells x {SEEDS} seeds x {BUDGET_STEPS} steps)")
    print(f"    success threshold: recall > {RECALL_SUCCESS} "
          f"(anchors: 0.000 observed six times, 0.224 memorization)")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
