"""Phase 30 (protocol `EncoderIndependentPhase30Protocol.md`).

Does capacity or batch size unlock nonzero recall?

Twenty-nine phases produced recall 0.000 under every fresh-latent objective
tested.  The generator is the one component never varied, and Phase 28 showed it
*can* hold recall 0.224 when memorizing while exhausting its capacity somewhere
between 512 and 2048 images.

The pre-flight (`phase30_preflight.json`) returned NO-GO on the design first
proposed: width 256 at cloud 256 needs 5972 MiB on a 6141 MiB card, spills to
system memory and runs 95x slower than width 64 -- 27 h for one run against the
0.29 h the baseline takes.  Cloud size is therefore held at 256 for
comparability with the Phase 28/29 reference, which caps capacity at width 192.

Success is declared at **recall > 0.05**, calibrated against measured anchors:
0.000 across six independent objectives, 0.224 at the memorization ceiling,
0.496 for an autoencoder reconstruction, 0.737 for real data.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase30
"""

from __future__ import annotations

import argparse
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .config import MASTER_SEED, GeometryConfig, TrainConfig, derive_seed
from .device import configure, resolve_device
from .diagnose_phase20 import save_grid
from .diagnose_phase25 import rectangular_ess
from .diagnose_phase26 import score
from .diagnostics import provenance, write_json
from .fid import inception_features
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher
from .preflight_phase30 import RECALL_SUCCESS, TARGET_ESS

HERE = Path(__file__).resolve().parent

SEED_OFFSET = 50000
CLOUD = 256           # fixed: comparability with the Phase 28/29 reference
ETA = 0.5
EVAL = 512
RECORD_EVERY = 7500

# (label, width, positives) -- capacity ladder plus one batch contrast
ARMS = (("w64_p64", 64, 64),
        ("w64_p256", 64, 256),
        ("w128_p256", 128, 256),
        ("w192_p256", 192, 256))

ANCHORS = {"fresh_latent_objectives": 0.000, "memorization_ceiling": 0.224,
           "autoencoder_d512": 0.496}


def train_arm(width: int, positives: int, steps: int, seed: int, device,
              resolution: int, root: str | None, reference, real) -> dict:
    train = cifar.cifar_target(resolution, "train", root)
    train.device = device
    rng = np.random.default_rng(derive_seed(seed, "p30", width, positives))
    branch = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace",
                       target_ess_fraction=TARGET_ESS), 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=TARGET_ESS)
    config = TrainConfig(steps=steps, batch=positives, image_size=resolution,
                         width=width)
    model = OneStepGenerator(config.latent_dim, 3, resolution, width,
                             derive_seed(seed, "generator")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    probe = sample_latent(EVAL, config.latent_dim,
                          derive_seed(seed, "probe"), device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = []
    started = time.time()
    for step in range(steps + 1):
        if step % RECORD_EVERY == 0:
            with torch.no_grad():
                emitted = model(probe).detach()
            entry = {"step": step, **score(emitted, reference, real, device)}
            history.append(entry)
            print(f"      w{width}_p{positives:<4} step {step:6} "
                  f"KID={entry['kid']:+.5f} P={entry['precision']:.3f} "
                  f"R={entry['recall']:.3f} alpha={entry['alpha']:.3f} "
                  f"2nd={entry['second_moment']:.3f}", flush=True)
        if step == steps:
            break
        # Cosine schedule: the Phase 28/29 reference used it, and a constant
        # schedule would break comparability with the recall-0.000 baseline.
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * 0.5 * (
                1.0 + np.cos(np.pi * step / max(steps, 1)))
        latent = sample_latent(CLOUD, config.latent_dim,
                               derive_seed(seed, "latent", step), device)
        output = model(latent)
        with torch.no_grad():
            real_batch = train.sample(positives, rng)
            drift, _ = KG.field(output.detach(), real_batch, output.detach(),
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            teacher = corrected_teacher(output.detach() + ETA * drift,
                                        real_batch, mode="scalar")
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    wall = time.time() - started

    with torch.no_grad():
        emitted = model(probe).detach()
        realized, dead = rectangular_ess(kernel, branch, emitted,
                                         train.sample(positives, rng))
    return {"width": width, "positives": positives,
            "parameters": model.parameter_count(),
            "wall_seconds": wall, "history": history,
            "realized_ess": realized, "dead_row_fraction": dead,
            "peak_mib": (torch.cuda.max_memory_allocated() / 2 ** 20
                         if device.type == "cuda" else float("nan")),
            "final": history[-1], "sample": emitted}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase30.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(MASTER_SEED + SEED_OFFSET, "p30"))
    real = evaluation.sample(args.samples, rng)
    reference = inception_features(real, device).double().numpy()
    print(f"    success threshold recall > {RECALL_SUCCESS}"
          f"   anchors {ANCHORS}\n", flush=True)

    rows = []
    for label, width, positives in ARMS:
        for index in range(args.seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            out = train_arm(width, positives, args.steps, seed, device,
                            args.resolution, args.data_root, reference, real)
            sample = out.pop("sample")
            if index == 0:
                save_grid(sample[:64].cpu(),
                          HERE / f"phase30_samples_{label}.png")
            out |= {"arm": label, "seed": seed}
            rows.append(out)
            e = out["final"]
            print(f"    {label:11} seed{index} KID={e['kid']:+.5f} "
                  f"P={e['precision']:.3f} R={e['recall']:.3f} "
                  f"{out['wall_seconds']:6.0f}s "
                  f"{out['peak_mib']:.0f}MiB", flush=True)

    summary = {}
    for label, width, positives in ARMS:
        group = [r for r in rows if r["arm"] == label]
        summary[label] = {
            "width": width, "positives": positives,
            "parameters": group[0]["parameters"],
            **{k: float(np.median([r["final"][k] for r in group]))
               for k in ("kid", "precision", "recall", "alpha", "tail",
                         "second_moment")},
            "max_recall": float(max(r["final"]["recall"] for r in group)),
            "median_wall_seconds": float(np.median(
                [r["wall_seconds"] for r in group])),
            "realized_ess": float(np.median(
                [r["realized_ess"] for r in group])),
        }

    ladder = [summary[k]["recall"] for k in ("w64_p256", "w128_p256",
                                             "w192_p256")]
    best = max(rows, key=lambda r: r["final"]["recall"])
    verdict = {
        "recall_success_threshold": RECALL_SUCCESS, "anchors": ANCHORS,
        "summary": summary,
        "capacity_ladder_recall": ladder,
        "recall_rises_with_capacity": bool(
            all(b >= a - 1e-9 for a, b in pairwise(ladder))
            and ladder[-1] > ladder[0]),
        "batch_effect_recall": summary["w64_p256"]["recall"]
                               - summary["w64_p64"]["recall"],
        "any_arm_succeeds": bool(any(s["max_recall"] > RECALL_SUCCESS
                                     for s in summary.values())),
        "best": {"arm": best["arm"], "seed": best["seed"],
                 **best["final"]},
        "kid_ladder": [summary[k]["kid"] for k in ("w64_p256", "w128_p256",
                                                   "w192_p256")],
    }
    if verdict["any_arm_succeeds"]:
        verdict["reading"] = (
            "scale unlocks coverage -- recall leaves 0.000 for the first time "
            "in the program, and the encoder ladder becomes a real experiment")
    elif verdict["recall_rises_with_capacity"]:
        verdict["reading"] = (
            "recall rises with capacity but no arm crosses the threshold -- the "
            "direction is right and this harness is under-scaled")
    else:
        verdict["reading"] = (
            "capacity and batch are NOT the obstruction -- recall stays at "
            "0.000 across a 7.5x capacity range, leaving Phase 29's "
            "correspondence-stability problem as the remaining candidate")

    payload = {"status": "phase30-capacity-and-batch",
               "protocol": "numerics/EncoderIndependentPhase30Protocol.md",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "verdict": verdict, "rows": rows}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 30: DOES SCALE UNLOCK COVERAGE? ===")
    print(f"{'arm':11}{'params':>10}{'pos':>6}{'KID':>10}{'prec':>8}"
          f"{'recall':>8}{'alpha':>8}{'ESS':>7}{'hours':>7}")
    for label, _, _ in ARMS:
        s = summary[label]
        print(f"{label:11}{s['parameters']:>10,}{s['positives']:>6}"
              f"{s['kid']:+10.5f}{s['precision']:8.3f}{s['recall']:8.3f}"
              f"{s['alpha']:8.3f}{s['realized_ess']:7.3f}"
              f"{s['median_wall_seconds']/3600:7.2f}")
    for name, value in ANCHORS.items():
        print(f"{'(' + name[:9] + ')':11}{'':>10}{'':>6}{'':>10}{'':>8}"
              f"{value:8.3f}")
    print(f"\n    capacity ladder recall: "
          f"{[round(x, 4) for x in verdict['capacity_ladder_recall']]}")
    print(f"    capacity ladder KID:    "
          f"{[round(x, 4) for x in verdict['kid_ladder']]}")
    print(f"    batch effect on recall (p256 - p64): "
          f"{verdict['batch_effect_recall']:+.4f}")
    print(f"    best single run: {verdict['best']['arm']} "
          f"recall={verdict['best']['recall']:.4f} "
          f"KID={verdict['best']['kid']:+.5f}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
