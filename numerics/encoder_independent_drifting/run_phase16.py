"""Phase 16 (protocol `EncoderIndependentPhase16Protocol.md`).

Can drifting produce anything a moment-matched Gaussian cannot?

The metric audit showed energy distance in pixel space is saturated by
matching the first two moments -- a structureless Gaussian scores ED2 0.1079
against a real sample's 0.1118 while reading FID 240.6 -- and that R11, the
program's headline reform, is worth 6.2x on ED2 and 6% on FID.

So **FID is the primary readout here and ED2 is a diagnostic**, which
inverts thirteen phases of practice.  The bar is 240.6 and it does not move.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from . import metrics as M
from .config import (
    GeometryConfig, MASTER_SEED, TrainConfig, config_digest, derive_seed,
)
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnostics import provenance, write_json
from .encoders import encoder_branch
from .fid import frechet_distance
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher

HERE = Path(__file__).resolve().parent

SEED_OFFSET = 28000
GOOD_ESS = 0.9
POSITIVES = 64
EVAL_SAMPLES = 512
TAIL_KEEP = 32

# The bar, from `EncoderIndependentMetricAudit.md`.  Recomputed in this run
# so every number comes from one seed stream, but it does not move.
BUDGETS = (600, 3000, 10000, 30000)
STAGE2_BUDGET = 10000
STAGE2 = (("S2_base", 64, 256, "raw"),
          ("S2_wide", 256, 256, "raw"),
          ("S2_cloud", 64, 1024, "raw"),
          ("S2_encoder", 64, 256, "pretrained"))


def _tail(x: torch.Tensor) -> float:
    flat = x.reshape(len(x), -1)
    power = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True)) ** 2
    return float(power[TAIL_KEEP:].sum() / power.sum())


def _geometry_branch(kind: str, seed: int, device):
    if kind == "raw":
        geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                                  target_ess_fraction=GOOD_ESS)
        return build_family(geometry, 3).branches[0]
    return encoder_branch(kind, seed=seed, device=device)


def train_arm(kind: str, width: int, cloud: int, steps: int, seed: int,
              device, resolution: int, root: str | None,
              r11: bool = True) -> dict:
    train = cifar.cifar_target(resolution, "train", root)
    train.device = device
    rng = np.random.default_rng(derive_seed(seed, "p16", kind, width, cloud))
    branch = _geometry_branch(kind, seed, device)
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=GOOD_ESS)
    config = TrainConfig(steps=steps, batch=POSITIVES, image_size=resolution,
                         width=width)
    model = OneStepGenerator(config.latent_dim, 3, resolution, width,
                             derive_seed(seed, "generator")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    started = time.time()
    for step in range(steps):
        latent = sample_latent(cloud, config.latent_dim,
                               derive_seed(seed, "latent", step), device)
        output = model(latent)
        with torch.no_grad():
            positives = train.sample(POSITIVES, rng)
            drift, _ = KG.field(output.detach(), positives, output.detach(),
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            teacher = output.detach() + 0.5 * drift
            if r11:
                teacher = corrected_teacher(teacher, positives, mode="scalar")
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe = sample_latent(EVAL_SAMPLES, config.latent_dim,
                          derive_seed(seed, "probe"), device)
    with torch.no_grad():
        generated = model(probe)
    return {"sample": generated, "wall_seconds": time.time() - started}


def score(sample: torch.Tensor, real: torch.Tensor, device) -> dict:
    return {
        "fid": frechet_distance(sample, real, device)["fid"],
        "ed2": M.energy_distance2(sample.cpu(), real.cpu()),
        "tail": _tail(sample.cpu()),
        "second_moment": float(sample.flatten(1).var(0).mean().cpu()
                               / real.flatten(1).var(0).mean().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--budgets", type=str,
                        default=",".join(str(b) for b in BUDGETS))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase16.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    budgets = [int(b) for b in args.budgets.split(",")]

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(MASTER_SEED + SEED_OFFSET, "p16"))
    real_a = evaluation.sample(EVAL_SAMPLES, rng)
    real_b = evaluation.sample(EVAL_SAMPLES, rng)

    print("=== baselines, recomputed in this run ===", flush=True)
    baselines = {
        "real": score(real_b, real_a, device),
        "moment_matched_gaussian": score(
            gaussian_moment_match(real_a, EVAL_SAMPLES, rng), real_a, device),
        "pure_noise": score(
            torch.tensor(rng.normal(scale=0.5, size=tuple(real_b.shape)),
                         dtype=torch.float32), real_a, device),
    }
    for name, entry in baselines.items():
        print(f"    {name:26} FID={entry['fid']:8.2f} "
              f"ED2={entry['ed2']:8.4f}", flush=True)
    bar = baselines["moment_matched_gaussian"]["fid"]
    print(f"\n    THE BAR = {bar:.2f} (moment-matched Gaussian)", flush=True)

    rows = []
    print("\n=== Stage 1: the scaling curve (raw + R11, w64, cloud 256) ===",
          flush=True)
    for steps in budgets:
        for index in range(args.seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            out = train_arm("raw", 64, 256, steps, seed, device,
                            args.resolution, args.data_root)
            row = {"stage": "S1", "arm": f"S1_steps{steps}", "kind": "raw",
                   "width": 64, "cloud": 256, "steps": steps, "seed": seed,
                   "wall_seconds": out["wall_seconds"],
                   **score(out["sample"], real_a, device)}
            row["beats_bar"] = bool(row["fid"] < bar)
            rows.append(row)
            print(f"    steps={steps:6} seed{index} FID={row['fid']:8.2f} "
                  f"({'BEATS' if row['beats_bar'] else 'above'} bar)  "
                  f"ED2={row['ed2']:8.4f} tail={row['tail']:7.4f} "
                  f"2nd={row['second_moment']:6.3f} "
                  f"{row['wall_seconds']:6.0f}s", flush=True)

    print(f"\n=== Stage 2: other axes at {STAGE2_BUDGET} steps ===",
          flush=True)
    for label, width, cloud, kind in STAGE2:
        for index in range(args.seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            out = train_arm(kind, width, cloud, STAGE2_BUDGET, seed, device,
                            args.resolution, args.data_root)
            row = {"stage": "S2", "arm": label, "kind": kind, "width": width,
                   "cloud": cloud, "steps": STAGE2_BUDGET, "seed": seed,
                   "wall_seconds": out["wall_seconds"],
                   **score(out["sample"], real_a, device)}
            row["beats_bar"] = bool(row["fid"] < bar)
            rows.append(row)
            print(f"    {label:12} seed{index} FID={row['fid']:8.2f} "
                  f"({'BEATS' if row['beats_bar'] else 'above'} bar)  "
                  f"ED2={row['ed2']:8.4f} tail={row['tail']:7.4f} "
                  f"{row['wall_seconds']:6.0f}s", flush=True)

    summary = {}
    for arm in sorted({r["arm"] for r in rows}):
        group = [r for r in rows if r["arm"] == arm]
        summary[arm] = {
            "kind": group[0]["kind"], "width": group[0]["width"],
            "cloud": group[0]["cloud"], "steps": group[0]["steps"],
            **{k: float(np.median([r[k] for r in group]))
               for k in ("fid", "ed2", "tail", "second_moment",
                         "wall_seconds")},
            "beats_bar": all(r["beats_bar"] for r in group)}

    curve = [(summary[f"S1_steps{s}"]["steps"], summary[f"S1_steps{s}"]["fid"])
             for s in budgets if f"S1_steps{s}" in summary]
    falling = all(b[1] <= a[1] + 1e-9 for a, b in zip(curve, curve[1:]))
    best = min(summary.values(), key=lambda v: v["fid"])
    verdict = {
        "bar": bar, "floor": baselines["real"]["fid"],
        "ceiling": baselines["pure_noise"]["fid"],
        "scaling_curve": curve,
        "fid_falls_with_budget": falling,
        "best_fid": best["fid"], "best_arm_beats_bar": bool(best["fid"] < bar),
        "any_arm_beats_bar": any(v["beats_bar"] for v in summary.values()),
    }
    verdict["reading"] = (
        "drifting produces structure a moment-matched Gaussian cannot; the "
        "encoder ladder becomes meaningful"
        if verdict["best_arm_beats_bar"] else
        "no arm beats a moment-matched Gaussian -- at this scale the recipe "
        "produces moments, not structure")

    payload = {
        "status": "phase16-frozen-protocol",
        "protocol": "numerics/EncoderIndependentPhase16Protocol.md",
        "provenance": provenance(), "device": settings,
        "config": vars(args) | {"out": str(args.out)},
        "config_digest": config_digest(
            TrainConfig(image_size=args.resolution)),
        "elapsed_seconds": time.time() - started,
        "baselines": baselines, "summary": summary, "verdict": verdict,
        "rows": rows,
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 16 ===")
    print(f"{'arm':16}{'geom':12}{'w':>5}{'cloud':>7}{'steps':>8}"
          f"{'FID':>9}{'ED2':>9}{'tail':>8}{'2nd':>7}  vs bar")
    for key, e in summary.items():
        print(f"{key:16}{e['kind']:12}{e['width']:5}{e['cloud']:7}"
              f"{e['steps']:8}{e['fid']:9.2f}{e['ed2']:9.4f}"
              f"{e['tail']:8.4f}{e['second_moment']:7.3f}  "
              f"{'BEATS' if e['beats_bar'] else 'above'}")
    print(f"\n    floor (real) {verdict['floor']:.2f}   "
          f"BAR (gaussian) {bar:.2f}   ceiling (noise) "
          f"{verdict['ceiling']:.2f}")
    print(f"    scaling curve: {verdict['scaling_curve']}")
    print(f"    FID falls monotonically with budget: "
          f"{verdict['fid_falls_with_budget']}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
