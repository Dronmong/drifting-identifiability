"""Phase 18B: does a SHALLOW pretrained encoder work?

The invariance probe refuted the reading I had proposed.  A pretrained
ResNet18 is not invariant to the details that make images look real -- it is
**hypersensitive** to them, 12.6x the untrained network's response to
high-frequency noise at layer3 and 14.2x at layer4, rising monotonically with
depth.  That predicts the field chases fine detail, which is exactly the
spectral tail of 0.4853 Phase 17 measured against real data's 0.13.

If depth drives the failure, a shallow pretrained encoder should work -- and
then the answer to encoder dependence is **"a shallower one"**, not "none",
which is a far more useful result than a bare negative.

Screened at a reduced budget: Phase 14A saw the pretrained penalty at 600
steps (+38 FID) and Phase 17 at 30 000 (+138), so the effect is visible well
below the full budget and the ordering is what matters here.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase18b
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .config import (
    GeometryConfig, MASTER_SEED, TrainConfig, derive_seed,
)
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnostics import provenance, write_json
from .encoders import encoder_branch
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher
from .run_phase16 import EVAL_SAMPLES, score

HERE = Path(__file__).resolve().parent

SEED_OFFSET = 31000
GOOD_ESS = 0.9
POSITIVES = 64
WIDTH = 64
CLOUD = 256

# (label, kind, layer)
ARMS = (("raw", "raw", None),
        ("pre_layer1", "pretrained", "layer1"),
        ("pre_layer2", "pretrained", "layer2"),
        ("pre_layer3", "pretrained", "layer3"),
        ("pre_layer4", "pretrained", "layer4"),
        ("rand_layer3", "random", "layer3"))


def train(kind: str, layer: str | None, steps: int, seed: int, device,
          resolution: int, root: str | None) -> torch.Tensor:
    train_target = cifar.cifar_target(resolution, "train", root)
    train_target.device = device
    rng = np.random.default_rng(derive_seed(seed, "p18b", kind, layer))
    if kind == "raw":
        branch = build_family(
            GeometryConfig(family="raw", base_kernel="smooth_laplace",
                           target_ess_fraction=GOOD_ESS), 3).branches[0]
    else:
        branch = encoder_branch(kind, seed=seed, device=device, layer=layer)
    kernel = calibrate_block_kernel(
        branch, train_target.sample(256, rng), "smooth_laplace", 0.5, 1.0,
        1e-3, combine="sum", target_ess_fraction=GOOD_ESS)
    config = TrainConfig(steps=steps, batch=POSITIVES, image_size=resolution,
                         width=WIDTH)
    model = OneStepGenerator(config.latent_dim, 3, resolution, WIDTH,
                             derive_seed(seed, "generator")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for step in range(steps):
        latent = sample_latent(CLOUD, config.latent_dim,
                               derive_seed(seed, "latent", step), device)
        output = model(latent)
        with torch.no_grad():
            positives = train_target.sample(POSITIVES, rng)
            drift, _ = KG.field(output.detach(), positives, output.detach(),
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            teacher = corrected_teacher(output.detach() + 0.5 * drift,
                                        positives, mode="scalar")
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe = sample_latent(EVAL_SAMPLES, config.latent_dim,
                          derive_seed(seed, "probe"), device)
    with torch.no_grad():
        return model(probe)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase18b.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(MASTER_SEED + SEED_OFFSET, "p18b"))
    real_a = evaluation.sample(EVAL_SAMPLES, rng)
    bar = score(gaussian_moment_match(real_a, EVAL_SAMPLES, rng), real_a,
                device)["fid"]
    print(f"    BAR (moment-matched gaussian) = {bar:.2f}\n", flush=True)

    rows = []
    for label, kind, layer in ARMS:
        for index in range(args.seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            began = time.time()
            sample = train(kind, layer, args.steps, seed, device,
                           args.resolution, args.data_root)
            row = {"arm": label, "kind": kind, "layer": layer, "seed": seed,
                   "wall_seconds": time.time() - began,
                   **score(sample, real_a, device)}
            rows.append(row)
            print(f"    {label:14} seed{index} FID={row['fid']:8.2f} "
                  f"ED2={row['ed2']:8.4f} tail={row['tail']:7.4f} "
                  f"2nd={row['second_moment']:6.3f} "
                  f"{row['wall_seconds']:6.0f}s", flush=True)

    summary = {}
    for label, kind, layer in ARMS:
        group = [r for r in rows if r["arm"] == label]
        summary[label] = {
            "kind": kind, "layer": layer,
            **{k: float(np.median([r[k] for r in group]))
               for k in ("fid", "ed2", "tail", "second_moment")}}
    seeds = sorted({r["seed"] for r in rows})
    table = {(r["arm"], r["seed"]): r["fid"] for r in rows}
    paired = {label: float(np.mean([table[(label, s)] - table[("raw", s)]
                                    for s in seeds]))
              for label, _, _ in ARMS if label != "raw"}
    depth = [summary[f"pre_layer{i}"]["fid"] for i in (1, 2, 3, 4)]
    verdict = {
        "bar": bar, "paired_vs_raw": paired,
        "fid_by_depth": depth,
        "fid_rises_with_depth": all(b >= a - 1e-9
                                    for a, b in zip(depth, depth[1:])),
        "best_pretrained_layer": min(("layer1", "layer2", "layer3", "layer4"),
                                     key=lambda x: summary[
                                         f"pre_{x}"]["fid"]),
        "shallow_beats_raw": paired["pre_layer1"] < 0,
    }
    verdict["reading"] = (
        "a SHALLOW pretrained encoder beats raw pixels -- the answer to "
        "encoder dependence is a shallower encoder, not none"
        if verdict["shallow_beats_raw"] else
        "no pretrained depth beats raw pixels; the failure is not depth alone")

    payload = {"status": "phase18b-encoder-depth-sweep",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "bar": bar, "summary": summary, "verdict": verdict,
               "rows": rows}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 18B ===")
    print(f"{'arm':16}{'FID':>9}{'ED2':>9}{'tail':>8}{'2nd':>7}"
          f"{'paired vs raw':>15}")
    for key in sorted(summary, key=lambda k: summary[k]["fid"]):
        e = summary[key]
        p = paired.get(key)
        print(f"{key:16}{e['fid']:9.2f}{e['ed2']:9.4f}{e['tail']:8.4f}"
              f"{e['second_moment']:7.3f}"
              f"{('--' if p is None else f'{p:+.2f}'):>15}")
    print(f"\n    BAR {bar:.2f}   (real data tail ~= 0.13)")
    print(f"    FID by depth (layer1->4): "
          f"{[round(x, 1) for x in depth]}")
    print(f"    rises monotonically with depth: "
          f"{verdict['fid_rises_with_depth']}")
    print(f"    best pretrained layer: {verdict['best_pretrained_layer']}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
