"""Phase 22 (protocol `EncoderIndependentPhase22Protocol.md`).

Can any variant of this mechanism produce image structure?

Phase 20 rendered the best generator this program has produced and it holds
no image structure at all -- smooth colour swirls, no objects, no edges.  So
every FID in Phases 16-19 was ranking blobs.

Phase 21 found the first real lever and its limit.  The drift is a bi-softmax
weighted AVERAGE of the positives, so every target averages real images and
the row ESS sets how many.  Sharpening from the 0.9 of Phases 7-18 to 0.05 is
worth 17 FID, monotone over four rungs.  But calibrating to 0.05 *realizes*
0.609 -- a 12x gap -- because a point off the data manifold is nearly
equidistant from every real image.  Even the sharpest working setting still
averages 29 images.  Bandwidth alone cannot escape distance concentration.

This run tests the axes that attack concentration directly, chosen because
they are nearly free: measured at 30 000 steps, base 0.31 h, bandwidth
mixture 0.35 h, positives 256 0.35 h -- against cloud 1024 at 1.55 h, which
would spend the night on one axis.

KID is primary: Phase 20 measured FID's floor at 70.65 / 43.02 / 23.08 for
n = 512 / 1024 / 2048 while KID read 0.14200 / 0.14282 / 0.14210 over the
same draws.  Sample grids are a first-class output -- a metric that cannot
tell two blobs apart cannot be the only readout.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase22
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sstats

from . import cifar
from . import kernel_gradient as KG
from . import metrics as M
from .config import (
    MASTER_SEED,
    GeometryConfig,
    TrainConfig,
    config_digest,
    derive_seed,
)
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnose_phase20 import save_grid
from .diagnostics import provenance, write_json
from .fid import frechet_from_features, inception_features, kid_from_features
from .fixed_features import bandwidth_mixture, build_family
from .kernels import calibrate_block_kernel, geometric_multipliers
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher
from .run_phase16 import TAIL_KEEP
from .run_phase19 import CLOUD, WIDTH, learning_rate

HERE = Path(__file__).resolve().parent

SEED_OFFSET = 35000
SMALL_N = 512          # continuity with Phases 16-19
LARGE_N = 2048         # where FID is not bias-dominated
MIX_SPAN = 4.0

# (label, target ESS, bandwidth levels, positives)
#
# The axis that matters is how many real images each target AVERAGES, and
# the ESS *fraction* is roughly scale-free: holding it while raising the
# positive count raises the count (measured: 42 images at 64 positives,
# 168 at 256).  So more positives at a fixed bandwidth is BLURRIER, not
# sharper.  The mechanism question is whether sharpening past the point
# where Phase 21 reversed (0.02, 29 of 64 images) becomes viable once there
# are more positives to select among -- which is the C/E pair.
# 0.005 was measured DEAD: collapsed row fraction 1.0, affinity median
# exactly 0, half the denominators on the floor -- `exp(-d/tau)` underflows
# in float32 before the mechanism runs out.  0.01 is the sharpest verified
# viable setting (18.4 images averaged at 64 positives, 59.1 at 256).
ARMS = (("A_control", 0.500, 1, 64),
        ("B_sharp", 0.050, 1, 64),
        ("C_sharper", 0.010, 1, 64),
        ("D_pos", 0.050, 1, 256),
        ("E_sharper_pos", 0.010, 1, 256),
        ("F_mix", 0.050, 5, 64))

BASELINE = "A_control"
CONTRASTS = (("sharp", "A_control", "B_sharp"),
             ("sharper_at_pos64", "B_sharp", "C_sharper"),
             ("sharper_at_pos256", "D_pos", "E_sharper_pos"),
             ("positives_at_ess05", "B_sharp", "D_pos"),
             # The interaction the run exists for: does a bigger attractor
             # set rescue sharpening that fails with only 64 positives?
             ("positives_at_ess005", "C_sharper", "E_sharper_pos"),
             ("mixture", "B_sharp", "F_mix"))


def build_geometry(ess: float, levels: int):
    """Branch plus the declared bandwidth ladder for a mixture."""
    base = build_family(
        GeometryConfig(family="raw", base_kernel="smooth_laplace",
                       target_ess_fraction=ess), 3).branches[0]
    if levels == 1:
        return base, None
    return (bandwidth_mixture(base, levels),
            geometric_multipliers(levels, MIX_SPAN))


def train_arm(ess: float, levels: int, positives: int, steps: int, seed: int,
              device, resolution: int, root: str | None,
              pool: int) -> dict:
    """One run.  Every stream derives from the seed alone, so arms are paired."""
    train = cifar.cifar_target(resolution, "train", root)
    train.device = device
    rng = np.random.default_rng(derive_seed(seed, "p22"))
    branch, multipliers = build_geometry(ess, levels)
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=ess, tau_multipliers=multipliers)
    config = TrainConfig(steps=steps, batch=positives, image_size=resolution,
                         width=WIDTH)
    model = OneStepGenerator(config.latent_dim, 3, resolution, WIDTH,
                             derive_seed(seed, "generator")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    began = time.time()
    realized = []
    for step in range(steps):
        lr = learning_rate(config.learning_rate, "cosine", step, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        latent = sample_latent(CLOUD, config.latent_dim,
                               derive_seed(seed, "latent", step), device)
        output = model(latent)
        want = step % max(steps // 10, 1) == 0
        with torch.no_grad():
            batch = train.sample(positives, rng)
            drift, health = KG.field(output.detach(), batch, output.detach(),
                                     branch, kernel, direction_mode="paper",
                                     normalization="rms", diagnostics=want)
            teacher = corrected_teacher(output.detach() + 0.5 * drift, batch,
                                        mode="scalar")
            if want:
                realized.append({"step": step,
                                 "ess_fraction": health["ess_fraction"],
                                 "ess_count": health["ess_mean"],
                                 "collapsed": health["collapsed_row_fraction"]})
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    wall = time.time() - began

    probe = sample_latent(pool, config.latent_dim, derive_seed(seed, "probe"),
                          device)
    with torch.no_grad():
        generated = torch.cat([model(probe[i:i + 512])
                               for i in range(0, len(probe), 512)])
    return {"sample": generated, "wall_seconds": wall, "realized": realized,
            "bandwidth_median": float(kernel.taus.median())}


def tail(x: torch.Tensor) -> float:
    flat = x.reshape(len(x), -1)
    power = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True)) ** 2
    return float(power[TAIL_KEEP:].sum() / power.sum())


def score(generated: torch.Tensor, real: torch.Tensor, device) -> dict:
    """KID (primary, unbiased) plus FID at both sample counts."""
    gen_f = inception_features(generated, device).double().numpy()
    real_f = inception_features(real, device).double().numpy()
    out = {
        "kid": kid_from_features(gen_f[:LARGE_N], real_f[:LARGE_N]),
        "fid_512": frechet_from_features(gen_f[:SMALL_N], real_f[:SMALL_N]),
        "fid_2048": frechet_from_features(gen_f[:LARGE_N], real_f[:LARGE_N]),
        "tail": tail(generated[:LARGE_N].cpu()),
        "second_moment": float(
            generated[:LARGE_N].flatten(1).var(0).mean().cpu()
            / real[:LARGE_N].flatten(1).var(0).mean().cpu()),
        "ed2": M.energy_distance2(generated[:SMALL_N].cpu(),
                                  real[:SMALL_N].cpu()),
    }
    return out


def paired(table: dict, seeds: list[int], key: str, a: str, b: str) -> dict:
    """Paired b - a with mean, sem and a paired t.  No sign test (Phase 19)."""
    diff = np.array([table[(b, s)][key] - table[(a, s)][key] for s in seeds])
    t, p = sstats.ttest_1samp(diff, 0.0) if len(diff) > 1 else (np.nan, np.nan)
    return {"mean": float(diff.mean()),
            "sem": float(diff.std(ddof=1) / np.sqrt(len(diff)))
            if len(diff) > 1 else float("nan"),
            "t": float(t), "p": float(p),
            "per_seed": [float(x) for x in diff],
            "significant": bool(np.isfinite(p) and p < 0.05)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--pool", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase22.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(MASTER_SEED + SEED_OFFSET, "p22"))
    real = evaluation.sample(args.pool, rng)
    real_b = evaluation.sample(args.pool, rng)
    floor = score(real_b, real, device)
    bar = score(gaussian_moment_match(real, args.pool, rng), real, device)
    print(f"    floor  KID {floor['kid']:+.5f}  FID512 {floor['fid_512']:7.2f}"
          f"  FID2048 {floor['fid_2048']:7.2f}  tail {floor['tail']:.4f}",
          flush=True)
    print(f"    BAR    KID {bar['kid']:+.5f}  FID512 {bar['fid_512']:7.2f}"
          f"  FID2048 {bar['fid_2048']:7.2f}\n", flush=True)

    rows = []
    for label, ess, levels, positives in ARMS:
        for index in range(args.seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            out = train_arm(ess, levels, positives, args.steps, seed, device,
                            args.resolution, args.data_root, args.pool)
            scored = score(out["sample"], real, device)
            if index == 0:
                save_grid(out["sample"], HERE / f"phase22_samples_{label}.png")
            rows.append({"arm": label, "target_ess": ess, "levels": levels,
                         "positives": positives, "seed": seed,
                         "wall_seconds": out["wall_seconds"],
                         "realized_ess": out["realized"][-1],
                         "bandwidth_median": out["bandwidth_median"],
                         **scored})
            health = out["realized"][-1]
            # A dead kernel must never be reported as a quiet result: at
            # ESS 0.005 every row underflowed and the floor, not the data,
            # decided each update.
            dead = (not np.isfinite(health["ess_fraction"])
                    or health["collapsed"] > 0.0)
            print(f"    {label:17} seed{index} KID={scored['kid']:+.5f} "
                  f"FID512={scored['fid_512']:7.2f} "
                  f"FID2048={scored['fid_2048']:7.2f} "
                  f"tail={scored['tail']:.4f} "
                  f"ess={health['ess_fraction']:.3f} "
                  f"{out['wall_seconds']:5.0f}s"
                  f"{'   *** KERNEL COLLAPSED ***' if dead else ''}",
                  flush=True)

    seeds = sorted({r["seed"] for r in rows})
    table = {(r["arm"], r["seed"]): r for r in rows}
    summary = {}
    for label, ess, levels, positives in ARMS:
        group = [r for r in rows if r["arm"] == label]
        summary[label] = {
            "target_ess": ess, "levels": levels, "positives": positives,
            **{k: float(np.median([r[k] for r in group]))
               for k in ("kid", "fid_512", "fid_2048", "tail",
                         "second_moment", "ed2", "wall_seconds")},
            "realized_ess": float(np.median(
                [r["realized_ess"]["ess_fraction"] for r in group])),
            "realized_images": float(np.median(
                [r["realized_ess"]["ess_count"] for r in group])),
        }

    contrasts = {}
    for name, a, b in CONTRASTS:
        contrasts[name] = {key: paired(table, seeds, key, a, b)
                           for key in ("kid", "fid_2048")}
    best = min(summary, key=lambda k: summary[k]["kid"])
    contrasts["best_vs_baseline"] = {
        key: paired(table, seeds, key, BASELINE, best)
        for key in ("kid", "fid_2048")}

    sharp = contrasts["sharp"]["kid"]
    added = any(contrasts[n]["kid"]["significant"]
                and contrasts[n]["kid"]["mean"] < 0
                for n, _, _ in CONTRASTS if n != "sharp")
    verdict = {
        "floor": floor, "bar": bar, "seeds": seeds, "steps": args.steps,
        "best_arm": best, "best_kid": summary[best]["kid"],
        "baseline_kid": summary[BASELINE]["kid"],
        "sharpening_replicates": bool(sharp["significant"]
                                      and sharp["mean"] < 0),
        "an_axis_adds": bool(added),
        "contrasts": contrasts,
    }
    verdict["reading"] = {
        (True, True): "sharpening replicates AND an axis adds -- the "
                      "mechanism has headroom; take the winner to a scaling "
                      "run",
        (True, False): "sharpening replicates, nothing adds -- ESS was the "
                       "whole lever and distance concentration bounds the "
                       "method",
        (False, True): "sharpening does not replicate but another axis "
                       "moves -- Phase 21 was budget-specific",
        (False, False): "nothing resolves at 30k -- the configuration work "
                        "is finished; write up the mechanism result",
    }[(verdict["sharpening_replicates"], bool(added))]

    payload = {"status": "phase22-frozen-protocol",
               "protocol": "numerics/EncoderIndependentPhase22Protocol.md",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "config_digest": config_digest(
                   TrainConfig(image_size=args.resolution)),
               "elapsed_seconds": time.time() - started,
               "floor": floor, "bar": bar,
               "summary": summary, "verdict": verdict, "rows": rows}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 22 ===")
    print(f"{'arm':18}{'ESS':>6}{'lv':>4}{'pos':>5}{'KID':>10}"
          f"{'FID2048':>10}{'FID512':>9}{'tail':>8}{'realized':>10}"
          f"{'imgs':>7}")
    for label, _, _, _ in ARMS:
        e = summary[label]
        print(f"{label:18}{e['target_ess']:6.3f}{e['levels']:4}"
              f"{e['positives']:5}{e['kid']:+10.5f}{e['fid_2048']:10.2f}"
              f"{e['fid_512']:9.2f}{e['tail']:8.4f}{e['realized_ess']:10.3f}"
              f"{e['realized_images']:7.1f}")
    print(f"\n    floor KID {floor['kid']:+.5f} / FID2048 "
          f"{floor['fid_2048']:.2f}    BAR KID {bar['kid']:+.5f} / "
          f"FID2048 {bar['fid_2048']:.2f}    real tail {floor['tail']:.4f}")
    print("\n    paired contrasts on KID (negative = better)")
    for name in list(dict.fromkeys([n for n, _, _ in CONTRASTS]
                                   + ["best_vs_baseline"])):
        e = contrasts[name]["kid"]
        mark = "p<0.05" if e["significant"] else "n.s."
        print(f"      {name:22}{e['mean']:+10.5f} +- {e['sem']:.5f}"
              f"   t={e['t']:+6.2f}  {mark}")
    print(f"\n    best arm: {best} at KID {summary[best]['kid']:+.5f} "
          f"(baseline {summary[BASELINE]['kid']:+.5f})")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
