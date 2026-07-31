"""Phase 19 (protocol `EncoderIndependentPhase19Protocol.md`).

Two cheap fixes, screened before the long-budget run.

**F1 -- the bandwidth is set to the value our own audit called worse.**
`run_phase16.py` fixes `GOOD_ESS = 0.9` and Phases 16/17/18B inherit it, but
`EncoderIndependentMetricAudit.md` measured ESS 0.5 at FID 244.0 against
0.9's 258.8 with R11 (260.3 vs 266.8 without).  The program sits at 0.9
because Phase 7 found it 4.9x better on ED2 -- the metric the same audit
showed a structureless Gaussian saturates.

**F2 -- the recipe has no annealing path.**  The drift is RMS-normalized, so
the teacher stays a constant distance from the output however close the cloud
gets; `step_eta` is inert under Adam (R24); and the learning rate is a
constant 2e-3 with no scheduler and no weight EMA anywhere in the package.
The generator therefore takes constant-magnitude steps forever and has no
fixed point to reach -- it jitters around a statistical equilibrium, which is
what the shape law describes.

EMA does not change training, so one run scores both the live and the EMA
weights from an identical trajectory.  That makes it an *evaluation* factor,
perfectly paired, and leaves four training arms: {ESS 0.9, 0.5} x {constant,
cosine}.  4 seeds x 15 000 steps -- both raised deliberately after Phase 18B,
where 2 seeds at 5 000 steps gave >100 FID spreads and a control that
reversed sign.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase19
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from . import kernel_gradient as KG
from .config import (
    MASTER_SEED,
    GeometryConfig,
    TrainConfig,
    config_digest,
    derive_seed,
)
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnostics import provenance, write_json
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher
from .run_phase16 import EVAL_SAMPLES, score

HERE = Path(__file__).resolve().parent

SEED_OFFSET = 32000          # fresh: 28000/30000/31000 are already spent
POSITIVES = 64
CLOUD = 256
WIDTH = 64

# (label, target ESS, lr schedule)
ARMS = (("A_ess9_const", 0.9, "constant"),
        ("B_ess5_const", 0.5, "constant"),
        ("C_ess9_cos", 0.9, "cosine"),
        ("D_ess5_cos", 0.5, "cosine"))

# `ema999` is the pre-registered primary (protocol section 3).  `ema9999` is
# free to compute and is reported for information only -- the protocol
# forbids selecting the long-run configuration on it.
EMA_DECAYS = {"ema999": 0.999, "ema9999": 0.9999}
PRIMARY_EMA = "ema999"
WEIGHT_SETS = ("live", *EMA_DECAYS)

BASELINE = ("A_ess9_const", "live")   # the as-is configuration


class Ema:
    """Shadow weights, updated alongside training and never fed back.

    Warmup ``d_t = min(d, (1+t)/(10+t))`` keeps the average from being pinned
    to the random initialization over the first steps.  Declared, not tuned.
    """

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.steps = 0
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model: torch.nn.Module) -> None:
        self.steps += 1
        d = min(self.decay, (1.0 + self.steps) / (10.0 + self.steps))
        with torch.no_grad():
            for key, value in model.state_dict().items():
                if key in self.shadow:
                    self.shadow[key].mul_(d).add_(value.detach().float(),
                                                  alpha=1.0 - d)

    def copy_into(self, model: torch.nn.Module) -> None:
        state = model.state_dict()
        with torch.no_grad():
            for key, value in self.shadow.items():
                state[key].copy_(value.to(state[key].dtype))


def learning_rate(base: float, schedule: str, step: int, total: int) -> float:
    if schedule == "constant":
        return base
    if schedule == "cosine":
        return base * 0.5 * (1.0 + math.cos(math.pi * step / max(total, 1)))
    raise ValueError(f"unknown schedule {schedule!r}")


def train_arm(ess: float, schedule: str, steps: int, seed: int, device,
              resolution: int, root: str | None) -> dict:
    """One training run; returns a sample per weight set.

    Every stochastic stream is derived from the seed ALONE -- not from the
    arm -- so all four arms share a seed's generator initialization, latent
    draws and target order.  That is what makes the comparisons paired.
    """
    train = cifar.cifar_target(resolution, "train", root)
    train.device = device
    rng = np.random.default_rng(derive_seed(seed, "p19"))
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
    emas = {name: Ema(model, decay) for name, decay in EMA_DECAYS.items()}

    started = time.time()
    for step in range(steps):
        lr = learning_rate(config.learning_rate, schedule, step, steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        latent = sample_latent(CLOUD, config.latent_dim,
                               derive_seed(seed, "latent", step), device)
        output = model(latent)
        with torch.no_grad():
            positives = train.sample(POSITIVES, rng)
            drift, _ = KG.field(output.detach(), positives, output.detach(),
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            teacher = corrected_teacher(output.detach() + 0.5 * drift,
                                        positives, mode="scalar")
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for ema in emas.values():
            ema.update(model)
    wall = time.time() - started

    probe = sample_latent(EVAL_SAMPLES, config.latent_dim,
                          derive_seed(seed, "probe"), device)
    samples = {}
    with torch.no_grad():
        samples["live"] = model(probe)
        shadow = OneStepGenerator(config.latent_dim, 3, resolution, WIDTH,
                                  derive_seed(seed, "generator")).to(device)
        for name, ema in emas.items():
            ema.copy_into(shadow)
            samples[name] = shadow(probe)
    return {"samples": samples, "wall_seconds": wall,
            "final_lr": lr, "bandwidth_median": float(kernel.taus.median())}


def paired(table: dict, seeds: list[int], a: tuple, b: tuple) -> dict:
    """Paired difference b - a, with the sign test the protocol declares."""
    diffs = [table[(b[0], b[1], s)] - table[(a[0], a[1], s)] for s in seeds]
    signs = {d < 0 for d in diffs}
    return {"mean": float(np.mean(diffs)), "per_seed": [float(d) for d in diffs],
            "spread": float(max(diffs) - min(diffs)),
            "sign_consistent": len(signs) == 1,
            "improves": bool(np.mean(diffs) < 0 and len(signs) == 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase19.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(MASTER_SEED + SEED_OFFSET, "p19"))
    real_a = evaluation.sample(EVAL_SAMPLES, rng)
    real_b = evaluation.sample(EVAL_SAMPLES, rng)
    floor = score(real_b, real_a, device)["fid"]
    bar = score(gaussian_moment_match(real_a, EVAL_SAMPLES, rng), real_a,
                device)["fid"]
    print(f"    floor (real) {floor:.2f}    BAR (gaussian) {bar:.2f}\n",
          flush=True)

    rows = []
    for label, ess, schedule in ARMS:
        for index in range(args.seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            out = train_arm(ess, schedule, args.steps, seed, device,
                            args.resolution, args.data_root)
            scored = {name: score(sample, real_a, device)
                      for name, sample in out["samples"].items()}
            rows.append({"arm": label, "ess": ess, "schedule": schedule,
                         "seed": seed, "wall_seconds": out["wall_seconds"],
                         "bandwidth_median": out["bandwidth_median"],
                         "scores": scored})
            live, primary = scored["live"], scored[PRIMARY_EMA]
            print(f"    {label:14} seed{index} "
                  f"live={live['fid']:8.2f} {PRIMARY_EMA}={primary['fid']:8.2f}"
                  f"  ed2={live['ed2']:7.4f} tail={live['tail']:6.4f}"
                  f" {out['wall_seconds']:6.0f}s", flush=True)

    seeds = sorted({r["seed"] for r in rows})
    table = {(r["arm"], w, r["seed"]): r["scores"][w]["fid"]
             for r in rows for w in WEIGHT_SETS}
    summary = {}
    for label, ess, schedule in ARMS:
        group = [r for r in rows if r["arm"] == label]
        summary[label] = {"ess": ess, "schedule": schedule, **{
            w: {k: float(np.median([r["scores"][w][k] for r in group]))
                for k in ("fid", "ed2", "tail", "second_moment")}
            for w in WEIGHT_SETS}}

    contrasts = {
        # F1: the bandwidth, at each schedule, on live weights
        "ess_at_constant": paired(table, seeds, ("A_ess9_const", "live"),
                                  ("B_ess5_const", "live")),
        "ess_at_cosine": paired(table, seeds, ("C_ess9_cos", "live"),
                                ("D_ess5_cos", "live")),
        # F2a: the schedule, at each bandwidth, on live weights
        "cosine_at_ess9": paired(table, seeds, ("A_ess9_const", "live"),
                                 ("C_ess9_cos", "live")),
        "cosine_at_ess5": paired(table, seeds, ("B_ess5_const", "live"),
                                 ("D_ess5_cos", "live")),
    }
    # F2b: EMA, within every arm (same trajectory, so exactly paired)
    for label, _, _ in ARMS:
        contrasts[f"ema_in_{label}"] = paired(
            table, seeds, (label, "live"), (label, PRIMARY_EMA))

    cells = [(a, w) for a, _, _ in ARMS for w in ("live", PRIMARY_EMA)]
    best = min(cells, key=lambda c: float(np.median(
        [table[(c[0], c[1], s)] for s in seeds])))
    contrasts["best_vs_baseline"] = paired(table, seeds, BASELINE, best)

    bandwidth = (contrasts["ess_at_constant"]["improves"]
                 or contrasts["ess_at_cosine"]["improves"])
    annealing = any(contrasts[k]["improves"] for k in contrasts
                    if k.startswith(("ema_in_", "cosine_at_")))
    verdict = {
        "floor": floor, "bar": bar, "seeds": seeds, "steps": args.steps,
        "baseline_fid": float(np.median(
            [table[(*BASELINE, s)] for s in seeds])),
        "best_cell": {"arm": best[0], "weights": best[1],
                      "fid": float(np.median(
                          [table[(best[0], best[1], s)] for s in seeds]))},
        "bandwidth_helps": bool(bandwidth),
        "annealing_helps": bool(annealing),
        "contrasts": contrasts,
    }
    verdict["reading"] = {
        (True, True): "both fixes help -- the 232 figure measured a "
                      "misconfigured system; long run goes to the best cell",
        (True, False): "bandwidth helps, annealing does not -- the audit's "
                       "ESS finding survives; long run at ESS 0.5",
        (False, True): "annealing helps, bandwidth does not -- the ESS-0.5 "
                       "advantage was specific to the audit's short budget",
        (False, False): "neither helps -- the recipe is at its ceiling here; "
                        "run the long budget as-is and write up",
    }[(bool(bandwidth), bool(annealing))]

    payload = {"status": "phase19-frozen-protocol",
               "protocol": "numerics/EncoderIndependentPhase19Protocol.md",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "config_digest": config_digest(
                   TrainConfig(image_size=args.resolution)),
               "elapsed_seconds": time.time() - started,
               "floor": floor, "bar": bar,
               "summary": summary, "verdict": verdict, "rows": rows}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 19 ===")
    print(f"{'arm':16}{'ess':>5}{'sched':>10}"
          f"{'live':>10}{'ema999':>10}{'ema9999':>10}")
    for label, _, _ in ARMS:
        e = summary[label]
        print(f"{label:16}{e['ess']:5.1f}{e['schedule']:>10}"
              f"{e['live']['fid']:10.2f}{e['ema999']['fid']:10.2f}"
              f"{e['ema9999']['fid']:10.2f}")
    print(f"\n    floor {floor:.2f}    BAR {bar:.2f}    "
          f"baseline (as-is) {verdict['baseline_fid']:.2f}")
    print("\n    paired contrasts (negative = better; 4/4 sign needed)")
    for name, entry in contrasts.items():
        mark = "REAL" if entry["improves"] else (
            "worse" if entry["sign_consistent"] else "unresolved")
        print(f"      {name:22}{entry['mean']:+9.2f}  "
              f"spread {entry['spread']:6.1f}  {mark}")
    print(f"\n    best cell: {verdict['best_cell']['arm']} / "
          f"{verdict['best_cell']['weights']} at "
          f"FID {verdict['best_cell']['fid']:.2f}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
