"""Phase 17: the encoder ladder, at a budget where FID discriminates.

Phase 14A ran this comparison at 600 steps and could not interpret it: every
arm sat at FID 255-302, where a moment-matched Gaussian also sits (254.8), so
the ladder was ranking geometries by how well they match moments -- which is
not what an encoder is for.

Phase 16 then found the budget is the binding constraint.  Raw pixels at
30 000 steps reach **FID 232**, below the Gaussian bar and still falling,
while ED2 moves the *opposite* way.  So the ladder is worth running again at
that budget, and only there.

Three geometries, the ones that carry the thesis:

  raw          encoder-free;
  random       untrained ResNet18 -- same architecture, no pretraining;
  pretrained   ImageNet ResNet18 -- a real semantic encoder.

`random` is the highest-information arm: if it matches `pretrained`, the
paper's "semantic encoder" story does not survive and encoder-independence is
nearly free.  If `pretrained` wins decisively, the thing Branch B failed to
replace has been quantified.

**FID is the outcome.  ED2, the second moment and the spectral tail are
diagnostics and cannot declare success** -- Phase 16 showed FID and ED2
disagree on the sign of the budget effect.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase17
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from .config import MASTER_SEED, TrainConfig, config_digest, derive_seed
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnostics import provenance, write_json
from .encoders import ENCODER_INPUT
from .run_phase16 import EVAL_SAMPLES, score, train_arm

HERE = Path(__file__).resolve().parent

SEED_OFFSET = 29000
BUDGET = 30000
WIDTH = 64
CLOUD = 256

# (label, kind, note)
GEOMETRIES = (
    ("raw", "raw", "encoder-free"),
    ("random_resnet", "random",
     "untrained ResNet18 -- separates pretraining from architecture"),
    ("pretrained_resnet", "pretrained",
     "ImageNet ResNet18 (NOT the paper's self-supervised encoder)"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=BUDGET)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase17.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(MASTER_SEED + SEED_OFFSET, "p17"))
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
        print(f"    {name:26} FID={entry['fid']:8.2f}", flush=True)
    bar = baselines["moment_matched_gaussian"]["fid"]
    floor = baselines["real"]["fid"]
    print(f"\n    floor {floor:.2f}   BAR {bar:.2f}\n", flush=True)

    rows = []
    print(f"=== the ladder at {args.steps} steps "
          f"(encoder input {ENCODER_INPUT}px) ===", flush=True)
    for label, kind, note in GEOMETRIES:
        for index in range(args.seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            out = train_arm(kind, WIDTH, CLOUD, args.steps, seed, device,
                            args.resolution, args.data_root)
            row = {"arm": label, "kind": kind, "note": note, "seed": seed,
                   "steps": args.steps, "wall_seconds": out["wall_seconds"],
                   **score(out["sample"], real_a, device)}
            row["beats_bar"] = bool(row["fid"] < bar)
            rows.append(row)
            print(f"    {label:20} seed{index} FID={row['fid']:8.2f} "
                  f"({'BEATS' if row['beats_bar'] else 'above'} bar)  "
                  f"ED2={row['ed2']:8.4f} tail={row['tail']:7.4f} "
                  f"2nd={row['second_moment']:6.3f} "
                  f"{row['wall_seconds']:6.0f}s", flush=True)

    summary = {}
    for label, kind, note in GEOMETRIES:
        group = [r for r in rows if r["arm"] == label]
        summary[label] = {
            "kind": kind, "note": note,
            **{k: float(np.median([r[k] for r in group]))
               for k in ("fid", "ed2", "tail", "second_moment",
                         "wall_seconds")},
            "beats_bar": all(r["beats_bar"] for r in group)}

    by_fid = sorted(summary, key=lambda k: summary[k]["fid"])
    by_ed2 = sorted(summary, key=lambda k: summary[k]["ed2"])
    verdict = {
        "bar": bar, "floor": floor,
        "ranking_by_fid": by_fid, "ranking_by_ed2": by_ed2,
        "metrics_agree_on_best": by_fid[0] == by_ed2[0],
        "pretrained_beats_raw": (summary["pretrained_resnet"]["fid"]
                                 < summary["raw"]["fid"]),
        "pretrained_beats_random": (summary["pretrained_resnet"]["fid"]
                                    < summary["random_resnet"]["fid"]),
        "encoder_gap_fid": (summary["raw"]["fid"]
                            - summary["pretrained_resnet"]["fid"]),
    }
    verdict["reading"] = (
        "a pretrained encoder helps: the geometry the paper depends on is "
        "doing measurable work here"
        if verdict["pretrained_beats_raw"] else
        "a real pretrained encoder does NOT beat raw pixels at this scale -- "
        "encoder-free drifting is not paying a penalty in this harness")
    if verdict["pretrained_beats_raw"] and not verdict[
            "pretrained_beats_random"]:
        verdict["reading"] += ("; but it does not beat an UNTRAINED network "
                               "of the same architecture, so the benefit is "
                               "the feature map's structure and not its "
                               "pretraining")

    payload = {
        "status": "phase17-encoder-ladder-at-discriminating-budget",
        "provenance": provenance(), "device": settings,
        "config": vars(args) | {"out": str(args.out),
                                "encoder_input_px": ENCODER_INPUT,
                                "width": WIDTH, "cloud": CLOUD},
        "config_digest": config_digest(
            TrainConfig(image_size=args.resolution, width=WIDTH)),
        "elapsed_seconds": time.time() - started,
        "baselines": baselines, "summary": summary, "verdict": verdict,
        "rows": rows,
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 17 ===")
    print(f"{'geometry':22}{'FID':>9}{'ED2':>9}{'tail':>8}{'2nd':>7}"
          f"{'minutes':>9}  vs bar")
    for key in by_fid:
        e = summary[key]
        print(f"{key:22}{e['fid']:9.2f}{e['ed2']:9.4f}{e['tail']:8.4f}"
              f"{e['second_moment']:7.3f}{e['wall_seconds'] / 60:9.1f}  "
              f"{'BEATS' if e['beats_bar'] else 'above'}")
    print(f"\n    floor {floor:.2f}   BAR {bar:.2f}")
    print(f"    ranking by FID: {' < '.join(by_fid)}")
    print(f"    ranking by ED2: {' < '.join(by_ed2)}")
    print(f"    pretrained beats raw:    "
          f"{verdict['pretrained_beats_raw']}  "
          f"(gap {verdict['encoder_gap_fid']:+.2f} FID)")
    print(f"    pretrained beats random: "
          f"{verdict['pretrained_beats_random']}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
