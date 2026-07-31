"""Phase 14A (plan `EncoderIndependentPhase14Plan.md`).

Does this harness reproduce the paper's finding that quality tracks encoder
quality?

This is a **validity check and a precondition**, not the comparison.  The
program's only encoder-vs-encoder-free measurement (Phase 1) has raw pixels
beating learned geometry ~5x, while the paper's central ablation has FID
tracking encoder quality closely.  Those point opposite ways, and nothing in
thirteen phases distinguishes "semantic geometry genuinely does not help
here" from "the harness cannot see what the encoder contributes".  Until it
does, the comparison would produce an uninterpretable number.

Five geometries, one recipe, everything else fixed.  Each gets its OWN
target-ESS calibration -- Phases 2-6 compared geometries at an uncalibrated
bandwidth and Phase 7 later showed bandwidth is worth 4.9x, so not repeating
that is the single most important design constraint here.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.run_phase14a
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
from .diagnostics import kernel_admissible, provenance, write_json
from .encoders import encoder_branch
from .evaluate import evaluation_pools, null_reference
from .fid import frechet_distance
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher
from .reference_encoder import encoder_family, train_reference_encoder

HERE = Path(__file__).resolve().parent

SEED_OFFSET = 26000
GOOD_ESS = 0.9
FIELD_CLOUD = 256
POSITIVES = 64
TEACHER_ETA = 0.5

# (label, kind, note)
GEOMETRIES = (
    ("G1_pretrained_resnet", "pretrained",
     "ImageNet ResNet18 -- a real semantic encoder (NOT the paper's DINO)"),
    ("G2_random_resnet", "random",
     "same architecture, untrained -- separates pretraining from depth"),
    ("G3_autoencoder", "autoencoder",
     "the Phase-1 A8 stand-in, for continuity with the old record"),
    ("G4_raw_pixels", "raw", "encoder-free"),
    ("G5_degraded", "degraded", "the bad end of the ladder"),
)


def _branch_for(kind: str, seed: int, device, resolution: int,
                train, rng):
    """One geometry, plus whether it is a learned map."""
    if kind == "raw":
        geometry = GeometryConfig(family="raw",
                                  base_kernel="smooth_laplace",
                                  target_ess_fraction=GOOD_ESS)
        return build_family(geometry, 3).branches[0], False
    if kind == "autoencoder":
        # Trained here, not pretrained -- and it sees target data, so if
        # anything it is *favoured* against a genuinely external encoder.
        # Carried for continuity with the Phase-1 A8 row.
        geometry = GeometryConfig(family="reference_encoder",
                                  base_kernel="smooth_laplace",
                                  target_ess_fraction=GOOD_ESS)
        model = train_reference_encoder(
            train.sample, 3, 32, derive_seed(seed, "refenc"),
            steps=400, batch=64, learning_rate=2e-3, rng=rng).to(device)
        return encoder_family(model, geometry).branches[0], True
    return encoder_branch(kind, seed=seed, device=device), True


def run_geometry(label: str, kind: str, seed: int, device, resolution: int,
                 steps: int, root: str | None, r11: bool) -> dict:
    train = cifar.cifar_target(resolution, "train", root)
    evaluation = cifar.cifar_target(resolution, "eval", root)
    # Training runs on the device; the evaluation pools stay on the CPU
    # because the metric suite is numpy-backed.  FID moves what it needs.
    train.device = device
    rng = np.random.default_rng(derive_seed(seed, "p14a", kind))
    config = TrainConfig(steps=steps, batch=POSITIVES, eval_samples=512,
                         image_size=resolution)
    pools = evaluation_pools(evaluation, config, seed)
    null = null_reference(evaluation, pools, seed)

    branch, learned = _branch_for(kind, seed, device, resolution, train, rng)
    # Each geometry gets its OWN bandwidth solved to the same target ESS.
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=GOOD_ESS)

    model = OneStepGenerator(config.latent_dim, 3, resolution, config.width,
                             derive_seed(seed, "generator")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    health: dict = {}
    for step in range(steps):
        latent = sample_latent(FIELD_CLOUD, config.latent_dim,
                               derive_seed(seed, "latent", step), device)
        output = model(latent)
        with torch.no_grad():
            positives = train.sample(POSITIVES, rng)
            drift, stats = KG.field(
                output.detach(), positives, output.detach(), branch, kernel,
                direction_mode="paper", normalization="rms",
                diagnostics=(step == 0))
            if step == 0:
                health = stats
            teacher = output.detach() + TEACHER_ETA * drift
            if r11:
                teacher = corrected_teacher(teacher, positives, mode="scalar")
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    probe = sample_latent(config.eval_samples, config.latent_dim,
                          derive_seed(seed, "probe"), device)
    with torch.no_grad():
        generated = model(probe)
    measured = M.raw_metrics(
        generated.cpu(), pools["eval"].cpu(), pools["cal_a"].cpu(),
        pools["cal_b"].cpu(),
        np.random.default_rng(derive_seed(seed, "p14a-m")), None,
        target_null=pools["null"].cpu())
    fid = frechet_distance(generated, pools["eval"], device)
    admissible = kernel_admissible(health, FIELD_CLOUD)
    return {
        "arm": label, "kind": kind, "seed": seed, "r11": r11,
        "learned_geometry": learned,
        "ed2": measured["ed2"],
        "geometry_score_v2": M.normalized_geometry_score_v2(
            measured, null)["geometry_score"],
        "fid": fid["fid"], "fid_samples": fid["samples_generated"],
        "second_moment_ratio": float(
            generated.flatten(1).var(0).mean()
            / pools["eval"].flatten(1).var(0).mean()),
        "ess_fraction": float(health.get("ess_fraction", float("nan"))),
        "collapsed_row_fraction": float(
            health.get("collapsed_row_fraction", float("nan"))),
        "admissible": admissible["admissible"],
        "admissibility_reasons": admissible.get("reasons", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--r11", action="store_true",
                        help="carry R11 on every arm (14B uses this)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase14a.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)

    started = time.time()
    print(f"=== Phase 14A: the encoder-quality ladder on {device} ===",
          flush=True)
    rows = []
    for label, kind, note in GEOMETRIES:
        for index in range(args.seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            row = run_geometry(label, kind, seed, device, args.resolution,
                               args.steps, args.data_root, args.r11)
            row["note"] = note
            rows.append(row)
            print(f"    {label:22} seed{index} "
                  f"fid={row['fid']:8.2f} ed2={row['ed2']:8.4f} "
                  f"score={row['geometry_score_v2']:7.3f} "
                  f"ESS={row['ess_fraction']:6.3f} "
                  f"{'OK ' if row['admissible'] else 'DEAD'}", flush=True)

    summary = {}
    for label, kind, note in GEOMETRIES:
        group = [r for r in rows if r["arm"] == label]
        summary[label] = {
            "kind": kind, "note": note,
            "learned_geometry": group[0]["learned_geometry"],
            "median_fid": float(np.median([r["fid"] for r in group])),
            "median_ed2": float(np.median([r["ed2"] for r in group])),
            "median_score_v2": float(np.median(
                [r["geometry_score_v2"] for r in group])),
            "median_second_moment": float(np.median(
                [r["second_moment_ratio"] for r in group])),
            "median_ess": float(np.median([r["ess_fraction"]
                                           for r in group])),
            "admissible": all(r["admissible"] for r in group)}

    scored = {k: v for k, v in summary.items() if v["admissible"]}
    by_fid = sorted(scored, key=lambda k: scored[k]["median_fid"])
    by_ed2 = sorted(scored, key=lambda k: scored[k]["median_ed2"])
    pretrained = scored.get("G1_pretrained_resnet")
    random_arm = scored.get("G2_random_resnet")
    raw = scored.get("G4_raw_pixels")
    verdict = {
        "ranking_by_fid": by_fid, "ranking_by_ed2": by_ed2,
        "metrics_agree": by_fid[:1] == by_ed2[:1] if scored else None,
        "pretrained_beats_random_fid": (
            pretrained["median_fid"] < random_arm["median_fid"]
            if pretrained and random_arm else None),
        "pretrained_beats_raw_fid": (
            pretrained["median_fid"] < raw["median_fid"]
            if pretrained and raw else None),
        "raw_is_best_fid": bool(by_fid and by_fid[0] == "G4_raw_pixels"),
    }
    verdict["reading"] = (
        "the harness sees encoder quality; the comparison in 14B is "
        "meaningful" if verdict["pretrained_beats_raw_fid"]
        else "raw pixels match or beat a real pretrained encoder here -- "
             "either semantic geometry does not help at this scale, or the "
             "harness is blind to it; 14B cannot be interpreted until this "
             "is resolved")

    payload = {
        "status": "phase14a-validity-check",
        "plan": "numerics/EncoderIndependentPhase14Plan.md",
        "provenance": provenance(), "device": settings,
        "config": vars(args) | {"out": str(args.out)},
        "config_digest": config_digest(
            TrainConfig(steps=args.steps, image_size=args.resolution)),
        "elapsed_seconds": time.time() - started,
        "summary": summary, "verdict": verdict, "rows": rows,
    }
    digest = write_json(args.out, payload)

    print("\n=== PHASE 14A ===")
    print(f"{'geometry':24}{'learned':>9}{'FID':>10}{'ed2':>10}{'score':>8}"
          f"{'2nd':>8}{'ESS':>8}  health")
    for key in sorted(summary, key=lambda k: summary[k]["median_fid"]):
        e = summary[key]
        print(f"{key:24}{str(e['learned_geometry']):>9}"
              f"{e['median_fid']:10.2f}{e['median_ed2']:10.4f}"
              f"{e['median_score_v2']:8.3f}{e['median_second_moment']:8.3f}"
              f"{e['median_ess']:8.3f}  "
              f"{'OK' if e['admissible'] else 'DEAD'}")
    print(f"\n  ranking by FID: {' < '.join(by_fid)}")
    print(f"  ranking by ED2: {' < '.join(by_ed2)}")
    print(f"  pretrained beats random: "
          f"{verdict['pretrained_beats_random_fid']}")
    print(f"  pretrained beats raw:    "
          f"{verdict['pretrained_beats_raw_fid']}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
