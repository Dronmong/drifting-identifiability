"""The autoencoder ceiling: what latent drifting could possibly achieve.

Latent-space drifting cannot beat its own decoder.  `decode(encode(real))` is
therefore a hard upper bound on the entire approach, and measuring it costs
minutes where the alternative is discovering the bound after an overnight run.
Phase 22 already spent 7.31 h establishing a ceiling the arithmetic could have
predicted; this probe is the correction to that habit.

Two things are being decided here, and they pull against each other:

  a bigger latent reconstructs better, raising the ceiling;
  a bigger latent is a harder kernel problem, which is the whole reason for
  leaving pixel space -- 3072 dimensions with 256 particles is where Phase 22's
  two walls came from.

So the ceiling is measured at three latent sizes (128 / 256 / 512 dimensions)
and the operating point is chosen knowing both sides, rather than by assuming
256 is fine.  Convergence of the reconstruction fit is reported as evidence
rather than asserted.

Judged on the measures validated in `diagnose_phase23.py`: KID and FID2048 for
distribution agreement, and **precision/recall**, since Phase 22 showed KID and
FID can rank garish texture above blurry photographs and an autoencoder's
failure mode is exactly blur.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase24
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from . import cifar
from .appearance import precision_recall, spectrum_slope
from .autoencoder import latent_dimension, latent_statistics, train_autoencoder
from .config import MASTER_SEED, derive_seed
from .device import configure, resolve_device
from .diagnose_phase15 import gaussian_moment_match
from .diagnose_phase20 import save_grid
from .diagnostics import provenance, write_json
from .fid import frechet_from_features, inception_features, kid_from_features
from .run_phase16 import _tail

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 37000
LATENT_CHANNELS = (8, 16, 32)


def score(images: torch.Tensor, reference: np.ndarray, real: torch.Tensor,
          device) -> dict:
    features = inception_features(images, device).double().numpy()
    return {"kid": kid_from_features(features, reference),
            "fid_2048": frechet_from_features(features, reference),
            **precision_recall(features, reference),
            "alpha": spectrum_slope(images)["alpha"],
            "tail": _tail(images.cpu()),
            "second_moment": float(
                images.flatten(1).var(0).mean().cpu()
                / real.flatten(1).var(0).mean().cpu())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--ae-steps", type=int, default=8000)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--channels", type=str,
                        default=",".join(str(c) for c in LATENT_CHANNELS))
    parser.add_argument("--out", type=Path, default=HERE / "phase24_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    channels = [int(c) for c in args.channels.split(",")]

    started = time.time()
    train = cifar.cifar_target(args.resolution, "train", args.data_root)
    train.device = device
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    seed = MASTER_SEED + SEED_OFFSET
    rng = np.random.default_rng(derive_seed(seed, "p24"))
    real_a = evaluation.sample(args.samples, rng)
    real_b = evaluation.sample(args.samples, rng)
    reference = inception_features(real_a, device).double().numpy()

    print("=== reference points, on the validated measures ===", flush=True)
    baselines = {
        "real": score(real_b, reference, real_a, device),
        "gaussian_mm": score(
            gaussian_moment_match(real_a, args.samples, rng), reference,
            real_a, device),
    }
    for name, entry in baselines.items():
        print(f"    {name:14} KID={entry['kid']:+.5f} "
              f"FID={entry['fid_2048']:7.2f} P={entry['precision']:.3f} "
              f"R={entry['recall']:.3f} alpha={entry['alpha']:.3f}", flush=True)
    save_grid(real_b[:64], HERE / "phase24_samples_real.png")

    rows = []
    print(f"\n=== autoencoder ceiling at {args.ae_steps} steps ===", flush=True)
    for latent_channels in channels:
        fit = train_autoencoder(train, args.ae_steps, seed, device,
                                image_size=args.resolution,
                                latent_channels=latent_channels,
                                report_every=max(args.ae_steps // 4, 1))
        model = fit["model"]
        with torch.no_grad():
            reconstructed = model(real_b.to(device)).cpu()
        entry = {"latent_channels": latent_channels,
                 "latent_dimension": latent_dimension(latent_channels),
                 "ae_parameters": fit["parameters"],
                 "ae_wall_seconds": fit["wall_seconds"],
                 "ae_final_mse": fit["history"][-1]["mse"],
                 "ae_history": fit["history"],
                 **latent_statistics(model, train, 512, rng),
                 **score(reconstructed, reference, real_a, device)}
        rows.append(entry)
        save_grid(reconstructed[:64],
                  HERE / f"phase24_samples_recon_d{entry['latent_dimension']}.png")
        print(f"    d={entry['latent_dimension']:4} KID={entry['kid']:+.5f} "
              f"FID={entry['fid_2048']:7.2f} P={entry['precision']:.3f} "
              f"R={entry['recall']:.3f} alpha={entry['alpha']:.3f} "
              f"mse={entry['ae_final_mse']:.5f}", flush=True)

    pixel_best_kid = 0.13116          # Phase 22 F_mix, the arm to beat
    by_dim = {r["latent_dimension"]: r for r in rows}
    verdict = {
        "real": baselines["real"], "gaussian_mm": baselines["gaussian_mm"],
        "pixel_space_best_kid": pixel_best_kid,
        "ceiling_by_dimension": {
            str(r["latent_dimension"]): {
                "kid": r["kid"], "fid_2048": r["fid_2048"],
                "precision": r["precision"], "recall": r["recall"]}
            for r in rows},
        "every_ceiling_beats_pixel_best": all(
            r["kid"] < pixel_best_kid for r in rows),
        "best_dimension": min(by_dim, key=lambda d: by_dim[d]["kid"]),
    }
    headroom = {str(d): pixel_best_kid - by_dim[d]["kid"] for d in by_dim}
    verdict["kid_headroom_vs_pixel_best"] = headroom
    verdict["reading"] = (
        "every autoencoder ceiling is below the best pixel-space arm, so "
        "latent drifting has room to be an improvement and Phase 25 is worth "
        "running"
        if verdict["every_ceiling_beats_pixel_best"] else
        "at least one ceiling is ABOVE the best pixel-space arm -- for those "
        "latent sizes the approach cannot help and must not be run")

    payload = {"status": "phase24-autoencoder-ceiling-probe",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "baselines": baselines, "rows": rows, "verdict": verdict}
    digest = write_json(args.out, payload)

    print("\n=== PHASE 24: THE CEILING ===")
    print(f"{'latent d':>10}{'KID':>10}{'FID2048':>10}{'prec':>8}{'recall':>8}"
          f"{'alpha':>8}{'tail':>8}{'ae mse':>9}")
    for row in rows:
        print(f"{row['latent_dimension']:10}{row['kid']:+10.5f}"
              f"{row['fid_2048']:10.2f}{row['precision']:8.3f}"
              f"{row['recall']:8.3f}{row['alpha']:8.3f}{row['tail']:8.4f}"
              f"{row['ae_final_mse']:9.5f}")
    real = baselines["real"]
    print(f"{'real':>10}{real['kid']:+10.5f}{real['fid_2048']:10.2f}"
          f"{real['precision']:8.3f}{real['recall']:8.3f}"
          f"{real['alpha']:8.3f}{real['tail']:8.4f}")
    print(f"\n    pixel-space best (Phase 22 F_mix) KID = {pixel_best_kid:+.5f}")
    print("    KID headroom, ceiling below that best:")
    for dimension, gap in sorted(headroom.items(), key=lambda kv: int(kv[0])):
        print(f"      d={dimension:>4}  {gap:+.5f}"
              f"{'  (ceiling is WORSE)' if gap <= 0 else ''}")
    print(f"\n  {verdict['reading']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
