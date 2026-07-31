"""Is the METRIC the bottleneck, and what do the samples actually look like?

Phase 19 failed to resolve any contrast: within-arm seed sd was 26.68 FID
against effects near 15.  That was read as training variance, but it was
never decomposed.  FID here is computed at 512 samples from 2048-dimensional
Inception features, so **both sample covariances have rank <= 511 and are
singular** -- which is why real-vs-real reads ~70 instead of 0 in every run
this program has done.  A biased estimator can also be a noisy one, and if a
large share of that 26.68 is measurement noise then raising the eval sample
count buys resolution for nothing.

Three questions, none of which needs a new training idea:

  M1  **The floor, versus sample count.**  FID and KID between two disjoint
      REAL subsets.  FID's floor should fall steeply with n (its bias is a
      finite-sample artifact); KID's should sit at ~0 at every n, because
      the U-statistic is unbiased.

  M2  **Measurement noise, versus sample count.**  One fixed trained
      generator, scored repeatedly against fresh subsets.  Every seed and
      every weight is held constant, so all the spread here is the metric.
      Compare the sd against Phase 19's 26.68.

  M3  **Look at the samples.**  Eighteen phases of statistics and nobody has
      rendered an image.  Saved as a PNG grid.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase20
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
from .diagnostics import provenance, write_json
from .fid import frechet_from_features, inception_features, kid_from_features
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher
from .run_phase19 import CLOUD, POSITIVES, WIDTH, learning_rate

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 33000
SIZES = (512, 1024, 2048)
REPEATS = 8


def train_model(ess: float, schedule: str, steps: int, seed: int, device,
                resolution: int, root: str | None) -> OneStepGenerator:
    """Arm-D training, returning the MODEL.

    This mirrors `run_phase19.train_arm` deliberately rather than importing
    it: that function returns only a 512-sample draw, and this probe needs
    thousands.  `run_phase19.py` is left untouched because its artifact is
    already sealed and its recorded provenance must keep matching the file.
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
    return model


def gpu_frechet(a: torch.Tensor, b: torch.Tensor) -> float:
    """FID from features, with the two matrix square roots done on GPU.

    Identical mathematics to `fid.frechet_from_features`; numpy's `eigh` on
    2048x2048 doubles costs seconds and this probe needs ~50 of them.
    Agreement with the production path is checked once and reported.
    """
    mu_a, mu_b = a.mean(0), b.mean(0)
    cov_a, cov_b = torch.cov(a.T), torch.cov(b.T)

    def root(matrix: torch.Tensor) -> torch.Tensor:
        values, vectors = torch.linalg.eigh(matrix)
        return (vectors * values.clamp_min(0).sqrt()) @ vectors.T

    root_a = root(cov_a)
    middle = root(root_a @ cov_b @ root_a)
    value = (((mu_a - mu_b) ** 2).sum()
             + cov_a.trace() + cov_b.trace() - 2.0 * middle.trace())
    return max(float(value), 0.0)


def curve(left_pool: torch.Tensor, right_pool: torch.Tensor, rng,
          label: str) -> dict:
    """FID and KID at each sample count, over independent draws.

    For the floor both sides come from the real pool and are drawn
    DISJOINTLY, so no shared sample can deflate the distance.
    """
    out = {}
    same = left_pool is right_pool
    for n in SIZES:
        need = 2 * n if same else n
        if len(left_pool) < need or len(right_pool) < n:
            continue
        fids, kids = [], []
        for _ in range(REPEATS):
            if same:
                pick = rng.choice(len(left_pool), 2 * n, replace=False)
                left, right = left_pool[pick[:n]], left_pool[pick[n:]]
            else:
                left = left_pool[rng.choice(len(left_pool), n, replace=False)]
                right = right_pool[rng.choice(len(right_pool), n,
                                              replace=False)]
            fids.append(gpu_frechet(left, right))
            kids.append(kid_from_features(left.cpu().numpy(),
                                          right.cpu().numpy()))
        out[str(n)] = {
            "fid_mean": float(np.mean(fids)),
            "fid_sd": float(np.std(fids, ddof=1)),
            "kid_mean": float(np.mean(kids)),
            "kid_sd": float(np.std(kids, ddof=1)),
        }
        print(f"    {label:6} n={n:5}  FID {np.mean(fids):7.2f} "
              f"+- {np.std(fids, ddof=1):5.2f}   "
              f"KID {np.mean(kids):+.5f} +- {np.std(kids, ddof=1):.5f}",
              flush=True)
    return out


def save_grid(images: torch.Tensor, path: Path, rows: int = 8) -> None:
    from torchvision.utils import make_grid, save_image  # noqa: PLC0415
    grid = make_grid((images[:rows * rows].detach().cpu() + 1.0) * 0.5,
                     nrow=rows, padding=2)
    save_image(grid.clamp(0.0, 1.0), path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--pool", type=int, default=8192)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "phase20_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    seed = MASTER_SEED + SEED_OFFSET

    started = time.time()
    print(f"=== training one generator (arm D config), {args.steps} steps ===",
          flush=True)
    began = time.time()
    model = train_model(0.5, "cosine", args.steps, seed, device,
                        args.resolution, args.data_root)
    print(f"    {time.time() - began:.0f}s", flush=True)

    latent = sample_latent(args.pool, TrainConfig().latent_dim,
                           derive_seed(seed, "pool"), device)
    with torch.no_grad():
        generated = torch.cat([model(latent[i:i + 512])
                               for i in range(0, len(latent), 512)])
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(seed, "p20"))
    real = evaluation.sample(args.pool, rng)

    print("\n=== M3: rendering samples ===", flush=True)
    save_grid(generated, HERE / "phase20_samples_generated.png")
    save_grid(real, HERE / "phase20_samples_real.png")
    print("    wrote phase20_samples_generated.png / _real.png", flush=True)

    print("\n=== Inception features ===", flush=True)
    began = time.time()
    real_features = inception_features(real, device).double().to(device)
    gen_features = inception_features(generated, device).double().to(device)
    print(f"    real {tuple(real_features.shape)} "
          f"gen {tuple(gen_features.shape)}  {time.time() - began:.0f}s",
          flush=True)

    fast = gpu_frechet(real_features[:512], real_features[512:1024])
    slow = frechet_from_features(real_features[:512].cpu().numpy(),
                                 real_features[512:1024].cpu().numpy())
    agreement = abs(fast - slow) / max(slow, 1e-12)
    print(f"\n    gpu/numpy FID agreement: {fast:.4f} vs {slow:.4f} "
          f"(rel {agreement:.2e})", flush=True)

    print("\n=== M1: the floor, real vs real (disjoint) ===", flush=True)
    floor = curve(real_features, real_features, rng, "floor")
    print("\n=== M2: measurement noise, ONE fixed generator ===", flush=True)
    measured = curve(gen_features, real_features, rng, "gen")

    sd512 = measured.get("512", {}).get("fid_sd")
    verdict = {
        "phase19_seed_sd": 26.68,
        "floor_sd_at_512": floor.get("512", {}).get("fid_sd"),
        "generator_measurement_sd_at_512": sd512,
        "measurement_share_of_phase19_spread": (
            None if sd512 is None else sd512 / 26.68),
        "gpu_numpy_relative_agreement": agreement,
    }
    payload = {"status": "phase20-probe-feeds-no-gate",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "floor_curve": floor, "generator_curve": measured,
               "verdict": verdict}
    digest = write_json(args.out, payload)

    print("\n=== READING ===")
    if sd512 is not None:
        print(f"    measurement sd at n=512 is {sd512:.2f} FID, against "
              f"Phase 19's within-arm seed sd of 26.68")
        for n in SIZES:
            if str(n) not in measured:
                continue
            e = measured[str(n)]
            # The floor needs 2n disjoint real samples, so it drops out at
            # sizes the generator curve still covers.
            f = floor.get(str(n))
            shown = f"{f['fid_mean']:6.2f}" if f else "    --"
            print(f"      n={n:5}  gen FID {e['fid_mean']:7.2f} "
                  f"+- {e['fid_sd']:5.2f}   floor {shown}"
                  f"   KID {e['kid_mean']:+.5f} +- {e['kid_sd']:.5f}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
