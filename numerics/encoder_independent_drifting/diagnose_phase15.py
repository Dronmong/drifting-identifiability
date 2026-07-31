"""What does FID 280 mean, and do this program's reforms move it?

Phase 14A produced two results that reach backwards.  Every arm sits at FID
255-302 against a floor of 70.7 and a noise ceiling of 416.9; and ED2 and FID
disagree in rank order, most sharply on free particles -- best in the program
by energy distance (~0.07-0.2) and near-noise by FID (381).

Three questions, in order of how much they would change:

  E1  **Calibrate the scale.**  FID is only interpretable against known
      degradations.  Where does "FID 280" sit relative to real data that has
      been blurred, noised, downsampled, or replaced by a Gaussian with the
      data's exact mean and covariance?  That last one matters most: it
      matches every second-order statistic and carries no image content, so
      it is the direct probe of whether FID is measuring something ED2
      cannot see.

  E2  **Do the program's headline reforms move FID?**  R11 is worth 3.1x on
      ED2 and the ESS-0.9 bandwidth 4.9x.  If neither moves FID, thirteen
      phases improved a pixel statistic rather than a generative model, and
      that has to be known before anything is written up.

  E3  **Is FID at 512 samples discriminating or noisy?**  The covariance is
      2048x2048; at 512 samples it is rank-deficient and the estimate is
      biased.  Comparing the same arms at 512 and 2048 says whether the
      ordering is stable.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase15
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import cifar
from . import kernel_gradient as KG
from . import metrics as M
from .config import (
    GeometryConfig, MASTER_SEED, TrainConfig, derive_seed,
)
from .device import configure, resolve_device
from .diagnostics import provenance, write_json
from .fid import frechet_distance
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent
from .objectives import corrected_teacher

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 27000
POSITIVES = 64
FIELD_CLOUD = 256


def _tail(x: torch.Tensor, keep: int = 32) -> float:
    flat = x.reshape(len(x), -1)
    power = torch.linalg.svdvals(flat - flat.mean(dim=0, keepdim=True)) ** 2
    return float(power[keep:].sum() / power.sum())


def gaussian_moment_match(real: torch.Tensor, count: int,
                          rng: np.random.Generator) -> torch.Tensor:
    """A Gaussian with the data's exact mean and covariance.

    Matches every first- and second-order statistic of the real data and
    contains no image structure whatever.  If FID separates this from real
    data while ED2 does not, that is the clearest possible statement of what
    the program's metric has been unable to see.
    """
    flat = real.reshape(len(real), -1).double()
    mean = flat.mean(dim=0)
    centred = flat - mean
    # Sample in the data's own principal basis; with n < d the covariance is
    # rank-deficient, so this reproduces it exactly on its support.
    left, singular, right = torch.linalg.svd(centred, full_matrices=False)
    scale = singular / max(len(flat) - 1, 1) ** 0.5
    coefficients = torch.tensor(
        rng.normal(size=(count, len(scale))), dtype=torch.float64)
    samples = mean + (coefficients * scale) @ right
    return samples.reshape((count,) + real.shape[1:]).float()


def e1_scale(real_a: torch.Tensor, real_b: torch.Tensor, device,
             rng: np.random.Generator) -> dict:
    """Map the FID scale onto degradations we can describe in words."""
    variants: dict[str, torch.Tensor] = {
        "real (floor)": real_b,
        "gaussian_moment_match": gaussian_moment_match(real_a, len(real_a),
                                                       rng),
        "blur_2x": F.interpolate(F.avg_pool2d(real_b, 2), scale_factor=2,
                                 mode="nearest"),
        "blur_4x": F.interpolate(F.avg_pool2d(real_b, 4), scale_factor=4,
                                 mode="nearest"),
        "blur_8x": F.interpolate(F.avg_pool2d(real_b, 8), scale_factor=8,
                                 mode="nearest"),
        "noise_sigma_0.1": real_b + 0.1 * torch.randn(real_b.shape),
        "noise_sigma_0.3": real_b + 0.3 * torch.randn(real_b.shape),
        "noise_sigma_0.6": real_b + 0.6 * torch.randn(real_b.shape),
        "shuffled_pixels": real_b.reshape(len(real_b), 3, -1)[
            :, :, torch.randperm(real_b.shape[-1] ** 2)].reshape(
                real_b.shape),
        "pure_noise (ceiling)": torch.tensor(
            rng.normal(scale=0.5, size=tuple(real_b.shape)),
            dtype=torch.float32),
    }
    rows = []
    for name, sample in variants.items():
        fid = frechet_distance(sample, real_a, device)["fid"]
        ed2 = M.energy_distance2(sample, real_a)
        rows.append({"variant": name, "fid": fid, "ed2": ed2,
                     "tail": _tail(sample),
                     "second_moment": float(
                         sample.flatten(1).var(0).mean()
                         / real_a.flatten(1).var(0).mean())})
        print(f"    E1 {name:24} FID={fid:8.2f}  ED2={ed2:9.4f}  "
              f"2nd={rows[-1]['second_moment']:6.3f}", flush=True)
    return {"rows": rows}


def _train(kind: str, ess: float, r11: bool, device, resolution: int,
           steps: int, root: str | None, seed: int) -> torch.Tensor:
    train = cifar.cifar_target(resolution, "train", root)
    train.device = device
    rng = np.random.default_rng(derive_seed(seed, "p15", kind, ess, r11))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=ess)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace", 0.5, 1.0, 1e-3,
        combine="sum", target_ess_fraction=ess)
    config = TrainConfig(steps=steps, batch=POSITIVES, image_size=resolution)
    if kind == "particles":
        cloud = torch.tensor(
            rng.normal(scale=0.5,
                       size=(512, 3, resolution, resolution)),
            dtype=torch.float32).to(device)
        for _ in range(steps):
            drift, _ = KG.field(cloud, train.sample(POSITIVES, rng), cloud,
                                branch, kernel, direction_mode="paper",
                                normalization="rms", diagnostics=False)
            cloud = cloud + 0.2 * drift
        return cloud
    model = OneStepGenerator(config.latent_dim, 3, resolution, config.width,
                             derive_seed(seed, "generator")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for step in range(steps):
        latent = sample_latent(FIELD_CLOUD, config.latent_dim,
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
    probe = sample_latent(512, config.latent_dim, derive_seed(seed, "probe"),
                          device)
    with torch.no_grad():
        return model(probe)


def e2_reforms(real: torch.Tensor, device, resolution: int, steps: int,
               root: str | None, seeds: int) -> dict:
    """Do R11 and the ESS-0.9 bandwidth -- the program's two wins -- move FID?"""
    arms = [("generator_ess0.5_noR11", "generator", 0.5, False),
            ("generator_ess0.5_R11", "generator", 0.5, True),
            ("generator_ess0.9_noR11", "generator", 0.9, False),
            ("generator_ess0.9_R11", "generator", 0.9, True),
            ("particles_ess0.9", "particles", 0.9, False)]
    rows = []
    for label, kind, ess, r11 in arms:
        for index in range(seeds):
            seed = MASTER_SEED + SEED_OFFSET + index
            sample = _train(kind, ess, r11, device, resolution, steps, root,
                            seed)
            fid = frechet_distance(sample, real, device)["fid"]
            ed2 = M.energy_distance2(sample.cpu(), real.cpu())
            rows.append({"arm": label, "kind": kind, "ess": ess, "r11": r11,
                         "seed": seed, "fid": fid, "ed2": ed2,
                         "tail": _tail(sample.cpu()),
                         "second_moment": float(
                             sample.flatten(1).var(0).mean()
                             / real.flatten(1).var(0).mean().to(
                                 sample.device))})
            print(f"    E2 {label:24} seed{index} FID={fid:8.2f} "
                  f"ED2={ed2:8.4f} 2nd={rows[-1]['second_moment']:6.3f}",
                  flush=True)
    summary = {}
    for label, kind, ess, r11 in arms:
        group = [r for r in rows if r["arm"] == label]
        summary[label] = {k: float(np.median([r[k] for r in group]))
                          for k in ("fid", "ed2", "tail", "second_moment")}
    return {"rows": rows, "summary": summary}


def e3_samples(real_pool: torch.Tensor, samples: dict, device) -> dict:
    """Is the FID ordering stable between 512 and 2048 samples?"""
    rows = []
    for count in (512, 2048):
        if len(real_pool) < count:
            continue
        reference = real_pool[:count]
        for name, sample in samples.items():
            if len(sample) < count:
                continue
            fid = frechet_distance(sample[:count], reference, device)["fid"]
            rows.append({"samples": count, "arm": name, "fid": fid})
            print(f"    E3 n={count:5} {name:22} FID={fid:8.2f}", flush=True)
    orders = {}
    for count in sorted({r["samples"] for r in rows}):
        subset = [r for r in rows if r["samples"] == count]
        orders[count] = [r["arm"] for r in sorted(subset,
                                                  key=lambda r: r["fid"])]
    keys = sorted(orders)
    return {"rows": rows, "orderings": orders,
            "ordering_stable": (orders[keys[0]] == orders[keys[-1]]
                                if len(keys) > 1 else None)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase15_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    torch.manual_seed(MASTER_SEED + SEED_OFFSET)

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(MASTER_SEED + SEED_OFFSET, "p15"))
    real_a = evaluation.sample(2048, rng)
    real_b = evaluation.sample(2048, rng)

    print("=== E1: what does FID mean, in interpretable degradations ===",
          flush=True)
    e1 = e1_scale(real_a[:512], real_b[:512], device, rng)

    print("\n=== E2: do the program's reforms move FID? ===", flush=True)
    e2 = e2_reforms(real_a[:512], device, args.resolution, args.steps,
                    args.data_root, args.seeds)

    print("\n=== E3: is FID at 512 samples stable? ===", flush=True)
    probes = {
        "real": real_b,
        "gaussian_moment_match": gaussian_moment_match(real_a, 2048, rng),
        "blur_4x": F.interpolate(F.avg_pool2d(real_b, 4), scale_factor=4,
                                 mode="nearest"),
        "pure_noise": torch.tensor(
            rng.normal(scale=0.5, size=tuple(real_b.shape)),
            dtype=torch.float32),
    }
    e3 = e3_samples(real_a, probes, device)

    payload = {"status": "phase14a-followup-probe-feeds-no-gate",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out)},
               "elapsed_seconds": time.time() - started,
               "e1_scale": e1, "e2_reforms": e2, "e3_samples": e3}
    digest = write_json(args.out, payload)

    print("\n=== PHASE-14A FOLLOW-UP ===")
    print(f"\nE1  {'degradation':26}{'FID':>9}{'ED2':>10}{'2nd':>8}")
    for row in e1["rows"]:
        print(f"    {row['variant']:26}{row['fid']:9.2f}{row['ed2']:10.4f}"
              f"{row['second_moment']:8.3f}")
    print(f"\nE2  {'arm':26}{'FID':>9}{'ED2':>10}{'tail':>8}{'2nd':>8}")
    for name, entry in e2["summary"].items():
        print(f"    {name:26}{entry['fid']:9.2f}{entry['ed2']:10.4f}"
              f"{entry['tail']:8.4f}{entry['second_moment']:8.3f}")
    print(f"\nE3  ordering stable between 512 and 2048 samples: "
          f"{e3['ordering_stable']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
