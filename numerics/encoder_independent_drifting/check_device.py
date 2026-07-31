"""CPU-versus-GPU equivalence and speed, before any GPU result is trusted.

Every phase of this program was measured on the CPU-only torch wheel.  A CUDA
build changes accumulation order and algorithm selection, so GPU numbers are
not automatically comparable with the sealed artifacts.  This quantifies the
difference instead of assuming it away, and measures what the move actually
buys.

Random draws are made on the CPU and moved (see `device.py`), so the sample
path is bit-identical across devices and every difference reported here is
floating-point arithmetic alone.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.check_device
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
from .fixed_features import build_family
from .kernels import calibrate_block_kernel
from .models import OneStepGenerator, sample_latent

HERE = Path(__file__).resolve().parent
GOOD_ESS = 0.9


def _train(device: torch.device, resolution: int, steps: int, cloud: int,
           root: str | None, seed: int) -> torch.Tensor:
    """The real recipe, seeded identically, run wherever it is told."""
    configure(device)
    train = cifar.cifar_target(resolution, "train", root)
    train.device = device
    rng = np.random.default_rng(derive_seed(seed, "check"))
    geometry = GeometryConfig(family="raw", base_kernel="smooth_laplace",
                              target_ess_fraction=GOOD_ESS)
    branch = build_family(geometry, 3).branches[0]
    kernel = calibrate_block_kernel(
        branch, train.sample(256, rng), "smooth_laplace",
        geometry.bandwidth_quantile, geometry.bandwidth_multiplier,
        geometry.kernel_eps, combine=geometry.combine,
        target_ess_fraction=GOOD_ESS)
    config = TrainConfig(steps=steps, batch=64, image_size=resolution)
    model = OneStepGenerator(config.latent_dim, 3, resolution, config.width,
                             derive_seed(seed, "generator")).to(device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)
    for step in range(steps):
        latent = sample_latent(cloud, config.latent_dim,
                               derive_seed(seed, "latent", step), device)
        output = model(latent)
        with torch.no_grad():
            drift, _ = KG.field(output.detach(), train.sample(64, rng),
                                output.detach(), branch, kernel,
                                direction_mode="paper", normalization="rms",
                                diagnostics=False)
            teacher = output.detach() + 0.5 * drift
        loss = ((output - teacher) ** 2).flatten(1).sum(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    probe = sample_latent(512, config.latent_dim, derive_seed(seed, "probe"),
                          device)
    with torch.no_grad():
        return model(probe)


def _time(device: torch.device, resolution: int, steps: int, cloud: int,
          root: str | None, seed: int) -> float:
    if device.type == "cuda":
        _train(device, resolution, 3, cloud, root, seed)   # warm up
        torch.cuda.synchronize()
    started = time.time()
    _train(device, resolution, steps, cloud, root, seed)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--clouds", type=str, default="64,256,512")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "device_check.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    clouds = [int(x) for x in args.clouds.split(",")]
    seed = MASTER_SEED + 99000

    cpu = resolve_device("cpu")
    gpu = resolve_device("cuda")
    payload = {"status": "device-equivalence-and-speed",
               "provenance": provenance(),
               "config": vars(args) | {"out": str(args.out)},
               "cpu": configure(cpu), "gpu": configure(gpu),
               "note": "random draws are made on CPU and moved, so the "
                       "sample path is identical and every difference below "
                       "is floating-point arithmetic"}

    # The arithmetic check: one forward pass, no optimizer, so round-off is
    # not yet amplified.  This is the quantity that says whether the two
    # devices compute the same thing.  The trained comparison below is a
    # *trajectory* comparison, and training is chaotic -- round-off at 1e-7
    # grows to ~1e-4 over tens of steps on either device, which is why the
    # program uses multiple seeds and not why a device would be untrustworthy.
    print("=== arithmetic: one forward pass, no training ===")
    arithmetic = []
    for cloud in clouds:
        a = _train(cpu, args.resolution, 0, cloud, args.data_root,
                   seed).double()
        b = _train(gpu, args.resolution, 0, cloud, args.data_root,
                   seed).detach().cpu().double()
        relative = float((a - b).norm() / a.norm().clamp_min(1e-30))
        arithmetic.append({"cloud": cloud, "relative_difference": relative})
        print(f"    cloud={cloud:4}  relative diff={relative:.3e}", flush=True)

    print("\n=== trajectory: identical seeds, both devices, after training ===")
    equivalence = []
    for cloud in clouds:
        a = _train(cpu, args.resolution, args.steps, cloud, args.data_root,
                   seed).double()
        b = _train(gpu, args.resolution, args.steps, cloud, args.data_root,
                   seed).detach().cpu().double()
        relative = float((a - b).norm() / a.norm().clamp_min(1e-30))
        row = {"cloud": cloud, "relative_difference": relative,
               "max_absolute": float((a - b).abs().max()),
               "cpu_second_moment": float(a.flatten(1).var(0).mean()),
               "gpu_second_moment": float(b.flatten(1).var(0).mean())}
        equivalence.append(row)
        print(f"    cloud={cloud:4}  relative diff={relative:.3e}  "
              f"2nd moment cpu={row['cpu_second_moment']:.6f} "
              f"gpu={row['gpu_second_moment']:.6f}", flush=True)

    print("\n=== speed ===")
    speed = []
    for cloud in clouds:
        cpu_seconds = _time(cpu, args.resolution, args.steps, cloud,
                            args.data_root, seed)
        gpu_seconds = _time(gpu, args.resolution, args.steps, cloud,
                            args.data_root, seed)
        row = {"cloud": cloud, "cpu_seconds": cpu_seconds,
               "gpu_seconds": gpu_seconds,
               "speedup": cpu_seconds / max(gpu_seconds, 1e-9)}
        speed.append(row)
        print(f"    cloud={cloud:4}  cpu={cpu_seconds:7.2f}s  "
              f"gpu={gpu_seconds:7.2f}s  speedup={row['speedup']:5.2f}x",
              flush=True)

    payload |= {"arithmetic": arithmetic, "equivalence": equivalence,
                "speed": speed}
    worst_arithmetic = max(r["relative_difference"] for r in arithmetic)
    worst_trajectory = max(r["relative_difference"] for r in equivalence)
    payload |= {"worst_arithmetic": worst_arithmetic,
                "worst_trajectory": worst_trajectory,
                "verdict": ("the two devices compute the same thing to "
                            "float32 round-off"
                            if worst_arithmetic < 1e-5 else
                            "the devices DISAGREE arithmetically -- do not "
                            "mix GPU results with the sealed CPU artifacts")}
    digest = write_json(args.out, payload)
    print(f"\n  arithmetic (1 forward):  {worst_arithmetic:.3e}   "
          f"<- the criterion")
    print(f"  trajectory (trained):    {worst_trajectory:.3e}   "
          f"<- chaotic amplification, expected")
    print(f"  verdict: {payload['verdict']}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
