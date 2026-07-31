"""Why is a pretrained encoder a bad kernel, and is the fix a shallower one?

Phase 17: a pretrained ResNet18 geometry reaches FID 373.19 against raw
pixels' 234.72 and an *untrained* ResNet18's 226.17 -- paired +138.47 over
raw, both seeds agreeing, with a spectral tail of 0.4853 against real data's
~0.13.  The arm is not collapsing the cloud; it is injecting high-frequency
content.

The proposed reading is **invariance**: a pretrained encoder is trained to
discard texture, colour and position, so matching its features leaves those
details unconstrained and the field fills them with noise.  An untrained
network discards less, so matching it constrains more.

That reading makes three checkable predictions, and the third is a fix:

  R1  **Invariance, measured directly.**  Perturb real images in controlled
      ways and compare how far the *features* move against how far the
      *pixels* move.  If pretrained features are markedly less responsive to
      high-frequency perturbation than untrained ones, the mechanism is
      confirmed without training anything.

  R2  **Depth.**  Invariance is built up with depth, so the failure should
      track the layer the features are taken from.  layer1 should behave
      like raw pixels and layer4 should be worst.

  R3  **If depth is the axis, a shallow pretrained encoder should WORK** --
      which would mean the answer to encoder dependence is not "no encoder"
      but "a shallower one", a much more useful result.

    uv run --python 3.12 \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      --index-strategy unsafe-best-match \
      --with torch==2.7.1+cu126 --with torchvision==0.22.1+cu126 \
      --with numpy --with scipy \
      python -m numerics.encoder_independent_drifting.diagnose_phase18
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import cifar
from .config import MASTER_SEED, derive_seed
from .device import configure, resolve_device
from .diagnostics import provenance, write_json
from .encoders import ENCODER_INPUT, _prepare

HERE = Path(__file__).resolve().parent
SEED_OFFSET = 30000
LAYERS = ("layer1", "layer2", "layer3", "layer4")


def trunk_to(layer: str, pretrained: bool, seed: int, device):
    """ResNet18 truncated after a named stage."""
    from torchvision.models import ResNet18_Weights, resnet18  # noqa: PLC0415
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        torch.manual_seed(int(seed) % (2 ** 31))
        model = resnet18(weights=None)
    stages = [model.conv1, model.bn1, model.relu, model.maxpool,
              model.layer1, model.layer2, model.layer3, model.layer4]
    keep = {"layer1": 5, "layer2": 6, "layer3": 7, "layer4": 8}[layer]
    return torch.nn.Sequential(*stages[:keep]).eval().requires_grad_(
        False).to(device)


def features(trunk, images: torch.Tensor, pool: int = 2) -> torch.Tensor:
    with torch.no_grad():
        out = F.adaptive_avg_pool2d(trunk(_prepare(images)), pool).flatten(1)
        return out / out.norm(dim=1, keepdim=True).clamp_min(1e-12)


def perturbations(images: torch.Tensor, rng) -> dict:
    """Controlled distortions, each with a measurable pixel-space size."""
    generator = torch.Generator().manual_seed(7)
    noise = torch.randn(images.shape, generator=generator).to(images.device)
    shifted = torch.roll(images, shifts=(2, 2), dims=(2, 3))
    blurred = F.interpolate(F.avg_pool2d(images, 2), scale_factor=2,
                            mode="nearest")
    colour = images * torch.tensor(
        [1.15, 1.0, 0.85], device=images.device).view(1, 3, 1, 1)
    return {
        "high_freq_noise": images + 0.15 * noise,
        "translation_2px": shifted,
        "blur_2x": blurred,
        "colour_shift": colour,
    }


def r1_invariance(real: torch.Tensor, device, seed: int) -> dict:
    """How far do features move, per unit of pixel movement?

    A *low* ratio means the encoder is invariant to that perturbation -- it
    cannot tell the distorted image from the original, so a kernel built on
    it will not penalize that distortion either.
    """
    rng = np.random.default_rng(derive_seed(seed, "p18-r1"))
    variants = perturbations(real, rng)
    rows = []
    for layer in LAYERS:
        for pretrained in (True, False):
            trunk = trunk_to(layer, pretrained, seed, device)
            base = features(trunk, real)
            for name, distorted in variants.items():
                moved = features(trunk, distorted)
                d_feature = float((moved - base).norm(dim=1).mean())
                d_pixel = float((distorted - real).flatten(1).norm(
                    dim=1).mean())
                rows.append({
                    "layer": layer,
                    "weights": "pretrained" if pretrained else "random",
                    "perturbation": name,
                    "feature_shift": d_feature,
                    "pixel_shift": d_pixel,
                    "sensitivity": d_feature / max(d_pixel, 1e-12)})
    summary = {}
    for name in variants:
        for layer in LAYERS:
            pre = next(r["sensitivity"] for r in rows
                       if r["layer"] == layer and r["weights"] == "pretrained"
                       and r["perturbation"] == name)
            rnd = next(r["sensitivity"] for r in rows
                       if r["layer"] == layer and r["weights"] == "random"
                       and r["perturbation"] == name)
            summary[f"{name}_{layer}"] = {
                "pretrained": pre, "random": rnd,
                "pretrained_over_random": pre / max(rnd, 1e-12)}
    return {"rows": rows, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--out", type=Path,
                        default=HERE / "phase18_probe.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    settings = configure(device)
    seed = MASTER_SEED + SEED_OFFSET

    started = time.time()
    evaluation = cifar.cifar_target(args.resolution, "eval", args.data_root)
    rng = np.random.default_rng(derive_seed(seed, "p18"))
    real = evaluation.sample(args.samples, rng).to(device)

    print("=== R1: feature sensitivity per unit of pixel movement ===",
          flush=True)
    r1 = r1_invariance(real, device, seed)

    payload = {"status": "phase17-followup-probe-feeds-no-gate",
               "provenance": provenance(), "device": settings,
               "config": vars(args) | {"out": str(args.out),
                                       "encoder_input_px": ENCODER_INPUT},
               "elapsed_seconds": time.time() - started,
               "r1_invariance": r1}
    digest = write_json(args.out, payload)

    print("\n=== SENSITIVITY (feature shift / pixel shift) ===")
    print("low = INVARIANT = the kernel cannot see that distortion\n")
    for name in ("high_freq_noise", "translation_2px", "blur_2x",
                 "colour_shift"):
        print(f"  {name}")
        print(f"    {'layer':10}{'pretrained':>13}{'random':>11}"
              f"{'pre/rand':>11}")
        for layer in LAYERS:
            e = r1["summary"][f"{name}_{layer}"]
            print(f"    {layer:10}{e['pretrained']:13.5f}"
                  f"{e['random']:11.5f}{e['pretrained_over_random']:11.3f}")
    print(f"\nwrote {args.out} sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
