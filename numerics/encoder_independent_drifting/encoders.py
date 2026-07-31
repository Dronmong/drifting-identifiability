"""Learned-geometry branches, including a REAL pretrained encoder.

Until now this program could not test its own thesis.  `reference_encoder.py`
records why: *"This repository has no pretrained image encoder and no network
access during runs, so the paper's arm cannot be reproduced here."*  That is
no longer true -- torchvision's ImageNet weights download -- so the
encoder-quality ladder the paper's central ablation implies can finally be
run.

Three geometries live here, all exposing the same `Branch` interface the
fixed families use, so the kernel calibration, the field and every
diagnostic treat them identically:

  `pretrained`  ResNet18 with ImageNet weights -- a genuine semantic encoder;
  `random`      the SAME architecture, untrained.  The highest-information
                arm in the ladder: if it matches `pretrained`, the paper's
                "semantic encoder" story does not survive, and
                encoder-independence is nearly free;
  `degraded`    a deliberately lossy map, the bad end of the ladder.

**Scope note that must travel with any G1 number.** ResNet18/ImageNet is a
supervised classifier, not the self-supervised (DINO-family) encoder the
paper uses.  It is a real pretrained semantic encoder and an enormous step up
from a locally-trained autoencoder stand-in, but it is a substitution.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .fixed_features import Branch

# Declared, not tuned.  Images arrive in [-1, 1]; ImageNet statistics are
# applied after mapping to [0, 1].
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# The encoder sees images at this size.  CIFAR at 32 is far below ResNet's
# training resolution, so some upsampling is required for the pretrained
# filters to mean anything; 128 is the declared compromise between fidelity
# and the cost of running the encoder inside every field evaluation.
ENCODER_INPUT = 128


def _prepare(images: torch.Tensor) -> torch.Tensor:
    """[-1, 1] images -> ImageNet-normalized tensor at the encoder's size."""
    x = (images + 1.0) * 0.5
    if x.shape[-1] != ENCODER_INPUT:
        x = F.interpolate(x, size=(ENCODER_INPUT, ENCODER_INPUT),
                          mode="bilinear", align_corners=False)
    mean = torch.tensor(_IMAGENET_MEAN, dtype=x.dtype,
                        device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=x.dtype,
                       device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


_STAGE_END = {"layer1": 5, "layer2": 6, "layer3": 7, "layer4": 8}


def _resnet_trunk(pretrained: bool, seed: int, device,
                  layer: str = "layer3") -> torch.nn.Module:
    """ResNet18 truncated after a named stage.

    Depth is exposed because the invariance probe found feature sensitivity
    to high-frequency perturbation rises monotonically with it -- 4.3x the
    untrained network's at layer1, 14.2x at layer4 -- and Phase 17's failing
    arm was layer3.
    """
    from torchvision.models import ResNet18_Weights, resnet18  # noqa: PLC0415
    if layer not in _STAGE_END:
        raise ValueError(f"unknown layer {layer!r}")
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        # Same architecture, same initialization scheme, no pretraining --
        # the control that separates "semantic features" from "deep conv
        # architecture".
        torch.manual_seed(int(seed) % (2 ** 31))
        model = resnet18(weights=None)
    stages = [model.conv1, model.bn1, model.relu, model.maxpool,
              model.layer1, model.layer2, model.layer3, model.layer4]
    trunk = torch.nn.Sequential(*stages[:_STAGE_END[layer]])
    return trunk.eval().requires_grad_(False).to(device)


def encoder_branch(kind: str, seed: int = 0, device=None,
                   layer: str = "layer3",
                   normalize: str = "l2", pool: int = 2) -> Branch:
    """A learned-geometry branch with a declared normalization.

    ``normalize="l2"`` puts every feature vector on the unit sphere, which is
    what makes the paper's temperature grid meaningful: those values are
    calibrated for *normalized encoder features*, and applying them to raw
    pixels is what collapsed the kernel in Phase 4 (93.8% dead rows).
    """
    if normalize not in ("l2", "none"):
        raise ValueError(f"unknown normalization {normalize!r}")
    if kind in ("pretrained", "random"):
        trunk = _resnet_trunk(kind == "pretrained", seed, device, layer)

        def extract(images: torch.Tensor) -> list[torch.Tensor]:
            with torch.no_grad():
                features = trunk(_prepare(images))
                features = F.adaptive_avg_pool2d(features, pool)
                flat = features.flatten(1)
                if normalize == "l2":
                    flat = flat / flat.norm(dim=1, keepdim=True
                                            ).clamp_min(1e-12)
            return [flat]
    elif kind == "degraded":
        # The bad end of the ladder: heavy blur then a coarse pool, so most
        # of the image's structure is discarded before the kernel sees it.
        def extract(images: torch.Tensor) -> list[torch.Tensor]:
            with torch.no_grad():
                x = F.avg_pool2d(images, 4)
                x = F.interpolate(x, scale_factor=2, mode="nearest")
                flat = x.flatten(1)
                if normalize == "l2":
                    flat = flat / flat.norm(dim=1, keepdim=True
                                            ).clamp_min(1e-12)
            return [flat]
    else:
        raise ValueError(f"unknown encoder geometry {kind!r}")
    suffix = f"_{layer}" if kind in ("pretrained", "random") else ""
    return Branch(f"encoder_{kind}{suffix}", extract)
