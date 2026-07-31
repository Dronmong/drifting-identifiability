"""A self-trained latent space to drift inside.

Phase 22 located the ceiling of pixel-space kernel drifting precisely.  The
teacher is a kernel-weighted average of the real batch, and the realized
effective sample size never falls below ~18 images at any viable setting --
below that the process degenerates, measured at p = 0.0001.  **A weighted
average of 18 CIFAR images is mush**, so no bandwidth choice can produce a
sharp target.  Both walls come from estimating a kernel in 3072 dimensions
from 64-256 samples, where pairwise distances concentrate.

Averaging in a *learned* latent space and decoding does not have that problem:
the decoder maps the latent average back onto the image manifold, whereas a
pixel-space average leaves it.  This is the standard result in the MMD
generative literature -- moment matching fails in data space and works in
autoencoder code space -- and it is also what the paper itself does, since it
drifts *inside* an encoder's latent space rather than using encoder features
to weight a pixel-space drift.

**This does not abandon encoder independence; it sharpens what the claim
means.**  Two dependencies must not be conflated:

  *external pretrained semantic encoder* (ImageNet / DINO) -- needs outside
      data and supervision.  This is the paper's dependency, the one the
      program set out to remove, and the one Phases 17-18 showed is actively
      harmful when used as a pixel-space kernel.

  *autoencoder trained on the target distribution only* -- no outside data, no
      labels, no semantic supervision, fitted by reconstruction on the same
      40k CIFAR training images the generator already sees.

The second is self-contained, so drifting in it remains encoder-independent in
the sense that matters here.  It is also the first configuration in which the
paper's own ablation becomes directly testable: with a decoder available, one
can drift inside a self-trained latent space, a pretrained encoder's and an
untrained one's and compare.

**The ceiling must be measured before anything is concluded.**  Latent
drifting cannot beat its own autoencoder, so `decode(encode(real))` bounds the
whole approach and has to be reported next to every result.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn

# Declared, not tuned.  4x4x16 = 256 latent dimensions against 3072 pixels: a
# 12x reduction, which moves a 256-particle cloud from hopeless to marginal on
# the particle-count-versus-dimension axis Phase 22 identified as the cause.
LATENT_CHANNELS = 16
LATENT_GRID = 4
WIDTH = 64


def latent_dimension(channels: int = LATENT_CHANNELS,
                     grid: int = LATENT_GRID) -> int:
    return int(channels) * int(grid) * int(grid)


class ConvAutoencoder(nn.Module):
    """Deterministic convolutional autoencoder, [-1, 1] images to [-1, 1].

    Strided convolutions down to ``LATENT_GRID``, mirrored nearest-upsample
    convolutions back.  GroupNorm rather than BatchNorm so that a forward pass
    does not depend on batch composition -- the field evaluates this encoder on
    clouds of varying size and a batch-dependent normalization would make the
    geometry depend on how many particles happened to be present.
    """

    def __init__(self, image_size: int = 32, channels: int = 3,
                 latent_channels: int = LATENT_CHANNELS,
                 width: int = WIDTH, seed: int = 0) -> None:
        super().__init__()
        if image_size < LATENT_GRID or (image_size & (image_size - 1)) != 0:
            raise ValueError(
                f"image size {image_size} must be a power of two and at least "
                f"{LATENT_GRID}")
        stages = 0
        size = image_size
        while size > LATENT_GRID:
            size //= 2
            stages += 1
        self.image_size = int(image_size)
        self.latent_channels = int(latent_channels)

        down: list[nn.Module] = []
        in_channels = channels
        for _ in range(stages):
            down += [
                nn.Conv2d(in_channels, width, 4, stride=2, padding=1),
                nn.GroupNorm(4, width),
                nn.SiLU(),
            ]
            in_channels = width
        down += [nn.Conv2d(width, latent_channels, 3, padding=1,
                           padding_mode="reflect")]
        self.encoder = nn.Sequential(*down)

        up: list[nn.Module] = [
            nn.Conv2d(latent_channels, width, 3, padding=1,
                      padding_mode="reflect"),
            nn.GroupNorm(4, width),
            nn.SiLU(),
        ]
        for _ in range(stages):
            up += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
                nn.GroupNorm(4, width),
                nn.SiLU(),
            ]
        up += [nn.Conv2d(width, channels, 3, padding=1,
                         padding_mode="reflect")]
        self.decoder = nn.Sequential(*up)
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) % (2 ** 63 - 1))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = module.weight[0].numel()
                bound = 1.0 / max(fan_in, 1) ** 0.5
                with torch.no_grad():
                    module.weight.copy_(
                        (torch.rand(module.weight.shape, generator=generator)
                         * 2 - 1) * bound)
                    if module.bias is not None:
                        module.bias.zero_()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 2:
            latent = latent.reshape(len(latent), self.latent_channels,
                                    LATENT_GRID, LATENT_GRID)
        return self.decoder(latent)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def train_autoencoder(target, steps: int, seed: int, device,
                      image_size: int = 32,
                      latent_channels: int = LATENT_CHANNELS,
                      width: int = WIDTH, batch: int = 128,
                      learning_rate: float = 2e-3,
                      report_every: int = 0) -> dict:
    """Fit by reconstruction MSE on the target split alone.

    Cosine-decayed learning rate, matching what Phase 19 measured as the
    better schedule for the generator.  The returned loss history is the
    evidence that the autoencoder converged, and it is written into every
    artifact that uses the resulting latent space.
    """
    from .config import derive_seed

    model = ConvAutoencoder(image_size, 3, latent_channels, width,
                            derive_seed(seed, "autoencoder")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    rng = np.random.default_rng(derive_seed(seed, "autoencoder-batches"))
    history = []
    started = time.time()
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * 0.5 * (
                1.0 + np.cos(np.pi * step / max(steps, 1)))
        images = target.sample(batch, rng)
        loss = ((model(images) - images) ** 2).flatten(1).mean(1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if report_every and (step % report_every == 0 or step == steps - 1):
            history.append({"step": step, "mse": float(loss)})
            print(f"      ae step {step:6} mse={float(loss):.5f}", flush=True)
    model.eval().requires_grad_(False)
    return {"model": model, "history": history,
            "wall_seconds": time.time() - started,
            "parameters": model.parameter_count(),
            "latent_dimension": latent_dimension(latent_channels)}


def latent_statistics(model: ConvAutoencoder, target, samples: int,
                      rng) -> dict:
    """Scale of the learned code space, for the record.

    The kernel calibration is median-based and therefore scale free, but the
    numbers are reported so a later reader can tell whether the latent space
    drifted in scale between runs.
    """
    with torch.no_grad():
        codes = model.encode(target.sample(samples, rng)).flatten(1)
        return {"latent_rms": float(codes.pow(2).mean().sqrt()),
                "latent_abs_max": float(codes.abs().max()),
                "latent_per_dim_std_mean": float(codes.std(dim=0).mean())}
