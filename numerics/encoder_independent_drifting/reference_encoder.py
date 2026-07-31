"""Arm A8 reference geometry: a LOCALLY TRAINED encoder stand-in.

Read this before quoting any A8 number.

The plan's Phase-1 arm table lists ``A8`` as "pretrained paper encoder,
paper protocol".  This repository has no pretrained image encoder and no
network access during runs, so the paper's arm **cannot** be reproduced
here.  What follows is a *stand-in*: a small convolutional autoencoder
trained from scratch on samples from the same target family, used only to
supply a learned similarity geometry for comparison.

Consequences, which every report must carry:

* A8 is not evidence about the paper's ImageNet encoder or its FID ladder.
* A8 sees target data at training time and is therefore, if anything,
  *favoured* relative to a genuinely external encoder on these targets.
* A8 is excluded from the Phase-1 exit gate, which is defined only over
  A1/A4/A5 (plan section 9). It is a context arm.

It is included because the alternative -- silently dropping the only
learned-geometry comparison -- would make the encoder-free arms look
unopposed.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .fixed_features import Branch, FeatureFamily, local_pool
from .config import GeometryConfig


class SmallEncoder(nn.Module):
    """Conv encoder/decoder pair trained by denoising reconstruction."""

    def __init__(self, channels: int, width: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) % (2 ** 63 - 1))
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1, padding_mode="reflect"),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
            nn.SiLU(),
            nn.Conv2d(width, channels, 3, padding=1, padding_mode="reflect"),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = module.weight[0].numel()
                bound = 1.0 / max(fan_in, 1) ** 0.5
                with torch.no_grad():
                    module.weight.copy_(
                        (torch.rand(module.weight.shape, generator=generator)
                         * 2 - 1) * bound)
                    module.bias.zero_()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images))


def train_reference_encoder(target_sampler, channels: int, width: int,
                            seed: int, steps: int, batch: int,
                            learning_rate: float, rng) -> SmallEncoder:
    """Denoising reconstruction on target-family samples.  Trained, not
    pretrained; see the module docstring."""
    model = SmallEncoder(channels, width, seed)
    torch_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(int(seed + 1) % (2 ** 63 - 1))
    # Follow the target's device rather than assuming CPU, so this arm runs
    # wherever the rest of the phase does.
    probe = target_sampler(1, rng)
    model = model.to(probe.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(steps):
        clean = target_sampler(batch, rng)
        # Drawn on the CPU and moved, so the noise sequence is identical
        # whichever device trains (see `device.py`).
        noisy = clean + 0.3 * torch.randn(
            clean.shape, generator=torch_generator).to(clean.device)
        loss = ((model(noisy) - clean) ** 2).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def encoder_family(model: SmallEncoder, config: GeometryConfig
                   ) -> FeatureFamily:
    """Expose the frozen encoder as a single geometry branch."""
    def extract(images: torch.Tensor) -> list[torch.Tensor]:
        codes = model.encoder(images)
        pooled = local_pool(codes, config.pool)
        b, c, g, _ = pooled.shape
        flat = pooled.reshape(b, c, g * g)
        return [flat[:, :, r] for r in range(g * g)]

    branch = Branch("reference_encoder", extract)
    return FeatureFamily("reference_encoder", [branch], config,
                         list(model.parameters()))
