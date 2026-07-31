"""One-step image generator (plan section 1, item 3).

Small, fixed-capacity, deterministic from a declared seed, and identical
across every arm so that a screen compares objectives rather than
architectures.  Nothing here is pretrained and no arm may change the
generator: matched capacity is part of the plan's matched-accounting rule.

Inference is one forward pass -- NFE = 1 -- which is the property the
drifting family is being kept for.
"""

from __future__ import annotations

import torch
from torch import nn


class OneStepGenerator(nn.Module):
    """latent -> 4x4 feature map -> nearest-upsample conv stack -> image."""

    def __init__(self, latent_dim: int, channels: int, image_size: int,
                 width: int, seed: int) -> None:
        super().__init__()
        # The stack upsamples by 2 from a 4x4 projection, so only sizes of
        # the form 4 * 2^k are reachable.  Checking divisibility by 4 alone
        # let image_size=24 through and silently produced 32x32 output --
        # caught when a downstream kernel saw mismatched dimensions.
        if image_size < 4 or (image_size & (image_size - 1)) != 0:
            raise ValueError(
                f"image size {image_size} is not reachable: this generator "
                "upsamples by 2 from 4x4, so it must be 4 * 2^k "
                "(4, 8, 16, 32, 64, ...)")
        self.latent_dim = int(latent_dim)
        self.channels = int(channels)
        self.image_size = int(image_size)
        stages = 0
        size = 4
        while size < image_size:
            size *= 2
            stages += 1
        self.project = nn.Linear(latent_dim, width * 4 * 4)
        blocks: list[nn.Module] = []
        for _ in range(stages):
            blocks += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
                nn.GroupNorm(4, width),
                nn.SiLU(),
            ]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Conv2d(width, channels, 3, padding=1,
                              padding_mode="reflect")
        self.width = int(width)
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) % (2 ** 63 - 1))
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                fan_in = module.weight[0].numel()
                bound = 1.0 / max(fan_in, 1) ** 0.5
                with torch.no_grad():
                    module.weight.copy_(
                        (torch.rand(module.weight.shape, generator=generator)
                         * 2 - 1) * bound)
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent dim {latent.shape[1]} != {self.latent_dim}")
        x = self.project(latent).reshape(len(latent), self.width, 4, 4)
        return self.head(self.blocks(x))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LatentGenerator(nn.Module):
    """noise -> autoencoder code, for drifting inside a learned latent space.

    Phase 24 measured the autoencoder ceiling: a 512-dimensional code
    reconstructs CIFAR at KID 0.031 with *recognizable objects*, against the
    best pixel-space drifting arm's 0.131.  Drifting in that code space needs a
    generator whose output IS a code, so this produces a ``channels x grid x
    grid`` tensor at fixed resolution rather than upsampling to an image.

    ``OneStepGenerator`` would technically serve at ``image_size=4``, but with
    zero upsampling stages it degenerates to a linear map plus one convolution.
    That is far shallower than the pixel-space generator it would be compared
    against, which would confound a geometry comparison with a capacity one --
    the mistake Phase 1 made with kernel smoothness and Phase 14A made with
    collapse.  Depth here is explicit and reported.

    Inference is still one forward pass; the decoder is a fixed post-map, so
    NFE = 1 is preserved.
    """

    def __init__(self, latent_dim: int, channels: int, grid: int, width: int,
                 seed: int, blocks: int = 3) -> None:
        super().__init__()
        if grid < 1:
            raise ValueError("the code grid must be at least 1")
        if blocks < 0:
            raise ValueError("the block count must be nonnegative")
        self.latent_dim = int(latent_dim)
        self.channels = int(channels)
        self.grid = int(grid)
        self.width = int(width)
        self.project = nn.Linear(latent_dim, width * grid * grid)
        stack: list[nn.Module] = []
        for _ in range(blocks):
            stack += [
                nn.Conv2d(width, width, 3, padding=1, padding_mode="reflect"),
                nn.GroupNorm(4, width),
                nn.SiLU(),
            ]
        self.blocks = nn.Sequential(*stack)
        self.head = nn.Conv2d(width, channels, 3, padding=1,
                              padding_mode="reflect")
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) % (2 ** 63 - 1))
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                fan_in = module.weight[0].numel()
                bound = 1.0 / max(fan_in, 1) ** 0.5
                with torch.no_grad():
                    module.weight.copy_(
                        (torch.rand(module.weight.shape, generator=generator)
                         * 2 - 1) * bound)
                    if module.bias is not None:
                        module.bias.zero_()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent dim {latent.shape[1]} != {self.latent_dim}")
        x = self.project(latent).reshape(len(latent), self.width, self.grid,
                                         self.grid)
        return self.head(self.blocks(x))

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def sample_latent(count: int, latent_dim: int, seed: int,
                  device: torch.device | str | None = None) -> torch.Tensor:
    """Latents drawn on the CPU, then moved.

    The draw stays on the CPU so the sample path is identical whichever
    device trains -- a CUDA generator does not reproduce a CPU one, and a
    CPU/GPU comparison in which the random numbers also differ would measure
    nothing.  See `device.py`.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2 ** 63 - 1))
    latent = torch.randn(count, latent_dim, generator=generator)
    return latent if device is None else latent.to(device)
