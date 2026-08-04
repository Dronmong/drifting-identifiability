"""Patch-2 U-ViT with AdaLN-Zero conditioning and a local pixel refiner.

Two departures from the S3R trunk, both deliberate and both recorded in
protocol section 5:

* **AdaLN-Zero replaces the two prepended conditioning tokens.**  S3R fed
  ``cat((time_token, interval_token, patches))``, so its sequence was
  ``256 + 2`` and the image grid was ``tokens[:, 2:]``.  Here the sequence *is*
  the image grid, exactly 256 tokens, no slice.  ASFD extracts features from
  that grid, and an off-by-two slice is invisible to a forward-parity test
  because a read-only hook genuinely does not change model outputs.
* **A shallow convolutional refiner follows unpatchify**, targeting the HH
  deficit S3R measured (0.159 of target).  It is inside the single inference
  call and introduces no learned external representation.

The trunk is a U-ViT: encoder blocks then decoder blocks fusing reversed
encoder skips.  Width and token count are uniform throughout — **there is no
spatial bottleneck**, and no caller may assume one.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import FEATURE_LEVELS, CAPModelConfig


class ScalarEmbedding(nn.Module):
    """Sinusoidal scalar features followed by a small MLP."""

    def __init__(self, dimension: int, output: int) -> None:
        super().__init__()
        half = dimension // 2
        frequencies = torch.exp(
            -math.log(10_000.0) * torch.arange(half) / max(half - 1, 1)
        )
        self.register_buffer("frequencies", frequencies, persistent=True)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, output), nn.SiLU(), nn.Linear(output, output)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 2 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 1:
            raise ValueError("scalar condition must have shape [batch]")
        angles = value[:, None] * 1_000.0 * self.frequencies[None]
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))


class SelfAttention(nn.Module):
    """Attention from primitives that support forward-mode AD.

    Written explicitly rather than through ``scaled_dot_product_attention`` so
    the float64 finite-difference-versus-JVP audit in the preflight can run
    against the real forward path.
    """

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.projection = nn.Linear(width, width)
        self.dropout = float(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, count, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        logits = torch.einsum("bthd,bshd->bhts", query, key) / math.sqrt(self.head_dim)
        weights = F.softmax(logits, dim=-1)
        weights = F.dropout(weights, self.dropout, self.training)
        attended = torch.einsum("bhts,bshd->bthd", weights, value).reshape(
            batch, count, width
        )
        return self.projection(attended)


class AdaLNZeroBlock(nn.Module):
    """Transformer block modulated by a conditioning vector, identity at init.

    The modulation projection is zero-initialized, so both residual gates start
    at zero and every block begins as the identity map.  That is the property
    AdaLN-*Zero* is named for and it is what keeps a 12-block stack stable
    before the conditioning has learned anything.
    """

    def __init__(
        self, width: int, heads: int, ratio: float, dropout: float, condition: int
    ) -> None:
        super().__init__()
        hidden = int(width * ratio)
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = SelfAttention(width, heads, dropout)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.gate = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)
        self.modulation = nn.Linear(condition, 6 * width)

    def forward(
        self, tokens: torch.Tensor, conditioning: torch.Tensor
    ) -> torch.Tensor:
        parts = self.modulation(conditioning).chunk(6, dim=-1)
        shift1, scale1, gate1, shift2, scale2, gate2 = (p[:, None] for p in parts)
        hidden = self.norm1(tokens) * (1.0 + scale1) + shift1
        tokens = tokens + gate1 * self.dropout(self.attention(hidden))
        hidden = self.norm2(tokens) * (1.0 + scale2) + shift2
        left, right = self.gate(hidden).chunk(2, dim=-1)
        feedforward = self.output(F.silu(left) * right)
        return tokens + gate2 * self.dropout(feedforward)


class PixelRefiner(nn.Module):
    """Shallow convolutional residual head; zero-initialized, so identity."""

    def __init__(self, channels: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, width, 3, padding=1)]
        for _ in range(depth - 1):
            layers += [nn.SiLU(), nn.Conv2d(width, width, 3, padding=1)]
        layers += [nn.SiLU(), nn.Conv2d(width, channels, 3, padding=1)]
        self.body = nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images + self.body(images)


class CAPPixelTransformer(nn.Module):
    """Image + ``(t, h)`` -> a direct denoised-image prediction in raw pixels."""

    def __init__(self, config: CAPModelConfig, seed: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        patch_dim = config.patch_size * config.patch_size * config.channels
        self.patch_count = config.tokens
        self.patch_embed = nn.Conv2d(
            config.channels,
            config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.position = nn.Parameter(torch.zeros(1, self.patch_count, config.width))

        self.time_embed = ScalarEmbedding(config.time_embedding_dim, config.condition_dim)
        self.interval_embed = ScalarEmbedding(
            config.time_embedding_dim, config.condition_dim
        )
        self.condition_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.condition_dim, config.condition_dim),
        )

        half = config.depth // 2
        self.encoder = nn.ModuleList(
            AdaLNZeroBlock(
                config.width,
                config.heads,
                config.mlp_ratio,
                config.dropout,
                config.condition_dim,
            )
            for _ in range(half)
        )
        self.skip_fusions = nn.ModuleList(
            nn.Linear(2 * config.width, config.width) for _ in range(half)
        )
        self.decoder = nn.ModuleList(
            AdaLNZeroBlock(
                config.width,
                config.heads,
                config.mlp_ratio,
                config.dropout,
                config.condition_dim,
            )
            for _ in range(half)
        )
        self.final_norm = nn.LayerNorm(config.width)
        self.pixel_head = nn.Linear(config.width, patch_dim)
        self.refiner = PixelRefiner(
            config.channels, config.refiner_width, config.refiner_depth
        )
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    fan_in = max(1, module.weight[0].numel())
                    # iMF/pMF variance 0.1/fan_in; the larger 1/fan_in draft was
                    # unstable in the S3 learning sanity despite passing purely
                    # mechanical checks.
                    std = (0.1 / fan_in) ** 0.5
                    module.weight.copy_(
                        torch.randn(module.weight.shape, generator=generator) * std
                    )
                    if module.bias is not None:
                        module.bias.zero_()
            self.position.copy_(
                torch.randn(self.position.shape, generator=generator) * 0.02
            )
            # AdaLN-Zero: zero modulation means zero residual gates, so every
            # block starts as the identity.
            for block in list(self.encoder) + list(self.decoder):
                block.modulation.weight.zero_()
                block.modulation.bias.zero_()
            # Direct pixel prediction starts explicit and coordinate-free.
            self.pixel_head.weight.zero_()
            self.pixel_head.bias.zero_()
            # The refiner starts as the identity so it cannot corrupt the
            # trunk's output before it has learned anything.
            final_conv = self.refiner.body[-1]
            final_conv.weight.zero_()
            final_conv.bias.zero_()

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch, count, patch_dim = patches.shape
        p = self.config.patch_size
        side = self.config.image_size // p
        expected = p * p * self.config.channels
        if count != side * side or patch_dim != expected:
            raise RuntimeError("invalid patch prediction shape")
        patches = patches.reshape(batch, side, side, self.config.channels, p, p)
        return (
            patches.permute(0, 3, 1, 4, 2, 5)
            .reshape(
                batch,
                self.config.channels,
                self.config.image_size,
                self.config.image_size,
            )
            .contiguous()
        )

    def _conditioning(
        self, time_value: torch.Tensor, interval: torch.Tensor
    ) -> torch.Tensor:
        return self.condition_mlp(
            self.time_embed(time_value) + self.interval_embed(interval)
        )

    def _validate(
        self, images: torch.Tensor, time_value: torch.Tensor, interval: torch.Tensor
    ) -> None:
        if images.ndim != 4 or images.shape[1] != self.config.channels:
            raise ValueError("images must have shape [batch, channels, H, W]")
        if images.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("image resolution differs from model config")
        if len(time_value) != len(images) or len(interval) != len(images):
            raise ValueError("one t and h value are required per image")
        if bool(((interval < -1e-7) | (interval > time_value + 1e-7)).any()):
            raise ValueError("conditions must satisfy 0 <= h <= t")

    def _trunk(
        self,
        images: torch.Tensor,
        time_value: torch.Tensor,
        interval: torch.Tensor,
        collect: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate(images, time_value, interval)
        conditioning = self._conditioning(time_value, interval)
        tokens = self.patch_embed(images).flatten(2).transpose(1, 2) + self.position

        taps: dict[str, dict[int, str]] = {"encoder": {}, "decoder": {}}
        if collect:
            for label, stack, index in FEATURE_LEVELS:
                taps[stack][index] = label
        features: dict[str, torch.Tensor] = {}

        skips: list[torch.Tensor] = []
        for index, block in enumerate(self.encoder):
            # Store each block's *input*.  Storing the output would make the
            # first decoder concatenate the deepest state with itself, wasting
            # the longest U-shaped skip.
            skips.append(tokens)
            tokens = block(tokens, conditioning)
            if index in taps["encoder"]:
                features[taps["encoder"][index]] = tokens
        for index, (fusion, block, skip) in enumerate(
            zip(self.skip_fusions, self.decoder, reversed(skips))
        ):
            tokens = fusion(torch.cat((tokens, skip), dim=-1))
            tokens = block(tokens, conditioning)
            if index in taps["decoder"]:
                features[taps["decoder"][index]] = tokens
        return tokens, features

    def forward(
        self, images: torch.Tensor, time_value: torch.Tensor, interval: torch.Tensor
    ) -> torch.Tensor:
        tokens, _ = self._trunk(images, time_value, interval, collect=False)
        base = self._unpatchify(self.pixel_head(self.final_norm(tokens)))
        return self.refiner(base)

    def forward_with_features(
        self, images: torch.Tensor, time_value: torch.Tensor, interval: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Ordinary prediction plus the declared feature levels.

        The returned token maps are ``[batch, 256, width]`` — the image grid
        itself, with **no conditioning-token slice**, which is the whole point
        of section 5.2.  ``forward`` must be bit-identical to this method's
        first return value; the preflight asserts it.
        """
        tokens, features = self._trunk(images, time_value, interval, collect=True)
        base = self._unpatchify(self.pixel_head(self.final_norm(tokens)))
        return self.refiner(base), features

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def one_step_sample(model: nn.Module, noise: torch.Tensor) -> torch.Tensor:
    """The complete sampler: exactly one network evaluation."""
    if noise.ndim != 4:
        raise ValueError("noise must be an image batch")
    ones = torch.ones(len(noise), device=noise.device, dtype=noise.dtype)
    return model(noise, ones, ones)
