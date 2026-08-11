"""A compact pixel U-ViT with direct denoised-image prediction.

The model accepts physical time ``t`` for interface validation but, following
released iMF/pMF, conditions its tokens only on interval length ``h = t-r``.
Absolute ``t`` remains in the analytic pixel/velocity conversion outside the
network. Its output is pixels, not a velocity.
Long encoder-to-decoder token skips preserve the U-ViT design without an
external image encoder or latent tokenizer.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import PMFModelConfig


class ScalarEmbedding(nn.Module):
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
    """Plain attention written from primitives supported by forward-mode AD."""

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
        # Training is full float32, so no mixed-precision upcast is needed.
        # Preserving dtype also permits a genuinely float64 finite-difference
        # audit of the forward-mode derivative.
        weights = F.softmax(logits, dim=-1)
        weights = F.dropout(weights, self.dropout, self.training)
        attended = torch.einsum("bhts,bshd->bthd", weights, value).reshape(
            batch, count, width
        )
        return self.projection(attended)


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(width * ratio)
        self.norm1 = nn.LayerNorm(width)
        self.attention = SelfAttention(width, heads, dropout)
        self.attention_scale = nn.Parameter(torch.zeros(width))
        self.norm2 = nn.LayerNorm(width)
        self.gate = nn.Linear(width, 2 * hidden)
        self.output = nn.Linear(hidden, width)
        self.mlp_scale = nn.Parameter(torch.zeros(width))
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.dropout(self.attention(self.norm1(tokens))) * (
            self.attention_scale
        )
        left, right = self.gate(self.norm2(tokens)).chunk(2, dim=-1)
        feedforward = self.output(F.silu(left) * right)
        return tokens + self.dropout(feedforward) * self.mlp_scale


class PixelMeanFlowTransformer(nn.Module):
    """Image + ``(t,h)`` -> a denoised-image prediction in raw pixels."""

    def __init__(self, config: PMFModelConfig, seed: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        patch_dim = config.patch_size * config.patch_size * config.channels
        side = config.image_size // config.patch_size
        self.patch_count = side * side
        self.patch_embed = nn.Conv2d(
            config.channels,
            config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.h_embed = ScalarEmbedding(config.time_embedding_dim, config.width)
        self.h_token = nn.Parameter(torch.zeros(1, 1, config.width))
        self.position = nn.Parameter(torch.zeros(1, self.patch_count + 1, config.width))
        half = config.depth // 2
        self.encoder = nn.ModuleList(
            TransformerBlock(
                config.width, config.heads, config.mlp_ratio, config.dropout
            )
            for _ in range(half)
        )
        self.skip_fusions = nn.ModuleList(
            nn.Linear(2 * config.width, config.width) for _ in range(half)
        )
        self.decoder = nn.ModuleList(
            TransformerBlock(
                config.width, config.heads, config.mlp_ratio, config.dropout
            )
            for _ in range(half)
        )
        self.final_norm = nn.LayerNorm(config.width)
        self.pixel_head = nn.Linear(config.width, patch_dim)
        # Training-only resource-scaled iMF auxiliary head.  It predicts the
        # instantaneous denoised-image field used for the JVP tangent and is
        # not evaluated by ``forward`` or the one-step sampler.
        self.auxiliary_pixel_head = nn.Linear(config.width, patch_dim)
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    fan_in = max(1, module.weight[0].numel())
                    # Match iMF/pMF's variance 0.1/fan_in.  The larger
                    # 1/fan_in draft made the JVP unstable in the learning
                    # sanity despite passing purely mechanical checks.
                    std = (0.1 / fan_in) ** 0.5
                    module.weight.copy_(
                        torch.randn(module.weight.shape, generator=generator) * std
                    )
                    if module.bias is not None:
                        module.bias.zero_()
            self.position.copy_(
                torch.randn(self.position.shape, generator=generator) * 0.02
            )
            self.h_token.copy_(
                torch.randn(self.h_token.shape, generator=generator)
                / math.sqrt(self.config.width)
            )
            # The released pMF network uses a zero-initialized prediction
            # layer.  This makes the initial direct pixel output explicit and
            # keeps the initialization independent of image coordinates.
            self.pixel_head.weight.zero_()
            self.pixel_head.bias.zero_()
            self.auxiliary_pixel_head.weight.zero_()
            self.auxiliary_pixel_head.bias.zero_()

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

    def _features(
        self, images: torch.Tensor, time_value: torch.Tensor, interval: torch.Tensor
    ) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != self.config.channels:
            raise ValueError("images must have shape [batch, channels, H, W]")
        if images.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("image resolution differs from model config")
        if len(time_value) != len(images) or len(interval) != len(images):
            raise ValueError("one t and h value are required per image")
        if bool(((interval < -1e-7) | (interval > time_value + 1e-7)).any()):
            raise ValueError("conditions must satisfy 0 <= h <= t")
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        h_token = self.h_token + self.h_embed(interval)[:, None]
        tokens = torch.cat((h_token, patches), dim=1) + self.position
        skips = []
        for block in self.encoder:
            # Store the input of each encoder block.  Storing the output here
            # would make the first decoder concatenate the deepest state with
            # itself, wasting the longest U-shaped skip.
            skips.append(tokens)
            tokens = block(tokens)
        for fusion, block, skip in zip(
            self.skip_fusions, self.decoder, reversed(skips)
        ):
            tokens = fusion(torch.cat((tokens, skip), dim=-1))
            tokens = block(tokens)
        return self.final_norm(tokens[:, 1:])

    def forward(
        self, images: torch.Tensor, time_value: torch.Tensor, interval: torch.Tensor
    ) -> torch.Tensor:
        features = self._features(images, time_value, interval)
        return self._unpatchify(self.pixel_head(features))

    def forward_with_auxiliary(
        self, images: torch.Tensor, time_value: torch.Tensor, interval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return main pixels and the training-only instantaneous head."""
        features = self._features(images, time_value, interval)
        return (
            self._unpatchify(self.pixel_head(features)),
            self._unpatchify(self.auxiliary_pixel_head(features)),
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def inference_parameter_count(self) -> int:
        auxiliary = sum(p.numel() for p in self.auxiliary_pixel_head.parameters())
        return self.parameter_count() - auxiliary
