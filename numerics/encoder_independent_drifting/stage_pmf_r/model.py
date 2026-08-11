"""Repaired one-step pixel transformer with a deep train-only auxiliary arm."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..stage_pmf.model import ScalarEmbedding, TransformerBlock
from .config import S3RModelConfig


class RepairedPixelMeanFlowTransformer(nn.Module):
    """Direct pixel predictor used by every matched S3R objective arm.

    The inference path remains the S3 U-shaped transformer.  The auxiliary
    prediction now passes through independent transformer blocks instead of
    differing only at the final linear projection.  Absolute time is a
    configurable token: AlphaFlow/EMF need it, while the continuous pMF arm
    can preserve the interval-only conditioning used by the frozen S3 model.
    """

    def __init__(self, config: S3RModelConfig, seed: int) -> None:
        super().__init__()
        config.validate()
        self.config = config
        patch_dim = config.patch_size**2 * config.channels
        side = config.image_size // config.patch_size
        self.patch_count = side**2
        self.patch_embed = nn.Conv2d(
            config.channels,
            config.width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.time_embed = ScalarEmbedding(config.time_embedding_dim, config.width)
        self.interval_embed = ScalarEmbedding(config.time_embedding_dim, config.width)
        self.time_token = nn.Parameter(torch.zeros(1, 1, config.width))
        self.interval_token = nn.Parameter(torch.zeros(1, 1, config.width))
        self.position = nn.Parameter(torch.zeros(1, self.patch_count + 2, config.width))

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

        self.auxiliary_blocks = nn.ModuleList(
            TransformerBlock(
                config.width, config.heads, config.mlp_ratio, config.dropout
            )
            for _ in range(config.auxiliary_depth)
        )
        self.auxiliary_norm = nn.LayerNorm(config.width)
        self.auxiliary_pixel_head = nn.Linear(config.width, patch_dim)
        self._initialize(seed)

    def _initialize(self, seed: int) -> None:
        generator = torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    fan_in = max(1, module.weight[0].numel())
                    module.weight.copy_(
                        torch.randn(module.weight.shape, generator=generator)
                        * math.sqrt(0.1 / fan_in)
                    )
                    if module.bias is not None:
                        module.bias.zero_()
            self.position.copy_(
                torch.randn(self.position.shape, generator=generator) * 0.02
            )
            for token in (self.time_token, self.interval_token):
                token.copy_(
                    torch.randn(token.shape, generator=generator)
                    / math.sqrt(self.config.width)
                )
            self.pixel_head.weight.zero_()
            self.pixel_head.bias.zero_()
            self.auxiliary_pixel_head.weight.zero_()
            self.auxiliary_pixel_head.bias.zero_()

    def _validate_inputs(
        self, images: torch.Tensor, time_value: torch.Tensor, interval: torch.Tensor
    ) -> None:
        if images.ndim != 4 or images.shape[1] != self.config.channels:
            raise ValueError("images must have shape [batch, channels, H, W]")
        if images.shape[-2:] != (self.config.image_size, self.config.image_size):
            raise ValueError("image resolution differs from model config")
        if len(time_value) != len(images) or len(interval) != len(images):
            raise ValueError("one time and interval are required per image")
        if bool(((interval < -1e-7) | (interval > time_value + 1e-7)).any()):
            raise ValueError("conditions must satisfy 0 <= interval <= time")

    def _trunk_features(
        self, images: torch.Tensor, time_value: torch.Tensor, interval: torch.Tensor
    ) -> torch.Tensor:
        self._validate_inputs(images, time_value, interval)
        patches = self.patch_embed(images).flatten(2).transpose(1, 2)
        time_input = (
            time_value
            if self.config.condition_on_absolute_time
            else torch.zeros_like(time_value)
        )
        time_token = self.time_token + self.time_embed(time_input)[:, None]
        interval_token = self.interval_token + self.interval_embed(interval)[:, None]
        tokens = torch.cat((time_token, interval_token, patches), dim=1)
        tokens = tokens + self.position
        skips: list[torch.Tensor] = []
        for block in self.encoder:
            skips.append(tokens)
            tokens = block(tokens)
        for fusion, block, skip in zip(
            self.skip_fusions, self.decoder, reversed(skips)
        ):
            tokens = fusion(torch.cat((tokens, skip), dim=-1))
            tokens = block(tokens)
        return tokens[:, 2:]

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

    def forward(
        self, images: torch.Tensor, time_value: torch.Tensor, interval: torch.Tensor
    ) -> torch.Tensor:
        features = self._trunk_features(images, time_value, interval)
        return self._unpatchify(self.pixel_head(self.final_norm(features)))

    def forward_with_auxiliary(
        self, images: torch.Tensor, time_value: torch.Tensor, interval: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._trunk_features(images, time_value, interval)
        main = self._unpatchify(self.pixel_head(self.final_norm(features)))
        auxiliary = features
        for block in self.auxiliary_blocks:
            auxiliary = block(auxiliary)
        auxiliary = self._unpatchify(
            self.auxiliary_pixel_head(self.auxiliary_norm(auxiliary))
        )
        return main, auxiliary

    def auxiliary_parameter_names(self) -> set[str]:
        return {
            name for name, _ in self.named_parameters() if name.startswith("auxiliary_")
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def inference_parameter_count(self) -> int:
        auxiliary = sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if name in self.auxiliary_parameter_names()
        )
        return self.parameter_count() - auxiliary
