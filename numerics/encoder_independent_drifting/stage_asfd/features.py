"""Reading the frozen trunk.

Freezing parameters is **not** the same as detaching inputs.  The generated
branch needs ``d h(x) / d x`` for the semantic energy to reach the online
generator at all, so the frozen forward runs under grad.  Implementing it under
``torch.no_grad()`` would leave a branch that still trains, still logs a falling
loss, and does nothing -- which is why a preflight check asserts the generated
input gradient is finite and nonzero.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..stage_cap.model import CAPPixelTransformer
from .config import FeatureConfig


def freeze_trunk(model: CAPPixelTransformer) -> CAPPixelTransformer:
    """Parameters frozen, evaluation mode, inputs still differentiable."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval()


def noise_images(
    images: torch.Tensor, t_f: float, generator: torch.Generator
) -> torch.Tensor:
    """The trunk's **native** corruption path, not an arbitrary perturbation.

    ``x_tf = (1 - t_f) x + t_f xi`` evaluated at absolute time ``t_f`` is
    in-distribution for a time-conditioned flow trunk.  CleanDIFT establishes
    that diffusion features are poor on clean inputs because training happened
    on noisy states, and that averaging arbitrary noise draws does not repair a
    mismatched extraction regime.
    """
    if not 0 < t_f < 1:
        raise ValueError("feature noise level must lie in (0,1)")
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype)
    return (1.0 - t_f) * images + t_f * noise.to(images.device)


def extract_tokens(
    trunk: CAPPixelTransformer,
    noised: torch.Tensor,
    t_f: float,
    levels: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """Token grids at the declared levels, evaluated at time ``t_f``, interval 0."""
    times = torch.full((len(noised),), float(t_f), device=noised.device)
    intervals = torch.zeros_like(times)
    _, features = trunk.forward_with_features(noised, times, intervals)
    missing = set(levels) - set(features)
    if missing:
        raise RuntimeError(f"trunk did not expose feature levels {sorted(missing)}")
    return {name: features[name] for name in levels}


def descriptors(tokens: torch.Tensor, pool: int) -> torch.Tensor:
    """``[batch, 256, C]`` -> ``[batch, 66, C]``: 64 pooled local, 2 global.

    The two global vectors are the only permutation-invariant part of the
    descriptor; the other 64 are position-locked, which :mod:`field` reports as
    an energy split rather than leaving implicit.
    """
    if tokens.ndim != 3:
        raise ValueError("tokens must be [batch, count, channels]")
    batch, count, channels = tokens.shape
    side = round(count**0.5)
    if side * side != count:
        raise ValueError(f"{count} tokens is not a square grid")
    if side % pool:
        raise ValueError("pool must divide the token grid")
    grid = tokens.reshape(batch, side, side, channels).permute(0, 3, 1, 2)
    pooled = torch.nn.functional.avg_pool2d(grid, pool)
    local = pooled.flatten(2).transpose(1, 2)
    mean = grid.mean(dim=(2, 3)).unsqueeze(1)
    # Unbiased=False so a batch of one is still well defined.
    std = grid.var(dim=(2, 3), unbiased=False).clamp_min(0).sqrt().unsqueeze(1)
    return torch.cat((local, mean, std), dim=1)


@dataclass(frozen=True)
class LevelNormalization:
    """Frozen target-only scales.  The metric must not move with the model."""

    channel_scale: torch.Tensor | None
    level_scale: float
    pc1_share: float
    per_channel_applied: bool

    def apply(self, values: torch.Tensor) -> torch.Tensor:
        if self.channel_scale is not None:
            values = values / self.channel_scale.to(values.device)
        return values / self.level_scale


def apply_normalization(
    descriptor_map: dict[str, torch.Tensor],
    normalization: dict[str, LevelNormalization],
) -> dict[str, torch.Tensor]:
    missing = set(descriptor_map) - set(normalization)
    if missing:
        raise RuntimeError(f"no frozen normalization for levels {sorted(missing)}")
    return {
        name: normalization[name].apply(values)
        for name, values in descriptor_map.items()
    }


def encode(
    trunk: CAPPixelTransformer,
    images: torch.Tensor,
    t_f: float,
    config: FeatureConfig,
    normalization: dict[str, LevelNormalization] | None = None,
    *,
    generator: torch.Generator | None = None,
    noised: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Images -> normalized ``[batch, 66, C]`` descriptors per level.

    Pass ``noised`` to reuse a corruption draw; pass ``generator`` to make one.
    Roles must never share a draw -- pairing a positive and a negative through
    the same noise tensor would correlate the two barycenters.
    """
    config.validate()
    if noised is None:
        if generator is None:
            raise ValueError("encode needs either a noise generator or noised images")
        noised = noise_images(images, t_f, generator)
    tokens = extract_tokens(trunk, noised, t_f, config.levels)
    result = {name: descriptors(value, config.pool) for name, value in tokens.items()}
    if normalization is not None:
        result = apply_normalization(result, normalization)
    return result


def to_locations(descriptor_map: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``[batch, 66, C]`` -> ``[66, batch, C]`` so locations batch the field."""
    return {name: value.transpose(0, 1) for name, value in descriptor_map.items()}


def input_jacobian_is_alive(
    trunk: CAPPixelTransformer, config: FeatureConfig, t_f: float
) -> dict:
    """Assert the frozen trunk still passes gradient to its input.

    This is the check that would have caught a ``no_grad`` frozen forward, the
    single most expensive mistake available in this design: everything would
    still run and nothing would be learned.
    """
    images = torch.randn(2, 3, 32, 32, requires_grad=True)
    generator = torch.Generator().manual_seed(0)
    encoded = encode(trunk, images, t_f, config, generator=generator)
    total = sum(value.square().sum() for value in encoded.values())
    total.backward()
    grad = images.grad
    alive = grad is not None and bool(torch.isfinite(grad).all())
    magnitude = 0.0 if grad is None else float(grad.abs().max())
    frozen_clean = all(
        parameter.grad is None or float(parameter.grad.abs().max()) == 0.0
        for parameter in trunk.parameters()
    )
    return {
        "input_gradient_alive": bool(alive and magnitude > 0.0),
        "input_gradient_max": magnitude,
        "frozen_parameters_received_no_gradient": frozen_clean,
    }
