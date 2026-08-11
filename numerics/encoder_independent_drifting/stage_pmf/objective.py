"""Pixel MeanFlow triangle sampling, JVP objective, and one-call sampler."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import PMFObjectiveConfig


@dataclass(frozen=True)
class TriangleSample:
    t: torch.Tensor
    r: torch.Tensor
    diagonal: torch.Tensor


def sample_time_triangle(
    batch: int,
    config: PMFObjectiveConfig,
    value_generator: torch.Generator,
    mask_generator: torch.Generator,
    *,
    dtype: torch.dtype = torch.float32,
) -> TriangleSample:
    """Draw the complete ``0 <= r <= t <= 1`` pMF training triangle.

    An exact rounded fraction of each batch is placed on the flow-matching
    diagonal.  The remaining samples have independently drawn endpoints,
    sorted into the triangle.  The random diagonal positions are chosen by a
    distinct stream so sample order cannot encode the arm.
    """
    config.validate()
    if batch <= 0:
        raise ValueError("batch must be positive")
    raw = torch.randn(batch, 2, generator=value_generator, dtype=dtype)
    raw = torch.sigmoid(raw * config.logit_std + config.logit_mean)
    first, second = raw.unbind(dim=1)
    upper = torch.maximum(first, second)
    lower = torch.minimum(first, second)
    count = round(batch * config.diagonal_fraction)
    diagonal = torch.zeros(batch, dtype=torch.bool)
    if count:
        positions = torch.randperm(batch, generator=mask_generator)[:count]
        diagonal[positions] = True
        # Match the released sampler: for flow-matching samples set the
        # second draw equal to the first *before* sorting.  Using max(first,
        # second) here would silently bias diagonal samples toward high noise.
        upper[diagonal] = first[diagonal]
        lower[diagonal] = first[diagonal]
    return TriangleSample(t=upper, r=lower, diagonal=diagonal)


def average_velocity(
    model: nn.Module,
    state: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    denominator_floor: float,
) -> torch.Tensor:
    """Convert the model's direct pixel prediction into average velocity."""
    interval = t - r
    prediction = model(state, t, interval)
    denominator = t.clamp_min(denominator_floor)[:, None, None, None]
    return (state - prediction) / denominator


@dataclass
class MeanFlowLoss:
    loss: torch.Tensor
    raw_mse: torch.Tensor
    compound: torch.Tensor
    target_velocity: torch.Tensor
    average_velocity: torch.Tensor
    directional_derivative: torch.Tensor
    auxiliary_velocity: torch.Tensor | None
    auxiliary_raw_mse: torch.Tensor
    triangle: TriangleSample


def meanflow_loss(
    model: nn.Module,
    clean: torch.Tensor,
    noise: torch.Tensor,
    triangle: TriangleSample,
    config: PMFObjectiveConfig,
) -> MeanFlowLoss:
    """Compute the unconditional boundary-velocity pMF loss.

    The JVP follows ``(z,t,r)`` in direction ``(v_theta,1,0)``.  Only the
    directional-derivative contribution is stopped.  The returned primal
    ``u`` therefore retains the ordinary first-order parameter gradient.
    """
    config.validate()
    if clean.shape != noise.shape or clean.ndim != 4:
        raise ValueError("clean and noise must be equal-shaped image batches")
    if len(clean) != len(triangle.t) or len(clean) != len(triangle.r):
        raise ValueError("triangle batch differs from image batch")
    device, dtype = clean.device, clean.dtype
    t = triangle.t.to(device=device, dtype=dtype)
    r = triangle.r.to(device=device, dtype=dtype)
    state = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    # Match the stabilized x->velocity conversion on both sides.  Away from
    # the tiny t < floor region this is exactly noise-clean.  Using the bare
    # conditional velocity below the floor would make the regression target
    # inconsistent with the released pMF denominator clamp.
    denominator = t.clamp_min(config.denominator_floor)[:, None, None, None]
    target = (state - clean) / denominator

    has_auxiliary = callable(getattr(model, "forward_with_auxiliary", None))
    if has_auxiliary:
        # The auxiliary head is a resource-scaled form of iMF's predicted
        # marginal-velocity tangent.  It is evaluated at h=0 for the boundary
        # direction and shares the backbone, but is unused at inference.
        with torch.no_grad():
            _, boundary_pixels = model.forward_with_auxiliary(
                state, t, torch.zeros_like(t)
            )
            boundary = (state - boundary_pixels) / denominator

        def u_fn(z_value, t_value, r_value):
            main_pixels, auxiliary_pixels = model.forward_with_auxiliary(
                z_value, t_value, t_value - r_value
            )
            divisor = t_value.clamp_min(config.denominator_floor)[:, None, None, None]
            return (z_value - main_pixels) / divisor, (
                z_value - auxiliary_pixels
            ) / divisor

        average, directional, auxiliary_velocity = torch.func.jvp(
            u_fn,
            (state, t, r),
            (boundary, torch.ones_like(t), torch.zeros_like(r)),
            has_aux=True,
        )
    else:
        # Small analytic test doubles intentionally implement only the main
        # field; the production transformer always takes the branch above.
        with torch.no_grad():
            boundary = average_velocity(model, state, t, t, config.denominator_floor)

        def u_fn(z_value, t_value, r_value):
            return average_velocity(
                model, z_value, t_value, r_value, config.denominator_floor
            )

        average, directional = torch.func.jvp(
            u_fn,
            (state, t, r),
            (boundary, torch.ones_like(t), torch.zeros_like(r)),
        )
        auxiliary_velocity = None
    compound = average + (t - r)[:, None, None, None] * directional.detach()
    residual = compound - target.detach()
    per_sample = residual.square().flatten(1).sum(dim=1)
    denominator = (
        (per_sample + config.adaptive_epsilon).pow(config.adaptive_power).detach()
    )
    loss = (per_sample / denominator).mean()
    if auxiliary_velocity is None:
        auxiliary_raw_mse = torch.zeros((), device=device, dtype=dtype)
    else:
        auxiliary_residual = auxiliary_velocity - target.detach()
        auxiliary_per_sample = auxiliary_residual.square().flatten(1).sum(dim=1)
        auxiliary_denominator = (
            (auxiliary_per_sample + config.adaptive_epsilon)
            .pow(config.adaptive_power)
            .detach()
        )
        loss = loss + (auxiliary_per_sample / auxiliary_denominator).mean()
        auxiliary_raw_mse = auxiliary_residual.square().mean()
    return MeanFlowLoss(
        loss=loss,
        raw_mse=residual.square().mean(),
        compound=compound,
        target_velocity=target,
        average_velocity=average,
        directional_derivative=directional,
        auxiliary_velocity=auxiliary_velocity,
        auxiliary_raw_mse=auxiliary_raw_mse,
        triangle=TriangleSample(t=t, r=r, diagonal=triangle.diagonal.to(device)),
    )


def one_step_sample(model: nn.Module, noise: torch.Tensor) -> torch.Tensor:
    """Map Gaussian noise to pixels using exactly one model invocation."""
    if noise.ndim != 4:
        raise ValueError("noise must be an image batch")
    ones = torch.ones(len(noise), device=noise.device, dtype=noise.dtype)
    return model(noise, ones, ones)
