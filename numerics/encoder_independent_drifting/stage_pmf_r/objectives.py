"""Auditable pMF, discrete AlphaFlow, and x1-EMF objectives for S3R."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..stage_pmf.objective import TriangleSample
from .config import S3RObjectiveConfig


@dataclass
class ObjectiveOutcome:
    loss: torch.Tensor
    raw_mse: torch.Tensor
    per_sample_raw_mse: torch.Tensor
    diagonal_raw_mse: torch.Tensor
    interior_raw_mse: torch.Tensor
    jvp_per_sample_rms: torch.Tensor
    auxiliary_raw_mse: torch.Tensor
    tfm_loss: torch.Tensor | None
    tc_loss: torch.Tensor | None
    alpha: float | None


def _conditions(
    clean: torch.Tensor,
    noise: torch.Tensor,
    triangle: TriangleSample,
    config: S3RObjectiveConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    config.validate()
    if clean.shape != noise.shape or clean.ndim != 4:
        raise ValueError("clean and noise must be equal-shaped image batches")
    if len(clean) != len(triangle.t) or len(clean) != len(triangle.r):
        raise ValueError("triangle and image batches differ")
    t = triangle.t.to(device=clean.device, dtype=clean.dtype)
    r = triangle.r.to(device=clean.device, dtype=clean.dtype)
    diagonal = triangle.diagonal.to(device=clean.device)
    state = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    denominator = t.clamp_min(config.denominator_floor)[:, None, None, None]
    conditional_velocity = (state - clean) / denominator
    return t, r, diagonal, state, conditional_velocity


def velocity_from_pixels(
    model: nn.Module,
    state: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    denominator_floor: float,
) -> torch.Tensor:
    prediction = model(state, t, t - r)
    divisor = t.clamp_min(denominator_floor)[:, None, None, None]
    return (state - prediction) / divisor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return selected.mean()


def _adaptive_loss(
    residual: torch.Tensor,
    config: S3RObjectiveConfig,
    numerator: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_sample_sum = residual.square().flatten(1).sum(dim=1)
    weights = numerator / (
        (per_sample_sum + config.adaptive_epsilon).pow(config.adaptive_power).detach()
    )
    return (weights * per_sample_sum).mean(), residual.square().flatten(1).mean(dim=1)


def _alpha_adaptive_loss(
    residual: torch.Tensor,
    diagonal: torch.Tensor,
    config: S3RObjectiveConfig,
    alpha: float,
) -> torch.Tensor:
    """Released AlphaFlow adaptive weighting, preserving FM boundary rows.

    The public implementation normalizes the per-sample *mean* squared error
    by ``mse.detach() + 1e-3``.  Its multiplier is one on the r=t
    flow-matching rows and alpha only on the discrete interior rows.
    """
    per_sample_mse = residual.square().flatten(1).mean(dim=1)
    numerator = torch.where(
        diagonal,
        torch.ones_like(per_sample_mse),
        torch.full_like(per_sample_mse, alpha),
    )
    weight = numerator / (per_sample_mse.detach() + config.alpha_adaptive_epsilon)
    return (weight * per_sample_mse).mean()


def _outcome(
    *,
    loss: torch.Tensor,
    residual: torch.Tensor,
    diagonal: torch.Tensor,
    jvp: torch.Tensor | None = None,
    auxiliary_raw_mse: torch.Tensor | None = None,
    tfm_loss: torch.Tensor | None = None,
    tc_loss: torch.Tensor | None = None,
    alpha: float | None = None,
) -> ObjectiveOutcome:
    per_sample = residual.square().flatten(1).mean(dim=1)
    if jvp is None:
        jvp_rms = torch.zeros_like(per_sample)
    else:
        jvp_rms = jvp.detach().square().flatten(1).mean(dim=1).sqrt()
    if auxiliary_raw_mse is None:
        auxiliary_raw_mse = torch.zeros(
            (), device=residual.device, dtype=residual.dtype
        )
    return ObjectiveOutcome(
        loss=loss,
        raw_mse=per_sample.mean(),
        per_sample_raw_mse=per_sample.detach(),
        diagonal_raw_mse=_masked_mean(per_sample, diagonal),
        interior_raw_mse=_masked_mean(per_sample, ~diagonal),
        jvp_per_sample_rms=jvp_rms,
        auxiliary_raw_mse=auxiliary_raw_mse,
        tfm_loss=tfm_loss,
        tc_loss=tc_loss,
        alpha=alpha,
    )


def pmf_loss(
    model: nn.Module,
    clean: torch.Tensor,
    noise: torch.Tensor,
    triangle: TriangleSample,
    config: S3RObjectiveConfig,
) -> ObjectiveOutcome:
    """Continuous pMF with a deep auxiliary marginal-velocity tangent."""
    t, r, diagonal, state, target = _conditions(clean, noise, triangle, config)
    if not callable(getattr(model, "forward_with_auxiliary", None)):
        raise TypeError("the repaired pMF arm requires a deep auxiliary branch")
    divisor = t.clamp_min(config.denominator_floor)[:, None, None, None]
    with torch.no_grad():
        _, boundary_pixels = model.forward_with_auxiliary(state, t, torch.zeros_like(t))
        boundary = (state - boundary_pixels) / divisor

    def field(z_value, t_value, r_value):
        main_pixels, auxiliary_pixels = model.forward_with_auxiliary(
            z_value, t_value, t_value - r_value
        )
        local_divisor = t_value.clamp_min(config.denominator_floor)[:, None, None, None]
        return (z_value - main_pixels) / local_divisor, (
            z_value - auxiliary_pixels
        ) / local_divisor

    average, directional, auxiliary = torch.func.jvp(
        field,
        (state, t, r),
        (boundary, torch.ones_like(t), torch.zeros_like(r)),
        has_aux=True,
    )
    interval = (t - r)[:, None, None, None]
    compound = average + interval * directional.detach()
    residual = compound - target.detach()
    main_loss, _ = _adaptive_loss(residual, config)
    auxiliary_residual = auxiliary - target.detach()
    auxiliary_loss, auxiliary_per_sample = _adaptive_loss(auxiliary_residual, config)

    # Raw decomposition diagnostics, intentionally without adaptive weighting.
    tfm_loss = (average - target.detach()).square().mean()
    tc_loss = 2 * (interval * average * directional.detach()).mean()
    return _outcome(
        loss=main_loss + auxiliary_loss,
        residual=residual,
        diagonal=diagonal,
        jvp=directional,
        auxiliary_raw_mse=auxiliary_per_sample.mean(),
        tfm_loss=tfm_loss,
        tc_loss=tc_loss,
    )


def alpha_schedule(
    update: int, total_updates: int, config: S3RObjectiveConfig
) -> float:
    """JVP-free AlphaFlow curriculum, floored at the reported 0.005 optimum.

    This intentionally differs from the paper's final continuous phase: values
    below ``alpha_floor`` remain at the floor instead of becoming alpha=0.
    """
    if total_updates <= 0 or not 0 <= update <= total_updates:
        raise ValueError("update must lie in [0,total_updates]")
    start = config.alpha_schedule_start_fraction * total_updates
    end = config.alpha_schedule_end_fraction * total_updates
    normalized = (update - (start + end) / 2) / (end - start)
    value = (
        1
        - torch.sigmoid(
            torch.tensor(normalized * config.alpha_temperature, dtype=torch.float64)
        ).item()
    )
    if value > 1 - config.alpha_floor:
        return 1.0
    return max(config.alpha_floor, float(value))


def alpha_flow_loss(
    model: nn.Module,
    clean: torch.Tensor,
    noise: torch.Tensor,
    triangle: TriangleSample,
    config: S3RObjectiveConfig,
    alpha: float,
) -> ObjectiveOutcome:
    """Discrete AlphaFlow loss in the repository's data-at-zero convention."""
    if not config.alpha_floor <= alpha <= 1:
        raise ValueError("the developmental AlphaFlow arm requires floor <= alpha <= 1")
    t, r, diagonal, state, conditional = _conditions(clean, noise, triangle, config)
    current = velocity_from_pixels(model, state, t, r, config.denominator_floor)
    s = alpha * r + (1 - alpha) * t
    state_s = (1 - s[:, None, None, None]) * clean + s[:, None, None, None] * noise
    with torch.no_grad():
        future = velocity_from_pixels(model, state_s, s, r, config.denominator_floor)
        discrete_target = alpha * conditional + (1 - alpha) * future
        # In the released objective, r=t rows remain ordinary trajectory flow
        # matching for every curriculum value.  Only interior rows receive the
        # discrete AlphaFlow target.
        target = torch.where(
            diagonal[:, None, None, None], conditional, discrete_target
        )
    residual = current - target
    loss = _alpha_adaptive_loss(residual, diagonal, config, alpha)
    tfm_loss = (current - conditional.detach()).square().mean()
    return _outcome(
        loss=loss,
        residual=residual,
        diagonal=diagonal,
        tfm_loss=tfm_loss,
        alpha=float(alpha),
    )


def emf_local_difference(
    model: nn.Module,
    state: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    delta: float,
    denominator_floor: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current x1 field, stopped local future field, and quotient.

    Only intervals longer than ``delta`` are advanced.  Other rows have zero
    quotient and reduce exactly to endpoint regression.
    """
    interval = t - r
    active = interval > delta
    advance = active.to(t.dtype) * delta
    current = model(state, t, interval)
    with torch.no_grad():
        boundary = model(state, t, torch.zeros_like(t))
        # The EMF paper explicitly clamps its (1-t) and (1-r) denominators to
        # 0.02.  In this repository's reversed clock those are t and r.
        if bool((t <= 0).any()):
            raise ValueError("x1-EMF requires strictly positive t")
        divisor = t.clamp_min(denominator_floor)[:, None, None, None]
        future_state = (
            state + advance[:, None, None, None] * (boundary - state) / divisor
        )
        future_t = t - advance
        future_interval = interval - advance
        future = model(future_state, future_t, future_interval)
        quotient = (future - current.detach()) / delta
        quotient = quotient * active[:, None, None, None]
    return current, future, quotient


def emf_x1_loss(
    model: nn.Module,
    clean: torch.Tensor,
    noise: torch.Tensor,
    triangle: TriangleSample,
    config: S3RObjectiveConfig,
) -> ObjectiveOutcome:
    """Time-reversed form of the direct-x1 Euler Mean Flow Equation 18."""
    t, r, diagonal, state, _ = _conditions(clean, noise, triangle, config)
    if bool((r <= 0).any()):
        raise ValueError("x1-EMF Equation 18 requires strictly positive r")
    current, _, quotient = emf_local_difference(
        model,
        state,
        t,
        r,
        config.emf_delta,
        config.emf_denominator_floor,
    )
    coefficient = (
        (t - r - config.emf_delta).clamp_min(0)
        * t.clamp_min(config.emf_denominator_floor)
        / r.clamp_min(config.emf_denominator_floor)
    )
    target = clean + coefficient[:, None, None, None] * quotient
    residual = current - target.detach()
    # The x1-prediction/u-loss correction is 1/(1-t_paper)^2.  Under the
    # data-at-zero clock used here this becomes 1/t^2.  Apply it outside the
    # stopped adaptive denominator so it remains an actual time weight.
    time_weight = t.clamp_min(config.emf_denominator_floor).pow(-2)
    loss, _ = _adaptive_loss(residual, config, numerator=time_weight)
    return _outcome(loss=loss, residual=residual, diagonal=diagonal)


def one_step_sample(model: nn.Module, noise: torch.Tensor) -> torch.Tensor:
    if noise.ndim != 4:
        raise ValueError("noise must be an image batch")
    ones = torch.ones(len(noise), device=noise.device, dtype=noise.dtype)
    return model(noise, ones, ones)
