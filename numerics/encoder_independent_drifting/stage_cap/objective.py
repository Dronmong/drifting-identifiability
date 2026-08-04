"""Direct-`x` Euler Mean Flow, JVP-free.

Ported from the audited S3R implementation.  The clock is this repository's
data-at-zero / noise-at-one convention, so the paper's ``t_paper = 1 - t``
substitution has already been applied: its ``1/(1-t_paper)^2`` x1/u loss weight
is ``1/t^2`` here, and its ``0.02`` clamps on ``1-t_paper`` and ``1-r_paper``
are clamps on local ``t`` and ``r``.

**The numbered equations, not the arXiv HTML pseudocode, are the source of
truth** — the rendered pseudocode contains transcription errors.  The preflight
checks this implementation against a directional JVP in float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .config import CAPObjectiveConfig


@dataclass(frozen=True)
class TriangleSample:
    t: torch.Tensor
    r: torch.Tensor
    diagonal: torch.Tensor


@dataclass
class ObjectiveOutcome:
    loss: torch.Tensor
    raw_mse: torch.Tensor
    per_sample_raw_mse: torch.Tensor
    diagonal_raw_mse: torch.Tensor
    interior_raw_mse: torch.Tensor


def sample_time_triangle(
    batch: int,
    config: CAPObjectiveConfig,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
) -> TriangleSample:
    """Logit-normal ``t`` with a fixed fraction of exactly diagonal rows.

    Diagonal rows (``r == t``) reduce the objective to plain endpoint
    regression, which is what makes short intervals well posed.
    """
    config.validate()
    normal = torch.randn(batch, generator=generator)
    t = torch.sigmoid(config.logit_mean + config.logit_std * normal)
    share = torch.rand(batch, generator=generator)
    r = t * share
    diagonal = torch.rand(batch, generator=generator) < config.diagonal_fraction
    r = torch.where(diagonal, t, r)
    # r is a denominator in Equation 18; keep it strictly positive.
    r = r.clamp_min(config.emf_denominator_floor * 0.5)
    r = torch.minimum(r, t)
    return TriangleSample(
        t=t.to(device), r=r.to(device), diagonal=diagonal.to(device)
    )


def _conditions(
    clean: torch.Tensor, noise: torch.Tensor, triangle: TriangleSample
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if clean.shape != noise.shape or clean.ndim != 4:
        raise ValueError("clean and noise must be equal-shaped image batches")
    if len(clean) != len(triangle.t) or len(clean) != len(triangle.r):
        raise ValueError("triangle and image batches differ")
    t = triangle.t.to(device=clean.device, dtype=clean.dtype)
    r = triangle.r.to(device=clean.device, dtype=clean.dtype)
    diagonal = triangle.diagonal.to(device=clean.device)
    state = (1 - t[:, None, None, None]) * clean + t[:, None, None, None] * noise
    return t, r, diagonal, state


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return torch.zeros((), device=values.device, dtype=values.dtype)
    return selected.mean()


def _adaptive_loss(
    residual: torch.Tensor,
    config: CAPObjectiveConfig,
    numerator: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    per_sample_sum = residual.square().flatten(1).sum(dim=1)
    weights = numerator / (
        (per_sample_sum + config.adaptive_epsilon).pow(config.adaptive_power).detach()
    )
    return (weights * per_sample_sum).mean()


def emf_local_difference(
    model: nn.Module,
    state: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    delta: float,
    denominator_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Current field and the stopped local difference quotient.

    Only intervals longer than ``delta`` are advanced; the remaining rows carry
    a zero quotient and reduce exactly to endpoint regression.
    """
    interval = t - r
    active = interval > delta
    advance = active.to(t.dtype) * delta
    current = model(state, t, interval)
    with torch.no_grad():
        if bool((t <= 0).any()):
            raise ValueError("direct-x EMF requires strictly positive t")
        boundary = model(state, t, torch.zeros_like(t))
        divisor = t.clamp_min(denominator_floor)[:, None, None, None]
        future_state = (
            state + advance[:, None, None, None] * (boundary - state) / divisor
        )
        future = model(future_state, t - advance, interval - advance)
        quotient = (future - current.detach()) / delta
        quotient = quotient * active[:, None, None, None]
    return current, quotient


def emf_loss(
    model: nn.Module,
    clean: torch.Tensor,
    noise: torch.Tensor,
    triangle: TriangleSample,
    config: CAPObjectiveConfig,
) -> ObjectiveOutcome:
    """Time-reversed direct-`x` Euler Mean Flow, Equation 18."""
    config.validate()
    t, r, diagonal, state = _conditions(clean, noise, triangle)
    if bool((r <= 0).any()):
        raise ValueError("Equation 18 requires strictly positive r")
    current, quotient = emf_local_difference(
        model, state, t, r, config.emf_delta, config.emf_denominator_floor
    )
    coefficient = (
        (t - r - config.emf_delta).clamp_min(0)
        * t.clamp_min(config.emf_denominator_floor)
        / r.clamp_min(config.emf_denominator_floor)
    )
    target = clean + coefficient[:, None, None, None] * quotient
    residual = current - target.detach()
    # Applied outside the stopped adaptive denominator so it stays an actual
    # time weight rather than being cancelled by the normalizer.
    time_weight = t.clamp_min(config.emf_denominator_floor).pow(-2)
    loss = _adaptive_loss(residual, config, numerator=time_weight)
    per_sample = residual.square().flatten(1).mean(dim=1)
    return ObjectiveOutcome(
        loss=loss,
        raw_mse=per_sample.mean(),
        per_sample_raw_mse=per_sample.detach(),
        diagonal_raw_mse=_masked_mean(per_sample, diagonal),
        interior_raw_mse=_masked_mean(per_sample, ~diagonal),
    )


def directional_jvp_reference(
    model: nn.Module,
    state: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    denominator_floor: float,
) -> torch.Tensor:
    """The exact derivative the local difference approximates.

    ``emf_local_difference`` steps to ``(z + δ·(x̂(z,t,0) − z)/t, t − δ, h − δ)``
    and divides by ``δ``.  The exact limit is therefore the directional
    derivative along ``(+velocity, −1, −1)`` — the state advances while both
    ``t`` and the interval retreat at unit rate.  Used only by the preflight,
    in float64, to bound the finite-difference error.
    """
    interval = t - r
    with torch.no_grad():
        boundary = model(state, t, torch.zeros_like(t))
        divisor = t.clamp_min(denominator_floor)[:, None, None, None]
        velocity = (boundary - state) / divisor

    def field(z_value, t_value, h_value):
        return model(z_value, t_value, h_value)

    _, tangent = torch.func.jvp(
        field,
        (state, t, interval),
        (velocity, -torch.ones_like(t), -torch.ones_like(interval)),
    )
    return tangent
