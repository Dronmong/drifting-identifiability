"""Direct-`x` Euler Mean Flow, JVP-free.

Ported from the audited S3R implementation.  The clock is this repository's
data-at-zero / noise-at-one convention, so the paper's ``t_paper = 1 - t``
substitution has already been applied: its ``1/(1-t_paper)^2`` x1/u loss weight
is ``1/t^2`` here, and its ``0.02`` clamps on ``1-t_paper`` and ``1-r_paper``
are clamps on local ``t`` and ``r``.

**The numbered equations, not the arXiv HTML pseudocode, are the source of
truth** — the rendered pseudocode contains transcription errors.  The preflight
checks this implementation against an exact forward-mode directional JVP of
the float32 network with TF32 disabled.
"""

from __future__ import annotations

from contextlib import contextmanager
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
    # Per-sample network evaluations actually performed. Not a constant: the
    # two stopped evaluations run only on active rows, so this is
    # ``batch + 2 * active`` rather than ``3 * batch``.
    model_evaluations: int
    # Actual objective-network invocations.  This differs from sample
    # evaluations because each batched model call may contain many rows.
    model_forward_calls: int
    # Sampled times, kept so training can bucket the error by t. The one-call
    # sampler runs at t=1, and logit-normal(0.8, 0.8) puts only ~4% of rows
    # above t=0.9 -- if the endpoint is undertrained this is where it shows.
    t: torch.Tensor
    r: torch.Tensor
    interval: torch.Tensor
    diagonal: torch.Tensor
    active: torch.Tensor
    coefficient: torch.Tensor
    adaptive_weight: torch.Tensor
    per_sample_weighted_loss: torch.Tensor
    per_sample_output_gradient_norm: torch.Tensor
    per_sample_target_rms: torch.Tensor
    per_sample_quotient_rms: torch.Tensor
    # Graph-carrying per-row objective, consumed immediately by sparse
    # parameter-gradient monitoring and never placed in a recovery artifact.
    per_sample_objective: torch.Tensor


def sample_time_triangle(
    batch: int,
    config: CAPObjectiveConfig,
    generator: torch.Generator,
    device: torch.device | str = "cpu",
    *,
    diagonal_generator: torch.Generator | None = None,
) -> TriangleSample:
    """Sample one of the declared time triangles.

    Diagonal rows (``r == t``) reduce the objective to plain endpoint
    regression, which is what makes short intervals well posed.  Ordered modes
    draw two iid endpoints and sort them, as prescribed by EMF.  The historical
    CAP mode is retained only as the matched control.
    """
    config.validate()
    if batch <= 0:
        raise ValueError("time sampler batch must be positive")
    if config.sampler_mode == "cap_conditional_logitnormal":
        normal = torch.randn(batch, generator=generator)
        first = torch.sigmoid(config.logit_mean + config.logit_std * normal)
        second = first * torch.rand(batch, generator=generator)
        t, r = first, second
    else:
        if config.sampler_mode == "ordered_logitnormal":
            draws = torch.sigmoid(
                config.logit_mean
                + config.logit_std * torch.randn((2, batch), generator=generator)
            )
        elif config.sampler_mode == "ordered_uniform":
            draws = torch.rand((2, batch), generator=generator)
        else:  # guarded by config.validate; defensive for foreign configs
            raise ValueError(f"unknown time sampler {config.sampler_mode!r}")
        first, second = draws.unbind(dim=0)
        t = torch.maximum(first, second)
        r = torch.minimum(first, second)

    if config.diagonal_sampling == "legacy_bernoulli":
        # Preserve CAP-EMF-1 byte-for-byte stochastic semantics.  In
        # particular, historical ordered experiments (if any) put diagonal
        # rows at the already-sorted upper endpoint.
        diagonal = torch.rand(batch, generator=generator) < config.diagonal_fraction
        r = torch.where(diagonal, t, r)
    else:
        if diagonal_generator is None:
            raise ValueError(
                "fixed-count diagonal sampling requires its own RNG stream"
            )
        count = round(batch * config.diagonal_fraction)
        diagonal = torch.zeros(batch, dtype=torch.bool)
        if count:
            positions = torch.randperm(batch, generator=diagonal_generator)[:count]
            diagonal[positions] = True
            # Released MeanFlow semantics: make the second draw equal to the
            # first *before* sorting.  Using the sorted maximum here changes
            # the diagonal marginal (uniform mean .5 -> 2/3) on half the batch.
            t = torch.where(diagonal, first, t)
            r = torch.where(diagonal, first, r)
    # The historical run clamped the *sampled endpoint*.  Repaired ordered
    # arms set this to zero and clamp only where r is used as a denominator.
    if config.sampled_r_floor > 0:
        r = r.clamp_min(config.sampled_r_floor)
    r = torch.minimum(r, t)
    return TriangleSample(t=t.to(device), r=r.to(device), diagonal=diagonal.to(device))


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


def _adaptive_terms(
    residual: torch.Tensor,
    config: CAPObjectiveConfig,
    numerator: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    per_sample_sum = residual.square().flatten(1).sum(dim=1)
    weights = numerator / (
        (per_sample_sum + config.adaptive_epsilon).pow(config.adaptive_power).detach()
    )
    weighted = weights * per_sample_sum
    # Exact residual-gradient norm of ``weighted.mean()``.  This is what the
    # retrospective tried to infer from raw MSE; with p=1 it decreases for
    # sufficiently large residuals instead of growing without bound.
    output_gradient_norm = (
        2.0 * weights * per_sample_sum.clamp_min(0).sqrt() / len(residual)
    )
    return weighted.mean(), weights, weighted, output_gradient_norm


@contextmanager
def _tf32_disabled(enabled: bool):
    """Disable TF32 around the stopped path while preserving global state."""
    if not enabled or not torch.cuda.is_available():
        yield
        return
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def emf_local_difference(
    model: nn.Module,
    state: torch.Tensor,
    t: torch.Tensor,
    r: torch.Tensor,
    delta: float,
    denominator_floor: float,
    evaluation_mode: str = "legacy_sparse",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Current field and the stopped local difference quotient.

    Only intervals longer than ``delta`` are advanced; the remaining rows carry
    a zero quotient and reduce exactly to endpoint regression.

    **The two stopped evaluations run on the active rows only.**  Inactive rows
    have their quotient multiplied by zero, so evaluating them was pure waste —
    and with ``diagonal_fraction = 0.5`` plus the short-interval rows, roughly
    half to sixty percent of every batch is inactive.

    The optimization is *mathematically* exact and is regression-tested against
    the dense path: inactive rows match bitwise, active rows to ~1e-13 relative
    in float64.  They cannot match bitwise, and never could — GEMM reduction
    order depends on batch shape, so the same rows evaluated alone round
    differently from the same rows evaluated inside a larger batch.
    """
    interval = t - r
    active = interval > delta
    current = model(state, t, interval)
    quotient = torch.zeros_like(current)
    with torch.no_grad():
        if bool((t < 0).any()):
            raise ValueError("direct-x EMF requires nonnegative t")
        if evaluation_mode == "legacy_sparse":
            index = active.nonzero(as_tuple=True)[0]
            if index.numel():
                sub_state = state[index]
                sub_t = t[index]
                sub_interval = interval[index]
                boundary = model(sub_state, sub_t, torch.zeros_like(sub_t))
                divisor = sub_t.clamp_min(denominator_floor)[:, None, None, None]
                future_state = sub_state + delta * (boundary - sub_state) / divisor
                future = model(future_state, sub_t - delta, sub_interval - delta)
                quotient[index] = (future - current.detach()[index]) / delta
        elif evaluation_mode in {"dense", "fp32_dense"}:
            # Matched full-batch shapes prevent the current/future subtraction
            # from inheriting batch-shape-dependent GEMM rounding.  In FP32
            # mode a separate stopped current is evaluated under the same
            # precision as the future value.
            advance = active.to(t.dtype) * delta
            with _tf32_disabled(evaluation_mode == "fp32_dense"):
                stopped_current = (
                    model(state, t, interval)
                    if evaluation_mode == "fp32_dense"
                    else current.detach()
                )
                boundary = model(state, t, torch.zeros_like(t))
                divisor = t.clamp_min(denominator_floor)[:, None, None, None]
                future_state = (
                    state + advance[:, None, None, None] * (boundary - state) / divisor
                )
                future = model(future_state, t - advance, interval - advance)
                quotient = (future - stopped_current) / delta
                quotient = quotient * active[:, None, None, None]
        else:
            raise ValueError(f"unknown stopped evaluation mode {evaluation_mode!r}")
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
    if bool((r < 0).any()):
        raise ValueError("Equation 18 requires nonnegative r")
    active = int(((t - r) > config.emf_delta).sum())
    current, quotient = emf_local_difference(
        model,
        state,
        t,
        r,
        config.emf_delta,
        config.emf_denominator_floor,
        config.stopped_evaluation,
    )
    coefficient = (
        (t - r - config.emf_delta).clamp_min(0)
        # In paper time the factor is (1-t)/(1-r).  Reversing the clock makes
        # this t/r: r is the denominator and is clamped for stability, while t
        # remains the mathematical numerator.  Clamping t here would
        # overweight the correction on the below-floor rows that already carry
        # the largest 1/t^2 regression weight.
        * t
        / r.clamp_min(config.resolved_coefficient_floor)
    )
    target = clean + coefficient[:, None, None, None] * quotient
    residual = current - target.detach()
    # Applied outside the stopped adaptive denominator so it stays an actual
    # time weight rather than being cancelled by the normalizer.
    time_weight = t.clamp_min(config.resolved_loss_weight_floor).pow(-2)
    loss, adaptive_weight, weighted_loss, output_gradient_norm = _adaptive_terms(
        residual, config, numerator=time_weight
    )
    per_sample = residual.square().flatten(1).mean(dim=1)
    if config.stopped_evaluation == "legacy_sparse":
        evaluations = len(clean) + 2 * active
        forward_calls = 1 + 2 * int(active > 0)
    elif config.stopped_evaluation == "dense":
        evaluations = 3 * len(clean)
        forward_calls = 3
    else:
        evaluations = 4 * len(clean)
        forward_calls = 4
    return ObjectiveOutcome(
        loss=loss,
        raw_mse=per_sample.mean(),
        per_sample_raw_mse=per_sample.detach(),
        diagonal_raw_mse=_masked_mean(per_sample, diagonal),
        interior_raw_mse=_masked_mean(per_sample, ~diagonal),
        model_evaluations=evaluations,
        model_forward_calls=forward_calls,
        t=t.detach(),
        r=r.detach(),
        interval=(t - r).detach(),
        diagonal=diagonal.detach(),
        active=((t - r) > config.emf_delta).detach(),
        coefficient=coefficient.detach(),
        adaptive_weight=adaptive_weight.detach(),
        per_sample_weighted_loss=weighted_loss.detach(),
        per_sample_output_gradient_norm=output_gradient_norm.detach(),
        per_sample_target_rms=target.detach().square().flatten(1).mean(1).sqrt(),
        per_sample_quotient_rms=quotient.detach().square().flatten(1).mean(1).sqrt(),
        per_sample_objective=weighted_loss,
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
    ``t`` and the interval retreat at unit rate.  Used only by the preflight
    with TF32 disabled to bound the finite-difference error; the derivative is
    exact forward-mode AD of the float32 network, not a float64 re-evaluation.
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
