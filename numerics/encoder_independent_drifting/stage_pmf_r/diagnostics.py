"""Train-only diagnostics that detect the S3 collapse before test evaluation."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import nn

from .objectives import ObjectiveOutcome, one_step_sample, pmf_loss


def effective_rank(samples: torch.Tensor) -> float:
    """Participation-ratio rank of centered samples (bounded by batch-1)."""
    if samples.ndim < 2 or len(samples) < 2:
        raise ValueError("effective rank requires at least two samples")
    flat = samples.detach().double().flatten(1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    singular_squared = torch.linalg.svdvals(centered).square()
    total = singular_squared.sum()
    if float(total) <= 0:
        return 0.0
    return float(total.square() / singular_squared.square().sum().clamp_min(1e-30))


def haar_subbands(images: torch.Tensor) -> dict[str, torch.Tensor]:
    """One-level orthonormal 2-D Haar decomposition with exact energy balance."""
    if images.ndim != 4 or images.shape[-1] % 2 or images.shape[-2] % 2:
        raise ValueError("Haar input must be BCHW with even spatial dimensions")
    x00 = images[..., 0::2, 0::2]
    x01 = images[..., 0::2, 1::2]
    x10 = images[..., 1::2, 0::2]
    x11 = images[..., 1::2, 1::2]
    return {
        "ll": (x00 + x01 + x10 + x11) / 2,
        "lh": (x00 - x01 + x10 - x11) / 2,
        "hl": (x00 + x01 - x10 - x11) / 2,
        "hh": (x00 - x01 - x10 + x11) / 2,
    }


def _moments(samples: torch.Tensor) -> tuple[float, float]:
    flat = samples.detach().double().flatten(1)
    second = float(flat.square().mean())
    centered = flat - flat.mean(dim=0, keepdim=True)
    variance = float(centered.square().mean())
    return second, variance


def endpoint_health(generated: torch.Tensor, target: torch.Tensor) -> dict:
    """Amplitude-aware endpoint and Haar-band collapse report."""
    if generated.shape[1:] != target.shape[1:]:
        raise ValueError("generated and target images must have matching shapes")
    generated_second, generated_variance = _moments(generated)
    target_second, target_variance = _moments(target)
    generated_rank = effective_rank(generated)
    target_rank = effective_rank(target)
    bands: dict[str, dict] = {}
    generated_bands = haar_subbands(generated)
    target_bands = haar_subbands(target)
    for name in generated_bands:
        g_second, g_variance = _moments(generated_bands[name])
        t_second, t_variance = _moments(target_bands[name])
        bands[name] = {
            "second_moment_ratio": g_second / max(t_second, 1e-30),
            "variance_ratio": g_variance / max(t_variance, 1e-30),
            "effective_rank": effective_rank(generated_bands[name]),
            "target_effective_rank": effective_rank(target_bands[name]),
        }
    moment_ratio = generated_second / max(target_second, 1e-30)
    variance_ratio = generated_variance / max(target_variance, 1e-30)
    return {
        "samples": len(generated),
        "finite": bool(torch.isfinite(generated).all()),
        "second_moment_ratio": moment_ratio,
        "variance_ratio": variance_ratio,
        "effective_rank": generated_rank,
        "target_effective_rank": target_rank,
        "effective_rank_ratio": generated_rank / max(target_rank, 1e-30),
        # A nonzero constant output can have healthy second moment but zero
        # diversity.  Rank is meaningful only after centered variance itself
        # clears the amplitude floor.
        "rank_interpretable": bool(variance_ratio > 0.15),
        "haar": bands,
    }


def tensor_quantiles(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().double().flatten().cpu()
    if flat.numel() == 0:
        return {"p50": 0.0, "p90": 0.0, "max": 0.0}
    return {
        "p50": float(torch.quantile(flat, 0.5)),
        "p90": float(torch.quantile(flat, 0.9)),
        "max": float(flat.max()),
    }


def objective_health(outcome: ObjectiveOutcome) -> dict:
    return {
        "raw_mse": float(outcome.raw_mse.detach()),
        "diagonal_raw_mse": float(outcome.diagonal_raw_mse.detach()),
        "interior_raw_mse": float(outcome.interior_raw_mse.detach()),
        "jvp_rms": tensor_quantiles(outcome.jvp_per_sample_rms),
        "sample_raw_mse": tensor_quantiles(outcome.per_sample_raw_mse),
        "auxiliary_raw_mse": float(outcome.auxiliary_raw_mse.detach()),
        "alpha": outcome.alpha,
    }


def gradient_cosine_from_losses(
    first: torch.Tensor, second: torch.Tensor, parameters: list[nn.Parameter]
) -> dict[str, float]:
    """Cosine and norms without concatenating a multi-million-vector gradient."""
    first_grads = torch.autograd.grad(
        first, parameters, retain_graph=True, allow_unused=True
    )
    second_grads = torch.autograd.grad(
        second, parameters, retain_graph=False, allow_unused=True
    )
    dot = torch.zeros((), device=first.device, dtype=torch.float64)
    first_sq = torch.zeros_like(dot)
    second_sq = torch.zeros_like(dot)
    for left, right in zip(first_grads, second_grads):
        if left is not None:
            first_sq += left.detach().double().square().sum()
        if right is not None:
            second_sq += right.detach().double().square().sum()
        if left is not None and right is not None:
            dot += (left.detach().double() * right.detach().double()).sum()
    first_norm = first_sq.sqrt()
    second_norm = second_sq.sqrt()
    cosine = dot / (first_norm * second_norm).clamp_min(1e-30)
    return {
        "cosine": float(cosine),
        "first_norm": float(first_norm),
        "second_norm": float(second_norm),
    }


def pmf_gradient_conflict(
    model: nn.Module,
    clean: torch.Tensor,
    noise: torch.Tensor,
    triangle,
    config,
) -> dict[str, float]:
    model.zero_grad(set_to_none=True)
    outcome = pmf_loss(model, clean, noise, triangle, config)
    if outcome.tfm_loss is None or outcome.tc_loss is None:
        raise RuntimeError("pMF decomposition losses were not produced")
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    result = gradient_cosine_from_losses(outcome.tfm_loss, outcome.tc_loss, parameters)
    model.zero_grad(set_to_none=True)
    return result


@contextlib.contextmanager
def swapped_state(model: nn.Module, state: dict[str, torch.Tensor]) -> Iterator[None]:
    original = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    model.load_state_dict(state, strict=True)
    try:
        yield
    finally:
        model.load_state_dict(original, strict=True)


@torch.no_grad()
def raw_and_ema_health(
    model: nn.Module,
    ema_state: dict[str, torch.Tensor],
    noise: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, dict]:
    """Use identical noise and target references for raw/EMA comparison."""
    was_training = model.training
    model.eval()
    raw = one_step_sample(model, noise)
    raw_report = endpoint_health(raw, target)
    with swapped_state(model, ema_state):
        model.eval()
        ema = one_step_sample(model, noise)
        ema_report = endpoint_health(ema, target)
    model.train(was_training)
    return {"raw": raw_report, "ema": ema_report}


def developmental_gate(report: dict, clipping_fraction: float) -> dict:
    """Outcome-blind health gate; never a substitute for sealed image metrics."""
    raw = report["raw"]
    ema = report["ema"]
    moment_ok = min(
        raw["second_moment_ratio"],
        ema["second_moment_ratio"],
        raw["variance_ratio"],
        ema["variance_ratio"],
    ) >= 0.5
    rank_ok = bool(
        raw["rank_interpretable"]
        and ema["rank_interpretable"]
        and raw["effective_rank_ratio"] >= 0.6
        and ema["effective_rank_ratio"] >= 0.6
    )
    clip_ok = clipping_fraction < 0.05
    return {
        "passes_current_snapshot": bool(moment_ok and rank_ok and clip_ok),
        "moment_ok": moment_ok,
        "rank_ok": rank_ok,
        "clip_ok": clip_ok,
        "warning": "a passing train-only health snapshot does not establish image quality",
    }


def developmental_series_gate(
    endpoint_history: list[dict], clipping_fraction: float
) -> dict:
    """Reject a final snapshot that merely follows an earlier rank collapse."""
    if not endpoint_history:
        return {"passes": False, "reason": "no endpoint-health snapshots"}
    final = developmental_gate(endpoint_history[-1], clipping_fraction)
    retention = {}
    retention_ok = True
    for state in ("raw", "ema"):
        interpretable = [
            row[state]["effective_rank_ratio"]
            for row in endpoint_history
            if row[state]["rank_interpretable"]
        ]
        if not interpretable:
            retention[state] = None
            retention_ok = False
            continue
        ratio = interpretable[-1] / max(max(interpretable), 1e-30)
        retention[state] = ratio
        retention_ok = retention_ok and ratio >= 0.8
    return {
        "passes": bool(final["passes_current_snapshot"] and retention_ok),
        "final_snapshot": final,
        "rank_retention_from_best_interpretable": retention,
        "rank_retention_ok": retention_ok,
        "warning": "train-only development gate; sealed image evaluation remains required",
    }
