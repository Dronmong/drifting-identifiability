"""Mechanism library for Large-Batch Quantile-Calibrated Drifting.

This module is development infrastructure, not a frozen confirmatory runner.
It deliberately leaves the sealed QLD-v1 implementation untouched.

The Run-Sort-ReRun (RSR) update has one essential invariant: all virtual-batch
matches are formed before backpropagation and all microbatch gradients are
summed into *one* Adam update.  Applying Adam once per microbatch would change
the optimization problem and is not RSR.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Iterable

import numpy as np

from identifiability_drift import compute_field
from run_identifiability_generator import (
    ADAM_EPS,
    ADAM_LR,
    BETA1,
    BETA2,
    TanhMLP,
)


@dataclass(frozen=True)
class StepWork:
    """Training-work ledger for one optimizer update."""

    optimizer_updates: int = 1
    generator_forward_calls: int = 0
    generator_example_evals: int = 0
    unique_latent_samples: int = 0
    target_samples: int = 0
    kernel_pairs: int = 0
    sort_work: float = 0.0


@dataclass(frozen=True)
class TauSelection:
    tau: float
    scores: dict[float, float]
    kernel_pairs: int
    generator_example_evals: int


def exact_rank_field(x: np.ndarray, target: np.ndarray,
                     latent: np.ndarray) -> np.ndarray:
    """One-dimensional empirical monotone-transport displacement."""
    if x.shape != target.shape or x.ndim != 2 or x.shape[1] != 1:
        raise ValueError("rank field requires equal (N, 1) arrays")
    if len(latent) != len(x):
        raise ValueError("one latent tie-break value is required per sample")
    tie = latent[:, 0]
    tie = (tie - tie.mean()) / max(float(tie.std()), 1e-12)
    ox = np.argsort(x[:, 0] + 1e-10 * tie, kind="stable")
    oy = np.argsort(target[:, 0], kind="stable")
    field = np.zeros_like(x)
    field[ox, 0] = target[oy, 0] - x[ox, 0]
    return field


def _stopgrad_grads(model: TanhMLP, cache: tuple[np.ndarray, ...],
                    field: np.ndarray, denominator: int) \
        -> dict[str, np.ndarray]:
    """Gradient contribution for ``-sum <G(z), field> / denominator``."""
    if denominator <= 0:
        raise ValueError("gradient denominator must be positive")
    z, h1, h2 = cache
    p = model.params
    dx = -field / denominator
    grads: dict[str, np.ndarray] = {}
    grads["W3"] = h2.T @ dx
    grads["b3"] = dx.sum(axis=0)
    dh2 = (dx @ p["W3"].T) * (1.0 - h2 * h2)
    grads["W2"] = h1.T @ dh2
    grads["b2"] = dh2.sum(axis=0)
    dh1 = (dh2 @ p["W2"].T) * (1.0 - h1 * h1)
    grads["W1"] = z.T @ dh1
    grads["b1"] = dh1.sum(axis=0)
    if not all(np.all(np.isfinite(g)) for g in grads.values()):
        raise FloatingPointError("non-finite accumulated generator gradient")
    return grads


def _sum_grads(model: TanhMLP,
               contributions: Iterable[dict[str, np.ndarray]]) \
        -> dict[str, np.ndarray]:
    total = {name: np.zeros_like(model.params[name]) for name in model.names}
    for contribution in contributions:
        for name in model.names:
            total[name] += contribution[name]
    return total


def _adam_step(model: TanhMLP, grads: dict[str, np.ndarray]) -> None:
    """Apply exactly one update using the optimizer owned by ``model``."""
    if not all(np.all(np.isfinite(grads[name])) for name in model.names):
        raise FloatingPointError("non-finite generator gradient")
    model.step_index += 1
    t = model.step_index
    for name in model.names:
        g = grads[name]
        model.m[name] = BETA1 * model.m[name] + (1.0 - BETA1) * g
        model.v[name] = BETA2 * model.v[name] + (1.0 - BETA2) * g * g
        mhat = model.m[name] / (1.0 - BETA1 ** t)
        vhat = model.v[name] / (1.0 - BETA2 ** t)
        model.params[name] -= ADAM_LR * mhat / (
            np.sqrt(vhat) + ADAM_EPS)


def direct_quantile_step(model: TanhMLP, latent: np.ndarray,
                         target: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, StepWork]:
    """Original QLD minibatch update, used as the exact compatibility arm."""
    x, cache = model.forward(latent, want_cache=True)
    field = exact_rank_field(x, target, latent)
    model.stopgrad_step(cache, field)
    n = len(latent)
    return x, field, StepWork(
        generator_forward_calls=1,
        generator_example_evals=n,
        unique_latent_samples=n,
        target_samples=n,
        sort_work=2.0 * n * math.log2(max(n, 2)),
    )


def rsr_quantile_step(model: TanhMLP, latent: np.ndarray,
                      target: np.ndarray, microbatch: int) \
        -> tuple[np.ndarray, np.ndarray, StepWork]:
    """Run-Sort-ReRun rank update with one virtual-batch Adam step.

    The first pass stores only outputs.  After global sorting, the second pass
    reconstructs activations one microbatch at a time.  Every contribution is
    normalized by the full virtual batch size and accumulated before Adam.
    """
    n = len(latent)
    if target.shape != (n, 1):
        raise ValueError("RSR target must have shape (len(latent), 1)")
    if microbatch <= 0:
        raise ValueError("microbatch must be positive")

    outputs: list[np.ndarray] = []
    for start in range(0, n, microbatch):
        outputs.append(model.forward(latent[start:start + microbatch]))
    x = np.concatenate(outputs, axis=0)
    field = exact_rank_field(x, target, latent)

    contributions: list[dict[str, np.ndarray]] = []
    for start in range(0, n, microbatch):
        stop = min(start + microbatch, n)
        x_again, cache = model.forward(latent[start:stop], want_cache=True)
        if not np.array_equal(x_again, x[start:stop]):
            raise AssertionError("RSR rerun changed outputs before the update")
        contributions.append(
            _stopgrad_grads(model, cache, field[start:stop], n))
    _adam_step(model, _sum_grads(model, contributions))

    chunks = math.ceil(n / microbatch)
    return x, field, StepWork(
        generator_forward_calls=2 * chunks,
        generator_example_evals=2 * n,
        unique_latent_samples=n,
        target_samples=n,
        sort_work=2.0 * n * math.log2(max(n, 2)),
    )


def paper_step(model: TanhMLP, latent: np.ndarray, target: np.ndarray,
               tau: float) -> tuple[np.ndarray, np.ndarray, StepWork]:
    """One exact paper Algorithm-2 generator update."""
    x, cache = model.forward(latent, want_cache=True)
    result = compute_field(
        x, target, tau=tau, gain="paper", mask=True,
        on_degenerate="zero")
    model.stopgrad_step(cache, result.V)
    n = len(latent)
    return x, result.V, StepWork(
        generator_forward_calls=1,
        generator_example_evals=n,
        unique_latent_samples=n,
        target_samples=n,
        kernel_pairs=result.kernel_pairs,
    )


def select_tau_by_alignment(model: TanhMLP, latent: np.ndarray,
                            target: np.ndarray,
                            taus: Iterable[float]) -> TauSelection:
    """Select the paper bandwidth aligned with held-out rank displacement.

    Candidate fields are compared by cosine alignment, so the selector is
    insensitive to the paper field's bandwidth-dependent scalar magnitude.
    The probe samples must be independent of the subsequent update batch.
    """
    x = model.forward(latent)
    rank = exact_rank_field(x, target, latent)
    rank_norm = float(np.linalg.norm(rank))
    scores: dict[float, float] = {}
    kernel_pairs = 0
    for raw_tau in taus:
        tau = float(raw_tau)
        if tau <= 0:
            raise ValueError("candidate bandwidths must be positive")
        result = compute_field(
            x, target, tau=tau, gain="paper", mask=True,
            on_degenerate="zero")
        kernel_pairs += result.kernel_pairs
        field_norm = float(np.linalg.norm(result.V))
        denom = rank_norm * field_norm
        scores[tau] = (-math.inf if denom <= 0 else
                       float(np.sum(rank * result.V) / denom))
    if not scores:
        raise ValueError("at least one bandwidth candidate is required")
    # Stable deterministic tie-break: prefer the smaller bandwidth.
    chosen = max(sorted(scores), key=lambda tau: scores[tau])
    return TauSelection(
        tau=chosen,
        scores=scores,
        kernel_pairs=kernel_pairs,
        generator_example_evals=len(latent),
    )


def example_matched_updates(reference_updates: int, batch: int,
                            virtual_batch: int,
                            warm_fraction: float) -> int:
    """Update count matching generator-example evaluations to QLD-v1.

    An RSR warm update evaluates the generator twice on ``virtual_batch``
    samples.  A paper refinement update evaluates it once on ``batch``.
    """
    if reference_updates <= 0 or batch <= 0 or virtual_batch <= 0:
        raise ValueError("update and batch counts must be positive")
    if not 0.0 <= warm_fraction <= 1.0:
        raise ValueError("warm_fraction must lie in [0, 1]")
    per_update = (warm_fraction * 2.0 * virtual_batch +
                  (1.0 - warm_fraction) * batch)
    return max(1, int(round(reference_updates * batch / per_update)))


def invariant_tests() -> None:
    """Fast algebraic and optimizer regression checks."""
    target_spec = SimpleNamespace(d=1, scale=2.0,
                                  means=np.asarray([[-1.0], [1.0]]))
    rng = np.random.default_rng(413)
    latent = rng.normal(size=(24, 2))
    target = rng.normal(size=(24, 1))

    direct = TanhMLP(target_spec, "concentrated", 99)
    rerun = deepcopy(direct)
    x_direct, field_direct, _ = direct_quantile_step(
        direct, latent, target)
    x_rerun, field_rerun, _ = rsr_quantile_step(
        rerun, latent, target, microbatch=len(latent))
    if not np.array_equal(x_direct, x_rerun):
        raise AssertionError("M=batch RSR forward mismatch")
    if not np.array_equal(field_direct, field_rerun):
        raise AssertionError("M=batch RSR field mismatch")
    for name in direct.names:
        if not np.array_equal(direct.params[name], rerun.params[name]):
            raise AssertionError(f"M=batch RSR optimizer mismatch in {name}")

    full = TanhMLP(target_spec, "missing", 101)
    chunked = deepcopy(full)
    rsr_quantile_step(full, latent, target, microbatch=len(latent))
    rsr_quantile_step(chunked, latent, target, microbatch=5)
    for name in full.names:
        if not np.allclose(full.params[name], chunked.params[name],
                           rtol=2e-12, atol=2e-12):
            raise AssertionError(f"RSR microbatch mismatch in {name}")

    if example_matched_updates(1200, 128, 512, 0.70) != 203:
        raise AssertionError("example-matched update calculation changed")
