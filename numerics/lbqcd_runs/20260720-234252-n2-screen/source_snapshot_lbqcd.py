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


@dataclass(frozen=True)
class QuantileResolutionDiagnosis:
    use_large_batch: bool
    segment_masses: tuple[float, ...]
    minimum_expected_batch_count: float
    significant_gap_count: int
    maximum_gap_ratio: float


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


def diagnose_quantile_resolution(
        target: np.ndarray, training_batch: int, *,
        minimum_expected_count: float = 8.0,
        gap_multiplier: float = 20.0,
        minimum_relative_gap: float = 0.015,
        cohesion_multiplier: float = 1.5,
        trim_fraction: float = 0.01) -> QuantileResolutionDiagnosis:
    """Detect separated target regions under-resolved by a training batch.

    This is a target-only routing diagnostic, not an oracle component label.
    Large interior spacings split the empirical quantile function into
    separated regions.  RSR is requested only when the smallest such region
    has fewer than ``minimum_expected_count`` expected representatives in an
    ordinary training batch.  Tail spacings are trimmed so a connected
    heavy-tailed law is not mistaken for a rare separated component.
    """
    values = np.asarray(target, dtype=float)
    if values.ndim != 2 or values.shape[1] != 1 or len(values) < 32:
        raise ValueError("diagnosis requires at least 32 one-dimensional samples")
    if training_batch <= 0 or minimum_expected_count <= 0:
        raise ValueError("batch and expected-count threshold must be positive")
    if (gap_multiplier <= 1 or minimum_relative_gap <= 0 or
            cohesion_multiplier <= 0 or
            not 0.0 <= trim_fraction < 0.25):
        raise ValueError("invalid gap diagnostic settings")
    ordered = np.sort(values[:, 0])
    gaps = np.diff(ordered)
    n = len(ordered)
    lo = max(1, int(math.floor(trim_fraction * n)))
    hi = min(n - 2, int(math.ceil((1.0 - trim_fraction) * n)))
    interior = gaps[lo:hi]
    positive = interior[interior > 0]
    baseline = float(np.median(positive)) if len(positive) else 0.0
    if baseline <= 0 or not math.isfinite(baseline):
        return QuantileResolutionDiagnosis(
            False, (1.0,), float(training_batch), 0, 0.0)
    ratios = gaps / baseline
    central_span = max(float(ordered[hi] - ordered[lo]), baseline)
    candidate = np.flatnonzero(
        (ratios >= gap_multiplier) &
        (gaps >= minimum_relative_gap * central_span))
    candidate = candidate[(candidate >= lo) & (candidate < hi)]
    # A real separated region has a gap larger than the local span of a small
    # neighborhood on *both* sides.  This rejects isolated order-statistic
    # gaps in connected heavy tails while retaining compact rare components.
    window = max(8, int(math.ceil(0.002 * n)))
    cohesive: list[int] = []
    for raw_i in candidate:
        i = int(raw_i)
        if i - window + 1 < 0 or i + window >= n:
            continue
        left_span = float(ordered[i] - ordered[i - window + 1])
        right_span = float(ordered[i + window] - ordered[i + 1])
        if gaps[i] >= cohesion_multiplier * max(left_span, right_span):
            cohesive.append(i)
    candidate = np.asarray(cohesive, dtype=int)
    # A spacing after index i creates a segment boundary at i+1.
    boundaries = [0, *(int(i) + 1 for i in candidate), n]
    masses = tuple((b - a) / n for a, b in zip(boundaries, boundaries[1:]))
    min_expected = training_batch * min(masses)
    max_ratio = float(np.max(ratios[candidate])) if len(candidate) else 0.0
    return QuantileResolutionDiagnosis(
        use_large_batch=(len(candidate) > 0 and
                         min_expected < minimum_expected_count),
        segment_masses=masses,
        minimum_expected_batch_count=float(min_expected),
        significant_gap_count=len(candidate),
        maximum_gap_ratio=max_ratio,
    )


def diagnose_quantile_resolution_stable(
        target: np.ndarray, training_batch: int, **kwargs) \
        -> QuantileResolutionDiagnosis:
    """Require the under-resolution diagnosis on two disjoint half-samples.

    The intersection rule suppresses routing caused by a single accidental
    heavy-tail spacing while retaining reproducible separated rare regions.
    """
    values = np.asarray(target)
    if len(values) < 64:
        raise ValueError("stable diagnosis requires at least 64 samples")
    left = diagnose_quantile_resolution(
        values[0::2], training_batch, **kwargs)
    right = diagnose_quantile_resolution(
        values[1::2], training_batch, **kwargs)
    return QuantileResolutionDiagnosis(
        use_large_batch=(left.use_large_batch and right.use_large_batch),
        segment_masses=left.segment_masses,
        minimum_expected_batch_count=max(
            left.minimum_expected_batch_count,
            right.minimum_expected_batch_count),
        significant_gap_count=min(
            left.significant_gap_count,
            right.significant_gap_count),
        maximum_gap_ratio=min(left.maximum_gap_ratio,
                              right.maximum_gap_ratio),
    )


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


def noise_restored_rsr_step(
        model: TanhMLP, latent: np.ndarray, target: np.ndarray,
        microbatch: int, gradient_batch: int, noise_mix: float,
        rng: np.random.Generator) \
        -> tuple[np.ndarray, np.ndarray, StepWork]:
    """RSR with a mean-zero minibatch fluctuation added to the full gradient.

    Let ``G_M`` be the full virtual-batch gradient and ``G_B`` the gradient of
    a uniformly sampled subset, both using the same globally matched pairs.
    This applies ``(1-lambda) G_M + lambda G_B``.  Conditional on the virtual
    batch, ``E[G_B] = G_M``; hence the added fluctuation is mean zero.  The
    global term retains rare-rank information even when the subset misses it.
    """
    n = len(latent)
    if not 0.0 <= noise_mix <= 1.0:
        raise ValueError("noise_mix must lie in [0, 1]")
    if gradient_batch <= 0 or gradient_batch > n:
        raise ValueError("gradient_batch must lie in [1, virtual batch]")
    if target.shape != (n, 1):
        raise ValueError("RSR target must have shape (len(latent), 1)")
    if microbatch <= 0:
        raise ValueError("microbatch must be positive")

    outputs = [model.forward(latent[start:start + microbatch])
               for start in range(0, n, microbatch)]
    x = np.concatenate(outputs, axis=0)
    field = exact_rank_field(x, target, latent)
    selected = np.zeros(n, dtype=bool)
    selected[rng.choice(n, size=gradient_batch, replace=False)] = True

    full_parts: list[dict[str, np.ndarray]] = []
    subset_parts: list[dict[str, np.ndarray]] = []
    for start in range(0, n, microbatch):
        stop = min(start + microbatch, n)
        x_again, cache = model.forward(latent[start:stop], want_cache=True)
        if not np.array_equal(x_again, x[start:stop]):
            raise AssertionError("RSR rerun changed outputs before the update")
        local_field = field[start:stop]
        full_parts.append(_stopgrad_grads(model, cache, local_field, n))
        subset_field = np.where(
            selected[start:stop, None], local_field, 0.0)
        subset_parts.append(_stopgrad_grads(
            model, cache, subset_field, gradient_batch))
    full = _sum_grads(model, full_parts)
    subset = _sum_grads(model, subset_parts)
    mixed = {
        name: (1.0 - noise_mix) * full[name] + noise_mix * subset[name]
        for name in model.names
    }
    _adam_step(model, mixed)

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


def pulse_example_matched_updates(reference_updates: int, batch: int,
                                  virtual_batch: int,
                                  warm_fraction: float,
                                  pulse_period: int) -> int:
    """Generator-evaluation-matched count for periodic RSR corrections."""
    if pulse_period <= 0:
        raise ValueError("pulse_period must be positive")
    warm_cost = (batch + (2.0 * virtual_batch - batch) / pulse_period)
    per_update = (warm_fraction * warm_cost +
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

    pure = TanhMLP(target_spec, "missing", 102)
    restored = deepcopy(pure)
    rsr_quantile_step(pure, latent, target, microbatch=5)
    noise_restored_rsr_step(
        restored, latent, target, microbatch=5, gradient_batch=7,
        noise_mix=0.0, rng=np.random.default_rng(17))
    for name in pure.names:
        if not np.allclose(pure.params[name], restored.params[name],
                           rtol=2e-12, atol=2e-12):
            raise AssertionError(f"zero-noise RSR mismatch in {name}")

    if example_matched_updates(1200, 128, 512, 0.70) != 203:
        raise AssertionError("example-matched update calculation changed")
    if pulse_example_matched_updates(1200, 128, 1024, 0.70, 8) != 519:
        raise AssertionError("pulse-matched update calculation changed")

    connected = np.linspace(-2.0, 2.0, 1000)[:, None]
    if diagnose_quantile_resolution(connected, 128).use_large_batch:
        raise AssertionError("connected quantiles triggered large-batch routing")
    rare = np.concatenate([
        np.linspace(-1.1, -0.9, 950),
        np.linspace(2.9, 3.1, 50),
    ])[:, None]
    diagnosis = diagnose_quantile_resolution(rare, 128)
    if not diagnosis.use_large_batch or not (6.0 <=
                                              diagnosis.minimum_expected_batch_count
                                              <= 7.0):
        raise AssertionError("rare separated quantiles evaded routing")
    if not diagnose_quantile_resolution_stable(rare, 128).use_large_batch:
        raise AssertionError("stable rare-region diagnosis failed")
