"""Persistent one-dimensional quantile-transport generator.

This module is development infrastructure for the transport-aligned generator
program.  It deliberately does not modify the frozen QLD or LB-QCD runners.

The generator is a monotone piecewise-linear map from ``Uniform(0, 1)`` to the
data line.  Every training update forms an empirical target quantile vector on
the generator's fixed probability grid and averages it into the stored knot
values.  Since both operands are ordered, monotonicity is preserved exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PQTWork:
    """Auditable work ledger for one Persistent Quantile Transport update."""

    optimizer_updates: int
    target_samples: int
    sort_work: float
    stored_scalars: int


class PersistentQuantileTransport:
    """Monotone piecewise-linear empirical quantile generator.

    ``prior_batches`` controls the strength of the supplied initialization.
    With the default value one, after ``t`` observations of empirical quantile
    vectors the knots are their exact running mean plus one equally weighted
    initialization vector.  This makes the update deterministic conditional on
    the target batches and avoids an unreported learning-rate search.
    """

    def __init__(self, grid: np.ndarray, initial_values: np.ndarray, *,
                 prior_batches: float = 1.0) -> None:
        grid = np.asarray(grid, dtype=float).reshape(-1)
        values = np.asarray(initial_values, dtype=float).reshape(-1)
        if len(grid) < 2 or values.shape != grid.shape:
            raise ValueError("grid and values must be equal vectors of length >= 2")
        if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(values)):
            raise ValueError("grid and initial values must be finite")
        if not np.all((0.0 < grid) & (grid < 1.0)):
            raise ValueError("probability knots must lie strictly inside (0, 1)")
        if not np.all(np.diff(grid) > 0.0):
            raise ValueError("probability knots must be strictly increasing")
        if not np.all(np.diff(values) >= 0.0):
            raise ValueError("initial quantile values must be nondecreasing")
        if not math.isfinite(prior_batches) or prior_batches < 0.0:
            raise ValueError("prior_batches must be finite and nonnegative")
        self.grid = grid.copy()
        self.values = values.copy()
        self.effective_batches = float(prior_batches)
        self.updates = 0

    @classmethod
    def from_ordered_initial_sample(
            cls, knot_count: int, sample: np.ndarray, *,
            prior_batches: float = 1.0) -> "PersistentQuantileTransport":
        """Initialize on the empirical quantiles of an independent sample."""
        if knot_count < 2:
            raise ValueError("knot_count must be at least two")
        grid = midpoint_grid(knot_count)
        values = empirical_quantiles(sample, grid)
        return cls(grid, values, prior_batches=prior_batches)

    def update(self, target: np.ndarray) -> PQTWork:
        """Average one independent empirical target quantile vector."""
        values = np.asarray(target, dtype=float)
        if values.ndim != 2 or values.shape[1] != 1 or len(values) < 2:
            raise ValueError("target must have shape (N, 1), N >= 2")
        if not np.all(np.isfinite(values)):
            raise ValueError("target batch must be finite")
        estimate = empirical_quantiles(values, self.grid)
        next_mass = self.effective_batches + 1.0
        if next_mass <= 0.0:
            raise AssertionError("nonpositive running-average mass")
        self.values = ((self.effective_batches / next_mass) * self.values +
                       (1.0 / next_mass) * estimate)
        self.effective_batches = next_mass
        self.updates += 1
        self._check_state()
        n = len(values)
        return PQTWork(
            optimizer_updates=1,
            target_samples=n,
            sort_work=float(n * math.log2(max(n, 2))),
            stored_scalars=len(self.grid) * 2,
        )

    def forward(self, uniform_latent: np.ndarray) -> np.ndarray:
        """Generate from untouched scalar uniforms using monotone interpolation.

        Values outside the interior knot range are clamped to the endpoint
        knots.  The omitted probability mass is exactly ``grid[0]`` on the
        left and ``1-grid[-1]`` on the right and is exposed by ``tail_mass``.
        """
        u = np.asarray(uniform_latent, dtype=float)
        if u.ndim == 2 and u.shape[1] == 1:
            flat = u[:, 0]
        elif u.ndim == 1:
            flat = u
        else:
            raise ValueError("uniform latent must have shape (N,) or (N, 1)")
        if not np.all(np.isfinite(flat)) or not np.all(
                (0.0 <= flat) & (flat <= 1.0)):
            raise ValueError("uniform latent values must lie in [0, 1]")
        return np.interp(flat, self.grid, self.values).reshape(-1, 1)

    @property
    def tail_mass(self) -> float:
        return float(self.grid[0] + (1.0 - self.grid[-1]))

    def _check_state(self) -> None:
        if not np.all(np.isfinite(self.values)):
            raise FloatingPointError("non-finite PQT knots")
        if not np.all(np.diff(self.values) >= -1e-12):
            raise AssertionError("PQT monotonicity invariant failed")


def midpoint_grid(knot_count: int) -> np.ndarray:
    """Equally spaced interior probability knots."""
    if knot_count < 2:
        raise ValueError("knot_count must be at least two")
    return (np.arange(knot_count, dtype=float) + 0.5) / knot_count


def empirical_quantiles(sample: np.ndarray, probabilities: np.ndarray) \
        -> np.ndarray:
    """Linear empirical quantiles as a nondecreasing vector."""
    values = np.asarray(sample, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    elif values.ndim != 1:
        raise ValueError("sample must have shape (N,) or (N, 1)")
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("sample must contain at least two finite values")
    if not np.all((0.0 <= probabilities) & (probabilities <= 1.0)):
        raise ValueError("quantile probabilities must lie in [0, 1]")
    result = np.quantile(values, probabilities, method="linear")
    # NumPy quantiles are ordered mathematically.  Maximum accumulation guards
    # against a one-ulp inversion without changing any larger-scale value.
    return np.maximum.accumulate(np.asarray(result, dtype=float))


def invariant_tests() -> None:
    """Fast deterministic invariants for CI and development runners."""
    grid = midpoint_grid(8)
    initial = np.linspace(-0.2, 0.2, len(grid))
    model = PersistentQuantileTransport(grid, initial, prior_batches=1.0)
    batch_a = np.asarray([[-2.0], [-1.0], [0.0], [3.0]])
    batch_b = np.asarray([[-4.0], [-0.5], [1.0], [2.0], [5.0]])
    estimate_a = empirical_quantiles(batch_a, grid)
    estimate_b = empirical_quantiles(batch_b, grid)
    model.update(batch_a)
    expected_a = 0.5 * (initial + estimate_a)
    if not np.allclose(model.values, expected_a, rtol=0.0, atol=1e-14):
        raise AssertionError("first PQT running-average update is incorrect")
    model.update(batch_b)
    expected_b = (initial + estimate_a + estimate_b) / 3.0
    if not np.allclose(model.values, expected_b, rtol=0.0, atol=1e-14):
        raise AssertionError("second PQT running-average update is incorrect")
    u = np.linspace(0.0, 1.0, 1001)
    generated = model.forward(u)[:, 0]
    if not np.all(np.diff(generated) >= -1e-12):
        raise AssertionError("PQT interpolation is not monotone")
    if model.tail_mass != 1.0 / len(grid):
        raise AssertionError("PQT endpoint tail-mass ledger is incorrect")


if __name__ == "__main__":
    invariant_tests()
    print("Persistent Quantile Transport invariants: PASS")

