"""Persistent projected-quantile transport for low-dimensional experiments.

The statistical state is a running average of empirical target quantiles on a
fixed bank of one-dimensional projections.  A persistent particle cloud is
then reconstructed against those projected quantiles by deterministic
tight-frame backprojection.

This is development infrastructure.  It intentionally does not modify the
frozen one-dimensional PQT implementation or any frozen baseline runner.  The
mathematical and experimental scope is recorded in
``PSQTHigherDimImplementationPlan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.linalg import norm

from persistent_quantile_transport import (
    PersistentQuantileTransport,
    empirical_quantiles,
    midpoint_grid,
)


@dataclass(frozen=True)
class PSQTWork:
    """Portable ledger for one persistent sliced-quantile update.

    ``projection_dot_products`` counts vector projections or backprojections,
    not floating-point multiply-adds.  ``sort_work`` uses the same
    ``n log2(n)`` proxy as scalar PQT.
    """

    optimizer_updates: int
    target_samples: int
    target_projection_sorts: int
    reconstruction_sorts: int
    projection_dot_products: int
    sort_work: float
    stored_scalars: int


def _validated_directions(directions: np.ndarray, dimension: int) \
        -> np.ndarray:
    out = np.asarray(directions, dtype=float)
    if out.ndim != 2 or out.shape[1] != dimension or len(out) < 1:
        raise ValueError("directions must have shape (L, dimension), L >= 1")
    if not np.all(np.isfinite(out)):
        raise ValueError("directions must be finite")
    lengths = norm(out, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("directions must be nonzero")
    return out / lengths[:, None]


def uniform_directions_2d(count: int, *, phase: float = 0.0) -> np.ndarray:
    """Uniform unoriented lines on ``[0, pi)``.

    For ``count >= 2`` these form a unit-norm tight frame:
    ``(2/count) * directions.T @ directions = I`` up to roundoff.
    """
    if count < 2:
        raise ValueError("a 2D uniform direction bank needs at least two lines")
    if not math.isfinite(phase):
        raise ValueError("phase must be finite")
    angles = phase + np.pi * np.arange(count, dtype=float) / count
    return np.column_stack((np.cos(angles), np.sin(angles)))


def coordinate_directions(dimension: int) -> np.ndarray:
    """Coordinate-only negative-control bank."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return np.eye(dimension, dtype=float)


def frame_operator(directions: np.ndarray) -> np.ndarray:
    """Return ``(d/L) sum theta theta^T`` for unit directions."""
    raw = np.asarray(directions, dtype=float)
    if raw.ndim != 2:
        raise ValueError("directions must be a matrix")
    dirs = _validated_directions(raw, raw.shape[1])
    d = dirs.shape[1]
    return (d / len(dirs)) * dirs.T @ dirs


def projected_quantile_rmse(a: np.ndarray, b: np.ndarray,
                            directions: np.ndarray, *,
                            knot_count: int = 128) -> float:
    """RMSE between projected quantile tables on an explicit direction bank."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("samples must be matrices with a common dimension")
    if len(a) < 2 or len(b) < 2:
        raise ValueError("each sample must contain at least two points")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("samples must be finite")
    dirs = _validated_directions(directions, a.shape[1])
    grid = midpoint_grid(knot_count)
    squared = 0.0
    for theta in dirs:
        qa = empirical_quantiles(a @ theta, grid)
        qb = empirical_quantiles(b @ theta, grid)
        squared += float(np.mean((qa - qb) ** 2))
    return math.sqrt(squared / len(dirs))


def _particle_quantiles(sample: np.ndarray, probabilities: np.ndarray) \
        -> np.ndarray:
    """Quantiles of an equally weighted persistent knot/particle set.

    Persistent particles represent values at midpoint probabilities.  This
    interpolation therefore leaves an already ordered length-``K`` knot vector
    unchanged on ``midpoint_grid(K)``.  Target minibatches intentionally use
    :func:`empirical_quantiles` instead, matching scalar PQT's estimator.
    """
    values = np.sort(np.asarray(sample, dtype=float).reshape(-1), kind="stable")
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("particle sample must contain at least two finite values")
    return np.interp(
        np.asarray(probabilities, dtype=float), midpoint_grid(len(values)),
        values)


class PersistentSlicedQuantileTransport:
    """Persistent particle reconstruction from projected target quantiles.

    The target table receives exactly one statistical update per target
    minibatch.  ``reconstruction_steps`` only improves geometric consistency;
    it does not reweight or reuse the minibatch in the running average.
    """

    def __init__(self, initial_particles: np.ndarray, directions: np.ndarray,
                 *, knot_count: int | None = None,
                 prior_batches: float = 1.0,
                 reconstruction_steps: int = 3,
                 reconstruction_step_size: float = 0.5) -> None:
        particles = np.asarray(initial_particles, dtype=float)
        if particles.ndim != 2 or len(particles) < 2 or particles.shape[1] < 1:
            raise ValueError(
                "initial_particles must have shape (N, d), N >= 2, d >= 1")
        if not np.all(np.isfinite(particles)):
            raise ValueError("initial particles must be finite")
        if knot_count is None:
            knot_count = len(particles)
        if knot_count < 2:
            raise ValueError("knot_count must be at least two")
        if not math.isfinite(prior_batches) or prior_batches < 0.0:
            raise ValueError("prior_batches must be finite and nonnegative")
        if reconstruction_steps < 1:
            raise ValueError("reconstruction_steps must be positive")
        if (not math.isfinite(reconstruction_step_size) or
                not 0.0 < reconstruction_step_size <= 1.0):
            raise ValueError("reconstruction_step_size must lie in (0, 1]")

        self.particles = particles.copy()
        self.dimension = particles.shape[1]
        self.directions = _validated_directions(
            directions, self.dimension).copy()
        self.grid = midpoint_grid(knot_count)
        self.effective_batches = float(prior_batches)
        self.reconstruction_steps = int(reconstruction_steps)
        self.reconstruction_step_size = float(reconstruction_step_size)
        self.updates = 0

        projected = self.particles @ self.directions.T
        self.target_quantiles = np.vstack([
            _particle_quantiles(projected[:, ell], self.grid)
            for ell in range(len(self.directions))
        ])
        self._check_state()

    @property
    def stored_scalars(self) -> int:
        return int(
            self.particles.size + self.directions.size +
            self.target_quantiles.size + self.grid.size + 1)

    @property
    def tail_mass_per_projection(self) -> float:
        return float(self.grid[0] + 1.0 - self.grid[-1])

    def _batch_quantiles(self, target: np.ndarray) -> np.ndarray:
        projected = target @ self.directions.T
        # NumPy sorts every projected column in compiled code.  This is
        # mathematically identical to calling ``empirical_quantiles`` on each
        # direction separately and avoids a direction-level Python loop.
        result = np.quantile(
            projected, self.grid, axis=0, method="linear").T
        return np.maximum.accumulate(np.asarray(result, dtype=float), axis=1)

    def _reconstruct_once(self, ordered_target: np.ndarray) -> None:
        count = len(self.directions)
        projected = self.particles @ self.directions.T
        desired = np.empty_like(projected)
        order = np.argsort(projected, axis=0, kind="stable")
        columns = np.arange(count)[None, :]
        desired[order, columns] = ordered_target
        residual = desired - projected
        correction = (self.dimension / count) * residual @ self.directions
        self.particles += self.reconstruction_step_size * correction

    def update(self, target: np.ndarray) -> PSQTWork:
        """Accumulate one target minibatch, then reconstruct the particles."""
        values = np.asarray(target, dtype=float)
        if (values.ndim != 2 or values.shape[1] != self.dimension or
                len(values) < 2):
            raise ValueError("target must have shape (B, dimension), B >= 2")
        if not np.all(np.isfinite(values)):
            raise ValueError("target minibatch must be finite")

        estimate = self._batch_quantiles(values)
        next_mass = self.effective_batches + 1.0
        if next_mass <= 0.0:
            raise AssertionError("nonpositive running-average mass")
        self.target_quantiles = (
            (self.effective_batches / next_mass) * self.target_quantiles +
            (1.0 / next_mass) * estimate)
        self.effective_batches = next_mass

        probabilities = midpoint_grid(len(self.particles))
        if (len(probabilities) == len(self.grid) and
                np.array_equal(probabilities, self.grid)):
            ordered_target = self.target_quantiles.T
        else:
            ordered_target = np.column_stack([
                np.interp(probabilities, self.grid, row)
                for row in self.target_quantiles
            ])
        for _ in range(self.reconstruction_steps):
            self._reconstruct_once(ordered_target)
        self.updates += 1
        self._check_state()

        batch = len(values)
        particles = len(self.particles)
        directions = len(self.directions)
        target_sort = directions
        reconstruction_sort = directions * self.reconstruction_steps
        sort_work = (
            directions * batch * math.log2(max(batch, 2)) +
            reconstruction_sort * particles *
            math.log2(max(particles, 2)))
        # Target projections, source projections, and residual
        # backprojections.  These are vector-level counts.
        dot_products = (
            batch * directions +
            self.reconstruction_steps * particles * directions * 2)
        return PSQTWork(
            optimizer_updates=1,
            target_samples=batch,
            target_projection_sorts=target_sort,
            reconstruction_sorts=reconstruction_sort,
            projection_dot_products=dot_products,
            sort_work=float(sort_work),
            stored_scalars=self.stored_scalars,
        )

    def training_projection_rmse(self) -> float:
        """Current inconsistency with the accumulated training projections."""
        projected = self.particles @ self.directions.T
        squared = 0.0
        for ell in range(len(self.directions)):
            current = _particle_quantiles(projected[:, ell], self.grid)
            squared += float(np.mean(
                (current - self.target_quantiles[ell]) ** 2))
        return math.sqrt(squared / len(self.directions))

    def sample(self, count: int, rng: np.random.Generator, *,
               jitter: float = 0.0) -> np.ndarray:
        """Sample the empirical particle generator, with optional set jitter.

        The benchmark uses ``jitter=0``.  Nonzero jitter is exposed only for
        visualization experiments and must be reported as a model change.
        """
        if count < 1:
            raise ValueError("count must be positive")
        if not math.isfinite(jitter) or jitter < 0.0:
            raise ValueError("jitter must be finite and nonnegative")
        indices = rng.integers(0, len(self.particles), size=count)
        out = self.particles[indices].copy()
        if jitter > 0.0:
            out += rng.normal(size=out.shape) * jitter
        return out

    def _check_state(self) -> None:
        if (not np.all(np.isfinite(self.particles)) or
                not np.all(np.isfinite(self.target_quantiles))):
            raise FloatingPointError("non-finite PSQT state")
        if np.any(np.diff(self.target_quantiles, axis=1) < -1e-12):
            raise AssertionError("projected target quantiles lost monotonicity")


def _collapsed_cdf_grid(values: np.ndarray, grid: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray]:
    """Strict source knots and representative probabilities for interpolation."""
    unique, inverse = np.unique(values, return_inverse=True)
    sums = np.zeros(len(unique), dtype=float)
    counts = np.zeros(len(unique), dtype=float)
    np.add.at(sums, inverse, grid)
    np.add.at(counts, inverse, 1.0)
    return unique, sums / counts


@dataclass
class RidgeQuantileLayer:
    """Replayable damped monotone correction along one direction."""

    direction: np.ndarray
    grid: np.ndarray
    source_quantiles: np.ndarray
    target_quantiles: np.ndarray
    step_size: float = 1.0

    def __post_init__(self) -> None:
        self.direction = np.asarray(self.direction, dtype=float).reshape(-1)
        length = norm(self.direction)
        if not np.all(np.isfinite(self.direction)) or length <= 0.0:
            raise ValueError("direction must be a finite nonzero vector")
        self.direction = self.direction / length
        self.grid = np.asarray(self.grid, dtype=float).reshape(-1).copy()
        self.source_quantiles = np.asarray(
            self.source_quantiles, dtype=float).reshape(-1).copy()
        self.target_quantiles = np.asarray(
            self.target_quantiles, dtype=float).reshape(-1).copy()
        if not (len(self.grid) >= 2 and
                self.source_quantiles.shape == self.grid.shape and
                self.target_quantiles.shape == self.grid.shape):
            raise ValueError("ridge quantile vectors must have equal length >= 2")
        if (not np.all(np.isfinite(self.grid)) or
                not np.all(np.isfinite(self.source_quantiles)) or
                not np.all(np.isfinite(self.target_quantiles))):
            raise ValueError("ridge layer values must be finite")
        if np.any(np.diff(self.grid) <= 0.0):
            raise ValueError("ridge probability grid must be strictly increasing")
        if (np.any(np.diff(self.source_quantiles) < 0.0) or
                np.any(np.diff(self.target_quantiles) < 0.0)):
            raise ValueError("ridge quantiles must be nondecreasing")
        if not math.isfinite(self.step_size) or not 0.0 < self.step_size <= 1.0:
            raise ValueError("ridge step_size must lie in (0, 1]")

    @classmethod
    def from_samples(cls, source: np.ndarray, target: np.ndarray,
                     direction: np.ndarray, *, knot_count: int = 128,
                     step_size: float = 1.0) -> "RidgeQuantileLayer":
        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        theta = np.asarray(direction, dtype=float).reshape(-1)
        if (source.ndim != 2 or target.ndim != 2 or
                source.shape[1] != target.shape[1] or
                len(theta) != source.shape[1]):
            raise ValueError("source, target, and direction dimensions disagree")
        theta = theta / norm(theta)
        grid = midpoint_grid(knot_count)
        return cls(
            theta,
            grid,
            empirical_quantiles(source @ theta, grid),
            empirical_quantiles(target @ theta, grid),
            step_size,
        )

    def apply(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != len(self.direction) or
                not np.all(np.isfinite(points))):
            raise ValueError("points have the wrong dimension or are non-finite")
        scalar = points @ self.direction
        source, probabilities = _collapsed_cdf_grid(
            self.source_quantiles, self.grid)
        rank = np.interp(
            scalar, source, probabilities,
            left=self.grid[0], right=self.grid[-1])
        destination = np.interp(
            rank, self.grid, self.target_quantiles)
        displacement = self.step_size * (destination - scalar)
        return points + displacement[:, None] * self.direction[None, :]

    @property
    def stored_scalars(self) -> int:
        return int(
            len(self.direction) + len(self.grid) +
            len(self.source_quantiles) + len(self.target_quantiles) + 1)


class CompositionalSlicedQuantileMap:
    """Finite composition of replayable ridge-quantile layers."""

    def __init__(self, dimension: int) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self.layers: list[RidgeQuantileLayer] = []

    def append(self, layer: RidgeQuantileLayer) -> None:
        if len(layer.direction) != self.dimension:
            raise ValueError("ridge layer dimension mismatch")
        self.layers.append(layer)

    def forward(self, base: np.ndarray) -> np.ndarray:
        out = np.asarray(base, dtype=float).copy()
        if out.ndim != 2 or out.shape[1] != self.dimension:
            raise ValueError("base sample has the wrong dimension")
        for layer in self.layers:
            out = layer.apply(out)
        return out

    @property
    def stored_scalars(self) -> int:
        return int(sum(layer.stored_scalars for layer in self.layers))


def invariant_tests() -> None:
    """Fast deterministic mathematical and implementation invariants."""
    # Exact unit-norm tight frame in 2D, up to floating-point roundoff.
    directions = uniform_directions_2d(16)
    if not np.allclose(frame_operator(directions), np.eye(2),
                       rtol=0.0, atol=2e-15):
        raise AssertionError("uniform 2D directions are not a tight frame")

    # Exact reduction to the existing scalar PQT knot update.
    grid = midpoint_grid(16)
    initial = np.linspace(-0.4, 0.7, len(grid))
    batch = np.asarray([[-3.0], [-1.0], [-0.25], [0.1], [0.8], [2.0]])
    scalar = PersistentQuantileTransport(
        grid, initial, prior_batches=1.0)
    sliced = PersistentSlicedQuantileTransport(
        initial[:, None], np.ones((1, 1)), knot_count=len(grid),
        prior_batches=1.0, reconstruction_steps=1,
        reconstruction_step_size=1.0)
    scalar.update(batch)
    sliced.update(batch)
    if not np.allclose(
            np.sort(sliced.particles[:, 0]), scalar.values,
            rtol=0.0, atol=2e-14):
        raise AssertionError("PSQT does not reduce exactly to scalar PQT")

    # Tight-frame normalization has the correct scale for a pure translation.
    shift = np.asarray([1.25, -0.7])
    collapsed = np.zeros((32, 2))
    translated = np.repeat(shift[None, :], 40, axis=0)
    translation_model = PersistentSlicedQuantileTransport(
        collapsed, directions, knot_count=32, prior_batches=1.0,
        reconstruction_steps=1, reconstruction_step_size=1.0)
    translation_model.update(translated)
    if not np.allclose(
            translation_model.particles, 0.5 * shift[None, :],
            rtol=0.0, atol=3e-14):
        raise AssertionError("PSQT tight-frame translation scaling is wrong")

    # Rotation equivariance when data, particles, and directions rotate
    # together.  A fixed unrotated finite bank is not claimed equivariant.
    rng = np.random.default_rng(20260721)
    x = rng.normal(size=(31, 2))
    y = rng.normal(size=(47, 2)) + np.asarray([0.5, -0.2])
    angle = 0.37
    rotation = np.asarray([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)],
    ])
    first = PersistentSlicedQuantileTransport(
        x, directions, knot_count=24, reconstruction_steps=2,
        reconstruction_step_size=0.4)
    second = PersistentSlicedQuantileTransport(
        x @ rotation.T, directions @ rotation.T, knot_count=24,
        reconstruction_steps=2, reconstruction_step_size=0.4)
    first.update(y)
    second.update(y @ rotation.T)
    if not np.allclose(
            second.particles, first.particles @ rotation.T,
            rtol=0.0, atol=2e-13):
        raise AssertionError("PSQT rotated-system equivariance failed")

    # Coordinate marginals cannot see dependence; off-axis slices can.
    diagonal = np.asarray([
        [-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]])
    antidiagonal = np.asarray([
        [-2.0, 2.0], [-1.0, 1.0], [1.0, -1.0], [2.0, -2.0]])
    coordinate_error = projected_quantile_rmse(
        diagonal, antidiagonal, coordinate_directions(2), knot_count=16)
    sliced_error = projected_quantile_rmse(
        diagonal, antidiagonal, uniform_directions_2d(8), knot_count=16)
    if coordinate_error > 1e-14 or sliced_error < 0.1:
        raise AssertionError("dependence-trap projection invariant failed")

    # A ridge layer is deterministic, replayable, and monotone on its line.
    base = rng.normal(size=(101, 2))
    target = base + np.asarray([1.0, 0.0])
    layer = RidgeQuantileLayer.from_samples(
        base, target, np.asarray([1.0, 0.0]), knot_count=64)
    mapping = CompositionalSlicedQuantileMap(2)
    mapping.append(layer)
    replay_a = mapping.forward(base)
    replay_b = layer.apply(base)
    if not np.array_equal(replay_a, replay_b):
        raise AssertionError("compositional ridge replay is not deterministic")
    order = np.argsort(base[:, 0], kind="stable")
    if np.any(np.diff(replay_a[order, 0]) < -1e-12):
        raise AssertionError("ridge quantile layer is not monotone")


if __name__ == "__main__":
    invariant_tests()
    print("Persistent Sliced Quantile Transport invariants: PASS")
