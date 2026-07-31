"""Target-statistic accumulators for the PSQT repair experiment.

The historical PSQT implementation averages complete minibatch quantile
vectors.  This module leaves that implementation untouched and provides
factorized alternatives that all expose the same projected quantile table.

The ``KLLStyleProjectedAccumulator`` is a deliberately simple fixed-capacity
random-compactor hierarchy.  It uses the KLL compaction mechanism, but not the
optimal level-dependent capacity schedule from the KLL paper; experiment
reports must therefore call it *KLL-style*, not claim the optimal KLL theorem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.linalg import norm

from persistent_quantile_transport import midpoint_grid


def _directions(values: np.ndarray, dimension: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 2 or result.shape[1] != dimension or len(result) < 1:
        raise ValueError("directions must have shape (L, dimension)")
    lengths = norm(result, axis=1)
    if not np.all(np.isfinite(result)) or np.any(lengths <= 0.0):
        raise ValueError("directions must be finite and nonzero")
    return result / lengths[:, None]


def _points(values: np.ndarray, dimension: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (result.ndim != 2 or result.shape[1] != dimension or
            len(result) < 1 or not np.all(np.isfinite(result))):
        raise ValueError("points must be a nonempty finite matrix")
    return result


def inverted_empirical_quantiles(sample: np.ndarray,
                                 probabilities: np.ndarray) -> np.ndarray:
    """Observed-value empirical quantiles (inverse right-continuous ECDF)."""
    values = np.asarray(sample, dtype=float).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(values) < 1 or not np.all(np.isfinite(values)):
        raise ValueError("sample must contain finite values")
    if not np.all((0.0 <= probabilities) & (probabilities <= 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    result = np.quantile(
        values, probabilities, method="inverted_cdf")
    return np.maximum.accumulate(np.asarray(result, dtype=float))


def projected_quantile_table(points: np.ndarray, directions: np.ndarray,
                             grid: np.ndarray, *,
                             method: str = "inverted_cdf") -> np.ndarray:
    """Return an ``(L, K)`` projected empirical-quantile table."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2:
        raise ValueError("points must be a matrix")
    dirs = _directions(directions, values.shape[1])
    probabilities = np.asarray(grid, dtype=float).reshape(-1)
    if len(probabilities) < 2:
        raise ValueError("grid must have at least two probabilities")
    projected = values @ dirs.T
    table = np.quantile(
        projected, probabilities, axis=0, method=method).T
    return np.maximum.accumulate(np.asarray(table, dtype=float), axis=1)


def particle_quantile_table(points: np.ndarray, directions: np.ndarray,
                            grid: np.ndarray) -> np.ndarray:
    """Linear midpoint interpolation used by historical PSQT particles."""
    values = np.asarray(points, dtype=float)
    dirs = _directions(directions, values.shape[1])
    probabilities = np.asarray(grid, dtype=float).reshape(-1)
    source_grid = midpoint_grid(len(values))
    projected = values @ dirs.T
    return np.vstack([
        np.interp(probabilities, source_grid,
                  np.sort(projected[:, ell], kind="stable"))
        for ell in range(len(dirs))
    ])


@dataclass(frozen=True)
class AccumulatorLedger:
    """Portable final ledger for a target-statistic accumulator."""

    target_samples: int
    projection_dot_products: int
    sort_work: float
    persistent_scalars: int
    peak_working_scalars: int
    retained_items: int
    retained_weight: int


class _ProjectedAccumulator:
    """Validation and work accounting shared by accumulator implementations."""

    def __init__(self, dimension: int, directions: np.ndarray,
                 knot_count: int) -> None:
        if dimension < 1 or knot_count < 2:
            raise ValueError("dimension must be positive and knots >= 2")
        self.dimension = int(dimension)
        self.directions = _directions(directions, dimension).copy()
        self.grid = midpoint_grid(knot_count)
        self.target_samples = 0
        self.projection_dot_products = 0
        self.sort_work = 0.0
        self._peak_working = 0

    def _project(self, points: np.ndarray) -> np.ndarray:
        values = _points(points, self.dimension)
        projected = values @ self.directions.T
        self.target_samples += len(values)
        self.projection_dot_products += len(values) * len(self.directions)
        self._peak_working = max(self._peak_working, int(projected.size))
        return projected

    @property
    def persistent_scalars(self) -> int:
        raise NotImplementedError

    @property
    def retained_items(self) -> int:
        return 0

    @property
    def retained_weight(self) -> int:
        return self.target_samples

    def table(self) -> np.ndarray:
        raise NotImplementedError

    def ledger(self) -> AccumulatorLedger:
        state = self.persistent_scalars
        return AccumulatorLedger(
            target_samples=self.target_samples,
            projection_dot_products=self.projection_dot_products,
            sort_work=float(self.sort_work),
            persistent_scalars=state,
            peak_working_scalars=int(state + self._peak_working),
            retained_items=self.retained_items,
            retained_weight=self.retained_weight,
        )


class BatchMeanProjectedAccumulator(_ProjectedAccumulator):
    """Historical equal-minibatch quantile mean, factorized from geometry."""

    def __init__(self, initial_points: np.ndarray, directions: np.ndarray,
                 knot_count: int, *, prior_batches: float = 1.0) -> None:
        values = np.asarray(initial_points, dtype=float)
        super().__init__(values.shape[1], directions, knot_count)
        if not math.isfinite(prior_batches) or prior_batches < 0.0:
            raise ValueError("prior_batches must be finite and nonnegative")
        self.quantiles = particle_quantile_table(
            values, self.directions, self.grid)
        self.effective_batches = float(prior_batches)

    def update(self, points: np.ndarray) -> None:
        projected = self._project(points)
        estimate = np.quantile(
            projected, self.grid, axis=0, method="linear").T
        estimate = np.maximum.accumulate(estimate, axis=1)
        batch = len(projected)
        self.sort_work += (
            len(self.directions) * batch * math.log2(max(batch, 2)))
        mass = self.effective_batches + 1.0
        self.quantiles = (
            self.effective_batches / mass * self.quantiles + estimate / mass)
        self.effective_batches = mass

    @property
    def persistent_scalars(self) -> int:
        return int(
            self.directions.size + self.grid.size + self.quantiles.size + 1)

    def table(self) -> np.ndarray:
        return self.quantiles.copy()


class BatchMedianProjectedAccumulator(_ProjectedAccumulator):
    """Diagnostic coordinate-wise median of complete batch quantile tables."""

    def __init__(self, dimension: int, directions: np.ndarray,
                 knot_count: int) -> None:
        super().__init__(dimension, directions, knot_count)
        self.tables: list[np.ndarray] = []

    def update(self, points: np.ndarray) -> None:
        projected = self._project(points)
        estimate = np.quantile(
            projected, self.grid, axis=0, method="linear").T
        self.tables.append(np.maximum.accumulate(estimate, axis=1))
        batch = len(projected)
        self.sort_work += (
            len(self.directions) * batch * math.log2(max(batch, 2)))

    @property
    def persistent_scalars(self) -> int:
        history = sum(table.size for table in self.tables)
        return int(self.directions.size + self.grid.size + history)

    def table(self) -> np.ndarray:
        if not self.tables:
            raise ValueError("cannot query an empty accumulator")
        count = len(self.tables)
        self.sort_work += (
            self.directions.shape[0] * len(self.grid) * count *
            math.log2(max(count, 2)))
        result = np.median(np.stack(self.tables), axis=0)
        return np.maximum.accumulate(result, axis=1)


class RawReservoirProjectedAccumulator(_ProjectedAccumulator):
    """Algorithm-R reservoir of raw points, reusable for new directions."""

    def __init__(self, dimension: int, directions: np.ndarray,
                 knot_count: int, capacity: int, *, seed: int) -> None:
        super().__init__(dimension, directions, knot_count)
        if capacity < 2:
            raise ValueError("reservoir capacity must be at least two")
        self.capacity = int(capacity)
        self.buffer = np.empty((capacity, dimension), dtype=float)
        self.size = 0
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def update(self, points: np.ndarray) -> None:
        values = _points(points, self.dimension)
        self.target_samples += len(values)
        for point in values:
            self.seen += 1
            if self.size < self.capacity:
                self.buffer[self.size] = point
                self.size += 1
            else:
                slot = int(self.rng.integers(0, self.seen))
                if slot < self.capacity:
                    self.buffer[slot] = point
        self._peak_working = max(self._peak_working, int(values.size))

    @property
    def persistent_scalars(self) -> int:
        return int(
            self.directions.size + self.grid.size +
            self.size * self.dimension + 2)

    @property
    def retained_items(self) -> int:
        return self.size

    def table(self) -> np.ndarray:
        if self.size < 2:
            raise ValueError("reservoir needs at least two observations")
        sample = self.buffer[:self.size]
        self.projection_dot_products += self.size * len(self.directions)
        self.sort_work += (
            len(self.directions) * self.size *
            math.log2(max(self.size, 2)))
        self._peak_working = max(
            self._peak_working, self.size * len(self.directions))
        return projected_quantile_table(
            sample, self.directions, self.grid, method="inverted_cdf")


class ExactPooledProjectedAccumulator(_ProjectedAccumulator):
    """Unbounded raw-point pool used only as a finite-stream ceiling."""

    def __init__(self, dimension: int, directions: np.ndarray,
                 knot_count: int) -> None:
        super().__init__(dimension, directions, knot_count)
        self.batches: list[np.ndarray] = []

    def update(self, points: np.ndarray) -> None:
        values = _points(points, self.dimension)
        self.batches.append(values.copy())
        self.target_samples += len(values)
        self._peak_working = max(self._peak_working, int(values.size))

    @property
    def persistent_scalars(self) -> int:
        observations = sum(batch.size for batch in self.batches)
        return int(self.directions.size + self.grid.size + observations)

    @property
    def retained_items(self) -> int:
        return self.target_samples

    def table(self) -> np.ndarray:
        if not self.batches:
            raise ValueError("cannot query an empty pool")
        sample = np.vstack(self.batches)
        count = len(sample)
        self.projection_dot_products += count * len(self.directions)
        self.sort_work += (
            len(self.directions) * count * math.log2(max(count, 2)))
        self._peak_working = max(
            self._peak_working, count * len(self.directions))
        return projected_quantile_table(
            sample, self.directions, self.grid, method="inverted_cdf")


class RandomCompactorSketch:
    """One-dimensional fixed-capacity KLL-style weighted compactor hierarchy."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if capacity < 8:
            raise ValueError("compactor capacity must be at least eight")
        self.capacity = int(capacity)
        self.levels: list[list[float]] = [[]]
        self.count = 0
        self.sort_work = 0.0
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        sample = np.asarray(values, dtype=float).reshape(-1)
        if not np.all(np.isfinite(sample)):
            raise ValueError("sketch values must be finite")
        self.count += len(sample)
        self.levels[0].extend(float(value) for value in sample)
        self._compress_from(0)
        if self.retained_weight != self.count:
            raise AssertionError("compaction did not preserve total weight")

    def _compress_from(self, start: int) -> None:
        level = start
        while level < len(self.levels):
            if len(self.levels[level]) <= self.capacity:
                level += 1
                continue
            values = np.sort(np.asarray(self.levels[level], dtype=float))
            size = len(values)
            self.sort_work += size * math.log2(max(size, 2))
            retained: list[float] = []
            compact = values
            if size % 2 == 1:
                if int(self.rng.integers(0, 2)) == 0:
                    retained.append(float(values[0]))
                    compact = values[1:]
                else:
                    retained.append(float(values[-1]))
                    compact = values[:-1]
            parity = int(self.rng.integers(0, 2))
            promoted = compact[parity::2]
            self.levels[level] = retained
            if level + 1 == len(self.levels):
                self.levels.append([])
            self.levels[level + 1].extend(
                float(value) for value in promoted)
            level += 1

    @property
    def retained_items(self) -> int:
        return sum(len(level) for level in self.levels)

    @property
    def retained_weight(self) -> int:
        return sum(
            len(level) * (1 << height)
            for height, level in enumerate(self.levels))

    def quantiles(self, probabilities: np.ndarray) -> np.ndarray:
        if self.count < 1:
            raise ValueError("cannot query an empty sketch")
        values: list[float] = []
        weights: list[int] = []
        for height, level in enumerate(self.levels):
            values.extend(level)
            weights.extend([1 << height] * len(level))
        raw = np.asarray(values, dtype=float)
        mass = np.asarray(weights, dtype=np.int64)
        order = np.argsort(raw, kind="stable")
        raw = raw[order]
        mass = mass[order]
        cumulative = np.cumsum(mass)
        probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        ranks = np.maximum(
            1, np.ceil(probabilities * self.count).astype(np.int64))
        indices = np.searchsorted(cumulative, ranks, side="left")
        self.sort_work += len(raw) * math.log2(max(len(raw), 2))
        result = raw[np.minimum(indices, len(raw) - 1)]
        return np.maximum.accumulate(result)


class KLLStyleProjectedAccumulator(_ProjectedAccumulator):
    """Independent fixed-capacity random-compactor sketch per projection."""

    def __init__(self, dimension: int, directions: np.ndarray,
                 knot_count: int, capacity: int, *, seed: int) -> None:
        super().__init__(dimension, directions, knot_count)
        self.capacity = int(capacity)
        sequence = np.random.SeedSequence(seed)
        children = sequence.spawn(len(self.directions))
        self.sketches = [
            RandomCompactorSketch(
                capacity, seed=int(child.generate_state(1)[0]))
            for child in children
        ]

    def update(self, points: np.ndarray) -> None:
        projected = self._project(points)
        for ell, sketch in enumerate(self.sketches):
            before = sketch.sort_work
            sketch.update(projected[:, ell])
            self.sort_work += sketch.sort_work - before

    @property
    def persistent_scalars(self) -> int:
        return int(
            self.directions.size + self.grid.size +
            sum(sketch.retained_items for sketch in self.sketches) +
            2 * len(self.sketches))

    @property
    def retained_items(self) -> int:
        return sum(sketch.retained_items for sketch in self.sketches)

    @property
    def retained_weight(self) -> int:
        weights = [sketch.retained_weight for sketch in self.sketches]
        if weights and any(weight != weights[0] for weight in weights):
            raise AssertionError("projection sketches have unequal weight")
        return weights[0] if weights else 0

    def table(self) -> np.ndarray:
        result = []
        for sketch in self.sketches:
            before = sketch.sort_work
            result.append(sketch.quantiles(self.grid))
            self.sort_work += sketch.sort_work - before
        return np.vstack(result)


def reconstruct_from_quantile_table(
        initial_particles: np.ndarray, directions: np.ndarray,
        target_table: np.ndarray, *, steps: int,
        step_size: float = 0.5) -> tuple[np.ndarray, float, int]:
    """Tight-frame particle reconstruction against one fixed table.

    Returns particles, the sort-work proxy, and vector-level projection or
    backprojection count.
    """
    particles = np.asarray(initial_particles, dtype=float).copy()
    if particles.ndim != 2 or len(particles) < 2:
        raise ValueError("initial particles must be a matrix with N >= 2")
    dirs = _directions(directions, particles.shape[1])
    table = np.asarray(target_table, dtype=float)
    if table.ndim != 2 or table.shape[0] != len(dirs):
        raise ValueError("target table must have one row per direction")
    if (steps < 1 or not math.isfinite(step_size) or
            not 0.0 < step_size <= 1.0):
        raise ValueError("invalid reconstruction schedule")
    probabilities = midpoint_grid(len(particles))
    source_grid = midpoint_grid(table.shape[1])
    if np.array_equal(probabilities, source_grid):
        ordered_target = table.T
    else:
        ordered_target = np.column_stack([
            np.interp(probabilities, source_grid, row) for row in table
        ])
    for _ in range(steps):
        projected = particles @ dirs.T
        desired = np.empty_like(projected)
        order = np.argsort(projected, axis=0, kind="stable")
        desired[order, np.arange(len(dirs))[None, :]] = ordered_target
        correction = (
            particles.shape[1] / len(dirs) *
            (desired - projected) @ dirs)
        particles += step_size * correction
    n = len(particles)
    sort_work = steps * len(dirs) * n * math.log2(max(n, 2))
    dot_products = steps * n * len(dirs) * 2
    return particles, float(sort_work), int(dot_products)


def invariant_tests() -> None:
    """Fast deterministic accumulator and reconstruction invariants."""
    rng = np.random.default_rng(20260722)
    directions = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    points = rng.normal(size=(33, 2))
    grid = midpoint_grid(16)

    pooled = ExactPooledProjectedAccumulator(2, directions, 16)
    pooled.update(points[:11])
    pooled.update(points[11:])
    expected = projected_quantile_table(points, directions, grid)
    if not np.array_equal(pooled.table(), expected):
        raise AssertionError("exact pooled accumulator disagrees with NumPy")

    mean = BatchMeanProjectedAccumulator(
        points[:16], directions, 16, prior_batches=0.0)
    mean.update(points[:11])
    mean.update(points[11:])
    first = projected_quantile_table(
        points[:11], directions, grid, method="linear")
    second = projected_quantile_table(
        points[11:], directions, grid, method="linear")
    if not np.allclose(mean.table(), 0.5 * (first + second),
                       rtol=0.0, atol=1e-14):
        raise AssertionError("batch-mean accumulator is incorrect")

    exact_sketch = RandomCompactorSketch(64, seed=7)
    exact_sketch.update(points[:, 0])
    if not np.array_equal(
            exact_sketch.quantiles(grid),
            inverted_empirical_quantiles(points[:, 0], grid)):
        raise AssertionError("uncompacted sketch must be exact")

    compacted_a = RandomCompactorSketch(8, seed=19)
    compacted_b = RandomCompactorSketch(8, seed=19)
    compacted_a.update(points[:, 0])
    compacted_b.update(points[:10, 0])
    compacted_b.update(points[10:, 0])
    if compacted_a.retained_weight != len(points):
        raise AssertionError("sketch did not preserve stream weight")
    qa = compacted_a.quantiles(grid)
    if np.any(np.diff(qa) < 0.0) or not np.all(np.isin(qa, points[:, 0])):
        raise AssertionError("sketch quantiles lost order or observed support")

    reservoir_a = RawReservoirProjectedAccumulator(
        2, directions, 16, 12, seed=23)
    reservoir_b = RawReservoirProjectedAccumulator(
        2, directions, 16, 12, seed=23)
    reservoir_a.update(points)
    reservoir_b.update(points)
    if not np.array_equal(reservoir_a.table(), reservoir_b.table()):
        raise AssertionError("reservoir replay is not deterministic")

    shift = np.asarray([0.7, -0.3])
    initial = np.zeros((16, 2))
    table = projected_quantile_table(
        np.repeat(shift[None, :], 20, axis=0), directions, grid)
    reconstructed, _, _ = reconstruct_from_quantile_table(
        initial, directions, table, steps=1, step_size=1.0)
    if not np.allclose(reconstructed, shift[None, :],
                       rtol=0.0, atol=1e-14):
        raise AssertionError("reconstruction translation invariant failed")


if __name__ == "__main__":
    invariant_tests()
    print("Projected quantile accumulator invariants: PASS")

