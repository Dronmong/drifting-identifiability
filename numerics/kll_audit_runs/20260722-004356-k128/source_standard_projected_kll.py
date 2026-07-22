"""Apache DataSketches KLL target statistics for confirmatory PSQT.

This module is intentionally separate from the local fixed-capacity
``KLLStyleProjectedAccumulator`` used during development.  The promoted
quality arm uses Apache DataSketches' maintained KLL implementation and pins
both Python and the package version in the protocol.

Apache's Python API does not expose a seed for KLL's internal randomized
compactions.  Exact trained states are therefore serialized into each run
artifact.  The target stream and every other RNG remain explicitly seeded.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata
import math

import numpy as np
from numpy.linalg import norm

try:
    import datasketches
except ImportError as error:  # pragma: no cover - exercised by environment
    datasketches = None
    _IMPORT_ERROR = error
else:
    _IMPORT_ERROR = None

from persistent_quantile_transport import midpoint_grid


REQUIRED_DATASKETCHES_VERSION = "5.2.0"


def require_datasketches() -> None:
    if datasketches is None:
        raise RuntimeError(
            "Apache DataSketches is required; run with Python 3.12 and "
            "`--with datasketches==5.2.0`") from _IMPORT_ERROR
    installed = importlib.metadata.version("datasketches")
    if installed != REQUIRED_DATASKETCHES_VERSION:
        raise RuntimeError(
            f"expected datasketches {REQUIRED_DATASKETCHES_VERSION}, "
            f"found {installed}")


def _directions(values: np.ndarray, dimension: int) -> np.ndarray:
    directions = np.asarray(values, dtype=float)
    if (directions.ndim != 2 or directions.shape[1] != dimension or
            len(directions) < 1 or not np.all(np.isfinite(directions))):
        raise ValueError("directions must be a finite (L, dimension) matrix")
    lengths = norm(directions, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("directions must be nonzero")
    return directions / lengths[:, None]


@dataclass(frozen=True)
class StandardKLLLedger:
    target_samples: int
    projection_dot_products: int
    retained_items: int
    serialized_bytes: int
    persistent_bytes: int
    peak_working_bytes: int
    normalized_rank_error: float


class ApacheKLLProjectedAccumulator:
    """One maintained Apache KLL float sketch per fixed projection."""

    def __init__(self, dimension: int, directions: np.ndarray,
                 knot_count: int, *, k: int = 128) -> None:
        require_datasketches()
        if dimension < 1 or knot_count < 2:
            raise ValueError("dimension must be positive and knots >= 2")
        if not 8 <= k <= 65535:
            raise ValueError("Apache KLL k must lie in [8, 65535]")
        self.dimension = int(dimension)
        self.directions = _directions(directions, dimension).copy()
        self.grid = midpoint_grid(knot_count)
        self.k = int(k)
        self.sketches = [
            datasketches.kll_floats_sketch(self.k)
            for _ in range(len(self.directions))
        ]
        self.target_samples = 0
        self.projection_dot_products = 0
        self._peak_projection_bytes = 0

    def update(self, points: np.ndarray) -> None:
        values = np.asarray(points, dtype=float)
        if (values.ndim != 2 or values.shape[1] != self.dimension or
                len(values) < 1 or not np.all(np.isfinite(values))):
            raise ValueError("points must be a nonempty finite matrix")
        projected = np.asarray(
            values @ self.directions.T, dtype=np.float32, order="F")
        for ell, sketch in enumerate(self.sketches):
            sketch.update(np.ascontiguousarray(projected[:, ell]))
        self.target_samples += len(values)
        self.projection_dot_products += len(values) * len(self.directions)
        self._peak_projection_bytes = max(
            self._peak_projection_bytes, int(projected.nbytes))
        self._check_counts()

    def table(self) -> np.ndarray:
        if self.target_samples < 1:
            raise ValueError("cannot query an empty accumulator")
        rows = [
            np.asarray(
                sketch.get_quantiles(self.grid.tolist(), inclusive=True),
                dtype=float)
            for sketch in self.sketches
        ]
        result = np.vstack(rows)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("KLL returned non-finite quantiles")
        result = np.maximum.accumulate(result, axis=1)
        if np.any(np.diff(result, axis=1) < 0.0):
            raise AssertionError("KLL quantile table is not monotone")
        return result

    def serialize(self) -> list[bytes]:
        self._check_counts()
        return [bytes(sketch.serialize()) for sketch in self.sketches]

    @classmethod
    def from_serialized(cls, dimension: int, directions: np.ndarray,
                        knot_count: int, payloads: list[bytes], *,
                        k: int = 128) -> "ApacheKLLProjectedAccumulator":
        result = cls(dimension, directions, knot_count, k=k)
        if len(payloads) != len(result.sketches):
            raise ValueError("one serialized sketch is required per direction")
        result.sketches = [
            datasketches.kll_floats_sketch.deserialize(payload)
            for payload in payloads
        ]
        counts = [int(sketch.n) for sketch in result.sketches]
        if counts and any(count != counts[0] for count in counts):
            raise ValueError("serialized projection sketches have unequal n")
        result.target_samples = counts[0] if counts else 0
        result.projection_dot_products = 0
        result._check_counts()
        return result

    def ledger(self) -> StandardKLLLedger:
        payloads = self.serialize()
        serialized = sum(len(payload) for payload in payloads)
        fixed = self.directions.nbytes + self.grid.nbytes
        retained = sum(int(sketch.num_retained) for sketch in self.sketches)
        return StandardKLLLedger(
            target_samples=self.target_samples,
            projection_dot_products=self.projection_dot_products,
            retained_items=retained,
            serialized_bytes=serialized,
            persistent_bytes=int(fixed + serialized),
            peak_working_bytes=int(fixed + serialized +
                                   self._peak_projection_bytes),
            normalized_rank_error=float(
                datasketches.kll_floats_sketch.get_normalized_rank_error(
                    self.k, False)),
        )

    def _check_counts(self) -> None:
        counts = [int(sketch.n) for sketch in self.sketches]
        if any(count != self.target_samples for count in counts):
            raise AssertionError("projection sketch stream counts disagree")


def empirical_rank_error(sample: np.ndarray, probabilities: np.ndarray,
                         estimates: np.ndarray) -> np.ndarray:
    """Smallest absolute rank error, respecting tie-valued rank intervals."""
    values = np.sort(np.asarray(sample, dtype=np.float32).reshape(-1))
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    estimates = np.asarray(estimates, dtype=np.float32).reshape(-1)
    if probabilities.shape != estimates.shape or len(values) < 1:
        raise ValueError("rank-error inputs have incompatible shapes")
    lower = np.searchsorted(values, estimates, side="left") / len(values)
    upper = np.searchsorted(values, estimates, side="right") / len(values)
    return np.maximum.reduce([
        lower - probabilities,
        probabilities - upper,
        np.zeros_like(probabilities),
    ])


def invariant_tests() -> None:
    require_datasketches()
    rng = np.random.default_rng(20260724)
    points = rng.normal(size=(1000, 2))
    directions = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    accumulator = ApacheKLLProjectedAccumulator(
        2, directions, 32, k=128)
    accumulator.update(points[:400])
    accumulator.update(points[400:])
    table = accumulator.table()
    if table.shape != (2, 32) or np.any(np.diff(table, axis=1) < 0.0):
        raise AssertionError("standard projected KLL table invariant failed")
    if any(int(sketch.n) != len(points) for sketch in accumulator.sketches):
        raise AssertionError("standard KLL stream count is wrong")
    restored = ApacheKLLProjectedAccumulator.from_serialized(
        2, directions, 32, accumulator.serialize(), k=128)
    if not np.array_equal(restored.table(), table):
        raise AssertionError("serialized KLL state does not replay exactly")
    ledger = accumulator.ledger()
    if (ledger.serialized_bytes <= 0 or ledger.retained_items <= 0 or
            ledger.persistent_bytes >= points.nbytes * len(directions)):
        raise AssertionError("standard KLL storage ledger is implausible")
    errors = empirical_rank_error(
        points[:, 0], accumulator.grid, table[0])
    if not np.all(np.isfinite(errors)) or errors.max() > 0.10:
        raise AssertionError("standard KLL rank-error smoke bound failed")


if __name__ == "__main__":
    invariant_tests()
    version = importlib.metadata.version("datasketches")
    error = datasketches.kll_floats_sketch.get_normalized_rank_error(
        128, False)
    print(
        "Apache projected KLL invariants: PASS "
        f"(datasketches={version}, k128_rank_error={error:.6f})")

