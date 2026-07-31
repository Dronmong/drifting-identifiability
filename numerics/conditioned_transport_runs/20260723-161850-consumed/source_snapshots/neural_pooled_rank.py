"""Neural amortization primitives for persistent pooled-rank supervision.

This is the Phase-0 implementation described in
``KLLPSQTNeuralAmortizationResearch.md``.  It intentionally contains no
training benchmark or model-selection logic.  The module supplies:

* immutable exact and Apache-KLL target quantile atlases;
* the scaled empirical sliced-W2 rank-matching loss;
* the matching free-particle PSQT correction;
* exact Run-Sort-ReRun (RSR) microbatch backpropagation; and
* finite-direction frame diagnostics.

The target atlas may persist across optimizer steps.  Generated ranks may not:
they must always be recomputed from a population generated at the current
parameter value.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor

from persistent_quantile_transport import midpoint_grid
from projected_quantile_accumulators import projected_quantile_table


@dataclass(frozen=True)
class QuantileAtlas:
    """Immutable quantile table for one fixed projection bank.

    ``quantiles`` has shape ``(directions, probability_knots)``.  KLL payloads
    are optional because an exact finalized table does not have a sketch state.
    Arrays are copied, normalized, and made read-only during construction.
    """

    directions: np.ndarray
    grid: np.ndarray
    quantiles: np.ndarray
    source: str
    target_count: int
    normalized_rank_error: float | None = None
    sketch_payloads: tuple[bytes, ...] | None = None

    def __post_init__(self) -> None:
        directions = np.asarray(self.directions, dtype=float)
        grid = np.asarray(self.grid, dtype=float).reshape(-1)
        quantiles = np.asarray(self.quantiles, dtype=float)
        if (
            directions.ndim != 2
            or directions.shape[0] < 1
            or directions.shape[1] < 1
            or not np.all(np.isfinite(directions))
        ):
            raise ValueError("directions must be a nonempty finite matrix")
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths <= 0.0):
            raise ValueError("atlas directions must be nonzero")
        directions = directions / lengths[:, None]
        if (
            len(grid) < 2
            or not np.all(np.isfinite(grid))
            or not np.all((0.0 < grid) & (grid < 1.0))
            or not np.all(np.diff(grid) > 0.0)
        ):
            raise ValueError("atlas grid must increase strictly inside (0,1)")
        if quantiles.shape != (len(directions), len(grid)):
            raise ValueError("quantile table shape does not match atlas")
        if not np.all(np.isfinite(quantiles)) or np.any(
            np.diff(quantiles, axis=1) < 0.0
        ):
            raise ValueError("atlas quantiles must be finite and nondecreasing")
        if not self.source:
            raise ValueError("atlas source must be nonempty")
        if self.target_count < 1:
            raise ValueError("atlas target_count must be positive")
        error = self.normalized_rank_error
        if error is not None and (not math.isfinite(error) or error < 0.0):
            raise ValueError("normalized rank error must be nonnegative")
        payloads = self.sketch_payloads
        if payloads is not None and len(payloads) != len(directions):
            raise ValueError("one serialized sketch is required per direction")

        directions = directions.copy()
        grid = grid.copy()
        quantiles = quantiles.copy()
        directions.setflags(write=False)
        grid.setflags(write=False)
        quantiles.setflags(write=False)
        object.__setattr__(self, "directions", directions)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "quantiles", quantiles)
        if payloads is not None:
            object.__setattr__(self, "sketch_payloads", tuple(payloads))

    @property
    def dimension(self) -> int:
        return int(self.directions.shape[1])

    @property
    def direction_count(self) -> int:
        return int(self.directions.shape[0])

    @property
    def knot_count(self) -> int:
        return int(len(self.grid))

    def ordered_quantiles(self, population_size: int) -> np.ndarray:
        """Return an ``(population_size, L)`` midpoint target table."""
        if population_size < 2:
            raise ValueError("population_size must be at least two")
        probabilities = midpoint_grid(population_size)
        if len(probabilities) == len(self.grid) and np.array_equal(
            probabilities, self.grid
        ):
            return self.quantiles.T.copy()
        return np.column_stack(
            [np.interp(probabilities, self.grid, row) for row in self.quantiles]
        )

    def torch_directions(self, reference: Tensor) -> Tensor:
        return torch.tensor(
            np.asarray(self.directions).copy(),
            dtype=reference.dtype,
            device=reference.device,
        )

    def torch_ordered_quantiles(
        self, population_size: int, reference: Tensor
    ) -> Tensor:
        return torch.tensor(
            self.ordered_quantiles(population_size),
            dtype=reference.dtype,
            device=reference.device,
        )

    def select_directions(self, indices: np.ndarray | list[int]) -> QuantileAtlas:
        """Return an immutable atlas restricted to distinct registered rows."""
        selected = np.asarray(indices, dtype=np.int64).reshape(-1)
        if (
            len(selected) < 1
            or np.any(selected < 0)
            or np.any(selected >= self.direction_count)
            or len(np.unique(selected)) != len(selected)
        ):
            raise ValueError("direction indices must be distinct and in range")
        payloads = (
            None
            if self.sketch_payloads is None
            else tuple(self.sketch_payloads[int(index)] for index in selected)
        )
        return QuantileAtlas(
            directions=self.directions[selected],
            grid=self.grid,
            quantiles=self.quantiles[selected],
            source=f"{self.source}|registered-subset-{len(selected)}",
            target_count=self.target_count,
            normalized_rank_error=self.normalized_rank_error,
            sketch_payloads=payloads,
        )


def balanced_orthogonal_block_schedule(
    directions: np.ndarray,
    active_direction_count: int,
    steps: int,
    rng: np.random.Generator,
    *,
    orthogonality_tolerance: float = 1e-10,
) -> tuple[np.ndarray, ...]:
    """Schedule complete registered orthogonal blocks with balanced exposure.

    Directions are interpreted as contiguous ``d x d`` blocks.  At every
    step the scheduler selects whole blocks among those with the least prior
    exposure, using ``rng`` only to break ties.  Consequently no direction is
    duplicated within a step and final block exposures differ by at most one.
    """
    values = np.asarray(directions, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 1
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("directions must be a nonempty finite matrix")
    count, dimension = values.shape
    active = int(active_direction_count)
    if count % dimension:
        raise ValueError("direction bank must contain complete d-by-d blocks")
    if active < dimension or active > count or active % dimension:
        raise ValueError("active directions must be complete in-range blocks")
    if steps < 1:
        raise ValueError("schedule steps must be positive")
    if (
        not math.isfinite(orthogonality_tolerance)
        or orthogonality_tolerance <= 0.0
    ):
        raise ValueError("orthogonality tolerance must be finite and positive")
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("directions must be nonzero")
    normalized = values / lengths[:, None]
    block_count = count // dimension
    for block in range(block_count):
        start = block * dimension
        current = normalized[start : start + dimension]
        defect = current @ current.T - np.eye(dimension)
        if np.linalg.norm(defect, ord=2) > orthogonality_tolerance:
            raise ValueError("registered direction block is not orthogonal")

    if active == count:
        full = np.arange(count, dtype=np.int64)
        full.setflags(write=False)
        return tuple(full for _ in range(steps))

    active_blocks = active // dimension
    exposures = np.zeros(block_count, dtype=np.int64)
    schedule: list[np.ndarray] = []
    for _ in range(steps):
        tie_break = rng.random(block_count)
        block_order = np.lexsort((tie_break, exposures))
        selected_blocks = np.sort(block_order[:active_blocks])
        exposures[selected_blocks] += 1
        selected = np.concatenate(
            [
                np.arange(
                    block * dimension,
                    (block + 1) * dimension,
                    dtype=np.int64,
                )
                for block in selected_blocks
            ]
        )
        selected.setflags(write=False)
        schedule.append(selected)
    if int(exposures.max() - exposures.min()) > 1:
        raise AssertionError("balanced block scheduler produced unequal exposure")
    return tuple(schedule)


def evenly_spaced_step_indices(total_steps: int, active_steps: int) -> np.ndarray:
    """Choose midpoint-stratified active steps with no temporal front-loading."""
    total = int(total_steps)
    active = int(active_steps)
    if total < 1 or active < 0 or active > total:
        raise ValueError("active steps must lie between zero and total steps")
    if active == 0:
        result = np.empty(0, dtype=np.int64)
    else:
        result = np.floor(
            (np.arange(active, dtype=float) + 0.5) * total / active
        ).astype(np.int64)
    if len(np.unique(result)) != len(result):
        raise AssertionError("stratified step schedule contains a duplicate")
    result.setflags(write=False)
    return result


def exact_target_atlas(
    points: np.ndarray, directions: np.ndarray, knot_count: int | None = None
) -> QuantileAtlas:
    """Finalize an exact inverted-ECDF target atlas.

    The raw target observations are not retained.  When ``knot_count`` equals
    the number of points, midpoint queries recover the sorted projections
    exactly.
    """
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("points must be a finite matrix with at least 2 rows")
    knots = len(values) if knot_count is None else int(knot_count)
    grid = midpoint_grid(knots)
    table = projected_quantile_table(values, directions, grid, method="inverted_cdf")
    return QuantileAtlas(
        directions=np.asarray(directions, dtype=float),
        grid=grid,
        quantiles=table,
        source="exact-inverted-ecdf",
        target_count=len(values),
    )


def apache_kll_target_atlas(
    target_batches: np.ndarray | Iterable[np.ndarray],
    directions: np.ndarray,
    knot_count: int,
    *,
    k: int = 128,
) -> QuantileAtlas:
    """Build a serialized Apache DataSketches KLL projection atlas.

    Import is lazy so exact-atlas users do not need DataSketches installed.
    DataSketches 5.2.0 currently requires Python 3.12 on the repository's
    Windows setup; ``standard_projected_kll`` enforces that pinned version.
    """
    from standard_projected_kll import ApacheKLLProjectedAccumulator

    dirs = np.asarray(directions, dtype=float)
    if dirs.ndim != 2:
        raise ValueError("directions must be a matrix")
    accumulator = ApacheKLLProjectedAccumulator(
        dirs.shape[1], dirs, int(knot_count), k=int(k)
    )
    batches: Iterable[np.ndarray]
    if isinstance(target_batches, np.ndarray):
        batches = (target_batches,)
    else:
        batches = target_batches
    updated = False
    for batch in batches:
        accumulator.update(np.asarray(batch, dtype=float))
        updated = True
    if not updated:
        raise ValueError("target_batches must contain at least one batch")
    ledger = accumulator.ledger()
    return QuantileAtlas(
        directions=accumulator.directions,
        grid=accumulator.grid,
        quantiles=accumulator.table(),
        source=f"apache-kll-k{int(k)}",
        target_count=accumulator.target_samples,
        normalized_rank_error=ledger.normalized_rank_error,
        sketch_payloads=tuple(accumulator.serialize()),
    )


def _validate_features(features: Tensor, atlas: QuantileAtlas) -> None:
    if (
        features.ndim != 2
        or features.shape[0] < 1
        or features.shape[1] != atlas.dimension
    ):
        raise ValueError("features must have shape (B, atlas.dimension) with B >= 1")
    if not features.is_floating_point():
        raise ValueError("features must have a floating dtype")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features must be finite")


@dataclass(frozen=True)
class RankAssignment:
    """Fixed target assignment produced by a detached global Sort step."""

    assigned_targets: Tensor
    order: Tensor
    ranks: Tensor
    ordered_targets: Tensor


def _rank_assignment_from_projected(
    projected: Tensor, atlas: QuantileAtlas, features: Tensor
) -> RankAssignment:
    """Assign fixed atlas targets from an already computed projection matrix."""
    expected = (features.shape[0], atlas.direction_count)
    if projected.shape != expected:
        raise ValueError("projected features have the wrong shape")
    with torch.no_grad():
        detached = projected.detach()
        order = torch.argsort(detached, dim=0, stable=True)
        ordered = atlas.torch_ordered_quantiles(len(features), features)
        assigned = torch.empty_like(detached)
        assigned.scatter_(0, order, ordered)
        rank_values = torch.arange(
            len(features), device=features.device, dtype=torch.long
        )
        rank_values = rank_values[:, None].expand_as(order)
        ranks = torch.empty_like(order)
        ranks.scatter_(0, order, rank_values)
    return RankAssignment(
        assigned_targets=assigned.detach(),
        order=order.detach(),
        ranks=ranks.detach(),
        ordered_targets=ordered.detach(),
    )


def assign_rank_targets(features: Tensor, atlas: QuantileAtlas) -> RankAssignment:
    """Assign atlas targets to generated features by global projected rank."""
    _validate_features(features, atlas)
    population = int(features.shape[0])
    if population < 2:
        raise ValueError("rank assignment requires at least two features")
    directions = atlas.torch_directions(features)
    projected = features.detach() @ directions.T
    return _rank_assignment_from_projected(
        projected, atlas, features
    )


def loss_from_rank_assignment(
    features: Tensor,
    atlas: QuantileAtlas,
    assignment: RankAssignment,
    *,
    normalizer_population: int | None = None,
) -> Tensor:
    """Evaluate a fixed-rank loss, optionally as one RSR microbatch.

    For a complete population this is

    ``d / (2 L B) * sum_(j,l) (dot(h_j,u_l) - target_(j,l))^2``.

    RSR chunks pass the full effective population as ``normalizer_population``
    and their scalar losses are summed before the optimizer update.
    """
    _validate_features(features, atlas)
    targets = assignment.assigned_targets
    expected = (features.shape[0], atlas.direction_count)
    if targets.shape != expected:
        raise ValueError("rank assignment shape does not match feature chunk")
    population = (
        int(features.shape[0])
        if normalizer_population is None
        else int(normalizer_population)
    )
    if population < int(features.shape[0]):
        raise ValueError("normalizer population cannot be smaller than chunk")
    directions = atlas.torch_directions(features)
    residual = features @ directions.T - targets
    scale = atlas.dimension / (2.0 * atlas.direction_count * population)
    return scale * residual.square().sum()


@dataclass(frozen=True)
class RankMatchedLoss:
    loss: Tensor
    assignment: RankAssignment


def rank_matched_loss(features: Tensor, atlas: QuantileAtlas) -> RankMatchedLoss:
    """Compute the full-population scaled empirical sliced-W2 loss."""
    assignment = assign_rank_targets(features, atlas)
    return RankMatchedLoss(
        loss=loss_from_rank_assignment(features, atlas, assignment),
        assignment=assignment,
    )


def psqt_feature_correction(features: Tensor, atlas: QuantileAtlas) -> Tensor:
    """Return the repository's free-particle tight-frame correction.

    If ``J`` is ``rank_matched_loss(features, atlas).loss`` and there are ``B``
    features away from a rank crossing, this correction is ``-B * grad J``.
    """
    _validate_features(features, atlas)
    if len(features) < 2:
        raise ValueError("PSQT correction requires at least two features")
    directions = atlas.torch_directions(features)
    projected = features @ directions.T
    assignment = _rank_assignment_from_projected(projected, atlas, features)
    return (
        (atlas.dimension / atlas.direction_count)
        * (assignment.assigned_targets - projected)
        @ directions
    )


@torch.no_grad()
def normalized_paper_field(
    query: Tensor, positive: Tensor, *, tau: float = 0.5, mask: bool = True
) -> Tensor:
    """Evaluate the row/column-normalized Algorithm-2 local field.

    If ``K`` is the concatenated positive/negative Laplace-kernel matrix, the
    geometric mean of its row and column softmaxes is exactly
    ``K / sqrt((K 1) (K.T 1))``.  The direct mass form uses one raw-kernel
    exponential.  A log-sum-exp path handles the exceptional case where an
    entire raw-kernel row or column underflows at the requested dtype.
    """
    if (
        query.ndim != 2
        or positive.ndim != 2
        or len(query) < 1
        or len(positive) < 1
        or query.shape[1] != positive.shape[1]
        or not query.is_floating_point()
        or not positive.is_floating_point()
        or query.dtype != positive.dtype
        or query.device != positive.device
        or not math.isfinite(tau)
        or tau <= 0.0
        or not bool(torch.isfinite(query).all())
        or not bool(torch.isfinite(positive).all())
    ):
        raise ValueError("invalid paper-field inputs")

    positive_distance = torch.cdist(query, positive, p=2.0)
    negative_distance = torch.cdist(query, query, p=2.0)
    logits = torch.cat(
        [-positive_distance / tau, -negative_distance / tau], dim=1
    )
    if mask:
        rows = torch.arange(len(query), device=query.device)
        logits[rows, len(positive) + rows] -= 1e6 / tau

    kernel = torch.exp(logits)
    row_mass = kernel.sum(dim=1, keepdim=True)
    column_mass = kernel.sum(dim=0, keepdim=True)
    mass_product = row_mass * column_mass
    if bool((mass_product == 0.0).any()):
        log_row_mass = torch.logsumexp(logits, dim=1, keepdim=True)
        log_column_mass = torch.logsumexp(logits, dim=0, keepdim=True)
        affinity = torch.exp(
            logits - 0.5 * (log_row_mass + log_column_mass)
        )
    else:
        affinity = kernel * torch.rsqrt(mass_product)

    positive_affinity = affinity[:, : len(positive)]
    negative_affinity = affinity[:, len(positive) :]
    positive_mass = positive_affinity.sum(dim=1, keepdim=True)
    negative_mass = negative_affinity.sum(dim=1, keepdim=True)
    positive_numerator = positive_affinity @ positive
    negative_numerator = negative_affinity @ query
    return (
        negative_mass * positive_numerator
        - positive_mass * negative_numerator
    )


@dataclass(frozen=True)
class WeightedRepresentativeSet:
    """Integer-multiplicity representatives of every row in a finite support."""

    centers: Tensor
    multiplicities: Tensor
    assignment: Tensor
    rms_radius: float
    max_radius: float
    strategy: str
    split_count: int
    unique_split_direction_count: int
    tail_reserve_count: int
    partition_projection_scalar_products: int
    partition_sort_work: float


REPRESENTATIVE_STRATEGIES = (
    "fixed-level",
    "variance-per-node",
    "radius-priority",
    "variance-with-tail-reserve",
)


def _representative_split(
    leaf: Tensor,
    projections: Tensor,
    *,
    fixed_direction: int | None,
) -> tuple[Tensor, Tensor, int, float]:
    """Median-split one nontrivial leaf along a registered direction."""
    if leaf.ndim != 1 or len(leaf) < 2:
        raise ValueError("only a nontrivial one-dimensional leaf may be split")
    if fixed_direction is None:
        leaf_projections = projections[leaf]
        variances = torch.var(leaf_projections, dim=0, correction=0)
        direction = int(torch.argmax(variances).cpu())
    else:
        direction = int(fixed_direction)
        leaf_projections = projections[leaf, direction][:, None]
    values = leaf_projections[:, direction if fixed_direction is None else 0]
    order = torch.argsort(values, stable=True)
    ordered = leaf[order]
    midpoint = len(ordered) // 2
    if midpoint < 1 or midpoint >= len(ordered):
        raise AssertionError("representative median split became empty")
    sort_work = float(len(leaf) * math.log2(len(leaf)))
    return ordered[:midpoint], ordered[midpoint:], direction, sort_work


def _representative_leaf_radius(points: Tensor, leaf: Tensor) -> float:
    centered = points[leaf] - torch.mean(points[leaf], dim=0, keepdim=True)
    return float(torch.max(torch.sum(centered.square(), dim=1)).cpu())


@torch.no_grad()
def projection_tree_representatives(
    points: Tensor,
    directions: Tensor,
    representative_count: int,
    *,
    strategy: str = "fixed-level",
    tail_reserve_fraction: float = 0.0625,
) -> WeightedRepresentativeSet:
    """Compress a support using an audited registered-direction partition.

    ``fixed-level`` is the historical control: one predetermined direction is
    used at every level. ``variance-per-node`` retains equal leaf occupancy
    but chooses each node's largest-variance registered direction.
    ``radius-priority`` spends capacity on the currently widest leaf.
    ``variance-with-tail-reserve`` first preserves points with the largest
    robust projected-extremeness scores as singleton leaves, then applies
    radius-priority variance splits to the remaining support.
    """
    if (
        points.ndim != 2
        or len(points) < 2
        or not points.is_floating_point()
        or not bool(torch.isfinite(points).all())
        or directions.ndim != 2
        or len(directions) < 1
        or directions.shape[1] != points.shape[1]
        or directions.dtype != points.dtype
        or directions.device != points.device
        or not bool(torch.isfinite(directions).all())
    ):
        raise ValueError("invalid representative-tree inputs")
    count = int(representative_count)
    if (
        count < 1
        or count > len(points)
        or count & (count - 1)
        or len(points) % count
    ):
        raise ValueError(
            "representative count must be a power of two dividing the support"
        )
    lengths = torch.linalg.vector_norm(directions, dim=1)
    if bool((lengths <= 0.0).any()):
        raise ValueError("representative split directions must be nonzero")
    if strategy not in REPRESENTATIVE_STRATEGIES:
        raise ValueError(
            "representative strategy must be one of "
            + ", ".join(REPRESENTATIVE_STRATEGIES)
        )
    if (
        not math.isfinite(tail_reserve_fraction)
        or tail_reserve_fraction < 0.0
        or tail_reserve_fraction >= 1.0
    ):
        raise ValueError("tail reserve fraction must lie in [0, 1)")
    unit_directions = directions / lengths[:, None]
    levels = count.bit_length() - 1
    all_indices = torch.arange(len(points), device=points.device, dtype=torch.long)
    split_directions: list[int] = []
    tail_reserve_count = 0
    partition_sort_work = 0.0

    if strategy == "fixed-level":
        ordered_indices = all_indices[None, :]
        selected_directions = unit_directions[
            torch.arange(levels, device=points.device) % len(unit_directions)
        ]
        projections = points @ selected_directions.T
        for level in range(levels):
            cluster_projection = projections[:, level][ordered_indices]
            order = torch.argsort(cluster_projection, dim=1, stable=True)
            ordered_indices = torch.gather(ordered_indices, 1, order)
            next_cluster_size = ordered_indices.shape[1] // 2
            if next_cluster_size < 1:
                raise AssertionError("balanced representative split became empty")
            partition_sort_work += float(
                sum(
                    len(row) * math.log2(len(row))
                    for row in ordered_indices
                )
            )
            split_directions.extend(
                [int(level % len(unit_directions))] * len(ordered_indices)
            )
            # Every row is one current cluster.  Reshaping its sorted low/high
            # halves creates all children in one tensor operation, avoiding a
            # Python loop and one tiny argsort launch per cluster.
            ordered_indices = ordered_indices.reshape(-1, next_cluster_size)
        leaves = [row for row in ordered_indices]
        projection_products = len(points) * levels
    else:
        projections = points @ unit_directions.T
        projection_products = len(points) * len(unit_directions)
        if strategy == "variance-with-tail-reserve" and count > 1:
            requested_tail = max(1, int(round(count * tail_reserve_fraction)))
            tail_reserve_count = min(requested_tail, count - 1, len(points) - 1)
            if tail_reserve_count:
                median = torch.median(projections, dim=0).values
                lower = torch.quantile(projections, 0.25, dim=0)
                upper = torch.quantile(projections, 0.75, dim=0)
                scale = (upper - lower).clamp_min(torch.finfo(points.dtype).eps)
                extremeness = torch.max(
                    torch.abs(projections - median[None, :]) / scale[None, :],
                    dim=1,
                ).values
                extreme_order = torch.argsort(
                    extremeness, descending=True, stable=True
                )
                reserved = extreme_order[:tail_reserve_count]
                retained_mask = torch.ones(
                    len(points), dtype=torch.bool, device=points.device
                )
                retained_mask[reserved] = False
                leaves = [index[None] for index in reserved]
                leaves.append(all_indices[retained_mask])
            else:
                leaves = [all_indices]
        else:
            leaves = [all_indices]

        if strategy == "variance-per-node":
            ordered_indices = all_indices[None, :]
            while len(ordered_indices) < count:
                leaf_projections = projections[ordered_indices]
                variances = torch.var(leaf_projections, dim=1, correction=0)
                chosen_directions = torch.argmax(variances, dim=1)
                chosen_values = torch.gather(
                    leaf_projections,
                    2,
                    chosen_directions[:, None, None].expand(
                        -1, leaf_projections.shape[1], 1
                    ),
                )[:, :, 0]
                order = torch.argsort(chosen_values, dim=1, stable=True)
                ordered_indices = torch.gather(ordered_indices, 1, order)
                leaf_size = ordered_indices.shape[1]
                partition_sort_work += float(
                    len(ordered_indices) * leaf_size * math.log2(leaf_size)
                )
                split_directions.extend(
                    map(int, chosen_directions.detach().cpu().tolist())
                )
                ordered_indices = ordered_indices.reshape(
                    -1, ordered_indices.shape[1] // 2
                )
            leaves = [leaf for leaf in ordered_indices]
        else:
            leaf_radii = [
                _representative_leaf_radius(points, leaf)
                if len(leaf) >= 2
                else 0.0
                for leaf in leaves
            ]
            while len(leaves) < count:
                candidates = [
                    (position, leaf, leaf_radii[position])
                    for position, leaf in enumerate(leaves)
                    if len(leaf) >= 2
                ]
                if not candidates:
                    raise AssertionError(
                        "representative refinement exhausted splittable leaves"
                    )
                position, leaf, _ = max(
                    candidates,
                    key=lambda item: (
                        item[2],
                        len(item[1]),
                        -item[0],
                    ),
                )
                low, high, direction, work = _representative_split(
                    leaf, projections, fixed_direction=None
                )
                leaves[position : position + 1] = [low, high]
                leaf_radii[position : position + 1] = [
                    _representative_leaf_radius(points, low)
                    if len(low) >= 2
                    else 0.0,
                    _representative_leaf_radius(points, high)
                    if len(high) >= 2
                    else 0.0,
                ]
                split_directions.append(direction)
                partition_sort_work += work

    if len(leaves) != count or any(len(leaf) < 1 for leaf in leaves):
        raise AssertionError("representative tree produced invalid leaves")

    centers = torch.stack([points[leaf].mean(dim=0) for leaf in leaves])
    multiplicities = torch.tensor(
        [len(leaf) for leaf in leaves],
        dtype=points.dtype,
        device=points.device,
    )
    assignment = torch.empty(len(points), dtype=torch.long, device=points.device)
    for representative, leaf in enumerate(leaves):
        assignment[leaf] = representative
    residual = points - centers[assignment]
    squared_radius = torch.sum(residual.square(), dim=1)
    rms_radius = float(torch.sqrt(torch.mean(squared_radius)).cpu())
    max_radius = float(torch.sqrt(torch.max(squared_radius)).cpu())
    return WeightedRepresentativeSet(
        centers=centers,
        multiplicities=multiplicities,
        assignment=assignment,
        rms_radius=rms_radius,
        max_radius=max_radius,
        strategy=strategy,
        split_count=len(split_directions),
        unique_split_direction_count=len(set(split_directions)),
        tail_reserve_count=tail_reserve_count,
        partition_projection_scalar_products=projection_products,
        partition_sort_work=partition_sort_work,
    )


def _validate_representatives(
    representatives: WeightedRepresentativeSet,
    reference: Tensor,
    *,
    require_support_size: int | None = None,
) -> None:
    centers = representatives.centers
    multiplicities = representatives.multiplicities
    assignment = representatives.assignment
    if (
        centers.ndim != 2
        or len(centers) < 1
        or centers.shape[1] != reference.shape[1]
        or centers.dtype != reference.dtype
        or centers.device != reference.device
        or multiplicities.shape != (len(centers),)
        or multiplicities.dtype != reference.dtype
        or multiplicities.device != reference.device
        or assignment.ndim != 1
        or assignment.dtype != torch.long
        or assignment.device != reference.device
        or not bool(torch.isfinite(centers).all())
        or not bool(torch.isfinite(multiplicities).all())
        or bool((multiplicities < 1.0).any())
        or not bool(torch.equal(multiplicities, torch.round(multiplicities)))
        or bool((assignment < 0).any())
        or bool((assignment >= len(centers)).any())
        or representatives.strategy not in REPRESENTATIVE_STRATEGIES
        or representatives.split_count < 0
        or representatives.unique_split_direction_count < 0
        or representatives.unique_split_direction_count
        > representatives.split_count
        or representatives.tail_reserve_count < 0
        or representatives.tail_reserve_count >= len(centers)
        or representatives.partition_projection_scalar_products < 0
        or not math.isfinite(representatives.partition_sort_work)
        or representatives.partition_sort_work < 0.0
    ):
        raise ValueError("invalid weighted representatives")
    counts = torch.bincount(assignment, minlength=len(centers)).to(
        dtype=reference.dtype
    )
    if not bool(torch.equal(counts, multiplicities)):
        raise ValueError("representative multiplicities do not match assignment")
    if require_support_size is not None and len(assignment) != require_support_size:
        raise ValueError("representatives have the wrong original support size")


@torch.no_grad()
def weighted_normalized_paper_field(
    query: Tensor,
    positive: WeightedRepresentativeSet,
    negative: WeightedRepresentativeSet,
    *,
    tau: float = 0.5,
    mask: bool = True,
) -> Tensor:
    """Evaluate the normalized field on duplicated weighted representatives.

    Every original support column remains present conceptually, but all points
    assigned to one leaf use the same representative center.  Column masses
    are retained per conceptual negative column, which permits the reused
    query's own negative entry to be deleted before aggregation.
    """
    if (
        query.ndim != 2
        or len(query) < 2
        or not query.is_floating_point()
        or not bool(torch.isfinite(query).all())
        or not math.isfinite(tau)
        or tau <= 0.0
    ):
        raise ValueError("invalid weighted paper-field query")
    _validate_representatives(positive, query)
    _validate_representatives(negative, query, require_support_size=len(query))

    positive_kernel = torch.exp(-torch.cdist(query, positive.centers) / tau)
    negative_kernel = torch.exp(-torch.cdist(query, negative.centers) / tau)
    rows = torch.arange(len(query), device=query.device)
    negative_self = negative_kernel[rows, negative.assignment]

    row_mass = (
        positive_kernel @ positive.multiplicities
        + negative_kernel @ negative.multiplicities
    )
    if mask:
        row_mass = row_mass - negative_self

    positive_column_mass = positive_kernel.sum(dim=0)
    negative_column_mass = negative_kernel.sum(dim=0)[negative.assignment]
    if mask:
        negative_column_mass = negative_column_mass - negative_self
    if (
        bool((row_mass <= 0.0).any())
        or bool((positive_column_mass <= 0.0).any())
        or bool((negative_column_mass <= 0.0).any())
        or not bool(torch.isfinite(row_mass).all())
    ):
        raise ValueError("weighted field has a zero or non-finite normalizer")

    inverse_row = torch.rsqrt(row_mass)
    positive_column_factor = (
        positive.multiplicities * torch.rsqrt(positive_column_mass)
    )
    positive_affinity = (
        inverse_row[:, None]
        * positive_kernel
        * positive_column_factor[None, :]
    )

    inverse_negative_column = torch.rsqrt(negative_column_mass)
    negative_cluster_factor = torch.zeros(
        len(negative.centers), dtype=query.dtype, device=query.device
    )
    negative_cluster_factor.scatter_add_(
        0, negative.assignment, inverse_negative_column
    )
    negative_affinity = negative_kernel * negative_cluster_factor[None, :]
    if mask:
        negative_affinity[rows, negative.assignment] -= (
            negative_self * inverse_negative_column
        )
    negative_affinity = inverse_row[:, None] * negative_affinity

    positive_mass = positive_affinity.sum(dim=1, keepdim=True)
    negative_mass = negative_affinity.sum(dim=1, keepdim=True)
    positive_numerator = positive_affinity @ positive.centers
    negative_numerator = negative_affinity @ negative.centers
    return (
        negative_mass * positive_numerator
        - positive_mass * negative_numerator
    )


@torch.no_grad()
def weighted_paper_field_audit(
    query: Tensor,
    original_positive: Tensor,
    positive: WeightedRepresentativeSet,
    negative: WeightedRepresentativeSet,
    *,
    tau: float = 0.5,
    mask: bool = True,
    approximate_field: Tensor | None = None,
) -> dict[str, float]:
    """Compare a representative field and its masses with the dense field.

    This intentionally materializes the dense ``B x 2B`` comparison and is
    therefore a diagnostic, not part of the compressed training path.
    """
    if (
        original_positive.ndim != 2
        or original_positive.shape[1] != query.shape[1]
        or original_positive.dtype != query.dtype
        or original_positive.device != query.device
        or not bool(torch.isfinite(original_positive).all())
    ):
        raise ValueError("invalid original positive support for field audit")
    _validate_representatives(
        positive, query, require_support_size=len(original_positive)
    )
    _validate_representatives(negative, query, require_support_size=len(query))
    dense_positive_kernel = torch.exp(-torch.cdist(query, original_positive) / tau)
    dense_negative_kernel = torch.exp(-torch.cdist(query, query) / tau)
    approximate_positive_kernel = torch.exp(
        -torch.cdist(query, positive.centers) / tau
    )[:, positive.assignment]
    approximate_negative_kernel = torch.exp(
        -torch.cdist(query, negative.centers) / tau
    )[:, negative.assignment]
    if mask:
        rows = torch.arange(len(query), device=query.device)
        dense_negative_kernel[rows, rows] = 0.0
        approximate_negative_kernel[rows, rows] = 0.0
    dense_kernel = torch.cat((dense_positive_kernel, dense_negative_kernel), dim=1)
    approximate_kernel = torch.cat(
        (approximate_positive_kernel, approximate_negative_kernel), dim=1
    )
    dense_row_mass = dense_kernel.sum(dim=1)
    dense_column_mass = dense_kernel.sum(dim=0)
    approximate_row_mass = approximate_kernel.sum(dim=1)
    approximate_column_mass = approximate_kernel.sum(dim=0)
    dense_field = normalized_paper_field(
        query, original_positive, tau=tau, mask=mask
    )
    if approximate_field is None:
        approximate_field = weighted_normalized_paper_field(
            query, positive, negative, tau=tau, mask=mask
        )

    tiny = torch.finfo(query.dtype).tiny
    relative_field_l2 = torch.linalg.vector_norm(approximate_field - dense_field) / (
        torch.linalg.vector_norm(dense_field).clamp_min(tiny)
    )
    cosine = torch.sum(approximate_field * dense_field) / (
        torch.linalg.vector_norm(approximate_field)
        * torch.linalg.vector_norm(dense_field)
    ).clamp_min(tiny)
    relative_row_mass_l2 = torch.linalg.vector_norm(
        approximate_row_mass - dense_row_mass
    ) / torch.linalg.vector_norm(dense_row_mass).clamp_min(tiny)
    relative_column_mass_l2 = torch.linalg.vector_norm(
        approximate_column_mass - dense_column_mass
    ) / torch.linalg.vector_norm(dense_column_mass).clamp_min(tiny)
    return {
        "field_relative_l2_error": float(relative_field_l2.cpu()),
        "field_cosine": float(cosine.cpu()),
        "row_mass_relative_l2_error": float(relative_row_mass_l2.cpu()),
        "column_mass_relative_l2_error": float(relative_column_mass_l2.cpu()),
        "row_mass_max_relative_error": float(
            torch.max(
                torch.abs(approximate_row_mass - dense_row_mass)
                / dense_row_mass.clamp_min(tiny)
            ).cpu()
        ),
        "column_mass_max_relative_error": float(
            torch.max(
                torch.abs(approximate_column_mass - dense_column_mass)
                / dense_column_mass.clamp_min(tiny)
            ).cpu()
        ),
    }


@dataclass(frozen=True)
class FrameDiagnostics:
    operator: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    condition_number: float
    spectral_tightness_error: float
    frobenius_tightness_error: float


@dataclass(frozen=True)
class QuadraticFrameDiagnostics:
    """Diagnostics for sensing symmetric matrices through ``u^T Sigma u``."""

    sensing_matrix: np.ndarray
    singular_values: np.ndarray
    parameter_dimension: int
    rank: int
    smallest_singular_value: float
    condition_number: float

    @property
    def is_full_rank(self) -> bool:
        return self.rank == self.parameter_dimension


def quadratic_sensing_matrix(directions: np.ndarray) -> np.ndarray:
    """Return the Frobenius-isometric symmetric outer-product design.

    Diagonal coordinates are ``u_i^2`` and off-diagonal coordinates are
    ``sqrt(2) u_i u_j``.  Pairing this row with the symmetric parameter vector
    ``(Sigma_ii, sqrt(2) Sigma_ij)`` gives exactly ``u^T Sigma u``.
    """
    dirs = np.asarray(directions, dtype=float)
    if (
        dirs.ndim != 2
        or dirs.shape[0] < 1
        or dirs.shape[1] < 1
        or not np.all(np.isfinite(dirs))
    ):
        raise ValueError("directions must be a nonempty finite matrix")
    lengths = np.linalg.norm(dirs, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("directions must be nonzero")
    dirs = dirs / lengths[:, None]
    dimension = dirs.shape[1]
    columns = [dirs[:, coordinate] ** 2 for coordinate in range(dimension)]
    root_two = math.sqrt(2.0)
    columns.extend(
        root_two * dirs[:, left] * dirs[:, right]
        for left in range(dimension)
        for right in range(left + 1, dimension)
    )
    return np.column_stack(columns)


def quadratic_frame_diagnostics(
    directions: np.ndarray, *, rank_tolerance: float = 1e-12
) -> QuadraticFrameDiagnostics:
    """Audit whether projected variances stably determine covariance."""
    if not math.isfinite(rank_tolerance) or rank_tolerance <= 0.0:
        raise ValueError("rank_tolerance must be finite and positive")
    sensing = quadratic_sensing_matrix(directions)
    singular = np.linalg.svd(sensing, compute_uv=False)
    parameter_dimension = sensing.shape[1]
    threshold = rank_tolerance * max(1.0, float(singular[0]))
    rank = int(np.count_nonzero(singular > threshold))
    if rank == parameter_dimension:
        smallest = float(singular[-1])
        condition = float(singular[0] / singular[-1])
    else:
        smallest = 0.0
        condition = math.inf
    sensing = sensing.copy()
    singular = singular.copy()
    sensing.setflags(write=False)
    singular.setflags(write=False)
    return QuadraticFrameDiagnostics(
        sensing_matrix=sensing,
        singular_values=singular,
        parameter_dimension=parameter_dimension,
        rank=rank,
        smallest_singular_value=smallest,
        condition_number=condition,
    )


def _random_orthogonal_block(rng: np.random.Generator, dimension: int) -> np.ndarray:
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
    return (q * signs[None, :]).T


def extend_to_conditioned_quadratic_frame(
    directions: np.ndarray,
    rng: np.random.Generator,
    *,
    maximum_condition_number: float = 25.0,
    maximum_blocks: int = 32,
    rank_tolerance: float = 1e-12,
) -> tuple[np.ndarray, QuadraticFrameDiagnostics]:
    """Append orthogonal blocks until covariance sensing is well conditioned.

    Existing directions are preserved exactly (apart from normalization).
    Failure is explicit if the registered block cap is exhausted.
    """
    dirs = np.asarray(directions, dtype=float)
    if (
        dirs.ndim != 2
        or dirs.shape[0] < 1
        or dirs.shape[1] < 1
        or not np.all(np.isfinite(dirs))
    ):
        raise ValueError("directions must be a nonempty finite matrix")
    if not math.isfinite(maximum_condition_number) or maximum_condition_number < 1.0:
        raise ValueError("maximum_condition_number must be finite and at least one")
    if maximum_blocks < 1:
        raise ValueError("maximum_blocks must be positive")
    lengths = np.linalg.norm(dirs, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("directions must be nonzero")
    current = dirs / lengths[:, None]
    dimension = current.shape[1]
    blocks_used = math.ceil(len(current) / dimension)
    if blocks_used > maximum_blocks:
        raise ValueError("existing directions exceed maximum_blocks")
    while True:
        diagnostics = quadratic_frame_diagnostics(
            current, rank_tolerance=rank_tolerance
        )
        if (
            diagnostics.is_full_rank
            and diagnostics.condition_number <= maximum_condition_number
        ):
            result = current.copy()
            result.setflags(write=False)
            return result, diagnostics
        if blocks_used >= maximum_blocks:
            raise RuntimeError(
                "failed to construct a condition-certified quadratic frame"
            )
        current = np.vstack([current, _random_orthogonal_block(rng, dimension)])
        blocks_used += 1


def frame_diagnostics(
    directions: np.ndarray, *, rank_tolerance: float = 1e-12
) -> FrameDiagnostics:
    """Diagnose the normalized frame operator ``(d/L) sum u u^T``."""
    dirs = np.asarray(directions, dtype=float)
    if (
        dirs.ndim != 2
        or dirs.shape[0] < 1
        or dirs.shape[1] < 1
        or not np.all(np.isfinite(dirs))
    ):
        raise ValueError("directions must be a nonempty finite matrix")
    lengths = np.linalg.norm(dirs, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("directions must be nonzero")
    dirs = dirs / lengths[:, None]
    dimension = dirs.shape[1]
    operator = dimension / len(dirs) * dirs.T @ dirs
    operator = (operator + operator.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(operator)
    threshold = rank_tolerance * max(1.0, float(eigenvalues[-1]))
    rank = int(np.count_nonzero(eigenvalues > threshold))
    condition = (
        math.inf if rank < dimension else float(eigenvalues[-1] / eigenvalues[0])
    )
    defect = operator - np.eye(dimension)
    return FrameDiagnostics(
        operator=operator,
        eigenvalues=eigenvalues,
        rank=rank,
        condition_number=condition,
        spectral_tightness_error=float(np.linalg.norm(defect, ord=2)),
        frobenius_tightness_error=float(np.linalg.norm(defect, ord="fro")),
    )


@dataclass(frozen=True)
class TorchRNGSnapshot:
    cpu: Tensor
    cuda: tuple[Tensor, ...]


def capture_torch_rng() -> TorchRNGSnapshot:
    """Capture Torch CPU and all initialized CUDA RNG streams."""
    cuda = (
        tuple(state.clone() for state in torch.cuda.get_rng_state_all())
        if torch.cuda.is_available()
        else ()
    )
    return TorchRNGSnapshot(cpu=torch.random.get_rng_state().clone(), cuda=cuda)


def restore_torch_rng(snapshot: TorchRNGSnapshot) -> None:
    torch.random.set_rng_state(snapshot.cpu)
    if snapshot.cuda:
        torch.cuda.set_rng_state_all(list(snapshot.cuda))


def _slice_assignment(
    assignment: RankAssignment, start: int, stop: int
) -> RankAssignment:
    return RankAssignment(
        assigned_targets=assignment.assigned_targets[start:stop],
        order=assignment.order,
        ranks=assignment.ranks[start:stop],
        ordered_targets=assignment.ordered_targets,
    )


@dataclass(frozen=True)
class RSRBackwardResult:
    loss: float
    run_features: Tensor
    assignment: RankAssignment
    effective_population: int
    microbatch_count: int
    generator_example_evaluations: int


@dataclass(frozen=True)
class ProtectedTailGuardResult:
    selected_weight: float
    total_loss_before: float
    tail_loss_before: float
    candidate_weights: tuple[float, ...]
    candidate_total_losses: tuple[float, ...]
    candidate_tail_losses: tuple[float, ...]
    candidate_safe: tuple[bool, ...]


@dataclass(frozen=True)
class CrossFittedLocalControllerResult:
    selected_weight: float
    candidate_weights: tuple[float, ...]
    candidate_total_losses: tuple[float, ...]
    candidate_tail_losses: tuple[float, ...]
    candidate_safe: tuple[bool, ...]
    global_total_loss: float
    global_tail_loss: float
    best_safe_total_loss: float


def select_cross_fitted_local_weight(
    run_features: Tensor,
    controller_atlas: QuantileAtlas,
    global_correction: Tensor,
    normalized_local_field: Tensor,
    *,
    particle_step: float,
    candidate_weights: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25),
    tail_fraction: float = 0.10,
    tail_relative_tolerance: float = 0.02,
    near_optimal_relative_tolerance: float = 0.01,
    absolute_tolerance: float = 1e-12,
) -> CrossFittedLocalControllerResult:
    """Choose a local weight on directions disjoint from final evaluation.

    Every candidate is reranked on the controller atlas. Tail safety is
    measured against the global-only candidate, then the smallest near-optimal
    safe weight is selected. The caller is responsible for constructing the
    controller atlas independently of both transport and evaluation banks.
    """
    _validate_features(run_features, controller_atlas)
    if global_correction.shape != run_features.shape:
        raise ValueError("global correction has the wrong shape")
    if normalized_local_field.shape != run_features.shape:
        raise ValueError("normalized local field has the wrong shape")
    if not 0.0 <= particle_step <= 1.0 or not math.isfinite(particle_step):
        raise ValueError("particle_step must lie in [0, 1]")
    if not 0.0 < tail_fraction <= 0.5 or not math.isfinite(tail_fraction):
        raise ValueError("tail_fraction must lie in (0, 0.5]")
    if (
        not math.isfinite(tail_relative_tolerance)
        or tail_relative_tolerance < 0.0
        or not math.isfinite(near_optimal_relative_tolerance)
        or near_optimal_relative_tolerance < 0.0
        or not math.isfinite(absolute_tolerance)
        or absolute_tolerance < 0.0
    ):
        raise ValueError("controller tolerances must be finite and nonnegative")
    weights = tuple(float(weight) for weight in candidate_weights)
    if (
        not weights
        or 0.0 not in weights
        or any(not math.isfinite(weight) or weight < 0.0 for weight in weights)
        or tuple(sorted(set(weights))) != weights
    ):
        raise ValueError(
            "candidate_weights must be unique, finite, increasing, and contain zero"
        )
    directions = controller_atlas.torch_directions(run_features)
    tail_count = max(1, math.ceil(tail_fraction * len(run_features)))
    totals: list[float] = []
    tails: list[float] = []
    with torch.no_grad():
        for weight in weights:
            candidate = run_features + particle_step * (
                global_correction + weight * normalized_local_field
            )
            assignment = assign_rank_targets(candidate, controller_atlas)
            residual = candidate @ directions.T - assignment.assigned_targets
            squares = residual.square()
            tail_mask = (assignment.ranks < tail_count) | (
                assignment.ranks >= len(candidate) - tail_count
            )
            totals.append(float(torch.mean(squares).cpu()))
            tails.append(float(torch.mean(squares[tail_mask]).cpu()))
    zero_index = weights.index(0.0)
    global_total = totals[zero_index]
    global_tail = tails[zero_index]
    tail_limit = (
        global_tail * (1.0 + tail_relative_tolerance) + absolute_tolerance
    )
    safe = [tail <= tail_limit for tail in tails]
    safe[zero_index] = True
    best_safe = min(
        total for total, is_safe in zip(totals, safe, strict=True) if is_safe
    )
    near_optimal_limit = (
        best_safe * (1.0 + near_optimal_relative_tolerance) + absolute_tolerance
    )
    selected = min(
        weight
        for weight, total, is_safe in zip(weights, totals, safe, strict=True)
        if is_safe and total <= near_optimal_limit
    )
    return CrossFittedLocalControllerResult(
        selected_weight=selected,
        candidate_weights=weights,
        candidate_total_losses=tuple(totals),
        candidate_tail_losses=tuple(tails),
        candidate_safe=tuple(safe),
        global_total_loss=global_total,
        global_tail_loss=global_tail,
        best_safe_total_loss=best_safe,
    )


def select_protected_local_weight(
    run_features: Tensor,
    atlas: QuantileAtlas,
    global_correction: Tensor,
    normalized_local_field: Tensor,
    *,
    particle_step: float,
    candidate_weights: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25),
    tail_fraction: float = 0.10,
    relative_tolerance: float = 1e-8,
) -> ProtectedTailGuardResult:
    """Choose the largest local weight safe relative to global-only transport.

    Safety uses the fixed pre-step rank assignment, but its loss baseline is
    the candidate with local weight zero.  A local term therefore cannot spend
    improvement supplied by the global correction.  This is an empirical
    one-step guard, not a theorem of mode preservation or convergence.
    """
    _validate_features(run_features, atlas)
    if global_correction.shape != run_features.shape:
        raise ValueError("global correction has the wrong shape")
    if normalized_local_field.shape != run_features.shape:
        raise ValueError("normalized local field has the wrong shape")
    if not 0.0 <= particle_step <= 1.0 or not math.isfinite(particle_step):
        raise ValueError("particle_step must lie in [0, 1]")
    if not 0.0 < tail_fraction <= 0.5 or not math.isfinite(tail_fraction):
        raise ValueError("tail_fraction must lie in (0, 0.5]")
    if relative_tolerance < 0.0 or not math.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be finite and nonnegative")
    if not candidate_weights or 0.0 not in candidate_weights:
        raise ValueError("candidate_weights must be nonempty and contain zero")
    weights = tuple(float(weight) for weight in candidate_weights)
    if (
        any(not math.isfinite(weight) or weight < 0.0 for weight in weights)
        or tuple(sorted(set(weights))) != weights
    ):
        raise ValueError("candidate_weights must be unique, finite, and increasing")

    assignment = assign_rank_targets(run_features, atlas)
    directions = atlas.torch_directions(run_features)
    tail_count = max(1, math.ceil(tail_fraction * len(run_features)))
    tail_mask = (assignment.ranks < tail_count) | (
        assignment.ranks >= len(run_features) - tail_count
    )

    def losses(features: Tensor) -> tuple[float, float]:
        residual = features @ directions.T - assignment.assigned_targets
        squares = residual.square()
        total = float(torch.mean(squares).cpu())
        tail = float(torch.mean(squares[tail_mask]).cpu())
        return total, tail

    with torch.no_grad():
        total_before, tail_before = losses(run_features)
        totals: list[float] = []
        tails: list[float] = []
        safe: list[bool] = []
        for weight in weights:
            candidate = run_features + particle_step * (
                global_correction + weight * normalized_local_field
            )
            total, tail = losses(candidate)
            totals.append(total)
            tails.append(tail)
        zero_index = weights.index(0.0)
        total_limit = (
            totals[zero_index] * (1.0 + relative_tolerance) + relative_tolerance
        )
        tail_limit = tails[zero_index] * (1.0 + relative_tolerance) + relative_tolerance
        safe.extend(
            total <= total_limit and tail <= tail_limit
            for total, tail in zip(totals, tails, strict=True)
        )
    accepted = [
        weight for weight, is_safe in zip(weights, safe, strict=True) if is_safe
    ]
    selected = max(accepted) if accepted else 0.0
    return ProtectedTailGuardResult(
        selected_weight=selected,
        total_loss_before=total_before,
        tail_loss_before=tail_before,
        candidate_weights=weights,
        candidate_total_losses=tuple(totals),
        candidate_tail_losses=tuple(tails),
        candidate_safe=tuple(safe),
    )


@dataclass(frozen=True)
class TransportAmortizationResult:
    teacher_features: Tensor
    run_features: Tensor
    global_correction: Tensor
    local_field: Tensor | None
    guard: ProtectedTailGuardResult | None
    controller: CrossFittedLocalControllerResult | None
    selected_local_weight: float
    global_correction_rms: float
    local_field_rms: float
    local_scale: float
    teacher_displacement_rms: float
    global_rollout_displacement_rms: float
    local_positive_weight_fraction: float
    tail_balanced_count: int
    mean_student_loss: float
    effective_population: int
    student_updates: int
    generator_forward_calls: int
    generator_example_evaluations: int


@torch.no_grad()
def reranked_psqt_rollout(
    features: Tensor,
    atlas: QuantileAtlas,
    *,
    steps: int,
    step_size: float,
) -> tuple[Tensor, Tensor, float]:
    """Rerank and transport one free-particle population repeatedly.

    The returned correction is the total displacement from the initial
    population. No generator evaluation occurs inside this rollout.
    """
    _validate_features(features, atlas)
    if steps < 1:
        raise ValueError("rollout steps must be positive")
    if not 0.0 <= step_size <= 1.0 or not math.isfinite(step_size):
        raise ValueError("rollout step size must lie in [0, 1]")
    initial = features
    current = features
    for _ in range(steps):
        current = current + step_size * psqt_feature_correction(current, atlas)
    displacement = current - initial
    displacement_rms = float(
        torch.sqrt(torch.mean(torch.sum(displacement.square(), dim=1))).cpu()
    )
    return current, displacement, displacement_rms


@torch.no_grad()
def per_particle_safe_local_weights(
    features: Tensor,
    atlas: QuantileAtlas,
    normalized_local_field: Tensor,
    *,
    candidate_weights: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25),
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-12,
) -> Tensor:
    """Choose the largest local weight that preserves each rank residual.

    The rank assignment is frozen at ``features``. Safety is particle-local:
    one bulk improvement cannot pay for worsening a different tail particle.
    """
    _validate_features(features, atlas)
    if normalized_local_field.shape != features.shape:
        raise ValueError("normalized local field has the wrong shape")
    if not bool(torch.isfinite(normalized_local_field).all()):
        raise ValueError("normalized local field must be finite")
    if (
        relative_tolerance < 0.0
        or not math.isfinite(relative_tolerance)
        or absolute_tolerance < 0.0
        or not math.isfinite(absolute_tolerance)
    ):
        raise ValueError("local safety tolerances must be finite and nonnegative")
    weights = tuple(float(weight) for weight in candidate_weights)
    if (
        not weights
        or 0.0 not in weights
        or any(not math.isfinite(weight) or weight < 0.0 for weight in weights)
        or tuple(sorted(set(weights))) != weights
    ):
        raise ValueError(
            "candidate_weights must be unique, increasing, and contain zero"
        )
    assignment = assign_rank_targets(features, atlas)
    directions = atlas.torch_directions(features)
    baseline_residual = features @ directions.T - assignment.assigned_targets
    baseline_loss = torch.mean(baseline_residual.square(), dim=1)
    limit = baseline_loss * (1.0 + relative_tolerance) + absolute_tolerance
    selected = torch.zeros(
        len(features), dtype=features.dtype, device=features.device
    )
    for weight in weights:
        candidate = features + weight * normalized_local_field
        residual = candidate @ directions.T - assignment.assigned_targets
        safe = torch.mean(residual.square(), dim=1) <= limit
        selected = torch.where(
            safe,
            torch.full_like(selected, weight),
            selected,
        )
    return selected


def tail_balanced_order(
    displacement: Tensor,
    base_order: Tensor,
    *,
    microbatch: int,
    tail_fraction: float,
) -> tuple[Tensor, int]:
    """Spread the largest-displacement particles across student minibatches."""
    if displacement.ndim != 2 or not displacement.is_floating_point():
        raise ValueError("displacement must be a floating matrix")
    population = len(displacement)
    if (
        base_order.shape != (population,)
        or base_order.dtype != torch.long
        or base_order.device != displacement.device
        or not torch.equal(
            torch.sort(base_order).values,
            torch.arange(population, device=displacement.device),
        )
    ):
        raise ValueError("base order must be a permutation")
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    if not 0.0 < tail_fraction <= 0.5 or not math.isfinite(tail_fraction):
        raise ValueError("tail fraction must lie in (0, 0.5]")
    tail_count = max(1, math.ceil(tail_fraction * population))
    scores = torch.sum(displacement.square(), dim=1)
    tail_indices = torch.topk(scores, tail_count, largest=True, sorted=False).indices
    is_tail = torch.zeros(population, dtype=torch.bool, device=displacement.device)
    is_tail[tail_indices] = True
    ordered_tail = base_order[is_tail[base_order]]
    ordered_bulk = base_order[~is_tail[base_order]]
    batch_sizes = [
        min(microbatch, population - start)
        for start in range(0, population, microbatch)
    ]
    batch_count = len(batch_sizes)
    tail_quotas = [
        tail_count // batch_count + (batch < tail_count % batch_count)
        for batch in range(batch_count)
    ]
    # A short final batch can be smaller than its even tail quota. Move the
    # excess deterministically to earlier batches with spare capacity.
    excess = 0
    for batch, size in enumerate(batch_sizes):
        if tail_quotas[batch] > size:
            excess += tail_quotas[batch] - size
            tail_quotas[batch] = size
    for batch, size in enumerate(batch_sizes):
        if excess == 0:
            break
        spare = size - tail_quotas[batch]
        moved = min(spare, excess)
        tail_quotas[batch] += moved
        excess -= moved
    if excess:
        raise AssertionError("tail quotas exceed minibatch capacity")
    chunks: list[Tensor] = []
    tail_start = 0
    bulk_start = 0
    for size, tail_quota in zip(batch_sizes, tail_quotas, strict=True):
        bulk_quota = size - tail_quota
        chunks.append(
            torch.cat(
                (
                    ordered_tail[tail_start : tail_start + tail_quota],
                    ordered_bulk[bulk_start : bulk_start + bulk_quota],
                )
            )
        )
        tail_start += tail_quota
        bulk_start += bulk_quota
    result = torch.cat(chunks)
    if (
        tail_start != len(ordered_tail)
        or bulk_start != len(ordered_bulk)
        or not torch.equal(torch.sort(result).values, torch.sort(base_order).values)
    ):
        raise AssertionError("tail-balanced ordering lost a particle")
    return result, tail_count


def transport_then_amortize_step(
    forward_features: Callable[[Tensor], Tensor],
    optimizer: torch.optim.Optimizer,
    latents: Tensor,
    atlas: QuantileAtlas,
    *,
    microbatch: int,
    particle_step: float = 0.5,
    local_field_builder: Callable[[Tensor], Tensor] | None = None,
    local_weight: float = 0.25,
    local_scale_floor: float = 1e-12,
    local_scale_cap: float = 256.0,
    permutation: Tensor | None = None,
    protect_tails: bool = False,
    controller_atlas: QuantileAtlas | None = None,
    guard_candidate_weights: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25),
    guard_tail_fraction: float = 0.10,
    controller_tail_relative_tolerance: float = 0.02,
    controller_near_optimal_relative_tolerance: float = 0.01,
    global_rollout_steps: int = 1,
    global_rollout_step_size: float | None = None,
    rollout_local_after_global: bool = False,
    per_particle_local_safety: bool = False,
    tail_balanced_amortization: bool = False,
    tail_balance_fraction: float = 0.10,
) -> TransportAmortizationResult:
    """Compute one global particle teacher and amortize it in micro-updates.

    The Run population is ranked once.  Its backprojected PSQT correction is a
    coherent feature-space target.  The optimizer intentionally steps after
    every student microbatch, so only the first student gradient has the exact
    rank-loss identity; later steps distill the frozen teacher.
    """
    if latents.ndim < 1 or latents.shape[0] < 2:
        raise ValueError("latents must contain at least two examples")
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    if not 0.0 <= particle_step <= 1.0 or not math.isfinite(particle_step):
        raise ValueError("particle_step must lie in [0, 1]")
    if not math.isfinite(local_weight) or local_weight < 0.0:
        raise ValueError("local_weight must be finite and nonnegative")
    if not math.isfinite(local_scale_floor) or local_scale_floor <= 0.0:
        raise ValueError("local_scale_floor must be finite and positive")
    if not math.isfinite(local_scale_cap) or local_scale_cap <= 0.0:
        raise ValueError("local_scale_cap must be finite and positive")
    if protect_tails and controller_atlas is not None:
        raise ValueError("tail guard and cross-fitted controller are mutually exclusive")
    if controller_atlas is not None and controller_atlas.dimension != atlas.dimension:
        raise ValueError("controller atlas dimension differs from transport atlas")
    if global_rollout_steps < 1:
        raise ValueError("global rollout steps must be positive")
    rollout_step_size = (
        particle_step
        if global_rollout_step_size is None
        else float(global_rollout_step_size)
    )
    if not 0.0 <= rollout_step_size <= 1.0 or not math.isfinite(
        rollout_step_size
    ):
        raise ValueError("global rollout step size must lie in [0, 1]")
    if per_particle_local_safety and not rollout_local_after_global:
        raise ValueError("per-particle local safety requires post-rollout local field")
    if (protect_tails or controller_atlas is not None) and rollout_local_after_global:
        raise ValueError("legacy scalar guards do not support post-rollout local fields")
    if (
        not 0.0 < tail_balance_fraction <= 0.5
        or not math.isfinite(tail_balance_fraction)
    ):
        raise ValueError("tail balance fraction must lie in (0, 0.5]")
    population = int(latents.shape[0])
    ranges = [
        (start, min(start + microbatch, population))
        for start in range(0, population, microbatch)
    ]
    with torch.no_grad():
        run_features = torch.cat(
            [forward_features(latents[start:stop]) for start, stop in ranges], dim=0
        )
        _validate_features(run_features, atlas)
        if rollout_local_after_global:
            (
                rollout_features,
                global_correction,
                global_rollout_displacement_rms,
            ) = reranked_psqt_rollout(
                run_features,
                atlas,
                steps=global_rollout_steps,
                step_size=rollout_step_size,
            )
            global_rms_tensor = torch.tensor(
                global_rollout_displacement_rms,
                dtype=run_features.dtype,
                device=run_features.device,
            )
            global_rms = global_rollout_displacement_rms
            local_query = rollout_features
        else:
            global_correction = psqt_feature_correction(run_features, atlas).detach()
            global_rms_tensor = torch.sqrt(
                torch.mean(torch.sum(global_correction.square(), dim=1))
            )
            global_rms = float(global_rms_tensor.cpu())
            global_rollout_displacement_rms = particle_step * global_rms
            rollout_features = run_features + particle_step * global_correction
            local_query = run_features

        local_field = None
        local_rms = 0.0
        local_scale = 0.0
        normalized_local = torch.zeros_like(global_correction)
        if local_field_builder is not None and local_weight > 0.0:
            local_field = local_field_builder(local_query).detach()
            if local_field.shape != local_query.shape:
                raise ValueError("local field has the wrong shape")
            if not bool(torch.isfinite(local_field).all()):
                raise ValueError("local field must be finite")
            local_rms_tensor = torch.sqrt(
                torch.mean(torch.sum(local_field.square(), dim=1))
            )
            local_rms = float(local_rms_tensor.cpu())
            if local_rms > local_scale_floor and global_rms > 0.0:
                local_scale = min(global_rms / local_rms, local_scale_cap)
                normalized_local = local_scale * local_field

        guard = None
        controller = None
        selected_weight = local_weight if local_field is not None else 0.0
        local_positive_weight_fraction = 1.0 if selected_weight > 0.0 else 0.0
        particle_local_weights: Tensor | None = None
        if protect_tails and local_field is not None and selected_weight > 0.0:
            eligible = tuple(
                weight
                for weight in guard_candidate_weights
                if weight <= local_weight + 1e-15
            )
            if 0.0 not in eligible:
                eligible = (0.0, *eligible)
            guard = select_protected_local_weight(
                run_features,
                atlas,
                global_correction,
                normalized_local,
                particle_step=particle_step,
                candidate_weights=tuple(sorted(set(eligible))),
                tail_fraction=guard_tail_fraction,
            )
            selected_weight = guard.selected_weight
        elif (
            controller_atlas is not None
            and local_field is not None
            and selected_weight > 0.0
        ):
            eligible = tuple(
                weight
                for weight in guard_candidate_weights
                if weight <= local_weight + 1e-15
            )
            if 0.0 not in eligible:
                eligible = (0.0, *eligible)
            controller = select_cross_fitted_local_weight(
                run_features,
                controller_atlas,
                global_correction,
                normalized_local,
                particle_step=particle_step,
                candidate_weights=tuple(sorted(set(eligible))),
                tail_fraction=guard_tail_fraction,
                tail_relative_tolerance=controller_tail_relative_tolerance,
                near_optimal_relative_tolerance=(
                    controller_near_optimal_relative_tolerance
                ),
            )
            selected_weight = controller.selected_weight
        if (
            rollout_local_after_global
            and per_particle_local_safety
            and local_field is not None
            and selected_weight > 0.0
        ):
            eligible = tuple(
                weight
                for weight in guard_candidate_weights
                if weight <= local_weight + 1e-15
            )
            if 0.0 not in eligible:
                eligible = (0.0, *eligible)
            particle_local_weights = per_particle_safe_local_weights(
                rollout_features,
                atlas,
                normalized_local,
                candidate_weights=tuple(sorted(set(eligible))),
            )
            selected_weight = float(torch.mean(particle_local_weights).cpu())
            local_positive_weight_fraction = float(
                torch.mean((particle_local_weights > 0.0).to(run_features.dtype)).cpu()
            )
        elif particle_local_weights is None:
            local_positive_weight_fraction = 1.0 if selected_weight > 0.0 else 0.0

        if rollout_local_after_global:
            if particle_local_weights is not None:
                teacher = (
                    rollout_features
                    + particle_local_weights[:, None] * normalized_local
                ).detach()
            else:
                teacher = (
                    rollout_features + selected_weight * normalized_local
                ).detach()
        else:
            teacher = (
                run_features
                + particle_step
                * (global_correction + selected_weight * normalized_local)
            ).detach()
        displacement_rms = float(
            torch.sqrt(
                torch.mean(torch.sum((teacher - run_features).square(), dim=1))
            ).cpu()
        )

    if permutation is None:
        order = torch.arange(population, device=latents.device)
    else:
        order = permutation.to(device=latents.device, dtype=torch.long)
        if order.shape != (population,) or not torch.equal(
            torch.sort(order).values, torch.arange(population, device=latents.device)
        ):
            raise ValueError("permutation must contain every population index once")
    tail_balanced_count = 0
    if tail_balanced_amortization:
        order, tail_balanced_count = tail_balanced_order(
            teacher - run_features,
            order,
            microbatch=microbatch,
            tail_fraction=tail_balance_fraction,
        )

    losses: list[float] = []
    for start, stop in ranges:
        indices = order[start:stop]
        optimizer.zero_grad(set_to_none=True)
        prediction = forward_features(latents[indices])
        _validate_features(prediction, atlas)
        loss = 0.5 * torch.mean(
            torch.sum((prediction - teacher[indices]).square(), dim=1)
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    return TransportAmortizationResult(
        teacher_features=teacher,
        run_features=run_features.detach(),
        global_correction=global_correction,
        local_field=local_field,
        guard=guard,
        controller=controller,
        selected_local_weight=selected_weight,
        global_correction_rms=global_rms,
        local_field_rms=local_rms,
        local_scale=local_scale,
        teacher_displacement_rms=displacement_rms,
        global_rollout_displacement_rms=global_rollout_displacement_rms,
        local_positive_weight_fraction=local_positive_weight_fraction,
        tail_balanced_count=tail_balanced_count,
        mean_student_loss=float(np.mean(losses)),
        effective_population=population,
        student_updates=len(ranges),
        generator_forward_calls=2 * len(ranges),
        generator_example_evaluations=2 * population,
    )


def run_sort_rerun_backward(
    forward_features: Callable[[Tensor], Tensor],
    latents: Tensor,
    atlas: QuantileAtlas,
    *,
    microbatch: int,
    fixed_feature_field: Tensor | None = None,
    feature_field_builder: Callable[[Tensor], Tensor] | None = None,
    rank_loss_weight: float = 1.0,
    feature_field_weight: float = 1.0,
    replay_torch_rng: bool = True,
    strict_replay: bool = True,
    replay_atol: float = 0.0,
    replay_rtol: float = 0.0,
) -> RSRBackwardResult:
    """Backpropagate one exact global-rank loss in replayed microbatches.

    This function accumulates gradients but deliberately does not clear them or
    call an optimizer.  The caller must ``zero_grad`` before and ``step`` only
    after this function returns.

    Torch RNG is replayed per chunk by default, which supports deterministic
    dropout.  NumPy/Python RNG and stateful buffers are outside this mechanism.
    ``strict_replay`` catches changed outputs, including typical training-mode
    BatchNorm misuse.
    """
    if latents.ndim < 1 or latents.shape[0] < 2:
        raise ValueError("latents must contain at least two examples")
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    if not math.isfinite(rank_loss_weight) or rank_loss_weight < 0.0:
        raise ValueError("rank_loss_weight must be finite and nonnegative")
    if not math.isfinite(feature_field_weight):
        raise ValueError("feature_field_weight must be finite")
    population = int(latents.shape[0])
    if fixed_feature_field is not None and feature_field_builder is not None:
        raise ValueError("provide a fixed field or a field builder, not both")
    ranges = [
        (start, min(start + microbatch, population))
        for start in range(0, population, microbatch)
    ]

    rng_before_chunks: list[TorchRNGSnapshot] = []
    run_chunks: list[Tensor] = []
    with torch.no_grad():
        for start, stop in ranges:
            if replay_torch_rng:
                rng_before_chunks.append(capture_torch_rng())
            features = forward_features(latents[start:stop])
            _validate_features(features, atlas)
            run_chunks.append(features.detach().clone())
    rng_after_run = capture_torch_rng() if replay_torch_rng else None
    run_features = torch.cat(run_chunks, dim=0)
    assignment = assign_rank_targets(run_features, atlas)
    field = None
    if fixed_feature_field is not None:
        field = fixed_feature_field.detach()
    elif feature_field_builder is not None:
        field = feature_field_builder(run_features).detach()
    if field is not None:
        if field.shape != (population, atlas.dimension):
            raise ValueError("fixed feature field has the wrong shape")
        if not bool(torch.isfinite(field).all()):
            raise ValueError("fixed feature field must be finite")

    total_loss = 0.0
    try:
        for chunk_index, (start, stop) in enumerate(ranges):
            if replay_torch_rng:
                restore_torch_rng(rng_before_chunks[chunk_index])
            rerun = forward_features(latents[start:stop])
            _validate_features(rerun, atlas)
            if strict_replay and not torch.allclose(
                rerun.detach(),
                run_chunks[chunk_index],
                atol=replay_atol,
                rtol=replay_rtol,
            ):
                maximum = float(
                    torch.max(torch.abs(rerun.detach() - run_chunks[chunk_index])).cpu()
                )
                raise RuntimeError(
                    "RSR rerun changed outputs before optimizer step; "
                    f"maximum difference {maximum:.6g}"
                )
            chunk_assignment = _slice_assignment(assignment, start, stop)
            chunk_loss = rank_loss_weight * loss_from_rank_assignment(
                rerun, atlas, chunk_assignment, normalizer_population=population
            )
            if field is not None:
                chunk_loss = chunk_loss - (
                    feature_field_weight / population
                ) * torch.sum(rerun * field[start:stop])
            total_loss += float(chunk_loss.detach().cpu())
            chunk_loss.backward()
    finally:
        if rng_after_run is not None:
            restore_torch_rng(rng_after_run)

    return RSRBackwardResult(
        loss=total_loss,
        run_features=run_features,
        assignment=assignment,
        effective_population=population,
        microbatch_count=len(ranges),
        generator_example_evaluations=2 * population,
    )
