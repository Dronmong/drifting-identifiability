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


def assign_rank_targets(features: Tensor, atlas: QuantileAtlas) -> RankAssignment:
    """Assign atlas targets to generated features by global projected rank."""
    _validate_features(features, atlas)
    population = int(features.shape[0])
    if population < 2:
        raise ValueError("rank assignment requires at least two features")
    with torch.no_grad():
        directions = atlas.torch_directions(features)
        projected = features.detach() @ directions.T
        order = torch.argsort(projected, dim=0, stable=True)
        ordered = atlas.torch_ordered_quantiles(population, features)
        assigned = torch.empty_like(projected)
        assigned.scatter_(0, order, ordered)
        rank_values = torch.arange(population, device=features.device, dtype=torch.long)
        rank_values = rank_values[:, None].expand_as(order)
        ranks = torch.empty_like(order)
        ranks.scatter_(0, order, rank_values)
    return RankAssignment(
        assigned_targets=assigned.detach(),
        order=order.detach(),
        ranks=ranks.detach(),
        ordered_targets=ordered.detach(),
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
    assignment = assign_rank_targets(features, atlas)
    directions = atlas.torch_directions(features)
    projected = features @ directions.T
    return (
        (atlas.dimension / atlas.direction_count)
        * (assignment.assigned_targets - projected)
        @ directions
    )


@dataclass(frozen=True)
class FrameDiagnostics:
    operator: np.ndarray
    eigenvalues: np.ndarray
    rank: int
    condition_number: float
    spectral_tightness_error: float
    frobenius_tightness_error: float


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
