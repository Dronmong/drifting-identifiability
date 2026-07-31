"""Phase-0 regression tests for neural pooled-rank PSQT.

Run from the repository root with the pinned Windows-compatible environment:

    uv run --python 3.12 --with torch==2.7.1 --with numpy \
      --with datasketches==5.2.0 \
      python numerics/neural_pooled_rank_tests.py

The tests are deterministic except for Apache KLL's internal compaction RNG.
The KLL equality test deliberately stays below compaction capacity; stress and
serialization behavior are separately audited by ``audit_projected_kll.py``.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.metadata
import json
import math

import numpy as np
import torch
from torch import nn

from neural_pooled_rank import (
    QuantileAtlas,
    apache_kll_target_atlas,
    assign_rank_targets,
    balanced_orthogonal_block_schedule,
    extend_to_conditioned_quadratic_frame,
    evenly_spaced_step_indices,
    exact_target_atlas,
    frame_diagnostics,
    normalized_paper_field,
    projection_tree_representatives,
    psqt_feature_correction,
    quadratic_frame_diagnostics,
    rank_matched_loss,
    run_sort_rerun_backward,
    select_protected_local_weight,
    transport_then_amortize_step,
    weighted_normalized_paper_field,
    weighted_paper_field_audit,
)
from projected_quantile_accumulators import reconstruct_from_quantile_table


SEED = 20260722
DTYPE = torch.float64


def _random_directions(
    rng: np.random.Generator, count: int, dimension: int
) -> np.ndarray:
    result = rng.normal(size=(count, dimension))
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def _maximum_parameter_gradient_difference(left: nn.Module, right: nn.Module) -> float:
    maximum = 0.0
    for left_parameter, right_parameter in zip(
        left.parameters(), right.parameters(), strict=True
    ):
        if left_parameter.grad is None or right_parameter.grad is None:
            raise AssertionError("expected every model parameter to have a grad")
        maximum = max(
            maximum,
            float(torch.max(torch.abs(left_parameter.grad - right_parameter.grad))),
        )
    return maximum


def exact_atlas_test() -> dict[str, float]:
    rng = np.random.default_rng(SEED + 1)
    target = rng.normal(size=(23, 3))
    directions = _random_directions(rng, 7, 3)
    atlas = exact_target_atlas(target, directions)
    expected = np.sort(target @ atlas.directions.T, axis=0, kind="stable")
    error = float(np.max(np.abs(atlas.quantiles.T - expected)))
    if error != 0.0:
        raise AssertionError(f"exact midpoint atlas error {error}")
    if atlas.ordered_quantiles(len(target)).shape != expected.shape:
        raise AssertionError("exact atlas population shape changed")
    if atlas.directions.flags.writeable or atlas.quantiles.flags.writeable:
        raise AssertionError("target atlas arrays are not immutable")
    return {"exact_midpoint_max_abs_error": error}


def direction_subset_and_schedule_test() -> dict[str, float | int]:
    rng = np.random.default_rng(SEED + 13)
    dimension, block_count = 3, 5
    blocks = []
    for _ in range(block_count):
        q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
        signs = np.where(np.diag(r) < 0.0, -1.0, 1.0)
        blocks.append((q * signs[None, :]).T)
    directions = np.vstack(blocks)
    grid = np.array([0.2, 0.5, 0.8])
    quantiles = np.sort(rng.normal(size=(len(directions), len(grid))), axis=1)
    payloads = tuple(f"direction-{index}".encode() for index in range(len(directions)))
    atlas = QuantileAtlas(
        directions=directions,
        grid=grid,
        quantiles=quantiles,
        source="schedule-test",
        target_count=101,
        normalized_rank_error=0.01,
        sketch_payloads=payloads,
    )
    selected_indices = np.array([0, 2, 4, 8, 11], dtype=np.int64)
    selected = atlas.select_directions(selected_indices)
    if (
        not np.array_equal(selected.directions, atlas.directions[selected_indices])
        or not np.array_equal(selected.quantiles, atlas.quantiles[selected_indices])
        or selected.sketch_payloads
        != tuple(payloads[int(index)] for index in selected_indices)
        or selected.target_count != atlas.target_count
        or selected.normalized_rank_error != atlas.normalized_rank_error
    ):
        raise AssertionError("direction-subset atlas lost registered state")
    if selected.directions.flags.writeable or selected.quantiles.flags.writeable:
        raise AssertionError("direction-subset atlas is mutable")

    steps = 11
    schedule = balanced_orthogonal_block_schedule(
        atlas.directions,
        2 * dimension,
        steps,
        np.random.default_rng(SEED + 14),
    )
    block_exposures = np.zeros(block_count, dtype=int)
    for indices in schedule:
        if len(indices) != 2 * dimension or len(np.unique(indices)) != len(indices):
            raise AssertionError("scheduled step has wrong or duplicate directions")
        scheduled_blocks = np.unique(indices // dimension)
        if len(scheduled_blocks) != 2:
            raise AssertionError("scheduled step split an orthogonal block")
        for block in scheduled_blocks:
            expected = np.arange(block * dimension, (block + 1) * dimension)
            if not np.all(np.isin(expected, indices)):
                raise AssertionError("scheduled step omitted part of a block")
        block_exposures[scheduled_blocks] += 1
    exposure_spread = int(block_exposures.max() - block_exposures.min())
    if exposure_spread > 1:
        raise AssertionError("orthogonal-block exposure is not balanced")

    full = balanced_orthogonal_block_schedule(
        atlas.directions,
        atlas.direction_count,
        3,
        np.random.default_rng(SEED + 15),
    )
    if any(
        not np.array_equal(indices, np.arange(atlas.direction_count))
        for indices in full
    ):
        raise AssertionError("full direction schedule changed registered ordering")
    local_twenty = evenly_spaced_step_indices(20, 20)
    local_ten = evenly_spaced_step_indices(20, 10)
    local_five = evenly_spaced_step_indices(20, 5)
    if (
        not np.array_equal(local_twenty, np.arange(20))
        or not np.array_equal(local_ten, np.arange(1, 20, 2))
        or not np.array_equal(local_five, np.arange(2, 20, 4))
    ):
        raise AssertionError("midpoint-stratified local schedule changed")
    return {
        "direction_subset_count": selected.direction_count,
        "scheduled_steps": len(schedule),
        "scheduled_block_exposure_spread": exposure_spread,
        "local_schedule_five_calls": len(local_five),
    }


def weighted_representative_field_test() -> dict[str, float]:
    """Certify weighted/self-masked semantics and expose compression error."""
    rng = np.random.default_rng(SEED + 17)
    support_size, dimension = 16, 4
    query = torch.tensor(
        rng.normal(size=(support_size, dimension)), dtype=DTYPE
    )
    positive = torch.tensor(
        rng.normal(size=(support_size, dimension)), dtype=DTYPE
    )
    directions = torch.tensor(
        _random_directions(rng, 7, dimension), dtype=DTYPE
    )

    full_positive = projection_tree_representatives(
        positive, directions, support_size
    )
    full_negative = projection_tree_representatives(query, directions, support_size)
    maximum_error = 0.0
    maximum_mass_error = 0.0
    for mask in (False, True):
        dense = normalized_paper_field(query, positive, tau=0.7, mask=mask)
        weighted = weighted_normalized_paper_field(
            query,
            full_positive,
            full_negative,
            tau=0.7,
            mask=mask,
        )
        maximum_error = max(
            maximum_error, float(torch.max(torch.abs(dense - weighted)))
        )
        audit = weighted_paper_field_audit(
            query,
            positive,
            full_positive,
            full_negative,
            tau=0.7,
            mask=mask,
            approximate_field=weighted,
        )
        maximum_mass_error = max(
            maximum_mass_error,
            audit["row_mass_max_relative_error"],
            audit["column_mass_max_relative_error"],
        )
    if maximum_error > 2e-13 or maximum_mass_error > 2e-13:
        raise AssertionError(
            "full weighted construction no longer matches dense: "
            f"field={maximum_error}, mass={maximum_mass_error}"
        )
    if (
        int(full_positive.multiplicities.sum()) != support_size
        or int(full_negative.multiplicities.sum()) != support_size
        or full_positive.max_radius > 1e-14
        or full_negative.max_radius > 1e-14
    ):
        raise AssertionError("full representative tree is not a singleton partition")

    compact_positive = projection_tree_representatives(positive, directions, 4)
    compact_negative = projection_tree_representatives(query, directions, 4)
    dense = normalized_paper_field(query, positive, tau=0.7, mask=True)
    compact = weighted_normalized_paper_field(
        query,
        compact_positive,
        compact_negative,
        tau=0.7,
        mask=True,
    )
    compact_audit = weighted_paper_field_audit(
        query,
        positive,
        compact_positive,
        compact_negative,
        tau=0.7,
        mask=True,
        approximate_field=compact,
    )
    relative_l2 = compact_audit["field_relative_l2_error"]
    cosine = compact_audit["field_cosine"]
    if not math.isfinite(relative_l2) or not -1.0000001 <= cosine <= 1.0000001:
        raise AssertionError("compressed representative field became non-finite")
    return {
        "weighted_full_dense_max_abs_error": maximum_error,
        "weighted_full_dense_max_relative_mass_error": maximum_mass_error,
        "weighted_compact_relative_l2_error": relative_l2,
        "weighted_compact_cosine": cosine,
        "weighted_compact_row_mass_relative_l2_error": compact_audit[
            "row_mass_relative_l2_error"
        ],
        "weighted_compact_column_mass_relative_l2_error": compact_audit[
            "column_mass_relative_l2_error"
        ],
        "weighted_compact_positive_rms_radius": compact_positive.rms_radius,
        "weighted_compact_negative_rms_radius": compact_negative.rms_radius,
    }


def kll_no_compaction_test() -> dict[str, float | int]:
    values = (np.arange(-15, 16, dtype=np.float32) / 4.0)[:, None]
    directions = np.asarray([[1.0], [-1.0]])
    exact = exact_target_atlas(values, directions)
    kll = apache_kll_target_atlas(
        (values[:9], values[9:21], values[21:]),
        directions,
        knot_count=len(values),
        k=128,
    )
    error = float(np.max(np.abs(kll.quantiles - exact.quantiles)))
    if error != 0.0:
        raise AssertionError(f"no-compaction KLL is not exact: {error}")
    if kll.target_count != len(values) or kll.sketch_payloads is None:
        raise AssertionError("KLL atlas metadata is incomplete")
    if len(kll.sketch_payloads) != len(directions):
        raise AssertionError("KLL atlas omitted a projection sketch")
    if kll.normalized_rank_error is None:
        raise AssertionError("KLL atlas omitted normalized rank error")
    return {
        "kll_no_compaction_max_abs_error": error,
        "kll_serialized_bytes": sum(map(len, kll.sketch_payloads)),
        "kll_normalized_rank_error": kll.normalized_rank_error,
    }


def finite_difference_and_psqt_test() -> dict[str, float]:
    rng = np.random.default_rng(SEED + 2)
    population, dimension, count = 7, 3, 11
    initial = rng.normal(size=(population, dimension))
    directions = _random_directions(rng, count, dimension)
    target = rng.normal(size=(population, dimension))
    atlas = exact_target_atlas(target, directions)

    features = torch.tensor(initial, dtype=DTYPE, requires_grad=True)
    result = rank_matched_loss(features, atlas)
    result.loss.backward()
    analytic = features.grad.detach().clone()

    epsilon = 1e-6
    finite = np.empty_like(initial)
    for row in range(population):
        for column in range(dimension):
            plus = initial.copy()
            minus = initial.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            plus_loss = rank_matched_loss(torch.tensor(plus, dtype=DTYPE), atlas).loss
            minus_loss = rank_matched_loss(torch.tensor(minus, dtype=DTYPE), atlas).loss
            finite[row, column] = float((plus_loss - minus_loss) / (2.0 * epsilon))
    finite_error = float(np.max(np.abs(analytic.numpy() - finite)))
    if finite_error > 2e-8:
        raise AssertionError(f"finite-difference error {finite_error}")

    detached = torch.tensor(initial, dtype=DTYPE)
    correction = psqt_feature_correction(detached, atlas)
    scaling_error = float(torch.max(torch.abs(correction + population * analytic)))
    if scaling_error > 2e-12:
        raise AssertionError(f"PSQT gradient scaling error {scaling_error}")

    reconstructed, _, _ = reconstruct_from_quantile_table(
        initial, directions, atlas.quantiles, steps=1, step_size=1.0
    )
    implementation_error = float(
        np.max(np.abs((reconstructed - initial) - correction.numpy()))
    )
    if implementation_error > 2e-12:
        raise AssertionError(
            f"existing PSQT correction mismatch {implementation_error}"
        )
    return {
        "finite_difference_max_abs_error": finite_error,
        "psqt_negative_B_gradient_max_abs_error": scaling_error,
        "existing_psqt_step_max_abs_error": implementation_error,
    }


def exact_cost_refactor_test() -> dict[str, float]:
    """Check the projection-reuse and normalized-kernel exact refactors."""
    rng = np.random.default_rng(SEED + 12)
    population, dimension, count = 17, 5, 19
    atlas = exact_target_atlas(
        rng.normal(size=(79, dimension)),
        _random_directions(rng, count, dimension),
        knot_count=43,
    )
    features = torch.tensor(
        rng.normal(size=(population, dimension)), dtype=DTYPE, requires_grad=True
    )
    assignment = assign_rank_targets(features, atlas)
    directions = atlas.torch_directions(features)
    legacy_correction = (
        (dimension / count)
        * (assignment.assigned_targets - features @ directions.T)
        @ directions
    )
    reused_correction = psqt_feature_correction(features, atlas)
    correction_error = float(
        torch.max(torch.abs(reused_correction - legacy_correction)).detach()
    )
    if correction_error != 0.0:
        raise AssertionError(f"projection reuse changed correction: {correction_error}")
    legacy_gradient = torch.autograd.grad(
        legacy_correction.square().sum(), features, retain_graph=True
    )[0]
    reused_gradient = torch.autograd.grad(
        reused_correction.square().sum(), features
    )[0]
    gradient_error = float(torch.max(torch.abs(reused_gradient - legacy_gradient)))
    if gradient_error != 0.0:
        raise AssertionError(f"projection reuse changed gradient: {gradient_error}")

    def legacy_field(
        query: torch.Tensor,
        positive: torch.Tensor,
        *,
        tau: float,
        mask: bool,
    ) -> torch.Tensor:
        positive_distance = torch.linalg.vector_norm(
            query[:, None, :] - positive[None, :, :], dim=2
        )
        negative_distance = torch.linalg.vector_norm(
            query[:, None, :] - query[None, :, :], dim=2
        )
        logits = torch.cat(
            [-positive_distance / tau, -negative_distance / tau], dim=1
        )
        if mask:
            rows = torch.arange(len(query), device=query.device)
            logits[rows, len(positive) + rows] -= 1e6 / tau
        affinity = torch.sqrt(
            torch.softmax(logits, dim=1) * torch.softmax(logits, dim=0)
        )
        positive_affinity = affinity[:, : len(positive)]
        negative_affinity = affinity[:, len(positive) :]
        return (
            negative_affinity.sum(dim=1, keepdim=True)
            * (positive_affinity @ positive)
            - positive_affinity.sum(dim=1, keepdim=True)
            * (negative_affinity @ query)
        )

    worst_float64 = 0.0
    worst_float32 = 0.0
    for dtype in (torch.float64, torch.float32):
        for mask in (False, True):
            query = torch.tensor(rng.normal(size=(23, 7)), dtype=dtype)
            positive = torch.tensor(rng.normal(size=(29, 7)), dtype=dtype)
            expected = legacy_field(query, positive, tau=0.7, mask=mask)
            observed = normalized_paper_field(
                query, positive, tau=0.7, mask=mask
            )
            error = float(torch.max(torch.abs(observed - expected)))
            if dtype == torch.float64:
                worst_float64 = max(worst_float64, error)
            else:
                worst_float32 = max(worst_float32, error)
    if worst_float64 > 2e-14 or worst_float32 > 2e-6:
        raise AssertionError(
            "normalized-kernel refactor mismatch: "
            f"float64={worst_float64}, float32={worst_float32}"
        )

    # Every raw-kernel entry underflows here.  The fallback must retain the
    # stable softmax result rather than returning NaNs from zero masses.
    extreme_query = torch.tensor([[1e6]], dtype=DTYPE)
    extreme_positive = torch.tensor([[-1e6]], dtype=DTYPE)
    extreme_expected = legacy_field(
        extreme_query, extreme_positive, tau=0.5, mask=True
    )
    extreme_observed = normalized_paper_field(
        extreme_query, extreme_positive, tau=0.5, mask=True
    )
    extreme_error = float(torch.max(torch.abs(extreme_observed - extreme_expected)))
    if extreme_error > 2e-14 or not bool(torch.isfinite(extreme_observed).all()):
        raise AssertionError(f"paper-field underflow fallback failed: {extreme_error}")
    return {
        "projection_reuse_correction_max_abs_error": correction_error,
        "projection_reuse_gradient_max_abs_error": gradient_error,
        "paper_field_factorization_float64_max_abs_error": worst_float64,
        "paper_field_factorization_float32_max_abs_error": worst_float32,
        "paper_field_underflow_fallback_max_abs_error": extreme_error,
    }


class TinyGenerator(nn.Module):
    def __init__(self, latent_dimension: int, feature_dimension: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dimension, 13),
            nn.Tanh(),
            nn.Linear(13, feature_dimension),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.layers(latent)


def rsr_gradient_test() -> dict[str, float]:
    rng = np.random.default_rng(SEED + 3)
    torch.manual_seed(SEED + 3)
    population, latent_dimension, dimension = 19, 5, 3
    latents = torch.tensor(rng.normal(size=(population, latent_dimension)), dtype=DTYPE)
    target = rng.normal(size=(67, dimension))
    directions = _random_directions(rng, 9, dimension)
    atlas = exact_target_atlas(target, directions, knot_count=31)

    base = TinyGenerator(latent_dimension, dimension).to(dtype=DTYPE)
    full = deepcopy(base)
    chunked = deepcopy(base)
    one_chunk = deepcopy(base)
    hybrid_full = deepcopy(base)
    hybrid_chunked = deepcopy(base)

    full.zero_grad(set_to_none=True)
    full_features = full(latents)
    full_loss = rank_matched_loss(full_features, atlas).loss
    full_loss.backward()

    chunked.zero_grad(set_to_none=True)
    chunked_result = run_sort_rerun_backward(chunked, latents, atlas, microbatch=4)
    one_chunk.zero_grad(set_to_none=True)
    one_chunk_result = run_sort_rerun_backward(
        one_chunk, latents, atlas, microbatch=population
    )

    fixed_field = torch.tensor(rng.normal(size=(population, dimension)), dtype=DTYPE)
    hybrid_full.zero_grad(set_to_none=True)
    hybrid_features = hybrid_full(latents)
    hybrid_loss = rank_matched_loss(hybrid_features, atlas).loss - (
        0.37 / population
    ) * torch.sum(hybrid_features * fixed_field)
    hybrid_loss.backward()
    hybrid_chunked.zero_grad(set_to_none=True)
    hybrid_result = run_sort_rerun_backward(
        hybrid_chunked,
        latents,
        atlas,
        microbatch=4,
        fixed_feature_field=fixed_field,
        feature_field_weight=0.37,
    )

    loss_error = abs(float(full_loss.detach()) - chunked_result.loss)
    gradient_error = _maximum_parameter_gradient_difference(full, chunked)
    partition_error = _maximum_parameter_gradient_difference(chunked, one_chunk)
    feature_error = float(
        torch.max(torch.abs(full_features.detach() - chunked_result.run_features))
    )
    one_chunk_loss_error = abs(one_chunk_result.loss - chunked_result.loss)
    hybrid_loss_error = abs(float(hybrid_loss.detach()) - hybrid_result.loss)
    hybrid_gradient_error = _maximum_parameter_gradient_difference(
        hybrid_full, hybrid_chunked
    )
    if loss_error > 2e-13 or one_chunk_loss_error > 2e-13:
        raise AssertionError("RSR loss changed under microbatching")
    if gradient_error > 2e-13 or partition_error > 2e-13:
        raise AssertionError("RSR gradient changed under microbatching")
    if feature_error > 2e-13:
        raise AssertionError("RSR Run features changed under microbatching")
    if hybrid_loss_error > 2e-13 or hybrid_gradient_error > 2e-13:
        raise AssertionError("fixed-field hybrid changed under microbatching")
    return {
        "rsr_full_loss_abs_error": loss_error,
        "rsr_full_gradient_max_abs_error": gradient_error,
        "rsr_partition_gradient_max_abs_error": partition_error,
        "rsr_run_feature_max_abs_error": feature_error,
        "rsr_hybrid_loss_abs_error": hybrid_loss_error,
        "rsr_hybrid_gradient_max_abs_error": hybrid_gradient_error,
    }


class DropoutGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(4, 17)
        self.dropout = nn.Dropout(p=0.35)
        self.output = nn.Linear(17, 2)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.dropout(self.input(latent))))


def stochastic_replay_test() -> dict[str, float]:
    rng = np.random.default_rng(SEED + 4)
    target = rng.normal(size=(47, 2))
    directions = _random_directions(rng, 8, 2)
    atlas = exact_target_atlas(target, directions, knot_count=23)
    latents = torch.tensor(rng.normal(size=(18, 4)), dtype=DTYPE)
    model = DropoutGenerator().to(dtype=DTYPE)
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(SEED + 4)
    result = run_sort_rerun_backward(
        model, latents, atlas, microbatch=5, replay_torch_rng=True, strict_replay=True
    )
    maximum_gradient = max(
        float(torch.max(torch.abs(parameter.grad)))
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not math.isfinite(result.loss) or not math.isfinite(maximum_gradient):
        raise AssertionError("stochastic RSR produced a non-finite result")
    if maximum_gradient == 0.0:
        raise AssertionError("stochastic RSR unexpectedly produced zero grad")
    return {
        "dropout_rsr_loss": result.loss,
        "dropout_rsr_max_abs_gradient": maximum_gradient,
    }


def atoms_gaps_and_rare_modes_test() -> dict[str, float | int]:
    directions = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
        ]
    )
    maximum_loss = 0.0
    cases = 0
    for minority in (1, 5):
        target = np.zeros((100, 2), dtype=float)
        target[-minority:, 0] = 8.0
        target[-minority:, 1] = -5.0
        exact = exact_target_atlas(target, directions, knot_count=64)
        kll = apache_kll_target_atlas((target[:37], target[37:]), directions, 64, k=128)
        features = torch.zeros((32, 2), dtype=DTYPE, requires_grad=True)
        first = assign_rank_targets(features, exact)
        second = assign_rank_targets(features, exact)
        if not torch.equal(first.assigned_targets, second.assigned_targets):
            raise AssertionError("stable tie assignment is not deterministic")
        for atlas in (exact, kll):
            candidate = features.detach().clone().requires_grad_(True)
            loss = rank_matched_loss(candidate, atlas).loss
            loss.backward()
            if not bool(torch.isfinite(loss)) or not bool(
                torch.isfinite(candidate.grad).all()
            ):
                raise AssertionError("atom/gap/rare case became non-finite")
            maximum_loss = max(maximum_loss, float(loss.detach()))
            cases += 1
        projected = np.asarray(target @ kll.directions.T, dtype=np.float32)
        for column in range(kll.direction_count):
            observed = set(projected[:, column].tolist())
            returned = set(np.asarray(kll.quantiles[column], dtype=np.float32).tolist())
            if not returned.issubset(observed):
                raise AssertionError(
                    "KLL returned a value outside projected float32 support"
                )
    return {
        "edge_case_count": cases,
        "edge_case_max_loss": maximum_loss,
    }


def frame_diagnostic_test() -> dict[str, float | int]:
    coordinate = np.eye(3)
    tight = frame_diagnostics(coordinate)
    if tight.rank != 3 or abs(tight.condition_number - 1.0) > 1e-14:
        raise AssertionError("coordinate tight frame diagnostic failed")
    if tight.spectral_tightness_error > 1e-14:
        raise AssertionError("coordinate frame was not recognized as tight")
    deficient = frame_diagnostics(
        np.asarray(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]
        )
    )
    if deficient.rank != 1 or not math.isinf(deficient.condition_number):
        raise AssertionError("rank-deficient frame was not detected")
    return {
        "coordinate_frame_rank": tight.rank,
        "coordinate_frame_condition": tight.condition_number,
        "deficient_frame_rank": deficient.rank,
    }


def quadratic_frame_diagnostic_test() -> dict[str, float | int]:
    """Catch frames that see vectors but not all covariance directions."""
    rng = np.random.default_rng(SEED + 8)
    dimension = 16
    blocks = []
    for _ in range(4):
        matrix = rng.normal(size=(dimension, dimension))
        q, _ = np.linalg.qr(matrix)
        blocks.append(q.T)
    vector_tight = np.vstack(blocks)
    vector = frame_diagnostics(vector_tight)
    quadratic = quadratic_frame_diagnostics(vector_tight)
    expected_rank = 4 * (dimension - 1) + 1
    if vector.rank != dimension or vector.spectral_tightness_error > 1e-12:
        raise AssertionError("orthogonal blocks should form a tight vector frame")
    if quadratic.rank != expected_rank or quadratic.is_full_rank:
        raise AssertionError("four-basis covariance nullspace was not detected")

    extended, certified = extend_to_conditioned_quadratic_frame(
        vector_tight,
        np.random.default_rng(SEED + 9),
        maximum_condition_number=25.0,
    )
    if not np.allclose(extended[: len(vector_tight)], vector_tight, atol=2e-15):
        raise AssertionError("quadratic extension changed registered directions")
    if not certified.is_full_rank or certified.condition_number > 25.0:
        raise AssertionError("quadratic extension did not satisfy its certificate")
    if len(extended) <= len(vector_tight):
        raise AssertionError("deficient frame was not extended")
    return {
        "d16_four_basis_quadratic_rank": quadratic.rank,
        "d16_quadratic_parameter_dimension": quadratic.parameter_dimension,
        "conditioned_direction_count": len(extended),
        "conditioned_quadratic_condition": certified.condition_number,
    }


def transport_gradient_identity_test() -> dict[str, float]:
    """Verify the exact first-step transport/amortization chain rule."""
    rng = np.random.default_rng(SEED + 10)
    torch.manual_seed(SEED + 10)
    population, latent_dimension, dimension = 29, 5, 3
    eta = 0.37
    latents = torch.tensor(rng.normal(size=(population, latent_dimension)), dtype=DTYPE)
    atlas = exact_target_atlas(
        rng.normal(size=(83, dimension)),
        _random_directions(rng, 13, dimension),
        knot_count=47,
    )
    base = TinyGenerator(latent_dimension, dimension).to(dtype=DTYPE)
    rank_model = deepcopy(base)
    student_model = deepcopy(base)

    rank_model.zero_grad(set_to_none=True)
    rank_features = rank_model(latents)
    rank_matched_loss(rank_features, atlas).loss.backward()

    with torch.no_grad():
        run_features = student_model(latents)
        teacher = run_features + eta * psqt_feature_correction(run_features, atlas)
    student_model.zero_grad(set_to_none=True)
    prediction = student_model(latents)
    student_loss = 0.5 * torch.mean(torch.sum((prediction - teacher).square(), dim=1))
    student_loss.backward()

    maximum = 0.0
    for rank_parameter, student_parameter in zip(
        rank_model.parameters(), student_model.parameters(), strict=True
    ):
        if rank_parameter.grad is None or student_parameter.grad is None:
            raise AssertionError("expected transport identity gradients")
        maximum = max(
            maximum,
            float(
                torch.max(torch.abs(student_parameter.grad - eta * rank_parameter.grad))
            ),
        )
    if maximum > 3e-13:
        raise AssertionError(f"transport gradient identity error {maximum}")
    return {"transport_first_gradient_max_abs_error": maximum}


def transport_step_and_guard_test() -> dict[str, float | int]:
    rng = np.random.default_rng(SEED + 11)
    torch.manual_seed(SEED + 11)
    population, latent_dimension, dimension = 24, 4, 2
    latents = torch.tensor(rng.normal(size=(population, latent_dimension)), dtype=DTYPE)
    atlas = exact_target_atlas(
        rng.normal(size=(71, dimension)),
        _random_directions(rng, 8, dimension),
        knot_count=41,
    )
    base = TinyGenerator(latent_dimension, dimension).to(dtype=DTYPE)
    first = deepcopy(base)
    second = deepcopy(base)
    permutation = torch.tensor(rng.permutation(population), dtype=torch.long)
    first_result = transport_then_amortize_step(
        first,
        torch.optim.Adam(first.parameters(), lr=1e-3),
        latents,
        atlas,
        microbatch=5,
        particle_step=0.4,
        permutation=permutation,
    )
    transport_then_amortize_step(
        second,
        torch.optim.Adam(second.parameters(), lr=1e-3),
        latents,
        atlas,
        microbatch=5,
        particle_step=0.4,
        permutation=permutation,
    )
    parameter_error = max(
        float(torch.max(torch.abs(left - right)))
        for left, right in zip(first.parameters(), second.parameters(), strict=True)
    )
    if parameter_error != 0.0:
        raise AssertionError("transport micro-updates are not deterministic")
    expected_updates = math.ceil(population / 5)
    if (
        first_result.student_updates != expected_updates
        or first_result.generator_forward_calls != 2 * expected_updates
        or first_result.generator_example_evaluations != 2 * population
    ):
        raise AssertionError("transport budget accounting is incorrect")

    zero = deepcopy(base)
    before = [parameter.detach().clone() for parameter in zero.parameters()]
    zero_result = transport_then_amortize_step(
        zero,
        torch.optim.Adam(zero.parameters(), lr=1e-3),
        latents,
        atlas,
        microbatch=6,
        particle_step=0.0,
    )
    zero_error = max(
        float(torch.max(torch.abs(old - new)))
        for old, new in zip(before, zero.parameters(), strict=True)
    )
    if zero_error != 0.0 or zero_result.teacher_displacement_rms != 0.0:
        raise AssertionError("zero particle step changed the model")

    global_only = deepcopy(base)
    zero_local = deepcopy(base)
    global_result = transport_then_amortize_step(
        global_only,
        torch.optim.Adam(global_only.parameters(), lr=1e-3),
        latents,
        atlas,
        microbatch=6,
        particle_step=0.4,
    )
    zero_local_result = transport_then_amortize_step(
        zero_local,
        torch.optim.Adam(zero_local.parameters(), lr=1e-3),
        latents,
        atlas,
        microbatch=6,
        particle_step=0.4,
        local_field_builder=lambda features: torch.zeros_like(features),
        local_weight=0.25,
    )
    zero_local_error = float(
        torch.max(
            torch.abs(
                global_result.teacher_features - zero_local_result.teacher_features
            )
        )
    )
    if zero_local_error != 0.0 or zero_local_result.local_scale != 0.0:
        raise AssertionError("zero local field changed the transport teacher")

    kll = apache_kll_target_atlas(
        (np.asarray(atlas.quantiles[:, :].T @ atlas.directions),),
        atlas.directions,
        atlas.knot_count,
        k=128,
    )
    # KLL execution is tested for finiteness here; exact no-compaction equality
    # is separately certified by kll_no_compaction_test.
    kll_model = deepcopy(base)
    kll_result = transport_then_amortize_step(
        kll_model,
        torch.optim.Adam(kll_model.parameters(), lr=1e-3),
        latents,
        kll,
        microbatch=6,
        particle_step=0.4,
    )
    if not math.isfinite(kll_result.mean_student_loss):
        raise AssertionError("KLL transport step became non-finite")

    features = base(latents).detach()
    correction = psqt_feature_correction(features, atlas)
    guard = select_protected_local_weight(
        features,
        atlas,
        correction,
        -40.0 * correction,
        particle_step=0.5,
    )
    if guard.selected_weight != 0.0 or not guard.candidate_safe[0]:
        raise AssertionError("tail guard did not reject an adversarial local field")
    return {
        "transport_determinism_max_abs_error": parameter_error,
        "transport_student_updates": first_result.student_updates,
        "transport_zero_step_max_abs_error": zero_error,
        "transport_zero_local_max_abs_error": zero_local_error,
        "transport_kll_student_loss": kll_result.mean_student_loss,
        "adversarial_guard_selected_weight": guard.selected_weight,
    }


def main() -> None:
    torch.use_deterministic_algorithms(True)
    report: dict[str, object] = {
        "seed": SEED,
        "python_datasketches": importlib.metadata.version("datasketches"),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    tests = (
        exact_atlas_test,
        direction_subset_and_schedule_test,
        weighted_representative_field_test,
        kll_no_compaction_test,
        finite_difference_and_psqt_test,
        exact_cost_refactor_test,
        rsr_gradient_test,
        stochastic_replay_test,
        atoms_gaps_and_rare_modes_test,
        frame_diagnostic_test,
        quadratic_frame_diagnostic_test,
        transport_gradient_identity_test,
        transport_step_and_guard_test,
    )
    for test in tests:
        report.update(test())
        print(f"{test.__name__}: PASS")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
