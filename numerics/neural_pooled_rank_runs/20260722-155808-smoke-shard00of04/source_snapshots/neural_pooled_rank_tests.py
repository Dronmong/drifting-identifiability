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
    apache_kll_target_atlas,
    assign_rank_targets,
    exact_target_atlas,
    frame_diagnostics,
    psqt_feature_correction,
    rank_matched_loss,
    run_sort_rerun_backward,
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
        kll_no_compaction_test,
        finite_difference_and_psqt_test,
        rsr_gradient_test,
        stochastic_replay_test,
        atoms_gaps_and_rare_modes_test,
        frame_diagnostic_test,
    )
    for test in tests:
        report.update(test())
        print(f"{test.__name__}: PASS")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
