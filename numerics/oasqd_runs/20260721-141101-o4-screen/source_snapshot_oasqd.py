"""Mechanisms for Occupancy-Adaptive Stratified Quantile Drifting.

This module is development infrastructure.  It does not alter either frozen
LB-QCD registry or runner.  Its main invariants are:

* target regions are inferred from samples, never mixture metadata;
* a global rank table is formed before any backward subsampling;
* every nonempty rank stratum has positive inclusion probability; and
* the weighted backward estimator is conditionally unbiased for the full
  virtual-table stop-gradient objective (before the nonlinear Adam update).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from lbqcd import (
    StepWork,
    _adam_step,
    _stopgrad_grads,
    _sum_grads,
    exact_rank_field,
)
from run_identifiability_generator import TanhMLP


@dataclass(frozen=True)
class AtlasConfig:
    """Sampling-only rules for a persistent one-dimensional quantile atlas."""

    minimum_segment_count: int = 16
    split_minimum_segment_count: int = 6
    # Eight neighbors is small enough to assess both sides of a 0.5% interior
    # component in a split half-sample.  A window of 32 accidentally crossed
    # both gaps around such a component and hid it from the cohesion test.
    local_window: int = 8
    gap_multiplier: float = 18.0
    minimum_relative_gap: float = 0.012
    cohesion_multiplier: float = 1.35
    bootstrap_reps: int = 32
    minimum_persistence: float = 0.55
    match_gap_fraction: float = 0.40
    match_iqr_fraction: float = 0.025
    confidence_z: float = 1.96


@dataclass(frozen=True)
class BoundaryEvidence:
    location: float
    gap: float
    gap_ratio: float
    left_count: int
    right_count: int
    split_persistent: bool
    bootstrap_persistence: float


@dataclass(frozen=True)
class QuantileAtlas:
    """An empirical target distribution and its persistent separated regions."""

    ordered: np.ndarray
    boundaries: tuple[float, ...]
    masses: tuple[float, ...]
    mass_lower: tuple[float, ...]
    mass_upper: tuple[float, ...]
    evidence: tuple[BoundaryEvidence, ...]
    raw_candidate_count: int
    unresolved: bool

    @property
    def region_count(self) -> int:
        return len(self.masses)

    @property
    def sample_count(self) -> int:
        return len(self.ordered)


@dataclass(frozen=True)
class OccupancyAssessment:
    target_mass: tuple[float, ...]
    generated_mass: tuple[float, ...]
    generated_counts: tuple[int, ...]
    deficit_scores: tuple[float, ...]
    active_regions: tuple[int, ...]

    @property
    def has_deficit(self) -> bool:
        return bool(self.active_regions)


@dataclass(frozen=True)
class VirtualBatchChoice:
    size: int
    active_mass: float
    conservative_mass: float
    expected_count: float
    conservative_expected_count: float
    cap_hit: bool


@dataclass(frozen=True)
class StratifiedSelection:
    indices: np.ndarray
    weights: np.ndarray
    allocation: tuple[int, ...]
    counts: tuple[int, ...]


@dataclass(frozen=True)
class OAStepWork:
    optimizer_updates: int = 1
    generator_forward_calls: int = 0
    generator_example_evals: int = 0
    unique_latent_samples: int = 0
    target_samples: int = 0
    kernel_pairs: int = 0
    sort_work: float = 0.0
    backward_example_evals: int = 0


def _wilson_interval(count: int, total: int, z: float) \
        -> tuple[float, float]:
    if total <= 0 or not 0 <= count <= total or z <= 0:
        raise ValueError("invalid binomial interval inputs")
    p = count / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    radius = (z * math.sqrt(
        p * (1.0 - p) / total + z2 / (4.0 * total * total)) /
        denominator)
    return max(0.0, center - radius), min(1.0, center + radius)


def _candidate_boundaries(values: np.ndarray, config: AtlasConfig,
                          minimum_count: int) \
        -> list[tuple[float, float, float, int]]:
    """Return location, gap, global ratio, and left index for plausible gaps."""
    ordered = np.sort(np.asarray(values, dtype=float).reshape(-1))
    n = len(ordered)
    if n < 2 * minimum_count + 2:
        return []
    gaps = np.diff(ordered)
    positive = gaps[gaps > 0]
    if not len(positive):
        return []
    baseline = float(np.median(positive))
    if not math.isfinite(baseline) or baseline <= 0:
        return []
    q25, q75 = np.quantile(ordered, [0.25, 0.75])
    scale = max(float(q75 - q25), baseline)
    window = min(config.local_window, max(minimum_count, n // 8))
    found: list[tuple[float, float, float, int]] = []
    for i, gap in enumerate(gaps):
        left_count = i + 1
        right_count = n - left_count
        if left_count < minimum_count or right_count < minimum_count:
            continue
        ratio = float(gap / baseline)
        if (ratio < config.gap_multiplier or
                gap < config.minimum_relative_gap * scale):
            continue
        left_start = max(0, i - window + 1)
        right_stop = min(n - 1, i + window)
        left_span = float(ordered[i] - ordered[left_start])
        right_span = float(ordered[right_stop] - ordered[i + 1])
        if gap < config.cohesion_multiplier * max(left_span, right_span):
            continue
        found.append((float((ordered[i] + ordered[i + 1]) / 2.0),
                      float(gap), ratio, i))
    return found


def _matches(location: float, reference: tuple[float, float, float, int],
             iqr: float, config: AtlasConfig) -> bool:
    tolerance = max(config.match_gap_fraction * reference[1],
                    config.match_iqr_fraction * iqr, 1e-12)
    return abs(location - reference[0]) <= tolerance


def build_quantile_atlas(samples: np.ndarray, rng: np.random.Generator, *,
                         config: AtlasConfig = AtlasConfig()) -> QuantileAtlas:
    """Build a tail-aware atlas using full, split, and bootstrap persistence."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 1:
        raise ValueError("quantile atlas requires an (N, 1) target sample")
    if len(values) < max(128, 4 * config.minimum_segment_count):
        raise ValueError("quantile atlas sample is too small")
    if (config.minimum_segment_count <= 0 or
            config.split_minimum_segment_count <= 0 or
            config.local_window <= 1 or config.gap_multiplier <= 1 or
            config.minimum_relative_gap <= 0 or
            config.cohesion_multiplier <= 0 or
            config.bootstrap_reps < 0 or
            not 0.0 <= config.minimum_persistence <= 1.0):
        raise ValueError("invalid atlas configuration")

    ordered = np.sort(values[:, 0])
    full = _candidate_boundaries(
        ordered, config, config.minimum_segment_count)
    q25, q75 = np.quantile(ordered, [0.25, 0.75])
    iqr = max(float(q75 - q25), 1e-12)
    permutation = rng.permutation(len(ordered))
    left_values = ordered[permutation[0::2]]
    right_values = ordered[permutation[1::2]]
    left = _candidate_boundaries(
        left_values, config, config.split_minimum_segment_count)
    right = _candidate_boundaries(
        right_values, config, config.split_minimum_segment_count)

    bootstrap_locations: list[list[float]] = []
    for _ in range(config.bootstrap_reps):
        resample = ordered[rng.integers(0, len(ordered), len(ordered))]
        bootstrap_locations.append([
            candidate[0] for candidate in _candidate_boundaries(
                resample, config, config.minimum_segment_count)
        ])

    retained: list[tuple[float, float, float, int, bool, float]] = []
    for candidate in full:
        split_persistent = (
            any(_matches(location, candidate, iqr, config)
                for location, *_ in left) and
            any(_matches(location, candidate, iqr, config)
                for location, *_ in right))
        if config.bootstrap_reps:
            persistence = float(np.mean([
                any(_matches(location, candidate, iqr, config)
                    for location in locations)
                for locations in bootstrap_locations
            ]))
        else:
            persistence = 1.0
        if split_persistent and persistence >= config.minimum_persistence:
            retained.append((*candidate, split_persistent, persistence))

    retained.sort(key=lambda item: item[0])
    # Recompute segment counts from midpoint locations.  This also prevents a
    # pathological pair of retained gaps from creating a sub-minimum region.
    locations: list[float] = []
    evidence: list[BoundaryEvidence] = []
    previous_index = 0
    for location, gap, ratio, _, split_ok, persistence in retained:
        index = int(np.searchsorted(ordered, location, side="right"))
        if index - previous_index < config.minimum_segment_count:
            continue
        if len(ordered) - index < config.minimum_segment_count:
            continue
        locations.append(location)
        evidence.append(BoundaryEvidence(
            location=location, gap=gap, gap_ratio=ratio,
            left_count=index, right_count=len(ordered) - index,
            split_persistent=split_ok,
            bootstrap_persistence=persistence))
        previous_index = index

    cut_indices = [0, *(
        int(np.searchsorted(ordered, location, side="right"))
        for location in locations), len(ordered)]
    counts = [stop - start for start, stop in
              zip(cut_indices, cut_indices[1:])]
    masses = tuple(count / len(ordered) for count in counts)
    intervals = [_wilson_interval(count, len(ordered), config.confidence_z)
                 for count in counts]
    unresolved = bool(full) and not bool(locations)
    return QuantileAtlas(
        ordered=ordered[:, None], boundaries=tuple(locations),
        masses=masses,
        mass_lower=tuple(interval[0] for interval in intervals),
        mass_upper=tuple(interval[1] for interval in intervals),
        evidence=tuple(evidence), raw_candidate_count=len(full),
        unresolved=unresolved)


def assign_regions(values: np.ndarray, atlas: QuantileAtlas) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2 and array.shape[1] == 1:
        flat = array[:, 0]
    elif array.ndim == 1:
        flat = array
    else:
        raise ValueError("region assignment requires one-dimensional values")
    return np.searchsorted(np.asarray(atlas.boundaries), flat,
                           side="right").astype(int)


def randomized_systematic_target(atlas: QuantileAtlas, size: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """Select actual atlas observations with randomized systematic ranks."""
    if size <= 0 or size > atlas.sample_count:
        raise ValueError("target-table size must lie in [1, atlas size]")
    offset = float(rng.random())
    levels = (np.arange(size, dtype=float) + offset) / size
    indices = np.minimum(
        (levels * atlas.sample_count).astype(int), atlas.sample_count - 1)
    return atlas.ordered[indices].copy()


def assess_occupancy(atlas: QuantileAtlas, generated: np.ndarray, *,
                     ordinary_batch: int, target_count: float,
                     deficit_z: float = 1.645,
                     tolerance: float = 0.0) -> OccupancyAssessment:
    """Find statistically supported, ordinary-batch-underresolved deficits.

    The score compares the generated and atlas proportions with a conservative
    plug-in standard error.  Two independent positive assessments are required
    by ``OccupancyController`` before a global pulse starts.
    """
    if ordinary_batch <= 0 or target_count <= 0 or deficit_z <= 0:
        raise ValueError("invalid occupancy settings")
    labels = assign_regions(generated, atlas)
    counts = np.bincount(labels, minlength=atlas.region_count)
    n = len(labels)
    if n <= 0:
        raise ValueError("occupancy probe cannot be empty")
    q = counts / n
    scores: list[float] = []
    active: list[int] = []
    for j, (p_j, q_j) in enumerate(zip(atlas.masses, q)):
        variance = (max(p_j * (1.0 - p_j), 1.0 / n) / n +
                    max(p_j * (1.0 - p_j), 1.0 / atlas.sample_count) /
                    atlas.sample_count)
        score = (p_j - float(q_j) - tolerance) / math.sqrt(variance)
        scores.append(score)
        if (ordinary_batch * p_j < target_count and
                score >= deficit_z and q_j < p_j - tolerance):
            active.append(j)
    return OccupancyAssessment(
        target_mass=atlas.masses,
        generated_mass=tuple(float(value) for value in q),
        generated_counts=tuple(int(value) for value in counts),
        deficit_scores=tuple(scores), active_regions=tuple(active))


class OccupancyController:
    """Two-observation hysteresis controller for bounded global pulses."""

    def __init__(self, pulse_length: int, cooldown_checks: int = 1,
                 clear_checks: int = 2) -> None:
        if pulse_length <= 0 or cooldown_checks < 0 or clear_checks <= 0:
            raise ValueError("invalid controller settings")
        self.pulse_length = pulse_length
        self.cooldown_checks = cooldown_checks
        self.clear_checks = clear_checks
        self.state = "local"
        self.armed_regions: tuple[int, ...] = ()
        self.active_regions: tuple[int, ...] = ()
        self.remaining_global = 0
        self.remaining_cooldown = 0
        self.observed_clear_checks = 0
        self.transition_count = 0
        self.global_updates = 0

    @property
    def use_global(self) -> bool:
        return self.state == "global" and self.remaining_global > 0

    def _set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.transition_count += 1

    def observe(self, assessment: OccupancyAssessment) -> None:
        active = assessment.active_regions
        if self.state == "local":
            if active:
                self.armed_regions = active
                self._set_state("armed")
        elif self.state == "armed":
            repeated = tuple(sorted(set(active).intersection(
                self.armed_regions)))
            if repeated:
                self.active_regions = repeated
                self.remaining_global = self.pulse_length
                self.observed_clear_checks = 0
                self._set_state("global")
            elif active:
                self.armed_regions = active
            else:
                self.armed_regions = ()
                self._set_state("local")
        elif self.state == "global":
            if active:
                self.active_regions = active
                self.observed_clear_checks = 0
            else:
                self.observed_clear_checks += 1
                if self.observed_clear_checks >= self.clear_checks:
                    self.remaining_global = 0
                    self.remaining_cooldown = self.cooldown_checks
                    self.observed_clear_checks = 0
                    self._set_state("cooldown" if self.cooldown_checks else
                                    "local")
        elif self.state == "cooldown":
            if self.remaining_cooldown > 0:
                self.remaining_cooldown -= 1
            elif active:
                self.armed_regions = active
                self._set_state("armed")
            else:
                self._set_state("local")
        else:
            raise AssertionError(f"unknown controller state {self.state}")

    def record_global_update(self) -> None:
        if not self.use_global:
            raise RuntimeError("global update recorded outside a pulse")
        self.remaining_global -= 1
        self.global_updates += 1
        if self.remaining_global == 0:
            self.remaining_cooldown = self.cooldown_checks
            self._set_state("cooldown" if self.cooldown_checks else "local")


def choose_virtual_batch(atlas: QuantileAtlas,
                         active_regions: Iterable[int], *,
                         target_count: float,
                         grid: tuple[int, ...] = (128, 256, 512, 1024, 2048)) \
        -> VirtualBatchChoice:
    regions = tuple(int(region) for region in active_regions)
    if not regions:
        raise ValueError("virtual batch requires at least one active region")
    if target_count <= 0 or not grid or any(size <= 0 for size in grid):
        raise ValueError("invalid virtual-batch rule")
    if tuple(sorted(set(grid))) != grid:
        raise ValueError("virtual-batch grid must be strictly increasing")
    if any(not 0 <= region < atlas.region_count for region in regions):
        raise ValueError("active region outside atlas")
    point_mass = min(atlas.masses[region] for region in regions)
    conservative = min(atlas.mass_lower[region] for region in regions)
    chosen = grid[-1]
    cap_hit = True
    for size in grid:
        if size * conservative >= target_count:
            chosen = size
            cap_hit = False
            break
    return VirtualBatchChoice(
        size=chosen, active_mass=point_mass,
        conservative_mass=conservative,
        expected_count=chosen * point_mass,
        conservative_expected_count=chosen * conservative,
        cap_hit=cap_hit)


def rank_field_and_strata(x: np.ndarray, target: np.ndarray,
                          latent: np.ndarray, atlas: QuantileAtlas) \
        -> tuple[np.ndarray, np.ndarray]:
    field = exact_rank_field(x, target, latent)
    ox = np.argsort(
        x[:, 0] + 1e-10 * ((latent[:, 0] - latent[:, 0].mean()) /
                           max(float(latent[:, 0].std()), 1e-12)),
        kind="stable")
    oy = np.argsort(target[:, 0], kind="stable")
    target_labels = assign_regions(target, atlas)
    strata = np.empty(len(x), dtype=int)
    strata[ox] = target_labels[oy]
    return field, strata


def stratified_rank_selection(strata: np.ndarray, backward_batch: int,
                              rng: np.random.Generator) \
        -> StratifiedSelection:
    labels = np.asarray(strata, dtype=int)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("strata must be a nonempty vector")
    if np.any(labels < 0):
        raise ValueError("strata labels must be nonnegative")
    total = len(labels)
    backward_batch = min(int(backward_batch), total)
    if backward_batch <= 0:
        raise ValueError("backward batch must be positive")
    region_count = int(labels.max()) + 1
    counts = np.bincount(labels, minlength=region_count)
    nonempty = np.flatnonzero(counts)
    if backward_batch < len(nonempty):
        raise ValueError(
            "backward batch must cover every nonempty rank stratum")

    allocation = np.zeros(region_count, dtype=int)
    allocation[nonempty] = 1
    desired = backward_batch * counts / total
    while int(allocation.sum()) < backward_batch:
        eligible = np.flatnonzero(allocation < counts)
        if not len(eligible):
            break
        deficits = desired[eligible] - allocation[eligible]
        chosen = int(eligible[int(np.argmax(deficits))])
        allocation[chosen] += 1

    selected: list[np.ndarray] = []
    selected_weights: list[np.ndarray] = []
    for region in nonempty:
        candidates = np.flatnonzero(labels == region)
        take = int(allocation[region])
        indices = rng.choice(candidates, size=take, replace=False)
        weight = counts[region] / (total * take)
        selected.append(np.asarray(indices, dtype=int))
        selected_weights.append(np.full(take, weight, dtype=float))
    indices = np.concatenate(selected)
    weights = np.concatenate(selected_weights)
    order = np.argsort(indices, kind="stable")
    return StratifiedSelection(
        indices=indices[order], weights=weights[order],
        allocation=tuple(int(value) for value in allocation),
        counts=tuple(int(value) for value in counts))


def full_virtual_table_gradient(model: TanhMLP, latent: np.ndarray,
                                target: np.ndarray) \
        -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    x, cache = model.forward(latent, want_cache=True)
    field = exact_rank_field(x, target, latent)
    return x, field, _stopgrad_grads(model, cache, field, len(latent))


def stratified_virtual_table_gradient(
        model: TanhMLP, latent: np.ndarray, target: np.ndarray,
        atlas: QuantileAtlas, backward_batch: int,
        rng: np.random.Generator) \
        -> tuple[np.ndarray, np.ndarray, StratifiedSelection,
                 dict[str, np.ndarray]]:
    x = model.forward(latent)
    field, strata = rank_field_and_strata(x, target, latent, atlas)
    selection = stratified_rank_selection(strata, backward_batch, rng)
    chosen = selection.indices
    x_again, cache = model.forward(latent[chosen], want_cache=True)
    if not np.array_equal(x_again, x[chosen]):
        raise AssertionError("stratified rerun changed outputs before update")
    weighted_field = field[chosen] * selection.weights[:, None]
    gradient = _stopgrad_grads(model, cache, weighted_field, 1)
    return x, field, selection, gradient


def stratified_rsr_quantile_step(
        model: TanhMLP, latent: np.ndarray, target: np.ndarray,
        atlas: QuantileAtlas, microbatch: int, backward_batch: int,
        rng: np.random.Generator) \
        -> tuple[np.ndarray, np.ndarray, StratifiedSelection, OAStepWork]:
    """Global rank construction with an importance-corrected backward subset."""
    n = len(latent)
    if target.shape != (n, 1) or microbatch <= 0:
        raise ValueError("invalid stratified RSR arrays or microbatch")
    outputs = [model.forward(latent[start:start + microbatch])
               for start in range(0, n, microbatch)]
    x = np.concatenate(outputs, axis=0)
    field, strata = rank_field_and_strata(x, target, latent, atlas)
    selection = stratified_rank_selection(strata, backward_batch, rng)

    contributions: list[dict[str, np.ndarray]] = []
    chosen = selection.indices
    weights = selection.weights
    for start in range(0, len(chosen), microbatch):
        stop = min(start + microbatch, len(chosen))
        local_indices = chosen[start:stop]
        x_again, cache = model.forward(
            latent[local_indices], want_cache=True)
        if not np.array_equal(x_again, x[local_indices]):
            raise AssertionError(
                "stratified RSR rerun changed outputs before the update")
        weighted_field = (field[local_indices] *
                          weights[start:stop, None])
        contributions.append(
            _stopgrad_grads(model, cache, weighted_field, 1))
    _adam_step(model, _sum_grads(model, contributions))
    return x, field, selection, OAStepWork(
        generator_forward_calls=(math.ceil(n / microbatch) +
                                 math.ceil(len(chosen) / microbatch)),
        generator_example_evals=n + len(chosen),
        unique_latent_samples=n,
        target_samples=n,
        sort_work=2.0 * n * math.log2(max(n, 2)),
        backward_example_evals=len(chosen))


def _assert_models_close(left: TanhMLP, right: TanhMLP,
                         message: str) -> None:
    for name in left.names:
        if not np.allclose(left.params[name], right.params[name],
                           rtol=2e-12, atol=2e-12):
            raise AssertionError(f"{message}: {name}")


def invariant_tests() -> None:
    """Fast deterministic mechanism checks, including estimator exactness."""
    from copy import deepcopy
    from types import SimpleNamespace

    rng = np.random.default_rng(20260721)
    common = rng.normal(0.0, 0.07, size=(7960, 1))
    rare = rng.normal(3.0, 0.035, size=(40, 1))
    atlas = build_quantile_atlas(
        np.concatenate([common, rare]), np.random.default_rng(100))
    if atlas.region_count < 2 or min(atlas.masses) > 0.01:
        raise AssertionError("tail-aware atlas missed a compact rare region")
    normal = build_quantile_atlas(
        rng.normal(size=(8000, 1)), np.random.default_rng(101))
    if normal.region_count != 1:
        raise AssertionError("connected Gaussian produced an atlas split")
    student = build_quantile_atlas(
        rng.standard_t(4.0, size=(8000, 1)), np.random.default_rng(102))
    if student.region_count != 1:
        raise AssertionError("connected Student target produced atlas split")

    table = randomized_systematic_target(
        atlas, 1024, np.random.default_rng(103))
    table_counts = np.bincount(assign_regions(table, atlas),
                               minlength=atlas.region_count)
    if np.any(np.abs(table_counts / len(table) - atlas.masses) > 2 / len(table)):
        raise AssertionError("systematic target table lost atlas mass")

    missing_probe = np.zeros((2048, 1))
    assessment = assess_occupancy(
        atlas, missing_probe, ordinary_batch=128, target_count=8)
    if not assessment.has_deficit:
        raise AssertionError("occupancy test missed an absent rare region")
    controller = OccupancyController(pulse_length=3)
    controller.observe(assessment)
    if controller.use_global:
        raise AssertionError("controller skipped confirmation")
    controller.observe(assessment)
    if not controller.use_global:
        raise AssertionError("controller failed to start a confirmed pulse")
    for _ in range(3):
        controller.record_global_update()
    if controller.use_global:
        raise AssertionError("controller failed to end a bounded pulse")

    choice = choose_virtual_batch(
        atlas, assessment.active_regions, target_count=4)
    if choice.size not in (128, 256, 512, 1024, 2048):
        raise AssertionError("adaptive virtual batch left its grid")

    target_spec = SimpleNamespace(
        d=1, scale=3.5, means=np.asarray([[-1.0], [1.0]]))
    model = TanhMLP(target_spec, "missing", 104)
    exact_model = deepcopy(model)
    stratified_model = deepcopy(model)
    latent = np.random.default_rng(105).normal(
        size=(64, model.latent_dim))
    target = randomized_systematic_target(
        atlas, 64, np.random.default_rng(106))
    # With every rank selected, the stratified update is exactly full RSR.
    from lbqcd import rsr_quantile_step
    rsr_quantile_step(exact_model, latent, target, microbatch=16)
    stratified_rsr_quantile_step(
        stratified_model, latent, target, atlas, microbatch=16,
        backward_batch=64, rng=np.random.default_rng(107))
    _assert_models_close(
        exact_model, stratified_model,
        "full-batch stratified RSR differs from exact RSR")

    # The importance weights sum to one and cover every present stratum.
    x = model.forward(latent)
    _, strata = rank_field_and_strata(x, target, latent, atlas)
    selection = stratified_rank_selection(
        strata, 32, np.random.default_rng(108))
    if not math.isclose(float(selection.weights.sum()), 1.0,
                        rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError("stratified inclusion weights do not sum to one")
    if any(count and not allocation for count, allocation in
           zip(selection.counts, selection.allocation)):
        raise AssertionError("a nonempty rank stratum was omitted")
