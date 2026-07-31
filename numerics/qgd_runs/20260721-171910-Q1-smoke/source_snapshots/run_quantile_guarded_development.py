"""Development runner for Quantile-Guarded Drifting (QGD).

The runner constructs each target/initialization/seed's frozen LB-QCD prefix
once, clones that exact handoff into all suffix arms, and evaluates guarded,
fixed-mix, periodic, and historical paper finishers on paired streams.  Every
arm receives the same independent two-bank checkpoint-selection opportunity.

This is a development mechanism screen.  It cannot support a confirmatory or
ImageNet-scale claim.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from conservative_finishers import invariant_tests as conservative_invariants  # noqa: E402
from identifiability_drift import invariant_tests as paper_invariants  # noqa: E402
from lbqcd import (  # noqa: E402
    StepWork,
    diagnose_quantile_resolution_stable,
    direct_quantile_step,
    invariant_tests as lbqcd_invariants,
    paper_step,
    rsr_quantile_step,
)
from lowdim_drift import energy_distance2, sliced_w1  # noqa: E402
from quantile_guarded_drifting import (  # noqa: E402
    CheckpointMetric,
    GuardConfig,
    checkpoint_score,
    fixed_mix_step,
    guarded_step,
    initialize_dual_state,
    invariant_tests as qgd_invariants,
    select_checkpoint,
)
from run_identifiability_generator import TanhMLP  # noqa: E402
from run_qld_confirmatory import (  # noqa: E402
    Target1D,
    build_target,
    target_diagnostics,
)


REGISTRY = HERE / "qgd_development_registry.json"
PLAN = HERE / "QuantileGuardedDriftingResearchPlan.md"
PROTOCOL = HERE / "QuantileGuardedDriftingProtocol.md"
RUNROOT = HERE / "qgd_runs"
REGISTRY_SHA256 = \
    "3CA4E4AC966513271D1FD1B5C7B8D721926C8EE7572F18A1395FE453F8C71701"
PAPER_TAU = 0.5
PREFIX_FRACTION = 0.70
VIRTUAL_BATCH = 1024
INITS = ("missing", "concentrated")


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    batch: int
    seeds: int
    eval_size: int
    ed_size: int
    checkpoint_period: int
    bank_a_size: int
    bank_a_reps: int
    bank_b_size: int
    bank_b_reps: int
    diagnostic_samples: int


PROFILES = {
    "smoke": Profile(
        "smoke", 60, 32, 1, 512, 256, 5,
        256, 2, 512, 2, 1024),
    "screen": Profile(
        "screen", 400, 64, 3, 2048, 512, 10,
        1024, 4, 2048, 4, 4096),
}


@dataclass(frozen=True)
class Arm:
    label: str
    kind: str
    rho: float = 0.0
    local_kind: str = "paper"
    quantile_weight: float = 0.0
    metric: str = "adam"


ARMS = (
    Arm("lbqcd", "paper"),
    Arm("periodic-4p1q", "periodic"),
    Arm("fixed-mix-q0.10", "fixed", quantile_weight=0.10),
    Arm("qgd-paper-r0.05", "guarded", rho=0.05),
    Arm("qgd-paper-r0.10", "guarded", rho=0.10),
    Arm("qgd-paper-r0.20", "guarded", rho=0.20),
    Arm("qgd-sharp-r0.10", "guarded", rho=0.10,
        local_kind="sharp"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git_text(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _canonical_target(spec: dict[str, Any]) -> str:
    ignored = {"name", "family"}
    payload = {key: value for key, value in spec.items()
               if key not in ignored}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _find_target_specs(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        if "kind" in value and any(key in value for key in
                                   ("means", "K", "df")):
            yield value
        for child in value.values():
            yield from _find_target_specs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _find_target_specs(child)


def load_registry() -> dict[str, Any]:
    actual = sha256_file(REGISTRY)
    if actual != REGISTRY_SHA256:
        raise RuntimeError(
            f"QGD registry hash mismatch: {actual} != {REGISTRY_SHA256}")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("registry") != "QGD-development-v1":
        raise RuntimeError("unexpected QGD registry identifier")
    targets = registry.get("targets", [])
    if len(targets) < 8:
        raise RuntimeError("QGD development requires at least eight targets")
    names = [str(spec["name"]) for spec in targets]
    if len(set(names)) != len(names):
        raise RuntimeError("QGD target names are not unique")
    prior: dict[str, str] = {}
    for path in HERE.glob("*registry*.json"):
        if path.resolve() == REGISTRY.resolve():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for spec in _find_target_specs(obj):
            prior[_canonical_target(spec)] = path.name
    for spec in targets:
        build_target(spec)
        duplicate = prior.get(_canonical_target(spec))
        if duplicate is not None:
            raise RuntimeError(
                f"QGD target {spec['name']} duplicates {duplicate}")
    return registry


def seed_base(master: int, target_index: int, init_index: int,
              seed: int) -> int:
    return int(master * 1_000_003 + target_index * 100_003 +
               init_index * 10_007 + seed * 101)


def clone_model(model: TanhMLP) -> TanhMLP:
    cloned = copy.deepcopy(model)
    for name in model.names:
        if np.shares_memory(model.params[name], cloned.params[name]):
            raise AssertionError("cloned model shares parameter storage")
        if np.shares_memory(model.m[name], cloned.m[name]) or \
                np.shares_memory(model.v[name], cloned.v[name]):
            raise AssertionError("cloned model shares optimizer storage")
    return cloned


def _empty_work() -> dict[str, float]:
    return {key: 0.0 for key in (
        "optimizer_updates", "generator_forward_calls",
        "generator_example_evals", "target_samples", "kernel_pairs",
        "sort_work", "backward_examples",
        "diagnostic_target_samples", "selection_generator_forward_calls",
        "selection_generator_example_evals", "selection_target_samples",
        "endpoint_generator_forward_calls", "endpoint_generator_example_evals",
        "endpoint_target_samples", "checkpoint_storage_bytes")}


def _add_step_work(total: dict[str, float], work: StepWork,
                   backward_multiplier: float = 1.0) -> None:
    total["optimizer_updates"] += float(work.optimizer_updates)
    total["generator_forward_calls"] += float(work.generator_forward_calls)
    total["generator_example_evals"] += float(work.generator_example_evals)
    total["target_samples"] += float(work.target_samples)
    total["kernel_pairs"] += float(work.kernel_pairs)
    total["sort_work"] += float(work.sort_work)
    total["backward_examples"] += (
        backward_multiplier * float(work.target_samples))


def _add_guarded_work(total: dict[str, float], diagnostics) -> None:
    total["optimizer_updates"] += float(diagnostics.optimizer_updates)
    total["generator_forward_calls"] += \
        float(diagnostics.generator_forward_calls)
    total["generator_example_evals"] += \
        float(diagnostics.generator_example_evals)
    total["target_samples"] += float(diagnostics.target_samples)
    total["kernel_pairs"] += float(diagnostics.kernel_pairs)
    total["sort_work"] += float(diagnostics.sort_work)
    total["backward_examples"] += float(diagnostics.backward_examples)


def _exact_empirical_w2(generated: np.ndarray,
                        reference: np.ndarray) -> float:
    """Exact equal-weight empirical W2 on the real line."""
    if generated.shape != reference.shape or generated.ndim != 2 or \
            generated.shape[1] != 1:
        raise ValueError("empirical W2 requires equal one-dimensional arrays")
    squared = np.mean(
        (np.sort(generated[:, 0]) - np.sort(reference[:, 0])) ** 2)
    return float(np.sqrt(max(float(squared), 0.0)))


def _extended_target_diagnostics(
        generated: np.ndarray, target: Target1D) \
        -> tuple[float, float, float]:
    """Coverage, component-mass L1, and worst component-mass error."""
    coverage, mass_l1 = target_diagnostics(generated, target)
    if target.kind == "student":
        return coverage, mass_l1, 0.0
    distance = np.abs(
        generated[:, 0, None] - target.means[:, 0][None, :])
    nearest = np.argmin(distance, axis=1)
    observed = np.bincount(
        nearest, minlength=len(target.weights)) / len(generated)
    worst = float(np.max(np.abs(observed - target.weights)))
    return coverage, mass_l1, worst


def _model_storage_bytes(model: TanhMLP) -> int:
    """Bytes retained by a cloned parameter/Adam checkpoint."""
    arrays = [*model.params.values(), *model.m.values(), *model.v.values()]
    return int(sum(array.nbytes for array in arrays))


def _endpoint_metrics(model: TanhMLP, target: Target1D,
                      latent: np.ndarray, reference: np.ndarray,
                      ed_indices: np.ndarray, metric_seed: int) \
        -> dict[str, float]:
    generated = model.forward(latent)
    ed2 = max(0.0, float(energy_distance2(
        generated[ed_indices], reference[ed_indices])))
    sw1 = float(sliced_w1(
        generated, reference, 1, np.random.default_rng(metric_seed)))
    coverage, mass_l1, worst_mode_error = \
        _extended_target_diagnostics(generated, target)
    return {
        "ed2": ed2,
        "sw1": sw1,
        "w2": _exact_empirical_w2(generated, reference),
        "coverage": coverage,
        "mass_l1": mass_l1,
        "worst_mode_error": worst_mode_error,
        "mean": float(np.mean(generated)),
        "std": float(np.std(generated)),
    }


def _bank_metric(model: TanhMLP, target: Target1D,
                 latent: np.ndarray, reference: np.ndarray,
                 metric_seed: int) -> CheckpointMetric:
    generated = model.forward(latent)
    ed2 = max(0.0, float(energy_distance2(generated, reference)))
    sw1 = float(sliced_w1(
        generated, reference, 1, np.random.default_rng(metric_seed)))
    coverage, mass_l1 = target_diagnostics(generated, target)
    return CheckpointMetric(ed2, sw1, coverage, mass_l1)


def _mean_metric(metrics: list[CheckpointMetric]) -> CheckpointMetric:
    return CheckpointMetric(
        ed2=float(np.mean([item.ed2 for item in metrics])),
        sw1=float(np.mean([item.sw1 for item in metrics])),
        coverage=float(np.mean([item.coverage for item in metrics])),
        mass_l1=float(np.mean([item.mass_l1 for item in metrics])))


def _select_branch_checkpoint(
        checkpoints: dict[int, TanhMLP], target: Target1D,
        bank_a_latents: list[np.ndarray], bank_a_targets: list[np.ndarray],
        bank_b_latents: list[np.ndarray], bank_b_targets: list[np.ndarray],
        seed: int, warm_step: int, work: dict[str, float]) \
        -> tuple[int, dict[str, Any]]:
    bank_a: dict[int, list[CheckpointMetric]] = {}
    for step, model in checkpoints.items():
        bank_a[step] = [
            _bank_metric(model, target, latent, reference,
                         seed + 1000 + replicate)
            for replicate, (latent, reference) in enumerate(
                zip(bank_a_latents, bank_a_targets))
        ]
    work["selection_generator_forward_calls"] += \
        len(checkpoints) * len(bank_a_latents)
    work["selection_generator_example_evals"] += sum(
        len(latent) for latent in bank_a_latents) * len(checkpoints)
    work["selection_target_samples"] += sum(
        len(reference) for reference in bank_a_targets) * len(checkpoints)

    handoff = _mean_metric(bank_a[warm_step])
    minimum_coverage = max(0.0, handoff.coverage - 0.05)
    maximum_mass_l1 = handoff.mass_l1 + 0.10
    eligible_steps = [
        step for step, metrics in bank_a.items()
        if (all(math.isfinite(checkpoint_score(item, handoff))
                for item in metrics) and
            float(np.mean([item.coverage for item in metrics])) >=
                minimum_coverage and
            float(np.mean([item.mass_l1 for item in metrics])) <=
                maximum_mass_l1)
    ]
    if not eligible_steps:
        eligible_steps = [warm_step]
    a_means = {
        step: float(np.mean([
            checkpoint_score(item, handoff) for item in bank_a[step]]))
        for step in eligible_steps
    }
    leaders = sorted(a_means, key=lambda step: (a_means[step], step))[
        :min(3, len(a_means))]
    bank_b: dict[int, list[CheckpointMetric]] = {}
    for step in leaders:
        model = checkpoints[step]
        bank_b[step] = [
            _bank_metric(model, target, latent, reference,
                         seed + 2000 + replicate)
            for replicate, (latent, reference) in enumerate(
                zip(bank_b_latents, bank_b_targets))
        ]
    work["selection_generator_forward_calls"] += \
        len(leaders) * len(bank_b_latents)
    work["selection_generator_example_evals"] += sum(
        len(latent) for latent in bank_b_latents) * len(leaders)
    work["selection_target_samples"] += sum(
        len(reference) for reference in bank_b_targets) * len(leaders)
    selection = select_checkpoint(
        bank_a, bank_b, handoff, top_k=3,
        minimum_coverage=minimum_coverage,
        maximum_mass_l1=maximum_mass_l1)
    return selection.selected_step, {
        **asdict(selection),
        "handoff": asdict(handoff),
        "minimum_coverage": minimum_coverage,
        "maximum_mass_l1": maximum_mass_l1,
        "bank_a": {
            str(step): [asdict(item) for item in metrics]
            for step, metrics in bank_a.items()
        },
        "bank_b": {
            str(step): [asdict(item) for item in metrics]
            for step, metrics in bank_b.items()
        },
    }


def _branch(
        arm: Arm, prefix: TanhMLP, profile: Profile, target: Target1D,
        warm_steps: int, suffix_latents: np.ndarray,
        suffix_targets: np.ndarray, endpoint_latent: np.ndarray,
        endpoint_target: np.ndarray, ed_indices: np.ndarray,
        bank_a_latents: list[np.ndarray], bank_a_targets: list[np.ndarray],
        bank_b_latents: list[np.ndarray], bank_b_targets: list[np.ndarray],
        metric_seed: int, prefix_work: dict[str, float]) \
        -> tuple[dict[str, Any], dict[str, Any]]:
    model = clone_model(prefix)
    work = dict(prefix_work)
    checkpoints: dict[int, TanhMLP] = {warm_steps: clone_model(model)}
    dual_state = initialize_dual_state(model)
    active_counts: dict[str, int] = {}
    trust_count = safe_count = incompatible_count = singular_count = 0
    correction_norms: list[float] = []
    q_derivatives: list[float] = []
    local_derivatives: list[float] = []
    gradient_cosines: list[float] = []
    proposal_cosines: list[float] = []
    diverged = False

    for offset in range(len(suffix_latents)):
        step = warm_steps + offset + 1
        latent = suffix_latents[offset]
        positive = suffix_targets[offset]
        if arm.kind == "paper":
            _, _, step_work = paper_step(
                model, latent, positive, PAPER_TAU)
            _add_step_work(work, step_work)
        elif arm.kind == "periodic":
            if (offset + 1) % 5 == 0:
                _, _, step_work = direct_quantile_step(
                    model, latent, positive)
            else:
                _, _, step_work = paper_step(
                    model, latent, positive, PAPER_TAU)
            _add_step_work(work, step_work)
        elif arm.kind == "fixed":
            result = fixed_mix_step(
                model, dual_state, latent, positive,
                quantile_weight=arm.quantile_weight,
                local_kind="paper", tau=PAPER_TAU,
                metric_kind=arm.metric)  # type: ignore[arg-type]
            dual_state = result.state
            diagnostics = result.diagnostics
            _add_guarded_work(work, diagnostics)
            active_counts[diagnostics.projection.active_set] = \
                active_counts.get(diagnostics.projection.active_set, 0) + 1
            correction_norms.append(
                diagnostics.projection.correction_metric_norm)
            gradient_cosines.append(
                diagnostics.projection.gradient_cosine)
            proposal_cosines.append(
                diagnostics.projection.proposal_cosine)
        elif arm.kind == "guarded":
            result = guarded_step(
                model, dual_state, latent, positive,
                local_kind=arm.local_kind, tau=PAPER_TAU,
                config=GuardConfig(rho=arm.rho,
                                   metric=arm.metric))  # type: ignore[arg-type]
            dual_state = result.state
            diagnostics = result.diagnostics
            projection = diagnostics.projection
            _add_guarded_work(work, diagnostics)
            active_counts[projection.active_set] = \
                active_counts.get(projection.active_set, 0) + 1
            trust_count += int(projection.trust_cap_active)
            safe_count += int(projection.safe_quantile_fallback)
            incompatible_count += int(projection.incompatible)
            singular_count += int(projection.singular_candidates)
            correction_norms.append(projection.correction_metric_norm)
            q_derivatives.append(projection.quantile_constraint_after)
            local_derivatives.append(projection.local_constraint_after)
            gradient_cosines.append(projection.gradient_cosine)
            proposal_cosines.append(projection.proposal_cosine)
        else:
            raise ValueError(f"unknown QGD arm kind {arm.kind}")
        if not model.finite():
            diverged = True
            break
        if (step == profile.steps or
                (step - warm_steps) % profile.checkpoint_period == 0):
            checkpoints[step] = clone_model(model)

    if profile.steps not in checkpoints and not diverged:
        checkpoints[profile.steps] = clone_model(model)
    work["checkpoint_storage_bytes"] = float(sum(
        _model_storage_bytes(checkpoint)
        for checkpoint in checkpoints.values()))
    endpoint = (_endpoint_metrics(
        model, target, endpoint_latent, endpoint_target,
        ed_indices, metric_seed)
        if not diverged else {
            "ed2": float("inf"), "sw1": float("inf"),
            "w2": float("inf"),
            "coverage": float("nan"), "mass_l1": float("nan"),
            "worst_mode_error": float("nan"),
            "mean": float("nan"), "std": float("nan")})
    work["endpoint_generator_forward_calls"] += 1.0
    work["endpoint_generator_example_evals"] += len(endpoint_latent)
    work["endpoint_target_samples"] += len(endpoint_target)

    if diverged:
        selected_step = max(checkpoints)
        selection_record: dict[str, Any] = {"status": "diverged"}
        selected = dict(endpoint)
    else:
        selected_step, selection_record = _select_branch_checkpoint(
            checkpoints, target, bank_a_latents, bank_a_targets,
            bank_b_latents, bank_b_targets, metric_seed, warm_steps, work)
        selected_model = checkpoints[selected_step]
        selected = _endpoint_metrics(
            selected_model, target, endpoint_latent, endpoint_target,
            ed_indices, metric_seed)
        work["endpoint_generator_forward_calls"] += 1.0
        work["endpoint_generator_example_evals"] += len(endpoint_latent)
        work["endpoint_target_samples"] += len(endpoint_target)

    handoff = _endpoint_metrics(
        checkpoints[warm_steps], target, endpoint_latent, endpoint_target,
        ed_indices, metric_seed)
    work["endpoint_generator_forward_calls"] += 1.0
    work["endpoint_generator_example_evals"] += len(endpoint_latent)
    work["endpoint_target_samples"] += len(endpoint_target)

    coverage_event_time: int | None = None
    if not diverged:
        bank_a = selection_record.get("bank_a", {})
        for raw_step in sorted(bank_a, key=int):
            metrics = bank_a[raw_step]
            mean_coverage = float(np.mean([
                float(metric["coverage"]) for metric in metrics]))
            if mean_coverage >= 0.90:
                coverage_event_time = int(raw_step)
                break

    suffix_steps = max(1, len(suffix_latents))
    finite_cosines = [value for value in gradient_cosines
                      if math.isfinite(value)]
    finite_proposal_cosines = [value for value in proposal_cosines
                               if math.isfinite(value)]
    row = {
        "arm": arm.label,
        "kind": arm.kind,
        "rho": arm.rho,
        "local_kind": arm.local_kind,
        "metric": arm.metric,
        "quantile_weight": arm.quantile_weight,
        "endpoint_ed2": endpoint["ed2"],
        "endpoint_sw1": endpoint["sw1"],
        "endpoint_w2": endpoint["w2"],
        "endpoint_coverage": endpoint["coverage"],
        "endpoint_mass_l1": endpoint["mass_l1"],
        "endpoint_worst_mode_error": endpoint["worst_mode_error"],
        "selected_ed2": selected["ed2"],
        "selected_sw1": selected["sw1"],
        "selected_w2": selected["w2"],
        "selected_coverage": selected["coverage"],
        "selected_mass_l1": selected["mass_l1"],
        "selected_worst_mode_error": selected["worst_mode_error"],
        "handoff_ed2": handoff["ed2"],
        "handoff_sw1": handoff["sw1"],
        "handoff_w2": handoff["w2"],
        "handoff_coverage": handoff["coverage"],
        "handoff_mass_l1": handoff["mass_l1"],
        "handoff_worst_mode_error": handoff["worst_mode_error"],
        "selected_ed2_change_from_handoff":
            selected["ed2"] - handoff["ed2"],
        "selected_sw1_change_from_handoff":
            selected["sw1"] - handoff["sw1"],
        "endpoint_ed2_change_from_handoff":
            endpoint["ed2"] - handoff["ed2"],
        "endpoint_sw1_change_from_handoff":
            endpoint["sw1"] - handoff["sw1"],
        "coverage_event_time": (
            coverage_event_time if coverage_event_time is not None
            else profile.steps),
        "coverage_event_censored": int(coverage_event_time is None),
        "selected_step": selected_step,
        "diverged": int(diverged),
        "projection_active_fraction": (
            1.0 - active_counts.get("none", 0) / suffix_steps
            if arm.kind == "guarded" else 0.0),
        "trust_fraction": trust_count / suffix_steps,
        "safe_quantile_fraction": safe_count / suffix_steps,
        "incompatible_fraction": incompatible_count / suffix_steps,
        "singular_candidates": singular_count,
        "median_correction_metric_norm": (
            float(np.median(correction_norms))
            if correction_norms else float("nan")),
        "p90_correction_metric_norm": (
            float(np.quantile(correction_norms, 0.9))
            if correction_norms else float("nan")),
        "median_gradient_cosine": (
            float(np.median(finite_cosines))
            if finite_cosines else float("nan")),
        "median_proposal_cosine": (
            float(np.median(finite_proposal_cosines))
            if finite_proposal_cosines else float("nan")),
        "median_quantile_directional_derivative": (
            float(np.median(q_derivatives))
            if q_derivatives else float("nan")),
        "median_local_directional_derivative": (
            float(np.median(local_derivatives))
            if local_derivatives else float("nan")),
        "active_set_counts": json.dumps(active_counts, sort_keys=True),
        **{f"work_{key}": value for key, value in work.items()},
    }
    selection_record.update({"arm": arm.label})
    return row, selection_record


def run_trial_group(task: dict[str, Any]) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target = build_target(task["target_spec"])
    profile = Profile(**task["profile"])
    base = int(task["base_seed"])
    seed = int(task["seed"])
    init_kind = str(task["init"])
    warm_steps = int(round(PREFIX_FRACTION * profile.steps))
    wall0 = time.perf_counter()

    template = TanhMLP(target, init_kind, base + 1)
    diagnosis_target = target.sample(
        profile.diagnostic_samples, np.random.default_rng(base + 2))
    diagnosis = diagnose_quantile_resolution_stable(
        diagnosis_target, profile.batch)
    prefix_latent_rng = np.random.default_rng(base + 3)
    prefix_target_rng = np.random.default_rng(base + 4)
    prefix = clone_model(template)
    prefix_work = _empty_work()
    prefix_work["diagnostic_target_samples"] = profile.diagnostic_samples
    for _ in range(warm_steps):
        n = VIRTUAL_BATCH if diagnosis.use_large_batch else profile.batch
        latent = prefix_latent_rng.normal(size=(n, prefix.latent_dim))
        positive = target.sample(n, prefix_target_rng)
        if diagnosis.use_large_batch:
            _, _, step_work = rsr_quantile_step(
                prefix, latent, positive, microbatch=profile.batch)
        else:
            _, _, step_work = direct_quantile_step(
                prefix, latent, positive)
        _add_step_work(prefix_work, step_work)
    if not prefix.finite():
        raise FloatingPointError("shared LB-QCD prefix diverged")

    suffix_steps = profile.steps - warm_steps
    suffix_latents = np.random.default_rng(base + 5).normal(
        size=(suffix_steps, profile.batch, prefix.latent_dim))
    suffix_target_rng = np.random.default_rng(base + 6)
    suffix_targets = np.asarray([
        target.sample(profile.batch, suffix_target_rng)
        for _ in range(suffix_steps)])

    endpoint_latent = np.random.default_rng(base + 7).normal(
        size=(profile.eval_size, prefix.latent_dim))
    endpoint_target = target.sample(
        profile.eval_size, np.random.default_rng(base + 8))
    ed_indices = np.random.default_rng(base + 9).choice(
        profile.eval_size, profile.ed_size, replace=False)

    bank_a_latents = [
        np.random.default_rng(base + 100 + replicate).normal(
            size=(profile.bank_a_size, prefix.latent_dim))
        for replicate in range(profile.bank_a_reps)]
    bank_a_targets = [
        target.sample(profile.bank_a_size,
                      np.random.default_rng(base + 200 + replicate))
        for replicate in range(profile.bank_a_reps)]
    bank_b_latents = [
        np.random.default_rng(base + 300 + replicate).normal(
            size=(profile.bank_b_size, prefix.latent_dim))
        for replicate in range(profile.bank_b_reps)]
    bank_b_targets = [
        target.sample(profile.bank_b_size,
                      np.random.default_rng(base + 400 + replicate))
        for replicate in range(profile.bank_b_reps)]

    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    for arm in ARMS:
        row, selection = _branch(
            arm, prefix, profile, target, warm_steps,
            suffix_latents, suffix_targets, endpoint_latent,
            endpoint_target, ed_indices, bank_a_latents, bank_a_targets,
            bank_b_latents, bank_b_targets, base + 500, prefix_work)
        row.update({
            "target": target.name,
            "family": target.family,
            "init": init_kind,
            "cell": f"{target.name}/{init_kind}",
            "seed": seed,
            "diagnostic_use_large_batch": int(diagnosis.use_large_batch),
            "diagnostic_min_expected_count":
                diagnosis.minimum_expected_batch_count,
            "diagnostic_gap_count": diagnosis.significant_gap_count,
            "diagnostic_max_gap_ratio": diagnosis.maximum_gap_ratio,
            "group_wall_seconds": time.perf_counter() - wall0,
        })
        selection.update({
            "target": target.name,
            "family": target.family,
            "init": init_kind,
            "seed": seed,
        })
        rows.append(row)
        selections.append(selection)
    return rows, selections


def make_tasks(registry: dict[str, Any], profile: Profile) \
        -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    master = int(registry["master_seed"])
    for target_index, spec in enumerate(registry["targets"]):
        for init_index, init in enumerate(INITS):
            for seed in range(profile.seeds):
                tasks.append({
                    "target_spec": spec,
                    "profile": asdict(profile),
                    "init": init,
                    "seed": seed,
                    "base_seed": seed_base(
                        master, target_index, init_index, seed),
                })
    return tasks


def execute(tasks: list[dict[str, Any]], workers: int) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    if workers == 1:
        for index, task in enumerate(tasks, 1):
            result_rows, result_selections = run_trial_group(task)
            rows.extend(result_rows)
            selections.extend(result_selections)
            print(f"[{index}/{len(tasks)}] {task['target_spec']['name']} "
                  f"{task['init']} seed={task['seed']}", flush=True)
        return rows, selections
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_trial_group, task): task for task in tasks}
        completed = 0
        for future in as_completed(futures):
            result_rows, result_selections = future.result()
            rows.extend(result_rows)
            selections.extend(result_selections)
            completed += 1
            task = futures[future]
            print(f"[{completed}/{len(tasks)}] {task['target_spec']['name']} "
                  f"{task['init']} seed={task['seed']}", flush=True)
    return rows, selections


def _geometric_mean(values: list[float]) -> float:
    safe = [max(float(value), 1e-300) for value in values
            if math.isfinite(float(value))]
    return float(math.exp(np.mean(np.log(safe)))) if safe else float("nan")


def _cell_medians(rows: list[dict[str, Any]], metric: str) \
        -> dict[str, dict[str, float]]:
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((str(row["arm"]), str(row["cell"])), []).append(
            float(row[metric]))
    result: dict[str, dict[str, float]] = {}
    for (arm, cell), values in grouped.items():
        result.setdefault(arm, {})[cell] = float(np.median(values))
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = "lbqcd"
    selected_ed = _cell_medians(rows, "selected_ed2")
    selected_sw = _cell_medians(rows, "selected_sw1")
    selected_w2 = _cell_medians(rows, "selected_w2")
    endpoint_ed = _cell_medians(rows, "endpoint_ed2")
    endpoint_sw = _cell_medians(rows, "endpoint_sw1")
    endpoint_w2 = _cell_medians(rows, "endpoint_w2")
    arms = sorted(selected_ed)

    def ratios(table: dict[str, dict[str, float]], arm: str,
               cells: list[str] | None = None) -> float:
        common = sorted(set(table[arm]) & set(table[baseline]))
        if cells is not None:
            allowed = set(cells)
            common = [cell for cell in common if cell in allowed]
        return _geometric_mean([
            table[arm][cell] / max(table[baseline][cell], 1e-300)
            for cell in common])

    selected_ed_ratios = {arm: ratios(selected_ed, arm) for arm in arms}
    selected_sw_ratios = {arm: ratios(selected_sw, arm) for arm in arms}
    selected_w2_ratios = {arm: ratios(selected_w2, arm) for arm in arms}
    endpoint_ed_ratios = {arm: ratios(endpoint_ed, arm) for arm in arms}
    endpoint_sw_ratios = {arm: ratios(endpoint_sw, arm) for arm in arms}
    endpoint_w2_ratios = {arm: ratios(endpoint_w2, arm) for arm in arms}
    families = sorted(set(str(row["family"]) for row in rows))
    family_cells = {
        family: sorted(set(str(row["cell"]) for row in rows
                           if row["family"] == family))
        for family in families
    }
    family_ratios = {
        arm: {family: ratios(selected_ed, arm, cells)
              for family, cells in family_cells.items()}
        for arm in arms
    }
    init_ratios: dict[str, dict[str, float]] = {}
    for arm in arms:
        init_ratios[arm] = {}
        for init in INITS:
            cells = sorted(set(str(row["cell"]) for row in rows
                               if row["init"] == init))
            init_ratios[arm][init] = ratios(selected_ed, arm, cells)
    mechanism: dict[str, dict[str, Any]] = {}
    outcome_diagnostics: dict[str, dict[str, float | int]] = {}
    work_diagnostics: dict[str, dict[str, float]] = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        active_totals: dict[str, int] = {}
        for row in arm_rows:
            for label, count in json.loads(
                    str(row["active_set_counts"])).items():
                active_totals[label] = active_totals.get(label, 0) + int(count)
        mechanism[arm] = {
            "projection_active_fraction": float(np.mean([
                float(row["projection_active_fraction"])
                for row in arm_rows])),
            "trust_fraction": float(np.mean([
                float(row["trust_fraction"]) for row in arm_rows])),
            "safe_quantile_fraction": float(np.mean([
                float(row["safe_quantile_fraction"]) for row in arm_rows])),
            "incompatible_fraction": float(np.mean([
                float(row["incompatible_fraction"]) for row in arm_rows])),
            "divergences": int(sum(int(row["diverged"])
                                    for row in arm_rows)),
            "active_set_counts": active_totals,
        }
        outcome_diagnostics[arm] = {
            "median_selected_coverage": float(np.median([
                float(row["selected_coverage"]) for row in arm_rows])),
            "median_selected_mass_l1": float(np.median([
                float(row["selected_mass_l1"]) for row in arm_rows])),
            "median_selected_worst_mode_error": float(np.median([
                float(row["selected_worst_mode_error"])
                for row in arm_rows])),
            "coverage_events_censored": int(sum(
                int(row["coverage_event_censored"]) for row in arm_rows)),
            "median_coverage_event_time": float(np.median([
                float(row["coverage_event_time"]) for row in arm_rows])),
            "median_selected_step": float(np.median([
                float(row["selected_step"]) for row in arm_rows])),
        }
        work_keys = sorted(
            key for key in arm_rows[0] if key.startswith("work_"))
        work_diagnostics[arm] = {
            key.removeprefix("work_"): float(np.median([
                float(row[key]) for row in arm_rows]))
            for key in work_keys
        }
    eligible_candidates = [arm for arm in arms if arm.startswith("qgd-paper")]
    best = min(eligible_candidates,
               key=lambda arm: selected_ed_ratios[arm])
    checks = {
        "selected_ed2_vs_lbqcd_at_most_0.95":
            selected_ed_ratios[best] <= 0.95,
        "selected_sw1_vs_lbqcd_at_most_0.98":
            selected_sw_ratios[best] <= 0.98,
        "all_family_ed2_ratios_at_most_1.05":
            all(value <= 1.05 for value in family_ratios[best].values()),
        "all_init_ed2_ratios_at_most_1.02":
            all(value <= 1.02 for value in init_ratios[best].values()),
        "no_divergence": mechanism[best]["divergences"] == 0,
        "safe_quantile_fraction_below_0.05":
            mechanism[best]["safe_quantile_fraction"] < 0.05,
        "trust_fraction_below_0.10":
            mechanism[best]["trust_fraction"] < 0.10,
    }
    return {
        "baseline": baseline,
        "selected_ed2_ratios_vs_lbqcd": selected_ed_ratios,
        "selected_sw1_ratios_vs_lbqcd": selected_sw_ratios,
        "selected_w2_ratios_vs_lbqcd": selected_w2_ratios,
        "endpoint_ed2_ratios_vs_lbqcd": endpoint_ed_ratios,
        "endpoint_sw1_ratios_vs_lbqcd": endpoint_sw_ratios,
        "endpoint_w2_ratios_vs_lbqcd": endpoint_w2_ratios,
        "family_selected_ed2_ratios_vs_lbqcd": family_ratios,
        "init_selected_ed2_ratios_vs_lbqcd": init_ratios,
        "mechanism": mechanism,
        "outcome_diagnostics": outcome_diagnostics,
        "median_work_per_run": work_diagnostics,
        "best_registered_paper_qgd": best,
        "advancement_checks": checks,
        "advancement_passed": all(checks.values()),
    }


def _format_ratio(value: Any) -> str:
    number = float(value)
    return f"{number:.4f}" if math.isfinite(number) else "NA"


def _markdown_report(summary: dict[str, Any], profile: Profile,
                     row_count: int, wall_seconds: float) -> str:
    """Human-readable companion to the machine-readable Q1 artifacts."""
    arms = sorted(summary["selected_ed2_ratios_vs_lbqcd"])
    lines = [
        f"# Quantile-Guarded Drifting {profile.name.title()} Results",
        "",
        "> Development evidence only. This is not a sealed confirmation and "
        "does not support an ImageNet or paper-FID claim.",
        "",
        f"The run produced {row_count} arm-level rows in "
        f"{wall_seconds:.1f} seconds. All ratios below use the same-run "
        "LB-QCD arm as 1.0 and aggregate target/initialization cells by the "
        "geometric mean of paired cell-median ratios.",
        "",
        "## Main outcomes",
        "",
        "| Arm | Selected ED2 | Selected SW1 | Selected W2 | Endpoint ED2 | Endpoint SW1 | Endpoint W2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        lines.append(
            f"| {arm} | "
            f"{_format_ratio(summary['selected_ed2_ratios_vs_lbqcd'][arm])} | "
            f"{_format_ratio(summary['selected_sw1_ratios_vs_lbqcd'][arm])} | "
            f"{_format_ratio(summary['selected_w2_ratios_vs_lbqcd'][arm])} | "
            f"{_format_ratio(summary['endpoint_ed2_ratios_vs_lbqcd'][arm])} | "
            f"{_format_ratio(summary['endpoint_sw1_ratios_vs_lbqcd'][arm])} | "
            f"{_format_ratio(summary['endpoint_w2_ratios_vs_lbqcd'][arm])} |")
    lines.extend([
        "",
        "Lower ratios are better. Checkpoint selection used independent "
        "Bank A ranking followed by Bank B earliest-within-one-SE confirmation.",
        "",
        "## Mechanism diagnostics",
        "",
        "| Arm | Projection active | Safe-Q fallback | Trust fallback | Incompatible | Divergences |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for arm in arms:
        item = summary["mechanism"][arm]
        lines.append(
            f"| {arm} | {float(item['projection_active_fraction']):.3f} | "
            f"{float(item['safe_quantile_fraction']):.3f} | "
            f"{float(item['trust_fraction']):.3f} | "
            f"{float(item['incompatible_fraction']):.3f} | "
            f"{int(item['divergences'])} |")
    lines.extend([
        "",
        "## Registered advancement decision",
        "",
        f"Best registered paper-QGD arm: "
        f"`{summary['best_registered_paper_qgd']}`.",
        "",
    ])
    for check, passed in summary["advancement_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — `{check}`")
    lines.extend([
        "",
        f"Overall advancement gate: "
        f"**{'PASS' if summary['advancement_passed'] else 'FAIL'}**.",
        "",
        "The complete family splits, initialization splits, active-set "
        "counts, outcome diagnostics, and compute ledger are in `summary.json`; "
        "per-run values are in `rows.csv`; checkpoint selection traces are in "
        "`checkpoint_selections.json`.",
        "",
    ])
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(
            rows, key=lambda row: (str(row["target"]), str(row["init"]),
                                   int(row["seed"]), str(row["arm"]))))


def invariant_tests(log=print) -> dict[str, Any]:
    lbqcd_invariants()
    paper_invariants(log=lambda _: None)
    conservative_invariants(log=lambda _: None)
    qgd_report = qgd_invariants(log=lambda _: None)
    registry = load_registry()
    report = {
        "status": "pass",
        "lbqcd": "pass",
        "paper": "pass",
        "conservative": "pass",
        "qgd": qgd_report,
        "registry_targets": len(registry["targets"]),
        "registry_sha256": REGISTRY_SHA256,
    }
    log("QGD runner invariants and disjoint registry: PASS")
    return report


def _run_directory(profile: Profile) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNROOT / f"{stamp}-Q1-{profile.name}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "source_snapshots").mkdir()
    return path


def _write_manifest(path: Path, profile: Profile, workers: int,
                    task_count: int, wall_seconds: float,
                    pre_status: str, invariant_report: dict[str, Any]) -> None:
    sources = [
        Path(__file__), HERE / "quantile_guarded_drifting.py", REGISTRY,
        PLAN, PROTOCOL, HERE / "lbqcd.py",
        HERE / "run_identifiability_generator.py",
        HERE / "identifiability_drift.py",
        HERE / "conservative_finishers.py", HERE / "lowdim_drift.py",
    ]
    manifest = {
        "protocol": "QGD-development-Q1-v1",
        "profile": asdict(profile),
        "arms": [asdict(arm) for arm in ARMS],
        "prefix_fraction": PREFIX_FRACTION,
        "virtual_batch": VIRTUAL_BATCH,
        "paper_tau": PAPER_TAU,
        "initializations": list(INITS),
        "workers": workers,
        "task_count": task_count,
        "registry_sha256": REGISTRY_SHA256,
        "commit": git_text("rev-parse", "HEAD"),
        "pre_run_git_status": pre_status.splitlines(),
        "command": sys.argv,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "wall_seconds": wall_seconds,
        "invariants": invariant_report,
        "source_sha256": {
            str(source.relative_to(ROOT)): sha256_file(source)
            for source in sources
        },
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    for source in sources:
        shutil.copy2(source, path / "source_snapshots" / source.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES),
                        default="smoke")
    parser.add_argument("--workers", type=int,
                        default=min(8, os.cpu_count() or 1))
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    invariant_report = invariant_tests()
    if args.self_check_only:
        return
    profile = PROFILES[args.profile]
    registry = load_registry()
    tasks = make_tasks(registry, profile)
    pre_status = git_text("status", "--porcelain")
    run_dir = _run_directory(profile)
    print(f"run directory: {run_dir}", flush=True)
    print(f"tasks: {len(tasks)} groups; arms/group: {len(ARMS)}", flush=True)
    wall0 = time.perf_counter()
    rows, selections = execute(tasks, max(1, args.workers))
    wall = time.perf_counter() - wall0
    summary = summarize(rows)
    _write_csv(run_dir / "rows.csv", rows)
    (run_dir / "checkpoint_selections.json").write_text(
        json.dumps(selections, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "RESULTS.md").write_text(
        _markdown_report(summary, profile, len(rows), wall),
        encoding="utf-8")
    _write_manifest(run_dir, profile, args.workers, len(tasks), wall,
                    pre_status, invariant_report)
    print(json.dumps({
        "selected_ed2_ratios_vs_lbqcd":
            summary["selected_ed2_ratios_vs_lbqcd"],
        "selected_sw1_ratios_vs_lbqcd":
            summary["selected_sw1_ratios_vs_lbqcd"],
        "endpoint_ed2_ratios_vs_lbqcd":
            summary["endpoint_ed2_ratios_vs_lbqcd"],
        "best_registered_paper_qgd":
            summary["best_registered_paper_qgd"],
        "advancement_checks": summary["advancement_checks"],
        "advancement_passed": summary["advancement_passed"],
    }, indent=2), flush=True)
    print(f"completed in {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
