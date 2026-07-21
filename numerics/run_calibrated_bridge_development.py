"""Causal optimizer screen for the QLD -> sharp -> paper bridge.

This runner implements the first gated experiment in
``CalibratedConservativeBridgePlan.md``.  It does not implement scale
normalization or multiscale kernels; those branches are prohibited unless the
registered optimizer-calibration gate passes.
"""

from __future__ import annotations

import argparse
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

from calibrated_bridge import (  # noqa: E402
    PARAMETER_ORDER,
    adam_step_configurable,
    capture_optimizer,
    clone_model,
    cosine,
    flatten_named,
    invariant_tests as bridge_invariant_tests,
    optimizer_matches,
    reset_optimizer,
    restore_optimizer,
    sharp_deleted_bridge_step,
)
from conservative_finishers import (  # noqa: E402
    invariant_tests as conservative_invariant_tests,
    sharp_laplace_field,
)
from identifiability_drift import compute_field as compute_paper_field  # noqa: E402
from lbqcd import _adam_step, _stopgrad_grads, exact_rank_field  # noqa: E402
from lowdim_drift import energy_distance2, sliced_w1  # noqa: E402
from run_conservative_finisher_development import (  # noqa: E402
    PRIMARY_INITS,
    REGISTRY_A,
    build_target,
    git_text,
    hierarchical_ratio_bootstrap,
    load_registry_a,
    regression_invariant_tests,
    seed_base,
    sha256_file,
    target_diagnostics,
)
from run_identifiability_generator import ADAM_LR, TanhMLP  # noqa: E402


PLAN = HERE / "CalibratedConservativeBridgePlan.md"
PROTOCOL = HERE / "CalibratedBridgeProtocol.md"
RUNROOT = HERE / "bridge_runs"
BRIDGE_LENGTH = 20
PREFIX_FRACTION = 0.70
SHARP_TAU = 0.5
SHARP_LR_FRACTION = 0.25
CAP_MULTIPLIER = 1.25
CALIBRATION_UPDATES = 5


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    batch: int
    seeds: int
    eval_size: int
    ed_size: int
    trajectory_period: int
    bootstrap_reps: int


PROFILES = {
    "smoke": Profile("smoke", 120, 32, 1, 512, 256, 25, 200),
    "screen": Profile("screen", 400, 64, 3, 2048, 512, 25, 2000),
}


@dataclass(frozen=True)
class Arm:
    label: str
    kind: str
    sharp_optimizer: str | None = None
    eligible: bool = True


ARMS = (
    Arm("qld-v1", "qld-paper"),
    Arm("qld-full", "qld-full"),
    Arm("bridge-reset-full-lr", "bridge", "reset-full"),
    Arm("bridge-reset-quarter", "bridge", "reset-quarter"),
    Arm("bridge-warm-quarter", "bridge", "warm-quarter"),
    Arm("bridge-calibrated", "bridge", "calibrated"),
    Arm("bridge-carry-copy", "bridge", "carry-copy"),
    Arm("qld-sharp20-only", "bridge-only", "calibrated", eligible=False),
)


def _empty_work() -> dict[str, float]:
    return {
        "optimizer_updates": 0.0,
        "generator_forward_calls": 0.0,
        "generator_example_evals": 0.0,
        "unique_latent_samples": 0.0,
        "positive_target_samples": 0.0,
        "kernel_pairs": 0.0,
        "sort_work": 0.0,
        "backward_examples": 0.0,
        "metric_generator_forward_calls": 0.0,
        "metric_generator_example_evals": 0.0,
        "metric_target_samples": 0.0,
        "diagnostic_generator_forward_calls": 0.0,
        "diagnostic_generator_example_evals": 0.0,
        "diagnostic_target_samples": 0.0,
        "calibration_optimizer_updates": 0.0,
        "calibration_generator_forward_calls": 0.0,
        "calibration_generator_example_evals": 0.0,
        "calibration_unique_latent_samples": 0.0,
        "calibration_target_samples": 0.0,
        "calibration_kernel_pairs": 0.0,
        "calibration_backward_examples": 0.0,
    }


def _add_work(total: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def _parameter_delta(before: dict[str, np.ndarray], model: TanhMLP) \
        -> dict[str, np.ndarray]:
    return {name: model.params[name] - before[name]
            for name in PARAMETER_ORDER}


def _standard_step(model: TanhMLP, phase: str, latent: np.ndarray,
                   positive: np.ndarray) -> dict[str, float]:
    queries, cache = model.forward(latent, want_cache=True)
    if phase == "QLD":
        field = exact_rank_field(queries, positive, latent)
        kernel_pairs = 0.0
        sort_work = 2.0 * len(queries) * math.log2(max(len(queries), 2))
    elif phase == "paper_stabilizer":
        result = compute_paper_field(
            queries, positive, tau=SHARP_TAU, gain="paper", mask=True,
            on_degenerate="zero")
        field = result.V
        kernel_pairs = float(result.kernel_pairs)
        sort_work = 0.0
    else:
        raise ValueError(phase)
    gradients = _stopgrad_grads(model, cache, field, len(field))
    before = {name: model.params[name].copy() for name in PARAMETER_ORDER}
    _adam_step(model, gradients)
    delta = _parameter_delta(before, model)
    flat_gradient = flatten_named(gradients)
    flat_delta = flatten_named(delta)
    n = len(latent)
    return {
        "field_rms": float(np.sqrt(np.mean(field * field))),
        "field_max": float(np.max(np.linalg.norm(field, axis=1))),
        "parameter_gradient_norm": float(np.linalg.norm(flat_gradient)),
        "pre_cap_step_norm": float(np.linalg.norm(flat_delta)),
        "committed_step_norm": float(np.linalg.norm(flat_delta)),
        "descent_step_cosine": cosine(-flat_gradient, flat_delta),
        "learning_rate": ADAM_LR,
        "trust_cap": float("nan"),
        "cap_activated": 0.0,
        "floor_activations": 0.0,
        "optimizer_updates": 1.0,
        "generator_forward_calls": 1.0,
        "generator_example_evals": float(n),
        "unique_latent_samples": float(n),
        "positive_target_samples": float(n),
        "kernel_pairs": kernel_pairs,
        "sort_work": sort_work,
        "backward_examples": float(n),
    }


def _sharp_learning_rate(optimizer: str, bridge_index: int) -> float:
    if optimizer in ("reset-full", "carry-copy"):
        return ADAM_LR
    if optimizer == "reset-quarter":
        return SHARP_LR_FRACTION * ADAM_LR
    if optimizer in ("warm-quarter", "calibrated"):
        return (SHARP_LR_FRACTION * ADAM_LR * bridge_index /
                BRIDGE_LENGTH)
    raise ValueError(f"unknown sharp optimizer {optimizer}")


def _sharp_step(model: TanhMLP, optimizer: str, bridge_index: int,
                latent: np.ndarray, positive: np.ndarray,
                trust_cap: float | None) -> dict[str, float]:
    learning_rate = _sharp_learning_rate(optimizer, bridge_index)
    cap = trust_cap if optimizer == "calibrated" else None
    result = sharp_deleted_bridge_step(
        model, latent, positive, tau=SHARP_TAU,
        learning_rate=learning_rate, maximum_parameter_step=cap)
    field = result.field_result.field
    gradient = flatten_named(result.gradients)
    n = len(latent)
    return {
        "field_rms": float(np.sqrt(np.mean(field * field))),
        "field_max": float(np.max(np.linalg.norm(field, axis=1))),
        "parameter_gradient_norm": float(np.linalg.norm(gradient)),
        "pre_cap_step_norm": result.adam.pre_cap_step_norm,
        "committed_step_norm": result.adam.committed_step_norm,
        "descent_step_cosine": result.adam.descent_step_cosine,
        "learning_rate": learning_rate,
        "trust_cap": cap if cap is not None else float("nan"),
        "cap_activated": float(result.adam.cap_activated),
        "floor_activations": float(
            result.field_result.diagnostics.floor_activations),
        "optimizer_updates": 1.0,
        "generator_forward_calls": 1.0,
        "generator_example_evals": float(n),
        "unique_latent_samples": float(n),
        "positive_target_samples": float(n),
        "kernel_pairs": float(
            result.field_result.diagnostics.kernel_pairs),
        "sort_work": 0.0,
        "backward_examples": float(n),
        **result.occupancy,
    }


def calibrate_paper_step(
        prefix: TanhMLP, latent_batches: np.ndarray,
        positive_batches: np.ndarray) -> tuple[float, dict[str, float]]:
    """Estimate a label-free carried-paper step scale on a discarded clone."""
    clone = clone_model(prefix)
    norms: list[float] = []
    work = _empty_work()
    for latent, positive in zip(latent_batches, positive_batches):
        diagnostics = _standard_step(
            clone, "paper_stabilizer", latent, positive)
        norms.append(diagnostics["committed_step_norm"])
        work["calibration_optimizer_updates"] += 1.0
        work["calibration_generator_forward_calls"] += 1.0
        work["calibration_generator_example_evals"] += len(latent)
        work["calibration_unique_latent_samples"] += len(latent)
        work["calibration_target_samples"] += len(positive)
        work["calibration_kernel_pairs"] += diagnostics["kernel_pairs"]
        work["calibration_backward_examples"] += len(latent)
    return float(np.median(norms)), work


def _evaluate(model: TanhMLP, target, evaluation_latent: np.ndarray,
              target_reference: np.ndarray, ed_indices: np.ndarray,
              metric_seed: int, minimum_mass: float) -> dict[str, float]:
    generated = model.forward(evaluation_latent)
    ed2 = max(0.0, float(energy_distance2(
        generated[ed_indices], target_reference[ed_indices])))
    sw1 = float(sliced_w1(
        generated, target_reference, 1,
        np.random.default_rng(metric_seed)))
    probabilities = np.arange(1, 32) / 32
    boundaries = np.quantile(target_reference[:, 0], probabilities)
    generated_cdf = np.asarray([
        np.mean(generated[:, 0] <= boundary) for boundary in boundaries])
    component = target_diagnostics(generated, target, minimum_mass)
    quantiles = np.quantile(
        generated[:, 0], [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "ed2": ed2,
        "sw1": sw1,
        "quantile_cdf_error": float(np.mean(
            np.abs(generated_cdf - probabilities))),
        "output_mean": float(np.mean(generated)),
        "output_std": float(np.std(generated)),
        "output_q01": float(quantiles[0]),
        "output_q10": float(quantiles[1]),
        "output_q50": float(quantiles[2]),
        "output_q90": float(quantiles[3]),
        "output_q99": float(quantiles[4]),
        **component,
    }


def _phase_steps(profile: Profile) -> tuple[int, int]:
    prefix_end = int(round(PREFIX_FRACTION * profile.steps))
    bridge_end = prefix_end + BRIDGE_LENGTH
    if bridge_end >= profile.steps:
        raise ValueError("profile leaves no paper stabilization updates")
    return prefix_end, bridge_end


def _trajectory_steps(profile: Profile) -> set[int]:
    prefix_end, bridge_end = _phase_steps(profile)
    steps = {0, prefix_end, bridge_end, profile.steps}
    steps.update(range(profile.trajectory_period, prefix_end + 1,
                       profile.trajectory_period))
    steps.update(range(prefix_end + 1, bridge_end + 1))
    steps.update(range(bridge_end + 1,
                       min(profile.steps, bridge_end + 5) + 1))
    steps.update(range(
        ((bridge_end // profile.trajectory_period) + 1) *
        profile.trajectory_period,
        profile.steps + 1, profile.trajectory_period))
    return steps


def _independent_gradient_diagnostics(
        model: TanhMLP, latent: np.ndarray,
        positive: np.ndarray) -> tuple[float, float]:
    queries, cache = model.forward(latent, want_cache=True)
    qld_field = exact_rank_field(queries, positive, latent)
    paper = compute_paper_field(
        queries, positive, tau=SHARP_TAU, gain="paper", mask=True,
        on_degenerate="zero").V
    sharp = sharp_laplace_field(
        queries, positive, tau=SHARP_TAU,
        reference_mode="reused_deleted").field
    qld_gradient = flatten_named(
        _stopgrad_grads(model, cache, qld_field, len(queries)))
    paper_gradient = flatten_named(
        _stopgrad_grads(model, cache, paper, len(queries)))
    sharp_gradient = flatten_named(
        _stopgrad_grads(model, cache, sharp, len(queries)))
    return (cosine(sharp_gradient, qld_gradient),
            cosine(sharp_gradient, paper_gradient))


def _trajectory_row(
        arm: Arm, step: int, phase: str, model: TanhMLP, target,
        evaluation_latent: np.ndarray, target_reference: np.ndarray,
        ed_indices: np.ndarray, metric_seed: int, minimum_mass: float,
        latest: dict[str, float] | None, work: dict[str, float],
        diagnostic_latent: np.ndarray, diagnostic_positive: np.ndarray,
        wall0: float, paper_state_step: int,
        paper_state_restored: bool, paper_calibration_step: float,
        trust_cap: float) -> dict[str, Any]:
    work["metric_generator_forward_calls"] += 1.0
    work["metric_generator_example_evals"] += len(evaluation_latent)
    work["metric_target_samples"] += len(target_reference)
    metrics = _evaluate(
        model, target, evaluation_latent, target_reference, ed_indices,
        metric_seed, minimum_mass)
    sharp_qld, sharp_paper = _independent_gradient_diagnostics(
        model, diagnostic_latent, diagnostic_positive)
    work["diagnostic_generator_forward_calls"] += 1.0
    work["diagnostic_generator_example_evals"] += len(diagnostic_latent)
    work["diagnostic_target_samples"] += len(diagnostic_positive)
    defaults = {
        "field_rms": float("nan"),
        "field_max": float("nan"),
        "parameter_gradient_norm": float("nan"),
        "pre_cap_step_norm": float("nan"),
        "committed_step_norm": float("nan"),
        "descent_step_cosine": float("nan"),
        "learning_rate": float("nan"),
        "trust_cap": trust_cap,
        "cap_activated": 0.0,
        "floor_activations": 0.0,
    }
    if latest is not None:
        defaults.update({key: latest[key] for key in defaults
                         if key in latest})
    occupancy = ({key: latest[key] for key in latest
                  if (key.startswith("positive_") or
                      key.startswith("negative_"))}
                 if latest is not None else {})
    return {
        "arm": arm.label,
        "step": step,
        "phase": phase,
        **metrics,
        **defaults,
        **occupancy,
        "sharp_qld_gradient_cosine": sharp_qld,
        "sharp_paper_gradient_cosine": sharp_paper,
        "paper_state_saved_step_index": paper_state_step,
        "paper_state_restored": int(paper_state_restored),
        "paper_calibration_median_step": paper_calibration_step,
        **{f"work_{key}": value for key, value in work.items()},
        "wall_seconds": time.perf_counter() - wall0,
    }


def _run_arm(
        arm: Arm, prefix: TanhMLP, profile: Profile, target,
        latent_batches: np.ndarray, positive_batches: np.ndarray,
        evaluation_latent: np.ndarray, target_reference: np.ndarray,
        ed_indices: np.ndarray, diagnostic_latent: np.ndarray,
        diagnostic_positive: np.ndarray, metric_seed: int,
        minimum_mass: float, prefix_rows: list[dict[str, Any]],
        prefix_work: dict[str, float], calibration_step: float,
        calibration_work: dict[str, float], wall0: float) \
        -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    prefix_end, bridge_end = _phase_steps(profile)
    final_step = bridge_end if arm.kind == "bridge-only" else profile.steps
    model = clone_model(prefix)
    saved_paper_state = capture_optimizer(model)
    paper_state_step = saved_paper_state.step_index
    paper_state_restored = False
    state_restoration_exact = arm.kind != "bridge"
    work = dict(prefix_work)
    if arm.sharp_optimizer == "calibrated":
        _add_work(work, calibration_work)
    trust_cap = CAP_MULTIPLIER * calibration_step
    trajectory = [dict(row, arm=arm.label) for row in prefix_rows]
    latest: dict[str, float] | None = None
    cap_activations = 0
    floor_activations = 0
    diverged = False
    steps_to_record = _trajectory_steps(profile)

    if arm.kind in ("bridge", "bridge-only"):
        if arm.sharp_optimizer != "carry-copy":
            reset_optimizer(model)

    for step in range(prefix_end + 1, final_step + 1):
        if arm.kind == "qld-full":
            phase = "QLD"
            latest = _standard_step(
                model, phase, latent_batches[step - 1],
                positive_batches[step - 1])
        elif arm.kind == "qld-paper":
            phase = "paper_stabilizer"
            latest = _standard_step(
                model, phase, latent_batches[step - 1],
                positive_batches[step - 1])
        elif step <= bridge_end:
            phase = "sharp_bridge"
            bridge_index = step - prefix_end
            latest = _sharp_step(
                model, str(arm.sharp_optimizer), bridge_index,
                latent_batches[step - 1], positive_batches[step - 1],
                trust_cap)
        else:
            phase = "paper_stabilizer"
            if not paper_state_restored:
                restore_optimizer(model, saved_paper_state)
                state_restoration_exact = optimizer_matches(
                    model, saved_paper_state)
                if not state_restoration_exact:
                    raise AssertionError(
                        "restored paper optimizer differs from snapshot")
                paper_state_restored = True
            latest = _standard_step(
                model, phase, latent_batches[step - 1],
                positive_batches[step - 1])
        _add_work(work, {key: latest[key] for key in _empty_work()
                         if key in latest})
        cap_activations += int(latest["cap_activated"])
        floor_activations += int(latest["floor_activations"])
        if not model.finite():
            diverged = True
            break
        if step in steps_to_record or step == final_step:
            trajectory.append(_trajectory_row(
                arm, step, phase, model, target, evaluation_latent,
                target_reference, ed_indices, metric_seed, minimum_mass,
                latest, work, diagnostic_latent, diagnostic_positive, wall0,
                paper_state_step, paper_state_restored, calibration_step,
                trust_cap))

    if not diverged:
        endpoint = trajectory[-1]
    else:
        endpoint = {
            "ed2": float("inf"), "sw1": float("inf"),
            "quantile_cdf_error": float("inf"),
            "mass_l1": float("nan"),
            "minimum_component_occupancy": float("nan"),
            "all_reportable_components_reached": float("nan"),
        }
    handoff = next(row for row in trajectory
                   if int(row["step"]) == prefix_end)
    bridge_exit = next((row for row in trajectory
                        if int(row["step"]) == bridge_end), None)
    if arm.kind == "bridge" and not paper_state_restored:
        raise AssertionError("bridge failed to restore paper optimizer state")
    expected_final_paper_step = (
        paper_state_step + profile.steps - bridge_end)
    if (arm.kind == "bridge" and not diverged and
            model.step_index != expected_final_paper_step):
        raise AssertionError("restored paper optimizer advanced incorrectly")
    row = {
        "arm": arm.label,
        "kind": arm.kind,
        "sharp_optimizer": arm.sharp_optimizer,
        "eligible": int(arm.eligible),
        "completed_step": (int(trajectory[-1]["step"])
                           if not diverged else int(trajectory[-1]["step"])),
        "ed2": endpoint["ed2"],
        "sw1": endpoint["sw1"],
        "quantile_cdf_error": endpoint["quantile_cdf_error"],
        "mass_l1": endpoint["mass_l1"],
        "minimum_component_occupancy": endpoint[
            "minimum_component_occupancy"],
        "all_reportable_components_reached": endpoint[
            "all_reportable_components_reached"],
        "handoff_ed2": handoff["ed2"],
        "handoff_sw1": handoff["sw1"],
        "bridge_exit_ed2": (bridge_exit["ed2"]
                            if bridge_exit is not None else float("nan")),
        "bridge_exit_sw1": (bridge_exit["sw1"]
                            if bridge_exit is not None else float("nan")),
        "bridge_ed2_change": ((bridge_exit["ed2"] - handoff["ed2"])
                              if bridge_exit is not None else float("nan")),
        "paper_ed2_change": ((endpoint["ed2"] - bridge_exit["ed2"])
                             if bridge_exit is not None and
                             arm.kind == "bridge" else float("nan")),
        "cap_activations": cap_activations,
        "floor_activations": floor_activations,
        "paper_calibration_median_step": calibration_step,
        "trust_cap": trust_cap,
        "paper_state_saved_step_index": paper_state_step,
        "paper_state_restoration_exact": int(state_restoration_exact),
        "diverged": int(diverged),
        "wall_seconds": time.perf_counter() - wall0,
        **{f"work_{key}": value for key, value in work.items()},
    }
    return row, trajectory, work


def run_trial_group(task: dict[str, Any]) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target = build_target(task["target_spec"])
    profile = Profile(**task["profile"])
    prefix_end, _ = _phase_steps(profile)
    base = int(task["base_seed"])
    seed = int(task["seed"])
    init = str(task["init"])
    minimum_mass = float(task["minimum_mass"])
    wall0 = time.perf_counter()

    template = TanhMLP(target, init, base + 1)
    latent_rng = np.random.default_rng(base + 2)
    data_rng = np.random.default_rng(base + 3)
    latent_batches = latent_rng.normal(
        size=(profile.steps, profile.batch, template.latent_dim))
    positive_batches = np.asarray([
        target.sample(profile.batch, data_rng) for _ in range(profile.steps)])
    calibration_latents = np.random.default_rng(base + 4).normal(
        size=(CALIBRATION_UPDATES, profile.batch, template.latent_dim))
    calibration_positive_rng = np.random.default_rng(base + 5)
    calibration_positives = np.asarray([
        target.sample(profile.batch, calibration_positive_rng)
        for _ in range(CALIBRATION_UPDATES)])
    evaluation_latent = np.random.default_rng(base + 6).normal(
        size=(profile.eval_size, template.latent_dim))
    target_reference = target.sample(
        profile.eval_size, np.random.default_rng(base + 7))
    ed_indices = np.random.default_rng(base + 8).choice(
        profile.eval_size, size=profile.ed_size, replace=False)
    diagnostic_latent = np.random.default_rng(base + 9).normal(
        size=(profile.batch, template.latent_dim))
    diagnostic_positive = target.sample(
        profile.batch, np.random.default_rng(base + 10))
    metric_seed = base + 11

    prefix = clone_model(template)
    prefix_work = _empty_work()
    prefix_rows: list[dict[str, Any]] = []
    dummy_arm = ARMS[0]
    prefix_rows.append(_trajectory_row(
        dummy_arm, 0, "initial", prefix, target, evaluation_latent,
        target_reference, ed_indices, metric_seed, minimum_mass, None,
        prefix_work, diagnostic_latent, diagnostic_positive, wall0, 0, False,
        float("nan"), float("nan")))
    record_steps = _trajectory_steps(profile)
    latest: dict[str, float] | None = None
    for step in range(1, prefix_end + 1):
        latest = _standard_step(
            prefix, "QLD", latent_batches[step - 1],
            positive_batches[step - 1])
        _add_work(prefix_work, {key: latest[key] for key in _empty_work()
                                if key in latest})
        if step in record_steps:
            prefix_rows.append(_trajectory_row(
                dummy_arm, step, "QLD", prefix, target,
                evaluation_latent, target_reference, ed_indices, metric_seed,
                minimum_mass, latest, prefix_work, diagnostic_latent,
                diagnostic_positive, wall0, prefix.step_index, False,
                float("nan"), float("nan")))

    prefix_snapshot = clone_model(prefix)
    calibration_step, calibration_work = calibrate_paper_step(
        prefix, calibration_latents, calibration_positives)
    if any(not np.array_equal(prefix.params[name], prefix_snapshot.params[name])
           or not np.array_equal(prefix.m[name], prefix_snapshot.m[name])
           or not np.array_equal(prefix.v[name], prefix_snapshot.v[name])
           for name in PARAMETER_ORDER):
        raise AssertionError("paper calibration mutated the shared prefix")
    if (calibration_work["calibration_optimizer_updates"] !=
            CALIBRATION_UPDATES):
        raise AssertionError("paper calibration work was not fully counted")

    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    for arm in ARMS:
        row, arm_trajectory, ledger = _run_arm(
            arm, prefix, profile, target, latent_batches, positive_batches,
            evaluation_latent, target_reference, ed_indices,
            diagnostic_latent, diagnostic_positive, metric_seed, minimum_mass,
            prefix_rows, prefix_work, calibration_step, calibration_work,
            wall0)
        metadata = {
            "target": target.name,
            "family": target.family,
            "init": init,
            "seed": seed,
            "cell": f"{target.name}/{init}",
        }
        rows.append({**metadata, **row})
        trajectories.extend([{**metadata, **item}
                             for item in arm_trajectory])
        ledgers.append({**metadata, "arm": arm.label, **ledger})
    return rows, trajectories, ledgers


def make_tasks(registry: dict[str, Any], profile: Profile) \
        -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    master = int(registry["master_seed"]) + 73_001
    for target_index, target_spec in enumerate(registry["targets"]):
        for init_index, init in enumerate(PRIMARY_INITS):
            for seed in range(profile.seeds):
                tasks.append({
                    "target_spec": target_spec,
                    "profile": asdict(profile),
                    "init": init,
                    "seed": seed,
                    "base_seed": seed_base(
                        master, target_index, init_index, seed),
                    "minimum_mass": registry[
                        "minimum_reportable_component_mass"],
                })
    return tasks


def execute(tasks: list[dict[str, Any]], workers: int) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    if workers <= 1:
        for index, task in enumerate(tasks, start=1):
            r, t, w = run_trial_group(task)
            rows.extend(r); trajectories.extend(t); ledgers.extend(w)
            print(f"completed trial group {index}/{len(tasks)}")
        return rows, trajectories, ledgers
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_trial_group, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            r, t, w = future.result()
            rows.extend(r); trajectories.extend(t); ledgers.extend(w)
            print(f"completed trial group {index}/{len(tasks)}")
    return rows, trajectories, ledgers


def _geomean(values: Iterable[float]) -> float:
    values = [max(float(value), 1e-300) for value in values]
    return float(math.exp(np.mean(np.log(values))))


def _cell_medians(rows: list[dict[str, Any]], metric: str) \
        -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for cell in sorted({str(row["cell"]) for row in rows}):
        for arm in sorted({str(row["arm"]) for row in rows}):
            selected = [float(row[metric]) for row in rows
                        if row["cell"] == cell and row["arm"] == arm]
            out[(cell, arm)] = float(np.median(selected))
    return out


def summarize(rows: list[dict[str, Any]], trajectories: list[dict[str, Any]],
              profile: Profile) -> dict[str, Any]:
    eligible_rows = [row for row in rows if int(row["eligible"]) == 1]
    med_ed = _cell_medians(rows, "ed2")
    med_sw = _cell_medians(rows, "sw1")
    cells = sorted({str(row["cell"]) for row in rows})
    arms = sorted({str(row["arm"]) for row in eligible_rows})
    ed_ratios = {
        arm: (1.0 if arm == "qld-v1" else _geomean(
            med_ed[(cell, arm)] /
            max(med_ed[(cell, "qld-v1")], 1e-300)
            for cell in cells))
        for arm in arms
    }
    sw_ratios = {
        arm: (1.0 if arm == "qld-v1" else _geomean(
            med_sw[(cell, arm)] /
            max(med_sw[(cell, "qld-v1")], 1e-300)
            for cell in cells))
        for arm in arms
    }
    connected_cells = sorted({str(row["cell"]) for row in rows
                              if str(row["family"]).startswith("connected")})
    connected = {
        arm: {cell: med_ed[(cell, arm)] /
              max(med_ed[(cell, "qld-v1")], 1e-300)
              for cell in connected_cells}
        for arm in arms
    }
    initialization = {
        arm: {init: _geomean(
            med_ed[(cell, arm)] /
            max(med_ed[(cell, "qld-v1")], 1e-300)
            for cell in cells if cell.endswith("/" + init))
            for init in PRIMARY_INITS}
        for arm in arms
    }
    primary = "bridge-calibrated"
    bridge_steps = [row for row in trajectories
                    if row["arm"] == primary and
                    row["phase"] == "sharp_bridge"]
    sharp_committed = np.asarray([
        float(row["committed_step_norm"]) for row in bridge_steps])
    # Compare each sharp update to the carried-paper scale calibrated at its
    # own prefix.  This is the predeclared label-free five-update reference,
    # and avoids estimating the denominator from sparse trajectory samples.
    sharp_to_paper = np.asarray([
        float(row["committed_step_norm"]) /
        max(float(row["paper_calibration_median_step"]), 1e-300)
        for row in bridge_steps])
    median_step_ratio = float(np.median(sharp_to_paper))
    p90_step_ratio = float(np.quantile(sharp_to_paper, 0.90))
    primary_rows = [row for row in rows if row["arm"] == primary]
    gates = {
        "ed2_vs_qld_at_most_0.98": ed_ratios[primary] <= 0.98,
        "sw1_vs_qld_at_most_0.99": sw_ratios[primary] <= 0.99,
        "worst_connected_ed2_at_most_1.05": all(
            value <= 1.05 for value in connected[primary].values()),
        "each_initialization_ed2_at_most_1.02": all(
            value <= 1.02 for value in initialization[primary].values()),
        "no_divergence": all(int(row["diverged"]) == 0
                             for row in primary_rows),
        "no_denominator_floor": all(int(row["floor_activations"]) == 0
                                    for row in primary_rows),
        "median_step_ratio_between_0.5_and_1.5":
            0.5 <= median_step_ratio <= 1.5,
        "p90_step_ratio_at_most_2": p90_step_ratio <= 2.0,
        "paper_state_restoration_exact": all(
            int(row["paper_state_restoration_exact"]) == 1
            for row in primary_rows),
    }
    bootstrap = hierarchical_ratio_bootstrap(
        rows, primary, "qld-v1", "ed2", profile.bootstrap_reps, 20260724)
    arm_phase = {
        arm: {
            "bridge_ed2_change_mean": float(np.nanmean([
                float(row["bridge_ed2_change"]) for row in rows
                if row["arm"] == arm])),
            "paper_ed2_change_mean": float(np.nanmean([
                float(row["paper_ed2_change"]) for row in rows
                if row["arm"] == arm])),
        }
        for arm in arms if arm.startswith("bridge-")
    }
    return {
        "profile": profile.name,
        "row_count": len(rows),
        "cell_count": len(cells),
        "ed2_ratios_vs_qld": ed_ratios,
        "sw1_ratios_vs_qld": sw_ratios,
        "connected_control_ed2_ratios": connected,
        "initialization_ed2_ratios": initialization,
        "primary_hierarchical_ed2_bootstrap": bootstrap,
        "primary_median_step_ratio": median_step_ratio,
        "primary_p90_step_ratio": p90_step_ratio,
        "primary_cap_activation_count": sum(
            int(row["cap_activations"]) for row in primary_rows),
        "arm_phase_changes": arm_phase,
        "gates": gates,
        "gate_pass": all(gates.values()),
        "decision": ("optimizer calibration passes; scale stage permitted"
                     if all(gates.values()) else
                     "optimizer calibration fails; stop before scale stage"),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _source_files() -> tuple[Path, ...]:
    return (
        Path(__file__), HERE / "calibrated_bridge.py", PLAN, PROTOCOL,
        REGISTRY_A, HERE / "conservative_finishers.py",
        HERE / "run_conservative_finisher_development.py",
        HERE / "identifiability_drift.py", HERE / "lbqcd.py",
        HERE / "lowdim_drift.py",
        HERE / "run_identifiability_generator.py",
    )


def _make_run_dir(profile: Profile) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNROOT / f"{stamp}-B1-{profile.name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_results(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Calibrated bridge B1 ({summary['profile']})", "",
        "This is an exposed Registry-A mechanism test, not confirmation.", "",
        f"Gate pass: **{summary['gate_pass']}**", "",
        f"Decision: {summary['decision']}", "",
        "| arm | ED2 / QLD-v1 | SW1 / QLD-v1 |", "|---|---:|---:|",
    ]
    for arm, value in sorted(summary["ed2_ratios_vs_qld"].items()):
        lines.append(
            f"| `{arm}` | {value:.6g} | "
            f"{summary['sw1_ratios_vs_qld'][arm]:.6g} |")
    lines.extend(["", "## Registered gates", ""])
    for gate, passed in summary["gates"].items():
        lines.append(f"- `{gate}`: **{passed}**")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_artifacts(
        run_dir: Path, args: argparse.Namespace, profile: Profile,
        invariants: dict[str, Any], rows: list[dict[str, Any]],
        trajectories: list[dict[str, Any]], ledgers: list[dict[str, Any]],
        summary: dict[str, Any]) -> None:
    write_csv(run_dir / "rows.csv", rows)
    write_csv(run_dir / "trajectories.csv", trajectories)
    (run_dir / "work_ledger.json").write_text(
        json.dumps(ledgers, indent=2, allow_nan=True), encoding="utf-8")
    occupancy = [{key: value for key, value in row.items()
                  if (key.startswith("positive_") or
                      key.startswith("negative_") or
                      key in ("target", "init", "seed", "arm", "step",
                              "phase"))}
                 for row in trajectories if row["phase"] == "sharp_bridge"]
    optimizer = [{key: row.get(key) for key in (
        "target", "init", "seed", "arm", "step", "phase",
        "learning_rate", "pre_cap_step_norm", "committed_step_norm",
        "trust_cap", "cap_activated", "descent_step_cosine",
        "paper_state_saved_step_index", "paper_state_restored")}
                 for row in trajectories]
    (run_dir / "occupancy_diagnostics.json").write_text(
        json.dumps(occupancy, indent=2, allow_nan=True), encoding="utf-8")
    (run_dir / "optimizer_diagnostics.json").write_text(
        json.dumps(optimizer, indent=2, allow_nan=True), encoding="utf-8")
    (run_dir / "invariants.json").write_text(
        json.dumps(invariants, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    hashes = {source.name: sha256_file(source)
              for source in _source_files()}
    (run_dir / "source_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8")
    snapshots = run_dir / "source_snapshots"; snapshots.mkdir()
    for source in _source_files():
        shutil.copy2(source, snapshots / source.name)
    manifest = {
        "stage": "B1-optimizer-calibration",
        "profile": asdict(profile),
        "arguments": vars(args),
        "registry_sha256": sha256_file(REGISTRY_A),
        "plan_sha256": sha256_file(PLAN),
        "protocol_sha256": sha256_file(PROTOCOL),
        "git_head": git_text("rev-parse", "HEAD"),
        "git_status": git_text("status", "--short"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "arms": [asdict(arm) for arm in ARMS],
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    _write_results(run_dir / "RESULTS.md", summary)


def runner_invariant_tests(log=print) -> dict[str, object]:
    registry = load_registry_a()
    target = build_target(registry["targets"][0])
    model = TanhMLP(target, "missing", 101)
    original = clone_model(model)
    rng = np.random.default_rng(20260724)
    latent = rng.normal(
        size=(CALIBRATION_UPDATES, 8, model.latent_dim))
    positive = np.asarray([
        target.sample(8, rng) for _ in range(CALIBRATION_UPDATES)])
    step, work = calibrate_paper_step(model, latent, positive)
    if not math.isfinite(step) or step <= 0.0:
        raise AssertionError("paper calibration returned invalid step")
    if any(not np.array_equal(model.params[name], original.params[name]) or
           not np.array_equal(model.m[name], original.m[name]) or
           not np.array_equal(model.v[name], original.v[name])
           for name in PARAMETER_ORDER):
        raise AssertionError("calibration mutated source model")
    if (work["calibration_optimizer_updates"] != CALIBRATION_UPDATES or
            work["calibration_generator_example_evals"] !=
            CALIBRATION_UPDATES * 8 or
            work["calibration_unique_latent_samples"] !=
            CALIBRATION_UPDATES * 8 or
            work["calibration_target_samples"] != CALIBRATION_UPDATES * 8):
        raise AssertionError("calibration ledger is incomplete")
    if (work["calibration_backward_examples"] !=
            CALIBRATION_UPDATES * 8):
        raise AssertionError("calibration backward work is not charged")
    report = {"status": "pass", "check_count": 2,
              "checks": ["calibration_clone_and_ledger",
                         "calibration_backward_and_unique_samples"]}
    log("calibrated bridge runner invariants: PASS (2 checks)")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("B0", "B1"), nargs="?",
                        default="B0")
    parser.add_argument("--profile", choices=tuple(PROFILES),
                        default="smoke")
    parser.add_argument("--workers", type=int,
                        default=max(1, min(4, os.cpu_count() or 1)))
    args = parser.parse_args()
    registry = load_registry_a()
    invariants = {
        "conservative_fields": conservative_invariant_tests(),
        "historical_baseline_regression": regression_invariant_tests(),
        "calibrated_bridge": bridge_invariant_tests(),
        "runner": runner_invariant_tests(),
    }
    if args.stage == "B0":
        print(json.dumps(invariants, indent=2)); return
    profile = PROFILES[args.profile]
    tasks = make_tasks(registry, profile)
    run_dir = _make_run_dir(profile)
    print(f"run directory: {run_dir}")
    rows, trajectories, ledgers = execute(tasks, args.workers)
    summary = summarize(rows, trajectories, profile)
    _write_artifacts(
        run_dir, args, profile, invariants, rows, trajectories, ledgers,
        summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
