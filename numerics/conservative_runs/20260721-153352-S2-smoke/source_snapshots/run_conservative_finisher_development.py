"""Staged common-prefix evaluation of conservative QLD finishers.

This runner implements S0--S2 of
``QLDSharpConservativeImplementationPlan.md``.  It intentionally uses the
repository's existing paper field, QLD rank field, ``TanhMLP``, and Adam
update.  New code is confined to conservative finishers, branch-safe state
handling, diagnostics, and immutable artifacts.
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

from conservative_finishers import (  # noqa: E402
    FieldResult,
    conservative_field,
    invariant_tests as conservative_invariant_tests,
    sharp_logkde_loss,
)
from identifiability_drift import (  # noqa: E402
    compute_field as compute_paper_field,
    invariant_tests as paper_field_invariant_tests,
)
from lbqcd import (  # noqa: E402
    _adam_step,
    _stopgrad_grads,
    exact_rank_field,
)
from lowdim_drift import (  # noqa: E402
    energy_distance2,
    field_invariants as paper_port_invariants,
    sliced_w1,
)
from run_identifiability_generator import TanhMLP  # noqa: E402


PLAN = HERE / "QLDSharpConservativeImplementationPlan.md"
PROTOCOL = HERE / "ConservativeFinisherProtocol.md"
REGISTRY_A = HERE / "conservative_registry_a.json"
RUNROOT = HERE / "conservative_runs"
REGISTRY_A_SHA256 = \
    "48AA09809A78413E2C2C2314618677FDC3B870C638EAD69C05E205DB6089AF12"
PRIMARY_INITS = ("missing", "concentrated")
STRESS_INITS = ("far",)
TAUS = (0.2, 0.5, 1.0)
PAPER_TAU = 0.5
PREFIX_FRACTION = 0.70
PARAMETER_ORDER = tuple(TanhMLP.names)


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
    "smoke": Profile("smoke", 60, 32, 1, 512, 256, 25, 200),
    "screen": Profile("screen", 400, 64, 3, 2048, 512, 25, 2000),
}


@dataclass(frozen=True)
class Arm:
    label: str
    finisher: str
    tau: float | None
    reference_mode: str | None
    handoff: str
    standalone_paper: bool = False


@dataclass
class Target1D:
    name: str
    family: str
    kind: str
    means: np.ndarray
    weights: np.ndarray
    sigmas: np.ndarray
    scale: float
    df: float | None = None
    log_sigma: float | None = None

    d: int = 1

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "student":
            return (rng.standard_t(float(self.df), size=(n, 1)) *
                    float(self.sigmas[0]))
        if self.kind == "shifted_lognormal":
            sigma = float(self.log_sigma)
            centered = (np.exp(sigma * rng.normal(size=(n, 1))) -
                        math.exp(0.5 * sigma * sigma))
            return centered * float(self.sigmas[0])
        component = rng.choice(len(self.means), size=n, p=self.weights)
        return (self.means[component] +
                rng.normal(size=(n, 1)) * self.sigmas[component, None])

    @property
    def has_components(self) -> bool:
        return self.kind == "mixture"


@dataclass(frozen=True)
class OptimizerState:
    """An explicit Adam state for later objective-specific alternation."""

    step_index: int
    m: dict[str, np.ndarray]
    v: dict[str, np.ndarray]


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
        if "kind" in value and any(key in value for key in (
                "means", "K", "df", "log_sigma")):
            yield value
        for child in value.values():
            yield from _find_target_specs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _find_target_specs(child)


def _reject_prior_duplicates(targets: list[dict[str, Any]]) -> None:
    prior: dict[str, str] = {}
    for path in HERE.glob("*registry*.json"):
        if path.resolve() == REGISTRY_A.resolve():
            continue
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for spec in _find_target_specs(obj):
            prior[_canonical_target(spec)] = path.name
    seen: set[str] = set()
    for spec in targets:
        canonical = _canonical_target(spec)
        if canonical in seen:
            raise RuntimeError(f"duplicate target inside Registry A: "
                               f"{spec['name']}")
        if canonical in prior:
            raise RuntimeError(
                f"Registry-A target {spec['name']} duplicates "
                f"{prior[canonical]}")
        seen.add(canonical)


def load_registry_a() -> dict[str, Any]:
    actual = sha256_file(REGISTRY_A)
    if actual != REGISTRY_A_SHA256:
        raise RuntimeError(
            f"Registry-A hash mismatch: {actual} != {REGISTRY_A_SHA256}")
    obj = json.loads(REGISTRY_A.read_text(encoding="utf-8"))
    if obj.get("registry") != "conservative-finisher-mechanism-v1":
        raise RuntimeError("unexpected Registry-A identifier")
    targets = obj.get("targets", [])
    if len(targets) < 8:
        raise RuntimeError("Registry A must contain at least eight targets")
    _reject_prior_duplicates(targets)
    return obj


def build_target(spec: dict[str, Any]) -> Target1D:
    kind = str(spec["kind"])
    name = str(spec["name"])
    family = str(spec["family"])
    if kind == "student":
        scale = float(spec["scale"])
        return Target1D(
            name, family, kind, np.asarray([[0.0]]), np.asarray([1.0]),
            np.asarray([scale]), 5.0 * scale, df=float(spec["df"]))
    if kind == "shifted_lognormal":
        scale = float(spec["scale"])
        sigma = float(spec["log_sigma"])
        if scale <= 0.0 or sigma <= 0.0:
            raise ValueError(f"invalid shifted lognormal {name}")
        return Target1D(
            name, family, kind, np.asarray([[0.0]]), np.asarray([1.0]),
            np.asarray([scale]), 5.0 * scale, log_sigma=sigma)
    if kind != "mixture":
        raise ValueError(f"unknown target kind {kind}")
    if "means" in spec:
        means = np.asarray(spec["means"], dtype=float)[:, None]
    else:
        k = int(spec["K"])
        spacing = float(spec["spacing"])
        means = ((np.arange(k) - (k - 1) / 2) * spacing)[:, None]
    k = len(means)
    sigma_spec = spec["sigmas"]
    sigmas = (np.full(k, float(sigma_spec)) if np.isscalar(sigma_spec)
              else np.asarray(sigma_spec, dtype=float))
    weight_spec = spec["weights"]
    weights = (np.ones(k) if weight_spec == "equal"
               else np.asarray(weight_spec, dtype=float))
    if len(sigmas) != k or len(weights) != k:
        raise ValueError(f"malformed target {name}")
    if np.any(sigmas <= 0.0) or np.any(weights <= 0.0):
        raise ValueError(f"nonpositive target parameter in {name}")
    weights = weights / weights.sum()
    scale = float(np.max(np.abs(means[:, 0])) + 4 * np.max(sigmas))
    return Target1D(name, family, kind, means, weights, sigmas,
                    max(scale, 0.1))


def make_arms() -> tuple[Arm, ...]:
    arms = [
        Arm("paper-0.5", "paper", PAPER_TAU, "reused_masked",
            "continuous", standalone_paper=True),
        Arm("qld-v1", "paper", PAPER_TAU, "reused_masked", "carry"),
        Arm("qld-full", "qld", None, None, "carry"),
        Arm("qld-paper-reset", "paper", PAPER_TAU, "reused_masked",
            "reset"),
    ]
    for tau in TAUS:
        suffix = f"t{tau:g}"
        arms.extend([
            Arm(f"qld-mean-crossfit-{suffix}", "mean", tau, "crossfit",
                "reset"),
            Arm(f"qld-sharp-deleted-{suffix}", "sharp", tau,
                "reused_deleted", "reset"),
            Arm(f"qld-sharp-crossfit-{suffix}", "sharp", tau, "crossfit",
                "reset"),
            Arm(f"qld-kgrad-crossfit-{suffix}", "kgrad", tau, "crossfit",
                "reset"),
        ])
    return tuple(arms)


ARMS = make_arms()


def seed_base(master: int, target_index: int, init_index: int,
              seed: int) -> int:
    return int(master * 1_000_003 + target_index * 100_003 +
               init_index * 10_007 + seed * 101)


def clone_model(model: TanhMLP) -> TanhMLP:
    clone = copy.deepcopy(model)
    for name in PARAMETER_ORDER:
        if np.shares_memory(model.params[name], clone.params[name]):
            raise AssertionError("cloned parameters share mutable storage")
        if np.shares_memory(model.m[name], clone.m[name]) or \
                np.shares_memory(model.v[name], clone.v[name]):
            raise AssertionError("cloned optimizer buffers share storage")
    return clone


def capture_optimizer(model: TanhMLP) -> OptimizerState:
    return OptimizerState(
        step_index=int(model.step_index),
        m={name: model.m[name].copy() for name in PARAMETER_ORDER},
        v={name: model.v[name].copy() for name in PARAMETER_ORDER})


def restore_optimizer(model: TanhMLP, state: OptimizerState) -> None:
    model.step_index = int(state.step_index)
    model.m = {name: state.m[name].copy() for name in PARAMETER_ORDER}
    model.v = {name: state.v[name].copy() for name in PARAMETER_ORDER}


def reset_optimizer(model: TanhMLP) -> None:
    model.step_index = 0
    model.m = {name: np.zeros_like(model.params[name])
               for name in PARAMETER_ORDER}
    model.v = {name: np.zeros_like(model.params[name])
               for name in PARAMETER_ORDER}


def flatten_named(values: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([np.ravel(values[name]) for name in PARAMETER_ORDER])


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.ravel(left)
    right = np.ravel(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def _empty_work() -> dict[str, float]:
    return {
        "optimizer_updates": 0.0,
        "generator_forward_calls": 0.0,
        "generator_example_evals": 0.0,
        "unique_latent_samples": 0.0,
        "positive_target_samples": 0.0,
        "negative_reference_samples": 0.0,
        "kernel_pairs": 0.0,
        "sort_work": 0.0,
        "backward_examples": 0.0,
        "metric_generator_forward_calls": 0.0,
        "metric_generator_example_evals": 0.0,
        "metric_target_samples": 0.0,
        "diagnostic_generator_forward_calls": 0.0,
        "diagnostic_generator_example_evals": 0.0,
        "diagnostic_target_samples": 0.0,
    }


def _add_work(total: dict[str, float], update: dict[str, float]) -> None:
    for key, value in update.items():
        total[key] = total.get(key, 0.0) + float(value)


def _field_for_arm(model: TanhMLP, arm: Arm, x: np.ndarray,
                   positive: np.ndarray, negative_latent: np.ndarray | None) \
        -> tuple[np.ndarray, dict[str, float]]:
    n = len(x)
    if arm.finisher == "qld":
        raise AssertionError("QLD field needs the latent tie breaker")
    if arm.finisher == "paper":
        result = compute_paper_field(
            x, positive, tau=float(arm.tau), gain="paper", mask=True,
            on_degenerate="zero")
        return result.V, {
            "kernel_pairs": float(result.kernel_pairs),
            "denominator_min": float("nan"),
            "floor_activations": 0.0,
            "negative_terms": float(n - 1),
            "extra_forwards": 0.0,
            "negative_samples": float(n),
        }
    negative: np.ndarray | None = None
    extra_forwards = 0
    negative_samples = n
    if arm.reference_mode == "crossfit":
        if negative_latent is None:
            raise ValueError("crossfit arm is missing negative latent input")
        negative = model.forward(negative_latent)
        extra_forwards = 1
    result: FieldResult = conservative_field(
        arm.finisher, x, positive, negative, tau=float(arm.tau),
        reference_mode=arm.reference_mode)  # type: ignore[arg-type]
    diagnostics = result.diagnostics
    return result.field, {
        "kernel_pairs": float(diagnostics.kernel_pairs),
        "denominator_min": float(min(
            diagnostics.positive_denominator_min,
            diagnostics.negative_denominator_min)),
        "floor_activations": float(diagnostics.floor_activations),
        "negative_terms": float(diagnostics.negative_terms_per_anchor),
        "extra_forwards": float(extra_forwards),
        "negative_samples": float(negative_samples),
    }


def apply_update(model: TanhMLP, arm: Arm, latent: np.ndarray,
                 positive: np.ndarray, negative_latent: np.ndarray | None) \
        -> dict[str, float]:
    """Apply one stop-gradient update and expose raw/effective directions."""
    x, cache = model.forward(latent, want_cache=True)
    if arm.finisher == "qld":
        field = exact_rank_field(x, positive, latent)
        detail = {
            "kernel_pairs": 0.0,
            "denominator_min": float("nan"),
            "floor_activations": 0.0,
            "negative_terms": 0.0,
            "extra_forwards": 0.0,
            "negative_samples": 0.0,
        }
        sort_work = 2.0 * len(x) * math.log2(max(len(x), 2))
    else:
        field, detail = _field_for_arm(
            model, arm, x, positive, negative_latent)
        sort_work = 0.0
    gradients = _stopgrad_grads(model, cache, field, len(field))
    before = {name: model.params[name].copy() for name in PARAMETER_ORDER}
    _adam_step(model, gradients)
    delta = {name: model.params[name] - before[name]
             for name in PARAMETER_ORDER}
    flat_gradient = flatten_named(gradients)
    flat_delta = flatten_named(delta)
    n = len(latent)
    return {
        "field_rms": float(np.sqrt(np.mean(field * field))),
        "field_max": float(np.max(np.linalg.norm(field, axis=1))),
        "parameter_gradient_norm": float(np.linalg.norm(flat_gradient)),
        "effective_step_norm": float(np.linalg.norm(flat_delta)),
        "descent_step_cosine": cosine(-flat_gradient, flat_delta),
        "denominator_min": detail["denominator_min"],
        "floor_activations": detail["floor_activations"],
        "negative_terms": detail["negative_terms"],
        "optimizer_updates": 1.0,
        "generator_forward_calls": 1.0 + detail["extra_forwards"],
        "generator_example_evals": n * (1.0 + detail["extra_forwards"]),
        "unique_latent_samples": n * (1.0 + detail["extra_forwards"]),
        "positive_target_samples": float(n),
        "negative_reference_samples": detail["negative_samples"],
        "kernel_pairs": detail["kernel_pairs"],
        "sort_work": sort_work,
        "backward_examples": float(n),
    }


def independent_gradient_cosine(
        model: TanhMLP, arm: Arm, latent: np.ndarray,
        positive: np.ndarray, negative_latent: np.ndarray) -> float:
    """Compare the arm and QLD gradients on an independent diagnostic batch."""
    x, cache = model.forward(latent, want_cache=True)
    qld = exact_rank_field(x, positive, latent)
    qld_gradient = _stopgrad_grads(model, cache, qld, len(qld))
    if arm.finisher == "qld":
        arm_gradient = qld_gradient
    else:
        field, _ = _field_for_arm(
            model, arm, x, positive, negative_latent)
        arm_gradient = _stopgrad_grads(model, cache, field, len(field))
    return cosine(flatten_named(arm_gradient), flatten_named(qld_gradient))


def target_diagnostics(q: np.ndarray, target: Target1D,
                       minimum_mass: float) -> dict[str, float]:
    if not target.has_components:
        return {
            "mass_l1": float("nan"),
            "minimum_component_occupancy": float("nan"),
            "all_reportable_components_reached": float("nan"),
        }
    distances = np.abs(q[:, 0, None] - target.means[:, 0][None, :])
    nearest = np.argmin(distances, axis=1)
    observed = np.bincount(nearest, minlength=len(target.means)) / len(q)
    reportable = target.weights >= minimum_mass - 1e-15
    return {
        "mass_l1": float(np.sum(np.abs(observed - target.weights))),
        "minimum_component_occupancy": float(np.min(observed[reportable])),
        "all_reportable_components_reached": float(
            np.all(observed[reportable] > 0.0)),
    }


def quantile_cdf_error(q: np.ndarray, target_reference: np.ndarray,
                       bins: int = 32) -> float:
    probabilities = np.arange(1, bins) / bins
    boundaries = np.quantile(target_reference[:, 0], probabilities)
    empirical = np.asarray([(q[:, 0] <= boundary).mean()
                            for boundary in boundaries])
    return float(np.mean(np.abs(empirical - probabilities)))


def evaluate_model(model: TanhMLP, target: Target1D,
                   evaluation_latent: np.ndarray,
                   target_reference: np.ndarray, ed_indices: np.ndarray,
                   metric_seed: int, minimum_mass: float) \
        -> dict[str, float]:
    q = model.forward(evaluation_latent)
    ed = max(0.0, float(energy_distance2(
        q[ed_indices], target_reference[ed_indices])))
    sw = float(sliced_w1(
        q, target_reference, 1, np.random.default_rng(metric_seed)))
    diagnostics = target_diagnostics(q, target, minimum_mass)
    quantiles = np.quantile(q[:, 0], [0.01, 0.1, 0.5, 0.9, 0.99])
    return {
        "ed2": ed,
        "sw1": sw,
        "quantile_cdf_error": quantile_cdf_error(q, target_reference),
        "output_mean": float(np.mean(q)),
        "output_std": float(np.std(q)),
        "output_q01": float(quantiles[0]),
        "output_q10": float(quantiles[1]),
        "output_q50": float(quantiles[2]),
        "output_q90": float(quantiles[3]),
        "output_q99": float(quantiles[4]),
        **diagnostics,
    }


def _trajectory_steps(profile: Profile, warm_steps: int) -> set[int]:
    steps = {0, warm_steps, profile.steps}
    steps.update(range(profile.trajectory_period, profile.steps + 1,
                       profile.trajectory_period))
    steps.update(range(warm_steps + 1,
                       min(profile.steps, warm_steps + 5) + 1))
    return steps


def _trajectory_row(
        arm: Arm, step: int, phase: str, model: TanhMLP,
        target: Target1D, evaluation_latent: np.ndarray,
        target_reference: np.ndarray, ed_indices: np.ndarray,
        metric_seed: int, minimum_mass: float,
        latest: dict[str, float] | None, work: dict[str, float],
        gradient_cosine: float, wall_seconds: float) -> dict[str, Any]:
    work["metric_generator_forward_calls"] += 1.0
    work["metric_generator_example_evals"] += float(len(evaluation_latent))
    work["metric_target_samples"] += float(len(target_reference))
    metrics = evaluate_model(
        model, target, evaluation_latent, target_reference, ed_indices,
        metric_seed, minimum_mass)
    defaults = {
        "field_rms": float("nan"),
        "field_max": float("nan"),
        "parameter_gradient_norm": float("nan"),
        "effective_step_norm": float("nan"),
        "descent_step_cosine": float("nan"),
        "denominator_min": float("nan"),
        "floor_activations": 0.0,
        "negative_terms": 0.0,
    }
    if latest is not None:
        defaults.update({key: latest[key] for key in defaults})
    return {
        "arm": arm.label,
        "step": step,
        "phase": phase,
        **metrics,
        **defaults,
        "finisher_qld_gradient_cosine": gradient_cosine,
        **{f"work_{key}": value for key, value in work.items()},
        "wall_seconds": wall_seconds,
    }


def _arrays_equal(left: TanhMLP, right: TanhMLP) -> bool:
    return (left.step_index == right.step_index and all(
        np.array_equal(left.params[name], right.params[name]) and
        np.array_equal(left.m[name], right.m[name]) and
        np.array_equal(left.v[name], right.v[name])
        for name in PARAMETER_ORDER))


def regression_invariant_tests(log=print) -> dict[str, Any]:
    """Check optimizer, clone, baseline, and parameter-gradient contracts."""
    checks: list[str] = []
    spec = {
        "name": "invariant-target", "family": "invariant",
        "kind": "mixture", "means": [-1.0, 0.7],
        "sigmas": [0.1, 0.2], "weights": [0.4, 0.6],
    }
    target = build_target(spec)
    rng = np.random.default_rng(20260722)
    latent = rng.normal(size=(11, 2))
    positive = target.sample(11, rng)

    direct = TanhMLP(target, "missing", 17)
    manual = clone_model(direct)
    x, cache = direct.forward(latent, want_cache=True)
    qld_field = exact_rank_field(x, positive, latent)
    direct.stopgrad_step(cache, qld_field)
    qld_arm = Arm("qld", "qld", None, None, "carry")
    apply_update(manual, qld_arm, latent, positive, None)
    if not _arrays_equal(direct, manual):
        raise AssertionError("QLD update no longer matches TanhMLP baseline")
    checks.append("qld_update_bitwise_compatible")

    carry = clone_model(direct)
    uninterrupted = clone_model(direct)
    apply_update(carry, qld_arm, latent, positive, None)
    apply_update(uninterrupted, qld_arm, latent, positive, None)
    if not _arrays_equal(carry, uninterrupted):
        raise AssertionError("carry state changed the next Adam update")
    checks.append("carry_matches_uninterrupted")

    reset = clone_model(direct)
    reset_optimizer(reset)
    if reset.step_index != 0 or any(
            np.any(reset.m[name]) or np.any(reset.v[name])
            for name in PARAMETER_ORDER):
        raise AssertionError("optimizer reset left nonzero state")
    checks.append("reset_is_zero")

    qld_state = capture_optimizer(direct)
    dual = clone_model(direct)
    reset_optimizer(dual)
    finisher_state = capture_optimizer(dual)
    restore_optimizer(dual, qld_state)
    if dual.step_index != direct.step_index:
        raise AssertionError("dual QLD optimizer state failed to restore")
    restore_optimizer(dual, finisher_state)
    if dual.step_index != 0:
        raise AssertionError("dual finisher optimizer state failed to restore")
    checks.append("dual_states_roundtrip")

    prefix = clone_model(direct)
    branch_a = clone_model(prefix)
    branch_b = clone_model(prefix)
    apply_update(branch_a, qld_arm, latent, positive, None)
    if not _arrays_equal(prefix, branch_b):
        raise AssertionError("branch mutation leaked into sibling or prefix")
    checks.append("branch_mutation_isolated")

    # Parameter-gradient check for the crossfit sharp scalar loss.  Negative
    # outputs are fixed to preserve the declared stop-gradient convention.
    gradient_model = TanhMLP(target, "concentrated", 23)
    zq = rng.normal(size=(7, gradient_model.latent_dim))
    zn = rng.normal(size=(8, gradient_model.latent_dim))
    q, cache = gradient_model.forward(zq, want_cache=True)
    negative = gradient_model.forward(zn).copy()
    positive_small = target.sample(9, rng)
    sharp = conservative_field(
        "sharp", q, positive_small, negative, tau=0.7,
        reference_mode="crossfit").field
    analytical = _stopgrad_grads(gradient_model, cache, sharp, len(sharp))
    flat_analytical = flatten_named(analytical)
    offsets: list[tuple[str, tuple[int, ...], int]] = []
    start = 0
    for name in PARAMETER_ORDER:
        for index in np.ndindex(gradient_model.params[name].shape):
            offsets.append((name, index, start))
            start += 1
    selected = np.linspace(0, len(offsets) - 1, 24, dtype=int)
    eps = 2e-6
    for selected_index in selected:
        name, index, flat_index = offsets[int(selected_index)]
        original = float(gradient_model.params[name][index])
        gradient_model.params[name][index] = original + eps
        plus = sharp_logkde_loss(
            gradient_model.forward(zq), positive_small, negative,
            tau=0.7, reference_mode="crossfit")
        gradient_model.params[name][index] = original - eps
        minus = sharp_logkde_loss(
            gradient_model.forward(zq), positive_small, negative,
            tau=0.7, reference_mode="crossfit")
        gradient_model.params[name][index] = original
        numerical = (plus - minus) / (2 * eps)
        np.testing.assert_allclose(
            numerical, flat_analytical[flat_index], rtol=4e-4, atol=3e-7)
    checks.append("sharp_parameter_gradient_matches_loss")

    # The exact paper implementation retains its own independent invariants.
    paper_field_invariant_tests(log=lambda _: None)
    paper_port_invariants(log=lambda _: None)
    checks.append("paper_baseline_invariants")

    report = {"status": "pass", "check_count": len(checks),
              "checks": checks}
    log(f"training/regression invariants: PASS ({len(checks)} checks)")
    return report


def _run_branch(
        arm: Arm, start_model: TanhMLP, start_step: int,
        profile: Profile, target: Target1D, latent_batches: np.ndarray,
        positive_batches: np.ndarray, negative_latent_batches: np.ndarray,
        evaluation_latent: np.ndarray, target_reference: np.ndarray,
        ed_indices: np.ndarray, diagnostic_latent: np.ndarray,
        diagnostic_positive: np.ndarray, diagnostic_negative: np.ndarray,
        metric_seed: int, minimum_mass: float,
        initial_metrics: dict[str, float],
        prefix_trajectories: list[dict[str, Any]],
        prefix_work: dict[str, float], wall0: float) \
        -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    model = clone_model(start_model)
    if arm.handoff == "reset":
        reset_optimizer(model)
    work = dict(prefix_work)
    trajectory = [dict(row, arm=arm.label) for row in prefix_trajectories]
    trajectory_steps = _trajectory_steps(profile,
                                         int(round(PREFIX_FRACTION *
                                                   profile.steps)))
    label_event: int | None = None
    occupancy_event: int | None = None
    best_ed2 = float("inf")
    best_step = start_step
    best_before_handoff = min(
        (float(row["ed2"]) for row in trajectory), default=float("nan"))
    handoff_metrics = trajectory[-1] if trajectory else None
    floor_total = 0
    diverged = False
    first_five_ed2: list[float] = []

    for step in range(start_step + 1, profile.steps + 1):
        latest = apply_update(
            model, arm, latent_batches[step - 1],
            positive_batches[step - 1], negative_latent_batches[step - 1])
        _add_work(work, {key: latest[key] for key in _empty_work()
                         if key in latest})
        floor_total += int(latest["floor_activations"])
        if not model.finite():
            diverged = True
            break
        if step in trajectory_steps:
            gradient_cos = independent_gradient_cosine(
                model, arm, diagnostic_latent,
                diagnostic_positive, diagnostic_negative)
            diagnostic_forwards = 1.0 + float(
                arm.reference_mode == "crossfit")
            work["diagnostic_generator_forward_calls"] += \
                diagnostic_forwards
            work["diagnostic_generator_example_evals"] += \
                diagnostic_forwards * len(diagnostic_latent)
            work["diagnostic_target_samples"] += float(
                len(diagnostic_positive))
            row = _trajectory_row(
                arm, step,
                "prefix" if step <= start_step else "finisher",
                model, target, evaluation_latent, target_reference,
                ed_indices, metric_seed, minimum_mass, latest, work,
                gradient_cos, time.perf_counter() - wall0)
            trajectory.append(row)
            ed2 = float(row["ed2"])
            if ed2 < best_ed2:
                best_ed2 = ed2
                best_step = step
            if step <= start_step + 5:
                first_five_ed2.append(ed2)
            if (label_event is None and
                    ed2 <= 0.25 * initial_metrics["ed2"] and
                    float(row["sw1"]) <= 0.25 * initial_metrics["sw1"]):
                label_event = step
            reached = row["all_reportable_components_reached"]
            if (occupancy_event is None and target.has_components and
                    reached == 1.0 and float(row["mass_l1"]) <= 0.25):
                occupancy_event = step

    if diverged:
        endpoint = {key: float("inf") for key in (
            "ed2", "sw1", "quantile_cdf_error")}
        endpoint.update({
            "mass_l1": float("nan"),
            "minimum_component_occupancy": float("nan"),
            "all_reportable_components_reached": float("nan"),
        })
    else:
        endpoint = evaluate_model(
            model, target, evaluation_latent, target_reference, ed_indices,
            metric_seed, minimum_mass)
        best_ed2 = min(best_ed2, endpoint["ed2"])
    five_step_change = float("nan")
    if handoff_metrics is not None and first_five_ed2:
        five_step_change = first_five_ed2[-1] - float(handoff_metrics["ed2"])
    row = {
        "arm": arm.label,
        "finisher": arm.finisher,
        "tau": arm.tau,
        "reference_mode": arm.reference_mode,
        "handoff": arm.handoff,
        **endpoint,
        "diverged": int(diverged),
        "floor_activations": floor_total,
        "label_free_event_time": (label_event if label_event is not None
                                  else profile.steps),
        "label_free_event_censored": int(label_event is None),
        "occupancy_event_time": (occupancy_event if occupancy_event is not None
                                 else profile.steps),
        "occupancy_event_censored": int(occupancy_event is None),
        "best_ed2": best_ed2,
        "best_ed2_step": best_step,
        "best_before_handoff_ed2": best_before_handoff,
        "handoff_ed2": (float(handoff_metrics["ed2"])
                        if handoff_metrics is not None else float("nan")),
        "five_step_ed2_change": five_step_change,
        "wall_seconds": time.perf_counter() - wall0,
    }
    return row, trajectory, work


def run_trial_group(task: dict[str, Any]) \
        -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    target = build_target(task["target_spec"])
    profile = Profile(**task["profile"])
    init_kind = str(task["init"])
    seed = int(task["seed"])
    base = int(task["base_seed"])
    minimum_mass = float(task["minimum_mass"])
    wall0 = time.perf_counter()
    warm_steps = int(round(PREFIX_FRACTION * profile.steps))

    template = TanhMLP(target, init_kind, base + 1)
    latent_rng = np.random.default_rng(base + 2)
    data_rng = np.random.default_rng(base + 3)
    negative_rng = np.random.default_rng(base + 4)
    latent_batches = latent_rng.normal(
        size=(profile.steps, profile.batch, template.latent_dim))
    positive_batches = np.asarray([
        target.sample(profile.batch, data_rng) for _ in range(profile.steps)])
    negative_latent_batches = negative_rng.normal(
        size=(profile.steps, profile.batch, template.latent_dim))

    evaluation_latent = np.random.default_rng(base + 5).normal(
        size=(profile.eval_size, template.latent_dim))
    target_reference = target.sample(
        profile.eval_size, np.random.default_rng(base + 6))
    ed_indices = np.random.default_rng(base + 7).choice(
        profile.eval_size, size=profile.ed_size, replace=False)
    diagnostic_latent = np.random.default_rng(base + 8).normal(
        size=(profile.batch, template.latent_dim))
    diagnostic_positive = target.sample(
        profile.batch, np.random.default_rng(base + 9))
    diagnostic_negative = np.random.default_rng(base + 10).normal(
        size=(profile.batch, template.latent_dim))
    metric_seed = base + 11

    initial_metrics = evaluate_model(
        template, target, evaluation_latent, target_reference, ed_indices,
        metric_seed, minimum_mass)
    initial_work = _empty_work()

    # Train the shared QLD prefix once.
    prefix = clone_model(template)
    prefix_work = _empty_work()
    prefix_rows: list[dict[str, Any]] = []
    qld_arm = next(arm for arm in ARMS if arm.label == "qld-full")
    prefix_steps = _trajectory_steps(profile, warm_steps)
    prefix_rows.append(_trajectory_row(
        qld_arm, 0, "initial", prefix, target, evaluation_latent,
        target_reference, ed_indices, metric_seed, minimum_mass, None,
        prefix_work, float("nan"), time.perf_counter() - wall0))
    for step in range(1, warm_steps + 1):
        latest = apply_update(
            prefix, qld_arm, latent_batches[step - 1],
            positive_batches[step - 1], None)
        _add_work(prefix_work, {key: latest[key] for key in _empty_work()
                                if key in latest})
        if step in prefix_steps:
            prefix_rows.append(_trajectory_row(
                qld_arm, step, "prefix", prefix, target,
                evaluation_latent, target_reference, ed_indices, metric_seed,
                minimum_mass, latest, prefix_work, 1.0,
                time.perf_counter() - wall0))

    rows: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    ledgers: list[dict[str, Any]] = []
    for arm in ARMS:
        if arm.standalone_paper:
            start_model = template
            start_step = 0
            arm_prefix_rows = [prefix_rows[0]]
            arm_prefix_work = initial_work
        else:
            start_model = prefix
            start_step = warm_steps
            arm_prefix_rows = prefix_rows
            arm_prefix_work = prefix_work
        row, arm_trajectory, work = _run_branch(
            arm, start_model, start_step, profile, target, latent_batches,
            positive_batches, negative_latent_batches, evaluation_latent,
            target_reference, ed_indices, diagnostic_latent,
            diagnostic_positive, diagnostic_negative, metric_seed,
            minimum_mass, initial_metrics, arm_prefix_rows,
            arm_prefix_work, wall0)
        metadata = {
            "target": target.name,
            "family": target.family,
            "init": init_kind,
            "seed": seed,
            "cell": f"{target.name}/{init_kind}",
        }
        rows.append({**metadata, **row})
        trajectories.extend([{**metadata, **item}
                             for item in arm_trajectory])
        ledgers.append({**metadata, "arm": arm.label, **work})
    return rows, trajectories, ledgers


def make_tasks(registry: dict[str, Any], profile: Profile,
               include_stress: bool) -> list[dict[str, Any]]:
    inits = PRIMARY_INITS + (STRESS_INITS if include_stress else ())
    tasks: list[dict[str, Any]] = []
    master = int(registry["master_seed"])
    for target_index, target_spec in enumerate(registry["targets"]):
        for init_index, init_kind in enumerate(inits):
            for seed in range(profile.seeds):
                tasks.append({
                    "target_spec": target_spec,
                    "profile": asdict(profile),
                    "init": init_kind,
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
            rows.extend(r)
            trajectories.extend(t)
            ledgers.extend(w)
            print(f"completed trial group {index}/{len(tasks)}")
        return rows, trajectories, ledgers
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_trial_group, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            r, t, w = future.result()
            rows.extend(r)
            trajectories.extend(t)
            ledgers.extend(w)
            print(f"completed trial group {index}/{len(tasks)}")
    return rows, trajectories, ledgers


def _positive_median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values
              if math.isfinite(float(value)) and float(value) >= 0.0]
    if not finite:
        return float("inf")
    return float(np.median(finite))


def _geomean(values: Iterable[float]) -> float:
    values = [max(float(value), 1e-300) for value in values]
    return float(math.exp(np.mean(np.log(values))))


def summarize(rows: list[dict[str, Any]], profile: Profile) -> dict[str, Any]:
    primary = [row for row in rows if row["init"] in PRIMARY_INITS]
    arms = sorted({str(row["arm"]) for row in primary})
    cells = sorted({str(row["cell"]) for row in primary})
    lookup: dict[tuple[str, str], float] = {}
    sw_lookup: dict[tuple[str, str], float] = {}
    for cell in cells:
        for arm in arms:
            selected = [row for row in primary
                        if row["cell"] == cell and row["arm"] == arm]
            lookup[(cell, arm)] = _positive_median(
                row["ed2"] for row in selected)
            sw_lookup[(cell, arm)] = _positive_median(
                row["sw1"] for row in selected)
    ratios: dict[str, float] = {}
    sw_ratios: dict[str, float] = {}
    for arm in arms:
        if arm == "qld-v1":
            ratios[arm] = 1.0
            sw_ratios[arm] = 1.0
            continue
        ratios[arm] = _geomean(
            lookup[(cell, arm)] / max(lookup[(cell, "qld-v1")], 1e-300)
            for cell in cells)
        sw_ratios[arm] = _geomean(
            sw_lookup[(cell, arm)] /
            max(sw_lookup[(cell, "qld-v1")], 1e-300)
            for cell in cells)
    eligible = [arm for arm in arms if arm.startswith("qld-") and
                arm not in ("qld-v1", "qld-full")]
    safe = [arm for arm in eligible if all(
        int(row["diverged"]) == 0 and int(row["floor_activations"]) == 0
        for row in primary if row["arm"] == arm)]
    ranked = sorted(safe, key=lambda arm: (ratios[arm], sw_ratios[arm], arm))
    return {
        "profile": profile.name,
        "trial_rows": len(rows),
        "primary_cells": len(cells),
        "ed2_ratios_vs_qld": ratios,
        "sw1_ratios_vs_qld": sw_ratios,
        "safe_candidate_ranking": ranked,
        "best_safe_candidate": ranked[0] if ranked else None,
        "divergence_counts": {
            arm: sum(int(row["diverged"]) for row in primary
                     if row["arm"] == arm)
            for arm in arms
        },
        "floor_activation_counts": {
            arm: sum(int(row["floor_activations"]) for row in primary
                     if row["arm"] == arm)
            for arm in arms
        },
        "mechanism_questions": {
            "paper_reset_ed2_ratio_vs_qld_v1": ratios.get(
                "qld-paper-reset"),
            "full_qld_ed2_ratio_vs_qld_v1": ratios.get("qld-full"),
            "primary_sharp_crossfit_t05_ed2_ratio_vs_qld_v1": ratios.get(
                "qld-sharp-crossfit-t0.5"),
        },
        "status": "mechanism screen only; no candidate promoted",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_run_dir(profile: Profile) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNROOT / f"{timestamp}-S2-{profile.name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_results(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Conservative finisher S2 ({summary['profile']})",
        "",
        "This is a mechanism screen, not a confirmation or superiority claim.",
        "",
        f"Primary cells: `{summary['primary_cells']}`",
        f"Best safe diagnostic arm: `{summary['best_safe_candidate']}`",
        "",
        "## Target-balanced ED2 ratios versus QLD-v1",
        "",
        "| arm | ratio | SW1 ratio | divergences | floor activations |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, ratio in sorted(summary["ed2_ratios_vs_qld"].items()):
        lines.append(
            f"| `{arm}` | {ratio:.6g} | "
            f"{summary['sw1_ratios_vs_qld'][arm]:.6g} | "
            f"{summary['divergence_counts'][arm]} | "
            f"{summary['floor_activation_counts'][arm]} |")
    lines.extend([
        "",
        "The ordering above is descriptive. S3 must apply the frozen",
        "lexicographic rule before any Registry-B run.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def source_files() -> tuple[Path, ...]:
    return (
        Path(__file__),
        HERE / "conservative_finishers.py",
        REGISTRY_A,
        PLAN,
        PROTOCOL,
        HERE / "identifiability_drift.py",
        HERE / "lbqcd.py",
        HERE / "lowdim_drift.py",
        HERE / "run_identifiability_generator.py",
    )


def write_artifacts(
        run_dir: Path, args: argparse.Namespace, profile: Profile,
        invariants: dict[str, Any], rows: list[dict[str, Any]],
        trajectories: list[dict[str, Any]], ledgers: list[dict[str, Any]],
        summary: dict[str, Any]) -> None:
    write_csv(run_dir / "rows.csv", rows)
    write_csv(run_dir / "trajectories.csv", trajectories)
    (run_dir / "work_ledger.json").write_text(
        json.dumps(ledgers, indent=2, allow_nan=True), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    (run_dir / "invariants.json").write_text(
        json.dumps(invariants, indent=2, allow_nan=True), encoding="utf-8")
    hashes = {path.name: sha256_file(path) for path in source_files()}
    (run_dir / "source_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8")
    snapshot_dir = run_dir / "source_snapshots"
    snapshot_dir.mkdir()
    for path in source_files():
        shutil.copy2(path, snapshot_dir / path.name)
    manifest = {
        "stage": "S2",
        "profile": asdict(profile),
        "arguments": vars(args),
        "registry_sha256": sha256_file(REGISTRY_A),
        "plan_sha256": sha256_file(PLAN),
        "git_head": git_text("rev-parse", "HEAD"),
        "git_status": git_text("status", "--short"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "arms": [asdict(arm) for arm in ARMS],
        "physical_common_prefix_sharing": True,
        "standalone_cost_reported_per_arm": True,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    write_results(run_dir / "RESULTS.md", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("S1", "S2"), nargs="?",
                        default="S1")
    parser.add_argument("--profile", choices=tuple(PROFILES),
                        default="smoke")
    parser.add_argument("--workers", type=int,
                        default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--include-stress", action="store_true")
    args = parser.parse_args()

    registry = load_registry_a()
    field_report = conservative_invariant_tests()
    regression_report = regression_invariant_tests()
    invariants = {"field": field_report, "regression": regression_report}
    if args.stage == "S1":
        print(json.dumps(invariants, indent=2))
        return

    profile = PROFILES[args.profile]
    tasks = make_tasks(registry, profile, args.include_stress)
    run_dir = make_run_dir(profile)
    print(f"run directory: {run_dir}")
    rows, trajectories, ledgers = execute(tasks, args.workers)
    summary = summarize(rows, profile)
    write_artifacts(
        run_dir, args, profile, invariants, rows, trajectories, ledgers,
        summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
