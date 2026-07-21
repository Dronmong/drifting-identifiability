"""Development runner for Large-Batch Quantile-Calibrated Drifting.

N1 tests whether virtual-large-batch Run-Sort-ReRun repairs QLD's
unequal-weight weakness.  N2 adds a held-out, rank-alignment selector for the
paper refinement bandwidth.  This is deliberately a mutable development
runner; any eventual claim requires a new frozen registry and protocol.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
from datetime import datetime
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

from lbqcd import (  # noqa: E402
    diagnose_quantile_resolution_stable,
    direct_quantile_step,
    example_matched_updates,
    invariant_tests,
    noise_restored_rsr_step,
    paper_step,
    pulse_example_matched_updates,
    rsr_quantile_step,
    select_tau_by_alignment,
    select_tau_by_model_lookahead,
)
from lowdim_drift import energy_distance2, sliced_w1  # noqa: E402
from run_identifiability_generator import TanhMLP  # noqa: E402
from run_qld_confirmatory import (  # noqa: E402
    Target1D,
    build_target,
    sha256_file,
    target_diagnostics,
)


REGISTRY = HERE / "lbqcd_development_registry.json"
RESEARCH = HERE / "QLDNextGenerationResearch.md"
RUNROOT = HERE / "lbqcd_runs"
PAPER_TAUS = (0.2, 0.5, 1.0, 2.0, 4.0)
SELECTED_PAPER_TAU = 0.5


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    batch: int
    seeds: int
    eval_size: int
    ed_size: int
    selector_probe: int


PROFILES = {
    "smoke": Profile("smoke", 30, 32, 1, 512, 256, 32),
    "screen": Profile("screen", 400, 128, 3, 2048, 512, 128),
    "standard": Profile("standard", 1200, 128, 8, 4096, 1024, 256),
}


@dataclass(frozen=True)
class Arm:
    label: str
    kind: str
    tau: float
    virtual_batch: int
    warm_fraction: float
    budget: str
    selector_period: int = 50
    pulse_period: int = 0
    noise_mix: float = 0.0


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _parse_numbers(text: str, cast) -> tuple:
    return tuple(cast(item.strip()) for item in text.split(",")
                 if item.strip())


def load_registry() -> dict[str, Any]:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("registry") != "LBQCD-development-v1":
        raise RuntimeError("unexpected LB-QCD development registry")
    targets = registry.get("targets", [])
    if len(targets) != 12:
        raise RuntimeError("development registry must contain 12 targets")
    names = [str(spec["name"]) for spec in targets]
    if len(set(names)) != len(names):
        raise RuntimeError("development target names are not unique")
    # Guard against accidental reuse of the sealed QLD target identifiers.
    if any(name.startswith("QLD-") for name in names):
        raise RuntimeError("development registry reuses a sealed target name")
    for spec in targets:
        build_target(spec)
    return registry


def seed_base(master: int, target_index: int, init_index: int,
              seed: int) -> int:
    return int(master * 1_000_003 + target_index * 100_003 +
               init_index * 10_007 + seed * 101)


def make_arms(stage: str, virtual_batches: tuple[int, ...],
              warm_fractions: tuple[float, ...], budgets: tuple[str, ...],
              selector_period: int,
              pulse_periods: tuple[int, ...],
              noise_mixes: tuple[float, ...],
              include_gated: bool,
              include_lookahead: bool) -> list[Arm]:
    arms = [Arm(f"paper-{tau:g}", "paper", tau, 0, 0.0, "update")
            for tau in PAPER_TAUS]
    arms.append(Arm("qld-v1", "qld", SELECTED_PAPER_TAU, 0, 0.70,
                    "update"))
    for virtual_batch in virtual_batches:
        for warm_fraction in warm_fractions:
            tag = f"M{virtual_batch}-f{warm_fraction:.2f}"
            for budget in budgets:
                suffix = "u" if budget == "update" else "e"
                arms.append(Arm(
                    f"rsr-{tag}-{suffix}", "rsr", SELECTED_PAPER_TAU,
                    virtual_batch, warm_fraction, budget, selector_period))
                for pulse_period in pulse_periods:
                    arms.append(Arm(
                        f"pulse-{tag}-p{pulse_period}-{suffix}", "pulse",
                        SELECTED_PAPER_TAU, virtual_batch, warm_fraction,
                        budget, selector_period, pulse_period))
                for noise_mix in noise_mixes:
                    arms.append(Arm(
                        f"nrsr-{tag}-l{noise_mix:.2f}-{suffix}", "nrsr",
                        SELECTED_PAPER_TAU, virtual_batch, warm_fraction,
                        budget, selector_period, 0, noise_mix))
                if include_gated:
                    arms.append(Arm(
                        f"gated-{tag}-{suffix}", "gated",
                        SELECTED_PAPER_TAU, virtual_batch, warm_fraction,
                        budget, selector_period))
            if stage == "n2":
                arms.append(Arm(
                    f"lbqcd-align-{tag}-u", "align", SELECTED_PAPER_TAU,
                    virtual_batch, warm_fraction, "update", selector_period))
                if include_gated:
                    arms.append(Arm(
                        f"gated-align-{tag}-u", "gated-align",
                        SELECTED_PAPER_TAU, virtual_batch, warm_fraction,
                        "update", selector_period))
                if include_lookahead:
                    arms.append(Arm(
                        f"lbqcd-lookahead-{tag}-u", "lookahead",
                        SELECTED_PAPER_TAU, virtual_batch, warm_fraction,
                        "update", selector_period))
                    if include_gated:
                        arms.append(Arm(
                            f"gated-lookahead-{tag}-u", "gated-lookahead",
                            SELECTED_PAPER_TAU, virtual_batch, warm_fraction,
                            "update", selector_period))
    return arms


def _trial_steps(arm: Arm, profile: Profile) -> int:
    if arm.kind == "pulse" and arm.budget == "examples":
        return pulse_example_matched_updates(
            profile.steps, profile.batch, arm.virtual_batch,
            arm.warm_fraction, arm.pulse_period)
    if arm.kind in ("rsr", "align", "nrsr", "gated", "gated-align",
                    "lookahead", "gated-lookahead") and arm.budget == "examples":
        return example_matched_updates(
            profile.steps, profile.batch, arm.virtual_batch,
            arm.warm_fraction)
    return profile.steps


def _add_work(total: dict[str, float], work) -> None:
    for key in ("optimizer_updates", "generator_forward_calls",
                "generator_example_evals", "unique_latent_samples",
                "target_samples", "kernel_pairs", "sort_work"):
        total[key] += float(getattr(work, key))


def run_trial(task: dict[str, Any]) -> dict[str, Any]:
    target: Target1D = build_target(task["target_spec"])
    profile = Profile(**task["profile"])
    arm = Arm(**task["arm"])
    base = int(task["base_seed"])
    seed = int(task["seed"])
    init_kind = str(task["init"])
    candidate_taus = tuple(float(x) for x in task["candidate_taus"])
    total_steps = _trial_steps(arm, profile)
    warm_steps = int(round(arm.warm_fraction * total_steps))
    wall0 = time.perf_counter()

    model = TanhMLP(target, init_kind, base + 1)
    latent_rng = np.random.default_rng(base + 2)
    data_rng = np.random.default_rng(base + 3)
    probe_latent_rng = np.random.default_rng(base + 31)
    probe_data_rng = np.random.default_rng(base + 32)
    gradient_rng = np.random.default_rng(base + 33)
    diagnostic_rng = np.random.default_rng(base + 34)
    diagnosis = None
    diagnostic_target_samples = 0
    if arm.kind in ("gated", "gated-align", "gated-lookahead"):
        diagnostic_target_samples = int(task["diagnostic_samples"])
        diagnostic_sample = target.sample(
            diagnostic_target_samples, diagnostic_rng)
        diagnosis = diagnose_quantile_resolution_stable(
            diagnostic_sample, profile.batch)
    current_tau = arm.tau
    tau_counts = {str(tau): 0 for tau in candidate_taus}
    selector_calls = 0
    selector_kernel_pairs = 0
    selector_example_evals = 0
    work = {key: 0.0 for key in (
        "optimizer_updates", "generator_forward_calls",
        "generator_example_evals", "unique_latent_samples",
        "target_samples", "kernel_pairs", "sort_work")}
    diverged = False
    event_time: int | None = None

    for step in range(1, total_steps + 1):
        if arm.kind in ("qld", "rsr", "align", "pulse", "nrsr", "gated",
                        "gated-align", "lookahead", "gated-lookahead") and step <= warm_steps:
            use_rsr = (arm.kind in ("rsr", "align", "nrsr", "lookahead") or
                       (arm.kind == "pulse" and
                        step % arm.pulse_period == 0) or
                       (arm.kind in ("gated", "gated-align",
                                    "gated-lookahead") and
                        diagnosis is not None and
                        diagnosis.use_large_batch))
            n = arm.virtual_batch if use_rsr else profile.batch
            z = latent_rng.normal(size=(n, model.latent_dim))
            positive = target.sample(n, data_rng)
            if not use_rsr:
                x, _, step_work = direct_quantile_step(
                    model, z, positive)
            elif arm.kind == "nrsr":
                x, _, step_work = noise_restored_rsr_step(
                    model, z, positive, microbatch=profile.batch,
                    gradient_batch=profile.batch,
                    noise_mix=arm.noise_mix, rng=gradient_rng)
            else:
                x, _, step_work = rsr_quantile_step(
                    model, z, positive, microbatch=profile.batch)
        else:
            if arm.kind in ("align", "gated-align", "lookahead",
                            "gated-lookahead") and (
                    step == warm_steps + 1 or
                    (step - warm_steps - 1) % arm.selector_period == 0):
                probe_n = profile.selector_probe
                probe_z = probe_latent_rng.normal(
                    size=(probe_n, model.latent_dim))
                probe_positive = target.sample(probe_n, probe_data_rng)
                if arm.kind in ("lookahead", "gated-lookahead"):
                    eval_z = probe_latent_rng.normal(
                        size=(probe_n, model.latent_dim))
                    eval_positive = target.sample(probe_n, probe_data_rng)
                    selection = select_tau_by_model_lookahead(
                        model, probe_z, probe_positive,
                        eval_z, eval_positive, candidate_taus)
                else:
                    selection = select_tau_by_alignment(
                        model, probe_z, probe_positive, candidate_taus)
                current_tau = selection.tau
                selector_calls += 1
                selector_kernel_pairs += selection.kernel_pairs
                selector_example_evals += selection.generator_example_evals
                tau_counts[str(current_tau)] += 1
            n = profile.batch
            z = latent_rng.normal(size=(n, model.latent_dim))
            positive = target.sample(n, data_rng)
            tau = arm.tau if arm.kind == "paper" else current_tau
            x, _, step_work = paper_step(model, z, positive, tau)
        _add_work(work, step_work)
        if (not model.finite() or
                not np.all(np.isfinite(x)) or
                np.linalg.norm(x) > 1e6 * max(target.scale, 1.0)):
            diverged = True
            break
        if event_time is None and (step == 1 or step % 50 == 0):
            reach, _ = target_diagnostics(x, target)
            if reach >= 0.90:
                event_time = step

    wall_seconds = time.perf_counter() - wall0
    common = {
        "arm": arm.label,
        "kind": arm.kind,
        "tau": arm.tau,
        "target": target.name,
        "family": target.family,
        "init": init_kind,
        "cell": f"{target.name}/{init_kind}",
        "seed": seed,
        "budget": arm.budget,
        "virtual_batch": arm.virtual_batch,
        "warm_fraction": arm.warm_fraction,
        "pulse_period": arm.pulse_period,
        "noise_mix": arm.noise_mix,
        "diagnostic_target_samples": diagnostic_target_samples,
        "diagnostic_use_large_batch": int(
            diagnosis.use_large_batch) if diagnosis is not None else 0,
        "diagnostic_min_expected_count": (
            diagnosis.minimum_expected_batch_count
            if diagnosis is not None else float("nan")),
        "diagnostic_gap_count": (
            diagnosis.significant_gap_count
            if diagnosis is not None else 0),
        "diagnostic_max_gap_ratio": (
            diagnosis.maximum_gap_ratio
            if diagnosis is not None else float("nan")),
        "planned_steps": total_steps,
        "completed_steps": int(work["optimizer_updates"]),
        "wall_seconds": wall_seconds,
        "generator_forward_calls": int(work["generator_forward_calls"]),
        "generator_example_evals": int(work["generator_example_evals"]),
        "unique_latent_samples": int(work["unique_latent_samples"]),
        "target_samples": int(work["target_samples"]),
        "kernel_pairs": int(work["kernel_pairs"]),
        "sort_work": work["sort_work"],
        "selector_calls": selector_calls,
        "selector_kernel_pairs": selector_kernel_pairs,
        "selector_example_evals": selector_example_evals,
        "tau_counts": json.dumps(tau_counts, sort_keys=True),
    }
    if diverged:
        return {
            **common,
            "ed2": float("inf"), "sw1": float("inf"),
            "weighted_reach": float("nan"), "mass_l1": float("nan"),
            "event_time": total_steps, "censored": 1, "diverged": 1,
        }

    eval_latent = np.random.default_rng(base + 4).normal(
        size=(profile.eval_size, model.latent_dim))
    q = model.forward(eval_latent)
    p = target.sample(profile.eval_size, np.random.default_rng(base + 5))
    metric_rng = np.random.default_rng(base + 6)
    metric_n = min(profile.ed_size, len(q), len(p))
    iq = metric_rng.choice(len(q), metric_n, replace=False)
    ip = metric_rng.choice(len(p), metric_n, replace=False)
    ed2 = max(float(energy_distance2(q[iq], p[ip])), 0.0)
    sw1 = sliced_w1(q, p, 1, metric_rng)
    reach, mass_l1 = target_diagnostics(q, target)
    return {
        **common,
        "ed2": ed2, "sw1": sw1,
        "weighted_reach": reach, "mass_l1": mass_l1,
        "event_time": event_time if event_time is not None else total_steps,
        "censored": int(event_time is None), "diverged": 0,
    }


def make_tasks(registry: dict[str, Any], profile: Profile,
               arms: list[Arm], inits: tuple[str, ...], seeds: int,
               candidate_taus: tuple[float, ...],
               target_filter: tuple[str, ...],
               diagnostic_samples: int) -> list[dict[str, Any]]:
    targets = list(registry["targets"])
    if target_filter:
        wanted = set(target_filter)
        targets = [spec for spec in targets
                   if str(spec["name"]) in wanted or
                   str(spec["family"]) in wanted]
        if not targets:
            raise ValueError("target filter selected no development targets")
    all_targets = list(registry["targets"])
    target_indices = {str(spec["name"]): i
                      for i, spec in enumerate(all_targets)}
    tasks: list[dict[str, Any]] = []
    for target_spec in targets:
        ti = target_indices[str(target_spec["name"])]
        for ii, init_kind in enumerate(inits):
            for arm in arms:
                for seed in range(seeds):
                    tasks.append({
                        "target_spec": target_spec,
                        "profile": asdict(profile),
                        "arm": asdict(arm),
                        "init": init_kind,
                        "seed": seed,
                        "candidate_taus": candidate_taus,
                        "diagnostic_samples": diagnostic_samples,
                        "base_seed": seed_base(
                            int(registry["master_seed"]), ti, ii, seed),
                    })
    return tasks


def execute(tasks: list[dict[str, Any]], workers: int) \
        -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_trial, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % 25 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                print(f"  {done}/{len(tasks)} trials ({elapsed:.1f}s)",
                      flush=True)
    return rows


def _positive_median(values: Iterable[float]) -> float:
    return max(float(np.median(list(values))), 1e-12)


def _cell_medians(rows: list[dict[str, Any]], metric: str) \
        -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["target"]), str(row["init"]), str(row["arm"]))
        grouped.setdefault(key, []).append(float(row[metric]))
    return {key: _positive_median(values)
            for key, values in grouped.items()}


def _geomean(values: Iterable[float]) -> float:
    values = list(values)
    return float(math.exp(np.mean(np.log(np.maximum(values, 1e-300)))))


def hierarchical_target_bootstrap(
        rows: list[dict[str, Any]], candidate: str, baseline: str,
        metric: str, reps: int, seed: int) -> list[float]:
    """Target-level bootstrap with paired seed resampling within each cell."""
    lookup = {
        (str(row["target"]), str(row["init"]), str(row["arm"]),
         int(row["seed"])): float(row[metric])
        for row in rows
    }
    targets = sorted({str(row["target"]) for row in rows})
    inits = sorted({str(row["init"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    rng = np.random.default_rng(seed)
    statistics = np.empty(reps)
    for b in range(reps):
        chosen_targets = [targets[i] for i in rng.integers(
            0, len(targets), size=len(targets))]
        log_ratios: list[float] = []
        for target in chosen_targets:
            for init in inits:
                indices = rng.integers(0, len(seeds), size=len(seeds))
                candidate_values = [lookup[
                    (target, init, candidate, seeds[i])] for i in indices]
                baseline_values = [lookup[
                    (target, init, baseline, seeds[i])] for i in indices]
                log_ratios.append(math.log(
                    _positive_median(candidate_values) /
                    _positive_median(baseline_values)))
        statistics[b] = math.exp(float(np.mean(log_ratios)))
    return [float(x) for x in np.quantile(statistics, [0.025, 0.975])]


def summarize(rows: list[dict[str, Any]], stage: str,
              selected_tau: float, bootstrap_reps: int = 5000,
              bootstrap_seed: int = 20260801) -> dict[str, Any]:
    baseline = f"paper-{selected_tau:g}"
    med_ed = _cell_medians(rows, "ed2")
    med_sw = _cell_medians(rows, "sw1")
    targets = sorted({str(row["target"]) for row in rows})
    inits = sorted({str(row["init"]) for row in rows})
    arms = sorted({str(row["arm"]) for row in rows})
    families = {str(row["target"]): str(row["family"]) for row in rows}

    paper_arms = [arm for arm in arms if arm.startswith("paper-")]
    candidates = [arm for arm in arms if not arm.startswith("paper-")]
    ratios_vs_paper: dict[str, float] = {}
    sw_ratios_vs_paper: dict[str, float] = {}
    ratios_vs_oracle: dict[str, float] = {}
    family_ratios_vs_paper: dict[str, dict[str, float]] = {}
    family_ratios_vs_qld: dict[str, dict[str, float]] = {}

    oracle: dict[tuple[str, str], float] = {}
    for target in targets:
        for init in inits:
            oracle[(target, init)] = min(
                med_ed[(target, init, arm)] for arm in paper_arms)

    for arm in candidates:
        cells = [(target, init) for target in targets for init in inits]
        ratios_vs_paper[arm] = _geomean(
            med_ed[(target, init, arm)] / med_ed[(target, init, baseline)]
            for target, init in cells)
        sw_ratios_vs_paper[arm] = _geomean(
            med_sw[(target, init, arm)] / med_sw[(target, init, baseline)]
            for target, init in cells)
        ratios_vs_oracle[arm] = _geomean(
            med_ed[(target, init, arm)] / oracle[(target, init)]
            for target, init in cells)
        family_ratios_vs_paper[arm] = {}
        family_ratios_vs_qld[arm] = {}
        for family in sorted(set(families.values())):
            fcells = [(target, init) for target, init in cells
                      if families[target] == family]
            family_ratios_vs_paper[arm][family] = _geomean(
                med_ed[(target, init, arm)] /
                med_ed[(target, init, baseline)]
                for target, init in fcells)
            if arm != "qld-v1":
                family_ratios_vs_qld[arm][family] = _geomean(
                    med_ed[(target, init, arm)] /
                    med_ed[(target, init, "qld-v1")]
                    for target, init in fcells)

    update_candidates = [
        arm for arm in candidates if arm.endswith("-u") and
        ((stage == "n1" and
          (arm.startswith("rsr-") or arm.startswith("pulse-") or
           arm.startswith("nrsr-") or arm.startswith("gated-"))) or
         (stage == "n2" and
          (arm.startswith("lbqcd-align-") or
           arm.startswith("gated-align-") or
           arm.startswith("lbqcd-lookahead-") or
           arm.startswith("gated-lookahead-"))))]
    eligible: list[str] = []
    if stage == "n1":
        for arm in update_candidates:
            unequal = family_ratios_vs_qld[arm].get("unequal", math.inf)
            controls = [value for family, value
                        in family_ratios_vs_qld[arm].items()
                        if family != "unequal"]
            if unequal < 1.0 and all(value <= 1.05 for value in controls):
                eligible.append(arm)
    selection_pool = eligible if eligible else update_candidates
    best = min(selection_pool, key=lambda arm: ratios_vs_paper[arm]) \
        if selection_pool else None
    gate: dict[str, bool] = {}
    if best is not None and stage == "n1":
        unequal = family_ratios_vs_qld[best].get("unequal", math.inf)
        controls = [value for family, value
                    in family_ratios_vs_qld[best].items()
                    if family != "unequal"]
        gate = {
            "unequal_ed2_improves_vs_qld_v1": unequal < 1.0,
            "no_control_family_loses_more_than_5pct_vs_qld_v1":
                all(value <= 1.05 for value in controls),
        }
    elif best is not None and stage == "n2":
        gate = {
            "ed2_ratio_vs_selected_paper_at_most_0.82":
                ratios_vs_paper[best] <= 0.82,
            "ed2_ratio_vs_paper_oracle_at_most_0.95":
                ratios_vs_oracle[best] <= 0.95,
        }

    work_summary: dict[str, dict[str, float]] = {}
    work_totals: dict[str, dict[str, float]] = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        work_summary[arm] = {
            key: float(np.median([float(row[key]) for row in arm_rows]))
            for key in ("wall_seconds", "generator_example_evals",
                        "unique_latent_samples", "kernel_pairs", "sort_work",
                        "selector_kernel_pairs")
        }
        work_summary[arm]["divergences"] = float(sum(
            int(row["diverged"]) for row in arm_rows))
        work_totals[arm] = {
            key: float(sum(float(row[key]) for row in arm_rows))
            for key in ("wall_seconds", "generator_example_evals",
                        "unique_latent_samples", "target_samples",
                        "diagnostic_target_samples", "kernel_pairs",
                        "selector_kernel_pairs")
        }

    best_details: dict[str, Any] = {}
    if best is not None:
        cells = [(target, init) for target in targets for init in inits]
        best_details = {
            "ed2_bootstrap_ci_vs_selected_paper":
                hierarchical_target_bootstrap(
                    rows, best, baseline, "ed2", bootstrap_reps,
                    bootstrap_seed),
            "sw1_bootstrap_ci_vs_selected_paper":
                hierarchical_target_bootstrap(
                    rows, best, baseline, "sw1", bootstrap_reps,
                    bootstrap_seed + 1),
            "cell_win_fraction_vs_selected_paper": float(np.mean([
                med_ed[(target, init, best)] <
                med_ed[(target, init, baseline)]
                for target, init in cells])),
            "init_ed2_ratios_vs_selected_paper": {
                init: _geomean(
                    med_ed[(target, init, best)] /
                    med_ed[(target, init, baseline)]
                    for target in targets)
                for init in inits
            },
            "summed_wall_ratio_vs_selected_paper":
                work_totals[best]["wall_seconds"] /
                max(work_totals[baseline]["wall_seconds"], 1e-300),
            "generator_eval_ratio_vs_selected_paper":
                work_totals[best]["generator_example_evals"] /
                max(work_totals[baseline]["generator_example_evals"],
                    1e-300),
            "kernel_pair_ratio_vs_selected_paper":
                (work_totals[best]["kernel_pairs"] +
                 work_totals[best]["selector_kernel_pairs"]) /
                max(work_totals[baseline]["kernel_pairs"], 1e-300),
            "divergences": int(work_summary[best]["divergences"]),
            "routing_rate_by_target": {
                target: float(np.mean([
                    int(row["diagnostic_use_large_batch"])
                    for row in rows
                    if row["arm"] == best and row["target"] == target]))
                for target in targets
            } if best.startswith("gated-") else {},
        }

    return {
        "program": "LBQCD-development-v1",
        "stage": stage,
        "selected_paper_tau": selected_tau,
        "selected_baseline": baseline,
        "targets": targets,
        "inits": inits,
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "ratios_vs_selected_paper": ratios_vs_paper,
        "sw1_ratios_vs_selected_paper": sw_ratios_vs_paper,
        "ratios_vs_paper_oracle": ratios_vs_oracle,
        "family_ratios_vs_selected_paper": family_ratios_vs_paper,
        "family_ratios_vs_qld_v1": family_ratios_vs_qld,
        "best_update_candidate": best,
        "eligible_n1_candidates": eligible,
        "advancement_gate": gate,
        "gate_passed": bool(gate) and all(gate.values()),
        "work_medians": work_summary,
        "work_totals": work_totals,
        "best_candidate_details": best_details,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(
            rows, key=lambda row: (str(row["target"]), str(row["init"]),
                                   str(row["arm"]), int(row["seed"]))))


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    best = summary["best_update_candidate"]
    lines = [
        f"# LB-QCD {summary['stage'].upper()} development results",
        "",
        "**Development evidence only. This registry is not sealed and these "
        "results are not a confirmatory claim.**",
        "",
        f"- selected paper baseline: `{summary['selected_baseline']}`",
        f"- best update-matched candidate: `{best}`",
        f"- advancement gate: **{'PASS' if summary['gate_passed'] else 'FAIL'}**",
        "",
        "## Aggregate ratios",
        "",
        "| arm | ED2 / selected paper | ED2 / paper oracle | SW1 / selected paper |",
        "|---|---:|---:|---:|",
    ]
    for arm, ratio in sorted(summary["ratios_vs_selected_paper"].items()):
        lines.append(
            f"| {arm} | {ratio:.4f} | "
            f"{summary['ratios_vs_paper_oracle'][arm]:.4f} | "
            f"{summary['sw1_ratios_vs_selected_paper'][arm]:.4f} |")
    lines.extend(["", "## Advancement checks", ""])
    if summary["advancement_gate"]:
        for name, passed in summary["advancement_gate"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    else:
        lines.append("- No eligible update-matched candidate was run.")
    if best:
        lines.extend(["", f"## Best candidate families: {best}", "",
                      "| family | vs selected paper | vs QLD-v1 |",
                      "|---|---:|---:|"])
        paper_families = summary["family_ratios_vs_selected_paper"][best]
        qld_families = summary["family_ratios_vs_qld_v1"][best]
        for family in sorted(paper_families):
            lines.append(f"| {family} | {paper_families[family]:.4f} | "
                         f"{qld_families[family]:.4f} |")
        details = summary["best_candidate_details"]
        lines.extend([
            "", "## Uncertainty and cost", "",
            "- target-bootstrap ED2 95% CI: "
            f"`[{details['ed2_bootstrap_ci_vs_selected_paper'][0]:.4f}, "
            f"{details['ed2_bootstrap_ci_vs_selected_paper'][1]:.4f}]`",
            "- target-bootstrap SW1 95% CI: "
            f"`[{details['sw1_bootstrap_ci_vs_selected_paper'][0]:.4f}, "
            f"{details['sw1_bootstrap_ci_vs_selected_paper'][1]:.4f}]`",
            "- cell win fraction: "
            f"`{details['cell_win_fraction_vs_selected_paper']:.4f}`",
            "- summed worker wall ratio: "
            f"`{details['summed_wall_ratio_vs_selected_paper']:.4f}`",
            "- generator-example-evaluation ratio: "
            f"`{details['generator_eval_ratio_vs_selected_paper']:.4f}`",
            "- kernel-pair ratio: "
            f"`{details['kernel_pair_ratio_vs_selected_paper']:.4f}`",
            f"- divergences: `{details['divergences']}`",
        ])
    lines.extend(["", "## Interpretation rule", "",
                  "N1 advances only when virtual batching improves the "
                  "unequal family relative to QLD-v1 without making any "
                  "control family more than 5% worse. N2 advances only at "
                  "ED2 ratios <= .82 against the selected paper baseline "
                  "and <= .95 against the per-cell paper oracle.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_run_dir(stage: str, profile: Profile) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNROOT / f"{stamp}-{stage}-{profile.name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_manifest(path: Path, args: argparse.Namespace, profile: Profile,
                   arms: list[Arm], tasks: list[dict[str, Any]],
                   wall_seconds: float) -> None:
    sources = [Path(__file__), HERE / "lbqcd.py", REGISTRY, RESEARCH,
               HERE / "identifiability_drift.py",
               HERE / "run_identifiability_generator.py",
               HERE / "lowdim_drift.py"]
    manifest = {
        "program": "LBQCD-development-v1",
        "stage": args.stage,
        "profile": asdict(profile),
        "arms": [asdict(arm) for arm in arms],
        "task_count": len(tasks),
        "workers": args.workers,
        "registry_sha256": sha256_file(REGISTRY),
        "commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--porcelain").splitlines(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "wall_seconds": wall_seconds,
        "command": sys.argv,
        "source_sha256": {
            str(source.relative_to(ROOT)): sha256_file(source)
            for source in sources
        },
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    for source in sources:
        shutil.copy2(source, path / f"source_snapshot_{source.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("n1", "n2"), default="n1")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="screen")
    parser.add_argument("--virtual-batches", default="512,1024")
    parser.add_argument("--warm-fractions", default="0.70")
    parser.add_argument("--budgets", default="update,examples")
    parser.add_argument("--candidate-taus", default="0.1,0.2,0.5,1,2")
    parser.add_argument("--selector-period", type=int, default=50)
    parser.add_argument("--pulse-periods", default="")
    parser.add_argument("--noise-mixes", default="")
    parser.add_argument("--include-gated", action="store_true")
    parser.add_argument("--include-lookahead", action="store_true")
    parser.add_argument("--diagnostic-samples", type=int, default=4096)
    parser.add_argument("--inits", default="missing,concentrated")
    parser.add_argument("--targets", default="")
    parser.add_argument("--arms", default="")
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--workers", type=int,
                        default=min(8, os.cpu_count() or 1))
    parser.add_argument("--self-check-only", action="store_true")
    parser.add_argument("--reanalyze", type=Path)
    args = parser.parse_args()

    invariant_tests()
    registry = load_registry()
    if args.reanalyze is not None:
        run_dir = args.reanalyze
        with (run_dir / "rows.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = summarize(rows, str(manifest["stage"]),
                            SELECTED_PAPER_TAU)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        write_summary_md(run_dir / "RESULTS.md", summary)
        print(json.dumps(summary["best_candidate_details"], indent=2))
        return
    if args.self_check_only:
        print("LB-QCD invariant tests: PASS")
        return
    profile = PROFILES[args.profile]
    virtual_batches = _parse_numbers(args.virtual_batches, int)
    warm_fractions = _parse_numbers(args.warm_fractions, float)
    budgets = _parse_numbers(args.budgets, str)
    if any(budget not in ("update", "examples") for budget in budgets):
        parser.error("budgets must contain only update and/or examples")
    candidate_taus = _parse_numbers(args.candidate_taus, float)
    pulse_periods = _parse_numbers(args.pulse_periods, int)
    if any(period <= 0 for period in pulse_periods):
        parser.error("pulse periods must be positive")
    noise_mixes = _parse_numbers(args.noise_mixes, float)
    if any(not 0.0 <= value <= 1.0 for value in noise_mixes):
        parser.error("noise mixes must lie in [0, 1]")
    if args.diagnostic_samples < 32:
        parser.error("diagnostic samples must be at least 32")
    inits = _parse_numbers(args.inits, str)
    target_filter = _parse_numbers(args.targets, str)
    seeds = args.seeds if args.seeds is not None else profile.seeds
    arms = make_arms(args.stage, virtual_batches, warm_fractions, budgets,
                     args.selector_period, pulse_periods, noise_mixes,
                     args.include_gated, args.include_lookahead)
    arm_filter = _parse_numbers(args.arms, str)
    if arm_filter:
        wanted_arms = set(arm_filter)
        arms = [arm for arm in arms if arm.label in wanted_arms]
        missing = wanted_arms - {arm.label for arm in arms}
        if missing:
            parser.error(f"unknown or disabled arms: {sorted(missing)}")
    tasks = make_tasks(registry, profile, arms, inits, seeds,
                       candidate_taus, target_filter,
                       args.diagnostic_samples)
    run_dir = make_run_dir(args.stage, profile)
    print(f"run directory: {run_dir}", flush=True)
    print(f"registry sha256: {sha256_file(REGISTRY)}", flush=True)
    print(f"tasks: {len(tasks)}", flush=True)
    wall0 = time.perf_counter()
    rows = execute(tasks, max(1, args.workers))
    wall = time.perf_counter() - wall0
    write_csv(run_dir / "rows.csv", rows)
    summary = summarize(rows, args.stage, SELECTED_PAPER_TAU)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    write_summary_md(run_dir / "RESULTS.md", summary)
    write_manifest(run_dir, args, profile, arms, tasks, wall)
    print(json.dumps({
        "best_update_candidate": summary["best_update_candidate"],
        "ratios_vs_selected_paper": summary["ratios_vs_selected_paper"],
        "ratios_vs_paper_oracle": summary["ratios_vs_paper_oracle"],
        "advancement_gate": summary["advancement_gate"],
        "gate_passed": summary["gate_passed"],
    }, indent=2), flush=True)
    print(f"completed in {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
