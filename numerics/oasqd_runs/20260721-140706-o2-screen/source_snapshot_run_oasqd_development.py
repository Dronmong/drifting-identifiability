"""Staged development runner for Occupancy-Adaptive Stratified QLD.

The runner implements stages O1--O5 from
``OccupancyAdaptiveQuantileResearch.md``.  It is deliberately development-only:
the registry is fresh and hash-guarded, but outcomes may be used for mechanism
selection and therefore cannot support a confirmatory claim.
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
    _stopgrad_grads,
    diagnose_quantile_resolution_stable,
    direct_quantile_step,
    invariant_tests as lbqcd_invariant_tests,
    paper_step,
    rsr_quantile_step,
)
from lowdim_drift import energy_distance2, sliced_w1  # noqa: E402
from oasqd import (  # noqa: E402
    AtlasConfig,
    OccupancyController,
    QuantileAtlas,
    assess_occupancy,
    assign_regions,
    build_quantile_atlas,
    choose_virtual_batch,
    full_virtual_table_gradient,
    invariant_tests as oasqd_invariant_tests,
    randomized_systematic_target,
    stratified_rsr_quantile_step,
    stratified_virtual_table_gradient,
)
from run_identifiability_generator import TanhMLP  # noqa: E402
from run_qld_confirmatory import (  # noqa: E402
    Target1D,
    build_target,
    sha256_file,
    target_diagnostics,
)


REGISTRY = HERE / "oasqd_development_registry.json"
REGISTRY_SHA256 = \
    "111FA056B30931F2BEBC6C95D7DFBB4CA0C810910ECAC2A8BF693782C6E858B8"
PROTOCOL = HERE / "OASQDDevelopmentProtocol.md"
RESEARCH = HERE / "OccupancyAdaptiveQuantileResearch.md"
RUNROOT = HERE / "oasqd_runs"
OLD_REGISTRIES = (
    HERE / "lbqcd_development_registry.json",
    HERE / "lbqcd_confirmatory_registry.json",
)
PAPER_TAUS = (0.2, 0.5, 1.0, 2.0, 4.0)
BASELINE_TAU = 0.5


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    batch: int
    seeds: int
    eval_size: int
    ed_size: int
    event_probe: int
    atlas_samples: int
    bootstrap_reps: int


PROFILES = {
    "smoke": Profile("smoke", 40, 32, 1, 512, 256, 256, 2048, 100),
    "screen": Profile("screen", 400, 128, 3, 2048, 512, 1024, 8192, 1000),
    "standard": Profile(
        "standard", 1200, 128, 8, 4096, 1024, 2048, 8192, 5000),
}


@dataclass(frozen=True)
class Arm:
    label: str
    kind: str
    tau: float = BASELINE_TAU
    virtual_batch: int = 1024
    warm_fraction: float = 0.70
    check_period: int = 25
    pulse_length: int = 25
    probe_size: int = 2048
    target_count: float = 8.0
    backward_batch: int = 128
    adaptive: bool = False
    systematic_target: bool = False
    one_shot: bool = False


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


def _canonical_target(spec: dict[str, Any]) -> str:
    return json.dumps({
        key: value for key, value in spec.items()
        if key not in ("name", "family")
    }, sort_keys=True, separators=(",", ":"))


def load_registry() -> dict[str, Any]:
    actual = sha256_file(REGISTRY)
    if actual != REGISTRY_SHA256:
        raise RuntimeError(
            f"OA-SQD registry hash mismatch: {actual} != {REGISTRY_SHA256}")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("registry") != "OASQD-development-v1":
        raise RuntimeError("unexpected OA-SQD registry")
    targets = list(registry.get("targets", []))
    if len(targets) != 16:
        raise RuntimeError("OA-SQD development registry must have 16 targets")
    names = [str(spec["name"]) for spec in targets]
    if len(set(names)) != len(names):
        raise RuntimeError("OA-SQD target names are not unique")
    old_specs: set[str] = set()
    for path in OLD_REGISTRIES:
        old = json.loads(path.read_text(encoding="utf-8"))
        old_specs.update(_canonical_target(spec)
                         for spec in old.get("targets", []))
    for spec in targets:
        build_target(spec)
        if _canonical_target(spec) in old_specs:
            raise RuntimeError(
                f"OA-SQD target duplicates a frozen specification: "
                f"{spec['name']}")
    return registry


def seed_base(master: int, target_index: int, init_index: int,
              seed: int) -> int:
    return int(master * 1_000_003 + target_index * 100_003 +
               init_index * 10_007 + seed * 101)


def atlas_config(bootstrap_reps: int) -> AtlasConfig:
    return AtlasConfig(bootstrap_reps=bootstrap_reps)


def make_arms(stage: str, *, check_periods: tuple[int, ...],
              pulse_lengths: tuple[int, ...], probe_sizes: tuple[int, ...],
              target_counts: tuple[float, ...], backward_batch: int) \
        -> list[Arm]:
    if stage == "o1":
        return []
    paper_taus = PAPER_TAUS if stage == "o5" else (BASELINE_TAU,)
    arms = [Arm(f"paper-{tau:g}", "paper", tau=tau)
            for tau in paper_taus]
    arms.extend([
        Arm("qld-v1", "qld"),
        Arm("fixed-lbqcd", "fixed"),
    ])
    for check in check_periods:
        for pulse in pulse_lengths:
            for probe in probe_sizes:
                for count in target_counts:
                    tag = f"H{check}-P{pulse}-N{probe}-K{count:g}"
                    if stage == "o2":
                        arms.extend([
                            Arm(f"oa-stop-full-{tag}", "oa-full",
                                check_period=check, pulse_length=-1,
                                probe_size=probe, target_count=count,
                                one_shot=True),
                            Arm(f"oa-once-full-{tag}", "oa-full",
                                check_period=check, pulse_length=pulse,
                                probe_size=probe, target_count=count,
                                one_shot=True),
                            Arm(f"oa-pulse-full-{tag}", "oa-full",
                                check_period=check, pulse_length=pulse,
                                probe_size=probe, target_count=count),
                        ])
                    elif stage in ("o3", "o4", "o5"):
                        arms.append(Arm(
                            f"oa-pulse-full-{tag}", "oa-full",
                            check_period=check, pulse_length=pulse,
                            probe_size=probe, target_count=count))
                        arms.append(Arm(
                            f"oa-pulse-strat-{tag}", "oa-stratified",
                            check_period=check, pulse_length=pulse,
                            probe_size=probe, target_count=count,
                            backward_batch=backward_batch))
                        if stage in ("o4", "o5"):
                            arms.append(Arm(
                                f"oasqd-adaptive-{tag}", "oa-stratified",
                                check_period=check, pulse_length=pulse,
                                probe_size=probe, target_count=count,
                                backward_batch=backward_batch,
                                adaptive=True, systematic_target=True))
    return arms


def _add_training_work(total: dict[str, float], step_work,
                       backward_examples: int) -> None:
    for key in ("optimizer_updates", "generator_forward_calls",
                "generator_example_evals", "unique_latent_samples",
                "kernel_pairs", "sort_work"):
        total[key] += float(getattr(step_work, key))
    total["backward_example_evals"] += float(backward_examples)


def _make_atlas(target: Target1D, profile: Profile, base: int) \
        -> QuantileAtlas:
    samples = target.sample(
        profile.atlas_samples, np.random.default_rng(base + 34))
    return build_quantile_atlas(
        samples, np.random.default_rng(base + 35),
        config=atlas_config(8 if profile.name == "smoke" else 32))


def run_trial(task: dict[str, Any]) -> dict[str, Any]:
    target: Target1D = build_target(task["target_spec"])
    profile = Profile(**task["profile"])
    arm = Arm(**task["arm"])
    base = int(task["base_seed"])
    seed = int(task["seed"])
    init_kind = str(task["init"])
    warm_steps = int(round(arm.warm_fraction * profile.steps))
    wall0 = time.perf_counter()

    model = TanhMLP(target, init_kind, base + 1)
    latent_rng = np.random.default_rng(base + 2)
    data_rng = np.random.default_rng(base + 3)
    eval_latent_rng = np.random.default_rng(base + 4)
    eval_data_rng = np.random.default_rng(base + 5)
    metric_rng = np.random.default_rng(base + 6)
    event_rng = np.random.default_rng(base + 30)
    probe_rng = np.random.default_rng(base + 36)
    selection_rng = np.random.default_rng(base + 37)
    systematic_rng = np.random.default_rng(base + 38)

    atlas: QuantileAtlas | None = None
    controller: OccupancyController | None = None
    assessment = None
    atlas_samples = 0
    if arm.kind.startswith("oa-"):
        atlas = _make_atlas(target, profile, base)
        atlas_samples = profile.atlas_samples
        pulse_length = warm_steps if arm.pulse_length < 0 else arm.pulse_length
        controller = OccupancyController(
            pulse_length=max(1, pulse_length), cooldown_checks=1,
            clear_checks=2)

    fixed_diagnosis = None
    diagnostic_samples = 0
    if arm.kind == "fixed":
        diagnostic_samples = 4096
        diagnostic = target.sample(
            diagnostic_samples, np.random.default_rng(base + 39))
        fixed_diagnosis = diagnose_quantile_resolution_stable(
            diagnostic, profile.batch)

    work = {key: 0.0 for key in (
        "optimizer_updates", "generator_forward_calls",
        "generator_example_evals", "unique_latent_samples",
        "kernel_pairs", "sort_work", "backward_example_evals")}
    training_target_samples = 0
    target_table_lookups = 0
    controller_probe_evals = 0
    event_probe_evals = 0
    controller_checks = 0
    global_updates = 0
    resolution_cap_hits = 0
    virtual_batch_total = 0
    controller_state_counts = {
        "local": 0, "armed": 0, "global": 0, "cooldown": 0}
    one_shot_finished = False
    one_shot_started = False
    event_time: int | None = None
    diverged = False
    x = np.zeros((profile.batch, 1))

    for step in range(1, profile.steps + 1):
        early_hybrid = (arm.kind in (
            "qld", "fixed", "oa-full", "oa-stratified") and
            step <= warm_steps)
        use_global = False
        if early_hybrid and arm.kind.startswith("oa-"):
            assert atlas is not None and controller is not None
            if (not one_shot_finished and
                    (step == 1 or (step - 1) % arm.check_period == 0)):
                probe_latent = probe_rng.normal(
                    size=(arm.probe_size, model.latent_dim))
                generated_probe = model.forward(probe_latent)
                controller_probe_evals += arm.probe_size
                controller_checks += 1
                assessment = assess_occupancy(
                    atlas, generated_probe, ordinary_batch=profile.batch,
                    target_count=arm.target_count)
                controller.observe(assessment)
            use_global = controller.use_global and not one_shot_finished

        if early_hybrid:
            if arm.kind == "qld":
                n = profile.batch
            elif arm.kind == "fixed":
                use_global = bool(fixed_diagnosis and
                                  fixed_diagnosis.use_large_batch)
                n = arm.virtual_batch if use_global else profile.batch
                if use_global:
                    virtual_batch_total += n
            elif use_global:
                assert atlas is not None and controller is not None
                if arm.adaptive:
                    active = controller.active_regions
                    if not active and assessment is not None:
                        active = assessment.active_regions
                    choice = choose_virtual_batch(
                        atlas, active, target_count=arm.target_count)
                    n = choice.size
                    resolution_cap_hits += int(choice.cap_hit)
                else:
                    n = arm.virtual_batch
                virtual_batch_total += n
            else:
                n = profile.batch

            latent = latent_rng.normal(size=(n, model.latent_dim))
            if use_global and arm.systematic_target:
                assert atlas is not None
                positive = randomized_systematic_target(
                    atlas, n, systematic_rng)
                target_table_lookups += n
            else:
                positive = target.sample(n, data_rng)
                training_target_samples += n

            if not use_global:
                x, _, step_work = direct_quantile_step(
                    model, latent, positive)
                backward_examples = n
            elif arm.kind == "oa-stratified":
                assert atlas is not None
                x, _, selection, step_work = stratified_rsr_quantile_step(
                    model, latent, positive, atlas,
                    microbatch=profile.batch,
                    backward_batch=min(arm.backward_batch, n),
                    rng=selection_rng)
                backward_examples = len(selection.indices)
                global_updates += 1
                assert controller is not None
                controller.record_global_update()
            else:
                x, _, step_work = rsr_quantile_step(
                    model, latent, positive, microbatch=profile.batch)
                backward_examples = n
                if arm.kind.startswith("oa-"):
                    global_updates += 1
                    assert controller is not None
                    controller.record_global_update()
                elif arm.kind == "fixed":
                    global_updates += 1
            _add_training_work(work, step_work, backward_examples)

            if arm.one_shot and controller is not None:
                one_shot_started = one_shot_started or global_updates > 0
                if one_shot_started and not controller.use_global:
                    one_shot_finished = True
        else:
            n = profile.batch
            latent = latent_rng.normal(size=(n, model.latent_dim))
            positive = target.sample(n, data_rng)
            training_target_samples += n
            tau = arm.tau if arm.kind == "paper" else BASELINE_TAU
            x, _, step_work = paper_step(model, latent, positive, tau)
            _add_training_work(work, step_work, n)

        if controller is not None:
            controller_state_counts[controller.state] += 1
        if (not model.finite() or not np.all(np.isfinite(x)) or
                np.linalg.norm(x) > 1e6 * max(target.scale, 1.0)):
            diverged = True
            break

        # Fair event timing: every arm is evaluated with the same independent
        # probe size.  The old LB-QCD runner used the current training batch,
        # which favored M=1024 arms on rare-mode reach.
        if event_time is None and (step == 1 or step % 50 == 0):
            event_latent = event_rng.normal(
                size=(profile.event_probe, model.latent_dim))
            event_q = model.forward(event_latent)
            event_probe_evals += profile.event_probe
            reach, _ = target_diagnostics(event_q, target)
            if reach >= 0.90:
                event_time = step

    wall_seconds = time.perf_counter() - wall0
    controller_transitions = controller.transition_count if controller else 0
    atlas_region_count = atlas.region_count if atlas else 0
    atlas_unresolved = int(atlas.unresolved) if atlas else 0
    min_atlas_mass = min(atlas.masses) if atlas else float("nan")
    total_generator_evals = int(
        work["generator_example_evals"] + controller_probe_evals +
        event_probe_evals)
    common = {
        "arm": arm.label, "kind": arm.kind, "tau": arm.tau,
        "target": target.name, "family": target.family,
        "init": init_kind, "cell": f"{target.name}/{init_kind}",
        "seed": seed, "planned_steps": profile.steps,
        "completed_steps": int(work["optimizer_updates"]),
        "wall_seconds": wall_seconds,
        "generator_forward_calls": int(work["generator_forward_calls"]),
        "training_generator_example_evals": int(
            work["generator_example_evals"]),
        "controller_probe_evals": controller_probe_evals,
        "event_probe_evals": event_probe_evals,
        "generator_example_evals": total_generator_evals,
        "unique_training_latents": int(work["unique_latent_samples"]),
        "backward_example_evals": int(work["backward_example_evals"]),
        "training_target_samples": training_target_samples,
        "atlas_samples": atlas_samples,
        "diagnostic_target_samples": diagnostic_samples,
        "target_table_lookups": target_table_lookups,
        "total_new_target_samples": (
            training_target_samples + atlas_samples + diagnostic_samples),
        "kernel_pairs": int(work["kernel_pairs"]),
        "sort_work": work["sort_work"],
        "global_updates": global_updates,
        "virtual_batch_total": virtual_batch_total,
        "resolution_cap_hits": resolution_cap_hits,
        "controller_checks": controller_checks,
        "controller_transitions": controller_transitions,
        "controller_state_counts": json.dumps(
            controller_state_counts, sort_keys=True),
        "atlas_region_count": atlas_region_count,
        "atlas_unresolved": atlas_unresolved,
        "min_atlas_mass": min_atlas_mass,
        "fixed_diagnostic_route": int(
            fixed_diagnosis.use_large_batch) if fixed_diagnosis else 0,
    }
    if diverged:
        return {
            **common, "ed2": float("inf"), "sw1": float("inf"),
            "weighted_reach": float("nan"), "mass_l1": float("nan"),
            "atlas_mass_l1": float("nan"),
            "event_time": profile.steps, "censored": 1, "diverged": 1,
        }

    q = model.forward(eval_latent_rng.normal(
        size=(profile.eval_size, model.latent_dim)))
    p = target.sample(profile.eval_size, eval_data_rng)
    metric_n = min(profile.ed_size, len(q), len(p))
    iq = metric_rng.choice(len(q), metric_n, replace=False)
    ip = metric_rng.choice(len(p), metric_n, replace=False)
    ed2 = max(float(energy_distance2(q[iq], p[ip])), 0.0)
    sw1 = sliced_w1(q, p, 1, metric_rng)
    reach, mass_l1 = target_diagnostics(q, target)
    if atlas is not None:
        q_counts = np.bincount(
            assign_regions(q, atlas), minlength=atlas.region_count)
        q_mass = q_counts / len(q)
        atlas_mass_l1 = float(np.sum(np.abs(q_mass - atlas.masses)))
    else:
        atlas_mass_l1 = float("nan")
    return {
        **common, "ed2": ed2, "sw1": sw1,
        "weighted_reach": reach, "mass_l1": mass_l1,
        "atlas_mass_l1": atlas_mass_l1,
        "event_time": event_time if event_time is not None else profile.steps,
        "censored": int(event_time is None), "diverged": 0,
    }


def make_tasks(registry: dict[str, Any], profile: Profile,
               arms: list[Arm], inits: tuple[str, ...], seeds: int,
               target_filter: tuple[str, ...]) -> list[dict[str, Any]]:
    all_targets = list(registry["targets"])
    targets = all_targets
    if target_filter:
        wanted = set(target_filter)
        targets = [spec for spec in targets
                   if str(spec["name"]) in wanted or
                   str(spec["family"]) in wanted]
        if not targets:
            raise ValueError("target filter selected no OA-SQD targets")
    target_indices = {str(spec["name"]): i
                      for i, spec in enumerate(all_targets)}
    tasks: list[dict[str, Any]] = []
    for spec in targets:
        ti = target_indices[str(spec["name"])]
        for ii, init in enumerate(inits):
            for arm in arms:
                for seed in range(seeds):
                    tasks.append({
                        "target_spec": spec, "profile": asdict(profile),
                        "arm": asdict(arm), "init": init, "seed": seed,
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
                print(f"  {done}/{len(tasks)} trials "
                      f"({time.perf_counter() - started:.1f}s)", flush=True)
    return rows


def _positive_median(values: Iterable[float]) -> float:
    return max(float(np.median(list(values))), 1e-12)


def _geomean(values: Iterable[float]) -> float:
    array = np.maximum(np.asarray(list(values), dtype=float), 1e-300)
    return float(math.exp(float(np.mean(np.log(array)))))


def _cell_medians(rows: list[dict[str, Any]], metric: str) \
        -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["target"]), str(row["init"]), str(row["arm"]))
        grouped.setdefault(key, []).append(float(row[metric]))
    return {key: _positive_median(values)
            for key, values in grouped.items()}


def hierarchical_target_bootstrap(
        rows: list[dict[str, Any]], candidate: str, baseline: str,
        metric: str, reps: int, seed: int) -> list[float]:
    lookup = {
        (str(row["target"]), str(row["init"]), str(row["arm"]),
         int(row["seed"])): float(row[metric]) for row in rows
    }
    targets = sorted({str(row["target"]) for row in rows})
    inits = sorted({str(row["init"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    rng = np.random.default_rng(seed)
    statistics = np.empty(reps)
    for rep in range(reps):
        chosen_targets = [targets[i] for i in rng.integers(
            0, len(targets), size=len(targets))]
        ratios: list[float] = []
        for target in chosen_targets:
            for init in inits:
                selected = rng.integers(0, len(seeds), size=len(seeds))
                c = [lookup[(target, init, candidate, seeds[i])]
                     for i in selected]
                b = [lookup[(target, init, baseline, seeds[i])]
                     for i in selected]
                ratios.append(_positive_median(c) / _positive_median(b))
        statistics[rep] = _geomean(ratios)
    return [float(value) for value in
            np.quantile(statistics, [0.025, 0.975])]


def summarize(rows: list[dict[str, Any]], stage: str,
              bootstrap_reps: int) -> dict[str, Any]:
    med_ed = _cell_medians(rows, "ed2")
    med_sw = _cell_medians(rows, "sw1")
    med_event = _cell_medians(rows, "event_time")
    targets = sorted({str(row["target"]) for row in rows})
    inits = sorted({str(row["init"]) for row in rows})
    arms = sorted({str(row["arm"]) for row in rows})
    families = {str(row["target"]): str(row["family"]) for row in rows}
    cells = [(target, init) for target in targets for init in inits]
    baseline = "paper-0.5"
    ratios: dict[str, dict[str, float]] = {}
    for arm in arms:
        ratios[arm] = {}
        for reference in (baseline, "qld-v1", "fixed-lbqcd"):
            if reference in arms:
                ratios[arm][reference] = _geomean(
                    med_ed[(target, init, arm)] /
                    med_ed[(target, init, reference)]
                    for target, init in cells)

    candidates = [arm for arm in arms if arm.startswith("oa")]
    best = min(candidates, key=lambda arm: ratios[arm][baseline]) \
        if candidates else None
    details: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    if best is not None:
        family_vs_qld = {
            family: _geomean(
                med_ed[(target, init, best)] /
                med_ed[(target, init, "qld-v1")]
                for target, init in cells if families[target] == family)
            for family in sorted(set(families.values()))
        }
        init_vs_qld = {
            init: _geomean(
                med_ed[(target, init, best)] /
                med_ed[(target, init, "qld-v1")] for target in targets)
            for init in inits
        }
        event_vs_fixed = _geomean(
            med_event[(target, init, best)] /
            med_event[(target, init, "fixed-lbqcd")]
            for target, init in cells)
        totals: dict[str, dict[str, float]] = {}
        for arm in arms:
            arm_rows = [row for row in rows if row["arm"] == arm]
            totals[arm] = {
                key: float(sum(float(row[key]) for row in arm_rows))
                for key in ("wall_seconds", "generator_example_evals",
                            "backward_example_evals", "kernel_pairs",
                            "total_new_target_samples", "global_updates")
            }
            totals[arm]["divergences"] = float(sum(
                int(row["diverged"]) for row in arm_rows))
        generator_ratio = (totals[best]["generator_example_evals"] /
                           max(totals[baseline]["generator_example_evals"],
                               1e-300))
        details = {
            "ed2_bootstrap_ci_vs_paper": hierarchical_target_bootstrap(
                rows, best, baseline, "ed2", bootstrap_reps, 20260918),
            "ed2_bootstrap_ci_vs_qld": hierarchical_target_bootstrap(
                rows, best, "qld-v1", "ed2", bootstrap_reps, 20260919),
            "cell_win_fraction_vs_paper": float(np.mean([
                med_ed[(target, init, best)] <
                med_ed[(target, init, baseline)]
                for target, init in cells])),
            "sw1_ratio_vs_paper": _geomean(
                med_sw[(target, init, best)] /
                med_sw[(target, init, baseline)]
                for target, init in cells),
            "family_ed2_ratios_vs_qld": family_vs_qld,
            "init_ed2_ratios_vs_qld": init_vs_qld,
            "event_time_ratio_vs_fixed": event_vs_fixed,
            "generator_eval_ratio_vs_paper": generator_ratio,
            "work_totals": totals,
        }
        if stage == "o2":
            gates = {
                "endpoint_improves_vs_fixed":
                    ratios[best]["fixed-lbqcd"] < 1.0,
                "fair_event_time_no_worse_than_fixed":
                    event_vs_fixed <= 1.0,
                "uses_fewer_global_updates_than_fixed":
                    totals[best]["global_updates"] <
                    totals["fixed-lbqcd"]["global_updates"],
            }
        elif stage == "o5":
            gates = {
                "ed2_vs_qld_at_most_0.95":
                    ratios[best]["qld-v1"] <= 0.95,
                "ed2_vs_paper_at_most_0.78":
                    ratios[best][baseline] <= 0.78,
                "each_init_vs_qld_at_most_0.98":
                    all(value <= 0.98 for value in init_vs_qld.values()),
                "worst_family_vs_qld_at_most_1.05":
                    all(value <= 1.05 for value in family_vs_qld.values()),
                "event_time_vs_fixed_at_most_1": event_vs_fixed <= 1.0,
                "generator_evals_vs_paper_at_most_3":
                    generator_ratio <= 3.0,
                "divergence_no_worse":
                    totals[best]["divergences"] <=
                    min(totals[baseline]["divergences"],
                        totals["qld-v1"]["divergences"],
                        totals["fixed-lbqcd"]["divergences"]),
            }
    paper_arms = [arm for arm in arms if arm.startswith("paper-")]
    oracle_ratio = None
    if best is not None and len(paper_arms) > 1:
        oracle_ratio = _geomean(
            med_ed[(target, init, best)] /
            min(med_ed[(target, init, arm)] for arm in paper_arms)
            for target, init in cells)
    return {
        "program": "OASQD-development-v1", "stage": stage,
        "targets": targets, "inits": inits,
        "seeds": sorted({int(row["seed"]) for row in rows}),
        "arms": arms, "ed2_ratios": ratios,
        "best_candidate": best, "best_vs_paper_oracle": oracle_ratio,
        "best_details": details, "advancement_gate": gates,
        "gate_passed": bool(gates) and all(gates.values()),
    }


def _flatten_gradient(model: TanhMLP,
                      gradient: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([gradient[name].reshape(-1)
                           for name in model.names])


def gradient_audit(registry: dict[str, Any], reps: int = 400) \
        -> dict[str, Any]:
    """Monte Carlo audit of conditional unbiasedness on one fixed rank table."""
    target = build_target(registry["targets"][7])
    base = seed_base(int(registry["master_seed"]), 7, 0, 0)
    profile = PROFILES["smoke"]
    atlas = _make_atlas(target, profile, base)
    model = TanhMLP(target, "missing", base + 1)
    m = 128
    b = 32
    latent = np.random.default_rng(base + 2).normal(
        size=(m, model.latent_dim))
    table = randomized_systematic_target(
        atlas, m, np.random.default_rng(base + 38))
    _, field, exact_gradient = full_virtual_table_gradient(
        model, latent, table)
    exact = _flatten_gradient(model, exact_gradient)
    stratified = np.empty((reps, len(exact)))
    uniform = np.empty_like(stratified)
    uniform_rng = np.random.default_rng(base + 41)
    for rep in range(reps):
        _, _, _, gradient = stratified_virtual_table_gradient(
            model, latent, table, atlas, b,
            np.random.default_rng(base + 50 + rep))
        stratified[rep] = _flatten_gradient(model, gradient)
        chosen = uniform_rng.choice(m, size=b, replace=False)
        _, cache = model.forward(latent[chosen], want_cache=True)
        uniform_gradient = _stopgrad_grads(
            model, cache, field[chosen], b)
        uniform[rep] = _flatten_gradient(model, uniform_gradient)
    stratified_mean = stratified.mean(axis=0)
    error_norm = float(np.linalg.norm(stratified_mean - exact))
    se_norm = float(np.sqrt(np.sum(
        np.var(stratified, axis=0, ddof=1) / reps)))
    stratified_mse = float(np.mean(np.sum(
        (stratified - exact[None, :]) ** 2, axis=1)))
    uniform_mse = float(np.mean(np.sum(
        (uniform - exact[None, :]) ** 2, axis=1)))
    return {
        "virtual_batch": m, "backward_batch": b, "repetitions": reps,
        "gradient_dimension": len(exact), "mean_error_norm": error_norm,
        "monte_carlo_se_norm": se_norm,
        "error_over_se": error_norm / max(se_norm, 1e-300),
        "stratified_gradient_mse": stratified_mse,
        "uniform_gradient_mse": uniform_mse,
        "stratified_over_uniform_mse":
            stratified_mse / max(uniform_mse, 1e-300),
        "bias_audit_passed": error_norm <= 4.0 * se_norm,
    }


def run_atlas_validation(registry: dict[str, Any], profile: Profile,
                         seeds: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    controls = {"overlap", "connected", "heavy-tail"}
    for ti, spec in enumerate(registry["targets"]):
        target = build_target(spec)
        exact_family = target.family in {
            "unequal-tail", "unequal-interior", "equal", "contaminated"}
        expected_regions = (len(target.means) if exact_family else
                            (1 if target.family in controls else 0))
        for seed in range(seeds):
            base = seed_base(int(registry["master_seed"]), ti, 0, seed)
            atlas = _make_atlas(target, profile, base)
            expected_control = target.family in controls
            rows.append({
                "target": target.name, "family": target.family,
                "seed": seed, "atlas_samples": profile.atlas_samples,
                "region_count": atlas.region_count,
                "boundary_count": len(atlas.boundaries),
                "raw_candidate_count": atlas.raw_candidate_count,
                "unresolved": int(atlas.unresolved),
                "minimum_mass": min(atlas.masses),
                "maximum_persistence": max(
                    (e.bootstrap_persistence for e in atlas.evidence),
                    default=0.0),
                "control_false_positive": int(
                    expected_control and atlas.region_count > 1),
                "expected_regions": expected_regions,
                "exact_region_recovery": int(
                    expected_regions > 0 and
                    atlas.region_count == expected_regions),
            })

    rng = np.random.default_rng(20260920)
    synthetic: list[tuple[str, np.ndarray, bool]] = [
        ("synthetic-normal", rng.normal(size=(profile.atlas_samples, 1)), True),
        ("synthetic-lognormal", rng.lognormal(
            0.0, 0.7, size=(profile.atlas_samples, 1)), True),
        ("synthetic-student3p2", rng.standard_t(
            3.2, size=(profile.atlas_samples, 1)), True),
    ]
    for mass in (0.05, 0.02, 0.01, 0.005, 0.0025):
        rare_count = max(1, int(round(profile.atlas_samples * mass)))
        common_count = profile.atlas_samples - rare_count
        sample = np.concatenate([
            rng.normal(0.0, 0.08, size=(common_count, 1)),
            rng.normal(3.0, 0.04, size=(rare_count, 1)),
        ])
        synthetic.append((f"synthetic-rare-{mass:g}", sample, False))
    for index, (name, sample, control) in enumerate(synthetic):
        atlas = build_quantile_atlas(
            sample, np.random.default_rng(20261000 + index),
            config=atlas_config(32))
        rows.append({
            "target": name, "family": "synthetic-control" if control
            else "synthetic-rare", "seed": 0,
            "atlas_samples": len(sample),
            "region_count": atlas.region_count,
            "boundary_count": len(atlas.boundaries),
            "raw_candidate_count": atlas.raw_candidate_count,
            "unresolved": int(atlas.unresolved),
            "minimum_mass": min(atlas.masses),
            "maximum_persistence": max(
                (e.bootstrap_persistence for e in atlas.evidence),
                default=0.0),
            "control_false_positive": int(control and atlas.region_count > 1),
            "expected_regions": 1 if control else 2,
            "exact_region_recovery": int(
                atlas.region_count == (1 if control else 2)),
        })
    control_rows = [row for row in rows if row["family"] in
                    controls | {"synthetic-control"}]
    rare_rows = [row for row in rows if row["family"] == "synthetic-rare"]
    separated_rows = [row for row in rows if row["family"] in {
        "unequal-tail", "unequal-interior", "equal", "contaminated"}]
    summary = {
        "program": "OASQD-development-v1", "stage": "o1",
        "rows": len(rows),
        "control_false_positive_rate": float(np.mean([
            row["control_false_positive"] for row in control_rows])),
        "synthetic_rare_detection_rate": float(np.mean([
            row["region_count"] > 1 for row in rare_rows])),
        "separated_exact_region_rate": float(np.mean([
            row["exact_region_recovery"] for row in separated_rows])),
        "synthetic_rare_results": {
            row["target"]: {
                "region_count": row["region_count"],
                "minimum_mass": row["minimum_mass"],
                "unresolved": row["unresolved"],
            } for row in rare_rows
        },
        "gate": {
            "no_control_false_positives": all(
                row["control_false_positive"] == 0 for row in control_rows),
            "detects_rare_mass_at_least_0.005": all(
                row["region_count"] > 1 for row in rare_rows
                if not row["target"].endswith("0.0025")),
            "recovers_declared_separated_regions": all(
                row["exact_region_recovery"] == 1
                for row in separated_rows),
        },
    }
    summary["gate_passed"] = all(summary["gate"].values())
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: tuple(
            str(row.get(key, "")) for key in
            ("target", "init", "arm", "seed"))))


def write_results_md(path: Path, summary: dict[str, Any]) -> None:
    stage = summary["stage"].upper()
    lines = [
        f"# OA-SQD {stage} development results", "",
        "**Development evidence only; this is not a confirmatory claim.**",
        "",
    ]
    if summary["stage"] == "o1":
        lines.extend([
            f"- control false-positive rate: "
            f"`{summary['control_false_positive_rate']:.4f}`",
            f"- synthetic rare detection rate: "
            f"`{summary['synthetic_rare_detection_rate']:.4f}`",
            f"- separated exact-region rate: "
            f"`{summary['separated_exact_region_rate']:.4f}`",
            f"- O1 gate: **{'PASS' if summary['gate_passed'] else 'FAIL'}**",
            "", "## Gate", "",
        ])
        lines.extend(f"- {key}: `{value}`"
                     for key, value in summary["gate"].items())
    else:
        best = summary["best_candidate"]
        lines.extend([
            f"- best OA candidate: `{best}`",
            f"- candidate / paper ED2: "
            f"`{summary['ed2_ratios'][best]['paper-0.5']:.4f}`"
            if best else "- no OA candidate",
            f"- candidate / QLD-v1 ED2: "
            f"`{summary['ed2_ratios'][best]['qld-v1']:.4f}`"
            if best else "",
            f"- candidate / fixed LB-QCD ED2: "
            f"`{summary['ed2_ratios'][best]['fixed-lbqcd']:.4f}`"
            if best else "",
            f"- stage gate: **{'PASS' if summary['gate_passed'] else 'FAIL'}**",
            "", "## Advancement gate", "",
        ])
        lines.extend(f"- {key}: `{value}`"
                     for key, value in summary["advancement_gate"].items())
        if best:
            lines.extend(["", "## Best-candidate details", "", "```json",
                          json.dumps(summary["best_details"], indent=2),
                          "```"])
    path.write_text("\n".join(line for line in lines if line is not None) +
                    "\n", encoding="utf-8")


def make_run_dir(stage: str, profile: Profile) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNROOT / f"{stamp}-{stage}-{profile.name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_manifest(path: Path, args: argparse.Namespace, profile: Profile,
                   arms: list[Arm], task_count: int,
                   wall_seconds: float) -> None:
    sources = [Path(__file__), HERE / "oasqd.py", REGISTRY, PROTOCOL,
               RESEARCH, HERE / "lbqcd.py", HERE / "identifiability_drift.py",
               HERE / "run_identifiability_generator.py",
               HERE / "run_qld_confirmatory.py", HERE / "lowdim_drift.py"]
    manifest = {
        "program": "OASQD-development-v1", "stage": args.stage,
        "profile": asdict(profile), "arms": [asdict(arm) for arm in arms],
        "task_count": task_count, "workers": args.workers,
        "registry_sha256": sha256_file(REGISTRY),
        "commit": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--porcelain").splitlines(),
        "python": sys.version, "numpy": np.__version__,
        "platform": platform.platform(), "wall_seconds": wall_seconds,
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
    parser.add_argument("--stage", choices=("o1", "o2", "o3", "o4", "o5"),
                        default="o1")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="screen")
    parser.add_argument("--check-periods", default="25")
    parser.add_argument("--pulse-lengths", default="25")
    parser.add_argument("--probe-sizes", default="2048")
    parser.add_argument("--target-counts", default="8")
    parser.add_argument("--backward-batch", type=int, default=128)
    parser.add_argument("--inits", default="missing,concentrated")
    parser.add_argument("--targets", default="")
    parser.add_argument("--arms", default="")
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--workers", type=int,
                        default=min(8, os.cpu_count() or 1))
    parser.add_argument("--gradient-audit-reps", type=int, default=400)
    parser.add_argument("--self-check-only", action="store_true")
    parser.add_argument("--reanalyze", type=Path)
    args = parser.parse_args()

    lbqcd_invariant_tests()
    oasqd_invariant_tests()
    registry = load_registry()
    if args.self_check_only:
        audit = gradient_audit(
            registry, reps=max(50, args.gradient_audit_reps))
        if not audit["bias_audit_passed"]:
            raise AssertionError(f"stratified gradient audit failed: {audit}")
        print(json.dumps(audit, indent=2))
        print("OA-SQD invariants, registry, and gradient audit: PASS")
        return

    if args.reanalyze is not None:
        run_dir = args.reanalyze
        with (run_dir / "rows.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8"))
        stage = str(manifest["stage"])
        if stage == "o1":
            raise RuntimeError("O1 reanalysis uses its stored summary")
        summary = summarize(rows, stage, PROFILES[
            str(manifest["profile"]["name"])].bootstrap_reps)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        write_results_md(run_dir / "RESULTS.md", summary)
        print(json.dumps(summary, indent=2))
        return

    profile = PROFILES[args.profile]
    seeds = args.seeds if args.seeds is not None else profile.seeds
    run_dir = make_run_dir(args.stage, profile)
    wall0 = time.perf_counter()
    if args.stage == "o1":
        rows, summary = run_atlas_validation(registry, profile, seeds)
        arms: list[Arm] = []
        audit = None
    else:
        arms = make_arms(
            args.stage,
            check_periods=_parse_numbers(args.check_periods, int),
            pulse_lengths=_parse_numbers(args.pulse_lengths, int),
            probe_sizes=_parse_numbers(args.probe_sizes, int),
            target_counts=_parse_numbers(args.target_counts, float),
            backward_batch=args.backward_batch)
        arm_filter = set(_parse_numbers(args.arms, str))
        if arm_filter:
            arms = [arm for arm in arms if arm.label in arm_filter]
            missing = arm_filter - {arm.label for arm in arms}
            if missing:
                parser.error(f"unknown or unavailable arms: {sorted(missing)}")
        inits = _parse_numbers(args.inits, str)
        targets = _parse_numbers(args.targets, str)
        tasks = make_tasks(registry, profile, arms, inits, seeds, targets)
        print(f"run directory: {run_dir}", flush=True)
        print(f"registry sha256: {sha256_file(REGISTRY)}", flush=True)
        print(f"tasks: {len(tasks)}", flush=True)
        rows = execute(tasks, max(1, args.workers))
        summary = summarize(rows, args.stage, profile.bootstrap_reps)
        audit = gradient_audit(registry, reps=args.gradient_audit_reps) \
            if args.stage in ("o3", "o4", "o5") else None
        if audit is not None and not audit["bias_audit_passed"]:
            raise AssertionError(f"stratified gradient audit failed: {audit}")
    wall = time.perf_counter() - wall0
    write_csv(run_dir / "rows.csv", rows)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    if audit is not None:
        (run_dir / "gradient_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8")
    write_results_md(run_dir / "RESULTS.md", summary)
    write_manifest(run_dir, args, profile, arms, len(rows), wall)
    print(json.dumps(summary, indent=2), flush=True)
    if audit is not None:
        print(json.dumps(audit, indent=2), flush=True)
    print(f"completed in {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
