"""Development runner for Persistent Quantile Transport (PQT).

The runner evaluates architecture-level candidates on the mutable LB-QCD
registry and compares them with archived, paired paper/QLD/LB-QCD rows.  It
never edits or reinterprets a frozen confirmatory artifact.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lbqcd import diagnose_quantile_resolution_stable  # noqa: E402
from lowdim_drift import energy_distance2, sliced_w1  # noqa: E402
from persistent_quantile_transport import (  # noqa: E402
    PersistentQuantileTransport,
    invariant_tests,
    midpoint_grid,
)
from run_identifiability_generator import TanhMLP  # noqa: E402
from run_lbqcd_development import (  # noqa: E402
    PROFILES as DEVELOPMENT_PROFILES,
    Profile,
    load_registry as load_development_registry,
    seed_base,
)
from run_lbqcd_confirmatory import (  # noqa: E402
    PROFILES as REPLICATION_PROFILES,
    load_registry as load_replication_registry,
)
from run_qld_confirmatory import (  # noqa: E402
    Target1D,
    build_target,
    target_diagnostics,
)


RUNROOT = HERE / "transport_aligned_runs"
PAPER = "paper-0.5"
QLD = "qld-v1"
LBQCD = "gated-M1024-f0.70-u"
ARMS = (
    "pqt-B128",
    "gated-pqt-M1024-f0.70",
    "gated-pqt-M1024",
)
BASELINE_ROWS = {
    ("development", "screen"):
        HERE / "lbqcd_runs/20260720-234146-n1-screen/rows.csv",
    ("development", "standard"):
        HERE / "lbqcd_runs/20260720-234857-n1-standard/rows.csv",
    ("replication", "standard"):
        HERE / "lbqcd_confirmatory_runs/20260721-000900-test-standard/rows.csv",
}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty row table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def initial_quantile_map(target: Target1D, init_kind: str, seed: int,
                         knot_count: int, sample_count: int,
                         prior_batches: float) \
        -> PersistentQuantileTransport:
    """Match the marginal distribution of the repository's TanhMLP init."""
    source = TanhMLP(target, init_kind, seed)
    rng = np.random.default_rng(seed + 200_003)
    latent = rng.normal(size=(sample_count, source.latent_dim))
    initial = source.forward(latent)
    return PersistentQuantileTransport.from_ordered_initial_sample(
        knot_count, initial, prior_batches=prior_batches)


def run_trial(task: dict[str, Any]) -> dict[str, Any]:
    target = build_target(task["target_spec"])
    profile = Profile(**task["profile"])
    arm = str(task["arm"])
    init_kind = str(task["init"])
    seed = int(task["seed"])
    base = int(task["base_seed"])
    knot_count = int(task["knot_count"])
    init_samples = int(task["initialization_samples"])
    prior_batches = float(task["prior_batches"])
    diagnostic_samples = int(task["diagnostic_samples"])
    wall0 = time.perf_counter()

    model = initial_quantile_map(
        target, init_kind, base + 1, knot_count, init_samples, prior_batches)
    data_rng = np.random.default_rng(base + 3)
    diagnosis = None
    if arm.startswith("gated-pqt-M1024"):
        diagnostic = target.sample(
            diagnostic_samples, np.random.default_rng(base + 34))
        diagnosis = diagnose_quantile_resolution_stable(
            diagnostic, profile.batch)
    routed = bool(diagnosis is not None and diagnosis.use_large_batch)
    maximum_training_batch = 1024 if routed else profile.batch
    target_samples = 0
    sort_work = 0.0
    large_batch_updates = 0
    event_time: int | None = None
    diverged = False

    # Event diagnostics use untouched uniforms and do not update the model.
    event_rng = np.random.default_rng(base + 30_001)
    for step in range(1, profile.steps + 1):
        use_large_batch = routed and (
            arm == "gated-pqt-M1024" or
            (arm == "gated-pqt-M1024-f0.70" and
             step <= int(round(0.70 * profile.steps))))
        training_batch = 1024 if use_large_batch else profile.batch
        large_batch_updates += int(use_large_batch)
        positive = target.sample(training_batch, data_rng)
        try:
            work = model.update(positive)
        except (FloatingPointError, ValueError, AssertionError):
            diverged = True
            break
        target_samples += work.target_samples
        sort_work += work.sort_work
        if event_time is None and (step == 1 or step % 50 == 0):
            probe = model.forward(event_rng.random((profile.batch, 1)))
            reach, _ = target_diagnostics(probe, target)
            if reach >= 0.90:
                event_time = step

    wall_seconds = time.perf_counter() - wall0
    common = {
        "arm": arm,
        "kind": "persistent-quantile-transport",
        "target": target.name,
        "family": target.family,
        "init": init_kind,
        "cell": f"{target.name}/{init_kind}",
        "seed": seed,
        "planned_steps": profile.steps,
        "completed_steps": model.updates,
        "maximum_training_batch": maximum_training_batch,
        "large_batch_updates": large_batch_updates,
        "ordinary_batch_updates": model.updates - large_batch_updates,
        "knot_count": knot_count,
        "prior_batches": prior_batches,
        "diagnostic_target_samples": diagnostic_samples if diagnosis else 0,
        "diagnostic_use_large_batch": int(routed),
        "diagnostic_min_expected_count": (
            diagnosis.minimum_expected_batch_count if diagnosis else float("nan")),
        "tail_mass": model.tail_mass,
        "wall_seconds": wall_seconds,
        "generator_example_evals": init_samples + profile.eval_size,
        "unique_latent_samples": init_samples + profile.eval_size,
        "target_samples": target_samples,
        "kernel_pairs": 0,
        "sort_work": sort_work,
        "stored_scalars": len(model.grid) * 2,
    }
    if diverged:
        return {
            **common,
            "ed2": float("inf"), "sw1": float("inf"),
            "w2_quantile_rmse": float("inf"),
            "weighted_reach": float("nan"), "mass_l1": float("nan"),
            "event_time": profile.steps, "censored": 1, "diverged": 1,
        }

    q = model.forward(np.random.default_rng(base + 4).random(
        (profile.eval_size, 1)))
    p = target.sample(profile.eval_size, np.random.default_rng(base + 5))
    metric_rng = np.random.default_rng(base + 6)
    metric_n = min(profile.ed_size, len(q), len(p))
    iq = metric_rng.choice(len(q), metric_n, replace=False)
    ip = metric_rng.choice(len(p), metric_n, replace=False)
    ed2 = max(float(energy_distance2(q[iq], p[ip])), 0.0)
    sw1 = sliced_w1(q, p, 1, metric_rng)
    w2 = float(np.sqrt(np.mean(
        (np.sort(q[:, 0]) - np.sort(p[:, 0])) ** 2)))
    reach, mass_l1 = target_diagnostics(q, target)
    return {
        **common,
        "ed2": ed2, "sw1": sw1, "w2_quantile_rmse": w2,
        "weighted_reach": reach, "mass_l1": mass_l1,
        "event_time": event_time if event_time is not None else profile.steps,
        "censored": int(event_time is None), "diverged": 0,
    }


def make_tasks(registry: dict[str, Any], profile: Profile,
               arms: tuple[str, ...], seeds: int, inits: tuple[str, ...],
               knot_count: int, initialization_samples: int,
               prior_batches: float, diagnostic_samples: int,
               target_filter: tuple[str, ...]) -> list[dict[str, Any]]:
    all_targets = list(registry["targets"])
    selected = [spec for spec in all_targets if not target_filter or
                str(spec["name"]) in target_filter or
                str(spec["family"]) in target_filter]
    if not selected:
        raise ValueError("target filter selected no targets")
    target_indices = {str(spec["name"]): i for i, spec in enumerate(all_targets)}
    tasks: list[dict[str, Any]] = []
    for spec in selected:
        ti = target_indices[str(spec["name"])]
        for ii, init_kind in enumerate(inits):
            for arm in arms:
                for seed in range(seeds):
                    tasks.append({
                        "target_spec": spec,
                        "profile": asdict(profile),
                        "arm": arm,
                        "init": init_kind,
                        "seed": seed,
                        "base_seed": seed_base(
                            int(registry["master_seed"]), ti, ii, seed),
                        "knot_count": knot_count,
                        "initialization_samples": initialization_samples,
                        "prior_batches": prior_batches,
                        "diagnostic_samples": diagnostic_samples,
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
            if done % 24 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} trials "
                      f"({time.perf_counter() - started:.1f}s)", flush=True)
    return rows


def _positive_median(values: Iterable[float]) -> float:
    return max(float(np.median(list(values))), 1e-12)


def _geomean(values: Iterable[float]) -> float:
    values = list(values)
    return float(math.exp(np.mean(np.log(np.maximum(values, 1e-300)))))


def _cell_medians(rows: list[dict[str, Any]], metric: str) \
        -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["target"]), str(row["init"]), str(row["arm"]))
        grouped.setdefault(key, []).append(float(row[metric]))
    return {key: _positive_median(values) for key, values in grouped.items()}


def validate_baseline_pairing(candidate_rows: list[dict[str, Any]],
                              baseline_rows: list[dict[str, Any]]) -> None:
    candidate_keys = {(str(r["target"]), str(r["init"]), int(r["seed"]))
                      for r in candidate_rows}
    for arm in (PAPER, QLD, LBQCD):
        keys = {(str(r["target"]), str(r["init"]), int(r["seed"]))
                for r in baseline_rows if str(r["arm"]) == arm}
        if keys != candidate_keys:
            missing = len(candidate_keys - keys)
            extra = len(keys - candidate_keys)
            raise RuntimeError(
                f"baseline pairing mismatch for {arm}: {missing=} {extra=}")


def hierarchical_target_bootstrap(
        rows: list[dict[str, Any]], candidate: str, baseline: str,
        metric: str, reps: int, seed: int) -> list[float]:
    """Paired target bootstrap with seed resampling inside each cell."""
    lookup = {
        (str(row["target"]), str(row["init"]), str(row["arm"]),
         int(row["seed"])): float(row[metric])
        for row in rows
        if str(row["arm"]) in (candidate, baseline)
    }
    targets = sorted({key[0] for key in lookup})
    inits = sorted({key[1] for key in lookup})
    seeds = sorted({key[3] for key in lookup})
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
    return [float(value) for value in np.quantile(
        statistics, [0.025, 0.975])]


def summarize(candidate_rows: list[dict[str, Any]],
              baseline_rows: list[dict[str, Any]], *,
              bootstrap_reps: int = 5000,
              bootstrap_seed: int = 20260831) -> dict[str, Any]:
    validate_baseline_pairing(candidate_rows, baseline_rows)
    all_rows = [*candidate_rows, *[
        row for row in baseline_rows if str(row["arm"]) in (PAPER, QLD, LBQCD)]]
    med_ed = _cell_medians(all_rows, "ed2")
    med_sw = _cell_medians(all_rows, "sw1")
    targets = sorted({str(row["target"]) for row in candidate_rows})
    inits = sorted({str(row["init"]) for row in candidate_rows})
    families = {str(row["target"]): str(row["family"])
                for row in candidate_rows}
    cells = [(target, init) for target in targets for init in inits]
    result: dict[str, Any] = {"comparisons": {}}
    for arm in sorted({str(row["arm"]) for row in candidate_rows}):
        comparisons: dict[str, Any] = {}
        for baseline in (PAPER, QLD, LBQCD):
            comparisons[f"ed2_vs_{baseline}"] = _geomean(
                med_ed[(target, init, arm)] / med_ed[(target, init, baseline)]
                for target, init in cells)
            comparisons[f"sw1_vs_{baseline}"] = _geomean(
                med_sw[(target, init, arm)] / med_sw[(target, init, baseline)]
                for target, init in cells)
            comparisons[f"ed2_ci_vs_{baseline}"] = \
                hierarchical_target_bootstrap(
                    all_rows, arm, baseline, "ed2", bootstrap_reps,
                    bootstrap_seed + 101 * len(comparisons))
            comparisons[f"sw1_ci_vs_{baseline}"] = \
                hierarchical_target_bootstrap(
                    all_rows, arm, baseline, "sw1", bootstrap_reps,
                    bootstrap_seed + 101 * len(comparisons) + 1)
        comparisons["family_ed2_vs_lbqcd"] = {
            family: _geomean(
                med_ed[(target, init, arm)] / med_ed[(target, init, LBQCD)]
                for target, init in cells if families[target] == family)
            for family in sorted(set(families.values()))
        }
        comparisons["init_ed2_vs_lbqcd"] = {
            init: _geomean(
                med_ed[(target, init, arm)] / med_ed[(target, init, LBQCD)]
                for target in targets)
            for init in inits
        }
        comparisons["cell_win_fraction_vs_lbqcd"] = float(np.mean([
            med_ed[(target, init, arm)] < med_ed[(target, init, LBQCD)]
            for target, init in cells]))
        arm_rows = [row for row in candidate_rows if str(row["arm"]) == arm]
        comparisons["median_work"] = {
            key: float(np.median([float(row[key]) for row in arm_rows]))
            for key in ("wall_seconds", "target_samples",
                        "generator_example_evals", "sort_work",
                        "stored_scalars", "w2_quantile_rmse")
        }
        comparisons["divergences"] = sum(
            int(row["diverged"]) for row in arm_rows)
        result["comparisons"][arm] = comparisons
    result["development_gate"] = {
        arm: {
            "ed2_vs_lbqcd_at_most_0.90":
                values["ed2_vs_gated-M1024-f0.70-u"] <= 0.90,
            "ed2_vs_paper_at_most_0.75":
                values["ed2_vs_paper-0.5"] <= 0.75,
            "sw1_improves_vs_lbqcd":
                values["sw1_vs_gated-M1024-f0.70-u"] < 1.0,
            "all_families_at_most_1.05_vs_lbqcd": all(
                ratio <= 1.05
                for ratio in values["family_ed2_vs_lbqcd"].values()),
            "both_inits_improve_vs_lbqcd": all(
                ratio < 1.0
                for ratio in values["init_ed2_vs_lbqcd"].values()),
            "no_divergences": values["divergences"] == 0,
        }
        for arm, values in result["comparisons"].items()
    }
    for checks in result["development_gate"].values():
        checks["passed"] = all(checks.values())
    return result


def render_results(summary: dict[str, Any], profile: str,
                   candidate_rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Persistent Quantile Transport: development results", "",
        f"**Profile:** `{profile}`  ",
        "**Status:** mutable mechanism development, not confirmatory evidence", "",
        "| arm | ED2 / paper | ED2 / QLD | ED2 / LB-QCD | SW1 / LB-QCD | cell wins | gate |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for arm, values in summary["comparisons"].items():
        gate = summary["development_gate"][arm]["passed"]
        lines.append(
            f"| {arm} | {values['ed2_vs_paper-0.5']:.4f} | "
            f"{values['ed2_vs_qld-v1']:.4f} | "
            f"{values['ed2_vs_gated-M1024-f0.70-u']:.4f} | "
            f"{values['sw1_vs_gated-M1024-f0.70-u']:.4f} | "
            f"{values['cell_win_fraction_vs_lbqcd']:.1%} | "
            f"{'PASS' if gate else 'FAIL'} |")
    lines += ["", "## Family ED2 ratios versus LB-QCD", ""]
    for arm, values in summary["comparisons"].items():
        lines.append(f"### {arm}")
        lines.append("")
        for family, ratio in values["family_ed2_vs_lbqcd"].items():
            lines.append(f"- `{family}`: `{ratio:.4f}`")
        lines.append("")
    lines += [
        "## Interpretation guardrail", "",
        "PQT is a one-dimensional nonparametric monotone generator. A favorable "
        "result tests persistent transport coordinates; it is not evidence for "
        "higher-dimensional or image generation. The matched `B128` arm is the "
        "architecture test. An improvement confined to gated `M1024` is a target-"
        "resolution result.", "",
        f"Candidate trial rows: `{len(candidate_rows)}`.", "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", choices=("development", "replication"),
                        default="development")
    parser.add_argument("--profile", choices=("smoke", "screen", "standard"),
                        default="screen")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--inits", default="missing,concentrated")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--workers", type=int,
                        default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--knots", type=int, default=1024)
    parser.add_argument("--initialization-samples", type=int, default=4096)
    parser.add_argument("--prior-batches", type=float, default=1.0)
    parser.add_argument("--diagnostic-samples", type=int, default=4096)
    parser.add_argument("--targets", default="")
    parser.add_argument("--baseline-rows", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--reanalyze-run", type=Path, default=None)
    parser.add_argument("--label", default="development")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invariant_tests()
    if args.registry == "development":
        registry = load_development_registry()
        profiles = DEVELOPMENT_PROFILES
        registry_source = HERE / "lbqcd_development_registry.json"
    else:
        registry = load_replication_registry()
        profiles = REPLICATION_PROFILES
        registry_source = HERE / "lbqcd_confirmatory_registry.json"
    if args.profile not in profiles:
        raise ValueError(
            f"profile {args.profile!r} is unavailable for {args.registry}")
    profile = profiles[args.profile]
    default_seeds = profile.seeds
    seeds = args.seeds if args.seeds is not None else default_seeds
    arms = tuple(item.strip() for item in args.arms.split(",") if item.strip())
    if not arms or not set(arms) <= set(ARMS):
        raise ValueError(f"arms must be drawn from {ARMS}")
    inits = tuple(item.strip() for item in args.inits.split(",") if item.strip())
    targets = tuple(item.strip() for item in args.targets.split(",") if item.strip())
    if args.reanalyze_run is None:
        tasks = make_tasks(
            registry, profile, arms, seeds, inits, args.knots,
            args.initialization_samples, args.prior_batches,
            args.diagnostic_samples, targets)
        print(f"Running {len(tasks)} PQT trials with {args.workers} workers")
        rows = execute(tasks, args.workers)
        rows.sort(key=lambda row: (
            str(row["arm"]), str(row["target"]), str(row["init"]), int(row["seed"])))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = RUNROOT / (
            f"{stamp}-{args.registry}-{args.profile}-{args.label}")
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_rows(run_dir / "candidate_rows.csv", rows)
    else:
        run_dir = args.reanalyze_run.resolve()
        rows = _read_rows(run_dir / "candidate_rows.csv")
        tasks = []
    baseline_path = args.baseline_rows or BASELINE_ROWS.get(
        (args.registry, args.profile))
    if baseline_path is not None and baseline_path.exists() and not targets and \
            seeds == default_seeds and set(inits) == {"missing", "concentrated"}:
        baseline = _read_rows(baseline_path)
        summary = summarize(
            rows, baseline, bootstrap_reps=args.bootstrap_reps,
            bootstrap_seed=int(registry["master_seed"]) + 9001)
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
        (run_dir / "RESULTS.md").write_text(
            render_results(summary, args.profile, rows), encoding="utf-8")
    else:
        summary = {"comparison_skipped": True,
                   "reason": "filtered/nonstandard run or no paired baseline"}
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
    if args.reanalyze_run is None:
        for source in (
                Path(__file__), HERE / "persistent_quantile_transport.py",
                HERE / "TransportAlignedGeneratorResearchPlan.md",
                registry_source):
            shutil.copy2(source, run_dir / f"source_{source.name}")
        manifest = {
            "program": "transport-aligned-generator-development-v1",
            "registry_role": args.registry,
            "registry_id": registry.get("registry"),
            "profile": args.profile,
            "arms": arms,
            "inits": inits,
            "seeds": seeds,
            "knots": args.knots,
            "initialization_samples": args.initialization_samples,
            "prior_batches": args.prior_batches,
            "diagnostic_samples": args.diagnostic_samples,
            "baseline_rows": str(baseline_path) if baseline_path else None,
            "tasks": len(rows),
            "bootstrap_reps": args.bootstrap_reps,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Artifacts: {run_dir}")
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
