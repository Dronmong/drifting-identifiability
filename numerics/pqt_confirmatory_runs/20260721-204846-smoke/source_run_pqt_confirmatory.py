"""Frozen confirmatory runner for Persistent Quantile Transport.

The protocol and registry hashes below were fixed before any confirmatory
outcomes were generated.  The runner compares the primary PQT candidate with
the selected paper field, a five-bandwidth paper oracle, QLD-v1, and gated
LB-QCD on paired target/initialization/seed cells.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
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

from lbqcd import invariant_tests as lbqcd_invariant_tests  # noqa: E402
from persistent_quantile_transport import (  # noqa: E402
    invariant_tests as pqt_invariant_tests,
)
from run_lbqcd_development import (  # noqa: E402
    Arm,
    Profile,
    run_trial as run_baseline_trial,
    seed_base,
)
from run_qld_confirmatory import build_target  # noqa: E402
from run_transport_aligned_development import (  # noqa: E402
    LBQCD,
    PAPER,
    QLD,
    run_trial as run_pqt_trial,
    summarize as summarize_transport,
)


PROTOCOL_ID = "PQT-confirmatory-v1"
REGISTRY = HERE / "pqt_confirmatory_registry.json"
PROTOCOL = HERE / "PQTConfirmatoryProtocol.md"
RUNROOT = HERE / "pqt_confirmatory_runs"
REGISTRY_SHA256 = \
    "E4B914DE6B94BB8359CCC3C3DF86E731EBE5F22855E7A6D311202D84B46DE755"
PROTOCOL_SHA256 = \
    "86D8D567B3B43B2A482527181DF95AAA4645FC0715FB81A371937BA71E4F9034"
PAPER_TAUS = (0.2, 0.5, 1.0, 2.0, 4.0)
INITS = ("missing", "concentrated", "far")
PRIMARY = "pqt-gated-M1024-f0.70-K128"
SECONDARY = "pqt-B128-K128"
INTERNAL_PRIMARY = "gated-pqt-M1024-f0.70"
INTERNAL_SECONDARY = "pqt-B128"

PROFILES = {
    "smoke": Profile("smoke", 30, 32, 2, 512, 256, 32),
    "standard": Profile("standard", 1200, 128, 20, 4096, 1024, 256),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_registry() -> dict[str, Any]:
    actual_registry = sha256_file(REGISTRY)
    actual_protocol = sha256_file(PROTOCOL)
    if actual_registry != REGISTRY_SHA256:
        raise RuntimeError(
            f"registry hash mismatch: {actual_registry} != {REGISTRY_SHA256}")
    if actual_protocol != PROTOCOL_SHA256:
        raise RuntimeError(
            f"protocol hash mismatch: {actual_protocol} != {PROTOCOL_SHA256}")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("registry") != PROTOCOL_ID:
        raise RuntimeError("unexpected PQT confirmatory registry ID")
    targets = registry.get("targets", [])
    if len(targets) != 20:
        raise RuntimeError("PQT confirmatory registry must contain 20 targets")
    names = [str(spec["name"]) for spec in targets]
    if len(set(names)) != len(names):
        raise RuntimeError("PQT confirmatory target names are not unique")
    for spec in targets:
        build_target(spec)
    return registry


def baseline_arms() -> list[Arm]:
    arms = [Arm(f"paper-{tau:g}", "paper", tau, 0, 0.0, "update")
            for tau in PAPER_TAUS]
    arms.extend([
        Arm(QLD, "qld", 0.5, 0, 0.70, "update"),
        Arm(LBQCD, "gated", 0.5, 1024, 0.70, "update"),
    ])
    return arms


def make_tasks(registry: dict[str, Any], profile: Profile,
               targets: list[dict[str, Any]], inits: tuple[str, ...],
               seeds: int) -> list[dict[str, Any]]:
    target_indices = {str(spec["name"]): i
                      for i, spec in enumerate(registry["targets"])}
    tasks: list[dict[str, Any]] = []
    for spec in targets:
        ti = target_indices[str(spec["name"])]
        for ii, init in enumerate(INITS):
            if init not in inits:
                continue
            for seed in range(seeds):
                base = seed_base(
                    int(registry["master_seed"]), ti, ii, seed)
                for arm in baseline_arms():
                    tasks.append({
                        "engine": "baseline",
                        "target_spec": spec,
                        "profile": asdict(profile),
                        "arm": asdict(arm),
                        "init": init,
                        "seed": seed,
                        "candidate_taus": PAPER_TAUS,
                        "diagnostic_samples": 4096,
                        "base_seed": base,
                    })
                for internal in (INTERNAL_SECONDARY, INTERNAL_PRIMARY):
                    tasks.append({
                        "engine": "pqt",
                        "target_spec": spec,
                        "profile": asdict(profile),
                        "arm": internal,
                        "init": init,
                        "seed": seed,
                        "base_seed": base,
                        "knot_count": 128,
                        "initialization_samples": 4096,
                        "prior_batches": 1.0,
                        "diagnostic_samples": 4096,
                    })
    return tasks


def run_task(task: dict[str, Any]) -> dict[str, Any]:
    payload = dict(task)
    engine = str(payload.pop("engine"))
    if engine == "baseline":
        return run_baseline_trial(payload)
    if engine != "pqt":
        raise ValueError(f"unknown trial engine {engine}")
    row = run_pqt_trial(payload)
    if row["arm"] == INTERNAL_PRIMARY:
        row["arm"] = PRIMARY
    elif row["arm"] == INTERNAL_SECONDARY:
        row["arm"] = SECONDARY
    else:
        raise AssertionError("unexpected internal PQT arm")
    return row


def execute(tasks: list[dict[str, Any]], workers: int) \
        -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % 100 == 0 or done == len(tasks):
                elapsed = time.perf_counter() - started
                print(f"  {done}/{len(tasks)} trials ({elapsed:.1f}s)",
                      flush=True)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty confirmatory rows")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (
            str(row["arm"]), str(row["target"]), str(row["init"]),
            int(row["seed"]))))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


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
    return {key: _positive_median(values)
            for key, values in grouped.items()}


def verify_complete(rows: list[dict[str, Any]], registry: dict[str, Any],
                    profile: Profile) -> None:
    targets = {str(spec["name"]) for spec in registry["targets"]}
    arms = {arm.label for arm in baseline_arms()} | {PRIMARY, SECONDARY}
    expected = {
        (target, init, arm, seed)
        for target in targets for init in INITS for arm in arms
        for seed in range(profile.seeds)
    }
    actual = {
        (str(row["target"]), str(row["init"]), str(row["arm"]),
         int(row["seed"]))
        for row in rows
    }
    if actual != expected:
        raise RuntimeError(
            f"incomplete confirmatory rows: missing={len(expected-actual)}, "
            f"extra={len(actual-expected)}")


def summarize(rows: list[dict[str, Any]], registry: dict[str, Any],
              bootstrap_reps: int) -> dict[str, Any]:
    candidates = [row for row in rows
                  if str(row["arm"]) in (PRIMARY, SECONDARY)]
    baselines = [row for row in rows
                 if str(row["arm"]) not in (PRIMARY, SECONDARY)]
    base_summary = summarize_transport(
        candidates, baselines, bootstrap_reps=bootstrap_reps,
        bootstrap_seed=int(registry["master_seed"]) + 9001)
    med_ed = _cell_medians(rows, "ed2")
    targets = sorted({str(row["target"]) for row in rows})
    cells = [(target, init) for target in targets for init in INITS]
    paper_arms = [f"paper-{tau:g}" for tau in PAPER_TAUS]
    oracle = {
        (target, init): min(
            med_ed[(target, init, arm)] for arm in paper_arms)
        for target, init in cells
    }
    for arm in (PRIMARY, SECONDARY):
        base_summary["comparisons"][arm]["ed2_vs_paper_oracle"] = _geomean(
            med_ed[(target, init, arm)] / oracle[(target, init)]
            for target, init in cells)

    primary = base_summary["comparisons"][PRIMARY]
    primary_rows = [row for row in rows if str(row["arm"]) == PRIMARY]
    incumbent_rows = [row for row in rows if str(row["arm"]) == LBQCD]
    incumbent_lookup = {
        (str(row["target"]), str(row["init"]), int(row["seed"])): row
        for row in incumbent_rows
    }
    budget_equal = True
    routing_equal = True
    for row in primary_rows:
        key = (str(row["target"]), str(row["init"]), int(row["seed"]))
        incumbent = incumbent_lookup[key]
        budget_equal &= int(float(row["target_samples"])) == int(float(
            incumbent["target_samples"]))
        routing_equal &= int(float(row["diagnostic_use_large_batch"])) == int(
            float(incumbent["diagnostic_use_large_batch"]))
    candidate_divergences = sum(int(float(row["diverged"]))
                                for row in primary_rows)
    incumbent_divergences = sum(int(float(row["diverged"]))
                                for row in incumbent_rows)
    ed_ci = primary[f"ed2_ci_vs_{LBQCD}"]
    sw_ci = primary[f"sw1_ci_vs_{LBQCD}"]
    family = primary["family_ed2_vs_lbqcd"]
    init = primary["init_ed2_vs_lbqcd"]
    checks = {
        "ed2_vs_lbqcd_at_most_0.80":
            primary[f"ed2_vs_{LBQCD}"] <= 0.80,
        "ed2_bootstrap_upper_vs_lbqcd_below_1": ed_ci[1] < 1.0,
        "sw1_vs_lbqcd_at_most_0.85":
            primary[f"sw1_vs_{LBQCD}"] <= 0.85,
        "sw1_bootstrap_upper_vs_lbqcd_below_1": sw_ci[1] < 1.0,
        "ed2_vs_selected_paper_at_most_0.70":
            primary[f"ed2_vs_{PAPER}"] <= 0.70,
        "ed2_vs_paper_oracle_at_most_0.85":
            primary["ed2_vs_paper_oracle"] <= 0.85,
        "cell_win_fraction_vs_lbqcd_at_least_0.70":
            primary["cell_win_fraction_vs_lbqcd"] >= 0.70,
        "all_family_ed2_ratios_vs_lbqcd_at_most_1.10":
            all(value <= 1.10 for value in family.values()),
        "all_init_ed2_ratios_vs_lbqcd_below_1":
            all(value < 1.0 for value in init.values()),
        "far_ed2_ratio_vs_lbqcd_at_most_0.80": init["far"] <= 0.80,
        "divergence_no_worse_than_lbqcd":
            candidate_divergences <= incumbent_divergences,
        "target_sample_budget_equal": budget_equal,
        "routing_decisions_equal": routing_equal,
    }
    return {
        "protocol": PROTOCOL_ID,
        "registry_sha256": REGISTRY_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "primary_candidate": PRIMARY,
        "comparisons": base_summary["comparisons"],
        "candidate_divergences": candidate_divergences,
        "incumbent_divergences": incumbent_divergences,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
    }


def render_results(summary: dict[str, Any]) -> str:
    primary = summary["comparisons"][PRIMARY]
    gate = summary["gate_checks"]
    lines = [
        "# Persistent Quantile Transport: frozen confirmatory results", "",
        f"**Protocol:** `{PROTOCOL_ID}`  ",
        f"**Verdict:** **{'PASS' if summary['gate_passed'] else 'FAIL'}**", "",
        "## Primary comparisons", "",
        "| comparison | ED2 ratio | 95% interval | SW1 ratio | 95% interval |",
        "|---|---:|---:|---:|---:|",
    ]
    for baseline, label in ((PAPER, "selected paper"), (QLD, "QLD-v1"),
                            (LBQCD, "gated LB-QCD")):
        lines.append(
            f"| PQT / {label} | {primary[f'ed2_vs_{baseline}']:.4f} | "
            f"`[{primary[f'ed2_ci_vs_{baseline}'][0]:.4f},"
            f"{primary[f'ed2_ci_vs_{baseline}'][1]:.4f}]` | "
            f"{primary[f'sw1_vs_{baseline}']:.4f} | "
            f"`[{primary[f'sw1_ci_vs_{baseline}'][0]:.4f},"
            f"{primary[f'sw1_ci_vs_{baseline}'][1]:.4f}]` |")
    lines += [
        "",
        f"ED2 ratio versus the per-cell paper oracle: "
        f"`{primary['ed2_vs_paper_oracle']:.4f}`.", "",
        "## Conjunctive gate", "",
    ]
    for name, passed in gate.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'}: `{name}`")
    lines += ["", "## Family ED2 ratios versus LB-QCD", ""]
    for family, value in primary["family_ed2_vs_lbqcd"].items():
        lines.append(f"- `{family}`: `{value:.4f}`")
    lines += ["", "## Initialization ED2 ratios versus LB-QCD", ""]
    for init, value in primary["init_ed2_vs_lbqcd"].items():
        lines.append(f"- `{init}`: `{value:.4f}`")
    lines += [
        "", "## Scope", "",
        "This verdict concerns the predeclared one-dimensional synthetic "
        "registry and its three initialization regimes. It does not establish "
        "higher-dimensional, encoder-feature, or image-generation superiority.",
        "",
    ]
    return "\n".join(lines)


def snapshot(run_dir: Path, args: argparse.Namespace,
             task_count: int, wall_seconds: float) -> None:
    sources = (
        Path(__file__), HERE / "persistent_quantile_transport.py",
        HERE / "run_transport_aligned_development.py",
        HERE / "run_lbqcd_development.py", HERE / "lbqcd.py",
        HERE / "run_identifiability_generator.py",
        HERE / "identifiability_drift.py", REGISTRY, PROTOCOL,
    )
    hashes: dict[str, str] = {}
    for source in sources:
        hashes[source.name] = sha256_file(source)
        shutil.copy2(source, run_dir / f"source_{source.name}")
    manifest = {
        "protocol": PROTOCOL_ID,
        "profile": args.profile,
        "workers": args.workers,
        "tasks": task_count,
        "bootstrap_reps": args.bootstrap_reps,
        "wall_seconds": wall_seconds,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "argv": sys.argv,
        "source_sha256": hashes,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=tuple(PROFILES), default="smoke")
    parser.add_argument("--workers", type=int,
                        default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--bootstrap-reps", type=int, default=10_000)
    parser.add_argument("--targets", default="")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--reanalyze", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = load_registry()
    lbqcd_invariant_tests()
    pqt_invariant_tests()
    profile = PROFILES[args.profile]
    if args.reanalyze is not None:
        if args.profile != "standard":
            raise RuntimeError("confirmatory reanalysis requires --profile standard")
        run_dir = args.reanalyze.resolve()
        rows = read_csv(run_dir / "rows.csv")
        verify_complete(rows, registry, profile)
        summary = summarize(rows, registry, args.bootstrap_reps)
        (run_dir / "results.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        (run_dir / "RESULTS.md").write_text(
            render_results(summary), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    selected_names = tuple(
        item.strip() for item in args.targets.split(",") if item.strip())
    if args.profile == "standard":
        if not args.confirm:
            raise RuntimeError("standard confirmatory run requires --confirm")
        if selected_names:
            raise RuntimeError("standard confirmatory run cannot filter targets")
    targets = list(registry["targets"])
    inits = INITS
    seeds = profile.seeds
    if selected_names:
        targets = [spec for spec in targets
                   if str(spec["name"]) in selected_names or
                   str(spec["family"]) in selected_names]
        if not targets:
            raise ValueError("target filter selected no targets")
    elif args.profile == "smoke":
        targets = targets[:2]
        inits = ("missing",)
    tasks = make_tasks(registry, profile, targets, inits, seeds)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNROOT / f"{stamp}-{args.profile}"
    run_dir.mkdir(parents=True, exist_ok=False)
    wall0 = time.perf_counter()
    print(f"Running {len(tasks)} frozen PQT trials with {args.workers} workers")
    rows = execute(tasks, args.workers)
    wall_seconds = time.perf_counter() - wall0
    write_csv(run_dir / "rows.csv", rows)
    if args.profile == "standard":
        verify_complete(rows, registry, profile)
        summary = summarize(rows, registry, args.bootstrap_reps)
        (run_dir / "results.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        (run_dir / "RESULTS.md").write_text(
            render_results(summary), encoding="utf-8")
    else:
        summary = {"smoke": "PASS", "rows": len(rows)}
        (run_dir / "results.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
    snapshot(run_dir, args, len(tasks), wall_seconds)
    print(f"Artifacts: {run_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

