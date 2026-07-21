"""Frozen confirmatory runner for resolution-gated LB-QCD."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lbqcd import invariant_tests  # noqa: E402
from run_lbqcd_development import (  # noqa: E402
    Arm,
    Profile,
    execute,
    run_trial,
    seed_base,
    sha256_file,
    summarize,
)
from run_qld_confirmatory import build_target  # noqa: E402


PROTOCOL_ID = "LBQCD-confirmatory-v1"
REGISTRY = HERE / "lbqcd_confirmatory_registry.json"
PROTOCOL = HERE / "LBQCDConfirmatoryProtocol.md"
RESEARCH = HERE / "LBQCDDevelopmentResults.md"
RUNROOT = HERE / "lbqcd_confirmatory_runs"
REGISTRY_SHA256 = \
    "C2A5F01048F732EC0574A70BBD249079E4E04250EAB6D522C3449D80DC310231"
PAPER_TAUS = (0.2, 0.5, 1.0, 2.0, 4.0)
BASELINE_TAU = 0.5
INITS = ("missing", "concentrated")
CANDIDATE = "gated-M1024-f0.70-u"

PROFILES = {
    "smoke": Profile("smoke", 30, 32, 2, 512, 256, 32),
    "standard": Profile("standard", 1200, 128, 20, 4096, 1024, 256),
}


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_registry() -> dict[str, Any]:
    actual = sha256_file(REGISTRY)
    if actual != REGISTRY_SHA256:
        raise RuntimeError(
            f"frozen registry hash mismatch: {actual} != {REGISTRY_SHA256}")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry.get("registry") != PROTOCOL_ID:
        raise RuntimeError("unexpected confirmatory registry ID")
    targets = registry.get("targets", [])
    if len(targets) != 16:
        raise RuntimeError("confirmatory registry must contain 16 targets")
    for spec in targets:
        build_target(spec)
    return registry


def frozen_arms() -> list[Arm]:
    arms = [Arm(f"paper-{tau:g}", "paper", tau, 0, 0.0, "update")
            for tau in PAPER_TAUS]
    arms.extend([
        Arm("qld-v1", "qld", BASELINE_TAU, 0, 0.70, "update"),
        Arm(CANDIDATE, "gated", BASELINE_TAU, 1024, 0.70, "update"),
    ])
    return arms


def make_tasks(registry: dict[str, Any], profile: Profile) \
        -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for ti, target_spec in enumerate(registry["targets"]):
        for ii, init in enumerate(INITS):
            for arm in frozen_arms():
                for seed in range(profile.seeds):
                    tasks.append({
                        "target_spec": target_spec,
                        "profile": asdict(profile),
                        "arm": asdict(arm),
                        "init": init,
                        "seed": seed,
                        "candidate_taus": PAPER_TAUS,
                        "diagnostic_samples": 4096,
                        "base_seed": seed_base(
                            int(registry["master_seed"]), ti, ii, seed),
                    })
    return tasks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(
            rows, key=lambda row: (str(row["target"]), str(row["init"]),
                                   str(row["arm"]), int(row["seed"]))))


def confirmatory_result(summary: dict[str, Any]) -> dict[str, Any]:
    baseline = "paper-0.5"
    ratio = float(summary["ratios_vs_selected_paper"][CANDIDATE])
    oracle_ratio = float(summary["ratios_vs_paper_oracle"][CANDIDATE])
    families = summary["family_ratios_vs_selected_paper"][CANDIDATE]
    details = summary["best_candidate_details"]
    candidate_divergence = int(summary["work_medians"][CANDIDATE][
        "divergences"])
    baseline_divergence = int(summary["work_medians"][baseline][
        "divergences"])
    checks = {
        "ed2_ratio_at_most_0.80": ratio <= 0.80,
        "bootstrap_upper_below_1":
            details["ed2_bootstrap_ci_vs_selected_paper"][1] < 1.0,
        "oracle_ed2_ratio_at_most_0.95": oracle_ratio <= 0.95,
        "cell_win_fraction_at_least_0.60":
            details["cell_win_fraction_vs_selected_paper"] >= 0.60,
        "all_family_ratios_at_most_1.10":
            all(float(value) <= 1.10 for value in families.values()),
        "divergence_no_worse": candidate_divergence <= baseline_divergence,
    }
    return {
        "protocol": PROTOCOL_ID,
        "registry_sha256": REGISTRY_SHA256,
        "candidate": CANDIDATE,
        "selected_baseline": baseline,
        "ed2_ratio_vs_selected_paper": ratio,
        "ed2_bootstrap_ci_vs_selected_paper":
            details["ed2_bootstrap_ci_vs_selected_paper"],
        "ed2_ratio_vs_paper_oracle": oracle_ratio,
        "sw1_ratio_vs_selected_paper":
            summary["sw1_ratios_vs_selected_paper"][CANDIDATE],
        "sw1_bootstrap_ci_vs_selected_paper":
            details["sw1_bootstrap_ci_vs_selected_paper"],
        "cell_win_fraction":
            details["cell_win_fraction_vs_selected_paper"],
        "family_ed2_ratios": families,
        "init_ed2_ratios": details["init_ed2_ratios_vs_selected_paper"],
        "routing_rate_by_target": details["routing_rate_by_target"],
        "summed_wall_ratio":
            details["summed_wall_ratio_vs_selected_paper"],
        "generator_eval_ratio":
            details["generator_eval_ratio_vs_selected_paper"],
        "kernel_pair_ratio": details["kernel_pair_ratio_vs_selected_paper"],
        "candidate_divergences": candidate_divergence,
        "baseline_divergences": baseline_divergence,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
        "development_summary": summary,
    }


def write_results_md(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# Resolution-gated LB-QCD: confirmatory results",
        "",
        f"**Protocol:** `{PROTOCOL_ID}`  ",
        "**Verdict:** " + ("**PASS**" if result["gate_passed"] else
                           "**FAIL**"),
        "",
        "## Primary results",
        "",
        "- ED2 ratio versus selected paper: "
        f"`{result['ed2_ratio_vs_selected_paper']:.4f}`",
        "- target-bootstrap ED2 95% CI: "
        f"`[{result['ed2_bootstrap_ci_vs_selected_paper'][0]:.4f}, "
        f"{result['ed2_bootstrap_ci_vs_selected_paper'][1]:.4f}]`",
        "- ED2 ratio versus per-cell paper oracle: "
        f"`{result['ed2_ratio_vs_paper_oracle']:.4f}`",
        f"- cell win fraction: `{result['cell_win_fraction']:.4f}`",
        "- SW1 ratio versus selected paper: "
        f"`{result['sw1_ratio_vs_selected_paper']:.4f}`",
        "",
        "## Gate",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in result["gate_checks"].items())
    lines.extend(["", "## Family ED2 ratios", ""])
    lines.extend(
        f"- {family}: `{float(value):.4f}`"
        for family, value in sorted(result["family_ed2_ratios"].items()))
    lines.extend([
        "", "## Cost", "",
        f"- summed worker wall ratio: `{result['summed_wall_ratio']:.4f}`",
        "- generator-example-evaluation ratio: "
        f"`{result['generator_eval_ratio']:.4f}`",
        f"- kernel-pair ratio: `{result['kernel_pair_ratio']:.4f}`",
        "", "## Scope", "",
        "This is a frozen one-dimensional missing/concentrated-initialization "
        "test. It does not cover far starts, dimensions above one, or "
        "ImageNet-scale generation.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_run_dir(profile: Profile) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNROOT / f"{stamp}-test-{profile.name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_manifest(path: Path, profile: Profile, workers: int,
                   task_count: int, wall_seconds: float,
                   pre_run_status: str) -> None:
    sources = [Path(__file__), REGISTRY, PROTOCOL, RESEARCH,
               HERE / "lbqcd.py", HERE / "run_lbqcd_development.py",
               HERE / "identifiability_drift.py",
               HERE / "run_identifiability_generator.py",
               HERE / "lowdim_drift.py"]
    manifest = {
        "protocol": PROTOCOL_ID,
        "profile": asdict(profile),
        "workers": workers,
        "task_count": task_count,
        "registry_sha256": REGISTRY_SHA256,
        "pre_run_git_status": pre_run_status.splitlines(),
        "commit": _git("rev-parse", "HEAD"),
        "python": sys.version,
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
    parser.add_argument("--profile", choices=tuple(PROFILES),
                        default="standard")
    parser.add_argument("--workers", type=int,
                        default=min(10, os.cpu_count() or 1))
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    invariant_tests()
    registry = load_registry()
    if args.self_check_only:
        print("LBQCD confirmatory invariants and registry: PASS")
        return
    pre_run_status = _git("status", "--porcelain")
    if args.profile == "standard" and pre_run_status:
        raise RuntimeError(
            "standard confirmatory run requires a clean committed tree")
    profile = PROFILES[args.profile]
    tasks = make_tasks(registry, profile)
    run_dir = make_run_dir(profile)
    print(f"run directory: {run_dir}", flush=True)
    print(f"registry sha256: {REGISTRY_SHA256}", flush=True)
    print(f"tasks: {len(tasks)}", flush=True)
    wall0 = time.perf_counter()
    rows = execute(tasks, max(1, args.workers))
    wall = time.perf_counter() - wall0
    write_csv(run_dir / "rows.csv", rows)
    summary = summarize(
        rows, "n1", BASELINE_TAU,
        bootstrap_reps=10_000,
        bootstrap_seed=int(registry["master_seed"]) + 9001)
    result = confirmatory_result(summary)
    (run_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    write_results_md(run_dir / "RESULTS.md", result)
    write_manifest(run_dir, profile, args.workers, len(tasks), wall,
                   pre_run_status)
    print(json.dumps({
        key: result[key] for key in (
            "ed2_ratio_vs_selected_paper",
            "ed2_bootstrap_ci_vs_selected_paper",
            "ed2_ratio_vs_paper_oracle", "sw1_ratio_vs_selected_paper",
            "cell_win_fraction", "gate_checks", "gate_passed")
    }, indent=2), flush=True)
    print(f"completed in {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
