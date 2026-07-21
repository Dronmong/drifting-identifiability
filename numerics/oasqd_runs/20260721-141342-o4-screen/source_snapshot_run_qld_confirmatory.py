"""Frozen validation and confirmatory test for Quantile-to-Laplace Drifting.

This runner is intentionally standalone at the protocol layer.  It reuses the
repository's exact Algorithm-2 field and TanhMLP/Adam implementation, verifies
the sealed registry hash, and writes immutable run artifacts.
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
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from identifiability_drift import compute_field  # noqa: E402
from lowdim_drift import energy_distance2, sliced_w1  # noqa: E402
from run_identifiability_generator import TanhMLP  # noqa: E402


REGISTRY = HERE / "qld_confirmatory_registry.json"
PROTOCOL = HERE / "QuantileFissionConfirmatoryProtocol.md"
PLAN = HERE / "QuantileFissionDriftingPlan.md"
RUNROOT = HERE / "qld_runs"
REGISTRY_SHA256 = \
    "7FEF49789904464A3103E09A166AD1C49A28C72DCAE00C61B87D72AA0CB1B8F8"
PAPER_TAUS = (0.2, 0.5, 1.0, 2.0, 4.0)
QLD_TAU = 0.5
PRIMARY_INITS = ("missing", "concentrated")


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    batch: int
    validation_seeds: int
    test_seeds: int
    eval_size: int
    ed_size: int
    bootstrap_reps: int


PROFILES = {
    "smoke": Profile("smoke", 60, 32, 2, 2, 512, 256, 200),
    "standard": Profile("standard", 1200, 128, 8, 20, 4096, 1024, 10_000),
}


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

    d: int = 1

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.kind == "student":
            return (rng.standard_t(float(self.df), size=(n, 1)) *
                    float(self.sigmas[0]))
        comp = rng.choice(len(self.means), size=n, p=self.weights)
        return (self.means[comp] +
                rng.normal(size=(n, 1)) * self.sigmas[comp, None])


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


def load_registry() -> dict[str, Any]:
    actual = sha256_file(REGISTRY)
    if actual != REGISTRY_SHA256:
        raise RuntimeError(
            f"frozen registry hash mismatch: {actual} != {REGISTRY_SHA256}")
    obj = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if obj.get("registry") != "QLD-confirmatory-v1":
        raise RuntimeError("unexpected registry ID")
    if len(obj.get("test_targets", [])) != 16:
        raise RuntimeError("confirmatory registry must contain 16 test targets")
    return obj


def build_target(spec: dict[str, Any]) -> Target1D:
    kind = str(spec["kind"])
    if kind == "student":
        scale = float(spec["scale"])
        return Target1D(
            str(spec["name"]), str(spec["family"]), kind,
            np.asarray([[0.0]]), np.asarray([1.0]), np.asarray([scale]),
            3.5 * scale, float(spec["df"]))
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
        raise ValueError(f"malformed target {spec['name']}")
    if np.any(sigmas <= 0) or np.any(weights <= 0):
        raise ValueError(f"nonpositive target parameter in {spec['name']}")
    weights = weights / weights.sum()
    scale = float(np.max(np.abs(means[:, 0])) + 4 * np.max(sigmas))
    return Target1D(
        str(spec["name"]), str(spec["family"]), kind,
        means, weights, sigmas, max(scale, 0.1))


def seed_base(master: int, split_offset: int, target_index: int,
              init_index: int, seed: int) -> int:
    return int(master * 1_000_003 + split_offset * 1_000_000_007 +
               target_index * 100_003 + init_index * 10_007 + seed * 101)


def exact_rank_field(x: np.ndarray, target: np.ndarray,
                     latent: np.ndarray) -> np.ndarray:
    if x.shape != target.shape or x.shape[1] != 1:
        raise ValueError("QLD rank field requires equal 1-D batches")
    tie = latent[:, 0]
    tie = (tie - tie.mean()) / max(float(tie.std()), 1e-12)
    ox = np.argsort(x[:, 0] + 1e-10 * tie, kind="stable")
    oy = np.argsort(target[:, 0], kind="stable")
    out = np.zeros_like(x)
    out[ox, 0] = target[oy, 0] - x[ox, 0]
    return out


def target_diagnostics(q: np.ndarray, target: Target1D) \
        -> tuple[float, float]:
    distance = np.abs(q[:, 0, None] - target.means[:, 0][None, :])
    nearest = np.argmin(distance, axis=1)
    reached = np.asarray([
        np.any(distance[:, j] <= 3.0 * target.sigmas[j])
        for j in range(len(target.means))
    ])
    weighted_reach = float(np.sum(target.weights[reached]))
    observed = np.bincount(nearest, minlength=len(target.means)) / len(q)
    mass_l1 = float(np.sum(np.abs(observed - target.weights)))
    return weighted_reach, mass_l1


def run_trial(task: dict[str, Any]) -> dict[str, Any]:
    target = build_target(task["target_spec"])
    profile = Profile(**task["profile"])
    arm = str(task["arm"])
    tau = float(task["tau"])
    base = int(task["base_seed"])
    seed = int(task["seed"])
    init_kind = str(task["init"])
    wall0 = time.perf_counter()

    model = TanhMLP(target, init_kind, base + 1)
    latent_rng = np.random.default_rng(base + 2)
    data_rng = np.random.default_rng(base + 3)
    warm_steps = int(round(0.70 * profile.steps))
    diverged = False
    event_time: int | None = None
    kernel_pairs = 0
    sort_work = 0.0

    for step in range(1, profile.steps + 1):
        z = latent_rng.normal(size=(profile.batch, model.latent_dim))
        x, cache = model.forward(z, want_cache=True)
        positive = target.sample(profile.batch, data_rng)
        if arm == "qld" and step <= warm_steps:
            field = exact_rank_field(x, positive, z)
            sort_work += 2 * profile.batch * math.log2(profile.batch)
        else:
            field = compute_field(
                x, positive, tau=tau, gain="paper", mask=True,
                on_degenerate="zero").V
            kernel_pairs += 2 * profile.batch * profile.batch
        try:
            model.stopgrad_step(cache, field)
        except FloatingPointError:
            diverged = True
            break
        if (not model.finite() or
                np.linalg.norm(x) > 1e6 * max(target.scale, 1.0)):
            diverged = True
            break
        if event_time is None and (step == 1 or step % 50 == 0):
            reach, _ = target_diagnostics(x, target)
            if reach >= 0.90:
                event_time = step

    if diverged:
        return {
            "arm": arm, "tau": tau, "target": target.name,
            "family": target.family, "init": init_kind,
            "cell": f"{target.name}/{init_kind}", "seed": seed,
            "ed2": float("inf"), "sw1": float("inf"),
            "weighted_reach": float("nan"), "mass_l1": float("nan"),
            "event_time": profile.steps, "censored": 1, "diverged": 1,
            "wall_seconds": time.perf_counter() - wall0,
            "kernel_pairs": kernel_pairs, "sort_work": sort_work,
        }

    eval_latent = np.random.default_rng(base + 4).normal(
        size=(profile.eval_size, model.latent_dim))
    q = model.forward(eval_latent)
    p = target.sample(profile.eval_size, np.random.default_rng(base + 5))
    metric_rng = np.random.default_rng(base + 6)
    iq = metric_rng.choice(len(q), profile.ed_size, replace=False)
    ip = metric_rng.choice(len(p), profile.ed_size, replace=False)
    ed2 = max(float(energy_distance2(q[iq], p[ip])), 0.0)
    sw1 = sliced_w1(q, p, 1, metric_rng)
    reach, mass_l1 = target_diagnostics(q, target)
    return {
        "arm": arm, "tau": tau, "target": target.name,
        "family": target.family, "init": init_kind,
        "cell": f"{target.name}/{init_kind}", "seed": seed,
        "ed2": ed2, "sw1": sw1, "weighted_reach": reach,
        "mass_l1": mass_l1,
        "event_time": event_time if event_time is not None else profile.steps,
        "censored": int(event_time is None), "diverged": 0,
        "wall_seconds": time.perf_counter() - wall0,
        "kernel_pairs": kernel_pairs, "sort_work": sort_work,
    }


def rank_field_self_check() -> None:
    x = np.asarray([[2.0], [-1.0], [0.5], [4.0]])
    y = np.asarray([[3.0], [-2.0], [1.0], [0.0]])
    z = np.asarray([[0.2, -0.1], [0.7, 0.4], [-0.3, 0.8], [1.1, -0.5]])
    got = exact_rank_field(x, y, z)
    ox = np.argsort(x[:, 0], kind="stable")
    oy = np.argsort(y[:, 0], kind="stable")
    want = np.zeros_like(x)
    want[ox, 0] = y[oy, 0] - x[ox, 0]
    if not np.allclose(got, want, rtol=1e-12, atol=1e-12):
        raise AssertionError("rank-field self-check failed")


def make_tasks(stage: str, registry: dict[str, Any], profile: Profile) \
        -> list[dict[str, Any]]:
    validation = stage == "validation"
    targets = registry["validation_targets" if validation else "test_targets"]
    seeds = profile.validation_seeds if validation else profile.test_seeds
    tasks: list[dict[str, Any]] = []
    for ti, target_spec in enumerate(targets):
        for ii, init_kind in enumerate(PRIMARY_INITS):
            arms = [(f"paper-{tau:g}", tau) for tau in PAPER_TAUS]
            if not validation:
                arms = [("qld", QLD_TAU), *arms]
            for arm, tau in arms:
                for seed in range(seeds):
                    tasks.append({
                        "target_spec": target_spec,
                        "profile": asdict(profile),
                        "arm": arm,
                        "tau": tau,
                        "seed": seed,
                        "init": init_kind,
                        "base_seed": seed_base(
                            int(registry["master_seed"]),
                            0 if validation else 1, ti, ii, seed),
                    })
    return tasks


def execute(tasks: list[dict[str, Any]], workers: int) \
        -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    total = len(tasks)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_trial, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if done % 100 == 0 or done == total:
                elapsed = time.perf_counter() - started
                print(f"  {done}/{total} trials ({elapsed:.1f}s)", flush=True)
    return rows


def positive_median(values: list[float]) -> float:
    return max(float(np.median(values)), 1e-12)


def cell_medians(rows: list[dict[str, Any]], metric: str) \
        -> dict[tuple[str, str, str], float]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row["target"]), str(row["init"]), str(row["arm"]))
        grouped.setdefault(key, []).append(float(row[metric]))
    return {key: positive_median(values) for key, values in grouped.items()}


def select_baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    medians = cell_medians(rows, "ed2")
    targets = sorted({str(row["target"]) for row in rows})
    scores: dict[str, float] = {}
    for tau in PAPER_TAUS:
        arm = f"paper-{tau:g}"
        values = [medians[(target, init, arm)]
                  for target in targets for init in PRIMARY_INITS]
        scores[str(tau)] = float(np.exp(np.mean(np.log(values))))
    chosen = min(PAPER_TAUS, key=lambda tau: scores[str(tau)])
    return {
        "protocol": "QLD-confirmatory-v1",
        "profile": "standard",
        "registry_sha256": REGISTRY_SHA256,
        "selected_tau": chosen,
        "selection_rule": "minimum geometric mean of cell-median ED2",
        "scores": scores,
    }


def hierarchical_target_bootstrap(
        rows: list[dict[str, Any]], baseline_arm: str, metric: str,
        reps: int, seed: int) -> tuple[float, list[float]]:
    lookup = {
        (str(row["target"]), str(row["init"]), str(row["arm"]),
         int(row["seed"])): float(row[metric])
        for row in rows
    }
    targets = sorted({str(row["target"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})

    def cell_log_ratio(target: str, init: str, indices: np.ndarray) -> float:
        qld = [lookup[(target, init, "qld", seeds[i])] for i in indices]
        base = [lookup[(target, init, baseline_arm, seeds[i])]
                for i in indices]
        return math.log(positive_median(qld) / positive_median(base))

    def point_aggregate() -> float:
        log_ratios = []
        indices = np.arange(len(seeds))
        for target in targets:
            for init in PRIMARY_INITS:
                log_ratios.append(cell_log_ratio(target, init, indices))
        return float(math.exp(np.mean(log_ratios)))

    point = point_aggregate()
    rng = np.random.default_rng(seed)
    stats = np.empty(reps)
    for b in range(reps):
        chosen = [targets[i] for i in rng.integers(0, len(targets),
                                                   size=len(targets))]
        log_ratios = []
        # Each occurrence of a resampled target receives an independent paired
        # seed resample.  Reusing one seed draw for duplicate target occurrences
        # would understate the hierarchical bootstrap variance.
        for target in chosen:
            for init in PRIMARY_INITS:
                indices = rng.integers(0, len(seeds), size=len(seeds))
                log_ratios.append(cell_log_ratio(target, init, indices))
        stats[b] = math.exp(float(np.mean(log_ratios)))
    ci = [float(x) for x in np.quantile(stats, [0.025, 0.975])]
    return point, ci


def summarize_test(rows: list[dict[str, Any]], selection: dict[str, Any],
                   profile: Profile, registry: dict[str, Any]) -> dict[str, Any]:
    selected_tau = float(selection["selected_tau"])
    baseline_arm = f"paper-{selected_tau:g}"
    med_ed = cell_medians(rows, "ed2")
    med_sw = cell_medians(rows, "sw1")
    target_specs = {str(spec["name"]): spec
                    for spec in registry["test_targets"]}
    targets = sorted(target_specs)
    cell_ratios: dict[str, float] = {}
    cell_sw_ratios: dict[str, float] = {}
    family_values: dict[str, list[float]] = {}
    for target in targets:
        family = str(target_specs[target]["family"])
        for init in PRIMARY_INITS:
            cell = f"{target}/{init}"
            ratio = (med_ed[(target, init, "qld")] /
                     med_ed[(target, init, baseline_arm)])
            sw_ratio = (med_sw[(target, init, "qld")] /
                        med_sw[(target, init, baseline_arm)])
            cell_ratios[cell] = ratio
            cell_sw_ratios[cell] = sw_ratio
            family_values.setdefault(family, []).append(ratio)
    primary_ratio, primary_ci = hierarchical_target_bootstrap(
        rows, baseline_arm, "ed2", profile.bootstrap_reps,
        int(registry["master_seed"]) + 8001)
    sw_ratio, sw_ci = hierarchical_target_bootstrap(
        rows, baseline_arm, "sw1", profile.bootstrap_reps,
        int(registry["master_seed"]) + 8002)
    family_ratios = {
        family: float(np.exp(np.mean(np.log(values))))
        for family, values in family_values.items()
    }
    win_fraction = float(np.mean([value < 1.0
                                  for value in cell_ratios.values()]))
    qld_diverged = sum(int(row["diverged"]) for row in rows
                       if row["arm"] == "qld")
    base_diverged = sum(int(row["diverged"]) for row in rows
                        if row["arm"] == baseline_arm)

    oracle_tau: dict[str, float] = {}
    oracle_cell_ratios: dict[str, float] = {}
    for target in targets:
        best = min(PAPER_TAUS, key=lambda tau: np.exp(np.mean(np.log([
            med_ed[(target, init, f"paper-{tau:g}")]
            for init in PRIMARY_INITS
        ]))))
        oracle_tau[target] = best
        for init in PRIMARY_INITS:
            oracle_cell_ratios[f"{target}/{init}"] = (
                med_ed[(target, init, "qld")] /
                med_ed[(target, init, f"paper-{best:g}")])
    oracle_ratio = float(np.exp(np.mean(np.log(
        list(oracle_cell_ratios.values())))))

    gate_checks = {
        "ratio_at_most_0.80": primary_ratio <= 0.80,
        "bootstrap_upper_below_1": primary_ci[1] < 1.0,
        "win_fraction_at_least_0.60": win_fraction >= 0.60,
        "all_family_ratios_at_most_1.10":
            all(value <= 1.10 for value in family_ratios.values()),
        "divergence_no_worse": qld_diverged <= base_diverged,
    }
    total_wall = {
        arm: float(sum(float(row["wall_seconds"]) for row in rows
                       if row["arm"] == arm))
        for arm in ("qld", baseline_arm)
    }
    return {
        "protocol": "QLD-confirmatory-v1",
        "profile": profile.name,
        "selected_paper_tau": selected_tau,
        "selected_baseline_arm": baseline_arm,
        "primary_ed2_ratio": primary_ratio,
        "primary_ed2_bootstrap_ci": primary_ci,
        "primary_win_fraction": win_fraction,
        "family_ed2_ratios": family_ratios,
        "cell_ed2_ratios": cell_ratios,
        "secondary_sw1_ratio": sw_ratio,
        "secondary_sw1_bootstrap_ci": sw_ci,
        "cell_sw1_ratios": cell_sw_ratios,
        "oracle_paper_tau_by_target": oracle_tau,
        "oracle_target_ed2_ratio": oracle_ratio,
        "oracle_cell_ed2_ratios": oracle_cell_ratios,
        "qld_diverged": qld_diverged,
        "selected_paper_diverged": base_diverged,
        "summed_worker_wall_seconds": total_wall,
        "gate_checks": gate_checks,
        "gate_passed": all(gate_checks.values()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(sorted(
            rows, key=lambda row: (str(row["target"]), str(row["init"]),
                                   str(row["arm"]), int(row["seed"]))))


def write_results_md(path: Path, results: dict[str, Any]) -> None:
    checks = results["gate_checks"]
    families = results["family_ed2_ratios"]
    lines = [
        "# QLD confirmatory results",
        "",
        f"**Gate: {'PASS' if results['gate_passed'] else 'FAIL'}**",
        "",
        f"- selected paper tau: `{results['selected_paper_tau']}`",
        f"- primary ED2 ratio: `{results['primary_ed2_ratio']:.4f}`",
        "- paired target-bootstrap 95% CI: "
        f"`[{results['primary_ed2_bootstrap_ci'][0]:.4f}, "
        f"{results['primary_ed2_bootstrap_ci'][1]:.4f}]`",
        f"- cell win fraction: `{results['primary_win_fraction']:.3f}`",
        f"- secondary SW1 ratio: `{results['secondary_sw1_ratio']:.4f}`",
        "- oracle-per-target paper ED2 ratio: "
        f"`{results['oracle_target_ed2_ratio']:.4f}`",
        "",
        "## Gate checks",
        "",
    ]
    lines.extend(f"- {'PASS' if value else 'FAIL'}: `{name}`"
                 for name, value in checks.items())
    lines.extend(["", "## Family ED2 ratios", ""])
    lines.extend(f"- {family}: `{value:.4f}`"
                 for family, value in sorted(families.items()))
    lines.extend(["", "## Scope", "",
                  "This gate covers only the sealed one-dimensional fission "
                  "suite and its missing/concentrated initializations. It is "
                  "not an ImageNet, high-dimensional, or arbitrary-start "
                  "claim.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_run_dir(stage: str, profile: Profile) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = RUNROOT / f"{stamp}-{stage}-{profile.name}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_manifest(path: Path, stage: str, profile: Profile,
                   workers: int, selection_path: Path | None,
                   wall_seconds: float) -> None:
    sources = [Path(__file__), REGISTRY, PROTOCOL, PLAN,
               HERE / "identifiability_drift.py",
               HERE / "run_identifiability_generator.py",
               HERE / "lowdim_drift.py"]
    manifest = {
        "protocol": "QLD-confirmatory-v1",
        "stage": stage,
        "profile": asdict(profile),
        "workers": workers,
        "registry_sha256": REGISTRY_SHA256,
        "selection_path": str(selection_path) if selection_path else None,
        "selection_sha256": (sha256_file(selection_path)
                             if selection_path else None),
        "commit": git_text("rev-parse", "HEAD"),
        "git_status": git_text("status", "--porcelain").splitlines(),
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "wall_seconds": wall_seconds,
        "source_sha256": {str(source.relative_to(ROOT)): sha256_file(source)
                          for source in sources},
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    for source in sources:
        shutil.copy2(source, path / f"source_snapshot_{source.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validation", "test"), required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="standard")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    registry = load_registry()
    rank_field_self_check()
    if args.stage == "test" and args.selection is None:
        parser.error("--selection is required for the test stage")
    if args.stage == "validation" and args.selection is not None:
        parser.error("validation does not accept --selection")

    selection: dict[str, Any] | None = None
    if args.selection is not None:
        selection = json.loads(args.selection.read_text(encoding="utf-8"))
        if selection.get("registry_sha256") != REGISTRY_SHA256:
            raise RuntimeError("selection was produced for another registry")
        if selection.get("profile") != args.profile:
            raise RuntimeError("selection profile does not match test profile")

    run_dir = make_run_dir(args.stage, profile)
    print(f"run directory: {run_dir}", flush=True)
    print(f"registry hash: {REGISTRY_SHA256}", flush=True)
    tasks = make_tasks(args.stage, registry, profile)
    wall0 = time.perf_counter()
    rows = execute(tasks, max(1, args.workers))
    wall = time.perf_counter() - wall0
    write_csv(run_dir / "rows.csv", rows)

    if args.stage == "validation":
        chosen = select_baseline(rows)
        chosen["profile"] = profile.name
        (run_dir / "selection.json").write_text(
            json.dumps(chosen, indent=2), encoding="utf-8")
        print(json.dumps(chosen, indent=2), flush=True)
    else:
        assert selection is not None
        results = summarize_test(rows, selection, profile, registry)
        (run_dir / "results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
        write_results_md(run_dir / "RESULTS.md", results)
        print(json.dumps({
            key: results[key] for key in (
                "selected_paper_tau", "primary_ed2_ratio",
                "primary_ed2_bootstrap_ci", "primary_win_fraction",
                "secondary_sw1_ratio", "oracle_target_ed2_ratio",
                "gate_checks", "gate_passed")
        }, indent=2), flush=True)
    write_manifest(run_dir, args.stage, profile, args.workers,
                   args.selection, wall)
    print(f"completed in {wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
