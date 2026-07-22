"""Paired development experiment for PSQT target-quantile accumulators.

This runner factorizes target-statistic estimation from particle geometry and
compares bounded-memory streaming repairs with the historical online PSQT,
the selected paper estimator, and an unbounded exact-pooled ceiling.

Usage:

    uv run --with numpy --with scipy --with matplotlib \
        python numerics/run_psqt_accumulator_repair.py --profile smoke
    uv run --with numpy --with scipy --with matplotlib \
        python numerics/run_psqt_accumulator_repair.py --profile screen
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from numpy.linalg import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNROOT = HERE / "psqt_accumulator_runs"
sys.path.insert(0, str(HERE))

from lowdim_drift import (  # noqa: E402
    TargetSpec,
    drift_paper,
    energy_distance2,
    mode_metrics,
)
from persistent_quantile_transport import midpoint_grid  # noqa: E402
from persistent_sliced_quantile_transport import (  # noqa: E402
    PersistentSlicedQuantileTransport,
    invariant_tests as psqt_invariant_tests,
    projected_quantile_rmse,
    uniform_directions_2d,
)
from projected_quantile_accumulators import (  # noqa: E402
    BatchMeanProjectedAccumulator,
    BatchMedianProjectedAccumulator,
    ExactPooledProjectedAccumulator,
    KLLStyleProjectedAccumulator,
    RawReservoirProjectedAccumulator,
    invariant_tests as accumulator_invariant_tests,
    projected_quantile_table,
    reconstruct_from_quantile_table,
)
from run_psqt_2d_development import (  # noqa: E402
    development_targets,
    fixed_sliced_w1,
    initial_particles,
)

MASTER = 20260723


@dataclass(frozen=True)
class Profile:
    name: str
    updates: int
    seeds: int
    particles: int
    batch: int
    reference: int
    directions: int
    knots: int
    reconstruction_steps: int
    reconstruction_step_size: float
    target_mode: str


PROFILES = {
    "smoke": Profile(
        "smoke", 40, 1, 32, 32, 512, 16, 32, 60, 0.5,
        "smoke"),
    "screen": Profile(
        "screen", 300, 5, 64, 64, 1024, 32, 64, 100, 0.5,
        "all"),
}


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str
    parameter: int | None = None


ARMS = (
    Arm("paper-tau1", "paper"),
    Arm("historical-online-psqt", "historical"),
    Arm("batch-mean-table", "batch-mean"),
    Arm("batch-median-table", "batch-median"),
    Arm("reservoir-R512", "reservoir", 512),
    Arm("reservoir-R1024", "reservoir", 1024),
    Arm("reservoir-R2048", "reservoir", 2048),
    Arm("kll-style-k64", "kll-style", 64),
    Arm("kll-style-k128", "kll-style", 128),
    Arm("exact-pooled-ceiling", "exact-pooled"),
)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Artifact:
    def __init__(self, profile: Profile) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.directory = RUNROOT / f"{stamp}-{profile.name}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.time()
        self.lines: list[str] = []
        sources = [
            HERE / "run_psqt_accumulator_repair.py",
            HERE / "projected_quantile_accumulators.py",
            HERE / "persistent_sliced_quantile_transport.py",
            HERE / "persistent_quantile_transport.py",
            HERE / "run_psqt_2d_development.py",
            HERE / "lowdim_drift.py",
            HERE / "PSQTQuantileAccumulatorDefectAnalysis.md",
            HERE / "PSQTQuantileAccumulatorRepairPlan.md",
        ]
        self.manifest = {
            "protocol_status": "development-not-frozen",
            "master_seed": MASTER,
            "profile": asdict(profile),
            "arms": [asdict(arm) for arm in ARMS],
            "quantile_convention": {
                "historical": "linear per-batch quantiles",
                "repairs": "inverted empirical CDF where applicable",
            },
            "git_commit": _git("rev-parse", "HEAD"),
            "git_status": _git("status", "--short").splitlines(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "command": sys.argv,
            "started": stamp,
            "source_sha256": {
                str(source.relative_to(ROOT)): _sha256(source)
                for source in sources
            },
        }
        for source in sources:
            shutil.copy2(source, self.directory / f"source_{source.name}")

    def log(self, line: str = "") -> None:
        print(line, flush=True)
        self.lines.append(line)

    def finish(self) -> None:
        self.manifest["wall_seconds"] = time.time() - self.started
        with (self.directory / "manifest.json").open(
                "w", encoding="utf-8") as stream:
            json.dump(self.manifest, stream, indent=2)
        with (self.directory / "RESULTS.md").open(
                "w", encoding="utf-8") as stream:
            stream.write("\n".join(self.lines) + "\n")


def weighted_diagonal_target(name: str, minority: float) -> TargetSpec:
    if not 0.0 < minority <= 0.5:
        raise ValueError("minority mass must lie in (0, 0.5]")
    means = np.asarray([[-1.0, -1.0], [1.0, 1.0]])
    weights = np.asarray([1.0 - minority, minority])

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        labels = rng.choice(2, size=n, p=weights)
        return means[labels] + rng.normal(size=(n, 2)) * 0.10

    return TargetSpec(
        name, 2, sampler, "rare", means, weights, scale=1.5)


def targets_for(profile: Profile) -> list[TargetSpec]:
    original = development_targets()
    rare = [
        weighted_diagonal_target("PS2-diagonal-minority-01", 0.01),
        weighted_diagonal_target("PS2-diagonal-minority-05", 0.05),
        weighted_diagonal_target("PS2-diagonal-minority-10", 0.10),
    ]
    if profile.target_mode == "smoke":
        # Include the original failure, a connected control, and two rare modes.
        return [original[0], original[3], original[-1], rare[1], rare[2]]
    return [*original, *rare]


def endpoint_metrics(q: np.ndarray, target: TargetSpec,
                     reference: np.ndarray,
                     heldout_directions: np.ndarray) -> dict[str, float]:
    coverage, mass_l1 = mode_metrics(q, target)
    return {
        "ed2": float(energy_distance2(q, reference)),
        "heldout_sw1": fixed_sliced_w1(
            q, reference, heldout_directions),
        "heldout_projection_rmse": projected_quantile_rmse(
            q, reference, heldout_directions, knot_count=128),
        "coverage": float(coverage),
        "mass_l1": float(mass_l1),
    }


def bridge_fraction(points: np.ndarray, target: TargetSpec) -> float:
    if (target.family not in {"dependence", "rare"} or
            target.means is None or len(target.means) != 2):
        return float("nan")
    delta = target.means[1] - target.means[0]
    half_distance = 0.5 * norm(delta)
    direction = delta / norm(delta)
    midpoint = 0.5 * (target.means[0] + target.means[1])
    coordinate = (points - midpoint) @ direction
    return float(np.mean(np.abs(coordinate) < 0.60 * half_distance))


def minority_diagnostics(q: np.ndarray,
                         target: TargetSpec) -> tuple[float, int]:
    if target.family != "rare" or target.means is None:
        return float("nan"), 0
    labels = norm(
        q[:, None, :] - target.means[None, :, :], axis=2).argmin(axis=1)
    masses = np.bincount(labels, minlength=2) / len(q)
    minority_index = int(np.argmin(target.weights))
    target_mass = float(target.weights[minority_index])
    recovered = int(masses[minority_index] >= 0.5 * target_mass)
    return float(masses[minority_index]), recovered


def training_table_rmse(q: np.ndarray, directions: np.ndarray,
                        table: np.ndarray) -> float:
    current = projected_quantile_table(
        q, directions, midpoint_grid(table.shape[1]), method="linear")
    return float(np.sqrt(np.mean((current - table) ** 2)))


def run_paper(q0: np.ndarray, batches: np.ndarray, target: TargetSpec,
              reference: np.ndarray, heldout: np.ndarray) \
        -> tuple[dict, np.ndarray]:
    q = q0.copy()
    started = time.perf_counter()
    eta = 0.15
    kernel_pairs = 0
    diverged = False
    for batch in batches:
        q += eta * drift_paper(q, batch, 1.0, True)
        kernel_pairs += len(q) * (len(batch) + len(q))
        if not np.all(np.isfinite(q)) or norm(q) > 1e6:
            diverged = True
            break
    elapsed = time.perf_counter() - started
    metrics = endpoint_metrics(q, target, reference, heldout)
    metrics.update({
        "diverged": int(diverged),
        "wall_seconds": elapsed,
        "target_samples": int(batches.shape[0] * batches.shape[1]),
        "kernel_pairs": int(kernel_pairs),
        "sort_work": 0.0,
        "projection_dot_products": 0,
        "persistent_scalars": int(q.size),
        "peak_working_scalars": int(q.size + len(batches[0]) * 2),
        "retained_items": 0,
        "retained_weight": 0,
        "reconstruction_sweeps": 0,
        "training_projection_rmse": float("nan"),
        "target_table_rmse": float("nan"),
    })
    return metrics, q


def run_historical(q0: np.ndarray, batches: np.ndarray,
                   target: TargetSpec, reference: np.ndarray,
                   heldout: np.ndarray, directions: np.ndarray,
                   profile: Profile, exact_table: np.ndarray) \
        -> tuple[dict, np.ndarray]:
    model = PersistentSlicedQuantileTransport(
        q0, directions, knot_count=profile.knots, prior_batches=1.0,
        reconstruction_steps=3, reconstruction_step_size=0.5)
    started = time.perf_counter()
    sort_work = 0.0
    dot_products = 0
    for batch in batches:
        work = model.update(batch)
        sort_work += work.sort_work
        dot_products += work.projection_dot_products
    elapsed = time.perf_counter() - started
    q = model.particles.copy()
    metrics = endpoint_metrics(q, target, reference, heldout)
    metrics.update({
        "diverged": int(not np.all(np.isfinite(q))),
        "wall_seconds": elapsed,
        "target_samples": int(batches.shape[0] * batches.shape[1]),
        "kernel_pairs": 0,
        "sort_work": float(sort_work),
        "projection_dot_products": int(dot_products),
        "persistent_scalars": model.stored_scalars,
        "peak_working_scalars": int(
            model.stored_scalars + len(batches[0]) * len(directions)),
        "retained_items": 0,
        "retained_weight": int(batches.shape[0] * batches.shape[1]),
        "reconstruction_sweeps": int(3 * len(batches)),
        "training_projection_rmse": model.training_projection_rmse(),
        "target_table_rmse": float(np.sqrt(np.mean(
            (model.target_quantiles - exact_table) ** 2))),
    })
    return metrics, q


def make_accumulator(arm: Arm, q0: np.ndarray, directions: np.ndarray,
                     profile: Profile, seed: int):
    if arm.kind == "batch-mean":
        return BatchMeanProjectedAccumulator(
            q0, directions, profile.knots, prior_batches=1.0)
    if arm.kind == "batch-median":
        return BatchMedianProjectedAccumulator(
            2, directions, profile.knots)
    if arm.kind == "reservoir":
        return RawReservoirProjectedAccumulator(
            2, directions, profile.knots, int(arm.parameter), seed=seed)
    if arm.kind == "kll-style":
        return KLLStyleProjectedAccumulator(
            2, directions, profile.knots, int(arm.parameter), seed=seed)
    if arm.kind == "exact-pooled":
        return ExactPooledProjectedAccumulator(2, directions, profile.knots)
    raise ValueError(f"arm {arm.name} has no accumulator")


def run_accumulator_arm(
        arm: Arm, q0: np.ndarray, batches: np.ndarray, target: TargetSpec,
        reference: np.ndarray, heldout: np.ndarray, directions: np.ndarray,
        profile: Profile, exact_table: np.ndarray, seed: int) \
        -> tuple[dict, np.ndarray, np.ndarray]:
    accumulator = make_accumulator(arm, q0, directions, profile, seed)
    started = time.perf_counter()
    for batch in batches:
        accumulator.update(batch)
    table = accumulator.table()
    q, reconstruction_sort, reconstruction_dots = (
        reconstruct_from_quantile_table(
            q0, directions, table, steps=profile.reconstruction_steps,
            step_size=profile.reconstruction_step_size))
    elapsed = time.perf_counter() - started
    ledger = accumulator.ledger()
    metrics = endpoint_metrics(q, target, reference, heldout)
    metrics.update({
        "diverged": int(not np.all(np.isfinite(q))),
        "wall_seconds": elapsed,
        "target_samples": ledger.target_samples,
        "kernel_pairs": 0,
        "sort_work": float(ledger.sort_work + reconstruction_sort),
        "projection_dot_products": int(
            ledger.projection_dot_products + reconstruction_dots),
        "persistent_scalars": int(
            ledger.persistent_scalars + q.size),
        "peak_working_scalars": int(
            ledger.peak_working_scalars + q.size + table.size),
        "retained_items": ledger.retained_items,
        "retained_weight": ledger.retained_weight,
        "reconstruction_sweeps": profile.reconstruction_steps,
        "training_projection_rmse": training_table_rmse(
            q, directions, table),
        "target_table_rmse": float(np.sqrt(np.mean(
            (table - exact_table) ** 2))),
    })
    return metrics, q, table


def split_table_disagreement(batches: np.ndarray, directions: np.ndarray,
                             knots: int) -> float:
    even = batches[::2].reshape(-1, batches.shape[-1])
    odd = batches[1::2].reshape(-1, batches.shape[-1])
    grid = midpoint_grid(knots)
    a = projected_quantile_table(even, directions, grid)
    b = projected_quantile_table(odd, directions, grid)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def complete_diagnostics(metrics: dict, q: np.ndarray, target: TargetSpec,
                         reference: np.ndarray, split_rmse: float) -> None:
    bridge = bridge_fraction(q, target)
    reference_bridge = bridge_fraction(reference, target)
    minority_mass, minority_recovered = minority_diagnostics(q, target)
    metrics.update({
        "bridge_fraction": bridge,
        "reference_bridge_fraction": reference_bridge,
        "excess_bridge_fraction": (
            bridge - reference_bridge
            if math.isfinite(bridge) and math.isfinite(reference_bridge)
            else float("nan")),
        "minority_particle_mass": minority_mass,
        "minority_recovered": minority_recovered,
        "split_target_table_rmse": split_rmse,
    })


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def summarize(rows: list[dict], particles: int) -> dict:
    metrics = (
        "ed2", "heldout_sw1", "heldout_projection_rmse", "mass_l1",
        "bridge_fraction", "excess_bridge_fraction", "minority_recovered",
        "target_table_rmse", "wall_seconds", "persistent_scalars",
        "peak_working_scalars", "sort_work", "projection_dot_products",
    )
    arms = sorted({row["arm"] for row in rows})
    targets = sorted({row["target"] for row in rows})
    inits = sorted({row["init"] for row in rows})
    cells: dict[tuple[str, str, str], dict[str, float]] = {}
    for arm in arms:
        for target in targets:
            for init in inits:
                group = [
                    row for row in rows
                    if row["arm"] == arm and row["target"] == target and
                    row["init"] == init
                ]
                if group:
                    cells[(arm, target, init)] = {
                        metric: _median([float(row[metric]) for row in group])
                        for metric in metrics
                    }

    def ratio(candidate: str, baseline: str, metric: str,
              family: str | None = None, *,
              exclude_family: str | None = None) -> float:
        logs = []
        for target in targets:
            target_family = next(
                row["family"] for row in rows if row["target"] == target)
            if family is not None and target_family != family:
                continue
            if exclude_family is not None and target_family == exclude_family:
                continue
            for init in inits:
                top = cells[(candidate, target, init)][metric]
                bottom = cells[(baseline, target, init)][metric]
                logs.append(math.log(max(top, 1e-12) / max(bottom, 1e-12)))
        return math.exp(float(np.mean(logs))) if logs else float("nan")

    sample_count = min(int(row["target_samples"]) for row in rows)
    deployable = []
    for arm in arms:
        if arm.startswith("kll-style-"):
            deployable.append(arm)
        elif arm.startswith("reservoir-"):
            capacity = int(arm.rsplit("R", maxsplit=1)[1])
            # A reservoir at least as large as the entire smoke stream is
            # exact pooling with metadata, not a bounded approximation.
            if capacity < sample_count:
                deployable.append(arm)
    deployable_scores = {
        # Select on the original nine-family screen. Rare modes are a stress
        # test and must not dominate selection through very large ratios.
        arm: ratio(
            arm, "historical-online-psqt", "ed2",
            exclude_family="rare")
        for arm in deployable
    }
    selected = min(deployable_scores, key=deployable_scores.get)
    families = sorted({row["family"] for row in rows})
    family_ratios = {
        family: ratio(
            selected, "historical-online-psqt", "ed2", family)
        for family in families
    }
    family_paper_ratios = {
        family: ratio(selected, "paper-tau1", "ed2", family)
        for family in families
    }
    rare_rows = [
        row for row in rows
        if row["arm"] == selected and row["family"] == "rare" and
        not row["target"].endswith("-01")
    ]
    bridge_values = [
        float(row["excess_bridge_fraction"])
        for row in rows if row["arm"] == selected and
        row["family"] in {"dependence", "rare"}
    ]
    exact_memory = _median([
        float(row["persistent_scalars"]) for row in rows
        if row["arm"] == "exact-pooled-ceiling"
    ])
    selected_memory = _median([
        float(row["persistent_scalars"]) for row in rows
        if row["arm"] == selected
    ])
    gates = {
        "finite_and_no_divergence": all(
            int(row["diverged"]) == 0 and math.isfinite(float(row["ed2"]))
            for row in rows if row["arm"] == selected),
        "diagonal_better_than_historical": (
            ratio(selected, "historical-online-psqt", "ed2", "dependence")
            < 1.0),
        "diagonal_better_than_paper": (
            ratio(selected, "paper-tau1", "ed2", "dependence") < 1.0),
        "aggregate_ed2_better_than_historical": (
            ratio(
                selected, "historical-online-psqt", "ed2",
                exclude_family="rare") < 1.0),
        "aggregate_sw1_better_than_historical": (
            ratio(
                selected, "historical-online-psqt", "heldout_sw1",
                exclude_family="rare") < 1.0),
        "no_family_regression_over_5pct": max(family_ratios.values()) <= 1.05,
        "median_excess_bridge_at_most_one_particle": (
            _median(bridge_values) <= 1.0 / particles),
        "rare_05_10_majority_recovered": bool(
            np.mean([int(row["minority_recovered"]) for row in rare_rows])
            >= 0.5) if rare_rows else False,
        "bounded_memory_below_exact_pool": selected_memory < exact_memory,
    }
    arm_summary = {}
    for arm in arms:
        arm_summary[arm] = {
            "original_ed2_vs_historical": ratio(
                arm, "historical-online-psqt", "ed2",
                exclude_family="rare"),
            "original_ed2_vs_paper": ratio(
                arm, "paper-tau1", "ed2", exclude_family="rare"),
            "original_sw1_vs_historical": ratio(
                arm, "historical-online-psqt", "heldout_sw1",
                exclude_family="rare"),
            "rare_ed2_vs_historical": ratio(
                arm, "historical-online-psqt", "ed2", "rare"),
            "median_table_rmse": _median([
                float(row["target_table_rmse"])
                for row in rows if row["arm"] == arm]),
            "median_wall_seconds": _median([
                float(row["wall_seconds"])
                for row in rows if row["arm"] == arm]),
            "median_persistent_scalars": _median([
                float(row["persistent_scalars"])
                for row in rows if row["arm"] == arm]),
        }
    return {
        "development_selected_bounded_arm": selected,
        "selected_original_ed2_ratio_vs_historical": ratio(
            selected, "historical-online-psqt", "ed2",
            exclude_family="rare"),
        "selected_original_ed2_ratio_vs_paper": ratio(
            selected, "paper-tau1", "ed2", exclude_family="rare"),
        "selected_original_sw1_ratio_vs_historical": ratio(
            selected, "historical-online-psqt", "heldout_sw1",
            exclude_family="rare"),
        "selected_rare_ed2_ratio_vs_historical": ratio(
            selected, "historical-online-psqt", "ed2", "rare"),
        "exact_original_ed2_ratio_vs_historical": ratio(
            "exact-pooled-ceiling", "historical-online-psqt", "ed2",
            exclude_family="rare"),
        "exact_original_ed2_ratio_vs_paper": ratio(
            "exact-pooled-ceiling", "paper-tau1", "ed2",
            exclude_family="rare"),
        "selected_family_ed2_ratios_vs_historical": family_ratios,
        "selected_family_ed2_ratios_vs_paper": family_paper_ratios,
        "decision_gates": gates,
        "all_decision_gates_pass": bool(all(gates.values())),
        "arm_summary": arm_summary,
        "cell_medians": {
            "|".join(key): value for key, value in cells.items()
        },
    }


def make_visuals(directory: Path, payload: dict, selected: str) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = [
        "target", "initial", "paper-tau1", "historical-online-psqt",
        selected, "exact-pooled-ceiling",
    ]
    for target, arrays in payload.items():
        if any(label not in arrays for label in labels):
            continue
        figure, axes = plt.subplots(1, len(labels), figsize=(18, 3.2),
                                    sharex=True, sharey=True)
        combined = np.vstack([arrays[label] for label in labels])
        low = np.quantile(combined, 0.005, axis=0)
        high = np.quantile(combined, 0.995, axis=0)
        center = 0.5 * (low + high)
        radius = 0.55 * max(*(high - low), 1e-6)
        for axis, label in zip(axes, labels):
            axis.scatter(
                arrays[label][:, 0], arrays[label][:, 1],
                s=7, alpha=0.65, linewidths=0)
            axis.set_title(label, fontsize=7)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_aspect("equal")
            axis.grid(alpha=0.15)
        figure.suptitle(target)
        figure.tight_layout()
        figure.savefig(
            directory / f"visual_{target.replace('/', '-')}.png", dpi=170)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    artifact = Artifact(profile)
    artifact.log("# PSQT target-quantile accumulator repair")
    artifact.log()
    artifact.log("Status: exploratory development; not frozen confirmation.")
    artifact.log()
    artifact.log("## Invariants")
    accumulator_invariant_tests()
    psqt_invariant_tests()
    artifact.log("- accumulator invariants: PASS")
    artifact.log("- historical PSQT invariants: PASS")

    targets = targets_for(profile)
    directions = uniform_directions_2d(profile.directions)
    heldout = uniform_directions_2d(64, phase=np.pi / 128)
    grid = midpoint_grid(profile.knots)
    rows: list[dict] = []
    visuals: dict[str, dict[str, np.ndarray]] = {}
    tables: dict[str, np.ndarray] = {}
    total = len(targets) * 2 * profile.seeds
    completed = 0
    for target_index, target in enumerate(targets):
        for init_index, init in enumerate(("concentrated", "far")):
            for seed in range(profile.seeds):
                base_seed = (
                    MASTER * 1_000_003 + target_index * 100_003 +
                    init_index * 10_007 + seed * 101)
                q0 = initial_particles(
                    init, target, profile.particles,
                    np.random.default_rng(base_seed + 1))
                data_rng = np.random.default_rng(base_seed + 2)
                batches = np.stack([
                    target.sample(profile.batch, data_rng)
                    for _ in range(profile.updates)
                ])
                pooled_points = batches.reshape(-1, 2)
                exact_table = projected_quantile_table(
                    pooled_points, directions, grid)
                split_rmse = split_table_disagreement(
                    batches, directions, profile.knots)
                reference = target.sample(
                    profile.reference, np.random.default_rng(base_seed + 3))
                outputs: dict[str, np.ndarray] = {}
                metrics, output = run_paper(
                    q0, batches, target, reference, heldout)
                complete_diagnostics(
                    metrics, output, target, reference, split_rmse)
                outputs["paper-tau1"] = output
                rows.append({
                    "arm": "paper-tau1", "kind": "paper",
                    "target": target.name, "family": target.family,
                    "init": init, "seed": seed, **metrics,
                })

                metrics, output = run_historical(
                    q0, batches, target, reference, heldout, directions,
                    profile, exact_table)
                complete_diagnostics(
                    metrics, output, target, reference, split_rmse)
                outputs["historical-online-psqt"] = output
                rows.append({
                    "arm": "historical-online-psqt", "kind": "historical",
                    "target": target.name, "family": target.family,
                    "init": init, "seed": seed, **metrics,
                })

                for arm_index, arm in enumerate(ARMS[2:], start=2):
                    metrics, output, table = run_accumulator_arm(
                        arm, q0, batches, target, reference, heldout,
                        directions, profile, exact_table,
                        base_seed + 1000 + arm_index)
                    complete_diagnostics(
                        metrics, output, target, reference, split_rmse)
                    outputs[arm.name] = output
                    rows.append({
                        "arm": arm.name, "kind": arm.kind,
                        "target": target.name, "family": target.family,
                        "init": init, "seed": seed, **metrics,
                    })
                    if (seed == 0 and init == "concentrated" and
                            target.family in {"dependence", "rare"}):
                        tables[f"{target.name}|{arm.name}"] = table

                if seed == 0 and init == "concentrated":
                    visuals[target.name] = {
                        "target": reference, "initial": q0, **outputs}
                completed += 1
                artifact.log(
                    f"- completed paired cell {completed}/{total}: "
                    f"{target.name}/{init}/seed{seed}")

    expected = total * len(ARMS)
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} rows, found {len(rows)}")
    write_rows(artifact.directory / "rows.csv", rows)
    summary = summarize(rows, profile.particles)
    with (artifact.directory / "summary.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    if tables:
        np.savez_compressed(
            artifact.directory / "representative_tables.npz", **tables)
    make_visuals(
        artifact.directory, visuals,
        summary["development_selected_bounded_arm"])

    artifact.log()
    artifact.log("## Development outcome")
    artifact.log(
        "- selected bounded accumulator: "
        f"`{summary['development_selected_bounded_arm']}`")
    artifact.log(
        "- ED2 ratio / historical online PSQT: "
        f"`{summary['selected_original_ed2_ratio_vs_historical']:.4f}`")
    artifact.log(
        "- ED2 ratio / selected paper: "
        f"`{summary['selected_original_ed2_ratio_vs_paper']:.4f}`")
    artifact.log(
        "- held-out SW1 ratio / historical online PSQT: "
        f"`{summary['selected_original_sw1_ratio_vs_historical']:.4f}`")
    artifact.log(
        "- rare-mode ED2 ratio / historical online PSQT: "
        f"`{summary['selected_rare_ed2_ratio_vs_historical']:.4f}`")
    artifact.log(
        "- exact-pooled ED2 ratio / historical online PSQT: "
        f"`{summary['exact_original_ed2_ratio_vs_historical']:.4f}`")
    artifact.log(
        "- exact-pooled ED2 ratio / selected paper: "
        f"`{summary['exact_original_ed2_ratio_vs_paper']:.4f}`")
    artifact.log("- family ED2 ratios, selected / historical:")
    for family, value in summary[
            "selected_family_ed2_ratios_vs_historical"].items():
        artifact.log(f"  - `{family}`: `{value:.4f}`")
    artifact.log("- decision gates:")
    for gate, passed in summary["decision_gates"].items():
        artifact.log(f"  - `{gate}`: **{'PASS' if passed else 'FAIL'}**")
    artifact.log(
        "- all development gates: "
        f"**{'PASS' if summary['all_decision_gates_pass'] else 'FAIL'}**")
    artifact.log()
    artifact.log("## Interpretation boundary")
    artifact.log()
    artifact.log(
        "The selected arm and every threshold were evaluated on reused "
        "development families. Exact pooling is an unbounded finite-stream "
        "ceiling. The KLL-style arm uses fixed-capacity random compactors and "
        "does not claim the optimal KLL space theorem. A fresh registry is "
        "required before a confirmatory superiority claim.")
    artifact.finish()
    print(f"artifact: {artifact.directory}", flush=True)


if __name__ == "__main__":
    main()
