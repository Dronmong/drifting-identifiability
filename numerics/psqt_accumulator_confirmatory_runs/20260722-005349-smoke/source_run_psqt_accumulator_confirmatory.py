"""Sealed target-level confirmation for repaired PSQT accumulators.

The runner supports a disposable smoke registry and a hash-verified sealed
registry.  It never selects hyperparameters from outcomes and writes the final
result whether promotion gates pass or fail.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
from numpy.linalg import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNROOT = HERE / "psqt_accumulator_confirmatory_runs"
sys.path.insert(0, str(HERE))

from lowdim_drift import drift_paper  # noqa: E402
from persistent_quantile_transport import midpoint_grid  # noqa: E402
from persistent_sliced_quantile_transport import (  # noqa: E402
    PersistentSlicedQuantileTransport,
    uniform_directions_2d,
)
from projected_quantile_accumulators import (  # noqa: E402
    ExactPooledProjectedAccumulator,
    RawReservoirProjectedAccumulator,
    invariant_tests as accumulator_invariant_tests,
    projected_quantile_table,
    reconstruct_from_quantile_table,
)
from psqt_confirmatory_targets import (  # noqa: E402
    FAMILIES,
    bridge_fraction,
    mode_metrics,
    sample_registry_target,
)
from standard_projected_kll import (  # noqa: E402
    ApacheKLLProjectedAccumulator,
    REQUIRED_DATASKETCHES_VERSION,
    invariant_tests as standard_kll_invariant_tests,
)

MASTER = 20260726
BOOTSTRAP_SEED = 2026072601
ARMS = (
    "paper-tau1",
    "historical-online-psqt",
    "pooled-rank-kll128",
    "pooled-rank-reservoir1024",
    "exact-pooled-ceiling",
)


@dataclass(frozen=True)
class Profile:
    name: str
    streams: int
    updates: int
    particles: int
    batch: int
    reference: int
    directions: int
    knots: int
    reconstruction_steps: int
    reconstruction_step_size: float
    bootstrap_draws: int


PROFILES = {
    "smoke": Profile(
        "smoke", 1, 40, 32, 32, 1024, 16, 32, 60, 0.5, 500),
    "confirmatory": Profile(
        "confirmatory", 5, 300, 64, 64, 4096, 32, 64, 100, 0.5,
        5000),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def verify_freeze(registry: Path, freeze_path: Path) -> dict:
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    expected_registry = freeze["registry"]
    if sha256(registry) != expected_registry["sha256"]:
        raise RuntimeError("sealed registry hash mismatch")
    for relative, expected in freeze["source_sha256"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen source hash mismatch: {relative}")
    if freeze["datasketches"] != REQUIRED_DATASKETCHES_VERSION:
        raise RuntimeError("freeze manifest has wrong DataSketches version")
    return freeze


class Artifact:
    def __init__(self, profile: Profile, registry: Path,
                 freeze: Path | None) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.directory = RUNROOT / f"{stamp}-{profile.name}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.time()
        self.lines: list[str] = []
        sources = [
            HERE / "run_psqt_accumulator_confirmatory.py",
            HERE / "standard_projected_kll.py",
            HERE / "projected_quantile_accumulators.py",
            HERE / "persistent_sliced_quantile_transport.py",
            HERE / "persistent_quantile_transport.py",
            HERE / "lowdim_drift.py",
            HERE / "psqt_confirmatory_targets.py",
            HERE / "generate_psqt_accumulator_registry.py",
            HERE / "PSQTAccumulatorConfirmatoryProtocol.md",
            HERE / "PSQTAccumulatorConfirmatoryRoadmap.md",
        ]
        self.manifest = {
            "status": (
                "sealed-confirmatory" if profile.name == "confirmatory"
                else "disposable-smoke"),
            "profile": asdict(profile),
            "arms": list(ARMS),
            "master_seed": MASTER,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "registry_sha256": sha256(registry),
            "freeze_sha256": sha256(freeze) if freeze else None,
            "git_commit": _git("rev-parse", "HEAD"),
            "git_status": _git("status", "--short").splitlines(),
            "python": sys.version,
            "numpy": np.__version__,
            "datasketches": importlib.metadata.version("datasketches"),
            "platform": platform.platform(),
            "command": sys.argv,
            "source_sha256": {
                str(path.relative_to(ROOT)): sha256(path) for path in sources
            },
        }
        for source in sources:
            shutil.copy2(source, self.directory / f"source_{source.name}")
        shutil.copy2(registry, self.directory / "registry.json")
        if freeze:
            shutil.copy2(freeze, self.directory / "freeze_manifest.json")

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


def initial_particles(kind: str, entry: dict, count: int,
                      rng: np.random.Generator) -> np.ndarray:
    scale = float(entry["scale"])
    if kind == "concentrated":
        return rng.normal(size=(count, 2)) * 0.08 * scale
    if kind == "far":
        center = 2.5 * scale * np.ones(2) / math.sqrt(2.0)
        return center + rng.normal(size=(count, 2)) * 0.10 * scale
    raise ValueError(kind)


def pairwise_mean_distance(a: np.ndarray, b: np.ndarray,
                           block: int = 128) -> float:
    total = 0.0
    for start in range(0, len(a), block):
        delta = a[start:start + block, None, :] - b[None, :, :]
        total += norm(delta, axis=2).sum()
    return float(total / (len(a) * len(b)))


@dataclass
class ReferenceStats:
    points: np.ndarray
    self_distance: float
    heldout_quantiles: np.ndarray


def reference_stats(points: np.ndarray, heldout: np.ndarray,
                    eval_grid: np.ndarray) -> ReferenceStats:
    return ReferenceStats(
        points=points,
        self_distance=pairwise_mean_distance(points, points),
        heldout_quantiles=projected_quantile_table(
            points, heldout, eval_grid, method="linear"),
    )


def quality_metrics(q: np.ndarray, reference: ReferenceStats,
                    heldout: np.ndarray, eval_grid: np.ndarray) -> dict:
    cross = pairwise_mean_distance(q, reference.points)
    within = pairwise_mean_distance(q, q)
    ed2 = max(0.0, 2.0 * cross - within - reference.self_distance)
    qtable = projected_quantile_table(
        q, heldout, eval_grid, method="linear")
    difference = qtable - reference.heldout_quantiles
    return {
        "ed2": float(ed2),
        "heldout_sw1": float(np.mean(np.abs(difference))),
        "heldout_projection_rmse": float(np.sqrt(np.mean(difference ** 2))),
    }


def target_table_rmse(table: np.ndarray, exact: np.ndarray) -> float:
    return float(np.sqrt(np.mean((table - exact) ** 2)))


def training_table_rmse(q: np.ndarray, directions: np.ndarray,
                        table: np.ndarray) -> float:
    current = projected_quantile_table(
        q, directions, midpoint_grid(table.shape[1]), method="linear")
    return float(np.sqrt(np.mean((current - table) ** 2)))


def common_cost(profile: Profile, q: np.ndarray) -> dict:
    return {
        "target_samples": profile.updates * profile.batch,
        "particle_bytes": int(q.nbytes),
    }


def run_paper(q0: np.ndarray, batches: np.ndarray,
              profile: Profile) -> tuple[np.ndarray, dict, None]:
    q = q0.copy()
    kernel_pairs = 0
    diverged = False
    started = time.perf_counter()
    for batch in batches:
        q += 0.15 * drift_paper(q, batch, 1.0, True)
        kernel_pairs += len(q) * (len(batch) + len(q))
        if not np.all(np.isfinite(q)) or norm(q) > 1e6:
            diverged = True
            break
    elapsed = time.perf_counter() - started
    cost = {
        **common_cost(profile, q),
        "diverged": int(diverged),
        "wall_seconds": elapsed,
        "kernel_pairs": kernel_pairs,
        "projection_dot_products": 0,
        "sort_work": 0.0,
        "sort_work_complete": 1,
        "reconstruction_sweeps": 0,
        "persistent_bytes": int(q.nbytes),
        "peak_working_bytes": int(q.nbytes + batches[0].nbytes),
        "retained_items": 0,
        "serialized_sketch_bytes": 0,
        "normalized_rank_error": float("nan"),
        "training_projection_rmse": float("nan"),
        "target_table_rmse": float("nan"),
    }
    return q, cost, None


def run_historical(q0: np.ndarray, batches: np.ndarray,
                   directions: np.ndarray, profile: Profile,
                   exact_table: np.ndarray) -> tuple[np.ndarray, dict, None]:
    model = PersistentSlicedQuantileTransport(
        q0, directions, knot_count=profile.knots, prior_batches=1.0,
        reconstruction_steps=3, reconstruction_step_size=0.5)
    sort_work = 0.0
    dots = 0
    started = time.perf_counter()
    for batch in batches:
        work = model.update(batch)
        sort_work += work.sort_work
        dots += work.projection_dot_products
    elapsed = time.perf_counter() - started
    q = model.particles.copy()
    persistent = int(
        model.particles.nbytes + model.directions.nbytes +
        model.target_quantiles.nbytes + model.grid.nbytes + 8)
    cost = {
        **common_cost(profile, q),
        "diverged": int(not np.all(np.isfinite(q))),
        "wall_seconds": elapsed,
        "kernel_pairs": 0,
        "projection_dot_products": dots,
        "sort_work": sort_work,
        "sort_work_complete": 1,
        "reconstruction_sweeps": 3 * profile.updates,
        "persistent_bytes": persistent,
        "peak_working_bytes": int(
            persistent + profile.batch * len(directions) * 8),
        "retained_items": 0,
        "serialized_sketch_bytes": 0,
        "normalized_rank_error": float("nan"),
        "training_projection_rmse": model.training_projection_rmse(),
        "target_table_rmse": target_table_rmse(
            model.target_quantiles, exact_table),
    }
    return q, cost, None


def run_kll(q0: np.ndarray, batches: np.ndarray,
            directions: np.ndarray, profile: Profile,
            exact_table: np.ndarray) -> tuple[np.ndarray, dict, list[bytes]]:
    accumulator = ApacheKLLProjectedAccumulator(
        2, directions, profile.knots, k=128)
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
    payloads = accumulator.serialize()
    cost = {
        **common_cost(profile, q),
        "diverged": int(not np.all(np.isfinite(q))),
        "wall_seconds": elapsed,
        "kernel_pairs": 0,
        "projection_dot_products": int(
            ledger.projection_dot_products + reconstruction_dots),
        "sort_work": reconstruction_sort,
        "sort_work_complete": 0,
        "reconstruction_sweeps": profile.reconstruction_steps,
        "persistent_bytes": int(ledger.persistent_bytes + q.nbytes),
        "peak_working_bytes": int(
            ledger.peak_working_bytes + q.nbytes + table.nbytes),
        "retained_items": ledger.retained_items,
        "serialized_sketch_bytes": ledger.serialized_bytes,
        "normalized_rank_error": ledger.normalized_rank_error,
        "training_projection_rmse": training_table_rmse(
            q, directions, table),
        "target_table_rmse": target_table_rmse(table, exact_table),
    }
    return q, cost, payloads


def run_reservoir(q0: np.ndarray, batches: np.ndarray,
                  directions: np.ndarray, profile: Profile,
                  exact_table: np.ndarray, seed: int) \
        -> tuple[np.ndarray, dict, None]:
    accumulator = RawReservoirProjectedAccumulator(
        2, directions, profile.knots, 1024, seed=seed)
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
    cost = {
        **common_cost(profile, q),
        "diverged": int(not np.all(np.isfinite(q))),
        "wall_seconds": elapsed,
        "kernel_pairs": 0,
        "projection_dot_products": int(
            ledger.projection_dot_products + reconstruction_dots),
        "sort_work": float(ledger.sort_work + reconstruction_sort),
        "sort_work_complete": 1,
        "reconstruction_sweeps": profile.reconstruction_steps,
        "persistent_bytes": int(8 * ledger.persistent_scalars + q.nbytes),
        "peak_working_bytes": int(
            8 * ledger.peak_working_scalars + q.nbytes + table.nbytes),
        "retained_items": ledger.retained_items,
        "serialized_sketch_bytes": 0,
        "normalized_rank_error": float("nan"),
        "training_projection_rmse": training_table_rmse(
            q, directions, table),
        "target_table_rmse": target_table_rmse(table, exact_table),
    }
    return q, cost, None


def run_exact(q0: np.ndarray, batches: np.ndarray,
              directions: np.ndarray, profile: Profile,
              exact_table: np.ndarray) -> tuple[np.ndarray, dict, None]:
    accumulator = ExactPooledProjectedAccumulator(
        2, directions, profile.knots)
    started = time.perf_counter()
    for batch in batches:
        accumulator.update(batch)
    table = accumulator.table()
    if not np.array_equal(table, exact_table):
        raise AssertionError("exact pooled arm disagrees with cached table")
    q, reconstruction_sort, reconstruction_dots = (
        reconstruct_from_quantile_table(
            q0, directions, table, steps=profile.reconstruction_steps,
            step_size=profile.reconstruction_step_size))
    elapsed = time.perf_counter() - started
    ledger = accumulator.ledger()
    cost = {
        **common_cost(profile, q),
        "diverged": int(not np.all(np.isfinite(q))),
        "wall_seconds": elapsed,
        "kernel_pairs": 0,
        "projection_dot_products": int(
            ledger.projection_dot_products + reconstruction_dots),
        "sort_work": float(ledger.sort_work + reconstruction_sort),
        "sort_work_complete": 1,
        "reconstruction_sweeps": profile.reconstruction_steps,
        "persistent_bytes": int(8 * ledger.persistent_scalars + q.nbytes),
        "peak_working_bytes": int(
            8 * ledger.peak_working_scalars + q.nbytes + table.nbytes),
        "retained_items": ledger.retained_items,
        "serialized_sketch_bytes": 0,
        "normalized_rank_error": float("nan"),
        "training_projection_rmse": training_table_rmse(
            q, directions, table),
        "target_table_rmse": 0.0,
    }
    return q, cost, None


def complete_row(q: np.ndarray, cost: dict, entry: dict,
                 reference: ReferenceStats, heldout: np.ndarray,
                 eval_grid: np.ndarray, reference_bridge: float) -> dict:
    if int(cost["diverged"]) or not np.all(np.isfinite(q)):
        penalty = 100.0 * float(entry["scale"])
        quality = {
            "ed2": penalty,
            "heldout_sw1": penalty,
            "heldout_projection_rmse": penalty,
        }
    else:
        quality = quality_metrics(q, reference, heldout, eval_grid)
    coverage, mass_l1, minority_mass, recovered = mode_metrics(q, entry)
    bridge = bridge_fraction(q, entry)
    excess = (
        bridge - reference_bridge
        if math.isfinite(bridge) and math.isfinite(reference_bridge)
        else float("nan"))
    return {
        **quality,
        "coverage": coverage,
        "mass_l1": mass_l1,
        "minority_particle_mass": minority_mass,
        "minority_recovered": recovered,
        "bridge_fraction": bridge,
        "reference_bridge_fraction": reference_bridge,
        "excess_bridge_fraction": excess,
        **cost,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.median(finite)) if finite else float("nan")


TARGET_METRICS = (
    "ed2", "heldout_sw1", "heldout_projection_rmse", "coverage",
    "mass_l1", "minority_particle_mass", "minority_recovered",
    "bridge_fraction", "excess_bridge_fraction", "diverged",
    "wall_seconds", "persistent_bytes", "peak_working_bytes",
    "target_table_rmse", "training_projection_rmse",
)


def aggregate_targets(rows: list[dict]) -> list[dict]:
    result = []
    targets = sorted({row["target"] for row in rows})
    for target in targets:
        family = next(row["family"] for row in rows if row["target"] == target)
        for arm in ARMS:
            group = [
                row for row in rows
                if row["target"] == target and row["arm"] == arm
            ]
            result.append({
                "target": target,
                "family": family,
                "arm": arm,
                **{
                    metric: finite_median([
                        float(row[metric]) for row in group])
                    for metric in TARGET_METRICS
                },
            })
    return result


def stratified_bootstrap(log_ratios: np.ndarray, families: np.ndarray,
                         draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = sorted(set(families.tolist()))
    indices = [np.flatnonzero(families == family) for family in unique]
    output = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected = np.concatenate([
            rng.choice(group, size=len(group), replace=True)
            for group in indices
        ])
        output[draw] = math.exp(float(np.mean(log_ratios[selected])))
    return output


def comparison(aggregates: list[dict], candidate: str, baseline: str,
               metric: str, draws: int, seed: int) -> tuple[dict, np.ndarray]:
    candidate_rows = {
        row["target"]: row for row in aggregates if row["arm"] == candidate
    }
    baseline_rows = {
        row["target"]: row for row in aggregates if row["arm"] == baseline
    }
    targets = sorted(candidate_rows)
    floor = 1e-12
    ratios = np.asarray([
        max(float(candidate_rows[target][metric]), floor) /
        max(float(baseline_rows[target][metric]), floor)
        for target in targets
    ])
    families = np.asarray([
        candidate_rows[target]["family"] for target in targets])
    logs = np.log(ratios)
    bootstrap = stratified_bootstrap(logs, families, draws, seed)
    family_ratios = {}
    for family in sorted(set(families.tolist())):
        family_ratios[family] = math.exp(float(np.mean(logs[families == family])))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "metric": metric,
        "geometric_mean_ratio": math.exp(float(np.mean(logs))),
        "median_target_ratio": float(np.median(ratios)),
        "target_win_fraction": float(np.mean(ratios < 1.0)),
        "ci95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "family_ratios": family_ratios,
        "target_count": len(targets),
    }, bootstrap


def summarize(rows: list[dict], aggregates: list[dict], profile: Profile,
              entries: dict[str, dict]) -> tuple[dict, dict[str, np.ndarray]]:
    names = {
        "kll_hist_ed2": ("pooled-rank-kll128", "historical-online-psqt", "ed2"),
        "kll_paper_ed2": ("pooled-rank-kll128", "paper-tau1", "ed2"),
        "res_hist_ed2": (
            "pooled-rank-reservoir1024", "historical-online-psqt", "ed2"),
        "res_paper_ed2": (
            "pooled-rank-reservoir1024", "paper-tau1", "ed2"),
        "kll_hist_sw1": (
            "pooled-rank-kll128", "historical-online-psqt", "heldout_sw1"),
        "kll_paper_sw1": (
            "pooled-rank-kll128", "paper-tau1", "heldout_sw1"),
        "res_hist_sw1": (
            "pooled-rank-reservoir1024", "historical-online-psqt",
            "heldout_sw1"),
        "exact_hist_ed2": (
            "exact-pooled-ceiling", "historical-online-psqt", "ed2"),
        "exact_paper_ed2": ("exact-pooled-ceiling", "paper-tau1", "ed2"),
    }
    comparisons = {}
    bootstrap = {}
    for index, (name, specification) in enumerate(names.items()):
        result, values = comparison(
            aggregates, *specification, profile.bootstrap_draws,
            BOOTSTRAP_SEED + index * 1009)
        comparisons[name] = result
        bootstrap[name] = values

    def rows_for(arm: str) -> list[dict]:
        return [row for row in rows if row["arm"] == arm]

    kll_rows = rows_for("pooled-rank-kll128")
    relevant_rare = [
        row for row in rows
        if row["family"] == "rare-mode" and
        float(entries[row["target"]]["parameters"]["minority"]) >= 0.05
    ]

    def recovery(arm: str) -> float:
        chosen = [
            int(row["minority_recovered"])
            for row in relevant_rare if row["arm"] == arm
        ]
        return float(np.mean(chosen)) if chosen else float("nan")

    bridge_rows = [
        row for row in kll_rows
        if math.isfinite(float(row["excess_bridge_fraction"])) and
        (row["family"] == "rare-mode" or
         (row["family"] == "dependence-trap" and
          entries[row["target"]]["parameters"]["kind"] == "offaxis-binary"))
    ]
    median_bridge = finite_median([
        float(row["excess_bridge_fraction"]) for row in bridge_rows])
    median_state = {
        arm: finite_median([
            float(row["persistent_bytes"]) for row in rows_for(arm)])
        for arm in ARMS
    }
    median_wall = {
        arm: finite_median([
            float(row["wall_seconds"]) for row in rows_for(arm)])
        for arm in ARMS
    }
    divergence = {
        arm: sum(int(row["diverged"]) for row in rows_for(arm))
        for arm in ARMS
    }

    kll_hist = comparisons["kll_hist_ed2"]
    kll_paper = comparisons["kll_paper_ed2"]
    res_hist = comparisons["res_hist_ed2"]
    res_paper = comparisons["res_paper_ed2"]
    gates_kll = {
        "ed2_vs_historical": (
            kll_hist["geometric_mean_ratio"] < 0.80 and
            kll_hist["ci95"][1] < 1.0),
        "ed2_vs_paper": (
            kll_paper["geometric_mean_ratio"] < 0.80 and
            kll_paper["ci95"][1] < 1.0),
        "sw1_vs_both": (
            comparisons["kll_hist_sw1"]["geometric_mean_ratio"] < 1.0 and
            comparisons["kll_paper_sw1"]["geometric_mean_ratio"] < 1.0),
        "target_wins": (
            kll_hist["target_win_fraction"] >= 0.70 and
            kll_paper["target_win_fraction"] >= 0.70),
        "family_robustness": max(
            kll_hist["family_ratios"].values()) <= 1.10,
        "zero_divergence": divergence["pooled-rank-kll128"] == 0,
        "rare_recovery": recovery("pooled-rank-kll128") >= 0.90,
        "bridge_occupancy": median_bridge <= 1.0 / profile.particles,
    }
    gates_reservoir = {
        "ed2_vs_historical": (
            res_hist["geometric_mean_ratio"] < 0.90 and
            res_hist["ci95"][1] < 1.0),
        "ed2_vs_paper": (
            res_paper["geometric_mean_ratio"] < 0.90 and
            res_paper["ci95"][1] < 1.0),
        "sw1_nonregression": (
            comparisons["res_hist_sw1"]["geometric_mean_ratio"] <= 1.05),
        "target_wins": (
            res_hist["target_win_fraction"] >= 0.60 and
            res_paper["target_win_fraction"] >= 0.60),
        "state": (
            median_state["pooled-rank-reservoir1024"] <=
            1.10 * median_state["historical-online-psqt"]),
        "wall_time": (
            median_wall["pooled-rank-reservoir1024"] <
            median_wall["historical-online-psqt"]),
        "zero_divergence": divergence["pooled-rank-reservoir1024"] == 0,
        "rare_recovery": (
            recovery("pooled-rank-reservoir1024") >=
            recovery("historical-online-psqt")),
    }
    return {
        "profile": asdict(profile),
        "comparisons": comparisons,
        "median_persistent_bytes": median_state,
        "median_wall_seconds": median_wall,
        "divergence_counts": divergence,
        "rare_recovery_05_10": {
            arm: recovery(arm) for arm in ARMS
        },
        "kll_median_excess_bridge": median_bridge,
        "kll_gates": gates_kll,
        "reservoir_gates": gates_reservoir,
        "kll_promoted": all(gates_kll.values()),
        "reservoir_promoted": all(gates_reservoir.values()),
        "general_improvement_claim": all(gates_kll.values()),
    }, bootstrap


def validate_registry(payload: dict, profile: Profile,
                      training: np.ndarray, heldout: np.ndarray) -> dict:
    if payload["target_count"] != len(payload["targets"]):
        raise ValueError("registry target count is inconsistent")
    if sorted(payload["families"]) != sorted(FAMILIES):
        raise ValueError("registry family list is inconsistent")
    if np.max(np.abs(training @ heldout.T)) >= 1.0 - 1e-12:
        raise ValueError("training and held-out directions overlap")
    parameter_hashes = [
        hashlib.sha256(json.dumps(
            entry["parameters"], sort_keys=True).encode()).hexdigest()
        for entry in payload["targets"]
    ]
    if len(parameter_hashes) != len(set(parameter_hashes)):
        raise ValueError("registry contains duplicate target parameters")

    stability = []
    first_by_family = {}
    for entry in payload["targets"]:
        first_by_family.setdefault(entry["family"], entry)
    check_count = min(profile.reference, 2048)
    grid = midpoint_grid(128)
    for entry in first_by_family.values():
        seed = int(entry["registry_seed"])
        first = sample_registry_target(
            entry, check_count, np.random.default_rng(seed + 7_000_001))
        second = sample_registry_target(
            entry, check_count, np.random.default_rng(seed + 7_000_002))
        a = reference_stats(first, heldout, grid)
        metrics = quality_metrics(second, a, heldout, grid)
        stability.append({
            "target": entry["id"],
            "family": entry["family"],
            "reference_ed2": metrics["ed2"],
            "reference_sw1": metrics["heldout_sw1"],
            "scale": float(entry["scale"]),
            "ed2_stable": metrics["ed2"] <= 0.08 * float(entry["scale"]),
            "sw1_stable": (
                metrics["heldout_sw1"] <= 0.08 * float(entry["scale"])),
        })
    if not all(row["ed2_stable"] and row["sw1_stable"] for row in stability):
        raise RuntimeError("target-only reference stability validation failed")
    return {
        "target_count": len(payload["targets"]),
        "families": sorted(first_by_family),
        "direction_overlap_max": float(np.max(np.abs(training @ heldout.T))),
        "duplicate_parameter_count": len(parameter_hashes) -
        len(set(parameter_hashes)),
        "reference_stability": stability,
        "pass": True,
    }


def append_kll_states(stream, index_rows: list[dict], output_index: int,
                      payloads: list[bytes]) -> None:
    for direction, payload in enumerate(payloads):
        offset = stream.tell()
        stream.write(payload)
        index_rows.append({
            "output_index": output_index,
            "direction": direction,
            "offset": offset,
            "length": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })


def make_visuals(directory: Path, payload: dict) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = ["target", "initial", *ARMS]
    for target, arrays in payload.items():
        if any(label not in arrays for label in labels):
            continue
        figure, axes = plt.subplots(
            1, len(labels), figsize=(3 * len(labels), 3.2),
            sharex=True, sharey=True)
        combined = np.vstack([arrays[label] for label in labels])
        low = np.quantile(combined, 0.005, axis=0)
        high = np.quantile(combined, 0.995, axis=0)
        center = 0.5 * (low + high)
        radius = 0.55 * max(*(high - low), 1e-6)
        for axis, label in zip(axes, labels):
            axis.scatter(
                arrays[label][:, 0], arrays[label][:, 1],
                s=6, alpha=0.65, linewidths=0)
            axis.set_title(label, fontsize=7)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_aspect("equal")
            axis.grid(alpha=0.15)
        figure.suptitle(target)
        figure.tight_layout()
        figure.savefig(directory / f"visual_{target}.png", dpi=170)
        plt.close(figure)


def warm_up(profile: Profile) -> None:
    directions = uniform_directions_2d(profile.directions)
    rng = np.random.default_rng(MASTER)
    q = rng.normal(size=(profile.particles, 2))
    batch = rng.normal(size=(profile.batch, 2))
    drift_paper(q, batch, 1.0, True)
    accumulator = ApacheKLLProjectedAccumulator(
        2, directions, profile.knots, k=128)
    accumulator.update(batch)
    table = accumulator.table()
    reconstruct_from_quantile_table(
        q, directions, table, steps=1, step_size=0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    registry = args.registry.resolve()
    payload = json.loads(registry.read_text(encoding="utf-8"))
    if profile.name == "confirmatory":
        if args.freeze is None:
            raise ValueError("confirmatory execution requires --freeze")
        if payload["status"] != "sealed-fresh":
            raise ValueError("confirmatory profile requires sealed registry")
        verify_freeze(registry, args.freeze.resolve())
    elif payload["status"] != "disposable-smoke":
        raise ValueError("smoke profile requires disposable registry")

    standard_kll_invariant_tests()
    accumulator_invariant_tests()
    warm_up(profile)
    artifact = Artifact(profile, registry, args.freeze.resolve()
                        if args.freeze else None)
    artifact.log("# PSQT accumulator confirmation")
    artifact.log()
    artifact.log(
        "Status: **sealed confirmatory execution**."
        if profile.name == "confirmatory"
        else "Status: disposable structural smoke run.")
    artifact.log()
    artifact.log("## Preflight")
    artifact.log("- Apache KLL invariants: PASS")
    artifact.log("- accumulator invariants: PASS")

    directions = uniform_directions_2d(profile.directions)
    heldout = uniform_directions_2d(64, phase=np.pi / 128)
    eval_grid = midpoint_grid(128)
    validation = validate_registry(payload, profile, directions, heldout)
    with (artifact.directory / "registry_validation.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(validation, stream, indent=2)
    artifact.log("- target-only registry validation: PASS")
    artifact.log(
        f"- registry SHA-256: `{sha256(registry)}`")

    entries = {entry["id"]: entry for entry in payload["targets"]}
    rows: list[dict] = []
    outputs: list[np.ndarray] = []
    references: list[np.ndarray] = []
    reference_ids: list[str] = []
    kll_index: list[dict] = []
    visuals: dict[str, dict[str, np.ndarray]] = {}
    first_family_target = {}
    for entry in payload["targets"]:
        first_family_target.setdefault(entry["family"], entry["id"])

    kll_stream = (artifact.directory / "kll_states.bin").open("wb")
    try:
        for target_index, entry in enumerate(payload["targets"]):
            target_seed = int(entry["registry_seed"])
            reference_points = sample_registry_target(
                entry, profile.reference,
                np.random.default_rng(target_seed + 9_000_001))
            reference = reference_stats(reference_points, heldout, eval_grid)
            reference_bridge = bridge_fraction(reference_points, entry)
            references.append(reference_points.astype(np.float32))
            reference_ids.append(entry["id"])
            for stream_index in range(profile.streams):
                data_seed = target_seed + 100_003 + stream_index * 1009
                data_rng = np.random.default_rng(data_seed)
                batches = np.stack([
                    sample_registry_target(entry, profile.batch, data_rng)
                    for _ in range(profile.updates)
                ])
                pooled = batches.reshape(-1, 2)
                exact_table = projected_quantile_table(
                    pooled, directions, midpoint_grid(profile.knots))
                for init_index, init in enumerate(("concentrated", "far")):
                    q0_seed = (
                        target_seed + 200_003 + stream_index * 10_007 +
                        init_index * 1009)
                    q0 = initial_particles(
                        init, entry, profile.particles,
                        np.random.default_rng(q0_seed))
                    arm_functions = (
                        ("paper-tau1", lambda: run_paper(
                            q0, batches, profile)),
                        ("historical-online-psqt", lambda: run_historical(
                            q0, batches, directions, profile, exact_table)),
                        ("pooled-rank-kll128", lambda: run_kll(
                            q0, batches, directions, profile, exact_table)),
                        ("pooled-rank-reservoir1024", lambda: run_reservoir(
                            q0, batches, directions, profile, exact_table,
                            target_seed + 300_007 + stream_index * 10_007 +
                            init_index * 1009)),
                        ("exact-pooled-ceiling", lambda: run_exact(
                            q0, batches, directions, profile, exact_table)),
                    )
                    trial_visuals = {}
                    for arm, function in arm_functions:
                        q, cost, sketches = function()
                        completed = complete_row(
                            q, cost, entry, reference, heldout, eval_grid,
                            reference_bridge)
                        output_index = len(outputs)
                        outputs.append(q.astype(np.float32))
                        rows.append({
                            "output_index": output_index,
                            "arm": arm,
                            "target": entry["id"],
                            "family": entry["family"],
                            "init": init,
                            "stream": stream_index,
                            "target_seed": target_seed,
                            "data_seed": data_seed,
                            "q0_seed": q0_seed,
                            **completed,
                        })
                        if sketches is not None:
                            append_kll_states(
                                kll_stream, kll_index, output_index, sketches)
                        trial_visuals[arm] = q
                    if (stream_index == 0 and init == "concentrated" and
                            first_family_target[entry["family"]] == entry["id"]):
                        visuals[entry["id"]] = {
                            "target": reference_points,
                            "initial": q0,
                            **trial_visuals,
                        }
            artifact.log(
                f"- completed target {target_index + 1}/"
                f"{len(payload['targets'])}: {entry['id']}")
    finally:
        kll_stream.close()

    expected = len(payload["targets"]) * profile.streams * 2 * len(ARMS)
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} rows, found {len(rows)}")
    write_csv(artifact.directory / "rows.csv", rows)
    write_csv(artifact.directory / "kll_state_index.csv", kll_index)
    np.save(
        artifact.directory / "outputs.npy",
        np.stack(outputs).astype(np.float32), allow_pickle=False)
    np.save(
        artifact.directory / "references.npy",
        np.stack(references).astype(np.float32), allow_pickle=False)
    (artifact.directory / "reference_ids.json").write_text(
        json.dumps(reference_ids, indent=2), encoding="utf-8")

    aggregates = aggregate_targets(rows)
    write_csv(artifact.directory / "target_aggregates.csv", aggregates)
    summary, bootstrap = summarize(rows, aggregates, profile, entries)
    with (artifact.directory / "summary.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    np.savez_compressed(
        artifact.directory / "bootstrap_draws.npz", **bootstrap)
    make_visuals(artifact.directory, visuals)

    artifact.log()
    artifact.log("## Target-level outcomes")
    for name in ("kll_hist_ed2", "kll_paper_ed2",
                 "res_hist_ed2", "res_paper_ed2"):
        result = summary["comparisons"][name]
        artifact.log(
            f"- `{name}`: ratio `{result['geometric_mean_ratio']:.4f}`, "
            f"95% CI `[{result['ci95'][0]:.4f}, "
            f"{result['ci95'][1]:.4f}]`, wins "
            f"`{result['target_win_fraction']:.1%}`")
    artifact.log("- KLL gates:")
    for gate, passed in summary["kll_gates"].items():
        artifact.log(f"  - `{gate}`: **{'PASS' if passed else 'FAIL'}**")
    artifact.log("- Reservoir gates:")
    for gate, passed in summary["reservoir_gates"].items():
        artifact.log(f"  - `{gate}`: **{'PASS' if passed else 'FAIL'}**")
    artifact.log(
        "- KLL promotion: "
        f"**{'PASS' if summary['kll_promoted'] else 'FAIL'}**")
    artifact.log(
        "- Reservoir promotion: "
        f"**{'PASS' if summary['reservoir_promoted'] else 'FAIL'}**")
    artifact.log(
        "- general low-dimensional improvement claim: "
        f"**{'SUPPORTED' if summary['general_improvement_claim'] else 'NOT SUPPORTED'}**")
    artifact.log()
    artifact.log("## Interpretation boundary")
    artifact.log()
    artifact.log(
        "This protocol concerns a nonparametric 2D particle generator under "
        "the declared sample and particle budgets. It is not an ImageNet, "
        "encoder-feature, neural-generator, or all-bandwidth paper claim.")
    artifact.finish()
    print(f"artifact: {artifact.directory}", flush=True)


if __name__ == "__main__":
    main()
