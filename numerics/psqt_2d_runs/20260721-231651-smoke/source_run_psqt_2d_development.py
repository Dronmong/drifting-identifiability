"""Paired two-dimensional development benchmark for PSQT.

This runner is intentionally separate from every frozen confirmatory runner.
It tests the mechanism described in ``PSQTHigherDimImplementationPlan.md`` on
the repository's exact paper estimator and low-dimensional target machinery.

Usage:

    uv run --with numpy --with scipy --with matplotlib \
        python numerics/run_psqt_2d_development.py --profile smoke
    uv run --with numpy --with scipy --with matplotlib \
        python numerics/run_psqt_2d_development.py --profile screen
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
RUNROOT = HERE / "psqt_2d_runs"
sys.path.insert(0, str(HERE))

from lowdim_drift import (  # noqa: E402
    TargetSpec,
    circles_target,
    drift_paper,
    energy_distance2,
    field_invariants,
    gauss_mixture,
    mode_metrics,
    moons_target,
    ring_target,
    skew_target,
)
from persistent_quantile_transport import empirical_quantiles, midpoint_grid  # noqa: E402
from persistent_sliced_quantile_transport import (  # noqa: E402
    PersistentSlicedQuantileTransport,
    coordinate_directions,
    invariant_tests as psqt_invariant_tests,
    projected_quantile_rmse,
    uniform_directions_2d,
)

MASTER = 20260722


@dataclass(frozen=True)
class Profile:
    name: str
    updates: int
    seeds: int
    particles: int
    batch: int
    reference: int
    cross_reference: int
    track_every: int
    target_limit: int | None


PROFILES = {
    "smoke": Profile("smoke", 40, 1, 32, 32, 256, 128, 10, 4),
    "screen": Profile("screen", 300, 5, 64, 64, 1024, 256, 20, None),
}

PAPER_TAUS = (0.2, 0.5, 1.0, 2.0)


@dataclass(frozen=True)
class PSQTConfig:
    directions: int
    knots: int
    reconstruction_steps: int
    reconstruction_step_size: float

    @property
    def name(self) -> str:
        return (
            f"psqt-L{self.directions}-K{self.knots}-"
            f"R{self.reconstruction_steps}-e{self.reconstruction_step_size:g}")


def psqt_configs(profile: Profile) -> list[PSQTConfig]:
    knots = profile.particles
    if profile.name == "smoke":
        return [
            PSQTConfig(8, knots, 2, 0.5),
            PSQTConfig(16, knots, 3, 0.5),
        ]
    return [
        PSQTConfig(8, knots, 3, 0.5),
        PSQTConfig(16, knots, 3, 0.5),
        PSQTConfig(32, knots, 3, 0.5),
        PSQTConfig(16, knots, 6, 0.5),
        PSQTConfig(16, knots, 3, 1.0),
    ]


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
    def __init__(self, profile: Profile, configs: list[PSQTConfig]) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.directory = RUNROOT / f"{stamp}-{profile.name}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.started = time.time()
        self.lines: list[str] = []
        sources = [
            HERE / "run_psqt_2d_development.py",
            HERE / "persistent_sliced_quantile_transport.py",
            HERE / "persistent_quantile_transport.py",
            HERE / "lowdim_drift.py",
            HERE / "PSQTHigherDimImplementationPlan.md",
        ]
        self.manifest = {
            "protocol_status": "development-not-frozen",
            "master_seed": MASTER,
            "profile": asdict(profile),
            "psqt_configs": [asdict(config) for config in configs],
            "paper_taus": PAPER_TAUS,
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


def correlated_gaussian(name: str, correlation: float) -> TargetSpec:
    covariance = np.asarray([[1.0, correlation], [correlation, 1.0]])
    factor = np.linalg.cholesky(covariance) * 0.55

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=(n, 2)) @ factor.T

    return TargetSpec(name, 2, sampler, "correlated", scale=1.5)


def diagonal_mixture(name: str, *, anti: bool = False) -> TargetSpec:
    second = -1.0 if anti else 1.0
    means = np.asarray([[-1.0, -second], [1.0, second]])
    weights = np.asarray([0.5, 0.5])

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        labels = rng.integers(0, 2, size=n)
        return means[labels] + rng.normal(size=(n, 2)) * 0.10

    return TargetSpec(name, 2, sampler, "dependence", means, weights, 1.5)


def development_targets() -> list[TargetSpec]:
    return [
        gauss_mixture("PS2-GMM4-equal", 4, 2, 0.12),
        gauss_mixture("PS2-GMM5-unequal", 5, 2, 0.11, unequal=True),
        gauss_mixture(
            "PS2-GMM4-hetero", 4, 2, [0.07, 0.12, 0.20, 0.30]),
        diagonal_mixture("PS2-diagonal-dependence"),
        correlated_gaussian("PS2-correlated-rho085", 0.85),
        ring_target("PS2-ring", radius=1.0, width=0.045),
        circles_target("PS2-circles", r1=0.45, r2=1.0, width=0.035),
        moons_target("PS2-moons", scale=1.0, noise=0.065),
        skew_target("PS2-skew", d=2),
    ]


def initial_particles(kind: str, target: TargetSpec, count: int,
                      rng: np.random.Generator) -> np.ndarray:
    if kind == "concentrated":
        return rng.normal(size=(count, 2)) * 0.08 * target.scale
    if kind == "far":
        center = 2.5 * target.scale * np.ones(2) / math.sqrt(2.0)
        return center + rng.normal(size=(count, 2)) * 0.10 * target.scale
    raise ValueError(kind)


def fixed_sliced_w1(a: np.ndarray, b: np.ndarray,
                    directions: np.ndarray, knot_count: int = 128) -> float:
    grid = midpoint_grid(knot_count)
    total = 0.0
    for theta in directions:
        qa = empirical_quantiles(a @ theta, grid)
        qb = empirical_quantiles(b @ theta, grid)
        total += float(np.mean(np.abs(qa - qb)))
    return total / len(directions)


def _endpoint_metrics(q: np.ndarray, target: TargetSpec,
                      reference: np.ndarray,
                      heldout_directions: np.ndarray) -> dict[str, float]:
    coverage, mass_l1 = mode_metrics(q, target)
    return {
        "ed2": float(energy_distance2(q, reference)),
        "heldout_sw1": fixed_sliced_w1(q, reference, heldout_directions),
        "heldout_projection_rmse": projected_quantile_rmse(
            q, reference, heldout_directions, knot_count=128),
        "coverage": float(coverage),
        "mass_l1": float(mass_l1),
    }


def _event_update(q: np.ndarray, cross_reference: np.ndarray, step: int,
                  threshold: float, current: int | None) -> int | None:
    if current is None and energy_distance2(q, cross_reference) < threshold:
        return step
    return current


def run_paper(q0: np.ndarray, batches: np.ndarray, target: TargetSpec,
              reference: np.ndarray, cross_reference: np.ndarray,
              heldout_directions: np.ndarray, tau: float,
              track_every: int) -> tuple[dict, np.ndarray]:
    q = q0.copy()
    started = time.perf_counter()
    kernel_pairs = 0
    event: int | None = None
    threshold = 0.05 * target.scale
    diverged = False
    eta = 0.15 * tau
    for index, batch in enumerate(batches, start=1):
        q += eta * drift_paper(q, batch, tau, True)
        kernel_pairs += len(q) * (len(batch) + len(q))
        if not np.all(np.isfinite(q)) or norm(q) > 1e6:
            diverged = True
            break
        if index == 1 or index % track_every == 0 or index == len(batches):
            event = _event_update(
                q, cross_reference, index, threshold, event)
    elapsed = time.perf_counter() - started
    metrics = _endpoint_metrics(
        q, target, reference, heldout_directions) if not diverged else {
            "ed2": float("inf"), "heldout_sw1": float("inf"),
            "heldout_projection_rmse": float("inf"),
            "coverage": float("nan"), "mass_l1": float("nan"),
        }
    metrics.update({
        "training_projection_rmse": float("nan"),
        "event_time": event if event is not None else len(batches),
        "censored": int(event is None),
        "diverged": int(diverged),
        "wall_seconds": elapsed,
        "target_samples": int(len(batches) * batches.shape[1]),
        "kernel_pairs": int(kernel_pairs),
        "sort_work": 0.0,
        "projection_dot_products": 0,
        "stored_scalars": int(q.size),
    })
    return metrics, q


def run_psqt(q0: np.ndarray, batches: np.ndarray, target: TargetSpec,
             reference: np.ndarray, cross_reference: np.ndarray,
             heldout_directions: np.ndarray, directions: np.ndarray,
             knot_count: int, reconstruction_steps: int,
             reconstruction_step_size: float,
             track_every: int) -> tuple[dict, np.ndarray]:
    model = PersistentSlicedQuantileTransport(
        q0, directions, knot_count=knot_count, prior_batches=1.0,
        reconstruction_steps=reconstruction_steps,
        reconstruction_step_size=reconstruction_step_size)
    started = time.perf_counter()
    sort_work = 0.0
    dot_products = 0
    target_samples = 0
    event: int | None = None
    threshold = 0.05 * target.scale
    diverged = False
    for index, batch in enumerate(batches, start=1):
        work = model.update(batch)
        sort_work += work.sort_work
        dot_products += work.projection_dot_products
        target_samples += work.target_samples
        q = model.particles
        if not np.all(np.isfinite(q)) or norm(q) > 1e6:
            diverged = True
            break
        if index == 1 or index % track_every == 0 or index == len(batches):
            event = _event_update(
                q, cross_reference, index, threshold, event)
    elapsed = time.perf_counter() - started
    q = model.particles.copy()
    metrics = _endpoint_metrics(
        q, target, reference, heldout_directions) if not diverged else {
            "ed2": float("inf"), "heldout_sw1": float("inf"),
            "heldout_projection_rmse": float("inf"),
            "coverage": float("nan"), "mass_l1": float("nan"),
        }
    metrics.update({
        "training_projection_rmse": (
            model.training_projection_rmse() if not diverged else float("inf")),
        "event_time": event if event is not None else len(batches),
        "censored": int(event is None),
        "diverged": int(diverged),
        "wall_seconds": elapsed,
        "target_samples": int(target_samples),
        "kernel_pairs": 0,
        "sort_work": float(sort_work),
        "projection_dot_products": int(dot_products),
        "stored_scalars": model.stored_scalars,
    })
    return metrics, q


def _finite_median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def summarize(rows: list[dict], psqt_names: list[str]) -> dict:
    cell_medians: dict[tuple[str, str, str], dict[str, float]] = {}
    arms = sorted({row["arm"] for row in rows})
    targets = sorted({row["target"] for row in rows})
    inits = sorted({row["init"] for row in rows})
    for arm in arms:
        for target in targets:
            for init in inits:
                group = [row for row in rows if row["arm"] == arm and
                         row["target"] == target and row["init"] == init]
                if not group:
                    continue
                cell_medians[(arm, target, init)] = {
                    metric: _finite_median([float(row[metric]) for row in group])
                    for metric in ("ed2", "heldout_sw1", "mass_l1",
                                   "event_time", "wall_seconds")
                }

    def geo_ratio(candidate: str, baseline: str, metric: str) -> float:
        ratios = []
        for target in targets:
            for init in inits:
                top = cell_medians[(candidate, target, init)][metric]
                bottom = cell_medians[(baseline, target, init)][metric]
                ratios.append(math.log(max(top, 1e-12) / max(bottom, 1e-12)))
        return float(math.exp(np.mean(ratios)))

    paper_names = [f"paper-tau{tau:g}" for tau in PAPER_TAUS]
    paper_scores = {
        arm: float(np.mean([
            math.log(max(values["ed2"], 1e-12))
            for (name, _, _), values in cell_medians.items() if name == arm
        ])) for arm in paper_names
    }
    selected_paper = min(paper_scores, key=paper_scores.get)
    psqt_scores = {
        arm: float(np.mean([
            math.log(max(values["ed2"], 1e-12))
            for (name, _, _), values in cell_medians.items() if name == arm
        ])) for arm in psqt_names
    }
    selected_psqt = min(psqt_scores, key=psqt_scores.get)

    oracle_ed = {}
    for target in targets:
        for init in inits:
            oracle_ed[(target, init)] = min(
                cell_medians[(arm, target, init)]["ed2"]
                for arm in paper_names)
    oracle_ratio = math.exp(float(np.mean([
        math.log(max(cell_medians[(selected_psqt, target, init)]["ed2"], 1e-12) /
                 max(oracle_ed[(target, init)], 1e-12))
        for target in targets for init in inits
    ])))

    wins = sum(
        cell_medians[(selected_psqt, target, init)]["ed2"] <
        cell_medians[(selected_paper, target, init)]["ed2"]
        for target in targets for init in inits)
    family_ratios = {}
    family_by_target = {row["target"]: row["family"] for row in rows}
    for family in sorted(set(family_by_target.values())):
        logs = []
        for target in targets:
            if family_by_target[target] != family:
                continue
            for init in inits:
                logs.append(math.log(max(
                    cell_medians[(selected_psqt, target, init)]["ed2"], 1e-12) /
                    max(cell_medians[(selected_paper, target, init)]["ed2"],
                        1e-12)))
        family_ratios[family] = math.exp(float(np.mean(logs)))

    return {
        "development_selected_paper": selected_paper,
        "development_selected_psqt": selected_psqt,
        "ed2_ratio_vs_selected_paper": geo_ratio(
            selected_psqt, selected_paper, "ed2"),
        "heldout_sw1_ratio_vs_selected_paper": geo_ratio(
            selected_psqt, selected_paper, "heldout_sw1"),
        "ed2_ratio_vs_coordinate_pqt": geo_ratio(
            selected_psqt, "coordinate-pqt", "ed2"),
        "ed2_ratio_vs_paper_cell_oracle": oracle_ratio,
        "cell_wins_vs_selected_paper": wins,
        "cell_count": len(targets) * len(inits),
        "family_ed2_ratios_vs_selected_paper": family_ratios,
        "cell_medians": {
            "|".join(key): value for key, value in cell_medians.items()
        },
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_visuals(directory: Path, payload: dict,
                 selected_paper: str, selected_psqt: str) -> None:
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    for target, arrays in payload.items():
        required = ["target", "initial", selected_paper,
                    "coordinate-pqt", selected_psqt]
        if any(name not in arrays for name in required):
            continue
        figure, axes = plt.subplots(1, 5, figsize=(15, 3.2), sharex=True,
                                    sharey=True)
        labels = ["target", "initial", selected_paper,
                  "coordinate-pqt", selected_psqt]
        combined = np.vstack([arrays[name] for name in required])
        low = np.quantile(combined, 0.005, axis=0)
        high = np.quantile(combined, 0.995, axis=0)
        center = 0.5 * (low + high)
        radius = 0.55 * max(*(high - low), 1e-6)
        for axis, name in zip(axes, labels):
            points = arrays[name]
            axis.scatter(points[:, 0], points[:, 1], s=7, alpha=0.65,
                         linewidths=0)
            axis.set_title(name, fontsize=8)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_aspect("equal")
            axis.grid(alpha=0.15)
        figure.suptitle(target)
        figure.tight_layout()
        safe = target.replace("/", "-")
        figure.savefig(directory / f"visual_{safe}.png", dpi=170)
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    configs = psqt_configs(profile)
    artifact = Artifact(profile, configs)
    artifact.log("# PSQT 2D development results")
    artifact.log()
    artifact.log("Status: exploratory development; not a frozen confirmation.")
    artifact.log()
    artifact.log("## Invariants")
    psqt_invariant_tests()
    artifact.log("- PSQT mathematical/implementation invariants: PASS")
    field_messages: list[str] = []
    field_invariants(field_messages.append)
    for message in field_messages:
        artifact.log(f"- {message.strip()}")

    targets = development_targets()
    if profile.target_limit is not None:
        targets = targets[:profile.target_limit]
    heldout = uniform_directions_2d(64, phase=np.pi / 128)
    rows: list[dict] = []
    visuals: dict[str, dict[str, np.ndarray]] = {}
    psqt_names = [config.name for config in configs]
    arms = [*(f"paper-tau{tau:g}" for tau in PAPER_TAUS),
            "coordinate-pqt", *psqt_names]
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
                reference = target.sample(
                    profile.reference, np.random.default_rng(base_seed + 3))
                cross = target.sample(
                    profile.cross_reference,
                    np.random.default_rng(base_seed + 4))
                outputs: dict[str, np.ndarray] = {}
                for tau in PAPER_TAUS:
                    arm = f"paper-tau{tau:g}"
                    metrics, output = run_paper(
                        q0, batches, target, reference, cross, heldout, tau,
                        profile.track_every)
                    outputs[arm] = output
                    rows.append({
                        "arm": arm, "kind": "paper", "target": target.name,
                        "family": target.family, "init": init, "seed": seed,
                        **metrics,
                    })
                metrics, output = run_psqt(
                    q0, batches, target, reference, cross, heldout,
                    coordinate_directions(2), profile.particles, 1, 1.0,
                    profile.track_every)
                outputs["coordinate-pqt"] = output
                rows.append({
                    "arm": "coordinate-pqt", "kind": "coordinate-negative-control",
                    "target": target.name, "family": target.family,
                    "init": init, "seed": seed, **metrics,
                })
                for config in configs:
                    metrics, output = run_psqt(
                        q0, batches, target, reference, cross, heldout,
                        uniform_directions_2d(config.directions), config.knots,
                        config.reconstruction_steps,
                        config.reconstruction_step_size,
                        profile.track_every)
                    outputs[config.name] = output
                    rows.append({
                        "arm": config.name, "kind": "psqt",
                        "target": target.name, "family": target.family,
                        "init": init, "seed": seed, **metrics,
                    })
                if seed == 0 and init == "concentrated":
                    visuals[target.name] = {
                        "target": reference, "initial": q0, **outputs}
                completed += 1
                artifact.log(
                    f"- completed paired cell {completed}/{total}: "
                    f"{target.name}/{init}/seed{seed}")

    if not rows:
        raise AssertionError("runner produced no rows")
    expected = total * len(arms)
    if len(rows) != expected:
        raise AssertionError(f"expected {expected} rows, found {len(rows)}")
    write_rows(artifact.directory / "rows.csv", rows)
    summary = summarize(rows, psqt_names)
    with (artifact.directory / "summary.json").open(
            "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    make_visuals(
        artifact.directory, visuals,
        summary["development_selected_paper"],
        summary["development_selected_psqt"])

    artifact.log()
    artifact.log("## Development selection")
    artifact.log(
        f"- selected paper arm: `{summary['development_selected_paper']}`")
    artifact.log(
        f"- selected PSQT arm: `{summary['development_selected_psqt']}`")
    artifact.log(
        "- target-balanced ED2 ratio, PSQT / selected paper: "
        f"`{summary['ed2_ratio_vs_selected_paper']:.4f}`")
    artifact.log(
        "- held-out SW1 ratio, PSQT / selected paper: "
        f"`{summary['heldout_sw1_ratio_vs_selected_paper']:.4f}`")
    artifact.log(
        "- ED2 ratio, PSQT / coordinate-PQT: "
        f"`{summary['ed2_ratio_vs_coordinate_pqt']:.4f}`")
    artifact.log(
        "- ED2 ratio, PSQT / per-cell paper oracle: "
        f"`{summary['ed2_ratio_vs_paper_cell_oracle']:.4f}`")
    artifact.log(
        "- cell wins versus selected paper: "
        f"`{summary['cell_wins_vs_selected_paper']}/"
        f"{summary['cell_count']}`")
    artifact.log("- family ED2 ratios versus selected paper:")
    for family, ratio in summary[
            "family_ed2_ratios_vs_selected_paper"].items():
        artifact.log(f"  - `{family}`: `{ratio:.4f}`")
    artifact.log()
    artifact.log("## Interpretation boundary")
    artifact.log()
    artifact.log(
        "These outcomes are development evidence on reused target families. "
        "They select hyperparameters and cannot support a confirmatory claim. "
        "A fresh registry, frozen arm, target-level uncertainty, and full "
        "sample/work gates are required next.")
    artifact.finish()
    print(f"artifact: {artifact.directory}", flush=True)


if __name__ == "__main__":
    main()

