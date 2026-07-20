"""Fresh D3b factorial attribution study.

The historical D0--D3 run remains untouched.  This runner implements the
pre-registered corrective protocol in LowDimAttributionProtocol.md:

    uv run --with numpy --with scipy python numerics/lowdim_attribution.py \
        validation --profile standard
    uv run --with numpy --with scipy python numerics/lowdim_attribution.py \
        test --profile standard
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lowdim_drift import (  # noqa: E402
    MASTER, Policy, Run, StepDecision, TargetSpec, WorkCounter,
    estimate_geometry, field_invariants, gauss_mixture, init_cloud, km_median,
    ring_target, circles_target, moons_target, support_residual, train,
)

PROTOCOL = HERE / "LowDimAttributionProtocol.md"
RUNROOT = HERE / "lowdim_runs"
BASE_TAU = 0.35
BASE_ETA = 0.0525


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    validation_seeds: int
    test_seeds: int
    ref_final: int
    ref_cross: int
    N: int
    batch: int


PROFILES = {
    "smoke": Profile("smoke", 80, 2, 3, 256, 128, 24, 32),
    "standard": Profile("standard", 400, 6, 20, 1024, 256, 48, 64),
}


# ---------------------------------------------------------------------------
# Fresh targets: no configuration below occurred in historical D3.
# ---------------------------------------------------------------------------


def banana_target(name: str, curvature: float, noise: float = 0.12,
                  xscale: float = 0.8) -> TargetSpec:
    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        x = rng.normal(size=n) * xscale
        y = curvature * (x ** 2 - xscale ** 2) + \
            rng.normal(size=n) * noise
        return np.stack([x, y], axis=1)

    return TargetSpec(name, 2, sampler, "banana",
                      scale=2.5 * max(xscale, curvature))


def sine_target(name: str, width: float = 0.10) -> TargetSpec:
    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        x = rng.uniform(-1.8, 1.8, size=n)
        y = 0.55 * np.sin(2.2 * x) + rng.normal(size=n) * width
        return np.stack([x, y], axis=1)

    return TargetSpec(name, 2, sampler, "sine", scale=2.0)


def attribution_validation_targets() -> list[TargetSpec]:
    return [
        gauss_mixture("AV-1d-K4-uneq", 4, 1, 0.18, L=0.8,
                      unequal=True),
        gauss_mixture("AV-2d-K5-hetero", 5, 2,
                      [0.10, 0.14, 0.18, 0.22, 0.26], L=1.1),
        ring_target("AV-ring", radius=0.8, width=0.07),
        circles_target("AV-circles", r1=0.30, r2=0.85, width=0.06),
        moons_target("AV-moons", scale=0.75, noise=0.10),
        banana_target("AV-banana", curvature=0.45),
    ]


def attribution_test_targets() -> list[TargetSpec]:
    return [
        gauss_mixture("AT-1d-K6-eq", 6, 1, 0.11, L=0.9),
        gauss_mixture("AT-2d-K4-uneq", 4, 2, 0.22, L=1.2,
                      unequal=True),
        ring_target("AT-ring", radius=1.25, width=0.035),
        circles_target("AT-circles", r1=0.45, r2=1.25, width=0.05),
        moons_target("AT-moons", scale=1.25, noise=0.05),
        banana_target("AT-banana", curvature=0.70, noise=0.10,
                      xscale=0.9),
        sine_target("AT-sine"),
    ]


# ---------------------------------------------------------------------------
# Factorial policies and audit metadata.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorSpec:
    label: str
    tau_mode: str       # fixed | geometry
    mask_mode: str      # on | auto
    eta_mode: str       # fixed | scaled

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "tau_mode": self.tau_mode,
            "mask_mode": self.mask_mode,
            "eta_mode": self.eta_mode,
        }


BASE_SPEC = FactorSpec("base", "fixed", "on", "fixed")


def factorial_specs() -> list[FactorSpec]:
    out = []
    for tau_mode in ("fixed", "geometry"):
        for mask_mode in ("on", "auto"):
            for eta_mode in ("fixed", "scaled"):
                label = f"tau-{tau_mode}__mask-{mask_mode}__eta-{eta_mode}"
                out.append(FactorSpec(label, tau_mode, mask_mode, eta_mode))
    return out


class FactorPolicy(Policy):
    def __init__(self, spec: FactorSpec):
        self.spec = spec
        self.name = spec.label
        self.geo: dict | None = None
        self.N = 0
        self.last = StepDecision(BASE_TAU, BASE_ETA, True)

    def setup(self, setup_sample, N, rng, counter):
        self.N = N
        if self.spec.tau_mode == "geometry" or self.spec.mask_mode == "auto":
            self.geo = estimate_geometry(setup_sample, rng)
        else:
            self.geo = None

    def decide(self, step, steps, obs):
        if self.spec.tau_mode == "geometry":
            assert self.geo is not None
            tau = float(np.sqrt(self.geo["sigma_hat"] * self.geo["L_hat"]))
        else:
            tau = BASE_TAU
        if self.spec.eta_mode == "scaled":
            eta = 0.15 * tau
        else:
            eta = BASE_ETA
        if self.spec.mask_mode == "auto":
            assert self.geo is not None
            mask = self.N >= 8 * self.geo["K_hat"]
        else:
            mask = True
        self.last = StepDecision(tau, eta, mask)
        return self.last

    def audit(self) -> dict:
        geo = self.geo or {}
        return {
            "K_hat": geo.get("K_hat"),
            "L_hat": geo.get("L_hat"),
            "sigma_hat": geo.get("sigma_hat"),
            "silhouette": geo.get("sil"),
            "tau": self.last.tau,
            "eta": self.last.eta,
            "mask": self.last.mask,
        }


# ---------------------------------------------------------------------------
# Trial runner and complete capture.
# ---------------------------------------------------------------------------


def _safe_key(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def run_trial(target: TargetSpec, target_index: int, init_kind: str,
              spec: FactorSpec, seed: int, profile: Profile,
              counter: WorkCounter) -> tuple[dict, dict[str, np.ndarray]]:
    trial_seed = 80_000 + seed
    init_rng = np.random.default_rng(MASTER * 17 + trial_seed * 1000 + 1)
    data_rng = np.random.default_rng(MASTER * 17 + trial_seed * 1000 + 2)
    policy_rng = np.random.default_rng(MASTER * 17 + trial_seed * 1000 + 3)
    q0 = init_cloud(init_kind, target, profile.N, init_rng)
    ref_final = target.sample(
        profile.ref_final,
        np.random.default_rng(MASTER * 17 + 900_000 + target_index),
    )
    ref_cross = target.sample(
        profile.ref_cross,
        np.random.default_rng(MASTER * 17 + 910_000 + target_index),
    )
    policy = FactorPolicy(spec)
    wall0 = time.perf_counter()
    result = train(
        q0, target, policy, profile.steps, profile.batch, data_rng, policy_rng,
        ref_final, ref_cross, 0.05 * target.scale, counter,
        estimator="paper", track_every=max(1, profile.steps // 40),
    )
    audit = policy.audit()
    residual = support_residual(
        result.q, target, audit["tau"], audit["mask"],
        np.random.default_rng(MASTER * 17 + 920_000 + target_index + seed),
        n_ref=512,
    )
    total_wall = time.perf_counter() - wall0
    row = {
        "arm": spec.label,
        "target": target.name,
        "family": target.family,
        "init": init_kind,
        "cell": f"{target.name}/{init_kind}",
        "seed": seed,
        "ed2": result.final_ed2,
        "sw1": result.final_sw1,
        "coverage": result.coverage,
        "mass_error": result.mass_error,
        "residual": residual,
        "event_time": result.event_time,
        "censored": int(result.censored),
        "kernel_pairs": result.kernel_pairs,
        "wall_seconds": total_wall,
        "diverged": int(result.diverged),
        **audit,
    }
    key = _safe_key(f"{spec.label}__{target.name}__{init_kind}__s{seed}")
    arrays = {
        f"trajectory__{key}": result.trajectory,
        f"final_q__{key}": result.q,
    }
    return row, arrays


CSV_FIELDS = [
    "arm", "target", "family", "init", "cell", "seed", "ed2", "sw1",
    "coverage", "mass_error", "residual", "event_time", "censored",
    "kernel_pairs", "wall_seconds", "diverged", "K_hat", "L_hat",
    "sigma_hat", "silhouette", "tau", "eta", "mask",
]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def geo_mean(values) -> float:
    x = np.maximum(np.asarray(values, dtype=float), 1e-12)
    return float(np.exp(np.mean(np.log(x))))


def validation_aggregates(rows: list[dict]) -> dict[str, float]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    for row in rows:
        grouped[row["arm"]][row["cell"]].append(float(row["ed2"]))
    return {
        arm: geo_mean([np.median(values) for values in cells.values()])
        for arm, cells in grouped.items()
    }


def _factor_from_dict(obj: dict, label: str | None = None) -> FactorSpec:
    return FactorSpec(
        label or obj["label"], obj["tau_mode"], obj["mask_mode"],
        obj["eta_mode"],
    )


# ---------------------------------------------------------------------------
# Target-aware statistics.
# ---------------------------------------------------------------------------


def _log_ratio(row_by_arm: dict[str, dict], arm: str) -> float:
    mod = max(float(row_by_arm[arm]["ed2"]), 1e-12)
    base = max(float(row_by_arm["base"]["ed2"]), 1e-12)
    return float(np.log(mod / base))


def paired_arm_rows(rows: list[dict], arm: str) -> dict[str, list[float]]:
    paired: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["arm"] in ("base", arm):
            paired[(row["cell"], int(row["seed"]))][row["arm"]] = row
    by_cell: dict[str, list[float]] = defaultdict(list)
    for (cell, _), arms in paired.items():
        if "base" in arms and arm in arms:
            by_cell[cell].append(_log_ratio(arms, arm))
    return by_cell


def bootstrap_stats(rows: list[dict], arm: str, n_boot: int = 10000,
                    seed: int = MASTER) -> dict:
    by_cell = paired_arm_rows(rows, arm)
    cells = sorted(by_cell)
    all_logs = np.asarray([x for cell in cells for x in by_cell[cell]])
    point = float(np.exp(np.median(all_logs)))
    cell_ratios = {
        cell: float(np.exp(np.median(by_cell[cell]))) for cell in cells
    }
    rng = np.random.default_rng(seed)
    hier = np.empty(n_boot)
    cell_only = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(cells, size=len(cells), replace=True)
        cell_only[b] = np.exp(np.median([np.median(by_cell[c])
                                         for c in chosen]))
        sampled: list[float] = []
        for cell in chosen:
            values = by_cell[cell]
            idx = rng.integers(0, len(values), size=len(values))
            sampled.extend(values[i] for i in idx)
        hier[b] = np.exp(np.median(sampled))
    # Fixed-suite diagnostic: resample paired seeds within the fixed cells.
    fixed = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(all_logs), size=len(all_logs))
        fixed[b] = np.exp(np.median(all_logs[idx]))
    return {
        "point_ratio": point,
        "hierarchical_ci": [float(x) for x in np.quantile(hier, [.025, .975])],
        "cell_only_ci": [float(x) for x in np.quantile(cell_only, [.025, .975])],
        "fixed_suite_row_ci": [float(x) for x in np.quantile(fixed, [.025, .975])],
        "cell_ratios": cell_ratios,
    }


def subset_rows(rows: list[dict], predicate) -> list[dict]:
    cells = {row["cell"] for row in rows if predicate(row)}
    return [row for row in rows if row["cell"] in cells]


def km_for(rows: list[dict], arm: str, init_kind: str) -> int | None:
    selected = [row for row in rows
                if row["arm"] == arm and row["init"] == init_kind]
    return km_median(
        [int(row["event_time"]) for row in selected],
        [bool(int(row["censored"])) for row in selected],
    )


# ---------------------------------------------------------------------------
# Stages.
# ---------------------------------------------------------------------------


def run_validation(profile: Profile) -> Path:
    specs = factorial_specs()
    targets = attribution_validation_targets()
    config = {
        "stage": "fresh-factorial-validation",
        "profile": profile.__dict__,
        "targets": [target.name for target in targets],
        "arms": [spec.as_dict() for spec in specs],
        "selection": "minimum geometric mean of cell-median ED2",
    }
    run = Run(
        "D3b-factor-validation", config,
        source_files=[Path(__file__), PROTOCOL],
    )
    run.log("D3b validation: fresh bandwidth x mask x step factorial")
    field_invariants(run.log)
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for target_index, target in enumerate(targets):
        for init_kind in ("missing", "covered"):
            for spec in specs:
                for seed in range(profile.validation_seeds):
                    row, captured = run_trial(
                        target, target_index, init_kind, spec, seed, profile,
                        run.counter,
                    )
                    rows.append(row)
                    arrays.update(captured)
    aggregates = validation_aggregates(rows)
    winner_label = min(aggregates, key=aggregates.get)
    winner = next(spec for spec in specs if spec.label == winner_label)
    base_label = next(spec.label for spec in specs
                      if spec == BASE_SPEC or
                      (spec.tau_mode, spec.mask_mode, spec.eta_mode) ==
                      ("fixed", "on", "fixed"))
    # Normalize the baseline label used by the test stage.
    base_score = aggregates[base_label]
    frozen = {
        "profile": profile.name,
        "winner": winner.as_dict(),
        "validation_aggregate": aggregates[winner_label],
        "baseline_factorial_label": base_label,
        "baseline_aggregate": base_score,
        "winner_to_base_ratio": aggregates[winner_label] / base_score,
        "all_aggregates": aggregates,
    }
    write_rows(run.dir / "d3b_validation_rows.csv", rows)
    run.save_npz("d3b_validation_trajectories.npz", arrays)
    run.save_json("d3b_policy_frozen.json", frozen)
    for label, value in sorted(aggregates.items(), key=lambda item: item[1]):
        run.log(f"  {label:43s} ED2={value:.6f}")
    run.log(f"  FROZEN winner={winner_label} ratio-to-base="
            f"{frozen['winner_to_base_ratio']:.3f}")
    run.finish()
    return run.dir


def load_frozen(profile_name: str) -> dict:
    candidates = sorted(RUNROOT.glob("*D3b-factor-validation*/d3b_policy_frozen.json"),
                        reverse=True)
    for candidate in candidates:
        obj = json.loads(candidate.read_text(encoding="utf-8"))
        if obj.get("profile") == profile_name:
            return obj
    raise FileNotFoundError(f"no D3b validation policy for {profile_name}")


def test_specs(frozen: dict) -> list[FactorSpec]:
    win = _factor_from_dict(frozen["winner"], "combined")
    specs = [
        BASE_SPEC,
        FactorSpec("tau-only", win.tau_mode, "on", "fixed"),
        FactorSpec("mask-only", "fixed", win.mask_mode, "fixed"),
        FactorSpec("step-only", "fixed", "on", win.eta_mode),
        win,
    ]
    # Retain labels even if factors coincide: exact duplicate trajectories are
    # an auditable indication that the selected factor was inactive.
    return specs


def run_test(profile: Profile) -> Path:
    frozen = load_frozen(profile.name)
    specs = test_specs(frozen)
    targets = attribution_test_targets()
    config = {
        "stage": "fresh-heldout-attribution",
        "profile": profile.__dict__,
        "targets": [target.name for target in targets],
        "frozen_validation_policy": frozen,
        "test_arms": [spec.as_dict() for spec in specs],
        "gate": {
            "ratio_le": 0.8,
            "hierarchical_ci_hi_lt": 1.0,
            "degraded_cells_gt_1.10_le_fraction": 0.20,
            "missing_km_not_worse": True,
            "nongaussian_hierarchical_ci_hi_lt": 1.0,
        },
    }
    run = Run(
        "D3b-fresh-test", config,
        source_files=[Path(__file__), PROTOCOL],
    )
    run.log("D3b test: frozen factorial candidate on fresh held-out targets")
    field_invariants(run.log)
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    for target_index, target in enumerate(targets):
        for init_kind in ("missing", "covered"):
            for spec in specs:
                for seed in range(profile.test_seeds):
                    row, captured = run_trial(
                        target, target_index, init_kind, spec, seed, profile,
                        run.counter,
                    )
                    rows.append(row)
                    arrays.update(captured)
    write_rows(run.dir / "d3b_test_rows.csv", rows)
    run.save_npz("d3b_test_trajectories.npz", arrays)

    arm_stats = {
        spec.label: bootstrap_stats(rows, spec.label,
                                    seed=MASTER + index * 101)
        for index, spec in enumerate(specs) if spec.label != "base"
    }
    combined = arm_stats["combined"]
    nong_rows = subset_rows(rows, lambda row: row["family"] != "gauss")
    nong = bootstrap_stats(nong_rows, "combined", seed=MASTER + 999)
    degraded = [cell for cell, ratio in combined["cell_ratios"].items()
                if ratio > 1.10]
    km_base_missing = km_for(rows, "base", "missing")
    km_mod_missing = km_for(rows, "combined", "missing")
    gate = {
        "aggregate_ratio": combined["point_ratio"],
        "hierarchical_ci": combined["hierarchical_ci"],
        "crit1_ratio_le_0.8": combined["point_ratio"] <= 0.8,
        "crit2_hierarchical_ci_hi_lt_1":
            combined["hierarchical_ci"][1] < 1.0,
        "degraded_cells": degraded,
        "crit3_degraded_le_20pct":
            len(degraded) <= 0.20 * len(combined["cell_ratios"]),
        "km_base_missing": km_base_missing,
        "km_modified_missing": km_mod_missing,
        "crit4_missing_time_not_worse":
            (km_mod_missing if km_mod_missing is not None else 10**9) <=
            (km_base_missing if km_base_missing is not None else 10**9),
        "nongaussian_ratio": nong["point_ratio"],
        "nongaussian_hierarchical_ci": nong["hierarchical_ci"],
        "crit5_nongaussian_ci_hi_lt_1": nong["hierarchical_ci"][1] < 1.0,
    }
    gate["PASS"] = all(gate[key] for key in (
        "crit1_ratio_le_0.8", "crit2_hierarchical_ci_hi_lt_1",
        "crit3_degraded_le_20pct", "crit4_missing_time_not_worse",
        "crit5_nongaussian_ci_hi_lt_1",
    ))
    run.save_json("d3b_arm_statistics.json", arm_stats)
    run.save_json("d3b_gate.json", gate)
    for arm, stats in arm_stats.items():
        run.log(f"  {arm:10s}: ratio={stats['point_ratio']:.3f} "
                f"hierCI={stats['hierarchical_ci']}")
    run.log(f"  nonGaussian combined ratio={nong['point_ratio']:.3f} "
            f"hierCI={nong['hierarchical_ci']}")
    run.log(f"  missing KM base={km_base_missing} "
            f"modified={km_mod_missing}; degraded={len(degraded)}/"
            f"{len(combined['cell_ratios'])}")
    run.log(f"  GATE D3b: {'PASS' if gate['PASS'] else 'FAIL'}")
    run.finish()
    return run.dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validation", "test"])
    parser.add_argument("--profile", default="standard",
                        choices=sorted(PROFILES))
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    if args.stage == "validation":
        run_validation(profile)
    else:
        run_test(profile)


if __name__ == "__main__":
    main()
