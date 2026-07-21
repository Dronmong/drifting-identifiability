"""Frozen learned-generator transfer for the identifiability-driven field.

The protocol is frozen in ``IdentifiabilityGeneratorProtocol.md``.  The MLP
and Adam update are implemented directly in NumPy so the paper's
stop-gradient semantics are explicit: the field is a fixed target vector and
is never differentiated through.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from numpy.linalg import norm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from identifiability_drift import compute_field, invariant_tests  # noqa: E402
from lowdim_drift import WorkCounter, km_median, sliced_w1  # noqa: E402
from run_identifiability_improvement import (  # noqa: E402
    StudyTarget,
    build_target,
    diagnostic_values,
    energy_distance2,
    energy_distance2_with_reference_self,
    hierarchical_stats,
    mean_pairwise_distance,
    mode_metrics,
    particle_spread,
    safe_key,
    support_coverage,
)

MASTER = 20260723
TAU = 0.35
ETA = 0.0525
NORM_CLIP = 2.0
ADAM_LR = 0.0025
BETA1 = 0.9
BETA2 = 0.999
ADAM_EPS = 1e-8
HIDDEN = 32
INITS = ("broad", "missing", "far", "concentrated")

PROTOCOL = HERE / "IdentifiabilityGeneratorProtocol.md"
REGISTRY = HERE / "identifiability_generator_registry.json"
FIELD_SOURCE = HERE / "identifiability_drift.py"
COMMON_SOURCE = HERE / "lowdim_drift.py"
PARTICLE_RUNNER = HERE / "run_identifiability_improvement.py"
PARTICLE_GATE = (HERE / "identifiability_runs" /
                 "20260720-011000-NCJ-test-standard" / "e4_gate.json")
RUNROOT = HERE / "identifiability_runs"
REGISTRY_SHA256 = \
    "B44BAD6F8ECE1DE7D9AE9532B037FB443378484920DEC822D23D67DFB361B93F"


@dataclass(frozen=True)
class Profile:
    name: str
    steps: int
    batch: int
    seeds: int
    ref_final: int
    ref_cross: int


PROFILES = {
    "smoke": Profile("smoke", 50, 32, 1, 256, 128),
    "standard": Profile("standard", 800, 64, 12, 1024, 256),
}


@dataclass(frozen=True)
class Arm:
    label: str
    gain: str
    crossfit: bool
    mask: bool
    matched_extra: bool
    clip: float | None


ARMS = (
    Arm("paper", "paper", False, True, False, None),
    Arm("paper-matched", "paper", False, True, True, None),
    Arm("normalized-only", "constant", False, True, False, NORM_CLIP),
    Arm("crossfit-only", "paper", True, False, False, None),
    Arm("jitter-only", "paper", False, True, False, None),
    Arm("normalized-crossfit", "constant", True, False, False, NORM_CLIP),
    Arm("ncj", "constant", True, False, False, NORM_CLIP),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest().upper()


def load_targets() -> tuple[dict, list[StudyTarget]]:
    if sha256_file(REGISTRY) != REGISTRY_SHA256:
        raise RuntimeError("frozen generator registry hash mismatch")
    obj = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return obj, [build_target(item) for item in obj["targets"]]


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def seed_base(registry_seed: int, target_index: int, init_index: int,
              seed: int) -> int:
    return int(registry_seed * 1_000_003 + target_index * 100_003 +
               init_index * 10_007 + seed * 101)


def clip_vectors(V: np.ndarray, bound: float | None) -> np.ndarray:
    if bound is None:
        return V
    lengths = norm(V, axis=1)
    scale = np.minimum(1.0, bound / np.maximum(lengths, 1e-300))
    return V * scale[:, None]


class TanhMLP:
    """Two-layer tanh generator with an explicit paired Adam optimizer."""

    names = ("W1", "b1", "W2", "b2", "W3", "b3")

    def __init__(self, target: StudyTarget, init_kind: str, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.latent_dim = max(2, target.d)
        self.forward_calls = 0
        self.step_index = 0
        self.params = {
            "W1": rng.normal(size=(self.latent_dim, HIDDEN)) /
                math.sqrt(self.latent_dim),
            "b1": np.zeros(HIDDEN),
            "W2": rng.normal(size=(HIDDEN, HIDDEN)) / math.sqrt(HIDDEN),
            "b2": np.zeros(HIDDEN),
            "W3": rng.normal(size=(HIDDEN, target.d)) / math.sqrt(HIDDEN),
            "b3": np.zeros(target.d),
        }
        if init_kind == "broad":
            self.params["W3"] *= 0.8 * target.scale
        elif init_kind == "missing":
            self.params["W3"] *= 0.12 * target.scale
            if target.means is not None:
                self.params["b3"] = target.means[0].copy()
        elif init_kind == "far":
            self.params["W3"] *= 0.10 * target.scale
            direction = np.ones(target.d) / math.sqrt(target.d)
            self.params["b3"] = 3.25 * target.scale * direction
        elif init_kind == "concentrated":
            self.params["W3"] *= 0.02 * target.scale
        else:
            raise ValueError(init_kind)
        self.m = {name: np.zeros_like(value)
                  for name, value in self.params.items()}
        self.v = {name: np.zeros_like(value)
                  for name, value in self.params.items()}

    def forward(self, z: np.ndarray, want_cache: bool = False):
        self.forward_calls += 1
        p = self.params
        h1 = np.tanh(z @ p["W1"] + p["b1"])
        h2 = np.tanh(h1 @ p["W2"] + p["b2"])
        x = h2 @ p["W3"] + p["b3"]
        return (x, (z, h1, h2)) if want_cache else x

    def stopgrad_step(self, cache, field: np.ndarray) -> None:
        """Minimize ``-mean <G(z), field>`` with field held constant."""
        z, h1, h2 = cache
        p = self.params
        dx = -field / len(field)
        grads: dict[str, np.ndarray] = {}
        grads["W3"] = h2.T @ dx
        grads["b3"] = dx.sum(axis=0)
        dh2 = (dx @ p["W3"].T) * (1.0 - h2 * h2)
        grads["W2"] = h1.T @ dh2
        grads["b2"] = dh2.sum(axis=0)
        dh1 = (dh2 @ p["W2"].T) * (1.0 - h1 * h1)
        grads["W1"] = z.T @ dh1
        grads["b1"] = dh1.sum(axis=0)
        if not all(np.all(np.isfinite(g)) for g in grads.values()):
            raise FloatingPointError("non-finite generator gradient")
        self.step_index += 1
        t = self.step_index
        for name in self.names:
            g = grads[name]
            self.m[name] = BETA1 * self.m[name] + (1 - BETA1) * g
            self.v[name] = BETA2 * self.v[name] + (1 - BETA2) * g * g
            mhat = self.m[name] / (1 - BETA1 ** t)
            vhat = self.v[name] / (1 - BETA2 ** t)
            self.params[name] -= ADAM_LR * mhat / (np.sqrt(vhat) + ADAM_EPS)

    def finite(self) -> bool:
        return all(np.all(np.isfinite(x)) for x in self.params.values())


TRAJECTORY_COLUMNS = (
    "step", "ed2", "spread_ratio", "pq_p10", "pq_median",
    "delta_norm_median", "ess_pos_median", "ess_neg_median",
    "field_norm_median", "degenerate_rows",
)


def run_trial(target: StudyTarget, target_index: int, init_kind: str,
              init_index: int, arm: Arm, seed: int, profile: Profile,
              registry_seed: int, reference_self: tuple[float, float]) \
        -> tuple[dict, np.ndarray, np.ndarray]:
    wall0 = time.perf_counter()
    base = seed_base(registry_seed, target_index, init_index, seed)
    model = TanhMLP(target, init_kind, base + 1)
    query_rng = np.random.default_rng(base + 2)
    reference_rng = np.random.default_rng(base + 3)
    data_rng = np.random.default_rng(base + 4)
    ref_cross = target.sample(profile.ref_cross,
                              np.random.default_rng(base + 5))
    ref_final = target.sample(profile.ref_final,
                              np.random.default_rng(base + 7))
    ref_final_self, ref_cross_self = reference_self
    target_spread = max(particle_spread(ref_final), 1e-12)
    counter = WorkCounter()
    track_every = max(1, profile.steps // 40)
    hist: list[list[float]] = []
    event_time: int | None = None
    diverged = False
    degenerate_total = 0

    for step in range(1, profile.steps + 1):
        zq = query_rng.normal(size=(profile.batch, model.latent_dim))
        x, cache = model.forward(zq, want_cache=True)
        zr = reference_rng.normal(size=(profile.batch, model.latent_dim))
        negative = None
        if arm.crossfit:
            negative = model.forward(zr)
        elif arm.matched_extra:
            model.forward(zr)
        positive = target.sample(profile.batch, data_rng)
        tracked = step == 1 or step % track_every == 0 or \
            step == profile.steps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = compute_field(
                x, positive, negative, tau=TAU, gain=arm.gain,
                mask=arm.mask, jitter_sigma=0.0, counter=counter,
                want_diagnostics=tracked, on_degenerate="zero")
        degenerate_total += result.n_degenerate_rows
        V = clip_vectors(result.V, arm.clip)
        try:
            model.stopgrad_step(cache, V)
        except FloatingPointError:
            diverged = True
            break
        if not model.finite() or norm(x) > 1e6 * max(target.scale, 1.0):
            diverged = True
            break
        ed = max(0.0, energy_distance2_with_reference_self(
            x, ref_cross, ref_cross_self))
        if event_time is None and ed <= 0.05 * target.scale:
            event_time = step
        if tracked:
            diag = diagnostic_values(result.diagnostics)
            hist.append([
                float(step), ed, particle_spread(x) / target_spread,
                diag["pq_p10"], diag["pq_median"],
                diag["delta_norm_median"], diag["ess_pos_median"],
                diag["ess_neg_median"], diag["field_norm_median"],
                float(result.n_degenerate_rows),
            ])

    if diverged:
        final_q = np.full((profile.ref_final, target.d), np.nan)
        final_ed = final_sw = residual = float("inf")
        coverage = mass_error = support = spread = float("nan")
        fdiag = {key: float("nan") for key in (
            "pq_p10", "pq_median", "delta_norm_median",
            "ess_pos_median", "ess_neg_median", "field_norm_median",
            "self_leverage_mean")}
    else:
        zfinal = np.random.default_rng(base + 6).normal(
            size=(profile.ref_final, model.latent_dim))
        final_q = model.forward(zfinal)
        final_ed = max(0.0, energy_distance2_with_reference_self(
            final_q, ref_final, ref_final_self))
        final_sw = sliced_w1(final_q, ref_final, 32,
                             np.random.default_rng(base + 8))
        coverage, mass_error = mode_metrics(final_q, target)
        support = support_coverage(final_q, ref_final, target)
        spread = particle_spread(final_q)
        rq = final_q[:profile.batch]
        zr = np.random.default_rng(base + 10).normal(
            size=(profile.batch, model.latent_dim))
        rneg = None
        if arm.crossfit:
            rneg = model.forward(zr)
        elif arm.matched_extra:
            model.forward(zr)
        rpos = target.sample(profile.batch, np.random.default_rng(base + 9))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rfield = compute_field(
                rq, rpos, rneg, tau=TAU, gain=arm.gain, mask=arm.mask,
                counter=counter, want_diagnostics=True,
                on_degenerate="zero")
        degenerate_total += rfield.n_degenerate_rows
        final_v = clip_vectors(rfield.V, arm.clip)
        residual = float(np.sqrt(np.mean(np.sum(final_v * final_v, axis=1))))
        fdiag = diagnostic_values(rfield.diagnostics)

    row = {
        "arm": arm.label,
        "target": target.name,
        "family": target.family,
        "dimension": target.d,
        "init": init_kind,
        "cell": f"{target.name}/{init_kind}",
        "seed": seed,
        "ed2": final_ed,
        "sw1": final_sw,
        "mode_coverage": coverage,
        "mass_error": mass_error,
        "support_coverage": support,
        "residual": residual,
        "event_time": event_time if event_time is not None else profile.steps,
        "censored": int(event_time is None),
        "diverged": int(diverged),
        "degenerate_rows": degenerate_total,
        "total_kernel_pairs": counter.kernel_pairs,
        "generator_forwards": model.forward_calls,
        "wall_seconds": time.perf_counter() - wall0,
        "particle_spread": spread,
        "spread_ratio": spread / target_spread,
        "tau": TAU,
        "adam_lr": ADAM_LR,
        "gain": arm.gain,
        "crossfit": int(arm.crossfit),
        "mask": int(arm.mask),
        "norm_clip": arm.clip,
        **fdiag,
    }
    return row, np.asarray(hist, dtype=float), final_q


CSV_FIELDS = (
    "arm", "target", "family", "dimension", "init", "cell", "seed",
    "ed2", "sw1", "mode_coverage", "mass_error", "support_coverage",
    "residual", "event_time", "censored", "diverged", "degenerate_rows",
    "total_kernel_pairs", "generator_forwards", "wall_seconds",
    "particle_spread", "spread_ratio", "tau", "adam_lr", "gain",
    "crossfit", "mask", "norm_clip", "pq_p10", "pq_median",
    "delta_norm_median", "ess_pos_median", "ess_neg_median",
    "field_norm_median", "self_leverage_mean",
)


class RunArtifact:
    def __init__(self, profile: Profile, expected_rows: int) -> None:
        status = _git("status", "--porcelain")
        if profile.name == "standard" and status:
            raise RuntimeError(
                "standard generator run requires a clean Git tree; "
                f"status was:\n{status}")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = RUNROOT / f"{stamp}-NCJ-generator-{profile.name}"
        self.dir.mkdir(parents=True, exist_ok=False)
        (self.dir / "source_snapshots").mkdir()
        self.start = time.time()
        self.lines: list[str] = []
        self.expected_rows = expected_rows
        sources = (Path(__file__), PROTOCOL, REGISTRY, FIELD_SOURCE,
                   COMMON_SOURCE, PARTICLE_RUNNER, PARTICLE_GATE)
        hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in sources}
        for source in sources:
            (self.dir / "source_snapshots" / source.name).write_bytes(
                source.read_bytes())
        (self.dir / "source_hashes.json").write_text(
            json.dumps(hashes, indent=2), encoding="utf-8")
        self.manifest = {
            "stage": "learned-generator-transfer",
            "profile": asdict(profile),
            "master_seed": MASTER,
            "commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(status),
            "git_status": status.splitlines() if status else [],
            "cmdline": sys.argv,
            "registry": REGISTRY.name,
            "registry_sha256": sha256_file(REGISTRY),
            "particle_gate_sha256": sha256_file(PARTICLE_GATE),
            "source_hashes": hashes,
            "architecture": {
                "hidden": [HIDDEN, HIDDEN], "activation": "tanh",
                "latent_dim": "max(2,d)",
            },
            "optimizer": {
                "name": "Adam", "lr": ADAM_LR, "beta1": BETA1,
                "beta2": BETA2, "eps": ADAM_EPS,
            },
            "arms": [asdict(arm) for arm in ARMS],
            "initializations": list(INITS),
            "trajectory_columns": list(TRAJECTORY_COLUMNS),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "expected_rows": expected_rows,
            "start": stamp,
        }

    def log(self, text: str) -> None:
        print(text, flush=True)
        self.lines.append(text)

    def finish(self, rows: int) -> None:
        if rows != self.expected_rows:
            raise RuntimeError(f"row mismatch {rows} != {self.expected_rows}")
        self.manifest.update({
            "realized_rows": rows,
            "wall_seconds": round(time.time() - self.start, 3),
        })
        (self.dir / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8")
        (self.dir / "stdout.log").write_text(
            "\n".join(self.lines) + "\n", encoding="utf-8")


def paired_mode_coverage_diff(rows: list[dict]) -> float:
    pairs: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["arm"] in ("paper", "ncj"):
            value = float(row["mode_coverage"])
            if math.isfinite(value):
                pairs[(row["cell"], int(row["seed"]))][row["arm"]] = value
    diffs: dict[str, list[float]] = defaultdict(list)
    for (cell, _), pair in pairs.items():
        if "paper" in pair and "ncj" in pair:
            diffs[cell].append(pair["ncj"] - pair["paper"])
    if not diffs:
        return float("nan")
    return float(np.mean([np.median(values) for values in diffs.values()]))


def endpoint_max_diff(rows: list[dict], a: str, b: str) -> float:
    paired: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["arm"] in (a, b):
            paired[(row["cell"], int(row["seed"]))][row["arm"]] = \
                float(row["ed2"])
    return max(abs(pair[a] - pair[b]) for pair in paired.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(PROFILES),
                        default="standard")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    particle_gate = json.loads(PARTICLE_GATE.read_text(encoding="utf-8"))
    if not particle_gate.get("PASS", False):
        raise RuntimeError("generator transfer is blocked: particle E4 failed")
    registry, targets = load_targets()
    expected = len(targets) * len(INITS) * len(ARMS) * profile.seeds
    run = RunArtifact(profile, expected)
    run.log(f"NCJ generator: {len(targets)} targets x {len(INITS)} inits x "
            f"{len(ARMS)} arms x {profile.seeds} seeds = {expected} rows")
    invariant_tests(run.log)
    rng = np.random.default_rng(MASTER + 909)
    a, b = rng.normal(size=(17, 2)), rng.normal(size=(23, 2))
    exact = energy_distance2(a, b)
    cached = energy_distance2_with_reference_self(
        a, b, mean_pairwise_distance(b, b))
    if exact != cached:
        raise RuntimeError("cached ED2 disagreement")
    run.log("  [PASS] cached ED2 is bitwise equal to frozen ED2")
    rows: list[dict] = []
    trajectories: dict[str, np.ndarray] = {}
    finals: dict[str, np.ndarray] = {}
    registry_seed = int(registry["master_seed"])
    for ti, target in enumerate(targets):
        started = time.perf_counter()
        for ii, init_kind in enumerate(INITS):
            self_terms: dict[int, tuple[float, float]] = {}
            for seed in range(profile.seeds):
                base = seed_base(registry_seed, ti, ii, seed)
                ref_cross = target.sample(
                    profile.ref_cross, np.random.default_rng(base + 5))
                ref_final = target.sample(
                    profile.ref_final, np.random.default_rng(base + 7))
                self_terms[seed] = (
                    mean_pairwise_distance(ref_final, ref_final),
                    mean_pairwise_distance(ref_cross, ref_cross),
                )
            for arm in ARMS:
                for seed in range(profile.seeds):
                    row, trajectory, final = run_trial(
                        target, ti, init_kind, ii, arm, seed, profile,
                        registry_seed, self_terms[seed])
                    rows.append(row)
                    key = safe_key(
                        f"{arm.label}__{target.name}__{init_kind}__s{seed}")
                    trajectories[key] = trajectory
                    finals[key] = final
        run.log(f"  completed {target.name} in "
                f"{time.perf_counter() - started:.1f}s")
    with (run.dir / "rows.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(run.dir / "trajectories.npz", **trajectories)
    np.savez_compressed(run.dir / "final_particles.npz", **finals)

    stats = {
        arm.label: hierarchical_stats(rows, arm.label,
                                      seed=MASTER + i * 101)
        for i, arm in enumerate(ARMS) if arm.label != "paper"
    }
    primary = stats["ncj"]
    # Re-express the matched comparison by temporarily relabeling the matched
    # baseline as the baseline expected by the shared bootstrap helper.
    matched_rows = [dict(row, arm="paper") if row["arm"] == "paper-matched"
                    else row for row in rows if row["arm"] in
                    ("paper-matched", "ncj")]
    matched = hierarchical_stats(
        matched_rows, "ncj", seed=MASTER + 9002)
    mixtures = {"gauss_mixture", "grid_mixture"}
    mix = hierarchical_stats(
        rows, "ncj", lambda row: row["family"] in mixtures,
        seed=MASTER + 9101)
    nong = hierarchical_stats(
        rows, "ncj", lambda row: row["family"] not in
        {"gaussian", "gauss_mixture", "grid_mixture"},
        seed=MASTER + 9102)
    win_fraction = float(np.mean(
        np.asarray(list(primary["cell_ratios"].values())) < 1.0))
    coverage_diff = paired_mode_coverage_diff(rows)
    paper_div = float(np.mean([
        row["diverged"] for row in rows if row["arm"] == "paper"]))
    ncj_div = float(np.mean([
        row["diverged"] for row in rows if row["arm"] == "ncj"]))
    pair_counts = {int(row["total_kernel_pairs"]) for row in rows}
    ncf_forwards = {int(row["generator_forwards"]) for row in rows
                    if row["arm"] == "ncj"}
    matched_forwards = {int(row["generator_forwards"]) for row in rows
                        if row["arm"] == "paper-matched"}
    regression = {
        "paper_vs_matched_max_ed2_diff": endpoint_max_diff(
            rows, "paper", "paper-matched"),
        "paper_vs_jitter_max_ed2_diff": endpoint_max_diff(
            rows, "paper", "jitter-only"),
        "ncj_vs_normalized_crossfit_max_ed2_diff": endpoint_max_diff(
            rows, "ncj", "normalized-crossfit"),
    }
    gate = {
        "ratio_vs_paper": primary["point_ratio"],
        "ci_vs_paper": primary["hierarchical_ci"],
        "ratio_vs_paper_matched": matched["point_ratio"],
        "ci_vs_paper_matched": matched["hierarchical_ci"],
        "crit1_ratio_le_0.80": primary["point_ratio"] <= 0.80,
        "crit2_ci_vs_paper_hi_lt_1": primary["hierarchical_ci"][1] < 1.0,
        "crit3_ci_vs_matched_hi_lt_1": matched["hierarchical_ci"][1] < 1.0,
        "winning_cells_fraction": win_fraction,
        "crit4_winning_cells_ge_0.60": win_fraction >= 0.60,
        "gaussian_mixture": mix,
        "non_gaussian": nong,
        "crit5_subgroup_cis_hi_lt_1":
            mix["hierarchical_ci"][1] < 1.0 and
            nong["hierarchical_ci"][1] < 1.0,
        "paired_mode_coverage_diff": coverage_diff,
        "paper_divergence_rate": paper_div,
        "ncj_divergence_rate": ncj_div,
        "crit6_coverage_and_divergence":
            coverage_diff >= -0.05 and ncj_div <= paper_div + 0.02,
        "kernel_pair_counts": sorted(pair_counts),
        "ncj_generator_forward_counts": sorted(ncf_forwards),
        "paper_matched_generator_forward_counts": sorted(matched_forwards),
        "crit7_equal_compute":
            len(pair_counts) == 1 and ncf_forwards == matched_forwards,
        "regression": regression,
        "crit8_zero_sigma_regressions":
            max(regression.values()) <= 1e-12,
    }
    criteria = [key for key in gate if key.startswith("crit")]
    gate["PASS"] = all(bool(gate[key]) for key in criteria)
    summary = {"stage": "learned-generator-transfer",
               "arm_statistics": stats, "gate": gate}
    (run.dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (run.dir / "e5_gate.json").write_text(
        json.dumps(gate, indent=2), encoding="utf-8")
    for arm, result in stats.items():
        run.log(f"  {arm:22s}: ratio={result['point_ratio']:.4f} "
                f"hierCI={result['hierarchical_ci']}")
    run.log(f"  NCF vs matched CI={matched['hierarchical_ci']}; "
            f"mixture CI={mix['hierarchical_ci']}; "
            f"nonGaussian CI={nong['hierarchical_ci']}")
    run.log(f"  winning cells={win_fraction:.3f}; "
            f"mode coverage diff={coverage_diff:.4f}")
    run.log(f"E5 GATE: {'PASS' if gate['PASS'] else 'FAIL'}")
    run.finish(len(rows))
    print(f"generator artifacts: {run.dir}")


if __name__ == "__main__":
    main()
