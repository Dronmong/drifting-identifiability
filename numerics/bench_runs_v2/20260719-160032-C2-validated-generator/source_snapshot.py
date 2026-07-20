"""Phase C validation pass: audit-corrected dynamics benchmarks.

This preserves ``driftbench.py`` as the historical first pass.  The validation
pass removes the bandwidth/step-size confound, differentiates the full coupled
particle field, runs the paper bi-softmax estimator in arbitrary finite
dimension, and retains per-seed trajectories and provenance.

Usage:
    uv run --with numpy --with scipy python numerics/driftbench_v2.py \
        [C1|C2|C3|all] [--profile smoke|standard|full]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import scipy
from numpy.linalg import eigvals, norm
from scipy.cluster.vq import kmeans2


MASTER = 20260719
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNROOT = HERE / "bench_runs_v2"


@dataclass(frozen=True)
class Profile:
    name: str
    c1_seeds: int
    c1_tune_seeds: int
    c1_steps: int
    c2_seeds: int
    c2_steps: int
    c3_seeds: int
    c3_steps: int
    c3_long_steps: int
    ref_n: int


PROFILES = {
    "smoke": Profile("smoke", 2, 1, 80, 2, 100, 2, 100, 200, 256),
    "standard": Profile("standard", 8, 3, 400, 8, 400, 6, 400, 1600, 1024),
    "full": Profile("full", 24, 8, 1000, 24, 1000, 20, 1000, 4000, 4096),
}


class WorkCounter:
    """Count kernel matrix entries, the dominant cost in these experiments."""

    def __init__(self) -> None:
        self.kernel_pairs = 0
        self.field_calls = 0

    def add_field(self, n_probe: int, n_pos: int, n_neg: int) -> None:
        self.field_calls += 1
        self.kernel_pairs += n_probe * (n_pos + n_neg)


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Run:
    def __init__(self, which: str, config: dict) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = RUNROOT / f"{stamp}-{which}"
        self.dir.mkdir(parents=True, exist_ok=False)
        self.start = time.time()
        self.which = which
        self.config = config
        self.lines: list[str] = []
        self.counter = WorkCounter()
        status = _git("status", "--porcelain")
        diff = _git("diff", "--binary", "HEAD")
        self.manifest = {
            "which": which,
            "seed": MASTER,
            "commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(status),
            "git_status": status.splitlines(),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "script_sha256": _sha256(Path(__file__)),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "cmdline": sys.argv,
            "config": config,
            "start": stamp,
        }

    def log(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def save_csv(self, name: str, header: Iterable[str], rows: Iterable[Iterable]) -> None:
        with (self.dir / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def save_npz(self, name: str, arrays: dict[str, np.ndarray]) -> None:
        np.savez_compressed(self.dir / name, **arrays)

    def finish(self) -> None:
        self.manifest.update(
            wall_seconds=round(time.time() - self.start, 3),
            kernel_pairs=self.counter.kernel_pairs,
            field_calls=self.counter.field_calls,
        )
        with (self.dir / "manifest.json").open("w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)
        with (self.dir / "summary.md").open("w", encoding="utf-8") as f:
            f.write("```text\n" + "\n".join(self.lines) + "\n```\n")
        shutil.copy2(Path(__file__), self.dir / "source_snapshot.py")


@dataclass
class Target:
    means: np.ndarray
    sigma: float
    weights: np.ndarray
    L: float

    @property
    def K(self) -> int:
        return len(self.means)

    @property
    def d(self) -> int:
        return self.means.shape[1]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        comp = rng.choice(self.K, size=n, p=self.weights)
        return self.means[comp] + rng.normal(size=(n, self.d)) * self.sigma


def _regular_simplex(K: int, d: int, L: float) -> np.ndarray:
    gram = np.eye(K) - np.ones((K, K)) / K
    val, vec = np.linalg.eigh(gram)
    coords = vec[:, val > 1e-10] * np.sqrt(val[val > 1e-10])
    coords *= L / np.sqrt(2.0)
    return np.pad(coords, ((0, 0), (0, d - coords.shape[1])))


def controlled_means(K: int, d: int, L: float) -> np.ndarray:
    """Means with *minimum* pair separation L, rather than a nominal radius."""
    if d == 1:
        return ((np.arange(K) - (K - 1) / 2) * L)[:, None]
    if d >= K - 1:
        return _regular_simplex(K, d, L)
    radius = L / (2 * np.sin(np.pi / K))
    theta = 2 * np.pi * np.arange(K) / K
    out = np.zeros((K, d))
    out[:, 0] = radius * np.cos(theta)
    out[:, 1] = radius * np.sin(theta)
    return out


def make_target(K: int, d: int, L: float, unequal: bool = False) -> Target:
    means = controlled_means(K, d, L)
    if unequal:
        raw = np.arange(1, K + 1, dtype=float)
        weights = raw / raw.sum()
    else:
        weights = np.full(K, 1 / K)
    return Target(means, 0.15 * L, weights, L)


def actual_min_separation(means: np.ndarray) -> float:
    D = norm(means[:, None, :] - means[None, :, :], axis=2)
    D[D == 0] = np.inf
    return float(D.min())


def estimate_geometry(
    data: np.ndarray, K: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Estimate nearest-mode separation and isotropic within-mode width."""
    centers, labels = kmeans2(data, K, iter=60, minit="++", seed=rng)
    D = norm(centers[:, None, :] - centers[None, :, :], axis=2)
    D[D == 0] = np.inf
    nearest = D.min(axis=1)
    Lhat = float(np.median(nearest))
    resid = data - centers[labels]
    sigma_hat = float(np.sqrt(np.mean(resid**2)))
    return max(Lhat, 1e-8), max(sigma_hat, 1e-8)


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


def mean_shift(X: np.ndarray, pts: np.ndarray, tau: float) -> np.ndarray:
    D = X[:, None, :] - pts[None, :, :]
    r = norm(D, axis=2)
    K = np.exp(-r / tau)
    Z = K.sum(axis=1, keepdims=True)
    Z = np.where(Z < 1e-300, np.inf, Z)
    return (K[:, :, None] * (-D)).sum(axis=1) / Z


def drift_snis(
    q: np.ndarray,
    data: np.ndarray,
    tau: float,
    mask: bool,
    counter: WorkCounter | None = None,
) -> np.ndarray:
    if counter is not None:
        counter.add_field(len(q), len(data), len(q))
    mp = mean_shift(q, data, tau)
    D = q[:, None, :] - q[None, :, :]
    K = np.exp(-norm(D, axis=2) / tau)
    if mask:
        np.fill_diagonal(K, 0.0)
    Z = K.sum(axis=1, keepdims=True)
    Z = np.where(Z < 1e-300, np.inf, Z)
    mq = (K[:, :, None] * (-D)).sum(axis=1) / Z
    return mp - mq


def drift_paper(
    q: np.ndarray,
    data: np.ndarray,
    tau: float,
    mask: bool,
    counter: WorkCounter | None = None,
) -> np.ndarray:
    """Dimension-general Algorithm-2 bi-softmax drift with q reused as y_neg."""
    if counter is not None:
        counter.add_field(len(q), len(data), len(q))
    dp = norm(q[:, None, :] - data[None, :, :], axis=2)
    dn = norm(q[:, None, :] - q[None, :, :], axis=2)
    logits = np.concatenate([-dp / tau, -dn / tau], axis=1)
    if mask:
        npos = len(data)
        ii = np.arange(len(q))
        logits[ii, npos + ii] -= 1e6 / tau
    row = np.exp(logits - logits.max(axis=1, keepdims=True))
    row /= row.sum(axis=1, keepdims=True)
    col = np.exp(logits - logits.max(axis=0, keepdims=True))
    col /= col.sum(axis=0, keepdims=True)
    A = np.sqrt(row * col)
    Ap, An = A[:, : len(data)], A[:, len(data) :]
    Wp = Ap * An.sum(axis=1, keepdims=True)
    Wn = An * Ap.sum(axis=1, keepdims=True)
    return Wp @ data - Wn @ q


FIELD = {"snis": drift_snis, "paper": drift_paper}


def mixed_field(
    estimator: str,
    q: np.ndarray,
    data: np.ndarray,
    taus: list[float],
    weights: list[float],
    mask: bool,
    counter: WorkCounter | None,
) -> np.ndarray:
    out = np.zeros_like(q)
    sw = float(sum(weights))
    for tau, weight in zip(taus, weights):
        out += weight * FIELD[estimator](q, data, tau, mask, counter)
    return out / sw


def field_invariants(run: Run) -> None:
    rng = np.random.default_rng(MASTER)
    q = rng.normal(size=(7, 2))
    y = rng.normal(size=(11, 2))
    t = rng.normal(size=2)
    for estimator in ("snis", "paper"):
        f = FIELD[estimator]
        v = f(q, y, 0.8, False)
        vt = f(q + t, y + t, 0.8, False)
        assert norm(v - vt) < 1e-11
        assert np.isfinite(v).all()
        run.log(f"  invariant {estimator}: translation/finite PASS")
    # Cross-check the ND transcription against the independent 1-D port.
    sys.path.insert(0, str(HERE))
    import driftlab  # pylint: disable=import-outside-toplevel

    x = rng.normal(size=6)
    yp = rng.normal(size=9)
    yn = x.copy()
    for mask in (False, True):
        got = drift_paper(x[:, None], yp[:, None], 0.7, mask)[:, 0]
        want = driftlab.compute_v_paper(x, yp, yn, 0.7, mask)
        assert norm(got - want) < 1e-11
    run.log("  invariant paper-ND vs driftlab.compute_v_paper: PASS")


# ---------------------------------------------------------------------------
# Metrics and training
# ---------------------------------------------------------------------------


def energy_distance2(A: np.ndarray, B: np.ndarray) -> float:
    def cross(P: np.ndarray, Q: np.ndarray) -> float:
        total = 0.0
        for i in range(0, len(P), 256):
            total += norm(P[i : i + 256, None, :] - Q[None, :, :], axis=2).sum()
        return float(total / (len(P) * len(Q)))

    return 2 * cross(A, B) - cross(A, A) - cross(B, B)


def mode_metrics(q: np.ndarray, target: Target) -> tuple[float, float]:
    labels = norm(q[:, None, :] - target.means[None, :, :], axis=2).argmin(axis=1)
    mass = np.bincount(labels, minlength=target.K) / len(q)
    coverage = float((mass >= 0.5 * target.weights).mean())
    return coverage, float(np.abs(mass - target.weights).sum())


@dataclass
class TrainResult:
    q: np.ndarray
    final_ed: float
    coverage: float
    mass_error: float
    event_time: int
    censored: bool
    trajectory: np.ndarray
    kernel_pairs: int
    wall_seconds: float


TauPolicy = Callable[[int, int], tuple[list[float], list[float], float]]


def train(
    q0: np.ndarray,
    target: Target,
    policy: TauPolicy,
    estimator: str,
    mask: bool,
    steps: int,
    batch: int,
    rng: np.random.Generator,
    ref_final: np.ndarray,
    ref_cross: np.ndarray,
    ed_tol: float,
    counter: WorkCounter,
    track_every: int = 10,
    fixed_data: np.ndarray | None = None,
) -> TrainResult:
    q = q0.copy()
    hist: list[tuple[float, ...]] = []
    event_time: int | None = None
    start_pairs = counter.kernel_pairs
    start = time.time()
    for step in range(1, steps + 1):
        data = fixed_data if fixed_data is not None else target.sample(batch, rng)
        taus, weights, eta = policy(step - 1, steps)
        V = mixed_field(estimator, q, data, taus, weights, mask, counter)
        q = q + eta * V
        # Check the event every step; retain a thinner trajectory for storage.
        ed_cross = energy_distance2(q, ref_cross)
        if event_time is None and ed_cross < ed_tol:
            event_time = step
        if step == 1 or step % track_every == 0 or step == steps:
            cov, me = mode_metrics(q, target)
            hist.append((step, ed_cross, cov, me, taus[0], eta))
    final_ed = energy_distance2(q, ref_final)
    cov, me = mode_metrics(q, target)
    return TrainResult(
        q=q,
        final_ed=final_ed,
        coverage=cov,
        mass_error=me,
        event_time=event_time if event_time is not None else steps,
        censored=event_time is None,
        trajectory=np.asarray(hist),
        kernel_pairs=counter.kernel_pairs - start_pairs,
        wall_seconds=time.time() - start,
    )


def constant_policy(taus: list[float], weights: list[float], eta: float) -> TauPolicy:
    return lambda _t, _n: (taus, weights, eta)


def anneal_policy(L: float, fine: float, eta_mode: str, eta_common: float) -> TauPolicy:
    def policy(t: int, steps: int) -> tuple[list[float], list[float], float]:
        frac = min(1.0, t / max(1.0, 0.7 * steps))
        tau = L * (fine / L) ** frac
        eta = 0.1 * tau if eta_mode == "joint" else eta_common
        return [tau], [1.0], eta

    return policy


def iqr(values: list[float]) -> tuple[float, float, float]:
    q = np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75])
    return float(q[0]), float(q[1]), float(q[2])


def km_median(times: list[int], censored: list[bool]) -> int | None:
    """Kaplan-Meier median for right-censored integer event times."""
    records = sorted(zip(times, censored))
    at_risk = len(records)
    survival = 1.0
    for t in sorted(set(times)):
        at_t = [(tt, c) for tt, c in records if tt == t]
        events = sum(not c for _, c in at_t)
        removed = len(at_t)
        if at_risk and events:
            survival *= 1.0 - events / at_risk
            if survival <= 0.5:
                return t
        at_risk -= removed
    return None


def key_name(*parts) -> str:
    text = "_".join(str(p) for p in parts)
    return "".join(c if c.isalnum() else "_" for c in text)


# ---------------------------------------------------------------------------
# C1: bandwidth and step-size factorial
# ---------------------------------------------------------------------------


def C1(profile: Profile) -> Path:
    config = {
        "profile": asdict(profile),
        "targets": [[4, 1], [4, 2], [8, 2], [4, 5]],
        "L_normalized": 1.0,
        "batch": 64,
        "particles_per_mode": 8,
        "paper_taus": [0.02, 0.05, 0.2],
        # Small enough to be in the flow-like regime even for the narrowest
        # normalized paper bandwidth 0.02.  The separate joint-policy arms
        # deliberately use eta_t = 0.1 tau_t.
        "common_eta": 0.01,
        "initialization": "central Gaussian, width 0.25 L (missing-mode stress test)",
        "equal_scale_budget": True,
    }
    run = Run("C1-validated-bandwidth", config)
    run.log("C1 validation: bandwidth-only factorial + joint schedule")
    field_invariants(run)
    rows = []
    trajectories: dict[str, np.ndarray] = {}
    L = 1.0
    for K, d in config["targets"]:
        target = make_target(K, d, L)
        fine = 0.5 * target.sigma
        common_eta = config["common_eta"] * L
        geom_data = target.sample(4096, np.random.default_rng(MASTER + 91 + K + d))
        Lhat, sighat = estimate_geometry(
            geom_data, K, np.random.default_rng(MASTER + 92 + K + d)
        )
        finehat = 0.5 * sighat
        run.log(
            f"  target K={K} d={d}: minSep={actual_min_separation(target.means):.3f} "
            f"oracle(L,sigma)=({L:.3f},{target.sigma:.3f}) "
            f"estimated=({Lhat:.3f},{sighat:.3f})"
        )
        N = 8 * K
        ref_final = target.sample(profile.ref_n, np.random.default_rng(MASTER + 100 + K + d))
        ref_cross = target.sample(min(256, profile.ref_n), np.random.default_rng(MASTER + 101 + K + d))
        ed_tol = 0.02 * L

        # Held-out, broader oracle grid under the same common eta and horizon.
        grid = np.geomspace(0.03 * L, 1.5 * L, 9)
        tune_scores = []
        tune_counter = WorkCounter()
        for gi, tau in enumerate(grid):
            vals = []
            for seed in range(profile.c1_tune_seeds):
                rng = np.random.default_rng(MASTER + 10000 + 100 * seed + K + d)
                q0 = (
                    np.average(target.means, axis=0, weights=target.weights)
                    + rng.normal(size=(N, d)) * 0.25 * L
                )
                result = train(
                    q0,
                    target,
                    constant_policy([float(tau)], [1.0], common_eta),
                    "snis",
                    False,
                    profile.c1_steps,
                    64,
                    rng,
                    ref_final,
                    ref_cross,
                    ed_tol,
                    tune_counter,
                    track_every=max(1, profile.c1_steps // 20),
                )
                vals.append(result.final_ed)
            tune_scores.append(float(np.median(vals)))
        best_tau = float(grid[int(np.argmin(tune_scores))])
        run.log(
            f"    held-out oracle grid: best tau={best_tau:.4f}; "
            f"selection cost={tune_counter.kernel_pairs:,} kernel pairs"
        )
        run.counter.kernel_pairs += tune_counter.kernel_pairs
        run.counter.field_calls += tune_counter.field_calls

        arm_defs: dict[str, tuple[TauPolicy, int, str]] = {
            "fine-common": (
                constant_policy([fine], [1.0], common_eta),
                profile.c1_steps,
                "bandwidth-only",
            ),
            "coarse-common": (
                constant_policy([L], [1.0], common_eta),
                profile.c1_steps,
                "bandwidth-only",
            ),
            "average-common": (
                constant_policy([fine, L], [1.0, 1.0], common_eta),
                max(1, profile.c1_steps // 2),
                "bandwidth-only/equal-compute",
            ),
            "paper-fixed-common": (
                constant_policy([0.02, 0.05, 0.2], [1.0] * 3, common_eta),
                max(1, profile.c1_steps // 3),
                "bandwidth-only/equal-compute",
            ),
            "anneal-oracle-common": (
                anneal_policy(L, fine, "common", common_eta),
                profile.c1_steps,
                "bandwidth-only",
            ),
            "anneal-estimated-common": (
                anneal_policy(Lhat, finehat, "common", common_eta),
                profile.c1_steps,
                "bandwidth-only",
            ),
            "oracle-grid-common": (
                constant_policy([best_tau], [1.0], common_eta),
                profile.c1_steps,
                "held-out oracle",
            ),
            "fine-joint": (
                constant_policy([fine], [1.0], 0.1 * fine),
                profile.c1_steps,
                "joint policy",
            ),
            "coarse-joint": (
                constant_policy([L], [1.0], 0.1 * L),
                profile.c1_steps,
                "joint policy",
            ),
            "anneal-oracle-joint": (
                anneal_policy(L, fine, "joint", common_eta),
                profile.c1_steps,
                "joint policy",
            ),
            "anneal-estimated-joint": (
                anneal_policy(Lhat, finehat, "joint", common_eta),
                profile.c1_steps,
                "joint policy",
            ),
        }
        for arm, (policy, steps, comparison) in arm_defs.items():
            finals: list[float] = []
            masses: list[float] = []
            event_times: list[int] = []
            censored: list[bool] = []
            for seed in range(profile.c1_seeds):
                rng = np.random.default_rng(MASTER + 20000 + 100 * seed + K + d)
                q0 = (
                    np.average(target.means, axis=0, weights=target.weights)
                    + rng.normal(size=(N, d)) * 0.25 * L
                )
                result = train(
                    q0,
                    target,
                    policy,
                    "snis",
                    False,
                    steps,
                    64,
                    rng,
                    ref_final,
                    ref_cross,
                    ed_tol,
                    run.counter,
                    track_every=max(1, steps // 40),
                )
                finals.append(result.final_ed)
                masses.append(result.mass_error)
                event_times.append(result.event_time)
                censored.append(result.censored)
                rows.append(
                    (
                        K,
                        d,
                        arm,
                        comparison,
                        seed,
                        steps,
                        result.final_ed,
                        result.coverage,
                        result.mass_error,
                        result.event_time,
                        int(result.censored),
                        result.kernel_pairs,
                        result.wall_seconds,
                        Lhat,
                        sighat,
                        best_tau,
                        tune_counter.kernel_pairs,
                    )
                )
                trajectories[key_name(K, d, arm, seed)] = result.trajectory
            q1, med, q3 = iqr(finals)
            mkm = km_median(event_times, censored)
            run.log(
                f"    K={K} d={d} {arm:24s}: ED {med:.4f} "
                f"[{q1:.4f},{q3:.4f}] massErr={np.median(masses):.3f} "
                f"KMsteps={mkm if mkm is not None else 'NR'} "
                f"censored={sum(censored)}/{len(censored)}"
            )
    run.save_csv(
        "c1_per_seed.csv",
        [
            "K",
            "d",
            "arm",
            "comparison",
            "seed",
            "steps",
            "final_ED2",
            "coverage",
            "mass_error",
            "event_time",
            "censored",
            "kernel_pairs",
            "wall_seconds",
            "L_est",
            "sigma_est",
            "best_tau",
            "oracle_tune_kernel_pairs",
        ],
        rows,
    )
    run.save_npz("c1_trajectories.npz", trajectories)
    run.finish()
    return run.dir


# ---------------------------------------------------------------------------
# C2: full coupled generator
# ---------------------------------------------------------------------------


def full_generator(
    q: np.ndarray,
    data: np.ndarray,
    tau: float,
    estimator: str,
    mask: bool,
    counter: WorkCounter,
) -> np.ndarray:
    shape = q.shape
    base = q.ravel()
    h = 1e-5 * tau
    J = np.empty((base.size, base.size))

    def f(v: np.ndarray) -> np.ndarray:
        qq = v.reshape(shape)
        return FIELD[estimator](qq, data, tau, mask, counter).ravel()

    for i in range(base.size):
        e = np.zeros_like(base)
        e[i] = h
        J[:, i] = (f(base + e) - f(base - e)) / (2 * h)
    return J


def euler_boundary(eigs: np.ndarray, tol: float = 1e-8) -> float | None:
    if np.max(eigs.real) >= -tol:
        return None
    return float(np.min(-2 * eigs.real / np.abs(eigs) ** 2))


def C2(profile: Profile) -> Path:
    config = {
        "profile": asdict(profile),
        "targets": [[4, 1], [4, 2]],
        "tau_regimes": ["sigma", "L"],
        "particles_per_mode": 6,
        "batch": 64,
        "generator": "full central-difference Nd-by-Nd at exact empirical truth",
    }
    run = Run("C2-validated-generator", config)
    run.log("C2 validation: full coupled generator + safety-factor benchmark")
    field_invariants(run)
    spectrum_rows = []
    result_rows = []
    trajectories: dict[str, np.ndarray] = {}
    for K, d in config["targets"]:
        target = make_target(K, d, 1.0)
        N = config["particles_per_mode"] * K
        ref_final = target.sample(profile.ref_n, np.random.default_rng(MASTER + 300 + K + d))
        ref_cross = target.sample(min(256, profile.ref_n), np.random.default_rng(MASTER + 301 + K + d))
        q_truth = target.sample(N, np.random.default_rng(MASTER + 302 + K + d))
        # Reusing the same empirical law on both sides makes q_truth an exact
        # equilibrium of the unmasked empirical field.
        data_truth = q_truth.copy()
        for regime in config["tau_regimes"]:
            tau = target.sigma if regime == "sigma" else target.L
            before_pairs = run.counter.kernel_pairs
            start = time.time()
            J = full_generator(q_truth, data_truth, tau, "snis", False, run.counter)
            generator_wall = time.time() - start
            generator_pairs = run.counter.kernel_pairs - before_pairs
            ev = eigvals(J)
            eta_star = euler_boundary(ev)
            residual = float(norm(drift_snis(q_truth, data_truth, tau, False), axis=1).max())
            rho_half = (
                float(np.abs(1 + 0.5 * eta_star * ev).max())
                if eta_star is not None else np.nan
            )
            rho_boundary = (
                float(np.abs(1 + eta_star * ev).max())
                if eta_star is not None else np.nan
            )
            if eta_star is not None:
                run.log(
                    f"  K={K} d={d} tau={regime}: maxRe={ev.real.max():+.3e} "
                    f"resid={residual:.2e} eta*={eta_star / tau:.3f}tau "
                    f"rho(.5eta*)={rho_half:.6f} rho(eta*)={rho_boundary:.6f}"
                )
            else:
                run.log(
                    f"  K={K} d={d} tau={regime}: maxRe={ev.real.max():+.3e} "
                    "NO POSITIVE STABLE EULER STEP"
                )
            for idx, lam in enumerate(ev):
                spectrum_rows.append(
                    (K, d, regime, idx, lam.real, lam.imag, eta_star, residual,
                     rho_half, rho_boundary, generator_pairs, generator_wall)
                )
            arms: list[tuple[str, float]] = [
                ("fixed-0.1tau", 0.1 * tau),
                ("fixed-0.5tau", 0.5 * tau),
                ("fixed-1.0tau", 1.0 * tau),
            ]
            if eta_star is not None:
                arms.extend(
                    [
                        ("0.1-eta-star", 0.1 * eta_star),
                        ("0.25-eta-star", 0.25 * eta_star),
                        ("0.5-eta-star", 0.5 * eta_star),
                        ("eta-star-boundary", eta_star),
                    ]
                )
            for arm, eta in arms:
                vals = []
                for seed in range(profile.c2_seeds):
                    rng = np.random.default_rng(MASTER + 30000 + 100 * seed + K + d)
                    q0 = (
                        np.average(target.means, axis=0, weights=target.weights)
                        + rng.normal(size=(N, d)) * 0.25 * target.L
                    )
                    result = train(
                        q0,
                        target,
                        constant_policy([tau], [1.0], eta),
                        "snis",
                        False,
                        profile.c2_steps,
                        64,
                        rng,
                        ref_final,
                        ref_cross,
                        0.05 * target.L,
                        run.counter,
                        track_every=max(1, profile.c2_steps // 40),
                    )
                    diverged = int(
                        not np.isfinite(result.final_ed)
                        or result.final_ed > 10 * target.L
                        or not np.isfinite(result.q).all()
                    )
                    vals.append(result.final_ed if not diverged else np.inf)
                    result_rows.append(
                        (
                            K,
                            d,
                            regime,
                            arm,
                            seed,
                            eta,
                            eta / tau,
                            eta_star,
                            result.final_ed,
                            result.coverage,
                            result.mass_error,
                            result.event_time,
                            int(result.censored),
                            diverged,
                            result.kernel_pairs,
                            result.wall_seconds,
                        )
                    )
                    trajectories[key_name(K, d, regime, arm, seed)] = result.trajectory
                finite = [v for v in vals if np.isfinite(v)]
                if finite:
                    lo, med, hi = iqr(finite)
                    run.log(
                        f"    {arm:18s} eta/tau={eta/tau:.3f}: "
                        f"ED {med:.4f} [{lo:.4f},{hi:.4f}] "
                        f"diverged={len(vals)-len(finite)}/{len(vals)}"
                    )
                else:
                    run.log(f"    {arm:18s}: all runs diverged")
    run.save_csv(
        "c2_spectrum.csv",
        ["K", "d", "tau_regime", "eig_index", "real", "imag", "eta_star",
         "truth_residual", "rho_half_eta_star", "rho_eta_star",
         "generator_kernel_pairs", "generator_wall_seconds"],
        spectrum_rows,
    )
    run.save_csv(
        "c2_per_seed.csv",
        ["K", "d", "tau_regime", "arm", "seed", "eta", "eta_over_tau",
         "eta_star", "final_ED2", "coverage", "mass_error", "event_time",
         "censored", "diverged", "kernel_pairs", "wall_seconds"],
        result_rows,
    )
    run.save_npz("c2_trajectories.npz", trajectories)
    run.finish()
    return run.dir


# ---------------------------------------------------------------------------
# C3: mask, particle count, batch size, and exact paper estimator
# ---------------------------------------------------------------------------


def local_endpoint_diagnostics(
    q: np.ndarray,
    data: np.ndarray,
    tau: float,
    estimator: str,
    mask: bool,
    counter: WorkCounter,
    compute_generator: bool,
) -> tuple[float, float]:
    V = FIELD[estimator](q, data, tau, mask, counter)
    residual = float(np.sqrt(np.mean(np.sum(V * V, axis=1))))
    if not compute_generator:
        return residual, np.nan
    J = full_generator(q, data, tau, estimator, mask, counter)
    return residual, float(eigvals(J).real.max())


def perturbation_contraction(
    q: np.ndarray,
    data: np.ndarray,
    tau: float,
    estimator: str,
    mask: bool,
    rng: np.random.Generator,
    counter: WorkCounter,
    steps: int = 100,
) -> float:
    delta = rng.normal(size=q.shape)
    delta *= (1e-3 * tau) / max(norm(delta), 1e-300)
    qa = q.copy()
    qb = q + delta
    initial = norm(qb - qa)
    for _ in range(steps):
        qa += 0.05 * tau * FIELD[estimator](qa, data, tau, mask, counter)
        qb += 0.05 * tau * FIELD[estimator](qb, data, tau, mask, counter)
    return float(norm(qb - qa) / initial)


def C3(profile: Profile) -> Path:
    per_modes = [1, 2, 4, 8, 16] if profile.name != "full" else [1, 2, 4, 8, 16, 32]
    batches = [16, 64] if profile.name != "full" else [16, 64, 256]
    config = {
        "profile": asdict(profile),
        "estimators": ["snis", "paper"],
        "target_weights": ["equal", "unequal"],
        "particles_per_mode": per_modes,
        "batches": batches,
        "tau": "sigma",
        "long_horizon_trigger": "coverage<1 or mass_error>0.4",
    }
    run = Run("C3-validated-mask", config)
    run.log("C3 validation: mask x N/K x batch x SNIS/paper estimator")
    field_invariants(run)
    rows = []
    trajectories: dict[str, np.ndarray] = {}
    for unequal in (False, True):
        target_name = "unequal" if unequal else "equal"
        target = make_target(4, 2, 1.0, unequal=unequal)
        tau = target.sigma
        ref_final = target.sample(profile.ref_n, np.random.default_rng(MASTER + 500 + int(unequal)))
        ref_cross = target.sample(min(256, profile.ref_n), np.random.default_rng(MASTER + 501 + int(unequal)))
        for estimator in ("snis", "paper"):
            for per_mode in per_modes:
                N = per_mode * target.K
                for batch in batches:
                    for mask in (False, True):
                        endpoints = []
                        candidate_count = 0
                        for seed in range(profile.c3_seeds):
                            rng = np.random.default_rng(
                                MASTER + 40000 + 1000 * int(unequal)
                                + 100 * seed + 10 * per_mode + batch
                            )
                            q0 = rng.normal(size=(N, target.d)) * target.L
                            result = train(
                                q0,
                                target,
                                constant_policy([tau], [1.0], 0.1 * tau),
                                estimator,
                                mask,
                                profile.c3_steps,
                                batch,
                                rng,
                                ref_final,
                                ref_cross,
                                0.05 * target.L,
                                run.counter,
                                track_every=max(1, profile.c3_steps // 40),
                            )
                            total_steps = profile.c3_steps
                            # A long continuation distinguishes endpoint failure
                            # from a finite-horizon transient.
                            if result.coverage < 1.0 or result.mass_error > 0.4:
                                result2 = train(
                                    result.q,
                                    target,
                                    constant_policy([tau], [1.0], 0.1 * tau),
                                    estimator,
                                    mask,
                                    profile.c3_long_steps,
                                    batch,
                                    rng,
                                    ref_final,
                                    ref_cross,
                                    0.05 * target.L,
                                    run.counter,
                                    track_every=max(1, profile.c3_long_steps // 40),
                                )
                                result2.trajectory[:, 0] += profile.c3_steps
                                result2.trajectory = np.vstack(
                                    [result.trajectory, result2.trajectory]
                                )
                                result2.kernel_pairs += result.kernel_pairs
                                result2.wall_seconds += result.wall_seconds
                                result = result2
                                total_steps += profile.c3_long_steps
                            eval_data = target.sample(
                                max(512, batch), np.random.default_rng(MASTER + 60000 + seed)
                            )
                            compute_gen = per_mode <= 2
                            residual, max_real = local_endpoint_diagnostics(
                                result.q,
                                eval_data,
                                tau,
                                estimator,
                                mask,
                                run.counter,
                                compute_gen,
                            )
                            contraction = np.nan
                            wrong = result.coverage < 1.0 or result.mass_error > 0.4
                            if wrong and compute_gen:
                                contraction = perturbation_contraction(
                                    result.q,
                                    eval_data,
                                    tau,
                                    estimator,
                                    mask,
                                    np.random.default_rng(MASTER + 70000 + seed),
                                    run.counter,
                                )
                            candidate = bool(
                                wrong
                                and residual < 1e-3 * target.L
                                and np.isfinite(max_real)
                                and max_real < 0
                                and np.isfinite(contraction)
                                and contraction < 1
                            )
                            candidate_count += int(candidate)
                            endpoints.append(result.final_ed)
                            rows.append(
                                (
                                    target_name,
                                    estimator,
                                    per_mode,
                                    N,
                                    batch,
                                    mask,
                                    seed,
                                    total_steps,
                                    result.final_ed,
                                    result.coverage,
                                    result.mass_error,
                                    residual,
                                    max_real,
                                    contraction,
                                    int(candidate),
                                    result.kernel_pairs,
                                    result.wall_seconds,
                                )
                            )
                            trajectories[
                                key_name(target_name, estimator, per_mode, batch, mask, seed)
                            ] = result.trajectory
                        lo, med, hi = iqr(endpoints)
                        run.log(
                            f"  {target_name:7s} {estimator:5s} N/K={per_mode:2d} "
                            f"B={batch:3d} mask={str(mask):5s}: ED {med:.4f} "
                            f"[{lo:.4f},{hi:.4f}] stable-wrong-candidates="
                            f"{candidate_count}/{profile.c3_seeds}"
                        )
    run.save_csv(
        "c3_per_seed.csv",
        ["target_weights", "estimator", "per_mode", "N", "batch", "mask",
         "seed", "total_steps", "final_ED2", "coverage", "mass_error",
         "residual_rms", "generator_max_real", "perturbation_ratio",
         "stable_wrong_candidate", "kernel_pairs", "wall_seconds"],
        rows,
    )
    run.save_npz("c3_trajectories.npz", trajectories)
    run.finish()
    return run.dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("which", nargs="?", default="all", choices=["C1", "C2", "C3", "all"])
    parser.add_argument("--profile", default="standard", choices=sorted(PROFILES))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    profile = PROFILES[args.profile]
    functions = {"C1": C1, "C2": C2, "C3": C3}
    for name, function in functions.items():
        if args.which in ("all", name):
            function(profile)
