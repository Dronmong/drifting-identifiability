"""Reusable exact-estimator drifting dynamics for the low-dimensional
performance program (LowDimPerformanceRoadmap.md, D0-D5).

Fresh namespace: the historical Phase C artifacts (driftbench*.py,
bench_runs*/) are untouched audit trail.  This module holds:

* target families (Gaussian mixtures incl. unequal/heteroscedastic, ring,
  concentric circles, two moons, skewed heavy-tailed) with a common
  interface and an explicit oracle/diagnostic separation;
* the exact dimension-general row/column bi-softmax Algorithm-2 drift
  (cross-checked against driftlab.compute_v_paper) plus the SNIS field as a
  labeled diagnostic;
* non-oracle geometry estimation (k-means with silhouette model selection);
* the training loop with pluggable policies, matched-compute accounting,
  right-censored threshold times, and full provenance;
* metrics: squared energy distance, sliced Wasserstein-1, mode metrics
  (oracle-diagnostic only), on-support residual;
* invariant tests (translation, finiteness, matched-batch cancellation,
  1-d agreement with driftlab).
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import scipy
from numpy.linalg import norm
from scipy.cluster.vq import kmeans2

MASTER = 20260719
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNROOT = HERE / "lowdim_runs"


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class WorkCounter:
    """Kernel matrix entries: the matched-compute currency."""

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


def _sha256_file(path: Path) -> str:
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
        self.lines: list[str] = []
        self.counter = WorkCounter()
        status = _git("status", "--porcelain")
        diff = _git("diff", "--binary", "HEAD")
        self.manifest = {
            "which": which,
            "seed": MASTER,
            "commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(status),
            "git_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(),
            "module_sha256": _sha256_file(HERE / "lowdim_drift.py"),
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "cmdline": sys.argv,
            "config": config,
            "start": stamp,
        }

    def log(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def save_csv(self, name: str, header: Iterable[str],
                 rows: Iterable[Iterable]) -> None:
        with (self.dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)

    def save_json(self, name: str, obj) -> None:
        with (self.dir / name).open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)

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


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass
class TargetSpec:
    """A sampling target.  `means/weights/sigmas` are ORACLE diagnostics for
    mode metrics on mixture families; policies must never read them."""

    name: str
    d: int
    sampler: Callable[[int, np.random.Generator], np.ndarray]
    family: str
    means: np.ndarray | None = None
    weights: np.ndarray | None = None
    scale: float = 1.0                 # rough data scale, diagnostics only

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return self.sampler(n, rng)


def _regular_simplex(K: int, d: int, L: float) -> np.ndarray:
    gram = np.eye(K) - np.ones((K, K)) / K
    val, vec = np.linalg.eigh(gram)
    coords = vec[:, val > 1e-10] * np.sqrt(val[val > 1e-10])
    coords *= L / np.sqrt(2.0)
    return np.pad(coords, ((0, 0), (0, d - coords.shape[1])))


def controlled_means(K: int, d: int, L: float) -> np.ndarray:
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


def gauss_mixture(name: str, K: int, d: int, sigma, L: float = 1.0,
                  unequal: bool = False) -> TargetSpec:
    means = controlled_means(K, d, L)
    if unequal:
        raw = np.arange(1, K + 1, dtype=float)
        weights = raw / raw.sum()
    else:
        weights = np.full(K, 1 / K)
    sigmas = np.full(K, sigma, dtype=float) if np.isscalar(sigma) \
        else np.asarray(sigma, dtype=float)

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        comp = rng.choice(K, size=n, p=weights)
        return means[comp] + rng.normal(size=(n, d)) * sigmas[comp, None]

    return TargetSpec(name, d, sampler, "gauss", means, weights,
                      scale=float(norm(means, axis=1).max() + sigmas.max()))


def ring_target(name: str, radius: float = 1.0,
                width: float = 0.05) -> TargetSpec:
    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        th = rng.uniform(0, 2 * np.pi, size=n)
        r = radius + rng.normal(size=n) * width
        return np.stack([r * np.cos(th), r * np.sin(th)], axis=1)

    return TargetSpec(name, 2, sampler, "ring", scale=radius + 3 * width)


def circles_target(name: str, r1: float = 0.5, r2: float = 1.0,
                   width: float = 0.04) -> TargetSpec:
    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        which = rng.integers(0, 2, size=n)
        rad = np.where(which == 0, r1, r2) + rng.normal(size=n) * width
        th = rng.uniform(0, 2 * np.pi, size=n)
        return np.stack([rad * np.cos(th), rad * np.sin(th)], axis=1)

    return TargetSpec(name, 2, sampler, "circles", scale=r2 + 3 * width)


def moons_target(name: str, scale: float = 1.0,
                 noise: float = 0.08) -> TargetSpec:
    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        which = rng.integers(0, 2, size=n)
        th = rng.uniform(0, np.pi, size=n)
        x = np.where(which == 0, np.cos(th), 1 - np.cos(th)) * scale
        y = np.where(which == 0, np.sin(th), 0.5 - np.sin(th)) * scale
        pts = np.stack([x, y], axis=1)
        return pts + rng.normal(size=(n, 2)) * noise * scale

    return TargetSpec(name, 2, sampler, "moons", scale=1.6 * scale)


def skew_target(name: str, d: int = 2) -> TargetSpec:
    """Connected, skewed, heavy-tailed: componentwise standardized
    log-normal mixed with a Student-t tail, rescaled to unit-ish spread."""

    def sampler(n: int, rng: np.random.Generator) -> np.ndarray:
        base = rng.lognormal(mean=0.0, sigma=0.55, size=(n, d)) - 1.0
        tail = rng.standard_t(df=3, size=(n, d)) * 0.15
        return 0.5 * (base + tail)

    return TargetSpec(name, d, sampler, "skew", scale=1.5)


# ---------------------------------------------------------------------------
# Splits (fixed once; never tune on HELD_OUT)
# ---------------------------------------------------------------------------


def dev_targets() -> list[TargetSpec]:
    return [gauss_mixture("DEV-2d-K4", 4, 2, 0.15)]


def validation_targets() -> list[TargetSpec]:
    return [
        gauss_mixture("V-1d-K3", 3, 1, 0.15),
        gauss_mixture("V-2d-K5", 5, 2, 0.15),
        gauss_mixture("V-2d-K4-uneq", 4, 2, 0.15, unequal=True),
        gauss_mixture("V-3d-K4", 4, 3, 0.15),
        gauss_mixture("V-2d-K4-wide", 4, 2, 0.30),
    ]


def heldout_targets() -> list[TargetSpec]:
    return [
        gauss_mixture("H-1d-K4-eq", 4, 1, 0.15),
        gauss_mixture("H-1d-K5-uneq", 5, 1, 0.12, unequal=True),
        gauss_mixture("H-2d-K6-sep", 6, 2, 0.10),
        gauss_mixture("H-2d-K3-overlap", 3, 2, 0.35),
        gauss_mixture("H-2d-K4-hetero", 4, 2, [0.08, 0.15, 0.22, 0.30]),
        ring_target("H-ring"),
        circles_target("H-circles"),
        moons_target("H-moons"),
        skew_target("H-skew"),
    ]


def init_cloud(kind: str, target: TargetSpec, N: int,
               rng: np.random.Generator) -> np.ndarray:
    """`missing`: central cloud that misses outer structure;
    `covered`: target sample plus mild jitter."""
    if kind == "missing":
        return rng.normal(size=(N, target.d)) * 0.25 * target.scale
    if kind == "covered":
        return target.sample(N, rng) + \
            rng.normal(size=(N, target.d)) * 0.05 * target.scale
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Fields (exact estimator + SNIS diagnostic)
# ---------------------------------------------------------------------------


def drift_paper(q: np.ndarray, data: np.ndarray, tau: float, mask: bool,
                counter: WorkCounter | None = None) -> np.ndarray:
    """Exact dimension-general Algorithm-2 row/column bi-softmax drift with
    the particle cloud reused as negatives (paper reuse pattern)."""
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
    Ap, An = A[:, : len(data)], A[:, len(data):]
    Wp = Ap * An.sum(axis=1, keepdims=True)
    Wn = An * Ap.sum(axis=1, keepdims=True)
    return Wp @ data - Wn @ q


def drift_snis(q: np.ndarray, data: np.ndarray, tau: float, mask: bool,
               counter: WorkCounter | None = None) -> np.ndarray:
    """SNIS mean-shift field: DIAGNOSTIC ONLY in this program."""
    if counter is not None:
        counter.add_field(len(q), len(data), len(q))

    def ms(X: np.ndarray, pts: np.ndarray, diag_mask: bool) -> np.ndarray:
        D = X[:, None, :] - pts[None, :, :]
        Km = np.exp(-norm(D, axis=2) / tau)
        if diag_mask:
            np.fill_diagonal(Km, 0.0)
        Z = Km.sum(axis=1, keepdims=True)
        Z = np.where(Z < 1e-300, np.inf, Z)
        return (Km[:, :, None] * (-D)).sum(axis=1) / Z

    return ms(q, data, False) - ms(q, q, mask)


FIELD = {"paper": drift_paper, "snis": drift_snis}


# ---------------------------------------------------------------------------
# Non-oracle geometry estimation
# ---------------------------------------------------------------------------


def _silhouette_lite(data: np.ndarray, centers: np.ndarray,
                     labels: np.ndarray) -> float:
    D = norm(data[:, None, :] - centers[None, :, :], axis=2)
    a = D[np.arange(len(data)), labels]
    D2 = D.copy()
    D2[np.arange(len(data)), labels] = np.inf
    b = D2.min(axis=1)
    return float(np.mean((b - a) / np.maximum(np.maximum(a, b), 1e-12)))


def estimate_geometry(data: np.ndarray, rng: np.random.Generator,
                      k_max: int = 8) -> dict:
    """Estimate (K_hat, L_hat, sigma_hat) from unlabeled target samples.
    Silhouette model selection; K_hat = 1 when no cluster structure."""
    best = {"K_hat": 1, "sil": -1.0}
    for k in range(2, k_max + 1):
        if k >= len(data):
            break
        try:
            centers, labels = kmeans2(data, k, iter=40, minit="++", seed=rng)
        except Exception:
            continue
        if len(np.unique(labels)) < k:
            continue
        sil = _silhouette_lite(data, centers, labels)
        if sil > best["sil"]:
            resid = data - centers[labels]
            D = norm(centers[:, None, :] - centers[None, :, :], axis=2)
            D[D == 0] = np.inf
            best = {
                "K_hat": k,
                "sil": sil,
                "L_hat": float(np.median(D.min(axis=1))),
                "sigma_hat": float(np.sqrt(np.mean(resid ** 2))),
            }
    spread = float(np.sqrt(np.mean((data - data.mean(0)) ** 2)))
    if best["K_hat"] == 1 or best["sil"] < 0.25:
        return {"K_hat": 1, "sil": best.get("sil", -1.0),
                "L_hat": 2.0 * spread, "sigma_hat": spread}
    best["L_hat"] = max(best["L_hat"], 1e-8)
    best["sigma_hat"] = max(min(best["sigma_hat"], best["L_hat"]), 1e-8)
    return best


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def energy_distance2(A: np.ndarray, B: np.ndarray) -> float:
    def cross(P: np.ndarray, Q: np.ndarray) -> float:
        total = 0.0
        for i in range(0, len(P), 256):
            total += norm(P[i:i + 256, None, :] - Q[None, :, :], axis=2).sum()
        return float(total / (len(P) * len(Q)))

    return 2 * cross(A, B) - cross(A, A) - cross(B, B)


def sliced_w1(A: np.ndarray, B: np.ndarray, n_proj: int,
              rng: np.random.Generator) -> float:
    d = A.shape[1]
    if d == 1:
        dirs = np.ones((1, 1))
    else:
        dirs = rng.normal(size=(n_proj, d))
        dirs /= norm(dirs, axis=1, keepdims=True)
    grid = np.linspace(0.01, 0.99, 99)
    total = 0.0
    for u in dirs:
        pa = np.quantile(A @ u, grid)
        pb = np.quantile(B @ u, grid)
        total += np.mean(np.abs(pa - pb))
    return float(total / len(dirs))


def mode_metrics(q: np.ndarray, target: TargetSpec) -> tuple[float, float]:
    """Oracle diagnostic; NaN for nonparametric families."""
    if target.means is None:
        return float("nan"), float("nan")
    labels = norm(q[:, None, :] - target.means[None, :, :],
                  axis=2).argmin(axis=1)
    mass = np.bincount(labels, minlength=len(target.means)) / len(q)
    coverage = float((mass >= 0.5 * target.weights).mean())
    return coverage, float(np.abs(mass - target.weights).sum())


def support_residual(q: np.ndarray, target: TargetSpec, tau: float,
                     mask: bool, rng: np.random.Generator,
                     n_ref: int = 512) -> float:
    ref = target.sample(n_ref, rng)
    V = drift_paper(q, ref, tau, mask)
    return float(np.sqrt(np.mean(np.sum(V ** 2, axis=1))))


# ---------------------------------------------------------------------------
# Training loop with pluggable policy
# ---------------------------------------------------------------------------


@dataclass
class StepDecision:
    tau: float
    eta: float
    mask: bool


class Policy:
    """A frozen algorithm: geometry setup + per-step decisions.  Policies see
    only target SAMPLES (via setup) and observable trainer state."""

    name = "base"

    def setup(self, setup_sample: np.ndarray, N: int,
              rng: np.random.Generator, counter: WorkCounter) -> None:
        pass

    def decide(self, step: int, steps: int, obs: dict) -> StepDecision:
        raise NotImplementedError


@dataclass
class TrainResult:
    q: np.ndarray
    final_ed2: float
    final_sw1: float
    coverage: float
    mass_error: float
    event_time: int
    censored: bool
    trajectory: np.ndarray
    kernel_pairs: int
    wall_seconds: float
    diverged: bool


def train(q0: np.ndarray, target: TargetSpec, policy: Policy,
          steps: int, batch: int, data_rng: np.random.Generator,
          policy_rng: np.random.Generator,
          ref_final: np.ndarray, ref_cross: np.ndarray, ed_tol: float,
          counter: WorkCounter, estimator: str = "paper",
          track_every: int = 10, setup_n: int = 256) -> TrainResult:
    """Paired randomness: `data_rng` drives the setup sample and every target
    minibatch (identical stream across arms for a given seed); `policy_rng`
    drives policy-internal randomness only."""
    q = q0.copy()
    setup_sample = target.sample(setup_n, data_rng)
    policy.setup(setup_sample, len(q), policy_rng, counter)
    hist: list[tuple[float, ...]] = []
    event_time: int | None = None
    start_pairs = counter.kernel_pairs
    t0 = time.time()
    diverged = False
    obs: dict = {"last_V": None, "last_ed": None, "counter": counter}
    for step in range(1, steps + 1):
        data = target.sample(batch, data_rng)
        dec = policy.decide(step - 1, steps, obs)
        V = FIELD[estimator](q, data, dec.tau, dec.mask, counter)
        obs["last_V"] = V
        obs["data"] = data
        obs["q"] = q
        q = q + dec.eta * V
        if not np.isfinite(q).all() or norm(q) > 1e6 * max(target.scale, 1.0):
            diverged = True
            break
        ed_cross = energy_distance2(q, ref_cross)
        obs["last_ed"] = ed_cross
        if event_time is None and ed_cross < ed_tol:
            event_time = step
        if step == 1 or step % track_every == 0 or step == steps:
            cov, me = mode_metrics(q, target)
            hist.append((step, ed_cross, cov, me, dec.tau, dec.eta,
                         float(dec.mask)))
    final_ed2 = energy_distance2(q, ref_final) if not diverged else float("inf")
    final_sw1 = sliced_w1(q, ref_final, 32,
                          np.random.default_rng(MASTER + 91)) \
        if not diverged else float("inf")
    cov, me = mode_metrics(q, target)
    return TrainResult(
        q=q, final_ed2=final_ed2, final_sw1=final_sw1, coverage=cov,
        mass_error=me,
        event_time=event_time if event_time is not None else steps,
        censored=event_time is None,
        trajectory=np.asarray(hist),
        kernel_pairs=counter.kernel_pairs - start_pairs,
        wall_seconds=time.time() - t0, diverged=diverged)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def km_median(times: list[int], censored: list[bool]) -> int | None:
    records = sorted(zip(times, censored))
    at_risk = len(records)
    survival = 1.0
    for t in sorted(set(times)):
        at_t = [(tt, c) for tt, c in records if tt == t]
        events = sum(not c for _, c in at_t)
        if at_risk and events:
            survival *= 1.0 - events / at_risk
            if survival <= 0.5:
                return t
        at_risk -= len(at_t)
    return None


def paired_bootstrap_ratio(mod: np.ndarray, base: np.ndarray,
                           n_boot: int = 4000,
                           seed: int = MASTER) -> tuple[float, float, float]:
    """Median of paired log-ratios, exponentiated, with a bootstrap 95% CI.
    Values are per-pair positive errors (e.g. final ED^2)."""
    ratios = np.log(np.maximum(mod, 1e-12) / np.maximum(base, 1e-12))
    rng = np.random.default_rng(seed)
    point = float(np.exp(np.median(ratios)))
    stats = np.empty(n_boot)
    n = len(ratios)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = np.exp(np.median(ratios[idx]))
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Invariants (D0 gate)
# ---------------------------------------------------------------------------


def field_invariants(log: Callable[[str], None]) -> None:
    rng = np.random.default_rng(MASTER)
    q = rng.normal(size=(7, 2))
    y = rng.normal(size=(11, 2))
    t = rng.normal(size=2)
    for name, f in FIELD.items():
        v = f(q, y, 0.8, False)
        vt = f(q + t, y + t, 0.8, False)
        assert norm(v - vt) < 1e-11, f"translation {name}"
        assert np.isfinite(v).all(), f"finite {name}"
        log(f"  invariant {name}: translation/finite PASS")
    # matched positive/negative cancellation (certified matched-batch lemma)
    vz = drift_paper(q, q, 0.8, False)
    assert norm(vz) < 1e-11, "matched-batch cancellation"
    log("  invariant paper matched-batch cancellation: PASS")
    # 1-d agreement with the independent driftlab port
    sys.path.insert(0, str(HERE))
    import driftlab  # noqa: PLC0415

    x = rng.normal(size=6)
    yp = rng.normal(size=9)
    for mask in (False, True):
        got = drift_paper(x[:, None], yp[:, None], 0.7, mask)[:, 0]
        want = driftlab.compute_v_paper(x, yp, x.copy(), 0.7, mask)
        assert norm(got - want) < 1e-11, f"driftlab agreement mask={mask}"
    log("  invariant paper-ND vs driftlab.compute_v_paper: PASS")
