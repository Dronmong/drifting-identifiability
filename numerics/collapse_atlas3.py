"""Collapse Atlas pass 3: complete the numerics program.

P3A  mask-shift N-scaling (masked equilibrium distance from p, stability)
P3B  full basin matrix (P1+P2, tau x N x spread, dual criterion, 6-way
     classes, collapse-biased inits, Wilson CIs, quantization floors)
P3C  zero-finding coverage (1-d bracketing + outer bound; 2-d seed stability)
P3D  figures from stored raw data -> numerics/atlas_figs/
P3E  bandwidth-ladder design curve (mixture weight x second bandwidth)

Usage:
    uv run --with numpy --with scipy --with matplotlib python numerics/collapse_atlas3.py [P3A|P3B|P3C|P3D|P3E|all]
"""
import glob
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from numpy.linalg import eigvals, norm
from scipy.optimize import brentq, root

MASTER = 20260718
RUNROOT = "numerics/atlas_runs"
FIGDIR = "numerics/atlas_figs"


class Run:
    def __init__(self, which):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = os.path.join(RUNROOT, f"{stamp}-{which}")
        os.makedirs(self.dir, exist_ok=True)
        try:
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            commit = "unknown"
        self.manifest = {
            "which": which, "seed": MASTER, "commit": commit,
            "python": sys.version, "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "cmdline": " ".join(sys.argv), "start": stamp}
        self.lines = []

    def log(self, s=""):
        print(s)
        self.lines.append(s)

    def save_csv(self, name, header, rows):
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(str(x) for x in r) + "\n")

    def finish(self):
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=1)
        with open(os.path.join(self.dir, "summary.md"), "w",
                  encoding="utf-8") as f:
            f.write("```\n" + "\n".join(self.lines) + "\n```\n")


def mean_shift(X, pts, w, tau):
    X = np.atleast_2d(X)
    D = X[:, None, :] - pts[None, :, :]
    r = norm(D, axis=2)
    K = w[None, :] * np.exp(-r / tau)
    Z = K.sum(axis=1)
    return (K[:, :, None] * (-D)).sum(axis=1) / Z[:, None]


def drift(X, p_pts, p_w, q_pts, q_w, tau):
    return mean_shift(X, p_pts, p_w, tau) - mean_shift(X, q_pts, q_w, tau)


def drift_masked(q_pts, p_pts, p_w, q_w, tau):
    N = len(q_pts)
    D = q_pts[:, None, :] - q_pts[None, :, :]
    r = norm(D, axis=2)
    K = q_w[None, :] * np.exp(-r / tau)
    np.fill_diagonal(K, 0.0)
    Z = K.sum(axis=1)
    mq = (K[:, :, None] * (-D)).sum(axis=1) / Z[:, None]
    return mean_shift(q_pts, p_pts, p_w, tau) - mq


def ed_sq(p_pts, p_w, q_pts, q_w):
    def cross(A, a, B, b):
        return (a[:, None] * b[None, :]
                * norm(A[:, None, :] - B[None, :, :], axis=2)).sum()
    return 2 * cross(p_pts, p_w, q_pts, q_w) - cross(p_pts, p_w, p_pts, p_w) \
        - cross(q_pts, q_w, q_pts, q_w)


def two_cluster(rng, d, L, sig, weights=(0.5, 0.5), atoms=3):
    A = rng.normal(size=(atoms, d)) * sig
    B = rng.normal(size=(atoms, d)) * sig
    B[:, 0] += L
    pts = np.vstack([A, B])
    w = np.concatenate([np.full(atoms, weights[0] / atoms),
                        np.full(atoms, weights[1] / atoms)])
    return pts, w


# ---------------- P3A: mask-shift N-scaling ----------------

def P3A():
    run = Run("P3A-mask-scaling")
    run.log("P3A: masked-field equilibrium distance from p vs N")
    d, L, tau = 2, 5.0, 1.0
    rng = np.random.default_rng(MASTER + 10)
    p_pts, p_w = two_cluster(rng, d, L, 0.3)
    rows = []
    for k in (1, 2, 4, 8, 16):
        N = 6 * k
        q = np.repeat(p_pts, k, axis=0) + \
            np.random.default_rng(MASTER + k).normal(size=(N, d)) * 1e-8
        q_w = np.full(N, 1 / N)
        eta = 0.05 * tau
        for t in range(60000):
            V = drift_masked(q, p_pts, p_w, q_w, tau)
            q = q + eta * V
            if t % 500 == 0 and norm(V, axis=1).max() < 1e-12:
                break
        res = norm(drift_masked(q, p_pts, p_w, q_w, tau), axis=1).max()
        dmax = min_att = np.max(
            [norm(q[j] - p_pts, axis=1).min() for j in range(N)])
        eds = ed_sq(p_pts, p_w, q, q_w)
        # stability of the masked equilibrium
        h = 1e-7

        def field(v):
            return drift_masked(v.reshape(N, d), p_pts, p_w, q_w, tau).ravel()

        v0 = q.ravel()
        G = np.zeros((N * d, N * d))
        for i in range(N * d):
            e = np.zeros(N * d); e[i] = h
            G[:, i] = (field(v0 + e) - field(v0 - e)) / (2 * h)
        mre = eigvals(G).real.max()
        rows.append((N, res, dmax, eds, mre))
        run.log(f"  N={N:3d}: resid={res:.1e} max-shift={dmax:.4e} "
                f"EDsq={eds:.3e} maxReEig={mre:+.3e}")
    arr = np.array(rows)
    sl = np.polyfit(np.log(arr[:, 0]), np.log(np.maximum(arr[:, 2], 1e-16)),
                    1)[0]
    run.log(f"  log-log slope of max-shift vs N: {sl:.3f} "
            "(certified prediction O(1/N) ~ -1)")
    run.save_csv("p3a_scaling.csv",
                 ["N", "resid", "max_shift", "ED_sq", "max_re_eig"], rows)
    run.finish()


# ---------------- P3B: full basin matrix ----------------

def classify(p_pts, p_w, q, q_w, tau, L, alpha_p, ed_floor):
    res = norm(drift(q, p_pts, p_w, q, q_w, tau), axis=1).max()
    eds = ed_sq(p_pts, p_w, q, q_w)
    diam = norm(q - q.mean(axis=0), axis=1).max()
    alpha = (q[:, 0] < L / 2).mean()
    ed_tol = max(0.02 * L, 2.0 * ed_floor)
    if eds < ed_tol and res < 1e-3 * tau:
        return "target_stationary"
    if eds < ed_tol:
        return "target_moving"
    if diam < 0.05 * L:
        return "collapsed"
    if res < 1e-3 * tau and abs(alpha - alpha_p) > 0.1:
        return "metastable"
    return "unresolved"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    den = 1 + z * z / n
    ctr = (ph + z * z / (2 * n)) / den
    hw = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, ctr - hw), min(1.0, ctr + hw))


def P3B():
    run = Run("P3B-basins")
    run.log("P3B: full basin matrix; 100 inits/cell (50 for N=128); "
            "6-way classes; 4x extension of unresolved; Wilson CI for collapse")
    d, L = 2, 5.0
    rng0 = np.random.default_rng(MASTER + 10)
    fams = {"P1": two_cluster(rng0, d, L, 0.3),
            "P2": two_cluster(rng0, d, L, 0.3, weights=(0.3, 0.7))}
    alphas = {"P1": 0.5, "P2": 0.3}
    rows = []
    for fam, (p_pts, p_w) in fams.items():
        for tau_f in (0.2, 0.5, 1.0):
            tau = tau_f * L
            for N in (8, 32, 128):
                # quantization floor for this (fam, N)
                kA = int(round(alphas[fam] * N))
                idx = np.concatenate([
                    np.tile(np.arange(3), (kA // 3 + 1))[:kA],
                    np.tile(np.arange(3, 6), ((N - kA) // 3 + 1))[:N - kA]])
                ed_floor = ed_sq(p_pts, p_w, p_pts[idx], np.full(N, 1 / N))
                for spread_f in (0.5, 2.0, 5.0):
                    spread = spread_f * L
                    n_init = 50 if N == 128 else 100
                    counts = {}
                    for rep in range(n_init):
                        rr = np.random.default_rng(
                            MASTER + (hash((fam, tau_f, N, spread_f, rep))
                                      % 10 ** 8))
                        q = rr.normal(size=(N, d)) * spread \
                            + p_pts.mean(axis=0)
                        q_w = np.full(N, 1 / N)
                        eta = 0.1 * tau
                        steps = 1200
                        for _ in range(steps):
                            q = q + eta * drift(q, p_pts, p_w, q, q_w, tau)
                        c = classify(p_pts, p_w, q, q_w, tau, L,
                                     alphas[fam], ed_floor)
                        if c == "unresolved":       # extend 4x
                            for _ in range(4 * steps):
                                q = q + eta * drift(q, p_pts, p_w, q, q_w,
                                                    tau)
                            c = classify(p_pts, p_w, q, q_w, tau, L,
                                         alphas[fam], ed_floor)
                            if c == "unresolved":
                                c = "unresolved_extended"
                        counts[c] = counts.get(c, 0) + 1
                    ncol = counts.get("collapsed", 0)
                    lo, hi = wilson(ncol, n_init)
                    run.log(f"  {fam} tau={tau_f}L N={N:3d} s={spread_f}L: "
                            f"{counts}  collapseCI=[{lo:.3f},{hi:.3f}] "
                            f"(edfloor={ed_floor:.3f})")
                    rows.append((fam, tau_f, N, spread_f, json.dumps(counts),
                                 lo, hi, ed_floor))
    # collapse-biased initializations: start inside a sink's fission zone
    run.log("  collapse-biased inits (all particles at a cluster sink "
            "+ 0.05*tau jitter):")
    p_pts, p_w = fams["P1"]
    for tau_f, N in ((0.2, 8), (0.2, 32), (1.0, 8)):
        tau = tau_f * L
        f = lambda x: mean_shift(x[None, :], p_pts, p_w, tau)[0]
        sol = root(f, p_pts[:3].mean(axis=0), method="hybr", tol=1e-12)
        sink = sol.x
        counts = {}
        for rep in range(50):
            rr = np.random.default_rng(MASTER + 9000 + rep)
            q = sink + rr.normal(size=(N, d)) * 0.05 * tau
            q_w = np.full(N, 1 / N)
            eta = 0.1 * tau
            for _ in range(6000):
                q = q + eta * drift(q, p_pts, p_w, q, q_w, tau)
            kA = int(round(0.5 * N))
            idx = np.concatenate([
                np.tile(np.arange(3), (kA // 3 + 1))[:kA],
                np.tile(np.arange(3, 6), ((N - kA) // 3 + 1))[:N - kA]])
            ed_floor = ed_sq(p_pts, p_w, p_pts[idx], np.full(N, 1 / N))
            c = classify(p_pts, p_w, q, q_w, tau, L, 0.5, ed_floor)
            counts[c] = counts.get(c, 0) + 1
        run.log(f"    tau={tau_f}L N={N}: {counts} "
                "(escape-from-collapse test)")
        rows.append(("P1-biased", tau_f, N, 0.05, json.dumps(counts),
                     np.nan, np.nan, np.nan))
    run.save_csv("p3b_basins.csv",
                 ["family", "tau_over_L", "N", "spread_over_L", "counts",
                  "collapse_ci_lo", "collapse_ci_hi", "ed_floor"], rows)
    run.finish()


# ---------------- P3C: zero-finding coverage ----------------

def P3C():
    run = Run("P3C-zero-coverage")
    run.log("P3C: hardened zero finding")
    rng0 = np.random.default_rng(MASTER + 10)
    # 1-d: bracketing + tangency detection + outer bound
    d, L = 1, 5.0
    p_pts, p_w = two_cluster(rng0, d, L, 0.3)
    run.log("  1-d exhaustive bracketing (outer bound: zeros confined to "
            "atom hull, since m_p points inward outside it):")
    for tau in (0.5, 1.0, 2.5, 6.0):
        f = lambda x: mean_shift(np.array([[x]]), p_pts, p_w, tau)[0][0]
        lo, hi = p_pts.min() - 1e-9, p_pts.max() + 1e-9
        xs = np.linspace(lo, hi, 4001)
        vals = np.array([f(x) for x in xs])
        zeros = []
        for i in range(len(xs) - 1):
            if vals[i] == 0.0:
                zeros.append(xs[i])
            elif vals[i] * vals[i + 1] < 0:
                zeros.append(brentq(f, xs[i], xs[i + 1], xtol=1e-14))
        # tangency scan: local minima of |m| below tolerance w/o sign change
        av = np.abs(vals)
        tang = [xs[i] for i in range(1, len(xs) - 1)
                if av[i] < av[i - 1] and av[i] < av[i + 1]
                and av[i] < 1e-8 and vals[i - 1] * vals[i + 1] > 0]
        run.log(f"    tau={tau:4.1f}: bracketed zeros={len(zeros)} "
                f"tangencies={len(tang)} "
                f"(pass-1 census on this family: cross-check)")
    # 2-d seed-count stability
    d = 2
    p_pts, p_w = two_cluster(np.random.default_rng(MASTER + 10), d, L, 0.3)
    run.log("  2-d multistart stability (zeros found vs seed count):")
    for tau in (0.5, 1.0):
        f = lambda x: mean_shift(x[None, :], p_pts, p_w, tau)[0]
        for n_seeds in (250, 1000, 4000):
            rng = np.random.default_rng(MASTER + 77)
            lo = p_pts.min(axis=0) - 3 * tau
            hi = p_pts.max(axis=0) + 3 * tau
            seeds = np.vstack([rng.uniform(lo, hi, size=(n_seeds, d)),
                               p_pts])
            zeros = []
            for s in seeds:
                sol = root(f, s, method="hybr", tol=1e-12)
                if sol.success and norm(f(sol.x)) < 1e-9:
                    if all(norm(sol.x - z) > 1e-4 for z in zeros):
                        zeros.append(sol.x)
            run.log(f"    tau={tau} seeds={n_seeds}: zeros={len(zeros)}")
    run.finish()


# ---------------- P3D: figures ----------------

def P3D():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run = Run("P3D-figures")
    os.makedirs(FIGDIR, exist_ok=True)
    # (1) metastability from the latest P2C csv
    cands = sorted(glob.glob(os.path.join(RUNROOT, "*P2C*", "p2c_relax.csv")))
    if cands:
        data = np.genfromtxt(cands[-1], delimiter=",", names=True)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for L, mk in ((5.0, "o"), (8.0, "s")):
            sel = (data["L"] == L) & (data["censored"] == 0)
            ax.semilogy(data["L_over_tau"][sel], data["T_over_tau"][sel],
                        mk, label=f"L={L}", alpha=0.7)
        ax.set_xlabel("L / tau"); ax.set_ylabel("relaxation time T / tau")
        ax.set_title("Mass-imbalance metastability (slope ~ e^{1.7 L/tau})")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "metastability.png"), dpi=140)
        run.log("  wrote atlas_figs/metastability.png")
    # (2) 1-d splitting-index landscape
    rng0 = np.random.default_rng(MASTER + 10)
    p_pts, p_w = two_cluster(rng0, 1, 5.0, 0.3)
    tau = 1.0
    xs = np.linspace(-2, 7, 800)

    def mp(x):
        return mean_shift(np.array([[x]]), p_pts, p_w, tau)[0][0]

    ms = np.array([mp(x) for x in xs])
    h = 1e-6
    sig = np.array([1 + (mp(x + h) - mp(x - h)) / (2 * h) for x in xs])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, ms, label="m_p(x)")
    ax.plot(xs, sig, label="sigma(x) = 1 + m_p'(x)")
    ax.axhline(0, color="k", lw=0.5)
    for a in p_pts[:, 0]:
        ax.axvline(a, color="gray", ls=":", lw=0.5)
    ax.set_title("1-d field and splitting index (P1, tau=1)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "splitting_1d.png"), dpi=140)
    run.log("  wrote atlas_figs/splitting_1d.png")
    # (3) mask-shift scaling
    cands = sorted(glob.glob(os.path.join(RUNROOT, "*P3A*",
                                          "p3a_scaling.csv")))
    if cands:
        data = np.genfromtxt(cands[-1], delimiter=",", names=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.loglog(data["N"], data["max_shift"], "o-")
        ax.loglog(data["N"], data["max_shift"][0] * data["N"][0]
                  / data["N"], "--", label="~1/N")
        ax.set_xlabel("N"); ax.set_ylabel("masked equilibrium shift")
        ax.set_title("Self-mask equilibrium shift vs N")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(FIGDIR, "mask_shift.png"), dpi=140)
        run.log("  wrote atlas_figs/mask_shift.png")
    run.finish()


# ---------------- P3E: bandwidth-ladder design curve ----------------

def P3E():
    run = Run("P3E-ladder")
    run.log("P3E: two-bandwidth design curve at L/tau=5 "
            "(mixture V = (1-b) V_tau + b V_tau')")
    d, N, L, alpha0 = 1, 40, 5.0, 0.2
    ratio = 5.0
    tau = L / ratio
    sig = 0.25 * tau
    rng = np.random.default_rng(MASTER + 700)
    A = rng.normal(size=(6, d)) * sig
    B = rng.normal(size=(6, d)) * sig + L * np.eye(1, d)[0]
    p_pts, p_w = np.vstack([A, B]), np.full(12, 1 / 12)
    nA = int(alpha0 * N)
    q0 = np.vstack([rng.normal(size=(nA, d)) * sig,
                    rng.normal(size=(N - nA, d)) * sig + L * np.eye(1, d)[0]])
    q_w = np.full(N, 1 / N)
    rows = []
    for tp_f in (0.25, 0.5, 1.0, 2.0):
        tp = tp_f * L
        for beta in (0.25, 0.5):
            q = q0.copy()
            eta = 0.1 * tau
            T, cens = None, True
            for t in range(150000):
                V = (1 - beta) * drift(q, p_pts, p_w, q, q_w, tau) \
                    + beta * drift(q, p_pts, p_w, q, q_w, tp)
                q = q + eta * V
                if abs((q[:, 0] < L / 2).mean() - 0.5) \
                        < 0.5 * abs(alpha0 - 0.5):
                    T, cens = t * eta, False
                    break
            rows.append((tp_f, beta, (T or np.nan) / tau, int(cens)))
            run.log(f"  tau'={tp_f}L beta={beta}: T/tau="
                    f"{(T or np.nan)/tau:.1f}{' (censored)' if cens else ''}")
    run.save_csv("p3e_ladder.csv",
                 ["taup_over_L", "beta", "T_over_tau", "censored"], rows)
    run.finish()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"P3A": P3A, "P3B": P3B, "P3C": P3C, "P3D": P3D, "P3E": P3E}
    for name, fn in fns.items():
        if which in ("all", name):
            fn()
