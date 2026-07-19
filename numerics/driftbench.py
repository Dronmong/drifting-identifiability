"""Phase C: benchmark the certified drifting design rules at the finite-sample
(estimator) level.  Spec: numerics/PhaseC_DesignRules.md.

Usage:
    uv run --with numpy --with scipy --with matplotlib python numerics/driftbench.py [C1|C2|C3|all]

Runs write numerics/bench_runs/<id>/ (manifest + csv/npz + summary).
"""
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from numpy.linalg import norm

MASTER = 20260719
RUNROOT = "numerics/bench_runs"


# ---------------- provenance ----------------

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
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "start": stamp}
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


# ---------------- target distributions ----------------

def make_target(K, d, L, rng):
    """K isotropic Gaussians, separation ~ L, width 0.15 L.  Returns
    (means [K,d], sigma, sampler)."""
    if d == 1:
        means = (np.arange(K) - (K - 1) / 2)[:, None] * L
    else:
        # spread modes on a scaled grid / simplex-ish layout
        pts = rng.normal(size=(K, d))
        pts = pts / norm(pts, axis=1, keepdims=True)
        means = pts * L * (0.6 + 0.4 * rng.random((K, 1)))
    sigma = 0.15 * L

    def sample(n, r):
        comp = r.integers(0, K, size=n)
        return means[comp] + r.normal(size=(n, d)) * sigma
    return means, sigma, sample


# ---------------- kernel field (SNIS mean-shift estimator) ----------------

def mean_shift_est(X, pts, tau, eye_mask=False):
    """SNIS mean shift m_hat at probes X (n,d) using support atoms pts (m,d)."""
    D = X[:, None, :] - pts[None, :, :]          # (n,m,d)
    r = norm(D, axis=2)                          # (n,m)
    K = np.exp(-r / tau)
    if eye_mask and X.shape[0] == pts.shape[0] and X is pts:
        np.fill_diagonal(K, 0.0)
    Z = K.sum(axis=1, keepdims=True)
    Z = np.where(Z < 1e-300, np.inf, Z)          # dead zone -> 0 field
    return (K[:, :, None] * (-D)).sum(axis=1) / Z


def drift_est(q, data_batch, tau, eye_mask=False):
    """V_hat(x_j) = m_p_hat - m_q_hat, single bandwidth."""
    mp = mean_shift_est(q, data_batch, tau)
    # m_q with optional self-mask (particle excludes itself)
    D = q[:, None, :] - q[None, :, :]
    r = norm(D, axis=2)
    Kq = np.exp(-r / tau)
    if eye_mask:
        np.fill_diagonal(Kq, 0.0)
    Zq = Kq.sum(axis=1, keepdims=True)
    Zq = np.where(Zq < 1e-300, np.inf, Zq)
    mq = (Kq[:, :, None] * (-D)).sum(axis=1) / Zq
    return mp - mq


def drift_est_multi(q, data_batch, taus, weights, eye_mask=False):
    """Mixture-of-bandwidths drift: weighted average of per-tau drifts."""
    V = np.zeros_like(q)
    wsum = 0.0
    for tau, w in zip(taus, weights):
        V += w * drift_est(q, data_batch, tau, eye_mask)
        wsum += w
    return V / wsum


# ---------------- metrics ----------------

def energy_distance(A, B):
    """Sample energy distance^2 between point sets A, B (unweighted)."""
    def cross(P, Q):
        # mean pairwise distance; chunk to bound memory
        s, nP = 0.0, len(P)
        for i in range(0, nP, 256):
            s += norm(P[i:i + 256, None, :] - Q[None, :, :],
                      axis=2).sum()
        return s / (len(P) * len(Q))
    return 2 * cross(A, B) - cross(A, A) - cross(B, B)


def mode_metrics(q, means, K):
    """coverage fraction and mode-mass L1 error."""
    lbl = norm(q[:, None, :] - means[None, :, :], axis=2).argmin(axis=1)
    mass = np.bincount(lbl, minlength=K) / len(q)
    coverage = (mass >= 0.5 / K).mean()
    mass_err = np.abs(mass - 1.0 / K).sum()
    return coverage, mass_err


# ---------------- training loop ----------------

def train(q0, sample_p, tau_cfg, eta, steps, batch, rng, means, K,
          ref, eye_mask=False, track_every=20):
    q = q0.copy()
    hist = []
    for t in range(steps):
        Y = sample_p(batch, rng)
        if callable(tau_cfg):                 # coarse-to-fine schedule
            tau_t = tau_cfg(t, steps)
            V = drift_est(q, Y, tau_t, eye_mask)
            eta_t = 0.1 * tau_t
        elif isinstance(tau_cfg, tuple):
            taus, weights = tau_cfg
            V = drift_est_multi(q, Y, taus, weights, eye_mask)
            eta_t = eta
        else:
            V = drift_est(q, Y, tau_cfg, eye_mask)
            eta_t = eta
        q = q + eta_t * V
        if t % track_every == 0:
            ed = energy_distance(q, ref)
            cov, me = mode_metrics(q, means, K)
            hist.append((t, ed, cov, me))
    ed = energy_distance(q, ref)
    cov, me = mode_metrics(q, means, K)
    hist.append((steps, ed, cov, me))
    return q, np.array(hist)


def steps_to_target(hist, ed_tol):
    for t, ed, cov, me in hist:
        if ed < ed_tol:
            return int(t)
    return None


# ---------------- C1: bandwidth ladder ----------------

def C1():
    run = Run("C1-bandwidth-ladder")
    run.log("C1: bandwidth ladder vs single-best vs paper-multi")
    run.log("target: K Gaussians, sep L, width 0.15L; N=20K particles, "
            "batch=64, 800 steps, eta=0.1*tau_fine")
    rows = []
    for K, d, L in [(4, 1, 5.0), (4, 2, 5.0), (8, 2, 5.0), (4, 5, 5.0)]:
        rng0 = np.random.default_rng(MASTER + K * 100 + d * 10)
        means, sigma, sample_p = make_target(K, d, L, rng0)
        tau_fine = 0.5 * sigma           # sharp final placement
        # single-best: grid over tau, pick best final ED (baseline's best case)
        grid = [0.3 * sigma, sigma, 0.3 * L, L]
        N = 20 * K
        ref = sample_p(2000, np.random.default_rng(MASTER + 7))
        def anneal(t, steps, Lv=L, tf=tau_fine):
            # geometric coarse-to-fine from L to tau_fine over 70% of the run
            frac = min(1.0, t / (0.7 * steps))
            return Lv * (tf / Lv) ** frac

        arms = {
            "single-fine": tau_fine,
            "single-L": L,
            "paper-multi": ([0.3 * sigma, sigma, 0.3 * L],
                            [1.0, 1.0, 1.0]),
            "ladder-eq": ([tau_fine, L], [1.0, 1.0]),
            "anneal": anneal,
        }
        # find single-best across grid
        best_ed, best_tau = np.inf, None
        for tau in grid:
            eds = []
            for s in range(2):
                rr = np.random.default_rng(MASTER + 1000 + s)
                q0 = rr.normal(size=(N, d)) * L
                _, h = train(q0, sample_p, tau, 0.1 * tau, 400, 64, rr,
                             means, K, ref)
                eds.append(h[-1, 1])
            if np.median(eds) < best_ed:
                best_ed, best_tau = np.median(eds), tau
        arms["single-best"] = best_tau
        run.log(f"  --- K={K} d={d} L={L}: single-best tau={best_tau:.3f} "
                f"(sigma={sigma:.3f}) ---")
        ed_tol = 0.05 * L
        for name, cfg in arms.items():
            if callable(cfg):
                eta_tau = tau_fine       # unused (schedule sets its own eta)
            elif isinstance(cfg, tuple):
                eta_tau = cfg[0][0]
            else:
                eta_tau = cfg
            finals, covs, mes, tts = [], [], [], []
            for s in range(4):
                rr = np.random.default_rng(MASTER + 2000 + s)
                q0 = rr.normal(size=(N, d)) * L
                _, h = train(q0, sample_p, cfg, 0.1 * eta_tau, 800, 64, rr,
                             means, K, ref)
                finals.append(h[-1, 1]); covs.append(h[-1, 2])
                mes.append(h[-1, 3])
                tt = steps_to_target(h, ed_tol)
                tts.append(tt if tt is not None else 9999)
            run.log(f"    {name:12s}: ED={np.median(finals):.4f} "
                    f"cover={np.median(covs):.2f} massErr={np.median(mes):.3f} "
                    f"steps->tol={int(np.median(tts))}")
            rows.append((K, d, L, name, np.median(finals), np.median(covs),
                         np.median(mes), int(np.median(tts))))
    run.save_csv("c1_results.csv",
                 ["K", "d", "L", "arm", "ED", "coverage", "mass_err",
                  "steps_to_tol"], rows)
    run.finish()
    return run.dir


# ---------------- C2: step-size rule ----------------

def local_eta_star(q, sample_p, tau, rng, batch=64):
    """Estimate eta* = min -2 Re lam / |lam|^2 from a finite-difference of the
    empirical field Jacobian on a small particle subset."""
    idx = rng.choice(len(q), size=min(6, len(q)), replace=False)
    Y = sample_p(batch, rng)
    h = 1e-4 * tau
    eigs = []
    for j in idx:
        d = q.shape[1]
        J = np.zeros((d, d))
        base = drift_est(q[j:j + 1], Y, tau)[0]
        for k in range(d):
            e = np.zeros(d); e[k] = h
            Jp = drift_est((q[j:j + 1] + e), Y, tau)[0]
            J[:, k] = (Jp - base) / h
        ev = np.linalg.eigvals(J)
        for lam in ev:
            if lam.real < -1e-9:
                eigs.append(-2 * lam.real / (abs(lam) ** 2))
    return min(eigs) if eigs else tau


def C2():
    run = Run("C2-step-size")
    run.log("C2: generator eta* vs fixed-eta grid")
    rows = []
    for K, d, L in [(4, 1, 5.0), (4, 2, 5.0)]:
        rng0 = np.random.default_rng(MASTER + K * 100 + d * 10)
        means, sigma, sample_p = make_target(K, d, L, rng0)
        tau = sigma
        N = 20 * K
        ref = sample_p(2000, np.random.default_rng(MASTER + 7))
        run.log(f"  --- K={K} d={d} L={L} tau={tau:.3f} ---")
        # estimate eta* once from a settled-ish cloud
        rr = np.random.default_rng(MASTER + 3300)
        q_probe = sample_p(N, rr) + rr.normal(size=(N, d)) * sigma
        est = local_eta_star(q_probe, sample_p, tau, rr)
        run.log(f"    estimated eta* = {est/tau:.3f}*tau")
        for name, eta in [("fixed-0.1tau", 0.1 * tau),
                          ("fixed-0.5tau", 0.5 * tau),
                          ("fixed-1.0tau", 1.0 * tau),
                          ("fixed-2.0tau", 2.0 * tau),
                          ("eta*", est)]:
            finals, divg = [], 0
            for s in range(4):
                rs = np.random.default_rng(MASTER + 4400 + s)
                q0 = rs.normal(size=(N, d)) * L
                _, h = train(q0, sample_p, tau, eta, 600, 64, rs,
                             means, K, ref)
                ed = h[-1, 1]
                if not np.isfinite(ed) or ed > 10 * L:
                    divg += 1
                else:
                    finals.append(ed)
            med = np.median(finals) if finals else np.inf
            run.log(f"    {name:14s}: ED={med:.4f} diverged={divg}/4")
            rows.append((K, d, L, name, eta / tau, med, divg))
    run.save_csv("c2_results.csv",
                 ["K", "d", "L", "arm", "eta_over_tau", "ED", "diverged"],
                 rows)
    run.finish()
    return run.dir


# ---------------- C3: mask policy ----------------

def C3():
    run = Run("C3-mask")
    run.log("C3: eye-mask on/off x particles-per-mode (SNIS mean-shift drift)")
    rows = []
    K, d, L = 4, 2, 5.0
    rng0 = np.random.default_rng(MASTER + 555)
    means, sigma, sample_p = make_target(K, d, L, rng0)
    tau = sigma
    ref = sample_p(2000, np.random.default_rng(MASTER + 7))
    for per_mode in (2, 8, 32):
        N = per_mode * K
        for mask in (False, True):
            finals, covs, mes = [], [], []
            for s in range(5):
                rs = np.random.default_rng(MASTER + 6600 + s)
                q0 = rs.normal(size=(N, d)) * L
                _, h = train(q0, sample_p, tau, 0.1 * tau, 800, 64, rs,
                             means, K, ref, eye_mask=mask)
                finals.append(h[-1, 1]); covs.append(h[-1, 2])
                mes.append(h[-1, 3])
            run.log(f"  N/K={per_mode:2d} N={N:3d} mask={str(mask):5s}: "
                    f"ED={np.median(finals):.4f} "
                    f"cover={np.median(covs):.2f} "
                    f"massErr={np.median(mes):.3f}")
            rows.append((per_mode, N, mask, np.median(finals),
                         np.median(covs), np.median(mes)))
    run.save_csv("c3_results.csv",
                 ["per_mode", "N", "mask", "ED", "coverage", "mass_err"],
                 rows)
    run.finish()
    return run.dir


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"C1": C1, "C2": C2, "C3": C3}
    for name, fn in fns.items():
        if which in ("all", name):
            fn()
