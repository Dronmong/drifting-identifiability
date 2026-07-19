"""Collapse Atlas pass 2: audit-hardened validation of the two central
discoveries (fission instability; exponential mass metastability) plus the
flow-level Lyapunov test and the continuous-generator stability boundary.

Spec: CollapseAtlas.md (Pass 2).  Audit: CollapseAtlasAudit.md.

Usage:
    uv run --with numpy --with scipy python numerics/collapse_atlas2.py [P2A|P2B|P2C|P2D|P2E|all]

Each invocation writes numerics/atlas_runs/<run-id>/ with manifest.json,
raw CSVs, and summary.md.  Nothing is appended to CollapseAtlasResults.md.
"""
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from numpy.linalg import eigvals, norm, svd
from scipy.optimize import differential_evolution, minimize, root

MASTER = 20260718
RUNROOT = "numerics/atlas_runs"


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
            "python": sys.version, "numpy": np.__version__,
            "scipy": scipy.__version__, "platform": platform.platform(),
            "cmdline": " ".join(sys.argv), "start": stamp,
        }
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
        self.manifest["wall_seconds"] = round(
            time.time() - time.mktime(time.strptime(
                self.manifest["start"], "%Y%m%d-%H%M%S")), 1)
        with open(os.path.join(self.dir, "manifest.json"), "w") as f:
            json.dump(self.manifest, f, indent=1)
        with open(os.path.join(self.dir, "summary.md"), "w",
                  encoding="utf-8") as f:
            f.write("```\n" + "\n".join(self.lines) + "\n```\n")


# ---------------- fields ----------------

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
    """Paper-style eye mask: particle j's own mass removed from m_q."""
    N = len(q_pts)
    D = q_pts[:, None, :] - q_pts[None, :, :]
    r = norm(D, axis=2)
    K = q_w[None, :] * np.exp(-r / tau)
    np.fill_diagonal(K, 0.0)
    Z = K.sum(axis=1)
    mq = (K[:, :, None] * (-D)).sum(axis=1) / Z[:, None]
    return mean_shift(q_pts, p_pts, p_w, tau) - mq


def jac_fd(f, x, h):
    d = x.size
    J = np.zeros((d, d))
    for i in range(d):
        e = np.zeros(d); e[i] = h
        J[:, i] = (f(x + e) - f(x - e)) / (2 * h)
    return J


def find_zero_near(p_pts, p_w, tau, seed):
    f = lambda x: mean_shift(x[None, :], p_pts, p_w, tau)[0]
    sol = root(f, seed, method="hybr", tol=1e-13)
    if sol.success and norm(f(sol.x)) < 1e-11:
        return sol.x
    return None


# ---------------- invariant tests (audit R0.5) ----------------

def invariants(run):
    rng = np.random.default_rng(MASTER)
    d, m, tau = 2, 5, 0.8
    pts = rng.normal(size=(m, d))
    w = rng.dirichlet(np.ones(m))
    X = rng.normal(size=(7, d)) * 2
    ok = []
    # V(p,p) = 0
    ok.append(("V(p,p)=0",
               norm(drift(X, pts, w, pts, w, tau)) < 1e-14))
    # translation equivariance of m
    t = rng.normal(size=d)
    ok.append(("translation",
               norm(mean_shift(X + t, pts + t, w, tau)
                    - mean_shift(X, pts, w, tau)) < 1e-12))
    # duplicated atom with split weight = same law
    pts2 = np.vstack([pts, pts[:1]])
    w2 = np.concatenate([w, [0.0]])
    w2[0] /= 2; w2[-1] = w2[0]
    ok.append(("atom-split",
               norm(mean_shift(X, pts2, w2, tau)
                    - mean_shift(X, pts, w, tau)) < 1e-12))
    # weights positive, sum 1
    ok.append(("weights", abs(w.sum() - 1) < 1e-12 and (w > 0).all()))
    for name, good in ok:
        run.log(f"  invariant {name}: {'PASS' if good else 'FAIL'}")
    assert all(g for _, g in ok)


# ---------------- P2A: linearization-scale study ----------------

def P2A():
    run = Run("P2A-linearization")
    run.log("P2A: fission-law relative error vs perturbation radius")
    invariants(run)
    rng = np.random.default_rng(MASTER + 21)
    d, L = 2, 5.0
    A = rng.normal(size=(3, d)) * 0.3
    B = rng.normal(size=(3, d)) * 0.3; B[:, 0] += L
    pts, w = np.vstack([A, B]), np.full(6, 1 / 6)
    tau = 1.0
    c = find_zero_near(pts, w, tau, np.zeros(d))
    f = lambda x: mean_shift(x[None, :], pts, w, tau)[0]
    A_pred = np.eye(d) + jac_fd(f, c, 1e-6 * tau)
    rows = []
    for expo in range(2, 9):
        radius = 10.0 ** (-expo) * tau
        errs = []
        for _ in range(8):
            u = rng.normal(size=d); u *= radius / norm(u)
            q_pts = np.vstack([c + u, c - u]); q_w = np.array([.5, .5])
            V = drift(q_pts, pts, w, q_pts, q_w, tau)
            errs.append(norm((V[0] - V[1]) / 2 - A_pred @ u) / norm(u))
        rows.append((radius, max(errs)))
        run.log(f"  radius=1e-{expo}*tau: max rel err = {max(errs):.3e}")
    rr = np.array(rows)
    mask = rr[:, 1] > 1e-12          # above roundoff
    slope = np.polyfit(np.log(rr[mask, 0]), np.log(rr[mask, 1]), 1)[0]
    run.log(f"  log-log slope over pre-roundoff radii: {slope:.3f} "
            "(first-order remainder predicts ~1)")
    run.save_csv("p2a_linerr.csv", ["radius", "max_rel_err"], rows)
    run.finish()


# ---------------- P2B: adversarial splitting search v2 ----------------

def sigma_multi_scale(pts, w, tau, c):
    """Splitting index at three FD scales + Jacobian conditioning."""
    f = lambda x: mean_shift(x[None, :], pts, w, tau)[0]
    sigs, smin = [], np.inf
    for h in (1e-5, 1e-6, 1e-7):
        Dm = jac_fd(f, c, h * tau)
        A = np.eye(len(c)) + Dm
        sigs.append(np.max(eigvals(A).real))
        smin = min(smin, svd(A, compute_uv=False)[-1])
    return sigs, smin


def P2B():
    run = Run("P2B-adversarial")
    run.log("P2B: symmetry-quotiented adversarial minimization of sigma")
    run.log("  parameterization: zero pinned at origin (penalty), first atom"
            " on +x axis, positions in ball 6*tau, weight floor 0.04")
    tau = 1.0
    results = []
    for d, m_atoms, n_de in ((2, 4, 3), (3, 5, 2)):
        nn = m_atoms * d - (d - 1)      # first atom: only its x-coord free
        bounds = [(-6, 6)] * nn + [(-3, 3)] * m_atoms

        def unpack(v):
            first = np.zeros(d); first[0] = abs(v[0])
            rest = np.array(v[1:nn]).reshape(m_atoms - 1, d)
            pts = np.vstack([first, rest])
            w = np.exp(v[nn:]); w = w / w.sum()
            w = 0.96 * w + 0.04 / m_atoms
            return pts, w

        def obj(v):
            pts, w = unpack(v)
            f = lambda x: mean_shift(x[None, :], pts, w, tau)[0]
            pen = norm(f(np.zeros(d))) / tau
            Dm = jac_fd(f, np.zeros(d), 1e-6)
            sig = np.max(eigvals(np.eye(d) + Dm).real)
            return sig + 80.0 * pen

        best = np.inf
        best_rec = None
        # global: differential evolution
        for k in range(n_de):
            de = differential_evolution(
                obj, bounds, seed=MASTER + 40 + k, maxiter=60,
                popsize=18, tol=1e-10, polish=True)
            cands = [de.x]
            for r in range(40):
                rr = np.random.default_rng(MASTER + 500 + 100 * k + r)
                x0 = np.array([rr.uniform(lo, hi) for lo, hi in bounds])
                nm = minimize(obj, x0, method="Nelder-Mead",
                              options={"maxiter": 2500})
                cands.append(nm.x)
            for v in cands:
                pts, w = unpack(v)
                z = find_zero_near(pts, w, tau, np.zeros(d))
                if z is None or norm(z) > 3:
                    continue
                sigs, smin = sigma_multi_scale(pts, w, tau, z)
                spread = max(sigs) - min(sigs)
                if min(sigs) < best:
                    best = min(sigs)
                    best_rec = (d, min(sigs), max(sigs), spread, smin)
        if best_rec:
            d_, s_lo, s_hi, spr, smin = best_rec
            run.log(f"  d={d_}: adversarial min sigma = {s_lo:+.5f} "
                    f"(FD-scale spread {spr:.1e}, min sing val {smin:.2e})")
            results.append(best_rec)
    run.save_csv("p2b_adversarial.csv",
                 ["d", "sigma_min", "sigma_max_scale", "fd_spread",
                  "min_singval"], results)
    run.log("  verdict: " + (
        "NO negative splitting index found (dichotomy conjecture survives)"
        if all(r[1] > 0 for r in results) else
        "NEGATIVE sigma found - stable collapse candidate, investigate!"))
    run.finish()


# ---------------- P2C: metastability v2 ----------------

def relax_time(p_pts, p_w, q0, q_w, tau, eta, mid, alpha0, cap_steps,
               taus_mix=None):
    q = q0.copy()
    for t in range(cap_steps):
        if taus_mix is None:
            V = drift(q, p_pts, p_w, q, q_w, tau)
        else:
            V = sum(drift(q, p_pts, p_w, q, q_w, tt)
                    for tt in taus_mix) / len(taus_mix)
        q = q + eta * V
        alpha = (q[:, 0] < mid).mean()
        if abs(alpha - 0.5) < 0.5 * abs(alpha0 - 0.5):
            return t * eta, False
    return cap_steps * eta, True


def P2C():
    run = Run("P2C-metastability")
    run.log("P2C: scale-consistent metastability; censoring-aware fit;")
    run.log("     L/tau control test; eta control; two-bandwidth rescue")
    d, N, alpha0 = 1, 40, 0.2
    rows = []
    for L in (5.0, 8.0):
        for ratio in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0):
            tau = L / ratio
            sig = 0.25 * tau               # scale-consistent cluster width
            eta = 0.1 * tau
            for rep in range(3):
                rng = np.random.default_rng(MASTER + 700 + rep)
                A = rng.normal(size=(6, d)) * sig
                B = rng.normal(size=(6, d)) * sig + L * np.eye(1, d)[0]
                p_pts, p_w = np.vstack([A, B]), np.full(12, 1 / 12)
                nA = int(alpha0 * N)
                q0 = np.vstack([rng.normal(size=(nA, d)) * sig,
                                rng.normal(size=(N - nA, d)) * sig
                                + L * np.eye(1, d)[0]])
                T, cens = relax_time(p_pts, p_w, q0, np.full(N, 1 / N),
                                     tau, eta, L / 2, alpha0, 150000)
                # report in units of tau (scale-invariant flow time)
                rows.append((L, ratio, rep, T / tau, int(cens)))
    run.save_csv("p2c_relax.csv",
                 ["L", "L_over_tau", "rep", "T_over_tau", "censored"], rows)
    arr = np.array(rows)
    for L in (5.0, 8.0):
        sel = (arr[:, 0] == L) & (arr[:, 4] == 0)
        x, y = arr[sel, 1], np.log(arr[sel, 3])
        cens = int(((arr[:, 0] == L) & (arr[:, 4] == 1)).sum())
        if sel.sum() >= 4:
            c1 = np.polyfit(x, y, 1)
            # competing model: log T = a + c x + p log x
            X2 = np.vstack([np.ones_like(x), x, np.log(x)]).T
            beta, res2, *_ = np.linalg.lstsq(X2, y, rcond=None)
            res1 = ((np.polyval(c1, x) - y) ** 2).sum()
            n = len(x)
            aic1 = n * np.log(res1 / n) + 2 * 2
            aic2 = n * np.log(res2[0] / n) + 2 * 3 if len(res2) else np.nan
            run.log(f"  L={L}: exp-law slope c={c1[0]:.3f} "
                    f"(censored cells: {cens}); AIC pure-exp={aic1:.1f} "
                    f"vs poly*exp={aic2:.1f} (poly coeff {beta[2]:+.2f})")
    # same L/tau via different L: compare T/tau at ratio 4.0
    for ratio in (3.0, 4.0):
        vals = [arr[(arr[:, 0] == L) & (arr[:, 1] == ratio)
                    & (arr[:, 4] == 0), 3] for L in (5.0, 8.0)]
        ms = [f"L={L}: {np.median(v):.1f}" if len(v) else f"L={L}: censored"
              for L, v in zip((5.0, 8.0), vals)]
        run.log(f"  L/tau={ratio} median T/tau -> " + "; ".join(ms) +
                "  (agreement = ratio controls)")
    # eta control at one cell
    L, ratio = 5.0, 4.0
    tau = L / ratio; sig = 0.25 * tau
    rng = np.random.default_rng(MASTER + 700)
    A = rng.normal(size=(6, d)) * sig
    B = rng.normal(size=(6, d)) * sig + L * np.eye(1, d)[0]
    p_pts, p_w = np.vstack([A, B]), np.full(12, 1 / 12)
    nA = int(alpha0 * N)
    q0 = np.vstack([rng.normal(size=(nA, d)) * sig,
                    rng.normal(size=(N - nA, d)) * sig + L * np.eye(1, d)[0]])
    for eta_f in (0.1, 0.05):
        T, cens = relax_time(p_pts, p_w, q0, np.full(N, 1 / N), tau,
                             eta_f * tau, L / 2, alpha0, 300000)
        run.log(f"  eta={eta_f}*tau at L/tau=4: T/tau="
                f"{T/tau:.1f}{' (censored)' if cens else ''}")
    # fixed absolute sigma cell (isolates sigma/tau)
    sig_abs = 0.3
    rngc = np.random.default_rng(MASTER + 700)
    A = rngc.normal(size=(6, d)) * sig_abs
    B = rngc.normal(size=(6, d)) * sig_abs + L * np.eye(1, d)[0]
    pp, pw = np.vstack([A, B]), np.full(12, 1 / 12)
    q0c = np.vstack([rngc.normal(size=(nA, d)) * sig_abs,
                     rngc.normal(size=(N - nA, d)) * sig_abs
                     + L * np.eye(1, d)[0]])
    T, cens = relax_time(pp, pw, q0c, np.full(N, 1 / N), tau, 0.1 * tau,
                         L / 2, alpha0, 150000)
    run.log(f"  fixed-abs sigma=0.3 at L/tau=4: T/tau="
            f"{T/tau:.1f}{' (censored)' if cens else ''} "
            "(compare scale-consistent cell)")
    # two-bandwidth rescue
    run.log("  two-bandwidth intervention V = (V_tau + V_tau')/2:")
    for ratio in (4.0, 5.0):
        tau = L / ratio; sig = 0.25 * tau
        rng = np.random.default_rng(MASTER + 700)
        A = rng.normal(size=(6, d)) * sig
        B = rng.normal(size=(6, d)) * sig + L * np.eye(1, d)[0]
        p_pts, p_w = np.vstack([A, B]), np.full(12, 1 / 12)
        q0 = np.vstack([rng.normal(size=(nA, d)) * sig,
                        rng.normal(size=(N - nA, d)) * sig
                        + L * np.eye(1, d)[0]])
        T1, c1 = relax_time(p_pts, p_w, q0, np.full(N, 1 / N), tau,
                            0.1 * tau, L / 2, alpha0, 150000)
        for tp in (L / 2, L):
            T2, c2 = relax_time(p_pts, p_w, q0, np.full(N, 1 / N), tau,
                                0.1 * tau, L / 2, alpha0, 150000,
                                taus_mix=(tau, tp))
            run.log(f"    L/tau={ratio}: single T/tau="
                    f"{T1/tau:.1f}{'(cens)' if c1 else ''}  "
                    f"mix(tau'={tp:.1f}) T/tau="
                    f"{T2/tau:.1f}{'(cens)' if c2 else ''}  "
                    f"speedup x{T1/max(T2,1e-9):.1f}")
    run.finish()


# ---------------- P2D: flow-level Lyapunov ----------------

def grad_mmd2_q(p_pts, p_w, q_pts, q_w, tau):
    """d/dx_j MMD^2(p, q) for the Laplace kernel (a.e. formula)."""
    def gk(A, B):                      # grad_1 k(a,b) summed vs B-weights
        D = A[:, None, :] - B[None, :, :]
        r = norm(D, axis=2)
        r = np.where(r < 1e-12, np.inf, r)
        return -(np.exp(-np.where(np.isinf(r), 0, r) / tau)
                 * (1 / tau))[:, :, None] * D / r[:, :, None]
    gqq = (q_w[None, :, None] * gk(q_pts, q_pts)).sum(axis=1)
    gqp = (p_w[None, :, None] * gk(q_pts, p_pts)).sum(axis=1)
    return 2 * q_w[:, None] * (gqq - gqp)


def grad_ed_q(p_pts, p_w, q_pts, q_w):
    """d/dx_j ED(p,q) (squared-form energy distance)."""
    def gu(A, B):
        D = A[:, None, :] - B[None, :, :]
        r = norm(D, axis=2)
        r = np.where(r < 1e-12, np.inf, r)
        return D / r[:, :, None]
    gqp = (p_w[None, :, None] * gu(q_pts, p_pts)).sum(axis=1)
    gqq = (q_w[None, :, None] * gu(q_pts, q_pts)).sum(axis=1)
    return 2 * q_w[:, None] * (gqp - gqq)


def P2D():
    run = Run("P2D-lyapunov-flow")
    run.log("P2D: orbital derivatives grad F . V at flow level")
    d, L = 1, 5.0
    rng0 = np.random.default_rng(MASTER + 10)
    A = rng0.normal(size=(3, d)) * 0.3
    B = rng0.normal(size=(3, d)) * 0.3; B[:, 0] += L
    p_pts, p_w = np.vstack([A, B]), np.full(6, 1 / 6)
    tot = 0
    pos = {"MMD2": 0, "ED": 0}
    witnesses = []
    for rep in range(20):
        rr = np.random.default_rng(MASTER + 800 + rep)
        N = 24
        q = rr.normal(size=(N, d)) * 3.0 + p_pts.mean(axis=0)
        q_w = np.full(N, 1 / N)
        tau = float(rr.choice([1.0, 2.5]))
        for t in range(600):
            V = drift(q, p_pts, p_w, q, q_w, tau)
            if t % 5 == 0:
                tot += 1
                dm = (grad_mmd2_q(p_pts, p_w, q, q_w, tau) * V).sum()
                de = (grad_ed_q(p_pts, p_w, q, q_w) * V).sum()
                if dm > 1e-12:
                    pos["MMD2"] += 1
                    witnesses.append(("MMD2", rep, t, dm))
                if de > 1e-12:
                    pos["ED"] += 1
                    witnesses.append(("ED", rep, t, de))
            q = q + 0.1 * tau * V
    for k in pos:
        run.log(f"  {k}: orbital derivative > 0 at {pos[k]}/{tot} states "
                f"({100*pos[k]/tot:.1f}%)")
    if witnesses:
        w0 = witnesses[0]
        run.log(f"  first witness: {w0[0]} rep={w0[1]} t={w0[2]} "
                f"dF/dt={w0[3]:+.3e}")
        run.log("  -> genuine FLOW-level violations "
                "(not Euler overshoot): these functionals are not Lyapunov "
                "for the continuous population flow")
    else:
        run.log("  -> NO flow-level violations: pass-1 E6 increases were "
                "Euler overshoot; MMD2/ED are candidate Lyapunov functionals!")
    run.save_csv("p2d_witnesses.csv", ["cand", "rep", "t", "dFdt"],
                 witnesses[:200])
    run.finish()


# ---------------- P2E: generator spectrum + eta* ----------------

def P2E():
    run = Run("P2E-truth-stability")
    run.log("P2E: continuous generator at q=p, Euler boundary, self-mask")
    d, L = 2, 5.0
    rng0 = np.random.default_rng(MASTER + 10)
    A = rng0.normal(size=(3, d)) * 0.3
    B = rng0.normal(size=(3, d)) * 0.3; B[:, 0] += L
    p_pts, p_w = np.vstack([A, B]), np.full(6, 1 / 6)
    N = len(p_pts)
    for tau in (1.0, 2.5):
        for masked in (False, True):
            def field(v):
                qq = v.reshape(N, d)
                if masked:
                    return drift_masked(qq, p_pts, p_w, p_w, tau).ravel()
                return drift(qq, p_pts, p_w, qq, p_w, tau).ravel()

            v0 = p_pts.ravel()
            h = 1e-7
            G = np.zeros((N * d, N * d))
            for i in range(N * d):
                e = np.zeros(N * d); e[i] = h
                G[:, i] = (field(v0 + e) - field(v0 - e)) / (2 * h)
            ev = eigvals(G)
            stab = ev[ev.real < -1e-12]
            eta_pred = np.min(-2 * stab.real / np.abs(stab) ** 2) \
                if len(stab) else np.inf
            # bisection on the actual map spectral radius
            def rho(eta):
                return np.abs(eigvals(np.eye(N * d) + eta * G)).max()
            lo_, hi_ = 0.01 * tau, 5 * tau
            if rho(hi_) <= 1 + 1e-12:
                eta_star = np.inf
            else:
                for _ in range(40):
                    mid = (lo_ + hi_) / 2
                    if rho(mid) <= 1 + 1e-12:
                        lo_ = mid
                    else:
                        hi_ = mid
                eta_star = lo_
            run.log(f"  tau={tau} masked={masked}: max Re eig(G)="
                    f"{ev.real.max():+.3e}, predicted eta*="
                    f"{eta_pred/tau:.3f}*tau, bisected eta*="
                    f"{eta_star/tau:.3f}*tau")
    run.finish()


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = {"P2A": P2A, "P2B": P2B, "P2C": P2C, "P2D": P2D, "P2E": P2E}
    for name, fn in fns.items():
        if which in ("all", name):
            fn()
