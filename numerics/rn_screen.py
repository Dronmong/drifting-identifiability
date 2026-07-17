"""Rn-converse screening numerics (2026-07-16).

Two decision experiments for the general-Rn Laplace converse roadmap
(DriftingIdentifiability/LaplaceRnRoadmap.md):

  EXP 1  Counterexample screen for the full n-d conjecture on atomic measures
         (D2.c falsification pass called for in LaplaceHigherDim.md section 4.8
         "Numerics side-quests (ii)", never previously run):
         (a) linearized injectivity: at p = q atomic, the drift response
             operator T[h](x) = (1/Z_p(x)) * sum_j h_j k(x,y_j)(y_j - x - m_p(x))
             restricted to sum(h) = 0 must have trivial kernel; report the
             normalized smallest singular value over random configurations.
         (b) nonlinear residual descent: from random q != p, gradient-descend
             the probe-summed squared drift residual over q's weights AND
             positions; a floor >> 0 with q != p (and collapse to p when the
             optimizer finds it) supports the conjecture.

  EXP 2  The n-d centroid-monotonicity conjecture (section 4.6(e)):
         F(x) = x + m_p(x) monotone, i.e. <F(x)-F(x'), x-x'> >= 0, and its
         infinitesimal form lambda_min( sym(I + Dm_p) ) >= 0.  This is the sign
         control the general endgame and the RadialSlack removal both want.
         1-d: proved.  Radial n=3 slack scan (RsiScan.cs, 2026-07-15): the
         related RSI holds with margin.  Here: NON-radial atomic p in R^2/R^3.

tau = 1 throughout WLOG (configs scanned over scales 0.1..10 covers bandwidth).
Run:  uv run --with numpy --with scipy python numerics/rn_screen.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

RNG = np.random.default_rng(20260716)
TAU = 1.0


def z_d_m(x: np.ndarray, ys: np.ndarray, a: np.ndarray):
    """Normalizer, displacement numerator and mean shift at probes x.

    x: (P, n) probes; ys: (N, n) atoms; a: (N,) positive weights.
    Returns Z: (P,), D: (P, n), m: (P, n).
    """
    diff = ys[None, :, :] - x[:, None, :]            # (P, N, n)
    d = np.linalg.norm(diff, axis=2)                 # (P, N)
    w = a[None, :] * np.exp(-d / TAU)                # (P, N)
    z = w.sum(axis=1)                                # (P,)
    dnum = (w[:, :, None] * diff).sum(axis=1)        # (P, n)
    return z, dnum, dnum / z[:, None]


def random_config(n: int, n_atoms: int, scale: float):
    ys = RNG.normal(0.0, scale, size=(n_atoms, n))
    a = RNG.dirichlet(np.ones(n_atoms))
    return ys, a


def probe_cloud(n: int, scale: float, count: int) -> np.ndarray:
    """Probes at the config scale, plus a mid-field and a far ring."""
    near = RNG.normal(0.0, scale, size=(count // 2, n))
    mid = RNG.normal(0.0, 2.0 * scale + 2.0, size=(count // 4, n))
    far = RNG.normal(0.0, 4.0 * scale + 6.0, size=(count - count // 2 - count // 4, n))
    return np.vstack([near, mid, far])


# ---------------------------------------------------------------- EXP 1a ----
def linearized_min_sv(n: int, trials: int) -> tuple[float, float]:
    """Normalized smallest singular value of the linearized drift response.

    Perturbation h lives on the atom set of p with sum(h)=0 (signed measure,
    both laws stay probabilities to first order when h is split p+eh, q-eh —
    only the difference enters the linearization at p=q).
    """
    worst = np.inf
    worst_rel = np.inf
    for _ in range(trials):
        n_atoms = int(RNG.integers(2, 7))
        scale = float(np.exp(RNG.uniform(np.log(0.1), np.log(10.0))))
        ys, a = random_config(n, n_atoms, scale)
        x = probe_cloud(n, scale, 240)
        z, _, m = z_d_m(x, ys, a)
        diff = ys[None, :, :] - x[:, None, :]
        d = np.linalg.norm(diff, axis=2)
        k = np.exp(-d / TAU)
        # column j: (1/Z) k(x,y_j) (y_j - x - m(x)) stacked over probe coords
        cols = k[:, :, None] * (diff - m[:, None, :]) / z[:, None, None]
        mat = cols.transpose(0, 2, 1).reshape(-1, n_atoms)  # (P*n, N)
        # restrict to sum-zero h: basis e_j - e_N
        basis = np.vstack([np.eye(n_atoms - 1), -np.ones((1, n_atoms - 1))])
        sub = mat @ basis
        s = np.linalg.svd(sub, compute_uv=False)
        worst = min(worst, s[-1])
        worst_rel = min(worst_rel, s[-1] / s[0])
    return worst, worst_rel


# ---------------------------------------------------------------- EXP 1b ----
def residual(params: np.ndarray, n: int, m_atoms: int,
             ys_p: np.ndarray, a_p: np.ndarray, x: np.ndarray) -> float:
    ys_q = params[: m_atoms * n].reshape(m_atoms, n)
    logits = params[m_atoms * n:]
    a_q = np.exp(logits - logits.max())
    a_q = a_q / a_q.sum()
    _, _, mp = z_d_m(x, ys_p, a_p)
    _, _, mq = z_d_m(x, ys_q, a_q)
    return float(np.sum((mp - mq) ** 2))


def nonlinear_floor(n: int, trials: int) -> tuple[float, float, float]:
    """Descend the drift residual from random q != p.

    Returns (min residual with q far from p, max transport-ish distance at
    near-zero residual, fraction of runs that collapsed onto p).
    """
    floor_when_distinct = np.inf
    dist_at_zero = 0.0
    collapsed = 0
    for _ in range(trials):
        n_atoms = int(RNG.integers(2, 5))
        m_atoms = n_atoms + int(RNG.integers(0, 2))
        scale = float(np.exp(RNG.uniform(np.log(0.3), np.log(3.0))))
        ys_p, a_p = random_config(n, n_atoms, scale)
        x = probe_cloud(n, scale, 160)
        ys0 = ys_p[RNG.integers(0, n_atoms, size=m_atoms)] + \
            RNG.normal(0, 0.5 * scale, size=(m_atoms, n))
        p0 = np.concatenate([ys0.ravel(), RNG.normal(0, 0.5, m_atoms)])
        res = minimize(residual, p0, args=(n, m_atoms, ys_p, a_p, x),
                       method="L-BFGS-B",
                       options={"maxiter": 800, "ftol": 1e-16, "gtol": 1e-12})
        val = res.fun
        ys_q = res.x[: m_atoms * n].reshape(m_atoms, n)
        logits = res.x[m_atoms * n:]
        a_q = np.exp(logits - logits.max())
        a_q = a_q / a_q.sum()
        # crude distance of q to p: matched nearest-atom mass-weighted gap
        gap = 0.0
        for j in range(m_atoms):
            dj = np.linalg.norm(ys_p - ys_q[j], axis=1).min()
            gap += a_q[j] * dj
        if val < 1e-14:
            collapsed += 1
            dist_at_zero = max(dist_at_zero, gap)
        else:
            floor_when_distinct = min(floor_when_distinct, val)
    return floor_when_distinct, dist_at_zero, collapsed / trials


# ----------------------------------------------------------------- EXP 2 ----
def monotonicity_scan(n: int, trials: int) -> tuple[float, float]:
    """Min pair form <F(x)-F(x'),x-x'>/|x-x'|^2 and min sym-Jacobian eigenvalue."""
    min_pair = np.inf
    min_eig = np.inf
    h = 1e-5
    for _ in range(trials):
        n_atoms = int(RNG.integers(1, 7))
        scale = float(np.exp(RNG.uniform(np.log(0.1), np.log(10.0))))
        ys, a = random_config(n, n_atoms, scale)
        x = probe_cloud(n, scale, 120)
        _, _, m = z_d_m(x, ys, a)
        f = x + m
        # random probe pairs
        idx = RNG.integers(0, len(x), size=(400, 2))
        i, j = idx[:, 0], idx[:, 1]
        keep = i != j
        i, j = i[keep], j[keep]
        dx = x[i] - x[j]
        df = f[i] - f[j]
        vals = np.einsum("ij,ij->i", df, dx) / np.einsum("ij,ij->i", dx, dx)
        min_pair = min(min_pair, float(vals.min()))
        # symmetrized Jacobian at a few probes by central differences
        for probe in x[RNG.integers(0, len(x), size=8)]:
            jac = np.zeros((n, n))
            for c in range(n):
                e = np.zeros(n)
                e[c] = h
                _, _, mp = z_d_m((probe + e)[None, :], ys, a)
                _, _, mm = z_d_m((probe - e)[None, :], ys, a)
                jac[:, c] = (mp[0] - mm[0]) / (2 * h)
            sym = np.eye(n) + 0.5 * (jac + jac.T)
            min_eig = min(min_eig, float(np.linalg.eigvalsh(sym).min()))
    return min_pair, min_eig


def main() -> None:
    print("Rn screen (tau = 1, seed 20260716)")
    for n in (2, 3):
        sv, sv_rel = linearized_min_sv(n, trials=400)
        print(f"[EXP1a n={n}] linearized drift response on sum-zero atomic "
              f"perturbations: min sigma_min = {sv:.3e} "
              f"(relative to sigma_max: {sv_rel:.3e}) over 400 configs")
    for n in (2, 3):
        floor, dz, frac = nonlinear_floor(n, trials=60)
        print(f"[EXP1b n={n}] nonlinear residual descent: floor when q!=p = "
              f"{floor:.3e}; max q-to-p gap at residual<1e-14 = {dz:.3e}; "
              f"collapse fraction = {frac:.2f}")
    for n in (2, 3):
        mp, me = monotonicity_scan(n, trials=800)
        print(f"[EXP2  n={n}] centroid monotonicity: min pair quotient = "
              f"{mp:+.4f}; min eig sym(I+Dm) = {me:+.4f} over 800 configs")


if __name__ == "__main__":
    main()
