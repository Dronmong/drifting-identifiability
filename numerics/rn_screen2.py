"""Follow-up to rn_screen.py (2026-07-16): sharpened EXP1b + adversarial EXP2.

EXP1b': nonlinear residual floor reported ONLY over runs whose optimized q is
genuinely distinct from p (matched-atom gap > 1e-3), separating true floors
from optimizer collapses onto p.

EXP2-adv: adversarial search for the WORST violation of the centroid
monotonicity lambda_min(sym(I + Dm_p)) — maximize the violation over atom
positions, weights, and the probe point (L-BFGS from many random starts).
rn_screen.py already falsified the conjecture (min eig ~ -0.16 random); the
question here is whether the violation is BOUNDED (a weaker uniform bound
sym(I+Dm) >= -c could still feed a propagation argument) or unbounded.

Run:  uv run --with numpy --with scipy python numerics/rn_screen2.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

RNG = np.random.default_rng(20260717)
TAU = 1.0


def z_d_m(x: np.ndarray, ys: np.ndarray, a: np.ndarray):
    diff = ys[None, :, :] - x[:, None, :]
    d = np.linalg.norm(diff, axis=2)
    w = a[None, :] * np.exp(-d / TAU)
    z = w.sum(axis=1)
    dnum = (w[:, :, None] * diff).sum(axis=1)
    return z, dnum, dnum / z[:, None]


def probe_cloud(n: int, scale: float, count: int) -> np.ndarray:
    near = RNG.normal(0.0, scale, size=(count // 2, n))
    mid = RNG.normal(0.0, 2.0 * scale + 2.0, size=(count // 4, n))
    far = RNG.normal(0.0, 4.0 * scale + 6.0, size=(count - count // 2 - count // 4, n))
    return np.vstack([near, mid, far])


# --------------------------------------------------------------- EXP 1b' ----
def residual(params, n, m_atoms, ys_p, a_p, x):
    ys_q = params[: m_atoms * n].reshape(m_atoms, n)
    logits = params[m_atoms * n:]
    a_q = np.exp(logits - logits.max())
    a_q = a_q / a_q.sum()
    _, _, mp = z_d_m(x, ys_p, a_p)
    _, _, mq = z_d_m(x, ys_q, a_q)
    return float(np.sum((mp - mq) ** 2))


def nonlinear_floor_sharp(n: int, trials: int):
    floor_distinct = np.inf
    max_gap_small_resid = 0.0
    for _ in range(trials):
        n_atoms = int(RNG.integers(2, 5))
        m_atoms = n_atoms + int(RNG.integers(0, 2))
        scale = float(np.exp(RNG.uniform(np.log(0.3), np.log(3.0))))
        ys_p = RNG.normal(0.0, scale, size=(n_atoms, n))
        a_p = RNG.dirichlet(np.ones(n_atoms))
        x = probe_cloud(n, scale, 160)
        ys0 = ys_p[RNG.integers(0, n_atoms, size=m_atoms)] + \
            RNG.normal(0, 0.5 * scale, size=(m_atoms, n))
        p0 = np.concatenate([ys0.ravel(), RNG.normal(0, 0.5, m_atoms)])
        res = minimize(residual, p0, args=(n, m_atoms, ys_p, a_p, x),
                       method="L-BFGS-B",
                       options={"maxiter": 1500, "ftol": 1e-18, "gtol": 1e-14})
        ys_q = res.x[: m_atoms * n].reshape(m_atoms, n)
        logits = res.x[m_atoms * n:]
        a_q = np.exp(logits - logits.max())
        a_q = a_q / a_q.sum()
        gap = 0.0
        for j in range(m_atoms):
            dj = np.linalg.norm(ys_p - ys_q[j], axis=1).min()
            gap += a_q[j] * dj
        if gap > 1e-3:
            floor_distinct = min(floor_distinct, res.fun)
        elif res.fun < 1e-10:
            max_gap_small_resid = max(max_gap_small_resid, gap)
    return floor_distinct, max_gap_small_resid


# -------------------------------------------------------------- EXP2-adv ----
def violation(params: np.ndarray, n: int, n_atoms: int) -> float:
    """Negative of lambda_min(sym(I+Dm)) at the probe -> minimize returns worst."""
    ys = params[: n_atoms * n].reshape(n_atoms, n)
    logits = params[n_atoms * n: n_atoms * n + n_atoms]
    probe = params[n_atoms * n + n_atoms:]
    a = np.exp(logits - logits.max())
    a = a / a.sum()
    h = 1e-5
    jac = np.zeros((n, n))
    for c in range(n):
        e = np.zeros(n)
        e[c] = h
        _, _, mp = z_d_m((probe + e)[None, :], ys, a)
        _, _, mm = z_d_m((probe - e)[None, :], ys, a)
        jac[:, c] = (mp[0] - mm[0]) / (2 * h)
    sym = np.eye(n) + 0.5 * (jac + jac.T)
    return float(np.linalg.eigvalsh(sym).min())


def adversarial_monotonicity(n: int, starts: int) -> float:
    worst = np.inf
    for _ in range(starts):
        n_atoms = int(RNG.integers(2, 7))
        scale = float(np.exp(RNG.uniform(np.log(0.2), np.log(5.0))))
        ys = RNG.normal(0.0, scale, size=(n_atoms, n))
        p0 = np.concatenate([
            ys.ravel(),
            RNG.normal(0, 0.5, n_atoms),
            RNG.normal(0, scale, n),
        ])
        res = minimize(violation, p0, args=(n, n_atoms), method="Nelder-Mead",
                       options={"maxiter": 4000, "fatol": 1e-10, "xatol": 1e-8})
        worst = min(worst, res.fun)
    return worst


def main() -> None:
    print("Rn screen follow-up (tau = 1, seed 20260717)")
    for n in (2, 3):
        floor, gap = nonlinear_floor_sharp(n, trials=60)
        print(f"[EXP1b' n={n}] residual floor over genuinely-distinct optima "
              f"(gap>1e-3): {floor:.3e}; max gap among residual<1e-10 runs: "
              f"{gap:.3e}")
    for n in (2, 3):
        worst = adversarial_monotonicity(n, starts=40)
        print(f"[EXP2adv n={n}] adversarial min eig sym(I+Dm) = {worst:+.4f} "
              f"over 40 optimized starts")


if __name__ == "__main__":
    main()
