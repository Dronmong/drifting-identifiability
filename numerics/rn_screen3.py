"""Third Rn-screen pass (2026-07-16): the DIRECTIONAL monotonicity conjecture.

Analysis behind this script (recorded in LaplaceRnRoadmap.md):
  sym(I + Dm_p)(x) = (1/tau) * sym Cov_w(X, X/|X|),  X = y - x,
so the full-matrix centroid monotonicity of section 4.6(e) is UNBOUNDEDLY
false in n >= 2: with atom 1 at the probe and atom 2 at distance d in
direction u2, masses tuned so both effective weights are 1/2, the quadratic
form along v with c = <u2,v> is (d*c/4)(c-1)/tau -> -infinity.  The violating
direction is TRANSVERSE to m; along m-hat the same example is +(d/4)(1-c) >= 0.

NEW CONJECTURE tested here (directional centroid monotonicity):
    1 + <m_hat, Dm_p(x) m_hat>  >=  0   at every x with m(x) != 0,
equivalently <m_hat, Cov_w(X, X/|X|) m_hat> >= -tau.  In 1-d this IS the
proved m'+1 >= 0.  Along rays for radial measures it is the (r + m_tilde)' >= 0
radial form.  It is exactly the coefficient a flow-line (characteristic)
propagation along m would consume.

Also verifies the explicit transverse counterexample formula numerically.

Run:  uv run --with numpy --with scipy python numerics/rn_screen3.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

RNG = np.random.default_rng(20260718)
TAU = 1.0
BOX = 40.0


def z_d_m_stable(x: np.ndarray, ys: np.ndarray, a: np.ndarray):
    """Stabilized mean shift: weights shifted by the min distance (m invariant)."""
    diff = ys[None, :, :] - x[:, None, :]
    d = np.linalg.norm(diff, axis=2)
    d0 = d.min(axis=1, keepdims=True)
    w = a[None, :] * np.exp(-(d - d0) / TAU)
    z = w.sum(axis=1)
    dnum = (w[:, :, None] * diff).sum(axis=1)
    return dnum / z[:, None]


def directional_value(probe: np.ndarray, ys: np.ndarray, a: np.ndarray,
                      n: int) -> float:
    """1 + <m_hat, Dm m_hat> at probe via central differences along m_hat."""
    m0 = z_d_m_stable(probe[None, :], ys, a)[0]
    nm = np.linalg.norm(m0)
    if nm < 1e-9:
        return np.inf  # zero of m: conjecture not asserted there
    mh = m0 / nm
    h = 1e-5
    mp = z_d_m_stable((probe + h * mh)[None, :], ys, a)[0]
    mm = z_d_m_stable((probe - h * mh)[None, :], ys, a)[0]
    return 1.0 + float(mh @ (mp - mm)) / (2 * h)


def random_scan(n: int, trials: int) -> float:
    worst = np.inf
    for _ in range(trials):
        n_atoms = int(RNG.integers(1, 8))
        scale = float(np.exp(RNG.uniform(np.log(0.1), np.log(20.0))))
        ys = RNG.normal(0.0, scale, size=(n_atoms, n))
        raw = RNG.uniform(-12.0, 2.0, n_atoms)  # heavy-tailed masses
        a = np.exp(raw)
        a = a / a.sum()
        for _ in range(40):
            probe = RNG.normal(0.0, 1.5 * scale + 2.0, n)
            worst = min(worst, directional_value(probe, ys, a, n))
    return worst


def adversarial(n: int, starts: int) -> float:
    def objective(params, n_atoms):
        ys = np.clip(params[: n_atoms * n].reshape(n_atoms, n), -BOX, BOX)
        logits = np.clip(params[n_atoms * n: n_atoms * n + n_atoms], -30, 30)
        probe = np.clip(params[n_atoms * n + n_atoms:], -BOX, BOX)
        a = np.exp(logits - logits.max())
        a = a / a.sum()
        v = directional_value(probe, ys, a, n)
        return v if np.isfinite(v) else 10.0

    worst = np.inf
    for _ in range(starts):
        n_atoms = int(RNG.integers(2, 7))
        scale = float(np.exp(RNG.uniform(np.log(0.3), np.log(15.0))))
        ys = RNG.normal(0.0, scale, size=(n_atoms, n))
        p0 = np.concatenate([ys.ravel(), RNG.uniform(-8, 2, n_atoms),
                             RNG.normal(0, scale, n)])
        res = minimize(objective, p0, args=(n_atoms,), method="Nelder-Mead",
                       options={"maxiter": 6000, "fatol": 1e-12, "xatol": 1e-9})
        worst = min(worst, res.fun)
    return worst


def transverse_counterexample_check() -> None:
    """Verify sym(I+Dm) transverse blowup ~ (d*c/4)(c-1)/tau and the positive
    directional value on the SAME configuration."""
    print("[EXP3] explicit 2-atom transverse counterexample (n=2):")
    for d in (10.0, 30.0, 100.0):
        c = 0.5
        u2 = np.array([c, np.sqrt(1 - c * c)])
        ys = np.stack([np.array([1e-6, 0.0]), d * u2])
        # tune masses so effective weights at the origin are 1/2 each
        a = np.array([np.exp(-d / TAU), 1.0])
        a = a / a.sum()
        probe = np.zeros(2)
        h = 1e-5
        jac = np.zeros((2, 2))
        for cidx in range(2):
            e = np.zeros(2)
            e[cidx] = h
            mp = z_d_m_stable((probe + e)[None, :], ys, a)[0]
            mm = z_d_m_stable((probe - e)[None, :], ys, a)[0]
            jac[:, cidx] = (mp - mm) / (2 * h)
        sym = np.eye(2) + 0.5 * (jac + jac.T)
        eigs = np.linalg.eigvalsh(sym)
        predicted = (d * c / 4.0) * (c - 1.0) / TAU
        dval = directional_value(probe, ys, a, 2)
        print(f"    d={d:6.1f}: min eig sym(I+Dm) = {eigs[0]:+10.3f} "
              f"(transverse prediction ~ {predicted:+10.3f}); "
              f"directional value along m = {dval:+8.4f}")


def main() -> None:
    print("Rn screen 3: directional centroid monotonicity (tau=1, seed 20260718)")
    transverse_counterexample_check()
    for n in (2, 3):
        w = random_scan(n, trials=1500)
        print(f"[EXP4rand n={n}] min of 1 + <m_hat, Dm m_hat> over ~60k "
              f"random (config, probe): {w:+.6f}")
    for n in (2, 3):
        w = adversarial(n, starts=50)
        print(f"[EXP4adv  n={n}] adversarial min of 1 + <m_hat, Dm m_hat>: "
              f"{w:+.6f} over 50 optimized starts")


if __name__ == "__main__":
    main()
