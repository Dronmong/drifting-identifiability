"""Fourth Rn-screen pass (2026-07-16): clean, optimizer-free verification that
BOTH monotonicity conjectures (full-matrix section 4.6(e) AND the directional
refinement along m-hat) are unboundedly false in n >= 2.

Construction A (transverse, kills full-matrix form):
  atom 1 at probe + delta*e1 (delta=0.01), atom 2 at distance L in direction
  u2 = (c, sqrt(1-c^2)); masses tuned so effective kernel weights at the probe
  are 1/2 each.  Then I + Dm = (1/tau) Cov_w(X, u) = (L/4) u2 (x) (u2 - u1)
  + O(delta), whose symmetric form at v = e1 is (L c / 4)(c - 1)/tau -> -inf.
  Along m-hat = u2 the value is +(L/4)(1-c) > 0 (2 atoms never violate
  directionally).

Construction B (directional, kills the flow-line refinement):
  atom 0 at probe + delta*e1; a symmetric far pair at L*(eps, +-sqrt(1-eps^2))
  with masses tuned to effective weights (p1, p2/2, p2/2).  Transverse first
  moments cancel, m-hat = e1, and
      1 + <m_hat, Dm m_hat> = 1 - p1 p2 L eps (1-eps)/tau + O(delta, angle^2).
  Grows linearly in L: unbounded violation ALONG the drift direction.

Run:  uv run --with numpy python numerics/rn_screen4.py
"""

from __future__ import annotations

import numpy as np

TAU = 1.0


def m_stable(x: np.ndarray, ys: np.ndarray, a: np.ndarray) -> np.ndarray:
    diff = ys[None, :, :] - x[:, None, :]
    d = np.linalg.norm(diff, axis=2)
    d0 = d.min(axis=1, keepdims=True)
    w = a[None, :] * np.exp(-(d - d0) / TAU)
    z = w.sum(axis=1)
    return (w[:, :, None] * diff).sum(axis=1) / z[:, None]


def sym_jac_eigmin_and_directional(probe, ys, a, n, h=1e-6):
    jac = np.zeros((n, n))
    for c in range(n):
        e = np.zeros(n)
        e[c] = h
        mp = m_stable((probe + e)[None, :], ys, a)[0]
        mm = m_stable((probe - e)[None, :], ys, a)[0]
        jac[:, c] = (mp - mm) / (2 * h)
    sym = np.eye(n) + 0.5 * (jac + jac.T)
    m0 = m_stable(probe[None, :], ys, a)[0]
    mh = m0 / np.linalg.norm(m0)
    return float(np.linalg.eigvalsh(sym).min()), float(mh @ sym @ mh)


def construction_a() -> None:
    print("[A] transverse violation, 2 atoms, n=2 (prediction (Lc/4)(c-1)):")
    delta, c = 0.01, 0.5
    u2 = np.array([c, np.sqrt(1 - c * c)])
    for L in (10.0, 30.0, 100.0, 300.0):
        ys = np.stack([np.array([delta, 0.0]), L * u2])
        a = np.array([np.exp(-(L - delta) / TAU), 1.0])
        a = a / a.sum()
        eigmin, dirv = sym_jac_eigmin_and_directional(np.zeros(2), ys, a, 2)
        pred = (L * c / 4.0) * (c - 1.0) / TAU
        print(f"    L={L:6.0f}: eig_min = {eigmin:+9.3f} (pred {pred:+9.3f}); "
              f"directional = {dirv:+9.3f} (pred {L * (1 - c) / 4:+9.3f})")


def construction_b() -> None:
    print("[B] DIRECTIONAL violation, 3 atoms, n=2 "
          "(prediction 1 - p1 p2 L eps (1-eps)):")
    delta, eps = 0.01, 0.5
    p1 = 0.5
    up = np.array([eps, np.sqrt(1 - eps * eps)])
    um = np.array([eps, -np.sqrt(1 - eps * eps)])
    for L in (10.0, 30.0, 100.0, 300.0):
        ys = np.stack([np.array([delta, 0.0]), L * up, L * um])
        # effective weights (p1, (1-p1)/2, (1-p1)/2) at the probe:
        # a_i proportional to p_i * exp(+d_i/tau); rescale by exp(-L/tau)
        a = np.array([p1 * np.exp((delta - L) / TAU),
                      (1 - p1) / 2, (1 - p1) / 2])
        a = a / a.sum()
        eigmin, dirv = sym_jac_eigmin_and_directional(np.zeros(2), ys, a, 2)
        pred = -p1 * (1 - p1) * L * eps * (1 - eps) / TAU
        print(f"    L={L:6.0f}: directional = {dirv:+9.3f} (pred {pred:+9.3f}); "
              f"eig_min = {eigmin:+9.3f}")


if __name__ == "__main__":
    print("Rn screen 4: optimizer-free unboundedness verification (tau=1)")
    construction_a()
    construction_b()
