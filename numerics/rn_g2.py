"""G2 numerics (2026-07-17): can RadialSlackN be removed?

Decomposition under test (derived on paper, recorded in LaplaceRnRoadmap.md):
with the tilted shell-mixture written through the law of total covariance,

  tau*(m'+1) = Cov(X, X/d) = E_pi[Cov_within(s)] + Cov_pi(m(s), z(s)),

  m(s) = E[X | shell s],  z(s) = E[X/d | shell s]   (tilted zonal averages),
  pi(ds) proportional to Zbar(s) nu(ds).

Facts already provable per shell:
  (P1) s >= r  =>  Cov_within(s) >= 0   (X and X/d comonotone in u:
       d(X/d)/du = s^2 (s - r u)/d^3 >= 0 for s >= r).
  (P2) s < r   =>  m(s) < 0 <= r, and the per-shell AM-GM + (T) argument
       gives Cov_within(s) >= -(n-1)*tau*(r - |m(s)|)/(2r) >= -(n-1)tau/2.

So G2 (slack removal, i.e. Cov >= -(n-1)tau on the region m > r; numerics say
Cov >= 0 there) follows if the BETWEEN-shell term obeys
Cov_pi(m, z) >= -(n-1)tau/2 for every mixture — in particular if BOTH
s -> m(s) and s -> z(s) are nondecreasing (association: Cov_pi >= 0 for any
pi).  This script tests:

  [A] monotonicity of m(s) and z(s) in s, over r/tau in a wide grid and
      n in {3,4,5,10};
  [B] the within-shell floor Cov_within(s) >= -(n-1)tau/2 (and locates the
      worst shell);
  [C] adversarial two-shell mixtures: the full Cov + (n-1)tau margin, and
      the between-term alone.

tau = 1 WLOG.  Run: uv run --with numpy --with scipy python numerics/rn_g2.py
"""

from __future__ import annotations

import numpy as np

RNG = np.random.default_rng(20260717)
TAU = 1.0
NU = 4001  # u-quadrature points


def shell_stats(n: int, r: float, s: float):
    """Tilted zonal averages on one shell: returns (Zbar, m, z, q, cov_within).

    Zbar = integral of weight*K (unnormalized shell normalizer),
    m = E[X], z = E[X/d], q = E[X^2/d], cov = Cov(X, X/d), all tilted.
    """
    u = np.linspace(-1.0, 1.0, NU)
    w = (1.0 - u * u) ** ((n - 3) / 2.0)
    d = np.sqrt(np.maximum(r * r + s * s - 2 * r * s * u, 0.0))
    K = np.exp(-(d - abs(r - s)) / TAU)  # stabilized; cancels in averages
    x = s * u - r
    xod = np.divide(x, d, out=np.zeros_like(x), where=d > 1e-300)
    base = w * K
    zb = np.trapezoid(base, u)
    m = np.trapezoid(base * x, u) / zb
    z = np.trapezoid(base * xod, u) / zb
    q = np.trapezoid(base * x * xod, u) / zb
    cov = q - m * z
    return zb * np.exp(-abs(r - s) / TAU), m, z, q, cov


def scan_monotonicity(n: int) -> None:
    worst_dm = np.inf
    worst_dz = np.inf
    worst_at = (0.0, 0.0, "")
    for r in np.geomspace(0.02, 50.0, 40):
        smax = 8.0 * r + 40.0
        ss = np.linspace(1e-4, smax, 900)
        ms, zs = [], []
        for s in ss:
            _, m, z, _, _ = shell_stats(n, r, s)
            ms.append(m)
            zs.append(z)
        dm = np.diff(ms)
        dz = np.diff(zs)
        if dm.min() < worst_dm:
            worst_dm = dm.min()
            worst_at = (r, ss[int(dm.argmin())], "m")
        if dz.min() < worst_dz:
            worst_dz = dz.min()
    print(f"[A n={n}] min step of m(s): {worst_dm:+.2e} "
          f"(worst near r={worst_at[0]:.3g}, s={worst_at[1]:.3g}); "
          f"min step of z(s): {worst_dz:+.2e}  "
          f"({'MONOTONE' if worst_dm > -1e-10 and worst_dz > -1e-10 else 'VIOLATION'})")


def scan_within_floor(n: int) -> None:
    floor = np.inf
    at = (0.0, 0.0)
    for r in np.geomspace(0.02, 50.0, 40):
        for s in np.linspace(1e-3, 4.0 * r + 20.0, 500):
            _, _, _, _, cov = shell_stats(n, r, s)
            if cov < floor:
                floor = cov
                at = (r, s)
    print(f"[B n={n}] within-shell Cov floor: {floor:+.5f}  "
          f"(bound -(n-1)/2 = {-(n - 1) / 2:.1f}; worst r={at[0]:.3g}, "
          f"s={at[1]:.3g}, s/r={at[1] / at[0]:.3f})")


def scan_two_shell(n: int, trials: int) -> None:
    worst_total = np.inf
    worst_between = np.inf
    for _ in range(trials):
        r = float(np.exp(RNG.uniform(np.log(0.05), np.log(20.0))))
        s1 = float(np.exp(RNG.uniform(np.log(0.01 * r), np.log(10 * r + 30))))
        s2 = float(np.exp(RNG.uniform(np.log(0.01 * r), np.log(10 * r + 30))))
        a = float(RNG.uniform(0.02, 0.98))
        zb1, m1, z1, q1, c1 = shell_stats(n, r, s1)
        zb2, m2, z2, q2, c2 = shell_stats(n, r, s2)
        p1 = a * zb1 / (a * zb1 + (1 - a) * zb2)
        p2 = 1.0 - p1
        m = p1 * m1 + p2 * m2
        z = p1 * z1 + p2 * z2
        q = p1 * q1 + p2 * q2
        total = q - m * z
        between = p1 * (m1 - m) * (z1 - z) + p2 * (m2 - m) * (z2 - z)
        worst_total = min(worst_total, total)
        worst_between = min(worst_between, between)
    print(f"[C n={n}] two-shell mixtures ({trials} random): "
          f"min Cov = {worst_total:+.5f} (needs > {-(n - 1):.0f}); "
          f"min between-term = {worst_between:+.5f} "
          f"(association predicts >= 0)")


if __name__ == "__main__":
    print("G2 screen: between-shell association (tau=1, seed 20260717)")
    for n in (3, 4, 5, 10):
        scan_monotonicity(n)
    for n in (3, 5):
        scan_within_floor(n)
    for n in (3, 5):
        scan_two_shell(n, trials=400)
