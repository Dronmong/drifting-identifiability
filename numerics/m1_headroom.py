"""M1 headroom precondition (ModeRecoveryRoadmap.md, design invariant L2).

Demonstrates -- BEFORE any candidate is built -- that a single fixed bandwidth
cannot both REACH distant modes and RESOLVE them under missing-mode
initialization, so there is a real coverage deficit for a multi-scale field to
attack. Each fixed bandwidth is given its best shot via an eta sweep.

reach   = basin coverage (mass in the Voronoi basin)      -- needs coarse tau
resolve = sigma coverage (mass at intra-mode precision)   -- needs fine tau
spread  = cloud std / target std                          -- did the swarm split
calib   = L1 mass-calibration error                       -- lower is better

Verdict: headroom exists iff no single tau reaches AND resolves together.

    uv run --with numpy --with scipy python numerics/m1_headroom.py
"""
import sys
import numpy as np

sys.path.insert(0, ".")
from lowdim_drift import drift_paper, controlled_means           # noqa: E402
from mode_recovery import (coverage, basin_radius, sigma_radius,  # noqa: E402
                           mass_calibration)

TAUS = (0.02, 0.05, 0.1, 0.2, 0.4, 0.8)
ETAS = (0.05, 0.1, 0.2)
REGIMES = [(16, 2, 1.0, 0.03), (16, 5, 1.0, 0.03), (32, 2, 1.0, 0.02)]


def run(K, d, L, sigma, N=128, steps=800, seeds=3):
    modes = controlled_means(K, d, L)
    w = np.full(K, 1.0 / K)
    rb, rs = basin_radius(modes), sigma_radius(sigma, K=K)
    tgt_spread = float(np.std(modes, axis=0).mean())

    def target(n, rng):
        i = rng.integers(0, K, n)
        return modes[i] + rng.normal(size=(n, d)) * sigma

    def init(rng):
        return modes[0] + rng.normal(size=(N, d)) * sigma

    print(f"\n=== K={K} d={d} L/sigma={L/sigma:.0f} "
          f"(basin r~{rb.mean():.3f}, sigma r~{rs.mean():.3f}) ===")
    print(f"{'tau':>6}{'reach':>7}{'resolv':>7}{'spread':>7}{'calib':>7}"
          "  (best over eta)")
    best_reach_resolve = 0.0
    for tau in TAUS:
        best = None
        for eta in ETAS:
            rc, rz, sp, cal = [], [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(100 + s)
                drng = np.random.default_rng(200 + s)
                q = init(rng)
                for _ in range(steps):
                    q = q + eta * drift_paper(q, target(N, drng), tau, True)
                    if not np.all(np.isfinite(q)):
                        break
                if not np.all(np.isfinite(q)):
                    continue
                rc.append(coverage(q, modes, rb, w)["unweighted"])
                rz.append(coverage(q, modes, rs, w)["unweighted"])
                sp.append(float(np.std(q, axis=0).mean()) / tgt_spread)
                cal.append(mass_calibration(q, modes, rb, w)["l1_all"])
            if not rc:
                continue
            key = float(np.median(rc))
            if best is None or key > best[0]:
                best = (key, float(np.median(rz)), float(np.median(sp)),
                        float(np.median(cal)))
        if best:
            print(f"{tau:>6}{best[0]:>7.2f}{best[1]:>7.2f}"
                  f"{best[2]:>7.2f}{best[3]:>7.2f}")
            if best[0] >= 0.9:
                best_reach_resolve = max(best_reach_resolve, best[1])
    print(f"  >> best RESOLUTION at full reach (reach>=0.9): "
          f"{best_reach_resolve:.2f}  "
          f"({'HEADROOM' if best_reach_resolve < 0.9 else 'no headroom'})")


if __name__ == "__main__":
    for reg in REGIMES:
        run(*reg)
