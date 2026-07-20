"""Mode-recovery metrics (ModeRecoveryRoadmap.md M0).

The primary scoreboard for the mode-recovery program: how many target modes a
particle cloud has *found* (coverage), how fast (time-to-cover), and whether the
found modes carry the right mass (calibration). Aggregate distributional error
is deliberately NOT here -- it is the old scoreboard this program replaces.

**Reach vs resolution (why radius is explicit).** "Found mode k" must mean
"cloud mass reached mode k's basin", not "cloud resolved mode k to intra-mode
precision". Tying the radius to the intra-mode width `sigma` conflates the two
(a tiny sigma makes coverage demand sub-sigma convergence). So coverage takes an
explicit per-mode absolute `radius`, and two builders are provided:

* `basin_radius(modes, frac)` -- radius = `frac x` nearest-neighbour distance
  (`frac < 0.5` keeps it strictly inside the Voronoi basin). This is the
  **reach** metric and the program's PRIMARY coverage radius.
* `sigma_radius(sigmas, rho)` -- radius = `rho x sigma`. This is the tight
  **resolution** radius, used for calibration-style checks.

Oracle discipline (design invariant L3): every function takes the TRUE modes /
scales / weights explicitly. These are EVALUATION inputs only -- no field or
policy in this program may read the true modes.

Pure numpy; no dependency on the heavy runner modules.
"""

from __future__ import annotations

import numpy as np

# Frozen metric constants (fixed before any M1 gate run; document changes).
BASIN_FRAC = 0.4   # reach radius = BASIN_FRAC * nearest-neighbour distance
RHO = 3.0          # resolution radius = RHO * sigma
KAPPA = 0.25       # mode found iff it holds >= KAPPA * fair-share particles
COVER_THETA = 1.0  # time-to-cover target: fraction of modes found (1.0 = all)


def _prep(modes, weights):
    modes = np.asarray(modes, dtype=float)
    K = len(modes)
    if weights is None:
        w = np.full(K, 1.0 / K)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
    if w.shape != (K,):
        raise ValueError("weights shape mismatch")
    return modes, w


def nn_distance(modes) -> np.ndarray:
    """Per-mode distance to the nearest other mode."""
    modes = np.asarray(modes, dtype=float)
    d2 = ((modes[:, None, :] - modes[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min(1))


def basin_radius(modes, frac: float = BASIN_FRAC) -> np.ndarray:
    """Reach radius: a fraction (<0.5) of the nearest-neighbour distance, so the
    ball sits strictly inside the mode's Voronoi basin."""
    if frac >= 0.5:
        raise ValueError("basin frac must be < 0.5 to stay inside the basin")
    return frac * nn_distance(modes)


def sigma_radius(sigmas, rho: float = RHO, K: int | None = None) -> np.ndarray:
    """Resolution radius: rho * sigma (broadcast scalar sigma to K modes)."""
    sig = np.asarray(sigmas, dtype=float)
    if sig.ndim == 0:
        if K is None:
            raise ValueError("scalar sigma needs K")
        sig = np.full(K, float(sig))
    return rho * sig


def mode_counts(q: np.ndarray, modes, radius):
    """Particles per mode: each particle is assigned to its NEAREST mode and
    kept iff within that mode's `radius`. Nearest-mode assignment guarantees a
    particle counts for at most one mode."""
    q = np.asarray(q, dtype=float)
    modes = np.asarray(modes, dtype=float)
    radius = np.asarray(radius, dtype=float)
    d2 = ((q[:, None, :] - modes[None, :, :]) ** 2).sum(-1)
    nearest = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(q)), nearest])
    within = dist <= radius[nearest]
    counts = np.zeros(len(modes), dtype=int)
    for k in range(len(modes)):
        counts[k] = int(np.sum((nearest == k) & within))
    return counts, nearest, within


def coverage(q: np.ndarray, modes, radius, weights=None,
             kappa: float = KAPPA) -> dict:
    """Mode coverage. A mode is "found" iff it holds >= max(1, kappa*N*w_k)
    particles within `radius`. Returns unweighted coverage (fraction of modes
    found -- the missing-mode question) and weighted coverage (target mass on
    found modes)."""
    q = np.asarray(q, dtype=float)
    modes, w = _prep(modes, weights)
    N = len(q)
    counts, _, _ = mode_counts(q, modes, radius)
    thresh = np.maximum(1.0, kappa * N * w)
    found = counts >= thresh
    return {"unweighted": float(found.mean()),
            "weighted": float((w * found).sum()),
            "found": found, "counts": counts,
            "n_found": int(found.sum()), "n_modes": len(modes)}


def mass_calibration(q: np.ndarray, modes, radius, weights=None) -> dict:
    """L1 error between empirical per-mode particle fraction and true weights;
    `l1_found_only` restricts to found modes (right modes, wrong proportions)."""
    q = np.asarray(q, dtype=float)
    modes, w = _prep(modes, weights)
    N = len(q)
    counts, _, _ = mode_counts(q, modes, radius)
    emp = counts / max(N, 1)
    found = coverage(q, modes, radius, w)["found"]
    l1_all = float(np.abs(emp - w).sum())
    if found.any():
        wf = w[found] / w[found].sum()
        ef = counts[found] / max(counts[found].sum(), 1)
        l1_found = float(np.abs(ef - wf).sum())
    else:
        l1_found = float("nan")
    return {"l1_all": l1_all, "l1_found_only": l1_found,
            "empirical_fraction": emp}


def time_to_cover(cov_series, theta: float = COVER_THETA,
                  steps=None) -> tuple[int | None, bool]:
    """First recorded step where unweighted coverage reaches `theta`,
    right-censored. Returns `(step, censored)`."""
    cov = np.asarray(cov_series, dtype=float)
    steps = np.arange(1, len(cov) + 1) if steps is None else np.asarray(steps)
    hit = np.nonzero(cov >= theta - 1e-12)[0]
    if len(hit) == 0:
        return None, True
    return int(steps[hit[0]]), False


def mode_drop_rate(q, modes, radius, weights=None, kappa: float = KAPPA) -> float:
    return 1.0 - coverage(q, modes, radius, weights, kappa)["unweighted"]


# ---------------------------------------------------------------------------
# Invariant tests (M0 acceptance)
# ---------------------------------------------------------------------------


def invariant_tests(log=print) -> None:
    rng = np.random.default_rng(20260720)
    fails: list[str] = []

    def check(name: str, ok: bool) -> None:
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails.append(name)

    K, d, N = 8, 3, 400
    L, sigma = 1.0, 0.02
    ang = 2 * np.pi * np.arange(K) / K
    modes = np.zeros((K, d))
    modes[:, 0] = L * np.cos(ang)
    modes[:, 1] = L * np.sin(ang)
    weights = np.full(K, 1.0 / K)
    rad_basin = basin_radius(modes)
    rad_sigma = sigma_radius(sigma, K=K)

    def sample_from(mode_ids, n):
        ids = np.asarray(mode_ids)
        pick = ids[rng.integers(0, len(ids), n)]
        return modes[pick] + rng.normal(size=(n, d)) * sigma

    q_all = sample_from(np.arange(K), N)

    # 1. basin radius sits strictly inside the Voronoi cell (< half nn).
    check("1. basin radius < 0.5 * nn distance",
          bool(np.all(rad_basin < 0.5 * nn_distance(modes))))

    # 2. all-mode cloud -> coverage 1 under BOTH radii (well-separated).
    check("2. all-mode cloud -> coverage 1 (basin & sigma)",
          abs(coverage(q_all, modes, rad_basin, weights)["unweighted"] - 1) < 1e-9
          and abs(coverage(q_all, modes, rad_sigma, weights)["unweighted"] - 1)
          < 1e-9)

    # 3. single-mode cloud -> unweighted 1/K, weighted w_0.
    q_one = sample_from([0], N)
    c_one = coverage(q_one, modes, rad_basin, weights)
    check("3. single-mode cloud -> 1/K unweighted, w_0 weighted",
          abs(c_one["unweighted"] - 1 / K) < 1e-9 and
          abs(c_one["weighted"] - weights[0]) < 1e-9)

    # 4. reach vs resolution: points inside the basin but outside the tight
    #    sigma ball (0.3*nn from each mode, radially outward, since
    #    3*sigma < 0.3*nn < 0.4*nn) are credited by the basin radius but not by
    #    the resolution radius.
    nn = nn_distance(modes)
    dirs = modes / np.linalg.norm(modes, axis=1, keepdims=True)
    reps = N // K
    edge_pts = (np.repeat(modes, reps, axis=0)
                + 0.3 * np.repeat((nn[:, None] * dirs), reps, axis=0))
    cov_b = coverage(edge_pts, modes, rad_basin, weights)["unweighted"]
    cov_s = coverage(edge_pts, modes, rad_sigma, weights)["unweighted"]
    check("4. basin credits reach where sigma (resolution) does not",
          cov_b > cov_s and cov_b > 0.99 and cov_s < 0.01)

    # 5. translation invariance.
    shift = np.array([5.0, -3.0, 2.0])
    check("5. translation invariance",
          abs(coverage(q_all + shift, modes + shift, rad_basin, weights)["unweighted"]
              - coverage(q_all, modes, rad_basin, weights)["unweighted"]) < 1e-12)

    # 6. relabel invariance.
    perm = rng.permutation(K)
    check("6. relabel invariance",
          abs(coverage(q_all, modes[perm], basin_radius(modes[perm]), weights[perm])["unweighted"]
              - coverage(q_all, modes, rad_basin, weights)["unweighted"]) < 1e-12)

    # 7. far cloud -> coverage 0.
    q_far = np.full((N, d), 100.0) + rng.normal(size=(N, d)) * sigma
    check("7. far cloud -> coverage 0",
          coverage(q_far, modes, rad_basin, weights)["unweighted"] == 0.0)

    # 8. no double counting.
    counts, _, within = mode_counts(q_all, modes, rad_basin)
    check("8. counts sum == particles within radius (no double count)",
          int(counts.sum()) == int(within.sum()))

    # 9. time_to_cover first-crossing and censoring.
    t, cens = time_to_cover([0.2, 0.4, 0.4, 0.8, 1.0])
    tc, censc = time_to_cover([0.2, 0.4, 0.6])
    check("9. time_to_cover first-crossing and censoring",
          t == 5 and not cens and tc is None and censc)

    # 10. mass calibration low for fair share, high for single mode.
    check("10. mass calibration separates fair-share from single-mode",
          mass_calibration(q_all, modes, rad_basin, weights)["l1_all"] < 0.15
          and mass_calibration(q_one, modes, rad_basin, weights)["l1_all"] > 1.0)

    if fails:
        raise SystemExit(f"mode-recovery invariant tests FAILED: {fails}")
    log("all mode-recovery invariant tests passed")


if __name__ == "__main__":
    invariant_tests()
