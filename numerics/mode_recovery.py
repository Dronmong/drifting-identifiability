"""Mode-recovery metrics (ModeRecoveryRoadmap.md M0).

The primary scoreboard for the mode-recovery program: how many target modes a
particle cloud has *found* (coverage), how fast (time-to-cover), and whether the
found modes carry the right mass (calibration). Aggregate distributional error
is deliberately NOT here -- it is the old scoreboard this program replaces.

Oracle discipline (design invariant L3): every function takes the TRUE modes /
scales / weights explicitly. These are EVALUATION inputs only. No field or policy
in this program may call these functions or otherwise read the true modes; a
grep for `mode_recovery` in any field/policy module is a bug.

Pure numpy; no dependency on the heavy runner modules, so it stays trivially
testable and reusable by the M1/M2 harness.
"""

from __future__ import annotations

import numpy as np

# Frozen metric constants (fixed before any M1 run; document changes explicitly).
RHO = 3.0        # a particle is "at" mode k iff within RHO * sigma_k of it
KAPPA = 0.25     # mode k is "found" iff it holds >= KAPPA * fair-share particles
COVER_THETA = 1.0  # time-to-cover target: fraction of modes found (1.0 = all)


def _as_arrays(modes, sigmas, weights):
    modes = np.asarray(modes, dtype=float)
    K = len(modes)
    sig = np.asarray(sigmas, dtype=float)
    if sig.ndim == 0:
        sig = np.full(K, float(sig))
    if weights is None:
        w = np.full(K, 1.0 / K)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / w.sum()
    if sig.shape != (K,) or w.shape != (K,):
        raise ValueError("modes/sigmas/weights shape mismatch")
    return modes, sig, w


def mode_counts(q: np.ndarray, modes, sigmas, rho: float = RHO):
    """Particles per mode, assigning each particle to its NEAREST mode and
    keeping it only if within `rho * sigma` of that mode. Nearest-mode
    assignment guarantees a particle is counted for at most one mode."""
    q = np.asarray(q, dtype=float)
    modes = np.asarray(modes, dtype=float)
    sig = np.asarray(sigmas, dtype=float)
    if sig.ndim == 0:
        sig = np.full(len(modes), float(sig))
    d2 = ((q[:, None, :] - modes[None, :, :]) ** 2).sum(-1)   # (N, K)
    nearest = d2.argmin(1)
    dist = np.sqrt(d2[np.arange(len(q)), nearest])
    within = dist <= rho * sig[nearest]
    counts = np.zeros(len(modes), dtype=int)
    for k in range(len(modes)):
        counts[k] = int(np.sum((nearest == k) & within))
    return counts, nearest, within


def coverage(q: np.ndarray, modes, sigmas, weights=None,
             rho: float = RHO, kappa: float = KAPPA) -> dict:
    """Mode coverage of a cloud.

    A mode is "found" iff it holds at least `max(1, kappa * N * w_k)` particles
    within its radius. Returns both the unweighted coverage (fraction of modes
    found -- the missing-mode question) and the weighted coverage (fraction of
    target mass on found modes)."""
    q = np.asarray(q, dtype=float)
    modes, sig, w = _as_arrays(modes, sigmas, weights)
    N = len(q)
    counts, _, _ = mode_counts(q, modes, sig, rho)
    thresh = np.maximum(1.0, kappa * N * w)
    found = counts >= thresh
    return {
        "unweighted": float(found.mean()),
        "weighted": float((w * found).sum()),
        "found": found,
        "counts": counts,
        "n_found": int(found.sum()),
        "n_modes": len(modes),
    }


def mass_calibration(q: np.ndarray, modes, sigmas, weights=None,
                     rho: float = RHO) -> dict:
    """L1 error between the empirical per-mode particle fraction and the true
    weights. `found_only` restricts to modes that were found, isolating "right
    modes, wrong proportions" from "modes missing entirely"."""
    q = np.asarray(q, dtype=float)
    modes, sig, w = _as_arrays(modes, sigmas, weights)
    N = len(q)
    counts, _, _ = mode_counts(q, modes, sig, rho)
    emp = counts / max(N, 1)
    cov = coverage(q, modes, sig, w, rho)
    found = cov["found"]
    l1_all = float(np.abs(emp - w).sum())
    if found.any():
        wf = w[found] / w[found].sum()
        ef = (counts[found] / max(counts[found].sum(), 1))
        l1_found = float(np.abs(ef - wf).sum())
    else:
        l1_found = float("nan")
    return {"l1_all": l1_all, "l1_found_only": l1_found,
            "empirical_fraction": emp}


def time_to_cover(cov_series, theta: float = COVER_THETA,
                  steps=None) -> tuple[int | None, bool]:
    """First step at which unweighted coverage reaches `theta`, right-censored.

    `cov_series` is a 1-D sequence of unweighted-coverage values, one per
    recorded step. Returns `(step, censored)`; `censored=True` and `step=None`
    if the threshold is never reached."""
    cov = np.asarray(cov_series, dtype=float)
    if steps is None:
        steps = np.arange(1, len(cov) + 1)
    else:
        steps = np.asarray(steps)
    hit = np.nonzero(cov >= theta - 1e-12)[0]
    if len(hit) == 0:
        return None, True
    return int(steps[hit[0]]), False


def mode_drop_rate(q: np.ndarray, modes, sigmas, weights=None,
                   rho: float = RHO, kappa: float = KAPPA) -> float:
    """Fraction of modes NOT found (1 - unweighted coverage)."""
    return 1.0 - coverage(q, modes, sigmas, weights, rho, kappa)["unweighted"]


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

    # A well-separated K-mode mixture in d dims.
    K, d, N = 8, 3, 400
    L, sigma = 1.0, 0.02
    ang = 2 * np.pi * np.arange(K) / K
    modes = np.zeros((K, d))
    modes[:, 0] = L * np.cos(ang)
    modes[:, 1] = L * np.sin(ang)
    sig = np.full(K, sigma)
    weights = np.full(K, 1.0 / K)

    def sample_from(mode_ids, n):
        ids = np.asarray(mode_ids)
        pick = ids[rng.integers(0, len(ids), n)]
        return modes[pick] + rng.normal(size=(n, d)) * sigma

    # 1. Cloud from ALL modes -> unweighted coverage 1.
    q_all = sample_from(np.arange(K), N)
    c_all = coverage(q_all, modes, sig, weights)
    check("1. all-mode cloud -> unweighted coverage 1.0",
          abs(c_all["unweighted"] - 1.0) < 1e-9)

    # 2. Cloud at ONE mode -> unweighted 1/K, weighted w_0.
    q_one = sample_from([0], N)
    c_one = coverage(q_one, modes, sig, weights)
    check("2. single-mode cloud -> unweighted 1/K and weighted w_0",
          abs(c_one["unweighted"] - 1.0 / K) < 1e-9 and
          abs(c_one["weighted"] - weights[0]) < 1e-9)

    # 3. Translation invariance (shift cloud AND modes by same vector).
    shift = np.array([5.0, -3.0, 2.0])
    c_shift = coverage(q_all + shift, modes + shift, sig, weights)
    check("3. translation invariance",
          abs(c_shift["unweighted"] - c_all["unweighted"]) < 1e-12)

    # 4. Relabel invariance (permute modes, sigmas, weights together).
    perm = rng.permutation(K)
    c_perm = coverage(q_all, modes[perm], sig[perm], weights[perm])
    check("4. relabel invariance",
          abs(c_perm["unweighted"] - c_all["unweighted"]) < 1e-12)

    # 5. Far cloud -> coverage 0.
    q_far = np.full((N, d), 100.0) + rng.normal(size=(N, d)) * sigma
    check("5. far cloud -> coverage 0",
          coverage(q_far, modes, sig, weights)["unweighted"] == 0.0)

    # 6. No double counting: every found particle is counted once.
    counts, nearest, within = mode_counts(q_all, modes, sig)
    check("6. counts sum == particles within radius (no double count)",
          int(counts.sum()) == int(within.sum()))

    # 7. time_to_cover: reaching vs censored.
    series = [0.2, 0.4, 0.4, 0.8, 1.0, 1.0]
    t, cens = time_to_cover(series)
    tc, censc = time_to_cover([0.2, 0.4, 0.6])
    check("7. time_to_cover first-crossing and censoring",
          t == 5 and not cens and tc is None and censc)

    # 8. mass calibration: fair-share cloud ~0 error; single-mode large.
    cal_all = mass_calibration(q_all, modes, sig, weights)
    cal_one = mass_calibration(q_one, modes, sig, weights)
    check("8. mass calibration low for fair share, high for single mode",
          cal_all["l1_all"] < 0.15 and cal_one["l1_all"] > 1.0)

    # 9. Unequal weights: mode found iff it clears its OWN fair share.
    w_uneq = np.arange(1, K + 1, dtype=float)
    w_uneq /= w_uneq.sum()
    # Put a fair-share cloud for unequal weights: everything should be found.
    ids = rng.choice(K, size=N, p=w_uneq)
    q_uneq = modes[ids] + rng.normal(size=(N, d)) * sigma
    c_uneq = coverage(q_uneq, modes, sig, w_uneq)
    check("9. unequal-weight fair-share cloud -> coverage ~1",
          c_uneq["unweighted"] >= 0.99)

    if fails:
        raise SystemExit(f"mode-recovery invariant tests FAILED: {fails}")
    log("all mode-recovery invariant tests passed")


if __name__ == "__main__":
    invariant_tests()
