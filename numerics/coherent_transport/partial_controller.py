"""Partial-transport coverage controller + geometry guard (plan sections 4.3,
4.5, 4.6; Stage 1B).

Moves only DEMONSTRABLY DEFICIENT mass: rank particles by surplus (off-support /
over-dense), rank target regions by deficit (under-covered), and transport a
fraction rho of surplus particles onto distinct deficit targets. Leaves
well-placed particles to the local geometry learner (trust region, RQ2/H2). A
geometry guard backtracks any repair step that would worsen support precision.

Oracle discipline: uses only generated particles + target samples (no component
labels, no evaluation pool). Pure numpy.
"""

from __future__ import annotations

import numpy as np

from metrics import knn_radius, _pairwise


def target_deficit(q: np.ndarray, Y: np.ndarray, k: int = 5):
    """Uncovered target points (no generated particle within their k-NN radius)
    and the deficit fraction. Label-free, target-calibrated."""
    rt = knn_radius(Y, k)
    D = _pairwise(Y, q)                       # (M, N)
    covered = (D <= rt[:, None]).any(axis=1)
    return ~covered, float((~covered).mean())


def surplus_order(q: np.ndarray, Y: np.ndarray, k: int = 5) -> np.ndarray:
    """Particles ranked most-surplus first: off-support particles (far from the
    target manifold relative to its local scale) come first, then over-dense."""
    rt = knn_radius(Y, k)
    Dqy = _pairwise(q, Y)                     # (N, M)
    nearest = Dqy.argmin(axis=1)
    offsup = Dqy.min(axis=1) / (rt[nearest] + 1e-12)   # >1 ~ off-support
    return np.argsort(-offsup)


def partial_repair(q: np.ndarray, Y: np.ndarray, rho: float, k: int = 5,
                   precision_tol: float = 0.01) -> np.ndarray:
    """One deficit-fill repair step. Fill each uncovered target with its nearest
    available surplus particle (1-1, maximising coverage gain), then backtrack
    the whole repair if it lowers support precision beyond tolerance."""
    if rho <= 0.0:
        return q
    uncovered_mask, _ = target_deficit(q, Y, k)
    unc = np.nonzero(uncovered_mask)[0]
    if len(unc) == 0:
        return q
    order = surplus_order(q, Y, k)
    n_move = max(1, int(round(rho * len(q))))
    movers = order[:n_move]
    Du = _pairwise(Y[unc], q[movers])         # (n_unc, n_move)
    taken = np.zeros(len(movers), dtype=bool)
    dest_idx, mover_idx = [], []
    for j in range(len(unc)):
        for mi in np.argsort(Du[j]):
            if not taken[mi]:
                taken[mi] = True
                dest_idx.append(unc[j]); mover_idx.append(movers[mi])
                break
    if not mover_idx:
        return q
    mover_idx = np.asarray(mover_idx); dest = Y[np.asarray(dest_idx)]
    base_prec = _precision(q, Y, k)
    delta = dest - q[mover_idx]
    for eta in (1.0, 0.5, 0.25, 0.125, 0.0):   # geometry guard / backtracking
        if eta == 0.0:
            return q
        qtry = q.copy()
        qtry[mover_idx] = q[mover_idx] + eta * delta
        if _precision(qtry, Y, k) >= base_prec - precision_tol:
            return qtry
    return q


def _precision(q, Y, k):
    rt = knn_radius(Y, k)
    return float((_pairwise(q, Y) <= rt[None, :]).any(axis=1).mean())


def adaptive_rho(q: np.ndarray, Y: np.ndarray, k: int = 5,
                 candidates=(0.0, 0.05, 0.10, 0.20, 0.40, 1.0)) -> float:
    """Trust-region rho: smallest predeclared candidate that covers the current
    deficit. rho ~ deficit fraction, so it shrinks to 0 as coverage fills
    (never pinned at 1). Uses only generated+target (no eval)."""
    _, deficit = target_deficit(q, Y, k)
    for c in candidates:
        if c >= deficit:
            return c
    return candidates[-1]


def _tests(log=print) -> None:
    rng = np.random.default_rng(11)
    fails = []

    def check(name, ok):
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails.append(name)

    # Two far clusters; generated covers only one -> other is deficit.
    A = rng.normal(size=(80, 2)) * 0.05 + [-5, 0]
    B = rng.normal(size=(80, 2)) * 0.05 + [5, 0]
    Y = np.vstack([A, B])
    q = np.vstack([rng.normal(size=(120, 2)) * 0.05 + [-5, 0],   # all on A
                   rng.normal(size=(40, 2)) * 3.0])              # + off-support
    mask, deficit = target_deficit(q, Y)
    check("1. uncovered cluster flagged as deficit", deficit > 0.3)

    # 2. rho=0 is exactly no repair.
    check("2. rho=0 -> identity", np.array_equal(partial_repair(q, Y, 0.0), q))

    # 3. a repair step raises coverage of the deficit cluster.
    cov_before = 1.0 - deficit
    q2 = partial_repair(q, Y, 0.4)
    _, deficit2 = target_deficit(q2, Y)
    check("3. repair reduces deficit", (1.0 - deficit2) > cov_before)

    # 4. exact match -> negligible deficit, adaptive rho ~ 0.
    qm = Y + rng.normal(size=Y.shape) * 0.001
    check("4. matched cloud -> adaptive rho small",
          adaptive_rho(qm, Y) <= 0.05)

    # 5. surplus order puts an off-support particle first.
    qs = np.vstack([Y[:5], np.array([[100.0, 100.0]])])
    check("5. off-support particle ranked most-surplus",
          surplus_order(qs, Y)[0] == len(qs) - 1)

    # 6. geometry guard: repair never lowers precision below tol.
    p_before = _precision(q, Y, 5)
    p_after = _precision(partial_repair(q, Y, 0.4), Y, 5)
    check("6. guard preserves precision", p_after >= p_before - 0.01 - 1e-9)

    if fails:
        raise SystemExit(f"partial controller tests FAILED: {fails}")
    log("all partial controller tests passed")


if __name__ == "__main__":
    _tests()
