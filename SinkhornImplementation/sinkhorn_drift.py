"""Sinkhorn-balanced drifting: EXTENSION library (not part of the paper).

Generalizes Algorithm 2 of arXiv:2602.04770v2 by iterating the geometric-mean
row/column balancing that the paper applies exactly once
(`A = sqrt(A_row * A_col)`).  `iters = 1` reproduces the paper's affinity
bit-for-bit (asserted against `numerics/driftlab.py`); `iters = 0` is the
row-softmax-only (InfoNCE-style) baseline; `iters -> inf` approaches a
doubly-stochastic (entropic-OT) coupling.

The Lean-certified counterpart is `DriftingIdentifiability/SinkhornBalanced.lean`:
every balancing iterate is a positive diagonal rescaling `u(x) k(x,y) v(y)` of
the paper kernel, and identifiability transfers along the whole orbit
(`sinkhornOrbit01Setup`, `interactionFrameBound_of_biScaling`).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Distances and balanced affinities
# ----------------------------------------------------------------------------


def cdist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Euclidean pairwise distances; accepts [n] or [n, d] arrays."""
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    sq = (x**2).sum(1)[:, None] + (y**2).sum(1)[None, :] - 2.0 * (x @ y.T)
    return np.sqrt(np.maximum(sq, 0.0))


def balanced_affinity(logit: np.ndarray, iters: int) -> np.ndarray:
    """Geometric-mean Sinkhorn balancing of `exp(logit)`, in the log domain.

    One step maps M to D_r^{-1/2} M D_c^{-1/2} (D_r, D_c the row/column sums).
    `iters = 1` equals the paper's `sqrt(A_row * A_col)` exactly;
    `iters = 0` returns the row softmax only.
    """
    L = logit.astype(float).copy()
    if iters == 0:
        L = L - L.max(axis=1, keepdims=True)
        e = np.exp(L)
        return e / e.sum(axis=1, keepdims=True)
    for _ in range(iters):
        # log-domain: subtract half the row/col log-sum-exp
        row = _logsumexp(L, axis=1, keepdims=True)
        col = _logsumexp(L, axis=0, keepdims=True)
        L = L - 0.5 * row - 0.5 * col
    return np.exp(L)


def _logsumexp(a: np.ndarray, axis: int, keepdims: bool) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


# ----------------------------------------------------------------------------
# Drift computation (Algorithm 2 with balancing depth as a parameter)
# ----------------------------------------------------------------------------


def compute_v_sinkhorn(
    x: np.ndarray,
    y_pos: np.ndarray,
    y_neg: np.ndarray,
    T: float,
    iters: int,
    self_mask: bool,
) -> np.ndarray:
    """Algorithm 2 with `iters` balancing steps (paper = 1).

    Keeps the paper's weight recipe `W_pos = A_pos * A_neg.sum`,
    `W_neg = A_neg * A_pos.sum`, and the `eye * 1e6` self-mask convention."""
    dist_pos = cdist(x, y_pos)
    dist_neg = cdist(x, y_neg)
    if self_mask:
        assert dist_neg.shape[0] == dist_neg.shape[1]
        dist_neg = dist_neg + np.eye(dist_neg.shape[0]) * 1e6
    logit = np.concatenate([-dist_pos / T, -dist_neg / T], axis=1)
    A = balanced_affinity(logit, iters)
    npos = dist_pos.shape[1]
    A_pos, A_neg = A[:, :npos], A[:, npos:]
    W_pos = A_pos * A_neg.sum(axis=1, keepdims=True)
    W_neg = A_neg * A_pos.sum(axis=1, keepdims=True)
    ypos2 = y_pos if y_pos.ndim > 1 else y_pos[:, None]
    yneg2 = y_neg if y_neg.ndim > 1 else y_neg[:, None]
    V = W_pos @ ypos2 - W_neg @ yneg2
    return V if y_pos.ndim > 1 else V[:, 0]


def affinity_diagnostics(
    x: np.ndarray,
    y_pos: np.ndarray,
    y_neg: np.ndarray,
    T: float,
    iters: int,
    self_mask: bool,
) -> dict:
    """Row/column mass dispersion and per-anchor ESS of the balanced affinity.

    Mass coefficient-of-variation is the numerical dial for the certified
    denominator floor: CV -> 0 means the random SNIS denominator becomes
    deterministic (the slack `DenominatorTail.lean` exists to absorb)."""
    dist_pos = cdist(x, y_pos)
    dist_neg = cdist(x, y_neg)
    if self_mask:
        dist_neg = dist_neg + np.eye(dist_neg.shape[0]) * 1e6
    logit = np.concatenate([-dist_pos / T, -dist_neg / T], axis=1)
    A = balanced_affinity(logit, iters)
    row = A.sum(axis=1)
    col = A.sum(axis=0)
    npos = dist_pos.shape[1]
    A_neg = A[:, npos:]
    ess = (A_neg.sum(axis=1) ** 2) / np.maximum((A_neg**2).sum(axis=1), 1e-300)
    return {
        "row_cv": float(row.std() / row.mean()),
        "col_cv": float(col.std() / col.mean()),
        "row_min_over_mean": float(row.min() / row.mean()),
        "ess_median": float(np.median(ess)),
        "ess_min": float(ess.min()),
    }


# ----------------------------------------------------------------------------
# Toy particle descent (the paper's Figure-3 methodology, no network)
# ----------------------------------------------------------------------------


def bimodal_sampler(weights: np.ndarray, centers: np.ndarray, std: float, rng):
    """Sampler for a 2-D Gaussian mixture."""

    def sample(n: int) -> np.ndarray:
        comp = rng.random(n) < weights[1]
        pts = centers[comp.astype(int)] + rng.normal(0, std, (n, 2))
        return pts

    return sample


def mode_mass_error(
    particles: np.ndarray, centers: np.ndarray, weights: np.ndarray
) -> float:
    """|empirical mass of mode 1 - target weight of mode 1| by nearest center."""
    d = cdist(particles, centers)
    frac1 = float((d.argmin(axis=1) == 1).mean())
    return abs(frac1 - float(weights[1]))


def particle_descent(
    weights: np.ndarray,
    centers: np.ndarray,
    std: float,
    init: str,
    n_particles: int,
    n_pos: int,
    steps: int,
    eta: float,
    T: float,
    iters: int,
    rng,
    record_every: int = 10,
) -> list[tuple[int, float]]:
    """Move particles by the (balanced) drift field; return mode-mass error
    trajectory.  Negatives are the particles themselves with the eye mask,
    matching the paper's reuse convention."""
    sample = bimodal_sampler(weights, centers, std, rng)
    if init == "between":
        pts = rng.normal(0, 0.25, (n_particles, 2))
    elif init == "far":
        pts = np.array([5.0, 5.0]) + rng.normal(0, 0.25, (n_particles, 2))
    elif init == "collapsed":
        pts = centers[0] + rng.normal(0, 0.1, (n_particles, 2))
    else:
        raise ValueError(init)
    traj = []
    for step in range(steps + 1):
        if step % record_every == 0:
            traj.append((step, mode_mass_error(pts, centers, weights)))
        ypos = sample(n_pos)
        V = compute_v_sinkhorn(pts, ypos, pts, T, iters, self_mask=True)
        pts = pts + eta * V
    return traj
