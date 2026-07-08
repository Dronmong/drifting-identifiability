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

Second extension layer (`gain=...` in `compute_v_sinkhorn`): the exact
identity `algorithm2Drift = (P*Q) . (C+ - C-)` (Lean:
`algorithm2Drift_eq_massProduct_centroidDiff`, Algorithm2Estimator.lean)
splits the paper's drift into a raw-mass GAIN and a self-normalized-centroid
SIGNAL.  Only the signal carries identifiability; the gain is a per-query
positive rescaling, the exact class covered by
`interactionFrameBound_of_probeScaling`.  `gain_schedule` supplies
alternatives to the paper's `P*Q` gain, which collapses exponentially far
from support (`algorithm2Drift_norm_le_affinityMass`).  See
`PROPOSAL_CERTIFIED_GAIN.md` for the full derivation and status.
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
    gain: str = "paper",
    gain_kwargs: dict | None = None,
) -> np.ndarray:
    """Algorithm 2 with `iters` balancing steps (paper = 1), and a pluggable
    per-query GAIN multiplying the self-normalized centroid-difference signal
    (Lean: `algorithm2Drift_eq_massProduct_centroidDiff`).

    `gain = "paper"` (default) takes the ORIGINAL code path unchanged --
    `W_pos = A_pos * A_neg.sum`, `W_neg = A_neg * A_pos.sum` -- and is
    bit-for-bit identical to the pre-extension implementation (S0-checked).
    Other modes keep the same balanced affinity and the same centroid-
    difference signal, replacing only the scalar gain; see `gain_schedule`
    and `PROPOSAL_CERTIFIED_GAIN.md`."""
    dist_pos = cdist(x, y_pos)
    dist_neg = cdist(x, y_neg)
    if self_mask:
        assert dist_neg.shape[0] == dist_neg.shape[1]
        dist_neg = dist_neg + np.eye(dist_neg.shape[0]) * 1e6
    logit = np.concatenate([-dist_pos / T, -dist_neg / T], axis=1)
    A = balanced_affinity(logit, iters)
    npos = dist_pos.shape[1]
    A_pos, A_neg = A[:, :npos], A[:, npos:]
    ypos2 = y_pos if y_pos.ndim > 1 else y_pos[:, None]
    yneg2 = y_neg if y_neg.ndim > 1 else y_neg[:, None]

    if gain == "paper":
        W_pos = A_pos * A_neg.sum(axis=1, keepdims=True)
        W_neg = A_neg * A_pos.sum(axis=1, keepdims=True)
        V = W_pos @ ypos2 - W_neg @ yneg2
        return V if y_pos.ndim > 1 else V[:, 0]

    P = A_pos.sum(axis=1)
    Q = A_neg.sum(axis=1)
    Cpos = (A_pos @ ypos2) / np.maximum(P, 1e-300)[:, None]
    Cneg = (A_neg @ yneg2) / np.maximum(Q, 1e-300)[:, None]
    g = gain_schedule(gain, P, Q, A_pos, A_neg, ypos2, yneg2, **(gain_kwargs or {}))
    V = g[:, None] * (Cpos - Cneg)
    return V if y_pos.ndim > 1 else V[:, 0]


def _ess(sq_row_sum: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """Effective sample size of one branch's row weights: mass^2 / sum(w^2)."""
    return mass**2 / np.maximum(sq_row_sum, 1e-300)


def _robust_diameter(*point_sets: np.ndarray) -> float:
    """Coordinate-range diameter of the union of point sets: the empirical
    hull-radius proxy used by the certificate gain (matches the convex-hull
    bound in `selfNormalizedCentroid_relative_perturbation`, B1)."""
    pts = np.concatenate([p if p.ndim > 1 else p[:, None] for p in point_sets], axis=0)
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def gain_schedule(
    mode: str,
    P: np.ndarray,
    Q: np.ndarray,
    A_pos: np.ndarray,
    A_neg: np.ndarray,
    y_pos: np.ndarray,
    y_neg: np.ndarray,
    *,
    gamma: float = 0.5,
    gmax: float | None = None,
    lam: float = 1.0,
    diameter: float | None = None,
) -> np.ndarray:
    """Per-query STRICTLY POSITIVE gain, replacing the paper's implicit
    `P * Q` mass product.  Every mode is a positive per-probe rescaling of
    the centroid-difference signal -- exactly the class already certified by
    `interactionFrameBound_of_probeScaling` (SinkhornBalanced.lean), so
    identifiability of the resulting estimator is not a new question.

    - "power": `(P*Q)**gamma` (gamma = 1 reduces to the paper's `P*Q`;
      gamma < 1 shrinks less aggressively off-support -- the cheapest fix).
    - "min":   `min(P, Q)`.
    - "const": a single scalar per call: `gmax` if given, else this batch's
      `median(P*Q)`.  CAUTION: the no-`gmax` fallback recomputes the scale
      from the CURRENT batch, so on a frozen/far-from-support batch it is
      just as small as the paper's gain -- callers that need the far-init
      unfreezing property (e.g. `run_sinkhorn.py`'s S7) MUST pass a `gmax`
      calibrated once from a healthy/matched batch (`calibrate_gmax`).
    - "cert": `gmax / (1 + lam * Ehat)`, `Ehat` the plug-in shadow of the
      certified finite-sample deviation bounds -- a per-branch ESS term
      capped by the empirical hull diameter (see
      `PROPOSAL_CERTIFIED_GAIN.md`).  Same `gmax` caveat as "const"."""
    default_scale = float(np.median(P * Q)) if gmax is None else gmax
    if mode == "power":
        return (P * Q) ** gamma
    if mode == "min":
        return np.minimum(P, Q)
    if mode == "const":
        return np.full(P.shape, default_scale)
    if mode == "cert":
        ess_pos = _ess((A_pos**2).sum(axis=1), P)
        ess_neg = _ess((A_neg**2).sum(axis=1), Q)
        D = _robust_diameter(y_pos, y_neg) if diameter is None else diameter
        Ehat = D * (
            1.0 / np.sqrt(np.maximum(ess_pos, 1e-300))
            + 1.0 / np.sqrt(np.maximum(ess_neg, 1e-300))
        )
        return default_scale / (1.0 + lam * Ehat)
    raise ValueError(f"unknown gain mode {mode!r}")


def calibrate_gmax(
    weights: np.ndarray,
    centers: np.ndarray,
    std: float,
    n_particles: int,
    n_pos: int,
    T: float,
    iters: int,
    rng,
) -> float:
    """Reference gain scale for `gain="const"`/`gain="cert"`: `median(P*Q)`
    on a HEALTHY batch (particles drawn from the target itself, not from a
    frozen/degenerate init).  This fixed reference is what lets the
    certificate gain move at near-paper speed close to support while still
    granting a bounded floor far from it -- using the current (possibly
    frozen) batch's own median instead would reintroduce the paper's
    collapse (see `gain_schedule`'s "const" caveat)."""
    sample = bimodal_sampler(weights, centers, std, rng)
    pts = sample(n_particles)
    ypos = sample(n_pos)
    dist_pos = cdist(pts, ypos)
    dist_neg = cdist(pts, pts) + np.eye(n_particles) * 1e6
    logit = np.concatenate([-dist_pos / T, -dist_neg / T], axis=1)
    A = balanced_affinity(logit, iters)
    npos = dist_pos.shape[1]
    P = A[:, :npos].sum(axis=1)
    Q = A[:, npos:].sum(axis=1)
    return float(np.median(P * Q))


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
    gain: str = "paper",
    gain_kwargs: dict | None = None,
) -> list[tuple[int, float]]:
    """Move particles by the (balanced) drift field; return mode-mass error
    trajectory.  Negatives are the particles themselves with the eye mask,
    matching the paper's reuse convention.  `gain`/`gain_kwargs` forward to
    `compute_v_sinkhorn` (default `gain="paper"` is the unmodified estimator)."""
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
        V = compute_v_sinkhorn(
            pts, ypos, pts, T, iters, self_mask=True, gain=gain, gain_kwargs=gain_kwargs
        )
        pts = pts + eta * V
    return traj
