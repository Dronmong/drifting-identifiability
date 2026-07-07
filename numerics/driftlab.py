"""driftlab: numerical companion to the DriftingIdentifiability Lean stack.

Every formula here transcribes a *specific* Lean declaration (named in the
docstrings) or the paper's own pseudo-code (Algorithm 2, arXiv:2602.04770v2).
The point of this module is Objective 7: evaluate whether the machine-checked
conditions are realistic at the paper's actual operating point.

Conventions
-----------
* Data space is R^1 unless stated; norms are absolute values.
* "Normalized units": the paper normalizes features so that the mean pairwise
  distance is sqrt(C) and uses kernel exp(-dist/(tau*sqrt(C))) (eqs. 18-22).
  In units of u = dist/sqrt(C) the kernel is exp(-u/tau); typical u ~ 1.
* Pi-type norms in Lean are sup norms; frame inequalities use ||.||_inf.
"""

from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------


def gaussian_kernel(sigma: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Paper.gaussianKernel: exp(-(1/(2 sigma^2)) |x-y|^2)."""
    return np.exp(-((x - y) ** 2) / (2.0 * sigma**2))


def alg2_kernel(tau: float, dist: np.ndarray) -> np.ndarray:
    """Algorithm2.algorithm2Kernel (= Paper.laplaceKernel): exp(-dist/tau)."""
    return np.exp(-dist / tau)


# ----------------------------------------------------------------------------
# Structured Gaussian empirical family (EmpiricalFrameBound.lean)
# ----------------------------------------------------------------------------


def strict_pairs(m: int) -> list[tuple[int, int]]:
    """StrictPair m, in lexicographic order."""
    return [(i, j) for i in range(m) for j in range(i + 1, m)]


def interaction_matrix(z: np.ndarray, probes: np.ndarray) -> np.ndarray:
    """M[n, p] = U_p(probe n) for the empirical point basis, unit Gaussian.

    Lean: inducedInteractionVector (empiricalFin z)
            (meanShiftInteractionKernel (gaussianKernel 1))
            (empiricalPointDensity z) probes,
    which collapses to U_ij(n) = k(x_n, z_i) k(x_n, z_j) (z_i - z_j).
    """
    pairs = strict_pairs(len(z))
    M = np.empty((len(probes), len(pairs)))
    for p, (i, j) in enumerate(pairs):
        M[:, p] = (
            gaussian_kernel(1.0, probes, z[i])
            * gaussian_kernel(1.0, probes, z[j])
            * (z[i] - z[j])
        )
    return M


def certified_frame_constant(M: np.ndarray) -> float:
    """gaussianEmpiricalPointCertifiedFrameConstant: (sum |M^{-1}|)^{-1}.

    Requires M square (probes = strict pairs).  Returns 0.0 if M is
    numerically singular (the honest report for a collapsed certificate).
    """
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return 0.0
    mass = np.abs(Minv).sum()
    if not np.isfinite(mass) or mass <= 0:
        return 0.0
    return 1.0 / mass


def frame_ceiling(z: np.ndarray) -> float:
    """gaussianEmpiricalPoint_frameConstant_le: c <= min |dz| e^{-dz^2/4}."""
    vals = [
        abs(z[i] - z[j]) * np.exp(-((z[i] - z[j]) ** 2) / 4.0)
        for i, j in strict_pairs(len(z))
    ]
    return float(min(vals))


def frame_violation(M: np.ndarray, c: float, trials: int, rng) -> float:
    """Max violation of  c * ||w||_1 <= ||M w||_inf  over random coefficient
    vectors (InteractionFrameBound, sup norm on the probe axis).

    Returns the largest value of  c*||w||_1 - ||Mw||_inf  seen (should be
    <= ~1e-12 when c is a valid certificate)."""
    worst = -np.inf
    P = M.shape[1]
    for _ in range(trials):
        w = rng.standard_normal(P)
        lhs = c * np.abs(w).sum()
        rhs = np.abs(M @ w).max()
        worst = max(worst, lhs - rhs)
    return float(worst)


# ----------------------------------------------------------------------------
# Two-atom Laplace / column-reweighted class ({0,1} atoms)
# ----------------------------------------------------------------------------


def u01_bare(anchors: np.ndarray, tau: float) -> np.ndarray:
    """Two-atom bare interaction vector (empirical01Laplace family).

    basisInteraction_empirical2 with atoms {0,1}:
      U_01(x) = (1/4)(2*2 - 0*0) * k(x,0) k(x,1) * (0-1) = -k(x,0)k(x,1).
    Returns |U_01| entries; frame constant = sup (Pi norm)."""
    return alg2_kernel(tau, np.abs(anchors - 0.0)) * alg2_kernel(
        tau, np.abs(anchors - 1.0)
    )


def column_mass(anchors: np.ndarray, tau: float, y: np.ndarray) -> np.ndarray:
    """algorithm2ColumnKernelMass: g(y) = sum_r k(x_r, y)."""
    return alg2_kernel(tau, np.abs(anchors[:, None] - y[None, :])).sum(axis=0)


def col_reweight_scale(anchors: np.ndarray, tau: float) -> float:
    """inducedInteractionVector_columnReweighted01_eq_smul scale:
    U^col = (1/sqrt(g(0) g(1))) * U^bare."""
    g = column_mass(anchors, tau, np.array([0.0, 1.0]))
    return float(1.0 / np.sqrt(g[0] * g[1]))


def column_reweighted_weight(
    anchors: np.ndarray, tau: float, i: int, y: np.ndarray
) -> np.ndarray:
    """algorithm2ColumnReweightedWeight: k(x_i,y)/sqrt(g(y))."""
    k = alg2_kernel(tau, np.abs(anchors[i] - y))
    return k / np.sqrt(column_mass(anchors, tau, y))


def modified_field_two_atom(
    anchors: np.ndarray, tau: float, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Column-reweighted population centroid difference at each anchor.

    The limiting no-mask field (ColumnReweightedMeanShift): with atoms {0,1},
    c_p(x_i) = sum_z a_z w_i(z) z / sum_z a_z w_i(z),  V = c_p - c_q."""
    atoms = np.array([0.0, 1.0])
    out = np.empty(len(anchors))
    for i in range(len(anchors)):
        w = column_reweighted_weight(anchors, tau, i, atoms)
        cp = (a * w * atoms).sum() / (a * w).sum()
        cq = (b * w * atoms).sum() / (b * w).sum()
        out[i] = cp - cq
    return out


def bare_field_two_atom(
    anchors: np.ndarray, tau: float, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Paper.meanShiftDrift (eqs. 8/10) with the bare kernel, two atoms."""
    atoms = np.array([0.0, 1.0])
    out = np.empty(len(anchors))
    for i in range(len(anchors)):
        k = alg2_kernel(tau, np.abs(anchors[i] - atoms))
        cp = (a * k * atoms).sum() / (a * k).sum()
        cq = (b * k * atoms).sum() / (b * k).sum()
        out[i] = cp - cq
    return out


# ----------------------------------------------------------------------------
# Algorithm 2 (two independent implementations)
# ----------------------------------------------------------------------------


def compute_v_paper(
    x: np.ndarray,
    y_pos: np.ndarray,
    y_neg: np.ndarray,
    T: float,
    self_mask: bool,
) -> np.ndarray:
    """Verbatim vectorized port of the paper's Algorithm 2 pseudo-code.

    x: [N] (1-D data), y_pos: [Npos], y_neg: [Nneg].  self_mask adds
    eye(N)*1e6 to dist_neg (requires N == Nneg, the reuse pattern)."""
    dist_pos = np.abs(x[:, None] - y_pos[None, :])
    dist_neg = np.abs(x[:, None] - y_neg[None, :])
    if self_mask:
        assert len(x) == len(y_neg)
        dist_neg = dist_neg + np.eye(len(x)) * 1e6
    logit = np.concatenate([-dist_pos / T, -dist_neg / T], axis=1)
    # softmax over samples (dim -1), shift-invariant per row
    lr = logit - logit.max(axis=1, keepdims=True)
    er = np.exp(lr)
    A_row = er / er.sum(axis=1, keepdims=True)
    # softmax over anchors (dim -2), shift-invariant per column
    lc = logit - logit.max(axis=0, keepdims=True)
    ec = np.exp(lc)
    A_col = ec / ec.sum(axis=0, keepdims=True)
    A = np.sqrt(A_row * A_col)
    npos = y_pos.shape[0]
    A_pos, A_neg = A[:, :npos], A[:, npos:]
    W_pos = A_pos * A_neg.sum(axis=1, keepdims=True)
    W_neg = A_neg * A_pos.sum(axis=1, keepdims=True)
    return W_pos @ y_pos - W_neg @ y_neg


def compute_v_lean(
    x: np.ndarray,
    y_pos: np.ndarray,
    y_neg: np.ndarray,
    T: float,
    mask: np.ndarray,
) -> np.ndarray:
    """Literal transcription of Paper.algorithm2Drift (finiteSoftmax pipeline).

    mask: bool [N, Nneg] (selfMask).  Element-wise, no max-shift tricks, so it
    doubles as an independent numerical audit of compute_v_paper."""
    N, npos, nneg = len(x), len(y_pos), len(y_neg)
    logit = np.empty((N, npos + nneg))
    for i in range(N):
        for j in range(npos):
            logit[i, j] = -abs(x[i] - y_pos[j]) / T
        for l in range(nneg):
            pen = 1e6 if mask[i, l] else 0.0
            logit[i, npos + l] = -(abs(x[i] - y_neg[l]) + pen) / T
    e = np.exp(logit)
    A_row = e / e.sum(axis=1, keepdims=True)  # finiteSoftmax over samples
    A_col = e / e.sum(axis=0, keepdims=True)  # finiteSoftmax over anchors
    A = np.sqrt(A_row * A_col)  # algorithm2Affinity
    A_pos, A_neg = A[:, :npos], A[:, npos:]
    W_pos = A_pos * A_neg.sum(axis=1, keepdims=True)  # algorithm2PositiveWeight
    W_neg = A_neg * A_pos.sum(axis=1, keepdims=True)  # algorithm2NegativeWeight
    return W_pos @ y_pos - W_neg @ y_neg


def centroid_diff(
    x: np.ndarray,
    y_pos: np.ndarray,
    y_neg: np.ndarray,
    T: float,
    self_mask: bool,
) -> np.ndarray:
    """Algorithm2.noMaskCentroidDrift / masked analogue: C_pos - C_neg.

    Uses the affinity pipeline; drift = P*Q*(C_pos - C_neg)
    (algorithm2Drift_eq_massProduct_centroidDiff)."""
    dist_pos = np.abs(x[:, None] - y_pos[None, :])
    dist_neg = np.abs(x[:, None] - y_neg[None, :])
    if self_mask:
        dist_neg = dist_neg + np.eye(len(x)) * 1e6
    logit = np.concatenate([-dist_pos / T, -dist_neg / T], axis=1)
    lr = logit - logit.max(axis=1, keepdims=True)
    er = np.exp(lr)
    A_row = er / er.sum(axis=1, keepdims=True)
    lc = logit - logit.max(axis=0, keepdims=True)
    ec = np.exp(lc)
    A_col = ec / ec.sum(axis=0, keepdims=True)
    A = np.sqrt(A_row * A_col)
    npos = y_pos.shape[0]
    A_pos, A_neg = A[:, :npos], A[:, npos:]
    P = A_pos.sum(axis=1)
    Q = A_neg.sum(axis=1)
    c_pos = (A_pos @ y_pos) / P
    c_neg = (A_neg @ y_neg) / Q
    return c_pos - c_neg


# ----------------------------------------------------------------------------
# Sampling and diagnostics
# ----------------------------------------------------------------------------


def sample_two_atom(coeffs: np.ndarray, n: int, rng) -> np.ndarray:
    """iid draws from the two-atom mixture on {0,1}."""
    return (rng.random(n) < coeffs[1]).astype(float)


def ess(weights: np.ndarray) -> float:
    """Effective sample size of importance weights: (sum w)^2 / sum w^2."""
    s = weights.sum()
    q = (weights**2).sum()
    return float(s * s / q) if q > 0 else 0.0


def row_softmax_ess(dists: np.ndarray, tau: float) -> np.ndarray:
    """Per-anchor ESS of the row softmax over a distance matrix [N, M]."""
    logit = -dists / tau
    logit = logit - logit.max(axis=1, keepdims=True)
    w = np.exp(logit)
    return np.array([ess(w[i]) for i in range(w.shape[0])])
