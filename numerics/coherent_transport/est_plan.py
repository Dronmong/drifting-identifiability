"""Expected Sliced Transport (EST) plan primitives.

AnchoredCoherentTransportResearchPlan.md sections 4.4-4.5, unit-tested per
section 11. The central hypothesis (section 2.2 / RQ1) is that the *independent*
sliced correction -- averaging back-projected 1-D rank destinations -- sends a
particle to a point BETWEEN incompatible target components, creating bridges.
EST instead averages the lifted permutation COUPLINGS (a valid doubly-stochastic
joint object). A row-wise modal route sends each source to a single target, but
does NOT in general preserve the target marginal. A balanced EST assignment
selects a permutation from the sparse EST support and therefore preserves both
empirical marginals without averaging incompatible target identities.

Reference: Expected Sliced Transport Plans (ICLR 2025). Pure numpy.

Conventions: `X` generated (N,d), `Y` target (N,d) with matched counts (the
coupling is a permutation coupling, so |X|=|Y|). Directions `U` are (L,d) unit
rows. Ties in projection sorts are broken by a stable sort on (value, index) so
every result is deterministic.
"""

from __future__ import annotations

import numpy as np


def unit_directions(L: int, d: int, seed: int) -> np.ndarray:
    """L deterministic random unit directions in R^d."""
    g = np.random.default_rng(seed).standard_normal((L, d))
    return g / np.linalg.norm(g, axis=1, keepdims=True)


def _stable_argsort(v: np.ndarray) -> np.ndarray:
    # Lexicographic (value, index) so equal projections order by original index.
    return np.lexsort((np.arange(len(v)), v))


def rank_match(px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """1-D rank-matching permutation for one direction: returns `pi` with
    `pi[i]` = index of the target point matched to generated point `i` (the
    target at the same sorted rank)."""
    gs = _stable_argsort(px)
    ts = _stable_argsort(py)
    pi = np.empty(len(px), dtype=int)
    pi[gs] = ts
    return pi


def all_rank_matches(X: np.ndarray, Y: np.ndarray, U: np.ndarray) -> np.ndarray:
    """`Pi` of shape (L, N): rank-matching permutation per direction."""
    PX = X @ U.T   # (N, L)
    PY = Y @ U.T
    L = U.shape[0]
    Pi = np.empty((L, len(X)), dtype=int)
    for l in range(L):
        Pi[l] = rank_match(PX[:, l], PY[:, l])
    return Pi


def est_coupling(Pi: np.ndarray, N: int) -> np.ndarray:
    """Averaged lifted coupling Gamma_EST (N,N): (1/L) sum_l (1/N) P_{pi_l}.
    Row and column sums are exactly 1/N (valid doubly-stochastic/N object)."""
    L = Pi.shape[0]
    G = np.zeros((N, N))
    for l in range(L):
        G[np.arange(N), Pi[l]] += 1.0
    return G / (L * N)


def est_barycenter(Y: np.ndarray, Pi: np.ndarray) -> np.ndarray:
    """Barycentric destination T_i = (1/L) sum_l y_{pi_l(i)} (section 4.4).
    Required diagnostic arm; can still average disconnected modes."""
    return Y[Pi].mean(axis=0)


def hard_consensus(Y: np.ndarray, Pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible name for :func:`modal_route`.

    This is a deterministic single-destination route, not a balanced coupling.
    Callers must not claim that it preserves the target empirical marginal.
    """
    return modal_route(Y, Pi)


def modal_route(Y: np.ndarray, Pi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-source modal target across sliced permutation plans.

    Ties are broken toward the smallest target index. Every source commits to
    one observed target, so target identities are not barycentrically averaged.
    Several sources may select the same target, however, and some targets may
    receive no source. This route is therefore not a transport *coupling* with
    the target marginal.
    """
    L, N = Pi.shape
    assign = np.empty(N, dtype=int)
    for i in range(N):
        vals, counts = np.unique(Pi[:, i], return_counts=True)
        # np.unique returns sorted vals; argmax picks the first (smallest-index)
        # among ties -> deterministic.
        assign[i] = vals[counts.argmax()]
    return Y[assign], assign


def match_counts(Pi: np.ndarray, N: int | None = None) -> np.ndarray:
    """Number of sliced rank plans using each source-target edge."""
    if Pi.ndim != 2:
        raise ValueError("Pi must have shape (L, N)")
    L, n_source = Pi.shape
    n_target = n_source if N is None else int(N)
    if n_target != n_source:
        raise ValueError("balanced EST currently requires equal source/target counts")
    if Pi.size and (Pi.min() < 0 or Pi.max() >= n_target):
        raise ValueError("Pi contains an out-of-range target index")
    counts = np.zeros((n_source, n_target), dtype=np.int32)
    for ell in range(L):
        counts[np.arange(n_source), Pi[ell]] += 1
    return counts


def target_marginal_l1(assign: np.ndarray, N: int | None = None) -> float:
    """L1 error of a deterministic assignment's target marginal from uniform."""
    n = len(assign) if N is None else int(N)
    counts = np.bincount(np.asarray(assign, dtype=int), minlength=n)
    return float(np.abs(counts / len(assign) - 1.0 / n).sum())


def balanced_est_assignment(
    Y: np.ndarray,
    Pi: np.ndarray,
    *,
    method: str = "sparse",
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-consensus balanced assignment supported by the EST graph.

    The support is the union of the sliced permutation plans. It always
    contains a perfect matching because it contains every constituent
    permutation. The objective minimizes ``L - count(i,j)`` over supported
    edges, equivalently maximizing total sliced agreement.

    ``method="dense"`` uses the dense Hungarian solver. ``method="sparse"``
    uses SciPy's sparse full-bipartite matching routine and stores only the
    at-most ``L*N`` observed EST edges. Both return a permutation assignment,
    hence exact empirical source and target marginals.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    L, N = Pi.shape
    if len(Y) != N:
        raise ValueError("balanced EST requires len(Y) == Pi.shape[1]")
    counts = match_counts(Pi, N)
    support = counts > 0

    if method == "dense":
        # Any supported perfect matching has total cost at most L*N. Make one
        # unsupported edge more expensive than that entire feasible matching.
        forbidden = L * N + 1
        cost = np.where(support, L - counts, forbidden)
        rows, cols = linear_sum_assignment(cost)
    elif method == "sparse":
        rows0, cols0 = np.nonzero(support)
        # Sparse graph routines discard explicit zero weights. Add one to keep
        # even unanimous (cost-zero) supported edges in the graph.
        weights = (L - counts[rows0, cols0] + 1).astype(float)
        graph = csr_matrix((weights, (rows0, cols0)), shape=(N, N))
        rows, cols = min_weight_full_bipartite_matching(graph)
    else:
        raise ValueError("method must be 'dense' or 'sparse'")

    assign = np.empty(N, dtype=int)
    assign[rows] = cols
    if not np.array_equal(np.sort(assign), np.arange(N)):
        raise AssertionError("balanced EST solver did not return a permutation")
    if not np.all(support[np.arange(N), assign]):
        raise AssertionError("balanced EST solver used an edge outside EST support")
    return Y[assign], assign


def balanced_hybrid_assignment(
    X: np.ndarray,
    Y: np.ndarray,
    Pi: np.ndarray,
    *,
    agreement_weight: float,
    method: str = "sparse",
) -> tuple[np.ndarray, np.ndarray]:
    """Balanced EST-support assignment with distance/consensus tradeoff.

    On every supported edge ``(i,j)`` the dimensionless cost is

    ``||x_i-y_j||^2 / median_supported_distance^2
       - agreement_weight * log(count(i,j)/L)``.

    ``agreement_weight=0`` is the shortest balanced route restricted to EST
    support. Larger values prefer edges proposed by more sliced plans. The
    assignment remains a permutation on EST support for every nonnegative
    finite weight.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    if (
        not np.isfinite(agreement_weight)
        or agreement_weight < 0
    ):
        raise ValueError("agreement_weight must be finite and nonnegative")
    L, N = Pi.shape
    if X.shape != Y.shape or len(X) != N:
        raise ValueError("X, Y, and Pi must have matched equal populations")
    counts = match_counts(Pi, N)
    support = counts > 0
    rows0, cols0 = np.nonzero(support)
    squared = ((X[rows0] - Y[cols0]) ** 2).sum(axis=1)
    positive = squared[squared > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    scale = max(scale, np.finfo(float).tiny)
    agreement = counts[rows0, cols0].astype(float) / L
    edge_cost = squared / scale - agreement_weight * np.log(agreement)

    if method == "dense":
        # A supported perfect matching exists. Choose a forbidden cost greater
        # than the worst possible sum of N supported edge costs.
        forbidden = N * (float(edge_cost.max()) + 1.0)
        cost = np.full((N, N), forbidden)
        cost[rows0, cols0] = edge_cost
        rows, cols = linear_sum_assignment(cost)
    elif method == "sparse":
        # Sparse matching drops explicit zeros, so shift all observed weights
        # by one. A constant per selected edge does not change the optimizer.
        graph = csr_matrix(
            (edge_cost + 1.0, (rows0, cols0)), shape=(N, N))
        rows, cols = min_weight_full_bipartite_matching(graph)
    else:
        raise ValueError("method must be 'dense' or 'sparse'")

    assign = np.empty(N, dtype=int)
    assign[rows] = cols
    if not np.array_equal(np.sort(assign), np.arange(N)):
        raise AssertionError("hybrid EST solver did not return a permutation")
    if not np.all(support[np.arange(N), assign]):
        raise AssertionError("hybrid EST solver used an unsupported edge")
    return Y[assign], assign


def consensus_fraction(Pi: np.ndarray, assign: np.ndarray) -> np.ndarray:
    """Fraction of directions agreeing with the consensus target per source.
    Low values flag sources whose slices disagree (the bridge-prone particles)."""
    L = Pi.shape[0]
    return (Pi == assign[None, :]).mean(axis=0)


# ---------------------------------------------------------------------------
# Property tests (research plan section 11: EST plan)
# ---------------------------------------------------------------------------


def _tests(log=print) -> None:
    rng = np.random.default_rng(7)
    fails = []

    def check(name, ok):
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails.append(name)

    N, d, L = 40, 3, 24
    X = rng.normal(size=(N, d))
    Y = rng.normal(size=(N, d)) + np.array([2.0, 0.0, -1.0])
    U = unit_directions(L, d, seed=1)
    Pi = all_rank_matches(X, Y, U)

    # 1. every rank plan is a permutation.
    check("1. each direction's match is a permutation",
          all(np.array_equal(np.sort(Pi[l]), np.arange(N)) for l in range(L)))

    # 2. row and column masses equal 1/N.
    G = est_coupling(Pi, N)
    check("2. coupling row/col masses == 1/N",
          np.allclose(G.sum(1), 1.0 / N) and np.allclose(G.sum(0), 1.0 / N))

    # 3. averaged plan preserves both marginals (== test 2 for the average; also
    #    total mass 1).
    check("3. total mass 1 and doubly stochastic/N",
          abs(G.sum() - 1.0) < 1e-12 and np.all(G >= 0))

    # 4. identical source and target admit the identity plan.
    Pi_id = all_rank_matches(X, X, U)
    check("4. identical samples -> identity plan (every direction)",
          all(np.array_equal(Pi_id[l], np.arange(N)) for l in range(L)))

    # 5. deterministic under the frozen tie rule (ties present).
    Xt = np.zeros((6, 2)); Yt = np.zeros((6, 2))
    Xt[:, 0] = [0, 0, 0, 1, 1, 1]; Yt[:, 0] = [0, 0, 0, 1, 1, 1]
    Ut = unit_directions(5, 2, seed=3)
    a = all_rank_matches(Xt, Yt, Ut)
    b = all_rank_matches(Xt, Yt, Ut)
    check("5. deterministic under tie rule", np.array_equal(a, b))

    # 6. barycenter of identical samples is the identity (T_i == x_i).
    check("6. barycenter of matched identical samples == points",
          np.allclose(est_barycenter(X, Pi_id), X))

    # 7. modal route of identical samples assigns each to itself.
    _, assign_id = hard_consensus(X, Pi_id)
    check("7. modal route identity on identical samples",
          np.array_equal(assign_id, np.arange(N)))

    # 8. consensus separates coherent from incompatible: two far-apart clusters,
    #    matched cluster-to-cluster, give consensus fraction ~1; barycenter of a
    #    source whose slices split across both clusters lands between them.
    A = rng.normal(size=(N // 2, 2)) * 0.05 + np.array([-5.0, 0])
    B = rng.normal(size=(N // 2, 2)) * 0.05 + np.array([5.0, 0])
    Xc = np.vstack([A, B]); Yc = np.vstack([A, B]).copy()
    Uc = unit_directions(40, 2, seed=9)
    Pic = all_rank_matches(Xc, Yc, Uc)
    _, ac = hard_consensus(Yc, Pic)
    cf = consensus_fraction(Pic, ac)
    check("8. coherent matched clusters -> high consensus fraction",
          float(cf.mean()) > 0.9)

    # 9. Row-wise modal routing need not preserve the target marginal.
    Pi_collision = np.array([
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
    ])
    Y3 = np.arange(3.0)[:, None]
    _, modal = modal_route(Y3, Pi_collision)
    check("9. modal route is correctly detected as unbalanced",
          target_marginal_l1(modal) > 0)

    # 10. Dense and sparse balanced assignments are permutations on EST
    # support and attain the same maximum-consensus objective.
    _, ad = balanced_est_assignment(Y, Pi, method="dense")
    _, ass = balanced_est_assignment(Y, Pi, method="sparse")
    counts = match_counts(Pi)
    check("10. dense/sparse balanced EST preserve target marginal",
          target_marginal_l1(ad) == 0.0 and target_marginal_l1(ass) == 0.0)
    check("11. dense/sparse balanced EST stay on EST support",
          np.all(counts[np.arange(N), ad] > 0) and
          np.all(counts[np.arange(N), ass] > 0))
    check("12. dense/sparse balanced EST have equal consensus objective",
          int(counts[np.arange(N), ad].sum()) ==
          int(counts[np.arange(N), ass].sum()))

    # 13. Hybrid assignments preserve marginals/support for several weights.
    hybrid_ok = True
    for weight in (0.0, 0.1, 1.0, 10.0):
        _, ah = balanced_hybrid_assignment(
            X, Y, Pi, agreement_weight=weight, method="sparse")
        hybrid_ok &= (
            target_marginal_l1(ah) == 0.0
            and np.all(counts[np.arange(N), ah] > 0)
        )
    check("13. hybrid EST family preserves marginals and support", hybrid_ok)

    # 14. Dense and sparse hybrid solvers attain the same declared objective.
    weight = 0.25
    _, hd = balanced_hybrid_assignment(
        X, Y, Pi, agreement_weight=weight, method="dense")
    _, hs = balanced_hybrid_assignment(
        X, Y, Pi, agreement_weight=weight, method="sparse")
    squared_all = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)
    support_sq = squared_all[counts > 0]
    pos = support_sq[support_sq > 0]
    scale = float(np.median(pos)) if len(pos) else 1.0

    def hybrid_objective(a):
        return float(np.sum(
            squared_all[np.arange(N), a] / scale
            - weight * np.log(counts[np.arange(N), a] / L)
        ))

    check("14. dense/sparse hybrid EST have equal objective",
          abs(hybrid_objective(hd) - hybrid_objective(hs)) < 1e-8)

    if fails:
        raise SystemExit(f"EST plan tests FAILED: {fails}")
    log("all EST plan tests passed")


if __name__ == "__main__":
    _tests()
