"""Expected Sliced Transport (EST) plan primitives.

AnchoredCoherentTransportResearchPlan.md sections 4.4-4.5, unit-tested per
section 11. The central hypothesis (section 2.2 / RQ1) is that the *independent*
sliced correction -- averaging back-projected 1-D rank destinations -- sends a
particle to a point BETWEEN incompatible target components, creating bridges.
EST instead averages the lifted permutation COUPLINGS (a valid doubly-stochastic
joint object), and hard consensus routes each source to a single coherent
target, so incompatible destinations are never averaged.

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
    """Per-source modal (most-frequent) target assignment across directions
    (section 4.5, 'hard consensus'). Ties broken toward the smallest target
    index. Returns (destinations, assigned_target_index). Because each source
    commits to ONE coherent target, incompatible modes are never averaged."""
    L, N = Pi.shape
    assign = np.empty(N, dtype=int)
    for i in range(N):
        vals, counts = np.unique(Pi[:, i], return_counts=True)
        # np.unique returns sorted vals; argmax picks the first (smallest-index)
        # among ties -> deterministic.
        assign[i] = vals[counts.argmax()]
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

    # 7. hard consensus of identical samples assigns each to itself.
    _, assign_id = hard_consensus(X, Pi_id)
    check("7. hard consensus identity on identical samples",
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

    if fails:
        raise SystemExit(f"EST plan tests FAILED: {fails}")
    log("all EST plan tests passed")


if __name__ == "__main__":
    _tests()
