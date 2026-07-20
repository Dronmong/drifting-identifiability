"""NCJ field library (IdentifiabilityDrivenImprovementPlan.md, step 2).

Fresh namespace for the identifiability-driven improvement program.  Historical
artifacts (lowdim_*.py runs, driftbench*.py, atlas) are untouched audit trail.

This module holds the *field layer* only:

* the exact dimension-general row/column bi-softmax Algorithm-2 affinity,
  generalized to an arbitrary independent negative batch, written so that the
  paper-gain path with reused/masked negatives is BITWISE identical to the
  audited `lowdim_drift.drift_paper`;
* the audited factorization `Vpaper_i = (P_i * Q_i) * Delta_i`
  (Algorithm2Estimator.lean: `algorithm2Drift_eq_massProduct_centroidDiff`)
  exposed as diagnostics `P, Q, Cpos, Cneg, Delta, ESS`;
* gain modes: "paper" (exact `Wp@pos - Wn@neg`, never divides by mass),
  "constant" (`Delta` computed directly from the two centroids, never divides
  the raw drift by a tiny `P*Q`), "power" (`(P*Q)**gamma * Delta`, diagnostic);
* symmetric Gaussian jitter with three independent, seed-reproducible streams
  (query/positive/negative), `sigma = 0` bitwise-skipped;
* strict non-finiteness handling: degenerate rows (underflowed masses) either
  raise or are zeroed WITH an explicit count in the result -- never silent.

The training loop, registries, and protocol live in
`run_identifiability_improvement.py` / `IdentifiabilityImprovementProtocol.md`.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import norm

from lowdim_drift import WorkCounter, drift_paper

GAINS = ("paper", "constant", "power", "abc")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FieldResult:
    """One field evaluation with matched-compute accounting.

    `V` is aligned with the query index.  When jitter is active the field is
    evaluated at the jittered queries but indexed by (and applied to) the
    original queries: the jitter is an additive perturbation with identity
    derivative in a differentiable implementation (plan section 2.4).
    """

    V: np.ndarray
    kernel_pairs: int
    wall_time: float
    jitter_sigma: float = 0.0
    jitter_seed: int | None = None
    n_degenerate_rows: int = 0
    diagnostics: dict | None = None


class DegenerateFieldError(RuntimeError):
    """Raised when affinity masses underflow or inputs are non-finite."""


# ---------------------------------------------------------------------------
# Core affinity (exact replica of the audited bi-softmax, general negatives)
# ---------------------------------------------------------------------------


def biaffinity(queries: np.ndarray, positives: np.ndarray,
               negatives: np.ndarray, tau: float,
               mask: bool) -> tuple[np.ndarray, np.ndarray]:
    """Row/column bi-softmax affinity split into positive/negative blocks.

    Operation-for-operation identical to `lowdim_drift.drift_paper` up to the
    `A = sqrt(row * col)` line, with the reused cloud generalized to an
    arbitrary `negatives` batch.  `mask=True` requires the paper reuse shape
    (one negative per query) and suppresses the paired self entry.
    """
    if mask and len(negatives) != len(queries):
        raise ValueError("eye mask requires len(negatives) == len(queries)")
    dp = norm(queries[:, None, :] - positives[None, :, :], axis=2)
    dn = norm(queries[:, None, :] - negatives[None, :, :], axis=2)
    logits = np.concatenate([-dp / tau, -dn / tau], axis=1)
    if mask:
        npos = len(positives)
        ii = np.arange(len(queries))
        logits[ii, npos + ii] -= 1e6 / tau
    row = np.exp(logits - logits.max(axis=1, keepdims=True))
    row /= row.sum(axis=1, keepdims=True)
    col = np.exp(logits - logits.max(axis=0, keepdims=True))
    col /= col.sum(axis=0, keepdims=True)
    A = np.sqrt(row * col)
    return A[:, : len(positives)], A[:, len(positives):]


def _jitter_streams(shapes: list[tuple[int, ...]], sigma: float,
                    seed: int) -> list[np.ndarray]:
    """Three independent standard-normal streams, reproducible from `seed`."""
    children = np.random.SeedSequence(seed).spawn(len(shapes))
    return [sigma * np.random.default_rng(c).standard_normal(s)
            for c, s in zip(children, shapes)]


# ---------------------------------------------------------------------------
# Unified field evaluation
# ---------------------------------------------------------------------------


def compute_field(queries: np.ndarray, positives: np.ndarray,
                  negatives: np.ndarray | None = None, *, tau: float,
                  gain: str = "paper", gamma: float = 1.0,
                  mask: bool = False, jitter_sigma: float = 0.0,
                  jitter_seed: int | None = None,
                  counter: WorkCounter | None = None,
                  want_diagnostics: bool = False,
                  on_degenerate: str = "raise") -> FieldResult:
    """Evaluate one drifting field.

    `negatives=None` selects the paper reuse pattern (queries as negatives).
    `gain`: "paper" = exact Algorithm-2 field (never divides by mass);
    "constant" = `Cpos - Cneg` computed directly from the centroids;
    "power" = `(P*Q)**gamma * Delta` (diagnostic bridge, gamma=1 ~ paper).
    `jitter_sigma > 0` requires `jitter_seed`; sigma = 0 draws nothing so the
    result is bitwise identical to the corresponding no-jitter arm.
    """
    if gain not in GAINS:
        raise ValueError(f"unknown gain {gain!r}")
    if on_degenerate not in ("raise", "zero"):
        raise ValueError(f"unknown on_degenerate {on_degenerate!r}")
    t0 = time.perf_counter()
    reuse = negatives is None
    neg = queries if reuse else negatives
    for name, arr in (("queries", queries), ("positives", positives),
                      ("negatives", neg)):
        if not np.all(np.isfinite(arr)):
            raise DegenerateFieldError(f"non-finite input in {name}")

    xq, pos = queries, positives
    if jitter_sigma < 0:
        raise ValueError("jitter_sigma must be >= 0")
    if jitter_sigma > 0:
        if jitter_seed is None:
            raise ValueError("jitter_sigma > 0 requires jitter_seed")
        eq, ep, en = _jitter_streams(
            [queries.shape, positives.shape, neg.shape], jitter_sigma,
            jitter_seed)
        xq, pos, neg = queries + eq, positives + ep, neg + en

    if counter is not None:
        counter.add_field(len(xq), len(pos), len(neg))
    kernel_pairs = len(xq) * (len(pos) + len(neg))

    Ap, An = biaffinity(xq, pos, neg, tau, mask)
    P = Ap.sum(axis=1)
    Q = An.sum(axis=1)
    bad = ~(np.isfinite(P) & np.isfinite(Q) & (P > 0.0) & (Q > 0.0))
    n_bad = int(bad.sum())
    if n_bad and on_degenerate == "raise":
        raise DegenerateFieldError(
            f"{n_bad} degenerate affinity rows (underflowed or non-finite "
            "mass); rerun with on_degenerate='zero' to zero and count them")
    if n_bad:
        warnings.warn(f"zeroed {n_bad} degenerate affinity rows",
                      RuntimeWarning, stacklevel=2)
    Psafe = np.where(bad, 1.0, P)[:, None]
    Qsafe = np.where(bad, 1.0, Q)[:, None]

    if gain == "paper":
        # Exact audited op order: bitwise-identical to drift_paper.
        Wp = Ap * An.sum(axis=1, keepdims=True)
        Wn = An * Ap.sum(axis=1, keepdims=True)
        V = Wp @ pos - Wn @ neg
        if n_bad:
            V[bad] = 0.0
    else:
        Cpos = (Ap / Psafe) @ pos
        Cneg = (An / Qsafe) @ neg
        if gain == "abc":
            # Standard first-order analytical bias correction (ABC) of the
            # self-normalized minibatch centroid.  With normalized weights
            # a_j = A_j / sum_k A_k, the leading O(1/N) SNIS ratio bias is
            # -sum_j a_j^2 (y_j - Chat) (Kong-Liu-Wong self-normalized ratio
            # expansion), so the corrected centroid adds sum_j a_j^2 (y_j - C).
            # This is independently derived, NOT a verified reproduction of the
            # post-cutoff arXiv:2604.27239; it is the textbook SNIS correction.
            ap = Ap / Psafe
            an = An / Qsafe
            Cpos = Cpos + (ap ** 2) @ pos - \
                (ap ** 2).sum(axis=1, keepdims=True) * Cpos
            Cneg = Cneg + (an ** 2) @ neg - \
                (an ** 2).sum(axis=1, keepdims=True) * Cneg
        Delta = Cpos - Cneg
        if n_bad:
            Delta[bad] = 0.0
        if gain in ("constant", "abc"):
            V = Delta
        else:
            V = ((P * Q) ** gamma)[:, None] * Delta

    if not np.all(np.isfinite(V)):
        raise DegenerateFieldError("non-finite drift output")

    diag = None
    if want_diagnostics:
        Cpos = (Ap / Psafe) @ pos
        Cneg = (An / Qsafe) @ neg
        w2p = (Ap ** 2).sum(axis=1)
        w2n = (An ** 2).sum(axis=1)
        diag = {
            "P": P, "Q": Q, "PQ": P * Q,
            "Cpos": Cpos, "Cneg": Cneg, "Delta": Cpos - Cneg,
            "ESSpos": np.where(w2p > 0, P ** 2 / np.where(w2p > 0, w2p, 1.0),
                               0.0),
            "ESSneg": np.where(w2n > 0, Q ** 2 / np.where(w2n > 0, w2n, 1.0),
                               0.0),
            "field_norm": norm(V, axis=1),
            "delta_norm": norm(Cpos - Cneg, axis=1),
            "gain_mode": gain,
        }
        if reuse and not mask:
            ii = np.arange(len(xq))
            diag["self_leverage"] = An[ii, ii] / np.where(Q > 0, Q, 1.0)
        if jitter_sigma > 0:
            diag["x_tilde"] = xq

    return FieldResult(V=V, kernel_pairs=kernel_pairs,
                       wall_time=time.perf_counter() - t0,
                       jitter_sigma=jitter_sigma, jitter_seed=jitter_seed,
                       n_degenerate_rows=n_bad, diagnostics=diag)


# ---------------------------------------------------------------------------
# Plan-facing wrappers (section 4 API)
# ---------------------------------------------------------------------------


def compute_paper_field(queries, positives, negatives=None, *, tau,
                        mask=True, **kw) -> FieldResult:
    """Exact Algorithm-2 field.  Default = paper reuse pattern with eye mask."""
    return compute_field(queries, positives, negatives, tau=tau, gain="paper",
                         mask=mask, **kw)


def compute_centroid_field(queries, positives, negatives=None, *, tau,
                           gain="constant", mask=False, **kw) -> FieldResult:
    """Normalized field `g * Delta` (version 1: constant gain g = 1)."""
    return compute_field(queries, positives, negatives, tau=tau, gain=gain,
                         mask=mask, **kw)


def compute_ncj_field(queries, positives, negatives, *, tau, jitter_sigma,
                      jitter_seed=None, gain="constant", **kw) -> FieldResult:
    """NCJ field: constant gain + cross-fitted negatives + symmetric jitter.

    `negatives` is REQUIRED and must be an independent model-reference batch;
    there is no eye mask and no index-dependent diagonal operation.
    """
    if negatives is None:
        raise ValueError("cross-fitted mode requires an independent negatives "
                         "batch")
    if kw.pop("mask", False):
        raise ValueError("cross-fitted mode admits no eye mask")
    return compute_field(queries, positives, negatives, tau=tau, gain=gain,
                         mask=False, jitter_sigma=jitter_sigma,
                         jitter_seed=jitter_seed, **kw)


# ---------------------------------------------------------------------------
# Invariant tests (plan section 4, items 1-9)
# ---------------------------------------------------------------------------


def invariant_tests(log=print) -> None:
    rng = np.random.default_rng(20260719)
    fails: list[str] = []

    def check(name: str, ok: bool) -> None:
        log(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            fails.append(name)

    q = rng.normal(size=(24, 2))
    data = rng.normal(size=(32, 2)) + np.array([1.5, -0.5])
    ref = rng.normal(size=(24, 2)) * 1.1
    tau = 0.35

    # 1. gain="paper" reproduces the audited historical field bitwise
    #    (reuse pattern, both mask settings).
    for m in (True, False):
        got = compute_paper_field(q, data, tau=tau, mask=m).V
        ref_v = drift_paper(q, data, tau, m)
        check(f"1. paper gain bitwise == drift_paper (mask={m})",
              np.array_equal(got, ref_v))

    # 2. gain="power", gamma=1 reproduces the paper gain (tolerance level:
    #    same value, different audited op order).
    a = compute_field(q, data, ref, tau=tau, gain="paper").V
    b = compute_field(q, data, ref, tau=tau, gain="power", gamma=1.0).V
    check("2. power(gamma=1) == paper gain (rtol 1e-9)",
          np.allclose(a, b, rtol=1e-9, atol=1e-12))

    # 3. Constant gain returns exactly Cpos - Cneg.
    r3 = compute_field(q, data, ref, tau=tau, gain="constant",
                       want_diagnostics=True)
    check("3. constant gain == Cpos - Cneg (bitwise)",
          np.array_equal(r3.V, r3.diagnostics["Delta"]))

    # 4. Identical positive/negative batches, unmasked -> zero for every gain.
    same = rng.normal(size=(20, 2))
    for g in GAINS:
        v = compute_field(q, same, same.copy(), tau=tau, gain=g).V
        check(f"4. matched batches -> zero field (gain={g})",
              float(np.abs(v).max()) < 1e-12)

    # 4b. ABC bias correction: identical unmasked batches still cancel (the
    #     masked case intentionally does not -- that residual is the self-mask
    #     distortion ABC/cross-fit exist to remove); the correction vanishes as
    #     weights flatten (large tau) and is nonzero for sharp weights.
    v = compute_field(q, same, same.copy(), tau=tau, gain="abc", mask=False).V
    check("4b. ABC identical unmasked batches -> zero",
          float(np.abs(v).max()) < 1e-12)
    v_abc_flat = compute_field(q, data, ref, tau=50.0, gain="abc").V
    v_const_flat = compute_field(q, data, ref, tau=50.0, gain="constant").V
    check("4c. ABC -> constant gain as weights flatten (large tau)",
          np.allclose(v_abc_flat, v_const_flat, atol=2e-3))
    v_abc_sharp = compute_field(q, data, ref, tau=0.15, gain="abc").V
    v_const_sharp = compute_field(q, data, ref, tau=0.15, gain="constant").V
    check("4d. ABC differs from constant under sharp weights",
          not np.allclose(v_abc_sharp, v_const_sharp, atol=1e-6))

    # 5. Cross-fitted mode has no index-dependent diagonal operation:
    #    permuting the negative batch leaves the field unchanged, and
    #    requesting a mask is rejected.
    perm = rng.permutation(len(ref))
    v1 = compute_ncj_field(q, data, ref, tau=tau, jitter_sigma=0.0).V
    v2 = compute_ncj_field(q, data, ref[perm], tau=tau, jitter_sigma=0.0).V
    check("5a. negative-batch permutation invariance",
          np.allclose(v1, v2, rtol=1e-9, atol=1e-12))
    try:
        compute_ncj_field(q, data, ref, tau=tau, jitter_sigma=0.0, mask=True)
        check("5b. cross-fitted mask rejected", False)
    except ValueError:
        check("5b. cross-fitted mask rejected", True)

    # 6. sigma=0 is bitwise identical to the no-jitter arm.
    v0 = compute_ncj_field(q, data, ref, tau=tau, jitter_sigma=0.0,
                           jitter_seed=7).V
    vn = compute_centroid_field(q, data, ref, tau=tau).V
    check("6. sigma=0 bitwise == no-jitter arm", np.array_equal(v0, vn))

    # 7. Jitter streams are reproducible from the seed and mutually
    #    independent (three distinct streams; different seeds differ).
    ja = compute_ncj_field(q, data, ref, tau=tau, jitter_sigma=0.1,
                           jitter_seed=11).V
    jb = compute_ncj_field(q, data, ref, tau=tau, jitter_sigma=0.1,
                           jitter_seed=11).V
    jc = compute_ncj_field(q, data, ref, tau=tau, jitter_sigma=0.1,
                           jitter_seed=12).V
    eq, ep, en = _jitter_streams([(24, 2), (24, 2), (24, 2)], 0.1, 11)
    check("7a. same seed -> bitwise identical jittered field",
          np.array_equal(ja, jb))
    check("7b. different seed -> different field",
          not np.array_equal(ja, jc))
    check("7c. query/pos/neg jitter streams mutually distinct",
          not np.array_equal(eq, ep) and not np.array_equal(ep, en))

    # 8. Compute accounting: kernel pairs and wall time recorded per call
    #    and mirrored into a WorkCounter.
    wc = WorkCounter()
    r8 = compute_ncj_field(q, data, ref, tau=tau, jitter_sigma=0.1,
                           jitter_seed=3, counter=wc)
    expect = len(q) * (len(data) + len(ref))
    check("8. kernel pairs + wall time recorded",
          r8.kernel_pairs == expect and wc.kernel_pairs == expect
          and r8.wall_time > 0.0)

    # 9. No NaN/Inf is silently converted into a finite value: underflowed
    #    masses raise by default, are counted when zeroed, and non-finite
    #    inputs raise.
    far_pos = np.full((8, 2), 1e6)
    near = rng.normal(size=(8, 2)) * 0.01
    try:
        compute_field(near, far_pos, near.copy(), tau=tau, gain="constant")
        check("9a. underflowed mass raises by default", False)
    except DegenerateFieldError:
        check("9a. underflowed mass raises by default", True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        r9 = compute_field(near, far_pos, near.copy(), tau=tau,
                           gain="constant", on_degenerate="zero")
    check("9b. zeroed degenerate rows are counted",
          r9.n_degenerate_rows == 8 and np.all(np.isfinite(r9.V)))
    badq = q.copy()
    badq[0, 0] = np.nan
    try:
        compute_field(badq, data, ref, tau=tau, gain="paper")
        check("9c. non-finite input raises", False)
    except DegenerateFieldError:
        check("9c. non-finite input raises", True)

    if fails:
        raise SystemExit(f"invariant tests FAILED: {fails}")
    log("all invariant tests passed")


if __name__ == "__main__":
    invariant_tests()
