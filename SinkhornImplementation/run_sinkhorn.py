"""Sinkhorn-balanced drifting: EXTENSION experiments (not part of the paper).

Run:  uv run --with numpy python SinkhornImplementation/run_sinkhorn.py
Writes SinkhornImplementation/RESULTS.md (deterministic, seed 20260707).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "numerics"))
import driftlab as dl  # noqa: E402  (paper-side reference implementation)
import sinkhorn_drift as sk  # noqa: E402

OUT = HERE / "RESULTS.md"
R: list[str] = []
TAUS = [0.02, 0.05, 0.2]
ITERS_GRID = [0, 1, 2, 3, 5, 10]


def sec(title: str) -> None:
    print(f"\n=== {title}")
    R.append(f"\n## {title}\n")


def out(line: str = "") -> None:
    print(line)
    R.append(line)


# ----------------------------------------------------------------------------
# S0. Agreement audit: iters = 1 reproduces the paper / Lean pipeline
# ----------------------------------------------------------------------------


def s0_agreement(rng) -> None:
    sec("S0. Agreement: one balancing step == paper Algorithm 2")
    worst = 0.0
    for _ in range(20):
        n, npos = int(rng.integers(2, 12)), int(rng.integers(1, 12))
        x = rng.normal(0, 1, n)
        ypos = rng.normal(0, 1, npos)
        yneg = rng.normal(0, 1, n)
        for T in (0.05, 0.2, 1.0):
            for masked in (False, True):
                v_paper = dl.compute_v_paper(x, ypos, yneg, T, masked)
                v_sink = sk.compute_v_sinkhorn(x, ypos, yneg, T, 1, masked)
                worst = max(worst, float(np.abs(v_paper - v_sink).max()))
    out(f"- max |paper - sinkhorn(iters=1)| over 120 random configs: `{worst:.2e}`")
    out("  (`iters = 1` IS the paper's estimator; larger `iters` is the extension.)")


# ----------------------------------------------------------------------------
# S1. Estimator dispersion vs balancing depth
# ----------------------------------------------------------------------------


def _orbit_sharp_constant(anchors: np.ndarray, tau: float, t: int) -> float:
    """Sharp two-atom frame constant of the t-step balanced anchor-atom
    matrix (Lean: frameConstant of `sinkhornOrbit01Setup` via the exact
    rescaling identity)."""
    atoms = np.array([0.0, 1.0])
    M = np.exp(-np.abs(anchors[:, None] - atoms[None, :]) / tau)
    for _ in range(t):
        M = M / np.sqrt(M.sum(axis=1))[:, None] / np.sqrt(M.sum(axis=0))[None, :]
    return float((M[:, 0] * M[:, 1]).max())


def s1_dispersion(rng) -> None:
    sec("S1. Estimator dispersion vs balancing depth (two-atom, N = 64)")
    out("Fixed anchors [0, 0.3, 1], a = [0.3, 0.7], b = [0.6, 0.4], 300 reps.")
    out("Balancing rescales the field itself (S4: the sharp frame constant")
    out("grows with depth), so raw dispersion is reported alongside the")
    out("signal-normalized dispersion std/c_t — the quantity the coefficient")
    out("bound (2B/c)(...) actually depends on.")
    out("")
    a = np.array([0.3, 0.7])
    b = np.array([0.6, 0.4])
    anchors = np.array([0.0, 0.3, 1.0])
    N, reps = 64, 300
    for tau in TAUS:
        out(f"**tau = {tau}**")
        out("")
        out("| t | raw std | c_t (sharp) | std / c_t |")
        out("|---|---------|-------------|-----------|")
        for t in ITERS_GRID:
            vs = np.empty((reps, len(anchors)))
            r2 = np.random.default_rng(rng.integers(2**32))
            for k in range(reps):
                ypos = dl.sample_two_atom(a, N, r2)
                yneg = dl.sample_two_atom(b, N, r2)
                vs[k] = sk.compute_v_sinkhorn(anchors, ypos, yneg, tau, t, False)
            raw = float(vs.std(axis=0).mean())
            ct = _orbit_sharp_constant(anchors, tau, t) if t >= 1 else float("nan")
            cell = f"{raw / ct:.2f}" if t >= 1 else "-"
            ctxt = f"{ct:.2e}" if t >= 1 else "-"
            out(f"| {t} | {raw:.2e} | {ctxt} | {cell} |")
        out("")
    out("Reading: raw dispersion grows because balancing *amplifies* the field")
    out("(the same scalings amplify the frame constant, S4).  The")
    out("signal-normalized dispersion std/c_t — what the certified coefficient")
    out("bound pays — improves or stays flat at moderate depth (t = 2-3) and")
    out("*degrades* at t = 10 for small tau: over-balancing amplifies noise")
    out("faster than signal once masses are nearly uniform.  Moderate depth is")
    out("the sweet spot, matching S3's choice of t = 3; the decisive gains are")
    out("deterministic denominators (S2) and mass recovery (S3).")


# ----------------------------------------------------------------------------
# S2. Mass uniformization: the certified-denominator dial
# ----------------------------------------------------------------------------


def s2_mass(rng) -> None:
    sec("S2. Row/column mass uniformization vs depth (the dmin dial)")
    out("2-D bimodal geometry (centers (+-2,0), std 0.5), 64 particles between")
    out("the modes, 64 positives, eye-masked reuse; median over 50 draws.")
    out("")
    centers = np.array([[-2.0, 0.0], [2.0, 0.0]])
    weights = np.array([0.5, 0.5])
    out("| tau | t | row CV | col CV | row min/mean | ESS med |")
    out("|-----|---|--------|--------|--------------|---------|")
    for tau in (0.05, 0.2):
        for t in (0, 1, 3, 10):
            vals = {"row_cv": [], "col_cv": [], "row_min_over_mean": [], "ess_median": []}
            r2 = np.random.default_rng(rng.integers(2**32))
            for _ in range(50):
                sample = sk.bimodal_sampler(weights, centers, 0.5, r2)
                pts = r2.normal(0, 0.25, (64, 2))
                ypos = sample(64)
                d = sk.affinity_diagnostics(pts, ypos, pts, tau, t, True)
                for key in vals:
                    vals[key].append(d[key])
            out(
                f"| {tau} | {t} | {np.median(vals['row_cv']):.3f} | "
                f"{np.median(vals['col_cv']):.3f} | "
                f"{np.median(vals['row_min_over_mean']):.3f} | "
                f"{np.median(vals['ess_median']):.1f} |"
            )
    out("")
    out("Row/column mass CV -> 0 with depth: the SNIS denominators become")
    out("deterministic, which is exactly the regime where the certified")
    out("constants stop paying the denominator-tail price.")


# ----------------------------------------------------------------------------
# S3. Particle descent (paper Figure-3 methodology), mass recovery
# ----------------------------------------------------------------------------


def s3_particles(rng) -> None:
    sec("S3. Particle descent: mode-mass recovery, paper (t=1) vs balanced (t=3)")
    out("2-D target with UNEQUAL weights (0.3, 0.7), centers (+-2, 0), std 0.5;")
    out("400 particles moved directly by the estimator field (no network),")
    out("400 positives per step, eta = 0.5, tau = 0.2, 300 steps, eye-masked")
    out("reuse.  Metric: |empirical mass of mode 1 - 0.7| (mode-mass error).")
    out("")
    centers = np.array([[-2.0, 0.0], [2.0, 0.0]])
    weights = np.array([0.3, 0.7])
    out("| init | t | err @ step 100 | err @ step 300 | mean err (last 100) |")
    out("|------|---|----------------|----------------|----------------------|")
    init_seed = {"between": 101, "far": 202, "collapsed": 303}
    for init in ("between", "far", "collapsed"):
        for t in (1, 3):
            r2 = np.random.default_rng(20260707 + init_seed[init])
            traj = sk.particle_descent(
                weights, centers, 0.5, init, 400, 400, 300, 0.5, 0.2, t, r2
            )
            d = dict(traj)
            tail = [e for s, e in traj if s >= 200]
            out(
                f"| {init} | {t} | {d[100]:.3f} | {d[300]:.3f} | "
                f"{np.mean(tail):.3f} |"
            )
    out("")
    out("This is the extension's toy-scale test of the paper's Figure-3 claim")
    out("(convergence without mode collapse), with the harder unequal-mass")
    out("target that exercises exactly what the identifiability theorems")
    out("certify: recovery of mode masses.")


# ----------------------------------------------------------------------------
# S4. Certified constants along the realized orbit
# ----------------------------------------------------------------------------


def s4_certified(rng) -> None:
    sec("S4. Certified constants along the realized orbit (Lean crosswalk)")
    out("Anchor-by-atom kernel matrix (anchors [0, 0.3, 1], atoms {0,1}),")
    out("balanced t steps with the cumulative diagonal scalings (u, v) tracked.")
    out("Checks the exact Lean identity `inducedInteractionVector_sinkhornOrbit01_eq`")
    out("(U_orbit(n) = u(x_n)^2 v(0) v(1) U_bare(n)) and the transfer lower bound")
    out("of `interactionFrameBound_of_biScaling` against the direct sharp constant.")
    out("")
    anchors = np.array([0.0, 0.3, 1.0])
    atoms = np.array([0.0, 1.0])
    tau = 0.2
    K = np.exp(-np.abs(anchors[:, None] - atoms[None, :]) / tau)
    c_bare = float((K[:, 0] * K[:, 1]).max())
    out(f"bare sharp constant c_bare = max_n k(x_n,0) k(x_n,1) = `{c_bare:.4e}`")
    out("")
    out("| t | identity max err | c_orbit (direct) | transfer lower bound | mass CV |")
    out("|---|-------------------|-------------------|-----------------------|---------|")
    for t in (1, 2, 3, 5, 10):
        M = K.copy()
        cu = np.ones(len(anchors))
        cv = np.ones(len(atoms))
        for _ in range(t):
            r = M.sum(axis=1)
            c = M.sum(axis=0)
            cu = cu / np.sqrt(r)
            cv = cv / np.sqrt(c)
            M = M / np.sqrt(r)[:, None] / np.sqrt(c)[None, :]
        # exact rescaling identity (Lean: inducedInteractionVector_sinkhornOrbit01_eq)
        u_orbit = M[:, 0] * M[:, 1]
        u_pred = cu**2 * (cv[0] * cv[1]) * (K[:, 0] * K[:, 1])
        ident_err = float(np.abs(u_orbit - u_pred).max())
        c_direct = float(u_orbit.max())
        transfer = float((cu**2).min() * (cv[0] * cv[1]) * c_bare)
        masses = np.concatenate([M.sum(axis=1), M.sum(axis=0)])
        cv_mass = float(masses.std() / masses.mean())
        out(
            f"| {t} | {ident_err:.1e} | {c_direct:.4e} | {transfer:.4e} | "
            f"{cv_mass:.3f} |"
        )
    out("")
    out("The transfer bound is valid (<= direct) at every depth, and the direct")
    out("constant stabilizes as the matrix balances — certified identifiability")
    out("rides along the whole orbit, as proved in `SinkhornBalanced.lean`.")


# ----------------------------------------------------------------------------
# S6. Numerical validation of the BalancedSampling.lean constants
# ----------------------------------------------------------------------------


def _two_step_weights(anchors: np.ndarray, tau: float, i: int, row_mass, y):
    """Weight from `BalancedSampling.lean::twoStepWeight` for scalar y."""
    y = np.asarray(y, dtype=float)
    K_y = np.exp(-np.abs(anchors[:, None] - y[None, :]) / tau)
    g = K_y.sum(axis=0)
    h = (K_y / np.sqrt(row_mass)[:, None]).sum(axis=0)
    return K_y[i] / (np.sqrt(np.sqrt(g)) * np.sqrt(h))


def s6_balanced_sampling_constants(rng) -> None:
    sec("S6. BalancedSampling constants: row-mass event -> weight/centroid bounds")
    out("Monte-Carlo sanity check for the Lean theorem")
    out("`balancedTwoStepCentroid_deviation_prob_le` at `t = 2`.")
    out("We condition on the good event")
    out("`|r_j - Mbar_j| <= delta Mbar_j` for all anchors, then check")
    out("`|W - Wbar| <= 4 delta Wbar` and centroid gap `<= 16 delta R`.")
    out("")
    anchors = np.array([0.0, 0.3, 1.0])
    atom_weights = np.array([0.3, 0.7])
    atoms = np.array([0.0, 1.0])
    tau = 0.2
    i_anchor = 1
    n = 512
    reps = 500
    delta = 1.0 / 8.0
    c = 0.5
    R_ball = 0.5
    K_atoms = np.exp(-np.abs(anchors[:, None] - atoms[None, :]) / tau)
    mbar = n * (K_atoms @ atom_weights)
    good = 0
    max_rel_over = 0.0
    max_centroid_over = 0.0
    for _ in range(reps):
        y = dl.sample_two_atom(atom_weights, n, rng)
        K_sample = np.exp(-np.abs(anchors[:, None] - y[None, :]) / tau)
        r = K_sample.sum(axis=1)
        rel = np.max(np.abs(r - mbar) / mbar)
        if rel > delta:
            continue
        good += 1
        w = _two_step_weights(anchors, tau, i_anchor, r, y)
        wbar = _two_step_weights(anchors, tau, i_anchor, mbar, y)
        rel_w = np.max(np.abs(w - wbar) / np.maximum(wbar, 1e-300))
        cref = float(np.dot(wbar, y) / wbar.sum())
        creal = float(np.dot(w, y) / w.sum())
        max_rel_over = max(max_rel_over, rel_w / (4.0 * delta))
        max_centroid_over = max(
            max_centroid_over, abs(creal - cref) / (16.0 * delta * R_ball)
        )
    out(f"- reps: `{reps}`, good-event reps: `{good}` (`delta = 1/8`, N = {n})")
    out(f"- max observed weight ratio / theorem bound: `{max_rel_over:.3f}`")
    out(f"- max observed centroid-gap ratio / theorem bound: `{max_centroid_over:.3f}`")
    out("")
    out("Ratios below 1 validate the explicit constants used by the Lean")
    out("finite-sample theorem on this two-atom testbed.  The constants are")
    out("deliberately loose; this is a guardrail check, not a tuning claim.")


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(20260707)
    R.append("# Sinkhorn-balanced drifting: extension experiments")
    R.append("")
    R.append("**EXTENSION — not part of the paper.**  Generated by")
    R.append("`SinkhornImplementation/run_sinkhorn.py` (seed 20260707).  Lean")
    R.append("counterpart: `DriftingIdentifiability/SinkhornBalanced.lean`; design")
    R.append("rationale and predictions: `SinkhornImplementation/PLAN.md`.")
    s0_agreement(rng)
    s1_dispersion(rng)
    s2_mass(rng)
    s3_particles(rng)
    s4_certified(rng)
    s6_balanced_sampling_constants(rng)
    R.append("")
    R.append(f"_Runtime: {time.time() - t0:.1f}s._")
    OUT.write_text("\n".join(R), encoding="utf-8")
    print(f"\nwrote {OUT} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
