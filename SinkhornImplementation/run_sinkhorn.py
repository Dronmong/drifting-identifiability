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


# ----------------------------------------------------------------------------
# S7. Certificate-scheduled gain: unfreezing the mass-product collapse
# ----------------------------------------------------------------------------


def s7_certified_gain(rng) -> None:
    sec("S7. Certificate-scheduled gain vs the paper's mass-product gain")
    out("The exact identity `algorithm2Drift = (P*Q) . (C+ - C-)`")
    out("(Lean: `algorithm2Drift_eq_massProduct_centroidDiff`) splits the")
    out("drift into a raw-mass GAIN and a self-normalized-centroid SIGNAL;")
    out("only the signal carries identifiability")
    out("(`interactionFrameBound_of_probeScaling` covers any positive")
    out("per-query rescaling of it).  This section swaps the paper's")
    out("`P*Q` gain -- proved to collapse exponentially off-support by")
    out("`algorithm2Drift_norm_le_affinityMass` -- for alternatives.")
    out("Full derivation: `PROPOSAL_CERTIFIED_GAIN.md`.")
    out("")
    out("Same S3 setup (2-D unequal-mass target (0.3, 0.7), 400 particles,")
    out("400 positives/step, eta = 0.5, tau = 0.2, 300 steps, eye-masked")
    out("reuse).  `gmax` for `const`/`cert` is calibrated ONCE per `t` from a")
    out("HEALTHY batch (particles drawn from the target itself, seed")
    out("independent of the descent) -- using the current/frozen batch's")
    out("own median would just recreate the paper's collapse at far init.")
    out("")
    centers = np.array([[-2.0, 0.0], [2.0, 0.0]])
    weights = np.array([0.3, 0.7])
    T = 0.2
    init_seed = {"between": 101, "far": 202, "collapsed": 303}
    out("| init | t | gain | err @ 100 | err @ 300 | mean err (last 100) |")
    out("|------|---|------|-----------|-----------|----------------------|")
    for t in (1, 3):
        cal_rng = np.random.default_rng(rng.integers(2**32))
        gmax_ref = sk.calibrate_gmax(weights, centers, 0.5, 400, 400, T, t, cal_rng)
        gain_modes = [
            ("paper", "paper", {}),
            ("power(g=0.25)", "power", {"gamma": 0.25}),
            ("min(P,Q)", "min", {}),
            ("const", "const", {"gmax": gmax_ref}),
            ("cert", "cert", {"gmax": gmax_ref, "lam": 1.0}),
        ]
        for init in ("between", "far", "collapsed"):
            for label, mode, kwargs in gain_modes:
                r2 = np.random.default_rng(20260707 + init_seed[init])
                traj = sk.particle_descent(
                    weights, centers, 0.5, init, 400, 400, 300, 0.5, T, t, r2,
                    gain=mode, gain_kwargs=kwargs,
                )
                d = dict(traj)
                tail = [e for s, e in traj if s >= 200]
                out(
                    f"| {init} | {t} | {label} | {d[100]:.3f} | {d[300]:.3f} | "
                    f"{np.mean(tail):.3f} |"
                )
        out(f"\n_t = {t}: reference gmax (median P*Q, healthy batch) = "
            f"{gmax_ref:.3e}._\n")
    out("`paper` rows use the same seeds/config as S3 and should reproduce")
    out("its table exactly -- a built-in cross-check of this section.")
    out("")
    out("**Equilibrium residual-noise check.**  Particles ARE distributed")
    out("like the target (population drift ~ 0 by symmetry of supply and")
    out("demand); only finite-sample noise remains.  Checks that the")
    out("alternative gains do not amplify noise relative to the paper at a")
    out("point the paper already handles well.  All five modes see the")
    out("IDENTICAL (particles, positives) draw per `t`.")
    out("")
    out("| t | gain | median norm(V) |")
    out("|---|------|-----------------|")
    for t in (1, 3):
        cal_rng = np.random.default_rng(rng.integers(2**32))
        gmax_ref = sk.calibrate_gmax(weights, centers, 0.5, 400, 400, T, t, cal_rng)
        r2 = np.random.default_rng(rng.integers(2**32))
        sample = sk.bimodal_sampler(weights, centers, 0.5, r2)
        pts = sample(400)
        ypos = sample(400)
        gain_modes = [
            ("paper", "paper", {}),
            ("power(g=0.25)", "power", {"gamma": 0.25}),
            ("min(P,Q)", "min", {}),
            ("const", "const", {"gmax": gmax_ref}),
            ("cert", "cert", {"gmax": gmax_ref, "lam": 1.0}),
        ]
        for label, mode, kwargs in gain_modes:
            V = sk.compute_v_sinkhorn(
                pts, ypos, pts, T, t, True, gain=mode, gain_kwargs=kwargs
            )
            med = float(np.median(np.linalg.norm(V, axis=1)))
            out(f"| {t} | {label} | {med:.3e} |")
    out("")
    out("**Diagnosis: unfreezing the step size is confirmed directly; full")
    out("mode-mass recovery from a collapsed swarm is a separate, harder")
    out("problem that a gain swap alone does not solve.**  Traced directly")
    out("(far init, t = 1, `cert` gain driving the descent):")
    out("")
    out("| step | median gain (cert) | swarm mean position | frac. nearest mode 1 (target 0.7) |")
    out("|------|---------------------|----------------------|--------------------------------------|")
    cal_rng2 = np.random.default_rng(rng.integers(2**32))
    gmax1 = sk.calibrate_gmax(weights, centers, 0.5, 400, 400, T, 1, cal_rng2)
    r4 = np.random.default_rng(20260707 + init_seed["far"])
    sample4 = sk.bimodal_sampler(weights, centers, 0.5, r4)
    pts4 = np.array([5.0, 5.0]) + r4.normal(0, 0.25, (400, 2))
    checkpoints = {0, 30, 100, 200, 300}
    for step in range(301):
        if step in checkpoints:
            frac1 = float((sk.cdist(pts4, centers).argmin(axis=1) == 1).mean())
            mean_pos = pts4.mean(axis=0)
        ypos4 = sample4(400)
        dist_pos4 = sk.cdist(pts4, ypos4)
        dist_neg4 = sk.cdist(pts4, pts4) + np.eye(400) * 1e6
        logit4 = np.concatenate([-dist_pos4 / T, -dist_neg4 / T], axis=1)
        A4 = sk.balanced_affinity(logit4, 1)
        P4, Q4 = A4[:, :400].sum(axis=1), A4[:, 400:].sum(axis=1)
        g4 = sk.gain_schedule(
            "cert", P4, Q4, A4[:, :400], A4[:, 400:], ypos4, pts4, gmax=gmax1, lam=1.0
        )
        if step in checkpoints:
            out(
                f"| {step} | {float(np.median(g4)):.3e} | "
                f"({mean_pos[0]:.2f}, {mean_pos[1]:.2f}) | {frac1:.3f} |"
            )
        V4 = sk.compute_v_sinkhorn(
            pts4, ypos4, pts4, T, 1, True, gain="cert", gain_kwargs={"gmax": gmax1, "lam": 1.0}
        )
        pts4 = pts4 + 0.5 * V4
    out("")
    out("Reading: the paper's gain is exactly frozen at far init (S3: error")
    out("flat at 0.300 for the full 300 steps at every `t`); the diagnosis")
    out("trace shows `cert`'s gain is five-to-six orders of magnitude larger")
    out("there, and the swarm's mean position genuinely travels almost the")
    out("entire distance from `(5, 5)` to mode 1's center `(2, 0)` within 300")
    out("steps -- the mass-product collapse is REAL and the certificate gain")
    out("DOES fix it, exactly as designed.  Yet `frac. nearest mode 1` barely")
    out("moves off `1.000`: because every particle in the (still tightly")
    out("clustered) swarm computes nearly the same local drift, the whole")
    out("swarm moves as one coherent body toward its nearest/dominant mode")
    out("(here mode 1, since `(5,5)` is Euclidean-closer to `(2,0)` than to")
    out("`(-2,0)`) instead of splitting into the correct 30/70 proportions.")
    out("This is a SEPARATE limitation of the swarm dynamics -- homogeneity,")
    out("not gain starvation -- that no per-particle gain rescaling can fix")
    out("alone, since it multiplies every particle's signal by a comparable")
    out("factor.  It explains the modest (not dramatic) mode-mass-error gains")
    out("in the main table above, and predicts the natural next experiment:")
    out("inject per-particle diversity (repulsion, or per-step noise) to")
    out("break the homogeneity, which should compose with the gain schedule")
    out("rather than substitute for it.")
    out("")
    out("Across the main table, the plainest alternative -- `const`, a fixed")
    out("reference gain with NO adaptivity, i.e. constant-speed centroid-")
    out("difference flow -- is the most consistently strong performer (best")
    out("or tied-best in most far/collapsed rows), while `cert` is the least")
    out("reliable of the four alternatives: it matches `const`'s gains at")
    out("far/t=3 and collapsed/t=3, gives NO improvement at collapsed/t=1,")
    out("and is worse than the paper at between/t=3.  Honest reading:")
    out("sophistication (a plug-in finite-sample certificate) did not beat")
    out("simplicity (a fixed positive floor) at this toy scale -- the win")
    out("here is dropping the mass-product attenuation at all, not the")
    out("particular schedule chosen to replace it.  `min(P,Q)` is the")
    out("weakest alternative, consistent with it still shrinking with the")
    out("smaller branch's own mass rather than carrying a fixed floor.")


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
    s7_certified_gain(rng)
    R.append("")
    R.append(f"_Runtime: {time.time() - t0:.1f}s._")
    OUT.write_text("\n".join(R), encoding="utf-8")
    print(f"\nwrote {OUT} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
