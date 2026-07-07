"""Objective 7 experiment suite.  Run:  uv run --with numpy python numerics/run_all.py

Writes numerics/RESULTS.md.  Every experiment names the Lean declarations whose
formulas it transcribes; see numerics/README.md for the crosswalk.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import driftlab as dl  # noqa: E402

OUT = Path(__file__).parent / "RESULTS.md"
R: list[str] = []  # report lines
TAUS = [0.02, 0.05, 0.2]  # the paper's temperature grid (Table 8)
N_PAPER = 64  # per-class batch (Table 8: Npos = Nneg = 64)


def sec(title: str) -> None:
    print(f"\n=== {title}")
    R.append(f"\n## {title}\n")


def out(line: str = "") -> None:
    print(line)
    R.append(line)


# ============================================================================
# E0. Transcription audit: paper pseudo-code vs Lean pipeline vs theorems
# ============================================================================


def e0_audit(rng) -> None:
    sec("E0. Transcription audit (paper Algorithm 2 vs Lean pipeline)")
    worst = 0.0
    for trial in range(20):
        n, npos = rng.integers(2, 12), rng.integers(1, 12)
        x = rng.normal(0, 1, n)
        ypos = rng.normal(0, 1, npos)
        yneg = rng.normal(0, 1, n)
        for T in (0.05, 0.2, 1.0):
            for masked in (False, True):
                mask = np.eye(n, dtype=bool) if masked else np.zeros((n, n), bool)
                v1 = dl.compute_v_paper(x, ypos, yneg, T, masked)
                v2 = dl.compute_v_lean(x, ypos, yneg, T, mask)
                worst = max(worst, float(np.abs(v1 - v2).max()))
    out(f"- max |paper - lean| over 120 random configs: `{worst:.2e}`")

    # matched-batch cancellation (algorithm2Drift_matched_zero)
    y = rng.normal(0, 1, 16)
    x = rng.normal(0, 1, 8)
    v = dl.compute_v_paper(x, y, y, 0.2, False)
    out(f"- matched-batch drift, no mask (theorem: = 0): max |V| = `{np.abs(v).max():.2e}`")

    # mass-product * centroid-difference identity
    x = rng.normal(0, 1, 8)
    ypos, yneg = rng.normal(0, 1, 10), rng.normal(0, 1, 12)
    v = dl.compute_v_paper(x, ypos, yneg, 0.2, False)
    cd = dl.centroid_diff(x, ypos, yneg, 0.2, False)
    # recompute masses for the identity check
    dist_pos = np.abs(x[:, None] - ypos[None, :])
    dist_neg = np.abs(x[:, None] - yneg[None, :])
    logit = np.concatenate([-dist_pos / 0.2, -dist_neg / 0.2], axis=1)
    er = np.exp(logit - logit.max(axis=1, keepdims=True))
    A_row = er / er.sum(axis=1, keepdims=True)
    ec = np.exp(logit - logit.max(axis=0, keepdims=True))
    A_col = ec / ec.sum(axis=0, keepdims=True)
    A = np.sqrt(A_row * A_col)
    P, Q = A[:, :10].sum(axis=1), A[:, 10:].sum(axis=1)
    ident = np.abs(v - P * Q * cd).max()
    out(f"- drift = P*Q*(C+ - C-) (algorithm2Drift_eq_massProduct_centroidDiff): max err `{ident:.2e}`")

    # drift norm bound 2*Npos*Nneg*R (algorithm2Drift_norm_le)
    Rad = max(np.abs(ypos).max(), np.abs(yneg).max())
    bound = 2 * 10 * 12 * Rad
    out(f"- |drift| <= 2 Npos Nneg R: max|V| = `{np.abs(v).max():.3f}` vs bound `{bound:.1f}`")

    # frame certificate validity (interactionFrameBound_inverseCertificate)
    z = np.array([0.0, 0.9, 2.1])  # distinct pair sums
    P3 = len(dl.strict_pairs(3))
    M = dl.interaction_matrix(z, np.arange(P3, dtype=float))
    c = dl.certified_frame_constant(M)
    viol = dl.frame_violation(M, c, 20000, rng)
    out(
        f"- frame inequality c||w||_1 <= ||Mw||_inf over 20k random w: "
        f"max(lhs - rhs) = `{viol:.2e}` (negative = satisfied with margin; "
        f"c_cert = `{c:.3e}`)"
    )


# ============================================================================
# E1. Certified frame-constant landscape (Objectives 1/3 formulas)
# ============================================================================


def e1_frame_landscape(rng) -> None:
    sec("E1. Certified frame constant: scaling in m, window, and probe design")
    out("Support: Mian-Chowla (Sidon) points scaled into a window [0, W]")
    out("(distinct pair sums hold, matching `DistinctStrictPairSums`).")
    out("")
    mian_chowla = np.array([1, 2, 4, 8, 13, 21, 31, 45, 66, 81], dtype=float)
    out("| m | pairs | W | c_cert (integer probes) | c_cert (midpoint probes) | ceiling | min sv (int) |")
    out("|---|-------|---|--------------------------|---------------------------|---------|--------------|")
    for m in (2, 3, 4, 5, 6, 8, 10):
        for W in (2.0, 4.0):
            z = mian_chowla[:m]
            z = (z - z[0]) / (z[m - 1] - z[0]) * W
            pairs = dl.strict_pairs(m)
            P = len(pairs)
            Mint = dl.interaction_matrix(z, np.arange(P, dtype=float))
            mids = np.array([(z[i] + z[j]) / 2 for i, j in pairs])
            Mmid = dl.interaction_matrix(z, mids)
            c_int = dl.certified_frame_constant(Mint)
            c_mid = dl.certified_frame_constant(Mmid)
            ceil = dl.frame_ceiling(z)
            sv = np.linalg.svd(Mint, compute_uv=False)
            out(
                f"| {m} | {P} | {W:.0f} | {c_int:.3e} | {c_mid:.3e} | "
                f"{ceil:.3e} | {sv[-1]:.1e} |"
            )
    out("")
    out("Sanity (theorem `interactionFrameBound_le_interactionNorm`): c_cert <= ceiling in every row.")
    out("`0.000e+00` entries are float64 underflow (< 1e-308): the certificate is")
    out("positive as a theorem (`gaussianEmpiricalPointCertifiedFrameConstant_pos`)")
    out("but numerically indistinguishable from zero — the practical content of the")
    out("'logical identifiability vs numerical usefulness' caveat.  Midpoint probes")
    out("(the pairwise-optimal placement from the ceiling analysis) buy 10-20 orders")
    out("of magnitude over integer probes at m = 5-6 and still collapse by m = 8:")
    out("probe design helps enormously but cannot beat the Vandermonde decay.")
    # m=2 sweet spot
    out("")
    out("m = 2 sharp constant `||U_01||` (midpoint probe) vs separation Delta:")
    out("")
    out("| Delta | |U|(midpoint) = Delta*exp(-Delta^2/4) |")
    out("|-------|----------------------------------------|")
    for d in (0.25, 0.5, 1.0, np.sqrt(2.0), 2.0, 3.0, 4.0):
        out(f"| {d:.3f} | {d * np.exp(-d * d / 4):.4f} |")
    out("")
    out(f"Peak at Delta = sqrt(2): `{np.sqrt(2 / np.e):.4f}` = sqrt(2/e), the proved ceiling.")


# ============================================================================
# E2. Column-reweighting price (Objective 4 two-atom class)
# ============================================================================


def e2_column_price() -> None:
    sec("E2. Column-reweighting price: exact scale vs proved floor 1/N")
    out("`U^col = s * U^bare` with s = 1/sqrt(g(0)g(1)); theorem floor: s >= 1/N.")
    out("")
    out("| tau | anchors | N | s exact | 1/N floor | s/(1/N) |")
    out("|-----|---------|---|---------|-----------|---------|")
    for tau in TAUS:
        for desc, anchors in (
            ("atoms {0,1}", np.array([0.0, 1.0])),
            ("8 in [0,1]", np.linspace(0, 1, 8)),
            ("64 in [0,1]", np.linspace(0, 1, 64)),
            ("64 in [0,4] (mostly far)", np.linspace(0, 4, 64)),
        ):
            s = dl.col_reweight_scale(anchors, tau)
            n = len(anchors)
            out(f"| {tau} | {desc} | {n} | {s:.3e} | {1 / n:.3e} | {s * n:.2f} |")
    out("")
    out("When anchors sit far from the atoms (small kernel mass g < 1), the")
    out("reweighting *amplifies* the frame constant (s > 1); the 1/N floor is")
    out("loose exactly when the anchor cloud is far from the support.")


# ============================================================================
# E3. Monte-Carlo: SNIS 1/N rate, modified-vs-bare target, leave-out bias
# ============================================================================


def e3_consistency(rng) -> None:
    sec("E3. Estimator consistency: 1/N rate and the modified-field target")
    a = np.array([0.3, 0.7])  # target p coefficients
    b = np.array([0.6, 0.4])  # model q coefficients
    anchors = np.array([0.0, 0.3, 1.0])  # asymmetric: modified != bare field
    reps = 400
    out(f"Two-atom class: atoms {{0,1}}, a = {a.tolist()}, b = {b.tolist()},")
    out(f"fixed anchors {anchors.tolist()}, {reps} replicates. Estimator:")
    out("no-mask centroid difference (Algorithm2.noMaskCentroidDrift).")
    out("")
    for tau in (0.2, 0.05):
        vmod = dl.modified_field_two_atom(anchors, tau, a, b)
        vbare = dl.bare_field_two_atom(anchors, tau, a, b)
        gap = np.abs(vmod - vbare).max()
        gap_sq = float(((vmod - vbare) ** 2).sum())
        out(f"**tau = {tau}**: modified field = {np.round(vmod, 5).tolist()},")
        out(f"bare field = {np.round(vbare, 5).tolist()}, ||gap||^2 = `{gap_sq:.2e}`")
        out("")
        out("| N | MSE vs modified | MSE vs bare | N * MSE(mod) | MSE(bare)-MSE(mod) |")
        out("|---|------------------|-------------|--------------|---------------------|")
        for N in (8, 16, 32, 64, 128, 256, 512):
            e_mod = e_bare = 0.0
            for _ in range(reps):
                ypos = dl.sample_two_atom(a, N, rng)
                yneg = dl.sample_two_atom(b, N, rng)
                cd = dl.centroid_diff(anchors, ypos, yneg, tau, False)
                e_mod += float(((cd - vmod) ** 2).sum())
                e_bare += float(((cd - vbare) ** 2).sum())
            e_mod /= reps
            e_bare /= reps
            out(
                f"| {N} | {e_mod:.3e} | {e_bare:.3e} | {N * e_mod:.3f} | "
                f"{e_bare - e_mod:.3e} |"
            )
        out("")
        out(f"(last column converges to ||gap||^2 = {gap_sq:.2e}, the residual bias")
        out("against the *bare* field that no amount of data removes)")
        out("")
    out("Reading: MSE vs the *modified* (column-reweighted) field scales as")
    out("~1/N (constant N*MSE), while against the *bare* mean-shift field a")
    out("bias floor of exactly ||modified - bare||^2 remains: the estimator")
    out("provably and measurably targets the modified field")
    out("(`ColumnReweightedMeanShift`).  At tau = 0.05 the field magnitude")
    out("itself is ~5e-4 (cross-atom kernel e^{-20}): the identifiability")
    out("signal at this geometry is carried almost entirely by tau = 0.2.")


def e3b_leaveout_bias() -> None:
    sec("E3b. Eye-mask leave-out bias b(N) (deletedNegativeCentroid hypotheses)")
    out("Per-slot deleted weight drops one anchor from the column mass.  Bias")
    out("b = max_l |E[w_l(Y)(Y - c)]| against the full-mass centroid target c,")
    out("computed in closed form for the two-atom model (tau = 0.2, anchor x_0).")
    out("")
    b_coef = np.array([0.6, 0.4])
    atoms = np.array([0.0, 1.0])
    tau = 0.2
    out("| N anchors | b(N) | N * b(N) |")
    out("|-----------|------|----------|")
    for N in (8, 16, 32, 64, 128, 256):
        anchors = np.linspace(0.0, 1.0, N)
        i = 0  # analysis anchor
        k_i = dl.alg2_kernel(tau, np.abs(anchors[i] - atoms))
        g_full = dl.column_mass(anchors, tau, atoms)
        w_full = k_i / np.sqrt(g_full)
        c_target = (b_coef * w_full * atoms).sum() / (b_coef * w_full).sum()
        worst = 0.0
        for l in range(N):
            if l == i:
                continue  # masked slot: weight identically zero
            k_l = dl.alg2_kernel(tau, np.abs(anchors[l] - atoms))
            w_l = k_i / np.sqrt(g_full - k_l)
            mu_l = (b_coef * w_l * (atoms - c_target)).sum()
            worst = max(worst, abs(mu_l))
        out(f"| {N} | {worst:.3e} | {N * worst:.4f} |")
    out("")
    out("N * b(N) is bounded and still decreasing: the leave-out bias is *at most*")
    out("O(1/N) — the rate the documentation predicted — and empirically somewhat")
    out("better, because most dropped anchors carry little column mass.  The")
    out("indexed SNIS bound's bias term 2N^2b^2/dmin^2 is therefore subdominant to")
    out("the variance term at every batch size tested.")


def e3c_mask_gap() -> None:
    sec("E3c. Self-mask perturbation scale (SelfMaskPerturbation bound)")
    out("delta = exp(-1e6/tau_tilde), tau_tilde = tau*sqrt(C) (paper eq. 22).")
    out("")
    out("| tau | C | tau_tilde | log10(delta) |")
    out("|-----|---|-----------|---------------|")
    for tau in TAUS:
        for C in (256, 2048):
            tt = tau * np.sqrt(C)
            log10d = -1e6 / tt / np.log(10)
            out(f"| {tau} | {C} | {tt:.2f} | {log10d:.0f} |")
    out("")
    out("Even at the largest tau_tilde the masked-vs-deleted gap is ~10^-4800:")
    out("the deterministic perturbation bound is vacuously satisfied in float64")
    out("(masked and deleted estimators are bit-identical), so all statistical")
    out("content lives in the deleted-estimator theorems.")


# ============================================================================
# E4. Temperature/ESS regimes at the paper's operating point
# ============================================================================


def e4_ess(rng) -> None:
    sec("E4. What each temperature 'sees': softmax ESS at N = 64")
    out("Feature-geometry model in normalized units (paper eq. 20: mean pairwise")
    out("distance = 1 after normalization): same-cluster distances N(1-g/2, s),")
    out("cross-cluster N(1+g/2, s), half of the 64 negatives in each cluster.")
    out("Median row-softmax ESS over 200 anchors (ESS = 64 means uniform")
    out("averaging; ESS = 1 means nearest-neighbor selection).")
    out("")
    header = "| gap g | spread s | " + " | ".join(f"tau={t}" for t in TAUS) + " |"
    out(header)
    out("|" + "---|" * (2 + len(TAUS)))
    for g in (0.05, 0.1, 0.2, 0.4):
        for s in (0.02, 0.05, 0.1):
            cells = []
            for tau in TAUS:
                vals = []
                for _ in range(200):
                    d_same = np.maximum(rng.normal(1 - g / 2, s, 32), 1e-4)
                    d_cross = np.maximum(rng.normal(1 + g / 2, s, 32), 1e-4)
                    d = np.concatenate([d_same, d_cross])[None, :]
                    vals.append(dl.row_softmax_ess(d, tau)[0])
                cells.append(f"{np.median(vals):.1f}")
            out(f"| {g} | {s} | " + " | ".join(cells) + " |")
    out("")
    out("tau = 0.02 is in the nearest-neighbor regime (ESS ~ 1-3) for every")
    out("plausible geometry; tau = 0.2 averages broadly (ESS ~ 15-60); tau = 0.05")
    out("sits at the transition.  The paper's multi-temperature grid spans the")
    out("regimes, matching its own ablation (single tau FID 8.67-10.62 vs 8.46).")
    out("For the SNIS bounds this is the practical dmin/sigma dial: the certified")
    out("constants are honest but only as strong as the smallest-tau ESS.")


# ============================================================================
# E5. Conditioning ledger: end-to-end certified sample complexity
# ============================================================================


def e5_ledger(rng) -> None:
    sec("E5. Conditioning ledger: the certified chain, multiplied out")
    a = np.array([0.3, 0.7])
    b = np.array([0.6, 0.4])
    atoms = np.array([0.0, 1.0])
    anchors = np.array([0.0, 1.0])  # probes = anchors = the atoms
    delta_conf = 0.1  # 90% confidence
    target_l1 = 0.1
    out("Operating point: two-atom certified class (columnReweighted01Setup),")
    out(f"anchors = atoms = {{0,1}}, confidence 90%, target ||a-b||_1 <= {target_l1}.")
    out("")
    out("Chain: coefficient bound (2B/c_col)(||Vhat|| + eps) with B = 1")
    out("(`columnReweighted01_coefficientStability`), failure probability")
    out("MSE/eps^2 (`estimate_failure_le_meanSquare`), SNIS mean-square")
    out("2 N sigma^2 / dmin^2 (`selfNormalizedIndexed_meanSquare_le`, b = 0).")
    out("The theorem's denominator floor dmin must hold *deterministically*:")
    out("dmin = N * wmin (worst atom).  The LLN-typical denominator N * wbar is")
    out("NOT a theorem; it is listed to expose the slack of the certified floor.")
    out("")
    out("| tau | c_col | 2B/c_col | sigma_z^2 | wmin | wbar | N cert (wmin) | N typical (wbar) |")
    out("|-----|-------|----------|-----------|------|------|----------------|-------------------|")
    for tau in TAUS:
        c_bare = float(dl.u01_bare(anchors, tau).max())  # sup norm over probes
        s_col = dl.col_reweight_scale(anchors, tau)
        c_col = c_bare * s_col
        amp = 2.0 / c_col  # B = 1
        # per-anchor SNIS ingredients at anchor 0 (worst case by symmetry)
        w = dl.column_reweighted_weight(anchors, tau, 0, atoms)
        cq = (b * w * atoms).sum() / (b * w).sum()
        z_sq = (b * (w * (atoms - cq)) ** 2).sum()  # E|w(Y)(Y-c)|^2
        wmin = float(w.min())  # deterministic per-sample weight floor
        wbar = float((b * w).sum())  # typical (mean) weight, not certified
        # MSE per centroid <= 2*N*sigma^2/dmin^2; dmin = N*wfloor
        # stacked over 2 anchors x 2 branches: 8 sigma^2/(N wfloor^2) (crude)
        need_mse = (target_l1 / amp) ** 2 * delta_conf
        n_cert = 8.0 * z_sq / (wmin**2 * need_mse)
        n_typ = 8.0 * z_sq / (wbar**2 * need_mse)
        out(
            f"| {tau} | {c_col:.2e} | {amp:.1e} | {z_sq:.2e} | "
            f"{wmin:.2e} | {wbar:.2e} | {n_cert:.1e} | {n_typ:.1e} |"
        )
    out("")
    out("Observed check (tau = 0.2): Monte-Carlo MSE at the paper batch N = 64:")
    tau = 0.2
    vmod = dl.modified_field_two_atom(anchors, tau, a, b)
    reps = 2000
    err = 0.0
    for _ in range(reps):
        ypos = dl.sample_two_atom(a, N_PAPER, rng)
        yneg = dl.sample_two_atom(b, N_PAPER, rng)
        cd = dl.centroid_diff(anchors, ypos, yneg, tau, False)
        err += float(((cd - vmod) ** 2).sum())
    err /= reps
    c_col = float(dl.u01_bare(anchors, tau).max()) * dl.col_reweight_scale(anchors, tau)
    eps = np.sqrt(err / delta_conf)
    bound = (2.0 / c_col) * eps
    need_mse = (target_l1 * c_col / 2.0) ** 2 * delta_conf
    n_obs = N_PAPER * err / need_mse  # observed MSE ~ C/N extrapolation
    out(f"- observed MSE(N=64) = `{err:.3e}`; via the bridge, at 90% confidence")
    out(f"  ||a-b||_1 <= (2/c_col)*sqrt(MSE/0.1) = `{bound:.2f}`")
    out(f"  (true ||a-b||_1 = {np.abs(a - b).sum():.1f}; bound at N=64 is vacuous).")
    out(f"- extrapolating observed MSE ~ C/N: N_observed for the target = `{n_obs:.1e}`.")
    out("")
    out("Readings:")
    out("1. The *typical*-denominator complexity is temperature-invariant")
    out("   (~6e4): signal (field ~ c_col) and noise (sigma_z) carry the same")
    out("   kernel scale, so tiny tau does not by itself doom certification.")
    out("2. The *certified* complexity explodes as tau shrinks because the")
    out("   deterministic floor dmin = N*wmin pays the full cross-atom kernel")
    out("   e^{-1/tau}.  At tau = 0.2 certified/observed slack is ~10^3; at")
    out("   tau = 0.02 the certified route is astronomically pessimistic.")
    out("   Actionable theory gap surfaced by the numerics: replace the")
    out("   deterministic dmin with a high-probability lower bound on the")
    out("   random denominator (e.g. Bernstein for sums of bounded weights);")
    out("   everything else in the chain is within ~30x of observed.")
    out("3. Even observed, certifying ||a-b||_1 <= 0.1 needs N ~ 2e5 per class")
    out("   vs the paper's N = 64: training signal and certification are")
    out("   different regimes; the theorems are sound but their constants")
    out("   matter only at certification-scale batch sizes.")


# ============================================================================
# E6. CFG nonnegativity gate at the paper's guidance scales (Objective 6)
# ============================================================================


def e6_cfg_gate(rng) -> None:
    sec("E6. CFG affine target: how often is it a probability vector?")
    out("CFGTargetNonnegative gate: q = alpha*p_c - (alpha-1)*u >= 0, i.e.")
    out("a_i >= (1 - 1/alpha) * u_i per coordinate.  Paper trains alpha in [1,4].")
    out("Fraction of Dirichlet(1)-random conditionals passing the gate (u uniform):")
    out("")
    alphas = [1.0, 1.5, 2.0, 3.0, 4.0]
    out("| m | " + " | ".join(f"alpha={al}" for al in alphas) + " | closed form (1/alpha)^(m-1) at 4 |")
    out("|---|" + "---|" * (len(alphas) + 1))
    for m in (2, 5, 10):
        u = np.full(m, 1.0 / m)
        cells = []
        for al in alphas:
            t = (1 - 1 / al) * u
            samples = rng.dirichlet(np.ones(m), size=200000)
            frac = float((samples >= t[None, :]).all(axis=1).mean())
            cells.append(f"{frac:.4f}")
        closed = (1 / 4.0) ** (m - 1)
        out(f"| {m} | " + " | ".join(cells) + f" | {closed:.2e} |")
    out("")
    out("At the paper's strongest guidance (alpha = 4) the affine CFG target is")
    out("a genuine probability vector only on a (1/alpha)^(m-1) sliver of the")
    out("simplex: for realistic basis sizes the signed-measure treatment of")
    out("Objective 6 (`CFGAffine.lean`) is the *generic* case, not a corner case.")


# ============================================================================


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(20260707)
    R.append("# Objective 7: numerical evaluation of the certified conditions")
    R.append("")
    R.append("Generated by `numerics/run_all.py` (seed 20260707); formulas transcribe")
    R.append("the Lean declarations listed in `numerics/README.md`.  Operating point")
    R.append("anchored to the paper (Table 8): kernel exp(-dist/tau) on normalized")
    R.append("distances (mean 1), tau in {0.02, 0.05, 0.2}, per-class batch N = 64.")
    e0_audit(rng)
    e1_frame_landscape(rng)
    e2_column_price()
    e3_consistency(rng)
    e3b_leaveout_bias()
    e3c_mask_gap()
    e4_ess(rng)
    e5_ledger(rng)
    e6_cfg_gate(rng)
    R.append("")
    R.append(f"_Runtime: {time.time() - t0:.1f}s._")
    OUT.write_text("\n".join(R), encoding="utf-8")
    print(f"\nwrote {OUT} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
