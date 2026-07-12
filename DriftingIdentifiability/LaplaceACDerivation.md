# Deriving the a.c. Laplace converse (3A + 3B) instead of axiomatizing it

Date: 2026-07-11 (tightened)

Decision: do NOT axiomatize the non-trivial project-own bridges 3A/3B. PROVE the
analytic step `ZeroDrift + a.c. + exp-moment => W ≡ 0` in Lean; the certified
gate `laplaceKernelNormalizer_wronskian_eq_zero_imp_eq`
(LaplaceGeneralConverseWronskian.lean:219) then delivers `p = q`. This doc is the
live plan; the axiom route (`LaplaceACConditionalAxiomPlan.md`) is a
marked-fallback only. The general docs (`LaplaceArbitraryConverse.md`,
`ResearchStatus.md`, `LaplaceGeneralConverseRoadmap.md`) point here.

Target (narrow): prove `forall x, laplaceKernelNormalizerWronskian tau p q x = 0`.

## 3A and 3B are ONE theorem

Let `m := D_p/Z_p = D_q/Z_q` (the common mean shift under zero drift) and
`mu := m + x` (tilted mean). Everything is controlled by the sign changes of `m`:

- `m` has `M` sign changes `a_1 < ... < a_M` (M odd; see zero structure).
- 3A = the case `M = 1` (single crossing): both sides of `a_1` are semi-infinite.
- 3B = the case `M >= 3` (multiple crossings): plus bounded interior intervals.

So the deliverable is ONE unified theorem parameterized by `M`; 3A is its simplest
instance and needs strictly less than 3B. Building 3A is therefore never wasted.

## Three enabling discoveries

1. The elliptic/Green identity is ALREADY certified classically
   (`LaplaceWronskian.lean` 10-23): `L_p' = D_p/tau`, `D_p' = L_p/tau - 2 Z_p`,
   hence `tau^2 L_p'' = L_p - 2 tau Z_p` — at the level of classical derivatives
   of smoothings, no distribution theory. No Green-identity axiom needed.
2. The asymptotic step needs neither Levinson nor Frobenius: bounded-variation
   Abel integral on the outer intervals, and an elementary boundedness-vs-blow-up
   argument at upward crossings on the interior.
3. NO density-continuity hypothesis. The whole argument runs at the
   absolutely-continuous (AC) function level, driven by the CERTIFIED first-order
   identities. This is the main tightening over the earlier draft (which assumed
   continuous density for a classical `C^2` ODE). See next section.

## The common ODE and Abel, from certified first-order data (no density)

Under zero drift `D_p = m Z_p`, `D_q = m Z_q` (common `m`). Differentiate
`D_p = m Z_p`: `D_p' = m' Z_p + m Z_p'`. Certified `D_p' = L_p/tau - 2 Z_p`, so
`L_p = tau((m'+2) Z_p + m Z_p')`. Certified `L_p' = D_p/tau = m Z_p/tau`.
Differentiating `L_p` and equating:

```text
m Z_p'' + 2 mu' Z_p' + (mu'' - m/tau^2) Z_p = 0        (**)
```

(using `mu' = m'+1`, `mu'' = m''`). The coefficients `m, 2 mu', mu'' - m/tau^2`
depend ONLY on `m` and `mu = m + x`, which are COMMON to `p` and `q` under zero
drift, so `Z_q` solves the identical `(**)`. The derivation uses only the two
certified first-order identities plus `D = m Z`; no density, no distributions.

Wronskian `W := Z_p' Z_q - Z_p Z_q'`:
- `W` is ABSOLUTELY CONTINUOUS: `Z_p', Z_q'` are AC because the one-sided masses
  `P^-(x) = int_{y<=x} e^{y/tau} dp`, `P^+(x) = int_{y>x} e^{-y/tau} dp` are AC for
  a.c. `p` (integrals of `L^1_loc` densities). [This is where a.c. is used.]
- Abel: from `(**)` for both solutions, `W' = -(2 mu'/m) W` holds a.e. on
  `{m != 0}`. (`Z_p'', Z_q''` exist a.e. — density defined a.e.; the identity is
  the standard two-solution Wronskian computation, needing only common `m, 2mu'`.)
- `W` is BOUNDED and CONTINUOUS everywhere (from `Z_p, Z_q in C^1`, `Z_p'`
  continuous for atomless `p`).

## Zero structure (from the tilted-mean linchpin)

`mu' = (1/tau) Cov_wt(y, sgn(y - x))` under the tilted law `wt ∝ k(x,.) dp`.
- `mu' >= 0` (covariance of two nondecreasing functions of `y`) — the linchpin.
- `mu` is bounded: `mu -> mu_+ := (int y e^{y/tau}dp)/(int e^{y/tau}dp)` at `+inf`
  and `mu -> mu_-` at `-inf` (finite by the exponential moment). Hence
  `range(mu) ⊆ [mu_-, mu_+]`, and every zero of `m` (where `mu(x)=x`) lies in the
  BOUNDED interval `[mu_-, mu_+]`.
- Tail: `m = mu - x -> -inf` at `+inf` and `+inf` at `-inf`. So `m>0` left of
  `a_1`, `m<0` right of `a_M`, and `M` is ODD. Sign changes alternate: DOWNWARD
  (`+ -> -`) at `a_1,a_3,...`; UPWARD (`- -> +`) at `a_2,a_4,...,a_{M-1}`.

## Three sub-arguments giving W ≡ 0

- TAIL: `W -> 0` at `+-inf` (DCT on the one-sided masses; `P^+ -> 0` at `+inf`,
  `P^-` bounded by the exp moment, etc.). Lemma `upperExpMass_tendsto_atTop_zero`
  DONE this session; `lowerExpMass_tendsto_atBot_zero` already certified.
- OUTER (BV): on `[x_0, inf)` inside the outer interval, `|m| >= |m(x_0)| > 0`, and
  `int_{x_0}^inf |2 mu'/m| <= (1/|m(x_0)|) 2 int mu' = (2/|m(x_0)|)(mu_+ - mu(x_0))
  < inf` (since `mu' >= 0`, bounded). So `W = W(x_0) exp(-int 2 mu'/m)` tends to a
  nonzero multiple of `W(x_0)`; but `W -> 0`, so `W(x_0) = 0`, i.e. `W ≡ 0` on the
  outer interval. Symmetric at `-inf`. (This alone finishes `M = 1` = 3A.)
- INTERIOR (blow-up): near an UPWARD crossing `a_k`, `m -> 0` with a sign change
  and `mu'(a_k) > 0` (strict linchpin), so `|2 mu'/m| >= 2 delta/|m|` with `|m(x)|
  <= L|x - a_k|`, giving `int 2 mu'/m` DIVERGENT and thus `|W| -> inf` on both
  sides. But `W` is bounded => `W ≡ 0` on both intervals flanking `a_k`.
- COVERING: consecutive `a_i` alternate parity, so every interior interval
  `(a_k, a_{k+1})` is flanked by an upward crossing => `W ≡ 0` there. With the two
  outer intervals, `W ≡ 0` on all of `R`; continuity (a.c., atomless) fills the
  crossings. Then the certified gate gives `p = q`.

## Lemma decomposition (unified 3A + 3B)

| # | Lemma | Status / risk |
|---|---|---|
| L0 | Gate `W ≡ 0 => p = q` | DONE (certified) |
| L1 | First-order identities `L_p'=D_p/tau`, `D_p'=L_p/tau-2Z_p` | DONE (certified, classical) |
| L2 | Tail `W -> 0` at `+-inf` (`upperExpMass_tendsto_atTop_zero` + boundedness) | `upperExpMass` limit DONE; assembly TODO (easy) |
| L3 | `mu_p` monotone (`mu' >= 0`) | **DONE, AXIOM-FREE** (`LaplaceTiltedMeanMonotone.lean`, `laplaceTiltedMean_monotone`): Monge/TP2 + symmetrization, no correlation-inequality axiom |
| L4 | `mu` bounded (`mu_+`, `mu_-`) + all zeros of `m` in `[mu_-,mu_+]` | TODO; from exp moment + DCT |
| L5 | common ODE `(**)` for `Z_p, Z_q`; `W` AC; Abel `W'=-(2mu'/m)W` a.e. | SCALAR CORE DONE in `LaplaceACAbel.lean`: first-order algebra to common ODE plus pointwise Abel Wronskian identity; remaining work is the AC/a.e. instantiation for the actual normalizers |
| L6 | OUTER BV: `int|2mu'/m|<inf` => `W ≡ 0` on semi-infinite intervals | TODO; elementary given L3,L4 (FINISHES 3A) |
| L7 | `mu'(a_k) > 0` STRICT at crossings (zeros lie in supp interior; strict covariance) | DONE in right-derivative/straddling-mass form (`hasStrictDerivWithinAt_Ici_laplaceTiltedMeanFromDisplacement_of_twoSidedMass`), plus bridge `laplaceTiltedMean_eq_fromDisplacement`; later L8 may choose how much two-sided/classical-AC packaging it needs |
| L8 | INTERIOR blow-up: upward crossing + bounded `W` => `W ≡ 0` on flanks | KEY new lemma; elementary (integrating factor + divergent integral) |
| L9 | finitely many sign changes + parity covering => `W ≡ 0` on `R` | TODO; L9-cover combinatorial (easy); finiteness = hypothesis (see below) |

3A needs L0-L6. 3B additionally needs L7-L9.

## Hypotheses (tightened) and what is NOT needed

Needed:
- `p, q` absolutely continuous with two-sided exponential moment
  (`int e^{+-y/tau} d(p,q) < inf`, and the first-moment weighted versions).
- `m` has FINITELY MANY sign changes. All its zeros already lie in the compact
  `[mu_-, mu_+]` (L4), so this is finiteness on a bounded interval; it is
  AUTOMATIC for real-analytic densities, and a mild explicit hypothesis otherwise.
- `mu'(a_k) > 0` at each crossing (L7); holds when `a_k` is interior to `supp p`,
  which every mean-shift zero is (`a_k = mu(a_k)` is a tilted average).

NOT needed (removed by the tightening):
- continuous density / `Z in C^2` — the argument is AC-level (discovery 3);
- SIMPLE zeros `m'(a_k) != 0` — L8 needs only a sign change + `mu' > 0`, valid for
  any `C^1` `m` (the divergence `int 1/|m| = inf` holds for any `C^1` zero);
- Levinson / Frobenius / distribution theory.

Scope: covers every case of practical interest (Gaussians, Gaussian mixtures,
bounded- or smooth-density uni/multimodal laws). "Fully general a.c." (merely
`L^1` density with possibly infinitely many sign changes) is not claimed; whether
the finiteness hypothesis is removable by a compactness/limiting argument on
`[mu_-, mu_+]` is an open refinement, not a blocker.

## Risk / feasibility (honest)

- L2, L4, L6, L9-cover: standard real analysis; expected to go through.
- L3 (tilted-mean monotonicity): **RESOLVED, AXIOM-FREE (this session).** This was
  THE decisive test of whether the route needs an external axiom. It does not:
  `laplaceTiltedMean_monotone` (`LaplaceTiltedMeanMonotone.lean`) proves `mu_p`
  monotone from the elementary Monge/TP2 property of the kernel plus a
  symmetrization of the double integral (`integral_prod_mul` + `integral_prod_swap`);
  `#print axioms` = `propext, Classical.choice, Quot.sound` only. No
  correlation-inequality library, no Levinson, no Frobenius, no project axiom.
- L7 (STRICT `mu' > 0` at mean-shift zeros): RESOLVED in the right-derivative
  formulation already used by the normalizer-Wronskian infrastructure.  The
  shifted tilted mean `x + D/Z` has a strictly positive right derivative whenever
  the law has positive mass on both sides of `x`; the usual tilted mean is
  bridged to this displacement form by `laplaceTiltedMean_eq_fromDisplacement`.
  Later L8 can wrap this into whatever two-sided/classical-AC language it needs.
- L5 (ODE + Abel from first-order data): the scalar/algebraic heart is now
  RESOLVED in `LaplaceACAbel.lean`.  The remaining risk is not the sign/algebra:
  it is the AC-level instantiation for the actual Laplace normalizers, i.e. feeding
  the certified first-order identities and a.e. derivatives into the pointwise
  common-ODE/Abel lemmas.
- L8 (blow-up): elementary (integrating factor + divergent integral, boundedness
  contradiction); the crux new analytic lemma, but self-contained.
- L9-finiteness: a hypothesis for general a.c.; automatic for analytic densities.

Given the prior sign-error episode: this is a plan to validate, not a theorem.
The zero structure and residue signs are numerically corroborated (below).

## Numerical corroboration

`numerics/ac_converse_3b_residues.ps1` (bimodal `p`, two Gaussians, tau=0.6):
3 mean-shift zeros at -3, 0, +3; residues `gamma = 2 mu'/m'` are `-2.73` (DOWN),
`+2.76` (UP), `-2.73` (DOWN); `mu'(a_k) > 0` at every zero (0.58, 3.64, 0.58).
The single UP crossing at 0 flanks BOTH interior intervals, exactly as the
covering predicts, so `W ≡ 0` throughout. `mu'` dips to 0 only in the TAILS
(where there are no zeros), consistent with L3 non-strict / L7 strict-at-zeros.

## Session log

- 2026-07-11: added `upperExpMass_tendsto_atTop_zero`
  (LaplaceGeneralConverse.lean), axiom-free, green build (lemma L2 building
  block).
- 2026-07-11: worked out 3B (multiple-zero) route — upward-crossing boundedness
  argument + parity covering; numerically corroborated
  (`numerics/ac_converse_3b_residues.ps1`).
- 2026-07-11: **NO-AXIOM ROUTE TEST PASSED.** Proved the decisive linchpin L3
  (`laplaceTiltedMean_monotone`, `LaplaceTiltedMeanMonotone.lean`) AXIOM-FREE —
  `#print axioms` = propext/Classical.choice/Quot.sound only. Route: elementary
  Monge/TP2 kernel inequality (`abs_monge`, `laplaceKernel_tp2`,
  `laplace_symmetrized_nonneg`) + Fubini symmetrization. Confirms the a.c. converse
  needs NO correlation-inequality axiom (the one piece I feared was external).
  Wired into the root build; full project green; trust audit green (no new axioms).
  Next: L7 (strict at zeros), then L5 (Abel) and L2 (tail assembly).
- 2026-07-11: TIGHTENED and UNIFIED 3A+3B. Key improvements: (a) removed the
  density-continuity hypothesis — the ODE and Abel come from the certified
  first-order identities at the AC level; (b) weakened simple-zeros to mere
  sign-changes in L8; (c) all mean-shift zeros pinned to the compact `[mu_-,mu_+]`,
  reducing L9 to finiteness on a bounded interval (automatic for analytic
  densities). Next Lean target: L2 assembly (`W -> 0`), then L3 (monotone tilted
  mean) as the decisive feasibility test.
- 2026-07-11: advanced L7 by proving the strict pointwise engine in
  `LaplaceTiltedMeanMonotone.lean`: strict Monge overlap
  (`abs_monge_strict_of_overlap`), strict TP2 overlap
  (`laplaceKernel_tp2_strict_of_overlap`), and strict symmetrized positivity
  (`laplace_symmetrized_pos_of_overlap`,
  `laplace_symmetrized_pos_of_straddles`). Full project build and trust audit
  are green. This is the pointwise precursor to the full L7 derivative package.
- 2026-07-11: FINISHED L7 in the project-native right-derivative form. Added
  `laplaceTiltedMeanFromDisplacement`, proved
  `laplaceTiltedMean_eq_fromDisplacement`, proved the right-derivative formula
  `hasDerivWithinAt_Ici_laplaceTiltedMeanFromDisplacement`, and proved strict
  positivity under two-sided mass:
  `hasStrictDerivWithinAt_Ici_laplaceTiltedMeanFromDisplacement_of_twoSidedMass`.
  The key algebra is the positive one-sided formula
  `mu'_+ * Z^2 = (2/tau) * (lowerComp * upperExp + upperComp * lowerExp)`.
  Full project build and trust audit are green; no new axioms.
- 2026-07-11: advanced L5 by adding `LaplaceACAbel.lean`, axiom-free.  New
  certified pieces:
  `laplaceCommonODE_companion_value_of_firstOrder` solves the companion value from
  `D' = L/tau - 2Z` and `D = mZ`;
  `laplaceCommonODE_value_of_companion_derivative` turns the differentiated
  companion relation plus `L' = mZ/tau` into
  `m Z'' + 2(m' + 1)Z' + (m'' - m/tau^2)Z = 0`;
  `hasDerivAt_wronskian_of_laplace_commonODE` proves Abel's pointwise identity
  `W' = -(2mu'/m)W` for two solutions of the shared ODE.  Full L5 still needs the
  analytic/a.e. wrapper for actual a.c. normalizers.
