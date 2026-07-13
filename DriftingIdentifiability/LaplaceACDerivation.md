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
3. Density-continuity question settled for the current Abel route: the
   pointwise common-ODE/Abel theorem needs classical C² normalizers. In Lean this
   is now an explicit `LaplaceC2NormalizerRegular` certificate
   (`LaplaceACRegularity.lean`). Continuous nonnegative density representations
   now imply that certificate in Lean
   (`laplaceC2NormalizerRegular_of_continuousDensity`,
   `LaplaceACDensityRegularity.lean`) via the classical identity
   `Z'' = (Z - 2 tau f)/tau^2`. The earlier "AC alone / no density continuity"
   wording was too optimistic and should not be used as a claim.

## The common ODE and Abel, from certified first-order data plus C² normalizers

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
certified first-order identities plus `D = m Z`, once the normalizer derivatives
`Z'` and `Z''` are supplied by the regularity certificate.

Wronskian `W := Z_p' Z_q - Z_p Z_q'`:
- `W` is differentiable pointwise under `LaplaceC2NormalizerRegular` for both
  laws; the theorem
  `hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular` constructs
  all needed `m'`, `m''`, `Z'`, and `Z''` facts and has no exposed `HasDerivAt`
  hypotheses.
- Abel: from `(**)` for both solutions, `W' = -(2 mu'/m) W` holds at every point
  where `m != 0`, under that C²-normalizer certificate.
- The density-level sufficient condition for this local regularity piece is now
  formalized: a continuous nonnegative density representation gives
  `LaplaceC2NormalizerRegular`. Exponential moments are not needed for this
  local differentiability theorem; they remain part of the separate tail and
  compactness hypotheses.
- `W` is BOUNDED and CONTINUOUS everywhere once the same regularity/atomlessness
  packaging is supplied; the propagation layer consumes continuity as an
  explicit deterministic hypothesis.

## Zero structure (from the tilted-mean linchpin)

`mu' = (1/tau) Cov_wt(y, sgn(y - x))` under the tilted law `wt ∝ k(x,.) dp`.
- `mu' >= 0` (covariance of two nondecreasing functions of `y`) — the linchpin.
- `mu` is bounded: `mu -> mu_+ := (int y e^{y/tau}dp)/(int e^{y/tau}dp)` at `+inf`
  and `mu -> mu_-` at `-inf` (finite by the exponential moment). Hence
  `range(mu) ⊆ [mu_-, mu_+]`, and every zero of `m` (where `mu(x)=x`) lies in the
  BOUNDED interval `[mu_-, mu_+]`.  This is now certified as L4 in
  `LaplaceACAsymptotics.lean`: the theorem
  `exists_bounds_for_laplaceMeanShiftRatio_zeros` gives explicit finite bounds
  for all zeros of `m = laplaceMeanShiftRatio tau p` under two-sided exponential
  first moments.
- Tail: `m = mu - x -> -inf` at `+inf` and `+inf` at `-inf`. So `m>0` left of
  `a_1`, `m<0` right of `a_M`, and `M` is ODD. Sign changes alternate: DOWNWARD
  (`+ -> -`) at `a_1,a_3,...`; UPWARD (`- -> +`) at `a_2,a_4,...,a_{M-1}`.

## Three sub-arguments giving W ≡ 0

- TAIL: `W -> 0` at `+-inf` (DCT on the one-sided masses; `P^+ -> 0` at `+inf`,
  `P^-` converges to the finite positive exponential moment, and the mirror
  statement at `-inf`).  This is now certified as L2:
  `laplaceKernelNormalizerWronskian_tendsto_atTop_zero` and
  `laplaceKernelNormalizerWronskian_tendsto_atBot_zero`.
- OUTER (sign-square): on the right outer interval, L4 gives `m < 0` and L3 gives
  `mu' >= 0`, hence `c = 2mu'/m <= 0`. From Abel `W' = -cW`, the derivative of
  `W^2` is `-2cW^2 >= 0`, so `W^2` is nondecreasing to the right. Since L2 gives
  `W -> 0` at `+inf`, every right-outer value must be zero. Symmetrically, on
  the left outer interval `m > 0`, hence `c >= 0`, `W^2` is nonincreasing to the
  right, and `W -> 0` at `-inf` forces `W ≡ 0` on the left ray. This alone
  finishes `M = 1` = 3A.
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
| L2 | Tail `W -> 0` at `+-inf` (`upperExpMass_tendsto_atTop_zero` + boundedness) | DONE under explicit two-sided exponential moments: `lowerExpMass_tendsto_atTop_integral`, `upperExpMass_tendsto_atBot_integral`, `laplaceKernelNormalizerWronskian_tendsto_atTop_zero`, `laplaceKernelNormalizerWronskian_tendsto_atBot_zero` |
| L3 | `mu_p` monotone (`mu' >= 0`) | **DONE, AXIOM-FREE** (`LaplaceTiltedMeanMonotone.lean`, `laplaceTiltedMean_monotone`): Monge/TP2 + symmetrization, no correlation-inequality axiom |
| L4 | `mu` bounded (`mu_+`, `mu_-`) + all zeros of `m` in `[mu_-,mu_+]` | DONE under explicit two-sided exponential first moments (`LaplaceACAsymptotics.lean`).  Proves the positive/negative tilted-mean tail limits, the `mu(x)-x -> -∞/+∞` consequences, and compact zero pinning via `exists_bounds_for_laplaceMeanShiftRatio_zeros` |
| L5 | common ODE `(**)` for `Z_p, Z_q`; Abel `W'=-(2mu'/m)W` | BRIDGE + REGULARITY DISCHARGE DONE under an explicit C²-normalizer certificate. `LaplaceACAbel.lean` proves the Abel algebra from derivative data; `LaplaceACRegularity.lean` defines `LaplaceC2NormalizerRegular`, derives the `m'`/`m''` data from certified first-order identities, and proves `hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular` with no exposed `HasDerivAt` hypotheses. `LaplaceACDensityRegularity.lean` now proves continuous nonnegative densities imply `LaplaceC2NormalizerRegular` |
| L6 | OUTER rays: `W ≡ 0` on the two semi-infinite sign intervals | DONE in `LaplaceACPropagation.lean` by a primitive-free sign argument. `abel_right_outer_zero_of_nonpos_coeff` and `abel_left_outer_zero_of_nonneg_coeff` show that if `W' = -cW`, `W -> 0` at the relevant tail, and `c` has the outer-ray sign, then `W` vanishes on the ray. The concrete wrappers `abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg` and `abel_left_outer_zero_of_muDeriv_nonneg_of_m_pos` consume exactly the Laplace coefficient `c = 2 mu'/m`: L3 gives `mu' >= 0`, while L4/outer sign geometry gives `m < 0` on the right outer ray and `m > 0` on the left. The older primitive/BV packages remain available but are no longer needed for L6. |
| L7 | `mu'(a_k) > 0` STRICT at crossings (zeros lie in supp interior; strict covariance) | DONE in right-derivative/straddling-mass form (`hasStrictDerivWithinAt_Ici_laplaceTiltedMeanFromDisplacement_of_twoSidedMass`), plus bridge `laplaceTiltedMean_eq_fromDisplacement`; later L8 may choose how much two-sided/classical-AC packaging it needs |
| L8 | INTERIOR blow-up: upward crossing + bounded `W` => `W ≡ 0` on flanks | DONE in `LaplaceACPropagation.lean`. The previous abstract barrier package remains, but the main route is now concrete: `primitive_tendsto_atBot_of_right_log_singularity` / left version prove `A -> -∞` from a logarithmic singular coefficient, and `abel_right_interval_zero_of_upwardCrossing_of_logSingularity` / left version feed that into the bounded Abel squeeze. The actual Laplace-coefficient wrappers `abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper` and `abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower` derive the log singularity from local `mu' >= δ > 0` plus one-sided linear control of `m` at the upward crossing. |
| L9 | finitely many sign changes + parity covering => `W ≡ 0` on `R` | **DONE at the deterministic packaging layer** in `LaplaceACPropagation.lean`: `continuous_eq_zero_of_zero_off_finset` proves a continuous `W` vanishes everywhere once the outer/flank arguments kill the complement of a finite breakpoint set; `continuous_eq_zero_of_vanishesOnOrderedGaps` packages the common finite-breakpoint cover; and `continuous_eq_zero_of_alternatingUpwardPairs` now converts the alternating down/up parity cover directly into global vanishing. |

3A needs L0-L6. 3B additionally needs L7-L9.

## Hypotheses (tightened) and what is NOT needed

Needed:
- `p, q` absolutely continuous with two-sided exponential moment
  (`int e^{+-y/tau} d(p,q) < inf`, and the first-moment weighted versions).
- For the current pointwise Abel route, `p` and `q` must additionally satisfy
  `LaplaceC2NormalizerRegular tau p/q`. This is now generated in Lean from a
  continuous nonnegative density representation
  `p = volume.withDensity (fun x => ENNReal.ofReal (rho x))` by
  `laplaceC2NormalizerRegular_of_continuousDensity`. The exponential moments
  above are not needed for that local C² implication; they are still needed for
  the tail/boundedness parts of the a.c. converse.
- `m` has FINITELY MANY sign changes. All its zeros already lie in the compact
  `[mu_-, mu_+]` (L4), so this is finiteness on a bounded interval; it is
  AUTOMATIC for real-analytic densities, and a mild explicit hypothesis otherwise.
- `mu'(a_k) > 0` at each crossing (L7); holds when `a_k` is interior to `supp p`,
  which every mean-shift zero is (`a_k = mu(a_k)` is a tilted average).

NOT needed:
- axiomatized Green identities or project-owned converse assumptions;
- SIMPLE zeros `m'(a_k) != 0` — L8 needs only a sign change + `mu' > 0`, valid for
  any `C^1` `m` (the divergence `int 1/|m| = inf` holds for any `C^1` zero);
- Levinson / Frobenius / distribution theory.

Scope: covers every case of practical interest (Gaussians, Gaussian mixtures,
bounded- or smooth-density uni/multimodal laws). "Fully general a.c." (merely
`L^1` density with possibly infinitely many sign changes) is not claimed; whether
the finiteness hypothesis is removable by a compactness/limiting argument on
`[mu_-, mu_+]` is an open refinement, not a blocker.

## Risk / feasibility (honest)

- L2: RESOLVED under the explicit exponential-moment hypotheses used by the
  a.c. theorem.
- L4: RESOLVED under the explicit two-sided exponential first-moment package.
  `LaplaceACAsymptotics.lean` proves the DCT cutoff limits, scaled wrong-side
  tail vanishing, `laplaceTiltedMean_tendsto_atTop/atBot`,
  `laplaceTiltedMean_sub_tendsto_atTop_atBot`,
  `laplaceTiltedMean_sub_tendsto_atBot_atTop`, and compact zero pinning for
  `laplaceMeanShiftRatio`.
- L6: RESOLVED. The earlier BV primitive route has been superseded by the
  primitive-free square-monotonicity sign package. On the right outer ray
  `m < 0` and `mu' >= 0`, so `c = 2mu'/m <= 0` and `W^2` is nondecreasing;
  since `W -> 0` at `+inf`, `W` must vanish on the ray. The left ray is the
  symmetric `m > 0`, `c >= 0`, `W -> 0` at `-inf` argument. No concrete tail-BV
  estimate remains for L6.
- L8: RESOLVED as a deterministic package. The concrete logarithmic barrier is
  now formalized: right side uses `mu' >= δ > 0` and `0 < m <= L(t-a)` to get
  `(2mu'/m) >= (2δ/L)/(t-a)`; left side uses `mu' >= δ > 0` and
  `-L(b-t) <= m < 0` to get `(2mu'/m) <= -(2δ/L)/(b-t)`. These singular bounds
  force the Abel primitive to tend to `-inf`, and boundedness of `W` kills both
  flanks. What remains for the final theorem is to supply these local `δ,L`
  witnesses from the selected crossing/regularity hypotheses, not to prove the
  L8 propagation/barrier mechanism.
- L9-cover: RESOLVED as a deterministic package.  The ordered-gap cover and the
  alternating down/up parity cover are both formalized; the theorem
  `continuous_eq_zero_of_alternatingUpwardPairs` converts the parity witness into
  the ordered-gap witness and then uses continuity to prove `W ≡ 0`.
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
- L5 (ODE + Abel from first-order data): RESOLVED as a Lean bridge in
  `LaplaceACAbel.lean` and discharged under explicit C²-normalizer regularity in
  `LaplaceACRegularity.lean`.  The new regularity theorem proves the actual
  zero-drift Wronskian Abel equation without caller-supplied derivative
  hypotheses.  The upstream density calculus is now also discharged:
  `LaplaceACDensityRegularity.lean` proves continuous nonnegative density
  representations imply `LaplaceC2NormalizerRegular`.  The moment hypotheses are
  still separate inputs for tails and boundedness, not for this local certificate.
- L6/L8/L9 certificate layer: the deterministic ODE/continuity endgame is now
  formalized in `LaplaceACPropagation.lean` without axioms.  L6 is now closed
  by the primitive-free sign package: the wrappers
  `abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg` and
  `abel_left_outer_zero_of_muDeriv_nonneg_of_m_pos` combine the Abel equation,
  L3 monotonicity, the L4 outer sign geometry, and the L2 tail limit to kill both
  outer rays.  The older primitive-tail-limit and tail-BV packages remain as
  reusable backup tools but are not part of the required L6 route.  L8 is now
  also closed at the deterministic level: concrete logarithmic singularity
  packages construct `A -> -inf`, and the actual `2mu'/m` wrappers derive those
  singularities from `mu' >= δ` plus one-sided linear control of `m`.  L9 now has
  the finite-set complement gluing theorem, the ordered-gap cover theorem, and
  the alternating parity-cover theorem: a finite `[down, up, down, ...]` witness
  carrying the L6 outer-ray and L8 flank vanishings implies `W ≡ 0` by
  continuity.  The final-assembly packaging is now in place; the only remaining
  automation layer is deriving the finite alternating certificate directly from
  a syntactic zero-list hypothesis such as "analytic density with finitely many
  simple sign-changing zeros."
- Final gate assembly: `LaplaceACFinal.lean` now provides the endgame socket.
  `laplaceAC_wronskian_eq_zero_of_finalAssembly` turns a concrete
  `LaplaceACFinalAssembly` certificate into `W ≡ 0`, and
  `laplaceAC_identifies_of_finalAssembly` feeds this into the certified
  Wronskian injectivity gate to conclude `p = q`.  Convenience constructors
  cover the single-downward-crossing case and the first nontrivial
  `[down, up, down]` parity case.  The same file now proves global Wronskian
  continuity from `LaplaceC2NormalizerRegular`, a concrete 3A/single-downward
  assembly theorem from L2+L5+L6, and a concrete first-3B
  `[down, up, down]` assembly theorem from L2+L5+L6+L8+L9.  It now also defines
  the arbitrary finite `LaplaceACAlternatingChain`, proves each local
  `LaplaceACUpwardCrossingCertificate` kills its two flanks, and proves the
  general arbitrary-length theorem `laplaceAC_identifies_of_alternatingChain`.
  The wrapper `laplaceAC_identifies_of_continuousDensity_finiteAlternating`
  now derives the needed C²-normalizer certificates internally from continuous
  nonnegative density representations and exposes the current theorem in its
  clean a.c. form: continuous densities + exponential moments + a finite
  alternating crossing certificate + zero drift imply `p = q`.
  The still-cleaner wrapper
  `laplaceAC_identifies_of_continuousDensity_finiteSimpleZeros` replaces those
  local crossing certificates by a finite alternating list of simple
  sign-changing zeros; the chain is converted automatically through
  `LaplaceACSimpleAlternatingZeros.toAlternatingChain`.
- L9-finiteness: a hypothesis for general a.c.; automatic for analytic densities.
  The arbitrary finite breakpoint induction is now formalized.  The lower-level
  constructor `LaplaceACUpwardCrossingCertificate.of_regular_withLocalBounds`
  generates primitive/FTC and W-boundedness fields once local `δ,L` estimates
  are supplied.  The stronger constructor
  `LaplaceACUpwardCrossingCertificate.of_regular` now derives those local
  estimates from a genuine upward sign-changing zero: adjacent signs, `m(up)=0`,
  two-sided mass for L7, continuity of `mu'`, and differentiability of `m`.
  `LaplaceACAlternatingChain.cons_regular_simpleZero` wires that constructor
  into the recursive chain.  The finite-simple-zero theorem now consumes a
  supplied finite zero/sign cover directly.  What remains beyond this theorem is
  the analytic extraction of such a zero/sign cover from a named concrete family
  such as a Gaussian mixture or a general analytic density class.

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
- 2026-07-11: TIGHTENED and UNIFIED 3A+3B. Key improvements then identified:
  (a) avoid project-owned analytic axioms by deriving the ODE/Abel bridge from
  certified first-order identities plus explicit regularity data; (b) weakened
  simple-zeros to mere sign-changes in L8; (c) all mean-shift zeros pinned to the
  compact `[mu_-,mu_+]`,
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
  `W' = -(2mu'/m)W` for two solutions of the shared ODE.  At this checkpoint,
  the project-native zero-drift/a.e. wrapper was still missing; see the next log
  entry for its completion.
- 2026-07-11: FINISHED the L5 zero-drift bridge layer. Added
  `laplaceMeanShiftRatio`, proved zero drift makes it a common ratio for both
  laws (`laplaceMeanShiftRatio_common_self`,
  `laplaceMeanShiftRatio_common_of_zeroDrift`), proved the project-native
  pointwise Abel theorem
  `hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift`, and proved the
  a.e. wrapper `ae_hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift`.
  L5 no longer has a sign/algebra/zero-drift gap; the next work is deriving the
  explicit differentiability hypotheses from a.c. exp-moment regularity.
- 2026-07-11: FINISHED L2 tail assembly. Added dominated-convergence moment
  limits `lowerExpMass_tendsto_atTop_integral` and
  `upperExpMass_tendsto_atBot_integral`, then proved the named normalizer
  Wronskian satisfies `W -> 0` at both tails under the matching exponential
  moments:
  `laplaceKernelNormalizerWronskian_tendsto_atTop_zero` and
  `laplaceKernelNormalizerWronskian_tendsto_atBot_zero`.
- 2026-07-11: advanced L6/L8/L9 by adding `LaplaceACPropagation.lean`,
  axiom-free.  New deterministic propagation tools:
  `abel_integratingFactor_const_Icc`, right/left tail squeezes
  (`abel_right_tail_zero_of_integratingFactor`,
  `abel_left_tail_zero_of_integratingFactor`), the L8 boundedness squeeze
  (`bounded_integratingFactor_const_zero_of_factor_tendsto_zero`), and finite
  interval zero continuation (`abel_zero_propagates_Icc`).  These close the
  reusable Abel-propagation core; remaining work is to derive the primitive
  finite-limit / zero-limit inputs from the AC sign-change hypotheses and then
  formalize the finite parity cover.
- 2026-07-11: strengthened L6/L8/L9 to certificate theorems.  L6 now has
  right/left outer-ray vanishing from a primitive with finite tail limit:
  `abel_right_outer_zero_of_integratingFactor_of_tendsto_primitive`,
  `abel_left_outer_zero_of_integratingFactor_of_tendsto_primitive`.  L8 now has
  right/left upward-crossing interval vanishing from primitive divergence
  `A -> -inf` and bounded `W`:
  `abel_right_interval_zero_of_upwardCrossing_of_tendsto_atBot`,
  `abel_left_interval_zero_of_upwardCrossing_of_tendsto_atBot`.  L9 now has the
  finite-breakpoint continuity gluing theorem
  `continuous_eq_zero_of_zero_off_finset`.  These are axiom-free.  The remaining
  work is no longer the propagation/gluing logic itself; it is constructing the
  finite-limit/divergence/finite-cover certificates from the raw a.c.
  mean-shift hypotheses.
- 2026-07-11: discharged the L5 Abel bridge's exposed regularity hypotheses
  under an explicit C²-normalizer certificate. Added
  `LaplaceACRegularity.lean`, defining `LaplaceC2NormalizerRegular`, deriving
  `laplaceMeanShiftRatioDeriv` and `laplaceMeanShiftRatioSecondDeriv` from the
  certified first-order Laplace identities plus the normalizer certificate, and
  proving
  `hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular`.  The new
  theorem has no leftover `HasDerivAt` assumptions in its statement.
- 2026-07-11: closed the continuous-density regularity bridge. Added
  `LaplaceACDensityRegularity.lean`, proving the interval-FTC derivatives for
  `lowerExpMass` and `upperExpMass`, assembling
  `Z' = laplaceKernelNormalizerRightDerivCoeff`, proving
  `Z'' = (Z - 2 tau rho)/tau^2`, and packaging the result as
  `laplaceC2NormalizerRegular_of_continuousDensity`.  This is axiom-free and
  requires only finite measure plus a continuous nonnegative density
  representation; exponential moments remain separate tail/compactness
  hypotheses.
- 2026-07-12: FINISHED L4 under explicit two-sided exponential first moments.
  Added `LaplaceACAsymptotics.lean`, defining the moment package
  `LaplaceTwoSidedExpFirstMoment`, the two tilted-mean tail limits, DCT cutoff
  limits for `∫ y e^{±y/tau}`, scaled wrong-side tail vanishing, scaled formulae
  for the tilted mean, and the compact zero-pinning theorem
  `exists_bounds_for_laplaceMeanShiftRatio_zeros`.  This proves that all zeros
  of `m = D/Z` lie in a bounded interval once the moment hypotheses hold.  No
  new axioms.
- 2026-07-12: advanced L6 upstream packaging.  Added the tail-Cauchy/BV
  constructors `cauchy_map_atTop_of_tail_norm_sub_le`,
  `exists_tendsto_atTop_of_tail_norm_sub_le`, and the left-tail analogue, then
  wrapped the existing outer-ray Abel certificate as
  `abel_right_outer_zero_of_tail_bvPrimitive` / left version.  This closes the
  abstract "BV estimate gives finite primitive tail limit" step without axioms.
  This route is now backup infrastructure rather than the main L6 route.
- 2026-07-12: advanced L8 upstream packaging.  Added the order squeeze
  `tendsto_atBot_of_eventually_le_of_tendsto_atBot` and wrapped the existing
  upward-crossing certificates as
  `abel_left_flank_zero_of_integratingFactor_of_atBotBarrier`,
  `abel_right_flank_zero_of_integratingFactor_of_atBotBarrier`,
  `abel_right_interval_zero_of_upwardCrossing_of_atBotBarrier`, and the left
  interval version.  This closes the abstract "upper barrier gives primitive
  divergence" step without axioms.  This route is now backup infrastructure;
  the concrete logarithmic-singularity route below is the main L8 route.
- 2026-07-12: advanced L9 upstream packaging.  Added
  `VanishesOnOrderedGapsFrom`, `VanishesOnOrderedGaps`, their pointwise
  off-breakpoint lemmas, and
  `continuous_eq_zero_of_vanishesOnOrderedGaps`.  This closes the abstract
  finite ordered-breakpoint cover step: once L6/L8 provide vanishing on the
  left ray, every adjacent open gap, and the right ray of a finite breakpoint
  list, continuity glues the breakpoint values.  The later parity-cover theorem
  below now builds this ordered-gap witness from the alternating
  sign-change/parity structure.
- 2026-07-12: FULLY CLOSED L6 by replacing the remaining concrete BV estimate
  with a primitive-free sign argument.  Added
  `abel_square_left_le_right_of_nonpos_coeff`,
  `abel_square_right_le_left_of_nonneg_coeff`, the tail-zero square squeeze
  lemmas, and the outer-ray theorems
  `abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg` /
  `abel_left_outer_zero_of_muDeriv_nonneg_of_m_pos`.  The proof uses only
  `W' = -(2mu'/m)W`, `mu' >= 0`, the L4 outer signs of `m`, and `W -> 0` at the
  corresponding tail.  No primitive, BV estimate, or new axiom is needed.
- 2026-07-12: FULLY CLOSED L8 at the deterministic barrier layer.  Added the
  concrete logarithmic barrier package:
  `primitive_tendsto_atBot_of_right_log_singularity` / left version prove that
  a primitive `A' = c` tends to `-inf` from the singular bounds
  `γ/(t-a) <= c(t)` and `c(t) <= -γ/(b-t)`.  Added algebraic Laplace-coefficient
  sources
  `right_laplaceCoeff_logSingularity_of_muDeriv_lower_m_upper` and
  `left_laplaceCoeff_logSingularity_of_muDeriv_lower_m_lower`, then wrapped the
  actual Abel interval conclusions as
  `abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper` and
  `abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower`.  The
  remaining final-assembly task is to provide the local `δ,L` witnesses from the
  chosen strict-crossing/regularity hypotheses; the L8 analytic mechanism itself
  is now axiom-free and closed.
- 2026-07-12: FULLY CLOSED L9 at the deterministic parity-cover layer.  Added
  `VanishesOnAlternatingUpwardPairsFrom`,
  `VanishesOnAlternatingUpwardPairs`,
  `VanishesOnOrderedGapsFrom.of_alternatingUpwardPairs`,
  `VanishesOnOrderedGaps.of_alternatingUpwardPairs`, and
  `continuous_eq_zero_of_alternatingUpwardPairs`.  A finite alternating
  `[down, up, down, ...]` cover now directly feeds the existing ordered-gap and
  continuity gluing theorem, with no new axioms.
- 2026-07-12: ADDED FINAL ASSEMBLY SOCKET.  `LaplaceACFinal.lean` defines
  `LaplaceACFinalAssembly` for the actual normalizer Wronskian, proves
  `laplaceAC_wronskian_eq_zero_of_finalAssembly`, and composes with the
  certified gate as `laplaceAC_identifies_of_finalAssembly`.  This closes the
  formal L9-to-`p=q` composition.  The remaining upstream work is now sharply
  localized: construct the concrete final-assembly certificate from the chosen
  zero/sign-change hypotheses and local crossing regularity.
- 2026-07-12: ADVANCED FINAL ASSEMBLY UPSTREAM.  Added
  `continuous_laplaceKernelNormalizerWronskian_of_regular`, discharging the
  global continuity field of `LaplaceACFinalAssembly` from the same C²
  regularity certificate used by L5.  Added
  `laplaceACFinalAssembly_singleDown_of_outerSigns` and
  `laplaceAC_identifies_singleDown_of_outerSigns`, which assemble the full 3A
  single-crossing route from L2 tails, the L5 Abel equation, L6 outer rays, and
  the certified Wronskian gate.  Added
  `laplaceACFinalAssembly_downUpDown_of_outerSigns_of_upwardCrossing` and
  `laplaceAC_identifies_downUpDown_of_outerSigns_of_upwardCrossing`, packaging
  the first nontrivial 3B pattern `[down, up, down]` via L6 outer rays and L8
  flank blow-up.  Remaining work is now the arbitrary-length parity-list
  induction and automatic production of local `δ,L` witnesses.
- 2026-07-12: CLOSED ARBITRARY-LENGTH FINAL ASSEMBLY.  Added the local
  certificate structure `LaplaceACUpwardCrossingCertificate`, its flank theorem
  `LaplaceACUpwardCrossingCertificate.vanishes`, the recursive finite
  `LaplaceACAlternatingChain`, and the general theorem
  `laplaceAC_identifies_of_alternatingChain`.  This handles any finite
  `[down, up, down, ..., down]` list by induction, using L6 for the two outer
  rays, L8 for every upward crossing, L9 for gluing, and the certified gate for
  `p = q`.  The remaining polish layer is to generate the local crossing
  certificates automatically from smoothness/strict-crossing hypotheses rather
  than supplying them as explicit certificate data.
- 2026-07-12: WIRED L3/L7 AND CONTINUOUS-DENSITY END-TO-END WRAPPER.  Added
  `hasDerivAt_laplaceTiltedMean_regular` and
  `laplaceMeanShiftRatioDeriv_add_one_nonneg_of_regular`, so every final
  assembly theorem now discharges `0 <= mu'` from the proved monotonicity theorem
  rather than exposing it as a hypothesis.  Added the strict bridge
  `laplaceMeanShiftRatioDeriv_add_one_pos_of_twoSidedMass_regular`, converting
  the L7 two-sided-mass theorem into positivity of the exact Abel coefficient
  `laplaceMeanShiftRatioDeriv + 1`.  Added
  `LaplaceACUpwardCrossingCertificate.of_regular`, which constructs the L8
  primitive/FTC and W-boundedness fields from `LaplaceC2NormalizerRegular` once
  the local sign, `delta`, and linear `m` bounds are supplied.  Added
  `LaplaceACAlternatingChain.cons_regular` and the clean theorem
  `laplaceAC_identifies_of_continuousDensity_finiteAlternating`, deriving C²
  regularity from continuous densities by
  `laplaceC2NormalizerRegular_of_continuousDensity`.  This entry predates the
  later Gaussian non-vacuity closure below; the standard-Gaussian
  finite-simple-zero inhabitant is now formalized in
  `LaplaceACGaussianCertificate.lean`.
- 2026-07-12: CLOSED LOCAL UPWARD-CROSSING CERTIFICATE AUTOMATION.  Renamed the
  earlier semi-regular constructor to
  `LaplaceACUpwardCrossingCertificate.of_regular_withLocalBounds`, then made
  `LaplaceACUpwardCrossingCertificate.of_regular` the fully automatic local
  constructor from a sign-changing zero.  It chooses a neighborhood where
  `laplaceMeanShiftRatioDeriv + 1` is bounded below using L7 strictness plus
  continuity, obtains a one-sided linear bound for
  `laplaceMeanShiftRatio` from differentiability at the zero via `dslope`, and
  feeds those witnesses into the existing primitive/FTC/W-boundedness
  constructor.  Added `LaplaceACAlternatingChain.cons_regular_simpleZero` so
  finite alternating chains can consume these zeros directly.
- 2026-07-12: CLOSED THE CLEAN FINITE-SIMPLE-ZERO THEOREM SURFACE.  Added
  `LaplaceACSimpleZero`, `LaplaceACSimpleUpwardCrossing`,
  `LaplaceACSimpleAlternatingZeros`, and the conversion
  `LaplaceACSimpleAlternatingZeros.toAlternatingChain`.  Added
  `LaplaceACContinuousDensityFiniteSimpleZeros` and the end-to-end theorem
  `laplaceAC_identifies_of_continuousDensity_finiteSimpleZeros`, plus the
  `IdentifiesAtZero` wrapper
  `laplaceAC_identifiesAtZero_of_continuousDensity_finiteSimpleZeros`.
  Also added Gaussian density/moment lemmas and a concrete distinct pair hook.
- 2026-07-12: CLOSED THE STANDARD-GAUSSIAN NON-VACUITY GAP.  Added
  `LaplaceACGaussianCertificate.lean`, which constructs the
  `StandardGaussianLaplaceSingleDownCertificate` axiom-free for every positive
  bandwidth.  The proof reduces signs to the displacement numerator, proves
  oddness at the origin by Gaussian symmetry, proves strict one-sided signs by
  pairing `x+s` with `x-s`, and proves the simple-zero derivative by a
  half-line integration-by-parts identity for
  `s * exp (-s / τ) * φ(s)`.  The public witness
  `standardGaussian_vs_shiftedGaussian_finiteSimpleZeros`, the distinct-pair
  theorem, and the legitimacy theorem are now hypothesis-free apart from
  `ValidBandwidth τ`.
