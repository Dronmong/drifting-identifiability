# The Laplace Endgame: routes to the unconditional 1-d converse

**Date: 2026-07-13 (research pass, no Lean written).**  This file records the
exploration and planning pass for closing the remaining open parts of the
arbitrary-target converse conjecture: *does pointwise zero raw mean-shift
drift `ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q` force `p = q` for
arbitrary probability measures on ℝ?*

> **★★★ UPDATE (2026-07-13, later the same day): FRONTIER A IS DONE — the
> conjecture is CLOSED, machine-checked and axiom-free.**
> `DriftingIdentifiability/LaplaceUnconditionalConverse.lean` proves
> `laplaceZeroDrift_identifies`: zero raw Laplace drift identifies **arbitrary**
> probability measures on ℝ — probability hypotheses only, no atomlessness, no
> moment, no density.  Every "[PD]" item in §1.1–§1.2 below is now "[MC]": the
> one-sided K-route went through essentially as designed.  `Check.ps1` green
> (58 files, **374 promoted declarations**, 15 paper + 5 conditional axioms
> unchanged); `#print axioms` on the headline = `propext, Classical.choice,
> Quot.sound` only; `--wfail`-clean.  Milestones A1–A5 all landed in one file
> (~1070 lines).  The Milestone-6 umbrella, the atomless theorem, the
> continuous-density theorem, and the "atoms on interval supports" gap are all
> now strict corollaries, and the `p`-first-moment wart is gone.  **Frontier C
> (Gaussian-kernel arbitrary converse) turned out to be ALREADY CLOSED in the
> repo (`gaussianMeanShiftDrift_identifiesAtZero`, arbitrary dimension) — see the
> correction in §4; `GaussianArbitraryConverse.lean` adds the direct-`p=q` and
> legitimacy surface for parity.  Remaining frontier is now higher-dimensional
> Laplace / Matérn kernels (§5, Frontier D) — whose dedicated research pass is
> recorded in `LaplaceHigherDim.md` (2026-07-14): the ℓ¹/product n-d case
> reduces to the 1-d theorem and is implementation-ready; the ℓ²/radial case
> has a structural framework and staged partials, endgame open.**
>
> Implementation notes worth keeping: (i) `laplaceMeanShiftRatioDeriv` and
> `laplaceKernelNormalizerWronskian` were *already* defined through the
> right-derivative coefficient, so the K-identities held for all measures once
> phrased one-sidedly — no redefinition needed.  (ii) At a zero of `m` the
> numerator `D` vanishes too, so `m` is genuinely two-sided differentiable
> there and the existing `exists_Ioo_linear_bound_of_hasDerivAt_zero` edge
> helper was reused verbatim (`hasDerivAt_laplaceMeanShiftRatio_of_root`).
> (iii) The moment dropped out via `laplaceTiltedMeanRightDerivCoeff ≥ 0` plus
> one-sided uniqueness (`UniqueDiffWithinAt.eq_deriv`).  (iv) Right-continuity
> of the one-sided masses is *easier* than the atomless two-sided version (no
> exceptional set); measurability is free from monotonicity/antitonicity.
> (v) The global bound `|Z'⁺| ≤ 1/τ` plus compactness for the two denominators
> `Z²`, `|m|` gave interval-integrability of the càdlàg coefficient, feeding
> the standard right-FTC (`integral_hasDerivWithinAt_right`) through an `Icc`
> sub-interval for the `FTCFilter` instance.  Mathlib name drift encountered:
> `le_or_lt`→`le_or_gt`, `abs_add`→`abs_add_le`, `div_le_div_iff`→
> `div_le_iff₀`/`le_div_iff₀`.

Verification legend used throughout:

- **[MC]** machine-checked in the repo today, hypothesis-verified at the cited
  `file:line` during this pass;
- **[PD]** paper-derived in this pass, believed correct, *not yet formalized*;
- **[?]** needs a check at implementation time.

The headline of this pass: **the conjecture appears to be true and provable
unconditionally — atoms and all — with existing repo machinery, because every
ingredient of the atomless K-route survives arbitrary measures once restated
with one-sided (right) derivatives, and the abstract propagation layer was
already built one-sided.**  A second, independent finding: the *Gaussian*-kernel
arbitrary-target converse is nearly free, because for the Gaussian kernel the
displacement integral is exactly `σ²·Z′` and the smoothing-injectivity endgame
already exists in the repo.

---

## 0. Where the conjecture stands

Current machine-checked coverage (all axiom-free at the theorem level):

| Class of pairs (p, q) | Theorem | Status |
|---|---|---|
| Right-dense zero-mass gaps (nowhere-dense supports: finite mixtures, Diracs, Cantor-type supports with gaps) | `laplace...` M3 route, `LaplaceGeneralConverseNowhereDense.lean` (`RightDenseZeroMassGaps` at :62) | [MC] |
| Continuous densities, no zero-set hypothesis (needs exponential moments via the a.c. package) | `laplaceAC_identifies_of_continuousDensity`, `LaplaceACFinal.lean` | [MC] |
| Atomless × atomless + `p` first moment (rough L¹ densities, singular-continuous laws, mixtures thereof) | `laplaceZeroDrift_identifies_of_noAtoms`, `LaplaceAtomlessConverse.lean:955` | [MC] |
| Umbrella: (gaps) ∨ (atomless + moment) | `laplaceZeroDrift_identifies_real`, `LaplaceRealConverse.lean` | [MC] |
| **Atoms on interval supports** (e.g. `½·Leb|[0,1] + ½·δ_{1/2}`) | — | **OPEN** |
| **Atomless without the `p` first moment** | — | **OPEN (wart)** |

The two open rows are exactly what Frontier A below closes — not by growing
the umbrella again, but by proving the *unconditional* statement that
subsumes every row of this table.

### The reduction that frames everything

`laplaceZeroDrift_imp_eq_of_companionAligned` (`LaplaceWronskian.lean:300`)
**[MC]** takes only `[IsProbabilityMeasure p] [IsProbabilityMeasure q]`, zero
drift, and the companion alignment `∀ x, L_p·Z_q = L_q·Z_p`, and concludes
`p = q`.  No moments, no density, no atomless.  Hence:

> **The whole remaining conjecture is the single scalar statement:**
> for arbitrary probability measures, zero drift ⟹ `K ≡ 0`, where
> `K := laplaceCompanionAlignmentDefect τ p q = L_p·Z_q − L_q·Z_p`
> (`LaplaceGeneralConverseCompanionWronskian.lean:47`).

Notation used below: `Z_ν = kernelNormalizer (laplaceKernel τ) ν`,
`D_ν x = ∫ (y−x)·k(x,y) dν`, `L_ν = kernelNormalizer (laplaceCompanionKernel τ) ν`,
`m = laplaceMeanShiftRatio τ p = D_p/Z_p` (the *common* mean-shift ratio:
under zero drift `D_ν = m·Z_ν` for both ν — `LaplaceACAbel.lean:209` **[MC]**,
probability instances only).

---

## 1. The structural discovery: only `Z` feels the atoms, and nothing needs two sides

What breaks at an atom of ν at x is precisely and only this: `Z_ν` has a
concave corner (`Z′⁺ − Z′⁻ = −(2/τ)·ν({x})`).  Everything else in the
K-route's first-order tower is still smooth or still exists one-sidedly:

1. **[MC]** `D_ν ∈ C¹` with `D′ = (1/τ)L − 2Z` for *arbitrary* probability
   measures — `hasDerivAt_laplaceDisplacementIntegral`,
   `LaplaceWronskian.lean:233`.  (The integrand `(y−x)k(x,y)` is `C¹` in x
   even across `y = x` because the corner of `‖x−y‖` is multiplied by a factor
   vanishing there.)
2. **[MC]** `L_ν ∈ C¹` with `L′ = (1/τ)D` for arbitrary probability measures —
   `hasDerivAt_laplaceCompanionNormalizer`, `LaplaceWronskian.lean:110`.
   (The companion kernel `(τ+‖x−y‖)e^{−‖x−y‖/τ}` is `C¹` across the diagonal.)
3. **[MC]** `Z_ν` is right-differentiable *everywhere* for `[IsFiniteMeasure]`:
   `hasDerivWithinAt_Ici_laplaceKernelNormalizer`,
   `LaplaceGeneralConverseBalance.lean:372`, with explicit coefficient
   `laplaceKernelNormalizerRightDerivCoeff` (:82)
   `= (1/τ)(−e^{−x/τ}·lowerExpMass + e^{x/τ}·upperExpMass)`.
4. **[MC]** The tilted mean `x + D_p/Z_p = x + m` is right-differentiable
   everywhere for arbitrary probability measures with the **manifestly
   nonnegative** coefficient `laplaceTiltedMeanRightDerivCoeff`
   (`LaplaceTiltedMeanMonotone.lean:210, :244`)
   `= (2/τ)(P·Q̂ + P̂·Q)/Z²` where P, Q, P̂, Q̂ are the four one-sided
   masses/compensated moments (`LaplaceGeneralConverse.lean:53–66`), each with
   a nonnegativity lemma (:168, :173).  **No moment hypothesis, no `NoAtoms`.**
   This is the one-sided L3 — it already exists and is strictly more general
   than the global monotone L3.
5. **[MC]** The entire abstract propagation layer
   (`LaplaceACPropagation.lean`) is *already stated with right derivatives*:
   every `abel_*` lemma takes `HasDerivWithinAt W (−c·W) (Ici t) t` and
   `HasDerivWithinAt A (c t) (Ici t) t` on `Ico` sets — including the *left*
   flank lemmas, which use right derivatives with left-approach filters
   (`abel_left_interval_zero_of_upwardCrossing`, :1109).  The outer-ray
   lemmas (:338, :358, :379, :402) are even primitive-free (W² squeeze).
   **No lemma in the layer requires continuity of the coefficient `c`.**
6. **[MC]** `K` is continuous and `K → 0` at ±∞ for arbitrary probability
   measures — `LaplaceAtomlessConverse.lean:407, :549, :564` take only
   probability instances; reusable verbatim.

### 1.1 The one-sided identities [PD — the mathematical core of Frontier A]

Fix arbitrary probability measures p, q with zero drift; write `m′⁺` for the
right derivative of the common ratio m (exists everywhere: quotient of
`D_p ∈ C¹` and right-differentiable `Z_p > 0`).  All identities hold at
**every** x ∈ ℝ.

**(I1) Right L-formula.**  For each ν ∈ {p, q}:
`L_ν = τ(m′⁺ + 2)·Z_ν + τ·m·Z′⁺_ν`.

*Derivation:* `D_ν = m·Z_ν` (common ratio).  Right-differentiate the product:
`D′_ν = m′⁺·Z_ν + m·Z′⁺_ν`.  The left side has the two-sided certified value
`(1/τ)L_ν − 2Z_ν`.  Uniqueness of derivatives within `Ici x`
(`uniqueDiffWithinAt_Ici`) equates them; solve for `L_ν`.  The crucial point:
**the same function m and the same m′⁺ appear for both measures** — this is
what zero drift buys.

**(I2) K in the right-Wronskian coordinate.**  With
`W⁺ := Z′⁺_p·Z_q − Z′⁺_q·Z_p` (an explicit function of the mass functionals):
`K = τ·m·W⁺`.

*Derivation:* substitute (I1) into `K = L_p Z_q − L_q Z_p`; the
`τ(m′⁺+2)`-terms cancel exactly.  **Corollary: `K = 0` pointwise on
`{m = 0}` — the zero set of m costs nothing, no flatness or closure
arguments** (this was already the shape of the atomless proof; it survives
verbatim).

**(I3) Right derivative of K.**
`K′⁺ = −τ(m′⁺ + 2)·W⁺` (as a `HasDerivWithinAt _ (Ici x) x`).

*Derivation:* one-sided product rule on `K = L_p Z_q − L_q Z_p` gives
`K′⁺ = (1/τ)(D_p Z_q − D_q Z_p) + (L_p Z′⁺_q − L_q Z′⁺_p)`; the first bracket
is 0 by zero drift; substitute (I1) into the second — the `τ m Z′⁺ Z′⁺` cross
terms cancel, the `τ(m′⁺+2)` terms collect to `−τ(m′⁺+2)W⁺`.

**(I4) Abel form on `{m ≠ 0}`.**  Combining (I2), (I3):
`K′⁺ = −(2·μ⁺/m)·K` with `μ⁺ := (m′⁺ + 2)/2`, which is *exactly* the
coefficient shape `c t = 2·μDeriv t / m t` consumed by the abstract layer.

**(I5) Global coefficient bound.**  `μ⁺ ≥ 1/2` everywhere: the tilted mean
`x + m` has right derivative `1 + m′⁺ = laplaceTiltedMeanRightDerivCoeff ≥ 0`
(pillar 4), so `m′⁺ ≥ −1`, so `μ⁺ = ((m′⁺+1)+1)/2 ≥ 1/2`.  Global δ = 1/2, no
moment needed — the `Integrable id p` wart of the atomless theorem
**disappears automatically** because this route never touches the global
monotone L3.

**(I6) Local Lipschitz bounds (replaces the edge-derivative trick).**
Global bounds, all elementary: `Z ≤ 1`, `L ≤ τ` (from
`(1+r/τ)e^{−r/τ} ≤ 1`, i.e. `Real.add_one_le_exp`), `|D| ≤ τ/e`
(`sup_{r≥0} r e^{−r/τ} = τ/e`), hence `|D′| = |(1/τ)L − 2Z| ≤ 3`; and
`|Z′⁺| ≤ Z/τ` because `e^{−x/τ}·lowerExpMass x = ∫_{y≤x} k` and
`e^{x/τ}·upperExpMass x = ∫_{y>x} k` split Z.  Therefore on any compact where
`Z_p ≥ ε`: `|m′⁺| ≤ (3·Z + (τ/e)(Z/τ))/Z² ≤ (3 + 1/e)/ε`, and a bounded
right derivative everywhere upgrades to a genuine Lipschitz bound
(`Convex.norm_image_sub_le_of_norm_hasDerivWithin_le` family).  Consequence:
at any edge point a with `m(a) = 0`, the linear bound `|m(t)| ≤ Λ·|t−a|`
needed by the log-singularity lemma is immediate — **no one-sided-Taylor
helper and no left derivatives needed anywhere in the whole proof.**

**(I7) CADLAG regularity of the coefficient [PD].**  The four mass
functionals are right-continuous *everywhere* for arbitrary finite measures:
for `xₙ ↓ x`, `1_{Iic xₙ} → 1_{Iic x}` and `1_{Ioi xₙ} → 1_{Ioi x}`
pointwise *without exception* (the failure set of two-sided continuity is
`{x}` — invisible from the right), so dominated convergence applies with the
same constant dominators used in the atomless file — *easier* than the
atomless two-sided lemmas.  Hence `Z′⁺`, `laplaceTiltedMeanRightDerivCoeff`,
`m′⁺`, `μ⁺`, and the Abel coefficient `c = 2μ⁺/m` (on `{m ≠ 0}`) are all
right-continuous.  Measurability is free: `lowerExpMass` and
`lowerCompensatedMoment` are monotone in x, `upperExpMass` and
`upperCompensatedMoment` antitone (integrand and domain both move one way),
so everything is Borel and locally bounded, hence interval-integrable.

**(I8) The primitive.**  `A(x) := ∫_{base}^{x} c` has
`HasDerivWithinAt A (c x) (Ici x) x` at every point of the working interval,
because c is right-continuous, measurable, locally bounded.  Mathlib tool:
`intervalIntegral.integral_hasDerivWithinAt_right`
(`Mathlib/MeasureTheory/Integral/IntervalIntegral/FundThmCalculus.lean:869`)
**[? exact filter-instance plumbing]**; fallback is a ~30-line manual squeeze
(average of c over `[x, x+h]` → `c(x)` by right-continuity).  This replaces
`intervalPrimitive_hasDerivWithinAt_Ici_of_continuousOn_Ioo`
(`LaplaceACFinal.lean:47`), which needed a continuous integrand.

**(I9) Bonus atom rigidity (‡) [PD, optional corollary].**  Mirroring (I2)
with left derivatives (cheapest via the reflection pushforward
`Measure.map Neg.neg`) gives `K = τ·m·W⁻` as well, so `m·(W⁺ − W⁻) ≡ 0`;
since `Z′⁺_ν − Z′⁻_ν = −(2/τ)ν({x})`, this reads:

> at every x: `m(x) · (Z_p(x)·q({x}) − Z_q(x)·p({x})) = 0`.

Wherever the common mean-shift ratio is nonzero, atom masses of p and q are
locked in the Z-weighted ratio.  Not needed for Frontier A; nice standalone
rigidity statement if we ever want the atomic structure explicitly.

### 1.2 Why this closes the conjecture

The assembly is *identical in shape* to `LaplaceAtomlessConverse.lean`'s
`laplaceCompanionAlignmentDefect_eq_zero_of_zeroDrift`: at any x₀, either
`m x₀ = 0` (then `K x₀ = 0` by (I2)), or trichotomize the zero set of the
continuous m around x₀ using `IsClosed.csSup_mem`/`csInf_mem`, IVT, and the
half-lower-bound extension `exists_Ioo_lower_bound_half_of_continuous_pos`
(`LaplaceACFinal.lean:138`, works for any continuous function — reusable),
then kill K on the component with:

- interior flanks: `abel_right/left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper/lower`
  (`LaplaceACPropagation.lean:1058` and mirror) — inputs are exactly
  (I4) + (I5) global δ = 1/2 + (I6) linear edge bound + (I8) primitive +
  boundedness of K near the edge (continuity);
- outer rays: `abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg` /
  `abel_left_outer_zero_of_muDeriv_nonneg_of_m_pos` (:379, :402) —
  primitive-free, inputs (I4) + sign of m + tails `K → 0` (pillar 6).

Then the gate finishes.  Every input exists for arbitrary probability
measures.  **No analytic obstruction was found anywhere in this pass.**

---

## 2. Frontier A — the unconditional theorem (the conjecture itself)

**Target.**
```
theorem laplaceZeroDrift_identifies
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    p = q
```
plus `IdentifiesAtZero (fun p q => IsProbabilityMeasure p ∧ IsProbabilityMeasure q) (meanShiftDrift (laplaceKernel τ))`,
its (trivial) legitimacy, and subsumption remarks retiring the umbrella.

**New file** `LaplaceUnconditionalConverse.lean` (~1000–1200 lines), importing
the atomless file (for tails/continuity/K-def) and the propagation layer.

### Milestones

- **A1 — CADLAG pack** (~200 lines).  Right-continuity of the four mass
  functionals (DCT along `𝓝[≥]`-sequences, constant local dominators, no
  exceptional set); monotonicity/antitonicity ⟹ measurability; right-
  continuity + measurability + local bounds for
  `laplaceKernelNormalizerRightDerivCoeff` and
  `laplaceTiltedMeanRightDerivCoeff`.  Also
  `laplaceTiltedMeanRightDerivCoeff_nonneg` (positivity from the four
  nonnegativity lemmas — check whether only the strict two-sided-mass version
  `:344` exists today [?]).
- **A2 — one-sided identity pack** (~300 lines).  `m′⁺` exists everywhere
  (`HasDerivWithinAt.div`); `m′⁺ = laplaceTiltedMeanRightDerivCoeff − 1 ≥ −1`
  (uniqueness within `Ici`); the right L-formula (I1) via
  `uniqueDiffWithinAt_Ici`; `K = τ·m·W⁺` (I2); `K′⁺ = −τ(m′⁺+2)W⁺` (I3)
  under zero drift (`laplaceMeanShiftRatio_common_of_zeroDrift` supplies the
  common ratio); Abel form on `{m ≠ 0}` (I4); global bounds and local
  Lipschitz for m (I6).
- **A3 — right-FTC helper** (~80 lines).
  `intervalPrimitive_hasDerivWithinAt_Ici_of_rightContinuous`: primitive of a
  measurable, locally bounded, right-continuous integrand (I8).  Try the
  Mathlib one-sided FTC-1 first; manual squeeze as fallback.
- **A4 — edge/ray wiring** (~250 lines).  One-sided analogues of
  `laplaceAlignmentDefect_eq_zero_on_Ioo_of_leftEdge/_rightEdge` and the two
  ray lemmas from the atomless file, with two simplifications: linear edge
  bounds come from A2's Lipschitz bound (no `exists_Ioo_linear_bound_...`),
  and δ = 1/2 is global.  The abstract layer is consumed as-is.
- **A5 — assembly + headline + audit** (~250 lines).  Mirror the atomless
  trichotomy; headline theorem; `IdentifiesAtZero` + legitimacy (Gaussian
  pair, τ-free); corollaries: `laplaceZeroDrift_identifies_of_noAtoms` and the
  M3/umbrella statements re-derived in one line each (do *not* delete the old
  proofs — they are independent certificates); AxiomAudit list; docs
  (Roadmap, ResearchStatus, ArbitraryConverse, Derivation log); Check.ps1.

### Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| Mathlib one-sided FTC-1 filter plumbing (`FTCFilter` instances for `𝓝[Ici x] x`) | low | manual 30-line squeeze using right-continuity |
| `m′⁺` bookkeeping: same m for both measures under zero drift | low | exactly the common-ratio dance the atomless file already performs |
| One-sided product/quotient rule names (`HasDerivWithinAt.mul/div`) | low | all exist; uniqueness via `uniqueDiffWithinAt_Ici` |
| Sign-convention drift between `W⁺` here and `laplaceWronskian`-style defs in older files | low | define `W⁺` fresh; only (I2)/(I3) consistency matters, checked by `ring` |
| Hidden two-sidedness in the atomless edge lemmas being mirrored | medium | the abstract layer is verified one-sided; A4 re-derives the *wiring* only, and each wiring input was re-checked this pass |

Honest overall assessment: **medium-low risk, 1–2 focused sessions.**  This
is the same effort class as the atomless milestone, with less new analysis
(the atomless file had to *invent* the K-route; here it only has to shed
two-sidedness, and the repo was accidentally-on-purpose built for it).

### What A makes obsolete

`LaplaceRealConverseCondition` and both branches become corollaries; the
"atoms on interval supports" row and the "`p` first moment" wart disappear
simultaneously.  The 1-d Laplace arbitrary-target converse is then **fully
closed, unconditionally**, and the interesting remaining questions move to
other kernels and other dimensions (Frontiers C/D).

---

## 3. Frontier B — retro-cleanup of the global monotone L3 (optional)

Frontier A does not use `laplaceTiltedMean_monotone`, so the moment wart dies
with A regardless.  Independently, the monotone theorem's `hint : Integrable
(fun y => y) p` is used **exactly once** — `LaplaceTiltedMeanMonotone.lean:379`,
`hAint := hint.bdd_mul ...` — to integrate `y·k(x,y)`.  But `|y·k(x,y)| ≤
|x| + τ/e` (split `|y| ≤ |x| + |y−x|` and use `sup r·e^{−r/τ} = τ/e`), so
`hAint` follows from boundedness alone.  Dropping the hypothesis is a ~20-line
edit plus ripples (callers at `LaplaceACRegularity.lean:218`,
`LaplaceAtomlessConverse.lean:279`; the `LaplaceAtomlessCondition` moment
field and its umbrella/legitimacy/audit entries).  Worth doing only as a
warm-up or if A is deferred; if A lands first, fold the cleanup into A5's
corollary section or skip.

---

## 4. Frontier C — the Gaussian-kernel arbitrary-target converse (cheap, high value)

> **★★★ CORRECTION + DONE (2026-07-13): this frontier was ALREADY CLOSED, and
> the recon below was wrong.**  The repo *does* have the general raw
> Gaussian-kernel converse: `gaussianMeanShiftDrift_identifiesAtZero`
> (`GaussianScoreRecovery.lean:295`) — for **arbitrary** probability measures on
> any finite-dimensional real inner-product space, pointwise zero Gaussian
> mean-shift drift forces `p = q`, axiom-free (already in the promoted list).
> It is even *stronger* than the 1-d target planned here (arbitrary dimension),
> and it is proved exactly as anticipated: the derivative identity
> `∇log Zₚ = σ⁻²·meanShiftₚ` is `hasFDerivAt_log_gaussianKernelNormalizer`, zero
> drift makes the two log-normalizers differ by a constant
> (`is_const_of_fderiv_eq_zero`), `gaussianKernelNormalizer_proportional_constant_eq_one`
> forces the constant to `1`, and `gaussianKernelNormalizer_injective` finishes.
> All that was missing was surface parity with the Laplace headline:
> `GaussianArbitraryConverse.lean` adds `gaussianZeroDrift_identifies` (direct
> `p = q` form) and `bothProbability_isLegitimate` (`IsLegitimateCondition` for
> the unconditional condition).  `Check.ps1` green; both new decls axiom-free.
> The structural moral below still stands.  Net: **both canonical kernels, all
> finite dimensions for Gaussian (1-d for Laplace), arbitrary targets,
> unconditional.**

**Status check this pass (as originally written — SUPERSEDED, see correction
above):** the repo has *no* general Gaussian-kernel
zero-drift converse — only the empirical/finite-support version
(`gaussianEmpiricalPoint_identifies`, `EmpiricalFrameBound.lean:722`) and the
conditional MMD track.  But the hard endgame **already exists, axiom-free**:

- `gaussianKernelNormalizer_injective` (`GaussianConvolutionInjectivity.lean:281`)
  **[MC]**: `Z_p ≡ Z_q ⟹ p = q` for arbitrary probability measures (via
  Mathlib's `Measure.ext_of_charFun`, `charFun_conv`, `charFun_gaussianReal` —
  all present in the pinned Mathlib).
- `gaussianKernelNormalizer_proportional_constant_eq_one` (:309) **[MC]**:
  `Z_p ≡ c·Z_q ⟹ c = 1`.

**The missing piece is one derivative identity.**  For the Gaussian kernel
`g_σ(x,y) = exp(−‖x−y‖²/(2σ²))` [? confirm exact normalization in
Paperaxioms at implementation time], `∂ₓ g = ((y−x)/σ²)·g`, so

> `Z′_ν(x) = (1/σ²)·D_ν(x)` — the displacement integral *is* the score of the
> smoothed measure, up to σ².

This is a `HasDerivAt` under the integral with a *constant global dominator*
(`sup_r r·e^{−r²/2σ²} < ∞`) — strictly easier than the Laplace analogues the
repo already certifies.  Then, for probability p, q with zero drift:
`D_p/Z_p ≡ D_q/Z_q ⟹ Z′_p/Z_p ≡ Z′_q/Z_q ⟹ (Z_p/Z_q)′ ≡ 0` (quotient rule,
`Z_q > 0`) `⟹ Z_p = c·Z_q` (derivative-zero-on-ℝ ⟹ constant) `⟹ c = 1` (:309)
`⟹ p = q` (:281).

**Target.**
```
theorem gaussianZeroDrift_identifies
    (σ : ℝ) (hσ : ValidBandwidth σ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (gaussianKernel σ)) p q) :
    p = q
```
**unconditional** — no atomless, no moments, nothing.  Plus
`IdentifiesAtZero` wrapper and legitimacy.  Estimate: **~250–350 lines, one
short session, low risk** (new file `GaussianArbitraryConverse.lean`:
derivative lemma ≈ 100 lines with integrability boilerplate, ratio-constant
argument ≈ 80, glue ≈ 80).

**Why this matters beyond the checkmark:** it explains *why* Laplace was the
hard case.  For the Gaussian kernel the drift is the gradient of the smoothed
log-density, so zero drift trivially pins the smooth; for the Laplace kernel
`Z′` is *not* proportional to `D` (it sees the one-sided masses separately),
and the companion tower + Abel ODE is genuinely needed.  The pair of theorems
makes a clean structural statement: *mean-shift identifiability is elementary
exactly when the kernel is its own displacement integrator.*

Multidimensional remark: the same Gaussian argument works in ℝⁿ
(`∇Z = D/σ²` coordinatewise, `ext_of_charFun` is already dimension-generic in
the repo's injectivity file), at the cost of Fréchet-derivative-under-integral
boilerplate.  Park as C+ until 1-d lands.

---

## 5. Frontier D — horizons after A and C

> **2026-07-14 update: the dedicated research pass exists — see
> `LaplaceHigherDim.md`.**  Outcome: in n-d the kernel bifurcates by norm.
> **The referenced paper's Eq. (12) uses the ℓ² norm** ("∥·∥ is ℓ2-distance",
> verified in `papers/2602.04770v2.pdf`), so the paper-faithful target is the
> radial case.  The ℓ¹/product case (`laplaceKernel` on `PiLp 1`; sklearn's
> `laplacian_kernel` — *not* the paper's kernel) **reduces completely to the
> 1-d unconditional theorem** by coordinate slicing with exponentially tilted
> measures — full proof on paper, implementation-ready, kept as an adjacent
> result.  The ℓ²/radial (Matérn-1/2) case — the primary objective — gets a
> structural framework (displacement potential `D = ∇ψ` with `ψ` the Matérn-3/2
> smoothing; companion PDE `(1−τ²Δ)D = (n+1)τ²∇Z`; vector Abel system) plus a
> staged partial-results program (far-field foundation → radial measures →
> finite support → endgame); its general endgame remains open research.

- **ℝⁿ Laplace kernel.**  No ODE structure in n > 1; the one-sided mass
  decomposition is a genuinely 1-d trick.  Plausible entry points: radial
  slicing through pairs of points, or the divergence-form identity
  `Δₓ e^{−‖x−y‖/τ} = (1/τ²)k − ((n−1)/(τ‖x−y‖))k` (Matérn-½ is the resolvent
  kernel of `(I − τ²Δ)` in odd dimensions ⟹ possible PDE route:
  zero drift ⟹ `(I − τ²Δ)`-resolvent smooths are proportional?).  *Research
  pass done — superseded by `LaplaceHigherDim.md`, which confirms the PDE-route
  intuition (the resolvent identity is exact, holds in every dimension at the
  potential level, and is Matérn-universal) and adds the ℓ¹ tensorization
  reduction.*
- **Poly-exponential (Matérn-type) kernels in 1-d.**  `f(r) = P(r)e^{−r/τ}`
  satisfies a constant-coefficient linear ODE; the D/L/Z tower generalizes to
  a `(deg P + 2)`-dimensional companion system.  Conjecture: the K-trick is
  an instance of an abstract statement about kernels whose translate module is
  ODE-finite.  A clean abstract theorem here would be a paper-worthy
  generalization.  Park.
- **Sharpness notes to record in the paper docs** (both trivial but worth
  stating): (i) the probability normalization is necessary — `m_{c·p} = m_p`
  for every `c > 0`, so the finite-measure version of the converse is false
  (p vs 2p); (ii) all identifying conditions found so far are bandwidth-free,
  and a single bandwidth suffices — no τ-family is ever needed.

---

## 6. Routes considered and rejected this pass (logged so we don't retry)

- **a.e./Rademacher route** (m locally Lipschitz ⟹ a.e. two-sided ODE +
  absolutely-continuous Gronwall): mathematically sound (atoms are countable
  hence Lebesgue-null, K is locally Lipschitz so it has no singular part),
  but it needs an AC/Lipschitz FTC that Mathlib support is uncertain for, and
  it is *strictly dominated* by the one-sided route because the abstract
  layer already speaks right derivatives.  Keep only as fallback if some
  right-continuity claim in (I7) fails unexpectedly (it won't — the DCT
  argument is the atomless file's own, minus the exceptional set).
- **Mollification** (convolve p, q with a small Gaussian and use the atomless
  theorem): dead end — the mean-shift drift is a *ratio* of ν-integrals, so
  zero drift for (p, q) does not transfer to (p∗η, q∗η) in any usable way.
- **Explicit atomic decomposition** (split p, q into atoms + continuous parts
  and match pieces): the (‡) rigidity from (I9) pins atoms only where m ≠ 0,
  and the cross terms in Z make the casework explode.  The K-coordinate does
  all of this implicitly; don't fight it by hand.

---

## 7. Recommended attack order and success criteria

1. **Frontier A** (the conjecture): milestones A1 → A5.  This is the main
   event; everything else is dessert.
2. **Frontier C** (Gaussian): one short session, immediately after — the
   program then reads "both canonical kernels, arbitrary 1-d targets,
   unconditional."
3. Frontier B only if it hasn't already died as a corollary of A.
4. Re-run the full discipline at each landing: `lake build
   DriftingIdentifiability`, `scripts/Check.ps1` (trust audit + axiom audit +
   promoted list), `#print axioms` on every new headline = `propext,
   Classical.choice, Quot.sound` only; no new axioms anywhere (neither A nor
   C needs any — C's charFun machinery is Mathlib-native).

Success criterion for the conjecture: the statement in §2 machine-checked and
promoted, with the umbrella condition demoted to a historical corollary.
