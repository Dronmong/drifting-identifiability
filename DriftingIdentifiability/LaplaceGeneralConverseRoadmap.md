# Roadmap: pushing the atomic Laplace converse to general measures

**Audience:** a fresh agent (or human) resuming this track cold.  Read this
file top to bottom before touching Lean.  Written 2026-07-10, immediately
after the atomic converse landed (`da71deb`).

**Goal (the open core):** prove, machine-checked and axiom-free,

```text
For τ > 0 and ARBITRARY probability measures p, q on ℝ:
ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q  →  p = q.
```

This is the last open cell of the paper's converse matrix in one dimension
("the general converse for arbitrary fields remains open" — the authors'
own concession).  The finite (atomic) case is DONE
(`laplaceZeroDrift_atomic_identifies`, `LaplaceAtomicConverse.lean`).  This
roadmap reduces the general case to one open analytic question and lays out
implementable milestones, two of which are new theorems achievable with
current technology.

---

## 0. Orientation: repo discipline (non-negotiable)

1. **Never** introduce `axiom`/`constant`/`opaque`/`sorry` without explicit
   user approval — the TrustAudit fails the build otherwise.  Only
   `TrustedBoundary.lean` imports `Paperaxioms.lean`.
2. **Record before implementing.**  Update this file (or
   `LaplaceArbitraryConverse.md`) with the plan/derivation BEFORE writing
   Lean.  The user has asked for this repeatedly (token-safety).
3. **Numerics first.**  Every identity gets verified numerically (~1e-14)
   before formalization.  No Python/Node on this machine — use PowerShell
   with `[math]::Exp` doubles.  **PowerShell trap:** variable names are
   case-insensitive; a loop counter `$k` silently clobbers a constant `$K`
   and the loop never runs.  Use distinct names (`$ncell`, `$j`).
4. After meaningful Lean changes run `scripts/Check.ps1` (build + trust +
   axiom audit).  New headline theorems go into the promoted list in
   `scripts/AxiomAudit.ps1` and must `#print axioms` to
   `propext, Classical.choice, Quot.sound` only.
5. New modules get imported from the root `DriftingIdentifiability.lean`.
6. Failed routes go to `LoggedFailures.md`.  Commits end with
   `Co-Authored-By:` line for the current model.
7. Lean/Mathlib pitfalls already paid for (don't rediscover them):
   - `HasDerivAt.mul` produces Pi-instance function products; type-ascribe
     the `have` with an explicit lambda and value.
   - `Filter.EventuallyEq` congr lemmas want `f₁ =ᶠ f` (not `.symm`);
     `convert` on `HasDerivAt` hits instance diamonds — prove a value
     equality `hval` and finish `exact hval ▸ hf`.
   - `linear_combination`/`ring` fail across `Finset.sum` binders; use
     rewrite chains (`rw [hgoal, mul_comm …, hcross, sub_self]` style).
   - `unfold kernelNormalizer at h` before integral rewrites.
   - Renames: `integral_finsetSum_measure`, `Measure.coe_finsetSum`,
     `Finset.notMem_empty`; `Set.infinite_coe_iff.mp` bridges subtype
     `Infinite`; `Set.Infinite.image` takes `InjOn` first.
   - Anything using `Real.exp` in a `def` needs `noncomputable`.

## 1. Where the mathematics stands

### Certified (Lean, axiom-free)

| Result | Declaration | File |
|---|---|---|
| Cross-displacement form of zero drift: `ZeroDrift ⟺ ∀x, Z_q·D_p = Z_p·D_q` | `laplaceZeroDrift_iff_crossDisplacement` | `LaplaceCompanion.lean` |
| No-moment regularity: everything integrable for arbitrary probability measures | `laplace_meanShiftRegularAt` etc. | `LaplaceCompanion.lean` |
| Tilt bridge: zero drift ⟹ exponential-tilt centroids agree (critical moment) | `laplaceZeroDrift_tiltCentroid_eq` | `LaplaceCompanion.lean` |
| 1-d Laplace smoothing injectivity (finite measures) | `laplaceKernelNormalizer_injective` | `LaplaceInjectivity.lean` |
| Dirac rigidity | `laplaceZeroDrift_dirac_identifies_real` | `LaplaceInjectivity.lean` |
| Classical ODE `τD_p′ = L_p − 2τZ_p`; alignment reduction (`K ≡ 0` ⟹ `p = q`) | `hasDerivAt_laplaceDisplacementIntegral`, `laplaceZeroDrift_imp_eq_of_companionAligned` | `LaplaceWronskian.lean` |
| **Atomic converse** (finite support, arbitrary atoms/weights) | `laplaceZeroDrift_atomic_identifies` | `LaplaceAtomicConverse.lean` |

Notation: `kτ(x,y) = e^{−|x−y|/τ}`, `Z_p = ∫kτ dp`, `D_p = ∫kτ·(y−x) dp`,
`L_p = ∫(τ+|x−y|)kτ dp`.

### The decomposition (the working coordinates)

Zero drift ⟺ `Φ(x) := Z_q(x)D_p(x) − Z_p(x)D_q(x) = 0` for all `x`, and

```text
Φ(x) = ∬ (y−y′) kτ(x,y) kτ(x,y′) dp(y) dq(y′).
```

Splitting `ℝ² = (Iic x ∪ Ioi x)²` and using `kτ(x,y) = e^{(y−x)/τ}` for
`y ≤ x`, `= e^{(x−y)/τ}` for `y > x`:

```text
Φ(x) = e^{−2x/τ}·𝔞(x) + 𝔟(x) + e^{2x/τ}·𝔠(x),

𝔞(x) = ∬_{y≤x, y′≤x} (y−y′) e^{(y+y′)/τ}  dp(y) dq(y′)   (truncated pairing)
𝔠(x) = ∬_{y>x, y′>x} (y−y′) e^{−(y+y′)/τ} dp(y) dq(y′)
𝔟(x) = ∬_{y≤x<y′} (y−y′)e^{(y−y′)/τ} + ∬_{y′≤x<y} (y−y′)e^{(y′−y)/τ}.
```

One-sided transforms (all finite for EVERY probability measure, no moment
hypotheses — integrands are bounded on the half-lines):

```text
P⁻(x) = ∫_{y≤x} e^{y/τ} dp        P(x) = ∫_{y≤x} (x−y) e^{y/τ} dp
P⁺(x) = ∫_{y>x} e^{−y/τ} dp       P̂(x) = ∫_{y>x} (y−x) e^{−y/τ} dp
```

and `Q⁻, Q, Q⁺, Q̂` likewise for `q`.  Bound for integrability of `P`:
`(x−y)e^{y/τ} = e^{x/τ}·(x−y)e^{−(x−y)/τ} ≤ τ·e^{x/τ}` — reuse
`mul_exp_neg_le'` from `LaplaceWronskian.lean`.

### NEW (2026-07-10, this roadmap; numerically verified): the bracket identities

**(I1)** `𝔞(x) = Q(x)·P⁻(x) − P(x)·Q⁻(x)` for every `x` — pure algebra:
`(y−y′) = (x−y′) − (x−y)` inside the double integral, then Fubini splits
into products of single integrals.  Verified numerically to `1.1e-14`
(random atomic pairs, seed 11, 60 probes, NOT zero drift — the identity is
unconditional).  Mirror: `𝔠(x) = P̂(x)Q⁺(x) − Q̂(x)P⁺(x)`.

**(I2)** `P(x) = ∫_{−∞}^{x} P⁻(t) dt` — Fubini
(`∫_{t≤x}∫_{y≤t} = ∫_{y≤x}(x−y)…`).  Verified numerically (midpoint rule,
residual = O(h) from atom cells, as predicted).

**Why (I1) changes the game:** it turns the sharpened open question
("does zero drift force `𝔞 ≡ 0`?", recorded in
`LaplaceArbitraryConverse.md` Stage 3c) into a two-part architecture that
mirrors the atomic proof exactly:

```text
[open core]      ZeroDrift  ⟹  𝔞 ≡ 0                     (Milestone 5)
[endgame, done*] 𝔞 ≡ 0      ⟹  p = q                      (Milestone 2)
```

(*done on paper, fully designed below, numerically sanity-checked;
formalization is mechanical.)  In the atomic case the two parts are
`frakA_eq_zero` and `weights_eq_of_frakA` respectively — the general
endgame is the measure-theoretic generalization of `weights_eq_of_frakA`.

Also note: `𝔞(+∞) = (∫ye^{y/τ}dp)(∫e^{y/τ}dq) − (∫e^{y/τ}dp)(∫ye^{y/τ}dq)`,
so the already-certified tilt bridge `laplaceZeroDrift_tiltCentroid_eq` is
precisely the statement `𝔞(+∞) = 0` (under the critical exponential
moment).  `𝔞 ≡ 0` is its finite-`x` sharpening.

---

## 2. Milestones, in implementation order

Suggested new module: `DriftingIdentifiability/LaplaceGeneralConverse.lean`
(or split into `…Endgame.lean` + `…NowhereDense.lean` if it grows past
~700 lines).

### Milestone 1 — one-sided transforms + interval decomposition (LOW RISK)

Definitions (`noncomputable`, section variables `(τ : ℝ) (p : Measure ℝ)`):
`lowerExpMass` (= `P⁻`), `lowerCompensatedMoment` (= `P`), `upperExpMass`,
`upperCompensatedMoment`, as set integrals over `Set.Iic x` / `Set.Ioi x`.

Lemmas:
1. Integrability on the half-lines (bounded integrands + finite measure;
   `Measure.integrableOn_of_bounded` or `Integrable.mono'` with constant).
2. `𝔞`-bracket definition: define
   `truncatedPairing τ p q x := Q x * P⁻ x − P x * Q⁻ x`
   and prove it equals the double integral (`integral_prod_mul` on
   `p.prod q` restricted to `Iic x ×ˢ Iic x`; the atomic file's
   `kernel_left/kernel_right` show the case-split style — they are
   `private`, so re-prove or de-private them).
3. **Decomposition theorem** (unconditional, no zero drift):
   `∀ x, Z_q x * D_p x − Z_p x * D_q x
        = exp (−2x/τ) * 𝔞 x + 𝔟 x + exp (2x/τ) * 𝔠 x`.
   Proof: write `Z_q·D_p` and `Z_p·D_q` as double integrals
   (`integral_prod_mul`), split the product measure over
   `(Iic x ∪ Ioi x) ×ˢ (Iic x ∪ Ioi x)` (`MeasureTheory.integral_union`,
   disjointness `Set.Iic_disjoint_Ioi`), rewrite kernels per piece.
4. Corollary under zero drift (via
   `laplaceZeroDrift_iff_crossDisplacement`, plus `smul_eq_mul` since
   `E = ℝ`): `e^{−2x/τ}𝔞(x) + 𝔟(x) + e^{2x/τ}𝔠(x) = 0` for all `x`.
5. Regularity of `𝔞`: right-continuity in `x` (dominated convergence as
   `Iic t ↓ Iic x`; `MeasureTheory.tendsto_integral_of_dominated_convergence`
   along sequences + `tendsto_of_seq_tendsto`, or monotone-class), and the
   trivial limit `𝔞(x) → 0` as `x → −∞`.

### Milestone 2 — the endgame: `𝔞 ≡ 0 ⟹ p = q` (✅ DONE 2026-07-10, machine-checked, axiom-free)

**Status: COMPLETE.**  Implemented in
`DriftingIdentifiability/LaplaceGeneralConverseEndgame.lean`, imported by the
root, promoted in `scripts/AxiomAudit.ps1`; `Check.ps1` green (42 files, 198
promoted decls); `#print axioms` on the headline reports only
`propext, Classical.choice, Quot.sound`.  Headline:
`laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero`.  Key supporting
theorems now certified: `hasDerivWithinAt_lowerCompensatedMoment` (E1),
`lowerCompensatedMoment_proportional` (E2–E4), `restrict_eq_of_lowerExpMass_prop`
(E6).  **The general-measure converse is now reduced to a single open
statement — Milestone 5's `ZeroDrift ⟹ 𝔞 ≡ 0`** (Milestone 3 discharges it
for nowhere-dense supports).

Notes on how the formalization went vs. the plan below:
* **E1 was proved by a clean squeeze, not distribution theory.**  For `x' > x₀`,
  `P(x')-P(x₀) = (x'-x₀)P⁻(x₀) + R` with `0 ≤ R ≤ (x'-x₀)(P⁻(x')-P⁻(x₀))`, so
  the difference quotient is squeezed between `P⁻(x₀)` and `P⁻(x')`.  The upper
  bound `→ P⁻(x₀)` is exactly the Milestone-1 right-continuity
  `lowerExpMass_continuousWithinAt_Ici` — **no measure-continuity lemma or
  convexity was needed.**  (`hasDerivWithinAt_iff_tendsto_slope` +
  `tendsto_of_tendsto_of_tendsto_of_le_of_le'`.)
* **E4 (support match) was simpler than sketched.**  With `A := {P>0}` an
  up-set, `s := sInf A` satisfies `P(s)=0` automatically (if `p(Iio s)>0` then
  `p(Iio (s-1/(m+1)))>0` for some `m` by `measure_iUnion_null`, contradicting
  `s = inf`); hence `A = Ioi s`, `Q = L·P` extends by continuity
  (`Set.EqOn.closure`, `closure_Ioi`) to `Q(s)=L·P(s)=0`, so `q(Iio s)=0`, and
  `q(Iio x)=0` for `x ≤ s`.  No atom-at-`s` case analysis or `N_q`/`N_p`
  moment computation was necessary.
* **E5** uses `derivWithin` uniqueness (`HasDerivWithinAt.derivWithin` +
  `uniqueDiffWithinAt_Ici`), not `HasDerivWithinAt.unique`.
* **E6** truncates to `Iic n`, weights by `withDensity (ofReal e^{y/τ})` (finite
  on `Iic n`), applies `Measure.ext_of_Iic`, undoes the density with the
  reciprocal `e^{-y/τ}` (`withDensity_mul` + `w·w' = 1`), then `n → ∞`
  (`tendsto_measure_iUnion_atTop`).  `λ = 1` falls out of total mass.

Original design (kept for reference):

```lean
theorem laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (h : ∀ x, truncatedPairing τ p q x = 0) : p = q
```

(Note: zero drift is NOT needed as a hypothesis — `𝔞 ≡ 0` alone suffices.
Numerically sanity-checked: `q = λp` gives `𝔞 ≡ 0` to 8.7e-19; a 1%
perturbation of one weight breaks it at 1e-3.)

Proof plan, step by step (each step has a named Mathlib tool):

- **E1. Right-derivative of `P`.**  For every `x`:
  `HasDerivWithinAt P (P⁻ x) (Set.Ici x) x`.  Direct squeeze, no FTC
  machinery: for `h > 0`,
  `P(x+h) − P(x) = h·P⁻(x) + ∫_{Ioc x (x+h)} (x+h−y)e^{y/τ} dp`,
  and the remainder is `≤ h·e^{(x+h)/τ}·p(Ioc x (x+h)) = o(h)` since
  `p(Ioc x (x+h)) → 0` (measure continuity from above,
  `MeasureTheory.tendsto_measure_iInter_atTop` or the `Ioc`-shrinking
  special case).  `P⁻` is right-continuous for the same reason, so the
  derivative value is exactly `P⁻ x` (no one-sided-limit correction).
  Also: `P` continuous (locally Lipschitz), `P` monotone nondecreasing.
- **E2. Positivity region.**  Define `s₀(p) := sInf` of the support in the
  CDF sense.  Show `P x > 0 ↔ p (Iio x) > 0` (split off mass at distance
  `δ` below `x`: `P x ≥ δ·e^{c/τ}·p(Iic (x−δ))`, continuity from below
  `Iio x = ⋃ₙ Iic (x − 1/n)`).
- **E3. Ratio constancy.**  On any `[a, b] ⊆ {P > 0}` the function
  `r := fun x => Q x / P x` is continuous with
  `HasDerivWithinAt r ((Q⁻x·P x − Q x·P⁻x)/(P x)^2) (Ici x) x`
  (`HasDerivWithinAt.div`), and the numerator is `−𝔞(x) = 0` by (I1) and
  the hypothesis.  Apply **`constant_of_has_deriv_right_zero`**
  (`Mathlib.Analysis.Calculus.MeanValue` — the same lemma that powers the
  Grönwall file), then exhaust `{P > 0} = (s₀(p), ∞)` by
  `a ↓ s₀, b ↑ ∞` to get `Q = λ·P` there, `λ := r(x₁)` for any anchor.
- **E4. Supports match, extension to all ℝ.**  `P ≡ 0` on `Iic s₀(p)`
  (an atom exactly at `s₀` contributes `(x−s₀) = 0` at `x = s₀`), and
  `Q(s₀⁺) = λ·P(s₀⁺) = 0` forces `q(Iio s₀) = 0` by E2.  Hence
  `Q = λ·P` on ALL of ℝ.  Rule out `λ = 0` (`Q ≡ 0` on `(s₀,∞)` would
  force `q = 0`, contradicting probability); if `{P > 0}` were empty,
  `p = δ` at a single point... it is never empty (E2 + probability).
- **E5. Differentiate the proportionality.**  `Q = λP` as functions plus
  E1 for both and uniqueness of derivatives within `Ici x`
  (`HasDerivWithinAt.unique` via `uniqueDiffWithinAt_Ici`, or subtract and
  use the squeeze directly) gives `Q⁻ = λ·P⁻` everywhere.
- **E6. Recover the measures.**  `∫_{Iic x} e^{y/τ} dq = λ·∫_{Iic x} e^{y/τ} dp`
  for all `x`.  Fix `b`; the measures
  `(q.restrict (Iic b)).withDensity (fun y => ENNReal.ofReal (exp (y/τ)))`
  and `λ`-scaled `p`-analogue are FINITE (density bounded by `e^{b/τ}`)
  and agree on every `Iic x` — apply the CDF-uniqueness extensionality
  (`MeasureTheory.Measure.ext_of_Iic` for finite measures; if the exact
  name differs, the charFun route in `LaplaceInjectivity.lean` shows the
  fallback pattern, but `ext_of_Iic` should exist).  Undo the density
  (`withDensity` of the reciprocal; `Measure.withDensity_withDensity` /
  `withDensity_mul`, density strictly positive) to get
  `q.restrict (Iic b) = (ENNReal.ofReal λ) • p.restrict (Iic b)`, then let
  `b → ∞` (`Measure.restrict_iUnion` or ext on `Iic` again).
- **E7. Normalize.**  Total mass `1 = λ·1` ⟹ `λ = 1` ⟹ `p = q`.

Risk notes: E6 is pure bookkeeping but fiddly (ENNReal/withDensity); if
`ext_of_Iic` proves elusive, an alternative is to show
`q (Ioc a b) = ofReal λ * p (Ioc a b)` directly from differences of E5 at
endpoints via monotone-class on the `Ioc` π-system
(`MeasureTheory.ext_of_generate_finite` with
`Real.borel_eq_generateFrom_Ioc`-style generators).

### Milestone 3 — free upgrade: the nowhere-dense-support converse (✅ DONE 2026-07-10, machine-checked, axiom-free)

The atomic proof's engine generalizes verbatim to any **gap** of the
combined support.  Let `S := closure (support p ∪ support q)` (measure
supports).  For any open interval `(u,v)` with `(p+q)((u,v)) = 0`, the
coefficients `𝔞, 𝔟, 𝔠` are CONSTANT on `(u,v)` (the truncation sets do not
change).  Multiplying the Milestone-1 decomposition by `e^{2x/τ}` and
substituting `w = e^{2x/τ}` (injective, infinitely many values in the
interval), a quadratic in `w` vanishes on an infinite set — reuse the
`quadratic_vanish` pattern from `LaplaceAtomicConverse.lean` (it is
`private`; de-private or copy).  Hence **`𝔞 = 0 on every gap of S`**.

If `S` is nowhere dense, gap points accumulate at every `x` from the
right, and `𝔞` is right-continuous (Milestone 1.5), so `𝔞 ≡ 0` on all of
ℝ, and Milestone 2 finishes:

```lean
theorem laplaceZeroDrift_identifies_of_nowhereDense_support … : p = q
```

This strictly extends the atomic theorem to: countably many atoms
(including accumulation points, e.g. support `{1/n} ∪ {0}` or all of a
discrete lattice), Cantor-type singular supports, etc. — a publishable
strengthening on its own, achievable with no new analytic ideas.
Formalization notes: state the gap hypothesis concretely (e.g.
`∀ x, ∀ ε > 0, ∃ u v, x < u ∧ u < v ∧ v < x + ε ∧ (p+q) (Ioo u v) = 0` —
"right-dense gaps" — rather than topological nowhere-density, to keep the
Lean statement assumption-checkable; then derive the topological corollary).

Implementation continuation (2026-07-10, Codex, after Milestone 2 was
cleared): implement Milestone 3 in a separate
`LaplaceGeneralConverseNowhereDense.lean` module with two layers.  First,
prove a reusable topology bridge: if a right-continuous function has zeros
arbitrarily close from the right of every point, then it is identically zero.
Applied to `truncatedPairing`, this immediately composes with the Milestone-2
endgame.  Second, prove that zero drift plus a concrete zero-mass gap
`(p+q)(Ioo u v)=0` supplies those right-dense zeros by freezing the three
one-sided coefficients on the gap and using the quadratic-vanishing pattern
from the atomic proof.  The final statement should use the explicit
right-dense zero-mass gap hypothesis; a topological nowhere-dense-support
corollary can be added later once the support-to-gap bridge is separately
formalized.

Status update (2026-07-10, Codex): implemented as planned in
`LaplaceGeneralConverseNowhereDense.lean`.  The module proves the topological
bridge `laplaceZeroDrift_identifies_of_rightDense_truncatedPairing_zeros`, the
zero-mass gap extraction `truncatedPairing_eq_zero_on_gap`, the right-dense
gap-to-zero theorem
`rightDenseZeros_truncatedPairing_of_zeroDrift_rightDenseZeroMassGaps`, and the
headline `laplaceZeroDrift_identifies_of_rightDense_zeroMassGaps` (plus the
roadmap-name alias `laplaceZeroDrift_identifies_of_nowhereDense_support` whose
formal hypothesis is still the explicit right-dense zero-mass gap condition).
No support/topological corollary has been added yet; that is deliberately left
as a separate support-bridge packaging task, not a theorem gap in the concrete
right-dense-gap result.

### Milestone 4 — the balance identity `e^{−2x/τ}𝔞 = e^{2x/τ}𝔠` (RECORDED, NEEDS RE-VERIFICATION)

Stage 3c (see `LaplaceArbitraryConverse.md`) derived, by Stieltjes
differentiation of the decomposition with cancellation of all `dp`/`dq`
boundary terms:

```text
under zero drift:   e^{−2x/τ}·𝔞(x) = e^{2x/τ}·𝔠(x)   for all x,
and   m·(Z_p dq − Z_q dp) = −(2/τ)·𝔟 dx   as measures on {m ≠ 0}.
```

Consequently `𝔟 = −2e^{−2x/τ}𝔞` under zero drift, so ALL THREE
coefficients are controlled by `𝔞` alone: `𝔞 ≡ 0 ⟺ 𝔟 ≡ 0 ⟺ 𝔠 ≡ 0`.

**Before formalizing:** the underlying UNCONDITIONAL identity (the
decomposition of `dΦ` for arbitrary pairs, zero drift not assumed) IS
numerically testable — test it on random non-drift-free atomic pairs
(the conditional identities themselves degenerate to `0 = 0` only if the
conjecture is true, so test the general form).  Derive first on paper:
`dΦ = (2/τ)(e^{2x/τ}𝔠 − e^{−2x/τ}𝔞) dx + [boundary measure that cancels]`.

**Formalization route (avoid Stieltjes calculus):** integrated/weak form.
For `s < t`, compute `∫_s^t e^{±2x/τ} Φ(x) dx` by Fubini — the inner
`x`-integrals against the kernels have closed forms — and subtract; the
result is the integrated balance identity, and right-continuity upgrades
it to pointwise.  All Fubini + explicit antiderivatives; no measure
derivatives anywhere.

### Milestone 5 — THE OPEN CORE: zero drift ⟹ `𝔞 ≡ 0` on interval supports

After Milestone 3 the enemy is measures whose combined support has
interior (e.g. absolutely continuous parts with full-interval support).
Nobody knows this step yet — this is the research frontier.  Attack
routes, in recommended order:

- **R1 (compact support first).**  Assume `supp(p+q) ⊆ [−M, M]`.  Then
  `𝔞 = 0` on `(M, ∞)` and `(−∞, −M)` by Milestone 3's gap lemma —
  in particular `𝔞(+∞) = 0` WITHOUT moment hypotheses, and under zero
  drift the balance identity (Milestone 4) transfers this to both tails.
  The unknown function `A(x) := e^{−2x/τ}𝔞(x)` satisfies: `A = C`
  (balance), `𝔟 = −2A`, `A → 0` at `±M`, plus the Stieltjes derivative
  `d𝔞 = e^{x/τ}(Q dp − P dq)` (derive + verify numerically first — this
  is the general-measure analogue of the telescoping step; its form is
  already implied by the endgame identity (I1) + E1).  Look for a
  maximum-principle / Lyapunov argument: at an interior extremum of `A`,
  the sign structure of `d𝔞` against `(Q dp − P dq)` should contradict
  extremality unless `A ≡ 0`.  Work this on paper + numerics BEFORE Lean.
- **R2 (the `K ≡ 0` alignment route, independent).**
  `K := L_p Z_q − L_q Z_p`; certified: `K ≡ 0` ⟹ `p = q`
  (`laplaceZeroDrift_imp_eq_of_companionAligned`).  Research notes:
  `K″ = −(2/τ)(L_p dq − L_q dp)` distributionally, `K → 0` at `±∞`.
  A max-principle attack at an interior extremum of `K` is natural.  Also
  compute `K`'s own `(e^{−2x/τ}, 1, e^{2x/τ})`-decomposition in the
  one-sided transforms and check (numerically) how it relates to
  `𝔞, 𝔟, 𝔠` — plausibly `K` and `𝔞` control each other, which would merge
  R1 and R2.
- **R3 (quantitative atomic + approximation).**  Make the atomic proof
  quantitative (`sup|Φ| ≤ ε ⟹ Σ|aᵢ−bᵢ| ≤ C(z,τ)·ε^θ`), then discretize a
  general zero-drift pair and pass to the limit.  DANGER: the constants in
  the telescoping induction degrade with the number of atoms and the gap
  structure; a uniform-in-`N` version is itself a hard estimate.  Log the
  attempt either way; treat as fallback.
- **R4 (counterexample search — take seriously!).**  The converse might be
  FALSE for general measures: a pair `p ≠ q` with zero drift would refute
  the paper's implicit claim and be a headline finding in its own right.
  Numeric experiment: parametrize pairs (atomic with many atoms, or
  discretized densities), minimize `sup_x |Φ(x)|` under a constraint
  keeping `dist(p,q)` bounded below; the atomic theorem forbids exact
  zeros, but if the infimum tends to 0 with mass/atoms NOT merging, the
  weak-* limit of a minimizing sequence is a candidate counterexample
  (or reveals which compactness fails).  Cheap, high information value.

### Milestone 6 — assembly and registration

Combine: `ZeroDrift ⟹ 𝔞 ≡ 0` (M5, or M3's hypothesis class) + endgame
(M2) ⟹ `laplaceZeroDrift_identifies_real`.  Then:
1. Root import in `DriftingIdentifiability.lean`.
2. Promoted list in `scripts/AxiomAudit.ps1` (headline theorems only).
3. `scripts/Check.ps1` green; `#print axioms` = foundations only.
4. Update `LaplaceArbitraryConverse.md` status header,
   `ResearchStatus.md`, and the auto-memory project-state file.
5. Update `LoggedFailures.md` with whatever died along the way.

---

## 3. Numerical verification record (2026-07-10)

Scratchpad script (PowerShell, seed 11, τ = 0.7, random atomic p (6 atoms)
and q (5 atoms), 60 random probes in [−4,4]):

- (I1) `𝔞 = Q·P⁻ − P·Q⁻`: worst residual `1.07e-14`.  Unconditional (no
  zero drift).
- (I2) `P = ∫P⁻`: midpoint rule residual `1.6e-4` at `h = 3.25e-4` — the
  predicted `O(h)` from atom-crossing cells.
- Endgame sanity: `q = 0.6·p` ⟹ `max|𝔞| = 8.7e-19`; multiplying one
  weight by 1.01 ⟹ `max|𝔞| ≈ 9.5e-4`.

Continuation note (2026-07-10, Codex): before starting
`LaplaceGeneralConverse.lean`, the Milestone-1 bracket formulas were
rechecked in PowerShell on an independent atomic pair (`tau = 0.7`, 6 vs. 5
atoms, 81 probes).  The lower bracket residual was `1.705e-13`; the upper
bracket residual was `2.998e-15`.  Implementation begins with the one-sided
transforms and algebraic bracket layer, leaving the full interval
decomposition and right-continuity items as the next Milestone-1 subgoals.

Lean continuation status (2026-07-10, Codex): `LaplaceGeneralConverse.lean`
now contains the one-sided transforms
`lowerExpMass`, `lowerCompensatedMoment`, `upperExpMass`,
`upperCompensatedMoment`; bounded integrability lemmas for the one-sided
kernels; the lower/upper bracket definitions `truncatedPairing` and
`upperPairing`; product-integral identities
`lowerBracketProductIntegral_eq_truncatedPairing` and
`upperBracketProductIntegral_eq_upperPairing`; and the direct restricted
double-integral identities
`lowerTruncatedPairingIntegral_eq_truncatedPairing` and
`upperTruncatedPairingIntegral_eq_upperPairing`.  The root import has been
added.  These were the first bracket/product pieces; later continuation
passes closed the four-region decomposition and lower-pairing regularity.

Second continuation status (2026-07-10, Codex): the four-region
decomposition and zero-drift corollary are now formalized:
`laplaceCrossDisplacementScalar_decomposition`,
`laplaceCrossDisplacementScalar_decomposition_exp`,
`laplaceZeroDrift_decomposition`, and
`laplaceZeroDrift_decomposition_exp`.  The decomposition is proved by
splitting the Laplace normalizer and displacement numerator into lower and
upper one-sided transforms, then doing the exact algebra; the exponential
version matches the roadmap's
`exp(-2x/tau) * 𝔞 + 𝔟 + exp(2x/tau) * 𝔠` notation.  The file also contains
the measure-tail lemma `tendsto_measure_Iic_atBot_zero` and the assembly
lemmas
`truncatedPairing_continuousWithinAt_of_lowerTransforms` and
`truncatedPairing_tendsto_atBot_zero_of_lowerTransforms`, which reduce the
remaining regularity part of Milestone 1 to the no-moment interval-shrinking
proofs for the lower one-sided transforms themselves.  `lake build --wfail`,
`scripts/Check.ps1`, and `#print axioms` for the new decomposition/assembly
lemmas are clean (Lean foundations only).

Third continuation status (2026-07-10, Codex): Milestone 1 is now closed.
The direct no-moment lower-transform regularity has been formalized via
filter dominated convergence:
`lowerExpMass_continuousWithinAt_Ici`,
`lowerCompensatedMoment_continuousWithinAt_Ici`,
`lowerExpMass_tendsto_atBot_zero`, and
`lowerCompensatedMoment_tendsto_atBot_zero`.  The advertised pairing-level
corollaries are now unconditional finite-measure theorems:
`truncatedPairing_continuousWithinAt_Ici` and
`truncatedPairing_tendsto_atBot_zero`.  Thus the four-region decomposition,
zero-drift corollary, bracket identities, and lower-pairing
right-continuity/tail-limit regularity are all Lean-certified; the project
can move to Milestone 2/3 without carrying a Milestone-1 regularity gap.

Still to verify numerically before formalizing (Milestone 4/5 inputs):
the unconditional `dΦ` decomposition, `d𝔞 = e^{x/τ}(Q dp − P dq)` in
integrated form (`𝔞(t) − 𝔞(s) = ∫_{(s,t]} e^{x/τ}(Q dp − P dq)`), and the
`K`-vs-`𝔞` relation of R2.

## 4. What NOT to redo (exhausted / forbidden routes)

- Escaping/tilted probes see only the sphere `‖a‖ = 1/τ` of tilt data —
  the asymptotic route is exhausted (recorded in
  `LaplaceArbitraryConverse.md`, "What the asymptotics cannot do").
- Distribution-theoretic `(τ²∂²−1)Z_p = −2τp`: bypassed; everything needed
  is classical (`LaplaceWronskian.lean`).  Do not import distribution
  machinery.
- The `V → 0` asymptotic-converse question stays OUT OF SCOPE
  (`RawFieldConverse.md` Part I §5).
- Higher-dimensional smoothing injectivity (Bessel transform) is a
  separate track (Stage 4); do not entangle it with the 1-d closure.
