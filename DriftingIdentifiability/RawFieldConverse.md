# Raw-field general converse: landscape record + axiom-route plan

**Status:** Part II **COMPLETED AXIOM-FREE 2026-07-08**.
The promoted theorem `gaussianMeanShiftDrift_identifiesAtZero` proves the exact
raw Gaussian converse for arbitrary probability measures in finite-dimensional
real inner-product spaces. `GaussianScoreRecovery.lean` proves the Gaussian
score identity and recovers equal scalar normalizers;
`GaussianConvolutionInjectivity.lean` proves those normalizers determine the
measure. `#print axioms` reports only Lean foundations. Part I records where
the question stands across the paper, the authors' reviewer rebuttal, and this
project; Part II records the now-completed staged proof.

Main track (not the Sinkhorn extension). Follows the candidate discipline in
`AGENTS.md` / `CandidateConditions.lean`: legitimacy check → counterexample
thought → written proof → Lean theorem.

---

## Part I. The landscape: paper vs rebuttal vs us

The theoretical claim at issue is the **converse / identifiability** of the
drifting field: does vanishing drift force the two laws to coincide? Written
for the paper's mean-shift field (equations 8/10):

```
meanShift k p x    = Z_p(x)⁻¹ · ∫ k(x,y)(y−x) dp(y)          -- eq 8
meanShiftDrift k p q x = meanShift k p x − meanShift k q x   -- eq 10  ("V")
Z_p(x) := kernelNormalizer k p x = ∫ k(x,y) dp(y)
```

The target predicate is `IdentifiesAtZero condition (meanShiftDrift k)`, i.e.
`∀ p q, condition p q → (∀ x, V p q x = 0) → p = q` (`TrustedBoundary.lean`).

### 1. The paper (arXiv:2602.04770v2)

Gives a **heuristic** identifiability argument in Appendix C.1: represent `p, q`
by finite coefficients (eq 29), expand the drift bilinearly (eqs 30–31), and
claim a nondegeneracy that forces equal coefficients hence `p = q`. It is a
sketch — the nondegeneracy is asserted, not proved, and there is no rigorous
statement of scope. This is precisely the "theory gap (W3)" a reviewer flagged.

### 2. The authors' reviewer rebuttal (the screenshot)

Two **informal analytic** arguments, "planned for the revised paper":

- **Rebuttal argument 1 (Gaussian kernel, general targets).** With a Gaussian
  kernel the mean-shift field equals `σ²[∇ₓ log p̂_σ(x) − ∇ₓ log q̂_σ(x)]`
  (score difference of the Gaussian-smoothed densities). If `V ≡ 0` then
  `p̂_σ = q̂_σ`, and Gaussian convolution is injective, so `p = q`.
- **Rebuttal argument 2 (Laplacian kernel, Gaussian targets).** For
  `k(x,y)=exp(−‖x−y‖/τ)` with `P,Q` Gaussian, the field probed along `x = ru`
  satisfies `V(ru) → (μ_p−μ_q) + (Σ_p−Σ_q)u/τ` as `r→∞`, determining both mean
  and covariance for the Gaussian family.

They explicitly concede: *"The general converse for arbitrary fields remains
open."*

### 3. What we have (this project)

Rigorous, machine-checked, and quantitative — the only non-heuristic artifact
in the picture. Three relevant pieces:

- **Trusted route (default, no conditional axioms).** The finite-basis
  interaction-frame / Vandermonde converse: for a finite probability-density
  basis whose induced interaction vectors satisfy a positive frame bound,
  `V = 0` at the selected probes forces `p = q` (`finitePopulationMeanShift_
  identifies`; frame discharged axiom-free for explicit Gaussian point-mass
  families in `EmpiricalFrameBound.lean` and a `C∞` bump basis). Crucially it is
  **exact and quantitative**: `‖a−b‖₁ ≤ (2B/c)‖V_probes‖`, plus a finite-sample
  bridge. Neither the paper nor the rebuttal has the quantitative form. Machine
  check: 160 promoted declarations, *no* conditional Gaussian/RKHS dependencies.

- **Conditional characteristic route (opt-in, `DriftingIdentifiability.
  Conditional`).** `characteristicKernel_identifiesAtZero` /
  `gaussianMmd_identifiesAtZero` (`CharacteristicIdentifiability.lean`) are
  conditional reductions of rebuttal argument 1 for the **MMD field**
  `mmdDrift kg p q x = ∫kg(x,·)dp −
  ∫kg(x,·)dq` (an *unnormalized* embedding difference), not the raw normalized
  `meanShiftDrift`. They rest on `characteristic_gradientEmbedding_injective`
  and `gaussian_gradient_isCharacteristic` (Sriperumbudur et al. 2010,
  axiomatized) and are explicitly labeled "reductions, not accepted project
  solutions."

- **Conditional asymptotic route.** `gaussianMmd_asymptoticallyIdentifies`:
  MMD-discrepancy → 0 implies weak convergence (Lévy–Prokhorov), via the
  metrization axiom `gaussian_mmd_metrizes_weakConvergence`. Again the *MMD
  discrepancy*, not the raw field norm.

### 4. Head-to-head

| claim | paper | rebuttal | us |
|---|---|---|---|
| exact `V≡0 ⟹ p=q`, finite/structured basis | heuristic | — | **proved, axiom-free, quantitative** |
| exact `V≡0 ⟹ p=q`, Gaussian, arbitrary target | — | arg 1 (informal) | **proved axiom-free for the raw field** |
| asymptotic `V→0 ⟹ qₙ→p` | — | arg 1/2 (informal) | conditional, MMD discrepancy only |
| Gaussian/Laplacian moment recovery (arg 2) | — | arg 2 (informal) | **proved axiom-free and promoted** (`laplaceGaussianMeanShiftDrift_identifiesAtZero`; see `LaplacianGaussianConverse.md`) |
| estimator-level finite-sample theory | — | — | **present (Algorithm 2 SNIS etc.)** |

### 5. The honest open problem (agreed by everyone)

The general converse for the **raw** drift field of **arbitrary** targets:
`V ≡ 0 ⟹ p = q` for `meanShiftDrift k`, general `p, q`. Two facts frame it:

- **Kernel structure is unavoidable.** The project's flat-kernel
  counterexample already shows that distinct equal-mean laws can have zero raw
  drift, so an arbitrary-kernel theorem is false. A band-limited
  translation-invariant kernel also fails ordinary embedding characteristicness,
  but transferring that failure to normalized mean shift needs a
  kernel-specific numerator/score relation. Conversely, ordinary
  characteristicness alone does not provide such a relation. The Gaussian
  theorem uses both full-spectrum convolution injectivity and its special score
  identity.
- **The raw field is normalized, and that is the technical crux** (see Part II).
  The asymptotic (`V→0`) version is strictly harder and may be false as literally
  stated for the raw field: kernel smoothing kills high-frequency differences, so
  `‖V‖` can be tiny while `qₙ ↛ p` (`LoggedFailures.md`). The asymptotic version
  needs a metrizing discrepancy (MMD) or tightness hypotheses and is **out of
  scope** for this plan, which targets the exact `V≡0` case only.

---

## Part II. Plan for the open piece (axiom route)

**Goal.** An opt-in conditional theorem: for the Gaussian kernel and
**arbitrary** probability measures, `meanShiftDrift (gaussianKernel σ) ≡ 0 ⟹
p = q`. This bridges the gap between "what we proved (MMD field)" and "the raw
mean-shift field the algorithm's population idealization actually uses," and
formalizes the Lean composition of rebuttal argument 1 modulo two
clearly-scoped, cited external analytic facts. It remains a conditional
reduction, not a promoted project solution.

### The technical crux (why the existing axioms do NOT already give this)

The existing conditional route closes the **MMD field** in one line because
`mmdDrift = ∫kg dp − ∫kg dq` **is** the embedding difference, so
`mmdDrift = 0 ⟺ embeddings match`, and `characteristic_gradientEmbedding_
injective` fires directly.

The raw field is different. Using `equation_11_bilinear_mean_shift`,
`meanShiftDrift k p q x = (Z_p(x) Z_q(x))⁻¹ · ∫∫ k(x,y₁)k(x,y₂)(y₁−y₂) d(p×q)`,
and since `Z_p, Z_q > 0` (from `MeanShiftRegularAt`),

```
V ≡ 0  ⟺  A_p(x)·Z_q(x) = Z_p(x)·A_q(x)  for all x,   A_p(x) := ∫ k(x,y) y dp
       ⟺  A_p(x)/Z_p(x) = A_q(x)/Z_q(x)              (normalized conditional means match)
```

This is `meanShift k p = meanShift k q` (immediate from the definition too:
`V = meanShift p − meanShift q`). It is **not** `A_p = A_q` — the MMD/embedding
condition. The two differ by the per-law normalizers `Z_p, Z_q`. So MMD
embedding injectivity does not apply as-is; the raw converse needs injectivity of
the **normalized conditional-mean map** `p ↦ meanShift k p`. For the Gaussian
this is exactly the score/convolution-injectivity fact of rebuttal argument 1
(the normalizer is the smoothed density, and `meanShift ∝ ∇log(p∗φ_σ)`), but as
a proof it has two conceptually separate stages.

### The two analytic stages

The score-recovery stage is the theorem
`gaussianMeanShift_eq_imp_kernelNormalizer_eq` in
`GaussianScoreRecovery.lean`:

```lean
theorem gaussianMeanShift_eq_imp_kernelNormalizer_eq
    {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
    [BorelSpace E] [SecondCountableTopology E]
    (σ : ℝ) (hσ : ValidBandwidth σ)
    (p q : Distribution E) [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (h : ∀ x, meanShift (gaussianKernel σ) p x = meanShift (gaussianKernel σ) q x) :
    ∀ x, kernelNormalizer (gaussianKernel σ) p x =
      kernelNormalizer (gaussianKernel σ) q x
```

The first stage formalizes Tweedie's formula plus the
score-determines-normalized-density step: equal mean-shift maps imply equal
Gaussian-smoothed scalar functions. The second stage, ordinary Gaussian
convolution/kernel-embedding injectivity, is now the proved theorem
`DriftingIdentifiability.gaussianKernelNormalizer_injective`. It
Fourier-transforms the scalar normalizer, factors it as the characteristic
function of the measure times a nowhere-zero Gaussian transform, cancels that
factor, and applies Mathlib's `Measure.ext_of_charFun`. The two stages have
independently meaningful intermediate conclusions; neither is the raw drift
converse under another name.

Sources:

- Bradley Efron, *Tweedie's Formula and Selection Bias*, JASA 106 (2011),
  §1–2, for the Gaussian posterior-mean/score identity.
- Sriperumbudur et al., *Hilbert Space Embeddings and Metrics on Probability
  Measures*, JMLR 11 (2010), Theorem 9 and the Gaussian example, for Gaussian
  characteristicness/injectivity.

Notes:

- The normalization mismatch means there is no direct
  `meanShiftDrift ⟹ mmdDrift` rewrite. The score stage must first recover the
  scalar normalizers.
- If a more general-kernel statement is wanted later, mirror the existing
  abstraction only after identifying a concrete analogue of the Gaussian score
  identity; ordinary kernel characteristicness alone does not imply normalized
  mean-shift injectivity.

### The theorem (promoted two-stage bridge)

The final theorem is in `GaussianScoreRecovery.lean`, reusing
`BothProbability`:

```lean
/-- **Raw-field identifiability for the Gaussian mean-shift drift.**  Zero raw
mean-shift drift (equation 10) between two probability measures forces equality.
Closes the gap left by `gaussianMmd_identifiesAtZero`, which handled only the
unnormalized MMD field. Rests on score recovery followed by the proved Gaussian
normalizer-injectivity theorem. -/
theorem gaussianMeanShiftDrift_identifiesAtZero (σ : ℝ) (hσ : ValidBandwidth σ) :
    IdentifiesAtZero (BothProbability (E := E)) (meanShiftDrift (gaussianKernel σ)) := by
  rintro p q ⟨hp, hq⟩ hzero
  haveI := hp; haveI := hq
  have hmeanShift : ∀ x,
      meanShift (gaussianKernel σ) p x =
        meanShift (gaussianKernel σ) q x :=
    fun x => sub_eq_zero.mp (hzero x)
  have hnormalizer :=
    gaussianMeanShift_eq_imp_kernelNormalizer_eq σ hσ p q hmeanShift
  exact gaussianKernelNormalizer_injective σ hσ p q hnormalizer
```

`ZeroDrift V p q` unfolds to `∀ x, V p q x = 0`, and
`meanShiftDrift k p q x = meanShift k p x − meanShift k q x` by definition, so
`hzero x : meanShift … p x − meanShift … q x = 0` and `sub_eq_zero.mp` gives the
first-stage hypothesis. The resulting normalizer equality is exactly the input
to Gaussian convolution injectivity.

### Legitimacy obligation (mandatory, per `AGENTS.md`)

The condition must be shown not to secretly encode `p = q`. `BothProbability`
already has `bothProbability_allowsDistinctPair` (two distinct Dirac masses) —
reuse it; the target is `ConditionAllowsDistinctPair (BothProbability)`, already
proved. No new legitimacy proof needed, but the plan must *cite* it so the
reviewer sees the condition is nonvacuous. Optionally add an
`IsExactCounterexample` note recording the band-limited-kernel obstruction from
Part I §5 (shows why the kernel hypothesis cannot be dropped).

### Audit / build checklist

- [x] The bundled `gaussianMeanShift_injective` axiom was removed because it was
      propositionally equivalent to the desired converse after unfolding
      `meanShiftDrift`.
- [x] `GaussianScoreRecovery.lean` proves positivity, Gaussian-weighted
      integrability, differentiation under the integral, and
      `D log Zₚ = σ⁻²⟪meanShiftₚ,·⟫`.
- [x] Equal mean-shift maps give a constant `log Zₚ-log Z_q`, hence
      proportional normalizers; probability normalization forces the constant
      to one.
- [x] The former `Paper.gaussianKernelNormalizer_injective` axiom was removed.
      `GaussianConvolutionInjectivity.lean` proves the same conclusion by
      Fourier calculation and characteristic-function uniqueness.
- [x] `gaussianMeanShiftDrift_identifiesAtZero` is imported by the default root
      and registered in `AxiomAudit.ps1`.
- [x] `gaussianKernelNormalizer_injective` is imported by the default root and
      registered in the promoted axiom audit.
- [x] `#print axioms gaussianKernelNormalizer_injective` reports only
      `propext`, `Classical.choice`, and `Quot.sound`.
- [x] `#print axioms gaussianMeanShiftDrift_identifiesAtZero` reports only
      `propext`, `Classical.choice`, and `Quot.sound` (not `equation_11` or any
      conditional Gaussian/RKHS axiom).

The legitimacy obligation is met by the existing
`bothProbability_allowsDistinctPair`. A formal counterexample showing that an
arbitrary kernel hypothesis is impossible remains an optional future addition;
ordinary characteristicness should not be conflated with the additional
score-recovery structure needed by normalized mean shift.

### Scope statement to carry into any write-up

This deliverable gives an **axiom-free Lean proof** of the exact
`V ≡ 0 ⟹ p = q` converse for the **raw** Gaussian mean-shift field and
arbitrary probability measures under the explicit finite-dimensional,
Borel/second-countable, complete real inner-product assumptions. It
deliberately does **not**: (a) address the asymptotic `V → 0` version; or
(b) extend the conclusion to arbitrary kernels.
Rebuttal argument 2 (Laplacian/Gaussian moment recovery) remains separate.

### Optional follow-ups (recorded, not in this deliverable)

- Investigate a general-kernel abstraction that includes both ordinary
  embedding characteristicness and a concrete score/normalizer recovery law;
  characteristicness by itself is insufficient for normalized mean shift.
- Rebuttal argument 2 as a Lean instance: an asymptotic `x = ru, r→∞`
  moment-recovery theorem for the Gaussian family under the Laplacian kernel
  (parametric family, not arbitrary targets). Detailed implementation plan:
  `LaplacianGaussianConverse.md`.
- The genuinely novel prize: a *quantitative* stability modulus `‖V‖ small ⟹
  D(p,q) small` for a characteristic kernel — beyond paper and rebuttal (both
  existence-only), in the spirit of our finite-basis `(2B/c)` bound. Obstruction:
  the modulus is frequency-dependent and degrades, which is exactly why the clean
  asymptotic statement resists formalization.
