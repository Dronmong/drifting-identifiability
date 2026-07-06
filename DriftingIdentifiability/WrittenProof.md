# Written proof: finite population mean-shift identifiability

## Theorem and scope

Let `μ` be a reference probability measure and let `φ₁,…,φₘ` be measurable,
nonnegative, integrable densities satisfying `∫φᵢ dμ = 1`. For simplex
coefficients `a,b`, define the genuine probability measures

```text
p = (∑ᵢ aᵢφᵢ) · μ,     q = (∑ᵢ bᵢφᵢ) · μ.
```

Fix a mean-shift kernel and finitely many probes. Assume all integrals in
equations (11) and (31) are valid, both normalizers are nonzero at every probe,
and the strict-pair interaction vectors have a positive frame lower bound.
Then zero finite squared normalized drift at the selected probes implies
`p=q`. Pointwise zero of the full field is a stronger corollary.

This statement is about the ideal population field in data space. It excludes
CFG/signed targets and does not identify the finite-batch estimator with the
population field.

## Proof

1. Each mixture density is measurable, nonnegative, integrable, and has
   integral one. Therefore `Measure.withDensity` produces probability measures
   `p` and `q`.
2. Applying the standard `withDensity` integration identity twice proves that
   the measure-level interaction integral against `p.prod q` equals the
   density-weighted iterated integral against `μ`.
3. Equation (11) writes normalized mean-shift drift as the inverse product of
   its two nonzero normalizers times the interaction numerator. Hence zero
   normalized drift forces the numerator to vanish at every probe.
4. Equation (31) expands the stacked numerator as
   `∑ᵢ∑ⱼ aᵢbⱼ Uᵢⱼ`.
5. Anti-symmetry groups this ordered sum into
   `∑_{i<j}(aᵢbⱼ-aⱼbᵢ)Uᵢⱼ`. This grouping is proved in Lean rather than
   axiomatized.
6. A positive frame bound implies linear independence, so every minor
   `aᵢbⱼ-aⱼbᵢ` is zero.
7. Probability normalization gives
   `aᵢ-bᵢ = ∑ⱼ(aᵢbⱼ-aⱼbᵢ)=0`; therefore `a=b`.
8. Equal coefficients give definitionally equal mixture measures, hence
   `p=q`.

No step assumes uniqueness of a zero-drift equilibrium or injectivity of the
drift-to-distribution map.

## Population-loss bridge

For the nonnegative function `x ↦ ‖Vₚ,q(x)‖²`, an integrable zero population
integral implies `Vₚ,q=0` almost everywhere under `q`. If the field is
continuous and `q` assigns positive measure to every nonempty open set,
continuous functions equal almost everywhere are equal everywhere. The main
theorem then applies.

Without full support this upgrade is false; `FailureCases.lean` gives the
continuous identity field under `dirac 0` as a counterexample.

## Stability

Let `c>0` satisfy

```text
c ∑_{i<j}|zᵢⱼ| ≤ ‖∑_{i<j} zᵢⱼ Uᵢⱼ‖.
```

The ordered minor mass is twice the strict-pair mass, while normalized
coefficients satisfy

```text
‖a-b‖₁ ≤ ∑ᵢ∑ⱼ |aᵢbⱼ-aⱼbᵢ|.
```

If the absolute normalizer product is at most `B` at every probe, then

```text
‖a-b‖₁ ≤ (2B/c) ‖normalized probe drift‖.
```

The same uniform estimate proves coefficient convergence from probe-drift
convergence. Nondegeneracy also requires
`card(StrictPair m) ≤ N · dim(E)`, which is now machine checked.

## Probe-local hypothesis (Objective 2)

The proof of the main theorem consumes the drift hypothesis only through step 3,
and only at the `N` probes: it uses `Vₚ,q(probe n) = 0`, never any other point.
The promoted theorem is therefore stated directly from the finite hypothesis

```text
∀ n, normalizedProbeDrift n = 0,     normalizedProbeDrift n := Vₚ,q(probe n),
```

as `finitePopulationMeanShift_identifies_of_probeZero`. The pointwise `ZeroDrift`
statement is recovered as the corollary obtained by instantiating this at the
probes, `fun n ↦ hzero (probe n)`. Because the finite hypothesis is exactly the
vanishing of the `N`-vector whose norm controls `‖a-b‖₁` in the stability
estimate, the hypothesis, the observable, and the stability bound all refer to
the same object. Full topological support is not used by this route; it is only
needed by the separate zero-energy corollary.

Define the deterministic finite probe loss by

```text
L_probe := ∑ₙ ‖normalizedProbeDrift n‖².
```

Because this is a finite sum of nonnegative terms, Lean proves
`L_probe = 0 ↔ ∀ n, normalizedProbeDrift n = 0`. Hence
`finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero` states the exact
result directly from zero finite loss, while quantitative stability becomes

```text
‖a-b‖₁ ≤ (2B/c) √L_probe.
```

## Explicit ceiling on the frame constant (Objective 1)

Instantiating the frame inequality at the coordinate indicator `Pi.single p 1`
(mass one, synthesis equal to `U_p`) gives, with no axiom,

```text
InteractionFrameBound U c  →  ∀ p, c ≤ ‖U_p‖,     hence  c ≤ min_p ‖U_p‖.
```

For the structured Gaussian family, `U_p(n) = e^{-n²}·column_p·base_pⁿ` with
`column_p = e^{-(zᵢ²+zⱼ²)/2}(zᵢ-zⱼ)` and `base_p = e^{zᵢ+zⱼ}`, so writing
`Δ = zᵢ-zⱼ` and completing the square in the exponent,

```text
|U_p(n)| = |Δ|·exp(-(n-(zᵢ+zⱼ)/2)²)·exp(-Δ²/4) ≤ |Δ| e^{-Δ²/4},
```

with equality only at the generally non-integer probe `n=(zᵢ+zⱼ)/2`. Hence

```text
c ≤ min_{i<j} |zᵢ-zⱼ| e^{-(zᵢ-zⱼ)²/4} ≤ √(2/e) ≈ 0.858.
```

This is a computable ceiling from the support geometry alone. It is proved as
`interactionFrameBound_le_interactionNorm` (general) and
`gaussianEmpiricalPoint_frameConstant_le` (Gaussian). It certifies numerical
uselessness whenever some pair of support points is close or far.

The complementary certified lower constant is obtained from the actual square
interaction matrix `M`, with probes as rows and strict pairs as columns:

```text
c_cert := (∑_{p,r}|(M⁻¹)_{p,r}|)⁻¹.
```

If the interaction vectors are independent, Lean proves `c_cert > 0` and
`c_cert ‖z‖₁ ≤ ‖Mz‖∞`. This is
`interactionFrameBound_inverseCertificate`; its structured-Gaussian
specialization is `gaussianEmpiricalPointCertifiedFrameBound`. The formula is
computable/certifiable, although it may be extremely small.

## Concrete general finite family

Let `m≥2`, let `z₀,…,zₘ₋₁∈ℝ` be distinct, and let the reference law be the
uniform empirical distribution on these points. Define the `i`th basis density
to equal `m` at `zᵢ` and zero at every other support point. These functions are
measurable, nonnegative, and have unit integral, so their simplex mixtures are
genuine probability measures.

For the unit-bandwidth Gaussian kernel and the mean-shift interaction, direct
finite integration gives

```text
Uᵢⱼ(n) = k(n,zᵢ)k(n,zⱼ)(zᵢ-zⱼ),
```

using one integer probe `n` for each strict pair. Expanding the Gaussian gives

```text
Uᵢⱼ(n)
 = exp(-n²)
   [exp(-(zᵢ²+zⱼ²)/2)(zᵢ-zⱼ)]
   [exp(zᵢ+zⱼ)]ⁿ.
```

The first factor is nonzero for every probe. The second is nonzero because the
support points are distinct. If all strict-pair sums `zᵢ+zⱼ` are distinct, the
geometric bases `exp(zᵢ+zⱼ)` are distinct. The square evaluation matrix is a
Vandermonde matrix and is nonsingular, so the actual induced vectors are
linearly independent. The inverse interaction-matrix formula above supplies a
specific positive frame constant.

This proves `gaussianEmpiricalPoint_identifies`: no frame or injectivity
conclusion is assumed by its caller. The main concrete setup uses
`gaussianEmpiricalPointCertifiedFrameConstant`, not the older existential
choice. Since the unit Gaussian satisfies `0<k≤1`, each normalizer lies in
`(0,1]`, so the product bound is explicitly `B=1`; the concrete energy estimate
is `‖a-b‖₁ ≤ (2/c_cert)√L_probe`.

## Higher-dimensional and practical model-class extensions

For a real inner-product data space `F`, choose a direction `u` and probes
`xₙ=n u`. The Gaussian interaction factors into a nonzero row scaling, a
geometric profile whose base is
`exp(⟪u,zᵢ+zⱼ⟫/σ²)`, and the vector weight `zᵢ-zⱼ`. Distinct projected pair sums
therefore give a vector-weighted Vandermonde family. This yields the complete
arbitrary-positive-bandwidth theorem
`gaussianEmpiricalPointND_identifies_of_probeEnergy_eq_zero`.

For arbitrary or adaptively selected probes, an `InteractionDualCertificate`
stores continuous linear functionals `Lₚ` satisfying

```text
Lₚ(U_q) = 1 when p=q, and 0 otherwise.
```

Then `c=(∑ₚ ‖Lₚ‖)⁻¹` is a valid frame constant. This is finite, checkable linear
algebra and supports fewer probes whenever the stacked output dimension is
large enough.

For continuous or smooth probability-density bases, uniform interaction
perturbation gives

```text
Frame(U,c),   supₚ ‖Uₚ-U'ₚ‖ ≤ δ < c
    ⟹ Frame(U',c-δ).
```

Thus a smooth model inherits exact identifiability and stability from a
certified baseline once the finite analytic/numerical error is bounded. The
same transfer covers moved probes and approximated kernels. Finally,
`empirical01Laplace_identifies_of_probeEnergy_eq_zero` instantiates the paper's
positive-bandwidth Laplace kernel for the two-atom, one-probe family.

## Concrete non-atomic smooth model

`SmoothBumpBasis.lean` instantiates the smooth interface without a perturbation
assumption. The reference law is the standard Gaussian on `ℝ`. Two normalized
`C∞` bump densities have disjoint ordered supports, one strictly negative and
one strictly positive. Their `withDensity` mixtures are genuine non-atomic
probability measures.

For the strict pair `(0,1)`, positivity of the Gaussian kernel and support
ordering force the interaction integrand to have a fixed negative sign on a
positive-measure product set. Hence its integral is strictly negative, so the
single interaction vector is nonzero and supplies the two-component frame
bound. For every positive Gaussian bandwidth,
`bumpGaussian_identifies_of_probeEnergy_eq_zero` concludes equality of the two
represented measures from zero finite drift energy at one probe. No
characteristic-kernel or synthetic Gaussian axiom is used.

## Legitimacy and failure audit

The registered `finiteBasisCandidate` admits distinct laws whenever two
basis-induced mixtures are unequal; the first two simplex vertices give a
convenient witness for `m≥2`. This is established before imposing zero drift.

Formal regressions cover zero normalizers, collapsed bases, duplicated
interaction vectors, insufficient probe dimension, missing full support, and
non-injective feature maps. A measurable embedding is sufficient to lift
feature-law equality back to source-law equality.

## Lean map and dependencies

- Probability measures and normalized bridge:
  `PopulationIdentifiability.lean`.
- Algebraic grouping and exact coefficient proof: `FiniteGrouping.lean` and
  `PaperFiniteIdentifiability.lean`.
- Quantitative frame estimates: `FiniteStability.lean`.
- Concrete empirical bases and the axiom-free Gaussian/Vandermonde theorem:
  `EmpiricalFrameBound.lean`.
- Higher-dimensional, adaptive-probe, smooth-transfer, and Laplace interfaces:
  `PracticalModelClasses.lean`.
- Concrete non-atomic `C∞` bump basis and direct sign-certified frame:
  `SmoothBumpBasis.lean`.
- Candidate legitimacy and regressions: `PopulationCandidate.lean` and
  `FailureCases.lean`.

The promoted theorem uses only the reviewed paper facts
`equation_11_bilinear_mean_shift`,
`equation_31_bilinear_expansion`, and
`antisymmetric_kernel_induces_basis_antisymmetry`, besides foundational
Mathlib axioms. It has no dependency on characteristic-kernel, Gaussian
metrization, or Gaussian Gram axioms; this is enforced by `AxiomAudit.ps1`.
