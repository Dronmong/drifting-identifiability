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

## Finite-sample bridge and self-normalized consistency

`FiniteSampleBridge.lean` separates the deterministic identifiability/stability
argument from sampling error: if a random estimator `Vhat` is within `ε` of the
ideal normalized probe drift, then the coefficient error is controlled by
`(2B/c)(‖Vhat‖+ε)`, and a Markov/MSE bound turns any estimator mean-squared error
estimate into a high-probability finite-sample guarantee.

`Algorithm2Estimator.lean` proves the exact algebraic shape of the paper's
`compute_V`: it is both an affinity-weighted pairwise interaction sum and a
mass-scaled centroid difference. With nonempty positive and negative batches,
the affinity masses are strictly positive, so
`algorithm2Drift_eq_massProduct_centroidDiff` rewrites the field as
`P Q • (C_pos-C_neg)` for two self-normalized affinity centroids. This is the
correct form for analyzing the softmax estimator, because the naive sup-norm
affinity perturbation loses the cancellation across the `N²` pair terms.

`SelfNormalizedConsistency.lean` now proves the generic ratio-estimator theorem
needed for that route. Under a deterministic positive lower bound on the weights
and the reviewed sample-mean MSE axiom, the self-normalized centroid
`(∑ᵢ w(Yᵢ))⁻¹ ∑ᵢ w(Yᵢ)Yᵢ` converges in mean square to the target `c` at rate
`σ²/(wmin² N)`.

`Algorithm2SNIS.lean` performs the no-self-mask fixed-anchor instantiation. It
defines the column-reweighted weight

```text
w_i(y) = sqrt(k(x_i,y) * k(x_i,y) / sum_r k(x_r,y)).
```

For `selfMask=false`, the row-softmax denominator is common to all samples in a
centroid and cancels. Lean proves that Algorithm 2's positive and negative
centroids are exactly the SNIS centroids with weight `w_i`, and hence inherit
the `O(1/N)` MSE bound from `selfNormalized_meanSquare_le`. It also proves that
raw no-mask drift is zero iff the normalized centroid difference is zero, and
that a lower mass-product bound converts raw drift norm into centroid-difference
norm.

This still does not identify the original bare-kernel population field. The
limiting no-mask estimator targets the column-reweighted kernel above.
`ColumnReweightedMeanShift.lean` now makes this modified kernel a first-class
population setup and reuses the finite-basis theorem and finite-sample bridge:
under an explicit interaction frame bound for the modified interaction vectors,
zero of the limiting no-mask centroid field identifies the represented
measures, and random estimates of that field inherit the same MSE-to-coefficient
guarantee. `FiniteStability.lean` also proves
`interactionFrameBound_of_strictPairScaling`, which transfers a frame certificate
through positive strict-pair column scalings.

`ColumnReweightedTwoAtom.lean` discharges that certification exactly in the
two-atom model class. Three observations make the frame certificate exact
rather than perturbative:

1. `algorithm2Kernel` is definitionally the paper's positive-bandwidth Laplace
   kernel (`-‖x-y‖/τ = -(1/τ)‖x-y‖`), so the bare baseline is the already
   studied paper kernel class.
2. Against the two-point empirical reference, the interaction integral has the
   kernel-generic closed form `basisInteraction_empirical2`, and the column
   reweighting factor `1/sqrt(g(y))` depends only on the sample slot, which the
   atoms pin to the support points `{0, 1}`. Hence

   ```text
   U^col_01(n) = (1/sqrt(g(0) g(1))) · U^bare_01(n)
   ```

   exactly (`inducedInteractionVector_columnReweighted01_eq_smul`), an
   axiom-free identity. The transfer lemma
   `interactionFrameBound_of_strictPairScaling` is instantiated with this
   constant scale (`columnReweighted01_frameBound_of_bare`).
3. Strict positivity of the column-reweighted kernel makes `U^col_01` nonzero,
   so `interactionFrameBound_two` certifies the sharp constant `‖U^col_01‖`
   directly. Since `g ≤ N` at positive temperature, the reweighting costs at
   most a factor `N` against the bare constant
   (`columnReweighted01_interactionNorm_ge`).

The packaged setup `columnReweighted01Setup` discharges every analytic field
(normalizer positivity from kernel positivity, integrability from the atomic
reference, continuity of the reweighted kernel from positivity of the column
mass), with the `B = 1` normalizer bound at the anchors because the column mass
dominates its own anchor term (`k/sqrt(g) ≤ sqrt(k) ≤ 1`). The promoted
theorems `columnReweighted01_identifies_of_probeEnergy_eq_zero`,
`columnReweighted01_coefficientStability`, and
`columnReweighted01_estimate_failure_le_meanSquare` give end-to-end
identifiability, stability, and the high-probability finite-sample bridge for
the concrete class, completing the chain: sampled no-mask Algorithm-2 centroids
→ SNIS mean-square consistency → certified column-reweighted population field →
`p = q`.

`SelfMaskPerturbation.lean` now supplies the separate implementation-mask
correction.  With `δ = exp(-1000000/temperature)`, the masked sample weight is
exactly the no-mask sample weight multiplied by `δ` on masked negative entries
and by `1` elsewhere.  Comparing the masked estimator to the estimator with
those entries hard-deleted gives a deterministic affinity bound
`maskAffinityErrorBound`, and hence

```text
‖algorithm2Drift(selfMask) - deletedDrift(selfMask)‖
  ≤ 4*Npos*Nneg*R0*maskAffinityErrorBound.
```

The eye-mask specialization proves the required unmasked-column condition when
there are at least two anchors.  This is deliberately a masked-vs-deleted
comparison, not a masked-vs-full-no-mask comparison on the same reused samples.
The remaining statistical proof obligation is consistency of the deleted
estimator for the column-reweighted limiting field.  Richer (more than two
atoms, non-atomic) certified classes for the modified kernel also remain
valuable model-design work.

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
- Finite-sample bridge, Algorithm 2 structure, generic self-normalized
  consistency, the no-mask Algorithm-2 SNIS specialization, the
  column-reweighted limiting-field bridge, and its concrete two-atom certified
  frame: `FiniteSampleBridge.lean`, `Algorithm2Estimator.lean`,
  `SelfNormalizedConsistency.lean`, `Algorithm2SNIS.lean`,
  `ColumnReweightedMeanShift.lean`, `ColumnReweightedTwoAtom.lean`, and
  `SelfMaskPerturbation.lean`.

The promoted theorem uses only the reviewed paper facts
`equation_11_bilinear_mean_shift`,
`equation_31_bilinear_expansion`, and
`antisymmetric_kernel_induces_basis_antisymmetry`, besides foundational
Mathlib axioms. It has no dependency on characteristic-kernel, Gaussian
metrization, or Gaussian Gram axioms; this is enforced by `AxiomAudit.ps1`.
