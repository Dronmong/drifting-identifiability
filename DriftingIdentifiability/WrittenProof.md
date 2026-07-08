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

## Feature-space lifting

If the drift theorem is applied after a feature map `φ : X → F`, the immediate
measure-level conclusion is equality of feature laws:

```text
φ♯p = φ♯q.
```

This is the correct default statement for feature-space training.  It does not
by itself imply `p=q`: a non-injective feature can collapse two distinct source
points, and hence the corresponding distinct Dirac source laws have identical
feature laws.

`FeatureSpaceIdentifiability.lean` formalizes the safe lift.  A measurable
feature model has a law `FeatureModel.law p = φ♯p`.  Any feature-space
identifiability theorem `ZeroDrift V (φ♯p) (φ♯q) → φ♯p = φ♯q` can be used as-is
to match feature laws.  To conclude source-law equality, the proof additionally
requires `MeasurableEmbedding φ`; then Mathlib's injectivity of `Measure.map`
along measurable embeddings gives `p=q`.  For a finite family of features,
matching every feature law plus one embedded feature is enough to lift back to
source laws.

The same module now handles heterogeneous feature families
`φ_j : X → F_j`.  The exact lift is packaged as
`HeterogeneousFeatureFamily.MeasureDetermining`: equality of every feature
pushforward law implies equality of source laws.  This condition is not assumed
silently; it is proved from one measurable embedding by
`measureDetermining_of_embedding`, or must be supplied later by a concrete
measure-determining theorem.

For approximate feature matching, the project uses
`FeatureStabilityCertificate`.  Given a source discrepancy `D_X`, feature
discrepancies `D_j`, and a constant `C`, the certificate states

```text
D_X(p,q) ≤ C ∑_j D_j((φ_j)♯p, (φ_j)♯q).
```

Lean proves that per-feature bounds `D_j ≤ ε_j` imply
`D_X(p,q) ≤ C∑_j ε_j`, and that zero feature discrepancies imply source-law
equality whenever the chosen source discrepancy separates measures.  Thus
approximate learned-feature claims must exhibit an actual stability
certificate and useful constants; they are not consequences of feature matching
alone.

## CFG as affine densities

Classifier-free guidance is not a probability theorem by default.  The paper's
equation (15) defines the effective negative density

```text
q_tilde = (1-γ) q + γ u,
```

where `u` is the unconditional data density.  Solving `q_tilde = p_cond` gives
equation (16):

```text
q = (1/(1-γ)) p_cond - ((1/(1-γ))-1) u.
```

The right finite object is therefore a normalized affine coefficient vector:
the coefficients sum to one, but they may be negative.  `CFGAffine.lean`
introduces `FiniteAffineVector` for this purpose and proves that the existing
anti-symmetric finite-basis algebra still works: vanishing minors plus affine
normalization imply coefficient equality, and a positive interaction frame
controls affine coefficient error by drift error.  No nonnegativity is used in
this algebraic step.

The CFG-specific setup compares the conditional coefficients with the
effective negative coefficients `(1-γ)q + γu`.  If the finite density-interaction
drift vanishes at the probes and the interaction family is nondegenerate, then
the effective negative coefficients equal the conditional coefficients.  For
`γ≠1`, elementary algebra then gives

```text
q = cfgTargetCoefficients γ p_cond u.
```

This is `CFGDriftFiniteSetup.generated_eq_cfgTarget`.  The density bridge
lemmas show that these coefficient formulas represent the paper's
`cfgNegativeDensity`, `cfgWeightedNegativeDensity`, and equation-(16) affine
density expressions.

To recover an ordinary probability vector from the affine CFG target, the proof
requires the explicit side condition `CFGTargetNonnegative`.  Without it, the
target is only an affine signed density, so no measure-level probability claim
is made.

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
the concrete **fixed-anchor/sample-split** class, completing the chain: sampled
no-mask centroids → SNIS mean-square consistency → certified
column-reweighted population field → `p = q`.

This is not yet a theorem for the paper's exact reuse pattern `x = y_neg`.
The probability theorems take deterministic `anchors : Fin Nx → F` separately
from random `Yneg : Fin Nneg → Ω → F`. Replacing the former by `Yneg ω` makes
the column weight random and jointly dependent on the whole negative batch, so
the fixed-weight SNIS hypotheses no longer apply. A separate coupled-batch
concentration argument is required.

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

`DeletedEstimatorConsistency.lean` discharges the statistical obligation for
that deleted estimator.  The written argument:

1. The deleted drift is bilinear in the deleted affinities, so the generic
   algebra `sum_sum_mul_smul_sub` gives the mass-scaled form and, with nonzero
   masses, `deletedDrift = (P_del · Q_del) • (C⁺_del − C⁻_del)`
   (`deletedDrift_eq_massProduct_centroidDiff`), mirroring the raw drift.
2. The deleted affinity is `dw/√(R_del·C_del)`.  Splitting the square root
   (`deletedAffinity_eq_rowScale_mul_deletedColumnWeight`) exposes a common row
   scale `√(R_del(i)⁻¹)` — which cancels in each self-normalized centroid via
   `selfNormalizedCentroid_eq_of_common_scale` — times the per-slot weight
   `deletedColumnWeight i s (y) = df(i,s)·k(x_i,y)/√(Σ_{i'} df(i',s)·k(x_{i'},y))`,
   whose column mass keeps exactly the anchors unmasked in slot `s`.
3. Positive slots are never masked, so the positive per-slot weight is the
   plain column-reweighted weight (`deletedColumnWeight_inl_eq`) and the
   deleted positive centroid **equals** the promoted no-mask positive centroid
   (`deletedPositiveCentroid_eq_algorithm2PositiveCentroid_false`).  Its
   mean-square bound `σ²/(wmin²·Npos)` transports verbatim.
4. The deleted negative centroid is a self-normalized average with a
   *different* weight function on each slot, and the per-slot reweighted means
   need not share a single target.  The generic theorem
   `selfNormalizedIndexed_meanSquare_le` covers exactly this: writing
   `Z_l = w_l(Y_l)•(Y_l − c)` with means `μ l = E Z_l`, the centered summands
   are pairwise independent and mean zero, so the reviewed sample-mean axiom
   bounds `E‖Σ(Z_l − μ_l)‖² ≤ N·σ²`; the deterministic denominator floor
   `dmin ≤ Σ_l w_l(Y_l)` and `‖Σμ_l‖ ≤ N·b` then give

   ```text
   E‖ĉ − c‖² ≤ (2·N·σ² + 2·N²·b²)/dmin².
   ```

   `deletedNegativeCentroid_meanSquare_le` instantiates this with
   `w_l := deletedNegativeColumnWeight … i l`.  The bias `b` (per-slot
   leave-out reweighting bias) and the floor `dmin` are honest caller-supplied
   hypotheses; for the eye mask they are conditioning quantities of the same
   kind as the kernel floors in the no-mask theorems.  Nothing here asserts
   identifiability: the bounds feed the estimator-agnostic bridge through the
   mean squared error.

Richer (more than two atoms, non-atomic) certified classes for the modified
kernel remain valuable model-design work, as does numerically discharging the
eye-mask bias/floor hypotheses.

## High-probability denominator refinement

The Objective-7 numerics measured that the deterministic denominator floor
`dmin = N·wmin` is the dominant slack of the certified chain.
`DenominatorTail.lean` replaces it with checkable mean/variance data on the
weights.  Write `W_l := w_l(Y_l) − μw_l` for the centered weights, `D := Σ_l
w_l(Y_l)` for the random denominator, and `M := Σ_l μw_l`.

**Lemma (weight-sum lower tail).**  If the `Y_l` are pairwise independent, the
weights are bounded (`|w_l(Y_l)| ≤ wmax`), have means `μw_l` and centered
second moments `E W_l² ≤ σw²`, then for every `t > 0`

```text
P{D < M − t} ≤ N σw² / t².
```

*Proof.*  `D − M = Σ W_l`, so on the event `{D < M − t}` we have
`t < |Σ W_l|`.  The `W_l` are pairwise independent (composition of independent
variables with measurable maps), mean zero (`∫W_l = μw_l − μw_l`), and have
second moments `≤ σw²`; the reviewed sample-mean axiom gives
`E|N⁻¹ Σ W_l|² ≤ σw²/N`, hence `E|Σ W_l|² ≤ N σw²`.  Markov on the squared
norm (`meas_gt_le_meanSquare_div`) bounds `P{t < |Σ W_l|}` by
`E|Σ W_l|²/t² ≤ N σw²/t²`.  ∎

**Theorem (deviation probability with a random denominator).**  Under the
hypotheses of the indexed ratio theorem — minus the deterministic floor — plus
the weight mean/variance data above, for every `ε > 0` and `0 < t < M`

```text
P{ε < ‖ĉ − c‖} ≤ (2Nσ² + 2N²b²) / ((M − t)² ε²)  +  N σw² / t².
```

*Proof.*  Split on the denominator event.  If `D ≥ M − t > 0`, the pointwise
ratio identity `ĉ − c = D⁻¹ • Σ Z_l` is valid and
`‖ĉ − c‖ ≤ (M − t)⁻¹ ‖Σ Z_l‖`, so the deviation event is contained in
`{(M − t)ε < ‖Σ Z_l‖}`; otherwise `ω` lies in the tail event of the lemma.
Subadditivity of the measure gives two terms.  For the first, centering
`Z_l = Z'_l + μ_l` gives the pointwise bound
`‖Σ Z_l‖² ≤ 2‖Σ Z'_l‖² + 2(Nb)²`; the sample-mean axiom bounds
`E‖Σ Z'_l‖² ≤ N σ²`, and Markov on the squared norm yields
`(2Nσ² + 2N²b²)/((M−t)²ε²)`.  The second term is the lemma.  ∎

Dependencies: the reviewed `sampleMean_meanSquare_le` axiom (twice: once for
the vector summands, once for the scalar weights) and the axiom-free Markov
lemma.  No step assumes identifiability, and no new axiom is introduced.  The
data `(μw, σw)` are checkable expectations of the weight function alone; the
theorem holds for every split point `t`, which the caller (or the numeric
ledger) optimizes.

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
- Feature-space pushforward conclusions and measurable-embedding lifts:
  `FeatureSpaceIdentifiability.lean`.
- CFG affine-density coefficients, finite affine identifiability/stability,
  density bridges for equations (15)/(16), and the CFG drift theorem:
  `CFGAffine.lean`.
- Finite-sample bridge, Algorithm 2 structure, generic self-normalized
  consistency (common-weight and indexed/bias-tolerant), the no-mask
  Algorithm-2 SNIS specialization, the column-reweighted limiting-field
  bridge, its concrete two-atom certified frame, the self-mask perturbation,
  and the deleted-estimator consistency: `FiniteSampleBridge.lean`,
  `Algorithm2Estimator.lean`, `SelfNormalizedConsistency.lean`,
  `Algorithm2SNIS.lean`, `ColumnReweightedMeanShift.lean`,
  `ColumnReweightedTwoAtom.lean`, `SelfMaskPerturbation.lean`, and
  `DeletedEstimatorConsistency.lean`.

The promoted theorem uses only the reviewed paper facts
`equation_11_bilinear_mean_shift`,
`equation_31_bilinear_expansion`, and
`antisymmetric_kernel_induces_basis_antisymmetry`, besides foundational
Mathlib axioms. It has no dependency on characteristic-kernel, Gaussian
metrization, or Gaussian Gram axioms; this is enforced by `AxiomAudit.ps1`.
