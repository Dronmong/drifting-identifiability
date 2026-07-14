# Research status

## Executive answer

The original mathematical existence goal has been achieved: the project gives
explicit, nonvacuous conditions under which zero ideal population mean-shift
drift forces `p = q`, and the result is machine checked without new or
conditional Gaussian/RKHS axioms.

This is an exact, restricted population theorem. No promoted theorem currently
shows that vanishing raw mean-shift drift along a sequence forces convergence
of distributions in a specified topology; the paper's asymptotic question
remains open.

The practical modeling goal is not yet complete, but Objectives 1--3 are now
complete at the deterministic population level. Objective 4 is complete for
the **fixed-anchor/sample-split** no-mask route (SNIS consistency → certified
column-reweighted field → identifiability), and supplies a deterministic
implementation-mask perturbation bound plus a fixed-anchor statistical
consistency theorem for the leave-masked-out/deleted estimator. It does **not**
yet cover the paper's coupled implementation where the random generated
negative batch is reused as the anchor batch, `x = y_neg`. Objective 3 includes a fully
instantiated non-atomic `C∞` bump basis on a Gaussian reference, in addition to
the higher-dimensional, variable-bandwidth, adaptive-probe, perturbative, and
Laplace infrastructure. What remains under Objective 4 is first a
concentration/consistency theorem for that reused-negative random-anchor
coupling, then numeric instantiation (per-slot leave-out bias and denominator
floors for concrete masks) and richer certified modified-kernel model classes.
Model-quality evaluation remains empirical work.

## What has been accomplished

### 1. The finite-dimensional heart of Appendix C.1 is formalized

The project now rigorously implements the paper's heuristic chain:

1. Equation (29): represent `p` and `q` using finite coefficients `a` and `b`.
2. Equation (30): define the integral-induced interaction vectors `Uᵢⱼ`.
3. Equation (31): expand the population drift as `∑ᵢ∑ⱼ aᵢbⱼUᵢⱼ`.
4. Antisymmetry: prove `Uᵢⱼ = -Uⱼᵢ` and group the expansion into
   `∑_{i<j}(aᵢbⱼ-aⱼbᵢ)Uᵢⱼ`.
5. Nondegeneracy: a linearly independent strict-pair family forces every
   coefficient minor `aᵢbⱼ-aⱼbᵢ` to vanish.
6. Probability normalization: vanishing minors imply `a=b`.
7. Equal coefficients produce equal represented probability measures.

The grouped-minor algebra and the normalized-coefficient conclusion are proved
in Lean rather than axiomatized. No step assumes zero-drift uniqueness or an
equivalent injectivity conclusion.

### 2. The density heuristic was upgraded to genuine probability measures

A `ProbabilityDensityBasis` requires every basis function to be measurable,
nonnegative, integrable, and unit mass relative to a reference probability
measure. Mixtures are constructed using `Measure.withDensity`, and Lean proves
that they are probability measures.

The project also proves that the measure-level product integral against the two
mixture laws agrees with the density-weighted interaction integral used in the
Appendix C.1 calculation.

### 3. The normalized paper field is connected to the bilinear numerator

Under explicit integrability and nonzero-normalizer assumptions, equation (11)
turns zero normalized population mean-shift drift into zero unnormalized
interaction drift at every required probe. Equation (31) then connects this to
the finite coefficient-minor argument.

The promoted generic result is
`finitePopulationMeanShift_identifies`. It proves equality of the represented
probability measures for any finite probability-density basis whose actual
integral-induced interaction vectors satisfy the frame condition.

### 4. Exact, approximate, and energy statements are separated

`InteractionFrameBound U c` strengthens qualitative independence to

```text
c ∑_{i<j}|zᵢⱼ| ≤ ‖∑_{i<j} zᵢⱼ Uᵢⱼ‖,    c>0.
```

The following consequences are proved:

- a positive frame bound implies qualitative nondegeneracy;
- in finite dimension, qualitative nondegeneracy implies that some positive
  frame constant exists;
- coefficient error satisfies
  `‖a-b‖₁ ≤ (2B/c)‖V_probes‖`;
- finite squared probe loss satisfies
  `‖a-b‖₁ ≤ (2B/c)√(∑ₙ ‖V(xₙ)‖²)`;
- vanishing probe drift gives coefficient convergence under uniform bounds;
- `card(StrictPair m) ≤ N · dim(E)` is necessary for nondegeneracy.

Separately, zero integrable population drift energy implies almost-everywhere
zero drift. Continuity and full topological support upgrade this to pointwise
zero and allow the exact theorem to be applied. A formal counterexample records
why full support cannot silently be omitted.

### 5. The frame hypothesis is discharged for an explicit general-`m` family

`EmpiricalFrameBound.lean` constructs, for every `m ≥ 2`:

- an injective support `z : Fin m → ℝ`;
- the uniform empirical probability reference on those points;
- point-mass functions proved to form a genuine probability-density basis;
- the unit-bandwidth Gaussian mean-shift kernel;
- `N = card(StrictPair m) = m(m-1)/2` integer probes.

For this basis, the actual equation (30) integral is computed exactly:

```text
Uᵢⱼ(n) = k(xₙ,zᵢ) k(xₙ,zⱼ) (zᵢ-zⱼ).
```

At the structured probes, the Gaussian interaction factors as

```text
Uᵢⱼ(n)
 = exp(-n²)
   [exp(-(zᵢ²+zⱼ²)/2)(zᵢ-zⱼ)]
   [exp(zᵢ+zⱼ)]ⁿ.
```

If the support points are distinct and all strict-pair sums `zᵢ+zⱼ` are
distinct, the geometric bases are distinct. An axiom-free Vandermonde argument
therefore proves that the actual integral-induced vectors are linearly
independent. This is not the synthetic Gaussian construction from the
conditional layer.

The resulting end-to-end theorem is `gaussianEmpiricalPoint_identifies`. Its
caller supplies the geometric support conditions, coefficients, and zero
population drift; it does not supply a frame bound or hidden injectivity
assumption. An explicit three-point support verifies that the geometric
condition is realizable, while the equally spaced four-point regression shows
that distinct points alone do not prevent pair-sum collisions.

The earlier `m=2` route is also fully packaged by
`empirical01Gaussian_identifies` for arbitrary positive Gaussian bandwidth.

### 6. Both stability constants are explicit

For the unit Gaussian, `0 < k(x,y) ≤ 1`. The concrete normalizers therefore lie
in `(0,1]`, and the product bound is `B=1`. The concrete stability theorem is

```text
‖a-b‖₁ ≤ (2/c) ‖V_probes‖.
```

The frame constant used by the concrete setup is now
`gaussianEmpiricalPointCertifiedFrameConstant z`, the reciprocal entrywise
`ℓ¹` mass of the inverse square interaction matrix. Thus both `B` and `c` are
finite formulas determined by the model geometry. The certificate can still
be extremely small; computing and optimizing its scaling is now an applied
conditioning problem rather than a logical gap in the theorem.

### 7. Trust and failure boundaries are enforced

- The promoted results use only the reviewed paper facts for equations (11)
  and (31) and basis-interaction antisymmetry.
- Conditional characteristic-kernel and synthetic Gaussian modules are
  excluded from the default proof route.
- Counterexamples cover collapsed bases, duplicated interactions,
  insufficient probe dimension, zero normalizers, missing support, and
  non-injective feature maps.
- The trust audit, warning-as-error build, conditional-module build, and
  promoted-theorem axiom audit pass.

### 8. An explicit computable ceiling on the frame constant

Testing the frame inequality on a single coordinate indicator proves, with no
axiom, that *every* valid frame constant is bounded above by the norm of each
individual interaction vector:

```text
InteractionFrameBound U c  →  ∀ p, c ≤ ‖U_p‖.
```

For the structured Gaussian family this norm has a closed-form ceiling. With
`Δ = zᵢ − zⱼ`, each probe value factors as
`|U_p(n)| = |Δ| · exp(-(n-(zᵢ+zⱼ)/2)²) · exp(-Δ²/4) ≤ |Δ| exp(-Δ²/4)`, so

```text
c ≤ min_{i<j} |zᵢ - zⱼ| · exp(-(zᵢ - zⱼ)² / 4).
```

The scalar function `|Δ| e^{-Δ²/4}` is maximized at `|Δ| = √2` with value
`√(2/e) ≈ 0.858`, and decays to `0` both as two support points collide
(`Δ → 0`) and — because the unit Gaussian suppresses distant interaction — as
they separate (`Δ → ∞`). This is a rigorous, computable description of when the
exact theorem is numerically useless, and an absolute `m`-independent ceiling on
any achievable constant. The complementary inverse-matrix lower certificate is
given in accomplishment 10. The promoted declarations here are
`interactionFrameBound_le_interactionNorm` and
`gaussianEmpiricalPoint_frameConstant_le`.

### 9. The exact theorem is driven by the finite probe-drift vector

The promoted population theorem is now stated and proved directly from vanishing
of the *finite observable* drift vector, not pointwise zero drift everywhere:

```text
(∀ n, normalizedProbeDrift n = 0)  →  p = q.
```

`finitePopulationMeanShift_identifies_of_probeZero` is the probe-local core; the
pointwise-`ZeroDrift` theorem `finitePopulationMeanShift_identifies` is now a
one-line corollary of it. The concrete end-to-end theorems
`gaussianEmpiricalPoint_identifies_of_probeZero` and
`empirical01Gaussian_identifies_of_probeZero` expose the same weaker hypothesis.
The drift vector whose vanishing is assumed is exactly the finite quantity whose
norm appears in the `‖a-b‖₁ ≤ (2B/c)‖V_probes‖` stability estimate, so the
hypothesis, the observable, and the stability control now all refer to the same
`N`-dimensional probe-drift vector. This removes the full-topological-support
requirement from the exact route (that hypothesis was only needed by the
zero-energy corollary).

### 10. Objective 1 is closed by an inverse-matrix certificate

For a square scalar interaction family, `strictPairInteractionMatrix` records
the actual interaction vectors as columns. Lean proves that if these columns
are independent, then

```text
c_cert = 1 / ∑_{p,r} |(M⁻¹)_{p,r}|
```

is positive and satisfies the required `ℓ¹` frame inequality. The theorem is
`interactionFrameBound_inverseCertificate`; the concrete Gaussian
specialization is `gaussianEmpiricalPointCertifiedFrameBound`. The main
`gaussianEmpiricalPointSetup` now uses this certified constant, so its exact and
stability theorems no longer depend on a classically selected existential
constant.

This is a rigorous computable lower bound, not a promise of good conditioning.
Together with the earlier ceiling, it cleanly separates logical
identifiability from numerical usefulness. Closed-form estimates in terms of
minimum separation and asymptotic scaling in `m` remain valuable engineering
and analysis questions, but are no longer missing proof obligations.

### 11. Objective 2 is closed at the deterministic population level

`probeDriftEnergy` is the finite loss

```text
∑ₙ ‖V(xₙ)‖².
```

Lean proves that it is zero exactly when every selected probe drift is zero.
Consequently `finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero` and
its concrete Gaussian corollaries identify the measures directly from zero
finite probe loss. The quantitative companion controls coefficient error by
the square root of that loss. No topology, full-support hypothesis, or
continuity upgrade is used in this finite route.

Random minibatch concentration is deliberately not folded into Objective 2:
it is the separate finite-sample problem recorded as Objective 4.

### 12. Objective 3 has a general deterministic infrastructure

`PracticalModelClasses.lean` adds four complementary routes:

1. `gaussianEmpiricalPointNDSetup` packages the vector-weighted Vandermonde
   proof into the complete normalized population theorem for any separable real
   inner-product data space and any positive Gaussian bandwidth.
2. `InteractionDualCertificate` gives an independently checkable vector-valued
   conditioning certificate. Its reciprocal total operator-norm mass is a valid
   frame constant. `gaussianEmpiricalPointCertifiedProbeSetup` therefore accepts
   arbitrary, fewer, or adaptively selected probes when they carry such a finite
   certificate; the existing dimension lower bound still applies.
3. `SmoothProbabilityDensityBasis` and `ContinuousProbabilityDensityBasis`
   describe genuine probability-density model classes. The robust perturbation
   theorem transfers a baseline frame constant `c` to the actual smooth
   interaction system with constant `c-δ` whenever its finite interaction error
   is at most `δ<c`.
4. `empirical01LaplaceSetup` gives a fully instantiated theorem for the paper's
   positive-bandwidth Laplace kernel using one probe in the two-atom family.

The perturbation theorem also applies to kernel approximation and probe
movement. None of these routes equates population loss with a minibatch
estimator or claims that a positive certificate is numerically large.

### 13. A fully instantiated continuum-supported smooth basis (Objective 3 gap)

`SmoothBumpBasis.lean` closes the last outstanding Objective-3 instantiation with
a genuinely continuum-supported model, not an atomic one:

- **Reference:** the standard Gaussian `gaussianReal 0 1` (full support, proved
  `IsOpenPosMeasure`).
- **Densities:** two `C∞` bump functions (`ContDiffBump.normed`) on *disjoint,
  ordered* supports — `φ₀` on `(-3/2,-1/2)`, `φ₁` on `(1/2,3/2)`. They form a
  genuine `SmoothProbabilityDensityBasis` (measurable, nonnegative, unit mass,
  and `C∞`); their mixtures are formally proved non-atomic probability
  measures.
- **Certified frame, proved directly:** because the supports are ordered,
  `y₊ - y₋ < 0` throughout the support of `φ₀(y₊)φ₁(y₋)`, so the single
  interaction double integral is *sign-definite* and hence nonzero
  (`basisInteraction_bump_neg`, a positivity argument — no Gaussian closed form).
  `interactionFrameBound_two` then yields a positive `InteractionFrameBound`
  (`bumpInteractionFrameBound`). The sign lemma itself uses only positivity and
  support ordering; the promoted population setup nevertheless requires the
  paper's intended positive Gaussian bandwidth.
- **End to end:** `bumpGaussianSetup` packages the complete population setup
  (all regularity discharged: normalizers positive because the kernel is
  positive and the mixture is a probability measure; integrability from the
  compactly supported bump densities via `prod_withDensity`). The promoted
  theorem `bumpGaussian_identifies_of_probeEnergy_eq_zero` concludes that zero
  finite normalized population-drift energy at a single probe and positive
  Gaussian bandwidth forces equality of the represented continuum probability
  measures.

This is the first *non-atomic* end-to-end identifiability result in the project.
It is axiom-free (no paper axioms beyond the reviewed equation-11/31 machinery
inherited by the population theorem).

## Strongest result and its exact scope

The strongest completely instantiated non-atomic result currently says:

> Let `p` and `q` be simplex mixtures of two normalized `C∞` bump densities on
> disjoint ordered intervals, relative to the standard Gaussian reference law.
> For any positive Gaussian bandwidth and any single probe, if the finite
> squared normalized population-drift loss is zero, then `p=q`.

The general atomic theorem remains broader in component count and data-space
dimension: it supports arbitrary finite `m` in separable real inner-product
spaces, structured probes, or arbitrary probes carrying a dual certificate.

This is an exact population theorem in data space. It does not presently claim:

- arbitrary-component smooth mixtures beyond the proved two-component bump
  family;
- equality of source laws from a non-injective feature representation;
- equivalence between Algorithm 2's finite bi-softmax estimator and the
  population field;
- a result for CFG or signed target laws;
- that gradient-based optimization reaches the zero-drift hypothesis;
- that the concrete conditions preserve practical model expressivity or
  generation quality.

The atomic mixtures do not have full topological support, and the smooth bump
mixtures have compact component support even though their Gaussian reference
has full support. This does not weaken either exact route: both consume drift at
the selected probes directly. Full support is needed only by the separate
global-energy-to-pointwise-zero corollary.

### 14. The accepted candidate now matches the theorem

`finiteBasisCandidate` is retained as the preliminary shared-family condition.
`finitePopulationMeanShiftCandidate` is the accepted theorem-level
`CandidateSpec`: its pair condition contains a complete
`PopulationMeanShiftFiniteSetup`, hence the fixed kernel, represented
probability measures, selected probes, regularity, integrability, and positive
interaction frame consumed by the proof.
`finitePopulationMeanShiftCandidate_identifiesAtZero` proves the canonical
`IdentifiesAtZero` statement for the fixed `meanShiftDrift` field, while
`finitePopulationMeanShiftCandidate_isLegitimate` requires an independently
distinct represented pair and makes no zero-drift assumption.

## Objectives and remaining priorities

### Objective 1: explicit lower frame certificate — complete

The concrete theorem now uses the finite formula

```text
c_cert = (∑_{p,r}|(M⁻¹)_{p,r}|)⁻¹,
```

where `M` is the actual square structured-Gaussian interaction matrix. Lean
proves `c_cert > 0` and `InteractionFrameBound U c_cert`. This replaces the
noncomputably selected compactness constant in the promoted concrete setup.
The earlier geometric ceiling remains useful for detecting bad configurations.

Further work may derive easier closed-form estimates from separation,
bandwidth, and `m`, or choose probes that maximize the certificate. Those are
conditioning/design improvements under Objectives 3 and 7, not gaps in exact
identifiability or stability.

### Objective 2: finite query-access population loss — complete

The exact theorem consumes the deterministic finite probe loss
`∑ₙ ‖V(xₙ)‖²`. Zero loss is proved equivalent to zero at every selected probe,
and small loss controls coefficient error through its square root. The concrete
Gaussian theorems expose both conclusions directly. This is a query-access
theorem: it assumes those selected probes can be evaluated. Structured probes
need not be generated anchors observed by the paper's loss. The separate
continuity/full-support energy theorem can recover pointwise zero from
generated-law population energy, but that route does not apply automatically
to finite atomic model laws.

Estimating this population quantity from random reused minibatches is not
silently assumed. Concentration, estimator bias/dependence, and high-probability
sample complexity remain Objective 4.

### Objective 3: practical model-class infrastructure — complete

Completed formally:

- arbitrary separable real inner-product data spaces;
- arbitrary positive Gaussian bandwidth;
- exact and quantitative finite-loss theorems in higher dimension;
- finite vector-valued dual certificates for arbitrary/adaptive probe sets;
- robust transfer to continuous or `C∞` probability-density bases;
- a concrete positive-bandwidth Laplace-kernel theorem;
- preservation of the necessary probe-dimension obstruction;
- **a fully instantiated continuum-supported smooth basis with a certified
  frame and end-to-end identifiability** (`SmoothBumpBasis.lean`, accomplishment
  13 below).

The one substantive instantiation that was previously outstanding — construct
and certify a *continuum-supported* smooth basis and prove its frame directly —
is now done. It is not the perturbation route (`δ<c`); the frame is certified
by an exact sign argument. What remains under Objective 3 is empirical
model-design tuning (approximation power, richer supports, adaptive probes),
which is engineering/analysis, not missing theorem infrastructure.

### Objective 4: finite-sample guarantees — fixed-anchor/sample-split routes complete; reused-negative coupling open

Progress (done, `FiniteSampleBridge.lean`):

- **Deterministic propagation of estimator error through `2/c`.**
  `coefficientStability_of_estimate`: if an observed estimate `Vhat` of the
  population probe-drift is within `ε` (sup norm) of the truth, then
  `‖a-b‖₁ ≤ (2B/c)(‖Vhat‖ + ε)`. This is the third desired result, and it isolates
  the statistical content (`ε`) as an explicit hypothesis rather than assuming
  it. `ε = 0`, `Vhat = normalizedProbeDrift` recovers `coefficientStability`.
- **High-probability lift.**  `estimate_failure_measure_le`: for a *random*
  estimate `Vhat : Ω → (Fin N → E)`, the event that the coefficient bound fails is
  contained in the event that the estimate is not within `ε` of the truth, so its
  probability is at most that of the estimation-error event (monotonicity only —
  no independence/measurability/distributional assumption). Composing with any
  concentration bound `P{ε < ‖V − Vhat‖} ≤ δ` gives
  `P{coefficient bound holds} ≥ 1 - δ`.

- **Markov concentration and the sample-complexity bound.**
  `meas_gt_le_meanSquare_div` proves `P{ε < ‖Z‖} ≤ E‖Z‖²/ε²` for any estimator
  error `Z` (Markov on the squared norm — no independence assumption).
  `estimate_failure_le_meanSquare` composes the three steps into the concrete
  bound: the probability that `‖a-b‖₁ ≤ (2B/c)(‖Vhat‖+ε)` *fails* is at most
  `E‖V − Vhat‖² / ε²`. Hence the bound holds with probability `≥ 1-δ` as soon as
  the estimator mean squared error satisfies `E‖V − Vhat‖² ≤ δε²`; for a
  sample-mean estimator with MSE `σ²/N` this is `N ≥ σ²/(δε²)`.

- **Explicit `1/M` sample-mean rate.**  `sampleMean_concentration` composes the
  Markov bound with the variance-of-a-sample-mean fact
  (`Paper.sampleMean_meanSquare_le`, a reviewed external statistical axiom) into
  `P{ε < ‖(1/M)∑Zᵢ‖} ≤ σ²/(Mε²)`.  This is the concrete `1/M`
  law-of-large-numbers rate: an unbiased sample-mean estimator with per-sample
  variance `σ²` achieves identifiability accuracy `ε` at confidence `1-δ` once
  `M ≥ σ²/(δε²)`.  The iid variance bound is the one place the finite-sample
  route uses an external statistical axiom (Micchelli-style: standard,
  non-conditional, allow-listed); the deterministic bridge and Markov step
  remain axiom-free.

- **Self-normalized ratio-estimator consistency.**  `SelfNormalizedConsistency.lean`
  proves `selfNormalized_meanSquare_le`: if weights satisfy a deterministic
  lower bound `wmin>0`, weighted centered samples
  `w(Yᵢ) • (Yᵢ-c)` are mean-zero with second moment at most `σ²`, and the samples
  are pairwise independent, then the ratio estimator
  `(∑ᵢ w(Yᵢ))⁻¹ • ∑ᵢ w(Yᵢ) • Yᵢ` has mean-squared error at most
  `σ²/(wmin² N)`.  This is the missing generic self-normalized-IS ingredient
  identified by the failed sup-affinity perturbation route.  It introduces no new
  axiom and its promoted axiom audit reports only `Paper.sampleMean_meanSquare_le`
  plus Lean foundations.

- **Algorithm-2 no-mask SNIS specialization.**  `Algorithm2SNIS.lean` implements
  the fixed-anchor, `selfMask=false` route.  It defines the exact
  column-reweighted weight
  `sqrt(k(x_i,y) * k(x_i,y) / sum_r k(x_r,y))`, proves that Algorithm 2's
  positive and negative centroids equal the corresponding self-normalized
  centroids (`algorithm2PositiveCentroid_false_eq_columnReweighted`,
  `algorithm2NegativeCentroid_false_eq_columnReweighted`), and lifts
  `selfNormalized_meanSquare_le` to Algorithm 2's positive and negative
  centroids (`algorithm2PositiveCentroid_false_meanSquare_le`,
  `algorithm2NegativeCentroid_false_meanSquare_le`).  It also proves that raw
  no-mask Algorithm-2 drift is zero exactly when the normalized centroid
  difference is zero, and that a lower bound on the mass product converts raw
  drift norm into centroid-difference norm.

- **Column-reweighted limiting field bridge.**  `ColumnReweightedMeanShift.lean`
  promotes the deterministic residual to a first-class setup. It defines the
  modified population kernel
  `sqrt(k(x,y) * k(x,y) / sum_r k(anchor_r,y))`, proves its positivity, packages a
  `ColumnReweightedMeanShiftFiniteSetup`, and reuses the verified finite-basis
  theorem and finite-sample bridge for this modified kernel. Thus zero of the
  limiting no-mask centroid field identifies the represented measures under an
  explicit `InteractionFrameBound` for the modified interaction vectors. It also
  adds the estimator bridge
  `ColumnReweightedMeanShiftFiniteSetup.estimate_failure_le_meanSquare`, so a
  random estimator of this modified population field inherits the same
  high-probability coefficient guarantee.

- **Frame transfer tool for modified kernels.**  `FiniteStability.lean` now
  includes `interactionFrameBound_of_strictPairScaling`: if a modified
  interaction family is obtained from a certified baseline family by scaling
  each strict-pair column by a factor bounded below by `smin>0`, the frame
  constant degrades only from `c` to `smin*c`. This is the intended bridge from
  empirical/bare-kernel certificates to column-reweighted certificates when the
  modified kernel factors into positive per-support-point column factors.

- **Concrete certified column-reweighted frame (two-atom class).**
  `ColumnReweightedTwoAtom.lean` discharges the modified-kernel frame condition
  exactly, closing the "prove an `InteractionFrameBound` for the actual
  column-reweighted interaction vectors" item:
  - `algorithm2Kernel_eq_laplaceKernel`: Algorithm 2's kernel *is* the paper's
    positive-bandwidth Laplace kernel, so the bare baseline is the certified
    paper kernel class.
  - Against the two-atom empirical basis, the kernel-generic closed form
    `basisInteraction_empirical2` pins the reweighting factor `1/sqrt(g(y))` to
    the support atoms, so the modified interaction vector is an **exact**
    strict-pair rescaling of the bare one:
    `U^col_01 = (1/sqrt(g(0)g(1))) • U^bare_01`
    (`inducedInteractionVector_columnReweighted01_eq_smul`, axiom-free), with
    the transfer lemma instantiated in `columnReweighted01_frameBound_of_bare`
    and the sharp norm identity in `columnReweighted01_interactionNorm_eq`.
  - `columnReweighted01_interactionNorm_ge`: at positive temperature `g ≤ N`,
    so the column reweighting costs at most a factor `N` (the anchor count) in
    the frame constant — the explicit conditioning price of the column softmax.
  - `columnReweighted01Setup` packages the complete certified population setup
    (frame certified directly by kernel positivity via
    `interactionFrameBound_two`; all analytic obligations discharged; `B = 1`
    at the anchors because `k/sqrt(g) ≤ sqrt(k) ≤ 1` there).  The promoted
    theorems `columnReweighted01_identifies_of_probeEnergy_eq_zero`,
    `columnReweighted01_coefficientStability`, and
    `columnReweighted01_estimate_failure_le_meanSquare` complete the
    **fixed-anchor/sample-split** chain: sampled no-mask centroids → SNIS
    mean-square consistency → certified column-reweighted population field →
    `p = q`, with an explicit high-probability sample-complexity bound. They do
    not permit substituting `anchors := Yneg ω`, because that makes the weight
    function random and jointly batch-dependent. The promoted theorems depend
    only on the three reviewed equation-11/31/antisymmetry paper axioms; the
    rescaling identity itself is axiom-free.

- **Deleted-estimator statistical consistency.**  `SelfNormalizedConsistency.lean`
  now also proves `selfNormalizedIndexed_meanSquare_le`, the indexed,
  bias-tolerant ratio theorem: per-index weight functions `w l`, an abstract
  deterministic denominator floor `dmin`, per-index reweighted mean shifts
  `μ l` bounded by `b`, and per-index centered second moments bounded by `σ²`
  give `E‖ĉ − c‖² ≤ (2Nσ² + 2N²b²)/dmin²`.  `DeletedEstimatorConsistency.lean`
  instantiates it for the leave-masked-out estimator targeted by the
  implementation mask, with the exact index-dependent structure:
  - `deletedDrift_eq_massProduct_centroidDiff` (axiom-free): the deleted drift
    is a mass product times a difference of self-normalized centroids, exactly
    like the raw drift.
  - Row cancellation (`deletedAffinity_eq_rowScale_mul_deletedColumnWeight`):
    the deleted affinity factors into a common row scale times the per-slot
    weight `deletedColumnWeight`, whose column mass drops precisely the anchors
    masked in that slot's column (`deletedNegativeColumnWeight` packages the
    negative-slot family).
  - **The mask does not move the positive centroid**
    (`deletedPositiveCentroid_eq_algorithm2PositiveCentroid_false`, axiom-free):
    positives are never masked and the row scale cancels, so the no-mask SNIS
    bound transports verbatim (`deletedPositiveCentroid_meanSquare_le`).
  - `deletedNegativeCentroid_meanSquare_le`: the negative centroid is a
    self-normalized average with a *different* weight function on each slot;
    its mean-square bound consumes the per-slot leave-out biases `‖μ l‖ ≤ b`
    and the mask-aware floor `dmin`.  Both consistency theorems depend only on
    the reviewed `sampleMean_meanSquare_le` axiom plus Lean foundations — no
    new axiom was needed.

Estimator structure (done, `Algorithm2Estimator.lean`): the bridge above is
estimator-*agnostic*.  This module discharges the estimator-specific facts about
Algorithm 2's actual bi-softmax `compute_V` (`Paper.algorithm2Drift`) that hold
without any distributional or limit hypothesis, all axiom-free (`#print axioms`
reports only `propext`/`Classical.choice`/`Quot.sound` — no paper axioms):

- **Range.**  The row and column softmax affinities and their geometric mean lie
  in `[0,1]` (`algorithm2Affinity_nonneg`, `algorithm2Affinity_le_one`); the
  positive/negative sample weights are nonnegative and bounded by the
  opposite-sample count (`algorithm2PositiveWeight_le` ≤ `Nneg`,
  `algorithm2NegativeWeight_le` ≤ `Npos`).
- **Exact structural form.**  `algorithm2Drift_eq_affinityPairSum`: the estimator
  equals the affinity-weighted pairwise attraction minus repulsion
  `∑_{j,l} A(i,+j) A(i,-l) • (yPos j − yNeg l)`, the finite-sample instance of the
  mean-shift interaction kernel `Paper.meanShiftInteractionKernel`
  `(k x y⁺)(k x y⁻) • (y⁺ − y⁻)` with the affinities `A` in the role of the kernel
  `k`.  Equivalently it is the mass-scaled centroid difference
  `Q • (Σ A(+) yPos) − P • (Σ A(−) yNeg)` (`algorithm2Drift_eq_massScaledCentroid`).
  With nonempty positive and negative batches, the affinity masses are strictly
  positive (`algorithm2PositiveMass_pos`, `algorithm2NegativeMass_pos`), and
  `algorithm2Drift_eq_massProduct_centroidDiff` rewrites the drift as
  `P Q • (C_pos − C_neg)`, where each `C` is a self-normalized affinity centroid.
  This is the formal algebraic hook connecting Algorithm 2 to
  `selfNormalized_meanSquare_le`.
- **Boundedness.**  `algorithm2Drift_norm_le`: with all samples in the ball of
  radius `R`, `‖algorithm2Drift i‖ ≤ 2 · Npos · Nneg · R`, uniformly in the
  temperature and self-mask — the bounded-range fact a bounded-differences
  (McDiarmid-type) concentration argument builds on.  `algorithm2Drift_norm_le_affinityMass`
  refines it to the data-adaptive convex-hull bound `2 · P · Q · R`, where
  `P = Σⱼ A(i,+j)` and `Q = Σₗ A(i,−l)` are the total affinity masses (each `≤`
  the sample count); this exposes that the softmax-weighted sample sums are
  mass-scaled convex combinations of the samples.
- **Matched-batch cancellation.**  `algorithm2Drift_matched_zero`: with coinciding
  positive/negative samples and no self-mask, `algorithm2Drift = 0`.  This is the
  sample-level analogue of `equation_17_matched_batch_drift_zero` (matched laws ⟹
  zero batch drift) and of the population reverse implication `p = q ⟹ V = 0`; it
  is the safe direction and assumes nothing about identifiability.

Remaining work after the current Objective-4 implementation:

- **Paper implementation's reused-negative coupling — open.** Algorithm 1 sets
  `x = y_neg`. The current SNIS and deleted-estimator probability theorems take
  `anchors : Fin Nx → F` as deterministic data separate from
  `Yneg : Fin Nneg → Ω → F`. Instantiating the former with `Yneg ω` invalidates
  the fixed per-slot weight-function hypothesis. Closing this requires a
  leave-one-out, conditional, U-statistic, bounded-difference, or comparable
  batch-functional concentration argument. Until then, the finite-sample
  certificate applies to fixed-anchor or sample-split variants, not the exact
  reused-negative training batch.

- the **quantitative bias/consistency** of the implementation estimator after
  deletion/leave-out masking — **now formalized**.  The no-mask fixed-anchor
  centroids have an SNIS consistency theorem, the finite `1e6` mask is
  deterministically close to the deleted estimator, and
  `DeletedEstimatorConsistency.lean` now proves the statistical theorem for the
  deleted estimator itself, with the exact index-dependent leave-masked-out
  structure in its hypotheses (per-slot weight functions, per-slot leave-out
  bias `b`, mask-aware denominator floor `dmin`).  It was proved directly from
  the reviewed sample-mean axiom rather than imported, and it does not assert
  the identifiability conclusion: it feeds the estimator-agnostic bridge
  through the mean squared error.

  An attempt to close this by an elementary *deterministic* reduction — bound the
  estimator error by the sup-norm deviation of the softmax affinities from their
  population values, using the bilinear stability of `driftOfAffinities` — was
  investigated and **fails by a scaling mismatch**: the crude bilinear bound is
  `ε ~ 4·Npos·Nneg·η`, but a single softmax weight is `O(1/N)` with sampling
  fluctuation `η ~ N^{-3/2}`, so `ε ~ √N → ∞`.  The estimator's consistency comes
  from *cancellation across the `N²` self-normalized pairs*, which a sup-norm
  perturbation bound discards.  The correct handle is the mass-scaled centroid form
  `algorithm2Drift_eq_massScaledCentroid` (each centroid a self-normalized average
  with the standard `O(1/N)` importance-sampling bias).  The generic
  self-normalized-IS consistency theorem and its no-mask Algorithm-2 centroid
  specialization are now formalized in `SelfNormalizedConsistency.lean` and
  `Algorithm2SNIS.lean`.

  The honest residual has been split cleanly.  The conditional modified-kernel
  identifiability theorem is formalized in `ColumnReweightedMeanShift.lean`, and
  `ColumnReweightedTwoAtom.lean` now **instantiates the condition concretely**:
  the two-atom class carries a certified `InteractionFrameBound` for the actual
  column-reweighted interaction vectors (exact strict-pair rescaling of the bare
  Laplace family — no perturbation argument), completing the no-mask
  fixed-anchor sampled-estimator-to-identifiability chain in a concrete model
  class. The additional fixed-anchor residuals are scoped and explicit:
  - the implementation **self-mask** (`selfMask = true` with the `1e6` penalty)
    is formalized in `SelfMaskPerturbation.lean` as a deterministic
    `δ = exp(-1000000/temperature)` perturbation of the leave-masked-out/deleted
    estimator, not of the full no-mask estimator on the same samples;
  - the statistical consistency of that deleted estimator is now proved
    (`DeletedEstimatorConsistency.lean`): the positive centroid coincides with
    the no-mask one, and the negative centroid satisfies the indexed
    bias-tolerant SNIS bound.  What remains is *numeric instantiation*: for a
    concrete mask (e.g. `eyeMask` at batch size `N`), discharging the per-slot
    bias bound `b` (the one-anchor-drop reweighting bias, expected `O(1/N)`)
    and the denominator floor `dmin` from kernel floors — conditioning
    estimates of the same kind as Objectives 3/7, not missing theorem
    infrastructure;
  - richer certified classes for the modified kernel (more than two atoms,
    non-atomic bases) — the same design/conditioning work already recorded
    under Objectives 3 and 7 for the bare kernel.
  This route and the earlier obstruction are recorded in `LoggedFailures.md`
  (2026-07-06 entry).

**Implemented and verified (2026-07-07): high-probability denominator
refinement.**
The Objective-7 conditioning ledger (`numerics/RESULTS.md`, E5) measured that
the single dominant slack in the certified finite-sample chain is the
*deterministic* SNIS denominator floor `dmin = N·wmin`, which pays the full
worst-case kernel value `e^{-1/tau}` and alone separates the certified sample
complexity (`~8.6e8` at `tau = 0.2`, astronomically worse at smaller `tau`)
from the LLN-typical (`~2.9e5`) and observed (`~2e5`) complexities.

`DenominatorTail.lean` now implements the planned repair while staying inside
the existing trusted boundary.  No new axiom is introduced: both promoted
theorems depend only on Lean foundations and the reviewed
`Paper.sampleMean_meanSquare_le` statistical axiom.

1. `SelfNormalized.weightSum_lower_tail_prob_le`: for independent per-slot
   weights with means `μw l` and centered second moments `≤ σw²`,
   `P{Σ w_l(Y_l) < Σ μw - t} ≤ N σw²/t²` (Chebyshev lower tail via centering
   and `meas_gt_le_meanSquare_div`).
2. `SelfNormalized.selfNormalizedIndexed_deviation_prob_le`: a
   deviation-probability form of the indexed ratio theorem.  Splitting on
   `{Σ w ≥ Σ μw - t}` gives
   `P{ε < ‖ĉ - c‖} ≤ (2Nσ² + 2N²b²)/((Σμw - t)² ε²) + N σw²/t²`, so the
   deterministic floor hypothesis is replaced by *checkable mean/variance
   hypotheses on the weights*.  This is a probability-level composition; it
   plugs into the coefficient bridge through the same event-inclusion lemma as
   before.

This does not touch identifiability itself: it sharpens the constants of the
finite-sample route only.  The theorem deliberately does not assume the
denominator's mean is large for free — `Σ μw`, `σw`, and the split point `t`
remain caller-supplied checkable data, with the side condition
`0 < t < Σ μw`.

The Objective-7 ledger has now been re-run with this theorem.  In the two-atom
certified class, optimizing `t = ρNμw` lowers the certified sample count from
`~8.6e8` to `~1.3e6` at `tau = 0.2`, and removes the astronomical small-`tau`
deterministic-floor explosion.  The refined certificate remains conservative
relative to the observed extrapolation (`~2e5`) because it still pays an
explicit denominator-tail term and a union bound over four centroids.

### Objective 5: handle feature-space training correctly — core infrastructure complete

The current data-space theorem gives equality of data-space measures. In
feature space, zero drift can generally identify only the pushforward laws.
`FeatureSpaceIdentifiability.lean` now formalizes this boundary.

Completed formally:

- `FeatureModel` packages a measurable feature map `φ : X → F`.
- `FeatureModel.law` is the paper's feature law `φ♯p`.
- `law_eq_of_feature_identifiesAtZero` says a feature-space zero-drift theorem
  gives equality of feature laws by default.
- `source_eq_of_law_eq` and
  `source_eq_of_feature_identifiesAtZero` lift feature-law equality to
  source-law equality only under an explicit `MeasurableEmbedding φ`
  hypothesis.
- `AllFeatureLawsEqual` and the multi-feature lift theorem show that one
  embedded feature in a finite feature family is enough to recover source-law
  equality after all feature laws have been matched.
- `featureLaw_collision_distinct_source_diracs` is the formal warning: in a
  measurable space that separates points, a non-injective feature collision
  gives distinct source Dirac laws with equal feature laws.
- `HeterogeneousFeatureFamily` handles finite feature families whose feature
  spaces differ by index, e.g. pixels, embeddings, logits, and auxiliary
  statistics in one training objective.
- `HeterogeneousFeatureFamily.MeasureDetermining` packages the exact
  measure-determining condition for a heterogeneous family, and
  `measureDetermining_of_embedding` proves it from one embedded feature.
- `FeatureStabilityCertificate` is the quantitative replacement for exact
  embedding: it certifies a real source discrepancy by a finite weighted sum of
  feature-law discrepancies. `sourceDist_le_of_featureDist_le` turns per-feature
  approximate bounds into a source-discrepancy bound, and
  `source_eq_of_featureDist_zero_of_stability` recovers exact source equality
  when the certified source discrepancy separates measures.

This closes the logical guardrail and the abstract quantitative route for
feature-space training. Remaining Objective-5 work is now concrete
instantiation: choose source and feature discrepancies (total variation,
Wasserstein, MMD, task metrics, etc.), prove or estimate a usable
`FeatureStabilityCertificate` for the actual representation, and evaluate
whether the resulting constants are meaningful for learned non-injective
features.

### Objective 6: treat CFG separately — affine-density core complete

CFG equation (16) can define a signed affine target rather than a probability
measure. It is outside the current probability-mixture theorem.
`CFGAffine.lean` now gives it a separate finite affine-density treatment.

Completed formally:

- `FiniteAffineVector` records finite coefficients with total mass one but no
  nonnegativity requirement.
- `affineParallelCoefficientsAreEqual`, `affineCoefficientIdentifiable`, and
  `AffineDriftFiniteSetup.identifiesCoefficients` generalize the finite
  anti-symmetric minor argument from probability coefficients to affine
  coefficients.
- `affineCoeffL1_le_of_frame_scaledDrift` gives the corresponding quantitative
  coefficient-stability estimate.
- `cfgNegativeCoefficients` formalizes equation (15) at coefficient level:
  `q̃ = (1-γ)q + γu`.
- `cfgTargetCoefficients` formalizes equation (16):
  `q = (1/(1-γ))p - ((1/(1-γ))-1)u`.
- `cfgGenerated_eq_target_of_effective_eq` proves the algebraic CFG solve:
  if the effective negative coefficients equal the conditional coefficients and
  `γ≠1`, then the generated coefficients equal the CFG affine target.
- `basisDensity_cfgNegativeCoefficients`,
  `basisDensity_cfgTargetCoefficients`, and
  `basisDensity_cfgWeightedNegativeCoefficients` connect the coefficient
  formulas to the paper's density-level CFG expressions.
- `CFGDriftFiniteSetup.generated_eq_cfgTarget` is the Objective-6 core theorem:
  zero drift identifying the effective negative density with the conditional
  density implies the generated finite affine density is the paper's CFG affine
  target, not necessarily the conditional density itself.
- `CFGTargetNonnegative` and `cfgTargetProbabilityVector` are the explicit gate
  for recovering an ordinary finite probability vector from the affine target.

Remaining Objective-6 work is concrete/probabilistic: determine when the CFG
affine target is nonnegative for useful model classes and guidance scales,
instantiate stability constants for those classes, and only then convert the
affine-density result back to genuine probability measures.

### Objective 7: evaluate whether the conditions are useful in practice — first numerical evaluation complete

After obtaining computable conditioning criteria:

- measure the interaction matrix singular values on realistic features;
- test sensitivity to support geometry, basis size, bandwidth, and probes;
- compare population and minibatch drift numerically;
- evaluate whether enforcing the conditions harms expressivity, optimization,
  FID/IS, or compute requirements.

**First numerical evaluation (2026-07-07, `numerics/`).**  A Python suite
(`numerics/driftlab.py`, `numerics/run_all.py`; report `numerics/RESULTS.md`)
transcribes the certified formulas (crosswalk in `numerics/README.md`) and
evaluates them at the paper's actual operating point (Table 8 / A.6: kernel
`exp(-dist/tau)` on normalized distances with mean 1, `tau ∈ {0.02, 0.05,
0.2}`, per-class batch `N = 64`, eye-masked reused negatives, CFG
`alpha ∈ [1,4]`).  Nothing in the suite is a proof; it prices the proofs.
Findings:

1. **Transcription audit.**  A verbatim port of the paper's Algorithm-2
   pseudo-code and a literal port of the Lean `finiteSoftmax` pipeline agree to
   `1e-16`; the proved identities (matched-batch zero, mass-product centroid
   form, drift bound, frame-certificate inequality over 20k random vectors)
   all check numerically.
2. **Frame certificates are the binding constraint, as the ceiling theorem
   predicted.**  The sharp two-atom constant peaks at `sqrt(2/e) ≈ 0.858` at
   separation `sqrt(2)`; the general-`m` inverse-matrix certificate collapses
   double-exponentially (`~1e-32` at `m = 5`, float64 zero by `m = 8`).
   Midpoint (pairwise-optimal) probes buy 10–20 orders of magnitude over
   integer probes and still collapse: probe design matters enormously but
   cannot beat the Vandermonde decay.  Certified identifiability is realistic
   for small bases at near-optimal separation, not for large `m`.
3. **The estimator provably and measurably targets the modified field.**
   Monte-Carlo MSE against the column-reweighted field decays at the proved
   `1/N` rate, while against the bare mean-shift field a bias floor of exactly
   `‖modified − bare‖²` remains — numerical confirmation of the central
   structural discovery of Objective 4.
4. **The masked-route design is validated.**  The eye-mask leave-out bias is
   at most `O(1/N)` (empirically better), so the indexed SNIS bias term is
   subdominant; the masked-vs-deleted gap `exp(-1e6/tau_tilde)` is
   `~10^{-135000}` or smaller at every paper configuration — bit-identical in
   float64 — so the deleted-estimator theorems carry all statistical content.
5. **What each temperature sees.**  Under a normalized-distance feature model
   at `N = 64`, `tau = 0.02` operates in the nearest-neighbor regime
   (softmax ESS ~ 1–3), `tau = 0.2` averages broadly (ESS ~ 30–60), `tau =
   0.05` sits at the transition; the cross-mode identifiability signal is
   carried almost entirely by the largest temperature.  This matches the
   paper's own multi-temperature ablation and gives the SNIS constants their
   practical meaning.
6. **Conditioning ledger (the headline).**  For the certified two-atom class,
   recovering `‖a-b‖₁ ≤ 0.1` at 90% confidence originally needed, per the
   deterministic-floor certified chain, `N ≈ 8.6e8` samples at `tau = 0.2`
   and astronomically more at smaller `tau`.  The new denominator-tail theorem
   cuts this to `N ≈ 1.1e6--1.3e6` across the paper's temperatures by replacing
   `dmin = N·wmin` with checkable denominator mean/variance data and an
   optimized lower-tail split.  The LLN-typical benchmark is
   `N ≈ 2.6e5--3.0e5`, and observed Monte-Carlo extrapolation is `N ≈ 2e5`.
   The paper trains at `N = 64`.  Consequences: (a) training-signal scale and
   certification scale are still different regimes; (b) the previous dominant
   proof slack has been repaired to within an order of magnitude of observed
   scaling; (c) remaining improvement is now constant-level sharpening
   (e.g. Bernstein-style tails, less crude union bounds), not an exponential
   denominator-floor pathology.
7. **CFG gate.**  At the paper's strongest guidance `alpha = 4`, the CFG
   affine target is a genuine probability vector on only a `(1/alpha)^{m-1}`
   sliver of the simplex under the chosen uniform Dirichlet prior (validated
   against closed form): Objective 6's signed/affine treatment is typical in
   that numerical model, not a claim about real ImageNet conditionals.

Remaining Objective-7 work: run `numerics/real_feature_diagnostics.py` on
*real* encoder features.  The runner is implemented and expects externally
supplied `.npy`/`.npz` feature arrays, but this workspace currently contains no
paper encoder checkpoints or real feature tensors.  Once those are supplied,
the report will measure bare and column-reweighted interaction-matrix
conditioning, finite dual-certificate constants, and softmax ESS in the actual
feature geometry.  The FID/IS-level question of whether enforcing certified
designs harms generation quality still requires training runs.

## Extension track: Sinkhorn-balanced drifting (beyond the paper)

**Not a paper claim.**  `SinkhornImplementation/` (plan, experiments) and
`DriftingIdentifiability/SinkhornBalanced.lean` (certified theory) develop a
*proposed modification* of Algorithm 2, motivated by this repo's results: the
paper's affinity `A = sqrt(A_row·A_col)` is one geometric-mean Sinkhorn
balancing step of the kernel matrix; the extension treats balancing depth `t`
as a design dimension (`t = 1` is the paper; `t → ∞` an entropic-OT coupling).

Certified (2026-07-07, no new axioms):

- `interactionFrameBound_of_probeScaling` and
  `interactionFrameBound_of_biScaling` (axiom-free): certified frames survive
  positive per-probe and per-pair rescalings with explicit constants — the
  exact shape `u(x)²·v(zᵢ)v(zⱼ)` produced by any diagonal kernel rescaling
  `u(x)k(x,y)v(y)`.
- `sinkhornOrbit01Setup` / `sinkhornOrbit01_identifies_of_probeEnergy_eq_zero`:
  the two-atom class is certified for **every** positive rescaling at once —
  the whole Sinkhorn orbit costs no identifiability.  The exact rescaling
  identity `inducedInteractionVector_sinkhornOrbit01_eq` is axiom-free.
- `oneStepBalanced01Setup`: the explicit one-full-step kernel
  `k/sqrt(r(x)·g(y))`.
- `BalancedSampling.lean` closes the first batch-dependence theorem for the
  extension: `balancedTwoStepCentroid_deviation_prob_le` proves a
  high-probability deviation bound for the realized `t = 2` balanced centroid.
  The proof decomposes the random scaling gap into two-sided row-mass
  concentration (`weightSum_deviation_prob_le`), deterministic relative
  centroid perturbation (`selfNormalizedCentroid_relative_perturbation`), and
  fixed-weight SNIS deviation.  It uses only the reviewed
  `sampleMean_meanSquare_le` statistical axiom; the perturbation algebra is
  axiom-free.
- `balancedTwoStepNormalizedDrift_deviation_prob_le_of_centroids` supplies the
  two-branch normalized drift assembly: positive and negative centroid
  high-probability bounds imply a high-probability bound for their difference.
  This closes the non-matrix normalized `t = 2` statistical bridge.
- `twoStepBalancedMatrixAffinity_eq_commonScale_mul_weight` and
  `twoStepBalancedMatrixCentroid_eq_weightCentroid` close the optional B4
  reconciliation: the literal two-step finite matrix implementation is a
  per-row common scaling of the weight-form estimator, and that common scaling
  cancels exactly in the centroid.
- `threeStepWeight_rel_of_rowMass_rel` and
  `balancedThreeStepCentroid_deviation_prob_le_of_mass_tails` implement the
  fixed `t = 3` unrolling core: if both raw row masses and first-balanced row
  masses obey explicit tail bounds around reference profiles, then the
  realized `t = 3` centroid has a high-probability bridge to the
  fixed-reference `t = 3` field.  The associated
  `balancedThreeStepNormalizedDrift_deviation_prob_le_of_centroids` gives the
  two-branch normalized drift assembly.  The primitive iid proof of the
  first-balanced row-mass tails remains open and is now isolated.

Toy-scale evidence (`SinkhornImplementation/RESULTS.md`, seed-deterministic):
one balancing step reproduces the paper's estimator to `1e-16`; mass CV falls
from 0.94 (paper, `t = 1`, `tau = 0.2`) to 0.03 (`t = 10`), making the SNIS
denominators deterministic — removing the dominant certified-chain slack
measured in `numerics/` E5; particle descent on an unequal-mass 2-D target
(the paper's Figure-3 methodology) shows `t = 3` beating `t = 1` on mode-mass
error in every initialization; signal-normalized dispersion favors moderate
depth (`t = 2–3`) and degrades by `t = 10` at small `tau`.  The S6 guardrail
check validates the `BalancedSampling.lean` perturbation constants on the
two-atom testbed.  Open: primitive concentration for the first-balanced
row-mass tails used by the conditional `t = 3` theorem, deeper `t ≥ 4`
unrolling, and any at-scale (FID) claim.  The `t = 2` full-matrix
reconciliation is certified.

### Gain-scheduling sub-track: separating signal from mass-product gain

**Not a paper claim.**  The exact structural
identity `algorithm2Drift_eq_massProduct_centroidDiff`
(`DriftingIdentifiability/Algorithm2Estimator.lean`) factors the paper's
drift as `(P·Q) · (C⁺ − C⁻)`: a raw affinity-mass GAIN times a
self-normalized-centroid SIGNAL.  Every identifiability theorem in this repo
factors through `C⁺ = C⁻`; the gain is identifiability-inert and provably
collapses exponentially off-support (`algorithm2Drift_norm_le_affinityMass`).
`SinkhornImplementation/sinkhorn_drift.py` now exposes a `gain=` parameter on
`compute_v_sinkhorn` (default `"paper"`, bit-identical to the original code
path — S0-regression-checked) that replaces `P·Q` with alternatives
(`power`, `min`, `const`, `cert`), every one a strictly positive per-query
rescaling of the same signal — exactly the class certified by
`interactionFrameBound_of_probeScaling`.  The named corollary
`interactionFrameBound_of_positiveGain` makes this explicit: from pointwise
positivity of the gain (what every `gain_schedule` mode guarantees) it
derives the positive finite-min lower bound and concludes a certified
positive frame constant for the gain-scheduled field, so no gain mode costs
identifiability.  Axiom-free (`#print axioms`: only the three foundational
axioms); registered in the audit (160 promoted declarations); Check.ps1
green.

S7 (`SinkhornImplementation/RESULTS.md`) verifies this empirically and
corrects an over-optimistic prediction from the design proposal
(`SinkhornImplementation/PROPOSAL_CERTIFIED_GAIN.md`): the alternative gains
DO unfreeze the paper's exponential collapse — traced directly, the median
gain rises from `~1e-6` to `~0.1–0.2` at the far initialization, and the
particle swarm's mean position genuinely travels almost the entire distance
from its start to the target mode within 300 steps — but this does not
translate into dramatic mode-mass-error recovery, because the far/collapsed
initializations start nearly homogeneous: every particle computes almost
the same local drift, so the whole swarm moves as one body toward its
nearest mode instead of splitting into the correct proportions.  This is a
separate limitation of swarm homogeneity, not gain starvation, and a
per-particle gain rescaling cannot fix it alone.  Where mode-mass error does
improve, the plainest alternative (`const`, a fixed reference gain with no
adaptivity) is the most consistently strong performer — sophistication (the
finite-sample-certificate gain) did not beat simplicity at this toy scale.
Open: testing whether injecting per-particle diversity (repulsion or
per-step noise) breaks the homogeneity and lets the gain fix compound with
it; any at-scale claim.

## What would complete the practical phase

The practical objective should be considered complete only when the project has
an implementable model/kernel/probe design, a useful certified or empirically
validated conditioning constant, a finite-sample bridge to the training
estimator, and evidence that the restrictions do not destroy model quality.

## Raw Gaussian general converse

`GaussianScoreRecovery.lean` and `GaussianConvolutionInjectivity.lean` now
prove the authors' reviewer-rebuttal argument 1 axiom-free. The promoted theorem
`gaussianMeanShiftDrift_identifiesAtZero` states that pointwise zero raw
Gaussian mean-shift drift identifies arbitrary probability measures on a
finite-dimensional complete real inner-product Borel space.

The proof establishes positivity and differentiability of the Gaussian
normalizer, proves the Fréchet score identity
`D log Zₚ = σ⁻²⟪meanShiftₚ,·⟫`, obtains proportional normalizers from equal
mean-shift maps, fixes the scalar using probability mass, and applies
Fourier/characteristic-function injectivity. Neither former raw-converse axiom
remains in `Paperaxioms.lean`.

Scope remains exact pointwise `V ≡ 0`; asymptotic `V → 0` and general
non-Gaussian kernels remain separate problems. Full proof record:
`RawFieldConverse.md`.

## Laplacian-kernel converse for Gaussian targets

`LaplacianGaussianConverse.lean` now proves the authors' reviewer-rebuttal
argument 2 axiom-free, for the paper's exact Laplace kernel
`exp (-‖x-y‖/τ)`. The promoted theorem
`laplaceGaussianMeanShiftDrift_identifiesAtZero` states that pointwise zero
raw Laplace-kernel mean-shift drift identifies two multivariate Gaussian laws
(arbitrary means, arbitrary positive-semidefinite — including degenerate —
covariances, any finite dimension).

Mechanism, fully internal: along radial probes `r • u` the compensated kernel
weight converges to an exponential tilt (rationalized radial identity +
dominated convergence with a sharp `exp (‖y‖/τ)` dominator, discharged for
Gaussians by Fernique); the Gaussian tilt centroid is computed exactly as
`μ + τ⁻¹ S u` via mathlib's scalar Gaussian MGF and tilted-measure API; so the
drift's radial limit is `(μp - μq) + τ⁻¹ (Sp - Sq) u`
(`multivariateGaussian_laplaceMeanShiftDrift_radial_tendsto`). Zero drift
forces every radial limit to vanish; directions `u` and `-u` recover the
means, positive scaling recovers the covariance action, and the matrix
transfer map is injective (`gaussianRadialLimit_zero_imp_parameters_eq`).

The candidate discipline is complete: `BothMultivariateGaussian` is the pair
condition, `laplaceGaussianCandidate` the registered `CandidateSpec`, with
`laplaceGaussianCandidate_identifiesAtZero` and (for every nonempty dimension)
`laplaceGaussianCandidate_isLegitimate` — the distinct pair is two
unit-covariance Gaussians with different means, witnessed by the exact first
moment, before any zero-drift assumption. `#print axioms` on all headline
declarations reports only Lean foundations — no paper axiom is used anywhere
in this route.

Scope: Gaussian-family targets only (not arbitrary distributions), exact
pointwise `V ≡ 0` only, ideal population field only; zero drift at finitely
many probes would not feed the radial argument, and `τ ≤ 0` is excluded. Plan
and proof record: `LaplacianGaussianConverse.md`. Both reviewer-rebuttal
converses are therefore now machine-checked; the rebuttal's own "general
converse for arbitrary fields" concession is answered for the Gaussian kernel
(arbitrary targets) and for the Laplace kernel (Gaussian targets).  The
Laplace-kernel arbitrary-target converse has also advanced beyond the
rebuttal: finite mixtures, nowhere-dense/right-dense-gap supports, and several
Wronskian gates are machine-checked below.  The continuous-density a.c. case
is now Lean-certified **unrestricted** (2026-07-12): zero drift + continuous
densities + two-sided exponential first moments + a `p` first moment forces
`p = q` with no hypothesis on the mean-shift zero set
(`laplaceAC_identifies_of_continuousDensity`), and the condition is
demonstrably legitimate via a certificate-free Gaussian witness.

## Laplacian-kernel arbitrary-target structural reduction

`LaplaceCompanion.lean` now proves the first structural reduction for the
paper's exact Laplace kernel with arbitrary probability targets. The full
arbitrary-target converse is **not** claimed. What is promoted is the
companion-kernel score identity:

```text
ℓτ(x,y) := (τ + ‖x-y‖) exp(-‖x-y‖/τ),
∇ₓ ∫ ℓτ(x,y) dp(y) = τ⁻¹ ∫ exp(-‖x-y‖/τ) (y-x) dp(y).
```

This is the Laplace analogue of the Gaussian score identity, except the
gradient potential and the normalizer are different smoothings. The module also
proves that the raw Laplace mean-shift field is globally well-defined for
arbitrary probability measures, with no moment hypotheses, because the
displacement integrand is uniformly bounded. Zero raw Laplace drift is
therefore reformulated as the exact cross-gradient equation

```text
Z_q(x) ∇L_p(x) = Z_p(x) ∇L_q(x)
```

and, under the critical exponential-moment hypothesis, bridges to equality of
exponential-tilt centroids in every radial direction.

Verification status: the new declarations are imported by the root module and
registered in `scripts/AxiomAudit.ps1`; `lake build --wfail`,
`scripts/Check.ps1`, and explicit `#print axioms` checks pass with only Lean
foundations (`propext`, `Classical.choice`, `Quot.sound`).

`LaplaceInjectivity.lean` then completes the real-line Stage-2 refinement:
the two-sided exponential profile has an explicitly computed nonzero Fourier
transform, yielding `laplaceKernelNormalizer_injective` for finite measures on
`ℝ`. The file registers the reusable predicate `LaplaceSmoothingInjective`,
proves `laplaceSmoothingInjective_real`, and proves the one-degenerate-side
converse `laplaceZeroDrift_dirac_identifies_real`: if a probability law on
`ℝ` has finite first moment and zero raw Laplace mean-shift drift against
`dirac c`, then it is exactly `dirac c`. This is a genuine arbitrary-target
Laplace result in the degenerate-opponent case, not the full arbitrary-pair
converse.

`LaplaceWronskian.lean` (Stage 3, 2026-07-10) makes the 1-d elliptic
structure classical and proves the **alignment reduction** of the open
converse. The displacement integrand `x ↦ kτ(x,y)(y-x)` is differentiable
EVERYWHERE — the `sgn` singularity of `∂ₓkτ` is killed by the vanishing
factor — with uniformly bounded derivative `(‖x-y‖/τ - 1)kτ`, so the
mean-shift numerator satisfies the pointwise classical ODE

```text
D_p′ = τ⁻¹ L_p - 2 Z_p     (equivalently  τ² L_p″ = L_p - 2τ Z_p),
```

with no distribution theory anywhere
(`hasDerivAt_laplaceDisplacementIntegral`). The headline
`laplaceZeroDrift_imp_eq_of_companionAligned` then proves: zero raw Laplace
drift between ARBITRARY probability measures on `ℝ` together with the
companion-alignment identity `L_p·Z_q = L_q·Z_p` forces `p = q` — zero drift
plus alignment kills the Wronskian `L_p′L_q - L_pL_q′`, so `L_p = c·L_q`;
alignment converts this to `Z_p = c·Z_q`; smoothing injectivity against the
scaled measure `c•q` and total mass force `c = 1`. **The open 1-d
arbitrary-target converse is thereby reduced to the single scalar identity
`K := L_p·Z_q - L_q·Z_p ≡ 0`.** All declarations are axiom-free
(foundations only), imported by the root, and registered in the promoted
axiom audit.

`LaplaceAtomicConverse.lean` (Stage 3b, 2026-07-10) **resolves the open
arbitrary-target converse on the finite-mixture class**: for the paper's
Laplace kernel and every positive bandwidth, pointwise zero raw mean-shift
drift between ANY two finitely-supported probability measures on the line
forces the measures to be equal (`laplaceZeroDrift_atomic_identifies`) —
arbitrary atoms, arbitrary support size, no frame conditions, probes,
moments, or bandwidth restrictions. This is the first arbitrary-PAIR
converse for the practical kernel, on exactly the finite representation
class of the paper's own Appendix C, and it answers the authors' open
"general converse" concession on that dense class. Mechanism (the
moment-parallelism argument): between consecutive atoms the zero-drift
bilinear identity is a quadratic in `exp(x/τ)²` with constant coefficients,
so a polynomial-vanishing argument makes the truncated antisymmetric
tilted-moment pairing `𝔞ₖ` vanish at every truncation; a strictly-signed-sum
argument matches the bottom atoms and a telescoping induction forces the
weight vectors proportional, with normalization pinning the constant to
one. Axiom-free (`#print axioms`: Lean foundations only), imported by the
root, registered in the promoted axiom audit; every step of the paper proof
was verified numerically to `1e-17` before formalization.

`LaplaceGeneralConverseEndgame.lean` and
`LaplaceGeneralConverseNowhereDense.lean` push this beyond finite mixtures.
The Milestone-2 endgame proves that vanishing of the lower truncated pairing
`𝔞(x) := truncatedPairing τ p q x` for all `x` already forces `p = q` for
arbitrary probability measures on `ℝ`
(`laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero`).  Milestone 3 then
proves a concrete nowhere-dense-support upgrade:
if zero raw Laplace drift holds and the combined measure has right-dense
zero-mass gaps,
`∀ x ε>0, ∃ u<v, x<u ∧ v<x+ε ∧ (p+q)(Ioo u v)=0`, then `p = q`
(`laplaceZeroDrift_identifies_of_rightDense_zeroMassGaps`, alias
`laplaceZeroDrift_identifies_of_nowhereDense_support`).  Mechanism:
on every zero-mass gap the three one-sided coefficients in the Laplace
decomposition are constant; zero drift becomes a quadratic in
`exp(x/τ)^2` on an infinite interval; polynomial vanishing forces
`𝔞=0` on the gap, and right-continuity plus right-dense gaps make
`𝔞≡0` globally.  This covers countable atomic supports with accumulation and
Cantor-type singular supports under the explicit right-dense-gap hypothesis.
The result is root-imported, promoted, and axiom-audited.

Milestone 4 is now closed in `LaplaceGeneralConverseBalance.lean`.  The file
defines the scaled coefficients
`scaledLowerPairing = exp(-2x/τ)𝔞(x)`,
`scaledUpperPairing = exp(2x/τ)𝔠(x)`, and their balance defect, proves the
needed right-continuity infrastructure, and certifies the mathematically
correct one-sided derivative route.  The raw normalizer right-derivative socket has
been proved as `hasDerivWithinAt_Ici_laplaceKernelNormalizer`: for `t ≥ x`,
the normalizer is exactly a fixed-tail exponential part plus the shrinking
strip remainder `laplaceNormalizerRightRemainder`; the fixed-tail part
differentiates by ordinary exponential rules and the strip remainder has zero
right derivative by dominated convergence.  This gives the unconditional
cross-displacement derivative theorem
`hasDerivWithinAt_Ici_laplaceCrossDisplacement` and the headline balance
identity `laplaceBalance_identity_of_zeroDrift`, valid for arbitrary
probability measures on `ℝ`.  No new axiom, `sorry`, or distribution-theory
import was introduced.

Milestone 5 has a new concrete Wronskian coordinate in
`LaplaceGeneralConverseWronskian.lean`.  The raw-normalizer Wronskian
`laplaceKernelNormalizerWronskian` is now a named function and is proved equal
to the determinant of upper/lower exponential masses:

```text
W(x) = (2/τ) * (P⁺(x) Q⁻(x) - P⁻(x) Q⁺(x)).
```

The promoted facts now include the determinant formula, a zero-Wronskian gate
`W ≡ 0 -> p = q`, constancy of `W` across zero-mass gaps, and vanishing of
`W` on one-sided tails under explicit support-side hypotheses.  This does not
close the full arbitrary-support converse yet; it converts the remaining
problem into a sharper scalar determinant-propagation question: prove from
zero raw Laplace drift that this mass determinant vanishes at every cut.
The first local propagation theorem for that determinant is now also certified:
on a zero-mass gap, zero drift forces `W = 0` at any cut where the `p`-side
lower/upper coordinates are connected
(`laplaceKernelNormalizerWronskian_eq_zero_on_gap_of_p_connection`; ergonomic
strictly two-sided version:
`laplaceKernelNormalizerWronskian_eq_zero_on_gap_of_p_twoSided`).  The next
Milestone-5 subproblem is to remove this nondegeneracy by formalizing the
one-sided support/zero-coordinate cases.

The a.c. route now has a **Lean-native finite-simple-zero theorem surface**.
`LaplaceACAbel.lean`, `LaplaceACRegularity.lean`,
`LaplaceACDensityRegularity.lean`, `LaplaceACAsymptotics.lean`,
`LaplaceACPropagation.lean`, and `LaplaceACFinal.lean` assemble the full
Wronskian route under explicit continuous-density, exponential-moment, and
finite alternating simple-zero hypotheses.  The user-facing theorem

```text
laplaceAC_identifies_of_continuousDensity_finiteSimpleZeros
```

states that for one-dimensional probability laws with continuous Lebesgue
densities, the required two-sided exponential moments, and a finite alternating
simple sign-changing zero list for the `p` mean-shift ratio, zero raw Laplace
mean-shift drift forces `p = q`.  No primitive functions, local `δ/L`
witnesses, exposed C²-normalizer certificates, or leftover `HasDerivAt`
hypotheses remain in the public statement; those are generated internally.

This theorem is also now demonstrably non-vacuous.  `LaplaceACFinal.lean`
contains the Gaussian density/moment infrastructure, and
`LaplaceACGaussianCertificate.lean` constructs the standard-Gaussian
single-downward-crossing certificate axiom-free for every positive bandwidth.
Consequently the public witness

```text
standardGaussian_vs_shiftedGaussian_finiteSimpleZeros
```

and the distinct-pair / legitimacy results

```text
laplaceACFiniteSimpleZerosCondition_allowsDistinctPair_of_standardGaussian
laplaceACFiniteSimpleZerosCondition_isLegitimate_of_standardGaussian
```

are hypothesis-free apart from `ValidBandwidth τ`; `#print axioms` reports only
Lean foundations (`propext`, `Classical.choice`, `Quot.sound`).

**2026-07-12 (Fable): the unrestricted continuous-density theorem is now
Lean-certified — Milestone 5 closed for this class.**  The anticipated need
for distributional identities and Levinson/ODE-asymptotic theory did not
materialize.  `LaplaceACFinal.lean` proves

```text
laplaceAC_identifies_of_continuousDensity
```

— continuous Lebesgue densities + two-sided exponential first moments + a `p`
first moment + zero raw Laplace drift imply `p = q`, with **no hypothesis on
the zero set of the mean-shift ratio** (no finiteness, simplicity,
alternation, or sign pattern).  The proof is a pointwise trichotomy: at any
point where `m ≠ 0`, either an outer-ray or an arbitrary-edge blow-up kills
the Wronskian (the edge needs only the one-sided slope limit `m' ≥ 0`, giving
`μ' ≥ 1` for free — this is what replaced the Levinson-type analysis); where
`m ≡ 0` locally, the certified first-order elliptic pair forces both
normalizer derivatives to vanish; the rest is closure/continuity.  The
condition form is `IsLegitimateCondition`
(`laplaceACContinuousDensityCondition_isLegitimate`) via a Gaussian witness
that needs no sign certificate at all.  `#print axioms` reports only Lean
foundations for the entire chain; the finite-simple-zeros theorem is now a
corollary surface.

Milestone 6 (assembly) is also closed: `LaplaceRealConverse.lean` records the
single umbrella theorem `laplaceZeroDrift_identifies_real` — zero raw Laplace
drift identifies probability measures satisfying
`LaplaceRealConverseCondition` = (right-dense zero-mass gaps, the
nowhere-dense/atomic/singular class) OR (unrestricted continuous densities
with exponential moments) — together with its legitimacy witness.  This is
the honest one-theorem statement of the 1-d Laplace converse as currently
machine-checked.

**2026-07-13 (Fable): the ATOMLESS converse is Lean-certified — rough
densities and singular-continuous laws cleared.**  `LaplaceAtomlessConverse.lean`
proves

```text
laplaceZeroDrift_identifies_of_noAtoms
```

— zero raw Laplace drift + `NoAtoms p` + `NoAtoms q` + a `p` first moment
imply `p = q`, with **no density regularity and no exponential moments**.
The route avoids the anticipated a.e.-Abel machinery entirely: the companion
alignment defect `K = L_p·Z_q − L_q·Z_p` has a purely first-order zero-drift
structure (`K = τ·m·W`, `K' = −τ(m'+2)·W`, both exact and pointwise for
atomless laws, with `m'+2 ≥ 1` globally from L3), so the same L6/L8
propagation closes `K ≡ 0` and the certified companion-alignment gate gives
`p = q`.  The condition `LaplaceAtomlessCondition` is bandwidth-free,
`IsLegitimateCondition`, and subsumes the continuous-density condition;
the Milestone-6 umbrella `LaplaceRealConverseCondition` is now
(right-dense zero-mass gaps) ∨ (atomless + `p` first moment), also
bandwidth-free.

Remaining research objectives are now narrower and split by trust level:

- Lean-native, general-measure track: remove the nondegeneracy from the
  gap-local `W = 0` theorem by formalizing the one-sided support/zero-coordinate
  cases.
- Conditional-axiom track: `LaplaceACConditionalAxiomPlan.md` is retained only
  as a historical FALLBACK; the atomless theorem is proved axiom-free, so no
  conditional axioms are needed for this class.
- General-measure track: measures with atoms on interval supports (the last
  structural gap between the two umbrella regimes), and dropping the `p`
  first moment from the atomless theorem.
- Higher-dimensional track: Laplace smoothing injectivity/Dirac rigidity and
  any multidimensional analogue of the ODE argument.

**2026-07-13 (Fable): THE ARBITRARY-TARGET CONJECTURE IS CLOSED — Frontier A
machine-checked, axiom-free.**  `LaplaceUnconditionalConverse.lean` proves
`laplaceZeroDrift_identifies`: zero raw Laplace mean-shift drift identifies
ARBITRARY probability measures on ℝ, with no hypotheses beyond
probability — no atomlessness, no moment, no density.  The one-sided K-route
planned in `LaplaceEndgame.md` executed as designed: the companion-alignment
gate is unconditional (whole conjecture ⟺ `K ≡ 0`); the abstract L6/L8
propagation layer was already right-derivative-based; the K-identities
(`K = τ·m·W⁺`, `K′⁺ = −τ(m′⁺+2)·W⁺`) hold for all measures because
`laplaceMeanShiftRatioDeriv`/`laplaceKernelNormalizerWronskian` are already
defined through the certified right-derivative coefficient; the moment
dropped out via `laplaceTiltedMeanRightDerivCoeff ≥ 0` plus one-sided
uniqueness; `m` is two-sided differentiable at its zeros (where `D = 0`), so
the existing edge linear-bound helper was reused verbatim; and the càdlàg
Abel coefficient is interval-integrable via `|Z'⁺| ≤ 1/τ` plus compactness,
feeding the standard right-FTC.  `Check.ps1` green (58 files, 374 promoted
declarations, 15 paper + 5 conditional axioms unchanged).  The Milestone-6
umbrella, the atomless theorem (`laplaceZeroDrift_identifies_of_noAtoms`), and
the continuous-density theorem are now strict corollaries; the previously-open
"atoms on interval supports" class and the `p`-first-moment hypothesis are both
eliminated.  The unconditional condition `LaplaceUnconditionalCondition`
(= a pair of probability measures) is `IsLegitimateCondition` and
bandwidth-free.  **Frontier C is also done** — but it was already closed in the
repo before this session: `gaussianMeanShiftDrift_identifiesAtZero`
(`GaussianScoreRecovery.lean`) proves the raw Gaussian mean-shift converse for
arbitrary probability measures in ANY finite dimension, axiom-free, via the
score identity `∇log Z = σ⁻²·meanShift`; `GaussianArbitraryConverse.lean` adds
the direct-`p=q` headline `gaussianZeroDrift_identifies` and
`bothProbability_isLegitimate` for parity with the Laplace result.  So both
canonical kernels now have unconditional arbitrary-target converses (Laplace in
1-d, Gaussian in all finite dimensions).  Remaining objectives: higher-
dimensional Laplace / Matérn-class kernels (no 1-d ODE structure — genuinely
open).  Milestones, derivations, and Lean gotchas: `LaplaceEndgame.md`.

**2026-07-14 (Fable): Frontier D research pass — higher-dimensional Laplace —
recorded in `LaplaceHigherDim.md` (no Lean written; findings + implementation
plan).**  The n-d Laplace kernel bifurcates by norm, since the repo's
`laplaceKernel` is norm-generic.  (1) The **ℓ¹/product** case (`laplaceKernel`
on `PiLp 1 (fun _ : ι => ℝ)`; sklearn's `laplacian_kernel`) **reduces entirely
to the 1-d unconditional theorem**: coordinate `i` of the n-d drift at probe
`x` is exactly the 1-d Laplace drift of an exponentially tilted slice measure,
so `laplaceZeroDrift_identifies` fires slice-by-slice (its unconditionality is
load-bearing — tilts inherit atoms/tails), forcing `Z_p = c·Z_q` with `c`
constant, `c = 1` by iterated 1-d Tonelli, and `p = q` by a `Finset`-induction
on rectangle coordinates powered by `laplaceKernelNormalizer_injective`.  Full
paper proof + staged Lean plan (est. 600–900 lines) in the doc, formulated as a
**tensorization meta-theorem** (any product kernel whose 1-d factors have an
unconditional converse + smoothing injectivity — also yields anisotropic
bandwidths and mixed Laplace/Gaussian tensor kernels).  (2) The **ℓ²/radial**
case (Matérn-1/2) is research-grade open; the doc derives the displacement
potential `D_p = ∇ψ_p` (`ψ_p` = Matérn-3/2 smoothing, elementary radial
identity), the Matérn-universal companion PDE `(1−τ²Δ)D = (n+1)τ²∇Z` (checked
against the proven 1-d companion identities), the vector Abel system
`2(Dm)W + (div W)m = −(n+1)W`, and a regularity result (no atomic deltas in
`ΔZ` for n ≥ 2 — cleaner than 1-d); staged program D2.a–D2.d (far-field
foundation largely pre-built in `LaplacianGaussianConverse.lean`; radial
measures reduce to a scalar Abel ODE; finite-support via exposed-point
peeling; general endgame paper-first).  **Paper determination (same day):**
the referenced paper's Eq. (12) explicitly takes ∥·∥ to be the **ℓ2-distance**
(verified in `papers/2602.04770v2.pdf`, restated in its appendix, `cdist` in
the pseudocode), so the paper-faithful n-d target is the ℓ²/radial case —
`laplaceKernel` on `EuclideanSpace ℝ ι` is exactly the paper's kernel — and
the sequencing in `LaplaceHigherDim.md` §5 now leads with the ℓ² program
(D2.a far-field → D2.b radial measures → D2.c/D2.d), keeping the ℓ¹
tensorization theorem as a derisked adjacent result.  **A second, directed ℓ²
pass (same day) added `LaplaceHigherDim.md` §4.6–§4.8**: new findings —
monotone compensated weights (moment-minimal far field), a no-concentration
principle (fixed bandwidth ⟹ no probe limit isolates atoms; exposed-point
peeling invalidated and D2.c corrected to singularity matching), **ball-average
atom alignment** (`q({a})·D_p(a) = p({a})·D_q(a)` for every point, arbitrary
measures, n ≥ 2 — a provable-now n-d theorem, milestone L3), the
difference-potential equation `(n+1)τ²∇Φ = m(Φ − τ²ΔΦ)` with the maximum
principle pinning extrema of `Φ` to `{m = 0}`, the n-d L3 conjecture (Laplace
centroid-map monotonicity — numerics first), classical-calculus feasibility
for radial measures (no distribution theory needed anywhere), and
subordination-based n-d smoothing injectivity (no closed-form transform
needed).  Plus a lemma-level implementation plan, milestones L0–L5
(foundations → far field/dimension reduction → atom alignment → injectivity →
radial-measure converse), with Mathlib support verified
(`hasFDerivAt_integral_of_dominated_loc_of_lip`, Gaussian Fourier transform,
ball-average API, generic charFun cancellation).  **A third pass (same day)
deep-dived L5 (`LaplaceHigherDim.md` §4.9)**: exact cylindrical derivative
formulas (`m̃'+1 = Cov_w(X, X/d)/τ`, reducing at n = 1 to the proven 1-d L3);
the tangential IBP identity `∫(ρ²/d)w dμ = ((n−1)τ/r)∫t·w dμ` for radial μ
(per-shell 1-d integration by parts — eliminates the Laplacian layer from L5
entirely); the substitution `v = r^{n−1}w` making the radial system exactly
1-d-shaped (`K = τm̃v`, `K' = −τ(m̃'+n+1)v`, slotting into the abstract
LaplaceACPropagation abel layer with `μDeriv_rad = m̃' + (n−1)`, whose four
consumed inputs were recon-verified at LaplaceUnconditionalConverse.lean:882–
1010); boundary conditions resolved (`v(0⁺) = 0` unconditionally even with
origin atoms; ray at ∞ under mild `(n−1)/2`-moments, subsequence suffices);
the needed sign condition reduced to the radial slack inequality
`E_w[X²/d] + (n−1)τ ≥ m̃·E_w[X/d]`, **proved for `m̃ ≤ 0` (Cauchy–Schwarz) and
free at zeros of `m̃`; the `m̃ > 0` case is the single remaining open point**
(numerics spec written); statement design via shell mixtures
(`Measure.bind` of a radial profile against `uniformShell`) sidesteps
Haar-orbit disintegration; L5 v1 targets n ≥ 3 (classical two-sided
derivatives), n = 2 as v2 (log-integrable `m̃'` via FTC form).  Lemma-grain
plan in §4.9(R8).

**2026-07-14 (Fable): milestones L0 and L1 IMPLEMENTED, machine-checked,
axiom-free, `--wfail`-clean.**  `LaplaceRadialFoundations.lean` (L0) and
`LaplaceRadialFarField.lean` (L1) are wired into the root module; Check.ps1 green
(61 Lean files; axiom audit over the promoted headlines shows only
propext/Classical.choice/Quot.sound).  **L0** proves, for arbitrary finite
measures on any real inner-product space: the Matérn-3/2 profile
`matern32Profile` with derivative `g'(r) = −r e^{−r/τ}` and the uniform
`τ·e⁻¹` gradient bound; `hasFDerivAt_matern32Profile_norm_sub`, the
through-the-diagonal gradient `∇ₓ g(‖x−y‖) = e^{−‖x−y‖/τ}(y−x)` (chain rule via
`√⟪·,·⟫` off-diagonal, a `g'(0)=0` little-o argument on the diagonal); the
displacement field `laplaceDisplacementField` (= D) and potential
`laplaceDisplacementPotential` (= ψ) with `τ/e`-bounded integrability; the
headline `hasFDerivAt_laplaceDisplacementPotential` (**`∇ψ = D`**, via
`hasFDerivAt_integral_of_dominated_of_fderiv_le` with the uniform gradient
dominator — no moment hypothesis); and `zeroDrift_displacementAligned` (**the
(PA) gate `Z_q·D_p = Z_p·D_q`**, proved from `meanShift = Z⁻¹·D` + normalizer
positivity, independent of the FDeriv).  **L1** proves
`laplaceCompensatedWeight_monotone` and the headline **`zeroDrift_tiltedCentroid_eq`**
(zero drift ⟹ exponential-tilt centroids agree in every direction, arbitrary
probability measures with exponential moments — D2.a far-field foundation), via
the existing `kernelCentroid_laplace_radial_tendsto`.  Lean gotchas recorded in
`LaplaceHigherDim.md` §4.8 (innerSL_apply_apply / innerSL_apply_norm /
SecondCountableTopologyEither haveI / omit-before-docstring / two_smul).  Next:
L2 (dimensional reduction + collinear corollary), then L3/L4/L5.

## Conditional research modules

`CharacteristicIdentifiability.lean` and `GaussianNondegeneracy.lean` remain
available only through `DriftingIdentifiability.Conditional`. They depend on
external Gaussian/RKHS axioms or a synthetic interaction construction and are
not accepted project solutions.

`CharacteristicIdentifiability.lean` still contains the conditional MMD/RKHS
and weak-convergence reductions. `GaussianNondegeneracy.lean` still contains a
synthetic interaction construction. These should not be conflated with the now
promoted raw Gaussian mean-shift theorem. The conditional-external class
contains 5 axioms.
