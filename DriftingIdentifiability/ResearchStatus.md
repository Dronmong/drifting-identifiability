# Research status

## Executive answer

The original mathematical existence goal has been achieved: the project gives
explicit, nonvacuous conditions under which zero ideal population mean-shift
drift forces `p = q`, and the result is machine checked without new or
conditional Gaussian/RKHS axioms.

The practical modeling goal is not yet complete, but Objectives 1--3 are now
complete at the deterministic population level. Objective 3 includes a fully
instantiated non-atomic `C∞` bump basis on a Gaussian reference, in addition to
the higher-dimensional, variable-bandwidth, adaptive-probe, perturbative, and
Laplace infrastructure. The next mathematical objective is the finite-sample
bridge to minibatch training; model-quality evaluation remains empirical work.

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

### Objective 2: finite observable population loss — complete

The exact theorem consumes the deterministic finite probe loss
`∑ₙ ‖V(xₙ)‖²`. Zero loss is proved equivalent to zero at every probe, and small
loss controls coefficient error through its square root. The concrete Gaussian
theorems expose both conclusions directly. This route needs neither pointwise
zero on all of data space nor a full-support sampling measure.

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

### Objective 4: prove finite-sample guarantees for Algorithm 2 — bridge and estimator structure complete; consistency open

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

Still open (a single genuine estimator-analysis question, not theorem
infrastructure):

- the **quantitative bias/consistency** of `algorithm2Drift`: its expectation
  against the ideal population mean-shift field as the temperature and sample
  counts vary.  The softmax normalization makes it a self-normalized estimator,
  generally biased at finite sample size; establishing the `→ 0` bias (and hence a
  vanishing MSE that the bridge turns into identifiability) is a real analysis of
  one algorithm.  The structural and boundedness facts above constrain this MSE
  (they give a deterministic `2·Npos·Nneg·R` envelope and the exact
  attraction–repulsion form) but do not compute the limit; that limit cannot be
  axiomatized without assuming the substantive consistency claim.

  An attempt to close this by an elementary *deterministic* reduction — bound the
  estimator error by the sup-norm deviation of the softmax affinities from their
  population values, using the bilinear stability of `driftOfAffinities` — was
  investigated and **fails by a scaling mismatch**: the crude bilinear bound is
  `ε ~ 4·Npos·Nneg·η`, but a single softmax weight is `O(1/N)` with sampling
  fluctuation `η ~ N^{-3/2}`, so `ε ~ √N → ∞`.  The estimator's consistency comes
  from *cancellation across the `N²` self-normalized pairs*, which a sup-norm
  perturbation bound discards.  The correct handle is the mass-scaled centroid form
  `algorithm2Drift_eq_massScaledCentroid` (each centroid a self-normalized average
  with the standard `O(1/N)` importance-sampling bias); the remaining work is a
  genuine self-normalized-IS / ratio-consistency analysis under an explicit iid
  sampling model.  This route and its obstruction are recorded in
  `LoggedFailures.md` (2026-07-06 entry).

### Objective 5: handle feature-space training correctly

The current data-space theorem gives equality of data-space measures. In
feature space, zero drift can generally identify only the pushforward laws.

Desired results:

- retain pushforward equality as the default conclusion;
- lift it to source-law equality only under an independently stated
  measure-determining or measurable-embedding condition;
- study approximate lifting when the learned feature map is not injective.

### Objective 6: treat CFG separately

CFG equation (16) can define a signed affine target rather than a probability
measure. It is outside the current probability-mixture theorem.

Desired result: develop a signed-measure or affine-density formulation with its
own normalization, identifiability, and stability analysis.

### Objective 7: evaluate whether the conditions are useful in practice

After obtaining computable conditioning criteria:

- measure the interaction matrix singular values on realistic features;
- test sensitivity to support geometry, basis size, bandwidth, and probes;
- compare population and minibatch drift numerically;
- evaluate whether enforcing the conditions harms expressivity, optimization,
  FID/IS, or compute requirements.

## What would complete the practical phase

The practical objective should be considered complete only when the project has
an implementable model/kernel/probe design, a useful certified or empirically
validated conditioning constant, a finite-sample bridge to the training
estimator, and evidence that the restrictions do not destroy model quality.

## Conditional research modules

`CharacteristicIdentifiability.lean` and `GaussianNondegeneracy.lean` remain
available only through `DriftingIdentifiability.Conditional`. They depend on
external Gaussian/RKHS axioms or a synthetic interaction construction and are
not accepted project solutions.
