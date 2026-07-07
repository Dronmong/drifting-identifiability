# Logged failures

Read this file before proposing a new condition. Preserve rejected approaches
so future agents do not repeat equivalent mistakes.

## Entry template

### YYYY-MM-DD — short candidate name

- **Exact condition:**
- **Intended mechanism:**
- **Stress test/model:**
- **Counterexample or obstruction:**
- **Why the argument fails:**
- **Fatal or repairable:**
- **Possible repair:**
- **Relevant Lean declarations/files:**

---

### 2026-07-04 — zero minors without normalization

- **Exact condition:** All coefficient minors
  `aᵢ bⱼ - aⱼ bᵢ` vanish, but `a` and `b` are not required to have equal
  total mass.
- **Intended mechanism:** Vanishing minors make the coefficient vectors
  parallel, which was hoped to imply equality.
- **Stress test/model:** `m = 2`, `a = (1,0)`, `b = (2,0)`.
- **Counterexample or obstruction:** Every minor vanishes but `a ≠ b`.
- **Why the argument fails:** Minors determine only a projective direction;
  they cannot recover scale.
- **Fatal or repairable:** Repairable.
- **Possible repair:** Require both coefficient vectors to have the same
  nonzero total mass. Probability normalization (`∑aᵢ = ∑bᵢ = 1`) is the
  natural condition.
- **Relevant Lean declarations/files:** `AllCoefficientMinorsZero`,
  `FiniteProbabilityVector`, `PaperFiniteIdentifiability.lean`.

---

### 2026-07-04 — degenerate interaction vectors or probes

- **Exact condition:** Finite-basis membership and probability normalization,
  without linear independence/separation of the induced vectors `Uᵢⱼ`.
- **Intended mechanism:** Use the equation
  `∑_{i<j}(aᵢbⱼ-aⱼbᵢ)Uᵢⱼ = 0` to recover every coefficient minor.
- **Stress test/model:** Set every `Uᵢⱼ = 0` (equivalently, use a completely
  uninformative interaction/probe system). Choose any distinct probability
  vectors `a,b`.
- **Counterexample or obstruction:** The bilinear drift is zero for every pair
  although `a ≠ b`.
- **Why the argument fails:** A linear combination equal to zero does not force
  its coefficients to vanish when the observable interaction family is
  dependent.
- **Fatal or repairable:** Repairable, but a genuine separation hypothesis is
  necessary.
- **Possible repair:** Require `{Uᵢⱼ | i<j}` to be linearly independent, or
  prove a weaker concrete restricted-injectivity condition from the kernel and
  probes. Merely assuming “the map is injective” without such a reduction is
  disallowed as circular.
- **Relevant Lean declarations/files:** `BasisInteractionNondegenerate`,
  `FiniteCoefficientSetup`, `PaperFiniteIdentifiability.lean`.

---

### 2026-07-04 — flat kernel gives only mean matching

- **Exact condition:** Use the normalized mean-shift field with the constant
  kernel `k(x,y)=1`, while retaining probability normalization and even full
  support.
- **Intended mechanism:** Hope that zero normalized mean-shift drift identifies
  the entire distribution.
- **Stress test/model:** On `ℝ`, let `p = Normal(0,1)` and let
  `q = ½ Normal(-1,1) + ½ Normal(1,1)`. Both have smooth full-support densities
  and mean zero, but they are distinct.
- **Counterexample or obstruction:** For `k=1`, `Zₚ=Z_q=1` and
  `Vₚ,q(x)=Eₚ[Y]-E_q[Y]=0` for every `x`.
- **Why the argument fails:** A flat kernel exposes only the first moment, not
  the distribution.
- **Fatal or repairable:** Fatal for flat kernels; informative kernel/probe
  separation is indispensable.
- **Possible repair:** Use a kernel whose induced interaction family separates
  the chosen model class. Characteristicness is relevant for MMD embeddings,
  but must still be connected to this normalized mean-shift field rather than
  assumed to transfer automatically.
- **Relevant Lean declarations/files:** `meanShiftDrift`, `laplaceKernel`,
  `BasisInteractionNondegenerate`, `Paperaxioms.lean`.

---

### 2026-07-04 — legitimacy witness collapses for `m ≤ 1`

- **Exact condition:** Try to certify the finite-basis condition as legitimate
  (admits a distinct pair before zero drift) by exhibiting normalized
  coefficient vectors `a ≠ b` for arbitrary `m`.
- **Intended mechanism:** Use the simplex vertices `a = e₀`, `b = e₁` as the
  distinct pair required by `ConditionAllowsDistinctPair`.
- **Stress test/model:** `m = 1` and `m = 0`.
- **Counterexample or obstruction:** For `m = 1` the only normalized
  nonnegative vector is `(1)`, so no distinct pair exists; for `m = 0` no
  normalized vector exists at all (empty sum `= 0 ≠ 1`). The vertices `e₀, e₁`
  need two distinct coordinates, i.e. `2 ≤ m`.
- **Why the argument fails:** Legitimacy is a real hypothesis on the model
  size, not automatic. Below `m = 2` the condition is *vacuously* satisfied by a
  single point and cannot demonstrate non-triviality.
- **Fatal or repairable:** Scoped, not fatal. The witness is correct exactly
  for `2 ≤ m`, which is the only regime where the finite route is interesting.
- **Possible repair:** State the legitimacy witnesses under the explicit
  hypothesis `2 ≤ m`. Separately, distinct *coefficients* only lift to distinct
  *densities* when the basis is linearly independent (`DensityBasisIndependent`);
  without it, `φ₀ = φ₁` collapses `e₀` and `e₁` to the same density. Both
  hypotheses are recorded in the Lean witnesses.
- **Relevant Lean declarations/files:** `stdBasisProbVector`,
  `exists_distinct_probVectors`, `basisDensity_injective_of_basisIndependent`,
  `finiteConditionAllowsDistinctDensities`, `FiniteLegitimacy.lean`.

---

### 2026-07-04 — raw drift-field norm does not obviously control weak convergence

- **Exact condition:** Try to state the asymptotic target directly on the drift
  field: `‖Vₚ,qₙ‖ → 0` (some norm of the MMD drift field, equation 41) ⟹
  `qₙ → p` in distribution, for a characteristic kernel.
- **Intended mechanism:** Import a metrization theorem to convert vanishing drift
  into convergence, mirroring the exact route.
- **Stress test/model:** Consider that `Vₚ,q(x) = Φₚ(x) − Φ_q(x)` is the
  difference of *gradient* mean embeddings, whereas the MMD (which metrizes weak
  convergence) is the norm of the *value* embedding difference.
- **Counterexample or obstruction:** A small drift gradient does not imply a
  small MMD. `V = -½∇(MMD²)` (Appendix C.2), so `V → 0` is a *critical-point*
  condition, satisfied at maxima/saddles of the MMD, not only at its zero. Sup-
  norm convergence of embeddings is also strictly weaker than RKHS-norm (MMD)
  convergence. So `‖V‖ → 0 ⟹ MMD → 0` is not a clean, known truth.
- **Why the argument fails:** The would-be axiom would assert something we are
  *not* confident is true, which violates the project's honesty policy — only
  genuinely-known theorems may be assumed.
- **Fatal or repairable:** Repairable by changing the driving quantity. The MMD
  discrepancy itself (equation 37) *is* the training objective the drift
  minimizes, and `MMD → 0 ⟹ weak convergence` is a solid, well-known theorem.
- **Possible repair (taken):** State the asymptotic result with the MMD
  discrepancy as the drift size — `gaussianMmd_asymptoticallyIdentifies` via
  `gaussian_mmd_metrizes_weakConvergence` — and explicitly flag that the raw
  field-norm version is not claimed.
- **Relevant Lean declarations/files:** `mmdDrift`, `mmdSquared`,
  `gaussian_mmd_metrizes_weakConvergence`, `gaussianMmd_asymptoticallyIdentifies`,
  `CharacteristicIdentifiability.lean`.

---

### 2026-07-05 — characteristic embedding axiom as a claimed solution

- **Exact condition:** Assume `characteristic_gradientEmbedding_injective` for
  the gradient embedding defining `mmdDrift`.
- **Intended mechanism:** Zero drift is equality of those embeddings, followed
  immediately by the assumed injectivity.
- **Counterexample or obstruction:** For this field the assumption contains
  the substantive target implication; renaming it “characteristic” does not
  independently solve the research problem.
- **Fatal or repairable:** Fatal as a promoted proof; useful as a conditional
  reduction.
- **Possible repair:** Prove the analytic injectivity theorem from lower-level
  Euclidean Fourier/RKHS facts, or keep the result explicitly conditional.
- **Relevant Lean declarations/files:** `CharacteristicIdentifiability.lean`,
  `DriftingIdentifiability.Conditional`.

---

### 2026-07-05 — synthetic Gaussian interactions mistaken for paper interactions

- **Exact condition:** Use an anti-symmetric extension of Gaussian Gram rows as
  the interaction vectors.
- **Counterexample or obstruction:** No theorem identifies these synthetic
  vectors with the paper's integral-induced `basisInteraction` vectors.
- **Fatal or repairable:** Repairable only by proving that analytic bridge.
- **Possible repair:** Treat the construction as a consistency example and
  estimate a frame bound for the actual integral system.
- **Relevant Lean declarations/files:** `GaussianNondegeneracy.lean`.

---

### 2026-07-05 — zero normalized drift with invalid normalizers

- **Exact condition:** Manipulate equation (11) without assuming both kernel
  normalizers are nonzero.
- **Counterexample or obstruction:** Multiplication by an inverse cannot be
  cancelled at a zero normalizer; the normalized field is not regular there.
- **Fatal or repairable:** Repairable with `MeanShiftRegularAt` at every probe.
- **Relevant Lean declarations/files:** `zero_target_normalizer_not_regular`,
  `PopulationMeanShiftFiniteSetup`.

---

### 2026-07-05 — sampled/a.e. zero silently promoted to global zero

- **Exact condition:** Infer pointwise `ZeroDrift` directly from zero expected
  squared drift under `q`.
- **Counterexample or obstruction:** Under `q=dirac 0`, the continuous identity
  field is zero almost everywhere but not globally.
- **Fatal or repairable:** Repairable with continuity and positive mass on
  every nonempty open set.
- **Relevant Lean declarations/files:** `identityDrift_zeroAE_at_diracZero`,
  `zeroDrift_of_ae_of_continuous_fullSupport`.

---

### 2026-07-05 — feature matching treated as source-law matching

- **Exact condition:** Conclude `p=q` from equality of `φ♯p` and `φ♯q` for an
  arbitrary feature map.
- **Counterexample or obstruction:** A non-injective feature collapses two
  distinct Dirac measures.
- **Fatal or repairable:** Repairable when `φ` is a measurable embedding; in
  general only pushforward equality may be claimed.
- **Relevant Lean declarations/files:** `noninjectiveFeature_collapses_dirac`,
  `sourceMeasure_eq_of_featureLaw_eq`.

---

### 2026-07-05 — population theorem applied to CFG or minibatches

- **Exact condition:** Treat Algorithm 2's finite bi-softmax estimator or an
  affine CFG target as the same object as the normalized population field.
- **Counterexample or obstruction:** The estimator has sampling and batch
  normalization error, while CFG can leave the probability simplex and require
  signed measures.
- **Fatal or repairable:** Separate research problems, not consequences of the
  current theorem.
- **Possible repair:** Prove finite-sample approximation bounds and formalize a
  signed-measure CFG extension.
- **Relevant Lean declarations/files:** `algorithm2Drift`, `cfgNegativeDensity`,
  `PopulationIdentifiability.lean`.

---

### 2026-07-05 — Gaussian product profiles with colliding pair sums

- **Exact condition:** Use one-dimensional empirical point densities and the
  structured integer-probe Gaussian construction, but allow two distinct
  strict pairs `(i,j) ≠ (k,l)` with `zᵢ+zⱼ = zₖ+zₗ`.
- **Intended mechanism:** Recover every coefficient minor from the Gaussian
  product profiles.
- **Counterexample or obstruction:** After removing nonzero row and column
  factors, both pair profiles have the same geometric base
  `exp(zᵢ+zⱼ)`, so their columns are proportional and the Vandermonde matrix
  is singular.
- **Why the argument fails:** Distinct support points alone do not distinguish
  unordered pairs; for example, equally spaced supports produce repeated pair
  sums.
- **Fatal or repairable:** Repairable for this construction.
- **Possible repair:** Require `DistinctStrictPairSums` (a finite Sidon-type
  condition), alter the support geometry, or use richer multidimensional
  probes that can separate colliding sums.
- **Relevant Lean declarations/files:** `DistinctStrictPairSums`,
  `gaussianEmpiricalPoint_basisNondegenerate`, `EmpiricalFrameBound.lean`.

---

### 2026-07-05 — qualitative frame existence treated as good conditioning

- **Exact condition:** Infer practical stability merely from linear
  independence of the interaction vectors.
- **Counterexample or obstruction:** Finite-dimensional compactness supplies
  some `c>0`, but gives no useful lower bound; nearly colliding Gaussian bases
  can make `c` arbitrarily small.
- **Why the argument fails:** Exact nonsingularity and numerical conditioning
  are different claims.
- **Fatal or repairable:** Repairable with explicit separation and
  singular-value/Vandermonde bounds.
- **Update (2026-07-05):** An explicit computable *ceiling* is now proved:
  `c ≤ min_{i<j} |zᵢ-zⱼ| e^{-(zᵢ-zⱼ)²/4} ≤ √(2/e)`
  (`interactionFrameBound_le_interactionNorm`,
  `gaussianEmpiricalPoint_frameConstant_le`). This *confirms* the concern
  rather than resolving it: the integer-probe Gaussian construction cannot be
  well-conditioned — the constant is capped below `0.858` and forced small
  whenever any support pair is close or far apart. A useful *lower* bound is
  still missing; it reduces to a weighted-Vandermonde smallest-singular-value
  estimate (a Lagrange-interpolation certificate is written out in
  `WrittenProof.md`) or to a better probe design (Objective 3).
- **Later update (2026-07-05):** The logical lower-bound gap is now repaired by
  `interactionFrameBound_inverseCertificate`. For the actual square
  interaction matrix `M`, Lean proves that
  `(∑_{p,r}|(M⁻¹)_{p,r}|)⁻¹` is positive and is a valid lower frame constant.
  `gaussianEmpiricalPointSetup` uses its concrete Gaussian specialization.
  This does **not** retract the failure warning: the certified value may be
  extremely small, so no claim of good conditioning is permitted without
  evaluating or further bounding it.
- **Relevant Lean declarations/files:**
  `interactionFrameBound_of_linearIndependent`,
  `interactionFrameBound_inverseCertificate`,
  `interactionFrameBound_le_interactionNorm`,
  `gaussianEmpiricalPoint_frameConstant_le`,
  `gaussianEmpiricalPointFrameConstant`, `FiniteStability.lean`,
  `EmpiricalFrameBound.lean`.

---

### 2026-07-05 — smooth-basis interface mistaken for a certified practical model

- **Exact condition:** Treat the existence of
  `SmoothProbabilityDensityBasis` or `continuousPerturbationSetup` as proof
  that a realistic continuum-supported architecture satisfies identifiability.
- **Counterexample or obstruction:** Smoothness of the component densities says
  nothing by itself about linear independence or conditioning of their induced
  interaction vectors. The perturbation route additionally requires an actual
  finite bound `δ<c` relative to a certified baseline.
- **Why the argument fails:** Regularity and identifiability are independent
  properties. A smooth basis may collapse interactions, and a formally positive
  frame constant may still be numerically useless.
- **Fatal or repairable:** Repairable by proving or interval-certifying the
  interaction error for a concrete continuum-supported basis, or by providing
  a direct `InteractionDualCertificate`.
- **Update (2026-07-05):** `SmoothBumpBasis.lean` now supplies one direct repair:
  a non-atomic two-component `C∞` basis whose frame is certified by an exact
  ordered-support sign argument. The warning remains applicable to every other
  smooth basis; smoothness alone is still not an identifiability condition.
- **Relevant Lean declarations/files:** `SmoothProbabilityDensityBasis`,
  `continuousPerturbationSetup`,
  `interactionFrameBound_of_uniformPerturbation`,
  `InteractionDualCertificate`, `bumpInteractionFrameBound`,
  `PracticalModelClasses.lean`, `SmoothBumpBasis.lean`.

---

### 2026-07-06 — affinity-perturbation reduction of Algorithm-2 consistency

- **Exact condition:** Close Objective 4 by instantiating the estimator-agnostic
  finite-sample bridge (`estimate_failure_le_meanSquare`) with
  `Vhat := algorithm2Drift`, discharging its `‖V − Vhat‖ ≤ ε` hypothesis through a
  deterministic *stability in the affinities*: compare `algorithm2Drift` (which is
  `driftOfAffinities A` for the softmax affinity `A`, by
  `algorithm2Drift_eq_affinityPairSum`) against `driftOfAffinities a*` for the
  population/expected affinity system `a*`, then bound the sup-affinity deviation
  `η = maxₛ |A s − a* s|` by a law of large numbers.
- **Intended mechanism:** `driftOfAffinities` is bilinear in the affinities, so
  `‖driftOfAffinities A − driftOfAffinities a*‖ ≤ 4·Npos·Nneg·R·η`
  (each of the `Npos·Nneg` pairs contributes `|A(+)A(−) − a*(+)a*(−)|·‖Δy‖ ≤
  2η·2R`).  Feed the resulting `ε = 4·Npos·Nneg·R·η` into the bridge.
- **Stress test/model:** iid samples `y_s`; softmax kernel `k(x,y)=e^{−‖x−y‖/τ}`,
  so the row affinity is the self-normalized weight `A_row(s)=k_s/Σ_{s'}k_{s'}`.
- **Counterexample or obstruction:** the crude bilinear bound has the wrong scaling
  and does **not** vanish.  A single softmax weight is `A ~ 1/N` (it is one of `N`
  normalized weights), while its sampling fluctuation about the ideal
  `a* = k/(N·Z)` is `η ~ N^{-3/2}` (numerator fixed `O(1)`, denominator
  `Σk = N·Z + O(√N)`, so `A − a* = k(N Z − Σk)/(Σk·N Z) ~ O(1)·√N / N² = N^{-3/2}`).
  The prefactor `4·Npos·Nneg ~ N²` then gives `ε ~ N² · N^{-3/2} = √N → ∞`.  The
  bound grows instead of shrinking.
- **Why the argument fails:** the estimator is a **self-normalized** (ratio) sum;
  its `N²` pairwise terms are each `O(1/N²)` and their consistency comes from
  *cancellation/averaging across the pairs*, not from smallness of any individual
  affinity error.  A sup-norm perturbation bound discards exactly that
  cancellation, so it over-counts by a factor that beats the per-affinity decay.
- **Fatal or repairable:** Repairable only by the genuine self-normalized analysis;
  the elementary reduction is fatally lossy.
- **Possible repair:** work at the level of the *mass-scaled centroid* form
  `algorithm2Drift = Q•S_pos − P•S_neg` (`algorithm2Drift_eq_massScaledCentroid`),
  where `S_pos/P` and `S_neg/Q` are softmax-weighted centroids lying in the convex
  hull of the samples (`algorithm2Drift_norm_le_affinityMass` records the
  convex-hull bound).  Each of `P`, `Q`, `S_pos`, `S_neg` is a properly-normalized
  self-normalized average whose LLN bias is the standard importance-sampling
  `O(1/N)`; ratio consistency (e.g. a delta-method / self-normalized-IS bound)
  then controls the centroids with the correct scaling.  Formalizing generic
  self-normalized IS consistency is a standalone (Mathlib-grade) analysis task and
  needs the explicit iid sampling model; it is the true content of the remaining
  Objective-4 gap, and it cannot be replaced by an axiom without assuming the
  estimator-consistency conclusion.
- **Update (2026-07-06):** the generic standalone ratio step is now formalized in
  `SelfNormalizedConsistency.lean` as `selfNormalized_meanSquare_le`, with no new
  axiom beyond the already reviewed `sampleMean_meanSquare_le`.  The obstruction
  recorded here remains relevant to the naive sup-affinity route, and the
  no-self-mask repair is now formalized in `Algorithm2SNIS.lean`: the row
  softmax factor cancels in the centroids, leaving SNIS estimators for the
  column-reweighted weight
  `sqrt(k(x_i,y) * k(x_i,y) / sum_r k(x_r,y))`.  The remaining gap is no longer
  the generic ratio theorem or the no-mask centroid coupling; it is the
  deterministic identifiability/frame condition for this modified kernel, plus a
  separate perturbative treatment of the implementation self-mask.
- **Update (2026-07-06, later):** the conditional deterministic bridge for the
  modified kernel is now formalized in `ColumnReweightedMeanShift.lean`.
  Under an explicit `InteractionFrameBound` for the column-reweighted interaction
  vectors, zero of the no-mask limiting centroid field identifies the represented
  measures and inherits the finite-sample MSE bridge.  `FiniteStability.lean` also
  contains `interactionFrameBound_of_strictPairScaling`, which is the intended
  tool for transferring a baseline frame through positive strict-pair column
  scalings.  The unresolved part is now concrete certification/design of that
  modified-kernel frame in useful model classes, and the implementation
  self-mask perturbation.
- **Update (2026-07-06, self-mask repair):** the implementation mask is now
  formalized in `SelfMaskPerturbation.lean` as an explicit deterministic
  perturbation, but the target of the comparison is the leave-masked-out/deleted
  estimator, not the full no-mask estimator on the same samples.  The proved
  facts include the exact multiplicative logit suppression
  `maskedWeight_eq_factor_mul_noMaskWeight`, the reconciliation
  `deletedDrift_false_eq_algorithm2Drift`, the eye-mask unmasked-column lemma,
  and the bound
  `algorithm2Drift_sub_deletedDrift_norm_le_eyeMask`.  This repairs the
  deterministic implementation-mask correction; the remaining statistical gap
  is a consistency theorem for the deleted estimator's index-dependent leave-out
  SNIS structure.
- **Relevant Lean declarations/files:** `algorithm2Drift_eq_affinityPairSum`,
  `algorithm2Drift_eq_massScaledCentroid`, `algorithm2Drift_norm_le_affinityMass`,
  `algorithm2Drift_eq_massProduct_centroidDiff`, `algorithm2Drift_matched_zero`,
  `estimate_failure_le_meanSquare`,
  `Algorithm2Estimator.lean`, `FiniteSampleBridge.lean`,
  `SelfNormalizedConsistency.lean`, `Algorithm2SNIS.lean`,
  `ColumnReweightedMeanShift.lean`, `SelfMaskPerturbation.lean`.

---

### 2026-07-05 — two-atom Laplace result overgeneralized

- **Exact condition:** Infer general-`m` or high-dimensional Laplace
  identifiability from `empirical01Laplace_identifies_of_probeEnergy_eq_zero`.
- **Counterexample or obstruction:** The proved theorem has exactly two atoms
  and one probe. Laplace interaction profiles on a line have substantially
  different rank behavior from the Gaussian Vandermonde family.
- **Why the argument fails:** Positivity of the Laplace kernel guarantees
  nonzero interaction for one strict pair, not independence of many pairs.
- **Fatal or repairable:** Repairable only with a separate general-`m` frame or
  dual-certificate construction.
- **Relevant Lean declarations/files:** `empirical01LaplaceSetup`,
  `empirical01Laplace_identifies_of_probeEnergy_eq_zero`,
  `PracticalModelClasses.lean`.
