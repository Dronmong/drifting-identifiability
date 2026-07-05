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
