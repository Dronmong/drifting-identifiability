import DriftingIdentifiability.PaperFiniteIdentifiability

/-!
# Legitimacy of the finite-basis interaction-separation condition

`PaperFiniteIdentifiability` proves that zero drift *plus* interaction
separation forces coefficient — hence density — equality.  The anti-cheating
protocol (`AGENT.md`, §1 and the anti-circularity checklist) additionally
requires a machine-checked demonstration that the *pre-zero-drift* condition is
not vacuous: it must admit a genuinely distinct pair `a ≠ b` **before** zero
drift is imposed, and that distinctness must survive the passage to densities.

This file supplies those witnesses.  Nothing here assumes `V = 0`, `a = b`,
`p = q`, or any uniqueness/injectivity statement; every result is a statement
about the freedom that remains once zero drift is *removed*.

The stress-test boundary `2 ≤ m` and the basis-independence requirement are
recorded in `LoggedFailures.md`; both appear as explicit hypotheses below.
-/

open scoped BigOperators

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

/-! ## Standard-basis probability vectors -/

/-- The `k`-th simplex vertex as a finite probability vector: unit mass on the
single coordinate `k`.  This is a legitimate probability coefficient vector and
makes no reference to zero drift. -/
def stdBasisProbVector (m : ℕ) (k : Fin m) : FiniteProbabilityVector m where
  weight := fun i => if i = k then 1 else 0
  nonnegative := by
    intro i
    by_cases h : i = k <;> simp [h]
  normalized := by simp

/-- Distinct vertices give coefficient-distinct probability vectors. -/
theorem stdBasisProbVector_coeff_distinct (m : ℕ) {k l : Fin m} (h : k ≠ l) :
    CoefficientsDistinct (stdBasisProbVector m k).weight
      (stdBasisProbVector m l).weight := by
  refine ⟨k, ?_⟩
  have hk1 : (stdBasisProbVector m k).weight k = 1 := if_pos rfl
  have hl0 : (stdBasisProbVector m l).weight k = 0 := if_neg h
  rw [hk1, hl0]
  norm_num

/-- For `2 ≤ m` the finite probability simplex contains two distinct points.
This is the concrete distinct pair required for legitimacy. -/
theorem exists_distinct_probVectors {m : ℕ} (hm : 2 ≤ m) :
    ∃ a b : FiniteProbabilityVector m, a.weight ≠ b.weight := by
  have hk : (0 : ℕ) < m := by omega
  have hl : (1 : ℕ) < m := by omega
  have hkl : (⟨0, hk⟩ : Fin m) ≠ ⟨1, hl⟩ := by
    intro hcon
    have hval : (0 : ℕ) = 1 := congrArg Fin.val hcon
    omega
  refine ⟨stdBasisProbVector m ⟨0, hk⟩, stdBasisProbVector m ⟨1, hl⟩, ?_⟩
  obtain ⟨i, hi⟩ := stdBasisProbVector_coeff_distinct m hkl
  intro heq
  exact hi (congrFun heq i)

/-! ## Distinct coefficients lift to distinct densities -/

/-- Under a linearly independent basis, the coefficient-to-density map is
injective.  This is the distinctness counterpart of `finiteBasisDensitiesEqual`:
it needs the same basis-independence hypothesis, and it does *not* invoke zero
drift. -/
theorem basisDensity_injective_of_basisIndependent
    {X : Type u} {m : ℕ} (φ : Fin m → X → ℝ)
    (hφ : DensityBasisIndependent φ) {a b : Fin m → ℝ}
    (h : basisDensity m φ a = basisDensity m φ b) : a = b := by
  have hzero : ∑ i, (a i - b i) • φ i = (0 : X → ℝ) := by
    funext y
    have hy := congrFun h y
    simp only [basisDensity] at hy
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, sub_mul,
      Finset.sum_sub_distrib, Pi.zero_apply]
    linarith [hy]
  have hcoef := (Fintype.linearIndependent_iff.mp hφ) (fun i => a i - b i) hzero
  funext i
  linarith [hcoef i]

/-- Distinct coefficient vectors yield distinct densities, provided the finite
basis is linearly independent. -/
theorem basisDensity_distinct_of_coeff_distinct
    {X : Type u} {m : ℕ} (φ : Fin m → X → ℝ)
    (hφ : DensityBasisIndependent φ) {a b : Fin m → ℝ}
    (h : a ≠ b) : basisDensity m φ a ≠ basisDensity m φ b :=
  fun heq => h (basisDensity_injective_of_basisIndependent φ hφ heq)

/-- **Legitimacy at the density level.**  For `2 ≤ m` and a linearly
independent basis, the finite condition admits two probability vectors whose
represented densities are genuinely different — established without assuming
zero drift.  Together with `finiteBasisDensitiesEqual` (which forces equality
*after* zero drift), this shows the condition is nonvacuous rather than a hidden
restatement of `p = q`. -/
theorem finiteConditionAllowsDistinctDensities
    {X : Type u} {m : ℕ} (hm : 2 ≤ m) (φ : Fin m → X → ℝ)
    (hφ : DensityBasisIndependent φ) :
    ∃ a b : FiniteProbabilityVector m,
      basisDensity m φ a.weight ≠ basisDensity m φ b.weight := by
  obtain ⟨a, b, hab⟩ := exists_distinct_probVectors hm
  exact ⟨a, b, basisDensity_distinct_of_coeff_distinct φ hφ hab⟩

/-! ## The separation hypothesis is itself satisfiable

Legitimacy could still be vacuous if no interaction system actually satisfies
both anti-symmetry *and* nondegeneracy: `BasisInteractionNondegenerate` demands
`{Uᵢⱼ : i<j}` be linearly independent, which is a real constraint.  We exhibit a
concrete such system for `m = 2` so that the separation premises are not empty
promises.  This addresses the "degenerate interaction vectors" failure logged in
`LoggedFailures.md` from the opposite side: there they were made to vanish; here
we show a genuinely nondegenerate choice exists. -/

/-- A concrete interaction system on `m = 2`, `N = 1`, valued in `ℝ`:
`Uᵢⱼ = i - j`.  It is anti-symmetric by construction and its single strict-pair
vector `U₀₁ = -1` is nonzero, hence linearly independent. -/
def witnessU : Fin 2 → Fin 2 → Fin 1 → ℝ :=
  fun i j _ => (i.val : ℝ) - (j.val : ℝ)

theorem witnessU_anti : AntiSymmetricBasisInteractions witnessU := by
  intro i j
  funext n
  simp only [witnessU, Pi.neg_apply]
  ring

/-- The strict-pair index set for `m = 2` is a singleton. -/
instance : Unique (StrictPair 2) where
  default := ⟨(0, 1), by decide⟩
  uniq := by decide

theorem witnessU_nondegenerate : BasisInteractionNondegenerate witnessU := by
  rw [BasisInteractionNondegenerate, linearIndependent_unique_iff]
  change witnessU (0 : Fin 2) (1 : Fin 2) ≠ 0
  intro hcon
  have h0 := congrFun hcon (0 : Fin 1)
  simp [witnessU] at h0

/-- **Legitimacy of the full separation condition.**  There is an interaction
system that is simultaneously anti-symmetric and nondegenerate — satisfying both
hypotheses of `FiniteCoefficientSetup` — while still admitting two distinct
normalized coefficient vectors.  Only adding the zero-drift equation
(`zeroBilinear`) collapses them, which is exactly `finiteCoefficientIdentifiable`.
This certifies the condition encodes no hidden `a = b`. -/
theorem exists_separation_with_distinct_pair :
    ∃ (N : ℕ) (U : Fin 2 → Fin 2 → Fin N → ℝ)
      (a b : FiniteProbabilityVector 2),
      AntiSymmetricBasisInteractions U ∧
      BasisInteractionNondegenerate U ∧
      a.weight ≠ b.weight := by
  obtain ⟨a, b, hab⟩ := exists_distinct_probVectors (le_refl 2)
  exact ⟨1, witnessU, a, b, witnessU_anti, witnessU_nondegenerate, hab⟩

end PaperFiniteIdentifiability
end DriftingIdentifiability
