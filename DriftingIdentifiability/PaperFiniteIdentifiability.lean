import DriftingIdentifiability.FiniteGrouping

/-!
# Finite-basis route suggested by Appendix C.1

This module packages and proves the paper's finite-dimensional coefficient
argument. It establishes coefficient and represented-density equality from the
grouped bilinear hypothesis.
-/

open scoped BigOperators

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

/-- Algebraic hypotheses remaining after the analytic expansion in Appendix
C.1.  Normalization is included explicitly because it is required to turn
parallel coefficient vectors into equal probability densities. -/
structure FiniteCoefficientSetup
    (E : Type u) [AddCommGroup E] [Module ℝ E]
    (m N : ℕ) where
  U : Fin m → Fin m → Fin N → E
  anti : AntiSymmetricBasisInteractions U
  nondegenerate : BasisInteractionNondegenerate U
  a : FiniteProbabilityVector m
  b : FiniteProbabilityVector m
  zeroBilinear : (∑ i, ∑ j, (a.weight i * b.weight j) • U i j) = 0

/-- Every coefficient minor vanishes.  This is not the forbidden conclusion
`a = b` or `p = q`.  It is proved from scratch in `FiniteGrouping.lean` (anti-
symmetry + Mathlib linear independence), so it no longer relies on the paper
paper's coefficient-minor conclusion, now proved in `FiniteGrouping.lean`. -/
theorem coefficientMinorsVanish
    {E : Type u} [AddCommGroup E] [Module ℝ E]
    {m N : ℕ} (setup : FiniteCoefficientSetup E m N) :
    AllCoefficientMinorsZero setup.a.weight setup.b.weight := by
  intro i j
  exact coefficientMinorsVanish_of_antisymm setup.U setup.anti setup.nondegenerate
    setup.a.weight setup.b.weight setup.zeroBilinear i j

/-- Probability normalization removes the scale ambiguity left by vanishing
coefficient minors. -/
theorem normalizedParallelCoefficientsAreEqual
    (m : ℕ) (a b : FiniteProbabilityVector m)
    (hminors : AllCoefficientMinorsZero a.weight b.weight) :
    a.weight = b.weight := by
  funext i
  have hsum : ∑ j, coefficientMinor a.weight b.weight i j = 0 := by
    apply Finset.sum_eq_zero
    intro j _hj
    exact hminors i j
  simp only [coefficientMinor, Finset.sum_sub_distrib] at hsum
  rw [← Finset.mul_sum, ← Finset.sum_mul, b.normalized, a.normalized,
    mul_one, one_mul] at hsum
  exact sub_eq_zero.mp hsum

/-- The finite coefficient vectors in Appendix C.1 are identifiable under the
paper's interaction nondegeneracy condition. -/
theorem finiteCoefficientIdentifiable
    {E : Type u} [AddCommGroup E] [Module ℝ E]
    {m N : ℕ} (setup : FiniteCoefficientSetup E m N) :
    setup.a.weight = setup.b.weight :=
  normalizedParallelCoefficientsAreEqual m setup.a setup.b
    (coefficientMinorsVanish setup)

/-- Equal identifiable coefficients yield equal densities in their common
finite basis. -/
theorem finiteBasisDensitiesEqual
    {E : Type u} [AddCommGroup E] [Module ℝ E]
    {X : Type*} {m N : ℕ} (setup : FiniteCoefficientSetup E m N)
    (φ : Fin m → X → ℝ) :
    basisDensity m φ setup.a.weight = basisDensity m φ setup.b.weight := by
  rw [finiteCoefficientIdentifiable setup]

end PaperFiniteIdentifiability
end DriftingIdentifiability
