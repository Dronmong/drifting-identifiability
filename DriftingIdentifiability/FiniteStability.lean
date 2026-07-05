import DriftingIdentifiability.PaperFiniteIdentifiability

/-!
# Quantitative coefficient stability

`normalizedParallelCoefficientsAreEqual` is the *exact* statement: vanishing
minors force `a = b`.  A genuinely asymptotic identifiability result needs its
quantitative shadow — a bound of the coefficient distance by the size of the
minors — so that *small* minors force *close* coefficients.  `WrittenProof.md`
flags this stability estimate as the missing ingredient for limit results.

This module supplies the axiom-free, finite-dimensional half of that estimate:
the identity `aᵢ - bᵢ = ∑ⱼ (aᵢbⱼ - aⱼbᵢ)` for normalized vectors, its `ℓ¹`
consequence, and the resulting convergence statement `minor mass → 0 ⟹
coefficient distance → 0`.  Nothing here uses a paper axiom; it depends only on
probability normalization.  Bounding the minor mass itself by the drift norm
(the linear-independence lower bound) remains the outstanding analytic step.
-/

open scoped BigOperators
open Filter Topology

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

/-- For normalized coefficient vectors, each coordinate difference is exactly the
row sum of the coefficient minors.  This is the quantitative form of the
computation inside `normalizedParallelCoefficientsAreEqual`. -/
theorem coeff_sub_eq_sum_minor (m : ℕ) (a b : FiniteProbabilityVector m)
    (i : Fin m) :
    a.weight i - b.weight i
      = ∑ j, coefficientMinor a.weight b.weight i j := by
  simp only [coefficientMinor, Finset.sum_sub_distrib]
  rw [← Finset.mul_sum, ← Finset.sum_mul, a.normalized, b.normalized,
    mul_one, one_mul]

/-- The `ℓ¹` coefficient distance is bounded by the total absolute mass of the
minors. -/
theorem coeff_l1_le_minor_mass (m : ℕ) (a b : FiniteProbabilityVector m) :
    ∑ i, |a.weight i - b.weight i|
      ≤ ∑ i, ∑ j, |coefficientMinor a.weight b.weight i j| := by
  apply Finset.sum_le_sum
  intro i _
  rw [coeff_sub_eq_sum_minor m a b i]
  exact Finset.abs_sum_le_sum_abs _ _

/-- **Coefficient stability.**  If the total minor mass of a sequence `bₙ`
against a fixed `a` tends to zero, then the `ℓ¹` coefficient distance tends to
zero.  This is the finite-coefficient shadow of `AsymptoticallyIdentifies`: it
turns the exact result into a convergence statement, still short of a full
distribution-level asymptotic theorem. -/
theorem coeffL1_tendsto_zero_of_minorMass_tendsto_zero
    (m : ℕ) (a : FiniteProbabilityVector m)
    (b : ℕ → FiniteProbabilityVector m)
    (h : Tendsto
      (fun n => ∑ i, ∑ j, |coefficientMinor a.weight (b n).weight i j|)
      atTop (𝓝 0)) :
    Tendsto (fun n => ∑ i, |a.weight i - (b n).weight i|) atTop (𝓝 0) :=
  squeeze_zero
    (fun _ => Finset.sum_nonneg fun _ _ => abs_nonneg _)
    (fun n => coeff_l1_le_minor_mass m a (b n))
    h

end PaperFiniteIdentifiability
end DriftingIdentifiability
