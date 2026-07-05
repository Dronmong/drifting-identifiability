import DriftingIdentifiability.PaperFiniteIdentifiability

/-!
# Discharging the nondegeneracy hypothesis for the Gaussian kernel

The finite route assumes `BasisInteractionNondegenerate` — linear independence of
the interaction vectors `{Uᵢⱼ : i<j}`.  Here we *derive* that hypothesis for a
concrete Gaussian-kernel interaction system, reducing it to the well-known fact
that the Gaussian kernel is strictly positive definite (Micchelli, 1986): for
distinct points the Gram matrix is nonsingular, so its rows are independent.

Two axiom-free pieces do the reduction:

* `antisymmExtend` builds an anti-symmetric interaction family from any family
  indexed by the strict pairs, and `antisymmExtend_nondegenerate` shows its
  strict-pair restriction is exactly that family — so independence transfers.

The only imported fact is the Gaussian Gram nonsingularity, and the resulting
identifiability theorem `gaussianInteractionIdentifiesCoefficients` therefore has
its nondegeneracy *discharged*, not assumed.
-/

open scoped BigOperators

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

variable {V : Type u} [AddCommGroup V] [Module ℝ V] {m : ℕ}

/-- Anti-symmetric extension of a family indexed by the strict pairs `i < j`:
`Uᵢⱼ = Wᵢⱼ` when `i < j`, `-Wⱼᵢ` when `j < i`, and `0` on the diagonal. -/
def antisymmExtend (W : StrictPair m → V) (i j : Fin m) : V :=
  if h : i.val < j.val then W ⟨(i, j), h⟩
  else if h' : j.val < i.val then -W ⟨(j, i), h'⟩
  else 0

omit [Module ℝ V] in
theorem antisymmExtend_anti (W : StrictPair m → V) (i j : Fin m) :
    antisymmExtend W i j = -antisymmExtend W j i := by
  unfold antisymmExtend
  rcases lt_trichotomy i.val j.val with h | h | h
  · rw [dif_pos h, dif_neg (not_lt.mpr h.le), dif_pos h, neg_neg]
  · have h1 : ¬ i.val < j.val := by omega
    have h2 : ¬ j.val < i.val := by omega
    rw [dif_neg h1, dif_neg h2, dif_neg h2, dif_neg h1, neg_zero]
  · rw [dif_neg (not_lt.mpr h.le), dif_pos h, dif_pos h]

omit [Module ℝ V] in
/-- The strict-pair restriction of the anti-symmetric extension is the original
family. -/
theorem antisymmExtend_strict (W : StrictPair m → V) :
    (fun p : StrictPair m => antisymmExtend W p.1.1 p.1.2) = W := by
  funext p
  unfold antisymmExtend
  rw [dif_pos p.2]

/-- **Nondegeneracy transfers.**  If the strict-pair family is linearly
independent, the anti-symmetric extension is nondegenerate. -/
theorem antisymmExtend_nondegenerate (W : StrictPair m → V)
    (hW : LinearIndependent ℝ W) :
    LinearIndependent ℝ (fun p : StrictPair m => antisymmExtend W p.1.1 p.1.2) := by
  rw [antisymmExtend_strict]
  exact hW

/-! ## The Gaussian instantiation -/

section Gaussian
variable {E : Type u} [NormedAddCommGroup E]

/-- The interaction directions built from Gaussian kernel values at points `x`
indexed by the strict pairs: `Wₚ(q) = k_σ(xₚ, x_q)`.  These are the rows of the
Gaussian Gram matrix. -/
noncomputable def gaussianGram (σ : ℝ) (x : StrictPair m → E) :
    StrictPair m → (StrictPair m → ℝ) :=
  fun p q => gaussianKernel σ (x p) (x q)

/-- **Nondegeneracy discharged for the Gaussian kernel.**  For distinct points,
the anti-symmetric interaction system built from Gaussian kernel values is
nondegenerate — reduced to Micchelli's strict positive definiteness, not
assumed. -/
theorem gaussianInteraction_nondegenerate (σ : ℝ) (hσ : ValidBandwidth σ)
    (x : StrictPair m → E) (hx : Function.Injective x) :
    LinearIndependent ℝ
      (fun p : StrictPair m => antisymmExtend (gaussianGram σ x) p.1.1 p.1.2) :=
  antisymmExtend_nondegenerate (gaussianGram σ x)
    (gaussian_gram_linearIndependent σ hσ x hx)

/-- **Finite identifiability for a Gaussian interaction system, with nondegeneracy
discharged.**  For the anti-symmetric Gaussian-Gram interaction system at distinct
points, zero bilinear drift forces the probability coefficient vectors to agree.
Unlike `finiteCoefficientIdentifiable`, the nondegeneracy hypothesis is *derived*
here from the Gaussian being strictly positive definite. -/
theorem gaussianInteractionIdentifiesCoefficients (σ : ℝ) (hσ : ValidBandwidth σ)
    (x : StrictPair m → E) (hx : Function.Injective x)
    (a b : FiniteProbabilityVector m)
    (hzero : (∑ i, ∑ j,
      (a.weight i * b.weight j) • antisymmExtend (gaussianGram σ x) i j) = 0) :
    a.weight = b.weight :=
  normalizedParallelCoefficientsAreEqual m a b (fun i j =>
    coefficientMinorsVanish_of_antisymm (antisymmExtend (gaussianGram σ x))
      (antisymmExtend_anti (gaussianGram σ x))
      (gaussianInteraction_nondegenerate σ hσ x hx)
      a.weight b.weight hzero i j)

end Gaussian

end PaperFiniteIdentifiability
end DriftingIdentifiability
