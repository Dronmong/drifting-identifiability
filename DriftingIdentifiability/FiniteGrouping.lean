import DriftingIdentifiability.CandidateConditions

/-!
# Deriving the coefficient-minor vanishing without the paper axiom

The coefficient-minor conclusion is elementary linear algebra, so it is proved
here from scratch: only anti-symmetry of the interaction family and a standard
Mathlib linear-independence fact are used. The former equation-(32) and
zero-minor axioms have been removed from the trusted boundary.

The key step is the grouping identity: an anti-symmetric bilinear sum over all
ordered pairs equals the minor-weighted sum over strict pairs.
-/

open scoped BigOperators

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

variable {V : Type u} [AddCommGroup V] [Module ℝ V] {m : ℕ}

/-- The minor-weighted summand `(aᵢbⱼ - aⱼbᵢ) • Uᵢⱼ`. -/
private def minorTerm (U : Fin m → Fin m → V) (a b : Fin m → ℝ) (i j : Fin m) : V :=
  (a i * b j - a j * b i) • U i j

/-- Under anti-symmetry the minor-weighted summand is symmetric in `i, j`. -/
private theorem minorTerm_symm (U : Fin m → Fin m → V)
    (hanti : ∀ i j, U i j = -U j i) (a b : Fin m → ℝ) (i j : Fin m) :
    minorTerm U a b i j = minorTerm U a b j i := by
  unfold minorTerm
  rw [hanti i j, smul_neg, ← neg_smul, neg_sub]

/-- The diagonal of the minor-weighted summand vanishes. -/
private theorem minorTerm_diag (U : Fin m → Fin m → V) (a b : Fin m → ℝ)
    (i : Fin m) : minorTerm U a b i i = 0 := by
  unfold minorTerm
  rw [sub_self, zero_smul]

/-- Grouping identity, doubled form: the ordered bilinear sum, doubled, equals
the full ordered sum of the minor-weighted terms. -/
private theorem two_smul_bilinear_eq_sum_minorTerm
    (U : Fin m → Fin m → V) (hanti : ∀ i j, U i j = -U j i) (a b : Fin m → ℝ) :
    (∑ i, ∑ j, minorTerm U a b i j)
      = (∑ i, ∑ j, (a i * b j) • U i j) + (∑ i, ∑ j, (a i * b j) • U i j) := by
  have hneg : (∑ i, ∑ j, (a j * b i) • U i j)
      = -(∑ i, ∑ j, (a i * b j) • U i j) := by
    rw [Finset.sum_comm]
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_neg_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [hanti j i, smul_neg]
  calc (∑ i, ∑ j, minorTerm U a b i j)
      = (∑ i, ∑ j, ((a i * b j) • U i j - (a j * b i) • U i j)) := by
        refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
        rw [minorTerm, sub_smul]
    _ = (∑ i, ∑ j, (a i * b j) • U i j) - (∑ i, ∑ j, (a j * b i) • U i j) := by
        rw [← Finset.sum_sub_distrib]
        refine Finset.sum_congr rfl fun i _ => ?_
        rw [Finset.sum_sub_distrib]
    _ = (∑ i, ∑ j, (a i * b j) • U i j) + (∑ i, ∑ j, (a i * b j) • U i j) := by
        rw [hneg, sub_neg_eq_add]

omit [Module ℝ V] in
/-- Trichotomy collapse: the full ordered sum of a symmetric, zero-diagonal
family equals twice the strict-upper sum. -/
theorem sum_symm_eq_two_upper
    (F : Fin m → Fin m → V) (hsym : ∀ i j, F i j = F j i)
    (hdiag : ∀ i, F i i = 0) :
    (∑ i, ∑ j, F i j)
      = (∑ i, ∑ j, if i < j then F i j else 0)
        + (∑ i, ∑ j, if i < j then F i j else 0) := by
  have hpoint : ∀ i j : Fin m, F i j
      = (if i < j then F i j else 0) + (if j < i then F i j else 0)
        + (if i = j then F i j else 0) := by
    intro i j
    rcases lt_trichotomy i j with h | h | h
    · rw [if_pos h, if_neg (lt_asymm h), if_neg (ne_of_lt h), add_zero, add_zero]
    · rw [if_neg (not_lt.mpr (le_of_eq h.symm)), if_neg (not_lt.mpr (le_of_eq h)),
        if_pos h, zero_add, zero_add]
    · rw [if_neg (not_lt.mpr h.le), if_pos h, if_neg (ne_of_lt h).symm, zero_add, add_zero]
  have hlower : (∑ i, ∑ j, if j < i then F i j else 0)
      = (∑ i, ∑ j, if i < j then F i j else 0) := by
    rw [Finset.sum_comm]
    refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
    by_cases h : i < j
    · rw [if_pos h, if_pos h, hsym]
    · rw [if_neg h, if_neg h]
  have hdiagSum : (∑ i, ∑ j : Fin m, if i = j then F i j else 0) = 0 := by
    refine Finset.sum_eq_zero fun i _ => ?_
    simp [hdiag]
  calc (∑ i, ∑ j, F i j)
      = ∑ i, ∑ j, ((if i < j then F i j else 0) + (if j < i then F i j else 0)
          + (if i = j then F i j else 0)) :=
        Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => hpoint i j
    _ = (∑ i, ∑ j, if i < j then F i j else 0)
          + (∑ i, ∑ j, if j < i then F i j else 0)
          + (∑ i, ∑ j, if i = j then F i j else 0) := by
        simp_rw [Finset.sum_add_distrib]
    _ = (∑ i, ∑ j, if i < j then F i j else 0)
          + (∑ i, ∑ j, if i < j then F i j else 0) := by
        rw [hlower, hdiagSum, add_zero]

omit [Module ℝ V] in
/-- The strict-upper double sum is the sum over the strict-pair index type. -/
theorem sum_if_lt_eq_strictPair (F : Fin m → Fin m → V) :
    (∑ i, ∑ j, if i < j then F i j else 0)
      = ∑ p : StrictPair m, F p.1.1 p.1.2 := by
  have key : (∑ p : StrictPair m, F p.1.1 p.1.2)
      = ∑ x ∈ Finset.univ.filter (fun q : Fin m × Fin m => q.1.val < q.2.val),
          F x.1 x.2 := by
    rw [← Finset.subtype_univ (fun q : Fin m × Fin m => q.1.val < q.2.val)]
    exact Finset.sum_subtype_eq_sum_filter (fun q : Fin m × Fin m => F q.1 q.2)
  rw [key, Finset.sum_filter, Fintype.sum_prod_type]
  rfl

/-- The ordered anti-symmetric bilinear sum is exactly the strict-pair
minor synthesis. -/
theorem bilinear_eq_sum_strictMinor
    (U : Fin m → Fin m → V) (hanti : ∀ i j, U i j = -U j i)
    (a b : Fin m → ℝ) :
    (∑ i, ∑ j, (a i * b j) • U i j) =
      ∑ p : StrictPair m,
        (a p.1.1 * b p.1.2 - a p.1.2 * b p.1.1) • U p.1.1 p.1.2 := by
  let B : V := ∑ i, ∑ j, (a i * b j) • U i j
  let S : V := ∑ p : StrictPair m,
    (a p.1.1 * b p.1.2 - a p.1.2 * b p.1.1) • U p.1.1 p.1.2
  have hfull : (∑ i, ∑ j, minorTerm U a b i j) = B + B := by
    exact two_smul_bilinear_eq_sum_minorTerm U hanti a b
  have hstrict : (∑ i, ∑ j, minorTerm U a b i j) = S + S := by
    rw [sum_symm_eq_two_upper (minorTerm U a b) (minorTerm_symm U hanti a b)
      (minorTerm_diag U a b)]
    simp only [sum_if_lt_eq_strictPair, minorTerm, S]
  have htwo : B + B = S + S := hfull.symm.trans hstrict
  have hscaled := congrArg (fun v : V => (2 : ℝ)⁻¹ • v) htwo
  simpa only [← two_smul ℝ, smul_smul, inv_mul_cancel₀ (by norm_num : (2 : ℝ) ≠ 0),
    one_smul, B, S] using hscaled

/-- **Coefficient minors vanish — proved, not assumed.**  This reproduces the
content of the paper's zero-minor conclusion using only
anti-symmetry of the interaction family and Mathlib's linear-independence
characterization.  No paper axiom is used. -/
theorem coefficientMinorsVanish_of_antisymm
    (U : Fin m → Fin m → V) (hanti : ∀ i j, U i j = -U j i)
    (hnondeg : LinearIndependent ℝ (fun p : StrictPair m => U p.1.1 p.1.2))
    (a b : Fin m → ℝ)
    (hzero : (∑ i, ∑ j, (a i * b j) • U i j) = 0) :
    ∀ i j, a i * b j - a j * b i = 0 := by
  have hmt0 : (∑ i, ∑ j, minorTerm U a b i j) = 0 := by
    rw [two_smul_bilinear_eq_sum_minorTerm U hanti a b, hzero, add_zero]
  have hup0 : (∑ i, ∑ j, if i < j then minorTerm U a b i j else 0)
      + (∑ i, ∑ j, if i < j then minorTerm U a b i j else 0) = 0 := by
    rw [← sum_symm_eq_two_upper (minorTerm U a b) (minorTerm_symm U hanti a b)
      (minorTerm_diag U a b)]
    exact hmt0
  have hupper0 : (∑ i, ∑ j, if i < j then minorTerm U a b i j else 0) = 0 := by
    have h2 : (2 : ℝ) • (∑ i, ∑ j, if i < j then minorTerm U a b i j else 0) = 0 := by
      rw [two_smul]; exact hup0
    have h3 := congrArg (fun v => (2 : ℝ)⁻¹ • v) h2
    simp only [smul_smul, smul_zero] at h3
    rwa [inv_mul_cancel₀ (two_ne_zero), one_smul] at h3
  have hSP : (∑ p : StrictPair m,
      (a p.1.1 * b p.1.2 - a p.1.2 * b p.1.1) • U p.1.1 p.1.2) = 0 := by
    rw [sum_if_lt_eq_strictPair (minorTerm U a b)] at hupper0
    simpa only [minorTerm] using hupper0
  have hcoef := (Fintype.linearIndependent_iff.mp hnondeg)
    (fun p => a p.1.1 * b p.1.2 - a p.1.2 * b p.1.1) hSP
  intro i j
  rcases lt_trichotomy i j with h | h | h
  · exact hcoef ⟨(i, j), h⟩
  · subst h; ring
  · have := hcoef ⟨(j, i), h⟩
    linarith [this]

end PaperFiniteIdentifiability
end DriftingIdentifiability
