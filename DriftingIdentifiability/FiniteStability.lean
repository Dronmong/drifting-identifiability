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
probability normalization.  The second half packages a checkable positive
frame bound and uses it to control coefficient error by drift error.
-/

open scoped BigOperators
open Filter Topology

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

variable {V : Type u} [NormedAddCommGroup V] [NormedSpace ℝ V]
  {m : ℕ}

/-- Coefficient minors restricted to the independent strict-pair index set. -/
def strictMinorVector (a b : Fin m → ℝ) (p : StrictPair m) : ℝ :=
  coefficientMinor a b p.1.1 p.1.2

/-- Linear synthesis by the strict-pair interaction vectors. -/
def interactionSynthesis (U : Fin m → Fin m → V) :
    (StrictPair m → ℝ) →ₗ[ℝ] V where
  toFun z := ∑ p, z p • U p.1.1 p.1.2
  map_add' z w := by
    simp only [Pi.add_apply, add_smul, Finset.sum_add_distrib]
  map_smul' r z := by
    simp only [Pi.smul_apply, RingHom.id_apply, smul_smul, Finset.smul_sum,
      smul_eq_mul]

@[simp]
theorem interactionSynthesis_apply (U : Fin m → Fin m → V)
    (z : StrictPair m → ℝ) :
    interactionSynthesis U z = ∑ p, z p • U p.1.1 p.1.2 := rfl

/-- A quantitative and numerically testable replacement for bare linear
independence. `c` is a lower frame bound in the coefficient `ℓ¹` norm. -/
def InteractionFrameBound (U : Fin m → Fin m → V) (c : ℝ) : Prop :=
  0 < c ∧ ∀ z : StrictPair m → ℝ,
    c * (∑ p, |z p|) ≤ ‖interactionSynthesis U z‖

/-- A positive frame bound implies qualitative linear independence. -/
theorem interactionFrameBound_linearIndependent
    (U : Fin m → Fin m → V) {c : ℝ} (hframe : InteractionFrameBound U c) :
    LinearIndependent ℝ (fun p : StrictPair m => U p.1.1 p.1.2) := by
  rw [Fintype.linearIndependent_iff]
  intro z hz p
  have hbound := hframe.2 z
  rw [interactionSynthesis_apply, hz, norm_zero] at hbound
  have hmass : (∑ q, |z q|) = 0 := by
    have hnonneg : 0 ≤ ∑ q, |z q| := Finset.sum_nonneg fun _ _ => abs_nonneg _
    nlinarith [hframe.1]
  have hp : |z p| = 0 := by
    exact (Finset.sum_eq_zero_iff_of_nonneg (fun q _ => abs_nonneg (z q))).mp hmass p
      (Finset.mem_univ p)
  exact abs_eq_zero.mp hp

/-- In finite dimension, qualitative nondegeneracy also yields some positive
frame constant.  This is an existence theorem; practical stability still
requires estimating a useful value of the constant. -/
theorem interactionFrameBound_of_linearIndependent
    [Nonempty (StrictPair m)] (U : Fin m → Fin m → V)
    (hindep : LinearIndependent ℝ (fun p : StrictPair m => U p.1.1 p.1.2)) :
    ∃ c > 0, InteractionFrameBound U c := by
  have hker : LinearMap.ker (interactionSynthesis U) = ⊥ := by
    rw [LinearMap.ker_eq_bot']
    intro z hz
    rw [interactionSynthesis_apply] at hz
    funext p
    exact (Fintype.linearIndependent_iff.mp hindep) z hz p
  obtain ⟨K, hKpos, hanti⟩ :=
    LinearMap.exists_antilipschitzWith (interactionSynthesis U) hker
  have hbound : ∀ z : StrictPair m → ℝ,
      ‖z‖ ≤ (K : ℝ) * ‖interactionSynthesis U z‖ := by
    intro z
    have h := hanti.le_mul_dist z 0
    rwa [dist_zero_right, map_zero, dist_zero_right] at h
  have hNpos : (0 : ℝ) < (Fintype.card (StrictPair m) : ℝ) := by
    exact_mod_cast Fintype.card_pos
  have hKR : (0 : ℝ) < (K : ℝ) := hKpos
  have hprod : (0 : ℝ) < (K : ℝ) * Fintype.card (StrictPair m) :=
    mul_pos hKR hNpos
  refine ⟨((K : ℝ) * Fintype.card (StrictPair m))⁻¹,
    by positivity, by positivity, fun z => ?_⟩
  have hsum : (∑ p, |z p|) ≤
      (Fintype.card (StrictPair m) : ℝ) * ‖z‖ := by
    calc
      (∑ p : StrictPair m, |z p|) ≤ ∑ _p : StrictPair m, ‖z‖ := by
        refine Finset.sum_le_sum fun p _ => ?_
        rw [← Real.norm_eq_abs]
        exact norm_le_pi_norm z p
      _ = (Fintype.card (StrictPair m) : ℝ) * ‖z‖ := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hchain : (∑ p, |z p|) ≤
      (K : ℝ) * Fintype.card (StrictPair m) * ‖interactionSynthesis U z‖ :=
    calc
      (∑ p, |z p|) ≤ (Fintype.card (StrictPair m) : ℝ) * ‖z‖ := hsum
      _ ≤ (Fintype.card (StrictPair m) : ℝ) *
          ((K : ℝ) * ‖interactionSynthesis U z‖) :=
        mul_le_mul_of_nonneg_left (hbound z) hNpos.le
      _ = (K : ℝ) * Fintype.card (StrictPair m) *
          ‖interactionSynthesis U z‖ := by ring
  calc
    ((K : ℝ) * Fintype.card (StrictPair m))⁻¹ * (∑ p, |z p|)
        ≤ ((K : ℝ) * Fintype.card (StrictPair m))⁻¹ *
            ((K : ℝ) * Fintype.card (StrictPair m) *
              ‖interactionSynthesis U z‖) :=
          mul_le_mul_of_nonneg_left hchain (by positivity)
    _ = ‖interactionSynthesis U z‖ := by
      rw [← mul_assoc, inv_mul_cancel₀ (ne_of_gt hprod), one_mul]

/-- **Ceiling on the frame constant.**  Testing the frame inequality on the
`p`-th coordinate indicator `Pi.single p 1` shows every valid frame constant is
at most the norm of the corresponding interaction vector.  Consequently
`c ≤ min_p ‖U_p‖`: a large frame constant *requires* every interaction vector to
be large, and closeness of any interaction vector to zero caps the achievable
conditioning.  This is the exact, computable upper bound complementing the
existential lower bound of `interactionFrameBound_of_linearIndependent`. -/
theorem interactionFrameBound_le_interactionNorm
    (U : Fin m → Fin m → V) {c : ℝ} (hframe : InteractionFrameBound U c)
    (p : StrictPair m) : c ≤ ‖U p.1.1 p.1.2‖ := by
  have hb := hframe.2 (Pi.single p 1 : StrictPair m → ℝ)
  rw [interactionSynthesis_apply] at hb
  have hsum : (∑ q, (Pi.single p 1 : StrictPair m → ℝ) q • U q.1.1 q.1.2)
      = U p.1.1 p.1.2 := by
    rw [Finset.sum_eq_single p
      (fun q _ hqp => by rw [Pi.single_eq_of_ne hqp, zero_smul])
      (fun h => absurd (Finset.mem_univ p) h), Pi.single_eq_same, one_smul]
  have habs : (∑ q, |(Pi.single p 1 : StrictPair m → ℝ) q|) = 1 := by
    have hpt : ∀ q, |(Pi.single p 1 : StrictPair m → ℝ) q|
        = (Pi.single p 1 : StrictPair m → ℝ) q := by
      intro q
      rcases eq_or_ne q p with h | h
      · subst h; rw [Pi.single_eq_same]; exact abs_one
      · rw [Pi.single_eq_of_ne h]; exact abs_zero
    simp_rw [hpt]
    rw [Finset.sum_pi_single']
    simp
  rw [hsum, habs, mul_one] at hb
  exact hb

/-- A finite family of nonzero row/column scalings of distinct geometric
profiles is linearly independent.  This is the axiom-free Vandermonde engine
used by the structured Gaussian probe construction. -/
theorem linearIndependent_weightedGeometricProfiles
    {ι : Type*} [Fintype ι] (r column : ι → ℝ)
    (hr : Function.Injective r) (hcolumn : ∀ i, column i ≠ 0)
    (row : Fin (Fintype.card ι) → ℝ) (hrow : ∀ n, row n ≠ 0) :
    LinearIndependent ℝ (fun i : ι => fun n =>
      row n * (column i * r i ^ (n : ℕ))) := by
  rw [Fintype.linearIndependent_iff]
  intro a ha i
  let e : ι ≃ Fin (Fintype.card ι) := Fintype.equivFin ι
  let v : Fin (Fintype.card ι) → ℝ := fun q =>
    a (e.symm q) * column (e.symm q)
  let f : Fin (Fintype.card ι) → ℝ := fun q => r (e.symm q)
  have hf : Function.Injective f := hr.comp e.symm.injective
  have hpower : ∀ n : Fin (Fintype.card ι),
      (∑ q, v q * f q ^ (n : ℕ)) = 0 := by
    intro n
    have hn := congrFun ha n
    simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Pi.zero_apply] at hn
    have hfactored : row n *
        (∑ p, a p * column p * r p ^ (n : ℕ)) = 0 := by
      calc
        row n * (∑ p, a p * column p * r p ^ (n : ℕ)) =
            ∑ p, a p * (row n * (column p * r p ^ (n : ℕ))) := by
          rw [Finset.mul_sum]
          apply Finset.sum_congr rfl
          intro p _
          ring
        _ = 0 := hn
    have hunscaled : (∑ p, a p * column p * r p ^ (n : ℕ)) = 0 :=
      (mul_eq_zero.mp hfactored).resolve_left (hrow n)
    let g : ι → ℝ := fun p => a p * column p * r p ^ (n : ℕ)
    change (∑ q, g (e.symm q)) = 0
    rw [e.symm.sum_comp]
    exact hunscaled
  have hv : v = 0 :=
    Matrix.eq_zero_of_forall_pow_sum_mul_pow_eq_zero hf hpower
  have hvi := congrFun hv (e i)
  simp only [v, Equiv.symm_apply_apply, Pi.zero_apply] at hvi
  exact (mul_eq_zero.mp hvi).resolve_right (hcolumn i)

/-- Specialization of the preceding result to the paper's probe-vector
interaction family. -/
theorem interactionFrameBound_basisNondegenerate
    {W : Type*} [NormedAddCommGroup W] [NormedSpace ℝ W] {N : ℕ}
    (U : Fin m → Fin m → Fin N → W) {c : ℝ}
    (hframe : InteractionFrameBound U c) : BasisInteractionNondegenerate U :=
  interactionFrameBound_linearIndependent U hframe

/-- Necessary dimension budget: the number of strict interaction pairs cannot
exceed the dimension available across all probes. -/
theorem nondegenerate_pairCount_le_probeDimension
    {W : Type*} [NormedAddCommGroup W] [NormedSpace ℝ W]
    [FiniteDimensional ℝ W] {N : ℕ}
    (U : Fin m → Fin m → Fin N → W) (h : BasisInteractionNondegenerate U) :
    Fintype.card (StrictPair m) ≤ N * Module.finrank ℝ W := by
  have hdim := h.fintype_card_le_finrank
  simpa [Module.finrank_pi_fintype] using hdim

/-- The full ordered minor mass is twice the strict-pair minor mass. -/
theorem minorMass_eq_two_strictMinorMass (a b : Fin m → ℝ) :
    (∑ i, ∑ j, |coefficientMinor a b i j|) =
      2 * ∑ p : StrictPair m, |strictMinorVector a b p| := by
  have hsym : ∀ i j : Fin m,
      |coefficientMinor a b i j| = |coefficientMinor a b j i| := by
    intro i j
    unfold coefficientMinor
    rw [show a j * b i - a i * b j = -(a i * b j - a j * b i) by ring, abs_neg]
  have hdiag : ∀ i : Fin m, |coefficientMinor a b i i| = 0 := by
    intro i
    simp [coefficientMinor]
  rw [sum_symm_eq_two_upper (fun i j => |coefficientMinor a b i j|) hsym hdiag]
  rw [sum_if_lt_eq_strictPair]
  simp only [strictMinorVector, two_mul]

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

/-- A frame lower bound turns drift control into coefficient control without
dividing by the conditioning constant. -/
theorem frame_mul_coeffL1_le_two_mul_driftNorm
    (U : Fin m → Fin m → V) (hanti : ∀ i j, U i j = -U j i)
    {c : ℝ} (hframe : InteractionFrameBound U c)
    (a b : FiniteProbabilityVector m) :
    c * (∑ i, |a.weight i - b.weight i|) ≤
      2 * ‖∑ i, ∑ j, (a.weight i * b.weight j) • U i j‖ := by
  have hcoeff := coeff_l1_le_minor_mass m a b
  rw [minorMass_eq_two_strictMinorMass a.weight b.weight] at hcoeff
  have hbound := hframe.2 (strictMinorVector a.weight b.weight)
  rw [interactionSynthesis_apply] at hbound
  simp only [strictMinorVector, coefficientMinor] at hbound
  rw [← bilinear_eq_sum_strictMinor U hanti] at hbound
  have hc := mul_le_mul_of_nonneg_left hcoeff hframe.1.le
  simp only [strictMinorVector, coefficientMinor] at hc
  nlinarith

/-- Division form of the practical stability estimate. -/
theorem coeffL1_le_two_div_frame_mul_driftNorm
    (U : Fin m → Fin m → V) (hanti : ∀ i j, U i j = -U j i)
    {c : ℝ} (hframe : InteractionFrameBound U c)
    (a b : FiniteProbabilityVector m) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 / c) * ‖∑ i, ∑ j, (a.weight i * b.weight j) • U i j‖ := by
  have h := frame_mul_coeffL1_le_two_mul_driftNorm U hanti hframe a b
  rw [div_mul_eq_mul_div]
  apply (le_div_iff₀ hframe.1).2
  nlinarith

/-- Stability after a bounded pointwise rescaling, the form needed for the
normalizers in the paper's mean-shift field. -/
theorem coeffL1_le_of_frame_scaledDrift
    {N : ℕ} (U : Fin m → Fin m → Fin N → V)
    (hanti : ∀ i j, U i j = -U j i)
    {c B : ℝ} (hframe : InteractionFrameBound U c) (hB0 : 0 ≤ B)
    (a b : FiniteProbabilityVector m) (scale : Fin N → ℝ) (v : Fin N → V)
    (hscale : ∀ n, |scale n| ≤ B)
    (hbilinear : (∑ i, ∑ j, (a.weight i * b.weight j) • U i j) =
      fun n => scale n • v n) :
    (∑ i, |a.weight i - b.weight i|) ≤ (2 * B / c) * ‖v‖ := by
  have hcoeff := coeffL1_le_two_div_frame_mul_driftNorm U hanti hframe a b
  rw [hbilinear] at hcoeff
  have hnorm : ‖fun n => scale n • v n‖ ≤ B * ‖v‖ := by
    apply (pi_norm_le_iff_of_nonneg (mul_nonneg hB0 (norm_nonneg v))).2
    intro n
    rw [norm_smul]
    exact mul_le_mul (hscale n) (norm_le_pi_norm v n) (norm_nonneg _) hB0
  calc
    (∑ i, |a.weight i - b.weight i|)
        ≤ (2 / c) * ‖fun n => scale n • v n‖ := hcoeff
    _ ≤ (2 / c) * (B * ‖v‖) :=
      mul_le_mul_of_nonneg_left hnorm (div_nonneg (by norm_num) hframe.1.le)
    _ = (2 * B / c) * ‖v‖ := by ring

/-- Uniform frame and scaling bounds turn vanishing probe drift into
coefficient convergence. -/
theorem coeffL1_tendsto_zero_of_frame_scaledDrift
    {N : ℕ} (U : Fin m → Fin m → Fin N → V)
    (hanti : ∀ i j, U i j = -U j i)
    {c B : ℝ} (hframe : InteractionFrameBound U c) (hB0 : 0 ≤ B)
    (a : FiniteProbabilityVector m) (b : ℕ → FiniteProbabilityVector m)
    (scale : ℕ → Fin N → ℝ) (v : ℕ → Fin N → V)
    (hscale : ∀ n i, |scale n i| ≤ B)
    (hbilinear : ∀ n,
      (∑ i, ∑ j, (a.weight i * (b n).weight j) • U i j) =
        fun i => scale n i • v n i)
    (hv : Tendsto (fun n => ‖v n‖) atTop (𝓝 0)) :
    Tendsto (fun n => ∑ i, |a.weight i - (b n).weight i|) atTop (𝓝 0) := by
  have hupper : Tendsto (fun n => (2 * B / c) * ‖v n‖) atTop (𝓝 0) := by
    simpa using tendsto_const_nhds.mul hv
  exact squeeze_zero
    (fun _ => Finset.sum_nonneg fun _ _ => abs_nonneg _)
    (fun n => coeffL1_le_of_frame_scaledDrift U hanti hframe hB0 a (b n)
      (scale n) (v n) (hscale n) (hbilinear n))
    hupper

end PaperFiniteIdentifiability
end DriftingIdentifiability
