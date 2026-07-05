import DriftingIdentifiability.PopulationIdentifiability

/-!
# Closed-form interaction vectors for an empirical reference

The finite population theorem assumes a positive `InteractionFrameBound` for the
paper's *actual* integral-induced interaction vectors.  Here we discharge that
obligation for a concrete, implementable model — a two-point empirical reference
with point-mass basis — by computing the double integral in closed form:

```text
Uᵢⱼ(n) = k(xₙ,zᵢ) · k(xₙ,zⱼ) · (zᵢ − zⱼ).
```

This is the actual `basisInteraction` of the mean-shift interaction kernel, not a
synthetic substitute.
-/

open scoped BigOperators ENNReal
open MeasureTheory

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

variable {E : Type u} [MeasurableSpace E] [MeasurableSingletonClass E]
  [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- The uniform empirical distribution on two points. -/
noncomputable def empirical2 (z0 z1 : E) : Distribution E :=
  (2⁻¹ : ℝ≥0∞) • (Measure.dirac z0 + Measure.dirac z1)

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Integral against the two-point empirical distribution is the average of the
two point evaluations. -/
theorem integral_empirical2 {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    [CompleteSpace F] (z0 z1 : E) (f : E → F) :
    ∫ y, f y ∂(empirical2 z0 z1) = (2⁻¹ : ℝ) • (f z0 + f z1) := by
  rw [empirical2, integral_smul_measure,
    integral_add_measure (integrable_dirac (by simp)) (integrable_dirac (by simp)),
    integral_dirac, integral_dirac]
  simp

/-- **Closed form of the actual integral-induced interaction.**  Against the
two-point empirical reference, the mean-shift interaction double integral of
equation (30) collapses to an explicit expression: the coefficient minor of the
basis values times the product of the two kernel values, along the direction
`z₀ − z₁`. -/
theorem basisInteraction_empirical2 (k : E → E → ℝ) (φi φj : E → ℝ) (z0 z1 x : E) :
    basisInteraction (empirical2 z0 z1) (meanShiftInteractionKernel k) φi φj x
      = ((2⁻¹ * 2⁻¹) * (φi z0 * φj z1 - φi z1 * φj z0) * (k x z0 * k x z1)) •
          (z0 - z1) := by
  unfold basisInteraction
  simp_rw [integral_empirical2, meanShiftInteractionKernel]
  simp only [sub_self, smul_zero, add_zero, zero_add, smul_smul]
  module

/-! ## The `m = 2` frame bound -/

/-- The strict-pair index set for `m = 2` is a singleton. -/
instance : Unique (StrictPair 2) where
  default := ⟨(0, 1), by decide⟩
  uniq := by decide

omit [MeasurableSpace E] [MeasurableSingletonClass E] [CompleteSpace E] in
/-- For `m = 2` a single nonzero interaction vector already gives a positive
frame bound (with constant its own norm). -/
theorem interactionFrameBound_two {N : ℕ} (U : Fin 2 → Fin 2 → Fin N → E)
    (hU : U 0 1 ≠ 0) : InteractionFrameBound U ‖U 0 1‖ := by
  refine ⟨norm_pos_iff.mpr hU, fun z => ?_⟩
  rw [interactionSynthesis_apply, Fintype.sum_unique, Fintype.sum_unique,
    norm_smul, Real.norm_eq_abs]
  change ‖U 0 1‖ * |z default| ≤ |z default| * ‖U 0 1‖
  rw [mul_comm]

/-- **Nondegeneracy gives a positive frame bound (general `m`).**  For any
linearly independent interaction family over a nonempty strict-pair index set,
some positive `InteractionFrameBound` holds.  With the reverse implication
(`interactionFrameBound_linearIndependent`), the frame-bound hypothesis is thus
*equivalent* to qualitative nondegeneracy: the general-`m` frame gap reduces to
pure linear independence of the induced vectors. -/
theorem interactionFrameBound_of_linearIndependent
    {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] {m : ℕ}
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
  have hbound : ∀ z : StrictPair m → ℝ, ‖z‖ ≤ (K : ℝ) * ‖interactionSynthesis U z‖ := by
    intro z
    have h := hanti.le_mul_dist z 0
    rwa [dist_zero_right, map_zero, dist_zero_right] at h
  have hNpos : (0 : ℝ) < (Fintype.card (StrictPair m) : ℝ) := by
    exact_mod_cast Fintype.card_pos
  have hKR : (0 : ℝ) < (K : ℝ) := hKpos
  have hprod : (0 : ℝ) < (K : ℝ) * Fintype.card (StrictPair m) := mul_pos hKR hNpos
  refine ⟨((K : ℝ) * Fintype.card (StrictPair m))⁻¹, by positivity, by positivity, fun z => ?_⟩
  have hsum : (∑ p, |z p|) ≤ (Fintype.card (StrictPair m) : ℝ) * ‖z‖ := by
    calc (∑ p : StrictPair m, |z p|) ≤ ∑ _p : StrictPair m, ‖z‖ := by
          refine Finset.sum_le_sum fun p _ => ?_
          rw [← Real.norm_eq_abs]; exact norm_le_pi_norm z p
      _ = (Fintype.card (StrictPair m) : ℝ) * ‖z‖ := by
          rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hchain : (∑ p, |z p|)
      ≤ (K : ℝ) * Fintype.card (StrictPair m) * ‖interactionSynthesis U z‖ :=
    calc (∑ p, |z p|) ≤ (Fintype.card (StrictPair m) : ℝ) * ‖z‖ := hsum
      _ ≤ (Fintype.card (StrictPair m) : ℝ) * ((K : ℝ) * ‖interactionSynthesis U z‖) :=
          mul_le_mul_of_nonneg_left (hbound z) hNpos.le
      _ = (K : ℝ) * Fintype.card (StrictPair m) * ‖interactionSynthesis U z‖ := by ring
  calc ((K : ℝ) * Fintype.card (StrictPair m))⁻¹ * (∑ p, |z p|)
      ≤ ((K : ℝ) * Fintype.card (StrictPair m))⁻¹ *
          ((K : ℝ) * Fintype.card (StrictPair m) * ‖interactionSynthesis U z‖) :=
        mul_le_mul_of_nonneg_left hchain (by positivity)
    _ = ‖interactionSynthesis U z‖ := by
        rw [← mul_assoc, inv_mul_cancel₀ (ne_of_gt hprod), one_mul]

/-! ## Positive frame bound for the actual induced vectors -/

variable {N : ℕ}

/-- For the two-point empirical reference and Gaussian kernel, the actual
integral-induced interaction vector `U₀₁` is nonzero as soon as the two points
differ and the basis value minor is nonzero (both concretely checkable). -/
theorem inducedInteractionVector_empirical2_ne_zero
    (σ : ℝ) (z0 z1 : E) (hz : z0 ≠ z1)
    (φ : Fin 2 → E → ℝ) (probes : Fin N → E) (n0 : Fin N)
    (hminor : φ 0 z0 * φ 1 z1 - φ 0 z1 * φ 1 z0 ≠ 0) :
    inducedInteractionVector (empirical2 z0 z1)
      (meanShiftInteractionKernel (gaussianKernel σ)) φ probes 0 1 ≠ 0 := by
  rw [Function.ne_iff]
  refine ⟨n0, ?_⟩
  simp only [inducedInteractionVector, basisInteraction_empirical2, Pi.zero_apply]
  apply smul_ne_zero
  · have h0 : (0 : ℝ) < gaussianKernel σ (probes n0) z0 := by
      rw [gaussianKernel]; exact Real.exp_pos _
    have h1 : (0 : ℝ) < gaussianKernel σ (probes n0) z1 := by
      rw [gaussianKernel]; exact Real.exp_pos _
    exact mul_ne_zero (mul_ne_zero (by norm_num) hminor) (ne_of_gt (mul_pos h0 h1))
  · exact sub_ne_zero.mpr hz

/-- **Gap closed for `m = 2`.**  The paper's *actual* integral-induced interaction
family — for a two-point empirical reference, a Gaussian kernel, distinct points,
and a basis with nonzero value minor — satisfies a positive
`InteractionFrameBound`.  This discharges the nondegeneracy/frame hypothesis of
the mean-shift population theorem in a concrete implementable model, using the
real `basisInteraction` double integral (not a synthetic substitute) and no
external axiom. -/
theorem empiricalInteractionFrameBound
    (σ : ℝ) (z0 z1 : E) (hz : z0 ≠ z1)
    (φ : Fin 2 → E → ℝ) (probes : Fin N → E) (n0 : Fin N)
    (hminor : φ 0 z0 * φ 1 z1 - φ 0 z1 * φ 1 z0 ≠ 0) :
    InteractionFrameBound (inducedInteractionVector (empirical2 z0 z1)
      (meanShiftInteractionKernel (gaussianKernel σ)) φ probes)
      ‖inducedInteractionVector (empirical2 z0 z1)
        (meanShiftInteractionKernel (gaussianKernel σ)) φ probes 0 1‖ :=
  interactionFrameBound_two _
    (inducedInteractionVector_empirical2_ne_zero σ z0 z1 hz φ probes n0 hminor)

end PaperFiniteIdentifiability
end DriftingIdentifiability
