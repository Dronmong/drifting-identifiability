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
