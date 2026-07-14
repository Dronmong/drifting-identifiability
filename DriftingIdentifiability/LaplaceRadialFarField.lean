import DriftingIdentifiability.LaplaceRadialFoundations

/-!
# Radial Laplace converse, milestone L1: the far-field foundation

This file is milestone `L1` of the ℓ²/radial higher-dimensional Laplace program
(`LaplaceHigherDim.md`, §4.8/§4.6(a), D2.a).  Its headline is
`zeroDrift_tiltedCentroid_eq`: pointwise zero Laplace mean-shift drift forces the
two exponential-tilt centroids to agree in **every** direction, for arbitrary
probability measures with an exponential moment — the n-d analogue of the 1-d
"radial-limit foundation".  The proof reads off the drift along a radial probe
`r • u`: since `meanShift = kernelCentroid − probe` and the probe cancels in the
difference, zero drift makes the two kernel centroids agree at every finite `r`,
and passing `r → ∞` (existing `kernelCentroid_laplace_radial_tendsto`) turns the
kernel centroids into the exponential-tilt centroids.

Also recorded is `laplaceCompensatedWeight_monotone` (§4.6(a)): the compensated
weight `e^{(r-‖r•u-y‖)/τ}` is monotone in `r`, the moment-minimal ingredient of
the far-field limits.

Design record: `LaplaceHigherDim.md`, §4.6(a) and §4.8 (L1).
-/

open MeasureTheory Filter Topology
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ## Monotonicity of the compensated weight (§4.6(a)) -/

/-- The compensated weight `r ↦ e^{(r-‖r•u-y‖)/τ}` is **monotone**: the exponent
`r - ‖r•u - y‖` increases because `‖r'•u-y‖ - ‖r•u-y‖ ≤ ‖(r'-r)•u‖ = r'-r`.  This
is the moment-free ingredient behind the far-field limits (the weight increases
to its exponential-tilt limit, so monotone/dominated convergence applies without
any moment hypothesis on the normalizer). -/
lemma laplaceCompensatedWeight_monotone {τ : ℝ} (hτ : 0 < τ) (u y : E) (hu : ‖u‖ = 1) :
    Monotone (fun r : ℝ => laplaceCompensatedWeight τ r u y) := by
  intro r r' hrr
  simp only [laplaceCompensatedWeight]
  apply Real.exp_le_exp.mpr
  apply div_le_div_of_nonneg_right _ hτ.le
  have hchain : ‖r' • u - y‖ - ‖r • u - y‖ ≤ r' - r := by
    calc ‖r' • u - y‖ - ‖r • u - y‖
        ≤ ‖(r' • u - y) - (r • u - y)‖ := norm_sub_norm_le _ _
      _ = ‖(r' - r) • u‖ := by rw [sub_sub_sub_cancel_right, ← sub_smul]
      _ = (r' - r) * ‖u‖ := by
          rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by linarith)]
      _ = r' - r := by rw [hu, mul_one]
  linarith

section FarField

variable [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]

omit [InnerProductSpace ℝ E] [CompleteSpace E] [SecondCountableTopology E] in
/-- An exponential moment `∫ e^{‖y‖/τ}‖y‖ < ∞` implies the first moment
`∫ ‖y‖ < ∞`, since `e^{‖y‖/τ} ≥ 1`. -/
lemma integrable_norm_of_integrable_exp_mul_norm {τ : ℝ} (hτ : 0 < τ) (P : Measure E)
    (h : Integrable (fun y => Real.exp (‖y‖ / τ) * ‖y‖) P) :
    Integrable (fun y : E => ‖y‖) P := by
  refine h.mono' continuous_norm.aestronglyMeasurable ?_
  filter_upwards with y
  rw [Real.norm_eq_abs, abs_of_nonneg (norm_nonneg _)]
  exact le_mul_of_one_le_left (norm_nonneg _)
    (Real.one_le_exp (div_nonneg (norm_nonneg _) hτ.le))

omit [CompleteSpace E] in
/-- The displacement-numerator integrand `laplaceKernel τ x · • ·` is integrable
against any measure with a first moment (`e^{-‖x-y‖/τ} ≤ 1`). -/
lemma laplaceKernel_smul_id_integrable_of_moment {τ : ℝ} (hτ : 0 < τ) (P : Measure E)
    (x : E) (hmom : Integrable (fun y : E => ‖y‖) P) :
    Integrable (fun y => laplaceKernel τ x y • y) P := by
  refine hmom.mono' ?_ ?_
  · apply Continuous.aestronglyMeasurable; unfold laplaceKernel; fun_prop
  · filter_upwards with y
    rw [norm_smul, laplaceKernel_eq_exp, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    have hle : Real.exp (-‖x - y‖ / τ) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      exact div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (norm_nonneg _)) hτ.le
    calc Real.exp (-‖x - y‖ / τ) * ‖y‖ ≤ 1 * ‖y‖ :=
          mul_le_mul_of_nonneg_right hle (norm_nonneg _)
      _ = ‖y‖ := one_mul _

/-- The paper's mean shift equals `kernelCentroid − probe` for any probability
measure with a first moment (general form of
`gaussian_meanShift_laplace_eq_kernelCentroid_sub`, no Gaussian hypothesis). -/
lemma meanShift_laplace_eq_kernelCentroid_sub_of_moment {τ : ℝ} (hτ : 0 < τ)
    (P : Measure E) [IsProbabilityMeasure P] (x : E)
    (hmom : Integrable (fun y : E => ‖y‖) P) :
    meanShift (laplaceKernel τ) P x = kernelCentroid (laplaceKernel τ) P x - x :=
  meanShift_eq_kernelCentroid_sub P (laplaceKernel τ) x
    (laplaceKernel_integrable P τ hτ x)
    (laplaceKernel_smul_id_integrable_of_moment hτ P x hmom)
    (laplaceKernelNormalizer_pos P τ hτ x).ne'

/-- **Far-field foundation (D2.a).**  Pointwise zero Laplace mean-shift drift
forces the exponential-tilt centroids of `P` and `Q` to agree in every unit
direction `u`, for arbitrary probability measures with the exponential moments
`∫ e^{‖y‖/τ}(1+‖y‖) < ∞`.  Proof: along the ray `r • u`, `meanShift =
kernelCentroid − r•u`, so zero drift makes the kernel centroids agree at every
`r`; letting `r → ∞` (via `kernelCentroid_laplace_radial_tendsto`) sends each to
its exponential-tilt centroid, and limits are unique.  This is the n-d analogue
of the 1-d radial-limit foundation; it is known to be *insufficient alone* (an
analytic difference of cumulant generating functions can have vanishing gradient
on a sphere), but it is shared infrastructure for the radial endgame (L5). -/
theorem zeroDrift_tiltedCentroid_eq (P Q : Measure E)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {τ : ℝ} (hτ : 0 < τ) (u : E) (hu : ‖u‖ = 1)
    (hP₀ : Integrable (fun y => Real.exp (‖y‖ / τ)) P)
    (hP₁ : Integrable (fun y => Real.exp (‖y‖ / τ) * ‖y‖) P)
    (hQ₀ : Integrable (fun y => Real.exp (‖y‖ / τ)) Q)
    (hQ₁ : Integrable (fun y => Real.exp (‖y‖ / τ) * ‖y‖) Q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) P Q) :
    exponentialTiltCentroid P τ u = exponentialTiltCentroid Q τ u := by
  have hPmom := integrable_norm_of_integrable_exp_mul_norm hτ P hP₁
  have hQmom := integrable_norm_of_integrable_exp_mul_norm hτ Q hQ₁
  have hP := kernelCentroid_laplace_radial_tendsto P τ hτ u hu hP₀ hP₁
  have hQ := kernelCentroid_laplace_radial_tendsto Q τ hτ u hu hQ₀ hQ₁
  have heq : (fun r : ℝ => kernelCentroid (laplaceKernel τ) P (r • u)) =
      fun r : ℝ => kernelCentroid (laplaceKernel τ) Q (r • u) := by
    funext r
    have hms : meanShift (laplaceKernel τ) P (r • u) =
        meanShift (laplaceKernel τ) Q (r • u) := by
      have h := hzero (r • u)
      simp only [meanShiftDrift, sub_eq_zero] at h
      exact h
    rw [meanShift_laplace_eq_kernelCentroid_sub_of_moment hτ P (r • u) hPmom,
      meanShift_laplace_eq_kernelCentroid_sub_of_moment hτ Q (r • u) hQmom] at hms
    have h2 := congrArg (· + r • u) hms
    simpa using h2
  have hQ' : Tendsto (fun r : ℝ => kernelCentroid (laplaceKernel τ) P (r • u)) atTop
      (𝓝 (exponentialTiltCentroid Q τ u)) := by rw [heq]; exact hQ
  exact tendsto_nhds_unique hP hQ'

end FarField

end DriftingIdentifiability
