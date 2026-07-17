import DriftingIdentifiability.LaplaceRadialRay2

/-!
# Radial Laplace converse, milestone G3 (`n = 2`): the system layer

Third G3 file: the ray mean `m̃₂ = D̃₂/Z̃₂`, its derivative `m̃₂'` (a quotient
of the first-order derivatives from `Ray2`), the covariance identity, the
closure `C̃₂ = Q̃₂ + 2τZ̃₂ + (τ/r)D̃₂`, and the named slack `RadialSlack₂`.

Everything is first-order — the constants are the `n = 2` specialization of the
general-`n` `System` layer (`3 ↦ 2`).
-/

open MeasureTheory Filter Topology Set

namespace DriftingIdentifiability

open Paper

/-! ## Positivity and elementary bounds -/

lemma radialRayZ₂_eq_kernelNormalizer (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayZ₂ τ ν r
      = kernelNormalizer (laplaceKernel τ) (radialMixture₂ ν) (rayProbe₂ r) :=
  (kernelNormalizer_radialMixture₂ τ hτ ν r).symm

lemma radialRayZ₂_pos (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) : 0 < radialRayZ₂ τ ν r := by
  rw [radialRayZ₂_eq_kernelNormalizer τ hτ ν r]
  exact laplaceKernelNormalizer_pos (radialMixture₂ ν) τ hτ (rayProbe₂ r)

lemma radialRayZ₂_le_one (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) : radialRayZ₂ τ ν r ≤ 1 := by
  rw [radialRayZ₂_eq_integral τ hτ ν r]
  calc ∫ y, laplaceKernel τ (rayProbe₂ r) y ∂(radialMixture₂ ν)
      ≤ ∫ _y, (1 : ℝ) ∂(radialMixture₂ ν) :=
        integral_mono (integrable_laplaceKernel_rayProbe₂ τ hτ _ r)
          (integrable_const 1)
          (fun y => laplaceKernel_rayProbe₂_le_one τ hτ r y)
    _ = 1 := by simp

lemma radialRayQ₂_nonneg (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) : 0 ≤ radialRayQ₂ τ ν r := by
  rw [radialRayQ₂_eq_integral τ ν r (integrable_Q_payload_rayProbe₂ τ hτ _ r)]
  exact integral_nonneg fun y => mul_nonneg
    (laplaceKernel_rayProbe₂_nonneg τ r y)
    (div_nonneg (sq_nonneg _) (norm_nonneg _))

lemma abs_radialRayZd₂_le (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) :
    |radialRayZd₂ τ ν r| ≤ radialRayZ₂ τ ν r := by
  rw [radialRayZd₂_eq_integral τ ν r (integrable_softsign_rayProbe₂ τ hτ _ r),
    radialRayZ₂_eq_integral τ hτ ν r]
  have hint := integrable_softsign_rayProbe₂ τ hτ (radialMixture₂ ν) r
  have hnorm := norm_integral_le_integral_norm
    (μ := radialMixture₂ ν)
    (f := fun y : EuclideanSpace ℝ (Fin 2) =>
      laplaceKernel τ (rayProbe₂ r) y *
        ((y 0 - r) / ‖rayProbe₂ r - y‖))
  simp only [Real.norm_eq_abs] at hnorm
  refine hnorm.trans ?_
  refine integral_mono hint.abs (integrable_laplaceKernel_rayProbe₂ τ hτ _ r)
    (fun y => ?_)
  rw [abs_mul, abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ r y)]
  simpa only [mul_one] using mul_le_mul_of_nonneg_left
    (abs_first_div_rayProbe₂_le_one r y)
    (laplaceKernel_rayProbe₂_nonneg τ r y)

/-! ## The ray mean and its derivative -/

noncomputable def radialRayM₂ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  radialRayD₂ τ ν r / radialRayZ₂ τ ν r

noncomputable def radialRayMDeriv₂ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  (((1 / τ) * radialRayQ₂ τ ν r - radialRayZ₂ τ ν r) * radialRayZ₂ τ ν r -
    radialRayD₂ τ ν r * ((1 / τ) * radialRayZd₂ τ ν r)) /
      (radialRayZ₂ τ ν r) ^ 2

theorem hasDerivAt_radialRayM₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayM₂ τ ν) (radialRayMDeriv₂ τ ν r) r :=
  (hasDerivAt_radialRayD₂ τ hτ ν hr).div
    (hasDerivAt_radialRayZ₂ τ hτ ν hr)
    (radialRayZ₂_pos τ hτ ν r).ne'

lemma radialRayMDeriv₂_cov (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) :
    τ * (radialRayMDeriv₂ τ ν r + 1) * (radialRayZ₂ τ ν r) ^ 2 =
      radialRayQ₂ τ ν r * radialRayZ₂ τ ν r -
        radialRayD₂ τ ν r * radialRayZd₂ τ ν r := by
  have hZne := (radialRayZ₂_pos τ hτ ν r).ne'
  rw [radialRayMDeriv₂]
  field_simp
  ring

/-- The named `n = 2` slack assumption (mirrors `RadialSlackN` at `n = 2`). -/
def RadialSlack₂ (τ : ℝ) (ν : Measure ℝ) : Prop :=
  ∀ r : ℝ, 0 < r → r < radialRayM₂ τ ν r →
    -(2 : ℝ) ≤ radialRayMDeriv₂ τ ν r

end DriftingIdentifiability
