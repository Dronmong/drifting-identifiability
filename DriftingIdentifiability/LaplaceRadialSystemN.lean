import DriftingIdentifiability.LaplaceRadialDifferentiationN
import DriftingIdentifiability.LaplaceACPropagation
import DriftingIdentifiability.LaplaceACFinal

/-!
# General-dimensional radial Laplace system

This file is the structural core of roadmap milestone G1.  It derives the
general-`n` ray mean, its sign layer, and the dimension-dependent Abel system
from the genuine Haar radial mixture.  The only external input inherited by
this file is the audited Haar-sphere first-coordinate formula used by the
physical/zonal bridge; all differentiation and system algebra below is proved.
-/

open MeasureTheory Filter Topology Set
open scoped RealInnerProductSpace

namespace DriftingIdentifiability
open Paper

/-! ## Positivity and elementary integral bounds -/

lemma radialRayZN_pos {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    0 < radialRayZN n τ ν r := by
  rw [radialRayZN_eq_kernelNormalizer hn hτ ν r]
  letI := radialMixtureN_isProbabilityMeasure hn ν
  exact laplaceKernelNormalizer_pos (radialMixtureN n ν) τ hτ
    (radialRayProbeN n (by omega) r)

lemma radialRayZN_nonneg {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    0 ≤ radialRayZN n τ ν r := (radialRayZN_pos hn τ hτ ν r).le

lemma radialRayZN_le_one {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayZN n τ ν r ≤ 1 := by
  rw [radialRayZN_eq_kernelNormalizer hn hτ ν r, kernelNormalizer]
  letI := radialMixtureN_isProbabilityMeasure hn ν
  calc
    (∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y
        ∂(radialMixtureN n ν))
      ≤ ∫ _y, (1 : ℝ) ∂(radialMixtureN n ν) :=
        integral_mono
          (integrable_laplaceKernel_radialRayProbeN
            (n := n) (by omega) τ hτ (radialMixtureN n ν) r)
          (integrable_const 1)
          (fun y => laplaceKernel_radialRayProbeN_le_one
            (n := n) (by omega) τ hτ r y)
    _ = 1 := by simp

lemma radialRayCN_nonneg {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    0 ≤ radialRayCN n τ ν r := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  rw [radialRayCN_eq_companionNormalizer hn τ ν r
    (integrable_laplaceCompanionKernel_radialRayProbeN
      (n := n) (by omega) τ hτ (radialMixtureN n ν) r), kernelNormalizer]
  exact integral_nonneg fun y =>
    laplaceCompanionKernel_radialRayProbeN_nonneg (n := n) (by omega) τ hτ r y

lemma radialRayCN_le {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayCN n τ ν r ≤ τ := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  rw [radialRayCN_eq_companionNormalizer hn τ ν r
    (integrable_laplaceCompanionKernel_radialRayProbeN
      (n := n) (by omega) τ hτ (radialMixtureN n ν) r), kernelNormalizer]
  calc
    (∫ y, laplaceCompanionKernel τ (radialRayProbeN n (by omega) r) y
        ∂(radialMixtureN n ν))
      ≤ ∫ _y, τ ∂(radialMixtureN n ν) :=
        integral_mono
          (integrable_laplaceCompanionKernel_radialRayProbeN
            (n := n) (by omega) τ hτ (radialMixtureN n ν) r)
          (integrable_const τ)
          (fun y => laplaceCompanionKernel_radialRayProbeN_le
            (n := n) (by omega) τ hτ r y)
    _ = τ := by simp

lemma abs_radialRayZdN_le {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    |radialRayZdN n (by omega) τ ν r| ≤ radialRayZN n τ ν r := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hn0 : 0 < n := by omega
  rw [radialRayZdN, radialRayZN_eq_kernelNormalizer hn hτ ν r,
    kernelNormalizer]
  have hint : Integrable (fun y : EuclideanSpace ℝ (Fin n) =>
      laplaceKernel τ (radialRayProbeN n (by omega) r) y *
        ((y ⟨0, by omega⟩ - r) /
          ‖radialRayProbeN n (by omega) r - y‖)) (radialMixtureN n ν) := by
    exact ⟨(((continuous_laplaceKernel_radialRayProbeN
        (n := n) (by omega) τ r).measurable).mul
          ((by fun_prop : Measurable (fun y : EuclideanSpace ℝ (Fin n) =>
            y ⟨0, hn0⟩ - r)).div
            (by fun_prop : Measurable (fun y : EuclideanSpace ℝ (Fin n) =>
              ‖radialRayProbeN n hn0 r - y‖)))).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := 1) (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs, abs_mul,
          abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg
            (n := n) (by omega) τ r y)]
        simpa only [mul_one] using (mul_le_mul
          (laplaceKernel_radialRayProbeN_le_one
            (n := n) (by omega) τ hτ r y)
          (abs_first_div_norm_radialRayProbeN_le_one
            (n := n) (by omega) r y)
          (abs_nonneg _) zero_le_one))⟩
  have hnorm := norm_integral_le_integral_norm
    (μ := radialMixtureN n ν)
    (f := fun y : EuclideanSpace ℝ (Fin n) =>
      laplaceKernel τ (radialRayProbeN n (by omega) r) y *
        ((y ⟨0, by omega⟩ - r) /
          ‖radialRayProbeN n (by omega) r - y‖))
  simp only [Real.norm_eq_abs] at hnorm
  refine hnorm.trans ?_
  exact integral_mono hint.abs
    (integrable_laplaceKernel_radialRayProbeN
      (n := n) (by omega) τ hτ (radialMixtureN n ν) r)
    (fun y => by
      rw [abs_mul,
        abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg
          (n := n) (by omega) τ r y)]
      simpa only [mul_one] using (mul_le_mul_of_nonneg_left
        (abs_first_div_norm_radialRayProbeN_le_one
          (n := n) (by omega) r y)
        (laplaceKernel_radialRayProbeN_nonneg
          (n := n) (by omega) τ r y)))

lemma radialRayQN_nonneg {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    0 ≤ radialRayQN n τ ν r := by
  rw [← radialRayQIntegralN_eq_radialRayQN hn τ hτ ν r,
    radialRayQIntegralN]
  exact integral_nonneg fun y => mul_nonneg
    (laplaceKernel_radialRayProbeN_nonneg (n := n) (by omega) τ r y)
    (div_nonneg (sq_nonneg _) (norm_nonneg _))

lemma abs_radialRayQN_le {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    |radialRayQN n τ ν r| ≤ τ * Real.exp (-1) := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  rw [← radialRayQIntegralN_eq_radialRayQN hn τ hτ ν r,
    radialRayQIntegralN]
  have hint := integrable_laplaceKernel_mul_first_sq_div
    (n := n) (by omega) τ hτ (radialMixtureN n ν) r
  have hnorm := norm_integral_le_integral_norm
    (μ := radialMixtureN n ν)
    (f := fun y : EuclideanSpace ℝ (Fin n) =>
      laplaceKernel τ (radialRayProbeN n (by omega) r) y *
        ((y ⟨0, by omega⟩ - r) ^ 2 /
          ‖radialRayProbeN n (by omega) r - y‖))
  simp only [Real.norm_eq_abs] at hnorm
  refine hnorm.trans ?_
  calc
    (∫ y, |laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          ((y ⟨0, by omega⟩ - r) ^ 2 /
            ‖radialRayProbeN n (by omega) r - y‖)|
        ∂(radialMixtureN n ν))
      ≤ ∫ _y, τ * Real.exp (-1) ∂(radialMixtureN n ν) :=
        integral_mono hint.abs (integrable_const _)
          (fun y => abs_laplaceKernel_mul_first_sq_div_le
            (n := n) (by omega) τ hτ r y)
    _ = τ * Real.exp (-1) := by simp

/-! ## The tilted ray displacement and its derivative -/

noncomputable def radialRayMN
    (n : ℕ) (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  radialRayDN n τ ν r / radialRayZN n τ ν r

noncomputable def radialRayMDerivN
    (n : ℕ) (hn : 0 < n) (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  (((1 / τ) * radialRayQN n τ ν r - radialRayZN n τ ν r) *
      radialRayZN n τ ν r -
    radialRayDN n τ ν r * ((1 / τ) * radialRayZdN n hn τ ν r)) /
      (radialRayZN n τ ν r) ^ 2

theorem hasDerivAt_radialRayMN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayMN n τ ν) (radialRayMDerivN n (by omega) τ ν r) r := by
  exact (hasDerivAt_radialRayDN_shell hn τ hτ ν hsupp hr).div
    (hasDerivAt_radialRayZN hn τ hτ ν hsupp hr)
    (radialRayZN_pos hn τ hτ ν r).ne'

lemma radialRayMDerivN_cov
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    τ * (radialRayMDerivN n (by omega) τ ν r + 1) * (radialRayZN n τ ν r) ^ 2 =
      radialRayQN n τ ν r * radialRayZN n τ ν r -
        radialRayDN n τ ν r * radialRayZdN n (by omega) τ ν r := by
  have hZne := (radialRayZN_pos hn τ hτ ν r).ne'
  rw [radialRayMDerivN]
  field_simp
  ring

/-- The named G1 slack assumption: only the far-tilt region remains outside
the proved AM--GM sign estimate. -/
def RadialSlackN (n : ℕ) (hn : 0 < n) (τ : ℝ) (ν : Measure ℝ) : Prop :=
  ∀ r : ℝ, 0 < r → r < radialRayMN n τ ν r →
    -(n : ℝ) ≤ radialRayMDerivN n hn τ ν r

/-! ## Closure identity -/

theorem radialRayCN_eq_firstOrder_closure
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    radialRayCN n τ ν r =
      radialRayQN n τ ν r + (n : ℝ) * τ * radialRayZN n τ ν r +
        (((n : ℝ) - 1) * τ / r) * radialRayDN n τ ν r := by
  have hC := radialRayCN_eq_closure_of_probability hn hτ ν hsupp hr
  have hP := radialRayRhoSqOverDistN_eq_const_mul_T hn hτ ν hsupp hr
  have hT := radialRayTN_eq_radialRayDN_add_rZ hn τ ν r
    (integrable_shellDN hn hτ ν) (integrable_shellZN hn hτ ν)
  rw [hP, hT] at hC
  rw [hC]
  field_simp [hr.ne']
  ring

/-! ## The proved part of the radial sign layer -/

lemma integrable_laplaceKernel_mul_norm_radialRayProbeN
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin n))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceKernel τ (radialRayProbeN n hn r) y *
      ‖radialRayProbeN n hn r - y‖) μ := by
  refine ⟨((continuous_laplaceKernel_radialRayProbeN hn τ r).mul
    ((continuous_const.sub continuous_id).norm)).aestronglyMeasurable,
    HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
      (ae_of_all _ fun y => ?_)⟩
  rw [Real.norm_eq_abs, abs_mul,
    abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg hn τ r y),
    abs_of_nonneg (norm_nonneg _)]
  calc
    laplaceKernel τ (radialRayProbeN n hn r) y *
        ‖radialRayProbeN n hn r - y‖
      = ‖radialRayProbeN n hn r - y‖ *
          Real.exp (-‖radialRayProbeN n hn r - y‖ / τ) := by
        simp only [laplaceKernel]
        rw [show -(1 / τ) * ‖radialRayProbeN n hn r - y‖ =
          -‖radialRayProbeN n hn r - y‖ / τ by ring]
        ring
    _ ≤ τ * Real.exp (-1) := mul_exp_neg_div_le hτ (norm_nonneg _)

lemma integral_laplaceKernel_mul_norm_eq_CN_sub
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    (∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y *
        ‖radialRayProbeN n (by omega) r - y‖ ∂(radialMixtureN n ν)) =
      radialRayCN n τ ν r - τ * radialRayZN n τ ν r := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hCint := integrable_laplaceCompanionKernel_radialRayProbeN
    (n := n) (by omega) τ hτ (radialMixtureN n ν) r
  have hZint := integrable_laplaceKernel_radialRayProbeN
    (n := n) (by omega) τ hτ (radialMixtureN n ν) r
  have hSint := integrable_laplaceKernel_mul_norm_radialRayProbeN
    (n := n) (by omega) τ hτ (radialMixtureN n ν) r
  rw [radialRayCN_eq_companionNormalizer hn τ ν r hCint,
    radialRayZN_eq_kernelNormalizer hn hτ ν r]
  simp only [kernelNormalizer]
  have hsplit :
      (∫ y, laplaceCompanionKernel τ (radialRayProbeN n (by omega) r) y
          ∂(radialMixtureN n ν)) =
        ∫ y, τ * laplaceKernel τ (radialRayProbeN n (by omega) r) y +
          laplaceKernel τ (radialRayProbeN n (by omega) r) y *
            ‖radialRayProbeN n (by omega) r - y‖ ∂(radialMixtureN n ν) := by
    exact integral_congr_ae (Filter.Eventually.of_forall fun y => by
      simp only [laplaceCompanionKernel]
      ring)
  rw [hsplit, integral_add (hZint.const_mul τ) hSint, integral_const_mul]
  ring

lemma two_mul_laplace_abs_first_le
    {n : ℕ} (hn : 0 < n) (τ r : ℝ)
    (y : EuclideanSpace ℝ (Fin n)) :
    2 * (laplaceKernel τ (radialRayProbeN n hn r) y *
        |y ⟨0, hn⟩ - r|) ≤
      laplaceKernel τ (radialRayProbeN n hn r) y *
          ((y ⟨0, hn⟩ - r) ^ 2 /
            ‖radialRayProbeN n hn r - y‖) +
        laplaceKernel τ (radialRayProbeN n hn r) y *
          ‖radialRayProbeN n hn r - y‖ := by
  have hK := laplaceKernel_radialRayProbeN_nonneg hn τ r y
  rcases eq_or_ne (‖radialRayProbeN n hn r - y‖) 0 with h0 | hne
  · have hy : radialRayProbeN n hn r = y := norm_sub_eq_zero_iff.mp h0
    have hX : y ⟨0, hn⟩ - r = 0 := by
      rw [← hy, radialRayProbeN_first, sub_self]
    simp [h0, hX]
  · have hdpos : 0 < ‖radialRayProbeN n hn r - y‖ :=
      (norm_nonneg _).lt_of_ne' hne
    have hAM : 2 * |y ⟨0, hn⟩ - r| ≤
        (y ⟨0, hn⟩ - r) ^ 2 / ‖radialRayProbeN n hn r - y‖ +
          ‖radialRayProbeN n hn r - y‖ := by
      rw [← sub_nonneg]
      have hkey :
          (y ⟨0, hn⟩ - r) ^ 2 / ‖radialRayProbeN n hn r - y‖ +
              ‖radialRayProbeN n hn r - y‖ - 2 * |y ⟨0, hn⟩ - r| =
            (|y ⟨0, hn⟩ - r| - ‖radialRayProbeN n hn r - y‖) ^ 2 /
              ‖radialRayProbeN n hn r - y‖ := by
        field_simp
        nlinarith [sq_abs (y ⟨0, hn⟩ - r)]
      rw [hkey]
      positivity
    calc
      2 * (laplaceKernel τ (radialRayProbeN n hn r) y *
          |y ⟨0, hn⟩ - r|) =
        laplaceKernel τ (radialRayProbeN n hn r) y *
          (2 * |y ⟨0, hn⟩ - r|) := by ring
      _ ≤ laplaceKernel τ (radialRayProbeN n hn r) y *
          ((y ⟨0, hn⟩ - r) ^ 2 /
              ‖radialRayProbeN n hn r - y‖ +
            ‖radialRayProbeN n hn r - y‖) :=
        mul_le_mul_of_nonneg_left hAM hK
      _ = _ := by ring

lemma abs_radialRayDN_le
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    |radialRayDN n τ ν r| ≤ radialRayQN n τ ν r +
      ((((n : ℝ) - 1) * τ) / (2 * r)) *
        (radialRayDN n τ ν r + r * radialRayZN n τ ν r) := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hDint := integrable_laplaceKernel_mul_first_radialRayProbeN
    (n := n) (by omega) τ hτ (radialMixtureN n ν) r
  have hQint := integrable_laplaceKernel_mul_first_sq_div
    (n := n) (by omega) τ hτ (radialMixtureN n ν) r
  have hSint := integrable_laplaceKernel_mul_norm_radialRayProbeN
    (n := n) (by omega) τ hτ (radialMixtureN n ν) r
  have habs : |radialRayDN n τ ν r| ≤
      ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y *
        |y ⟨0, by omega⟩ - r| ∂(radialMixtureN n ν) := by
    rw [radialRayDN_eq_integral_first hn τ hτ ν r]
    have h1 := norm_integral_le_integral_norm
      (μ := radialMixtureN n ν)
      (f := fun y : EuclideanSpace ℝ (Fin n) =>
        laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          (y ⟨0, by omega⟩ - r))
    simp only [Real.norm_eq_abs] at h1
    refine h1.trans (le_of_eq (integral_congr_ae
      (Filter.Eventually.of_forall fun y => ?_)))
    simp only [abs_mul, abs_of_nonneg
      (laplaceKernel_radialRayProbeN_nonneg (n := n) (by omega) τ r y)]
  have hintAbs : Integrable (fun y : EuclideanSpace ℝ (Fin n) =>
      laplaceKernel τ (radialRayProbeN n (by omega) r) y *
        |y ⟨0, by omega⟩ - r|) (radialMixtureN n ν) := by
    convert hDint.abs using 1
    funext y
    rw [abs_mul, abs_of_nonneg
      (laplaceKernel_radialRayProbeN_nonneg (n := n) (by omega) τ r y)]
  have hAM :
      (∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          |y ⟨0, by omega⟩ - r| ∂(radialMixtureN n ν)) ≤
        (radialRayQN n τ ν r +
          ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y *
            ‖radialRayProbeN n (by omega) r - y‖ ∂(radialMixtureN n ν)) / 2 := by
    calc
      (∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          |y ⟨0, by omega⟩ - r| ∂(radialMixtureN n ν))
        ≤ ∫ y, (laplaceKernel τ (radialRayProbeN n (by omega) r) y *
              ((y ⟨0, by omega⟩ - r) ^ 2 /
                ‖radialRayProbeN n (by omega) r - y‖) +
            laplaceKernel τ (radialRayProbeN n (by omega) r) y *
              ‖radialRayProbeN n (by omega) r - y‖) / 2
              ∂(radialMixtureN n ν) := by
          refine integral_mono hintAbs ((hQint.add hSint).div_const 2) (fun y => ?_)
          have h := two_mul_laplace_abs_first_le
            (n := n) (by omega) τ r y
          linarith
      _ = (radialRayQN n τ ν r +
          ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y *
            ‖radialRayProbeN n (by omega) r - y‖ ∂(radialMixtureN n ν)) / 2 := by
          rw [integral_div, integral_add hQint hSint,
            ← radialRayQIntegralN_eq_radialRayQN hn τ hτ ν r,
            radialRayQIntegralN]
  have hS := integral_laplaceKernel_mul_norm_eq_CN_sub hn τ hτ ν r
  have hC := radialRayCN_eq_firstOrder_closure hn τ hτ ν hsupp hr
  rw [hS, hC] at hAM
  have h := habs.trans hAM
  field_simp [hr.ne'] at h ⊢
  linarith

theorem radialRayMDerivN_ge_of_le
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) (hm : radialRayMN n τ ν r ≤ r) :
    -(n : ℝ) ≤ radialRayMDerivN n (by omega) τ ν r := by
  set Z := radialRayZN n τ ν r with hZ
  set D := radialRayDN n τ ν r with hD
  set Q := radialRayQN n τ ν r with hQ
  set Zd := radialRayZdN n (by omega) τ ν r with hZd
  have hZpos : 0 < Z := radialRayZN_pos hn τ hτ ν r
  have hDle : D ≤ r * Z := by
    rw [radialRayMN, div_le_iff₀ hZpos] at hm
    exact hm
  have hDabs := abs_radialRayDN_le hn τ hτ ν hsupp hr
  have hDabs2 : |D| ≤ Q + ((n : ℝ) - 1) * τ * Z := by
    have hn1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast (show 1 ≤ n by omega)
    have hcoef : 0 ≤ (((n : ℝ) - 1) * τ) / (2 * r) :=
      div_nonneg (mul_nonneg (sub_nonneg.mpr hn1) hτ.le)
        (mul_nonneg (by norm_num) hr.le)
    have hinside : D + r * Z ≤ 2 * r * Z := by linarith
    have hmul := mul_le_mul_of_nonneg_left hinside hcoef
    have hcalc : (((n : ℝ) - 1) * τ) / (2 * r) * (2 * r * Z) =
        ((n : ℝ) - 1) * τ * Z := by field_simp [hr.ne']
    rw [← hZ, ← hD, ← hQ] at hDabs
    linarith
  have hZdabs : |Zd| ≤ Z := by
    simpa [hZ, hZd] using abs_radialRayZdN_le hn τ hτ ν r
  have hDZd : D * Zd ≤ |D| * Z := by
    calc
      D * Zd ≤ |D * Zd| := le_abs_self _
      _ = |D| * |Zd| := abs_mul _ _
      _ ≤ |D| * Z := mul_le_mul_of_nonneg_left hZdabs (abs_nonneg _)
  have hcov := radialRayMDerivN_cov hn τ hτ ν r
  rw [← hZ, ← hD, ← hQ, ← hZd] at hcov
  have hlower : -(((n : ℝ) - 1) * τ) * Z ^ 2 ≤ Q * Z - D * Zd := by
    have htmp : D * Zd ≤ (Q + ((n : ℝ) - 1) * τ * Z) * Z :=
      hDZd.trans (mul_le_mul_of_nonneg_right hDabs2 hZpos.le)
    nlinarith [hZpos]
  have hkey : -(((n : ℝ) - 1) * τ) * Z ^ 2 ≤
      τ * (radialRayMDerivN n (by omega) τ ν r + 1) * Z ^ 2 := by
    rw [hcov]
    exact hlower
  have hZsq : 0 < Z ^ 2 := by positivity
  by_contra hcon
  have hlt : radialRayMDerivN n (by omega) τ ν r + 1 <
      -((n : ℝ) - 1) := by linarith
  have hstrict := mul_lt_mul_of_pos_right
    (mul_lt_mul_of_pos_left hlt hτ) hZsq
  have : τ * (radialRayMDerivN n (by omega) τ ν r + 1) * Z ^ 2 <
      -(((n : ℝ) - 1) * τ) * Z ^ 2 := by
    calc
      τ * (radialRayMDerivN n (by omega) τ ν r + 1) * Z ^ 2
        < τ * (-((n : ℝ) - 1)) * Z ^ 2 := hstrict
      _ = -(((n : ℝ) - 1) * τ) * Z ^ 2 := by ring
  linarith

theorem radialRayMDerivN_ge
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    (hslack : RadialSlackN n (by omega) τ ν) {r : ℝ} (hr : 0 < r) :
    -(n : ℝ) ≤ radialRayMDerivN n (by omega) τ ν r := by
  rcases le_or_gt (radialRayMN n τ ν r) r with hm | hm
  · exact radialRayMDerivN_ge_of_le hn τ hτ ν hsupp hr hm
  · exact hslack r hr hm

/-! ## Continuity of the quotient payloads -/

lemma continuous_radialRayProbeN_sub_norm
    {n : ℕ} (hn : 0 < n) (y : EuclideanSpace ℝ (Fin n)) :
    Continuous fun r : ℝ => ‖radialRayProbeN n hn r - y‖ := by
  unfold radialRayProbeN
  fun_prop

lemma continuous_laplaceKernel_radialRayProbeN_left
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (y : EuclideanSpace ℝ (Fin n)) :
    Continuous fun r : ℝ => laplaceKernel τ (radialRayProbeN n hn r) y := by
  simp only [laplaceKernel]
  exact Real.continuous_exp.comp
    ((continuous_radialRayProbeN_sub_norm hn y).const_mul (-(1 / τ)))

lemma continuousAt_radialRayQN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) : ContinuousAt (radialRayQN n τ ν) r := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hn0 : 0 < n := by omega
  have hfe : radialRayQN n τ ν = fun x =>
      ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) x) y *
        ((y ⟨0, by omega⟩ - x) ^ 2 /
          ‖radialRayProbeN n (by omega) x - y‖) ∂(radialMixtureN n ν) := by
    funext x
    exact (radialRayQIntegralN_eq_radialRayQN hn τ hτ ν x).symm
  rw [hfe]
  refine continuousAt_of_dominated (bound := fun _ => τ * Real.exp (-1)) ?_ ?_ ?_ ?_
  · exact Filter.Eventually.of_forall fun x =>
      (measurable_laplaceKernel_mul_first_sq_div
        (n := n) (by omega) τ x).aestronglyMeasurable
  · exact Filter.Eventually.of_forall fun x => ae_of_all _ fun y => by
      rw [Real.norm_eq_abs]
      exact abs_laplaceKernel_mul_first_sq_div_le
        (n := n) (by omega) τ hτ x y
  · exact integrable_const _
  · filter_upwards [radialMixtureN_ae_probe_ne hn ν hsupp] with y hy
    have hne := hy r hr
    exact (continuous_laplaceKernel_radialRayProbeN_left
      (n := n) (by omega) τ y).continuousAt.mul
        ((by fun_prop : ContinuousAt (fun x : ℝ =>
          (y ⟨0, hn0⟩ - x) ^ 2) r).div
          (continuous_radialRayProbeN_sub_norm
            (n := n) (by omega) y).continuousAt hne)

lemma continuousAt_radialRayZdN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayZdN n (by omega) τ ν) r := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hn0 : 0 < n := by omega
  change ContinuousAt (fun x =>
    ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) x) y *
      ((y ⟨0, by omega⟩ - x) /
        ‖radialRayProbeN n (by omega) x - y‖) ∂(radialMixtureN n ν)) r
  refine continuousAt_of_dominated (bound := fun _ => (1 : ℝ)) ?_ ?_ ?_ ?_
  · exact Filter.Eventually.of_forall fun x =>
      ((((continuous_laplaceKernel_radialRayProbeN
        (n := n) (by omega) τ x).measurable).mul
          ((by fun_prop : Measurable (fun y : EuclideanSpace ℝ (Fin n) =>
            y ⟨0, hn0⟩ - x)).div
            (by fun_prop : Measurable (fun y : EuclideanSpace ℝ (Fin n) =>
              ‖radialRayProbeN n hn0 x - y‖)))).aestronglyMeasurable)
  · exact Filter.Eventually.of_forall fun x => ae_of_all _ fun y => by
      rw [Real.norm_eq_abs, abs_mul,
        abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg
          (n := n) (by omega) τ x y)]
      simpa only [mul_one] using (mul_le_mul
        (laplaceKernel_radialRayProbeN_le_one
          (n := n) (by omega) τ hτ x y)
        (abs_first_div_norm_radialRayProbeN_le_one
          (n := n) (by omega) x y)
        (abs_nonneg _) zero_le_one)
  · exact integrable_const 1
  · filter_upwards [radialMixtureN_ae_probe_ne hn ν hsupp] with y hy
    have hne := hy r hr
    exact (continuous_laplaceKernel_radialRayProbeN_left
      (n := n) (by omega) τ y).continuousAt.mul
        ((by fun_prop : ContinuousAt (fun x : ℝ =>
          y ⟨0, hn0⟩ - x) r).div
          (continuous_radialRayProbeN_sub_norm
            (n := n) (by omega) y).continuousAt hne)

lemma continuousAt_radialRayMN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) : ContinuousAt (radialRayMN n τ ν) r :=
  (hasDerivAt_radialRayDN_shell hn τ hτ ν hsupp hr).continuousAt.div
    (hasDerivAt_radialRayZN hn τ hτ ν hsupp hr).continuousAt
    (radialRayZN_pos hn τ hτ ν r).ne'

lemma continuousAt_radialRayMDerivN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayMDerivN n (by omega) τ ν) r := by
  have hZ := (hasDerivAt_radialRayZN hn τ hτ ν hsupp hr).continuousAt
  have hD := (hasDerivAt_radialRayDN_shell hn τ hτ ν hsupp hr).continuousAt
  have hQ := continuousAt_radialRayQN hn τ hτ ν hsupp hr
  have hZd := continuousAt_radialRayZdN hn τ hτ ν hsupp hr
  exact ((((hQ.const_mul _).sub hZ).mul hZ).sub
    (hD.mul (hZd.const_mul _))).div (hZ.pow 2)
      (pow_ne_zero 2 (radialRayZN_pos hn τ hτ ν r).ne')

/-! ## Zero-drift reduction on the genuine radial ray -/

lemma radialRayDN_eq_weightedDisplacementCoord
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayDN n τ ν r =
      (∫ y, laplaceWeightedDisplacement τ
        (radialRayProbeN n (by omega) r) y ∂(radialMixtureN n ν))
          ⟨0, by omega⟩ := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hFint : Integrable (fun y => laplaceWeightedDisplacement τ
      (radialRayProbeN n (by omega) r) y) (radialMixtureN n ν) :=
    laplaceWeightedDisplacement_integrable τ hτ (radialMixtureN n ν)
      (radialRayProbeN n (by omega) r)
  have hproj := (EuclideanSpace.proj (⟨0, by omega⟩ : Fin n)).integral_comp_comm
    (μ := radialMixtureN n ν) hFint
  have hcoord :
      (∫ y, laplaceWeightedDisplacement τ
          (radialRayProbeN n (by omega) r) y ∂(radialMixtureN n ν))
          ⟨0, by omega⟩ =
        ∫ y, (laplaceWeightedDisplacement τ
          (radialRayProbeN n (by omega) r) y) ⟨0, by omega⟩
          ∂(radialMixtureN n ν) := by
    simpa [EuclideanSpace.coe_proj] using hproj.symm
  rw [hcoord]
  exact radialRayDN_eq_displacementCoord hn τ ν r
    (by
      simpa only [laplaceWeightedDisplacement, PiLp.smul_apply, PiLp.sub_apply,
        radialRayProbeN_first, smul_eq_mul] using
          integrable_laplaceKernel_mul_first_radialRayProbeN
            (n := n) (by omega) τ hτ (radialMixtureN n ν) r)

lemma zeroDrift_radialRay_meanShift_eq
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (νp νq : Measure ℝ)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq)) (r : ℝ) :
    meanShift (laplaceKernel τ) (radialMixtureN n νp)
        (radialRayProbeN n (by omega) r) =
      meanShift (laplaceKernel τ) (radialMixtureN n νq)
        (radialRayProbeN n (by omega) r) := by
  have h := hzero (radialRayProbeN n (by omega) r)
  rwa [meanShiftDrift, sub_eq_zero] at h

lemma zeroDrift_radialRay_D_mul_Z
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq)) (r : ℝ) :
    radialRayDN n τ νp r * radialRayZN n τ νq r =
      radialRayDN n τ νq r * radialRayZN n τ νp r := by
  have h := congrArg (fun v : EuclideanSpace ℝ (Fin n) => v ⟨0, by omega⟩)
    (zeroDrift_radialRay_meanShift_eq hn τ νp νq hzero r)
  simp only [meanShift, PiLp.smul_apply, smul_eq_mul] at h
  have hip :
      (∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y •
          (y - radialRayProbeN n (by omega) r) ∂(radialMixtureN n νp)) =
        ∫ y, laplaceWeightedDisplacement τ
          (radialRayProbeN n (by omega) r) y ∂(radialMixtureN n νp) := rfl
  have hiq :
      (∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y •
          (y - radialRayProbeN n (by omega) r) ∂(radialMixtureN n νq)) =
        ∫ y, laplaceWeightedDisplacement τ
          (radialRayProbeN n (by omega) r) y ∂(radialMixtureN n νq) := rfl
  rw [hip, hiq,
    ← radialRayZN_eq_kernelNormalizer hn hτ νp r,
    ← radialRayZN_eq_kernelNormalizer hn hτ νq r,
    ← radialRayDN_eq_weightedDisplacementCoord hn τ hτ νp r,
    ← radialRayDN_eq_weightedDisplacementCoord hn τ hτ νq r] at h
  have hZp := radialRayZN_pos hn τ hτ νp r
  have hZq := radialRayZN_pos hn τ hτ νq r
  field_simp [hZp.ne', hZq.ne'] at h
  linarith

lemma zeroDrift_radialRay_M_eq
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq)) (r : ℝ) :
    radialRayMN n τ νp r = radialRayMN n τ νq r := by
  have hp := radialRayZN_pos hn τ hτ νp r
  have hq := radialRayZN_pos hn τ hτ νq r
  rw [radialRayMN, radialRayMN, div_eq_div_iff hp.ne' hq.ne']
  exact zeroDrift_radialRay_D_mul_Z hn τ hτ νp νq hzero r

lemma zeroDrift_radialRay_MDeriv_eq
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq))
    {r : ℝ} (hr : 0 < r) :
    radialRayMDerivN n (by omega) τ νp r =
      radialRayMDerivN n (by omega) τ νq r := by
  have hp := hasDerivAt_radialRayMN hn τ hτ νp hsp hr
  have hq := hasDerivAt_radialRayMN hn τ hτ νq hsq hr
  have heq : radialRayMN n τ νp =ᶠ[𝓝 r] radialRayMN n τ νq :=
    Filter.Eventually.of_forall fun x =>
      zeroDrift_radialRay_M_eq hn τ hτ νp νq hzero x
  exact (hp.congr_of_eventuallyEq heq.symm).unique hq

lemma zeroDrift_radialRay_D_deriv_bridge
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp ν : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure ν]
    (hsp : νp (Iio 0) = 0) (hs : ν (Iio 0) = 0)
    (hM : ∀ x : ℝ, radialRayMN n τ νp x = radialRayMN n τ ν x)
    {r : ℝ} (hr : 0 < r) :
    (1 / τ) * radialRayQN n τ ν r - radialRayZN n τ ν r =
      radialRayMDerivN n (by omega) τ νp r * radialRayZN n τ ν r +
        radialRayMN n τ νp r *
          ((1 / τ) * radialRayZdN n (by omega) τ ν r) := by
  have hDν := hasDerivAt_radialRayDN_shell hn τ hτ ν hs hr
  have hprod := (hasDerivAt_radialRayMN hn τ hτ νp hsp hr).mul
    (hasDerivAt_radialRayZN hn τ hτ ν hs hr)
  have heq : (fun x => radialRayMN n τ νp x * radialRayZN n τ ν x)
      =ᶠ[𝓝 r] radialRayDN n τ ν := by
    exact Filter.Eventually.of_forall fun x => by
      change radialRayMN n τ νp x * radialRayZN n τ ν x = radialRayDN n τ ν x
      rw [hM x, radialRayMN, div_mul_cancel₀]
      exact (radialRayZN_pos hn τ hτ ν x).ne'
  exact ((hprod.congr_of_eventuallyEq heq.symm).unique hDν).symm

/-! ## The general-`n` Abel system -/

noncomputable def radialRayWN
    (n : ℕ) (hn : 0 < n) (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  (1 / τ) * (radialRayZdN n hn τ νp r * radialRayZN n τ νq r -
    radialRayZdN n hn τ νq r * radialRayZN n τ νp r)

noncomputable def radialRayVN
    (n : ℕ) (hn : 0 < n) (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  r ^ (n - 1) * radialRayWN n hn τ νp νq r

noncomputable def radialRayKN
    (n : ℕ) (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  radialRayCN n τ νp r * radialRayZN n τ νq r -
    radialRayCN n τ νq r * radialRayZN n τ νp r

noncomputable def radialRayKhatN
    (n : ℕ) (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  r ^ (n - 1) * radialRayKN n τ νp νq r

theorem radialRayKhatN_eq_M_mul_V
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq))
    {r : ℝ} (hr : 0 < r) :
    radialRayKhatN n τ νp νq r =
      τ * radialRayMN n τ νp r *
        radialRayVN n (by omega) τ νp νq r := by
  have hMp : ∀ x, radialRayMN n τ νp x = radialRayMN n τ νp x := fun _ => rfl
  have hMq : ∀ x, radialRayMN n τ νp x = radialRayMN n τ νq x :=
    fun x => zeroDrift_radialRay_M_eq hn τ hτ νp νq hzero x
  have hp := zeroDrift_radialRay_D_deriv_bridge
    hn τ hτ νp νp hsp hsp hMp hr
  have hq := zeroDrift_radialRay_D_deriv_bridge
    hn τ hτ νp νq hsp hsq hMq hr
  have hCp := radialRayCN_eq_firstOrder_closure hn τ hτ νp hsp hr
  have hCq := radialRayCN_eq_firstOrder_closure hn τ hτ νq hsq hr
  have hp' : radialRayQN n τ νp r =
      τ * radialRayZN n τ νp r +
        τ * radialRayMDerivN n (by omega) τ νp r * radialRayZN n τ νp r +
        radialRayMN n τ νp r * radialRayZdN n (by omega) τ νp r := by
    field_simp [hτ.ne'] at hp
    linarith
  have hq' : radialRayQN n τ νq r =
      τ * radialRayZN n τ νq r +
        τ * radialRayMDerivN n (by omega) τ νp r * radialRayZN n τ νq r +
        radialRayMN n τ νp r * radialRayZdN n (by omega) τ νq r := by
    field_simp [hτ.ne'] at hq
    linarith
  have hDp : radialRayDN n τ νp r =
      radialRayMN n τ νp r * radialRayZN n τ νp r := by
    rw [radialRayMN, div_mul_cancel₀]
    exact (radialRayZN_pos hn τ hτ νp r).ne'
  have hDq : radialRayDN n τ νq r =
      radialRayMN n τ νp r * radialRayZN n τ νq r := by
    calc
      radialRayDN n τ νq r =
          radialRayMN n τ νq r * radialRayZN n τ νq r := by
            rw [radialRayMN, div_mul_cancel₀]
            exact (radialRayZN_pos hn τ hτ νq r).ne'
      _ = _ := by
        rw [← zeroDrift_radialRay_M_eq hn τ hτ νp νq hzero r]
  have hbase : radialRayKN n τ νp νq r =
      τ * radialRayMN n τ νp r * radialRayWN n (by omega) τ νp νq r := by
    rw [radialRayKN, radialRayWN, hCp, hCq, hp', hq', hDp, hDq]
    field_simp [hτ.ne', hr.ne']
    ring
  rw [radialRayKhatN, radialRayVN, hbase]
  ring

theorem hasDerivAt_radialRayKhatN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq))
    {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayKhatN n τ νp νq)
      (-(τ * (radialRayMDerivN n (by omega) τ νp r + (n : ℝ) + 1)) *
        radialRayVN n (by omega) τ νp νq r) r := by
  have hCp := hasDerivAt_radialRayCN hn τ hτ νp hsp hr
  have hCq := hasDerivAt_radialRayCN hn τ hτ νq hsq hr
  have hZp := hasDerivAt_radialRayZN hn τ hτ νp hsp hr
  have hZq := hasDerivAt_radialRayZN hn τ hτ νq hsq hr
  have hKraw := (hCp.mul hZq).sub (hCq.mul hZp)
  have hK : HasDerivAt (radialRayKN n τ νp νq)
      ((1 / τ) * radialRayDN n τ νp r * radialRayZN n τ νq r +
          radialRayCN n τ νp r *
            ((1 / τ) * radialRayZdN n (by omega) τ νq r) -
        ((1 / τ) * radialRayDN n τ νq r * radialRayZN n τ νp r +
          radialRayCN n τ νq r *
            ((1 / τ) * radialRayZdN n (by omega) τ νp r))) r := by
    change HasDerivAt (fun x =>
      radialRayCN n τ νp x * radialRayZN n τ νq x -
        radialRayCN n τ νq x * radialRayZN n τ νp x) _ r
    convert hKraw using 1 <;> rfl
  have hpow : HasDerivAt (fun x : ℝ => x ^ (n - 1))
      ((n - 1 : ℕ) * r ^ (n - 1 - 1)) r := by
    simpa using hasDerivAt_pow (n - 1) r
  have hCpcl := radialRayCN_eq_firstOrder_closure hn τ hτ νp hsp hr
  have hCqcl := radialRayCN_eq_firstOrder_closure hn τ hτ νq hsq hr
  have hMp := zeroDrift_radialRay_M_eq hn τ hτ νp νq hzero r
  have hKM := radialRayKhatN_eq_M_mul_V hn τ hτ νp νq hsp hsq hzero hr
  have hbp := zeroDrift_radialRay_D_deriv_bridge hn τ hτ νp νp hsp hsp
    (fun _ => rfl) hr
  have hbq := zeroDrift_radialRay_D_deriv_bridge hn τ hτ νp νq hsp hsq
    (fun x => zeroDrift_radialRay_M_eq hn τ hτ νp νq hzero x) hr
  have hp' : radialRayQN n τ νp r =
      τ * radialRayZN n τ νp r +
        τ * radialRayMDerivN n (by omega) τ νp r * radialRayZN n τ νp r +
        radialRayMN n τ νp r * radialRayZdN n (by omega) τ νp r := by
    field_simp [hτ.ne'] at hbp
    linarith
  have hq' : radialRayQN n τ νq r =
      τ * radialRayZN n τ νq r +
        τ * radialRayMDerivN n (by omega) τ νp r * radialRayZN n τ νq r +
        radialRayMN n τ νp r * radialRayZdN n (by omega) τ νq r := by
    field_simp [hτ.ne'] at hbq
    linarith
  have hDp : radialRayDN n τ νp r =
      radialRayMN n τ νp r * radialRayZN n τ νp r := by
    rw [radialRayMN, div_mul_cancel₀]
    exact (radialRayZN_pos hn τ hτ νp r).ne'
  have hDq : radialRayDN n τ νq r =
      radialRayMN n τ νp r * radialRayZN n τ νq r := by
    calc
      radialRayDN n τ νq r =
          radialRayMN n τ νq r * radialRayZN n τ νq r := by
            rw [radialRayMN, div_mul_cancel₀]
            exact (radialRayZN_pos hn τ hτ νq r).ne'
      _ = _ := by rw [← hMp]
  have hKval :
      (1 / τ) * radialRayDN n τ νp r * radialRayZN n τ νq r +
          radialRayCN n τ νp r *
            ((1 / τ) * radialRayZdN n (by omega) τ νq r) -
        ((1 / τ) * radialRayDN n τ νq r * radialRayZN n τ νp r +
          radialRayCN n τ νq r *
            ((1 / τ) * radialRayZdN n (by omega) τ νp r)) =
      -(τ * (radialRayMDerivN n (by omega) τ νp r + (n : ℝ) + 1 +
          (((n : ℝ) - 1) * radialRayMN n τ νp r / r))) *
        radialRayWN n (by omega) τ νp νq r := by
    rw [hCpcl, hCqcl, hp', hq', hDp, hDq, radialRayWN]
    field_simp [hτ.ne', hr.ne']
    ring
  have hK' := hK.congr_deriv hKval
  have hrpow : r ^ (n - 1) ≠ 0 := pow_ne_zero _ hr.ne'
  have hKbase : radialRayKN n τ νp νq r =
      τ * radialRayMN n τ νp r * radialRayWN n (by omega) τ νp νq r := by
    apply mul_left_cancel₀ hrpow
    calc
      r ^ (n - 1) * radialRayKN n τ νp νq r =
          radialRayKhatN n τ νp νq r := rfl
      _ = τ * radialRayMN n τ νp r *
          radialRayVN n (by omega) τ νp νq r := hKM
      _ = r ^ (n - 1) *
          (τ * radialRayMN n τ νp r *
            radialRayWN n (by omega) τ νp νq r) := by
        rw [radialRayVN]
        ring
  have hprod := hpow.mul hK'
  have hval :
      ((n - 1 : ℕ) : ℝ) * r ^ (n - 1 - 1) * radialRayKN n τ νp νq r +
        r ^ (n - 1) *
          (-(τ * (radialRayMDerivN n (by omega) τ νp r + (n : ℝ) + 1 +
            (((n : ℝ) - 1) * radialRayMN n τ νp r / r))) *
              radialRayWN n (by omega) τ νp νq r) =
        -(τ * (radialRayMDerivN n (by omega) τ νp r + (n : ℝ) + 1)) *
          radialRayVN n (by omega) τ νp νq r := by
    rw [hKbase, radialRayVN]
    have hnsub : ((n - 1 : ℕ) : ℝ) = (n : ℝ) - 1 := by
      rw [Nat.cast_sub (by omega)]
      norm_num
    have hpowrel : r ^ (n - 1) = r * r ^ (n - 1 - 1) := by
      have heq : n - 1 = (n - 1 - 1) + 1 := by omega
      calc
        r ^ (n - 1) = r ^ ((n - 1 - 1) + 1) := congrArg (fun k : ℕ => r ^ k) heq
        _ = r * r ^ (n - 1 - 1) := by rw [pow_succ']
    rw [hnsub, hpowrel]
    field_simp [hτ.ne', hr.ne']
    ring
  have hfinal := hprod.congr_deriv hval
  change HasDerivAt (fun x =>
    x ^ (n - 1) * radialRayKN n τ νp νq x) _ r
  convert hfinal using 1 <;> rfl

end DriftingIdentifiability
