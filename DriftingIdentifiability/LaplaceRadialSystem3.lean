import DriftingIdentifiability.LaplaceRadialRay3

/-!
# Radial Laplace converse, milestone L5: the K̂ Abel system (`n = 3`)

This file derives the propagating Abel system of `LaplaceHigherDim.md §4.10
(F5)` from the R-layer.  Under zero drift, with the common tilted displacement
`m̃` and the Wronskian-type field `W̃ = Z̃_p'Z̃_q − Z̃_q'Z̃_p`:

* `radialRayKhat₃ = r²·(C̃_pZ̃_q − C̃_qZ̃_p)` satisfies **`K̂ = τ·m̃·v`** with
  `v = r²·W̃`, and
* **`K̂' = −τ·(m̃' + 4)·v`** — so on `{m̃ ≠ 0}` the Abel ODE
  `K̂' = −[(m̃'+4)/m̃]·K̂` has exactly the coefficient shape `2μ̂/m̃`,
  `μ̂ = (m̃'+4)/2`, consumed by `LaplaceACPropagation.lean`.

The derivation is the paper computation of §4.10(F5): zero drift kills every
`D̃`-cross defect, the closure identity rewrites `C̃` to first-order data, and
`HasDerivAt.unique` (applied to the two representations of `D̃'`) supplies the
covariance bridge with no division algebra.
-/

open MeasureTheory Filter Topology Set
open scoped RealInnerProductSpace

namespace DriftingIdentifiability
open Paper

section ZeroDriftRay

variable (τ : ℝ) (νp νq : Measure ℝ)
  [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]

omit [IsProbabilityMeasure νp] [IsProbabilityMeasure νq] in
/-- Zero drift evaluated on the ray: the two mean-shift fields agree at every
ray probe. -/
lemma zeroDrift_ray_meanShift_eq
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₃ νp) (radialMixture₃ νq)) (r : ℝ) :
    meanShift (laplaceKernel τ) (radialMixture₃ νp) (rayProbe r)
      = meanShift (laplaceKernel τ) (radialMixture₃ νq) (rayProbe r) := by
  have h := hzero (rayProbe r)
  rwa [meanShiftDrift, sub_eq_zero] at h

/-- **The zero-drift ray reduction**: `D̃_p·Z̃_q = D̃_q·Z̃_p` on the ray. -/
lemma zeroDrift_ray_D_mul_Z (hτ : 0 < τ)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₃ νp) (radialMixture₃ νq)) (r : ℝ) :
    radialRayD₃ τ νp r * radialRayZ₃ τ νq r
      = radialRayD₃ τ νq r * radialRayZ₃ τ νp r := by
  have h := congrArg (fun v : EuclideanSpace ℝ (Fin 3) => v 0)
    (zeroDrift_ray_meanShift_eq τ νp νq hzero r)
  simp only [meanShift, PiLp.smul_apply, smul_eq_mul] at h
  have hip : (∫ y, laplaceKernel τ (rayProbe r) y • (y - rayProbe r)
        ∂(radialMixture₃ νp))
      = ∫ y, laplaceWeightedDisplacement τ (rayProbe r) y ∂(radialMixture₃ νp) := rfl
  have hiq : (∫ y, laplaceKernel τ (rayProbe r) y • (y - rayProbe r)
        ∂(radialMixture₃ νq))
      = ∫ y, laplaceWeightedDisplacement τ (rayProbe r) y ∂(radialMixture₃ νq) := rfl
  rw [hip, hiq,
    ← radialRayZ₃_eq_kernelNormalizer τ hτ νp r,
    ← radialRayZ₃_eq_kernelNormalizer τ hτ νq r,
    ← radialRayD₃_eq_weightedDisplacementCoord τ hτ νp r,
    ← radialRayD₃_eq_weightedDisplacementCoord τ hτ νq r] at h
  have hZp := radialRayZ₃_pos τ hτ νp r
  have hZq := radialRayZ₃_pos τ hτ νq r
  field_simp [hZp.ne', hZq.ne'] at h
  linarith [h]

/-- Zero drift gives a **common tilted displacement** `m̃_p = m̃_q` on the
ray. -/
lemma zeroDrift_ray_M_eq (hτ : 0 < τ)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₃ νp) (radialMixture₃ νq)) (r : ℝ) :
    radialRayM₃ τ νp r = radialRayM₃ τ νq r := by
  have hZp := radialRayZ₃_pos τ hτ νp r
  have hZq := radialRayZ₃_pos τ hτ νq r
  have h := zeroDrift_ray_D_mul_Z τ νp νq hτ hzero r
  rw [radialRayM₃, radialRayM₃, div_eq_div_iff hZp.ne' hZq.ne']
  linarith [h]

/-- Zero drift gives a common tilted-displacement **derivative** on the open
ray (via `HasDerivAt.unique` on the eventually-equal functions). -/
lemma zeroDrift_ray_MDeriv_eq (hτ : 0 < τ)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₃ νp) (radialMixture₃ νq)) {r : ℝ} (hr : 0 < r) :
    radialRayMDeriv₃ τ νp r = radialRayMDeriv₃ τ νq r := by
  have hp := hasDerivAt_radialRayM₃ τ hτ νp hr
  have hq := hasDerivAt_radialRayM₃ τ hτ νq hr
  have heq : radialRayM₃ τ νp =ᶠ[𝓝 r] radialRayM₃ τ νq :=
    Filter.Eventually.of_forall fun x => zeroDrift_ray_M_eq τ νp νq hτ hzero x
  exact (hp.congr_of_eventuallyEq heq.symm).unique hq

/-- **The covariance bridge** (uniqueness of `D̃'`): for any radial mixture `ν`
whose tilted displacement agrees with `m̃_p` on the ray,
`(1/τ)Q̃_ν − Z̃_ν = m̃'·Z̃_ν + m̃·(1/τ)·Z̃d_ν`. -/
lemma zeroDrift_D_deriv_bridge (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (hM : ∀ x : ℝ, radialRayM₃ τ νp x = radialRayM₃ τ ν x) {r : ℝ} (hr : 0 < r) :
    (1 / τ) * radialRayQ₃ τ ν r - radialRayZ₃ τ ν r
      = radialRayMDeriv₃ τ νp r * radialRayZ₃ τ ν r
        + radialRayM₃ τ νp r * ((1 / τ) * radialRayZd₃ τ ν r) := by
  have hDν := hasDerivAt_radialRayD₃ τ hτ ν hr
  have hprod := (hasDerivAt_radialRayM₃ τ hτ νp hr).mul
    (hasDerivAt_radialRayZ₃ τ hτ ν hr)
  have heq : (fun x => radialRayM₃ τ νp x * radialRayZ₃ τ ν x)
      =ᶠ[𝓝 r] radialRayD₃ τ ν := by
    refine Filter.Eventually.of_forall fun x => ?_
    change radialRayM₃ τ νp x * radialRayZ₃ τ ν x = radialRayD₃ τ ν x
    rw [hM x, radialRayM₃, div_mul_cancel₀]
    exact (radialRayZ₃_pos τ hτ ν x).ne'
  exact ((hprod.congr_of_eventuallyEq heq.symm).unique hDν).symm

end ZeroDriftRay

/-! ## The system objects -/

/-- The Wronskian-type field `W̃ = Z̃_p'·Z̃_q − Z̃_q'·Z̃_p` in explicit payload
form. -/
noncomputable def radialRayW₃ (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  (1 / τ) * (radialRayZd₃ τ νp r * radialRayZ₃ τ νq r
    - radialRayZd₃ τ νq r * radialRayZ₃ τ νp r)

/-- `v = r²·W̃`, the geometric weight absorbing the `n = 3` `2/r`-term. -/
noncomputable def radialRayV₃ (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  r ^ 2 * radialRayW₃ τ νp νq r

/-- The companion alignment defect `K = C̃_pZ̃_q − C̃_qZ̃_p`. -/
noncomputable def radialRayK₃ (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  radialRayC₃ τ νp r * radialRayZ₃ τ νq r
    - radialRayC₃ τ νq r * radialRayZ₃ τ νp r

/-- The propagating object `K̂ = r²·K`. -/
noncomputable def radialRayKhat₃ (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  r ^ 2 * radialRayK₃ τ νp νq r

section System

variable (τ : ℝ) (νp νq : Measure ℝ)
  [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]

/-- The closure identity in first-order form: under zero drift,
`C̃_ν = τ·m̃'·Z̃_ν + m̃·Z̃d_ν + 4τ·Z̃_ν + (2τ/r)·m̃·Z̃_ν` for both mixtures.
This is the fully-substituted `g_ν`-form feeding the K̂ computation. -/
lemma zeroDrift_C_eq (hτ : 0 < τ)
    (ν' : Measure ℝ) [IsProbabilityMeasure ν'] (hsupp : ν' (Iio 0) = 0)
    (hM : ∀ x : ℝ, radialRayM₃ τ νp x = radialRayM₃ τ ν' x) {r : ℝ} (hr : 0 < r) :
    radialRayC₃ τ ν' r
      = τ * radialRayMDeriv₃ τ νp r * radialRayZ₃ τ ν' r
        + radialRayM₃ τ νp r * radialRayZd₃ τ ν' r
        + 4 * τ * radialRayZ₃ τ ν' r
        + (2 * τ / r) * (radialRayM₃ τ νp r * radialRayZ₃ τ ν' r) := by
  have hclosure := radialRayC₃_closure τ hτ ν' hsupp hr
  have hbridge := zeroDrift_D_deriv_bridge τ νp hτ ν' hM hr
  have hD : radialRayD₃ τ ν' r = radialRayM₃ τ νp r * radialRayZ₃ τ ν' r := by
    rw [hM r, radialRayM₃, div_mul_cancel₀]
    exact (radialRayZ₃_pos τ hτ ν' r).ne'
  -- from the bridge: Q̃ = τ·(m̃'·Z̃ + m̃·(1/τ)·Z̃d + Z̃)
  have hQ : radialRayQ₃ τ ν' r
      = τ * radialRayMDeriv₃ τ νp r * radialRayZ₃ τ ν' r
        + radialRayM₃ τ νp r * radialRayZd₃ τ ν' r
        + τ * radialRayZ₃ τ ν' r := by
    have h := hbridge
    field_simp [hτ.ne'] at h
    linarith [h]
  rw [hclosure, hQ, hD]
  ring

omit [IsProbabilityMeasure νp] in
/-- `D̃_ν = m̃·Z̃_ν` for any mixture sharing the tilted displacement. -/
lemma zeroDrift_D_eq (hτ : 0 < τ)
    (ν' : Measure ℝ) [IsProbabilityMeasure ν']
    (hM : ∀ x : ℝ, radialRayM₃ τ νp x = radialRayM₃ τ ν' x) (r : ℝ) :
    radialRayD₃ τ ν' r = radialRayM₃ τ νp r * radialRayZ₃ τ ν' r := by
  rw [hM r, radialRayM₃, div_mul_cancel₀]
  exact (radialRayZ₃_pos τ hτ ν' r).ne'

/-- **`K̂ = τ·m̃·v`** under zero drift: the companion alignment defect is the
tilted displacement times the geometric Wronskian weight (F5, first leg). -/
theorem radialRayKhat₃_eq_M_mul_V (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₃ νp) (radialMixture₃ νq)) {r : ℝ} (hr : 0 < r) :
    radialRayKhat₃ τ νp νq r
      = τ * radialRayM₃ τ νp r * radialRayV₃ τ νp νq r := by
  have gp := zeroDrift_C_eq τ νp hτ νp hsp (fun _ => rfl) hr
  have gq := zeroDrift_C_eq τ νp hτ νq hsq
    (fun x => zeroDrift_ray_M_eq τ νp νq hτ hzero x) hr
  rw [radialRayKhat₃, radialRayK₃, radialRayV₃, radialRayW₃, gp, gq]
  field_simp
  ring

/-- **`K̂' = −τ·(m̃'+4)·v`** under zero drift (F5, second leg): the Abel system
with exactly the `2μ̂/m̃` coefficient shape (`μ̂ = (m̃'+4)/2`) consumed by the
1-d propagation layer. -/
theorem hasDerivAt_radialRayKhat₃ (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₃ νp) (radialMixture₃ νq)) {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayKhat₃ τ νp νq)
      (-(τ * (radialRayMDeriv₃ τ νp r + 4)) * radialRayV₃ τ νp νq r) r := by
  have hCp := hasDerivAt_radialRayC₃ τ hτ νp hr
  have hCq := hasDerivAt_radialRayC₃ τ hτ νq hr
  have hZp := hasDerivAt_radialRayZ₃ τ hτ νp hr
  have hZq := hasDerivAt_radialRayZ₃ τ hτ νq hr
  have hr2 : HasDerivAt (fun x : ℝ => x ^ 2) (2 * r) r := by
    simpa using hasDerivAt_pow 2 r
  have hK := ((hCp.mul hZq).sub (hCq.mul hZp) :
    HasDerivAt (fun x => radialRayC₃ τ νp x * radialRayZ₃ τ νq x
        - radialRayC₃ τ νq x * radialRayZ₃ τ νp x)
      (((1 / τ) * radialRayD₃ τ νp r * radialRayZ₃ τ νq r
          + radialRayC₃ τ νp r * ((1 / τ) * radialRayZd₃ τ νq r))
        - ((1 / τ) * radialRayD₃ τ νq r * radialRayZ₃ τ νp r
          + radialRayC₃ τ νq r * ((1 / τ) * radialRayZd₃ τ νp r))) r)
  have hKhat := (hr2.mul hK :
    HasDerivAt (fun x => x ^ 2
        * (radialRayC₃ τ νp x * radialRayZ₃ τ νq x
          - radialRayC₃ τ νq x * radialRayZ₃ τ νp x))
      (2 * r * (radialRayC₃ τ νp r * radialRayZ₃ τ νq r
          - radialRayC₃ τ νq r * radialRayZ₃ τ νp r)
        + r ^ 2 * (((1 / τ) * radialRayD₃ τ νp r * radialRayZ₃ τ νq r
            + radialRayC₃ τ νp r * ((1 / τ) * radialRayZd₃ τ νq r))
          - ((1 / τ) * radialRayD₃ τ νq r * radialRayZ₃ τ νp r
            + radialRayC₃ τ νq r * ((1 / τ) * radialRayZd₃ τ νp r)))) r)
  have hfe : radialRayKhat₃ τ νp νq
      = fun x => x ^ 2 * (radialRayC₃ τ νp x * radialRayZ₃ τ νq x
          - radialRayC₃ τ νq x * radialRayZ₃ τ νp x) := rfl
  rw [hfe]
  have gp := zeroDrift_C_eq τ νp hτ νp hsp (fun _ => rfl) hr
  have gq := zeroDrift_C_eq τ νp hτ νq hsq
    (fun x => zeroDrift_ray_M_eq τ νp νq hτ hzero x) hr
  have hDp := zeroDrift_D_eq τ νp hτ νp (fun _ => rfl) r
  have hDq := zeroDrift_D_eq τ νp hτ νq
    (fun x => zeroDrift_ray_M_eq τ νp νq hτ hzero x) r
  have hval : (2 * r * (radialRayC₃ τ νp r * radialRayZ₃ τ νq r
          - radialRayC₃ τ νq r * radialRayZ₃ τ νp r)
        + r ^ 2 * (((1 / τ) * radialRayD₃ τ νp r * radialRayZ₃ τ νq r
            + radialRayC₃ τ νp r * ((1 / τ) * radialRayZd₃ τ νq r))
          - ((1 / τ) * radialRayD₃ τ νq r * radialRayZ₃ τ νp r
            + radialRayC₃ τ νq r * ((1 / τ) * radialRayZd₃ τ νp r))))
      = -(τ * (radialRayMDeriv₃ τ νp r + 4)) * radialRayV₃ τ νp νq r := by
    rw [radialRayV₃, radialRayW₃, gp, gq, hDp, hDq]
    field_simp
    ring
  rw [hval] at hKhat
  exact hKhat

end System

/-! ## Range bounds and the near-edge estimate `|K̂| ≤ τ·r²` -/

lemma radialRayZ₃_nonneg (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    0 ≤ radialRayZ₃ τ ν r := (radialRayZ₃_pos τ hτ ν r).le

lemma radialRayZ₃_le_one (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayZ₃ τ ν r ≤ 1 := by
  rw [radialRayZ₃_eq_kernelNormalizer τ hτ ν r, kernelNormalizer]
  calc (∫ y, laplaceKernel τ (rayProbe r) y ∂(radialMixture₃ ν))
      ≤ ∫ _, (1 : ℝ) ∂(radialMixture₃ ν) :=
        integral_mono (integrable_laplaceKernel_rayProbe τ hτ _ r)
          (integrable_const 1) (fun y => laplaceKernel_rayProbe_le_one τ hτ r y)
    _ = 1 := by simp

lemma radialRayC₃_nonneg (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    0 ≤ radialRayC₃ τ ν r := by
  rw [radialRayC₃_eq_companionNormalizer τ hτ ν r, kernelNormalizer]
  exact integral_nonneg fun y => laplaceCompanionKernel_rayProbe_nonneg τ hτ r y

lemma radialRayC₃_le (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayC₃ τ ν r ≤ τ := by
  rw [radialRayC₃_eq_companionNormalizer τ hτ ν r, kernelNormalizer]
  have hint : Integrable (fun y => laplaceCompanionKernel τ (rayProbe r) y)
      (radialMixture₃ ν) :=
    ⟨(continuous_laplaceCompanionKernel_rayProbe τ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := τ)
        (ae_of_all _ fun y => by
          rw [Real.norm_eq_abs,
            abs_of_nonneg (laplaceCompanionKernel_rayProbe_nonneg τ hτ r y)]
          exact laplaceCompanionKernel_rayProbe_le τ hτ r y)⟩
  calc (∫ y, laplaceCompanionKernel τ (rayProbe r) y ∂(radialMixture₃ ν))
      ≤ ∫ _, τ ∂(radialMixture₃ ν) :=
        integral_mono hint (integrable_const τ)
          (fun y => laplaceCompanionKernel_rayProbe_le τ hτ r y)
    _ = τ := by simp

/-- **The near-edge estimate** `|K̂| ≤ τ·r²`: super-linear vanishing at the
`r = 0` universal edge, and boundedness near every interior edge. -/
lemma abs_radialRayKhat₃_le (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (r : ℝ) :
    |radialRayKhat₃ τ νp νq r| ≤ τ * r ^ 2 := by
  have hCp0 := radialRayC₃_nonneg τ hτ νp r
  have hCq0 := radialRayC₃_nonneg τ hτ νq r
  have hCpτ := radialRayC₃_le τ hτ νp r
  have hCqτ := radialRayC₃_le τ hτ νq r
  have hZp0 := radialRayZ₃_nonneg τ hτ νp r
  have hZq0 := radialRayZ₃_nonneg τ hτ νq r
  have hZp1 := radialRayZ₃_le_one τ hτ νp r
  have hZq1 := radialRayZ₃_le_one τ hτ νq r
  have hK : |radialRayK₃ τ νp νq r| ≤ τ := by
    rw [radialRayK₃, abs_sub_le_iff]
    constructor
    · have h1 : radialRayC₃ τ νp r * radialRayZ₃ τ νq r ≤ τ * 1 :=
        mul_le_mul hCpτ hZq1 hZq0 hτ.le
      have h2 : 0 ≤ radialRayC₃ τ νq r * radialRayZ₃ τ νp r :=
        mul_nonneg hCq0 hZp0
      linarith
    · have h1 : radialRayC₃ τ νq r * radialRayZ₃ τ νp r ≤ τ * 1 :=
        mul_le_mul hCqτ hZp1 hZp0 hτ.le
      have h2 : 0 ≤ radialRayC₃ τ νp r * radialRayZ₃ τ νq r :=
        mul_nonneg hCp0 hZq0
      linarith
  calc |radialRayKhat₃ τ νp νq r|
      = r ^ 2 * |radialRayK₃ τ νp νq r| := by
        rw [radialRayKhat₃, abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ r ^ 2)]
    _ ≤ r ^ 2 * τ :=
        mul_le_mul_of_nonneg_left hK (by positivity)
    _ = τ * r ^ 2 := by ring

end DriftingIdentifiability
