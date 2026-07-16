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

/-! ## Decay at infinity: the tail lemma and per-shell decay bounds -/

/-- **The tail lemma**: `r·ν([r,∞)) → 0` for probability measures with a first
moment — genuinely to zero, by dominated convergence on the truncated first
moment (`LaplaceHigherDim.md §4.10 (F8)`). -/
lemma tendsto_mul_measureReal_Ici_atTop (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (hmom : Integrable id ν) :
    Tendsto (fun r : ℝ => r * ν.real (Ici r)) atTop (𝓝 0) := by
  have hupper : Tendsto (fun r : ℝ => ∫ s in Ici r, s ∂ν) atTop (𝓝 0) := by
    have h := tendsto_integral_filter_of_dominated_convergence (μ := ν)
      (l := atTop) (F := fun (r : ℝ) (s : ℝ) => Set.indicator (Ici r) id s)
      (f := fun _ => 0) (bound := fun s => |s|)
      (Filter.Eventually.of_forall fun _ =>
        (measurable_id.indicator measurableSet_Ici).aestronglyMeasurable)
      (Filter.Eventually.of_forall fun _ => ae_of_all _ fun s => by
        simpa [Real.norm_eq_abs] using norm_indicator_le_norm_self (f := (id : ℝ → ℝ)) s)
      hmom.abs
      (ae_of_all _ fun s => by
        have hev : ∀ᶠ r : ℝ in atTop, Set.indicator (Ici r) (id : ℝ → ℝ) s = 0 := by
          filter_upwards [eventually_gt_atTop s] with r hr
          exact Set.indicator_of_notMem (by simpa using not_le.mpr hr) _
        exact tendsto_const_nhds.congr' (hev.mono fun r h => h.symm))
    simp only [integral_zero] at h
    refine h.congr fun r => ?_
    exact integral_indicator measurableSet_Ici
  have hnn : ∀ᶠ r : ℝ in atTop, 0 ≤ r * ν.real (Ici r) := by
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with r hr
    exact mul_nonneg hr ENNReal.toReal_nonneg
  have hle : ∀ᶠ r : ℝ in atTop, r * ν.real (Ici r) ≤ ∫ s in Ici r, s ∂ν := by
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with r _
    have hmono : (∫ _ in Ici r, r ∂ν) ≤ ∫ s in Ici r, s ∂ν :=
      setIntegral_mono_on ((integrable_const r).integrableOn)
        (hmom.integrableOn) measurableSet_Ici (fun s hs => hs)
    calc r * ν.real (Ici r) = ν.real (Ici r) • r := by rw [smul_eq_mul]; ring
      _ = ∫ _ in Ici r, r ∂ν := (setIntegral_const r).symm
      _ ≤ ∫ s in Ici r, s ∂ν := hmono
  exact squeeze_zero' hnn hle hupper

/-- The shell distance dominates the radial gap when `r, s ≥ 0`. -/
lemma abs_sub_le_shellDist {r s u : ℝ} (hr : 0 ≤ r) (hs : 0 ≤ s) (hu : u ≤ 1) :
    |r - s| ≤ shellDist r s u := by
  rw [shellDist, ← Real.sqrt_sq_eq_abs]
  apply Real.sqrt_le_sqrt
  nlinarith [mul_nonneg hr hs]

lemma shellZ_le_one (τ : ℝ) (hτ : 0 < τ) (r s : ℝ) : shellZ τ r s ≤ 1 := by
  rw [shellZ]
  have hbound : ∀ u ∈ Ioc (-1 : ℝ) 1,
      Real.exp (-(1 / τ) * shellDist r s u) ≤ 1 := by
    intro u _
    rw [Real.exp_le_one_iff]
    have hd : 0 ≤ shellDist r s u := Real.sqrt_nonneg _
    have h1τ : 0 ≤ 1 / τ := by positivity
    nlinarith
  calc (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1, Real.exp (-(1 / τ) * shellDist r s u)
      ≤ (1 / 2) * ∫ _ in Ioc (-1 : ℝ) 1, (1 : ℝ) := by
        refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
        refine setIntegral_mono_on ?_
          (continuous_const.integrableOn_Icc.mono_set Ioc_subset_Icc_self)
          measurableSet_Ioc hbound
        have hc : Continuous fun u : ℝ => Real.exp (-(1 / τ) * shellDist r s u) := by
          unfold shellDist
          fun_prop
        exact hc.integrableOn_Icc.mono_set Ioc_subset_Icc_self
    _ = 1 := by
        rw [setIntegral_const, Real.volume_real_Ioc_of_le (by norm_num), smul_eq_mul]
        ring

lemma shellZ_nonneg (τ r s : ℝ) : 0 ≤ shellZ τ r s := by
  rw [shellZ]
  refine mul_nonneg (by norm_num) (integral_nonneg fun u => (Real.exp_pos _).le)

/-- Per-shell decay: `shellZ ≤ e^{−|r−s|/τ}` for nonnegative radii. -/
lemma shellZ_le_exp (τ : ℝ) (hτ : 0 < τ) {r s : ℝ} (hr : 0 ≤ r) (hs : 0 ≤ s) :
    shellZ τ r s ≤ Real.exp (-(1 / τ) * |r - s|) := by
  rw [shellZ]
  have hbound : ∀ u ∈ Ioc (-1 : ℝ) 1,
      Real.exp (-(1 / τ) * shellDist r s u) ≤ Real.exp (-(1 / τ) * |r - s|) := by
    intro u hu
    apply Real.exp_le_exp.mpr
    have h := abs_sub_le_shellDist hr hs hu.2
    have h1τ : (0 : ℝ) ≤ 1 / τ := by positivity
    nlinarith
  calc (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1, Real.exp (-(1 / τ) * shellDist r s u)
      ≤ (1 / 2) * ∫ _ in Ioc (-1 : ℝ) 1, Real.exp (-(1 / τ) * |r - s|) := by
        refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
        refine setIntegral_mono_on ?_
          (continuous_const.integrableOn_Icc.mono_set Ioc_subset_Icc_self)
          measurableSet_Ioc hbound
        have hc : Continuous fun u : ℝ => Real.exp (-(1 / τ) * shellDist r s u) := by
          unfold shellDist
          fun_prop
        exact hc.integrableOn_Icc.mono_set Ioc_subset_Icc_self
    _ = Real.exp (-(1 / τ) * |r - s|) := by
        rw [setIntegral_const, Real.volume_real_Ioc_of_le (by norm_num), smul_eq_mul]
        ring

/-- The Matérn-3/2 style profile `(τ+d)e^{−d/τ}` is antitone on `[0,∞)`. -/
lemma matern_antitone (τ : ℝ) (hτ : 0 < τ) {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) :
    (τ + b) * Real.exp (-(1 / τ) * b) ≤ (τ + a) * Real.exp (-(1 / τ) * a) := by
  have hc : 0 ≤ b - a := by linarith
  have hexp : (b - a) / τ + 1 ≤ Real.exp ((b - a) / τ) := Real.add_one_le_exp _
  have key : τ + b ≤ (τ + a) * Real.exp ((b - a) / τ) := by
    have h1 : (τ + a) * ((b - a) / τ + 1) ≤ (τ + a) * Real.exp ((b - a) / τ) :=
      mul_le_mul_of_nonneg_left hexp (by linarith)
    have h2 : τ + b ≤ (τ + a) * ((b - a) / τ + 1) := by
      have hexpand : (τ + a) * ((b - a) / τ + 1) = τ + b + a * (b - a) / τ := by
        field_simp
        ring
      have hnn : 0 ≤ a * (b - a) / τ := by positivity
      linarith
    linarith
  have hmul := mul_le_mul_of_nonneg_right key (Real.exp_pos (-(1 / τ) * b)).le
  rw [mul_assoc, ← Real.exp_add] at hmul
  have harg : (b - a) / τ + -(1 / τ) * b = -(1 / τ) * a := by
    field_simp
    ring
  rwa [harg] at hmul

/-- The Matérn-3/2 style profile is bounded by `τ` on `[0,∞)`. -/
lemma matern_le (τ : ℝ) (hτ : 0 < τ) {d : ℝ} (hd : 0 ≤ d) :
    (τ + d) * Real.exp (-(1 / τ) * d) ≤ τ := by
  have h := matern_antitone τ hτ (le_refl 0) hd
  simpa using h

/-- Per-shell companion decay: `shellC ≤ (τ+|r−s|)e^{−|r−s|/τ}` for nonnegative
radii. -/
lemma shellC_le_matern (τ : ℝ) (hτ : 0 < τ) {r s : ℝ} (hr : 0 ≤ r) (hs : 0 ≤ s) :
    shellC τ r s ≤ (τ + |r - s|) * Real.exp (-(1 / τ) * |r - s|) := by
  rw [shellC]
  have hbound : ∀ u ∈ Ioc (-1 : ℝ) 1,
      (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u)
        ≤ (τ + |r - s|) * Real.exp (-(1 / τ) * |r - s|) := by
    intro u hu
    exact matern_antitone τ hτ (abs_nonneg _) (abs_sub_le_shellDist hr hs hu.2)
  calc (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1,
        (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u)
      ≤ (1 / 2) * ∫ _ in Ioc (-1 : ℝ) 1,
          (τ + |r - s|) * Real.exp (-(1 / τ) * |r - s|) := by
        refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
        refine setIntegral_mono_on ?_
          (continuous_const.integrableOn_Icc.mono_set Ioc_subset_Icc_self)
          measurableSet_Ioc hbound
        have hc : Continuous fun u : ℝ =>
            (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u) := by
          unfold shellDist
          fun_prop
        exact hc.integrableOn_Icc.mono_set Ioc_subset_Icc_self
    _ = (τ + |r - s|) * Real.exp (-(1 / τ) * |r - s|) := by
        rw [setIntegral_const, Real.volume_real_Ioc_of_le (by norm_num), smul_eq_mul]
        ring

lemma shellC_nonneg (τ : ℝ) (hτ : 0 < τ) (r s : ℝ) : 0 ≤ shellC τ r s := by
  rw [shellC]
  refine mul_nonneg (by norm_num) (integral_nonneg fun u => ?_)
  have hd : 0 ≤ shellDist r s u := Real.sqrt_nonneg _
  positivity

lemma shellC_le (τ : ℝ) (hτ : 0 < τ) (r s : ℝ) : shellC τ r s ≤ τ := by
  rw [shellC]
  have hbound : ∀ u ∈ Ioc (-1 : ℝ) 1,
      (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u) ≤ τ := by
    intro u _
    exact matern_le τ hτ (Real.sqrt_nonneg _)
  calc (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1,
        (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u)
      ≤ (1 / 2) * ∫ _ in Ioc (-1 : ℝ) 1, τ := by
        refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
        refine setIntegral_mono_on ?_
          (continuous_const.integrableOn_Icc.mono_set Ioc_subset_Icc_self)
          measurableSet_Ioc hbound
        have hc : Continuous fun u : ℝ =>
            (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u) := by
          unfold shellDist
          fun_prop
        exact hc.integrableOn_Icc.mono_set Ioc_subset_Icc_self
    _ = τ := by
        rw [setIntegral_const, Real.volume_real_Ioc_of_le (by norm_num), smul_eq_mul]
        ring

lemma continuous_shellZ (τ : ℝ) (hτ : 0 < τ) (r : ℝ) :
    Continuous fun s : ℝ => shellZ τ r s := by
  unfold shellZ
  refine continuous_const.mul ?_
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-1 : ℝ) 1)) :=
    ⟨by rw [Measure.restrict_apply_univ, Real.volume_Ioc]; exact ENNReal.ofReal_lt_top⟩
  refine continuous_of_dominated (bound := fun _ => 1) (fun s => ?_) (fun s => ?_) ?_ ?_
  · have hc : Continuous fun u : ℝ => Real.exp (-(1 / τ) * shellDist r s u) := by
      unfold shellDist
      fun_prop
    exact hc.aestronglyMeasurable
  · refine ae_of_all _ fun u => ?_
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _), Real.exp_le_one_iff]
    have hd : 0 ≤ shellDist r s u := Real.sqrt_nonneg _
    have h1τ : (0 : ℝ) ≤ 1 / τ := by positivity
    nlinarith
  · exact integrable_const 1
  · refine ae_of_all _ fun u => ?_
    have hd : Continuous fun s : ℝ => shellDist r s u := by
      unfold shellDist
      fun_prop
    exact Real.continuous_exp.comp (hd.const_mul (-(1 / τ)))

lemma continuous_shellC (τ : ℝ) (hτ : 0 < τ) (r : ℝ) :
    Continuous fun s : ℝ => shellC τ r s := by
  unfold shellC
  refine continuous_const.mul ?_
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-1 : ℝ) 1)) :=
    ⟨by rw [Measure.restrict_apply_univ, Real.volume_Ioc]; exact ENNReal.ofReal_lt_top⟩
  refine continuous_of_dominated (bound := fun _ => τ) (fun s => ?_) (fun s => ?_) ?_ ?_
  · have hc : Continuous fun u : ℝ =>
        (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u) := by
      unfold shellDist
      fun_prop
    exact hc.aestronglyMeasurable
  · refine ae_of_all _ fun u => ?_
    have hd : 0 ≤ shellDist r s u := Real.sqrt_nonneg _
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    exact matern_le τ hτ hd
  · exact integrable_const τ
  · refine ae_of_all _ fun u => ?_
    have hd : Continuous fun s : ℝ => shellDist r s u := by
      unfold shellDist
      fun_prop
    exact (continuous_const.add hd).mul
      (Real.continuous_exp.comp (hd.const_mul (-(1 / τ))))

lemma prob_measureReal_le_one {ν : Measure ℝ} [IsProbabilityMeasure ν] (s : Set ℝ) :
    ν.real s ≤ 1 := by
  have h : ν s ≤ 1 := prob_le_one
  have h2 := ENNReal.toReal_mono ENNReal.one_ne_top h
  rw [ENNReal.toReal_one] at h2
  exact h2

lemma measureReal_nonneg' {ν : Measure ℝ} (s : Set ℝ) : 0 ≤ ν.real s :=
  ENNReal.toReal_nonneg

/-- **Normalizer split bound**: `Z̃(r) ≤ e^{−r/(2τ)} + ν([r/2,∞))`. -/
lemma radialRayZ₃_le_split (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 ≤ r) :
    radialRayZ₃ τ ν r
      ≤ Real.exp (-(1 / τ) * (r / 2)) + ν.real (Ici (r / 2)) := by
  have hint : Integrable (fun s => shellZ τ r s) ν :=
    ⟨(continuous_shellZ τ hτ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := 1) (ae_of_all _ fun s => by
        rw [Real.norm_eq_abs, abs_of_nonneg (shellZ_nonneg τ r s)]
        exact shellZ_le_one τ hτ r s)⟩
  rw [radialRayZ₃, ← integral_add_compl (measurableSet_Ici (a := r / 2)) hint]
  have h1 : (∫ s in Ici (r / 2), shellZ τ r s ∂ν) ≤ ν.real (Ici (r / 2)) := by
    calc (∫ s in Ici (r / 2), shellZ τ r s ∂ν)
        ≤ ∫ _ in Ici (r / 2), (1 : ℝ) ∂ν :=
          setIntegral_mono_on hint.integrableOn ((integrable_const 1).integrableOn)
            measurableSet_Ici (fun s _ => shellZ_le_one τ hτ r s)
      _ = ν.real (Ici (r / 2)) := by rw [setIntegral_const, smul_eq_mul, mul_one]
  have h2 : (∫ s in (Ici (r / 2))ᶜ, shellZ τ r s ∂ν)
      ≤ Real.exp (-(1 / τ) * (r / 2)) := by
    have hae : ∀ᵐ s ∂(ν.restrict (Ici (r / 2))ᶜ),
        shellZ τ r s ≤ Real.exp (-(1 / τ) * (r / 2)) := by
      filter_upwards [ae_restrict_of_ae (radial_ae_nonneg hsupp),
        ae_restrict_mem (measurableSet_Ici (a := r / 2)).compl] with s hs hmem
      have hlt : s < r / 2 := by
        simpa [mem_compl_iff, mem_Ici, not_le] using hmem
      refine (shellZ_le_exp τ hτ hr hs).trans ?_
      apply Real.exp_le_exp.mpr
      have habs : r / 2 ≤ |r - s| := by
        rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ r - s)]
        linarith
      have h1τ : (0 : ℝ) ≤ 1 / τ := by positivity
      nlinarith
    calc (∫ s in (Ici (r / 2))ᶜ, shellZ τ r s ∂ν)
        ≤ ∫ _ in (Ici (r / 2))ᶜ, Real.exp (-(1 / τ) * (r / 2)) ∂ν :=
          setIntegral_mono_ae_restrict hint.integrableOn
            ((integrable_const _).integrableOn) hae
      _ = ν.real (Ici (r / 2))ᶜ * Real.exp (-(1 / τ) * (r / 2)) := by
          rw [setIntegral_const, smul_eq_mul]
      _ ≤ 1 * Real.exp (-(1 / τ) * (r / 2)) :=
          mul_le_mul_of_nonneg_right (prob_measureReal_le_one _) (Real.exp_pos _).le
      _ = Real.exp (-(1 / τ) * (r / 2)) := one_mul _
  linarith

/-- **Companion split bound**:
`C̃(r) ≤ (τ+r/2)e^{−r/(2τ)} + τ·ν([r/2,∞))`. -/
lemma radialRayC₃_le_split (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 ≤ r) :
    radialRayC₃ τ ν r
      ≤ (τ + r / 2) * Real.exp (-(1 / τ) * (r / 2)) + τ * ν.real (Ici (r / 2)) := by
  have hint : Integrable (fun s => shellC τ r s) ν :=
    ⟨(continuous_shellC τ hτ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := τ) (ae_of_all _ fun s => by
        rw [Real.norm_eq_abs, abs_of_nonneg (shellC_nonneg τ hτ r s)]
        exact shellC_le τ hτ r s)⟩
  rw [radialRayC₃, ← integral_add_compl (measurableSet_Ici (a := r / 2)) hint]
  have h1 : (∫ s in Ici (r / 2), shellC τ r s ∂ν) ≤ τ * ν.real (Ici (r / 2)) := by
    calc (∫ s in Ici (r / 2), shellC τ r s ∂ν)
        ≤ ∫ _ in Ici (r / 2), τ ∂ν :=
          setIntegral_mono_on hint.integrableOn ((integrable_const τ).integrableOn)
            measurableSet_Ici (fun s _ => shellC_le τ hτ r s)
      _ = τ * ν.real (Ici (r / 2)) := by rw [setIntegral_const, smul_eq_mul]; ring
  have h2 : (∫ s in (Ici (r / 2))ᶜ, shellC τ r s ∂ν)
      ≤ (τ + r / 2) * Real.exp (-(1 / τ) * (r / 2)) := by
    have hae : ∀ᵐ s ∂(ν.restrict (Ici (r / 2))ᶜ),
        shellC τ r s ≤ (τ + r / 2) * Real.exp (-(1 / τ) * (r / 2)) := by
      filter_upwards [ae_restrict_of_ae (radial_ae_nonneg hsupp),
        ae_restrict_mem (measurableSet_Ici (a := r / 2)).compl] with s hs hmem
      have hlt : s < r / 2 := by
        simpa [mem_compl_iff, mem_Ici, not_le] using hmem
      refine (shellC_le_matern τ hτ hr hs).trans ?_
      refine matern_antitone τ hτ (by positivity) ?_
      rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ r - s)]
      linarith
    calc (∫ s in (Ici (r / 2))ᶜ, shellC τ r s ∂ν)
        ≤ ∫ _ in (Ici (r / 2))ᶜ, (τ + r / 2) * Real.exp (-(1 / τ) * (r / 2)) ∂ν :=
          setIntegral_mono_ae_restrict hint.integrableOn
            ((integrable_const _).integrableOn) hae
      _ = ν.real (Ici (r / 2))ᶜ * ((τ + r / 2) * Real.exp (-(1 / τ) * (r / 2))) := by
          rw [setIntegral_const, smul_eq_mul]
      _ ≤ 1 * ((τ + r / 2) * Real.exp (-(1 / τ) * (r / 2))) := by
          refine mul_le_mul_of_nonneg_right (prob_measureReal_le_one _) ?_
          have : (0 : ℝ) ≤ τ + r / 2 := by positivity
          positivity
      _ = (τ + r / 2) * Real.exp (-(1 / τ) * (r / 2)) := one_mul _
  linarith

/-! ## `K̂ → 0` at infinity -/

/-- Natural-power version of the polynomial×exponential decay. -/
lemma tendsto_nat_pow_mul_exp_neg_mul (n : ℕ) {c : ℝ} (hc : 0 < c) :
    Tendsto (fun r : ℝ => r ^ n * Real.exp (-c * r)) atTop (𝓝 0) := by
  have h := tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero (n : ℝ) c hc
  refine h.congr' ?_
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with r hr
  rw [Real.rpow_natCast]

/-- The scaled tail vanishes: `(r/2)·ν([r/2,∞)) → 0` under a first moment. -/
lemma tendsto_half_mul_tail (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (hmom : Integrable id ν) :
    Tendsto (fun r : ℝ => (r / 2) * ν.real (Ici (r / 2))) atTop (𝓝 0) := by
  have hhalf : Tendsto (fun r : ℝ => r / 2) atTop atTop :=
    tendsto_atTop_atTop.mpr fun b => ⟨2 * b, fun a ha => by linarith⟩
  have h := (tendsto_mul_measureReal_Ici_atTop ν hmom).comp hhalf
  exact h.congr fun r => rfl

/-- The pointwise decay envelope for `K̂`. -/
lemma abs_radialRayKhat₃_le_envelope (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    {r : ℝ} (hr : 0 ≤ r) :
    ‖radialRayKhat₃ τ νp νq r‖
      ≤ 2 * (τ * (r ^ 2 * Real.exp (-(1 / (2 * τ)) * r))
          + (1 / 2) * (r ^ 3 * Real.exp (-(1 / (2 * τ)) * r)))
        + (2 * (τ * (r ^ 2 * Real.exp (-(1 / (2 * τ)) * r))
            + (1 / 2) * (r ^ 3 * Real.exp (-(1 / (2 * τ)) * r)))
          + (2 * (τ * (r ^ 2 * Real.exp (-(1 / (2 * τ)) * r)))
            + 2 * (4 * τ * ((r / 2) * νp.real (Ici (r / 2))
              * ((r / 2) * νq.real (Ici (r / 2))))))) := by
  have hEexp : Real.exp (-(1 / τ) * (r / 2)) = Real.exp (-(1 / (2 * τ)) * r) := by
    congr 1
    field_simp
  set E : ℝ := Real.exp (-(1 / (2 * τ)) * r) with hEdef
  set Tp : ℝ := νp.real (Ici (r / 2)) with hTpdef
  set Tq : ℝ := νq.real (Ici (r / 2)) with hTqdef
  have hE0 : 0 < E := Real.exp_pos _
  have hE1 : E ≤ 1 := by
    rw [hEdef, Real.exp_le_one_iff]
    have hc : (0 : ℝ) < 1 / (2 * τ) := by positivity
    nlinarith
  have hTp0 : 0 ≤ Tp := ENNReal.toReal_nonneg
  have hTq0 : 0 ≤ Tq := ENNReal.toReal_nonneg
  have hTp1 : Tp ≤ 1 := prob_measureReal_le_one _
  have hTq1 : Tq ≤ 1 := prob_measureReal_le_one _
  have hZp := radialRayZ₃_le_split τ hτ νp hsp hr
  have hZq := radialRayZ₃_le_split τ hτ νq hsq hr
  have hCp := radialRayC₃_le_split τ hτ νp hsp hr
  have hCq := radialRayC₃_le_split τ hτ νq hsq hr
  rw [hEexp] at hZp hZq hCp hCq
  have hZp0 := radialRayZ₃_nonneg τ hτ νp r
  have hZq0 := radialRayZ₃_nonneg τ hτ νq r
  have hCp0 := radialRayC₃_nonneg τ hτ νp r
  have hCq0 := radialRayC₃_nonneg τ hτ νq r
  have hKabs : ‖radialRayKhat₃ τ νp νq r‖
      ≤ r ^ 2 * (radialRayC₃ τ νp r * radialRayZ₃ τ νq r
        + radialRayC₃ τ νq r * radialRayZ₃ τ νp r) := by
    rw [Real.norm_eq_abs, radialRayKhat₃, radialRayK₃, abs_mul,
      abs_of_nonneg (by positivity : (0:ℝ) ≤ r ^ 2)]
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    calc |radialRayC₃ τ νp r * radialRayZ₃ τ νq r
          - radialRayC₃ τ νq r * radialRayZ₃ τ νp r|
        ≤ |radialRayC₃ τ νp r * radialRayZ₃ τ νq r|
          + |radialRayC₃ τ νq r * radialRayZ₃ τ νp r| := abs_sub _ _
      _ = radialRayC₃ τ νp r * radialRayZ₃ τ νq r
          + radialRayC₃ τ νq r * radialRayZ₃ τ νp r := by
          rw [abs_of_nonneg (mul_nonneg hCp0 hZq0),
            abs_of_nonneg (mul_nonneg hCq0 hZp0)]
  have hτr : (0 : ℝ) ≤ τ + r / 2 := by positivity
  have hprod1 : radialRayC₃ τ νp r * radialRayZ₃ τ νq r
      ≤ ((τ + r / 2) * E + τ * Tp) * (E + Tq) := by
    refine mul_le_mul hCp hZq hZq0 ?_
    positivity
  have hprod2 : radialRayC₃ τ νq r * radialRayZ₃ τ νp r
      ≤ ((τ + r / 2) * E + τ * Tq) * (E + Tp) := by
    refine mul_le_mul hCq hZp hZp0 ?_
    positivity
  have hmid : ‖radialRayKhat₃ τ νp νq r‖
      ≤ r ^ 2 * (((τ + r / 2) * E + τ * Tp) * (E + Tq)
        + ((τ + r / 2) * E + τ * Tq) * (E + Tp)) := by
    refine hKabs.trans ?_
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    exact add_le_add hprod1 hprod2
  refine hmid.trans ?_
  have e1 : r ^ 2 * ((τ + r / 2) * E * E) ≤ r ^ 2 * ((τ + r / 2) * E) := by
    have hEE : (τ + r / 2) * E * E ≤ (τ + r / 2) * E * 1 := by
      refine mul_le_mul_of_nonneg_left hE1 ?_
      positivity
    nlinarith
  have e2 : r ^ 2 * ((τ + r / 2) * E * Tq) ≤ r ^ 2 * ((τ + r / 2) * E) := by
    have hET : (τ + r / 2) * E * Tq ≤ (τ + r / 2) * E * 1 := by
      refine mul_le_mul_of_nonneg_left hTq1 ?_
      positivity
    nlinarith
  have e2' : r ^ 2 * ((τ + r / 2) * E * Tp) ≤ r ^ 2 * ((τ + r / 2) * E) := by
    have hET : (τ + r / 2) * E * Tp ≤ (τ + r / 2) * E * 1 := by
      refine mul_le_mul_of_nonneg_left hTp1 ?_
      positivity
    nlinarith
  have e3 : r ^ 2 * (τ * Tp * E) ≤ r ^ 2 * (τ * E) := by
    have hTE : τ * Tp * E ≤ τ * 1 * E := by
      refine mul_le_mul_of_nonneg_right ?_ hE0.le
      exact mul_le_mul_of_nonneg_left hTp1 hτ.le
    nlinarith
  have e3' : r ^ 2 * (τ * Tq * E) ≤ r ^ 2 * (τ * E) := by
    have hTE : τ * Tq * E ≤ τ * 1 * E := by
      refine mul_le_mul_of_nonneg_right ?_ hE0.le
      exact mul_le_mul_of_nonneg_left hTq1 hτ.le
    nlinarith
  nlinarith [e1, e2, e2', e3, e3']

/-- **`K̂ → 0` at infinity** under first moments on both radial profiles —
the full-filter decay of `LaplaceHigherDim.md §4.10 (F8)`. -/
theorem tendsto_radialRayKhat₃_atTop (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable id νp) (hmq : Integrable id νq) :
    Tendsto (radialRayKhat₃ τ νp νq) atTop (𝓝 0) := by
  have hcpos : (0 : ℝ) < 1 / (2 * τ) := by positivity
  have hQ1 : Tendsto (fun r : ℝ => τ * (r ^ 2 * Real.exp (-(1 / (2 * τ)) * r))
      + (1 / 2) * (r ^ 3 * Real.exp (-(1 / (2 * τ)) * r))) atTop (𝓝 0) := by
    have h1 := (tendsto_nat_pow_mul_exp_neg_mul 2 hcpos).const_mul τ
    have h2 := (tendsto_nat_pow_mul_exp_neg_mul 3 hcpos).const_mul (1 / 2 : ℝ)
    have h := h1.add h2
    simpa using h
  have hQ3 : Tendsto (fun r : ℝ => τ * (r ^ 2 * Real.exp (-(1 / (2 * τ)) * r)))
      atTop (𝓝 0) := by
    have h := (tendsto_nat_pow_mul_exp_neg_mul 2 hcpos).const_mul τ
    simpa using h
  have hP4 : Tendsto (fun r : ℝ => 4 * τ * ((r / 2) * νp.real (Ici (r / 2))
      * ((r / 2) * νq.real (Ici (r / 2))))) atTop (𝓝 0) := by
    have h := ((tendsto_half_mul_tail νp hmp).mul
      (tendsto_half_mul_tail νq hmq)).const_mul (4 * τ)
    simpa using h
  have hΨ : Tendsto (fun r : ℝ =>
      2 * (τ * (r ^ 2 * Real.exp (-(1 / (2 * τ)) * r))
          + (1 / 2) * (r ^ 3 * Real.exp (-(1 / (2 * τ)) * r)))
        + (2 * (τ * (r ^ 2 * Real.exp (-(1 / (2 * τ)) * r))
            + (1 / 2) * (r ^ 3 * Real.exp (-(1 / (2 * τ)) * r)))
          + (2 * (τ * (r ^ 2 * Real.exp (-(1 / (2 * τ)) * r)))
            + 2 * (4 * τ * ((r / 2) * νp.real (Ici (r / 2))
              * ((r / 2) * νq.real (Ici (r / 2)))))))) atTop (𝓝 0) := by
    have h := (hQ1.const_mul 2).add
      ((hQ1.const_mul 2).add ((hQ3.const_mul 2).add (hP4.const_mul 2)))
    simpa using h
  refine squeeze_zero_norm' ?_ hΨ
  filter_upwards [eventually_ge_atTop (0 : ℝ)] with r hr
  exact abs_radialRayKhat₃_le_envelope τ hτ νp νq hsp hsq hr

end DriftingIdentifiability

