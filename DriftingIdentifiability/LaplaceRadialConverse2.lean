import DriftingIdentifiability.LaplaceRadialSystem2
import DriftingIdentifiability.LaplaceACPropagation

/-!
# Radial Laplace converse, milestone G3 (`n = 2`): the propagation layer

Fourth G3 file: consumes the `n = 2` ray system (`System2`) and drives the
already-audited one-dimensional Abel propagation machinery
(`LaplaceACPropagation`) to conclude `K̂₂ ≡ 0`, hence `v ≡ 0`, hence
`Z̃₂_p = c·Z̃₂_q`.

This chunk (Stage A) sets up the Abel equation and the universal origin edge:
continuity through the origin, the linear displacement bound, and the `2τ`
bound on `K̂₂` near zero.  It mirrors `ConverseN` with `n = 2` constants.
-/

open MeasureTheory Filter Topology Set

namespace DriftingIdentifiability

open Paper

/-! ## The Abel equation on `{M ≠ 0}` -/

theorem hasDerivAt_radialRayKhat₂_abel (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq))
    {r : ℝ} (hr : 0 < r) (hMr : radialRayM₂ τ νp r ≠ 0) :
    HasDerivAt (radialRayKhat₂ τ νp νq)
      (-((2 * ((radialRayMDeriv₂ τ νp r + 3) / 2)) /
          radialRayM₂ τ νp r) *
        radialRayKhat₂ τ νp νq r) r := by
  have hK := hasDerivAt_radialRayKhat₂ τ hτ νp νq hsp hsq hzero hr
  have hKM := radialRayKhat₂_eq_M_mul_V τ hτ νp νq hsp hsq hzero hr
  have hval :
      -(τ * (radialRayMDeriv₂ τ νp r + 3)) * radialRayV₂ τ νp νq r =
        -((2 * ((radialRayMDeriv₂ τ νp r + 3) / 2)) /
            radialRayM₂ τ νp r) * radialRayKhat₂ τ νp νq r := by
    rw [hKM]
    field_simp [hMr]
  exact hK.congr_deriv hval

/-! ## The universal origin edge -/

/-- The axial average vanishes at the origin: `∫ cos φ = 0`. -/
lemma shellD₂_zero_left (τ s : ℝ) : shellD₂ τ 0 s = 0 := by
  rw [shellD₂]
  have hpt : ∀ φ : ℝ,
      Real.exp (-(1 / τ) * shellDist 0 s (Real.cos φ)) * (s * Real.cos φ - 0)
        = (Real.exp (-(1 / τ) * |s|) * s) * Real.cos φ := by
    intro φ
    have hd : shellDist 0 s (Real.cos φ) = |s| := by
      rw [shellDist,
        show (0 : ℝ) ^ 2 + s ^ 2 - 2 * 0 * s * Real.cos φ = s ^ 2 by ring]
      exact Real.sqrt_sq_eq_abs s
    rw [hd]; ring
  rw [setIntegral_congr_fun measurableSet_Ioc (fun φ _ => hpt φ),
    integral_const_mul]
  have hcos : (∫ φ in Ioc (-Real.pi) Real.pi, Real.cos φ) = 0 := by
    rw [← intervalIntegral.integral_of_le (by linarith [Real.pi_pos]),
      integral_cos, Real.sin_pi, Real.sin_neg, Real.sin_pi]
    ring
  rw [hcos, mul_zero, mul_zero]

lemma radialRayD₂_zero (τ : ℝ) (ν : Measure ℝ) : radialRayD₂ τ ν 0 = 0 := by
  rw [radialRayD₂,
    show (fun s => shellD₂ τ 0 s) = fun _ => (0 : ℝ) from
      funext fun s => shellD₂_zero_left τ s]
  simp

lemma continuous_radialRayZ₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] : Continuous (radialRayZ₂ τ ν) := by
  have hfe : radialRayZ₂ τ ν = fun r =>
      ∫ y, laplaceKernel τ (rayProbe₂ r) y ∂(radialMixture₂ ν) := by
    funext r; exact radialRayZ₂_eq_integral τ hτ ν r
  rw [hfe]
  refine continuous_of_dominated (bound := fun _ => (1 : ℝ))
    (fun r => ?_) (fun r => ?_) ?_ ?_
  · exact (continuous_laplaceKernel_rayProbe₂ τ r).aestronglyMeasurable
  · exact ae_of_all _ fun y => by
      rw [Real.norm_eq_abs, abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ r y)]
      exact laplaceKernel_rayProbe₂_le_one τ hτ r y
  · exact integrable_const 1
  · exact ae_of_all _ fun y => continuous_laplaceKernel_rayProbe₂_left τ y

lemma continuous_radialRayD₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] : Continuous (radialRayD₂ τ ν) := by
  have hfe : radialRayD₂ τ ν = fun r =>
      ∫ y, laplaceKernel τ (rayProbe₂ r) y * (y 0 - r) ∂(radialMixture₂ ν) := by
    funext r
    exact radialRayD₂_eq_integral τ ν r (integrable_axial_rayProbe₂ τ hτ _ r)
  rw [hfe]
  refine continuous_of_dominated (bound := fun _ => τ * Real.exp (-1))
    (fun r => ?_) (fun r => ?_) ?_ ?_
  · exact ((continuous_laplaceKernel_rayProbe₂ τ r).mul
      (by fun_prop)).aestronglyMeasurable
  · refine ae_of_all _ fun y => ?_
    rw [Real.norm_eq_abs, abs_mul,
      abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ r y)]
    calc laplaceKernel τ (rayProbe₂ r) y * |y 0 - r|
        ≤ laplaceKernel τ (rayProbe₂ r) y * ‖rayProbe₂ r - y‖ :=
          mul_le_mul_of_nonneg_left (abs_first_sub_le_norm_rayProbe₂ r y)
            (laplaceKernel_rayProbe₂_nonneg τ r y)
      _ ≤ τ * Real.exp (-1) := laplaceKernel_rayProbe₂_mul_norm_le τ hτ r y
  · exact integrable_const _
  · exact ae_of_all _ fun y =>
      (continuous_laplaceKernel_rayProbe₂_left τ y).mul (by fun_prop)

/-- The displacement profile grows at most linearly away from the origin. -/
lemma abs_radialRayD₂_le_linear (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (_hsupp : ν (Iio 0) = 0) {t : ℝ} (ht : 0 < t) :
    |radialRayD₂ τ ν t| ≤ (Real.exp (-1) + 1) * t := by
  have hbound : ∀ x : ℝ, 0 < x →
      ‖(1 / τ) * radialRayQ₂ τ ν x - radialRayZ₂ τ ν x‖ ≤ Real.exp (-1) + 1 := by
    intro x _
    rw [Real.norm_eq_abs]
    have h1 : |(1 / τ) * radialRayQ₂ τ ν x| ≤ Real.exp (-1) := by
      rw [abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ 1 / τ)]
      have hQ : |radialRayQ₂ τ ν x| ≤ τ * Real.exp (-1) := by
        rw [abs_of_nonneg (radialRayQ₂_nonneg τ hτ ν x)]
        rw [radialRayQ₂_eq_integral τ ν x (integrable_Q_payload_rayProbe₂ τ hτ _ x)]
        have hb := norm_integral_le_integral_norm
          (μ := radialMixture₂ ν)
          (f := fun y : EuclideanSpace ℝ (Fin 2) =>
            laplaceKernel τ (rayProbe₂ x) y *
              ((y 0 - x) ^ 2 / ‖rayProbe₂ x - y‖))
        simp only [Real.norm_eq_abs] at hb
        refine (le_abs_self _).trans (hb.trans ?_)
        calc (∫ y, |laplaceKernel τ (rayProbe₂ x) y *
              ((y 0 - x) ^ 2 / ‖rayProbe₂ x - y‖)| ∂(radialMixture₂ ν))
            ≤ ∫ _y, τ * Real.exp (-1) ∂(radialMixture₂ ν) :=
              integral_mono
                (integrable_Q_payload_rayProbe₂ τ hτ _ x).abs (integrable_const _)
                (fun y => abs_kernel_mul_first_sq_div_rayProbe₂_le τ hτ x y)
          _ = τ * Real.exp (-1) := by simp
      calc (1 / τ) * |radialRayQ₂ τ ν x|
          ≤ (1 / τ) * (τ * Real.exp (-1)) :=
            mul_le_mul_of_nonneg_left hQ (by positivity)
        _ = Real.exp (-1) := by field_simp
    have h2 : |radialRayZ₂ τ ν x| ≤ 1 := by
      rw [abs_of_nonneg (radialRayZ₂_pos τ hτ ν x).le]
      exact radialRayZ₂_le_one τ hτ ν x
    exact (abs_sub _ _).trans (add_le_add h1 h2)
  have hev : ∀ᶠ ε in nhdsWithin (0 : ℝ) (Ioi 0), |radialRayD₂ τ ν t| ≤
      |radialRayD₂ τ ν ε| + (Real.exp (-1) + 1) * t := by
    filter_upwards [Ioo_mem_nhdsGT ht] with ε hε
    have hMVT : ‖radialRayD₂ τ ν t - radialRayD₂ τ ν ε‖ ≤
        (Real.exp (-1) + 1) * ‖t - ε‖ := by
      refine Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
        (f' := fun x => (1 / τ) * radialRayQ₂ τ ν x - radialRayZ₂ τ ν x)
        (fun x hx => ?_) (fun x hx => hbound x (lt_of_lt_of_le hε.1 hx.1))
        (convex_Icc ε t) (left_mem_Icc.mpr hε.2.le) (right_mem_Icc.mpr hε.2.le)
      exact (hasDerivAt_radialRayD₂ τ hτ ν
        (lt_of_lt_of_le hε.1 hx.1)).hasDerivWithinAt
    rw [Real.norm_eq_abs, Real.norm_eq_abs,
      abs_of_nonneg (by linarith [hε.2] : (0 : ℝ) ≤ t - ε)] at hMVT
    calc |radialRayD₂ τ ν t|
        ≤ |radialRayD₂ τ ν ε| +
            |radialRayD₂ τ ν t - radialRayD₂ τ ν ε| := by
          have h := abs_sub_abs_le_abs_sub
            (radialRayD₂ τ ν t) (radialRayD₂ τ ν ε)
          linarith [abs_nonneg (radialRayD₂ τ ν ε)]
      _ ≤ |radialRayD₂ τ ν ε| + (Real.exp (-1) + 1) * (t - ε) := by linarith
      _ ≤ |radialRayD₂ τ ν ε| + (Real.exp (-1) + 1) * t := by
          have : (Real.exp (-1) + 1) * (t - ε) ≤ (Real.exp (-1) + 1) * t :=
            mul_le_mul_of_nonneg_left (by linarith [hε.1]) (by positivity)
          linarith
  have htend : Tendsto
      (fun ε : ℝ => |radialRayD₂ τ ν ε| + (Real.exp (-1) + 1) * t)
      (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds ((Real.exp (-1) + 1) * t)) := by
    have h0 : Tendsto (fun ε : ℝ => |radialRayD₂ τ ν ε|)
        (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds 0) := by
      have hc : ContinuousAt (fun ε : ℝ => |radialRayD₂ τ ν ε|) 0 :=
        ((continuous_radialRayD₂ τ hτ ν).abs).continuousAt
      have hlim : Tendsto (fun ε : ℝ => |radialRayD₂ τ ν ε|)
          (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds |radialRayD₂ τ ν 0|) :=
        hc.tendsto.mono_left nhdsWithin_le_nhds
      simpa [radialRayD₂_zero] using hlim
    simpa using h0.add (tendsto_const_nhds
      (f := nhdsWithin (0 : ℝ) (Ioi 0)) (x := (Real.exp (-1) + 1) * t))
  exact ge_of_tendsto htend hev

lemma radialRayM₂_le_linear (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0) {b : ℝ} (hb : 0 < b) :
    ∃ L : ℝ, 0 < L ∧ ∀ t : ℝ, 0 < t → t ≤ b → radialRayM₂ τ ν t ≤ L * t := by
  obtain ⟨z₀, hz₀mem, hz₀min⟩ := isCompact_Icc.exists_isMinOn
    (Set.nonempty_Icc.mpr hb.le) ((continuous_radialRayZ₂ τ hτ ν).continuousOn)
  have hZmin : 0 < radialRayZ₂ τ ν z₀ := radialRayZ₂_pos τ hτ ν z₀
  refine ⟨(Real.exp (-1) + 1) / radialRayZ₂ τ ν z₀, by positivity, ?_⟩
  intro t ht htb
  have hZt : radialRayZ₂ τ ν z₀ ≤ radialRayZ₂ τ ν t :=
    hz₀min (Set.mem_Icc.mpr ⟨ht.le, htb⟩)
  have hZtpos : 0 < radialRayZ₂ τ ν t := radialRayZ₂_pos τ hτ ν t
  have hD := abs_radialRayD₂_le_linear τ hτ ν hsupp ht
  rw [radialRayM₂, div_le_iff₀ hZtpos]
  calc radialRayD₂ τ ν t ≤ |radialRayD₂ τ ν t| := le_abs_self _
    _ ≤ (Real.exp (-1) + 1) * t := hD
    _ = ((Real.exp (-1) + 1) / radialRayZ₂ τ ν z₀) * t *
        radialRayZ₂ τ ν z₀ := by field_simp
    _ ≤ ((Real.exp (-1) + 1) / radialRayZ₂ τ ν z₀) * t *
        radialRayZ₂ τ ν t :=
      mul_le_mul_of_nonneg_left hZt (by positivity)

/-- Bounds needed for the companion determinant. -/
lemma radialRayC₂_nonneg (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) : 0 ≤ radialRayC₂ τ ν r := by
  rw [radialRayC₂_eq_integral τ ν r (integrable_companion_rayProbe₂ τ hτ _ r)]
  refine integral_nonneg fun y => ?_
  rw [laplaceCompanionKernel]
  exact mul_nonneg (by positivity) (laplaceKernel_rayProbe₂_nonneg τ r y)

/-- Sharp pointwise companion bound: `(τ + d)·e^{-d/τ} ≤ τ`, via `1 + u ≤ eᵘ`. -/
lemma laplaceCompanionKernel_rayProbe₂_le (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin 2)) :
    laplaceCompanionKernel τ (rayProbe₂ r) y ≤ τ := by
  rw [laplaceCompanionKernel, laplaceKernel]
  set d := ‖rayProbe₂ r - y‖ with hd
  have hd0 : 0 ≤ d := norm_nonneg _
  have hexp : Real.exp (-(1 / τ) * d) = (Real.exp (d / τ))⁻¹ := by
    rw [← Real.exp_neg]; congr 1; field_simp
  rw [hexp, mul_inv_le_iff₀ (Real.exp_pos _)]
  have hstep : d / τ + 1 ≤ Real.exp (d / τ) := Real.add_one_le_exp (d / τ)
  have hmul : τ * (d / τ + 1) ≤ τ * Real.exp (d / τ) :=
    mul_le_mul_of_nonneg_left hstep hτ.le
  have heq : τ * (d / τ + 1) = τ + d := by field_simp; ring
  rw [heq] at hmul; linarith

lemma radialRayC₂_le (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) : radialRayC₂ τ ν r ≤ τ := by
  rw [radialRayC₂_eq_integral τ ν r (integrable_companion_rayProbe₂ τ hτ _ r)]
  calc ∫ y, laplaceCompanionKernel τ (rayProbe₂ r) y ∂(radialMixture₂ ν)
      ≤ ∫ _y, τ ∂(radialMixture₂ ν) :=
        integral_mono (integrable_companion_rayProbe₂ τ hτ _ r)
          (integrable_const τ)
          (fun y => laplaceCompanionKernel_rayProbe₂_le τ hτ r y)
    _ = τ := by simp

lemma abs_radialRayK₂_le_two_tau (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (r : ℝ) : |radialRayK₂ τ νp νq r| ≤ 2 * τ := by
  have hCp0 := radialRayC₂_nonneg τ hτ νp r
  have hCq0 := radialRayC₂_nonneg τ hτ νq r
  have hZp0 := (radialRayZ₂_pos τ hτ νp r).le
  have hZq0 := (radialRayZ₂_pos τ hτ νq r).le
  have hCp := radialRayC₂_le τ hτ νp r
  have hCq := radialRayC₂_le τ hτ νq r
  have hZp := radialRayZ₂_le_one τ hτ νp r
  have hZq := radialRayZ₂_le_one τ hτ νq r
  have hp : radialRayC₂ τ νp r * radialRayZ₂ τ νq r ≤ τ := by
    calc radialRayC₂ τ νp r * radialRayZ₂ τ νq r ≤ τ * 1 :=
          mul_le_mul hCp hZq hZq0 hτ.le
      _ = τ := mul_one _
  have hq : radialRayC₂ τ νq r * radialRayZ₂ τ νp r ≤ τ := by
    calc radialRayC₂ τ νq r * radialRayZ₂ τ νp r ≤ τ * 1 :=
          mul_le_mul hCq hZp hZp0 hτ.le
      _ = τ := mul_one _
  rw [radialRayK₂]
  calc |radialRayC₂ τ νp r * radialRayZ₂ τ νq r -
        radialRayC₂ τ νq r * radialRayZ₂ τ νp r|
      ≤ |radialRayC₂ τ νp r * radialRayZ₂ τ νq r| +
        |radialRayC₂ τ νq r * radialRayZ₂ τ νp r| := abs_sub _ _
    _ = radialRayC₂ τ νp r * radialRayZ₂ τ νq r +
          radialRayC₂ τ νq r * radialRayZ₂ τ νp r := by
        rw [abs_of_nonneg (mul_nonneg hCp0 hZq0),
          abs_of_nonneg (mul_nonneg hCq0 hZp0)]
    _ ≤ 2 * τ := by linarith

end DriftingIdentifiability
