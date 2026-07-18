import DriftingIdentifiability.LaplaceRadialSystem2
import DriftingIdentifiability.LaplaceACPropagation
import DriftingIdentifiability.LaplaceACFinal

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

lemma radialRayKhat₂_bounded_near_zero (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq] :
    IsBoundedUnder (· ≤ ·) (nhdsWithin (0 : ℝ) (Ioi 0))
      (norm ∘ radialRayKhat₂ τ νp νq) := by
  refine ⟨2 * τ, ?_⟩
  rw [Filter.eventually_map]
  filter_upwards [Ioo_mem_nhdsGT one_pos] with r hr
  have hK := abs_radialRayK₂_le_two_tau τ hτ νp νq r
  rw [Function.comp_apply, Real.norm_eq_abs, radialRayKhat₂, abs_mul,
    abs_of_nonneg hr.1.le]
  calc r * |radialRayK₂ τ νp νq r| ≤ 1 * (2 * τ) :=
        mul_le_mul hr.2.le hK (abs_nonneg _) (by positivity)
    _ = 2 * τ := one_mul _

/-! ## Propagation across sign components -/

section Trichotomy

variable (τ : ℝ) (νp νq : Measure ℝ)
  [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]

/-- Right-interval Abel propagation from a left edge (origin or interior zero). -/
lemma radialRayKhat₂_eq_zero_on_Ioo_of_leftEdge (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hslackp : RadialSlack₂ τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq))
    {a b r₂ L : ℝ} (ha : 0 ≤ a) (hab : a < b) (har₂ : a < r₂) (hr₂b : r₂ < b)
    (hL : 0 < L)
    (hmpos : ∀ t : ℝ, a < t → t < b → 0 < radialRayM₂ τ νp t)
    (hlin : ∀ t : ℝ, a < t → t < r₂ → radialRayM₂ τ νp t ≤ L * (t - a))
    (hKbdd : IsBoundedUnder (· ≤ ·) (nhdsWithin a (Ioi a))
      (norm ∘ radialRayKhat₂ τ νp νq)) :
    ∀ x : ℝ, a < x → x < b → radialRayKhat₂ τ νp νq x = 0 := by
  let c : ℝ → ℝ := fun t =>
    (2 * ((radialRayMDeriv₂ τ νp t + 3) / 2)) / radialRayM₂ τ νp t
  have hccontAt : ∀ t ∈ Ioo a b, ContinuousAt c t := by
    intro t ht
    have ht0 : 0 < t := lt_of_le_of_lt ha ht.1
    have hMne : radialRayM₂ τ νp t ≠ 0 := (hmpos t ht.1 ht.2).ne'
    exact ContinuousAt.div
      (((continuousAt_radialRayMDeriv₂ τ hτ νp ht0).add
        continuousAt_const).div_const 2 |>.const_mul 2)
      (continuousAt_radialRayM₂ τ hτ νp ht0) hMne
  have hcOn : ContinuousOn c (Ioo a b) :=
    fun t ht => (hccontAt t ht).continuousWithinAt
  have hcint : ∀ u v : ℝ, u ∈ Ioo a b → v ∈ Ioo a b →
      IntervalIntegrable c volume u v := by
    intro u v hu hv
    exact ContinuousOn.intervalIntegrable
      (hcOn.mono (OrdConnected.uIcc_subset inferInstance hu hv))
  let A : ℝ → ℝ := fun z => ∫ s in r₂..z, c s
  have hAderiv : ∀ t ∈ Ioo a b, HasDerivAt A (c t) t := by
    intro t ht
    exact intervalIntegral.integral_hasDerivAt_right
      (hcint r₂ t ⟨har₂, hr₂b⟩ ht)
      (hcOn.stronglyMeasurableAtFilter isOpen_Ioo t ht)
      (hccontAt t ht)
  refine abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper
    (W := radialRayKhat₂ τ νp νq) (A := A)
    (μDeriv := fun t => (radialRayMDeriv₂ τ νp t + 3) / 2)
    (m := radialRayM₂ τ νp)
    (a := a) (b := b) (r := r₂) (δ := 1 / 2) (L := L)
    hab har₂ hr₂b (by norm_num) hL ?_ ?_ ?_ ?_ hKbdd ?_ ?_ ?_
  · intro x y hx hxy hy
    have hsub : Icc x y ⊆ Ioo a b := fun t ht =>
      ⟨lt_of_lt_of_le hx ht.1, lt_of_le_of_lt ht.2 hy⟩
    refine ContinuousOn.mul ?_ (Real.continuous_exp.comp_continuousOn ?_)
    · intro t ht
      have ht0 : 0 < t := lt_of_le_of_lt ha (hsub ht).1
      exact ((hasDerivAt_radialRayKhat₂ τ hτ νp νq hsp hsq hzero
        ht0).continuousAt).continuousWithinAt
    · intro t ht
      exact ((hAderiv t (hsub ht)).continuousAt).continuousWithinAt
  · intro x y hx hxy hy t ht
    have ht' : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    have ht0 : 0 < t := lt_of_le_of_lt ha ht'.1
    have hMne : radialRayM₂ τ νp t ≠ 0 := (hmpos t ht'.1 ht'.2).ne'
    simpa [c] using (hasDerivAt_radialRayKhat₂_abel τ hτ νp νq
      hsp hsq hzero ht0 hMne).hasDerivWithinAt
  · intro x y hx hxy hy t ht
    have ht' : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    exact (hAderiv t ht').hasDerivWithinAt
  · intro x y hx hxy hy
    have hsub : Icc x y ⊆ Ioo a b := fun t ht =>
      ⟨lt_of_lt_of_le hx ht.1, lt_of_le_of_lt ht.2 hy⟩
    intro t ht
    exact ((hAderiv t (hsub ht)).continuousAt).continuousWithinAt
  · intro t hat htr
    have ht0 : 0 < t := lt_of_le_of_lt ha hat
    have hder := radialRayMDeriv₂_ge τ hτ νp hsp hslackp ht0
    linarith
  · intro t hat htr
    exact hmpos t hat (lt_trans htr hr₂b)
  · exact hlin

/-- Left-interval Abel propagation from a right edge at an interior zero. -/
lemma radialRayKhat₂_eq_zero_on_Ioo_of_rightEdge (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hslackp : RadialSlack₂ τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq))
    {a b l₂ L : ℝ} (ha : 0 < a) (hab : a < b) (hal₂ : a < l₂) (hl₂b : l₂ < b)
    (hL : 0 < L)
    (hmneg : ∀ t : ℝ, a < t → t < b → radialRayM₂ τ νp t < 0)
    (hlin : ∀ t : ℝ, l₂ ≤ t → t < b →
      -(L * (b - t)) ≤ radialRayM₂ τ νp t) :
    ∀ x : ℝ, a < x → x < b → radialRayKhat₂ τ νp νq x = 0 := by
  let c : ℝ → ℝ := fun t =>
    (2 * ((radialRayMDeriv₂ τ νp t + 3) / 2)) / radialRayM₂ τ νp t
  have hccontAt : ∀ t ∈ Ioo a b, ContinuousAt c t := by
    intro t ht
    have ht0 : 0 < t := lt_trans ha ht.1
    have hMne : radialRayM₂ τ νp t ≠ 0 := (hmneg t ht.1 ht.2).ne
    exact ContinuousAt.div
      (((continuousAt_radialRayMDeriv₂ τ hτ νp ht0).add
        continuousAt_const).div_const 2 |>.const_mul 2)
      (continuousAt_radialRayM₂ τ hτ νp ht0) hMne
  have hcOn : ContinuousOn c (Ioo a b) :=
    fun t ht => (hccontAt t ht).continuousWithinAt
  have hcint : ∀ u v : ℝ, u ∈ Ioo a b → v ∈ Ioo a b →
      IntervalIntegrable c volume u v := by
    intro u v hu hv
    exact ContinuousOn.intervalIntegrable
      (hcOn.mono (OrdConnected.uIcc_subset inferInstance hu hv))
  let A : ℝ → ℝ := fun z => ∫ s in l₂..z, c s
  have hAderiv : ∀ t ∈ Ioo a b, HasDerivAt A (c t) t := by
    intro t ht
    exact intervalIntegral.integral_hasDerivAt_right
      (hcint l₂ t ⟨hal₂, hl₂b⟩ ht)
      (hcOn.stronglyMeasurableAtFilter isOpen_Ioo t ht)
      (hccontAt t ht)
  have hb0 : 0 < b := lt_trans ha hab
  have hKbdd : IsBoundedUnder (· ≤ ·) (nhdsWithin b (Iio b))
      (norm ∘ radialRayKhat₂ τ νp νq) :=
    (((hasDerivAt_radialRayKhat₂ τ hτ νp νq hsp hsq hzero
      hb0).continuousAt).norm.isBoundedUnder_le).mono nhdsWithin_le_nhds
  refine abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower
    (W := radialRayKhat₂ τ νp νq) (A := A)
    (μDeriv := fun t => (radialRayMDeriv₂ τ νp t + 3) / 2)
    (m := radialRayM₂ τ νp)
    (a := a) (b := b) (l := l₂) (δ := 1 / 2) (L := L)
    hab hal₂ hl₂b (by norm_num) hL ?_ ?_ ?_ ?_ hKbdd ?_ ?_ ?_
  · intro x y hx hxy hy
    have hsub : Icc x y ⊆ Ioo a b := fun t ht =>
      ⟨lt_of_lt_of_le hx ht.1, lt_of_le_of_lt ht.2 hy⟩
    refine ContinuousOn.mul ?_ (Real.continuous_exp.comp_continuousOn ?_)
    · intro t ht
      have ht0 : 0 < t := lt_trans ha (hsub ht).1
      exact ((hasDerivAt_radialRayKhat₂ τ hτ νp νq hsp hsq hzero
        ht0).continuousAt).continuousWithinAt
    · intro t ht
      exact ((hAderiv t (hsub ht)).continuousAt).continuousWithinAt
  · intro x y hx hxy hy t ht
    have ht' : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    have ht0 : 0 < t := lt_trans ha ht'.1
    have hMne : radialRayM₂ τ νp t ≠ 0 := (hmneg t ht'.1 ht'.2).ne
    simpa [c] using (hasDerivAt_radialRayKhat₂_abel τ hτ νp νq
      hsp hsq hzero ht0 hMne).hasDerivWithinAt
  · intro x y hx hxy hy t ht
    have ht' : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    exact (hAderiv t ht').hasDerivWithinAt
  · intro x y hx hxy hy
    have hsub : Icc x y ⊆ Ioo a b := fun t ht =>
      ⟨lt_of_lt_of_le hx ht.1, lt_of_le_of_lt ht.2 hy⟩
    intro t ht
    exact ((hAderiv t (hsub ht)).continuousAt).continuousWithinAt
  · intro t hlt htb
    have ht0 : 0 < t := lt_trans ha (lt_of_lt_of_le hal₂ hlt)
    have hder := radialRayMDeriv₂_ge τ hτ νp hsp hslackp ht0
    linarith
  · intro t hlt htb
    exact hmneg t (lt_of_lt_of_le hal₂ hlt) htb
  · exact hlin

/-- Outer-ray propagation, parametrized by the tail decay of `Khat`. -/
lemma radialRayKhat₂_eq_zero_on_ray_of_tendsto (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hslackp : RadialSlack₂ τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq))
    (htail : Tendsto (radialRayKhat₂ τ νp νq) atTop (nhds 0))
    {a : ℝ} (ha : 0 < a)
    (hmneg : ∀ t : ℝ, a ≤ t → radialRayM₂ τ νp t < 0) :
    ∀ x : ℝ, a ≤ x → radialRayKhat₂ τ νp νq x = 0 := by
  refine abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg
    (W := radialRayKhat₂ τ νp νq)
    (muDeriv := fun t => (radialRayMDeriv₂ τ νp t + 3) / 2)
    (m := radialRayM₂ τ νp) (a := a) ?_ ?_ ?_ hmneg htail
  · intro x b hax hxb t ht
    have ht0 : 0 < t := lt_of_lt_of_le ha (le_trans hax ht.1)
    exact ((hasDerivAt_radialRayKhat₂ τ hτ νp νq hsp hsq hzero
      ht0).continuousAt).continuousWithinAt
  · intro x b hax hxb t ht
    have ht0 : 0 < t := lt_of_lt_of_le ha (le_trans hax ht.1)
    have hMne : radialRayM₂ τ νp t ≠ 0 := (hmneg t (le_trans hax ht.1)).ne
    exact (hasDerivAt_radialRayKhat₂_abel τ hτ νp νq hsp hsq hzero
      ht0 hMne).hasDerivWithinAt
  · intro t hat
    have ht0 : 0 < t := lt_of_lt_of_le ha hat
    have hder := radialRayMDeriv₂_ge τ hτ νp hsp hslackp ht0
    linarith

end Trichotomy

/-! ## The trichotomy driver -/

/-- If `Khat` has its required tail decay, it vanishes on the whole open ray. -/
theorem radialRayKhat₂_eq_zero_of_tendsto
    (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hslackp : RadialSlack₂ τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq))
    (htail : Tendsto (radialRayKhat₂ τ νp νq) atTop (nhds 0)) :
    ∀ x : ℝ, 0 < x → radialRayKhat₂ τ νp νq x = 0 := by
  intro x₀ hx₀
  by_cases hmx₀ : radialRayM₂ τ νp x₀ = 0
  · rw [radialRayKhat₂_eq_M_mul_V τ hτ νp νq hsp hsq hzero hx₀, hmx₀]
    ring
  rcases lt_or_gt_of_ne hmx₀ with hneg | hpos
  · by_cases hS : ∃ t : ℝ, x₀ ≤ t ∧ radialRayM₂ τ νp t = 0
    · obtain ⟨t₀, ht₀x, ht₀0⟩ := hS
      let S : Set ℝ := {t : ℝ | x₀ ≤ t ∧ radialRayM₂ τ νp t = 0}
      have hSne : S.Nonempty := ⟨t₀, ht₀x, ht₀0⟩
      have hSbdd : BddBelow S := ⟨x₀, fun t ht => ht.1⟩
      have hβx : x₀ ≤ sInf S := le_csInf hSne fun t ht => ht.1
      have hβ0 : 0 < sInf S := lt_of_lt_of_le hx₀ hβx
      have hmβ : radialRayM₂ τ νp (sInf S) = 0 := by
        have hβcl := csInf_mem_closure hSne hSbdd
        have himg := (continuousAt_radialRayM₂ τ hτ νp hβ0).continuousWithinAt
          |>.mem_closure_image hβcl
        have hsub : radialRayM₂ τ νp '' S ⊆ {0} := by
          rintro y ⟨t, ht, rfl⟩
          exact ht.2
        have h0 := closure_mono hsub himg
        rwa [closure_singleton, mem_singleton_iff] at h0
      have hβgt : x₀ < sInf S := by
        rcases eq_or_lt_of_le hβx with heq | hlt
        · exact absurd (heq ▸ hmβ) hneg.ne
        · exact hlt
      have hmid : ∀ t : ℝ, x₀ ≤ t → t < sInf S → radialRayM₂ τ νp t < 0 := by
        intro t hxt htβ
        rcases lt_trichotomy (radialRayM₂ τ νp t) 0 with hlt | heq | hgt
        · exact hlt
        · exact absurd (csInf_le hSbdd ⟨hxt, heq⟩) (not_le.mpr htβ)
        · exfalso
          have hcont : ContinuousOn (radialRayM₂ τ νp) (Icc x₀ t) := fun z hz =>
            (continuousAt_radialRayM₂ τ hτ νp
              (lt_of_lt_of_le hx₀ hz.1)).continuousWithinAt
          have h0mem : (0 : ℝ) ∈ Icc (radialRayM₂ τ νp x₀)
              (radialRayM₂ τ νp t) := ⟨hneg.le, hgt.le⟩
          obtain ⟨z, hzIcc, hz0⟩ := intermediate_value_Icc hxt hcont h0mem
          have hzβ := csInf_le hSbdd ⟨hzIcc.1, hz0⟩
          linarith [hzIcc.2]
      have hnb : (radialRayM₂ τ νp) ⁻¹' (Iio 0) ∈ nhds x₀ :=
        (continuousAt_radialRayM₂ τ hτ νp hx₀) (isOpen_Iio.mem_nhds hneg)
      obtain ⟨ε, hε, hball⟩ := Metric.mem_nhds_iff.mp hnb
      let l₁ := max (x₀ - ε / 2) (x₀ / 2)
      have hl₁0 : 0 < l₁ := by
        change 0 < max (x₀ - ε / 2) (x₀ / 2)
        exact lt_max_of_lt_right (by linarith)
      have hl₁x : l₁ < x₀ := by
        change max (x₀ - ε / 2) (x₀ / 2) < x₀
        exact max_lt (by linarith) (by linarith)
      have hl₁neg : ∀ t : ℝ, l₁ < t → t ≤ x₀ → radialRayM₂ τ νp t < 0 := by
        intro t hlt hle
        apply hball
        rw [Metric.mem_ball, Real.dist_eq, abs_sub_lt_iff]
        have hleft : x₀ - ε / 2 < t := lt_of_le_of_lt (le_max_left _ _) hlt
        exact ⟨by linarith, by linarith⟩
      have hmneg_ext : ∀ t : ℝ, l₁ < t → t < sInf S →
          radialRayM₂ τ νp t < 0 := by
        intro t hlt htβ
        rcases le_or_gt t x₀ with hle | hgt
        · exact hl₁neg t hlt hle
        · exact hmid t hgt.le htβ
      obtain ⟨l₂, _, L, hl₁l₂, hl₂β, _, _, hL, hlin, _⟩ :=
        exists_Ioo_linear_bound_of_hasDerivAt_zero (f := radialRayM₂ τ νp)
          (a := sInf S) (lower := l₁) (upper := sInf S + 1)
          (lt_of_lt_of_le hl₁x hβx) (lt_add_one _)
          (hasDerivAt_radialRayM₂ τ hτ νp hβ0) hmβ
      exact radialRayKhat₂_eq_zero_on_Ioo_of_rightEdge τ νp νq hτ hsp hsq
        hslackp hzero hl₁0 (lt_of_lt_of_le hl₁x hβx) hl₁l₂ hl₂β hL
        hmneg_ext hlin x₀ hl₁x hβgt
    · have hmray : ∀ t : ℝ, x₀ ≤ t → radialRayM₂ τ νp t < 0 := by
        intro t hxt
        rcases lt_trichotomy (radialRayM₂ τ νp t) 0 with hlt | heq | hgt
        · exact hlt
        · exact absurd ⟨t, hxt, heq⟩ hS
        · exfalso
          have hcont : ContinuousOn (radialRayM₂ τ νp) (Icc x₀ t) := fun z hz =>
            (continuousAt_radialRayM₂ τ hτ νp
              (lt_of_lt_of_le hx₀ hz.1)).continuousWithinAt
          have h0mem : (0 : ℝ) ∈ Icc (radialRayM₂ τ νp x₀)
              (radialRayM₂ τ νp t) := ⟨hneg.le, hgt.le⟩
          obtain ⟨z, hzIcc, hz0⟩ := intermediate_value_Icc hxt hcont h0mem
          exact hS ⟨z, hzIcc.1, hz0⟩
      exact radialRayKhat₂_eq_zero_on_ray_of_tendsto τ νp νq hτ hsp hsq
        hslackp hzero htail hx₀ hmray x₀ le_rfl
  · have hnb : (radialRayM₂ τ νp) ⁻¹' (Ioi 0) ∈ nhds x₀ :=
      (continuousAt_radialRayM₂ τ hτ νp hx₀) (isOpen_Ioi.mem_nhds hpos)
    obtain ⟨ε, hε, hball⟩ := Metric.mem_nhds_iff.mp hnb
    have hxr₁ : x₀ < x₀ + ε / 2 := by linarith
    have hposr : ∀ t : ℝ, x₀ ≤ t → t < x₀ + ε / 2 →
        0 < radialRayM₂ τ νp t := by
      intro t hxt htr
      apply hball
      rw [Metric.mem_ball, Real.dist_eq, abs_sub_lt_iff]
      exact ⟨by linarith, by linarith⟩
    by_cases hS : ∃ t : ℝ, 0 < t ∧ t ≤ x₀ ∧ radialRayM₂ τ νp t = 0
    · obtain ⟨t₀, ht₀0, ht₀x, ht₀z⟩ := hS
      let S : Set ℝ := {t : ℝ | 0 < t ∧ t ≤ x₀ ∧ radialRayM₂ τ νp t = 0}
      have hSne : S.Nonempty := ⟨t₀, ht₀0, ht₀x, ht₀z⟩
      have hSbdd : BddAbove S := ⟨x₀, fun t ht => ht.2.1⟩
      have hαx : sSup S ≤ x₀ := csSup_le hSne fun t ht => ht.2.1
      have hα0 : 0 < sSup S := lt_of_lt_of_le ht₀0 (le_csSup hSbdd ⟨ht₀0, ht₀x, ht₀z⟩)
      have hmα : radialRayM₂ τ νp (sSup S) = 0 := by
        have hαcl := csSup_mem_closure hSne hSbdd
        have himg := (continuousAt_radialRayM₂ τ hτ νp hα0).continuousWithinAt
          |>.mem_closure_image hαcl
        have hsub : radialRayM₂ τ νp '' S ⊆ {0} := by
          rintro y ⟨t, ht, rfl⟩
          exact ht.2.2
        have h0 := closure_mono hsub himg
        rwa [closure_singleton, mem_singleton_iff] at h0
      have hαlt : sSup S < x₀ := by
        rcases eq_or_lt_of_le hαx with heq | hlt
        · exact absurd (heq ▸ hmα) hpos.ne'
        · exact hlt
      have hmid : ∀ t : ℝ, sSup S < t → t ≤ x₀ →
          0 < radialRayM₂ τ νp t := by
        intro t hαt htx
        have ht0 : 0 < t := lt_trans hα0 hαt
        rcases lt_trichotomy (radialRayM₂ τ νp t) 0 with hlt | heq | hgt
        · exfalso
          have hcont : ContinuousOn (radialRayM₂ τ νp) (Icc t x₀) := fun z hz =>
            (continuousAt_radialRayM₂ τ hτ νp
              (lt_of_lt_of_le ht0 hz.1)).continuousWithinAt
          have h0mem : (0 : ℝ) ∈ Icc (radialRayM₂ τ νp t)
              (radialRayM₂ τ νp x₀) := ⟨hlt.le, hpos.le⟩
          obtain ⟨z, hzIcc, hz0⟩ := intermediate_value_Icc htx hcont h0mem
          have hzα := le_csSup hSbdd ⟨lt_of_lt_of_le ht0 hzIcc.1, hzIcc.2, hz0⟩
          linarith [hzIcc.1]
        · exact absurd (le_csSup hSbdd ⟨ht0, htx, heq⟩) (not_le.mpr hαt)
        · exact hgt
      have hmpos_ext : ∀ t : ℝ, sSup S < t → t < x₀ + ε / 2 →
          0 < radialRayM₂ τ νp t := by
        intro t hαt htr
        rcases le_or_gt t x₀ with hle | hgt
        · exact hmid t hαt hle
        · exact hposr t hgt.le htr
      obtain ⟨_, r₂, L, _, _, hαr₂, hr₂r₁, hL, _, hlin'⟩ :=
        exists_Ioo_linear_bound_of_hasDerivAt_zero (f := radialRayM₂ τ νp)
          (a := sSup S) (lower := 0) (upper := x₀ + ε / 2)
          hα0 (lt_trans hαlt hxr₁)
          (hasDerivAt_radialRayM₂ τ hτ νp hα0) hmα
      have hKbdd : IsBoundedUnder (· ≤ ·) (nhdsWithin (sSup S) (Ioi (sSup S)))
          (norm ∘ radialRayKhat₂ τ νp νq) :=
        (((hasDerivAt_radialRayKhat₂ τ hτ νp νq hsp hsq hzero
          hα0).continuousAt).norm.isBoundedUnder_le).mono nhdsWithin_le_nhds
      exact radialRayKhat₂_eq_zero_on_Ioo_of_leftEdge τ νp νq hτ hsp hsq
        hslackp hzero hα0.le (lt_trans hαlt hxr₁) hαr₂ hr₂r₁ hL hmpos_ext
        (fun t h1 h2 => hlin' t h1 h2.le) hKbdd x₀ hαlt hxr₁
    · have hmray : ∀ t : ℝ, 0 < t → t ≤ x₀ → 0 < radialRayM₂ τ νp t := by
        intro t ht0 htx
        rcases lt_trichotomy (radialRayM₂ τ νp t) 0 with hlt | heq | hgt
        · exfalso
          have hcont : ContinuousOn (radialRayM₂ τ νp) (Icc t x₀) := fun z hz =>
            (continuousAt_radialRayM₂ τ hτ νp
              (lt_of_lt_of_le ht0 hz.1)).continuousWithinAt
          have h0mem : (0 : ℝ) ∈ Icc (radialRayM₂ τ νp t)
              (radialRayM₂ τ νp x₀) := ⟨hlt.le, hpos.le⟩
          obtain ⟨z, hzIcc, hz0⟩ := intermediate_value_Icc htx hcont h0mem
          exact hS ⟨z, lt_of_lt_of_le ht0 hzIcc.1, hzIcc.2, hz0⟩
        · exact absurd ⟨t, ht0, htx, heq⟩ hS
        · exact hgt
      have hmpos_ext : ∀ t : ℝ, 0 < t → t < x₀ + ε / 2 →
          0 < radialRayM₂ τ νp t := by
        intro t ht0 htr
        rcases le_or_gt t x₀ with hle | hgt
        · exact hmray t ht0 hle
        · exact hposr t hgt.le htr
      obtain ⟨L, hL, hlin⟩ := radialRayM₂_le_linear τ hτ νp hsp hx₀
      refine radialRayKhat₂_eq_zero_on_Ioo_of_leftEdge τ νp νq hτ hsp hsq
        hslackp hzero (le_refl 0) (lt_trans hx₀ hxr₁) hx₀ hxr₁ hL
        hmpos_ext ?_ (radialRayKhat₂_bounded_near_zero τ hτ νp νq)
        x₀ hx₀ hxr₁
      intro t ht0 htx
      have h := hlin t ht0 htx.le
      rwa [sub_zero]

end DriftingIdentifiability
