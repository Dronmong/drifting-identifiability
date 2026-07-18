import DriftingIdentifiability.LaplaceRadialConverse2
import DriftingIdentifiability.LaplaceRadialFourier
import DriftingIdentifiability.LaplaceRadialFarField

/-!
# Radial Laplace converse, milestone G3 (`n = 2`): the global endgame

Fifth and final G3 file.  The `n = 2` chart uses the uniform angle
`φ ↦ (cos φ, sin φ)` rather than a Haar sphere, so the master radiality step
`Z_ν(x) = Z̃₂(‖x‖)` is a genuine circle-rotation argument: a nonzero probe
`x` has a polar angle `θ` (`x = ‖x‖·(cos θ, sin θ)`), and the periodic shift
`φ ↦ φ - θ` moves the arbitrary-probe circle average onto the axial one.

With radiality in hand the endgame is identical to every dimension: feed ray
proportionality `Z̃₂_p = c·Z̃₂_q` (from `Converse2`) into the already-certified
Euclidean Laplace smoothing injectivity `L4`, then force `c = 1` by comparing
total masses.
-/

open MeasureTheory Filter Topology Set Metric

namespace DriftingIdentifiability

open Paper

/-! ## Circle geometry -/

lemma norm_circleChart (φ : ℝ) : ‖circleChart φ‖ = 1 := by
  rw [← Real.sqrt_sq (norm_nonneg (circleChart φ)), EuclideanSpace.real_norm_sq_eq,
    Fin.sum_univ_two, circleChart_apply_zero, circleChart_apply_one,
    Real.cos_sq_add_sin_sq, Real.sqrt_one]

/-- Every nonzero planar probe has a polar angle. -/
private lemma exists_polar₂ (x : EuclideanSpace ℝ (Fin 2)) (hx : x ≠ 0) :
    ∃ θ : ℝ, x 0 = ‖x‖ * Real.cos θ ∧ x 1 = ‖x‖ * Real.sin θ := by
  have hr : 0 < ‖x‖ := norm_pos_iff.mpr hx
  have hnormsq : ‖x‖ ^ 2 = (x 0) ^ 2 + (x 1) ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two]
  set a := x 0 / ‖x‖ with ha
  set b := x 1 / ‖x‖ with hb
  have hab : a ^ 2 + b ^ 2 = 1 := by
    rw [ha, hb, div_pow, div_pow, ← add_div, ← hnormsq, div_self (by positivity)]
  have ha1 : a ≤ 1 := by nlinarith [sq_nonneg b]
  have ha1' : -1 ≤ a := by nlinarith [sq_nonneg b]
  refine ⟨if 0 ≤ b then Real.arccos a else -Real.arccos a, ?_, ?_⟩
  · split_ifs with h
    · rw [Real.cos_arccos ha1' ha1, ha]; field_simp
    · rw [Real.cos_neg, Real.cos_arccos ha1' ha1, ha]; field_simp
  · have hsin : Real.sin (Real.arccos a) = Real.sqrt (1 - a ^ 2) := Real.sin_arccos a
    have hb2 : Real.sqrt (1 - a ^ 2) = |b| := by
      rw [show 1 - a ^ 2 = b ^ 2 by linarith [hab], Real.sqrt_sq_eq_abs]
    split_ifs with h
    · rw [hsin, hb2, abs_of_nonneg h, hb]; field_simp
    · rw [Real.sin_neg, hsin, hb2, abs_of_neg (not_le.mp h), hb]; field_simp

/-- The probe distance at an arbitrary planar `x`, in terms of the polar angle. -/
lemma norm_sub_smul_circleChart {x : EuclideanSpace ℝ (Fin 2)} (s φ : ℝ)
    {θ : ℝ} (hx0 : x 0 = ‖x‖ * Real.cos θ) (hx1 : x 1 = ‖x‖ * Real.sin θ) :
    ‖x - s • circleChart φ‖ = shellDist ‖x‖ s (Real.cos (φ - θ)) := by
  rw [shellDist, ← Real.sqrt_sq (norm_nonneg (x - s • circleChart φ))]
  congr 1
  rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two]
  simp only [PiLp.sub_apply, PiLp.smul_apply, circleChart_apply_zero,
    circleChart_apply_one, smul_eq_mul]
  rw [Real.cos_sub, hx0, hx1]
  linear_combination ‖x‖ ^ 2 * Real.cos_sq_add_sin_sq θ +
    s ^ 2 * Real.cos_sq_add_sin_sq φ

/-! ## Master radiality -/

/-- The uniform-`φ` circle average of the Laplace kernel at an arbitrary probe
`x` is the axial shell profile evaluated at `‖x‖`. -/
lemma circleAverage_laplaceKernel_eq_shellZ₂ (τ : ℝ)
    (x : EuclideanSpace ℝ (Fin 2)) (s : ℝ) :
    ∫ φ, laplaceKernel τ x (chartMap₂ (s, φ)) ∂chartBase₂ = shellZ₂ τ ‖x‖ s := by
  rw [integral_chartBase₂, shellZ₂]
  congr 1
  by_cases hx : x = 0
  · subst hx
    rw [norm_zero]
    refine setIntegral_congr_fun measurableSet_Ioc (fun φ _ => ?_)
    simp only [chartMap₂_mk, laplaceKernel]
    congr 1
    rw [zero_sub, norm_neg, norm_smul, Real.norm_eq_abs, norm_circleChart, mul_one,
      shellDist, show (0 : ℝ) ^ 2 + s ^ 2 - 2 * 0 * s * Real.cos φ = s ^ 2 by ring,
      Real.sqrt_sq_eq_abs]
  · obtain ⟨θ, hx0, hx1⟩ := exists_polar₂ x hx
    set G : ℝ → ℝ := fun ψ => Real.exp (-(1 / τ) * shellDist ‖x‖ s (Real.cos ψ))
      with hG
    have hstep1 : (∫ φ in Ioc (-Real.pi) Real.pi,
          laplaceKernel τ x (chartMap₂ (s, φ)))
        = ∫ φ in Ioc (-Real.pi) Real.pi, G (φ - θ) := by
      refine setIntegral_congr_fun measurableSet_Ioc (fun φ _ => ?_)
      simp only [chartMap₂_mk, laplaceKernel, hG]
      rw [norm_sub_smul_circleChart s φ hx0 hx1]
    have hπle : (-Real.pi : ℝ) ≤ Real.pi := by linarith [Real.pi_pos]
    have hper : Function.Periodic G (2 * Real.pi) := by
      intro ψ
      simp only [hG]
      rw [Real.cos_add_two_pi]
    rw [hstep1, ← intervalIntegral.integral_of_le hπle,
      ← intervalIntegral.integral_of_le hπle,
      intervalIntegral.integral_comp_sub_right G θ]
    have hshift := hper.intervalIntegral_add_eq (-Real.pi - θ) (-Real.pi)
    rw [show -Real.pi - θ + 2 * Real.pi = Real.pi - θ by ring,
      show -Real.pi + 2 * Real.pi = Real.pi by ring] at hshift
    exact hshift

/-- The Laplace normalizer of a genuine `n = 2` radial mixture is radial, with
profile exactly `radialRayZ₂`. -/
theorem kernelNormalizer_radialMixture₂_radial (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (x : EuclideanSpace ℝ (Fin 2)) :
    kernelNormalizer (laplaceKernel τ) (radialMixture₂ ν) x =
      radialRayZ₂ τ ν ‖x‖ := by
  have hf : Integrable (fun y => laplaceKernel τ x y) (radialMixture₂ ν) :=
    laplaceKernel_integrable (radialMixture₂ ν) τ hτ x
  rw [kernelNormalizer, integral_radialMixture₂ ν hf, radialRayZ₂]
  apply integral_congr_ae
  exact Filter.Eventually.of_forall fun s =>
    circleAverage_laplaceKernel_eq_shellZ₂ τ x s

/-! ## From ray proportionality to measure identity -/

/-- Ray proportionality extended to the origin, with a positive constant. -/
lemma radialRayZ₂_proportional_nonneg (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable (fun s : ℝ => s ^ 1) νp)
    (hmq : Integrable (fun s : ℝ => s ^ 1) νq)
    (hslackp : RadialSlack₂ τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq)) :
    ∃ c : ℝ, 0 < c ∧ ∀ r : ℝ, 0 ≤ r →
      radialRayZ₂ τ νp r = c * radialRayZ₂ τ νq r := by
  obtain ⟨c, hprop⟩ := radialRayZ₂_proportional τ hτ νp νq hsp hsq
    hmp hmq hslackp hzero
  have hcpos : 0 < c := by
    have h1 := hprop 1 one_pos
    have hp1 := radialRayZ₂_pos τ hτ νp 1
    have hq1 := radialRayZ₂_pos τ hτ νq 1
    nlinarith
  refine ⟨c, hcpos, fun r hr => ?_⟩
  rcases eq_or_lt_of_le hr with h0 | hpos
  · subst r
    have hf : Tendsto (radialRayZ₂ τ νp) (nhdsWithin (0 : ℝ) (Ioi 0))
        (nhds (radialRayZ₂ τ νp 0)) :=
      ((continuous_radialRayZ₂ τ hτ νp).tendsto 0).mono_left nhdsWithin_le_nhds
    have hg : Tendsto (fun r => c * radialRayZ₂ τ νq r)
        (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds (c * radialRayZ₂ τ νq 0)) :=
      (((continuous_radialRayZ₂ τ hτ νq).const_mul c).tendsto 0).mono_left
        nhdsWithin_le_nhds
    have heq : radialRayZ₂ τ νp =ᶠ[nhdsWithin (0 : ℝ) (Ioi 0)]
        fun r => c * radialRayZ₂ τ νq r := by
      filter_upwards [self_mem_nhdsWithin] with r hr'
      exact hprop r hr'
    exact tendsto_nhds_unique_of_eventuallyEq hf hg heq
  · exact hprop r hpos

/-- **G3 headline at center zero.**  Zero Laplace drift identifies genuine
`n = 2` radial mixtures, under `RadialSlack₂` and finite first radial moments. -/
theorem laplaceZeroDrift_identifies_of_radialMixture₂ (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable (fun s : ℝ => s ^ 1) νp)
    (hmq : Integrable (fun s : ℝ => s ^ 1) νq)
    (hslackp : RadialSlack₂ τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq)) :
    radialMixture₂ νp = radialMixture₂ νq := by
  obtain ⟨c, hcpos, hprop⟩ := radialRayZ₂_proportional_nonneg τ hτ νp νq
    hsp hsq hmp hmq hslackp hzero
  have hZ : ∀ x : EuclideanSpace ℝ (Fin 2),
      kernelNormalizer (laplaceKernel τ) (radialMixture₂ νp) x =
        kernelNormalizer (laplaceKernel τ)
          ((ENNReal.ofReal c) • radialMixture₂ νq) x := by
    intro x
    have hsm : kernelNormalizer (laplaceKernel τ)
        ((ENNReal.ofReal c) • radialMixture₂ νq) x =
        c * kernelNormalizer (laplaceKernel τ) (radialMixture₂ νq) x := by
      have h1 := MeasureTheory.integral_smul_measure
        (μ := radialMixture₂ νq) (f := fun y => laplaceKernel τ x y)
        (c := ENNReal.ofReal c)
      rw [smul_eq_mul] at h1
      rw [kernelNormalizer, h1, ENNReal.toReal_ofReal hcpos.le]
      rfl
    rw [hsm, kernelNormalizer_radialMixture₂_radial τ hτ νp x,
      kernelNormalizer_radialMixture₂_radial τ hτ νq x]
    exact hprop ‖x‖ (norm_nonneg x)
  haveI hfin : IsFiniteMeasure ((ENNReal.ofReal c) • radialMixture₂ νq) := by
    constructor
    rw [Measure.smul_apply, smul_eq_mul, measure_univ, mul_one]
    exact ENNReal.ofReal_lt_top
  have hL4 := laplaceSmoothingInjective_euclideanSpace (ι := Fin 2) τ hτ
  have hpq := hL4 (radialMixture₂ νp) ((ENNReal.ofReal c) • radialMixture₂ νq)
    inferInstance hfin hZ
  have hmass : (radialMixture₂ νp) univ =
      ((ENNReal.ofReal c) • radialMixture₂ νq) univ := by rw [hpq]
  rw [measure_univ, Measure.smul_apply, smul_eq_mul, measure_univ, mul_one] at hmass
  have hc1 : c = 1 := ENNReal.ofReal_eq_one.mp hmass.symm
  rw [hpq, hc1, ENNReal.ofReal_one, one_smul]

end DriftingIdentifiability
