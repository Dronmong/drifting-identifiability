import DriftingIdentifiability.LaplaceRadialShell2
import DriftingIdentifiability.LaplaceRadialDifferentiationN

/-!
# Radial Laplace converse, milestone G3 (`n = 2`): the ray / differentiation layer

Second G3 file (`LaplaceRnRoadmap.md` §3(G3), first-order path).  The ray
objects `Z̃₂/D̃₂/C̃₂/Q̃₂/Z̃d₂` as `ν`-mixtures of the `Shell2` per-shell
`φ`-averages, their mixture-integral forms, and the **first-order**
`r`-derivatives `Z̃₂' = (1/τ)Z̃d₂`, `C̃₂' = (1/τ)D̃₂`, `D̃₂' = (1/τ)Q̃₂ − Z̃₂`.

The research pass (roadmap G3) established that the whole radial program is
first-order with *constant* dominators (`|X/d| ≤ 1`, `X²/d ≤ d`), so the
`n = 2` log-divergence (which lives in `Z̃''`) never enters and this layer is
a direct port of `DifferentiationN`.  We reuse the `0 < n` pointwise kernel
lemmas from `DifferentiationN` at `n = 2` through the probe bridge
`rayProbe₂ = radialRayProbeN 2`.
-/

open MeasureTheory Filter Topology Set

namespace DriftingIdentifiability

open Paper

/-! ## The ray probe as a scaled chart direction -/

/-- The ray probe is `r` times the chart's zero-angle unit vector. -/
lemma rayProbe₂_eq_smul (r : ℝ) : rayProbe₂ r = r • circleChart 0 := by
  ext i
  fin_cases i
  · change rayProbe₂ r 0 = (r • circleChart 0) 0
    rw [rayProbe₂_apply_zero, PiLp.smul_apply, circleChart_apply_zero,
      Real.cos_zero, smul_eq_mul, mul_one]
  · change rayProbe₂ r 1 = (r • circleChart 0) 1
    rw [rayProbe₂_apply_one, PiLp.smul_apply, circleChart_apply_one,
      Real.sin_zero, smul_eq_mul, mul_zero]

/-! ## Additional per-shell objects (`n = 2`) -/

/-- Per-shell average of the companion kernel `(τ+d)e^{-d/τ}` (`n = 2`). -/
noncomputable def shellC₂ (τ r s : ℝ) : ℝ :=
  (2 * Real.pi)⁻¹ * ∫ φ in Ioc (-Real.pi) Real.pi,
    (τ + shellDist r s (Real.cos φ)) *
      Real.exp (-(1 / τ) * shellDist r s (Real.cos φ))

/-- Per-shell average of the soft-sign payload `(X/d) e^{-d/τ}` (`n = 2`). -/
noncomputable def shellZd₂ (τ r s : ℝ) : ℝ :=
  (2 * Real.pi)⁻¹ * ∫ φ in Ioc (-Real.pi) Real.pi,
    Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
      ((s * Real.cos φ - r) / shellDist r s (Real.cos φ))

/-- Per-shell average of the `X²/d` payload `(X²/d) e^{-d/τ}` (`n = 2`). -/
noncomputable def shellQ₂ (τ r s : ℝ) : ℝ :=
  (2 * Real.pi)⁻¹ * ∫ φ in Ioc (-Real.pi) Real.pi,
    Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
      ((s * Real.cos φ - r) ^ 2 / shellDist r s (Real.cos φ))

/-! ## The ray objects -/

/-- Ray normalizer profile (`n = 2`). -/
noncomputable def radialRayZ₂ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellZ₂ τ r s ∂ν

/-- Ray axial drift-numerator profile (`n = 2`). -/
noncomputable def radialRayD₂ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellD₂ τ r s ∂ν

/-- Ray companion-normalizer profile (`n = 2`). -/
noncomputable def radialRayC₂ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellC₂ τ r s ∂ν

/-- Ray soft-sign profile (`n = 2`), the `Z̃₂'`-payload. -/
noncomputable def radialRayZd₂ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellZd₂ τ r s ∂ν

/-- Ray `X²/d` profile (`n = 2`), the `D̃₂'`-payload. -/
noncomputable def radialRayQ₂ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellQ₂ τ r s ∂ν

/-! ## The mixture-collapse bridges -/

/-- Generic collapse: every mixture integral is a `ν`-integral of the uniform
`φ`-average of the chart integrand. -/
lemma integral_radialMixture₂_eq (ν : Measure ℝ) [IsProbabilityMeasure ν]
    {g : EuclideanSpace ℝ (Fin 2) → ℝ} (hf : Integrable g (radialMixture₂ ν)) :
    ∫ y, g y ∂(radialMixture₂ ν)
      = ∫ s, ((2 * Real.pi)⁻¹ *
          ∫ φ in Ioc (-Real.pi) Real.pi, g (s • circleChart φ)) ∂ν := by
  rw [integral_radialMixture₂ ν hf]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))
  dsimp only
  rw [← integral_chartBase₂ (fun φ => g (s • circleChart φ))]
  simp only [chartMap₂_mk]

/-- The mixture normalizer is the ray normalizer. -/
lemma radialRayZ₂_eq_integral (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayZ₂ τ ν r
      = ∫ y, laplaceKernel τ (rayProbe₂ r) y ∂(radialMixture₂ ν) := by
  rw [radialRayZ₂, ← kernelNormalizer_radialMixture₂ τ hτ ν r, kernelNormalizer]

/-- The mixture axial integral is the ray drift numerator. -/
lemma radialRayD₂_eq_integral (τ : ℝ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ)
    (hf : Integrable (fun y => laplaceKernel τ (rayProbe₂ r) y *
      (y 0 - r)) (radialMixture₂ ν)) :
    radialRayD₂ τ ν r
      = ∫ y, laplaceKernel τ (rayProbe₂ r) y * (y 0 - r) ∂(radialMixture₂ ν) := by
  rw [radialRayD₂, integral_radialMixture₂_eq ν hf]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))
  rw [shellD₂]
  congr 1
  refine setIntegral_congr_fun measurableSet_Ioc (fun φ _ => ?_)
  have hc : (s • circleChart φ) 0 = s * Real.cos φ := by
    rw [PiLp.smul_apply, circleChart_apply_zero, smul_eq_mul]
  rw [laplaceKernel_rayProbe₂_chart, hc]

/-! ## Pointwise kernel derivative along the ray -/

/-- The probe distance along the ray is differentiable away from a collision. -/
lemma hasDerivAt_norm_rayProbe₂_sub {r : ℝ} {y : EuclideanSpace ℝ (Fin 2)}
    (hne : ‖rayProbe₂ r - y‖ ≠ 0) :
    HasDerivAt (fun x => ‖rayProbe₂ x - y‖)
      ((r - y 0) / ‖rayProbe₂ r - y‖) r := by
  have hline : HasDerivAt (fun x => rayProbe₂ x - y) (circleChart 0) r := by
    have h : HasDerivAt (fun x : ℝ => x • circleChart 0 - y) (circleChart 0) r := by
      simpa using ((hasDerivAt_id r).smul_const (circleChart 0)).sub_const y
    refine h.congr_of_eventuallyEq ?_
    filter_upwards with x
    rw [rayProbe₂_eq_smul]
  have hsqne : ‖rayProbe₂ r - y‖ ^ 2 ≠ 0 := pow_ne_zero 2 hne
  have hsqrt := (hline.norm_sq).sqrt hsqne
  have hfun : (fun x => Real.sqrt (‖rayProbe₂ x - y‖ ^ 2))
      = fun x => ‖rayProbe₂ x - y‖ := by
    funext x; rw [Real.sqrt_sq (norm_nonneg _)]
  rw [hfun] at hsqrt
  have hinner : inner ℝ (rayProbe₂ r - y) (circleChart 0) = r - y 0 := by
    rw [PiLp.inner_apply, Fin.sum_univ_two]
    simp only [RCLike.inner_apply, conj_trivial, PiLp.sub_apply,
      rayProbe₂_apply_zero, rayProbe₂_apply_one, circleChart_apply_zero,
      circleChart_apply_one, Real.cos_zero, Real.sin_zero]
    ring
  have hval : (2 * inner ℝ (rayProbe₂ r - y) (circleChart 0)) /
        (2 * Real.sqrt (‖rayProbe₂ r - y‖ ^ 2))
      = (r - y 0) / ‖rayProbe₂ r - y‖ := by
    rw [hinner, Real.sqrt_sq (norm_nonneg _)]
    exact mul_div_mul_left _ _ two_ne_zero
  rw [hval] at hsqrt
  exact hsqrt

lemma hasDerivAt_laplaceKernel_rayProbe₂ {τ r : ℝ}
    {y : EuclideanSpace ℝ (Fin 2)} (hne : ‖rayProbe₂ r - y‖ ≠ 0) :
    HasDerivAt (fun x => laplaceKernel τ (rayProbe₂ x) y)
      ((1 / τ) * (laplaceKernel τ (rayProbe₂ r) y *
        ((y 0 - r) / ‖rayProbe₂ r - y‖))) r := by
  have hd := hasDerivAt_norm_rayProbe₂_sub hne
  have hexp := (hd.const_mul (-(1 / τ))).exp
  have hfe : (fun x => Real.exp (-(1 / τ) * ‖rayProbe₂ x - y‖))
      = fun x => laplaceKernel τ (rayProbe₂ x) y := by
    funext x; rw [laplaceKernel]
  rw [hfe] at hexp
  have hval : Real.exp (-(1 / τ) * ‖rayProbe₂ r - y‖) *
        (-(1 / τ) * ((r - y 0) / ‖rayProbe₂ r - y‖))
      = (1 / τ) * (laplaceKernel τ (rayProbe₂ r) y *
        ((y 0 - r) / ‖rayProbe₂ r - y‖)) := by
    rw [laplaceKernel, div_eq_mul_inv, div_eq_mul_inv]
    ring
  rw [hval] at hexp
  exact hexp

/-! ## Coordinate bounds -/

lemma abs_first_sub_le_norm_rayProbe₂ (r : ℝ) (y : EuclideanSpace ℝ (Fin 2)) :
    |y 0 - r| ≤ ‖rayProbe₂ r - y‖ := by
  have h := abs_coord_le_norm_N (rayProbe₂ r - y) 0
  rwa [PiLp.sub_apply, rayProbe₂_apply_zero, abs_sub_comm] at h

lemma abs_first_div_rayProbe₂_le_one (r : ℝ) (y : EuclideanSpace ℝ (Fin 2)) :
    |(y 0 - r) / ‖rayProbe₂ r - y‖| ≤ 1 := by
  rcases eq_or_ne (‖rayProbe₂ r - y‖) 0 with h0 | hne
  · rw [h0, div_zero, abs_zero]; norm_num
  · have hpos : 0 < ‖rayProbe₂ r - y‖ := (norm_nonneg _).lt_of_ne' hne
    rw [abs_div, abs_of_pos hpos, div_le_one hpos]
    exact abs_first_sub_le_norm_rayProbe₂ r y

lemma integrable_laplaceKernel_rayProbe₂ (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin 2))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceKernel τ (rayProbe₂ r) y) μ :=
  ⟨(continuous_laplaceKernel_rayProbe₂ τ r).aestronglyMeasurable,
    HasFiniteIntegral.of_bounded (C := 1)
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs, abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ r y)]
        exact laplaceKernel_rayProbe₂_le_one τ hτ r y)⟩

/-! ## The soft-sign collapse -/

lemma radialRayZd₂_eq_integral (τ : ℝ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ)
    (hf : Integrable (fun y => laplaceKernel τ (rayProbe₂ r) y *
      ((y 0 - r) / ‖rayProbe₂ r - y‖)) (radialMixture₂ ν)) :
    radialRayZd₂ τ ν r
      = ∫ y, laplaceKernel τ (rayProbe₂ r) y *
          ((y 0 - r) / ‖rayProbe₂ r - y‖) ∂(radialMixture₂ ν) := by
  rw [radialRayZd₂, integral_radialMixture₂_eq ν hf]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))
  rw [shellZd₂]
  congr 1
  refine setIntegral_congr_fun measurableSet_Ioc (fun φ _ => ?_)
  have hc : (s • circleChart φ) 0 = s * Real.cos φ := by
    rw [PiLp.smul_apply, circleChart_apply_zero, smul_eq_mul]
  have hnorm : ‖rayProbe₂ r - s • circleChart φ‖ = shellDist r s (Real.cos φ) :=
    norm_rayProbe₂_sub_smul_circleChart r s φ
  rw [laplaceKernel_rayProbe₂_chart, hc, hnorm]

lemma integrable_softsign_rayProbe₂ (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin 2))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceKernel τ (rayProbe₂ r) y *
      ((y 0 - r) / ‖rayProbe₂ r - y‖)) μ := by
  refine ⟨?_, HasFiniteIntegral.of_bounded (C := 1) (ae_of_all _ fun y => ?_)⟩
  · have hcoord : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => y 0 - r) := by fun_prop
    have hnorm : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => ‖rayProbe₂ r - y‖) := by fun_prop
    exact (((continuous_laplaceKernel_rayProbe₂ τ r).measurable).mul
      (hcoord.div hnorm)).aestronglyMeasurable
  · rw [Real.norm_eq_abs, abs_mul,
      abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ r y)]
    calc laplaceKernel τ (rayProbe₂ r) y * |(y 0 - r) / ‖rayProbe₂ r - y‖|
        ≤ 1 * 1 :=
          mul_le_mul (laplaceKernel_rayProbe₂_le_one τ hτ r y)
            (abs_first_div_rayProbe₂_le_one r y) (abs_nonneg _) zero_le_one
      _ = 1 := mul_one _

/-! ## The moving probe is `radialMixture₂`-a.e. off the sample -/

lemma radialMixture₂_ae_probe_ne (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    ∀ᵐ y ∂(radialMixture₂ ν), ∀ x : ℝ, 0 < x → ‖rayProbe₂ x - y‖ ≠ 0 := by
  set A : Set (EuclideanSpace ℝ (Fin 2)) := {y | y 1 = 0 ∧ 0 < y 0} with hA
  have hAmeas : MeasurableSet A := by
    refine MeasurableSet.inter ?_ ?_
    · exact (by fun_prop : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => y 1)) (measurableSet_singleton 0)
    · exact (by fun_prop : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => y 0)) measurableSet_Ioi
  have hpre : chartMap₂ ⁻¹' A ⊆ (univ : Set ℝ) ×ˢ {φ | Real.sin φ = 0} := by
    rintro ⟨s, φ⟩ hz
    rw [mem_preimage, hA] at hz
    simp only [mem_setOf_eq, chartMap₂_mk] at hz
    obtain ⟨h1, h0⟩ := hz
    have hval1 : (s • circleChart φ) 1 = s * Real.sin φ := by
      rw [PiLp.smul_apply, circleChart_apply_one, smul_eq_mul]
    have hval0 : (s • circleChart φ) 0 = s * Real.cos φ := by
      rw [PiLp.smul_apply, circleChart_apply_zero, smul_eq_mul]
    rw [hval1] at h1
    rw [hval0] at h0
    refine ⟨mem_univ _, ?_⟩
    simp only [mem_setOf_eq]
    rcases mul_eq_zero.mp h1 with hs | hsin
    · exfalso; rw [hs, zero_mul] at h0; exact lt_irrefl 0 h0
    · exact hsin
  have hsin_null : chartBase₂ {φ | Real.sin φ = 0} = 0 := by
    have hcount : {φ : ℝ | Real.sin φ = 0}.Countable := by
      refine Set.Countable.mono ?_ (Set.countable_range (fun n : ℤ => (n : ℝ) * Real.pi))
      intro φ hφ
      rw [mem_setOf_eq, Real.sin_eq_zero_iff] at hφ
      obtain ⟨n, hn⟩ := hφ
      exact ⟨n, hn⟩
    have hvol : volume {φ : ℝ | Real.sin φ = 0} = 0 := hcount.measure_zero volume
    have hsinmeas : MeasurableSet {φ : ℝ | Real.sin φ = 0} :=
      Real.continuous_sin.measurable (measurableSet_singleton 0)
    rw [chartBase₂, Measure.smul_apply, smul_eq_mul]
    have hres : (volume.restrict (Ioc (-Real.pi) Real.pi)) {φ | Real.sin φ = 0}
        = 0 := by
      rw [Measure.restrict_apply hsinmeas]
      exact measure_mono_null Set.inter_subset_left hvol
    rw [hres, mul_zero]
  have hslab_null : (ν.prod chartBase₂)
      ((univ : Set ℝ) ×ˢ {φ | Real.sin φ = 0}) = 0 := by
    rw [Measure.prod_prod, hsin_null, mul_zero]
  have hAnull : radialMixture₂ ν A = 0 := by
    rw [radialMixture₂, Measure.map_apply continuous_chartMap₂.measurable hAmeas]
    exact measure_mono_null hpre hslab_null
  rw [ae_iff]
  refine measure_mono_null ?_ hAnull
  intro y hy
  rcases not_forall.mp hy with ⟨x, hx⟩
  rw [Classical.not_imp, not_not] at hx
  obtain ⟨hxpos, hnorm0⟩ := hx
  have heq : rayProbe₂ x = y := by rwa [norm_sub_eq_zero_iff] at hnorm0
  rw [hA, mem_setOf_eq]
  refine ⟨?_, ?_⟩
  · rw [← heq, rayProbe₂_apply_one]
  · rw [← heq, rayProbe₂_apply_zero]; exact hxpos

/-! ## The first derivative `Z̃₂' = (1/τ) Z̃d₂` -/

theorem hasDerivAt_radialRayZ₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayZ₂ τ ν) ((1 / τ) * radialRayZd₂ τ ν r) r := by
  have hfe : radialRayZ₂ τ ν = fun x =>
      ∫ y, laplaceKernel τ (rayProbe₂ x) y ∂(radialMixture₂ ν) := by
    funext x
    exact radialRayZ₂_eq_integral τ hτ ν x
  have hmeasF' : AEStronglyMeasurable
      (fun y : EuclideanSpace ℝ (Fin 2) =>
        (1 / τ) * (laplaceKernel τ (rayProbe₂ r) y *
          ((y 0 - r) / ‖rayProbe₂ r - y‖))) (radialMixture₂ ν) := by
    have hcoord : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => y 0 - r) := by fun_prop
    have hnorm : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => ‖rayProbe₂ r - y‖) := by fun_prop
    exact ((((continuous_laplaceKernel_rayProbe₂ τ r).measurable).mul
      (hcoord.div hnorm)).const_mul _).aestronglyMeasurable
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := radialMixture₂ ν) (x₀ := r)
    (F := fun x (y : EuclideanSpace ℝ (Fin 2)) => laplaceKernel τ (rayProbe₂ x) y)
    (F' := fun x (y : EuclideanSpace ℝ (Fin 2)) =>
      (1 / τ) * (laplaceKernel τ (rayProbe₂ x) y *
        ((y 0 - x) / ‖rayProbe₂ x - y‖)))
    (bound := fun _ => 1 / τ)
    (Ioi_mem_nhds hr)
    (Filter.Eventually.of_forall fun x =>
      (continuous_laplaceKernel_rayProbe₂ τ x).aestronglyMeasurable)
    (integrable_laplaceKernel_rayProbe₂ τ hτ _ r)
    hmeasF'
    (ae_of_all _ fun y => by
      intro x _
      rw [Real.norm_eq_abs, abs_mul, abs_mul,
        abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ x y),
        abs_of_nonneg (by positivity : (0 : ℝ) ≤ 1 / τ)]
      calc 1 / τ * (laplaceKernel τ (rayProbe₂ x) y *
            |(y 0 - x) / ‖rayProbe₂ x - y‖|)
          ≤ 1 / τ * (1 * 1) := by
            apply mul_le_mul_of_nonneg_left _ (by positivity)
            exact mul_le_mul (laplaceKernel_rayProbe₂_le_one τ hτ x y)
              (abs_first_div_rayProbe₂_le_one x y) (abs_nonneg _) zero_le_one
        _ = 1 / τ := by ring)
    (integrable_const _)
    (by
      filter_upwards [radialMixture₂_ae_probe_ne ν] with y hy
      intro x hx
      exact hasDerivAt_laplaceKernel_rayProbe₂ (hy x hx))
  rw [hfe]
  have hval : (∫ y, (1 / τ) *
      (laplaceKernel τ (rayProbe₂ r) y *
        ((y 0 - r) / ‖rayProbe₂ r - y‖)) ∂(radialMixture₂ ν))
      = (1 / τ) * radialRayZd₂ τ ν r := by
    rw [radialRayZd₂_eq_integral τ ν r
      (integrable_softsign_rayProbe₂ τ hτ _ r), integral_const_mul]
  exact hval ▸ key.2

/-! ## The `X²/d` bound and payload derivatives -/

lemma first_sq_div_rayProbe₂_le (r : ℝ) (y : EuclideanSpace ℝ (Fin 2)) :
    (y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖ ≤ ‖rayProbe₂ r - y‖ := by
  rcases eq_or_ne (‖rayProbe₂ r - y‖) 0 with h0 | hne
  · rw [h0, div_zero]
  · have hpos : 0 < ‖rayProbe₂ r - y‖ := (norm_nonneg _).lt_of_ne' hne
    rw [div_le_iff₀ hpos]
    have h := abs_first_sub_le_norm_rayProbe₂ r y
    nlinarith [mul_self_le_mul_self (abs_nonneg (y 0 - r)) h, sq_abs (y 0 - r)]

/-- The core `K·d ≤ τe⁻¹` bound along the ray. -/
lemma laplaceKernel_rayProbe₂_mul_norm_le (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin 2)) :
    laplaceKernel τ (rayProbe₂ r) y * ‖rayProbe₂ r - y‖ ≤ τ * Real.exp (-1) := by
  rw [laplaceKernel, mul_comm]
  have he : -(1 / τ) * ‖rayProbe₂ r - y‖ = -‖rayProbe₂ r - y‖ / τ := by ring
  rw [he]
  exact mul_exp_neg_div_le hτ (norm_nonneg _)

lemma abs_kernel_mul_first_sq_div_rayProbe₂_le (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin 2)) :
    |laplaceKernel τ (rayProbe₂ r) y *
      ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖)| ≤ τ * Real.exp (-1) := by
  have hKnn : 0 ≤ laplaceKernel τ (rayProbe₂ r) y :=
    laplaceKernel_rayProbe₂_nonneg τ r y
  have hpnn : 0 ≤ (y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖ :=
    div_nonneg (sq_nonneg _) (norm_nonneg _)
  rw [abs_of_nonneg (mul_nonneg hKnn hpnn)]
  have hle : laplaceKernel τ (rayProbe₂ r) y *
        ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖)
      ≤ laplaceKernel τ (rayProbe₂ r) y * ‖rayProbe₂ r - y‖ :=
    mul_le_mul_of_nonneg_left (first_sq_div_rayProbe₂_le r y) hKnn
  exact hle.trans (laplaceKernel_rayProbe₂_mul_norm_le τ hτ r y)

/-- Pointwise derivative of the axial weighted-displacement integrand. -/
lemma hasDerivAt_laplaceKernel_mul_first_rayProbe₂ {τ r : ℝ}
    {y : EuclideanSpace ℝ (Fin 2)} (hne : ‖rayProbe₂ r - y‖ ≠ 0) :
    HasDerivAt (fun x => laplaceKernel τ (rayProbe₂ x) y * (y 0 - x))
      ((1 / τ) * (laplaceKernel τ (rayProbe₂ r) y *
          ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖)) -
        laplaceKernel τ (rayProbe₂ r) y) r := by
  have hk := hasDerivAt_laplaceKernel_rayProbe₂ (τ := τ) hne
  have hlin : HasDerivAt (fun x : ℝ => y 0 - x) (-1) r := by
    simpa using (hasDerivAt_id r).const_sub (y 0)
  have hprod := hk.mul hlin
  have hval :
      (1 / τ) * (laplaceKernel τ (rayProbe₂ r) y *
          ((y 0 - r) / ‖rayProbe₂ r - y‖)) * (y 0 - r) +
        laplaceKernel τ (rayProbe₂ r) y * (-1)
      = (1 / τ) * (laplaceKernel τ (rayProbe₂ r) y *
          ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖)) -
        laplaceKernel τ (rayProbe₂ r) y := by
    rw [div_eq_mul_inv, div_eq_mul_inv]
    ring
  rw [hval] at hprod
  exact hprod

/-- Pointwise derivative of the companion kernel along the ray. -/
lemma hasDerivAt_laplaceCompanionKernel_rayProbe₂ {τ r : ℝ}
    {y : EuclideanSpace ℝ (Fin 2)} (hτ : 0 < τ)
    (hne : ‖rayProbe₂ r - y‖ ≠ 0) :
    HasDerivAt (fun x => laplaceCompanionKernel τ (rayProbe₂ x) y)
      ((1 / τ) * (laplaceKernel τ (rayProbe₂ r) y * (y 0 - r))) r := by
  have hd := hasDerivAt_norm_rayProbe₂_sub hne
  have hk := hasDerivAt_laplaceKernel_rayProbe₂ (τ := τ) hne
  have hprod := (hd.const_add τ).mul hk
  have hval :
      (r - y 0) / ‖rayProbe₂ r - y‖ * laplaceKernel τ (rayProbe₂ r) y +
        (τ + ‖rayProbe₂ r - y‖) *
          ((1 / τ) * (laplaceKernel τ (rayProbe₂ r) y *
            ((y 0 - r) / ‖rayProbe₂ r - y‖)))
      = (1 / τ) * (laplaceKernel τ (rayProbe₂ r) y * (y 0 - r)) := by
    field_simp [hne, hτ.ne']
    ring
  rw [hval] at hprod
  simp only [laplaceCompanionKernel]
  exact hprod

/-! ## `Q` and companion collapses -/

lemma radialRayQ₂_eq_integral (τ : ℝ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ)
    (hf : Integrable (fun y => laplaceKernel τ (rayProbe₂ r) y *
      ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖)) (radialMixture₂ ν)) :
    radialRayQ₂ τ ν r
      = ∫ y, laplaceKernel τ (rayProbe₂ r) y *
          ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖) ∂(radialMixture₂ ν) := by
  rw [radialRayQ₂, integral_radialMixture₂_eq ν hf]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))
  rw [shellQ₂]
  congr 1
  refine setIntegral_congr_fun measurableSet_Ioc (fun φ _ => ?_)
  have hc : (s • circleChart φ) 0 = s * Real.cos φ := by
    rw [PiLp.smul_apply, circleChart_apply_zero, smul_eq_mul]
  have hnorm : ‖rayProbe₂ r - s • circleChart φ‖ = shellDist r s (Real.cos φ) :=
    norm_rayProbe₂_sub_smul_circleChart r s φ
  rw [laplaceKernel_rayProbe₂_chart, hc, hnorm]

lemma radialRayC₂_eq_integral (τ : ℝ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ)
    (hf : Integrable (fun y => laplaceCompanionKernel τ (rayProbe₂ r) y)
      (radialMixture₂ ν)) :
    radialRayC₂ τ ν r
      = ∫ y, laplaceCompanionKernel τ (rayProbe₂ r) y ∂(radialMixture₂ ν) := by
  rw [radialRayC₂, integral_radialMixture₂_eq ν hf]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))
  rw [shellC₂]
  congr 1
  refine setIntegral_congr_fun measurableSet_Ioc (fun φ _ => ?_)
  rw [laplaceCompanionKernel, norm_rayProbe₂_sub_smul_circleChart,
    laplaceKernel_rayProbe₂_chart]

/-! ## Integrability of the axial and companion payloads -/

lemma integrable_axial_rayProbe₂ (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin 2))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceKernel τ (rayProbe₂ r) y * (y 0 - r)) μ := by
  refine ⟨?_, HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
    (ae_of_all _ fun y => ?_)⟩
  · have hcoord : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => y 0 - r) := by fun_prop
    exact (((continuous_laplaceKernel_rayProbe₂ τ r).measurable).mul
      hcoord).aestronglyMeasurable
  · rw [Real.norm_eq_abs, abs_mul,
      abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ r y)]
    have hXd : |y 0 - r| ≤ ‖rayProbe₂ r - y‖ := abs_first_sub_le_norm_rayProbe₂ r y
    calc laplaceKernel τ (rayProbe₂ r) y * |y 0 - r|
        ≤ laplaceKernel τ (rayProbe₂ r) y * ‖rayProbe₂ r - y‖ :=
          mul_le_mul_of_nonneg_left hXd (laplaceKernel_rayProbe₂_nonneg τ r y)
      _ ≤ τ * Real.exp (-1) := laplaceKernel_rayProbe₂_mul_norm_le τ hτ r y

lemma integrable_companion_rayProbe₂ (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin 2))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceCompanionKernel τ (rayProbe₂ r) y) μ := by
  have hcont : Continuous (fun y : EuclideanSpace ℝ (Fin 2) =>
      laplaceCompanionKernel τ (rayProbe₂ r) y) := by
    simp only [laplaceCompanionKernel]
    exact ((continuous_const.add
      (continuous_const.sub continuous_id).norm).mul
      (continuous_laplaceKernel_rayProbe₂ τ r))
  refine ⟨hcont.aestronglyMeasurable,
    HasFiniteIntegral.of_bounded (C := τ + τ * Real.exp (-1))
      (ae_of_all _ fun y => ?_)⟩
  rw [Real.norm_eq_abs, laplaceCompanionKernel]
  have hKnn : 0 ≤ laplaceKernel τ (rayProbe₂ r) y :=
    laplaceKernel_rayProbe₂_nonneg τ r y
  have hnn : 0 ≤ (τ + ‖rayProbe₂ r - y‖) * laplaceKernel τ (rayProbe₂ r) y :=
    mul_nonneg (by positivity) hKnn
  rw [abs_of_nonneg hnn, add_mul]
  have h1 : τ * laplaceKernel τ (rayProbe₂ r) y ≤ τ := by
    nlinarith [laplaceKernel_rayProbe₂_le_one τ hτ r y, hτ.le]
  have h2 : ‖rayProbe₂ r - y‖ * laplaceKernel τ (rayProbe₂ r) y
      ≤ τ * Real.exp (-1) := by
    rw [mul_comm]
    exact laplaceKernel_rayProbe₂_mul_norm_le τ hτ r y
  linarith

lemma integrable_Q_payload_rayProbe₂ (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin 2))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceKernel τ (rayProbe₂ r) y *
      ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖)) μ := by
  refine ⟨?_, HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
    (ae_of_all _ fun y => ?_)⟩
  · have hcoord : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => (y 0 - r) ^ 2) := by fun_prop
    have hnorm : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => ‖rayProbe₂ r - y‖) := by fun_prop
    exact (((continuous_laplaceKernel_rayProbe₂ τ r).measurable).mul
      (hcoord.div hnorm)).aestronglyMeasurable
  · rw [Real.norm_eq_abs]
    exact abs_kernel_mul_first_sq_div_rayProbe₂_le τ hτ r y

/-! ## The first derivatives `D̃₂'` and `C̃₂'` -/

theorem hasDerivAt_radialRayD₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayD₂ τ ν)
      ((1 / τ) * radialRayQ₂ τ ν r - radialRayZ₂ τ ν r) r := by
  have hfe : radialRayD₂ τ ν = fun x =>
      ∫ y, laplaceKernel τ (rayProbe₂ x) y * (y 0 - x) ∂(radialMixture₂ ν) := by
    funext x
    exact radialRayD₂_eq_integral τ ν x (integrable_axial_rayProbe₂ τ hτ _ x)
  have hmeasF' : AEStronglyMeasurable
      (fun y : EuclideanSpace ℝ (Fin 2) =>
        (1 / τ) * (laplaceKernel τ (rayProbe₂ r) y *
          ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖)) -
        laplaceKernel τ (rayProbe₂ r) y) (radialMixture₂ ν) := by
    have hcoord : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => (y 0 - r) ^ 2) := by fun_prop
    have hnorm : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => ‖rayProbe₂ r - y‖) := by fun_prop
    exact ((((continuous_laplaceKernel_rayProbe₂ τ r).measurable).mul
      (hcoord.div hnorm)).const_mul _).sub
      (continuous_laplaceKernel_rayProbe₂ τ r).measurable
      |>.aestronglyMeasurable
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := radialMixture₂ ν) (x₀ := r)
    (F := fun x (y : EuclideanSpace ℝ (Fin 2)) =>
      laplaceKernel τ (rayProbe₂ x) y * (y 0 - x))
    (F' := fun x (y : EuclideanSpace ℝ (Fin 2)) =>
      (1 / τ) * (laplaceKernel τ (rayProbe₂ x) y *
        ((y 0 - x) ^ 2 / ‖rayProbe₂ x - y‖)) -
      laplaceKernel τ (rayProbe₂ x) y)
    (bound := fun _ => Real.exp (-1) + 1)
    (Ioi_mem_nhds hr)
    (Filter.Eventually.of_forall fun x =>
      ((continuous_laplaceKernel_rayProbe₂ τ x).mul
        (by fun_prop)).aestronglyMeasurable)
    (integrable_axial_rayProbe₂ τ hτ _ r)
    hmeasF'
    (ae_of_all _ fun y => by
      intro x _
      rw [Real.norm_eq_abs]
      have hA : |(1 / τ) * (laplaceKernel τ (rayProbe₂ x) y *
            ((y 0 - x) ^ 2 / ‖rayProbe₂ x - y‖))| ≤ Real.exp (-1) := by
        rw [abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ 1 / τ)]
        have hb := abs_kernel_mul_first_sq_div_rayProbe₂_le τ hτ x y
        calc 1 / τ * |laplaceKernel τ (rayProbe₂ x) y *
              ((y 0 - x) ^ 2 / ‖rayProbe₂ x - y‖)|
            ≤ 1 / τ * (τ * Real.exp (-1)) :=
              mul_le_mul_of_nonneg_left hb (by positivity)
          _ = Real.exp (-1) := by field_simp
      have hB : |laplaceKernel τ (rayProbe₂ x) y| ≤ 1 := by
        rw [abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ x y)]
        exact laplaceKernel_rayProbe₂_le_one τ hτ x y
      calc |(1 / τ) * (laplaceKernel τ (rayProbe₂ x) y *
              ((y 0 - x) ^ 2 / ‖rayProbe₂ x - y‖)) -
            laplaceKernel τ (rayProbe₂ x) y|
          ≤ |(1 / τ) * (laplaceKernel τ (rayProbe₂ x) y *
              ((y 0 - x) ^ 2 / ‖rayProbe₂ x - y‖))| +
            |laplaceKernel τ (rayProbe₂ x) y| := abs_sub _ _
        _ ≤ Real.exp (-1) + 1 := add_le_add hA hB)
    (integrable_const _)
    (by
      filter_upwards [radialMixture₂_ae_probe_ne ν] with y hy
      intro x hx
      exact hasDerivAt_laplaceKernel_mul_first_rayProbe₂ (hy x hx))
  rw [hfe]
  have hiQ : Integrable (fun y => laplaceKernel τ (rayProbe₂ r) y *
      ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖)) (radialMixture₂ ν) :=
    integrable_Q_payload_rayProbe₂ τ hτ _ r
  have hval : (∫ y, ((1 / τ) * (laplaceKernel τ (rayProbe₂ r) y *
        ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖)) -
      laplaceKernel τ (rayProbe₂ r) y) ∂(radialMixture₂ ν))
      = (1 / τ) * radialRayQ₂ τ ν r - radialRayZ₂ τ ν r := by
    rw [integral_sub (hiQ.const_mul (1 / τ))
      (integrable_laplaceKernel_rayProbe₂ τ hτ _ r), integral_const_mul,
      ← radialRayQ₂_eq_integral τ ν r hiQ, ← radialRayZ₂_eq_integral τ hτ ν r]
  exact hval ▸ key.2

/-- The companion-normalizer derivative `C̃₂' = (1/τ) D̃₂`. -/
theorem hasDerivAt_radialRayC₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayC₂ τ ν) ((1 / τ) * radialRayD₂ τ ν r) r := by
  have hfe : radialRayC₂ τ ν = fun x =>
      ∫ y, laplaceCompanionKernel τ (rayProbe₂ x) y ∂(radialMixture₂ ν) := by
    funext x
    exact radialRayC₂_eq_integral τ ν x (integrable_companion_rayProbe₂ τ hτ _ x)
  have hmeasF' : AEStronglyMeasurable
      (fun y : EuclideanSpace ℝ (Fin 2) =>
        (1 / τ) * (laplaceKernel τ (rayProbe₂ r) y * (y 0 - r)))
      (radialMixture₂ ν) := by
    have hcoord : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => y 0 - r) := by fun_prop
    exact ((((continuous_laplaceKernel_rayProbe₂ τ r).measurable).mul
      hcoord).const_mul _).aestronglyMeasurable
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := radialMixture₂ ν) (x₀ := r)
    (F := fun x (y : EuclideanSpace ℝ (Fin 2)) =>
      laplaceCompanionKernel τ (rayProbe₂ x) y)
    (F' := fun x (y : EuclideanSpace ℝ (Fin 2)) =>
      (1 / τ) * (laplaceKernel τ (rayProbe₂ x) y * (y 0 - x)))
    (bound := fun _ => Real.exp (-1))
    (Ioi_mem_nhds hr)
    (Filter.Eventually.of_forall fun x => by
      have hc : Continuous (fun y : EuclideanSpace ℝ (Fin 2) =>
          laplaceCompanionKernel τ (rayProbe₂ x) y) := by
        simp only [laplaceCompanionKernel]
        exact (continuous_const.add
          (continuous_const.sub continuous_id).norm).mul
          (continuous_laplaceKernel_rayProbe₂ τ x)
      exact hc.aestronglyMeasurable)
    (integrable_companion_rayProbe₂ τ hτ _ r)
    hmeasF'
    (ae_of_all _ fun y => by
      intro x _
      rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg
        (by positivity : (0 : ℝ) ≤ 1 / τ)]
      have hb := abs_kernel_mul_first_sq_div_rayProbe₂_le τ hτ x y
      have hXd : |laplaceKernel τ (rayProbe₂ x) y * (y 0 - x)|
          ≤ τ * Real.exp (-1) := by
        rw [abs_mul, abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ x y)]
        have hXle : |y 0 - x| ≤ ‖rayProbe₂ x - y‖ :=
          abs_first_sub_le_norm_rayProbe₂ x y
        calc laplaceKernel τ (rayProbe₂ x) y * |y 0 - x|
            ≤ laplaceKernel τ (rayProbe₂ x) y * ‖rayProbe₂ x - y‖ :=
              mul_le_mul_of_nonneg_left hXle
                (laplaceKernel_rayProbe₂_nonneg τ x y)
          _ ≤ τ * Real.exp (-1) :=
              laplaceKernel_rayProbe₂_mul_norm_le τ hτ x y
      calc 1 / τ * |laplaceKernel τ (rayProbe₂ x) y * (y 0 - x)|
          ≤ 1 / τ * (τ * Real.exp (-1)) :=
            mul_le_mul_of_nonneg_left hXd (by positivity)
        _ = Real.exp (-1) := by field_simp)
    (integrable_const _)
    (by
      filter_upwards [radialMixture₂_ae_probe_ne ν] with y hy
      intro x hx
      exact hasDerivAt_laplaceCompanionKernel_rayProbe₂ hτ (hy x hx))
  rw [hfe]
  have hval : (∫ y, (1 / τ) *
      (laplaceKernel τ (rayProbe₂ r) y * (y 0 - r)) ∂(radialMixture₂ ν))
      = (1 / τ) * radialRayD₂ τ ν r := by
    rw [radialRayD₂_eq_integral τ ν r (integrable_axial_rayProbe₂ τ hτ _ r),
      integral_const_mul]
  exact hval ▸ key.2

end DriftingIdentifiability
