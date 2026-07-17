import DriftingIdentifiability.LaplaceRadialShellN

/-!
# Radial Laplace converse, milestone G3 (`n = 2`): the shell layer

The `n = 2` radial program (`LaplaceRnRoadmap.md` §3(G3)).  The zonal
"`u = cos θ`" parametrization used for `n ≥ 3` would give the *singular*
arcsine weight `(1-u²)^{-1/2}` here; instead we parametrize the circle
`S¹ ⊂ ℝ²` directly by its **angle** `φ`, whose Haar measure is *uniform* —
so every per-shell object is a clean bounded `φ`-integral with no singular
weight, exactly mirroring the explicit-chart approach of the `n = 3` file
`LaplaceRadialShell3`.

Content:

* `circleChart φ = (cos φ, sin φ)`, the ray probe `r•e₁`, and the probe-shell
  distance identity `‖r•e₁ - s•circleChart φ‖ = shellDist r s (cos φ)`
  (reusing `shellDist` from the `n ≥ 3` layer at `u = cos φ`).
* `radialMixture₂ ν` — the rotation-invariant probability measure on `ℝ²`
  with radial profile `ν`, as a single pushforward of `ν ⊗ chartBase₂` with
  `chartBase₂` uniform on `[-π,π]`.  No sphere-measure API, no axiom.
* the per-shell objects `shellZ₂`, `shellD₂`, `shellT₂`, `shellQ₂`,
  `shellRhoSqOverDist₂` as uniform `φ`-averages, and the normalizer bridge
  `Z̃₂(r) = ∫ shellZ₂ dν`.
* **the `n = 2` tangential identity (T)** `shellRhoSqOverDist₂ = (τ/r)·shellT₂`
  — one `φ`-integration by parts against the primitive `-(τ/r)·s·sin φ`
  (boundary terms vanish at `φ = ±π` because `sin(±π) = 0`), valid off the
  collision shell `s = r` and extended across it by `s`-continuity.
-/

open MeasureTheory Filter Topology Set

namespace DriftingIdentifiability

open Paper

/-! ## The circle chart and the ray probe -/

/-- The explicit chart of the unit circle `S¹ ⊂ ℝ²`: `φ ↦ (cos φ, sin φ)`. -/
noncomputable def circleChart (φ : ℝ) : EuclideanSpace ℝ (Fin 2) :=
  !₂[Real.cos φ, Real.sin φ]

/-- The ray probe `r • e₁ = (r, 0)` in `ℝ²`. -/
noncomputable def rayProbe₂ (r : ℝ) : EuclideanSpace ℝ (Fin 2) := !₂[r, 0]

@[simp] lemma circleChart_apply_zero (φ : ℝ) : circleChart φ 0 = Real.cos φ := rfl
@[simp] lemma circleChart_apply_one (φ : ℝ) : circleChart φ 1 = Real.sin φ := rfl
@[simp] lemma rayProbe₂_apply_zero (r : ℝ) : rayProbe₂ r 0 = r := rfl
@[simp] lemma rayProbe₂_apply_one (r : ℝ) : rayProbe₂ r 1 = 0 := rfl

lemma continuous_circleChart : Continuous circleChart := by
  have htoLp : Continuous
      (fun v : Fin 2 → ℝ => (WithLp.toLp 2 v : EuclideanSpace ℝ (Fin 2))) := by
    fun_prop
  have h : Continuous fun φ : ℝ =>
      (![Real.cos φ, Real.sin φ] : Fin 2 → ℝ) := by
    refine continuous_pi fun i => ?_
    fin_cases i
    · change Continuous Real.cos; exact Real.continuous_cos
    · change Continuous Real.sin; exact Real.continuous_sin
  exact htoLp.comp h

/-- **The probe-distance identity.**  `‖r•e₁ - s•circleChart φ‖² = r²+s²-2rs cos φ`
`= shellDist r s (cos φ)²`. -/
lemma dist_sq_rayProbe₂_smul_circleChart (r s φ : ℝ) :
    ‖rayProbe₂ r - s • circleChart φ‖ ^ 2 = r ^ 2 + s ^ 2 - 2 * r * s * Real.cos φ := by
  rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_two]
  simp only [PiLp.sub_apply, PiLp.smul_apply, rayProbe₂_apply_zero,
    rayProbe₂_apply_one, circleChart_apply_zero, circleChart_apply_one,
    smul_eq_mul]
  have hcs : Real.cos φ ^ 2 + Real.sin φ ^ 2 = 1 := Real.cos_sq_add_sin_sq φ
  nlinarith [hcs]

lemma norm_rayProbe₂_sub_smul_circleChart (r s φ : ℝ) :
    ‖rayProbe₂ r - s • circleChart φ‖ = shellDist r s (Real.cos φ) := by
  rw [shellDist, ← dist_sq_rayProbe₂_smul_circleChart r s φ]
  exact (Real.sqrt_sq (norm_nonneg _)).symm

/-! ## The radial-mixture measure and its integral collapse -/

/-- The base measure on the angle domain `[-π,π]`, normalised to a
probability measure (total mass `2π · (2π)⁻¹ = 1`). -/
noncomputable def chartBase₂ : Measure ℝ :=
  (ENNReal.ofReal (2 * Real.pi))⁻¹ • volume.restrict (Ioc (-Real.pi) Real.pi)

instance : IsProbabilityMeasure chartBase₂ := by
  constructor
  have hπ : (0 : ℝ) < 2 * Real.pi := by positivity
  rw [chartBase₂, Measure.smul_apply, smul_eq_mul, Measure.restrict_apply_univ,
    Real.volume_Ioc]
  rw [show Real.pi - -Real.pi = 2 * Real.pi by ring]
  exact ENNReal.inv_mul_cancel (ENNReal.ofReal_pos.mpr hπ).ne' ENNReal.ofReal_ne_top

/-- The chart pushforward map `(s, φ) ↦ s • circleChart φ`. -/
noncomputable def chartMap₂ (z : ℝ × ℝ) : EuclideanSpace ℝ (Fin 2) :=
  z.1 • circleChart z.2

@[simp] lemma chartMap₂_mk (s φ : ℝ) : chartMap₂ (s, φ) = s • circleChart φ := rfl

lemma continuous_chartMap₂ : Continuous chartMap₂ := by
  unfold chartMap₂
  exact continuous_fst.smul (continuous_circleChart.comp continuous_snd)

/-- **The `n = 2` radial-mixture measure**: the rotation-invariant probability
measure on `ℝ²` with radial profile `ν`, a single pushforward of
`ν ⊗ chartBase₂`. -/
noncomputable def radialMixture₂ (ν : Measure ℝ) : Measure (EuclideanSpace ℝ (Fin 2)) :=
  (ν.prod chartBase₂).map chartMap₂

instance radialMixture₂_isProbabilityMeasure
    (ν : Measure ℝ) [IsProbabilityMeasure ν] : IsProbabilityMeasure (radialMixture₂ ν) :=
  Measure.isProbabilityMeasure_map continuous_chartMap₂.aemeasurable

/-- **Integral collapse.**  Every integral against `radialMixture₂ ν` reduces
to a `ν`-integral of a `φ`-integral over `[-π,π]`. -/
lemma integral_radialMixture₂ (ν : Measure ℝ) [IsProbabilityMeasure ν]
    {f : EuclideanSpace ℝ (Fin 2) → ℝ} (hf : Integrable f (radialMixture₂ ν)) :
    ∫ y, f y ∂(radialMixture₂ ν)
      = ∫ s, ∫ φ, f (chartMap₂ (s, φ)) ∂chartBase₂ ∂ν := by
  have hmap : AEMeasurable chartMap₂ (ν.prod chartBase₂) :=
    continuous_chartMap₂.aemeasurable
  have hint : Integrable (fun z => f (chartMap₂ z)) (ν.prod chartBase₂) :=
    (integrable_map_measure hf.aestronglyMeasurable hmap).mp hf
  rw [radialMixture₂, integral_map hmap hf.aestronglyMeasurable]
  exact integral_prod _ hint

/-- The uniform-`φ` average reduces to a normalized interval integral. -/
lemma integral_chartBase₂ (g : ℝ → ℝ) :
    ∫ φ, g φ ∂chartBase₂ = (2 * Real.pi)⁻¹ * ∫ φ in Ioc (-Real.pi) Real.pi, g φ := by
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-Real.pi) Real.pi)) := by
    constructor
    rw [Measure.restrict_apply_univ, Real.volume_Ioc]
    exact ENNReal.ofReal_lt_top
  rw [chartBase₂, integral_smul_measure, ENNReal.toReal_inv,
    ENNReal.toReal_ofReal (by positivity), smul_eq_mul]

/-! ## The per-shell objects -/

/-- Per-shell uniform-`φ` average of the Laplace kernel (`n = 2`). -/
noncomputable def shellZ₂ (τ r s : ℝ) : ℝ :=
  (2 * Real.pi)⁻¹ * ∫ φ in Ioc (-Real.pi) Real.pi,
    Real.exp (-(1 / τ) * shellDist r s (Real.cos φ))

/-- Per-shell average of the axial coordinate `s cos φ · e^{-d/τ}`. -/
noncomputable def shellT₂ (τ r s : ℝ) : ℝ :=
  (2 * Real.pi)⁻¹ * ∫ φ in Ioc (-Real.pi) Real.pi,
    Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) * (s * Real.cos φ)

/-- Per-shell average of the axial drift numerator `(s cos φ - r) e^{-d/τ}`. -/
noncomputable def shellD₂ (τ r s : ℝ) : ℝ :=
  (2 * Real.pi)⁻¹ * ∫ φ in Ioc (-Real.pi) Real.pi,
    Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) * (s * Real.cos φ - r)

/-- Per-shell average of `(ρ²/d) e^{-d/τ}` with `ρ² = s² sin²φ`. -/
noncomputable def shellRhoSqOverDist₂ (τ r s : ℝ) : ℝ :=
  (2 * Real.pi)⁻¹ * ∫ φ in Ioc (-Real.pi) Real.pi,
    Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
      (s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ))

/-! ## The normalizer bridge -/

lemma laplaceKernel_rayProbe₂_chart (τ r s φ : ℝ) :
    laplaceKernel τ (rayProbe₂ r) (s • circleChart φ)
      = Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) := by
  simp only [laplaceKernel]
  rw [norm_rayProbe₂_sub_smul_circleChart]

lemma continuous_laplaceKernel_rayProbe₂ (τ r : ℝ) :
    Continuous (fun y : EuclideanSpace ℝ (Fin 2) => laplaceKernel τ (rayProbe₂ r) y) := by
  simp only [laplaceKernel]
  exact Real.continuous_exp.comp
    (((continuous_const.sub continuous_id).norm).const_mul (-(1 / τ)))

lemma laplaceKernel_rayProbe₂_nonneg (τ r : ℝ) (y : EuclideanSpace ℝ (Fin 2)) :
    0 ≤ laplaceKernel τ (rayProbe₂ r) y := (Real.exp_pos _).le

lemma laplaceKernel_rayProbe₂_le_one (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin 2)) : laplaceKernel τ (rayProbe₂ r) y ≤ 1 := by
  have h0 : -(1 / τ) * ‖rayProbe₂ r - y‖ ≤ 0 := by
    have hnn : 0 ≤ 1 / τ * ‖rayProbe₂ r - y‖ :=
      mul_nonneg (one_div_pos.mpr hτ).le (norm_nonneg _)
    linarith [neg_mul (1 / τ) ‖rayProbe₂ r - y‖]
  simp only [laplaceKernel]
  have h := Real.exp_le_exp.mpr h0
  rwa [Real.exp_zero] at h

/-- Continuity of the reduced `φ`-integrand of `shellZ₂`. -/
private lemma continuous_shellKernel₂ (τ r s : ℝ) :
    Continuous fun φ : ℝ => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) :=
  Real.continuous_exp.comp
    (((continuous_shellDist_u r s).comp Real.continuous_cos).const_mul (-(1 / τ)))

/-- **The ray normalizer is a `ν`-mixture of per-shell `φ`-averages.** -/
theorem kernelNormalizer_radialMixture₂ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    kernelNormalizer (laplaceKernel τ) (radialMixture₂ ν) (rayProbe₂ r)
      = ∫ s, shellZ₂ τ r s ∂ν := by
  have hf : Integrable (fun y => laplaceKernel τ (rayProbe₂ r) y) (radialMixture₂ ν) :=
    ⟨(continuous_laplaceKernel_rayProbe₂ τ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := 1)
        (ae_of_all _ (fun y => by
          rw [Real.norm_eq_abs, abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ r y)]
          exact laplaceKernel_rayProbe₂_le_one τ hτ r y))⟩
  rw [kernelNormalizer, integral_radialMixture₂ ν hf]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))
  rw [shellZ₂, ← integral_chartBase₂
    (fun φ => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)))]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun φ => ?_))
  simp only [chartMap₂_mk]
  exact laplaceKernel_rayProbe₂_chart τ r s φ

/-! ## The tangential identity (T) -/

/-- Derivative in `φ` of the reduced kernel, off the collision shell (in the
raw `.exp` form; the `IBP` massages it as needed). -/
private lemma hasDerivAt_shellKernel₂ {τ r s φ : ℝ}
    (hq : 0 < r ^ 2 + s ^ 2 - 2 * r * s * Real.cos φ) :
    HasDerivAt (fun t => Real.exp (-(1 / τ) * shellDist r s (Real.cos t)))
      (Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
        (-(1 / τ) * (-(r * s / shellDist r s (Real.cos φ)) * -Real.sin φ))) φ := by
  have hcos : HasDerivAt Real.cos (-Real.sin φ) φ := Real.hasDerivAt_cos φ
  exact (((hasDerivAt_shellDist_u hq).comp φ hcos).const_mul (-(1 / τ))).exp

/-- The reduced `φ`-integrands of `shellRhoSqOverDist₂` and `shellT₂`. -/
private noncomputable def rhoInt₂ (τ r s : ℝ) : ℝ :=
  ∫ φ in Ioc (-Real.pi) Real.pi,
    Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
      (s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ))

private noncomputable def tInt₂ (τ r s : ℝ) : ℝ :=
  ∫ φ in Ioc (-Real.pi) Real.pi,
    Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) * (s * Real.cos φ)

private lemma shellDist_cos_pos_of_ne {r s : ℝ} (hr : 0 < r) (hs : 0 ≤ s)
    (hne : s ≠ r) (φ : ℝ) : 0 < r ^ 2 + s ^ 2 - 2 * r * s * Real.cos φ := by
  have h1 : r - s ≠ 0 := sub_ne_zero.mpr (Ne.symm hne)
  have h2 : 0 < (r - s) ^ 2 :=
    lt_of_le_of_ne (sq_nonneg (r - s)) (Ne.symm (pow_ne_zero 2 h1))
  nlinarith [Real.neg_one_le_cos φ, Real.cos_le_one φ,
    mul_nonneg (mul_nonneg hr.le hs) (by linarith [Real.cos_le_one φ] :
      (0 : ℝ) ≤ 1 - Real.cos φ)]

private lemma rhoInt₂_eq_of_ne {τ r s : ℝ} (hτ : 0 < τ) (hr : 0 < r)
    (hs : 0 ≤ s) (hne : s ≠ r) :
    rhoInt₂ τ r s = (τ / r) * tInt₂ τ r s := by
  have huIcc : uIcc (-Real.pi) Real.pi = Icc (-Real.pi) Real.pi :=
    uIcc_of_le (by linarith [Real.pi_pos])
  have hq : ∀ φ : ℝ, 0 < r ^ 2 + s ^ 2 - 2 * r * s * Real.cos φ :=
    fun φ => shellDist_cos_pos_of_ne hr hs hne φ
  have hd0 : ∀ φ : ℝ, shellDist r s (Real.cos φ) ≠ 0 :=
    fun φ => (Real.sqrt_pos.mpr (hq φ)).ne'
  -- IBP with u = -(τ/r)·s·sin, v = K (kernel); u', v' their derivatives
  have hgd : ∀ φ ∈ uIcc (-Real.pi) Real.pi,
      HasDerivAt (fun t => -(τ / r) * s * Real.sin t)
        (-(τ / r) * s * Real.cos φ) φ :=
    fun φ _ => (Real.hasDerivAt_sin φ).const_mul (-(τ / r) * s)
  have hKd : ∀ φ ∈ uIcc (-Real.pi) Real.pi,
      HasDerivAt (fun t => Real.exp (-(1 / τ) * shellDist r s (Real.cos t)))
        (Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
          (-(1 / τ) * (-(r * s / shellDist r s (Real.cos φ)) * -Real.sin φ))) φ :=
    fun φ _ => hasDerivAt_shellKernel₂ (hq φ)
  have hgi : IntervalIntegrable (fun φ => -(τ / r) * s * Real.cos φ)
      volume (-Real.pi) Real.pi :=
    (continuous_const.mul Real.continuous_cos).intervalIntegrable _ _
  have hKi : IntervalIntegrable
      (fun φ => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
        (-(1 / τ) * (-(r * s / shellDist r s (Real.cos φ)) * -Real.sin φ)))
      volume (-Real.pi) Real.pi := by
    refine ContinuousOn.intervalIntegrable ?_
    have hden : ∀ φ ∈ uIcc (-Real.pi) Real.pi,
        shellDist r s (Real.cos φ) ≠ 0 := fun φ _ => hd0 φ
    refine (continuous_shellKernel₂ τ r s).continuousOn.mul ?_
    refine continuousOn_const.mul ?_
    refine ContinuousOn.mul ?_ Real.continuous_sin.continuousOn.neg
    exact (continuousOn_const.div
      ((continuous_shellDist_u r s).comp Real.continuous_cos).continuousOn
      hden).neg
  have hibp := intervalIntegral.integral_mul_deriv_eq_deriv_mul hgd hKd hgi hKi
  -- boundary terms vanish (`sin (±π) = 0`)
  have hb1 : -(τ / r) * s * Real.sin Real.pi = 0 := by rw [Real.sin_pi]; ring
  have hb0 : -(τ / r) * s * Real.sin (-Real.pi) = 0 := by
    rw [Real.sin_neg, Real.sin_pi]; ring
  rw [hb1, hb0, zero_mul, zero_mul, sub_zero, zero_sub] at hibp
  -- hibp : ∫ u·v' = -∫ u'·v
  have hlhs : (∫ φ in (-Real.pi)..Real.pi,
        -(τ / r) * s * Real.sin φ *
          (Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
            (-(1 / τ) * (-(r * s / shellDist r s (Real.cos φ)) * -Real.sin φ))))
      = rhoInt₂ τ r s := by
    rw [rhoInt₂, ← intervalIntegral.integral_of_le (by linarith [Real.pi_pos])]
    refine intervalIntegral.integral_congr fun φ _ => ?_
    rw [mul_comm (Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)))
      (s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ))]
    field_simp [hd0 φ]
  have hrhs : (∫ φ in (-Real.pi)..Real.pi,
        -(τ / r) * s * Real.cos φ *
          Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)))
      = -(τ / r) * tInt₂ τ r s := by
    rw [tInt₂, ← intervalIntegral.integral_of_le (by linarith [Real.pi_pos]),
      ← intervalIntegral.integral_const_mul]
    refine intervalIntegral.integral_congr fun φ _ => ?_
    ring
  rw [hlhs, hrhs] at hibp
  linarith [hibp]

private lemma continuous_shellDist_cos_s (r φ : ℝ) :
    Continuous (fun s => shellDist r s (Real.cos φ)) := by
  have hfe : (fun s => shellDist r s (Real.cos φ))
      = fun s => Real.sqrt (r ^ 2 + s ^ 2 - 2 * r * s * Real.cos φ) := rfl
  rw [hfe]
  refine Real.continuous_sqrt.comp ?_
  exact (continuous_const.add (continuous_pow 2)).sub
    ((continuous_const.mul continuous_id).mul continuous_const)

private lemma continuousAt_tInt₂ {τ r : ℝ} (hτ : 0 < τ) (hr : 0 < r) :
    ContinuousAt (fun s => tInt₂ τ r s) r := by
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-Real.pi) Real.pi)) := by
    constructor
    rw [Measure.restrict_apply_univ, Real.volume_Ioc]
    exact ENNReal.ofReal_lt_top
  unfold tInt₂
  refine continuousAt_of_dominated (bound := fun _ => 2 * r) ?_ ?_ ?_ ?_
  · filter_upwards with s
    refine Continuous.aestronglyMeasurable ?_
    exact (continuous_shellKernel₂ τ r s).mul
      (continuous_const.mul Real.continuous_cos)
  · filter_upwards [Metric.ball_mem_nhds r hr] with s hs
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with φ _
    have hsr : |s - r| < r := by
      have h := Metric.mem_ball.mp hs; rwa [Real.dist_eq] at h
    obtain ⟨hs1, hs2⟩ := abs_lt.mp hsr
    have hK : |Real.exp (-(1 / τ) * shellDist r s (Real.cos φ))| ≤ 1 := by
      rw [abs_of_pos (Real.exp_pos _)]
      exact exp_shellDist_le_one hτ r s (Real.cos φ)
    have hsc : |s * Real.cos φ| ≤ 2 * r := by
      rw [abs_mul]
      have hcos : |Real.cos φ| ≤ 1 := Real.abs_cos_le_one φ
      have hsabs : |s| ≤ 2 * r := by
        rw [abs_of_pos (by linarith : (0 : ℝ) < s)]; linarith
      calc |s| * |Real.cos φ| ≤ (2 * r) * 1 :=
            mul_le_mul hsabs hcos (abs_nonneg _) (by linarith)
        _ = 2 * r := mul_one _
    rw [Real.norm_eq_abs, abs_mul]
    calc |Real.exp (-(1 / τ) * shellDist r s (Real.cos φ))| * |s * Real.cos φ|
        ≤ 1 * (2 * r) := mul_le_mul hK hsc (abs_nonneg _) zero_le_one
      _ = 2 * r := by ring
  · exact integrable_const _
  · filter_upwards with φ
    exact ((Real.continuous_exp.comp
      ((continuous_shellDist_cos_s r φ).const_mul (-(1 / τ)))).mul
      (continuous_id.mul continuous_const)).continuousAt

private lemma continuousAt_rhoInt₂ {τ r : ℝ} (hτ : 0 < τ) (hr : 0 < r) :
    ContinuousAt (fun s => rhoInt₂ τ r s) r := by
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-Real.pi) Real.pi)) := by
    constructor
    rw [Measure.restrict_apply_univ, Real.volume_Ioc]
    exact ENNReal.ofReal_lt_top
  unfold rhoInt₂
  refine continuousAt_of_dominated (bound := fun _ => 4 * r) ?_ ?_ ?_ ?_
  · filter_upwards with s
    refine Measurable.aestronglyMeasurable ?_
    refine ((continuous_shellKernel₂ τ r s).measurable).mul ?_
    have hnum : Measurable fun φ => s ^ 2 * Real.sin φ ^ 2 := by fun_prop
    exact hnum.div ((continuous_shellDist_u r s).comp
      Real.continuous_cos).measurable
  · filter_upwards [Metric.ball_mem_nhds r hr] with s hs
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with φ _
    have hsr : |s - r| < r := by
      have h := Metric.mem_ball.mp hs; rwa [Real.dist_eq] at h
    obtain ⟨hs1, hs2⟩ := abs_lt.mp hsr
    have hspos : (0 : ℝ) < s := by linarith
    have hK : |Real.exp (-(1 / τ) * shellDist r s (Real.cos φ))| ≤ 1 := by
      rw [abs_of_pos (Real.exp_pos _)]
      exact exp_shellDist_le_one hτ r s (Real.cos φ)
    have hpay : |s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ)| ≤ 4 * r := by
      have hu2 : Real.cos φ ^ 2 ≤ 1 := by
        nlinarith [Real.cos_le_one φ, Real.neg_one_le_cos φ]
      have heq : s ^ 2 * Real.sin φ ^ 2 = shellRhoSq s (Real.cos φ) := by
        rw [shellRhoSq, Real.sin_sq]
      rw [heq, abs_of_nonneg (shellRhoSq_div_shellDist_nonneg hu2 r)]
      have hle : shellRhoSq s (Real.cos φ) / shellDist r s (Real.cos φ)
          ≤ shellDist r s (Real.cos φ) := shellRhoSq_div_shellDist_le hu2 r
      have hdle := shellDist_le_add hr.le hspos.le (Real.neg_one_le_cos φ)
      linarith
    rw [Real.norm_eq_abs, abs_mul]
    calc |Real.exp (-(1 / τ) * shellDist r s (Real.cos φ))|
          * |s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ)|
        ≤ 1 * (4 * r) := mul_le_mul hK hpay (abs_nonneg _) zero_le_one
      _ = 4 * r := by ring
  · exact integrable_const _
  · filter_upwards with φ
    have hK : Continuous fun s => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) :=
      Real.continuous_exp.comp
        ((continuous_shellDist_cos_s r φ).const_mul (-(1 / τ)))
    rcases eq_or_ne (Real.sin φ) 0 with hsin | hsin
    · have hzero : (fun s => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
          (s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ)))
          = fun s => (0 : ℝ) := by
        funext s; rw [hsin]; simp
      rw [hzero]; exact continuousAt_const
    · have hdpos : 0 < shellDist r r (Real.cos φ) := by
        refine Real.sqrt_pos.mpr ?_
        have hcos_lt : Real.cos φ < 1 :=
          lt_of_le_of_ne (Real.cos_le_one φ)
            (fun h => hsin (Real.sin_eq_zero_iff_cos_eq.mpr (Or.inl h)))
        nlinarith [hcos_lt, mul_pos hr hr]
      have hdiv : ContinuousAt
          (fun s => s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ)) r :=
        ((continuous_pow 2).continuousAt.mul continuousAt_const).div
          (continuous_shellDist_cos_s r φ).continuousAt hdpos.ne'
      exact hK.continuousAt.mul hdiv

/-- **The `n = 2` tangential identity (T)**: for `τ > 0`, probe radius `r > 0`,
and every shell `s ≥ 0` (the collision shell `s = r` included, by
`s`-continuity), `shellRhoSqOverDist₂ = (τ/r)·shellT₂`. -/
theorem shellRhoSqOverDist₂_eq_shellT₂ {τ : ℝ} (hτ : 0 < τ) {r : ℝ}
    (hr : 0 < r) {s : ℝ} (hs : 0 ≤ s) :
    shellRhoSqOverDist₂ τ r s = (τ / r) * shellT₂ τ r s := by
  have hbridge : rhoInt₂ τ r s = (τ / r) * tInt₂ τ r s := by
    rcases ne_or_eq s r with hne | heq
    · exact rhoInt₂_eq_of_ne hτ hr hs hne
    · rw [heq]
      have hR : Tendsto (fun s => rhoInt₂ τ r s) (𝓝[≠] r)
          (𝓝 (rhoInt₂ τ r r)) :=
        Filter.Tendsto.mono_left (continuousAt_rhoInt₂ hτ hr) nhdsWithin_le_nhds
      have hT : Tendsto (fun s => (τ / r) * tInt₂ τ r s) (𝓝[≠] r)
          (𝓝 ((τ / r) * tInt₂ τ r r)) :=
        Filter.Tendsto.mono_left ((continuousAt_tInt₂ hτ hr).const_mul (τ / r))
          nhdsWithin_le_nhds
      have hball : Metric.ball r r ∈ 𝓝[≠] r :=
        nhdsWithin_le_nhds (Metric.ball_mem_nhds r hr)
      have heqev : (fun s => rhoInt₂ τ r s)
          =ᶠ[𝓝[≠] r] fun s => (τ / r) * tInt₂ τ r s := by
        filter_upwards [self_mem_nhdsWithin, hball] with s hs1 hs2
        have hspos : (0 : ℝ) < s := by
          have h := Metric.mem_ball.mp hs2; rw [Real.dist_eq] at h
          obtain ⟨h1, h2⟩ := abs_lt.mp h; linarith
        exact rhoInt₂_eq_of_ne hτ hr hspos.le hs1
      exact tendsto_nhds_unique_of_eventuallyEq hR hT heqev
  rw [shellRhoSqOverDist₂, shellT₂,
    show (∫ φ in Ioc (-Real.pi) Real.pi,
        Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
          (s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ)))
      = rhoInt₂ τ r s from rfl,
    show (∫ φ in Ioc (-Real.pi) Real.pi,
        Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) * (s * Real.cos φ))
      = tInt₂ τ r s from rfl, hbridge]
  ring

end DriftingIdentifiability
