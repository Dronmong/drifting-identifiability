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

/-! ## Shell integrability -/

private lemma continuous_shellDistPair₂ (r : ℝ) :
    Continuous fun z : ℝ × ℝ => shellDist r z.1 (Real.cos z.2) := by
  have hfe : (fun z : ℝ × ℝ => shellDist r z.1 (Real.cos z.2))
      = fun z => Real.sqrt (r ^ 2 + z.1 ^ 2 - 2 * r * z.1 * Real.cos z.2) := rfl
  rw [hfe]
  exact Real.continuous_sqrt.comp (by fun_prop)

private lemma stronglyMeasurable_setIntegral₂ {F : ℝ × ℝ → ℝ}
    (hF : StronglyMeasurable F) :
    StronglyMeasurable
      (fun s : ℝ => ∫ φ in Ioc (-Real.pi) Real.pi, F (s, φ)) :=
  hF.integral_prod_right'

private lemma isFiniteMeasure_IocPi :
    IsFiniteMeasure (volume.restrict (Ioc (-Real.pi) Real.pi)) := by
  constructor
  rw [Measure.restrict_apply_univ, Real.volume_Ioc]
  exact ENNReal.ofReal_lt_top

/-- The generic per-shell bound: a uniform `φ`-average of a `C`-bounded
integrand is `C`-bounded. -/
private lemma abs_shell₂_le {g : ℝ → ℝ} {C : ℝ}
    (hint : IntegrableOn g (Ioc (-Real.pi) Real.pi))
    (hbd : ∀ φ, |g φ| ≤ C) :
    |(2 * Real.pi)⁻¹ * ∫ φ in Ioc (-Real.pi) Real.pi, g φ| ≤ C := by
  haveI := isFiniteMeasure_IocPi
  rw [abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ (2 * Real.pi)⁻¹)]
  have hcint : IntegrableOn (fun _ : ℝ => C) (Ioc (-Real.pi) Real.pi) :=
    (continuous_const.integrableOn_Icc).mono_set Ioc_subset_Icc_self
  have h2 : (∫ φ in Ioc (-Real.pi) Real.pi, |g φ|)
      ≤ ∫ _φ in Ioc (-Real.pi) Real.pi, C :=
    setIntegral_mono_on hint.abs hcint measurableSet_Ioc (fun φ _ => hbd φ)
  have hconst : (∫ _φ in Ioc (-Real.pi) Real.pi, C) = C * (2 * Real.pi) := by
    rw [setIntegral_const, Real.volume_real_Ioc_of_le (by linarith [Real.pi_pos])]
    rw [smul_eq_mul]; ring
  have hpi : (0 : ℝ) < 2 * Real.pi := by positivity
  calc (2 * Real.pi)⁻¹ * |∫ φ in Ioc (-Real.pi) Real.pi, g φ|
      ≤ (2 * Real.pi)⁻¹ * (C * (2 * Real.pi)) := by
        apply mul_le_mul_of_nonneg_left _ (by positivity)
        calc |∫ φ in Ioc (-Real.pi) Real.pi, g φ|
            ≤ ∫ φ in Ioc (-Real.pi) Real.pi, |g φ| := abs_integral_le_integral_abs
          _ ≤ C * (2 * Real.pi) := by rw [← hconst]; exact h2
    _ = C := by field_simp

private lemma continuous_shellKernel₂' (τ r s : ℝ) :
    Continuous fun φ : ℝ => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) :=
  Real.continuous_exp.comp
    (((continuous_shellDist_u r s).comp Real.continuous_cos).const_mul _)

/-- `|X| ≤ d` at the chart: `|s cos φ − r| ≤ shellDist r s (cos φ)`. -/
private lemma abs_axial_le_shellDist₂ (r s φ : ℝ) :
    |s * Real.cos φ - r| ≤ shellDist r s (Real.cos φ) := by
  have hu2 : Real.cos φ ^ 2 ≤ 1 := by
    nlinarith [Real.neg_one_le_cos φ, Real.cos_le_one φ]
  have hd2 : shellDist r s (Real.cos φ) ^ 2
      = (s * Real.cos φ - r) ^ 2 + s ^ 2 * (1 - Real.cos φ ^ 2) := by
    have h := shellDist_sq_eq_axial_add_rho (u := Real.cos φ) hu2 r s
    rw [h]; rfl
  have hle : (s * Real.cos φ - r) ^ 2 ≤ shellDist r s (Real.cos φ) ^ 2 := by
    rw [hd2]
    have hnn : 0 ≤ s ^ 2 * (1 - Real.cos φ ^ 2) := by
      apply mul_nonneg (sq_nonneg s)
      nlinarith [Real.cos_le_one φ, Real.neg_one_le_cos φ]
    linarith
  calc |s * Real.cos φ - r| = Real.sqrt ((s * Real.cos φ - r) ^ 2) :=
        (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt (shellDist r s (Real.cos φ) ^ 2) := Real.sqrt_le_sqrt hle
    _ = shellDist r s (Real.cos φ) := Real.sqrt_sq (shellDist_nonneg r s _)

/-- The core per-shell `K·payload ≤ τe⁻¹` bound (payload `≤ d`). -/
private lemma exp_mul_le_τe {τ r s : ℝ} (hτ : 0 < τ) {p : ℝ} (φ : ℝ)
    (hpd : p ≤ shellDist r s (Real.cos φ)) :
    Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) * p ≤ τ * Real.exp (-1) := by
  have hExp : Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
        shellDist r s (Real.cos φ) ≤ τ * Real.exp (-1) := by
    rw [mul_comm]
    have he : -(1 / τ) * shellDist r s (Real.cos φ)
        = -shellDist r s (Real.cos φ) / τ := by ring
    rw [he]
    exact mul_exp_neg_div_le hτ (shellDist_nonneg r s _)
  calc Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) * p
      ≤ Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
          shellDist r s (Real.cos φ) :=
        mul_le_mul_of_nonneg_left hpd (Real.exp_pos _).le
    _ ≤ τ * Real.exp (-1) := hExp

/-! ## The ray tangential objects -/

noncomputable def radialRayT₂ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellT₂ τ r s ∂ν

noncomputable def radialRayRhoSqOverDist₂ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellRhoSqOverDist₂ τ r s ∂ν

/-! ## Integrability of the shell profiles -/

private lemma measurable_shellDistPair₂ (r : ℝ) :
    Measurable (fun z : ℝ × ℝ => shellDist r z.1 (Real.cos z.2)) :=
  (continuous_shellDistPair₂ r).measurable

private lemma sq_le_one_cos (φ : ℝ) : Real.cos φ ^ 2 ≤ 1 := by
  nlinarith [Real.neg_one_le_cos φ, Real.cos_le_one φ]

private lemma rho_div_le_shellDist₂ (r s φ : ℝ) :
    s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ)
      ≤ shellDist r s (Real.cos φ) := by
  have heq : s ^ 2 * Real.sin φ ^ 2 = shellRhoSq s (Real.cos φ) := by
    rw [shellRhoSq, Real.sin_sq]
  rw [heq]
  exact shellRhoSq_div_shellDist_le (sq_le_one_cos φ) r

private lemma sqsub_div_le_shellDist₂ (r s φ : ℝ) :
    (s * Real.cos φ - r) ^ 2 / shellDist r s (Real.cos φ)
      ≤ shellDist r s (Real.cos φ) := by
  rcases eq_or_lt_of_le (shellDist_nonneg r s (Real.cos φ)) with h0 | hpos
  · simp [← h0]
  · rw [div_le_iff₀ hpos]
    have h := abs_axial_le_shellDist₂ r s φ
    nlinarith [sq_abs (s * Real.cos φ - r), h, abs_nonneg (s * Real.cos φ - r)]

/-- Generic shell integrability: a `C`-bounded measurable chart integrand
gives a `ν`-integrable per-shell average. -/
private lemma integrable_shell₂_of_bound {F : ℝ × ℝ → ℝ} {C : ℝ}
    (hF : Measurable F) (hbd : ∀ s φ, |F (s, φ)| ≤ C)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    Integrable (fun s => (2 * Real.pi)⁻¹ *
      ∫ φ in Ioc (-Real.pi) Real.pi, F (s, φ)) ν := by
  haveI := isFiniteMeasure_IocPi
  refine Integrable.of_bound ?_ C (ae_of_all _ fun s => ?_)
  · exact (StronglyMeasurable.const_mul
      (stronglyMeasurable_setIntegral₂ hF.stronglyMeasurable) _).aestronglyMeasurable
  · rw [Real.norm_eq_abs]
    have hint : IntegrableOn (fun φ => F (s, φ)) (Ioc (-Real.pi) Real.pi) := by
      refine Integrable.mono' (integrable_const C)
        ((hF.comp (by fun_prop : Measurable
          (fun φ : ℝ => (s, φ)))).aestronglyMeasurable) ?_
      filter_upwards with φ
      rw [Real.norm_eq_abs]
      exact hbd s φ
    exact abs_shell₂_le hint (fun φ => hbd s φ)

lemma integrable_shellZ₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) :
    Integrable (fun s => shellZ₂ τ r s) ν :=
  integrable_shell₂_of_bound
    (F := fun z => Real.exp (-(1 / τ) * shellDist r z.1 (Real.cos z.2))) (C := 1)
    (Real.measurable_exp.comp ((measurable_shellDistPair₂ r).const_mul _))
    (fun s φ => by
      rw [abs_of_pos (Real.exp_pos _)]
      exact exp_shellDist_le_one hτ r s (Real.cos φ)) ν

lemma integrable_shellD₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) :
    Integrable (fun s => shellD₂ τ r s) ν :=
  integrable_shell₂_of_bound
    (F := fun z => Real.exp (-(1 / τ) * shellDist r z.1 (Real.cos z.2)) *
      (z.1 * Real.cos z.2 - r)) (C := τ * Real.exp (-1))
    ((Real.measurable_exp.comp ((measurable_shellDistPair₂ r).const_mul _)).mul
      (by fun_prop))
    (fun s φ => by
      rw [abs_mul, abs_of_pos (Real.exp_pos _)]
      exact exp_mul_le_τe hτ φ (abs_axial_le_shellDist₂ r s φ)) ν

lemma integrable_shellQ₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) :
    Integrable (fun s => shellQ₂ τ r s) ν :=
  integrable_shell₂_of_bound
    (F := fun z => Real.exp (-(1 / τ) * shellDist r z.1 (Real.cos z.2)) *
      ((z.1 * Real.cos z.2 - r) ^ 2 / shellDist r z.1 (Real.cos z.2)))
    (C := τ * Real.exp (-1))
    ((Real.measurable_exp.comp ((measurable_shellDistPair₂ r).const_mul _)).mul
      ((by fun_prop : Measurable (fun z : ℝ × ℝ => (z.1 * Real.cos z.2 - r) ^ 2)).div
        (measurable_shellDistPair₂ r)))
    (fun s φ => by
      rw [abs_of_nonneg (mul_nonneg (Real.exp_pos _).le
        (div_nonneg (sq_nonneg _) (shellDist_nonneg _ _ _)))]
      exact exp_mul_le_τe hτ φ (sqsub_div_le_shellDist₂ r s φ)) ν

lemma integrable_shellRhoSqOverDist₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) :
    Integrable (fun s => shellRhoSqOverDist₂ τ r s) ν :=
  integrable_shell₂_of_bound
    (F := fun z => Real.exp (-(1 / τ) * shellDist r z.1 (Real.cos z.2)) *
      (z.1 ^ 2 * Real.sin z.2 ^ 2 / shellDist r z.1 (Real.cos z.2)))
    (C := τ * Real.exp (-1))
    ((Real.measurable_exp.comp ((measurable_shellDistPair₂ r).const_mul _)).mul
      ((by fun_prop : Measurable (fun z : ℝ × ℝ => z.1 ^ 2 * Real.sin z.2 ^ 2)).div
        (measurable_shellDistPair₂ r)))
    (fun s φ => by
      rw [abs_of_nonneg (mul_nonneg (Real.exp_pos _).le
        (div_nonneg (by positivity) (shellDist_nonneg _ _ _)))]
      exact exp_mul_le_τe hτ φ (rho_div_le_shellDist₂ r s φ)) ν

/-! ## Per-shell decompositions -/

lemma shellT₂_eq (τ r s : ℝ) :
    shellT₂ τ r s = shellD₂ τ r s + r * shellZ₂ τ r s := by
  haveI := isFiniteMeasure_IocPi
  have hZi : IntegrableOn
      (fun φ => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)))
      (Ioc (-Real.pi) Real.pi) :=
    (continuous_shellKernel₂' τ r s).integrableOn_Icc.mono_set Ioc_subset_Icc_self
  have hDi : IntegrableOn
      (fun φ => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
        (s * Real.cos φ - r)) (Ioc (-Real.pi) Real.pi) :=
    ((continuous_shellKernel₂' τ r s).mul (by fun_prop)).integrableOn_Icc.mono_set
      Ioc_subset_Icc_self
  rw [shellT₂, shellD₂, shellZ₂,
    show r * ((2 * Real.pi)⁻¹ *
        ∫ φ in Ioc (-Real.pi) Real.pi,
          Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)))
      = (2 * Real.pi)⁻¹ * ∫ φ in Ioc (-Real.pi) Real.pi,
          r * Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) from by
      rw [integral_const_mul]; ring,
    ← mul_add, ← integral_add hDi (hZi.const_mul r)]
  congr 1
  refine setIntegral_congr_fun measurableSet_Ioc (fun φ _ => ?_)
  ring

/-- The per-shell distance decomposition `d = X²/d + ρ²/d`. -/
private lemma shellDist_decomp₂ (r s φ : ℝ) :
    shellDist r s (Real.cos φ)
      = (s * Real.cos φ - r) ^ 2 / shellDist r s (Real.cos φ)
        + s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ) := by
  rw [← add_div]
  rcases eq_or_lt_of_le (shellDist_nonneg r s (Real.cos φ)) with h0 | hpos
  · rw [← h0]; simp
  · rw [eq_div_iff hpos.ne']
    have hsq : (s * Real.cos φ - r) ^ 2 + s ^ 2 * Real.sin φ ^ 2
        = shellDist r s (Real.cos φ) ^ 2 := by
      rw [shellDist_sq_eq_axial_add_rho (u := Real.cos φ) (sq_le_one_cos φ) r s]
      simp only [shellAxial, shellRhoSq]
      rw [Real.sin_sq]
    rw [hsq]; ring

lemma shellC₂_decomp (τ : ℝ) (hτ : 0 < τ) (r s : ℝ) :
    shellC₂ τ r s
      = τ * shellZ₂ τ r s + shellQ₂ τ r s + shellRhoSqOverDist₂ τ r s := by
  haveI := isFiniteMeasure_IocPi
  have hZi : IntegrableOn
      (fun φ => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)))
      (Ioc (-Real.pi) Real.pi) :=
    (continuous_shellKernel₂' τ r s).integrableOn_Icc.mono_set Ioc_subset_Icc_self
  have hQi : IntegrableOn
      (fun φ => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
        ((s * Real.cos φ - r) ^ 2 / shellDist r s (Real.cos φ)))
      (Ioc (-Real.pi) Real.pi) := by
    refine Integrable.mono' (integrable_const (τ * Real.exp (-1)))
      (((continuous_shellKernel₂' τ r s).measurable).mul
        ((by fun_prop : Measurable (fun φ : ℝ => (s * Real.cos φ - r) ^ 2)).div
          ((continuous_shellDist_u r s).comp Real.continuous_cos).measurable)).aestronglyMeasurable
      (ae_of_all _ fun φ => ?_)
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (Real.exp_pos _).le
      (div_nonneg (sq_nonneg _) (shellDist_nonneg _ _ _)))]
    exact exp_mul_le_τe hτ φ (sqsub_div_le_shellDist₂ r s φ)
  have hRi : IntegrableOn
      (fun φ => Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
        (s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ)))
      (Ioc (-Real.pi) Real.pi) := by
    refine Integrable.mono' (integrable_const (τ * Real.exp (-1)))
      (((continuous_shellKernel₂' τ r s).measurable).mul
        ((by fun_prop : Measurable (fun φ : ℝ => s ^ 2 * Real.sin φ ^ 2)).div
          ((continuous_shellDist_u r s).comp Real.continuous_cos).measurable)).aestronglyMeasurable
      (ae_of_all _ fun φ => ?_)
    rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg (Real.exp_pos _).le
      (div_nonneg (by positivity) (shellDist_nonneg _ _ _)))]
    exact exp_mul_le_τe hτ φ (rho_div_le_shellDist₂ r s φ)
  have hpt : (∫ φ in Ioc (-Real.pi) Real.pi,
        (τ + shellDist r s (Real.cos φ)) *
          Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)))
      = ∫ φ in Ioc (-Real.pi) Real.pi,
          (τ * Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) +
            Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
              ((s * Real.cos φ - r) ^ 2 / shellDist r s (Real.cos φ)) +
            Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
              (s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ))) :=
    setIntegral_congr_fun measurableSet_Ioc (fun φ _ => by
      linear_combination (Real.exp (-(1 / τ) * shellDist r s (Real.cos φ))) *
        shellDist_decomp₂ r s φ)
  have hs1 : (∫ φ in Ioc (-Real.pi) Real.pi,
        (τ * Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) +
          Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
            ((s * Real.cos φ - r) ^ 2 / shellDist r s (Real.cos φ)) +
          Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
            (s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ))))
      = (∫ φ in Ioc (-Real.pi) Real.pi,
          (τ * Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) +
            Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
              ((s * Real.cos φ - r) ^ 2 / shellDist r s (Real.cos φ))))
        + ∫ φ in Ioc (-Real.pi) Real.pi,
            Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
              (s ^ 2 * Real.sin φ ^ 2 / shellDist r s (Real.cos φ)) :=
    integral_add ((hZi.const_mul τ).add hQi) hRi
  have hs2 : (∫ φ in Ioc (-Real.pi) Real.pi,
        (τ * Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) +
          Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
            ((s * Real.cos φ - r) ^ 2 / shellDist r s (Real.cos φ))))
      = (∫ φ in Ioc (-Real.pi) Real.pi,
          τ * Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)))
        + ∫ φ in Ioc (-Real.pi) Real.pi,
            Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) *
              ((s * Real.cos φ - r) ^ 2 / shellDist r s (Real.cos φ)) :=
    integral_add (hZi.const_mul τ) hQi
  have hs3 : (∫ φ in Ioc (-Real.pi) Real.pi,
      τ * Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)))
      = τ * ∫ φ in Ioc (-Real.pi) Real.pi,
          Real.exp (-(1 / τ) * shellDist r s (Real.cos φ)) := integral_const_mul _ _
  rw [shellC₂, shellZ₂, shellQ₂, shellRhoSqOverDist₂, hpt, hs1, hs2, hs3]
  ring

/-! ## The ray-level closure -/

private lemma radial_ae_nonneg₂ {ν : Measure ℝ} (hsupp : ν (Iio 0) = 0) :
    ∀ᵐ s ∂ν, 0 ≤ s := by
  rw [ae_iff]
  have hset : {s : ℝ | ¬ 0 ≤ s} = Iio 0 := by ext s; simp [not_le]
  rw [hset]; exact hsupp

lemma radialRayT₂_eq (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayT₂ τ ν r = radialRayD₂ τ ν r + r * radialRayZ₂ τ ν r := by
  rw [radialRayT₂, radialRayD₂, radialRayZ₂, ← integral_const_mul,
    ← integral_add (integrable_shellD₂ τ hτ ν r)
      ((integrable_shellZ₂ τ hτ ν r).const_mul r)]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))
  exact shellT₂_eq τ r s

lemma radialRayRhoSqOverDist₂_eq (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0) {r : ℝ} (hr : 0 < r) :
    radialRayRhoSqOverDist₂ τ ν r = (τ / r) * radialRayT₂ τ ν r := by
  rw [radialRayRhoSqOverDist₂, radialRayT₂, ← integral_const_mul]
  refine integral_congr_ae ?_
  filter_upwards [radial_ae_nonneg₂ hsupp] with s hs
  exact shellRhoSqOverDist₂_eq_shellT₂ hτ hr hs

lemma radialRayC₂_eq_decomp (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayC₂ τ ν r = τ * radialRayZ₂ τ ν r + radialRayQ₂ τ ν r
      + radialRayRhoSqOverDist₂ τ ν r := by
  have hpt : (∫ s, shellC₂ τ r s ∂ν)
      = ∫ s, (τ * shellZ₂ τ r s + shellQ₂ τ r s + shellRhoSqOverDist₂ τ r s) ∂ν :=
    integral_congr_ae (Filter.Eventually.of_forall (fun s => shellC₂_decomp τ hτ r s))
  have hs1 : (∫ s, (τ * shellZ₂ τ r s + shellQ₂ τ r s
        + shellRhoSqOverDist₂ τ r s) ∂ν)
      = (∫ s, (τ * shellZ₂ τ r s + shellQ₂ τ r s) ∂ν)
        + ∫ s, shellRhoSqOverDist₂ τ r s ∂ν :=
    integral_add
      (((integrable_shellZ₂ τ hτ ν r).const_mul τ).add (integrable_shellQ₂ τ hτ ν r))
      (integrable_shellRhoSqOverDist₂ τ hτ ν r)
  have hs2 : (∫ s, (τ * shellZ₂ τ r s + shellQ₂ τ r s) ∂ν)
      = (∫ s, τ * shellZ₂ τ r s ∂ν) + ∫ s, shellQ₂ τ r s ∂ν :=
    integral_add ((integrable_shellZ₂ τ hτ ν r).const_mul τ)
      (integrable_shellQ₂ τ hτ ν r)
  have hs3 : (∫ s, τ * shellZ₂ τ r s ∂ν) = τ * ∫ s, shellZ₂ τ r s ∂ν :=
    integral_const_mul _ _
  rw [radialRayC₂, hpt, hs1, hs2, hs3, radialRayZ₂, radialRayQ₂,
    radialRayRhoSqOverDist₂]

/-- **The `n = 2` first-order closure** `C̃₂ = Q̃₂ + 2τZ̃₂ + (τ/r)D̃₂`. -/
theorem radialRayC₂_eq_closure (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0) {r : ℝ} (hr : 0 < r) :
    radialRayC₂ τ ν r = radialRayQ₂ τ ν r + 2 * τ * radialRayZ₂ τ ν r
      + (τ / r) * radialRayD₂ τ ν r := by
  rw [radialRayC₂_eq_decomp τ hτ ν r,
    radialRayRhoSqOverDist₂_eq τ hτ ν hsupp hr, radialRayT₂_eq τ hτ ν r]
  field_simp [hr.ne']
  ring

/-! ## The proved part of the sign layer (AM–GM near branch) -/

lemma two_mul_laplace_abs_first_rayProbe₂_le (τ r : ℝ)
    (y : EuclideanSpace ℝ (Fin 2)) :
    2 * (laplaceKernel τ (rayProbe₂ r) y * |y 0 - r|) ≤
      laplaceKernel τ (rayProbe₂ r) y *
          ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖) +
        laplaceKernel τ (rayProbe₂ r) y * ‖rayProbe₂ r - y‖ := by
  have hK := laplaceKernel_rayProbe₂_nonneg τ r y
  rcases eq_or_ne (‖rayProbe₂ r - y‖) 0 with h0 | hne
  · have hy : rayProbe₂ r = y := norm_sub_eq_zero_iff.mp h0
    have hX : y 0 - r = 0 := by rw [← hy, rayProbe₂_apply_zero, sub_self]
    simp [h0, hX]
  · have hdpos : 0 < ‖rayProbe₂ r - y‖ := (norm_nonneg _).lt_of_ne' hne
    have hAM : 2 * |y 0 - r| ≤
        (y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖ + ‖rayProbe₂ r - y‖ := by
      rw [← sub_nonneg]
      have hkey :
          (y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖ + ‖rayProbe₂ r - y‖ -
              2 * |y 0 - r| =
            (|y 0 - r| - ‖rayProbe₂ r - y‖) ^ 2 / ‖rayProbe₂ r - y‖ := by
        field_simp
        nlinarith [sq_abs (y 0 - r)]
      rw [hkey]; positivity
    calc 2 * (laplaceKernel τ (rayProbe₂ r) y * |y 0 - r|)
        = laplaceKernel τ (rayProbe₂ r) y * (2 * |y 0 - r|) := by ring
      _ ≤ laplaceKernel τ (rayProbe₂ r) y *
          ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖ + ‖rayProbe₂ r - y‖) :=
        mul_le_mul_of_nonneg_left hAM hK
      _ = _ := by ring

lemma integrable_laplaceKernel_mul_norm_rayProbe₂ (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin 2))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceKernel τ (rayProbe₂ r) y *
      ‖rayProbe₂ r - y‖) μ := by
  refine ⟨((continuous_laplaceKernel_rayProbe₂ τ r).mul
    ((continuous_const.sub continuous_id).norm)).aestronglyMeasurable,
    HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
      (ae_of_all _ fun y => ?_)⟩
  rw [Real.norm_eq_abs, abs_mul,
    abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ r y),
    abs_of_nonneg (norm_nonneg _)]
  exact laplaceKernel_rayProbe₂_mul_norm_le τ hτ r y

lemma integral_laplaceKernel_mul_norm_eq_C₂_sub (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    (∫ y, laplaceKernel τ (rayProbe₂ r) y *
        ‖rayProbe₂ r - y‖ ∂(radialMixture₂ ν)) =
      radialRayC₂ τ ν r - τ * radialRayZ₂ τ ν r := by
  have hCint := integrable_companion_rayProbe₂ τ hτ (radialMixture₂ ν) r
  have hZint := integrable_laplaceKernel_rayProbe₂ τ hτ (radialMixture₂ ν) r
  have hSint := integrable_laplaceKernel_mul_norm_rayProbe₂ τ hτ
    (radialMixture₂ ν) r
  rw [radialRayC₂_eq_integral τ ν r hCint, radialRayZ₂_eq_integral τ hτ ν r]
  have hdecomp : (∫ y, laplaceCompanionKernel τ (rayProbe₂ r) y
        ∂(radialMixture₂ ν))
      = τ * (∫ y, laplaceKernel τ (rayProbe₂ r) y ∂(radialMixture₂ ν)) +
        ∫ y, laplaceKernel τ (rayProbe₂ r) y *
          ‖rayProbe₂ r - y‖ ∂(radialMixture₂ ν) := by
    have hpt : (∫ y, laplaceCompanionKernel τ (rayProbe₂ r) y
          ∂(radialMixture₂ ν))
        = ∫ y, (τ * laplaceKernel τ (rayProbe₂ r) y +
            laplaceKernel τ (rayProbe₂ r) y * ‖rayProbe₂ r - y‖)
            ∂(radialMixture₂ ν) :=
      integral_congr_ae (Filter.Eventually.of_forall (fun y => by
        rw [laplaceCompanionKernel]; ring))
    have hadd : (∫ y, (τ * laplaceKernel τ (rayProbe₂ r) y +
          laplaceKernel τ (rayProbe₂ r) y * ‖rayProbe₂ r - y‖)
          ∂(radialMixture₂ ν))
        = (∫ y, τ * laplaceKernel τ (rayProbe₂ r) y ∂(radialMixture₂ ν))
          + ∫ y, laplaceKernel τ (rayProbe₂ r) y *
              ‖rayProbe₂ r - y‖ ∂(radialMixture₂ ν) :=
      integral_add (hZint.const_mul τ) hSint
    rw [hpt, hadd, integral_const_mul]
  rw [hdecomp]; ring

lemma abs_radialRayD₂_le (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0) {r : ℝ} (hr : 0 < r) :
    |radialRayD₂ τ ν r| ≤ radialRayQ₂ τ ν r +
      (τ / (2 * r)) * (radialRayD₂ τ ν r + r * radialRayZ₂ τ ν r) := by
  have hDint := integrable_axial_rayProbe₂ τ hτ (radialMixture₂ ν) r
  have hQint := integrable_Q_payload_rayProbe₂ τ hτ (radialMixture₂ ν) r
  have hSint := integrable_laplaceKernel_mul_norm_rayProbe₂ τ hτ
    (radialMixture₂ ν) r
  have habs : |radialRayD₂ τ ν r| ≤
      ∫ y, laplaceKernel τ (rayProbe₂ r) y *
        |y 0 - r| ∂(radialMixture₂ ν) := by
    rw [radialRayD₂_eq_integral τ ν r hDint]
    have h1 := norm_integral_le_integral_norm
      (μ := radialMixture₂ ν)
      (f := fun y : EuclideanSpace ℝ (Fin 2) =>
        laplaceKernel τ (rayProbe₂ r) y * (y 0 - r))
    simp only [Real.norm_eq_abs] at h1
    refine h1.trans (le_of_eq (integral_congr_ae
      (Filter.Eventually.of_forall fun y => ?_)))
    simp only [abs_mul, abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ r y)]
  have hintAbs : Integrable (fun y : EuclideanSpace ℝ (Fin 2) =>
      laplaceKernel τ (rayProbe₂ r) y * |y 0 - r|) (radialMixture₂ ν) := by
    convert hDint.abs using 1
    funext y
    rw [abs_mul, abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ r y)]
  have hAM :
      (∫ y, laplaceKernel τ (rayProbe₂ r) y *
          |y 0 - r| ∂(radialMixture₂ ν)) ≤
        (radialRayQ₂ τ ν r +
          ∫ y, laplaceKernel τ (rayProbe₂ r) y *
            ‖rayProbe₂ r - y‖ ∂(radialMixture₂ ν)) / 2 := by
    calc (∫ y, laplaceKernel τ (rayProbe₂ r) y *
            |y 0 - r| ∂(radialMixture₂ ν))
        ≤ ∫ y, (laplaceKernel τ (rayProbe₂ r) y *
              ((y 0 - r) ^ 2 / ‖rayProbe₂ r - y‖) +
            laplaceKernel τ (rayProbe₂ r) y * ‖rayProbe₂ r - y‖) / 2
            ∂(radialMixture₂ ν) := by
          refine integral_mono hintAbs ((hQint.add hSint).div_const 2)
            (fun y => ?_)
          have h := two_mul_laplace_abs_first_rayProbe₂_le τ r y
          linarith
      _ = (radialRayQ₂ τ ν r +
          ∫ y, laplaceKernel τ (rayProbe₂ r) y *
            ‖rayProbe₂ r - y‖ ∂(radialMixture₂ ν)) / 2 := by
          rw [integral_div, integral_add hQint hSint,
            ← radialRayQ₂_eq_integral τ ν r hQint]
  have hS := integral_laplaceKernel_mul_norm_eq_C₂_sub τ hτ ν r
  have hC := radialRayC₂_eq_closure τ hτ ν hsupp hr
  rw [hS, hC] at hAM
  have h := habs.trans hAM
  field_simp [hr.ne'] at h ⊢
  linarith

theorem radialRayMDeriv₂_ge_of_le (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0) {r : ℝ} (hr : 0 < r)
    (hm : radialRayM₂ τ ν r ≤ r) :
    -(2 : ℝ) ≤ radialRayMDeriv₂ τ ν r := by
  set Z := radialRayZ₂ τ ν r with hZ
  set D := radialRayD₂ τ ν r with hD
  set Q := radialRayQ₂ τ ν r with hQ
  set Zd := radialRayZd₂ τ ν r with hZd
  have hZpos : 0 < Z := radialRayZ₂_pos τ hτ ν r
  have hDle : D ≤ r * Z := by
    rw [radialRayM₂, div_le_iff₀ hZpos] at hm; exact hm
  have hDabs := abs_radialRayD₂_le τ hτ ν hsupp hr
  have hDabs2 : |D| ≤ Q + τ * Z := by
    have hcoef : 0 ≤ τ / (2 * r) :=
      div_nonneg hτ.le (by positivity)
    have hinside : D + r * Z ≤ 2 * r * Z := by linarith
    have hmul := mul_le_mul_of_nonneg_left hinside hcoef
    have hcalc : τ / (2 * r) * (2 * r * Z) = τ * Z := by field_simp [hr.ne']
    rw [← hZ, ← hD, ← hQ] at hDabs
    linarith
  have hZdabs : |Zd| ≤ Z := by
    simpa [hZ, hZd] using abs_radialRayZd₂_le τ hτ ν r
  have hDZd : D * Zd ≤ |D| * Z := by
    calc D * Zd ≤ |D * Zd| := le_abs_self _
      _ = |D| * |Zd| := abs_mul _ _
      _ ≤ |D| * Z := mul_le_mul_of_nonneg_left hZdabs (abs_nonneg _)
  have hcov := radialRayMDeriv₂_cov τ hτ ν r
  rw [← hZ, ← hD, ← hQ, ← hZd] at hcov
  have hlower : -(τ) * Z ^ 2 ≤ Q * Z - D * Zd := by
    have htmp : D * Zd ≤ (Q + τ * Z) * Z :=
      hDZd.trans (mul_le_mul_of_nonneg_right hDabs2 hZpos.le)
    nlinarith [hZpos]
  have hkey : -(τ) * Z ^ 2 ≤ τ * (radialRayMDeriv₂ τ ν r + 1) * Z ^ 2 := by
    rw [hcov]; exact hlower
  have hZsq : 0 < Z ^ 2 := by positivity
  by_contra hcon
  have hlt : radialRayMDeriv₂ τ ν r + 1 < -1 := by linarith
  nlinarith [mul_pos hτ hZsq, hkey, hlt]

theorem radialRayMDeriv₂_ge (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    (hslack : RadialSlack₂ τ ν) {r : ℝ} (hr : 0 < r) :
    -(2 : ℝ) ≤ radialRayMDeriv₂ τ ν r := by
  rcases le_or_gt (radialRayM₂ τ ν r) r with hm | hm
  · exact radialRayMDeriv₂_ge_of_le τ hτ ν hsupp hr hm
  · exact hslack r hr hm

/-! ## Continuity lemmas -/

lemma continuous_rayProbe₂_sub_norm (y : EuclideanSpace ℝ (Fin 2)) :
    Continuous fun r : ℝ => ‖rayProbe₂ r - y‖ := by
  have hfe : (fun r : ℝ => rayProbe₂ r - y)
      = fun r => r • circleChart 0 - y := by
    funext r; rw [rayProbe₂_eq_smul]
  rw [show (fun r : ℝ => ‖rayProbe₂ r - y‖)
      = fun r => ‖r • circleChart 0 - y‖ from by funext r; rw [rayProbe₂_eq_smul]]
  fun_prop

lemma continuous_laplaceKernel_rayProbe₂_left (τ : ℝ)
    (y : EuclideanSpace ℝ (Fin 2)) :
    Continuous fun r : ℝ => laplaceKernel τ (rayProbe₂ r) y := by
  simp only [laplaceKernel]
  exact Real.continuous_exp.comp
    ((continuous_rayProbe₂_sub_norm y).const_mul (-(1 / τ)))

lemma continuousAt_radialRayQ₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayQ₂ τ ν) r := by
  have hfe : radialRayQ₂ τ ν = fun x =>
      ∫ y, laplaceKernel τ (rayProbe₂ x) y *
        ((y 0 - x) ^ 2 / ‖rayProbe₂ x - y‖) ∂(radialMixture₂ ν) := by
    funext x
    exact radialRayQ₂_eq_integral τ ν x (integrable_Q_payload_rayProbe₂ τ hτ _ x)
  rw [hfe]
  refine continuousAt_of_dominated (bound := fun _ => τ * Real.exp (-1)) ?_ ?_ ?_ ?_
  · refine Filter.Eventually.of_forall fun x => ?_
    have hcoord : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => (y 0 - x) ^ 2) := by fun_prop
    have hnorm : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => ‖rayProbe₂ x - y‖) := by fun_prop
    exact (((continuous_laplaceKernel_rayProbe₂ τ x).measurable).mul
      (hcoord.div hnorm)).aestronglyMeasurable
  · refine Filter.Eventually.of_forall fun x => ae_of_all _ fun y => ?_
    rw [Real.norm_eq_abs]
    exact abs_kernel_mul_first_sq_div_rayProbe₂_le τ hτ x y
  · exact integrable_const _
  · filter_upwards [radialMixture₂_ae_probe_ne ν] with y hy
    have hne := hy r hr
    exact (continuous_laplaceKernel_rayProbe₂_left τ y).continuousAt.mul
      ((by fun_prop : ContinuousAt (fun x : ℝ => (y 0 - x) ^ 2) r).div
        (continuous_rayProbe₂_sub_norm y).continuousAt hne)

lemma continuousAt_radialRayZd₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayZd₂ τ ν) r := by
  have hfe : radialRayZd₂ τ ν = fun x =>
      ∫ y, laplaceKernel τ (rayProbe₂ x) y *
        ((y 0 - x) / ‖rayProbe₂ x - y‖) ∂(radialMixture₂ ν) := by
    funext x
    exact radialRayZd₂_eq_integral τ ν x (integrable_softsign_rayProbe₂ τ hτ _ x)
  rw [hfe]
  refine continuousAt_of_dominated (bound := fun _ => (1 : ℝ)) ?_ ?_ ?_ ?_
  · refine Filter.Eventually.of_forall fun x => ?_
    have hcoord : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => y 0 - x) := by fun_prop
    have hnorm : Measurable
        (fun y : EuclideanSpace ℝ (Fin 2) => ‖rayProbe₂ x - y‖) := by fun_prop
    exact (((continuous_laplaceKernel_rayProbe₂ τ x).measurable).mul
      (hcoord.div hnorm)).aestronglyMeasurable
  · refine Filter.Eventually.of_forall fun x => ae_of_all _ fun y => ?_
    rw [Real.norm_eq_abs, abs_mul,
      abs_of_nonneg (laplaceKernel_rayProbe₂_nonneg τ x y)]
    simpa only [mul_one] using mul_le_mul
      (laplaceKernel_rayProbe₂_le_one τ hτ x y)
      (abs_first_div_rayProbe₂_le_one x y) (abs_nonneg _) zero_le_one
  · exact integrable_const 1
  · filter_upwards [radialMixture₂_ae_probe_ne ν] with y hy
    have hne := hy r hr
    exact (continuous_laplaceKernel_rayProbe₂_left τ y).continuousAt.mul
      ((by fun_prop : ContinuousAt (fun x : ℝ => y 0 - x) r).div
        (continuous_rayProbe₂_sub_norm y).continuousAt hne)

lemma continuousAt_radialRayM₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayM₂ τ ν) r :=
  (hasDerivAt_radialRayD₂ τ hτ ν hr).continuousAt.div
    (hasDerivAt_radialRayZ₂ τ hτ ν hr).continuousAt
    (radialRayZ₂_pos τ hτ ν r).ne'

lemma continuousAt_radialRayMDeriv₂ (τ : ℝ) (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayMDeriv₂ τ ν) r := by
  have hZ := (hasDerivAt_radialRayZ₂ τ hτ ν hr).continuousAt
  have hD := (hasDerivAt_radialRayD₂ τ hτ ν hr).continuousAt
  have hQ := continuousAt_radialRayQ₂ τ hτ ν hr
  have hZd := continuousAt_radialRayZd₂ τ hτ ν hr
  exact ((((hQ.const_mul _).sub hZ).mul hZ).sub
    (hD.mul (hZd.const_mul _))).div (hZ.pow 2)
      (pow_ne_zero 2 (radialRayZ₂_pos τ hτ ν r).ne')

/-! ## Zero-drift reduction -/

lemma radialRayD₂_eq_weightedDisplacementCoord (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayD₂ τ ν r =
      (∫ y, laplaceWeightedDisplacement τ (rayProbe₂ r) y
        ∂(radialMixture₂ ν)) 0 := by
  have hFint : Integrable (fun y => laplaceWeightedDisplacement τ
      (rayProbe₂ r) y) (radialMixture₂ ν) :=
    laplaceWeightedDisplacement_integrable τ hτ (radialMixture₂ ν) (rayProbe₂ r)
  have hproj := (EuclideanSpace.proj (0 : Fin 2)).integral_comp_comm
    (μ := radialMixture₂ ν) hFint
  have hcoord :
      (∫ y, laplaceWeightedDisplacement τ (rayProbe₂ r) y
        ∂(radialMixture₂ ν)) 0
      = ∫ y, (laplaceWeightedDisplacement τ (rayProbe₂ r) y) 0
        ∂(radialMixture₂ ν) := by
    simpa [EuclideanSpace.coe_proj] using hproj.symm
  rw [hcoord, radialRayD₂_eq_integral τ ν r (integrable_axial_rayProbe₂ τ hτ _ r)]
  refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
  simp only [laplaceWeightedDisplacement, PiLp.smul_apply, PiLp.sub_apply,
    rayProbe₂_apply_zero, smul_eq_mul]

lemma zeroDrift_radialRay₂_meanShift_eq (τ : ℝ) (νp νq : Measure ℝ)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq)) (r : ℝ) :
    meanShift (laplaceKernel τ) (radialMixture₂ νp) (rayProbe₂ r) =
      meanShift (laplaceKernel τ) (radialMixture₂ νq) (rayProbe₂ r) := by
  have h := hzero (rayProbe₂ r)
  rwa [meanShiftDrift, sub_eq_zero] at h

lemma zeroDrift_radialRay₂_D_mul_Z (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq)) (r : ℝ) :
    radialRayD₂ τ νp r * radialRayZ₂ τ νq r =
      radialRayD₂ τ νq r * radialRayZ₂ τ νp r := by
  have h := congrArg (fun v : EuclideanSpace ℝ (Fin 2) => v 0)
    (zeroDrift_radialRay₂_meanShift_eq τ νp νq hzero r)
  simp only [meanShift, PiLp.smul_apply, smul_eq_mul] at h
  have hip :
      (∫ y, laplaceKernel τ (rayProbe₂ r) y • (y - rayProbe₂ r)
          ∂(radialMixture₂ νp)) =
        ∫ y, laplaceWeightedDisplacement τ (rayProbe₂ r) y
          ∂(radialMixture₂ νp) := rfl
  have hiq :
      (∫ y, laplaceKernel τ (rayProbe₂ r) y • (y - rayProbe₂ r)
          ∂(radialMixture₂ νq)) =
        ∫ y, laplaceWeightedDisplacement τ (rayProbe₂ r) y
          ∂(radialMixture₂ νq) := rfl
  rw [hip, hiq,
    ← radialRayZ₂_eq_kernelNormalizer τ hτ νp r,
    ← radialRayZ₂_eq_kernelNormalizer τ hτ νq r,
    ← radialRayD₂_eq_weightedDisplacementCoord τ hτ νp r,
    ← radialRayD₂_eq_weightedDisplacementCoord τ hτ νq r] at h
  have hZp := radialRayZ₂_pos τ hτ νp r
  have hZq := radialRayZ₂_pos τ hτ νq r
  field_simp [hZp.ne', hZq.ne'] at h
  linarith

lemma zeroDrift_radialRay₂_M_eq (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq)) (r : ℝ) :
    radialRayM₂ τ νp r = radialRayM₂ τ νq r := by
  have hp := radialRayZ₂_pos τ hτ νp r
  have hq := radialRayZ₂_pos τ hτ νq r
  rw [radialRayM₂, radialRayM₂, div_eq_div_iff hp.ne' hq.ne']
  exact zeroDrift_radialRay₂_D_mul_Z τ hτ νp νq hzero r

lemma zeroDrift_radialRay₂_D_deriv_bridge (τ : ℝ) (hτ : 0 < τ)
    (νp ν : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure ν]
    (hM : ∀ x : ℝ, radialRayM₂ τ νp x = radialRayM₂ τ ν x) {r : ℝ} (hr : 0 < r) :
    (1 / τ) * radialRayQ₂ τ ν r - radialRayZ₂ τ ν r =
      radialRayMDeriv₂ τ νp r * radialRayZ₂ τ ν r +
        radialRayM₂ τ νp r * ((1 / τ) * radialRayZd₂ τ ν r) := by
  have hDν := hasDerivAt_radialRayD₂ τ hτ ν hr
  have hprod := (hasDerivAt_radialRayM₂ τ hτ νp hr).mul
    (hasDerivAt_radialRayZ₂ τ hτ ν hr)
  have heq : (fun x => radialRayM₂ τ νp x * radialRayZ₂ τ ν x)
      =ᶠ[𝓝 r] radialRayD₂ τ ν := by
    refine Filter.Eventually.of_forall fun x => ?_
    change radialRayM₂ τ νp x * radialRayZ₂ τ ν x = radialRayD₂ τ ν x
    rw [hM x, radialRayM₂, div_mul_cancel₀]
    exact (radialRayZ₂_pos τ hτ ν x).ne'
  exact ((hprod.congr_of_eventuallyEq heq.symm).unique hDν).symm

/-! ## The `n = 2` Abel system -/

noncomputable def radialRayW₂ (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  (1 / τ) * (radialRayZd₂ τ νp r * radialRayZ₂ τ νq r -
    radialRayZd₂ τ νq r * radialRayZ₂ τ νp r)

noncomputable def radialRayV₂ (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  r * radialRayW₂ τ νp νq r

noncomputable def radialRayK₂ (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  radialRayC₂ τ νp r * radialRayZ₂ τ νq r -
    radialRayC₂ τ νq r * radialRayZ₂ τ νp r

noncomputable def radialRayKhat₂ (τ : ℝ) (νp νq : Measure ℝ) (r : ℝ) : ℝ :=
  r * radialRayK₂ τ νp νq r

theorem radialRayKhat₂_eq_M_mul_V (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq)) {r : ℝ} (hr : 0 < r) :
    radialRayKhat₂ τ νp νq r =
      τ * radialRayM₂ τ νp r * radialRayV₂ τ νp νq r := by
  have hMq : ∀ x, radialRayM₂ τ νp x = radialRayM₂ τ νq x :=
    fun x => zeroDrift_radialRay₂_M_eq τ hτ νp νq hzero x
  have hp := zeroDrift_radialRay₂_D_deriv_bridge τ hτ νp νp (fun _ => rfl) hr
  have hq := zeroDrift_radialRay₂_D_deriv_bridge τ hτ νp νq hMq hr
  have hCp := radialRayC₂_eq_closure τ hτ νp hsp hr
  have hCq := radialRayC₂_eq_closure τ hτ νq hsq hr
  have hp' : radialRayQ₂ τ νp r =
      τ * radialRayZ₂ τ νp r +
        τ * radialRayMDeriv₂ τ νp r * radialRayZ₂ τ νp r +
        radialRayM₂ τ νp r * radialRayZd₂ τ νp r := by
    field_simp [hτ.ne'] at hp; linarith
  have hq' : radialRayQ₂ τ νq r =
      τ * radialRayZ₂ τ νq r +
        τ * radialRayMDeriv₂ τ νp r * radialRayZ₂ τ νq r +
        radialRayM₂ τ νp r * radialRayZd₂ τ νq r := by
    field_simp [hτ.ne'] at hq; linarith
  have hDp : radialRayD₂ τ νp r =
      radialRayM₂ τ νp r * radialRayZ₂ τ νp r := by
    rw [radialRayM₂, div_mul_cancel₀]; exact (radialRayZ₂_pos τ hτ νp r).ne'
  have hDq : radialRayD₂ τ νq r =
      radialRayM₂ τ νp r * radialRayZ₂ τ νq r := by
    calc radialRayD₂ τ νq r =
          radialRayM₂ τ νq r * radialRayZ₂ τ νq r := by
            rw [radialRayM₂, div_mul_cancel₀]; exact (radialRayZ₂_pos τ hτ νq r).ne'
      _ = _ := by rw [← zeroDrift_radialRay₂_M_eq τ hτ νp νq hzero r]
  have hbase : radialRayK₂ τ νp νq r =
      τ * radialRayM₂ τ νp r * radialRayW₂ τ νp νq r := by
    rw [radialRayK₂, radialRayW₂, hCp, hCq, hp', hq', hDp, hDq]
    field_simp [hτ.ne', hr.ne']; ring
  rw [radialRayKhat₂, radialRayV₂, hbase]; ring

theorem hasDerivAt_radialRayKhat₂ (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₂ νp) (radialMixture₂ νq)) {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayKhat₂ τ νp νq)
      (-(τ * (radialRayMDeriv₂ τ νp r + 3)) *
        radialRayV₂ τ νp νq r) r := by
  have hCp := hasDerivAt_radialRayC₂ τ hτ νp hr
  have hCq := hasDerivAt_radialRayC₂ τ hτ νq hr
  have hZp := hasDerivAt_radialRayZ₂ τ hτ νp hr
  have hZq := hasDerivAt_radialRayZ₂ τ hτ νq hr
  have hKraw := (hCp.mul hZq).sub (hCq.mul hZp)
  have hK : HasDerivAt (radialRayK₂ τ νp νq)
      ((1 / τ) * radialRayD₂ τ νp r * radialRayZ₂ τ νq r +
          radialRayC₂ τ νp r * ((1 / τ) * radialRayZd₂ τ νq r) -
        ((1 / τ) * radialRayD₂ τ νq r * radialRayZ₂ τ νp r +
          radialRayC₂ τ νq r * ((1 / τ) * radialRayZd₂ τ νp r))) r := by
    change HasDerivAt (fun x =>
      radialRayC₂ τ νp x * radialRayZ₂ τ νq x -
        radialRayC₂ τ νq x * radialRayZ₂ τ νp x) _ r
    exact hKraw
  have hCpcl := radialRayC₂_eq_closure τ hτ νp hsp hr
  have hCqcl := radialRayC₂_eq_closure τ hτ νq hsq hr
  have hMp := zeroDrift_radialRay₂_M_eq τ hτ νp νq hzero r
  have hbp := zeroDrift_radialRay₂_D_deriv_bridge τ hτ νp νp (fun _ => rfl) hr
  have hbq := zeroDrift_radialRay₂_D_deriv_bridge τ hτ νp νq
    (fun x => zeroDrift_radialRay₂_M_eq τ hτ νp νq hzero x) hr
  have hp' : radialRayQ₂ τ νp r =
      τ * radialRayZ₂ τ νp r +
        τ * radialRayMDeriv₂ τ νp r * radialRayZ₂ τ νp r +
        radialRayM₂ τ νp r * radialRayZd₂ τ νp r := by
    field_simp [hτ.ne'] at hbp; linarith
  have hq' : radialRayQ₂ τ νq r =
      τ * radialRayZ₂ τ νq r +
        τ * radialRayMDeriv₂ τ νp r * radialRayZ₂ τ νq r +
        radialRayM₂ τ νp r * radialRayZd₂ τ νq r := by
    field_simp [hτ.ne'] at hbq; linarith
  have hDp : radialRayD₂ τ νp r =
      radialRayM₂ τ νp r * radialRayZ₂ τ νp r := by
    rw [radialRayM₂, div_mul_cancel₀]; exact (radialRayZ₂_pos τ hτ νp r).ne'
  have hDq : radialRayD₂ τ νq r =
      radialRayM₂ τ νp r * radialRayZ₂ τ νq r := by
    calc radialRayD₂ τ νq r =
          radialRayM₂ τ νq r * radialRayZ₂ τ νq r := by
            rw [radialRayM₂, div_mul_cancel₀]; exact (radialRayZ₂_pos τ hτ νq r).ne'
      _ = _ := by rw [← hMp]
  have hKval :
      (1 / τ) * radialRayD₂ τ νp r * radialRayZ₂ τ νq r +
          radialRayC₂ τ νp r * ((1 / τ) * radialRayZd₂ τ νq r) -
        ((1 / τ) * radialRayD₂ τ νq r * radialRayZ₂ τ νp r +
          radialRayC₂ τ νq r * ((1 / τ) * radialRayZd₂ τ νp r)) =
      -(τ * (radialRayMDeriv₂ τ νp r + 3 +
          radialRayM₂ τ νp r / r)) * radialRayW₂ τ νp νq r := by
    rw [hCpcl, hCqcl, hp', hq', hDp, hDq, radialRayW₂]
    field_simp [hτ.ne', hr.ne']; ring
  have hK' := hK.congr_deriv hKval
  have hprod := (hasDerivAt_id r).mul hK'
  have hval :
      1 * radialRayK₂ τ νp νq r +
        r * (-(τ * (radialRayMDeriv₂ τ νp r + 3 +
          radialRayM₂ τ νp r / r)) * radialRayW₂ τ νp νq r) =
      -(τ * (radialRayMDeriv₂ τ νp r + 3)) * radialRayV₂ τ νp νq r := by
    have hKb : radialRayK₂ τ νp νq r =
        τ * radialRayM₂ τ νp r * radialRayW₂ τ νp νq r := by
      have hKhat := radialRayKhat₂_eq_M_mul_V τ hτ νp νq hsp hsq hzero hr
      rw [radialRayKhat₂, radialRayV₂] at hKhat
      have hr0 : r ≠ 0 := hr.ne'
      field_simp [hr0] at hKhat ⊢
      linarith [hKhat]
    rw [hKb, radialRayV₂]
    field_simp [hr.ne']; ring
  have hfinal := hprod.congr_deriv hval
  exact hfinal

end DriftingIdentifiability
