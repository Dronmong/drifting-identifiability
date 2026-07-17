import DriftingIdentifiability.LaplaceRadialRayN

/-!
# General-n radial shell integrability

This is the analytic discharge layer for G1.  The shell closure in
`LaplaceRadialRayN` was deliberately stated with its exact integrability
requirements.  Here we prove the three *zonal-variable* requirements from the
Laplace exponential bound.  No radial-moment assumption is needed at this
stage: the factors `d * exp (-d / tau)` are uniformly bounded.
-/

open MeasureTheory Filter Topology Set

namespace DriftingIdentifiability

open Paper

/-! ## The axial quotient has the same elementary bound as the tangential one -/

/-- On a shell, the axial square-over-distance quotient is at most the
distance.  The zero-distance convention is the ordinary Lean division
convention, so the collision case is included. -/
lemma shellAxial_sq_div_shellDist_le {r s u : ℝ} (hu : u ^ 2 ≤ 1) :
    shellAxial r s u ^ 2 / shellDist r s u ≤ shellDist r s u := by
  rcases eq_or_lt_of_le (shellDist_nonneg r s u) with hzero | hpos
  · rw [← hzero, div_zero]
  · rw [div_le_iff₀ hpos]
    have hsq : shellDist r s u ^ 2 =
        shellAxial r s u ^ 2 + shellRhoSq s u :=
      shellDist_sq_eq_axial_add_rho hu r s
    nlinarith [shellRhoSq_nonneg hu s]

lemma shellAxial_sq_div_shellDist_nonneg {r s u : ℝ} :
    0 ≤ shellAxial r s u ^ 2 / shellDist r s u :=
  div_nonneg (sq_nonneg _) (shellDist_nonneg r s u)

/-! ## Zonal-variable integrability -/

private lemma shell_exp_dist_bound {τ r s u : ℝ} (hτ : 0 < τ) :
    shellDist r s u * Real.exp (-(1 / τ) * shellDist r s u)
      ≤ τ * Real.exp (-1) := by
  have h := mul_exp_neg_div_le hτ (shellDist_nonneg r s u)
  have heq : -(1 / τ) * shellDist r s u = -shellDist r s u / τ := by
    field_simp [hτ.ne']
  rw [heq]
  exact h

/-- The shell normalizer integrand is integrable on the compact zonal interval. -/
lemma integrableOn_shellKernelN {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ}
    (hτ : 0 < τ) :
    IntegrableOn (fun u : ℝ => zonalWeight n u *
      Real.exp (-(1 / τ) * shellDist r s u)) (Ioc (-1 : ℝ) 1) := by
  refine Measure.integrableOn_of_bounded (M := 1) measure_Ioc_lt_top.ne ?_ ?_
  · exact ((continuous_zonalWeight hn).measurable.mul
      (Real.continuous_exp.comp
        ((continuous_shellDist_u r s).const_mul (-(1 / τ)))).measurable).aestronglyMeasurable
  · filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
    have hW0 : 0 ≤ zonalWeight n u := zonalWeight_nonneg hu2
    have hW1 : zonalWeight n u ≤ 1 := zonalWeight_le_one hn hu2
    have hE0 : 0 ≤ Real.exp (-(1 / τ) * shellDist r s u) := (Real.exp_pos _).le
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg hW0, abs_of_nonneg hE0]
    calc zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)
        ≤ 1 * 1 := mul_le_mul hW1 (exp_shellDist_le_one hτ r s u) hE0 zero_le_one
      _ = 1 := by norm_num

/-- The axial square-over-distance shell integrand is integrable.  The
uniform bound is the standard maximum of `d exp (-d/tau)`. -/
lemma integrableOn_shellAxialPayloadN {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ}
    (hτ : 0 < τ) :
    IntegrableOn (fun u : ℝ => zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) *
        (shellAxial r s u ^ 2 / shellDist r s u))) (Ioc (-1 : ℝ) 1) := by
  refine Measure.integrableOn_of_bounded (M := τ * Real.exp (-1))
    measure_Ioc_lt_top.ne ?_ ?_
  · have hA : Measurable fun u : ℝ => shellAxial r s u ^ 2 := by
      unfold shellAxial
      fun_prop
    exact (((continuous_zonalWeight hn).measurable).mul
      (((Real.continuous_exp.comp
        ((continuous_shellDist_u r s).const_mul (-(1 / τ)))).measurable).mul
          (hA.div (continuous_shellDist_u r s).measurable))).aestronglyMeasurable
  · filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
    have hW0 : 0 ≤ zonalWeight n u := zonalWeight_nonneg hu2
    have hW1 : zonalWeight n u ≤ 1 := zonalWeight_le_one hn hu2
    have hE0 : 0 ≤ Real.exp (-(1 / τ) * shellDist r s u) := (Real.exp_pos _).le
    have hA0 : 0 ≤ shellAxial r s u ^ 2 / shellDist r s u :=
      shellAxial_sq_div_shellDist_nonneg
    have hAd : shellAxial r s u ^ 2 / shellDist r s u ≤ shellDist r s u :=
      shellAxial_sq_div_shellDist_le hu2
    have hEd : Real.exp (-(1 / τ) * shellDist r s u) *
        (shellAxial r s u ^ 2 / shellDist r s u)
        ≤ τ * Real.exp (-1) := by
      calc
        Real.exp (-(1 / τ) * shellDist r s u) *
            (shellAxial r s u ^ 2 / shellDist r s u)
          ≤ Real.exp (-(1 / τ) * shellDist r s u) * shellDist r s u :=
            mul_le_mul_of_nonneg_left hAd hE0
        _ = shellDist r s u * Real.exp (-(1 / τ) * shellDist r s u) := by ring
        _ ≤ τ * Real.exp (-1) := shell_exp_dist_bound hτ
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg hW0,
      abs_mul, abs_of_nonneg hE0, abs_of_nonneg hA0]
    calc zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellAxial r s u ^ 2 / shellDist r s u))
        ≤ 1 * (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellAxial r s u ^ 2 / shellDist r s u)) :=
          mul_le_mul_of_nonneg_right hW1 (mul_nonneg hE0 hA0)
      _ = Real.exp (-(1 / τ) * shellDist r s u) *
            (shellAxial r s u ^ 2 / shellDist r s u) := by ring
      _ ≤ τ * Real.exp (-1) := hEd

/-- The tangential square-over-distance shell integrand is integrable, with
the same uniform exponential bound. -/
lemma integrableOn_shellRhoPayloadN {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ}
    (hτ : 0 < τ) :
    IntegrableOn (fun u : ℝ => zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) *
        (shellRhoSq s u / shellDist r s u))) (Ioc (-1 : ℝ) 1) := by
  refine Measure.integrableOn_of_bounded (M := τ * Real.exp (-1))
    measure_Ioc_lt_top.ne ?_ ?_
  · have hR : Measurable fun u : ℝ => shellRhoSq s u := by
      unfold shellRhoSq
      fun_prop
    exact (((continuous_zonalWeight hn).measurable).mul
      (((Real.continuous_exp.comp
        ((continuous_shellDist_u r s).const_mul (-(1 / τ)))).measurable).mul
          (hR.div (continuous_shellDist_u r s).measurable))).aestronglyMeasurable
  · filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
    have hW0 : 0 ≤ zonalWeight n u := zonalWeight_nonneg hu2
    have hW1 : zonalWeight n u ≤ 1 := zonalWeight_le_one hn hu2
    have hE0 : 0 ≤ Real.exp (-(1 / τ) * shellDist r s u) := (Real.exp_pos _).le
    have hR0 : 0 ≤ shellRhoSq s u / shellDist r s u :=
      shellRhoSq_div_shellDist_nonneg hu2 r
    have hRd : shellRhoSq s u / shellDist r s u ≤ shellDist r s u :=
      shellRhoSq_div_shellDist_le hu2 r
    have hEd : Real.exp (-(1 / τ) * shellDist r s u) *
        (shellRhoSq s u / shellDist r s u)
        ≤ τ * Real.exp (-1) := by
      calc
        Real.exp (-(1 / τ) * shellDist r s u) *
            (shellRhoSq s u / shellDist r s u)
          ≤ Real.exp (-(1 / τ) * shellDist r s u) * shellDist r s u :=
            mul_le_mul_of_nonneg_left hRd hE0
        _ = shellDist r s u * Real.exp (-(1 / τ) * shellDist r s u) := by ring
        _ ≤ τ * Real.exp (-1) := shell_exp_dist_bound hτ
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg hW0,
      abs_mul, abs_of_nonneg hE0, abs_of_nonneg hR0]
    calc zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellRhoSq s u / shellDist r s u))
        ≤ 1 * (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellRhoSq s u / shellDist r s u)) :=
          mul_le_mul_of_nonneg_right hW1 (mul_nonneg hE0 hR0)
      _ = Real.exp (-(1 / τ) * shellDist r s u) *
            (shellRhoSq s u / shellDist r s u) := by ring
      _ ≤ τ * Real.exp (-1) := hEd

/-- The general-n ray closure no longer needs its three zonal-variable
integrability hypotheses.  The remaining radial integrability hypotheses are
kept explicit because they are the next, genuinely radial, G1 obligation. -/
theorem radialRayCN_eq_closure_of_shell_zonalIntegrable
    {n : ℕ} (hn : 3 ≤ n) {τ : ℝ} (hτ : 0 < τ) (ν : Measure ℝ)
    (hsupp : ν (Iio 0) = 0) {r : ℝ} (hr : 0 < r)
    (hZ : Integrable (fun s => shellZN n τ r s) ν)
    (hQray : Integrable (fun s => shellQN n τ r s) ν)
    (hPray : Integrable (fun s => shellRhoSqOverDistN n τ r s) ν) :
    radialRayCN n τ ν r = τ * radialRayZN n τ ν r + radialRayQN n τ ν r
      + radialRayRhoSqOverDistN n τ ν r := by
  apply radialRayCN_eq_closure_of_shell hn hτ ν hsupp hr
  · exact Filter.Eventually.of_forall fun s => integrableOn_shellKernelN hn hτ
  · exact Filter.Eventually.of_forall fun s => integrableOn_shellAxialPayloadN hn hτ
  · exact Filter.Eventually.of_forall fun s => integrableOn_shellRhoPayloadN hn hτ
  · exact hZ
  · exact hQray
  · exact hPray

/-! ## Uniform shell-profile bounds and radial integrability -/

private lemma stronglyMeasurable_setIntegral_right {F : ℝ × ℝ → ℝ}
    (hF : StronglyMeasurable F) :
    StronglyMeasurable (fun s : ℝ => ∫ u in Ioc (-1 : ℝ) 1, F (s, u)) := by
  exact hF.integral_prod_right'

private lemma continuous_shellDist_pair (r : ℝ) :
    Continuous (fun z : ℝ × ℝ => shellDist r z.1 z.2) := by
  unfold shellDist
  refine Real.continuous_sqrt.comp ?_
  exact (continuous_const.add (continuous_fst.pow 2)).sub
    ((continuous_const.mul continuous_fst).mul continuous_snd)

private lemma stronglyMeasurable_shellZN (n : ℕ) (hn : 3 ≤ n) (τ r : ℝ) :
    StronglyMeasurable (fun s : ℝ => shellZN n τ r s) := by
  unfold shellZN
  apply StronglyMeasurable.const_mul
  apply stronglyMeasurable_setIntegral_right
    (F := fun z : ℝ × ℝ => zonalWeight n z.2 *
      Real.exp (-(1 / τ) * shellDist r z.1 z.2))
  apply Measurable.stronglyMeasurable
  exact ((continuous_zonalWeight hn).comp continuous_snd).measurable.mul
    (Real.continuous_exp.comp
      ((continuous_shellDist_pair r).const_mul (-(1 / τ)))).measurable

private lemma stronglyMeasurable_shellDN (n : ℕ) (hn : 3 ≤ n) (τ r : ℝ) :
    StronglyMeasurable (fun s : ℝ => shellDN n τ r s) := by
  unfold shellDN
  apply StronglyMeasurable.const_mul
  apply stronglyMeasurable_setIntegral_right
    (F := fun z : ℝ × ℝ => zonalWeight n z.2 *
      (Real.exp (-(1 / τ) * shellDist r z.1 z.2) *
        shellAxial r z.1 z.2))
  apply Measurable.stronglyMeasurable
  exact ((continuous_zonalWeight hn).comp continuous_snd).measurable.mul
    ((Real.continuous_exp.comp
      ((continuous_shellDist_pair r).const_mul (-(1 / τ)))).mul
      (by
        unfold shellAxial
        fun_prop)).measurable

private lemma stronglyMeasurable_shellQN (n : ℕ) (hn : 3 ≤ n) (τ r : ℝ) :
    StronglyMeasurable (fun s : ℝ => shellQN n τ r s) := by
  unfold shellQN
  apply StronglyMeasurable.const_mul
  apply stronglyMeasurable_setIntegral_right
    (F := fun z : ℝ × ℝ => zonalWeight n z.2 *
      (Real.exp (-(1 / τ) * shellDist r z.1 z.2) *
        (shellAxial r z.1 z.2 ^ 2 / shellDist r z.1 z.2)))
  apply Measurable.stronglyMeasurable
  have hA : Measurable fun z : ℝ × ℝ => shellAxial r z.1 z.2 ^ 2 := by
    unfold shellAxial
    fun_prop
  exact ((continuous_zonalWeight hn).comp continuous_snd).measurable.mul
    ((Real.continuous_exp.comp
      ((continuous_shellDist_pair r).const_mul (-(1 / τ)))).measurable.mul
      (hA.div (continuous_shellDist_pair r).measurable))

private lemma stronglyMeasurable_shellRhoSqOverDistN (n : ℕ) (hn : 3 ≤ n)
    (τ r : ℝ) :
    StronglyMeasurable (fun s : ℝ => shellRhoSqOverDistN n τ r s) := by
  unfold shellRhoSqOverDistN
  apply StronglyMeasurable.const_mul
  apply stronglyMeasurable_setIntegral_right
    (F := fun z : ℝ × ℝ => zonalWeight n z.2 *
      (Real.exp (-(1 / τ) * shellDist r z.1 z.2) *
        (shellRhoSq z.1 z.2 / shellDist r z.1 z.2)))
  apply Measurable.stronglyMeasurable
  have hR : Measurable fun z : ℝ × ℝ => shellRhoSq z.1 z.2 := by
    unfold shellRhoSq
    fun_prop
  exact ((continuous_zonalWeight hn).comp continuous_snd).measurable.mul
    ((Real.continuous_exp.comp
      ((continuous_shellDist_pair r).const_mul (-(1 / τ)))).measurable.mul
      (hR.div (continuous_shellDist_pair r).measurable))

private lemma integral_abs_shellKernelN_le_zonalMass {n : ℕ} (hn : 3 ≤ n)
    {τ r s : ℝ} (hτ : 0 < τ) :
    |∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)|
      ≤ zonalMass n := by
  have hF := integrableOn_shellKernelN hn hτ (r := r) (s := s)
  have hW := integrableOn_zonalWeight hn
  have hmono : (fun u : ℝ =>
      |zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)|)
        ≤ᵐ[volume.restrict (Ioc (-1 : ℝ) 1)] zonalWeight n := by
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
    rw [abs_mul, abs_of_nonneg (zonalWeight_nonneg hu2),
      abs_of_pos (Real.exp_pos _)]
    calc zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)
        ≤ zonalWeight n u * 1 :=
          mul_le_mul_of_nonneg_left (exp_shellDist_le_one hτ r s u)
            (zonalWeight_nonneg hu2)
      _ = zonalWeight n u := by ring
  change Integrable (fun u : ℝ => zonalWeight n u *
    Real.exp (-(1 / τ) * shellDist r s u))
      (volume.restrict (Ioc (-1 : ℝ) 1)) at hF
  change Integrable (zonalWeight n) (volume.restrict (Ioc (-1 : ℝ) 1)) at hW
  have hmono' : (fun u : ℝ => ‖zonalWeight n u *
      Real.exp (-(1 / τ) * shellDist r s u)‖)
      ≤ᵐ[volume.restrict (Ioc (-1 : ℝ) 1)] zonalWeight n := by
    simpa only [Real.norm_eq_abs] using hmono
  exact (norm_integral_le_integral_norm _).trans
    (integral_mono_ae hF.norm hW hmono')

private lemma integral_abs_shellAxialPayloadN_le {n : ℕ} (hn : 3 ≤ n)
    {τ r s : ℝ} (hτ : 0 < τ) :
    |∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellAxial r s u ^ 2 / shellDist r s u))|
      ≤ (τ * Real.exp (-1)) * zonalMass n := by
  have hF := integrableOn_shellAxialPayloadN hn hτ (r := r) (s := s)
  have hW := (integrableOn_zonalWeight hn).const_mul (τ * Real.exp (-1))
  have hC0 : 0 ≤ τ * Real.exp (-1) := mul_nonneg hτ.le (Real.exp_pos _).le
  have hmono : (fun u : ℝ =>
      |zonalWeight n u *
        (Real.exp (-(1 / τ) * shellDist r s u) *
          (shellAxial r s u ^ 2 / shellDist r s u))|)
        ≤ᵐ[volume.restrict (Ioc (-1 : ℝ) 1)]
          fun u => (τ * Real.exp (-1)) * zonalWeight n u := by
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
    have hW0 : 0 ≤ zonalWeight n u := zonalWeight_nonneg hu2
    have hW1 : zonalWeight n u ≤ 1 := zonalWeight_le_one hn hu2
    have hE0 : 0 ≤ Real.exp (-(1 / τ) * shellDist r s u) := (Real.exp_pos _).le
    have hA0 : 0 ≤ shellAxial r s u ^ 2 / shellDist r s u :=
      shellAxial_sq_div_shellDist_nonneg
    have hAd := shellAxial_sq_div_shellDist_le (r := r) (s := s) hu2
    have hEd : Real.exp (-(1 / τ) * shellDist r s u) *
        (shellAxial r s u ^ 2 / shellDist r s u) ≤ τ * Real.exp (-1) := by
      calc
        Real.exp (-(1 / τ) * shellDist r s u) *
            (shellAxial r s u ^ 2 / shellDist r s u)
          ≤ Real.exp (-(1 / τ) * shellDist r s u) * shellDist r s u :=
            mul_le_mul_of_nonneg_left hAd hE0
        _ = shellDist r s u * Real.exp (-(1 / τ) * shellDist r s u) := by ring
        _ ≤ τ * Real.exp (-1) := shell_exp_dist_bound hτ
    rw [abs_mul, abs_of_nonneg hW0, abs_mul, abs_of_nonneg hE0,
      abs_of_nonneg hA0]
    calc zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellAxial r s u ^ 2 / shellDist r s u))
        ≤ zonalWeight n u * (τ * Real.exp (-1)) :=
          mul_le_mul_of_nonneg_left hEd hW0
      _ = (τ * Real.exp (-1)) * zonalWeight n u := by ring
  calc
    |∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellAxial r s u ^ 2 / shellDist r s u))|
      ≤ ∫ u in Ioc (-1 : ℝ) 1,
          |zonalWeight n u *
            (Real.exp (-(1 / τ) * shellDist r s u) *
              (shellAxial r s u ^ 2 / shellDist r s u))| := by
        simpa only [Real.norm_eq_abs] using
          (norm_integral_le_integral_norm (fun u : ℝ => zonalWeight n u *
            (Real.exp (-(1 / τ) * shellDist r s u) *
              (shellAxial r s u ^ 2 / shellDist r s u))))
    _ ≤ ∫ u in Ioc (-1 : ℝ) 1, (τ * Real.exp (-1)) * zonalWeight n u :=
        integral_mono_ae hF.abs hW hmono
    _ = (τ * Real.exp (-1)) * zonalMass n := by
      rw [integral_const_mul]
      rfl

private lemma abs_shellAxial_le_shellDist {r s u : ℝ} (hu : u ^ 2 ≤ 1) :
    |shellAxial r s u| ≤ shellDist r s u := by
  have hsq : shellAxial r s u ^ 2 ≤ shellDist r s u ^ 2 := by
    rw [shellDist_sq_eq_axial_add_rho hu r s]
    exact le_add_of_nonneg_right (shellRhoSq_nonneg hu s)
  rw [sq_le_sq, abs_of_nonneg (shellDist_nonneg r s u)] at hsq
  exact hsq

private lemma integral_abs_shellDisplacementPayloadN_le
    {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ} (hτ : 0 < τ) :
    |∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) * shellAxial r s u)|
      ≤ (τ * Real.exp (-1)) * zonalMass n := by
  have hF : IntegrableOn (fun u : ℝ => zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) * shellAxial r s u))
      (Ioc (-1 : ℝ) 1) := by
    exact ((continuous_zonalWeight hn).mul
      ((Real.continuous_exp.comp
        ((continuous_shellDist_u r s).const_mul (-(1 / τ)))).mul
        ((continuous_const.mul continuous_id).sub continuous_const))).integrableOn_Icc.mono_set
          Ioc_subset_Icc_self
  have hW := (integrableOn_zonalWeight hn).const_mul (τ * Real.exp (-1))
  have hmono : (fun u : ℝ =>
      |zonalWeight n u *
        (Real.exp (-(1 / τ) * shellDist r s u) * shellAxial r s u)|)
        ≤ᵐ[volume.restrict (Ioc (-1 : ℝ) 1)]
          fun u => (τ * Real.exp (-1)) * zonalWeight n u := by
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    have hu2 := sq_le_one_of_mem_Ioc hu
    have hW0 := zonalWeight_nonneg (n := n) hu2
    have hA := abs_shellAxial_le_shellDist (r := r) (s := s) hu2
    rw [abs_mul, abs_of_nonneg hW0, abs_mul, abs_of_pos (Real.exp_pos _)]
    calc
      zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) * |shellAxial r s u|)
        ≤ zonalWeight n u *
            (Real.exp (-(1 / τ) * shellDist r s u) * shellDist r s u) :=
          mul_le_mul_of_nonneg_left
            (mul_le_mul_of_nonneg_left hA (Real.exp_pos _).le) hW0
      _ ≤ zonalWeight n u * (τ * Real.exp (-1)) :=
          mul_le_mul_of_nonneg_left (by
            calc
              Real.exp (-(1 / τ) * shellDist r s u) * shellDist r s u
                = shellDist r s u *
                    Real.exp (-(1 / τ) * shellDist r s u) := by ring
              _ ≤ τ * Real.exp (-1) := shell_exp_dist_bound hτ) hW0
      _ = (τ * Real.exp (-1)) * zonalWeight n u := by ring
  calc
    |∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) * shellAxial r s u)|
      ≤ ∫ u in Ioc (-1 : ℝ) 1,
          |zonalWeight n u *
            (Real.exp (-(1 / τ) * shellDist r s u) * shellAxial r s u)| := by
        simpa only [Real.norm_eq_abs] using
          (norm_integral_le_integral_norm (fun u : ℝ => zonalWeight n u *
            (Real.exp (-(1 / τ) * shellDist r s u) * shellAxial r s u)))
    _ ≤ ∫ u in Ioc (-1 : ℝ) 1,
          (τ * Real.exp (-1)) * zonalWeight n u :=
        integral_mono_ae hF.abs hW hmono
    _ = (τ * Real.exp (-1)) * zonalMass n := by
      rw [integral_const_mul]
      rfl

private lemma integral_abs_shellRhoPayloadN_le {n : ℕ} (hn : 3 ≤ n)
    {τ r s : ℝ} (hτ : 0 < τ) :
    |∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellRhoSq s u / shellDist r s u))|
      ≤ (τ * Real.exp (-1)) * zonalMass n := by
  have hF := integrableOn_shellRhoPayloadN hn hτ (r := r) (s := s)
  have hW := (integrableOn_zonalWeight hn).const_mul (τ * Real.exp (-1))
  have hmono : (fun u : ℝ =>
      |zonalWeight n u *
        (Real.exp (-(1 / τ) * shellDist r s u) *
          (shellRhoSq s u / shellDist r s u))|)
        ≤ᵐ[volume.restrict (Ioc (-1 : ℝ) 1)]
          fun u => (τ * Real.exp (-1)) * zonalWeight n u := by
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
    have hW0 : 0 ≤ zonalWeight n u := zonalWeight_nonneg hu2
    have hE0 : 0 ≤ Real.exp (-(1 / τ) * shellDist r s u) := (Real.exp_pos _).le
    have hR0 : 0 ≤ shellRhoSq s u / shellDist r s u :=
      shellRhoSq_div_shellDist_nonneg hu2 r
    have hRd := shellRhoSq_div_shellDist_le (s := s) hu2 r
    have hEd : Real.exp (-(1 / τ) * shellDist r s u) *
        (shellRhoSq s u / shellDist r s u) ≤ τ * Real.exp (-1) := by
      calc
        Real.exp (-(1 / τ) * shellDist r s u) *
            (shellRhoSq s u / shellDist r s u)
          ≤ Real.exp (-(1 / τ) * shellDist r s u) * shellDist r s u :=
            mul_le_mul_of_nonneg_left hRd hE0
        _ = shellDist r s u * Real.exp (-(1 / τ) * shellDist r s u) := by ring
        _ ≤ τ * Real.exp (-1) := shell_exp_dist_bound hτ
    rw [abs_mul, abs_of_nonneg hW0, abs_mul, abs_of_nonneg hE0,
      abs_of_nonneg hR0]
    calc zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellRhoSq s u / shellDist r s u))
        ≤ zonalWeight n u * (τ * Real.exp (-1)) :=
          mul_le_mul_of_nonneg_left hEd hW0
      _ = (τ * Real.exp (-1)) * zonalWeight n u := by ring
  calc
    |∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellRhoSq s u / shellDist r s u))|
      ≤ ∫ u in Ioc (-1 : ℝ) 1,
          |zonalWeight n u *
            (Real.exp (-(1 / τ) * shellDist r s u) *
              (shellRhoSq s u / shellDist r s u))| := by
        simpa only [Real.norm_eq_abs] using
          (norm_integral_le_integral_norm (fun u : ℝ => zonalWeight n u *
            (Real.exp (-(1 / τ) * shellDist r s u) *
              (shellRhoSq s u / shellDist r s u))))
    _ ≤ ∫ u in Ioc (-1 : ℝ) 1, (τ * Real.exp (-1)) * zonalWeight n u :=
        integral_mono_ae hF.abs hW hmono
    _ = (τ * Real.exp (-1)) * zonalMass n := by
      rw [integral_const_mul]
      rfl

lemma abs_shellZN_le_one {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ} (hτ : 0 < τ) :
    |shellZN n τ r s| ≤ 1 := by
  rw [shellZN, abs_mul, abs_of_nonneg (inv_nonneg.mpr (zonalMass_pos hn).le)]
  calc
    (zonalMass n)⁻¹ *
        |∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)|
      ≤ (zonalMass n)⁻¹ * zonalMass n :=
        mul_le_mul_of_nonneg_left
          (integral_abs_shellKernelN_le_zonalMass (r := r) (s := s) hn hτ)
          (inv_nonneg.mpr (zonalMass_pos hn).le)
    _ = 1 := inv_mul_cancel₀ (zonalMass_ne_zero hn)

lemma abs_shellQN_le {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ} (hτ : 0 < τ) :
    |shellQN n τ r s| ≤ τ * Real.exp (-1) := by
  rw [shellQN, abs_mul, abs_of_nonneg (inv_nonneg.mpr (zonalMass_pos hn).le)]
  calc
    (zonalMass n)⁻¹ *
        |∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u *
            (Real.exp (-(1 / τ) * shellDist r s u) *
              (shellAxial r s u ^ 2 / shellDist r s u))|
      ≤ (zonalMass n)⁻¹ * ((τ * Real.exp (-1)) * zonalMass n) :=
        mul_le_mul_of_nonneg_left
          (integral_abs_shellAxialPayloadN_le (r := r) (s := s) hn hτ)
          (inv_nonneg.mpr (zonalMass_pos hn).le)
    _ = τ * Real.exp (-1) := by
      field_simp [zonalMass_ne_zero hn]

lemma abs_shellRhoSqOverDistN_le {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ}
    (hτ : 0 < τ) :
    |shellRhoSqOverDistN n τ r s| ≤ τ * Real.exp (-1) := by
  rw [shellRhoSqOverDistN, abs_mul,
    abs_of_nonneg (inv_nonneg.mpr (zonalMass_pos hn).le)]
  calc
    (zonalMass n)⁻¹ *
        |∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u *
            (Real.exp (-(1 / τ) * shellDist r s u) *
              (shellRhoSq s u / shellDist r s u))|
      ≤ (zonalMass n)⁻¹ * ((τ * Real.exp (-1)) * zonalMass n) :=
        mul_le_mul_of_nonneg_left
          (integral_abs_shellRhoPayloadN_le (r := r) (s := s) hn hτ)
          (inv_nonneg.mpr (zonalMass_pos hn).le)
    _ = τ * Real.exp (-1) := by
      field_simp [zonalMass_ne_zero hn]

lemma abs_shellDN_le {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ} (hτ : 0 < τ) :
    |shellDN n τ r s| ≤ τ * Real.exp (-1) := by
  rw [shellDN, abs_mul,
    abs_of_nonneg (inv_nonneg.mpr (zonalMass_pos hn).le)]
  calc
    (zonalMass n)⁻¹ *
        |∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u *
            (Real.exp (-(1 / τ) * shellDist r s u) * shellAxial r s u)|
      ≤ (zonalMass n)⁻¹ * ((τ * Real.exp (-1)) * zonalMass n) :=
        mul_le_mul_of_nonneg_left
          (integral_abs_shellDisplacementPayloadN_le (r := r) (s := s) hn hτ)
          (inv_nonneg.mpr (zonalMass_pos hn).le)
    _ = τ * Real.exp (-1) := by
      field_simp [zonalMass_ne_zero hn]

lemma integrable_shellZN {n : ℕ} (hn : 3 ≤ n) {τ r : ℝ} (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    Integrable (fun s => shellZN n τ r s) ν :=
  Integrable.of_bound (stronglyMeasurable_shellZN n hn τ r).aestronglyMeasurable 1
    (ae_of_all _ fun s => by
      rw [Real.norm_eq_abs]
      exact abs_shellZN_le_one hn hτ)

lemma integrable_shellDN {n : ℕ} (hn : 3 ≤ n) {τ r : ℝ} (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    Integrable (fun s => shellDN n τ r s) ν :=
  Integrable.of_bound (stronglyMeasurable_shellDN n hn τ r).aestronglyMeasurable
    (τ * Real.exp (-1))
    (ae_of_all _ fun s => by
      rw [Real.norm_eq_abs]
      exact abs_shellDN_le hn hτ)

lemma integrable_shellQN {n : ℕ} (hn : 3 ≤ n) {τ r : ℝ} (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    Integrable (fun s => shellQN n τ r s) ν :=
  Integrable.of_bound (stronglyMeasurable_shellQN n hn τ r).aestronglyMeasurable
    (τ * Real.exp (-1))
    (ae_of_all _ fun s => by
      rw [Real.norm_eq_abs]
      exact abs_shellQN_le hn hτ)

lemma integrable_shellRhoSqOverDistN {n : ℕ} (hn : 3 ≤ n) {τ r : ℝ}
    (hτ : 0 < τ) (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    Integrable (fun s => shellRhoSqOverDistN n τ r s) ν :=
  Integrable.of_bound
    (stronglyMeasurable_shellRhoSqOverDistN n hn τ r).aestronglyMeasurable
    (τ * Real.exp (-1))
    (ae_of_all _ fun s => by
      rw [Real.norm_eq_abs]
      exact abs_shellRhoSqOverDistN_le hn hτ)

/-- **Analytic G1 closure.**  For any radial probability profile, all six
integrability hypotheses in the general-n shell closure follow from the
Laplace exponential bound. -/
theorem radialRayCN_eq_closure_of_probability
    {n : ℕ} (hn : 3 ≤ n) {τ : ℝ} (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0) {r : ℝ} (hr : 0 < r) :
    radialRayCN n τ ν r = τ * radialRayZN n τ ν r + radialRayQN n τ ν r
      + radialRayRhoSqOverDistN n τ ν r := by
  apply radialRayCN_eq_closure_of_shell_zonalIntegrable hn hτ ν hsupp hr
  · exact integrable_shellZN hn hτ ν
  · exact integrable_shellQN hn hτ ν
  · exact integrable_shellRhoSqOverDistN hn hτ ν

end DriftingIdentifiability
