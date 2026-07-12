import DriftingIdentifiability.LaplaceACDensityRegularity

/-!
# L4 asymptotics for the a.c. Laplace converse

This file formalizes the tail/boundedness part of `LaplaceACDerivation.md`.
The main analytic fact is that, under two-sided exponential first moments, the
Laplace tilted mean

`μ_p(x) = ∫ y kτ(x,y) dp(y) / ∫ kτ(x,y) dp(y)`

has finite limits at both tails.  Consequently zeros of the mean-shift ratio
`m(x) = μ_p(x) - x` are confined to a compact interval once the monotonicity
lemma L3 is available.

The local C²-density regularity from L5 is not used here; L4 only needs moment
assumptions and the already-certified positivity of the Laplace normalizer.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

set_option linter.style.multiGoal false

/-! ## Moment package and tail coordinates -/

/-- Two-sided exponential first moments for the one-dimensional Laplace kernel.

The `exp_*` fields are the denominator moments.  The `first_*` fields are the
numerator moments; since these are `Integrable` statements for real-valued
functions, they assert integrability of the absolute value. -/
structure LaplaceTwoSidedExpFirstMoment (τ : ℝ) (p : Measure ℝ) : Prop where
  exp_pos : Integrable (fun y : ℝ => Real.exp (y / τ)) p
  exp_neg : Integrable (fun y : ℝ => Real.exp (-y / τ)) p
  first_pos : Integrable (fun y : ℝ => y * Real.exp (y / τ)) p
  first_neg : Integrable (fun y : ℝ => y * Real.exp (-y / τ)) p

/-- Positive-tail tilted mean limit:
`μ₊ = ∫ y e^{y/τ} dp / ∫ e^{y/τ} dp`. -/
noncomputable def laplaceTiltedMeanLimitAtTop
    (τ : ℝ) (p : Measure ℝ) : ℝ :=
  (∫ y, y * Real.exp (y / τ) ∂p) / (∫ y, Real.exp (y / τ) ∂p)

/-- Negative-tail tilted mean limit:
`μ₋ = ∫ y e^{-y/τ} dp / ∫ e^{-y/τ} dp`. -/
noncomputable def laplaceTiltedMeanLimitAtBot
    (τ : ℝ) (p : Measure ℝ) : ℝ :=
  (∫ y, y * Real.exp (-y / τ) ∂p) / (∫ y, Real.exp (-y / τ) ∂p)

/-- Lower positive first moment cut off at `x`. -/
noncomputable def lowerPosFirstMoment
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iic x, y * Real.exp (y / τ) ∂p

/-- Upper negative first moment cut off at `x`. -/
noncomputable def upperNegFirstMoment
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Ioi x, y * Real.exp (-y / τ) ∂p

/-- The upper-tail denominator term after multiplying the tilted mean by the
right-tail scale `e^{x/τ}`. -/
noncomputable def upperPosScaledExpTail
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Ioi x, Real.exp ((2 * x - y) / τ) ∂p

/-- The upper-tail numerator term after multiplying the tilted mean by the
right-tail scale `e^{x/τ}`. -/
noncomputable def upperPosScaledFirstTail
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Ioi x, y * Real.exp ((2 * x - y) / τ) ∂p

/-- The lower-tail denominator term after multiplying the tilted mean by the
left-tail scale `e^{-x/τ}`. -/
noncomputable def lowerNegScaledExpTail
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iic x, Real.exp ((y - 2 * x) / τ) ∂p

/-- The lower-tail numerator term after multiplying the tilted mean by the
left-tail scale `e^{-x/τ}`. -/
noncomputable def lowerNegScaledFirstTail
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iic x, y * Real.exp ((y - 2 * x) / τ) ∂p

/-! ## Denominator moments are strictly positive for probability measures -/

lemma exp_pos_integral_pos
    (τ : ℝ) (p : Measure ℝ) [IsProbabilityMeasure p]
    (hInt : Integrable (fun y : ℝ => Real.exp (y / τ)) p) :
    0 < ∫ y, Real.exp (y / τ) ∂p := by
  rw [integral_pos_iff_support_of_nonneg
    (fun y : ℝ => (Real.exp_pos (y / τ)).le) hInt]
  have hsupp : Function.support (fun y : ℝ => Real.exp (y / τ)) = Set.univ := by
    ext y
    simp only [Function.mem_support, ne_eq, Set.mem_univ, iff_true]
    exact (Real.exp_pos (y / τ)).ne'
  rw [hsupp, measure_univ]
  exact one_pos

lemma exp_neg_integral_pos
    (τ : ℝ) (p : Measure ℝ) [IsProbabilityMeasure p]
    (hInt : Integrable (fun y : ℝ => Real.exp (-y / τ)) p) :
    0 < ∫ y, Real.exp (-y / τ) ∂p := by
  rw [integral_pos_iff_support_of_nonneg
    (fun y : ℝ => (Real.exp_pos (-y / τ)).le) hInt]
  have hsupp : Function.support (fun y : ℝ => Real.exp (-y / τ)) = Set.univ := by
    ext y
    simp only [Function.mem_support, ne_eq, Set.mem_univ, iff_true]
    exact (Real.exp_pos (-y / τ)).ne'
  rw [hsupp, measure_univ]
  exact one_pos

/-! ## Cutoff first-moment limits -/

theorem lowerPosFirstMoment_tendsto_atTop_integral
    (τ : ℝ) (p : Measure ℝ)
    (hInt : Integrable (fun y : ℝ => y * Real.exp (y / τ)) p) :
    Tendsto (fun x => lowerPosFirstMoment τ p x) atTop
      (𝓝 (∫ y, y * Real.exp (y / τ) ∂p)) := by
  unfold lowerPosFirstMoment
  simp_rw [← integral_indicator measurableSet_Iic]
  refine tendsto_integral_filter_of_dominated_convergence
    (μ := p) (l := (atTop : Filter ℝ))
    (bound := fun y : ℝ => ‖y * Real.exp (y / τ)‖)
    ?h_meas ?h_bound hInt.norm ?h_lim
  · exact Eventually.of_forall fun x =>
      ((by fun_prop :
        AEStronglyMeasurable (fun y : ℝ => y * Real.exp (y / τ)) p).indicator
          measurableSet_Iic)
  · exact Eventually.of_forall fun x => ae_of_all p fun y => by
      by_cases hy : y ≤ x
      · rw [Set.indicator_of_mem (show y ∈ Set.Iic x by simpa using hy)
            (fun y : ℝ => y * Real.exp (y / τ))]
      · rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hy)
            (fun y : ℝ => y * Real.exp (y / τ)),
          norm_zero]
        exact norm_nonneg _
  · exact ae_of_all p fun y => by
      refine tendsto_const_nhds.congr' ?_
      filter_upwards [(Ici_mem_atTop y : Set.Ici y ∈ (atTop : Filter ℝ))] with x hx
      rw [Set.indicator_of_mem (show y ∈ Set.Iic x by simpa using hx)
        (fun y : ℝ => y * Real.exp (y / τ))]

theorem upperNegFirstMoment_tendsto_atBot_integral
    (τ : ℝ) (p : Measure ℝ)
    (hInt : Integrable (fun y : ℝ => y * Real.exp (-y / τ)) p) :
    Tendsto (fun x => upperNegFirstMoment τ p x) atBot
      (𝓝 (∫ y, y * Real.exp (-y / τ) ∂p)) := by
  unfold upperNegFirstMoment
  simp_rw [← integral_indicator measurableSet_Ioi]
  refine tendsto_integral_filter_of_dominated_convergence
    (μ := p) (l := (atBot : Filter ℝ))
    (bound := fun y : ℝ => ‖y * Real.exp (-y / τ)‖)
    ?h_meas ?h_bound hInt.norm ?h_lim
  · exact Eventually.of_forall fun x =>
      ((by fun_prop :
        AEStronglyMeasurable (fun y : ℝ => y * Real.exp (-y / τ)) p).indicator
          measurableSet_Ioi)
  · exact Eventually.of_forall fun x => ae_of_all p fun y => by
      by_cases hy : x < y
      · rw [Set.indicator_of_mem (show y ∈ Set.Ioi x by simpa using hy)
            (fun y : ℝ => y * Real.exp (-y / τ))]
      · rw [Set.indicator_of_notMem (show y ∉ Set.Ioi x by simpa using hy)
            (fun y : ℝ => y * Real.exp (-y / τ)),
          norm_zero]
        exact norm_nonneg _
  · exact ae_of_all p fun y => by
      refine tendsto_const_nhds.congr' ?_
      filter_upwards [(Iio_mem_atBot y : Set.Iio y ∈ (atBot : Filter ℝ))] with x hx
      rw [Set.indicator_of_mem (show y ∈ Set.Ioi x by simpa using hx)
        (fun y : ℝ => y * Real.exp (-y / τ))]

/-! ## Scaled wrong-side tails vanish -/

theorem upperPosScaledExpTail_tendsto_atTop_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    (hInt : Integrable (fun y : ℝ => Real.exp (y / τ)) p) :
    Tendsto (fun x => upperPosScaledExpTail τ p x) atTop (𝓝 0) := by
  unfold upperPosScaledExpTail
  simp_rw [← integral_indicator measurableSet_Ioi]
  have hDCT :
      Tendsto
        (fun x => ∫ y,
          (Set.Ioi x).indicator
            (fun y : ℝ => Real.exp ((2 * x - y) / τ)) y ∂p)
        atTop (𝓝 (∫ _ : ℝ, (0 : ℝ) ∂p)) := by
    refine tendsto_integral_filter_of_dominated_convergence
      (μ := p) (l := (atTop : Filter ℝ))
      (bound := fun y : ℝ => Real.exp (y / τ))
      ?h_meas ?h_bound hInt ?h_lim
    · exact Eventually.of_forall fun x =>
        ((by fun_prop :
          AEStronglyMeasurable (fun y : ℝ => Real.exp ((2 * x - y) / τ)) p).indicator
            measurableSet_Ioi)
    · exact Eventually.of_forall fun x => ae_of_all p fun y => by
        by_cases hy : x < y
        · rw [Set.indicator_of_mem (show y ∈ Set.Ioi x by simpa using hy)
              (fun y : ℝ => Real.exp ((2 * x - y) / τ)),
            Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
          exact Real.exp_le_exp.mpr
            (div_le_div_of_nonneg_right (by linarith : 2 * x - y ≤ y) hτ.le)
        · rw [Set.indicator_of_notMem (show y ∉ Set.Ioi x by simpa using hy)
              (fun y : ℝ => Real.exp ((2 * x - y) / τ)),
            norm_zero]
          exact (Real.exp_pos _).le
    · exact ae_of_all p fun y => by
        refine tendsto_nhds_of_eventually_eq ?_
        filter_upwards [(Ici_mem_atTop y : Set.Ici y ∈ (atTop : Filter ℝ))] with x hx
        have hyx : ¬ x < y := not_lt.mpr hx
        rw [Set.indicator_of_notMem (show y ∉ Set.Ioi x by simpa using hyx)
          (fun y : ℝ => Real.exp ((2 * x - y) / τ))]
  simpa using hDCT

theorem upperPosScaledFirstTail_tendsto_atTop_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    (hInt : Integrable (fun y : ℝ => y * Real.exp (y / τ)) p) :
    Tendsto (fun x => upperPosScaledFirstTail τ p x) atTop (𝓝 0) := by
  unfold upperPosScaledFirstTail
  simp_rw [← integral_indicator measurableSet_Ioi]
  have hDCT :
      Tendsto
        (fun x => ∫ y,
          (Set.Ioi x).indicator
            (fun y : ℝ => y * Real.exp ((2 * x - y) / τ)) y ∂p)
        atTop (𝓝 (∫ _ : ℝ, (0 : ℝ) ∂p)) := by
    refine tendsto_integral_filter_of_dominated_convergence
      (μ := p) (l := (atTop : Filter ℝ))
      (bound := fun y : ℝ => ‖y * Real.exp (y / τ)‖)
      ?h_meas ?h_bound hInt.norm ?h_lim
    · exact Eventually.of_forall fun x =>
        ((by fun_prop :
          AEStronglyMeasurable (fun y : ℝ => y * Real.exp ((2 * x - y) / τ)) p).indicator
            measurableSet_Ioi)
    · exact Eventually.of_forall fun x => ae_of_all p fun y => by
        by_cases hy : x < y
        · rw [Set.indicator_of_mem (show y ∈ Set.Ioi x by simpa using hy)
              (fun y : ℝ => y * Real.exp ((2 * x - y) / τ))]
          rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_mul, abs_mul,
            abs_of_pos (Real.exp_pos ((2 * x - y) / τ)),
            abs_of_pos (Real.exp_pos (y / τ))]
          exact mul_le_mul_of_nonneg_left
            (Real.exp_le_exp.mpr
            (div_le_div_of_nonneg_right (by linarith : 2 * x - y ≤ y) hτ.le)
            ) (abs_nonneg y)
        · rw [Set.indicator_of_notMem (show y ∉ Set.Ioi x by simpa using hy)
              (fun y : ℝ => y * Real.exp ((2 * x - y) / τ)),
            norm_zero]
          exact norm_nonneg _
    · exact ae_of_all p fun y => by
        refine tendsto_nhds_of_eventually_eq ?_
        filter_upwards [(Ici_mem_atTop y : Set.Ici y ∈ (atTop : Filter ℝ))] with x hx
        have hyx : ¬ x < y := not_lt.mpr hx
        rw [Set.indicator_of_notMem (show y ∉ Set.Ioi x by simpa using hyx)
          (fun y : ℝ => y * Real.exp ((2 * x - y) / τ))]
  simpa using hDCT

theorem lowerNegScaledExpTail_tendsto_atBot_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    (hInt : Integrable (fun y : ℝ => Real.exp (-y / τ)) p) :
    Tendsto (fun x => lowerNegScaledExpTail τ p x) atBot (𝓝 0) := by
  unfold lowerNegScaledExpTail
  simp_rw [← integral_indicator measurableSet_Iic]
  have hDCT :
      Tendsto
        (fun x => ∫ y,
          (Set.Iic x).indicator
            (fun y : ℝ => Real.exp ((y - 2 * x) / τ)) y ∂p)
        atBot (𝓝 (∫ _ : ℝ, (0 : ℝ) ∂p)) := by
    refine tendsto_integral_filter_of_dominated_convergence
      (μ := p) (l := (atBot : Filter ℝ))
      (bound := fun y : ℝ => Real.exp (-y / τ))
      ?h_meas ?h_bound hInt ?h_lim
    · exact Eventually.of_forall fun x =>
        ((by fun_prop :
          AEStronglyMeasurable (fun y : ℝ => Real.exp ((y - 2 * x) / τ)) p).indicator
            measurableSet_Iic)
    · exact Eventually.of_forall fun x => ae_of_all p fun y => by
        by_cases hy : y ≤ x
        · rw [Set.indicator_of_mem (show y ∈ Set.Iic x by simpa using hy)
              (fun y : ℝ => Real.exp ((y - 2 * x) / τ)),
            Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
          exact Real.exp_le_exp.mpr
            (div_le_div_of_nonneg_right (by linarith : y - 2 * x ≤ -y) hτ.le)
        · rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hy)
              (fun y : ℝ => Real.exp ((y - 2 * x) / τ)),
            norm_zero]
          exact (Real.exp_pos _).le
    · exact ae_of_all p fun y => by
        refine tendsto_nhds_of_eventually_eq ?_
        filter_upwards [(Iio_mem_atBot y : Set.Iio y ∈ (atBot : Filter ℝ))] with x hx
        have hyx : ¬ y ≤ x := not_le.mpr hx
        rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hyx)
          (fun y : ℝ => Real.exp ((y - 2 * x) / τ))]
  simpa using hDCT

theorem lowerNegScaledFirstTail_tendsto_atBot_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    (hInt : Integrable (fun y : ℝ => y * Real.exp (-y / τ)) p) :
    Tendsto (fun x => lowerNegScaledFirstTail τ p x) atBot (𝓝 0) := by
  unfold lowerNegScaledFirstTail
  simp_rw [← integral_indicator measurableSet_Iic]
  have hDCT :
      Tendsto
        (fun x => ∫ y,
          (Set.Iic x).indicator
            (fun y : ℝ => y * Real.exp ((y - 2 * x) / τ)) y ∂p)
        atBot (𝓝 (∫ _ : ℝ, (0 : ℝ) ∂p)) := by
    refine tendsto_integral_filter_of_dominated_convergence
      (μ := p) (l := (atBot : Filter ℝ))
      (bound := fun y : ℝ => ‖y * Real.exp (-y / τ)‖)
      ?h_meas ?h_bound hInt.norm ?h_lim
    · exact Eventually.of_forall fun x =>
        ((by fun_prop :
          AEStronglyMeasurable (fun y : ℝ => y * Real.exp ((y - 2 * x) / τ)) p).indicator
            measurableSet_Iic)
    · exact Eventually.of_forall fun x => ae_of_all p fun y => by
        by_cases hy : y ≤ x
        · rw [Set.indicator_of_mem (show y ∈ Set.Iic x by simpa using hy)
              (fun y : ℝ => y * Real.exp ((y - 2 * x) / τ))]
          rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_mul, abs_mul,
            abs_of_pos (Real.exp_pos ((y - 2 * x) / τ)),
            abs_of_pos (Real.exp_pos (-y / τ))]
          exact mul_le_mul_of_nonneg_left
            (Real.exp_le_exp.mpr
            (div_le_div_of_nonneg_right (by linarith : y - 2 * x ≤ -y) hτ.le)
            ) (abs_nonneg y)
        · rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hy)
              (fun y : ℝ => y * Real.exp ((y - 2 * x) / τ)),
            norm_zero]
          exact norm_nonneg _
    · exact ae_of_all p fun y => by
        refine tendsto_nhds_of_eventually_eq ?_
        filter_upwards [(Iio_mem_atBot y : Set.Iio y ∈ (atBot : Filter ℝ))] with x hx
        have hyx : ¬ y ≤ x := not_le.mpr hx
        rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hyx)
          (fun y : ℝ => y * Real.exp ((y - 2 * x) / τ))]
  simpa using hDCT

/-! ## Scaled formulae for the tilted mean -/

private lemma laplaceKernel_mul_right_scale_Iic
    (τ : ℝ) (hτ : ValidBandwidth τ) {x y : ℝ} (hy : y ≤ x) :
    Real.exp (x / τ) * laplaceKernel τ x y = Real.exp (y / τ) := by
  rw [laplaceKernel_real]
  have hxy : |x - y| = x - y := abs_of_nonneg (sub_nonneg.mpr hy)
  rw [hxy, ← Real.exp_add]
  congr 1
  field_simp [hτ.ne']
  ring

private lemma laplaceKernel_mul_right_scale_Ioi
    (τ : ℝ) (hτ : ValidBandwidth τ) {x y : ℝ} (hy : x < y) :
    Real.exp (x / τ) * laplaceKernel τ x y =
      Real.exp ((2 * x - y) / τ) := by
  rw [laplaceKernel_real]
  have hxy : |x - y| = y - x := by
    rw [abs_of_neg (sub_neg.mpr hy)]
    ring
  rw [hxy, ← Real.exp_add]
  congr 1
  field_simp [hτ.ne']
  ring

private lemma laplaceKernel_mul_left_scale_Iic
    (τ : ℝ) (hτ : ValidBandwidth τ) {x y : ℝ} (hy : y ≤ x) :
    Real.exp (-x / τ) * laplaceKernel τ x y =
      Real.exp ((y - 2 * x) / τ) := by
  rw [laplaceKernel_real]
  have hxy : |x - y| = x - y := abs_of_nonneg (sub_nonneg.mpr hy)
  rw [hxy, ← Real.exp_add]
  congr 1
  field_simp [hτ.ne']
  ring

private lemma laplaceKernel_mul_left_scale_Ioi
    (τ : ℝ) (hτ : ValidBandwidth τ) {x y : ℝ} (hy : x < y) :
    Real.exp (-x / τ) * laplaceKernel τ x y = Real.exp (-y / τ) := by
  rw [laplaceKernel_real]
  have hxy : |x - y| = y - x := by
    rw [abs_of_neg (sub_neg.mpr hy)]
    ring
  rw [hxy, ← Real.exp_add]
  congr 1
  field_simp [hτ.ne']
  ring

private lemma exp_neg_mul_exp_self (z : ℝ) :
    Real.exp (-z) * Real.exp z = 1 := by
  rw [← Real.exp_add]
  ring_nf
  simp

private lemma laplaceKernel_eq_Iic
    (τ : ℝ) (hτ : ValidBandwidth τ) {x y : ℝ} (hy : y ≤ x) :
    laplaceKernel τ x y = Real.exp (-x / τ) * Real.exp (y / τ) := by
  calc
    laplaceKernel τ x y = 1 * laplaceKernel τ x y := by ring
    _ = (Real.exp (-(x / τ)) * Real.exp (x / τ)) * laplaceKernel τ x y := by
          rw [exp_neg_mul_exp_self (x / τ)]
    _ = Real.exp (-x / τ) * (Real.exp (x / τ) * laplaceKernel τ x y) := by ring_nf
    _ = Real.exp (-x / τ) * Real.exp (y / τ) := by
          rw [laplaceKernel_mul_right_scale_Iic τ hτ hy]

private lemma laplaceKernel_eq_Ioi
    (τ : ℝ) (hτ : ValidBandwidth τ) {x y : ℝ} (hy : x < y) :
    laplaceKernel τ x y = Real.exp (x / τ) * Real.exp (-y / τ) := by
  calc
    laplaceKernel τ x y = 1 * laplaceKernel τ x y := by ring
    _ = (Real.exp (x / τ) * Real.exp (-(x / τ))) * laplaceKernel τ x y := by
          rw [show Real.exp (x / τ) * Real.exp (-(x / τ)) = 1 by
            rw [mul_comm, exp_neg_mul_exp_self (x / τ)]]
    _ = Real.exp (x / τ) * (Real.exp (-x / τ) * laplaceKernel τ x y) := by ring_nf
    _ = Real.exp (x / τ) * Real.exp (-y / τ) := by
          rw [laplaceKernel_mul_left_scale_Ioi τ hτ hy]

/-- Right-tail scaled expression for the tilted mean. -/
theorem laplaceTiltedMean_eq_rightScaled
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hmom : LaplaceTwoSidedExpFirstMoment τ p)
    (x : ℝ) :
    laplaceTiltedMean τ p x =
      (lowerPosFirstMoment τ p x + upperPosScaledFirstTail τ p x) /
        (lowerExpMass τ p x + upperPosScaledExpTail τ p x) := by
  have hZpos : 0 < kernelNormalizer (laplaceKernel τ) p x :=
    laplaceKernelNormalizer_pos p τ hτ x
  have hscale_pos : Real.exp (x / τ) ≠ 0 := (Real.exp_pos _).ne'
  have hKint : Integrable (fun y => laplaceKernel τ x y) p :=
    laplaceKernel_integrable p τ hτ x
  have hYKint : Integrable (fun y => laplaceKernel τ x y * y) p := by
    -- The two-sided first moments dominate the two half-line pieces.
    rw [← integrableOn_univ]
    rw [← Iic_union_Ioi (a := x)]
    refine IntegrableOn.union ?_ ?_
    · refine Integrable.mono'
        ((hmom.first_pos.const_mul (Real.exp (-x / τ))).norm.restrict) ?_ ?_
      · unfold laplaceKernel
        fun_prop
      · filter_upwards [ae_restrict_mem measurableSet_Iic] with y hy
        have hyx : y ≤ x := by simpa using hy
        have hEq : laplaceKernel τ x y = Real.exp (-x / τ) * Real.exp (y / τ) := by
          exact laplaceKernel_eq_Iic τ hτ hyx
        rw [hEq]
        rw [show Real.exp (-x / τ) * Real.exp (y / τ) * y =
            Real.exp (-x / τ) * (y * Real.exp (y / τ)) by ring]
    · refine Integrable.mono'
        ((hmom.first_neg.const_mul (Real.exp (x / τ))).norm.restrict) ?_ ?_
      · unfold laplaceKernel
        fun_prop
      · filter_upwards [ae_restrict_mem measurableSet_Ioi] with y hy
        have hxy : x < y := by simpa using hy
        have hEq : laplaceKernel τ x y = Real.exp (x / τ) * Real.exp (-y / τ) := by
          exact laplaceKernel_eq_Ioi τ hτ hxy
        rw [hEq]
        rw [show Real.exp (x / τ) * Real.exp (-y / τ) * y =
            Real.exp (x / τ) * (y * Real.exp (-y / τ)) by ring]
  unfold laplaceTiltedMean
  have hden_split :
      Real.exp (x / τ) * kernelNormalizer (laplaceKernel τ) p x =
        lowerExpMass τ p x + upperPosScaledExpTail τ p x := by
    unfold kernelNormalizer lowerExpMass upperPosScaledExpTail
    have hKint_scaled : Integrable (fun y => Real.exp (x / τ) * laplaceKernel τ x y) p :=
      hKint.const_mul (Real.exp (x / τ))
    rw [← integral_const_mul]
    rw [← integral_add_compl measurableSet_Iic hKint_scaled]
    rw [compl_Iic]
    congr 1
    · apply setIntegral_congr_fun measurableSet_Iic
      intro y hy
      have hyx : y ≤ x := by simpa using hy
      change Real.exp (x / τ) * laplaceKernel τ x y = Real.exp (y / τ)
      exact laplaceKernel_mul_right_scale_Iic τ hτ hyx
    · apply setIntegral_congr_fun measurableSet_Ioi
      intro y hy
      have hxy : x < y := by simpa using hy
      change Real.exp (x / τ) * laplaceKernel τ x y =
        Real.exp ((2 * x - y) / τ)
      exact laplaceKernel_mul_right_scale_Ioi τ hτ hxy
  have hnum_split :
      Real.exp (x / τ) * (∫ y, laplaceKernel τ x y * y ∂p) =
        lowerPosFirstMoment τ p x + upperPosScaledFirstTail τ p x := by
    have hYKint_scaled : Integrable
        (fun y => Real.exp (x / τ) * (laplaceKernel τ x y * y)) p :=
      hYKint.const_mul (Real.exp (x / τ))
    rw [← integral_const_mul]
    rw [← integral_add_compl measurableSet_Iic hYKint_scaled]
    rw [compl_Iic]
    unfold lowerPosFirstMoment upperPosScaledFirstTail
    congr 1
    · apply setIntegral_congr_fun measurableSet_Iic
      intro y hy
      have hyx : y ≤ x := by simpa using hy
      calc
        Real.exp (x / τ) * (laplaceKernel τ x y * y)
            = (Real.exp (x / τ) * laplaceKernel τ x y) * y := by ring
        _ = Real.exp (y / τ) * y := by
              rw [laplaceKernel_mul_right_scale_Iic τ hτ hyx]
        _ = y * Real.exp (y / τ) := by ring
    · apply setIntegral_congr_fun measurableSet_Ioi
      intro y hy
      have hxy : x < y := by simpa using hy
      calc
        Real.exp (x / τ) * (laplaceKernel τ x y * y)
            = (Real.exp (x / τ) * laplaceKernel τ x y) * y := by ring
        _ = Real.exp ((2 * x - y) / τ) * y := by
              rw [laplaceKernel_mul_right_scale_Ioi τ hτ hxy]
        _ = y * Real.exp ((2 * x - y) / τ) := by ring
  have hden_ne : lowerExpMass τ p x + upperPosScaledExpTail τ p x ≠ 0 := by
    rw [← hden_split]
    exact mul_ne_zero hscale_pos hZpos.ne'
  rw [← hnum_split, ← hden_split]
  field_simp [hZpos.ne', hscale_pos]

/-- Left-tail scaled expression for the tilted mean. -/
theorem laplaceTiltedMean_eq_leftScaled
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hmom : LaplaceTwoSidedExpFirstMoment τ p)
    (x : ℝ) :
    laplaceTiltedMean τ p x =
      (lowerNegScaledFirstTail τ p x + upperNegFirstMoment τ p x) /
        (lowerNegScaledExpTail τ p x + upperExpMass τ p x) := by
  have hZpos : 0 < kernelNormalizer (laplaceKernel τ) p x :=
    laplaceKernelNormalizer_pos p τ hτ x
  have hscale_pos : Real.exp (-x / τ) ≠ 0 := (Real.exp_pos _).ne'
  have hKint : Integrable (fun y => laplaceKernel τ x y) p :=
    laplaceKernel_integrable p τ hτ x
  have hYKint : Integrable (fun y => laplaceKernel τ x y * y) p := by
    rw [← integrableOn_univ]
    rw [← Iic_union_Ioi (a := x)]
    refine IntegrableOn.union ?_ ?_
    · refine Integrable.mono'
        ((hmom.first_pos.const_mul (Real.exp (-x / τ))).norm.restrict) ?_ ?_
      · unfold laplaceKernel
        fun_prop
      · filter_upwards [ae_restrict_mem measurableSet_Iic] with y hy
        have hyx : y ≤ x := by simpa using hy
        have hEq : laplaceKernel τ x y = Real.exp (-x / τ) * Real.exp (y / τ) := by
          exact laplaceKernel_eq_Iic τ hτ hyx
        rw [hEq]
        rw [show Real.exp (-x / τ) * Real.exp (y / τ) * y =
            Real.exp (-x / τ) * (y * Real.exp (y / τ)) by ring]
    · refine Integrable.mono'
        ((hmom.first_neg.const_mul (Real.exp (x / τ))).norm.restrict) ?_ ?_
      · unfold laplaceKernel
        fun_prop
      · filter_upwards [ae_restrict_mem measurableSet_Ioi] with y hy
        have hxy : x < y := by simpa using hy
        have hEq : laplaceKernel τ x y = Real.exp (x / τ) * Real.exp (-y / τ) := by
          exact laplaceKernel_eq_Ioi τ hτ hxy
        rw [hEq]
        rw [show Real.exp (x / τ) * Real.exp (-y / τ) * y =
            Real.exp (x / τ) * (y * Real.exp (-y / τ)) by ring]
  unfold laplaceTiltedMean
  have hden_split :
      Real.exp (-x / τ) * kernelNormalizer (laplaceKernel τ) p x =
        lowerNegScaledExpTail τ p x + upperExpMass τ p x := by
    unfold kernelNormalizer lowerNegScaledExpTail upperExpMass
    have hKint_scaled : Integrable (fun y => Real.exp (-x / τ) * laplaceKernel τ x y) p :=
      hKint.const_mul (Real.exp (-x / τ))
    rw [← integral_const_mul]
    rw [← integral_add_compl measurableSet_Iic hKint_scaled]
    rw [compl_Iic]
    congr 1
    · apply setIntegral_congr_fun measurableSet_Iic
      intro y hy
      have hyx : y ≤ x := by simpa using hy
      change Real.exp (-x / τ) * laplaceKernel τ x y =
        Real.exp ((y - 2 * x) / τ)
      exact laplaceKernel_mul_left_scale_Iic τ hτ hyx
    · apply setIntegral_congr_fun measurableSet_Ioi
      intro y hy
      have hxy : x < y := by simpa using hy
      change Real.exp (-x / τ) * laplaceKernel τ x y = Real.exp (-y / τ)
      exact laplaceKernel_mul_left_scale_Ioi τ hτ hxy
  have hnum_split :
      Real.exp (-x / τ) * (∫ y, laplaceKernel τ x y * y ∂p) =
        lowerNegScaledFirstTail τ p x + upperNegFirstMoment τ p x := by
    have hYKint_scaled : Integrable
        (fun y => Real.exp (-x / τ) * (laplaceKernel τ x y * y)) p :=
      hYKint.const_mul (Real.exp (-x / τ))
    rw [← integral_const_mul]
    rw [← integral_add_compl measurableSet_Iic hYKint_scaled]
    rw [compl_Iic]
    unfold lowerNegScaledFirstTail upperNegFirstMoment
    congr 1
    · apply setIntegral_congr_fun measurableSet_Iic
      intro y hy
      have hyx : y ≤ x := by simpa using hy
      calc
        Real.exp (-x / τ) * (laplaceKernel τ x y * y)
            = (Real.exp (-x / τ) * laplaceKernel τ x y) * y := by ring
        _ = Real.exp ((y - 2 * x) / τ) * y := by
              rw [laplaceKernel_mul_left_scale_Iic τ hτ hyx]
        _ = y * Real.exp ((y - 2 * x) / τ) := by ring
    · apply setIntegral_congr_fun measurableSet_Ioi
      intro y hy
      have hxy : x < y := by simpa using hy
      calc
        Real.exp (-x / τ) * (laplaceKernel τ x y * y)
            = (Real.exp (-x / τ) * laplaceKernel τ x y) * y := by ring
        _ = Real.exp (-y / τ) * y := by
              rw [laplaceKernel_mul_left_scale_Ioi τ hτ hxy]
        _ = y * Real.exp (-y / τ) := by ring
  have hden_ne : lowerNegScaledExpTail τ p x + upperExpMass τ p x ≠ 0 := by
    rw [← hden_split]
    exact mul_ne_zero hscale_pos hZpos.ne'
  rw [← hnum_split, ← hden_split]
  field_simp [hZpos.ne', hscale_pos]

/-! ## L4 tail limits and compact zero pinning -/

theorem laplaceTiltedMean_tendsto_atTop
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hmom : LaplaceTwoSidedExpFirstMoment τ p) :
    Tendsto (fun x => laplaceTiltedMean τ p x) atTop
      (𝓝 (laplaceTiltedMeanLimitAtTop τ p)) := by
  have hnum₁ := lowerPosFirstMoment_tendsto_atTop_integral τ p hmom.first_pos
  have hnum₂ := upperPosScaledFirstTail_tendsto_atTop_zero τ hτ p hmom.first_pos
  have hden₁ := lowerExpMass_tendsto_atTop_integral τ p hmom.exp_pos
  have hden₂ := upperPosScaledExpTail_tendsto_atTop_zero τ hτ p hmom.exp_pos
  have hnum := hnum₁.add hnum₂
  have hden := hden₁.add hden₂
  have hden_pos : 0 < ∫ y, Real.exp (y / τ) ∂p :=
    exp_pos_integral_pos τ p hmom.exp_pos
  have hratio :
      Tendsto
        (fun x =>
          (lowerPosFirstMoment τ p x + upperPosScaledFirstTail τ p x) /
            (lowerExpMass τ p x + upperPosScaledExpTail τ p x))
        atTop (𝓝 (laplaceTiltedMeanLimitAtTop τ p)) := by
    unfold laplaceTiltedMeanLimitAtTop
    change Tendsto
      (((fun x =>
          lowerPosFirstMoment τ p x + upperPosScaledFirstTail τ p x) /
        fun x =>
          lowerExpMass τ p x + upperPosScaledExpTail τ p x))
      atTop
      (𝓝 ((∫ y, y * Real.exp (y / τ) ∂p) / (∫ y, Real.exp (y / τ) ∂p)))
    simpa [zero_add, add_zero] using
      hnum.div hden (by simpa [zero_add] using hden_pos.ne')
  refine hratio.congr' ?_
  exact Eventually.of_forall fun x =>
    (laplaceTiltedMean_eq_rightScaled τ hτ p hmom x).symm

theorem laplaceTiltedMean_tendsto_atBot
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hmom : LaplaceTwoSidedExpFirstMoment τ p) :
    Tendsto (fun x => laplaceTiltedMean τ p x) atBot
      (𝓝 (laplaceTiltedMeanLimitAtBot τ p)) := by
  have hnum₁ := lowerNegScaledFirstTail_tendsto_atBot_zero τ hτ p hmom.first_neg
  have hnum₂ := upperNegFirstMoment_tendsto_atBot_integral τ p hmom.first_neg
  have hden₁ := lowerNegScaledExpTail_tendsto_atBot_zero τ hτ p hmom.exp_neg
  have hden₂ := upperExpMass_tendsto_atBot_integral τ p hmom.exp_neg
  have hnum := hnum₁.add hnum₂
  have hden := hden₁.add hden₂
  have hden_pos : 0 < ∫ y, Real.exp (-y / τ) ∂p :=
    exp_neg_integral_pos τ p hmom.exp_neg
  have hratio :
      Tendsto
        (fun x =>
          (lowerNegScaledFirstTail τ p x + upperNegFirstMoment τ p x) /
            (lowerNegScaledExpTail τ p x + upperExpMass τ p x))
        atBot (𝓝 (laplaceTiltedMeanLimitAtBot τ p)) := by
    unfold laplaceTiltedMeanLimitAtBot
    change Tendsto
      (((fun x =>
          lowerNegScaledFirstTail τ p x + upperNegFirstMoment τ p x) /
        fun x =>
          lowerNegScaledExpTail τ p x + upperExpMass τ p x))
      atBot
      (𝓝 ((∫ y, y * Real.exp (-y / τ) ∂p) / (∫ y, Real.exp (-y / τ) ∂p)))
    simpa [zero_add, add_zero] using
      hnum.div hden (by simpa [zero_add] using hden_pos.ne')
  refine hratio.congr' ?_
  exact Eventually.of_forall fun x =>
    (laplaceTiltedMean_eq_leftScaled τ hτ p hmom x).symm

/-- Under zero drift, the common mean-shift ratio `m = D/Z` satisfies
`m(x) = 0` iff the tilted mean is equal to the probe `x`. -/
theorem laplaceMeanShiftRatio_eq_zero_iff_tiltedMean_eq
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ) :
    laplaceMeanShiftRatio τ p x = 0 ↔ laplaceTiltedMean τ p x = x := by
  rw [laplaceTiltedMean_eq_fromDisplacement τ hτ p x]
  unfold laplaceTiltedMeanFromDisplacement laplaceMeanShiftRatio
  constructor <;> intro h
  · linarith
  · linarith

/-- L4 compact pinning, abstracted from the tail limits: if `μ(x) - x` tends to
`-∞` at `+∞` and `+∞` at `-∞`, then all zeros of `μ(x)-x` lie in a bounded
interval.  This purely topological form is the one later sign-change
combinatorics should consume. -/
theorem exists_bounds_for_zeros_of_tendsto_sub
    {μ : ℝ → ℝ}
    (hTop : Tendsto (fun x => μ x - x) atTop atBot)
    (hBot : Tendsto (fun x => μ x - x) atBot atTop) :
    ∃ A B : ℝ, A ≤ B ∧ ∀ x : ℝ, μ x - x = 0 → A ≤ x ∧ x ≤ B := by
  have hRightEventually : ∀ᶠ x in atTop, μ x - x < 0 := by
    exact hTop.eventually (Iio_mem_atBot 0)
  have hLeftEventually : ∀ᶠ x in atBot, 0 < μ x - x := by
    exact hBot.eventually (Ioi_mem_atTop 0)
  rcases eventually_atTop.1 hRightEventually with ⟨B, hB⟩
  rcases eventually_atBot.1 hLeftEventually with ⟨A, hA⟩
  refine ⟨A, max A B, le_max_left _ _, ?_⟩
  intro x hxzero
  constructor
  · by_contra hxA
    have hxlt : x < A := lt_of_not_ge hxA
    have hpos : 0 < μ x - x := hA x (le_of_lt hxlt)
    linarith
  · have hxB : x ≤ B := by
      by_contra hxB
      have hBlt : B < x := lt_of_not_ge hxB
      have hneg : μ x - x < 0 := hB x (le_of_lt hBlt)
      linarith
    exact hxB.trans (le_max_right _ _)

theorem laplaceTiltedMean_sub_tendsto_atTop_atBot
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hmom : LaplaceTwoSidedExpFirstMoment τ p) :
    Tendsto (fun x => laplaceTiltedMean τ p x - x) atTop atBot := by
  have hμ := laplaceTiltedMean_tendsto_atTop τ hτ p hmom
  rw [tendsto_atTop_atBot]
  intro b
  let L := laplaceTiltedMeanLimitAtTop τ p
  have hμlt : ∀ᶠ x in atTop, laplaceTiltedMean τ p x < L + 1 := by
    exact hμ.eventually (Iio_mem_nhds (by linarith : L < L + 1))
  rcases eventually_atTop.1 hμlt with ⟨M, hM⟩
  refine ⟨max M (L + 1 - b), ?_⟩
  intro x hx
  have hxM : M ≤ x := (le_max_left M (L + 1 - b)).trans hx
  have hxB : L + 1 - b ≤ x := (le_max_right M (L + 1 - b)).trans hx
  have hlt := hM x hxM
  linarith

theorem laplaceTiltedMean_sub_tendsto_atBot_atTop
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hmom : LaplaceTwoSidedExpFirstMoment τ p) :
    Tendsto (fun x => laplaceTiltedMean τ p x - x) atBot atTop := by
  have hμ := laplaceTiltedMean_tendsto_atBot τ hτ p hmom
  rw [tendsto_atBot_atTop]
  intro b
  let L := laplaceTiltedMeanLimitAtBot τ p
  have hμgt : ∀ᶠ x in atBot, L - 1 < laplaceTiltedMean τ p x := by
    exact hμ.eventually (Ioi_mem_nhds (by linarith : L - 1 < L))
  rcases eventually_atBot.1 hμgt with ⟨M, hM⟩
  refine ⟨min M (L - 1 - b), ?_⟩
  intro x hx
  have hxM : x ≤ M := hx.trans (min_le_left M (L - 1 - b))
  have hxB : x ≤ L - 1 - b := hx.trans (min_le_right M (L - 1 - b))
  have hgt := hM x hxM
  linarith

/-- **L4 compact zero pinning.**  Under two-sided exponential first moments,
all zeros of the Laplace mean-shift ratio lie in a compact interval. -/
theorem exists_bounds_for_laplaceMeanShiftRatio_zeros
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hmom : LaplaceTwoSidedExpFirstMoment τ p) :
    ∃ A B : ℝ, A ≤ B ∧
      ∀ x : ℝ, laplaceMeanShiftRatio τ p x = 0 → A ≤ x ∧ x ≤ B := by
  rcases exists_bounds_for_zeros_of_tendsto_sub
      (μ := laplaceTiltedMean τ p)
      (laplaceTiltedMean_sub_tendsto_atTop_atBot τ hτ p hmom)
      (laplaceTiltedMean_sub_tendsto_atBot_atTop τ hτ p hmom) with
    ⟨A, B, hAB, hzero⟩
  refine ⟨A, B, hAB, ?_⟩
  intro x hx
  have hμeq : laplaceTiltedMean τ p x - x = 0 := by
    have h := (laplaceMeanShiftRatio_eq_zero_iff_tiltedMean_eq τ hτ p x).mp hx
    linarith
  exact hzero x hμeq

end DriftingIdentifiability
