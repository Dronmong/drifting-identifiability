import DriftingIdentifiability.LaplaceAtomlessConverse

/-!
# The unconditional 1-d Laplace converse (Frontier A)

This file removes the last hypotheses from the atomless converse: zero raw
Laplace mean-shift drift identifies **arbitrary** probability measures on ℝ.
No atomlessness, no moment, no density.

The engine is unchanged — the companion alignment defect
`K := L_p·Z_q − L_q·Z_p` — but everything is phrased with the **one-sided
(right) derivative** of the raw normalizer, which exists for every finite
measure (`hasDerivWithinAt_Ici_laplaceKernelNormalizer`).  The key structural
facts that make atoms harmless:

* `D` and `L` are `C¹` for any probability measure; only `Z` has corners at
  atoms, and its right derivative is the certified `rightDerivCoeff`;
* the mean-shift ratio derivative `laplaceMeanShiftRatioDeriv` and the raw
  Wronskian `laplaceKernelNormalizerWronskian` are already *defined* through
  the right-derivative coefficient, so the identities `K = τ·m·W` and
  `K'⁺ = −τ(m'+2)·W` hold pointwise for arbitrary measures once phrased with
  right derivatives;
* the abstract L6/L8 propagation layer was already stated with right
  derivatives (`HasDerivWithinAt _ (Ici t) t`) and needs no continuity of its
  coefficient — only right-continuity of `c = 2μ'/m` for the FTC primitive,
  which holds because the one-sided masses are càdlàg for every finite
  measure;
* the coefficient lower bound `m' + 1 ≥ 0` is the *moment-free* one-sided L3
  (`laplaceTiltedMeanRightDerivCoeff ≥ 0`), so the `p` first moment disappears;
* at a **zero of `m`** the numerator `D` also vanishes, so the atom-sensitive
  term `−D·Z'` drops and `m` is genuinely two-sided differentiable there —
  the existing edge linear-bound helper applies verbatim.

Design record: `LaplaceEndgame.md` (Frontier A).
-/

open MeasureTheory Set Filter Topology ProbabilityTheory
open scoped intervalIntegral

namespace DriftingIdentifiability

open Paper

/-! ## A1. Càdlàg regularity of the one-sided masses -/

/-- The lower exponential mass is **right-continuous** for every finite
measure — no atomlessness needed, because the atom at the cut point sits inside
`Iic t` on both sides of the approach from the right. -/
theorem lowerExpMass_rightContinuous
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    ContinuousWithinAt (fun t => lowerExpMass τ p t) (Set.Ici x) x := by
  have hτ0 : 0 < τ := hτ
  unfold ContinuousWithinAt lowerExpMass
  simp_rw [← integral_indicator measurableSet_Iic]
  refine tendsto_integral_filter_of_dominated_convergence
    (fun _ : ℝ => Real.exp ((x + 1) / τ)) ?_ ?_ (integrable_const _) ?_
  · exact Eventually.of_forall fun t =>
      (Continuous.aestronglyMeasurable (by fun_prop)).indicator measurableSet_Iic
  · filter_upwards [mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds (lt_add_one x))] with t ht
    refine ae_of_all p fun y => ?_
    by_cases hy : y ∈ Iic t
    · rw [indicator_of_mem hy, Real.norm_eq_abs, abs_of_nonneg (Real.exp_pos _).le]
      refine Real.exp_le_exp.mpr ?_
      have hy' : y ≤ x + 1 := le_trans (mem_Iic.mp hy) (le_of_lt ht)
      rw [div_eq_mul_inv, div_eq_mul_inv]
      exact mul_le_mul_of_nonneg_right hy' (inv_nonneg.mpr hτ0.le)
    · rw [indicator_of_notMem hy]; simp [(Real.exp_pos _).le]
  · refine ae_of_all p fun y => ?_
    rcases le_or_gt y x with hyx | hxy
    · have hev : (fun _ : ℝ => indicator (Iic x) (fun z : ℝ => Real.exp (z / τ)) y)
          =ᶠ[𝓝[Ici x] x]
            (fun t => indicator (Iic t) (fun z : ℝ => Real.exp (z / τ)) y) := by
        filter_upwards [self_mem_nhdsWithin] with t ht
        rw [indicator_of_mem (mem_Iic.mpr hyx),
          indicator_of_mem (mem_Iic.mpr (le_trans hyx (mem_Ici.mp ht)))]
      exact Filter.Tendsto.congr' hev tendsto_const_nhds
    · have hev : (fun _ : ℝ => indicator (Iic x) (fun z : ℝ => Real.exp (z / τ)) y)
          =ᶠ[𝓝[Ici x] x]
            (fun t => indicator (Iic t) (fun z : ℝ => Real.exp (z / τ)) y) := by
        filter_upwards [mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds hxy)] with t ht
        rw [indicator_of_notMem (by simp only [mem_Iic]; exact not_le.mpr hxy),
          indicator_of_notMem (by simp only [mem_Iic]; exact not_le.mpr ht)]
      exact Filter.Tendsto.congr' hev tendsto_const_nhds

/-- The upper exponential mass is **right-continuous** for every finite
measure. -/
theorem upperExpMass_rightContinuous
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    ContinuousWithinAt (fun t => upperExpMass τ p t) (Set.Ici x) x := by
  have hτ0 : 0 < τ := hτ
  unfold ContinuousWithinAt upperExpMass
  simp_rw [← integral_indicator measurableSet_Ioi]
  refine tendsto_integral_filter_of_dominated_convergence
    (fun _ : ℝ => Real.exp (-x / τ)) ?_ ?_ (integrable_const _) ?_
  · exact Eventually.of_forall fun t =>
      (Continuous.aestronglyMeasurable (by fun_prop)).indicator measurableSet_Ioi
  · filter_upwards [self_mem_nhdsWithin] with t ht
    refine ae_of_all p fun y => ?_
    by_cases hy : y ∈ Ioi t
    · rw [indicator_of_mem hy, Real.norm_eq_abs, abs_of_nonneg (Real.exp_pos _).le]
      refine Real.exp_le_exp.mpr ?_
      have hxy : x < y := lt_of_le_of_lt (mem_Ici.mp ht) (mem_Ioi.mp hy)
      rw [div_eq_mul_inv, div_eq_mul_inv, neg_mul, neg_mul]
      exact neg_le_neg (mul_le_mul_of_nonneg_right hxy.le (inv_nonneg.mpr hτ0.le))
    · rw [indicator_of_notMem hy]; simp [(Real.exp_pos _).le]
  · refine ae_of_all p fun y => ?_
    rcases le_or_gt y x with hyx | hxy
    · have hev : (fun _ : ℝ => indicator (Ioi x) (fun z : ℝ => Real.exp (-z / τ)) y)
          =ᶠ[𝓝[Ici x] x]
            (fun t => indicator (Ioi t) (fun z : ℝ => Real.exp (-z / τ)) y) := by
        filter_upwards [self_mem_nhdsWithin] with t ht
        rw [indicator_of_notMem (by simp only [mem_Ioi]; exact not_lt.mpr hyx),
          indicator_of_notMem
            (by simp only [mem_Ioi]; exact not_lt.mpr (le_trans hyx (mem_Ici.mp ht)))]
      exact Filter.Tendsto.congr' hev tendsto_const_nhds
    · have hev : (fun _ : ℝ => indicator (Ioi x) (fun z : ℝ => Real.exp (-z / τ)) y)
          =ᶠ[𝓝[Ici x] x]
            (fun t => indicator (Ioi t) (fun z : ℝ => Real.exp (-z / τ)) y) := by
        filter_upwards [mem_nhdsWithin_of_mem_nhds (Iio_mem_nhds hxy)] with t ht
        rw [indicator_of_mem (mem_Ioi.mpr hxy), indicator_of_mem (mem_Ioi.mpr ht)]
      exact Filter.Tendsto.congr' hev tendsto_const_nhds

/-- The lower exponential mass is monotone (hence measurable). -/
theorem lowerExpMass_monotone
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] :
    Monotone (fun t => lowerExpMass τ p t) := by
  have hτ0 : 0 < τ := hτ
  intro a b hab
  simp only [lowerExpMass]
  refine setIntegral_mono_set (integrable_lowerExpKernel τ hτ0 p b)
    (ae_of_all _ fun y => (Real.exp_pos _).le) ?_
  exact HasSubset.Subset.eventuallyLE (Iic_subset_Iic.mpr hab)

/-- The upper exponential mass is antitone (hence measurable). -/
theorem upperExpMass_antitone
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] :
    Antitone (fun t => upperExpMass τ p t) := by
  have hτ0 : 0 < τ := hτ
  intro a b hab
  simp only [upperExpMass]
  refine setIntegral_mono_set (integrable_upperExpKernel τ hτ0 p a)
    (ae_of_all _ fun y => (Real.exp_pos _).le) ?_
  exact HasSubset.Subset.eventuallyLE (Ioi_subset_Ioi hab)

theorem lowerExpMass_measurable
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] :
    Measurable (fun t => lowerExpMass τ p t) :=
  (lowerExpMass_monotone τ hτ p).measurable

theorem upperExpMass_measurable
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] :
    Measurable (fun t => upperExpMass τ p t) :=
  (upperExpMass_antitone τ hτ p).measurable

/-- The certified right-derivative coefficient of the raw normalizer is
right-continuous for every finite measure. -/
theorem laplaceKernelNormalizerRightDerivCoeff_rightContinuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    ContinuousWithinAt (laplaceKernelNormalizerRightDerivCoeff τ p) (Set.Ici x) x := by
  have hlow := lowerExpMass_rightContinuous τ hτ p x
  have hup := upperExpMass_rightContinuous τ hτ p x
  have he1 : ContinuousWithinAt (fun t : ℝ => Real.exp (-t / τ)) (Set.Ici x) x :=
    (by fun_prop : Continuous fun t : ℝ => Real.exp (-t / τ)).continuousWithinAt
  have he2 : ContinuousWithinAt (fun t : ℝ => Real.exp (t / τ)) (Set.Ici x) x :=
    (by fun_prop : Continuous fun t : ℝ => Real.exp (t / τ)).continuousWithinAt
  unfold laplaceKernelNormalizerRightDerivCoeff
  exact continuousWithinAt_const.mul ((he1.neg.mul hlow).add (he2.mul hup))

theorem laplaceKernelNormalizerRightDerivCoeff_measurable
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] :
    Measurable (laplaceKernelNormalizerRightDerivCoeff τ p) := by
  have hlow := lowerExpMass_measurable τ hτ p
  have hup := upperExpMass_measurable τ hτ p
  have he1 : Measurable (fun t : ℝ => Real.exp (-t / τ)) := by fun_prop
  have he2 : Measurable (fun t : ℝ => Real.exp (t / τ)) := by fun_prop
  unfold laplaceKernelNormalizerRightDerivCoeff
  exact measurable_const.mul ((he1.neg.mul hlow).add (he2.mul hup))

/-! ## A1′. Global bound on the right-derivative coefficient -/

/-- The raw Laplace normalizer of a probability measure is `≤ 1` (the kernel is
`≤ 1`). -/
theorem laplaceKernelNormalizer_le_one
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsProbabilityMeasure p] (x : ℝ) :
    kernelNormalizer (laplaceKernel τ) p x ≤ 1 := by
  have hτ0 : 0 < τ := hτ
  have hk_le : ∀ y, laplaceKernel τ x y ≤ 1 := by
    intro y
    rw [laplaceKernel_real]
    refine Real.exp_le_one_iff.mpr ?_
    have : (0 : ℝ) ≤ 1 / τ * |x - y| :=
      mul_nonneg (one_div_nonneg.mpr hτ0.le) (abs_nonneg _)
    nlinarith
  have hint : Integrable (fun y => laplaceKernel τ x y) p :=
    laplaceKernel_integrable p τ hτ x
  calc kernelNormalizer (laplaceKernel τ) p x
      = ∫ y, laplaceKernel τ x y ∂p := rfl
    _ ≤ ∫ _, (1 : ℝ) ∂p := integral_mono hint (integrable_const 1) hk_le
    _ = 1 := by simp

/-- **Global one-sided-derivative bound.**  `|Z'⁺(x)| ≤ 1/τ` for every
probability measure, because the two one-sided kernel masses are nonnegative
and sum to `Z ≤ 1`. -/
theorem abs_laplaceKernelNormalizerRightDerivCoeff_le
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsProbabilityMeasure p] (x : ℝ) :
    |laplaceKernelNormalizerRightDerivCoeff τ p x| ≤ 1 / τ := by
  have hτ0 : 0 < τ := hτ
  set a₁ : ℝ := Real.exp (-x / τ) * lowerExpMass τ p x with ha₁
  set a₂ : ℝ := Real.exp (x / τ) * upperExpMass τ p x with ha₂
  have ha₁nn : 0 ≤ a₁ := mul_nonneg (Real.exp_pos _).le (lowerExpMass_nonneg τ p x)
  have ha₂nn : 0 ≤ a₂ := mul_nonneg (Real.exp_pos _).le (upperExpMass_nonneg τ p x)
  have hsum : a₁ + a₂ ≤ 1 := by
    rw [ha₁, ha₂, ← laplaceKernelNormalizer_eq_lower_upper τ hτ p x]
    exact laplaceKernelNormalizer_le_one τ hτ p x
  have hrdc : laplaceKernelNormalizerRightDerivCoeff τ p x = 1 / τ * (a₂ - a₁) := by
    rw [laplaceKernelNormalizerRightDerivCoeff, ha₁, ha₂]; ring
  rw [hrdc, abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ 1 / τ)]
  have habs : |a₂ - a₁| ≤ 1 := by
    rw [abs_le]
    constructor
    · linarith
    · linarith
  calc 1 / τ * |a₂ - a₁| ≤ 1 / τ * 1 :=
        mul_le_mul_of_nonneg_left habs (by positivity)
    _ = 1 / τ := by ring

/-! ## A1″. Regularity of the mean-shift ratio derivative -/

theorem laplaceMeanShiftRatioDeriv_measurable
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsProbabilityMeasure p] :
    Measurable (laplaceMeanShiftRatioDeriv τ p) := by
  have hZ : Measurable (fun t => kernelNormalizer (laplaceKernel τ) p t) :=
    (continuous_laplaceKernelNormalizer τ hτ p).measurable
  have hL : Measurable (fun t => kernelNormalizer (laplaceCompanionKernel τ) p t) :=
    (continuous_iff_continuousAt.mpr fun t =>
      (hasDerivAt_laplaceCompanionNormalizer τ hτ p t).continuousAt).measurable
  have hD : Measurable (laplaceMeanShiftNumerator τ p) :=
    (continuous_iff_continuousAt.mpr fun t =>
      (hasDerivAt_laplaceMeanShiftNumerator τ hτ p t).continuousAt).measurable
  have hrdc : Measurable (laplaceKernelNormalizerRightDerivCoeff τ p) :=
    laplaceKernelNormalizerRightDerivCoeff_measurable τ hτ p
  unfold laplaceMeanShiftRatioDeriv laplaceMeanShiftRatioDerivNumerator
    laplaceMeanShiftRatioDerivDenominator laplaceMeanShiftNumeratorDeriv
  exact ((((measurable_const.mul hL).sub (measurable_const.mul hZ)).mul hZ).sub
    (hD.mul hrdc)).div (hZ.pow_const 2)

theorem laplaceMeanShiftRatioDeriv_rightContinuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsProbabilityMeasure p] (x : ℝ) :
    ContinuousWithinAt (laplaceMeanShiftRatioDeriv τ p) (Set.Ici x) x := by
  have hZ : ContinuousWithinAt (fun t => kernelNormalizer (laplaceKernel τ) p t)
      (Set.Ici x) x := (continuous_laplaceKernelNormalizer τ hτ p).continuousWithinAt
  have hL : ContinuousWithinAt (fun t => kernelNormalizer (laplaceCompanionKernel τ) p t)
      (Set.Ici x) x :=
    ((continuous_iff_continuousAt.mpr fun t =>
      (hasDerivAt_laplaceCompanionNormalizer τ hτ p t).continuousAt)).continuousWithinAt
  have hD : ContinuousWithinAt (laplaceMeanShiftNumerator τ p) (Set.Ici x) x :=
    ((continuous_iff_continuousAt.mpr fun t =>
      (hasDerivAt_laplaceMeanShiftNumerator τ hτ p t).continuousAt)).continuousWithinAt
  have hrdc : ContinuousWithinAt (laplaceKernelNormalizerRightDerivCoeff τ p)
      (Set.Ici x) x :=
    laplaceKernelNormalizerRightDerivCoeff_rightContinuousWithinAt_Ici τ hτ p x
  have hZne : kernelNormalizer (laplaceKernel τ) p x ≠ 0 :=
    (laplaceKernelNormalizer_pos p τ hτ x).ne'
  unfold laplaceMeanShiftRatioDeriv laplaceMeanShiftRatioDerivNumerator
    laplaceMeanShiftRatioDerivDenominator laplaceMeanShiftNumeratorDeriv
  refine ContinuousWithinAt.div ?_ (hZ.pow 2) (pow_ne_zero 2 hZne)
  exact (((continuousWithinAt_const.mul hL).sub (continuousWithinAt_const.mul hZ)).mul hZ).sub
    (hD.mul hrdc)

/-! ## A2. One-sided identity pack -/

/-- A product with one factor vanishing and differentiable, the other merely
continuous, is differentiable — the slope splits as `slope f · g`. -/
theorem hasDerivAt_mul_of_eq_zero_of_continuousAt
    {f g : ℝ → ℝ} {f' a : ℝ}
    (hf : HasDerivAt f f' a) (hfa : f a = 0) (hg : ContinuousAt g a) :
    HasDerivAt (fun x => f x * g x) (f' * g a) a := by
  rw [hasDerivAt_iff_tendsto_slope] at hf ⊢
  have hg' : Tendsto g (𝓝[≠] a) (𝓝 (g a)) := hg.tendsto.mono_left nhdsWithin_le_nhds
  refine Filter.Tendsto.congr' ?_ (hf.mul hg')
  filter_upwards [self_mem_nhdsWithin] with t _
  simp only [slope_def_field, hfa, sub_zero, zero_mul, div_mul_eq_mul_div]

/-- One-sided (right) derivative of the mean-shift ratio, for **every**
probability measure — the numerator is `C¹` and the denominator has the
certified right derivative. -/
theorem hasDerivWithinAt_Ici_laplaceMeanShiftRatio
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsProbabilityMeasure p] (x : ℝ) :
    HasDerivWithinAt (laplaceMeanShiftRatio τ p)
      (laplaceMeanShiftRatioDeriv τ p x) (Set.Ici x) x := by
  have hD := (hasDerivAt_laplaceMeanShiftNumerator τ hτ p x).hasDerivWithinAt (s := Set.Ici x)
  have hZ := hasDerivWithinAt_Ici_laplaceKernelNormalizer τ hτ p x
  have hZne : kernelNormalizer (laplaceKernel τ) p x ≠ 0 :=
    (laplaceKernelNormalizer_pos p τ hτ x).ne'
  have hdiv := hD.div hZ hZne
  change HasDerivWithinAt
    (laplaceMeanShiftNumerator τ p /
      fun s : ℝ => kernelNormalizer (laplaceKernel τ) p s)
    (laplaceMeanShiftRatioDeriv τ p x) (Set.Ici x) x
  simpa [laplaceMeanShiftRatioDeriv, laplaceMeanShiftRatioDerivNumerator,
    laplaceMeanShiftRatioDerivDenominator, laplaceMeanShiftNumerator] using hdiv

/-- At a **zero of the mean-shift ratio** the ratio is genuinely two-sided
differentiable, atoms notwithstanding: `m a = 0 ⟹ D a = 0`, so the
atom-sensitive term `−D·Z'` in the quotient rule vanishes. -/
theorem hasDerivAt_laplaceMeanShiftRatio_of_root
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsProbabilityMeasure p] {a : ℝ}
    (hma : laplaceMeanShiftRatio τ p a = 0) :
    HasDerivAt (laplaceMeanShiftRatio τ p) (laplaceMeanShiftRatioDeriv τ p a) a := by
  have hZpos : 0 < kernelNormalizer (laplaceKernel τ) p a :=
    laplaceKernelNormalizer_pos p τ hτ a
  have hZne : kernelNormalizer (laplaceKernel τ) p a ≠ 0 := hZpos.ne'
  have hDa : laplaceMeanShiftNumerator τ p a = 0 := by
    have h0 : laplaceMeanShiftNumerator τ p a /
        kernelNormalizer (laplaceKernel τ) p a = 0 := by
      simpa [laplaceMeanShiftRatio, laplaceMeanShiftNumerator] using hma
    rcases div_eq_zero_iff.mp h0 with h | h
    · exact h
    · exact absurd h hZne
  have hD := hasDerivAt_laplaceMeanShiftNumerator τ hτ p a
  have hg : ContinuousAt (fun x => (kernelNormalizer (laplaceKernel τ) p x)⁻¹) a :=
    ((continuous_laplaceKernelNormalizer τ hτ p).continuousAt).inv₀ hZne
  have hmul := hasDerivAt_mul_of_eq_zero_of_continuousAt hD hDa hg
  have hfun : (fun x => laplaceMeanShiftNumerator τ p x *
      (kernelNormalizer (laplaceKernel τ) p x)⁻¹) = laplaceMeanShiftRatio τ p := by
    funext x
    rw [laplaceMeanShiftRatio, laplaceMeanShiftNumerator, div_eq_mul_inv]
  have hval : laplaceMeanShiftNumeratorDeriv τ p a *
      (kernelNormalizer (laplaceKernel τ) p a)⁻¹ = laplaceMeanShiftRatioDeriv τ p a := by
    rw [laplaceMeanShiftRatioDeriv, laplaceMeanShiftRatioDerivNumerator,
      laplaceMeanShiftRatioDerivDenominator, hDa, zero_mul, sub_zero, pow_two]
    field_simp
  rw [hfun, hval] at hmul
  exact hmul

/-- Nonnegativity of the certified tilted-mean right-derivative coefficient
(all four one-sided mass functionals are nonnegative). -/
theorem laplaceTiltedMeanRightDerivCoeff_nonneg
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsProbabilityMeasure p] (x : ℝ) :
    0 ≤ laplaceTiltedMeanRightDerivCoeff τ p x := by
  have hτ0 : 0 < τ := hτ
  unfold laplaceTiltedMeanRightDerivCoeff
  refine div_nonneg ?_ (sq_nonneg _)
  refine mul_nonneg (div_nonneg (by norm_num) hτ0.le) ?_
  exact add_nonneg
    (mul_nonneg (lowerCompensatedMoment_nonneg τ p x) (upperExpMass_nonneg τ p x))
    (mul_nonneg (upperCompensatedMoment_nonneg τ p x) (lowerExpMass_nonneg τ p x))

/-- **Moment-free L3.**  `m' + 1 ≥ 0` everywhere, because the certified
right-derivative of the tilted mean equals `1 + m'` (one-sided uniqueness) and
is manifestly nonnegative.  This is the ingredient that drops the `p` first
moment from the atomless theorem. -/
theorem laplaceMeanShiftRatioDeriv_add_one_nonneg
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsProbabilityMeasure p] :
    ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1 := by
  intro t
  have hc := hasDerivWithinAt_Ici_laplaceTiltedMeanFromDisplacement τ hτ p t
  have hm := hasDerivWithinAt_Ici_laplaceMeanShiftRatio τ hτ p t
  have hid : HasDerivWithinAt (fun s : ℝ => s) 1 (Set.Ici t) t :=
    (hasDerivAt_id t).hasDerivWithinAt
  have hsum : HasDerivWithinAt (fun s => laplaceTiltedMeanFromDisplacement τ p s)
      (1 + laplaceMeanShiftRatioDeriv τ p t) (Set.Ici t) t := by
    have h0 : HasDerivWithinAt (fun s : ℝ => s + laplaceMeanShiftRatio τ p s)
        (1 + laplaceMeanShiftRatioDeriv τ p t) (Set.Ici t) t := hid.add hm
    exact h0.congr_of_eventuallyEq (Eventually.of_forall fun s => rfl) rfl
  have heq : laplaceTiltedMeanRightDerivCoeff τ p t =
      1 + laplaceMeanShiftRatioDeriv τ p t :=
    UniqueDiffWithinAt.eq_deriv _ (uniqueDiffWithinAt_Ici t) hc hsum
  have hnn := laplaceTiltedMeanRightDerivCoeff_nonneg τ hτ p t
  rw [heq] at hnn
  linarith

/-- Pointwise one-sided companion-normalizer formula
`L_ν = τ((m'+2)·Z_ν + m·Z'⁺_ν)` for arbitrary probability measures. -/
private theorem laplaceCompanionNormalizer_formula_gen
    (τ : ℝ) (hτ : ValidBandwidth τ) (p ν : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure ν]
    (hcommon : ∀ t : ℝ, (∫ y, laplaceWeightedDisplacement τ t y ∂ν) =
      laplaceMeanShiftRatio τ p t * kernelNormalizer (laplaceKernel τ) ν t)
    (x : ℝ) :
    kernelNormalizer (laplaceCompanionKernel τ) ν x =
      τ * ((laplaceMeanShiftRatioDeriv τ p x + 2) *
          kernelNormalizer (laplaceKernel τ) ν x +
        laplaceMeanShiftRatio τ p x *
          laplaceKernelNormalizerRightDerivCoeff τ ν x) := by
  have hm := hasDerivWithinAt_Ici_laplaceMeanShiftRatio τ hτ p x
  have hZ := hasDerivWithinAt_Ici_laplaceKernelNormalizer τ hτ ν x
  have hprod := hm.mul hZ
  have hD2 : HasDerivWithinAt
      (fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂ν)
      (laplaceMeanShiftRatioDeriv τ p x *
          kernelNormalizer (laplaceKernel τ) ν x +
        laplaceMeanShiftRatio τ p x *
          laplaceKernelNormalizerRightDerivCoeff τ ν x) (Set.Ici x) x :=
    hprod.congr_of_eventuallyEq (Eventually.of_forall hcommon) (hcommon x)
  have hD1 := (hasDerivAt_laplaceDisplacementIntegral τ hτ ν x).hasDerivWithinAt
    (s := Set.Ici x)
  have hval := UniqueDiffWithinAt.eq_deriv _ (uniqueDiffWithinAt_Ici x) hD1 hD2
  have hτne : (τ : ℝ) ≠ 0 := hτ.ne'
  set L : ℝ := kernelNormalizer (laplaceCompanionKernel τ) ν x with hLdef
  set Z : ℝ := kernelNormalizer (laplaceKernel τ) ν x with hZdef
  set G : ℝ := laplaceKernelNormalizerRightDerivCoeff τ ν x with hGdef
  set m : ℝ := laplaceMeanShiftRatio τ p x with hmdef
  set m' : ℝ := laplaceMeanShiftRatioDeriv τ p x with hm'def
  have h1 : (1 / τ) * L = m' * Z + m * G + 2 * Z := by linarith
  calc L = 1 * L := (one_mul L).symm
    _ = (τ * (1 / τ)) * L := by rw [mul_one_div, div_self hτne]
    _ = τ * ((1 / τ) * L) := by ring
    _ = τ * (m' * Z + m * G + 2 * Z) := by rw [h1]
    _ = τ * ((m' + 2) * Z + m * G) := by ring

/-- **`K = τ·m·W` for arbitrary probability measures** under zero drift. -/
theorem laplaceCompanionAlignmentDefect_eq_of_zeroDrift_gen
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ) :
    laplaceCompanionAlignmentDefect τ p q x =
      τ * laplaceMeanShiftRatio τ p x *
        laplaceKernelNormalizerWronskian τ p q x := by
  have hp := laplaceCompanionNormalizer_formula_gen τ hτ p p
    (fun t => laplaceMeanShiftRatio_common_self τ hτ p t) x
  have hq := laplaceCompanionNormalizer_formula_gen τ hτ p q
    (fun t => laplaceMeanShiftRatio_common_of_zeroDrift τ hτ p q hzero t) x
  unfold laplaceCompanionAlignmentDefect laplaceKernelNormalizerWronskian
  rw [hp, hq]
  ring

/-- **`K'⁺ = −τ(m'+2)·W` for arbitrary probability measures** — the alignment
defect has a one-sided derivative of the anticipated first-order form. -/
theorem hasDerivWithinAt_Ici_laplaceCompanionAlignmentDefect_of_zeroDrift
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ) :
    HasDerivWithinAt (fun t => laplaceCompanionAlignmentDefect τ p q t)
      (-(τ * (laplaceMeanShiftRatioDeriv τ p x + 2)) *
        laplaceKernelNormalizerWronskian τ p q x) (Set.Ici x) x := by
  have hLp := (hasDerivAt_laplaceCompanionNormalizer τ hτ p x).hasDerivWithinAt
    (s := Set.Ici x)
  have hLq := (hasDerivAt_laplaceCompanionNormalizer τ hτ q x).hasDerivWithinAt
    (s := Set.Ici x)
  have hZp := hasDerivWithinAt_Ici_laplaceKernelNormalizer τ hτ p x
  have hZq := hasDerivWithinAt_Ici_laplaceKernelNormalizer τ hτ q x
  have hK := (hLp.mul hZq).sub (hLq.mul hZp)
  have hval :
      (1 / τ) * (∫ y, laplaceWeightedDisplacement τ x y ∂p) *
            kernelNormalizer (laplaceKernel τ) q x +
          kernelNormalizer (laplaceCompanionKernel τ) p x *
            laplaceKernelNormalizerRightDerivCoeff τ q x -
        ((1 / τ) * (∫ y, laplaceWeightedDisplacement τ x y ∂q) *
            kernelNormalizer (laplaceKernel τ) p x +
          kernelNormalizer (laplaceCompanionKernel τ) q x *
            laplaceKernelNormalizerRightDerivCoeff τ p x) =
      -(τ * (laplaceMeanShiftRatioDeriv τ p x + 2)) *
        laplaceKernelNormalizerWronskian τ p q x := by
    rw [laplaceMeanShiftRatio_common_self τ hτ p x,
      laplaceMeanShiftRatio_common_of_zeroDrift τ hτ p q hzero x,
      laplaceCompanionNormalizer_formula_gen τ hτ p p
        (fun t => laplaceMeanShiftRatio_common_self τ hτ p t) x,
      laplaceCompanionNormalizer_formula_gen τ hτ p q
        (fun t => laplaceMeanShiftRatio_common_of_zeroDrift τ hτ p q hzero t) x]
    unfold laplaceKernelNormalizerWronskian
    ring
  rw [hval] at hK
  exact hK.congr_of_eventuallyEq (Eventually.of_forall fun t => rfl) rfl

/-- The Abel-shaped one-sided ODE on `{m ≠ 0}`, for arbitrary probability
measures. -/
theorem hasDerivWithinAt_Ici_laplaceCompanionAlignmentDefect_of_ne
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ)
    (hmx : laplaceMeanShiftRatio τ p x ≠ 0) :
    HasDerivWithinAt (fun t => laplaceCompanionAlignmentDefect τ p q t)
      (-(2 * ((laplaceMeanShiftRatioDeriv τ p x + 2) / 2) /
          laplaceMeanShiftRatio τ p x) *
        laplaceCompanionAlignmentDefect τ p q x) (Set.Ici x) x := by
  have h := hasDerivWithinAt_Ici_laplaceCompanionAlignmentDefect_of_zeroDrift
    τ hτ p q hzero x
  have heq :
      -(τ * (laplaceMeanShiftRatioDeriv τ p x + 2)) *
          laplaceKernelNormalizerWronskian τ p q x =
        -(2 * ((laplaceMeanShiftRatioDeriv τ p x + 2) / 2) /
            laplaceMeanShiftRatio τ p x) *
          laplaceCompanionAlignmentDefect τ p q x := by
    rw [laplaceCompanionAlignmentDefect_eq_of_zeroDrift_gen τ hτ p q hzero x]
    field_simp
  rw [heq] at h
  exact h

/-! ## A3. Right-continuous FTC primitive helpers -/

/-- Continuity of the mean-shift ratio for every probability measure (`D/Z`
with `D` continuous and `Z` continuous positive). -/
theorem continuous_laplaceMeanShiftRatio
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsProbabilityMeasure p] :
    Continuous (laplaceMeanShiftRatio τ p) := by
  have hZne : ∀ x, kernelNormalizer (laplaceKernel τ) p x ≠ 0 := fun x =>
    (laplaceKernelNormalizer_pos p τ hτ x).ne'
  have h := ((continuous_iff_continuousAt.mpr fun t =>
      (hasDerivAt_laplaceMeanShiftNumerator τ hτ p t).continuousAt)).div
    (continuous_laplaceKernelNormalizer τ hτ p) hZne
  have heq : (laplaceMeanShiftNumerator τ p /
      fun x => kernelNormalizer (laplaceKernel τ) p x) = laplaceMeanShiftRatio τ p := by
    funext x
    simp only [Pi.div_apply, laplaceMeanShiftRatio, laplaceMeanShiftNumerator]
  rwa [heq] at h

/-- A measurable function bounded on the closed interval is interval-integrable. -/
theorem intervalIntegrable_of_measurable_bounded
    {c : ℝ → ℝ} {a b M : ℝ}
    (hmeas : Measurable c) (hbd : ∀ t ∈ Set.uIcc a b, |c t| ≤ M) :
    IntervalIntegrable c volume a b := by
  rw [intervalIntegrable_iff]
  refine Measure.integrableOn_of_bounded ?_ hmeas.aestronglyMeasurable
    (M := M) ?_
  · rw [Set.uIoc]
    exact measure_Ioc_lt_top.ne
  · filter_upwards [ae_restrict_mem measurableSet_uIoc] with t ht
    rw [Real.norm_eq_abs]
    exact hbd t (Set.Ioc_subset_Icc_self ht)

/-- The FTC primitive of a globally measurable, right-continuous, locally
interval-integrable coefficient has the expected right derivative. -/
theorem intervalPrimitive_hasDerivWithinAt_Ici_of_rightContinuous
    {c : ℝ → ℝ} {base t : ℝ}
    (hint : IntervalIntegrable c volume base t)
    (hmeas : Measurable c)
    (hrc : ContinuousWithinAt c (Set.Ici t) t) :
    HasDerivWithinAt (fun z : ℝ => ∫ s in base..z, c s) (c t) (Set.Ici t) t := by
  have htlt : t < t + 1 := lt_add_one t
  set J : Set ℝ := Set.Icc t (t + 1) with hJdef
  haveI : Fact (t ∈ J) := ⟨⟨le_rfl, htlt.le⟩⟩
  have hrcJ : ContinuousWithinAt c J t :=
    hrc.mono (by rw [hJdef]; exact Set.Icc_subset_Ici_self)
  have hderivJ : HasDerivWithinAt (fun z : ℝ => ∫ s in base..z, c s) (c t) J t :=
    intervalIntegral.integral_hasDerivWithinAt_right hint
      hmeas.stronglyMeasurable.stronglyMeasurableAtFilter hrcJ
  exact hderivJ.mono_of_mem_nhdsWithin (Icc_mem_nhdsGE htlt)

/-- Continuity of the same FTC primitive on compact subintervals. -/
theorem intervalPrimitive_continuousOn_of_intervalIntegrable
    {c : ℝ → ℝ} {base x y : ℝ}
    (hint : IntervalIntegrable c volume (min base x) (max base y)) :
    ContinuousOn (fun z : ℝ => ∫ s in base..z, c s) (Set.Icc x y) :=
  fun b₀ _ => intervalIntegral.continuousWithinAt_primitive
    (by simp : volume ({b₀} : Set ℝ) = 0) hint

/-! ## A4. The coefficient bound, edge lemmas, and ray lemmas -/

/-- On any compact interval avoiding the zeros of the mean-shift ratio, the
Abel coefficient `c = 2μ'/m` is bounded.  The numerator uses the global bound
`|Z'⁺| ≤ 1/τ`; the denominator `Z²` and `|m|` are bounded below by compactness. -/
private theorem exists_uIcc_bound_laplaceGapCoeff
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsProbabilityMeasure p]
    {u v : ℝ}
    (hm : ∀ t ∈ Set.uIcc u v, laplaceMeanShiftRatio τ p t ≠ 0) :
    ∃ M : ℝ, ∀ t ∈ Set.uIcc u v,
      |(2 : ℝ) * ((laplaceMeanShiftRatioDeriv τ p t + 2) / 2) /
        laplaceMeanShiftRatio τ p t| ≤ M := by
  have hτ0 : 0 < τ := hτ
  set K := Set.uIcc u v with hKdef
  have hK : IsCompact K := isCompact_uIcc
  have hKne : K.Nonempty := nonempty_uIcc
  have hZcont : Continuous (fun t => kernelNormalizer (laplaceKernel τ) p t) :=
    continuous_laplaceKernelNormalizer τ hτ p
  have hLcont : Continuous (fun t => kernelNormalizer (laplaceCompanionKernel τ) p t) :=
    continuous_iff_continuousAt.mpr fun t =>
      (hasDerivAt_laplaceCompanionNormalizer τ hτ p t).continuousAt
  have hDcont : Continuous (laplaceMeanShiftNumerator τ p) :=
    continuous_iff_continuousAt.mpr fun t =>
      (hasDerivAt_laplaceMeanShiftNumerator τ hτ p t).continuousAt
  have hD'cont : Continuous (laplaceMeanShiftNumeratorDeriv τ p) := by
    unfold laplaceMeanShiftNumeratorDeriv
    exact (continuous_const.mul hLcont).sub (continuous_const.mul hZcont)
  have hmcont : Continuous (laplaceMeanShiftRatio τ p) :=
    continuous_laplaceMeanShiftRatio τ hτ p
  -- lower bound on Z
  obtain ⟨zZ, hzZ, hminZ⟩ := hK.exists_isMinOn hKne hZcont.continuousOn
  set εZ : ℝ := kernelNormalizer (laplaceKernel τ) p zZ with hεZdef
  have hεZpos : 0 < εZ := laplaceKernelNormalizer_pos p τ hτ zZ
  -- lower bound on |m|
  obtain ⟨zm, hzm, hminm⟩ := hK.exists_isMinOn hKne hmcont.abs.continuousOn
  set εm : ℝ := |laplaceMeanShiftRatio τ p zm| with hεmdef
  have hεmpos : 0 < εm := abs_pos.mpr (hm zm hzm)
  -- upper bound on the numerator majorant
  set g : ℝ → ℝ := fun t => |laplaceMeanShiftNumeratorDeriv τ p t *
      kernelNormalizer (laplaceKernel τ) p t| +
    |laplaceMeanShiftNumerator τ p t| * (1 / τ) with hgdef
  have hgcont : Continuous g :=
    ((hD'cont.mul hZcont).abs).add (hDcont.abs.mul continuous_const)
  obtain ⟨zg, hzg, hmaxg⟩ := hK.exists_isMaxOn hKne hgcont.continuousOn
  set Mnum : ℝ := g zg with hMnumdef
  refine ⟨(Mnum / εZ ^ 2 + 2) / εm, fun t ht => ?_⟩
  have hZt_pos : 0 < kernelNormalizer (laplaceKernel τ) p t :=
    laplaceKernelNormalizer_pos p τ hτ t
  have hZt_ge : εZ ≤ kernelNormalizer (laplaceKernel τ) p t := isMinOn_iff.mp hminZ t ht
  have hmt_ge : εm ≤ |laplaceMeanShiftRatio τ p t| := isMinOn_iff.mp hminm t ht
  have hg_le : g t ≤ Mnum := isMaxOn_iff.mp hmaxg t ht
  -- |numerator| ≤ Mnum
  have hrdc : |laplaceMeanShiftNumerator τ p t *
      laplaceKernelNormalizerRightDerivCoeff τ p t| ≤
      |laplaceMeanShiftNumerator τ p t| * (1 / τ) := by
    rw [abs_mul]
    exact mul_le_mul_of_nonneg_left
      (abs_laplaceKernelNormalizerRightDerivCoeff_le τ hτ p t) (abs_nonneg _)
  have hnum : |laplaceMeanShiftRatioDerivNumerator τ p t| ≤ Mnum := by
    rw [laplaceMeanShiftRatioDerivNumerator]
    calc |laplaceMeanShiftNumeratorDeriv τ p t *
            kernelNormalizer (laplaceKernel τ) p t -
          laplaceMeanShiftNumerator τ p t *
            laplaceKernelNormalizerRightDerivCoeff τ p t|
        ≤ |laplaceMeanShiftNumeratorDeriv τ p t *
            kernelNormalizer (laplaceKernel τ) p t| +
          |laplaceMeanShiftNumerator τ p t *
            laplaceKernelNormalizerRightDerivCoeff τ p t| := by
          rw [sub_eq_add_neg]
          exact (abs_add_le _ _).trans_eq (by rw [abs_neg])
      _ ≤ |laplaceMeanShiftNumeratorDeriv τ p t *
            kernelNormalizer (laplaceKernel τ) p t| +
          |laplaceMeanShiftNumerator τ p t| * (1 / τ) := by linarith
      _ = g t := rfl
      _ ≤ Mnum := hg_le
  have hMnum_nonneg : 0 ≤ Mnum := le_trans (abs_nonneg _) hnum
  -- |msrDeriv t| ≤ Mnum / εZ²
  have hZsq : (0 : ℝ) < kernelNormalizer (laplaceKernel τ) p t ^ 2 := by positivity
  have hεZsq : (0 : ℝ) < εZ ^ 2 := by positivity
  have hd_abs : |laplaceMeanShiftRatioDeriv τ p t| ≤ Mnum / εZ ^ 2 := by
    rw [laplaceMeanShiftRatioDeriv, laplaceMeanShiftRatioDerivDenominator, abs_div,
      abs_of_pos hZsq, div_le_iff₀ hZsq, div_mul_eq_mul_div, le_div_iff₀ hεZsq]
    have hle : εZ ^ 2 ≤ kernelNormalizer (laplaceKernel τ) p t ^ 2 := by
      nlinarith [hZt_ge, hεZpos.le]
    nlinarith [hnum, abs_nonneg (laplaceMeanShiftRatioDerivNumerator τ p t), hle,
      hMnum_nonneg, hεZsq.le]
  -- assemble
  have hcabs : |(2 : ℝ) * ((laplaceMeanShiftRatioDeriv τ p t + 2) / 2) /
      laplaceMeanShiftRatio τ p t| =
      |laplaceMeanShiftRatioDeriv τ p t + 2| / |laplaceMeanShiftRatio τ p t| := by
    rw [show (2 : ℝ) * ((laplaceMeanShiftRatioDeriv τ p t + 2) / 2) =
      laplaceMeanShiftRatioDeriv τ p t + 2 from by ring, abs_div]
  rw [hcabs]
  have hcm_nonneg : (0 : ℝ) ≤ Mnum / εZ ^ 2 + 2 :=
    add_nonneg (div_nonneg hMnum_nonneg (sq_nonneg εZ)) (by norm_num)
  have hnumer : |laplaceMeanShiftRatioDeriv τ p t + 2| ≤ Mnum / εZ ^ 2 + 2 := by
    calc |laplaceMeanShiftRatioDeriv τ p t + 2|
        ≤ |laplaceMeanShiftRatioDeriv τ p t| + |(2 : ℝ)| := abs_add_le _ _
      _ = |laplaceMeanShiftRatioDeriv τ p t| + 2 := by norm_num
      _ ≤ Mnum / εZ ^ 2 + 2 := by linarith
  have hmt_pos : (0 : ℝ) < |laplaceMeanShiftRatio τ p t| :=
    lt_of_lt_of_le hεmpos hmt_ge
  rw [div_le_iff₀ hmt_pos, div_mul_eq_mul_div, le_div_iff₀ hεmpos]
  nlinarith [hnumer, hmt_ge, hεmpos.le, hcm_nonneg,
    abs_nonneg (laplaceMeanShiftRatioDeriv τ p t + 2)]

/-- **Left-edge blow-up (unconditional).**  If `m a = 0` and `m > 0` on `(a,b)`,
the alignment defect vanishes on `(a,b)`.  No atomlessness: the coefficient is
merely càdlàg, which is all the FTC primitive needs. -/
theorem laplaceAlignmentDefect_eq_zero_on_Ioo_of_leftEdge_gen
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hμ1 : ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    {a b : ℝ} (hab : a < b)
    (hma : laplaceMeanShiftRatio τ p a = 0)
    (hmpos : ∀ t : ℝ, a < t → t < b → 0 < laplaceMeanShiftRatio τ p t) :
    ∀ x : ℝ, a < x → x < b →
      laplaceCompanionAlignmentDefect τ p q x = 0 := by
  let K : ℝ → ℝ := fun x => laplaceCompanionAlignmentDefect τ p q x
  let μDeriv : ℝ → ℝ := fun x => (laplaceMeanShiftRatioDeriv τ p x + 2) / 2
  let m : ℝ → ℝ := fun x => laplaceMeanShiftRatio τ p x
  let c : ℝ → ℝ := fun x => ((2 : ℝ) * μDeriv x) / m x
  have hm_cont : Continuous m := continuous_laplaceMeanShiftRatio τ hτ p
  have hKcont : Continuous K := continuous_laplaceCompanionAlignmentDefect τ hτ p q
  have hc_meas : Measurable c := by
    have hμmeas : Measurable μDeriv :=
      ((laplaceMeanShiftRatioDeriv_measurable τ hτ p).add measurable_const).div_const 2
    exact (measurable_const.mul hμmeas).div hm_cont.measurable
  have hc_rc : ∀ t : ℝ, a < t → t < b → ContinuousWithinAt c (Set.Ici t) t := by
    intro t hat htb
    have hmt : m t ≠ 0 := (hmpos t hat htb).ne'
    refine ContinuousWithinAt.div ?_ hm_cont.continuousWithinAt hmt
    exact continuousWithinAt_const.mul
      (((laplaceMeanShiftRatioDeriv_rightContinuousWithinAt_Ici τ hτ p t).add
        continuousWithinAt_const).div_const 2)
  have hc_int : ∀ w z : ℝ, w ∈ Ioo a b → z ∈ Ioo a b →
      IntervalIntegrable c volume w z := by
    intro w z hw hz
    obtain ⟨M, hM⟩ := exists_uIcc_bound_laplaceGapCoeff τ hτ p (u := w) (v := z)
      (fun t ht => by
        have ht' : t ∈ Ioo a b := OrdConnected.uIcc_subset inferInstance hw hz ht
        exact (hmpos t ht'.1 ht'.2).ne')
    exact intervalIntegrable_of_measurable_bounded hc_meas
      (fun t ht => by simpa [c, μDeriv, m] using hM t ht)
  have hm_root : HasDerivAt m (laplaceMeanShiftRatioDeriv τ p a) a :=
    hasDerivAt_laplaceMeanShiftRatio_of_root τ hτ p (by simpa [m] using hma)
  obtain ⟨_, r₂, L, _, _, har₂, hr₂b, hL, _, hlin⟩ :=
    exists_Ioo_linear_bound_of_hasDerivAt_zero (f := m) (a := a)
      (lower := a - 1) (upper := b) (by linarith) hab hm_root
      (by simpa [m] using hma)
  refine abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper
    (W := K) (A := fun z : ℝ => ∫ s in r₂..z, c s)
    (μDeriv := μDeriv) (m := m)
    (a := a) (b := b) (r := r₂) (δ := 1 / 2) (L := L)
    hab har₂ hr₂b (by norm_num) hL ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro x y hx hxy hy
    have hxg : x ∈ Ioo a b := ⟨hx, lt_of_le_of_lt hxy hy⟩
    have hyg : y ∈ Ioo a b := ⟨lt_of_lt_of_le hx hxy, hy⟩
    refine hKcont.continuousOn.mul (Real.continuous_exp.comp_continuousOn ?_)
    exact intervalPrimitive_continuousOn_of_intervalIntegrable
      (hc_int (min r₂ x) (max r₂ y)
        ⟨lt_min har₂ hxg.1, lt_of_le_of_lt (min_le_left _ _) hr₂b⟩
        ⟨lt_of_lt_of_le har₂ (le_max_left _ _), max_lt hr₂b hyg.2⟩)
  · intro x y hx hxy hy t ht
    have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
      (hmpos t (lt_of_lt_of_le hx ht.1) (lt_trans ht.2 hy)).ne'
    have hderiv := hasDerivWithinAt_Ici_laplaceCompanionAlignmentDefect_of_ne
      τ hτ p q hzero t hmt
    simpa [K, μDeriv, m] using hderiv
  · intro x y hx hxy hy t ht
    have htGap : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    exact intervalPrimitive_hasDerivWithinAt_Ici_of_rightContinuous
      (hc_int r₂ t ⟨har₂, hr₂b⟩ htGap) hc_meas (hc_rc t htGap.1 htGap.2)
  · intro x y hx hxy hy
    have hxg : x ∈ Ioo a b := ⟨hx, lt_of_le_of_lt hxy hy⟩
    have hyg : y ∈ Ioo a b := ⟨lt_of_lt_of_le hx hxy, hy⟩
    exact intervalPrimitive_continuousOn_of_intervalIntegrable
      (hc_int (min r₂ x) (max r₂ y)
        ⟨lt_min har₂ hxg.1, lt_of_le_of_lt (min_le_left _ _) hr₂b⟩
        ⟨lt_of_lt_of_le har₂ (le_max_left _ _), max_lt hr₂b hyg.2⟩)
  · exact (hKcont.continuousAt.norm.isBoundedUnder_le).mono nhdsWithin_le_nhds
  · intro t _ _
    have := hμ1 t
    dsimp [μDeriv]
    linarith
  · intro t hat htr
    exact hmpos t hat (lt_trans htr hr₂b)
  · intro t hat htr
    exact hlin t hat htr.le

/-- **Right-edge blow-up (unconditional).** -/
theorem laplaceAlignmentDefect_eq_zero_on_Ioo_of_rightEdge_gen
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hμ1 : ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    {a b : ℝ} (hab : a < b)
    (hmb : laplaceMeanShiftRatio τ p b = 0)
    (hmneg : ∀ t : ℝ, a < t → t < b → laplaceMeanShiftRatio τ p t < 0) :
    ∀ x : ℝ, a < x → x < b →
      laplaceCompanionAlignmentDefect τ p q x = 0 := by
  let K : ℝ → ℝ := fun x => laplaceCompanionAlignmentDefect τ p q x
  let μDeriv : ℝ → ℝ := fun x => (laplaceMeanShiftRatioDeriv τ p x + 2) / 2
  let m : ℝ → ℝ := fun x => laplaceMeanShiftRatio τ p x
  let c : ℝ → ℝ := fun x => ((2 : ℝ) * μDeriv x) / m x
  have hm_cont : Continuous m := continuous_laplaceMeanShiftRatio τ hτ p
  have hKcont : Continuous K := continuous_laplaceCompanionAlignmentDefect τ hτ p q
  have hc_meas : Measurable c := by
    have hμmeas : Measurable μDeriv :=
      ((laplaceMeanShiftRatioDeriv_measurable τ hτ p).add measurable_const).div_const 2
    exact (measurable_const.mul hμmeas).div hm_cont.measurable
  have hc_rc : ∀ t : ℝ, a < t → t < b → ContinuousWithinAt c (Set.Ici t) t := by
    intro t hat htb
    have hmt : m t ≠ 0 := (hmneg t hat htb).ne
    refine ContinuousWithinAt.div ?_ hm_cont.continuousWithinAt hmt
    exact continuousWithinAt_const.mul
      (((laplaceMeanShiftRatioDeriv_rightContinuousWithinAt_Ici τ hτ p t).add
        continuousWithinAt_const).div_const 2)
  have hc_int : ∀ w z : ℝ, w ∈ Ioo a b → z ∈ Ioo a b →
      IntervalIntegrable c volume w z := by
    intro w z hw hz
    obtain ⟨M, hM⟩ := exists_uIcc_bound_laplaceGapCoeff τ hτ p (u := w) (v := z)
      (fun t ht => by
        have ht' : t ∈ Ioo a b := OrdConnected.uIcc_subset inferInstance hw hz ht
        exact (hmneg t ht'.1 ht'.2).ne)
    exact intervalIntegrable_of_measurable_bounded hc_meas
      (fun t ht => by simpa [c, μDeriv, m] using hM t ht)
  have hm_root : HasDerivAt m (laplaceMeanShiftRatioDeriv τ p b) b :=
    hasDerivAt_laplaceMeanShiftRatio_of_root τ hτ p (by simpa [m] using hmb)
  obtain ⟨l₂, _, L, hal₂, hl₂b, _, _, hL, hlin, _⟩ :=
    exists_Ioo_linear_bound_of_hasDerivAt_zero (f := m) (a := b)
      (lower := a) (upper := b + 1) hab (lt_add_one b) hm_root
      (by simpa [m] using hmb)
  refine abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower
    (W := K) (A := fun z : ℝ => ∫ s in l₂..z, c s)
    (μDeriv := μDeriv) (m := m)
    (a := a) (b := b) (l := l₂) (δ := 1 / 2) (L := L)
    hab hal₂ hl₂b (by norm_num) hL ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
  · intro x y hx hxy hy
    have hxg : x ∈ Ioo a b := ⟨hx, lt_of_le_of_lt hxy hy⟩
    have hyg : y ∈ Ioo a b := ⟨lt_of_lt_of_le hx hxy, hy⟩
    refine hKcont.continuousOn.mul (Real.continuous_exp.comp_continuousOn ?_)
    exact intervalPrimitive_continuousOn_of_intervalIntegrable
      (hc_int (min l₂ x) (max l₂ y)
        ⟨lt_min hal₂ hxg.1, lt_of_le_of_lt (min_le_left _ _) hl₂b⟩
        ⟨lt_of_lt_of_le hal₂ (le_max_left _ _), max_lt hl₂b hyg.2⟩)
  · intro x y hx hxy hy t ht
    have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
      (hmneg t (lt_of_lt_of_le hx ht.1) (lt_trans ht.2 hy)).ne
    have hderiv := hasDerivWithinAt_Ici_laplaceCompanionAlignmentDefect_of_ne
      τ hτ p q hzero t hmt
    simpa [K, μDeriv, m] using hderiv
  · intro x y hx hxy hy t ht
    have htGap : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    exact intervalPrimitive_hasDerivWithinAt_Ici_of_rightContinuous
      (hc_int l₂ t ⟨hal₂, hl₂b⟩ htGap) hc_meas (hc_rc t htGap.1 htGap.2)
  · intro x y hx hxy hy
    have hxg : x ∈ Ioo a b := ⟨hx, lt_of_le_of_lt hxy hy⟩
    have hyg : y ∈ Ioo a b := ⟨lt_of_lt_of_le hx hxy, hy⟩
    exact intervalPrimitive_continuousOn_of_intervalIntegrable
      (hc_int (min l₂ x) (max l₂ y)
        ⟨lt_min hal₂ hxg.1, lt_of_le_of_lt (min_le_left _ _) hl₂b⟩
        ⟨lt_of_lt_of_le hal₂ (le_max_left _ _), max_lt hl₂b hyg.2⟩)
  · exact (hKcont.continuousAt.norm.isBoundedUnder_le).mono nhdsWithin_le_nhds
  · intro t _ _
    have := hμ1 t
    dsimp [μDeriv]
    linarith
  · intro t hlt htb
    exact hmneg t (lt_of_lt_of_le hal₂ hlt) htb
  · intro t hlt htb
    exact hlin t hlt htb

/-- **Left outer ray (unconditional).** -/
theorem laplaceAlignmentDefect_eq_zero_on_left_ray_gen
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hμ1 : ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    {a : ℝ} (hmpos : ∀ t : ℝ, t ≤ a → 0 < laplaceMeanShiftRatio τ p t) :
    ∀ x : ℝ, x ≤ a → laplaceCompanionAlignmentDefect τ p q x = 0 := by
  have hKcont : Continuous (fun x => laplaceCompanionAlignmentDefect τ p q x) :=
    continuous_laplaceCompanionAlignmentDefect τ hτ p q
  refine abel_left_outer_zero_of_muDeriv_nonneg_of_m_pos
    (W := fun x : ℝ => laplaceCompanionAlignmentDefect τ p q x)
    (muDeriv := fun t : ℝ => (laplaceMeanShiftRatioDeriv τ p t + 2) / 2)
    (m := fun t : ℝ => laplaceMeanShiftRatio τ p t) (a := a)
    ?_ ?_ ?_ ?_ (laplaceCompanionAlignmentDefect_tendsto_atBot_zero τ hτ p q)
  · intro b x _ _
    exact hKcont.continuousOn
  · intro b x _ hxa t ht
    have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
      (hmpos t (le_trans ht.2.le hxa)).ne'
    exact hasDerivWithinAt_Ici_laplaceCompanionAlignmentDefect_of_ne
      τ hτ p q hzero t hmt
  · intro t _
    have := hμ1 t
    linarith
  · intro t ht
    exact hmpos t ht

/-- **Right outer ray (unconditional).** -/
theorem laplaceAlignmentDefect_eq_zero_on_right_ray_gen
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hμ1 : ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    {a : ℝ} (hmneg : ∀ t : ℝ, a ≤ t → laplaceMeanShiftRatio τ p t < 0) :
    ∀ x : ℝ, a ≤ x → laplaceCompanionAlignmentDefect τ p q x = 0 := by
  have hKcont : Continuous (fun x => laplaceCompanionAlignmentDefect τ p q x) :=
    continuous_laplaceCompanionAlignmentDefect τ hτ p q
  refine abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg
    (W := fun x : ℝ => laplaceCompanionAlignmentDefect τ p q x)
    (muDeriv := fun t : ℝ => (laplaceMeanShiftRatioDeriv τ p t + 2) / 2)
    (m := fun t : ℝ => laplaceMeanShiftRatio τ p t) (a := a)
    ?_ ?_ ?_ ?_ (laplaceCompanionAlignmentDefect_tendsto_atTop_zero τ hτ p q)
  · intro x b _ _
    exact hKcont.continuousOn
  · intro x b hax _ t ht
    have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
      (hmneg t (le_trans hax ht.1)).ne
    exact hasDerivWithinAt_Ici_laplaceCompanionAlignmentDefect_of_ne
      τ hτ p q hzero t hmt
  · intro t _
    have := hμ1 t
    linarith
  · intro t ht
    exact hmneg t ht

/-! ## A5. Global vanishing, headline theorem, legitimacy -/

/-- **The alignment defect vanishes identically under zero drift — for
arbitrary probability measures.**  No atomlessness, no moment, no density. -/
theorem laplaceCompanionAlignmentDefect_eq_zero_of_zeroDrift_gen
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    ∀ x : ℝ, laplaceCompanionAlignmentDefect τ p q x = 0 := by
  let m : ℝ → ℝ := fun x => laplaceMeanShiftRatio τ p x
  have hμ1 : ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1 :=
    laplaceMeanShiftRatioDeriv_add_one_nonneg τ hτ p
  have hm_cont : Continuous m := continuous_laplaceMeanShiftRatio τ hτ p
  intro x₀
  by_cases hmx₀ : m x₀ = 0
  · rw [laplaceCompanionAlignmentDefect_eq_of_zeroDrift_gen τ hτ p q hzero x₀]
    have h0 : laplaceMeanShiftRatio τ p x₀ = 0 := hmx₀
    rw [h0]; ring
  · rcases lt_or_gt_of_ne hmx₀ with hneg | hpos
    · by_cases hS : ∃ t : ℝ, x₀ ≤ t ∧ m t = 0
      · obtain ⟨t₀, ht₀x, ht₀0⟩ := hS
        have hSne : (Ici x₀ ∩ m ⁻¹' {0}).Nonempty :=
          ⟨t₀, mem_Ici.mpr ht₀x, by simpa using ht₀0⟩
        have hSclosed : IsClosed (Ici x₀ ∩ m ⁻¹' {0}) :=
          isClosed_Ici.inter (isClosed_singleton.preimage hm_cont)
        have hSbdd : BddBelow (Ici x₀ ∩ m ⁻¹' {0}) :=
          ⟨x₀, fun t ht => mem_Ici.mp ht.1⟩
        set β : ℝ := sInf (Ici x₀ ∩ m ⁻¹' {0}) with hβdef
        have hβmem := hSclosed.csInf_mem hSne hSbdd
        have hxβ : x₀ ≤ β := mem_Ici.mp hβmem.1
        have hmβ : m β = 0 := by simpa using hβmem.2
        have hβgt : x₀ < β := by
          rcases eq_or_lt_of_le hxβ with heq | hlt
          · exfalso; rw [← heq] at hmβ; exact hneg.ne hmβ
          · exact hlt
        have hmid : ∀ t : ℝ, x₀ ≤ t → t < β → m t < 0 := by
          intro t hxt htβ
          rcases lt_trichotomy (m t) 0 with hlt | heq | hgt
          · exact hlt
          · exfalso
            have : β ≤ t := csInf_le hSbdd ⟨mem_Ici.mpr hxt, by simpa using heq⟩
            linarith
          · exfalso
            have h0mem : (0 : ℝ) ∈ Icc (m x₀) (m t) := ⟨hneg.le, hgt.le⟩
            obtain ⟨z, hzIcc, hz0⟩ :=
              intermediate_value_Icc hxt hm_cont.continuousOn h0mem
            have : β ≤ z :=
              csInf_le hSbdd ⟨mem_Ici.mpr hzIcc.1, by simpa using hz0⟩
            linarith [hzIcc.2]
        obtain ⟨l₁, r₁, δ₁, _, hl₁x, hxr₁, _, hδ₁, hbound₁⟩ :=
          exists_Ioo_lower_bound_half_of_continuous_pos
            (g := fun t : ℝ => -(m t)) (a := x₀)
            (lower := x₀ - 1) (upper := β) hm_cont.neg
            (by linarith) hβgt (by simpa using hneg)
        have hmneg_ext : ∀ t : ℝ, l₁ < t → t < β → m t < 0 := by
          intro t hlt htβ
          rcases le_or_gt x₀ t with hle | hgt
          · exact hmid t hle htβ
          · have := hbound₁ t hlt (lt_trans hgt hxr₁); linarith
        have hvan := laplaceAlignmentDefect_eq_zero_on_Ioo_of_rightEdge_gen
          τ hτ p q hzero hμ1 (lt_trans hl₁x hβgt) hmβ hmneg_ext
        exact hvan x₀ hl₁x hβgt
      · push Not at hS
        have hmray : ∀ t : ℝ, x₀ ≤ t → m t < 0 := by
          intro t hxt
          rcases lt_trichotomy (m t) 0 with hlt | heq | hgt
          · exact hlt
          · exact absurd heq (hS t hxt)
          · exfalso
            have h0mem : (0 : ℝ) ∈ Icc (m x₀) (m t) := ⟨hneg.le, hgt.le⟩
            obtain ⟨z, hzIcc, hz0⟩ :=
              intermediate_value_Icc hxt hm_cont.continuousOn h0mem
            exact hS z hzIcc.1 hz0
        exact laplaceAlignmentDefect_eq_zero_on_right_ray_gen
          τ hτ p q hzero hμ1 hmray x₀ le_rfl
    · by_cases hS : ∃ t : ℝ, t ≤ x₀ ∧ m t = 0
      · obtain ⟨t₀, ht₀x, ht₀0⟩ := hS
        have hSne : (Iic x₀ ∩ m ⁻¹' {0}).Nonempty :=
          ⟨t₀, mem_Iic.mpr ht₀x, by simpa using ht₀0⟩
        have hSclosed : IsClosed (Iic x₀ ∩ m ⁻¹' {0}) :=
          isClosed_Iic.inter (isClosed_singleton.preimage hm_cont)
        have hSbdd : BddAbove (Iic x₀ ∩ m ⁻¹' {0}) :=
          ⟨x₀, fun t ht => mem_Iic.mp ht.1⟩
        set α : ℝ := sSup (Iic x₀ ∩ m ⁻¹' {0}) with hαdef
        have hαmem := hSclosed.csSup_mem hSne hSbdd
        have hαx : α ≤ x₀ := mem_Iic.mp hαmem.1
        have hmα : m α = 0 := by simpa using hαmem.2
        have hαlt : α < x₀ := by
          rcases eq_or_lt_of_le hαx with heq | hlt
          · exfalso; rw [heq] at hmα; exact hpos.ne' hmα
          · exact hlt
        have hmid : ∀ t : ℝ, α < t → t ≤ x₀ → 0 < m t := by
          intro t hαt htx
          rcases lt_trichotomy (m t) 0 with hlt | heq | hgt
          · exfalso
            have h0mem : (0 : ℝ) ∈ Icc (m t) (m x₀) := ⟨hlt.le, hpos.le⟩
            obtain ⟨z, hzIcc, hz0⟩ :=
              intermediate_value_Icc htx hm_cont.continuousOn h0mem
            have : z ≤ α :=
              le_csSup hSbdd ⟨mem_Iic.mpr hzIcc.2, by simpa using hz0⟩
            linarith [hzIcc.1]
          · exfalso
            have : t ≤ α := le_csSup hSbdd ⟨mem_Iic.mpr htx, by simpa using heq⟩
            linarith
          · exact hgt
        obtain ⟨l₁, r₁, δ₁, _, hl₁x, hxr₁, _, hδ₁, hbound₁⟩ :=
          exists_Ioo_lower_bound_half_of_continuous_pos
            (g := m) (a := x₀) (lower := α) (upper := x₀ + 1)
            hm_cont hαlt (lt_add_one x₀) hpos
        have hmpos_ext : ∀ t : ℝ, α < t → t < r₁ → 0 < m t := by
          intro t hαt htr
          rcases le_or_gt t x₀ with hle | hgt
          · exact hmid t hαt hle
          · have := hbound₁ t (lt_trans hl₁x hgt) htr; linarith
        have hvan := laplaceAlignmentDefect_eq_zero_on_Ioo_of_leftEdge_gen
          τ hτ p q hzero hμ1 (lt_trans hαlt hxr₁) hmα hmpos_ext
        exact hvan x₀ hαlt hxr₁
      · push Not at hS
        have hmray : ∀ t : ℝ, t ≤ x₀ → 0 < m t := by
          intro t htx
          rcases lt_trichotomy (m t) 0 with hlt | heq | hgt
          · exfalso
            have h0mem : (0 : ℝ) ∈ Icc (m t) (m x₀) := ⟨hlt.le, hpos.le⟩
            obtain ⟨z, hzIcc, hz0⟩ :=
              intermediate_value_Icc htx hm_cont.continuousOn h0mem
            exact hS z hzIcc.2 hz0
          · exact absurd heq (hS t htx)
          · exact hgt
        exact laplaceAlignmentDefect_eq_zero_on_left_ray_gen
          τ hτ p q hzero hμ1 hmray x₀ le_rfl

/-- **The unconditional 1-d Laplace converse.**  Zero raw Laplace mean-shift
drift identifies *arbitrary* probability measures on ℝ — no atomlessness, no
moment, no density.  This closes the arbitrary-target converse and subsumes
every previous coverage class. -/
theorem laplaceZeroDrift_identifies
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    p = q := by
  refine laplaceZeroDrift_imp_eq_of_companionAligned τ hτ p q hzero fun x => ?_
  have h := laplaceCompanionAlignmentDefect_eq_zero_of_zeroDrift_gen τ hτ p q hzero x
  unfold laplaceCompanionAlignmentDefect at h
  exact sub_eq_zero.mp h

/-- The unconditional identifiability condition: just being a pair of
probability measures.  (Bandwidth-free — identified for every valid `τ`.) -/
def LaplaceUnconditionalCondition (p q : Measure ℝ) : Prop :=
  IsProbabilityMeasure p ∧ IsProbabilityMeasure q

theorem laplaceZeroDrift_identifiesAtZero
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    IdentifiesAtZero
      LaplaceUnconditionalCondition
      (meanShiftDrift (laplaceKernel τ)) := by
  intro p q hcond hzero
  obtain ⟨hpProb, hqProb⟩ := hcond
  letI : IsProbabilityMeasure p := hpProb
  letI : IsProbabilityMeasure q := hqProb
  exact laplaceZeroDrift_identifies τ hτ p q hzero

theorem laplaceUnconditionalCondition_gaussianPair :
    LaplaceUnconditionalCondition
      (gaussianReal 0 (1 : NNReal)) (gaussianReal 1 (1 : NNReal)) :=
  ⟨inferInstance, inferInstance⟩

theorem laplaceUnconditionalCondition_allowsDistinctPair :
    ConditionAllowsDistinctPair LaplaceUnconditionalCondition :=
  ⟨gaussianReal 0 (1 : NNReal), gaussianReal 1 (1 : NNReal),
    laplaceUnconditionalCondition_gaussianPair,
    gaussianReal_zero_ne_one_unitVariance⟩

theorem laplaceUnconditionalCondition_isLegitimate :
    IsLegitimateCondition LaplaceUnconditionalCondition :=
  ⟨⟨gaussianReal 0 (1 : NNReal), gaussianReal 1 (1 : NNReal),
      laplaceUnconditionalCondition_gaussianPair⟩,
    laplaceUnconditionalCondition_allowsDistinctPair⟩

/-- The unconditional theorem subsumes the atomless one: no moment is needed. -/
theorem laplaceZeroDrift_identifies_of_noAtoms'
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    p = q :=
  laplaceZeroDrift_identifies τ hτ p q hzero

/-- Every previously identified pair is a pair of probability measures, so the
unconditional condition subsumes the atomless / real-converse conditions. -/
theorem laplaceUnconditionalCondition_of_atomlessCondition
    {p q : Measure ℝ} (h : LaplaceAtomlessCondition p q) :
    LaplaceUnconditionalCondition p q :=
  ⟨h.1, h.2.1⟩

end DriftingIdentifiability
