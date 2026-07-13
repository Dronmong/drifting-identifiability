import DriftingIdentifiability.LaplaceACFinal
import DriftingIdentifiability.LaplaceGeneralConverseCompanionWronskian

/-!
# The atomless 1-d Laplace converse via the alignment-defect coordinate

This file removes BOTH the density-continuity and the exponential-moment
hypotheses from the a.c. Laplace converse: zero raw Laplace mean-shift drift
identifies **atomless** probability measures on ℝ, assuming only a first
moment for `p`.  This covers general L¹ densities, singular-continuous laws,
and their mixtures — everything except atoms.

The engine is the companion alignment defect `K := L_p·Z_q − L_q·Z_p`
(`laplaceCompanionAlignmentDefect`), which — unlike the normalizer Wronskian —
has a purely FIRST-ORDER zero-drift structure:

* for atomless measures the raw normalizer `Z` is genuinely `C¹`
  (the certified one-sided derivative has a continuous coefficient once the
  one-sided exponential masses are continuous, which is exactly
  atomlessness), and `D`, `L` are `C¹` for any probability measure;
* differentiating `D_μ = m·Z_μ` once and substituting the certified
  `D' = (1/τ)L − 2Z` gives `L_μ = τ((m'+2)Z_μ + m·Z_μ')` pointwise, whence
  the two exact identities `K = τ·m·W` and `K' = −τ(m'+2)·W`;
* on `{m ≠ 0}` this is the first-order ODE `K' = −((m'+2)/m)·K`, holding
  everywhere, with continuous coefficient; L3 gives `m'+2 ≥ 1` globally, so
  the L8 blow-up lemmas apply at arbitrary zero-set edges with `δ = 1/2`;
* where `m = 0` the identity `K = τ·m·W` kills `K` pointwise; and `K → 0`
  at `±∞` for any finite measure (bounded kernels), so the L6 outer rays
  close the components reaching infinity.

The certified gate `laplaceZeroDrift_imp_eq_of_companionAligned` then gives
`p = q`.  Design record: `LaplaceACDerivation.md`, section
"ATOMLESS UPGRADE PLAN".
-/

open MeasureTheory Set Filter Topology ProbabilityTheory

namespace DriftingIdentifiability

open Paper

/-! ## Continuity of the one-sided exponential masses at atomless points -/

private theorem ae_ne_of_noAtoms (p : Measure ℝ) [NoAtoms p] (x : ℝ) :
    ∀ᵐ y ∂p, y ≠ x := by
  rw [ae_iff]
  have hset : {y : ℝ | ¬y ≠ x} = {x} := by
    ext y
    simp
  rw [hset]
  exact measure_singleton x

/-- The lower exponential mass is (two-sidedly) continuous for atomless
finite measures. -/
theorem lowerExpMass_continuousAt
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsFiniteMeasure p] [NoAtoms p] (x : ℝ) :
    ContinuousAt (fun t => lowerExpMass τ p t) x := by
  have hτ0 : 0 < τ := hτ
  unfold ContinuousAt lowerExpMass
  simp_rw [← integral_indicator measurableSet_Iic]
  refine tendsto_integral_filter_of_dominated_convergence
    (fun _ : ℝ => Real.exp ((x + 1) / τ)) ?_ ?_ (integrable_const _) ?_
  · refine Eventually.of_forall fun t => ?_
    exact ((Continuous.aestronglyMeasurable (by fun_prop)).indicator
      measurableSet_Iic)
  · filter_upwards [Ioo_mem_nhds (by linarith : x - 1 < x)
      (by linarith : x < x + 1)] with t ht
    refine ae_of_all p fun y => ?_
    by_cases hy : y ∈ Iic t
    · rw [indicator_of_mem hy, Real.norm_eq_abs,
        abs_of_nonneg (Real.exp_pos _).le]
      refine Real.exp_le_exp.mpr ?_
      have hy' : y ≤ x + 1 := le_trans (mem_Iic.mp hy) ht.2.le
      rw [div_eq_mul_inv, div_eq_mul_inv]
      exact mul_le_mul_of_nonneg_right hy' (inv_nonneg.mpr hτ0.le)
    · rw [indicator_of_notMem hy]
      simp [(Real.exp_pos _).le]
  · filter_upwards [ae_ne_of_noAtoms p x] with y hy
    rcases lt_or_gt_of_ne hy with hlt | hgt
    · have hev : ∀ᶠ t in 𝓝 x,
          indicator (Iic x) (fun z : ℝ => Real.exp (z / τ)) y
            = indicator (Iic t) (fun z : ℝ => Real.exp (z / τ)) y := by
        filter_upwards [Ioi_mem_nhds hlt] with t ht
        rw [indicator_of_mem (mem_Iic.mpr hlt.le),
          indicator_of_mem (mem_Iic.mpr (le_of_lt ht))]
      exact Filter.Tendsto.congr' hev tendsto_const_nhds
    · have hev : ∀ᶠ t in 𝓝 x,
          indicator (Iic x) (fun z : ℝ => Real.exp (z / τ)) y
            = indicator (Iic t) (fun z : ℝ => Real.exp (z / τ)) y := by
        filter_upwards [Iio_mem_nhds hgt] with t ht
        rw [indicator_of_notMem (by simpa using not_le.mpr hgt),
          indicator_of_notMem (by simpa using not_le.mpr ht)]
      exact Filter.Tendsto.congr' hev tendsto_const_nhds

/-- The upper exponential mass is (two-sidedly) continuous for atomless
finite measures. -/
theorem upperExpMass_continuousAt
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsFiniteMeasure p] [NoAtoms p] (x : ℝ) :
    ContinuousAt (fun t => upperExpMass τ p t) x := by
  have hτ0 : 0 < τ := hτ
  unfold ContinuousAt upperExpMass
  simp_rw [← integral_indicator measurableSet_Ioi]
  refine tendsto_integral_filter_of_dominated_convergence
    (fun _ : ℝ => Real.exp ((1 - x) / τ)) ?_ ?_ (integrable_const _) ?_
  · refine Eventually.of_forall fun t => ?_
    exact ((Continuous.aestronglyMeasurable (by fun_prop)).indicator
      measurableSet_Ioi)
  · filter_upwards [Ioo_mem_nhds (by linarith : x - 1 < x)
      (by linarith : x < x + 1)] with t ht
    refine ae_of_all p fun y => ?_
    by_cases hy : y ∈ Ioi t
    · rw [indicator_of_mem hy, Real.norm_eq_abs,
        abs_of_nonneg (Real.exp_pos _).le]
      refine Real.exp_le_exp.mpr ?_
      have hy' : -y ≤ 1 - x := by
        have h1 : x - 1 < y := lt_trans ht.1 (mem_Ioi.mp hy)
        linarith
      rw [div_eq_mul_inv, div_eq_mul_inv]
      exact mul_le_mul_of_nonneg_right hy' (inv_nonneg.mpr hτ0.le)
    · rw [indicator_of_notMem hy]
      simp [(Real.exp_pos _).le]
  · filter_upwards [ae_ne_of_noAtoms p x] with y hy
    rcases lt_or_gt_of_ne hy with hlt | hgt
    · have hev : ∀ᶠ t in 𝓝 x,
          indicator (Ioi x) (fun z : ℝ => Real.exp (-z / τ)) y
            = indicator (Ioi t) (fun z : ℝ => Real.exp (-z / τ)) y := by
        filter_upwards [Ioi_mem_nhds hlt] with t ht
        rw [indicator_of_notMem (by simpa using not_lt.mpr hlt.le),
          indicator_of_notMem (by simpa using not_lt.mpr ht.le)]
      exact Filter.Tendsto.congr' hev tendsto_const_nhds
    · have hev : ∀ᶠ t in 𝓝 x,
          indicator (Ioi x) (fun z : ℝ => Real.exp (-z / τ)) y
            = indicator (Ioi t) (fun z : ℝ => Real.exp (-z / τ)) y := by
        filter_upwards [Iio_mem_nhds hgt] with t ht
        rw [indicator_of_mem (mem_Ioi.mpr hgt),
          indicator_of_mem (mem_Ioi.mpr ht)]
      exact Filter.Tendsto.congr' hev tendsto_const_nhds

/-- For atomless finite measures the certified one-sided normalizer
derivative coefficient is continuous. -/
theorem laplaceKernelNormalizerRightDerivCoeff_continuous
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsFiniteMeasure p] [NoAtoms p] :
    Continuous (laplaceKernelNormalizerRightDerivCoeff τ p) := by
  have hlower : Continuous (fun t => lowerExpMass τ p t) :=
    continuous_iff_continuousAt.mpr fun x => lowerExpMass_continuousAt τ hτ p x
  have hupper : Continuous (fun t => upperExpMass τ p t) :=
    continuous_iff_continuousAt.mpr fun x => upperExpMass_continuousAt τ hτ p x
  unfold laplaceKernelNormalizerRightDerivCoeff
  have hexp₁ : Continuous fun t : ℝ => Real.exp (-t / τ) := by fun_prop
  have hexp₂ : Continuous fun t : ℝ => Real.exp (t / τ) := by fun_prop
  exact continuous_const.mul ((hexp₁.neg.mul hlower).add (hexp₂.mul hupper))

/-! ## Right-derivative-to-derivative upgrade -/

/-- A continuous function with an everywhere right-sided derivative given by a
continuous coefficient is genuinely differentiable with that derivative. -/
theorem hasDerivAt_of_continuous_of_hasDerivWithinAt_Ici
    {f g : ℝ → ℝ} (hf : Continuous f) (hg : Continuous g)
    (h : ∀ t : ℝ, HasDerivWithinAt f (g t) (Ici t) t) (x : ℝ) :
    HasDerivAt f (g x) x := by
  set P : ℝ → ℝ := fun t => ∫ s in (x - 1)..t, g s with hPdef
  have hPderiv : ∀ t : ℝ, HasDerivAt P (g t) t := by
    intro t
    exact intervalIntegral.integral_hasDerivAt_right
      (hg.intervalIntegrable _ _)
      hg.stronglyMeasurable.stronglyMeasurableAtFilter
      hg.continuousAt
  have hPcont : Continuous P :=
    continuous_iff_continuousAt.mpr fun t => (hPderiv t).continuousAt
  have hconst : ∀ t ∈ Icc (x - 1) (x + 1),
      (fun s : ℝ => f s - P s) t = (fun s : ℝ => f s - P s) (x - 1) := by
    refine constant_of_has_deriv_right_zero ((hf.sub hPcont).continuousOn) ?_
    intro t _
    have hsub := (h t).sub (hPderiv t).hasDerivWithinAt
    rw [sub_self] at hsub
    exact hsub
  have hev : f =ᶠ[𝓝 x] fun t => P t + (f (x - 1) - P (x - 1)) := by
    filter_upwards [Ioo_mem_nhds (by linarith : x - 1 < x)
      (by linarith : x < x + 1)] with t ht
    have := hconst t ⟨ht.1.le, ht.2.le⟩
    simp only at this
    linarith
  have hsum : HasDerivAt (fun t => P t + (f (x - 1) - P (x - 1))) (g x) x := by
    simpa using (hPderiv x).add_const (f (x - 1) - P (x - 1))
  exact hsum.congr_of_eventuallyEq hev

/-- **`C¹` raw normalizer for atomless measures.**  The Laplace kernel
normalizer of an atomless finite measure is genuinely differentiable
everywhere, with derivative the certified one-sided coefficient. -/
theorem hasDerivAt_laplaceKernelNormalizer_of_noAtoms
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsFiniteMeasure p] [NoAtoms p] (x : ℝ) :
    HasDerivAt (fun t => kernelNormalizer (laplaceKernel τ) p t)
      (laplaceKernelNormalizerRightDerivCoeff τ p x) x :=
  hasDerivAt_of_continuous_of_hasDerivWithinAt_Ici
    (continuous_laplaceKernelNormalizer τ hτ p)
    (laplaceKernelNormalizerRightDerivCoeff_continuous τ hτ p)
    (fun t => hasDerivWithinAt_Ici_laplaceKernelNormalizer τ hτ p t) x

/-! ## The mean-shift ratio is `C¹` for atomless measures -/

/-- Atomless analogue of `hasDerivAt_laplaceMeanShiftRatio_regular`: no
density regularity is needed. -/
theorem hasDerivAt_laplaceMeanShiftRatio_of_noAtoms
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] [NoAtoms p] (x : ℝ) :
    HasDerivAt (laplaceMeanShiftRatio τ p)
      (laplaceMeanShiftRatioDeriv τ p x) x := by
  have hD := hasDerivAt_laplaceMeanShiftNumerator τ hτ p x
  have hZ := hasDerivAt_laplaceKernelNormalizer_of_noAtoms τ hτ p x
  have hZne : kernelNormalizer (laplaceKernel τ) p x ≠ 0 :=
    (laplaceKernelNormalizer_pos p τ hτ x).ne'
  have hdiv := hD.div hZ hZne
  change HasDerivAt
    (laplaceMeanShiftNumerator τ p /
      fun s : ℝ => kernelNormalizer (laplaceKernel τ) p s)
    (laplaceMeanShiftRatioDeriv τ p x) x
  simpa [laplaceMeanShiftRatioDeriv, laplaceMeanShiftRatioDerivNumerator,
    laplaceMeanShiftRatioDerivDenominator, laplaceMeanShiftNumerator] using hdiv

/-- The mean-shift ratio derivative is continuous for atomless measures. -/
theorem laplaceMeanShiftRatioDeriv_continuous_of_noAtoms
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] [NoAtoms p] :
    Continuous (laplaceMeanShiftRatioDeriv τ p) := by
  have hZcont : Continuous (fun t => kernelNormalizer (laplaceKernel τ) p t) :=
    continuous_laplaceKernelNormalizer τ hτ p
  have hLc : Continuous (kernelNormalizer (laplaceCompanionKernel τ) p) :=
    continuous_iff_continuousAt.mpr fun x =>
      (hasDerivAt_laplaceCompanionNormalizer τ hτ p x).continuousAt
  have hDcont : Continuous (laplaceMeanShiftNumerator τ p) :=
    continuous_iff_continuousAt.mpr fun x =>
      (hasDerivAt_laplaceMeanShiftNumerator τ hτ p x).continuousAt
  have hrdc := laplaceKernelNormalizerRightDerivCoeff_continuous τ hτ p
  have hZne : ∀ x : ℝ, kernelNormalizer (laplaceKernel τ) p x ≠ 0 := fun x =>
    (laplaceKernelNormalizer_pos p τ hτ x).ne'
  unfold laplaceMeanShiftRatioDeriv laplaceMeanShiftRatioDerivNumerator
    laplaceMeanShiftRatioDerivDenominator laplaceMeanShiftNumeratorDeriv
  refine Continuous.div ?_ (hZcont.pow 2)
    (fun x => pow_ne_zero 2 (hZne x))
  exact (((continuous_const.mul hLc).sub (continuous_const.mul hZcont)).mul
    hZcont).sub (hDcont.mul hrdc)

/-- Atomless analogue of the tilted-mean derivative wiring. -/
theorem hasDerivAt_laplaceTiltedMean_of_noAtoms
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] [NoAtoms p] (x : ℝ) :
    HasDerivAt (laplaceTiltedMean τ p)
      (laplaceMeanShiftRatioDeriv τ p x + 1) x := by
  have hratio := hasDerivAt_laplaceMeanShiftRatio_of_noAtoms τ hτ p x
  have hdisp : HasDerivAt (laplaceTiltedMeanFromDisplacement τ p)
      (1 + laplaceMeanShiftRatioDeriv τ p x) x := by
    have hsum : HasDerivAt (fun t : ℝ => t + laplaceMeanShiftRatio τ p t)
        (1 + laplaceMeanShiftRatioDeriv τ p x) x :=
      (hasDerivAt_id x).add hratio
    change HasDerivAt
      (fun t : ℝ => t + (∫ y, laplaceWeightedDisplacement τ t y ∂p) /
        kernelNormalizer (laplaceKernel τ) p t)
      (1 + laplaceMeanShiftRatioDeriv τ p x) x
    simpa [laplaceMeanShiftRatio] using hsum
  have heq : laplaceTiltedMean τ p = laplaceTiltedMeanFromDisplacement τ p := by
    funext y
    exact laplaceTiltedMean_eq_fromDisplacement τ hτ p y
  simpa [heq, add_comm] using hdisp

/-- **L3 bridge, atomless form.**  Monotonicity of the tilted mean gives
`m' + 1 ≥ 0` everywhere, hence `m' + 2 ≥ 1` — the global coefficient bound
used by the alignment-defect blow-up argument. -/
theorem laplaceMeanShiftRatioDeriv_add_one_nonneg_of_noAtoms
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] [NoAtoms p]
    (hint : Integrable (fun y : ℝ => y) p) :
    ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1 := by
  intro t
  exact (hasDerivAt_laplaceTiltedMean_of_noAtoms τ hτ p t).nonneg_of_monotone
    (laplaceTiltedMean_monotone hτ p hint)

/-! ## First-order zero-drift structure of the alignment defect -/

/-- Pointwise companion-normalizer formula from the common mean-shift
relation: `L_μ = τ((m'+2)·Z_μ + m·Z_μ')`, using only certified first-order
identities and the atomless `C¹` normalizer. -/
private theorem laplaceCompanionNormalizer_formula
    (τ : ℝ) (hτ : ValidBandwidth τ) (p ν : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure ν]
    [NoAtoms p] [NoAtoms ν]
    (hcommon : ∀ t : ℝ, (∫ y, laplaceWeightedDisplacement τ t y ∂ν) =
      laplaceMeanShiftRatio τ p t * kernelNormalizer (laplaceKernel τ) ν t)
    (x : ℝ) :
    kernelNormalizer (laplaceCompanionKernel τ) ν x =
      τ * ((laplaceMeanShiftRatioDeriv τ p x + 2) *
          kernelNormalizer (laplaceKernel τ) ν x +
        laplaceMeanShiftRatio τ p x *
          laplaceKernelNormalizerRightDerivCoeff τ ν x) := by
  have hm := hasDerivAt_laplaceMeanShiftRatio_of_noAtoms τ hτ p x
  have hZ := hasDerivAt_laplaceKernelNormalizer_of_noAtoms τ hτ ν x
  have hprod := hm.mul hZ
  have hD2 : HasDerivAt
      (fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂ν)
      (laplaceMeanShiftRatioDeriv τ p x *
          kernelNormalizer (laplaceKernel τ) ν x +
        laplaceMeanShiftRatio τ p x *
          laplaceKernelNormalizerRightDerivCoeff τ ν x) x := by
    refine hprod.congr_of_eventuallyEq ?_
    exact Eventually.of_forall fun t => by
      simpa [Pi.mul_apply] using hcommon t
  have hD1 := hasDerivAt_laplaceDisplacementIntegral τ hτ ν x
  have hval := hD1.unique hD2
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

/-- **`K = τ·m·W` pointwise under zero drift.**  In particular the alignment
defect vanishes at every zero of the mean shift. -/
theorem laplaceCompanionAlignmentDefect_eq_of_zeroDrift
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    [NoAtoms p] [NoAtoms q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ) :
    laplaceCompanionAlignmentDefect τ p q x =
      τ * laplaceMeanShiftRatio τ p x *
        laplaceKernelNormalizerWronskian τ p q x := by
  have hp := laplaceCompanionNormalizer_formula τ hτ p p
    (fun t => laplaceMeanShiftRatio_common_self τ hτ p t) x
  have hq := laplaceCompanionNormalizer_formula τ hτ p q
    (fun t => laplaceMeanShiftRatio_common_of_zeroDrift τ hτ p q hzero t) x
  unfold laplaceCompanionAlignmentDefect laplaceKernelNormalizerWronskian
  rw [hp, hq]
  ring

/-- **`K' = −τ(m'+2)·W` everywhere under zero drift** — the alignment defect
is `C¹` with a purely first-order derivative formula. -/
theorem hasDerivAt_laplaceCompanionAlignmentDefect_of_zeroDrift
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    [NoAtoms p] [NoAtoms q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ) :
    HasDerivAt (fun t => laplaceCompanionAlignmentDefect τ p q t)
      (-(τ * (laplaceMeanShiftRatioDeriv τ p x + 2)) *
        laplaceKernelNormalizerWronskian τ p q x) x := by
  have hLp := hasDerivAt_laplaceCompanionNormalizer τ hτ p x
  have hLq := hasDerivAt_laplaceCompanionNormalizer τ hτ q x
  have hZp := hasDerivAt_laplaceKernelNormalizer_of_noAtoms τ hτ p x
  have hZq := hasDerivAt_laplaceKernelNormalizer_of_noAtoms τ hτ q x
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
      laplaceCompanionNormalizer_formula τ hτ p p
        (fun t => laplaceMeanShiftRatio_common_self τ hτ p t) x,
      laplaceCompanionNormalizer_formula τ hτ p q
        (fun t => laplaceMeanShiftRatio_common_of_zeroDrift τ hτ p q hzero t) x]
    unfold laplaceKernelNormalizerWronskian
    ring
  rw [hval] at hK
  exact hK.congr_of_eventuallyEq (Eventually.of_forall fun t => rfl)

/-- The Abel-shaped form on `{m ≠ 0}`: the alignment defect satisfies the
first-order ODE `K' = −(2·((m'+2)/2)/m)·K` — exactly the coefficient shape
consumed by the abstract L6/L8 propagation layer. -/
theorem hasDerivAt_laplaceCompanionAlignmentDefect_of_ne
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    [NoAtoms p] [NoAtoms q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ)
    (hmx : laplaceMeanShiftRatio τ p x ≠ 0) :
    HasDerivAt (fun t => laplaceCompanionAlignmentDefect τ p q t)
      (-(2 * ((laplaceMeanShiftRatioDeriv τ p x + 2) / 2) /
          laplaceMeanShiftRatio τ p x) *
        laplaceCompanionAlignmentDefect τ p q x) x := by
  have h := hasDerivAt_laplaceCompanionAlignmentDefect_of_zeroDrift
    τ hτ p q hzero x
  have heq :
      -(τ * (laplaceMeanShiftRatioDeriv τ p x + 2)) *
          laplaceKernelNormalizerWronskian τ p q x =
        -(2 * ((laplaceMeanShiftRatioDeriv τ p x + 2) / 2) /
            laplaceMeanShiftRatio τ p x) *
          laplaceCompanionAlignmentDefect τ p q x := by
    rw [laplaceCompanionAlignmentDefect_eq_of_zeroDrift τ hτ p q hzero x]
    field_simp
  rw [heq] at h
  exact h

/-- The alignment defect is continuous (any probability measures). -/
theorem continuous_laplaceCompanionAlignmentDefect
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] :
    Continuous (fun x => laplaceCompanionAlignmentDefect τ p q x) := by
  have hLp : Continuous (kernelNormalizer (laplaceCompanionKernel τ) p) :=
    continuous_iff_continuousAt.mpr fun x =>
      (hasDerivAt_laplaceCompanionNormalizer τ hτ p x).continuousAt
  have hLq : Continuous (kernelNormalizer (laplaceCompanionKernel τ) q) :=
    continuous_iff_continuousAt.mpr fun x =>
      (hasDerivAt_laplaceCompanionNormalizer τ hτ q x).continuousAt
  have hZp := continuous_laplaceKernelNormalizer τ hτ p
  have hZq := continuous_laplaceKernelNormalizer τ hτ q
  unfold laplaceCompanionAlignmentDefect
  exact (hLp.mul hZq).sub (hLq.mul hZp)

/-! ## Tails: the alignment defect vanishes at infinity (no moments) -/

private theorem abs_sub_tendsto_atTop (y : ℝ) :
    Tendsto (fun x : ℝ => ‖x - y‖) atTop atTop := by
  simp_rw [Real.norm_eq_abs]
  refine tendsto_atTop_mono (fun x => le_abs_self (x - y)) ?_
  exact (tendsto_atTop_add_const_right atTop (-y) tendsto_id).congr
    fun x => by simp only [id_eq]; ring

private theorem abs_sub_tendsto_atBot (y : ℝ) :
    Tendsto (fun x : ℝ => ‖x - y‖) atBot atTop := by
  simp_rw [Real.norm_eq_abs]
  refine tendsto_atTop_mono (fun x => neg_le_abs (x - y)) ?_
  have h1 : Tendsto (fun x : ℝ => -x + y) atBot atTop :=
    tendsto_atTop_add_const_right atBot y tendsto_neg_atBot_atTop
  exact h1.congr fun x => by ring

private theorem laplaceKernel_comp_tendsto_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) {l : Filter ℝ} (y : ℝ)
    (habs : Tendsto (fun x : ℝ => ‖x - y‖) l atTop) :
    Tendsto (fun x : ℝ => laplaceKernel τ x y) l (𝓝 0) := by
  have hτ0 : 0 < τ := hτ
  have harg : Tendsto (fun x : ℝ => -(1 / τ) * ‖x - y‖) l atBot := by
    have hpos : Tendsto (fun x : ℝ => (1 / τ) * ‖x - y‖) l atTop :=
      habs.const_mul_atTop (one_div_pos.mpr hτ0)
    have hneg := tendsto_neg_atTop_atBot.comp hpos
    exact hneg.congr fun x => by
      simp only [Function.comp_apply]
      ring
  exact (Real.tendsto_exp_atBot.comp harg).congr fun x => rfl

private theorem laplaceCompanionKernel_comp_tendsto_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) {l : Filter ℝ} (y : ℝ)
    (habs : Tendsto (fun x : ℝ => ‖x - y‖) l atTop) :
    Tendsto (fun x : ℝ => laplaceCompanionKernel τ x y) l (𝓝 0) := by
  have hτ0 : 0 < τ := hτ
  have hτne : (τ : ℝ) ≠ 0 := hτ0.ne'
  have hu : Tendsto (fun x : ℝ => ‖x - y‖ / τ) l atTop :=
    habs.atTop_div_const hτ0
  have h₁ : Tendsto (fun u : ℝ => Real.exp (-u)) atTop (𝓝 0) :=
    Real.tendsto_exp_atBot.comp tendsto_neg_atTop_atBot
  have h₂ : Tendsto (fun u : ℝ => u * Real.exp (-u)) atTop (𝓝 0) := by
    simpa using Real.tendsto_pow_mul_exp_neg_atTop_nhds_zero 1
  have hcomb : Tendsto
      (fun u : ℝ => τ * Real.exp (-u) + τ * (u * Real.exp (-u)))
      atTop (𝓝 0) := by
    have := (h₁.const_mul τ).add (h₂.const_mul τ)
    simpa using this
  refine (hcomb.comp hu).congr fun x => ?_
  simp only [Function.comp_apply]
  unfold laplaceCompanionKernel laplaceKernel
  rw [show -(1 / τ) * ‖x - y‖ = -(‖x - y‖ / τ) from by ring]
  have h3 : τ * (‖x - y‖ / τ) = ‖x - y‖ := by
    rw [mul_comm, div_mul_cancel₀ _ hτne]
  calc τ * Real.exp (-(‖x - y‖ / τ)) +
        τ * (‖x - y‖ / τ * Real.exp (-(‖x - y‖ / τ)))
      = (τ + τ * (‖x - y‖ / τ)) * Real.exp (-(‖x - y‖ / τ)) := by ring
    _ = (τ + ‖x - y‖) * Real.exp (-(‖x - y‖ / τ)) := by rw [h3]

private theorem laplaceKernelNormalizer_tendsto_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p]
    {l : Filter ℝ} [l.IsCountablyGenerated]
    (habs : ∀ y : ℝ, Tendsto (fun x : ℝ => ‖x - y‖) l atTop) :
    Tendsto (fun x => kernelNormalizer (laplaceKernel τ) p x) l (𝓝 0) := by
  have hτ0 : 0 < τ := hτ
  have h : Tendsto (fun x => ∫ y, laplaceKernel τ x y ∂p) l
      (𝓝 (∫ _ : ℝ, (0 : ℝ) ∂p)) := by
    refine tendsto_integral_filter_of_dominated_convergence
      (fun _ : ℝ => (1 : ℝ)) ?_ ?_ (integrable_const _) ?_
    · refine Eventually.of_forall fun x => ?_
      refine Continuous.aestronglyMeasurable ?_
      unfold laplaceKernel
      fun_prop
    · refine Eventually.of_forall fun x => ae_of_all p fun y => ?_
      unfold laplaceKernel
      rw [Real.norm_eq_abs, abs_of_nonneg (Real.exp_pos _).le]
      refine Real.exp_le_one_iff.mpr ?_
      have h1 : (0 : ℝ) ≤ (1 / τ) * ‖x - y‖ :=
        mul_nonneg (one_div_nonneg.mpr hτ0.le) (norm_nonneg _)
      linarith
    · exact ae_of_all p fun y => laplaceKernel_comp_tendsto_zero τ hτ y (habs y)
  unfold kernelNormalizer
  simpa using h

private theorem laplaceCompanionNormalizer_tendsto_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p]
    {l : Filter ℝ} [l.IsCountablyGenerated]
    (habs : ∀ y : ℝ, Tendsto (fun x : ℝ => ‖x - y‖) l atTop) :
    Tendsto (fun x => kernelNormalizer (laplaceCompanionKernel τ) p x)
      l (𝓝 0) := by
  have hτ0 : 0 < τ := hτ
  have hτne : (τ : ℝ) ≠ 0 := hτ0.ne'
  have h : Tendsto (fun x => ∫ y, laplaceCompanionKernel τ x y ∂p) l
      (𝓝 (∫ _ : ℝ, (0 : ℝ) ∂p)) := by
    refine tendsto_integral_filter_of_dominated_convergence
      (fun _ : ℝ => τ) ?_ ?_ (integrable_const _) ?_
    · refine Eventually.of_forall fun x => ?_
      refine Continuous.aestronglyMeasurable ?_
      unfold laplaceCompanionKernel laplaceKernel
      fun_prop
    · refine Eventually.of_forall fun x => ae_of_all p fun y => ?_
      have hr : (0 : ℝ) ≤ ‖x - y‖ := norm_nonneg _
      have hker_pos : (0 : ℝ) < Real.exp (-(1 / τ) * ‖x - y‖) := Real.exp_pos _
      have hcomp_nonneg : (0 : ℝ) ≤ laplaceCompanionKernel τ x y := by
        unfold laplaceCompanionKernel laplaceKernel
        exact mul_nonneg (by linarith) hker_pos.le
      rw [Real.norm_eq_abs, abs_of_nonneg hcomp_nonneg]
      unfold laplaceCompanionKernel laplaceKernel
      have hkey : τ + ‖x - y‖ ≤ τ * Real.exp (‖x - y‖ / τ) := by
        have hexp := Real.add_one_le_exp (‖x - y‖ / τ)
        have h2 : τ * (‖x - y‖ / τ + 1) ≤ τ * Real.exp (‖x - y‖ / τ) :=
          mul_le_mul_of_nonneg_left hexp hτ0.le
        have h3 : τ * (‖x - y‖ / τ + 1) = τ + ‖x - y‖ := by
          rw [mul_add, mul_one, mul_comm τ (‖x - y‖ / τ),
            div_mul_cancel₀ _ hτne]
          ring
        linarith
      rw [show -(1 / τ) * ‖x - y‖ = -(‖x - y‖ / τ) from by ring,
        Real.exp_neg, ← div_eq_mul_inv,
        div_le_iff₀ (Real.exp_pos _)]
      exact hkey
    · exact ae_of_all p fun y =>
        laplaceCompanionKernel_comp_tendsto_zero τ hτ y (habs y)
  unfold kernelNormalizer
  simpa using h

/-- The alignment defect tends to zero at `+∞` — with no moment hypotheses. -/
theorem laplaceCompanionAlignmentDefect_tendsto_atTop_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] :
    Tendsto (fun x => laplaceCompanionAlignmentDefect τ p q x)
      atTop (𝓝 0) := by
  have hZq := laplaceKernelNormalizer_tendsto_zero τ hτ q abs_sub_tendsto_atTop
  have hZp := laplaceKernelNormalizer_tendsto_zero τ hτ p abs_sub_tendsto_atTop
  have hLp := laplaceCompanionNormalizer_tendsto_zero τ hτ p
    abs_sub_tendsto_atTop
  have hLq := laplaceCompanionNormalizer_tendsto_zero τ hτ q
    abs_sub_tendsto_atTop
  unfold laplaceCompanionAlignmentDefect
  simpa using (hLp.mul hZq).sub (hLq.mul hZp)

/-- The alignment defect tends to zero at `−∞` — with no moment hypotheses. -/
theorem laplaceCompanionAlignmentDefect_tendsto_atBot_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] :
    Tendsto (fun x => laplaceCompanionAlignmentDefect τ p q x)
      atBot (𝓝 0) := by
  have hZq := laplaceKernelNormalizer_tendsto_zero τ hτ q abs_sub_tendsto_atBot
  have hZp := laplaceKernelNormalizer_tendsto_zero τ hτ p abs_sub_tendsto_atBot
  have hLp := laplaceCompanionNormalizer_tendsto_zero τ hτ p
    abs_sub_tendsto_atBot
  have hLq := laplaceCompanionNormalizer_tendsto_zero τ hτ q
    abs_sub_tendsto_atBot
  unfold laplaceCompanionAlignmentDefect
  simpa using (hLp.mul hZq).sub (hLq.mul hZp)

/-! ## Propagation: edges, rays, assembly -/

/-- Left-edge blow-up for the alignment defect: if the mean shift vanishes at
`a` and is positive on `(a, b)`, the defect vanishes on `(a, b)`.  The `δ`
lower bound is global (`(m'+2)/2 ≥ 1/2` from L3), so only the linear bound
on `m` at the edge is extracted locally. -/
theorem laplaceAlignmentDefect_eq_zero_on_Ioo_of_leftEdge
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    [NoAtoms p] [NoAtoms q]
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
  have hm_all : ∀ t : ℝ, HasDerivAt m (laplaceMeanShiftRatioDeriv τ p t) t :=
    fun t => hasDerivAt_laplaceMeanShiftRatio_of_noAtoms τ hτ p t
  have hμ_cont : Continuous μDeriv :=
    ((laplaceMeanShiftRatioDeriv_continuous_of_noAtoms τ hτ p).add
      continuous_const).div_const 2
  have hm_cont : Continuous m := by
    rw [continuous_iff_continuousAt]
    intro x
    exact (hm_all x).continuousAt
  have hKcont : Continuous K :=
    continuous_laplaceCompanionAlignmentDefect τ hτ p q
  obtain ⟨_, r₂, L, _, _, har₂, hr₂b, hL, _, hlin⟩ :=
    exists_Ioo_linear_bound_of_hasDerivAt_zero (f := m) (a := a)
      (lower := a - 1) (upper := b) (by linarith) hab (hm_all a)
      (by simpa [m] using hma)
  have hc_cont : ContinuousOn c (Ioo a b) := by
    dsimp [c]
    exact ((continuous_const.mul hμ_cont).continuousOn).div hm_cont.continuousOn
      (fun t ht => (hmpos t ht.1 ht.2).ne')
  have hvanish : ∀ x : ℝ, a < x → x < b → K x = 0 := by
    refine abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper
      (W := K) (A := fun z : ℝ => ∫ s in r₂..z, c s)
      (μDeriv := μDeriv) (m := m)
      (a := a) (b := b) (r := r₂) (δ := 1 / 2) (L := L)
      hab har₂ hr₂b (by norm_num) hL ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
    · intro x y hx hxy hy
      refine hKcont.continuousOn.mul
        (Real.continuous_exp.comp_continuousOn ?_)
      exact intervalPrimitive_continuousOn_Icc_of_continuousOn_Ioo
        (c := c) (a := a) (b := b) (base := r₂) (x := x) (y := y)
        hc_cont ⟨har₂, hr₂b⟩ ⟨hx, lt_of_le_of_lt hxy hy⟩
        ⟨lt_of_lt_of_le hx hxy, hy⟩
    · intro x y hx hxy hy t ht
      have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
        (hmpos t (lt_of_lt_of_le hx ht.1) (lt_trans ht.2 hy)).ne'
      have hderiv :=
        hasDerivAt_laplaceCompanionAlignmentDefect_of_ne
          τ hτ p q hzero t hmt
      simpa [K, μDeriv, m] using hderiv.hasDerivWithinAt
    · intro x y hx hxy hy t ht
      have htGap : t ∈ Ioo a b :=
        ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
      exact intervalPrimitive_hasDerivWithinAt_Ici_of_continuousOn_Ioo
        (c := c) (a := a) (b := b) (base := r₂) (t := t)
        hc_cont ⟨har₂, hr₂b⟩ htGap
    · intro x y hx hxy hy
      exact intervalPrimitive_continuousOn_Icc_of_continuousOn_Ioo
        (c := c) (a := a) (b := b) (base := r₂) (x := x) (y := y)
        hc_cont ⟨har₂, hr₂b⟩ ⟨hx, lt_of_le_of_lt hxy hy⟩
        ⟨lt_of_lt_of_le hx hxy, hy⟩
    · exact (hKcont.continuousAt.norm.isBoundedUnder_le).mono
        nhdsWithin_le_nhds
    · intro t _ _
      have := hμ1 t
      dsimp [μDeriv]
      linarith
    · intro t hat htr
      exact hmpos t hat (lt_trans htr hr₂b)
    · intro t hat htr
      exact hlin t hat htr.le
  exact hvanish

/-- Right-edge blow-up for the alignment defect (mirror). -/
theorem laplaceAlignmentDefect_eq_zero_on_Ioo_of_rightEdge
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    [NoAtoms p] [NoAtoms q]
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
  have hm_all : ∀ t : ℝ, HasDerivAt m (laplaceMeanShiftRatioDeriv τ p t) t :=
    fun t => hasDerivAt_laplaceMeanShiftRatio_of_noAtoms τ hτ p t
  have hμ_cont : Continuous μDeriv :=
    ((laplaceMeanShiftRatioDeriv_continuous_of_noAtoms τ hτ p).add
      continuous_const).div_const 2
  have hm_cont : Continuous m := by
    rw [continuous_iff_continuousAt]
    intro x
    exact (hm_all x).continuousAt
  have hKcont : Continuous K :=
    continuous_laplaceCompanionAlignmentDefect τ hτ p q
  obtain ⟨l₂, _, L, hal₂, hl₂b, _, _, hL, hlin, _⟩ :=
    exists_Ioo_linear_bound_of_hasDerivAt_zero (f := m) (a := b)
      (lower := a) (upper := b + 1) hab (lt_add_one b) (hm_all b)
      (by simpa [m] using hmb)
  have hc_cont : ContinuousOn c (Ioo a b) := by
    dsimp [c]
    exact ((continuous_const.mul hμ_cont).continuousOn).div hm_cont.continuousOn
      (fun t ht => (hmneg t ht.1 ht.2).ne)
  have hvanish : ∀ x : ℝ, a < x → x < b → K x = 0 := by
    refine abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower
      (W := K) (A := fun z : ℝ => ∫ s in l₂..z, c s)
      (μDeriv := μDeriv) (m := m)
      (a := a) (b := b) (l := l₂) (δ := 1 / 2) (L := L)
      hab hal₂ hl₂b (by norm_num) hL ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
    · intro x y hx hxy hy
      refine hKcont.continuousOn.mul
        (Real.continuous_exp.comp_continuousOn ?_)
      exact intervalPrimitive_continuousOn_Icc_of_continuousOn_Ioo
        (c := c) (a := a) (b := b) (base := l₂) (x := x) (y := y)
        hc_cont ⟨hal₂, hl₂b⟩ ⟨hx, lt_of_le_of_lt hxy hy⟩
        ⟨lt_of_lt_of_le hx hxy, hy⟩
    · intro x y hx hxy hy t ht
      have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
        (hmneg t (lt_of_lt_of_le hx ht.1) (lt_trans ht.2 hy)).ne
      have hderiv :=
        hasDerivAt_laplaceCompanionAlignmentDefect_of_ne
          τ hτ p q hzero t hmt
      simpa [K, μDeriv, m] using hderiv.hasDerivWithinAt
    · intro x y hx hxy hy t ht
      have htGap : t ∈ Ioo a b :=
        ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
      exact intervalPrimitive_hasDerivWithinAt_Ici_of_continuousOn_Ioo
        (c := c) (a := a) (b := b) (base := l₂) (t := t)
        hc_cont ⟨hal₂, hl₂b⟩ htGap
    · intro x y hx hxy hy
      exact intervalPrimitive_continuousOn_Icc_of_continuousOn_Ioo
        (c := c) (a := a) (b := b) (base := l₂) (x := x) (y := y)
        hc_cont ⟨hal₂, hl₂b⟩ ⟨hx, lt_of_le_of_lt hxy hy⟩
        ⟨lt_of_lt_of_le hx hxy, hy⟩
    · exact (hKcont.continuousAt.norm.isBoundedUnder_le).mono
        nhdsWithin_le_nhds
    · intro t _ _
      have := hμ1 t
      dsimp [μDeriv]
      linarith
    · intro t hlt htb
      exact hmneg t (lt_of_lt_of_le hal₂ hlt) htb
    · intro t hlt htb
      exact hlin t hlt htb
  exact hvanish

/-- Left outer ray for the alignment defect. -/
theorem laplaceAlignmentDefect_eq_zero_on_left_ray
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    [NoAtoms p] [NoAtoms q]
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
    have hderiv :=
      hasDerivAt_laplaceCompanionAlignmentDefect_of_ne τ hτ p q hzero t hmt
    simpa using hderiv.hasDerivWithinAt
  · intro t _
    have := hμ1 t
    linarith
  · intro t ht
    exact hmpos t ht

/-- Right outer ray for the alignment defect. -/
theorem laplaceAlignmentDefect_eq_zero_on_right_ray
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    [NoAtoms p] [NoAtoms q]
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
    have hderiv :=
      hasDerivAt_laplaceCompanionAlignmentDefect_of_ne τ hτ p q hzero t hmt
    simpa using hderiv.hasDerivWithinAt
  · intro t _
    have := hμ1 t
    linarith
  · intro t ht
    exact hmneg t ht

/-- **The alignment defect vanishes identically under zero drift for atomless
laws** — with NO hypothesis on the mean-shift zero set and NO moment
hypotheses beyond the `p` first moment (used only for L3 monotonicity). -/
theorem laplaceCompanionAlignmentDefect_eq_zero_of_zeroDrift
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    [NoAtoms p] [NoAtoms q]
    (hpFirst : Integrable (fun y : ℝ => y) p)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    ∀ x : ℝ, laplaceCompanionAlignmentDefect τ p q x = 0 := by
  let m : ℝ → ℝ := fun x => laplaceMeanShiftRatio τ p x
  have hμ1 : ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1 :=
    laplaceMeanShiftRatioDeriv_add_one_nonneg_of_noAtoms τ hτ p hpFirst
  have hm_cont : Continuous m := by
    rw [continuous_iff_continuousAt]
    intro x
    exact (hasDerivAt_laplaceMeanShiftRatio_of_noAtoms τ hτ p x).continuousAt
  intro x₀
  by_cases hmx₀ : m x₀ = 0
  · -- `K = τ·m·W` kills the zero set pointwise
    rw [laplaceCompanionAlignmentDefect_eq_of_zeroDrift τ hτ p q hzero x₀]
    have h0 : laplaceMeanShiftRatio τ p x₀ = 0 := hmx₀
    rw [h0]
    ring
  · rcases lt_or_gt_of_ne hmx₀ with hneg | hpos
    · -- `m(x₀) < 0`: look right
      by_cases hS : ∃ t : ℝ, x₀ ≤ t ∧ m t = 0
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
          · exfalso
            rw [← heq] at hmβ
            exact hneg.ne hmβ
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
          · have := hbound₁ t hlt (lt_trans hgt hxr₁)
            linarith
        have hvan := laplaceAlignmentDefect_eq_zero_on_Ioo_of_rightEdge
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
        exact laplaceAlignmentDefect_eq_zero_on_right_ray
          τ hτ p q hzero hμ1 hmray x₀ le_rfl
    · -- `m(x₀) > 0`: look left
      by_cases hS : ∃ t : ℝ, t ≤ x₀ ∧ m t = 0
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
          · exfalso
            rw [heq] at hmα
            exact hpos.ne' hmα
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
          · have := hbound₁ t (lt_trans hl₁x hgt) htr
            linarith
        have hvan := laplaceAlignmentDefect_eq_zero_on_Ioo_of_leftEdge
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
        exact laplaceAlignmentDefect_eq_zero_on_left_ray
          τ hτ p q hzero hμ1 hmray x₀ le_rfl

/-! ## Headline theorem, condition, legitimacy -/

/-- **Atomless 1-d Laplace converse.**  Zero raw Laplace mean-shift drift
identifies atomless probability measures on ℝ, assuming only a first moment
for `p`.  No density (let alone a continuous one), no exponential moments,
and no hypothesis on the mean-shift zero set.

This subsumes `laplaceAC_identifies_of_continuousDensity` (continuous
densities are absolutely continuous, hence atomless) and extends the
identifiable class to rough L¹ densities, singular-continuous laws, and
their mixtures. -/
theorem laplaceZeroDrift_identifies_of_noAtoms
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    [NoAtoms p] [NoAtoms q]
    (hpFirst : Integrable (fun y : ℝ => y) p)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    p = q := by
  refine laplaceZeroDrift_imp_eq_of_companionAligned τ hτ p q hzero fun x => ?_
  have h := laplaceCompanionAlignmentDefect_eq_zero_of_zeroDrift
    τ hτ p q hpFirst hzero x
  unfold laplaceCompanionAlignmentDefect at h
  exact sub_eq_zero.mp h

/-- Condition form of the atomless converse.  Note that the condition is
bandwidth-free: the same class is identified for EVERY valid bandwidth. -/
def LaplaceAtomlessCondition (p q : Measure ℝ) : Prop :=
  IsProbabilityMeasure p ∧ IsProbabilityMeasure q ∧
    NoAtoms p ∧ NoAtoms q ∧ Integrable (fun y : ℝ => y) p

/-- `IdentifiesAtZero` wrapper for the atomless condition. -/
theorem laplaceZeroDrift_identifiesAtZero_of_noAtoms
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    IdentifiesAtZero
      LaplaceAtomlessCondition
      (meanShiftDrift (laplaceKernel τ)) := by
  intro p q hcond hzero
  obtain ⟨hpProb, hqProb, hpAtomless, hqAtomless, hpFirst⟩ := hcond
  letI : IsProbabilityMeasure p := hpProb
  letI : IsProbabilityMeasure q := hqProb
  letI : NoAtoms p := hpAtomless
  letI : NoAtoms q := hqAtomless
  exact laplaceZeroDrift_identifies_of_noAtoms τ hτ p q hpFirst hzero

/-- Atomlessness transfers through absolute continuity. -/
theorem noAtoms_of_absolutelyContinuous
    {μ ν : Measure ℝ} [NoAtoms ν] (h : μ ≪ ν) : NoAtoms μ :=
  ⟨fun x => h (measure_singleton x)⟩

/-- The Gaussian pair witnesses the atomless condition. -/
theorem laplaceAtomlessCondition_gaussianPair :
    LaplaceAtomlessCondition
      (gaussianReal 0 (1 : NNReal)) (gaussianReal 1 (1 : NNReal)) :=
  ⟨inferInstance, inferInstance,
    noAtoms_gaussianReal (by norm_num : (1 : NNReal) ≠ 0),
    noAtoms_gaussianReal (by norm_num : (1 : NNReal) ≠ 0),
    integrable_id_gaussianReal_as_fun 0 (1 : NNReal)⟩

/-- The atomless condition admits a concrete distinct pair. -/
theorem laplaceAtomlessCondition_allowsDistinctPair :
    ConditionAllowsDistinctPair LaplaceAtomlessCondition :=
  ⟨gaussianReal 0 (1 : NNReal), gaussianReal 1 (1 : NNReal),
    laplaceAtomlessCondition_gaussianPair,
    gaussianReal_zero_ne_one_unitVariance⟩

/-- The atomless condition is formally legitimate. -/
theorem laplaceAtomlessCondition_isLegitimate :
    IsLegitimateCondition LaplaceAtomlessCondition := by
  constructor
  · exact ⟨gaussianReal 0 (1 : NNReal), gaussianReal 1 (1 : NNReal),
      laplaceAtomlessCondition_gaussianPair⟩
  · exact laplaceAtomlessCondition_allowsDistinctPair

/-- **Subsumption**: every continuous-density package satisfies the atomless
condition, so the atomless theorem strictly extends the Milestone-5
continuous-density theorem (which additionally assumed exponential
moments). -/
theorem laplaceAtomlessCondition_of_continuousDensityCondition
    {τ : ℝ} {p q : Measure ℝ}
    (h : LaplaceACContinuousDensityCondition τ p q) :
    LaplaceAtomlessCondition p q := by
  obtain ⟨hpProb, hqProb, ⟨pkg⟩⟩ := h
  refine ⟨hpProb, hqProb, ?_, ?_, pkg.hpFirstMoment⟩
  · rw [pkg.hpρ.measure_eq]
    exact noAtoms_of_absolutelyContinuous (withDensity_absolutelyContinuous _ _)
  · rw [pkg.hqρ.measure_eq]
    exact noAtoms_of_absolutelyContinuous (withDensity_absolutelyContinuous _ _)

end DriftingIdentifiability
