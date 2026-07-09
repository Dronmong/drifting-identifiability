import DriftingIdentifiability.GaussianConvolutionInjectivity
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# Gaussian score recovery

This file proves the remaining analytic stage of the raw Gaussian mean-shift
converse. For an arbitrary probability measure, the Gaussian kernel normalizer
is positive and differentiable, with

`D log Zₚ(x) = σ⁻² ⟪meanShiftₚ(x), ·⟫`.

Consequently equal Gaussian mean-shift maps give proportional normalizers.
The probability normalization theorem from
`GaussianConvolutionInjectivity.lean` forces the proportionality constant to
be one.
-/

open MeasureTheory
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

universe u

section GaussianScore

variable {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
  [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [BorelSpace E] [SecondCountableTopology E]

private noncomputable def gaussianKernelFDeriv
    (σ : ℝ) (x y : E) : E →L[ℝ] ℝ :=
  ((σ ^ 2)⁻¹ * gaussianKernel σ x y) • innerSL ℝ (y - x)

omit [MeasurableSpace E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [BorelSpace E] [SecondCountableTopology E] in
private lemma hasFDerivAt_gaussianKernel
    (σ : ℝ) (hσ : 0 < σ) (x y : E) :
    HasFDerivAt (fun x : E => gaussianKernel σ x y)
      (gaussianKernelFDeriv σ x y) x := by
  have hsub : HasFDerivAt (fun x : E => x - y)
      (ContinuousLinearMap.id ℝ E) x :=
    (hasFDerivAt_id x).sub_const y
  have hnorm := hsub.norm_sq
  have hscale := hnorm.const_mul (-(1 / (2 * σ ^ 2)))
  have hexp := hscale.exp
  convert hexp using 1
  · funext z
    rfl
  · ext z
    simp only [gaussianKernelFDeriv, smul_apply, innerSL_apply_apply,
      ContinuousLinearMap.comp_apply, ContinuousLinearMap.id_apply, smul_eq_mul]
    unfold gaussianKernel
    field_simp
    have hinner : ⟪y - x, z⟫ = -⟪x - y, z⟫ := by
      simp only [inner_sub_left]
      ring
    rw [hinner]
    ring

private lemma mul_gaussian_le {a r : ℝ} (ha : 0 < a) :
    r * Real.exp (-a * r ^ 2) ≤ 1 + a⁻¹ := by
  have hexp_nonneg : 0 ≤ Real.exp (-a * r ^ 2) := (Real.exp_pos _).le
  have hr_le : r ≤ 1 + r ^ 2 := by
    nlinarith [sq_nonneg (r - 1 / 2)]
  have hk_le_one : Real.exp (-a * r ^ 2) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    nlinarith [sq_nonneg r]
  have hu := Real.mul_exp_neg_le_exp_neg_one (a * r ^ 2)
  have he_le_one : Real.exp (-1) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    norm_num
  have hu_le : (a * r ^ 2) * Real.exp (-(a * r ^ 2)) ≤ 1 :=
    hu.trans he_le_one
  have ha_inv_nonneg : 0 ≤ a⁻¹ := (inv_pos.mpr ha).le
  have hrsq_le : r ^ 2 * Real.exp (-a * r ^ 2) ≤ a⁻¹ := by
    calc
      r ^ 2 * Real.exp (-a * r ^ 2) =
          a⁻¹ * ((a * r ^ 2) * Real.exp (-(a * r ^ 2))) := by
            field_simp
      _ ≤ a⁻¹ * 1 := mul_le_mul_of_nonneg_left hu_le ha_inv_nonneg
      _ = a⁻¹ := mul_one _
  calc
    r * Real.exp (-a * r ^ 2) ≤
        (1 + r ^ 2) * Real.exp (-a * r ^ 2) :=
      mul_le_mul_of_nonneg_right hr_le hexp_nonneg
    _ = Real.exp (-a * r ^ 2) +
        r ^ 2 * Real.exp (-a * r ^ 2) := by ring
    _ ≤ 1 + a⁻¹ := add_le_add hk_le_one hrsq_le

private noncomputable def gaussianFDerivBound (σ : ℝ) : ℝ :=
  |(σ ^ 2)⁻¹| * (1 + (1 / (2 * σ ^ 2))⁻¹)

omit [MeasurableSpace E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [BorelSpace E] [SecondCountableTopology E] in
private lemma norm_gaussianKernelFDeriv_le
    (σ : ℝ) (hσ : 0 < σ) (x y : E) :
    ‖gaussianKernelFDeriv σ x y‖ ≤ gaussianFDerivBound σ := by
  have ha : 0 < 1 / (2 * σ ^ 2) := by positivity
  have hcore := mul_gaussian_le
    (a := 1 / (2 * σ ^ 2)) (r := ‖y - x‖) ha
  unfold gaussianKernelFDeriv gaussianFDerivBound gaussianKernel
  rw [norm_smul, innerSL_apply_norm, Real.norm_eq_abs, abs_mul,
    abs_of_pos (Real.exp_pos _)]
  calc
    |(σ ^ 2)⁻¹| * Real.exp (-(1 / (2 * σ ^ 2)) * ‖x - y‖ ^ 2) *
        ‖y - x‖ =
      |(σ ^ 2)⁻¹| *
        (‖y - x‖ * Real.exp (-(1 / (2 * σ ^ 2)) * ‖y - x‖ ^ 2)) := by
          rw [norm_sub_rev x y]
          ring
    _ ≤ |(σ ^ 2)⁻¹| * (1 + (1 / (2 * σ ^ 2))⁻¹) :=
      mul_le_mul_of_nonneg_left hcore (abs_nonneg _)

omit [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [SecondCountableTopology E] in
private lemma gaussianKernel_integrable
    (σ : ℝ) (hσ : 0 < σ)
    (p : Measure E) [IsFiniteMeasure p] (x : E) :
    Integrable (fun y => gaussianKernel σ x y) p := by
  refine Integrable.of_bound ?_ 1 ?_
  · apply Continuous.aestronglyMeasurable
    unfold gaussianKernel
    fun_prop
  · filter_upwards with y
    rw [Real.norm_eq_abs]
    unfold gaussianKernel
    rw [abs_of_pos (Real.exp_pos _), Real.exp_le_one_iff]
    have : 0 < 1 / (2 * σ ^ 2) := by positivity
    nlinarith [sq_nonneg ‖x - y‖]

private noncomputable def gaussianWeightedDisplacement
    (σ : ℝ) (x y : E) : E :=
  gaussianKernel σ x y • (y - x)

omit [CompleteSpace E] [FiniteDimensional ℝ E] in
private lemma gaussianWeightedDisplacement_integrable
    (σ : ℝ) (hσ : 0 < σ)
    (p : Measure E) [IsFiniteMeasure p] (x : E) :
    Integrable (gaussianWeightedDisplacement σ x) p := by
  have ha : 0 < 1 / (2 * σ ^ 2) := by positivity
  refine Integrable.of_bound ?_ (1 + (1 / (2 * σ ^ 2))⁻¹) ?_
  · apply Continuous.aestronglyMeasurable
    unfold gaussianWeightedDisplacement gaussianKernel
    fun_prop
  · filter_upwards with y
    rw [gaussianWeightedDisplacement, norm_smul, Real.norm_eq_abs]
    unfold gaussianKernel
    rw [abs_of_pos (Real.exp_pos _), norm_sub_rev y x]
    simpa only [mul_comm] using
      (mul_gaussian_le (a := 1 / (2 * σ ^ 2)) (r := ‖x - y‖) ha)

omit [CompleteSpace E] [SecondCountableTopology E] in
private lemma gaussianKernelFDeriv_integrable
    (σ : ℝ) (hσ : 0 < σ)
    (p : Measure E) [IsFiniteMeasure p] (x : E) :
    Integrable (gaussianKernelFDeriv σ x) p := by
  refine Integrable.of_bound ?_ (gaussianFDerivBound σ) ?_
  · apply Continuous.aestronglyMeasurable
    unfold gaussianKernelFDeriv gaussianKernel
    fun_prop
  · exact ae_of_all _ (norm_gaussianKernelFDeriv_le σ hσ x)

private lemma integral_gaussianKernelFDeriv_eq
    (σ : ℝ) (hσ : 0 < σ)
    (p : Measure E) [IsFiniteMeasure p] (x : E) :
    (∫ y, gaussianKernelFDeriv σ x y ∂p) =
      (σ ^ 2)⁻¹ •
        innerSL ℝ (∫ y, gaussianWeightedDisplacement σ x y ∂p) := by
  ext z
  rw [ContinuousLinearMap.integral_apply
    (gaussianKernelFDeriv_integrable σ hσ p x)]
  simp only [gaussianKernelFDeriv, smul_apply, innerSL_apply_apply, smul_eq_mul]
  conv_rhs => rw [real_inner_comm]
  rw [← integral_inner (gaussianWeightedDisplacement_integrable σ hσ p x) z]
  rw [← integral_const_mul]
  congr 1
  funext y
  simp only [gaussianWeightedDisplacement, real_inner_smul_right]
  rw [real_inner_comm (y - x) z]
  ring

private lemma hasFDerivAt_gaussianKernelNormalizer
    (σ : ℝ) (hσ : 0 < σ)
    (p : Measure E) [IsFiniteMeasure p] (x : E) :
    HasFDerivAt
      (kernelNormalizer (gaussianKernel σ) p)
      ((σ ^ 2)⁻¹ •
        innerSL ℝ (∫ y, gaussianWeightedDisplacement σ x y ∂p))
      x := by
  have hraw :
      HasFDerivAt
        (fun x => ∫ y, gaussianKernel σ x y ∂p)
        (∫ y, gaussianKernelFDeriv σ x y ∂p) x := by
    apply hasFDerivAt_integral_of_dominated_of_fderiv_le
      (s := Set.univ) (bound := fun _ => gaussianFDerivBound σ)
    · exact Filter.univ_mem
    · filter_upwards with z
      apply Continuous.aestronglyMeasurable
      unfold gaussianKernel
      fun_prop
    · exact gaussianKernel_integrable σ hσ p x
    · apply Continuous.aestronglyMeasurable
      unfold gaussianKernelFDeriv gaussianKernel
      fun_prop
    · exact ae_of_all _ fun y z _ => norm_gaussianKernelFDeriv_le σ hσ z y
    · exact integrable_const _
    · exact ae_of_all _ fun y z _ => hasFDerivAt_gaussianKernel σ hσ z y
  rw [integral_gaussianKernelFDeriv_eq σ hσ p x] at hraw
  exact hraw

omit [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [SecondCountableTopology E] in
/-- A Gaussian kernel normalizer is strictly positive for every probability
measure and every probe. -/
theorem gaussianKernelNormalizer_pos
    (σ : ℝ) (hσ : ValidBandwidth σ)
    (p : Measure E) [IsProbabilityMeasure p] (x : E) :
    0 < kernelNormalizer (gaussianKernel σ) p x := by
  unfold kernelNormalizer gaussianKernel
  exact integral_exp_pos (gaussianKernel_integrable σ hσ p x)

/-- Gaussian score identity in Fréchet-derivative form:
`D log Zₚ(x) = σ⁻² ⟪meanShiftₚ(x), ·⟫`. -/
theorem hasFDerivAt_log_gaussianKernelNormalizer
    (σ : ℝ) (hσ : ValidBandwidth σ)
    (p : Measure E) [IsProbabilityMeasure p] (x : E) :
    HasFDerivAt
      (fun x => Real.log (kernelNormalizer (gaussianKernel σ) p x))
      ((σ ^ 2)⁻¹ • innerSL ℝ (meanShift (gaussianKernel σ) p x))
      x := by
  have hnorm := hasFDerivAt_gaussianKernelNormalizer σ hσ p x
  have hlog := hnorm.log (ne_of_gt (gaussianKernelNormalizer_pos σ hσ p x))
  convert hlog using 1
  ext z
  simp only [meanShift, gaussianWeightedDisplacement, smul_apply,
    innerSL_apply_apply, real_inner_smul_left, smul_eq_mul]
  ring

/-- Equal Gaussian mean-shift maps determine equal Gaussian-smoothed scalar
normalizers. This is the score-recovery stage of the raw-field converse. -/
theorem gaussianMeanShift_eq_imp_kernelNormalizer_eq
    (σ : ℝ) (hσ : ValidBandwidth σ)
    (p q : Measure E) [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (h : ∀ x,
      meanShift (gaussianKernel σ) p x =
        meanShift (gaussianKernel σ) q x) :
    ∀ x,
      kernelNormalizer (gaussianKernel σ) p x =
        kernelNormalizer (gaussianKernel σ) q x := by
  let lp : E → ℝ :=
    fun x => Real.log (kernelNormalizer (gaussianKernel σ) p x)
  let lq : E → ℝ :=
    fun x => Real.log (kernelNormalizer (gaussianKernel σ) q x)
  have hp_deriv : ∀ x, HasFDerivAt lp
      ((σ ^ 2)⁻¹ • innerSL ℝ (meanShift (gaussianKernel σ) p x)) x :=
    fun x => hasFDerivAt_log_gaussianKernelNormalizer σ hσ p x
  have hq_deriv : ∀ x, HasFDerivAt lq
      ((σ ^ 2)⁻¹ • innerSL ℝ (meanShift (gaussianKernel σ) q x)) x :=
    fun x => hasFDerivAt_log_gaussianKernelNormalizer σ hσ q x
  have hdiff : Differentiable ℝ (fun x => lp x - lq x) :=
    fun x => (hp_deriv x).differentiableAt.sub (hq_deriv x).differentiableAt
  have hfderiv : ∀ x, fderiv ℝ (fun x => lp x - lq x) x = 0 := by
    intro x
    rw [fderiv_fun_sub (hp_deriv x).differentiableAt
      (hq_deriv x).differentiableAt, (hp_deriv x).fderiv,
      (hq_deriv x).fderiv, h x, sub_self]
  let c : ℝ := Real.exp (lp 0 - lq 0)
  have hprop : ∀ x,
      kernelNormalizer (gaussianKernel σ) p x =
        c * kernelNormalizer (gaussianKernel σ) q x := by
    intro x
    have hconst :=
      is_const_of_fderiv_eq_zero hdiff hfderiv x (0 : E)
    have hexp := congrArg Real.exp hconst
    dsimp only [lp, lq] at hexp
    rw [Real.exp_sub,
      Real.exp_log (gaussianKernelNormalizer_pos σ hσ p x),
      Real.exp_log (gaussianKernelNormalizer_pos σ hσ q x)] at hexp
    exact
      (div_eq_iff (ne_of_gt (gaussianKernelNormalizer_pos σ hσ q x))).mp hexp
  have hc : c = 1 :=
    gaussianKernelNormalizer_proportional_constant_eq_one σ hσ p q c hprop
  intro x
  simpa [hc] using hprop x

/-- **Raw Gaussian mean-shift identifiability.** For arbitrary probability
measures on a finite-dimensional real inner-product space, pointwise zero
Gaussian mean-shift drift forces equality of the measures. -/
theorem gaussianMeanShiftDrift_identifiesAtZero
    (σ : ℝ) (hσ : ValidBandwidth σ) :
    IdentifiesAtZero (BothProbability (E := E))
      (meanShiftDrift (gaussianKernel σ)) := by
  rintro p q ⟨hp, hq⟩ hzero
  haveI := hp
  haveI := hq
  have hmeanShift : ∀ x,
      meanShift (gaussianKernel σ) p x =
        meanShift (gaussianKernel σ) q x :=
    fun x => sub_eq_zero.mp (hzero x)
  have hnormalizer :=
    gaussianMeanShift_eq_imp_kernelNormalizer_eq σ hσ p q hmeanShift
  exact gaussianKernelNormalizer_injective σ hσ p q hnormalizer

end GaussianScore

end DriftingIdentifiability
