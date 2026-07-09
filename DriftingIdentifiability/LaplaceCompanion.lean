import DriftingIdentifiability.LaplacianGaussianConverse
import Mathlib.Analysis.Calculus.ParametricIntegral

/-!
# The Laplace companion-kernel score identity

Stage 1 of the attack on the open Laplace-kernel arbitrary-target converse
(`LaplaceArbitraryConverse.md`).  For the Gaussian kernel the raw mean shift is
the score of the smoothed density — the fact behind the promoted raw Gaussian
converse.  For the paper's Laplace kernel that identity fails, but an exact
replacement holds: the mean-shift numerator is the gradient of a *different*
radial smoothing, by the **companion kernel**

`ℓτ(x,y) = (τ + ‖x-y‖) · exp (-‖x-y‖/τ)`.

Pointwise, `∇ₓ ℓτ(x,y) = -(1/τ)(x-y) · kτ(x,y)` — including at `x = y`, where
the bare kernel has its cone singularity but the companion kernel is
differentiable with vanishing gradient.  Integrating against any probability
measure `p` (no moment assumptions are needed: every integrand is bounded),

`∇ (ℓτ * p)(x) = (1/τ) ∫ kτ(x,y)(y - x) dp = (1/τ) Z_p(x) · meanShift_p(x)`,

so zero raw drift is equivalent to the cross-gradient equation
`Z_q ∇L_p = Z_p ∇L_q` (`laplaceZeroDrift_iff_crossDisplacement`).  The open
converse is thereby isolated in the mismatch between the two smoothings
`ℓτ ≠ c · kτ` (for the Gaussian they coincide, up to constants).

This module also certifies that the Laplace field is globally well defined:
`MeanShiftRegularAt (laplaceKernel τ) p q x` holds for ARBITRARY probability
measures at every probe, and bridges zero drift to the exponential-tilt data
of `LaplacianGaussianConverse.lean` for measures with the critical exponential
moment (`laplaceZeroDrift_tiltCentroid_eq`).  Everything is axiom-free.
-/

open MeasureTheory Filter Topology
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ## Scalar inequalities -/

/-- The Laplace displacement weight is uniformly bounded: `s·e^{-s/τ} ≤ τ`. -/
private lemma mul_exp_neg_le {τ : ℝ} (hτ : 0 < τ) {s : ℝ} (hs : 0 ≤ s) :
    s * Real.exp (-(1 / τ) * s) ≤ τ := by
  have h1 : s / τ + 1 ≤ Real.exp (s / τ) := Real.add_one_le_exp (s / τ)
  have h2 := mul_le_mul_of_nonneg_left h1 hτ.le
  have h3 : τ * (s / τ + 1) = s + τ := by field_simp
  have hexp : Real.exp (-(1 / τ) * s) = (Real.exp (s / τ))⁻¹ := by
    rw [← Real.exp_neg]
    congr 1
    field_simp
  rw [hexp, mul_inv_le_iff₀ (Real.exp_pos _)]
  nlinarith [Real.exp_pos (s / τ), hs]

/-- The companion profile never exceeds its value at the origin:
`(τ+s)e^{-s/τ} ≤ τ`. -/
private lemma companion_profile_le {τ : ℝ} (hτ : 0 < τ) {s : ℝ} :
    (τ + s) * Real.exp (-(1 / τ) * s) ≤ τ := by
  have h1 : s / τ + 1 ≤ Real.exp (s / τ) := Real.add_one_le_exp (s / τ)
  have h2 := mul_le_mul_of_nonneg_left h1 hτ.le
  have h3 : τ * (s / τ + 1) = τ + s := by field_simp; ring
  have hpos := (Real.exp_pos (-(1 / τ) * s)).le
  have hrel : Real.exp (s / τ) * Real.exp (-(1 / τ) * s) = 1 := by
    rw [← Real.exp_add, show s / τ + -(1 / τ) * s = 0 by ring, Real.exp_zero]
  calc (τ + s) * Real.exp (-(1 / τ) * s)
      ≤ (τ * Real.exp (s / τ)) * Real.exp (-(1 / τ) * s) := by nlinarith
    _ = τ * (Real.exp (s / τ) * Real.exp (-(1 / τ) * s)) := by ring
    _ = τ := by rw [hrel, mul_one]

/-- Quadratic Taylor control of the companion profile at the origin:
`|(τ+s)e^{-s/τ} - τ| ≤ s²/τ`. -/
private lemma companion_taylor_bound {τ : ℝ} (hτ : 0 < τ) {s : ℝ} (hs : 0 ≤ s) :
    |(τ + s) * Real.exp (-(1 / τ) * s) - τ| ≤ s ^ 2 / τ := by
  have hupper := companion_profile_le hτ (s := s)
  have hlower : τ - s ^ 2 / τ ≤ (τ + s) * Real.exp (-(1 / τ) * s) := by
    have h1 : 1 - (1 / τ) * s ≤ Real.exp (-(1 / τ) * s) := by
      have := Real.add_one_le_exp (-(1 / τ) * s)
      linarith
    have hτs : (0 : ℝ) ≤ τ + s := by linarith
    have h2 := mul_le_mul_of_nonneg_left h1 hτs
    have heq : (τ + s) * (1 - (1 / τ) * s) = τ - s ^ 2 / τ := by
      field_simp
      ring
    linarith
  have hsq : (0 : ℝ) ≤ s ^ 2 / τ := by positivity
  rw [abs_le]
  constructor <;> linarith

/-! ## The companion kernel and its pointwise gradient -/

/-- **The companion kernel** `ℓτ(x,y) = (τ + ‖x-y‖)·kτ(x,y)`.  Its smoothing of
`p` is the potential whose gradient is the Laplace mean-shift numerator. -/
noncomputable def laplaceCompanionKernel (τ : ℝ) (x y : E) : ℝ :=
  (τ + ‖x - y‖) * laplaceKernel τ x y

private noncomputable def laplaceCompanionFDeriv (τ : ℝ) (x y : E) : E →L[ℝ] ℝ :=
  (-(1 / τ) * laplaceKernel τ x y) • innerSL ℝ (x - y)

private lemma norm_laplaceCompanionFDeriv_le (τ : ℝ) (hτ : 0 < τ) (x y : E) :
    ‖laplaceCompanionFDeriv τ x y‖ ≤ 1 := by
  unfold laplaceCompanionFDeriv laplaceKernel
  rw [norm_smul, innerSL_apply_norm, Real.norm_eq_abs, abs_mul, abs_neg,
    abs_of_pos (by positivity : (0 : ℝ) < 1 / τ), abs_of_pos (Real.exp_pos _)]
  have hb := mul_exp_neg_le hτ (norm_nonneg (x - y))
  calc 1 / τ * Real.exp (-(1 / τ) * ‖x - y‖) * ‖x - y‖
      = (‖x - y‖ * Real.exp (-(1 / τ) * ‖x - y‖)) / τ := by ring
    _ ≤ 1 := by
        rw [div_le_one hτ]
        exact hb

/-- The norm-difference map is differentiable away from the base point, with
the expected unit-radial gradient. -/
private lemma hasFDerivAt_norm_sub (y x : E) (hne : x ≠ y) :
    HasFDerivAt (fun z : E => ‖z - y‖)
      (‖x - y‖⁻¹ • innerSL ℝ (x - y)) x := by
  have hxy : x - y ≠ 0 := sub_ne_zero.mpr hne
  have hs : (0 : ℝ) < ‖x - y‖ := norm_pos_iff.mpr hxy
  have hsub : HasFDerivAt (fun z : E => z - y)
      (ContinuousLinearMap.id ℝ E) x := (hasFDerivAt_id x).sub_const y
  have hnsq := hsub.norm_sq
  have hsqrt :=
    (Real.hasDerivAt_sqrt (ne_of_gt (pow_pos hs 2))).comp_hasFDerivAt x hnsq
  have hfun : ((fun t : ℝ => Real.sqrt t) ∘ fun z : E => ‖z - y‖ ^ 2) =
      fun z : E => ‖z - y‖ :=
    funext fun z => Real.sqrt_sq (norm_nonneg _)
  rw [hfun] at hsqrt
  refine hsqrt.congr_fderiv ?_
  rw [Real.sqrt_sq hs.le]
  ext z
  simp only [smul_apply, innerSL_apply_apply,
    ContinuousLinearMap.coe_comp, Function.comp_apply,
    ContinuousLinearMap.id_apply, smul_eq_mul]
  field_simp [hs.ne']
  ring

/-- **Pointwise companion gradient identity.**  The companion kernel is
differentiable in the probe EVERYWHERE — including at `x = y` — with gradient
`-(1/τ)(x-y)·kτ(x,y)`. -/
private lemma hasFDerivAt_laplaceCompanionKernel
    (τ : ℝ) (hτ : 0 < τ) (y x : E) :
    HasFDerivAt (fun z : E => laplaceCompanionKernel τ z y)
      (laplaceCompanionFDeriv τ x y) x := by
  rcases eq_or_ne x y with rfl | hne
  · -- at the base point: quadratic remainder, zero derivative
    have h0 : laplaceCompanionFDeriv τ x x = 0 := by
      unfold laplaceCompanionFDeriv
      rw [sub_self]
      simp
    rw [h0, hasFDerivAt_iff_isLittleO_nhds_zero]
    simp only [zero_apply, sub_zero]
    rw [Asymptotics.isLittleO_iff]
    intro c hc
    filter_upwards [Metric.ball_mem_nhds (0 : E) (mul_pos hc hτ)] with h hh
    rw [mem_ball_zero_iff] at hh
    have hfx : laplaceCompanionKernel τ x x = τ := by
      unfold laplaceCompanionKernel laplaceKernel
      simp
    have hfxh : laplaceCompanionKernel τ (x + h) x =
        (τ + ‖h‖) * Real.exp (-(1 / τ) * ‖h‖) := by
      unfold laplaceCompanionKernel laplaceKernel
      rw [add_sub_cancel_left]
    rw [hfx, hfxh, Real.norm_eq_abs]
    calc |(τ + ‖h‖) * Real.exp (-(1 / τ) * ‖h‖) - τ|
        ≤ ‖h‖ ^ 2 / τ := companion_taylor_bound hτ (norm_nonneg h)
      _ ≤ c * ‖h‖ := by
          rw [div_le_iff₀ hτ]
          nlinarith [mul_le_mul_of_nonneg_left hh.le (norm_nonneg h),
            norm_nonneg h]
  · -- away from the base point: chain rule through the norm
    have hs : (0 : ℝ) < ‖x - y‖ := norm_pos_iff.mpr (sub_ne_zero.mpr hne)
    have hnorm := hasFDerivAt_norm_sub y x hne
    have hplus : HasFDerivAt (fun z : E => τ + ‖z - y‖)
        (‖x - y‖⁻¹ • innerSL ℝ (x - y)) x := hnorm.const_add τ
    have hexp : HasFDerivAt
        (fun z : E => Real.exp (-(1 / τ) * ‖z - y‖))
        (Real.exp (-(1 / τ) * ‖x - y‖) •
          ((-(1 / τ)) • (‖x - y‖⁻¹ • innerSL ℝ (x - y)))) x :=
      (hnorm.const_mul (-(1 / τ))).exp
    have hmul := hplus.mul hexp
    have hfun_eq : (fun z : E => laplaceCompanionKernel τ z y) =
        ((fun z : E => τ + ‖z - y‖) * fun z : E =>
          Real.exp (-(1 / τ) * ‖z - y‖)) := by
      funext z
      simp only [Pi.mul_apply]
      unfold laplaceCompanionKernel laplaceKernel
      rfl
    rw [hfun_eq]
    refine hmul.congr_fderiv ?_
    unfold laplaceCompanionFDeriv laplaceKernel
    ext z
    simp only [smul_apply, add_apply, innerSL_apply_apply, smul_eq_mul]
    field_simp [hs.ne']
    ring

/-- The Laplace-weighted displacement `kτ(x,y)•(y - x)` — the mean-shift
numerator's integrand. -/
noncomputable def laplaceWeightedDisplacement (τ : ℝ) (x y : E) : E :=
  laplaceKernel τ x y • (y - x)

private lemma norm_laplaceWeightedDisplacement_le
    (τ : ℝ) (hτ : 0 < τ) (x y : E) :
    ‖laplaceWeightedDisplacement τ x y‖ ≤ τ := by
  unfold laplaceWeightedDisplacement laplaceKernel
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _),
    norm_sub_rev y x]
  simpa [mul_comm] using mul_exp_neg_le hτ (norm_nonneg (x - y))

/-! ## Integrated form: the score identity for arbitrary probability measures -/

section Integral

variable [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
  [SecondCountableTopology E]

omit [CompleteSpace E] in
/-- The Laplace mean-shift numerator is integrable for EVERY finite measure —
no moment assumptions: the integrand is uniformly bounded by `τ`. -/
lemma laplaceWeightedDisplacement_integrable
    (τ : ℝ) (hτ : 0 < τ) (p : Measure E) [IsFiniteMeasure p] (x : E) :
    Integrable (laplaceWeightedDisplacement τ x) p := by
  refine Integrable.of_bound ?_ τ ?_
  · apply Continuous.aestronglyMeasurable
    unfold laplaceWeightedDisplacement laplaceKernel
    fun_prop
  · filter_upwards with y
    exact norm_laplaceWeightedDisplacement_le τ hτ x y

omit [CompleteSpace E] in
/-- The identity-weighted Laplace integrand is integrable for every finite
measure: `kτ(x,y)‖y‖ ≤ τ + ‖x‖`.  Generalizes the Gaussian-target version of
`LaplacianGaussianConverse.lean` to arbitrary laws. -/
lemma laplaceKernel_smul_id_integrable'
    (τ : ℝ) (hτ : 0 < τ) (p : Measure E) [IsFiniteMeasure p] (x : E) :
    Integrable (fun y => laplaceKernel τ x y • y) p := by
  refine Integrable.of_bound ?_ (τ + ‖x‖) ?_
  · apply Continuous.aestronglyMeasurable
    unfold laplaceKernel
    fun_prop
  · filter_upwards with y
    rw [norm_smul, Real.norm_eq_abs]
    unfold laplaceKernel
    rw [abs_of_pos (Real.exp_pos _)]
    have h1 : ‖y‖ ≤ ‖x - y‖ + ‖x‖ := by
      have := norm_sub_norm_le y x
      rw [norm_sub_rev y x] at this
      linarith
    have h2 := mul_exp_neg_le hτ (norm_nonneg (x - y))
    have h3 : Real.exp (-(1 / τ) * ‖x - y‖) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      have : (0 : ℝ) ≤ 1 / τ := by positivity
      nlinarith [norm_nonneg (x - y)]
    nlinarith [Real.exp_pos (-(1 / τ) * ‖x - y‖), norm_nonneg x,
      norm_nonneg (x - y)]

omit [InnerProductSpace ℝ E] [CompleteSpace E] [SecondCountableTopology E] in
private lemma laplaceCompanionKernel_integrable
    (τ : ℝ) (hτ : 0 < τ) (p : Measure E) [IsFiniteMeasure p] (x : E) :
    Integrable (fun y => laplaceCompanionKernel τ x y) p := by
  refine Integrable.of_bound ?_ τ ?_
  · apply Continuous.aestronglyMeasurable
    unfold laplaceCompanionKernel laplaceKernel
    fun_prop
  · filter_upwards with y
    rw [Real.norm_eq_abs]
    unfold laplaceCompanionKernel laplaceKernel
    rw [abs_of_nonneg (by positivity)]
    exact companion_profile_le hτ

omit [CompleteSpace E] in
private lemma laplaceCompanionFDeriv_integrable
    (τ : ℝ) (hτ : 0 < τ) (p : Measure E) [IsFiniteMeasure p] (x : E) :
    Integrable (laplaceCompanionFDeriv τ x) p := by
  refine Integrable.of_bound ?_ 1 ?_
  · apply Continuous.aestronglyMeasurable
    unfold laplaceCompanionFDeriv laplaceKernel
    fun_prop
  · exact ae_of_all _ (norm_laplaceCompanionFDeriv_le τ hτ x)

private lemma integral_laplaceCompanionFDeriv_eq
    (τ : ℝ) (hτ : 0 < τ) (p : Measure E) [IsFiniteMeasure p] (x : E) :
    (∫ y, laplaceCompanionFDeriv τ x y ∂p) =
      (1 / τ) • innerSL ℝ (∫ y, laplaceWeightedDisplacement τ x y ∂p) := by
  ext z
  rw [ContinuousLinearMap.integral_apply
    (laplaceCompanionFDeriv_integrable τ hτ p x)]
  simp only [laplaceCompanionFDeriv, smul_apply,
    innerSL_apply_apply, smul_eq_mul]
  conv_rhs => rw [real_inner_comm]
  rw [← integral_inner (laplaceWeightedDisplacement_integrable τ hτ p x) z]
  rw [← integral_const_mul]
  congr 1
  funext y
  simp only [laplaceWeightedDisplacement, real_inner_smul_right]
  have hswap : ⟪x - y, z⟫ = -⟪y - x, z⟫ := by
    rw [← inner_neg_left, neg_sub]
  rw [hswap, real_inner_comm (y - x) z]
  ring

/-- **The companion score identity, integral form.**  For EVERY probability
measure, the companion-kernel normalizer is differentiable with Fréchet
derivative `(1/τ)⟪∫ kτ(x,y)(y-x) dp, ·⟫` — the Laplace mean-shift numerator is
an exact gradient.  The Laplace analogue of
`hasFDerivAt_gaussianKernelNormalizer`; no moment assumptions. -/
theorem hasFDerivAt_laplaceCompanionNormalizer
    (τ : ℝ) (hτ : ValidBandwidth τ)
    (p : Measure E) [IsProbabilityMeasure p] (x : E) :
    HasFDerivAt (kernelNormalizer (laplaceCompanionKernel τ) p)
      ((1 / τ) • innerSL ℝ (∫ y, laplaceWeightedDisplacement τ x y ∂p)) x := by
  have hτ0 : 0 < τ := hτ
  have hraw : HasFDerivAt (fun x => ∫ y, laplaceCompanionKernel τ x y ∂p)
      (∫ y, laplaceCompanionFDeriv τ x y ∂p) x := by
    apply hasFDerivAt_integral_of_dominated_of_fderiv_le
      (s := Set.univ) (bound := fun _ => (1 : ℝ))
    · exact Filter.univ_mem
    · filter_upwards with z
      apply Continuous.aestronglyMeasurable
      unfold laplaceCompanionKernel laplaceKernel
      fun_prop
    · exact laplaceCompanionKernel_integrable τ hτ0 p x
    · exact (laplaceCompanionFDeriv_integrable τ hτ0 p x).aestronglyMeasurable
    · exact ae_of_all _ fun y z _ => norm_laplaceCompanionFDeriv_le τ hτ0 z y
    · exact integrable_const _
    · exact ae_of_all _ fun y z _ => hasFDerivAt_laplaceCompanionKernel τ hτ0 y z
  rw [integral_laplaceCompanionFDeriv_eq τ hτ0 p x] at hraw
  exact hraw

omit [SecondCountableTopology E] in
/-- The mean-shift numerator equals the normalizer times the mean shift. -/
theorem integral_laplaceWeightedDisplacement_eq
    (τ : ℝ) (hτ : ValidBandwidth τ)
    (p : Measure E) [IsProbabilityMeasure p] (x : E) :
    (∫ y, laplaceWeightedDisplacement τ x y ∂p) =
      kernelNormalizer (laplaceKernel τ) p x •
        meanShift (laplaceKernel τ) p x := by
  have hτ0 : 0 < τ := hτ
  have hZ := laplaceKernelNormalizer_pos p τ hτ0 x
  unfold Paper.meanShift
  rw [smul_smul, mul_inv_cancel₀ (ne_of_gt hZ), one_smul]
  rfl

/-- **The Laplace score-analogue identity.**  The companion-normalizer
gradient is `(Z_p/τ)·meanShift_p`: the raw Laplace mean shift is an exact
weighted gradient, for every probability measure and every probe. -/
theorem hasFDerivAt_laplaceCompanionNormalizer_meanShift
    (τ : ℝ) (hτ : ValidBandwidth τ)
    (p : Measure E) [IsProbabilityMeasure p] (x : E) :
    HasFDerivAt (kernelNormalizer (laplaceCompanionKernel τ) p)
      ((1 / τ * kernelNormalizer (laplaceKernel τ) p x) •
        innerSL ℝ (meanShift (laplaceKernel τ) p x)) x := by
  have h := hasFDerivAt_laplaceCompanionNormalizer τ hτ p x
  rw [integral_laplaceWeightedDisplacement_eq τ hτ p x, map_smul,
    smul_smul] at h
  exact h

/-- **Global well-definedness of the Laplace field.**  The equation-11
regularity package holds for ARBITRARY probability measures at EVERY probe —
the paper's practical kernel needs no moment or support assumptions.  All
integrands are uniformly bounded (`s·e^{-s/τ} ≤ τ`). -/
theorem laplace_meanShiftRegularAt
    (τ : ℝ) (hτ : ValidBandwidth τ)
    (p q : Measure E) [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (x : E) :
    MeanShiftRegularAt (laplaceKernel τ) p q x := by
  have hτ0 : 0 < τ := hτ
  have hone : ∀ z y : E, Real.exp (-(1 / τ) * ‖z - y‖) ≤ 1 := by
    intro z y
    rw [Real.exp_le_one_iff]
    have : (0 : ℝ) ≤ 1 / τ := by positivity
    nlinarith [norm_nonneg (z - y)]
  refine ⟨(laplaceKernelNormalizer_pos p τ hτ0 x).ne',
    (laplaceKernelNormalizer_pos q τ hτ0 x).ne', ?_, ?_, ?_⟩
  · exact laplaceWeightedDisplacement_integrable τ hτ0 p x
  · exact laplaceWeightedDisplacement_integrable τ hτ0 q x
  · refine Integrable.of_bound ?_ (2 * τ) ?_
    · apply Continuous.aestronglyMeasurable
      unfold laplaceKernel
      fun_prop
    · filter_upwards with y
      rw [norm_smul, Real.norm_eq_abs]
      unfold laplaceKernel
      rw [abs_mul, abs_of_pos (Real.exp_pos _), abs_of_pos (Real.exp_pos _)]
      have htri : ‖y.1 - y.2‖ ≤ ‖x - y.1‖ + ‖x - y.2‖ := by
        calc ‖y.1 - y.2‖ = ‖(y.1 - x) + (x - y.2)‖ := by
              rw [sub_add_sub_cancel]
          _ ≤ ‖y.1 - x‖ + ‖x - y.2‖ := norm_add_le _ _
          _ = ‖x - y.1‖ + ‖x - y.2‖ := by rw [norm_sub_rev]
      have hb1 := mul_exp_neg_le hτ0 (norm_nonneg (x - y.1))
      have hb2 := mul_exp_neg_le hτ0 (norm_nonneg (x - y.2))
      have he1 := hone x y.1
      have he2 := hone x y.2
      have hp1 := (Real.exp_pos (-(1 / τ) * ‖x - y.1‖)).le
      have hp2 := (Real.exp_pos (-(1 / τ) * ‖x - y.2‖)).le
      calc Real.exp (-(1 / τ) * ‖x - y.1‖) * Real.exp (-(1 / τ) * ‖x - y.2‖) *
          ‖y.1 - y.2‖
          ≤ Real.exp (-(1 / τ) * ‖x - y.1‖) * Real.exp (-(1 / τ) * ‖x - y.2‖) *
            (‖x - y.1‖ + ‖x - y.2‖) :=
            mul_le_mul_of_nonneg_left htri (mul_nonneg hp1 hp2)
        _ = Real.exp (-(1 / τ) * ‖x - y.2‖) *
              (‖x - y.1‖ * Real.exp (-(1 / τ) * ‖x - y.1‖)) +
            Real.exp (-(1 / τ) * ‖x - y.1‖) *
              (‖x - y.2‖ * Real.exp (-(1 / τ) * ‖x - y.2‖)) := by ring
        _ ≤ 1 * τ + 1 * τ := by
            apply add_le_add
            · exact mul_le_mul he2 hb1 (by positivity) one_pos.le
            · exact mul_le_mul he1 hb2 (by positivity) one_pos.le
        _ = 2 * τ := by ring

omit [SecondCountableTopology E] in
/-- **Stage-1 headline: the cross-gradient reformulation of the open
converse.**  Pointwise zero raw Laplace drift between ARBITRARY probability
measures is equivalent to the cross-multiplied gradient equation
`Z_q(x)·∇L_p(x) = Z_p(x)·∇L_q(x)` (stated via the mean-shift numerators,
which the companion score identity exhibits as the gradients `τ·∇L`). -/
theorem laplaceZeroDrift_iff_crossDisplacement
    (τ : ℝ) (hτ : ValidBandwidth τ)
    (p q : Measure E) [IsProbabilityMeasure p] [IsProbabilityMeasure q] :
    ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q ↔
      ∀ x : E,
        kernelNormalizer (laplaceKernel τ) q x •
            (∫ y, laplaceWeightedDisplacement τ x y ∂p) =
          kernelNormalizer (laplaceKernel τ) p x •
            (∫ y, laplaceWeightedDisplacement τ x y ∂q) := by
  have hτ0 : 0 < τ := hτ
  constructor
  · intro hzero x
    have hZp := laplaceKernelNormalizer_pos p τ hτ0 x
    have hZq := laplaceKernelNormalizer_pos q τ hτ0 x
    have hms : meanShift (laplaceKernel τ) p x =
        meanShift (laplaceKernel τ) q x := sub_eq_zero.mp (hzero x)
    rw [integral_laplaceWeightedDisplacement_eq τ hτ p x,
      integral_laplaceWeightedDisplacement_eq τ hτ q x, hms,
      smul_smul, smul_smul, mul_comm]
  · intro hcross x
    have hZp := laplaceKernelNormalizer_pos p τ hτ0 x
    have hZq := laplaceKernelNormalizer_pos q τ hτ0 x
    have h := hcross x
    rw [integral_laplaceWeightedDisplacement_eq τ hτ p x,
      integral_laplaceWeightedDisplacement_eq τ hτ q x,
      smul_smul, smul_smul,
      mul_comm (kernelNormalizer (laplaceKernel τ) p x)
        (kernelNormalizer (laplaceKernel τ) q x)] at h
    have hc : kernelNormalizer (laplaceKernel τ) q x *
        kernelNormalizer (laplaceKernel τ) p x ≠ 0 :=
      mul_ne_zero (ne_of_gt hZq) (ne_of_gt hZp)
    exact sub_eq_zero.mpr (smul_right_injective E hc h)

/-- **Bridge to the exponential-tilt data.**  For measures with the critical
exponential moment, pointwise zero raw Laplace drift forces the
exponential-tilt centroids to agree in every unit direction.  This is ALL the
information radial asymptotics can extract (escaping probes see only the tilt
sphere; see `LaplaceArbitraryConverse.md`) — the finite-probe content is
carried by the cross-gradient reformulation above. -/
theorem laplaceZeroDrift_tiltCentroid_eq
    (τ : ℝ) (hτ : ValidBandwidth τ)
    (p q : Measure E) [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp₀ : Integrable (fun y : E => Real.exp (‖y‖ / τ)) p)
    (hp₁ : Integrable (fun y : E => Real.exp (‖y‖ / τ) * ‖y‖) p)
    (hq₀ : Integrable (fun y : E => Real.exp (‖y‖ / τ)) q)
    (hq₁ : Integrable (fun y : E => Real.exp (‖y‖ / τ) * ‖y‖) q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (u : E) (hu : ‖u‖ = 1) :
    exponentialTiltCentroid p τ u = exponentialTiltCentroid q τ u := by
  have hτ0 : 0 < τ := hτ
  have hcp := kernelCentroid_laplace_radial_tendsto p τ hτ0 u hu hp₀ hp₁
  have hcq := kernelCentroid_laplace_radial_tendsto q τ hτ0 u hu hq₀ hq₁
  have hdiff := hcp.sub hcq
  have hfun : (fun r : ℝ => kernelCentroid (laplaceKernel τ) p (r • u) -
      kernelCentroid (laplaceKernel τ) q (r • u)) = fun _ : ℝ => (0 : E) := by
    funext r
    have h0 := hzero (r • u)
    have hmp := meanShift_eq_kernelCentroid_sub p (laplaceKernel τ) (r • u)
      (laplaceKernel_integrable p τ hτ0 (r • u))
      (laplaceKernel_smul_id_integrable' τ hτ0 p (r • u))
      (ne_of_gt (laplaceKernelNormalizer_pos p τ hτ0 (r • u)))
    have hmq := meanShift_eq_kernelCentroid_sub q (laplaceKernel τ) (r • u)
      (laplaceKernel_integrable q τ hτ0 (r • u))
      (laplaceKernel_smul_id_integrable' τ hτ0 q (r • u))
      (ne_of_gt (laplaceKernelNormalizer_pos q τ hτ0 (r • u)))
    have hexpand : meanShiftDrift (laplaceKernel τ) p q (r • u) =
        kernelCentroid (laplaceKernel τ) p (r • u) -
          kernelCentroid (laplaceKernel τ) q (r • u) := by
      unfold Paper.meanShiftDrift
      rw [hmp, hmq]
      abel
    rw [hexpand] at h0
    exact h0
  rw [hfun] at hdiff
  exact sub_eq_zero.mp (tendsto_nhds_unique hdiff tendsto_const_nhds)

end Integral

end DriftingIdentifiability
