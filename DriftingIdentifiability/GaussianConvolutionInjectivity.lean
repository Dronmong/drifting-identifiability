import DriftingIdentifiability.TrustedBoundary
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Analysis.SpecialFunctions.Gaussian.FourierTransform

/-!
# Gaussian convolution injectivity

This file discharges the Gaussian-smoothing injectivity part of the raw
mean-shift converse. It is deliberately independent of the conditional
Gaussian/RKHS axioms in `Paperaxioms.lean`.

Besides the abstract convolution-cancellation theorem, the final result uses a
direct Fourier calculation for the project's unnormalized Gaussian kernel
normalizer. This avoids requiring a separate density-of-convolution API.
-/

open MeasureTheory ProbabilityTheory
open scoped FourierTransform RealInnerProductSpace

namespace DriftingIdentifiability

universe u

section CharacteristicCancellation

variable {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
  [InnerProductSpace ℝ E] [CompleteSpace E] [BorelSpace E]
  [SecondCountableTopology E]

/-- Convolution by a finite measure whose characteristic function is nowhere
zero is injective on finite measures. -/
theorem Measure.eq_of_conv_eq_conv_of_charFun_ne_zero
    (p q ν : Measure E) [IsFiniteMeasure p] [IsFiniteMeasure q]
    [IsFiniteMeasure ν] (hν : ∀ t, charFun ν t ≠ 0)
    (hconv : p ∗ ν = q ∗ ν) :
    p = q := by
  apply Measure.ext_of_charFun
  funext t
  have h := congrFun (congrArg charFun hconv) t
  rw [charFun_conv, charFun_conv] at h
  exact mul_right_cancel₀ (hν t) h

end CharacteristicCancellation

section ScaledGaussian

variable {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
  [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [BorelSpace E] [SecondCountableTopology E]

/-- The centered Gaussian probability measure with covariance `σ² I`,
obtained by scaling Mathlib's standard Gaussian. -/
noncomputable def scaledStdGaussian (σ : ℝ) : Measure E :=
  (stdGaussian E).map (σ • ·)

instance (σ : ℝ) : IsProbabilityMeasure (scaledStdGaussian (E := E) σ) := by
  unfold scaledStdGaussian
  exact Measure.isProbabilityMeasure_map (Measurable.aemeasurable (by fun_prop))

omit [CompleteSpace E] [SecondCountableTopology E] in
@[simp]
theorem charFun_scaledStdGaussian (σ : ℝ) (t : E) :
    charFun (scaledStdGaussian (E := E) σ) t =
      Complex.exp (-‖σ • t‖ ^ 2 / 2) := by
  rw [scaledStdGaussian, charFun_map_smul, charFun_stdGaussian]

omit [CompleteSpace E] [SecondCountableTopology E] in
theorem charFun_scaledStdGaussian_ne_zero (σ : ℝ) (t : E) :
    charFun (scaledStdGaussian (E := E) σ) t ≠ 0 := by
  rw [charFun_scaledStdGaussian]
  exact Complex.exp_ne_zero _

/-- Convolution by a scaled standard Gaussian is injective.  The proof works
even at `σ = 0` (where the scaling is a Dirac mass), although the raw-field
application assumes a positive bandwidth. -/
theorem Measure.eq_of_conv_scaledStdGaussian_eq
    (σ : ℝ) (p q : Measure E) [IsFiniteMeasure p] [IsFiniteMeasure q]
    (hconv :
      p ∗ scaledStdGaussian (E := E) σ =
        q ∗ scaledStdGaussian (E := E) σ) :
    p = q :=
  Measure.eq_of_conv_eq_conv_of_charFun_ne_zero p q
    (scaledStdGaussian (E := E) σ)
    (charFun_scaledStdGaussian_ne_zero σ) hconv

end ScaledGaussian

section GaussianKernelNormalizer

variable {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
  [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [BorelSpace E] [SecondCountableTopology E]

private noncomputable def gaussianFourierRate (σ : ℝ) : ℝ :=
  1 / (2 * σ ^ 2)

private lemma gaussianFourierRate_pos {σ : ℝ} (hσ : 0 < σ) :
    0 < gaussianFourierRate σ := by
  unfold gaussianFourierRate
  positivity

private noncomputable def gaussianFourierBase (σ : ℝ) (x : E) : ℂ :=
  Complex.exp (-(gaussianFourierRate σ : ℂ) * ‖x‖ ^ 2)

omit [CompleteSpace E] [SecondCountableTopology E] in
private lemma gaussianFourierBase_integrable {σ : ℝ} (hσ : 0 < σ) :
    Integrable (gaussianFourierBase (E := E) σ) := by
  unfold gaussianFourierBase
  simpa only [zero_mul, add_zero] using
    (GaussianFourier.integrable_cexp_neg_mul_sq_norm_add (V := E)
      (show 0 < (gaussianFourierRate σ : ℂ).re by
        simpa using gaussianFourierRate_pos hσ)
      0 (0 : E))

omit [MeasurableSpace E] [InnerProductSpace ℝ E] [CompleteSpace E]
  [FiniteDimensional ℝ E] [BorelSpace E] [SecondCountableTopology E] in
private lemma gaussianKernel_coe_eq_fourierBase_sub {σ : ℝ} (x y : E) :
    (Paper.gaussianKernel σ x y : ℂ) =
      gaussianFourierBase σ (x - y) := by
  unfold Paper.gaussianKernel gaussianFourierBase gaussianFourierRate
  rw [Complex.ofReal_exp]
  congr 1
  push_cast
  ring

omit [CompleteSpace E] [SecondCountableTopology E] in
private lemma gaussianKernel_coe_integrable {σ : ℝ} (hσ : 0 < σ) (y : E) :
    Integrable (fun x : E => (Paper.gaussianKernel σ x y : ℂ)) := by
  simpa only [gaussianKernel_coe_eq_fourierBase_sub] using
    (gaussianFourierBase_integrable (E := E) hσ).comp_sub_right y

omit [CompleteSpace E] in
private lemma phase_mul_gaussianKernel_integrable
    {σ : ℝ} (hσ : 0 < σ) (t y : E) :
    Integrable (fun x : E =>
      (Real.fourierChar (-⟪x, t⟫) : ℂ) *
        (Paper.gaussianKernel σ x y : ℂ)) := by
  apply (gaussianKernel_coe_integrable hσ y).bdd_mul (c := 1)
  · fun_prop
  · filter_upwards with x
    simp

omit [CompleteSpace E] [SecondCountableTopology E] in
private lemma norm_phase_mul_gaussianKernel_integral {σ : ℝ} (t y : E) :
    (∫ x : E,
        ‖(Real.fourierChar (-⟪x, t⟫) : ℂ) *
          (Paper.gaussianKernel σ x y : ℂ)‖) =
      ∫ x : E, ‖gaussianFourierBase (E := E) σ x‖ := by
  simp only [norm_mul, Circle.norm_coe, one_mul,
    gaussianKernel_coe_eq_fourierBase_sub]
  exact integral_sub_right_eq_self
    (fun x : E => ‖gaussianFourierBase (E := E) σ x‖) y

private lemma phase_mul_gaussianKernel_integrable_prod
    {σ : ℝ} (hσ : 0 < σ) (t : E) (p : Measure E)
    [IsFiniteMeasure p] :
    Integrable (fun z : E × E =>
      (Real.fourierChar (-⟪z.1, t⟫) : ℂ) *
        (Paper.gaussianKernel σ z.1 z.2 : ℂ))
      (volume.prod p) := by
  have hmeas : AEStronglyMeasurable (fun z : E × E =>
      (Real.fourierChar (-⟪z.1, t⟫) : ℂ) *
        (Paper.gaussianKernel σ z.1 z.2 : ℂ))
      (volume.prod p) := by
    apply Continuous.aestronglyMeasurable
    simp only [Real.fourierChar_apply]
    unfold Paper.gaussianKernel
    fun_prop
  refine (integrable_prod_iff' hmeas).2 ⟨?_, ?_⟩
  · exact ae_of_all _ (phase_mul_gaussianKernel_integrable hσ t)
  · have hc : Integrable
        (fun _ : E => ∫ x : E, ‖gaussianFourierBase (E := E) σ x‖) p :=
      integrable_const _
    convert hc using 1
    funext y
    exact norm_phase_mul_gaussianKernel_integral t y

omit [CompleteSpace E] [SecondCountableTopology E] in
private lemma fourier_gaussianKernel {σ : ℝ} (t y : E) :
    𝓕 (fun x : E => (Paper.gaussianKernel σ x y : ℂ)) t =
      (Real.fourierChar (-⟪y, t⟫) : ℂ) •
        𝓕 (gaussianFourierBase (E := E) σ) t := by
  have h := congrFun
    (VectorFourier.fourierIntegral_comp_add_right Real.fourierChar volume
      (innerₗ E) (gaussianFourierBase (E := E) σ) (-y)) t
  change 𝓕 (gaussianFourierBase (E := E) σ ∘ fun x : E => x + -y) t =
      Real.fourierChar (((innerₗ E) (-y)) t) •
        𝓕 (gaussianFourierBase (E := E) σ) t at h
  calc
    𝓕 (fun x : E => (Paper.gaussianKernel σ x y : ℂ)) t =
        𝓕 (gaussianFourierBase (E := E) σ ∘ fun x : E => x + -y) t := by
          apply congrArg (fun f : E → ℂ => 𝓕 f t)
          funext x
          simpa only [Function.comp_apply, sub_eq_add_neg] using
            (gaussianKernel_coe_eq_fourierBase_sub (σ := σ) x y)
    _ = Real.fourierChar (((innerₗ E) (-y)) t) •
          𝓕 (gaussianFourierBase (E := E) σ) t := h
    _ = (Real.fourierChar (-⟪y, t⟫) : ℂ) •
          𝓕 (gaussianFourierBase (E := E) σ) t := by
          simp [Circle.smul_def]

private noncomputable def complexKernelNormalizer
    (σ : ℝ) (p : Measure E) (x : E) : ℂ :=
  ∫ y, (Paper.gaussianKernel σ x y : ℂ) ∂p

omit [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [BorelSpace E] [SecondCountableTopology E] in
private lemma complexKernelNormalizer_eq_coe
    (σ : ℝ) (p : Measure E) (x : E) :
    complexKernelNormalizer σ p x =
      (Paper.kernelNormalizer (Paper.gaussianKernel σ) p x : ℂ) := by
  unfold complexKernelNormalizer Paper.kernelNormalizer
  exact integral_complex_ofReal

private lemma fourier_complexKernelNormalizer
    {σ : ℝ} (hσ : 0 < σ) (p : Measure E) [IsFiniteMeasure p] (t : E) :
    𝓕 (complexKernelNormalizer σ p) t =
      charFun p ((-2 * Real.pi) • t) *
        𝓕 (gaussianFourierBase (E := E) σ) t := by
  have hprod := phase_mul_gaussianKernel_integrable_prod hσ t p
  rw [Real.fourier_eq]
  simp only [Circle.smul_def]
  unfold complexKernelNormalizer
  calc
    (∫ x : E,
        (Real.fourierChar (-⟪x, t⟫) : ℂ) *
          ∫ y : E, (Paper.gaussianKernel σ x y : ℂ) ∂p) =
        ∫ x : E, ∫ y : E,
          (Real.fourierChar (-⟪x, t⟫) : ℂ) *
            (Paper.gaussianKernel σ x y : ℂ) ∂p := by
              congr 1
              funext x
              rw [integral_const_mul]
    _ = ∫ y : E, (∫ x : E,
          (Real.fourierChar (-⟪x, t⟫) : ℂ) *
            (Paper.gaussianKernel σ x y : ℂ)) ∂p := by
              exact integral_integral_swap
                (f := fun x y =>
                  (Real.fourierChar (-⟪x, t⟫) : ℂ) *
                    (Paper.gaussianKernel σ x y : ℂ))
                hprod
    _ = ∫ y : E,
          (Real.fourierChar (-⟪y, t⟫) : ℂ) •
            𝓕 (gaussianFourierBase (E := E) σ) t ∂p := by
              congr 1
              funext y
              rw [← fourier_gaussianKernel (σ := σ) t y, Real.fourier_eq]
              rfl
    _ = (∫ y : E, (Real.fourierChar (-⟪y, t⟫) : ℂ) ∂p) *
          𝓕 (gaussianFourierBase (E := E) σ) t := by
              simp only [smul_eq_mul]
              rw [integral_mul_const]
    _ = charFun p ((-2 * Real.pi) • t) *
          𝓕 (gaussianFourierBase (E := E) σ) t := by
              rw [charFun_apply]
              congr 2
              funext y
              simp only [Real.fourierChar_apply, real_inner_smul_right, neg_mul]
              congr 1
              push_cast
              ring

omit [CompleteSpace E] [SecondCountableTopology E] in
private lemma fourier_gaussianFourierBase_ne_zero
    {σ : ℝ} (hσ : 0 < σ) (t : E) :
    𝓕 (gaussianFourierBase (E := E) σ) t ≠ 0 := by
  unfold gaussianFourierBase
  rw [fourier_gaussian_innerProductSpace
    (show 0 < (gaussianFourierRate σ : ℂ).re by
      simpa using gaussianFourierRate_pos hσ)]
  apply mul_ne_zero
  · exact Complex.cpow_ne_zero_iff.mpr <| Or.inl <|
      div_ne_zero (Complex.ofReal_ne_zero.mpr Real.pi_ne_zero)
        (Complex.ofReal_ne_zero.mpr (ne_of_gt (gaussianFourierRate_pos hσ)))
  · exact Complex.exp_ne_zero _

/-- Equality of all unnormalized Gaussian kernel normalizers determines a
finite-dimensional probability measure.  The proof Fourier-transforms the
normalizer, cancels the nowhere-zero Gaussian transform, and invokes
characteristic-function uniqueness. -/
theorem gaussianKernelNormalizer_injective
    (σ : ℝ) (hσ : Paper.ValidBandwidth σ)
    (p q : Measure E) [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (h : ∀ x,
      Paper.kernelNormalizer (Paper.gaussianKernel σ) p x =
        Paper.kernelNormalizer (Paper.gaussianKernel σ) q x) :
    p = q := by
  apply Measure.ext_of_charFun
  funext s
  obtain ⟨t, rfl⟩ : ∃ t : E, (-2 * Real.pi) • t = s := by
    refine ⟨((-2 * Real.pi)⁻¹) • s, ?_⟩
    rw [smul_smul]
    convert one_smul ℝ s using 1
    field_simp [Real.pi_ne_zero]
  have hcomplex :
      complexKernelNormalizer σ p = complexKernelNormalizer σ q := by
    funext x
    rw [complexKernelNormalizer_eq_coe, complexKernelNormalizer_eq_coe, h x]
  have hfourier :=
    congrFun (congrArg (fun f : E → ℂ => 𝓕 f) hcomplex) t
  rw [fourier_complexKernelNormalizer hσ,
    fourier_complexKernelNormalizer hσ] at hfourier
  exact mul_right_cancel₀
    (fourier_gaussianFourierBase_ne_zero hσ t) hfourier

/-- If two probability measures have Gaussian kernel normalizers differing by
a global scalar, that scalar is one. This is the normalization step needed
after equality of Gaussian scores yields proportional smoothings. -/
theorem gaussianKernelNormalizer_proportional_constant_eq_one
    (σ : ℝ) (hσ : Paper.ValidBandwidth σ)
    (p q : Measure E) [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (c : ℝ)
    (h : ∀ x,
      Paper.kernelNormalizer (Paper.gaussianKernel σ) p x =
        c * Paper.kernelNormalizer (Paper.gaussianKernel σ) q x) :
    c = 1 := by
  have hcomplex :
      complexKernelNormalizer σ p =
        (c : ℂ) • complexKernelNormalizer σ q := by
    funext x
    simp only [Pi.smul_apply, smul_eq_mul]
    rw [complexKernelNormalizer_eq_coe, complexKernelNormalizer_eq_coe]
    exact_mod_cast h x
  have hfourier_smul := congrFun
    (VectorFourier.fourierIntegral_const_smul Real.fourierChar volume
      (innerₗ E) (complexKernelNormalizer σ q) (c : ℂ)) 0
  change 𝓕 ((c : ℂ) • complexKernelNormalizer σ q) 0 =
      (c : ℂ) • 𝓕 (complexKernelNormalizer σ q) 0 at hfourier_smul
  have hfourier :=
    congrFun (congrArg (fun f : E → ℂ => 𝓕 f) hcomplex) 0
  rw [fourier_complexKernelNormalizer hσ, hfourier_smul,
    fourier_complexKernelNormalizer hσ] at hfourier
  simp only [smul_zero, charFun_zero, probReal_univ, Complex.ofReal_one,
    one_mul] at hfourier
  have hc : (c : ℂ) = 1 := by
    apply mul_right_cancel₀
      (fourier_gaussianFourierBase_ne_zero hσ (0 : E))
    simpa [mul_comm] using hfourier.symm
  exact_mod_cast hc

end GaussianKernelNormalizer

end DriftingIdentifiability
