import DriftingIdentifiability.LaplaceCompanion

/-!
# Laplace smoothing injectivity on the line

Stage 2 of the attack on the open Laplace-kernel arbitrary-target converse
(`LaplaceArbitraryConverse.md`).  This file proves that the paper's Laplace
kernel smoothing determines a finite measure on `ℝ`:

`(∀ x, ∫ e^{-|x-y|/τ} dμ = ∫ e^{-|x-y|/τ} dν)  →  μ = ν`.

Mathlib has no ready Fourier transform for the two-sided exponential, so it is
computed here from first principles: on each half-line the phase-twisted
profile is a complex exponential `exp (c x)` with nonzero real part, whose
antiderivative decays, so the fundamental theorem of calculus on semi-infinite
intervals evaluates the positive half, and reflection handles the negative
half.  The result

`𝓕 (e^{-|·|/τ}) (t) = (2/τ) / ((1/τ)² + (2πt)²)`

is real, positive, and nowhere zero.  The remainder mirrors the
battle-tested Gaussian skeleton (`GaussianConvolutionInjectivity.lean`):
Fourier-transform the kernel normalizer, factor it through the characteristic
function by Fubini and translation invariance, cancel the nowhere-zero
profile transform, and finish with characteristic-function uniqueness.
Axiom-free throughout.
-/

open MeasureTheory ProbabilityTheory Set Filter Topology
open scoped FourierTransform RealInnerProductSpace Real

namespace DriftingIdentifiability

/-! ## Half-line integrals of decaying complex exponentials -/

private lemma cexp_mul_hasDerivAt {c : ℂ} (hc : c ≠ 0) (x : ℝ) :
    HasDerivAt (fun x : ℝ => Complex.exp (c * x) / c)
      (Complex.exp (c * x)) x := by
  have h1 : HasDerivAt (fun x : ℝ => c * (x : ℂ)) c x := by
    simpa using (Complex.ofRealCLM.hasDerivAt (x := x)).const_mul c
  have h2 := (h1.cexp).div_const c
  have hne : Complex.exp (c * x) * c / c = Complex.exp (c * x) := by
    field_simp
  rwa [hne] at h2

private lemma cexp_integrableOn_Ioi {c : ℂ} (hc : c.re < 0) :
    IntegrableOn (fun x : ℝ => Complex.exp (c * x)) (Ioi 0) := by
  have hb : (0 : ℝ) < -c.re := by linarith
  refine (exp_neg_integrableOn_Ioi 0 hb).mono' ?_ ?_
  · apply Continuous.aestronglyMeasurable
    fun_prop
  · filter_upwards with x
    rw [Complex.norm_exp]
    have h1 : (c * (x : ℂ)).re = c.re * x := by
      simp [Complex.mul_re]
    have h2 : -(-c.re) * x = c.re * x := by ring
    rw [h1, h2]

private lemma cexp_tendsto_atTop_zero {c : ℂ} (hc : c.re < 0) :
    Tendsto (fun x : ℝ => Complex.exp (c * x) / c) atTop (𝓝 0) := by
  rw [tendsto_zero_iff_norm_tendsto_zero]
  simp only [norm_div, Complex.norm_exp]
  have hre : ∀ x : ℝ, (c * (x : ℂ)).re = c.re * x := by
    intro x
    simp [Complex.mul_re]
  simp only [hre]
  have h1 : Tendsto (fun x : ℝ => c.re * x) atTop atBot :=
    (tendsto_const_mul_atBot_of_neg hc).mpr tendsto_id
  have h2 : Tendsto (fun x : ℝ => Real.exp (c.re * x)) atTop (𝓝 0) :=
    Real.tendsto_exp_atBot.comp h1
  simpa using h2.div_const ‖c‖

/-- `∫_{(0,∞)} e^{cx} dx = -1/c` for `Re c < 0`. -/
private lemma integral_cexp_Ioi {c : ℂ} (hc : c.re < 0) :
    ∫ x : ℝ in Ioi 0, Complex.exp (c * x) = -c⁻¹ := by
  have hcne : c ≠ 0 := fun h => by simp [h] at hc
  have h := integral_Ioi_of_hasDerivAt_of_tendsto'
    (f := fun x : ℝ => Complex.exp (c * x) / c)
    (fun x _ => cexp_mul_hasDerivAt hcne x)
    (cexp_integrableOn_Ioi hc) (cexp_tendsto_atTop_zero hc)
  rw [h]
  simp [one_div]

private lemma cexp_integrableOn_Iic {c : ℂ} (hc : 0 < c.re) :
    IntegrableOn (fun x : ℝ => Complex.exp (c * x)) (Iic 0) := by
  rw [← Measure.map_neg_eq_self (volume : Measure ℝ)]
  have m : MeasurableEmbedding fun x : ℝ => -x :=
    (Homeomorph.neg ℝ).measurableEmbedding
  rw [m.integrableOn_map_iff]
  have hfun : ((fun x : ℝ => Complex.exp (c * x)) ∘ fun x : ℝ => -x) =
      fun x : ℝ => Complex.exp ((-c) * x) := by
    funext x
    simp only [Function.comp_apply, Complex.ofReal_neg]
    congr 1
    ring
  rw [hfun]
  simp only [neg_preimage, neg_Iic, neg_zero]
  rw [integrableOn_Ici_iff_integrableOn_Ioi]
  exact cexp_integrableOn_Ioi (by rw [Complex.neg_re]; linarith)

/-- `∫_{(-∞,0]} e^{cx} dx = 1/c` for `Re c > 0`, by reflection. -/
private lemma integral_cexp_Iic {c : ℂ} (hc : 0 < c.re) :
    ∫ x : ℝ in Iic 0, Complex.exp (c * x) = c⁻¹ := by
  have h := integral_comp_neg_Ioi (0 : ℝ) (fun x : ℝ => Complex.exp (c * x))
  rw [neg_zero] at h
  rw [← h]
  have hfun : ∀ x : ℝ, Complex.exp (c * ((-x : ℝ) : ℂ)) =
      Complex.exp ((-c) * x) := by
    intro x
    congr 1
    push_cast
    ring
  simp_rw [hfun]
  rw [integral_cexp_Ioi (by rw [Complex.neg_re]; linarith)]
  simp

/-! ## The Fourier transform of the two-sided exponential profile -/

private noncomputable def laplaceFourierBase (τ : ℝ) (x : ℝ) : ℂ :=
  (Real.exp (-(1 / τ) * |x|) : ℂ)

private lemma laplaceFourierBase_integrable {τ : ℝ} (hτ : 0 < τ) :
    Integrable (laplaceFourierBase τ) := by
  have hb : (0 : ℝ) < 1 / τ := by positivity
  have h1 : IntegrableOn (fun x : ℝ => Real.exp (-(1 / τ) * |x|)) (Ioi 0) := by
    refine ((exp_neg_integrableOn_Ioi 0 hb).congr_fun ?_ measurableSet_Ioi)
    intro x hx
    simp only
    rw [abs_of_pos (Set.mem_Ioi.mp hx)]
  have h2 : IntegrableOn (fun x : ℝ => Real.exp (-(1 / τ) * |x|)) (Iic 0) := by
    rw [← Measure.map_neg_eq_self (volume : Measure ℝ)]
    have m : MeasurableEmbedding fun x : ℝ => -x :=
      (Homeomorph.neg ℝ).measurableEmbedding
    rw [m.integrableOn_map_iff]
    have hfun : ((fun x : ℝ => Real.exp (-(1 / τ) * |x|)) ∘ fun x : ℝ => -x) =
        fun x : ℝ => Real.exp (-(1 / τ) * |x|) := by
      funext x
      simp [Function.comp_apply, abs_neg]
    rw [hfun]
    simp only [neg_preimage, neg_Iic, neg_zero]
    rw [integrableOn_Ici_iff_integrableOn_Ioi]
    exact h1
  have hreal : Integrable (fun x : ℝ => Real.exp (-(1 / τ) * |x|)) := by
    rw [← integrableOn_univ, ← Iic_union_Ioi (a := (0 : ℝ))]
    exact h2.union h1
  exact hreal.ofReal

/-- The closed-form transform value: real and strictly positive. -/
private noncomputable def laplaceFourierValue (τ t : ℝ) : ℝ :=
  (2 / τ) / ((1 / τ) ^ 2 + (2 * π * t) ^ 2)

private lemma laplaceFourierValue_pos {τ : ℝ} (hτ : 0 < τ) (t : ℝ) :
    0 < laplaceFourierValue τ t := by
  unfold laplaceFourierValue
  positivity

/-- The phase-twisted profile on the positive half-line. -/
private lemma phase_base_eq_pos {τ : ℝ} (t : ℝ) {x : ℝ} (hx : 0 < x) :
    Complex.exp (((-2 * π * x * t : ℝ) : ℂ) * Complex.I) •
        laplaceFourierBase τ x =
      Complex.exp ((-(1 / τ : ℂ) - (2 * π * t : ℝ) * Complex.I) * x) := by
  unfold laplaceFourierBase
  rw [abs_of_pos hx, Complex.ofReal_exp, smul_eq_mul, ← Complex.exp_add]
  congr 1
  push_cast
  ring

/-- The phase-twisted profile on the negative half-line. -/
private lemma phase_base_eq_neg {τ : ℝ} (t : ℝ) {x : ℝ} (hx : x ≤ 0) :
    Complex.exp (((-2 * π * x * t : ℝ) : ℂ) * Complex.I) •
        laplaceFourierBase τ x =
      Complex.exp (((1 / τ : ℂ) - (2 * π * t : ℝ) * Complex.I) * x) := by
  unfold laplaceFourierBase
  rw [abs_of_nonpos hx, Complex.ofReal_exp, smul_eq_mul, ← Complex.exp_add]
  congr 1
  push_cast
  ring

private lemma phase_base_integrable {τ : ℝ} (hτ : 0 < τ) (t : ℝ) :
    Integrable (fun x : ℝ =>
      Complex.exp (((-2 * π * x * t : ℝ) : ℂ) * Complex.I) •
        laplaceFourierBase τ x) := by
  have h : Integrable (fun x : ℝ =>
      Complex.exp (((-2 * π * x * t : ℝ) : ℂ) * Complex.I) *
        laplaceFourierBase τ x) := by
    apply (laplaceFourierBase_integrable hτ).bdd_mul (c := 1)
    · apply Continuous.aestronglyMeasurable
      fun_prop
    · filter_upwards with x
      simp [Complex.norm_exp]
  apply h.congr
  filter_upwards with x
  rw [smul_eq_mul]

/-- **The Fourier transform of the two-sided exponential**, computed from
first principles.  It is a positive real number for every frequency. -/
private lemma fourier_laplaceFourierBase {τ : ℝ} (hτ : 0 < τ) (t : ℝ) :
    𝓕 (laplaceFourierBase τ) t = (laplaceFourierValue τ t : ℂ) := by
  have hb : (0 : ℝ) < 1 / τ := by positivity
  set cP : ℂ := -(1 / τ : ℂ) - (2 * π * t : ℝ) * Complex.I with hcP
  set cN : ℂ := (1 / τ : ℂ) - (2 * π * t : ℝ) * Complex.I with hcN
  have hcPre : cP.re = -(1 / τ) := by
    simp [hcP, Complex.sub_re, Complex.neg_re, Complex.ofReal_re]
  have hcNre : cN.re = 1 / τ := by
    simp [hcN, Complex.sub_re, Complex.ofReal_re]
  have hIoi : IntegrableOn (fun x : ℝ =>
      Complex.exp (((-2 * π * x * t : ℝ) : ℂ) * Complex.I) •
        laplaceFourierBase τ x) (Ioi 0) :=
    (phase_base_integrable hτ t).integrableOn
  have hIic : IntegrableOn (fun x : ℝ =>
      Complex.exp (((-2 * π * x * t : ℝ) : ℂ) * Complex.I) •
        laplaceFourierBase τ x) (Iic 0) :=
    (phase_base_integrable hτ t).integrableOn
  rw [Real.fourier_real_eq_integral_exp_smul]
  rw [← setIntegral_univ, ← Iic_union_Ioi (a := (0 : ℝ)),
    setIntegral_union (Iic_disjoint_Ioi le_rfl) measurableSet_Ioi hIic hIoi]
  have hIoiEq : (∫ x in Ioi (0 : ℝ),
      Complex.exp (((-2 * π * x * t : ℝ) : ℂ) * Complex.I) •
        laplaceFourierBase τ x) = -cP⁻¹ := by
    rw [setIntegral_congr_fun measurableSet_Ioi
      (fun x hx => phase_base_eq_pos t (Set.mem_Ioi.mp hx))]
    exact integral_cexp_Ioi (by rw [hcPre]; linarith)
  have hIicEq : (∫ x in Iic (0 : ℝ),
      Complex.exp (((-2 * π * x * t : ℝ) : ℂ) * Complex.I) •
        laplaceFourierBase τ x) = cN⁻¹ := by
    rw [setIntegral_congr_fun measurableSet_Iic
      (fun x hx => phase_base_eq_neg t (Set.mem_Iic.mp hx))]
    exact integral_cexp_Iic (by rw [hcNre]; exact hb)
  rw [hIoiEq, hIicEq]
  -- final algebra: 1/cN - 1/cP = (2/τ) / ((1/τ)² + (2πt)²)
  have hcPne : cP ≠ 0 := by
    intro h
    have := congrArg Complex.re h
    rw [hcPre] at this
    simp at this
    linarith
  have hcNne : cN ≠ 0 := by
    intro h
    have := congrArg Complex.re h
    rw [hcNre] at this
    simp at this
    linarith
  have hprod : cN * cP = -((((1 / τ) ^ 2 + (2 * π * t) ^ 2 : ℝ)) : ℂ) := by
    rw [hcN, hcP]
    push_cast
    ring_nf
    rw [Complex.I_sq]
    ring
  have hsub : cP - cN = -((2 / τ : ℝ) : ℂ) := by
    rw [hcN, hcP]
    push_cast
    ring
  have hkey : cN⁻¹ + -cP⁻¹ = (cP - cN) / (cN * cP) := by
    field_simp
    ring
  rw [hkey, hsub, hprod, neg_div_neg_eq]
  unfold laplaceFourierValue
  push_cast
  ring

private lemma fourier_laplaceFourierBase_ne_zero {τ : ℝ} (hτ : 0 < τ) (t : ℝ) :
    𝓕 (laplaceFourierBase τ) t ≠ 0 := by
  rw [fourier_laplaceFourierBase hτ t]
  exact_mod_cast (laplaceFourierValue_pos hτ t).ne'

/-! ## Mirrored injectivity skeleton for the Laplace kernel on `ℝ` -/

private lemma laplaceKernel_coe_eq_base_sub {τ : ℝ} (x y : ℝ) :
    (Paper.laplaceKernel τ x y : ℂ) = laplaceFourierBase τ (x - y) := by
  unfold Paper.laplaceKernel laplaceFourierBase
  rw [Real.norm_eq_abs]

private lemma laplaceKernel_coe_integrable {τ : ℝ} (hτ : 0 < τ) (y : ℝ) :
    Integrable (fun x : ℝ => (Paper.laplaceKernel τ x y : ℂ)) := by
  simpa only [laplaceKernel_coe_eq_base_sub] using
    (laplaceFourierBase_integrable hτ).comp_sub_right y

private lemma phase_mul_laplaceKernel_integrable
    {τ : ℝ} (hτ : 0 < τ) (t y : ℝ) :
    Integrable (fun x : ℝ =>
      (Real.fourierChar (-⟪x, t⟫) : ℂ) * (Paper.laplaceKernel τ x y : ℂ)) := by
  apply (laplaceKernel_coe_integrable hτ y).bdd_mul (c := 1)
  · apply Continuous.aestronglyMeasurable
    simp only [Real.fourierChar_apply]
    fun_prop
  · filter_upwards with x
    simp

private lemma norm_phase_mul_laplaceKernel_integral {τ : ℝ} (t y : ℝ) :
    (∫ x : ℝ,
        ‖(Real.fourierChar (-⟪x, t⟫) : ℂ) *
          (Paper.laplaceKernel τ x y : ℂ)‖) =
      ∫ x : ℝ, ‖laplaceFourierBase τ x‖ := by
  simp only [norm_mul, Circle.norm_coe, one_mul,
    laplaceKernel_coe_eq_base_sub]
  exact integral_sub_right_eq_self
    (fun x : ℝ => ‖laplaceFourierBase τ x‖) y

private lemma phase_mul_laplaceKernel_integrable_prod
    {τ : ℝ} (hτ : 0 < τ) (t : ℝ) (p : Measure ℝ) [IsFiniteMeasure p] :
    Integrable (fun z : ℝ × ℝ =>
      (Real.fourierChar (-⟪z.1, t⟫) : ℂ) *
        (Paper.laplaceKernel τ z.1 z.2 : ℂ))
      (volume.prod p) := by
  have hmeas : AEStronglyMeasurable (fun z : ℝ × ℝ =>
      (Real.fourierChar (-⟪z.1, t⟫) : ℂ) *
        (Paper.laplaceKernel τ z.1 z.2 : ℂ))
      (volume.prod p) := by
    apply Continuous.aestronglyMeasurable
    simp only [Real.fourierChar_apply]
    unfold Paper.laplaceKernel
    fun_prop
  refine (integrable_prod_iff' hmeas).2 ⟨?_, ?_⟩
  · exact ae_of_all _ (phase_mul_laplaceKernel_integrable hτ t)
  · have hc : Integrable
        (fun _ : ℝ => ∫ x : ℝ, ‖laplaceFourierBase τ x‖) p :=
      integrable_const _
    convert hc using 1
    funext y
    exact norm_phase_mul_laplaceKernel_integral t y

private lemma fourier_laplaceKernel {τ : ℝ} (t y : ℝ) :
    𝓕 (fun x : ℝ => (Paper.laplaceKernel τ x y : ℂ)) t =
      (Real.fourierChar (-⟪y, t⟫) : ℂ) •
        𝓕 (laplaceFourierBase τ) t := by
  have h := congrFun
    (VectorFourier.fourierIntegral_comp_add_right Real.fourierChar volume
      (innerₗ ℝ) (laplaceFourierBase τ) (-y)) t
  change 𝓕 (laplaceFourierBase τ ∘ fun x : ℝ => x + -y) t =
      Real.fourierChar (((innerₗ ℝ) (-y)) t) •
        𝓕 (laplaceFourierBase τ) t at h
  calc
    𝓕 (fun x : ℝ => (Paper.laplaceKernel τ x y : ℂ)) t =
        𝓕 (laplaceFourierBase τ ∘ fun x : ℝ => x + -y) t := by
          apply congrArg (fun f : ℝ → ℂ => 𝓕 f t)
          funext x
          simpa only [Function.comp_apply, sub_eq_add_neg] using
            (laplaceKernel_coe_eq_base_sub (τ := τ) x y)
    _ = Real.fourierChar (((innerₗ ℝ) (-y)) t) •
          𝓕 (laplaceFourierBase τ) t := h
    _ = (Real.fourierChar (-⟪y, t⟫) : ℂ) •
          𝓕 (laplaceFourierBase τ) t := by
          simp [Circle.smul_def]

private noncomputable def complexLaplaceNormalizer
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℂ :=
  ∫ y, (Paper.laplaceKernel τ x y : ℂ) ∂p

private lemma complexLaplaceNormalizer_eq_coe
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) :
    complexLaplaceNormalizer τ p x =
      (Paper.kernelNormalizer (Paper.laplaceKernel τ) p x : ℂ) := by
  unfold complexLaplaceNormalizer Paper.kernelNormalizer
  exact integral_complex_ofReal

private lemma fourier_complexLaplaceNormalizer
    {τ : ℝ} (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p] (t : ℝ) :
    𝓕 (complexLaplaceNormalizer τ p) t =
      charFun p ((-2 * π) • t) * 𝓕 (laplaceFourierBase τ) t := by
  have hprod := phase_mul_laplaceKernel_integrable_prod hτ t p
  rw [Real.fourier_eq]
  simp only [Circle.smul_def]
  unfold complexLaplaceNormalizer
  calc
    (∫ x : ℝ,
        (Real.fourierChar (-⟪x, t⟫) : ℂ) *
          ∫ y : ℝ, (Paper.laplaceKernel τ x y : ℂ) ∂p) =
        ∫ x : ℝ, ∫ y : ℝ,
          (Real.fourierChar (-⟪x, t⟫) : ℂ) *
            (Paper.laplaceKernel τ x y : ℂ) ∂p := by
              congr 1
              funext x
              rw [integral_const_mul]
    _ = ∫ y : ℝ, (∫ x : ℝ,
          (Real.fourierChar (-⟪x, t⟫) : ℂ) *
            (Paper.laplaceKernel τ x y : ℂ)) ∂p := by
              exact integral_integral_swap
                (f := fun x y =>
                  (Real.fourierChar (-⟪x, t⟫) : ℂ) *
                    (Paper.laplaceKernel τ x y : ℂ))
                hprod
    _ = ∫ y : ℝ,
          (Real.fourierChar (-⟪y, t⟫) : ℂ) •
            𝓕 (laplaceFourierBase τ) t ∂p := by
              congr 1
              funext y
              rw [← fourier_laplaceKernel (τ := τ) t y, Real.fourier_eq]
              rfl
    _ = (∫ y : ℝ, (Real.fourierChar (-⟪y, t⟫) : ℂ) ∂p) *
          𝓕 (laplaceFourierBase τ) t := by
              simp only [smul_eq_mul]
              rw [integral_mul_const]
    _ = charFun p ((-2 * π) • t) * 𝓕 (laplaceFourierBase τ) t := by
              rw [charFun_apply]
              congr 2
              funext y
              simp only [Real.fourierChar_apply, real_inner_smul_right, neg_mul]
              congr 1
              push_cast
              ring

/-- **Laplace smoothing injectivity on the line.**  Equality of all Laplace
kernel normalizers determines a finite measure on `ℝ`: Fourier-transform the
normalizer, factor through the characteristic function, cancel the
nowhere-zero transform of the two-sided exponential, and apply
characteristic-function uniqueness.  This is the analytic heart of Stage 2 of
`LaplaceArbitraryConverse.md`. -/
theorem laplaceKernelNormalizer_injective
    (τ : ℝ) (hτ : Paper.ValidBandwidth τ)
    (p q : Measure ℝ) [IsFiniteMeasure p] [IsFiniteMeasure q]
    (h : ∀ x,
      Paper.kernelNormalizer (Paper.laplaceKernel τ) p x =
        Paper.kernelNormalizer (Paper.laplaceKernel τ) q x) :
    p = q := by
  have hτ0 : 0 < τ := hτ
  apply Measure.ext_of_charFun
  funext s
  obtain ⟨t, rfl⟩ : ∃ t : ℝ, (-2 * π) • t = s := by
    refine ⟨((-2 * π)⁻¹) • s, ?_⟩
    rw [smul_smul]
    convert one_smul ℝ s using 1
    field_simp [Real.pi_ne_zero]
  have hcomplex :
      complexLaplaceNormalizer τ p = complexLaplaceNormalizer τ q := by
    funext x
    rw [complexLaplaceNormalizer_eq_coe, complexLaplaceNormalizer_eq_coe, h x]
  have hfourier :=
    congrFun (congrArg (fun f : ℝ → ℂ => 𝓕 f) hcomplex) t
  rw [fourier_complexLaplaceNormalizer hτ0,
    fourier_complexLaplaceNormalizer hτ0] at hfourier
  exact mul_right_cancel₀
    (fourier_laplaceFourierBase_ne_zero hτ0 t) hfourier

/-! ## Stage-2 interface and Dirac-rigidity infrastructure -/

/-- A kernel-level injectivity predicate for Laplace smoothing.  This predicate
is intentionally about normalizers of arbitrary finite measures; it does not
assert any mean-shift identifiability conclusion. -/
def LaplaceSmoothingInjective
    (E : Type*) [MeasurableSpace E] [NormedAddCommGroup E] (τ : ℝ) : Prop :=
  ∀ p q : Measure E, IsFiniteMeasure p → IsFiniteMeasure q →
    (∀ x : E,
      Paper.kernelNormalizer (Paper.laplaceKernel τ) p x =
        Paper.kernelNormalizer (Paper.laplaceKernel τ) q x) →
      p = q

/-- The Fourier calculation above gives Laplace-smoothing injectivity on the
real line for every positive bandwidth. -/
theorem laplaceSmoothingInjective_real
    (τ : ℝ) (hτ : Paper.ValidBandwidth τ) :
    LaplaceSmoothingInjective ℝ τ := by
  intro p q hp hq h
  letI := hp
  letI := hq
  exact laplaceKernelNormalizer_injective τ hτ p q h

/-- Positive part of the signed first-moment measure `(y-c)·p`. -/
noncomputable def laplacePositiveMomentMeasure (p : Measure ℝ) (c : ℝ) :
    Measure ℝ :=
  p.withDensity fun y => ENNReal.ofReal (y - c)

/-- Negative part of the signed first-moment measure `(y-c)·p`. -/
noncomputable def laplaceNegativeMomentMeasure (p : Measure ℝ) (c : ℝ) :
    Measure ℝ :=
  p.withDensity fun y => ENNReal.ofReal (c - y)

private lemma lintegral_ofReal_norm_sub_lt_top
    (p : Measure ℝ) (c : ℝ)
    (hp : Integrable (fun y : ℝ => ‖y - c‖) p) :
    (∫⁻ y, ENNReal.ofReal (‖y - c‖) ∂p) < ⊤ := by
  have hfin := hp.hasFiniteIntegral
  rw [hasFiniteIntegral_iff_enorm] at hfin
  have hEq : (fun y : ℝ => ENNReal.ofReal (‖y - c‖)) =
      fun y : ℝ => ‖(‖y - c‖ : ℝ)‖ₑ := by
    funext y
    exact (Real.enorm_eq_ofReal (norm_nonneg (y - c))).symm
  rw [hEq]
  exact hfin

private lemma isFiniteMeasure_laplacePositiveMomentMeasure
    (p : Measure ℝ) (c : ℝ)
    (hp : Integrable (fun y : ℝ => ‖y - c‖) p) :
    IsFiniteMeasure (laplacePositiveMomentMeasure p c) := by
  rw [laplacePositiveMomentMeasure]
  apply isFiniteMeasure_withDensity
  exact ne_of_lt (lt_of_le_of_lt
    (lintegral_mono fun y => ENNReal.ofReal_le_ofReal (le_abs_self (y - c)))
    (lintegral_ofReal_norm_sub_lt_top p c hp))

private lemma isFiniteMeasure_laplaceNegativeMomentMeasure
    (p : Measure ℝ) (c : ℝ)
    (hp : Integrable (fun y : ℝ => ‖y - c‖) p) :
    IsFiniteMeasure (laplaceNegativeMomentMeasure p c) := by
  rw [laplaceNegativeMomentMeasure]
  apply isFiniteMeasure_withDensity
  exact ne_of_lt (lt_of_le_of_lt
    (lintegral_mono fun y => ENNReal.ofReal_le_ofReal (le_abs_self (c - y)))
    (by
      simpa [norm_sub_rev] using lintegral_ofReal_norm_sub_lt_top p c hp))

private lemma toReal_ofReal_sub_toReal_ofReal_neg (a : ℝ) :
    (ENNReal.ofReal a).toReal - (ENNReal.ofReal (-a)).toReal = a := by
  by_cases ha : 0 ≤ a
  · rw [ENNReal.toReal_ofReal ha]
    have hzero : (ENNReal.ofReal (-a)).toReal = 0 := by
      rw [ENNReal.toReal_ofReal']
      exact max_eq_right (by linarith)
    rw [hzero]
    · ring
  · have hneg : 0 ≤ -a := by linarith
    have hnonpos : a ≤ 0 := le_of_not_ge ha
    rw [ENNReal.toReal_ofReal hneg]
    have hzero : (ENNReal.ofReal a).toReal = 0 := by
      rw [ENNReal.toReal_ofReal']
      exact max_eq_right hnonpos
    rw [hzero]
    ring

private lemma kernelNormalizer_laplacePositiveMomentMeasure
    (τ : ℝ) (p : Measure ℝ) (c x : ℝ) :
    Paper.kernelNormalizer (Paper.laplaceKernel τ)
        (laplacePositiveMomentMeasure p c) x =
      ∫ y, (ENNReal.ofReal (y - c)).toReal *
        Paper.laplaceKernel τ x y ∂p := by
  rw [Paper.kernelNormalizer, laplacePositiveMomentMeasure,
    integral_withDensity_eq_integral_toReal_smul
      (by fun_prop : Measurable fun y : ℝ => ENNReal.ofReal (y - c))
      (Filter.Eventually.of_forall fun _ => ENNReal.ofReal_lt_top)
      (fun y : ℝ => Paper.laplaceKernel τ x y)]
  simp [smul_eq_mul]

private lemma kernelNormalizer_laplaceNegativeMomentMeasure
    (τ : ℝ) (p : Measure ℝ) (c x : ℝ) :
    Paper.kernelNormalizer (Paper.laplaceKernel τ)
        (laplaceNegativeMomentMeasure p c) x =
      ∫ y, (ENNReal.ofReal (c - y)).toReal *
        Paper.laplaceKernel τ x y ∂p := by
  rw [Paper.kernelNormalizer, laplaceNegativeMomentMeasure,
    integral_withDensity_eq_integral_toReal_smul
      (by fun_prop : Measurable fun y : ℝ => ENNReal.ofReal (c - y))
      (Filter.Eventually.of_forall fun _ => ENNReal.ofReal_lt_top)
      (fun y : ℝ => Paper.laplaceKernel τ x y)]
  simp [smul_eq_mul]

private lemma laplacePositiveNegative_normalizers_eq_of_signed_moment_zero
    (τ : ℝ) (hτ : Paper.ValidBandwidth τ) (p : Measure ℝ) (c : ℝ)
    (hp : Integrable (fun y : ℝ => ‖y - c‖) p)
    (hmoment : ∀ x : ℝ,
      ∫ y, Paper.laplaceKernel τ x y * (y - c) ∂p = 0) :
    ∀ x : ℝ,
      Paper.kernelNormalizer (Paper.laplaceKernel τ)
          (laplacePositiveMomentMeasure p c) x =
        Paper.kernelNormalizer (Paper.laplaceKernel τ)
          (laplaceNegativeMomentMeasure p c) x := by
  intro x
  rw [kernelNormalizer_laplacePositiveMomentMeasure,
    kernelNormalizer_laplaceNegativeMomentMeasure]
  have hposInt : Integrable
      (fun y : ℝ => (ENNReal.ofReal (y - c)).toReal *
        Paper.laplaceKernel τ x y) p := by
    refine hp.mono' ?_ ?_
    · have hfun :
          (fun y : ℝ => (ENNReal.ofReal (y - c)).toReal *
            Paper.laplaceKernel τ x y) =
          fun y : ℝ => max (y - c) 0 * Paper.laplaceKernel τ x y := by
        funext y
        rw [ENNReal.toReal_ofReal']
      rw [hfun]
      apply Continuous.aestronglyMeasurable
      unfold Paper.laplaceKernel
      fun_prop
    · filter_upwards with y
      rw [Real.norm_eq_abs]
      have hk_le : Paper.laplaceKernel τ x y ≤ 1 := by
        unfold Paper.laplaceKernel
        rw [Real.exp_le_one_iff]
        have hτ0 : 0 < τ := hτ
        have hτinv : 0 ≤ 1 / τ := by
          simpa [one_div] using inv_nonneg.mpr hτ0.le
        nlinarith [norm_nonneg (x - y)]
      have hk_nonneg : 0 ≤ Paper.laplaceKernel τ x y := by
        unfold Paper.laplaceKernel
        positivity
      have hto : (ENNReal.ofReal (y - c)).toReal ≤ ‖y - c‖ := by
        rw [ENNReal.toReal_ofReal']
        exact max_le (le_abs_self _) (norm_nonneg _)
      have hto_nonneg : 0 ≤ (ENNReal.ofReal (y - c)).toReal :=
        ENNReal.toReal_nonneg
      calc |(ENNReal.ofReal (y - c)).toReal * Paper.laplaceKernel τ x y|
          = (ENNReal.ofReal (y - c)).toReal * Paper.laplaceKernel τ x y := by
              rw [abs_of_nonneg (mul_nonneg hto_nonneg hk_nonneg)]
        _ ≤ ‖y - c‖ := by nlinarith
  have hnegInt : Integrable
      (fun y : ℝ => (ENNReal.ofReal (c - y)).toReal *
        Paper.laplaceKernel τ x y) p := by
    refine hp.mono' ?_ ?_
    · have hfun :
          (fun y : ℝ => (ENNReal.ofReal (c - y)).toReal *
            Paper.laplaceKernel τ x y) =
          fun y : ℝ => max (c - y) 0 * Paper.laplaceKernel τ x y := by
        funext y
        rw [ENNReal.toReal_ofReal']
      rw [hfun]
      apply Continuous.aestronglyMeasurable
      unfold Paper.laplaceKernel
      fun_prop
    · filter_upwards with y
      rw [Real.norm_eq_abs]
      have hk_le : Paper.laplaceKernel τ x y ≤ 1 := by
        unfold Paper.laplaceKernel
        rw [Real.exp_le_one_iff]
        have hτ0 : 0 < τ := hτ
        have hτinv : 0 ≤ 1 / τ := by
          simpa [one_div] using inv_nonneg.mpr hτ0.le
        nlinarith [norm_nonneg (x - y)]
      have hk_nonneg : 0 ≤ Paper.laplaceKernel τ x y := by
        unfold Paper.laplaceKernel
        positivity
      have hto : (ENNReal.ofReal (c - y)).toReal ≤ ‖y - c‖ := by
        rw [ENNReal.toReal_ofReal']
        have habs : |c - y| = ‖y - c‖ := by
          rw [Real.norm_eq_abs, abs_sub_comm]
        exact max_le (by simpa [habs] using le_abs_self (c - y)) (norm_nonneg _)
      have hto_nonneg : 0 ≤ (ENNReal.ofReal (c - y)).toReal :=
        ENNReal.toReal_nonneg
      calc |(ENNReal.ofReal (c - y)).toReal * Paper.laplaceKernel τ x y|
          = (ENNReal.ofReal (c - y)).toReal * Paper.laplaceKernel τ x y := by
              rw [abs_of_nonneg (mul_nonneg hto_nonneg hk_nonneg)]
        _ ≤ ‖y - c‖ := by nlinarith
  have hcalc :
      (fun y : ℝ =>
          (ENNReal.ofReal (y - c)).toReal * Paper.laplaceKernel τ x y -
            (ENNReal.ofReal (c - y)).toReal * Paper.laplaceKernel τ x y)
        =
      fun y : ℝ => Paper.laplaceKernel τ x y * (y - c) := by
    funext y
    have hsplit := toReal_ofReal_sub_toReal_ofReal_neg (y - c)
    rw [show c - y = -(y - c) by ring]
    rw [← sub_mul, hsplit]
    ring
  have hdiff :
      (∫ y, (ENNReal.ofReal (y - c)).toReal *
          Paper.laplaceKernel τ x y ∂p) -
        (∫ y, (ENNReal.ofReal (c - y)).toReal *
      Paper.laplaceKernel τ x y ∂p) = 0 := by
    rw [← integral_sub hposInt hnegInt, hcalc, hmoment x]
  exact sub_eq_zero.mp hdiff

private lemma meanShift_laplace_dirac_real
    (τ c x : ℝ) :
    Paper.meanShift (Paper.laplaceKernel τ) (Measure.dirac c) x = c - x := by
  unfold Paper.meanShift Paper.kernelNormalizer
  rw [integral_dirac, integral_dirac]
  rw [smul_smul]
  have hk : Paper.laplaceKernel τ x c ≠ 0 := by
    unfold Paper.laplaceKernel
    exact Real.exp_ne_zero _
  rw [inv_mul_cancel₀ hk, one_smul]

private lemma laplace_signed_moment_zero_of_zeroDrift_dirac_real
    (τ : ℝ) (hτ : Paper.ValidBandwidth τ)
    (p : Measure ℝ) [IsProbabilityMeasure p] (c : ℝ)
    (hzero : Paper.ZeroDrift
      (Paper.meanShiftDrift (Paper.laplaceKernel τ)) p (Measure.dirac c)) :
    ∀ x : ℝ, ∫ y, Paper.laplaceKernel τ x y * (y - c) ∂p = 0 := by
  intro x
  have hτ0 : 0 < τ := hτ
  have hms : Paper.meanShift (Paper.laplaceKernel τ) p x = c - x := by
    have hz := hzero x
    unfold Paper.meanShiftDrift at hz
    rw [meanShift_laplace_dirac_real τ c x] at hz
    exact sub_eq_zero.mp hz
  have hZpos := laplaceKernelNormalizer_pos p τ hτ0 x
  have hDp :
      ∫ y, Paper.laplaceKernel τ x y • (y - x) ∂p =
        Paper.kernelNormalizer (Paper.laplaceKernel τ) p x • (c - x) := by
    have h := congrArg
      (fun v : ℝ => Paper.kernelNormalizer (Paper.laplaceKernel τ) p x • v) hms
    unfold Paper.meanShift at h
    rw [smul_smul, mul_inv_cancel₀ (ne_of_gt hZpos), one_smul] at h
    exact h
  have hweighted : Integrable
      (fun y : ℝ => Paper.laplaceKernel τ x y • (y - x)) p :=
    laplaceWeightedDisplacement_integrable τ hτ0 p x
  have hk : Integrable (fun y : ℝ => Paper.laplaceKernel τ x y) p :=
    laplaceKernel_integrable p τ hτ0 x
  have hconst : Integrable
      (fun y : ℝ => (x - c) * Paper.laplaceKernel τ x y) p :=
    hk.const_mul (x - c)
  calc
    ∫ y, Paper.laplaceKernel τ x y * (y - c) ∂p
        = ∫ y, Paper.laplaceKernel τ x y • (y - x) +
            (x - c) * Paper.laplaceKernel τ x y ∂p := by
            congr 1
            funext y
            simp [smul_eq_mul]
            ring
    _ = (∫ y, Paper.laplaceKernel τ x y • (y - x) ∂p) +
          ∫ y, (x - c) * Paper.laplaceKernel τ x y ∂p := by
            rw [integral_add hweighted hconst]
    _ = Paper.kernelNormalizer (Paper.laplaceKernel τ) p x • (c - x) +
          (x - c) * Paper.kernelNormalizer (Paper.laplaceKernel τ) p x := by
            rw [hDp]
            unfold Paper.kernelNormalizer
            rw [integral_const_mul]
    _ = 0 := by
            simp [smul_eq_mul]
            ring

/-- Zero Laplace drift against a Dirac makes the positive and negative parts of
the signed first-moment measure have identical Laplace smoothings; by the
Fourier injectivity theorem on `ℝ`, the two finite measures are equal. -/
theorem laplaceMomentParts_eq_of_zeroDrift_dirac_real
    (τ : ℝ) (hτ : Paper.ValidBandwidth τ)
    (p : Measure ℝ) [IsProbabilityMeasure p] (c : ℝ)
    (hp : Integrable (fun y : ℝ => ‖y - c‖) p)
    (hzero : Paper.ZeroDrift
      (Paper.meanShiftDrift (Paper.laplaceKernel τ)) p (Measure.dirac c)) :
    laplacePositiveMomentMeasure p c = laplaceNegativeMomentMeasure p c := by
  haveI : IsFiniteMeasure (laplacePositiveMomentMeasure p c) :=
    isFiniteMeasure_laplacePositiveMomentMeasure p c hp
  haveI : IsFiniteMeasure (laplaceNegativeMomentMeasure p c) :=
    isFiniteMeasure_laplaceNegativeMomentMeasure p c hp
  apply laplaceKernelNormalizer_injective τ hτ
  exact laplacePositiveNegative_normalizers_eq_of_signed_moment_zero τ hτ p c hp
    (laplace_signed_moment_zero_of_zeroDrift_dirac_real τ hτ p c hzero)

private lemma measure_eq_dirac_of_ae_eq_real
    (p : Measure ℝ) [IsProbabilityMeasure p] (c : ℝ)
    (h : ∀ᵐ y ∂p, y = c) :
    p = Measure.dirac c := by
  have hmem : ∀ᵐ y ∂p, y ∈ ({c} : Finset ℝ) := by
    filter_upwards [h] with y hy
    simp [hy]
  have hsum := (Measure.ae_mem_finset_iff (μ := p) (s := ({c} : Finset ℝ))).mp hmem
  have hmass : p {c} = 1 := by
    have huniv := congrArg (fun μ : Measure ℝ => μ Set.univ) hsum
    simpa using huniv.symm
  simpa [hmass] using hsum

/-- **Stage-2 Dirac rigidity on the real line.**  For the paper's Laplace
kernel, a probability law with finite first moment cannot have zero raw
mean-shift drift against a Dirac unless it is that Dirac.  This is a genuine
one-degenerate-side instance of the arbitrary-target Laplace converse. -/
theorem laplaceZeroDrift_dirac_identifies_real
    (τ : ℝ) (hτ : Paper.ValidBandwidth τ)
    (p : Measure ℝ) [IsProbabilityMeasure p] (c : ℝ)
    (hp : Integrable (fun y : ℝ => ‖y - c‖) p)
    (hzero : Paper.ZeroDrift
      (Paper.meanShiftDrift (Paper.laplaceKernel τ)) p (Measure.dirac c)) :
    p = Measure.dirac c := by
  have hparts := laplaceMomentParts_eq_of_zeroDrift_dirac_real τ hτ p c hp hzero
  have hsing :
      laplacePositiveMomentMeasure p c ⟂ₘ laplaceNegativeMomentMeasure p c := by
    unfold laplacePositiveMomentMeasure laplaceNegativeMomentMeasure
    simpa [sub_eq_add_neg, neg_sub] using
      (withDensity_ofReal_mutuallySingular (μ := p)
        (f := fun y : ℝ => y - c) (by fun_prop))
  have hposSelf : laplacePositiveMomentMeasure p c ⟂ₘ
      laplacePositiveMomentMeasure p c := by
    simpa [hparts] using hsing
  have hposZero : laplacePositiveMomentMeasure p c = 0 :=
    (Measure.MutuallySingular.self_iff (laplacePositiveMomentMeasure p c)).mp hposSelf
  have hnegZero : laplaceNegativeMomentMeasure p c = 0 := by
    rw [← hparts]
    exact hposZero
  have hae_le : ∀ᵐ y ∂p, y ≤ c := by
    have hden :
        (fun y : ℝ => ENNReal.ofReal (y - c)) =ᵐ[p] 0 := by
      rw [← withDensity_eq_zero_iff
        ((by fun_prop : Measurable fun y : ℝ => ENNReal.ofReal (y - c))).aemeasurable]
      simpa [laplacePositiveMomentMeasure] using hposZero
    filter_upwards [hden] with y hy
    have hyc : y - c ≤ 0 := ENNReal.ofReal_eq_zero.mp hy
    linarith
  have hae_ge : ∀ᵐ y ∂p, c ≤ y := by
    have hden :
        (fun y : ℝ => ENNReal.ofReal (c - y)) =ᵐ[p] 0 := by
      rw [← withDensity_eq_zero_iff
        ((by fun_prop : Measurable fun y : ℝ => ENNReal.ofReal (c - y))).aemeasurable]
      simpa [laplaceNegativeMomentMeasure] using hnegZero
    filter_upwards [hden] with y hy
    have hcy : c - y ≤ 0 := ENNReal.ofReal_eq_zero.mp hy
    linarith
  have hae_eq : ∀ᵐ y ∂p, y = c := by
    filter_upwards [hae_le, hae_ge] with y hy_le hy_ge
    linarith
  exact measure_eq_dirac_of_ae_eq_real p c hae_eq

end DriftingIdentifiability
