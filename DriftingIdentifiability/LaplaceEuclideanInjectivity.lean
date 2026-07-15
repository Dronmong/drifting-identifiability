import DriftingIdentifiability.LaplaceInjectivity

/-!
# Euclidean Laplace smoothing injectivity: Fourier cancellation shell

`LaplaceInjectivity.lean` proves the complete one-dimensional theorem by
computing the Fourier transform of the two-sided exponential.  The higher-
dimensional paper kernel on `EuclideanSpace ℝ ι` is the radial profile

`x ↦ exp (-‖x‖ / τ)`.

This file proves its integrability axiom-free by product domination in
coordinates.  The remaining genuinely analytic L4 task is the standard radial
Fourier fact that this profile has nowhere-zero Fourier transform for positive
bandwidth.  Once that fact is supplied, equality of all Euclidean Laplace
normalizers determines the finite measure.

This mirrors the Gaussian cancellation skeleton in
`GaussianConvolutionInjectivity.lean`, but keeps the Laplace radial transform as
an explicit hypothesis rather than hiding it behind a new axiom.
-/

open MeasureTheory ProbabilityTheory Set Filter Topology
open scoped FourierTransform RealInnerProductSpace

namespace DriftingIdentifiability

universe u

/-- The radial Euclidean Laplace profile used by the paper kernel. -/
noncomputable def laplaceEuclideanFourierBase
    {E : Type u} [NormedAddCommGroup E] (τ : ℝ) (x : E) : ℂ :=
  (Real.exp (-‖x‖ / τ) : ℂ)

section EuclideanLaplaceProfileIntegrability

variable {ι : Type*} [Fintype ι]

/-- One-dimensional exponential decay with absolute value is integrable as a
complex-valued function. -/
private lemma integrable_complex_exp_neg_abs_div {a : ℝ} (ha : 0 < a) :
    Integrable (fun x : ℝ => (Real.exp (-|x| / a) : ℂ)) := by
  have hb : (0 : ℝ) < 1 / a := by positivity
  have h1 : IntegrableOn (fun x : ℝ => Real.exp (-(1 / a) * |x|)) (Ioi 0) := by
    refine ((exp_neg_integrableOn_Ioi 0 hb).congr_fun ?_ measurableSet_Ioi)
    intro x hx
    simp only
    rw [abs_of_pos (Set.mem_Ioi.mp hx)]
  have h2 : IntegrableOn (fun x : ℝ => Real.exp (-(1 / a) * |x|)) (Iic 0) := by
    rw [← Measure.map_neg_eq_self (volume : Measure ℝ)]
    have m : MeasurableEmbedding fun x : ℝ => -x :=
      (Homeomorph.neg ℝ).measurableEmbedding
    rw [m.integrableOn_map_iff]
    have hfun : ((fun x : ℝ => Real.exp (-(1 / a) * |x|)) ∘ fun x : ℝ => -x) =
        fun x : ℝ => Real.exp (-(1 / a) * |x|) := by
      funext x
      simp [Function.comp_apply, abs_neg]
    rw [hfun]
    simp only [neg_preimage, neg_Iic, neg_zero]
    rw [integrableOn_Ici_iff_integrableOn_Ioi]
    exact h1
  have hreal : Integrable (fun x : ℝ => Real.exp (-(1 / a) * |x|)) := by
    rw [← integrableOn_univ, ← Iic_union_Ioi (a := (0 : ℝ))]
    exact h2.union h1
  have hcomplex :
      Integrable (fun x : ℝ => ((Real.exp (-(1 / a) * |x|) : ℝ) : ℂ)) :=
    hreal.ofReal
  simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hcomplex

/-- The radial Euclidean Laplace profile is integrable for every positive
bandwidth on a finite-dimensional coordinate Euclidean space.  The proof uses
the elementary domination

`‖x‖₂ ≥ (∑ᵢ |xᵢ|) / max 1 √n`

and then factors the dominating product of one-dimensional exponentials. -/
theorem laplaceEuclideanFourierBase_integrable
    {τ : ℝ} (hτ : 0 < τ) :
    Integrable
      (laplaceEuclideanFourierBase (E := EuclideanSpace ℝ ι) τ) := by
  classical
  let C : ℝ := max 1 (Real.sqrt (Fintype.card ι))
  have hCpos : 0 < C := lt_of_lt_of_le zero_lt_one (le_max_left _ _)
  have hCτ : 0 < C * τ := mul_pos hCpos hτ
  have hprod : Integrable
      (fun x : ι → ℝ => ∏ i, (Real.exp (-|x i| / (C * τ)) : ℂ))
      (Measure.pi fun _ : ι => (volume : Measure ℝ)) :=
    Integrable.fintype_prod
      (μ := fun _ : ι => (volume : Measure ℝ))
      (fun _ => integrable_complex_exp_neg_abs_div hCτ)
  have hprod_volume : Integrable
      (fun x : ι → ℝ => ∏ i, (Real.exp (-|x i| / (C * τ)) : ℂ)) :=
    by simpa [MeasureTheory.volume_pi] using hprod
  rw [← (PiLp.volume_preserving_toLp ι).integrable_comp_emb
    (MeasurableEquiv.toLp 2 _).measurableEmbedding]
  refine hprod_volume.mono ?_ ?_
  · apply Continuous.aestronglyMeasurable
    unfold laplaceEuclideanFourierBase
    fun_prop
  · filter_upwards with x
    have hboundReal :
        Real.exp (-‖WithLp.toLp 2 x‖ / τ) ≤
          ∏ i, Real.exp (-|x i| / (C * τ)) := by
      calc
        Real.exp (-‖WithLp.toLp 2 x‖ / τ)
          ≤ Real.exp (-((∑ i, |x i|) / C) / τ) := by
              apply Real.exp_monotone
              have hsum_le : ∑ i, |x i| ≤ C * ‖WithLp.toLp 2 x‖ := by
                have hA_nonneg : 0 ≤ ∑ i, |x i| :=
                  Finset.sum_nonneg fun i _ => abs_nonneg (x i)
                have hB_nonneg : 0 ≤ ∑ i, |x i| ^ 2 :=
                  Finset.sum_nonneg fun i _ => sq_nonneg (|x i|)
                have hcard_nonneg : 0 ≤ (Fintype.card ι : ℝ) := by positivity
                have hmul_nonneg :
                    0 ≤ (Fintype.card ι : ℝ) * ∑ i, |x i| ^ 2 :=
                  mul_nonneg hcard_nonneg hB_nonneg
                have hsq :
                    (∑ i, |x i|) ^ 2 ≤
                      (Fintype.card ι : ℝ) * ∑ i, |x i| ^ 2 := by
                  simpa using
                    (sq_sum_le_card_mul_sum_sq
                      (s := Finset.univ) (f := fun i : ι => |x i|))
                have hsqrt :
                    ∑ i, |x i| ≤
                      Real.sqrt ((Fintype.card ι : ℝ) * ∑ i, |x i| ^ 2) :=
                  (Real.le_sqrt hA_nonneg hmul_nonneg).2 hsq
                have hcs :
                    ∑ i, |x i| ≤
                      Real.sqrt (Fintype.card ι) *
                        Real.sqrt (∑ i, |x i| ^ 2) := by
                  calc
                    ∑ i, |x i| ≤
                        Real.sqrt ((Fintype.card ι : ℝ) * ∑ i, |x i| ^ 2) := hsqrt
                    _ = Real.sqrt (Fintype.card ι) *
                          Real.sqrt (∑ i, |x i| ^ 2) := by
                        rw [Real.sqrt_mul hcard_nonneg]
                calc
                  ∑ i, |x i|
                      ≤ Real.sqrt (Fintype.card ι) *
                          Real.sqrt (∑ i, |x i| ^ 2) := hcs
                  _ ≤ C * Real.sqrt (∑ i, |x i| ^ 2) := by
                        gcongr
                        exact le_max_right 1 (Real.sqrt (Fintype.card ι))
                  _ = C * ‖WithLp.toLp 2 x‖ := by
                        congr 1
                        simp [EuclideanSpace.norm_eq, Real.norm_eq_abs, sq_abs]
              have hdiv : (∑ i, |x i|) / C ≤ ‖WithLp.toLp 2 x‖ := by
                rw [div_le_iff₀ hCpos]
                simpa [mul_comm] using hsum_le
              have hdivτ : ((∑ i, |x i|) / C) / τ ≤ ‖WithLp.toLp 2 x‖ / τ := by
                gcongr
              have hneg := neg_le_neg hdivτ
              simpa [neg_div] using hneg
        _ = ∏ i, Real.exp (-|x i| / (C * τ)) := by
              rw [← Real.exp_sum]
              congr 1
              field_simp [hCpos.ne', hτ.ne']
              rw [Finset.mul_sum, ← Finset.sum_neg_distrib]
              apply Finset.sum_congr rfl
              intro i _hi
              field_simp [hCpos.ne', hτ.ne']
    have htarget :
        Real.exp (-‖WithLp.toLp 2 x‖ / τ) ≤
          ∏ i, Real.exp
            ((-((|x i| : ℝ) : ℂ) / ((C : ℂ) * (τ : ℂ))).re) := by
      refine hboundReal.trans_eq ?_
      apply Finset.prod_congr rfl
      intro i _hi
      have harg :
          (-((|x i| : ℝ) : ℂ) / ((C : ℂ) * (τ : ℂ))) =
            ((-|x i| / (C * τ) : ℝ) : ℂ) := by
        push_cast
        ring
      have hre :
          (-((|x i| : ℝ) : ℂ) / ((C : ℂ) * (τ : ℂ))).re =
            -|x i| / (C * τ) := by
        rw [harg]
        rfl
      rw [hre]
    simpa [Function.comp_apply, laplaceEuclideanFourierBase, Complex.norm_exp] using htarget

end EuclideanLaplaceProfileIntegrability

section EuclideanLaplaceKernelNormalizer

variable {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
  [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [BorelSpace E] [SecondCountableTopology E]

omit [MeasurableSpace E] [InnerProductSpace ℝ E] [CompleteSpace E]
  [FiniteDimensional ℝ E] [BorelSpace E] [SecondCountableTopology E] in
private lemma laplaceKernel_coe_eq_euclideanFourierBase_sub
    {τ : ℝ} (x y : E) :
    (Paper.laplaceKernel τ x y : ℂ) =
      laplaceEuclideanFourierBase (E := E) τ (x - y) := by
  unfold Paper.laplaceKernel laplaceEuclideanFourierBase
  congr 1
  ring_nf

omit [CompleteSpace E] [SecondCountableTopology E] in
private lemma laplaceKernel_coe_integrable_of_euclideanFourierBase
    {τ : ℝ}
    (hbase : Integrable (laplaceEuclideanFourierBase (E := E) τ))
    (y : E) :
    Integrable (fun x : E => (Paper.laplaceKernel τ x y : ℂ)) := by
  simpa only [laplaceKernel_coe_eq_euclideanFourierBase_sub] using
    hbase.comp_sub_right y

omit [CompleteSpace E] in
private lemma phase_mul_laplaceKernel_integrable_of_euclideanFourierBase
    {τ : ℝ}
    (hbase : Integrable (laplaceEuclideanFourierBase (E := E) τ))
    (t y : E) :
    Integrable (fun x : E =>
      (Real.fourierChar (-⟪x, t⟫) : ℂ) *
        (Paper.laplaceKernel τ x y : ℂ)) := by
  apply (laplaceKernel_coe_integrable_of_euclideanFourierBase hbase y).bdd_mul
    (c := 1)
  · fun_prop
  · filter_upwards with x
    simp

omit [CompleteSpace E] [SecondCountableTopology E] in
private lemma norm_phase_mul_laplaceKernel_integral_euclidean
    {τ : ℝ} (t y : E) :
    (∫ x : E,
        ‖(Real.fourierChar (-⟪x, t⟫) : ℂ) *
          (Paper.laplaceKernel τ x y : ℂ)‖) =
      ∫ x : E, ‖laplaceEuclideanFourierBase (E := E) τ x‖ := by
  simp only [norm_mul, Circle.norm_coe, one_mul,
    laplaceKernel_coe_eq_euclideanFourierBase_sub]
  exact integral_sub_right_eq_self
    (fun x : E => ‖laplaceEuclideanFourierBase (E := E) τ x‖) y

private lemma phase_mul_laplaceKernel_integrable_prod_of_euclideanFourierBase
    {τ : ℝ}
    (hbase : Integrable (laplaceEuclideanFourierBase (E := E) τ))
    (t : E) (p : Measure E) [IsFiniteMeasure p] :
    Integrable (fun z : E × E =>
      (Real.fourierChar (-⟪z.1, t⟫) : ℂ) *
        (Paper.laplaceKernel τ z.1 z.2 : ℂ))
      (volume.prod p) := by
  have hmeas : AEStronglyMeasurable (fun z : E × E =>
      (Real.fourierChar (-⟪z.1, t⟫) : ℂ) *
        (Paper.laplaceKernel τ z.1 z.2 : ℂ))
      (volume.prod p) := by
    apply Continuous.aestronglyMeasurable
    simp only [Real.fourierChar_apply]
    unfold Paper.laplaceKernel
    fun_prop
  refine (integrable_prod_iff' hmeas).2 ⟨?_, ?_⟩
  · exact ae_of_all _
      (phase_mul_laplaceKernel_integrable_of_euclideanFourierBase hbase t)
  · have hc : Integrable
        (fun _ : E =>
          ∫ x : E, ‖laplaceEuclideanFourierBase (E := E) τ x‖) p :=
      integrable_const _
    convert hc using 1
    funext y
    exact norm_phase_mul_laplaceKernel_integral_euclidean (τ := τ) t y

omit [CompleteSpace E] [SecondCountableTopology E] in
private lemma fourier_laplaceKernel_euclidean {τ : ℝ} (t y : E) :
    𝓕 (fun x : E => (Paper.laplaceKernel τ x y : ℂ)) t =
      (Real.fourierChar (-⟪y, t⟫) : ℂ) •
        𝓕 (laplaceEuclideanFourierBase (E := E) τ) t := by
  have h := congrFun
    (VectorFourier.fourierIntegral_comp_add_right Real.fourierChar volume
      (innerₗ E) (laplaceEuclideanFourierBase (E := E) τ) (-y)) t
  change 𝓕 (laplaceEuclideanFourierBase (E := E) τ ∘ fun x : E => x + -y) t =
      Real.fourierChar (((innerₗ E) (-y)) t) •
        𝓕 (laplaceEuclideanFourierBase (E := E) τ) t at h
  calc
    𝓕 (fun x : E => (Paper.laplaceKernel τ x y : ℂ)) t =
        𝓕 (laplaceEuclideanFourierBase (E := E) τ ∘ fun x : E => x + -y) t := by
          apply congrArg (fun f : E → ℂ => 𝓕 f t)
          funext x
          simpa only [Function.comp_apply, sub_eq_add_neg] using
            (laplaceKernel_coe_eq_euclideanFourierBase_sub (τ := τ) x y)
    _ = Real.fourierChar (((innerₗ E) (-y)) t) •
          𝓕 (laplaceEuclideanFourierBase (E := E) τ) t := h
    _ = (Real.fourierChar (-⟪y, t⟫) : ℂ) •
          𝓕 (laplaceEuclideanFourierBase (E := E) τ) t := by
          simp [Circle.smul_def]

private noncomputable def complexLaplaceEuclideanNormalizer
    (τ : ℝ) (p : Measure E) (x : E) : ℂ :=
  ∫ y, (Paper.laplaceKernel τ x y : ℂ) ∂p

omit [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [BorelSpace E] [SecondCountableTopology E] in
private lemma complexLaplaceEuclideanNormalizer_eq_coe
    (τ : ℝ) (p : Measure E) (x : E) :
    complexLaplaceEuclideanNormalizer τ p x =
      (Paper.kernelNormalizer (Paper.laplaceKernel τ) p x : ℂ) := by
  unfold complexLaplaceEuclideanNormalizer Paper.kernelNormalizer
  exact integral_complex_ofReal

private lemma fourier_complexLaplaceEuclideanNormalizer
    {τ : ℝ}
    (hbase : Integrable (laplaceEuclideanFourierBase (E := E) τ))
    (p : Measure E) [IsFiniteMeasure p] (t : E) :
    𝓕 (complexLaplaceEuclideanNormalizer τ p) t =
      charFun p ((-2 * Real.pi) • t) *
        𝓕 (laplaceEuclideanFourierBase (E := E) τ) t := by
  have hprod :=
    phase_mul_laplaceKernel_integrable_prod_of_euclideanFourierBase hbase t p
  rw [Real.fourier_eq]
  simp only [Circle.smul_def]
  unfold complexLaplaceEuclideanNormalizer
  calc
    (∫ x : E,
        (Real.fourierChar (-⟪x, t⟫) : ℂ) *
          ∫ y : E, (Paper.laplaceKernel τ x y : ℂ) ∂p) =
        ∫ x : E, ∫ y : E,
          (Real.fourierChar (-⟪x, t⟫) : ℂ) *
            (Paper.laplaceKernel τ x y : ℂ) ∂p := by
              congr 1
              funext x
              rw [integral_const_mul]
    _ = ∫ y : E, (∫ x : E,
          (Real.fourierChar (-⟪x, t⟫) : ℂ) *
            (Paper.laplaceKernel τ x y : ℂ)) ∂p := by
              exact integral_integral_swap
                (f := fun x y =>
                  (Real.fourierChar (-⟪x, t⟫) : ℂ) *
                    (Paper.laplaceKernel τ x y : ℂ))
                hprod
    _ = ∫ y : E,
          (Real.fourierChar (-⟪y, t⟫) : ℂ) •
            𝓕 (laplaceEuclideanFourierBase (E := E) τ) t ∂p := by
              congr 1
              funext y
              rw [← fourier_laplaceKernel_euclidean (τ := τ) t y, Real.fourier_eq]
              rfl
    _ = (∫ y : E, (Real.fourierChar (-⟪y, t⟫) : ℂ) ∂p) *
          𝓕 (laplaceEuclideanFourierBase (E := E) τ) t := by
              simp only [smul_eq_mul]
              rw [integral_mul_const]
    _ = charFun p ((-2 * Real.pi) • t) *
          𝓕 (laplaceEuclideanFourierBase (E := E) τ) t := by
              rw [charFun_apply]
              congr 2
              funext y
              simp only [Real.fourierChar_apply, real_inner_smul_right, neg_mul]
              congr 1
              push_cast
              ring

/-- **Conditional Euclidean Laplace smoothing injectivity.**

This is the L4 Fourier-cancellation theorem.  The only supplied hypotheses are
the two analytic radial-profile facts still isolated by the roadmap:

* `laplaceEuclideanFourierBase τ` is integrable;
* its Fourier transform is nowhere zero.

Under those hypotheses, equality of all unnormalized radial Laplace normalizers
determines an arbitrary finite measure on a finite-dimensional real inner-
product space. -/
theorem laplaceKernelNormalizer_injective_euclidean_of_fourier_ne_zero
    (τ : ℝ)
    (hbase : Integrable (laplaceEuclideanFourierBase (E := E) τ))
    (hbase_ne : ∀ t : E, 𝓕 (laplaceEuclideanFourierBase (E := E) τ) t ≠ 0)
    (p q : Measure E) [IsFiniteMeasure p] [IsFiniteMeasure q]
    (h : ∀ x : E,
      Paper.kernelNormalizer (Paper.laplaceKernel τ) p x =
        Paper.kernelNormalizer (Paper.laplaceKernel τ) q x) :
    p = q := by
  apply Measure.ext_of_charFun
  funext s
  obtain ⟨t, rfl⟩ : ∃ t : E, (-2 * Real.pi) • t = s := by
    refine ⟨((-2 * Real.pi)⁻¹) • s, ?_⟩
    rw [smul_smul]
    convert one_smul ℝ s using 1
    field_simp [Real.pi_ne_zero]
  have hcomplex :
      complexLaplaceEuclideanNormalizer τ p =
        complexLaplaceEuclideanNormalizer τ q := by
    funext x
    rw [complexLaplaceEuclideanNormalizer_eq_coe,
      complexLaplaceEuclideanNormalizer_eq_coe, h x]
  have hfourier :=
    congrFun (congrArg (fun f : E → ℂ => 𝓕 f) hcomplex) t
  rw [fourier_complexLaplaceEuclideanNormalizer hbase,
    fourier_complexLaplaceEuclideanNormalizer hbase] at hfourier
  exact mul_right_cancel₀ (hbase_ne t) hfourier

/-- Predicate-level form of the conditional Euclidean Laplace smoothing
injectivity theorem.  This is the plug-in L4 interface for higher-dimensional
arguments: prove the radial Fourier profile facts once, and the generic
`LaplaceSmoothingInjective` predicate follows. -/
theorem laplaceSmoothingInjective_euclidean_of_fourier_ne_zero
    (τ : ℝ)
    (hbase : Integrable (laplaceEuclideanFourierBase (E := E) τ))
    (hbase_ne : ∀ t : E, 𝓕 (laplaceEuclideanFourierBase (E := E) τ) t ≠ 0) :
    LaplaceSmoothingInjective E τ := by
  intro p q hp hq h
  letI := hp
  letI := hq
  exact laplaceKernelNormalizer_injective_euclidean_of_fourier_ne_zero
    (E := E) τ hbase hbase_ne p q h

end EuclideanLaplaceKernelNormalizer

section EuclideanSpaceLaplaceKernelNormalizer

variable {ι : Type*} [Fintype ι]

/-- Euclidean-space L4 normalizer injectivity with integrability discharged.
The only remaining analytic hypothesis is the nowhere-vanishing Fourier
transform of the radial Laplace profile. -/
theorem laplaceKernelNormalizer_injective_euclideanSpace_of_fourier_ne_zero
    (τ : ℝ) (hτ : Paper.ValidBandwidth τ)
    (hbase_ne : ∀ t : EuclideanSpace ℝ ι,
      𝓕 (laplaceEuclideanFourierBase (E := EuclideanSpace ℝ ι) τ) t ≠ 0)
    (p q : Measure (EuclideanSpace ℝ ι))
    [IsFiniteMeasure p] [IsFiniteMeasure q]
    (h : ∀ x : EuclideanSpace ℝ ι,
      Paper.kernelNormalizer (Paper.laplaceKernel τ) p x =
        Paper.kernelNormalizer (Paper.laplaceKernel τ) q x) :
    p = q :=
  laplaceKernelNormalizer_injective_euclidean_of_fourier_ne_zero
    (E := EuclideanSpace ℝ ι) τ
    (laplaceEuclideanFourierBase_integrable (ι := ι) hτ)
    hbase_ne p q h

/-- Predicate-level Euclidean-space L4 interface with integrability
discharged.  Proving the remaining radial Fourier nonvanishing theorem will
instantiate `LaplaceSmoothingInjective` for the paper's Euclidean kernel. -/
theorem laplaceSmoothingInjective_euclideanSpace_of_fourier_ne_zero
    (τ : ℝ) (hτ : Paper.ValidBandwidth τ)
    (hbase_ne : ∀ t : EuclideanSpace ℝ ι,
      𝓕 (laplaceEuclideanFourierBase (E := EuclideanSpace ℝ ι) τ) t ≠ 0) :
    LaplaceSmoothingInjective (EuclideanSpace ℝ ι) τ := by
  intro p q hp hq h
  letI := hp
  letI := hq
  exact laplaceKernelNormalizer_injective_euclideanSpace_of_fourier_ne_zero
    (ι := ι) τ hτ hbase_ne p q h

end EuclideanSpaceLaplaceKernelNormalizer

end DriftingIdentifiability
