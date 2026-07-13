import DriftingIdentifiability.LaplaceACFinal
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics

open MeasureTheory Set Filter Topology ProbabilityTheory
open scoped intervalIntegral ENNReal NNReal
namespace DriftingIdentifiability
open Paper

private lemma integral_eq_zero_of_map_neg_eq_self
    {p : Measure ℝ} {f : ℝ → ℝ}
    (hmap : p.map (fun x : ℝ => -x) = p)
    (hf : Integrable f p)
    (hodd : ∀ x : ℝ, f (-x) = - f x) :
    (∫ x, f x ∂p) = 0 := by
  have hsm : AEStronglyMeasurable f (p.map fun x : ℝ => -x) := by
    simpa [hmap] using hf.aestronglyMeasurable
  have hmapInt :
      (∫ x, f x ∂p.map (fun x : ℝ => -x)) = ∫ x, f (-x) ∂p := by
    exact integral_map (by fun_prop) hsm
  have hself : (∫ x, f x ∂p) = ∫ x, f (-x) ∂p := by
    calc
      (∫ x, f x ∂p) = (∫ x, f x ∂p.map (fun x : ℝ => -x)) := by rw [hmap]
      _ = ∫ x, f (-x) ∂p := hmapInt
  have hneg : (∫ x, f (-x) ∂p) = - ∫ x, f x ∂p := by
    simp_rw [hodd]
    rw [integral_neg]
  have h : (∫ x, f x ∂p) = - ∫ x, f x ∂p := hself.trans hneg
  linarith

private lemma stdGaussian_map_neg :
    (gaussianReal 0 (1 : NNReal)).map (fun x : ℝ => -x) =
      gaussianReal 0 (1 : NNReal) := by
  simpa using (gaussianReal_map_neg (μ := (0 : ℝ)) (v := (1 : NNReal)))

private lemma laplaceWeightedDisplacement_zero_odd (τ : ℝ) :
    ∀ y : ℝ, laplaceWeightedDisplacement τ 0 (-y) =
      - laplaceWeightedDisplacement τ 0 y := by
  intro y
  unfold laplaceWeightedDisplacement laplaceKernel
  rw [smul_eq_mul, smul_eq_mul]
  have hnorm : ‖(0 : ℝ) - -y‖ = ‖(0 : ℝ) - y‖ := by simp
  rw [hnorm]
  ring

private theorem standardGaussian_laplaceDisplacementIntegral_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    (∫ y, laplaceWeightedDisplacement τ 0 y ∂gaussianReal 0 (1 : NNReal)) = 0 := by
  exact integral_eq_zero_of_map_neg_eq_self
    stdGaussian_map_neg
    (laplaceWeightedDisplacement_integrable τ hτ (gaussianReal 0 (1 : NNReal)) 0)
    (laplaceWeightedDisplacement_zero_odd τ)

private theorem standardGaussian_laplaceMeanShiftRatio_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    laplaceMeanShiftRatio τ (gaussianReal 0 (1 : NNReal)) 0 = 0 := by
  unfold laplaceMeanShiftRatio
  rw [standardGaussian_laplaceDisplacementIntegral_zero τ hτ]
  simp

private lemma laplaceWeightedDisplacement_shift (τ x s : ℝ) :
    laplaceWeightedDisplacement τ x (x + s) =
      Real.exp (-(1 / τ) * |s|) * s := by
  unfold laplaceWeightedDisplacement laplaceKernel
  rw [smul_eq_mul, Real.norm_eq_abs]
  have hsub : x - (x + s) = -s := by ring
  rw [hsub, abs_neg]
  ring

private lemma standardGaussian_displacementIntegral_shift
    (τ : ℝ) (_hτ : ValidBandwidth τ) (x : ℝ) :
    (∫ y, laplaceWeightedDisplacement τ x y ∂gaussianReal 0 (1 : NNReal)) =
      ∫ s, gaussianPDFReal 0 (1 : NNReal) (x + s) *
        (Real.exp (-(1 / τ) * |s|) * s) := by
  rw [integral_gaussianReal_eq_integral_smul (by norm_num : (1 : NNReal) ≠ 0)]
  rw [← integral_add_left_eq_self
    (fun y : ℝ => gaussianPDFReal 0 (1 : NNReal) y •
      laplaceWeightedDisplacement τ x y) x]
  apply integral_congr_ae
  filter_upwards with s
  rw [laplaceWeightedDisplacement_shift]
  simp [smul_eq_mul]

private lemma standardGaussian_shiftedDisplacementDensity_integrable
    (τ : ℝ) (hτ : ValidBandwidth τ) (x : ℝ) :
    Integrable
      (fun s : ℝ => gaussianPDFReal 0 (1 : NNReal) (x + s) *
        (Real.exp (-(1 / τ) * |s|) * s)) volume := by
  let G : ℝ → ℝ := fun y =>
    gaussianPDFReal 0 (1 : NNReal) y • laplaceWeightedDisplacement τ x y
  have hG : Integrable G volume := by
    have hwd : Integrable (fun y : ℝ => laplaceWeightedDisplacement τ x y)
        (gaussianReal 0 (1 : NNReal)) :=
      laplaceWeightedDisplacement_integrable τ hτ
        (gaussianReal 0 (1 : NNReal)) x
    rw [gaussianReal_of_var_ne_zero (μ := (0 : ℝ))
      (by norm_num : (1 : NNReal) ≠ 0)] at hwd
    have hiff := (integrable_withDensity_iff_integrable_smul'
      (f := gaussianPDF 0 (1 : NNReal))
      (μ := (volume : Measure ℝ))
      (E := ℝ)
      (g := fun y : ℝ => laplaceWeightedDisplacement τ x y)
      (measurable_gaussianPDF 0 (1 : NNReal))
      (ae_of_all _ fun y =>
        (gaussianPDF_lt_top (μ := (0 : ℝ)) (v := (1 : NNReal)) (x := y))))
    have h := hiff.mp hwd
    simpa [G, gaussianPDF, smul_eq_mul, ENNReal.toReal_ofReal,
      gaussianPDFReal_nonneg] using h
  have hshift : Integrable (fun s : ℝ => G (x + s)) volume :=
    hG.comp_add_left x
  refine hshift.congr ?_
  filter_upwards with s
  dsimp [G]
  rw [laplaceWeightedDisplacement_shift]

private lemma integral_eq_integral_Ioi_add_reflect
    {F : ℝ → ℝ} (hF : Integrable F volume) :
    (∫ s, F s) = ∫ s in Set.Ioi (0 : ℝ), F s + F (-s) := by
  have hsplit := integral_add_compl (s := Set.Ioi (0 : ℝ)) measurableSet_Ioi hF
  have hneg := integral_comp_neg_Ioi (0 : ℝ) F
  rw [neg_zero] at hneg
  have hFnegOn : IntegrableOn (fun s : ℝ => F (-s)) (Set.Ioi (0 : ℝ)) := by
    rw [← Measure.map_neg_eq_self (volume : Measure ℝ)]
    have m : MeasurableEmbedding (fun x : ℝ => -x) :=
      (Homeomorph.neg ℝ).measurableEmbedding
    rw [m.integrableOn_map_iff]
    simpa [Function.comp_def, Set.preimage, Set.Ioi, Set.Iio]
      using (hF.integrableOn : IntegrableOn F (Set.Iio (0 : ℝ)) volume)
  have hFOn : IntegrableOn F (Set.Ioi (0 : ℝ)) := hF.integrableOn
  calc
    (∫ s, F s) =
        (∫ s in Set.Ioi (0 : ℝ), F s) + (∫ s in (Set.Ioi (0 : ℝ))ᶜ, F s) :=
      hsplit.symm
    _ = (∫ s in Set.Ioi (0 : ℝ), F s) + (∫ s in Set.Iic (0 : ℝ), F s) := by simp
    _ = (∫ s in Set.Ioi (0 : ℝ), F s) + (∫ s in Set.Ioi (0 : ℝ), F (-s)) := by rw [hneg]
    _ = ∫ s in Set.Ioi (0 : ℝ), F s + F (-s) := by
      exact (integral_add hFOn hFnegOn).symm

private lemma reflected_pair_integrableOn_Ioi
    {F : ℝ → ℝ} (hF : Integrable F volume) :
    IntegrableOn (fun s : ℝ => F s + F (-s)) (Set.Ioi (0 : ℝ)) volume := by
  have hFnegOn : IntegrableOn (fun s : ℝ => F (-s)) (Set.Ioi (0 : ℝ)) := by
    rw [← Measure.map_neg_eq_self (volume : Measure ℝ)]
    have m : MeasurableEmbedding (fun x : ℝ => -x) :=
      (Homeomorph.neg ℝ).measurableEmbedding
    rw [m.integrableOn_map_iff]
    simpa [Function.comp_def, Set.preimage, Set.Ioi, Set.Iio]
      using (hF.integrableOn : IntegrableOn F (Set.Iio (0 : ℝ)) volume)
  exact hF.integrableOn.add hFnegOn

private lemma setIntegral_Ioi_reflected_pair_neg
    {F : ℝ → ℝ} (hF : Integrable F volume)
    (hneg : ∀ s : ℝ, 0 < s → F s + F (-s) < 0) :
    (∫ s in Set.Ioi (0 : ℝ), F s + F (-s)) < 0 := by
  let G : ℝ → ℝ := fun s => -(F s + F (-s))
  have hG_nonneg : 0 ≤ᵐ[volume.restrict (Set.Ioi (0 : ℝ))] G := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with s hs
    exact le_of_lt (neg_pos.mpr (hneg s hs))
  have hG_int : Integrable G (volume.restrict (Set.Ioi (0 : ℝ))) := by
    change IntegrableOn (fun s : ℝ => -(F s + F (-s))) (Set.Ioi (0 : ℝ)) volume
    exact (reflected_pair_integrableOn_Ioi hF).neg
  have hG_pos : 0 < ∫ s in Set.Ioi (0 : ℝ), G s := by
    rw [setIntegral_pos_iff_support_of_nonneg_ae hG_nonneg hG_int]
    have hsupport : Function.support G ∩ Set.Ioi (0 : ℝ) = Set.Ioi (0 : ℝ) := by
      ext s
      constructor
      · intro hs
        exact hs.2
      · intro hs
        refine ⟨?_, hs⟩
        change G s ≠ 0
        exact ne_of_gt (neg_pos.mpr (hneg s hs))
    rw [hsupport]
    simp
  have hnegint :
      (∫ s in Set.Ioi (0 : ℝ), G s) =
        - ∫ s in Set.Ioi (0 : ℝ), F s + F (-s) := by
    change (∫ s in Set.Ioi (0 : ℝ), -(F s + F (-s))) =
      - ∫ s in Set.Ioi (0 : ℝ), F s + F (-s)
    rw [integral_neg]
  linarith

private lemma gaussianPDFReal_standard_lt_of_abs_lt {a b : ℝ}
    (h : |a| < |b|) :
    gaussianPDFReal 0 (1 : NNReal) b < gaussianPDFReal 0 (1 : NNReal) a := by
  have hsq : a ^ 2 < b ^ 2 := by
    rwa [sq_lt_sq]
  unfold gaussianPDFReal
  have hdenpos : 0 < √(2 * Real.pi * (1 : NNReal)) := by positivity
  apply mul_lt_mul_of_pos_left _ (inv_pos.mpr hdenpos)
  apply Real.exp_lt_exp.mpr
  norm_num
  nlinarith

private lemma abs_sub_lt_add_of_pos {x s : ℝ} (hx : 0 < x) (hs : 0 < s) :
    |x - s| < x + s := by
  rw [abs_lt]
  constructor <;> linarith

private lemma standardGaussian_shifted_pair_neg
    (τ : ℝ) (_hτ : ValidBandwidth τ) {x s : ℝ} (hx : 0 < x) (hs : 0 < s) :
    gaussianPDFReal 0 (1 : NNReal) (x + s) *
        (Real.exp (-(1 / τ) * |s|) * s) +
      gaussianPDFReal 0 (1 : NNReal) (x + -s) *
        (Real.exp (-(1 / τ) * |-s|) * (-s)) < 0 := by
  have hpdf :
      gaussianPDFReal 0 (1 : NNReal) (x + s) <
        gaussianPDFReal 0 (1 : NNReal) (x - s) := by
    have h_abs : |x - s| < |x + s| := by
      rw [abs_of_pos (by linarith : 0 < x + s)]
      exact abs_sub_lt_add_of_pos hx hs
    exact gaussianPDFReal_standard_lt_of_abs_lt h_abs
  have hscale : 0 < Real.exp (-(1 / τ) * |s|) * s :=
    mul_pos (Real.exp_pos _) hs
  rw [show x + -s = x - s by ring, abs_neg]
  nlinarith

private theorem standardGaussian_laplaceDisplacementIntegral_neg_of_pos
    (τ : ℝ) (hτ : ValidBandwidth τ) {x : ℝ} (hx : 0 < x) :
    (∫ y, laplaceWeightedDisplacement τ x y ∂gaussianReal 0 (1 : NNReal)) < 0 := by
  let F : ℝ → ℝ := fun s =>
    gaussianPDFReal 0 (1 : NNReal) (x + s) *
      (Real.exp (-(1 / τ) * |s|) * s)
  have hF : Integrable F volume :=
    standardGaussian_shiftedDisplacementDensity_integrable τ hτ x
  rw [standardGaussian_displacementIntegral_shift τ hτ x]
  rw [integral_eq_integral_Ioi_add_reflect hF]
  exact setIntegral_Ioi_reflected_pair_neg hF
    (fun s hs => standardGaussian_shifted_pair_neg τ hτ hx hs)

private lemma laplaceWeightedDisplacement_neg_neg (τ x y : ℝ) :
    laplaceWeightedDisplacement τ (-x) (-y) =
      - laplaceWeightedDisplacement τ x y := by
  unfold laplaceWeightedDisplacement laplaceKernel
  rw [smul_eq_mul, smul_eq_mul]
  have hnorm : ‖(-x : ℝ) - -y‖ = ‖x - y‖ := by
    rw [Real.norm_eq_abs, Real.norm_eq_abs]
    have h : (-x : ℝ) - -y = -(x - y) := by ring
    rw [h, abs_neg]
  rw [hnorm]
  ring

private theorem standardGaussian_laplaceDisplacementIntegral_odd
    (τ : ℝ) (hτ : ValidBandwidth τ) (x : ℝ) :
    (∫ y, laplaceWeightedDisplacement τ (-x) y ∂gaussianReal 0 (1 : NNReal)) =
      - (∫ y, laplaceWeightedDisplacement τ x y ∂gaussianReal 0 (1 : NNReal)) := by
  let f : ℝ → ℝ := fun y => laplaceWeightedDisplacement τ (-x) y
  have hsm : AEStronglyMeasurable f
      ((gaussianReal 0 (1 : NNReal)).map fun y : ℝ => -y) := by
    have hf : Integrable f (gaussianReal 0 (1 : NNReal)) :=
      laplaceWeightedDisplacement_integrable τ hτ (gaussianReal 0 (1 : NNReal)) (-x)
    simpa [stdGaussian_map_neg] using hf.aestronglyMeasurable
  have hmapInt :
      (∫ y, f y ∂(gaussianReal 0 (1 : NNReal)).map (fun y : ℝ => -y)) =
        ∫ y, f (-y) ∂gaussianReal 0 (1 : NNReal) := by
    exact integral_map (by fun_prop) hsm
  calc
    (∫ y, laplaceWeightedDisplacement τ (-x) y ∂gaussianReal 0 (1 : NNReal))
        = ∫ y, f y ∂(gaussianReal 0 (1 : NNReal)).map (fun y : ℝ => -y) := by
          rw [stdGaussian_map_neg]
    _ = ∫ y, f (-y) ∂gaussianReal 0 (1 : NNReal) := hmapInt
    _ = ∫ y, - laplaceWeightedDisplacement τ x y ∂gaussianReal 0 (1 : NNReal) := by
          apply integral_congr_ae
          filter_upwards with y
          dsimp [f]
          exact laplaceWeightedDisplacement_neg_neg τ x y
    _ = - (∫ y, laplaceWeightedDisplacement τ x y ∂gaussianReal 0 (1 : NNReal)) := by
          rw [integral_neg]

private theorem standardGaussian_laplaceDisplacementIntegral_pos_of_neg
    (τ : ℝ) (hτ : ValidBandwidth τ) {x : ℝ} (hx : x < 0) :
    0 < (∫ y, laplaceWeightedDisplacement τ x y ∂gaussianReal 0 (1 : NNReal)) := by
  have hpos : 0 < -x := neg_pos.mpr hx
  have hneg := standardGaussian_laplaceDisplacementIntegral_neg_of_pos τ hτ hpos
  have hodd := standardGaussian_laplaceDisplacementIntegral_odd τ hτ (-x)
  have hodd' :
      (∫ y, laplaceWeightedDisplacement τ x y ∂gaussianReal 0 (1 : NNReal)) =
        - (∫ y, laplaceWeightedDisplacement τ (-x) y ∂gaussianReal 0 (1 : NNReal)) := by
    simpa using hodd
  rw [hodd']
  exact neg_pos.mpr hneg

private lemma hasDerivAt_gaussianPDFReal_standard (s : ℝ) :
    HasDerivAt (gaussianPDFReal 0 (1 : NNReal))
      (-s * gaussianPDFReal 0 (1 : NNReal) s) s := by
  unfold gaussianPDFReal
  have hfun :
      (fun t : ℝ =>
        (√(2 * Real.pi * (1 : NNReal)))⁻¹ *
          Real.exp (-(t - 0) ^ 2 / (2 * (1 : NNReal)))) =
      fun t : ℝ =>
        (√(2 * Real.pi * (1 : NNReal)))⁻¹ *
          Real.exp (-((t - 0) * (t - 0)) / (2 * (1 : NNReal))) := by
    funext t
    congr 2
    ring
  rw [hfun]
  have hvalue :
      -s * ((√(2 * Real.pi * (1 : NNReal)))⁻¹ *
        Real.exp (-(s - 0) ^ 2 / (2 * (1 : NNReal)))) =
      -s * ((√(2 * Real.pi * (1 : NNReal)))⁻¹ *
        Real.exp (-((s - 0) * (s - 0)) / (2 * (1 : NNReal)))) := by
    congr 2
    congr 1
    ring
  rw [hvalue]
  change HasDerivAt
    ((fun _ : ℝ => (√(2 * Real.pi * (1 : NNReal)))⁻¹) *
      fun t : ℝ => Real.exp (-((t - 0) * (t - 0)) / (2 * (1 : NNReal))))
    (-s * ((√(2 * Real.pi * (1 : NNReal)))⁻¹ *
      Real.exp (-((s - 0) * (s - 0)) / (2 * (1 : NNReal))))) s
  have hc : HasDerivAt
      (fun _ : ℝ => (√(2 * Real.pi * (1 : NNReal)))⁻¹)
      0 s := hasDerivAt_const s _
  have hsq : HasDerivAt (fun t : ℝ => (t - 0) * (t - 0))
      (2 * (s - 0)) s := by
    have hsub : HasDerivAt (fun t : ℝ => t - 0) 1 s :=
      (hasDerivAt_id s).sub_const 0
    have hmul := hsub.mul hsub
    change HasDerivAt ((fun t : ℝ => t - 0) * fun t : ℝ => t - 0)
      (2 * (s - 0)) s
    have hval : 1 * (s - 0) + (s - 0) * 1 = 2 * (s - 0) := by
      ring
    rw [← hval]
    exact hmul
  have hquad : HasDerivAt (fun t : ℝ => -((t - 0) * (t - 0)) / (2 * (1 : NNReal)))
      (-s) s := by
    have hneg : HasDerivAt (fun t : ℝ => -((t - 0) * (t - 0)))
        (-(2 * (s - 0))) s := hsq.neg
    have hdiv := hneg.div_const (2 * (1 : NNReal))
    have hval : -(2 * (s - 0)) / (2 * (1 : NNReal)) = -s := by
      norm_num
      ring
    rw [← hval]
    exact hdiv
  have hexp := hquad.exp
  have hmul := hc.mul hexp
  have hval :
      0 * Real.exp (-((s - 0) * (s - 0)) / (2 * (1 : NNReal))) +
          (√(2 * Real.pi * (1 : NNReal)))⁻¹ *
            (Real.exp (-((s - 0) * (s - 0)) / (2 * (1 : NNReal))) * -s) =
        -s * ((√(2 * Real.pi * (1 : NNReal)))⁻¹ *
          Real.exp (-((s - 0) * (s - 0)) / (2 * (1 : NNReal)))) := by
    ring
  rw [← hval]
  exact hmul

private lemma hasDerivAt_standardGaussian_halfPrimitive (τ : ℝ) (s : ℝ) :
    HasDerivAt
      (fun t : ℝ =>
        t * Real.exp (-t / τ) * gaussianPDFReal 0 (1 : NNReal) t)
      ((1 - s / τ - s ^ 2) * Real.exp (-s / τ) *
        gaussianPDFReal 0 (1 : NNReal) s) s := by
  have hid : HasDerivAt (fun t : ℝ => t) 1 s := hasDerivAt_id s
  have hlin : HasDerivAt (fun t : ℝ => -t / τ) (-(1 / τ)) s := by
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
      ((hasDerivAt_id s).const_mul (-(1 / τ)))
  have hexp : HasDerivAt (fun t : ℝ => Real.exp (-t / τ))
      (-(1 / τ) * Real.exp (-s / τ)) s := by
    simpa [mul_comm, mul_left_comm, mul_assoc] using hlin.exp
  have hpdf := hasDerivAt_gaussianPDFReal_standard s
  have hmul := (hid.mul hexp).mul hpdf
  have hval :
      (1 * Real.exp (-s / τ) + s * (-(1 / τ) * Real.exp (-s / τ))) *
          gaussianPDFReal 0 (1 : NNReal) s +
        (s * Real.exp (-s / τ)) *
          (-s * gaussianPDFReal 0 (1 : NNReal) s) =
      (1 - s / τ - s ^ 2) * Real.exp (-s / τ) *
        gaussianPDFReal 0 (1 : NNReal) s := by
    ring
  change HasDerivAt
    (((fun t : ℝ => t) * fun t : ℝ => Real.exp (-t / τ)) *
      gaussianPDFReal 0 (1 : NNReal))
    ((1 - s / τ - s ^ 2) * Real.exp (-s / τ) *
      gaussianPDFReal 0 (1 : NNReal) s) s
  rw [← hval]
  exact hmul

private lemma standardGaussian_pdf_pow_integrable (n : ℕ) :
    Integrable (fun x : ℝ => x ^ n * gaussianPDFReal 0 (1 : NNReal) x) volume := by
  have hg : Integrable (fun x : ℝ => x ^ n) (gaussianReal 0 (1 : NNReal)) := by
    have h := integrable_pow_mul_exp_of_mem_interior_integrableExpSet
      (X := id) (μ := gaussianReal 0 (1 : NNReal))
      (by simp : (0 : ℝ) ∈ interior (integrableExpSet id (gaussianReal 0 (1 : NNReal))))
      n
    simpa using h
  rw [gaussianReal_of_var_ne_zero (μ := (0 : ℝ))
    (by norm_num : (1 : NNReal) ≠ 0)] at hg
  have hiff := (integrable_withDensity_iff_integrable_smul'
    (f := gaussianPDF 0 (1 : NNReal))
    (μ := (volume : Measure ℝ))
    (E := ℝ)
    (g := fun x : ℝ => x ^ n)
    (measurable_gaussianPDF 0 (1 : NNReal))
    (ae_of_all _ fun x =>
      (gaussianPDF_lt_top (μ := (0 : ℝ)) (v := (1 : NNReal)) (x := x))))
  have hvol := hiff.mp hg
  simpa [gaussianPDF, smul_eq_mul, ENNReal.toReal_ofReal,
    gaussianPDFReal_nonneg, mul_comm, mul_left_comm, mul_assoc] using hvol

private lemma standardGaussian_exp_neg_pow_integrableOn_Ioi
    (τ : ℝ) (hτ : ValidBandwidth τ) (n : ℕ) :
    IntegrableOn
      (fun s : ℝ => s ^ n * Real.exp (-s / τ) *
        gaussianPDFReal 0 (1 : NNReal) s)
      (Set.Ioi (0 : ℝ)) volume := by
  have hbase : Integrable (fun s : ℝ => s ^ n * gaussianPDFReal 0 (1 : NNReal) s)
      (volume.restrict (Set.Ioi (0 : ℝ))) :=
    (standardGaussian_pdf_pow_integrable n).integrableOn
  refine hbase.mono' ?_ ?_
  · measurability
  · filter_upwards [ae_restrict_mem measurableSet_Ioi] with s hs
    have hexp_le : Real.exp (-s / τ) ≤ 1 := by
      have hnonneg : 0 ≤ s / τ := div_nonneg hs.le hτ.le
      have hle : -s / τ ≤ 0 := by
        rw [neg_div]
        exact neg_nonpos.mpr hnonneg
      exact Real.exp_le_one_iff.mpr hle
    have hexp_nonneg : 0 ≤ Real.exp (-s / τ) := (Real.exp_pos _).le
    have hbase_nonneg :
        0 ≤ s ^ n * gaussianPDFReal 0 (1 : NNReal) s := by
      exact mul_nonneg (pow_nonneg hs.le n)
        (gaussianPDFReal_nonneg 0 (1 : NNReal) s)
    calc
      ‖s ^ n * Real.exp (-s / τ) * gaussianPDFReal 0 (1 : NNReal) s‖
          = Real.exp (-s / τ) *
              (s ^ n * gaussianPDFReal 0 (1 : NNReal) s) := by
            rw [show s ^ n * Real.exp (-s / τ) *
                gaussianPDFReal 0 (1 : NNReal) s =
                Real.exp (-s / τ) *
                  (s ^ n * gaussianPDFReal 0 (1 : NNReal) s) by ring,
              Real.norm_eq_abs, abs_of_nonneg (mul_nonneg hexp_nonneg hbase_nonneg)]
      _ ≤ 1 * (s ^ n * gaussianPDFReal 0 (1 : NNReal) s) := by
            exact mul_le_mul_of_nonneg_right hexp_le hbase_nonneg
      _ = s ^ n * gaussianPDFReal 0 (1 : NNReal) s := by ring

private lemma standardGaussian_halfPrimitive_deriv_integrableOn_Ioi
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    IntegrableOn
      (fun s : ℝ => (1 - s / τ - s ^ 2) * Real.exp (-s / τ) *
        gaussianPDFReal 0 (1 : NNReal) s)
      (Set.Ioi (0 : ℝ)) volume := by
  have h0 := standardGaussian_exp_neg_pow_integrableOn_Ioi τ hτ 0
  have h1 := standardGaussian_exp_neg_pow_integrableOn_Ioi τ hτ 1
  have h2 := standardGaussian_exp_neg_pow_integrableOn_Ioi τ hτ 2
  have h1scaled :
      IntegrableOn
        (fun s : ℝ => (s / τ) * Real.exp (-s / τ) *
          gaussianPDFReal 0 (1 : NNReal) s)
        (Set.Ioi (0 : ℝ)) volume := by
    have hc := h1.const_mul (1 / τ)
    change Integrable
      (fun s : ℝ => (s / τ) * Real.exp (-s / τ) *
        gaussianPDFReal 0 (1 : NNReal) s)
      (volume.restrict (Set.Ioi (0 : ℝ)))
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hc
  have hsum := (h0.sub h1scaled).sub h2
  refine hsum.congr_fun ?_ measurableSet_Ioi
  intro s hs
  simp
  ring

private lemma setIntegral_standardGaussian_eq_density
    (s : Set ℝ) (hs : MeasurableSet s) (f : ℝ → ℝ) :
    (∫ x in s, f x ∂(gaussianReal 0 (1 : NNReal))) =
      ∫ x in s, gaussianPDFReal 0 (1 : NNReal) x * f x := by
  rw [← integral_indicator hs]
  rw [integral_gaussianReal_eq_integral_smul (by norm_num : (1 : NNReal) ≠ 0)]
  conv_rhs => rw [← integral_indicator hs]
  apply integral_congr_ae
  filter_upwards with x
  by_cases hx : x ∈ s
  · simp [Set.indicator_of_mem hx, smul_eq_mul]
  · simp [Set.indicator, hx, smul_eq_mul]

private lemma standardGaussian_upperExpMass_eq_density
    (τ : ℝ) :
    upperExpMass τ (gaussianReal 0 (1 : NNReal)) 0 =
      ∫ s in Set.Ioi (0 : ℝ),
        gaussianPDFReal 0 (1 : NNReal) s * Real.exp (-s / τ) := by
  unfold upperExpMass
  rw [setIntegral_standardGaussian_eq_density (Set.Ioi (0 : ℝ)) measurableSet_Ioi]

private lemma standardGaussian_upperCompensatedMoment_eq_density
    (τ : ℝ) :
    upperCompensatedMoment τ (gaussianReal 0 (1 : NNReal)) 0 =
      ∫ s in Set.Ioi (0 : ℝ),
        gaussianPDFReal 0 (1 : NNReal) s * (s * Real.exp (-s / τ)) := by
  unfold upperCompensatedMoment
  rw [setIntegral_standardGaussian_eq_density (Set.Ioi (0 : ℝ)) measurableSet_Ioi]
  apply setIntegral_congr_fun measurableSet_Ioi
  intro s hs
  ring

private lemma standardGaussian_lowerExpMass_eq_upperExpMass
    (τ : ℝ) :
    lowerExpMass τ (gaussianReal 0 (1 : NNReal)) 0 =
      upperExpMass τ (gaussianReal 0 (1 : NNReal)) 0 := by
  unfold lowerExpMass
  rw [setIntegral_standardGaussian_eq_density (Set.Iic (0 : ℝ)) measurableSet_Iic]
  rw [standardGaussian_upperExpMass_eq_density τ]
  let f : ℝ → ℝ := fun s => gaussianPDFReal 0 (1 : NNReal) s * Real.exp (-s / τ)
  have hcomp :
      (∫ s in Set.Ioi (0 : ℝ), f s) =
        ∫ y in Set.Iic (0 : ℝ), f (-y) := by
    have h := (integral_comp_neg_Iic (0 : ℝ) f).symm
    rw [neg_zero] at h
    exact h
  rw [hcomp]
  apply setIntegral_congr_fun measurableSet_Iic
  intro y hy
  dsimp [f]
  have hpdf : gaussianPDFReal 0 (1 : NNReal) (-y) =
      gaussianPDFReal 0 (1 : NNReal) y := by
    unfold gaussianPDFReal
    congr 2
    ring
  rw [hpdf]
  ring_nf

private lemma standardGaussian_lowerCompensatedMoment_eq_upperCompensatedMoment
    (τ : ℝ) :
    lowerCompensatedMoment τ (gaussianReal 0 (1 : NNReal)) 0 =
      upperCompensatedMoment τ (gaussianReal 0 (1 : NNReal)) 0 := by
  unfold lowerCompensatedMoment
  rw [setIntegral_standardGaussian_eq_density (Set.Iic (0 : ℝ)) measurableSet_Iic]
  rw [standardGaussian_upperCompensatedMoment_eq_density τ]
  let f : ℝ → ℝ := fun s => gaussianPDFReal 0 (1 : NNReal) s *
    (s * Real.exp (-s / τ))
  have hcomp :
      (∫ s in Set.Ioi (0 : ℝ), f s) =
        ∫ y in Set.Iic (0 : ℝ), f (-y) := by
    have h := (integral_comp_neg_Iic (0 : ℝ) f).symm
    rw [neg_zero] at h
    exact h
  rw [hcomp]
  apply setIntegral_congr_fun measurableSet_Iic
  intro y hy
  dsimp [f]
  have hpdf : gaussianPDFReal 0 (1 : NNReal) (-y) =
      gaussianPDFReal 0 (1 : NNReal) y := by
    unfold gaussianPDFReal
    congr 2
    ring
  rw [hpdf]
  ring_nf

private lemma standardGaussian_halfPrimitive_tendsto_atTop_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    Tendsto
      (fun s : ℝ => s * Real.exp (-s / τ) * gaussianPDFReal 0 (1 : NNReal) s)
      atTop (𝓝 0) := by
  let C : ℝ := (√(2 * Real.pi * (1 : NNReal)))⁻¹
  have hbase :
      Tendsto (fun s : ℝ => C * (s * Real.exp (-(1 / τ) * s))) atTop (𝓝 0) := by
    have hraw :=
      tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero (1 : ℝ) (1 / τ)
        (one_div_pos.mpr hτ)
    have hlin :
        Tendsto (fun s : ℝ => s * Real.exp (-(1 / τ) * s)) atTop (𝓝 0) := by
      refine hraw.congr' ?_
      filter_upwards [eventually_ge_atTop (0 : ℝ)] with s hs
      rw [Real.rpow_one]
    simpa using (tendsto_const_nhds.mul hlin)
  refine squeeze_zero' ?_ ?_ hbase
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with s hs
    unfold gaussianPDFReal
    positivity
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with s hs
    unfold gaussianPDFReal C
    have hCnonneg : 0 ≤ (√(2 * Real.pi * (1 : NNReal)))⁻¹ := by positivity
    have hrest_nonneg : 0 ≤ s * Real.exp (-(1 / τ) * s) := by
      positivity
    have hquad_le_one : Real.exp (-(s - 0) ^ 2 / (2 * (1 : NNReal))) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      have hsq : 0 ≤ (s - 0) ^ 2 := sq_nonneg (s - 0)
      norm_num
      nlinarith
    calc
      s * Real.exp (-s / τ) *
          ((√(2 * Real.pi * (1 : NNReal)))⁻¹ *
            Real.exp (-(s - 0) ^ 2 / (2 * (1 : NNReal))))
          =
        ((√(2 * Real.pi * (1 : NNReal)))⁻¹ *
          Real.exp (-(s - 0) ^ 2 / (2 * (1 : NNReal)))) *
            (s * Real.exp (-(1 / τ) * s)) := by
            ring_nf
      _ ≤ (√(2 * Real.pi * (1 : NNReal)))⁻¹ *
            (s * Real.exp (-(1 / τ) * s)) := by
            exact mul_le_mul_of_nonneg_right
              (mul_le_of_le_one_right hCnonneg hquad_le_one) hrest_nonneg

private lemma standardGaussian_halfline_deriv_integral_eq_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    (∫ s in Set.Ioi (0 : ℝ),
      (1 - s / τ - s ^ 2) * Real.exp (-s / τ) *
        gaussianPDFReal 0 (1 : NNReal) s) = 0 := by
  have h := integral_Ioi_of_hasDerivAt_of_tendsto'
    (a := (0 : ℝ)) (m := (0 : ℝ))
    (f := fun s : ℝ =>
      s * Real.exp (-s / τ) * gaussianPDFReal 0 (1 : NNReal) s)
    (f' := fun s : ℝ =>
      (1 - s / τ - s ^ 2) * Real.exp (-s / τ) *
        gaussianPDFReal 0 (1 : NNReal) s)
    (fun s hs => hasDerivAt_standardGaussian_halfPrimitive τ s)
    (standardGaussian_halfPrimitive_deriv_integrableOn_Ioi τ hτ)
    (standardGaussian_halfPrimitive_tendsto_atTop_zero τ hτ)
  simpa using h

private lemma standardGaussian_halfline_square_integral_pos
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    0 < ∫ s in Set.Ioi (0 : ℝ),
      s ^ 2 * Real.exp (-s / τ) * gaussianPDFReal 0 (1 : NNReal) s := by
  have hnonneg :
      0 ≤ᵐ[volume.restrict (Set.Ioi (0 : ℝ))]
        fun s : ℝ => s ^ 2 * Real.exp (-s / τ) *
          gaussianPDFReal 0 (1 : NNReal) s := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with s hs
    exact mul_nonneg
      (mul_nonneg (sq_nonneg s) (Real.exp_pos _).le)
      (gaussianPDFReal_nonneg 0 (1 : NNReal) s)
  have hint := standardGaussian_exp_neg_pow_integrableOn_Ioi τ hτ 2
  rw [setIntegral_pos_iff_support_of_nonneg_ae hnonneg hint]
  have hsupport :
      Function.support
          (fun s : ℝ => s ^ 2 * Real.exp (-s / τ) *
            gaussianPDFReal 0 (1 : NNReal) s) ∩
        Set.Ioi (0 : ℝ) = Set.Ioi (0 : ℝ) := by
    ext s
    constructor
    · intro hs
      exact hs.2
    · intro hs
      refine ⟨?_, hs⟩
      rw [Function.mem_support]
      exact ne_of_gt
        (mul_pos
          (mul_pos (sq_pos_of_ne_zero (ne_of_gt hs)) (Real.exp_pos _))
          (gaussianPDFReal_pos 0 (1 : NNReal) s (by norm_num)))
  rw [hsupport]
  simp

private lemma standardGaussian_upperComp_div_tau_sub_upperExp_neg
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    upperCompensatedMoment τ (gaussianReal 0 (1 : NNReal)) 0 / τ -
        upperExpMass τ (gaussianReal 0 (1 : NNReal)) 0 < 0 := by
  let A : ℝ := ∫ s in Set.Ioi (0 : ℝ),
      gaussianPDFReal 0 (1 : NNReal) s * Real.exp (-s / τ)
  let C : ℝ := ∫ s in Set.Ioi (0 : ℝ),
      gaussianPDFReal 0 (1 : NNReal) s * (s * Real.exp (-s / τ))
  let S : ℝ := ∫ s in Set.Ioi (0 : ℝ),
      s ^ 2 * Real.exp (-s / τ) * gaussianPDFReal 0 (1 : NNReal) s
  have hderiv := standardGaussian_halfline_deriv_integral_eq_zero τ hτ
  have hAint0 := standardGaussian_exp_neg_pow_integrableOn_Ioi τ hτ 0
  have hAint :
      IntegrableOn
        (fun s : ℝ => gaussianPDFReal 0 (1 : NNReal) s * Real.exp (-s / τ))
        (Set.Ioi (0 : ℝ)) volume := by
    refine hAint0.congr_fun ?_ measurableSet_Ioi
    intro s hs
    simp
    ring
  have hCint0 := standardGaussian_exp_neg_pow_integrableOn_Ioi τ hτ 1
  have hCint :
      IntegrableOn
        (fun s : ℝ => gaussianPDFReal 0 (1 : NNReal) s *
          (s * Real.exp (-s / τ)))
        (Set.Ioi (0 : ℝ)) volume := by
    refine hCint0.congr_fun ?_ measurableSet_Ioi
    intro s hs
    simp
    ring
  have hSint := standardGaussian_exp_neg_pow_integrableOn_Ioi τ hτ 2
  have hCscaled :
      IntegrableOn
        (fun s : ℝ => (s / τ) * Real.exp (-s / τ) *
          gaussianPDFReal 0 (1 : NNReal) s)
        (Set.Ioi (0 : ℝ)) volume := by
    have h := hCint0.const_mul (1 / τ)
    change Integrable
      (fun s : ℝ => (s / τ) * Real.exp (-s / τ) *
        gaussianPDFReal 0 (1 : NNReal) s)
      (volume.restrict (Set.Ioi (0 : ℝ)))
    simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using h
  have hdecomp :
      (∫ s in Set.Ioi (0 : ℝ),
        (1 - s / τ - s ^ 2) * Real.exp (-s / τ) *
          gaussianPDFReal 0 (1 : NNReal) s) =
        A - C / τ - S := by
    calc
      (∫ s in Set.Ioi (0 : ℝ),
        (1 - s / τ - s ^ 2) * Real.exp (-s / τ) *
          gaussianPDFReal 0 (1 : NNReal) s)
          =
        ∫ s in Set.Ioi (0 : ℝ),
          (gaussianPDFReal 0 (1 : NNReal) s * Real.exp (-s / τ) -
            (s / τ) * Real.exp (-s / τ) *
              gaussianPDFReal 0 (1 : NNReal) s) -
            s ^ 2 * Real.exp (-s / τ) *
              gaussianPDFReal 0 (1 : NNReal) s := by
            apply setIntegral_congr_fun measurableSet_Ioi
            intro s hs
            ring
      _ =
        (∫ s in Set.Ioi (0 : ℝ),
          gaussianPDFReal 0 (1 : NNReal) s * Real.exp (-s / τ) -
            (s / τ) * Real.exp (-s / τ) *
              gaussianPDFReal 0 (1 : NNReal) s) -
          ∫ s in Set.Ioi (0 : ℝ),
            s ^ 2 * Real.exp (-s / τ) *
              gaussianPDFReal 0 (1 : NNReal) s := by
            rw [integral_sub]
            · exact hAint.sub hCscaled
            · exact hSint
      _ =
        ((∫ s in Set.Ioi (0 : ℝ),
          gaussianPDFReal 0 (1 : NNReal) s * Real.exp (-s / τ)) -
          ∫ s in Set.Ioi (0 : ℝ),
            (s / τ) * Real.exp (-s / τ) *
              gaussianPDFReal 0 (1 : NNReal) s) - S := by
            rw [integral_sub hAint hCscaled]
      _ = A - C / τ - S := by
            have hCeq :
                (∫ s in Set.Ioi (0 : ℝ),
                  (s / τ) * Real.exp (-s / τ) *
                    gaussianPDFReal 0 (1 : NNReal) s) = C / τ := by
              calc
                (∫ s in Set.Ioi (0 : ℝ),
                  (s / τ) * Real.exp (-s / τ) *
                    gaussianPDFReal 0 (1 : NNReal) s)
                    =
                  (1 / τ) * C := by
                    rw [← integral_const_mul]
                    apply setIntegral_congr_fun measurableSet_Ioi
                    intro s hs
                    dsimp [C]
                    ring
                _ = C / τ := by ring
            rw [hCeq]
  have hSpos : 0 < S := standardGaussian_halfline_square_integral_pos τ hτ
  have hzero : A - C / τ - S = 0 := by
    rw [← hdecomp]
    exact hderiv
  rw [standardGaussian_upperCompensatedMoment_eq_density,
    standardGaussian_upperExpMass_eq_density]
  change C / τ - A < 0
  linarith

private lemma standardGaussian_upperExpMass_pos
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    0 < upperExpMass τ (gaussianReal 0 (1 : NNReal)) 0 := by
  rw [standardGaussian_upperExpMass_eq_density]
  have hnonneg :
      0 ≤ᵐ[volume.restrict (Set.Ioi (0 : ℝ))]
        fun s : ℝ => gaussianPDFReal 0 (1 : NNReal) s * Real.exp (-s / τ) := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with s hs
    exact mul_nonneg (gaussianPDFReal_nonneg 0 (1 : NNReal) s) (Real.exp_pos _).le
  have hint0 := standardGaussian_exp_neg_pow_integrableOn_Ioi τ hτ 0
  have hint :
      IntegrableOn
        (fun s : ℝ => gaussianPDFReal 0 (1 : NNReal) s * Real.exp (-s / τ))
        (Set.Ioi (0 : ℝ)) volume := by
    refine hint0.congr_fun ?_ measurableSet_Ioi
    intro s hs
    simp
    ring
  rw [setIntegral_pos_iff_support_of_nonneg_ae hnonneg hint]
  have hsupport :
      Function.support
          (fun s : ℝ => gaussianPDFReal 0 (1 : NNReal) s * Real.exp (-s / τ)) ∩
        Set.Ioi (0 : ℝ) = Set.Ioi (0 : ℝ) := by
    ext s
    constructor
    · intro hs
      exact hs.2
    · intro hs
      refine ⟨?_, hs⟩
      rw [Function.mem_support]
      exact ne_of_gt
        (mul_pos (gaussianPDFReal_pos 0 (1 : NNReal) s (by norm_num))
          (Real.exp_pos _))
  rw [hsupport]
  simp

private theorem standardGaussian_laplaceTiltedMeanRightDerivCoeff_zero_lt_one
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    laplaceTiltedMeanRightDerivCoeff τ (gaussianReal 0 (1 : NNReal)) 0 < 1 := by
  let A : ℝ := upperExpMass τ (gaussianReal 0 (1 : NNReal)) 0
  let C : ℝ := upperCompensatedMoment τ (gaussianReal 0 (1 : NNReal)) 0
  have hApos : 0 < A := by
    dsimp [A]
    exact standardGaussian_upperExpMass_pos τ hτ
  have hCtau_lt_A : C / τ < A := by
    dsimp [C, A]
    have h := standardGaussian_upperComp_div_tau_sub_upperExp_neg τ hτ
    linarith
  have hZ :
      kernelNormalizer (laplaceKernel τ) (gaussianReal 0 (1 : NNReal)) 0 =
        2 * A := by
    rw [laplaceKernelNormalizer_eq_lower_upper τ hτ
      (gaussianReal 0 (1 : NNReal)) 0]
    rw [standardGaussian_lowerExpMass_eq_upperExpMass τ]
    dsimp [A]
    norm_num
    ring
  unfold laplaceTiltedMeanRightDerivCoeff
  rw [standardGaussian_lowerExpMass_eq_upperExpMass τ,
    standardGaussian_lowerCompensatedMoment_eq_upperCompensatedMoment τ, hZ]
  dsimp [A, C] at *
  have hτpos : 0 < τ := hτ
  have hAne : A ≠ 0 := ne_of_gt hApos
  field_simp [hτ.ne', hAne]
  have hClt :
      upperCompensatedMoment τ (gaussianReal 0 (1 : NNReal)) 0 <
        τ * upperExpMass τ (gaussianReal 0 (1 : NNReal)) 0 := by
    have h := (div_lt_iff₀ hτpos).mp hCtau_lt_A
    nlinarith
  nlinarith

private theorem standardGaussian_laplaceMeanShiftRatioDeriv_zero_neg
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    laplaceMeanShiftRatioDeriv τ (gaussianReal 0 (1 : NNReal)) 0 < 0 := by
  let p : Measure ℝ := gaussianReal 0 (1 : NNReal)
  have hpReg : LaplaceC2NormalizerRegular τ p :=
    laplaceC2NormalizerRegular_of_continuousDensity τ hτ p
      (gaussianPDFReal 0 (1 : NNReal))
      (continuousDensityMeasure_gaussianReal 0 (by norm_num : (1 : NNReal) ≠ 0))
  have hregular :
      HasDerivWithinAt (laplaceTiltedMean τ p)
        (laplaceMeanShiftRatioDeriv τ p 0 + 1) (Set.Ici (0 : ℝ)) 0 :=
    (hasDerivAt_laplaceTiltedMean_regular τ hτ p hpReg 0).hasDerivWithinAt
  have hright :
      HasDerivWithinAt (laplaceTiltedMean τ p)
        (laplaceTiltedMeanRightDerivCoeff τ p 0) (Set.Ici (0 : ℝ)) 0 := by
    have hdisp :=
      hasDerivWithinAt_Ici_laplaceTiltedMeanFromDisplacement τ hτ p 0
    have heq : laplaceTiltedMean τ p = laplaceTiltedMeanFromDisplacement τ p := by
      funext y
      exact laplaceTiltedMean_eq_fromDisplacement τ hτ p y
    simpa [heq] using hdisp
  have hEq :
      laplaceMeanShiftRatioDeriv τ p 0 + 1 =
        laplaceTiltedMeanRightDerivCoeff τ p 0 := by
    have h₁ := hregular.derivWithin (uniqueDiffWithinAt_Ici (0 : ℝ))
    have h₂ := hright.derivWithin (uniqueDiffWithinAt_Ici (0 : ℝ))
    exact h₁.symm.trans h₂
  have hcoeff :
      laplaceTiltedMeanRightDerivCoeff τ p 0 < 1 := by
    dsimp [p]
    exact standardGaussian_laplaceTiltedMeanRightDerivCoeff_zero_lt_one τ hτ
  linarith

private theorem standardGaussian_laplaceMeanShiftRatio_neg_of_pos
    (τ : ℝ) (hτ : ValidBandwidth τ) {x : ℝ} (hx : 0 < x) :
    laplaceMeanShiftRatio τ (gaussianReal 0 (1 : NNReal)) x < 0 := by
  unfold laplaceMeanShiftRatio
  exact div_neg_of_neg_of_pos
    (standardGaussian_laplaceDisplacementIntegral_neg_of_pos τ hτ hx)
    (laplaceKernelNormalizer_pos (gaussianReal 0 (1 : NNReal)) τ hτ x)

private theorem standardGaussian_laplaceMeanShiftRatio_pos_of_neg
    (τ : ℝ) (hτ : ValidBandwidth τ) {x : ℝ} (hx : x < 0) :
    0 < laplaceMeanShiftRatio τ (gaussianReal 0 (1 : NNReal)) x := by
  unfold laplaceMeanShiftRatio
  exact div_pos
    (standardGaussian_laplaceDisplacementIntegral_pos_of_neg τ hτ hx)
    (laplaceKernelNormalizer_pos (gaussianReal 0 (1 : NNReal)) τ hτ x)

/-- Axiom-free standard-Gaussian one-crossing certificate for the 1-d Laplace
mean-shift ratio.

For every positive bandwidth, the centered unit Gaussian has its unique
downward crossing at `0`: the ratio is positive to the left, negative to the
right, and its derivative at `0` is strictly negative. -/
noncomputable def standardGaussianLaplaceSingleDownCertificate
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    StandardGaussianLaplaceSingleDownCertificate τ :=
  { zero :=
      { zero := standardGaussian_laplaceMeanShiftRatio_zero τ hτ
        simple := ne_of_lt (standardGaussian_laplaceMeanShiftRatioDeriv_zero_neg τ hτ) }
    hm_left := fun t ht => standardGaussian_laplaceMeanShiftRatio_pos_of_neg τ hτ (x := t) ht
    hm_right := fun t ht => standardGaussian_laplaceMeanShiftRatio_neg_of_pos τ hτ (x := t) ht }

/-- Concrete standard-vs-shifted Gaussian finite-simple-zero package, with the
Gaussian sign certificate constructed internally. -/
noncomputable def standardGaussian_vs_shiftedGaussian_finiteSimpleZeros
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    LaplaceACContinuousDensityFiniteSimpleZeros τ
      (gaussianReal 0 (1 : NNReal)) (gaussianReal 1 (1 : NNReal)) :=
  standardGaussian_vs_shiftedGaussian_finiteSimpleZeros_of_certificate τ
    (standardGaussianLaplaceSingleDownCertificate τ hτ)

/-- The finite-simple-zero condition admits a concrete distinct Gaussian pair
for every positive bandwidth. -/
theorem laplaceACFiniteSimpleZerosCondition_allowsDistinctPair_of_standardGaussian
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    ConditionAllowsDistinctPair
      (LaplaceACContinuousDensityFiniteSimpleZerosCondition τ) :=
  laplaceACFiniteSimpleZerosCondition_allowsDistinctPair_of_package
    (standardGaussian_vs_shiftedGaussian_finiteSimpleZeros τ hτ)
    gaussianReal_zero_ne_one_unitVariance

/-- The finite-simple-zero a.c. Laplace condition is formally legitimate:
it is inhabited and it allows a distinct pair before any zero-drift hypothesis
is imposed. -/
theorem laplaceACFiniteSimpleZerosCondition_isLegitimate_of_standardGaussian
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    IsLegitimateCondition
      (LaplaceACContinuousDensityFiniteSimpleZerosCondition τ) := by
  constructor
  · exact ⟨gaussianReal 0 (1 : NNReal), gaussianReal 1 (1 : NNReal),
      laplaceACFiniteSimpleZerosCondition_of_package
        (standardGaussian_vs_shiftedGaussian_finiteSimpleZeros τ hτ)⟩
  · exact laplaceACFiniteSimpleZerosCondition_allowsDistinctPair_of_standardGaussian τ hτ

end DriftingIdentifiability
