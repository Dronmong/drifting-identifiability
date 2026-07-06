import DriftingIdentifiability.PracticalModelClasses

/-!
# A concrete continuum-supported smooth probability-density basis

`PracticalModelClasses.lean` provides the smooth/continuous model-class
*interfaces* (`SmoothProbabilityDensityBasis`, `continuousPerturbationSetup`) but
leaves open the single substantive instantiation that Objective 3 still needs: an
actual continuum-supported smooth basis with a *certified* frame bound.

This module supplies it. The reference law is the standard Gaussian on `ℝ`; the
two component densities are `C∞` bump functions normalized against it, placed on
disjoint, *ordered* supports (`φ₀` entirely negative, `φ₁` entirely positive).
Their mixtures are genuine continuum-supported probability measures.

The frame bound is proved *directly*, with no Gaussian-integral closed form:
because the supports are ordered, the sign of `y₊ - y₋` is constant on the
support of `φ₀(y₊)φ₁(y₋)`, so the single interaction double integral is
sign-definite and therefore nonzero. `interactionFrameBound_two` then yields a
positive frame constant.
-/

open scoped BigOperators ENNReal
open MeasureTheory ProbabilityTheory Metric

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

/-! ## Continuum Gaussian reference with full support -/

/-- The continuum reference law: the standard Gaussian probability measure. -/
noncomputable def bumpReference : Distribution ℝ := gaussianReal 0 1

instance : IsProbabilityMeasure bumpReference := by
  unfold bumpReference; infer_instance

/-- The reference is genuinely non-atomic. -/
instance : NoAtoms bumpReference := by
  unfold bumpReference
  exact noAtoms_gaussianReal one_ne_zero

instance : bumpReference.IsOpenPosMeasure := by
  refine ⟨fun U hU hUne hzero => ?_⟩
  have hmeas : Measurable (gaussianPDF 0 1) :=
    (measurable_gaussianPDFReal 0 1).ennreal_ofReal
  unfold bumpReference at hzero
  rw [gaussianReal_of_var_ne_zero 0 one_ne_zero,
    withDensity_apply _ hU.measurableSet] at hzero
  have hpos : 0 < ∫⁻ x in U, gaussianPDF 0 1 x ∂volume := by
    rw [setLIntegral_pos_iff hmeas]
    have hsupp : Function.support (gaussianPDF 0 1) = Set.univ := by
      ext x
      simp only [Function.mem_support, ne_eq, Set.mem_univ, iff_true]
      exact (gaussianPDF_pos 0 one_ne_zero x).ne'
    rw [hsupp, Set.univ_inter]
    exact hU.measure_pos _ hUne
  exact hpos.ne' hzero

/-! ## Two normalized smooth bumps on disjoint ordered supports -/

/-- Smooth bump centered at `-1` with support `ball (-1) (1/2) = (-3/2, -1/2)`. -/
noncomputable def leftBump : ContDiffBump (-1 : ℝ) :=
  ⟨1 / 4, 1 / 2, by norm_num, by norm_num⟩

/-- Smooth bump centered at `1` with support `ball 1 (1/2) = (1/2, 3/2)`. -/
noncomputable def rightBump : ContDiffBump (1 : ℝ) :=
  ⟨1 / 4, 1 / 2, by norm_num, by norm_num⟩

/-- Left component density: a normalized bump, all of whose mass sits at
negative points. -/
noncomputable def leftDensity : ℝ → ℝ := leftBump.normed bumpReference

/-- Right component density: a normalized bump, all of whose mass sits at
positive points. -/
noncomputable def rightDensity : ℝ → ℝ := rightBump.normed bumpReference

/-- The two smooth densities, indexed by `Fin 2`. -/
noncomputable def bumpDensity : Fin 2 → ℝ → ℝ := ![leftDensity, rightDensity]

theorem leftDensity_support :
    Function.support leftDensity = ball (-1 : ℝ) (1 / 2) :=
  leftBump.support_normed_eq

theorem rightDensity_support :
    Function.support rightDensity = ball (1 : ℝ) (1 / 2) :=
  rightBump.support_normed_eq

theorem bumpDensity_hasCompactSupport (i : Fin 2) :
    HasCompactSupport (bumpDensity i) := by
  fin_cases i
  · exact leftBump.hasCompactSupport_normed
  · exact rightBump.hasCompactSupport_normed

theorem continuous_bumpDensity (i : Fin 2) : Continuous (bumpDensity i) := by
  fin_cases i
  · exact leftBump.continuous_normed
  · exact rightBump.continuous_normed

/-- Everywhere in the support of `leftDensity`, the point is strictly negative. -/
theorem leftDensity_ne_zero_neg {y : ℝ} (hy : leftDensity y ≠ 0) : y < 0 := by
  have hmem : y ∈ ball (-1 : ℝ) (1 / 2) := by
    rw [← leftDensity_support]; exact hy
  rw [mem_ball, Real.dist_eq] at hmem
  rw [abs_lt] at hmem
  linarith [hmem.2]

/-- Everywhere in the support of `rightDensity`, the point is strictly
positive. -/
theorem rightDensity_ne_zero_pos {y : ℝ} (hy : rightDensity y ≠ 0) : 0 < y := by
  have hmem : y ∈ ball (1 : ℝ) (1 / 2) := by
    rw [← rightDensity_support]; exact hy
  rw [mem_ball, Real.dist_eq] at hmem
  rw [abs_lt] at hmem
  linarith [hmem.1]

/-- Each component density is `C∞` (smooth), recorded separately: the
`ContinuousProbabilityDensityBasis` interface only asks for continuity, but these
bumps are genuinely smooth. -/
theorem bumpDensity_contDiff (i : Fin 2) :
    ContDiff ℝ (⊤ : ℕ∞) (bumpDensity i) := by
  fin_cases i
  · exact leftBump.contDiff_normed
  · exact rightBump.contDiff_normed

/-- The bump densities form a genuine continuum-supported probability-density
basis with continuous (indeed `C∞`) components. -/
noncomputable def bumpContinuousBasis :
    ContinuousProbabilityDensityBasis ℝ bumpReference 2 where
  density := bumpDensity
  measurable_density i := by
    fin_cases i
    · exact leftBump.continuous_normed.measurable
    · exact rightBump.continuous_normed.measurable
  nonnegative i x := by
    fin_cases i
    · exact leftBump.nonneg_normed x
    · exact rightBump.nonneg_normed x
  integrable_density i := by
    fin_cases i
    · exact leftBump.integrable_normed
    · exact rightBump.integrable_normed
  integral_density i := by
    fin_cases i
    · exact leftBump.integral_normed
    · exact rightBump.integral_normed
  continuous_density i := by
    fin_cases i
    · exact leftBump.continuous_normed
    · exact rightBump.continuous_normed

/-- The same concrete basis registered through the project's `C∞` interface. -/
noncomputable def bumpSmoothBasis :
    SmoothProbabilityDensityBasis ℝ bumpReference 2 where
  toProbabilityDensityBasis := bumpContinuousBasis.toProbabilityDensityBasis
  smooth_density i := by
    simpa only [bumpContinuousBasis] using bumpDensity_contDiff i

/-- Every represented bump-mixture law is non-atomic, not merely defined over
a continuum reference. -/
instance bumpBasisMeasure_noAtoms (a : FiniteProbabilityVector 2) :
    NoAtoms (bumpContinuousBasis.toProbabilityDensityBasis.basisMeasure a) := by
  unfold ProbabilityDensityBasis.basisMeasure
  infer_instance

/-! ## Sign-definite interaction and the certified frame -/

section BumpFrame

variable (σ : ℝ)

/-- The scalar mean-shift interaction integrand for the bump basis.  The grouping
matches the `smul`-expanded interaction kernel. -/
noncomputable def bumpIntegrand (x : ℝ) : ℝ × ℝ → ℝ :=
  fun y => leftDensity y.1 * rightDensity y.2 *
    (gaussianKernel σ x y.1 * gaussianKernel σ x y.2 * (y.1 - y.2))

theorem continuous_bumpIntegrand (x : ℝ) : Continuous (bumpIntegrand σ x) := by
  have hL : Continuous leftDensity := leftBump.continuous_normed
  have hR : Continuous rightDensity := rightBump.continuous_normed
  have hk : Continuous (fun y : ℝ => gaussianKernel σ x y) := by
    unfold gaussianKernel; fun_prop
  exact (((hL.comp continuous_fst).mul (hR.comp continuous_snd)).mul
    (((hk.comp continuous_fst).mul (hk.comp continuous_snd)).mul
      (continuous_fst.sub continuous_snd)))

theorem hasCompactSupport_bumpIntegrand (x : ℝ) :
    HasCompactSupport (bumpIntegrand σ x) := by
  apply HasCompactSupport.intro
    (K := tsupport leftDensity ×ˢ tsupport rightDensity)
    (IsCompact.prod leftBump.hasCompactSupport_normed
      rightBump.hasCompactSupport_normed)
  intro y hy
  rw [Set.mem_prod, not_and_or] at hy
  unfold bumpIntegrand
  rcases hy with h | h
  · rw [image_eq_zero_of_notMem_tsupport h]; ring
  · rw [image_eq_zero_of_notMem_tsupport h]; ring

/-- The bump interaction integrand is integrable against the product reference:
it is continuous with compact support. -/
theorem integrable_bumpIntegrand (x : ℝ) :
    Integrable (bumpIntegrand σ x) (bumpReference.prod bumpReference) :=
  (continuous_bumpIntegrand σ x).integrable_of_hasCompactSupport
    (hasCompactSupport_bumpIntegrand σ x)

/-- The interaction double integral equals the product integral of
`bumpIntegrand`. -/
theorem basisInteraction_bump_eq (x : ℝ) :
    basisInteraction bumpReference (meanShiftInteractionKernel (gaussianKernel σ))
      leftDensity rightDensity x =
      ∫ y, bumpIntegrand σ x y ∂(bumpReference.prod bumpReference) := by
  rw [integral_prod _ (integrable_bumpIntegrand σ x)]
  unfold basisInteraction
  simp only [meanShiftInteractionKernel, bumpIntegrand, smul_eq_mul]

/-- The negated integrand is everywhere nonnegative: the ordered supports force
`y₂ - y₁ > 0` wherever both densities are nonzero. -/
theorem neg_bumpIntegrand_nonneg (x : ℝ) (y : ℝ × ℝ) :
    0 ≤ -bumpIntegrand σ x y := by
  rcases eq_or_ne (leftDensity y.1) 0 with hL | hL
  · simp [bumpIntegrand, hL]
  rcases eq_or_ne (rightDensity y.2) 0 with hR | hR
  · simp [bumpIntegrand, hR]
  have hy1 : y.1 < 0 := leftDensity_ne_zero_neg hL
  have hy2 : 0 < y.2 := rightDensity_ne_zero_pos hR
  have hLpos : 0 < leftDensity y.1 :=
    lt_of_le_of_ne (leftBump.nonneg_normed y.1) (Ne.symm hL)
  have hRpos : 0 < rightDensity y.2 :=
    lt_of_le_of_ne (rightBump.nonneg_normed y.2) (Ne.symm hR)
  have hk1 : 0 < gaussianKernel σ x y.1 := by rw [gaussianKernel]; exact Real.exp_pos _
  have hk2 : 0 < gaussianKernel σ x y.2 := by rw [gaussianKernel]; exact Real.exp_pos _
  have : bumpIntegrand σ x y < 0 := by
    unfold bumpIntegrand
    have hsub : y.1 - y.2 < 0 := by linarith
    nlinarith [mul_pos (mul_pos (mul_pos hLpos hRpos) (mul_pos hk1 hk2)) (neg_pos.mpr hsub)]
  linarith

/-- **The single interaction vector is nonzero.** The negated integrand is
nonnegative and strictly positive on `ball (-1) (1/2) ×ˢ ball 1 (1/2)`, a set of
positive product-reference measure, so the interaction double integral is
strictly negative. -/
theorem basisInteraction_bump_neg (x : ℝ) :
    basisInteraction bumpReference (meanShiftInteractionKernel (gaussianKernel σ))
      leftDensity rightDensity x < 0 := by
  rw [basisInteraction_bump_eq]
  have hnonneg : 0 ≤ fun y => -bumpIntegrand σ x y := fun y => neg_bumpIntegrand_nonneg σ x y
  have hint : Integrable (fun y => -bumpIntegrand σ x y)
      (bumpReference.prod bumpReference) := (integrable_bumpIntegrand σ x).neg
  have hball : ball (-1 : ℝ) (1 / 2) ×ˢ ball (1 : ℝ) (1 / 2) ⊆
      Function.support (fun y => -bumpIntegrand σ x y) := by
    intro y hy
    rw [Set.mem_prod] at hy
    have hL : leftDensity y.1 ≠ 0 := by
      rw [← leftDensity_support] at hy; exact hy.1
    have hR : rightDensity y.2 ≠ 0 := by
      rw [← rightDensity_support] at hy; exact hy.2
    have hy1 : y.1 < 0 := leftDensity_ne_zero_neg hL
    have hy2 : 0 < y.2 := rightDensity_ne_zero_pos hR
    have hLpos : 0 < leftDensity y.1 :=
      lt_of_le_of_ne (leftBump.nonneg_normed y.1) (Ne.symm hL)
    have hRpos : 0 < rightDensity y.2 :=
      lt_of_le_of_ne (rightBump.nonneg_normed y.2) (Ne.symm hR)
    have hk1 : 0 < gaussianKernel σ x y.1 := by rw [gaussianKernel]; exact Real.exp_pos _
    have hk2 : 0 < gaussianKernel σ x y.2 := by rw [gaussianKernel]; exact Real.exp_pos _
    rw [Function.mem_support]
    have : bumpIntegrand σ x y < 0 := by
      unfold bumpIntegrand
      nlinarith [mul_pos (mul_pos (mul_pos hLpos hRpos) (mul_pos hk1 hk2))
        (neg_pos.mpr (by linarith : y.1 - y.2 < 0))]
    linarith
  have hballpos : 0 < (bumpReference.prod bumpReference)
      (ball (-1 : ℝ) (1 / 2) ×ˢ ball (1 : ℝ) (1 / 2)) := by
    rw [Measure.prod_prod]
    have h1 : 0 < bumpReference (ball (-1 : ℝ) (1 / 2)) :=
      measure_ball_pos bumpReference (-1) (by norm_num)
    have h2 : 0 < bumpReference (ball (1 : ℝ) (1 / 2)) :=
      measure_ball_pos bumpReference 1 (by norm_num)
    exact ENNReal.mul_pos h1.ne' h2.ne'
  have hpos : 0 < ∫ y, -bumpIntegrand σ x y ∂(bumpReference.prod bumpReference) := by
    rw [integral_pos_iff_support_of_nonneg hnonneg hint]
    exact lt_of_lt_of_le hballpos (measure_mono hball)
  rw [integral_neg] at hpos
  linarith

/-- **Certified frame bound for the smooth continuum basis.** With one probe, the
actual integral-induced interaction family of the bump basis satisfies a positive
`InteractionFrameBound`, discharged directly through the sign argument. -/
theorem bumpInteractionFrameBound (x0 : ℝ) :
    InteractionFrameBound
      (inducedInteractionVector bumpReference
        (meanShiftInteractionKernel (gaussianKernel σ)) bumpDensity
        (fun _ : Fin 1 => x0))
      ‖inducedInteractionVector bumpReference
        (meanShiftInteractionKernel (gaussianKernel σ)) bumpDensity
        (fun _ : Fin 1 => x0) 0 1‖ := by
  apply interactionFrameBound_two
  rw [Function.ne_iff]
  refine ⟨0, ?_⟩
  simp only [inducedInteractionVector, Pi.zero_apply]
  have hne : basisInteraction bumpReference
      (meanShiftInteractionKernel (gaussianKernel σ))
      (bumpDensity 0) (bumpDensity 1) x0 ≠ 0 := by
    have h := basisInteraction_bump_neg σ x0
    simp only [bumpDensity, Matrix.cons_val_zero, Matrix.cons_val_one]
    exact ne_of_lt h
  exact hne

end BumpFrame

/-! ## End-to-end identifiability for the smooth continuum basis -/

section BumpSetup

variable (σ : ℝ)

/-- The bump mixture density is continuous. -/
theorem continuous_bumpMixture (a : FiniteProbabilityVector 2) :
    Continuous (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity a) := by
  unfold ProbabilityDensityBasis.mixtureDensity basisDensity
  exact continuous_finsetSum _ fun i _ =>
    continuous_const.mul (bumpContinuousBasis.continuous_density i)

/-- The bump mixture density has compact support. -/
theorem hasCompactSupport_bumpMixture (a : FiniteProbabilityVector 2) :
    HasCompactSupport (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity a) := by
  apply HasCompactSupport.intro
    (K := tsupport leftDensity ∪ tsupport rightDensity)
    (IsCompact.union leftBump.hasCompactSupport_normed
      rightBump.hasCompactSupport_normed)
  intro y hy
  rw [Set.mem_union, not_or] at hy
  have h0 : bumpContinuousBasis.toProbabilityDensityBasis.density 0 y = 0 :=
    image_eq_zero_of_notMem_tsupport hy.1
  have h1 : bumpContinuousBasis.toProbabilityDensityBasis.density 1 y = 0 :=
    image_eq_zero_of_notMem_tsupport hy.2
  simp only [ProbabilityDensityBasis.mixtureDensity, basisDensity, Fin.sum_univ_two,
    h0, h1, mul_zero, add_zero]

/-- Any continuous function is integrable against a bump-basis mixture measure:
the compactly supported mixture density makes the `withDensity` integrand
compactly supported. -/
theorem bumpBasis_integrable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (a : FiniteProbabilityVector 2) (f : ℝ → F) (hf : Continuous f) :
    Integrable f (bumpContinuousBasis.toProbabilityDensityBasis.basisMeasure a) := by
  unfold ProbabilityDensityBasis.basisMeasure
  rw [integrable_withDensity_iff_integrable_smul'
    (bumpContinuousBasis.toProbabilityDensityBasis.measurable_mixtureDensity a).ennreal_ofReal
    (Filter.Eventually.of_forall fun _ => ENNReal.ofReal_lt_top)]
  have hEq : (fun y => (ENNReal.ofReal
        (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity a y)).toReal • f y)
      = fun y => bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity a y • f y := by
    funext y
    rw [ENNReal.toReal_ofReal
      (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity_nonnegative a y)]
  rw [hEq]
  exact ((continuous_bumpMixture a).smul hf).integrable_of_hasCompactSupport
    (hasCompactSupport_bumpMixture a).smul_right

/-- Every empirical Gaussian normalizer of a bump mixture is strictly positive:
the kernel is positive everywhere and the mixture is a probability measure. -/
theorem bumpBasis_normalizer_pos (a : FiniteProbabilityVector 2) (x : ℝ) :
    0 < kernelNormalizer (gaussianKernel σ)
      (bumpContinuousBasis.toProbabilityDensityBasis.basisMeasure a) x := by
  letI := bumpContinuousBasis.toProbabilityDensityBasis.basisMeasure_isProbability a
  unfold kernelNormalizer
  rw [integral_pos_iff_support_of_nonneg
    (fun y => by rw [gaussianKernel]; exact (Real.exp_pos _).le)
    (bumpBasis_integrable a _ (by unfold gaussianKernel; fun_prop))]
  have hsupp : Function.support (fun y => gaussianKernel σ x y) = Set.univ := by
    ext y
    simp only [Function.mem_support, ne_eq, Set.mem_univ, iff_true]
    rw [gaussianKernel]; exact (Real.exp_pos _).ne'
  rw [hsupp, measure_univ]
  exact one_pos

/-- The product of two bump mixture densities has compact support. -/
theorem hasCompactSupport_bumpProdMixture (a b : FiniteProbabilityVector 2) :
    HasCompactSupport (fun z : ℝ × ℝ =>
      bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity a z.1 *
        bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity b z.2) := by
  apply HasCompactSupport.intro
    (K := tsupport (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity a) ×ˢ
      tsupport (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity b))
    (IsCompact.prod (hasCompactSupport_bumpMixture a) (hasCompactSupport_bumpMixture b))
  intro z hz
  rw [Set.mem_prod, not_and_or] at hz
  rcases hz with h | h
  · rw [image_eq_zero_of_notMem_tsupport h, zero_mul]
  · rw [image_eq_zero_of_notMem_tsupport h, mul_zero]

/-- Continuity of the mean-shift interaction kernel in the two sample slots. -/
theorem continuous_meanShiftKernel_bump (x : ℝ) :
    Continuous (fun z : ℝ × ℝ =>
      meanShiftInteractionKernel (gaussianKernel σ) x z.1 z.2) := by
  have hk : Continuous (fun y : ℝ => gaussianKernel σ x y) := by
    unfold gaussianKernel; fun_prop
  simp only [meanShiftInteractionKernel]
  exact (((hk.comp continuous_fst).mul (hk.comp continuous_snd)).smul
    (continuous_fst.sub continuous_snd))

/-- The interaction kernel is integrable against a product of two bump mixtures:
converting through `prod_withDensity`, the integrand acquires the compactly
supported product mixture density as a factor. -/
theorem bumpBasis_integrable_interaction (a b : FiniteProbabilityVector 2) (x : ℝ) :
    Integrable (fun z : ℝ × ℝ =>
      meanShiftInteractionKernel (gaussianKernel σ) x z.1 z.2)
      ((bumpContinuousBasis.toProbabilityDensityBasis.basisMeasure a).prod
        (bumpContinuousBasis.toProbabilityDensityBasis.basisMeasure b)) := by
  unfold ProbabilityDensityBasis.basisMeasure
  rw [prod_withDensity
    (bumpContinuousBasis.toProbabilityDensityBasis.measurable_mixtureDensity a).ennreal_ofReal
    (bumpContinuousBasis.toProbabilityDensityBasis.measurable_mixtureDensity b).ennreal_ofReal,
    integrable_withDensity_iff_integrable_smul'
    (show Measurable (fun z : ℝ × ℝ =>
        ENNReal.ofReal
            (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity a z.1) *
          ENNReal.ofReal
            (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity b z.2))
      from (((bumpContinuousBasis.toProbabilityDensityBasis.measurable_mixtureDensity a).comp
          measurable_fst).ennreal_ofReal.mul
        ((bumpContinuousBasis.toProbabilityDensityBasis.measurable_mixtureDensity b).comp
          measurable_snd).ennreal_ofReal))
    (Filter.Eventually.of_forall fun _ => ENNReal.mul_lt_top ENNReal.ofReal_lt_top
      ENNReal.ofReal_lt_top)]
  have hEq : (fun z : ℝ × ℝ =>
        (ENNReal.ofReal
            (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity a z.1) *
          ENNReal.ofReal
            (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity b z.2)).toReal •
          meanShiftInteractionKernel (gaussianKernel σ) x z.1 z.2)
      = fun z => (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity a z.1 *
          bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity b z.2) •
          meanShiftInteractionKernel (gaussianKernel σ) x z.1 z.2 := by
    funext z
    rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal
        (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity_nonnegative a z.1),
      ENNReal.toReal_ofReal
        (bumpContinuousBasis.toProbabilityDensityBasis.mixtureDensity_nonnegative b z.2)]
  rw [hEq]
  refine Continuous.integrable_of_hasCompactSupport ?_ ?_
  · exact (((continuous_bumpMixture a).comp continuous_fst).mul
      ((continuous_bumpMixture b).comp continuous_snd)).smul
      (continuous_meanShiftKernel_bump σ x)
  · exact (hasCompactSupport_bumpProdMixture a b).smul_right

/-- All mean-shift regularity obligations hold for the bump Gaussian model. -/
theorem bumpGaussian_meanShiftRegular (a b : FiniteProbabilityVector 2) (x : ℝ) :
    MeanShiftRegularAt (gaussianKernel σ)
      (bumpContinuousBasis.toProbabilityDensityBasis.basisMeasure a)
      (bumpContinuousBasis.toProbabilityDensityBasis.basisMeasure b) x where
  zp_ne_zero := (bumpBasis_normalizer_pos σ a x).ne'
  zq_ne_zero := (bumpBasis_normalizer_pos σ b x).ne'
  integrable_p := bumpBasis_integrable a _ (by
    have hk : Continuous (fun y : ℝ => gaussianKernel σ x y) := by
      unfold gaussianKernel; fun_prop
    exact hk.smul (continuous_id.sub continuous_const))
  integrable_q := bumpBasis_integrable b _ (by
    have hk : Continuous (fun y : ℝ => gaussianKernel σ x y) := by
      unfold gaussianKernel; fun_prop
    exact hk.smul (continuous_id.sub continuous_const))
  integrable_product := by
    have := bumpBasis_integrable_interaction σ a b x
    simpa only [meanShiftInteractionKernel] using this

/-- The basis-density interaction integrand has compact support (both densities
are bumps), so it is integrable against the product reference. -/
theorem bumpBasis_integrable_basisInteraction (i j : Fin 2) (x : ℝ) :
    Integrable (fun y : ℝ × ℝ =>
      (bumpContinuousBasis.toProbabilityDensityBasis.density i y.1 *
          bumpContinuousBasis.toProbabilityDensityBasis.density j y.2) •
        meanShiftInteractionKernel (gaussianKernel σ) x y.1 y.2)
      (bumpReference.prod bumpReference) := by
  have key : ∀ di dj : ℝ → ℝ, Continuous di → Continuous dj →
      HasCompactSupport di → HasCompactSupport dj →
      Integrable (fun y : ℝ × ℝ => (di y.1 * dj y.2) •
          meanShiftInteractionKernel (gaussianKernel σ) x y.1 y.2)
        (bumpReference.prod bumpReference) := by
    intro di dj hdi hdj hcdi hcdj
    have hcont : Continuous (fun y : ℝ × ℝ => (di y.1 * dj y.2) •
        meanShiftInteractionKernel (gaussianKernel σ) x y.1 y.2) :=
      ((hdi.comp continuous_fst).mul (hdj.comp continuous_snd)).smul
        (continuous_meanShiftKernel_bump σ x)
    have hcs : HasCompactSupport (fun y : ℝ × ℝ => (di y.1 * dj y.2) •
        meanShiftInteractionKernel (gaussianKernel σ) x y.1 y.2) := by
      apply HasCompactSupport.intro (K := tsupport di ×ˢ tsupport dj)
        (IsCompact.prod hcdi hcdj)
      intro z hz
      rw [Set.mem_prod, not_and_or] at hz
      rcases hz with h | h
      · rw [image_eq_zero_of_notMem_tsupport h]; simp
      · rw [image_eq_zero_of_notMem_tsupport h]; simp
    exact hcont.integrable_of_hasCompactSupport hcs
  exact key _ _ (bumpContinuousBasis.continuous_density i)
    (bumpContinuousBasis.continuous_density j)
    (bumpDensity_hasCompactSupport i) (bumpDensity_hasCompactSupport j)

/-- **Complete population setup for the smooth continuum bump basis.**  A single
probe suffices; the frame bound is discharged directly by the ordered-support
sign argument, not by a supplied hypothesis. The sign lemma itself only uses
kernel positivity; the promoted setup still requires the paper's intended
positive Gaussian bandwidth. -/
noncomputable def bumpGaussianSetup
    (_hσ : ValidBandwidth σ) (a b : FiniteProbabilityVector 2) (x0 : ℝ) :
    PopulationMeanShiftFiniteSetup ℝ 2 1 where
  reference := bumpReference
  refProb := inferInstance
  basis := bumpContinuousBasis.toProbabilityDensityBasis
  kernel := gaussianKernel σ
  probes := fun _ => x0
  a := a
  b := b
  meanShiftRegular _ := bumpGaussian_meanShiftRegular σ a b x0
  interactionIntegrable _ := bumpBasis_integrable_interaction σ a b x0
  basisInteractionIntegrable i j _ := bumpBasis_integrable_basisInteraction σ i j x0
  frameConstant :=
    ‖inducedInteractionVector bumpReference
      (meanShiftInteractionKernel (gaussianKernel σ))
      bumpContinuousBasis.toProbabilityDensityBasis.density
      (fun _ : Fin 1 => x0) 0 1‖
  frameBound := bumpInteractionFrameBound σ x0

/-- **Concrete continuum-supported smooth-basis identifiability (Objective 3).**
For the non-atomic two-component bump model on the continuum, zero
finite normalized population drift energy at the single probe forces equality of
the represented probability measures. -/
theorem bumpGaussian_identifies_of_probeEnergy_eq_zero
    (hσ : ValidBandwidth σ) (a b : FiniteProbabilityVector 2) (x0 : ℝ)
    (henergy :
      (bumpGaussianSetup σ hσ a b x0).normalizedProbeDriftEnergy = 0) :
    bumpContinuousBasis.toProbabilityDensityBasis.basisMeasure a =
      bumpContinuousBasis.toProbabilityDensityBasis.basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero
    (bumpGaussianSetup σ hσ a b x0) henergy

end BumpSetup

end PaperFiniteIdentifiability
end DriftingIdentifiability
