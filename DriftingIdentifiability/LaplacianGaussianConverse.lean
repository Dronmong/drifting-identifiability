import DriftingIdentifiability.TrustedBoundary
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.Distributions.Gaussian.Fernique
import Mathlib.Probability.Moments.Tilted
import Mathlib.MeasureTheory.Integral.DominatedConvergence

open MeasureTheory Filter Topology ProbabilityTheory
open scoped RealInnerProductSpace

/-!
# Laplacian-kernel converse for Gaussian targets

This module develops the analytic foundation for the second converse described
in the paper authors' rebuttal.  Along a radial probe `r • u`, the common
Laplace-kernel factor `exp (-r / τ)` is removed exactly.  Dominated convergence
then identifies the limiting kernel centroid with an exponential tilt.

The present first stage is distribution-generic.  The subsequent stage will
specialize the tilt to a multivariate Gaussian and recover its mean and
covariance from the limiting raw drift.  No identifiability statement is
assumed here.
-/

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

lemma radial_identity (u y : E) (hu : ‖u‖ = 1) {r : ℝ} (hr : 0 < r) :
    r - ‖r • u - y‖ =
      (2 * ⟪u, y⟫ - ‖y‖ ^ 2 / r) /
        (1 + ‖u - r⁻¹ • y‖) := by
  have hr0 : r ≠ 0 := ne_of_gt hr
  have hvec : r • u - y = r • (u - r⁻¹ • y) := by
    rw [smul_sub, smul_smul, mul_inv_cancel₀ hr0, one_smul]
  have hnorm : ‖r • u - y‖ = r * ‖u - r⁻¹ • y‖ := by
    rw [hvec, norm_smul, Real.norm_eq_abs, abs_of_pos hr]
  have hsq :
      ‖u - r⁻¹ • y‖ ^ 2 =
        1 - 2 * r⁻¹ * ⟪u, y⟫ + (r⁻¹) ^ 2 * ‖y‖ ^ 2 := by
    rw [← real_inner_self_eq_norm_sq]
    simp only [inner_sub_left, inner_sub_right, real_inner_smul_left,
      real_inner_smul_right]
    rw [real_inner_comm y u, real_inner_self_eq_norm_sq,
      real_inner_self_eq_norm_sq, hu]
    ring
  rw [hnorm]
  have hden : 1 + ‖u - r⁻¹ • y‖ ≠ 0 := by positivity
  apply (eq_div_iff hden).2
  calc
    (r - r * ‖u - r⁻¹ • y‖) * (1 + ‖u - r⁻¹ • y‖) =
        r * (1 - ‖u - r⁻¹ • y‖ ^ 2) := by ring
    _ = 2 * ⟪u, y⟫ - ‖y‖ ^ 2 / r := by
      rw [hsq]
      field_simp [hr0]
      ring

lemma radial_tendsto (u y : E) (hu : ‖u‖ = 1) :
    Tendsto (fun r : ℝ => r - ‖r • u - y‖) atTop (𝓝 ⟪u, y⟫) := by
  have hinv : Tendsto (fun r : ℝ => r⁻¹) atTop (𝓝 0) :=
    tendsto_inv_atTop_zero
  have hvec :
      Tendsto (fun r : ℝ => u - r⁻¹ • y) atTop (𝓝 u) := by
    simpa using tendsto_const_nhds.sub (hinv.smul_const y)
  have hnorm :
      Tendsto (fun r : ℝ => ‖u - r⁻¹ • y‖) atTop (𝓝 1) := by
    change Tendsto
      ((fun z : E => ‖z‖) ∘ (fun r : ℝ => u - r⁻¹ • y))
      atTop (𝓝 1)
    simpa [hu] using (continuous_norm.tendsto u).comp hvec
  have hsmall :
      Tendsto (fun r : ℝ => ‖y‖ ^ 2 / r) atTop (𝓝 0) := by
    simpa [div_eq_mul_inv] using
      (tendsto_const_nhds.mul hinv : Tendsto
        (fun r : ℝ => ‖y‖ ^ 2 * r⁻¹) atTop (𝓝 (‖y‖ ^ 2 * 0)))
  have hnum :
      Tendsto (fun r : ℝ => 2 * ⟪u, y⟫ - ‖y‖ ^ 2 / r) atTop
        (𝓝 (2 * ⟪u, y⟫)) := by
    simpa using tendsto_const_nhds.sub hsmall
  have hden :
      Tendsto (fun r : ℝ => 1 + ‖u - r⁻¹ • y‖) atTop (𝓝 2) := by
    convert
      ((tendsto_const_nhds :
        Tendsto (fun _ : ℝ => (1 : ℝ)) atTop (𝓝 1)).add hnorm) using 1
    all_goals norm_num
  have hquot :
      Tendsto
        (fun r : ℝ =>
          (2 * ⟪u, y⟫ - ‖y‖ ^ 2 / r) /
            (1 + ‖u - r⁻¹ • y‖))
        atTop (𝓝 ⟪u, y⟫) := by
    change Tendsto
      ((fun r : ℝ => 2 * ⟪u, y⟫ - ‖y‖ ^ 2 / r) /
        (fun r : ℝ => 1 + ‖u - r⁻¹ • y‖))
      atTop (𝓝 ⟪u, y⟫)
    simpa using hnum.div hden (by norm_num : (2 : ℝ) ≠ 0)
  apply hquot.congr'
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with r hr
  exact (radial_identity u y hu hr).symm

noncomputable def laplaceCompensatedWeight
    (τ r : ℝ) (u y : E) : ℝ :=
  Real.exp ((r - ‖r • u - y‖) / τ)

noncomputable def exponentialTiltWeight
    (τ : ℝ) (u y : E) : ℝ :=
  Real.exp (⟪u, y⟫ / τ)

noncomputable def laplaceCompensatedCentroid
    [MeasurableSpace E] (P : Measure E)
    (τ r : ℝ) (u : E) : E :=
  (∫ y, laplaceCompensatedWeight τ r u y ∂P)⁻¹ •
    ∫ y, laplaceCompensatedWeight τ r u y • y ∂P

noncomputable def exponentialTiltCentroid
    [MeasurableSpace E] (P : Measure E)
    (τ : ℝ) (u : E) : E :=
  (∫ y, exponentialTiltWeight τ u y ∂P)⁻¹ •
    ∫ y, exponentialTiltWeight τ u y • y ∂P

noncomputable def kernelCentroid
    [MeasurableSpace E] (k : E → E → ℝ)
    (P : Measure E) (x : E) : E :=
  (∫ y, k x y ∂P)⁻¹ • ∫ y, k x y • y ∂P

lemma laplaceCompensatedWeight_tendsto
    (τ : ℝ) (u y : E) (hu : ‖u‖ = 1) :
    Tendsto (fun r : ℝ => laplaceCompensatedWeight τ r u y)
      atTop (𝓝 (exponentialTiltWeight τ u y)) := by
  unfold laplaceCompensatedWeight exponentialTiltWeight
  exact Real.continuous_exp.continuousAt.tendsto.comp
    ((radial_tendsto u y hu).div_const τ)

lemma exp_mul_laplaceKernel_eq_compensated
    (τ r : ℝ) (u y : E) :
    Real.exp (r / τ) * laplaceKernel τ (r • u) y =
      laplaceCompensatedWeight τ r u y := by
  unfold laplaceKernel laplaceCompensatedWeight
  rw [← Real.exp_add]
  congr 1
  ring

lemma radialExponent_le_norm
    (u y : E) (hu : ‖u‖ = 1) {r : ℝ} (hr : 0 ≤ r) :
    r - ‖r • u - y‖ ≤ ‖y‖ := by
  have hru : ‖r • u‖ = r := by
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg hr, hu, mul_one]
  have htri : ‖r • u‖ ≤ ‖r • u - y‖ + ‖y‖ := by
    calc
      ‖r • u‖ = ‖(r • u - y) + y‖ := by rw [sub_add_cancel]
      _ ≤ ‖r • u - y‖ + ‖y‖ := norm_add_le _ _
  rw [hru] at htri
  linarith

lemma laplaceCompensatedWeight_le
    (τ : ℝ) (hτ : 0 < τ) (u y : E) (hu : ‖u‖ = 1)
    {r : ℝ} (hr : 0 ≤ r) :
    laplaceCompensatedWeight τ r u y ≤ Real.exp (‖y‖ / τ) := by
  unfold laplaceCompensatedWeight
  exact Real.exp_le_exp.mpr
    (div_le_div_of_nonneg_right (radialExponent_le_norm u y hu hr) hτ.le)

lemma norm_laplaceCompensatedWeight_smul_le
    (τ : ℝ) (hτ : 0 < τ) (u y : E) (hu : ‖u‖ = 1)
    {r : ℝ} (hr : 0 ≤ r) :
    ‖laplaceCompensatedWeight τ r u y • y‖ ≤
      Real.exp (‖y‖ / τ) * ‖y‖ := by
  unfold laplaceCompensatedWeight
  rw [norm_smul, Real.norm_eq_abs,
    abs_of_pos (Real.exp_pos ((r - ‖r • u - y‖) / τ))]
  exact mul_le_mul_of_nonneg_right
    (laplaceCompensatedWeight_le τ hτ u y hu hr) (norm_nonneg y)

section IntegralLimits

variable [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
  [SecondCountableTopology E]

omit [CompleteSpace E] [SecondCountableTopology E] in
lemma integral_laplaceCompensatedWeight_tendsto
    (P : Measure E) (τ : ℝ) (hτ : 0 < τ)
    (u : E) (hu : ‖u‖ = 1)
    (hdom : Integrable (fun y : E => Real.exp (‖y‖ / τ)) P) :
    Tendsto
      (fun r : ℝ => ∫ y, laplaceCompensatedWeight τ r u y ∂P)
      atTop
      (𝓝 (∫ y, exponentialTiltWeight τ u y ∂P)) := by
  apply tendsto_integral_filter_of_dominated_convergence
    (bound := fun y : E => Real.exp (‖y‖ / τ))
  · filter_upwards with r
    apply Continuous.aestronglyMeasurable
    unfold laplaceCompensatedWeight
    fun_prop
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with r hr
    filter_upwards with y
    rw [Real.norm_eq_abs]
    unfold laplaceCompensatedWeight
    rw [abs_of_pos (Real.exp_pos _)]
    exact laplaceCompensatedWeight_le τ hτ u y hu hr
  · exact hdom
  · filter_upwards with y
    exact laplaceCompensatedWeight_tendsto τ u y hu

omit [CompleteSpace E] in
lemma integral_laplaceCompensatedWeight_smul_tendsto
    (P : Measure E) (τ : ℝ) (hτ : 0 < τ)
    (u : E) (hu : ‖u‖ = 1)
    (hdom :
      Integrable (fun y : E => Real.exp (‖y‖ / τ) * ‖y‖) P) :
    Tendsto
      (fun r : ℝ =>
        ∫ y, laplaceCompensatedWeight τ r u y • y ∂P)
      atTop
      (𝓝 (∫ y, exponentialTiltWeight τ u y • y ∂P)) := by
  apply tendsto_integral_filter_of_dominated_convergence
    (bound := fun y : E => Real.exp (‖y‖ / τ) * ‖y‖)
  · filter_upwards with r
    apply Continuous.aestronglyMeasurable
    unfold laplaceCompensatedWeight
    fun_prop
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with r hr
    filter_upwards with y
    exact norm_laplaceCompensatedWeight_smul_le τ hτ u y hu hr
  · exact hdom
  · filter_upwards with y
    exact (laplaceCompensatedWeight_tendsto τ u y hu).smul_const y

omit [CompleteSpace E] [SecondCountableTopology E] in
lemma exponentialTiltWeight_integrable
    (P : Measure E) (τ : ℝ) (hτ : 0 < τ)
    (u : E) (hu : ‖u‖ = 1)
    (hdom : Integrable (fun y : E => Real.exp (‖y‖ / τ)) P) :
    Integrable (exponentialTiltWeight τ u) P := by
  refine hdom.mono' ?_ ?_
  · apply Continuous.aestronglyMeasurable
    unfold exponentialTiltWeight
    fun_prop
  · filter_upwards with y
    rw [Real.norm_eq_abs]
    unfold exponentialTiltWeight
    rw [abs_of_pos (Real.exp_pos _)]
    apply Real.exp_le_exp.mpr
    apply div_le_div_of_nonneg_right _ hτ.le
    simpa [hu] using real_inner_le_norm u y

omit [CompleteSpace E] in
lemma laplaceCompensatedCentroid_tendsto
    (P : Measure E) [IsProbabilityMeasure P]
    (τ : ℝ) (hτ : 0 < τ) (u : E) (hu : ‖u‖ = 1)
    (hdom₀ : Integrable (fun y : E => Real.exp (‖y‖ / τ)) P)
    (hdom₁ :
      Integrable (fun y : E => Real.exp (‖y‖ / τ) * ‖y‖) P) :
    Tendsto (fun r : ℝ => laplaceCompensatedCentroid P τ r u)
      atTop (𝓝 (exponentialTiltCentroid P τ u)) := by
  have hmass :=
    integral_laplaceCompensatedWeight_tendsto P τ hτ u hu hdom₀
  have hmoment :=
    integral_laplaceCompensatedWeight_smul_tendsto P τ hτ u hu hdom₁
  have htilt :
      Integrable (exponentialTiltWeight τ u) P :=
    exponentialTiltWeight_integrable P τ hτ u hu hdom₀
  have hpos :
      0 < ∫ y, exponentialTiltWeight τ u y ∂P := by
    unfold exponentialTiltWeight
    exact integral_exp_pos htilt
  unfold laplaceCompensatedCentroid exponentialTiltCentroid
  exact (hmass.inv₀ (ne_of_gt hpos)).smul hmoment

omit [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E] in
lemma kernelCentroid_laplace_eq_compensated
    (P : Measure E) (τ r : ℝ) (u : E) :
    kernelCentroid (laplaceKernel τ) P (r • u) =
      laplaceCompensatedCentroid P τ r u := by
  let c : ℝ := Real.exp (r / τ)
  have hc : c ≠ 0 := ne_of_gt (Real.exp_pos _)
  have hmass :
      ∫ y, laplaceCompensatedWeight τ r u y ∂P =
        c * ∫ y, laplaceKernel τ (r • u) y ∂P := by
    rw [← integral_const_mul]
    congr 1
    funext y
    exact (exp_mul_laplaceKernel_eq_compensated τ r u y).symm
  have hmoment :
      ∫ y, laplaceCompensatedWeight τ r u y • y ∂P =
        c • ∫ y, laplaceKernel τ (r • u) y • y ∂P := by
    rw [← integral_smul]
    congr 1
    funext y
    rw [smul_smul]
    congr 1
    exact (exp_mul_laplaceKernel_eq_compensated τ r u y).symm
  unfold kernelCentroid laplaceCompensatedCentroid
  rw [hmass, hmoment]
  rw [smul_smul]
  have hscalar :
      (c * ∫ y, laplaceKernel τ (r • u) y ∂P)⁻¹ * c =
        (∫ y, laplaceKernel τ (r • u) y ∂P)⁻¹ := by
    rw [mul_inv]
    field_simp
  rw [hscalar]

omit [CompleteSpace E] in
lemma kernelCentroid_laplace_radial_tendsto
    (P : Measure E) [IsProbabilityMeasure P]
    (τ : ℝ) (hτ : 0 < τ) (u : E) (hu : ‖u‖ = 1)
    (hdom₀ : Integrable (fun y : E => Real.exp (‖y‖ / τ)) P)
    (hdom₁ :
      Integrable (fun y : E => Real.exp (‖y‖ / τ) * ‖y‖) P) :
    Tendsto
      (fun r : ℝ =>
        kernelCentroid (laplaceKernel τ) P (r • u))
      atTop (𝓝 (exponentialTiltCentroid P τ u)) := by
  simpa only [kernelCentroid_laplace_eq_compensated] using
    laplaceCompensatedCentroid_tendsto P τ hτ u hu hdom₀ hdom₁

omit [BorelSpace E] [SecondCountableTopology E] in
lemma meanShift_eq_kernelCentroid_sub
    (P : Measure E) [IsProbabilityMeasure P]
    (k : E → E → ℝ) (x : E)
    (hk : Integrable (fun y => k x y) P)
    (hky : Integrable (fun y => k x y • y) P)
    (hz : (∫ y, k x y ∂P) ≠ 0) :
    meanShift k P x =
      kernelCentroid k P x - x := by
  have hkx : Integrable (fun y => k x y • x) P :=
    hk.smul_const x
  have hint :
      (∫ y, k x y • (y - x) ∂P) =
        (∫ y, k x y • y ∂P) - (∫ y, k x y ∂P) • x := by
    rw [← integral_smul_const]
    rw [← integral_sub hky hkx]
    congr 1
    funext y
    exact smul_sub (k x y) y x
  unfold meanShift kernelNormalizer kernelCentroid
  rw [hint, smul_sub, smul_smul, inv_mul_cancel₀ hz, one_smul]

section GaussianDomination

/-- Fernique's theorem implies integrability of every linear exponential of
the norm of a Gaussian random vector. -/
lemma gaussian_integrable_exp_norm
    (P : Measure E) [IsGaussian P] (b : ℝ) :
    Integrable (fun y : E => Real.exp (b * ‖y‖)) P := by
  obtain ⟨C, hC, hfern⟩ := IsGaussian.exists_integrable_exp_sq P
  let K : ℝ := Real.exp (C⁻¹ * (b / 2) ^ 2)
  refine (hfern.const_mul K).mono' ?_ ?_
  · apply Continuous.aestronglyMeasurable
    fun_prop
  · filter_upwards with y
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    change Real.exp (b * ‖y‖) ≤
      Real.exp (C⁻¹ * (b / 2) ^ 2) *
        Real.exp (C * ‖y‖ ^ 2)
    rw [← Real.exp_add]
    apply Real.exp_le_exp.mpr
    have hyoung :=
      two_mul_le_add_mul_sq (a := ‖y‖) (b := b / 2) hC
    nlinarith

/-- The extra factor `‖y‖` needed for the first moment is also absorbed by
Fernique domination. -/
lemma gaussian_integrable_exp_norm_mul_norm
    (P : Measure E) [IsGaussian P] (b : ℝ) :
    Integrable (fun y : E => Real.exp (b * ‖y‖) * ‖y‖) P := by
  have hbase := gaussian_integrable_exp_norm P (b + 1)
  refine hbase.mono' ?_ ?_
  · apply Continuous.aestronglyMeasurable
    fun_prop
  · filter_upwards with y
    rw [Real.norm_eq_abs, abs_of_nonneg
      (mul_nonneg (Real.exp_pos _).le (norm_nonneg y))]
    have hnorm_le_exp : ‖y‖ ≤ Real.exp ‖y‖ := by
      linarith [Real.add_one_le_exp ‖y‖]
    calc
      Real.exp (b * ‖y‖) * ‖y‖ ≤
          Real.exp (b * ‖y‖) * Real.exp ‖y‖ :=
        mul_le_mul_of_nonneg_left hnorm_le_exp (Real.exp_pos _).le
      _ = Real.exp ((b + 1) * ‖y‖) := by
        rw [← Real.exp_add]
        congr 1
        ring

/-- Every real linear exponential is integrable under a Gaussian law. -/
lemma gaussian_integrable_exp_inner
    (P : Measure E) [IsGaussian P] (a : E) :
    Integrable (fun y : E => Real.exp ⟪a, y⟫) P := by
  refine (gaussian_integrable_exp_norm P ‖a‖).mono' ?_ ?_
  · apply Continuous.aestronglyMeasurable
    fun_prop
  · filter_upwards with y
    rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact Real.exp_le_exp.mpr (real_inner_le_norm a y)

/-- The tilted first-moment integrand is integrable under a Gaussian law. -/
lemma gaussian_integrable_exp_inner_smul_id
    (P : Measure E) [IsGaussian P] (a : E) :
    Integrable (fun y : E => Real.exp ⟪a, y⟫ • y) P := by
  refine (gaussian_integrable_exp_norm_mul_norm P ‖a‖).mono' ?_ ?_
  · apply Continuous.aestronglyMeasurable
    fun_prop
  · filter_upwards with y
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    exact mul_le_mul_of_nonneg_right
      (Real.exp_le_exp.mpr (real_inner_le_norm a y)) (norm_nonneg y)

/-- The exact Gaussian normalizer for an exponential linear tilt.  This is the
scalar Gaussian MGF transported through `IsGaussian.map_eq_gaussianReal`. -/
lemma gaussian_integral_exp_inner
    (P : Measure E) [IsGaussian P] (a : E) :
    (∫ y, Real.exp ⟪a, y⟫ ∂P) =
      Real.exp (⟪a, ∫ y, y ∂P⟫ + covarianceBilin P a a / 2) := by
  let L : StrongDual ℝ E := (InnerProductSpace.toDualMap ℝ E) a
  have hmap :
      P.map L =
        gaussianReal (∫ x, L x ∂P) (Var[L; P]).toNNReal :=
    IsGaussian.map_eq_gaussianReal L
  have hmgf := mgf_gaussianReal (p := P) (X := L) hmap (1 : ℝ)
  rw [mgf] at hmgf
  simp only [one_mul] at hmgf
  calc
    (∫ y, Real.exp ⟪a, y⟫ ∂P) = ∫ y, Real.exp (L y) ∂P := by
      congr with y
    _ = Real.exp (⟪a, ∫ y, y ∂P⟫ + covarianceBilin P a a / 2) := by
      rw [hmgf]
      congr 1
      have hmean : (∫ x, L x ∂P) = ⟪a, ∫ y, y ∂P⟫ := by
        simpa [L] using
          (ContinuousLinearMap.integral_comp_id_comm
            (μ := P) (h := IsGaussian.integrable_id (μ := P)) L)
      have hvarReal : Var[L; P] = covarianceBilin P a a := by
        dsimp [L]
        change Var[fun u : E => ⟪a, u⟫; P] = covarianceBilin P a a
        exact (covarianceBilin_self (μ := P)
          (h := IsGaussian.memLp_two_id (μ := P)) a).symm
      have hvar : ((Var[L; P]).toNNReal : ℝ) = covarianceBilin P a a := by
        rw [← hvarReal, Real.coe_toNNReal']
        exact sup_eq_left.mpr (variance_nonneg L P)
      rw [hmean, hvar]
      ring

/-- After tilting a Gaussian by `exp ⟪a, ·⟫`, every one-dimensional
projection still has exponential moments of all orders. -/
lemma gaussian_tilted_integrableExpSet_eq_univ
    (P : Measure E) [IsGaussian P] (a b : E) :
    integrableExpSet (fun y : E => ⟪b, y⟫)
      (P.tilted fun y : E => ⟪a, y⟫) = Set.univ := by
  ext t
  simp only [integrableExpSet, Set.mem_setOf_eq, Set.mem_univ]
  rw [MeasureTheory.integrable_tilted_iff
    (gaussian_integrable_exp_inner P a)]
  constructor
  · intro _
    trivial
  · intro _
    convert gaussian_integrable_exp_inner P (a + t • b) using 1
    ext y
    simp [inner_add_left, real_inner_smul_left, Real.exp_add, smul_eq_mul]

/-- The cumulant-generating function of a one-dimensional projection under a
linearly tilted Gaussian. -/
lemma gaussian_tilted_cgf_inner
    (P : Measure E) [IsGaussian P] (a b : E) (t : ℝ) :
    cgf (fun y : E => ⟪b, y⟫)
        (P.tilted fun y : E => ⟪a, y⟫) t =
      (⟪a + t • b, ∫ y, y ∂P⟫ +
          covarianceBilin P (a + t • b) (a + t • b) / 2) -
        (⟪a, ∫ y, y ∂P⟫ + covarianceBilin P a a / 2) := by
  rw [cgf, mgf]
  rw [MeasureTheory.integral_exp_tilted]
  have hfun :
      (fun y : E => ⟪a, y⟫) + (fun y : E => t * ⟪b, y⟫)
        = fun y : E => ⟪a + t • b, y⟫ := by
    funext y
    simp [inner_add_left, real_inner_smul_left]
  rw [hfun]
  rw [gaussian_integral_exp_inner P (a + t • b),
    gaussian_integral_exp_inner P a]
  rw [Real.log_div (Real.exp_ne_zero _) (Real.exp_ne_zero _)]
  simp only [Real.log_exp]

omit [CompleteSpace E] [SecondCountableTopology E] in
lemma deriv_gaussian_tilted_cgf_expr
    (P : Measure E) (a b : E) :
    deriv
      (fun t : ℝ =>
        (⟪a + t • b, ∫ y, y ∂P⟫ +
            covarianceBilin P (a + t • b) (a + t • b) / 2) -
          (⟪a, ∫ y, y ∂P⟫ + covarianceBilin P a a / 2))
      0 =
      ⟪b, ∫ y, y ∂P⟫ + covarianceBilin P b a := by
  have hfun :
      (fun t : ℝ =>
        (⟪a + t • b, ∫ y, y ∂P⟫ +
            covarianceBilin P (a + t • b) (a + t • b) / 2) -
          (⟪a, ∫ y, y ∂P⟫ + covarianceBilin P a a / 2))
      =
      (fun t : ℝ =>
        t * (⟪b, ∫ y, y ∂P⟫ + covarianceBilin P b a) +
          (t ^ 2 / 2) * covarianceBilin P b b) := by
    funext t
    simp [inner_add_left, real_inner_smul_left, map_add, map_smul,
      smul_eq_mul, covarianceBilin_comm (μ := P) a b]
    ring
  rw [hfun]
  apply HasDerivAt.deriv
  convert
    (((hasDerivAt_id (0 : ℝ)).mul_const
        (⟪b, ∫ y, y ∂P⟫ + covarianceBilin P b a)).add
      ((((hasDerivAt_id (0 : ℝ)).pow 2).div_const 2).mul_const
        (covarianceBilin P b b))) using 1
  · ext t
    simp only [Pi.add_apply, Pi.pow_apply, id_eq]
  · simp only [id_eq]
    ring_nf

/-- The mean of a Gaussian after a linear exponential tilt, tested against an
arbitrary vector. -/
lemma gaussian_tilted_integral_inner
    (P : Measure E) [IsGaussian P] (a b : E) :
    (∫ y, ⟪b, y⟫ ∂(P.tilted fun y : E => ⟪a, y⟫)) =
      ⟪b, ∫ y, y ∂P⟫ + covarianceBilin P b a := by
  let f : E → ℝ := fun y => ⟪a, y⟫
  let X : E → ℝ := fun y => ⟪b, y⟫
  have hf : Integrable (fun y : E => Real.exp (f y)) P := by
    simpa [f] using gaussian_integrable_exp_inner P a
  haveI : IsProbabilityMeasure (P.tilted f) :=
    MeasureTheory.isProbabilityMeasure_tilted hf
  have hset :
      integrableExpSet X (P.tilted f) = Set.univ := by
    simpa [X, f] using gaussian_tilted_integrableExpSet_eq_univ P a b
  have ht0 : (0 : ℝ) ∈ interior (integrableExpSet X (P.tilted f)) := by
    rw [hset, interior_univ]
    trivial
  have hmain :=
    integral_tilted_mul_self (μ := P.tilted f) (X := X) (t := 0) ht0
  have hcgf :
      (fun t : ℝ => cgf X (P.tilted f) t) =
      fun t : ℝ =>
        (⟪a + t • b, ∫ y, y ∂P⟫ +
            covarianceBilin P (a + t • b) (a + t • b) / 2) -
          (⟪a, ∫ y, y ∂P⟫ + covarianceBilin P a a / 2) := by
    funext t
    simpa [X, f] using gaussian_tilted_cgf_inner P a b t
  calc
    (∫ y, ⟪b, y⟫ ∂(P.tilted fun y : E => ⟪a, y⟫)) =
        (∫ y, X y ∂P.tilted f) := by rfl
    _ = deriv (cgf X (P.tilted f)) 0 := by
      simpa [X] using hmain
    _ = deriv
      (fun t : ℝ =>
        (⟪a + t • b, ∫ y, y ∂P⟫ +
            covarianceBilin P (a + t • b) (a + t • b) / 2) -
          (⟪a, ∫ y, y ∂P⟫ + covarianceBilin P a a / 2))
      0 := by
        change deriv (fun t : ℝ => cgf X (P.tilted f) t) 0 = _
        rw [hcgf]
    _ = ⟪b, ∫ y, y ∂P⟫ + covarianceBilin P b a :=
      deriv_gaussian_tilted_cgf_expr P a b

omit [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E] in
/-- The paper-facing tilted centroid is the mean under mathlib's exponentially
tilted measure. -/
lemma exponentialTiltCentroid_eq_tilted_mean
    (P : Measure E) (τ : ℝ) (u : E) :
    exponentialTiltCentroid P τ u =
      ∫ y, y ∂(P.tilted fun y : E => ⟪(1 / τ) • u, y⟫) := by
  let a : E := (1 / τ) • u
  have hweight :
      (fun y : E => exponentialTiltWeight τ u y) =
        fun y : E => Real.exp ⟪a, y⟫ := by
    funext y
    simp [exponentialTiltWeight, a, real_inner_smul_left, div_eq_mul_inv,
      mul_comm]
  have hden :
      (∫ y, exponentialTiltWeight τ u y ∂P) =
        ∫ y, Real.exp ⟪a, y⟫ ∂P := by
    rw [hweight]
  have hnum :
      (∫ y, exponentialTiltWeight τ u y • y ∂P) =
        ∫ y, Real.exp ⟪a, y⟫ • y ∂P := by
    congr with y
    rw [congrFun hweight y]
  unfold exponentialTiltCentroid
  rw [hden, hnum]
  rw [MeasureTheory.integral_tilted]
  rw [← integral_smul]
  congr with y
  rw [smul_smul]
  congr 1
  rw [div_eq_mul_inv, mul_comm]

/-- Weak form of the Gaussian exponential-tilt centroid identity.  This avoids
introducing an abstract covariance operator; the multivariate specialization
below converts the bilinear covariance into `S *ᵥ u`. -/
lemma gaussian_inner_exponentialTiltCentroid
    (P : Measure E) [IsGaussian P] (τ : ℝ) (_hτ : 0 < τ) (u b : E) :
    ⟪b, exponentialTiltCentroid P τ u⟫ =
      ⟪b, ∫ y, y ∂P⟫ + (1 / τ) * covarianceBilin P b u := by
  rw [exponentialTiltCentroid_eq_tilted_mean P τ u]
  let a : E := (1 / τ) • u
  have hf : Integrable (fun y : E => Real.exp ⟪a, y⟫) P :=
    gaussian_integrable_exp_inner P a
  have hIntTilt :
      Integrable (id : E → E)
        (P.tilted fun y : E => ⟪a, y⟫) := by
    rw [MeasureTheory.integrable_tilted_iff hf]
    simpa [a] using gaussian_integrable_exp_inner_smul_id P a
  change
    ⟪b, ∫ y, id y ∂(P.tilted fun y : E => ⟪a, y⟫)⟫ =
      ⟪b, ∫ y, y ∂P⟫ + (1 / τ) * covarianceBilin P b u
  rw [← integral_inner hIntTilt b]
  change
    (∫ y, ⟪b, y⟫ ∂(P.tilted fun y : E => ⟪a, y⟫)) =
      ⟪b, ∫ y, y ∂P⟫ + (1 / τ) * covarianceBilin P b u
  rw [gaussian_tilted_integral_inner P a b]
  simp [a, map_smul]

section MultivariateTilt

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- For a multivariate Gaussian, the exponential tilt centroid has the expected
closed form `μ + τ⁻¹ S u`. -/
theorem multivariateGaussian_exponentialTiltCentroid
    (μ : EuclideanSpace ℝ ι) (S : Matrix ι ι ℝ) (hS : S.PosSemidef)
    (τ : ℝ) (hτ : 0 < τ) (u : EuclideanSpace ℝ ι) :
    exponentialTiltCentroid (multivariateGaussian μ S) τ u =
      μ + (1 / τ) • (((Matrix.toEuclideanCLM (n := ι) (𝕜 := ℝ)) S) u) := by
  apply ext_inner_left ℝ
  intro b
  rw [gaussian_inner_exponentialTiltCentroid
    (P := multivariateGaussian μ S) τ hτ u b]
  rw [integral_id_multivariateGaussian,
    covarianceBilin_multivariateGaussian hS]
  rw [inner_add_right, real_inner_smul_right, Matrix.inner_toEuclideanCLM]

end MultivariateTilt

/-- The scalar dominator required by the compensated-normalizer limit is
automatic for every Gaussian law and every positive bandwidth. -/
lemma gaussian_integrable_laplace_scalar_dominator
    (P : Measure E) [IsGaussian P] (τ : ℝ) :
    Integrable (fun y : E => Real.exp (‖y‖ / τ)) P := by
  simpa [div_eq_mul_inv, mul_comm] using
    gaussian_integrable_exp_norm P τ⁻¹

/-- The vector first-moment dominator required by the radial centroid limit is
automatic for every Gaussian law and every positive bandwidth. -/
lemma gaussian_integrable_laplace_vector_dominator
    (P : Measure E) [IsGaussian P] (τ : ℝ) :
    Integrable (fun y : E => Real.exp (‖y‖ / τ) * ‖y‖) P := by
  simpa [div_eq_mul_inv, mul_comm] using
    gaussian_integrable_exp_norm_mul_norm P τ⁻¹

/-- A Gaussian law satisfies the complete generic radial-centroid limit. -/
theorem gaussian_kernelCentroid_laplace_radial_tendsto
    (P : Measure E) [IsGaussian P] [IsProbabilityMeasure P]
    (τ : ℝ) (hτ : 0 < τ) (u : E) (hu : ‖u‖ = 1) :
    Tendsto
      (fun r : ℝ => kernelCentroid (laplaceKernel τ) P (r • u))
      atTop (𝓝 (exponentialTiltCentroid P τ u)) :=
  kernelCentroid_laplace_radial_tendsto P τ hτ u hu
    (gaussian_integrable_laplace_scalar_dominator P τ)
    (gaussian_integrable_laplace_vector_dominator P τ)

omit [InnerProductSpace ℝ E] [CompleteSpace E]
  [SecondCountableTopology E] in
lemma laplaceKernel_integrable
    (P : Measure E) [IsFiniteMeasure P]
    (τ : ℝ) (hτ : 0 < τ) (x : E) :
    Integrable (fun y => laplaceKernel τ x y) P := by
  refine Integrable.of_bound ?_ 1 ?_
  · apply Continuous.aestronglyMeasurable
    unfold laplaceKernel
    fun_prop
  · filter_upwards with y
    rw [Real.norm_eq_abs]
    unfold laplaceKernel
    rw [abs_of_pos (Real.exp_pos _), Real.exp_le_one_iff]
    have hinv : 0 ≤ (1 / τ) := by positivity
    exact mul_nonpos_of_nonpos_of_nonneg
      (neg_nonpos.mpr hinv) (norm_nonneg (x - y))

lemma laplaceKernel_smul_id_integrable
    (P : Measure E) [IsGaussian P]
    (τ : ℝ) (hτ : 0 < τ) (x : E) :
    Integrable (fun y => laplaceKernel τ x y • y) P := by
  have hid : Integrable (id : E → E) P :=
    IsGaussian.integrable_id (μ := P)
  refine hid.norm.mono'
    (f := fun y : E => laplaceKernel τ x y • y) ?_ ?_
  · apply Continuous.aestronglyMeasurable
    unfold laplaceKernel
    fun_prop
  · filter_upwards with y
    rw [norm_smul, Real.norm_eq_abs]
    unfold laplaceKernel
    rw [abs_of_pos (Real.exp_pos _)]
    have hk_le : Real.exp (-(1 / τ) * ‖x - y‖) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      have hinv : 0 ≤ (1 / τ) := by positivity
      exact mul_nonpos_of_nonpos_of_nonneg
        (neg_nonpos.mpr hinv) (norm_nonneg (x - y))
    simpa using
      (mul_le_mul_of_nonneg_right hk_le (norm_nonneg y))

omit [InnerProductSpace ℝ E] [CompleteSpace E]
  [SecondCountableTopology E] in
lemma laplaceKernelNormalizer_pos
    (P : Measure E) [IsProbabilityMeasure P]
    (τ : ℝ) (hτ : 0 < τ) (x : E) :
    0 < kernelNormalizer (laplaceKernel τ) P x := by
  unfold kernelNormalizer laplaceKernel
  exact integral_exp_pos (laplaceKernel_integrable P τ hτ x)

/-- For Gaussian laws the paper's mean shift is the kernel centroid minus the
probe.  The Gaussian hypothesis is used only to supply the first moment. -/
theorem gaussian_meanShift_laplace_eq_kernelCentroid_sub
    (P : Measure E) [IsGaussian P] [IsProbabilityMeasure P]
    (τ : ℝ) (hτ : 0 < τ) (x : E) :
    meanShift (laplaceKernel τ) P x =
      kernelCentroid (laplaceKernel τ) P x - x :=
  meanShift_eq_kernelCentroid_sub P (laplaceKernel τ) x
    (laplaceKernel_integrable P τ hτ x)
    (laplaceKernel_smul_id_integrable P τ hτ x)
    (ne_of_gt (laplaceKernelNormalizer_pos P τ hτ x))

/-- Before evaluating the Gaussian exponential tilt explicitly, the raw
Laplacian drift already has a rigorous radial limit: the difference of the two
tilted centroids. -/
theorem gaussian_laplaceMeanShiftDrift_radial_tendsto
    (P Q : Measure E)
    [IsGaussian P] [IsGaussian Q]
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (τ : ℝ) (hτ : 0 < τ) (u : E) (hu : ‖u‖ = 1) :
    Tendsto
      (fun r : ℝ =>
        meanShiftDrift (laplaceKernel τ) P Q (r • u))
      atTop
      (𝓝 (exponentialTiltCentroid P τ u -
        exponentialTiltCentroid Q τ u)) := by
  have hp :=
    gaussian_kernelCentroid_laplace_radial_tendsto P τ hτ u hu
  have hq :=
    gaussian_kernelCentroid_laplace_radial_tendsto Q τ hτ u hu
  have hsub := hp.sub hq
  apply hsub.congr'
  filter_upwards with r
  unfold meanShiftDrift
  rw [gaussian_meanShift_laplace_eq_kernelCentroid_sub P τ hτ,
    gaussian_meanShift_laplace_eq_kernelCentroid_sub Q τ hτ]
  abel

section MultivariateRadial

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Explicit version of the Laplacian/Gaussian radial limit from the authors'
rebuttal: along `r • u`, the raw Laplace-kernel drift converges to the
difference of the Gaussian means plus the covariance-action difference scaled
by `τ⁻¹`. -/
theorem multivariateGaussian_laplaceMeanShiftDrift_radial_tendsto
    (μp μq : EuclideanSpace ℝ ι)
    (Sp Sq : Matrix ι ι ℝ) (hSp : Sp.PosSemidef) (hSq : Sq.PosSemidef)
    (τ : ℝ) (hτ : 0 < τ) (u : EuclideanSpace ℝ ι) (hu : ‖u‖ = 1) :
    Tendsto
      (fun r : ℝ =>
        meanShiftDrift (laplaceKernel τ)
          (multivariateGaussian μp Sp)
          (multivariateGaussian μq Sq) (r • u))
      atTop
      (𝓝
        ((μp + (1 / τ) • (((Matrix.toEuclideanCLM (n := ι) (𝕜 := ℝ)) Sp) u)) -
          (μq + (1 / τ) • (((Matrix.toEuclideanCLM (n := ι) (𝕜 := ℝ)) Sq) u)))) := by
  simpa [multivariateGaussian_exponentialTiltCentroid μp Sp hSp τ hτ u,
    multivariateGaussian_exponentialTiltCentroid μq Sq hSq τ hτ u] using
    gaussian_laplaceMeanShiftDrift_radial_tendsto
      (P := multivariateGaussian μp Sp)
      (Q := multivariateGaussian μq Sq) τ hτ u hu

end MultivariateRadial

end GaussianDomination

end IntegralLimits

end DriftingIdentifiability
