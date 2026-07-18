import DriftingIdentifiability.LaplaceFoliationCancellation
import Mathlib.Analysis.InnerProductSpace.Trace

/-!
# The classical Hessian of the Euclidean Laplace displacement potential

This file closes the analytic part of G4/P1 without an almost-everywhere
regularity detour.  Although the Laplace kernel itself is not differentiable
on the diagonal, its displacement-weighted vector field is differentiable
there: the derivative is `-I`.  Its derivative kernel is uniformly bounded,
so differentiation under every finite measure is legitimate, including at
atoms.

The resulting Hessian is symmetric and its trace gives the pointwise
companion elliptic identity used by the foliation argument.
-/

open MeasureTheory Filter Topology Asymptotics
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The point-source derivative of
`x ↦ exp (-‖x-y‖/τ) • (y-x)`.

Away from the diagonal this is
`exp(-r/τ) ((r/τ) u ⊗ u - I)`.  At the diagonal the rank-one term vanishes
continuously and the derivative is `-I`. -/
noncomputable def laplaceDisplacementKernelHessian
    (τ : ℝ) (x y : E) : E →L[ℝ] E :=
  by
    classical
    exact if h : x = y then
      -(ContinuousLinearMap.id ℝ E)
    else
      laplaceKernel τ x y • (-(ContinuousLinearMap.id ℝ E)) +
        ((-(Real.exp (-‖x - y‖ / τ) / (τ * ‖x - y‖))) •
          innerSL ℝ (x - y)).smulRight (y - x)

/-- Off the diagonal, the scalar Laplace kernel has the usual radial
Fréchet derivative. -/
private lemma hasFDerivAt_laplaceKernel_left_of_ne
    {τ : ℝ} (_hτ : 0 < τ) {x y : E} (hxy : x ≠ y) :
    HasFDerivAt (fun z : E => laplaceKernel τ z y)
      ((-(Real.exp (-‖x - y‖ / τ) / (τ * ‖x - y‖))) •
        innerSL ℝ (x - y)) x := by
  have hne : ‖x - y‖ ≠ 0 := norm_ne_zero_iff.mpr (sub_ne_zero.mpr hxy)
  have hne2 : ‖x - y‖ ^ 2 ≠ 0 := pow_ne_zero 2 hne
  have hsq : HasFDerivAt (fun z : E => ‖z - y‖ ^ 2)
      (2 • innerSL ℝ (x - y)) x := by
    have h := (hasStrictFDerivAt_norm_sq (x - y)).hasFDerivAt.comp x
      ((hasFDerivAt_id x).sub_const y)
    rwa [ContinuousLinearMap.comp_id] at h
  have hsqrt : HasFDerivAt (fun z : E => Real.sqrt (‖z - y‖ ^ 2))
      ((1 / (2 * Real.sqrt (‖x - y‖ ^ 2))) •
        (2 • innerSL ℝ (x - y))) x :=
    (Real.hasDerivAt_sqrt hne2).comp_hasFDerivAt x hsq
  have hnorm : HasFDerivAt (fun z : E => ‖z - y‖)
      ((1 / (2 * Real.sqrt (‖x - y‖ ^ 2))) •
        (2 • innerSL ℝ (x - y))) x := by
    have hfun : (fun z : E => Real.sqrt (‖z - y‖ ^ 2)) =
        fun z : E => ‖z - y‖ := by
      funext z
      rw [Real.sqrt_sq (norm_nonneg _)]
    rwa [hfun] at hsqrt
  have harg : HasFDerivAt (fun z : E => -(1 / τ) * ‖z - y‖)
      ((-(1 / τ)) •
        ((1 / (2 * Real.sqrt (‖x - y‖ ^ 2))) •
          (2 • innerSL ℝ (x - y)))) x :=
    hnorm.const_mul (-(1 / τ))
  have hexp := harg.exp
  unfold laplaceKernel
  convert hexp using 1
  ext v
  simp only [smul_apply, innerSL_apply_apply, Real.sqrt_sq (norm_nonneg _)]
  field_simp
  ring

/-- The displacement-weighted kernel is differentiable through the diagonal,
with derivative `laplaceDisplacementKernelHessian`. -/
theorem hasFDerivAt_laplaceDisplacementKernel
    {τ : ℝ} (hτ : 0 < τ) (y x : E) :
    HasFDerivAt (fun z : E => laplaceKernel τ z y • (y - z))
      (laplaceDisplacementKernelHessian τ x y) x := by
  by_cases hxy : x = y
  · subst x
    rw [laplaceDisplacementKernelHessian, dif_pos rfl,
      hasFDerivAt_iff_isLittleO_nhds_zero]
    have hk : Tendsto
        (fun h : E => 1 - laplaceKernel τ (y + h) y)
        (𝓝 0) (𝓝 0) := by
      have hc : ContinuousAt (fun h : E => laplaceKernel τ (y + h) y) 0 := by
        unfold laplaceKernel
        fun_prop
      convert tendsto_const_nhds.sub hc using 1
      all_goals simp [laplaceKernel]
    have ho : (fun h : E => 1 - laplaceKernel τ (y + h) y) =o[𝓝 0]
        (fun _ : E => (1 : ℝ)) := (isLittleO_one_iff ℝ).2 hk
    have hsmall := ho.smul_isBigO (isBigO_refl (fun h : E => h) (𝓝 0))
    simpa only [one_smul] using hsmall.congr'
      (Filter.Eventually.of_forall fun h => by
        simp [laplaceKernel, sub_eq_add_neg]
        module)
      Filter.EventuallyEq.rfl
  · have hk := hasFDerivAt_laplaceKernel_left_of_ne hτ hxy
    have hv : HasFDerivAt (fun z : E => y - z)
        (-(ContinuousLinearMap.id ℝ E)) x := by
      change HasFDerivAt ((fun _ : E => y) - id)
        (-(ContinuousLinearMap.id ℝ E)) x
      simpa only [zero_sub] using
        (hasFDerivAt_const (𝕜 := ℝ) y x).sub (hasFDerivAt_id (𝕜 := ℝ) x)
    have hprod := hk.smul hv
    rw [laplaceDisplacementKernelHessian, dif_neg hxy]
    change HasFDerivAt
      ((fun z : E => laplaceKernel τ z y) • fun z : E => y - z) _ x
    exact hprod

/-- The point-source Hessian kernel is symmetric. -/
theorem laplaceDisplacementKernelHessian_symmetric
    (τ : ℝ) (x y v w : E) :
    ⟪laplaceDisplacementKernelHessian τ x y v, w⟫ =
      ⟪laplaceDisplacementKernelHessian τ x y w, v⟫ := by
  by_cases hxy : x = y
  · simp [laplaceDisplacementKernelHessian, hxy, real_inner_comm]
  · have hyx : y - x = -(x - y) := by abel
    rw [laplaceDisplacementKernelHessian, dif_neg hxy, hyx]
    simp only [add_apply,
      smul_apply, neg_apply, ContinuousLinearMap.id_apply,
      ContinuousLinearMap.smulRight_apply, innerSL_apply_apply,
      inner_add_left, real_inner_smul_left, inner_neg_left, smul_eq_mul]
    rw [real_inner_comm v w]
    ring

/-- A uniform operator-norm bound for the point-source Hessian. -/
theorem norm_laplaceDisplacementKernelHessian_le
    {τ : ℝ} (hτ : 0 < τ) (x y : E) :
    ‖laplaceDisplacementKernelHessian τ x y‖ ≤ 2 := by
  by_cases hxy : x = y
  · rw [laplaceDisplacementKernelHessian, dif_pos hxy, norm_neg]
    exact (ContinuousLinearMap.norm_id_le.trans (by norm_num))
  · have hr : 0 < ‖x - y‖ := norm_pos_iff.mpr (sub_ne_zero.mpr hxy)
    have hepos : 0 < Real.exp (-‖x - y‖ / τ) := Real.exp_pos _
    have heone : Real.exp (-‖x - y‖ / τ) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      exact div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (norm_nonneg _)) hτ.le
    have hrad0 : 0 ≤ ‖x - y‖ * Real.exp (-‖x - y‖ / τ) / τ := by positivity
    have hrad : ‖x - y‖ * Real.exp (-‖x - y‖ / τ) / τ ≤ 1 := by
      have hmax := mul_exp_neg_div_le hτ (norm_nonneg (x - y))
      have hdiv : ‖x - y‖ * Real.exp (-‖x - y‖ / τ) / τ ≤ Real.exp (-1) :=
        (div_le_iff₀ hτ).2 (by simpa [mul_comm] using hmax)
      exact hdiv.trans (by
        rw [Real.exp_le_one_iff]
        norm_num)
    rw [laplaceDisplacementKernelHessian, dif_neg hxy]
    calc
      ‖laplaceKernel τ x y • (-(ContinuousLinearMap.id ℝ E)) +
          ((-(Real.exp (-‖x - y‖ / τ) / (τ * ‖x - y‖))) •
            innerSL ℝ (x - y)).smulRight (y - x)‖
          ≤ ‖laplaceKernel τ x y • (-(ContinuousLinearMap.id ℝ E))‖ +
            ‖((-(Real.exp (-‖x - y‖ / τ) / (τ * ‖x - y‖))) •
              innerSL ℝ (x - y)).smulRight (y - x)‖ :=
        norm_add_le _ _
      _ ≤ 1 + 1 := by
        apply add_le_add
        · rw [norm_smul, norm_neg, Real.norm_eq_abs,
            abs_of_pos (by unfold laplaceKernel; exact Real.exp_pos _)]
          have hkone : laplaceKernel τ x y ≤ 1 := by
            unfold laplaceKernel
            rw [Real.exp_le_one_iff]
            exact mul_nonpos_of_nonpos_of_nonneg
              (neg_nonpos.mpr (by positivity : 0 ≤ 1 / τ)) (norm_nonneg _)
          simpa only [mul_one] using
            (mul_le_mul hkone ContinuousLinearMap.norm_id_le
              (norm_nonneg (ContinuousLinearMap.id ℝ E)) zero_le_one)
        · rw [ContinuousLinearMap.norm_smulRight_apply, norm_smul,
            innerSL_apply_norm]
          rw [norm_sub_rev y x, Real.norm_eq_abs,
            abs_of_neg (neg_lt_zero.mpr (div_pos hepos (mul_pos hτ hr)))]
          have heq :
              (Real.exp (-‖x - y‖ / τ) / (τ * ‖x - y‖)) *
                  (‖x - y‖ * ‖x - y‖) =
                ‖x - y‖ * Real.exp (-‖x - y‖ / τ) / τ := by
            field_simp [hτ.ne', hr.ne']
          simp only [neg_neg]
          rw [mul_assoc, heq]
          exact hrad
      _ = 2 := by norm_num

section Measure

variable [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
  [SecondCountableTopology E] [FiniteDimensional ℝ E]

/-- The actual Hessian of the displacement potential, obtained by integrating
the bounded point-source Hessian. -/
noncomputable def laplaceDisplacementHessian
    (τ : ℝ) (μ : Measure E) (x : E) : E →L[ℝ] E :=
  ∫ y, laplaceDisplacementKernelHessian τ x y ∂μ

/-- Trace as a continuous linear functional on continuous endomorphisms of a
finite-dimensional Euclidean space. -/
noncomputable def continuousLinearMapTrace :
    (E →L[ℝ] E) →L[ℝ] ℝ :=
  LinearMap.toContinuousLinearMap
    { toFun := fun A => A.toLinearMap.trace ℝ E
      map_add' := by
        intro A B
        exact map_add (LinearMap.trace ℝ E) A.toLinearMap B.toLinearMap
      map_smul' := by
        intro c A
        exact map_smul (LinearMap.trace ℝ E) c A.toLinearMap }

set_option linter.unusedSectionVars false in
@[simp]
theorem continuousLinearMapTrace_apply (A : E →L[ℝ] E) :
    continuousLinearMapTrace A = A.toLinearMap.trace ℝ E := rfl

/-- The classical Laplacian supplied by the integrated Hessian. -/
noncomputable def laplaceDisplacementLaplacian
    (τ : ℝ) (μ : Measure E) (x : E) : ℝ :=
  continuousLinearMapTrace (laplaceDisplacementHessian τ μ x)

set_option linter.unusedSectionVars false in
private theorem laplaceDisplacementKernelHessian_eq_rankOne
    (τ : ℝ) {x y : E} (hxy : x ≠ y) :
    laplaceDisplacementKernelHessian τ x y =
      (Real.exp (-‖x - y‖ / τ) / (τ * ‖x - y‖)) •
          InnerProductSpace.rankOne ℝ (x - y) (x - y) -
        Real.exp (-‖x - y‖ / τ) • ContinuousLinearMap.id ℝ E := by
  rw [laplaceDisplacementKernelHessian, dif_neg hxy]
  ext v
  have hyx : y - x = -(x - y) := by abel
  rw [hyx, laplaceKernel_eq_exp]
  simp only [add_apply, sub_apply, smul_apply, neg_apply,
    ContinuousLinearMap.id_apply, ContinuousLinearMap.smulRight_apply,
    InnerProductSpace.rankOne_apply, innerSL_apply_apply]
  module

/-- Pointwise trace of the source Hessian. -/
theorem continuousLinearMapTrace_laplaceDisplacementKernelHessian
    {τ : ℝ} (hτ : 0 < τ) (x y : E) :
    continuousLinearMapTrace (laplaceDisplacementKernelHessian τ x y) =
      Real.exp (-‖x - y‖ / τ) *
        (‖x - y‖ / τ - (Module.finrank ℝ E : ℝ)) := by
  by_cases hxy : x = y
  · subst x
    simp [laplaceDisplacementKernelHessian, continuousLinearMapTrace,
      LinearMap.trace_id]
  · have hr : ‖x - y‖ ≠ 0 := norm_ne_zero_iff.mpr (sub_ne_zero.mpr hxy)
    rw [laplaceDisplacementKernelHessian_eq_rankOne τ hxy]
    simp only [continuousLinearMapTrace_apply]
    simp [InnerProductSpace.trace_rankOne, LinearMap.trace_id]
    field_simp [hτ.ne', hr]
    rw [real_inner_comm y x]
    have hnorm := norm_sub_sq_real x y
    rw [real_inner_comm y x] at hnorm
    nlinarith [hnorm]

/-- The point-source Hessian is strongly measurable in the source variable.
This is obtained from Mathlib's measurability theorem for a parameterized
Fréchet derivative, avoiding a separate diagonal-continuity proof. -/
private theorem stronglyMeasurable_laplaceDisplacementKernelHessian
    {τ : ℝ} (hτ : 0 < τ) (x : E) :
    StronglyMeasurable (fun y : E => laplaceDisplacementKernelHessian τ x y) := by
  let F : E → E → E := fun y z => laplaceKernel τ z y • (y - z)
  have hFcont : Continuous F.uncurry := by
    dsimp [F]
    unfold laplaceKernel
    fun_prop
  have hmeas : Measurable
      (fun y : E => fderiv ℝ (F y) x) := by
    have hjoint := measurable_fderiv_with_param ℝ hFcont
    exact hjoint.comp (by fun_prop : Measurable (fun y : E => (y, x)))
  have heq : (fun y : E => laplaceDisplacementKernelHessian τ x y) =
      fun y : E => fderiv ℝ (F y) x := by
    funext y
    exact (hasFDerivAt_laplaceDisplacementKernel hτ y x).fderiv.symm
  rw [heq]
  exact hmeas.stronglyMeasurable

/-- The Hessian kernel is integrable against every finite measure. -/
theorem integrable_laplaceDisplacementKernelHessian
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    Integrable (fun y => laplaceDisplacementKernelHessian τ x y) μ := by
  refine Integrable.of_bound
    (stronglyMeasurable_laplaceDisplacementKernelHessian hτ x).aestronglyMeasurable 2 ?_
  exact ae_of_all _ fun y => norm_laplaceDisplacementKernelHessian_le hτ x y

/-- Trace commutes with the Hessian integral. -/
theorem laplaceDisplacementLaplacian_eq_integral_trace
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    laplaceDisplacementLaplacian τ μ x =
      ∫ y, continuousLinearMapTrace
        (laplaceDisplacementKernelHessian τ x y) ∂μ := by
  have hint := integrable_laplaceDisplacementKernelHessian hτ μ x
  unfold laplaceDisplacementLaplacian laplaceDisplacementHessian
  exact (continuousLinearMapTrace.integral_comp_comm hint).symm

/-- **Pointwise companion elliptic identity.**  For every finite measure and
at every point,

`ψ - τ² Δψ = (finrank E + 1) τ² Z`.

Here `Δψ` is the trace of the everywhere-classical Hessian constructed above.
No a.e. exceptional set and no regularity assumption on the measure occurs. -/
theorem laplaceDisplacementPotential_elliptic
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    laplaceDisplacementPotential τ μ x -
        τ ^ 2 * laplaceDisplacementLaplacian τ μ x =
      ((Module.finrank ℝ E : ℝ) + 1) * τ ^ 2 *
        kernelNormalizer (laplaceKernel τ) μ x := by
  have hψ : Integrable (fun y => matern32Profile τ ‖x - y‖) μ :=
    integrable_matern32Profile_norm hτ μ x
  have hA := integrable_laplaceDisplacementKernelHessian hτ μ x
  have htr : Integrable (fun y => continuousLinearMapTrace
      (laplaceDisplacementKernelHessian τ x y)) μ :=
    continuousLinearMapTrace.integrable_comp hA
  have hK : Integrable (fun y => laplaceKernel τ x y) μ :=
    laplaceKernel_integrable μ τ hτ x
  rw [laplaceDisplacementLaplacian_eq_integral_trace hτ μ x]
  unfold laplaceDisplacementPotential kernelNormalizer
  rw [← integral_const_mul, ← integral_sub hψ (htr.const_mul (τ ^ 2))]
  calc
    ∫ y, matern32Profile τ ‖x - y‖ -
          τ ^ 2 * continuousLinearMapTrace
            (laplaceDisplacementKernelHessian τ x y) ∂μ =
        ∫ y, (((Module.finrank ℝ E : ℝ) + 1) * τ ^ 2) *
          laplaceKernel τ x y ∂μ := by
      apply integral_congr_ae
      exact ae_of_all _ fun y => by
        change matern32Profile τ ‖x - y‖ -
            τ ^ 2 * continuousLinearMapTrace
              (laplaceDisplacementKernelHessian τ x y) =
          ((Module.finrank ℝ E : ℝ) + 1) * τ ^ 2 * laplaceKernel τ x y
        rw [continuousLinearMapTrace_laplaceDisplacementKernelHessian hτ]
        unfold matern32Profile
        rw [laplaceKernel_eq_exp]
        field_simp [hτ.ne']
        ring
    _ = ((Module.finrank ℝ E : ℝ) + 1) * τ ^ 2 *
        ∫ y, laplaceKernel τ x y ∂μ := by
      rw [integral_const_mul]

/-- **Everywhere classical P1 bridge.**  The displacement field is Fréchet
differentiable at every point, for every finite measure, with no atomlessness,
density, support, or moment hypothesis. -/
theorem hasFDerivAt_laplaceDisplacementField
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    HasFDerivAt (laplaceDisplacementField τ μ)
      (laplaceDisplacementHessian τ μ x) x := by
  let F : E → E → E := fun z y => laplaceKernel τ z y • (y - z)
  let F' : E → E → (E →L[ℝ] E) :=
    fun z y => laplaceDisplacementKernelHessian τ z y
  have hmain : HasFDerivAt (fun z => ∫ y, F z y ∂μ) (∫ y, F' x y ∂μ) x := by
    refine hasFDerivAt_integral_of_dominated_of_fderiv_le
      (bound := fun _ => (2 : ℝ)) (F' := F') (s := Set.univ)
      Filter.univ_mem ?_ ?_ ?_ ?_ (integrable_const 2) ?_
    · exact Filter.Eventually.of_forall fun z => by
        apply Continuous.aestronglyMeasurable
        dsimp [F]
        unfold laplaceKernel
        fun_prop
    · simpa [F] using integrable_laplaceDisplacementField_integrand hτ μ x
    · simpa [F'] using
        (stronglyMeasurable_laplaceDisplacementKernelHessian hτ x).aestronglyMeasurable
    · exact ae_of_all _ fun y z _ =>
        norm_laplaceDisplacementKernelHessian_le hτ z y
    · exact ae_of_all _ fun y z _ => by
        simpa [F, F'] using hasFDerivAt_laplaceDisplacementKernel hτ y z
  change HasFDerivAt (fun z => ∫ y, laplaceKernel τ z y • (y - z) ∂μ)
    (∫ y, laplaceDisplacementKernelHessian τ x y ∂μ) x
  exact hmain

/-- The integrated Hessian is symmetric. -/
theorem laplaceDisplacementHessian_symmetric
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ]
    (x v w : E) :
    ⟪laplaceDisplacementHessian τ μ x v, w⟫ =
      ⟪laplaceDisplacementHessian τ μ x w, v⟫ := by
  have hint := integrable_laplaceDisplacementKernelHessian hτ μ x
  have hintv := (ContinuousLinearMap.apply ℝ E v).integrable_comp hint
  have hintw := (ContinuousLinearMap.apply ℝ E w).integrable_comp hint
  have hintv' : Integrable
      (fun y => (laplaceDisplacementKernelHessian τ x y) v) μ := by
    simpa using hintv
  have hintw' : Integrable
      (fun y => (laplaceDisplacementKernelHessian τ x y) w) μ := by
    simpa using hintw
  rw [laplaceDisplacementHessian,
    ContinuousLinearMap.integral_apply hint,
    ContinuousLinearMap.integral_apply hint]
  calc
    ⟪∫ y, (laplaceDisplacementKernelHessian τ x y) v ∂μ, w⟫
        = ∫ y, ⟪w, (laplaceDisplacementKernelHessian τ x y) v⟫ ∂μ := by
          rw [real_inner_comm, integral_inner hintv']
    _ = ∫ y, ⟪v, (laplaceDisplacementKernelHessian τ x y) w⟫ ∂μ := by
          apply integral_congr_ae
          exact ae_of_all _ fun y => by
            simpa [real_inner_comm] using
              laplaceDisplacementKernelHessian_symmetric τ x y v w
    _ = ⟪∫ y, (laplaceDisplacementKernelHessian τ x y) w ∂μ, v⟫ := by
          rw [integral_inner hintw', real_inner_comm]

end Measure

end DriftingIdentifiability
