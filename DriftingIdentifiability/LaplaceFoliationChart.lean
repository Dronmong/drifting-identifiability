import DriftingIdentifiability.LaplaceDisplacementHessian
import Mathlib.Analysis.Calculus.Implicit

/-!
# G4 regular foliation: classical ratio differentiation and cancellation

The preceding Hessian module removes all a.e. regularity from G4/P1.  This
file applies that result on the regular set of the `q` displacement potential.
It proves that the normalizer ratio is classically differentiable there,
differentiates the actual zero-drift alignment, and derives the exact
measure-level cancellation equation.

The implicit-function chart itself is exposed separately below.  Its first
coordinate is definitionally the `q` potential; no abstract leaf chart is
assumed.
-/

open MeasureTheory Filter Topology
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
  [SecondCountableTopology E] [FiniteDimensional ℝ E]

/-- A regular-set formula for the normalizer ratio using only the two
displacement fields. -/
noncomputable def laplaceRegularRatioFormula
    (τ : ℝ) (p q : Measure E) (x : E) : ℝ :=
  ⟪laplaceDisplacementField τ p x, laplaceDisplacementField τ q x⟫ /
    ⟪laplaceDisplacementField τ q x, laplaceDisplacementField τ q x⟫

/-- The integrated Laplace displacement field is continuous everywhere. -/
theorem continuous_laplaceDisplacementField
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] :
    Continuous (laplaceDisplacementField τ μ) := by
  rw [continuous_iff_continuousAt]
  intro x
  exact (hasFDerivAt_laplaceDisplacementField hτ μ x).continuousAt

/-- The displacement potential is strictly differentiable everywhere.  This
is the `C¹` input required by Mathlib's implicit-function theorem. -/
theorem hasStrictFDerivAt_laplaceDisplacementPotential
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    HasStrictFDerivAt (laplaceDisplacementPotential τ μ)
      (innerSL ℝ (laplaceDisplacementField τ μ x)) x := by
  apply hasStrictFDerivAt_of_hasFDerivAt_of_continuousAt
    (f' := fun y => innerSL ℝ (laplaceDisplacementField τ μ y))
  · exact Filter.Eventually.of_forall fun y =>
      hasFDerivAt_laplaceDisplacementPotential hτ μ y
  · exact (innerSL ℝ).continuous.continuousAt.comp
      (continuous_laplaceDisplacementField hτ μ).continuousAt

omit [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
    [SecondCountableTopology E] [FiniteDimensional ℝ E] in
/-- A nonzero gradient makes the derivative of a real-valued potential
surjective. -/
theorem innerSL_range_eq_top {v : E} (hv : v ≠ 0) :
    (innerSL ℝ v).range = ⊤ := by
  apply LinearMap.range_eq_top.mpr
  intro r
  refine ⟨(r / ⟪v, v⟫) • v, ?_⟩
  change ⟪v, (r / ⟪v, v⟫) • v⟫ = r
  rw [real_inner_smul_right]
  exact div_mul_cancel₀ r (inner_self_ne_zero.mpr hv)

/-- The canonical implicit-function chart at a regular point.  Its first
coordinate is the actual displacement potential, while its second coordinate
lies in the tangent hyperplane `ker ⟨D_q(x), ·⟩`. -/
noncomputable def laplaceRegularLeafChart
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    OpenPartialHomeomorph E
      (ℝ × (innerSL ℝ (laplaceDisplacementField τ q x)).ker) :=
  (hasStrictFDerivAt_laplaceDisplacementPotential hτ q x).implicitToOpenPartialHomeomorph
    (laplaceDisplacementPotential τ q)
    (innerSL ℝ (laplaceDisplacementField τ q x))
    (innerSL_range_eq_top hreg)

/-- The regular point belongs to the source of its canonical leaf chart. -/
theorem laplaceRegularLeafChart_mem_source
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    x ∈ (laplaceRegularLeafChart hτ q x hreg).source := by
  exact HasStrictFDerivAt.mem_implicitToOpenPartialHomeomorph_source
    (hasStrictFDerivAt_laplaceDisplacementPotential hτ q x)
    (innerSL_range_eq_top hreg)

/-- The chart sends its base point to potential value plus zero tangent
coordinate. -/
@[simp] theorem laplaceRegularLeafChart_apply_self
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    laplaceRegularLeafChart hτ q x hreg x =
      (laplaceDisplacementPotential τ q x, 0) := by
  exact HasStrictFDerivAt.implicitToOpenPartialHomeomorph_self
    (hasStrictFDerivAt_laplaceDisplacementPotential hτ q x)
    (innerSL_range_eq_top hreg)

/-- The first chart coordinate is exactly the displacement potential. -/
@[simp] theorem laplaceRegularLeafChart_fst
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) (z : E) :
    (laplaceRegularLeafChart hτ q x hreg z).1 =
      laplaceDisplacementPotential τ q z := by
  exact HasStrictFDerivAt.implicitToOpenPartialHomeomorph_fst
    (hasStrictFDerivAt_laplaceDisplacementPotential hτ q x)
    (innerSL_range_eq_top hreg) z

/-- Under zero drift, the displacement quotient equals the normalizer ratio
wherever the `q` field is nonzero. -/
theorem laplaceRegularRatioFormula_eq
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {x : E} (hreg : laplaceDisplacementField τ q x ≠ 0) :
    laplaceRegularRatioFormula τ p q x = laplaceNormalizerRatio τ p q x := by
  rw [laplaceRegularRatioFormula,
    laplaceDisplacementField_eq_ratio_smul_of_zeroDrift hτ p q hzero x,
    real_inner_smul_left]
  exact mul_div_cancel_right₀ _ (inner_self_ne_zero.mpr hreg)

set_option maxHeartbeats 800000 in
-- Inner-product quotient differentiation needs extra elaboration budget in finite dimension.
/-- The displacement quotient is differentiable at every regular point. -/
theorem differentiableAt_laplaceRegularRatioFormula
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsFiniteMeasure p] [IsFiniteMeasure q]
    {x : E} (hreg : laplaceDisplacementField τ q x ≠ 0) :
    DifferentiableAt ℝ (laplaceRegularRatioFormula τ p q) x := by
  have hp := (hasFDerivAt_laplaceDisplacementField hτ p x).differentiableAt
  have hq := (hasFDerivAt_laplaceDisplacementField hτ q x).differentiableAt
  unfold laplaceRegularRatioFormula
  have hn : DifferentiableAt ℝ
      (fun z : E =>
        ⟪laplaceDisplacementField τ p z, laplaceDisplacementField τ q z⟫) x :=
    DifferentiableAt.inner ℝ hp hq
  have hd : DifferentiableAt ℝ
      (fun z : E =>
        ⟪laplaceDisplacementField τ q z, laplaceDisplacementField τ q z⟫) x :=
    DifferentiableAt.inner ℝ hq hq
  change DifferentiableAt ℝ
    ((fun z : E =>
        ⟪laplaceDisplacementField τ p z, laplaceDisplacementField τ q z⟫) *
      (fun z : E =>
        ⟪laplaceDisplacementField τ q z, laplaceDisplacementField τ q z⟫)⁻¹) x
  exact hn.mul (hd.inv (inner_self_ne_zero.mpr hreg))

/-- **Classical ratio regularity on the regular set.**  Although the raw
Laplace normalizers need not be differentiable at atoms, their ratio under
zero drift is differentiable wherever `D_q ≠ 0`, because it agrees locally
with a quotient of the everywhere-`C¹` displacement fields. -/
theorem differentiableAt_laplaceNormalizerRatio_of_zeroDrift
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {x : E} (hreg : laplaceDisplacementField τ q x ≠ 0) :
    DifferentiableAt ℝ (laplaceNormalizerRatio τ p q) x := by
  have hqcont : ContinuousAt (laplaceDisplacementField τ q) x :=
    (hasFDerivAt_laplaceDisplacementField hτ q x).continuousAt
  have hne : ∀ᶠ z in 𝓝 x, laplaceDisplacementField τ q z ≠ 0 :=
    hqcont.eventually_ne hreg
  have heq : laplaceNormalizerRatio τ p q =ᶠ[𝓝 x]
      laplaceRegularRatioFormula τ p q :=
    hne.mono fun z hz => (laplaceRegularRatioFormula_eq hτ p q hzero hz).symm
  exact (differentiableAt_laplaceRegularRatioFormula hτ p q hreg).congr_of_eventuallyEq heq

/-- At a regular point, differentiating the actual aligned fields gives the
classical product rule with the actual integrated Hessians. -/
theorem laplaceDisplacementHessian_product_of_zeroDrift
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {x : E} (hreg : laplaceDisplacementField τ q x ≠ 0) :
    laplaceDisplacementHessian τ p x =
      laplaceNormalizerRatio τ p q x • laplaceDisplacementHessian τ q x +
        (fderiv ℝ (laplaceNormalizerRatio τ p q) x).smulRight
          (laplaceDisplacementField τ q x) := by
  let R := laplaceNormalizerRatio τ p q
  let Dp := laplaceDisplacementField τ p
  let Dq := laplaceDisplacementField τ q
  have hfun : Dp = fun z => R z • Dq z := by
    funext z
    exact laplaceDisplacementField_eq_ratio_smul_of_zeroDrift hτ p q hzero z
  have hR :=
    (differentiableAt_laplaceNormalizerRatio_of_zeroDrift hτ p q hzero hreg).hasFDerivAt
  have hp := hasFDerivAt_laplaceDisplacementField hτ p x
  have hq := hasFDerivAt_laplaceDisplacementField hτ q x
  exact fderiv_eq_of_vectorField_eq_smul hfun hR hp hq

/-- The derivative of the actual normalizer ratio annihilates every tangent
direction to a regular `q`-potential leaf. -/
theorem laplaceNormalizerRatio_fderiv_tangent_eq_zero
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {x v : E} (hreg : laplaceDisplacementField τ q x ≠ 0)
    (hv : ⟪laplaceDisplacementField τ q x, v⟫ = 0) :
    fderiv ℝ (laplaceNormalizerRatio τ p q) x v = 0 := by
  apply laplaceFoliation_differential_tangent_eq_zero
    (laplaceNormalizerRatio τ p q x)
    (laplaceDisplacementField τ q x)
    (fderiv ℝ (laplaceNormalizerRatio τ p q) x)
    (laplaceDisplacementHessian τ p x)
    (laplaceDisplacementHessian τ q x)
  · intro w
    exact congrArg (fun A : E →L[ℝ] E => A w)
      (laplaceDisplacementHessian_product_of_zeroDrift hτ p q hzero hreg)
  · exact laplaceDisplacementHessian_symmetric hτ p x
  · exact laplaceDisplacementHessian_symmetric hτ q x
  · exact hreg
  · exact hv

/-- Trace of the differentiated alignment. -/
theorem laplaceDisplacementLaplacian_product_of_zeroDrift
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {x : E} (hreg : laplaceDisplacementField τ q x ≠ 0) :
    laplaceDisplacementLaplacian τ p x =
      laplaceNormalizerRatio τ p q x * laplaceDisplacementLaplacian τ q x +
        fderiv ℝ (laplaceNormalizerRatio τ p q) x
          (laplaceDisplacementField τ q x) := by
  have hprod := laplaceDisplacementHessian_product_of_zeroDrift
    hτ p q hzero hreg
  unfold laplaceDisplacementLaplacian
  rw [hprod]
  simp [continuousLinearMapTrace, LinearMap.trace_smulRight]

/-- **Actual measure-level foliation cancellation.**  At every regular point,
the pointwise elliptic identities and the differentiated zero-drift alignment
give

`τ² dR(D_q) = ψ_p - R ψ_q`.

There are no chart, Hessian, Laplacian, or a.e.-regularity hypotheses left in
the statement. -/
theorem laplaceFoliation_measureCancellation
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {x : E} (hreg : laplaceDisplacementField τ q x ≠ 0) :
    τ ^ 2 * fderiv ℝ (laplaceNormalizerRatio τ p q) x
        (laplaceDisplacementField τ q x) =
      laplaceDisplacementPotential τ p x -
        laplaceNormalizerRatio τ p q x *
          laplaceDisplacementPotential τ q x := by
  have hpde := laplaceDisplacementPotential_elliptic hτ p x
  have hqde := laplaceDisplacementPotential_elliptic hτ q x
  have hZ : kernelNormalizer (laplaceKernel τ) p x =
      laplaceNormalizerRatio τ p q x *
        kernelNormalizer (laplaceKernel τ) q x := by
    unfold laplaceNormalizerRatio
    field_simp [(laplaceKernelNormalizer_pos q τ hτ x).ne']
  have hlap := laplaceDisplacementLaplacian_product_of_zeroDrift
    hτ p q hzero hreg
  linear_combination -hpde + laplaceNormalizerRatio τ p q x * hqde -
    (((Module.finrank ℝ E : ℝ) + 1) * τ ^ 2) * hZ - τ ^ 2 * hlap

end DriftingIdentifiability
