import DriftingIdentifiability.LaplaceRadialFoundations
import DriftingIdentifiability.LaplaceUnconditionalConverse

/-!
# Radial Laplace converse, milestone L1: the far-field foundation

This file is milestone `L1` of the ℓ²/radial higher-dimensional Laplace program
(`LaplaceHigherDim.md`, §4.8/§4.6(a), D2.a).  Its headline is
`zeroDrift_tiltedCentroid_eq`: pointwise zero Laplace mean-shift drift forces the
two exponential-tilt centroids to agree in **every** direction, for arbitrary
probability measures with an exponential moment — the n-d analogue of the 1-d
"radial-limit foundation".  The proof reads off the drift along a radial probe
`r • u`: since `meanShift = kernelCentroid − probe` and the probe cancels in the
difference, zero drift makes the two kernel centroids agree at every finite `r`,
and passing `r → ∞` (existing `kernelCentroid_laplace_radial_tendsto`) turns the
kernel centroids into the exponential-tilt centroids.

Also recorded is `laplaceCompensatedWeight_monotone` (§4.6(a)): the compensated
weight `e^{(r-‖r•u-y‖)/τ}` is monotone in `r`, the moment-minimal ingredient of
the far-field limits.

The final section is milestone `L2`: an affine-isometric dimensional-reduction
lemma.  If two laws are pushed forward from a lower-dimensional Hilbert space
through an affine isometry, ambient zero drift reduces to zero drift in the
source.  The collinear corollary instantiates the source as `ℝ` and fires the
unconditional one-dimensional theorem `laplaceZeroDrift_identifies`.

Design record: `LaplaceHigherDim.md`, §4.6(a) and §4.8 (L1).
-/

open MeasureTheory Filter Topology
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-! ## Monotonicity of the compensated weight (§4.6(a)) -/

/-- The compensated weight `r ↦ e^{(r-‖r•u-y‖)/τ}` is **monotone**: the exponent
`r - ‖r•u - y‖` increases because `‖r'•u-y‖ - ‖r•u-y‖ ≤ ‖(r'-r)•u‖ = r'-r`.  This
is the moment-free ingredient behind the far-field limits (the weight increases
to its exponential-tilt limit, so monotone/dominated convergence applies without
any moment hypothesis on the normalizer). -/
lemma laplaceCompensatedWeight_monotone {τ : ℝ} (hτ : 0 < τ) (u y : E) (hu : ‖u‖ = 1) :
    Monotone (fun r : ℝ => laplaceCompensatedWeight τ r u y) := by
  intro r r' hrr
  simp only [laplaceCompensatedWeight]
  apply Real.exp_le_exp.mpr
  apply div_le_div_of_nonneg_right _ hτ.le
  have hchain : ‖r' • u - y‖ - ‖r • u - y‖ ≤ r' - r := by
    calc ‖r' • u - y‖ - ‖r • u - y‖
        ≤ ‖(r' • u - y) - (r • u - y)‖ := norm_sub_norm_le _ _
      _ = ‖(r' - r) • u‖ := by rw [sub_sub_sub_cancel_right, ← sub_smul]
      _ = (r' - r) * ‖u‖ := by
          rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (by linarith)]
      _ = r' - r := by rw [hu, mul_one]
  linarith

section FarField

variable [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]

omit [InnerProductSpace ℝ E] [CompleteSpace E] [SecondCountableTopology E] in
/-- An exponential moment `∫ e^{‖y‖/τ}‖y‖ < ∞` implies the first moment
`∫ ‖y‖ < ∞`, since `e^{‖y‖/τ} ≥ 1`. -/
lemma integrable_norm_of_integrable_exp_mul_norm {τ : ℝ} (hτ : 0 < τ) (P : Measure E)
    (h : Integrable (fun y => Real.exp (‖y‖ / τ) * ‖y‖) P) :
    Integrable (fun y : E => ‖y‖) P := by
  refine h.mono' continuous_norm.aestronglyMeasurable ?_
  filter_upwards with y
  rw [Real.norm_eq_abs, abs_of_nonneg (norm_nonneg _)]
  exact le_mul_of_one_le_left (norm_nonneg _)
    (Real.one_le_exp (div_nonneg (norm_nonneg _) hτ.le))

omit [CompleteSpace E] in
/-- The displacement-numerator integrand `laplaceKernel τ x · • ·` is integrable
against any measure with a first moment (`e^{-‖x-y‖/τ} ≤ 1`). -/
lemma laplaceKernel_smul_id_integrable_of_moment {τ : ℝ} (hτ : 0 < τ) (P : Measure E)
    (x : E) (hmom : Integrable (fun y : E => ‖y‖) P) :
    Integrable (fun y => laplaceKernel τ x y • y) P := by
  refine hmom.mono' ?_ ?_
  · apply Continuous.aestronglyMeasurable; unfold laplaceKernel; fun_prop
  · filter_upwards with y
    rw [norm_smul, laplaceKernel_eq_exp, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    have hle : Real.exp (-‖x - y‖ / τ) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      exact div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (norm_nonneg _)) hτ.le
    calc Real.exp (-‖x - y‖ / τ) * ‖y‖ ≤ 1 * ‖y‖ :=
          mul_le_mul_of_nonneg_right hle (norm_nonneg _)
      _ = ‖y‖ := one_mul _

/-- The paper's mean shift equals `kernelCentroid − probe` for any probability
measure with a first moment (general form of
`gaussian_meanShift_laplace_eq_kernelCentroid_sub`, no Gaussian hypothesis). -/
lemma meanShift_laplace_eq_kernelCentroid_sub_of_moment {τ : ℝ} (hτ : 0 < τ)
    (P : Measure E) [IsProbabilityMeasure P] (x : E)
    (hmom : Integrable (fun y : E => ‖y‖) P) :
    meanShift (laplaceKernel τ) P x = kernelCentroid (laplaceKernel τ) P x - x :=
  meanShift_eq_kernelCentroid_sub P (laplaceKernel τ) x
    (laplaceKernel_integrable P τ hτ x)
    (laplaceKernel_smul_id_integrable_of_moment hτ P x hmom)
    (laplaceKernelNormalizer_pos P τ hτ x).ne'

/-- **Far-field foundation (D2.a).**  Pointwise zero Laplace mean-shift drift
forces the exponential-tilt centroids of `P` and `Q` to agree in every unit
direction `u`, for arbitrary probability measures with the exponential moments
`∫ e^{‖y‖/τ}(1+‖y‖) < ∞`.  Proof: along the ray `r • u`, `meanShift =
kernelCentroid − r•u`, so zero drift makes the kernel centroids agree at every
`r`; letting `r → ∞` (via `kernelCentroid_laplace_radial_tendsto`) sends each to
its exponential-tilt centroid, and limits are unique.  This is the n-d analogue
of the 1-d radial-limit foundation; it is known to be *insufficient alone* (an
analytic difference of cumulant generating functions can have vanishing gradient
on a sphere), but it is shared infrastructure for the radial endgame (L5). -/
theorem zeroDrift_tiltedCentroid_eq (P Q : Measure E)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    {τ : ℝ} (hτ : 0 < τ) (u : E) (hu : ‖u‖ = 1)
    (hP₀ : Integrable (fun y => Real.exp (‖y‖ / τ)) P)
    (hP₁ : Integrable (fun y => Real.exp (‖y‖ / τ) * ‖y‖) P)
    (hQ₀ : Integrable (fun y => Real.exp (‖y‖ / τ)) Q)
    (hQ₁ : Integrable (fun y => Real.exp (‖y‖ / τ) * ‖y‖) Q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) P Q) :
    exponentialTiltCentroid P τ u = exponentialTiltCentroid Q τ u := by
  have hPmom := integrable_norm_of_integrable_exp_mul_norm hτ P hP₁
  have hQmom := integrable_norm_of_integrable_exp_mul_norm hτ Q hQ₁
  have hP := kernelCentroid_laplace_radial_tendsto P τ hτ u hu hP₀ hP₁
  have hQ := kernelCentroid_laplace_radial_tendsto Q τ hτ u hu hQ₀ hQ₁
  have heq : (fun r : ℝ => kernelCentroid (laplaceKernel τ) P (r • u)) =
      fun r : ℝ => kernelCentroid (laplaceKernel τ) Q (r • u) := by
    funext r
    have hms : meanShift (laplaceKernel τ) P (r • u) =
        meanShift (laplaceKernel τ) Q (r • u) := by
      have h := hzero (r • u)
      simp only [meanShiftDrift, sub_eq_zero] at h
      exact h
    rw [meanShift_laplace_eq_kernelCentroid_sub_of_moment hτ P (r • u) hPmom,
      meanShift_laplace_eq_kernelCentroid_sub_of_moment hτ Q (r • u) hQmom] at hms
    have h2 := congrArg (· + r • u) hms
    simpa using h2
  have hQ' : Tendsto (fun r : ℝ => kernelCentroid (laplaceKernel τ) P (r • u)) atTop
      (𝓝 (exponentialTiltCentroid Q τ u)) := by rw [heq]; exact hQ
  exact tendsto_nhds_unique hP hQ'

end FarField

/-! ## L2: dimensional reduction along affine isometries -/

section DimensionalReduction

variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
  [MeasurableSpace F] [BorelSpace F] [CompleteSpace F] [SecondCountableTopology F]
  [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]

/-- The affine embedding `z ↦ a + L z` associated to a linear isometry. -/
def affineIsometryEmbedding (a : E) (L : F →ₗᵢ[ℝ] E) : F → E :=
  fun z => a + L z

omit [MeasurableSpace F] [BorelSpace F] [CompleteSpace F] [SecondCountableTopology F]
  [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E] in
lemma continuous_affineIsometryEmbedding (a : E) (L : F →ₗᵢ[ℝ] E) :
    Continuous (affineIsometryEmbedding (F := F) a L) :=
  continuous_const.add L.toContinuousLinearMap.continuous

omit [MeasurableSpace F] [BorelSpace F] [CompleteSpace F] [SecondCountableTopology F]
  [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E] in
lemma affineIsometryEmbedding_sub (a : E) (L : F →ₗᵢ[ℝ] E) (x y : F) :
    affineIsometryEmbedding (F := F) a L x - affineIsometryEmbedding (F := F) a L y =
      L (x - y) := by
  simp [affineIsometryEmbedding, map_sub]

omit [MeasurableSpace F] [BorelSpace F] [CompleteSpace F] [SecondCountableTopology F]
  [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E] in
lemma norm_affineIsometryEmbedding_sub (a : E) (L : F →ₗᵢ[ℝ] E) (x y : F) :
    ‖affineIsometryEmbedding (F := F) a L x -
        affineIsometryEmbedding (F := F) a L y‖ = ‖x - y‖ := by
  rw [affineIsometryEmbedding_sub]
  exact L.norm_map (x - y)

omit [MeasurableSpace F] [BorelSpace F] [CompleteSpace F] [SecondCountableTopology F]
  [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E] in
lemma laplaceKernel_affineIsometryEmbedding (τ : ℝ) (a : E) (L : F →ₗᵢ[ℝ] E)
    (x y : F) :
    laplaceKernel τ (affineIsometryEmbedding (F := F) a L x)
        (affineIsometryEmbedding (F := F) a L y) =
      laplaceKernel τ x y := by
  rw [laplaceKernel_eq_exp, laplaceKernel_eq_exp,
    norm_affineIsometryEmbedding_sub]

/- Laplace normalizers commute with affine-isometric pushforward.  This is the
scalar half of L2: distances are preserved by `z ↦ a + L z`, so the mapped
kernel integral is literally the source kernel integral. -/
omit [CompleteSpace F] [SecondCountableTopology F] [CompleteSpace E]
  [SecondCountableTopology E] in
lemma kernelNormalizer_laplace_affineIsometryMap (τ : ℝ) (a : E) (L : F →ₗᵢ[ℝ] E)
    (μ : Measure F) [IsFiniteMeasure μ] (x : F) :
    kernelNormalizer (laplaceKernel τ)
        (μ.map (affineIsometryEmbedding (F := F) a L))
        (affineIsometryEmbedding (F := F) a L x) =
      kernelNormalizer (laplaceKernel τ) μ x := by
  unfold kernelNormalizer
  rw [integral_map]
  · refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
    exact laplaceKernel_affineIsometryEmbedding τ a L x y
  · exact (continuous_affineIsometryEmbedding (F := F) a L).aemeasurable
  · apply Continuous.aestronglyMeasurable
    unfold laplaceKernel
    fun_prop

/-- Laplace displacement numerators commute with affine-isometric pushforward:
the ambient numerator is the isometric image of the source numerator. -/
lemma laplaceDisplacementField_affineIsometryMap {τ : ℝ} (hτ : 0 < τ)
    (a : E) (L : F →ₗᵢ[ℝ] E) (μ : Measure F) [IsFiniteMeasure μ] (x : F) :
    laplaceDisplacementField τ
        (μ.map (affineIsometryEmbedding (F := F) a L))
        (affineIsometryEmbedding (F := F) a L x) =
      L (laplaceDisplacementField τ μ x) := by
  unfold laplaceDisplacementField
  rw [integral_map]
  · have hint :
        Integrable (fun y : F => laplaceKernel τ x y • (y - x)) μ :=
      integrable_laplaceDisplacementField_integrand hτ μ x
    change ∫ x_1 : F,
        laplaceKernel τ (affineIsometryEmbedding (F := F) a L x)
            (affineIsometryEmbedding (F := F) a L x_1) •
          (affineIsometryEmbedding (F := F) a L x_1 -
            affineIsometryEmbedding (F := F) a L x) ∂μ =
        L.toContinuousLinearMap (∫ y : F, laplaceKernel τ x y • (y - x) ∂μ)
    rw [← L.toContinuousLinearMap.integral_comp_comm hint]
    refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
    change laplaceKernel τ (affineIsometryEmbedding (F := F) a L x)
          (affineIsometryEmbedding (F := F) a L y) •
        (affineIsometryEmbedding (F := F) a L y -
          affineIsometryEmbedding (F := F) a L x) =
        L.toContinuousLinearMap (laplaceKernel τ x y • (y - x))
    rw [laplaceKernel_affineIsometryEmbedding τ a L x y,
      affineIsometryEmbedding_sub]
    simp
  · exact (continuous_affineIsometryEmbedding (F := F) a L).aemeasurable
  · apply Continuous.aestronglyMeasurable
    unfold laplaceKernel
    fun_prop

/-- Mean shifts commute with affine-isometric pushforward. -/
lemma meanShift_laplace_affineIsometryMap {τ : ℝ} (hτ : 0 < τ)
    (a : E) (L : F →ₗᵢ[ℝ] E) (μ : Measure F) [IsFiniteMeasure μ] (x : F) :
    meanShift (laplaceKernel τ)
        (μ.map (affineIsometryEmbedding (F := F) a L))
        (affineIsometryEmbedding (F := F) a L x) =
      L (meanShift (laplaceKernel τ) μ x) := by
  rw [meanShift_laplace_eq, meanShift_laplace_eq,
    kernelNormalizer_laplace_affineIsometryMap,
    laplaceDisplacementField_affineIsometryMap hτ]
  simp

/-- **Dimensional reduction (L2).**  If two measures are affine-isometric
pushforwards from `F` into `E`, then ambient zero drift implies zero drift in
the source space.  This is the reusable WLOG lemma: any lower-dimensional
configuration can be studied in its intrinsic coordinates. -/
theorem zeroDrift_of_affineIsometryMap_zeroDrift {τ : ℝ} (hτ : 0 < τ)
    (a : E) (L : F →ₗᵢ[ℝ] E) (μ ν : Measure F)
    [IsProbabilityMeasure μ] [IsProbabilityMeasure ν]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (μ.map (affineIsometryEmbedding (F := F) a L))
      (ν.map (affineIsometryEmbedding (F := F) a L))) :
    ZeroDrift (meanShiftDrift (laplaceKernel τ)) μ ν := by
  intro x
  have h := hzero (affineIsometryEmbedding (F := F) a L x)
  simp only [meanShiftDrift, sub_eq_zero] at h ⊢
  rw [meanShift_laplace_affineIsometryMap hτ a L μ x,
    meanShift_laplace_affineIsometryMap hτ a L ν x] at h
  exact L.injective h

/-- The one-dimensional linear isometry `t ↦ t • u` generated by a unit vector. -/
def realLineLinearIsometry (u : E) (hu : ‖u‖ = 1) : ℝ →ₗᵢ[ℝ] E :=
  LinearIsometry.mk
    ({ toFun := fun t : ℝ => t • u
       map_add' := by intro x y; rw [add_smul]
       map_smul' := by intro c x; simp [smul_eq_mul, smul_smul] } : ℝ →ₗ[ℝ] E)
    (by
      intro t
      change ‖t • u‖ = ‖t‖
      rw [norm_smul, hu, mul_one])

/-- The affine parametrization of the line through `a` in unit direction `u`. -/
def affineLineEmbedding (a u : E) : ℝ → E :=
  fun t => a + t • u

omit [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E] in
lemma affineIsometryEmbedding_realLineLinearIsometry (a u : E) (hu : ‖u‖ = 1) :
    affineIsometryEmbedding (F := ℝ) a (realLineLinearIsometry u hu) =
      affineLineEmbedding a u := by
  rfl

/-- **Collinear corollary (L2).**  If two ambient laws are presented as
pushforwards of one-dimensional probability measures along the same affine
line, ambient zero Laplace drift identifies them.  This is the first fully
unconditional higher-dimensional `ℓ²` statement supplied by the 1-d theorem. -/
theorem laplaceZeroDrift_identifies_of_collinear
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (a u : E) (hu : ‖u‖ = 1)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (p.map (affineLineEmbedding a u))
      (q.map (affineLineEmbedding a u))) :
    p.map (affineLineEmbedding a u) = q.map (affineLineEmbedding a u) := by
  let L := realLineLinearIsometry u hu
  have hzero' : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (p.map (affineIsometryEmbedding (F := ℝ) a L))
      (q.map (affineIsometryEmbedding (F := ℝ) a L)) := by
    simpa [L, affineIsometryEmbedding_realLineLinearIsometry]
      using hzero
  have h1 : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q :=
    zeroDrift_of_affineIsometryMap_zeroDrift (F := ℝ) hτ a L p q hzero'
  have hpq : p = q := laplaceZeroDrift_identifies τ hτ p q h1
  rw [hpq]

end DimensionalReduction

end DriftingIdentifiability
