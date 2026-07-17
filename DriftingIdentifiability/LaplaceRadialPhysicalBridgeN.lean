import DriftingIdentifiability.LaplaceRadialZonalBridgeN
import DriftingIdentifiability.LaplaceRadialIntegrabilityN

/-!
# The physical general-`n` ray bridge

This file connects the genuine Haar radial mixture to the one-dimensional
zonal ray profiles.  The only geometric input is carried explicitly by
`ZonalSphereBridge`; no coordinate-density fact is assumed globally.
-/

open MeasureTheory Filter Topology Set Metric
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

/-- The first-coordinate ray in `ℝⁿ`. -/
noncomputable def radialRayProbeN (n : ℕ) (hn : 0 < n) (r : ℝ) :
    EuclideanSpace ℝ (Fin n) :=
  r • EuclideanSpace.basisFun (Fin n) ℝ ⟨0, hn⟩

@[simp] lemma norm_radialRayProbeN (n : ℕ) (hn : 0 < n) (r : ℝ) :
    ‖radialRayProbeN n hn r‖ ^ 2 = r ^ 2 := by
  have hb : ‖(EuclideanSpace.basisFun (Fin n) ℝ) ⟨0, hn⟩‖ = 1 :=
    (EuclideanSpace.basisFun (Fin n) ℝ).norm_eq_one ⟨0, hn⟩
  rw [radialRayProbeN, norm_smul, hb]
  simp only [mul_one, Real.norm_eq_abs, sq_abs]

@[simp] lemma radialRayProbeN_first (n : ℕ) (hn : 0 < n) (r : ℝ) :
    radialRayProbeN n hn r ⟨0, hn⟩ = r := by
  simp [radialRayProbeN]

@[simp] lemma inner_radialRayProbeN_smul_sphere
    {n : ℕ} (hn : 0 < n) (r s : ℝ)
    (ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1) :
    inner ℝ (radialRayProbeN n hn r) (s • (ω : EuclideanSpace ℝ (Fin n)))
      = r * s * (ω : EuclideanSpace ℝ (Fin n)) ⟨0, hn⟩ := by
  simp [radialRayProbeN, PiLp.inner_apply]
  ring

lemma norm_sphere_coe_eq_one
    {n : ℕ} (ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1) :
    ‖(ω : EuclideanSpace ℝ (Fin n))‖ = 1 := by
  have h := ω.property
  rw [Metric.mem_sphere, dist_zero_right] at h
  exact h

/-- The probe-to-shell distance depends only on the first sphere coordinate. -/
lemma norm_radialRayProbeN_sub_smul_sphere
    {n : ℕ} (hn : 0 < n) (r s : ℝ)
    (ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1) :
    ‖radialRayProbeN n hn r - s • (ω : EuclideanSpace ℝ (Fin n))‖
      = shellDist r s (sphereFirstCoord n hn ω) := by
  rw [shellDist, ← Real.sqrt_sq (norm_nonneg _)]
  congr 1
  rw [norm_sub_sq_real, norm_radialRayProbeN,
    norm_smul, norm_sphere_coe_eq_one, Real.norm_eq_abs,
    inner_radialRayProbeN_smul_sphere hn]
  simp only [mul_one, sq_abs]
  simp [sphereFirstCoord]
  ring

/-! ## Zonal transfer to the genuine Haar shell -/

private lemma continuous_shellKernelN (τ r s : ℝ) :
    Continuous fun u : ℝ => Real.exp (-(1 / τ) * shellDist r s u) :=
  Real.continuous_exp.comp
    ((continuous_shellDist_u r s).const_mul (-(1 / τ)))

/-- Given the explicitly supplied spherical-slicing identity, the genuine
Haar-shell average of the Laplace kernel is exactly the zonal shell profile. -/
lemma integral_uniformSphere_laplaceKernel_eq_shellZN_of_bridge
    {n : ℕ} {hn : 3 ≤ n} (hbridge : ZonalSphereBridge n hn)
    (τ r s : ℝ) :
    (∫ ω, laplaceKernel τ (radialRayProbeN n (by omega) r)
        (s • (ω : EuclideanSpace ℝ (Fin n))) ∂(uniformSphereMeasure n))
      = shellZN n τ r s := by
  have hpoint : ∀ ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1,
      laplaceKernel τ (radialRayProbeN n (by omega) r)
          (s • (ω : EuclideanSpace ℝ (Fin n)))
        = Real.exp (-(1 / τ) * shellDist r s
            (sphereFirstCoord n (by omega) ω)) := by
    intro ω
    unfold laplaceKernel
    rw [norm_radialRayProbeN_sub_smul_sphere (by omega)]
  calc
    (∫ ω, laplaceKernel τ (radialRayProbeN n (by omega) r)
        (s • (ω : EuclideanSpace ℝ (Fin n))) ∂(uniformSphereMeasure n))
      = ∫ ω, Real.exp (-(1 / τ) * shellDist r s
          (sphereFirstCoord n (by omega) ω)) ∂(uniformSphereMeasure n) :=
        integral_congr_ae (Filter.Eventually.of_forall hpoint)
    _ = (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u) := by
      exact integral_uniformSphere_zonal_of_bridge hbridge _
        (continuous_shellKernelN τ r s)
    _ = shellZN n τ r s := rfl

/-- The normalizer of the genuine Haar radial mixture equals the general-`n`
zonal ray normalizer, once the (still explicit) spherical-slicing bridge is
provided.  This is the measure-to-ray connection needed by the later system
layer. -/
theorem radialRayZN_eq_kernelNormalizer_of_zonalBridge
    {n : ℕ} {hn : 3 ≤ n} (hbridge : ZonalSphereBridge n hn)
    {τ : ℝ} (hτ : 0 < τ) (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayZN n τ ν r =
      kernelNormalizer (laplaceKernel τ) (radialMixtureN n ν)
        (radialRayProbeN n (by omega) r) := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  rw [kernelNormalizer,
    integral_radialMixtureN hn ν
      (laplaceKernel_integrable (radialMixtureN n ν) τ hτ
        (radialRayProbeN n (by omega) r))]
  exact integral_congr_ae (Filter.Eventually.of_forall fun s =>
    (integral_uniformSphere_laplaceKernel_eq_shellZN_of_bridge hbridge τ r s).symm)

/-- The genuine radial-mixture normalizer is the zonal ray normalizer.  The
classical sphere-coordinate theorem is supplied through the reviewed standard
bridge, so callers do not need to thread a geometric hypothesis. -/
theorem radialRayZN_eq_kernelNormalizer
    {n : ℕ} (hn : 3 ≤ n) {τ : ℝ} (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayZN n τ ν r =
      kernelNormalizer (laplaceKernel τ) (radialMixtureN n ν)
        (radialRayProbeN n (by omega) r) :=
  radialRayZN_eq_kernelNormalizer_of_zonalBridge
    (zonalSphereBridge_standard n hn) hτ ν r

private lemma continuous_shellCompanionN (τ r s : ℝ) :
    Continuous fun u : ℝ =>
      (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u) :=
  (continuous_const.add (continuous_shellDist_u r s)).mul
    (continuous_shellKernelN τ r s)

/-- Given the spherical-slicing bridge, the genuine Haar-shell average of the
companion kernel is the zonal companion shell profile. -/
lemma integral_uniformSphere_laplaceCompanion_eq_shellCN_of_bridge
    {n : ℕ} {hn : 3 ≤ n} (hbridge : ZonalSphereBridge n hn)
    (τ r s : ℝ) :
    (∫ ω, laplaceCompanionKernel τ (radialRayProbeN n (by omega) r)
        (s • (ω : EuclideanSpace ℝ (Fin n))) ∂(uniformSphereMeasure n))
      = shellCN n τ r s := by
  have hpoint : ∀ ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1,
      laplaceCompanionKernel τ (radialRayProbeN n (by omega) r)
          (s • (ω : EuclideanSpace ℝ (Fin n)))
        = (τ + shellDist r s (sphereFirstCoord n (by omega) ω)) *
          Real.exp (-(1 / τ) * shellDist r s
            (sphereFirstCoord n (by omega) ω)) := by
    intro ω
    unfold laplaceCompanionKernel laplaceKernel
    rw [norm_radialRayProbeN_sub_smul_sphere (by omega)]
  calc
    (∫ ω, laplaceCompanionKernel τ (radialRayProbeN n (by omega) r)
        (s • (ω : EuclideanSpace ℝ (Fin n))) ∂(uniformSphereMeasure n))
      = ∫ ω, (τ + shellDist r s (sphereFirstCoord n (by omega) ω)) *
          Real.exp (-(1 / τ) * shellDist r s
            (sphereFirstCoord n (by omega) ω)) ∂(uniformSphereMeasure n) :=
        integral_congr_ae (Filter.Eventually.of_forall hpoint)
    _ = (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * ((τ + shellDist r s u) *
          Real.exp (-(1 / τ) * shellDist r s u)) := by
      exact integral_uniformSphere_zonal_of_bridge hbridge _
        (continuous_shellCompanionN τ r s)
    _ = shellCN n τ r s := rfl

/-- The companion normalizer of the genuine radial mixture equals the zonal
ray companion profile, subject only to the visible standard integrability
premise for that bounded smoothing. -/
theorem radialRayCN_eq_companionNormalizer_of_zonalBridge
    {n : ℕ} {hn : 3 ≤ n} (hbridge : ZonalSphereBridge n hn)
    (τ : ℝ) (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ)
    (hcomp : Integrable
      (fun y => laplaceCompanionKernel τ (radialRayProbeN n (by omega) r) y)
      (radialMixtureN n ν)) :
    radialRayCN n τ ν r =
      kernelNormalizer (laplaceCompanionKernel τ) (radialMixtureN n ν)
        (radialRayProbeN n (by omega) r) := by
  rw [kernelNormalizer, integral_radialMixtureN hn ν hcomp]
  exact integral_congr_ae (Filter.Eventually.of_forall fun s =>
    (integral_uniformSphere_laplaceCompanion_eq_shellCN_of_bridge hbridge τ r s).symm)

/-- Standard-bridge form of the companion-normalizer identity. -/
theorem radialRayCN_eq_companionNormalizer
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ)
    (hcomp : Integrable
      (fun y => laplaceCompanionKernel τ (radialRayProbeN n (by omega) r) y)
      (radialMixtureN n ν)) :
    radialRayCN n τ ν r =
      kernelNormalizer (laplaceCompanionKernel τ) (radialMixtureN n ν)
        (radialRayProbeN n (by omega) r) :=
  radialRayCN_eq_companionNormalizer_of_zonalBridge
    (zonalSphereBridge_standard n hn) τ ν r hcomp

private lemma continuous_shellDisplacementN (τ r s : ℝ) :
    Continuous fun u : ℝ =>
      Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r) :=
  (continuous_shellKernelN τ r s).mul
    ((continuous_const.mul continuous_id).sub continuous_const)

/-- The differentiated axial payload has a removable collision singularity
on the physical coordinate interval.  Its absolute value is squeezed by the
continuous shell distance. -/
private lemma continuousOn_shellAxialSqDivN (r s : ℝ) :
    ContinuousOn (fun u => shellAxial r s u ^ 2 / shellDist r s u)
      (Icc (-1 : ℝ) 1) := by
  intro u hu
  rcases eq_or_ne (shellDist r s u) 0 with hd0 | hdne
  · have hnonneg : ∀ᶠ v in 𝓝[Icc (-1 : ℝ) 1] u,
        0 ≤ shellAxial r s v ^ 2 / shellDist r s v :=
      Filter.Eventually.of_forall fun v => shellAxial_sq_div_shellDist_nonneg
    have hupper : ∀ᶠ v in 𝓝[Icc (-1 : ℝ) 1] u,
        shellAxial r s v ^ 2 / shellDist r s v ≤ shellDist r s v := by
      filter_upwards [self_mem_nhdsWithin] with v hv
      have hv2 : v ^ 2 ≤ 1 := by nlinarith [hv.1, hv.2]
      exact shellAxial_sq_div_shellDist_le hv2
    have hdlim : Tendsto (shellDist r s) (𝓝[Icc (-1 : ℝ) 1] u) (𝓝 0) := by
      have h : ContinuousWithinAt (shellDist r s) (Icc (-1 : ℝ) 1) u :=
        (continuous_shellDist_u r s).continuousWithinAt
      change Tendsto (shellDist r s) (𝓝[Icc (-1 : ℝ) 1] u)
        (𝓝 (shellDist r s u)) at h
      rw [hd0] at h
      exact h
    have hlim : Tendsto
        (fun v => shellAxial r s v ^ 2 / shellDist r s v)
        (𝓝[Icc (-1 : ℝ) 1] u) (𝓝 0) :=
      squeeze_zero' hnonneg hupper hdlim
    change Tendsto (fun v => shellAxial r s v ^ 2 / shellDist r s v)
      (𝓝[Icc (-1 : ℝ) 1] u)
      (𝓝 (shellAxial r s u ^ 2 / shellDist r s u))
    rw [hd0, div_zero]
    exact hlim
  · exact (((continuous_const.mul continuous_id).sub continuous_const).pow 2).continuousAt.div
      (continuous_shellDist_u r s).continuousAt hdne |>.continuousWithinAt

private lemma continuousOn_shellQPayloadN (τ r s : ℝ) :
    ContinuousOn (fun u => Real.exp (-(1 / τ) * shellDist r s u) *
      (shellAxial r s u ^ 2 / shellDist r s u)) (Icc (-1 : ℝ) 1) :=
  (continuous_shellKernelN τ r s).continuousOn.mul
    (continuousOn_shellAxialSqDivN r s)

/-- The Haar-shell integral of the `X₀²/d` derivative payload is exactly the
zonal `shellQN` profile. -/
lemma integral_uniformSphere_laplaceAxialSqDiv_eq_shellQN
    {n : ℕ} (hn : 3 ≤ n) (τ r s : ℝ) :
    (∫ ω, laplaceKernel τ (radialRayProbeN n (by omega) r)
          (s • (ω : EuclideanSpace ℝ (Fin n))) *
        (((s • (ω : EuclideanSpace ℝ (Fin n))) ⟨0, by omega⟩ - r) ^ 2 /
          ‖radialRayProbeN n (by omega) r -
            s • (ω : EuclideanSpace ℝ (Fin n))‖)
        ∂(uniformSphereMeasure n)) = shellQN n τ r s := by
  have hpoint : ∀ ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1,
      laplaceKernel τ (radialRayProbeN n (by omega) r)
          (s • (ω : EuclideanSpace ℝ (Fin n))) *
        (((s • (ω : EuclideanSpace ℝ (Fin n))) ⟨0, by omega⟩ - r) ^ 2 /
          ‖radialRayProbeN n (by omega) r -
            s • (ω : EuclideanSpace ℝ (Fin n))‖) =
      Real.exp (-(1 / τ) * shellDist r s
          (sphereFirstCoord n (by omega) ω)) *
        (shellAxial r s (sphereFirstCoord n (by omega) ω) ^ 2 /
          shellDist r s (sphereFirstCoord n (by omega) ω)) := by
    intro ω
    unfold laplaceKernel shellAxial sphereFirstCoord
    rw [norm_radialRayProbeN_sub_smul_sphere (by omega)]
    simp [PiLp.smul_apply, smul_eq_mul, sphereFirstCoord]
  calc
    (∫ ω, laplaceKernel τ (radialRayProbeN n (by omega) r)
          (s • (ω : EuclideanSpace ℝ (Fin n))) *
        (((s • (ω : EuclideanSpace ℝ (Fin n))) ⟨0, by omega⟩ - r) ^ 2 /
          ‖radialRayProbeN n (by omega) r -
            s • (ω : EuclideanSpace ℝ (Fin n))‖)
        ∂(uniformSphereMeasure n)) =
      ∫ ω, Real.exp (-(1 / τ) * shellDist r s
          (sphereFirstCoord n (by omega) ω)) *
        (shellAxial r s (sphereFirstCoord n (by omega) ω) ^ 2 /
          shellDist r s (sphereFirstCoord n (by omega) ω))
        ∂(uniformSphereMeasure n) :=
      integral_congr_ae (Filter.Eventually.of_forall hpoint)
    _ = (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellAxial r s u ^ 2 / shellDist r s u)) := by
      exact integral_uniformSphere_zonalOn_of_bridge
        (zonalSphereBridge_standard n hn) _ (continuousOn_shellQPayloadN τ r s)
    _ = shellQN n τ r s := rfl

/-- The first-coordinate Haar-shell integral of the Laplace displacement is
the zonal axial displacement profile. -/
lemma integral_uniformSphere_laplaceDisplacementCoord_eq_shellDN_of_bridge
    {n : ℕ} {hn : 3 ≤ n} (hbridge : ZonalSphereBridge n hn)
    (τ r s : ℝ) :
    (∫ ω, (laplaceWeightedDisplacement τ (radialRayProbeN n (by omega) r)
        (s • (ω : EuclideanSpace ℝ (Fin n)))) ⟨0, by omega⟩
        ∂(uniformSphereMeasure n)) = shellDN n τ r s := by
  have hpoint : ∀ ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1,
      (laplaceWeightedDisplacement τ (radialRayProbeN n (by omega) r)
        (s • (ω : EuclideanSpace ℝ (Fin n)))) ⟨0, by omega⟩
        = Real.exp (-(1 / τ) * shellDist r s
            (sphereFirstCoord n (by omega) ω)) *
          (s * sphereFirstCoord n (by omega) ω - r) := by
    intro ω
    simp only [laplaceWeightedDisplacement, PiLp.smul_apply, PiLp.sub_apply,
      smul_eq_mul]
    rw [radialRayProbeN_first]
    unfold laplaceKernel
    rw [norm_radialRayProbeN_sub_smul_sphere (by omega)]
    simp [sphereFirstCoord]
  calc
    (∫ ω, (laplaceWeightedDisplacement τ (radialRayProbeN n (by omega) r)
        (s • (ω : EuclideanSpace ℝ (Fin n)))) ⟨0, by omega⟩
        ∂(uniformSphereMeasure n))
      = ∫ ω, Real.exp (-(1 / τ) * shellDist r s
          (sphereFirstCoord n (by omega) ω)) *
          (s * sphereFirstCoord n (by omega) ω - r)
          ∂(uniformSphereMeasure n) :=
        integral_congr_ae (Filter.Eventually.of_forall hpoint)
    _ = (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) *
          (s * u - r)) := by
      exact integral_uniformSphere_zonal_of_bridge hbridge _
        (continuous_shellDisplacementN τ r s)
    _ = shellDN n τ r s := rfl

/-- The scalar first-coordinate displacement integral of the genuine radial
mixture equals the zonal ray displacement profile whenever that scalar
integrand is integrable. -/
theorem radialRayDN_eq_displacementCoord_of_zonalBridge
    {n : ℕ} {hn : 3 ≤ n} (hbridge : ZonalSphereBridge n hn)
    (τ : ℝ) (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ)
    (hdisp : Integrable
      (fun y => (laplaceWeightedDisplacement τ (radialRayProbeN n (by omega) r) y)
        ⟨0, by omega⟩) (radialMixtureN n ν)) :
    radialRayDN n τ ν r =
      ∫ y, (laplaceWeightedDisplacement τ (radialRayProbeN n (by omega) r) y)
        ⟨0, by omega⟩ ∂(radialMixtureN n ν) := by
  rw [integral_radialMixtureN hn ν hdisp]
  exact integral_congr_ae (Filter.Eventually.of_forall fun s =>
    (integral_uniformSphere_laplaceDisplacementCoord_eq_shellDN_of_bridge
      hbridge τ r s).symm)

/-- Standard-bridge form of the axial displacement identity. -/
theorem radialRayDN_eq_displacementCoord
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ)
    (hdisp : Integrable
      (fun y => (laplaceWeightedDisplacement τ (radialRayProbeN n (by omega) r) y)
        ⟨0, by omega⟩) (radialMixtureN n ν)) :
    radialRayDN n τ ν r =
      ∫ y, (laplaceWeightedDisplacement τ (radialRayProbeN n (by omega) r) y)
        ⟨0, by omega⟩ ∂(radialMixtureN n ν) :=
  radialRayDN_eq_displacementCoord_of_zonalBridge
    (zonalSphereBridge_standard n hn) τ ν r hdisp

end DriftingIdentifiability
