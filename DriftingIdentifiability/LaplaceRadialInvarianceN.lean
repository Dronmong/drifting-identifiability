import DriftingIdentifiability.LaplaceRadialConverseN
import DriftingIdentifiability.LaplaceRadialFourier
import DriftingIdentifiability.LaplaceRadialFarField

/-!
# General-dimensional radial Laplace converse: global endgame

The manifest Haar-sphere construction makes the global radiality step much
shorter than the earlier chart-based `n = 3` proof.  The standard directional
coordinate marginal of the uniform sphere shows directly that a shell
average depends only on the probe norm.  Ray proportionality can therefore be
fed to the already-certified Euclidean Laplace smoothing injectivity theorem.
-/

open MeasureTheory Filter Topology Set Metric
open scoped RealInnerProductSpace ENNReal

namespace DriftingIdentifiability
open Paper

private lemma norm_inv_smul_self
    {n : ℕ} (x : EuclideanSpace ℝ (Fin n)) (hx : x ≠ 0) :
    ‖‖x‖⁻¹ • x‖ = 1 := by
  rw [norm_smul, Real.norm_eq_abs, abs_inv, abs_of_nonneg (norm_nonneg x)]
  exact inv_mul_cancel₀ (norm_ne_zero_iff.mpr hx)

/-- The uniform-shell Laplace average at an arbitrary nonzero probe is the
same zonal shell profile evaluated at the probe norm. -/
lemma integral_uniformSphere_laplaceKernel_eq_shellZN_norm
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ)
    (x : EuclideanSpace ℝ (Fin n)) (hx : x ≠ 0) (s : ℝ) :
    (∫ ω, laplaceKernel τ x (s • (ω : EuclideanSpace ℝ (Fin n)))
        ∂(uniformSphereMeasure n)) = shellZN n τ ‖x‖ s := by
  let v : EuclideanSpace ℝ (Fin n) := ‖x‖⁻¹ • x
  have hv : ‖v‖ = 1 := norm_inv_smul_self x hx
  let g : ℝ → ℝ := fun u =>
    Real.exp (-(1 / τ) * Real.sqrt (‖x‖ ^ 2 + s ^ 2 - 2 * ‖x‖ * s * u))
  have hg : Continuous g := by
    dsimp [g]
    fun_prop
  have hdist : ∀ ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1,
      ‖x - s • (ω : EuclideanSpace ℝ (Fin n))‖ =
        Real.sqrt (‖x‖ ^ 2 + s ^ 2 -
          2 * ‖x‖ * s * inner ℝ v (ω : EuclideanSpace ℝ (Fin n))) := by
    intro ω
    rw [← Real.sqrt_sq (norm_nonneg _), norm_sub_sq_real,
      norm_smul, norm_sphere_coe_eq_one, Real.norm_eq_abs, mul_one, sq_abs]
    congr 1
    have hxnorm : ‖x‖ ≠ 0 := norm_ne_zero_iff.mpr hx
    have hvinner : inner ℝ v (ω : EuclideanSpace ℝ (Fin n)) =
        ‖x‖⁻¹ * inner ℝ x (ω : EuclideanSpace ℝ (Fin n)) := by
      simp [v, real_inner_smul_left]
    rw [real_inner_smul_right, hvinner]
    field_simp [hxnorm]
    ring
  have hpoint : ∀ ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1,
      laplaceKernel τ x (s • (ω : EuclideanSpace ℝ (Fin n))) =
        g (inner ℝ v (ω : EuclideanSpace ℝ (Fin n))) := by
    intro ω
    simp only [laplaceKernel, g]
    rw [hdist ω]
  calc
    (∫ ω, laplaceKernel τ x (s • (ω : EuclideanSpace ℝ (Fin n)))
        ∂(uniformSphereMeasure n)) =
        ∫ ω, g (inner ℝ v (ω : EuclideanSpace ℝ (Fin n)))
          ∂(uniformSphereMeasure n) :=
      integral_congr_ae (Filter.Eventually.of_forall hpoint)
    _ = (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * g u :=
      integral_uniformSphere_directional hn v hv g hg.continuousOn
    _ = shellZN n τ ‖x‖ s := by rfl

/-- The Laplace normalizer of a genuine Haar radial mixture is a radial
function, with profile exactly `radialRayZN`. -/
theorem kernelNormalizer_radialMixtureN_radial
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (x : EuclideanSpace ℝ (Fin n)) :
    kernelNormalizer (laplaceKernel τ) (radialMixtureN n ν) x =
      radialRayZN n τ ν ‖x‖ := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  by_cases hx : x = 0
  · subst x
    have hray := radialRayZN_eq_kernelNormalizer hn hτ ν 0
    simpa [radialRayProbeN] using hray.symm
  · have hf : Integrable (fun y => laplaceKernel τ x y) (radialMixtureN n ν) :=
      laplaceKernel_integrable (radialMixtureN n ν) τ hτ x
    rw [kernelNormalizer, integral_radialMixtureN hn ν hf, radialRayZN]
    apply integral_congr_ae
    exact Filter.Eventually.of_forall fun s =>
      integral_uniformSphere_laplaceKernel_eq_shellZN_norm hn τ x hx s

/-- Ray proportionality extended to the origin, with a positive constant. -/
lemma radialRayZN_proportional_nonneg
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable (fun s : ℝ => s ^ (n - 1)) νp)
    (hmq : Integrable (fun s : ℝ => s ^ (n - 1)) νq)
    (hslackp : RadialSlackN n (by omega) τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq)) :
    ∃ c : ℝ, 0 < c ∧ ∀ r : ℝ, 0 ≤ r →
      radialRayZN n τ νp r = c * radialRayZN n τ νq r := by
  obtain ⟨c, hprop⟩ := radialRayZN_proportional hn τ hτ νp νq hsp hsq
    hmp hmq hslackp hzero
  have hcpos : 0 < c := by
    have h1 := hprop 1 one_pos
    have hp1 := radialRayZN_pos hn τ hτ νp 1
    have hq1 := radialRayZN_pos hn τ hτ νq 1
    nlinarith
  refine ⟨c, hcpos, fun r hr => ?_⟩
  rcases eq_or_lt_of_le hr with h0 | hpos
  · subst r
    have hf : Tendsto (radialRayZN n τ νp) (nhdsWithin (0 : ℝ) (Ioi 0))
        (nhds (radialRayZN n τ νp 0)) :=
      ((continuous_radialRayZN hn τ hτ νp).tendsto 0).mono_left nhdsWithin_le_nhds
    have hg : Tendsto (fun r => c * radialRayZN n τ νq r)
        (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds (c * radialRayZN n τ νq 0)) :=
      (((continuous_radialRayZN hn τ hτ νq).const_mul c).tendsto 0).mono_left
        nhdsWithin_le_nhds
    have heq : radialRayZN n τ νp =ᶠ[nhdsWithin (0 : ℝ) (Ioi 0)]
        fun r => c * radialRayZN n τ νq r := by
      filter_upwards [self_mem_nhdsWithin] with r hr'
      exact hprop r hr'
    exact tendsto_nhds_unique_of_eventuallyEq hf hg heq
  · exact hprop r hpos

/-- **G1 headline at center zero.**  Zero Laplace drift identifies genuine
Haar radial mixtures in every dimension `n ≥ 3`, under `RadialSlackN` and
finite natural `(n-1)` radial moments. -/
theorem laplaceZeroDrift_identifies_of_radialMixtureN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable (fun s : ℝ => s ^ (n - 1)) νp)
    (hmq : Integrable (fun s : ℝ => s ^ (n - 1)) νq)
    (hslackp : RadialSlackN n (by omega) τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq)) :
    radialMixtureN n νp = radialMixtureN n νq := by
  letI hp := radialMixtureN_isProbabilityMeasure hn νp
  letI hq := radialMixtureN_isProbabilityMeasure hn νq
  obtain ⟨c, hcpos, hprop⟩ := radialRayZN_proportional_nonneg hn τ hτ νp νq
    hsp hsq hmp hmq hslackp hzero
  have hZ : ∀ x : EuclideanSpace ℝ (Fin n),
      kernelNormalizer (laplaceKernel τ) (radialMixtureN n νp) x =
        kernelNormalizer (laplaceKernel τ)
          ((ENNReal.ofReal c) • radialMixtureN n νq) x := by
    intro x
    have hsm : kernelNormalizer (laplaceKernel τ)
        ((ENNReal.ofReal c) • radialMixtureN n νq) x =
        c * kernelNormalizer (laplaceKernel τ) (radialMixtureN n νq) x := by
      have h1 := MeasureTheory.integral_smul_measure
        (μ := radialMixtureN n νq) (f := fun y => laplaceKernel τ x y)
        (c := ENNReal.ofReal c)
      rw [smul_eq_mul] at h1
      rw [kernelNormalizer, h1, ENNReal.toReal_ofReal hcpos.le]
      rfl
    rw [hsm, kernelNormalizer_radialMixtureN_radial hn τ hτ νp x,
      kernelNormalizer_radialMixtureN_radial hn τ hτ νq x]
    exact hprop ‖x‖ (norm_nonneg x)
  haveI hfin : IsFiniteMeasure ((ENNReal.ofReal c) • radialMixtureN n νq) := by
    constructor
    rw [Measure.smul_apply, smul_eq_mul, measure_univ, mul_one]
    exact ENNReal.ofReal_lt_top
  have hL4 := laplaceSmoothingInjective_euclideanSpace (ι := Fin n) τ hτ
  have hpq := hL4 (radialMixtureN n νp) ((ENNReal.ofReal c) • radialMixtureN n νq)
    inferInstance hfin hZ
  have hmass : (radialMixtureN n νp) univ =
      ((ENNReal.ofReal c) • radialMixtureN n νq) univ := by rw [hpq]
  rw [measure_univ, Measure.smul_apply, smul_eq_mul, measure_univ, mul_one] at hmass
  have hc1 : c = 1 := ENNReal.ofReal_eq_one.mp hmass.symm
  rw [hpq, hc1, ENNReal.ofReal_one, one_smul]

/-- **G1 headline at an arbitrary common center.**  Translation is an affine
isometry, so centered zero drift reduces exactly to the center-zero theorem
and the identified source measures can be pushed forward again. -/
theorem laplaceZeroDrift_identifies_of_radialMixtureN_centered
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (c : EuclideanSpace ℝ (Fin n))
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable (fun s : ℝ => s ^ (n - 1)) νp)
    (hmq : Integrable (fun s : ℝ => s ^ (n - 1)) νq)
    (hslackp : RadialSlackN n (by omega) τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN_centered n νp c) (radialMixtureN_centered n νq c)) :
    radialMixtureN_centered n νp c = radialMixtureN_centered n νq c := by
  letI hp := radialMixtureN_isProbabilityMeasure hn νp
  letI hq := radialMixtureN_isProbabilityMeasure hn νq
  let L : EuclideanSpace ℝ (Fin n) →ₗᵢ[ℝ] EuclideanSpace ℝ (Fin n) :=
    (LinearIsometryEquiv.refl ℝ (EuclideanSpace ℝ (Fin n))).toLinearIsometry
  have hemb : affineIsometryEmbedding c L =
      fun x : EuclideanSpace ℝ (Fin n) => c + x := by
    funext x
    simp [affineIsometryEmbedding, L]
  have hzeroMap : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      ((radialMixtureN n νp).map (affineIsometryEmbedding c L))
      ((radialMixtureN n νq).map (affineIsometryEmbedding c L)) := by
    rw [hemb]
    simpa [radialMixtureN_centered] using hzero
  have hzeroRoot : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq) :=
    zeroDrift_of_affineIsometryMap_zeroDrift hτ c L
      (radialMixtureN n νp) (radialMixtureN n νq) hzeroMap
  have hpq := laplaceZeroDrift_identifies_of_radialMixtureN hn τ hτ νp νq
    hsp hsq hmp hmq hslackp hzeroRoot
  simp [radialMixtureN_centered, hpq]

end DriftingIdentifiability
