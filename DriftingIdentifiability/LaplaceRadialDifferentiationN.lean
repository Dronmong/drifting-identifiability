import DriftingIdentifiability.LaplaceRadialPhysicalBridgeN
import Mathlib.Analysis.Convex.Measure

/-!
# General-dimensional radial ray differentiation

This file starts the structural part of G1 after the reviewed spherical
coordinate bridge.  It proves the pointwise calculus used to differentiate
the Laplace ray profiles in every dimension `n ≥ 3`.  No derivative,
propagation, or identifiability statement is assumed.
-/

open MeasureTheory Filter Topology Set Metric
open scoped RealInnerProductSpace Pointwise

namespace DriftingIdentifiability

open Paper

/-! ## Dimension-free ray geometry -/

/-- Coordinates of a Euclidean vector are dominated by its norm. -/
lemma abs_coord_le_norm_N {n : ℕ} (x : EuclideanSpace ℝ (Fin n)) (i : Fin n) :
    |x i| ≤ ‖x‖ := by
  have h := PiLp.norm_apply_le x i
  simpa [Real.norm_eq_abs] using h

/-- Axial displacement is bounded by the probe-to-sample distance. -/
lemma abs_first_sub_le_norm_radialRayProbeN
    {n : ℕ} (hn : 0 < n) (r : ℝ) (y : EuclideanSpace ℝ (Fin n)) :
    |y ⟨0, hn⟩ - r| ≤ ‖radialRayProbeN n hn r - y‖ := by
  have h := abs_coord_le_norm_N (radialRayProbeN n hn r - y) ⟨0, hn⟩
  simpa [PiLp.sub_apply, radialRayProbeN_first, abs_sub_comm] using h

/-- The normalized axial displacement has absolute value at most one, with
the totalized division convention making the collision case harmless. -/
lemma abs_first_div_norm_radialRayProbeN_le_one
    {n : ℕ} (hn : 0 < n) (r : ℝ) (y : EuclideanSpace ℝ (Fin n)) :
    |(y ⟨0, hn⟩ - r) / ‖radialRayProbeN n hn r - y‖| ≤ 1 := by
  rcases eq_or_ne (‖radialRayProbeN n hn r - y‖) 0 with h0 | hne
  · rw [h0, div_zero, abs_zero]
    norm_num
  · have hpos : 0 < ‖radialRayProbeN n hn r - y‖ :=
      (norm_nonneg _).lt_of_ne' hne
    rw [abs_div, abs_of_pos hpos, div_le_one hpos]
    exact abs_first_sub_le_norm_radialRayProbeN hn r y

/-- The squared axial displacement divided by distance is bounded by the
distance itself, again including the totalized collision value. -/
lemma first_sq_div_norm_radialRayProbeN_le
    {n : ℕ} (hn : 0 < n) (r : ℝ) (y : EuclideanSpace ℝ (Fin n)) :
    (y ⟨0, hn⟩ - r) ^ 2 / ‖radialRayProbeN n hn r - y‖
      ≤ ‖radialRayProbeN n hn r - y‖ := by
  rcases eq_or_ne (‖radialRayProbeN n hn r - y‖) 0 with h0 | hne
  · rw [h0, div_zero]
  · have hpos : 0 < ‖radialRayProbeN n hn r - y‖ :=
      (norm_nonneg _).lt_of_ne' hne
    rw [div_le_iff₀ hpos]
    have habs := abs_first_sub_le_norm_radialRayProbeN hn r y
    nlinarith [mul_self_le_mul_self (abs_nonneg (y ⟨0, hn⟩ - r)) habs,
      sq_abs (y ⟨0, hn⟩ - r)]

/-- The moving ray probe has constant vector derivative `e₀`. -/
lemma hasDerivAt_radialRayProbeN
    {n : ℕ} (hn : 0 < n) (r : ℝ) :
    HasDerivAt (radialRayProbeN n hn)
      (EuclideanSpace.basisFun (Fin n) ℝ ⟨0, hn⟩) r := by
  change HasDerivAt
    (fun x : ℝ => x • EuclideanSpace.basisFun (Fin n) ℝ ⟨0, hn⟩)
    (EuclideanSpace.basisFun (Fin n) ℝ ⟨0, hn⟩) r
  simpa only [id_eq, one_smul] using
    (hasDerivAt_id r).smul_const
      (EuclideanSpace.basisFun (Fin n) ℝ ⟨0, hn⟩)

/-- The probe distance is differentiable away from collisions, with derivative
`(r-y₀)/‖r e₀-y‖`. -/
lemma hasDerivAt_norm_radialRayProbeN_sub
    {n : ℕ} (hn : 0 < n) {r : ℝ} {y : EuclideanSpace ℝ (Fin n)}
    (hne : ‖radialRayProbeN n hn r - y‖ ≠ 0) :
    HasDerivAt (fun x => ‖radialRayProbeN n hn x - y‖)
      ((r - y ⟨0, hn⟩) / ‖radialRayProbeN n hn r - y‖) r := by
  let e : EuclideanSpace ℝ (Fin n) :=
    EuclideanSpace.basisFun (Fin n) ℝ ⟨0, hn⟩
  have hline : HasDerivAt (fun x => radialRayProbeN n hn x - y) e r := by
    simpa [e] using (hasDerivAt_radialRayProbeN hn r).sub_const y
  have hsq := hline.norm_sq
  have hsqne : ‖radialRayProbeN n hn r - y‖ ^ 2 ≠ 0 := pow_ne_zero 2 hne
  have hsqrt := hsq.sqrt hsqne
  have hfun :
      (fun x => Real.sqrt (‖radialRayProbeN n hn x - y‖ ^ 2)) =
        fun x => ‖radialRayProbeN n hn x - y‖ := by
    funext x
    rw [Real.sqrt_sq (norm_nonneg _)]
  rw [hfun] at hsqrt
  have hinner :
      inner ℝ (radialRayProbeN n hn r - y) e = r - y ⟨0, hn⟩ := by
    simp [e, radialRayProbeN, PiLp.inner_apply]
  have hval :
      (2 * inner ℝ (radialRayProbeN n hn r - y) e) /
          (2 * Real.sqrt (‖radialRayProbeN n hn r - y‖ ^ 2)) =
        (r - y ⟨0, hn⟩) / ‖radialRayProbeN n hn r - y‖ := by
    rw [hinner, Real.sqrt_sq (norm_nonneg _)]
    exact mul_div_mul_left _ _ two_ne_zero
  rw [hval] at hsqrt
  exact hsqrt

/-! ## Collision-null geometry -/

/-- A singleton direction has zero normalized surface measure in dimensions
`n ≥ 3`.  This is proved directly from Mathlib's polar `toSphere` measure: the
corresponding radial segment lies in a strict one-dimensional affine
subspace.  It does not use the external zonal-coordinate theorem. -/
lemma uniformSphereMeasure_singleton
    {n : ℕ} (hn : 3 ≤ n)
    (ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1) :
    uniformSphereMeasure n {ω} = 0 := by
  let E := EuclideanSpace ℝ (Fin n)
  let L : Submodule ℝ E := Submodule.span ℝ {(ω : E)}
  have hωne : (ω : E) ≠ 0 := by
    intro hω
    have hnorm := norm_sphere_coe_eq_one ω
    rw [hω, norm_zero] at hnorm
    norm_num at hnorm
  have hLne : L ≠ ⊤ := by
    intro htop
    have hspan : Module.finrank ℝ L = 1 := by
      simpa [L] using finrank_span_singleton hωne
    rw [htop] at hspan
    have : n = 1 := by simpa [E] using hspan
    omega
  have hLAne : L.toAffineSubspace ≠ ⊤ := by
    intro htop
    apply hLne
    have hdir := congrArg AffineSubspace.direction htop
    simpa using hdir
  have hsub :
      Ioo (0 : ℝ) 1 • ((fun z : sphere (0 : E) 1 => (z : E)) '' {ω})
        ⊆ (L.toAffineSubspace : Set E) := by
    rintro z ⟨a, ha, b, ⟨ω', hω', rfl⟩, rfl⟩
    have hω'eq : ω' = ω := by simpa using hω'
    subst ω'
    change a • (ω : E) ∈ L
    exact L.smul_mem a (Submodule.subset_span (by simp))
  have hvol :
      (volume : Measure E)
        (Ioo (0 : ℝ) 1 • ((fun z : sphere (0 : E) 1 => (z : E)) '' {ω})) = 0 :=
    measure_mono_null hsub
      (Measure.addHaar_affineSubspace (volume : Measure E) L.toAffineSubspace hLAne)
  rw [uniformSphereMeasure, Measure.smul_apply, Measure.toSphere_apply'
    (volume : Measure E) (measurableSet_singleton ω), hvol]
  simp

/-- The positive first-axis direction as a point of the unit sphere. -/
noncomputable def radialAxisSphereN (n : ℕ) (hn : 0 < n) :
    sphere (0 : EuclideanSpace ℝ (Fin n)) 1 :=
  ⟨EuclideanSpace.basisFun (Fin n) ℝ ⟨0, hn⟩, by
    rw [Metric.mem_sphere, dist_zero_right]
    exact (EuclideanSpace.basisFun (Fin n) ℝ).norm_eq_one ⟨0, hn⟩⟩

@[simp] lemma coe_radialAxisSphereN (n : ℕ) (hn : 0 < n) :
    ((radialAxisSphereN n hn : sphere
      (0 : EuclideanSpace ℝ (Fin n)) 1) : EuclideanSpace ℝ (Fin n)) =
      EuclideanSpace.basisFun (Fin n) ℝ ⟨0, hn⟩ := rfl

/-- A nonnegative-profile radial mixture gives zero mass to the positive
first-coordinate axis.  Hence a moving positive ray probe collides with the
sample only on a null set. -/
lemma radialMixtureN_ae_probe_ne
    {n : ℕ} (hn : 3 ≤ n) (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (hsupp : ν (Iio 0) = 0) :
    ∀ᵐ y ∂(radialMixtureN n ν), ∀ x : ℝ, 0 < x →
      ‖radialRayProbeN n (by omega) x - y‖ ≠ 0 := by
  let σ := uniformSphereMeasure n
  letI : IsProbabilityMeasure σ := uniformSphereMeasure_isProbability hn
  let e₀ := radialAxisSphereN n (by omega)
  let B : Set (ℝ × sphere (0 : EuclideanSpace ℝ (Fin n)) 1) :=
    (Iio 0 ×ˢ univ) ∪ (univ ×ˢ {e₀})
  let A : Set (EuclideanSpace ℝ (Fin n)) :=
    (fun y => y - radialRayProbeN n (by omega) (y ⟨0, by omega⟩)) ⁻¹' {0} ∩
      (fun y => y ⟨0, by omega⟩) ⁻¹' Ioi 0
  have hAmeas : MeasurableSet A := by
    have hcoord : Continuous
        (fun y : EuclideanSpace ℝ (Fin n) => y ⟨0, by omega⟩) := by
      fun_prop
    have haxis : Continuous
        (fun y : EuclideanSpace ℝ (Fin n) =>
          y - radialRayProbeN n (by omega) (y ⟨0, by omega⟩)) := by
      unfold radialRayProbeN
      fun_prop
    exact (haxis.measurable (measurableSet_singleton 0)).inter
      (hcoord.measurable measurableSet_Ioi)
  have hBnull : (ν.prod σ) B = 0 := by
    apply measure_union_null
    · rw [Measure.prod_prod, hsupp, zero_mul]
    · rw [Measure.prod_prod, uniformSphereMeasure_singleton hn e₀, mul_zero]
  have hpre : radialScaleMap n ⁻¹' A ⊆ B := by
    rintro ⟨s, ω⟩ hz
    simp only [A, mem_inter_iff, mem_preimage, mem_singleton_iff, mem_Ioi,
      radialScaleMap] at hz
    obtain ⟨haxis, hxpos⟩ := hz
    have heq : s • (ω : EuclideanSpace ℝ (Fin n)) =
        radialRayProbeN n (by omega)
          ((s • (ω : EuclideanSpace ℝ (Fin n))) ⟨0, by omega⟩) :=
      sub_eq_zero.mp haxis
    rcases lt_or_ge s 0 with hsneg | hsnonneg
    · exact Or.inl ⟨hsneg, mem_univ _⟩
    · right
      refine ⟨mem_univ _, ?_⟩
      let x := (s • (ω : EuclideanSpace ℝ (Fin n))) ⟨0, by omega⟩
      have hxpos' : 0 < x := by simpa [x] using hxpos
      have hnormeq := congrArg norm heq
      have hprobeNorm : ‖radialRayProbeN n (by omega) x‖ = x := by
        rw [radialRayProbeN, norm_smul,
          (EuclideanSpace.basisFun (Fin n) ℝ).norm_eq_one]
        simp [Real.norm_eq_abs, abs_of_pos hxpos']
      have hsphereNorm : ‖(ω : EuclideanSpace ℝ (Fin n))‖ = 1 :=
        norm_sphere_coe_eq_one ω
      have hsx : s = x := by
        rw [norm_smul, hsphereNorm, mul_one, Real.norm_eq_abs,
          abs_of_nonneg hsnonneg, hprobeNorm] at hnormeq
        exact hnormeq
      apply Subtype.ext
      have hscaled :
          x • (ω : EuclideanSpace ℝ (Fin n)) =
            x • EuclideanSpace.basisFun (Fin n) ℝ ⟨0, by omega⟩ := by
        change s • (ω : EuclideanSpace ℝ (Fin n)) =
          radialRayProbeN n (by omega) x at heq
        rw [hsx] at heq
        simpa [radialRayProbeN] using heq
      have hinv := congrArg
        (fun v : EuclideanSpace ℝ (Fin n) => x⁻¹ • v) hscaled
      simpa [e₀, radialAxisSphereN, smul_smul, hxpos'.ne'] using hinv
  have hAnull : radialMixtureN n ν A = 0 := by
    rw [radialMixtureN, Measure.map_apply
      (continuous_radialScaleMap n).measurable hAmeas]
    exact measure_mono_null hpre hBnull
  rw [ae_iff]
  refine measure_mono_null ?_ hAnull
  intro y hy
  rcases not_forall.mp hy with ⟨x, hx⟩
  have hxpos : 0 < x := by
    by_contra hxnonpos
    exact hx fun h => (hxnonpos h).elim
  have hnorm0 : ‖radialRayProbeN n (by omega) x - y‖ = 0 := by
    by_contra hne
    exact hx fun _ => hne
  have heq : radialRayProbeN n (by omega) x = y := by
    rwa [norm_sub_eq_zero_iff] at hnorm0
  simp only [A, mem_inter_iff, mem_preimage, mem_singleton_iff, mem_Ioi]
  constructor
  · rw [← heq, radialRayProbeN_first]
    exact sub_self _
  · rw [← heq, radialRayProbeN_first]
    exact hxpos

/-! ## Pointwise kernel derivatives -/

/-- Pointwise derivative of the Laplace kernel along the first-coordinate
ray, away from a probe/sample collision. -/
lemma hasDerivAt_laplaceKernel_radialRayProbeN
    {n : ℕ} (hn : 0 < n) {τ r : ℝ} {y : EuclideanSpace ℝ (Fin n)}
    (hne : ‖radialRayProbeN n hn r - y‖ ≠ 0) :
    HasDerivAt (fun x => laplaceKernel τ (radialRayProbeN n hn x) y)
      ((1 / τ) * (laplaceKernel τ (radialRayProbeN n hn r) y *
        ((y ⟨0, hn⟩ - r) / ‖radialRayProbeN n hn r - y‖))) r := by
  have hd := hasDerivAt_norm_radialRayProbeN_sub hn hne
  have hexp := (hd.const_mul (-(1 / τ))).exp
  convert hexp using 1 <;> simp only [laplaceKernel]
  ring

/-- Pointwise derivative of the companion kernel along the ray. -/
lemma hasDerivAt_laplaceCompanionKernel_radialRayProbeN
    {n : ℕ} (hn : 0 < n) {τ r : ℝ} {y : EuclideanSpace ℝ (Fin n)}
    (hτ : 0 < τ) (hne : ‖radialRayProbeN n hn r - y‖ ≠ 0) :
    HasDerivAt (fun x => laplaceCompanionKernel τ (radialRayProbeN n hn x) y)
      ((1 / τ) * (laplaceKernel τ (radialRayProbeN n hn r) y *
        (y ⟨0, hn⟩ - r))) r := by
  have hd := hasDerivAt_norm_radialRayProbeN_sub hn hne
  have hk := hasDerivAt_laplaceKernel_radialRayProbeN hn (τ := τ) hne
  have hprod := ((hd.const_add τ).mul hk :
    HasDerivAt
      (fun x => (τ + ‖radialRayProbeN n hn x - y‖) *
        laplaceKernel τ (radialRayProbeN n hn x) y)
      ((r - y ⟨0, hn⟩) / ‖radialRayProbeN n hn r - y‖ *
          laplaceKernel τ (radialRayProbeN n hn r) y +
        (τ + ‖radialRayProbeN n hn r - y‖) *
          ((1 / τ) * (laplaceKernel τ (radialRayProbeN n hn r) y *
            ((y ⟨0, hn⟩ - r) / ‖radialRayProbeN n hn r - y‖)))) r)
  have hval :
      (r - y ⟨0, hn⟩) / ‖radialRayProbeN n hn r - y‖ *
          laplaceKernel τ (radialRayProbeN n hn r) y +
        (τ + ‖radialRayProbeN n hn r - y‖) *
          ((1 / τ) * (laplaceKernel τ (radialRayProbeN n hn r) y *
            ((y ⟨0, hn⟩ - r) / ‖radialRayProbeN n hn r - y‖))) =
      (1 / τ) * (laplaceKernel τ (radialRayProbeN n hn r) y *
        (y ⟨0, hn⟩ - r)) := by
    field_simp [hne, hτ.ne']
    ring
  rw [hval] at hprod
  simp only [laplaceCompanionKernel]
  exact hprod

/-- Pointwise derivative of the axial weighted-displacement integrand. -/
lemma hasDerivAt_laplaceKernel_mul_first_radialRayProbeN
    {n : ℕ} (hn : 0 < n) {τ r : ℝ} {y : EuclideanSpace ℝ (Fin n)}
    (hne : ‖radialRayProbeN n hn r - y‖ ≠ 0) :
    HasDerivAt
      (fun x => laplaceKernel τ (radialRayProbeN n hn x) y *
        (y ⟨0, hn⟩ - x))
      ((1 / τ) * (laplaceKernel τ (radialRayProbeN n hn r) y *
          ((y ⟨0, hn⟩ - r) ^ 2 / ‖radialRayProbeN n hn r - y‖)) -
        laplaceKernel τ (radialRayProbeN n hn r) y) r := by
  have hk := hasDerivAt_laplaceKernel_radialRayProbeN hn (τ := τ) hne
  have hlin : HasDerivAt (fun x : ℝ => y ⟨0, hn⟩ - x) (-1) r := by
    simpa using (hasDerivAt_id r).const_sub (y ⟨0, hn⟩)
  have hprod := (hk.mul hlin :
    HasDerivAt
      (fun x => laplaceKernel τ (radialRayProbeN n hn x) y *
        (y ⟨0, hn⟩ - x))
      ((1 / τ) * (laplaceKernel τ (radialRayProbeN n hn r) y *
          ((y ⟨0, hn⟩ - r) / ‖radialRayProbeN n hn r - y‖)) *
          (y ⟨0, hn⟩ - r) +
        laplaceKernel τ (radialRayProbeN n hn r) y * (-1)) r)
  have hval :
      (1 / τ) * (laplaceKernel τ (radialRayProbeN n hn r) y *
          ((y ⟨0, hn⟩ - r) / ‖radialRayProbeN n hn r - y‖)) *
          (y ⟨0, hn⟩ - r) +
        laplaceKernel τ (radialRayProbeN n hn r) y * (-1) =
      (1 / τ) * (laplaceKernel τ (radialRayProbeN n hn r) y *
          ((y ⟨0, hn⟩ - r) ^ 2 / ‖radialRayProbeN n hn r - y‖)) -
        laplaceKernel τ (radialRayProbeN n hn r) y := by
    ring
  rw [hval] at hprod
  exact hprod

/-! ## Integrated `Z` and `C` derivatives -/

/-- The `X₀/d` payload in the derivative of the radial normalizer. -/
noncomputable def radialRayZdN
    (n : ℕ) (hn : 0 < n) (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ y, laplaceKernel τ (radialRayProbeN n hn r) y *
      ((y ⟨0, hn⟩ - r) /
        ‖radialRayProbeN n hn r - y‖) ∂(radialMixtureN n ν)

/-- The `X₀²/d` payload in the derivative of the axial displacement. -/
noncomputable def radialRayQIntegralN
    (n : ℕ) (hn : 0 < n) (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ y, laplaceKernel τ (radialRayProbeN n hn r) y *
      ((y ⟨0, hn⟩ - r) ^ 2 /
        ‖radialRayProbeN n hn r - y‖) ∂(radialMixtureN n ν)

lemma laplaceKernel_radialRayProbeN_nonneg
    {n : ℕ} (hn : 0 < n) (τ r : ℝ) (y : EuclideanSpace ℝ (Fin n)) :
    0 ≤ laplaceKernel τ (radialRayProbeN n hn r) y :=
  (Real.exp_pos _).le

lemma laplaceKernel_radialRayProbeN_le_one
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (hτ : 0 < τ)
    (r : ℝ) (y : EuclideanSpace ℝ (Fin n)) :
    laplaceKernel τ (radialRayProbeN n hn r) y ≤ 1 := by
  have h0 : -(1 / τ) * ‖radialRayProbeN n hn r - y‖ ≤ 0 := by
    have hnn : 0 ≤ (1 / τ) * ‖radialRayProbeN n hn r - y‖ :=
      mul_nonneg (one_div_pos.mpr hτ).le (norm_nonneg _)
    linarith [neg_mul (1 / τ) (‖radialRayProbeN n hn r - y‖)]
  simp only [laplaceKernel]
  simpa using Real.exp_le_exp.mpr h0

lemma continuous_laplaceKernel_radialRayProbeN
    {n : ℕ} (hn : 0 < n) (τ r : ℝ) :
    Continuous (fun y : EuclideanSpace ℝ (Fin n) =>
      laplaceKernel τ (radialRayProbeN n hn r) y) := by
  simp only [laplaceKernel]
  exact Real.continuous_exp.comp
    (((continuous_const.sub continuous_id).norm).const_mul (-(1 / τ)))

lemma integrable_laplaceKernel_radialRayProbeN
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin n))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceKernel τ (radialRayProbeN n hn r) y) μ :=
  ⟨(continuous_laplaceKernel_radialRayProbeN hn τ r).aestronglyMeasurable,
    HasFiniteIntegral.of_bounded (C := 1)
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs,
          abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg hn τ r y)]
        exact laplaceKernel_radialRayProbeN_le_one hn τ hτ r y)⟩

/-- The general-dimensional radial normalizer is differentiable on the open
ray. -/
theorem hasDerivAt_radialRayZN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayZN n τ ν)
      ((1 / τ) * radialRayZdN n (by omega) τ ν r) r := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hfe : radialRayZN n τ ν =
      fun x => ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) x) y
        ∂(radialMixtureN n ν) := by
    funext x
    rw [radialRayZN_eq_kernelNormalizer hn hτ ν x]
    rfl
  have hmeasF' : AEStronglyMeasurable
      (fun y : EuclideanSpace ℝ (Fin n) =>
        (1 / τ) * (laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          ((y ⟨0, by omega⟩ - r) /
            ‖radialRayProbeN n (by omega) r - y‖))) (radialMixtureN n ν) := by
    have hcoord : Measurable
        (fun y : EuclideanSpace ℝ (Fin n) => y ⟨0, by omega⟩ - r) := by
      fun_prop
    have hnorm : Measurable
        (fun y : EuclideanSpace ℝ (Fin n) =>
          ‖radialRayProbeN n (by omega) r - y‖) := by
      fun_prop
    exact ((((continuous_laplaceKernel_radialRayProbeN (by omega) τ r).measurable).mul
      (hcoord.div hnorm)).const_mul _).aestronglyMeasurable
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := radialMixtureN n ν) (x₀ := r)
    (F := fun x (y : EuclideanSpace ℝ (Fin n)) =>
      laplaceKernel τ (radialRayProbeN n (by omega) x) y)
    (F' := fun x (y : EuclideanSpace ℝ (Fin n)) =>
      (1 / τ) * (laplaceKernel τ (radialRayProbeN n (by omega) x) y *
        ((y ⟨0, by omega⟩ - x) /
          ‖radialRayProbeN n (by omega) x - y‖)))
    (bound := fun _ => 1 / τ)
    (Ioi_mem_nhds hr)
    (Filter.Eventually.of_forall fun x =>
      (continuous_laplaceKernel_radialRayProbeN (by omega) τ x).aestronglyMeasurable)
    (integrable_laplaceKernel_radialRayProbeN (by omega) τ hτ _ r)
    hmeasF'
    (ae_of_all _ fun y => by
      intro x _
      rw [Real.norm_eq_abs, abs_mul, abs_mul,
        abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg (by omega) τ x y),
        abs_of_nonneg (by positivity : (0 : ℝ) ≤ 1 / τ)]
      calc
        1 / τ * (laplaceKernel τ (radialRayProbeN n (by omega) x) y *
            |(y ⟨0, by omega⟩ - x) /
              ‖radialRayProbeN n (by omega) x - y‖|)
          ≤ 1 / τ * (1 * 1) := by
            apply mul_le_mul_of_nonneg_left _ (by positivity)
            exact mul_le_mul
              (laplaceKernel_radialRayProbeN_le_one (by omega) τ hτ x y)
              (abs_first_div_norm_radialRayProbeN_le_one (by omega) x y)
              (abs_nonneg _) zero_le_one
        _ = 1 / τ := by ring)
    (integrable_const _)
    (by
      filter_upwards [radialMixtureN_ae_probe_ne hn ν hsupp] with y hy
      intro x hx
      exact hasDerivAt_laplaceKernel_radialRayProbeN (by omega) (hy x hx))
  rw [hfe]
  have hval :
      (∫ y, (1 / τ) *
        (laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          ((y ⟨0, by omega⟩ - r) /
            ‖radialRayProbeN n (by omega) r - y‖)) ∂(radialMixtureN n ν)) =
        (1 / τ) * radialRayZdN n (by omega) τ ν r := by
    rw [radialRayZdN, integral_const_mul]
  exact hval ▸ key.2

lemma laplaceCompanionKernel_radialRayProbeN_nonneg
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (hτ : 0 < τ)
    (r : ℝ) (y : EuclideanSpace ℝ (Fin n)) :
    0 ≤ laplaceCompanionKernel τ (radialRayProbeN n hn r) y := by
  simp only [laplaceCompanionKernel]
  exact mul_nonneg (add_nonneg hτ.le (norm_nonneg _))
    (laplaceKernel_radialRayProbeN_nonneg hn τ r y)

lemma laplaceCompanionKernel_radialRayProbeN_le
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (hτ : 0 < τ)
    (r : ℝ) (y : EuclideanSpace ℝ (Fin n)) :
    laplaceCompanionKernel τ (radialRayProbeN n hn r) y ≤ τ := by
  simp only [laplaceCompanionKernel, laplaceKernel]
  set d := ‖radialRayProbeN n hn r - y‖
  have h1 : τ + d ≤ τ * Real.exp (d / τ) := by
    have hle : d / τ + 1 ≤ Real.exp (d / τ) := Real.add_one_le_exp (d / τ)
    have hcancel : τ * (d / τ + 1) = τ + d := by
      field_simp [hτ.ne']
      ring
    nlinarith [mul_le_mul_of_nonneg_left hle hτ.le, hcancel]
  have hcancel2 : Real.exp (d / τ) * Real.exp (-(1 / τ) * d) = 1 := by
    rw [← Real.exp_add, show d / τ + -(1 / τ) * d = 0 by ring, Real.exp_zero]
  have h3 := mul_le_mul_of_nonneg_right h1 (Real.exp_pos (-(1 / τ) * d)).le
  rw [mul_assoc, hcancel2, mul_one] at h3
  exact h3

lemma continuous_laplaceCompanionKernel_radialRayProbeN
    {n : ℕ} (hn : 0 < n) (τ r : ℝ) :
    Continuous (fun y : EuclideanSpace ℝ (Fin n) =>
      laplaceCompanionKernel τ (radialRayProbeN n hn r) y) := by
  simp only [laplaceCompanionKernel]
  exact (continuous_const.add ((continuous_const.sub continuous_id).norm)).mul
    (continuous_laplaceKernel_radialRayProbeN hn τ r)

lemma integrable_laplaceCompanionKernel_radialRayProbeN
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin n))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable
      (fun y => laplaceCompanionKernel τ (radialRayProbeN n hn r) y) μ :=
  ⟨(continuous_laplaceCompanionKernel_radialRayProbeN hn τ r).aestronglyMeasurable,
    HasFiniteIntegral.of_bounded (C := τ)
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs,
          abs_of_nonneg (laplaceCompanionKernel_radialRayProbeN_nonneg
            hn τ hτ r y)]
        exact laplaceCompanionKernel_radialRayProbeN_le hn τ hτ r y)⟩

/-- The companion radial profile satisfies `C' = D/τ` on the open ray. -/
theorem hasDerivAt_radialRayCN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayCN n τ ν) ((1 / τ) * radialRayDN n τ ν r) r := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hfe : radialRayCN n τ ν =
      fun x => ∫ y, laplaceCompanionKernel τ (radialRayProbeN n (by omega) x) y
        ∂(radialMixtureN n ν) := by
    funext x
    rw [radialRayCN_eq_companionNormalizer hn τ ν x
      (integrable_laplaceCompanionKernel_radialRayProbeN
        (by omega) τ hτ _ x)]
    rfl
  have hdisp (x : ℝ) : Integrable
      (fun y => (laplaceWeightedDisplacement τ
        (radialRayProbeN n (by omega) x) y) ⟨0, by omega⟩)
      (radialMixtureN n ν) := by
    have hc : Continuous (fun y : EuclideanSpace ℝ (Fin n) =>
        laplaceKernel τ (radialRayProbeN n (by omega) x) y *
          (y ⟨0, by omega⟩ - x)) := by
      exact (continuous_laplaceKernel_radialRayProbeN (by omega) τ x).mul (by
        fun_prop)
    refine ⟨?_, HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
      (ae_of_all _ fun y => ?_)⟩
    · simpa only [laplaceWeightedDisplacement, PiLp.smul_apply, PiLp.sub_apply,
        radialRayProbeN_first, smul_eq_mul] using hc.aestronglyMeasurable
    · rw [Real.norm_eq_abs]
      simp only [laplaceWeightedDisplacement, PiLp.smul_apply, PiLp.sub_apply,
        radialRayProbeN_first, smul_eq_mul]
      rw [abs_mul,
        abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg (by omega) τ x y)]
      calc
        laplaceKernel τ (radialRayProbeN n (by omega) x) y *
            |y ⟨0, by omega⟩ - x|
          ≤ laplaceKernel τ (radialRayProbeN n (by omega) x) y *
              ‖radialRayProbeN n (by omega) x - y‖ :=
            mul_le_mul_of_nonneg_left
              (abs_first_sub_le_norm_radialRayProbeN (by omega) x y)
              (laplaceKernel_radialRayProbeN_nonneg (by omega) τ x y)
        _ = ‖radialRayProbeN n (by omega) x - y‖ *
              Real.exp (-‖radialRayProbeN n (by omega) x - y‖ / τ) := by
            simp only [laplaceKernel]
            rw [show -(1 / τ) * ‖radialRayProbeN n (by omega) x - y‖ =
              -‖radialRayProbeN n (by omega) x - y‖ / τ by ring]
            ring
        _ ≤ τ * Real.exp (-1) :=
          mul_exp_neg_div_le hτ (norm_nonneg _)
  have hD (x : ℝ) :
      radialRayDN n τ ν x =
        ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) x) y *
          (y ⟨0, by omega⟩ - x) ∂(radialMixtureN n ν) := by
    rw [radialRayDN_eq_displacementCoord hn τ ν x (hdisp x)]
    apply integral_congr_ae
    filter_upwards [] with y
    simp only [laplaceWeightedDisplacement, PiLp.smul_apply, PiLp.sub_apply,
      radialRayProbeN_first, smul_eq_mul]
  have hmeasF' : AEStronglyMeasurable
      (fun y : EuclideanSpace ℝ (Fin n) =>
        (1 / τ) * (laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          (y ⟨0, by omega⟩ - r))) (radialMixtureN n ν) := by
    have hc : Continuous (fun y : EuclideanSpace ℝ (Fin n) =>
        (1 / τ) * (laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          (y ⟨0, by omega⟩ - r))) := by
      exact continuous_const.mul
        ((continuous_laplaceKernel_radialRayProbeN (by omega) τ r).mul (by
          fun_prop))
    exact hc.aestronglyMeasurable
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := radialMixtureN n ν) (x₀ := r)
    (F := fun x (y : EuclideanSpace ℝ (Fin n)) =>
      laplaceCompanionKernel τ (radialRayProbeN n (by omega) x) y)
    (F' := fun x (y : EuclideanSpace ℝ (Fin n)) =>
      (1 / τ) * (laplaceKernel τ (radialRayProbeN n (by omega) x) y *
        (y ⟨0, by omega⟩ - x)))
    (bound := fun _ => Real.exp (-1))
    (Ioi_mem_nhds hr)
    (Filter.Eventually.of_forall fun x =>
      (continuous_laplaceCompanionKernel_radialRayProbeN
        (by omega) τ x).aestronglyMeasurable)
    (integrable_laplaceCompanionKernel_radialRayProbeN (by omega) τ hτ _ r)
    hmeasF'
    (ae_of_all _ fun y => by
      intro x _
      rw [Real.norm_eq_abs, abs_mul,
        abs_of_nonneg (by positivity : (0 : ℝ) ≤ 1 / τ)]
      have hpoint :
          |laplaceKernel τ (radialRayProbeN n (by omega) x) y *
            (y ⟨0, by omega⟩ - x)| ≤ τ * Real.exp (-1) := by
        rw [abs_mul,
          abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg (by omega) τ x y)]
        calc
          laplaceKernel τ (radialRayProbeN n (by omega) x) y *
              |y ⟨0, by omega⟩ - x|
            ≤ laplaceKernel τ (radialRayProbeN n (by omega) x) y *
                ‖radialRayProbeN n (by omega) x - y‖ :=
              mul_le_mul_of_nonneg_left
                (abs_first_sub_le_norm_radialRayProbeN (by omega) x y)
                (laplaceKernel_radialRayProbeN_nonneg (by omega) τ x y)
          _ = ‖radialRayProbeN n (by omega) x - y‖ *
                Real.exp (-‖radialRayProbeN n (by omega) x - y‖ / τ) := by
              simp only [laplaceKernel]
              rw [show -(1 / τ) * ‖radialRayProbeN n (by omega) x - y‖ =
                -‖radialRayProbeN n (by omega) x - y‖ / τ by ring]
              ring
          _ ≤ τ * Real.exp (-1) :=
            mul_exp_neg_div_le hτ (norm_nonneg _)
      calc
        1 / τ * |laplaceKernel τ (radialRayProbeN n (by omega) x) y *
            (y ⟨0, by omega⟩ - x)|
          ≤ 1 / τ * (τ * Real.exp (-1)) :=
            mul_le_mul_of_nonneg_left hpoint (by positivity)
        _ = Real.exp (-1) := by field_simp)
    (integrable_const _)
    (by
      filter_upwards [radialMixtureN_ae_probe_ne hn ν hsupp] with y hy
      intro x hx
      exact hasDerivAt_laplaceCompanionKernel_radialRayProbeN
        (by omega) hτ (hy x hx))
  rw [hfe]
  have hval :
      (∫ y, (1 / τ) *
        (laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          (y ⟨0, by omega⟩ - r)) ∂(radialMixtureN n ν)) =
        (1 / τ) * radialRayDN n τ ν r := by
    rw [integral_const_mul, hD]
  exact hval ▸ key.2

/-! ## Integrated axial-displacement derivative -/

lemma abs_laplaceKernel_mul_first_radialRayProbeN_le
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin n)) :
    |laplaceKernel τ (radialRayProbeN n hn r) y * (y ⟨0, hn⟩ - r)|
      ≤ τ * Real.exp (-1) := by
  rw [abs_mul,
    abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg hn τ r y)]
  calc
    laplaceKernel τ (radialRayProbeN n hn r) y * |y ⟨0, hn⟩ - r|
      ≤ laplaceKernel τ (radialRayProbeN n hn r) y *
          ‖radialRayProbeN n hn r - y‖ :=
        mul_le_mul_of_nonneg_left
          (abs_first_sub_le_norm_radialRayProbeN hn r y)
          (laplaceKernel_radialRayProbeN_nonneg hn τ r y)
    _ = ‖radialRayProbeN n hn r - y‖ *
          Real.exp (-‖radialRayProbeN n hn r - y‖ / τ) := by
        simp only [laplaceKernel]
        rw [show -(1 / τ) * ‖radialRayProbeN n hn r - y‖ =
          -‖radialRayProbeN n hn r - y‖ / τ by ring]
        ring
    _ ≤ τ * Real.exp (-1) := mul_exp_neg_div_le hτ (norm_nonneg _)

lemma integrable_laplaceKernel_mul_first_radialRayProbeN
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin n))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceKernel τ (radialRayProbeN n hn r) y *
      (y ⟨0, hn⟩ - r)) μ := by
  have hc : Continuous (fun y : EuclideanSpace ℝ (Fin n) =>
      laplaceKernel τ (radialRayProbeN n hn r) y * (y ⟨0, hn⟩ - r)) :=
    (continuous_laplaceKernel_radialRayProbeN hn τ r).mul (by fun_prop)
  exact ⟨hc.aestronglyMeasurable,
    HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs]
        exact abs_laplaceKernel_mul_first_radialRayProbeN_le hn τ hτ r y)⟩

/-- Raw scalar-integral form of the zonal axial displacement. -/
lemma radialRayDN_eq_integral_first
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayDN n τ ν r =
      ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y *
        (y ⟨0, by omega⟩ - r) ∂(radialMixtureN n ν) := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hdisp : Integrable
      (fun y => (laplaceWeightedDisplacement τ
        (radialRayProbeN n (by omega) r) y) ⟨0, by omega⟩)
      (radialMixtureN n ν) := by
    have hraw := integrable_laplaceKernel_mul_first_radialRayProbeN
      (n := n) (by omega) τ hτ (radialMixtureN n ν) r
    convert hraw using 1
    funext y
    simp only [laplaceWeightedDisplacement, PiLp.smul_apply, PiLp.sub_apply,
      radialRayProbeN_first, smul_eq_mul]
  rw [radialRayDN_eq_displacementCoord hn τ ν r hdisp]
  apply integral_congr_ae
  filter_upwards [] with y
  simp only [laplaceWeightedDisplacement, PiLp.smul_apply, PiLp.sub_apply,
    radialRayProbeN_first, smul_eq_mul]

lemma abs_laplaceKernel_mul_first_sq_div_le
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin n)) :
    |laplaceKernel τ (radialRayProbeN n hn r) y *
      ((y ⟨0, hn⟩ - r) ^ 2 / ‖radialRayProbeN n hn r - y‖)|
      ≤ τ * Real.exp (-1) := by
  have hK := laplaceKernel_radialRayProbeN_nonneg hn τ r y
  have hdiv : 0 ≤ (y ⟨0, hn⟩ - r) ^ 2 /
      ‖radialRayProbeN n hn r - y‖ :=
    div_nonneg (sq_nonneg _) (norm_nonneg _)
  rw [abs_mul, abs_of_nonneg hK, abs_of_nonneg hdiv]
  calc
    laplaceKernel τ (radialRayProbeN n hn r) y *
        ((y ⟨0, hn⟩ - r) ^ 2 / ‖radialRayProbeN n hn r - y‖)
      ≤ laplaceKernel τ (radialRayProbeN n hn r) y *
          ‖radialRayProbeN n hn r - y‖ :=
        mul_le_mul_of_nonneg_left
          (first_sq_div_norm_radialRayProbeN_le hn r y) hK
    _ = ‖radialRayProbeN n hn r - y‖ *
          Real.exp (-‖radialRayProbeN n hn r - y‖ / τ) := by
        simp only [laplaceKernel]
        rw [show -(1 / τ) * ‖radialRayProbeN n hn r - y‖ =
          -‖radialRayProbeN n hn r - y‖ / τ by ring]
        ring
    _ ≤ τ * Real.exp (-1) := mul_exp_neg_div_le hτ (norm_nonneg _)

lemma measurable_laplaceKernel_mul_first_sq_div
    {n : ℕ} (hn : 0 < n) (τ r : ℝ) :
    Measurable (fun y : EuclideanSpace ℝ (Fin n) =>
      laplaceKernel τ (radialRayProbeN n hn r) y *
        ((y ⟨0, hn⟩ - r) ^ 2 / ‖radialRayProbeN n hn r - y‖)) := by
  have hcoord : Measurable (fun y : EuclideanSpace ℝ (Fin n) =>
      (y ⟨0, hn⟩ - r) ^ 2) := by fun_prop
  have hnorm : Measurable (fun y : EuclideanSpace ℝ (Fin n) =>
      ‖radialRayProbeN n hn r - y‖) := by fun_prop
  exact (continuous_laplaceKernel_radialRayProbeN hn τ r).measurable.mul
    (hcoord.div hnorm)

lemma integrable_laplaceKernel_mul_first_sq_div
    {n : ℕ} (hn : 0 < n) (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin n))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y : EuclideanSpace ℝ (Fin n) =>
      laplaceKernel τ (radialRayProbeN n hn r) y *
        ((y ⟨0, hn⟩ - r) ^ 2 / ‖radialRayProbeN n hn r - y‖)) μ := by
  exact ⟨(measurable_laplaceKernel_mul_first_sq_div hn τ r).aestronglyMeasurable,
    HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs]
        exact abs_laplaceKernel_mul_first_sq_div_le hn τ hτ r y)⟩

/-- The physical `X₀²/d` derivative payload agrees with the zonal shell
payload used by the general-dimensional ray system. -/
theorem radialRayQIntegralN_eq_radialRayQN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayQIntegralN n (by omega) τ ν r = radialRayQN n τ ν r := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  rw [radialRayQIntegralN,
    integral_radialMixtureN hn ν
      (integrable_laplaceKernel_mul_first_sq_div
        (n := n) (by omega) τ hτ (radialMixtureN n ν) r),
    radialRayQN]
  exact integral_congr_ae (Filter.Eventually.of_forall fun s =>
    integral_uniformSphere_laplaceAxialSqDiv_eq_shellQN hn τ r s)

/-- The axial displacement profile satisfies
`D' = Q_integral/τ - Z` on the open ray. -/
theorem hasDerivAt_radialRayDN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayDN n τ ν)
      ((1 / τ) * radialRayQIntegralN n (by omega) τ ν r -
        radialRayZN n τ ν r) r := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hfe : radialRayDN n τ ν = fun x =>
      ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) x) y *
        (y ⟨0, by omega⟩ - x) ∂(radialMixtureN n ν) := by
    funext x
    exact radialRayDN_eq_integral_first hn τ hτ ν x
  have hmeasF' : AEStronglyMeasurable
      (fun y : EuclideanSpace ℝ (Fin n) =>
        (1 / τ) * (laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          ((y ⟨0, by omega⟩ - r) ^ 2 /
            ‖radialRayProbeN n (by omega) r - y‖)) -
        laplaceKernel τ (radialRayProbeN n (by omega) r) y)
      (radialMixtureN n ν) :=
    ((((measurable_laplaceKernel_mul_first_sq_div
      (n := n) (by omega) τ r).const_mul _).sub
        (continuous_laplaceKernel_radialRayProbeN
          (by omega) τ r).measurable).aestronglyMeasurable)
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := radialMixtureN n ν) (x₀ := r)
    (F := fun x (y : EuclideanSpace ℝ (Fin n)) =>
      laplaceKernel τ (radialRayProbeN n (by omega) x) y *
        (y ⟨0, by omega⟩ - x))
    (F' := fun x (y : EuclideanSpace ℝ (Fin n)) =>
      (1 / τ) * (laplaceKernel τ (radialRayProbeN n (by omega) x) y *
        ((y ⟨0, by omega⟩ - x) ^ 2 /
          ‖radialRayProbeN n (by omega) x - y‖)) -
      laplaceKernel τ (radialRayProbeN n (by omega) x) y)
    (bound := fun _ => Real.exp (-1) + 1)
    (Ioi_mem_nhds hr)
    (Filter.Eventually.of_forall fun x =>
      ((continuous_laplaceKernel_radialRayProbeN (by omega) τ x).mul
        (by fun_prop)).aestronglyMeasurable)
    (integrable_laplaceKernel_mul_first_radialRayProbeN
      (by omega) τ hτ _ r)
    hmeasF'
    (ae_of_all _ fun y => by
      intro x _
      have hA :
          |(1 / τ) * (laplaceKernel τ (radialRayProbeN n (by omega) x) y *
            ((y ⟨0, by omega⟩ - x) ^ 2 /
              ‖radialRayProbeN n (by omega) x - y‖))| ≤ Real.exp (-1) := by
        rw [abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ 1 / τ)]
        have hb := abs_laplaceKernel_mul_first_sq_div_le
          (n := n) (by omega) τ hτ x y
        calc
          1 / τ * |laplaceKernel τ (radialRayProbeN n (by omega) x) y *
              ((y ⟨0, by omega⟩ - x) ^ 2 /
                ‖radialRayProbeN n (by omega) x - y‖)|
            ≤ 1 / τ * (τ * Real.exp (-1)) :=
              mul_le_mul_of_nonneg_left hb (by positivity)
          _ = Real.exp (-1) := by field_simp
      have hK : |laplaceKernel τ (radialRayProbeN n (by omega) x) y| ≤ 1 := by
        rw [abs_of_nonneg
          (laplaceKernel_radialRayProbeN_nonneg (by omega) τ x y)]
        exact laplaceKernel_radialRayProbeN_le_one (by omega) τ hτ x y
      rw [Real.norm_eq_abs]
      exact (abs_sub _ _).trans (add_le_add hA hK))
    (integrable_const _)
    (by
      filter_upwards [radialMixtureN_ae_probe_ne hn ν hsupp] with y hy
      intro x hx
      exact hasDerivAt_laplaceKernel_mul_first_radialRayProbeN
        (by omega) (hy x hx))
  rw [hfe]
  have hint1 : Integrable
      (fun y : EuclideanSpace ℝ (Fin n) =>
        (1 / τ) * (laplaceKernel τ (radialRayProbeN n (by omega) r) y *
          ((y ⟨0, by omega⟩ - r) ^ 2 /
            ‖radialRayProbeN n (by omega) r - y‖)) ) (radialMixtureN n ν) :=
    ⟨((measurable_laplaceKernel_mul_first_sq_div
        (n := n) (by omega) τ r).const_mul _).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := Real.exp (-1))
        (ae_of_all _ fun y => by
          rw [Real.norm_eq_abs, abs_mul,
            abs_of_nonneg (by positivity : (0 : ℝ) ≤ 1 / τ)]
          have hb := abs_laplaceKernel_mul_first_sq_div_le
            (n := n) (by omega) τ hτ r y
          calc
            1 / τ * |laplaceKernel τ (radialRayProbeN n (by omega) r) y *
                ((y ⟨0, by omega⟩ - r) ^ 2 /
                  ‖radialRayProbeN n (by omega) r - y‖)|
              ≤ 1 / τ * (τ * Real.exp (-1)) :=
                mul_le_mul_of_nonneg_left hb (by positivity)
            _ = Real.exp (-1) := by field_simp)⟩
  have hval :
      (∫ y, ((1 / τ) *
          (laplaceKernel τ (radialRayProbeN n (by omega) r) y *
            ((y ⟨0, by omega⟩ - r) ^ 2 /
              ‖radialRayProbeN n (by omega) r - y‖)) -
        laplaceKernel τ (radialRayProbeN n (by omega) r) y)
        ∂(radialMixtureN n ν)) =
      (1 / τ) * radialRayQIntegralN n (by omega) τ ν r -
        radialRayZN n τ ν r := by
    rw [integral_sub hint1
      (integrable_laplaceKernel_radialRayProbeN (by omega) τ hτ _ r),
      integral_const_mul, radialRayQIntegralN,
      radialRayZN_eq_kernelNormalizer hn hτ ν r]
    rfl
  exact hval ▸ key.2

/-- Shell form of the axial-displacement derivative:
`D' = Q/τ - Z` on the open ray. -/
theorem hasDerivAt_radialRayDN_shell
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayDN n τ ν)
      ((1 / τ) * radialRayQN n τ ν r - radialRayZN n τ ν r) r := by
  simpa [radialRayQIntegralN_eq_radialRayQN hn τ hτ ν r] using
    hasDerivAt_radialRayDN hn τ hτ ν hsupp hr

end DriftingIdentifiability
