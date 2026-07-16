import DriftingIdentifiability.LaplaceRadialShell3

/-!
# Radial Laplace converse, milestone L5: ray layer (`n = 3`)

This file starts the R-layer promised in `LaplaceHigherDim.md §4.10` and
`LaplaceL5_HANDOFF.md`.  The shell layer has reduced ray evaluations of the
Laplace normalizer, companion normalizer, and drift numerator to per-shell
zonal averages.  Here we package those shell mixtures as honest ray-profile
functions of the probe radius `r`.

The heavy R-layer still to come is the differentiability/system layer
(`Z̃'`, `C̃'`, common tilted displacement, closure identity, and the RSI sign
split).  This file deliberately contains only definitions and shell-bridge
facts, so it stays axiom-free and does not smuggle in any endgame theorem.
-/

open MeasureTheory Filter Topology Set
open scoped RealInnerProductSpace

namespace DriftingIdentifiability
open Paper

/-- Ray Laplace normalizer profile for a radial mixture. -/
noncomputable def radialRayZ₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellZ τ r s ∂ν

/-- Ray companion-normalizer profile for a radial mixture. -/
noncomputable def radialRayC₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellC τ r s ∂ν

/-- Ray first-coordinate drift-numerator profile for a radial mixture. -/
noncomputable def radialRayD₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellD τ r s ∂ν

/-- Ray `T` profile, the shell average of the axial coordinate `t=s·u`. -/
noncomputable def radialRayT₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellT τ r s ∂ν

/-- Ray tangential `ρ²/d` profile. -/
noncomputable def radialRayRhoSqOverDist₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellRhoSqOverDist τ r s ∂ν

lemma radialRayZ₃_eq_kernelNormalizer (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayZ₃ τ ν r =
      kernelNormalizer (laplaceKernel τ) (radialMixture₃ ν) (rayProbe r) := by
  rw [radialRayZ₃, kernelNormalizer_radialMixture₃ τ hτ ν r]

lemma radialRayC₃_eq_companionNormalizer (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayC₃ τ ν r =
      kernelNormalizer (laplaceCompanionKernel τ) (radialMixture₃ ν) (rayProbe r) := by
  rw [radialRayC₃, kernelNormalizer_companion_radialMixture₃ τ hτ ν r]

lemma radialRayD₃_eq_weightedDisplacementCoord (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayD₃ τ ν r =
      (∫ y, laplaceWeightedDisplacement τ (rayProbe r) y ∂(radialMixture₃ ν)) 0 := by
  rw [radialRayD₃, laplaceWeightedDisplacement_coord_radialMixture₃ τ hτ ν r]

/-! ## Pointwise probe-derivative layer

Everything below differentiates in the probe radius `x ↦ r•e₁` at a fixed
sample `y`, valid whenever the probe does not collide with the sample
(`‖rayProbe x − y‖ ≠ 0`).  The collision set is `radialMixture₃`-null
(`radialMixture₃_ae_probe_ne`), which is exactly what the dominated
differentiation of the ray objects needs. -/

/-- The probe distance as an explicit square root of a quadratic in the probe
radius. -/
lemma norm_rayProbe_sub_eq_sqrt (r : ℝ) (y : EuclideanSpace ℝ (Fin 3)) :
    ‖rayProbe r - y‖ = Real.sqrt ((r - y 0) ^ 2 + ((y 1) ^ 2 + (y 2) ^ 2)) := by
  have hsq : ‖rayProbe r - y‖ ^ 2 = (r - y 0) ^ 2 + ((y 1) ^ 2 + (y 2) ^ 2) := by
    rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
    simp only [PiLp.sub_apply, rayProbe_apply_zero, rayProbe_apply_one, rayProbe_apply_two]
    ring
  rw [← hsq, Real.sqrt_sq (norm_nonneg _)]

/-- Axial displacement is dominated by the probe distance: `|y₀ − r| ≤ d`. -/
lemma abs_coord_sub_le_norm_rayProbe_sub (r : ℝ) (y : EuclideanSpace ℝ (Fin 3)) :
    |y 0 - r| ≤ ‖rayProbe r - y‖ := by
  rw [norm_rayProbe_sub_eq_sqrt, abs_sub_comm]
  calc |r - y 0| = Real.sqrt ((r - y 0) ^ 2) := (Real.sqrt_sq_eq_abs _).symm
    _ ≤ Real.sqrt ((r - y 0) ^ 2 + ((y 1) ^ 2 + (y 2) ^ 2)) :=
        Real.sqrt_le_sqrt (le_add_of_nonneg_right (by positivity))

/-- `|(y₀−r)/d| ≤ 1`, junk-safe at collisions (`x/0 = 0`). -/
lemma abs_coord_div_norm_rayProbe_le_one (r : ℝ) (y : EuclideanSpace ℝ (Fin 3)) :
    |(y 0 - r) / ‖rayProbe r - y‖| ≤ 1 := by
  rcases eq_or_ne (‖rayProbe r - y‖) 0 with h0 | hne
  · rw [h0, div_zero, abs_zero]; norm_num
  · have hpos : 0 < ‖rayProbe r - y‖ := (norm_nonneg _).lt_of_ne' hne
    rw [abs_div, abs_of_pos hpos, div_le_one hpos]
    exact abs_coord_sub_le_norm_rayProbe_sub r y

/-- `X²/d ≤ d`, junk-safe at collisions. -/
lemma sq_coord_div_norm_rayProbe_le (r : ℝ) (y : EuclideanSpace ℝ (Fin 3)) :
    (y 0 - r) ^ 2 / ‖rayProbe r - y‖ ≤ ‖rayProbe r - y‖ := by
  rcases eq_or_ne (‖rayProbe r - y‖) 0 with h0 | hne
  · rw [h0, div_zero]
  · have hpos : 0 < ‖rayProbe r - y‖ := (norm_nonneg _).lt_of_ne' hne
    rw [div_le_iff₀ hpos]
    have habs := abs_coord_sub_le_norm_rayProbe_sub r y
    nlinarith [mul_self_le_mul_self (abs_nonneg (y 0 - r)) habs, sq_abs (y 0 - r)]

/-- The probe distance is differentiable in the probe radius away from
collisions, with the expected radial derivative `(r − y₀)/d`. -/
lemma hasDerivAt_norm_rayProbe_sub {r : ℝ} {y : EuclideanSpace ℝ (Fin 3)}
    (hne : ‖rayProbe r - y‖ ≠ 0) :
    HasDerivAt (fun x => ‖rayProbe x - y‖)
      ((r - y 0) / ‖rayProbe r - y‖) r := by
  set c : ℝ := (y 1) ^ 2 + (y 2) ^ 2 with hc
  have hq : HasDerivAt (fun x : ℝ => (x - y 0) ^ 2 + c) (2 * (r - y 0)) r := by
    have h1 : HasDerivAt (fun x : ℝ => x - y 0) 1 r := (hasDerivAt_id r).sub_const _
    simpa using (h1.pow 2).add_const c
  have hqne : (r - y 0) ^ 2 + c ≠ 0 := by
    intro h0
    apply hne
    rw [norm_rayProbe_sub_eq_sqrt, ← hc, h0, Real.sqrt_zero]
  have hsq := hq.sqrt hqne
  have hfun : (fun x : ℝ => Real.sqrt ((x - y 0) ^ 2 + c))
      = fun x => ‖rayProbe x - y‖ := by
    funext x
    rw [norm_rayProbe_sub_eq_sqrt]
  rw [hfun] at hsq
  have hval : 2 * (r - y 0) / (2 * Real.sqrt ((r - y 0) ^ 2 + c))
      = (r - y 0) / ‖rayProbe r - y‖ := by
    rw [norm_rayProbe_sub_eq_sqrt, ← hc]
    exact mul_div_mul_left _ _ two_ne_zero
  rw [hval] at hsq
  exact hsq

/-- The Laplace kernel from the moving probe is differentiable away from
collisions: `∂_r e^{−d/τ} = (1/τ)·e^{−d/τ}·(y₀−r)/d`. -/
lemma hasDerivAt_laplaceKernel_rayProbe' {τ r : ℝ} {y : EuclideanSpace ℝ (Fin 3)}
    (hne : ‖rayProbe r - y‖ ≠ 0) :
    HasDerivAt (fun x => laplaceKernel τ (rayProbe x) y)
      ((1 / τ) * (laplaceKernel τ (rayProbe r) y
        * ((y 0 - r) / ‖rayProbe r - y‖))) r := by
  have hd := hasDerivAt_norm_rayProbe_sub hne
  have hexp := ((hd.const_mul (-(1 / τ))).exp :
    HasDerivAt (fun x => Real.exp (-(1 / τ) * ‖rayProbe x - y‖))
      (Real.exp (-(1 / τ) * ‖rayProbe r - y‖)
        * (-(1 / τ) * ((r - y 0) / ‖rayProbe r - y‖))) r)
  have hval : Real.exp (-(1 / τ) * ‖rayProbe r - y‖)
        * (-(1 / τ) * ((r - y 0) / ‖rayProbe r - y‖))
      = (1 / τ) * (Real.exp (-(1 / τ) * ‖rayProbe r - y‖)
        * ((y 0 - r) / ‖rayProbe r - y‖)) := by
    ring
  rw [hval] at hexp
  simp only [laplaceKernel]
  exact hexp

/-- The companion kernel from the moving probe is differentiable away from
collisions, with the **collision-free** derivative `(1/τ)·K·(y₀−r)` — this is
the pointwise form of `C̃' = D̃/τ`. -/
lemma hasDerivAt_laplaceCompanionKernel_rayProbe' {τ r : ℝ}
    {y : EuclideanSpace ℝ (Fin 3)} (hτ : 0 < τ) (hne : ‖rayProbe r - y‖ ≠ 0) :
    HasDerivAt (fun x => laplaceCompanionKernel τ (rayProbe x) y)
      ((1 / τ) * (laplaceKernel τ (rayProbe r) y * (y 0 - r))) r := by
  have hd := hasDerivAt_norm_rayProbe_sub hne
  have hk := hasDerivAt_laplaceKernel_rayProbe' (τ := τ) hne
  have hprod := ((hd.const_add τ).mul hk :
    HasDerivAt (fun x => (τ + ‖rayProbe x - y‖) * laplaceKernel τ (rayProbe x) y)
      ((r - y 0) / ‖rayProbe r - y‖ * laplaceKernel τ (rayProbe r) y
        + (τ + ‖rayProbe r - y‖)
          * ((1 / τ) * (laplaceKernel τ (rayProbe r) y
            * ((y 0 - r) / ‖rayProbe r - y‖)))) r)
  have hval : (r - y 0) / ‖rayProbe r - y‖ * laplaceKernel τ (rayProbe r) y
        + (τ + ‖rayProbe r - y‖)
          * ((1 / τ) * (laplaceKernel τ (rayProbe r) y
            * ((y 0 - r) / ‖rayProbe r - y‖)))
      = (1 / τ) * (laplaceKernel τ (rayProbe r) y * (y 0 - r)) := by
    field_simp [hne, hτ.ne']
    ring
  rw [hval] at hprod
  simp only [laplaceCompanionKernel]
  exact hprod

/-- The axial weighted displacement from the moving probe is differentiable away
from collisions: `∂_r [K·(y₀−r)] = (1/τ)·K·X²/d − K`. -/
lemma hasDerivAt_laplaceKernel_mul_coord' {τ r : ℝ}
    {y : EuclideanSpace ℝ (Fin 3)} (hne : ‖rayProbe r - y‖ ≠ 0) :
    HasDerivAt (fun x => laplaceKernel τ (rayProbe x) y * (y 0 - x))
      ((1 / τ) * (laplaceKernel τ (rayProbe r) y
          * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖))
        - laplaceKernel τ (rayProbe r) y) r := by
  have hk := hasDerivAt_laplaceKernel_rayProbe' (τ := τ) hne
  have hlin : HasDerivAt (fun x : ℝ => y 0 - x) (-1) r := by
    simpa using ((hasDerivAt_id r).const_sub (y 0))
  have hprod := (hk.mul hlin :
    HasDerivAt (fun x => laplaceKernel τ (rayProbe x) y * (y 0 - x))
      ((1 / τ) * (laplaceKernel τ (rayProbe r) y
          * ((y 0 - r) / ‖rayProbe r - y‖)) * (y 0 - r)
        + laplaceKernel τ (rayProbe r) y * (-1)) r)
  have hval : (1 / τ) * (laplaceKernel τ (rayProbe r) y
          * ((y 0 - r) / ‖rayProbe r - y‖)) * (y 0 - r)
        + laplaceKernel τ (rayProbe r) y * (-1)
      = (1 / τ) * (laplaceKernel τ (rayProbe r) y
          * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖))
        - laplaceKernel τ (rayProbe r) y := by
    ring
  rw [hval] at hprod
  exact hprod

/-- **The collision set is null**: for `radialMixture₃`-a.e. sample `y`, the
probe never collides with `y` along the open ray.  (The only candidates are
axis points `(t,0,0)` with `t > 0`, whose chart preimage lives in the
`chartBase`-null slice `u = 1`.) -/
lemma radialMixture₃_ae_probe_ne (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    ∀ᵐ y ∂(radialMixture₃ ν), ∀ x : ℝ, 0 < x → ‖rayProbe x - y‖ ≠ 0 := by
  set A : Set (EuclideanSpace ℝ (Fin 3)) :=
    ((fun y : EuclideanSpace ℝ (Fin 3) => y 1) ⁻¹' {0})
      ∩ (((fun y : EuclideanSpace ℝ (Fin 3) => y 2) ⁻¹' {0})
        ∩ ((fun y : EuclideanSpace ℝ (Fin 3) => y 0) ⁻¹' Ioi 0)) with hA
  have hAmeas : MeasurableSet A := by
    have h1 : Continuous fun y : EuclideanSpace ℝ (Fin 3) => y 1 := by fun_prop
    have h2 : Continuous fun y : EuclideanSpace ℝ (Fin 3) => y 2 := by fun_prop
    have h0 : Continuous fun y : EuclideanSpace ℝ (Fin 3) => y 0 := by fun_prop
    exact (h1.measurable (measurableSet_singleton 0)).inter
      ((h2.measurable (measurableSet_singleton 0)).inter
        (h0.measurable measurableSet_Ioi))
  have hnull : radialMixture₃ ν A = 0 := by
    rw [radialMixture₃, Measure.map_apply continuous_chartMap.measurable hAmeas]
    have hsub : chartMap ⁻¹' A ⊆
        (univ : Set ℝ) ×ˢ (({u : ℝ | 1 - u ^ 2 ≤ 0} : Set ℝ) ×ˢ (univ : Set ℝ)) := by
      rintro ⟨s, u, φ⟩ hz
      simp only [hA, mem_preimage, mem_inter_iff, mem_singleton_iff, mem_Ioi,
        chartMap_mk, PiLp.smul_apply, sphereChart_apply_zero, sphereChart_apply_one,
        sphereChart_apply_two, smul_eq_mul] at hz
      obtain ⟨h1, h2, h0⟩ := hz
      refine ⟨mem_univ _, ?_, mem_univ _⟩
      by_contra hu
      have hupos : 0 < 1 - u ^ 2 := not_le.mp hu
      have hsqrt_pos : 0 < Real.sqrt (1 - u ^ 2) := Real.sqrt_pos.mpr hupos
      have hs_ne : s ≠ 0 := by
        intro h0s
        rw [h0s, zero_mul] at h0
        exact lt_irrefl 0 h0
      have hcos : Real.cos φ = 0 := by
        have := mul_eq_zero.mp h1
        rcases this with h | h
        · exact absurd h hs_ne
        · rcases mul_eq_zero.mp h with h' | h'
          · exact absurd h' hsqrt_pos.ne'
          · exact h'
      have hsin : Real.sin φ = 0 := by
        have := mul_eq_zero.mp h2
        rcases this with h | h
        · exact absurd h hs_ne
        · rcases mul_eq_zero.mp h with h' | h'
          · exact absurd h' hsqrt_pos.ne'
          · exact h'
      have := Real.sin_sq_add_cos_sq φ
      rw [hsin, hcos] at this
      norm_num at this
    refine measure_mono_null hsub ?_
    rw [Measure.prod_prod]
    have hBu : MeasurableSet ({u : ℝ | 1 - u ^ 2 ≤ 0} : Set ℝ) := by
      have : Continuous fun u : ℝ => 1 - u ^ 2 := by fun_prop
      exact this.measurable measurableSet_Iic
    have hcb : chartBase (({u : ℝ | 1 - u ^ 2 ≤ 0} : Set ℝ) ×ˢ (univ : Set ℝ)) = 0 := by
      rw [chartBase, Measure.smul_apply, Measure.prod_prod,
        Measure.restrict_apply hBu]
      have hset : {u : ℝ | 1 - u ^ 2 ≤ 0} ∩ Ioc (-1 : ℝ) 1 = {1} := by
        ext u
        constructor
        · rintro ⟨hu, hlo, hhi⟩
          have hsq : 1 ≤ u ^ 2 := by simpa using hu
          have : u ≤ -1 ∨ 1 ≤ u := by
            rcases le_or_gt u 0 with h | h
            · left; nlinarith
            · right; nlinarith
          rcases this with h | h
          · exact absurd h (not_le.mpr hlo)
          · exact le_antisymm hhi h
        · rintro rfl
          exact ⟨by norm_num, by norm_num, le_refl 1⟩
      rw [hset]
      simp
    rw [hcb, mul_zero]
  rw [ae_iff]
  refine measure_mono_null (fun y hy => ?_) hnull
  simp only [mem_setOf_eq] at hy
  rcases not_forall.mp hy with ⟨x, hx⟩
  have hxpos : 0 < x := by
    by_contra hxneg
    exact hx fun h => absurd h hxneg
  have hnorm0 : ‖rayProbe x - y‖ = 0 := by
    by_contra hne
    exact hx fun _ => hne
  have hyx : rayProbe x = y := by
    rwa [norm_sub_eq_zero_iff] at hnorm0
  simp only [hA, mem_inter_iff, mem_preimage, mem_singleton_iff, mem_Ioi]
  refine ⟨?_, ?_, ?_⟩ <;> rw [← hyx]
  · exact rayProbe_apply_one x
  · exact rayProbe_apply_two x
  · rw [rayProbe_apply_zero]; exact hxpos

/-! ## The C¹ layer: dominated differentiation of the ray objects -/

/-- The `X/d`-weighted ray integral — the derivative payload of `Z̃`. -/
noncomputable def radialRayZd₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ y, laplaceKernel τ (rayProbe r) y * ((y 0 - r) / ‖rayProbe r - y‖)
    ∂(radialMixture₃ ν)

/-- The `X²/d`-weighted ray integral — the second-moment payload of `D̃'`. -/
noncomputable def radialRayQ₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ y, laplaceKernel τ (rayProbe r) y * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖)
    ∂(radialMixture₃ ν)

lemma integrable_laplaceKernel_rayProbe (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin 3))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceKernel τ (rayProbe r) y) μ :=
  ⟨(continuous_laplaceKernel_rayProbe τ r).aestronglyMeasurable,
    HasFiniteIntegral.of_bounded (C := 1)
      (ae_of_all _ (fun y => by
        rw [Real.norm_eq_abs, abs_of_nonneg (laplaceKernel_rayProbe_nonneg τ r y)]
        exact laplaceKernel_rayProbe_le_one τ hτ r y))⟩

/-- **`Z̃` is differentiable on the open ray** with derivative
`Z̃'(r) = (1/τ)·∫ (X/d)·e^{−d/τ}`. -/
theorem hasDerivAt_radialRayZ₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayZ₃ τ ν) ((1 / τ) * radialRayZd₃ τ ν r) r := by
  have hfe : radialRayZ₃ τ ν
      = fun x => ∫ y, laplaceKernel τ (rayProbe x) y ∂(radialMixture₃ ν) := by
    funext x
    rw [radialRayZ₃_eq_kernelNormalizer τ hτ ν x]
    rfl
  have hmeasF' : AEStronglyMeasurable
      (fun y : EuclideanSpace ℝ (Fin 3) =>
        (1 / τ) * (laplaceKernel τ (rayProbe r) y
          * ((y 0 - r) / ‖rayProbe r - y‖))) (radialMixture₃ ν) := by
    have hcoord : Measurable fun y : EuclideanSpace ℝ (Fin 3) => y 0 - r := by
      have : Continuous fun y : EuclideanSpace ℝ (Fin 3) => y 0 - r := by fun_prop
      exact this.measurable
    have hnorm : Measurable fun y : EuclideanSpace ℝ (Fin 3) => ‖rayProbe r - y‖ := by
      have : Continuous fun y : EuclideanSpace ℝ (Fin 3) => ‖rayProbe r - y‖ := by
        exact (continuous_const.sub continuous_id).norm
      exact this.measurable
    exact ((((continuous_laplaceKernel_rayProbe τ r).measurable).mul
      (hcoord.div hnorm)).const_mul _).aestronglyMeasurable
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := radialMixture₃ ν) (x₀ := r)
    (F := fun x (y : EuclideanSpace ℝ (Fin 3)) => laplaceKernel τ (rayProbe x) y)
    (F' := fun x (y : EuclideanSpace ℝ (Fin 3)) =>
      (1 / τ) * (laplaceKernel τ (rayProbe x) y * ((y 0 - x) / ‖rayProbe x - y‖)))
    (bound := fun _ => 1 / τ)
    (Ioi_mem_nhds hr)
    (Filter.Eventually.of_forall fun x =>
      (continuous_laplaceKernel_rayProbe τ x).aestronglyMeasurable)
    (integrable_laplaceKernel_rayProbe τ hτ _ r)
    hmeasF'
    (ae_of_all _ fun y => by
      intro x _
      rw [Real.norm_eq_abs, abs_mul, abs_mul,
        abs_of_nonneg (laplaceKernel_rayProbe_nonneg τ x y),
        abs_of_nonneg (by positivity : (0:ℝ) ≤ 1 / τ)]
      calc 1 / τ * (laplaceKernel τ (rayProbe x) y * |(y 0 - x) / ‖rayProbe x - y‖|)
          ≤ 1 / τ * (1 * 1) := by
            apply mul_le_mul_of_nonneg_left _ (by positivity)
            exact mul_le_mul (laplaceKernel_rayProbe_le_one τ hτ x y)
              (abs_coord_div_norm_rayProbe_le_one x y) (abs_nonneg _) zero_le_one
        _ = 1 / τ := by ring)
    (integrable_const _)
    (by
      filter_upwards [radialMixture₃_ae_probe_ne ν] with y hy
      intro x hx
      exact hasDerivAt_laplaceKernel_rayProbe' (hy x hx))
  rw [hfe]
  have hval : (∫ y, (1 / τ) * (laplaceKernel τ (rayProbe r) y
      * ((y 0 - r) / ‖rayProbe r - y‖)) ∂(radialMixture₃ ν))
      = (1 / τ) * radialRayZd₃ τ ν r := by
    rw [radialRayZd₃, integral_const_mul]
  exact hval ▸ key.2

end DriftingIdentifiability
