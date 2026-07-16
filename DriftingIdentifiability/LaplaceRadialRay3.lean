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

/-- The axial weighted-displacement integrand is uniformly bounded by `τ·e⁻¹`. -/
lemma abs_laplaceKernel_mul_coord_le (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin 3)) :
    |laplaceKernel τ (rayProbe r) y * (y 0 - r)| ≤ τ * Real.exp (-1) := by
  rw [abs_mul, abs_of_nonneg (laplaceKernel_rayProbe_nonneg τ r y)]
  calc laplaceKernel τ (rayProbe r) y * |y 0 - r|
      ≤ laplaceKernel τ (rayProbe r) y * ‖rayProbe r - y‖ :=
        mul_le_mul_of_nonneg_left (abs_coord_sub_le_norm_rayProbe_sub r y)
          (laplaceKernel_rayProbe_nonneg τ r y)
    _ = ‖rayProbe r - y‖ * Real.exp (-‖rayProbe r - y‖ / τ) := by
        simp only [laplaceKernel]
        rw [show -(1 / τ) * ‖rayProbe r - y‖ = -‖rayProbe r - y‖ / τ by ring]
        ring
    _ ≤ τ * Real.exp (-1) := mul_exp_neg_div_le hτ (norm_nonneg _)

/-- The `X²/d`-weighted kernel integrand is uniformly bounded by `τ·e⁻¹`. -/
lemma abs_laplaceKernel_mul_sq_div_le (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin 3)) :
    |laplaceKernel τ (rayProbe r) y * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖)|
      ≤ τ * Real.exp (-1) := by
  have hK := laplaceKernel_rayProbe_nonneg τ r y
  have hdiv : (0 : ℝ) ≤ (y 0 - r) ^ 2 / ‖rayProbe r - y‖ :=
    div_nonneg (by positivity) (norm_nonneg _)
  rw [abs_mul, abs_of_nonneg hK, abs_of_nonneg hdiv]
  calc laplaceKernel τ (rayProbe r) y * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖)
      ≤ laplaceKernel τ (rayProbe r) y * ‖rayProbe r - y‖ :=
        mul_le_mul_of_nonneg_left (sq_coord_div_norm_rayProbe_le r y) hK
    _ = ‖rayProbe r - y‖ * Real.exp (-‖rayProbe r - y‖ / τ) := by
        simp only [laplaceKernel]
        rw [show -(1 / τ) * ‖rayProbe r - y‖ = -‖rayProbe r - y‖ / τ by ring]
        ring
    _ ≤ τ * Real.exp (-1) := mul_exp_neg_div_le hτ (norm_nonneg _)

lemma measurable_laplaceKernel_mul_sq_div (τ r : ℝ) :
    Measurable (fun y : EuclideanSpace ℝ (Fin 3) =>
      laplaceKernel τ (rayProbe r) y * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖)) := by
  have hcoord : Measurable fun y : EuclideanSpace ℝ (Fin 3) => (y 0 - r) ^ 2 := by
    have : Continuous fun y : EuclideanSpace ℝ (Fin 3) => (y 0 - r) ^ 2 := by fun_prop
    exact this.measurable
  have hnorm : Measurable fun y : EuclideanSpace ℝ (Fin 3) => ‖rayProbe r - y‖ :=
    ((continuous_const.sub continuous_id).norm).measurable
  exact ((continuous_laplaceKernel_rayProbe τ r).measurable).mul (hcoord.div hnorm)

/-- `D̃` in its raw axial-integral form (component-0 of the drift numerator,
commuted through the Bochner integral). -/
lemma radialRayD₃_eq_integral_coord (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayD₃ τ ν r
      = ∫ y, laplaceKernel τ (rayProbe r) y * (y 0 - r) ∂(radialMixture₃ ν) := by
  rw [radialRayD₃_eq_weightedDisplacementCoord τ hτ ν r]
  have hFint : Integrable (fun y => laplaceWeightedDisplacement τ (rayProbe r) y)
      (radialMixture₃ ν) :=
    laplaceWeightedDisplacement_integrable τ hτ (radialMixture₃ ν) (rayProbe r)
  have hproj :=
    (EuclideanSpace.proj (0 : Fin 3)).integral_comp_comm (μ := radialMixture₃ ν) hFint
  have hcoord : (∫ y, laplaceWeightedDisplacement τ (rayProbe r) y
        ∂(radialMixture₃ ν)) 0
      = ∫ y, (laplaceWeightedDisplacement τ (rayProbe r) y) 0
        ∂(radialMixture₃ ν) := by
    simpa [EuclideanSpace.coe_proj] using hproj.symm
  rw [hcoord]
  refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
  simp only [laplaceWeightedDisplacement, PiLp.smul_apply, PiLp.sub_apply,
    rayProbe_apply_zero, smul_eq_mul]

lemma integrable_laplaceKernel_mul_coord (τ : ℝ) (hτ : 0 < τ)
    (μ : Measure (EuclideanSpace ℝ (Fin 3))) [IsFiniteMeasure μ] (r : ℝ) :
    Integrable (fun y => laplaceKernel τ (rayProbe r) y * (y 0 - r)) μ := by
  have hc : Continuous (fun y : EuclideanSpace ℝ (Fin 3) =>
      laplaceKernel τ (rayProbe r) y * (y 0 - r)) := by
    have h0 : Continuous fun y : EuclideanSpace ℝ (Fin 3) => y 0 - r := by fun_prop
    exact (continuous_laplaceKernel_rayProbe τ r).mul h0
  exact ⟨hc.aestronglyMeasurable,
    HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
      (ae_of_all _ fun y => by
        rw [Real.norm_eq_abs]
        exact abs_laplaceKernel_mul_coord_le τ hτ r y)⟩

/-- **`C̃` is differentiable on the open ray with `C̃' = (1/τ)·D̃`** — the
integrated companion-derivative identity, and the first leg of the closure. -/
theorem hasDerivAt_radialRayC₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayC₃ τ ν) ((1 / τ) * radialRayD₃ τ ν r) r := by
  have hfe : radialRayC₃ τ ν
      = fun x => ∫ y, laplaceCompanionKernel τ (rayProbe x) y ∂(radialMixture₃ ν) := by
    funext x
    rw [radialRayC₃_eq_companionNormalizer τ hτ ν x]
    rfl
  have hmeasF' : AEStronglyMeasurable
      (fun y : EuclideanSpace ℝ (Fin 3) =>
        (1 / τ) * (laplaceKernel τ (rayProbe r) y * (y 0 - r))) (radialMixture₃ ν) := by
    have h0 : Continuous fun y : EuclideanSpace ℝ (Fin 3) =>
        (1 / τ) * (laplaceKernel τ (rayProbe r) y * (y 0 - r)) := by
      have := continuous_laplaceKernel_rayProbe τ r
      fun_prop
    exact h0.aestronglyMeasurable
  have hCint : Integrable (fun y => laplaceCompanionKernel τ (rayProbe r) y)
      (radialMixture₃ ν) :=
    ⟨(continuous_laplaceCompanionKernel_rayProbe τ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := τ)
        (ae_of_all _ fun y => by
          rw [Real.norm_eq_abs,
            abs_of_nonneg (laplaceCompanionKernel_rayProbe_nonneg τ hτ r y)]
          exact laplaceCompanionKernel_rayProbe_le τ hτ r y)⟩
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := radialMixture₃ ν) (x₀ := r)
    (F := fun x (y : EuclideanSpace ℝ (Fin 3)) => laplaceCompanionKernel τ (rayProbe x) y)
    (F' := fun x (y : EuclideanSpace ℝ (Fin 3)) =>
      (1 / τ) * (laplaceKernel τ (rayProbe x) y * (y 0 - x)))
    (bound := fun _ => Real.exp (-1))
    (Ioi_mem_nhds hr)
    (Filter.Eventually.of_forall fun x =>
      (continuous_laplaceCompanionKernel_rayProbe τ x).aestronglyMeasurable)
    hCint
    hmeasF'
    (ae_of_all _ fun y => by
      intro x _
      rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ 1 / τ)]
      have hb := abs_laplaceKernel_mul_coord_le τ hτ x y
      calc 1 / τ * |laplaceKernel τ (rayProbe x) y * (y 0 - x)|
          ≤ 1 / τ * (τ * Real.exp (-1)) :=
            mul_le_mul_of_nonneg_left hb (by positivity)
        _ = Real.exp (-1) := by field_simp)
    (integrable_const _)
    (by
      filter_upwards [radialMixture₃_ae_probe_ne ν] with y hy
      intro x hx
      exact hasDerivAt_laplaceCompanionKernel_rayProbe' hτ (hy x hx))
  rw [hfe]
  have hval : (∫ y, (1 / τ) * (laplaceKernel τ (rayProbe r) y * (y 0 - r))
      ∂(radialMixture₃ ν)) = (1 / τ) * radialRayD₃ τ ν r := by
    rw [integral_const_mul, radialRayD₃_eq_integral_coord τ hτ ν r]
  exact hval ▸ key.2

/-- **`D̃` is differentiable on the open ray with
`D̃' = (1/τ)·Q̃ − Z̃`** — the second-moment derivative formula feeding both the
closure identity and the sign layer. -/
theorem hasDerivAt_radialRayD₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    HasDerivAt (radialRayD₃ τ ν)
      ((1 / τ) * radialRayQ₃ τ ν r - radialRayZ₃ τ ν r) r := by
  have hfe : radialRayD₃ τ ν
      = fun x => ∫ y, laplaceKernel τ (rayProbe x) y * (y 0 - x)
          ∂(radialMixture₃ ν) := by
    funext x
    exact radialRayD₃_eq_integral_coord τ hτ ν x
  have hmeasF' : AEStronglyMeasurable
      (fun y : EuclideanSpace ℝ (Fin 3) =>
        (1 / τ) * (laplaceKernel τ (rayProbe r) y
          * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖))
          - laplaceKernel τ (rayProbe r) y) (radialMixture₃ ν) := by
    exact (((measurable_laplaceKernel_mul_sq_div τ r).const_mul _).sub
      (continuous_laplaceKernel_rayProbe τ r).measurable).aestronglyMeasurable
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := radialMixture₃ ν) (x₀ := r)
    (F := fun x (y : EuclideanSpace ℝ (Fin 3)) =>
      laplaceKernel τ (rayProbe x) y * (y 0 - x))
    (F' := fun x (y : EuclideanSpace ℝ (Fin 3)) =>
      (1 / τ) * (laplaceKernel τ (rayProbe x) y
        * ((y 0 - x) ^ 2 / ‖rayProbe x - y‖))
        - laplaceKernel τ (rayProbe x) y)
    (bound := fun _ => Real.exp (-1) + 1)
    (Ioi_mem_nhds hr)
    (Filter.Eventually.of_forall fun x => by
      have h0 : Continuous (fun y : EuclideanSpace ℝ (Fin 3) =>
          laplaceKernel τ (rayProbe x) y * (y 0 - x)) := by
        have := continuous_laplaceKernel_rayProbe τ x
        fun_prop
      exact h0.aestronglyMeasurable)
    (integrable_laplaceKernel_mul_coord τ hτ _ r)
    hmeasF'
    (ae_of_all _ fun y => by
      intro x _
      have hA : |(1 / τ) * (laplaceKernel τ (rayProbe x) y
          * ((y 0 - x) ^ 2 / ‖rayProbe x - y‖))| ≤ Real.exp (-1) := by
        rw [abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ 1 / τ)]
        have hb := abs_laplaceKernel_mul_sq_div_le τ hτ x y
        calc 1 / τ * |laplaceKernel τ (rayProbe x) y
            * ((y 0 - x) ^ 2 / ‖rayProbe x - y‖)|
            ≤ 1 / τ * (τ * Real.exp (-1)) :=
              mul_le_mul_of_nonneg_left hb (by positivity)
          _ = Real.exp (-1) := by field_simp
      have hK : |laplaceKernel τ (rayProbe x) y| ≤ 1 := by
        rw [abs_of_nonneg (laplaceKernel_rayProbe_nonneg τ x y)]
        exact laplaceKernel_rayProbe_le_one τ hτ x y
      rw [Real.norm_eq_abs]
      calc |(1 / τ) * (laplaceKernel τ (rayProbe x) y
            * ((y 0 - x) ^ 2 / ‖rayProbe x - y‖))
            - laplaceKernel τ (rayProbe x) y|
          ≤ |(1 / τ) * (laplaceKernel τ (rayProbe x) y
            * ((y 0 - x) ^ 2 / ‖rayProbe x - y‖))|
            + |laplaceKernel τ (rayProbe x) y| := abs_sub _ _
        _ ≤ Real.exp (-1) + 1 := add_le_add hA hK)
    (integrable_const _)
    (by
      filter_upwards [radialMixture₃_ae_probe_ne ν] with y hy
      intro x hx
      exact hasDerivAt_laplaceKernel_mul_coord' (hy x hx))
  rw [hfe]
  have hint1 : Integrable (fun y : EuclideanSpace ℝ (Fin 3) =>
      (1 / τ) * (laplaceKernel τ (rayProbe r) y
        * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖))) (radialMixture₃ ν) :=
    ⟨((measurable_laplaceKernel_mul_sq_div τ r).const_mul _).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := Real.exp (-1))
        (ae_of_all _ fun y => by
          rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ 1 / τ)]
          have hb := abs_laplaceKernel_mul_sq_div_le τ hτ r y
          calc 1 / τ * |laplaceKernel τ (rayProbe r) y
              * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖)|
              ≤ 1 / τ * (τ * Real.exp (-1)) :=
                mul_le_mul_of_nonneg_left hb (by positivity)
            _ = Real.exp (-1) := by field_simp)⟩
  have hval : (∫ y, ((1 / τ) * (laplaceKernel τ (rayProbe r) y
      * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖))
      - laplaceKernel τ (rayProbe r) y) ∂(radialMixture₃ ν))
      = (1 / τ) * radialRayQ₃ τ ν r - radialRayZ₃ τ ν r := by
    rw [integral_sub hint1 (integrable_laplaceKernel_rayProbe τ hτ _ r),
      integral_const_mul, radialRayZ₃_eq_kernelNormalizer τ hτ ν r]
    rfl
  exact hval ▸ key.2

/-! ## The closure identity `C̃ = Q̃ + 3τ·Z̃ + (2τ/r)·D̃`

This is the `n = 3` closure (F4 of `LaplaceHigherDim.md §4.10`) in its algebraic
form; combined with `hasDerivAt_radialRayD₃` it yields
`C̃ = τ·D̃' + 4τ·Z̃ + (2τ/r)·D̃`.  The proof chains: the companion split
`C̃ = τZ̃ + ∫d·w̄`, the pointwise cylindrical split `d = X²/d + ρ²/d`
(junk-safe), the T₃ mixture `P̃ = (2τ/r)·T̃`, and the axial split
`T̃ = D̃ + r·Z̃`. -/

/-- The cylindrical norm split `d² = X² + ρ²` in coordinates. -/
lemma norm_rayProbe_sub_sq_eq (r : ℝ) (y : EuclideanSpace ℝ (Fin 3)) :
    ‖rayProbe r - y‖ ^ 2 = (y 0 - r) ^ 2 + ((y 1) ^ 2 + (y 2) ^ 2) := by
  rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
  simp only [PiLp.sub_apply, rayProbe_apply_zero, rayProbe_apply_one, rayProbe_apply_two]
  ring

/-- `ρ²/d ≤ d`, junk-safe at collisions. -/
lemma rhoSq_div_norm_rayProbe_le (r : ℝ) (y : EuclideanSpace ℝ (Fin 3)) :
    ((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖ ≤ ‖rayProbe r - y‖ := by
  rcases eq_or_ne (‖rayProbe r - y‖) 0 with h0 | hne
  · rw [h0, div_zero]
  · have hpos : 0 < ‖rayProbe r - y‖ := (norm_nonneg _).lt_of_ne' hne
    rw [div_le_iff₀ hpos]
    nlinarith [norm_rayProbe_sub_sq_eq r y, sq_nonneg (y 0 - r)]

/-- The `ρ²/d`-weighted kernel integrand is uniformly bounded by `τ·e⁻¹`. -/
lemma abs_laplaceKernel_mul_rhoSq_div_le (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin 3)) :
    |laplaceKernel τ (rayProbe r) y * (((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖)|
      ≤ τ * Real.exp (-1) := by
  have hK := laplaceKernel_rayProbe_nonneg τ r y
  have hdiv : (0 : ℝ) ≤ ((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖ :=
    div_nonneg (by positivity) (norm_nonneg _)
  rw [abs_mul, abs_of_nonneg hK, abs_of_nonneg hdiv]
  calc laplaceKernel τ (rayProbe r) y * (((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖)
      ≤ laplaceKernel τ (rayProbe r) y * ‖rayProbe r - y‖ :=
        mul_le_mul_of_nonneg_left (rhoSq_div_norm_rayProbe_le r y) hK
    _ = ‖rayProbe r - y‖ * Real.exp (-‖rayProbe r - y‖ / τ) := by
        simp only [laplaceKernel]
        rw [show -(1 / τ) * ‖rayProbe r - y‖ = -‖rayProbe r - y‖ / τ by ring]
        ring
    _ ≤ τ * Real.exp (-1) := mul_exp_neg_div_le hτ (norm_nonneg _)

lemma measurable_laplaceKernel_mul_rhoSq_div (τ r : ℝ) :
    Measurable (fun y : EuclideanSpace ℝ (Fin 3) =>
      laplaceKernel τ (rayProbe r) y
        * (((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖)) := by
  have hrho : Measurable fun y : EuclideanSpace ℝ (Fin 3) => (y 1) ^ 2 + (y 2) ^ 2 := by
    have : Continuous fun y : EuclideanSpace ℝ (Fin 3) => (y 1) ^ 2 + (y 2) ^ 2 := by
      fun_prop
    exact this.measurable
  have hnorm : Measurable fun y : EuclideanSpace ℝ (Fin 3) => ‖rayProbe r - y‖ :=
    ((continuous_const.sub continuous_id).norm).measurable
  exact ((continuous_laplaceKernel_rayProbe τ r).measurable).mul (hrho.div hnorm)

/-- Measurable-integrand generalisation of `integral_chartBase_zonal` (needed for
the `ρ²/d` integrand, whose chart section is discontinuous at the collision
`s = r, u = 1`). -/
lemma integral_chartBase_zonal_measurable {G : ℝ × ℝ → ℝ} {g : ℝ → ℝ} {C : ℝ}
    (hG : Measurable G) (hC : ∀ w, |G w| ≤ C)
    (hEq : ∀ u φ, u ∈ Ioc (-1 : ℝ) 1 → G (u, φ) = g u) :
    ∫ w : ℝ × ℝ, G w ∂chartBase = (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1, g u := by
  have hBvol : volume (Ioc (-Real.pi) Real.pi) = ENNReal.ofReal (2 * Real.pi) := by
    rw [Real.volume_Ioc]; congr 1; ring
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-1 : ℝ) 1)) :=
    ⟨by rw [Measure.restrict_apply_univ, Real.volume_Ioc]; exact ENNReal.ofReal_lt_top⟩
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-Real.pi) Real.pi)) :=
    ⟨by rw [Measure.restrict_apply_univ, hBvol]; exact ENNReal.ofReal_lt_top⟩
  have hint : Integrable G
      ((volume.restrict (Ioc (-1 : ℝ) 1)).prod (volume.restrict (Ioc (-Real.pi) Real.pi))) :=
    ⟨hG.aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := C)
        (ae_of_all _ (fun w => by rw [Real.norm_eq_abs]; exact hC w))⟩
  rw [chartBase, integral_smul_measure, integral_prod _ hint]
  have hcong : ∀ u ∈ Ioc (-1 : ℝ) 1,
      ∫ φ in Ioc (-Real.pi) Real.pi, G (u, φ) ∂volume = 2 * Real.pi * g u := by
    intro u hu
    rw [setIntegral_congr_fun measurableSet_Ioc (fun φ _ => hEq u φ hu), setIntegral_const,
      Real.volume_real_Ioc_of_le (by linarith [Real.pi_pos]), smul_eq_mul]
    ring
  rw [setIntegral_congr_fun measurableSet_Ioc hcong, integral_const_mul]
  rw [ENNReal.toReal_inv, ENNReal.toReal_ofReal (by positivity), smul_eq_mul]
  have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp
  ring

/-- **`T̃` bridge**: the axial ray profile is the y-level integral `∫ K·y₀`. -/
lemma radialRayT₃_eq_integral (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayT₃ τ ν r
      = ∫ y, laplaceKernel τ (rayProbe r) y * y 0 ∂(radialMixture₃ ν) := by
  have hcont : Continuous (fun y : EuclideanSpace ℝ (Fin 3) =>
      laplaceKernel τ (rayProbe r) y * y 0) := by
    have := continuous_laplaceKernel_rayProbe τ r
    fun_prop
  have habs : ∀ y : EuclideanSpace ℝ (Fin 3),
      |laplaceKernel τ (rayProbe r) y * y 0| ≤ τ * Real.exp (-1) + |r| := by
    intro y
    have h1 : |y 0| ≤ |y 0 - r| + |r| := by
      calc |y 0| = |(y 0 - r) + r| := by ring_nf
        _ ≤ |y 0 - r| + |r| := abs_add_le _ _
    have hK := laplaceKernel_rayProbe_nonneg τ r y
    have hK1 := laplaceKernel_rayProbe_le_one τ hτ r y
    rw [abs_mul, abs_of_nonneg hK]
    calc laplaceKernel τ (rayProbe r) y * |y 0|
        ≤ laplaceKernel τ (rayProbe r) y * (|y 0 - r| + |r|) :=
          mul_le_mul_of_nonneg_left h1 hK
      _ = laplaceKernel τ (rayProbe r) y * |y 0 - r|
          + laplaceKernel τ (rayProbe r) y * |r| := by ring
      _ ≤ τ * Real.exp (-1) + 1 * |r| := by
          refine add_le_add ?_ (mul_le_mul_of_nonneg_right hK1 (abs_nonneg _))
          have := abs_laplaceKernel_mul_coord_le τ hτ r y
          rwa [abs_mul, abs_of_nonneg hK] at this
      _ = τ * Real.exp (-1) + |r| := by ring
  have hf : Integrable (fun y : EuclideanSpace ℝ (Fin 3) =>
      laplaceKernel τ (rayProbe r) y * y 0) (radialMixture₃ ν) :=
    ⟨hcont.aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1) + |r|)
        (ae_of_all _ fun y => by rw [Real.norm_eq_abs]; exact habs y)⟩
  rw [radialRayT₃, integral_radialMixture₃ ν hf]
  refine (integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))).symm
  refine integral_chartBase_zonal
    (G := fun w : ℝ × ℝ => laplaceKernel τ (rayProbe r) (chartMap (s, w))
      * (chartMap (s, w)) 0)
    (g := fun u => Real.exp (-(1 / τ) * shellDist r s u) * (s * u))
    (C := τ * Real.exp (-1) + |r|) ?_ ?_ ?_
  · exact hcont.comp (continuous_chartMap.comp (f := fun w : ℝ × ℝ => ((s, w) : ℝ × ℝ × ℝ))
      (by fun_prop))
  · intro w
    exact habs _
  · intro u φ hu
    have hu2 : u ^ 2 ≤ 1 := by nlinarith [hu.1, hu.2]
    simp only [chartMap_mk_pair, laplaceKernel_rayProbe_chart hu2 τ r s φ,
      PiLp.smul_apply, sphereChart_apply_zero, smul_eq_mul]

/-- **`P̃` bridge**: the tangential `ρ²/d` ray profile is the y-level integral
`∫ K·(ρ²/d)`. -/
lemma radialRayRhoSqOverDist₃_eq_integral (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayRhoSqOverDist₃ τ ν r
      = ∫ y, laplaceKernel τ (rayProbe r) y
          * (((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖) ∂(radialMixture₃ ν) := by
  have hmeas := measurable_laplaceKernel_mul_rhoSq_div τ r
  have hf : Integrable (fun y : EuclideanSpace ℝ (Fin 3) =>
      laplaceKernel τ (rayProbe r) y
        * (((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖)) (radialMixture₃ ν) :=
    ⟨hmeas.aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
        (ae_of_all _ fun y => by
          rw [Real.norm_eq_abs]
          exact abs_laplaceKernel_mul_rhoSq_div_le τ hτ r y)⟩
  rw [radialRayRhoSqOverDist₃, integral_radialMixture₃ ν hf]
  refine (integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))).symm
  refine integral_chartBase_zonal_measurable
    (G := fun w : ℝ × ℝ => laplaceKernel τ (rayProbe r) (chartMap (s, w))
      * ((((chartMap (s, w)) 1) ^ 2 + ((chartMap (s, w)) 2) ^ 2)
        / ‖rayProbe r - chartMap (s, w)‖))
    (g := fun u => Real.exp (-(1 / τ) * shellDist r s u)
      * (shellRhoSq s u / shellDist r s u))
    (C := τ * Real.exp (-1)) ?_ ?_ ?_
  · have hsec : Measurable (fun w : ℝ × ℝ => chartMap (s, w)) :=
      (continuous_chartMap.comp (f := fun w : ℝ × ℝ => ((s, w) : ℝ × ℝ × ℝ))
        (by fun_prop)).measurable
    exact hmeas.comp hsec
  · intro w
    exact abs_laplaceKernel_mul_rhoSq_div_le τ hτ r _
  · intro u φ hu
    have hu2 : u ^ 2 ≤ 1 := by nlinarith [hu.1, hu.2]
    have h1u : (0 : ℝ) ≤ 1 - u ^ 2 := by linarith
    rw [chartMap_mk_pair, laplaceKernel_rayProbe_chart hu2 τ r s φ,
      norm_rayProbe_sub_smul_sphereChart hu2]
    congr 2
    have hsq : Real.sqrt (1 - u ^ 2) ^ 2 = 1 - u ^ 2 := Real.sq_sqrt h1u
    have hcs : Real.cos φ ^ 2 + Real.sin φ ^ 2 = 1 := Real.cos_sq_add_sin_sq φ
    simp only [PiLp.smul_apply, sphereChart_apply_one, sphereChart_apply_two, smul_eq_mul]
    unfold shellRhoSq
    linear_combination (s ^ 2 * (Real.cos φ ^ 2 + Real.sin φ ^ 2)) * hsq
      + (s ^ 2 * (1 - u ^ 2)) * hcs

lemma shellT_zero_right (τ r : ℝ) : shellT τ r 0 = 0 := by
  unfold shellT
  simp

lemma shellRhoSqOverDist_zero_right (τ r : ℝ) : shellRhoSqOverDist τ r 0 = 0 := by
  unfold shellRhoSqOverDist shellRhoSq
  simp

/-- Supported-on-`[0,∞)` gives a.e. nonnegativity of the shell radius. -/
lemma radial_ae_nonneg {ν : Measure ℝ} (hsupp : ν (Iio 0) = 0) :
    ∀ᵐ s ∂ν, 0 ≤ s := by
  rw [ae_iff]
  convert hsupp using 2
  ext s
  simp [not_le]

/-- **The mixture T₃ identity**: `P̃ = (2τ/r)·T̃` for radial mixtures supported
on `[0,∞)`.  Per-shell this is the closed T₃ theorem of the shell layer; the
`s = 0` origin atom contributes zero to both sides. -/
theorem radialRayRhoSqOverDist₃_eq_T (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    radialRayRhoSqOverDist₃ τ ν r = (2 * τ / r) * radialRayT₃ τ ν r := by
  rw [radialRayRhoSqOverDist₃, radialRayT₃, ← integral_const_mul]
  refine integral_congr_ae ?_
  filter_upwards [radial_ae_nonneg hsupp] with s hs
  rcases eq_or_lt_of_le hs with h0 | hs0
  · rw [← h0, shellRhoSqOverDist_zero_right, shellT_zero_right, mul_zero]
  · exact shellRhoSqOverDist_eq_two_tau_div_r_mul_shellT hτ hr hs0

/-- **The axial split**: `T̃ = D̃ + r·Z̃`. -/
lemma radialRayT₃_eq_D_add_Z (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayT₃ τ ν r = radialRayD₃ τ ν r + r * radialRayZ₃ τ ν r := by
  rw [radialRayT₃_eq_integral τ hτ ν r, radialRayD₃_eq_integral_coord τ hτ ν r,
    radialRayZ₃_eq_kernelNormalizer τ hτ ν r]
  have hZint : Integrable (fun y => laplaceKernel τ (rayProbe r) y) (radialMixture₃ ν) :=
    integrable_laplaceKernel_rayProbe τ hτ _ r
  have hDint : Integrable (fun y => laplaceKernel τ (rayProbe r) y * (y 0 - r))
      (radialMixture₃ ν) := integrable_laplaceKernel_mul_coord τ hτ _ r
  have hsplit : (∫ y, laplaceKernel τ (rayProbe r) y * y 0 ∂(radialMixture₃ ν))
      = (∫ y, (laplaceKernel τ (rayProbe r) y * (y 0 - r)
          + r * laplaceKernel τ (rayProbe r) y) ∂(radialMixture₃ ν)) := by
    refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
    ring
  rw [hsplit, integral_add hDint (hZint.const_mul r), integral_const_mul]
  rfl

/-- **The closure identity (F4), algebraic form**:
`C̃ = Q̃ + 3τ·Z̃ + (2τ/r)·D̃` on the open ray, for radial mixtures supported on
`[0,∞)`.  With `hasDerivAt_radialRayD₃` this is exactly
`C̃ = τ·D̃' + 4τ·Z̃ + (2τ/r)·D̃`. -/
theorem radialRayC₃_closure (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    radialRayC₃ τ ν r
      = radialRayQ₃ τ ν r + 3 * τ * radialRayZ₃ τ ν r
        + (2 * τ / r) * radialRayD₃ τ ν r := by
  have hZint : Integrable (fun y => laplaceKernel τ (rayProbe r) y) (radialMixture₃ ν) :=
    integrable_laplaceKernel_rayProbe τ hτ _ r
  have hQint : Integrable (fun y : EuclideanSpace ℝ (Fin 3) =>
      laplaceKernel τ (rayProbe r) y * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖))
      (radialMixture₃ ν) :=
    ⟨(measurable_laplaceKernel_mul_sq_div τ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
        (ae_of_all _ fun y => by
          rw [Real.norm_eq_abs]
          exact abs_laplaceKernel_mul_sq_div_le τ hτ r y)⟩
  have hPint : Integrable (fun y : EuclideanSpace ℝ (Fin 3) =>
      laplaceKernel τ (rayProbe r) y
        * (((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖)) (radialMixture₃ ν) :=
    ⟨(measurable_laplaceKernel_mul_rhoSq_div τ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
        (ae_of_all _ fun y => by
          rw [Real.norm_eq_abs]
          exact abs_laplaceKernel_mul_rhoSq_div_le τ hτ r y)⟩
  -- Step 1: the companion split `C̃ = τ·Z̃ + ∫ K·d`.
  have hCsplit : radialRayC₃ τ ν r
      = τ * radialRayZ₃ τ ν r
        + ∫ y, laplaceKernel τ (rayProbe r) y * ‖rayProbe r - y‖ ∂(radialMixture₃ ν) := by
    rw [radialRayC₃_eq_companionNormalizer τ hτ ν r,
      radialRayZ₃_eq_kernelNormalizer τ hτ ν r]
    have hdint : Integrable (fun y : EuclideanSpace ℝ (Fin 3) =>
        laplaceKernel τ (rayProbe r) y * ‖rayProbe r - y‖) (radialMixture₃ ν) := by
      refine ⟨?_, HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
        (ae_of_all _ fun y => ?_)⟩
      · have : Continuous (fun y : EuclideanSpace ℝ (Fin 3) =>
            laplaceKernel τ (rayProbe r) y * ‖rayProbe r - y‖) := by
          have h1 := continuous_laplaceKernel_rayProbe τ r
          have h2 : Continuous fun y : EuclideanSpace ℝ (Fin 3) => ‖rayProbe r - y‖ :=
            (continuous_const.sub continuous_id).norm
          exact h1.mul h2
        exact this.aestronglyMeasurable
      · rw [Real.norm_eq_abs, abs_mul,
          abs_of_nonneg (laplaceKernel_rayProbe_nonneg τ r y),
          abs_of_nonneg (norm_nonneg _)]
        calc laplaceKernel τ (rayProbe r) y * ‖rayProbe r - y‖
            = ‖rayProbe r - y‖ * Real.exp (-‖rayProbe r - y‖ / τ) := by
              simp only [laplaceKernel]
              rw [show -(1 / τ) * ‖rayProbe r - y‖ = -‖rayProbe r - y‖ / τ by ring]
              ring
          _ ≤ τ * Real.exp (-1) := mul_exp_neg_div_le hτ (norm_nonneg _)
    have hsplit : kernelNormalizer (laplaceCompanionKernel τ) (radialMixture₃ ν) (rayProbe r)
        = ∫ y, (τ * laplaceKernel τ (rayProbe r) y
            + laplaceKernel τ (rayProbe r) y * ‖rayProbe r - y‖) ∂(radialMixture₃ ν) := by
      refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
      simp only [laplaceCompanionKernel]
      ring
    rw [hsplit, integral_add (hZint.const_mul τ) hdint, integral_const_mul]
    rfl
  -- Step 2: the cylindrical split `∫ K·d = Q̃ + P̃` (junk-safe pointwise).
  have hdsplit : (∫ y, laplaceKernel τ (rayProbe r) y * ‖rayProbe r - y‖
        ∂(radialMixture₃ ν))
      = radialRayQ₃ τ ν r
        + ∫ y, laplaceKernel τ (rayProbe r) y
            * (((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖) ∂(radialMixture₃ ν) := by
    rw [radialRayQ₃, ← integral_add hQint hPint]
    refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
    rcases eq_or_ne (‖rayProbe r - y‖) 0 with h0 | hne
    · simp only [h0, div_zero, mul_zero, add_zero]
    · have hd : (y 0 - r) ^ 2 / ‖rayProbe r - y‖
          + ((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖ = ‖rayProbe r - y‖ := by
        rw [← add_div, ← norm_rayProbe_sub_sq_eq r y, pow_two, mul_div_assoc,
          div_self hne, mul_one]
      calc laplaceKernel τ (rayProbe r) y * ‖rayProbe r - y‖
          = laplaceKernel τ (rayProbe r) y
            * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖
              + ((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖) := by rw [hd]
        _ = laplaceKernel τ (rayProbe r) y * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖)
            + laplaceKernel τ (rayProbe r) y
              * (((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖) := by ring
  -- Step 3: assemble via the T₃ mixture and the axial split.
  have hP : (∫ y, laplaceKernel τ (rayProbe r) y
        * (((y 1) ^ 2 + (y 2) ^ 2) / ‖rayProbe r - y‖) ∂(radialMixture₃ ν))
      = (2 * τ / r) * (radialRayD₃ τ ν r + r * radialRayZ₃ τ ν r) := by
    rw [← radialRayRhoSqOverDist₃_eq_integral τ hτ ν r,
      radialRayRhoSqOverDist₃_eq_T τ hτ ν hsupp hr,
      radialRayT₃_eq_D_add_Z τ hτ ν r]
  rw [hCsplit, hdsplit, hP]
  field_simp
  ring

end DriftingIdentifiability
