import DriftingIdentifiability.LaplaceRadialSystem3
import DriftingIdentifiability.LaplaceACPropagation
import DriftingIdentifiability.LaplaceACFinal

/-!
# Radial Laplace converse, milestone L5: propagation and endgame (`n = 3`)

This file consumes the K̂ Abel system of `LaplaceRadialSystem3.lean` through the
abstract propagation layer (`LaplaceACPropagation.lean`), mirroring the 1-d
trichotomy driver `LaplaceUnconditionalConverse.lean:882–1010`:

* the coefficient `c = (m̃'+4)/m̃ = 2μ̂/m̃` (`μ̂ = (m̃'+4)/2`) is **continuous**
  on `{m̃ ≠ 0} ∩ (0,∞)` — unlike 1-d there are no one-sided contortions, so the
  primitive `A` comes from the plain FTC;
* the sign layer supplies `μ̂ ≥ ½` on `{m̃ ≤ r}` (proved) and on `{m̃ > r}`
  (`RadialSlack₃`), covering every edge with the same margins as 1-d;
* `r = 0` is a universal edge (`|K̂| ≤ τr²`, `m̃ ≤ L·r` near 0), and `K̂ → 0`
  at `+∞` under first moments.

Design record: `LaplaceHigherDim.md §4.10 (F7–F9)`.
-/

open MeasureTheory Filter Topology Set
open scoped RealInnerProductSpace

namespace DriftingIdentifiability
open Paper

/-! ## Continuity of the derivative payloads (hence of `m̃'`) on the open ray -/

lemma continuous_norm_rayProbe_sub (y : EuclideanSpace ℝ (Fin 3)) :
    Continuous fun r : ℝ => ‖rayProbe r - y‖ := by
  have hfe : (fun r : ℝ => ‖rayProbe r - y‖)
      = fun r : ℝ => Real.sqrt ((r - y 0) ^ 2 + ((y 1) ^ 2 + (y 2) ^ 2)) := by
    funext r
    exact norm_rayProbe_sub_eq_sqrt r y
  rw [hfe]
  fun_prop

lemma continuous_laplaceKernel_rayProbe_left (τ : ℝ)
    (y : EuclideanSpace ℝ (Fin 3)) :
    Continuous fun r : ℝ => laplaceKernel τ (rayProbe r) y := by
  simp only [laplaceKernel]
  exact Real.continuous_exp.comp
    ((continuous_norm_rayProbe_sub y).const_mul (-(1 / τ)))

/-- `Q̃` is continuous at every positive radius (dominated continuity; the
collision set is null). -/
lemma continuousAt_radialRayQ₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayQ₃ τ ν) r := by
  have hfe : radialRayQ₃ τ ν = fun x =>
      ∫ y, laplaceKernel τ (rayProbe x) y * ((y 0 - x) ^ 2 / ‖rayProbe x - y‖)
        ∂(radialMixture₃ ν) := by
    funext x
    rw [radialRayQ₃]
  rw [hfe]
  refine continuousAt_of_dominated (bound := fun _ => τ * Real.exp (-1)) ?_ ?_ ?_ ?_
  · exact Filter.Eventually.of_forall fun x =>
      (measurable_laplaceKernel_mul_sq_div τ x).aestronglyMeasurable
  · refine Filter.Eventually.of_forall fun x => ae_of_all _ fun y => ?_
    rw [Real.norm_eq_abs]
    exact abs_laplaceKernel_mul_sq_div_le τ hτ x y
  · exact integrable_const _
  · filter_upwards [radialMixture₃_ae_probe_ne ν] with y hy
    have hne : ‖rayProbe r - y‖ ≠ 0 := hy r hr
    have h1 : ContinuousAt (fun x : ℝ => laplaceKernel τ (rayProbe x) y) r :=
      (continuous_laplaceKernel_rayProbe_left τ y).continuousAt
    have h2 : ContinuousAt (fun x : ℝ => (y 0 - x) ^ 2 / ‖rayProbe x - y‖) r := by
      refine ContinuousAt.div ?_ ?_ hne
      · fun_prop
      · exact (continuous_norm_rayProbe_sub y).continuousAt
    exact h1.mul h2

/-- `Z̃d` is continuous at every positive radius. -/
lemma continuousAt_radialRayZd₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayZd₃ τ ν) r := by
  have hfe : radialRayZd₃ τ ν = fun x =>
      ∫ y, laplaceKernel τ (rayProbe x) y * ((y 0 - x) / ‖rayProbe x - y‖)
        ∂(radialMixture₃ ν) := by
    funext x
    rw [radialRayZd₃]
  rw [hfe]
  refine continuousAt_of_dominated (bound := fun _ => (1 : ℝ)) ?_ ?_ ?_ ?_
  · exact Filter.Eventually.of_forall fun x =>
      (measurable_laplaceKernel_mul_coord_div τ x).aestronglyMeasurable
  · refine Filter.Eventually.of_forall fun x => ae_of_all _ fun y => ?_
    rw [Real.norm_eq_abs, abs_mul, abs_of_nonneg (laplaceKernel_rayProbe_nonneg τ x y)]
    calc laplaceKernel τ (rayProbe x) y * |(y 0 - x) / ‖rayProbe x - y‖|
        ≤ 1 * 1 := mul_le_mul (laplaceKernel_rayProbe_le_one τ hτ x y)
          (abs_coord_div_norm_rayProbe_le_one x y) (abs_nonneg _) zero_le_one
      _ = 1 := by ring
  · exact integrable_const (1 : ℝ)
  · filter_upwards [radialMixture₃_ae_probe_ne ν] with y hy
    have hne : ‖rayProbe r - y‖ ≠ 0 := hy r hr
    have h1 : ContinuousAt (fun x : ℝ => laplaceKernel τ (rayProbe x) y) r :=
      (continuous_laplaceKernel_rayProbe_left τ y).continuousAt
    have h2 : ContinuousAt (fun x : ℝ => (y 0 - x) / ‖rayProbe x - y‖) r := by
      refine ContinuousAt.div ?_ ?_ hne
      · fun_prop
      · exact (continuous_norm_rayProbe_sub y).continuousAt
    exact h1.mul h2

/-- `m̃'` (the explicit quotient-rule value) is continuous at every positive
radius. -/
lemma continuousAt_radialRayMDeriv₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayMDeriv₃ τ ν) r := by
  have hZ : ContinuousAt (radialRayZ₃ τ ν) r :=
    (hasDerivAt_radialRayZ₃ τ hτ ν hr).continuousAt
  have hD : ContinuousAt (radialRayD₃ τ ν) r :=
    (hasDerivAt_radialRayD₃ τ hτ ν hr).continuousAt
  have hQ := continuousAt_radialRayQ₃ τ hτ ν hr
  have hZd := continuousAt_radialRayZd₃ τ hτ ν hr
  have hZne : (radialRayZ₃ τ ν r) ^ 2 ≠ 0 :=
    pow_ne_zero 2 (radialRayZ₃_pos τ hτ ν r).ne'
  have hfe : radialRayMDeriv₃ τ ν = fun x =>
      (((1 / τ) * radialRayQ₃ τ ν x - radialRayZ₃ τ ν x) * radialRayZ₃ τ ν x
        - radialRayD₃ τ ν x * ((1 / τ) * radialRayZd₃ τ ν x))
        / (radialRayZ₃ τ ν x) ^ 2 := rfl
  rw [hfe]
  exact ((((hQ.const_mul _).sub hZ).mul hZ).sub
    (hD.mul (hZd.const_mul _))).div (hZ.pow 2) hZne

/-- `m̃` is continuous at every positive radius. -/
lemma continuousAt_radialRayM₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayM₃ τ ν) r := by
  have hZ : ContinuousAt (radialRayZ₃ τ ν) r :=
    (hasDerivAt_radialRayZ₃ τ hτ ν hr).continuousAt
  have hD : ContinuousAt (radialRayD₃ τ ν) r :=
    (hasDerivAt_radialRayD₃ τ hτ ν hr).continuousAt
  have hfe : radialRayM₃ τ ν = fun x => radialRayD₃ τ ν x / radialRayZ₃ τ ν x := rfl
  rw [hfe]
  exact hD.div hZ (radialRayZ₃_pos τ hτ ν r).ne'

/-! ## The Abel shape on `{m̃ ≠ 0}` -/

/-- **The Abel ODE for `K̂`**: on `{m̃ ≠ 0} ∩ (0,∞)`,
`K̂' = −(2μ̂/m̃)·K̂` with `μ̂ = (m̃'+4)/2` — exactly the coefficient consumed
by the 1-d propagation wrappers. -/
theorem hasDerivAt_radialRayKhat₃_abel (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₃ νp) (radialMixture₃ νq))
    {t : ℝ} (ht : 0 < t) (hmt : radialRayM₃ τ νp t ≠ 0) :
    HasDerivAt (radialRayKhat₃ τ νp νq)
      (-((2 * ((radialRayMDeriv₃ τ νp t + 4) / 2)) / radialRayM₃ τ νp t)
        * radialRayKhat₃ τ νp νq t) t := by
  have hK := hasDerivAt_radialRayKhat₃ τ νp νq hτ hsp hsq hzero ht
  have hKeq := radialRayKhat₃_eq_M_mul_V τ νp νq hτ hsp hsq hzero ht
  have hval : -(τ * (radialRayMDeriv₃ τ νp t + 4)) * radialRayV₃ τ νp νq t
      = -((2 * ((radialRayMDeriv₃ τ νp t + 4) / 2)) / radialRayM₃ τ νp t)
        * radialRayKhat₃ τ νp νq t := by
    rw [hKeq]
    field_simp [hmt]
  rw [hval] at hK
  exact hK

end DriftingIdentifiability
