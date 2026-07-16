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

/-! ## The propagation trichotomy: edge lemmas -/

section Trichotomy

variable (τ : ℝ) (νp νq : Measure ℝ)
  [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]

/-- **Right-interval propagation from a left edge** (an interior zero of `m̃`
or the origin): if `m̃ > 0` on `(a,b)` with the linear bound `m̃ ≤ L(t−a)`
near `a` and `K̂` bounded as `t → a⁺`, then `K̂ ≡ 0` on `(a,b)`. -/
lemma radialRayKhat₃_eq_zero_on_Ioo_of_leftEdge (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hslackp : RadialSlack₃ τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₃ νp) (radialMixture₃ νq))
    {a b r₂ L : ℝ} (ha : 0 ≤ a) (hab : a < b) (har₂ : a < r₂) (hr₂b : r₂ < b)
    (hL : 0 < L)
    (hmpos : ∀ t : ℝ, a < t → t < b → 0 < radialRayM₃ τ νp t)
    (hlin : ∀ t : ℝ, a < t → t < r₂ → radialRayM₃ τ νp t ≤ L * (t - a))
    (hKbdd : IsBoundedUnder (· ≤ ·) (𝓝[>] a)
      (norm ∘ radialRayKhat₃ τ νp νq)) :
    ∀ x : ℝ, a < x → x < b → radialRayKhat₃ τ νp νq x = 0 := by
  have hccontAt : ∀ t ∈ Ioo a b, ContinuousAt
      (fun t => (2 * ((radialRayMDeriv₃ τ νp t + 4) / 2)) / radialRayM₃ τ νp t) t := by
    intro t ht
    have ht0 : 0 < t := lt_of_le_of_lt ha ht.1
    have hMne : radialRayM₃ τ νp t ≠ 0 := (hmpos t ht.1 ht.2).ne'
    refine ContinuousAt.div ?_ (continuousAt_radialRayM₃ τ hτ νp ht0) hMne
    exact (((continuousAt_radialRayMDeriv₃ τ hτ νp ht0).add
      continuousAt_const).div_const 2).const_mul 2
  have hcOn : ContinuousOn
      (fun t => (2 * ((radialRayMDeriv₃ τ νp t + 4) / 2)) / radialRayM₃ τ νp t)
      (Ioo a b) := fun t ht => (hccontAt t ht).continuousWithinAt
  have hcint : ∀ u v : ℝ, u ∈ Ioo a b → v ∈ Ioo a b →
      IntervalIntegrable (fun t => (2 * ((radialRayMDeriv₃ τ νp t + 4) / 2))
        / radialRayM₃ τ νp t) volume u v := by
    intro u v hu hv
    exact ContinuousOn.intervalIntegrable
      (hcOn.mono (OrdConnected.uIcc_subset inferInstance hu hv))
  have hAderiv : ∀ t ∈ Ioo a b, HasDerivAt
      (fun z => ∫ s in r₂..z, (2 * ((radialRayMDeriv₃ τ νp s + 4) / 2))
        / radialRayM₃ τ νp s)
      ((2 * ((radialRayMDeriv₃ τ νp t + 4) / 2)) / radialRayM₃ τ νp t) t := by
    intro t ht
    exact intervalIntegral.integral_hasDerivAt_right
      (hcint r₂ t ⟨har₂, hr₂b⟩ ht)
      (hcOn.stronglyMeasurableAtFilter isOpen_Ioo t ht)
      (hccontAt t ht)
  refine abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper
    (W := radialRayKhat₃ τ νp νq)
    (A := fun z => ∫ s in r₂..z, (2 * ((radialRayMDeriv₃ τ νp s + 4) / 2))
      / radialRayM₃ τ νp s)
    (μDeriv := fun t => (radialRayMDeriv₃ τ νp t + 4) / 2)
    (m := radialRayM₃ τ νp)
    (a := a) (b := b) (r := r₂) (δ := 1 / 2) (L := L)
    hab har₂ hr₂b (by norm_num) hL ?_ ?_ ?_ ?_ hKbdd ?_ ?_ ?_
  · intro x y hx hxy hy
    have hsub : Icc x y ⊆ Ioo a b := fun t ht =>
      ⟨lt_of_lt_of_le hx ht.1, lt_of_le_of_lt ht.2 hy⟩
    refine ContinuousOn.mul ?_ (Real.continuous_exp.comp_continuousOn ?_)
    · intro t ht
      have ht0 : 0 < t := lt_of_le_of_lt ha (hsub ht).1
      exact ((hasDerivAt_radialRayKhat₃ τ νp νq hτ hsp hsq hzero
        ht0).continuousAt).continuousWithinAt
    · intro t ht
      exact ((hAderiv t (hsub ht)).continuousAt).continuousWithinAt
  · intro x y hx hxy hy t ht
    have ht' : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    have ht0 : 0 < t := lt_of_le_of_lt ha ht'.1
    have hMne : radialRayM₃ τ νp t ≠ 0 := (hmpos t ht'.1 ht'.2).ne'
    exact (hasDerivAt_radialRayKhat₃_abel τ hτ νp νq hsp hsq hzero
      ht0 hMne).hasDerivWithinAt
  · intro x y hx hxy hy t ht
    have ht' : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    exact (hAderiv t ht').hasDerivWithinAt
  · intro x y hx hxy hy
    have hsub : Icc x y ⊆ Ioo a b := fun t ht =>
      ⟨lt_of_lt_of_le hx ht.1, lt_of_le_of_lt ht.2 hy⟩
    intro t ht
    exact ((hAderiv t (hsub ht)).continuousAt).continuousWithinAt
  · intro t hat htr
    have ht0 : 0 < t := lt_of_le_of_lt ha hat
    have h3 := radialRayMDeriv₃_ge τ hτ νp hsp hslackp ht0
    linarith
  · intro t hat htr
    exact hmpos t hat (lt_trans htr hr₂b)
  · exact hlin

/-- **Left-interval propagation from a right edge** (an interior zero of `m̃`
at `b`): if `m̃ < 0` on `(a,b)` with the linear bound `−L(b−t) ≤ m̃` near `b`,
then `K̂ ≡ 0` on `(a,b)`. -/
lemma radialRayKhat₃_eq_zero_on_Ioo_of_rightEdge (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hslackp : RadialSlack₃ τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₃ νp) (radialMixture₃ νq))
    {a b l₂ L : ℝ} (ha : 0 < a) (hab : a < b) (hal₂ : a < l₂) (hl₂b : l₂ < b)
    (hL : 0 < L)
    (hmneg : ∀ t : ℝ, a < t → t < b → radialRayM₃ τ νp t < 0)
    (hlin : ∀ t : ℝ, l₂ ≤ t → t < b → -(L * (b - t)) ≤ radialRayM₃ τ νp t) :
    ∀ x : ℝ, a < x → x < b → radialRayKhat₃ τ νp νq x = 0 := by
  have hccontAt : ∀ t ∈ Ioo a b, ContinuousAt
      (fun t => (2 * ((radialRayMDeriv₃ τ νp t + 4) / 2)) / radialRayM₃ τ νp t) t := by
    intro t ht
    have ht0 : 0 < t := lt_trans ha ht.1
    have hMne : radialRayM₃ τ νp t ≠ 0 := (hmneg t ht.1 ht.2).ne
    refine ContinuousAt.div ?_ (continuousAt_radialRayM₃ τ hτ νp ht0) hMne
    exact (((continuousAt_radialRayMDeriv₃ τ hτ νp ht0).add
      continuousAt_const).div_const 2).const_mul 2
  have hcOn : ContinuousOn
      (fun t => (2 * ((radialRayMDeriv₃ τ νp t + 4) / 2)) / radialRayM₃ τ νp t)
      (Ioo a b) := fun t ht => (hccontAt t ht).continuousWithinAt
  have hcint : ∀ u v : ℝ, u ∈ Ioo a b → v ∈ Ioo a b →
      IntervalIntegrable (fun t => (2 * ((radialRayMDeriv₃ τ νp t + 4) / 2))
        / radialRayM₃ τ νp t) volume u v := by
    intro u v hu hv
    exact ContinuousOn.intervalIntegrable
      (hcOn.mono (OrdConnected.uIcc_subset inferInstance hu hv))
  have hAderiv : ∀ t ∈ Ioo a b, HasDerivAt
      (fun z => ∫ s in l₂..z, (2 * ((radialRayMDeriv₃ τ νp s + 4) / 2))
        / radialRayM₃ τ νp s)
      ((2 * ((radialRayMDeriv₃ τ νp t + 4) / 2)) / radialRayM₃ τ νp t) t := by
    intro t ht
    exact intervalIntegral.integral_hasDerivAt_right
      (hcint l₂ t ⟨hal₂, hl₂b⟩ ht)
      (hcOn.stronglyMeasurableAtFilter isOpen_Ioo t ht)
      (hccontAt t ht)
  have hb0 : 0 < b := lt_trans ha hab
  have hKbdd : IsBoundedUnder (· ≤ ·) (𝓝[<] b)
      (norm ∘ radialRayKhat₃ τ νp νq) :=
    (((hasDerivAt_radialRayKhat₃ τ νp νq hτ hsp hsq hzero
      hb0).continuousAt).norm.isBoundedUnder_le).mono nhdsWithin_le_nhds
  refine abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower
    (W := radialRayKhat₃ τ νp νq)
    (A := fun z => ∫ s in l₂..z, (2 * ((radialRayMDeriv₃ τ νp s + 4) / 2))
      / radialRayM₃ τ νp s)
    (μDeriv := fun t => (radialRayMDeriv₃ τ νp t + 4) / 2)
    (m := radialRayM₃ τ νp)
    (a := a) (b := b) (l := l₂) (δ := 1 / 2) (L := L)
    hab hal₂ hl₂b (by norm_num) hL ?_ ?_ ?_ ?_ hKbdd ?_ ?_ ?_
  · intro x y hx hxy hy
    have hsub : Icc x y ⊆ Ioo a b := fun t ht =>
      ⟨lt_of_lt_of_le hx ht.1, lt_of_le_of_lt ht.2 hy⟩
    refine ContinuousOn.mul ?_ (Real.continuous_exp.comp_continuousOn ?_)
    · intro t ht
      have ht0 : 0 < t := lt_trans ha (hsub ht).1
      exact ((hasDerivAt_radialRayKhat₃ τ νp νq hτ hsp hsq hzero
        ht0).continuousAt).continuousWithinAt
    · intro t ht
      exact ((hAderiv t (hsub ht)).continuousAt).continuousWithinAt
  · intro x y hx hxy hy t ht
    have ht' : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    have ht0 : 0 < t := lt_trans ha ht'.1
    have hMne : radialRayM₃ τ νp t ≠ 0 := (hmneg t ht'.1 ht'.2).ne
    exact (hasDerivAt_radialRayKhat₃_abel τ hτ νp νq hsp hsq hzero
      ht0 hMne).hasDerivWithinAt
  · intro x y hx hxy hy t ht
    have ht' : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    exact (hAderiv t ht').hasDerivWithinAt
  · intro x y hx hxy hy
    have hsub : Icc x y ⊆ Ioo a b := fun t ht =>
      ⟨lt_of_lt_of_le hx ht.1, lt_of_le_of_lt ht.2 hy⟩
    intro t ht
    exact ((hAderiv t (hsub ht)).continuousAt).continuousWithinAt
  · intro t hlt htb
    have ht0 : 0 < t := lt_trans ha (lt_of_lt_of_le hal₂ hlt)
    have h3 := radialRayMDeriv₃_ge τ hτ νp hsp hslackp ht0
    linarith
  · intro t hlt htb
    exact hmneg t (lt_of_lt_of_le hal₂ hlt) htb
  · exact hlin

/-- **Outer-ray propagation**: if `m̃ < 0` on all of `[a,∞)` then `K̂ ≡ 0`
there (the square/tail mechanism with `K̂ → 0` at `∞`). -/
lemma radialRayKhat₃_eq_zero_on_ray (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable id νp) (hmq : Integrable id νq)
    (hslackp : RadialSlack₃ τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixture₃ νp) (radialMixture₃ νq))
    {a : ℝ} (ha : 0 < a)
    (hmneg : ∀ t : ℝ, a ≤ t → radialRayM₃ τ νp t < 0) :
    ∀ x : ℝ, a ≤ x → radialRayKhat₃ τ νp νq x = 0 := by
  refine abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg
    (W := radialRayKhat₃ τ νp νq)
    (muDeriv := fun t => (radialRayMDeriv₃ τ νp t + 4) / 2)
    (m := radialRayM₃ τ νp) (a := a) ?_ ?_ ?_ hmneg
    (tendsto_radialRayKhat₃_atTop τ hτ νp νq hsp hsq hmp hmq)
  · intro x b hax hxb t ht
    have ht0 : 0 < t := lt_of_lt_of_le ha (le_trans hax ht.1)
    exact ((hasDerivAt_radialRayKhat₃ τ νp νq hτ hsp hsq hzero
      ht0).continuousAt).continuousWithinAt
  · intro x b hax hxb t ht
    have ht0 : 0 < t := lt_of_lt_of_le ha (le_trans hax ht.1)
    have hMne : radialRayM₃ τ νp t ≠ 0 := (hmneg t (le_trans hax ht.1)).ne
    exact (hasDerivAt_radialRayKhat₃_abel τ hτ νp νq hsp hsq hzero
      ht0 hMne).hasDerivWithinAt
  · intro t hat
    have ht0 : 0 < t := lt_of_lt_of_le ha hat
    have h3 := radialRayMDeriv₃_ge τ hτ νp hsp hslackp ht0
    linarith

end Trichotomy

end DriftingIdentifiability
