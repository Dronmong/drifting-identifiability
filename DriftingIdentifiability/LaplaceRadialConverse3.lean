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

/-! ## The `r = 0` universal edge: providers -/

/-- The axial displacement average vanishes at the origin probe (odd
`u`-integrand). -/
lemma shellD_zero_left (τ s : ℝ) : shellD τ 0 s = 0 := by
  unfold shellD
  have hdist : ∀ u : ℝ, shellDist 0 s u = |s| := by
    intro u
    unfold shellDist
    rw [show (0:ℝ) ^ 2 + s ^ 2 - 2 * 0 * s * u = s ^ 2 by ring]
    exact Real.sqrt_sq_eq_abs s
  have hcong : (∫ u in Ioc (-1 : ℝ) 1,
        Real.exp (-(1 / τ) * shellDist 0 s u) * (s * u - 0))
      = ∫ u in Ioc (-1 : ℝ) 1, (Real.exp (-(1 / τ) * |s|) * s) * u := by
    refine setIntegral_congr_fun measurableSet_Ioc fun u _ => ?_
    rw [hdist u]
    ring
  rw [hcong, integral_const_mul, integral_Ioc_neg_one_one_eq_interval]
  rw [integral_id]
  norm_num

/-- `D̃(0) = 0`: the drift numerator vanishes at the origin. -/
lemma radialRayD₃_zero (τ : ℝ) (ν : Measure ℝ) : radialRayD₃ τ ν 0 = 0 := by
  rw [radialRayD₃]
  have : (fun s => shellD τ 0 s) = fun _ => (0 : ℝ) := funext fun s => shellD_zero_left τ s
  rw [this]
  simp

/-- The second-moment payload is globally bounded: `|Q̃| ≤ τ·e⁻¹`. -/
lemma abs_radialRayQ₃_le (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    |radialRayQ₃ τ ν r| ≤ τ * Real.exp (-1) := by
  rw [radialRayQ₃]
  have hint : Integrable (fun y : EuclideanSpace ℝ (Fin 3) =>
      laplaceKernel τ (rayProbe r) y * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖))
      (radialMixture₃ ν) :=
    ⟨(measurable_laplaceKernel_mul_sq_div τ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := τ * Real.exp (-1))
        (ae_of_all _ fun y => by
          rw [Real.norm_eq_abs]
          exact abs_laplaceKernel_mul_sq_div_le τ hτ r y)⟩
  have h1 := norm_integral_le_integral_norm (μ := radialMixture₃ ν)
    (f := fun y => laplaceKernel τ (rayProbe r) y * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖))
  simp only [Real.norm_eq_abs] at h1
  refine h1.trans ?_
  calc (∫ y, |laplaceKernel τ (rayProbe r) y * ((y 0 - r) ^ 2 / ‖rayProbe r - y‖)|
        ∂(radialMixture₃ ν))
      ≤ ∫ _, τ * Real.exp (-1) ∂(radialMixture₃ ν) :=
        integral_mono hint.abs (integrable_const _)
          (fun y => abs_laplaceKernel_mul_sq_div_le τ hτ r y)
    _ = τ * Real.exp (-1) := by simp

/-- `Z̃` is globally continuous. -/
lemma continuous_radialRayZ₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    Continuous (radialRayZ₃ τ ν) := by
  have hfe : radialRayZ₃ τ ν
      = fun x => ∫ y, laplaceKernel τ (rayProbe x) y ∂(radialMixture₃ ν) := by
    funext x
    rw [radialRayZ₃_eq_kernelNormalizer τ hτ ν x]
    rfl
  rw [hfe]
  refine continuous_of_dominated (bound := fun _ => (1 : ℝ)) (fun x => ?_) (fun x => ?_) ?_ ?_
  · exact (continuous_laplaceKernel_rayProbe τ x).aestronglyMeasurable
  · refine ae_of_all _ fun y => ?_
    rw [Real.norm_eq_abs, abs_of_nonneg (laplaceKernel_rayProbe_nonneg τ x y)]
    exact laplaceKernel_rayProbe_le_one τ hτ x y
  · exact integrable_const 1
  · exact ae_of_all _ fun y => continuous_laplaceKernel_rayProbe_left τ y

/-- `D̃` is globally continuous. -/
lemma continuous_radialRayD₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    Continuous (radialRayD₃ τ ν) := by
  have hfe : radialRayD₃ τ ν
      = fun x => ∫ y, laplaceKernel τ (rayProbe x) y * (y 0 - x)
          ∂(radialMixture₃ ν) := by
    funext x
    exact radialRayD₃_eq_integral_coord τ hτ ν x
  rw [hfe]
  refine continuous_of_dominated (bound := fun _ => τ * Real.exp (-1))
    (fun x => ?_) (fun x => ?_) ?_ ?_
  · have hc : Continuous (fun y : EuclideanSpace ℝ (Fin 3) =>
        laplaceKernel τ (rayProbe x) y * (y 0 - x)) := by
      have := continuous_laplaceKernel_rayProbe τ x
      fun_prop
    exact hc.aestronglyMeasurable
  · refine ae_of_all _ fun y => ?_
    rw [Real.norm_eq_abs]
    exact abs_laplaceKernel_mul_coord_le τ hτ x y
  · exact integrable_const _
  · refine ae_of_all _ fun y => ?_
    have h1 := continuous_laplaceKernel_rayProbe_left τ y
    fun_prop

/-- **The origin linear bound**: `|D̃(t)| ≤ (e⁻¹+1)·t` for `t ≥ 0`, from the
global derivative bound `|D̃'| ≤ e⁻¹+1` and `D̃(0) = 0`. -/
lemma abs_radialRayD₃_le_linear (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] {t : ℝ} (ht : 0 < t) :
    |radialRayD₃ τ ν t| ≤ (Real.exp (-1) + 1) * t := by
  have hC : (0 : ℝ) ≤ Real.exp (-1) + 1 := by positivity
  have hbound : ∀ x : ℝ, 0 < x →
      ‖(1 / τ) * radialRayQ₃ τ ν x - radialRayZ₃ τ ν x‖ ≤ Real.exp (-1) + 1 := by
    intro x _
    rw [Real.norm_eq_abs]
    have h1 : |(1 / τ) * radialRayQ₃ τ ν x| ≤ Real.exp (-1) := by
      rw [abs_mul, abs_of_nonneg (by positivity : (0:ℝ) ≤ 1 / τ)]
      have := abs_radialRayQ₃_le τ hτ ν x
      calc (1 / τ) * |radialRayQ₃ τ ν x| ≤ (1 / τ) * (τ * Real.exp (-1)) :=
            mul_le_mul_of_nonneg_left this (by positivity)
        _ = Real.exp (-1) := by field_simp
    have h2 : |radialRayZ₃ τ ν x| ≤ 1 := by
      rw [abs_of_nonneg (radialRayZ₃_nonneg τ hτ ν x)]
      exact radialRayZ₃_le_one τ hτ ν x
    calc |(1 / τ) * radialRayQ₃ τ ν x - radialRayZ₃ τ ν x|
        ≤ |(1 / τ) * radialRayQ₃ τ ν x| + |radialRayZ₃ τ ν x| := abs_sub _ _
      _ ≤ Real.exp (-1) + 1 := add_le_add h1 h2
  have hev : ∀ᶠ ε in 𝓝[>] (0 : ℝ), |radialRayD₃ τ ν t|
      ≤ |radialRayD₃ τ ν ε| + (Real.exp (-1) + 1) * t := by
    filter_upwards [Ioo_mem_nhdsGT ht] with ε hε
    have hMVT : ‖radialRayD₃ τ ν t - radialRayD₃ τ ν ε‖
        ≤ (Real.exp (-1) + 1) * ‖t - ε‖ := by
      refine Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
        (f' := fun x => (1 / τ) * radialRayQ₃ τ ν x - radialRayZ₃ τ ν x)
        (fun x hx => ?_) (fun x hx => hbound x (lt_of_lt_of_le hε.1 hx.1))
        (convex_Icc ε t) (left_mem_Icc.mpr hε.2.le) (right_mem_Icc.mpr hε.2.le)
      exact (hasDerivAt_radialRayD₃ τ hτ ν
        (lt_of_lt_of_le hε.1 hx.1)).hasDerivWithinAt
    rw [Real.norm_eq_abs, Real.norm_eq_abs,
      abs_of_nonneg (by linarith [hε.2] : (0:ℝ) ≤ t - ε)] at hMVT
    calc |radialRayD₃ τ ν t|
        ≤ |radialRayD₃ τ ν ε| + |radialRayD₃ τ ν t - radialRayD₃ τ ν ε| := by
          have := abs_sub_abs_le_abs_sub (radialRayD₃ τ ν t) (radialRayD₃ τ ν ε)
          linarith [abs_nonneg (radialRayD₃ τ ν ε)]
      _ ≤ |radialRayD₃ τ ν ε| + (Real.exp (-1) + 1) * (t - ε) := by linarith
      _ ≤ |radialRayD₃ τ ν ε| + (Real.exp (-1) + 1) * t := by nlinarith [hε.1]
  have htend : Tendsto (fun ε : ℝ => |radialRayD₃ τ ν ε| + (Real.exp (-1) + 1) * t)
      (𝓝[>] (0 : ℝ)) (𝓝 ((Real.exp (-1) + 1) * t)) := by
    have h0 : Tendsto (fun ε : ℝ => |radialRayD₃ τ ν ε|) (𝓝[>] (0 : ℝ)) (𝓝 0) := by
      have hc : ContinuousAt (fun ε : ℝ => |radialRayD₃ τ ν ε|) 0 :=
        ((continuous_radialRayD₃ τ hτ ν).abs).continuousAt
      have h1 : Tendsto (fun ε : ℝ => |radialRayD₃ τ ν ε|) (𝓝[>] (0 : ℝ))
          (𝓝 |radialRayD₃ τ ν 0|) :=
        hc.tendsto.mono_left nhdsWithin_le_nhds
      rwa [radialRayD₃_zero, abs_zero] at h1
    have := h0.add (tendsto_const_nhds (α := ℝ) (x := (Real.exp (-1) + 1) * t)
      (f := 𝓝[>] (0:ℝ)))
    simpa using this
  exact ge_of_tendsto htend hev

/-- **The origin linear `m̃`-bound**: on `(0, b]`, `m̃(t) ≤ L·t` with
`L = (e⁻¹+1)/min Z̃`. -/
lemma radialRayM₃_le_linear (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] {b : ℝ} (hb : 0 < b) :
    ∃ L : ℝ, 0 < L ∧ ∀ t : ℝ, 0 < t → t ≤ b → radialRayM₃ τ ν t ≤ L * t := by
  obtain ⟨z₀, hz₀mem, hz₀min⟩ := isCompact_Icc.exists_isMinOn
    (Set.nonempty_Icc.mpr hb.le) ((continuous_radialRayZ₃ τ hτ ν).continuousOn)
  have hZmin : 0 < radialRayZ₃ τ ν z₀ := radialRayZ₃_pos τ hτ ν z₀
  refine ⟨(Real.exp (-1) + 1) / radialRayZ₃ τ ν z₀, by positivity, ?_⟩
  intro t ht htb
  have hZt : radialRayZ₃ τ ν z₀ ≤ radialRayZ₃ τ ν t :=
    hz₀min (Set.mem_Icc.mpr ⟨ht.le, htb⟩)
  have hZtpos : 0 < radialRayZ₃ τ ν t := radialRayZ₃_pos τ hτ ν t
  have hD := abs_radialRayD₃_le_linear τ hτ ν ht
  rw [radialRayM₃, div_le_iff₀ hZtpos]
  calc radialRayD₃ τ ν t ≤ |radialRayD₃ τ ν t| := le_abs_self _
    _ ≤ (Real.exp (-1) + 1) * t := hD
    _ = ((Real.exp (-1) + 1) / radialRayZ₃ τ ν z₀) * t * radialRayZ₃ τ ν z₀ := by
        field_simp
    _ ≤ ((Real.exp (-1) + 1) / radialRayZ₃ τ ν z₀) * t * radialRayZ₃ τ ν t := by
        refine mul_le_mul_of_nonneg_left hZt ?_
        positivity

/-- `K̂` is bounded as `r → 0⁺` (indeed `≤ τ` on `(0,1)`). -/
lemma radialRayKhat₃_bounded_near_zero (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq] :
    IsBoundedUnder (· ≤ ·) (𝓝[>] (0 : ℝ)) (norm ∘ radialRayKhat₃ τ νp νq) := by
  refine ⟨τ, ?_⟩
  rw [Filter.eventually_map]
  filter_upwards [Ioo_mem_nhdsGT one_pos] with r hr
  have h := abs_radialRayKhat₃_le τ hτ νp νq r
  have hr2 : r ^ 2 ≤ 1 := by nlinarith [hr.1, hr.2]
  calc ‖radialRayKhat₃ τ νp νq r‖ = |radialRayKhat₃ τ νp νq r| := Real.norm_eq_abs _
    _ ≤ τ * r ^ 2 := h
    _ ≤ τ * 1 := mul_le_mul_of_nonneg_left hr2 hτ.le
    _ = τ := mul_one τ

end DriftingIdentifiability
