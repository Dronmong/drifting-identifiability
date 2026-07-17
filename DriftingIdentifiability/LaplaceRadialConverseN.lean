import DriftingIdentifiability.LaplaceRadialSystemN
import DriftingIdentifiability.LaplaceACPropagation

/-!
# Radial Laplace converse in general dimension

This file is the propagation layer for roadmap milestone G1.  It consumes the
general-`n` ray system proved in `LaplaceRadialSystemN.lean`.  In particular,
the first theorem below rewrites the differentiated weighted determinant into
the Abel equation used by the already-audited one-dimensional propagation
machinery.

No new analytic fact is assumed here.  The only external theorem below the
import boundary is the separately audited Haar-sphere coordinate formula used
to construct the physical general-`n` ray profiles.
-/

open MeasureTheory Filter Topology Set
open scoped RealInnerProductSpace

namespace DriftingIdentifiability
open Paper

/-! ## The Abel equation on `{M ≠ 0}` -/

/-- **General-dimensional radial Abel equation.**  Under zero population
drift, on every positive radius where the common radial mean-shift ratio is
nonzero,

`Khat' = -((M' + n + 1) / M) * Khat`.

The displayed coefficient is written as `2 * ((M' + n + 1) / 2) / M` so it
can be passed directly to the common Abel propagation lemmas. -/
theorem hasDerivAt_radialRayKhatN_abel
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq))
    {r : ℝ} (hr : 0 < r) (hMr : radialRayMN n τ νp r ≠ 0) :
    HasDerivAt (radialRayKhatN n τ νp νq)
      (-((2 * ((radialRayMDerivN n (by omega) τ νp r + (n : ℝ) + 1) / 2)) /
          radialRayMN n τ νp r) *
        radialRayKhatN n τ νp νq r) r := by
  have hK := hasDerivAt_radialRayKhatN hn τ hτ νp νq hsp hsq hzero hr
  have hKM := radialRayKhatN_eq_M_mul_V hn τ hτ νp νq hsp hsq hzero hr
  have hval :
      -(τ * (radialRayMDerivN n (by omega) τ νp r + (n : ℝ) + 1)) *
          radialRayVN n (by omega) τ νp νq r =
        -((2 * ((radialRayMDerivN n (by omega) τ νp r + (n : ℝ) + 1) / 2)) /
            radialRayMN n τ νp r) *
          radialRayKhatN n τ νp νq r := by
    rw [hKM]
    field_simp [hMr]
  exact hK.congr_deriv hval

/-! ## The universal origin edge -/

private lemma zonalWeight_neg (n : ℕ) (u : ℝ) :
    zonalWeight n (-u) = zonalWeight n u := by
  unfold zonalWeight
  congr 1
  ring

/-- Every weighted zonal shell has zero axial displacement at the origin.
This is the general-`n` replacement for `shellD_zero_left`: the zonal weight
and the Laplace factor are even in the sphere coordinate, while `s * u` is
odd. -/
lemma shellDN_zero_left (n : ℕ) (τ s : ℝ) :
    shellDN n τ 0 s = 0 := by
  unfold shellDN
  have hdist : ∀ u : ℝ, shellDist 0 s u = |s| := by
    intro u
    unfold shellDist
    rw [show (0 : ℝ) ^ 2 + s ^ 2 - 2 * 0 * s * u = s ^ 2 by ring]
    exact Real.sqrt_sq_eq_abs s
  let f : ℝ → ℝ := fun u =>
    zonalWeight n u * (Real.exp (-(1 / τ) * |s|) * (s * u))
  have hfodd : ∀ u : ℝ, f (-u) = -f u := by
    intro u
    dsimp [f]
    rw [zonalWeight_neg]
    ring
  have hcong :
      (∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist 0 s u) * (s * u - 0))) =
        ∫ u in Ioc (-1 : ℝ) 1, f u := by
    refine setIntegral_congr_fun measurableSet_Ioc fun u _ => ?_
    rw [hdist]
    simp [f]
  rw [hcong, integral_Ioc_neg_one_one_eq_interval]
  have hreflect := intervalIntegral.integral_comp_neg (f := f)
      (a := (-1 : ℝ)) (b := 1)
  have hneg : (∫ u in (-1 : ℝ)..1, f (-u)) =
      -∫ u in (-1 : ℝ)..1, f u := by
    rw [intervalIntegral.integral_congr (fun u _ => hfodd u)]
    exact intervalIntegral.integral_neg
  rw [hneg] at hreflect
  norm_num at hreflect
  have : (∫ u in (-1 : ℝ)..1, f u) = 0 := by linarith
  rw [this, mul_zero]

/-- The general-dimensional radial displacement numerator vanishes at the
origin. -/
lemma radialRayDN_zero (n : ℕ) (τ : ℝ) (ν : Measure ℝ) :
    radialRayDN n τ ν 0 = 0 := by
  rw [radialRayDN]
  have hfun : (fun s => shellDN n τ 0 s) = fun _ => (0 : ℝ) :=
    funext fun s => shellDN_zero_left n τ s
  rw [hfun]
  simp

/-- The general-dimensional ray normalizer is globally continuous, including
at the origin. -/
lemma continuous_radialRayZN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    Continuous (radialRayZN n τ ν) := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hfe : radialRayZN n τ ν = fun r =>
      ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y
        ∂(radialMixtureN n ν) := by
    funext r
    rw [radialRayZN_eq_kernelNormalizer hn hτ ν r]
    rfl
  rw [hfe]
  refine continuous_of_dominated (bound := fun _ => (1 : ℝ))
    (fun r => ?_) (fun r => ?_) ?_ ?_
  · exact (continuous_laplaceKernel_radialRayProbeN
      (n := n) (by omega) τ r).aestronglyMeasurable
  · exact ae_of_all _ fun y => by
      rw [Real.norm_eq_abs,
        abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg
          (n := n) (by omega) τ r y)]
      exact laplaceKernel_radialRayProbeN_le_one
        (n := n) (by omega) τ hτ r y
  · exact integrable_const 1
  · exact ae_of_all _ fun y =>
      continuous_laplaceKernel_radialRayProbeN_left
        (n := n) (by omega) τ y

/-- The general-dimensional axial displacement profile is globally
continuous, including at the origin. -/
lemma continuous_radialRayDN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    Continuous (radialRayDN n τ ν) := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hfe : radialRayDN n τ ν = fun r =>
      ∫ y, laplaceKernel τ (radialRayProbeN n (by omega) r) y *
        (y ⟨0, by omega⟩ - r) ∂(radialMixtureN n ν) := by
    funext r
    exact radialRayDN_eq_integral_first hn τ hτ ν r
  rw [hfe]
  refine continuous_of_dominated (bound := fun _ => τ * Real.exp (-1))
    (fun r => ?_) (fun r => ?_) ?_ ?_
  · exact ((continuous_laplaceKernel_radialRayProbeN
      (n := n) (by omega) τ r).mul (by fun_prop)).aestronglyMeasurable
  · exact ae_of_all _ fun y => by
      rw [Real.norm_eq_abs]
      exact abs_laplaceKernel_mul_first_radialRayProbeN_le
        (n := n) (by omega) τ hτ r y
  · exact integrable_const _
  · exact ae_of_all _ fun y =>
      (continuous_laplaceKernel_radialRayProbeN_left
        (n := n) (by omega) τ y).mul (by fun_prop)

/-- The displacement profile grows at most linearly away from the origin.
The constant is dimension-free because `|Q| ≤ τ e⁻¹` and `Z ≤ 1`. -/
lemma abs_radialRayDN_le_linear
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (hsupp : ν (Iio 0) = 0) {t : ℝ} (ht : 0 < t) :
    |radialRayDN n τ ν t| ≤ (Real.exp (-1) + 1) * t := by
  have hC : (0 : ℝ) ≤ Real.exp (-1) + 1 := by positivity
  have hbound : ∀ x : ℝ, 0 < x →
      ‖(1 / τ) * radialRayQN n τ ν x - radialRayZN n τ ν x‖ ≤
        Real.exp (-1) + 1 := by
    intro x _
    rw [Real.norm_eq_abs]
    have h1 : |(1 / τ) * radialRayQN n τ ν x| ≤ Real.exp (-1) := by
      rw [abs_mul, abs_of_nonneg (by positivity : (0 : ℝ) ≤ 1 / τ)]
      have hQ := abs_radialRayQN_le hn τ hτ ν x
      calc
        (1 / τ) * |radialRayQN n τ ν x|
            ≤ (1 / τ) * (τ * Real.exp (-1)) :=
              mul_le_mul_of_nonneg_left hQ (by positivity)
        _ = Real.exp (-1) := by field_simp
    have h2 : |radialRayZN n τ ν x| ≤ 1 := by
      rw [abs_of_nonneg (radialRayZN_nonneg hn τ hτ ν x)]
      exact radialRayZN_le_one hn τ hτ ν x
    exact (abs_sub _ _).trans (add_le_add h1 h2)
  have hev : ∀ᶠ ε in nhdsWithin (0 : ℝ) (Ioi 0), |radialRayDN n τ ν t| ≤
      |radialRayDN n τ ν ε| + (Real.exp (-1) + 1) * t := by
    filter_upwards [Ioo_mem_nhdsGT ht] with ε hε
    have hMVT : ‖radialRayDN n τ ν t - radialRayDN n τ ν ε‖ ≤
        (Real.exp (-1) + 1) * ‖t - ε‖ := by
      refine Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
        (f' := fun x => (1 / τ) * radialRayQN n τ ν x - radialRayZN n τ ν x)
        (fun x hx => ?_) (fun x hx => hbound x (lt_of_lt_of_le hε.1 hx.1))
        (convex_Icc ε t) (left_mem_Icc.mpr hε.2.le) (right_mem_Icc.mpr hε.2.le)
      exact (hasDerivAt_radialRayDN_shell hn τ hτ ν hsupp
        (lt_of_lt_of_le hε.1 hx.1)).hasDerivWithinAt
    rw [Real.norm_eq_abs, Real.norm_eq_abs,
      abs_of_nonneg (by linarith [hε.2] : (0 : ℝ) ≤ t - ε)] at hMVT
    calc
      |radialRayDN n τ ν t|
          ≤ |radialRayDN n τ ν ε| +
              |radialRayDN n τ ν t - radialRayDN n τ ν ε| := by
            have h := abs_sub_abs_le_abs_sub
              (radialRayDN n τ ν t) (radialRayDN n τ ν ε)
            linarith [abs_nonneg (radialRayDN n τ ν ε)]
      _ ≤ |radialRayDN n τ ν ε| + (Real.exp (-1) + 1) * (t - ε) := by
            linarith
      _ ≤ |radialRayDN n τ ν ε| + (Real.exp (-1) + 1) * t := by
            nlinarith [hε.1]
  have htend : Tendsto
      (fun ε : ℝ => |radialRayDN n τ ν ε| + (Real.exp (-1) + 1) * t)
      (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds ((Real.exp (-1) + 1) * t)) := by
    have h0 : Tendsto (fun ε : ℝ => |radialRayDN n τ ν ε|)
        (nhdsWithin (0 : ℝ) (Ioi 0)) (nhds 0) := by
      have hc : ContinuousAt (fun ε : ℝ => |radialRayDN n τ ν ε|) 0 :=
        ((continuous_radialRayDN hn τ hτ ν).abs).continuousAt
      have hlim : Tendsto (fun ε : ℝ => |radialRayDN n τ ν ε|)
          (nhdsWithin (0 : ℝ) (Ioi 0))
          (nhds |radialRayDN n τ ν 0|) :=
        hc.tendsto.mono_left nhdsWithin_le_nhds
      simpa [radialRayDN_zero] using hlim
    simpa using h0.add (tendsto_const_nhds
      (f := nhdsWithin (0 : ℝ) (Ioi 0)) (x := (Real.exp (-1) + 1) * t))
  exact ge_of_tendsto htend hev

/-- On every compact initial ray, the mean-shift ratio has a positive linear
upper bound at the origin. -/
lemma radialRayMN_le_linear
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (hsupp : ν (Iio 0) = 0) {b : ℝ} (hb : 0 < b) :
    ∃ L : ℝ, 0 < L ∧ ∀ t : ℝ, 0 < t → t ≤ b →
      radialRayMN n τ ν t ≤ L * t := by
  obtain ⟨z₀, hz₀mem, hz₀min⟩ := isCompact_Icc.exists_isMinOn
    (Set.nonempty_Icc.mpr hb.le) ((continuous_radialRayZN hn τ hτ ν).continuousOn)
  have hZmin : 0 < radialRayZN n τ ν z₀ := radialRayZN_pos hn τ hτ ν z₀
  refine ⟨(Real.exp (-1) + 1) / radialRayZN n τ ν z₀, by positivity, ?_⟩
  intro t ht htb
  have hZt : radialRayZN n τ ν z₀ ≤ radialRayZN n τ ν t :=
    hz₀min (Set.mem_Icc.mpr ⟨ht.le, htb⟩)
  have hZtpos : 0 < radialRayZN n τ ν t := radialRayZN_pos hn τ hτ ν t
  have hD := abs_radialRayDN_le_linear hn τ hτ ν hsupp ht
  rw [radialRayMN, div_le_iff₀ hZtpos]
  calc
    radialRayDN n τ ν t ≤ |radialRayDN n τ ν t| := le_abs_self _
    _ ≤ (Real.exp (-1) + 1) * t := hD
    _ = ((Real.exp (-1) + 1) / radialRayZN n τ ν z₀) * t *
        radialRayZN n τ ν z₀ := by field_simp
    _ ≤ ((Real.exp (-1) + 1) / radialRayZN n τ ν z₀) * t *
        radialRayZN n τ ν t := by
      exact mul_le_mul_of_nonneg_left hZt (by positivity)

/-- The raw alignment determinant is bounded by `2τ`. -/
lemma abs_radialRayKN_le_two_tau
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (r : ℝ) :
    |radialRayKN n τ νp νq r| ≤ 2 * τ := by
  have hCp0 := radialRayCN_nonneg hn τ hτ νp r
  have hCq0 := radialRayCN_nonneg hn τ hτ νq r
  have hZp0 := radialRayZN_nonneg hn τ hτ νp r
  have hZq0 := radialRayZN_nonneg hn τ hτ νq r
  have hCp := radialRayCN_le hn τ hτ νp r
  have hCq := radialRayCN_le hn τ hτ νq r
  have hZp := radialRayZN_le_one hn τ hτ νp r
  have hZq := radialRayZN_le_one hn τ hτ νq r
  have hp : radialRayCN n τ νp r * radialRayZN n τ νq r ≤ τ := by
    calc
      radialRayCN n τ νp r * radialRayZN n τ νq r
          ≤ τ * 1 := mul_le_mul hCp hZq hZq0 hτ.le
      _ = τ := mul_one _
  have hq : radialRayCN n τ νq r * radialRayZN n τ νp r ≤ τ := by
    calc
      radialRayCN n τ νq r * radialRayZN n τ νp r
          ≤ τ * 1 := mul_le_mul hCq hZp hZp0 hτ.le
      _ = τ := mul_one _
  rw [radialRayKN]
  calc
    |radialRayCN n τ νp r * radialRayZN n τ νq r -
        radialRayCN n τ νq r * radialRayZN n τ νp r|
        ≤ |radialRayCN n τ νp r * radialRayZN n τ νq r| +
          |radialRayCN n τ νq r * radialRayZN n τ νp r| := abs_sub _ _
    _ = radialRayCN n τ νp r * radialRayZN n τ νq r +
          radialRayCN n τ νq r * radialRayZN n τ νp r := by
      rw [abs_of_nonneg (mul_nonneg hCp0 hZq0),
        abs_of_nonneg (mul_nonneg hCq0 hZp0)]
    _ ≤ 2 * τ := by linarith

/-- `Khat` is bounded as `r → 0⁺`.  In fact, on `(0,1)` its norm is at
most `2τ`; no moment assumption is involved at the origin edge. -/
lemma radialRayKhatN_bounded_near_zero
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq] :
    IsBoundedUnder (· ≤ ·) (nhdsWithin (0 : ℝ) (Ioi 0))
      (norm ∘ radialRayKhatN n τ νp νq) := by
  refine ⟨2 * τ, ?_⟩
  rw [Filter.eventually_map]
  filter_upwards [Ioo_mem_nhdsGT one_pos] with r hr
  have hK := abs_radialRayKN_le_two_tau hn τ hτ νp νq r
  have hrpow : r ^ (n - 1) ≤ 1 := by
    exact pow_le_one₀ hr.1.le hr.2.le
  rw [Function.comp_apply, Real.norm_eq_abs, radialRayKhatN, abs_mul,
    abs_of_nonneg (pow_nonneg hr.1.le (n - 1))]
  calc
    r ^ (n - 1) * |radialRayKN n τ νp νq r|
        ≤ 1 * (2 * τ) :=
      mul_le_mul hrpow hK (abs_nonneg _) (by positivity)
    _ = 2 * τ := one_mul _

/-! ## Propagation across sign components -/

section Trichotomy

variable {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (νp νq : Measure ℝ)
  [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]

/-- Right-interval Abel propagation from a left edge (either the origin or
an interior zero of `M`). -/
lemma radialRayKhatN_eq_zero_on_Ioo_of_leftEdge (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hslackp : RadialSlackN n (by omega) τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq))
    {a b r₂ L : ℝ} (ha : 0 ≤ a) (hab : a < b) (har₂ : a < r₂) (hr₂b : r₂ < b)
    (hL : 0 < L)
    (hmpos : ∀ t : ℝ, a < t → t < b → 0 < radialRayMN n τ νp t)
    (hlin : ∀ t : ℝ, a < t → t < r₂ → radialRayMN n τ νp t ≤ L * (t - a))
    (hKbdd : IsBoundedUnder (· ≤ ·) (nhdsWithin a (Ioi a))
      (norm ∘ radialRayKhatN n τ νp νq)) :
    ∀ x : ℝ, a < x → x < b → radialRayKhatN n τ νp νq x = 0 := by
  have hn0 : 0 < n := by omega
  let c : ℝ → ℝ := fun t =>
    (2 * ((radialRayMDerivN n hn0 τ νp t + (n : ℝ) + 1) / 2)) /
      radialRayMN n τ νp t
  have hccontAt : ∀ t ∈ Ioo a b, ContinuousAt c t := by
    intro t ht
    have ht0 : 0 < t := lt_of_le_of_lt ha ht.1
    have hMne : radialRayMN n τ νp t ≠ 0 := (hmpos t ht.1 ht.2).ne'
    exact ContinuousAt.div
      ((((continuousAt_radialRayMDerivN hn τ hτ νp hsp ht0).add
        continuousAt_const).add continuousAt_const).div_const 2 |>.const_mul 2)
      (continuousAt_radialRayMN hn τ hτ νp hsp ht0) hMne
  have hcOn : ContinuousOn c (Ioo a b) :=
    fun t ht => (hccontAt t ht).continuousWithinAt
  have hcint : ∀ u v : ℝ, u ∈ Ioo a b → v ∈ Ioo a b →
      IntervalIntegrable c volume u v := by
    intro u v hu hv
    exact ContinuousOn.intervalIntegrable
      (hcOn.mono (OrdConnected.uIcc_subset inferInstance hu hv))
  let A : ℝ → ℝ := fun z => ∫ s in r₂..z, c s
  have hAderiv : ∀ t ∈ Ioo a b, HasDerivAt A (c t) t := by
    intro t ht
    exact intervalIntegral.integral_hasDerivAt_right
      (hcint r₂ t ⟨har₂, hr₂b⟩ ht)
      (hcOn.stronglyMeasurableAtFilter isOpen_Ioo t ht)
      (hccontAt t ht)
  refine abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper
    (W := radialRayKhatN n τ νp νq) (A := A)
    (μDeriv := fun t => (radialRayMDerivN n hn0 τ νp t + (n : ℝ) + 1) / 2)
    (m := radialRayMN n τ νp)
    (a := a) (b := b) (r := r₂) (δ := 1 / 2) (L := L)
    hab har₂ hr₂b (by norm_num) hL ?_ ?_ ?_ ?_ hKbdd ?_ ?_ ?_
  · intro x y hx hxy hy
    have hsub : Icc x y ⊆ Ioo a b := fun t ht =>
      ⟨lt_of_lt_of_le hx ht.1, lt_of_le_of_lt ht.2 hy⟩
    refine ContinuousOn.mul ?_ (Real.continuous_exp.comp_continuousOn ?_)
    · intro t ht
      have ht0 : 0 < t := lt_of_le_of_lt ha (hsub ht).1
      exact ((hasDerivAt_radialRayKhatN hn τ hτ νp νq hsp hsq hzero
        ht0).continuousAt).continuousWithinAt
    · intro t ht
      exact ((hAderiv t (hsub ht)).continuousAt).continuousWithinAt
  · intro x y hx hxy hy t ht
    have ht' : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    have ht0 : 0 < t := lt_of_le_of_lt ha ht'.1
    have hMne : radialRayMN n τ νp t ≠ 0 := (hmpos t ht'.1 ht'.2).ne'
    simpa [c] using (hasDerivAt_radialRayKhatN_abel hn τ hτ νp νq
      hsp hsq hzero ht0 hMne).hasDerivWithinAt
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
    have hder := radialRayMDerivN_ge hn τ hτ νp hsp hslackp ht0
    linarith
  · intro t hat htr
    exact hmpos t hat (lt_trans htr hr₂b)
  · exact hlin

/-- Left-interval Abel propagation from a right edge at an interior zero of
`M`. -/
lemma radialRayKhatN_eq_zero_on_Ioo_of_rightEdge (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hslackp : RadialSlackN n (by omega) τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq))
    {a b l₂ L : ℝ} (ha : 0 < a) (hab : a < b) (hal₂ : a < l₂) (hl₂b : l₂ < b)
    (hL : 0 < L)
    (hmneg : ∀ t : ℝ, a < t → t < b → radialRayMN n τ νp t < 0)
    (hlin : ∀ t : ℝ, l₂ ≤ t → t < b →
      -(L * (b - t)) ≤ radialRayMN n τ νp t) :
    ∀ x : ℝ, a < x → x < b → radialRayKhatN n τ νp νq x = 0 := by
  have hn0 : 0 < n := by omega
  let c : ℝ → ℝ := fun t =>
    (2 * ((radialRayMDerivN n hn0 τ νp t + (n : ℝ) + 1) / 2)) /
      radialRayMN n τ νp t
  have hccontAt : ∀ t ∈ Ioo a b, ContinuousAt c t := by
    intro t ht
    have ht0 : 0 < t := lt_trans ha ht.1
    have hMne : radialRayMN n τ νp t ≠ 0 := (hmneg t ht.1 ht.2).ne
    exact ContinuousAt.div
      ((((continuousAt_radialRayMDerivN hn τ hτ νp hsp ht0).add
        continuousAt_const).add continuousAt_const).div_const 2 |>.const_mul 2)
      (continuousAt_radialRayMN hn τ hτ νp hsp ht0) hMne
  have hcOn : ContinuousOn c (Ioo a b) :=
    fun t ht => (hccontAt t ht).continuousWithinAt
  have hcint : ∀ u v : ℝ, u ∈ Ioo a b → v ∈ Ioo a b →
      IntervalIntegrable c volume u v := by
    intro u v hu hv
    exact ContinuousOn.intervalIntegrable
      (hcOn.mono (OrdConnected.uIcc_subset inferInstance hu hv))
  let A : ℝ → ℝ := fun z => ∫ s in l₂..z, c s
  have hAderiv : ∀ t ∈ Ioo a b, HasDerivAt A (c t) t := by
    intro t ht
    exact intervalIntegral.integral_hasDerivAt_right
      (hcint l₂ t ⟨hal₂, hl₂b⟩ ht)
      (hcOn.stronglyMeasurableAtFilter isOpen_Ioo t ht)
      (hccontAt t ht)
  have hb0 : 0 < b := lt_trans ha hab
  have hKbdd : IsBoundedUnder (· ≤ ·) (nhdsWithin b (Iio b))
      (norm ∘ radialRayKhatN n τ νp νq) :=
    (((hasDerivAt_radialRayKhatN hn τ hτ νp νq hsp hsq hzero
      hb0).continuousAt).norm.isBoundedUnder_le).mono nhdsWithin_le_nhds
  refine abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower
    (W := radialRayKhatN n τ νp νq) (A := A)
    (μDeriv := fun t => (radialRayMDerivN n hn0 τ νp t + (n : ℝ) + 1) / 2)
    (m := radialRayMN n τ νp)
    (a := a) (b := b) (l := l₂) (δ := 1 / 2) (L := L)
    hab hal₂ hl₂b (by norm_num) hL ?_ ?_ ?_ ?_ hKbdd ?_ ?_ ?_
  · intro x y hx hxy hy
    have hsub : Icc x y ⊆ Ioo a b := fun t ht =>
      ⟨lt_of_lt_of_le hx ht.1, lt_of_le_of_lt ht.2 hy⟩
    refine ContinuousOn.mul ?_ (Real.continuous_exp.comp_continuousOn ?_)
    · intro t ht
      have ht0 : 0 < t := lt_trans ha (hsub ht).1
      exact ((hasDerivAt_radialRayKhatN hn τ hτ νp νq hsp hsq hzero
        ht0).continuousAt).continuousWithinAt
    · intro t ht
      exact ((hAderiv t (hsub ht)).continuousAt).continuousWithinAt
  · intro x y hx hxy hy t ht
    have ht' : t ∈ Ioo a b := ⟨lt_of_lt_of_le hx ht.1, lt_trans ht.2 hy⟩
    have ht0 : 0 < t := lt_trans ha ht'.1
    have hMne : radialRayMN n τ νp t ≠ 0 := (hmneg t ht'.1 ht'.2).ne
    simpa [c] using (hasDerivAt_radialRayKhatN_abel hn τ hτ νp νq
      hsp hsq hzero ht0 hMne).hasDerivWithinAt
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
    have hder := radialRayMDerivN_ge hn τ hτ νp hsp hslackp ht0
    linarith
  · intro t hlt htb
    exact hmneg t (lt_of_lt_of_le hal₂ hlt) htb
  · exact hlin

/-- Outer-ray propagation, parametrized by the (later moment-theoretic) tail
decay of `Khat`.  Separating this endpoint fact keeps the Abel argument fully
independent of the chosen moment hypothesis. -/
lemma radialRayKhatN_eq_zero_on_ray_of_tendsto (hτ : 0 < τ)
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hslackp : RadialSlackN n (by omega) τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq))
    (htail : Tendsto (radialRayKhatN n τ νp νq) atTop (nhds 0))
    {a : ℝ} (ha : 0 < a)
    (hmneg : ∀ t : ℝ, a ≤ t → radialRayMN n τ νp t < 0) :
    ∀ x : ℝ, a ≤ x → radialRayKhatN n τ νp νq x = 0 := by
  have hn0 : 0 < n := by omega
  refine abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg
    (W := radialRayKhatN n τ νp νq)
    (muDeriv := fun t => (radialRayMDerivN n hn0 τ νp t + (n : ℝ) + 1) / 2)
    (m := radialRayMN n τ νp) (a := a) ?_ ?_ ?_ hmneg htail
  · intro x b hax hxb t ht
    have ht0 : 0 < t := lt_of_lt_of_le ha (le_trans hax ht.1)
    exact ((hasDerivAt_radialRayKhatN hn τ hτ νp νq hsp hsq hzero
      ht0).continuousAt).continuousWithinAt
  · intro x b hax hxb t ht
    have ht0 : 0 < t := lt_of_lt_of_le ha (le_trans hax ht.1)
    have hMne : radialRayMN n τ νp t ≠ 0 := (hmneg t (le_trans hax ht.1)).ne
    exact (hasDerivAt_radialRayKhatN_abel hn τ hτ νp νq hsp hsq hzero
      ht0 hMne).hasDerivWithinAt
  · intro t hat
    have ht0 : 0 < t := lt_of_lt_of_le ha hat
    have hder := radialRayMDerivN_ge hn τ hτ νp hsp hslackp ht0
    linarith

end Trichotomy

/-! ## The trichotomy driver -/

/-- If the weighted determinant has its required tail decay, it vanishes on
the whole open ray.  This theorem isolates the purely ODE/topological content
of G1 from the moment estimate used to prove the tail hypothesis. -/
theorem radialRayKhatN_eq_zero_of_tendsto
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hslackp : RadialSlackN n (by omega) τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq))
    (htail : Tendsto (radialRayKhatN n τ νp νq) atTop (nhds 0)) :
    ∀ x : ℝ, 0 < x → radialRayKhatN n τ νp νq x = 0 := by
  intro x₀ hx₀
  by_cases hmx₀ : radialRayMN n τ νp x₀ = 0
  · rw [radialRayKhatN_eq_M_mul_V hn τ hτ νp νq hsp hsq hzero hx₀, hmx₀]
    ring
  rcases lt_or_gt_of_ne hmx₀ with hneg | hpos
  · -- A negative component either ends at its first zero or reaches infinity.
    by_cases hS : ∃ t : ℝ, x₀ ≤ t ∧ radialRayMN n τ νp t = 0
    · obtain ⟨t₀, ht₀x, ht₀0⟩ := hS
      let S : Set ℝ := {t : ℝ | x₀ ≤ t ∧ radialRayMN n τ νp t = 0}
      have hSne : S.Nonempty := ⟨t₀, ht₀x, ht₀0⟩
      have hSbdd : BddBelow S := ⟨x₀, fun t ht => ht.1⟩
      have hβx : x₀ ≤ sInf S := le_csInf hSne fun t ht => ht.1
      have hβ0 : 0 < sInf S := lt_of_lt_of_le hx₀ hβx
      have hmβ : radialRayMN n τ νp (sInf S) = 0 := by
        have hβcl := csInf_mem_closure hSne hSbdd
        have himg := (continuousAt_radialRayMN hn τ hτ νp hsp hβ0).continuousWithinAt
          |>.mem_closure_image hβcl
        have hsub : radialRayMN n τ νp '' S ⊆ {0} := by
          rintro y ⟨t, ht, rfl⟩
          exact ht.2
        have h0 := closure_mono hsub himg
        rwa [closure_singleton, mem_singleton_iff] at h0
      have hβgt : x₀ < sInf S := by
        rcases eq_or_lt_of_le hβx with heq | hlt
        · exfalso
          apply hneg.ne
          rw [heq]
          exact hmβ
        · exact hlt
      have hmid : ∀ t : ℝ, x₀ ≤ t → t < sInf S → radialRayMN n τ νp t < 0 := by
        intro t hxt htβ
        rcases lt_trichotomy (radialRayMN n τ νp t) 0 with hlt | heq | hgt
        · exact hlt
        · exact absurd (csInf_le hSbdd ⟨hxt, heq⟩) (not_le.mpr htβ)
        · exfalso
          have hcont : ContinuousOn (radialRayMN n τ νp) (Icc x₀ t) := fun z hz =>
            (continuousAt_radialRayMN hn τ hτ νp hsp
              (lt_of_lt_of_le hx₀ hz.1)).continuousWithinAt
          have h0mem : (0 : ℝ) ∈ Icc (radialRayMN n τ νp x₀)
              (radialRayMN n τ νp t) := ⟨hneg.le, hgt.le⟩
          obtain ⟨z, hzIcc, hz0⟩ := intermediate_value_Icc hxt hcont h0mem
          have hzβ := csInf_le hSbdd ⟨hzIcc.1, hz0⟩
          linarith [hzIcc.2]
      have hnb : (radialRayMN n τ νp) ⁻¹' (Iio 0) ∈ nhds x₀ :=
        (continuousAt_radialRayMN hn τ hτ νp hsp hx₀) (isOpen_Iio.mem_nhds hneg)
      obtain ⟨ε, hε, hball⟩ := Metric.mem_nhds_iff.mp hnb
      let l₁ := max (x₀ - ε / 2) (x₀ / 2)
      have hl₁0 : 0 < l₁ := by
        change 0 < max (x₀ - ε / 2) (x₀ / 2)
        exact lt_max_of_lt_right (by linarith)
      have hl₁x : l₁ < x₀ := by
        change max (x₀ - ε / 2) (x₀ / 2) < x₀
        exact max_lt (by linarith) (by linarith)
      have hl₁neg : ∀ t : ℝ, l₁ < t → t ≤ x₀ → radialRayMN n τ νp t < 0 := by
        intro t hlt hle
        apply hball
        rw [Metric.mem_ball, Real.dist_eq, abs_sub_lt_iff]
        have hleft : x₀ - ε / 2 < t := lt_of_le_of_lt (le_max_left _ _) hlt
        exact ⟨by linarith, by linarith⟩
      have hmneg_ext : ∀ t : ℝ, l₁ < t → t < sInf S →
          radialRayMN n τ νp t < 0 := by
        intro t hlt htβ
        rcases le_or_gt t x₀ with hle | hgt
        · exact hl₁neg t hlt hle
        · exact hmid t hgt.le htβ
      obtain ⟨l₂, _, L, hl₁l₂, hl₂β, _, _, hL, hlin, _⟩ :=
        exists_Ioo_linear_bound_of_hasDerivAt_zero (f := radialRayMN n τ νp)
          (a := sInf S) (lower := l₁) (upper := sInf S + 1)
          (lt_of_lt_of_le hl₁x hβx) (lt_add_one _)
          (hasDerivAt_radialRayMN hn τ hτ νp hsp hβ0) hmβ
      exact radialRayKhatN_eq_zero_on_Ioo_of_rightEdge hn τ νp νq hτ hsp hsq
        hslackp hzero hl₁0 (lt_of_lt_of_le hl₁x hβx) hl₁l₂ hl₂β hL
        hmneg_ext hlin x₀ hl₁x hβgt
    · have hmray : ∀ t : ℝ, x₀ ≤ t → radialRayMN n τ νp t < 0 := by
        intro t hxt
        rcases lt_trichotomy (radialRayMN n τ νp t) 0 with hlt | heq | hgt
        · exact hlt
        · exact absurd ⟨t, hxt, heq⟩ hS
        · exfalso
          have hcont : ContinuousOn (radialRayMN n τ νp) (Icc x₀ t) := fun z hz =>
            (continuousAt_radialRayMN hn τ hτ νp hsp
              (lt_of_lt_of_le hx₀ hz.1)).continuousWithinAt
          have h0mem : (0 : ℝ) ∈ Icc (radialRayMN n τ νp x₀)
              (radialRayMN n τ νp t) := ⟨hneg.le, hgt.le⟩
          obtain ⟨z, hzIcc, hz0⟩ := intermediate_value_Icc hxt hcont h0mem
          exact hS ⟨z, hzIcc.1, hz0⟩
      exact radialRayKhatN_eq_zero_on_ray_of_tendsto hn τ νp νq hτ hsp hsq
        hslackp hzero htail hx₀ hmray x₀ le_rfl
  · -- A positive component either starts at its last interior zero or at 0.
    have hnb : (radialRayMN n τ νp) ⁻¹' (Ioi 0) ∈ nhds x₀ :=
      (continuousAt_radialRayMN hn τ hτ νp hsp hx₀) (isOpen_Ioi.mem_nhds hpos)
    obtain ⟨ε, hε, hball⟩ := Metric.mem_nhds_iff.mp hnb
    have hxr₁ : x₀ < x₀ + ε / 2 := by linarith
    have hposr : ∀ t : ℝ, x₀ ≤ t → t < x₀ + ε / 2 →
        0 < radialRayMN n τ νp t := by
      intro t hxt htr
      apply hball
      rw [Metric.mem_ball, Real.dist_eq, abs_sub_lt_iff]
      exact ⟨by linarith, by linarith⟩
    by_cases hS : ∃ t : ℝ, 0 < t ∧ t ≤ x₀ ∧ radialRayMN n τ νp t = 0
    · obtain ⟨t₀, ht₀0, ht₀x, ht₀z⟩ := hS
      let S : Set ℝ := {t : ℝ | 0 < t ∧ t ≤ x₀ ∧ radialRayMN n τ νp t = 0}
      have hSne : S.Nonempty := ⟨t₀, ht₀0, ht₀x, ht₀z⟩
      have hSbdd : BddAbove S := ⟨x₀, fun t ht => ht.2.1⟩
      have hαx : sSup S ≤ x₀ := csSup_le hSne fun t ht => ht.2.1
      have hα0 : 0 < sSup S := lt_of_lt_of_le ht₀0 (le_csSup hSbdd ⟨ht₀0, ht₀x, ht₀z⟩)
      have hmα : radialRayMN n τ νp (sSup S) = 0 := by
        have hαcl := csSup_mem_closure hSne hSbdd
        have himg := (continuousAt_radialRayMN hn τ hτ νp hsp hα0).continuousWithinAt
          |>.mem_closure_image hαcl
        have hsub : radialRayMN n τ νp '' S ⊆ {0} := by
          rintro y ⟨t, ht, rfl⟩
          exact ht.2.2
        have h0 := closure_mono hsub himg
        rwa [closure_singleton, mem_singleton_iff] at h0
      have hαlt : sSup S < x₀ := by
        rcases eq_or_lt_of_le hαx with heq | hlt
        · exfalso
          apply hpos.ne'
          rw [← heq]
          exact hmα
        · exact hlt
      have hmid : ∀ t : ℝ, sSup S < t → t ≤ x₀ →
          0 < radialRayMN n τ νp t := by
        intro t hαt htx
        have ht0 : 0 < t := lt_trans hα0 hαt
        rcases lt_trichotomy (radialRayMN n τ νp t) 0 with hlt | heq | hgt
        · exfalso
          have hcont : ContinuousOn (radialRayMN n τ νp) (Icc t x₀) := fun z hz =>
            (continuousAt_radialRayMN hn τ hτ νp hsp
              (lt_of_lt_of_le ht0 hz.1)).continuousWithinAt
          have h0mem : (0 : ℝ) ∈ Icc (radialRayMN n τ νp t)
              (radialRayMN n τ νp x₀) := ⟨hlt.le, hpos.le⟩
          obtain ⟨z, hzIcc, hz0⟩ := intermediate_value_Icc htx hcont h0mem
          have hzα := le_csSup hSbdd ⟨lt_of_lt_of_le ht0 hzIcc.1, hzIcc.2, hz0⟩
          linarith [hzIcc.1]
        · exact absurd (le_csSup hSbdd ⟨ht0, htx, heq⟩) (not_le.mpr hαt)
        · exact hgt
      have hmpos_ext : ∀ t : ℝ, sSup S < t → t < x₀ + ε / 2 →
          0 < radialRayMN n τ νp t := by
        intro t hαt htr
        rcases le_or_gt t x₀ with hle | hgt
        · exact hmid t hαt hle
        · exact hposr t hgt.le htr
      obtain ⟨_, r₂, L, _, _, hαr₂, hr₂r₁, hL, _, hlin'⟩ :=
        exists_Ioo_linear_bound_of_hasDerivAt_zero (f := radialRayMN n τ νp)
          (a := sSup S) (lower := 0) (upper := x₀ + ε / 2)
          hα0 (lt_trans hαlt hxr₁)
          (hasDerivAt_radialRayMN hn τ hτ νp hsp hα0) hmα
      have hKbdd : IsBoundedUnder (· ≤ ·) (nhdsWithin (sSup S) (Ioi (sSup S)))
          (norm ∘ radialRayKhatN n τ νp νq) :=
        (((hasDerivAt_radialRayKhatN hn τ hτ νp νq hsp hsq hzero
          hα0).continuousAt).norm.isBoundedUnder_le).mono nhdsWithin_le_nhds
      exact radialRayKhatN_eq_zero_on_Ioo_of_leftEdge hn τ νp νq hτ hsp hsq
        hslackp hzero hα0.le (lt_trans hαlt hxr₁) hαr₂ hr₂r₁ hL hmpos_ext
        (fun t h1 h2 => hlin' t h1 h2.le) hKbdd x₀ hαlt hxr₁
    · have hmray : ∀ t : ℝ, 0 < t → t ≤ x₀ → 0 < radialRayMN n τ νp t := by
        intro t ht0 htx
        rcases lt_trichotomy (radialRayMN n τ νp t) 0 with hlt | heq | hgt
        · exfalso
          have hcont : ContinuousOn (radialRayMN n τ νp) (Icc t x₀) := fun z hz =>
            (continuousAt_radialRayMN hn τ hτ νp hsp
              (lt_of_lt_of_le ht0 hz.1)).continuousWithinAt
          have h0mem : (0 : ℝ) ∈ Icc (radialRayMN n τ νp t)
              (radialRayMN n τ νp x₀) := ⟨hlt.le, hpos.le⟩
          obtain ⟨z, hzIcc, hz0⟩ := intermediate_value_Icc htx hcont h0mem
          exact hS ⟨z, lt_of_lt_of_le ht0 hzIcc.1, hzIcc.2, hz0⟩
        · exact absurd ⟨t, ht0, htx, heq⟩ hS
        · exact hgt
      have hmpos_ext : ∀ t : ℝ, 0 < t → t < x₀ + ε / 2 →
          0 < radialRayMN n τ νp t := by
        intro t ht0 htr
        rcases le_or_gt t x₀ with hle | hgt
        · exact hmray t ht0 hle
        · exact hposr t hgt.le htr
      obtain ⟨L, hL, hlin⟩ := radialRayMN_le_linear hn τ hτ νp hsp hx₀
      refine radialRayKhatN_eq_zero_on_Ioo_of_leftEdge hn τ νp νq hτ hsp hsq
        hslackp hzero (le_refl 0) (lt_trans hx₀ hxr₁) hx₀ hxr₁ hL
        hmpos_ext ?_ (radialRayKhatN_bounded_near_zero hn τ hτ νp νq)
        x₀ hx₀ hxr₁
      intro t ht0 htx
      have h := hlin t ht0 htx.le
      rwa [sub_zero]

/-! ## A moment-based tail provider -/

/-- A natural-power moment forces the correspondingly scaled upper tail to
vanish.  This is a direct dominated-convergence/Markov-tail argument. -/
lemma tendsto_pow_mul_measureReal_Ici_atTop
    (k : ℕ) (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (hmom : Integrable (fun s : ℝ => s ^ k) ν) :
    Tendsto (fun r : ℝ => r ^ k * ν.real (Ici r)) atTop (nhds 0) := by
  have hupper : Tendsto (fun r : ℝ => ∫ s in Ici r, s ^ k ∂ν) atTop (nhds 0) := by
    have h := tendsto_integral_filter_of_dominated_convergence (μ := ν)
      (l := atTop)
      (F := fun (r : ℝ) (s : ℝ) => Set.indicator (Ici r) (fun z => z ^ k) s)
      (f := fun _ => 0) (bound := fun s => |s ^ k|)
      (Filter.Eventually.of_forall fun _ =>
        ((measurable_id.pow_const k).indicator measurableSet_Ici).aestronglyMeasurable)
      (Filter.Eventually.of_forall fun _ => ae_of_all _ fun s => by
        simpa [Real.norm_eq_abs] using
          norm_indicator_le_norm_self (f := fun z : ℝ => z ^ k) s)
      hmom.abs
      (ae_of_all _ fun s => by
        have hev : ∀ᶠ r : ℝ in atTop,
            Set.indicator (Ici r) (fun z : ℝ => z ^ k) s = 0 := by
          filter_upwards [eventually_gt_atTop s] with r hr
          exact Set.indicator_of_notMem (by simpa using not_le.mpr hr) _
        exact tendsto_const_nhds.congr' (hev.mono fun r hr => hr.symm))
    simp only [integral_zero] at h
    refine h.congr fun r => ?_
    exact integral_indicator measurableSet_Ici
  have hnn : ∀ᶠ r : ℝ in atTop, 0 ≤ r ^ k * ν.real (Ici r) := by
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with r hr
    exact mul_nonneg (pow_nonneg hr k) ENNReal.toReal_nonneg
  have hle : ∀ᶠ r : ℝ in atTop,
      r ^ k * ν.real (Ici r) ≤ ∫ s in Ici r, s ^ k ∂ν := by
    filter_upwards [eventually_ge_atTop (0 : ℝ)] with r hr
    have hmono : (∫ _ in Ici r, r ^ k ∂ν) ≤ ∫ s in Ici r, s ^ k ∂ν :=
      setIntegral_mono_on ((integrable_const (r ^ k)).integrableOn)
        hmom.integrableOn measurableSet_Ici (fun s hs =>
          pow_le_pow_left₀ hr hs k)
    calc
      r ^ k * ν.real (Ici r) = ν.real (Ici r) • r ^ k := by
        rw [smul_eq_mul]
        ring
      _ = ∫ _ in Ici r, r ^ k ∂ν := (setIntegral_const (r ^ k)).symm
      _ ≤ ∫ s in Ici r, s ^ k ∂ν := hmono
  exact squeeze_zero' hnn hle hupper

private lemma shellZN_nonneg
    {n : ℕ} (hn : 3 ≤ n) (τ r s : ℝ) : 0 ≤ shellZN n τ r s := by
  rw [shellZN]
  exact mul_nonneg (inv_nonneg.mpr (zonalMass_pos hn).le)
    (setIntegral_nonneg measurableSet_Ioc fun u hu => mul_nonneg
      (zonalWeight_nonneg (sq_le_one_of_mem_Ioc hu)) (Real.exp_pos _).le)

private lemma abs_sub_le_shellDistN {r s u : ℝ}
    (hr : 0 ≤ r) (hs : 0 ≤ s) (hu : u ≤ 1) :
    |r - s| ≤ shellDist r s u := by
  rw [shellDist, ← Real.sqrt_sq_eq_abs]
  apply Real.sqrt_le_sqrt
  nlinarith [mul_nonneg hr hs]

/-- A weighted general-dimensional shell normalizer is bounded by the radial
gap exponential. -/
lemma shellZN_le_exp
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    {r s : ℝ} (hr : 0 ≤ r) (hs : 0 ≤ s) :
    shellZN n τ r s ≤ Real.exp (-(1 / τ) * |r - s|) := by
  rw [shellZN]
  have hbound : ∀ u ∈ Ioc (-1 : ℝ) 1,
      zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u) ≤
        zonalWeight n u * Real.exp (-(1 / τ) * |r - s|) := by
    intro u hu
    refine mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr ?_)
      (zonalWeight_nonneg (sq_le_one_of_mem_Ioc hu))
    have hgap := abs_sub_le_shellDistN hr hs hu.2
    have hrec : (0 : ℝ) ≤ 1 / τ := by positivity
    nlinarith
  have hleft : IntegrableOn
      (fun u : ℝ => zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u))
      (Ioc (-1 : ℝ) 1) :=
    (((continuous_zonalWeight hn).mul
      (Real.continuous_exp.comp
        ((continuous_shellDist_u r s).const_mul (-(1 / τ))))).integrableOn_Icc).mono_set
      Ioc_subset_Icc_self
  have hright : IntegrableOn
      (fun u : ℝ => zonalWeight n u * Real.exp (-(1 / τ) * |r - s|))
      (Ioc (-1 : ℝ) 1) :=
    ((integrableOn_zonalWeight hn).mul_const _)
  have hmono := setIntegral_mono_on hleft hright measurableSet_Ioc hbound
  calc
    (zonalMass n)⁻¹ *
        ∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)
      ≤ (zonalMass n)⁻¹ *
        ∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * Real.exp (-(1 / τ) * |r - s|) :=
        mul_le_mul_of_nonneg_left hmono (inv_nonneg.mpr (zonalMass_pos hn).le)
    _ = Real.exp (-(1 / τ) * |r - s|) := by
      have hfun : (fun u : ℝ =>
          zonalWeight n u * Real.exp (-(1 / τ) * |r - s|)) =
          fun u => Real.exp (-(1 / τ) * |r - s|) * zonalWeight n u := by
        funext u
        ring
      rw [hfun, integral_const_mul, zonalMass]
      field_simp [zonalMass_ne_zero hn]
      exact div_self (zonalMass_ne_zero hn)

private lemma prob_measureReal_le_one_N
    {ν : Measure ℝ} [IsProbabilityMeasure ν] (s : Set ℝ) : ν.real s ≤ 1 := by
  have h := ENNReal.toReal_mono ENNReal.one_ne_top (show ν s ≤ 1 from prob_le_one)
  exact h

/-- General-`n` normalizer split: near shells contribute only an exponential,
and far shells are controlled by the radial tail probability. -/
lemma radialRayZN_le_split
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    {r : ℝ} (hr : 0 ≤ r) :
    radialRayZN n τ ν r ≤
      Real.exp (-(1 / τ) * (r / 2)) + ν.real (Ici (r / 2)) := by
  have hint := integrable_shellZN hn hτ ν (r := r)
  rw [radialRayZN, ← integral_add_compl (measurableSet_Ici (a := r / 2)) hint]
  have htail : (∫ s in Ici (r / 2), shellZN n τ r s ∂ν) ≤
      ν.real (Ici (r / 2)) := by
    calc
      (∫ s in Ici (r / 2), shellZN n τ r s ∂ν)
          ≤ ∫ _ in Ici (r / 2), (1 : ℝ) ∂ν :=
        setIntegral_mono_on hint.integrableOn ((integrable_const 1).integrableOn)
          measurableSet_Ici (fun s _ => by
            exact (le_abs_self _).trans (abs_shellZN_le_one hn hτ))
      _ = ν.real (Ici (r / 2)) := by
        rw [setIntegral_const, smul_eq_mul, mul_one]
  have hnear : (∫ s in (Ici (r / 2))ᶜ, shellZN n τ r s ∂ν) ≤
      Real.exp (-(1 / τ) * (r / 2)) := by
    have hae : ∀ᵐ s ∂(ν.restrict (Ici (r / 2))ᶜ),
        shellZN n τ r s ≤ Real.exp (-(1 / τ) * (r / 2)) := by
      filter_upwards [ae_restrict_of_ae (radial_ae_nonneg_N hsupp),
        ae_restrict_mem (measurableSet_Ici (a := r / 2)).compl] with s hs hmem
      have hlt : s < r / 2 := by
        simpa [mem_compl_iff, mem_Ici, not_le] using hmem
      refine (shellZN_le_exp hn τ hτ hr hs).trans ?_
      apply Real.exp_le_exp.mpr
      have habs : r / 2 ≤ |r - s| := by
        rw [abs_of_nonneg (by linarith : (0 : ℝ) ≤ r - s)]
        linarith
      have hrec : (0 : ℝ) ≤ 1 / τ := by positivity
      nlinarith
    calc
      (∫ s in (Ici (r / 2))ᶜ, shellZN n τ r s ∂ν)
          ≤ ∫ _ in (Ici (r / 2))ᶜ,
              Real.exp (-(1 / τ) * (r / 2)) ∂ν :=
        setIntegral_mono_ae_restrict hint.integrableOn
          ((integrable_const _).integrableOn) hae
      _ = ν.real (Ici (r / 2))ᶜ * Real.exp (-(1 / τ) * (r / 2)) := by
        rw [setIntegral_const, smul_eq_mul]
      _ ≤ 1 * Real.exp (-(1 / τ) * (r / 2)) :=
        mul_le_mul_of_nonneg_right (prob_measureReal_le_one_N _)
          (Real.exp_pos _).le
      _ = Real.exp (-(1 / τ) * (r / 2)) := one_mul _
  linarith

private lemma tendsto_nat_pow_mul_exp_neg_mul_N (k : ℕ) {c : ℝ} (hc : 0 < c) :
    Tendsto (fun r : ℝ => r ^ k * Real.exp (-c * r)) atTop (nhds 0) := by
  have h := tendsto_rpow_mul_exp_neg_mul_atTop_nhds_zero (k : ℝ) c hc
  refine h.congr' ?_
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with r hr
  rw [Real.rpow_natCast]

private lemma tendsto_pow_mul_half_tail
    (k : ℕ) (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (hmom : Integrable (fun s : ℝ => s ^ k) ν) :
    Tendsto (fun r : ℝ => r ^ k * ν.real (Ici (r / 2))) atTop (nhds 0) := by
  have hhalf : Tendsto (fun r : ℝ => r / 2) atTop atTop :=
    tendsto_atTop_atTop.mpr fun b => ⟨2 * b, fun a ha => by linarith⟩
  have h := (tendsto_pow_mul_measureReal_Ici_atTop k ν hmom).comp hhalf
  have hscaled := h.const_mul ((2 : ℝ) ^ k)
  convert hscaled using 1
  · funext r
    simp only [Function.comp_apply]
    rw [← mul_assoc, ← mul_pow]
    rw [show (2 : ℝ) * (r / 2) = r by ring]
  · simp

/-- A natural `(n-1)`-moment is a simple sufficient condition for scaled
normalizer decay.  The sharper symmetric half-moment condition can replace
this provider later without changing any propagation theorem. -/
lemma tendsto_pow_radialRayZN_atTop
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0)
    (hmom : Integrable (fun s : ℝ => s ^ (n - 1)) ν) :
    Tendsto (fun r : ℝ => r ^ (n - 1) * radialRayZN n τ ν r)
      atTop (nhds 0) := by
  have hc : 0 < 1 / (2 * τ) := by positivity
  have hexp : Tendsto
      (fun r : ℝ => r ^ (n - 1) * Real.exp (-(1 / (2 * τ)) * r))
      atTop (nhds 0) := tendsto_nat_pow_mul_exp_neg_mul_N (n - 1) hc
  have htail := tendsto_pow_mul_half_tail (n - 1) ν hmom
  have hupper : Tendsto (fun r : ℝ =>
      r ^ (n - 1) *
        (Real.exp (-(1 / τ) * (r / 2)) + ν.real (Ici (r / 2))))
      atTop (nhds 0) := by
    have hadd := hexp.add htail
    convert hadd using 1
    · funext r
      have heq : -(1 / τ) * (r / 2) = -(1 / (2 * τ)) * r := by
        field_simp [hτ.ne']
      rw [mul_add, heq]
    · simp
  refine squeeze_zero' ?_ ?_ hupper
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with r hr
    exact mul_nonneg (pow_nonneg hr _) (radialRayZN_nonneg hn τ hτ ν r)
  · filter_upwards [eventually_ge_atTop (0 : ℝ)] with r hr
    exact mul_le_mul_of_nonneg_left (radialRayZN_le_split hn τ hτ ν hsupp hr)
      (pow_nonneg hr _)

private lemma abs_radialRayKN_le_tau_mul_Zadd
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (r : ℝ) :
    |radialRayKN n τ νp νq r| ≤
      τ * (radialRayZN n τ νp r + radialRayZN n τ νq r) := by
  have hCp0 := radialRayCN_nonneg hn τ hτ νp r
  have hCq0 := radialRayCN_nonneg hn τ hτ νq r
  have hZp0 := radialRayZN_nonneg hn τ hτ νp r
  have hZq0 := radialRayZN_nonneg hn τ hτ νq r
  have hCp := radialRayCN_le hn τ hτ νp r
  have hCq := radialRayCN_le hn τ hτ νq r
  rw [radialRayKN]
  calc
    |radialRayCN n τ νp r * radialRayZN n τ νq r -
        radialRayCN n τ νq r * radialRayZN n τ νp r|
      ≤ |radialRayCN n τ νp r * radialRayZN n τ νq r| +
        |radialRayCN n τ νq r * radialRayZN n τ νp r| := abs_sub _ _
    _ = radialRayCN n τ νp r * radialRayZN n τ νq r +
        radialRayCN n τ νq r * radialRayZN n τ νp r := by
      rw [abs_of_nonneg (mul_nonneg hCp0 hZq0),
        abs_of_nonneg (mul_nonneg hCq0 hZp0)]
    _ ≤ τ * radialRayZN n τ νq r +
        τ * radialRayZN n τ νp r :=
      add_le_add (mul_le_mul_of_nonneg_right hCp hZq0)
        (mul_le_mul_of_nonneg_right hCq hZp0)
    _ = τ * (radialRayZN n τ νp r + radialRayZN n τ νq r) := by ring

/-- `Khat → 0` under finite natural `(n-1)` moments of both radial laws. -/
theorem tendsto_radialRayKhatN_atTop
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable (fun s : ℝ => s ^ (n - 1)) νp)
    (hmq : Integrable (fun s : ℝ => s ^ (n - 1)) νq) :
    Tendsto (radialRayKhatN n τ νp νq) atTop (nhds 0) := by
  have hp := tendsto_pow_radialRayZN_atTop hn τ hτ νp hsp hmp
  have hq := tendsto_pow_radialRayZN_atTop hn τ hτ νq hsq hmq
  have hupper : Tendsto (fun r : ℝ =>
      τ * (r ^ (n - 1) * radialRayZN n τ νp r +
        r ^ (n - 1) * radialRayZN n τ νq r)) atTop (nhds 0) := by
    simpa using (hp.add hq).const_mul τ
  refine squeeze_zero_norm' ?_ hupper
  filter_upwards [eventually_ge_atTop (0 : ℝ)] with r hr
  rw [Real.norm_eq_abs, radialRayKhatN, abs_mul,
    abs_of_nonneg (pow_nonneg hr (n - 1))]
  calc
    r ^ (n - 1) * |radialRayKN n τ νp νq r|
      ≤ r ^ (n - 1) *
          (τ * (radialRayZN n τ νp r + radialRayZN n τ νq r)) :=
        mul_le_mul_of_nonneg_left (abs_radialRayKN_le_tau_mul_Zadd hn τ hτ νp νq r)
          (pow_nonneg hr _)
    _ = τ * (r ^ (n - 1) * radialRayZN n τ νp r +
          r ^ (n - 1) * radialRayZN n τ νq r) := by ring

/-- **General-`n` determinant propagation with an explicit moment provider.**
This closes the ODE part of G1 under `RadialSlackN` and finite natural
`(n-1)` moments. -/
theorem radialRayKhatN_eq_zero
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable (fun s : ℝ => s ^ (n - 1)) νp)
    (hmq : Integrable (fun s : ℝ => s ^ (n - 1)) νq)
    (hslackp : RadialSlackN n (by omega) τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq)) :
    ∀ r : ℝ, 0 < r → radialRayKhatN n τ νp νq r = 0 :=
  radialRayKhatN_eq_zero_of_tendsto hn τ hτ νp νq hsp hsq hslackp hzero
    (tendsto_radialRayKhatN_atTop hn τ hτ νp νq hsp hsq hmp hmq)

/-! ## From `Khat = 0` to ray proportionality -/

lemma continuousAt_radialRayVN
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    {r : ℝ} (hr : 0 < r) :
    ContinuousAt (radialRayVN n (by omega) τ νp νq) r := by
  have hn0 : 0 < n := by omega
  have h1 := continuousAt_radialRayZdN hn τ hτ νp hsp hr
  have h2 := continuousAt_radialRayZdN hn τ hτ νq hsq hr
  have h3 := (hasDerivAt_radialRayZN hn τ hτ νp hsp hr).continuousAt
  have h4 := (hasDerivAt_radialRayZN hn τ hτ νq hsq hr).continuousAt
  change ContinuousAt (fun x => x ^ (n - 1) *
    ((1 / τ) * (radialRayZdN n hn0 τ νp x * radialRayZN n τ νq x -
      radialRayZdN n hn0 τ νq x * radialRayZN n τ νp x))) r
  exact ((continuous_pow (n - 1)).continuousAt).mul
    (((h1.mul h4).sub (h2.mul h3)).const_mul _)

/-- The weighted Wronskian `V` vanishes on the open ray. -/
theorem radialRayVN_eq_zero
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable (fun s : ℝ => s ^ (n - 1)) νp)
    (hmq : Integrable (fun s : ℝ => s ^ (n - 1)) νq)
    (hslackp : RadialSlackN n (by omega) τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq)) :
    ∀ r : ℝ, 0 < r → radialRayVN n (by omega) τ νp νq r = 0 := by
  have hn0 : 0 < n := by omega
  have hK0 := radialRayKhatN_eq_zero hn τ hτ νp νq hsp hsq hmp hmq hslackp hzero
  intro r hr
  by_cases hm : radialRayMN n τ νp r = 0
  · by_cases hev : ∀ᶠ t in nhds r, radialRayMN n τ νp t = 0
    · have hMD0 : radialRayMDerivN n hn0 τ νp r = 0 := by
        have h1 := hasDerivAt_radialRayMN hn τ hτ νp hsp hr
        have heq : radialRayMN n τ νp =ᶠ[nhds r] fun _ => (0 : ℝ) := hev
        exact ((h1.congr_of_eventuallyEq heq.symm).unique (hasDerivAt_const r 0))
      have hK'0 :
          -(τ * (radialRayMDerivN n hn0 τ νp r + (n : ℝ) + 1)) *
            radialRayVN n hn0 τ νp νq r = 0 := by
        have h1 := hasDerivAt_radialRayKhatN hn τ hτ νp νq hsp hsq hzero hr
        have heq : radialRayKhatN n τ νp νq =ᶠ[nhds r] fun _ => (0 : ℝ) := by
          filter_upwards [Ioi_mem_nhds hr] with t ht
          exact hK0 t ht
        exact ((h1.congr_of_eventuallyEq heq.symm).unique (hasDerivAt_const r 0))
      rw [hMD0] at hK'0
      have hne : -(τ * ((0 : ℝ) + (n : ℝ) + 1)) ≠ 0 := by
        have hnpos : (0 : ℝ) < (n : ℝ) := by exact_mod_cast hn0
        apply neg_ne_zero.mpr
        exact mul_ne_zero hτ.ne' (by linarith)
      exact (mul_eq_zero.mp hK'0).resolve_left hne
    · have hfreq : ∃ᶠ t in nhds r, radialRayMN n τ νp t ≠ 0 :=
        Filter.not_eventually.mp hev
      have hpos : ∀ᶠ t in nhds r, 0 < t := Ioi_mem_nhds hr
      have hfreq0 : ∃ᶠ t in nhds r, radialRayVN n hn0 τ νp νq t = 0 := by
        refine (hfreq.and_eventually hpos).mono ?_
        rintro t ⟨hmne, ht0⟩
        have hKM := radialRayKhatN_eq_M_mul_V hn τ hτ νp νq hsp hsq hzero ht0
        rw [hK0 t ht0] at hKM
        rcases mul_eq_zero.mp hKM.symm with h | h
        · rcases mul_eq_zero.mp h with h' | h'
          · exact absurd h' hτ.ne'
          · exact absurd h' hmne
        · exact h
      exact tendsto_nhds_unique_of_frequently_eq
        (continuousAt_radialRayVN hn τ hτ νp νq hsp hsq hr)
        tendsto_const_nhds hfreq0
  · have hKM := radialRayKhatN_eq_M_mul_V hn τ hτ νp νq hsp hsq hzero hr
    rw [hK0 r hr] at hKM
    rcases mul_eq_zero.mp hKM.symm with h | h
    · rcases mul_eq_zero.mp h with h' | h'
      · exact absurd h' hτ.ne'
      · exact absurd h' hm
    · exact h

/-- The two radial normalizer profiles are proportional on the open ray. -/
theorem radialRayZN_proportional
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ)
    (νp νq : Measure ℝ) [IsProbabilityMeasure νp] [IsProbabilityMeasure νq]
    (hsp : νp (Iio 0) = 0) (hsq : νq (Iio 0) = 0)
    (hmp : Integrable (fun s : ℝ => s ^ (n - 1)) νp)
    (hmq : Integrable (fun s : ℝ => s ^ (n - 1)) νq)
    (hslackp : RadialSlackN n (by omega) τ νp)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (radialMixtureN n νp) (radialMixtureN n νq)) :
    ∃ c : ℝ, ∀ r : ℝ, 0 < r →
      radialRayZN n τ νp r = c * radialRayZN n τ νq r := by
  have hn0 : 0 < n := by omega
  have hV0 := radialRayVN_eq_zero hn τ hτ νp νq hsp hsq hmp hmq hslackp hzero
  have hRderiv : ∀ r : ℝ, 0 < r → HasDerivAt
      (fun x => radialRayZN n τ νp x / radialRayZN n τ νq x) 0 r := by
    intro r hr
    have h1 := (hasDerivAt_radialRayZN hn τ hτ νp hsp hr).div
      (hasDerivAt_radialRayZN hn τ hτ νq hsq hr)
      (radialRayZN_pos hn τ hτ νq r).ne'
    have hnum : (1 / τ) * radialRayZdN n hn0 τ νp r * radialRayZN n τ νq r -
        radialRayZN n τ νp r * ((1 / τ) * radialRayZdN n hn0 τ νq r) = 0 := by
      have hv := hV0 r hr
      rw [radialRayVN, radialRayWN] at hv
      have hrpow : r ^ (n - 1) ≠ 0 := pow_ne_zero _ hr.ne'
      have hW := (mul_eq_zero.mp hv).resolve_left hrpow
      rcases mul_eq_zero.mp hW with h | h
      · exact absurd h (one_div_ne_zero hτ.ne')
      · linear_combination (1 / τ) * h
    have hzero' : ((1 / τ) * radialRayZdN n hn0 τ νp r * radialRayZN n τ νq r -
        radialRayZN n τ νp r * ((1 / τ) * radialRayZdN n hn0 τ νq r)) /
        (radialRayZN n τ νq r) ^ 2 = 0 := by rw [hnum]; simp
    rw [← hzero']
    exact h1
  refine ⟨radialRayZN n τ νp 1 / radialRayZN n τ νq 1, ?_⟩
  intro r hr
  have hconst : radialRayZN n τ νp r / radialRayZN n τ νq r =
      radialRayZN n τ νp 1 / radialRayZN n τ νq 1 := by
    have hu0 : 0 < min r 1 := lt_min hr one_pos
    have hMVT := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun x => radialRayZN n τ νp x / radialRayZN n τ νq x)
      (f' := fun _ => (0 : ℝ)) (s := Icc (min r 1) (max r 1)) (C := 0)
      (fun x hx => (hRderiv x (lt_of_lt_of_le hu0 hx.1)).hasDerivWithinAt)
      (fun x _ => by simp) (convex_Icc _ _)
      (Set.mem_Icc.mpr ⟨min_le_left r 1, le_max_left r 1⟩)
      (Set.mem_Icc.mpr ⟨min_le_right r 1, le_max_right r 1⟩)
    rw [zero_mul] at hMVT
    have hz := norm_le_zero_iff.mp hMVT
    have hsub := sub_eq_zero.mp hz
    linarith [hsub]
  have hZq : radialRayZN n τ νq r ≠ 0 := (radialRayZN_pos hn τ hτ νq r).ne'
  calc
    radialRayZN n τ νp r =
        (radialRayZN n τ νp r / radialRayZN n τ νq r) *
          radialRayZN n τ νq r := by field_simp
    _ = (radialRayZN n τ νp 1 / radialRayZN n τ νq 1) *
          radialRayZN n τ νq r := by rw [hconst]

end DriftingIdentifiability
