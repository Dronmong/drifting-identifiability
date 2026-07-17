import DriftingIdentifiability.LaplaceRadialShell3

/-!
# Radial Laplace converse, milestone G1 (general `n ≥ 3`): the zonal shell layer

First file of the G1 generalization (`LaplaceRnRoadmap.md` §3): the per-shell
zonal objects of the `n`-dimensional radial program, over the weighted zonal
base `(1-u²)^((n-3)/2) du` on `[-1,1]` (the height-coordinate density of the
uniform sphere measure).  At `n = 3` the weight is `1` and these objects
reduce to the `Shell3` ones (Archimedes' hat-box theorem).

Key content:

* `zonalWeight n u = (1-u²)^((n-3)/2)` (real power), its bounds/continuity,
  and the abstract normalizer `zonalMass n` (positive, finite — its closed
  form is never needed; no Gamma functions anywhere).
* the per-shell objects `shellZN`, `shellCN`, `shellDN`, `shellTN`,
  `shellRhoSqOverDistN` as weighted zonal averages.
* **the general-`n` tangential identity (T)** —
  `shellRhoSqOverDistN = ((n-1)τ/r) · shellTN` — proved by ONE integration by
  parts in `u` against the primitive `(1-u²)^((n-1)/2)` (whose boundary
  values vanish for every `n ≥ 3`), valid off the collision shell `s = r`,
  and extended across the collision by continuity in `s` (the same
  `𝓝[≠] r`-gluing pattern as `chartBase_tilted_eq_shellZ'` at `n = 3`).
  This replaces the `n = 3` polynomial reverse-distance substitution, which
  does not generalize to even `n`.

Design record: `LaplaceHigherDim.md` §4.9(R2) for the identity,
`LaplaceRnRoadmap.md` §3(G1) for the file plan.
-/

open MeasureTheory Filter Topology Set

namespace DriftingIdentifiability

open Paper

/-! ## The zonal weight and its mass -/

/-- The zonal exponent `(n-3)/2` of the height-coordinate density of the
uniform measure on `S^{n-1}`. -/
noncomputable def zonalExponent (n : ℕ) : ℝ := ((n : ℝ) - 3) / 2

lemma zonalExponent_nonneg {n : ℕ} (hn : 3 ≤ n) : 0 ≤ zonalExponent n := by
  have h3 : (3 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  unfold zonalExponent
  linarith

lemma one_le_zonalExponent_add_one {n : ℕ} (hn : 3 ≤ n) :
    (1 : ℝ) ≤ zonalExponent n + 1 := by
  linarith [zonalExponent_nonneg hn]

/-- The zonal weight `(1-u²)^((n-3)/2)` (real power). -/
noncomputable def zonalWeight (n : ℕ) (u : ℝ) : ℝ :=
  (1 - u ^ 2) ^ zonalExponent n

lemma zonalWeight_nonneg {n : ℕ} {u : ℝ} (hu : u ^ 2 ≤ 1) :
    0 ≤ zonalWeight n u :=
  Real.rpow_nonneg (by linarith) _

lemma zonalWeight_le_one {n : ℕ} (hn : 3 ≤ n) {u : ℝ} (hu : u ^ 2 ≤ 1) :
    zonalWeight n u ≤ 1 :=
  Real.rpow_le_one (by linarith) (by linarith [sq_nonneg u])
    (zonalExponent_nonneg hn)

lemma zonalWeight_pos {n : ℕ} {u : ℝ} (hu : u ^ 2 < 1) :
    0 < zonalWeight n u :=
  Real.rpow_pos_of_pos (by linarith) _

lemma continuous_zonalWeight {n : ℕ} (hn : 3 ≤ n) :
    Continuous (zonalWeight n) := by
  refine continuous_iff_continuousAt.mpr fun u => ?_
  exact (Real.continuousAt_rpow_const _ _
    (Or.inr (zonalExponent_nonneg hn))).comp
    ((continuous_const.sub (continuous_pow 2)).continuousAt)

lemma sq_le_one_of_mem_Ioc {u : ℝ} (hu : u ∈ Ioc (-1 : ℝ) 1) : u ^ 2 ≤ 1 := by
  nlinarith [hu.1, hu.2]

private lemma isFiniteMeasure_zonalRestrict :
    IsFiniteMeasure (volume.restrict (Ioc (-1 : ℝ) 1)) := by
  constructor
  rw [Measure.restrict_apply_univ, Real.volume_Ioc]
  exact ENNReal.ofReal_lt_top

lemma integrableOn_zonalWeight {n : ℕ} (hn : 3 ≤ n) :
    IntegrableOn (zonalWeight n) (Ioc (-1 : ℝ) 1) :=
  ((continuous_zonalWeight hn).integrableOn_Icc).mono_set Ioc_subset_Icc_self

/-- The zonal mass `∫_{-1}^{1} (1-u²)^((n-3)/2) du` — the (unnormalized) total
weight.  Positive and finite; its closed form (a Beta value) is never used. -/
noncomputable def zonalMass (n : ℕ) : ℝ :=
  ∫ u in Ioc (-1 : ℝ) 1, zonalWeight n u

lemma zonalMass_pos {n : ℕ} (hn : 3 ≤ n) : 0 < zonalMass n := by
  have hsub : Ioc (-(1 : ℝ) / 2) (1 / 2) ⊆ Ioc (-1 : ℝ) 1 := by
    intro u hu
    exact ⟨by linarith [hu.1], by linarith [hu.2]⟩
  have hconst : (0 : ℝ) < (3 / 4 : ℝ) ^ zonalExponent n :=
    Real.rpow_pos_of_pos (by norm_num) _
  have hlow : ∀ u ∈ Ioc (-(1 : ℝ) / 2) (1 / 2),
      (3 / 4 : ℝ) ^ zonalExponent n ≤ zonalWeight n u := by
    intro u hu
    have hbase : (3 / 4 : ℝ) ≤ 1 - u ^ 2 := by nlinarith [hu.1, hu.2]
    exact Real.rpow_le_rpow (by norm_num) hbase (zonalExponent_nonneg hn)
  have hsmall : (3 / 4 : ℝ) ^ zonalExponent n
      ≤ ∫ u in Ioc (-(1 : ℝ) / 2) (1 / 2), zonalWeight n u := by
    have hint : IntegrableOn (zonalWeight n) (Ioc (-(1 : ℝ) / 2) (1 / 2)) :=
      (integrableOn_zonalWeight hn).mono_set hsub
    have hcint : IntegrableOn (fun _ : ℝ => (3 / 4 : ℝ) ^ zonalExponent n)
        (Ioc (-(1 : ℝ) / 2) (1 / 2)) :=
      (continuous_const.integrableOn_Icc).mono_set Ioc_subset_Icc_self
    have hmono := setIntegral_mono_on hcint hint measurableSet_Ioc hlow
    have hcv : (∫ _ in Ioc (-(1 : ℝ) / 2) (1 / 2),
          ((3 / 4 : ℝ) ^ zonalExponent n) ∂volume)
        = (volume : Measure ℝ).real (Ioc (-(1 : ℝ) / 2) (1 / 2))
            • ((3 / 4 : ℝ) ^ zonalExponent n) :=
      setIntegral_const _
    have hvol : (volume : Measure ℝ).real (Ioc (-(1 : ℝ) / 2) (1 / 2)) = 1 := by
      rw [Real.volume_real_Ioc_of_le (by norm_num)]
      norm_num
    rw [hcv, hvol, one_smul] at hmono
    exact hmono
  have hmono2 : ∫ u in Ioc (-(1 : ℝ) / 2) (1 / 2), zonalWeight n u
      ≤ zonalMass n := by
    refine setIntegral_mono_set (integrableOn_zonalWeight hn) ?_
      (HasSubset.Subset.eventuallyLE hsub)
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    exact zonalWeight_nonneg (sq_le_one_of_mem_Ioc hu)
  exact lt_of_lt_of_le (lt_of_lt_of_le hconst hsmall) hmono2

lemma zonalMass_ne_zero {n : ℕ} (hn : 3 ≤ n) : zonalMass n ≠ 0 :=
  (zonalMass_pos hn).ne'

/-! ## The per-shell zonal objects -/

/-- Per-shell zonal average of the Laplace kernel in dimension `n`. -/
noncomputable def shellZN (n : ℕ) (τ r s : ℝ) : ℝ :=
  (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
    zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)

/-- Per-shell zonal average of the companion kernel `(τ + d)e^{-d/τ}`. -/
noncomputable def shellCN (n : ℕ) (τ r s : ℝ) : ℝ :=
  (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
    zonalWeight n u *
      ((τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u))

/-- Per-shell zonal average of the axial drift numerator `(su - r)e^{-d/τ}`. -/
noncomputable def shellDN (n : ℕ) (τ r s : ℝ) : ℝ :=
  (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
    zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r))

/-- Per-shell zonal average of the axial coordinate `su·e^{-d/τ}`. -/
noncomputable def shellTN (n : ℕ) (τ r s : ℝ) : ℝ :=
  (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
    zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u))

/-- Per-shell zonal average of `(ρ²/d)·e^{-d/τ}`. -/
noncomputable def shellRhoSqOverDistN (n : ℕ) (τ r s : ℝ) : ℝ :=
  (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
    zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) *
        (shellRhoSq s u / shellDist r s u))

/-! ## Elementary distance facts -/

lemma shellDist_nonneg (r s u : ℝ) : 0 ≤ shellDist r s u := Real.sqrt_nonneg _

lemma shellDist_sq_pos_of_ne {r s : ℝ} (hr : 0 < r) (hs : 0 ≤ s) (hne : s ≠ r)
    {u : ℝ} (hu : u ≤ 1) : 0 < r ^ 2 + s ^ 2 - 2 * r * s * u := by
  have h1 : r - s ≠ 0 := sub_ne_zero.mpr (Ne.symm hne)
  have h2 : 0 < (r - s) ^ 2 :=
    lt_of_le_of_ne (sq_nonneg (r - s)) (Ne.symm (pow_ne_zero 2 h1))
  nlinarith [mul_nonneg (mul_nonneg hr.le hs) (sub_nonneg.mpr hu)]

lemma shellDist_pos_of_ne {r s : ℝ} (hr : 0 < r) (hs : 0 ≤ s) (hne : s ≠ r)
    {u : ℝ} (hu : u ≤ 1) : 0 < shellDist r s u :=
  Real.sqrt_pos.mpr (shellDist_sq_pos_of_ne hr hs hne hu)

lemma shellDist_le_add {r s : ℝ} (hr : 0 ≤ r) (hs : 0 ≤ s) {u : ℝ}
    (hu : -1 ≤ u) : shellDist r s u ≤ r + s := by
  have hq : r ^ 2 + s ^ 2 - 2 * r * s * u ≤ (r + s) ^ 2 := by
    nlinarith [mul_nonneg (mul_nonneg hr hs) (by linarith : (0 : ℝ) ≤ 1 + u)]
  calc shellDist r s u ≤ Real.sqrt ((r + s) ^ 2) := Real.sqrt_le_sqrt hq
    _ = r + s := Real.sqrt_sq (by linarith)

lemma shellRhoSq_div_shellDist_le {s u : ℝ} (hu : u ^ 2 ≤ 1) (r : ℝ) :
    shellRhoSq s u / shellDist r s u ≤ shellDist r s u := by
  rcases eq_or_lt_of_le (shellDist_nonneg r s u) with h0 | hpos
  · rw [← h0, div_zero]
  · rw [div_le_iff₀ hpos]
    have hd2 : shellDist r s u ^ 2 = shellAxial r s u ^ 2 + shellRhoSq s u :=
      shellDist_sq_eq_axial_add_rho hu r s
    nlinarith [sq_nonneg (shellAxial r s u)]

lemma shellRhoSq_div_shellDist_nonneg {s u : ℝ} (hu : u ^ 2 ≤ 1) (r : ℝ) :
    0 ≤ shellRhoSq s u / shellDist r s u :=
  div_nonneg (shellRhoSq_nonneg hu s) (shellDist_nonneg r s u)

lemma exp_shellDist_le_one {τ : ℝ} (hτ : 0 < τ) (r s u : ℝ) :
    Real.exp (-(1 / τ) * shellDist r s u) ≤ 1 := by
  have h0 : -(1 / τ) * shellDist r s u ≤ 0 := by
    have hnn := mul_nonneg (one_div_pos.mpr hτ).le (shellDist_nonneg r s u)
    linarith [neg_mul (1 / τ) (shellDist r s u)]
  have h := Real.exp_le_exp.mpr h0
  rwa [Real.exp_zero] at h

lemma continuous_shellDist_u (r s : ℝ) :
    Continuous (fun u => shellDist r s u) := by
  have hfe : (fun u => shellDist r s u)
      = fun u => Real.sqrt (r ^ 2 + s ^ 2 - 2 * r * s * u) := rfl
  rw [hfe]
  exact Real.continuous_sqrt.comp
    (continuous_const.sub (continuous_const.mul continuous_id))

lemma continuous_shellDist_s (r u : ℝ) :
    Continuous (fun s => shellDist r s u) := by
  have hfe : (fun s => shellDist r s u)
      = fun s => Real.sqrt (r ^ 2 + s ^ 2 - 2 * r * s * u) := rfl
  rw [hfe]
  refine Real.continuous_sqrt.comp ?_
  have h2 : Continuous fun s : ℝ => 2 * r * s * u :=
    (continuous_const.mul continuous_id).mul continuous_const
  exact (continuous_const.add (continuous_pow 2)).sub h2

/-! ## The zonal primitive and its derivative -/

/-- Derivative of the boundary-vanishing primitive `(1-u²)^((n-1)/2)`:
`-(n-1)·u·(1-u²)^((n-3)/2)`. -/
lemma hasDerivAt_zonalPrimitive {n : ℕ} (hn : 3 ≤ n) (u : ℝ) :
    HasDerivAt (fun t => (1 - t ^ 2) ^ (zonalExponent n + 1))
      (-(((n : ℝ) - 1) * u * zonalWeight n u)) u := by
  have hg : HasDerivAt (fun t : ℝ => 1 - t ^ 2) (-(2 * u)) u := by
    have h := (hasDerivAt_const u (1 : ℝ)).sub (hasDerivAt_pow 2 u)
    have hval : (0 : ℝ) - ((2 : ℕ) : ℝ) * u ^ (2 - 1) = -(2 * u) := by
      norm_num
    rw [hval] at h
    exact h
  have hh : HasDerivAt (fun x : ℝ => x ^ (zonalExponent n + 1))
      ((zonalExponent n + 1) * (1 - u ^ 2) ^ (zonalExponent n + 1 - 1))
      (1 - u ^ 2) :=
    Real.hasDerivAt_rpow_const (Or.inr (one_le_zonalExponent_add_one hn))
  have h := hh.comp u hg
  have hexp : zonalExponent n + 1 - 1 = zonalExponent n := by ring
  rw [hexp] at h
  have h2 : 2 * (zonalExponent n + 1) = (n : ℝ) - 1 := by
    unfold zonalExponent
    ring
  have hval : (zonalExponent n + 1) * (1 - u ^ 2) ^ zonalExponent n * -(2 * u)
      = -(((n : ℝ) - 1) * u * zonalWeight n u) := by
    unfold zonalWeight
    linear_combination (-(u * ((1 - u ^ 2) ^ zonalExponent n))) * h2
  rw [hval] at h
  exact h

/-- On `u² ≤ 1` the primitive factors as `(1-u²)·zonalWeight`. -/
lemma zonalPrimitive_eq_weight_mul {n : ℕ} (hn : 3 ≤ n) {u : ℝ}
    (hu : u ^ 2 ≤ 1) :
    (1 - u ^ 2) ^ (zonalExponent n + 1) = (1 - u ^ 2) * zonalWeight n u := by
  rcases lt_or_eq_of_le hu with hlt | heq
  · have hbase : (1 - u ^ 2) ≠ 0 := by linarith
    rw [Real.rpow_add_one hbase, zonalWeight, mul_comm]
  · have hbase : 1 - u ^ 2 = 0 := by linarith
    rw [hbase, Real.zero_rpow, zero_mul]
    have h1 := one_le_zonalExponent_add_one hn
    linarith

lemma zonalPrimitive_boundary {n : ℕ} (hn : 3 ≤ n) {u : ℝ} (hu : u ^ 2 = 1) :
    (1 - u ^ 2) ^ (zonalExponent n + 1) = 0 := by
  rw [hu, sub_self, Real.zero_rpow]
  have h1 := one_le_zonalExponent_add_one hn
  linarith

/-! ## The zonal kernel derivative in `u` (off the collision shell) -/

lemma hasDerivAt_shellDist_u {r s u : ℝ}
    (hq : 0 < r ^ 2 + s ^ 2 - 2 * r * s * u) :
    HasDerivAt (fun t => shellDist r s t)
      (-(r * s / shellDist r s u)) u := by
  have hqd : HasDerivAt (fun t : ℝ => r ^ 2 + s ^ 2 - 2 * r * s * t)
      (-(2 * r * s)) u := by
    have h := (hasDerivAt_const u (r ^ 2 + s ^ 2)).sub
      ((hasDerivAt_id u).const_mul (2 * r * s))
    have hval : (0 : ℝ) - 2 * r * s * 1 = -(2 * r * s) := by ring
    rw [hval] at h
    exact h
  have hsq := (Real.hasDerivAt_sqrt hq.ne').comp u hqd
  have hd0 : Real.sqrt (r ^ 2 + s ^ 2 - 2 * r * s * u) ≠ 0 :=
    (Real.sqrt_pos.mpr hq).ne'
  have hval : 1 / (2 * Real.sqrt (r ^ 2 + s ^ 2 - 2 * r * s * u))
        * -(2 * r * s)
      = -(r * s / shellDist r s u) := by
    have hde : shellDist r s u
        = Real.sqrt (r ^ 2 + s ^ 2 - 2 * r * s * u) := rfl
    rw [hde]
    field_simp
  rw [hval] at hsq
  exact hsq

lemma hasDerivAt_shellKernel_u {τ r s u : ℝ} (hτ : 0 < τ)
    (hq : 0 < r ^ 2 + s ^ 2 - 2 * r * s * u) :
    HasDerivAt (fun t => Real.exp (-(1 / τ) * shellDist r s t))
      (r * s / (τ * shellDist r s u)
        * Real.exp (-(1 / τ) * shellDist r s u)) u := by
  have hd := (hasDerivAt_shellDist_u hq).const_mul (-(1 / τ))
  have hexp := hd.exp
  have hd0 : shellDist r s u ≠ 0 := (Real.sqrt_pos.mpr hq).ne'
  have hval : Real.exp (-(1 / τ) * shellDist r s u)
        * (-(1 / τ) * -(r * s / shellDist r s u))
      = r * s / (τ * shellDist r s u)
        * Real.exp (-(1 / τ) * shellDist r s u) := by
    field_simp
  rw [hval] at hexp
  exact hexp

/-! ## The integration by parts -/

/-- **The zonal IBP** (off-collision): for `n ≥ 3`, `r > 0`, `0 ≤ s ≠ r`,

`∫ (n-1)·u·W(u)·K(u) du = (rs/τ) ∫ (1-u²)·W(u)·(K(u)/d(u)) du`

over `u ∈ [-1,1]`, where `W` is the zonal weight and `K` the shell kernel.
Boundary terms vanish because the primitive `(1-u²)^((n-1)/2)` is zero at
`u = ±1` for every `n ≥ 3`. -/
lemma zonal_ibp {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ} (hτ : 0 < τ) (hr : 0 < r)
    (hs : 0 ≤ s) (hne : s ≠ r) :
    (∫ u in (-1 : ℝ)..1, ((n : ℝ) - 1) * u * zonalWeight n u
        * Real.exp (-(1 / τ) * shellDist r s u))
      = (r * s / τ) * ∫ u in (-1 : ℝ)..1,
          (1 - u ^ 2) * zonalWeight n u
            * (Real.exp (-(1 / τ) * shellDist r s u) / shellDist r s u) := by
  have huIcc : uIcc (-1 : ℝ) 1 = Icc (-1 : ℝ) 1 := uIcc_of_le (by norm_num)
  have hsq_of_mem : ∀ x ∈ uIcc (-1 : ℝ) 1, x ^ 2 ≤ 1 := by
    intro x hx
    rw [huIcc] at hx
    nlinarith [hx.1, hx.2]
  have hq_of_mem : ∀ x ∈ uIcc (-1 : ℝ) 1,
      0 < r ^ 2 + s ^ 2 - 2 * r * s * x := by
    intro x hx
    rw [huIcc] at hx
    exact shellDist_sq_pos_of_ne hr hs hne hx.2
  have hFd : ∀ x ∈ uIcc (-1 : ℝ) 1,
      HasDerivAt (fun t => (1 - t ^ 2) ^ (zonalExponent n + 1))
        (-(((n : ℝ) - 1) * x * zonalWeight n x)) x :=
    fun x _ => hasDerivAt_zonalPrimitive hn x
  have hKd : ∀ x ∈ uIcc (-1 : ℝ) 1,
      HasDerivAt (fun t => Real.exp (-(1 / τ) * shellDist r s t))
        (r * s / (τ * shellDist r s x)
          * Real.exp (-(1 / τ) * shellDist r s x)) x :=
    fun x hx => hasDerivAt_shellKernel_u hτ (hq_of_mem x hx)
  have hFi : IntervalIntegrable
      (fun x => -(((n : ℝ) - 1) * x * zonalWeight n x)) volume (-1 : ℝ) 1 := by
    refine Continuous.intervalIntegrable ?_ _ _
    exact ((continuous_const.mul continuous_id).mul
      (continuous_zonalWeight hn)).neg
  have hKi : IntervalIntegrable
      (fun x => r * s / (τ * shellDist r s x)
        * Real.exp (-(1 / τ) * shellDist r s x)) volume (-1 : ℝ) 1 := by
    refine ContinuousOn.intervalIntegrable ?_
    have hden : ∀ x ∈ uIcc (-1 : ℝ) 1, τ * shellDist r s x ≠ 0 := by
      intro x hx
      rw [huIcc] at hx
      exact (mul_pos hτ (shellDist_pos_of_ne hr hs hne hx.2)).ne'
    refine ContinuousOn.mul ?_ ?_
    · exact continuousOn_const.div
        (continuous_const.mul (continuous_shellDist_u r s)).continuousOn hden
    · exact (Real.continuous_exp.comp
        ((continuous_shellDist_u r s).const_mul (-(1 / τ)))).continuousOn
  have hibp := intervalIntegral.integral_mul_deriv_eq_deriv_mul hFd hKd hFi hKi
  have hb1 : (1 - (1 : ℝ) ^ 2) ^ (zonalExponent n + 1) = 0 :=
    zonalPrimitive_boundary hn (by norm_num)
  have hbm : (1 - (-1 : ℝ) ^ 2) ^ (zonalExponent n + 1) = 0 :=
    zonalPrimitive_boundary hn (by norm_num)
  rw [hb1, hbm, zero_mul, zero_mul, sub_zero, zero_sub] at hibp
  have hleft : (∫ x in (-1 : ℝ)..1,
        -(((n : ℝ) - 1) * x * zonalWeight n x)
          * Real.exp (-(1 / τ) * shellDist r s x))
      = -∫ x in (-1 : ℝ)..1, ((n : ℝ) - 1) * x * zonalWeight n x
          * Real.exp (-(1 / τ) * shellDist r s x) := by
    rw [← intervalIntegral.integral_neg]
    exact intervalIntegral.integral_congr fun x _ => by ring
  have hright : (∫ x in (-1 : ℝ)..1,
        (1 - x ^ 2) ^ (zonalExponent n + 1)
          * (r * s / (τ * shellDist r s x)
            * Real.exp (-(1 / τ) * shellDist r s x)))
      = (r * s / τ) * ∫ x in (-1 : ℝ)..1,
          (1 - x ^ 2) * zonalWeight n x
            * (Real.exp (-(1 / τ) * shellDist r s x) / shellDist r s x) := by
    rw [← intervalIntegral.integral_const_mul]
    refine intervalIntegral.integral_congr fun x hx => ?_
    rw [zonalPrimitive_eq_weight_mul hn (hsq_of_mem x hx)]
    have hd0 : shellDist r s x ≠ 0 := by
      rw [huIcc] at hx
      exact (shellDist_pos_of_ne hr hs hne hx.2).ne'
    have hτ0 : τ ≠ 0 := hτ.ne'
    field_simp
  rw [hleft, hright] at hibp
  linarith [hibp]

/-! ## The tangential identity (T) -/

private noncomputable def shellTInt (n : ℕ) (τ r s : ℝ) : ℝ :=
  ∫ u in Ioc (-1 : ℝ) 1,
    zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u))

private noncomputable def shellRhoInt (n : ℕ) (τ r s : ℝ) : ℝ :=
  ∫ u in Ioc (-1 : ℝ) 1,
    zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u)
        * (shellRhoSq s u / shellDist r s u))

private lemma shellTN_eq_int (n : ℕ) (τ r s : ℝ) :
    shellTN n τ r s = (zonalMass n)⁻¹ * shellTInt n τ r s := rfl

private lemma shellRhoSqOverDistN_eq_int (n : ℕ) (τ r s : ℝ) :
    shellRhoSqOverDistN n τ r s = (zonalMass n)⁻¹ * shellRhoInt n τ r s := rfl

private lemma shellRhoInt_eq_of_ne {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ}
    (hτ : 0 < τ) (hr : 0 < r) (hs : 0 ≤ s) (hne : s ≠ r) :
    shellRhoInt n τ r s = (((n : ℝ) - 1) * τ / r) * shellTInt n τ r s := by
  rcases eq_or_lt_of_le hs with hs0 | hspos
  · -- `s = 0`: both integrands vanish identically
    have hzero1 : ∀ u : ℝ, zonalWeight n u
        * (Real.exp (-(1 / τ) * shellDist r s u)
          * (shellRhoSq s u / shellDist r s u)) = 0 := by
      intro u
      have hfe : shellRhoSq s u = s ^ 2 * (1 - u ^ 2) := rfl
      rw [hfe, ← hs0]
      norm_num
    have hzero2 : ∀ u : ℝ, zonalWeight n u
        * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u)) = 0 := by
      intro u
      rw [← hs0]
      norm_num
    unfold shellRhoInt shellTInt
    rw [setIntegral_congr_fun measurableSet_Ioc fun u _ => hzero1 u,
      setIntegral_congr_fun measurableSet_Ioc fun u _ => hzero2 u]
    simp
  · -- `0 < s ≠ r`: the IBP plus algebra (the `s`-powers cancel)
    have hibp := zonal_ibp hn hτ hr hs hne
    have hn1 : ((n : ℝ) - 1) ≠ 0 := by
      have h3 : (3 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
      linarith
    unfold shellRhoInt shellTInt
    rw [integral_Ioc_neg_one_one_eq_interval,
      integral_Ioc_neg_one_one_eq_interval]
    have hTform : (∫ u in (-1 : ℝ)..1, zonalWeight n u
          * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u)))
        = (s / ((n : ℝ) - 1)) * ∫ u in (-1 : ℝ)..1,
            ((n : ℝ) - 1) * u * zonalWeight n u
              * Real.exp (-(1 / τ) * shellDist r s u) := by
      rw [← intervalIntegral.integral_const_mul]
      refine intervalIntegral.integral_congr fun x _ => ?_
      field_simp
    have hRform : (∫ u in (-1 : ℝ)..1, zonalWeight n u
          * (Real.exp (-(1 / τ) * shellDist r s u)
            * (shellRhoSq s u / shellDist r s u)))
        = s ^ 2 * ∫ u in (-1 : ℝ)..1,
            (1 - u ^ 2) * zonalWeight n u
              * (Real.exp (-(1 / τ) * shellDist r s u) / shellDist r s u) := by
      rw [← intervalIntegral.integral_const_mul]
      refine intervalIntegral.integral_congr fun x _ => ?_
      have hfe : shellRhoSq s x = s ^ 2 * (1 - x ^ 2) := rfl
      rw [hfe]
      ring
    rw [hRform, hTform, hibp]
    have hτ0 : τ ≠ 0 := hτ.ne'
    have hr0 : r ≠ 0 := hr.ne'
    field_simp

/-! ### Continuity in `s` at the collision shell -/

private lemma continuousAt_shellTInt {n : ℕ} (hn : 3 ≤ n) {τ r : ℝ}
    (hτ : 0 < τ) (hr : 0 < r) :
    ContinuousAt (fun s => shellTInt n τ r s) r := by
  haveI := isFiniteMeasure_zonalRestrict
  unfold shellTInt
  refine continuousAt_of_dominated (bound := fun _ => 2 * r) ?_ ?_ ?_ ?_
  · filter_upwards with s
    refine Continuous.aestronglyMeasurable ?_
    exact (continuous_zonalWeight hn).mul
      ((Real.continuous_exp.comp
        ((continuous_shellDist_u r s).const_mul (-(1 / τ)))).mul
        (continuous_id.const_mul s))
  · filter_upwards [Metric.ball_mem_nhds r hr] with s hs
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
    have hsr : |s - r| < r := by
      have h := Metric.mem_ball.mp hs
      rwa [Real.dist_eq] at h
    obtain ⟨hs1, hs2⟩ := abs_lt.mp hsr
    have hW : |zonalWeight n u| ≤ 1 := by
      rw [abs_of_nonneg (zonalWeight_nonneg hu2)]
      exact zonalWeight_le_one hn hu2
    have hK : |Real.exp (-(1 / τ) * shellDist r s u)| ≤ 1 := by
      rw [abs_of_pos (Real.exp_pos _)]
      exact exp_shellDist_le_one hτ r s u
    have hsu : |s * u| ≤ 2 * r := by
      rw [abs_mul]
      have hu1 : |u| ≤ 1 := abs_le.mpr ⟨hu.1.le, hu.2⟩
      have hsabs : |s| ≤ 2 * r := by
        rw [abs_of_pos (by linarith : (0 : ℝ) < s)]
        linarith
      calc |s| * |u| ≤ (2 * r) * 1 :=
            mul_le_mul hsabs hu1 (abs_nonneg u) (by linarith)
        _ = 2 * r := mul_one _
    rw [Real.norm_eq_abs, abs_mul, abs_mul]
    calc |zonalWeight n u|
          * (|Real.exp (-(1 / τ) * shellDist r s u)| * |s * u|)
        ≤ 1 * (1 * (2 * r)) := by
          refine mul_le_mul hW ?_ (by positivity) zero_le_one
          exact mul_le_mul hK hsu (abs_nonneg _) zero_le_one
      _ = 2 * r := by ring
  · exact integrable_const _
  · filter_upwards with u
    have hK : Continuous fun s => Real.exp (-(1 / τ) * shellDist r s u) :=
      Real.continuous_exp.comp
        ((continuous_shellDist_s r u).const_mul (-(1 / τ)))
    exact (continuous_const.mul
      (hK.mul (continuous_id.mul continuous_const))).continuousAt

private lemma continuousAt_shellRhoInt {n : ℕ} (hn : 3 ≤ n) {τ r : ℝ}
    (hτ : 0 < τ) (hr : 0 < r) :
    ContinuousAt (fun s => shellRhoInt n τ r s) r := by
  haveI := isFiniteMeasure_zonalRestrict
  unfold shellRhoInt
  refine continuousAt_of_dominated (bound := fun _ => 3 * r) ?_ ?_ ?_ ?_
  · filter_upwards with s
    refine Measurable.aestronglyMeasurable ?_
    refine ((continuous_zonalWeight hn).measurable).mul ?_
    refine ((Real.continuous_exp.comp
      ((continuous_shellDist_u r s).const_mul (-(1 / τ)))).measurable).mul ?_
    have hnum : Measurable fun u => shellRhoSq s u := by
      have hfe : (fun u => shellRhoSq s u)
          = fun u => s ^ 2 * (1 - u ^ 2) := rfl
      rw [hfe]
      exact (continuous_const.mul
        (continuous_const.sub (continuous_pow 2))).measurable
    exact hnum.div (continuous_shellDist_u r s).measurable
  · filter_upwards [Metric.ball_mem_nhds r hr] with s hs
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
    have hsr : |s - r| < r := by
      have h := Metric.mem_ball.mp hs
      rwa [Real.dist_eq] at h
    obtain ⟨hs1, hs2⟩ := abs_lt.mp hsr
    have hspos : (0 : ℝ) < s := by linarith
    have hW : |zonalWeight n u| ≤ 1 := by
      rw [abs_of_nonneg (zonalWeight_nonneg hu2)]
      exact zonalWeight_le_one hn hu2
    have hK : |Real.exp (-(1 / τ) * shellDist r s u)| ≤ 1 := by
      rw [abs_of_pos (Real.exp_pos _)]
      exact exp_shellDist_le_one hτ r s u
    have hpay : |shellRhoSq s u / shellDist r s u| ≤ 3 * r := by
      rw [abs_of_nonneg (shellRhoSq_div_shellDist_nonneg hu2 r)]
      calc shellRhoSq s u / shellDist r s u
          ≤ shellDist r s u := shellRhoSq_div_shellDist_le hu2 r
        _ ≤ r + s := shellDist_le_add hr.le hspos.le hu.1.le
        _ ≤ 3 * r := by linarith
    rw [Real.norm_eq_abs, abs_mul, abs_mul]
    calc |zonalWeight n u|
          * (|Real.exp (-(1 / τ) * shellDist r s u)|
            * |shellRhoSq s u / shellDist r s u|)
        ≤ 1 * (1 * (3 * r)) := by
          refine mul_le_mul hW ?_ (by positivity) zero_le_one
          exact mul_le_mul hK hpay (abs_nonneg _) zero_le_one
      _ = 3 * r := by ring
  · exact integrable_const _
  · filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
    rcases eq_or_lt_of_le hu.2 with h1 | hlt
    · -- `u = 1`: the integrand is identically zero in `s`
      have hzero : (fun s => zonalWeight n u
          * (Real.exp (-(1 / τ) * shellDist r s u)
            * (shellRhoSq s u / shellDist r s u))) = fun _ => (0 : ℝ) := by
        funext s
        have hfe : shellRhoSq s u = s ^ 2 * (1 - u ^ 2) := rfl
        have hrho : shellRhoSq s u = 0 := by
          rw [hfe, h1]
          norm_num
        rw [hrho, zero_div, mul_zero, mul_zero]
      rw [hzero]
      exact continuousAt_const
    · -- `u < 1`: the denominator is bounded away from zero near `s = r`
      have hdpos : 0 < shellDist r r u := by
        refine Real.sqrt_pos.mpr ?_
        nlinarith [mul_pos hr hr, hlt]
      have hK : Continuous fun s => Real.exp (-(1 / τ) * shellDist r s u) :=
        Real.continuous_exp.comp
          ((continuous_shellDist_s r u).const_mul (-(1 / τ)))
      have hnum : Continuous fun s => shellRhoSq s u := by
        have hfe : (fun s => shellRhoSq s u)
            = fun s => s ^ 2 * (1 - u ^ 2) := rfl
        rw [hfe]
        exact (continuous_pow 2).mul continuous_const
      have hdiv : ContinuousAt
          (fun s => shellRhoSq s u / shellDist r s u) r :=
        hnum.continuousAt.div (continuous_shellDist_s r u).continuousAt
          hdpos.ne'
      exact continuousAt_const.mul (hK.continuousAt.mul hdiv)

/-- **The general-`n` tangential identity (T)**: for every `n ≥ 3`, `τ > 0`,
probe radius `r > 0`, and every shell `s ≥ 0` (the collision shell `s = r`
included, by continuity in `s`),

`shellRhoSqOverDistN = ((n-1)τ/r) · shellTN`.

This is the radial form of the Matérn companion PDE, obtained with no second
derivatives — it replaces the entire `Δ`-layer of the `n`-dimensional radial
program (design record `LaplaceHigherDim.md` §4.9(R2)). -/
theorem shellRhoSqOverDistN_eq_shellTN {n : ℕ} (hn : 3 ≤ n) {τ : ℝ}
    (hτ : 0 < τ) {r : ℝ} (hr : 0 < r) {s : ℝ} (hs : 0 ≤ s) :
    shellRhoSqOverDistN n τ r s
      = (((n : ℝ) - 1) * τ / r) * shellTN n τ r s := by
  rcases ne_or_eq s r with hne | heq
  · rw [shellRhoSqOverDistN_eq_int, shellTN_eq_int,
      shellRhoInt_eq_of_ne hn hτ hr hs hne]
    ring
  · subst s
    have hRT : shellRhoInt n τ r r
        = (((n : ℝ) - 1) * τ / r) * shellTInt n τ r r := by
      have hR : Tendsto (fun s => shellRhoInt n τ r s) (𝓝 r)
          (𝓝 (shellRhoInt n τ r r)) := continuousAt_shellRhoInt hn hτ hr
      have hT : Tendsto
          (fun s => (((n : ℝ) - 1) * τ / r) * shellTInt n τ r s) (𝓝 r)
          (𝓝 ((((n : ℝ) - 1) * τ / r) * shellTInt n τ r r)) :=
        (continuousAt_shellTInt hn hτ hr).const_mul _
      have hR' : Tendsto (fun s => shellRhoInt n τ r s) (𝓝[≠] r)
          (𝓝 (shellRhoInt n τ r r)) := hR.mono_left nhdsWithin_le_nhds
      have hT' : Tendsto
          (fun s => (((n : ℝ) - 1) * τ / r) * shellTInt n τ r s) (𝓝[≠] r)
          (𝓝 ((((n : ℝ) - 1) * τ / r) * shellTInt n τ r r)) :=
        hT.mono_left nhdsWithin_le_nhds
      have hball : Metric.ball r r ∈ 𝓝[≠] r :=
        nhdsWithin_le_nhds (Metric.ball_mem_nhds r hr)
      have heqev : (fun s => shellRhoInt n τ r s)
          =ᶠ[𝓝[≠] r] fun s => (((n : ℝ) - 1) * τ / r) * shellTInt n τ r s := by
        filter_upwards [self_mem_nhdsWithin, hball] with s hs1 hs2
        have hsr : s ≠ r := hs1
        have hspos : (0 : ℝ) < s := by
          have h := Metric.mem_ball.mp hs2
          rw [Real.dist_eq] at h
          obtain ⟨h1, h2⟩ := abs_lt.mp h
          linarith
        exact shellRhoInt_eq_of_ne hn hτ hr hspos.le hsr
      exact tendsto_nhds_unique_of_eventuallyEq hR' hT' heqev
    rw [shellRhoSqOverDistN_eq_int, shellTN_eq_int, hRT]
    ring

end DriftingIdentifiability
