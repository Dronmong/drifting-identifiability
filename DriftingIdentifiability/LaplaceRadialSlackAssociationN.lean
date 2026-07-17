import DriftingIdentifiability.LaplaceRadialSystemN

/-!
# G2: the slack-association reduction (per-shell layer)

This file begins the removal of the named `RadialSlackN` hypothesis
(`LaplaceRnRoadmap.md` §3(G2), sixth-pass reduction).  The mixture covariance
`τ(m̃'+1)Z² = QZ − D·Zd` decomposes over shell pairs through the doubling
identity

`q(s) + q(s') − m(s)z(s') − m(s')z(s)
   = Cov_within(s) + Cov_within(s') + (m(s) − m(s'))(z(s) − z(s'))`,

so `RadialSlackN` follows from (i) a per-shell covariance floor — provable
for every single shell — and (ii) the measure-free between-shell association
`(m(s) − m(s'))(z(s) − z(s')) ≥ 0` (`ZonalShellAssociation` below;
numerically verified across dimensions and scales in `numerics/rn_g2.py`).

This file provides:
* `shellZdN` — the per-shell tilted soft-sign profile `E[X/d]`-numerator,
  and its bridge to the y-level `radialRayZdN` (the collision shell `s = r`
  carries a removable singularity, squeezed through the exact identity
  `2r|X| = d²` valid on the collision shell);
* Dirac-profile collapse: every ray object at `ν = dirac s` is the
  corresponding shell object — so the PROVED mixture sign estimate
  `radialRayMDerivN_ge_of_le` instantiates to the near-shell (`s < r`)
  per-shell floor `shellRSIN_near` for free;
* the `ZonalShellAssociation` predicate and the pointwise two-shell bracket
  lemma `shell_pair_bracket` (pure algebra) consumed by the assembly.

The far branch (`s ≥ r`, comonotonicity in `u`) and the final assembly
`radialSlackN_of_zonalShellAssociation` follow in the companion file.
-/

open MeasureTheory Filter Topology Set Metric

namespace DriftingIdentifiability

open Paper

/-! ## The per-shell soft-sign profile -/

/-- Per-shell zonal average of the normalized axial displacement
`(su - r)/d · e^{-d/τ}` — the per-shell "soft sign" numerator. -/
noncomputable def shellZdN (n : ℕ) (τ r s : ℝ) : ℝ :=
  (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
    zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) *
        ((s * u - r) / shellDist r s u))

private lemma continuous_shellKernelN' (τ r s : ℝ) :
    Continuous fun u : ℝ => Real.exp (-(1 / τ) * shellDist r s u) :=
  Real.continuous_exp.comp
    ((continuous_shellDist_u r s).const_mul (-(1 / τ)))

/-- On the collision shell the axial-over-distance payload obeys the exact
quadratic identity `2r·|X| ≤ d²` (with equality), which drives the removable
singularity squeeze. -/
private lemma two_r_mul_abs_axial_le_sq_collision {r u : ℝ} (hr : 0 ≤ r)
    (hu : u ≤ 1) :
    2 * r * |r * u - r| ≤ shellDist r r u ^ 2 := by
  have harg : 0 ≤ r ^ 2 + r ^ 2 - 2 * r * r * u := by nlinarith
  have hde : shellDist r r u
      = Real.sqrt (r ^ 2 + r ^ 2 - 2 * r * r * u) := rfl
  have hd2 : shellDist r r u ^ 2 = r ^ 2 + r ^ 2 - 2 * r * r * u := by
    rw [hde]
    exact Real.sq_sqrt harg
  have habs : |r * u - r| = r - r * u := by
    rw [abs_of_nonpos (by nlinarith)]
    ring
  rw [hd2, habs]
  nlinarith

/-- The normalized axial payload `X/d` is bounded by `d/(2r)` on the
collision shell. -/
private lemma abs_axialDiv_le_dist_div_collision {r u : ℝ} (hr : 0 < r)
    (hu : u ≤ 1) :
    |(r * u - r) / shellDist r r u| ≤ shellDist r r u / (2 * r) := by
  rcases eq_or_lt_of_le (shellDist_nonneg r r u) with h0 | hpos
  · rw [← h0, div_zero, abs_zero, zero_div]
  · have h2r : (0 : ℝ) < 2 * r := by linarith
    have h := two_r_mul_abs_axial_le_sq_collision hr.le hu
    rw [abs_div, abs_of_pos hpos, div_le_div_iff₀ hpos h2r]
    nlinarith [h]

/-- The normalized axial payload is continuous on the physical coordinate
interval, the collision shell included (removable singularity). -/
private lemma continuousOn_shellAxialDivN {r s : ℝ} (hr : 0 < r)
    (hs : 0 ≤ s) :
    ContinuousOn (fun u => (s * u - r) / shellDist r s u)
      (Icc (-1 : ℝ) 1) := by
  intro u hu
  rcases eq_or_ne (shellDist r s u) 0 with hd0 | hdne
  · -- collision: `s = r` and `u = 1`
    have hrs : 0 ≤ 2 * r * s * (1 - u) :=
      mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) hr.le) hs)
        (sub_nonneg.mpr hu.2)
    have harg : 0 ≤ r ^ 2 + s ^ 2 - 2 * r * s * u := by
      nlinarith [sq_nonneg (r - s)]
    have hq0 : r ^ 2 + s ^ 2 - 2 * r * s * u = 0 :=
      (Real.sqrt_eq_zero harg).mp hd0
    have hsq0 : (r - s) ^ 2 = 0 :=
      le_antisymm (by nlinarith) (sq_nonneg _)
    have hsr : s = r := by
      have h3 := (pow_eq_zero_iff two_ne_zero).mp hsq0
      linarith [h3]
    subst hsr
    have hfac : 2 * s ^ 2 * (1 - u) = 0 := by linear_combination hq0
    have hu1 : u = 1 := by
      rcases mul_eq_zero.mp hfac with h | h
      · exfalso
        nlinarith [mul_pos hr hr]
      · linarith
    -- squeeze `|X/d| ≤ d/(2s)` along the interval
    have hbd : ∀ᶠ v in 𝓝[Icc (-1 : ℝ) 1] u,
        ‖(s * v - s) / shellDist s s v‖ ≤ shellDist s s v / (2 * s) := by
      filter_upwards [self_mem_nhdsWithin] with v hv
      rw [Real.norm_eq_abs]
      exact abs_axialDiv_le_dist_div_collision hr hv.2
    have hdlim : Tendsto (fun v => shellDist s s v / (2 * s))
        (𝓝[Icc (-1 : ℝ) 1] u) (𝓝 0) := by
      have hc : ContinuousWithinAt
          (fun v => shellDist s s v / (2 * s)) (Icc (-1 : ℝ) 1) u :=
        ((continuous_shellDist_u s s).div_const (2 * s)).continuousWithinAt
      have hval : shellDist s s u / (2 * s) = 0 := by
        rw [hd0, zero_div]
      rw [← hval]
      exact hc
    have hlim : Tendsto (fun v => (s * v - s) / shellDist s s v)
        (𝓝[Icc (-1 : ℝ) 1] u) (𝓝 0) :=
      squeeze_zero_norm' hbd hdlim
    have hval0 : (s * u - s) / shellDist s s u = 0 := by
      rw [hu1, mul_one, sub_self, zero_div]
    change Tendsto (fun v => (s * v - s) / shellDist s s v)
      (𝓝[Icc (-1 : ℝ) 1] u) (𝓝 ((s * u - s) / shellDist s s u))
    rw [hval0]
    exact hlim
  · exact (((continuous_const.mul continuous_id).sub
      continuous_const).continuousAt.div
      (continuous_shellDist_u r s).continuousAt hdne).continuousWithinAt

private lemma continuousOn_shellZdPayloadN {τ r s : ℝ} (hr : 0 < r)
    (hs : 0 ≤ s) :
    ContinuousOn (fun u => Real.exp (-(1 / τ) * shellDist r s u) *
      ((s * u - r) / shellDist r s u)) (Icc (-1 : ℝ) 1) :=
  (continuous_shellKernelN' τ r s).continuousOn.mul
    (continuousOn_shellAxialDivN hr hs)

/-! ## The sphere and mixture bridges -/

/-- The Haar-shell integral of the normalized axial payload is exactly the
zonal `shellZdN` profile. -/
lemma integral_uniformSphere_laplaceAxialDiv_eq_shellZdN
    {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ} (hr : 0 < r) (hs : 0 ≤ s) :
    (∫ ω, laplaceKernel τ (radialRayProbeN n (by omega) r)
          (s • (ω : EuclideanSpace ℝ (Fin n))) *
        (((s • (ω : EuclideanSpace ℝ (Fin n))) ⟨0, by omega⟩ - r) /
          ‖radialRayProbeN n (by omega) r -
            s • (ω : EuclideanSpace ℝ (Fin n))‖)
        ∂(uniformSphereMeasure n)) = shellZdN n τ r s := by
  have hpoint : ∀ ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1,
      laplaceKernel τ (radialRayProbeN n (by omega) r)
          (s • (ω : EuclideanSpace ℝ (Fin n))) *
        (((s • (ω : EuclideanSpace ℝ (Fin n))) ⟨0, by omega⟩ - r) /
          ‖radialRayProbeN n (by omega) r -
            s • (ω : EuclideanSpace ℝ (Fin n))‖) =
      Real.exp (-(1 / τ) * shellDist r s
          (sphereFirstCoord n (by omega) ω)) *
        ((s * sphereFirstCoord n (by omega) ω - r) /
          shellDist r s (sphereFirstCoord n (by omega) ω)) := by
    intro ω
    unfold laplaceKernel
    rw [norm_radialRayProbeN_sub_smul_sphere (by omega)]
    simp [PiLp.smul_apply, smul_eq_mul, sphereFirstCoord]
  calc
    (∫ ω, laplaceKernel τ (radialRayProbeN n (by omega) r)
          (s • (ω : EuclideanSpace ℝ (Fin n))) *
        (((s • (ω : EuclideanSpace ℝ (Fin n))) ⟨0, by omega⟩ - r) /
          ‖radialRayProbeN n (by omega) r -
            s • (ω : EuclideanSpace ℝ (Fin n))‖)
        ∂(uniformSphereMeasure n)) =
      ∫ ω, Real.exp (-(1 / τ) * shellDist r s
          (sphereFirstCoord n (by omega) ω)) *
        ((s * sphereFirstCoord n (by omega) ω - r) /
          shellDist r s (sphereFirstCoord n (by omega) ω))
        ∂(uniformSphereMeasure n) :=
      integral_congr_ae (Filter.Eventually.of_forall hpoint)
    _ = (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            ((s * u - r) / shellDist r s u)) := by
      exact integral_uniformSphere_zonalOn_of_bridge
        (zonalSphereBridge_standard n hn) _
        (continuousOn_shellZdPayloadN hr hs)
    _ = shellZdN n τ r s := rfl

/-- The y-level soft-sign ray profile is the `ν`-mixture of the per-shell
`shellZdN` profiles. -/
theorem radialRayZdN_eq_integral_shellZdN
    {n : ℕ} (hn : 3 ≤ n) {τ : ℝ} (hτ : 0 < τ) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] (hsupp : ν (Iio 0) = 0) {r : ℝ} (hr : 0 < r) :
    radialRayZdN n (by omega) τ ν r = ∫ s, shellZdN n τ r s ∂ν := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  have hint : Integrable
      (fun y => laplaceKernel τ (radialRayProbeN n (by omega) r) y *
        ((y ⟨0, by omega⟩ - r) / ‖radialRayProbeN n (by omega) r - y‖))
      (radialMixtureN n ν) := by
    refine ⟨?_, HasFiniteIntegral.of_bounded (C := 1)
      (ae_of_all _ fun y => ?_)⟩
    · have hcoord : Measurable
          (fun y : EuclideanSpace ℝ (Fin n) => y ⟨0, by omega⟩ - r) := by
        fun_prop
      have hnorm : Measurable
          (fun y : EuclideanSpace ℝ (Fin n) =>
            ‖radialRayProbeN n (by omega) r - y‖) := by
        fun_prop
      exact (((continuous_laplaceKernel_radialRayProbeN (by omega)
        τ r).measurable).mul (hcoord.div hnorm)).aestronglyMeasurable
    · rw [Real.norm_eq_abs, abs_mul,
        abs_of_nonneg (laplaceKernel_radialRayProbeN_nonneg (by omega) τ r y)]
      calc laplaceKernel τ (radialRayProbeN n (by omega) r) y *
            |(y ⟨0, by omega⟩ - r) /
              ‖radialRayProbeN n (by omega) r - y‖|
          ≤ 1 * 1 :=
            mul_le_mul
              (laplaceKernel_radialRayProbeN_le_one (by omega) τ hτ r y)
              (abs_first_div_norm_radialRayProbeN_le_one (by omega) r y)
              (abs_nonneg _) zero_le_one
        _ = 1 := mul_one _
  rw [radialRayZdN, integral_radialMixtureN hn ν hint]
  refine integral_congr_ae ?_
  have hae : ∀ᵐ s ∂ν, (0 : ℝ) ≤ s := by
    rw [ae_iff]
    have hset : {s : ℝ | ¬ (0 : ℝ) ≤ s} = Iio 0 := by
      ext s
      simp [not_le]
    rw [hset]
    exact hsupp
  filter_upwards [hae] with s hs
  exact integral_uniformSphere_laplaceAxialDiv_eq_shellZdN hn hr hs

/-! ## Dirac-profile collapse -/

lemma dirac_Iio_zero {s : ℝ} (hs : 0 ≤ s) :
    (Measure.dirac s) (Iio (0 : ℝ)) = 0 := by
  rw [Measure.dirac_apply' s measurableSet_Iio]
  exact Set.indicator_of_notMem (by simpa using hs) _

lemma radialRayZN_dirac (n : ℕ) (τ r s : ℝ) :
    radialRayZN n τ (Measure.dirac s) r = shellZN n τ r s := by
  rw [radialRayZN, integral_dirac]

lemma radialRayDN_dirac (n : ℕ) (τ r s : ℝ) :
    radialRayDN n τ (Measure.dirac s) r = shellDN n τ r s := by
  rw [radialRayDN, integral_dirac]

lemma radialRayQN_dirac (n : ℕ) (τ r s : ℝ) :
    radialRayQN n τ (Measure.dirac s) r = shellQN n τ r s := by
  rw [radialRayQN, integral_dirac]

lemma radialRayZdN_dirac {n : ℕ} (hn : 3 ≤ n) {τ : ℝ} (hτ : 0 < τ)
    {r s : ℝ} (hr : 0 < r) (hs : 0 ≤ s) :
    radialRayZdN n (by omega) τ (Measure.dirac s) r = shellZdN n τ r s := by
  rw [radialRayZdN_eq_integral_shellZdN hn hτ (Measure.dirac s)
    (dirac_Iio_zero hs) hr, integral_dirac]

/-- Per-shell normalizer positivity, through the Dirac collapse. -/
lemma shellZN_pos {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (hτ : 0 < τ) (r s : ℝ) :
    0 < shellZN n τ r s := by
  have h := radialRayZN_pos hn τ hτ (Measure.dirac s) r
  rwa [radialRayZN_dirac] at h

/-- Near shells have axial mean below the probe: `D̄(s) ≤ r·Z̄(s)` whenever
`s ≤ 2r`. -/
lemma shellDN_le_r_mul_shellZN {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ}
    (hs : 0 ≤ s) (hs2 : s ≤ 2 * r) :
    shellDN n τ r s ≤ r * shellZN n τ r s := by
  have hker : IntegrableOn
      (fun u => zonalWeight n u *
        (Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r)))
      (Ioc (-1 : ℝ) 1) := by
    refine (Continuous.integrableOn_Icc ?_).mono_set Ioc_subset_Icc_self
    exact (continuous_zonalWeight hn).mul
      ((continuous_shellKernelN' τ r s).mul
        ((continuous_const.mul continuous_id).sub continuous_const))
  have hker2 : IntegrableOn
      (fun u => zonalWeight n u *
        (Real.exp (-(1 / τ) * shellDist r s u) * r))
      (Ioc (-1 : ℝ) 1) := by
    refine (Continuous.integrableOn_Icc ?_).mono_set Ioc_subset_Icc_self
    exact (continuous_zonalWeight hn).mul
      ((continuous_shellKernelN' τ r s).mul continuous_const)
  have hmono : (∫ u in Ioc (-1 : ℝ) 1, zonalWeight n u *
        (Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r)))
      ≤ ∫ u in Ioc (-1 : ℝ) 1, zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) * r) := by
    refine setIntegral_mono_on hker hker2 measurableSet_Ioc fun u hu => ?_
    have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
    have hW : 0 ≤ zonalWeight n u := zonalWeight_nonneg hu2
    have hK : 0 ≤ Real.exp (-(1 / τ) * shellDist r s u) :=
      (Real.exp_pos _).le
    have hX : s * u - r ≤ r := by nlinarith [hu.2, hu.1]
    exact mul_le_mul_of_nonneg_left
      (mul_le_mul_of_nonneg_left hX hK) hW
  have hconst : (∫ u in Ioc (-1 : ℝ) 1, zonalWeight n u *
        (Real.exp (-(1 / τ) * shellDist r s u) * r))
      = r * ∫ u in Ioc (-1 : ℝ) 1, zonalWeight n u *
          Real.exp (-(1 / τ) * shellDist r s u) := by
    rw [← integral_const_mul]
    exact setIntegral_congr_fun measurableSet_Ioc fun u _ => by ring
  have hmass : (0 : ℝ) ≤ (zonalMass n)⁻¹ :=
    inv_nonneg.mpr (zonalMass_pos hn).le
  rw [shellDN, shellZN]
  calc (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1, zonalWeight n u *
        (Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r))
      ≤ (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1, zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) * r) :=
        mul_le_mul_of_nonneg_left hmono hmass
    _ = r * ((zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1, zonalWeight n u *
          Real.exp (-(1 / τ) * shellDist r s u)) := by
        rw [hconst]
        ring

/-! ## The near-shell per-shell covariance floor -/

/-- **Per-shell RSI, near branch (`s < r`)**: instantiating the PROVED
mixture sign estimate at the Dirac profile `ν = δ_s` gives the per-shell
covariance floor for every shell strictly inside the probe radius. -/
theorem shellRSIN_near {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ}
    (hτ : 0 < τ) (hr : 0 < r) (hs : 0 ≤ s) (hlt : s < r) :
    -(((n : ℝ) - 1) * τ) * shellZN n τ r s ^ 2 ≤
      shellQN n τ r s * shellZN n τ r s -
        shellDN n τ r s * shellZdN n τ r s := by
  have hsupp : (Measure.dirac s) (Iio (0 : ℝ)) = 0 := dirac_Iio_zero hs
  have hZpos : 0 < shellZN n τ r s := shellZN_pos hn τ hτ r s
  have hm : radialRayMN n τ (Measure.dirac s) r ≤ r := by
    rw [radialRayMN, radialRayZN_dirac, radialRayDN_dirac,
      div_le_iff₀ hZpos]
    exact shellDN_le_r_mul_shellZN hn hs (by linarith)
  have hder := radialRayMDerivN_ge_of_le hn τ hτ (Measure.dirac s) hsupp hr hm
  have hcov := radialRayMDerivN_cov hn τ hτ (Measure.dirac s) r
  rw [radialRayZN_dirac, radialRayDN_dirac, radialRayQN_dirac,
    radialRayZdN_dirac hn hτ hr hs] at hcov
  have hlow : -(((n : ℝ) - 1) * τ) * shellZN n τ r s ^ 2 ≤
      τ * (radialRayMDerivN n (by omega) τ (Measure.dirac s) r + 1) *
        shellZN n τ r s ^ 2 := by
    have hZsq : (0 : ℝ) ≤ shellZN n τ r s ^ 2 := sq_nonneg _
    nlinarith [hder, mul_nonneg hτ.le hZsq]
  rw [hcov] at hlow
  exact hlow

/-! ## The measure-free association predicate and the pair bracket -/

/-- **Between-shell association** (the single remaining G2 lemma;
numerically verified across `n ∈ {3,4,5,10}` and `r/τ ∈ [0.02, 50]` in
`numerics/rn_g2.py`): the per-shell tilted axial mean `m(s) = D̄/Z̄` and the
per-shell tilted soft-sign mean `z(s) = Z̄d/Z̄` are comonotone in the shell
radius.  Stated with cleared denominators; the profile measure `ν` does not
appear — this is a property of the kernel geometry alone. -/
def ZonalShellAssociation (n : ℕ) (τ : ℝ) : Prop :=
  ∀ r : ℝ, 0 < r → ∀ s s' : ℝ, 0 ≤ s → 0 ≤ s' →
    0 ≤ (shellDN n τ r s * shellZN n τ r s' -
          shellDN n τ r s' * shellZN n τ r s) *
        (shellZdN n τ r s * shellZN n τ r s' -
          shellZdN n τ r s' * shellZN n τ r s)

/-- The pointwise two-shell bracket: per-shell floors at `s` and `s'` plus
the association inequality force the symmetrized cross bracket, after
cancelling one factor of `Z̄Z̄' > 0`. -/
lemma shell_pair_bracket {c : ℝ} {Z Z' Q Q' D D' W W' : ℝ}
    (hZ : 0 < Z) (hZ' : 0 < Z')
    (h1 : -c * Z ^ 2 ≤ Q * Z - D * W)
    (h2 : -c * Z' ^ 2 ≤ Q' * Z' - D' * W')
    (hassoc : 0 ≤ (D * Z' - D' * Z) * (W * Z' - W' * Z)) :
    -(2 * c) * (Z * Z') ≤ (Q * Z' + Q' * Z) - (D * W' + D' * W) := by
  have hA := mul_le_mul_of_nonneg_left h1 (sq_nonneg Z')
  have hB := mul_le_mul_of_nonneg_left h2 (sq_nonneg Z)
  have hkey : Z * Z' * ((Q * Z' + Q' * Z) - (D * W' + D' * W))
      = Z' ^ 2 * (Q * Z - D * W) + Z ^ 2 * (Q' * Z' - D' * W')
        + (D * Z' - D' * Z) * (W * Z' - W' * Z) := by ring
  have hprod : (Z * Z') * (-(2 * c) * (Z * Z'))
      = Z' ^ 2 * (-c * Z ^ 2) + Z ^ 2 * (-c * Z' ^ 2) := by ring
  have hsum : (Z * Z') * (-(2 * c) * (Z * Z'))
      ≤ (Z * Z') * ((Q * Z' + Q' * Z) - (D * W' + D' * W)) := by
    rw [hprod, hkey]
    linarith [hA, hB, hassoc]
  exact le_of_mul_le_mul_left hsum (mul_pos hZ hZ')

/-! ## The far-shell branch: comonotonicity in `u` -/

lemma abs_axialDiv_le_one {r s u : ℝ} (hu : u ^ 2 ≤ 1) :
    |(s * u - r) / shellDist r s u| ≤ 1 := by
  rcases eq_or_lt_of_le (shellDist_nonneg r s u) with h0 | hpos
  · rw [← h0, div_zero, abs_zero]
    exact zero_le_one
  · rw [abs_div, abs_of_pos hpos, div_le_one hpos]
    have hd2 : shellDist r s u ^ 2 = shellAxial r s u ^ 2 + shellRhoSq s u :=
      shellDist_sq_eq_axial_add_rho hu r s
    have hax : shellAxial r s u = s * u - r := rfl
    rw [hax] at hd2
    have h1 : (s * u - r) ^ 2 ≤ shellDist r s u ^ 2 := by
      nlinarith [shellRhoSq_nonneg hu s]
    calc |s * u - r| = Real.sqrt ((s * u - r) ^ 2) :=
          (Real.sqrt_sq_eq_abs _).symm
      _ ≤ Real.sqrt (shellDist r s u ^ 2) := Real.sqrt_le_sqrt h1
      _ = shellDist r s u := Real.sqrt_sq hpos.le

private lemma shellDist_sq_pos_far {r s u : ℝ} (hr : 0 < r) (hge : r ≤ s)
    (hu : u < 1) : 0 < r ^ 2 + s ^ 2 - 2 * r * s * u := by
  have hs : 0 < s := lt_of_lt_of_le hr hge
  nlinarith [sq_nonneg (r - s), mul_pos (mul_pos hr hs) (sub_pos.mpr hu)]

private lemma hasDerivAt_axialDiv {r s u : ℝ} (hr : 0 < r) (hge : r ≤ s)
    (hu2 : u < 1) :
    HasDerivAt (fun v => (s * v - r) / shellDist r s v)
      ((s * shellDist r s u - (s * u - r) * -(r * s / shellDist r s u)) /
        shellDist r s u ^ 2) u := by
  have hq := shellDist_sq_pos_far hr hge hu2
  have hdne : shellDist r s u ≠ 0 := (Real.sqrt_pos.mpr hq).ne'
  have hX : HasDerivAt (fun v : ℝ => s * v - r) s u := by
    simpa using ((hasDerivAt_id u).const_mul s).sub_const r
  exact hX.div (hasDerivAt_shellDist_u hq) hdne

private lemma axialDiv_deriv_nonneg {r s u : ℝ} (hr : 0 < r) (hge : r ≤ s)
    (hu2 : u ≤ 1) (hq : 0 < r ^ 2 + s ^ 2 - 2 * r * s * u) :
    0 ≤ (s * shellDist r s u - (s * u - r) * -(r * s / shellDist r s u)) /
      shellDist r s u ^ 2 := by
  have hs : 0 < s := lt_of_lt_of_le hr hge
  have hdpos : 0 < shellDist r s u := Real.sqrt_pos.mpr hq
  have hde : shellDist r s u
      = Real.sqrt (r ^ 2 + s ^ 2 - 2 * r * s * u) := rfl
  have hd2 : shellDist r s u ^ 2 = r ^ 2 + s ^ 2 - 2 * r * s * u := by
    rw [hde]
    exact Real.sq_sqrt hq.le
  refine div_nonneg ?_ (sq_nonneg _)
  have hnum : s * shellDist r s u - (s * u - r) * -(r * s / shellDist r s u)
      = (s * shellDist r s u ^ 2 + (s * u - r) * (r * s))
          / shellDist r s u := by
    field_simp
    ring
  rw [hnum]
  refine div_nonneg ?_ hdpos.le
  rw [hd2]
  have hru : r * u ≤ s := by nlinarith
  have hkey : s * (r ^ 2 + s ^ 2 - 2 * r * s * u) + (s * u - r) * (r * s)
      = s ^ 2 * (s - r * u) := by ring
  have h2 : 0 ≤ s ^ 2 * (s - r * u) :=
    mul_nonneg (sq_nonneg s) (sub_nonneg.mpr hru)
  linarith [hkey, h2]

private lemma monotoneOn_axialDiv {r s : ℝ} (hr : 0 < r) (hge : r ≤ s) :
    MonotoneOn (fun u => (s * u - r) / shellDist r s u) (Icc (-1 : ℝ) 1) := by
  have hs0 : 0 ≤ s := (lt_of_lt_of_le hr hge).le
  refine monotoneOn_of_deriv_nonneg (convex_Icc _ _)
    (continuousOn_shellAxialDivN hr hs0) ?_ ?_
  · rw [interior_Icc]
    intro u hu
    exact (hasDerivAt_axialDiv hr hge
      hu.2).differentiableAt.differentiableWithinAt
  · rw [interior_Icc]
    intro u hu
    rw [(hasDerivAt_axialDiv hr hge hu.2).deriv]
    exact axialDiv_deriv_nonneg hr hge hu.2.le
      (shellDist_sq_pos_far hr hge hu.2)

/-! ## Weighted Chebyshev doubling -/

/-- **Weighted Chebyshev/FKG doubling**: on a full-measure carrier where the
weight is nonnegative and `f, g` are comonotone, the weighted correlation is
nonnegative.  Proved by expanding the doubled bracket
`F(x)F(y)(f(x)-f(y))(g(x)-g(y)) ≥ 0` over the product measure. -/
private lemma weighted_chebyshev {μ : Measure ℝ} [SFinite μ]
    {F f g : ℝ → ℝ} {A : Set ℝ} (hAfull : μ Aᶜ = 0)
    (hFnn : ∀ x ∈ A, 0 ≤ F x)
    (hmono : ∀ x ∈ A, ∀ y ∈ A, x ≤ y → f x ≤ f y ∧ g x ≤ g y)
    (hiF : Integrable F μ) (hiFf : Integrable (fun x => F x * f x) μ)
    (hiFg : Integrable (fun x => F x * g x) μ)
    (hiFfg : Integrable (fun x => F x * (f x * g x)) μ) :
    (∫ x, F x * f x ∂μ) * ∫ x, F x * g x ∂μ ≤
      (∫ x, F x * (f x * g x) ∂μ) * ∫ x, F x ∂μ := by
  have hpair : ∀ᵐ z ∂(μ.prod μ), z.1 ∈ A ∧ z.2 ∈ A := by
    rw [ae_iff]
    have hsub : {z : ℝ × ℝ | ¬(z.1 ∈ A ∧ z.2 ∈ A)}
        ⊆ (Aᶜ ×ˢ (univ : Set ℝ)) ∪ ((univ : Set ℝ) ×ˢ Aᶜ) := by
      intro z hz
      rw [mem_setOf_eq, not_and] at hz
      by_cases h1 : z.1 ∈ A
      · exact Or.inr ⟨mem_univ _, hz h1⟩
      · exact Or.inl ⟨h1, mem_univ _⟩
    refine measure_mono_null hsub (measure_union_null ?_ ?_)
    · rw [Measure.prod_prod, hAfull, zero_mul]
    · rw [Measure.prod_prod, hAfull, mul_zero]
  have hprod_nonneg : 0 ≤ ∫ z : ℝ × ℝ,
      (F z.1 * F z.2) * ((f z.1 - f z.2) * (g z.1 - g z.2)) ∂(μ.prod μ) := by
    refine integral_nonneg_of_ae ?_
    filter_upwards [hpair] with z hz
    obtain ⟨h1, h2⟩ := hz
    have hF1 := hFnn z.1 h1
    have hF2 := hFnn z.2 h2
    have hbr : 0 ≤ (f z.1 - f z.2) * (g z.1 - g z.2) := by
      rcases le_total z.1 z.2 with h | h
      · obtain ⟨hf, hg⟩ := hmono z.1 h1 z.2 h2 h
        have hprod := mul_nonneg (neg_nonneg.mpr (by linarith : f z.1 - f z.2 ≤ 0))
          (neg_nonneg.mpr (by linarith : g z.1 - g z.2 ≤ 0))
        rwa [neg_mul_neg] at hprod
      · obtain ⟨hf, hg⟩ := hmono z.2 h2 z.1 h1 h
        exact mul_nonneg (by linarith) (by linarith)
    exact mul_nonneg (mul_nonneg hF1 hF2) hbr
  have hiT1 : Integrable
      (fun z : ℝ × ℝ => (F z.1 * (f z.1 * g z.1)) * F z.2) (μ.prod μ) :=
    hiFfg.mul_prod hiF
  have hiT2 : Integrable
      (fun z : ℝ × ℝ => F z.1 * (F z.2 * (f z.2 * g z.2))) (μ.prod μ) :=
    hiF.mul_prod hiFfg
  have hiT3 : Integrable
      (fun z : ℝ × ℝ => (F z.1 * f z.1) * (F z.2 * g z.2)) (μ.prod μ) :=
    hiFf.mul_prod hiFg
  have hiT4 : Integrable
      (fun z : ℝ × ℝ => (F z.1 * g z.1) * (F z.2 * f z.2)) (μ.prod μ) :=
    hiFg.mul_prod hiFf
  have hexpand : (fun z : ℝ × ℝ =>
        (F z.1 * F z.2) * ((f z.1 - f z.2) * (g z.1 - g z.2)))
      = fun z : ℝ × ℝ =>
          ((F z.1 * (f z.1 * g z.1)) * F z.2
            + F z.1 * (F z.2 * (f z.2 * g z.2)))
          - ((F z.1 * f z.1) * (F z.2 * g z.2)
            + (F z.1 * g z.1) * (F z.2 * f z.2)) := by
    funext z
    ring
  rw [hexpand] at hprod_nonneg
  have hT1 : (∫ z : ℝ × ℝ, (F z.1 * (f z.1 * g z.1)) * F z.2 ∂(μ.prod μ))
      = (∫ x, F x * (f x * g x) ∂μ) * ∫ x, F x ∂μ :=
    integral_prod_mul (fun x => F x * (f x * g x)) (fun y => F y)
  have hT2 : (∫ z : ℝ × ℝ, F z.1 * (F z.2 * (f z.2 * g z.2)) ∂(μ.prod μ))
      = (∫ x, F x ∂μ) * ∫ x, F x * (f x * g x) ∂μ :=
    integral_prod_mul (fun x => F x) (fun y => F y * (f y * g y))
  have hT3 : (∫ z : ℝ × ℝ, (F z.1 * f z.1) * (F z.2 * g z.2) ∂(μ.prod μ))
      = (∫ x, F x * f x ∂μ) * ∫ x, F x * g x ∂μ :=
    integral_prod_mul (fun x => F x * f x) (fun y => F y * g y)
  have hT4 : (∫ z : ℝ × ℝ, (F z.1 * g z.1) * (F z.2 * f z.2) ∂(μ.prod μ))
      = (∫ x, F x * g x ∂μ) * ∫ x, F x * f x ∂μ :=
    integral_prod_mul (fun x => F x * g x) (fun y => F y * f y)
  have hbig : (∫ z : ℝ × ℝ,
        F z.1 * (f z.1 * g z.1) * F z.2 + F z.1 * (F z.2 * (f z.2 * g z.2))
          - (F z.1 * f z.1 * (F z.2 * g z.2)
            + F z.1 * g z.1 * (F z.2 * f z.2)) ∂(μ.prod μ))
      = (∫ z : ℝ × ℝ,
          F z.1 * (f z.1 * g z.1) * F z.2
            + F z.1 * (F z.2 * (f z.2 * g z.2)) ∂(μ.prod μ))
        - (∫ z : ℝ × ℝ,
          F z.1 * f z.1 * (F z.2 * g z.2)
            + F z.1 * g z.1 * (F z.2 * f z.2) ∂(μ.prod μ)) :=
    integral_sub (hiT1.add hiT2) (hiT3.add hiT4)
  have hadd1 : (∫ z : ℝ × ℝ,
        F z.1 * (f z.1 * g z.1) * F z.2
          + F z.1 * (F z.2 * (f z.2 * g z.2)) ∂(μ.prod μ))
      = (∫ z : ℝ × ℝ, F z.1 * (f z.1 * g z.1) * F z.2 ∂(μ.prod μ))
        + ∫ z : ℝ × ℝ, F z.1 * (F z.2 * (f z.2 * g z.2)) ∂(μ.prod μ) :=
    integral_add hiT1 hiT2
  have hadd2 : (∫ z : ℝ × ℝ,
        F z.1 * f z.1 * (F z.2 * g z.2)
          + F z.1 * g z.1 * (F z.2 * f z.2) ∂(μ.prod μ))
      = (∫ z : ℝ × ℝ, F z.1 * f z.1 * (F z.2 * g z.2) ∂(μ.prod μ))
        + ∫ z : ℝ × ℝ, F z.1 * g z.1 * (F z.2 * f z.2) ∂(μ.prod μ) :=
    integral_add hiT3 hiT4
  rw [hbig, hadd1, hadd2, hT1, hT2, hT3, hT4] at hprod_nonneg
  have e1 : (∫ x, F x ∂μ) * ∫ x, F x * (f x * g x) ∂μ
      = (∫ x, F x * (f x * g x) ∂μ) * ∫ x, F x ∂μ := mul_comm _ _
  have e2 : (∫ x, F x * g x ∂μ) * ∫ x, F x * f x ∂μ
      = (∫ x, F x * f x ∂μ) * ∫ x, F x * g x ∂μ := mul_comm _ _
  linarith [hprod_nonneg, e1, e2]

/-- **Per-shell RSI, far branch (`s ≥ r`)**: on far shells `X` and `X/d` are
comonotone in `u`, so the per-shell covariance is nonnegative outright. -/
theorem shellRSIN_far {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ}
    (hτ : 0 < τ) (hr : 0 < r) (hge : r ≤ s) :
    0 ≤ shellQN n τ r s * shellZN n τ r s -
      shellDN n τ r s * shellZdN n τ r s := by
  have hs0 : 0 ≤ s := (lt_of_lt_of_le hr hge).le
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-1 : ℝ) 1)) := by
    constructor
    rw [Measure.restrict_apply_univ, Real.volume_Ioc]
    exact ENNReal.ofReal_lt_top
  -- integrable families over the zonal base
  have hiF : Integrable
      (fun u => zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u))
      (volume.restrict (Ioc (-1 : ℝ) 1)) :=
    ((continuous_zonalWeight hn).mul
      (continuous_shellKernelN' τ r s)).integrableOn_Icc.mono_set
      Ioc_subset_Icc_self
  have hiFf : Integrable
      (fun u => (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u))
        * (s * u - r)) (volume.restrict (Ioc (-1 : ℝ) 1)) :=
    (((continuous_zonalWeight hn).mul
      (continuous_shellKernelN' τ r s)).mul
      ((continuous_const.mul continuous_id).sub
        continuous_const)).integrableOn_Icc.mono_set Ioc_subset_Icc_self
  have hmeasG : Measurable fun u => (s * u - r) / shellDist r s u :=
    (((continuous_const.mul continuous_id).sub
      continuous_const).measurable).div
      (continuous_shellDist_u r s).measurable
  have hiFg : Integrable
      (fun u => (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u))
        * ((s * u - r) / shellDist r s u))
      (volume.restrict (Ioc (-1 : ℝ) 1)) := by
    refine ⟨?_, HasFiniteIntegral.of_bounded (C := 1) ?_⟩
    · exact (((continuous_zonalWeight hn).mul
        (continuous_shellKernelN' τ r s)).measurable.mul
        hmeasG).aestronglyMeasurable
    · filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
      have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
      rw [Real.norm_eq_abs, abs_mul, abs_mul]
      have hW : |zonalWeight n u| ≤ 1 := by
        rw [abs_of_nonneg (zonalWeight_nonneg hu2)]
        exact zonalWeight_le_one hn hu2
      have hK : |Real.exp (-(1 / τ) * shellDist r s u)| ≤ 1 := by
        rw [abs_of_pos (Real.exp_pos _)]
        exact exp_shellDist_le_one hτ r s u
      calc |zonalWeight n u| * |Real.exp (-(1 / τ) * shellDist r s u)|
            * |(s * u - r) / shellDist r s u|
          ≤ 1 * 1 * 1 :=
            mul_le_mul (mul_le_mul hW hK (abs_nonneg _) zero_le_one)
              (abs_axialDiv_le_one hu2) (abs_nonneg _) (by norm_num)
        _ = 1 := by norm_num
  have hiFfg : Integrable
      (fun u => (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u))
        * ((s * u - r) * ((s * u - r) / shellDist r s u)))
      (volume.restrict (Ioc (-1 : ℝ) 1)) := by
    refine ⟨?_, HasFiniteIntegral.of_bounded (C := s + r) ?_⟩
    · exact (((continuous_zonalWeight hn).mul
        (continuous_shellKernelN' τ r s)).measurable.mul
        ((((continuous_const.mul continuous_id).sub
          continuous_const).measurable).mul hmeasG)).aestronglyMeasurable
    · filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
      have hu2 : u ^ 2 ≤ 1 := sq_le_one_of_mem_Ioc hu
      rw [Real.norm_eq_abs, abs_mul, abs_mul]
      have hW : |zonalWeight n u| ≤ 1 := by
        rw [abs_of_nonneg (zonalWeight_nonneg hu2)]
        exact zonalWeight_le_one hn hu2
      have hK : |Real.exp (-(1 / τ) * shellDist r s u)| ≤ 1 := by
        rw [abs_of_pos (Real.exp_pos _)]
        exact exp_shellDist_le_one hτ r s u
      have hu1 : |u| ≤ 1 := abs_le.mpr ⟨hu.1.le, hu.2⟩
      have hX : |s * u - r| ≤ s + r := by
        calc |s * u - r| ≤ |s * u| + |r| := abs_sub _ _
          _ = s * |u| + r := by
            rw [abs_mul, abs_of_nonneg hs0, abs_of_pos hr]
          _ ≤ s * 1 + r := by
            have := mul_le_mul_of_nonneg_left hu1 hs0
            linarith
          _ = s + r := by ring
      calc |zonalWeight n u| * |Real.exp (-(1 / τ) * shellDist r s u)|
            * |(s * u - r) * ((s * u - r) / shellDist r s u)|
          ≤ 1 * 1 * ((s + r) * 1) := by
            refine mul_le_mul (mul_le_mul hW hK (abs_nonneg _) zero_le_one)
              ?_ (abs_nonneg _) (by norm_num)
            rw [abs_mul]
            exact mul_le_mul hX (abs_axialDiv_le_one hu2) (abs_nonneg _)
              (by linarith)
        _ = s + r := by ring
  -- the carrier `Icc (-1) 1` is full for the restricted measure
  have hAfull : (volume.restrict (Ioc (-1 : ℝ) 1)) (Icc (-1 : ℝ) 1)ᶜ = 0 := by
    rw [Measure.restrict_apply measurableSet_Icc.compl]
    have hsub : (Icc (-1 : ℝ) 1)ᶜ ∩ Ioc (-1 : ℝ) 1 ⊆ ∅ :=
      fun x hx => hx.1 (Ioc_subset_Icc_self hx.2)
    rw [Set.subset_empty_iff.mp hsub, measure_empty]
  have hFnn : ∀ u ∈ Icc (-1 : ℝ) 1,
      0 ≤ zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u) := by
    intro u hu
    exact mul_nonneg (zonalWeight_nonneg (by nlinarith [hu.1, hu.2]))
      (Real.exp_pos _).le
  have hmono : ∀ u ∈ Icc (-1 : ℝ) 1, ∀ v ∈ Icc (-1 : ℝ) 1, u ≤ v →
      (s * u - r ≤ s * v - r) ∧
      ((s * u - r) / shellDist r s u ≤ (s * v - r) / shellDist r s v) := by
    intro u hu v hv huv
    refine ⟨by nlinarith [mul_le_mul_of_nonneg_left huv hs0], ?_⟩
    exact monotoneOn_axialDiv hr hge hu hv huv
  -- the doubling inequality
  have hcheb := weighted_chebyshev (μ := volume.restrict (Ioc (-1 : ℝ) 1))
    (F := fun u => zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u))
    (f := fun u => s * u - r)
    (g := fun u => (s * u - r) / shellDist r s u)
    (A := Icc (-1 : ℝ) 1) hAfull hFnn hmono
    hiF hiFf hiFg hiFfg
  -- rewrite shell objects into the `∫F`, `∫F·f`, `∫F·g`, `∫F·(f·g)` shapes
  have hZN : shellZN n τ r s = (zonalMass n)⁻¹ *
      ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u) := rfl
  have hDN : shellDN n τ r s = (zonalMass n)⁻¹ *
      ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u) *
          (s * u - r) := by
    rw [shellDN]
    congr 1
    exact setIntegral_congr_fun measurableSet_Ioc (fun u _ => by ring)
  have hZdN : shellZdN n τ r s = (zonalMass n)⁻¹ *
      ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u) *
          ((s * u - r) / shellDist r s u) := by
    rw [shellZdN]
    congr 1
    exact setIntegral_congr_fun measurableSet_Ioc (fun u _ => by ring)
  have hQN : shellQN n τ r s = (zonalMass n)⁻¹ *
      ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u) *
          ((s * u - r) * ((s * u - r) / shellDist r s u)) := by
    rw [shellQN]
    congr 1
    refine setIntegral_congr_fun measurableSet_Ioc (fun u _ => ?_)
    have hax : shellAxial r s u = s * u - r := rfl
    rw [hax]
    ring
  rw [hZN, hDN, hZdN, hQN]
  have hcinv : 0 ≤ (zonalMass n)⁻¹ := inv_nonneg.mpr (zonalMass_pos hn).le
  have hkey :
      ((zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u) *
            (s * u - r)) *
        ((zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u) *
            ((s * u - r) / shellDist r s u))
      ≤ ((zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u) *
            ((s * u - r) * ((s * u - r) / shellDist r s u))) *
        ((zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)) := by
    have hsq : (0 : ℝ) ≤ (zonalMass n)⁻¹ * (zonalMass n)⁻¹ :=
      mul_nonneg hcinv hcinv
    nlinarith [hcheb, hsq]
  linarith [hkey]

end DriftingIdentifiability
