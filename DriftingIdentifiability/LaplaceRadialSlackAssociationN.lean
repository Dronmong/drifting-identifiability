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

end DriftingIdentifiability
