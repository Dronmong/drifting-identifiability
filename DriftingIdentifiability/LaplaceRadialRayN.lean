import DriftingIdentifiability.LaplaceRadialShellN

/-!
# Radial Laplace converse, milestone G1: the general-`n` ray layer

This file is the first ray-level packaging for the general zonal shell layer.
The shell theorem is already proved in `LaplaceRadialShellN`; here we define
the corresponding profile integrals over a radial law and lift the tangential
identity to those profiles.  The lift is deliberately stated with the exact
integrability hypotheses needed by Bochner integration.  No spherical-chart
or injectivity claim is hidden in these definitions; the measure/chart bridge
is a subsequent G1 layer.
-/

open MeasureTheory Filter Topology Set

namespace DriftingIdentifiability

open Paper

/-! ## Ray profiles -/

/-- The general-`n` ray normalizer profile obtained by mixing the shell profile
against a radial measure `ν`. -/
noncomputable def radialRayZN (n : ℕ) (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellZN n τ r s ∂ν

/-- The general-`n` ray companion-normalizer profile. -/
noncomputable def radialRayCN (n : ℕ) (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellCN n τ r s ∂ν

/-- The general-`n` ray axial displacement numerator. -/
noncomputable def radialRayDN (n : ℕ) (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellDN n τ r s ∂ν

/-- The general-`n` ray axial-coordinate profile. -/
noncomputable def radialRayTN (n : ℕ) (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellTN n τ r s ∂ν

/-- The general-`n` ray tangential `ρ²/d` profile. -/
noncomputable def radialRayRhoSqOverDistN (n : ℕ) (τ : ℝ) (ν : Measure ℝ)
    (r : ℝ) : ℝ :=
  ∫ s, shellRhoSqOverDistN n τ r s ∂ν


/-! ## Elementary shell values at the origin -/

lemma shellTN_zero_right_N (n : ℕ) (τ r : ℝ) :
    shellTN n τ r 0 = 0 := by
  unfold shellTN
  simp

lemma shellRhoSqOverDistN_zero_right (n : ℕ) (τ r : ℝ) :
    shellRhoSqOverDistN n τ r 0 = 0 := by
  unfold shellRhoSqOverDistN shellRhoSq
  simp

/-! ## Support bookkeeping -/

/-- A radial law supported on `[0,∞)` has nonnegative radii almost everywhere. -/
lemma radial_ae_nonneg_N {ν : Measure ℝ} (hsupp : ν (Iio 0) = 0) :
    ∀ᵐ s ∂ν, 0 ≤ s := by
  rw [ae_iff]
  convert hsupp using 2
  ext s
  simp [not_le]

/-! ## The ray-level tangential identity -/

/-- **General-`n` ray tangential identity.**  Mixing the per-shell identity
`P̄ = ((n-1)τ/r) T̄` against a radial law preserves it.  The two explicit
integrability hypotheses are the precise assumptions needed to commute the
constant through the integral and to use integral congruence; they are not
axioms and can later be discharged from bounded radial laws. -/
theorem radialRayRhoSqOverDistN_eq_const_mul_T
    {n : ℕ} (hn : 3 ≤ n) {τ : ℝ} (hτ : 0 < τ)
    (ν : Measure ℝ) (hsupp : ν (Iio 0) = 0) {r : ℝ} (hr : 0 < r)
    :
    radialRayRhoSqOverDistN n τ ν r
      = (((n : ℝ) - 1) * τ / r) * radialRayTN n τ ν r := by
  rw [radialRayRhoSqOverDistN, radialRayTN, ← integral_const_mul]
  refine integral_congr_ae ?_
  filter_upwards [radial_ae_nonneg_N hsupp] with s hs
  rcases eq_or_lt_of_le hs with h0 | hs0
  · rw [← h0, shellRhoSqOverDistN_zero_right, shellTN_zero_right_N, mul_zero]
  · exact shellRhoSqOverDistN_eq_shellTN hn hτ hr hs0.le

/-! ## The axial split, still purely shell algebra -/

lemma shellTN_eq_shellDN_add_rZ
    {n : ℕ} (hn : 3 ≤ n) (τ r s : ℝ) :
    shellTN n τ r s = shellDN n τ r s + r * shellZN n τ r s := by
  unfold shellTN shellDN shellZN
  have hI (f : ℝ → ℝ) (hf : Continuous f) :
      IntegrableOn (fun u => zonalWeight n u * f u) (Ioc (-1 : ℝ) 1) := by
    exact ((continuous_zonalWeight hn).mul hf).integrableOn_Icc.mono_set
      Ioc_subset_Icc_self
  have hK : Continuous fun u : ℝ => Real.exp (-(1 / τ) * shellDist r s u) :=
    Real.continuous_exp.comp ((continuous_shellDist_u r s).const_mul (-(1 / τ)))
  have hsu : Continuous fun u : ℝ => s * u := continuous_const.mul continuous_id
  have hsub : Continuous fun u : ℝ => s * u - r := hsu.sub continuous_const
  have hB : IntegrableOn
      (fun u : ℝ => zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r)))
      (Ioc (-1 : ℝ) 1) := hI (fun u => Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r))
      (hK.mul hsub)
  have hC : IntegrableOn
      (fun u : ℝ => r * (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)))
      (Ioc (-1 : ℝ) 1) := by
    simpa [mul_assoc, mul_left_comm, mul_comm] using
      (hI (fun u => r * Real.exp (-(1 / τ) * shellDist r s u))
        (continuous_const.mul hK))
  have hcongr : ∀ u : ℝ,
      zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u)) =
        (zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u)
          * (s * u - r))) +
          r * (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)) := by
    intro u
    ring
  have hEq : (∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u))) =
      (∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r))) +
        ∫ u in Ioc (-1 : ℝ) 1,
          r * (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)) := by
    calc
      (∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u))) =
          ∫ u in Ioc (-1 : ℝ) 1,
            (zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r))
              + r * (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u))) := by
        exact integral_congr_ae (Filter.Eventually.of_forall hcongr)
      _ = (∫ u in Ioc (-1 : ℝ) 1,
            zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r))) +
          ∫ u in Ioc (-1 : ℝ) 1,
            r * (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)) :=
        integral_add hB hC
  rw [hEq, integral_const_mul]
  ring

/-- Mixing the shell axial split gives the corresponding ray split, under
the integrability needed for each profile. -/
lemma radialRayTN_eq_radialRayDN_add_rZ
    {n : ℕ} (hn : 3 ≤ n) (τ : ℝ) (ν : Measure ℝ) (r : ℝ)
    (hD : Integrable (fun s => shellDN n τ r s) ν)
    (hZ : Integrable (fun s => shellZN n τ r s) ν) :
    radialRayTN n τ ν r = radialRayDN n τ ν r + r * radialRayZN n τ ν r := by
  rw [radialRayTN, radialRayDN, radialRayZN]
  calc
    (∫ s, shellTN n τ r s ∂ν) =
        ∫ s, shellDN n τ r s + r * shellZN n τ r s ∂ν := by
      exact integral_congr_ae (Filter.Eventually.of_forall fun s =>
        shellTN_eq_shellDN_add_rZ hn τ r s)
    _ = (∫ s, shellDN n τ r s ∂ν) +
        ∫ s, r * shellZN n τ r s ∂ν := integral_add hD (hZ.const_mul r)
    _ = (∫ s, shellDN n τ r s ∂ν) +
        r * ∫ s, shellZN n τ r s ∂ν := by rw [integral_const_mul]

/-! ## The `Q` payload and the shell closure -/

/-- The axial square-over-distance shell payload.  As in the `n = 3` layer,
the quotient is interpreted as zero at a collision (`/ 0 = 0`). -/
noncomputable def shellQN (n : ℕ) (τ r s : ℝ) : ℝ :=
  (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
    zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) *
        (shellAxial r s u ^ 2 / shellDist r s u))

/-- The ray axial square-over-distance payload. -/
noncomputable def radialRayQN (n : ℕ) (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellQN n τ r s ∂ν

lemma shellCN_eq_shellQN_add_of_integrable
    {n : ℕ} (hn : 3 ≤ n) {τ r s : ℝ} (hτ : 0 < τ) (hr : 0 < r)
    (hs : 0 ≤ s)
    (hK : IntegrableOn (fun u : ℝ => zonalWeight n u *
      Real.exp (-(1 / τ) * shellDist r s u)) (Ioc (-1 : ℝ) 1))
    (hQ : IntegrableOn (fun u : ℝ => zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) *
        (shellAxial r s u ^ 2 / shellDist r s u))) (Ioc (-1 : ℝ) 1))
    (hP : IntegrableOn (fun u : ℝ => zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) *
        (shellRhoSq s u / shellDist r s u))) (Ioc (-1 : ℝ) 1)) :
    shellCN n τ r s = τ * shellZN n τ r s + shellQN n τ r s
      + shellRhoSqOverDistN n τ r s := by
  have hsq : ∀ u ∈ Ioc (-1 : ℝ) 1,
      shellDist r s u ^ 2 = shellAxial r s u ^ 2 + shellRhoSq s u := by
    intro u hu
    exact shellDist_sq_eq_axial_add_rho (sq_le_one_of_mem_Ioc hu) r s
  have hEq : ∫ u in Ioc (-1 : ℝ) 1,
      zonalWeight n u * ((τ + shellDist r s u) *
        Real.exp (-(1 / τ) * shellDist r s u)) =
      (∫ u in Ioc (-1 : ℝ) 1,
        τ * (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u))) +
      (∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) *
          (shellAxial r s u ^ 2 / shellDist r s u))) +
      ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) *
          (shellRhoSq s u / shellDist r s u)) := by
    have hK' := hK.const_mul τ
    have hsum : IntegrableOn
        (fun u : ℝ => zonalWeight n u *
          (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellAxial r s u ^ 2 / shellDist r s u)) +
          zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) *
            (shellRhoSq s u / shellDist r s u)))
        (Ioc (-1 : ℝ) 1) := hQ.add hP
    calc
      (∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * ((τ + shellDist r s u) *
            Real.exp (-(1 / τ) * shellDist r s u))) =
          ∫ u in Ioc (-1 : ℝ) 1,
            (τ * (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u)) +
              (zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) *
                (shellAxial r s u ^ 2 / shellDist r s u)) +
                zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) *
                  (shellRhoSq s u / shellDist r s u)))) := by
        refine integral_congr_ae ?_
        filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
        by_cases hd : shellDist r s u = 0
        · simp [hd]
          ring
        · have hsq' := hsq u hu
          field_simp [hd]
          linear_combination (zonalWeight n u) * hsq'
      _ = (∫ u in Ioc (-1 : ℝ) 1,
          τ * (zonalWeight n u * Real.exp (-(1 / τ) * shellDist r s u))) +
          ∫ u in Ioc (-1 : ℝ) 1,
            (zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) *
              (shellAxial r s u ^ 2 / shellDist r s u)) +
              zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) *
                (shellRhoSq s u / shellDist r s u))) :=
        integral_add hK' hsum
      _ = _ := by
        rw [integral_add hQ hP]
        ring
  unfold shellCN shellZN shellQN
  rw [hEq, integral_const_mul]
  have hPdef : (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
      zonalWeight n u * (Real.exp (-(1 / τ) * shellDist r s u) *
        (shellRhoSq s u / shellDist r s u)) =
      shellRhoSqOverDistN n τ r s := rfl
  simp only [mul_add]
  rw [hPdef, shellRhoSqOverDistN_eq_shellTN hn hτ hr hs]
  ring

/-- **Ray closure (general `n`).**  The shell closure lifts through a radial
mixture whenever the shell-level integrands are integrable almost everywhere
and the three resulting payloads are integrable in the radial variable.  The
statement is intentionally parameterized by these analytic hypotheses; the
next G1 files discharge them for the chosen radial-measure/chart model. -/
theorem radialRayCN_eq_closure_of_shell
    {n : ℕ} (hn : 3 ≤ n) {τ : ℝ} (hτ : 0 < τ) (ν : Measure ℝ)
    (hsupp : ν (Iio 0) = 0) {r : ℝ} (hr : 0 < r)
    (hK : ∀ᵐ s ∂ν, IntegrableOn (fun u : ℝ => zonalWeight n u *
      Real.exp (-(1 / τ) * shellDist r s u)) (Ioc (-1 : ℝ) 1))
    (hQ : ∀ᵐ s ∂ν, IntegrableOn (fun u : ℝ => zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) *
        (shellAxial r s u ^ 2 / shellDist r s u))) (Ioc (-1 : ℝ) 1))
    (hP : ∀ᵐ s ∂ν, IntegrableOn (fun u : ℝ => zonalWeight n u *
      (Real.exp (-(1 / τ) * shellDist r s u) *
        (shellRhoSq s u / shellDist r s u))) (Ioc (-1 : ℝ) 1))
    (hZ : Integrable (fun s => shellZN n τ r s) ν)
    (hQray : Integrable (fun s => shellQN n τ r s) ν)
    (hPray : Integrable (fun s => shellRhoSqOverDistN n τ r s) ν) :
    radialRayCN n τ ν r = τ * radialRayZN n τ ν r + radialRayQN n τ ν r
      + radialRayRhoSqOverDistN n τ ν r := by
  rw [radialRayCN, radialRayZN, radialRayQN, radialRayRhoSqOverDistN]
  have hshell : ∀ᵐ s ∂ν,
      shellCN n τ r s = τ * shellZN n τ r s + shellQN n τ r s
        + shellRhoSqOverDistN n τ r s := by
    filter_upwards [hK, hQ, hP, radial_ae_nonneg_N hsupp] with s hKs hQs hPs hs
    exact shellCN_eq_shellQN_add_of_integrable hn hτ hr hs hKs hQs hPs
  calc
    (∫ s, shellCN n τ r s ∂ν) =
        ∫ s, τ * shellZN n τ r s + shellQN n τ r s
          + shellRhoSqOverDistN n τ r s ∂ν := integral_congr_ae hshell
    _ = ∫ s, τ * shellZN n τ r s +
          (shellQN n τ r s + shellRhoSqOverDistN n τ r s) ∂ν := by
      refine integral_congr_ae (Filter.Eventually.of_forall fun s => ?_)
      ring
    _ = (∫ s, τ * shellZN n τ r s ∂ν) +
        ∫ s, shellQN n τ r s + shellRhoSqOverDistN n τ r s ∂ν := by
      simpa only [Pi.add_apply] using
        (integral_add (μ := ν) (hZ.const_mul τ) (hQray.add hPray))
    _ = τ * (∫ s, shellZN n τ r s ∂ν) +
        (∫ s, shellQN n τ r s ∂ν) +
        ∫ s, shellRhoSqOverDistN n τ r s ∂ν := by
      rw [integral_const_mul, integral_add hQray hPray]
      ring

end DriftingIdentifiability
