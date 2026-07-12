import DriftingIdentifiability.LaplaceACRegularity

/-!
# Continuous densities imply the L5 C²-normalizer certificate

This file proves the density-level calculus promised in `LaplaceACDerivation.md`.
For a probability/finite measure represented as

`p = volume.withDensity (fun x => ENNReal.ofReal (ρ x))`

with `ρ` nonnegative and continuous, the one-dimensional Laplace normalizer is
classically C² in the sense needed by `LaplaceC2NormalizerRegular`.

No exponential moment is needed for this local differentiability statement; the
tail/compactness parts of the a.c. converse still use the separate exponential
moment hypotheses already documented in the roadmap.
-/

open MeasureTheory Set Filter Topology
open scoped intervalIntegral

namespace DriftingIdentifiability

open Paper

/-! ## Density representation helpers -/

/-- A nonnegative continuous density representation of a real measure. -/
structure ContinuousDensityMeasure (p : Measure ℝ) (ρ : ℝ → ℝ) : Prop where
  nonneg : ∀ x : ℝ, 0 ≤ ρ x
  continuous : Continuous ρ
  measure_eq : p = volume.withDensity fun x => ENNReal.ofReal (ρ x)

private lemma setIntegral_withDensity_ofReal_eq
    (ρ g : ℝ → ℝ) (hρ_nonneg : ∀ x : ℝ, 0 ≤ ρ x)
    (hρ_cont : Continuous ρ) {s : Set ℝ} (hs : MeasurableSet s) :
    (∫ y in s, g y ∂(volume.withDensity fun y => ENNReal.ofReal (ρ y))) =
      ∫ y in s, ρ y * g y ∂volume := by
  have hmeas : Measurable (fun y : ℝ => ENNReal.ofReal (ρ y)) :=
    hρ_cont.measurable.ennreal_ofReal
  have htop : ∀ᵐ y ∂volume.restrict s, ENNReal.ofReal (ρ y) < ⊤ := by
    filter_upwards with y
    exact ENNReal.ofReal_lt_top
  rw [setIntegral_withDensity_eq_setIntegral_toReal_smul hmeas htop g hs]
  apply setIntegral_congr_fun hs
  intro y _
  simp [smul_eq_mul, ENNReal.toReal_ofReal (hρ_nonneg y)]

private lemma lowerExpMass_density_interval
    (τ : ℝ) (ρ : ℝ → ℝ) (hρ_nonneg : ∀ x : ℝ, 0 ≤ ρ x)
    (hρ_cont : Continuous ρ) {x t : ℝ} (hxt : x ≤ t) :
    ∫ y in Set.Ioc x t, Real.exp (y / τ)
        ∂(volume.withDensity fun y => ENNReal.ofReal (ρ y)) =
      ∫ y in x..t, Real.exp (y / τ) * ρ y := by
  rw [setIntegral_withDensity_ofReal_eq ρ (fun y => Real.exp (y / τ))
      hρ_nonneg hρ_cont measurableSet_Ioc]
  rw [intervalIntegral.integral_of_le hxt]
  apply setIntegral_congr_fun measurableSet_Ioc
  intro y _
  ring

private lemma upperExpMass_density_interval
    (τ : ℝ) (ρ : ℝ → ℝ) (hρ_nonneg : ∀ x : ℝ, 0 ≤ ρ x)
    (hρ_cont : Continuous ρ) {x t : ℝ} (hxt : x ≤ t) :
    ∫ y in Set.Ioc x t, Real.exp (-y / τ)
        ∂(volume.withDensity fun y => ENNReal.ofReal (ρ y)) =
      ∫ y in x..t, Real.exp (-y / τ) * ρ y := by
  rw [setIntegral_withDensity_ofReal_eq ρ (fun y => Real.exp (-y / τ))
      hρ_nonneg hρ_cont measurableSet_Ioc]
  rw [intervalIntegral.integral_of_le hxt]
  apply setIntegral_congr_fun measurableSet_Ioc
  intro y _
  ring

/-! ## One-sided masses are differentiable under a continuous density -/

theorem hasDerivAt_lowerExpMass_of_continuousDensity
    (τ : ℝ) (hτ : ValidBandwidth τ) (ρ : ℝ → ℝ)
    [IsFiniteMeasure (volume.withDensity fun y => ENNReal.ofReal (ρ y))]
    (hρ_nonneg : ∀ x : ℝ, 0 ≤ ρ x) (hρ_cont : Continuous ρ) (x : ℝ) :
    HasDerivAt
      (fun t => lowerExpMass τ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) t)
      (Real.exp (x / τ) * ρ x) x := by
  let w : ℝ → ℝ := fun y => Real.exp (y / τ) * ρ y
  have hwcont : Continuous w := by
    unfold w
    fun_prop
  have hF := hwcont.integral_hasStrictDerivAt x x
  have hlocal :
      (fun t => lowerExpMass τ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) t)
        =ᶠ[𝓝 x]
      fun t =>
        lowerExpMass τ (volume.withDensity fun y => ENNReal.ofReal (ρ y)) x +
          ∫ y in x..t, w y := by
    refine Eventually.of_forall ?_
    intro t
    rcases le_total x t with hxt | htx
    · have hsub := lowerExpMass_sub τ hτ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) hxt
      have hint := lowerExpMass_density_interval τ ρ hρ_nonneg hρ_cont hxt
      unfold w
      linarith
    · have hsub := lowerExpMass_sub τ hτ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) htx
      have hint := lowerExpMass_density_interval τ ρ hρ_nonneg hρ_cont htx
      unfold w
      rw [intervalIntegral.integral_symm x t] at hint
      linarith
  have hderiv_const :
      HasDerivAt
        (fun t =>
          lowerExpMass τ (volume.withDensity fun y => ENNReal.ofReal (ρ y)) x +
            ∫ y in x..t, w y)
        (w x) x := by
    simpa using hF.hasDerivAt.const_add
      (lowerExpMass τ (volume.withDensity fun y => ENNReal.ofReal (ρ y)) x)
  have hderiv := hderiv_const.congr_of_eventuallyEq hlocal
  simpa [w] using hderiv

theorem hasDerivAt_upperExpMass_of_continuousDensity
    (τ : ℝ) (hτ : ValidBandwidth τ) (ρ : ℝ → ℝ)
    [IsFiniteMeasure (volume.withDensity fun y => ENNReal.ofReal (ρ y))]
    (hρ_nonneg : ∀ x : ℝ, 0 ≤ ρ x) (hρ_cont : Continuous ρ) (x : ℝ) :
    HasDerivAt
      (fun t => upperExpMass τ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) t)
      (-(Real.exp (-x / τ) * ρ x)) x := by
  let w : ℝ → ℝ := fun y => Real.exp (-y / τ) * ρ y
  have hwcont : Continuous w := by
    unfold w
    fun_prop
  have hF := hwcont.integral_hasStrictDerivAt x x
  have hlocal :
      (fun t => upperExpMass τ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) t)
        =ᶠ[𝓝 x]
      fun t =>
        upperExpMass τ (volume.withDensity fun y => ENNReal.ofReal (ρ y)) x -
          ∫ y in x..t, w y := by
    refine Eventually.of_forall ?_
    intro t
    rcases le_total x t with hxt | htx
    · have hsub := upperExpMass_sub τ hτ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) hxt
      have hint := upperExpMass_density_interval τ ρ hρ_nonneg hρ_cont hxt
      unfold w
      linarith
    · have hsub := upperExpMass_sub τ hτ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) htx
      have hint := upperExpMass_density_interval τ ρ hρ_nonneg hρ_cont htx
      unfold w
      rw [intervalIntegral.integral_symm x t] at hint
      linarith
  have hderiv_neg :
      HasDerivAt
        (fun t =>
          upperExpMass τ (volume.withDensity fun y => ENNReal.ofReal (ρ y)) x -
            ∫ y in x..t, w y)
        (-(w x)) x := by
    simpa using
      (HasDerivAt.const_sub
        (upperExpMass τ (volume.withDensity fun y => ENNReal.ofReal (ρ y)) x)
        hF.hasDerivAt)
  have hderiv := hderiv_neg.congr_of_eventuallyEq hlocal
  simpa [w] using hderiv

/-! ## Assembly: the Laplace normalizer is C² -/

noncomputable def laplaceNormalizerDensitySecondDeriv
    (τ : ℝ) (ρ : ℝ → ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (kernelNormalizer (laplaceKernel τ) p x - 2 * τ * ρ x) / τ ^ 2

theorem hasDerivAt_laplaceKernelNormalizer_of_continuousDensity
    (τ : ℝ) (hτ : ValidBandwidth τ) (ρ : ℝ → ℝ)
    [IsFiniteMeasure (volume.withDensity fun y => ENNReal.ofReal (ρ y))]
    (hρ_nonneg : ∀ x : ℝ, 0 ≤ ρ x) (hρ_cont : Continuous ρ) (x : ℝ) :
    HasDerivAt
      (fun s => kernelNormalizer (laplaceKernel τ)
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) s)
      (laplaceKernelNormalizerRightDerivCoeff τ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) x) x := by
  let p : Measure ℝ := volume.withDensity fun y => ENNReal.ofReal (ρ y)
  have hL := hasDerivAt_lowerExpMass_of_continuousDensity τ hτ ρ hρ_nonneg hρ_cont x
  have hU := hasDerivAt_upperExpMass_of_continuousDensity τ hτ ρ hρ_nonneg hρ_cont x
  have hExpNeg : HasDerivAt (fun t : ℝ => Real.exp (-t / τ))
      (-(1 / τ) * Real.exp (-x / τ)) x := by
    have hlin : HasDerivAt (fun t : ℝ => -t / τ) (-(1 / τ)) x := by
      simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
        ((hasDerivAt_id x).const_mul (-(1 / τ)))
    simpa [mul_comm, mul_left_comm, mul_assoc] using hlin.exp
  have hExpPos : HasDerivAt (fun t : ℝ => Real.exp (t / τ))
      ((1 / τ) * Real.exp (x / τ)) x := by
    have hlin : HasDerivAt (fun t : ℝ => t / τ) (1 / τ) x := by
      simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
        ((hasDerivAt_id x).const_mul (1 / τ))
    simpa [mul_comm, mul_left_comm, mul_assoc] using hlin.exp
  have hsum := (hExpNeg.mul hL).add (hExpPos.mul hU)
  have hformula :
      (fun s => kernelNormalizer (laplaceKernel τ) p s)
        = fun s =>
          Real.exp (-s / τ) * lowerExpMass τ p s +
            Real.exp (s / τ) * upperExpMass τ p s := by
    funext s
    exact laplaceKernelNormalizer_eq_lower_upper τ hτ p s
  have hsum' :
      HasDerivAt
        (fun s =>
          Real.exp (-s / τ) * lowerExpMass τ p s +
            Real.exp (s / τ) * upperExpMass τ p s)
        (laplaceKernelNormalizerRightDerivCoeff τ p x) x := by
    have hval :
        (-(1 / τ) * Real.exp (-x / τ) * lowerExpMass τ p x +
              Real.exp (-x / τ) * (Real.exp (x / τ) * ρ x)) +
            ((1 / τ) * Real.exp (x / τ) * upperExpMass τ p x +
              Real.exp (x / τ) * (-(Real.exp (-x / τ) * ρ x))) =
          laplaceKernelNormalizerRightDerivCoeff τ p x := by
      unfold laplaceKernelNormalizerRightDerivCoeff
      field_simp [hτ.ne']
      ring
    change HasDerivAt
      (((fun t : ℝ => Real.exp (-t / τ)) *
          fun t : ℝ => lowerExpMass τ p t) +
        ((fun t : ℝ => Real.exp (t / τ)) *
          fun t : ℝ => upperExpMass τ p t))
      (laplaceKernelNormalizerRightDerivCoeff τ p x) x
    rw [← hval]
    simpa [p] using hsum
  exact hsum'.congr_of_eventuallyEq (Eventually.of_forall fun s => by
    rw [hformula])

theorem hasDerivAt_laplaceKernelNormalizerRightDerivCoeff_of_continuousDensity
    (τ : ℝ) (hτ : ValidBandwidth τ) (ρ : ℝ → ℝ)
    [IsFiniteMeasure (volume.withDensity fun y => ENNReal.ofReal (ρ y))]
    (hρ_nonneg : ∀ x : ℝ, 0 ≤ ρ x) (hρ_cont : Continuous ρ) (x : ℝ) :
    HasDerivAt
      (laplaceKernelNormalizerRightDerivCoeff τ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)))
      (laplaceNormalizerDensitySecondDeriv τ ρ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y)) x) x := by
  let p : Measure ℝ := volume.withDensity fun y => ENNReal.ofReal (ρ y)
  have hL := hasDerivAt_lowerExpMass_of_continuousDensity τ hτ ρ hρ_nonneg hρ_cont x
  have hU := hasDerivAt_upperExpMass_of_continuousDensity τ hτ ρ hρ_nonneg hρ_cont x
  have hExpNeg : HasDerivAt (fun t : ℝ => Real.exp (-t / τ))
      (-(1 / τ) * Real.exp (-x / τ)) x := by
    have hlin : HasDerivAt (fun t : ℝ => -t / τ) (-(1 / τ)) x := by
      simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
        ((hasDerivAt_id x).const_mul (-(1 / τ)))
    simpa [mul_comm, mul_left_comm, mul_assoc] using hlin.exp
  have hExpPos : HasDerivAt (fun t : ℝ => Real.exp (t / τ))
      ((1 / τ) * Real.exp (x / τ)) x := by
    have hlin : HasDerivAt (fun t : ℝ => t / τ) (1 / τ) x := by
      simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
        ((hasDerivAt_id x).const_mul (1 / τ))
    simpa [mul_comm, mul_left_comm, mul_assoc] using hlin.exp
  have hA := (hExpNeg.mul hL).neg
  have hB := hExpPos.mul hU
  have hsum := (hA.add hB).const_mul (1 / τ)
  have hval :
      (1 / τ) *
        (- (-(1 / τ) * Real.exp (-x / τ) * lowerExpMass τ p x +
              Real.exp (-x / τ) * (Real.exp (x / τ) * ρ x)) +
          ((1 / τ) * Real.exp (x / τ) * upperExpMass τ p x +
            Real.exp (x / τ) * (-(Real.exp (-x / τ) * ρ x)))) =
      laplaceNormalizerDensitySecondDeriv τ ρ p x := by
    unfold laplaceNormalizerDensitySecondDeriv
    rw [laplaceKernelNormalizer_eq_lower_upper τ hτ p x]
    field_simp [hτ.ne']
    ring_nf
    have hexp_cancel : Real.exp (-(x * τ⁻¹)) * Real.exp (x * τ⁻¹) = 1 := by
      rw [← Real.exp_add]
      ring_nf
      simp
    rw [show τ * Real.exp (-(x * τ⁻¹)) * Real.exp (x * τ⁻¹) * ρ x * 2 =
        τ * ρ x * 2 by
      rw [mul_assoc τ, hexp_cancel]
      ring]
  unfold laplaceKernelNormalizerRightDerivCoeff
  rw [← hval]
  simpa [p, Pi.mul_apply] using hsum

/-- Continuous nonnegative densities give the C²-normalizer certificate consumed
by the L5 Abel bridge.

The two-sided exponential moment assumptions used elsewhere in the a.c. converse
are not needed here: this is a local regularity statement. -/
noncomputable def laplaceC2NormalizerRegular_of_continuousDensity
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p]
    (ρ : ℝ → ℝ) (hpρ : ContinuousDensityMeasure p ρ) :
    LaplaceC2NormalizerRegular τ p := by
  haveI : IsFiniteMeasure (volume.withDensity fun y => ENNReal.ofReal (ρ y)) := by
    rw [← hpρ.measure_eq]
    infer_instance
  rw [hpρ.measure_eq]
  refine
    { secondDeriv := laplaceNormalizerDensitySecondDeriv τ ρ
        (volume.withDensity fun y => ENNReal.ofReal (ρ y))
      hasDerivAt_normalizer := ?_
      hasDerivAt_rightDerivCoeff := ?_ }
  · intro x
    exact hasDerivAt_laplaceKernelNormalizer_of_continuousDensity
      τ hτ ρ hpρ.nonneg hpρ.continuous x
  · intro x
    exact hasDerivAt_laplaceKernelNormalizerRightDerivCoeff_of_continuousDensity
      τ hτ ρ hpρ.nonneg hpρ.continuous x

end DriftingIdentifiability
