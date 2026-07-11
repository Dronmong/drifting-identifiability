import DriftingIdentifiability.LaplaceGeneralConverseNowhereDense

/-!
# Balance-identity scaffold for the general 1-d Laplace converse

This module starts Milestone 4 of `LaplaceGeneralConverseRoadmap.md`.

The target balance identity under zero drift is

`exp (-(2*x)/τ) * 𝔞(x) = exp ((2*x)/τ) * 𝔠(x)`,

where `𝔞 = truncatedPairing` and `𝔠 = upperPairing`.

The full theorem still requires the unconditional derivative/weak-Fubini
identity for the cross-displacement scalar.  This file formalizes the
non-controversial infrastructure around that target:

* scaled lower/upper pairings and the balance defect;
* right-continuity of the upper pairing and of the balance defect;
* conditional bridges showing that the derivative identity immediately gives
  the pointwise balance law under zero drift.

Important: for arbitrary measures the two-sided classical derivative of the
cross-displacement scalar can fail at atoms.  The mathematically correct target
is the right-derivative/weak identity; the older `HasDerivAt` bridge is kept
only as a convenient stronger conditional.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-- The lower coefficient in the zero-drift decomposition, scaled as it occurs
in the cross-displacement scalar. -/
noncomputable def scaledLowerPairing (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  Real.exp (-(2 * x) / τ) * truncatedPairing τ p q x

/-- The upper coefficient in the zero-drift decomposition, scaled as it occurs
in the cross-displacement scalar. -/
noncomputable def scaledUpperPairing (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  Real.exp ((2 * x) / τ) * upperPairing τ p q x

/-- The Milestone-4 balance defect.  The desired balance identity is exactly
`laplaceBalanceDefect τ p q x = 0`. -/
noncomputable def laplaceBalanceDefect (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  scaledUpperPairing τ p q x - scaledLowerPairing τ p q x

/-- The expected right derivative of the nonsmooth Laplace normalizer.  This
is the single remaining analytic derivative formula needed for Milestone 4. -/
noncomputable def laplaceKernelNormalizerRightDerivCoeff
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (1 / τ) *
    (-(Real.exp (-x / τ)) * lowerExpMass τ p x +
      Real.exp (x / τ) * upperExpMass τ p x)

/-- The derivative coefficient of the smooth displacement numerator, expressed
in one-sided coordinates. -/
noncomputable def laplaceDisplacementIntegralDerivCoeff
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (1 / τ) *
      (Real.exp (-x / τ) * lowerCompensatedMoment τ p x +
        Real.exp (x / τ) * upperCompensatedMoment τ p x) -
    (Real.exp (-x / τ) * lowerExpMass τ p x +
      Real.exp (x / τ) * upperExpMass τ p x)

/-- The already-certified classical derivative of the displacement numerator,
rewritten in the one-sided coordinates used by the balance identity. -/
theorem laplaceDisplacementIntegral_derivCoeff_eq
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ) :
    (1 / τ) * kernelNormalizer (laplaceCompanionKernel τ) p x -
        2 * kernelNormalizer (laplaceKernel τ) p x =
      laplaceDisplacementIntegralDerivCoeff τ p x := by
  rw [laplaceCompanionNormalizer_eq_lower_upper τ hτ p x,
    laplaceKernelNormalizer_eq_lower_upper τ hτ p x]
  unfold laplaceDisplacementIntegralDerivCoeff
  field_simp [hτ.ne']
  ring

/-- Right-continuity of the upper pairing follows from right-continuity of the
upper one-sided transforms. -/
theorem upperPairing_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    ContinuousWithinAt (fun t => upperPairing τ p q t) (Set.Ici x) x := by
  unfold upperPairing
  exact ((upperCompensatedMoment_continuousWithinAt_Ici τ hτ p x).mul
      (upperExpMass_continuousWithinAt_Ici τ hτ q x)).sub
    ((upperCompensatedMoment_continuousWithinAt_Ici τ hτ q x).mul
      (upperExpMass_continuousWithinAt_Ici τ hτ p x))

theorem scaledLowerPairing_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    ContinuousWithinAt (fun t => scaledLowerPairing τ p q t) (Set.Ici x) x := by
  unfold scaledLowerPairing
  exact (by fun_prop : ContinuousWithinAt (fun t : ℝ => Real.exp (-(2 * t) / τ))
      (Set.Ici x) x).mul
    (truncatedPairing_continuousWithinAt_Ici τ hτ p q x)

theorem scaledUpperPairing_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    ContinuousWithinAt (fun t => scaledUpperPairing τ p q t) (Set.Ici x) x := by
  unfold scaledUpperPairing
  exact (by fun_prop : ContinuousWithinAt (fun t : ℝ => Real.exp ((2 * t) / τ))
      (Set.Ici x) x).mul
    (upperPairing_continuousWithinAt_Ici τ hτ p q x)

/-- Right-continuity of the balance defect.  This is the regularity needed to
upgrade a weak/a.e. balance identity to a pointwise one. -/
theorem laplaceBalanceDefect_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    ContinuousWithinAt (fun t => laplaceBalanceDefect τ p q t) (Set.Ici x) x := by
  unfold laplaceBalanceDefect
  exact (scaledUpperPairing_continuousWithinAt_Ici τ hτ p q x).sub
    (scaledLowerPairing_continuousWithinAt_Ici τ hτ p q x)

/-- Algebraic reduction of the Milestone-4 right-derivative identity to the
right derivative of the Laplace normalizer.

The displacement numerator is already classically differentiable.  Therefore,
once the nonsmooth normalizer has the expected right derivative, the
cross-displacement scalar has right derivative `(2/τ) * balanceDefect`. -/
theorem hasDerivWithinAt_Ici_crossDisplacement_of_kernelNormalizerRightDeriv
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hZ : ∀ r : Measure ℝ, IsProbabilityMeasure r →
      ∀ x : ℝ,
        HasDerivWithinAt (fun t => kernelNormalizer (laplaceKernel τ) r t)
          (laplaceKernelNormalizerRightDerivCoeff τ r x) (Set.Ici x) x) :
    ∀ x : ℝ,
      HasDerivWithinAt (fun t => laplaceCrossDisplacementScalar τ p q t)
        ((2 / τ) * laplaceBalanceDefect τ p q x) (Set.Ici x) x := by
  intro x
  letI hp : IsProbabilityMeasure p := ‹IsProbabilityMeasure p›
  letI hq : IsProbabilityMeasure q := ‹IsProbabilityMeasure q›
  have hZp := hZ p hp x
  have hZq := hZ q hq x
  have hDp₀ := (hasDerivAt_laplaceDisplacementIntegral τ hτ p x).hasDerivWithinAt
    (s := Set.Ici x)
  have hDq₀ := (hasDerivAt_laplaceDisplacementIntegral τ hτ q x).hasDerivWithinAt
    (s := Set.Ici x)
  have hDp : HasDerivWithinAt
      (fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂p)
      (laplaceDisplacementIntegralDerivCoeff τ p x) (Set.Ici x) x := by
    rw [← laplaceDisplacementIntegral_derivCoeff_eq τ hτ p x]
    exact hDp₀
  have hDq : HasDerivWithinAt
      (fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂q)
      (laplaceDisplacementIntegralDerivCoeff τ q x) (Set.Ici x) x := by
    rw [← laplaceDisplacementIntegral_derivCoeff_eq τ hτ q x]
    exact hDq₀
  have hprod₁ := hZq.mul hDp
  have hprod₂ := hZp.mul hDq
  have hsub := hprod₁.sub hprod₂
  unfold laplaceCrossDisplacementScalar
  change HasDerivWithinAt
    (((fun t : ℝ => kernelNormalizer (laplaceKernel τ) q t) *
        fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂p) -
      ((fun t : ℝ => kernelNormalizer (laplaceKernel τ) p t) *
        fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂q))
    ((2 / τ) * laplaceBalanceDefect τ p q x) (Set.Ici x) x
  have hvalue :
      laplaceKernelNormalizerRightDerivCoeff τ q x *
            (∫ y, laplaceWeightedDisplacement τ x y ∂p) +
          kernelNormalizer (laplaceKernel τ) q x *
            laplaceDisplacementIntegralDerivCoeff τ p x -
        (laplaceKernelNormalizerRightDerivCoeff τ p x *
            (∫ y, laplaceWeightedDisplacement τ x y ∂q) +
          kernelNormalizer (laplaceKernel τ) p x *
            laplaceDisplacementIntegralDerivCoeff τ q x)
        = (2 / τ) * laplaceBalanceDefect τ p q x := by
    rw [laplaceKernelNormalizer_eq_lower_upper τ hτ p x,
      laplaceKernelNormalizer_eq_lower_upper τ hτ q x,
      laplaceDisplacementIntegral_eq_lower_upper τ hτ p x,
      laplaceDisplacementIntegral_eq_lower_upper τ hτ q x]
    unfold laplaceKernelNormalizerRightDerivCoeff laplaceDisplacementIntegralDerivCoeff
      laplaceBalanceDefect scaledUpperPairing scaledLowerPairing
      truncatedPairing upperPairing
    have hneg : (Real.exp (-(x * τ⁻¹))) ^ 2 = Real.exp (-(x * τ⁻¹ * 2)) := by
      rw [sq, ← Real.exp_add]
      congr 1
      ring
    have hpos : (Real.exp (x * τ⁻¹)) ^ 2 = Real.exp (x * τ⁻¹ * 2) := by
      rw [sq, ← Real.exp_add]
      congr 1
      ring
    field_simp [hτ.ne']
    ring_nf
    rw [hneg, hpos]
    ring
  rw [← hvalue]
  exact hsub

/-- Correct one-sided Milestone-4 bridge.

If the unconditional **right-derivative** identity

`D⁺(laplaceCrossDisplacementScalar τ p q)(x) =
  (2/τ) * laplaceBalanceDefect τ p q x`

is available pointwise as a `HasDerivWithinAt` statement on `Ici x`, then zero
drift forces the balance identity pointwise.  This is the socket the remaining
weak/Stieltjes proof should target. -/
theorem laplaceBalance_identity_of_hasDerivWithinAt_Ici_crossDisplacement
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hderiv : ∀ x : ℝ,
      HasDerivWithinAt (fun t => laplaceCrossDisplacementScalar τ p q t)
        ((2 / τ) * laplaceBalanceDefect τ p q x) (Set.Ici x) x) :
    ∀ x : ℝ, scaledLowerPairing τ p q x = scaledUpperPairing τ p q x := by
  intro x
  have hscalar_zero : ∀ t : ℝ, laplaceCrossDisplacementScalar τ p q t = 0 := by
    intro t
    have hcross := (laplaceZeroDrift_iff_crossDisplacement τ hτ p q).mp hzero t
    simp only [smul_eq_mul] at hcross
    unfold laplaceCrossDisplacementScalar
    exact sub_eq_zero.mpr hcross
  have hconst :
      (fun t => laplaceCrossDisplacementScalar τ p q t) = fun _ : ℝ => (0 : ℝ) := by
    funext t
    exact hscalar_zero t
  have hzeroDeriv :
      HasDerivWithinAt (fun t => laplaceCrossDisplacementScalar τ p q t) 0
        (Set.Ici x) x := by
    rw [hconst]
    exact hasDerivWithinAt_const x (Set.Ici x) (0 : ℝ)
  have h₁ := (hderiv x).derivWithin (uniqueDiffWithinAt_Ici x)
  have h₂ := hzeroDeriv.derivWithin (uniqueDiffWithinAt_Ici x)
  have huniq : (2 / τ) * laplaceBalanceDefect τ p q x = 0 := by
    rw [← h₁, h₂]
  have hfactor : (2 : ℝ) / τ ≠ 0 := div_ne_zero two_ne_zero hτ.ne'
  have hdefect : laplaceBalanceDefect τ p q x = 0 := by
    rcases mul_eq_zero.mp huniq with hbad | hgood
    · exact absurd hbad hfactor
    · exact hgood
  unfold laplaceBalanceDefect at hdefect
  exact (sub_eq_zero.mp hdefect).symm

/-- Milestone-4 reduction to the single nonsmooth normalizer derivative.

The displacement numerator side is already differentiable.  Therefore, to get
the pointwise balance identity from zero drift, it is enough to prove the
right-derivative formula for the raw Laplace normalizer for arbitrary
probability measures. -/
theorem laplaceBalance_identity_of_kernelNormalizerRightDeriv
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hZ : ∀ r : Measure ℝ, IsProbabilityMeasure r →
      ∀ x : ℝ,
        HasDerivWithinAt (fun t => kernelNormalizer (laplaceKernel τ) r t)
          (laplaceKernelNormalizerRightDerivCoeff τ r x) (Set.Ici x) x) :
    ∀ x : ℝ, scaledLowerPairing τ p q x = scaledUpperPairing τ p q x :=
  laplaceBalance_identity_of_hasDerivWithinAt_Ici_crossDisplacement τ hτ p q hzero
    (hasDerivWithinAt_Ici_crossDisplacement_of_kernelNormalizerRightDeriv τ hτ p q hZ)

/-- Conditional Milestone-4 bridge.

If the unconditional derivative identity

`(laplaceCrossDisplacementScalar τ p q)' =
  (2/τ) * laplaceBalanceDefect τ p q`

is available pointwise, then zero drift forces the balance identity
`scaledLowerPairing = scaledUpperPairing` pointwise.  The remaining Milestone-4
work is to prove the derivative/weak-Fubini identity itself for arbitrary
probability measures. -/
theorem laplaceBalance_identity_of_hasDerivAt_crossDisplacement
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hderiv : ∀ x : ℝ,
      HasDerivAt (fun t => laplaceCrossDisplacementScalar τ p q t)
        ((2 / τ) * laplaceBalanceDefect τ p q x) x) :
    ∀ x : ℝ, scaledLowerPairing τ p q x = scaledUpperPairing τ p q x := by
  intro x
  have hscalar_zero : ∀ t : ℝ, laplaceCrossDisplacementScalar τ p q t = 0 := by
    intro t
    have hcross := (laplaceZeroDrift_iff_crossDisplacement τ hτ p q).mp hzero t
    simp only [smul_eq_mul] at hcross
    unfold laplaceCrossDisplacementScalar
    exact sub_eq_zero.mpr hcross
  have hconst :
      (fun t => laplaceCrossDisplacementScalar τ p q t) = fun _ : ℝ => (0 : ℝ) := by
    funext t
    exact hscalar_zero t
  have hzeroDeriv :
      HasDerivAt (fun t => laplaceCrossDisplacementScalar τ p q t) 0 x := by
    rw [hconst]
    exact hasDerivAt_const x (0 : ℝ)
  have huniq := (hderiv x).unique hzeroDeriv
  have hfactor : (2 : ℝ) / τ ≠ 0 := div_ne_zero two_ne_zero hτ.ne'
  have hdefect : laplaceBalanceDefect τ p q x = 0 := by
    rcases mul_eq_zero.mp huniq with hbad | hgood
    · exact absurd hbad hfactor
    · exact hgood
  unfold laplaceBalanceDefect at hdefect
  exact (sub_eq_zero.mp hdefect).symm

end DriftingIdentifiability
