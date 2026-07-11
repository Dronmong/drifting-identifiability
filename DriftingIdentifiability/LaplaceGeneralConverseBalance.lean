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
