import DriftingIdentifiability.LaplaceRadialShell3

/-!
# Radial Laplace converse, milestone L5: ray layer (`n = 3`)

This file starts the R-layer promised in `LaplaceHigherDim.md §4.10` and
`LaplaceL5_HANDOFF.md`.  The shell layer has reduced ray evaluations of the
Laplace normalizer, companion normalizer, and drift numerator to per-shell
zonal averages.  Here we package those shell mixtures as honest ray-profile
functions of the probe radius `r`.

The heavy R-layer still to come is the differentiability/system layer
(`Z̃'`, `C̃'`, common tilted displacement, closure identity, and the RSI sign
split).  This file deliberately contains only definitions and shell-bridge
facts, so it stays axiom-free and does not smuggle in any endgame theorem.
-/

open MeasureTheory Filter Topology Set
open scoped RealInnerProductSpace

namespace DriftingIdentifiability
open Paper

/-- Ray Laplace normalizer profile for a radial mixture. -/
noncomputable def radialRayZ₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellZ τ r s ∂ν

/-- Ray companion-normalizer profile for a radial mixture. -/
noncomputable def radialRayC₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellC τ r s ∂ν

/-- Ray first-coordinate drift-numerator profile for a radial mixture. -/
noncomputable def radialRayD₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellD τ r s ∂ν

/-- Ray `T` profile, the shell average of the axial coordinate `t=s·u`. -/
noncomputable def radialRayT₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellT τ r s ∂ν

/-- Ray tangential `ρ²/d` profile. -/
noncomputable def radialRayRhoSqOverDist₃ (τ : ℝ) (ν : Measure ℝ) (r : ℝ) : ℝ :=
  ∫ s, shellRhoSqOverDist τ r s ∂ν

lemma radialRayZ₃_eq_kernelNormalizer (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayZ₃ τ ν r =
      kernelNormalizer (laplaceKernel τ) (radialMixture₃ ν) (rayProbe r) := by
  rw [radialRayZ₃, kernelNormalizer_radialMixture₃ τ hτ ν r]

lemma radialRayC₃_eq_companionNormalizer (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayC₃ τ ν r =
      kernelNormalizer (laplaceCompanionKernel τ) (radialMixture₃ ν) (rayProbe r) := by
  rw [radialRayC₃, kernelNormalizer_companion_radialMixture₃ τ hτ ν r]

lemma radialRayD₃_eq_weightedDisplacementCoord (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    radialRayD₃ τ ν r =
      (∫ y, laplaceWeightedDisplacement τ (rayProbe r) y ∂(radialMixture₃ ν)) 0 := by
  rw [radialRayD₃, laplaceWeightedDisplacement_coord_radialMixture₃ τ hτ ν r]

end DriftingIdentifiability
