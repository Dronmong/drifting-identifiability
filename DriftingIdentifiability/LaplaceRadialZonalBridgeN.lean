import DriftingIdentifiability.LaplaceRadialMeasureN

/-!
# The general-n zonal-coordinate bridge boundary

`LaplaceRadialMeasureN` constructs the genuine Haar law on the unit sphere and
the radius-times-direction mixture.  The remaining geometric input in G1 is
the classical spherical-slicing formula: the first coordinate of normalized
Haar measure has density proportional to
`(1 - u^2)^((n-3)/2)` on `[-1,1]`.

That formula is not currently available in Mathlib.  This file does **not**
turn it into an axiom.  Instead it records the exact proposition as an
explicit boundary (`ZonalSphereBridge`) and proves the bookkeeping around it.
Consequently any theorem using the bridge has a visible, independently
checkable geometric hypothesis, while the measure construction itself remains
fully genuine and axiom-free.
-/

open MeasureTheory Filter Topology Set Metric
open scoped ENNReal

namespace DriftingIdentifiability

open Paper

/-! ## The first sphere coordinate -/

/-- The coordinate used by the zonal chart.  The proof argument is only used
to construct the `Fin n` index; propositions are proof-irrelevant. -/
noncomputable def sphereFirstCoord (n : ℕ) (hn : 0 < n)
    (ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1) : ℝ :=
  (ω : EuclideanSpace ℝ (Fin n)) ⟨0, hn⟩

lemma continuous_sphereFirstCoord {n : ℕ} (hn : 0 < n) :
    Continuous (sphereFirstCoord n hn) := by
  unfold sphereFirstCoord
  fun_prop

lemma measurable_sphereFirstCoord {n : ℕ} (hn : 0 < n) :
    Measurable (sphereFirstCoord n hn) :=
  (continuous_sphereFirstCoord hn).measurable

/-! ## The explicit geometric boundary -/

/--
The spherical-slicing identity needed by the general-n radial converse.

For every continuous zonal function `g`, normalized Haar integration on the
unit sphere is the normalized weighted one-dimensional integral.  The
statement is intentionally a structure rather than an axiom: a future proof
of the classical slicing theorem may inhabit it, and all downstream results
must explicitly carry that inhabitant.
-/
structure ZonalSphereBridge (n : ℕ) (hn : 3 ≤ n) : Prop where
  integral_eq :
    ∀ g : ℝ → ℝ, Continuous g →
      (∫ ω, g (sphereFirstCoord n (by omega) ω)
          ∂(uniformSphereMeasure n)) =
        (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * g u

/-- The bridge rewrites any continuous zonal sphere average. -/
lemma integral_uniformSphere_zonal_of_bridge
    {n : ℕ} {hn : 3 ≤ n} (hbridge : ZonalSphereBridge n hn)
    (g : ℝ → ℝ) (hg : Continuous g) :
    (∫ ω, g (sphereFirstCoord n (by omega) ω)
        ∂(uniformSphereMeasure n)) =
      (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * g u :=
  hbridge.integral_eq g hg

/-! ## Pushforward bookkeeping (proved without the bridge) -/

/-- The first-coordinate pushforward of Haar measure is a probability measure.
This fact does not identify its density; it is useful independently for
boundedness and weak-convergence arguments. -/
lemma uniformSphereFirstCoord_map_isProbability {n : ℕ} (hn : 3 ≤ n) :
    IsProbabilityMeasure
      ((uniformSphereMeasure n).map (sphereFirstCoord n (by omega))) := by
  letI := uniformSphereMeasure_isProbability hn
  exact Measure.isProbabilityMeasure_map
    (continuous_sphereFirstCoord (by omega)).aemeasurable

/-! ## A clean downstream interface -/

/-- Any continuous radial/zonal integrand can be transferred from the genuine
sphere to the one-dimensional shell layer once the explicit bridge is
provided.  Keeping this theorem separate prevents an accidental use of the
bridge as if it were already proved. -/
theorem radialShellIntegral_of_zonalBridge
    {n : ℕ} {hn : 3 ≤ n} (hbridge : ZonalSphereBridge n hn)
    (g : ℝ → ℝ) (hg : Continuous g) :
    (∫ ω, g (sphereFirstCoord n (by omega) ω)
        ∂(uniformSphereMeasure n)) =
      (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * g u :=
  integral_uniformSphere_zonal_of_bridge hbridge g hg

end DriftingIdentifiability
