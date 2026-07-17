import DriftingIdentifiability.LaplaceRadialMeasureN
import DriftingIdentifiability.TrustedBoundary

/-!
# The general-n zonal-coordinate bridge boundary

`LaplaceRadialMeasureN` constructs the genuine Haar law on the unit sphere and
the radius-times-direction mixture.  The remaining geometric input in G1 is
the classical spherical-slicing formula: the first coordinate of normalized
Haar measure has density proportional to
`(1 - u^2)^((n-3)/2)` on `[-1,1]`.

That formula is not currently available in Mathlib.  The exact classical
coordinate-marginal theorem is therefore kept as one reviewed standard
external analytic fact in `Paperaxioms.lean`.  It is independent of the drift
problem: its statement mentions only normalized surface measure and a
continuous scalar test function.  This file packages that theorem as the
downstream `ZonalSphereBridge`, keeping the dependency explicit and preventing
any normalization convention from being hidden in later radial arguments.
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
    ∀ g : ℝ → ℝ, ContinuousOn g (Icc (-1 : ℝ) 1) →
      (∫ ω, g (sphereFirstCoord n (by omega) ω)
          ∂(uniformSphereMeasure n)) =
        (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * g u

/-- The reviewed classical first-coordinate marginal theorem supplies the
general-dimensional zonal bridge.  This is the only externally trusted input
in the bridge; all later shell, ODE, and identifiability steps remain proof
obligations. -/
theorem zonalSphereBridge_standard (n : ℕ) (hn : 3 ≤ n) :
    ZonalSphereBridge n hn := by
  constructor
  intro g hg
  let e₀ : EuclideanSpace ℝ (Fin n) :=
    EuclideanSpace.basisFun (Fin n) ℝ ⟨0, by omega⟩
  have he₀ : ‖e₀‖ = 1 :=
    (EuclideanSpace.basisFun (Fin n) ℝ).norm_eq_one ⟨0, by omega⟩
  have hcoord : ∀ ω : sphere (0 : EuclideanSpace ℝ (Fin n)) 1,
      inner ℝ e₀ (ω : EuclideanSpace ℝ (Fin n)) =
        sphereFirstCoord n (by omega) ω := by
    intro ω
    simp [e₀, sphereFirstCoord, PiLp.inner_apply]
  calc
    (∫ ω, g (sphereFirstCoord n (by omega) ω) ∂(uniformSphereMeasure n)) =
        ∫ ω, g (inner ℝ e₀ (ω : EuclideanSpace ℝ (Fin n)))
          ∂(uniformSphereMeasure n) := by
      apply integral_congr_ae
      exact Filter.Eventually.of_forall fun ω => congrArg g (hcoord ω).symm
    _ = (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
          zonalWeight n u * g u := by
      simpa [uniformSphereMeasure, zonalMass, zonalWeight, zonalExponent] using
        (Paper.uniformSphere_directionalCoordinate_integral n hn e₀ he₀ g hg)

/-- The same reviewed sphere-coordinate theorem in an arbitrary unit
direction.  This is the rotation-invariant form used to prove that the
normalizer of a Haar radial mixture depends only on the probe norm. -/
theorem integral_uniformSphere_directional
    {n : ℕ} (hn : 3 ≤ n) (v : EuclideanSpace ℝ (Fin n)) (hv : ‖v‖ = 1)
    (g : ℝ → ℝ) (hg : ContinuousOn g (Icc (-1 : ℝ) 1)) :
    (∫ ω, g (inner ℝ v (ω : EuclideanSpace ℝ (Fin n)))
        ∂(uniformSphereMeasure n)) =
      (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * g u := by
  simpa [uniformSphereMeasure, zonalMass, zonalWeight, zonalExponent] using
    (Paper.uniformSphere_directionalCoordinate_integral n hn v hv g hg)

/-- The bridge rewrites any continuous zonal sphere average. -/
lemma integral_uniformSphere_zonal_of_bridge
    {n : ℕ} {hn : 3 ≤ n} (hbridge : ZonalSphereBridge n hn)
    (g : ℝ → ℝ) (hg : Continuous g) :
    (∫ ω, g (sphereFirstCoord n (by omega) ω)
        ∂(uniformSphereMeasure n)) =
      (zonalMass n)⁻¹ * ∫ u in Ioc (-1 : ℝ) 1,
        zonalWeight n u * g u :=
  hbridge.integral_eq g hg.continuousOn

/-- Continuous-on-the-coordinate-interval form of the bridge.  This is the
natural interface for removable endpoint singularities in differentiated
shell payloads. -/
lemma integral_uniformSphere_zonalOn_of_bridge
    {n : ℕ} {hn : 3 ≤ n} (hbridge : ZonalSphereBridge n hn)
    (g : ℝ → ℝ) (hg : ContinuousOn g (Icc (-1 : ℝ) 1)) :
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
