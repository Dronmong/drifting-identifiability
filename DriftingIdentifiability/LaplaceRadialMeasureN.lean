import DriftingIdentifiability.LaplaceRadialRayN
import Mathlib.MeasureTheory.Constructions.HaarToSphere

/-!
# General-`n` radial measures: the uniform-sphere bridge boundary

This file supplies the genuine measure-theoretic object requested by G1.  It
uses Mathlib's finite-dimensional Haar-to-sphere construction, normalizes the
result to a probability measure, and mixes it with a radial profile.  The
integral bridge is proved by `Measure.map`/product integration.  The remaining
zonal-coordinate identity is intentionally left as a named boundary: it is a
geometric pushforward computation, not an assumption hidden in the radial
measure definition.
-/

open MeasureTheory Filter Topology Set Metric
open scoped ENNReal

namespace DriftingIdentifiability

open Paper

/-- The normalized Haar measure on the unit sphere in `ℝⁿ`.  For `n = 0` the
normalizing factor is harmlessly defined by `0⁻¹ = 0`; all probability results
below assume `n ≥ 3`. -/
noncomputable def uniformSphereMeasure (n : ℕ) :
    Measure (sphere (0 : EuclideanSpace ℝ (Fin n)) 1) :=
  ((Measure.toSphere (volume : Measure (EuclideanSpace ℝ (Fin n))) univ)⁻¹) •
    Measure.toSphere (volume : Measure (EuclideanSpace ℝ (Fin n)))

lemma uniformSphereMeasure_isProbability {n : ℕ} (hn : 3 ≤ n) :
    IsProbabilityMeasure (uniformSphereMeasure n) := by
  letI : Nonempty (Fin n) := ⟨⟨0, by omega⟩⟩
  letI : Nontrivial (EuclideanSpace ℝ (Fin n)) := by infer_instance
  constructor
  have hmass : (Measure.toSphere (volume : Measure (EuclideanSpace ℝ (Fin n)))) univ ≠ 0 :=
    Measure.measure_univ_ne_zero.2
      (Measure.toSphere_ne_zero (volume : Measure (EuclideanSpace ℝ (Fin n))))
  have htop : (Measure.toSphere (volume : Measure (EuclideanSpace ℝ (Fin n)))) univ ≠ ∞ :=
    (IsFiniteMeasure.measure_univ_lt_top (μ :=
      Measure.toSphere (volume : Measure (EuclideanSpace ℝ (Fin n))))).ne
  rw [uniformSphereMeasure, Measure.smul_apply, smul_eq_mul,
    ENNReal.inv_mul_cancel hmass htop]

/-- The radius-times-direction map from `ℝ × Sⁿ⁻¹` to `ℝⁿ`. -/
noncomputable def radialScaleMap (n : ℕ) :
    ℝ × sphere (0 : EuclideanSpace ℝ (Fin n)) 1 → EuclideanSpace ℝ (Fin n) :=
  fun z => z.1 • (z.2 : EuclideanSpace ℝ (Fin n))

lemma continuous_radialScaleMap (n : ℕ) : Continuous (radialScaleMap n) := by
  unfold radialScaleMap
  exact continuous_fst.smul (continuous_subtype_val.comp continuous_snd)

/-- The centered-at-zero radial mixture with radial profile `ν`. -/
noncomputable def radialMixtureN (n : ℕ) (ν : Measure ℝ) :
    Measure (EuclideanSpace ℝ (Fin n)) :=
  (ν.prod (uniformSphereMeasure n)).map (radialScaleMap n)

lemma radialMixtureN_isProbabilityMeasure {n : ℕ} (hn : 3 ≤ n)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] :
    IsProbabilityMeasure (radialMixtureN n ν) := by
  letI := uniformSphereMeasure_isProbability hn
  exact Measure.isProbabilityMeasure_map (continuous_radialScaleMap n).aemeasurable

/-- **Radial integral bridge.**  Every integrable function against the genuine
radial mixture reduces to a product integral over the profile and the uniform
unit-sphere law. -/
lemma integral_radialMixtureN {n : ℕ} (hn : 3 ≤ n) (ν : Measure ℝ)
    [IsProbabilityMeasure ν] {f : EuclideanSpace ℝ (Fin n) → ℝ}
    (hf : Integrable f (radialMixtureN n ν)) :
    ∫ y, f y ∂(radialMixtureN n ν) =
      ∫ s, ∫ ω, f (s • (ω : EuclideanSpace ℝ (Fin n)))
        ∂(uniformSphereMeasure n) ∂ν := by
  letI := uniformSphereMeasure_isProbability hn
  have hmap : AEMeasurable (radialScaleMap n)
      (ν.prod (uniformSphereMeasure n)) :=
    (continuous_radialScaleMap n).aemeasurable
  have hint : Integrable (fun z => f (radialScaleMap n z))
      (ν.prod (uniformSphereMeasure n)) :=
    (integrable_map_measure hf.aestronglyMeasurable hmap).mp hf
  rw [radialMixtureN, integral_map hmap hf.aestronglyMeasurable]
  simpa [radialScaleMap] using (integral_prod _ hint)

/-- The translated radial mixture. -/
noncomputable def radialMixtureN_centered (n : ℕ) (ν : Measure ℝ)
    (c : EuclideanSpace ℝ (Fin n)) : Measure (EuclideanSpace ℝ (Fin n)) :=
  (radialMixtureN n ν).map (fun x => c + x)

lemma radialMixtureN_centered_isProbabilityMeasure {n : ℕ} (hn : 3 ≤ n)
    (ν : Measure ℝ) [IsProbabilityMeasure ν]
    (c : EuclideanSpace ℝ (Fin n)) :
    IsProbabilityMeasure (radialMixtureN_centered n ν c) := by
  letI := radialMixtureN_isProbabilityMeasure hn ν
  exact Measure.isProbabilityMeasure_map (by fun_prop :
    AEMeasurable (fun x : EuclideanSpace ℝ (Fin n) => c + x)
      (radialMixtureN n ν))

end DriftingIdentifiability
