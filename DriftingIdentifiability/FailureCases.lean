import DriftingIdentifiability.PopulationCandidate

/-!
# Machine-checked boundary failures

Small lemmas recording why each practical side condition is present.  They are
regressions against accidentally weakening the promoted theorem.
-/

open MeasureTheory

namespace DriftingIdentifiability

open Paper
open PaperFiniteIdentifiability
open CandidateConditions

universe u

theorem zero_target_normalizer_not_regular
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) (p q : Distribution E) (x : E)
    (hzero : kernelNormalizer k p x = 0) :
    ¬ MeanShiftRegularAt k p q x := by
  intro h
  exact h.zp_ne_zero hzero

theorem insufficientProbeDimension_not_nondegenerate
    {W : Type*} [NormedAddCommGroup W] [NormedSpace ℝ W]
    [FiniteDimensional ℝ W] {m N : ℕ}
    (U : Fin m → Fin m → Fin N → W)
    (hdim : N * Module.finrank ℝ W < Fintype.card (StrictPair m)) :
    ¬ BasisInteractionNondegenerate U := by
  intro h
  exact (Nat.not_lt_of_ge (nondegenerate_pairCount_le_probeDimension U h)) hdim

theorem duplicateInteraction_not_nondegenerate
    {W : Type*} [AddCommGroup W] [Module ℝ W] {m N : ℕ}
    (U : Fin m → Fin m → Fin N → W) {p q : StrictPair m}
    (hpq : p ≠ q) (hduplicate : U p.1.1 p.1.2 = U q.1.1 q.1.2) :
    ¬ BasisInteractionNondegenerate U := by
  intro h
  exact hpq (h.injective hduplicate)

theorem collapsedFiniteFamily_not_legitimate
    {E : Type u} [MeasurableSpace E] {reference : Distribution E} {m : ℕ}
    (basis : ProbabilityDensityBasis reference m)
    (hcollapse : ∀ a b : FiniteProbabilityVector m,
      basis.basisMeasure a = basis.basisMeasure b) :
    ¬ ConditionAllowsDistinctPair (FiniteBasisFamily basis) := by
  rintro ⟨p, q, ⟨a, b, rfl, rfl⟩, hpq⟩
  exact hpq (hcollapse a b)

/-- Continuous a.e.-zero need not be pointwise zero when the sampling measure
lacks full support. -/
def identityDrift : DriftingField ℝ := fun _ _ x => x

theorem identityDrift_zeroAE_at_diracZero :
    ZeroDriftAE identityDrift (Measure.dirac 0) (Measure.dirac 0) := by
  simp [ZeroDriftAE, identityDrift]

theorem identityDrift_not_zeroDrift_at_diracZero :
    ¬ ZeroDrift identityDrift (Measure.dirac 0) (Measure.dirac 0) := by
  intro h
  have := h 1
  norm_num [identityDrift] at this

/-- A non-injective feature can collapse distinct Dirac laws, so feature-space
matching alone proves only pushforward equality. -/
theorem noninjectiveFeature_collapses_dirac
    {X : Type*} {F : Type*} [MeasurableSpace X] [MeasurableSpace F]
    (φ : X → F) (hφ : Measurable φ) {a b : X} (_hab : a ≠ b)
    (hcollapse : φ a = φ b) :
    featureLaw φ (Measure.dirac a) hφ.aemeasurable =
      featureLaw φ (Measure.dirac b) hφ.aemeasurable := by
  simp [featureLaw, pushforward, Measure.map_dirac' hφ, hcollapse]

/-- Source-law equality can be recovered from feature-law equality under the
independently checkable stronger condition that the feature is a measurable
embedding. -/
theorem sourceMeasure_eq_of_featureLaw_eq
    {X : Type*} {F : Type*} [MeasurableSpace X] [MeasurableSpace F]
    (φ : X → F) (hφ : MeasurableEmbedding φ) (p q : Distribution X)
    (h : featureLaw φ p hφ.measurable.aemeasurable =
      featureLaw φ q hφ.measurable.aemeasurable) : p = q := by
  apply hφ.map_injective
  simpa [featureLaw, pushforward] using h

/-- Two temperature components can cancel exactly even when neither vanishes.
This is the minimal obstruction to applying a single-temperature zero theorem
to the paper's summed multi-temperature field. -/
noncomputable def cancellingTemperatureDrift
    {F : Type*} [AddCommGroup F] (v : F) : ℝ → F :=
  fun τ => if τ = 0 then v else if τ = 1 then -v else 0

theorem aggregateTemperatureDrift_cancels
    {F : Type*} [AddCommGroup F] (v : F) :
    aggregateTemperatureDrift ({0, 1} : Finset ℝ)
      (cancellingTemperatureDrift v) = 0 := by
  simp [aggregateTemperatureDrift, cancellingTemperatureDrift]

theorem aggregateTemperatureDrift_zero_with_nonzero_component
    {F : Type*} [AddCommGroup F] {v : F} (hv : v ≠ 0) :
    aggregateTemperatureDrift ({0, 1} : Finset ℝ)
        (cancellingTemperatureDrift v) = 0 ∧
      cancellingTemperatureDrift v 0 ≠ 0 := by
  exact ⟨aggregateTemperatureDrift_cancels v, by
    simpa [cancellingTemperatureDrift] using hv⟩

end DriftingIdentifiability
