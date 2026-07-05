import DriftingIdentifiability.PopulationIdentifiability

/-!
# Accepted finite-population candidate

This file records the actual model-family condition used by the promoted
paper-native theorem. Kernel regularity and the interaction frame bound are
model-level obligations carried by `PopulationMeanShiftFiniteSetup`; the pair
condition below says only that both laws lie in the same finite probability
density family.
-/

namespace DriftingIdentifiability
namespace CandidateConditions

open Paper
open PaperFiniteIdentifiability

universe u

variable {E : Type u} [MeasurableSpace E]
  {reference : Distribution E} {m : ℕ}

/-- Both distributions are mixtures in the same finite probability-density
basis. -/
def FiniteBasisFamily (basis : ProbabilityDensityBasis reference m) :
    Distribution E → Distribution E → Prop :=
  fun p q => ∃ a b : FiniteProbabilityVector m,
    p = basis.basisMeasure a ∧ q = basis.basisMeasure b

/-- Registered candidate corresponding to the verified population theorem. -/
def finiteBasisCandidate (basis : ProbabilityDensityBasis reference m) :
    CandidateSpec E where
  name := "finite probability-density family with conditioned interactions"
  condition := FiniteBasisFamily basis
  rationale :=
    "A frame bound identifies coefficients; mean-shift regularity connects the population field."

/-- Legitimacy is independent of zero drift: two unequal basis-induced
mixtures witness a distinct admissible pair. -/
theorem finiteBasisCandidate_isLegitimate
    (basis : ProbabilityDensityBasis reference m)
    (hdistinct : ∃ a b : FiniteProbabilityVector m,
      basis.basisMeasure a ≠ basis.basisMeasure b) :
    (finiteBasisCandidate basis).IsLegitimate := by
  obtain ⟨a, b, hab⟩ := hdistinct
  constructor
  · exact ⟨basis.basisMeasure a, basis.basisMeasure b, a, b, rfl, rfl⟩
  · exact ⟨basis.basisMeasure a, basis.basisMeasure b,
      ⟨a, b, rfl, rfl⟩, hab⟩

/-- A convenient legitimacy criterion using the first two simplex vertices. -/
theorem finiteBasisCandidate_isLegitimate_of_two_vertices
    (basis : ProbabilityDensityBasis reference m) (hm : 2 ≤ m)
    (hdistinct :
      basis.basisMeasure (stdBasisProbVector m ⟨0, by omega⟩) ≠
      basis.basisMeasure (stdBasisProbVector m ⟨1, by omega⟩)) :
    (finiteBasisCandidate basis).IsLegitimate := by
  apply finiteBasisCandidate_isLegitimate basis
  exact ⟨stdBasisProbVector m ⟨0, by omega⟩,
    stdBasisProbVector m ⟨1, by omega⟩, hdistinct⟩

end CandidateConditions
end DriftingIdentifiability
