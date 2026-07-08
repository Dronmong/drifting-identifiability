import DriftingIdentifiability.PopulationIdentifiability

/-!
# Finite-population candidates

`finiteBasisCandidate` records only shared finite-family membership and is a
useful preliminary model-family candidate. The accepted theorem-level candidate
is `finitePopulationMeanShiftCandidate`: its pair condition also carries a
complete `PopulationMeanShiftFiniteSetup`, so the kernel, probes, regularity,
integrability, and positive frame certificate consumed by the promoted theorem
are all present and independently checkable before zero drift is assumed.
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

/-- Preliminary model-family candidate. This records shared finite-basis
membership, but by itself does not imply identifiability for an arbitrary
kernel or probe family. -/
def finiteBasisCandidate (basis : ProbabilityDensityBasis reference m) :
    CandidateSpec E where
  name := "shared finite probability-density family"
  condition := FiniteBasisFamily basis
  rationale :=
    "A finite family exposes coefficient algebra; a separate theorem-level \
      condition must supply regularity and a positive interaction frame."

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

section AcceptedPopulationCandidate

variable [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- Complete independently checkable pair condition for a fixed mean-shift
kernel. A witness is a finite population setup whose represented target/model
measures are exactly `p` and `q` and whose kernel is the fixed field kernel.

The setup stores no zero-drift hypothesis and no equality or injectivity
conclusion. In particular, its positive frame bound is a concrete finite linear
algebra certificate rather than a restatement of identifiability. -/
def FinitePopulationMeanShiftCondition
    (kernel : E → E → ℝ) : Distribution E → Distribution E → Prop :=
  fun p q => ∃ (m N : ℕ) (setup : PopulationMeanShiftFiniteSetup E m N),
    setup.kernel = kernel ∧
      setup.targetMeasure = p ∧ setup.modelMeasure = q

/-- Accepted candidate corresponding exactly to the promoted finite population
mean-shift theorem. -/
def finitePopulationMeanShiftCandidate (kernel : E → E → ℝ) :
    CandidateSpec E where
  name := "finite population mean-shift setup with certified interactions"
  condition := FinitePopulationMeanShiftCondition kernel
  rationale :=
    "Finite probability-density membership plus explicit regularity, \
      integrability, probes, and a positive interaction frame turns zero \
      normalized mean-shift drift into coefficient and measure equality."

/-- The complete candidate condition proves the project's canonical exact
target for the fixed paper mean-shift field. -/
theorem finitePopulationMeanShiftCandidate_identifiesAtZero
    (kernel : E → E → ℝ) :
    IdentifiesAtZero
      (finitePopulationMeanShiftCandidate kernel).condition
      (meanShiftDrift kernel) := by
  intro p q hcondition hzero
  obtain ⟨m, N, setup, hkernel, hp, hq⟩ := hcondition
  have hzeroSetup :
      ZeroDrift (meanShiftDrift setup.kernel)
        setup.targetMeasure setup.modelMeasure := by
    simpa [hkernel, hp, hq] using hzero
  have heq := finitePopulationMeanShift_identifies setup hzeroSetup
  simpa [hp, hq] using heq

/-- Legitimacy of the accepted candidate is witnessed entirely before zero
drift: any complete setup representing two distinct measures supplies a
distinct admissible pair. -/
theorem finitePopulationMeanShiftCandidate_isLegitimate
    {m N : ℕ} (setup : PopulationMeanShiftFiniteSetup E m N)
    (hdistinct : setup.targetMeasure ≠ setup.modelMeasure) :
    (finitePopulationMeanShiftCandidate setup.kernel).IsLegitimate := by
  constructor
  · exact ⟨setup.targetMeasure, setup.modelMeasure,
      m, N, setup, rfl, rfl, rfl⟩
  · exact ⟨setup.targetMeasure, setup.modelMeasure,
      ⟨m, N, setup, rfl, rfl, rfl⟩, hdistinct⟩

end AcceptedPopulationCandidate

end CandidateConditions
end DriftingIdentifiability
