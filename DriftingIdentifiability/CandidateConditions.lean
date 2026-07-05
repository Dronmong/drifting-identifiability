import DriftingIdentifiability.CounterexampleHarness

/-!
# Candidate conditions

New candidates belong here only after they have a mathematically meaningful
definition.  Do not place axioms, theorem stubs, or `sorry` in this file.

For each candidate `c`, the expected progression is:

1. define `c : CandidateSpec E`;
2. prove `c.IsLegitimate` (in particular, exhibit a distinct pair satisfying
   the condition before zero drift is assumed);
3. attempt to prove `RefutesCondition c.condition V` on small/discrete models;
4. write a complete proof in `WrittenProof.md`;
5. only then prove `IdentifiesAtZero c.condition V` in Lean.
-/

namespace DriftingIdentifiability
namespace CandidateConditions

/-- Research stages are metadata, not evidence that a condition is correct. -/
inductive ResearchStage where
  | proposed
  | stressTesting
  | writtenProof
  | leanProof
  | accepted
  | rejected
  deriving DecidableEq, Repr

/-- Human-readable tracking information for a candidate. -/
structure CandidateMetadata where
  name : String
  stage : ResearchStage
  motivation : String
  testsAttempted : List String := []

/-- A candidate may advance to proof-writing only after legitimacy is proved
and no counterexample has been found in the declared test family.  This does
not replace the global theorem. -/
def ReadyForWrittenProof
    {E : Type*} [MeasurableSpace E] [Zero E]
    (family : Set (Paper.Distribution E))
    (candidate : CandidateSpec E) (V : Paper.DriftingField E) : Prop :=
  candidate.IsLegitimate ∧ PassesFamilyTest family candidate.condition V

end CandidateConditions
end DriftingIdentifiability
