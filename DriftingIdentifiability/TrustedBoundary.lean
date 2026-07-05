import DriftingIdentifiability.Paperaxioms

/-!
# Trusted boundary and theorem targets

This is the only project module permitted to import `Paperaxioms` directly.
It translates the paper interface into research targets without asserting any
identifiability result.
-/

open Filter MeasureTheory Topology

namespace DriftingIdentifiability

open Paper

universe u

/-- A condition proves exact identifiability for `V` when zero drift forces
equality for every pair satisfying that condition.  This is a definition of
the target, never an axiom. -/
def IdentifiesAtZero
    {E : Type u} [MeasurableSpace E] [Zero E]
    (condition : Distribution E → Distribution E → Prop)
    (V : DriftingField E) : Prop :=
  ∀ p q, condition p q → ZeroDrift V p q → p = q

/-- An exact counterexample to a proposed condition. -/
def IsExactCounterexample
    {E : Type u} [MeasurableSpace E] [Zero E]
    (condition : Distribution E → Distribution E → Prop)
    (V : DriftingField E) (p q : Distribution E) : Prop :=
  condition p q ∧ ZeroDrift V p q ∧ p ≠ q

/-- A candidate condition must allow some unequal distributions before the
zero-drift hypothesis is imposed.  This rules out conditions that merely hide
`p = q` (or a logically equivalent statement) in their assumptions. -/
def ConditionAllowsDistinctPair
    {E : Type u} [MeasurableSpace E]
    (condition : Distribution E → Distribution E → Prop) : Prop :=
  ∃ p q, condition p q ∧ p ≠ q

/-- Minimal formal legitimacy check for a candidate condition. -/
def IsLegitimateCondition
    {E : Type u} [MeasurableSpace E]
    (condition : Distribution E → Distribution E → Prop) : Prop :=
  (∃ p q, condition p q) ∧ ConditionAllowsDistinctPair condition

/-- A condition restricted to a model family, useful for finite and numerical
stress tests. -/
def RestrictedCondition
    {E : Type u} [MeasurableSpace E]
    (family : Set (Distribution E))
    (condition : Distribution E → Distribution E → Prop) :
    Distribution E → Distribution E → Prop :=
  fun p q => p ∈ family ∧ q ∈ family ∧ condition p q

/-- Abstract asymptotic target.  `driftSize` and `distributionDistance` must be
specified explicitly, preventing accidental conflation of `V = 0` with
`V → 0` or of measure equality with convergence in an unnamed topology. -/
def AsymptoticallyIdentifies
    {E : Type u} [MeasurableSpace E]
    (condition : Distribution E → Distribution E → Prop)
    (driftSize distributionDistance : Distribution E → Distribution E → ℝ) : Prop :=
  ∀ p (q : ℕ → Distribution E),
    (∀ n, condition p (q n)) →
    Tendsto (fun n => driftSize p (q n)) atTop (𝓝 0) →
    Tendsto (fun n => distributionDistance p (q n)) atTop (𝓝 0)

/-- Metadata for a proposed condition.  Mathematical acceptance still
requires a proof of `IsLegitimateCondition`, counterexample testing, a written
proof, and finally a Lean theorem. -/
structure CandidateSpec
    (E : Type u) [MeasurableSpace E] where
  name : String
  condition : Distribution E → Distribution E → Prop
  rationale : String

def CandidateSpec.IsLegitimate
    {E : Type u} [MeasurableSpace E] (candidate : CandidateSpec E) : Prop :=
  IsLegitimateCondition candidate.condition

end DriftingIdentifiability
