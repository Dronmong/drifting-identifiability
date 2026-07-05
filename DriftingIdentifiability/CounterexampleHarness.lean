import DriftingIdentifiability.TrustedBoundary

/-!
# Counterexample and finite-model harness

Definitions in this module make the mandatory stress-test phase precise.  They
do not certify any candidate condition.
-/

open scoped BigOperators

namespace DriftingIdentifiability

open Paper

universe u

/-- A proposed condition is refuted when an unequal zero-drift pair satisfies
it. -/
def RefutesCondition
    {E : Type u} [MeasurableSpace E] [Zero E]
    (condition : Distribution E → Distribution E → Prop)
    (V : DriftingField E) : Prop :=
  ∃ p q, IsExactCounterexample condition V p q

/-- Passing a test family means only that no counterexample occurs inside that
family.  It is deliberately weaker than a global theorem. -/
def PassesFamilyTest
    {E : Type u} [MeasurableSpace E] [Zero E]
    (family : Set (Distribution E))
    (condition : Distribution E → Distribution E → Prop)
    (V : DriftingField E) : Prop :=
  ¬RefutesCondition (RestrictedCondition family condition) V

/-- Probability coefficients for finite-basis and finite-support tests. -/
structure FiniteProbabilityVector (m : ℕ) where
  weight : Fin m → ℝ
  nonnegative : ∀ i, 0 ≤ weight i
  normalized : ∑ i, weight i = 1

/-- The `2 × 2` coefficient minor from Appendix C.1. -/
def coefficientMinor {m : ℕ}
    (a b : Fin m → ℝ) (i j : Fin m) : ℝ :=
  a i * b j - a j * b i

/-- All coefficient minors vanish, the intermediate conclusion supplied by
Appendix C.1 under its nondegeneracy hypotheses. -/
def AllCoefficientMinorsZero {m : ℕ} (a b : Fin m → ℝ) : Prop :=
  ∀ i j, coefficientMinor a b i j = 0

/-- A finite witness that two coefficient vectors differ. -/
def CoefficientsDistinct {m : ℕ} (a b : Fin m → ℝ) : Prop :=
  ∃ i, a i ≠ b i

/-- Data recorded for a computational or hand-built counterexample. -/
structure CounterexampleRecord where
  candidateName : String
  model : String
  witness : String
  violatedClaim : String

end DriftingIdentifiability
