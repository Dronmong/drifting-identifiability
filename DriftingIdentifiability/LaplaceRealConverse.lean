import DriftingIdentifiability.LaplaceGeneralConverseNowhereDense
import DriftingIdentifiability.LaplaceACFinal

/-!
# Milestone 6: the assembled one-dimensional Laplace raw-drift converse

This file is the single named assembly promised by the roadmap
(`LaplaceGeneralConverseRoadmap.md`, Milestone 6): one condition and one
`IdentifiesAtZero` theorem recording exactly the class on which the
one-dimensional raw Laplace mean-shift converse is machine-checked today.

The condition is the disjunction of the two certified regimes:

* **nowhere-dense-style supports** (Milestone 3): the combined measure has
  right-dense zero-mass gaps — this covers finite mixtures, countable atomic
  laws, Cantor-type singular supports, with no moment hypotheses at all;
* **unrestricted continuous densities** (Milestone 5 closure): continuous
  Lebesgue densities with two-sided exponential first moments and a `p` first
  moment — with no hypothesis on the mean-shift zero set.

The two regimes are genuinely complementary: the first allows atoms and
singular parts but no support interior; the second demands absolute
continuity but allows full-interval supports.  The still-open frontier is
their common refinement (general L¹ densities, and mixed measures with atoms
or singular parts on interval supports).
-/

open MeasureTheory

namespace DriftingIdentifiability

open Paper

/-- Combined condition for the assembled real-line Laplace converse: either
right-dense zero-mass gaps for the combined measure (the Milestone-3
nowhere-dense-style class), or the unrestricted continuous-density package
(the Milestone-5 closure class). -/
def LaplaceRealConverseCondition (τ : ℝ) (p q : Measure ℝ) : Prop :=
  (IsProbabilityMeasure p ∧ IsProbabilityMeasure q ∧ RightDenseZeroMassGaps p q) ∨
    LaplaceACContinuousDensityCondition τ p q

/-- **Assembled 1-d Laplace raw-drift converse (Milestone 6).**  Zero raw
Laplace mean-shift drift identifies probability measures on ℝ that satisfy
either certified regime: nowhere-dense-style supports, or unrestricted
continuous densities with exponential moments. -/
theorem laplaceZeroDrift_identifies_real
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    IdentifiesAtZero
      (LaplaceRealConverseCondition τ)
      (meanShiftDrift (laplaceKernel τ)) := by
  intro p q hcond hzero
  rcases hcond with ⟨hpProb, hqProb, hgaps⟩ | hac
  · letI : IsProbabilityMeasure p := hpProb
    letI : IsProbabilityMeasure q := hqProb
    exact laplaceZeroDrift_identifies_of_rightDense_zeroMassGaps
      τ hτ p q hzero hgaps
  · exact laplaceAC_identifiesAtZero_of_continuousDensity τ hτ p q hac hzero

/-- The assembled condition admits a concrete distinct pair (the Gaussian
witness enters through the continuous-density disjunct). -/
theorem laplaceRealConverseCondition_allowsDistinctPair
    (τ : ℝ) :
    ConditionAllowsDistinctPair (LaplaceRealConverseCondition τ) := by
  obtain ⟨p, q, hcond, hne⟩ :=
    laplaceACContinuousDensityCondition_allowsDistinctPair τ
  exact ⟨p, q, Or.inr hcond, hne⟩

/-- The assembled condition is formally legitimate: inhabited and allowing a
distinct pair before any zero-drift hypothesis. -/
theorem laplaceRealConverseCondition_isLegitimate
    (τ : ℝ) :
    IsLegitimateCondition (LaplaceRealConverseCondition τ) := by
  constructor
  · obtain ⟨p, q, hcond, _⟩ :=
      laplaceACContinuousDensityCondition_allowsDistinctPair τ
    exact ⟨p, q, Or.inr hcond⟩
  · exact laplaceRealConverseCondition_allowsDistinctPair τ

end DriftingIdentifiability
