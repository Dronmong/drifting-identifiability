import DriftingIdentifiability.LaplaceGeneralConverseNowhereDense
import DriftingIdentifiability.LaplaceAtomlessConverse

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
* **atomless laws** (the post-Milestone-5 upgrade,
  `LaplaceAtomlessConverse.lean`): `p` and `q` atomless with a `p` first
  moment — no density regularity, no exponential moments, and no hypothesis
  on the mean-shift zero set.  This strictly subsumes the earlier
  continuous-density regime
  (`laplaceAtomlessCondition_of_continuousDensityCondition`).

Both regimes — hence the combined condition — are bandwidth-free: the same
class of measure pairs is identified for EVERY valid bandwidth `τ`.

The two regimes are genuinely complementary: the first allows atoms and
singular parts but no support interior; the second allows full-interval
supports but no atoms.  The still-open frontier is their common refinement:
measures with atoms on interval supports (and dropping the `p` first
moment). -/

open MeasureTheory ProbabilityTheory

namespace DriftingIdentifiability

open Paper

/-- Combined condition for the assembled real-line Laplace converse: either
right-dense zero-mass gaps for the combined measure (the Milestone-3
nowhere-dense-style class), or both laws atomless with a `p` first moment
(the alignment-defect class). -/
def LaplaceRealConverseCondition (p q : Measure ℝ) : Prop :=
  (IsProbabilityMeasure p ∧ IsProbabilityMeasure q ∧ RightDenseZeroMassGaps p q) ∨
    LaplaceAtomlessCondition p q

/-- **Assembled 1-d Laplace raw-drift converse (Milestone 6).**  Zero raw
Laplace mean-shift drift identifies probability measures on ℝ that satisfy
either certified regime: nowhere-dense-style supports, or atomlessness with
a `p` first moment. -/
theorem laplaceZeroDrift_identifies_real
    (τ : ℝ) (hτ : ValidBandwidth τ) :
    IdentifiesAtZero
      LaplaceRealConverseCondition
      (meanShiftDrift (laplaceKernel τ)) := by
  intro p q hcond hzero
  rcases hcond with ⟨hpProb, hqProb, hgaps⟩ | hatomless
  · letI : IsProbabilityMeasure p := hpProb
    letI : IsProbabilityMeasure q := hqProb
    exact laplaceZeroDrift_identifies_of_rightDense_zeroMassGaps
      τ hτ p q hzero hgaps
  · exact laplaceZeroDrift_identifiesAtZero_of_noAtoms τ hτ p q hatomless hzero

/-- The earlier continuous-density disjunct is subsumed: any pair satisfying
the Milestone-5 continuous-density condition satisfies the assembled
condition through its atomless disjunct. -/
theorem laplaceRealConverseCondition_of_continuousDensity
    {τ : ℝ} {p q : Measure ℝ}
    (h : LaplaceACContinuousDensityCondition τ p q) :
    LaplaceRealConverseCondition p q :=
  Or.inr (laplaceAtomlessCondition_of_continuousDensityCondition h)

/-- The assembled condition admits a concrete distinct pair (the Gaussian
witness enters through the atomless disjunct). -/
theorem laplaceRealConverseCondition_allowsDistinctPair :
    ConditionAllowsDistinctPair LaplaceRealConverseCondition := by
  obtain ⟨p, q, hcond, hne⟩ := laplaceAtomlessCondition_allowsDistinctPair
  exact ⟨p, q, Or.inr hcond, hne⟩

/-- The assembled condition is formally legitimate: inhabited and allowing a
distinct pair before any zero-drift hypothesis. -/
theorem laplaceRealConverseCondition_isLegitimate :
    IsLegitimateCondition LaplaceRealConverseCondition := by
  constructor
  · exact ⟨gaussianReal 0 (1 : NNReal), gaussianReal 1 (1 : NNReal),
      Or.inr laplaceAtomlessCondition_gaussianPair⟩
  · exact laplaceRealConverseCondition_allowsDistinctPair

end DriftingIdentifiability
