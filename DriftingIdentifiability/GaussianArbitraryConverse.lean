import DriftingIdentifiability.GaussianScoreRecovery

/-!
# The Gaussian-kernel arbitrary-target converse (Frontier C)

The mathematical core is already proved, axiom-free, in
`GaussianScoreRecovery.lean`: `gaussianMeanShiftDrift_identifiesAtZero` shows
that for **arbitrary** probability measures on a finite-dimensional real
inner-product space, pointwise zero Gaussian mean-shift drift forces `p = q`.
The engine is the exact score identity `∇log Zₚ = σ⁻²·meanShiftₚ`
(`hasFDerivAt_log_gaussianKernelNormalizer`): zero drift makes the two smoothed
log-normalizers differ by a constant, so `Zₚ = c·Zₚ` with `c = 1`, and the
charFun injectivity of Gaussian convolution (`gaussianKernelNormalizer_injective`)
finishes.  This is *why* the Gaussian case is elementary while the Laplace case
needed the whole companion/Abel machinery: for the Gaussian kernel the mean
shift is exactly the score of the smoothed measure, so zero drift pins the
smooth immediately; for the Laplace kernel `Z'` is not proportional to the
displacement integral.

This file adds the thin surface that brings the Gaussian result to parity with
the Laplace headline `laplaceZeroDrift_identifies`: a direct `p = q` form and an
explicit legitimacy witness for the (unconditional) `BothProbability` condition.
Design record: `LaplaceEndgame.md`, Frontier C.
-/

open MeasureTheory

namespace DriftingIdentifiability

open Paper

universe u

variable {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
  [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [BorelSpace E] [SecondCountableTopology E]

/-- **The unconditional Gaussian-kernel converse, direct form.**  Zero raw
Gaussian mean-shift drift identifies arbitrary probability measures on a
finite-dimensional real inner-product space — no atomlessness, no moment, no
density.  (Parity headline for `laplaceZeroDrift_identifies`; the underlying
work is `gaussianMeanShiftDrift_identifiesAtZero`.) -/
theorem gaussianZeroDrift_identifies
    (σ : ℝ) (hσ : ValidBandwidth σ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (gaussianKernel σ)) p q) :
    p = q :=
  gaussianMeanShiftDrift_identifiesAtZero σ hσ p q ⟨inferInstance, inferInstance⟩ hzero

omit [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  [FiniteDimensional ℝ E] [BorelSpace E] [SecondCountableTopology E] in
/-- The unconditional `BothProbability` condition is formally legitimate: it is
satisfiable and admits distinct pairs before any zero-drift hypothesis. -/
theorem bothProbability_isLegitimate
    [Nontrivial E] [MeasurableSpace.SeparatesPoints E] :
    IsLegitimateCondition (BothProbability (E := E)) := by
  obtain ⟨p, q, hpq, _⟩ := bothProbability_allowsDistinctPair (E := E)
  exact ⟨⟨p, q, hpq⟩, bothProbability_allowsDistinctPair⟩

end DriftingIdentifiability
