import DriftingIdentifiability.LaplaceFoliationMaximum
import DriftingIdentifiability.LaplaceFoliationEndgame

/-!
# The general Euclidean Laplace converse

**The target theorem of the ℝⁿ program** (`LaplaceRnRoadmap.md` §0): zero
ℓ²-Laplace mean-shift drift identifies arbitrary probability measures on every
finite-dimensional Euclidean space, with no support, moment, density, atom,
radiality, or slack hypothesis.

The proof composes the G4 chain: the global maximum principle for the
foliation defect (`LaplaceFoliationMaximum.lean`) forces `H ≡ 0`, hence the
normalizer ratio `Z_p/Z_q` is globally constant, and the certified
gluing/mass endgame (`LaplaceFoliationEndgame.lean`) — finite-measure Laplace
smoothing injectivity via Bernstein subordination plus total mass — yields
`p = q`.
-/

open MeasureTheory

namespace DriftingIdentifiability

open Paper

/-- **The general Euclidean Laplace converse.**  Zero Laplace mean-shift
drift identifies arbitrary probability measures on any finite-dimensional
Euclidean space. -/
theorem laplaceZeroDrift_identifies_euclidean
    {ι : Type*} [Fintype ι] (τ : ℝ) (hτ : 0 < τ)
    (p q : Measure (EuclideanSpace ℝ ι))
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    p = q :=
  eq_of_laplaceNormalizerRatio_isLocallyConstant_euclideanSpace hτ p q
    (laplaceNormalizerRatio_isLocallyConstant hτ p q hzero)

/-- The roadmap-form statement on `ℝⁿ` for every finite `n`. -/
theorem laplaceZeroDrift_identifies_rn
    (τ : ℝ) (hτ : 0 < τ) (n : ℕ)
    (p q : Measure (EuclideanSpace ℝ (Fin n)))
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    p = q :=
  laplaceZeroDrift_identifies_euclidean τ hτ p q hzero

end DriftingIdentifiability
