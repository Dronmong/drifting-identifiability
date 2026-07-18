import DriftingIdentifiability.LaplaceFoliationCancellation
import DriftingIdentifiability.LaplaceRadialFourier

/-!
# General Euclidean Laplace converse, G4/P3: gluing and mass endgame

This file isolates the part of G4 that becomes completely mechanical once the
geometric P1/P2 argument has shown that the positive normalizer ratio is
locally constant.  Connectedness glues the local constants, finite-measure
Laplace smoothing injectivity identifies `p` with a scalar multiple of `q`,
and probability mass forces that scalar to be one.

The local-constancy premise is intentionally visible.  The theorem is an
endgame interface, not the still-open arbitrary-measure converse: P2/P3 must
derive this premise from zero drift before it can be removed.
-/

open MeasureTheory Topology

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasureSpace E] [BorelSpace E] [CompleteSpace E]
  [SecondCountableTopology E] [PreconnectedSpace E]

/-- On a preconnected space, local constancy of the normalizer ratio glues to
one global proportionality constant. -/
theorem laplaceKernelNormalizer_proportional_of_ratio_isLocallyConstant
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hlocal : IsLocallyConstant (laplaceNormalizerRatio τ p q)) :
    ∃ c : ℝ, 0 < c ∧ ∀ x : E,
      kernelNormalizer (laplaceKernel τ) p x =
        c * kernelNormalizer (laplaceKernel τ) q x := by
  let c := laplaceNormalizerRatio τ p q 0
  have hc : 0 < c := laplaceNormalizerRatio_pos hτ p q 0
  refine ⟨c, hc, ?_⟩
  intro x
  have hr : laplaceNormalizerRatio τ p q x = c :=
    hlocal.apply_eq_of_preconnectedSpace x 0
  unfold laplaceNormalizerRatio at hr
  exact (div_eq_iff (laplaceKernelNormalizer_pos q τ hτ x).ne').mp hr

/-- **G4 mass endgame.**  Local constancy of the normalizer ratio identifies
two probability laws whenever finite-measure Laplace smoothing is injective.

This packages the dimension-free `c • q` argument used by the radial
converses.  It has no zero-drift premise because all drift geometry has already
been compressed into `hlocal`. -/
theorem eq_of_laplaceNormalizerRatio_isLocallyConstant
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hL4 : LaplaceSmoothingInjective E τ)
    (hlocal : IsLocallyConstant (laplaceNormalizerRatio τ p q)) :
    p = q := by
  obtain ⟨c, hc, hZ⟩ :=
    laplaceKernelNormalizer_proportional_of_ratio_isLocallyConstant
      hτ p q hlocal
  have hscaledFinite : IsFiniteMeasure ((ENNReal.ofReal c) • q) := by
    constructor
    rw [Measure.smul_apply, smul_eq_mul]
    exact ENNReal.mul_lt_top ENNReal.ofReal_lt_top (measure_lt_top q _)
  have hZscaled : ∀ x : E,
      kernelNormalizer (laplaceKernel τ) p x =
        kernelNormalizer (laplaceKernel τ) ((ENNReal.ofReal c) • q) x := by
    intro x
    unfold kernelNormalizer
    rw [integral_smul_measure, ENNReal.toReal_ofReal hc.le, smul_eq_mul]
    exact hZ x
  have hpScaled : p = (ENNReal.ofReal c) • q :=
    hL4 p ((ENNReal.ofReal c) • q) inferInstance hscaledFinite hZscaled
  have hmass : ENNReal.ofReal c = 1 := by
    have hpMass : p Set.univ = 1 := measure_univ
    rw [hpScaled, Measure.smul_apply, smul_eq_mul, measure_univ, mul_one] at hpMass
    exact hpMass
  rw [hpScaled, hmass, one_smul]

/-- Fully discharged Euclidean specialization of the G4 gluing/mass endgame.
Only the geometric local-constancy conclusion remains as an input. -/
theorem eq_of_laplaceNormalizerRatio_isLocallyConstant_euclideanSpace
    {ι : Type*} [Fintype ι] {τ : ℝ} (hτ : 0 < τ)
    (p q : Measure (EuclideanSpace ℝ ι))
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hlocal : IsLocallyConstant (laplaceNormalizerRatio τ p q)) :
    p = q := by
  exact eq_of_laplaceNormalizerRatio_isLocallyConstant hτ p q
    (laplaceSmoothingInjective_euclideanSpace τ hτ) hlocal

end DriftingIdentifiability
