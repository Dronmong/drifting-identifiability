import DriftingIdentifiability.FiniteSampleBridge
import DriftingIdentifiability.Algorithm2SNIS

/-!
# Column-reweighted Algorithm-2 limiting mean-shift field

With fixed anchors and `selfMask = false`, `Algorithm2SNIS.lean` proves that
the row-softmax factor cancels inside Algorithm 2's positive/negative
centroids.  The limiting population centroids therefore use the modified,
column-reweighted kernel

```text
k_col(x,y) = sqrt(k_alg(x,y) * k_alg(x,y) / sum_r k_alg(anchor_r,y)).
```

This file makes that modified kernel a first-class population mean-shift
kernel and reuses the finite-basis/frame infrastructure from
`PopulationIdentifiability.lean` and `FiniteSampleBridge.lean`.  The important
guard rail is explicit: identifiability is proved only under a frame bound for
the modified interaction vectors, not for the original bare paper kernel.
-/

open scoped BigOperators
open MeasureTheory

namespace DriftingIdentifiability

namespace Algorithm2

open Paper

universe u

section ColumnReweightedKernel

variable {E : Type u} [NormedAddCommGroup E]
variable {Nx Npos Nneg : ℕ}

/-- The population kernel targeted by the fixed-anchor, no-self-mask Algorithm-2
centroids after the common row-normalization factor cancels. -/
noncomputable def columnReweightedKernel
    (anchors : Fin Nx → E) (temperature : ℝ) (x y : E) : ℝ :=
  Real.sqrt
    (algorithm2Kernel temperature x y *
      (algorithm2Kernel temperature x y /
        algorithm2ColumnKernelMass anchors temperature y))

/-- At an anchor, the population kernel is exactly the SNIS weight already used
in `Algorithm2SNIS.lean`. -/
theorem columnReweightedKernel_apply_anchor
    (anchors : Fin Nx → E) (temperature : ℝ) (i : Fin Nx) (y : E) :
    columnReweightedKernel anchors temperature (anchors i) y =
      algorithm2ColumnReweightedWeight anchors temperature i y := rfl

theorem columnReweightedKernel_nonneg
    (anchors : Fin Nx → E) (temperature : ℝ) (x y : E) :
    0 ≤ columnReweightedKernel anchors temperature x y := by
  unfold columnReweightedKernel
  exact Real.sqrt_nonneg _

theorem columnReweightedKernel_pos [Nonempty (Fin Nx)]
    (anchors : Fin Nx → E) (temperature : ℝ) (x y : E) :
    0 < columnReweightedKernel anchors temperature x y := by
  unfold columnReweightedKernel
  exact Real.sqrt_pos.2
    (mul_pos (algorithm2Kernel_pos temperature x y)
      (div_pos (algorithm2Kernel_pos temperature x y)
        (algorithm2ColumnKernelMass_pos anchors temperature y)))

variable [NormedSpace ℝ E]

/-- The normalized centroid-difference estimator isolated from Algorithm 2's
raw mass-scaled drift.  This is the statistic that converges to the
column-reweighted population mean-shift field. -/
noncomputable def noMaskCentroidDrift
    (anchors : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) : Fin Nx → E :=
  fun i =>
    algorithm2PositiveCentroid anchors yPos yNeg temperature (fun _ _ => false) i -
      algorithm2NegativeCentroid anchors yPos yNeg temperature (fun _ _ => false) i

@[simp]
theorem noMaskCentroidDrift_apply
    (anchors : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) :
    noMaskCentroidDrift anchors yPos yNeg temperature i =
      algorithm2PositiveCentroid anchors yPos yNeg temperature (fun _ _ => false) i -
        algorithm2NegativeCentroid anchors yPos yNeg temperature (fun _ _ => false) i := rfl

/-- Vector form of the zero-set equivalence: for nonempty positive and negative
batches, raw no-mask Algorithm-2 drift vanishes at every anchor iff the
normalized centroid-difference estimator vanishes at every anchor. -/
theorem algorithm2Drift_false_eq_zero_iff_noMaskCentroidDrift_eq_zero
    [Nonempty (Fin Npos)] [Nonempty (Fin Nneg)]
    (anchors : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) :
    (∀ i : Fin Nx,
        algorithm2Drift anchors yPos yNeg temperature (fun _ _ => false) i = 0) ↔
      noMaskCentroidDrift anchors yPos yNeg temperature = 0 := by
  constructor
  · intro h
    funext i
    exact (algorithm2Drift_false_eq_zero_iff_centroidDiff_eq_zero
      anchors yPos yNeg temperature i).mp (h i)
  · intro h i
    exact (algorithm2Drift_false_eq_zero_iff_centroidDiff_eq_zero
      anchors yPos yNeg temperature i).mpr (by simpa [noMaskCentroidDrift] using congrFun h i)

end ColumnReweightedKernel

end Algorithm2

namespace PaperFiniteIdentifiability

open Paper

universe u

variable {E : Type u} [MeasurableSpace E]
  [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- A finite-basis population setup for the column-reweighted kernel targeted by
fixed-anchor, no-self-mask Algorithm 2.

All analytic and conditioning assumptions are explicit.  In particular,
`frameBound` is for
`meanShiftInteractionKernel (Algorithm2.columnReweightedKernel anchors temperature)`;
this is intentionally not the same hypothesis as a frame bound for the bare
paper/Algorithm-2 exponential kernel. -/
structure ColumnReweightedMeanShiftFiniteSetup
    (E : Type u) [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (m N : ℕ) where
  reference : Distribution E
  refProb : IsProbabilityMeasure reference
  basis : ProbabilityDensityBasis reference m
  anchors : Fin N → E
  temperature : ℝ
  a : FiniteProbabilityVector m
  b : FiniteProbabilityVector m
  meanShiftRegular : ∀ n, MeanShiftRegularAt
    (Algorithm2.columnReweightedKernel anchors temperature)
    (basis.basisMeasure a) (basis.basisMeasure b) (anchors n)
  interactionIntegrable : ∀ n, Integrable
    (fun y : E × E =>
      meanShiftInteractionKernel
        (Algorithm2.columnReweightedKernel anchors temperature)
        (anchors n) y.1 y.2)
    ((basis.basisMeasure a).prod (basis.basisMeasure b))
  basisInteractionIntegrable : ∀ i j n, Integrable
    (fun y : E × E =>
      (basis.density i y.1 * basis.density j y.2) •
        meanShiftInteractionKernel
          (Algorithm2.columnReweightedKernel anchors temperature)
          (anchors n) y.1 y.2)
    (reference.prod reference)
  frameConstant : ℝ
  frameBound : InteractionFrameBound
    (inducedInteractionVector reference
      (meanShiftInteractionKernel
        (Algorithm2.columnReweightedKernel anchors temperature))
      basis.density anchors) frameConstant

namespace ColumnReweightedMeanShiftFiniteSetup

variable {m N : ℕ}

noncomputable def toPopulationSetup
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N) :
    PopulationMeanShiftFiniteSetup E m N where
  reference := setup.reference
  refProb := setup.refProb
  basis := setup.basis
  kernel := Algorithm2.columnReweightedKernel setup.anchors setup.temperature
  probes := setup.anchors
  a := setup.a
  b := setup.b
  meanShiftRegular := setup.meanShiftRegular
  interactionIntegrable := setup.interactionIntegrable
  basisInteractionIntegrable := setup.basisInteractionIntegrable
  frameConstant := setup.frameConstant
  frameBound := setup.frameBound

noncomputable def targetMeasure
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N) : Distribution E :=
  setup.toPopulationSetup.targetMeasure

noncomputable def modelMeasure
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N) : Distribution E :=
  setup.toPopulationSetup.modelMeasure

/-- The deterministic population field that the no-mask Algorithm-2 centroid
estimator targets at the fixed anchors. -/
noncomputable def modifiedProbeDrift
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N) : Fin N → E :=
  setup.toPopulationSetup.normalizedProbeDrift

noncomputable def normalizerProduct
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N) (n : Fin N) : ℝ :=
  setup.toPopulationSetup.normalizerProduct n

noncomputable def modifiedProbeDriftEnergy
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N) : ℝ :=
  probeDriftEnergy setup.modifiedProbeDrift

theorem modifiedProbeDriftEnergy_eq_zero_iff
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N) :
    setup.modifiedProbeDriftEnergy = 0 ↔ ∀ n, setup.modifiedProbeDrift n = 0 :=
  probeDriftEnergy_eq_zero_iff setup.modifiedProbeDrift

/-- Deterministic identifiability for the column-reweighted Algorithm-2 limiting
field under the modified-kernel frame condition. -/
theorem identifies_of_probeZero
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N)
    (hzero : ∀ n, setup.modifiedProbeDrift n = 0) :
    setup.targetMeasure = setup.modelMeasure := by
  simpa [toPopulationSetup, targetMeasure, modelMeasure, modifiedProbeDrift]
    using finitePopulationMeanShift_identifies_of_probeZero
      setup.toPopulationSetup hzero

/-- Finite-loss form for the column-reweighted Algorithm-2 limiting field. -/
theorem identifies_of_probeEnergy_eq_zero
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N)
    (henergy : setup.modifiedProbeDriftEnergy = 0) :
    setup.targetMeasure = setup.modelMeasure :=
  setup.identifies_of_probeZero
    (setup.modifiedProbeDriftEnergy_eq_zero_iff.mp henergy)

/-- Coefficient stability for the column-reweighted Algorithm-2 limiting field. -/
theorem coefficientStability
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N)
    {B : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n, |setup.normalizerProduct n| ≤ B) :
    (∑ i, |setup.a.weight i - setup.b.weight i|) ≤
      (2 * B / setup.frameConstant) * ‖setup.modifiedProbeDrift‖ := by
  simpa [toPopulationSetup, normalizerProduct, modifiedProbeDrift]
    using setup.toPopulationSetup.coefficientStability hB0 hnormalizer

/-- Deterministic estimator bridge specialized to the modified Algorithm-2
population field.  `Vhat` can be instantiated with
`Algorithm2.noMaskCentroidDrift anchors yPos yNeg temperature`. -/
theorem coefficientStability_of_estimate
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N)
    {B ε : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n, |setup.normalizerProduct n| ≤ B)
    (Vhat : Fin N → E)
    (hclose : ‖setup.modifiedProbeDrift - Vhat‖ ≤ ε) :
    (∑ i, |setup.a.weight i - setup.b.weight i|) ≤
      (2 * B / setup.frameConstant) * (‖Vhat‖ + ε) := by
  simpa [toPopulationSetup, normalizerProduct, modifiedProbeDrift]
    using setup.toPopulationSetup.coefficientStability_of_estimate
      hB0 hnormalizer Vhat hclose

/-- High-probability finite-sample bridge for any random estimator of the
column-reweighted limiting field.  The SNIS lemmas in `Algorithm2SNIS.lean`
provide the natural way to bound the mean-square term for no-mask Algorithm 2. -/
theorem estimate_failure_le_meanSquare
    (setup : ColumnReweightedMeanShiftFiniteSetup E m N)
    {B ε : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n, |setup.normalizerProduct n| ≤ B) (hε : 0 < ε)
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (Vhat : Ω → (Fin N → E))
    (hmeas : AEStronglyMeasurable
      (fun ω => setup.modifiedProbeDrift - Vhat ω) P)
    (hint : Integrable
      (fun ω => ‖setup.modifiedProbeDrift - Vhat ω‖ ^ 2) P) :
    P {ω | (2 * B / setup.frameConstant) * (‖Vhat ω‖ + ε) <
        ∑ i, |setup.a.weight i - setup.b.weight i|}
      ≤ ENNReal.ofReal
          ((∫ ω, ‖setup.modifiedProbeDrift - Vhat ω‖ ^ 2 ∂P) / ε ^ 2) := by
  simpa [toPopulationSetup, normalizerProduct, modifiedProbeDrift]
    using setup.toPopulationSetup.estimate_failure_le_meanSquare
      hB0 hnormalizer hε P Vhat hmeas hint

end ColumnReweightedMeanShiftFiniteSetup

end PaperFiniteIdentifiability
end DriftingIdentifiability
