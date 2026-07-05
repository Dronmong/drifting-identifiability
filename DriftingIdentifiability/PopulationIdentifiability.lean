import DriftingIdentifiability.PaperDriftIdentifiability
import DriftingIdentifiability.FiniteStability

/-!
# Verified population-level finite identifiability

This module connects the paper's finite-basis algebra to genuine probability
measures.  It treats the ideal, normalized population mean-shift field in data
space.  It does not model CFG, a minibatch estimator, or inversion of a
non-injective feature map.
-/

open scoped BigOperators ENNReal
open MeasureTheory

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

variable {E : Type u} [MeasurableSpace E]

/-- A finite family of genuine probability densities with respect to a fixed
reference probability measure. -/
structure ProbabilityDensityBasis (reference : Distribution E) (m : ℕ) where
  density : Fin m → E → ℝ
  measurable_density : ∀ i, Measurable (density i)
  nonnegative : ∀ i x, 0 ≤ density i x
  integrable_density : ∀ i, Integrable (density i) reference
  integral_density : ∀ i, ∫ x, density i x ∂reference = 1

namespace ProbabilityDensityBasis

variable {reference : Distribution E} {m : ℕ}

/-- The real-valued mixture density associated with simplex coefficients. -/
def mixtureDensity (basis : ProbabilityDensityBasis reference m)
    (a : FiniteProbabilityVector m) : E → ℝ :=
  basisDensity m basis.density a.weight

theorem measurable_mixtureDensity (basis : ProbabilityDensityBasis reference m)
    (a : FiniteProbabilityVector m) : Measurable (basis.mixtureDensity a) := by
  unfold mixtureDensity basisDensity
  exact Finset.measurable_sum _ fun i _ => measurable_const.mul (basis.measurable_density i)

theorem mixtureDensity_nonnegative (basis : ProbabilityDensityBasis reference m)
    (a : FiniteProbabilityVector m) (x : E) : 0 ≤ basis.mixtureDensity a x := by
  unfold mixtureDensity basisDensity
  apply Finset.sum_nonneg
  intro i _
  exact mul_nonneg (a.nonnegative i) (basis.nonnegative i x)

theorem integrable_mixtureDensity (basis : ProbabilityDensityBasis reference m)
    (a : FiniteProbabilityVector m) : Integrable (basis.mixtureDensity a) reference := by
  unfold mixtureDensity basisDensity
  apply integrable_finsetSum Finset.univ
  intro i _
  exact (basis.integrable_density i).const_mul (a.weight i)

theorem integral_mixtureDensity (basis : ProbabilityDensityBasis reference m)
    (a : FiniteProbabilityVector m) :
    ∫ x, basis.mixtureDensity a x ∂reference = 1 := by
  simp only [mixtureDensity, basisDensity]
  rw [integral_finsetSum]
  · simp_rw [integral_const_mul, basis.integral_density, mul_one]
    exact a.normalized
  · intro i _
    exact (basis.integrable_density i).const_mul (a.weight i)

/-- The probability measure represented by a finite mixture density. -/
noncomputable def basisMeasure (basis : ProbabilityDensityBasis reference m)
    (a : FiniteProbabilityVector m) : Distribution E :=
  reference.withDensity (fun x => ENNReal.ofReal (basis.mixtureDensity a x))

theorem basisMeasure_apply (basis : ProbabilityDensityBasis reference m)
    (a : FiniteProbabilityVector m) {s : Set E} (hs : MeasurableSet s) :
    basis.basisMeasure a s = ∫⁻ x in s, ENNReal.ofReal (basis.mixtureDensity a x) ∂reference := by
  exact withDensity_apply _ hs

theorem basisMeasure_isProbability (basis : ProbabilityDensityBasis reference m)
    (a : FiniteProbabilityVector m) : IsProbabilityMeasure (basis.basisMeasure a) := by
  refine ⟨?_⟩
  rw [basisMeasure_apply basis a MeasurableSet.univ, setLIntegral_univ,
    ← ofReal_integral_eq_lintegral_ofReal (basis.integrable_mixtureDensity a)
      (Filter.Eventually.of_forall (basis.mixtureDensity_nonnegative a)),
    basis.integral_mixtureDensity]
  exact ENNReal.ofReal_one

theorem basisMeasure_eq_of_weight_eq (basis : ProbabilityDensityBasis reference m)
    {a b : FiniteProbabilityVector m} (h : a.weight = b.weight) :
    basis.basisMeasure a = basis.basisMeasure b := by
  cases a
  cases b
  simp_all [basisMeasure, mixtureDensity]

theorem basisMeasure_eq_of_mixtureDensity_eq
    (basis : ProbabilityDensityBasis reference m) {a b : FiniteProbabilityVector m}
    (h : basis.mixtureDensity a = basis.mixtureDensity b) :
    basis.basisMeasure a = basis.basisMeasure b := by
  unfold basisMeasure
  rw [h]

theorem integral_basisMeasure_eq_density_smul
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]
    (basis : ProbabilityDensityBasis reference m) (a : FiniteProbabilityVector m)
    (g : E → F) :
    ∫ x, g x ∂basis.basisMeasure a =
      ∫ x, basis.mixtureDensity a x • g x ∂reference := by
  rw [basisMeasure, integral_withDensity_eq_integral_toReal_smul
    (basis.measurable_mixtureDensity a).ennreal_ofReal
    (Filter.Eventually.of_forall fun _ => ENNReal.ofReal_lt_top) g]
  apply integral_congr_ae
  filter_upwards with x
  rw [ENNReal.toReal_ofReal (basis.mixtureDensity_nonnegative a x)]

end ProbabilityDensityBasis

section InteractionBridge

variable [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
  {reference : Distribution E} {m : ℕ}

/-- The paper's measure-level bilinear interaction agrees with its
density-weighted version when both measures are finite-basis mixtures over the
same reference law. -/
theorem interactionDrift_basisMeasure_eq_densityInteractionDrift
    (basis : ProbabilityDensityBasis reference m)
    (K : E → E → E → E) (a b : FiniteProbabilityVector m) (x : E)
    (hintegrable : Integrable (fun y : E × E => K x y.1 y.2)
      ((basis.basisMeasure a).prod (basis.basisMeasure b))) :
    interactionDrift K (basis.basisMeasure a) (basis.basisMeasure b) x =
      densityInteractionDrift reference K
        (basis.mixtureDensity a) (basis.mixtureDensity b) x := by
  letI := basis.basisMeasure_isProbability a
  letI := basis.basisMeasure_isProbability b
  unfold interactionDrift densityInteractionDrift
  rw [integral_prod _ hintegrable]
  rw [basis.integral_basisMeasure_eq_density_smul a]
  apply integral_congr_ae
  filter_upwards with yplus
  rw [basis.integral_basisMeasure_eq_density_smul b]
  rw [← integral_smul]
  apply integral_congr_ae
  filter_upwards with yminus
  rw [smul_smul]

end InteractionBridge

section NormalizedMeanShift

variable [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- All independently checkable data needed by the verified finite population
theorem.  In particular, no zero-drift or equality conclusion is stored here. -/
structure PopulationMeanShiftFiniteSetup (E : Type u) [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] (m N : ℕ) where
  reference : Distribution E
  refProb : IsProbabilityMeasure reference
  basis : ProbabilityDensityBasis reference m
  kernel : E → E → ℝ
  probes : Fin N → E
  a : FiniteProbabilityVector m
  b : FiniteProbabilityVector m
  meanShiftRegular : ∀ n, MeanShiftRegularAt kernel
    (basis.basisMeasure a) (basis.basisMeasure b) (probes n)
  interactionIntegrable : ∀ n, Integrable
    (fun y : E × E => meanShiftInteractionKernel kernel (probes n) y.1 y.2)
    ((basis.basisMeasure a).prod (basis.basisMeasure b))
  basisInteractionIntegrable : ∀ i j n, Integrable
    (fun y : E × E =>
      (basis.density i y.1 * basis.density j y.2) •
        meanShiftInteractionKernel kernel (probes n) y.1 y.2)
    (reference.prod reference)
  frameConstant : ℝ
  frameBound : InteractionFrameBound
    (inducedInteractionVector reference (meanShiftInteractionKernel kernel)
      basis.density probes) frameConstant

namespace PopulationMeanShiftFiniteSetup

variable {m N : ℕ}

noncomputable def targetMeasure (setup : PopulationMeanShiftFiniteSetup E m N) :
    Distribution E := setup.basis.basisMeasure setup.a

noncomputable def modelMeasure (setup : PopulationMeanShiftFiniteSetup E m N) :
    Distribution E := setup.basis.basisMeasure setup.b

theorem targetMeasure_isProbability (setup : PopulationMeanShiftFiniteSetup E m N) :
    IsProbabilityMeasure setup.targetMeasure :=
  setup.basis.basisMeasure_isProbability setup.a

theorem modelMeasure_isProbability (setup : PopulationMeanShiftFiniteSetup E m N) :
    IsProbabilityMeasure setup.modelMeasure :=
  setup.basis.basisMeasure_isProbability setup.b

noncomputable def normalizerProduct
    (setup : PopulationMeanShiftFiniteSetup E m N) (n : Fin N) : ℝ :=
  kernelNormalizer setup.kernel setup.targetMeasure (setup.probes n) *
    kernelNormalizer setup.kernel setup.modelMeasure (setup.probes n)

noncomputable def normalizedProbeDrift
    (setup : PopulationMeanShiftFiniteSetup E m N) : Fin N → E :=
  fun n => meanShiftDrift setup.kernel setup.targetMeasure setup.modelMeasure
    (setup.probes n)

/-- **Probe-local core.**  Only the finite drift values at the probes are used:
if the normalized mean-shift drift vanishes at each probe, the unnormalized
density interaction numerator vanishes there.  The full pointwise `ZeroDrift`
hypothesis is never needed. -/
theorem densityInteraction_zero_at_probe_of_probeZero
    (setup : PopulationMeanShiftFiniteSetup E m N)
    (hzero : ∀ n, setup.normalizedProbeDrift n = 0) (n : Fin N) :
    densityInteractionDrift setup.reference
      (meanShiftInteractionKernel setup.kernel)
      (setup.basis.mixtureDensity setup.a)
      (setup.basis.mixtureDensity setup.b) (setup.probes n) = 0 := by
  letI := setup.targetMeasure_isProbability
  letI := setup.modelMeasure_isProbability
  have h11 := equation_11_bilinear_mean_shift setup.kernel
    setup.targetMeasure setup.modelMeasure (setup.probes n) (setup.meanShiftRegular n)
  have hn : meanShiftDrift setup.kernel setup.targetMeasure setup.modelMeasure
      (setup.probes n) = 0 := hzero n
  rw [hn] at h11
  have hscale :
      (kernelNormalizer setup.kernel setup.targetMeasure (setup.probes n) *
        kernelNormalizer setup.kernel setup.modelMeasure (setup.probes n))⁻¹ ≠ 0 :=
    inv_ne_zero (mul_ne_zero (setup.meanShiftRegular n).zp_ne_zero
      (setup.meanShiftRegular n).zq_ne_zero)
  have hnumerator : interactionDrift (meanShiftInteractionKernel setup.kernel)
      setup.targetMeasure setup.modelMeasure (setup.probes n) = 0 := by
    exact (smul_eq_zero.mp h11.symm).resolve_left hscale
  rw [← interactionDrift_basisMeasure_eq_densityInteractionDrift setup.basis
    (meanShiftInteractionKernel setup.kernel) setup.a setup.b (setup.probes n)
    (setup.interactionIntegrable n)]
  simpa only [targetMeasure, modelMeasure] using hnumerator

/-- At each probe, zero normalized mean-shift drift forces the unnormalized
density interaction numerator to vanish.  A corollary of the probe-local core
specialized to the full pointwise `ZeroDrift` hypothesis. -/
theorem densityInteraction_zero_at_probe
    (setup : PopulationMeanShiftFiniteSetup E m N)
    (hzero : ZeroDrift (meanShiftDrift setup.kernel)
      setup.targetMeasure setup.modelMeasure) (n : Fin N) :
    densityInteractionDrift setup.reference
      (meanShiftInteractionKernel setup.kernel)
      (setup.basis.mixtureDensity setup.a)
      (setup.basis.mixtureDensity setup.b) (setup.probes n) = 0 :=
  setup.densityInteraction_zero_at_probe_of_probeZero
    (fun k => hzero (setup.probes k)) n

/-- The finite bilinear interaction vector is the normalized probe drift
rescaled by the product of the two kernel normalizers. -/
theorem bilinear_eq_normalizer_smul_probeDrift
    (setup : PopulationMeanShiftFiniteSetup E m N) :
    (∑ i, ∑ j, (setup.a.weight i * setup.b.weight j) •
      inducedInteractionVector setup.reference
        (meanShiftInteractionKernel setup.kernel) setup.basis.density
        setup.probes i j) =
      fun n => setup.normalizerProduct n • setup.normalizedProbeDrift n := by
  letI := setup.refProb
  letI := setup.targetMeasure_isProbability
  letI := setup.modelMeasure_isProbability
  have h31 := equation_31_bilinear_expansion setup.reference
    (meanShiftInteractionKernel setup.kernel) setup.basis.density setup.probes
    setup.a.weight setup.b.weight setup.basisInteractionIntegrable
  rw [← h31]
  funext n
  change densityInteractionDrift setup.reference
    (meanShiftInteractionKernel setup.kernel)
    (setup.basis.mixtureDensity setup.a) (setup.basis.mixtureDensity setup.b)
    (setup.probes n) = _
  rw [← interactionDrift_basisMeasure_eq_densityInteractionDrift setup.basis
    (meanShiftInteractionKernel setup.kernel) setup.a setup.b (setup.probes n)
    (setup.interactionIntegrable n)]
  have h11 := equation_11_bilinear_mean_shift setup.kernel
    setup.targetMeasure setup.modelMeasure (setup.probes n) (setup.meanShiftRegular n)
  change setup.normalizedProbeDrift n =
    (setup.normalizerProduct n)⁻¹ •
      interactionDrift (meanShiftInteractionKernel setup.kernel)
        setup.targetMeasure setup.modelMeasure (setup.probes n) at h11
  have hscale : setup.normalizerProduct n ≠ 0 :=
    mul_ne_zero (setup.meanShiftRegular n).zp_ne_zero
      (setup.meanShiftRegular n).zq_ne_zero
  rw [h11, smul_smul, mul_inv_cancel₀ hscale, one_smul]
  rfl

/-- Practical finite stability for the actual normalized population field. -/
theorem coefficientStability
    (setup : PopulationMeanShiftFiniteSetup E m N) {B : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n, |setup.normalizerProduct n| ≤ B) :
    (∑ i, |setup.a.weight i - setup.b.weight i|) ≤
      (2 * B / setup.frameConstant) * ‖setup.normalizedProbeDrift‖ := by
  letI := setup.refProb
  have hanti := antisymmetric_kernel_induces_basis_antisymmetry setup.reference
    (meanShiftInteractionKernel setup.kernel)
    (meanShiftInteractionKernel_antisymmetric setup.kernel)
    setup.basis.density setup.probes setup.basisInteractionIntegrable
  exact coeffL1_le_of_frame_scaledDrift _ hanti setup.frameBound hB0
    setup.a setup.b setup.normalizerProduct setup.normalizedProbeDrift
    hnormalizer setup.bilinear_eq_normalizer_smul_probeDrift

/-- Probe-local reduction to the axiom-free finite drift setup: consumes only
the finite probe-drift vanishing hypothesis. -/
noncomputable def toDriftFiniteSetup_of_probeZero
    (setup : PopulationMeanShiftFiniteSetup E m N)
    (hzero : ∀ n, setup.normalizedProbeDrift n = 0) : DriftFiniteSetup E m N where
  reference := setup.reference
  refProb := setup.refProb
  K := meanShiftInteractionKernel setup.kernel
  Kanti := meanShiftInteractionKernel_antisymmetric setup.kernel
  φ := setup.basis.density
  probes := setup.probes
  regular := setup.basisInteractionIntegrable
  a := setup.a
  b := setup.b
  nondegenerate := interactionFrameBound_basisNondegenerate _ setup.frameBound
  driftZero := setup.densityInteraction_zero_at_probe_of_probeZero hzero

/-- Pointwise-`ZeroDrift` specialization of the probe-local reduction. -/
noncomputable def toDriftFiniteSetup
    (setup : PopulationMeanShiftFiniteSetup E m N)
    (hzero : ZeroDrift (meanShiftDrift setup.kernel)
      setup.targetMeasure setup.modelMeasure) : DriftFiniteSetup E m N :=
  setup.toDriftFiniteSetup_of_probeZero (fun n => hzero (setup.probes n))

end PopulationMeanShiftFiniteSetup

/-- **Promoted probe-local result (Objective 2).**  Identifiability needs only
the *finite* observable drift vector to vanish: if the normalized population
mean-shift drift is zero at each of the `N` probes, the represented probability
measures coincide.  This is strictly weaker than pointwise `ZeroDrift` and does
not require full topological support; it is the exact hypothesis the proof
consumes and matches the finite quantity a training loss can observe. -/
theorem finitePopulationMeanShift_identifies_of_probeZero
    {m N : ℕ} (setup : PopulationMeanShiftFiniteSetup E m N)
    (hzero : ∀ n, setup.normalizedProbeDrift n = 0) :
    setup.targetMeasure = setup.modelMeasure := by
  have hweights := driftProbeZeroIdentifiesCoefficients
    (setup.toDriftFiniteSetup_of_probeZero hzero)
  exact setup.basis.basisMeasure_eq_of_weight_eq hweights

/-- **Promoted paper-native result.** For finite mixtures of genuine probability
densities, pointwise zero of the paper's normalized population mean-shift field
forces equality of the represented probability measures, provided the paper's
finite interaction family is nondegenerate and all displayed integrals and
normalizers are valid.  A corollary of the probe-local theorem. -/
theorem finitePopulationMeanShift_identifies
    {m N : ℕ} (setup : PopulationMeanShiftFiniteSetup E m N)
    (hzero : ZeroDrift (meanShiftDrift setup.kernel)
      setup.targetMeasure setup.modelMeasure) :
    setup.targetMeasure = setup.modelMeasure :=
  finitePopulationMeanShift_identifies_of_probeZero setup
    (fun n => hzero (setup.probes n))

/-- Population-loss corollary.  If the model law has full topological support
and the mean-shift field is continuous, zero ideal squared-drift energy implies
the same measure equality. -/
theorem finitePopulationMeanShift_identifies_of_energy
    {m N : ℕ} (setup : PopulationMeanShiftFiniteSetup E m N)
    [setup.modelMeasure.IsOpenPosMeasure]
    (hcontinuous : Continuous
      (meanShiftDrift setup.kernel setup.targetMeasure setup.modelMeasure))
    (hintegrable : Integrable (fun x =>
      ‖meanShiftDrift setup.kernel setup.targetMeasure setup.modelMeasure x‖ ^ 2)
      setup.modelMeasure)
    (henergy : DriftingIdentifiability.driftEnergy (meanShiftDrift setup.kernel)
      setup.targetMeasure setup.modelMeasure = 0) :
    setup.targetMeasure = setup.modelMeasure := by
  apply finitePopulationMeanShift_identifies setup
  apply DriftingIdentifiability.zeroDrift_of_ae_of_continuous_fullSupport
    _ _ _ hcontinuous
  exact DriftingIdentifiability.zeroDriftAE_of_driftEnergy_eq_zero
    _ _ _ hintegrable henergy

end NormalizedMeanShift

end PaperFiniteIdentifiability
end DriftingIdentifiability
