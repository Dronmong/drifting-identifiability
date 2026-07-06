import DriftingIdentifiability.EmpiricalFrameBound

/-!
# Practical model-class infrastructure

This module develops the deterministic population part of Objective 3.  It
packages the higher-dimensional Gaussian construction end to end, exposes a
continuous-density interface, and transfers certified frame bounds to smooth
bases, perturbed probes, and the paper's Laplace kernel through an explicit
uniform interaction-error condition.

It does not identify a random minibatch estimator with the population field;
that remains the finite-sample problem of Objective 4.
-/

open scoped BigOperators ENNReal RealInnerProductSpace
open MeasureTheory

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

/-! ## Continuous probability-density bases -/

/-- A paper-native probability-density basis whose component densities are
continuous.  All probability and normalization obligations remain inherited
from `ProbabilityDensityBasis`; continuity is extra model-class structure, not
a hidden identifiability assumption. -/
structure ContinuousProbabilityDensityBasis
    (E : Type u) [TopologicalSpace E] [MeasurableSpace E]
    (reference : Distribution E) (m : ℕ)
    extends ProbabilityDensityBasis reference m where
  continuous_density : ∀ i, Continuous (toProbabilityDensityBasis.density i)

/-- A probability-density basis with `C∞` component densities.  It can enter
the continuous perturbation theorem through `toContinuous`. -/
structure SmoothProbabilityDensityBasis
    (E : Type u) [NormedAddCommGroup E] [NormedSpace ℝ E] [MeasurableSpace E]
    (reference : Distribution E) (m : ℕ)
  extends ProbabilityDensityBasis reference m where
  smooth_density : ∀ i,
    ContDiff ℝ (⊤ : ℕ∞) (toProbabilityDensityBasis.density i)

namespace SmoothProbabilityDensityBasis

variable {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [MeasurableSpace E] {reference : Distribution E} {m : ℕ}

/-- Forget smoothness while retaining the continuity needed by the robust
population setup. -/
def toContinuous (basis : SmoothProbabilityDensityBasis E reference m) :
    ContinuousProbabilityDensityBasis E reference m where
  toProbabilityDensityBasis := basis.toProbabilityDensityBasis
  continuous_density i := (basis.smooth_density i).continuous

end SmoothProbabilityDensityBasis

/-! ## Higher-dimensional, variable-bandwidth Gaussian population theorem -/

section HigherDimensionalGaussianPopulation

variable {F : Type u} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
  [CompleteSpace F] [MeasurableSpace F] [BorelSpace F]
  [MeasurableSingletonClass F] [SecondCountableTopology F]

omit [InnerProductSpace ℝ F] [CompleteSpace F] [BorelSpace F]
  [SecondCountableTopology F] in
/-- The empirical-mixture Gaussian normalizer in an arbitrary real
inner-product data space is a finite weighted kernel sum. -/
theorem gaussianKernelNormalizer_empiricalPointND
    {m : ℕ} (hm : 0 < m) (z : Fin m → F) (hz : Function.Injective z)
    (σ : ℝ) (x : F) (a : FiniteProbabilityVector m) :
    kernelNormalizer (gaussianKernel σ)
      ((empiricalPointBasis hm z hz).basisMeasure a) x =
        ∑ r, a.weight r * gaussianKernel σ x (z r) := by
  unfold kernelNormalizer
  rw [(empiricalPointBasis hm z hz).integral_basisMeasure_eq_density_smul,
    integral_empiricalFin]
  simp_rw [empiricalPointBasis_mixtureDensity_apply hm z hz a]
  simp only [smul_eq_mul]
  have hm0 : (m : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hm)
  rw [show (∑ r, (m : ℝ) * a.weight r * gaussianKernel σ x (z r)) =
      (m : ℝ) * ∑ r, a.weight r * gaussianKernel σ x (z r) by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun r _ => by ring]
  rw [inv_mul_cancel_left₀ hm0]

omit [InnerProductSpace ℝ F] [CompleteSpace F] [BorelSpace F]
  [SecondCountableTopology F] in
/-- Gaussian empirical normalizers are strictly positive in arbitrary
dimension. -/
theorem gaussianKernelNormalizer_empiricalPointND_pos
    {m : ℕ} (hm : 0 < m) (z : Fin m → F) (hz : Function.Injective z)
    (σ : ℝ) (x : F) (a : FiniteProbabilityVector m) :
    0 < kernelNormalizer (gaussianKernel σ)
      ((empiricalPointBasis hm z hz).basisMeasure a) x := by
  rw [gaussianKernelNormalizer_empiricalPointND hm z hz]
  have hexists : ∃ i : Fin m, 0 < a.weight i := by
    by_contra h
    push Not at h
    have hzero : a.weight = 0 := by
      funext i
      exact le_antisymm (h i) (a.nonnegative i)
    have hnorm := a.normalized
    rw [hzero] at hnorm
    simp at hnorm
  rcases hexists with ⟨i, hi⟩
  apply Finset.sum_pos'
  · intro r _
    exact mul_nonneg (a.nonnegative r) (Real.exp_pos _).le
  · exact ⟨i, Finset.mem_univ i, mul_pos hi (Real.exp_pos _)⟩

omit [InnerProductSpace ℝ F] [CompleteSpace F] [MeasurableSpace F]
  [BorelSpace F] [MeasurableSingletonClass F] [SecondCountableTopology F] in
/-- A nonzero-bandwidth Gaussian kernel takes values at most one in any normed
data space. -/
theorem gaussianKernel_le_one_of_ne (σ : ℝ) (hσ : σ ≠ 0) (x y : F) :
    gaussianKernel σ x y ≤ 1 := by
  rw [gaussianKernel, Real.exp_le_one_iff]
  have hden : 0 < 2 * σ ^ 2 := mul_pos (by norm_num) (sq_pos_of_ne_zero hσ)
  exact mul_nonpos_of_nonpos_of_nonneg
    (neg_nonpos.mpr (by positivity : 0 ≤ (1 / (2 * σ ^ 2) : ℝ)))
    (sq_nonneg ‖x - y‖)

omit [InnerProductSpace ℝ F] [CompleteSpace F] [BorelSpace F]
  [SecondCountableTopology F] in
/-- Consequently every empirical Gaussian normalizer is bounded by one. -/
theorem gaussianKernelNormalizer_empiricalPointND_le_one
    {m : ℕ} (hm : 0 < m) (z : Fin m → F) (hz : Function.Injective z)
    (σ : ℝ) (hσ : σ ≠ 0) (x : F) (a : FiniteProbabilityVector m) :
    kernelNormalizer (gaussianKernel σ)
      ((empiricalPointBasis hm z hz).basisMeasure a) x ≤ 1 := by
  rw [gaussianKernelNormalizer_empiricalPointND hm z hz]
  calc
    (∑ r, a.weight r * gaussianKernel σ x (z r)) ≤
        ∑ r, a.weight r * 1 := by
      refine Finset.sum_le_sum fun r _ => ?_
      exact mul_le_mul_of_nonneg_left
        (gaussianKernel_le_one_of_ne σ hσ x (z r)) (a.nonnegative r)
    _ = 1 := by simpa using a.normalized

/-- All regularity obligations for the normalized Gaussian field are automatic
for higher-dimensional empirical mixtures. -/
theorem gaussianEmpiricalPointND_meanShiftRegular
    {m : ℕ} (hm : 0 < m) (σ : ℝ) (z : Fin m → F)
    (hz : Function.Injective z) (x : F)
    (a b : FiniteProbabilityVector m) :
    MeanShiftRegularAt (gaussianKernel σ)
      ((empiricalPointBasis hm z hz).basisMeasure a)
      ((empiricalPointBasis hm z hz).basisMeasure b) x := by
  refine
    { zp_ne_zero := ne_of_gt
        (gaussianKernelNormalizer_empiricalPointND_pos hm z hz σ x a)
      zq_ne_zero := ne_of_gt
        (gaussianKernelNormalizer_empiricalPointND_pos hm z hz σ x b)
      integrable_p := empiricalPointBasis_integrable hm z hz a _
      integrable_q := empiricalPointBasis_integrable hm z hz b _
      integrable_product := empiricalPointBasis_integrable_prod hm z hz a b _ ?_ }
  apply Continuous.aestronglyMeasurable
  unfold gaussianKernel
  fun_prop

/-- A canonical positive frame constant for the higher-dimensional structured
Gaussian construction.  Its existence is proved by the vector-weighted
Vandermonde argument; quantitative certification can use the perturbation and
dual-certificate interfaces below. -/
noncomputable def gaussianEmpiricalPointNDFrameConstant
    {m : ℕ} (hm : 2 ≤ m) (σ : ℝ) (hσ : σ ≠ 0) (u : F) (z : Fin m → F)
    (hz : Function.Injective z) (hsums : DistinctProjectedPairSums u z) : ℝ :=
  Classical.choose
    (gaussianEmpiricalPointND_exists_frameBound hm σ hσ u z hz hsums)

omit [BorelSpace F] [SecondCountableTopology F] in
theorem gaussianEmpiricalPointNDFrameBound
    {m : ℕ} (hm : 2 ≤ m) (σ : ℝ) (hσ : σ ≠ 0) (u : F) (z : Fin m → F)
    (hz : Function.Injective z) (hsums : DistinctProjectedPairSums u z) :
    InteractionFrameBound
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel σ))
        (empiricalPointDensity z) (structuredGaussianProbesND m u))
      (gaussianEmpiricalPointNDFrameConstant hm σ hσ u z hz hsums) :=
  (Classical.choose_spec
    (gaussianEmpiricalPointND_exists_frameBound hm σ hσ u z hz hsums)).2

/-- Complete higher-dimensional, arbitrary-positive-bandwidth population
setup. The frame hypothesis is derived from explicit support/probe geometry. -/
noncomputable def gaussianEmpiricalPointNDSetup
    {m : ℕ} (hm : 2 ≤ m) (σ : ℝ) (hσ : ValidBandwidth σ)
    (u : F) (z : Fin m → F) (hz : Function.Injective z)
    (hsums : DistinctProjectedPairSums u z)
    (a b : FiniteProbabilityVector m) :
    PopulationMeanShiftFiniteSetup F m (Fintype.card (StrictPair m)) where
  reference := empiricalFin z
  refProb := empiricalFin_isProbability (by omega) z
  basis := empiricalPointBasis (by omega) z hz
  kernel := gaussianKernel σ
  probes := structuredGaussianProbesND m u
  a := a
  b := b
  meanShiftRegular n :=
    gaussianEmpiricalPointND_meanShiftRegular (by omega) σ z hz
      (structuredGaussianProbesND m u n) a b
  interactionIntegrable n := by
    apply empiricalPointBasis_integrable_prod (by omega) z hz
    apply Continuous.aestronglyMeasurable
    simp only [meanShiftInteractionKernel]
    unfold gaussianKernel
    fun_prop
  basisInteractionIntegrable i j n := by
    apply integrable_empiricalFin_prod (by omega) z
    apply Measurable.aestronglyMeasurable
    have hi : Measurable (fun y : F × F => empiricalPointDensity z i y.1) :=
      ((empiricalPointBasis (by omega) z hz).measurable_density i).comp measurable_fst
    have hj : Measurable (fun y : F × F => empiricalPointDensity z j y.2) :=
      ((empiricalPointBasis (by omega) z hz).measurable_density j).comp measurable_snd
    have hK : Measurable (fun y : F × F =>
        meanShiftInteractionKernel (gaussianKernel σ)
          (structuredGaussianProbesND m u n) y.1 y.2) := by
      apply Continuous.measurable
      simp only [meanShiftInteractionKernel]
      unfold gaussianKernel
      fun_prop
    exact (hi.mul hj).smul hK
  frameConstant :=
    gaussianEmpiricalPointNDFrameConstant hm σ (ne_of_gt hσ) u z hz hsums
  frameBound :=
    gaussianEmpiricalPointNDFrameBound hm σ (ne_of_gt hσ) u z hz hsums

/-- The higher-dimensional Gaussian setup has the explicit normalizer-product
bound `B = 1`. -/
theorem gaussianEmpiricalPointND_normalizerProduct_abs_le_one
    {m : ℕ} (hm : 2 ≤ m) (σ : ℝ) (hσ : ValidBandwidth σ)
    (u : F) (z : Fin m → F) (hz : Function.Injective z)
    (hsums : DistinctProjectedPairSums u z)
    (a b : FiniteProbabilityVector m)
    (n : Fin (Fintype.card (StrictPair m))) :
    |(gaussianEmpiricalPointNDSetup hm σ hσ u z hz hsums a b).normalizerProduct n| ≤
      1 := by
  change |kernelNormalizer (gaussianKernel σ)
      ((empiricalPointBasis (by omega) z hz).basisMeasure a)
      (structuredGaussianProbesND m u n) *
    kernelNormalizer (gaussianKernel σ)
      ((empiricalPointBasis (by omega) z hz).basisMeasure b)
      (structuredGaussianProbesND m u n)| ≤ 1
  have hpa := gaussianKernelNormalizer_empiricalPointND_pos
    (by omega) z hz σ (structuredGaussianProbesND m u n) a
  have hpb := gaussianKernelNormalizer_empiricalPointND_pos
    (by omega) z hz σ (structuredGaussianProbesND m u n) b
  rw [abs_of_pos (mul_pos hpa hpb)]
  exact mul_le_one₀
    (gaussianKernelNormalizer_empiricalPointND_le_one
      (by omega) z hz σ (ne_of_gt hσ) (structuredGaussianProbesND m u n) a)
    hpb.le
    (gaussianKernelNormalizer_empiricalPointND_le_one
      (by omega) z hz σ (ne_of_gt hσ) (structuredGaussianProbesND m u n) b)

/-- **End-to-end Objective 3 Gaussian theorem.** In any separable real
inner-product data space, zero finite normalized population drift loss
identifies two empirical mixtures on a common injective support whenever one
projection separates all strict-pair sums. -/
theorem gaussianEmpiricalPointND_identifies_of_probeEnergy_eq_zero
    {m : ℕ} (hm : 2 ≤ m) (σ : ℝ) (hσ : ValidBandwidth σ)
    (u : F) (z : Fin m → F) (hz : Function.Injective z)
    (hsums : DistinctProjectedPairSums u z)
    (a b : FiniteProbabilityVector m)
    (henergy :
      (gaussianEmpiricalPointNDSetup hm σ hσ u z hz hsums a b).normalizedProbeDriftEnergy =
        0) :
    (empiricalPointBasis (by omega) z hz).basisMeasure a =
      (empiricalPointBasis (by omega) z hz).basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero
    (gaussianEmpiricalPointNDSetup hm σ hσ u z hz hsums a b) henergy

/-- Quantitative higher-dimensional Gaussian stability with `B=1`. -/
theorem gaussianEmpiricalPointND_coefficientStability_probeEnergy
    {m : ℕ} (hm : 2 ≤ m) (σ : ℝ) (hσ : ValidBandwidth σ)
    (u : F) (z : Fin m → F) (hz : Function.Injective z)
    (hsums : DistinctProjectedPairSums u z)
    (a b : FiniteProbabilityVector m) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 / gaussianEmpiricalPointNDFrameConstant hm σ (ne_of_gt hσ) u z hz hsums) *
        Real.sqrt
          (gaussianEmpiricalPointNDSetup hm σ hσ u z hz hsums a b).normalizedProbeDriftEnergy := by
  have h :=
    (gaussianEmpiricalPointNDSetup hm σ hσ u z hz hsums a b).coefficientStability_probeEnergy
      (B := 1) (by norm_num)
      (gaussianEmpiricalPointND_normalizerProduct_abs_le_one
        hm σ hσ u z hz hsums a b)
  simpa only [gaussianEmpiricalPointNDSetup, mul_one] using h

/-! ### Arbitrary and adaptive probe families -/

/-- A higher-dimensional empirical Gaussian setup with an arbitrary probe
family and an explicit vector-valued dual certificate.  This permits fewer or
adaptively selected probes whenever their stacked interaction vectors admit a
biorthogonal certificate; the necessary dimension obstruction remains enforced
by `nondegenerate_pairCount_le_probeDimension`. -/
noncomputable def gaussianEmpiricalPointCertifiedProbeSetup
    {m N : ℕ} (hm : 2 ≤ m) (σ : ℝ) (_hσ : ValidBandwidth σ)
    (z : Fin m → F) (hz : Function.Injective z) (probes : Fin N → F)
    (a b : FiniteProbabilityVector m)
    (certificate : InteractionDualCertificate
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel σ))
        (empiricalPointDensity z) probes)) :
    PopulationMeanShiftFiniteSetup F m N := by
  let p : StrictPair m :=
    ⟨(⟨0, by omega⟩, ⟨1, by omega⟩), by simp⟩
  letI : Nonempty (StrictPair m) := ⟨p⟩
  exact
    { reference := empiricalFin z
      refProb := empiricalFin_isProbability (by omega) z
      basis := empiricalPointBasis (by omega) z hz
      kernel := gaussianKernel σ
      probes := probes
      a := a
      b := b
      meanShiftRegular n :=
        gaussianEmpiricalPointND_meanShiftRegular (by omega) σ z hz (probes n) a b
      interactionIntegrable n := by
        apply empiricalPointBasis_integrable_prod (by omega) z hz
        apply Continuous.aestronglyMeasurable
        simp only [meanShiftInteractionKernel]
        unfold gaussianKernel
        fun_prop
      basisInteractionIntegrable i j n := by
        apply integrable_empiricalFin_prod (by omega) z
        apply Measurable.aestronglyMeasurable
        have hi : Measurable (fun y : F × F => empiricalPointDensity z i y.1) :=
          ((empiricalPointBasis (by omega) z hz).measurable_density i).comp
            measurable_fst
        have hj : Measurable (fun y : F × F => empiricalPointDensity z j y.2) :=
          ((empiricalPointBasis (by omega) z hz).measurable_density j).comp
            measurable_snd
        have hK : Measurable (fun y : F × F =>
            meanShiftInteractionKernel (gaussianKernel σ) (probes n) y.1 y.2) := by
          apply Continuous.measurable
          simp only [meanShiftInteractionKernel]
          unfold gaussianKernel
          fun_prop
        exact (hi.mul hj).smul hK
      frameConstant := certificate.mass⁻¹
      frameBound := interactionFrameBound_of_dualCertificate _ certificate }

theorem gaussianEmpiricalPointCertifiedProbe_normalizerProduct_abs_le_one
    {m N : ℕ} (hm : 2 ≤ m) (σ : ℝ) (hσ : ValidBandwidth σ)
    (z : Fin m → F) (hz : Function.Injective z) (probes : Fin N → F)
    (a b : FiniteProbabilityVector m)
    (certificate : InteractionDualCertificate
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel σ))
        (empiricalPointDensity z) probes)) (n : Fin N) :
    |(gaussianEmpiricalPointCertifiedProbeSetup hm σ hσ z hz probes a b
      certificate).normalizerProduct n| ≤ 1 := by
  change |kernelNormalizer (gaussianKernel σ)
      ((empiricalPointBasis (by omega) z hz).basisMeasure a) (probes n) *
    kernelNormalizer (gaussianKernel σ)
      ((empiricalPointBasis (by omega) z hz).basisMeasure b) (probes n)| ≤ 1
  have hpa := gaussianKernelNormalizer_empiricalPointND_pos
    (by omega) z hz σ (probes n) a
  have hpb := gaussianKernelNormalizer_empiricalPointND_pos
    (by omega) z hz σ (probes n) b
  rw [abs_of_pos (mul_pos hpa hpb)]
  exact mul_le_one₀
    (gaussianKernelNormalizer_empiricalPointND_le_one
      (by omega) z hz σ (ne_of_gt hσ) (probes n) a)
    hpb.le
    (gaussianKernelNormalizer_empiricalPointND_le_one
      (by omega) z hz σ (ne_of_gt hσ) (probes n) b)

/-- Exact identifiability for a certified arbitrary/adaptive probe design. -/
theorem gaussianEmpiricalPointCertifiedProbe_identifies_of_probeEnergy_eq_zero
    {m N : ℕ} (hm : 2 ≤ m) (σ : ℝ) (hσ : ValidBandwidth σ)
    (z : Fin m → F) (hz : Function.Injective z) (probes : Fin N → F)
    (a b : FiniteProbabilityVector m)
    (certificate : InteractionDualCertificate
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel σ))
        (empiricalPointDensity z) probes))
    (henergy :
      (gaussianEmpiricalPointCertifiedProbeSetup hm σ hσ z hz probes a b
        certificate).normalizedProbeDriftEnergy = 0) :
    (empiricalPointBasis (by omega) z hz).basisMeasure a =
      (empiricalPointBasis (by omega) z hz).basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero
    (gaussianEmpiricalPointCertifiedProbeSetup hm σ hσ z hz probes a b certificate)
    henergy

/-- Quantitative stability for a certified arbitrary/adaptive probe design. -/
theorem gaussianEmpiricalPointCertifiedProbe_coefficientStability_probeEnergy
    {m N : ℕ} (hm : 2 ≤ m) (σ : ℝ) (hσ : ValidBandwidth σ)
    (z : Fin m → F) (hz : Function.Injective z) (probes : Fin N → F)
    (a b : FiniteProbabilityVector m)
    (certificate : InteractionDualCertificate
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel σ))
        (empiricalPointDensity z) probes)) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 / certificate.mass⁻¹) *
        Real.sqrt
          (gaussianEmpiricalPointCertifiedProbeSetup hm σ hσ z hz probes a b
            certificate).normalizedProbeDriftEnergy := by
  have h :=
    (gaussianEmpiricalPointCertifiedProbeSetup hm σ hσ z hz probes a b
      certificate).coefficientStability_probeEnergy
      (B := 1) (by norm_num)
      (gaussianEmpiricalPointCertifiedProbe_normalizerProduct_abs_le_one
        hm σ hσ z hz probes a b certificate)
  simpa only [gaussianEmpiricalPointCertifiedProbeSetup, mul_one] using h

end HigherDimensionalGaussianPopulation

/-! ## Certified transfer to continuous/smooth model classes -/

section ContinuousPerturbation

variable {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
  [NormedSpace ℝ E] [CompleteSpace E]

/-- Build a fully verified population setup for a continuous density basis by
showing that its actual interaction vectors lie within `δ` of any already
certified baseline system.  The usable frame constant is explicitly `c-δ`.

This is the formal smooth-basis route: analytic estimates or interval
arithmetic establish `hclose`; no injectivity conclusion is hidden in the
interface. -/
noncomputable def continuousPerturbationSetup
    {m N : ℕ} (reference : Distribution E)
    (refProb : IsProbabilityMeasure reference)
    (basis : ContinuousProbabilityDensityBasis E reference m)
    (kernel : E → E → ℝ) (probes : Fin N → E)
    (a b : FiniteProbabilityVector m)
    (meanShiftRegular : ∀ n, MeanShiftRegularAt kernel
      (basis.toProbabilityDensityBasis.basisMeasure a)
      (basis.toProbabilityDensityBasis.basisMeasure b) (probes n))
    (interactionIntegrable : ∀ n, Integrable
      (fun y : E × E => meanShiftInteractionKernel kernel (probes n) y.1 y.2)
      ((basis.toProbabilityDensityBasis.basisMeasure a).prod
        (basis.toProbabilityDensityBasis.basisMeasure b)))
    (basisInteractionIntegrable : ∀ i j n, Integrable
      (fun y : E × E =>
        (basis.toProbabilityDensityBasis.density i y.1 *
          basis.toProbabilityDensityBasis.density j y.2) •
          meanShiftInteractionKernel kernel (probes n) y.1 y.2)
      (reference.prod reference))
    (baseline : Fin m → Fin m → Fin N → E) (c δ : ℝ)
    (baselineFrame : InteractionFrameBound baseline c) (hδ : δ < c)
    (hclose : ∀ p : StrictPair m,
      ‖baseline p.1.1 p.1.2 -
        inducedInteractionVector reference (meanShiftInteractionKernel kernel)
          basis.toProbabilityDensityBasis.density probes p.1.1 p.1.2‖ ≤ δ) :
    PopulationMeanShiftFiniteSetup E m N where
  reference := reference
  refProb := refProb
  basis := basis.toProbabilityDensityBasis
  kernel := kernel
  probes := probes
  a := a
  b := b
  meanShiftRegular := meanShiftRegular
  interactionIntegrable := interactionIntegrable
  basisInteractionIntegrable := basisInteractionIntegrable
  frameConstant := c - δ
  frameBound := interactionFrameBound_of_uniformPerturbation baseline _
    baselineFrame hδ hclose

/-- Exact identifiability for the continuous/smooth perturbation setup from
zero deterministic probe energy. -/
theorem continuousPerturbation_identifies_of_probeEnergy_eq_zero
    {m N : ℕ} (reference : Distribution E)
    (refProb : IsProbabilityMeasure reference)
    (basis : ContinuousProbabilityDensityBasis E reference m)
    (kernel : E → E → ℝ) (probes : Fin N → E)
    (a b : FiniteProbabilityVector m)
    (meanShiftRegular : ∀ n, MeanShiftRegularAt kernel
      (basis.toProbabilityDensityBasis.basisMeasure a)
      (basis.toProbabilityDensityBasis.basisMeasure b) (probes n))
    (interactionIntegrable : ∀ n, Integrable
      (fun y : E × E => meanShiftInteractionKernel kernel (probes n) y.1 y.2)
      ((basis.toProbabilityDensityBasis.basisMeasure a).prod
        (basis.toProbabilityDensityBasis.basisMeasure b)))
    (basisInteractionIntegrable : ∀ i j n, Integrable
      (fun y : E × E =>
        (basis.toProbabilityDensityBasis.density i y.1 *
          basis.toProbabilityDensityBasis.density j y.2) •
          meanShiftInteractionKernel kernel (probes n) y.1 y.2)
      (reference.prod reference))
    (baseline : Fin m → Fin m → Fin N → E) (c δ : ℝ)
    (baselineFrame : InteractionFrameBound baseline c) (hδ : δ < c)
    (hclose : ∀ p : StrictPair m,
      ‖baseline p.1.1 p.1.2 -
        inducedInteractionVector reference (meanShiftInteractionKernel kernel)
          basis.toProbabilityDensityBasis.density probes p.1.1 p.1.2‖ ≤ δ)
    (henergy :
      (continuousPerturbationSetup reference refProb basis kernel probes a b
        meanShiftRegular interactionIntegrable basisInteractionIntegrable
        baseline c δ baselineFrame hδ hclose).normalizedProbeDriftEnergy = 0) :
    basis.toProbabilityDensityBasis.basisMeasure a =
      basis.toProbabilityDensityBasis.basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero
    (continuousPerturbationSetup reference refProb basis kernel probes a b
      meanShiftRegular interactionIntegrable basisInteractionIntegrable
      baseline c δ baselineFrame hδ hclose) henergy

end ContinuousPerturbation

/-! ## A concrete instance of the paper's Laplace kernel -/

section LaplaceTwoAtom

/-- One probe at the first support point.  For two atoms this is already the
minimum possible nonempty probe family. -/
def empirical01LaplaceProbe : Fin 1 → ℝ := fun _ => 0

/-- Closed form of the two-atom Laplace normalizer. -/
theorem laplaceKernelNormalizer_empirical01
    (τ x : ℝ) (a : FiniteProbabilityVector 2) :
    kernelNormalizer (laplaceKernel τ) (empirical01Basis.basisMeasure a) x =
      a.weight 0 * laplaceKernel τ x 0 +
        a.weight 1 * laplaceKernel τ x 1 := by
  unfold kernelNormalizer
  rw [empirical01Basis.integral_basisMeasure_eq_density_smul,
    integral_empirical2]
  simp only [ProbabilityDensityBasis.mixtureDensity, basisDensity,
    empirical01Basis, Fin.sum_univ_two, empirical01Density_zero_zero,
    empirical01Density_one_zero, empirical01Density_zero_one,
    empirical01Density_one_one, mul_zero, add_zero, mul_two, zero_add,
    smul_eq_mul]
  ring

theorem laplaceKernelNormalizer_empirical01_pos
    (τ x : ℝ) (a : FiniteProbabilityVector 2) :
    0 < kernelNormalizer (laplaceKernel τ)
      (empirical01Basis.basisMeasure a) x := by
  rw [laplaceKernelNormalizer_empirical01]
  have hk0 : 0 < laplaceKernel τ x 0 := Real.exp_pos _
  have hk1 : 0 < laplaceKernel τ x 1 := Real.exp_pos _
  have hsum : a.weight 0 + a.weight 1 = 1 := by
    simpa [Fin.sum_univ_two] using a.normalized
  have hor : 0 < a.weight 0 ∨ 0 < a.weight 1 := by
    have h0 := a.nonnegative 0
    have h1 := a.nonnegative 1
    rcases lt_or_eq_of_le h0 with h0pos | h0zero
    · exact Or.inl h0pos
    · right
      nlinarith
  rcases hor with h0 | h1
  · exact add_pos_of_pos_of_nonneg (mul_pos h0 hk0)
      (mul_nonneg (a.nonnegative 1) hk1.le)
  · exact add_pos_of_nonneg_of_pos
      (mul_nonneg (a.nonnegative 0) hk0.le) (mul_pos h1 hk1)

/-- Positive-bandwidth Laplace values are at most one. -/
theorem laplaceKernel_le_one (τ : ℝ) (hτ : ValidBandwidth τ) (x y : ℝ) :
    laplaceKernel τ x y ≤ 1 := by
  rw [laplaceKernel, Real.exp_le_one_iff]
  exact mul_nonpos_of_nonpos_of_nonneg
    (neg_nonpos.mpr (one_div_nonneg.mpr hτ.le)) (norm_nonneg _)

theorem laplaceKernelNormalizer_empirical01_le_one
    (τ : ℝ) (hτ : ValidBandwidth τ) (x : ℝ)
    (a : FiniteProbabilityVector 2) :
    kernelNormalizer (laplaceKernel τ)
      (empirical01Basis.basisMeasure a) x ≤ 1 := by
  rw [laplaceKernelNormalizer_empirical01]
  calc
    a.weight 0 * laplaceKernel τ x 0 + a.weight 1 * laplaceKernel τ x 1 ≤
        a.weight 0 * 1 + a.weight 1 * 1 := by
      exact add_le_add
        (mul_le_mul_of_nonneg_left (laplaceKernel_le_one τ hτ x 0)
          (a.nonnegative 0))
        (mul_le_mul_of_nonneg_left (laplaceKernel_le_one τ hτ x 1)
          (a.nonnegative 1))
    _ = 1 := by simpa [Fin.sum_univ_two] using a.normalized

/-- All normalized mean-shift regularity obligations are automatic for the
two-atom Laplace model. -/
theorem empirical01Laplace_meanShiftRegular
    (τ x : ℝ) (a b : FiniteProbabilityVector 2) :
    MeanShiftRegularAt (laplaceKernel τ)
      (empirical01Basis.basisMeasure a)
      (empirical01Basis.basisMeasure b) x := by
  refine
    { zp_ne_zero := ne_of_gt (laplaceKernelNormalizer_empirical01_pos τ x a)
      zq_ne_zero := ne_of_gt (laplaceKernelNormalizer_empirical01_pos τ x b)
      integrable_p := empirical01Basis_integrable a _
      integrable_q := empirical01Basis_integrable b _
      integrable_product := empirical01Basis_integrable_prod a b _ ?_ }
  apply Continuous.aestronglyMeasurable
  unfold laplaceKernel
  fun_prop

/-- The actual integral-induced Laplace interaction is nonzero at the single
probe. -/
theorem inducedInteractionVector_empirical01Laplace_ne_zero
    (τ : ℝ) :
    inducedInteractionVector (empirical2 (0 : ℝ) 1)
      (meanShiftInteractionKernel (laplaceKernel τ)) empirical01Density
      empirical01LaplaceProbe 0 1 ≠ 0 := by
  rw [Function.ne_iff]
  refine ⟨0, ?_⟩
  simp only [inducedInteractionVector, basisInteraction_empirical2,
    empirical01Density_zero_zero, empirical01Density_one_one,
    empirical01Density_zero_one, empirical01Density_one_zero, Pi.zero_apply]
  apply smul_ne_zero
  · exact mul_ne_zero (by norm_num)
      (mul_ne_zero (ne_of_gt (Real.exp_pos _)) (ne_of_gt (Real.exp_pos _)))
  · norm_num

/-- Complete two-atom population setup for equation (12)'s Laplace kernel. -/
noncomputable def empirical01LaplaceSetup
    (τ : ℝ) (_hτ : ValidBandwidth τ)
    (a b : FiniteProbabilityVector 2) :
    PopulationMeanShiftFiniteSetup ℝ 2 1 where
  reference := empirical2 0 1
  refProb := empirical2_isProbability 0 1
  basis := empirical01Basis
  kernel := laplaceKernel τ
  probes := empirical01LaplaceProbe
  a := a
  b := b
  meanShiftRegular n := empirical01Laplace_meanShiftRegular τ
    (empirical01LaplaceProbe n) a b
  interactionIntegrable n := by
    apply empirical01Basis_integrable_prod
    apply Continuous.aestronglyMeasurable
    simp only [meanShiftInteractionKernel]
    unfold laplaceKernel
    fun_prop
  basisInteractionIntegrable i j n := by
    apply integrable_empirical2_prod
    apply Measurable.aestronglyMeasurable
    have hi : Measurable (fun y : ℝ × ℝ => empirical01Density i y.1) :=
      (empirical01Basis.measurable_density i).comp measurable_fst
    have hj : Measurable (fun y : ℝ × ℝ => empirical01Density j y.2) :=
      (empirical01Basis.measurable_density j).comp measurable_snd
    have hK : Measurable (fun y : ℝ × ℝ =>
        meanShiftInteractionKernel (laplaceKernel τ)
          (empirical01LaplaceProbe n) y.1 y.2) := by
      apply Continuous.measurable
      simp only [meanShiftInteractionKernel]
      unfold laplaceKernel
      fun_prop
    exact (hi.mul hj).smul hK
  frameConstant :=
    ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
      (meanShiftInteractionKernel (laplaceKernel τ)) empirical01Density
      empirical01LaplaceProbe 0 1‖
  frameBound := interactionFrameBound_two _
    (inducedInteractionVector_empirical01Laplace_ne_zero τ)

theorem empirical01Laplace_normalizerProduct_abs_le_one
    (τ : ℝ) (hτ : ValidBandwidth τ)
    (a b : FiniteProbabilityVector 2) (n : Fin 1) :
    |(empirical01LaplaceSetup τ hτ a b).normalizerProduct n| ≤ 1 := by
  change |kernelNormalizer (laplaceKernel τ) (empirical01Basis.basisMeasure a)
      (empirical01LaplaceProbe n) *
    kernelNormalizer (laplaceKernel τ) (empirical01Basis.basisMeasure b)
      (empirical01LaplaceProbe n)| ≤ 1
  have hpa := laplaceKernelNormalizer_empirical01_pos τ
    (empirical01LaplaceProbe n) a
  have hpb := laplaceKernelNormalizer_empirical01_pos τ
    (empirical01LaplaceProbe n) b
  rw [abs_of_pos (mul_pos hpa hpb)]
  exact mul_le_one₀
    (laplaceKernelNormalizer_empirical01_le_one τ hτ
      (empirical01LaplaceProbe n) a)
    hpb.le
    (laplaceKernelNormalizer_empirical01_le_one τ hτ
      (empirical01LaplaceProbe n) b)

/-- **Concrete paper-kernel theorem.** Zero single-probe population loss for
the positive-bandwidth Laplace mean-shift field identifies the two-atom laws. -/
theorem empirical01Laplace_identifies_of_probeEnergy_eq_zero
    (τ : ℝ) (hτ : ValidBandwidth τ)
    (a b : FiniteProbabilityVector 2)
    (henergy : (empirical01LaplaceSetup τ hτ a b).normalizedProbeDriftEnergy = 0) :
    empirical01Basis.basisMeasure a = empirical01Basis.basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero
    (empirical01LaplaceSetup τ hτ a b) henergy

/-- Quantitative single-probe Laplace stability with `B=1`. -/
theorem empirical01Laplace_coefficientStability_probeEnergy
    (τ : ℝ) (hτ : ValidBandwidth τ)
    (a b : FiniteProbabilityVector 2) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 / (empirical01LaplaceSetup τ hτ a b).frameConstant) *
        Real.sqrt (empirical01LaplaceSetup τ hτ a b).normalizedProbeDriftEnergy := by
  have h := (empirical01LaplaceSetup τ hτ a b).coefficientStability_probeEnergy
    (B := 1) (by norm_num)
    (empirical01Laplace_normalizerProduct_abs_le_one τ hτ a b)
  simpa only [empirical01LaplaceSetup, mul_one] using h

end LaplaceTwoAtom

end PaperFiniteIdentifiability
end DriftingIdentifiability
