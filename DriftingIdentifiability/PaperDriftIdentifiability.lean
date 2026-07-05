import DriftingIdentifiability.FiniteLegitimacy

/-!
# From actual drift-zero to finite identifiability

`PaperFiniteIdentifiability` starts from the *already grouped* bilinear
hypothesis `∑ᵢ ∑ⱼ aᵢ bⱼ • Uᵢⱼ = 0`.  Using the reviewed paper axioms
`equation_31_bilinear_expansion` and
`antisymmetric_kernel_induces_basis_antisymmetry`, this module derives that
hypothesis from the vanishing of the *actual* density-interaction drift
(equations (28)/(30)/(31)) at the finite probe set, and hence proves finite
identifiability directly from zero drift.

The zero-drift hypothesis is probe-wise: the drift vanishes at each of the `N`
probes, exactly as in Appendix C.1.  No global pointwise statement, and no
`a = b`, is assumed; only the reviewed paper interface and the nondegeneracy
condition already shown satisfiable in `FiniteLegitimacy.lean` are used.
-/

open scoped BigOperators
open MeasureTheory

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

variable {E : Type u} [MeasurableSpace E]
  [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- Drift-level hypotheses of Appendix C.1: an anti-symmetric interaction kernel
`K`, a finite basis `φ`, probes, and probability coefficient vectors, with the
actual density-interaction drift vanishing at every probe. -/
structure DriftFiniteSetup (E : Type u) [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] (m N : ℕ) where
  reference : Distribution E
  refProb : IsProbabilityMeasure reference
  K : E → E → E → E
  Kanti : AntiSymmetricInteractionKernel K
  φ : Fin m → E → ℝ
  probes : Fin N → E
  regular : ∀ i j n, Integrable
    (fun y : E × E => (φ i y.1 * φ j y.2) • K (probes n) y.1 y.2)
    (reference.prod reference)
  a : FiniteProbabilityVector m
  b : FiniteProbabilityVector m
  nondegenerate :
    BasisInteractionNondegenerate (inducedInteractionVector reference K φ probes)
  driftZero : ∀ n, densityInteractionDrift reference K
    (basisDensity m φ a.weight) (basisDensity m φ b.weight) (probes n) = 0

namespace DriftFiniteSetup

variable {m N : ℕ}

/-- The induced interaction vectors are anti-symmetric, by the reviewed paper
axiom `antisymmetric_kernel_induces_basis_antisymmetry`. -/
theorem anti (setup : DriftFiniteSetup E m N) :
    AntiSymmetricBasisInteractions
      (inducedInteractionVector setup.reference setup.K setup.φ setup.probes) := by
  haveI := setup.refProb
  exact antisymmetric_kernel_induces_basis_antisymmetry setup.reference setup.K
    setup.Kanti setup.φ setup.probes setup.regular

/-- The grouped bilinear expression vanishes, derived from probe-wise drift-zero
via `equation_31_bilinear_expansion`. -/
theorem zeroBilinear (setup : DriftFiniteSetup E m N) :
    (∑ i, ∑ j, (setup.a.weight i * setup.b.weight j) •
        inducedInteractionVector setup.reference setup.K setup.φ setup.probes i j)
      = 0 := by
  haveI := setup.refProb
  have h31 := equation_31_bilinear_expansion setup.reference setup.K setup.φ
    setup.probes setup.a.weight setup.b.weight setup.regular
  have hz : (fun n => densityInteractionDrift setup.reference setup.K
      (basisDensity m setup.φ setup.a.weight)
      (basisDensity m setup.φ setup.b.weight) (setup.probes n)) = 0 := by
    funext n
    exact setup.driftZero n
  rw [hz] at h31
  exact h31.symm

/-- Reduction to the purely algebraic finite setup, discharging both the
anti-symmetry and the grouped zero-drift fields with the paper axioms. -/
noncomputable def toCoefficientSetup (setup : DriftFiniteSetup E m N) :
    FiniteCoefficientSetup E m N where
  U := inducedInteractionVector setup.reference setup.K setup.φ setup.probes
  anti := setup.anti
  nondegenerate := setup.nondegenerate
  a := setup.a
  b := setup.b
  zeroBilinear := setup.zeroBilinear

end DriftFiniteSetup

/-- **Drift-level finite identifiability.**  If the density-interaction drift of
Appendix C.1 vanishes at every probe, the induced interaction system is
nondegenerate, and the kernel is anti-symmetric, then the coefficient vectors
coincide. -/
theorem driftProbeZeroIdentifiesCoefficients
    {m N : ℕ} (setup : DriftFiniteSetup E m N) :
    setup.a.weight = setup.b.weight :=
  finiteCoefficientIdentifiable setup.toCoefficientSetup

/-- The represented densities coincide as well. -/
theorem driftProbeZeroIdentifiesDensities
    {m N : ℕ} (setup : DriftFiniteSetup E m N) :
    basisDensity m setup.φ setup.a.weight
      = basisDensity m setup.φ setup.b.weight := by
  rw [driftProbeZeroIdentifiesCoefficients setup]

/-- Specialization to the paper's actual mean-shift interaction kernel
(equation (33)), whose anti-symmetry is the reviewed axiom
`meanShiftInteractionKernel_antisymmetric`.  Probe-wise vanishing of the induced
drift, together with nondegeneracy, identifies the basis densities. -/
theorem meanShiftInteractionIdentifiesDensities
    {m N : ℕ} (k : E → E → ℝ) (reference : Distribution E)
    [IsProbabilityMeasure reference]
    (φ : Fin m → E → ℝ) (probes : Fin N → E)
    (regular : ∀ i j n, Integrable
      (fun y : E × E =>
        (φ i y.1 * φ j y.2) • meanShiftInteractionKernel k (probes n) y.1 y.2)
      (reference.prod reference))
    (a b : FiniteProbabilityVector m)
    (nondeg : BasisInteractionNondegenerate
      (inducedInteractionVector reference (meanShiftInteractionKernel k) φ probes))
    (driftZero : ∀ n, densityInteractionDrift reference
      (meanShiftInteractionKernel k)
      (basisDensity m φ a.weight) (basisDensity m φ b.weight) (probes n) = 0) :
    basisDensity m φ a.weight = basisDensity m φ b.weight :=
  driftProbeZeroIdentifiesDensities
    { reference := reference
      refProb := ‹IsProbabilityMeasure reference›
      K := meanShiftInteractionKernel k
      Kanti := meanShiftInteractionKernel_antisymmetric k
      φ := φ
      probes := probes
      regular := regular
      a := a
      b := b
      nondegenerate := nondeg
      driftZero := driftZero }

end PaperFiniteIdentifiability
end DriftingIdentifiability
