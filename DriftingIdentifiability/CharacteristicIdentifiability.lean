import DriftingIdentifiability.FiniteStability

/-!
# Distribution-level identifiability from a characteristic kernel

This module reaches the project's actual target — `Vₚ,q = 0 → p = q` for a
concrete drifting field on genuine probability measures — through the paper's
MMD drift (equation 41) and the reviewed RKHS facts in `Paperaxioms.lean`.

The condition is that the kernel gradient is **characteristic** (`IsCharacteristic`),
a concrete, checkable property of the kernel that is independent of any target
identity and is witnessed by the Gaussian kernel.  The zero-drift hypothesis is
used *only* to match the two mean embeddings; the identification itself is the
external, well-known injectivity of a characteristic embedding.  No step assumes
`p = q`, injectivity of the target map, or uniqueness of the equilibrium.
-/

open MeasureTheory
open scoped BigOperators

namespace DriftingIdentifiability

open Paper

universe u

/-- The pair condition: both laws are probability measures.  Concrete, checkable,
and independent of the identifiability conclusion. -/
def BothProbability {E : Type u} [MeasurableSpace E] (p q : Distribution E) : Prop :=
  IsProbabilityMeasure p ∧ IsProbabilityMeasure q

section Generic
variable {E : Type u} [MeasurableSpace E]
  [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- **Distribution-level identifiability from a characteristic kernel.**  If the
kernel gradient is characteristic, then zero MMD drift (equation 41) between two
probability measures forces the measures to be equal.  Zero drift is used only to
match the mean embeddings; the identification is the reviewed RKHS fact
`characteristic_gradientEmbedding_injective`. -/
theorem characteristicKernel_identifiesAtZero
    (kg : E → E → E) (hkg : IsCharacteristic kg) :
    IdentifiesAtZero BothProbability (mmdDrift kg) := by
  rintro p q ⟨hp, hq⟩ hzero
  haveI := hp
  haveI := hq
  refine characteristic_gradientEmbedding_injective kg hkg p q ?_
  intro x
  exact sub_eq_zero.mp (hzero x)

end Generic

section Gaussian
variable {E : Type u} [MeasurableSpace E]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- **Identifiability for the paper's Gaussian MMD drift.**  Specializing the
characteristic-kernel result to the Gaussian kernel (equations 43–45), which is
characteristic, zero drift of the radial MMD field identifies the target measure
among probability measures. -/
theorem gaussianMmd_identifiesAtZero (σ : ℝ) (hσ : ValidBandwidth σ) :
    IdentifiesAtZero (BothProbability (E := E))
      (radialMmdDrift (gaussianRadialDeriv σ)) :=
  characteristicKernel_identifiesAtZero _ (gaussian_gradient_isCharacteristic σ hσ)

end Gaussian

section Legitimacy
variable {E : Type u} [MeasurableSpace E]

/-- The pair condition is legitimate: it admits distinct probability measures
(two Dirac masses) before any zero-drift hypothesis is imposed, so it does not
secretly encode `p = q`. -/
theorem bothProbability_allowsDistinctPair
    [Nontrivial E] [MeasurableSpace.SeparatesPoints E] :
    ConditionAllowsDistinctPair (BothProbability (E := E)) := by
  obtain ⟨a, b, hab⟩ := exists_pair_ne E
  exact ⟨Measure.dirac a, Measure.dirac b, ⟨inferInstance, inferInstance⟩,
    dirac_ne_dirac hab⟩

end Legitimacy

end DriftingIdentifiability
