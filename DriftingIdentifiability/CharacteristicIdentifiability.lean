import DriftingIdentifiability.TrustedBoundary

/-!
# Conditional distribution-level reduction to a characteristic kernel

This opt-in module reduces `Vₚ,q = 0 → p = q` for the paper's MMD drift to
external RKHS assumptions recorded in `Paperaxioms.lean`.

The condition is that the kernel gradient is **characteristic**
(`IsCharacteristic`) and is conditionally witnessed by the Gaussian kernel.
For this field, embedding injectivity is equivalent in substance to the target
implication; these theorems are reductions, not accepted project solutions.
-/

open MeasureTheory
open scoped BigOperators

namespace DriftingIdentifiability

open Paper

universe u

section Generic
variable {E : Type u} [MeasurableSpace E]
  [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
  [BorelSpace E] [SecondCountableTopology E]

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
  [FiniteDimensional ℝ E] [BorelSpace E] [SecondCountableTopology E]

/-- **Identifiability for the paper's Gaussian MMD drift.**  Specializing the
characteristic-kernel result to the Gaussian kernel (equations 43–45), which is
characteristic, zero drift of the radial MMD field identifies the target measure
among probability measures. -/
theorem gaussianMmd_identifiesAtZero (σ : ℝ) (hσ : ValidBandwidth σ) :
    IdentifiesAtZero (BothProbability (E := E))
      (radialMmdDrift (gaussianRadialDeriv σ)) :=
  characteristicKernel_identifiesAtZero _ (gaussian_gradient_isCharacteristic σ hσ)

end Gaussian

section Asymptotic
variable {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
  [InnerProductSpace ℝ E] [CompleteSpace E] [FiniteDimensional ℝ E]
  [BorelSpace E] [SecondCountableTopology E]

/-- **Asymptotic identifiability for the Gaussian MMD discrepancy.**  An instance
of `AsymptoticallyIdentifies` with the MMD discrepancy of equation (37) as the
drift size and the Lévy–Prokhorov metric as the distribution distance: if the
Gaussian MMD between `p` and a sequence `qₙ` tends to zero, then `qₙ → p` in
distribution.  Rests on the reviewed metrization axiom
`gaussian_mmd_metrizes_weakConvergence`.

Honesty note: the "drift size" named here is the **MMD training discrepancy**
(equation 37) — one of the discrepancy measures `AGENT.md` explicitly permits —
*not* the raw sup-norm of the drift field.  Tying the field norm itself to weak
convergence is genuinely subtle (a small drift *gradient* need not mean a small
MMD; see `LoggedFailures.md`) and is deliberately not claimed here. -/
theorem gaussianMmd_asymptoticallyIdentifies (σ : ℝ) (hσ : ValidBandwidth σ) :
    AsymptoticallyIdentifies (BothProbability (E := E))
      (fun p q => mmdSquared (gaussianKernel σ) p q)
      (fun p q => (levyProkhorovEDist p q).toReal) := by
  intro p qn hcond hdrift
  exact gaussian_mmd_metrizes_weakConvergence σ hσ p qn (hcond 0).1
    (fun n => (hcond n).2) hdrift

end Asymptotic

end DriftingIdentifiability
