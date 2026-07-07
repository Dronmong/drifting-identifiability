import DriftingIdentifiability.FiniteStability

/-!
# Objective 6: classifier-free guidance as affine densities

CFG in the paper is not, in general, a probability-measure statement.  Equation
(15) defines an effective negative density

```text
q̃ = (1 - γ) q + γ u,
```

and equation (16) solves `q̃ = p` as

```text
q = α p - (α - 1) u,   α = 1/(1-γ).
```

The right object is therefore an affine density/finite-coefficient vector:
coefficients sum to one, but may be negative.  This module keeps that boundary
visible.  It proves the finite-basis identifiability and stability algebra for
normalized affine coefficients, derives the CFG affine target, and provides an
explicit nonnegativity gate for recovering an ordinary probability vector.

No theorem here treats the CFG affine target as a probability law unless a
separate nonnegativity hypothesis is supplied.
-/

open scoped BigOperators
open MeasureTheory

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

variable {V : Type u} [NormedAddCommGroup V] [NormedSpace ℝ V]
  {m : ℕ}

/-! ## Affine coefficient vectors -/

/-- A finite coefficient vector with total mass one, but no nonnegativity
requirement.  This is the finite-dimensional representation appropriate for
CFG's affine/signed density target. -/
structure FiniteAffineVector (m : ℕ) where
  weight : Fin m → ℝ
  normalized : ∑ i, weight i = 1

namespace FiniteAffineVector

/-- Forgetting nonnegativity: every probability vector is an affine vector. -/
def ofProbabilityVector {m : ℕ} (a : FiniteProbabilityVector m) :
    FiniteAffineVector m where
  weight := a.weight
  normalized := a.normalized

/-- If an affine vector is nonnegative, it is an ordinary finite probability
vector.  This is the explicit gate needed before interpreting a CFG affine
target as a probability law. -/
def toProbabilityVector {m : ℕ} (a : FiniteAffineVector m)
    (hnonneg : ∀ i, 0 ≤ a.weight i) : FiniteProbabilityVector m where
  weight := a.weight
  nonnegative := hnonneg
  normalized := a.normalized

@[simp]
theorem ofProbabilityVector_weight {m : ℕ} (a : FiniteProbabilityVector m) :
    (ofProbabilityVector a).weight = a.weight := rfl

end FiniteAffineVector

/-- Vanishing coefficient minors identify normalized affine coefficient
vectors.  Nonnegativity is not used; total mass one is the only scale-fixing
condition needed. -/
theorem affineParallelCoefficientsAreEqual
    (m : ℕ) (a b : FiniteAffineVector m)
    (hminors : AllCoefficientMinorsZero a.weight b.weight) :
    a.weight = b.weight := by
  funext i
  have hsum : ∑ j, coefficientMinor a.weight b.weight i j = 0 := by
    apply Finset.sum_eq_zero
    intro j _hj
    exact hminors i j
  simp only [coefficientMinor, Finset.sum_sub_distrib] at hsum
  rw [← Finset.mul_sum, ← Finset.sum_mul, b.normalized, a.normalized,
    mul_one, one_mul] at hsum
  exact sub_eq_zero.mp hsum

/-! ## Affine finite-basis identifiability -/

/-- Algebraic finite setup for affine coefficients.  This is the CFG analogue
of `FiniteCoefficientSetup`: no nonnegativity/probability interpretation is
stored here. -/
structure AffineCoefficientSetup
    (E : Type u) [AddCommGroup E] [Module ℝ E]
    (m N : ℕ) where
  U : Fin m → Fin m → Fin N → E
  anti : AntiSymmetricBasisInteractions U
  nondegenerate : BasisInteractionNondegenerate U
  a : FiniteAffineVector m
  b : FiniteAffineVector m
  zeroBilinear : (∑ i, ∑ j, (a.weight i * b.weight j) • U i j) = 0

/-- The same anti-symmetric minor algebra proves vanishing minors for affine
coefficients. -/
theorem affineCoefficientMinorsVanish
    {E : Type u} [AddCommGroup E] [Module ℝ E]
    {m N : ℕ} (setup : AffineCoefficientSetup E m N) :
    AllCoefficientMinorsZero setup.a.weight setup.b.weight := by
  intro i j
  exact coefficientMinorsVanish_of_antisymm setup.U setup.anti setup.nondegenerate
    setup.a.weight setup.b.weight setup.zeroBilinear i j

/-- Finite affine coefficient vectors are identifiable under the same
anti-symmetry and interaction nondegeneracy condition. -/
theorem affineCoefficientIdentifiable
    {E : Type u} [AddCommGroup E] [Module ℝ E]
    {m N : ℕ} (setup : AffineCoefficientSetup E m N) :
    setup.a.weight = setup.b.weight :=
  affineParallelCoefficientsAreEqual m setup.a setup.b
    (affineCoefficientMinorsVanish setup)

/-- Equal identifiable affine coefficients yield equal represented densities
in their common finite basis. -/
theorem affineBasisDensitiesEqual
    {E : Type u} [AddCommGroup E] [Module ℝ E]
    {X : Type*} {m N : ℕ} (setup : AffineCoefficientSetup E m N)
    (φ : Fin m → X → ℝ) :
    basisDensity m φ setup.a.weight = basisDensity m φ setup.b.weight := by
  rw [affineCoefficientIdentifiable setup]

/-! ## Quantitative affine stability -/

/-- For normalized affine coefficient vectors, each coordinate difference is
the row sum of coefficient minors. -/
theorem affineCoeff_sub_eq_sum_minor (m : ℕ)
    (a b : FiniteAffineVector m) (i : Fin m) :
    a.weight i - b.weight i =
      ∑ j, coefficientMinor a.weight b.weight i j := by
  simp only [coefficientMinor, Finset.sum_sub_distrib]
  rw [← Finset.mul_sum, ← Finset.sum_mul, a.normalized, b.normalized,
    mul_one, one_mul]

/-- The `ℓ¹` affine coefficient distance is bounded by total minor mass. -/
theorem affineCoeff_l1_le_minor_mass (m : ℕ)
    (a b : FiniteAffineVector m) :
    ∑ i, |a.weight i - b.weight i|
      ≤ ∑ i, ∑ j, |coefficientMinor a.weight b.weight i j| := by
  apply Finset.sum_le_sum
  intro i _hi
  rw [affineCoeff_sub_eq_sum_minor m a b i]
  exact Finset.abs_sum_le_sum_abs _ _

/-- A frame lower bound turns affine drift control into coefficient control. -/
theorem affineFrame_mul_coeffL1_le_two_mul_driftNorm
    (U : Fin m → Fin m → V) (hanti : ∀ i j, U i j = -U j i)
    {c : ℝ} (hframe : InteractionFrameBound U c)
    (a b : FiniteAffineVector m) :
    c * (∑ i, |a.weight i - b.weight i|) ≤
      2 * ‖∑ i, ∑ j, (a.weight i * b.weight j) • U i j‖ := by
  have hcoeff := affineCoeff_l1_le_minor_mass m a b
  rw [minorMass_eq_two_strictMinorMass a.weight b.weight] at hcoeff
  have hbound := hframe.2 (strictMinorVector a.weight b.weight)
  rw [interactionSynthesis_apply] at hbound
  simp only [strictMinorVector, coefficientMinor] at hbound
  rw [← bilinear_eq_sum_strictMinor U hanti] at hbound
  have hc := mul_le_mul_of_nonneg_left hcoeff hframe.1.le
  simp only [strictMinorVector, coefficientMinor] at hc
  nlinarith

/-- Division form of affine coefficient stability. -/
theorem affineCoeffL1_le_two_div_frame_mul_driftNorm
    (U : Fin m → Fin m → V) (hanti : ∀ i j, U i j = -U j i)
    {c : ℝ} (hframe : InteractionFrameBound U c)
    (a b : FiniteAffineVector m) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 / c) * ‖∑ i, ∑ j, (a.weight i * b.weight j) • U i j‖ := by
  have h := affineFrame_mul_coeffL1_le_two_mul_driftNorm U hanti hframe a b
  rw [div_mul_eq_mul_div]
  apply (le_div_iff₀ hframe.1).2
  nlinarith

/-- Affine stability after a bounded pointwise rescaling. -/
theorem affineCoeffL1_le_of_frame_scaledDrift
    {N : ℕ} (U : Fin m → Fin m → Fin N → V)
    (hanti : ∀ i j, U i j = -U j i)
    {c B : ℝ} (hframe : InteractionFrameBound U c) (hB0 : 0 ≤ B)
    (a b : FiniteAffineVector m) (scale : Fin N → ℝ) (v : Fin N → V)
    (hscale : ∀ n, |scale n| ≤ B)
    (hbilinear : (∑ i, ∑ j, (a.weight i * b.weight j) • U i j) =
      fun n => scale n • v n) :
    (∑ i, |a.weight i - b.weight i|) ≤ (2 * B / c) * ‖v‖ := by
  have hcoeff := affineCoeffL1_le_two_div_frame_mul_driftNorm U hanti hframe a b
  rw [hbilinear] at hcoeff
  have hnorm : ‖fun n => scale n • v n‖ ≤ B * ‖v‖ := by
    apply (pi_norm_le_iff_of_nonneg (mul_nonneg hB0 (norm_nonneg v))).2
    intro n
    rw [norm_smul]
    exact mul_le_mul (hscale n) (norm_le_pi_norm v n) (norm_nonneg _) hB0
  calc
    (∑ i, |a.weight i - b.weight i|)
        ≤ (2 / c) * ‖fun n => scale n • v n‖ := hcoeff
    _ ≤ (2 / c) * (B * ‖v‖) :=
      mul_le_mul_of_nonneg_left hnorm (div_nonneg (by norm_num) hframe.1.le)
    _ = (2 * B / c) * ‖v‖ := by ring

/-! ## CFG coefficient algebra -/

/-- Equation (15) at the finite coefficient level:
`q̃ = (1-γ)q + γu`. -/
def cfgNegativeCoefficients {m : ℕ} (γ : ℝ)
    (q unconditional : FiniteAffineVector m) : FiniteAffineVector m where
  weight i := (1 - γ) * q.weight i + γ * unconditional.weight i
  normalized := by
    calc
      (∑ i, ((1 - γ) * q.weight i + γ * unconditional.weight i))
          = (1 - γ) * (∑ i, q.weight i) +
              γ * (∑ i, unconditional.weight i) := by
            rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      _ = 1 := by
            rw [q.normalized, unconditional.normalized]
            ring

/-- Equation (16) at the finite coefficient level:
`q = αp - (α-1)u`, with `α = 1/(1-γ)`. -/
noncomputable def cfgTargetCoefficients {m : ℕ} (γ : ℝ)
    (conditional unconditional : FiniteAffineVector m) :
    FiniteAffineVector m where
  weight i :=
    (1 / (1 - γ)) * conditional.weight i -
      ((1 / (1 - γ)) - 1) * unconditional.weight i
  normalized := by
    calc
      (∑ i,
        ((1 / (1 - γ)) * conditional.weight i -
          ((1 / (1 - γ)) - 1) * unconditional.weight i))
          = (1 / (1 - γ)) * (∑ i, conditional.weight i) -
              ((1 / (1 - γ)) - 1) * (∑ i, unconditional.weight i) := by
            rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      _ = 1 := by
            rw [conditional.normalized, unconditional.normalized]
            ring

@[simp]
theorem cfgNegativeCoefficients_weight {m : ℕ} (γ : ℝ)
    (q unconditional : FiniteAffineVector m) (i : Fin m) :
    (cfgNegativeCoefficients γ q unconditional).weight i =
      (1 - γ) * q.weight i + γ * unconditional.weight i := rfl

@[simp]
theorem cfgTargetCoefficients_weight {m : ℕ} (γ : ℝ)
    (conditional unconditional : FiniteAffineVector m) (i : Fin m) :
    (cfgTargetCoefficients γ conditional unconditional).weight i =
      (1 / (1 - γ)) * conditional.weight i -
        ((1 / (1 - γ)) - 1) * unconditional.weight i := rfl

/-- Solving equation (15): if the effective negative coefficients equal the
conditional coefficients, then the generated coefficients equal the CFG affine
target. -/
theorem cfgGenerated_eq_target_of_effective_eq
    {m : ℕ} {γ : ℝ} (hγ : γ ≠ 1)
    (generated conditional unconditional : FiniteAffineVector m)
    (hmatch :
      (cfgNegativeCoefficients γ generated unconditional).weight =
        conditional.weight) :
    generated.weight =
      (cfgTargetCoefficients γ conditional unconditional).weight := by
  funext i
  have hden : 1 - γ ≠ 0 := sub_ne_zero.mpr hγ.symm
  have hi := congrFun hmatch i
  simp only [cfgNegativeCoefficients_weight] at hi
  simp only [cfgTargetCoefficients_weight]
  field_simp [hden]
  nlinarith

/-- Conversely, substituting the CFG target into the effective negative
coefficients recovers the conditional coefficients. -/
theorem cfgEffective_eq_conditional_of_generated_eq_target
    {m : ℕ} {γ : ℝ} (hγ : γ ≠ 1)
    (generated conditional unconditional : FiniteAffineVector m)
    (hgen :
      generated.weight =
        (cfgTargetCoefficients γ conditional unconditional).weight) :
    (cfgNegativeCoefficients γ generated unconditional).weight =
      conditional.weight := by
  funext i
  have hden : 1 - γ ≠ 0 := sub_ne_zero.mpr hγ.symm
  have hi := congrFun hgen i
  simp only [cfgTargetCoefficients_weight] at hi
  simp only [cfgNegativeCoefficients_weight, hi]
  field_simp [hden]
  ring

/-- At `γ=0`, CFG has no affine extrapolation and the target is the conditional
density. -/
theorem cfgTargetCoefficients_zero_eq {m : ℕ}
    (conditional unconditional : FiniteAffineVector m) :
    (cfgTargetCoefficients 0 conditional unconditional).weight =
      conditional.weight := by
  funext i
  simp [cfgTargetCoefficients]

/-- Explicit nonnegativity gate for interpreting the CFG affine target as a
probability vector. -/
def CFGTargetNonnegative {m : ℕ} (γ : ℝ)
    (conditional unconditional : FiniteAffineVector m) : Prop :=
  ∀ i, 0 ≤ (cfgTargetCoefficients γ conditional unconditional).weight i

/-- A nonnegative CFG affine target is an ordinary finite probability vector. -/
noncomputable def cfgTargetProbabilityVector {m : ℕ} (γ : ℝ)
    (conditional unconditional : FiniteAffineVector m)
    (hnonneg : CFGTargetNonnegative γ conditional unconditional) :
    FiniteProbabilityVector m :=
  (cfgTargetCoefficients γ conditional unconditional).toProbabilityVector hnonneg

/-! ## CFG coefficient/density bridge -/

theorem basisDensity_cfgNegativeCoefficients
    {X : Type*} {m : ℕ} (φ : Fin m → X → ℝ) (γ : ℝ)
    (q unconditional : FiniteAffineVector m) :
    basisDensity m φ (cfgNegativeCoefficients γ q unconditional).weight =
      cfgNegativeDensity γ (basisDensity m φ q.weight)
        (basisDensity m φ unconditional.weight) := by
  funext x
  simp only [basisDensity, cfgNegativeDensity, cfgNegativeCoefficients_weight]
  calc
    (∑ i, ((1 - γ) * q.weight i + γ * unconditional.weight i) * φ i x)
        = ∑ i, ((1 - γ) * (q.weight i * φ i x) +
            γ * (unconditional.weight i * φ i x)) := by
          apply Finset.sum_congr rfl
          intro i _hi
          ring
    _ = (1 - γ) * (∑ i, q.weight i * φ i x) +
          γ * (∑ i, unconditional.weight i * φ i x) := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

theorem basisDensity_cfgTargetCoefficients
    {X : Type*} {m : ℕ} (φ : Fin m → X → ℝ) (γ : ℝ)
    (conditional unconditional : FiniteAffineVector m) :
    basisDensity m φ (cfgTargetCoefficients γ conditional unconditional).weight =
      fun x =>
        (1 / (1 - γ)) * basisDensity m φ conditional.weight x -
          ((1 / (1 - γ)) - 1) * basisDensity m φ unconditional.weight x := by
  funext x
  simp only [basisDensity, cfgTargetCoefficients_weight]
  calc
    (∑ i,
      ((1 / (1 - γ)) * conditional.weight i -
        (1 / (1 - γ) - 1) * unconditional.weight i) * φ i x)
        = ∑ i, ((1 / (1 - γ)) * (conditional.weight i * φ i x) -
            (1 / (1 - γ) - 1) * (unconditional.weight i * φ i x)) := by
          apply Finset.sum_congr rfl
          intro i _hi
          ring
    _ = (1 / (1 - γ)) * (∑ i, conditional.weight i * φ i x) -
          (1 / (1 - γ) - 1) * (∑ i, unconditional.weight i * φ i x) := by
          rw [Finset.sum_sub_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

/-! ## Weighted finite-sample CFG coefficients -/

/-- Appendix A.7's weighted finite-sample effective negative coefficients,
under the explicit denominator side condition needed for normalization. -/
noncomputable def cfgWeightedNegativeCoefficients {m : ℕ}
    (Nneg Nunc : ℕ) (w : ℝ)
    (hden : (Nneg : ℝ) - 1 + (Nunc : ℝ) * w ≠ 0)
    (q unconditional : FiniteAffineVector m) : FiniteAffineVector m where
  weight i :=
    (((Nneg : ℝ) - 1) * q.weight i +
      (Nunc : ℝ) * w * unconditional.weight i) /
      ((Nneg : ℝ) - 1 + (Nunc : ℝ) * w)
  normalized := by
    let D : ℝ := (Nneg : ℝ) - 1 + (Nunc : ℝ) * w
    calc
      (∑ i,
        (((Nneg : ℝ) - 1) * q.weight i +
          (Nunc : ℝ) * w * unconditional.weight i) / D)
          = (∑ i,
              (((Nneg : ℝ) - 1) * q.weight i +
                (Nunc : ℝ) * w * unconditional.weight i)) / D := by
            rw [Finset.sum_div]
      _ = (((Nneg : ℝ) - 1) * (∑ i, q.weight i) +
            (Nunc : ℝ) * w * (∑ i, unconditional.weight i)) / D := by
            rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]
      _ = 1 := by
            rw [q.normalized, unconditional.normalized]
            field_simp [D, hden]
            ring

/-- The weighted finite-sample coefficients are exactly equation (15) with the
paper's mixing rate. -/
theorem cfgWeightedNegativeCoefficients_eq_cfgNegativeCoefficients
    {m : ℕ} (Nneg Nunc : ℕ) (w : ℝ)
    (hden : (Nneg : ℝ) - 1 + (Nunc : ℝ) * w ≠ 0)
    (q unconditional : FiniteAffineVector m) :
    (cfgWeightedNegativeCoefficients Nneg Nunc w hden q unconditional).weight =
      (cfgNegativeCoefficients (cfgMixingRate Nneg Nunc w) q unconditional).weight := by
  funext i
  simp only [cfgWeightedNegativeCoefficients, cfgNegativeCoefficients_weight,
    cfgMixingRate]
  field_simp [hden]
  ring

/-- Coefficient version of the paper's weighted negative density identity. -/
theorem basisDensity_cfgWeightedNegativeCoefficients
    {X : Type*} {m : ℕ} (φ : Fin m → X → ℝ)
    (Nneg Nunc : ℕ) (w : ℝ)
    (hden : (Nneg : ℝ) - 1 + (Nunc : ℝ) * w ≠ 0)
    (q unconditional : FiniteAffineVector m) :
    basisDensity m φ
        (cfgWeightedNegativeCoefficients Nneg Nunc w hden q unconditional).weight =
      cfgWeightedNegativeDensity Nneg Nunc w
        (basisDensity m φ q.weight) (basisDensity m φ unconditional.weight) := by
  funext x
  simp only [basisDensity, cfgWeightedNegativeCoefficients,
    cfgWeightedNegativeDensity]
  let D : ℝ := (Nneg : ℝ) - 1 + (Nunc : ℝ) * w
  calc
    (∑ i,
      ((((Nneg : ℝ) - 1) * q.weight i +
        (Nunc : ℝ) * w * unconditional.weight i) / D) * φ i x)
        = ∑ i,
            ((((Nneg : ℝ) - 1) * q.weight i +
              (Nunc : ℝ) * w * unconditional.weight i) * φ i x) / D := by
          apply Finset.sum_congr rfl
          intro i _hi
          ring
    _ = (∑ i,
            (((Nneg : ℝ) - 1) * q.weight i +
              (Nunc : ℝ) * w * unconditional.weight i) * φ i x) / D := by
          rw [Finset.sum_div]
    _ = (∑ i,
            (((Nneg : ℝ) - 1) * (q.weight i * φ i x) +
              (Nunc : ℝ) * w * (unconditional.weight i * φ i x))) / D := by
          apply congrArg (fun z : ℝ => z / D)
          apply Finset.sum_congr rfl
          intro i _hi
          ring
    _ = (((Nneg : ℝ) - 1) * (∑ i, q.weight i * φ i x) +
          (Nunc : ℝ) * w * (∑ i, unconditional.weight i * φ i x)) / D := by
          rw [Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum]

/-! ## Drift-level CFG finite setup -/

section CFGDrift

variable {E : Type u} [MeasurableSpace E]
  [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- Drift-level affine finite setup: equation (31) plus anti-symmetry and
nondegeneracy identify affine coefficient vectors, not necessarily probability
measures. -/
structure AffineDriftFiniteSetup (E : Type u) [MeasurableSpace E]
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
  a : FiniteAffineVector m
  b : FiniteAffineVector m
  nondegenerate :
    BasisInteractionNondegenerate (inducedInteractionVector reference K φ probes)
  driftZero : ∀ n, densityInteractionDrift reference K
    (basisDensity m φ a.weight) (basisDensity m φ b.weight) (probes n) = 0

namespace AffineDriftFiniteSetup

variable {m N : ℕ}

theorem anti (setup : AffineDriftFiniteSetup E m N) :
    AntiSymmetricBasisInteractions
      (inducedInteractionVector setup.reference setup.K setup.φ setup.probes) := by
  haveI := setup.refProb
  exact antisymmetric_kernel_induces_basis_antisymmetry setup.reference setup.K
    setup.Kanti setup.φ setup.probes setup.regular

theorem zeroBilinear (setup : AffineDriftFiniteSetup E m N) :
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

noncomputable def toAffineCoefficientSetup (setup : AffineDriftFiniteSetup E m N) :
    AffineCoefficientSetup E m N where
  U := inducedInteractionVector setup.reference setup.K setup.φ setup.probes
  anti := setup.anti
  nondegenerate := setup.nondegenerate
  a := setup.a
  b := setup.b
  zeroBilinear := setup.zeroBilinear

theorem identifiesCoefficients (setup : AffineDriftFiniteSetup E m N) :
    setup.a.weight = setup.b.weight :=
  affineCoefficientIdentifiable setup.toAffineCoefficientSetup

theorem identifiesDensities (setup : AffineDriftFiniteSetup E m N) :
    basisDensity m setup.φ setup.a.weight =
      basisDensity m setup.φ setup.b.weight := by
  rw [setup.identifiesCoefficients]

end AffineDriftFiniteSetup

/-- CFG-specialized finite drift setup.  The drift compares the conditional
positive density against the effective negative density `(1-γ)q + γu`. -/
structure CFGDriftFiniteSetup (E : Type u) [MeasurableSpace E]
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
  γ : ℝ
  gamma_ne_one : γ ≠ 1
  conditional : FiniteAffineVector m
  generated : FiniteAffineVector m
  unconditional : FiniteAffineVector m
  nondegenerate : BasisInteractionNondegenerate
    (inducedInteractionVector reference K φ probes)
  driftZero : ∀ n, densityInteractionDrift reference K
    (basisDensity m φ conditional.weight)
    (basisDensity m φ
      (cfgNegativeCoefficients γ generated unconditional).weight)
    (probes n) = 0

namespace CFGDriftFiniteSetup

variable {m N : ℕ}

noncomputable def toAffineDriftFiniteSetup
    (setup : CFGDriftFiniteSetup E m N) : AffineDriftFiniteSetup E m N where
  reference := setup.reference
  refProb := setup.refProb
  K := setup.K
  Kanti := setup.Kanti
  φ := setup.φ
  probes := setup.probes
  regular := setup.regular
  a := setup.conditional
  b := cfgNegativeCoefficients setup.γ setup.generated setup.unconditional
  nondegenerate := setup.nondegenerate
  driftZero := setup.driftZero

/-- Zero drift identifies the effective negative coefficients with the
conditional coefficients. -/
theorem effectiveNegative_eq_conditional
    (setup : CFGDriftFiniteSetup E m N) :
    (cfgNegativeCoefficients setup.γ setup.generated setup.unconditional).weight =
      setup.conditional.weight := by
  symm
  exact setup.toAffineDriftFiniteSetup.identifiesCoefficients

/-- **CFG affine target theorem.**  If zero drift matches the effective
negative density to the conditional density, then the generated density is the
paper's affine CFG target `α p_cond - (α-1) p_uncond`. -/
theorem generated_eq_cfgTarget (setup : CFGDriftFiniteSetup E m N) :
    setup.generated.weight =
      (cfgTargetCoefficients setup.γ setup.conditional setup.unconditional).weight :=
  cfgGenerated_eq_target_of_effective_eq setup.gamma_ne_one
    setup.generated setup.conditional setup.unconditional
    setup.effectiveNegative_eq_conditional

theorem generatedDensity_eq_cfgTargetDensity
    (setup : CFGDriftFiniteSetup E m N) :
    basisDensity m setup.φ setup.generated.weight =
      basisDensity m setup.φ
        (cfgTargetCoefficients setup.γ setup.conditional setup.unconditional).weight := by
  rw [setup.generated_eq_cfgTarget]

end CFGDriftFiniteSetup

end CFGDrift

end PaperFiniteIdentifiability
end DriftingIdentifiability
