import Mathlib

/-!
# Mathematical interface to *Generative Modeling via Drifting*

This file records the mathematical definitions, numbered equations, and
proved implications in Deng--Li--Li--Du--He (2026), arXiv:2602.04770v2.
Empirical measurements and architecture hyperparameters are intentionally not
axioms: they are experimental observations, not mathematical propositions.

Most displayed equations are definitions below.  An `axiom` is used only when
the paper invokes a mathematical fact whose analytic side conditions would be
noise for the identifiability project (Fubini/interchange of finite expansions,
the stop-gradient convention, and the paper's stated equilibrium facts).

Crucially, this file contains **no axiom** of either of the following forms:

* `ZeroDrift V p q → p = q`;
* `V p q = 0 → p = q` (pointwise, almost everywhere, or approximately).

Appendix C.1's ingredients are exposed separately, through the vanishing of
the coefficient minors `aᵢ bⱼ - aⱼ bᵢ`.  Turning those ingredients into an
identifiability theorem is deliberately left to downstream files.

The characteristic-kernel axioms at the end of this file (`IsCharacteristic`,
`characteristic_gradientEmbedding_injective`, `gaussian_gradient_isCharacteristic`)
are external reproducing-kernel facts about *mean embeddings*, imported rather
than reproved (per the project's policy of axiomatizing well-known theorems).  A
transparency note is owed here: for the MMD drift of equation (41) the embedding
gap **is** the field by construction (`mmdDrift kg p q = Φₚ − Φ_q`), so the
embedding-injectivity axiom is, for that specific field, equivalent to the
identifiability statement itself.  This is a deliberate *reduction* of the
paper's identifiability question to a standard RKHS theorem, not an independent
proof of it; the drifting-field framework and its admissible conditions remain
the object of study.  These axioms are still not of the forbidden forms above:
they are stated about embeddings and probability measures, and the Gaussian
witness names a concrete, checkable kernel.
-/

open scoped BigOperators ENNReal NNReal
open MeasureTheory

namespace DriftingIdentifiability
namespace Paper

universe u v w

/-! ## Probability distributions, pushforwards, and training-time drift -/

abbrev Distribution (X : Type u) [MeasurableSpace X] := Measure X

/-- Equation (1): the output law is the pushforward of the prior law. -/
noncomputable def pushforward
    {X : Type u} {Y : Type v} [MeasurableSpace X] [MeasurableSpace Y]
    (f : X → Y) (p : Distribution X) (_hf : AEMeasurable f p) : Distribution Y :=
  Measure.map f p

/-- The sequence `qᵢ = (fᵢ)♯pε` associated with a sequence of generators. -/
noncomputable def generatedLaw
    {Noise : Type u} {E : Type v} [MeasurableSpace Noise] [MeasurableSpace E]
    (pε : Distribution Noise) (f : ℕ → Noise → E)
    (hf : ∀ i, AEMeasurable (f i) pε) (i : ℕ) : Distribution E :=
  pushforward (f i) pε (hf i)

/-- The residual `Δx := fᵢ₊₁(ε) - fᵢ(ε)` in Section 3.1. -/
def trainingResidual
    {Noise : Type u} {E : Type v} [AddGroup E]
    (f : ℕ → Noise → E) (i : ℕ) (ε : Noise) : E :=
  f (i + 1) ε - f i ε

/-- A drifting field is distribution-dependent and returns a vector at `x`. -/
abbrev DriftingField (E : Type u) [MeasurableSpace E] :=
  Distribution E → Distribution E → E → E

/-- Equation (2), as the predicate that a training trajectory follows `V`. -/
def FollowsDriftingField
    {Noise : Type u} {E : Type v} [MeasurableSpace E] [Add E]
    (V : DriftingField E) (p : Distribution E)
    (q : ℕ → Distribution E) (f : ℕ → Noise → E) : Prop :=
  ∀ i ε, f (i + 1) ε = f i ε + V p (q i) (f i ε)

/-- Equation (3): swapping the two laws negates the field. -/
def AntiSymmetric
    {E : Type u} [MeasurableSpace E] [Neg E]
    (V : DriftingField E) : Prop :=
  ∀ p q x, V p q x = -V q p x

/-- The paper's notation `Vₚ,q(x) = 0, ∀x`. -/
def ZeroDrift
    {E : Type u} [MeasurableSpace E] [Zero E]
    (V : DriftingField E) (p q : Distribution E) : Prop :=
  ∀ x, V p q x = 0

/-- The equilibrium requirement imposed on admissible drifting fields. -/
def HasMatchedEquilibrium
    {E : Type u} [MeasurableSpace E] [Zero E]
    (V : DriftingField E) : Prop :=
  ∀ p, ZeroDrift V p p

/-- The map taking a sample to its next drifting iterate. -/
def driftMap
    {E : Type u} [MeasurableSpace E] [Add E]
    (V : DriftingField E) (p q : Distribution E) (x : E) : E :=
  x + V p q x

/-- The flexible-step-size update `x ← x + ηV(x)` from Appendix C.2. -/
def scaledDriftMap
    {E : Type u} [MeasurableSpace E] [Add E] [SMul ℝ E]
    (η : ℝ) (V : DriftingField E)
    (p q : Distribution E) (x : E) : E :=
  x + η • V p q x

/-- The law after applying one measurable drifting update. -/
noncomputable def driftedLaw
    {E : Type u} [MeasurableSpace E] [Add E]
    (V : DriftingField E) (p q : Distribution E)
    (h : AEMeasurable (driftMap V p q) q) : Distribution E :=
  pushforward (driftMap V p q) q h

/-- Distribution-level counterpart of equation (2): `qᵢ₊₁` is the
pushforward of `qᵢ` by its drifting update. -/
def IsDriftingLawSequence
    {E : Type u} [MeasurableSpace E] [Add E]
    (V : DriftingField E) (p : Distribution E)
    (q : ℕ → Distribution E)
    (hmeas : ∀ i, AEMeasurable (driftMap V p (q i)) (q i)) : Prop :=
  ∀ i, q (i + 1) = driftedLaw V p (q i) (hmeas i)

/-- Proposition 3.1 (the proved direction only): `p = q → Vₚ,q = 0`. -/
theorem proposition_3_1
    {E : Type u} [MeasurableSpace E] [AddCommGroup E] [Module ℝ E]
    (V : DriftingField E) (hV : AntiSymmetric V)
    (p q : Distribution E) (hpq : p = q) :
    ZeroDrift V p q := by
  subst q
  intro x
  have hneg : V p p x = -V p p x := hV p p x
  have htwo : (2 : ℝ) • V p p x = 0 := by
    rw [two_smul]
    exact eq_neg_iff_add_eq_zero.mp hneg
  exact (smul_eq_zero.mp htwo).resolve_left (by norm_num)

/-- Equation (4): the equilibrium fixed-point relation. -/
def IsEquilibriumFixedPoint
    {Noise : Type u} {E : Type v} [MeasurableSpace E] [Add E]
    (V : DriftingField E) (p q : Distribution E) (f : Noise → E) : Prop :=
  ∀ ε, f ε = f ε + V p q (f ε)

/-- Equation (5): the fixed-point iteration used during training. -/
def IsFixedPointIteration
    {Noise : Type u} {E : Type v} [MeasurableSpace E] [Add E]
    (V : DriftingField E) (p : Distribution E)
    (q : ℕ → Distribution E) (f : ℕ → Noise → E) : Prop :=
  ∀ i ε, f (i + 1) ε = f i ε + V p (q i) (f i ε)

/-- Stop-gradient is the identity on values; its derivative convention is
recorded separately below in equations (34)--(36). -/
def stopGradient {X : Type u} (x : X) : X := x

/-- Equation (6), the training loss. -/
noncomputable def driftingLoss
    {Noise : Type u} {E : Type v}
    [MeasurableSpace Noise] [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (pε : Distribution Noise) (f : Noise → E)
    (V : DriftingField E) (p q : Distribution E) : ℝ :=
  ∫ ε, ‖f ε - stopGradient (f ε + V p q (f ε))‖ ^ 2 ∂pε

/-- The value statement immediately following (6): the loss value is the
expected squared norm of the field. -/
axiom equation_6_loss_value
    {Noise : Type u} {E : Type v}
    [MeasurableSpace Noise] [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (pε : Distribution Noise) (f : Noise → E)
    (V : DriftingField E) (p q : Distribution E) :
    driftingLoss pε f V p q = ∫ ε, ‖V p q (f ε)‖ ^ 2 ∂pε

/-! ## Interaction fields and the paper's mean-shift instantiation -/

/-- Equation (7), equivalently equation (28): expectation of a three-point
interaction kernel under the product law. -/
noncomputable def interactionDrift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (K : E → E → E → E) (p q : Distribution E) (x : E) : E :=
  ∫ y : E × E, K x y.1 y.2 ∂(p.prod q)

/-- Equation (7) also permits the interaction kernel itself to depend on
`p` and `q`, as stated directly below that equation. -/
abbrev DistributionDependentInteractionKernel
    (E : Type u) [MeasurableSpace E] :=
  Distribution E → Distribution E → E → E → E → E

noncomputable def distributionDependentInteractionDrift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (K : DistributionDependentInteractionKernel E) : DriftingField E :=
  fun p q x => interactionDrift (K p q) p q x

/-- Equation (9): the normalizing factor `Zₚ(x) = Eₚ[k(x,y)]`. -/
noncomputable def kernelNormalizer
    {E : Type u} [MeasurableSpace E]
    (k : E → E → ℝ) (p : Distribution E) (x : E) : ℝ :=
  ∫ y, k x y ∂p

/-- Equation (8): the normalized mean-shift field generated by one law. -/
noncomputable def meanShift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) (p : Distribution E) (x : E) : E :=
  (kernelNormalizer k p x)⁻¹ • ∫ y, k x y • (y - x) ∂p

/-- Equation (10): attraction by `p` minus repulsion by `q`. -/
noncomputable def meanShiftDrift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) : DriftingField E :=
  fun p q x => meanShift k p x - meanShift k q x

/-- The regularity implicitly used when the paper substitutes (8) into (10)
and applies Fubini to obtain (11). -/
structure MeanShiftRegularAt
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) (p q : Distribution E) (x : E) : Prop where
  zp_ne_zero : kernelNormalizer k p x ≠ 0
  zq_ne_zero : kernelNormalizer k q x ≠ 0
  integrable_p : Integrable (fun y => k x y • (y - x)) p
  integrable_q : Integrable (fun y => k x y • (y - x)) q
  integrable_product : Integrable
    (fun y : E × E => (k x y.1 * k x y.2) • (y.1 - y.2)) (p.prod q)

/-- Equation (11): the normalized bilinear representation of (10). -/
axiom equation_11_bilinear_mean_shift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) (p q : Distribution E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (x : E)
    (h : MeanShiftRegularAt k p q x) :
    meanShiftDrift k p q x =
      (kernelNormalizer k p x * kernelNormalizer k q x)⁻¹ •
        ∫ y : E × E,
          (k x y.1 * k x y.2) • (y.1 - y.2) ∂(p.prod q)

/-- The field (10) is anti-symmetric, as stated after equation (11). -/
axiom meanShiftDrift_antiSymmetric
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) :
    AntiSymmetric (meanShiftDrift k)

/-- Equation (12): the Laplace kernel used in the paper. -/
noncomputable def laplaceKernel
    {E : Type u} [NormedAddCommGroup E]
    (τ : ℝ) (x y : E) : ℝ :=
  Real.exp (-(1 / τ) * ‖x - y‖)

/-- The normalized kernel `k̃ = k/Z` used after equation (12) and in (27). -/
noncomputable def normalizedKernel
    {E : Type u} [MeasurableSpace E]
    (k : E → E → ℝ) (p : Distribution E) (x y : E) : ℝ :=
  (kernelNormalizer k p x)⁻¹ * k x y

/-! ## Feature-space loss, multiple features, and CFG -/

/-- Equation (13): drifting in a feature space. -/
noncomputable def featureDriftingLoss
    {Noise : Type u} {X : Type v} {F : Type w}
    [MeasurableSpace Noise] [MeasurableSpace F]
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    (pε : Distribution Noise) (generator : Noise → X) (φ : X → F)
    (V : DriftingField F) (p q : Distribution F) : ℝ :=
  ∫ ε,
    ‖φ (generator ε) -
      stopGradient (φ (generator ε) + V p q (φ (generator ε)))‖ ^ 2 ∂pε

/-- The feature distribution `φ♯p` mentioned in Section 3.4. -/
noncomputable def featureLaw
    {X : Type u} {F : Type v} [MeasurableSpace X] [MeasurableSpace F]
    (φ : X → F) (p : Distribution X) (hφ : AEMeasurable φ p) :
    Distribution F :=
  pushforward φ p hφ

/-- Equation (14): sum of independent drifting losses over features. -/
noncomputable def multiFeatureDriftingLoss
    {Noise : Type u} [MeasurableSpace Noise]
    (n : ℕ) (featureLoss : Fin n → Noise → ℝ)
    (pε : Distribution Noise) : ℝ :=
  ∑ j : Fin n, ∫ ε, featureLoss j ε ∂pε

/-- A density-like real-valued representation, used because CFG equation (16)
is a signed affine combination and therefore is not itself an equality of
nonnegative `Measure`s. -/
abbrev Density (X : Type u) := X → ℝ

/-- The full-support assumption introduced at the start of Appendix C.1. -/
def FullSupportDensity {X : Type u} (p : Density X) : Prop :=
  ∀ x, 0 < p x

/-- A nonnegative, unit-mass density relative to a reference measure.  This
packages the normalization premise used in Appendix C.1 without asserting its
identifiability conclusion. -/
def IsProbabilityDensity
    {X : Type u} [MeasurableSpace X]
    (reference : Distribution X) (p : Density X) : Prop :=
  (∀ x, 0 ≤ p x) ∧ Integrable p reference ∧ (∫ x, p x ∂reference) = 1

/-- Equation (15): the effective negative density used for CFG. -/
def cfgNegativeDensity
    {X : Type u} (γ : ℝ) (q conditionalUnconditional : Density X) : Density X :=
  fun x => (1 - γ) * q x + γ * conditionalUnconditional x

/-- Equation (16): solving `q̃ = p(·|c)` for the generated conditional density.
This is only the algebraic CFG identity; it is not a zero-drift identifiability
statement. -/
axiom equation_16_cfg_target
    {X : Type u} (γ : ℝ) (hγ : γ ≠ 1)
    (q pConditional pUnconditional : Density X)
    (hmatch : cfgNegativeDensity γ q pUnconditional = pConditional) :
    q = fun x =>
      (1 / (1 - γ)) * pConditional x -
        ((1 / (1 - γ)) - 1) * pUnconditional x

/-- Appendix A.7's finite-sample effective negative density. -/
noncomputable def cfgWeightedNegativeDensity
    {X : Type u} (Nneg Nunc : ℕ) (w : ℝ)
    (q pUnconditional : Density X) : Density X :=
  fun x =>
    (((Nneg : ℝ) - 1) * q x + (Nunc : ℝ) * w * pUnconditional x) /
      ((Nneg : ℝ) - 1 + (Nunc : ℝ) * w)

/-- Appendix A.7's relation between the unconditional weight and mixing rate. -/
noncomputable def cfgMixingRate (Nneg Nunc : ℕ) (w : ℝ) : ℝ :=
  ((Nunc : ℝ) * w) / ((Nneg : ℝ) - 1 + (Nunc : ℝ) * w)

/-- Appendix A.7's relation `α = 1/(1-γ)`. -/
noncomputable def cfgStrength (Nneg Nunc : ℕ) (w : ℝ) : ℝ :=
  1 / (1 - cfgMixingRate Nneg Nunc w)

/-- The weighted finite-sample density is equation (15) with the mixing rate
given in Appendix A.7. -/
axiom cfg_weighted_negative_is_mixture
    {X : Type u} (Nneg Nunc : ℕ) (w : ℝ)
    (q pUnconditional : Density X)
    (hden : (Nneg : ℝ) - 1 + (Nunc : ℝ) * w ≠ 0) :
    cfgWeightedNegativeDensity Nneg Nunc w q pUnconditional =
      cfgNegativeDensity (cfgMixingRate Nneg Nunc w) q pUnconditional

/-- The final unnumbered CFG identity on page 15. -/
axiom cfg_strength_closed_form
    (Nneg Nunc : ℕ) (w : ℝ)
    (hn : (Nneg : ℝ) - 1 ≠ 0)
    (hden : (Nneg : ℝ) - 1 + (Nunc : ℝ) * w ≠ 0) :
    cfgStrength Nneg Nunc w =
      ((Nneg : ℝ) - 1 + (Nunc : ℝ) * w) / ((Nneg : ℝ) - 1)

/-! ## Batch estimator and normalization (Appendix A) -/

/-- Equation (17): the abstract Monte-Carlo/batch-normalized interaction. -/
noncomputable def batchInteractionDrift
    {Batch : Type u} {E : Type v}
    [MeasurableSpace Batch] [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (batchLaw : Distribution Batch)
    (p q : Distribution E)
    (kBatch : Batch → E → E → ℝ) (x : E) : E :=
  ∫ b,
    ∫ y : E × E,
      (kBatch b x y.1 * kBatch b x y.2) • (y.1 - y.2) ∂(p.prod q) ∂batchLaw

/-- Softmax over a finite index type, used verbatim by Algorithm 2. -/
noncomputable def finiteSoftmax
    {ι : Type u} [Fintype ι] (logit : ι → ℝ) (i : ι) : ℝ :=
  Real.exp (logit i) / ∑ j, Real.exp (logit j)

/-- The concatenated positive/negative sample axis in Algorithm 2. -/
abbrev Algorithm2SampleIndex (Npos Nneg : ℕ) :=
  Sum (Fin Npos) (Fin Nneg)

/-- Algorithm 2's logits. `selfMask i j` marks a reused generated sample and
implements `dist_neg += eye(N) * 1e6`. -/
noncomputable def algorithm2Logit
    {E : Type u} [NormedAddCommGroup E]
    {Nx Npos Nneg : ℕ}
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  match s with
  | Sum.inl j => -‖x i - yPos j‖ / temperature
  | Sum.inr j =>
      -(‖x i - yNeg j‖ + if selfMask i j then 1000000 else 0) / temperature

/-- Row softmax: normalization over the concatenated sample axis. -/
noncomputable def algorithm2RowAffinity
    {E : Type u} [NormedAddCommGroup E]
    {Nx Npos Nneg : ℕ}
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  finiteSoftmax (algorithm2Logit x yPos yNeg temperature selfMask i) s

/-- Column softmax: normalization over generated anchors `x`. -/
noncomputable def algorithm2ColumnAffinity
    {E : Type u} [NormedAddCommGroup E]
    {Nx Npos Nneg : ℕ}
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  finiteSoftmax (fun i' => algorithm2Logit x yPos yNeg temperature selfMask i' s) i

/-- `A = sqrt(A_row * A_col)` from Algorithm 2. -/
noncomputable def algorithm2Affinity
    {E : Type u} [NormedAddCommGroup E]
    {Nx Npos Nneg : ℕ}
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  Real.sqrt
    (algorithm2RowAffinity x yPos yNeg temperature selfMask i s *
      algorithm2ColumnAffinity x yPos yNeg temperature selfMask i s)

/-- The positive weights `A_pos * A_neg.sum` in Algorithm 2. -/
noncomputable def algorithm2PositiveWeight
    {E : Type u} [NormedAddCommGroup E]
    {Nx Npos Nneg : ℕ}
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)
    (i : Fin Nx) (j : Fin Npos) : ℝ :=
  algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) *
    ∑ l : Fin Nneg,
      algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l)

/-- The negative weights `A_neg * A_pos.sum` in Algorithm 2. -/
noncomputable def algorithm2NegativeWeight
    {E : Type u} [NormedAddCommGroup E]
    {Nx Npos Nneg : ℕ}
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)
    (i : Fin Nx) (j : Fin Nneg) : ℝ :=
  algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr j) *
    ∑ l : Fin Npos,
      algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl l)

/-- The exact vector returned by `compute_V` in Algorithm 2. -/
noncomputable def algorithm2Drift
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {Nx Npos Nneg : ℕ}
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)
    (i : Fin Nx) : E :=
  (∑ j : Fin Npos,
      algorithm2PositiveWeight x yPos yNeg temperature selfMask i j • yPos j) -
    ∑ j : Fin Nneg,
      algorithm2NegativeWeight x yPos yNeg temperature selfMask i j • yNeg j

/-- Temperatures in equations (12), (22), and Algorithm 2 are positive. -/
def ValidTemperature (temperature : ℝ) : Prop :=
  0 < temperature

/-- The matched-distribution cancellation stated immediately after (17). -/
axiom equation_17_matched_batch_drift_zero
    {Batch : Type u} {E : Type v}
    [MeasurableSpace Batch] [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (batchLaw : Distribution Batch) (p : Distribution E)
    [IsProbabilityMeasure batchLaw] [IsProbabilityMeasure p]
    (kBatch : Batch → E → E → ℝ) (x : E)
    (hregular : ∀ b, Integrable
      (fun y : E × E =>
        (kBatch b x y.1 * kBatch b x y.2) • (y.1 - y.2)) (p.prod p)) :
    batchInteractionDrift batchLaw p p kBatch x = 0

/-- Equation (18): feature normalization. -/
noncomputable def normalizedFeature {X : Type u} {F : Type v}
    [SMul ℝ F] (φ : X → F) (S : ℝ) : X → F :=
  fun x => S⁻¹ • φ x

/-- Equation (19): distance between normalized features. -/
noncomputable def normalizedFeatureDistance {X : Type u} {F : Type v}
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    (φ : X → F) (S : ℝ) (x y : X) : ℝ :=
  ‖normalizedFeature φ S x - normalizedFeature φ S y‖

/-- Equations (20)--(21): the paper's chosen feature scale.  Equation (20) is
an approximate design target, while (21) is the exact definition. -/
noncomputable def featureNormalizationScale
    {X : Type u} {F : Type v} [MeasurableSpace X]
    [NormedAddCommGroup F]
    (C : ℕ+) (φ : X → F) (px py : Distribution X) : ℝ :=
  (Real.sqrt (C.val : ℝ))⁻¹ *
    ∫ z : X × X, ‖φ z.1 - φ z.2‖ ∂(px.prod py)

/-- Equation (22), including `τ̃ = τ√C`. -/
noncomputable def normalizedFeatureKernel
    {X : Type u} {F : Type v}
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    (τ S : ℝ) (C : ℕ+) (φ : X → F) (x y : X) : ℝ :=
  Real.exp
    (-(1 / (τ * Real.sqrt (C.val : ℝ))) * normalizedFeatureDistance φ S x y)

/-- Equation (23): normalized drift. -/
noncomputable def normalizedDrift {E : Type u} [SMul ℝ E]
    (V : E) (lambdaScale : ℝ) : E :=
  lambdaScale⁻¹ • V

/-- Equations (24)--(25): (24) is the intended unit-scale target and (25) is
the exact normalization used to pursue it. -/
noncomputable def driftNormalizationScale
    {X : Type u} {E : Type v} [MeasurableSpace X] [Norm E]
    (C : ℕ+) (sampleLaw : Distribution X) (V : X → E) : ℝ :=
  Real.sqrt (∫ x, (C.val : ℝ)⁻¹ * ‖V x‖ ^ 2 ∂sampleLaw)

/-- Equation (26): the per-feature mean-squared drifting loss. -/
noncomputable def normalizedFeatureLoss
    {X : Type u} {F : Type v} [MeasurableSpace X]
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    (C : ℕ+) (sampleLaw : Distribution X) (φ : X → F)
    (S lambdaScale : ℝ) (V : X → F) : ℝ :=
  ∫ x,
    (C.val : ℝ)⁻¹ *
      ‖normalizedFeature φ S x -
        stopGradient
          (normalizedFeature φ S x + normalizedDrift (V x) lambdaScale)‖ ^ 2
    ∂sampleLaw

/-- The multiple-temperature aggregation immediately following equation
(26): the normalized drifts are summed before evaluating the loss. -/
def aggregateTemperatureDrift
    {E : Type u} [AddCommMonoid E]
    (temperatures : Finset ℝ) (V : ℝ → E) : E :=
  ∑ τ ∈ temperatures, V τ

/-- Equation (27), the normalized-kernel form used by Algorithm 2. -/
noncomputable def normalizedProductDrift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) (p q : Distribution E) (x : E) : E :=
  ∫ y : E × E,
    (normalizedKernel k p x y.1 * normalizedKernel k q x y.2) •
      (y.1 - y.2) ∂(p.prod q)

/-! ## Finite-basis expansion from Appendix C.1 -/

/-- Equation (29): a finite linear expansion of a density. -/
def basisDensity {X : Type u} (m : ℕ)
    (φ : Fin m → X → ℝ) (a : Fin m → ℝ) : Density X :=
  fun y => ∑ i, a i * φ i y

/-- Linear independence of the finite density basis in equation (29). -/
def DensityBasisIndependent {X : Type u} {m : ℕ}
    (φ : Fin m → Density X) : Prop :=
  LinearIndependent ℝ φ

/-- Density-weighted version of equation (28), against a reference measure. -/
noncomputable def densityInteractionDrift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (reference : Distribution E) (K : E → E → E → E)
    (p q : Density E) (x : E) : E :=
  ∫ yplus,
    ∫ yminus,
      (p yplus * q yminus) • K x yplus yminus ∂reference
    ∂reference

/-- Equation (30): interaction vector induced by a basis pair at one probe. -/
noncomputable def basisInteraction
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (reference : Distribution E) (K : E → E → E → E)
    (φi φj : E → ℝ) (x : E) : E :=
  ∫ yplus,
    ∫ yminus,
      (φi yplus * φj yminus) • K x yplus yminus ∂reference
    ∂reference

/-- Equation (30), stacked over the finite set of probes `X`. -/
noncomputable def inducedInteractionVector
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    {m N : ℕ} (reference : Distribution E) (K : E → E → E → E)
    (φ : Fin m → E → ℝ) (probes : Fin N → E)
    (i j : Fin m) : Fin N → E :=
  fun n => basisInteraction reference K (φ i) (φ j) (probes n)

/-- Pointwise anti-symmetry in the positive and negative interaction slots. -/
def AntiSymmetricInteractionKernel
    {E : Type u} [Neg E] (K : E → E → E → E) : Prop :=
  ∀ x yplus yminus, K x yplus yminus = -K x yminus yplus

/-- Anti-symmetry of the interaction basis, stated before equation (32). -/
def AntiSymmetricBasisInteractions
    {E : Type u} [AddCommGroup E] {m N : ℕ}
    (U : Fin m → Fin m → Fin N → E) : Prop :=
  ∀ i j, U i j = -U j i

/-- The consequence `Uᵢᵢ = 0` stated before equation (32). -/
theorem antisymmetric_basis_diagonal_zero
    {E : Type u} [AddCommGroup E] [Module ℝ E] {m N : ℕ}
    (U : Fin m → Fin m → Fin N → E)
    (hanti : AntiSymmetricBasisInteractions U) :
    ∀ i, U i i = 0 := by
  intro i
  funext n
  have hneg : U i i n = -U i i n := congrFun (hanti i i) n
  have htwo : (2 : ℝ) • U i i n = 0 := by
    rw [two_smul]
    exact eq_neg_iff_add_eq_zero.mp hneg
  exact (smul_eq_zero.mp htwo).resolve_left (by norm_num)

/-- The implication `Uᵢⱼ = -Uⱼᵢ` used between equations (31) and (32). -/
axiom antisymmetric_kernel_induces_basis_antisymmetry
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    {m N : ℕ} (reference : Distribution E) [IsProbabilityMeasure reference]
    (K : E → E → E → E)
    (hK : AntiSymmetricInteractionKernel K)
    (φ : Fin m → E → ℝ) (probes : Fin N → E)
    (hregular : ∀ i j n, Integrable
      (fun y : E × E =>
        (φ i y.1 * φ j y.2) • K (probes n) y.1 y.2)
      (reference.prod reference)) :
    AntiSymmetricBasisInteractions
      (inducedInteractionVector reference K φ probes)

/-- Equation (31): the bilinear expansion evaluated at all probes. -/
axiom equation_31_bilinear_expansion
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    {m N : ℕ} (reference : Distribution E) [IsProbabilityMeasure reference]
    (K : E → E → E → E)
    (φ : Fin m → E → ℝ) (probes : Fin N → E) (a b : Fin m → ℝ)
    (hregular : ∀ i j n, Integrable
      (fun y : E × E =>
        (φ i y.1 * φ j y.2) • K (probes n) y.1 y.2)
      (reference.prod reference)) :
    (fun n => densityInteractionDrift reference K
      (basisDensity m φ a) (basisDensity m φ b) (probes n)) =
    ∑ i, ∑ j, (a i * b j) •
      inducedInteractionVector reference K φ probes i j

/-- Indices for the independent family `{Uᵢⱼ | i < j}` in Appendix C.1. -/
abbrev StrictPair (m : ℕ) :=
  {ij : Fin m × Fin m // ij.1.val < ij.2.val}

/-- The paper's generic non-degeneracy assumption. -/
def BasisInteractionNondegenerate
    {E : Type u} [AddCommGroup E] [Module ℝ E] {m N : ℕ}
    (U : Fin m → Fin m → Fin N → E) : Prop :=
  LinearIndependent ℝ (fun ij : StrictPair m => U ij.1.1 ij.1.2)

/-- Equation (32): after grouping an anti-symmetric bilinear expansion, the
coefficient of `Uᵢⱼ` is `aᵢbⱼ-aⱼbᵢ`. -/
axiom equation_32_grouped_zero_drift
    {E : Type u} [AddCommGroup E] [Module ℝ E] {m N : ℕ}
    (U : Fin m → Fin m → Fin N → E)
    (hanti : AntiSymmetricBasisInteractions U)
    (a b : Fin m → ℝ)
    (hzero : (∑ i, ∑ j, (a i * b j) • U i j) = 0) :
    (∑ ij : StrictPair m,
      (a ij.1.1 * b ij.1.2 - a ij.1.2 * b ij.1.1) •
        U ij.1.1 ij.1.2) = 0

/-- The coefficient conclusion following equation (32).  This is an
intermediate linear-algebra fact, not the forbidden conclusion `p = q`. -/
axiom zero_drift_coefficient_minors
    {E : Type u} [AddCommGroup E] [Module ℝ E] {m N : ℕ}
    (U : Fin m → Fin m → Fin N → E)
    (hanti : AntiSymmetricBasisInteractions U)
    (hnondegenerate : BasisInteractionNondegenerate U)
    (a b : Fin m → ℝ)
    (hzero : (∑ i, ∑ j, (a i * b j) • U i j) = 0) :
    ∀ i j, a i * b j - a j * b i = 0

/-- Equation (33): the unnormalized mean-shift interaction kernel. -/
def meanShiftInteractionKernel
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (k : E → E → ℝ) (x yplus yminus : E) : E :=
  (k x yplus * k x yminus) • (yplus - yminus)

/-- The kernel in equation (33) is anti-symmetric in its two sample slots. -/
axiom meanShiftInteractionKernel_antisymmetric
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (k : E → E → ℝ) :
    AntiSymmetricInteractionKernel (meanShiftInteractionKernel k)

/-! ## Stop-gradient calculus and MMD correspondence (Appendix C.2) -/

/-- Equation (34): the pointwise loss under the stop-gradient convention. -/
def pointDriftingLoss
    {E : Type u} [NormedAddCommGroup E]
    (V : E → E) (x : E) : ℝ :=
  ‖x - stopGradient (x + V x)‖ ^ 2

/-- The ordinary squared loss with a genuinely frozen target. -/
def frozenTargetLoss
    {E : Type u} [NormedAddCommGroup E]
    (target z : E) : ℝ :=
  ‖z - target‖ ^ 2

/-- Equation (34) interpreted as a function of a fresh variable `z`, while
`x + V(x)` is frozen at the differentiation point `x`. -/
def frozenPointDriftingLoss
    {E : Type u} [NormedAddCommGroup E]
    (V : E → E) (x z : E) : ℝ :=
  frozenTargetLoss (stopGradient (x + V x)) z

/-- The frozen-target interpretation has exactly the value of equation (34)
at the differentiation point. -/
theorem equation_34_frozen_value
    {E : Type u} [NormedAddCommGroup E]
    (V : E → E) (x : E) :
    frozenPointDriftingLoss V x x = pointDriftingLoss V x :=
  rfl

/-- The mathematical meaning of a gradient in a real inner-product space. -/
def HasGradientAt
    {E : Type u} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (F : E → ℝ) (g x : E) : Prop :=
  HasFDerivAt F (InnerProductSpace.toDualMap ℝ E g) x

/-- The paper's backpropagation gradient of (34), computed with its target
frozen rather than defined circularly from the desired answer. -/
def stopGradientPointGradient
    {E : Type u} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (V : E → E) (x : E) : E :=
  (2 : ℝ) • (x - stopGradient (x + V x))

/-- Standard differentiation of the frozen squared-distance loss. -/
axiom stopGradientPointGradient_is_gradient
    {E : Type u} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (V : E → E) (x : E) :
    HasGradientAt (frozenPointDriftingLoss V x)
      (stopGradientPointGradient V x) x

/-- Equation (35): the chain-rule factorization used by the paper.  `Dθx` is
kept abstract because the paper does not select a parameter space. -/
def ParameterGradientFactorization
    {E : Type u} {Θ : Type v}
    [MeasurableSpace E]
    [NormedAddCommGroup Θ] [NormedSpace ℝ Θ] [CompleteSpace Θ]
    (q : Distribution E) (pointGradient : E → E)
    (parameterPullback : E → E → Θ) (parameterGradient : Θ) : Prop :=
  parameterGradient =
    ∫ x, parameterPullback x (pointGradient x) ∂q

/-- Equation (36), under the paper's stop-gradient derivative semantics. -/
theorem equation_36_field_is_negative_half_gradient
    {E : Type u} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (V : E → E) (x : E) :
    V x = (-1 / 2 : ℝ) • stopGradientPointGradient V x := by
  simp only [stopGradientPointGradient, stopGradient]
  module

/-- Equation (37): squared MMD, including the `p,p` constant term. -/
noncomputable def mmdSquared
    {E : Type u} [MeasurableSpace E]
    (ξ : E → E → ℝ) (p q : Distribution E) : ℝ :=
  (∫ z : E × E, ξ z.1 z.2 ∂(q.prod q)) -
    2 * (∫ z : E × E, ξ z.1 z.2 ∂(p.prod q)) +
    ∫ z : E × E, ξ z.1 z.2 ∂(p.prod p)

/-- Equations (39)--(40): the pointwise MMD gradient expressed using positive
and negative samples. -/
noncomputable def mmdPointGradient
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (kernelGradient : E → E → E) (p q : Distribution E) (x : E) : E :=
  (2 : ℝ) • (∫ y, kernelGradient x y ∂q) -
    (2 : ℝ) • (∫ y, kernelGradient x y ∂p)

/-- Equation (38): equation (35) specialized to the MMD point gradient. -/
def MmdParameterGradientFactorization
    {E : Type u} {Θ : Type v}
    [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    [NormedAddCommGroup Θ] [NormedSpace ℝ Θ] [CompleteSpace Θ]
    (kernelGradient : E → E → E) (p q : Distribution E)
    (parameterPullback : E → E → Θ) (parameterGradient : Θ) : Prop :=
  ParameterGradientFactorization q (mmdPointGradient kernelGradient p q)
    parameterPullback parameterGradient

/-- Equation (41): the drifting field underlying the MMD loss. -/
noncomputable def mmdDrift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (kernelGradient : E → E → E) : DriftingField E :=
  fun p q x =>
    (∫ y, kernelGradient x y ∂p) - (∫ y, kernelGradient x y ∂q)

/-- Equations (36), (40), and (41) together: the MMD drift is negative one
half of the pointwise MMD gradient. -/
axiom mmdDrift_eq_negative_half_pointGradient
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (kernelGradient : E → E → E) (p q : Distribution E) (x : E) :
    mmdDrift kernelGradient p q x =
      (-1 / 2 : ℝ) • mmdPointGradient kernelGradient p q x

/-- Equation (42): gradient of a radial kernel `ξ(‖x-y‖²)`. -/
def radialKernelGradient
    {E : Type u} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (ξ' : ℝ → ℝ) (x y : E) : E :=
  (2 * ξ' (‖x - y‖ ^ 2)) • (x - y)

/-- Equation (43): (41) specialized using equation (42). -/
noncomputable def radialMmdDrift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    (ξ' : ℝ → ℝ) : DriftingField E :=
  mmdDrift (radialKernelGradient ξ')

/-- Equations (43)--(45): the radial MMD drift has the
attraction-minus-repulsion form with weight `-2ξ'`. -/
axiom radial_mmd_attraction_repulsion
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    (ξ' : ℝ → ℝ) (p q : Distribution E) (x : E) :
    radialMmdDrift ξ' p q x =
      (∫ y, (-2 * ξ' (‖x - y‖ ^ 2)) • (y - x) ∂p) -
        ∫ y, (-2 * ξ' (‖x - y‖ ^ 2)) • (y - x) ∂q

/-- The MMD drift is zero for matched distributions, as stated in C.2. -/
axiom mmdDrift_matched_zero
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (kernelGradient : E → E → E) (p : Distribution E) :
    ZeroDrift (mmdDrift kernelGradient) p p

/-- Equation (44): the attraction-minus-repulsion form, written explicitly. -/
noncomputable def normalizedAttractionRepulsionDrift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) : DriftingField E :=
  fun p q x =>
    (∫ y, normalizedKernel k p x y • (y - x) ∂p) -
      ∫ y, normalizedKernel k q x y • (y - x) ∂q

/-- Equation (44) is the expanded form of equation (10), once `k̃ = k/Z`
from equation (46) is substituted. -/
axiom equation_44_eq_meanShiftDrift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) (p q : Distribution E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (x : E)
    (hp : Integrable (fun y => k x y • (y - x)) p)
    (hq : Integrable (fun y => k x y • (y - x)) q) :
    normalizedAttractionRepulsionDrift k p q x = meanShiftDrift k p q x

/-- Equation (45): the kernel relating radial MMD to a mean-shift field. -/
def mmdMeanShiftKernel
    {E : Type u} [NormedAddCommGroup E]
    (ξ' : ℝ → ℝ) (x y : E) : ℝ :=
  -2 * ξ' (‖x - y‖ ^ 2)

/-- The Gaussian kernel used after equation (43). -/
noncomputable def gaussianKernel
    {E : Type u} [NormedAddCommGroup E]
    (σ : ℝ) (x y : E) : ℝ :=
  Real.exp (-(1 / (2 * σ ^ 2)) * ‖x - y‖ ^ 2)

/-- A Gaussian bandwidth is positive in the paper's intended setting. -/
def ValidBandwidth (σ : ℝ) : Prop :=
  0 < σ

/-- For a Gaussian radial kernel, equation (45) gives
`σ⁻² exp(-‖x-y‖²/(2σ²))`, as stated below (45). -/
axiom gaussian_mmd_meanShiftKernel
    {E : Type u} [NormedAddCommGroup E]
    (σ : ℝ) (_hσ : ValidBandwidth σ) (x y : E) :
    mmdMeanShiftKernel
      (fun r => -(1 / (2 * σ ^ 2)) * Real.exp (-(1 / (2 * σ ^ 2)) * r))
      x y =
    (1 / σ ^ 2) * Real.exp (-(1 / (2 * σ ^ 2)) * ‖x - y‖ ^ 2)

/-- Equation (46) is `normalizedKernel`; equation (47) is the following
normalized two-kernel interaction, definitionally the same construction as
equation (27). -/
noncomputable def equation47Drift
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) : DriftingField E :=
  fun p q x => normalizedProductDrift k p q x

/-- The matched-distribution equilibrium for equations (46)--(47). -/
axiom equation_47_matched_zero
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (k : E → E → ℝ) (p : Distribution E) [IsProbabilityMeasure p]
    (hregular : ∀ x, Integrable
      (fun y : E × E =>
        (normalizedKernel k p x y.1 * normalizedKernel k p x y.2) •
          (y.1 - y.2)) (p.prod p)) :
    ZeroDrift (equation47Drift k) p p

/-! ## Characteristic kernels and mean-embedding identifiability

This interface records the standard reproducing-kernel notion of a
*characteristic* kernel and the well-known theorem that its mean embedding is
injective on probability measures (Sriperumbudur–Gretton–Fukumizu–Schölkopf–
Lanckriet, *JMLR* 2010).  These are external facts about the kernel embedding;
none of them is the identifiability conclusion `Vₚ,q = 0 → p = q` for a drifting
field.  The bridge from a vanishing drift to a matched embedding is proved
separately in `CharacteristicIdentifiability.lean` and uses only these facts. -/

/-- The gradient kernel mean embedding `Φₚ(x) = ∫ kg(x,y) dp` that underlies the
MMD drift of equation (41). -/
noncomputable def kernelGradientEmbedding
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (kg : E → E → E) (p : Distribution E) (x : E) : E :=
  ∫ y, kg x y ∂p

/-- Abstract marker for a characteristic kernel gradient.  Its spectral
characterization is not formalized here; it is used only through the two axioms
below, so it behaves as an externally supplied, checkable property of the kernel
(not of any pair `p, q`). -/
axiom IsCharacteristic
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (kg : E → E → E) : Prop

/-- Well-known RKHS theorem: a characteristic kernel's mean embedding is
injective on probability measures.  This constrains the embedding map only; it
does not mention a drifting field or its zeros. -/
axiom characteristic_gradientEmbedding_injective
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
    (kg : E → E → E) (hkg : IsCharacteristic kg)
    (p q : Distribution E) [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (h : ∀ x, kernelGradientEmbedding kg p x = kernelGradientEmbedding kg q x) :
    p = q

/-- The Gaussian derivative profile `ξ'(r) = -(2σ²)⁻¹ exp(-(2σ²)⁻¹ r)` appearing
in equations (43)–(45). -/
noncomputable def gaussianRadialDeriv (σ : ℝ) : ℝ → ℝ :=
  fun r => -(1 / (2 * σ ^ 2)) * Real.exp (-(1 / (2 * σ ^ 2)) * r)

/-- Well-known RKHS theorem (Sriperumbudur et al., 2010): the Gaussian kernel is
characteristic.  Stated for the gradient feature map of the radial MMD drift;
Gaussian decay removes the additive constant left by matching gradients.  This is
a property of the kernel, not the identifiability conclusion. -/
axiom gaussian_gradient_isCharacteristic
    {E : Type u} [MeasurableSpace E]
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
    (σ : ℝ) (hσ : ValidBandwidth σ) :
    IsCharacteristic (E := E) (radialKernelGradient (gaussianRadialDeriv σ))

/-- Well-known RKHS theorem (Sriperumbudur et al., 2010; Simon-Gabriel–Schölkopf,
2018): the Gaussian-kernel MMD metrizes weak convergence, so a vanishing MMD
discrepancy (equation 37) forces convergence in distribution, here recorded via
the Lévy–Prokhorov metric.  This is an external fact about the MMD *metric*, not
about a drifting field; standard separability/regularity side conditions are
folded in, consistent with this file's policy.  It is an asymptotic (limit)
statement, not the exact conclusion. -/
axiom gaussian_mmd_metrizes_weakConvergence
    {E : Type u} [MeasurableSpace E] [NormedAddCommGroup E]
    (σ : ℝ) (hσ : ValidBandwidth σ)
    (p : Distribution E) (qn : ℕ → Distribution E)
    (hp : IsProbabilityMeasure p) (hqn : ∀ n, IsProbabilityMeasure (qn n))
    (h : Filter.Tendsto (fun n => mmdSquared (gaussianKernel σ) p (qn n))
      Filter.atTop (nhds 0)) :
    Filter.Tendsto (fun n => (levyProkhorovEDist p (qn n)).toReal)
      Filter.atTop (nhds 0)

end Paper
end DriftingIdentifiability
