import DriftingIdentifiability.ColumnReweightedTwoAtom

/-!
# EXTENSION TRACK: Sinkhorn-balanced drifting kernels

**This module is NOT part of the paper formalization.**  It certifies a
*proposed modification* of the paper's Algorithm 2, documented in
`SinkhornImplementation/PLAN.md`: iterating the geometric-mean row/column
balancing that the paper applies once (`A = sqrt(A_row * A_col)`).  Every
balancing iterate of a positive kernel matrix has the form
`u(x) * k(x,y) * v(y)` for positive scalings `u, v`, so certifying the whole
**orbit of positive diagonal rescalings** at once certifies every Sinkhorn
iterate — whatever the scalings turn out to be — without modeling the
batch-dependent balancing process itself.

Contents:

* `interactionFrameBound_of_probeScaling` — a positive per-probe (row) scaling
  transfers a certified frame bound with constant `ρmin * c`.
* `interactionFrameBound_of_biScaling` — simultaneous per-probe and per-pair
  scaling transfers with constant `ρmin * smin * c`; with
  `interactionFrameBound_of_strictPairScaling` this covers the full orbit
  shape `u(x)² · v(zᵢ)v(zⱼ) · U_ij(x)`.
* `orbitKernel`, `sinkhornOrbit01Setup` — the two-atom certified population
  setup for an arbitrary orbit kernel `u(x)·k(x,y)·v(y)` over the paper's
  exponential kernel, with the exact rescaling identity
  `inducedInteractionVector_sinkhornOrbit01_eq` and end-to-end
  identifiability/stability.
* `oneStepBalanced01Setup` — the explicit one-full-balancing-step kernel
  `k(x,y)/sqrt(r(x)·g(y))` (row mass over the atoms, column mass over the
  anchors), the population shadow of one geometric-mean Sinkhorn step of the
  anchor-by-atom matrix.

No new axioms: the scaling lemmas and rescaling identity are axiom-free, and
the identifiability theorems use only the reviewed equation-11/31/antisymmetry
machinery inherited from the population theorem.
-/

open scoped BigOperators
open MeasureTheory

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper Algorithm2

universe u

/-! ## Frame transfer under probe (row) and combined scalings -/

section Scaling

variable {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable {m N : ℕ}

/-- **Per-probe scaling transfer.**  If every probe coordinate of the
interaction family is rescaled by a factor bounded below by `ρmin > 0`, a
certified frame bound survives with constant `ρmin * c`. -/
theorem interactionFrameBound_of_probeScaling
    (U U' : Fin m → Fin m → Fin N → E) (ρ : Fin N → ℝ)
    {c ρmin : ℝ} (hframe : InteractionFrameBound U c) (hρmin : 0 < ρmin)
    (hρ : ∀ n, ρmin ≤ ρ n)
    (hU' : ∀ i j n, U' i j n = ρ n • U i j n) :
    InteractionFrameBound U' (ρmin * c) := by
  refine ⟨mul_pos hρmin hframe.1, fun z => ?_⟩
  have hρpos : ∀ n, 0 < ρ n := fun n => lt_of_lt_of_le hρmin (hρ n)
  have hsyn : ∀ n, interactionSynthesis U' z n = ρ n • interactionSynthesis U z n := by
    intro n
    rw [interactionSynthesis_apply, interactionSynthesis_apply, Finset.sum_apply,
      Finset.sum_apply, Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro p _
    rw [Pi.smul_apply, Pi.smul_apply, hU' p.1.1 p.1.2 n, smul_comm]
  have hnorm_le : ‖interactionSynthesis U z‖ ≤
      ρmin⁻¹ * ‖interactionSynthesis U' z‖ := by
    have h0 : 0 ≤ ρmin⁻¹ * ‖interactionSynthesis U' z‖ := by positivity
    rw [pi_norm_le_iff_of_nonneg h0]
    intro n
    have hval : interactionSynthesis U z n = (ρ n)⁻¹ • interactionSynthesis U' z n := by
      rw [hsyn n, smul_smul, inv_mul_cancel₀ (hρpos n).ne', one_smul]
    rw [hval, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr (hρpos n))]
    calc (ρ n)⁻¹ * ‖interactionSynthesis U' z n‖
        ≤ ρmin⁻¹ * ‖interactionSynthesis U' z n‖ := by
          refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
          simpa [one_div] using one_div_le_one_div_of_le hρmin (hρ n)
      _ ≤ ρmin⁻¹ * ‖interactionSynthesis U' z‖ :=
          mul_le_mul_of_nonneg_left (norm_le_pi_norm _ n) (by positivity)
  calc (ρmin * c) * (∑ p, |z p|) = ρmin * (c * ∑ p, |z p|) := by ring
    _ ≤ ρmin * ‖interactionSynthesis U z‖ :=
        mul_le_mul_of_nonneg_left (hframe.2 z) hρmin.le
    _ ≤ ‖interactionSynthesis U' z‖ := by
        have h := mul_le_mul_of_nonneg_left hnorm_le hρmin.le
        rwa [← mul_assoc, mul_inv_cancel₀ hρmin.ne', one_mul] at h

/-- **Bi-scaling transfer (the Sinkhorn-orbit shape).**  A simultaneous
positive per-probe scaling `ρ(n)` and per-strict-pair scaling `s(p)` — the
exact shape produced by a diagonal kernel rescaling `u(x)·k(x,y)·v(y)` —
transfers a certified frame bound with constant `ρmin * smin * c`. -/
theorem interactionFrameBound_of_biScaling
    (U U' : Fin m → Fin m → Fin N → E) (ρ : Fin N → ℝ) (s : StrictPair m → ℝ)
    {c ρmin smin : ℝ} (hframe : InteractionFrameBound U c)
    (hρmin : 0 < ρmin) (hsmin : 0 < smin)
    (hρ : ∀ n, ρmin ≤ ρ n) (hs : ∀ p, smin ≤ |s p|)
    (hU' : ∀ p : StrictPair m, ∀ n,
      U' p.1.1 p.1.2 n = ρ n • (s p • U p.1.1 p.1.2 n)) :
    InteractionFrameBound U' (ρmin * smin * c) := by
  refine ⟨mul_pos (mul_pos hρmin hsmin) hframe.1, fun z => ?_⟩
  have hρpos : ∀ n, 0 < ρ n := fun n => lt_of_lt_of_le hρmin (hρ n)
  have hsyn : ∀ n, interactionSynthesis U' z n =
      ρ n • interactionSynthesis U (fun q => z q * s q) n := by
    intro n
    rw [interactionSynthesis_apply, interactionSynthesis_apply, Finset.sum_apply,
      Finset.sum_apply, Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro p _
    rw [Pi.smul_apply, Pi.smul_apply, hU' p n]
    simp only [smul_smul]
    congr 1
    ring
  have hmass : smin * (∑ p, |z p|) ≤ ∑ p, |z p * s p| := by
    calc smin * (∑ p, |z p|) = ∑ p, smin * |z p| := by rw [Finset.mul_sum]
      _ ≤ ∑ p, |s p| * |z p| :=
          Finset.sum_le_sum fun p _ =>
            mul_le_mul_of_nonneg_right (hs p) (abs_nonneg _)
      _ = ∑ p, |z p * s p| := by
          apply Finset.sum_congr rfl
          intro p _
          rw [abs_mul, mul_comm]
  have hnorm_le : ‖interactionSynthesis U (fun q => z q * s q)‖ ≤
      ρmin⁻¹ * ‖interactionSynthesis U' z‖ := by
    have h0 : 0 ≤ ρmin⁻¹ * ‖interactionSynthesis U' z‖ := by positivity
    rw [pi_norm_le_iff_of_nonneg h0]
    intro n
    have hval : interactionSynthesis U (fun q => z q * s q) n =
        (ρ n)⁻¹ • interactionSynthesis U' z n := by
      rw [hsyn n, smul_smul, inv_mul_cancel₀ (hρpos n).ne', one_smul]
    rw [hval, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr (hρpos n))]
    calc (ρ n)⁻¹ * ‖interactionSynthesis U' z n‖
        ≤ ρmin⁻¹ * ‖interactionSynthesis U' z n‖ := by
          refine mul_le_mul_of_nonneg_right ?_ (norm_nonneg _)
          simpa [one_div] using one_div_le_one_div_of_le hρmin (hρ n)
      _ ≤ ρmin⁻¹ * ‖interactionSynthesis U' z‖ :=
          mul_le_mul_of_nonneg_left (norm_le_pi_norm _ n) (by positivity)
  calc (ρmin * smin * c) * (∑ p, |z p|)
      = ρmin * (c * (smin * ∑ p, |z p|)) := by ring
    _ ≤ ρmin * (c * ∑ p, |z p * s p|) := by
        refine mul_le_mul_of_nonneg_left ?_ hρmin.le
        exact mul_le_mul_of_nonneg_left hmass hframe.1.le
    _ ≤ ρmin * ‖interactionSynthesis U (fun q => z q * s q)‖ :=
        mul_le_mul_of_nonneg_left (hframe.2 _) hρmin.le
    _ ≤ ‖interactionSynthesis U' z‖ := by
        have h := mul_le_mul_of_nonneg_left hnorm_le hρmin.le
        rwa [← mul_assoc, mul_inv_cancel₀ hρmin.ne', one_mul] at h

end Scaling

/-! ## The two-atom orbit kernel and its certified setup -/

section Orbit

variable {N : ℕ}

/-- A point of the Sinkhorn orbit: the paper's exponential kernel rescaled by
arbitrary positive row and column functions.  Every geometric-mean balancing
iterate of the anchor-by-sample kernel matrix has this form. -/
noncomputable def orbitKernel (τ : ℝ) (u v : ℝ → ℝ) (x y : ℝ) : ℝ :=
  u x * algorithm2Kernel τ x y * v y

theorem orbitKernel_pos (τ : ℝ) {u v : ℝ → ℝ}
    (hu : ∀ x, 0 < u x) (hv : ∀ y, 0 < v y) (x y : ℝ) :
    0 < orbitKernel τ u v x y :=
  mul_pos (mul_pos (hu x) (algorithm2Kernel_pos τ x y)) (hv y)

/-- The paper kernel is continuous in the sample argument. -/
theorem algorithm2Kernel_continuous_snd (τ x : ℝ) :
    Continuous fun y : ℝ => algorithm2Kernel τ x y := by
  unfold Algorithm2.algorithm2Kernel
  exact Real.continuous_exp.comp
    (((continuous_const.sub continuous_id).norm).neg.div_const τ)

/-- At a fixed probe the orbit kernel is measurable in the sample argument. -/
theorem orbitKernel_measurable_snd (τ : ℝ) (u : ℝ → ℝ) {v : ℝ → ℝ}
    (hvm : Measurable v) (x : ℝ) :
    Measurable fun y : ℝ => orbitKernel τ u v x y := by
  unfold orbitKernel
  exact ((measurable_const.mul
    (algorithm2Kernel_continuous_snd τ x).measurable)).mul hvm

/-- The paired mean-shift interaction integrand of the orbit kernel is
measurable. -/
theorem orbitKernel_interaction_measurable (τ : ℝ) (u : ℝ → ℝ) {v : ℝ → ℝ}
    (hvm : Measurable v) (x : ℝ) :
    Measurable fun y : ℝ × ℝ =>
      meanShiftInteractionKernel (orbitKernel τ u v) x y.1 y.2 := by
  simp only [meanShiftInteractionKernel]
  exact (((orbitKernel_measurable_snd τ u hvm x).comp measurable_fst).mul
    ((orbitKernel_measurable_snd τ u hvm x).comp measurable_snd)).smul
    (measurable_fst.sub measurable_snd)

/-- All mean-shift regularity obligations hold for the orbit kernel on the
two-atom model, at every probe. -/
theorem sinkhornOrbit01_meanShiftRegular (τ : ℝ) {u v : ℝ → ℝ}
    (hu : ∀ x, 0 < u x) (hv : ∀ y, 0 < v y) (hvm : Measurable v)
    (x : ℝ) (a b : FiniteProbabilityVector 2) :
    MeanShiftRegularAt (orbitKernel τ u v)
      (empirical01Basis.basisMeasure a)
      (empirical01Basis.basisMeasure b) x :=
  { zp_ne_zero := ne_of_gt (kernelNormalizer_empirical01_pos _ x a
      (orbitKernel_pos τ hu hv x 0) (orbitKernel_pos τ hu hv x 1))
    zq_ne_zero := ne_of_gt (kernelNormalizer_empirical01_pos _ x b
      (orbitKernel_pos τ hu hv x 0) (orbitKernel_pos τ hu hv x 1))
    integrable_p := empirical01Basis_integrable a _
    integrable_q := empirical01Basis_integrable b _
    integrable_product := empirical01Basis_integrable_prod a b _
      (Measurable.aestronglyMeasurable (by
        exact (((orbitKernel_measurable_snd τ u hvm x).comp measurable_fst).mul
          ((orbitKernel_measurable_snd τ u hvm x).comp measurable_snd)).smul
          (measurable_fst.sub measurable_snd))) }

/-- The orbit interaction vector is nonzero: the closed two-atom form exposes
it as a positive multiple of `(0 - 1)`. -/
theorem inducedInteractionVector_sinkhornOrbit01_ne_zero [Nonempty (Fin N)]
    (anchors : Fin N → ℝ) (τ : ℝ) {u v : ℝ → ℝ}
    (hu : ∀ x, 0 < u x) (hv : ∀ y, 0 < v y) :
    inducedInteractionVector (empirical2 (0 : ℝ) 1)
      (meanShiftInteractionKernel (orbitKernel τ u v))
      empirical01Density anchors 0 1 ≠ 0 := by
  rw [Function.ne_iff]
  obtain ⟨n0⟩ := (inferInstance : Nonempty (Fin N))
  refine ⟨n0, ?_⟩
  simp only [inducedInteractionVector, basisInteraction_empirical2,
    empirical01Density_zero_zero, empirical01Density_one_one,
    empirical01Density_zero_one, empirical01Density_one_zero, Pi.zero_apply]
  apply smul_ne_zero
  · exact mul_ne_zero (by norm_num)
      (ne_of_gt (mul_pos (orbitKernel_pos τ hu hv (anchors n0) 0)
        (orbitKernel_pos τ hu hv (anchors n0) 1)))
  · norm_num

/-- **Exact orbit rescaling identity.**  Against the two-atom basis the orbit
interaction vector is the bare interaction vector rescaled per probe by
`u(xₙ)²` and per pair by `v(0)v(1)` — the exact shape consumed by
`interactionFrameBound_of_biScaling`.  Axiom-free. -/
theorem inducedInteractionVector_sinkhornOrbit01_eq
    (anchors : Fin N → ℝ) (τ : ℝ) (u v : ℝ → ℝ) :
    inducedInteractionVector (empirical2 (0 : ℝ) 1)
      (meanShiftInteractionKernel (orbitKernel τ u v))
      empirical01Density anchors 0 1 =
    fun n => (u (anchors n) ^ 2 * (v 0 * v 1)) •
      inducedInteractionVector (empirical2 (0 : ℝ) 1)
        (meanShiftInteractionKernel (algorithm2Kernel τ))
        empirical01Density anchors 0 1 n := by
  funext n
  simp only [inducedInteractionVector, basisInteraction_empirical2, smul_smul]
  congr 1
  simp only [orbitKernel]
  ring

/-- **Certified two-atom setup for the whole Sinkhorn orbit.**  For any
positive measurable rescalings `u, v`, the orbit kernel carries a complete
certified population setup: analytic obligations from positivity and the
atomic reference, the frame certified directly by kernel positivity. -/
noncomputable def sinkhornOrbit01Setup [Nonempty (Fin N)]
    (anchors : Fin N → ℝ) (τ : ℝ) (u v : ℝ → ℝ)
    (hu : ∀ x, 0 < u x) (hv : ∀ y, 0 < v y) (hvm : Measurable v)
    (a b : FiniteProbabilityVector 2) :
    PopulationMeanShiftFiniteSetup ℝ 2 N where
  reference := empirical2 0 1
  refProb := empirical2_isProbability 0 1
  basis := empirical01Basis
  kernel := orbitKernel τ u v
  probes := anchors
  a := a
  b := b
  meanShiftRegular n :=
    sinkhornOrbit01_meanShiftRegular τ hu hv hvm (anchors n) a b
  interactionIntegrable n := empirical01Basis_integrable_prod a b _
    (Measurable.aestronglyMeasurable
      (orbitKernel_interaction_measurable τ u hvm (anchors n)))
  basisInteractionIntegrable i j n := by
    apply integrable_empirical2_prod
    apply Measurable.aestronglyMeasurable
    have hi : Measurable fun y : ℝ × ℝ => empirical01Density i y.1 :=
      (empirical01Basis.measurable_density i).comp measurable_fst
    have hj : Measurable fun y : ℝ × ℝ => empirical01Density j y.2 :=
      (empirical01Basis.measurable_density j).comp measurable_snd
    exact (hi.mul hj).smul
      (orbitKernel_interaction_measurable τ u hvm (anchors n))
  frameConstant :=
    ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
      (meanShiftInteractionKernel (orbitKernel τ u v))
      empirical01Density anchors 0 1‖
  frameBound := interactionFrameBound_two _
    (inducedInteractionVector_sinkhornOrbit01_ne_zero anchors τ hu hv)

/-- **Orbit-invariant identifiability.**  Zero finite probe energy of the
population field for *any* positive-rescaled (Sinkhorn-orbit) kernel
identifies the two-atom mixtures.  The frame is certified inside the setup,
not assumed. -/
theorem sinkhornOrbit01_identifies_of_probeEnergy_eq_zero [Nonempty (Fin N)]
    (anchors : Fin N → ℝ) (τ : ℝ) (u v : ℝ → ℝ)
    (hu : ∀ x, 0 < u x) (hv : ∀ y, 0 < v y) (hvm : Measurable v)
    (a b : FiniteProbabilityVector 2)
    (henergy : (sinkhornOrbit01Setup anchors τ u v hu hv hvm
      a b).normalizedProbeDriftEnergy = 0) :
    empirical01Basis.basisMeasure a = empirical01Basis.basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero
    (sinkhornOrbit01Setup anchors τ u v hu hv hvm a b) henergy

/-- Quantitative orbit stability with an abstract normalizer bound `B`. -/
theorem sinkhornOrbit01_coefficientStability [Nonempty (Fin N)]
    (anchors : Fin N → ℝ) (τ : ℝ) (u v : ℝ → ℝ)
    (hu : ∀ x, 0 < u x) (hv : ∀ y, 0 < v y) (hvm : Measurable v)
    (a b : FiniteProbabilityVector 2) {B : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n, |(sinkhornOrbit01Setup anchors τ u v hu hv hvm
      a b).normalizerProduct n| ≤ B) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 * B / (sinkhornOrbit01Setup anchors τ u v hu hv hvm a b).frameConstant) *
        ‖(sinkhornOrbit01Setup anchors τ u v hu hv hvm
          a b).normalizedProbeDrift‖ := by
  have h := (sinkhornOrbit01Setup anchors τ u v hu hv hvm
    a b).coefficientStability hB0 hnormalizer
  simpa only [sinkhornOrbit01Setup] using h

end Orbit

/-! ## The explicit one-step balanced kernel -/

section OneStep

variable {N : ℕ}

/-- Row kernel mass over the two-atom support. -/
noncomputable def oneStepRowMass (τ x : ℝ) : ℝ :=
  algorithm2Kernel τ x 0 + algorithm2Kernel τ x 1

theorem oneStepRowMass_pos (τ x : ℝ) : 0 < oneStepRowMass τ x :=
  add_pos (algorithm2Kernel_pos τ x 0) (algorithm2Kernel_pos τ x 1)

theorem oneStepRowMass_measurable (τ : ℝ) : Measurable (oneStepRowMass τ) := by
  unfold oneStepRowMass
  exact ((Real.continuous_exp.comp
      (((continuous_id.sub continuous_const).norm).neg.div_const τ)).measurable).add
    ((Real.continuous_exp.comp
      (((continuous_id.sub continuous_const).norm).neg.div_const τ)).measurable)

/-- **One full geometric-mean balancing step.**  The population shadow of one
Sinkhorn iteration of the anchor-by-atom kernel matrix:
`k₁(x,y) = k(x,y)/sqrt(r(x)·g(y))` with `r` the row mass over the atoms and
`g` the column mass over the anchors.  This instantiates the orbit setup at
the scalings the paper's Algorithm 2 realizes after its bi-softmax. -/
noncomputable def oneStepBalanced01Setup [Nonempty (Fin N)]
    (anchors : Fin N → ℝ) (τ : ℝ) (a b : FiniteProbabilityVector 2) :
    PopulationMeanShiftFiniteSetup ℝ 2 N :=
  sinkhornOrbit01Setup anchors τ
    (fun x => (Real.sqrt (oneStepRowMass τ x))⁻¹)
    (fun y => (Real.sqrt (algorithm2ColumnKernelMass anchors τ y))⁻¹)
    (fun x => inv_pos.mpr (Real.sqrt_pos.mpr (oneStepRowMass_pos τ x)))
    (fun y => inv_pos.mpr (Real.sqrt_pos.mpr
      (algorithm2ColumnKernelMass_pos anchors τ y)))
    ((Real.continuous_sqrt.measurable.comp (by
      unfold Algorithm2.algorithm2ColumnKernelMass
      exact Finset.measurable_sum _ fun r _ =>
        (algorithm2Kernel_continuous_snd τ (anchors r)).measurable)).inv)
    a b

/-- Identifiability for the explicit one-step balanced kernel. -/
theorem oneStepBalanced01_identifies_of_probeEnergy_eq_zero [Nonempty (Fin N)]
    (anchors : Fin N → ℝ) (τ : ℝ) (a b : FiniteProbabilityVector 2)
    (henergy : (oneStepBalanced01Setup anchors τ
      a b).normalizedProbeDriftEnergy = 0) :
    empirical01Basis.basisMeasure a = empirical01Basis.basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero
    (oneStepBalanced01Setup anchors τ a b) henergy

end OneStep

end PaperFiniteIdentifiability
end DriftingIdentifiability
