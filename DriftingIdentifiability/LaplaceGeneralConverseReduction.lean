import DriftingIdentifiability.LaplaceGeneralConverseBalance
import DriftingIdentifiability.LaplaceGeneralConverseEndgame

/-!
# Milestone 5 groundwork: the single-coefficient reduction

Milestones 3 and 4 give, under zero raw Laplace drift on `ℝ`:

* the **decomposition** `laplaceZeroDrift_decomposition_exp`:
  `e^{-2x/τ}·𝔞(x) + 𝔟(x) + e^{2x/τ}·𝔠(x) = 0` for all `x`
  (`𝔞 = truncatedPairing`, `𝔟 = middlePairing`, `𝔠 = upperPairing`); and
* the **balance identity** `laplaceBalance_identity_of_zeroDrift`:
  `e^{-2x/τ}·𝔞(x) = e^{2x/τ}·𝔠(x)` for all `x`
  (`scaledLowerPairing = scaledUpperPairing`).

Together these collapse the three coefficient families to one:

* `𝔟(x) = -2·e^{-2x/τ}·𝔞(x)` (`zeroDrift_middlePairing_eq_neg_two_scaledLower`);
* `𝔞(x) = 0 ↔ 𝔠(x) = 0` and `𝔞(x) = 0 ↔ 𝔟(x) = 0` pointwise.

Consequence (composing with the Milestone-2 endgame
`laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero`): **the identically
vanishing of ANY one of the three families forces `p = q`.**  So the open core
(Milestone 5, `ZeroDrift ⟹ 𝔞 ≡ 0`) may be attacked through whichever of the
lower/middle/upper coefficients is most convenient — they are equivalent.

These are pointwise algebraic corollaries of the certified Milestone-3/4
theorems; the genuine Milestone-5 analysis (forcing `𝔞 ≡ 0` for measures with
interior support) remains open, with strong numerical evidence that it holds
(linearized injectivity of `q ↦ meanShift_q` is full-rank across bandwidths and
densities; see the roadmap).
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-- Under zero drift the middle coefficient is minus twice the scaled lower
coefficient: `𝔟 = -2·e^{-2x/τ}·𝔞`. -/
theorem zeroDrift_middlePairing_eq_neg_two_scaledLower
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ) :
    middlePairing τ p q x = -2 * scaledLowerPairing τ p q x := by
  have hdecomp := laplaceZeroDrift_decomposition_exp τ hτ p q hzero x
  have hbal := laplaceBalance_identity_of_zeroDrift τ hτ p q hzero x
  simp only [scaledLowerPairing, scaledUpperPairing] at hbal ⊢
  linarith [hdecomp, hbal]

/-- Under zero drift, the lower and upper truncated pairings vanish at the same
points. -/
theorem zeroDrift_truncatedPairing_eq_zero_iff_upperPairing
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ) :
    truncatedPairing τ p q x = 0 ↔ upperPairing τ p q x = 0 := by
  have hbal := laplaceBalance_identity_of_zeroDrift τ hτ p q hzero x
  simp only [scaledLowerPairing, scaledUpperPairing] at hbal
  constructor
  · intro h
    have hz : Real.exp ((2 * x) / τ) * upperPairing τ p q x = 0 := by
      rw [← hbal, h, mul_zero]
    exact (mul_eq_zero.mp hz).resolve_left (Real.exp_ne_zero _)
  · intro h
    have hz : Real.exp (-(2 * x) / τ) * truncatedPairing τ p q x = 0 := by
      rw [hbal, h, mul_zero]
    exact (mul_eq_zero.mp hz).resolve_left (Real.exp_ne_zero _)

/-- Under zero drift, the lower and middle pairings vanish at the same points. -/
theorem zeroDrift_truncatedPairing_eq_zero_iff_middlePairing
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ) :
    truncatedPairing τ p q x = 0 ↔ middlePairing τ p q x = 0 := by
  have hm := zeroDrift_middlePairing_eq_neg_two_scaledLower τ hτ p q hzero x
  simp only [scaledLowerPairing] at hm
  constructor
  · intro h
    rw [hm, h, mul_zero, mul_zero]
  · intro h
    rw [h] at hm
    have hz : Real.exp (-(2 * x) / τ) * truncatedPairing τ p q x = 0 := by linarith [hm]
    exact (mul_eq_zero.mp hz).resolve_left (Real.exp_ne_zero _)

/-- **Reduction headline (upper).**  If the UPPER truncated pairing vanishes
identically, then zero drift forces `p = q`.  (Equivalent to the lower
condition of Milestone 2 by the balance identity.) -/
theorem laplaceZeroDrift_identifies_of_upperPairing_eq_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hup : ∀ x, upperPairing τ p q x = 0) : p = q :=
  laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero τ hτ p q
    (fun x => (zeroDrift_truncatedPairing_eq_zero_iff_upperPairing τ hτ p q hzero x).mpr (hup x))

/-- **Reduction headline (middle).**  If the MIDDLE pairing vanishes
identically, then zero drift forces `p = q`. -/
theorem laplaceZeroDrift_identifies_of_middlePairing_eq_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hmid : ∀ x, middlePairing τ p q x = 0) : p = q :=
  laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero τ hτ p q
    (fun x => (zeroDrift_truncatedPairing_eq_zero_iff_middlePairing τ hτ p q hzero x).mpr (hmid x))

end DriftingIdentifiability
