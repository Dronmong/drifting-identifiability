import DriftingIdentifiability.PopulationIdentifiability

/-!
# Finite-sample stability bridge (Objective 4)

Objectives 1--3 concern the ideal population field.  Practical training observes
a random minibatch estimator of that field, not the field itself.  This module
supplies the *deterministic* half of the finite-sample bridge: if an observed
estimate `Vhat` of the population probe-drift is within `ε` of the true value,
the `ℓ¹` coefficient error is controlled by the observed estimate plus the
estimation error, through the same `2B/c` conditioning constant.

This isolates the *statistical* content — a concentration bound producing `ε`
with high probability — as an explicit, independently checkable hypothesis; it
does not assume it.  The high-probability lift and a concrete Chebyshev
concentration are layered on top.
-/

open scoped BigOperators
open MeasureTheory

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

variable {E : Type u} [MeasurableSpace E]
  [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

namespace PopulationMeanShiftFiniteSetup

variable {m N : ℕ}

/-- **Deterministic finite-sample stability bridge (Objective 4).**  If an
observed estimate `Vhat` of the ideal population probe-drift lies within `ε`
(sup norm) of the true value, then the `ℓ¹` coefficient error is bounded by the
*observed* estimate norm plus the estimation error, scaled by `2B/c`.

The estimation error `ε` is an explicit input, discharged separately by a
concentration bound.  Setting `ε = 0` and `Vhat = normalizedProbeDrift` recovers
`coefficientStability`; the identifiability theorems are the special case where
the true probe-drift vanishes. -/
theorem coefficientStability_of_estimate
    (setup : PopulationMeanShiftFiniteSetup E m N) {B ε : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n, |setup.normalizerProduct n| ≤ B)
    (Vhat : Fin N → E)
    (hclose : ‖setup.normalizedProbeDrift - Vhat‖ ≤ ε) :
    (∑ i, |setup.a.weight i - setup.b.weight i|) ≤
      (2 * B / setup.frameConstant) * (‖Vhat‖ + ε) := by
  have hstab := setup.coefficientStability hB0 hnormalizer
  have htri : ‖setup.normalizedProbeDrift‖ ≤ ‖Vhat‖ + ε := by
    have hsplit : setup.normalizedProbeDrift
        = Vhat + (setup.normalizedProbeDrift - Vhat) := by abel
    calc ‖setup.normalizedProbeDrift‖
        = ‖Vhat + (setup.normalizedProbeDrift - Vhat)‖ := by rw [← hsplit]
      _ ≤ ‖Vhat‖ + ‖setup.normalizedProbeDrift - Vhat‖ := norm_add_le _ _
      _ ≤ ‖Vhat‖ + ε := by linarith
  have hfactor : 0 ≤ 2 * B / setup.frameConstant :=
    div_nonneg (by linarith) setup.frameBound.1.le
  calc (∑ i, |setup.a.weight i - setup.b.weight i|)
      ≤ (2 * B / setup.frameConstant) * ‖setup.normalizedProbeDrift‖ := hstab
    _ ≤ (2 * B / setup.frameConstant) * (‖Vhat‖ + ε) :=
        mul_le_mul_of_nonneg_left htri hfactor

/-- Small-observed-estimate corollary: once the observed estimate has norm at
most `η`, the coefficient error is at most `(2B/c)(η+ε)`.  In particular, driving
the *observed* minibatch drift to `≈0` and the estimation error to `≈0` forces
`a ≈ b`; exact equality needs both to be exactly zero. -/
theorem coefficientStability_of_estimate_le
    (setup : PopulationMeanShiftFiniteSetup E m N) {B ε η : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n, |setup.normalizerProduct n| ≤ B)
    (Vhat : Fin N → E)
    (hclose : ‖setup.normalizedProbeDrift - Vhat‖ ≤ ε)
    (hsmall : ‖Vhat‖ ≤ η) :
    (∑ i, |setup.a.weight i - setup.b.weight i|) ≤
      (2 * B / setup.frameConstant) * (η + ε) := by
  refine le_trans (setup.coefficientStability_of_estimate hB0 hnormalizer Vhat hclose) ?_
  have hfactor : 0 ≤ 2 * B / setup.frameConstant :=
    div_nonneg (by linarith) setup.frameBound.1.le
  exact mul_le_mul_of_nonneg_left (by linarith) hfactor

/-- **High-probability finite-sample identifiability (Objective 4).**  For a
*random* estimate `Vhat : Ω → (Fin N → E)` of the population probe-drift, the
event that the coefficient bound *fails* is contained in the event that the
estimate is *not* within `ε` of the truth.  Hence its probability is at most that
of the estimation-error event — no independence, measurability, or distributional
assumption is used, only monotonicity of the measure.

Composing with a concentration bound `P{ε < ‖V − Vhat‖} ≤ δ` yields
`P{coefficient bound holds} ≥ 1 - δ`. -/
theorem estimate_failure_measure_le
    (setup : PopulationMeanShiftFiniteSetup E m N) {B ε : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n, |setup.normalizerProduct n| ≤ B)
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (Vhat : Ω → (Fin N → E)) :
    P {ω | (2 * B / setup.frameConstant) * (‖Vhat ω‖ + ε) <
        ∑ i, |setup.a.weight i - setup.b.weight i|}
      ≤ P {ω | ε < ‖setup.normalizedProbeDrift - Vhat ω‖} := by
  apply measure_mono
  intro ω hω
  simp only [Set.mem_setOf_eq] at hω ⊢
  by_contra hle
  push Not at hle
  exact absurd (setup.coefficientStability_of_estimate hB0 hnormalizer (Vhat ω) hle)
    (not_le.mpr hω)

end PopulationMeanShiftFiniteSetup

end PaperFiniteIdentifiability
end DriftingIdentifiability
