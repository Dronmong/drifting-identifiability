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

/-- **Generic Chebyshev/Markov concentration.**  For any estimator error `Z`
with finite mean squared error, the probability that `‖Z‖` exceeds `ε` is at most
`E‖Z‖² / ε²`.  No independence or distributional assumption is used — this is
Markov applied to the squared norm.  For a sample-mean estimator over `N` iid
draws the mean squared error scales as `σ²/N`, giving the usual sample
complexity `N ≳ σ²/(δε²)`. -/
theorem meas_gt_le_meanSquare_div
    {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    {F : Type*} [NormedAddCommGroup F]
    (Z : Ω → F) (hZ : AEStronglyMeasurable Z P)
    (hint : Integrable (fun ω => ‖Z ω‖ ^ 2) P)
    {ε : ℝ} (hε : 0 < ε) :
    P {ω | ε < ‖Z ω‖} ≤ ENNReal.ofReal ((∫ ω, ‖Z ω‖ ^ 2 ∂P) / ε ^ 2) := by
  have hε2 : (0 : ℝ) < ε ^ 2 := by positivity
  have hfmeas : AEMeasurable (fun ω => ENNReal.ofReal (‖Z ω‖ ^ 2)) P :=
    ((hZ.norm.aemeasurable.pow_const 2).ennreal_ofReal)
  have hsubset : {ω | ε < ‖Z ω‖} ⊆
      {ω | ENNReal.ofReal (ε ^ 2) ≤ ENNReal.ofReal (‖Z ω‖ ^ 2)} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    exact ENNReal.ofReal_le_ofReal (by nlinarith [norm_nonneg (Z ω)])
  calc P {ω | ε < ‖Z ω‖}
      ≤ P {ω | ENNReal.ofReal (ε ^ 2) ≤ ENNReal.ofReal (‖Z ω‖ ^ 2)} :=
        measure_mono hsubset
    _ ≤ (∫⁻ ω, ENNReal.ofReal (‖Z ω‖ ^ 2) ∂P) / ENNReal.ofReal (ε ^ 2) :=
        meas_ge_le_lintegral_div hfmeas
          (ENNReal.ofReal_pos.mpr hε2).ne' ENNReal.ofReal_ne_top
    _ = ENNReal.ofReal (∫ ω, ‖Z ω‖ ^ 2 ∂P) / ENNReal.ofReal (ε ^ 2) := by
        rw [← ofReal_integral_eq_lintegral_ofReal hint
          (Filter.Eventually.of_forall fun _ => sq_nonneg _)]
    _ = ENNReal.ofReal ((∫ ω, ‖Z ω‖ ^ 2 ∂P) / ε ^ 2) := by
        rw [← ENNReal.ofReal_div_of_pos hε2]

/-- **Explicit `1/M` sample-mean concentration (Objective 4).**  Composing the
Markov bound with the well-known variance-of-a-sample-mean fact
(`Paper.sampleMean_meanSquare_le`): the empirical mean of `M` independent,
mean-zero, `L²` random vectors deviates from `0` by more than `ε` with
probability at most `σ²/(M ε²)`.  This is the concrete `1/M` rate that instantiates
the abstract concentration hypothesis of `estimate_failure_measure_le`; combined
with the bridge it gives the sample complexity `M ≥ σ²/(δε²)` for
identifiability accuracy `ε` at confidence `1-δ`. -/
theorem sampleMean_concentration
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
    [MeasurableSpace F] [BorelSpace F] [CompleteSpace F] [SecondCountableTopology F]
    {M : ℕ} (hM : 0 < M) (Z : Fin M → Ω → F)
    (hindep : ∀ i j, i ≠ j → ProbabilityTheory.IndepFun (Z i) (Z j) P)
    (hZmeas : ∀ i, Measurable (Z i))
    (haemeas : AEStronglyMeasurable (fun ω => (M : ℝ)⁻¹ • ∑ i, Z i ω) P)
    (hint2 : ∀ i, Integrable (fun ω => ‖Z i ω‖ ^ 2) P)
    (hintmean : Integrable (fun ω => ‖(M : ℝ)⁻¹ • ∑ i, Z i ω‖ ^ 2) P)
    (hmean : ∀ i, ∫ ω, Z i ω ∂P = 0)
    {σ ε : ℝ} (hε : 0 < ε) (hσ : ∀ i, ∫ ω, ‖Z i ω‖ ^ 2 ∂P ≤ σ ^ 2) :
    P {ω | ε < ‖(M : ℝ)⁻¹ • ∑ i, Z i ω‖} ≤ ENNReal.ofReal (σ ^ 2 / (M * ε ^ 2)) := by
  refine le_trans (meas_gt_le_meanSquare_div _ haemeas hintmean hε) ?_
  apply ENNReal.ofReal_le_ofReal
  have hmse := Paper.sampleMean_meanSquare_le P hM Z hindep hZmeas hint2 hmean hσ
  rw [show σ ^ 2 / ((M : ℝ) * ε ^ 2) = σ ^ 2 / (M : ℝ) / ε ^ 2 from (div_div _ _ _).symm]
  gcongr

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

/-- **Concrete finite-sample identifiability bound (Objective 4).**  Combining the
deterministic bridge, the high-probability lift, and Markov concentration: the
probability that the coefficient bound `‖a-b‖₁ ≤ (2B/c)(‖Vhat‖+ε)` *fails* is at
most the estimator mean squared error divided by `ε²`.

To guarantee the bound with probability `≥ 1-δ` it therefore suffices that
`E‖V - Vhat‖² ≤ δ ε²`.  For a sample-mean estimator with mean squared error
`σ²/N` this is the sample complexity `N ≥ σ²/(δ ε²)`. -/
theorem estimate_failure_le_meanSquare
    (setup : PopulationMeanShiftFiniteSetup E m N) {B ε : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n, |setup.normalizerProduct n| ≤ B) (hε : 0 < ε)
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (Vhat : Ω → (Fin N → E))
    (hmeas : AEStronglyMeasurable
      (fun ω => setup.normalizedProbeDrift - Vhat ω) P)
    (hint : Integrable
      (fun ω => ‖setup.normalizedProbeDrift - Vhat ω‖ ^ 2) P) :
    P {ω | (2 * B / setup.frameConstant) * (‖Vhat ω‖ + ε) <
        ∑ i, |setup.a.weight i - setup.b.weight i|}
      ≤ ENNReal.ofReal
          ((∫ ω, ‖setup.normalizedProbeDrift - Vhat ω‖ ^ 2 ∂P) / ε ^ 2) :=
  le_trans (setup.estimate_failure_measure_le hB0 hnormalizer P Vhat)
    (meas_gt_le_meanSquare_div _ hmeas hint hε)

end PopulationMeanShiftFiniteSetup

end PaperFiniteIdentifiability
end DriftingIdentifiability
