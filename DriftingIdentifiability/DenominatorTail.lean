import DriftingIdentifiability.FiniteSampleBridge
import DriftingIdentifiability.SelfNormalizedConsistency

/-!
# High-probability denominator refinement for the SNIS chain

The Objective-7 conditioning ledger (`numerics/RESULTS.md`, E5) measured that
the single dominant slack in the certified finite-sample chain is the
*deterministic* denominator floor `dmin = N * wmin` of
`selfNormalizedIndexed_meanSquare_le`, which pays the full worst-case kernel
value.  This module replaces it with checkable mean/variance data on the
weights:

* `weightSum_lower_tail_prob_le` — the random denominator `Σ wₗ(Yₗ)` falls
  below its mean sum by more than `t` with probability at most `N σw²/t²`
  (Chebyshev lower tail: center the weights, apply the reviewed sample-mean
  axiom, finish with the axiom-free Markov lemma).
* `selfNormalizedIndexed_deviation_prob_le` — splitting on the denominator
  event, the ratio estimator deviates from its target by more than `ε` with
  probability at most

  `(2Nσ² + 2N²b²)/((Σμw − t)² ε²)  +  N σw²/t²`,

  for every split point `0 < t < Σμw`.  The deterministic floor hypothesis is
  gone; in its place are the weight means `μw` and a centered second-moment
  bound `σw²`, both expectations of the weight function alone.

No new axiom: both results rest on `Paper.sampleMean_meanSquare_le` (applied
once to the vector summands and once to the scalar weights) plus the existing
Markov lemma.  Nothing here concerns identifiability itself — this sharpens
the constants of the finite-sample route only.
-/

open scoped BigOperators
open MeasureTheory ProbabilityTheory

namespace DriftingIdentifiability
namespace SelfNormalized

open Paper

universe u

/-- **Weight-sum lower tail.**  For pairwise-independent samples and bounded
per-slot weights with means `μw l` and centered second moments `≤ σw²`, the
random denominator falls below `Σ μw − t` with probability at most
`N σw²/t²`. -/
theorem weightSum_lower_tail_prob_le
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    {E : Type u} [MeasurableSpace E]
    {N : ℕ} (hN : 0 < N) (Y : Fin N → Ω → E) (w : Fin N → E → ℝ)
    (μw : Fin N → ℝ) {wmax σw t : ℝ}
    (hwmax : 0 ≤ wmax) (ht : 0 < t)
    (hYmeas : ∀ l, Measurable (Y l)) (hw : ∀ l, Measurable (w l))
    (hindep : ∀ l k, l ≠ k → IndepFun (Y l) (Y k) P)
    (hwabs : ∀ l ω, |w l (Y l ω)| ≤ wmax)
    (hμw : ∀ l, ∫ ω, w l (Y l ω) ∂P = μw l)
    (hσw : ∀ l, ∫ ω, (w l (Y l ω) - μw l) ^ 2 ∂P ≤ σw ^ 2) :
    P {ω | (∑ l, w l (Y l ω)) < (∑ l, μw l) - t} ≤
      ENNReal.ofReal ((N * σw ^ 2) / t ^ 2) := by
  set W : Fin N → Ω → ℝ := fun l ω => w l (Y l ω) - μw l with hWdef
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  -- measurability
  have hgw : ∀ l, Measurable (fun y : E => w l y - μw l) := fun l =>
    (hw l).sub measurable_const
  have hWmeas : ∀ l, Measurable (W l) := fun l => (hgw l).comp (hYmeas l)
  have hsum_meas : Measurable (fun ω => ∑ l, W l ω) :=
    Finset.measurable_sum _ (fun l _ => hWmeas l)
  -- weight means are bounded by the weight bound
  have hWint0 : ∀ l, Integrable (fun ω => w l (Y l ω)) P := fun l =>
    (integrable_const wmax).mono' ((hw l).comp (hYmeas l)).aestronglyMeasurable
      (Filter.Eventually.of_forall fun ω => by
        simpa [Real.norm_eq_abs] using hwabs l ω)
  have hμw_abs : ∀ l, |μw l| ≤ wmax := by
    intro l
    rw [← hμw l]
    calc |∫ ω, w l (Y l ω) ∂P| = ‖∫ ω, w l (Y l ω) ∂P‖ := (Real.norm_eq_abs _).symm
      _ ≤ ∫ ω, ‖w l (Y l ω)‖ ∂P := norm_integral_le_integral_norm _
      _ ≤ ∫ ω, wmax ∂P := by
          refine integral_mono (hWint0 l).norm (integrable_const _) fun ω => ?_
          simpa [Real.norm_eq_abs] using hwabs l ω
      _ = wmax := by simp
  have hWbd : ∀ l ω, ‖W l ω‖ ≤ 2 * wmax := by
    intro l ω
    calc ‖W l ω‖ ≤ ‖w l (Y l ω)‖ + ‖μw l‖ := norm_sub_le _ _
      _ ≤ wmax + wmax := by
          refine add_le_add ?_ ?_
          · simpa [Real.norm_eq_abs] using hwabs l ω
          · simpa [Real.norm_eq_abs] using hμw_abs l
      _ = 2 * wmax := by ring
  -- centered weights: mean zero, independent, second moments
  have hWmean : ∀ l, ∫ ω, W l ω ∂P = 0 := by
    intro l
    simp only [hWdef]
    rw [integral_sub (hWint0 l) (integrable_const _), hμw l, integral_const]
    simp
  have hWindep : ∀ l k, l ≠ k → IndepFun (W l) (W k) P := fun l k hlk =>
    (hindep l k hlk).comp (hgw l) (hgw k)
  have hWint2 : ∀ l, Integrable (fun ω => ‖W l ω‖ ^ 2) P := fun l =>
    (integrable_const ((2 * wmax) ^ 2)).mono'
      ((hWmeas l).norm.pow_const 2).aestronglyMeasurable
      (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
        nlinarith [hWbd l ω, norm_nonneg (W l ω)])
  have hWσ : ∀ l, ∫ ω, ‖W l ω‖ ^ 2 ∂P ≤ σw ^ 2 := by
    intro l
    have heq : (fun ω => ‖W l ω‖ ^ 2) = fun ω => (w l (Y l ω) - μw l) ^ 2 := by
      funext ω
      rw [Real.norm_eq_abs, sq_abs]
    rw [heq]
    exact hσw l
  -- the reviewed sample-mean axiom at F := ℝ
  have haxiomW : ∫ ω, ‖(N : ℝ)⁻¹ • ∑ l, W l ω‖ ^ 2 ∂P ≤ σw ^ 2 / N :=
    Paper.sampleMean_meanSquare_le P hN W hWindep hWint2 hWmean hWσ
  -- event inclusion: below-mean tail forces a large centered sum
  have hsub : {ω | (∑ l, w l (Y l ω)) < (∑ l, μw l) - t} ⊆
      {ω | t < ‖∑ l, W l ω‖} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have hsplit : (∑ l, W l ω) = (∑ l, w l (Y l ω)) - ∑ l, μw l := by
      simp only [hWdef]
      rw [Finset.sum_sub_distrib]
    rw [Real.norm_eq_abs, hsplit]
    calc t < -((∑ l, w l (Y l ω)) - ∑ l, μw l) := by linarith
      _ ≤ |(∑ l, w l (Y l ω)) - ∑ l, μw l| := neg_le_abs _
  -- integrability of the squared sum (bounded)
  have hWsum_bd : ∀ ω, ‖∑ l, W l ω‖ ≤ (N : ℝ) * (2 * wmax) := by
    intro ω
    calc ‖∑ l, W l ω‖ ≤ ∑ l, ‖W l ω‖ := norm_sum_le _ _
      _ ≤ ∑ _l : Fin N, (2 * wmax) := Finset.sum_le_sum fun l _ => hWbd l ω
      _ = (N : ℝ) * (2 * wmax) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hWsq_int : Integrable (fun ω => ‖∑ l, W l ω‖ ^ 2) P :=
    (integrable_const (((N : ℝ) * (2 * wmax)) ^ 2)).mono'
      (hsum_meas.norm.pow_const 2).aestronglyMeasurable
      (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
        nlinarith [hWsum_bd ω, norm_nonneg (∑ l, W l ω),
          mul_nonneg (le_of_lt hNpos) (by linarith : (0 : ℝ) ≤ 2 * wmax)])
  -- Markov on the squared norm, then the axiom
  have hmarkov := PaperFiniteIdentifiability.meas_gt_le_meanSquare_div
    (fun ω => ∑ l, W l ω) hsum_meas.aestronglyMeasurable hWsq_int ht
  refine le_trans (measure_mono hsub) (le_trans hmarkov (ENNReal.ofReal_le_ofReal ?_))
  rw [div_eq_mul_one_div (∫ ω, ‖∑ l, W l ω‖ ^ 2 ∂P),
    div_eq_mul_one_div ((N : ℝ) * σw ^ 2)]
  refine mul_le_mul_of_nonneg_right ?_ (by positivity)
  have hscaleW : ∀ ω,
      ‖∑ l, W l ω‖ ^ 2 = (N : ℝ) ^ 2 * ‖(N : ℝ)⁻¹ • ∑ l, W l ω‖ ^ 2 := by
    intro ω
    rw [norm_smul]
    simp only [Real.norm_eq_abs]
    rw [abs_of_pos (inv_pos.mpr hNpos), mul_pow]
    field_simp
  calc ∫ ω, ‖∑ l, W l ω‖ ^ 2 ∂P
      = (N : ℝ) ^ 2 * ∫ ω, ‖(N : ℝ)⁻¹ • ∑ l, W l ω‖ ^ 2 ∂P := by
        rw [← integral_const_mul]
        apply integral_congr_ae
        filter_upwards with ω
        rw [hscaleW ω]
    _ ≤ (N : ℝ) ^ 2 * (σw ^ 2 / N) :=
        mul_le_mul_of_nonneg_left haxiomW (by positivity)
    _ = (N : ℝ) * σw ^ 2 := by
        field_simp

/-- **Deviation probability with a random denominator.**  The indexed ratio
estimator deviates from its target by more than `ε` with probability at most
the numerator-concentration term plus the denominator lower tail.  The
deterministic floor hypothesis of `selfNormalizedIndexed_meanSquare_le` is
replaced by the weight means `μw` and centered second-moment bound `σw²` —
checkable expectations of the weight function alone — and the bound holds for
every split point `0 < t < Σ μw`. -/
theorem selfNormalizedIndexed_deviation_prob_le
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    {E : Type u} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
      [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]
    {N : ℕ} (hN : 0 < N) (Y : Fin N → Ω → E) (w : Fin N → E → ℝ) (c : E)
    (μ : Fin N → E) (μw : Fin N → ℝ) {wmax R b σ σw t ε : ℝ}
    (hwmax : 0 ≤ wmax) (_hR : 0 ≤ R) (hε : 0 < ε) (ht : 0 < t)
    (hdlow : 0 < (∑ l, μw l) - t)
    (hYmeas : ∀ l, Measurable (Y l)) (hw : ∀ l, Measurable (w l))
    (hindep : ∀ l k, l ≠ k → IndepFun (Y l) (Y k) P)
    (hwabs : ∀ l ω, |w l (Y l ω)| ≤ wmax)
    (hYbd : ∀ l ω, ‖Y l ω - c‖ ≤ R)
    (hμ : ∀ l, ∫ ω, w l (Y l ω) • (Y l ω - c) ∂P = μ l)
    (hb : ∀ l, ‖μ l‖ ≤ b)
    (hσ : ∀ l, ∫ ω, ‖w l (Y l ω) • (Y l ω - c) - μ l‖ ^ 2 ∂P ≤ σ ^ 2)
    (hμw : ∀ l, ∫ ω, w l (Y l ω) ∂P = μw l)
    (hσw : ∀ l, ∫ ω, (w l (Y l ω) - μw l) ^ 2 ∂P ≤ σw ^ 2) :
    P {ω | ε < ‖(∑ l, w l (Y l ω))⁻¹ • (∑ l, w l (Y l ω) • Y l ω) - c‖} ≤
      ENNReal.ofReal ((2 * N * σ ^ 2 + 2 * N ^ 2 * b ^ 2) /
          (((∑ l, μw l) - t) ^ 2 * ε ^ 2)) +
        ENNReal.ofReal ((N * σw ^ 2) / t ^ 2) := by
  set Z : Fin N → Ω → E := fun l ω => w l (Y l ω) • (Y l ω - c) with hZdef
  set Z' : Fin N → Ω → E := fun l ω => Z l ω - μ l with hZ'def
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  have hb0 : 0 ≤ b := le_trans (norm_nonneg (μ ⟨0, hN⟩)) (hb ⟨0, hN⟩)
  -- measurability
  have hgz : ∀ l, Measurable (fun y : E => w l y • (y - c)) := fun l =>
    (hw l).smul (measurable_id.sub measurable_const)
  have hgz' : ∀ l, Measurable (fun y : E => w l y • (y - c) - μ l) := fun l =>
    (hgz l).sub measurable_const
  have hZmeas : ∀ l, Measurable (Z l) := fun l => (hgz l).comp (hYmeas l)
  have hZ'meas : ∀ l, Measurable (Z' l) := fun l => (hZmeas l).sub measurable_const
  have hsummeas : Measurable (fun ω => ∑ l, Z l ω) :=
    Finset.measurable_sum _ (fun l _ => hZmeas l)
  have hsum'meas : Measurable (fun ω => ∑ l, Z' l ω) :=
    Finset.measurable_sum _ (fun l _ => hZ'meas l)
  -- per-sample bounds
  have hZbd : ∀ l ω, ‖Z l ω‖ ≤ wmax * R := by
    intro l ω
    simp only [hZdef, norm_smul, Real.norm_eq_abs]
    exact mul_le_mul (hwabs l ω) (hYbd l ω) (norm_nonneg _) hwmax
  have hZ'bd : ∀ l ω, ‖Z' l ω‖ ≤ wmax * R + b := by
    intro l ω
    calc ‖Z' l ω‖ ≤ ‖Z l ω‖ + ‖μ l‖ := norm_sub_le _ _
      _ ≤ wmax * R + b := add_le_add (hZbd l ω) (hb l)
  -- integrability and centered moments
  have hZint : ∀ l, Integrable (Z l) P := fun l =>
    (integrable_const (wmax * R)).mono' (hZmeas l).aestronglyMeasurable
      (Filter.Eventually.of_forall fun ω => by simpa using hZbd l ω)
  have hZ'mean : ∀ l, ∫ ω, Z' l ω ∂P = 0 := by
    intro l
    simp only [hZ'def]
    rw [integral_sub (hZint l) (integrable_const _),
      show ∫ ω, Z l ω ∂P = μ l from hμ l, integral_const]
    simp
  have hZ'indep : ∀ l k, l ≠ k → IndepFun (Z' l) (Z' k) P := fun l k hlk =>
    (hindep l k hlk).comp (hgz' l) (hgz' k)
  have hZ'int2 : ∀ l, Integrable (fun ω => ‖Z' l ω‖ ^ 2) P := fun l =>
    (integrable_const ((wmax * R + b) ^ 2)).mono'
      ((hZ'meas l).norm.pow_const 2).aestronglyMeasurable
      (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
        nlinarith [hZ'bd l ω, norm_nonneg (Z' l ω)])
  have haxiom : ∫ ω, ‖(N : ℝ)⁻¹ • ∑ l, Z' l ω‖ ^ 2 ∂P ≤ σ ^ 2 / N :=
    Paper.sampleMean_meanSquare_le P hN Z' hZ'indep hZ'int2 hZ'mean hσ
  -- numerator second moment
  have hμsum : ‖∑ l, μ l‖ ≤ (N : ℝ) * b := by
    calc ‖∑ l, μ l‖ ≤ ∑ l, ‖μ l‖ := norm_sum_le _ _
      _ ≤ ∑ _l : Fin N, b := Finset.sum_le_sum fun l _ => hb l
      _ = (N : ℝ) * b := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hZsplit : ∀ ω, (∑ l, Z l ω) = (∑ l, Z' l ω) + ∑ l, μ l := by
    intro ω
    rw [← Finset.sum_add_distrib]
    apply Finset.sum_congr rfl
    intro l _
    simp [hZ'def]
  have hscale : ∀ ω,
      ‖∑ l, Z' l ω‖ ^ 2 = (N : ℝ) ^ 2 * ‖(N : ℝ)⁻¹ • ∑ l, Z' l ω‖ ^ 2 := by
    intro ω
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hNpos), mul_pow]
    field_simp
  have hptw : ∀ ω, ‖∑ l, Z l ω‖ ^ 2 ≤
      (2 * (N : ℝ) ^ 2) * ‖(N : ℝ)⁻¹ • ∑ l, Z' l ω‖ ^ 2 + 2 * ((N : ℝ) * b) ^ 2 := by
    intro ω
    have h1 : ‖∑ l, Z l ω‖ ≤ ‖∑ l, Z' l ω‖ + (N : ℝ) * b := by
      rw [hZsplit ω]
      calc ‖(∑ l, Z' l ω) + ∑ l, μ l‖
          ≤ ‖∑ l, Z' l ω‖ + ‖∑ l, μ l‖ := norm_add_le _ _
        _ ≤ ‖∑ l, Z' l ω‖ + (N : ℝ) * b := by linarith [hμsum]
    have h2 : ‖∑ l, Z l ω‖ ^ 2 ≤ (‖∑ l, Z' l ω‖ + (N : ℝ) * b) ^ 2 := by
      nlinarith [norm_nonneg (∑ l, Z l ω), norm_nonneg (∑ l, Z' l ω),
        mul_nonneg (le_of_lt hNpos) hb0]
    have h3 : (‖∑ l, Z' l ω‖ + (N : ℝ) * b) ^ 2 ≤
        2 * ‖∑ l, Z' l ω‖ ^ 2 + 2 * ((N : ℝ) * b) ^ 2 := by
      nlinarith [sq_nonneg (‖∑ l, Z' l ω‖ - (N : ℝ) * b)]
    calc ‖∑ l, Z l ω‖ ^ 2
        ≤ 2 * ‖∑ l, Z' l ω‖ ^ 2 + 2 * ((N : ℝ) * b) ^ 2 := by linarith
      _ = (2 * (N : ℝ) ^ 2) * ‖(N : ℝ)⁻¹ • ∑ l, Z' l ω‖ ^ 2 + 2 * ((N : ℝ) * b) ^ 2 := by
          rw [hscale ω]
          ring
  have hZsum_bd : ∀ ω, ‖∑ l, Z l ω‖ ≤ (N : ℝ) * (wmax * R) := by
    intro ω
    calc ‖∑ l, Z l ω‖ ≤ ∑ l, ‖Z l ω‖ := norm_sum_le _ _
      _ ≤ ∑ _l : Fin N, (wmax * R) := Finset.sum_le_sum fun l _ => hZbd l ω
      _ = (N : ℝ) * (wmax * R) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hZsq_int : Integrable (fun ω => ‖∑ l, Z l ω‖ ^ 2) P :=
    (integrable_const (((N : ℝ) * (wmax * R)) ^ 2)).mono'
      (hsummeas.norm.pow_const 2).aestronglyMeasurable
      (Filter.Eventually.of_forall fun ω => by
        rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
        nlinarith [hZsum_bd ω, norm_nonneg (∑ l, Z l ω)])
  have hmsq_int : Integrable (fun ω => ‖(N : ℝ)⁻¹ • ∑ l, Z' l ω‖ ^ 2) P := by
    refine (integrable_const ((wmax * R + b) ^ 2)).mono'
      ((hsum'meas.const_smul _).norm.pow_const 2).aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    have hmean : ‖(N : ℝ)⁻¹ • (∑ l, Z' l ω)‖ ≤ wmax * R + b := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hNpos)]
      calc (N : ℝ)⁻¹ * ‖∑ l, Z' l ω‖
          ≤ (N : ℝ)⁻¹ * ∑ l, ‖Z' l ω‖ :=
            mul_le_mul_of_nonneg_left (norm_sum_le _ _) (by positivity)
        _ ≤ (N : ℝ)⁻¹ * ∑ _l : Fin N, (wmax * R + b) :=
            mul_le_mul_of_nonneg_left
              (Finset.sum_le_sum fun l _ => hZ'bd l ω) (by positivity)
        _ = wmax * R + b := by
            rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
              ← mul_assoc, inv_mul_cancel₀ (ne_of_gt hNpos), one_mul]
    nlinarith [hmean, norm_nonneg ((N : ℝ)⁻¹ • (∑ l, Z' l ω))]
  have hnum : ∫ ω, ‖∑ l, Z l ω‖ ^ 2 ∂P ≤ 2 * N * σ ^ 2 + 2 * N ^ 2 * b ^ 2 := by
    calc ∫ ω, ‖∑ l, Z l ω‖ ^ 2 ∂P
        ≤ ∫ ω, ((2 * (N : ℝ) ^ 2) * ‖(N : ℝ)⁻¹ • ∑ l, Z' l ω‖ ^ 2 +
            2 * ((N : ℝ) * b) ^ 2) ∂P :=
          integral_mono hZsq_int ((hmsq_int.const_mul _).add (integrable_const _)) hptw
      _ = (2 * (N : ℝ) ^ 2) * (∫ ω, ‖(N : ℝ)⁻¹ • ∑ l, Z' l ω‖ ^ 2 ∂P) +
            2 * ((N : ℝ) * b) ^ 2 := by
          rw [integral_add (hmsq_int.const_mul _) (integrable_const _),
            integral_const_mul, integral_const]
          simp
      _ ≤ (2 * (N : ℝ) ^ 2) * (σ ^ 2 / N) + 2 * ((N : ℝ) * b) ^ 2 := by
          have hcoef : (0 : ℝ) ≤ 2 * (N : ℝ) ^ 2 := by positivity
          have := mul_le_mul_of_nonneg_left haxiom hcoef
          linarith
      _ = 2 * N * σ ^ 2 + 2 * N ^ 2 * b ^ 2 := by
          field_simp
  -- event split on the denominator
  have hsplitset :
      {ω | ε < ‖(∑ l, w l (Y l ω))⁻¹ • (∑ l, w l (Y l ω) • Y l ω) - c‖} ⊆
        {ω | ((∑ l, μw l) - t) * ε < ‖∑ l, Z l ω‖} ∪
          {ω | (∑ l, w l (Y l ω)) < (∑ l, μw l) - t} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
    by_cases hD : (∑ l, μw l) - t ≤ ∑ l, w l (Y l ω)
    · left
      have hDpos : 0 < ∑ l, w l (Y l ω) := lt_of_lt_of_le hdlow hD
      have hpoint : (∑ l, w l (Y l ω))⁻¹ • (∑ l, w l (Y l ω) • Y l ω) - c
          = (∑ l, w l (Y l ω))⁻¹ • (∑ l, Z l ω) := by
        have hDne : (∑ l, w l (Y l ω)) ≠ 0 := hDpos.ne'
        have hsplit2 : (∑ l, Z l ω)
            = (∑ l, w l (Y l ω) • Y l ω) - (∑ l, w l (Y l ω)) • c := by
          rw [Finset.sum_smul, ← Finset.sum_sub_distrib]
          apply Finset.sum_congr rfl
          intro l _
          simp only [hZdef, smul_sub]
        rw [hsplit2, smul_sub, smul_smul, inv_mul_cancel₀ hDne, one_smul]
      rw [hpoint, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hDpos)] at hω
      have hinv : (∑ l, w l (Y l ω))⁻¹ ≤ ((∑ l, μw l) - t)⁻¹ := by
        simpa [one_div] using one_div_le_one_div_of_le hdlow hD
      have hεlt : ε < ((∑ l, μw l) - t)⁻¹ * ‖∑ l, Z l ω‖ :=
        lt_of_lt_of_le hω (mul_le_mul_of_nonneg_right hinv (norm_nonneg _))
      calc ((∑ l, μw l) - t) * ε
          < ((∑ l, μw l) - t) * (((∑ l, μw l) - t)⁻¹ * ‖∑ l, Z l ω‖) :=
            mul_lt_mul_of_pos_left hεlt hdlow
        _ = ‖∑ l, Z l ω‖ := by
            rw [← mul_assoc, mul_inv_cancel₀ hdlow.ne', one_mul]
    · right
      exact not_le.mp hD
  -- assemble via subadditivity, Markov, and the tail lemma
  have hterm1 : P {ω | ((∑ l, μw l) - t) * ε < ‖∑ l, Z l ω‖} ≤
      ENNReal.ofReal ((2 * N * σ ^ 2 + 2 * N ^ 2 * b ^ 2) /
        (((∑ l, μw l) - t) ^ 2 * ε ^ 2)) := by
    have hmarkov := PaperFiniteIdentifiability.meas_gt_le_meanSquare_div
      (fun ω => ∑ l, Z l ω) hsummeas.aestronglyMeasurable hZsq_int
      (mul_pos hdlow hε)
    refine le_trans hmarkov (ENNReal.ofReal_le_ofReal ?_)
    rw [mul_pow, div_eq_mul_one_div (∫ ω, ‖∑ l, Z l ω‖ ^ 2 ∂P),
      div_eq_mul_one_div (2 * (N : ℝ) * σ ^ 2 + 2 * (N : ℝ) ^ 2 * b ^ 2)]
    exact mul_le_mul_of_nonneg_right hnum (by positivity)
  have hterm2 := weightSum_lower_tail_prob_le P hN Y w μw hwmax ht
    hYmeas hw hindep hwabs hμw hσw
  calc P {ω | ε < ‖(∑ l, w l (Y l ω))⁻¹ • (∑ l, w l (Y l ω) • Y l ω) - c‖}
      ≤ P ({ω | ((∑ l, μw l) - t) * ε < ‖∑ l, Z l ω‖} ∪
          {ω | (∑ l, w l (Y l ω)) < (∑ l, μw l) - t}) :=
        measure_mono hsplitset
    _ ≤ P {ω | ((∑ l, μw l) - t) * ε < ‖∑ l, Z l ω‖} +
          P {ω | (∑ l, w l (Y l ω)) < (∑ l, μw l) - t} :=
        measure_union_le _ _
    _ ≤ ENNReal.ofReal ((2 * N * σ ^ 2 + 2 * N ^ 2 * b ^ 2) /
          (((∑ l, μw l) - t) ^ 2 * ε ^ 2)) +
        ENNReal.ofReal ((N * σw ^ 2) / t ^ 2) := add_le_add hterm1 hterm2

end SelfNormalized
end DriftingIdentifiability
