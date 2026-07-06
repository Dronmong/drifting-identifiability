import DriftingIdentifiability.TrustedBoundary

/-!
# Self-normalized importance-sampling consistency (Objective 4)

This module proves the abstract L²-consistency of the **self-normalized (ratio)
estimator**, the engine behind Algorithm 2's softmax-weighted centroids.  It is
proved from the existing `Paper.sampleMean_meanSquare_le` axiom (variance of a
sample mean) — *no new axiom*.

The key observation is that with a weight bounded below by `wmin > 0`, the
denominator `Σⱼ w(Yⱼ)` is bounded below **deterministically** (`≥ N·wmin`), so the
ratio estimator error is exactly `Σ w(Yⱼ)⁻¹` times a **mean-zero** sample mean of
`w(Yⱼ)(Yⱼ − c)`, whose mean squared error is `σ²/N`.  No high-probability
denominator event and no delta method are needed.
-/

open scoped BigOperators
open MeasureTheory ProbabilityTheory

namespace DriftingIdentifiability
namespace SelfNormalized

open Paper

universe u

/-- **Self-normalized (ratio) estimator L²-consistency.**  For iid samples `Y`
with a weight `w` bounded in `[wmin, wmax]` (`wmin > 0`) and centered samples of
norm `≤ R`, the self-normalized estimator
`(Σⱼ w(Yⱼ)·Yⱼ)/(Σⱼ w(Yⱼ))` has mean squared error at most `σ²/(wmin²·N)` about the
target `c` characterized by `E[w(Y)(Y − c)] = 0` (i.e. `c = E[wY]/E[w]`).  Built on
`Paper.sampleMean_meanSquare_le`; introduces no new axiom. -/
theorem selfNormalized_meanSquare_le
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    {E : Type u} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
      [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]
    {N : ℕ} (hN : 0 < N) (Y : Fin N → Ω → E) (w : E → ℝ) (c : E)
    {wmin wmax R σ : ℝ} (hwmin : 0 < wmin) (hwmax : 0 ≤ wmax) (_hR : 0 ≤ R)
    (hYmeas : ∀ i, Measurable (Y i)) (hw : Measurable w)
    (hindep : ∀ i j, i ≠ j → IndepFun (Y i) (Y j) P)
    (hwlb : ∀ i ω, wmin ≤ w (Y i ω)) (hwub : ∀ i ω, w (Y i ω) ≤ wmax)
    (hYbd : ∀ i ω, ‖Y i ω - c‖ ≤ R)
    (hmean : ∀ i, ∫ ω, w (Y i ω) • (Y i ω - c) ∂P = 0)
    (hσ : ∀ i, ∫ ω, ‖w (Y i ω) • (Y i ω - c)‖ ^ 2 ∂P ≤ σ ^ 2) :
    ∫ ω, ‖(∑ i, w (Y i ω))⁻¹ • (∑ i, w (Y i ω) • Y i ω) - c‖ ^ 2 ∂P
      ≤ σ ^ 2 / (wmin ^ 2 * N) := by
  set Z : Fin N → Ω → E := fun i ω => w (Y i ω) • (Y i ω - c) with hZdef
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  -- measurability building blocks
  have hZmeas : ∀ i, Measurable (Z i) := fun i =>
    (hw.comp (hYmeas i)).smul ((hYmeas i).sub measurable_const)
  have hg : Measurable (fun y : E => w y • (y - c)) :=
    hw.smul (measurable_id.sub measurable_const)
  have hsummeas : Measurable (fun ω => ∑ i, Z i ω) :=
    Finset.measurable_sum _ (fun i _ => hZmeas i)
  have hDmeas : Measurable (fun ω => ∑ i, w (Y i ω)) :=
    Finset.measurable_sum _ (fun i _ => hw.comp (hYmeas i))
  have hNummeas : Measurable (fun ω => ∑ i, w (Y i ω) • Y i ω) :=
    Finset.measurable_sum _ (fun i _ => (hw.comp (hYmeas i)).smul (hYmeas i))
  have hcmeas : Measurable
      (fun ω => (∑ i, w (Y i ω))⁻¹ • (∑ i, w (Y i ω) • Y i ω) - c) :=
    (hDmeas.inv.smul hNummeas).sub measurable_const
  -- denominator lower bound: Σ w(Yᵢ) ≥ N·wmin > 0
  have hge : ∀ ω, (N : ℝ) * wmin ≤ ∑ i, w (Y i ω) := by
    intro ω
    have h := Finset.sum_le_sum (fun i (_ : i ∈ Finset.univ) => hwlb i ω)
    simpa [Finset.sum_const, Finset.card_univ, nsmul_eq_mul] using h
  have hDpos : ∀ ω, 0 < ∑ i, w (Y i ω) := fun ω =>
    lt_of_lt_of_le (mul_pos hNpos hwmin) (hge ω)
  -- pointwise identity: ĉ − c = D⁻¹ • Σ Z
  have hpoint : ∀ ω, (∑ i, w (Y i ω))⁻¹ • (∑ i, w (Y i ω) • Y i ω) - c
      = (∑ i, w (Y i ω))⁻¹ • (∑ i, Z i ω) := by
    intro ω
    have hD : (∑ i, w (Y i ω)) ≠ 0 := (hDpos ω).ne'
    have hsplit : (∑ i, Z i ω)
        = (∑ i, w (Y i ω) • Y i ω) - (∑ i, w (Y i ω)) • c := by
      rw [Finset.sum_smul, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro i _
      simp only [hZdef, smul_sub]
    rw [hsplit, smul_sub, smul_smul, inv_mul_cancel₀ hD, one_smul]
  -- per-sample bound ‖Z i‖ ≤ wmax·R
  have hZbd : ∀ i ω, ‖Z i ω‖ ≤ wmax * R := by
    intro i ω
    simp only [hZdef, norm_smul, Real.norm_eq_abs,
      abs_of_pos (lt_of_lt_of_le hwmin (hwlb i ω))]
    exact mul_le_mul (hwub i ω) (hYbd i ω) (norm_nonneg _) hwmax
  -- bound ‖N⁻¹ Σ Z‖ ≤ wmax·R
  have hmeanbd : ∀ ω, ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖ ≤ wmax * R := by
    intro ω
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hNpos)]
    calc (N : ℝ)⁻¹ * ‖∑ i, Z i ω‖
        ≤ (N : ℝ)⁻¹ * ∑ i, ‖Z i ω‖ :=
          mul_le_mul_of_nonneg_left (norm_sum_le _ _) (by positivity)
      _ ≤ (N : ℝ)⁻¹ * ∑ _i : Fin N, (wmax * R) :=
          mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => hZbd i ω) (by positivity)
      _ = wmax * R := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
            ← mul_assoc, inv_mul_cancel₀ (ne_of_gt hNpos), one_mul]
  -- pointwise MSE bound: ‖ĉ − c‖² ≤ wmin⁻² ‖N⁻¹ Σ Z‖²
  have hbound : ∀ ω, ‖(∑ i, w (Y i ω))⁻¹ • (∑ i, w (Y i ω) • Y i ω) - c‖ ^ 2
      ≤ wmin⁻¹ ^ 2 * ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖ ^ 2 := by
    intro ω
    have hnorm : ‖(∑ i, w (Y i ω))⁻¹ • (∑ i, w (Y i ω) • Y i ω) - c‖
        ≤ wmin⁻¹ * ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖ := by
      rw [hpoint ω, norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
        abs_of_pos (inv_pos.mpr (hDpos ω)), abs_of_pos (inv_pos.mpr hNpos), ← mul_assoc]
      apply mul_le_mul_of_nonneg_right _ (norm_nonneg _)
      rw [← mul_inv]
      have hge' : wmin * (N : ℝ) ≤ ∑ i, w (Y i ω) := by
        simpa [mul_comm] using hge ω
      simpa [one_div] using one_div_le_one_div_of_le (mul_pos hwmin hNpos) hge'
    calc ‖(∑ i, w (Y i ω))⁻¹ • (∑ i, w (Y i ω) • Y i ω) - c‖ ^ 2
        ≤ (wmin⁻¹ * ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖) ^ 2 :=
          sq_le_sq' (by
            have hl : 0 ≤ ‖(∑ i, w (Y i ω))⁻¹ • (∑ i, w (Y i ω) • Y i ω) - c‖ :=
              norm_nonneg _
            have hr : 0 ≤ wmin⁻¹ * ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖ := by positivity
            linarith) hnorm
      _ = wmin⁻¹ ^ 2 * ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖ ^ 2 := by rw [mul_pow]
  -- integrability of the sample-mean square (bounded, hence integrable)
  have hmsq_int : Integrable (fun ω => ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖ ^ 2) P := by
    refine (integrable_const ((wmax * R) ^ 2)).mono'
      ((hsummeas.const_smul _).norm.pow_const 2).aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    nlinarith [hmeanbd ω, norm_nonneg ((N : ℝ)⁻¹ • (∑ i, Z i ω))]
  have hg_int : Integrable
      (fun ω => wmin⁻¹ ^ 2 * ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖ ^ 2) P :=
    hmsq_int.const_mul _
  have hf_int : Integrable
      (fun ω => ‖(∑ i, w (Y i ω))⁻¹ • (∑ i, w (Y i ω) • Y i ω) - c‖ ^ 2) P := by
    refine hg_int.mono' (hcmeas.norm.pow_const 2).aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    exact hbound ω
  -- the sample-mean variance axiom applied to Z
  have haxiom : ∫ ω, ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖ ^ 2 ∂P ≤ σ ^ 2 / N := by
    refine Paper.sampleMean_meanSquare_le P hN Z (fun i j hij => (hindep i j hij).comp hg hg)
      (fun i => ?_) hmean hσ
    refine (integrable_const ((wmax * R) ^ 2)).mono'
      ((hZmeas i).norm.pow_const 2).aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    nlinarith [hZbd i ω, norm_nonneg (Z i ω)]
  -- assemble
  calc ∫ ω, ‖(∑ i, w (Y i ω))⁻¹ • (∑ i, w (Y i ω) • Y i ω) - c‖ ^ 2 ∂P
      ≤ ∫ ω, wmin⁻¹ ^ 2 * ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖ ^ 2 ∂P :=
        integral_mono hf_int hg_int hbound
    _ = wmin⁻¹ ^ 2 * ∫ ω, ‖(N : ℝ)⁻¹ • (∑ i, Z i ω)‖ ^ 2 ∂P := integral_const_mul _ _
    _ ≤ wmin⁻¹ ^ 2 * (σ ^ 2 / N) := by
        apply mul_le_mul_of_nonneg_left haxiom (by positivity)
    _ = σ ^ 2 / (wmin ^ 2 * N) := by
        rw [inv_pow]
        field_simp

end SelfNormalized
end DriftingIdentifiability
