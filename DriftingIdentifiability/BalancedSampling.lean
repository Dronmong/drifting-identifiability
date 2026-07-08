import DriftingIdentifiability.DenominatorTail
import DriftingIdentifiability.SinkhornBalanced

/-!
# EXTENSION TRACK: sampling theory for two-step balanced affinities

**Not part of the paper formalization.**  Spec and proof sketches:
`SinkhornImplementation/PLAN_BALANCED_STATS.md`.

At balancing depth `t ≥ 2` the affinity weights depend on the whole batch, so
the fixed-weight SNIS theorems no longer apply directly.  The structural fact
that saves the day: after row-cancellation, the entire batch-coupling of the
`t = 2` weights is mediated by the `M`-dimensional vector of *row masses*
`r_j = Σ_s k_j(Y_s)` — bounded sums of independent kernel values, exactly what
the existing weight-sum machinery concentrates.  The theorem chain:

* `weightSum_deviation_prob_le` — two-sided Chebyshev for the row masses.
* `selfNormalizedCentroid_relative_perturbation` — a deterministic bound: a
  relative weight error `η < 1` moves a self-normalized centroid by at most
  `η·D/(1-η)` (`D` = point diameter).  Axiom-free.
* `abs_inv_sqrt_sub_inv_sqrt_le`, `abs_sum_sub_sum_le_of_rel`,
  `twoStepWeight_rel_of_rowMass_rel` — relative errors propagate through the
  balancing level with explicit (loose) constants: row masses within `δ`
  give weights within `4δ`.  Axiom-free.
* `balancedTwoStepCentroid_deviation_prob_le` — the headline: the realized
  two-step balanced centroid deviates from the population target by more than
  `ε + 16δR` with probability at most the fixed-weight SNIS bound at the
  *reference* weight plus one row-mass tail per anchor.
* `twoStepBalancedMatrixCentroid_eq_weightCentroid` — the full finite
  matrix-form two-step centroid equals the weight-form centroid used by the
  sampling theorem, because the remaining per-row factor cancels.
* `balancedThreeStepCentroid_deviation_prob_le_of_mass_tails` — a fixed
  `t = 3` bridge conditional on explicit tails for raw and first-balanced row
  masses; the primitive proof of the first-balanced tails is intentionally
  left as the isolated remaining concentration problem.

No new axioms: the probabilistic inputs are the reviewed sample-mean axiom
(through the existing lemmas); everything else is finite algebra.
-/

open scoped BigOperators ENNReal
open MeasureTheory ProbabilityTheory

namespace DriftingIdentifiability
namespace SelfNormalized

open Paper

universe u

/-! ## B0: two-sided weight-sum deviation -/

/-- **Two-sided weight-sum deviation.**  The random weight sum deviates from
its mean sum by more than `t` with probability at most `N σw²/t²`. -/
theorem weightSum_deviation_prob_le
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
    P {ω | t < |(∑ l, w l (Y l ω)) - ∑ l, μw l|} ≤
      ENNReal.ofReal ((N * σw ^ 2) / t ^ 2) := by
  set W : Fin N → Ω → ℝ := fun l ω => w l (Y l ω) - μw l with hWdef
  have hNpos : (0 : ℝ) < N := by exact_mod_cast hN
  have hgw : ∀ l, Measurable (fun y : E => w l y - μw l) := fun l =>
    (hw l).sub measurable_const
  have hWmeas : ∀ l, Measurable (W l) := fun l => (hgw l).comp (hYmeas l)
  have hsum_meas : Measurable (fun ω => ∑ l, W l ω) :=
    Finset.measurable_sum _ (fun l _ => hWmeas l)
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
  have haxiomW : ∫ ω, ‖(N : ℝ)⁻¹ • ∑ l, W l ω‖ ^ 2 ∂P ≤ σw ^ 2 / N :=
    Paper.sampleMean_meanSquare_le P hN W hWindep hWint2 hWmean hWσ
  have hsub : {ω | t < |(∑ l, w l (Y l ω)) - ∑ l, μw l|} ⊆
      {ω | t < ‖∑ l, W l ω‖} := by
    intro ω hω
    simp only [Set.mem_setOf_eq] at hω ⊢
    have hsplit : (∑ l, W l ω) = (∑ l, w l (Y l ω)) - ∑ l, μw l := by
      simp only [hWdef]
      rw [Finset.sum_sub_distrib]
    rw [Real.norm_eq_abs, hsplit]
    exact hω
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

/-! ## B1: deterministic centroid perturbation -/

/-- **Relative weight perturbation of a self-normalized centroid.**  Weights
within relative error `η < 1` of positive reference weights move the centroid
by at most `η·D/(1-η)`, `D` the point diameter.  Axiom-free. -/
theorem selfNormalizedCentroid_relative_perturbation
    {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
    {ι : Type*} [Fintype ι] [Nonempty ι]
    (W Wb : ι → ℝ) (y : ι → E) {η D : ℝ}
    (hη0 : 0 ≤ η) (hη1 : η < 1) (hWb : ∀ s, 0 < Wb s)
    (hrel : ∀ s, |W s - Wb s| ≤ η * Wb s)
    (hdiam : ∀ s t, ‖y s - y t‖ ≤ D) :
    ‖(∑ s, W s)⁻¹ • (∑ s, W s • y s) -
        (∑ s, Wb s)⁻¹ • (∑ s, Wb s • y s)‖ ≤ η * D / (1 - η) := by
  set Sb : ℝ := ∑ s, Wb s with hSbdef
  set cb : E := Sb⁻¹ • (∑ s, Wb s • y s) with hcbdef
  have hSbpos : 0 < Sb := Finset.sum_pos (fun s _ => hWb s) Finset.univ_nonempty
  have hWlow : ∀ s, (1 - η) * Wb s ≤ W s := by
    intro s
    have h := abs_le.mp (hrel s)
    nlinarith [(hWb s).le]
  have hSlow : (1 - η) * Sb ≤ ∑ s, W s := by
    rw [hSbdef, Finset.mul_sum]
    exact Finset.sum_le_sum fun s _ => hWlow s
  have hSpos : 0 < ∑ s, W s :=
    lt_of_lt_of_le (mul_pos (by linarith) hSbpos) hSlow
  -- the reference centroid lies within diameter of every point
  have hhull : ∀ s, ‖y s - cb‖ ≤ D := by
    intro s
    have hexp : y s - cb = Sb⁻¹ • (∑ t, Wb t • (y s - y t)) := by
      have hsum : (∑ t, Wb t • (y s - y t)) = Sb • y s - ∑ t, Wb t • y t := by
        rw [hSbdef, Finset.sum_smul, ← Finset.sum_sub_distrib]
        apply Finset.sum_congr rfl
        intro t _
        rw [smul_sub]
      rw [hsum, smul_sub, hcbdef, smul_smul, inv_mul_cancel₀ hSbpos.ne', one_smul]
    rw [hexp, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hSbpos)]
    have hbd : ‖∑ t, Wb t • (y s - y t)‖ ≤ Sb * D := by
      refine le_trans (norm_sum_le _ _) ?_
      calc ∑ t, ‖Wb t • (y s - y t)‖
          = ∑ t, Wb t * ‖y s - y t‖ := by
            apply Finset.sum_congr rfl
            intro t _
            rw [norm_smul, Real.norm_eq_abs, abs_of_pos (hWb t)]
        _ ≤ ∑ t, Wb t * D :=
            Finset.sum_le_sum fun t _ =>
              mul_le_mul_of_nonneg_left (hdiam s t) (hWb t).le
        _ = Sb * D := by rw [← Finset.sum_mul, hSbdef]
    calc Sb⁻¹ * ‖∑ t, Wb t • (y s - y t)‖ ≤ Sb⁻¹ * (Sb * D) :=
          mul_le_mul_of_nonneg_left hbd (inv_pos.mpr hSbpos).le
      _ = D := by field_simp
  -- centered expansion of the perturbed centroid
  have hpoint : (∑ s, W s)⁻¹ • (∑ s, W s • y s) - cb
      = (∑ s, W s)⁻¹ • (∑ s, W s • (y s - cb)) := by
    have hsplit : (∑ s, W s • (y s - cb))
        = (∑ s, W s • y s) - (∑ s, W s) • cb := by
      rw [Finset.sum_smul, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro s _
      rw [smul_sub]
    rw [hsplit, smul_sub, smul_smul, inv_mul_cancel₀ hSpos.ne', one_smul]
  -- the reference weights center the differences
  have hzero : (∑ s, Wb s • (y s - cb)) = 0 := by
    have hsplit : (∑ s, Wb s • (y s - cb))
        = (∑ s, Wb s • y s) - Sb • cb := by
      rw [hSbdef, Finset.sum_smul, ← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro s _
      rw [smul_sub]
    rw [hsplit, hcbdef, smul_smul, mul_inv_cancel₀ hSbpos.ne', one_smul, sub_self]
  have hnum : ‖∑ s, W s • (y s - cb)‖ ≤ η * Sb * D := by
    have hsplit2 : (∑ s, W s • (y s - cb))
        = ∑ s, (W s - Wb s) • (y s - cb) := by
      rw [← sub_zero (∑ s, W s • (y s - cb))]
      nth_rewrite 1 [← hzero]
      rw [← Finset.sum_sub_distrib]
      apply Finset.sum_congr rfl
      intro s _
      rw [← sub_smul]
    rw [hsplit2]
    refine le_trans (norm_sum_le _ _) ?_
    calc ∑ s, ‖(W s - Wb s) • (y s - cb)‖
        = ∑ s, |W s - Wb s| * ‖y s - cb‖ := by
          apply Finset.sum_congr rfl
          intro s _
          rw [norm_smul, Real.norm_eq_abs]
      _ ≤ ∑ s, (η * Wb s) * D :=
          Finset.sum_le_sum fun s _ =>
            mul_le_mul (hrel s) (hhull s) (norm_nonneg _)
              (mul_nonneg hη0 (hWb s).le)
      _ = η * Sb * D := by
          rw [← Finset.sum_mul, ← Finset.mul_sum, hSbdef]
  rw [hpoint, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hSpos)]
  have hinv : (∑ s, W s)⁻¹ ≤ ((1 - η) * Sb)⁻¹ := by
    simpa [one_div] using
      one_div_le_one_div_of_le (mul_pos (by linarith) hSbpos) hSlow
  calc (∑ s, W s)⁻¹ * ‖∑ s, W s • (y s - cb)‖
      ≤ ((1 - η) * Sb)⁻¹ * (η * Sb * D) := by
        refine mul_le_mul hinv hnum (norm_nonneg _) ?_
        exact inv_nonneg.mpr (mul_pos (by linarith) hSbpos).le
    _ = η * D / (1 - η) := by
        field_simp

/-! ## B2: relative-error propagation through a balancing level -/

/-- B2a: a relative error `η ≤ 1/2` on a positive quantity gives a relative
error at most `2η` on its inverse square root. -/
theorem abs_inv_sqrt_sub_inv_sqrt_le {a abar η : ℝ}
    (habar : 0 < abar) (hη0 : 0 ≤ η) (hη : η ≤ 1 / 2)
    (hrel : |a - abar| ≤ η * abar) :
    |(Real.sqrt a)⁻¹ - (Real.sqrt abar)⁻¹| ≤ 2 * η * (Real.sqrt abar)⁻¹ := by
  have halow : (1 - η) * abar ≤ a := by
    have h := abs_le.mp hrel
    nlinarith
  have hapos : 0 < a := lt_of_lt_of_le (by nlinarith) halow
  have hsa := Real.sqrt_pos.mpr hapos
  have hsabar := Real.sqrt_pos.mpr habar
  have hsumpos : 0 < Real.sqrt abar + Real.sqrt a := by positivity
  -- |√abar − √a| ≤ η √abar
  have hsqrtdiff : |Real.sqrt abar - Real.sqrt a| ≤ η * Real.sqrt abar := by
    have hfac : (Real.sqrt abar - Real.sqrt a) *
        (Real.sqrt abar + Real.sqrt a) = abar - a := by
      have h1 := Real.sq_sqrt habar.le
      have h2 := Real.sq_sqrt hapos.le
      nlinarith
    have heq : |Real.sqrt abar - Real.sqrt a| =
        |abar - a| / (Real.sqrt abar + Real.sqrt a) := by
      rw [eq_div_iff hsumpos.ne', ← abs_of_pos hsumpos, ← abs_mul, hfac]
    rw [heq]
    calc |abar - a| / (Real.sqrt abar + Real.sqrt a)
        ≤ |abar - a| / Real.sqrt abar := by
          exact div_le_div_of_nonneg_left (abs_nonneg _) hsabar
            (le_add_of_nonneg_right hsa.le)
      _ ≤ (η * abar) / Real.sqrt abar := by
          exact div_le_div_of_nonneg_right (by simpa [abs_sub_comm] using hrel)
            hsabar.le
      _ = η * Real.sqrt abar := by
          field_simp [hsabar.ne']
          rw [Real.sq_sqrt habar.le]
  -- |1/√a − 1/√abar| ≤ η/√a
  have hmain : |(Real.sqrt a)⁻¹ - (Real.sqrt abar)⁻¹| ≤ η / Real.sqrt a := by
    have heq : (Real.sqrt a)⁻¹ - (Real.sqrt abar)⁻¹ =
        (Real.sqrt abar - Real.sqrt a) / (Real.sqrt a * Real.sqrt abar) := by
      field_simp
    rw [heq, abs_div, abs_of_pos (mul_pos hsa hsabar)]
    calc |Real.sqrt abar - Real.sqrt a| / (Real.sqrt a * Real.sqrt abar)
        ≤ (η * Real.sqrt abar) / (Real.sqrt a * Real.sqrt abar) := by
          exact div_le_div_of_nonneg_right hsqrtdiff
            (mul_nonneg hsa.le hsabar.le)
      _ = η / Real.sqrt a := by
          field_simp [hsa.ne', hsabar.ne']
  -- √a ≥ √abar/√2, so η/√a ≤ √2 η/√abar ≤ 2η/√abar
  have hage : abar / 2 ≤ a := by nlinarith
  have hsage : Real.sqrt abar / Real.sqrt 2 ≤ Real.sqrt a := by
    rw [← Real.sqrt_div habar.le]
    exact Real.sqrt_le_sqrt hage
  have hs2 : Real.sqrt 2 ≤ 2 := by
    rw [show (2 : ℝ) = Real.sqrt 4 by
      rw [show (4 : ℝ) = 2 ^ 2 by norm_num, Real.sqrt_sq (by norm_num)]]
    exact Real.sqrt_le_sqrt (by norm_num)
  have hsdiv : 0 < Real.sqrt abar / Real.sqrt 2 := by positivity
  calc |(Real.sqrt a)⁻¹ - (Real.sqrt abar)⁻¹| ≤ η / Real.sqrt a := hmain
    _ ≤ η / (Real.sqrt abar / Real.sqrt 2) := by
        exact div_le_div_of_nonneg_left hη0 hsdiv hsage
    _ = η * Real.sqrt 2 / Real.sqrt abar := by
        field_simp [Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 2)]
    _ ≤ 2 * η * (Real.sqrt abar)⁻¹ := by
        rw [div_eq_mul_inv]
        refine mul_le_mul_of_nonneg_right ?_ (inv_nonneg.mpr hsabar.le)
        nlinarith

/-- B2b: positive-coefficient sums preserve relative error bounds. -/
theorem abs_sum_sub_sum_le_of_rel {ι : Type*} [Fintype ι]
    (k aval abar : ι → ℝ) {η : ℝ}
    (hk : ∀ j, 0 ≤ k j)
    (hrel : ∀ j, |aval j - abar j| ≤ η * abar j) :
    |(∑ j, k j * aval j) - ∑ j, k j * abar j| ≤ η * ∑ j, k j * abar j := by
  rw [← Finset.sum_sub_distrib]
  calc |∑ j, (k j * aval j - k j * abar j)|
      ≤ ∑ j, |k j * aval j - k j * abar j| := Finset.abs_sum_le_sum_abs _ _
    _ = ∑ j, k j * |aval j - abar j| := by
        apply Finset.sum_congr rfl
        intro j _
        rw [← mul_sub, abs_mul, abs_of_nonneg (hk j)]
    _ ≤ ∑ j, k j * (η * abar j) :=
        Finset.sum_le_sum fun j _ =>
          mul_le_mul_of_nonneg_left (hrel j) (hk j)
    _ = η * ∑ j, k j * abar j := by
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro j _
        ring

/-- Product version of relative-error propagation.  If each factor is close
to a nonnegative reference factor, then the product is close with the usual
`η + θ + ηθ` loss. -/
theorem abs_mul_sub_mul_le_of_rel {a b abar bbar η θ : ℝ}
    (hη0 : 0 ≤ η) (_hθ0 : 0 ≤ θ) (habar : 0 ≤ abar) (hbbar : 0 ≤ bbar)
    (ha : |a - abar| ≤ η * abar) (hb : |b - bbar| ≤ θ * bbar) :
    |a * b - abar * bbar| ≤ (η + θ + η * θ) * (abar * bbar) := by
  have hterm :
      a * b - abar * bbar =
        (a - abar) * (b - bbar) + (a - abar) * bbar + abar * (b - bbar) := by
    ring
  rw [hterm]
  calc
    |(a - abar) * (b - bbar) + (a - abar) * bbar + abar * (b - bbar)|
        ≤ |(a - abar) * (b - bbar)| +
            |(a - abar) * bbar| + |abar * (b - bbar)| := by
          calc
            |(a - abar) * (b - bbar) + (a - abar) * bbar +
                abar * (b - bbar)|
                ≤ |(a - abar) * (b - bbar) + (a - abar) * bbar| +
                    |abar * (b - bbar)| := abs_add_le _ _
            _ ≤ |(a - abar) * (b - bbar)| + |(a - abar) * bbar| +
                    |abar * (b - bbar)| := by
                  nlinarith [abs_add_le ((a - abar) * (b - bbar))
                    ((a - abar) * bbar)]
    _ = |a - abar| * |b - bbar| + |a - abar| * bbar +
          abar * |b - bbar| := by
          rw [abs_mul, abs_mul, abs_mul, abs_of_nonneg hbbar, abs_of_nonneg habar]
    _ ≤ (η * abar) * (θ * bbar) + (η * abar) * bbar +
          abar * (θ * bbar) := by
          gcongr
    _ = (η + θ + η * θ) * (abar * bbar) := by ring

end SelfNormalized

/-! ## B3: two-step balanced weights and the headline theorem -/

namespace Algorithm2

open Paper SelfNormalized PaperFiniteIdentifiability

universe u

section TwoStep

variable {M N Npos Nneg : ℕ}

/-- Level-one column-mass profile at row masses `r`:
`h(y, r) = Σ_j k_j(y)/√(r_j)`. -/
noncomputable def balancedLevelMass (anchors : Fin M → ℝ) (τ : ℝ)
    (r : Fin M → ℝ) (y : ℝ) : ℝ :=
  ∑ j, algorithm2Kernel τ (anchors j) y * (Real.sqrt (r j))⁻¹

/-- The `t = 2` per-sample weight (after row-cancellation):
`k_i(y) / (g(y)^{1/4} · √h(y, r))`. -/
noncomputable def twoStepWeight (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (r : Fin M → ℝ) (y : ℝ) : ℝ :=
  algorithm2Kernel τ (anchors i) y /
    (Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ y)) *
      Real.sqrt (balancedLevelMass anchors τ r y))

theorem balancedLevelMass_pos [Nonempty (Fin M)]
    (anchors : Fin M → ℝ) (τ : ℝ) {r : Fin M → ℝ}
    (hr : ∀ j, 0 < r j) (y : ℝ) :
    0 < balancedLevelMass anchors τ r y := by
  unfold balancedLevelMass
  exact Finset.sum_pos (fun j _ =>
    mul_pos (algorithm2Kernel_pos τ (anchors j) y)
      (inv_pos.mpr (Real.sqrt_pos.mpr (hr j)))) Finset.univ_nonempty

theorem twoStepWeight_pos [Nonempty (Fin M)]
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M) {r : Fin M → ℝ}
    (hr : ∀ j, 0 < r j) (y : ℝ) :
    0 < twoStepWeight anchors τ i r y := by
  unfold twoStepWeight
  refine div_pos (algorithm2Kernel_pos τ (anchors i) y) (mul_pos ?_ ?_)
  · exact Real.sqrt_pos.mpr (Real.sqrt_pos.mpr
      (algorithm2ColumnKernelMass_pos anchors τ y))
  · exact Real.sqrt_pos.mpr (balancedLevelMass_pos anchors τ hr y)

theorem twoStepWeight_measurable (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (r : Fin M → ℝ) :
    Measurable (twoStepWeight anchors τ i r) := by
  unfold twoStepWeight balancedLevelMass
  refine Measurable.div ?_ (Measurable.mul ?_ ?_)
  · exact (PaperFiniteIdentifiability.algorithm2Kernel_continuous_snd τ
      (anchors i)).measurable
  · refine Real.continuous_sqrt.measurable.comp
      (Real.continuous_sqrt.measurable.comp ?_)
    unfold Algorithm2.algorithm2ColumnKernelMass
    exact Finset.measurable_sum _ fun j _ =>
      (PaperFiniteIdentifiability.algorithm2Kernel_continuous_snd τ
        (anchors j)).measurable
  · refine Real.continuous_sqrt.measurable.comp ?_
    exact Finset.measurable_sum _ fun j _ =>
      ((PaperFiniteIdentifiability.algorithm2Kernel_continuous_snd τ
        (anchors j)).measurable).mul_const _

/-- B2c: row masses within relative `δ ≤ 1/4` give two-step weights within
relative `4δ`.  Axiom-free. -/
theorem twoStepWeight_rel_of_rowMass_rel [Nonempty (Fin M)]
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    {r Mbar : Fin M → ℝ} {δ : ℝ}
    (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 4) (hMbar : ∀ j, 0 < Mbar j)
    (hrel : ∀ j, |r j - Mbar j| ≤ δ * Mbar j) (y : ℝ) :
    |twoStepWeight anchors τ i r y - twoStepWeight anchors τ i Mbar y| ≤
      4 * δ * twoStepWeight anchors τ i Mbar y := by
  have hrpos : ∀ j, 0 < r j := by
    intro j
    have h := abs_le.mp (hrel j)
    nlinarith [hMbar j]
  -- level masses within relative 2δ
  have hlevel : |balancedLevelMass anchors τ r y -
      balancedLevelMass anchors τ Mbar y| ≤
      (2 * δ) * balancedLevelMass anchors τ Mbar y := by
    unfold balancedLevelMass
    refine abs_sum_sub_sum_le_of_rel _ _ _
      (fun j => (algorithm2Kernel_pos τ (anchors j) y).le) ?_
    intro j
    exact abs_inv_sqrt_sub_inv_sqrt_le (hMbar j) hδ0 (by linarith) (hrel j)
  have hlevelpos := balancedLevelMass_pos anchors τ hMbar y
  -- inverse square roots of the level masses within relative 4δ
  have hinvsqrt : |(Real.sqrt (balancedLevelMass anchors τ r y))⁻¹ -
      (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹| ≤
      (4 * δ) * (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹ := by
    have h := abs_inv_sqrt_sub_inv_sqrt_le hlevelpos
      (by linarith : (0 : ℝ) ≤ 2 * δ) (by linarith) hlevel
    calc |(Real.sqrt (balancedLevelMass anchors τ r y))⁻¹ -
        (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹|
        ≤ 2 * (2 * δ) *
          (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹ := h
      _ = (4 * δ) *
          (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹ := by ring
  -- factor out the common positive prefactor
  have hq : 0 < Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ y)) :=
    Real.sqrt_pos.mpr (Real.sqrt_pos.mpr
      (algorithm2ColumnKernelMass_pos anchors τ y))
  have hW : ∀ rr : Fin M → ℝ, twoStepWeight anchors τ i rr y =
      (algorithm2Kernel τ (anchors i) y *
        (Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ y)))⁻¹) *
        (Real.sqrt (balancedLevelMass anchors τ rr y))⁻¹ := by
    intro rr
    unfold twoStepWeight
    rw [div_eq_mul_inv, mul_inv]
    ring
  have hpre : 0 ≤ algorithm2Kernel τ (anchors i) y *
      (Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ y)))⁻¹ :=
    mul_nonneg (algorithm2Kernel_nonneg τ (anchors i) y) (inv_nonneg.mpr hq.le)
  rw [hW r, hW Mbar, ← mul_sub, abs_mul, abs_of_nonneg hpre]
  calc (algorithm2Kernel τ (anchors i) y *
        (Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ y)))⁻¹) *
        |(Real.sqrt (balancedLevelMass anchors τ r y))⁻¹ -
          (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹|
      ≤ (algorithm2Kernel τ (anchors i) y *
          (Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ y)))⁻¹) *
        ((4 * δ) * (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹) :=
        mul_le_mul_of_nonneg_left hinvsqrt hpre
    _ = 4 * δ * ((algorithm2Kernel τ (anchors i) y *
          (Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ y)))⁻¹) *
          (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹) := by ring

/-! ## B4: full finite-matrix reconciliation -/

/-- Row mass of a finite anchor-by-sample matrix. -/
noncomputable def balancedMatrixRowMass (A : Fin M → Fin N → ℝ) (i : Fin M) : ℝ :=
  ∑ s, A i s

/-- Column mass of a finite anchor-by-sample matrix. -/
noncomputable def balancedMatrixColumnMass (A : Fin M → Fin N → ℝ) (s : Fin N) : ℝ :=
  ∑ i, A i s

/-- One geometric-mean row/column balancing step on a finite matrix. -/
noncomputable def balancedMatrixStep (A : Fin M → Fin N → ℝ) (i : Fin M) (s : Fin N) : ℝ :=
  A i s / (Real.sqrt (balancedMatrixRowMass A i) *
    Real.sqrt (balancedMatrixColumnMass A s))

/-- The raw Algorithm-2 kernel matrix for fixed anchors and one sample branch. -/
noncomputable def balancedKernelMatrix (anchors : Fin M → ℝ) (τ : ℝ)
    (Y : Fin N → ℝ) (i : Fin M) (s : Fin N) : ℝ :=
  algorithm2Kernel τ (anchors i) (Y s)

/-- Two full finite balancing steps, still in matrix form. -/
noncomputable def twoStepBalancedMatrixAffinity (anchors : Fin M → ℝ) (τ : ℝ)
    (Y : Fin N → ℝ) : Fin M → Fin N → ℝ :=
  balancedMatrixStep (balancedMatrixStep (balancedKernelMatrix anchors τ Y))

/-- Self-normalized centroid of one row of a finite affinity matrix. -/
noncomputable def balancedMatrixCentroid (A : Fin M → Fin N → ℝ)
    (Y : Fin N → ℝ) (i : Fin M) : ℝ :=
  (∑ s, A i s)⁻¹ • ∑ s, A i s • Y s

/-- The literal two-step matrix-form centroid. -/
noncomputable def twoStepBalancedMatrixCentroid (anchors : Fin M → ℝ) (τ : ℝ)
    (i : Fin M) (Y : Fin N → ℝ) : ℝ :=
  balancedMatrixCentroid (twoStepBalancedMatrixAffinity anchors τ Y) Y i

/-- The deterministic weight-form centroid used by the two-step sampling
theorem, with row masses supplied by the same finite batch. -/
noncomputable def balancedTwoStepWeightCentroid (anchors : Fin M → ℝ) (τ : ℝ)
    (i : Fin M) (Y : Fin N → ℝ) : ℝ :=
  (∑ s, twoStepWeight anchors τ i
      (fun j => balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) (Y s))⁻¹ •
    ∑ s, twoStepWeight anchors τ i
      (fun j => balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) (Y s) • Y s

theorem balancedMatrixRowMass_pos_of_pos [Nonempty (Fin N)]
    {A : Fin M → Fin N → ℝ} (hA : ∀ i s, 0 < A i s) (i : Fin M) :
    0 < balancedMatrixRowMass A i := by
  unfold balancedMatrixRowMass
  exact Finset.sum_pos (fun s _ => hA i s) Finset.univ_nonempty

theorem balancedMatrixColumnMass_pos_of_pos [Nonempty (Fin M)]
    {A : Fin M → Fin N → ℝ} (hA : ∀ i s, 0 < A i s) (s : Fin N) :
    0 < balancedMatrixColumnMass A s := by
  unfold balancedMatrixColumnMass
  exact Finset.sum_pos (fun i _ => hA i s) Finset.univ_nonempty

theorem balancedMatrixStep_pos [Nonempty (Fin M)] [Nonempty (Fin N)]
    {A : Fin M → Fin N → ℝ} (hA : ∀ i s, 0 < A i s) (i : Fin M) (s : Fin N) :
    0 < balancedMatrixStep A i s := by
  unfold balancedMatrixStep
  refine div_pos (hA i s) (mul_pos ?_ ?_)
  · exact Real.sqrt_pos.mpr (balancedMatrixRowMass_pos_of_pos hA i)
  · exact Real.sqrt_pos.mpr (balancedMatrixColumnMass_pos_of_pos hA s)

theorem balancedKernelMatrix_pos (anchors : Fin M → ℝ) (τ : ℝ)
    (Y : Fin N → ℝ) (i : Fin M) (s : Fin N) :
    0 < balancedKernelMatrix anchors τ Y i s := by
  unfold balancedKernelMatrix
  exact algorithm2Kernel_pos τ (anchors i) (Y s)

theorem balancedKernelMatrix_columnMass (anchors : Fin M → ℝ) (τ : ℝ)
    (Y : Fin N → ℝ) (s : Fin N) :
    balancedMatrixColumnMass (balancedKernelMatrix anchors τ Y) s =
      algorithm2ColumnKernelMass anchors τ (Y s) := by
  rfl

theorem balancedMatrixStep_kernel_eq (anchors : Fin M → ℝ) (τ : ℝ)
    (Y : Fin N → ℝ) (i : Fin M) (s : Fin N) :
    balancedMatrixStep (balancedKernelMatrix anchors τ Y) i s =
      algorithm2Kernel τ (anchors i) (Y s) /
        (Real.sqrt (balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) i) *
          Real.sqrt (algorithm2ColumnKernelMass anchors τ (Y s))) := by
  simp [balancedMatrixStep, balancedKernelMatrix, balancedMatrixColumnMass,
    algorithm2ColumnKernelMass]

/-- First balancing column mass in the form consumed by `twoStepWeight`. -/
theorem balancedMatrixColumnMass_firstStep_kernel [Nonempty (Fin M)] [Nonempty (Fin N)]
    (anchors : Fin M → ℝ) (τ : ℝ) (Y : Fin N → ℝ) (s : Fin N) :
    balancedMatrixColumnMass
        (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) s =
      balancedLevelMass anchors τ
          (fun j => balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) (Y s) *
        (Real.sqrt (algorithm2ColumnKernelMass anchors τ (Y s)))⁻¹ := by
  calc
    balancedMatrixColumnMass
        (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) s
        = ∑ j, algorithm2Kernel τ (anchors j) (Y s) /
            (Real.sqrt (balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) *
              Real.sqrt (algorithm2ColumnKernelMass anchors τ (Y s))) := by
          simp [balancedMatrixColumnMass, balancedMatrixStep, balancedKernelMatrix,
            algorithm2ColumnKernelMass]
    _ = ∑ j, (algorithm2Kernel τ (anchors j) (Y s) *
          (Real.sqrt (balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j))⁻¹) *
          (Real.sqrt (algorithm2ColumnKernelMass anchors τ (Y s)))⁻¹ := by
          apply Finset.sum_congr rfl
          intro j _
          rw [div_eq_mul_inv, mul_inv]
          ring
    _ = (∑ j, algorithm2Kernel τ (anchors j) (Y s) *
          (Real.sqrt (balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j))⁻¹) *
          (Real.sqrt (algorithm2ColumnKernelMass anchors τ (Y s)))⁻¹ := by
          rw [Finset.sum_mul]
    _ = balancedLevelMass anchors τ
          (fun j => balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) (Y s) *
        (Real.sqrt (algorithm2ColumnKernelMass anchors τ (Y s)))⁻¹ := by
          rfl

/-- Square-root algebra behind the `t = 2` matrix-to-weight cancellation. -/
theorem sqrt_mul_sqrt_mul_inv_sqrt_eq {a b : ℝ} (ha : 0 < a) (hb : 0 ≤ b) :
    Real.sqrt a * Real.sqrt (b * (Real.sqrt a)⁻¹) =
      Real.sqrt (Real.sqrt a) * Real.sqrt b := by
  have hsa : 0 < Real.sqrt a := Real.sqrt_pos.mpr ha
  have hbinv : 0 ≤ b * (Real.sqrt a)⁻¹ := mul_nonneg hb (inv_nonneg.mpr hsa.le)
  refine (sq_eq_sq₀ ?_ ?_).mp ?_
  · exact mul_nonneg (Real.sqrt_nonneg a) (Real.sqrt_nonneg _)
  · exact mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg b)
  · rw [mul_pow, mul_pow, Real.sq_sqrt ha.le, Real.sq_sqrt hbinv,
      Real.sq_sqrt hsa.le, Real.sq_sqrt hb]
    field_simp [hsa.ne']
    rw [Real.sq_sqrt ha.le]
    ring

/-- The full two-step matrix affinity is a per-row common scale times the
existing `twoStepWeight`.  The common scale cancels in row centroids. -/
theorem twoStepBalancedMatrixAffinity_eq_commonScale_mul_weight
    [Nonempty (Fin M)] [Nonempty (Fin N)]
    (anchors : Fin M → ℝ) (τ : ℝ) (Y : Fin N → ℝ) (i : Fin M) (s : Fin N) :
    twoStepBalancedMatrixAffinity anchors τ Y i s =
      ((Real.sqrt (balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) i) *
          Real.sqrt (balancedMatrixRowMass
            (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) i))⁻¹) *
        twoStepWeight anchors τ i
          (fun j => balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) (Y s) := by
  let K : Fin M → Fin N → ℝ := balancedKernelMatrix anchors τ Y
  let A₁ : Fin M → Fin N → ℝ := balancedMatrixStep K
  have hKpos : ∀ j t, 0 < K j t := by
    intro j t
    exact balancedKernelMatrix_pos anchors τ Y j t
  have hRpos : ∀ j, 0 < balancedMatrixRowMass K j := fun j =>
    balancedMatrixRowMass_pos_of_pos hKpos j
  have hA₁pos : ∀ j t, 0 < A₁ j t := by
    intro j t
    exact balancedMatrixStep_pos hKpos j t
  have hA₁rowpos : ∀ j, 0 < balancedMatrixRowMass A₁ j := fun j =>
    balancedMatrixRowMass_pos_of_pos hA₁pos j
  have hGpos : 0 < algorithm2ColumnKernelMass anchors τ (Y s) :=
    algorithm2ColumnKernelMass_pos anchors τ (Y s)
  have hLpos : 0 < balancedLevelMass anchors τ
      (fun j => balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) (Y s) :=
    balancedLevelMass_pos anchors τ (by
      intro j
      exact balancedMatrixRowMass_pos_of_pos
        (balancedKernelMatrix_pos anchors τ Y) j) (Y s)
  have hcol :
      balancedMatrixColumnMass
          (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) s =
        balancedLevelMass anchors τ
            (fun j => balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) (Y s) *
          (Real.sqrt (algorithm2ColumnKernelMass anchors τ (Y s)))⁻¹ :=
    balancedMatrixColumnMass_firstStep_kernel anchors τ Y s
  have hsqrtcol :
      Real.sqrt (algorithm2ColumnKernelMass anchors τ (Y s)) *
        Real.sqrt (balancedMatrixColumnMass
          (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) s) =
      Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ (Y s))) *
        Real.sqrt (balancedLevelMass anchors τ
          (fun j => balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) (Y s)) := by
    rw [hcol]
    exact sqrt_mul_sqrt_mul_inv_sqrt_eq hGpos hLpos.le
  have hsqrtcol_ne :
      Real.sqrt (algorithm2ColumnKernelMass anchors τ (Y s)) *
        Real.sqrt (balancedMatrixColumnMass
          (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) s) ≠ 0 := by
    rw [hsqrtcol]
    exact mul_ne_zero
      (Real.sqrt_pos.mpr (Real.sqrt_pos.mpr hGpos)).ne'
      (Real.sqrt_pos.mpr hLpos).ne'
  have hrow_ne :
      Real.sqrt (balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) i) *
        Real.sqrt (balancedMatrixRowMass
          (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) i) ≠ 0 := by
    exact mul_ne_zero
      (Real.sqrt_pos.mpr (balancedMatrixRowMass_pos_of_pos
        (balancedKernelMatrix_pos anchors τ Y) i)).ne'
      (Real.sqrt_pos.mpr (by
        exact balancedMatrixRowMass_pos_of_pos (by
          intro j t
          exact balancedMatrixStep_pos
            (balancedKernelMatrix_pos anchors τ Y) j t) i)).ne'
  unfold twoStepBalancedMatrixAffinity
  change (balancedMatrixStep (balancedKernelMatrix anchors τ Y) i s) /
      (Real.sqrt (balancedMatrixRowMass
          (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) i) *
        Real.sqrt (balancedMatrixColumnMass
          (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) s)) =
    ((Real.sqrt (balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) i) *
        Real.sqrt (balancedMatrixRowMass
          (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) i))⁻¹) *
      twoStepWeight anchors τ i
        (fun j => balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) (Y s)
  rw [balancedMatrixStep_kernel_eq anchors τ Y i s]
  unfold twoStepWeight
  rw [← hsqrtcol]
  rw [div_eq_mul_inv, div_eq_mul_inv, div_eq_mul_inv, mul_inv, mul_inv]
  field_simp [hrow_ne, hsqrtcol_ne]

/-- **B4 matrix reconciliation.**  The literal finite matrix obtained by two
geometric-mean balancing rounds yields exactly the same row centroid as the
weight-form `t = 2` estimator used in the sampling theorem.  This is the
formal bridge between the full matrix implementation and the B3 SNIS-style
analysis. -/
theorem twoStepBalancedMatrixCentroid_eq_weightCentroid
    [Nonempty (Fin M)] [Nonempty (Fin N)]
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M) (Y : Fin N → ℝ) :
    twoStepBalancedMatrixCentroid anchors τ i Y =
      balancedTwoStepWeightCentroid anchors τ i Y := by
  let lam : ℝ :=
    (Real.sqrt (balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) i) *
      Real.sqrt (balancedMatrixRowMass
        (balancedMatrixStep (balancedKernelMatrix anchors τ Y)) i))⁻¹
  let w : Fin N → ℝ := fun s =>
    twoStepWeight anchors τ i
      (fun j => balancedMatrixRowMass (balancedKernelMatrix anchors τ Y) j) (Y s)
  have hlam_ne : lam ≠ 0 := by
    have hKpos : ∀ j s, 0 < balancedKernelMatrix anchors τ Y j s :=
      balancedKernelMatrix_pos anchors τ Y
    have hA₁pos : ∀ j s,
        0 < balancedMatrixStep (balancedKernelMatrix anchors τ Y) j s := by
      intro j s
      exact balancedMatrixStep_pos hKpos j s
    unfold lam
    exact inv_ne_zero (mul_ne_zero
      (Real.sqrt_pos.mpr (balancedMatrixRowMass_pos_of_pos hKpos i)).ne'
      (Real.sqrt_pos.mpr (balancedMatrixRowMass_pos_of_pos hA₁pos i)).ne')
  have hpoint : ∀ s, twoStepBalancedMatrixAffinity anchors τ Y i s = lam * w s := by
    intro s
    unfold lam w
    exact twoStepBalancedMatrixAffinity_eq_commonScale_mul_weight anchors τ Y i s
  have hsum :
      (∑ s, twoStepBalancedMatrixAffinity anchors τ Y i s) = ∑ s, lam * w s := by
    apply Finset.sum_congr rfl
    intro s _
    exact hpoint s
  have hsumY :
      (∑ s, twoStepBalancedMatrixAffinity anchors τ Y i s • Y s) =
        ∑ s, (lam * w s) • Y s := by
    apply Finset.sum_congr rfl
    intro s _
    rw [hpoint s]
  unfold twoStepBalancedMatrixCentroid balancedMatrixCentroid balancedTwoStepWeightCentroid
  rw [hsum, hsumY]
  simpa [w] using selfNormalizedCentroid_eq_of_common_scale lam w Y hlam_ne

/-! ## B5: fixed `t = 3` finite unrolling core -/

/-- Level-two numerator profile:
`Σ_j k_j(y)/(sqrt(r_j) sqrt(q_j))`, where `r` is the raw row-mass profile and
`q` is the first-balanced row-mass profile. -/
noncomputable def twoStepLevelMass (anchors : Fin M → ℝ) (τ : ℝ)
    (r q : Fin M → ℝ) (y : ℝ) : ℝ :=
  ∑ j, algorithm2Kernel τ (anchors j) y *
    ((Real.sqrt (r j))⁻¹ * (Real.sqrt (q j))⁻¹)

/-- Column mass profile of the second balanced matrix, written in the
row-cancelled coordinates used by the `t = 3` centroid. -/
noncomputable def secondBalancedColumnProfile (anchors : Fin M → ℝ) (τ : ℝ)
    (r q : Fin M → ℝ) (y : ℝ) : ℝ :=
  twoStepLevelMass anchors τ r q y /
    (Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ y)) *
      Real.sqrt (balancedLevelMass anchors τ r y))

/-- The `t = 3` per-sample weight after cancelling all per-anchor row factors
in the row centroid. -/
noncomputable def threeStepWeight (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (r q : Fin M → ℝ) (y : ℝ) : ℝ :=
  twoStepWeight anchors τ i r y /
    Real.sqrt (secondBalancedColumnProfile anchors τ r q y)

theorem twoStepLevelMass_pos [Nonempty (Fin M)]
    (anchors : Fin M → ℝ) (τ : ℝ) {r q : Fin M → ℝ}
    (hr : ∀ j, 0 < r j) (hq : ∀ j, 0 < q j) (y : ℝ) :
    0 < twoStepLevelMass anchors τ r q y := by
  unfold twoStepLevelMass
  exact Finset.sum_pos (fun j _ =>
    mul_pos (algorithm2Kernel_pos τ (anchors j) y)
      (mul_pos (inv_pos.mpr (Real.sqrt_pos.mpr (hr j)))
        (inv_pos.mpr (Real.sqrt_pos.mpr (hq j))))) Finset.univ_nonempty

theorem secondBalancedColumnProfile_pos [Nonempty (Fin M)]
    (anchors : Fin M → ℝ) (τ : ℝ) {r q : Fin M → ℝ}
    (hr : ∀ j, 0 < r j) (hq : ∀ j, 0 < q j) (y : ℝ) :
    0 < secondBalancedColumnProfile anchors τ r q y := by
  unfold secondBalancedColumnProfile
  refine div_pos (twoStepLevelMass_pos anchors τ hr hq y) (mul_pos ?_ ?_)
  · exact Real.sqrt_pos.mpr (Real.sqrt_pos.mpr
      (algorithm2ColumnKernelMass_pos anchors τ y))
  · exact Real.sqrt_pos.mpr (balancedLevelMass_pos anchors τ hr y)

theorem threeStepWeight_pos [Nonempty (Fin M)]
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M) {r q : Fin M → ℝ}
    (hr : ∀ j, 0 < r j) (hq : ∀ j, 0 < q j) (y : ℝ) :
    0 < threeStepWeight anchors τ i r q y := by
  unfold threeStepWeight
  exact div_pos (twoStepWeight_pos anchors τ i hr y)
    (Real.sqrt_pos.mpr (secondBalancedColumnProfile_pos anchors τ hr hq y))

theorem twoStepLevelMass_measurable (anchors : Fin M → ℝ) (τ : ℝ)
    (r q : Fin M → ℝ) :
    Measurable (twoStepLevelMass anchors τ r q) := by
  unfold twoStepLevelMass
  exact Finset.measurable_sum _ fun j _ =>
    ((PaperFiniteIdentifiability.algorithm2Kernel_continuous_snd τ
      (anchors j)).measurable).mul_const _

theorem secondBalancedColumnProfile_measurable (anchors : Fin M → ℝ) (τ : ℝ)
    (r q : Fin M → ℝ) :
    Measurable (secondBalancedColumnProfile anchors τ r q) := by
  unfold secondBalancedColumnProfile
  refine (twoStepLevelMass_measurable anchors τ r q).div (Measurable.mul ?_ ?_)
  · refine Real.continuous_sqrt.measurable.comp
      (Real.continuous_sqrt.measurable.comp ?_)
    unfold Algorithm2.algorithm2ColumnKernelMass
    exact Finset.measurable_sum _ fun j _ =>
      (PaperFiniteIdentifiability.algorithm2Kernel_continuous_snd τ
        (anchors j)).measurable
  · refine Real.continuous_sqrt.measurable.comp ?_
    unfold balancedLevelMass
    exact Finset.measurable_sum _ fun j _ =>
      ((PaperFiniteIdentifiability.algorithm2Kernel_continuous_snd τ
        (anchors j)).measurable).mul_const _

theorem threeStepWeight_measurable (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (r q : Fin M → ℝ) :
    Measurable (threeStepWeight anchors τ i r q) := by
  unfold threeStepWeight
  exact (twoStepWeight_measurable anchors τ i r).div
    (Real.continuous_sqrt.measurable.comp
      (secondBalancedColumnProfile_measurable anchors τ r q))

/-- Relative error for the level-two numerator profile. -/
theorem twoStepLevelMass_rel_of_rowMass_rel [Nonempty (Fin M)]
    (anchors : Fin M → ℝ) (τ : ℝ)
    {r q Mbar Qbar : Fin M → ℝ} {δ : ℝ}
    (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 4)
    (hMbar : ∀ j, 0 < Mbar j) (hQbar : ∀ j, 0 < Qbar j)
    (hrrel : ∀ j, |r j - Mbar j| ≤ δ * Mbar j)
    (hqrel : ∀ j, |q j - Qbar j| ≤ δ * Qbar j) (y : ℝ) :
    |twoStepLevelMass anchors τ r q y -
        twoStepLevelMass anchors τ Mbar Qbar y| ≤
      5 * δ * twoStepLevelMass anchors τ Mbar Qbar y := by
  unfold twoStepLevelMass
  refine abs_sum_sub_sum_le_of_rel _ _ _
    (fun j => (algorithm2Kernel_pos τ (anchors j) y).le) ?_
  intro j
  have hrinv := abs_inv_sqrt_sub_inv_sqrt_le (hMbar j) hδ0 (by linarith) (hrrel j)
  have hqinv := abs_inv_sqrt_sub_inv_sqrt_le (hQbar j) hδ0 (by linarith) (hqrel j)
  have hprod := abs_mul_sub_mul_le_of_rel
    (by nlinarith : 0 ≤ 2 * δ) (by nlinarith : 0 ≤ 2 * δ)
    (inv_nonneg.mpr (Real.sqrt_pos.mpr (hMbar j)).le)
    (inv_nonneg.mpr (Real.sqrt_pos.mpr (hQbar j)).le)
    hrinv hqinv
  calc
    |(Real.sqrt (r j))⁻¹ * (Real.sqrt (q j))⁻¹ -
        (Real.sqrt (Mbar j))⁻¹ * (Real.sqrt (Qbar j))⁻¹|
        ≤ (2 * δ + 2 * δ + (2 * δ) * (2 * δ)) *
            ((Real.sqrt (Mbar j))⁻¹ * (Real.sqrt (Qbar j))⁻¹) := hprod
    _ ≤ 5 * δ * ((Real.sqrt (Mbar j))⁻¹ * (Real.sqrt (Qbar j))⁻¹) := by
        refine mul_le_mul_of_nonneg_right ?_ ?_
        · nlinarith [hδ0, hδ]
        · exact mul_nonneg
            (inv_nonneg.mpr (Real.sqrt_pos.mpr (hMbar j)).le)
            (inv_nonneg.mpr (Real.sqrt_pos.mpr (hQbar j)).le)

/-- Relative error for the second balanced column profile. -/
theorem secondBalancedColumnProfile_rel_of_rowMass_rel [Nonempty (Fin M)]
    (anchors : Fin M → ℝ) (τ : ℝ)
    {r q Mbar Qbar : Fin M → ℝ} {δ : ℝ}
    (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 8)
    (hMbar : ∀ j, 0 < Mbar j) (hQbar : ∀ j, 0 < Qbar j)
    (hrrel : ∀ j, |r j - Mbar j| ≤ δ * Mbar j)
    (hqrel : ∀ j, |q j - Qbar j| ≤ δ * Qbar j) (y : ℝ) :
    |secondBalancedColumnProfile anchors τ r q y -
        secondBalancedColumnProfile anchors τ Mbar Qbar y| ≤
      12 * δ * secondBalancedColumnProfile anchors τ Mbar Qbar y := by
  have hlevel : |balancedLevelMass anchors τ r y -
      balancedLevelMass anchors τ Mbar y| ≤
      (2 * δ) * balancedLevelMass anchors τ Mbar y := by
    unfold balancedLevelMass
    refine abs_sum_sub_sum_le_of_rel _ _ _
      (fun j => (algorithm2Kernel_pos τ (anchors j) y).le) ?_
    intro j
    exact abs_inv_sqrt_sub_inv_sqrt_le (hMbar j) hδ0 (by linarith) (hrrel j)
  have hlevelpos := balancedLevelMass_pos anchors τ hMbar y
  have hlevelInv : |(Real.sqrt (balancedLevelMass anchors τ r y))⁻¹ -
      (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹| ≤
      (4 * δ) * (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹ := by
    have h := abs_inv_sqrt_sub_inv_sqrt_le hlevelpos
      (by nlinarith : (0 : ℝ) ≤ 2 * δ) (by nlinarith) hlevel
    calc |(Real.sqrt (balancedLevelMass anchors τ r y))⁻¹ -
        (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹|
        ≤ 2 * (2 * δ) *
          (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹ := h
      _ = (4 * δ) *
          (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹ := by ring
  have htwo := twoStepLevelMass_rel_of_rowMass_rel anchors τ hδ0
    (by linarith) hMbar hQbar hrrel hqrel y
  have htwoPos := twoStepLevelMass_pos anchors τ hMbar hQbar y
  have hprod := abs_mul_sub_mul_le_of_rel
    (by nlinarith : 0 ≤ 5 * δ) (by nlinarith : 0 ≤ 4 * δ)
    htwoPos.le (inv_nonneg.mpr (Real.sqrt_pos.mpr hlevelpos).le)
    htwo hlevelInv
  have hconst : 0 ≤ (Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ y)))⁻¹ :=
    inv_nonneg.mpr (Real.sqrt_pos.mpr (Real.sqrt_pos.mpr
      (algorithm2ColumnKernelMass_pos anchors τ y))).le
  unfold secondBalancedColumnProfile
  rw [div_eq_mul_inv, div_eq_mul_inv, mul_inv, mul_inv]
  set C := (Real.sqrt (Real.sqrt (algorithm2ColumnKernelMass anchors τ y)))⁻¹ with hC
  set A := twoStepLevelMass anchors τ r q y with hA
  set B := (Real.sqrt (balancedLevelMass anchors τ r y))⁻¹ with hB
  set Abar := twoStepLevelMass anchors τ Mbar Qbar y with hAbar
  set Bbar := (Real.sqrt (balancedLevelMass anchors τ Mbar y))⁻¹ with hBbar
  change |A * (C * B) - Abar * (C * Bbar)| ≤ 12 * δ * (Abar * (C * Bbar))
  have hCnonneg : 0 ≤ C := by
    simp [C, hconst]
  have hrewrite : A * (C * B) - Abar * (C * Bbar) = C * (A * B - Abar * Bbar) := by
    ring
  rw [hrewrite, abs_mul, abs_of_nonneg hCnonneg]
  calc
    C * |A * B - Abar * Bbar|
        ≤ C *
            ((5 * δ + 4 * δ + (5 * δ) * (4 * δ)) *
              (Abar * Bbar)) := by
          exact mul_le_mul_of_nonneg_left (by simpa [A, B, Abar, Bbar] using hprod) hCnonneg
    _ ≤ C *
            ((12 * δ) *
              (Abar * Bbar)) := by
          refine mul_le_mul_of_nonneg_left ?_ hCnonneg
          refine mul_le_mul_of_nonneg_right ?_ ?_
          · nlinarith [hδ0, hδ]
          · exact mul_nonneg htwoPos.le
              (inv_nonneg.mpr (Real.sqrt_pos.mpr hlevelpos).le)
    _ = 12 * δ * (Abar * (C * Bbar)) := by ring

/-- If both the raw row masses and first-balanced row masses are within
relative `δ`, then the `t = 3` row-cancelled weights are within relative
`32δ`.  Constants are intentionally loose. -/
theorem threeStepWeight_rel_of_rowMass_rel [Nonempty (Fin M)]
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    {r q Mbar Qbar : Fin M → ℝ} {δ : ℝ}
    (hδ0 : 0 ≤ δ) (hδ : δ ≤ 1 / 24)
    (hMbar : ∀ j, 0 < Mbar j) (hQbar : ∀ j, 0 < Qbar j)
    (hrrel : ∀ j, |r j - Mbar j| ≤ δ * Mbar j)
    (hqrel : ∀ j, |q j - Qbar j| ≤ δ * Qbar j) (y : ℝ) :
    |threeStepWeight anchors τ i r q y -
        threeStepWeight anchors τ i Mbar Qbar y| ≤
      32 * δ * threeStepWeight anchors τ i Mbar Qbar y := by
  have hW2 := twoStepWeight_rel_of_rowMass_rel anchors τ i hδ0
    (by linarith) hMbar hrrel y
  have hprof := secondBalancedColumnProfile_rel_of_rowMass_rel anchors τ
    hδ0 (by linarith) hMbar hQbar hrrel hqrel y
  have hprofPos := secondBalancedColumnProfile_pos anchors τ hMbar hQbar y
  have hprofInv : |(Real.sqrt (secondBalancedColumnProfile anchors τ r q y))⁻¹ -
      (Real.sqrt (secondBalancedColumnProfile anchors τ Mbar Qbar y))⁻¹| ≤
      (24 * δ) *
        (Real.sqrt (secondBalancedColumnProfile anchors τ Mbar Qbar y))⁻¹ := by
    have h := abs_inv_sqrt_sub_inv_sqrt_le hprofPos
      (by nlinarith : (0 : ℝ) ≤ 12 * δ) (by nlinarith) hprof
    calc |(Real.sqrt (secondBalancedColumnProfile anchors τ r q y))⁻¹ -
        (Real.sqrt (secondBalancedColumnProfile anchors τ Mbar Qbar y))⁻¹|
        ≤ 2 * (12 * δ) *
          (Real.sqrt (secondBalancedColumnProfile anchors τ Mbar Qbar y))⁻¹ := h
      _ = (24 * δ) *
          (Real.sqrt (secondBalancedColumnProfile anchors τ Mbar Qbar y))⁻¹ := by ring
  have hW2Pos := twoStepWeight_pos anchors τ i hMbar y
  have hprod := abs_mul_sub_mul_le_of_rel
    (by nlinarith : 0 ≤ 4 * δ) (by nlinarith : 0 ≤ 24 * δ)
    hW2Pos.le (inv_nonneg.mpr (Real.sqrt_pos.mpr hprofPos).le)
    hW2 hprofInv
  unfold threeStepWeight
  rw [div_eq_mul_inv, div_eq_mul_inv]
  calc
    |twoStepWeight anchors τ i r y *
          (Real.sqrt (secondBalancedColumnProfile anchors τ r q y))⁻¹ -
        twoStepWeight anchors τ i Mbar y *
          (Real.sqrt (secondBalancedColumnProfile anchors τ Mbar Qbar y))⁻¹|
        ≤ (4 * δ + 24 * δ + (4 * δ) * (24 * δ)) *
          (twoStepWeight anchors τ i Mbar y *
            (Real.sqrt (secondBalancedColumnProfile anchors τ Mbar Qbar y))⁻¹) := hprod
    _ ≤ 32 * δ *
          (twoStepWeight anchors τ i Mbar y *
            (Real.sqrt (secondBalancedColumnProfile anchors τ Mbar Qbar y))⁻¹) := by
        refine mul_le_mul_of_nonneg_right ?_ ?_
        · nlinarith [hδ0, hδ]
        · exact mul_nonneg hW2Pos.le
            (inv_nonneg.mpr (Real.sqrt_pos.mpr hprofPos).le)

/-- The realized (batch) row masses. -/
noncomputable def realizedRowMass {Ω : Type*} (anchors : Fin M → ℝ) (τ : ℝ)
    (Y : Fin N → Ω → ℝ) (ω : Ω) : Fin M → ℝ :=
  fun j => ∑ s, algorithm2Kernel τ (anchors j) (Y s ω)

/-- First-balanced row masses of a realized batch. -/
noncomputable def realizedFirstBalancedRowMass {Ω : Type*} (anchors : Fin M → ℝ)
    (τ : ℝ) (Y : Fin N → Ω → ℝ) (ω : Ω) : Fin M → ℝ :=
  fun j => balancedMatrixRowMass
    (balancedMatrixStep (balancedKernelMatrix anchors τ (fun s => Y s ω))) j

/-- The realized `t = 3` balanced centroid in row-cancelled weight form. -/
noncomputable def balancedThreeStepCentroid {Ω : Type*} (anchors : Fin M → ℝ)
    (τ : ℝ) (i : Fin M) (Y : Fin N → Ω → ℝ) (ω : Ω) : ℝ :=
  (∑ l, threeStepWeight anchors τ i
      (realizedRowMass anchors τ Y ω) (realizedFirstBalancedRowMass anchors τ Y ω)
      (Y l ω))⁻¹ •
    ∑ l, threeStepWeight anchors τ i
      (realizedRowMass anchors τ Y ω) (realizedFirstBalancedRowMass anchors τ Y ω)
      (Y l ω) • Y l ω

/-- Fixed-reference `t = 3` balanced centroid. -/
noncomputable def balancedThreeStepReferenceCentroid {Ω : Type*}
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (Mbar Qbar : Fin M → ℝ) (Y : Fin N → Ω → ℝ) (ω : Ω) : ℝ :=
  (∑ l, threeStepWeight anchors τ i Mbar Qbar (Y l ω))⁻¹ •
    ∑ l, threeStepWeight anchors τ i Mbar Qbar (Y l ω) • Y l ω

/-- Normalized two-branch `t = 3` balanced drift. -/
noncomputable def balancedThreeStepNormalizedDrift {Ω : Type*}
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (Ypos : Fin Npos → Ω → ℝ) (Yneg : Fin Nneg → Ω → ℝ) (ω : Ω) : ℝ :=
  balancedThreeStepCentroid anchors τ i Ypos ω -
    balancedThreeStepCentroid anchors τ i Yneg ω

/-- **Two-branch normalized drift assembly at `t = 3`.**  This is the same
metric/event bookkeeping as the `t = 2` assembly theorem. -/
theorem balancedThreeStepNormalizedDrift_deviation_prob_le_of_centroids
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (Ypos : Fin Npos → Ω → ℝ) (Yneg : Fin Nneg → Ω → ℝ)
    (cPos cNeg : ℝ) {εPos εNeg : ℝ} {Bpos Bneg : ℝ≥0∞}
    (hpos : P {ω | εPos <
        ‖balancedThreeStepCentroid anchors τ i Ypos ω - cPos‖} ≤ Bpos)
    (hneg : P {ω | εNeg <
        ‖balancedThreeStepCentroid anchors τ i Yneg ω - cNeg‖} ≤ Bneg) :
    P {ω | εPos + εNeg <
        ‖balancedThreeStepNormalizedDrift anchors τ i Ypos Yneg ω -
          (cPos - cNeg)‖} ≤ Bpos + Bneg := by
  have hsub :
      {ω | εPos + εNeg <
          ‖balancedThreeStepNormalizedDrift anchors τ i Ypos Yneg ω -
            (cPos - cNeg)‖} ⊆
        {ω | εPos <
          ‖balancedThreeStepCentroid anchors τ i Ypos ω - cPos‖} ∪
        {ω | εNeg <
          ‖balancedThreeStepCentroid anchors τ i Yneg ω - cNeg‖} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
    by_cases hp : εPos <
        ‖balancedThreeStepCentroid anchors τ i Ypos ω - cPos‖
    · exact Or.inl hp
    · right
      by_contra hn
      have hp_le : ‖balancedThreeStepCentroid anchors τ i Ypos ω - cPos‖ ≤ εPos :=
        le_of_not_gt hp
      have hn_le : ‖balancedThreeStepCentroid anchors τ i Yneg ω - cNeg‖ ≤ εNeg :=
        le_of_not_gt hn
      have hrewrite :
          balancedThreeStepNormalizedDrift anchors τ i Ypos Yneg ω - (cPos - cNeg) =
            (balancedThreeStepCentroid anchors τ i Ypos ω - cPos) -
              (balancedThreeStepCentroid anchors τ i Yneg ω - cNeg) := by
        unfold balancedThreeStepNormalizedDrift
        ring
      have htri : ‖balancedThreeStepNormalizedDrift anchors τ i Ypos Yneg ω -
          (cPos - cNeg)‖ ≤ εPos + εNeg := by
        rw [hrewrite]
        calc ‖(balancedThreeStepCentroid anchors τ i Ypos ω - cPos) -
              (balancedThreeStepCentroid anchors τ i Yneg ω - cNeg)‖
            ≤ ‖balancedThreeStepCentroid anchors τ i Ypos ω - cPos‖ +
                ‖balancedThreeStepCentroid anchors τ i Yneg ω - cNeg‖ := norm_sub_le _ _
          _ ≤ εPos + εNeg := add_le_add hp_le hn_le
      exact (not_lt_of_ge htri) hω
  calc P {ω | εPos + εNeg <
        ‖balancedThreeStepNormalizedDrift anchors τ i Ypos Yneg ω -
          (cPos - cNeg)‖}
      ≤ P ({ω | εPos <
          ‖balancedThreeStepCentroid anchors τ i Ypos ω - cPos‖} ∪
        {ω | εNeg <
          ‖balancedThreeStepCentroid anchors τ i Yneg ω - cNeg‖}) := measure_mono hsub
    _ ≤ P {ω | εPos <
          ‖balancedThreeStepCentroid anchors τ i Ypos ω - cPos‖} +
        P {ω | εNeg <
          ‖balancedThreeStepCentroid anchors τ i Yneg ω - cNeg‖} := measure_union_le _ _
    _ ≤ Bpos + Bneg := add_le_add hpos hneg

/-- **Fixed-depth `t = 3` batch-dependence bridge.**  If the raw row masses
and first-balanced row masses are both close to fixed reference profiles, then
the realized `t = 3` centroid is close to the fixed-reference centroid.  The
first-balanced mass tails are hypotheses: proving them from primitive iid
assumptions is the remaining deeper concentration step. -/
theorem balancedThreeStepCentroid_deviation_prob_le_of_mass_tails
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [Nonempty (Fin M)]
    (hN : 0 < N)
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (Y : Fin N → Ω → ℝ) (c : ℝ) (Mbar Qbar : Fin M → ℝ)
    {R δ ε : ℝ} {Bref : ℝ≥0∞} (B0 B1 : Fin M → ℝ≥0∞)
    (hR : 0 ≤ R) (hδ0 : 0 < δ) (hδ : δ ≤ 1 / 64)
    (hMbar : ∀ j, 0 < Mbar j) (hQbar : ∀ j, 0 < Qbar j)
    (hYbd : ∀ l ω, ‖Y l ω - c‖ ≤ R)
    (href : P {ω | ε <
        ‖balancedThreeStepReferenceCentroid anchors τ i Mbar Qbar Y ω - c‖} ≤ Bref)
    (hrow0tail : ∀ j, P {ω |
        δ * Mbar j < |realizedRowMass anchors τ Y ω j - Mbar j|} ≤ B0 j)
    (hrow1tail : ∀ j, P {ω |
        δ * Qbar j < |realizedFirstBalancedRowMass anchors τ Y ω j - Qbar j|} ≤ B1 j) :
    P {ω | ε + 128 * δ * R <
        ‖balancedThreeStepCentroid anchors τ i Y ω - c‖} ≤
      (Bref + ∑ j, B0 j) + ∑ j, B1 j := by
  let Cref : Ω → ℝ := fun ω =>
    balancedThreeStepReferenceCentroid anchors τ i Mbar Qbar Y ω
  let Brow0 : Fin M → Set Ω := fun j =>
    {ω | δ * Mbar j < |realizedRowMass anchors τ Y ω j - Mbar j|}
  let Brow1 : Fin M → Set Ω := fun j =>
    {ω | δ * Qbar j < |realizedFirstBalancedRowMass anchors τ Y ω j - Qbar j|}
  haveI : Nonempty (Fin N) := ⟨⟨0, hN⟩⟩
  have hpert : ∀ ω, (∀ j, ω ∉ Brow0 j) → (∀ j, ω ∉ Brow1 j) →
      ‖balancedThreeStepCentroid anchors τ i Y ω - Cref ω‖ ≤ 128 * δ * R := by
    intro ω hgood0 hgood1
    have hδnonneg : 0 ≤ δ := hδ0.le
    have heta0 : 0 ≤ 32 * δ := by nlinarith
    have heta1 : 32 * δ < 1 := by nlinarith [hδ]
    have hrelω : ∀ l,
        |threeStepWeight anchors τ i
            (realizedRowMass anchors τ Y ω) (realizedFirstBalancedRowMass anchors τ Y ω)
            (Y l ω) -
          threeStepWeight anchors τ i Mbar Qbar (Y l ω)| ≤
        (32 * δ) * threeStepWeight anchors τ i Mbar Qbar (Y l ω) := by
      intro l
      refine threeStepWeight_rel_of_rowMass_rel anchors τ i hδnonneg
        (by linarith) hMbar hQbar ?_ ?_ (Y l ω)
      · intro j
        have hj := hgood0 j
        simp only [Brow0, Set.mem_setOf_eq, not_lt] at hj
        exact hj
      · intro j
        have hj := hgood1 j
        simp only [Brow1, Set.mem_setOf_eq, not_lt] at hj
        exact hj
    have hdiam : ∀ s t, ‖Y s ω - Y t ω‖ ≤ 2 * R := by
      intro s t
      have hrewrite : Y s ω - Y t ω = (Y s ω - c) - (Y t ω - c) := by ring
      rw [hrewrite]
      calc ‖(Y s ω - c) - (Y t ω - c)‖
          ≤ ‖Y s ω - c‖ + ‖Y t ω - c‖ := norm_sub_le _ _
        _ ≤ R + R := add_le_add (hYbd s ω) (hYbd t ω)
        _ = 2 * R := by ring
    have hcent := selfNormalizedCentroid_relative_perturbation
      (fun l => threeStepWeight anchors τ i
        (realizedRowMass anchors τ Y ω) (realizedFirstBalancedRowMass anchors τ Y ω)
        (Y l ω))
      (fun l => threeStepWeight anchors τ i Mbar Qbar (Y l ω))
      (fun l => Y l ω) heta0 heta1
      (fun l => threeStepWeight_pos anchors τ i hMbar hQbar (Y l ω))
      hrelω hdiam
    have hsmall :
        (32 * δ) * (2 * R) / (1 - 32 * δ) ≤ 128 * δ * R := by
      have hden : (1 / 2 : ℝ) ≤ 1 - 32 * δ := by nlinarith [hδ]
      have hnum : 0 ≤ 64 * δ * R := by nlinarith [hδnonneg, hR]
      calc (32 * δ) * (2 * R) / (1 - 32 * δ)
          = (64 * δ * R) / (1 - 32 * δ) := by ring
        _ ≤ (64 * δ * R) / (1 / 2 : ℝ) :=
            div_le_div_of_nonneg_left hnum (by norm_num) hden
        _ = 128 * δ * R := by ring
    simpa [balancedThreeStepCentroid, balancedThreeStepReferenceCentroid, Cref]
      using le_trans hcent hsmall
  have hsub :
      {ω | ε + 128 * δ * R <
          ‖balancedThreeStepCentroid anchors τ i Y ω - c‖} ⊆
        ({ω | ε < ‖Cref ω - c‖} ∪ ⋃ j, Brow0 j) ∪ ⋃ j, Brow1 j := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union, Set.mem_iUnion] at hω ⊢
    by_cases hgood0 : ∀ j, ω ∉ Brow0 j
    · by_cases hgood1 : ∀ j, ω ∉ Brow1 j
      · left
        left
        have hgap := hpert ω hgood0 hgood1
        have htri : ‖balancedThreeStepCentroid anchors τ i Y ω - c‖ ≤
            ‖balancedThreeStepCentroid anchors τ i Y ω - Cref ω‖ +
              ‖Cref ω - c‖ := by
          have hrewrite : balancedThreeStepCentroid anchors τ i Y ω - c =
              (balancedThreeStepCentroid anchors τ i Y ω - Cref ω) + (Cref ω - c) := by
            ring
          rw [hrewrite]
          exact norm_add_le _ _
        have hupper : ‖balancedThreeStepCentroid anchors τ i Y ω - c‖ ≤
            128 * δ * R + ‖Cref ω - c‖ := by linarith
        linarith
      · right
        push Not at hgood1
        exact hgood1
    · left
      right
      push Not at hgood0
      exact hgood0
  have hbad0 : P (⋃ j, Brow0 j) ≤ ∑ j, B0 j := by
    calc P (⋃ j, Brow0 j) ≤ ∑ j, P (Brow0 j) :=
        measure_iUnion_fintype_le P Brow0
      _ ≤ ∑ j, B0 j :=
        Finset.sum_le_sum fun j _ => by
          simpa [Brow0] using hrow0tail j
  have hbad1 : P (⋃ j, Brow1 j) ≤ ∑ j, B1 j := by
    calc P (⋃ j, Brow1 j) ≤ ∑ j, P (Brow1 j) :=
        measure_iUnion_fintype_le P Brow1
      _ ≤ ∑ j, B1 j :=
        Finset.sum_le_sum fun j _ => by
          simpa [Brow1] using hrow1tail j
  have hrefrow0 :
      P ({ω | ε < ‖Cref ω - c‖} ∪ ⋃ j, Brow0 j) ≤
        P {ω | ε < ‖Cref ω - c‖} + P (⋃ j, Brow0 j) :=
    measure_union_le _ _
  calc P {ω | ε + 128 * δ * R <
          ‖balancedThreeStepCentroid anchors τ i Y ω - c‖}
      ≤ P (({ω | ε < ‖Cref ω - c‖} ∪ ⋃ j, Brow0 j) ∪ ⋃ j, Brow1 j) :=
        measure_mono hsub
    _ ≤ P ({ω | ε < ‖Cref ω - c‖} ∪ ⋃ j, Brow0 j) + P (⋃ j, Brow1 j) :=
        measure_union_le _ _
    _ ≤ (P {ω | ε < ‖Cref ω - c‖} + P (⋃ j, Brow0 j)) + P (⋃ j, Brow1 j) := by
        exact add_le_add hrefrow0 le_rfl
    _ ≤ (Bref + ∑ j, B0 j) + ∑ j, B1 j := by
        exact add_le_add (add_le_add (by simpa [Cref] using href) hbad0) hbad1

/-- The realized two-step balanced centroid (weight form; the per-anchor
balancing factor cancels by `selfNormalizedCentroid_eq_of_common_scale`). -/
noncomputable def balancedTwoStepCentroid {Ω : Type*} (anchors : Fin M → ℝ)
    (τ : ℝ) (i : Fin M) (Y : Fin N → Ω → ℝ) (ω : Ω) : ℝ :=
  (∑ l, twoStepWeight anchors τ i (realizedRowMass anchors τ Y ω) (Y l ω))⁻¹ •
    ∑ l, twoStepWeight anchors τ i (realizedRowMass anchors τ Y ω) (Y l ω) •
      Y l ω

/-- The fixed-reference `t = 2` balanced centroid: same weight formula, but
with caller-supplied row masses `Mbar` instead of the random realized row
masses. -/
noncomputable def balancedTwoStepReferenceCentroid {Ω : Type*}
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M) (Mbar : Fin M → ℝ)
    (Y : Fin N → Ω → ℝ) (ω : Ω) : ℝ :=
  (∑ l, twoStepWeight anchors τ i Mbar (Y l ω))⁻¹ •
    ∑ l, twoStepWeight anchors τ i Mbar (Y l ω) • Y l ω

/-- The normalized two-branch `t = 2` balanced drift, i.e. positive centroid
minus negative centroid.  This is the identifiability-relevant normalized field;
the raw Algorithm-2-style field is a positive mass product times this object. -/
noncomputable def balancedTwoStepNormalizedDrift {Ω : Type*}
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (Ypos : Fin Npos → Ω → ℝ) (Yneg : Fin Nneg → Ω → ℝ) (ω : Ω) : ℝ :=
  balancedTwoStepCentroid anchors τ i Ypos ω -
    balancedTwoStepCentroid anchors τ i Yneg ω

/-- Fixed-reference normalized two-branch `t = 2` balanced drift. -/
noncomputable def balancedTwoStepReferenceDrift {Ω : Type*}
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (MbarPos MbarNeg : Fin M → ℝ)
    (Ypos : Fin Npos → Ω → ℝ) (Yneg : Fin Nneg → Ω → ℝ) (ω : Ω) : ℝ :=
  balancedTwoStepReferenceCentroid anchors τ i MbarPos Ypos ω -
    balancedTwoStepReferenceCentroid anchors τ i MbarNeg Yneg ω

/-- Closed-form right-hand side for the `t = 2` balanced centroid deviation
theorem. -/
noncomputable def balancedTwoStepCentroidDeviationBound
    (μw : Fin N → ℝ) (Mbar : Fin M → ℝ)
    (tw ε σ σw b δ σrow : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal ((2 * N * σ ^ 2 + 2 * N ^ 2 * b ^ 2) /
      (((∑ l, μw l) - tw) ^ 2 * ε ^ 2)) +
    ENNReal.ofReal ((N * σw ^ 2) / tw ^ 2) +
    ∑ j, ENNReal.ofReal ((N * σrow ^ 2) / (δ * Mbar j) ^ 2)

/-- **Two-branch normalized drift assembly.**  Any high-probability bounds for
the positive and negative `t = 2` centroids compose into a high-probability
bound for their normalized drift difference.  This is pure metric/event
bookkeeping; it is independent of the probabilistic mechanism used to obtain
the two centroid bounds. -/
theorem balancedTwoStepNormalizedDrift_deviation_prob_le_of_centroids
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (Ypos : Fin Npos → Ω → ℝ) (Yneg : Fin Nneg → Ω → ℝ)
    (cPos cNeg : ℝ) {εPos εNeg : ℝ} {Bpos Bneg : ℝ≥0∞}
    (hpos : P {ω | εPos <
        ‖balancedTwoStepCentroid anchors τ i Ypos ω - cPos‖} ≤ Bpos)
    (hneg : P {ω | εNeg <
        ‖balancedTwoStepCentroid anchors τ i Yneg ω - cNeg‖} ≤ Bneg) :
    P {ω | εPos + εNeg <
        ‖balancedTwoStepNormalizedDrift anchors τ i Ypos Yneg ω -
          (cPos - cNeg)‖} ≤ Bpos + Bneg := by
  have hsub :
      {ω | εPos + εNeg <
          ‖balancedTwoStepNormalizedDrift anchors τ i Ypos Yneg ω -
            (cPos - cNeg)‖} ⊆
        {ω | εPos <
          ‖balancedTwoStepCentroid anchors τ i Ypos ω - cPos‖} ∪
        {ω | εNeg <
          ‖balancedTwoStepCentroid anchors τ i Yneg ω - cNeg‖} := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union] at hω ⊢
    by_cases hp : εPos <
        ‖balancedTwoStepCentroid anchors τ i Ypos ω - cPos‖
    · exact Or.inl hp
    · right
      by_contra hn
      have hp_le : ‖balancedTwoStepCentroid anchors τ i Ypos ω - cPos‖ ≤ εPos :=
        le_of_not_gt hp
      have hn_le : ‖balancedTwoStepCentroid anchors τ i Yneg ω - cNeg‖ ≤ εNeg :=
        le_of_not_gt hn
      have hrewrite :
          balancedTwoStepNormalizedDrift anchors τ i Ypos Yneg ω - (cPos - cNeg) =
            (balancedTwoStepCentroid anchors τ i Ypos ω - cPos) -
              (balancedTwoStepCentroid anchors τ i Yneg ω - cNeg) := by
        unfold balancedTwoStepNormalizedDrift
        ring
      have htri : ‖balancedTwoStepNormalizedDrift anchors τ i Ypos Yneg ω -
          (cPos - cNeg)‖ ≤ εPos + εNeg := by
        rw [hrewrite]
        calc ‖(balancedTwoStepCentroid anchors τ i Ypos ω - cPos) -
              (balancedTwoStepCentroid anchors τ i Yneg ω - cNeg)‖
            ≤ ‖balancedTwoStepCentroid anchors τ i Ypos ω - cPos‖ +
                ‖balancedTwoStepCentroid anchors τ i Yneg ω - cNeg‖ := norm_sub_le _ _
          _ ≤ εPos + εNeg := add_le_add hp_le hn_le
      exact (not_lt_of_ge htri) hω
  calc P {ω | εPos + εNeg <
        ‖balancedTwoStepNormalizedDrift anchors τ i Ypos Yneg ω -
          (cPos - cNeg)‖}
      ≤ P ({ω | εPos <
          ‖balancedTwoStepCentroid anchors τ i Ypos ω - cPos‖} ∪
        {ω | εNeg <
          ‖balancedTwoStepCentroid anchors τ i Yneg ω - cNeg‖}) := measure_mono hsub
    _ ≤ P {ω | εPos <
          ‖balancedTwoStepCentroid anchors τ i Ypos ω - cPos‖} +
        P {ω | εNeg <
          ‖balancedTwoStepCentroid anchors τ i Yneg ω - cNeg‖} := measure_union_le _ _
    _ ≤ Bpos + Bneg := add_le_add hpos hneg

/-- **Headline: deviation probability for the two-step balanced centroid.**
The realized `t = 2` centroid deviates from the target `c` by more than
`ε + 16δR` with probability at most the fixed-weight SNIS deviation bound at
the *reference* weight (population row masses `Mbar`) plus one row-mass
Chebyshev tail per anchor.  The batch-dependence of the balanced weights —
the open item of the extension track — is fully absorbed by the row-mass
concentration and the deterministic `4δ` propagation. -/
theorem balancedTwoStepCentroid_deviation_prob_le
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
    [Nonempty (Fin M)] (hN : 0 < N)
    (anchors : Fin M → ℝ) (τ : ℝ) (i : Fin M)
    (Y : Fin N → Ω → ℝ) (c : ℝ) (μ : Fin N → ℝ) (μw : Fin N → ℝ)
    (rowμ : Fin M → Fin N → ℝ) (Mbar : Fin M → ℝ)
    {kmax wmax R b σ σw tw δ ε σrow : ℝ}
    (hkmax : 0 ≤ kmax) (hwmax : 0 ≤ wmax) (hR : 0 ≤ R)
    (hε : 0 < ε) (htw : 0 < tw) (hδ0 : 0 < δ) (hδ : δ ≤ 1 / 8)
    (hdlow : 0 < (∑ l, μw l) - tw)
    (hMbar : ∀ j, 0 < Mbar j)
    (hYmeas : ∀ l, Measurable (Y l))
    (hindep : ∀ l k, l ≠ k → IndepFun (Y l) (Y k) P)
    (hkb : ∀ j l ω, |algorithm2Kernel τ (anchors j) (Y l ω)| ≤ kmax)
    (hrowμ : ∀ j l, ∫ ω, algorithm2Kernel τ (anchors j) (Y l ω) ∂P = rowμ j l)
    (hMbar_eq : ∀ j, Mbar j = ∑ l, rowμ j l)
    (hσrow : ∀ j l,
      ∫ ω, (algorithm2Kernel τ (anchors j) (Y l ω) - rowμ j l) ^ 2 ∂P ≤ σrow ^ 2)
    (hwabs : ∀ l ω, |twoStepWeight anchors τ i Mbar (Y l ω)| ≤ wmax)
    (hYbd : ∀ l ω, ‖Y l ω - c‖ ≤ R)
    (hμ : ∀ l, ∫ ω, twoStepWeight anchors τ i Mbar (Y l ω) • (Y l ω - c) ∂P = μ l)
    (hb : ∀ l, ‖μ l‖ ≤ b)
    (hσ : ∀ l, ∫ ω, ‖twoStepWeight anchors τ i Mbar (Y l ω) • (Y l ω - c) -
      μ l‖ ^ 2 ∂P ≤ σ ^ 2)
    (hμw : ∀ l, ∫ ω, twoStepWeight anchors τ i Mbar (Y l ω) ∂P = μw l)
    (hσw : ∀ l, ∫ ω, (twoStepWeight anchors τ i Mbar (Y l ω) - μw l) ^ 2 ∂P ≤
      σw ^ 2) :
    P {ω | ε + 16 * δ * R <
        ‖balancedTwoStepCentroid anchors τ i Y ω - c‖} ≤
      (ENNReal.ofReal ((2 * N * σ ^ 2 + 2 * N ^ 2 * b ^ 2) /
          (((∑ l, μw l) - tw) ^ 2 * ε ^ 2)) +
        ENNReal.ofReal ((N * σw ^ 2) / tw ^ 2)) +
      ∑ j, ENNReal.ofReal ((N * σrow ^ 2) / (δ * Mbar j) ^ 2) := by
  -- reference centroid and its SNIS deviation bound (existing theorem)
  have hsnis := selfNormalizedIndexed_deviation_prob_le P hN Y
    (fun _ => twoStepWeight anchors τ i Mbar) c μ μw
    hwmax hR hε htw hdlow hYmeas
    (fun _ => twoStepWeight_measurable anchors τ i Mbar)
    hindep hwabs hYbd hμ hb hσ hμw hσw
  -- per-anchor row-mass tails (two-sided Chebyshev)
  have htail : ∀ j, P {ω |
      δ * Mbar j < |realizedRowMass anchors τ Y ω j - Mbar j|} ≤
      ENNReal.ofReal ((N * σrow ^ 2) / (δ * Mbar j) ^ 2) := by
    intro j
    have h := weightSum_deviation_prob_le P hN Y
      (fun _ y => algorithm2Kernel τ (anchors j) y) (fun l => rowμ j l)
      hkmax (mul_pos hδ0 (hMbar j)) hYmeas
      (fun _ => (PaperFiniteIdentifiability.algorithm2Kernel_continuous_snd τ
        (anchors j)).measurable)
      hindep (fun l ω => hkb j l ω) (fun l => hrowμ j l) (fun l => hσrow j l)
    have hev : {ω | δ * Mbar j <
        |realizedRowMass anchors τ Y ω j - Mbar j|} =
        {ω | δ * Mbar j <
          |(∑ l, algorithm2Kernel τ (anchors j) (Y l ω)) - ∑ l, rowμ j l|} := by
      ext ω
      simp [realizedRowMass, hMbar_eq j]
    simpa [hev] using h
  let Cref : Ω → ℝ := fun ω =>
    (∑ l, twoStepWeight anchors τ i Mbar (Y l ω))⁻¹ •
      ∑ l, twoStepWeight anchors τ i Mbar (Y l ω) • Y l ω
  let Brow : Fin M → Set Ω := fun j =>
    {ω | δ * Mbar j < |realizedRowMass anchors τ Y ω j - Mbar j|}
  haveI : Nonempty (Fin N) := ⟨⟨0, hN⟩⟩
  have hpert : ∀ ω, (∀ j, ω ∉ Brow j) →
      ‖balancedTwoStepCentroid anchors τ i Y ω - Cref ω‖ ≤ 16 * δ * R := by
    intro ω hgood
    have hδnonneg : 0 ≤ δ := hδ0.le
    have heta0 : 0 ≤ 4 * δ := by nlinarith
    have heta1 : 4 * δ < 1 := by nlinarith [hδ]
    have hrelω : ∀ l,
        |twoStepWeight anchors τ i (realizedRowMass anchors τ Y ω) (Y l ω) -
          twoStepWeight anchors τ i Mbar (Y l ω)| ≤
        (4 * δ) * twoStepWeight anchors τ i Mbar (Y l ω) := by
      intro l
      refine twoStepWeight_rel_of_rowMass_rel anchors τ i hδnonneg
        (by linarith) hMbar ?_ (Y l ω)
      intro j
      have hj := hgood j
      simp only [Brow, Set.mem_setOf_eq, not_lt] at hj
      exact hj
    have hdiam : ∀ s t, ‖Y s ω - Y t ω‖ ≤ 2 * R := by
      intro s t
      have hrewrite : Y s ω - Y t ω = (Y s ω - c) - (Y t ω - c) := by ring
      rw [hrewrite]
      calc ‖(Y s ω - c) - (Y t ω - c)‖
          ≤ ‖Y s ω - c‖ + ‖Y t ω - c‖ := norm_sub_le _ _
        _ ≤ R + R := add_le_add (hYbd s ω) (hYbd t ω)
        _ = 2 * R := by ring
    have hcent := selfNormalizedCentroid_relative_perturbation
      (fun l => twoStepWeight anchors τ i (realizedRowMass anchors τ Y ω) (Y l ω))
      (fun l => twoStepWeight anchors τ i Mbar (Y l ω))
      (fun l => Y l ω) heta0 heta1
      (fun l => twoStepWeight_pos anchors τ i hMbar (Y l ω))
      hrelω hdiam
    have hsmall :
        (4 * δ) * (2 * R) / (1 - 4 * δ) ≤ 16 * δ * R := by
      have hden : (1 / 2 : ℝ) ≤ 1 - 4 * δ := by nlinarith [hδ]
      have hnum : 0 ≤ 8 * δ * R := by nlinarith [hδnonneg, hR]
      calc (4 * δ) * (2 * R) / (1 - 4 * δ)
          = (8 * δ * R) / (1 - 4 * δ) := by ring
        _ ≤ (8 * δ * R) / (1 / 2 : ℝ) :=
            div_le_div_of_nonneg_left hnum (by norm_num) hden
        _ = 16 * δ * R := by ring
    simpa [balancedTwoStepCentroid, Cref] using le_trans hcent hsmall
  have hsub :
      {ω | ε + 16 * δ * R <
          ‖balancedTwoStepCentroid anchors τ i Y ω - c‖} ⊆
        {ω | ε < ‖Cref ω - c‖} ∪ ⋃ j, Brow j := by
    intro ω hω
    simp only [Set.mem_setOf_eq, Set.mem_union, Set.mem_iUnion] at hω ⊢
    by_cases hgood : ∀ j, ω ∉ Brow j
    · left
      have hgap := hpert ω hgood
      have htri : ‖balancedTwoStepCentroid anchors τ i Y ω - c‖ ≤
          ‖balancedTwoStepCentroid anchors τ i Y ω - Cref ω‖ +
            ‖Cref ω - c‖ := by
        have hrewrite : balancedTwoStepCentroid anchors τ i Y ω - c =
            (balancedTwoStepCentroid anchors τ i Y ω - Cref ω) + (Cref ω - c) := by
          ring
        rw [hrewrite]
        exact norm_add_le _ _
      have hupper : ‖balancedTwoStepCentroid anchors τ i Y ω - c‖ ≤
          16 * δ * R + ‖Cref ω - c‖ := by linarith
      linarith
    · right
      push Not at hgood
      exact hgood
  have hbadRows : P (⋃ j, Brow j) ≤
      ∑ j, ENNReal.ofReal ((N * σrow ^ 2) / (δ * Mbar j) ^ 2) := by
    calc P (⋃ j, Brow j) ≤ ∑ j, P (Brow j) :=
        measure_iUnion_fintype_le P Brow
      _ ≤ ∑ j, ENNReal.ofReal ((N * σrow ^ 2) / (δ * Mbar j) ^ 2) :=
        Finset.sum_le_sum fun j _ => by
          simpa [Brow] using htail j
  calc P {ω | ε + 16 * δ * R <
          ‖balancedTwoStepCentroid anchors τ i Y ω - c‖}
      ≤ P ({ω | ε < ‖Cref ω - c‖} ∪ ⋃ j, Brow j) :=
        measure_mono hsub
    _ ≤ P {ω | ε < ‖Cref ω - c‖} + P (⋃ j, Brow j) :=
        measure_union_le _ _
    _ ≤ (ENNReal.ofReal ((2 * N * σ ^ 2 + 2 * N ^ 2 * b ^ 2) /
          (((∑ l, μw l) - tw) ^ 2 * ε ^ 2)) +
        ENNReal.ofReal ((N * σw ^ 2) / tw ^ 2)) +
      ∑ j, ENNReal.ofReal ((N * σrow ^ 2) / (δ * Mbar j) ^ 2) := by
        exact add_le_add (by simpa [Cref] using hsnis) hbadRows

end TwoStep

end Algorithm2
end DriftingIdentifiability
