import DriftingIdentifiability.ColumnReweightedTwoAtom

/-!
# The self-mask as an explicit perturbation (Objective 4)

The promoted Objective-4 route analyzes Algorithm 2 with `selfMask = false`.
The paper's implementation masks reused generated samples with
`dist_neg += eye(N) * 1e6`, i.e. `selfMask = eye`.  This module quantifies that
difference exactly.

The key observation is that the `1e6` penalty acts *multiplicatively* on the
exponentiated logits: each masked entry's weight is exactly
`maskPenaltyFactor = exp(-10⁶/τ)` times its unmasked value
(`maskedWeight_eq_factor_mul_noMaskWeight`).  The masked estimator is therefore
a `δ`-perturbation, `δ = exp(-10⁶/τ)`, of the **deleted** estimator — the one
with masked entries removed — *not* of the plain no-mask estimator on the same
samples (the self sample sits at distance zero and can dominate its softmax
row, which is exactly why the implementation masks it).

The comparison chain proved here:

* `deletedDrift` — the leave-masked-out estimator; at `selfMask = false` it is
  *definitionally reconciled* with `algorithm2Drift`
  (`deletedDrift_false_eq_algorithm2Drift`), so it is the promoted no-mask
  object, not a new ad-hoc target.
* A generic perturbation inequality for bi-softmax weights `w/√(r·c)`
  (`div_sqrt_mass_perturbation`), with the geometric-mean square root handled
  by `√b - √a ≤ √(b-a)`.
* `algorithm2Affinity_sub_deletedAffinity_abs_le`: every masked affinity is
  within `η := (δ + √(3(Npos+Nneg)·Nx·δ))/√(Npos·wmin²)` of its deleted
  counterpart, under a positive temperature, a kernel floor `wmin`, and the
  hypothesis that every negative column keeps at least one unmasked anchor
  (true for the paper's `eye(N)` mask once `Nx ≥ 2`:
  `eyeMask_column_unmasked`).
* `algorithm2Drift_sub_deletedDrift_norm_le`: the drifts differ by at most
  `4·Npos·Nneg·R₀·η`.

The constant is astronomically small (`δ = exp(-10⁶/τ)`), but the point is
that it is *explicit and deterministic*: no distributional assumption is used.
The sup-affinity route is legitimate here — unlike in the logged consistency
failure — because `δ` is an absolute constant, not a `1/N`-sized fluctuation.

What this module does **not** claim: that the masked estimator is close to the
*full* no-mask estimator on the same samples (false in general), or any
statistical property of the deleted estimator (its per-anchor reduced-sample
SNIS analysis is future work).
-/

open scoped BigOperators

namespace DriftingIdentifiability
namespace Algorithm2

open Paper

universe u

/-! ## Generic real inequalities -/

section RealLemmas

/-- Square-root subadditivity. -/
theorem sqrt_add_le_sqrt_add_sqrt {a b : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) :
    Real.sqrt (a + b) ≤ Real.sqrt a + Real.sqrt b := by
  have hsq : a + b ≤ (Real.sqrt a + Real.sqrt b) ^ 2 := by
    have haa := Real.sq_sqrt ha
    have hbb := Real.sq_sqrt hb
    nlinarith [mul_nonneg (Real.sqrt_nonneg a) (Real.sqrt_nonneg b)]
  calc Real.sqrt (a + b) ≤ Real.sqrt ((Real.sqrt a + Real.sqrt b) ^ 2) :=
        Real.sqrt_le_sqrt hsq
    _ = Real.sqrt a + Real.sqrt b := Real.sqrt_sq (by positivity)

/-- `√b - √a ≤ √(b - a)` for `0 ≤ a ≤ b`. -/
theorem sqrt_sub_sqrt_le_sqrt_sub {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) :
    Real.sqrt b - Real.sqrt a ≤ Real.sqrt (b - a) := by
  have h : Real.sqrt b ≤ Real.sqrt a + Real.sqrt (b - a) := by
    calc Real.sqrt b = Real.sqrt (a + (b - a)) := by rw [show a + (b - a) = b by ring]
      _ ≤ Real.sqrt a + Real.sqrt (b - a) :=
          sqrt_add_le_sqrt_add_sqrt ha (by linarith)
  linarith

/-- **Bi-softmax weight perturbation.**  If a weight grows from `v` to `w` and
its row/column masses grow from `(rv, cv)` to `(rw, cw)`, all masses staying
above the floors `r₀, c₀ > 0` and the weight below its own masses, then the
normalized bi-softmax weight `w/√(r·c)` moves by at most the weight increment
plus the square root of the mass-product increment, both over `√(r₀·c₀)`. -/
theorem div_sqrt_mass_perturbation
    {v w rv rw cv cw r0 c0 : ℝ}
    (hv : 0 ≤ v) (hvw : v ≤ w)
    (hr0 : 0 < r0) (hc0 : 0 < c0)
    (hr : r0 ≤ rv) (hc : c0 ≤ cv)
    (hrw : rv ≤ rw) (hcw : cv ≤ cw)
    (hvr : v ≤ rv) (hvc : v ≤ cv) :
    |w / Real.sqrt (rw * cw) - v / Real.sqrt (rv * cv)| ≤
      (w - v) / Real.sqrt (r0 * c0) +
        Real.sqrt (rw * cw - rv * cv) / Real.sqrt (r0 * c0) := by
  have hrv0 : (0 : ℝ) < rv := lt_of_lt_of_le hr0 hr
  have hcv0 : (0 : ℝ) < cv := lt_of_lt_of_le hc0 hc
  have ha0 : (0 : ℝ) < rv * cv := mul_pos hrv0 hcv0
  have hb0 : (0 : ℝ) < rw * cw :=
    mul_pos (lt_of_lt_of_le hrv0 hrw) (lt_of_lt_of_le hcv0 hcw)
  have hab : rv * cv ≤ rw * cw := mul_le_mul hrw hcw hcv0.le (le_trans hrv0.le hrw)
  have h00 : (0 : ℝ) < r0 * c0 := mul_pos hr0 hc0
  have hsa : 0 < Real.sqrt (rv * cv) := Real.sqrt_pos.mpr ha0
  have hsb : 0 < Real.sqrt (rw * cw) := Real.sqrt_pos.mpr hb0
  have hs0 : 0 < Real.sqrt (r0 * c0) := Real.sqrt_pos.mpr h00
  have hsab : Real.sqrt (rv * cv) ≤ Real.sqrt (rw * cw) := Real.sqrt_le_sqrt hab
  have hs0a : Real.sqrt (r0 * c0) ≤ Real.sqrt (rv * cv) :=
    Real.sqrt_le_sqrt (mul_le_mul hr hc hc0.le hrv0.le)
  have hvsa : v ≤ Real.sqrt (rv * cv) := by
    have h2 : v ^ 2 ≤ rv * cv := by nlinarith
    calc v = Real.sqrt (v ^ 2) := (Real.sqrt_sq hv).symm
      _ ≤ Real.sqrt (rv * cv) := Real.sqrt_le_sqrt h2
  have hw0 : 0 ≤ w := le_trans hv hvw
  rw [abs_sub_le_iff]
  constructor
  · -- upper direction: dominated by the weight-increment term
    have h1 : w / Real.sqrt (rw * cw) ≤ w / Real.sqrt (rv * cv) := by
      rw [div_eq_mul_one_div w, div_eq_mul_one_div w]
      exact mul_le_mul_of_nonneg_left (one_div_le_one_div_of_le hsa hsab) hw0
    have h3 : (w - v) / Real.sqrt (rv * cv) ≤ (w - v) / Real.sqrt (r0 * c0) := by
      rw [div_eq_mul_one_div (w - v), div_eq_mul_one_div (w - v)]
      exact mul_le_mul_of_nonneg_left (one_div_le_one_div_of_le hs0 hs0a) (by linarith)
    have h4 : 0 ≤ Real.sqrt (rw * cw - rv * cv) / Real.sqrt (r0 * c0) :=
      div_nonneg (Real.sqrt_nonneg _) hs0.le
    have h2 : w / Real.sqrt (rv * cv) - v / Real.sqrt (rv * cv)
        = (w - v) / Real.sqrt (rv * cv) := (sub_div _ _ _).symm
    linarith
  · -- lower direction: dominated by the mass-increment term
    have hstep1 : v / Real.sqrt (rv * cv) - w / Real.sqrt (rw * cw) ≤
        v / Real.sqrt (rv * cv) - v / Real.sqrt (rw * cw) := by
      have hvb : v / Real.sqrt (rw * cw) ≤ w / Real.sqrt (rw * cw) := by
        rw [div_eq_mul_one_div v, div_eq_mul_one_div w]
        exact mul_le_mul_of_nonneg_right hvw (by positivity)
      linarith
    have hfactor : v / Real.sqrt (rv * cv) - v / Real.sqrt (rw * cw)
        = v * ((Real.sqrt (rw * cw) - Real.sqrt (rv * cv)) /
            (Real.sqrt (rv * cv) * Real.sqrt (rw * cw))) := by
      field_simp
    have hquot_nonneg : 0 ≤ (Real.sqrt (rw * cw) - Real.sqrt (rv * cv)) /
        (Real.sqrt (rv * cv) * Real.sqrt (rw * cw)) :=
      div_nonneg (by linarith) (by positivity)
    have hv_le : v * ((Real.sqrt (rw * cw) - Real.sqrt (rv * cv)) /
          (Real.sqrt (rv * cv) * Real.sqrt (rw * cw)))
        ≤ Real.sqrt (rv * cv) * ((Real.sqrt (rw * cw) - Real.sqrt (rv * cv)) /
          (Real.sqrt (rv * cv) * Real.sqrt (rw * cw))) :=
      mul_le_mul_of_nonneg_right hvsa hquot_nonneg
    have hcollapse : Real.sqrt (rv * cv) * ((Real.sqrt (rw * cw) - Real.sqrt (rv * cv)) /
          (Real.sqrt (rv * cv) * Real.sqrt (rw * cw)))
        = (Real.sqrt (rw * cw) - Real.sqrt (rv * cv)) / Real.sqrt (rw * cw) := by
      rw [mul_div_assoc']
      exact mul_div_mul_left _ _ hsa.ne'
    have hnum := sqrt_sub_sqrt_le_sqrt_sub ha0.le hab
    have hfinal : (Real.sqrt (rw * cw) - Real.sqrt (rv * cv)) / Real.sqrt (rw * cw)
        ≤ Real.sqrt (rw * cw - rv * cv) / Real.sqrt (r0 * c0) := by
      calc (Real.sqrt (rw * cw) - Real.sqrt (rv * cv)) / Real.sqrt (rw * cw)
          ≤ Real.sqrt (rw * cw - rv * cv) / Real.sqrt (rw * cw) := by
            rw [div_eq_mul_one_div _ (Real.sqrt (rw * cw)),
              div_eq_mul_one_div (Real.sqrt (rw * cw - rv * cv)) (Real.sqrt (rw * cw))]
            exact mul_le_mul_of_nonneg_right hnum (by positivity)
        _ ≤ Real.sqrt (rw * cw - rv * cv) / Real.sqrt (r0 * c0) := by
            rw [div_eq_mul_one_div (Real.sqrt (rw * cw - rv * cv)) (Real.sqrt (rw * cw)),
              div_eq_mul_one_div (Real.sqrt (rw * cw - rv * cv)) (Real.sqrt (r0 * c0))]
            exact mul_le_mul_of_nonneg_left
              (one_div_le_one_div_of_le hs0 (le_trans hs0a hsab)) (Real.sqrt_nonneg _)
    have hterm1 : 0 ≤ (w - v) / Real.sqrt (r0 * c0) := div_nonneg (by linarith) hs0.le
    calc v / Real.sqrt (rv * cv) - w / Real.sqrt (rw * cw)
        ≤ v / Real.sqrt (rv * cv) - v / Real.sqrt (rw * cw) := hstep1
      _ = v * ((Real.sqrt (rw * cw) - Real.sqrt (rv * cv)) /
            (Real.sqrt (rv * cv) * Real.sqrt (rw * cw))) := hfactor
      _ ≤ (Real.sqrt (rw * cw) - Real.sqrt (rv * cv)) / Real.sqrt (rw * cw) := by
          rw [← hcollapse]; exact hv_le
      _ ≤ Real.sqrt (rw * cw - rv * cv) / Real.sqrt (r0 * c0) := hfinal
      _ ≤ (w - v) / Real.sqrt (r0 * c0) +
            Real.sqrt (rw * cw - rv * cv) / Real.sqrt (r0 * c0) := by linarith

/-- Products of `[0,1]` quantities move by at most the sum of the moves. -/
theorem abs_mul_sub_mul_le {a b c d : ℝ}
    (hb : 0 ≤ b) (hb1 : b ≤ 1) (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    |a * b - c * d| ≤ |a - c| + |b - d| := by
  have h : a * b - c * d = (a - c) * b + c * (b - d) := by ring
  rw [h]
  calc |(a - c) * b + c * (b - d)| ≤ |(a - c) * b| + |c * (b - d)| := abs_add_le _ _
    _ = |a - c| * |b| + |c| * |b - d| := by rw [abs_mul, abs_mul]
    _ ≤ |a - c| * 1 + 1 * |b - d| := by
        refine add_le_add ?_ ?_
        · exact mul_le_mul_of_nonneg_left (by rw [abs_of_nonneg hb]; exact hb1)
            (abs_nonneg _)
        · exact mul_le_mul_of_nonneg_right (by rw [abs_of_nonneg hc]; exact hc1)
            (abs_nonneg _)
    _ = |a - c| + |b - d| := by ring

end RealLemmas

/-! ## The masked, no-mask, and deleted weight systems -/

section Definitions

variable {E : Type u} [NormedAddCommGroup E]
variable {Nx Npos Nneg : ℕ}
variable (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
variable (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)

/-- The exact multiplicative suppression applied by the `1e6` penalty. -/
noncomputable def maskPenaltyFactor : ℝ :=
  Real.exp (-(1000000 : ℝ) / temperature)

/-- The exponentiated no-mask logit: the raw sample weight. -/
noncomputable def noMaskWeight (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  Real.exp (algorithm2Logit x yPos yNeg temperature (fun _ _ => false) i s)

/-- The exponentiated masked logit. -/
noncomputable def maskedWeight (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  Real.exp (algorithm2Logit x yPos yNeg temperature selfMask i s)

/-- Per-entry multiplicative factor of the mask: `δ` on masked entries, `1`
elsewhere. -/
noncomputable def maskWeightFactor (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  match s with
  | Sum.inl _ => 1
  | Sum.inr j => if selfMask i j then maskPenaltyFactor temperature else 1

/-- Hard-deletion factor: `0` on masked entries, `1` elsewhere. -/
def deletedFactor (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  match s with
  | Sum.inl _ => 1
  | Sum.inr j => if selfMask i j then 0 else 1

/-- The leave-masked-out sample weight. -/
noncomputable def deletedWeight (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  deletedFactor selfMask i s * noMaskWeight x yPos yNeg temperature i s

noncomputable def maskedRowMass (i : Fin Nx) : ℝ :=
  ∑ s, maskedWeight x yPos yNeg temperature selfMask i s

noncomputable def maskedColumnMass (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  ∑ i, maskedWeight x yPos yNeg temperature selfMask i s

noncomputable def deletedRowMass (i : Fin Nx) : ℝ :=
  ∑ s, deletedWeight x yPos yNeg temperature selfMask i s

noncomputable def deletedColumnMass (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  ∑ i, deletedWeight x yPos yNeg temperature selfMask i s

/-- The leave-masked-out bi-softmax affinity. -/
noncomputable def deletedAffinity (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) : ℝ :=
  deletedWeight x yPos yNeg temperature selfMask i s /
    Real.sqrt (deletedRowMass x yPos yNeg temperature selfMask i *
      deletedColumnMass x yPos yNeg temperature selfMask s)

/-- The leave-masked-out drift estimator, in affinity pair-sum form. -/
noncomputable def deletedDrift [NormedSpace ℝ E] (i : Fin Nx) : E :=
  ∑ j : Fin Npos, ∑ l : Fin Nneg,
    (deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j) *
      deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l)) •
      (yPos j - yNeg l)

end Definitions

/-! ## Exact factorization and elementary bounds -/

section WeightBounds

variable {E : Type u} [NormedAddCommGroup E]
variable {Nx Npos Nneg : ℕ}
variable (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
variable (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)

theorem maskPenaltyFactor_pos : 0 < maskPenaltyFactor temperature :=
  Real.exp_pos _

theorem maskPenaltyFactor_le_one (hτ : ValidTemperature temperature) :
    maskPenaltyFactor temperature ≤ 1 := by
  have hτ' : (0 : ℝ) < temperature := hτ
  unfold maskPenaltyFactor
  rw [Real.exp_le_one_iff, neg_div]
  exact neg_nonpos.mpr (div_nonneg (by norm_num) hτ'.le)

/-- **Exact mask factorization.**  The `1e6` penalty multiplies each
exponentiated logit by exactly `maskPenaltyFactor` on masked entries and
leaves every other entry unchanged. -/
theorem maskedWeight_eq_factor_mul_noMaskWeight (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    maskedWeight x yPos yNeg temperature selfMask i s =
      maskWeightFactor temperature selfMask i s *
        noMaskWeight x yPos yNeg temperature i s := by
  cases s with
  | inl j =>
      simp [maskedWeight, noMaskWeight, maskWeightFactor, algorithm2Logit]
  | inr j =>
      by_cases h : selfMask i j = true
      · simp only [maskedWeight, noMaskWeight, maskWeightFactor, algorithm2Logit, h,
          if_true, Bool.false_eq_true, if_false, maskPenaltyFactor]
        rw [← Real.exp_add]
        congr 1
        ring
      · simp [maskedWeight, noMaskWeight, maskWeightFactor, algorithm2Logit, h]

theorem noMaskWeight_pos (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    0 < noMaskWeight x yPos yNeg temperature i s :=
  Real.exp_pos _

theorem noMaskWeight_le_one (hτ : ValidTemperature temperature)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    noMaskWeight x yPos yNeg temperature i s ≤ 1 := by
  unfold noMaskWeight
  rw [exp_algorithm2Logit_false]
  exact PaperFiniteIdentifiability.algorithm2Kernel_le_one hτ _ _

theorem deletedFactor_nonneg (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    0 ≤ deletedFactor selfMask i s := by
  cases s with
  | inl j => norm_num [deletedFactor]
  | inr j =>
      by_cases h : selfMask i j = true <;> simp [deletedFactor, h]

theorem deletedFactor_le_maskFactor (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedFactor selfMask i s ≤ maskWeightFactor temperature selfMask i s := by
  cases s with
  | inl j => simp [deletedFactor, maskWeightFactor]
  | inr j =>
      by_cases h : selfMask i j = true <;>
        simp [deletedFactor, maskWeightFactor, h, (maskPenaltyFactor_pos temperature).le]

theorem maskFactor_le_one (hτ : ValidTemperature temperature) (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    maskWeightFactor temperature selfMask i s ≤ 1 := by
  cases s with
  | inl j => simp [maskWeightFactor]
  | inr j =>
      by_cases h : selfMask i j = true <;>
        simp [maskWeightFactor, h, maskPenaltyFactor_le_one temperature hτ]

theorem maskFactor_sub_deletedFactor_le (hτ : ValidTemperature temperature)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    maskWeightFactor temperature selfMask i s - deletedFactor selfMask i s ≤
      maskPenaltyFactor temperature := by
  have hδ := maskPenaltyFactor_pos temperature
  have hδ1 := maskPenaltyFactor_le_one temperature hτ
  cases s with
  | inl j => simp only [maskWeightFactor, deletedFactor, sub_self]; linarith
  | inr j =>
      by_cases h : selfMask i j = true
      · simp [maskWeightFactor, deletedFactor, h]
      · simp [maskWeightFactor, deletedFactor, h]
        linarith

theorem deletedWeight_nonneg (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    0 ≤ deletedWeight x yPos yNeg temperature selfMask i s :=
  mul_nonneg (deletedFactor_nonneg selfMask i s)
    (noMaskWeight_pos x yPos yNeg temperature i s).le

theorem deletedWeight_le_maskedWeight (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedWeight x yPos yNeg temperature selfMask i s ≤
      maskedWeight x yPos yNeg temperature selfMask i s := by
  rw [maskedWeight_eq_factor_mul_noMaskWeight]
  exact mul_le_mul_of_nonneg_right (deletedFactor_le_maskFactor temperature selfMask i s)
    (noMaskWeight_pos x yPos yNeg temperature i s).le

theorem maskedWeight_sub_deletedWeight_le (hτ : ValidTemperature temperature)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    maskedWeight x yPos yNeg temperature selfMask i s -
        deletedWeight x yPos yNeg temperature selfMask i s ≤
      maskPenaltyFactor temperature := by
  rw [maskedWeight_eq_factor_mul_noMaskWeight]
  unfold deletedWeight
  rw [← sub_mul]
  calc (maskWeightFactor temperature selfMask i s - deletedFactor selfMask i s) *
        noMaskWeight x yPos yNeg temperature i s
      ≤ maskPenaltyFactor temperature * 1 := by
        refine mul_le_mul (maskFactor_sub_deletedFactor_le temperature selfMask hτ i s)
          (noMaskWeight_le_one x yPos yNeg temperature hτ i s)
          (noMaskWeight_pos x yPos yNeg temperature i s).le
          (maskPenaltyFactor_pos temperature).le
    _ = maskPenaltyFactor temperature := mul_one _

theorem maskedWeight_le_one (hτ : ValidTemperature temperature)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    maskedWeight x yPos yNeg temperature selfMask i s ≤ 1 := by
  rw [maskedWeight_eq_factor_mul_noMaskWeight]
  calc maskWeightFactor temperature selfMask i s *
        noMaskWeight x yPos yNeg temperature i s
      ≤ 1 * 1 := by
        refine mul_le_mul (maskFactor_le_one temperature selfMask hτ i s)
          (noMaskWeight_le_one x yPos yNeg temperature hτ i s)
          (noMaskWeight_pos x yPos yNeg temperature i s).le (by norm_num)
    _ = 1 := mul_one 1

end WeightBounds

/-! ## Deleted affinities and the no-mask reconciliation -/

section DeletedReconciliation

variable {E : Type u} [NormedAddCommGroup E]
variable {Nx Npos Nneg : ℕ}
variable (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
variable (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)

/-- Algebraic square-root form behind the geometric bi-softmax affinity. -/
theorem sqrt_div_mul_div_eq_div_sqrt {w r c : ℝ} (hw : 0 ≤ w) :
    Real.sqrt ((w / r) * (w / c)) = w / Real.sqrt (r * c) := by
  rw [show (w / r) * (w / c) = w ^ 2 / (r * c) by ring,
    Real.sqrt_div (sq_nonneg w), Real.sqrt_sq hw]

/-- Any Algorithm-2 affinity is the raw exponentiated logit divided by the
square root of its row and column masses. -/
theorem algorithm2Affinity_eq_maskedWeight_div_sqrt
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    algorithm2Affinity x yPos yNeg temperature selfMask i s =
      maskedWeight x yPos yNeg temperature selfMask i s /
        Real.sqrt (maskedRowMass x yPos yNeg temperature selfMask i *
          maskedColumnMass x yPos yNeg temperature selfMask s) := by
  unfold algorithm2Affinity algorithm2RowAffinity algorithm2ColumnAffinity
    finiteSoftmax maskedWeight maskedRowMass maskedColumnMass
  exact sqrt_div_mul_div_eq_div_sqrt (Real.exp_pos _).le

theorem deletedFactor_false (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    deletedFactor (fun _ _ => false) i s = 1 := by
  cases s <;> simp [deletedFactor]

theorem deletedWeight_false_eq_noMaskWeight
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    deletedWeight x yPos yNeg temperature (fun _ _ => false) i s =
      noMaskWeight x yPos yNeg temperature i s := by
  simp [deletedWeight, deletedFactor_false]

theorem maskedWeight_false_eq_noMaskWeight
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    maskedWeight x yPos yNeg temperature (fun _ _ => false) i s =
      noMaskWeight x yPos yNeg temperature i s := by
  cases s <;> simp [maskedWeight, noMaskWeight, algorithm2Logit]

theorem deletedRowMass_false_eq_maskedRowMass_false (i : Fin Nx) :
    deletedRowMass x yPos yNeg temperature (fun _ _ => false) i =
      maskedRowMass x yPos yNeg temperature (fun _ _ => false) i := by
  unfold deletedRowMass maskedRowMass
  apply Finset.sum_congr rfl
  intro s _
  rw [deletedWeight_false_eq_noMaskWeight, maskedWeight_false_eq_noMaskWeight]

theorem deletedColumnMass_false_eq_maskedColumnMass_false
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedColumnMass x yPos yNeg temperature (fun _ _ => false) s =
      maskedColumnMass x yPos yNeg temperature (fun _ _ => false) s := by
  unfold deletedColumnMass maskedColumnMass
  apply Finset.sum_congr rfl
  intro i _
  rw [deletedWeight_false_eq_noMaskWeight, maskedWeight_false_eq_noMaskWeight]

/-- With no mask, the deleted affinity is exactly Algorithm 2's ordinary
affinity. -/
theorem deletedAffinity_false_eq_algorithm2Affinity
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    deletedAffinity x yPos yNeg temperature (fun _ _ => false) i s =
      algorithm2Affinity x yPos yNeg temperature (fun _ _ => false) i s := by
  rw [algorithm2Affinity_eq_maskedWeight_div_sqrt]
  unfold deletedAffinity
  rw [deletedWeight_false_eq_noMaskWeight, maskedWeight_false_eq_noMaskWeight,
    deletedRowMass_false_eq_maskedRowMass_false,
    deletedColumnMass_false_eq_maskedColumnMass_false]

/-- The deleted estimator reconciles exactly with the promoted no-mask
Algorithm-2 drift when no entries are deleted. -/
theorem deletedDrift_false_eq_algorithm2Drift [NormedSpace ℝ E] (i : Fin Nx) :
    deletedDrift x yPos yNeg temperature (fun _ _ => false) i =
      algorithm2Drift x yPos yNeg temperature (fun _ _ => false) i := by
  rw [algorithm2Drift_eq_affinityPairSum]
  unfold deletedDrift
  apply Finset.sum_congr rfl
  intro j _
  apply Finset.sum_congr rfl
  intro l _
  rw [deletedAffinity_false_eq_algorithm2Affinity,
    deletedAffinity_false_eq_algorithm2Affinity]

/-- Paper-style eye mask: generated anchor `i` deletes its own negative sample
`j=i`. -/
def eyeMask {N : ℕ} (i j : Fin N) : Bool := decide (i = j)

/-- If there are at least two anchors, each negative column has some unmasked
anchor under the eye mask. -/
theorem eyeMask_column_unmasked {N : ℕ} (hN : 2 ≤ N) :
    ∀ j : Fin N, ∃ i : Fin N, eyeMask i j = false := by
  intro j
  have hcard : 1 < Fintype.card (Fin N) := by
    have hN1 : 1 < N := lt_of_lt_of_le (by norm_num) hN
    simpa [Fintype.card_fin] using hN1
  haveI : Nontrivial (Fin N) := Fintype.one_lt_card_iff_nontrivial.mp hcard
  rcases exists_ne j with ⟨i, hij⟩
  exact ⟨i, by simp [eyeMask, hij]⟩

end DeletedReconciliation

/-! ## Explicit mass bounds for the mask perturbation -/

section MassBounds

variable {E : Type u} [NormedAddCommGroup E]
variable {Nx Npos Nneg : ℕ}
variable (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
variable (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)

/-- Every negative sample column keeps at least one unmasked anchor.  This is
the structural hypothesis used by the explicit deleted-column mass lower bound.
For the paper's eye mask it follows from `eyeMask_column_unmasked` when
`Nx ≥ 2`. -/
def HasUnmaskedNegativeColumn : Prop :=
  ∀ j : Fin Nneg, ∃ i : Fin Nx, selfMask i j = false

/-- The explicit affinity-error scale produced by the deterministic mask
perturbation argument. -/
noncomputable def maskAffinityErrorBound
    (Nx Npos Nneg : ℕ) (wmin δ : ℝ) : ℝ :=
  δ / Real.sqrt (((Npos : ℝ) * wmin) * wmin) +
    Real.sqrt (3 * ((Npos + Nneg : ℕ) : ℝ) * (Nx : ℝ) * δ) /
      Real.sqrt (((Npos : ℝ) * wmin) * wmin)

theorem deletedWeight_le_one (hτ : ValidTemperature temperature)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    deletedWeight x yPos yNeg temperature selfMask i s ≤ 1 :=
  le_trans (deletedWeight_le_maskedWeight x yPos yNeg temperature selfMask i s)
    (maskedWeight_le_one x yPos yNeg temperature selfMask hτ i s)

theorem maskedRowMass_nonneg (i : Fin Nx) :
    0 ≤ maskedRowMass x yPos yNeg temperature selfMask i := by
  unfold maskedRowMass
  exact Finset.sum_nonneg fun s _ => (Real.exp_pos _).le

theorem maskedColumnMass_nonneg (s : Algorithm2SampleIndex Npos Nneg) :
    0 ≤ maskedColumnMass x yPos yNeg temperature selfMask s := by
  unfold maskedColumnMass
  exact Finset.sum_nonneg fun i _ => (Real.exp_pos _).le

theorem deletedRowMass_nonneg (i : Fin Nx) :
    0 ≤ deletedRowMass x yPos yNeg temperature selfMask i := by
  unfold deletedRowMass
  exact Finset.sum_nonneg fun s _ =>
    deletedWeight_nonneg x yPos yNeg temperature selfMask i s

theorem deletedColumnMass_nonneg (s : Algorithm2SampleIndex Npos Nneg) :
    0 ≤ deletedColumnMass x yPos yNeg temperature selfMask s := by
  unfold deletedColumnMass
  exact Finset.sum_nonneg fun i _ =>
    deletedWeight_nonneg x yPos yNeg temperature selfMask i s

theorem deletedWeight_le_deletedRowMass_of_term (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedWeight x yPos yNeg temperature selfMask i s ≤
      deletedRowMass x yPos yNeg temperature selfMask i := by
  unfold deletedRowMass
  exact Finset.single_le_sum
    (fun t _ => deletedWeight_nonneg x yPos yNeg temperature selfMask i t)
    (Finset.mem_univ s)

theorem deletedWeight_le_deletedColumnMass_of_term (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedWeight x yPos yNeg temperature selfMask i s ≤
      deletedColumnMass x yPos yNeg temperature selfMask s := by
  unfold deletedColumnMass
  exact Finset.single_le_sum
    (fun r _ => deletedWeight_nonneg x yPos yNeg temperature selfMask r s)
    (Finset.mem_univ i)

theorem deletedRowMass_le_maskedRowMass (i : Fin Nx) :
    deletedRowMass x yPos yNeg temperature selfMask i ≤
      maskedRowMass x yPos yNeg temperature selfMask i := by
  unfold deletedRowMass maskedRowMass
  exact Finset.sum_le_sum fun s _ =>
    deletedWeight_le_maskedWeight x yPos yNeg temperature selfMask i s

theorem deletedColumnMass_le_maskedColumnMass
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedColumnMass x yPos yNeg temperature selfMask s ≤
      maskedColumnMass x yPos yNeg temperature selfMask s := by
  unfold deletedColumnMass maskedColumnMass
  exact Finset.sum_le_sum fun i _ =>
    deletedWeight_le_maskedWeight x yPos yNeg temperature selfMask i s

theorem deletedRowMass_le_card (hτ : ValidTemperature temperature) (i : Fin Nx) :
    deletedRowMass x yPos yNeg temperature selfMask i ≤ (Npos + Nneg : ℝ) := by
  unfold deletedRowMass
  calc
    (∑ s : Algorithm2SampleIndex Npos Nneg,
        deletedWeight x yPos yNeg temperature selfMask i s)
        ≤ ∑ _s : Algorithm2SampleIndex Npos Nneg, (1 : ℝ) := by
          exact Finset.sum_le_sum fun s _ =>
            deletedWeight_le_one x yPos yNeg temperature selfMask hτ i s
    _ = (Npos + Nneg : ℝ) := by
        simp [Algorithm2SampleIndex, Fintype.card_sum]

theorem deletedColumnMass_le_card (hτ : ValidTemperature temperature)
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedColumnMass x yPos yNeg temperature selfMask s ≤ (Nx : ℝ) := by
  unfold deletedColumnMass
  calc
    (∑ i : Fin Nx, deletedWeight x yPos yNeg temperature selfMask i s)
        ≤ ∑ _i : Fin Nx, (1 : ℝ) := by
          exact Finset.sum_le_sum fun i _ =>
            deletedWeight_le_one x yPos yNeg temperature selfMask hτ i s
    _ = (Nx : ℝ) := by
        simp

theorem maskedRowMass_sub_deletedRowMass_le (hτ : ValidTemperature temperature)
    (i : Fin Nx) :
    maskedRowMass x yPos yNeg temperature selfMask i -
        deletedRowMass x yPos yNeg temperature selfMask i ≤
      (Npos + Nneg : ℝ) * maskPenaltyFactor temperature := by
  unfold maskedRowMass deletedRowMass
  rw [← Finset.sum_sub_distrib]
  calc
    (∑ s : Algorithm2SampleIndex Npos Nneg,
        (maskedWeight x yPos yNeg temperature selfMask i s -
          deletedWeight x yPos yNeg temperature selfMask i s))
        ≤ ∑ _s : Algorithm2SampleIndex Npos Nneg, maskPenaltyFactor temperature := by
          exact Finset.sum_le_sum fun s _ =>
            maskedWeight_sub_deletedWeight_le x yPos yNeg temperature selfMask hτ i s
    _ = (Npos + Nneg : ℝ) * maskPenaltyFactor temperature := by
        simp [Algorithm2SampleIndex, Fintype.card_sum]

theorem maskedColumnMass_sub_deletedColumnMass_le (hτ : ValidTemperature temperature)
    (s : Algorithm2SampleIndex Npos Nneg) :
    maskedColumnMass x yPos yNeg temperature selfMask s -
        deletedColumnMass x yPos yNeg temperature selfMask s ≤
      (Nx : ℝ) * maskPenaltyFactor temperature := by
  unfold maskedColumnMass deletedColumnMass
  rw [← Finset.sum_sub_distrib]
  calc
    (∑ i : Fin Nx,
        (maskedWeight x yPos yNeg temperature selfMask i s -
          deletedWeight x yPos yNeg temperature selfMask i s))
        ≤ ∑ _i : Fin Nx, maskPenaltyFactor temperature := by
          exact Finset.sum_le_sum fun i _ =>
            maskedWeight_sub_deletedWeight_le x yPos yNeg temperature selfMask hτ i s
    _ = (Nx : ℝ) * maskPenaltyFactor temperature := by
        simp

theorem deletedRowMass_lower_of_noMaskWeight_floor
    {wmin : ℝ}
    (hfloor : ∀ i s, wmin ≤ noMaskWeight x yPos yNeg temperature i s)
    (i : Fin Nx) :
    (Npos : ℝ) * wmin ≤ deletedRowMass x yPos yNeg temperature selfMask i := by
  unfold deletedRowMass
  rw [Fintype.sum_sum_type]
  calc
    (Npos : ℝ) * wmin = ∑ _j : Fin Npos, wmin := by
      simp
    _ ≤ ∑ j : Fin Npos,
          deletedWeight x yPos yNeg temperature selfMask i (Sum.inl j) := by
        exact Finset.sum_le_sum fun j _ => by
          simpa [deletedWeight, deletedFactor] using hfloor i (Sum.inl j)
    _ ≤ (∑ j : Fin Npos,
          deletedWeight x yPos yNeg temperature selfMask i (Sum.inl j)) +
        ∑ j : Fin Nneg,
          deletedWeight x yPos yNeg temperature selfMask i (Sum.inr j) := by
        exact le_add_of_nonneg_right
          (Finset.sum_nonneg fun j _ =>
            deletedWeight_nonneg x yPos yNeg temperature selfMask i (Sum.inr j))

theorem deletedColumnMass_lower_of_noMaskWeight_floor
    [Nonempty (Fin Nx)]
    {wmin : ℝ}
    (hfloor : ∀ i s, wmin ≤ noMaskWeight x yPos yNeg temperature i s)
    (hcol : HasUnmaskedNegativeColumn selfMask)
    (s : Algorithm2SampleIndex Npos Nneg) :
    wmin ≤ deletedColumnMass x yPos yNeg temperature selfMask s := by
  unfold deletedColumnMass
  cases s with
  | inl j =>
      let i0 : Fin Nx := Classical.choice inferInstance
      have hsingle :
          deletedWeight x yPos yNeg temperature selfMask i0 (Sum.inl j) ≤
            ∑ i : Fin Nx,
              deletedWeight x yPos yNeg temperature selfMask i (Sum.inl j) :=
        Finset.single_le_sum
          (fun i _ => deletedWeight_nonneg x yPos yNeg temperature selfMask i (Sum.inl j))
          (Finset.mem_univ i0)
      exact le_trans (hfloor i0 (Sum.inl j)) (by
        simpa [deletedWeight, deletedFactor, i0] using hsingle)
  | inr j =>
      rcases hcol j with ⟨i0, hi0⟩
      have hsingle :
          deletedWeight x yPos yNeg temperature selfMask i0 (Sum.inr j) ≤
            ∑ i : Fin Nx,
              deletedWeight x yPos yNeg temperature selfMask i (Sum.inr j) :=
        Finset.single_le_sum
          (fun i _ => deletedWeight_nonneg x yPos yNeg temperature selfMask i (Sum.inr j))
          (Finset.mem_univ i0)
      exact le_trans (hfloor i0 (Sum.inr j)) (by
        simpa [deletedWeight, deletedFactor, hi0] using hsingle)

theorem maskedMassProduct_sub_deletedMassProduct_le (hτ : ValidTemperature temperature)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    maskedRowMass x yPos yNeg temperature selfMask i *
        maskedColumnMass x yPos yNeg temperature selfMask s -
      deletedRowMass x yPos yNeg temperature selfMask i *
        deletedColumnMass x yPos yNeg temperature selfMask s ≤
      3 * ((Npos + Nneg : ℕ) : ℝ) * (Nx : ℝ) *
        maskPenaltyFactor temperature := by
  set δ := maskPenaltyFactor temperature
  set M : ℝ := ((Npos + Nneg : ℕ) : ℝ)
  set A : ℝ := (Nx : ℝ)
  set rv := deletedRowMass x yPos yNeg temperature selfMask i
  set rw := maskedRowMass x yPos yNeg temperature selfMask i
  set cv := deletedColumnMass x yPos yNeg temperature selfMask s
  set cw := maskedColumnMass x yPos yNeg temperature selfMask s
  have hδ0 : 0 ≤ δ := by exact (maskPenaltyFactor_pos temperature).le
  have hδ1 : δ ≤ 1 := by exact maskPenaltyFactor_le_one temperature hτ
  have hM0 : 0 ≤ M := by positivity
  have hA0 : 0 ≤ A := by positivity
  have hrv0 : 0 ≤ rv := by
    exact deletedRowMass_nonneg x yPos yNeg temperature selfMask i
  have hcv0 : 0 ≤ cv := by
    exact deletedColumnMass_nonneg x yPos yNeg temperature selfMask s
  have hdr0 : 0 ≤ rw - rv := by
    exact sub_nonneg.mpr (deletedRowMass_le_maskedRowMass x yPos yNeg temperature selfMask i)
  have hdc0 : 0 ≤ cw - cv := by
    exact sub_nonneg.mpr
      (deletedColumnMass_le_maskedColumnMass x yPos yNeg temperature selfMask s)
  have hrvM : rv ≤ M := by
    simpa [rv, M] using deletedRowMass_le_card x yPos yNeg temperature selfMask hτ i
  have hcvA : cv ≤ A := by
    simpa [cv, A] using deletedColumnMass_le_card x yPos yNeg temperature selfMask hτ s
  have hdr : rw - rv ≤ M * δ := by
    simpa [rw, rv, M, δ] using
      maskedRowMass_sub_deletedRowMass_le x yPos yNeg temperature selfMask hτ i
  have hdc : cw - cv ≤ A * δ := by
    simpa [cw, cv, A, δ] using
      maskedColumnMass_sub_deletedColumnMass_le x yPos yNeg temperature selfMask hτ s
  have hidentity : rw * cw - rv * cv =
      (rw - rv) * cv + rv * (cw - cv) + (rw - rv) * (cw - cv) := by ring
  rw [show maskedRowMass x yPos yNeg temperature selfMask i *
        maskedColumnMass x yPos yNeg temperature selfMask s -
      deletedRowMass x yPos yNeg temperature selfMask i *
        deletedColumnMass x yPos yNeg temperature selfMask s =
      rw * cw - rv * cv by rfl, hidentity]
  have h1 : (rw - rv) * cv ≤ (M * δ) * A :=
    mul_le_mul hdr hcvA hcv0 (mul_nonneg hM0 hδ0)
  have h2 : rv * (cw - cv) ≤ M * (A * δ) :=
    mul_le_mul hrvM hdc hdc0 hM0
  have h3 : (rw - rv) * (cw - cv) ≤ (M * δ) * (A * δ) :=
    mul_le_mul hdr hdc hdc0 (mul_nonneg hM0 hδ0)
  calc
    (rw - rv) * cv + rv * (cw - cv) + (rw - rv) * (cw - cv)
        ≤ (M * δ) * A + M * (A * δ) + (M * δ) * (A * δ) := by
          linarith
    _ ≤ 3 * M * A * δ := by
          have hMAδ0 : 0 ≤ M * A * δ := by positivity
          have hthird : (M * δ) * (A * δ) ≤ M * A * δ := by
            calc
              (M * δ) * (A * δ) = (M * A * δ) * δ := by ring
              _ ≤ (M * A * δ) * 1 := mul_le_mul_of_nonneg_left hδ1 hMAδ0
              _ = M * A * δ := by ring
          calc
            (M * δ) * A + M * (A * δ) + (M * δ) * (A * δ)
                ≤ (M * A * δ) + (M * A * δ) + (M * A * δ) := by
                  nlinarith
            _ = 3 * M * A * δ := by ring

/-- **Explicit masked-vs-deleted affinity comparison.**  Under a positive
temperature, a no-mask raw-weight floor `wmin`, and a guarantee that every
negative sample column keeps at least one unmasked anchor, each masked
bi-softmax affinity is within the deterministic `δ`-scale error of the deleted
affinity. -/
theorem algorithm2Affinity_sub_deletedAffinity_abs_le
    [Nonempty (Fin Nx)] (hτ : ValidTemperature temperature)
    {wmin : ℝ} (hNpos : 0 < Npos) (hwmin : 0 < wmin)
    (hfloor : ∀ i s, wmin ≤ noMaskWeight x yPos yNeg temperature i s)
    (hcol : HasUnmaskedNegativeColumn selfMask)
    (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    |algorithm2Affinity x yPos yNeg temperature selfMask i s -
        deletedAffinity x yPos yNeg temperature selfMask i s| ≤
      maskAffinityErrorBound Nx Npos Nneg wmin (maskPenaltyFactor temperature) := by
  rw [algorithm2Affinity_eq_maskedWeight_div_sqrt]
  unfold deletedAffinity
  set w := maskedWeight x yPos yNeg temperature selfMask i s
  set v := deletedWeight x yPos yNeg temperature selfMask i s
  set rw := maskedRowMass x yPos yNeg temperature selfMask i
  set rv := deletedRowMass x yPos yNeg temperature selfMask i
  set cw := maskedColumnMass x yPos yNeg temperature selfMask s
  set cv := deletedColumnMass x yPos yNeg temperature selfMask s
  set r0 : ℝ := (Npos : ℝ) * wmin
  set c0 : ℝ := wmin
  set δ := maskPenaltyFactor temperature
  set K : ℝ := 3 * ((Npos + Nneg : ℕ) : ℝ) * (Nx : ℝ) * δ
  have hv : 0 ≤ v := by exact deletedWeight_nonneg x yPos yNeg temperature selfMask i s
  have hvw : v ≤ w := by exact deletedWeight_le_maskedWeight x yPos yNeg temperature selfMask i s
  have hr0 : 0 < r0 := by
    exact mul_pos (Nat.cast_pos.mpr hNpos) hwmin
  have hc0 : 0 < c0 := by exact hwmin
  have hr : r0 ≤ rv := by
    simpa [r0, rv] using
      deletedRowMass_lower_of_noMaskWeight_floor x yPos yNeg temperature selfMask hfloor i
  have hc : c0 ≤ cv := by
    simpa [c0, cv] using
      deletedColumnMass_lower_of_noMaskWeight_floor x yPos yNeg temperature selfMask
        hfloor hcol s
  have hrw : rv ≤ rw := by
    exact deletedRowMass_le_maskedRowMass x yPos yNeg temperature selfMask i
  have hcw : cv ≤ cw := by
    exact deletedColumnMass_le_maskedColumnMass x yPos yNeg temperature selfMask s
  have hvr : v ≤ rv := by
    exact deletedWeight_le_deletedRowMass_of_term x yPos yNeg temperature selfMask i s
  have hvc : v ≤ cv := by
    exact deletedWeight_le_deletedColumnMass_of_term x yPos yNeg temperature selfMask i s
  have hmain := div_sqrt_mass_perturbation hv hvw hr0 hc0 hr hc hrw hcw hvr hvc
  have hdenpos : 0 < Real.sqrt (r0 * c0) := Real.sqrt_pos.mpr (mul_pos hr0 hc0)
  have hwdiff : w - v ≤ δ := by
    simpa [w, v, δ] using
      maskedWeight_sub_deletedWeight_le x yPos yNeg temperature selfMask hτ i s
  have hprod : rw * cw - rv * cv ≤ K := by
    simpa [rw, rv, cw, cv, K, δ] using
      maskedMassProduct_sub_deletedMassProduct_le x yPos yNeg temperature selfMask hτ i s
  have hK0 : 0 ≤ K := by
    have hδ0 : 0 ≤ δ := (maskPenaltyFactor_pos temperature).le
    positivity
  calc
    |w / Real.sqrt (rw * cw) - v / Real.sqrt (rv * cv)|
        ≤ (w - v) / Real.sqrt (r0 * c0) +
            Real.sqrt (rw * cw - rv * cv) / Real.sqrt (r0 * c0) := hmain
    _ ≤ δ / Real.sqrt (r0 * c0) + Real.sqrt K / Real.sqrt (r0 * c0) := by
        apply add_le_add
        · exact div_le_div_of_nonneg_right hwdiff hdenpos.le
        · exact div_le_div_of_nonneg_right (Real.sqrt_le_sqrt hprod) hdenpos.le
    _ = maskAffinityErrorBound Nx Npos Nneg wmin (maskPenaltyFactor temperature) := by
        simp [maskAffinityErrorBound, r0, c0, K, δ]

end MassBounds

/-! ## Drift stability from an affinity perturbation bound -/

section DriftPerturbation

variable {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable {Nx Npos Nneg : ℕ}
variable (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
variable (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)

omit [NormedSpace ℝ E] in
theorem deletedAffinity_nonneg (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    0 ≤ deletedAffinity x yPos yNeg temperature selfMask i s := by
  unfold deletedAffinity
  exact div_nonneg (deletedWeight_nonneg x yPos yNeg temperature selfMask i s)
    (Real.sqrt_nonneg _)

omit [NormedSpace ℝ E] in
theorem deletedWeight_le_deletedRowMass (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedWeight x yPos yNeg temperature selfMask i s ≤
      deletedRowMass x yPos yNeg temperature selfMask i := by
  unfold deletedRowMass
  exact Finset.single_le_sum
    (fun t _ => deletedWeight_nonneg x yPos yNeg temperature selfMask i t)
    (Finset.mem_univ s)

omit [NormedSpace ℝ E] in
theorem deletedWeight_le_deletedColumnMass (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedWeight x yPos yNeg temperature selfMask i s ≤
      deletedColumnMass x yPos yNeg temperature selfMask s := by
  unfold deletedColumnMass
  exact Finset.single_le_sum
    (fun r _ => deletedWeight_nonneg x yPos yNeg temperature selfMask r s)
    (Finset.mem_univ i)

omit [NormedSpace ℝ E] in
theorem deletedAffinity_le_one_of_mass_pos (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg)
    (hrow : 0 < deletedRowMass x yPos yNeg temperature selfMask i)
    (hcol : 0 < deletedColumnMass x yPos yNeg temperature selfMask s) :
    deletedAffinity x yPos yNeg temperature selfMask i s ≤ 1 := by
  unfold deletedAffinity
  set v := deletedWeight x yPos yNeg temperature selfMask i s
  set r := deletedRowMass x yPos yNeg temperature selfMask i
  set c := deletedColumnMass x yPos yNeg temperature selfMask s
  have hv : 0 ≤ v := by
    exact deletedWeight_nonneg x yPos yNeg temperature selfMask i s
  have hvr : v ≤ r := by
    exact deletedWeight_le_deletedRowMass x yPos yNeg temperature selfMask i s
  have hvc : v ≤ c := by
    exact deletedWeight_le_deletedColumnMass x yPos yNeg temperature selfMask i s
  have hs : 0 < Real.sqrt (r * c) := Real.sqrt_pos.mpr (mul_pos hrow hcol)
  have hv_sqrt : v ≤ Real.sqrt (r * c) := by
    have hsq : v ^ 2 ≤ r * c := by nlinarith
    calc v = Real.sqrt (v ^ 2) := (Real.sqrt_sq hv).symm
      _ ≤ Real.sqrt (r * c) := Real.sqrt_le_sqrt hsq
  rw [div_le_one hs]
  exact hv_sqrt

/-- If every bi-softmax affinity is within `η` of the corresponding deleted
affinity, then the pair-sum drift changes by at most
`4 * Npos * Nneg * R0 * η`.  This is the deterministic downstream step in the
self-mask perturbation argument. -/
theorem algorithm2Drift_sub_deletedDrift_norm_le_of_affinity
    (i : Fin Nx) {R0 η : ℝ} (_hR0 : 0 ≤ R0) (hη : 0 ≤ η)
    (hyPos : ∀ j, ‖yPos j‖ ≤ R0) (hyNeg : ∀ j, ‖yNeg j‖ ≤ R0)
    (hdel_le_one : ∀ s : Algorithm2SampleIndex Npos Nneg,
      deletedAffinity x yPos yNeg temperature selfMask i s ≤ 1)
    (haff : ∀ s : Algorithm2SampleIndex Npos Nneg,
      |algorithm2Affinity x yPos yNeg temperature selfMask i s -
        deletedAffinity x yPos yNeg temperature selfMask i s| ≤ η) :
    ‖algorithm2Drift x yPos yNeg temperature selfMask i -
        deletedDrift x yPos yNeg temperature selfMask i‖ ≤
      4 * Npos * Nneg * R0 * η := by
  rw [algorithm2Drift_eq_affinityPairSum]
  unfold deletedDrift
  rw [← Finset.sum_sub_distrib]
  refine le_trans (norm_sum_le _ _) ?_
  calc
    (∑ j : Fin Npos,
        ‖(∑ l : Fin Nneg,
          (algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) *
              algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l)) •
              (yPos j - yNeg l)) -
          (∑ l : Fin Nneg,
            (deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j) *
              deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l)) •
              (yPos j - yNeg l))‖)
      ≤ ∑ j : Fin Npos, ∑ l : Fin Nneg,
          4 * R0 * η := by
        apply Finset.sum_le_sum
        intro j _
        rw [← Finset.sum_sub_distrib]
        refine le_trans (norm_sum_le _ _) ?_
        apply Finset.sum_le_sum
        intro l _
        rw [← sub_smul, norm_smul, Real.norm_eq_abs]
        have hdist : ‖yPos j - yNeg l‖ ≤ 2 * R0 := by
          calc ‖yPos j - yNeg l‖ ≤ ‖yPos j‖ + ‖yNeg l‖ := by
                simpa [sub_eq_add_neg] using norm_add_le (yPos j) (-yNeg l)
            _ ≤ R0 + R0 := add_le_add (hyPos j) (hyNeg l)
            _ = 2 * R0 := by ring
        have hprod :
            |algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) *
                algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l) -
              deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j) *
                deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l)| ≤
              2 * η := by
          have hmul := abs_mul_sub_mul_le
            (a := algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j))
            (b := algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l))
            (c := deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j))
            (d := deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l))
            (algorithm2Affinity_nonneg x yPos yNeg temperature selfMask i (Sum.inr l))
            (algorithm2Affinity_le_one x yPos yNeg temperature selfMask i (Sum.inr l))
            (deletedAffinity_nonneg x yPos yNeg temperature selfMask i (Sum.inl j))
            (hdel_le_one (Sum.inl j))
          calc
            |algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) *
                algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l) -
              deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j) *
                deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l)|
                ≤
              |algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) -
                deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j)| +
              |algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l) -
                deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l)| := hmul
            _ ≤ η + η := add_le_add (haff (Sum.inl j)) (haff (Sum.inr l))
            _ = 2 * η := by ring
        calc
          |algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) *
                algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l) -
              deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j) *
                deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l)| *
              ‖yPos j - yNeg l‖
              ≤ (2 * η) * (2 * R0) :=
                mul_le_mul hprod hdist (norm_nonneg _) (by positivity)
          _ = 4 * R0 * η := by ring
    _ = 4 * Npos * Nneg * R0 * η := by
        rw [Finset.sum_const, Finset.sum_const, Finset.card_univ,
          Finset.card_univ, Fintype.card_fin, Fintype.card_fin,
          nsmul_eq_mul, nsmul_eq_mul]
        ring

end DriftPerturbation

/-! ## Fully explicit self-mask perturbation bound -/

section ExplicitDriftPerturbation

variable {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable {Nx Npos Nneg : ℕ} [Nonempty (Fin Nx)]
variable (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
variable (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)

omit [Nonempty (Fin Nx)] in
theorem maskAffinityErrorBound_nonneg
    {wmin δ : ℝ} (hNpos : 0 < Npos) (hwmin : 0 < wmin) (hδ : 0 ≤ δ) :
    0 ≤ maskAffinityErrorBound Nx Npos Nneg wmin δ := by
  unfold maskAffinityErrorBound
  have hden : 0 < Real.sqrt (((Npos : ℝ) * wmin) * wmin) := by
    exact Real.sqrt_pos.mpr (mul_pos (mul_pos (Nat.cast_pos.mpr hNpos) hwmin) hwmin)
  have hK : 0 ≤ 3 * ((Npos + Nneg : ℕ) : ℝ) * (Nx : ℝ) * δ := by positivity
  positivity

/-- **Self-mask perturbation bound.**  The Algorithm-2 estimator with an
arbitrary Boolean self-mask is close to the estimator with the masked entries
hard-deleted.  The comparison is deterministic and explicit in
`δ = exp(-1000000/temperature)`.

This theorem deliberately compares masked vs. deleted, not masked vs. the full
no-mask estimator on the same samples. -/
theorem algorithm2Drift_sub_deletedDrift_norm_le
    (hτ : ValidTemperature temperature)
    {wmin R0 : ℝ} (hNpos : 0 < Npos) (hwmin : 0 < wmin) (hR0 : 0 ≤ R0)
    (hfloor : ∀ i s, wmin ≤ noMaskWeight x yPos yNeg temperature i s)
    (hcol : HasUnmaskedNegativeColumn selfMask)
    (hyPos : ∀ j, ‖yPos j‖ ≤ R0) (hyNeg : ∀ j, ‖yNeg j‖ ≤ R0)
    (i : Fin Nx) :
    ‖algorithm2Drift x yPos yNeg temperature selfMask i -
        deletedDrift x yPos yNeg temperature selfMask i‖ ≤
      4 * Npos * Nneg * R0 *
        maskAffinityErrorBound Nx Npos Nneg wmin (maskPenaltyFactor temperature) := by
  let η := maskAffinityErrorBound Nx Npos Nneg wmin (maskPenaltyFactor temperature)
  have hη : 0 ≤ η := by
    exact maskAffinityErrorBound_nonneg (Nx := Nx) (Npos := Npos) (Nneg := Nneg)
      hNpos hwmin (maskPenaltyFactor_pos temperature).le
  have hrowpos : 0 < deletedRowMass x yPos yNeg temperature selfMask i := by
    have hrow := deletedRowMass_lower_of_noMaskWeight_floor
      x yPos yNeg temperature selfMask hfloor i
    exact lt_of_lt_of_le (mul_pos (Nat.cast_pos.mpr hNpos) hwmin) hrow
  have hcolpos : ∀ s : Algorithm2SampleIndex Npos Nneg,
      0 < deletedColumnMass x yPos yNeg temperature selfMask s := by
    intro s
    have hc := deletedColumnMass_lower_of_noMaskWeight_floor
      x yPos yNeg temperature selfMask hfloor hcol s
    exact lt_of_lt_of_le hwmin hc
  have hdel_le_one : ∀ s : Algorithm2SampleIndex Npos Nneg,
      deletedAffinity x yPos yNeg temperature selfMask i s ≤ 1 := by
    intro s
    exact deletedAffinity_le_one_of_mass_pos x yPos yNeg temperature selfMask
      i s hrowpos (hcolpos s)
  have haff : ∀ s : Algorithm2SampleIndex Npos Nneg,
      |algorithm2Affinity x yPos yNeg temperature selfMask i s -
        deletedAffinity x yPos yNeg temperature selfMask i s| ≤ η := by
    intro s
    exact algorithm2Affinity_sub_deletedAffinity_abs_le
      x yPos yNeg temperature selfMask hτ hNpos hwmin hfloor hcol i s
  simpa [η] using
    algorithm2Drift_sub_deletedDrift_norm_le_of_affinity
      x yPos yNeg temperature selfMask i hR0 hη hyPos hyNeg hdel_le_one haff

/-- Eye-mask specialization of the deterministic self-mask perturbation. -/
theorem algorithm2Drift_sub_deletedDrift_norm_le_eyeMask
    {N : ℕ} (hN : 2 ≤ N)
    (x : Fin N → E) (yPos : Fin Npos → E) (yNeg : Fin N → E)
    (temperature : ℝ) (hτ : ValidTemperature temperature)
    {wmin R0 : ℝ} (hNpos : 0 < Npos) (hwmin : 0 < wmin) (hR0 : 0 ≤ R0)
    (hfloor : ∀ i s, wmin ≤ noMaskWeight x yPos yNeg temperature i s)
    (hyPos : ∀ j, ‖yPos j‖ ≤ R0) (hyNeg : ∀ j, ‖yNeg j‖ ≤ R0)
    (i : Fin N) :
    ‖algorithm2Drift x yPos yNeg temperature eyeMask i -
        deletedDrift x yPos yNeg temperature eyeMask i‖ ≤
      4 * Npos * N * R0 *
        maskAffinityErrorBound N Npos N wmin (maskPenaltyFactor temperature) := by
  haveI : Nonempty (Fin N) := ⟨i⟩
  exact algorithm2Drift_sub_deletedDrift_norm_le
    x yPos yNeg temperature eyeMask hτ hNpos hwmin hR0 hfloor
    (eyeMask_column_unmasked hN) hyPos hyNeg i

end ExplicitDriftPerturbation

end Algorithm2
end DriftingIdentifiability
