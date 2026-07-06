import DriftingIdentifiability.TrustedBoundary

/-!
# Structure and boundedness of Algorithm 2's estimator (Objective 4)

`FiniteSampleBridge.lean` supplies the *estimator-agnostic* finite-sample bridge:
any estimator `Vhat` of the population probe-drift transfers its mean-squared
error to identifiability accuracy through the `2B/c` conditioning constant.  The
remaining Objective-4 work is *estimator-specific*: understanding the actual
minibatch field `Paper.algorithm2Drift` (Algorithm 2's bi-softmax `compute_V`).

This module proves the properties of that estimator that are available without
any distributional/limit hypothesis, staying strictly inside the trust
boundary (these are theorems *about* the reviewed `Paper` definitions; they add
no axioms):

* **Softmax/affinity range.**  The row and column softmax affinities, and their
  geometric mean, take values in `[0,1]`; the sample weights are nonnegative and
  bounded by the opposite-sample count.
* **Exact structural form.**  `algorithm2Drift` equals the affinity-weighted
  pairwise attraction minus repulsion
  `∑_{j,l} A(i,+j) A(i,-l) • (yPos j - yNeg l)`, the finite-sample analogue of
  the mean-shift interaction kernel `Paper.meanShiftInteractionKernel`
  `(k x y⁺)(k x y⁻) • (y⁺ - y⁻)`, with the affinities `A` in the role of the
  kernel `k`.  It also equals the mass-scaled centroid difference
  `Q • (Σ A(+) yPos) - P • (Σ A(-) yNeg)`.
* **Boundedness.**  If every sample lies in a ball of radius `R`, then
  `‖algorithm2Drift i‖ ≤ 2 · Npos · Nneg · R`.  This is exactly the bounded-range
  hypothesis a bounded-differences concentration inequality consumes, so it is
  the concrete input the finite-sample bridge needs from this estimator.
* **Matched-batch cancellation.**  When the positive and negative samples
  coincide and no self-mask is applied, `algorithm2Drift = 0`.  This is the
  sample-level analogue of `Paper.equation_17_matched_batch_drift_zero`
  (matched laws ⟹ zero batch drift) and of the population identity
  `p = q ⟹ V = 0`; it is the *safe* direction and assumes nothing about
  identifiability.

What is deliberately **not** proved here is the quantitative bias/consistency of
`algorithm2Drift` against the ideal population field (its expectation as the
temperature and sample counts vary).  That is a genuine estimator-analysis
research question; the bridge reaches it only through the estimator's mean
squared error, which these structural facts constrain but do not compute.
-/

open scoped BigOperators
open MeasureTheory

namespace DriftingIdentifiability
namespace Algorithm2

open Paper

universe u

/-! ## Softmax basics -/

section Softmax

variable {ι : Type*} [Fintype ι] (logit : ι → ℝ)

/-- A softmax weight is nonnegative. -/
theorem finiteSoftmax_nonneg (i : ι) : 0 ≤ finiteSoftmax logit i := by
  unfold finiteSoftmax
  exact div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun j _ => (Real.exp_pos _).le)

/-- A softmax weight is at most one (the numerator is one of the summands). -/
theorem finiteSoftmax_le_one (i : ι) : finiteSoftmax logit i ≤ 1 := by
  unfold finiteSoftmax
  rw [div_le_one (Finset.sum_pos (fun j _ => Real.exp_pos _) ⟨i, Finset.mem_univ i⟩)]
  exact Finset.single_le_sum (fun j _ => (Real.exp_pos _).le) (Finset.mem_univ i)

/-- The softmax weights over a nonempty index sum to one. -/
theorem finiteSoftmax_sum_eq_one [Nonempty ι] : ∑ i, finiteSoftmax logit i = 1 := by
  unfold finiteSoftmax
  rw [← Finset.sum_div, div_self]
  exact (Finset.sum_pos (fun j _ => Real.exp_pos _) Finset.univ_nonempty).ne'

end Softmax

/-! ## Affinity and weight range -/

section Affinity

variable {E : Type u} [NormedAddCommGroup E]
variable {Nx Npos Nneg : ℕ}
variable (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
variable (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)

theorem algorithm2RowAffinity_nonneg (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    0 ≤ algorithm2RowAffinity x yPos yNeg temperature selfMask i s := by
  unfold algorithm2RowAffinity; exact finiteSoftmax_nonneg _ _

theorem algorithm2RowAffinity_le_one (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    algorithm2RowAffinity x yPos yNeg temperature selfMask i s ≤ 1 := by
  unfold algorithm2RowAffinity; exact finiteSoftmax_le_one _ _

theorem algorithm2ColumnAffinity_nonneg (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    0 ≤ algorithm2ColumnAffinity x yPos yNeg temperature selfMask i s := by
  unfold algorithm2ColumnAffinity; exact finiteSoftmax_nonneg _ _

theorem algorithm2ColumnAffinity_le_one (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    algorithm2ColumnAffinity x yPos yNeg temperature selfMask i s ≤ 1 := by
  unfold algorithm2ColumnAffinity; exact finiteSoftmax_le_one _ _

/-- The geometric-mean affinity is nonnegative. -/
theorem algorithm2Affinity_nonneg (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    0 ≤ algorithm2Affinity x yPos yNeg temperature selfMask i s := by
  unfold algorithm2Affinity; exact Real.sqrt_nonneg _

/-- The geometric-mean affinity is at most one. -/
theorem algorithm2Affinity_le_one (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    algorithm2Affinity x yPos yNeg temperature selfMask i s ≤ 1 := by
  unfold algorithm2Affinity
  rw [show (1 : ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
  apply Real.sqrt_le_sqrt
  nlinarith [algorithm2RowAffinity_nonneg x yPos yNeg temperature selfMask i s,
    algorithm2RowAffinity_le_one x yPos yNeg temperature selfMask i s,
    algorithm2ColumnAffinity_nonneg x yPos yNeg temperature selfMask i s,
    algorithm2ColumnAffinity_le_one x yPos yNeg temperature selfMask i s]

theorem algorithm2PositiveWeight_nonneg (i : Fin Nx) (j : Fin Npos) :
    0 ≤ algorithm2PositiveWeight x yPos yNeg temperature selfMask i j := by
  unfold algorithm2PositiveWeight
  exact mul_nonneg (algorithm2Affinity_nonneg x yPos yNeg temperature selfMask i (Sum.inl j))
    (Finset.sum_nonneg fun l _ =>
      algorithm2Affinity_nonneg x yPos yNeg temperature selfMask i (Sum.inr l))

/-- Each positive weight is bounded by the number of negative samples. -/
theorem algorithm2PositiveWeight_le (i : Fin Nx) (j : Fin Npos) :
    algorithm2PositiveWeight x yPos yNeg temperature selfMask i j ≤ (Nneg : ℝ) := by
  unfold algorithm2PositiveWeight
  have hsum : ∑ l : Fin Nneg, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l)
      ≤ (Nneg : ℝ) := by
    calc ∑ l : Fin Nneg, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l)
        ≤ ∑ _l : Fin Nneg, (1 : ℝ) :=
          Finset.sum_le_sum fun l _ =>
            algorithm2Affinity_le_one x yPos yNeg temperature selfMask i (Sum.inr l)
      _ = (Nneg : ℝ) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  have hsum0 :
      0 ≤ ∑ l : Fin Nneg, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l) :=
    Finset.sum_nonneg fun l _ =>
      algorithm2Affinity_nonneg x yPos yNeg temperature selfMask i (Sum.inr l)
  calc algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) *
        ∑ l : Fin Nneg, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l)
      ≤ 1 * (Nneg : ℝ) :=
        mul_le_mul (algorithm2Affinity_le_one x yPos yNeg temperature selfMask i (Sum.inl j))
          hsum hsum0 (by norm_num)
    _ = (Nneg : ℝ) := one_mul _

theorem algorithm2NegativeWeight_nonneg (i : Fin Nx) (j : Fin Nneg) :
    0 ≤ algorithm2NegativeWeight x yPos yNeg temperature selfMask i j := by
  unfold algorithm2NegativeWeight
  exact mul_nonneg (algorithm2Affinity_nonneg x yPos yNeg temperature selfMask i (Sum.inr j))
    (Finset.sum_nonneg fun l _ =>
      algorithm2Affinity_nonneg x yPos yNeg temperature selfMask i (Sum.inl l))

/-- Each negative weight is bounded by the number of positive samples. -/
theorem algorithm2NegativeWeight_le (i : Fin Nx) (j : Fin Nneg) :
    algorithm2NegativeWeight x yPos yNeg temperature selfMask i j ≤ (Npos : ℝ) := by
  unfold algorithm2NegativeWeight
  have hsum : ∑ l : Fin Npos, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl l)
      ≤ (Npos : ℝ) := by
    calc ∑ l : Fin Npos, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl l)
        ≤ ∑ _l : Fin Npos, (1 : ℝ) :=
          Finset.sum_le_sum fun l _ =>
            algorithm2Affinity_le_one x yPos yNeg temperature selfMask i (Sum.inl l)
      _ = (Npos : ℝ) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]
  have hsum0 :
      0 ≤ ∑ l : Fin Npos, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl l) :=
    Finset.sum_nonneg fun l _ =>
      algorithm2Affinity_nonneg x yPos yNeg temperature selfMask i (Sum.inl l)
  calc algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr j) *
        ∑ l : Fin Npos, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl l)
      ≤ 1 * (Npos : ℝ) :=
        mul_le_mul (algorithm2Affinity_le_one x yPos yNeg temperature selfMask i (Sum.inr j))
          hsum hsum0 (by norm_num)
    _ = (Npos : ℝ) := one_mul _

end Affinity

/-! ## Exact structural form -/

section Structure

variable {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable {Nx Npos Nneg : ℕ}
variable (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
variable (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)

/-- **Exact structural form of `compute_V`.**  Algorithm 2's returned drift is
the affinity-weighted pairwise attraction minus repulsion, summed over all
positive/negative sample pairs.  With the geometric-mean affinities `A` in the
role of the kernel `k`, each summand
`A(i,+j) A(i,-l) • (yPos j - yNeg l)` is exactly the finite-sample instance of
the mean-shift interaction kernel `Paper.meanShiftInteractionKernel`
`(k x y⁺)(k x y⁻) • (y⁺ - y⁻)`.  No positivity, temperature, or distributional
hypothesis is needed. -/
theorem algorithm2Drift_eq_affinityPairSum (i : Fin Nx) :
    algorithm2Drift x yPos yNeg temperature selfMask i
      = ∑ j : Fin Npos, ∑ l : Fin Nneg,
          (algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) *
            algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l))
          • (yPos j - yNeg l) := by
  have hpos :
      (∑ j : Fin Npos, algorithm2PositiveWeight x yPos yNeg temperature selfMask i j • yPos j)
        = ∑ j : Fin Npos, ∑ l : Fin Nneg,
            (algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) *
              algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l)) • yPos j := by
    apply Finset.sum_congr rfl
    intro j _
    unfold algorithm2PositiveWeight
    rw [Finset.mul_sum, Finset.sum_smul]
  have hneg :
      (∑ j : Fin Nneg, algorithm2NegativeWeight x yPos yNeg temperature selfMask i j • yNeg j)
        = ∑ j : Fin Npos, ∑ l : Fin Nneg,
            (algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) *
              algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l)) • yNeg l := by
    have expand :
        (∑ j : Fin Nneg, algorithm2NegativeWeight x yPos yNeg temperature selfMask i j • yNeg j)
          = ∑ j : Fin Nneg, ∑ l : Fin Npos,
              (algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl l) *
                algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr j)) • yNeg j := by
      apply Finset.sum_congr rfl
      intro j _
      unfold algorithm2NegativeWeight
      rw [Finset.mul_sum, Finset.sum_smul]
      apply Finset.sum_congr rfl
      intro l _
      rw [mul_comm]
    rw [expand, Finset.sum_comm]
  unfold algorithm2Drift
  rw [hpos, hneg, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  rw [← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro l _
  rw [smul_sub]

/-- **Mass-scaled centroid form.**  Equivalently, the drift is the total
negative-affinity mass times the affinity-weighted positive sample sum, minus
the total positive-affinity mass times the affinity-weighted negative sample
sum.  This is the discrete "attraction by positives minus repulsion by
negatives" reading of `compute_V`. -/
theorem algorithm2Drift_eq_massScaledCentroid (i : Fin Nx) :
    algorithm2Drift x yPos yNeg temperature selfMask i
      = (∑ l : Fin Nneg, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l)) •
          (∑ j : Fin Npos,
            algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j) • yPos j)
        - (∑ l : Fin Npos, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl l)) •
          (∑ j : Fin Nneg,
            algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr j) • yNeg j) := by
  unfold algorithm2Drift algorithm2PositiveWeight algorithm2NegativeWeight
  congr 1
  · rw [Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [smul_smul, mul_comm]
  · rw [Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [smul_smul, mul_comm]

/-- **Boundedness of `compute_V`.**  If every positive and negative sample lies
in the ball of radius `R`, the estimator is bounded by `2 · Npos · Nneg · R`,
uniformly in the temperature and self-mask.  This is the bounded-range
hypothesis that a bounded-differences (McDiarmid-type) concentration inequality
requires, i.e. the concrete estimator input to the finite-sample bridge. -/
theorem algorithm2Drift_norm_le (i : Fin Nx) {R : ℝ}
    (hyPos : ∀ j, ‖yPos j‖ ≤ R) (hyNeg : ∀ j, ‖yNeg j‖ ≤ R) :
    ‖algorithm2Drift x yPos yNeg temperature selfMask i‖ ≤ 2 * Npos * Nneg * R := by
  unfold algorithm2Drift
  refine le_trans (norm_sub_le _ _) ?_
  have hpos :
      ‖∑ j, algorithm2PositiveWeight x yPos yNeg temperature selfMask i j • yPos j‖
        ≤ (Npos : ℝ) * Nneg * R := by
    refine le_trans (norm_sum_le _ _) ?_
    calc ∑ j, ‖algorithm2PositiveWeight x yPos yNeg temperature selfMask i j • yPos j‖
        ≤ ∑ _j : Fin Npos, ((Nneg : ℝ) * R) := by
          apply Finset.sum_le_sum
          intro j _
          rw [norm_smul, Real.norm_eq_abs,
            abs_of_nonneg (algorithm2PositiveWeight_nonneg x yPos yNeg temperature selfMask i j)]
          exact mul_le_mul (algorithm2PositiveWeight_le x yPos yNeg temperature selfMask i j)
            (hyPos j) (norm_nonneg _) (by positivity)
      _ = (Npos : ℝ) * Nneg * R := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  have hneg :
      ‖∑ j, algorithm2NegativeWeight x yPos yNeg temperature selfMask i j • yNeg j‖
        ≤ (Npos : ℝ) * Nneg * R := by
    refine le_trans (norm_sum_le _ _) ?_
    calc ∑ j, ‖algorithm2NegativeWeight x yPos yNeg temperature selfMask i j • yNeg j‖
        ≤ ∑ _j : Fin Nneg, ((Npos : ℝ) * R) := by
          apply Finset.sum_le_sum
          intro j _
          rw [norm_smul, Real.norm_eq_abs,
            abs_of_nonneg (algorithm2NegativeWeight_nonneg x yPos yNeg temperature selfMask i j)]
          exact mul_le_mul (algorithm2NegativeWeight_le x yPos yNeg temperature selfMask i j)
            (hyNeg j) (norm_nonneg _) (by positivity)
      _ = (Npos : ℝ) * Nneg * R := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring
  calc ‖∑ j, algorithm2PositiveWeight x yPos yNeg temperature selfMask i j • yPos j‖ +
        ‖∑ j, algorithm2NegativeWeight x yPos yNeg temperature selfMask i j • yNeg j‖
      ≤ (Npos : ℝ) * Nneg * R + (Npos : ℝ) * Nneg * R := add_le_add hpos hneg
    _ = 2 * Npos * Nneg * R := by ring

/-- **Affinity-mass boundedness (adaptive convex-hull bound).**  A tighter,
data-adaptive magnitude bound than `algorithm2Drift_norm_le`.  Using the
mass-scaled centroid form and that each softmax-weighted sample sum lies in the
mass-scaled convex hull of the samples, `‖algorithm2Drift i‖ ≤ 2 · P · Q · R`,
where `P = Σⱼ A(i,+j)` and `Q = Σₗ A(i,−l)` are the total positive and negative
affinity masses (each at most `Npos`, `Nneg`, so this refines the crude bound).
It shrinks as the softmax concentrates its mass on few samples. -/
theorem algorithm2Drift_norm_le_affinityMass (i : Fin Nx) {R : ℝ}
    (hyPos : ∀ j, ‖yPos j‖ ≤ R) (hyNeg : ∀ j, ‖yNeg j‖ ≤ R) :
    ‖algorithm2Drift x yPos yNeg temperature selfMask i‖ ≤
      2 * (∑ j : Fin Npos, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inl j))
        * (∑ l : Fin Nneg, algorithm2Affinity x yPos yNeg temperature selfMask i (Sum.inr l))
        * R := by
  rw [algorithm2Drift_eq_massScaledCentroid]
  set A := algorithm2Affinity x yPos yNeg temperature selfMask i with hA
  have hAnn : ∀ s, 0 ≤ A s := fun s =>
    algorithm2Affinity_nonneg x yPos yNeg temperature selfMask i s
  have hPnn : 0 ≤ ∑ j : Fin Npos, A (Sum.inl j) := Finset.sum_nonneg fun j _ => hAnn _
  have hQnn : 0 ≤ ∑ l : Fin Nneg, A (Sum.inr l) := Finset.sum_nonneg fun l _ => hAnn _
  have hSpos : ‖∑ j : Fin Npos, A (Sum.inl j) • yPos j‖
      ≤ (∑ j : Fin Npos, A (Sum.inl j)) * R := by
    refine le_trans (norm_sum_le _ _) ?_
    rw [Finset.sum_mul]
    apply Finset.sum_le_sum
    intro j _
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hAnn _)]
    exact mul_le_mul_of_nonneg_left (hyPos j) (hAnn _)
  have hSneg : ‖∑ j : Fin Nneg, A (Sum.inr j) • yNeg j‖
      ≤ (∑ j : Fin Nneg, A (Sum.inr j)) * R := by
    refine le_trans (norm_sum_le _ _) ?_
    rw [Finset.sum_mul]
    apply Finset.sum_le_sum
    intro j _
    rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hAnn _)]
    exact mul_le_mul_of_nonneg_left (hyNeg j) (hAnn _)
  refine le_trans (norm_sub_le _ _) ?_
  rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs, abs_of_nonneg hQnn,
    abs_of_nonneg hPnn]
  calc (∑ l : Fin Nneg, A (Sum.inr l)) * ‖∑ j : Fin Npos, A (Sum.inl j) • yPos j‖
        + (∑ l : Fin Npos, A (Sum.inl l)) * ‖∑ j : Fin Nneg, A (Sum.inr j) • yNeg j‖
      ≤ (∑ l : Fin Nneg, A (Sum.inr l)) * ((∑ j : Fin Npos, A (Sum.inl j)) * R)
        + (∑ l : Fin Npos, A (Sum.inl l)) * ((∑ j : Fin Nneg, A (Sum.inr j)) * R) :=
        add_le_add (mul_le_mul_of_nonneg_left hSpos hQnn)
          (mul_le_mul_of_nonneg_left hSneg hPnn)
    _ = 2 * (∑ j : Fin Npos, A (Sum.inl j)) * (∑ l : Fin Nneg, A (Sum.inr l)) * R := by ring

end Structure

/-! ## Matched-batch cancellation -/

section Matched

variable {E : Type u} [NormedAddCommGroup E]
variable {Nx n : ℕ} (x : Fin Nx → E) (y : Fin n → E) (temperature : ℝ)

/-- With coinciding positive/negative samples and no self-mask, the logits of a
positive and its negative copy agree. -/
theorem algorithm2Logit_matched_eq (i : Fin Nx) (j : Fin n) :
    algorithm2Logit x y y temperature (fun _ _ => false) i (Sum.inl j)
      = algorithm2Logit x y y temperature (fun _ _ => false) i (Sum.inr j) := by
  simp [algorithm2Logit]

theorem algorithm2RowAffinity_matched_eq (i : Fin Nx) (j : Fin n) :
    algorithm2RowAffinity x y y temperature (fun _ _ => false) i (Sum.inl j)
      = algorithm2RowAffinity x y y temperature (fun _ _ => false) i (Sum.inr j) := by
  unfold algorithm2RowAffinity finiteSoftmax
  rw [algorithm2Logit_matched_eq x y temperature i j]

theorem algorithm2ColumnAffinity_matched_eq (i : Fin Nx) (j : Fin n) :
    algorithm2ColumnAffinity x y y temperature (fun _ _ => false) i (Sum.inl j)
      = algorithm2ColumnAffinity x y y temperature (fun _ _ => false) i (Sum.inr j) := by
  unfold algorithm2ColumnAffinity
  have hFG :
      (fun i' => algorithm2Logit x y y temperature (fun _ _ => false) i' (Sum.inl j))
        = (fun i' => algorithm2Logit x y y temperature (fun _ _ => false) i' (Sum.inr j)) := by
    funext i'
    exact algorithm2Logit_matched_eq x y temperature i' j
  rw [hFG]

theorem algorithm2Affinity_matched_eq (i : Fin Nx) (j : Fin n) :
    algorithm2Affinity x y y temperature (fun _ _ => false) i (Sum.inl j)
      = algorithm2Affinity x y y temperature (fun _ _ => false) i (Sum.inr j) := by
  unfold algorithm2Affinity
  rw [algorithm2RowAffinity_matched_eq x y temperature i j,
    algorithm2ColumnAffinity_matched_eq x y temperature i j]

/-- **Matched-batch cancellation (sample-level analogue of equation (17)).**
When the positive and negative samples coincide and no self-mask is applied,
Algorithm 2's estimator returns exactly zero.  This mirrors
`Paper.equation_17_matched_batch_drift_zero` (matched laws force zero batch
drift) and the population reverse implication `p = q ⟹ V = 0`; it is the safe
direction and assumes nothing about identifiability. -/
theorem algorithm2Drift_matched_zero [NormedSpace ℝ E] (i : Fin Nx) :
    algorithm2Drift x y y temperature (fun _ _ => false) i = 0 := by
  unfold algorithm2Drift
  have hweight : ∀ j : Fin n,
      algorithm2PositiveWeight x y y temperature (fun _ _ => false) i j
        = algorithm2NegativeWeight x y y temperature (fun _ _ => false) i j := by
    intro j
    unfold algorithm2PositiveWeight algorithm2NegativeWeight
    rw [algorithm2Affinity_matched_eq x y temperature i j]
    congr 1
    apply Finset.sum_congr rfl
    intro l _
    exact (algorithm2Affinity_matched_eq x y temperature i l).symm
  rw [sub_eq_zero]
  apply Finset.sum_congr rfl
  intro j _
  rw [hweight j]

end Matched

end Algorithm2
end DriftingIdentifiability
