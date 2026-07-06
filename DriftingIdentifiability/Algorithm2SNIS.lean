import DriftingIdentifiability.Algorithm2Estimator
import DriftingIdentifiability.SelfNormalizedConsistency

/-!
# Algorithm 2 as a self-normalized importance-sampling estimator

This module implements the deterministic half of the remaining Objective-4
route.  With fixed anchors and `selfMask = false`, Algorithm 2's geometric
bi-softmax affinity has a common row-normalization factor at each anchor.  That
factor cancels in the positive and negative centroids, leaving genuine
self-normalized centroids with the column-reweighted kernel

```text
w_i(y) = sqrt(k(x_i,y) * k(x_i,y) / sum_r k(x_r,y)).
```

This is the exact algebraic bridge to
`SelfNormalized.selfNormalized_meanSquare_le`.  The remaining statistical work
is to instantiate that generic theorem for an explicit minibatch sampling
model.  No new axiom is introduced here.
-/

open scoped BigOperators
open MeasureTheory ProbabilityTheory

namespace DriftingIdentifiability
namespace Algorithm2

open Paper

universe u

section ColumnReweighted

variable {E : Type u} [NormedAddCommGroup E]
variable {Nx Npos Nneg : ℕ}

/-- The positive/negative sample selected by a concatenated Algorithm-2 sample
index. -/
def algorithm2SampleValue (yPos : Fin Npos → E) (yNeg : Fin Nneg → E) :
    Algorithm2SampleIndex Npos Nneg → E
  | Sum.inl j => yPos j
  | Sum.inr j => yNeg j

/-- The unnormalized Algorithm-2 kernel corresponding exactly to the logits
`-‖x-y‖/temperature`. -/
noncomputable def algorithm2Kernel (temperature : ℝ) (x y : E) : ℝ :=
  Real.exp (-‖x - y‖ / temperature)

/-- Row kernel mass over the concatenated positive/negative sample axis. -/
noncomputable def algorithm2RowKernelMass
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) : ℝ :=
  ∑ s : Algorithm2SampleIndex Npos Nneg,
    algorithm2Kernel temperature (x i) (algorithm2SampleValue yPos yNeg s)

/-- Column kernel mass over the fixed anchor/probe axis. -/
noncomputable def algorithm2ColumnKernelMass
    (x : Fin Nx → E) (temperature : ℝ) (y : E) : ℝ :=
  ∑ r : Fin Nx, algorithm2Kernel temperature (x r) y

/-- The column-reweighted SNIS weight left after the row softmax factor cancels
inside Algorithm 2's centroids.  This is definitionally equivalent to
`k(x_i,y) / sqrt(sum_r k(x_r,y))` whenever the column mass is positive, but the
`sqrt(k * k/g)` form is friendlier for exact Lean rewriting from the geometric
mean affinity. -/
noncomputable def algorithm2ColumnReweightedWeight
    (x : Fin Nx → E) (temperature : ℝ) (i : Fin Nx) (y : E) : ℝ :=
  Real.sqrt
    (algorithm2Kernel temperature (x i) y *
      (algorithm2Kernel temperature (x i) y /
        algorithm2ColumnKernelMass x temperature y))

theorem algorithm2Kernel_pos (temperature : ℝ) (x y : E) :
    0 < algorithm2Kernel temperature x y := by
  unfold algorithm2Kernel
  exact Real.exp_pos _

theorem algorithm2Kernel_nonneg (temperature : ℝ) (x y : E) :
    0 ≤ algorithm2Kernel temperature x y :=
  (algorithm2Kernel_pos temperature x y).le

theorem algorithm2ColumnKernelMass_pos [Nonempty (Fin Nx)]
    (x : Fin Nx → E) (temperature : ℝ) (y : E) :
    0 < algorithm2ColumnKernelMass x temperature y := by
  unfold algorithm2ColumnKernelMass
  exact Finset.sum_pos
    (fun r _ => algorithm2Kernel_pos temperature (x r) y)
    Finset.univ_nonempty

theorem algorithm2ColumnKernelMass_nonneg
    (x : Fin Nx → E) (temperature : ℝ) (y : E) :
    0 ≤ algorithm2ColumnKernelMass x temperature y := by
  unfold algorithm2ColumnKernelMass
  exact Finset.sum_nonneg fun r _ => algorithm2Kernel_nonneg temperature (x r) y

theorem algorithm2RowKernelMass_pos_of_posSample
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) (j : Fin Npos) :
    0 < algorithm2RowKernelMass x yPos yNeg temperature i := by
  unfold algorithm2RowKernelMass
  exact Finset.sum_pos
    (fun s _ =>
      algorithm2Kernel_pos temperature (x i) (algorithm2SampleValue yPos yNeg s))
    ⟨Sum.inl j, Finset.mem_univ _⟩

theorem algorithm2RowKernelMass_pos_of_negSample
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) (j : Fin Nneg) :
    0 < algorithm2RowKernelMass x yPos yNeg temperature i := by
  unfold algorithm2RowKernelMass
  exact Finset.sum_pos
    (fun s _ =>
      algorithm2Kernel_pos temperature (x i) (algorithm2SampleValue yPos yNeg s))
    ⟨Sum.inr j, Finset.mem_univ _⟩

theorem algorithm2ColumnReweightedWeight_nonneg
    (x : Fin Nx → E) (temperature : ℝ) (i : Fin Nx) (y : E) :
    0 ≤ algorithm2ColumnReweightedWeight x temperature i y := by
  unfold algorithm2ColumnReweightedWeight
  exact Real.sqrt_nonneg _

theorem algorithm2ColumnReweightedWeight_pos [Nonempty (Fin Nx)]
    (x : Fin Nx → E) (temperature : ℝ) (i : Fin Nx) (y : E) :
    0 < algorithm2ColumnReweightedWeight x temperature i y := by
  unfold algorithm2ColumnReweightedWeight
  exact Real.sqrt_pos.2
    (mul_pos (algorithm2Kernel_pos temperature (x i) y)
      (div_pos (algorithm2Kernel_pos temperature (x i) y)
        (algorithm2ColumnKernelMass_pos x temperature y)))

theorem exp_algorithm2Logit_false
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    Real.exp (algorithm2Logit x yPos yNeg temperature (fun _ _ => false) i s) =
      algorithm2Kernel temperature (x i) (algorithm2SampleValue yPos yNeg s) := by
  cases s <;> simp [algorithm2Logit, algorithm2Kernel, algorithm2SampleValue]

theorem algorithm2RowAffinity_false_eq_kernel_div_rowMass
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    algorithm2RowAffinity x yPos yNeg temperature (fun _ _ => false) i s =
      algorithm2Kernel temperature (x i) (algorithm2SampleValue yPos yNeg s) /
        algorithm2RowKernelMass x yPos yNeg temperature i := by
  unfold algorithm2RowAffinity finiteSoftmax algorithm2RowKernelMass
  rw [exp_algorithm2Logit_false]
  congr 1
  apply Finset.sum_congr rfl
  intro t _
  rw [exp_algorithm2Logit_false]

theorem algorithm2ColumnAffinity_false_eq_kernel_div_columnMass
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg) :
    algorithm2ColumnAffinity x yPos yNeg temperature (fun _ _ => false) i s =
      algorithm2Kernel temperature (x i) (algorithm2SampleValue yPos yNeg s) /
        algorithm2ColumnKernelMass x temperature
          (algorithm2SampleValue yPos yNeg s) := by
  unfold algorithm2ColumnAffinity finiteSoftmax algorithm2ColumnKernelMass
  rw [exp_algorithm2Logit_false]
  congr 1
  apply Finset.sum_congr rfl
  intro r _
  rw [exp_algorithm2Logit_false]

/-- With `selfMask = false`, the geometric affinity is a common row factor
times a column-reweighted single-sample weight. -/
theorem algorithm2Affinity_false_eq_rowScale_mul_columnReweightedWeight
    [Nonempty (Fin Nx)]
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) (s : Algorithm2SampleIndex Npos Nneg)
    (hrow : 0 < algorithm2RowKernelMass x yPos yNeg temperature i) :
    algorithm2Affinity x yPos yNeg temperature (fun _ _ => false) i s =
      Real.sqrt ((algorithm2RowKernelMass x yPos yNeg temperature i)⁻¹) *
        algorithm2ColumnReweightedWeight x temperature i
          (algorithm2SampleValue yPos yNeg s) := by
  unfold algorithm2Affinity algorithm2ColumnReweightedWeight
  rw [algorithm2RowAffinity_false_eq_kernel_div_rowMass,
    algorithm2ColumnAffinity_false_eq_kernel_div_columnMass]
  set k := algorithm2Kernel temperature (x i) (algorithm2SampleValue yPos yNeg s)
  set D := algorithm2RowKernelMass x yPos yNeg temperature i
  set G := algorithm2ColumnKernelMass x temperature (algorithm2SampleValue yPos yNeg s)
  have hDinv : 0 ≤ D⁻¹ := inv_nonneg.mpr hrow.le
  have hrewrite : k / D * (k / G) = D⁻¹ * (k * (k / G)) := by
    rw [div_eq_inv_mul]
    ring
  rw [hrewrite, Real.sqrt_mul hDinv]

section CentroidCancellation

variable {ι : Type*} [Fintype ι]
variable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- Multiplying every SNIS weight by the same nonzero scalar leaves the
self-normalized centroid unchanged. -/
theorem selfNormalizedCentroid_eq_of_common_scale
    (lam : ℝ) (w : ι → ℝ) (y : ι → F) (hlam : lam ≠ 0) :
    (∑ j, lam * w j)⁻¹ • (∑ j, (lam * w j) • y j) =
      (∑ j, w j)⁻¹ • (∑ j, w j • y j) := by
  set S : ℝ := ∑ j, w j with hS
  set T : F := ∑ j, w j • y j with hT
  have hsum : (∑ j, lam * w j) = lam * S := by
    simp [hS, Finset.mul_sum]
  have hvec : (∑ j, (lam * w j) • y j) = lam • T := by
    rw [hT, Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [smul_smul]
  have hcoef : (lam * S)⁻¹ * lam = S⁻¹ := by
    by_cases hS0 : S = 0
    · simp [hS0]
    · field_simp [hlam, hS0]
  rw [hsum, hvec, smul_smul, hcoef]

end CentroidCancellation

variable [NormedSpace ℝ E]

/-- The positive centroid in Algorithm 2 is exactly a self-normalized centroid
with the column-reweighted weight; the row normalization cancels. -/
theorem algorithm2PositiveCentroid_false_eq_columnReweighted
    [Nonempty (Fin Nx)] [Nonempty (Fin Npos)]
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) :
    algorithm2PositiveCentroid x yPos yNeg temperature (fun _ _ => false) i =
      (∑ j : Fin Npos, algorithm2ColumnReweightedWeight x temperature i (yPos j))⁻¹ •
        (∑ j : Fin Npos,
          algorithm2ColumnReweightedWeight x temperature i (yPos j) • yPos j) := by
  classical
  let lam : ℝ := Real.sqrt ((algorithm2RowKernelMass x yPos yNeg temperature i)⁻¹)
  let w : Fin Npos → ℝ := fun j =>
    algorithm2ColumnReweightedWeight x temperature i (yPos j)
  have hrow : 0 < algorithm2RowKernelMass x yPos yNeg temperature i :=
    algorithm2RowKernelMass_pos_of_posSample x yPos yNeg temperature i
      (Classical.choice ‹Nonempty (Fin Npos)›)
  have hlam : lam ≠ 0 := by
    have hlampos : 0 < lam := by
      unfold lam
      exact Real.sqrt_pos.2 (inv_pos.mpr hrow)
    exact ne_of_gt hlampos
  have hA : ∀ j : Fin Npos,
      algorithm2Affinity x yPos yNeg temperature (fun _ _ => false) i (Sum.inl j) =
        lam * w j := by
    intro j
    unfold lam w
    simpa [algorithm2SampleValue] using
      algorithm2Affinity_false_eq_rowScale_mul_columnReweightedWeight
        x yPos yNeg temperature i (Sum.inl j) hrow
  unfold algorithm2PositiveCentroid algorithm2PositiveMass
  calc
    (∑ j : Fin Npos,
        algorithm2Affinity x yPos yNeg temperature (fun _ _ => false) i (Sum.inl j))⁻¹ •
        (∑ j : Fin Npos,
          algorithm2Affinity x yPos yNeg temperature (fun _ _ => false) i (Sum.inl j) •
            yPos j)
        =
      (∑ j : Fin Npos, lam * w j)⁻¹ •
        (∑ j : Fin Npos, (lam * w j) • yPos j) := by simp_rw [hA]
    _ =
      (∑ j : Fin Npos, w j)⁻¹ • (∑ j : Fin Npos, w j • yPos j) :=
        selfNormalizedCentroid_eq_of_common_scale lam w yPos hlam
    _ =
      (∑ j : Fin Npos, algorithm2ColumnReweightedWeight x temperature i (yPos j))⁻¹ •
        (∑ j : Fin Npos,
          algorithm2ColumnReweightedWeight x temperature i (yPos j) • yPos j) := rfl

/-- The negative centroid in Algorithm 2 is exactly a self-normalized centroid
with the column-reweighted weight; the row normalization cancels. -/
theorem algorithm2NegativeCentroid_false_eq_columnReweighted
    [Nonempty (Fin Nx)] [Nonempty (Fin Nneg)]
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) :
    algorithm2NegativeCentroid x yPos yNeg temperature (fun _ _ => false) i =
      (∑ j : Fin Nneg, algorithm2ColumnReweightedWeight x temperature i (yNeg j))⁻¹ •
        (∑ j : Fin Nneg,
          algorithm2ColumnReweightedWeight x temperature i (yNeg j) • yNeg j) := by
  classical
  let lam : ℝ := Real.sqrt ((algorithm2RowKernelMass x yPos yNeg temperature i)⁻¹)
  let w : Fin Nneg → ℝ := fun j =>
    algorithm2ColumnReweightedWeight x temperature i (yNeg j)
  have hrow : 0 < algorithm2RowKernelMass x yPos yNeg temperature i :=
    algorithm2RowKernelMass_pos_of_negSample x yPos yNeg temperature i
      (Classical.choice ‹Nonempty (Fin Nneg)›)
  have hlam : lam ≠ 0 := by
    have hlampos : 0 < lam := by
      unfold lam
      exact Real.sqrt_pos.2 (inv_pos.mpr hrow)
    exact ne_of_gt hlampos
  have hA : ∀ j : Fin Nneg,
      algorithm2Affinity x yPos yNeg temperature (fun _ _ => false) i (Sum.inr j) =
        lam * w j := by
    intro j
    unfold lam w
    simpa [algorithm2SampleValue] using
      algorithm2Affinity_false_eq_rowScale_mul_columnReweightedWeight
        x yPos yNeg temperature i (Sum.inr j) hrow
  unfold algorithm2NegativeCentroid algorithm2NegativeMass
  calc
    (∑ j : Fin Nneg,
        algorithm2Affinity x yPos yNeg temperature (fun _ _ => false) i (Sum.inr j))⁻¹ •
        (∑ j : Fin Nneg,
          algorithm2Affinity x yPos yNeg temperature (fun _ _ => false) i (Sum.inr j) •
            yNeg j)
        =
      (∑ j : Fin Nneg, lam * w j)⁻¹ •
        (∑ j : Fin Nneg, (lam * w j) • yNeg j) := by simp_rw [hA]
    _ =
      (∑ j : Fin Nneg, w j)⁻¹ • (∑ j : Fin Nneg, w j • yNeg j) :=
        selfNormalizedCentroid_eq_of_common_scale lam w yNeg hlam
    _ =
      (∑ j : Fin Nneg, algorithm2ColumnReweightedWeight x temperature i (yNeg j))⁻¹ •
        (∑ j : Fin Nneg,
          algorithm2ColumnReweightedWeight x temperature i (yNeg j) • yNeg j) := rfl

/-- Raw Algorithm-2 drift is zero exactly when its self-normalized centroid
difference is zero, provided both sample sides are nonempty. -/
theorem algorithm2Drift_false_eq_zero_iff_centroidDiff_eq_zero
    [Nonempty (Fin Npos)] [Nonempty (Fin Nneg)]
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) :
    algorithm2Drift x yPos yNeg temperature (fun _ _ => false) i = 0 ↔
      algorithm2PositiveCentroid x yPos yNeg temperature (fun _ _ => false) i -
        algorithm2NegativeCentroid x yPos yNeg temperature (fun _ _ => false) i = 0 := by
  classical
  have hPpos : 0 <
      algorithm2PositiveMass x yPos yNeg temperature (fun _ _ => false) i :=
    algorithm2PositiveMass_pos x yPos yNeg temperature (fun _ _ => false) i
  have hQpos : 0 <
      algorithm2NegativeMass x yPos yNeg temperature (fun _ _ => false) i :=
    algorithm2NegativeMass_pos x yPos yNeg temperature (fun _ _ => false) i
  have hPQ : algorithm2PositiveMass x yPos yNeg temperature (fun _ _ => false) i *
      algorithm2NegativeMass x yPos yNeg temperature (fun _ _ => false) i ≠ 0 :=
    mul_ne_zero (ne_of_gt hPpos) (ne_of_gt hQpos)
  rw [algorithm2Drift_eq_massProduct_centroidDiff x yPos yNeg temperature
    (fun _ _ => false) i (ne_of_gt hPpos) (ne_of_gt hQpos)]
  constructor
  · intro h
    exact (smul_eq_zero.mp h).resolve_left hPQ
  · intro h
    rw [h, smul_zero]

/-- If the product of the two affinity masses is bounded below by `L>0`, then
the normalized centroid difference is controlled by the raw Algorithm-2 drift. -/
theorem centroidDiff_norm_le_inv_massProduct_mul_drift_norm
    [Nonempty (Fin Npos)] [Nonempty (Fin Nneg)]
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) {L : ℝ}
    (hL : 0 < L)
    (hmass : L ≤
      algorithm2PositiveMass x yPos yNeg temperature (fun _ _ => false) i *
        algorithm2NegativeMass x yPos yNeg temperature (fun _ _ => false) i) :
    ‖algorithm2PositiveCentroid x yPos yNeg temperature (fun _ _ => false) i -
        algorithm2NegativeCentroid x yPos yNeg temperature (fun _ _ => false) i‖
      ≤ L⁻¹ * ‖algorithm2Drift x yPos yNeg temperature (fun _ _ => false) i‖ := by
  classical
  set Pm := algorithm2PositiveMass x yPos yNeg temperature (fun _ _ => false) i
  set Qm := algorithm2NegativeMass x yPos yNeg temperature (fun _ _ => false) i
  set Δ := algorithm2PositiveCentroid x yPos yNeg temperature (fun _ _ => false) i -
    algorithm2NegativeCentroid x yPos yNeg temperature (fun _ _ => false) i
  have hPpos : 0 < Pm := by
    simpa [Pm] using algorithm2PositiveMass_pos x yPos yNeg temperature (fun _ _ => false) i
  have hQpos : 0 < Qm := by
    simpa [Qm] using algorithm2NegativeMass_pos x yPos yNeg temperature (fun _ _ => false) i
  have hprodpos : 0 < Pm * Qm := mul_pos hPpos hQpos
  have hdrift :
      algorithm2Drift x yPos yNeg temperature (fun _ _ => false) i =
        (Pm * Qm) • Δ := by
    simpa [Pm, Qm, Δ] using
      algorithm2Drift_eq_massProduct_centroidDiff x yPos yNeg temperature
        (fun _ _ => false) i (ne_of_gt hPpos) (ne_of_gt hQpos)
  have hnorm : ‖algorithm2Drift x yPos yNeg temperature (fun _ _ => false) i‖ =
      (Pm * Qm) * ‖Δ‖ := by
    rw [hdrift, norm_smul, Real.norm_eq_abs, abs_of_pos hprodpos]
  have hmul : L * ‖Δ‖ ≤ ‖algorithm2Drift x yPos yNeg temperature (fun _ _ => false) i‖ := by
    rw [hnorm]
    exact mul_le_mul_of_nonneg_right (by simpa [Pm, Qm] using hmass) (norm_nonneg Δ)
  calc ‖Δ‖
      = L⁻¹ * (L * ‖Δ‖) := by
          rw [← mul_assoc, inv_mul_cancel₀ (ne_of_gt hL), one_mul]
    _ ≤ L⁻¹ * ‖algorithm2Drift x yPos yNeg temperature (fun _ _ => false) i‖ :=
        mul_le_mul_of_nonneg_left hmul (inv_nonneg.mpr hL.le)

end ColumnReweighted

section Algorithm2SNISConsistency

variable {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
  [MeasurableSpace F] [BorelSpace F] [CompleteSpace F] [SecondCountableTopology F]
variable {Nx Npos Nneg : ℕ} [Nonempty (Fin Nx)]

/-- The positive Algorithm-2 centroid, with `selfMask = false`, inherits the
generic self-normalized MSE bound.  The negative batch is allowed to be random:
after the row-normalization cancellation it no longer appears in the positive
centroid. -/
theorem algorithm2PositiveCentroid_false_meanSquare_le
    {Npos : ℕ} (hNpos : 0 < Npos)
    (anchors : Fin Nx → F) (temperature : ℝ) (i : Fin Nx)
    (Ypos : Fin Npos → Ω → F) (Yneg : Fin Nneg → Ω → F) (c : F)
    {wmin wmax R σ : ℝ} (hwmin : 0 < wmin) (hwmax : 0 ≤ wmax) (hR : 0 ≤ R)
    (hYmeas : ∀ j, Measurable (Ypos j))
    (hw : Measurable (algorithm2ColumnReweightedWeight anchors temperature i))
    (hindep : ∀ j k, j ≠ k → IndepFun (Ypos j) (Ypos k) P)
    (hwlb : ∀ j ω,
      wmin ≤ algorithm2ColumnReweightedWeight anchors temperature i (Ypos j ω))
    (hwub : ∀ j ω,
      algorithm2ColumnReweightedWeight anchors temperature i (Ypos j ω) ≤ wmax)
    (hYbd : ∀ j ω, ‖Ypos j ω - c‖ ≤ R)
    (hmean : ∀ j,
      ∫ ω, algorithm2ColumnReweightedWeight anchors temperature i (Ypos j ω) •
        (Ypos j ω - c) ∂P = 0)
    (hσ : ∀ j,
      ∫ ω, ‖algorithm2ColumnReweightedWeight anchors temperature i (Ypos j ω) •
        (Ypos j ω - c)‖ ^ 2 ∂P ≤ σ ^ 2) :
    ∫ ω,
        ‖algorithm2PositiveCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature (fun _ _ => false) i - c‖ ^ 2 ∂P
      ≤ σ ^ 2 / (wmin ^ 2 * Npos) := by
  haveI : Nonempty (Fin Npos) := ⟨⟨0, hNpos⟩⟩
  let w : F → ℝ := algorithm2ColumnReweightedWeight anchors temperature i
  have hcent : ∀ ω,
      algorithm2PositiveCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature (fun _ _ => false) i =
        (∑ j : Fin Npos, w (Ypos j ω))⁻¹ •
          (∑ j : Fin Npos, w (Ypos j ω) • Ypos j ω) := by
    intro ω
    simpa [w] using
      algorithm2PositiveCentroid_false_eq_columnReweighted
        anchors (fun j => Ypos j ω) (fun l => Yneg l ω) temperature i
  have hSNIS := SelfNormalized.selfNormalized_meanSquare_le
    P hNpos Ypos w c hwmin hwmax hR hYmeas hw hindep hwlb hwub hYbd hmean hσ
  calc
    ∫ ω,
        ‖algorithm2PositiveCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature (fun _ _ => false) i - c‖ ^ 2 ∂P
        =
      ∫ ω, ‖(∑ j : Fin Npos, w (Ypos j ω))⁻¹ •
          (∑ j : Fin Npos, w (Ypos j ω) • Ypos j ω) - c‖ ^ 2 ∂P := by
        apply integral_congr_ae
        filter_upwards with ω
        rw [hcent ω]
    _ ≤ σ ^ 2 / (wmin ^ 2 * Npos) := hSNIS

/-- The negative Algorithm-2 centroid, with `selfMask = false`, inherits the
generic self-normalized MSE bound.  The positive batch is allowed to be random:
after the row-normalization cancellation it no longer appears in the negative
centroid. -/
theorem algorithm2NegativeCentroid_false_meanSquare_le
    {Nneg : ℕ} (hNneg : 0 < Nneg)
    (anchors : Fin Nx → F) (temperature : ℝ) (i : Fin Nx)
    (Ypos : Fin Npos → Ω → F) (Yneg : Fin Nneg → Ω → F) (c : F)
    {wmin wmax R σ : ℝ} (hwmin : 0 < wmin) (hwmax : 0 ≤ wmax) (hR : 0 ≤ R)
    (hYmeas : ∀ j, Measurable (Yneg j))
    (hw : Measurable (algorithm2ColumnReweightedWeight anchors temperature i))
    (hindep : ∀ j k, j ≠ k → IndepFun (Yneg j) (Yneg k) P)
    (hwlb : ∀ j ω,
      wmin ≤ algorithm2ColumnReweightedWeight anchors temperature i (Yneg j ω))
    (hwub : ∀ j ω,
      algorithm2ColumnReweightedWeight anchors temperature i (Yneg j ω) ≤ wmax)
    (hYbd : ∀ j ω, ‖Yneg j ω - c‖ ≤ R)
    (hmean : ∀ j,
      ∫ ω, algorithm2ColumnReweightedWeight anchors temperature i (Yneg j ω) •
        (Yneg j ω - c) ∂P = 0)
    (hσ : ∀ j,
      ∫ ω, ‖algorithm2ColumnReweightedWeight anchors temperature i (Yneg j ω) •
        (Yneg j ω - c)‖ ^ 2 ∂P ≤ σ ^ 2) :
    ∫ ω,
        ‖algorithm2NegativeCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature (fun _ _ => false) i - c‖ ^ 2 ∂P
      ≤ σ ^ 2 / (wmin ^ 2 * Nneg) := by
  haveI : Nonempty (Fin Nneg) := ⟨⟨0, hNneg⟩⟩
  let w : F → ℝ := algorithm2ColumnReweightedWeight anchors temperature i
  have hcent : ∀ ω,
      algorithm2NegativeCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature (fun _ _ => false) i =
        (∑ j : Fin Nneg, w (Yneg j ω))⁻¹ •
          (∑ j : Fin Nneg, w (Yneg j ω) • Yneg j ω) := by
    intro ω
    simpa [w] using
      algorithm2NegativeCentroid_false_eq_columnReweighted
        anchors (fun j => Ypos j ω) (fun l => Yneg l ω) temperature i
  have hSNIS := SelfNormalized.selfNormalized_meanSquare_le
    P hNneg Yneg w c hwmin hwmax hR hYmeas hw hindep hwlb hwub hYbd hmean hσ
  calc
    ∫ ω,
        ‖algorithm2NegativeCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature (fun _ _ => false) i - c‖ ^ 2 ∂P
        =
      ∫ ω, ‖(∑ j : Fin Nneg, w (Yneg j ω))⁻¹ •
          (∑ j : Fin Nneg, w (Yneg j ω) • Yneg j ω) - c‖ ^ 2 ∂P := by
        apply integral_congr_ae
        filter_upwards with ω
        rw [hcent ω]
    _ ≤ σ ^ 2 / (wmin ^ 2 * Nneg) := hSNIS

end Algorithm2SNISConsistency

section MeanSquareGlue

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω}
variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

omit [InnerProductSpace ℝ F] in
/-- Combining two mean-squared centroid errors controls the mean-squared error
of their difference.  This is the finite-sample glue used after applying the
SNIS theorem separately to the positive and negative minibatches. -/
theorem meanSquare_sub_sub_le_two_add
    (X Y : Ω → F) (x y : F)
    (hXint : Integrable (fun ω => ‖X ω - x‖ ^ 2) P)
    (hYint : Integrable (fun ω => ‖Y ω - y‖ ^ 2) P)
    (hZint : Integrable (fun ω => ‖(X ω - Y ω) - (x - y)‖ ^ 2) P)
    {AX AY : ℝ}
    (hX : ∫ ω, ‖X ω - x‖ ^ 2 ∂P ≤ AX)
    (hY : ∫ ω, ‖Y ω - y‖ ^ 2 ∂P ≤ AY) :
    ∫ ω, ‖(X ω - Y ω) - (x - y)‖ ^ 2 ∂P ≤ 2 * AX + 2 * AY := by
  have hright_int : Integrable
      (fun ω => 2 * ‖X ω - x‖ ^ 2 + 2 * ‖Y ω - y‖ ^ 2) P :=
    (hXint.const_mul 2).add (hYint.const_mul 2)
  have hpoint : ∀ ω, ‖(X ω - Y ω) - (x - y)‖ ^ 2 ≤
      2 * ‖X ω - x‖ ^ 2 + 2 * ‖Y ω - y‖ ^ 2 := by
    intro ω
    have hrewrite : (X ω - Y ω) - (x - y) = (X ω - x) - (Y ω - y) := by abel
    rw [hrewrite]
    have hnorm : ‖(X ω - x) - (Y ω - y)‖ ≤ ‖X ω - x‖ + ‖Y ω - y‖ := by
      have h := norm_add_le (X ω - x) (-(Y ω - y))
      rw [norm_neg] at h
      simpa [sub_eq_add_neg] using h
    have hsq : ‖(X ω - x) - (Y ω - y)‖ ^ 2 ≤
        (‖X ω - x‖ + ‖Y ω - y‖) ^ 2 :=
      sq_le_sq' (by
        have h0 : 0 ≤ ‖(X ω - x) - (Y ω - y)‖ := norm_nonneg _
        have h1 : 0 ≤ ‖X ω - x‖ + ‖Y ω - y‖ :=
          add_nonneg (norm_nonneg _) (norm_nonneg _)
        linarith) hnorm
    nlinarith [hsq, sq_nonneg (‖X ω - x‖ - ‖Y ω - y‖)]
  calc ∫ ω, ‖(X ω - Y ω) - (x - y)‖ ^ 2 ∂P
      ≤ ∫ ω, (2 * ‖X ω - x‖ ^ 2 + 2 * ‖Y ω - y‖ ^ 2) ∂P :=
        integral_mono hZint hright_int hpoint
    _ = 2 * ∫ ω, ‖X ω - x‖ ^ 2 ∂P +
        2 * ∫ ω, ‖Y ω - y‖ ^ 2 ∂P := by
          rw [integral_add (hXint.const_mul 2) (hYint.const_mul 2),
            integral_const_mul, integral_const_mul]
    _ ≤ 2 * AX + 2 * AY := by
        gcongr

end MeanSquareGlue

end Algorithm2
end DriftingIdentifiability
