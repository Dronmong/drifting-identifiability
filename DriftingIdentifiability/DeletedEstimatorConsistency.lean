import DriftingIdentifiability.SelfMaskPerturbation

/-!
# Statistical consistency of the deleted (leave-masked-out) estimator

`SelfMaskPerturbation.lean` proves that the implementation self-mask is a
deterministic `exp(-10⁶/τ)` perturbation of the **deleted** estimator.  This
module supplies the missing statistical half: the deleted estimator's centroids
satisfy explicit mean-square consistency bounds, with the *exact*
index-dependent leave-masked-out structure in the hypotheses.

Structure of the argument:

* `deletedDrift_eq_massProduct_centroidDiff` — like the raw Algorithm-2 drift,
  the deleted drift is a mass product times a difference of two self-normalized
  centroids.
* **Row cancellation** (`deletedAffinity_eq_rowScale_mul_deletedColumnWeight`):
  the deleted affinity factors into a common row scale times the per-slot
  weight `deletedColumnWeight`, whose column mass drops exactly the anchors
  masked in that slot's column.  The row scale cancels in each centroid.
* **Positives are never masked**, so the deleted positive-slot weight is the
  plain column-reweighted weight (`deletedColumnWeight_inl_eq`) and the deleted
  positive centroid *equals* the promoted no-mask positive centroid
  (`deletedPositiveCentroid_eq_algorithm2PositiveCentroid_false`).  Its
  mean-square bound transports verbatim
  (`deletedPositiveCentroid_meanSquare_le`).
* The deleted **negative** centroid is a self-normalized average with a
  *different* weight function on each slot.  Its mean-square bound
  (`deletedNegativeCentroid_meanSquare_le`) instantiates the indexed,
  bias-tolerant ratio theorem `selfNormalizedIndexed_meanSquare_le`: per-slot
  reweighted mean shifts `μ l` (the leave-out bias) enter through `‖μ l‖ ≤ b`,
  and the denominator floor `dmin` accounts for slots zeroed by the mask.

No theorem here asserts identifiability; the bounds feed the estimator-agnostic
finite-sample bridge through the mean squared error, exactly like the no-mask
route.
-/

open scoped BigOperators
open MeasureTheory ProbabilityTheory

namespace DriftingIdentifiability
namespace Algorithm2

open Paper

universe u

/-! ## Generic bilinear pair-sum algebra -/

/-- The affinity pair sum collapses to mass-scaled centroid sums.  Pure
algebra, shared by the raw and deleted drifts. -/
theorem sum_sum_mul_smul_sub {ι κ : Type*} [Fintype ι] [Fintype κ]
    {V : Type*} [AddCommGroup V] [Module ℝ V]
    (a : ι → ℝ) (b : κ → ℝ) (u : ι → V) (v : κ → V) :
    (∑ j, ∑ l, (a j * b l) • (u j - v l)) =
      (∑ l, b l) • (∑ j, a j • u j) - (∑ j, a j) • (∑ l, b l • v l) := by
  have hsplit : (∑ j, ∑ l, (a j * b l) • (u j - v l)) =
      (∑ j, ∑ l, (a j * b l) • u j) - ∑ j, ∑ l, (a j * b l) • v l := by
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro j _
    rw [← Finset.sum_sub_distrib]
    apply Finset.sum_congr rfl
    intro l _
    rw [smul_sub]
  rw [hsplit]
  congr 1
  · rw [Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro j _
    rw [smul_smul, ← Finset.sum_smul, ← Finset.mul_sum, mul_comm (a j) (∑ l, b l)]
  · rw [Finset.sum_comm, Finset.smul_sum]
    apply Finset.sum_congr rfl
    intro l _
    rw [smul_smul, ← Finset.sum_smul, ← Finset.sum_mul]

/-! ## Deleted masses, centroids, and per-slot weights -/

section DeletedCentroids

variable {E : Type u} [NormedAddCommGroup E]
variable {Nx Npos Nneg : ℕ}
variable (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
variable (temperature : ℝ) (selfMask : Fin Nx → Fin Nneg → Bool)

/-- Total positive affinity mass of the deleted system at anchor `i`. -/
noncomputable def deletedPositiveMass (i : Fin Nx) : ℝ :=
  ∑ j : Fin Npos, deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j)

/-- Total negative affinity mass of the deleted system at anchor `i`. -/
noncomputable def deletedNegativeMass (i : Fin Nx) : ℝ :=
  ∑ l : Fin Nneg, deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l)

/-- Self-normalized positive centroid of the deleted system. -/
noncomputable def deletedPositiveCentroid [NormedSpace ℝ E] (i : Fin Nx) : E :=
  (deletedPositiveMass x yPos yNeg temperature selfMask i)⁻¹ •
    ∑ j : Fin Npos,
      deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j) • yPos j

/-- Self-normalized negative centroid of the deleted system. -/
noncomputable def deletedNegativeCentroid [NormedSpace ℝ E] (i : Fin Nx) : E :=
  (deletedNegativeMass x yPos yNeg temperature selfMask i)⁻¹ •
    ∑ l : Fin Nneg,
      deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l) • yNeg l

/-- Column-mass profile of the deleted system in slot `s`, as a function of the
sample value: the kernel mass of exactly the anchors left unmasked in that
column. -/
noncomputable def deletedColumnMassFn (s : Algorithm2SampleIndex Npos Nneg)
    (y : E) : ℝ :=
  ∑ i' : Fin Nx, deletedFactor selfMask i' s * algorithm2Kernel temperature (x i') y

/-- Per-slot deleted SNIS weight for anchor `i`: the kernel renormalized by the
square root of that slot's unmasked column mass.  This is the index-dependent
weight family of the leave-masked-out estimator. -/
noncomputable def deletedColumnWeight (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) (y : E) : ℝ :=
  deletedFactor selfMask i s * algorithm2Kernel temperature (x i) y /
    Real.sqrt (deletedColumnMassFn x temperature selfMask s y)

/-- The deleted weight is the mask factor times the kernel at the sample. -/
theorem deletedWeight_eq_factor_mul_kernel (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedWeight x yPos yNeg temperature selfMask i s =
      deletedFactor selfMask i s *
        algorithm2Kernel temperature (x i) (algorithm2SampleValue yPos yNeg s) := by
  unfold deletedWeight noMaskWeight
  rw [exp_algorithm2Logit_false]

/-- The deleted column mass is the column-mass profile at the sample value. -/
theorem deletedColumnMass_eq_fn (s : Algorithm2SampleIndex Npos Nneg) :
    deletedColumnMass x yPos yNeg temperature selfMask s =
      deletedColumnMassFn x temperature selfMask s
        (algorithm2SampleValue yPos yNeg s) := by
  unfold deletedColumnMass deletedColumnMassFn
  apply Finset.sum_congr rfl
  intro i' _
  exact deletedWeight_eq_factor_mul_kernel x yPos yNeg temperature selfMask i' s

/-- Mass-scaled centroid form of the deleted drift. -/
theorem deletedDrift_eq_massScaledCentroid [NormedSpace ℝ E] (i : Fin Nx) :
    deletedDrift x yPos yNeg temperature selfMask i =
      deletedNegativeMass x yPos yNeg temperature selfMask i •
        (∑ j : Fin Npos,
          deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j) • yPos j) -
      deletedPositiveMass x yPos yNeg temperature selfMask i •
        (∑ l : Fin Nneg,
          deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l) • yNeg l) := by
  unfold deletedDrift deletedPositiveMass deletedNegativeMass
  exact sum_sum_mul_smul_sub _ _ _ _

/-- **Self-normalized centroid form of the deleted drift.**  With both deleted
affinity masses nonzero, the deleted drift is the mass product times the
difference of the two deleted centroids — the algebraic hook from the deleted
estimator to the indexed ratio-consistency theorem. -/
theorem deletedDrift_eq_massProduct_centroidDiff [NormedSpace ℝ E] (i : Fin Nx)
    (hP : deletedPositiveMass x yPos yNeg temperature selfMask i ≠ 0)
    (hQ : deletedNegativeMass x yPos yNeg temperature selfMask i ≠ 0) :
    deletedDrift x yPos yNeg temperature selfMask i =
      (deletedPositiveMass x yPos yNeg temperature selfMask i *
        deletedNegativeMass x yPos yNeg temperature selfMask i) •
        (deletedPositiveCentroid x yPos yNeg temperature selfMask i -
          deletedNegativeCentroid x yPos yNeg temperature selfMask i) := by
  rw [deletedDrift_eq_massScaledCentroid]
  unfold deletedPositiveCentroid deletedNegativeCentroid
  set Pm := deletedPositiveMass x yPos yNeg temperature selfMask i with hPm
  set Qm := deletedNegativeMass x yPos yNeg temperature selfMask i with hQm
  have hfirst : (Pm * Qm) * Pm⁻¹ = Qm := by field_simp
  have hsecond : (Pm * Qm) * Qm⁻¹ = Pm := by field_simp
  rw [smul_sub, smul_smul, smul_smul, hfirst, hsecond]

/-- **Row cancellation for the deleted system.**  The deleted affinity factors
into a common row scale times the per-slot deleted column weight evaluated at
the sample. -/
theorem deletedAffinity_eq_rowScale_mul_deletedColumnWeight (i : Fin Nx)
    (s : Algorithm2SampleIndex Npos Nneg) :
    deletedAffinity x yPos yNeg temperature selfMask i s =
      Real.sqrt ((deletedRowMass x yPos yNeg temperature selfMask i)⁻¹) *
        deletedColumnWeight x temperature selfMask i s
          (algorithm2SampleValue yPos yNeg s) := by
  unfold deletedAffinity deletedColumnWeight
  rw [← deletedColumnMass_eq_fn x yPos yNeg temperature selfMask s,
    ← deletedWeight_eq_factor_mul_kernel x yPos yNeg temperature selfMask i s,
    Real.sqrt_mul (deletedRowMass_nonneg x yPos yNeg temperature selfMask i),
    Real.sqrt_inv, inv_mul_eq_div, div_div,
    mul_comm (Real.sqrt (deletedColumnMass x yPos yNeg temperature selfMask s))
      (Real.sqrt (deletedRowMass x yPos yNeg temperature selfMask i))]

/-- Positive slots are never masked, so the per-slot deleted weight on a
positive slot is the plain column-reweighted weight. -/
theorem deletedColumnWeight_inl_eq (i : Fin Nx) (j : Fin Npos) (y : E) :
    deletedColumnWeight x temperature selfMask i (Sum.inl j) y =
      algorithm2ColumnReweightedWeight x temperature i y := by
  unfold deletedColumnWeight deletedColumnMassFn
  have hdf : ∀ i' : Fin Nx,
      deletedFactor selfMask i' (Sum.inl j : Algorithm2SampleIndex Npos Nneg)
        = (1 : ℝ) := fun i' => rfl
  simp only [hdf, one_mul]
  rw [← columnReweightedKernel_apply_anchor,
    PaperFiniteIdentifiability.columnReweightedKernel_eq_kernel_div_sqrtMass]
  rfl

/-- The negative-slot deleted weight, packaged without any reference to the
positive batch: the mask zero/one factor times the kernel, renormalized by the
square root of the column's unmasked kernel mass. -/
noncomputable def deletedNegativeColumnWeight (i : Fin Nx) (l : Fin Nneg)
    (y : E) : ℝ :=
  (if selfMask i l then 0 else 1) * algorithm2Kernel temperature (x i) y /
    Real.sqrt (∑ i' : Fin Nx,
      (if selfMask i' l then 0 else 1) * algorithm2Kernel temperature (x i') y)

/-- On negative slots the generic per-slot weight is the packaged negative
weight. -/
theorem deletedColumnWeight_inr_eq (i : Fin Nx) (l : Fin Nneg) (y : E) :
    deletedColumnWeight x temperature selfMask i
        (Sum.inr l : Algorithm2SampleIndex Npos Nneg) y =
      deletedNegativeColumnWeight x temperature selfMask i l y := rfl

/-- With a positive sample present, the deleted row mass is strictly positive:
positives are never masked. -/
theorem deletedRowMass_pos [Nonempty (Fin Npos)] (i : Fin Nx) :
    0 < deletedRowMass x yPos yNeg temperature selfMask i := by
  obtain ⟨j0⟩ := (inferInstance : Nonempty (Fin Npos))
  have hterm : 0 < deletedWeight x yPos yNeg temperature selfMask i (Sum.inl j0) := by
    unfold deletedWeight
    rw [show deletedFactor selfMask i
        (Sum.inl j0 : Algorithm2SampleIndex Npos Nneg) = (1 : ℝ) from rfl, one_mul]
    exact noMaskWeight_pos x yPos yNeg temperature i (Sum.inl j0)
  calc (0 : ℝ) < deletedWeight x yPos yNeg temperature selfMask i (Sum.inl j0) := hterm
    _ ≤ deletedRowMass x yPos yNeg temperature selfMask i := by
        unfold deletedRowMass
        exact Finset.single_le_sum
          (fun s _ => deletedWeight_nonneg x yPos yNeg temperature selfMask i s)
          (Finset.mem_univ _)

variable [NormedSpace ℝ E]

/-- The deleted positive centroid is a self-normalized centroid for the plain
column-reweighted weight: the row scale cancels and positives are unmasked. -/
theorem deletedPositiveCentroid_eq_columnReweighted [Nonempty (Fin Npos)]
    (i : Fin Nx) :
    deletedPositiveCentroid x yPos yNeg temperature selfMask i =
      (∑ j : Fin Npos, algorithm2ColumnReweightedWeight x temperature i (yPos j))⁻¹ •
        (∑ j : Fin Npos,
          algorithm2ColumnReweightedWeight x temperature i (yPos j) • yPos j) := by
  have hrow := deletedRowMass_pos x yPos yNeg temperature selfMask i
  have hlam :
      Real.sqrt ((deletedRowMass x yPos yNeg temperature selfMask i)⁻¹) ≠ 0 :=
    ne_of_gt (Real.sqrt_pos.2 (inv_pos.mpr hrow))
  have hA : ∀ j : Fin Npos,
      deletedAffinity x yPos yNeg temperature selfMask i (Sum.inl j) =
        Real.sqrt ((deletedRowMass x yPos yNeg temperature selfMask i)⁻¹) *
          algorithm2ColumnReweightedWeight x temperature i (yPos j) := by
    intro j
    have h := deletedAffinity_eq_rowScale_mul_deletedColumnWeight
      x yPos yNeg temperature selfMask i (Sum.inl j)
    rw [deletedColumnWeight_inl_eq] at h
    simpa [algorithm2SampleValue] using h
  unfold deletedPositiveCentroid deletedPositiveMass
  simp_rw [hA]
  exact selfNormalizedCentroid_eq_of_common_scale _ _ yPos hlam

/-- The deleted negative centroid is a self-normalized centroid for the
index-dependent deleted column weights: the row scale cancels, but each slot
keeps its own leave-masked-out column normalization. -/
theorem deletedNegativeCentroid_eq_deletedColumnWeight [Nonempty (Fin Npos)]
    (i : Fin Nx) :
    deletedNegativeCentroid x yPos yNeg temperature selfMask i =
      (∑ l : Fin Nneg,
        deletedNegativeColumnWeight x temperature selfMask i l (yNeg l))⁻¹ •
        (∑ l : Fin Nneg,
          deletedNegativeColumnWeight x temperature selfMask i l (yNeg l) • yNeg l) := by
  have hrow := deletedRowMass_pos x yPos yNeg temperature selfMask i
  have hlam :
      Real.sqrt ((deletedRowMass x yPos yNeg temperature selfMask i)⁻¹) ≠ 0 :=
    ne_of_gt (Real.sqrt_pos.2 (inv_pos.mpr hrow))
  have hA : ∀ l : Fin Nneg,
      deletedAffinity x yPos yNeg temperature selfMask i (Sum.inr l) =
        Real.sqrt ((deletedRowMass x yPos yNeg temperature selfMask i)⁻¹) *
          deletedNegativeColumnWeight x temperature selfMask i l (yNeg l) := by
    intro l
    have h := deletedAffinity_eq_rowScale_mul_deletedColumnWeight
      x yPos yNeg temperature selfMask i (Sum.inr l)
    rw [deletedColumnWeight_inr_eq] at h
    simpa [algorithm2SampleValue] using h
  unfold deletedNegativeCentroid deletedNegativeMass
  simp_rw [hA]
  exact selfNormalizedCentroid_eq_of_common_scale _ _ yNeg hlam

/-- **The mask does not move the positive centroid.**  The deleted positive
centroid coincides with the promoted no-mask positive centroid, because
positives are never masked and the row normalization cancels. -/
theorem deletedPositiveCentroid_eq_algorithm2PositiveCentroid_false
    [Nonempty (Fin Nx)] [Nonempty (Fin Npos)] (i : Fin Nx) :
    deletedPositiveCentroid x yPos yNeg temperature selfMask i =
      algorithm2PositiveCentroid x yPos yNeg temperature (fun _ _ => false) i := by
  rw [deletedPositiveCentroid_eq_columnReweighted,
    algorithm2PositiveCentroid_false_eq_columnReweighted]

end DeletedCentroids

/-! ## Mean-square consistency of the deleted centroids -/

section DeletedConsistency

variable {Ω : Type*} [MeasurableSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]
variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
  [MeasurableSpace F] [BorelSpace F] [CompleteSpace F] [SecondCountableTopology F]
variable {Nx Nneg : ℕ}

/-- **Deleted positive centroid consistency.**  Since the mask does not move the
positive centroid, the no-mask SNIS mean-square bound transports verbatim to
the deleted estimator, for any mask. -/
theorem deletedPositiveCentroid_meanSquare_le
    {Npos : ℕ} (hNpos : 0 < Npos) [Nonempty (Fin Nx)]
    (anchors : Fin Nx → F) (temperature : ℝ)
    (selfMask : Fin Nx → Fin Nneg → Bool) (i : Fin Nx)
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
        ‖deletedPositiveCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature selfMask i - c‖ ^ 2 ∂P
      ≤ σ ^ 2 / (wmin ^ 2 * Npos) := by
  haveI : Nonempty (Fin Npos) := ⟨⟨0, hNpos⟩⟩
  calc
    ∫ ω,
        ‖deletedPositiveCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature selfMask i - c‖ ^ 2 ∂P
        =
      ∫ ω,
        ‖algorithm2PositiveCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature (fun _ _ => false) i - c‖ ^ 2 ∂P := by
        apply integral_congr_ae
        filter_upwards with ω
        rw [deletedPositiveCentroid_eq_algorithm2PositiveCentroid_false]
    _ ≤ σ ^ 2 / (wmin ^ 2 * Npos) :=
        algorithm2PositiveCentroid_false_meanSquare_le (P := P) hNpos anchors
          temperature i Ypos Yneg c hwmin hwmax hR hYmeas hw hindep hwlb hwub
          hYbd hmean hσ

/-- **Deleted negative centroid consistency.**  The statistical theorem for the
leave-masked-out estimator, with the exact index-dependent structure: each
negative slot `l` carries its own weight function
`deletedColumnWeight … (Sum.inr l)` (its column mass drops the anchors masked
in column `l`), its own reweighted mean shift `μ l` bounded by the leave-out
bias `b`, and the mask-aware denominator floor `dmin`.  Instantiates the
indexed bias-tolerant ratio theorem `selfNormalizedIndexed_meanSquare_le`. -/
theorem deletedNegativeCentroid_meanSquare_le
    {Npos : ℕ} [Nonempty (Fin Npos)] (hNneg : 0 < Nneg)
    (anchors : Fin Nx → F) (temperature : ℝ)
    (selfMask : Fin Nx → Fin Nneg → Bool) (i : Fin Nx)
    (Ypos : Fin Npos → Ω → F) (Yneg : Fin Nneg → Ω → F) (c : F)
    (μ : Fin Nneg → F) {dmin wmax R b σ : ℝ}
    (hdmin : 0 < dmin) (hwmax : 0 ≤ wmax) (hR : 0 ≤ R)
    (hYmeas : ∀ l, Measurable (Yneg l))
    (hw : ∀ l, Measurable
      (deletedNegativeColumnWeight anchors temperature selfMask i l))
    (hindep : ∀ l k, l ≠ k → IndepFun (Yneg l) (Yneg k) P)
    (hD : ∀ ω, dmin ≤ ∑ l,
      deletedNegativeColumnWeight anchors temperature selfMask i l (Yneg l ω))
    (hwabs : ∀ l ω,
      |deletedNegativeColumnWeight anchors temperature selfMask i l (Yneg l ω)|
        ≤ wmax)
    (hYbd : ∀ l ω, ‖Yneg l ω - c‖ ≤ R)
    (hμ : ∀ l,
      ∫ ω, deletedNegativeColumnWeight anchors temperature selfMask i l
        (Yneg l ω) • (Yneg l ω - c) ∂P = μ l)
    (hb : ∀ l, ‖μ l‖ ≤ b)
    (hσ : ∀ l,
      ∫ ω, ‖deletedNegativeColumnWeight anchors temperature selfMask i l
        (Yneg l ω) • (Yneg l ω - c) - μ l‖ ^ 2 ∂P ≤ σ ^ 2) :
    ∫ ω,
        ‖deletedNegativeCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature selfMask i - c‖ ^ 2 ∂P
      ≤ (2 * Nneg * σ ^ 2 + 2 * Nneg ^ 2 * b ^ 2) / dmin ^ 2 := by
  have hSNIS := SelfNormalized.selfNormalizedIndexed_meanSquare_le P hNneg Yneg
    (fun l => deletedNegativeColumnWeight anchors temperature selfMask i l) c μ
    hdmin hwmax hR hYmeas hw hindep hD hwabs hYbd hμ hb hσ
  calc
    ∫ ω,
        ‖deletedNegativeCentroid anchors (fun j => Ypos j ω) (fun l => Yneg l ω)
          temperature selfMask i - c‖ ^ 2 ∂P
        =
      ∫ ω, ‖(∑ l, deletedNegativeColumnWeight anchors temperature selfMask i l
            (Yneg l ω))⁻¹ •
          (∑ l, deletedNegativeColumnWeight anchors temperature selfMask i l
            (Yneg l ω) • Yneg l ω) - c‖ ^ 2 ∂P := by
        apply integral_congr_ae
        filter_upwards with ω
        rw [deletedNegativeCentroid_eq_deletedColumnWeight]
    _ ≤ (2 * Nneg * σ ^ 2 + 2 * Nneg ^ 2 * b ^ 2) / dmin ^ 2 := hSNIS

end DeletedConsistency

end Algorithm2
end DriftingIdentifiability
