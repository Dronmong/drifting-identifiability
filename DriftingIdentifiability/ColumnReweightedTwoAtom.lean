import DriftingIdentifiability.ColumnReweightedMeanShift
import DriftingIdentifiability.EmpiricalFrameBound

/-!
# Concrete certified frame for the column-reweighted Algorithm-2 field

`ColumnReweightedMeanShift.lean` proves identifiability for the limiting
no-mask Algorithm-2 centroid field *conditionally* on an
`InteractionFrameBound` for the column-reweighted interaction vectors.  This
module discharges that condition in a concrete model class: the two-atom
empirical basis on `{0, 1}` with an arbitrary nonempty anchor family.

Three facts make the instantiation exact rather than perturbative:

1. `algorithm2Kernel` **is** the paper's positive-bandwidth Laplace kernel
   (`algorithm2Kernel_eq_laplaceKernel`), so the bare baseline here is the same
   kernel class as the certified `empirical01LaplaceSetup`.
2. Against a two-point empirical reference the interaction integral has the
   kernel-generic closed form `basisInteraction_empirical2`, and the column
   reweighting factor `1/sqrt(g(y))` depends only on the sample slot `y`, which
   the atoms pin to the support points.  Hence the modified interaction vector
   is an **exact strict-pair rescaling** of the bare one
   (`inducedInteractionVector_columnReweighted01_eq_smul`), with the explicit
   positive scale `1/sqrt(g(0)·g(1))`.
3. Strict positivity of the column-reweighted kernel makes the modified vector
   nonzero, so `interactionFrameBound_two` certifies the sharp frame constant
   `‖U^col₀₁‖` directly; the rescaling identity converts it into the bare
   constant exactly, and the anchor-count bound `g ≤ N` shows the reweighting
   costs at most a factor `N` (`columnReweighted01_interactionNorm_ge`).

`columnReweighted01Setup` packages the complete certified population setup, and
the promoted theorems give end-to-end identifiability, stability with `B = 1`,
and the high-probability finite-sample bridge for the concrete class.  Combined
with `Algorithm2SNIS.lean` this closes the "prove an `InteractionFrameBound`
for the actual column-reweighted interaction vectors" item of Objective 4.

The self-masked implementation is *not* covered: these theorems are for the
`selfMask = false` estimator, matching the promoted no-mask route.
-/

open scoped BigOperators
open MeasureTheory

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper Algorithm2

universe u

/-! ## The Algorithm-2 kernel is the paper's Laplace kernel -/

section KernelFacts

variable {E : Type u} [NormedAddCommGroup E]

/-- Algorithm 2's exponential kernel coincides with the paper's Laplace kernel
at bandwidth `τ`.  The bare baseline of the column-reweighted family is
therefore the certified paper kernel class, not a new object. -/
theorem algorithm2Kernel_eq_laplaceKernel (τ : ℝ) :
    algorithm2Kernel (E := E) τ = laplaceKernel τ := by
  funext x y
  unfold Algorithm2.algorithm2Kernel laplaceKernel
  congr 1
  ring

/-- Positive temperature makes the Algorithm-2 kernel at most one. -/
theorem algorithm2Kernel_le_one {τ : ℝ} (hτ : ValidTemperature τ) (x y : E) :
    algorithm2Kernel τ x y ≤ 1 := by
  have hτ' : (0 : ℝ) < τ := hτ
  unfold Algorithm2.algorithm2Kernel
  rw [Real.exp_le_one_iff, neg_div]
  exact neg_nonpos.mpr (div_nonneg (norm_nonneg _) hτ'.le)

/-- The column kernel mass is bounded by the anchor count. -/
theorem algorithm2ColumnKernelMass_le_card {N : ℕ}
    (anchors : Fin N → E) {τ : ℝ} (hτ : ValidTemperature τ) (y : E) :
    algorithm2ColumnKernelMass anchors τ y ≤ (N : ℝ) := by
  unfold Algorithm2.algorithm2ColumnKernelMass
  calc ∑ r, algorithm2Kernel τ (anchors r) y
      ≤ ∑ _r : Fin N, (1 : ℝ) :=
        Finset.sum_le_sum fun r _ => algorithm2Kernel_le_one hτ (anchors r) y
    _ = (N : ℝ) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one]

end KernelFacts

section KernelQuotient

variable {E : Type u} [NormedAddCommGroup E]

/-- Quotient form of the column-reweighted kernel: `k(x,y)/sqrt(g(y))`.  This
is the SNIS weight shape used throughout `Algorithm2SNIS.lean`. -/
theorem columnReweightedKernel_eq_kernel_div_sqrtMass {N : ℕ}
    (anchors : Fin N → E) (τ : ℝ) (x y : E) :
    columnReweightedKernel anchors τ x y =
      algorithm2Kernel τ x y /
        Real.sqrt (algorithm2ColumnKernelMass anchors τ y) := by
  unfold Algorithm2.columnReweightedKernel
  rw [show algorithm2Kernel τ x y *
        (algorithm2Kernel τ x y / algorithm2ColumnKernelMass anchors τ y)
      = algorithm2Kernel τ x y ^ 2 / algorithm2ColumnKernelMass anchors τ y by ring,
    Real.sqrt_div (sq_nonneg _), Real.sqrt_sq (algorithm2Kernel_nonneg τ x y)]

/-- At an anchor probe the column-reweighted kernel is at most one: the column
mass dominates its own anchor term, so `k/sqrt(g) ≤ sqrt(k) ≤ 1`. -/
theorem columnReweightedKernel_le_one_at_anchor {N : ℕ}
    (anchors : Fin N → E) {τ : ℝ} (hτ : ValidTemperature τ) (r : Fin N) (y : E) :
    columnReweightedKernel anchors τ (anchors r) y ≤ 1 := by
  have hk := algorithm2Kernel_pos τ (anchors r) y
  have hmem : algorithm2Kernel τ (anchors r) y ≤ algorithm2ColumnKernelMass anchors τ y := by
    unfold Algorithm2.algorithm2ColumnKernelMass
    exact Finset.single_le_sum
      (fun i _ => (algorithm2Kernel_pos τ (anchors i) y).le) (Finset.mem_univ r)
  have hsk : 0 < Real.sqrt (algorithm2Kernel τ (anchors r) y) := Real.sqrt_pos.mpr hk
  have hstep : 1 / Real.sqrt (algorithm2ColumnKernelMass anchors τ y)
      ≤ 1 / Real.sqrt (algorithm2Kernel τ (anchors r) y) :=
    one_div_le_one_div_of_le hsk (Real.sqrt_le_sqrt hmem)
  rw [columnReweightedKernel_eq_kernel_div_sqrtMass]
  calc algorithm2Kernel τ (anchors r) y /
        Real.sqrt (algorithm2ColumnKernelMass anchors τ y)
      = algorithm2Kernel τ (anchors r) y *
          (1 / Real.sqrt (algorithm2ColumnKernelMass anchors τ y)) := by
        rw [mul_one_div]
    _ ≤ algorithm2Kernel τ (anchors r) y *
          (1 / Real.sqrt (algorithm2Kernel τ (anchors r) y)) :=
        mul_le_mul_of_nonneg_left hstep hk.le
    _ = algorithm2Kernel τ (anchors r) y /
          Real.sqrt (algorithm2Kernel τ (anchors r) y) := by
        rw [mul_one_div]
    _ = Real.sqrt (algorithm2Kernel τ (anchors r) y) := Real.div_sqrt
    _ ≤ 1 := by
        rw [show (1 : ℝ) = Real.sqrt 1 from Real.sqrt_one.symm]
        exact Real.sqrt_le_sqrt (algorithm2Kernel_le_one hτ (anchors r) y)

end KernelQuotient

/-! ## Two-atom normalizers for a general kernel -/

/-- Closed form of the two-atom kernel normalizer.  This is the kernel-generic
statement behind `laplaceKernelNormalizer_empirical01`. -/
theorem kernelNormalizer_empirical01 (k : ℝ → ℝ → ℝ) (x : ℝ)
    (a : FiniteProbabilityVector 2) :
    kernelNormalizer k (empirical01Basis.basisMeasure a) x =
      a.weight 0 * k x 0 + a.weight 1 * k x 1 := by
  unfold kernelNormalizer
  rw [empirical01Basis.integral_basisMeasure_eq_density_smul, integral_empirical2]
  simp only [ProbabilityDensityBasis.mixtureDensity, basisDensity,
    empirical01Basis, Fin.sum_univ_two, empirical01Density_zero_zero,
    empirical01Density_one_zero, empirical01Density_zero_one,
    empirical01Density_one_one, mul_zero, add_zero, mul_two, zero_add,
    smul_eq_mul]
  ring

/-- A kernel positive at both atoms has a positive two-atom normalizer. -/
theorem kernelNormalizer_empirical01_pos (k : ℝ → ℝ → ℝ) (x : ℝ)
    (a : FiniteProbabilityVector 2) (hk0 : 0 < k x 0) (hk1 : 0 < k x 1) :
    0 < kernelNormalizer k (empirical01Basis.basisMeasure a) x := by
  rw [kernelNormalizer_empirical01]
  have hor : 0 < a.weight 0 ∨ 0 < a.weight 1 := by
    have h0 := a.nonnegative 0
    have h1 := a.nonnegative 1
    have hsum : a.weight 0 + a.weight 1 = 1 := by
      simpa [Fin.sum_univ_two] using a.normalized
    rcases lt_or_eq_of_le h0 with h0pos | h0zero
    · exact Or.inl h0pos
    · right
      nlinarith
  rcases hor with h0 | h1
  · exact add_pos_of_pos_of_nonneg (mul_pos h0 hk0)
      (mul_nonneg (a.nonnegative 1) hk1.le)
  · exact add_pos_of_nonneg_of_pos
      (mul_nonneg (a.nonnegative 0) hk0.le) (mul_pos h1 hk1)

/-- A kernel bounded by one at both atoms has a two-atom normalizer at most
one. -/
theorem kernelNormalizer_empirical01_le_one (k : ℝ → ℝ → ℝ) (x : ℝ)
    (a : FiniteProbabilityVector 2) (hk0 : k x 0 ≤ 1) (hk1 : k x 1 ≤ 1) :
    kernelNormalizer k (empirical01Basis.basisMeasure a) x ≤ 1 := by
  rw [kernelNormalizer_empirical01]
  calc a.weight 0 * k x 0 + a.weight 1 * k x 1
      ≤ a.weight 0 * 1 + a.weight 1 * 1 :=
        add_le_add
          (mul_le_mul_of_nonneg_left hk0 (a.nonnegative 0))
          (mul_le_mul_of_nonneg_left hk1 (a.nonnegative 1))
    _ = 1 := by simpa [Fin.sum_univ_two] using a.normalized

/-! ## Regularity of the column-reweighted kernel on the two-atom model -/

section Regularity

variable {N : ℕ} [Nonempty (Fin N)]

/-- The column-reweighted kernel is continuous in the sample slot: the column
mass is a positive continuous finite sum, so the quotient and square root are
continuous. -/
theorem columnReweightedKernel_continuous_snd
    (anchors : Fin N → ℝ) (τ x : ℝ) :
    Continuous fun y : ℝ => columnReweightedKernel anchors τ x y := by
  have hk : ∀ x' : ℝ, Continuous fun y : ℝ => algorithm2Kernel τ x' y := by
    intro x'
    unfold Algorithm2.algorithm2Kernel
    exact Real.continuous_exp.comp
      (((continuous_const.sub continuous_id).norm).neg.div_const τ)
  have hg : Continuous fun y : ℝ => algorithm2ColumnKernelMass anchors τ y := by
    unfold Algorithm2.algorithm2ColumnKernelMass
    exact continuous_finsetSum _ fun r _ => hk (anchors r)
  have hdiv : Continuous fun y : ℝ =>
      algorithm2Kernel τ x y *
        (algorithm2Kernel τ x y / algorithm2ColumnKernelMass anchors τ y) :=
    (hk x).mul ((hk x).div hg fun y =>
      (algorithm2ColumnKernelMass_pos anchors τ y).ne')
  unfold Algorithm2.columnReweightedKernel
  exact Real.continuous_sqrt.comp hdiv

/-- The paired interaction integrand of the column-reweighted kernel is
continuous. -/
theorem columnReweighted_interaction_continuous
    (anchors : Fin N → ℝ) (τ x : ℝ) :
    Continuous fun y : ℝ × ℝ =>
      meanShiftInteractionKernel (columnReweightedKernel anchors τ) x y.1 y.2 := by
  simp only [meanShiftInteractionKernel]
  exact ((((columnReweightedKernel_continuous_snd anchors τ x).comp continuous_fst).mul
    ((columnReweightedKernel_continuous_snd anchors τ x).comp continuous_snd)).smul
    (continuous_fst.sub continuous_snd))

/-- All mean-shift regularity obligations hold for the column-reweighted kernel
on the two-atom model, at every probe point. -/
theorem columnReweighted01_meanShiftRegular
    (anchors : Fin N → ℝ) (τ x : ℝ) (a b : FiniteProbabilityVector 2) :
    MeanShiftRegularAt (columnReweightedKernel anchors τ)
      (empirical01Basis.basisMeasure a)
      (empirical01Basis.basisMeasure b) x :=
  { zp_ne_zero := ne_of_gt (kernelNormalizer_empirical01_pos _ x a
      (columnReweightedKernel_pos anchors τ x 0)
      (columnReweightedKernel_pos anchors τ x 1))
    zq_ne_zero := ne_of_gt (kernelNormalizer_empirical01_pos _ x b
      (columnReweightedKernel_pos anchors τ x 0)
      (columnReweightedKernel_pos anchors τ x 1))
    integrable_p := empirical01Basis_integrable a _
    integrable_q := empirical01Basis_integrable b _
    integrable_product := empirical01Basis_integrable_prod a b _
      (columnReweighted_interaction_continuous anchors τ x).aestronglyMeasurable }

/-! ## The certified frame for the modified interaction vectors -/

/-- The actual integral-induced column-reweighted interaction vector is nonzero:
the closed form `basisInteraction_empirical2` exposes it as a positive multiple
of `(0 - 1)` because the modified kernel is strictly positive. -/
theorem inducedInteractionVector_columnReweighted01_ne_zero
    (anchors : Fin N → ℝ) (τ : ℝ) :
    inducedInteractionVector (empirical2 (0 : ℝ) 1)
      (meanShiftInteractionKernel (columnReweightedKernel anchors τ))
      empirical01Density anchors 0 1 ≠ 0 := by
  rw [Function.ne_iff]
  obtain ⟨n0⟩ := (inferInstance : Nonempty (Fin N))
  refine ⟨n0, ?_⟩
  simp only [inducedInteractionVector, basisInteraction_empirical2,
    empirical01Density_zero_zero, empirical01Density_one_one,
    empirical01Density_zero_one, empirical01Density_one_zero, Pi.zero_apply]
  apply smul_ne_zero
  · exact mul_ne_zero (by norm_num)
      (ne_of_gt (mul_pos (columnReweightedKernel_pos anchors τ (anchors n0) 0)
        (columnReweightedKernel_pos anchors τ (anchors n0) 1)))
  · norm_num

/-- **Exact bare-to-modified rescaling.**  Against the two-atom basis the
column-reweighted interaction vector equals the bare Algorithm-2 (Laplace)
interaction vector scaled by the explicit positive constant
`1/sqrt(g(0)·g(1))`, where `g` is the anchor column mass.  The reweighting
factor depends only on the sample slots, which the atoms pin to the support
points, so the identity is exact rather than perturbative. -/
theorem inducedInteractionVector_columnReweighted01_eq_smul
    (anchors : Fin N → ℝ) (τ : ℝ) :
    inducedInteractionVector (empirical2 (0 : ℝ) 1)
      (meanShiftInteractionKernel (columnReweightedKernel anchors τ))
      empirical01Density anchors 0 1 =
    (Real.sqrt (algorithm2ColumnKernelMass anchors τ 0 *
        algorithm2ColumnKernelMass anchors τ 1))⁻¹ •
      inducedInteractionVector (empirical2 (0 : ℝ) 1)
        (meanShiftInteractionKernel (algorithm2Kernel τ))
        empirical01Density anchors 0 1 := by
  have hg0 := algorithm2ColumnKernelMass_pos anchors τ (0 : ℝ)
  have hg1 := algorithm2ColumnKernelMass_pos anchors τ (1 : ℝ)
  have hs0 : Real.sqrt (algorithm2ColumnKernelMass anchors τ 0) ≠ 0 :=
    (Real.sqrt_pos.mpr hg0).ne'
  have hs1 : Real.sqrt (algorithm2ColumnKernelMass anchors τ 1) ≠ 0 :=
    (Real.sqrt_pos.mpr hg1).ne'
  funext n
  simp only [inducedInteractionVector, basisInteraction_empirical2,
    Pi.smul_apply, smul_smul]
  congr 1
  rw [columnReweightedKernel_eq_kernel_div_sqrtMass,
    columnReweightedKernel_eq_kernel_div_sqrtMass, Real.sqrt_mul hg0.le]
  field_simp

/-- Norm form of the rescaling: the sharp modified frame constant is exactly
the bare one divided by `sqrt(g(0)·g(1))`. -/
theorem columnReweighted01_interactionNorm_eq
    (anchors : Fin N → ℝ) (τ : ℝ) :
    ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
        (meanShiftInteractionKernel (columnReweightedKernel anchors τ))
        empirical01Density anchors 0 1‖ =
      (Real.sqrt (algorithm2ColumnKernelMass anchors τ 0 *
          algorithm2ColumnKernelMass anchors τ 1))⁻¹ *
        ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
          (meanShiftInteractionKernel (algorithm2Kernel τ))
          empirical01Density anchors 0 1‖ := by
  have hg0 := algorithm2ColumnKernelMass_pos anchors τ (0 : ℝ)
  have hg1 := algorithm2ColumnKernelMass_pos anchors τ (1 : ℝ)
  rw [inducedInteractionVector_columnReweighted01_eq_smul, norm_smul,
    Real.norm_eq_abs,
    abs_of_pos (inv_pos.mpr (Real.sqrt_pos.mpr (mul_pos hg0 hg1)))]

/-- **The column reweighting costs at most the anchor count.**  At positive
temperature the column mass is at most `N`, so the modified frame constant is
at least the bare constant divided by `N`.  This is the explicit conditioning
price of the column softmax. -/
theorem columnReweighted01_interactionNorm_ge
    (anchors : Fin N → ℝ) {τ : ℝ} (hτ : ValidTemperature τ) :
    ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
        (meanShiftInteractionKernel (algorithm2Kernel τ))
        empirical01Density anchors 0 1‖ / (N : ℝ) ≤
      ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
        (meanShiftInteractionKernel (columnReweightedKernel anchors τ))
        empirical01Density anchors 0 1‖ := by
  have hg0 := algorithm2ColumnKernelMass_pos anchors τ (0 : ℝ)
  have hg1 := algorithm2ColumnKernelMass_pos anchors τ (1 : ℝ)
  have hsqrt_pos : 0 < Real.sqrt (algorithm2ColumnKernelMass anchors τ 0 *
      algorithm2ColumnKernelMass anchors τ 1) :=
    Real.sqrt_pos.mpr (mul_pos hg0 hg1)
  have hprod_le : algorithm2ColumnKernelMass anchors τ 0 *
      algorithm2ColumnKernelMass anchors τ 1 ≤ (N : ℝ) * N :=
    mul_le_mul (algorithm2ColumnKernelMass_le_card anchors hτ 0)
      (algorithm2ColumnKernelMass_le_card anchors hτ 1)
      hg1.le (Nat.cast_nonneg N)
  have hsqrt_le : Real.sqrt (algorithm2ColumnKernelMass anchors τ 0 *
      algorithm2ColumnKernelMass anchors τ 1) ≤ (N : ℝ) := by
    calc Real.sqrt (algorithm2ColumnKernelMass anchors τ 0 *
          algorithm2ColumnKernelMass anchors τ 1)
        ≤ Real.sqrt ((N : ℝ) * N) := Real.sqrt_le_sqrt hprod_le
      _ = (N : ℝ) := Real.sqrt_mul_self (Nat.cast_nonneg N)
  have hinv : ((N : ℝ))⁻¹ ≤ (Real.sqrt (algorithm2ColumnKernelMass anchors τ 0 *
      algorithm2ColumnKernelMass anchors τ 1))⁻¹ := by
    simpa [one_div] using one_div_le_one_div_of_le hsqrt_pos hsqrt_le
  rw [columnReweighted01_interactionNorm_eq]
  calc ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
        (meanShiftInteractionKernel (algorithm2Kernel τ))
        empirical01Density anchors 0 1‖ / (N : ℝ)
      = ((N : ℝ))⁻¹ * ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
          (meanShiftInteractionKernel (algorithm2Kernel τ))
          empirical01Density anchors 0 1‖ := by
        rw [div_eq_mul_inv, mul_comm]
    _ ≤ (Real.sqrt (algorithm2ColumnKernelMass anchors τ 0 *
          algorithm2ColumnKernelMass anchors τ 1))⁻¹ *
        ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
          (meanShiftInteractionKernel (algorithm2Kernel τ))
          empirical01Density anchors 0 1‖ :=
        mul_le_mul_of_nonneg_right hinv (norm_nonneg _)

/-- **Strict-pair scaling transfer, instantiated.**  Any certified frame bound
for the bare Algorithm-2 kernel interaction transfers to the column-reweighted
interaction with the explicit scale `1/sqrt(g(0)·g(1))`, via
`interactionFrameBound_of_strictPairScaling`.  For `m = 2` the direct
`interactionFrameBound_two` constant is sharper; this records the announced
bare-to-modified transfer route explicitly. -/
theorem columnReweighted01_frameBound_of_bare
    (anchors : Fin N → ℝ) (τ : ℝ) {c : ℝ}
    (hbare : InteractionFrameBound
      (inducedInteractionVector (empirical2 (0 : ℝ) 1)
        (meanShiftInteractionKernel (algorithm2Kernel τ))
        empirical01Density anchors) c) :
    InteractionFrameBound
      (inducedInteractionVector (empirical2 (0 : ℝ) 1)
        (meanShiftInteractionKernel (columnReweightedKernel anchors τ))
        empirical01Density anchors)
      ((Real.sqrt (algorithm2ColumnKernelMass anchors τ 0 *
          algorithm2ColumnKernelMass anchors τ 1))⁻¹ * c) := by
  have hg0 := algorithm2ColumnKernelMass_pos anchors τ (0 : ℝ)
  have hg1 := algorithm2ColumnKernelMass_pos anchors τ (1 : ℝ)
  have hs : 0 < (Real.sqrt (algorithm2ColumnKernelMass anchors τ 0 *
      algorithm2ColumnKernelMass anchors τ 1))⁻¹ :=
    inv_pos.mpr (Real.sqrt_pos.mpr (mul_pos hg0 hg1))
  refine interactionFrameBound_of_strictPairScaling _ _
    (fun _ => (Real.sqrt (algorithm2ColumnKernelMass anchors τ 0 *
      algorithm2ColumnKernelMass anchors τ 1))⁻¹)
    hbare hs (fun p => by rw [abs_of_pos hs]) (fun p => ?_)
  have hp : p = (⟨(0, 1), by decide⟩ : StrictPair 2) := Subsingleton.elim p _
  rw [hp]
  exact inducedInteractionVector_columnReweighted01_eq_smul anchors τ

/-! ## The concrete certified population setup -/

/-- **Concrete certified column-reweighted setup.**  The two-atom empirical
basis on `{0, 1}`, an arbitrary nonempty anchor family, and any temperature:
every analytic field is discharged, and the frame bound for the actual
column-reweighted interaction vectors is certified directly by kernel
positivity through `interactionFrameBound_two`. -/
noncomputable def columnReweighted01Setup
    (anchors : Fin N → ℝ) (τ : ℝ) (a b : FiniteProbabilityVector 2) :
    ColumnReweightedMeanShiftFiniteSetup ℝ 2 N where
  reference := empirical2 0 1
  refProb := empirical2_isProbability 0 1
  basis := empirical01Basis
  anchors := anchors
  temperature := τ
  a := a
  b := b
  meanShiftRegular n := columnReweighted01_meanShiftRegular anchors τ (anchors n) a b
  interactionIntegrable n := empirical01Basis_integrable_prod a b _
    (columnReweighted_interaction_continuous anchors τ (anchors n)).aestronglyMeasurable
  basisInteractionIntegrable i j n := by
    apply integrable_empirical2_prod
    apply Measurable.aestronglyMeasurable
    have hi : Measurable fun y : ℝ × ℝ => empirical01Density i y.1 :=
      (empirical01Basis.measurable_density i).comp measurable_fst
    have hj : Measurable fun y : ℝ × ℝ => empirical01Density j y.2 :=
      (empirical01Basis.measurable_density j).comp measurable_snd
    have hK : Measurable fun y : ℝ × ℝ =>
        meanShiftInteractionKernel (columnReweightedKernel anchors τ)
          (anchors n) y.1 y.2 :=
      (columnReweighted_interaction_continuous anchors τ (anchors n)).measurable
    exact (hi.mul hj).smul hK
  frameConstant :=
    ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
      (meanShiftInteractionKernel (columnReweightedKernel anchors τ))
      empirical01Density anchors 0 1‖
  frameBound := interactionFrameBound_two _
    (inducedInteractionVector_columnReweighted01_ne_zero anchors τ)

/-- At positive temperature the concrete normalizer product satisfies `B = 1`:
the probes are the anchors themselves, where the column-reweighted kernel is at
most one. -/
theorem columnReweighted01_normalizerProduct_abs_le_one
    (anchors : Fin N → ℝ) {τ : ℝ} (hτ : ValidTemperature τ)
    (a b : FiniteProbabilityVector 2) (n : Fin N) :
    |(columnReweighted01Setup anchors τ a b).normalizerProduct n| ≤ 1 := by
  have hpa := kernelNormalizer_empirical01_pos (columnReweightedKernel anchors τ)
    (anchors n) a (columnReweightedKernel_pos anchors τ (anchors n) 0)
    (columnReweightedKernel_pos anchors τ (anchors n) 1)
  have hpb := kernelNormalizer_empirical01_pos (columnReweightedKernel anchors τ)
    (anchors n) b (columnReweightedKernel_pos anchors τ (anchors n) 0)
    (columnReweightedKernel_pos anchors τ (anchors n) 1)
  have hla := kernelNormalizer_empirical01_le_one (columnReweightedKernel anchors τ)
    (anchors n) a (columnReweightedKernel_le_one_at_anchor anchors hτ n 0)
    (columnReweightedKernel_le_one_at_anchor anchors hτ n 1)
  have hlb := kernelNormalizer_empirical01_le_one (columnReweightedKernel anchors τ)
    (anchors n) b (columnReweightedKernel_le_one_at_anchor anchors hτ n 0)
    (columnReweightedKernel_le_one_at_anchor anchors hτ n 1)
  change |kernelNormalizer (columnReweightedKernel anchors τ)
      (empirical01Basis.basisMeasure a) (anchors n) *
    kernelNormalizer (columnReweightedKernel anchors τ)
      (empirical01Basis.basisMeasure b) (anchors n)| ≤ 1
  rw [abs_of_pos (mul_pos hpa hpb)]
  exact mul_le_one₀ hla hpb.le hlb

/-! ## End-to-end promoted theorems for the concrete class -/

/-- **Concrete column-reweighted identifiability.**  Zero finite probe energy of
the limiting no-mask Algorithm-2 centroid field, at any nonempty anchor family
and any temperature, identifies the two-atom mixtures.  The frame hypothesis is
not assumed: it is certified inside `columnReweighted01Setup`. -/
theorem columnReweighted01_identifies_of_probeEnergy_eq_zero
    (anchors : Fin N → ℝ) (τ : ℝ) (a b : FiniteProbabilityVector 2)
    (henergy :
      (columnReweighted01Setup anchors τ a b).modifiedProbeDriftEnergy = 0) :
    empirical01Basis.basisMeasure a = empirical01Basis.basisMeasure b :=
  (columnReweighted01Setup anchors τ a b).identifies_of_probeEnergy_eq_zero henergy

/-- Quantitative stability for the concrete class with `B = 1`. -/
theorem columnReweighted01_coefficientStability
    (anchors : Fin N → ℝ) {τ : ℝ} (hτ : ValidTemperature τ)
    (a b : FiniteProbabilityVector 2) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 / (columnReweighted01Setup anchors τ a b).frameConstant) *
        ‖(columnReweighted01Setup anchors τ a b).modifiedProbeDrift‖ := by
  have h := (columnReweighted01Setup anchors τ a b).coefficientStability
    (B := 1) (by norm_num)
    (columnReweighted01_normalizerProduct_abs_le_one anchors hτ a b)
  simpa only [columnReweighted01Setup, mul_one] using h

/-- **Concrete finite-sample bridge.**  For any random estimator `Vhat` of the
column-reweighted population field — in particular the no-mask Algorithm-2
centroid difference controlled by the SNIS mean-square bounds — the probability
that the `B = 1` coefficient bound fails is at most `MSE/ε²`. -/
theorem columnReweighted01_estimate_failure_le_meanSquare
    (anchors : Fin N → ℝ) {τ : ℝ} (hτ : ValidTemperature τ)
    (a b : FiniteProbabilityVector 2) {ε : ℝ} (hε : 0 < ε)
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (Vhat : Ω → (Fin N → ℝ))
    (hmeas : AEStronglyMeasurable
      (fun ω =>
        (columnReweighted01Setup anchors τ a b).modifiedProbeDrift - Vhat ω) P)
    (hint : Integrable
      (fun ω =>
        ‖(columnReweighted01Setup anchors τ a b).modifiedProbeDrift -
          Vhat ω‖ ^ 2) P) :
    P {ω | (2 / (columnReweighted01Setup anchors τ a b).frameConstant) *
        (‖Vhat ω‖ + ε) < ∑ i, |a.weight i - b.weight i|}
      ≤ ENNReal.ofReal
          ((∫ ω, ‖(columnReweighted01Setup anchors τ a b).modifiedProbeDrift -
            Vhat ω‖ ^ 2 ∂P) / ε ^ 2) := by
  have h := (columnReweighted01Setup anchors τ a b).estimate_failure_le_meanSquare
    (B := 1) (by norm_num)
    (columnReweighted01_normalizerProduct_abs_le_one anchors hτ a b) hε P Vhat hmeas hint
  simpa only [columnReweighted01Setup, mul_one] using h

end Regularity

end PaperFiniteIdentifiability
end DriftingIdentifiability
