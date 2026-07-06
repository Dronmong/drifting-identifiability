import DriftingIdentifiability.PopulationIdentifiability

/-!
# Closed-form interaction vectors for empirical references

The finite population theorem assumes a positive `InteractionFrameBound` for the
paper's *actual* integral-induced interaction vectors. Here we first compute the
double integral for arbitrary finite injective empirical point bases:

```text
Uᵢⱼ(n) = k(xₙ,zᵢ) · k(xₙ,zⱼ) · (zᵢ − zⱼ).
```

For `E = ℝ`, unit-bandwidth Gaussian kernel, one integer probe per strict pair,
and distinct pairwise sums `zᵢ+zⱼ`, a Vandermonde argument proves these actual
vectors linearly independent without external axioms. The file also retains a
fully explicit two-point specialization for arbitrary positive bandwidth.
-/

open scoped BigOperators ENNReal
open MeasureTheory

namespace DriftingIdentifiability
namespace PaperFiniteIdentifiability

open Paper

universe u

variable {E : Type u} [MeasurableSpace E] [MeasurableSingletonClass E]
  [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- The uniform empirical distribution on two points. -/
noncomputable def empirical2 (z0 z1 : E) : Distribution E :=
  (2⁻¹ : ℝ≥0∞) • (Measure.dirac z0 + Measure.dirac z1)

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Integral against the two-point empirical distribution is the average of the
two point evaluations. -/
theorem integral_empirical2 {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    [CompleteSpace F] (z0 z1 : E) (f : E → F) :
    ∫ y, f y ∂(empirical2 z0 z1) = (2⁻¹ : ℝ) • (f z0 + f z1) := by
  rw [empirical2, integral_smul_measure,
    integral_add_measure (integrable_dirac (by simp)) (integrable_dirac (by simp)),
    integral_dirac, integral_dirac]
  simp

omit [MeasurableSingletonClass E] [NormedAddCommGroup E] [NormedSpace ℝ E]
  [CompleteSpace E] in
/-- The two-point empirical reference really is a probability measure. -/
theorem empirical2_isProbability (z0 z1 : E) :
    IsProbabilityMeasure (empirical2 z0 z1) := by
  refine ⟨?_⟩
  simpa [empirical2] using ENNReal.inv_two_add_inv_two

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Every finite-valued function is integrable against the two-point
empirical reference. -/
theorem integrable_empirical2 {F : Type*} [NormedAddCommGroup F]
    [NormedSpace ℝ F] [CompleteSpace F] (z0 z1 : E) (f : E → F) :
    Integrable f (empirical2 z0 z1) := by
  rw [empirical2]
  exact (integrable_add_measure.2
    ⟨integrable_dirac (by simp), integrable_dirac (by simp)⟩).smul_measure (by simp)

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Product integrability for the finite empirical reference. -/
theorem integrable_empirical2_prod
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]
    (z0 z1 : E) (f : E × E → F)
    (hf : AEStronglyMeasurable f ((empirical2 z0 z1).prod (empirical2 z0 z1))) :
    Integrable f ((empirical2 z0 z1).prod (empirical2 z0 z1)) := by
  letI := empirical2_isProbability z0 z1
  rw [integrable_prod_iff hf]
  exact ⟨Filter.Eventually.of_forall fun x =>
      integrable_empirical2 z0 z1 (fun y => f (x, y)),
    integrable_empirical2 z0 z1 (fun x =>
      ∫ y, ‖f (x, y)‖ ∂empirical2 z0 z1)⟩

/-! ## Uniform empirical references of arbitrary finite size -/

/-- Uniform empirical measure on a finite indexed support. -/
noncomputable def empiricalFin {m : ℕ} (z : Fin m → E) : Distribution E :=
  ((m : ℝ≥0∞)⁻¹) • ∑ i, Measure.dirac (z i)

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Integration against a nonempty uniform empirical measure is finite
averaging. -/
theorem integral_empiricalFin {F : Type*} [NormedAddCommGroup F]
    [NormedSpace ℝ F] [CompleteSpace F] {m : ℕ}
    (z : Fin m → E) (f : E → F) :
    ∫ y, f y ∂empiricalFin z =
      (m : ℝ)⁻¹ • ∑ i, f (z i) := by
  rw [empiricalFin, integral_smul_measure,
    integral_finsetSum_measure (s := Finset.univ)]
  · simp
  · simp [integrable_dirac]

omit [MeasurableSingletonClass E] [NormedAddCommGroup E] [NormedSpace ℝ E]
  [CompleteSpace E] in
/-- A nonempty uniform empirical measure has total mass one. -/
theorem empiricalFin_isProbability {m : ℕ} (hm : 0 < m) (z : Fin m → E) :
    IsProbabilityMeasure (empiricalFin z) := by
  refine ⟨?_⟩
  have hm0 : (m : ℝ≥0∞) ≠ 0 := by
    apply ne_of_gt
    exact_mod_cast hm
  have hcancel := ENNReal.inv_mul_cancel (a := (m : ℝ≥0∞))
    hm0 (ENNReal.natCast_ne_top m)
  simpa [empiricalFin] using hcancel

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Every finite-valued function is integrable against a finite empirical
reference. -/
theorem integrable_empiricalFin
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]
    {m : ℕ} (hm : 0 < m) (z : Fin m → E) (f : E → F) :
    Integrable f (empiricalFin z) := by
  rw [empiricalFin]
  apply Integrable.smul_measure
  · rw [integrable_finsetSum_measure]
    simp [integrable_dirac]
  · apply (ENNReal.inv_ne_top).2
    apply ne_of_gt
    exact_mod_cast hm

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Product integrability for a nonempty finite empirical reference. -/
theorem integrable_empiricalFin_prod
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]
    {m : ℕ} (hm : 0 < m) (z : Fin m → E) (f : E × E → F)
    (hf : AEStronglyMeasurable f ((empiricalFin z).prod (empiricalFin z))) :
    Integrable f ((empiricalFin z).prod (empiricalFin z)) := by
  letI := empiricalFin_isProbability hm z
  rw [integrable_prod_iff hf]
  exact ⟨Filter.Eventually.of_forall fun x =>
      integrable_empiricalFin hm z (fun y => f (x, y)),
    integrable_empiricalFin hm z (fun x =>
      ∫ y, ‖f (x, y)‖ ∂empiricalFin z)⟩

/-- Atomic basis density relative to the uniform empirical reference. -/
noncomputable def empiricalPointDensity {m : ℕ} (z : Fin m → E)
    (i : Fin m) : E → ℝ :=
  ({z i} : Set E).indicator (fun _ => (m : ℝ))

omit [MeasurableSpace E] [MeasurableSingletonClass E] [NormedAddCommGroup E]
  [NormedSpace ℝ E] [CompleteSpace E] in
/-- Evaluation table of the atomic densities on an injective support. -/
theorem empiricalPointDensity_apply_support {m : ℕ} (z : Fin m → E)
    (hz : Function.Injective z) (i r : Fin m) :
    empiricalPointDensity z i (z r) = if r = i then (m : ℝ) else 0 := by
  by_cases hri : r = i
  · subst r
    simp [empiricalPointDensity]
  · have hzri : z r ≠ z i := fun h => hri (hz h)
    simp [empiricalPointDensity, hri, hzri]

/-- Distinct empirical support points give a genuine unit-mass point-density
basis. -/
noncomputable def empiricalPointBasis {m : ℕ} (hm : 0 < m)
    (z : Fin m → E) (hz : Function.Injective z) :
    ProbabilityDensityBasis (empiricalFin z) m where
  density := empiricalPointDensity z
  measurable_density i :=
    measurable_const.indicator (measurableSet_singleton (z i))
  nonnegative i x := by
    classical
    simp only [empiricalPointDensity, Set.indicator_apply]
    split_ifs <;> positivity
  integrable_density i := integrable_empiricalFin hm z (empiricalPointDensity z i)
  integral_density i := by
    rw [integral_empiricalFin]
    simp_rw [empiricalPointDensity_apply_support z hz i]
    simp only [Finset.sum_ite_eq', Finset.mem_univ, if_true, smul_eq_mul]
    have hm0 : (m : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hm)
    exact inv_mul_cancel₀ hm0

/-- Closed form of the actual interaction vectors for an arbitrary nonempty
injective empirical point basis.  The density and reference normalizations
cancel exactly. -/
theorem basisInteraction_empiricalPoint
    {m : ℕ} (hm : 0 < m) (z : Fin m → E) (hz : Function.Injective z)
    (K : E → E → E → E) (i j : Fin m) (x : E) :
    basisInteraction (empiricalFin z) K
      (empiricalPointDensity z i) (empiricalPointDensity z j) x =
        K x (z i) (z j) := by
  unfold basisInteraction
  rw [integral_empiricalFin]
  simp_rw [integral_empiricalFin]
  simp_rw [empiricalPointDensity_apply_support z hz]
  have hm0 : (m : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hm)
  simp [smul_smul, hm0]

/-- The general empirical point-basis interaction vectors for the paper's
mean-shift kernel. -/
theorem inducedInteractionVector_empiricalPoint
    {m N : ℕ} (hm : 0 < m) (z : Fin m → E) (hz : Function.Injective z)
    (k : E → E → ℝ) (probes : Fin N → E) (i j : Fin m) :
    inducedInteractionVector (empiricalFin z) (meanShiftInteractionKernel k)
      (empiricalPointDensity z) probes i j =
        fun n => (k (probes n) (z i) * k (probes n) (z j)) • (z i - z j) := by
  funext n
  exact basisInteraction_empiricalPoint hm z hz
    (meanShiftInteractionKernel k) i j (probes n)

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Mixture-density evaluation on its empirical support. -/
theorem empiricalPointBasis_mixtureDensity_apply
    {m : ℕ} (hm : 0 < m) (z : Fin m → E) (hz : Function.Injective z)
    (a : FiniteProbabilityVector m) (r : Fin m) :
    (empiricalPointBasis hm z hz).mixtureDensity a (z r) =
      (m : ℝ) * a.weight r := by
  simp only [ProbabilityDensityBasis.mixtureDensity, basisDensity,
    empiricalPointBasis, empiricalPointDensity_apply_support z hz,
    mul_ite, mul_zero]
  rw [Finset.sum_eq_single r]
  · simp [mul_comm]
  · intro b _ hbr
    simp [Ne.symm hbr]
  · simp

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Every finite-valued function remains integrable after applying a simplex
mixture density to the empirical reference. -/
theorem empiricalPointBasis_integrable
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]
    {m : ℕ} (hm : 0 < m) (z : Fin m → E) (hz : Function.Injective z)
    (a : FiniteProbabilityVector m) (f : E → F) :
    Integrable f ((empiricalPointBasis hm z hz).basisMeasure a) := by
  unfold ProbabilityDensityBasis.basisMeasure
  rw [integrable_withDensity_iff_integrable_smul'
    ((empiricalPointBasis hm z hz).measurable_mixtureDensity a).ennreal_ofReal
    (Filter.Eventually.of_forall fun _ => ENNReal.ofReal_lt_top)]
  exact integrable_empiricalFin hm z _

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Product integrability for two arbitrary empirical point-basis mixtures. -/
theorem empiricalPointBasis_integrable_prod
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]
    {m : ℕ} (hm : 0 < m) (z : Fin m → E) (hz : Function.Injective z)
    (a b : FiniteProbabilityVector m) (f : E × E → F)
    (hf : AEStronglyMeasurable f
      (((empiricalPointBasis hm z hz).basisMeasure a).prod
        ((empiricalPointBasis hm z hz).basisMeasure b))) :
    Integrable f (((empiricalPointBasis hm z hz).basisMeasure a).prod
      ((empiricalPointBasis hm z hz).basisMeasure b)) := by
  letI := (empiricalPointBasis hm z hz).basisMeasure_isProbability a
  letI := (empiricalPointBasis hm z hz).basisMeasure_isProbability b
  rw [integrable_prod_iff hf]
  exact ⟨Filter.Eventually.of_forall fun x =>
      empiricalPointBasis_integrable hm z hz b (fun y => f (x, y)),
    empiricalPointBasis_integrable hm z hz a (fun x =>
      ∫ y, ‖f (x, y)‖ ∂(empiricalPointBasis hm z hz).basisMeasure b)⟩

/-- Gaussian normalizer of a general empirical point-basis mixture. -/
theorem gaussianKernelNormalizer_empiricalPoint
    {m : ℕ} (hm : 0 < m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (σ x : ℝ) (a : FiniteProbabilityVector m) :
    kernelNormalizer (gaussianKernel σ)
      ((empiricalPointBasis hm z hz).basisMeasure a) x =
        ∑ r, a.weight r * gaussianKernel σ x (z r) := by
  unfold kernelNormalizer
  rw [(empiricalPointBasis hm z hz).integral_basisMeasure_eq_density_smul,
    integral_empiricalFin]
  simp_rw [empiricalPointBasis_mixtureDensity_apply hm z hz a]
  simp only [smul_eq_mul]
  have hm0 : (m : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hm)
  rw [show (∑ r, (m : ℝ) * a.weight r * gaussianKernel σ x (z r)) =
      (m : ℝ) * ∑ r, a.weight r * gaussianKernel σ x (z r) by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro r _
    ring]
  rw [inv_mul_cancel_left₀ hm0]

/-- The general empirical Gaussian normalizer is strictly positive. -/
theorem gaussianKernelNormalizer_empiricalPoint_pos
    {m : ℕ} (hm : 0 < m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (σ x : ℝ) (a : FiniteProbabilityVector m) :
    0 < kernelNormalizer (gaussianKernel σ)
      ((empiricalPointBasis hm z hz).basisMeasure a) x := by
  rw [gaussianKernelNormalizer_empiricalPoint hm z hz]
  have hexists : ∃ i : Fin m, 0 < a.weight i := by
    by_contra h
    push Not at h
    have hzero : a.weight = 0 := by
      funext i
      exact le_antisymm (h i) (a.nonnegative i)
    have hnorm := a.normalized
    rw [hzero] at hnorm
    simp at hnorm
  rcases hexists with ⟨i, hi⟩
  apply Finset.sum_pos'
  · intro r _
    exact mul_nonneg (a.nonnegative r) (Real.exp_pos _).le
  · refine ⟨i, Finset.mem_univ i, ?_⟩
    exact mul_pos hi (Real.exp_pos _)

/-- Unit-bandwidth Gaussian values are at most one. -/
theorem gaussianKernel_one_le_one (x y : ℝ) : gaussianKernel 1 x y ≤ 1 := by
  rw [gaussianKernel, Real.exp_le_one_iff]
  norm_num
  positivity

/-- Hence every unit-bandwidth empirical mixture normalizer is at most one. -/
theorem gaussianKernelNormalizer_empiricalPoint_le_one
    {m : ℕ} (hm : 0 < m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (x : ℝ) (a : FiniteProbabilityVector m) :
    kernelNormalizer (gaussianKernel 1)
      ((empiricalPointBasis hm z hz).basisMeasure a) x ≤ 1 := by
  rw [gaussianKernelNormalizer_empiricalPoint hm z hz]
  calc
    (∑ r, a.weight r * gaussianKernel 1 x (z r)) ≤
        ∑ r, a.weight r * 1 := by
      apply Finset.sum_le_sum
      intro r _
      exact mul_le_mul_of_nonneg_left (gaussianKernel_one_le_one x (z r))
        (a.nonnegative r)
    _ = 1 := by simpa using a.normalized

/-! ## Structured Gaussian probes on the real line -/

/-- One probe for each strict pair, placed at successive nonnegative
integers. -/
def structuredGaussianProbes (m : ℕ) :
    Fin (Fintype.card (StrictPair m)) → ℝ := fun n => (n : ℝ)

/-- Geometric base associated with a pair of support points. -/
noncomputable def gaussianPairBase {m : ℕ} (z : Fin m → ℝ)
    (p : StrictPair m) : ℝ :=
  Real.exp (z p.1.1 + z p.1.2)

/-- Nonzero column factor in the Gaussian product decomposition. -/
noncomputable def gaussianPairColumn {m : ℕ} (z : Fin m → ℝ)
    (p : StrictPair m) : ℝ :=
  Real.exp (-(1 / 2 : ℝ) * (z p.1.1 ^ 2 + z p.1.2 ^ 2)) *
    (z p.1.1 - z p.1.2)

/-- Nonzero row factor in the Gaussian product decomposition. -/
noncomputable def gaussianProbeRow {m : ℕ}
    (n : Fin (Fintype.card (StrictPair m))) : ℝ :=
  Real.exp (-((n : ℝ) ^ 2))

/-- At unit bandwidth and integer probes, every Gaussian product interaction
is a nonzero row factor times a nonzero column factor times a geometric
profile. -/
theorem gaussianInteraction_eq_weightedGeometric
    {m : ℕ} (z : Fin m → ℝ) (p : StrictPair m)
    (n : Fin (Fintype.card (StrictPair m))) :
    meanShiftInteractionKernel (gaussianKernel 1)
      (structuredGaussianProbes m n) (z p.1.1) (z p.1.2) =
        gaussianProbeRow n *
          (gaussianPairColumn z p * gaussianPairBase z p ^ (n : ℕ)) := by
  simp only [meanShiftInteractionKernel, gaussianKernel, structuredGaussianProbes,
    gaussianProbeRow, gaussianPairColumn, gaussianPairBase, one_pow, mul_one,
    Real.norm_eq_abs, sq_abs, smul_eq_mul]
  let t : ℝ := n
  let u : ℝ := z p.1.1
  let v : ℝ := z p.1.2
  have hexponent :
      -(1 / (2 : ℝ)) * (t - u) ^ 2 + -(1 / 2 : ℝ) * (t - v) ^ 2 =
        -(t ^ 2) + (-(1 / 2 : ℝ) * (u ^ 2 + v ^ 2) + t * (u + v)) := by
    ring
  change Real.exp (-(1 / (2 : ℝ)) * (t - u) ^ 2) *
      Real.exp (-(1 / 2 : ℝ) * (t - v) ^ 2) * (u - v) = _
  calc
    Real.exp (-(1 / (2 : ℝ)) * (t - u) ^ 2) *
          Real.exp (-(1 / 2 : ℝ) * (t - v) ^ 2) * (u - v) =
        Real.exp (-(1 / (2 : ℝ)) * (t - u) ^ 2 +
          -(1 / 2 : ℝ) * (t - v) ^ 2) * (u - v) := by
            rw [Real.exp_add]
    _ = Real.exp (-(t ^ 2) +
          (-(1 / 2 : ℝ) * (u ^ 2 + v ^ 2) + t * (u + v))) * (u - v) := by
            rw [hexponent]
    _ = Real.exp (-(t ^ 2)) *
          (Real.exp (-(1 / 2 : ℝ) * (u ^ 2 + v ^ 2)) * (u - v) *
            Real.exp (u + v) ^ (n : ℕ)) := by
      rw [Real.exp_add, Real.exp_add, ← Real.exp_nat_mul]
      ring

/-- Independently checkable Sidon-type condition: different strict pairs have
different pairwise sums. -/
def DistinctStrictPairSums {m : ℕ} (z : Fin m → ℝ) : Prop :=
  Function.Injective (fun p : StrictPair m => z p.1.1 + z p.1.2)

/-- Regression: distinct points alone are insufficient.  Four equally spaced
points have the collision `z₀+z₃ = z₁+z₂`. -/
theorem equallySpacedFour_not_distinctStrictPairSums :
    ¬ DistinctStrictPairSums (fun i : Fin 4 => (i : ℝ)) := by
  intro h
  let p : StrictPair 4 := ⟨(0, 3), by decide⟩
  let q : StrictPair 4 := ⟨(1, 2), by decide⟩
  have hpq : p ≠ q := by decide
  apply hpq
  apply h
  norm_num [p, q]

/-- A concrete nonvacuity witness for the geometric hypotheses. -/
def sidonSupportThree : Fin 3 → ℝ := ![0, 1, 3]

theorem sidonSupportThree_injective : Function.Injective sidonSupportThree := by
  intro i j hij
  fin_cases i <;> fin_cases j <;>
    try { simp [sidonSupportThree] at hij ⊢ }

theorem sidonSupportThree_distinctStrictPairSums :
    DistinctStrictPairSums sidonSupportThree := by
  intro p q hpq
  fin_cases p <;> fin_cases q <;>
    try { norm_num [sidonSupportThree] at hpq ⊢ } <;>
    try { simp [sidonSupportThree] at hpq ⊢ }

/-- **Axiom-free general-`m` nondegeneracy.**  Distinct support points with
distinct strict-pair sums, unit-bandwidth Gaussian kernel, and one structured
probe per strict pair produce linearly independent *actual integral-induced*
interaction vectors. -/
theorem gaussianEmpiricalPoint_basisNondegenerate
    {m : ℕ} (hm : 0 < m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) :
    BasisInteractionNondegenerate
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel 1))
        (empiricalPointDensity z) (structuredGaussianProbes m)) := by
  have hbase : Function.Injective (gaussianPairBase z) := by
    intro p q hpq
    apply hsums
    exact Real.exp_injective hpq
  have hcolumn : ∀ p : StrictPair m, gaussianPairColumn z p ≠ 0 := by
    intro p
    apply mul_ne_zero (ne_of_gt (Real.exp_pos _))
    apply sub_ne_zero.mpr
    exact fun h => (Fin.ne_of_lt p.property) (hz h)
  have hrow : ∀ n : Fin (Fintype.card (StrictPair m)), gaussianProbeRow n ≠ 0 :=
    fun n => ne_of_gt (Real.exp_pos _)
  have hgeom := linearIndependent_weightedGeometricProfiles
    (gaussianPairBase z) (gaussianPairColumn z) hbase hcolumn gaussianProbeRow hrow
  have hfamily :
      (fun p : StrictPair m =>
        inducedInteractionVector (empiricalFin z)
          (meanShiftInteractionKernel (gaussianKernel 1))
          (empiricalPointDensity z) (structuredGaussianProbes m) p.1.1 p.1.2) =
      (fun p : StrictPair m => fun n =>
        gaussianProbeRow n *
          (gaussianPairColumn z p * gaussianPairBase z p ^ (n : ℕ))) := by
    funext p n
    change basisInteraction (empiricalFin z)
      (meanShiftInteractionKernel (gaussianKernel 1))
      (empiricalPointDensity z p.1.1) (empiricalPointDensity z p.1.2)
      (structuredGaussianProbes m n) = _
    rw [basisInteraction_empiricalPoint hm z hz]
    exact gaussianInteraction_eq_weightedGeometric z p n
  unfold BasisInteractionNondegenerate
  rw [hfamily]
  exact hgeom

/-- Explicit inverse-matrix lower frame constant for the concrete structured
Gaussian interaction system. Every entry is determined by `z`, the unit
bandwidth kernel, and the prescribed integer probes. -/
noncomputable def gaussianEmpiricalPointCertifiedFrameConstant
    {m : ℕ} (z : Fin m → ℝ) : ℝ :=
  (inverseInteractionCertificateMass
    (inducedInteractionVector (empiricalFin z)
      (meanShiftInteractionKernel (gaussianKernel 1))
      (empiricalPointDensity z) (structuredGaussianProbes m)))⁻¹

/-- The explicit inverse-matrix constant is positive under the concrete
support separation conditions. -/
theorem gaussianEmpiricalPointCertifiedFrameConstant_pos
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) :
    0 < gaussianEmpiricalPointCertifiedFrameConstant z := by
  let p : StrictPair m :=
    ⟨(⟨0, by omega⟩, ⟨1, by omega⟩), by simp⟩
  letI : Nonempty (StrictPair m) := ⟨p⟩
  exact inv_pos.mpr (inverseInteractionCertificateMass_pos _
    (gaussianEmpiricalPoint_basisNondegenerate (by omega) z hz hsums))

/-- **Certified concrete lower bound (Objective 1).** The reciprocal
entrywise `ℓ¹` mass of the inverse concrete interaction matrix satisfies the
frame inequality. Thus the earlier existential constant has been replaced by
a finite formula suitable for numerical or interval certification. -/
theorem gaussianEmpiricalPointCertifiedFrameBound
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) :
    InteractionFrameBound
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel 1))
        (empiricalPointDensity z) (structuredGaussianProbes m))
      (gaussianEmpiricalPointCertifiedFrameConstant z) := by
  let p : StrictPair m :=
    ⟨(⟨0, by omega⟩, ⟨1, by omega⟩), by simp⟩
  letI : Nonempty (StrictPair m) := ⟨p⟩
  exact interactionFrameBound_inverseCertificate _
    (gaussianEmpiricalPoint_basisNondegenerate (by omega) z hz hsums)

/-- The preceding concrete nondegeneracy theorem yields a positive frame
constant for every nontrivial (`m ≥ 2`) empirical family.  The constant exists
axiom-freely; obtaining a sharp numerical lower bound remains a conditioning
problem. -/
theorem gaussianEmpiricalPoint_exists_frameBound
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) :
    ∃ c > 0, InteractionFrameBound
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel 1))
        (empiricalPointDensity z) (structuredGaussianProbes m)) c := by
  let p : StrictPair m :=
    ⟨(⟨0, by omega⟩, ⟨1, by omega⟩), by simp⟩
  letI : Nonempty (StrictPair m) := ⟨p⟩
  apply interactionFrameBound_of_linearIndependent
  exact gaussianEmpiricalPoint_basisNondegenerate (by omega) z hz hsums

/-- A canonical (noncomputably selected) positive frame constant for the
structured Gaussian construction. -/
noncomputable def gaussianEmpiricalPointFrameConstant
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) : ℝ :=
  Classical.choose (gaussianEmpiricalPoint_exists_frameBound hm z hz hsums)

/-- The selected general-`m` constant satisfies the frame inequality. -/
theorem gaussianEmpiricalPointFrameBound
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) :
    InteractionFrameBound
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel 1))
        (empiricalPointDensity z) (structuredGaussianProbes m))
      (gaussianEmpiricalPointFrameConstant hm z hz hsums) :=
  (Classical.choose_spec
    (gaussianEmpiricalPoint_exists_frameBound hm z hz hsums)).2

/-! ## An explicit computable ceiling on the frame constant (Objective 1)

The frame constant produced by `interactionFrameBound_of_linearIndependent` is
selected noncomputably.  Testing the frame inequality on a single coordinate
indicator (`interactionFrameBound_le_interactionNorm`) shows that *any* valid
frame constant is bounded above by the norm of every individual interaction
vector.  For the structured Gaussian family this norm has the clean closed form
ceiling `|Δ| e^{-Δ²/4}`, `Δ = zᵢ − zⱼ`, maximized at `|Δ| = √2` and decaying to
`0` both as support points collide *and* as they separate.  This is a rigorous,
computable description of when the exact theorem is numerically useless, and an
absolute `m`-independent ceiling `√(2/e) ≈ 0.858` on the achievable constant. -/

/-- The actual structured-Gaussian induced interaction vector in closed form. -/
theorem inducedInteractionVector_structuredGaussian
    {m : ℕ} (hm : 0 < m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (p : StrictPair m) :
    inducedInteractionVector (empiricalFin z)
      (meanShiftInteractionKernel (gaussianKernel 1))
      (empiricalPointDensity z) (structuredGaussianProbes m) p.1.1 p.1.2 =
      fun n => gaussianProbeRow n *
        (gaussianPairColumn z p * gaussianPairBase z p ^ (n : ℕ)) := by
  funext n
  change basisInteraction (empiricalFin z)
    (meanShiftInteractionKernel (gaussianKernel 1))
    (empiricalPointDensity z p.1.1) (empiricalPointDensity z p.1.2)
    (structuredGaussianProbes m n) = _
  rw [basisInteraction_empiricalPoint hm z hz]
  exact gaussianInteraction_eq_weightedGeometric z p n

/-- Closed-form ceiling `|zᵢ - zⱼ| · exp(-(zᵢ-zⱼ)²/4)` for the norm of a
structured Gaussian interaction vector. -/
noncomputable def gaussianInteractionNormCeiling {m : ℕ} (z : Fin m → ℝ)
    (p : StrictPair m) : ℝ :=
  |z p.1.1 - z p.1.2| * Real.exp (-((z p.1.1 - z p.1.2) ^ 2) / 4)

/-- Pointwise ceiling: at every integer probe the interaction magnitude is at
most `|Δ| e^{-Δ²/4}`, with equality only at the (generally non-integer) probe
`n = (zᵢ+zⱼ)/2`. -/
theorem abs_structuredGaussianInteraction_le {m : ℕ} (z : Fin m → ℝ)
    (p : StrictPair m) (n : Fin (Fintype.card (StrictPair m))) :
    |gaussianProbeRow n *
        (gaussianPairColumn z p * gaussianPairBase z p ^ (n : ℕ))|
      ≤ gaussianInteractionNormCeiling z p := by
  have hbase : gaussianPairBase z p ^ (n : ℕ)
      = Real.exp ((n : ℝ) * (z p.1.1 + z p.1.2)) := by
    rw [gaussianPairBase, ← Real.exp_nat_mul]
  rw [gaussianProbeRow, gaussianPairColumn, hbase, gaussianInteractionNormCeiling,
    show Real.exp (-((n : ℝ) ^ 2)) *
        (Real.exp (-(1 / 2 : ℝ) * (z p.1.1 ^ 2 + z p.1.2 ^ 2)) *
            (z p.1.1 - z p.1.2) *
          Real.exp ((n : ℝ) * (z p.1.1 + z p.1.2)))
      = (z p.1.1 - z p.1.2) *
          Real.exp (-((n : ℝ) ^ 2) + -(1 / 2 : ℝ) * (z p.1.1 ^ 2 + z p.1.2 ^ 2)
            + (n : ℝ) * (z p.1.1 + z p.1.2)) from by
        rw [Real.exp_add, Real.exp_add]; ring]
  rw [abs_mul, abs_of_nonneg (Real.exp_pos _).le]
  refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
  rw [Real.exp_le_exp]
  nlinarith [sq_nonneg (2 * (n : ℝ) - (z p.1.1 + z p.1.2))]

/-- The structured Gaussian interaction vector has norm at most the closed-form
ceiling. -/
theorem norm_inducedInteractionVector_structuredGaussian_le
    {m : ℕ} (hm : 0 < m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (p : StrictPair m) :
    ‖inducedInteractionVector (empiricalFin z)
      (meanShiftInteractionKernel (gaussianKernel 1))
      (empiricalPointDensity z) (structuredGaussianProbes m) p.1.1 p.1.2‖ ≤
      gaussianInteractionNormCeiling z p := by
  rw [inducedInteractionVector_structuredGaussian hm z hz]
  have hnn : 0 ≤ gaussianInteractionNormCeiling z p := by
    rw [gaussianInteractionNormCeiling]; positivity
  refine (pi_norm_le_iff_of_nonneg hnn).2 (fun n => ?_)
  rw [Real.norm_eq_abs]
  exact abs_structuredGaussianInteraction_le z p n

/-- **Explicit ceiling on any frame constant (Objective 1).**  Every frame
constant valid for the structured Gaussian construction is bounded above by
`|zᵢ - zⱼ| · exp(-(zᵢ-zⱼ)²/4)` for *each* strict pair, hence by the minimum over
pairs.  The bound is computable from the support geometry alone. -/
theorem gaussianEmpiricalPoint_frameConstant_le
    {m : ℕ} (hm : 0 < m) (z : Fin m → ℝ) (hz : Function.Injective z)
    {c : ℝ} (hframe : InteractionFrameBound
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel 1))
        (empiricalPointDensity z) (structuredGaussianProbes m)) c)
    (p : StrictPair m) :
    c ≤ gaussianInteractionNormCeiling z p :=
  le_trans (interactionFrameBound_le_interactionNorm _ hframe p)
    (norm_inducedInteractionVector_structuredGaussian_le hm z hz p)

/-- The canonically selected general-`m` frame constant obeys the same explicit
ceiling. -/
theorem gaussianEmpiricalPointFrameConstant_le
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) (p : StrictPair m) :
    gaussianEmpiricalPointFrameConstant hm z hz hsums ≤
      gaussianInteractionNormCeiling z p :=
  gaussianEmpiricalPoint_frameConstant_le (by omega) z hz
    (gaussianEmpiricalPointFrameBound hm z hz hsums) p

/-- All mean-shift regularity obligations are automatic for the general
finite empirical Gaussian model. -/
theorem gaussianEmpiricalPoint_meanShiftRegular
    {m : ℕ} (hm : 0 < m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (x : ℝ) (a b : FiniteProbabilityVector m) :
    MeanShiftRegularAt (gaussianKernel 1)
      ((empiricalPointBasis hm z hz).basisMeasure a)
      ((empiricalPointBasis hm z hz).basisMeasure b) x := by
  refine
    { zp_ne_zero := ne_of_gt
        (gaussianKernelNormalizer_empiricalPoint_pos hm z hz 1 x a)
      zq_ne_zero := ne_of_gt
        (gaussianKernelNormalizer_empiricalPoint_pos hm z hz 1 x b)
      integrable_p := empiricalPointBasis_integrable hm z hz a _
      integrable_q := empiricalPointBasis_integrable hm z hz b _
      integrable_product := empiricalPointBasis_integrable_prod hm z hz a b _ ?_ }
  apply Continuous.aestronglyMeasurable
  unfold gaussianKernel
  fun_prop

/-- Complete general-`m` population setup under explicit geometric
conditions.  Its frame bound is derived, not assumed. -/
noncomputable def gaussianEmpiricalPointSetup
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) (a b : FiniteProbabilityVector m) :
    PopulationMeanShiftFiniteSetup ℝ m (Fintype.card (StrictPair m)) where
  reference := empiricalFin z
  refProb := empiricalFin_isProbability (by omega) z
  basis := empiricalPointBasis (by omega) z hz
  kernel := gaussianKernel 1
  probes := structuredGaussianProbes m
  a := a
  b := b
  meanShiftRegular n :=
    gaussianEmpiricalPoint_meanShiftRegular (by omega) z hz
      (structuredGaussianProbes m n) a b
  interactionIntegrable n := by
    apply empiricalPointBasis_integrable_prod (by omega) z hz
    apply Continuous.aestronglyMeasurable
    simp only [meanShiftInteractionKernel]
    unfold gaussianKernel
    fun_prop
  basisInteractionIntegrable i j n := by
    apply integrable_empiricalFin_prod (by omega) z
    apply Measurable.aestronglyMeasurable
    have hi : Measurable (fun y : ℝ × ℝ => empiricalPointDensity z i y.1) :=
      ((empiricalPointBasis (by omega) z hz).measurable_density i).comp measurable_fst
    have hj : Measurable (fun y : ℝ × ℝ => empiricalPointDensity z j y.2) :=
      ((empiricalPointBasis (by omega) z hz).measurable_density j).comp measurable_snd
    have hK : Measurable (fun y : ℝ × ℝ =>
        meanShiftInteractionKernel (gaussianKernel 1)
          (structuredGaussianProbes m n) y.1 y.2) := by
      apply Continuous.measurable
      simp only [meanShiftInteractionKernel]
      unfold gaussianKernel
      fun_prop
    exact (hi.mul hj).smul hK
  frameConstant := gaussianEmpiricalPointCertifiedFrameConstant z
  frameBound := gaussianEmpiricalPointCertifiedFrameBound hm z hz hsums

/-- The normalizer-product bound in the concrete unit-Gaussian setup is
explicit: `B = 1`. -/
theorem gaussianEmpiricalPoint_normalizerProduct_abs_le_one
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) (a b : FiniteProbabilityVector m)
    (n : Fin (Fintype.card (StrictPair m))) :
    |(gaussianEmpiricalPointSetup hm z hz hsums a b).normalizerProduct n| ≤ 1 := by
  change |kernelNormalizer (gaussianKernel 1)
      ((empiricalPointBasis (by omega) z hz).basisMeasure a)
      (structuredGaussianProbes m n) *
    kernelNormalizer (gaussianKernel 1)
      ((empiricalPointBasis (by omega) z hz).basisMeasure b)
      (structuredGaussianProbes m n)| ≤ 1
  have hpa := gaussianKernelNormalizer_empiricalPoint_pos
    (by omega) z hz 1 (structuredGaussianProbes m n) a
  have hpb := gaussianKernelNormalizer_empiricalPoint_pos
    (by omega) z hz 1 (structuredGaussianProbes m n) b
  rw [abs_of_pos (mul_pos hpa hpb)]
  exact mul_le_one₀
    (gaussianKernelNormalizer_empiricalPoint_le_one
      (by omega) z hz (structuredGaussianProbes m n) a)
    hpb.le
    (gaussianKernelNormalizer_empiricalPoint_le_one
      (by omega) z hz (structuredGaussianProbes m n) b)

/-- **General concrete exact-identifiability theorem.**  For any finite
empirical point family on `ℝ` with distinct support points and distinct
strict-pair sums, zero normalized population Gaussian mean-shift drift forces
equality of the represented probability measures. -/
theorem gaussianEmpiricalPoint_identifies
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) (a b : FiniteProbabilityVector m)
    (hzero : ZeroDrift (meanShiftDrift (gaussianKernel 1))
      ((empiricalPointBasis (by omega) z hz).basisMeasure a)
      ((empiricalPointBasis (by omega) z hz).basisMeasure b)) :
    (empiricalPointBasis (by omega) z hz).basisMeasure a =
      (empiricalPointBasis (by omega) z hz).basisMeasure b := by
  exact finitePopulationMeanShift_identifies
    (gaussianEmpiricalPointSetup hm z hz hsums a b) hzero

/-- **Probe-local general concrete theorem (Objective 2).**  Only the finite
drift values at the structured probes need to vanish — not the drift at every
point of `ℝ`.  This is the exact hypothesis the proof uses and the finite
quantity an empirical loss can observe. -/
theorem gaussianEmpiricalPoint_identifies_of_probeZero
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) (a b : FiniteProbabilityVector m)
    (hzero : ∀ n, meanShiftDrift (gaussianKernel 1)
      ((empiricalPointBasis (by omega) z hz).basisMeasure a)
      ((empiricalPointBasis (by omega) z hz).basisMeasure b)
      (structuredGaussianProbes m n) = 0) :
    (empiricalPointBasis (by omega) z hz).basisMeasure a =
      (empiricalPointBasis (by omega) z hz).basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeZero
    (gaussianEmpiricalPointSetup hm z hz hsums a b) hzero

/-- Zero deterministic squared loss at the structured probes is an equivalent
finite-loss hypothesis for the concrete exact-identifiability theorem. -/
theorem gaussianEmpiricalPoint_identifies_of_probeEnergy_eq_zero
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) (a b : FiniteProbabilityVector m)
    (henergy :
      (gaussianEmpiricalPointSetup hm z hz hsums a b).normalizedProbeDriftEnergy = 0) :
    (empiricalPointBasis (by omega) z hz).basisMeasure a =
      (empiricalPointBasis (by omega) z hz).basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero
    (gaussianEmpiricalPointSetup hm z hz hsums a b) henergy

/-- General concrete coefficient stability.  The remaining practical input is
an upper bound `B` on normalizer products; the frame constant is supplied by
the geometric construction. -/
theorem gaussianEmpiricalPoint_coefficientStability
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) (a b : FiniteProbabilityVector m)
    {B : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n,
      |(gaussianEmpiricalPointSetup hm z hz hsums a b).normalizerProduct n| ≤ B) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 * B / gaussianEmpiricalPointCertifiedFrameConstant z) *
        ‖(gaussianEmpiricalPointSetup hm z hz hsums a b).normalizedProbeDrift‖ := by
  exact (gaussianEmpiricalPointSetup hm z hz hsums a b).coefficientStability
    hB0 hnormalizer

/-- Stability with the concrete normalizer bound `B=1`; only the frame
conditioning constant remains. -/
theorem gaussianEmpiricalPoint_coefficientStability_one
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) (a b : FiniteProbabilityVector m) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 / gaussianEmpiricalPointCertifiedFrameConstant z) *
        ‖(gaussianEmpiricalPointSetup hm z hz hsums a b).normalizedProbeDrift‖ := by
  simpa using gaussianEmpiricalPoint_coefficientStability hm z hz hsums a b
    (B := 1) (by norm_num)
    (gaussianEmpiricalPoint_normalizerProduct_abs_le_one hm z hz hsums a b)

/-- Energy-form stability with both concrete constants exposed: `B=1` and the
inverse interaction-matrix frame certificate. -/
theorem gaussianEmpiricalPoint_coefficientStability_probeEnergy
    {m : ℕ} (hm : 2 ≤ m) (z : Fin m → ℝ) (hz : Function.Injective z)
    (hsums : DistinctStrictPairSums z) (a b : FiniteProbabilityVector m) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 / gaussianEmpiricalPointCertifiedFrameConstant z) *
        Real.sqrt
          (gaussianEmpiricalPointSetup hm z hz hsums a b).normalizedProbeDriftEnergy := by
  have h :=
    (gaussianEmpiricalPointSetup hm z hz hsums a b).coefficientStability_probeEnergy
      (B := 1) (by norm_num)
      (gaussianEmpiricalPoint_normalizerProduct_abs_le_one hm z hz hsums a b)
  simpa only [gaussianEmpiricalPointSetup, mul_one] using h

/-- **Closed form of the actual integral-induced interaction.**  Against the
two-point empirical reference, the mean-shift interaction double integral of
equation (30) collapses to an explicit expression: the coefficient minor of the
basis values times the product of the two kernel values, along the direction
`z₀ − z₁`. -/
theorem basisInteraction_empirical2 (k : E → E → ℝ) (φi φj : E → ℝ) (z0 z1 x : E) :
    basisInteraction (empirical2 z0 z1) (meanShiftInteractionKernel k) φi φj x
      = ((2⁻¹ * 2⁻¹) * (φi z0 * φj z1 - φi z1 * φj z0) * (k x z0 * k x z1)) •
          (z0 - z1) := by
  unfold basisInteraction
  simp_rw [integral_empirical2, meanShiftInteractionKernel]
  simp only [sub_self, smul_zero, add_zero, zero_add, smul_smul]
  module

/-! ## A genuine two-point probability-density basis -/

/-- Point-mass densities relative to `½(δ₀+δ₁)`.  The value `2` compensates
for the reference mass `½` at the selected point. -/
noncomputable def empirical01Density (i : Fin 2) : ℝ → ℝ :=
  if i = 0 then ({0} : Set ℝ).indicator (fun _ => 2)
  else ({1} : Set ℝ).indicator (fun _ => 2)

@[simp]
theorem empirical01Density_zero_zero : empirical01Density 0 0 = 2 := by
  simp [empirical01Density]

@[simp]
theorem empirical01Density_zero_one : empirical01Density 0 1 = 0 := by
  simp [empirical01Density]

@[simp]
theorem empirical01Density_one_zero : empirical01Density 1 0 = 0 := by
  simp [empirical01Density]

@[simp]
theorem empirical01Density_one_one : empirical01Density 1 1 = 2 := by
  simp [empirical01Density]

/-- The explicit point-mass functions are valid unit-mass densities. -/
noncomputable def empirical01Basis :
    ProbabilityDensityBasis (empirical2 (0 : ℝ) 1) 2 where
  density := empirical01Density
  measurable_density i := by
    classical
    unfold empirical01Density
    split
    · exact measurable_const.indicator (measurableSet_singleton (0 : ℝ))
    · exact measurable_const.indicator (measurableSet_singleton (1 : ℝ))
  nonnegative i x := by
    classical
    unfold empirical01Density
    split <;> simp only [Set.indicator_apply] <;> split_ifs <;> norm_num
  integrable_density i := integrable_empirical2 0 1 (empirical01Density i)
  integral_density i := by
    rw [integral_empirical2]
    fin_cases i <;> norm_num

/-- The concrete mixture measure, displayed as its two atomic masses. -/
noncomputable def empirical01MixtureMeasure (a : FiniteProbabilityVector 2) :
    Distribution ℝ :=
  ENNReal.ofReal (a.weight 0) • Measure.dirac 0 +
    ENNReal.ofReal (a.weight 1) • Measure.dirac 1

/-- The `withDensity` representation of the concrete basis is exactly the
expected two-atom mixture. -/
theorem empirical01Basis_basisMeasure (a : FiniteProbabilityVector 2) :
    empirical01Basis.basisMeasure a = empirical01MixtureMeasure a := by
  unfold ProbabilityDensityBasis.basisMeasure empirical01MixtureMeasure empirical2
  rw [withDensity_smul_measure, withDensity_add_measure,
    dirac_withDensity, dirac_withDensity]
  simp only [ProbabilityDensityBasis.mixtureDensity, basisDensity,
    empirical01Basis, Fin.sum_univ_two, empirical01Density_zero_zero,
    empirical01Density_one_zero, empirical01Density_zero_one,
    empirical01Density_one_one, mul_zero, add_zero, mul_two, zero_add]
  rw [show a.weight 0 + a.weight 0 = 2 * a.weight 0 by ring,
    show a.weight 1 + a.weight 1 = 2 * a.weight 1 by ring]
  rw [ENNReal.ofReal_mul (by positivity : (0 : ℝ) ≤ 2),
    ENNReal.ofReal_mul (by positivity : (0 : ℝ) ≤ 2)]
  have htwo : (2 : ℝ≥0∞)⁻¹ * 2 = 1 :=
    ENNReal.inv_mul_cancel (by norm_num) (by simp)
  simp [smul_smul, ← mul_assoc, htwo]

/-- Every finite-valued function is integrable against a concrete two-atom
mixture law. -/
theorem integrable_empirical01MixtureMeasure
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]
    (a : FiniteProbabilityVector 2) (f : ℝ → F) :
    Integrable f (empirical01MixtureMeasure a) := by
  unfold empirical01MixtureMeasure
  rw [integrable_add_measure]
  constructor <;>
    exact (integrable_dirac (by simp)).smul_measure ENNReal.ofReal_ne_top

/-- Consequently all finite-valued functions are integrable under the
`basisMeasure` used by the population theorem. -/
theorem empirical01Basis_integrable
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]
    (a : FiniteProbabilityVector 2) (f : ℝ → F) :
    Integrable f (empirical01Basis.basisMeasure a) := by
  rw [empirical01Basis_basisMeasure]
  exact integrable_empirical01MixtureMeasure a f

/-- Fubini's criterion reduces integrability on a product of two concrete
mixtures to the already established atomic marginal facts. -/
theorem empirical01Basis_integrable_prod
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]
    (a b : FiniteProbabilityVector 2) (f : ℝ × ℝ → F)
    (hf : AEStronglyMeasurable f
      ((empirical01Basis.basisMeasure a).prod
        (empirical01Basis.basisMeasure b))) :
    Integrable f ((empirical01Basis.basisMeasure a).prod
      (empirical01Basis.basisMeasure b)) := by
  letI := empirical01Basis.basisMeasure_isProbability a
  letI := empirical01Basis.basisMeasure_isProbability b
  rw [integrable_prod_iff hf]
  exact ⟨Filter.Eventually.of_forall fun x =>
      empirical01Basis_integrable b (fun y => f (x, y)),
    empirical01Basis_integrable a (fun x =>
      ∫ y, ‖f (x, y)‖ ∂empirical01Basis.basisMeasure b)⟩

/-- The Gaussian normalizer of a concrete mixture is the corresponding
positive weighted sum of the two kernel values. -/
theorem gaussianKernelNormalizer_empirical01
    (σ x : ℝ) (a : FiniteProbabilityVector 2) :
    kernelNormalizer (gaussianKernel σ) (empirical01Basis.basisMeasure a) x =
      a.weight 0 * gaussianKernel σ x 0 +
        a.weight 1 * gaussianKernel σ x 1 := by
  unfold kernelNormalizer
  rw [empirical01Basis.integral_basisMeasure_eq_density_smul,
    integral_empirical2]
  simp only [ProbabilityDensityBasis.mixtureDensity, basisDensity,
    empirical01Basis, Fin.sum_univ_two, empirical01Density_zero_zero,
    empirical01Density_one_zero, empirical01Density_zero_one,
    empirical01Density_one_one, mul_zero, add_zero, mul_two, zero_add,
    smul_eq_mul]
  ring

/-- Gaussian positivity and simplex normalization make every concrete
normalizer nonzero (indeed strictly positive). -/
theorem gaussianKernelNormalizer_empirical01_pos
    (σ x : ℝ) (a : FiniteProbabilityVector 2) :
    0 < kernelNormalizer (gaussianKernel σ)
      (empirical01Basis.basisMeasure a) x := by
  rw [gaussianKernelNormalizer_empirical01]
  have hk0 : 0 < gaussianKernel σ x 0 := by
    rw [gaussianKernel]
    exact Real.exp_pos _
  have hk1 : 0 < gaussianKernel σ x 1 := by
    rw [gaussianKernel]
    exact Real.exp_pos _
  have hsum : a.weight 0 + a.weight 1 = 1 := by
    simpa [Fin.sum_univ_two] using a.normalized
  have hor : 0 < a.weight 0 ∨ 0 < a.weight 1 := by
    have h0 := a.nonnegative 0
    have h1 := a.nonnegative 1
    rcases lt_or_eq_of_le h0 with h0pos | h0zero
    · exact Or.inl h0pos
    · right
      nlinarith
  rcases hor with h0 | h1
  · exact add_pos_of_pos_of_nonneg (mul_pos h0 hk0)
      (mul_nonneg (a.nonnegative 1) hk1.le)
  · exact add_pos_of_nonneg_of_pos
      (mul_nonneg (a.nonnegative 0) hk0.le) (mul_pos h1 hk1)

/-! ## The `m = 2` frame bound -/

/-- The strict-pair index set for `m = 2` is a singleton. -/
instance : Unique (StrictPair 2) where
  default := ⟨(0, 1), by decide⟩
  uniq := by decide

omit [MeasurableSpace E] [MeasurableSingletonClass E] [CompleteSpace E] in
/-- For `m = 2` a single nonzero interaction vector already gives a positive
frame bound (with constant its own norm). -/
theorem interactionFrameBound_two {N : ℕ} (U : Fin 2 → Fin 2 → Fin N → E)
    (hU : U 0 1 ≠ 0) : InteractionFrameBound U ‖U 0 1‖ := by
  refine ⟨norm_pos_iff.mpr hU, fun z => ?_⟩
  rw [interactionSynthesis_apply, Fintype.sum_unique, Fintype.sum_unique,
    norm_smul, Real.norm_eq_abs]
  change ‖U 0 1‖ * |z default| ≤ |z default| * ‖U 0 1‖
  rw [mul_comm]

/-! ## Positive frame bound for the actual induced vectors -/

variable {N : ℕ}

/-- For the two-point empirical reference and Gaussian kernel, the actual
integral-induced interaction vector `U₀₁` is nonzero as soon as the two points
differ and the basis value minor is nonzero (both concretely checkable). -/
theorem inducedInteractionVector_empirical2_ne_zero
    (σ : ℝ) (z0 z1 : E) (hz : z0 ≠ z1)
    (φ : Fin 2 → E → ℝ) (probes : Fin N → E) (n0 : Fin N)
    (hminor : φ 0 z0 * φ 1 z1 - φ 0 z1 * φ 1 z0 ≠ 0) :
    inducedInteractionVector (empirical2 z0 z1)
      (meanShiftInteractionKernel (gaussianKernel σ)) φ probes 0 1 ≠ 0 := by
  rw [Function.ne_iff]
  refine ⟨n0, ?_⟩
  simp only [inducedInteractionVector, basisInteraction_empirical2, Pi.zero_apply]
  apply smul_ne_zero
  · have h0 : (0 : ℝ) < gaussianKernel σ (probes n0) z0 := by
      rw [gaussianKernel]; exact Real.exp_pos _
    have h1 : (0 : ℝ) < gaussianKernel σ (probes n0) z1 := by
      rw [gaussianKernel]; exact Real.exp_pos _
    exact mul_ne_zero (mul_ne_zero (by norm_num) hminor) (ne_of_gt (mul_pos h0 h1))
  · exact sub_ne_zero.mpr hz

/-- **Gap closed for `m = 2`.**  The paper's *actual* integral-induced interaction
family — for a two-point empirical reference, a Gaussian kernel, distinct points,
and a basis with nonzero value minor — satisfies a positive
`InteractionFrameBound`.  This discharges the nondegeneracy/frame hypothesis of
the mean-shift population theorem in a concrete implementable model, using the
real `basisInteraction` double integral (not a synthetic substitute) and no
external axiom. -/
theorem empiricalInteractionFrameBound
    (σ : ℝ) (z0 z1 : E) (hz : z0 ≠ z1)
    (φ : Fin 2 → E → ℝ) (probes : Fin N → E) (n0 : Fin N)
    (hminor : φ 0 z0 * φ 1 z1 - φ 0 z1 * φ 1 z0 ≠ 0) :
    InteractionFrameBound (inducedInteractionVector (empirical2 z0 z1)
      (meanShiftInteractionKernel (gaussianKernel σ)) φ probes)
      ‖inducedInteractionVector (empirical2 z0 z1)
        (meanShiftInteractionKernel (gaussianKernel σ)) φ probes 0 1‖ :=
  interactionFrameBound_two _
    (inducedInteractionVector_empirical2_ne_zero σ z0 z1 hz φ probes n0 hminor)

/-! ## End-to-end concrete population theorem -/

/-- All regularity obligations for the two-atom Gaussian population model are
automatic: the Gaussian is positive and every relevant function is integrable
on the finite support. -/
theorem empirical01Gaussian_meanShiftRegular
    (σ x : ℝ) (a b : FiniteProbabilityVector 2) :
    MeanShiftRegularAt (gaussianKernel σ)
      (empirical01Basis.basisMeasure a)
      (empirical01Basis.basisMeasure b) x := by
  refine
    { zp_ne_zero := ne_of_gt (gaussianKernelNormalizer_empirical01_pos σ x a)
      zq_ne_zero := ne_of_gt (gaussianKernelNormalizer_empirical01_pos σ x b)
      integrable_p := empirical01Basis_integrable a _
      integrable_q := empirical01Basis_integrable b _
      integrable_product := empirical01Basis_integrable_prod a b _ ?_ }
  apply Continuous.aestronglyMeasurable
  unfold gaussianKernel
  fun_prop

/-- The complete population setup for the explicit two-atom family.  Unlike
the generic theorem, this construction does not ask the caller to supply a
frame bound, normalizer proof, or integrability proof. -/
noncomputable def empirical01GaussianSetup
    {N : ℕ} (σ : ℝ) (_hσ : ValidBandwidth σ)
    (a b : FiniteProbabilityVector 2)
    (probes : Fin N → ℝ) (n0 : Fin N) :
    PopulationMeanShiftFiniteSetup ℝ 2 N where
  reference := empirical2 0 1
  refProb := empirical2_isProbability 0 1
  basis := empirical01Basis
  kernel := gaussianKernel σ
  probes := probes
  a := a
  b := b
  meanShiftRegular n := empirical01Gaussian_meanShiftRegular σ (probes n) a b
  interactionIntegrable n := by
    apply empirical01Basis_integrable_prod
    apply Continuous.aestronglyMeasurable
    simp only [meanShiftInteractionKernel]
    unfold gaussianKernel
    fun_prop
  basisInteractionIntegrable i j n := by
    apply integrable_empirical2_prod
    apply Measurable.aestronglyMeasurable
    have hi : Measurable (fun y : ℝ × ℝ => empirical01Density i y.1) :=
      (empirical01Basis.measurable_density i).comp measurable_fst
    have hj : Measurable (fun y : ℝ × ℝ => empirical01Density j y.2) :=
      (empirical01Basis.measurable_density j).comp measurable_snd
    have hK : Measurable (fun y : ℝ × ℝ =>
        meanShiftInteractionKernel (gaussianKernel σ) (probes n) y.1 y.2) := by
      apply Continuous.measurable
      simp only [meanShiftInteractionKernel]
      unfold gaussianKernel
      fun_prop
    exact (hi.mul hj).smul hK
  frameConstant :=
    ‖inducedInteractionVector (empirical2 (0 : ℝ) 1)
      (meanShiftInteractionKernel (gaussianKernel σ)) empirical01Density probes 0 1‖
  frameBound := by
    apply empiricalInteractionFrameBound σ 0 1 (by norm_num)
      empirical01Density probes n0
    norm_num

/-- **Concrete exact result.**  In the explicit two-atom probability family,
zero of the actual normalized population Gaussian mean-shift field identifies
the target and model measures. -/
theorem empirical01Gaussian_identifies
    {N : ℕ} (σ : ℝ) (hσ : ValidBandwidth σ)
    (a b : FiniteProbabilityVector 2)
    (probes : Fin N → ℝ) (n0 : Fin N)
    (hzero : ZeroDrift (meanShiftDrift (gaussianKernel σ))
      (empirical01Basis.basisMeasure a)
      (empirical01Basis.basisMeasure b)) :
    empirical01Basis.basisMeasure a = empirical01Basis.basisMeasure b := by
  exact finitePopulationMeanShift_identifies
    (empirical01GaussianSetup σ hσ a b probes n0) hzero

/-- **Probe-local two-atom concrete theorem (Objective 2).**  Zero of the actual
normalized Gaussian mean-shift field at the finitely many supplied probes is
enough to identify the target and model measures. -/
theorem empirical01Gaussian_identifies_of_probeZero
    {N : ℕ} (σ : ℝ) (hσ : ValidBandwidth σ)
    (a b : FiniteProbabilityVector 2)
    (probes : Fin N → ℝ) (n0 : Fin N)
    (hzero : ∀ n, meanShiftDrift (gaussianKernel σ)
      (empirical01Basis.basisMeasure a)
      (empirical01Basis.basisMeasure b) (probes n) = 0) :
    empirical01Basis.basisMeasure a = empirical01Basis.basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeZero
    (empirical01GaussianSetup σ hσ a b probes n0) hzero

/-- Two-atom finite-loss form: zero deterministic squared loss over the
supplied probes identifies the two represented measures. -/
theorem empirical01Gaussian_identifies_of_probeEnergy_eq_zero
    {N : ℕ} (σ : ℝ) (hσ : ValidBandwidth σ)
    (a b : FiniteProbabilityVector 2)
    (probes : Fin N → ℝ) (n0 : Fin N)
    (henergy :
      (empirical01GaussianSetup σ hσ a b probes n0).normalizedProbeDriftEnergy = 0) :
    empirical01Basis.basisMeasure a = empirical01Basis.basisMeasure b :=
  finitePopulationMeanShift_identifies_of_probeEnergy_eq_zero
    (empirical01GaussianSetup σ hσ a b probes n0) henergy

/-- Concrete coefficient stability for the same two-atom population model. -/
theorem empirical01Gaussian_coefficientStability
    {N : ℕ} (σ : ℝ) (hσ : ValidBandwidth σ)
    (a b : FiniteProbabilityVector 2)
    (probes : Fin N → ℝ) (n0 : Fin N) {B : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n,
      |(empirical01GaussianSetup σ hσ a b probes n0).normalizerProduct n| ≤ B) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 * B /
        (empirical01GaussianSetup σ hσ a b probes n0).frameConstant) *
        ‖(empirical01GaussianSetup σ hσ a b probes n0).normalizedProbeDrift‖ := by
  exact (empirical01GaussianSetup σ hσ a b probes n0).coefficientStability
    hB0 hnormalizer

/-- Energy form of the two-atom stability estimate. -/
theorem empirical01Gaussian_coefficientStability_probeEnergy
    {N : ℕ} (σ : ℝ) (hσ : ValidBandwidth σ)
    (a b : FiniteProbabilityVector 2)
    (probes : Fin N → ℝ) (n0 : Fin N) {B : ℝ} (hB0 : 0 ≤ B)
    (hnormalizer : ∀ n,
      |(empirical01GaussianSetup σ hσ a b probes n0).normalizerProduct n| ≤ B) :
    (∑ i, |a.weight i - b.weight i|) ≤
      (2 * B /
        (empirical01GaussianSetup σ hσ a b probes n0).frameConstant) *
        Real.sqrt
          (empirical01GaussianSetup σ hσ a b probes n0).normalizedProbeDriftEnergy := by
  exact (empirical01GaussianSetup σ hσ a b probes n0).coefficientStability_probeEnergy
    hB0 hnormalizer

/-! ## Higher-dimensional data space (Objective 3)

The `basisInteraction` closed form holds for any data space, so the only
dimension-specific ingredient is the nondegeneracy of the induced vectors. On an
arbitrary real inner-product space `F`, placing the probes at integer multiples
of a fixed direction `u` reduces the Gaussian product to the same
weighted-geometric shape as on `ℝ`, except the interaction vectors are genuinely
vector valued (proportional to `zᵢ - zⱼ ∈ F`). The vector-weighted Vandermonde
engine then discharges nondegeneracy, replacing the distinct-pair-sum Sidon
condition by distinct *projected* pair-sums `⟪u, zᵢ+zⱼ⟫`. -/

section HigherDimensional

open scoped RealInnerProductSpace

variable {F : Type u} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
  [MeasurableSpace F] [MeasurableSingletonClass F] [CompleteSpace F]

/-- Probes: integer multiples of a fixed direction, one per strict pair. -/
noncomputable def structuredGaussianProbesND (m : ℕ) (u : F) :
    Fin (Fintype.card (StrictPair m)) → F := fun n => (n : ℝ) • u

/-- Row factor (depends only on the probe index and the direction length). -/
noncomputable def gaussianProbeRowND (m : ℕ) (σ : ℝ) (u : F) :
    Fin (Fintype.card (StrictPair m)) → ℝ :=
  fun n => Real.exp (-((n : ℝ) ^ 2 * ‖u‖ ^ 2) / σ ^ 2)

/-- Geometric base: exponential of the projected pair-sum. -/
noncomputable def gaussianPairBaseND (σ : ℝ) (u : F) {m : ℕ} (z : Fin m → F) :
    StrictPair m → ℝ :=
  fun p => Real.exp (inner ℝ u (z p.1.1 + z p.1.2) / σ ^ 2)

/-- Column scalar (depends only on the pair). -/
noncomputable def gaussianPairColumnScalarND (σ : ℝ) {m : ℕ} (z : Fin m → F) :
    StrictPair m → ℝ :=
  fun p => Real.exp (-(‖z p.1.1‖ ^ 2 + ‖z p.1.2‖ ^ 2) / (2 * σ ^ 2))

omit [MeasurableSpace F] [MeasurableSingletonClass F] [CompleteSpace F] in
/-- The higher-dimensional Gaussian interaction factors into a scalar
weighted-geometric profile times the vector direction `zᵢ - zⱼ`. -/
theorem gaussianInteractionND_eq_weightedGeometric {m : ℕ} (σ : ℝ) (hσ : σ ≠ 0)
    (u : F) (z : Fin m → F) (p : StrictPair m)
    (n : Fin (Fintype.card (StrictPair m))) :
    meanShiftInteractionKernel (gaussianKernel σ)
      (structuredGaussianProbesND m u n) (z p.1.1) (z p.1.2) =
      (gaussianProbeRowND m σ u n * gaussianPairBaseND σ u z p ^ (n : ℕ)) •
        (gaussianPairColumnScalarND σ z p • (z p.1.1 - z p.1.2)) := by
  have hσ2 : (σ : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 hσ
  have hnu : ‖((n : ℝ)) • u‖ ^ 2 = (n : ℝ) ^ 2 * ‖u‖ ^ 2 := by
    rw [norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
  have hscalar :
      gaussianKernel σ (structuredGaussianProbesND m u n) (z p.1.1) *
        gaussianKernel σ (structuredGaussianProbesND m u n) (z p.1.2) =
      gaussianProbeRowND m σ u n * gaussianPairBaseND σ u z p ^ (n : ℕ) *
        gaussianPairColumnScalarND σ z p := by
    simp only [structuredGaussianProbesND, gaussianProbeRowND, gaussianPairBaseND,
      gaussianPairColumnScalarND, gaussianKernel, ← Real.exp_nat_mul, ← Real.exp_add]
    congr 1
    rw [norm_sub_sq_real, norm_sub_sq_real]
    simp only [hnu, real_inner_smul_left, inner_add_right]
    field_simp
    ring
  rw [meanShiftInteractionKernel, hscalar, smul_smul]

/-- Independently checkable higher-dimensional separation condition: the
projected pair-sums `⟪u, zᵢ+zⱼ⟫` are distinct. -/
def DistinctProjectedPairSums {m : ℕ} (u : F) (z : Fin m → F) : Prop :=
  Function.Injective (fun p : StrictPair m => inner ℝ u (z p.1.1 + z p.1.2))

/-- **Axiom-free higher-dimensional nondegeneracy.**  On any real inner-product
space, an injective support with distinct projected pair-sums, unit-direction
integer probes and a Gaussian kernel yields linearly independent actual
integral-induced interaction vectors. -/
theorem gaussianEmpiricalPointND_basisNondegenerate
    {m : ℕ} (hm : 0 < m) (σ : ℝ) (hσ : σ ≠ 0) (u : F) (z : Fin m → F)
    (hz : Function.Injective z) (hsums : DistinctProjectedPairSums u z) :
    BasisInteractionNondegenerate
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel σ))
        (empiricalPointDensity z) (structuredGaussianProbesND m u)) := by
  have hσ2 : (σ : ℝ) ^ 2 ≠ 0 := pow_ne_zero 2 hσ
  have hbase : Function.Injective (gaussianPairBaseND σ u z) := by
    intro p q hpq
    apply hsums
    simp only [gaussianPairBaseND] at hpq
    have h2 := Real.exp_injective hpq
    exact (div_left_inj' hσ2).mp h2
  have hw : ∀ p : StrictPair m,
      gaussianPairColumnScalarND σ z p • (z p.1.1 - z p.1.2) ≠ 0 := by
    intro p
    refine smul_ne_zero (Real.exp_pos _).ne' ?_
    exact sub_ne_zero.mpr (fun h => (Fin.ne_of_lt p.property) (hz h))
  have hrow : ∀ n, gaussianProbeRowND m σ u n ≠ 0 :=
    fun _ => (Real.exp_pos _).ne'
  have hgeom := linearIndependent_vectorWeightedGeometricProfiles
    (gaussianPairBaseND σ u z)
    (fun p => gaussianPairColumnScalarND σ z p • (z p.1.1 - z p.1.2))
    hbase hw (gaussianProbeRowND m σ u) hrow
  have hfamily :
      (fun p : StrictPair m =>
        inducedInteractionVector (empiricalFin z)
          (meanShiftInteractionKernel (gaussianKernel σ))
          (empiricalPointDensity z) (structuredGaussianProbesND m u) p.1.1 p.1.2) =
      (fun p : StrictPair m => fun n =>
        (gaussianProbeRowND m σ u n * gaussianPairBaseND σ u z p ^ (n : ℕ)) •
          (gaussianPairColumnScalarND σ z p • (z p.1.1 - z p.1.2))) := by
    funext p n
    change basisInteraction (empiricalFin z)
      (meanShiftInteractionKernel (gaussianKernel σ))
      (empiricalPointDensity z p.1.1) (empiricalPointDensity z p.1.2)
      (structuredGaussianProbesND m u n) = _
    rw [basisInteraction_empiricalPoint hm z hz]
    exact gaussianInteractionND_eq_weightedGeometric σ hσ u z p n
  unfold BasisInteractionNondegenerate
  rw [hfamily]
  exact hgeom

/-- **Higher-dimensional frame bound.**  The interaction-frame hypothesis of the
population theorem is discharged for the arbitrary-dimensional structured
Gaussian construction (`m ≥ 2`), axiom-free.  Only the qualitative constant is
provided; the explicit ceiling of accomplishment 8 applies verbatim in each
coordinate. -/
theorem gaussianEmpiricalPointND_exists_frameBound
    {m : ℕ} (hm : 2 ≤ m) (σ : ℝ) (hσ : σ ≠ 0) (u : F) (z : Fin m → F)
    (hz : Function.Injective z) (hsums : DistinctProjectedPairSums u z) :
    ∃ c > 0, InteractionFrameBound
      (inducedInteractionVector (empiricalFin z)
        (meanShiftInteractionKernel (gaussianKernel σ))
        (empiricalPointDensity z) (structuredGaussianProbesND m u)) c := by
  let p : StrictPair m := ⟨(⟨0, by omega⟩, ⟨1, by omega⟩), by simp⟩
  letI : Nonempty (StrictPair m) := ⟨p⟩
  apply interactionFrameBound_of_linearIndependent
  exact gaussianEmpiricalPointND_basisNondegenerate (by omega) σ hσ u z hz hsums

end HigherDimensional

end PaperFiniteIdentifiability
end DriftingIdentifiability
