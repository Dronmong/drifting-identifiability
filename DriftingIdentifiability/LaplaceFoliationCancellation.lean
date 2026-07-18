import DriftingIdentifiability.LaplaceConeExtraction

/-!
# General Euclidean Laplace converse, G4/P1: foliation cancellation

This file formalizes the axiom-free part of the foliation--cancellation route
from `LaplaceRnRoadmap.md`, F-E and G4/P1.

There are two deliberately separate layers.

* The measure layer proves, for the actual Laplace normalizers and displacement
  potentials, that zero drift aligns the two gradients by the positive,
  continuous normalizer ratio.
* The local leaf layer proves the cancellation and non-degenerate-leaf algebra
  once a standard local submersion chart has supplied
  `ψp = G ∘ ψq`, its Laplacian chain rule, and the elliptic identities.

The chart/regularity inputs are explicit hypotheses.  They contain no
identifiability conclusion and are not promoted as the final G4 theorem.  Tube
rigidity and gluing remain separate downstream obligations.
-/

open MeasureTheory Topology
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasureSpace E] [BorelSpace E] [CompleteSpace E]
  [SecondCountableTopology E]

/-! ## The actual measure-level ratio and gradient alignment -/

/-- The positive normalizer ratio used by the G4 foliation argument. -/
noncomputable def laplaceNormalizerRatio
    (τ : ℝ) (p q : Measure E) (x : E) : ℝ :=
  kernelNormalizer (laplaceKernel τ) p x /
    kernelNormalizer (laplaceKernel τ) q x

set_option linter.unusedSectionVars false in
/-- The Laplace normalizer ratio is strictly positive for probability laws. -/
theorem laplaceNormalizerRatio_pos
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (x : E) :
    0 < laplaceNormalizerRatio τ p q x := by
  unfold laplaceNormalizerRatio
  exact div_pos (laplaceKernelNormalizer_pos p τ hτ x)
    (laplaceKernelNormalizer_pos q τ hτ x)

set_option linter.unusedSectionVars false in
/-- The Laplace normalizer ratio is continuous.  This uses only finiteness of
the two measures and positivity of the denominator. -/
theorem continuous_laplaceNormalizerRatio
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] :
    Continuous (laplaceNormalizerRatio τ p q) := by
  have hcont : ∀ μ : Measure E, IsProbabilityMeasure μ →
      Continuous (fun x : E => kernelNormalizer (laplaceKernel τ) μ x) := by
    intro μ hμ
    letI : IsProbabilityMeasure μ := hμ
    unfold kernelNormalizer
    refine continuous_of_dominated (bound := fun _ => (1 : ℝ))
      (fun x => ?_) (fun x => ?_) (integrable_const 1) ?_
    · apply Continuous.aestronglyMeasurable
      unfold laplaceKernel
      fun_prop
    · exact ae_of_all _ fun y => by
        rw [Real.norm_eq_abs, abs_of_pos (by
          unfold laplaceKernel
          exact Real.exp_pos _)]
        unfold laplaceKernel
        rw [Real.exp_le_one_iff]
        exact mul_nonpos_of_nonpos_of_nonneg
          (neg_nonpos.mpr (by positivity : 0 ≤ 1 / τ)) (norm_nonneg _)
    · exact ae_of_all _ fun y => by
        unfold laplaceKernel
        fun_prop
  unfold laplaceNormalizerRatio
  exact (hcont p inferInstance).div (hcont q inferInstance)
    (fun x => (laplaceKernelNormalizer_pos q τ hτ x).ne')

set_option linter.unusedSectionVars false in
/-- A probability-law Laplace normalizer lies in `(0,1]`.  The strict lower
bound is pointwise; the upper bound is the integral of the pointwise kernel
bound `exp (-‖x-y‖/τ) ≤ 1`. -/
theorem laplaceKernelNormalizer_mem_Ioc_one
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ] (x : E) :
    kernelNormalizer (laplaceKernel τ) μ x ∈ Set.Ioc 0 1 := by
  refine ⟨laplaceKernelNormalizer_pos μ τ hτ x, ?_⟩
  have hint : Integrable (fun y : E => laplaceKernel τ x y) μ := by
    refine Integrable.of_bound ?_ 1 ?_
    · apply Continuous.aestronglyMeasurable
      unfold laplaceKernel
      fun_prop
    · filter_upwards with y
      rw [Real.norm_eq_abs, abs_of_pos (by
        unfold laplaceKernel
        exact Real.exp_pos _)]
      unfold laplaceKernel
      rw [Real.exp_le_one_iff]
      exact mul_nonpos_of_nonpos_of_nonneg
        (neg_nonpos.mpr (by positivity : 0 ≤ 1 / τ)) (norm_nonneg _)
  unfold kernelNormalizer
  calc
    ∫ y, laplaceKernel τ x y ∂μ ≤ ∫ _y, (1 : ℝ) ∂μ := by
      apply integral_mono
      · exact hint
      · exact integrable_const 1
      · intro y
        unfold laplaceKernel
        rw [Real.exp_le_one_iff]
        exact mul_nonpos_of_nonpos_of_nonneg
          (neg_nonpos.mpr (by positivity : 0 ≤ 1 / τ)) (norm_nonneg _)
    _ = 1 := by simp

/-- Elementary quotient perturbation estimate used to turn the two global
normalizer Lipschitz bounds into a local bound for their ratio. -/
private theorem abs_div_sub_div_le_of_unit_interval
    {a b c d κ L δ : ℝ}
    (hc0 : 0 ≤ c) (hc1 : c ≤ 1)
    (hb0 : 0 < b) (hd0 : 0 < d) (hd1 : d ≤ 1)
    (hκ : 0 < κ) (hbκ : κ ≤ b) (hdκ : κ ≤ d)
    (hL : 0 ≤ L) (hδ : 0 ≤ δ)
    (hac : |a - c| ≤ L * δ) (hbd : |b - d| ≤ L * δ) :
    |a / b - c / d| ≤ (2 * L / κ ^ 2) * δ := by
  have hid : a / b - c / d = (d * (a - c) + c * (d - b)) / (b * d) := by
    field_simp
    ring
  rw [hid, abs_div, abs_of_pos (mul_pos hb0 hd0)]
  have hdabs : |d| ≤ 1 := by simpa [abs_of_pos hd0] using hd1
  have hcabs : |c| ≤ 1 := by simpa [abs_of_nonneg hc0] using hc1
  have hdb : |d - b| ≤ L * δ := by simpa [abs_sub_comm] using hbd
  have hnum : |d * (a - c) + c * (d - b)| ≤ 2 * L * δ := by
    calc
      |d * (a - c) + c * (d - b)|
          ≤ |d * (a - c)| + |c * (d - b)| := abs_add_le _ _
      _ = |d| * |a - c| + |c| * |d - b| := by rw [abs_mul, abs_mul]
      _ ≤ 1 * (L * δ) + 1 * (L * δ) :=
        add_le_add (mul_le_mul hdabs hac (abs_nonneg _) (by positivity))
          (mul_le_mul hcabs hdb (abs_nonneg _) (by positivity))
      _ = 2 * L * δ := by ring
  have hκsq : 0 < κ ^ 2 := sq_pos_of_pos hκ
  have hκbd : κ ^ 2 ≤ b * d := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hbκ) (sub_nonneg.mpr hdκ)]
  calc
    |d * (a - c) + c * (d - b)| / (b * d)
        ≤ (2 * L * δ) / (b * d) :=
      div_le_div_of_nonneg_right hnum (mul_pos hb0 hd0).le
    _ ≤ (2 * L * δ) / (κ ^ 2) := by
      exact div_le_div₀ (by positivity) le_rfl hκsq hκbd
    _ = (2 * L / κ ^ 2) * δ := by ring

/-! The explicit global Lipschitz estimates in `LaplaceConeExtraction` are
proved in the finite-dimensional Euclidean/Haar setting used by G4. -/

section EuclideanRegularity

variable [(volume : Measure E).IsAddHaarMeasure]
  [(volume : Measure E).IsNegInvariant]
  [FiniteDimensional ℝ E] [Nontrivial E]

set_option linter.unusedSectionVars false in
/-- A probability-law Laplace normalizer is globally `1/τ`-Lipschitz. -/
theorem lipschitzWith_laplaceKernelNormalizer
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ] :
    LipschitzWith (Real.toNNReal (1 / τ))
      (fun x : E => kernelNormalizer (laplaceKernel τ) μ x) := by
  refine LipschitzWith.of_dist_le' ?_
  intro x y
  rw [Real.dist_eq, dist_eq_norm]
  simpa using abs_kernelNormalizer_laplace_sub_le hτ μ x y

set_option linter.unusedSectionVars false in
/-- The displacement numerator, hence the gradient of the companion
potential, is globally `3`-Lipschitz for probability laws. -/
theorem lipschitzWith_laplaceDisplacementField_probability
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ] :
    LipschitzWith (Real.toNNReal 3)
      (laplaceDisplacementField τ μ) := by
  refine LipschitzWith.of_dist_le' ?_
  intro x y
  simpa [dist_eq_norm] using norm_laplaceDisplacementField_sub_le hτ μ x y

set_option linter.unusedSectionVars false in
/-- **Actual `C^{1,1}` data for P1.**  The companion potential has the stated
Frechet derivative everywhere and its representing gradient is globally
Lipschitz.  This is the precise regularity package used by the foliation
argument; no density or moment assumption is needed. -/
theorem laplaceDisplacementPotential_hasLipschitzGradient
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ] :
    (∀ x, HasFDerivAt (laplaceDisplacementPotential τ μ)
      (innerSL ℝ (laplaceDisplacementField τ μ x)) x) ∧
      LipschitzWith (Real.toNNReal 3) (laplaceDisplacementField τ μ) := by
  exact ⟨hasFDerivAt_laplaceDisplacementPotential hτ μ,
    lipschitzWith_laplaceDisplacementField_probability hτ μ⟩

set_option linter.unusedSectionVars false in
/-- The normalizer ratio is locally Lipschitz.  Its local constant is explicit:
near `x`, use the positive floor `Z_q(x)/2` and the global `1/τ` bounds for
both normalizers. -/
theorem locallyLipschitz_laplaceNormalizerRatio
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] :
    LocallyLipschitz (laplaceNormalizerRatio τ p q) := by
  intro x
  let Zp : E → ℝ := fun y => kernelNormalizer (laplaceKernel τ) p y
  let Zq : E → ℝ := fun y => kernelNormalizer (laplaceKernel τ) q y
  let κ : ℝ := Zq x / 2
  let U : Set E := {y | κ < Zq y}
  have hZqcont : Continuous Zq :=
    (lipschitzWith_laplaceKernelNormalizer hτ q).continuous
  have hκpos : 0 < κ := by
    have hxpos : 0 < Zq x := by
      dsimp [Zq]
      exact laplaceKernelNormalizer_pos q τ hτ x
    dsimp [κ]
    linarith
  have hUopen : IsOpen U := by
    exact isOpen_lt continuous_const hZqcont
  have hxU : x ∈ U := by
    have hxpos : 0 < Zq x := by
      dsimp [Zq]
      exact laplaceKernelNormalizer_pos q τ hτ x
    dsimp [U, κ]
    linarith
  refine ⟨Real.toNNReal (2 * (1 / τ) / κ ^ 2), U,
    hUopen.mem_nhds hxU, ?_⟩
  refine LipschitzOnWith.of_dist_le' ?_
  intro y hy z hz
  have hpY := laplaceKernelNormalizer_mem_Ioc_one hτ p y
  have hpZ := laplaceKernelNormalizer_mem_Ioc_one hτ p z
  have hqY := laplaceKernelNormalizer_mem_Ioc_one hτ q y
  have hqZ := laplaceKernelNormalizer_mem_Ioc_one hτ q z
  have hpLip := abs_kernelNormalizer_laplace_sub_le hτ p y z
  have hqLip := abs_kernelNormalizer_laplace_sub_le hτ q y z
  have hquot := abs_div_sub_div_le_of_unit_interval
    (a := Zp y) (b := Zq y) (c := Zp z) (d := Zq z)
    (κ := κ) (L := 1 / τ) (δ := dist y z)
    hpZ.1.le hpZ.2 hqY.1 hqZ.1 hqZ.2 hκpos (le_of_lt hy) (le_of_lt hz)
    (by positivity : 0 ≤ 1 / τ) (dist_nonneg)
    (by simpa [Real.dist_eq, dist_eq_norm] using hpLip)
    (by simpa [Real.dist_eq, dist_eq_norm] using hqLip)
  simpa [laplaceNormalizerRatio, Zp, Zq, Real.dist_eq] using hquot

end EuclideanRegularity

set_option linter.unusedSectionVars false in
/-- **G4 measure alignment.**  Under zero drift, the displacement field of
`p` is the normalizer ratio times the displacement field of `q`.

This is the divided form of the already-certified product identity
`Zq • Dp = Zp • Dq`; strict positivity of `Zq` justifies the division. -/
theorem laplaceDisplacementField_eq_ratio_smul_of_zeroDrift
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : E) :
    laplaceDisplacementField τ p x =
      laplaceNormalizerRatio τ p q x • laplaceDisplacementField τ q x := by
  set Zp := kernelNormalizer (laplaceKernel τ) p x with hZp
  set Zq := kernelNormalizer (laplaceKernel τ) q x with hZq
  set Dp := laplaceDisplacementField τ p x with hDp
  set Dq := laplaceDisplacementField τ q x with hDq
  have hZqpos : 0 < Zq := by
    rw [hZq]
    exact laplaceKernelNormalizer_pos q τ hτ x
  have halign : Zq • Dp = Zp • Dq := by
    rw [hZp, hZq, hDp, hDq]
    exact zeroDrift_displacementAligned hτ p q hzero x
  calc
    Dp = Zq⁻¹ • (Zq • Dp) := by
      rw [inv_smul_smul₀ hZqpos.ne']
    _ = Zq⁻¹ • (Zp • Dq) := by rw [halign]
    _ = (Zp / Zq) • Dq := by
      rw [smul_smul]
      congr 1
      rw [div_eq_mul_inv]
      exact mul_comm _ _
    _ = laplaceNormalizerRatio τ p q x • laplaceDisplacementField τ q x := by
      rw [hZp, hZq, hDq]
      rfl

/-- At every probe, zero drift aligns the Frechet derivatives of the actual
companion potentials by the positive normalizer ratio. -/
theorem hasFDerivAt_laplaceDisplacementPotential_ratio_of_zeroDrift
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : E) :
    HasFDerivAt (laplaceDisplacementPotential τ p)
      (innerSL ℝ
        (laplaceNormalizerRatio τ p q x •
          laplaceDisplacementField τ q x)) x := by
  have hp := hasFDerivAt_laplaceDisplacementPotential hτ p x
  rw [laplaceDisplacementField_eq_ratio_smul_of_zeroDrift hτ p q hzero x] at hp
  exact hp

/-- On the critical set of the `q` potential, zero drift also makes the `p`
potential critical. -/
theorem laplaceDisplacementField_eq_zero_of_zeroDrift_of_eq_zero
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) {x : E}
    (hq : laplaceDisplacementField τ q x = 0) :
    laplaceDisplacementField τ p x = 0 := by
  rw [laplaceDisplacementField_eq_ratio_smul_of_zeroDrift hτ p q hzero x,
    hq, smul_zero]

/-! ## The interior critical set -/

set_option linter.unusedSectionVars false in
/-- On a preconnected open region where the displacement field vanishes, the
actual companion potential is constant.  This is the rigorous first half of
the G4/P3 analysis of `int {D = 0}`. -/
theorem laplaceDisplacementPotential_eqOn_of_field_eq_zero
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ]
    {U : Set E} (hUopen : IsOpen U) (hUconn : IsPreconnected U)
    (hcrit : ∀ x ∈ U, laplaceDisplacementField τ μ x = 0)
    {a : E} (ha : a ∈ U) :
    U.EqOn (laplaceDisplacementPotential τ μ)
      (fun _ => laplaceDisplacementPotential τ μ a) := by
  have hdiff : DifferentiableOn ℝ (laplaceDisplacementPotential τ μ) U := by
    intro x _hx
    exact (hasFDerivAt_laplaceDisplacementPotential hτ μ x).differentiableAt.differentiableWithinAt
  have hfderiv : U.EqOn (fderiv ℝ (laplaceDisplacementPotential τ μ)) 0 := by
    intro x hx
    have h := (hasFDerivAt_laplaceDisplacementPotential hτ μ x).fderiv
    rw [hcrit x hx] at h
    simpa using h
  intro x hx
  exact hUopen.is_const_of_fderiv_eq_zero hUconn hdiff hfderiv hx ha

set_option linter.unusedSectionVars false in
/-- Under zero drift, both companion potentials are constant on every
preconnected open component of the `q`-critical set. -/
theorem laplaceDisplacementPotentials_eqOn_of_zeroDrift_of_qField_eq_zero
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {U : Set E} (hUopen : IsOpen U) (hUconn : IsPreconnected U)
    (hcrit : ∀ x ∈ U, laplaceDisplacementField τ q x = 0)
    {a : E} (ha : a ∈ U) :
    U.EqOn (laplaceDisplacementPotential τ p)
        (fun _ => laplaceDisplacementPotential τ p a) ∧
      U.EqOn (laplaceDisplacementPotential τ q)
        (fun _ => laplaceDisplacementPotential τ q a) := by
  have hpcrit : ∀ x ∈ U, laplaceDisplacementField τ p x = 0 := by
    intro x hx
    exact laplaceDisplacementField_eq_zero_of_zeroDrift_of_eq_zero
      hτ p q hzero (hcrit x hx)
  exact ⟨laplaceDisplacementPotential_eqOn_of_field_eq_zero
      hτ p hUopen hUconn hpcrit ha,
    laplaceDisplacementPotential_eqOn_of_field_eq_zero
      hτ q hUopen hUconn hcrit ha⟩

/-! ## Differential foliation at regular points -/

set_option linter.unusedSectionVars false in
/-- Product differentiation for an aligned pair of vector fields.  This is
kept abstract so it applies at every point where the Lipschitz fields in G4 are
classically differentiable (in particular, at the a.e. points supplied by
Rademacher). -/
theorem fderiv_eq_of_vectorField_eq_smul
    {R : E → ℝ} {Dp Dq : E → E} {x : E}
    {dR : E →L[ℝ] ℝ} {Ap Aq : E →L[ℝ] E}
    (hfun : Dp = fun z => R z • Dq z)
    (hR : HasFDerivAt R dR x)
    (hp : HasFDerivAt Dp Ap x)
    (hq : HasFDerivAt Dq Aq x) :
    Ap = R x • Aq + dR.smulRight (Dq x) := by
  have hrhs := hR.smul hq
  rw [hfun] at hp
  exact hp.unique hrhs

set_option linter.unusedSectionVars false in
/-- Symmetry of the two Hessians forces the exterior-product identity
`dR ∧ Dq = 0`.  It is written without differential-form infrastructure:
`dR(v) <Dq,w> = dR(w) <Dq,v>` for every pair of directions. -/
theorem laplaceFoliation_differential_parallel
    (ratio : ℝ) (Dq : E) (dR : E →L[ℝ] ℝ)
    (Ap Aq : E →L[ℝ] E)
    (hprod : ∀ v, Ap v = ratio • Aq v + dR v • Dq)
    (hpSymm : ∀ v w, ⟪Ap v, w⟫ = ⟪Ap w, v⟫)
    (hqSymm : ∀ v w, ⟪Aq v, w⟫ = ⟪Aq w, v⟫)
    (v w : E) :
    dR v * ⟪Dq, w⟫ = dR w * ⟪Dq, v⟫ := by
  have hv := congrArg (fun z : E => ⟪z, w⟫) (hprod v)
  have hw := congrArg (fun z : E => ⟪z, v⟫) (hprod w)
  simp only [inner_add_left, real_inner_smul_left] at hv hw
  rw [hpSymm v w, hqSymm v w] at hv
  linarith

/-- At a regular point (`Dq ≠ 0`), the ratio derivative annihilates every
direction tangent to the common potential leaf. -/
theorem laplaceFoliation_differential_tangent_eq_zero
    (ratio : ℝ) (Dq : E) (dR : E →L[ℝ] ℝ)
    (Ap Aq : E →L[ℝ] E)
    (hprod : ∀ v, Ap v = ratio • Aq v + dR v • Dq)
    (hpSymm : ∀ v w, ⟪Ap v, w⟫ = ⟪Ap w, v⟫)
    (hqSymm : ∀ v w, ⟪Aq v, w⟫ = ⟪Aq w, v⟫)
    (hDq : Dq ≠ 0) {v : E} (hv : ⟪Dq, v⟫ = 0) :
    dR v = 0 := by
  have hwedge := laplaceFoliation_differential_parallel ratio Dq dR Ap Aq
    hprod hpSymm hqSymm v Dq
  rw [hv] at hwedge
  simp only [mul_zero] at hwedge
  have hinner : 0 < ⟪Dq, Dq⟫ := real_inner_self_pos.mpr hDq
  nlinarith

/-! ## Local elliptic cancellation

The following scalar theorem is the exact algebra performed at a twice
differentiable point of a regular leaf chart.  `lapP` and `lapQ` denote the two
Laplacians; `slope` is `G'(ψq)`; `curvature` is `G''(ψq)`; and `gradSq` is
`‖∇ψq‖²`.
-/

/-- **G4/P1 cancellation, coordinate-free scalar core.**  The common elliptic
closure and the Laplacian chain rule cancel the unknown `lapQ` term exactly. -/
theorem laplaceFoliation_cancellation
    (τ C ψp ψq Zp Zq slope curvature gradSq lapP lapQ : ℝ)
    (hpde : ψp - τ ^ 2 * lapP = C * Zp)
    (hqde : ψq - τ ^ 2 * lapQ = C * Zq)
    (hratio : Zp = slope * Zq)
    (hchain : lapP = slope * lapQ + curvature * gradSq) :
    τ ^ 2 * curvature * gradSq = ψp - slope * ψq := by
  linear_combination -hpde + slope * hqde - C * hratio - τ ^ 2 * hchain

/-- The same cancellation in the local factorization notation
`ψp = G(ψq)`, `slope = G'(ψq)`, `curvature = G''(ψq)`. -/
theorem laplaceFoliation_cancellation_of_localFactorization
    (τ C s Gs Gs' Gs'' gradSq lapP lapQ Zp Zq : ℝ)
    (hpde : Gs - τ ^ 2 * lapP = C * Zp)
    (hqde : s - τ ^ 2 * lapQ = C * Zq)
    (hratio : Zp = Gs' * Zq)
    (hchain : lapP = Gs' * lapQ + Gs'' * gradSq) :
    τ ^ 2 * Gs'' * gradSq = Gs - s * Gs' := by
  simpa [mul_comm s Gs'] using
    laplaceFoliation_cancellation τ C Gs s Zp Zq Gs' Gs'' gradSq lapP lapQ
      hpde hqde hratio hchain

/-! ## The non-degenerate leaf branch -/

/-- Natural form of the preceding theorem with the common leaf right-hand
side `G(s) - s G'(s)`. -/
theorem laplaceFoliation_nondegenerateLeaf_at
    {τ s Gs Gs' Gs'' gradSq₁ gradSq₂ : ℝ} (hτ : τ ≠ 0)
    (hne : gradSq₁ ≠ gradSq₂)
    (h₁ : τ ^ 2 * Gs'' * gradSq₁ = Gs - s * Gs')
    (h₂ : τ ^ 2 * Gs'' * gradSq₂ = Gs - s * Gs') :
    Gs'' = 0 ∧ Gs = s * Gs' := by
  have hprod : τ ^ 2 * Gs'' * (gradSq₁ - gradSq₂) = 0 := by
    nlinarith [h₁, h₂]
  have hτsq : τ ^ 2 ≠ 0 := pow_ne_zero 2 hτ
  have hdiff : gradSq₁ - gradSq₂ ≠ 0 := sub_ne_zero.mpr hne
  have hcurv : Gs'' = 0 := by
    rcases mul_eq_zero.mp hprod with hpair | hdiff0
    · rcases mul_eq_zero.mp hpair with hτ0 | hcurv0
      · exact (hτsq hτ0).elim
      · exact hcurv0
    · exact (hdiff hdiff0).elim
  refine ⟨hcurv, ?_⟩
  have hzero : 0 = Gs - s * Gs' := by simpa [hcurv] using h₁
  linarith

/-- A zero second derivative together with the cancellation equation forces
the local affine factorization through the origin at that leaf value. -/
theorem laplaceFoliation_proportional_at
    {τ s Gs Gs' Gs'' gradSq : ℝ}
    (hcurv : Gs'' = 0)
    (hcancel : τ ^ 2 * Gs'' * gradSq = Gs - s * Gs') :
    Gs = s * Gs' := by
  have hzero : 0 = Gs - s * Gs' := by simpa [hcurv] using hcancel
  linarith

/-- The pointwise non-degenerate conclusion propagates across a connected open
interval of leaf values: zero derivative of `G'` makes the slope constant, and
`G(s)=sG'(s)` removes the affine intercept. -/
theorem laplaceFoliation_proportionalOn_of_nondegenerateInterval
    {I : Set ℝ} (hIopen : IsOpen I) (hIconn : IsPreconnected I)
    {G G' : ℝ → ℝ} (hG'diff : DifferentiableOn ℝ G' I)
    (hG'' : I.EqOn (deriv G') 0)
    (horigin : ∀ s ∈ I, G s = s * G' s)
    {s₀ : ℝ} (hs₀ : s₀ ∈ I) :
    I.EqOn G (fun s => G' s₀ * s) := by
  intro s hs
  have hslope : G' s = G' s₀ :=
    hIopen.is_const_of_deriv_eq_zero hIconn hG'diff hG'' hs hs₀
  rw [horigin s hs, hslope, mul_comm]

end DriftingIdentifiability
