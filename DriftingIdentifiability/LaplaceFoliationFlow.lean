import DriftingIdentifiability.LaplaceFoliationChart
import DriftingIdentifiability.LaplaceACPropagation
import Mathlib.Analysis.ODE.ExistUnique

/-!
# G4 gradient-flow propagation

The regular foliation equations admit a simpler one-dimensional reduction
than an explicit choice of leaf coordinates.  Along an integral curve
`gamma' = D_q`, define

`H = psi_p - (Z_p / Z_q) psi_q`.

The differentiated alignment gives `H' = -psi_q R'`, while the pointwise
elliptic cancellation gives `tau^2 R' = H`.  Hence

`H' = -(psi_q / tau^2) H`.

This file proves that identity for the actual measure objects and packages the
finite-interval uniqueness consequence.  It is the seeded part of G4/P2: one
zero of `H` propagates along the whole regular gradient orbit.  The remaining
geometric question is whether every seedless degenerate component can occur.
-/

open MeasureTheory Filter Set Topology
open scoped NNReal RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
  [SecondCountableTopology E] [FiniteDimensional ℝ E]

/-- The scalar defect left by the foliation cancellation equation. -/
noncomputable def laplaceFoliationDefect
    (τ : ℝ) (p q : Measure E) (x : E) : ℝ :=
  laplaceDisplacementPotential τ p x -
    laplaceNormalizerRatio τ p q x * laplaceDisplacementPotential τ q x

/-- The foliation defect is continuous. -/
theorem continuous_laplaceFoliationDefect
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] :
    Continuous (laplaceFoliationDefect τ p q) := by
  unfold laplaceFoliationDefect
  have hp : Continuous (laplaceDisplacementPotential τ p) := by
    rw [continuous_iff_continuousAt]
    exact fun x => (hasFDerivAt_laplaceDisplacementPotential hτ p x).continuousAt
  have hq : Continuous (laplaceDisplacementPotential τ q) := by
    rw [continuous_iff_continuousAt]
    exact fun x => (hasFDerivAt_laplaceDisplacementPotential hτ q x).continuousAt
  exact hp.sub ((continuous_laplaceNormalizerRatio hτ p q).mul hq)

omit [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]
    [FiniteDimensional ℝ E] in
/-- The displacement field of a probability measure has the uniform
`tau/e` bound inherited from its point-source integrand. -/
theorem norm_laplaceDisplacementField_le
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ]
    (x : E) :
    ‖laplaceDisplacementField τ μ x‖ ≤ τ * Real.exp (-1) := by
  unfold laplaceDisplacementField
  have hpoint : ∀ᵐ y ∂μ,
      ‖laplaceKernel τ x y • (y - x)‖ ≤ τ * Real.exp (-1) := by
    filter_upwards with y
    rw [norm_smul, laplaceKernel_eq_exp, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _), norm_sub_rev, mul_comm]
    exact mul_exp_neg_div_le hτ (norm_nonneg _)
  simpa using norm_integral_le_of_norm_le_const hpoint

omit [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]
    [FiniteDimensional ℝ E] in
/-- The integrated displacement Hessian inherits the uniform point-source
operator-norm bound `2`. -/
theorem norm_laplaceDisplacementHessian_le
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ]
    (x : E) :
    ‖laplaceDisplacementHessian τ μ x‖ ≤ 2 := by
  unfold laplaceDisplacementHessian
  have hpoint : ∀ᵐ y ∂μ, ‖laplaceDisplacementKernelHessian τ x y‖ ≤ 2 :=
    ae_of_all μ fun y => norm_laplaceDisplacementKernelHessian_le hτ x y
  simpa using norm_integral_le_of_norm_le_const hpoint

/-- The classical Hessian bound gives a dimension-free global Lipschitz
constant for the displacement field, without Haar or density assumptions. -/
theorem lipschitzWith_laplaceDisplacementField
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ] :
    LipschitzWith 2 (laplaceDisplacementField τ μ) := by
  apply lipschitzWith_of_nnnorm_fderiv_le (𝕜 := ℝ)
  · intro x
    exact (hasFDerivAt_laplaceDisplacementField hτ μ x).differentiableAt
  · intro x
    rw [(hasFDerivAt_laplaceDisplacementField hτ μ x).fderiv]
    exact_mod_cast norm_laplaceDisplacementHessian_le hτ μ x

/-- Every displacement field admits a local integral curve through every
point.  This discharges the curve-existence side of the seeded propagation
theorems using Picard--Lindelöf and the explicit global bounds above. -/
theorem exists_local_laplaceDisplacementGradientCurve
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ]
    (x : E) :
    ∃ ε : ℝ, 0 < ε ∧ ∃ γ : ℝ → E, γ 0 = x ∧
      ∀ t ∈ Ioo (-ε) ε,
        HasDerivAt γ (laplaceDisplacementField τ μ (γ t)) t := by
  let L : ℝ≥0 := ⟨τ * Real.exp (-1),
    mul_nonneg (le_of_lt hτ) (le_of_lt (Real.exp_pos _))⟩
  let ε : ℝ := 1 / (2 * ((L : ℝ) + 1))
  have hLnonneg : 0 ≤ (L : ℝ) := L.coe_nonneg
  have hε : 0 < ε := by
    dsimp [ε]
    positivity
  let t₀ : Icc (-ε) ε := ⟨0, by constructor <;> linarith⟩
  have hPL : IsPicardLindelof
      (fun _ : ℝ => laplaceDisplacementField τ μ) t₀ x 1 0 L 2 := by
    refine {
      lipschitzOnWith := ?_
      continuousOn := ?_
      norm_le := ?_
      mul_max_le := ?_ }
    · intro t ht
      exact (lipschitzWith_laplaceDisplacementField hτ μ).lipschitzOnWith
    · intro y hy
      exact continuous_const.continuousOn
    · intro t ht y hy
      exact norm_laplaceDisplacementField_le hτ μ y
    · change (L : ℝ) * max (ε - 0) (0 - -ε) ≤ (1 : ℝ) - 0
      rw [sub_zero, zero_sub, neg_neg, max_self, sub_zero]
      dsimp [ε]
      have hden : 0 < 2 * ((L : ℝ) + 1) := by positivity
      rw [one_div, ← div_eq_mul_inv]
      rw [div_le_iff₀ hden]
      nlinarith
  rcases hPL.exists_eq_forall_mem_Icc_hasDerivWithinAt₀ with ⟨γ, hγ₀, hγ⟩
  refine ⟨ε, hε, γ, ?_, ?_⟩
  · simpa [t₀] using hγ₀
  · intro t ht
    exact (hγ t (Ioo_subset_Icc_self ht)).hasDerivAt
      (Icc_mem_nhds ht.1 ht.2)

/-- On an open set where the displacement field vanishes identically, its
classical integrated Hessian vanishes pointwise as well. -/
theorem laplaceDisplacementHessian_eq_zero_of_eq_zero_on_open
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ]
    {U : Set E} (hUopen : IsOpen U)
    (hfield : ∀ x ∈ U, laplaceDisplacementField τ μ x = 0)
    {x : E} (hx : x ∈ U) :
    laplaceDisplacementHessian τ μ x = 0 := by
  have heq : laplaceDisplacementField τ μ =ᶠ[𝓝 x] fun _ : E => 0 := by
    filter_upwards [hUopen.mem_nhds hx] with y hy
    exact hfield y hy
  have hzero : HasFDerivAt (laplaceDisplacementField τ μ) 0 x :=
    (hasFDerivAt_const (𝕜 := ℝ) (0 : E) x).congr_of_eventuallyEq heq
  exact (hasFDerivAt_laplaceDisplacementField hτ μ x).unique hzero

/-- The companion Laplacian vanishes on the interior of a zero-displacement
region. -/
theorem laplaceDisplacementLaplacian_eq_zero_of_eq_zero_on_open
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ]
    {U : Set E} (hUopen : IsOpen U)
    (hfield : ∀ x ∈ U, laplaceDisplacementField τ μ x = 0)
    {x : E} (hx : x ∈ U) :
    laplaceDisplacementLaplacian τ μ x = 0 := by
  unfold laplaceDisplacementLaplacian
  rw [laplaceDisplacementHessian_eq_zero_of_eq_zero_on_open
    hτ μ hUopen hfield hx]
  simp [continuousLinearMapTrace]

/-- **Critical-component constancy.**  On every preconnected open component
of `{D_q = 0}`, the elliptic companion identity upgrades constancy of the two
potentials to constancy of both normalizers and hence of their ratio. -/
theorem laplaceNormalizerRatio_eqOn_of_zeroDrift_of_qField_eq_zero
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {U : Set E} (hUopen : IsOpen U) (hUconn : IsPreconnected U)
    (hcrit : ∀ x ∈ U, laplaceDisplacementField τ q x = 0)
    {a : E} (ha : a ∈ U) :
    U.EqOn (laplaceNormalizerRatio τ p q)
      (fun _ => laplaceNormalizerRatio τ p q a) := by
  have hpcrit : ∀ x ∈ U, laplaceDisplacementField τ p x = 0 := by
    intro x hx
    exact laplaceDisplacementField_eq_zero_of_zeroDrift_of_eq_zero
      hτ p q hzero (hcrit x hx)
  rcases laplaceDisplacementPotentials_eqOn_of_zeroDrift_of_qField_eq_zero
    hτ p q hzero hUopen hUconn hcrit ha with ⟨hpconst, hqconst⟩
  let C : ℝ := ((Module.finrank ℝ E : ℝ) + 1) * τ ^ 2
  have hCpos : 0 < C := by
    dsimp [C]
    positivity
  intro x hx
  have hpLapx := laplaceDisplacementLaplacian_eq_zero_of_eq_zero_on_open
    hτ p hUopen hpcrit hx
  have hpLapa := laplaceDisplacementLaplacian_eq_zero_of_eq_zero_on_open
    hτ p hUopen hpcrit ha
  have hqLapx := laplaceDisplacementLaplacian_eq_zero_of_eq_zero_on_open
    hτ q hUopen hcrit hx
  have hqLapa := laplaceDisplacementLaplacian_eq_zero_of_eq_zero_on_open
    hτ q hUopen hcrit ha
  have hpdeX := laplaceDisplacementPotential_elliptic hτ p x
  have hpdeA := laplaceDisplacementPotential_elliptic hτ p a
  have hqdeX := laplaceDisplacementPotential_elliptic hτ q x
  have hqdeA := laplaceDisplacementPotential_elliptic hτ q a
  change laplaceDisplacementPotential τ p x - τ ^ 2 *
      laplaceDisplacementLaplacian τ p x =
      C * kernelNormalizer (laplaceKernel τ) p x at hpdeX
  change laplaceDisplacementPotential τ p a - τ ^ 2 *
      laplaceDisplacementLaplacian τ p a =
      C * kernelNormalizer (laplaceKernel τ) p a at hpdeA
  change laplaceDisplacementPotential τ q x - τ ^ 2 *
      laplaceDisplacementLaplacian τ q x =
      C * kernelNormalizer (laplaceKernel τ) q x at hqdeX
  change laplaceDisplacementPotential τ q a - τ ^ 2 *
      laplaceDisplacementLaplacian τ q a =
      C * kernelNormalizer (laplaceKernel τ) q a at hqdeA
  have hZp : kernelNormalizer (laplaceKernel τ) p x =
      kernelNormalizer (laplaceKernel τ) p a := by
    rw [hpLapx, mul_zero, sub_zero] at hpdeX
    rw [hpLapa, mul_zero, sub_zero] at hpdeA
    nlinarith [hpconst hx]
  have hZq : kernelNormalizer (laplaceKernel τ) q x =
      kernelNormalizer (laplaceKernel τ) q a := by
    rw [hqLapx, mul_zero, sub_zero] at hqdeX
    rw [hqLapa, mul_zero, sub_zero] at hqdeA
    nlinarith [hqconst hx]
  unfold laplaceNormalizerRatio
  rw [hZp, hZq]

/-- The normalizer ratio is locally constant at every interior point of the
critical set `{D_q = 0}`.  Thus critical-set interiors are no longer a gluing
gap in G4/P3. -/
theorem laplaceNormalizerRatio_eventuallyEq_of_mem_interior_qField_eq_zero
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {x : E} (hx : x ∈ interior {y | laplaceDisplacementField τ q y = 0}) :
    ∀ᶠ y in 𝓝 x,
      laplaceNormalizerRatio τ p q y = laplaceNormalizerRatio τ p q x := by
  rcases Metric.mem_nhds_iff.mp
    (isOpen_interior.mem_nhds hx) with ⟨ε, hε, hball⟩
  have hcrit : ∀ y ∈ Metric.ball x ε,
      laplaceDisplacementField τ q y = 0 := by
    intro y hy
    have hyset : y ∈ {z | laplaceDisplacementField τ q z = 0} :=
      interior_subset (hball hy)
    exact hyset
  have heq := laplaceNormalizerRatio_eqOn_of_zeroDrift_of_qField_eq_zero
    hτ p q hzero Metric.isOpen_ball (convex_ball x ε).isPreconnected
      hcrit (Metric.mem_ball_self hε)
  filter_upwards [Metric.ball_mem_nhds x hε] with y hy
  exact heq hy

/-- Along a regular `q`-gradient curve, direct differentiation gives
`H' = -psi_q R'`. -/
theorem hasDerivAt_laplaceFoliationDefect_comp_gradientCurve
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {γ : ℝ → E} {t : ℝ}
    (hγ : HasDerivAt γ (laplaceDisplacementField τ q (γ t)) t)
    (hreg : laplaceDisplacementField τ q (γ t) ≠ 0) :
    HasDerivAt (fun u => laplaceFoliationDefect τ p q (γ u))
      (-laplaceDisplacementPotential τ q (γ t) *
        fderiv ℝ (laplaceNormalizerRatio τ p q) (γ t)
          (laplaceDisplacementField τ q (γ t))) t := by
  let R := laplaceNormalizerRatio τ p q
  let ψp := laplaceDisplacementPotential τ p
  let ψq := laplaceDisplacementPotential τ q
  let Dq := laplaceDisplacementField τ q
  have hR : HasDerivAt (R ∘ γ)
      (fderiv ℝ R (γ t) (Dq (γ t))) t :=
    (differentiableAt_laplaceNormalizerRatio_of_zeroDrift
      hτ p q hzero hreg).hasFDerivAt.comp_hasDerivAt t hγ
  have hp : HasDerivAt (ψp ∘ γ)
      (⟪laplaceDisplacementField τ p (γ t), Dq (γ t)⟫) t := by
    have hp0 :=
      (hasFDerivAt_laplaceDisplacementPotential hτ p (γ t)).comp_hasDerivAt t hγ
    simpa only [ψp, Dq, innerSL_apply_apply] using hp0
  have hq : HasDerivAt (ψq ∘ γ)
      (⟪Dq (γ t), Dq (γ t)⟫) t := by
    have hq0 :=
      (hasFDerivAt_laplaceDisplacementPotential hτ q (γ t)).comp_hasDerivAt t hγ
    simpa only [ψq, Dq, innerSL_apply_apply] using hq0
  have hprod := hR.mul hq
  have hsub := hp.sub hprod
  have halign := laplaceDisplacementField_eq_ratio_smul_of_zeroDrift
    hτ p q hzero (γ t)
  change HasDerivAt (laplaceFoliationDefect τ p q ∘ γ) _ t
  rw [show laplaceFoliationDefect τ p q ∘ γ =
      (ψp ∘ γ) - (R ∘ γ) * (ψq ∘ γ) by
    funext u
    rfl]
  apply hsub.congr_deriv
  rw [halign]
  simp only [real_inner_smul_left]
  simp only [R, ψq, Dq, Function.comp_apply]
  ring

/-- **Gradient-flow Abel equation.**  At a regular point of a `q`-gradient
curve the actual defect satisfies the homogeneous scalar ODE

`H' = -(psi_q / tau^2) H`.
-/
theorem hasDerivAt_laplaceFoliationDefect_comp_gradientCurve_abel
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {γ : ℝ → E} {t : ℝ}
    (hγ : HasDerivAt γ (laplaceDisplacementField τ q (γ t)) t)
    (hreg : laplaceDisplacementField τ q (γ t) ≠ 0) :
    HasDerivAt (fun u => laplaceFoliationDefect τ p q (γ u))
      (-(laplaceDisplacementPotential τ q (γ t) / τ ^ 2) *
        laplaceFoliationDefect τ p q (γ t)) t := by
  have hbase := hasDerivAt_laplaceFoliationDefect_comp_gradientCurve
    hτ p q hzero hγ hreg
  have hcancel := laplaceFoliation_measureCancellation hτ p q hzero hreg
  apply hbase.congr_deriv
  have hτsq : τ ^ 2 ≠ 0 := pow_ne_zero 2 (ne_of_gt hτ)
  unfold laplaceFoliationDefect
  rw [← hcancel]
  field_simp [hτsq]

/-- Vanishing foliation defect kills the full derivative of the normalizer
ratio at a regular point: cancellation kills the normal derivative and
Hessian symmetry already kills every tangent derivative. -/
theorem fderiv_laplaceNormalizerRatio_eq_zero_of_defect_eq_zero
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {x : E} (hreg : laplaceDisplacementField τ q x ≠ 0)
    (hdefect : laplaceFoliationDefect τ p q x = 0) :
    fderiv ℝ (laplaceNormalizerRatio τ p q) x = 0 := by
  let dR := fderiv ℝ (laplaceNormalizerRatio τ p q) x
  let Dq := laplaceDisplacementField τ q x
  have hnormal : dR Dq = 0 := by
    have hcancel := laplaceFoliation_measureCancellation hτ p q hzero hreg
    change τ ^ 2 * dR Dq = laplaceFoliationDefect τ p q x at hcancel
    rw [hdefect] at hcancel
    exact (mul_eq_zero.mp hcancel).resolve_left
      (pow_ne_zero 2 (ne_of_gt hτ))
  have hprod := laplaceDisplacementHessian_product_of_zeroDrift
    hτ p q hzero hreg
  ext v
  have hwedge := laplaceFoliation_differential_parallel
    (laplaceNormalizerRatio τ p q x) Dq dR
    (laplaceDisplacementHessian τ p x)
    (laplaceDisplacementHessian τ q x)
    (fun w => congrArg (fun A : E →L[ℝ] E => A w) hprod)
    (laplaceDisplacementHessian_symmetric hτ p x)
    (laplaceDisplacementHessian_symmetric hτ q x) v Dq
  rw [hnormal, zero_mul] at hwedge
  have hinner : ⟪Dq, Dq⟫ ≠ 0 := inner_self_ne_zero.mpr hreg
  exact (mul_eq_zero.mp hwedge).resolve_right hinner

/-- On an open connected regular region where the defect has been seeded to
zero, the actual normalizer ratio is constant. -/
theorem laplaceNormalizerRatio_eqOn_of_defect_eq_zero_on_regular_open
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {U : Set E} (hUopen : IsOpen U) (hUconn : IsPreconnected U)
    (hreg : ∀ x ∈ U, laplaceDisplacementField τ q x ≠ 0)
    (hdefect : ∀ x ∈ U, laplaceFoliationDefect τ p q x = 0)
    {a : E} (ha : a ∈ U) :
    U.EqOn (laplaceNormalizerRatio τ p q)
      (fun _ => laplaceNormalizerRatio τ p q a) := by
  have hdiff : DifferentiableOn ℝ (laplaceNormalizerRatio τ p q) U := by
    intro x hx
    exact (differentiableAt_laplaceNormalizerRatio_of_zeroDrift
      hτ p q hzero (hreg x hx)).differentiableWithinAt
  have hfderiv : U.EqOn (fderiv ℝ (laplaceNormalizerRatio τ p q)) 0 := by
    intro x hx
    exact fderiv_laplaceNormalizerRatio_eq_zero_of_defect_eq_zero
      hτ p q hzero (hreg x hx) (hdefect x hx)
  intro x hx
  exact hUopen.is_const_of_fderiv_eq_zero hUconn hdiff hfderiv hx ha

/-- Two-sided uniqueness for a scalar homogeneous ODE on a compact interval.
This is a symmetric wrapper around the one-sided Grönwall lemma used by the
one-dimensional Abel development. -/
theorem abel_zero_propagates_Icc_of_eq_zero_at
    {W c : ℝ → ℝ} {a b t₀ : ℝ}
    (ht₀ : t₀ ∈ Icc a b)
    (hcont : ContinuousOn W (Icc a b))
    (hderiv : ∀ t ∈ Icc a b, HasDerivAt W (c t * W t) t)
    (hbound : ∃ K : ℝ, 0 ≤ K ∧ ∀ t ∈ Icc a b, ‖c t‖ ≤ K)
    (hzero : W t₀ = 0) :
    ∀ t ∈ Icc a b, W t = 0 := by
  rcases hbound with ⟨K, hKnonneg, hK⟩
  intro t ht
  rcases le_total t t₀ with htt₀ | ht₀t
  · let Wr : ℝ → ℝ := W ∘ fun u => -u
    let cr : ℝ → ℝ := fun u => -c (-u)
    have hmap : MapsTo (fun u : ℝ => -u) (Icc (-t₀) (-t)) (Icc a b) := by
      intro u hu
      rcases hu with ⟨hu₁, hu₂⟩
      change a ≤ -u ∧ -u ≤ b
      constructor <;> linarith [ht.1, ht₀.2, hu₁, hu₂]
    have hWrcont : ContinuousOn Wr (Icc (-t₀) (-t)) :=
      hcont.comp (continuous_neg.comp continuous_id).continuousOn hmap
    have hWrderiv : ∀ u ∈ Ico (-t₀) (-t),
        HasDerivWithinAt Wr (cr u * Wr u) (Ici u) u := by
      intro u hu
      have hmem : -u ∈ Icc a b := hmap (Ico_subset_Icc_self hu)
      have hneg : HasDerivAt (fun z : ℝ => -z) (-1) u :=
        (hasDerivAt_id u).neg
      have hcomp := (hderiv (-u) hmem).comp u hneg
      have heq : c (-u) * W (-u) * (-1) = cr u * Wr u := by
        simp only [Wr, cr, Function.comp_apply]
        ring
      change HasDerivWithinAt (W ∘ fun z : ℝ => -z)
        (cr u * (W ∘ fun z : ℝ => -z) u) (Ici u) u
      exact (hcomp.congr_deriv heq).hasDerivWithinAt
    have hWrbound : ∃ L : ℝ, 0 ≤ L ∧
        ∀ u ∈ Ico (-t₀) (-t), ‖cr u‖ ≤ L := by
      refine ⟨K, hKnonneg, ?_⟩
      intro u hu
      simp only [cr, norm_neg]
      exact hK (-u) (hmap (Ico_subset_Icc_self hu))
    have hWrzero : Wr (-t₀) = 0 := by simpa [Wr] using hzero
    have hprop := abel_zero_propagates_Icc hWrcont hWrderiv hWrbound hWrzero
    have hnegmem : -t ∈ Icc (-t₀) (-t) := by constructor <;> linarith
    simpa [Wr] using hprop (-t) hnegmem
  · have hWcont : ContinuousOn W (Icc t₀ t) :=
      hcont.mono (Icc_subset_Icc ht₀.1 ht.2)
    have hWderiv : ∀ u ∈ Ico t₀ t,
        HasDerivWithinAt W (c u * W u) (Ici u) u := by
      intro u hu
      have humem : u ∈ Icc a b := by
        constructor <;> linarith [ht₀.1, ht.2, hu.1, hu.2]
      exact (hderiv u humem).hasDerivWithinAt
    have hWbound : ∃ L : ℝ, 0 ≤ L ∧
        ∀ u ∈ Ico t₀ t, ‖c u‖ ≤ L := by
      refine ⟨K, hKnonneg, ?_⟩
      intro u hu
      apply hK u
      constructor <;> linarith [ht₀.1, ht.2, hu.1, hu.2]
    exact abel_zero_propagates_Icc hWcont hWderiv hWbound hzero t
      (right_mem_Icc.mpr ht₀t)

/-- A zero defect propagates forward along every regular gradient-curve
segment.  The bounded-coefficient hypothesis needed by Grönwall is automatic
from continuity of the actual companion potential on the compact interval. -/
theorem laplaceFoliationDefect_eq_zero_on_Icc_of_gradientCurve
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {γ : ℝ → E} {a b : ℝ}
    (hγ : ∀ t ∈ Icc a b,
      HasDerivAt γ (laplaceDisplacementField τ q (γ t)) t)
    (hreg : ∀ t ∈ Ico a b, laplaceDisplacementField τ q (γ t) ≠ 0)
    (ha : laplaceFoliationDefect τ p q (γ a) = 0) :
    ∀ t ∈ Icc a b, laplaceFoliationDefect τ p q (γ t) = 0 := by
  let H : ℝ → ℝ := fun t => laplaceFoliationDefect τ p q (γ t)
  let c : ℝ → ℝ := fun t =>
    -(laplaceDisplacementPotential τ q (γ t) / τ ^ 2)
  have hγcont : ContinuousOn γ (Icc a b) :=
    fun t ht => (hγ t ht).continuousAt.continuousWithinAt
  have hHcont : ContinuousOn H (Icc a b) :=
    (continuous_laplaceFoliationDefect hτ p q).continuousOn.comp
      hγcont (mapsTo_image γ (Icc a b))
  have hψq : Continuous (laplaceDisplacementPotential τ q) := by
    rw [continuous_iff_continuousAt]
    exact fun x =>
      (hasFDerivAt_laplaceDisplacementPotential hτ q x).continuousAt
  have hψqcomp : ContinuousOn
      (fun t => laplaceDisplacementPotential τ q (γ t)) (Icc a b) :=
    hψq.continuousOn.comp hγcont (mapsTo_image γ (Icc a b))
  have hccont : ContinuousOn c (Icc a b) := by
    exact hψqcomp.div_const (τ ^ 2) |>.neg
  have hcbound : ∃ K : ℝ, 0 ≤ K ∧ ∀ t ∈ Ico a b, ‖c t‖ ≤ K := by
    rcases isCompact_Icc.bddAbove_image hccont.norm with ⟨K, hK⟩
    refine ⟨max 0 K, le_max_left _ _, ?_⟩
    intro t ht
    exact le_trans (hK (mem_image_of_mem (fun u => ‖c u‖)
      (Ico_subset_Icc_self ht))) (le_max_right _ _)
  apply abel_zero_propagates_Icc hHcont
  · intro t ht
    exact (hasDerivAt_laplaceFoliationDefect_comp_gradientCurve_abel
      hτ p q hzero (hγ t (Ico_subset_Icc_self ht)) (hreg t ht)).hasDerivWithinAt
  · exact hcbound
  · exact ha

/-- **Two-sided seeded-orbit propagation.**  On a compact regular segment of
an actual `q`-gradient curve, a zero of the foliation defect at any parameter
value forces the defect to vanish on the whole segment. -/
theorem laplaceFoliationDefect_eq_zero_on_Icc_of_gradientCurve_of_seed
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {γ : ℝ → E} {a b t₀ : ℝ} (ht₀ : t₀ ∈ Icc a b)
    (hγ : ∀ t ∈ Icc a b,
      HasDerivAt γ (laplaceDisplacementField τ q (γ t)) t)
    (hreg : ∀ t ∈ Icc a b, laplaceDisplacementField τ q (γ t) ≠ 0)
    (hseed : laplaceFoliationDefect τ p q (γ t₀) = 0) :
    ∀ t ∈ Icc a b, laplaceFoliationDefect τ p q (γ t) = 0 := by
  let H : ℝ → ℝ := fun t => laplaceFoliationDefect τ p q (γ t)
  let c : ℝ → ℝ := fun t =>
    -(laplaceDisplacementPotential τ q (γ t) / τ ^ 2)
  have hγcont : ContinuousOn γ (Icc a b) :=
    fun t ht => (hγ t ht).continuousAt.continuousWithinAt
  have hHcont : ContinuousOn H (Icc a b) :=
    (continuous_laplaceFoliationDefect hτ p q).continuousOn.comp
      hγcont (mapsTo_image γ (Icc a b))
  have hHderiv : ∀ t ∈ Icc a b, HasDerivAt H (c t * H t) t := by
    intro t ht
    exact hasDerivAt_laplaceFoliationDefect_comp_gradientCurve_abel
      hτ p q hzero (hγ t ht) (hreg t ht)
  have hψq : Continuous (laplaceDisplacementPotential τ q) := by
    rw [continuous_iff_continuousAt]
    exact fun x =>
      (hasFDerivAt_laplaceDisplacementPotential hτ q x).continuousAt
  have hψqcomp : ContinuousOn
      (fun t => laplaceDisplacementPotential τ q (γ t)) (Icc a b) :=
    hψq.continuousOn.comp hγcont (mapsTo_image γ (Icc a b))
  have hccont : ContinuousOn c (Icc a b) :=
    hψqcomp.div_const (τ ^ 2) |>.neg
  have hcbound : ∃ K : ℝ, 0 ≤ K ∧ ∀ t ∈ Icc a b, ‖c t‖ ≤ K := by
    rcases isCompact_Icc.bddAbove_image hccont.norm with ⟨K, hK⟩
    refine ⟨max 0 K, le_max_left _ _, ?_⟩
    intro t ht
    exact le_trans (hK (mem_image_of_mem (fun u => ‖c u‖) ht))
      (le_max_right _ _)
  exact abel_zero_propagates_Icc_of_eq_zero_at ht₀ hHcont hHderiv
    hcbound hseed

/-- **Local seeded flow, with all analytic side conditions discharged.**  A
regular point whose defect is zero lies on a nontrivial local gradient-curve
segment on which the defect remains zero. -/
theorem exists_local_gradientCurve_laplaceFoliationDefect_eq_zero
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {x : E} (hreg : laplaceDisplacementField τ q x ≠ 0)
    (hseed : laplaceFoliationDefect τ p q x = 0) :
    ∃ ε : ℝ, 0 < ε ∧ ∃ γ : ℝ → E,
      γ 0 = x ∧
      (∀ t ∈ Icc (-ε) ε,
        HasDerivAt γ (laplaceDisplacementField τ q (γ t)) t) ∧
      (∀ t ∈ Icc (-ε) ε,
        laplaceFoliationDefect τ p q (γ t) = 0) := by
  rcases exists_local_laplaceDisplacementGradientCurve hτ q x with
    ⟨ε₀, hε₀, γ, hγ₀, hγ⟩
  have hzero_mem : (0 : ℝ) ∈ Ioo (-ε₀) ε₀ := by constructor <;> linarith
  have hγcont : ContinuousAt γ 0 := (hγ 0 hzero_mem).continuousAt
  have hDcont : ContinuousAt
      (fun t => laplaceDisplacementField τ q (γ t)) 0 :=
    (continuous_laplaceDisplacementField hτ q).continuousAt.comp hγcont
  have hne : ∀ᶠ t in 𝓝 (0 : ℝ),
      laplaceDisplacementField τ q (γ t) ≠ 0 := by
    apply hDcont.eventually_ne
    simpa [hγ₀] using hreg
  have hdomain : ∀ᶠ t in 𝓝 (0 : ℝ), t ∈ Ioo (-ε₀) ε₀ :=
    Ioo_mem_nhds (by linarith) (by linarith)
  rcases Metric.mem_nhds_iff.mp (hne.and hdomain) with ⟨δ, hδ, hball⟩
  let ε := δ / 2
  have hε : 0 < ε := by dsimp [ε]; linarith
  have hsub : Icc (-ε) ε ⊆ Metric.ball (0 : ℝ) δ := by
    intro t ht
    rw [mem_ball_zero_iff, Real.norm_eq_abs]
    dsimp [ε] at ht
    rw [abs_lt]
    constructor <;> linarith [ht.1, ht.2, hδ]
  have hcurve : ∀ t ∈ Icc (-ε) ε,
      HasDerivAt γ (laplaceDisplacementField τ q (γ t)) t := by
    intro t ht
    exact hγ t (hball (hsub ht)).2
  have hregular : ∀ t ∈ Icc (-ε) ε,
      laplaceDisplacementField τ q (γ t) ≠ 0 := by
    intro t ht
    exact (hball (hsub ht)).1
  have hseed₀ : laplaceFoliationDefect τ p q (γ 0) = 0 := by
    simpa [hγ₀] using hseed
  have hprop := laplaceFoliationDefect_eq_zero_on_Icc_of_gradientCurve_of_seed
    hτ p q hzero (t₀ := 0) (by constructor <;> linarith)
      hcurve hregular hseed₀
  exact ⟨ε, hε, γ, hγ₀, hcurve, hprop⟩

end DriftingIdentifiability
