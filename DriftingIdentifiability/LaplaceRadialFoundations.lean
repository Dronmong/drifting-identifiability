import DriftingIdentifiability.LaplacianGaussianConverse

/-!
# Radial Laplace converse, milestone L0: the displacement potential

This file is the foundation layer (`L0`) of the ℓ²/radial higher-dimensional
Laplace program described in `LaplaceHigherDim.md`.  It establishes, for an
arbitrary finite measure on a real inner-product space, that the paper's
Laplace mean-shift **numerator**

`D_μ(x) = ∫ e^{-‖x-y‖/τ} • (y - x) dμ(y)`

is the gradient of the **displacement potential**

`ψ_μ(x) = ∫ τ(‖x-y‖ + τ) e^{-‖x-y‖/τ} dμ(y)`

(the Matérn-3/2 smoothing of `μ`).  The profile `g(r) = τ(r+τ)e^{-r/τ}` has
`g'(r) = -r e^{-r/τ}`, so `∇_x g(‖x-y‖) = e^{-‖x-y‖/τ}(y-x)`, an identity that
holds *through the diagonal* `x = y` because `g'(0) = 0` (the profile is `C¹`
across the origin even though the norm is not).

The headline consequences are `hasFDerivAt_laplaceDisplacementPotential`
(`∇ψ_μ = D_μ` for every finite `μ`, no moment or atomlessness hypothesis) and
the potential-alignment reformulation of zero drift
`zeroDrift_displacementAligned` (`Z_q • D_p = Z_p • D_q`), which is the ℓ²
analogue of the 1-d companion-alignment gate and the entry point for the
radial endgame (milestone L5).

Design record: `LaplaceHigherDim.md`, §4.1 and §4.8 (L0).
-/

open MeasureTheory Filter Topology
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

/-! ## The Matérn-3/2 radial profile `g(r) = τ(r+τ)e^{-r/τ}` -/

/-- The Matérn-3/2 profile `g(r) = τ(r+τ)e^{-r/τ}`, whose gradient composed with
the norm produces the Laplace displacement field. -/
noncomputable def matern32Profile (τ r : ℝ) : ℝ :=
  τ * (r + τ) * Real.exp (-r / τ)

/-- `g'(r) = -r e^{-r/τ}`; in particular `g'(0) = 0`, which is why the profile
is differentiable through the diagonal. -/
lemma matern32Profile_hasDerivAt {τ : ℝ} (hτ : 0 < τ) (r : ℝ) :
    HasDerivAt (matern32Profile τ) (-r * Real.exp (-r / τ)) r := by
  have hτ0 : τ ≠ 0 := hτ.ne'
  have hlin : HasDerivAt (fun r : ℝ => r + τ) 1 r := (hasDerivAt_id r).add_const τ
  have harg : HasDerivAt (fun r : ℝ => -r / τ) (-1 / τ) r := by
    simpa using ((hasDerivAt_id r).neg.div_const τ)
  have hexp : HasDerivAt (fun r : ℝ => Real.exp (-r / τ))
      (Real.exp (-r / τ) * (-1 / τ)) r := harg.exp
  have hprod : HasDerivAt (matern32Profile τ)
      (τ * 1 * Real.exp (-r / τ) + τ * (r + τ) * (Real.exp (-r / τ) * (-1 / τ))) r :=
    (hlin.const_mul τ).mul hexp
  convert hprod using 1
  field_simp
  ring

/-- The profile is nonnegative for `r ≥ 0`. -/
lemma matern32Profile_nonneg {τ : ℝ} (hτ : 0 < τ) {r : ℝ} (hr : 0 ≤ r) :
    0 ≤ matern32Profile τ r := by
  unfold matern32Profile
  have : 0 ≤ r + τ := by linarith
  positivity

/-- `t·e^{-t/τ} ≤ τ/e` for `t ≥ 0`: the standard `s ≤ e^{s-1}` bound with
`s = t/τ`.  Supplies the *uniform* integrable dominator for the displacement
field. -/
lemma mul_exp_neg_div_le {τ : ℝ} (hτ : 0 < τ) {t : ℝ} (ht : 0 ≤ t) :
    t * Real.exp (-t / τ) ≤ τ * Real.exp (-1) := by
  have hτ0 : τ ≠ 0 := hτ.ne'
  set s := t / τ with hs
  have hs0 : 0 ≤ s := div_nonneg ht hτ.le
  have key : s ≤ Real.exp s * Real.exp (-1) := by
    have h := Real.add_one_le_exp (s - 1)
    rw [sub_add_cancel] at h
    calc s ≤ Real.exp (s - 1) := h
      _ = Real.exp s * Real.exp (-1) := by rw [sub_eq_add_neg, Real.exp_add]
  have hexp : Real.exp s * Real.exp (-s) = 1 := by
    rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
  have hsexp : s * Real.exp (-s) ≤ Real.exp (-1) := by
    have h := mul_le_mul_of_nonneg_right key (Real.exp_pos (-s)).le
    calc s * Real.exp (-s) ≤ Real.exp s * Real.exp (-1) * Real.exp (-s) := h
      _ = Real.exp (-1) * (Real.exp s * Real.exp (-s)) := by ring
      _ = Real.exp (-1) := by rw [hexp, mul_one]
  have ht_eq : t = τ * s := by rw [hs]; field_simp
  have htexp : -t / τ = -s := by rw [hs, neg_div]
  rw [htexp, ht_eq, mul_assoc]
  exact mul_le_mul_of_nonneg_left hsexp hτ.le

/-- Crude uniform bound `g(r) ≤ τ²(1 + e⁻¹)` for `r ≥ 0`, splitting
`g = τ²e^{-r/τ} + τ·(r e^{-r/τ})`.  Supplies integrability of the potential. -/
lemma matern32Profile_le {τ : ℝ} (hτ : 0 < τ) {r : ℝ} (hr : 0 ≤ r) :
    matern32Profile τ r ≤ τ ^ 2 * (1 + Real.exp (-1)) := by
  unfold matern32Profile
  have h1 : Real.exp (-r / τ) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    exact div_nonpos_of_nonpos_of_nonneg (by linarith) hτ.le
  have h2 : r * Real.exp (-r / τ) ≤ τ * Real.exp (-1) := mul_exp_neg_div_le hτ hr
  have b2 : τ ^ 2 * Real.exp (-r / τ) ≤ τ ^ 2 * 1 :=
    mul_le_mul_of_nonneg_left h1 (by positivity)
  calc τ * (r + τ) * Real.exp (-r / τ)
      = τ * (r * Real.exp (-r / τ)) + τ ^ 2 * Real.exp (-r / τ) := by ring
    _ ≤ τ * (τ * Real.exp (-1)) + τ ^ 2 * 1 := by
        gcongr τ * ?_ + ?_
    _ = τ ^ 2 * (1 + Real.exp (-1)) := by ring

/-- The profile is continuous. -/
lemma continuous_matern32Profile (τ : ℝ) : Continuous (matern32Profile τ) := by
  unfold matern32Profile; fun_prop

/-! ## `∇ₓ g(‖x-y‖) = e^{-‖x-y‖/τ}(y-x)`, valid through the diagonal -/

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The Fréchet derivative of `x ↦ g(‖x-y‖)`.  Off the diagonal it is the chain
rule through `‖·‖ = √⟪·,·⟫`; **at** `x = y` the gradient vanishes because
`g'(0) = 0`, so the profile is `C¹` across the origin even though the norm is
not.  The gradient vector is `e^{-‖x-y‖/τ}(y-x)`, expressed as the inner-product
functional `innerSL ℝ (e^{-‖x-y‖/τ}(y-x))`. -/
lemma hasFDerivAt_matern32Profile_norm_sub {τ : ℝ} (hτ : 0 < τ) (y x : E) :
    HasFDerivAt (fun z : E => matern32Profile τ ‖z - y‖)
      (innerSL ℝ (Real.exp (-‖x - y‖ / τ) • (y - x))) x := by
  by_cases hxy : x = y
  · rw [hxy]
    have hzero_clm :
        (innerSL ℝ (Real.exp (-‖y - y‖ / τ) • (y - y)) : E →L[ℝ] ℝ) = 0 := by simp
    rw [hzero_clm, hasFDerivAt_iff_isLittleO]
    simp only [zero_apply, sub_zero, sub_self, norm_zero]
    have hd0 : HasDerivAt (matern32Profile τ) 0 0 := by
      simpa using matern32Profile_hasDerivAt hτ 0
    have hlo : (fun s : ℝ => matern32Profile τ s - matern32Profile τ 0) =o[𝓝 0]
        (fun s : ℝ => s) := by simpa using hd0.isLittleO
    have htend : Tendsto (fun z : E => ‖z - y‖) (𝓝 y) (𝓝 0) := by
      have hc : Continuous (fun z : E => ‖z - y‖) := by fun_prop
      simpa using hc.tendsto y
    exact Asymptotics.isLittleO_norm_right.mp (hlo.comp_tendsto htend)
  · have hne : ‖x - y‖ ≠ 0 := norm_ne_zero_iff.mpr (sub_ne_zero.mpr hxy)
    have hne2 : ‖x - y‖ ^ 2 ≠ 0 := pow_ne_zero 2 hne
    have hsq : HasFDerivAt (fun z : E => ‖z - y‖ ^ 2) (2 • innerSL ℝ (x - y)) x := by
      have h := (hasStrictFDerivAt_norm_sq (x - y)).hasFDerivAt.comp x
        ((hasFDerivAt_id x).sub_const y)
      rwa [ContinuousLinearMap.comp_id] at h
    have hsqrt : HasFDerivAt (fun z : E => Real.sqrt (‖z - y‖ ^ 2))
        ((1 / (2 * Real.sqrt (‖x - y‖ ^ 2))) • (2 • innerSL ℝ (x - y))) x :=
      (Real.hasDerivAt_sqrt hne2).comp_hasFDerivAt x hsq
    have hnorm : HasFDerivAt (fun z : E => ‖z - y‖)
        ((1 / (2 * Real.sqrt (‖x - y‖ ^ 2))) • (2 • innerSL ℝ (x - y))) x := by
      have hfe : (fun z : E => Real.sqrt (‖z - y‖ ^ 2)) = fun z : E => ‖z - y‖ := by
        funext z; rw [Real.sqrt_sq (norm_nonneg _)]
      rwa [hfe] at hsqrt
    have hcomp := (matern32Profile_hasDerivAt hτ ‖x - y‖).comp_hasFDerivAt x hnorm
    have hclm : (innerSL ℝ (Real.exp (-‖x - y‖ / τ) • (y - x)) : E →L[ℝ] ℝ) =
        (-‖x - y‖ * Real.exp (-‖x - y‖ / τ)) •
          ((1 / (2 * Real.sqrt (‖x - y‖ ^ 2))) • (2 • innerSL ℝ (x - y))) := by
      have hs : Real.sqrt (‖x - y‖ ^ 2) = ‖x - y‖ := Real.sqrt_sq (norm_nonneg _)
      ext v
      simp only [hs, smul_apply, add_apply,
        innerSL_apply_apply, real_inner_smul_left, inner_sub_left, two_smul, smul_eq_mul]
      field_simp
      ring
    rw [hclm]
    exact hcomp

/-! ## The displacement field `D` and potential `ψ` -/

omit [InnerProductSpace ℝ E] in
/-- `laplaceKernel τ x y = e^{-‖x-y‖/τ}`, the exponential form of the profile. -/
lemma laplaceKernel_eq_exp (τ : ℝ) (x y : E) :
    laplaceKernel τ x y = Real.exp (-‖x - y‖ / τ) := by
  unfold laplaceKernel; congr 1; ring

section Integrals

variable [MeasurableSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]

/-- The Laplace **displacement field** `D_μ(x) = ∫ e^{-‖x-y‖/τ} • (y-x) dμ`, i.e.
the mean-shift numerator. -/
noncomputable def laplaceDisplacementField (τ : ℝ) (μ : Measure E) (x : E) : E :=
  ∫ y, laplaceKernel τ x y • (y - x) ∂μ

/-- The Laplace **displacement potential** `ψ_μ(x) = ∫ τ(‖x-y‖+τ)e^{-‖x-y‖/τ} dμ`,
the Matérn-3/2 smoothing of `μ`. -/
noncomputable def laplaceDisplacementPotential (τ : ℝ) (μ : Measure E) (x : E) : ℝ :=
  ∫ y, matern32Profile τ ‖x - y‖ ∂μ

omit [CompleteSpace E] in
/-- The displacement-field integrand is bounded by `τ/e`, hence integrable for
every finite measure (no moment hypothesis). -/
lemma integrable_laplaceDisplacementField_integrand {τ : ℝ} (hτ : 0 < τ)
    (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    Integrable (fun y => laplaceKernel τ x y • (y - x)) μ := by
  refine Integrable.of_bound ?_ (τ * Real.exp (-1)) ?_
  · apply Continuous.aestronglyMeasurable; unfold laplaceKernel; fun_prop
  · filter_upwards with y
    rw [norm_smul, laplaceKernel_eq_exp, Real.norm_eq_abs,
      abs_of_pos (Real.exp_pos _), norm_sub_rev, mul_comm]
    exact mul_exp_neg_div_le hτ (norm_nonneg _)

omit [CompleteSpace E] in
/-- The `innerSL`-valued gradient integrand is bounded by `τ/e`, hence integrable
for every finite measure. -/
lemma integrable_matern32_gradient {τ : ℝ} (hτ : 0 < τ)
    (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    Integrable (fun y => innerSL ℝ (Real.exp (-‖x - y‖ / τ) • (y - x))) μ := by
  haveI : SecondCountableTopologyEither E (E →L[ℝ] ℝ) := ⟨Or.inl inferInstance⟩
  refine Integrable.of_bound ?_ (τ * Real.exp (-1)) ?_
  · apply Continuous.aestronglyMeasurable
    exact (innerSL ℝ).continuous.comp (by fun_prop)
  · filter_upwards with y
    rw [innerSL_apply_norm, norm_smul, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _),
      norm_sub_rev, mul_comm]
    exact mul_exp_neg_div_le hτ (norm_nonneg _)

omit [InnerProductSpace ℝ E] [CompleteSpace E] [SecondCountableTopology E] in
/-- The potential integrand is bounded, hence integrable for every finite
measure. -/
lemma integrable_matern32Profile_norm {τ : ℝ} (hτ : 0 < τ)
    (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    Integrable (fun y => matern32Profile τ ‖x - y‖) μ := by
  refine Integrable.of_bound ?_ (τ ^ 2 * (1 + Real.exp (-1))) ?_
  · apply Continuous.aestronglyMeasurable
    exact (continuous_matern32Profile τ).comp (by fun_prop)
  · filter_upwards with y
    rw [Real.norm_eq_abs, abs_of_nonneg (matern32Profile_nonneg hτ (norm_nonneg _))]
    exact matern32Profile_le hτ (norm_nonneg _)

/-- **`∇ψ_μ = D_μ`.**  For every finite measure, the displacement potential is
Fréchet-differentiable with gradient the displacement field — no atomlessness,
moment, or density hypothesis.  Differentiation under the integral is legitimate
because the gradient integrand is *uniformly* bounded by `τ/e`. -/
theorem hasFDerivAt_laplaceDisplacementPotential {τ : ℝ} (hτ : 0 < τ)
    (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    HasFDerivAt (laplaceDisplacementPotential τ μ)
      (innerSL ℝ (laplaceDisplacementField τ μ x)) x := by
  set F : E → E → ℝ := fun z y => matern32Profile τ ‖z - y‖ with hF
  set F' : E → E → (E →L[ℝ] ℝ) :=
    fun z y => innerSL ℝ (Real.exp (-‖z - y‖ / τ) • (y - z)) with hF'
  have hmain : HasFDerivAt (fun z => ∫ y, F z y ∂μ) (∫ y, F' x y ∂μ) x := by
    refine hasFDerivAt_integral_of_dominated_of_fderiv_le (bound := fun _ => τ * Real.exp (-1))
      (F' := F') (s := Set.univ) Filter.univ_mem ?_ ?_ ?_ ?_ (integrable_const _) ?_
    · exact Filter.Eventually.of_forall fun z => by
        apply Continuous.aestronglyMeasurable
        exact (continuous_matern32Profile τ).comp (by fun_prop)
    · exact integrable_matern32Profile_norm hτ μ x
    · simp only [hF']; exact (integrable_matern32_gradient hτ μ x).aestronglyMeasurable
    · refine Filter.Eventually.of_forall fun y z _ => ?_
      rw [hF', innerSL_apply_norm, norm_smul, Real.norm_eq_abs,
        abs_of_pos (Real.exp_pos _), norm_sub_rev, mul_comm]
      exact mul_exp_neg_div_le hτ (norm_nonneg _)
    · exact Filter.Eventually.of_forall fun y z _ =>
        hasFDerivAt_matern32Profile_norm_sub hτ y z
  have hconv : (∫ y, F' x y ∂μ) = innerSL ℝ (laplaceDisplacementField τ μ x) := by
    have hintF' := integrable_matern32_gradient hτ μ x
    have hintD := integrable_laplaceDisplacementField_integrand hτ μ x
    ext v
    rw [ContinuousLinearMap.integral_apply hintF', innerSL_apply_apply,
      show laplaceDisplacementField τ μ x = ∫ y, laplaceKernel τ x y • (y - x) ∂μ from rfl,
      real_inner_comm, ← integral_inner hintD]
    refine integral_congr_ae (Filter.Eventually.of_forall fun y => ?_)
    simp only [innerSL_apply_apply]
    rw [laplaceKernel_eq_exp, real_inner_comm]
  rw [hconv] at hmain
  exact hmain

/-! ## Potential alignment: the ℓ² companion-alignment gate -/

omit [BorelSpace E] [SecondCountableTopology E] in
/-- `meanShift = Z⁻¹ • D`: the paper's mean shift is the normalized displacement
field. -/
lemma meanShift_laplace_eq (τ : ℝ) (μ : Measure E) (x : E) :
    meanShift (laplaceKernel τ) μ x =
      (kernelNormalizer (laplaceKernel τ) μ x)⁻¹ • laplaceDisplacementField τ μ x :=
  rfl

omit [SecondCountableTopology E] in
/-- **Potential alignment (PA).**  Pointwise zero Laplace mean-shift drift is
equivalent to `Z_q • D_p = Z_p • D_q`, the ℓ² analogue of the 1-d
companion-alignment gate (`laplaceCompanionAlignmentDefect ≡ 0`) and the entry
point for the radial endgame.  Since `D = ∇ψ`
(`hasFDerivAt_laplaceDisplacementPotential`), this reads `Z_q ∇ψ_p = Z_p ∇ψ_q`. -/
theorem zeroDrift_displacementAligned {τ : ℝ} (hτ : 0 < τ) (P Q : Measure E)
    [IsProbabilityMeasure P] [IsProbabilityMeasure Q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) P Q) (x : E) :
    kernelNormalizer (laplaceKernel τ) Q x • laplaceDisplacementField τ P x =
      kernelNormalizer (laplaceKernel τ) P x • laplaceDisplacementField τ Q x := by
  have hzP := laplaceKernelNormalizer_pos P τ hτ x
  have hzQ := laplaceKernelNormalizer_pos Q τ hτ x
  have hms : meanShift (laplaceKernel τ) P x = meanShift (laplaceKernel τ) Q x := by
    have h := hzero x
    simp only [meanShiftDrift, sub_eq_zero] at h
    exact h
  rw [meanShift_laplace_eq, meanShift_laplace_eq] at hms
  set ZP := kernelNormalizer (laplaceKernel τ) P x with hZP
  set ZQ := kernelNormalizer (laplaceKernel τ) Q x with hZQ
  set DP := laplaceDisplacementField τ P x with hDP
  set DQ := laplaceDisplacementField τ Q x with hDQ
  calc ZQ • DP = ZQ • (ZP • (ZP⁻¹ • DP)) := by rw [smul_inv_smul₀ hzP.ne']
    _ = ZQ • (ZP • (ZQ⁻¹ • DQ)) := by rw [hms]
    _ = ZP • (ZQ • (ZQ⁻¹ • DQ)) := smul_comm ZQ ZP _
    _ = ZP • DQ := by rw [smul_inv_smul₀ hzQ.ne']

end Integrals

end DriftingIdentifiability
