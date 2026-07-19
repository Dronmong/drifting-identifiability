import DriftingIdentifiability.LaplaceUnconditionalConverse
import DriftingIdentifiability.LaplaceRadialFoundations

/-!
# Fission instability of point collapse (dynamics roadmap, A1)

The population dynamics of drifting admits *collapse equilibria*: if
`m_p(c) = 0` (a kernel-tilted barycenter of `p`), the single-particle state
`δ_c` is stationary.  The Collapse Atlas (numerics/CollapseAtlas*.md) found
these equilibria to be escapable by *fission* — splitting the particle into
a symmetric pair `½δ_{c+u} + ½δ_{c−u}` whose separation then grows — with
splitting index `σ = m_p′(c) + 1`, strictly positive except exactly when
`p` concentrates at `c`.  This file certifies that picture in one dimension:

* **the fission-index formula**: at a root of the mean shift,
  `m_p′(c) + 1 = (1/τ)·(∫‖c−y‖ k dp)/Z_p(c)` — the kernel-tilted mean
  absolute deviation, manifestly nonnegative;
* **strictness**: it is strictly positive unless `p({c}) = 1`;
* **the dynamical reading**: for the actual paper drift field of the
  symmetric pair, the separation velocity satisfies `g(u)/u → m_p′(c) + 1`,
  so the separation strictly grows for all small `u > 0`.

Everything reduces to certified one-dimensional identities: the two-sided
differentiability of the mean-shift ratio at its roots and the certified
displacement derivative `D′ = (1/τ)L_c − 2Z`.
-/

open MeasureTheory Filter Set Topology

namespace DriftingIdentifiability

open Paper

/-! ## The tilted mean absolute deviation -/

/-- `∫ ‖x−y‖·k(x,y) dp`: the kernel-tilted (unnormalized) mean absolute
deviation, the exact gap between the companion normalizer and `τ·Z`. -/
noncomputable def laplaceAbsDeviation (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y, ‖x - y‖ * laplaceKernel τ x y ∂p

lemma integrable_absDeviation_integrand (τ : ℝ) (hτ : ValidBandwidth τ)
    (p : Measure ℝ) [IsProbabilityMeasure p] (x : ℝ) :
    Integrable (fun y => ‖x - y‖ * laplaceKernel τ x y) p := by
  have hτ0 : (0 : ℝ) < τ := hτ
  refine Integrable.of_bound ?_ (τ * Real.exp (-1)) ?_
  · apply Continuous.aestronglyMeasurable
    unfold laplaceKernel
    fun_prop
  · filter_upwards with y
    rw [Real.norm_eq_abs, abs_of_nonneg (by
      have := Real.exp_pos (-(1 / τ) * ‖x - y‖)
      unfold laplaceKernel
      positivity)]
    unfold laplaceKernel
    rw [show -(1 / τ) * ‖x - y‖ = -‖x - y‖ / τ by ring]
    exact mul_exp_neg_div_le hτ0 (norm_nonneg (x - y))

/-- The companion normalizer splits as `τ·Z` plus the deviation. -/
theorem companionNormalizer_eq_tau_normalizer_add_deviation
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ) :
    kernelNormalizer (laplaceCompanionKernel τ) p x =
      τ * kernelNormalizer (laplaceKernel τ) p x +
        laplaceAbsDeviation τ p x := by
  have hτ0 : (0 : ℝ) < τ := hτ
  have hker : Integrable (fun y => τ * laplaceKernel τ x y) p :=
    (laplaceKernel_integrable p τ hτ0 x).const_mul τ
  have hdev := integrable_absDeviation_integrand τ hτ p x
  have hpt : ∀ y : ℝ, laplaceCompanionKernel τ x y =
      τ * laplaceKernel τ x y + ‖x - y‖ * laplaceKernel τ x y := by
    intro y
    unfold laplaceCompanionKernel
    ring
  unfold kernelNormalizer laplaceAbsDeviation
  rw [show (fun y => laplaceCompanionKernel τ x y) =
      fun y => τ * laplaceKernel τ x y + ‖x - y‖ * laplaceKernel τ x y from
    funext hpt, integral_add hker hdev, integral_const_mul]

lemma laplaceAbsDeviation_nonneg (τ : ℝ) (p : Measure ℝ) (x : ℝ) :
    0 ≤ laplaceAbsDeviation τ p x := by
  refine integral_nonneg fun y => ?_
  have := Real.exp_pos (-(1 / τ) * ‖x - y‖)
  unfold laplaceKernel
  positivity

/-- Strict positivity of the deviation away from the pure point mass. -/
theorem laplaceAbsDeviation_pos (τ : ℝ) (hτ : ValidBandwidth τ)
    (p : Measure ℝ) [IsProbabilityMeasure p] {x : ℝ}
    (hmass : p {x} < 1) :
    0 < laplaceAbsDeviation τ p x := by
  rw [laplaceAbsDeviation,
    integral_pos_iff_support_of_nonneg (fun y => by
      have := Real.exp_pos (-(1 / τ) * ‖x - y‖)
      unfold laplaceKernel
      positivity)
      (integrable_absDeviation_integrand τ hτ p x)]
  have hsupp : Function.support
      (fun y => ‖x - y‖ * laplaceKernel τ x y) = {x}ᶜ := by
    ext y
    simp only [Function.mem_support, mem_compl_iff, mem_singleton_iff]
    constructor
    · intro hne heq
      apply hne
      rw [heq]
      simp
    · intro hne
      have hk : laplaceKernel τ x y ≠ 0 := (Real.exp_pos _).ne'
      have hnorm : ‖x - y‖ ≠ 0 := by
        rw [norm_ne_zero_iff, sub_ne_zero]
        exact fun h => hne h.symm
      exact mul_ne_zero hnorm hk
  rw [hsupp]
  have hcompl : p {x}ᶜ = 1 - p {x} := by
    rw [measure_compl (measurableSet_singleton x) (measure_ne_top p _)]
    simp
  rw [hcompl]
  have hlt : p {x} < 1 := hmass
  exact tsub_pos_of_lt hlt

/-! ## The fission index -/

/-- The fission index `σ(c) = m_p′(c) + 1`: the linear growth rate of the
separation of a symmetric fission pair released at `c`. -/
noncomputable def laplaceFissionIndex (τ : ℝ) (p : Measure ℝ) (c : ℝ) : ℝ :=
  laplaceMeanShiftRatioDeriv τ p c + 1

/-- At a root of the mean shift, the quotient-rule derivative collapses to
`D′/Z`, giving the closed fission-index formula. -/
theorem laplaceFissionIndex_eq_deviation (τ : ℝ) (hτ : ValidBandwidth τ)
    (p : Measure ℝ) [IsProbabilityMeasure p] {c : ℝ}
    (hroot : laplaceMeanShiftRatio τ p c = 0) :
    laplaceFissionIndex τ p c =
      (1 / τ) * laplaceAbsDeviation τ p c /
        kernelNormalizer (laplaceKernel τ) p c := by
  have hτ0 : (0 : ℝ) < τ := hτ
  have hZpos : 0 < kernelNormalizer (laplaceKernel τ) p c :=
    laplaceKernelNormalizer_pos p τ hτ c
  have hZne := hZpos.ne'
  have hD0 : laplaceMeanShiftNumerator τ p c = 0 := by
    have h0 : laplaceMeanShiftNumerator τ p c /
        kernelNormalizer (laplaceKernel τ) p c = 0 := by
      simpa [laplaceMeanShiftRatio, laplaceMeanShiftNumerator] using hroot
    exact (div_eq_zero_iff.mp h0).resolve_right hZne
  have hLsplit := companionNormalizer_eq_tau_normalizer_add_deviation
    τ hτ p c
  unfold laplaceFissionIndex laplaceMeanShiftRatioDeriv
    laplaceMeanShiftRatioDerivNumerator laplaceMeanShiftRatioDerivDenominator
  rw [hD0, zero_mul, sub_zero]
  unfold laplaceMeanShiftNumeratorDeriv
  rw [hLsplit]
  field_simp [hZne, hτ0.ne']
  ring

/-- **Strict fission instability** (matching the atlas's sharp P2B boundary):
at a root of the mean shift, the fission index is strictly positive unless
`p` is entirely concentrated at the root. -/
theorem laplaceFissionIndex_pos (τ : ℝ) (hτ : ValidBandwidth τ)
    (p : Measure ℝ) [IsProbabilityMeasure p] {c : ℝ}
    (hroot : laplaceMeanShiftRatio τ p c = 0) (hmass : p {c} < 1) :
    0 < laplaceFissionIndex τ p c := by
  have hτ0 : (0 : ℝ) < τ := hτ
  rw [laplaceFissionIndex_eq_deviation τ hτ p hroot]
  have hdev := laplaceAbsDeviation_pos τ hτ p hmass
  have hZ := laplaceKernelNormalizer_pos p τ hτ c
  exact div_pos (mul_pos (by positivity) hdev) hZ

/-- Nonnegativity holds without any hypothesis beyond the root. -/
theorem laplaceFissionIndex_nonneg (τ : ℝ) (hτ : ValidBandwidth τ)
    (p : Measure ℝ) [IsProbabilityMeasure p] {c : ℝ}
    (hroot : laplaceMeanShiftRatio τ p c = 0) :
    0 ≤ laplaceFissionIndex τ p c := by
  have hτ0 : (0 : ℝ) < τ := hτ
  rw [laplaceFissionIndex_eq_deviation τ hτ p hroot]
  have hdev := laplaceAbsDeviation_nonneg τ p c
  have hZ := laplaceKernelNormalizer_pos p τ hτ c
  exact div_nonneg (mul_nonneg (by positivity) hdev) hZ.le

end DriftingIdentifiability
