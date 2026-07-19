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
open scoped ENNReal

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

/-! ## The symmetric fission pair -/

/-- The symmetric two-particle state `½δ_{c+u} + ½δ_{c−u}`. -/
noncomputable def laplaceFissionPair (c u : ℝ) : Measure ℝ :=
  (2⁻¹ : ℝ≥0∞) • (Measure.dirac (c + u) + Measure.dirac (c - u))

instance laplaceFissionPair_isProbabilityMeasure (c u : ℝ) :
    IsProbabilityMeasure (laplaceFissionPair c u) := by
  constructor
  rw [laplaceFissionPair, Measure.smul_apply, Measure.add_apply,
    Measure.dirac_apply_of_mem (mem_univ _),
    Measure.dirac_apply_of_mem (mem_univ _), smul_eq_mul]
  rw [one_add_one_eq_two]
  exact ENNReal.inv_mul_cancel two_ne_zero (by norm_num)

/-- Integration against the fission pair is the symmetric average. -/
lemma integral_laplaceFissionPair (c u : ℝ) {f : ℝ → ℝ}
    (hf : Integrable f (Measure.dirac (c + u)))
    (hg : Integrable f (Measure.dirac (c - u))) :
    ∫ y, f y ∂(laplaceFissionPair c u) =
      (f (c + u) + f (c - u)) / 2 := by
  rw [laplaceFissionPair, integral_smul_measure, integral_add_measure hf hg,
    integral_dirac, integral_dirac]
  rw [smul_eq_mul, ENNReal.toReal_inv]
  norm_num
  ring

lemma laplaceKernel_self (τ : ℝ) (x : ℝ) : laplaceKernel τ x x = 1 := by
  unfold laplaceKernel
  rw [sub_self, norm_zero, mul_zero, Real.exp_zero]

/-- The mean shift of the fission pair at its right particle: exact
repulsion `−2u·k/(1+k)` with `k` the inter-particle kernel value. -/
theorem meanShift_laplaceFissionPair_right (τ : ℝ) (hτ : ValidBandwidth τ)
    (c u : ℝ) :
    meanShift (laplaceKernel τ) (laplaceFissionPair c u) (c + u) =
      -(2 * u * laplaceKernel τ (c + u) (c - u)) /
        (1 + laplaceKernel τ (c + u) (c - u)) := by
  have hτ0 : (0 : ℝ) < τ := hτ
  set k2 := laplaceKernel τ (c + u) (c - u) with hk2
  have hk2pos : 0 < k2 := Real.exp_pos _
  have hZ : kernelNormalizer (laplaceKernel τ) (laplaceFissionPair c u)
      (c + u) = (1 + k2) / 2 := by
    unfold kernelNormalizer
    rw [integral_laplaceFissionPair c u
      (integrable_dirac (by finiteness)) (integrable_dirac (by finiteness))]
    rw [laplaceKernel_self]
  have hD : (∫ y, laplaceKernel τ (c + u) y • (y - (c + u))
      ∂(laplaceFissionPair c u)) = -(u * k2) := by
    rw [integral_laplaceFissionPair c u
      (integrable_dirac (by finiteness)) (integrable_dirac (by finiteness))]
    rw [laplaceKernel_self]
    have h1 : (c + u) - (c + u) = 0 := by ring
    have h2 : (c - u) - (c + u) = -(2 * u) := by ring
    rw [h1, h2]
    simp only [smul_eq_mul, hk2]
    ring
  unfold meanShift
  rw [hZ, hD, smul_eq_mul]
  have hden : (1 + k2) / 2 ≠ 0 := ne_of_gt (by linarith)
  field_simp

/-- Mirror form at the left particle. -/
theorem meanShift_laplaceFissionPair_left (τ : ℝ) (hτ : ValidBandwidth τ)
    (c u : ℝ) :
    meanShift (laplaceKernel τ) (laplaceFissionPair c u) (c - u) =
      (2 * u * laplaceKernel τ (c + u) (c - u)) /
        (1 + laplaceKernel τ (c + u) (c - u)) := by
  have hτ0 : (0 : ℝ) < τ := hτ
  set k2 := laplaceKernel τ (c + u) (c - u) with hk2
  have hk2sym : laplaceKernel τ (c - u) (c + u) = k2 := by
    rw [hk2]
    unfold laplaceKernel
    rw [norm_sub_rev]
  have hZ : kernelNormalizer (laplaceKernel τ) (laplaceFissionPair c u)
      (c - u) = (1 + k2) / 2 := by
    unfold kernelNormalizer
    rw [integral_laplaceFissionPair c u
      (integrable_dirac (by finiteness)) (integrable_dirac (by finiteness))]
    rw [laplaceKernel_self, hk2sym]
    ring
  have hD : (∫ y, laplaceKernel τ (c - u) y • (y - (c - u))
      ∂(laplaceFissionPair c u)) = u * k2 := by
    rw [integral_laplaceFissionPair c u
      (integrable_dirac (by finiteness)) (integrable_dirac (by finiteness))]
    rw [laplaceKernel_self, hk2sym]
    have h1 : (c + u) - (c - u) = 2 * u := by ring
    have h2 : (c - u) - (c - u) = 0 := by ring
    rw [h1, h2]
    simp only [smul_eq_mul]
    ring
  unfold meanShift
  rw [hZ, hD, smul_eq_mul]
  have hk2pos : 0 < k2 := Real.exp_pos _
  have hden : (1 + k2) / 2 ≠ 0 := ne_of_gt (by linarith)
  field_simp

/-- On the line, the paper's mean shift is the certified ratio. -/
lemma meanShift_eq_laplaceMeanShiftRatio (τ : ℝ) (p : Measure ℝ) (x : ℝ) :
    meanShift (laplaceKernel τ) p x = laplaceMeanShiftRatio τ p x := by
  unfold meanShift laplaceMeanShiftRatio laplaceWeightedDisplacement
  rw [smul_eq_mul, div_eq_inv_mul]

/-! ## The separation-growth theorem -/

/-- **Fission instability, dynamical form.**  At a collapse point that does
not carry all of `p`'s mass, the symmetric-pair separation velocity of the
actual paper drift field is strictly positive for every sufficiently small
separation: point collapse is escapable by fission. -/
theorem laplaceFission_separation_grows (τ : ℝ) (hτ : ValidBandwidth τ)
    (p : Measure ℝ) [IsProbabilityMeasure p] {c : ℝ}
    (hroot : laplaceMeanShiftRatio τ p c = 0) (hmass : p {c} < 1) :
    ∃ ε : ℝ, 0 < ε ∧ ∀ u ∈ Ioo (0 : ℝ) ε,
      0 < (meanShiftDrift (laplaceKernel τ) p (laplaceFissionPair c u)
            (c + u) -
          meanShiftDrift (laplaceKernel τ) p (laplaceFissionPair c u)
            (c - u)) / 2 := by
  have hτ0 : (0 : ℝ) < τ := hτ
  set m := laplaceMeanShiftRatio τ p with hm
  set m' := laplaceMeanShiftRatioDeriv τ p c with hm'
  have hderiv : HasDerivAt m m' c :=
    hasDerivAt_laplaceMeanShiftRatio_of_root τ hτ p hroot
  have hσ : 0 < m' + 1 := laplaceFissionIndex_pos τ hτ p hroot hmass
  -- the separation-velocity function and its slope form
  set k2 : ℝ → ℝ := fun u => laplaceKernel τ (c + u) (c - u) with hk2
  set g : ℝ → ℝ := fun u =>
    (m (c + u) - m (c - u)) / 2 + 2 * u * k2 u / (1 + k2 u) with hg
  -- slope convergence along both arms
  have hslope := hasDerivAt_iff_tendsto_slope.mp hderiv
  have hplus : Tendsto (fun u : ℝ => c + u) (nhdsWithin 0 (Ioi 0))
      (nhdsWithin c {c}ᶜ) := by
    apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within
    · have hc : Continuous fun u : ℝ => c + u := by fun_prop
      have := hc.tendsto (0 : ℝ)
      simpa using this.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with u hu
      have : (0 : ℝ) < u := hu
      simp only [mem_compl_iff, mem_singleton_iff]
      intro hbad
      have : u = 0 := by linarith [hbad ▸ (by ring : c + u - u = c)]
      nlinarith
  have hminus : Tendsto (fun u : ℝ => c - u) (nhdsWithin 0 (Ioi 0))
      (nhdsWithin c {c}ᶜ) := by
    apply tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within
    · have hc : Continuous fun u : ℝ => c - u := by fun_prop
      have := hc.tendsto (0 : ℝ)
      simpa using this.mono_left nhdsWithin_le_nhds
    · filter_upwards [self_mem_nhdsWithin] with u hu
      have hu0 : (0 : ℝ) < u := hu
      simp only [mem_compl_iff, mem_singleton_iff]
      intro hbad
      nlinarith [hbad ▸ (by ring : c - (c - u) = u)]
  have hTplus := hslope.comp hplus
  have hTminus := hslope.comp hminus
  -- the kernel factor tends to 1
  have hkcont : Tendsto (fun u : ℝ => 2 * k2 u / (1 + k2 u))
      (nhdsWithin 0 (Ioi 0)) (nhds 1) := by
    have hk20 : k2 0 = 1 := by
      rw [hk2]
      simp only [add_zero, sub_zero]
      exact laplaceKernel_self τ c
    have hcont : Continuous k2 := by
      rw [hk2]
      unfold laplaceKernel
      fun_prop
    have h1 : Tendsto (fun u : ℝ => 2 * k2 u / (1 + k2 u)) (nhds 0)
        (nhds (2 * k2 0 / (1 + k2 0))) := by
      apply Tendsto.div
      · exact (hcont.const_mul 2).tendsto 0
      · exact (hcont.const_add 1).tendsto 0
      · rw [hk20]; norm_num
    rw [hk20] at h1
    norm_num at h1
    exact h1.mono_left nhdsWithin_le_nhds
  -- the combined limit of g u / u
  have hcomb : Tendsto (fun u : ℝ =>
      (slope m c (c + u) + slope m c (c - u)) / 2 + 2 * k2 u / (1 + k2 u))
      (nhdsWithin 0 (Ioi 0)) (nhds (m' + 1)) := by
    have := ((hTplus.add hTminus).div_const 2).add hkcont
    rw [show (m' + m') / 2 + 1 = m' + 1 by ring] at this
    exact this
  -- identify with g u / u for u > 0
  have hev : ∀ᶠ u in nhdsWithin (0 : ℝ) (Ioi 0),
      (slope m c (c + u) + slope m c (c - u)) / 2 +
        2 * k2 u / (1 + k2 u) = g u / u := by
    filter_upwards [self_mem_nhdsWithin] with u hu
    have hu0 : (0 : ℝ) < u := hu
    have hk2pos : 0 < k2 u := Real.exp_pos _
    have hs1 : slope m c (c + u) = m (c + u) / u := by
      rw [slope_def_field]
      rw [hroot]
      congr 1
      · rw [sub_zero]
      · ring
    have hs2 : slope m c (c - u) = -(m (c - u) / u) := by
      rw [slope_def_field, hroot]
      rw [sub_zero]
      rw [show c - u - c = -u by ring]
      rw [div_neg]
    rw [hs1, hs2, hg]
    field_simp
    ring
  have htend : Tendsto (fun u : ℝ => g u / u)
      (nhdsWithin 0 (Ioi 0)) (nhds (m' + 1)) := hcomb.congr' hev
  -- positivity of g for small u
  have hpos : ∀ᶠ u in nhdsWithin (0 : ℝ) (Ioi 0), 0 < g u := by
    filter_upwards [htend.eventually (eventually_gt_nhds hσ),
      self_mem_nhdsWithin] with u hquot hu
    have hu0 : (0 : ℝ) < u := hu
    have : 0 < g u / u := hquot
    calc (0 : ℝ) < g u / u * u := mul_pos this hu0
      _ = g u := div_mul_cancel₀ _ hu0.ne'
  -- extract a uniform interval
  rw [eventually_nhdsWithin_iff, Metric.eventually_nhds_iff] at hpos
  obtain ⟨ε, hεpos, hε⟩ := hpos
  refine ⟨ε, hεpos, fun u hu => ?_⟩
  have hgu : 0 < g u := by
    apply hε
    · rw [Real.dist_eq, sub_zero, abs_of_pos hu.1]
      exact hu.2
    · exact hu.1
  -- rewrite the drift difference through the closed forms
  have hright := meanShift_laplaceFissionPair_right τ hτ c u
  have hleft := meanShift_laplaceFissionPair_left τ hτ c u
  have hk2pos : 0 < laplaceKernel τ (c + u) (c - u) := Real.exp_pos _
  unfold meanShiftDrift
  rw [hright, hleft, meanShift_eq_laplaceMeanShiftRatio,
    meanShift_eq_laplaceMeanShiftRatio]
  have hval : (m (c + u) -
      -(2 * u * laplaceKernel τ (c + u) (c - u)) /
        (1 + laplaceKernel τ (c + u) (c - u)) -
      (m (c - u) -
        2 * u * laplaceKernel τ (c + u) (c - u) /
          (1 + laplaceKernel τ (c + u) (c - u)))) / 2 = g u := by
    rw [hg]
    simp only [hk2]
    have hden : (0 : ℝ) < 1 + laplaceKernel τ (c + u) (c - u) := by
      linarith
    field_simp
    ring
  rw [hval]
  exact hgu

end DriftingIdentifiability
