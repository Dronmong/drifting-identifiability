import DriftingIdentifiability.LaplaceInjectivity

/-!
# The 1-d Wronskian program: classical ODE and the alignment reduction

Stage 3 of the attack on the open Laplace-kernel arbitrary-target converse
(`LaplaceArbitraryConverse.md`).  Two results, both axiom-free and valid for
ARBITRARY probability measures on `ℝ` with no moment hypotheses.

**The elliptic structure is classical.**  The displacement integrand
`x ↦ kτ(x,y)·(y-x)` is differentiable EVERYWHERE — the `sgn` singularity of
`∂ₓkτ` is killed by the vanishing factor — with derivative
`(‖x-y‖/τ - 1)·kτ(x,y)`, uniformly bounded.  Differentiating under the
integral, the mean-shift numerator `D_p` is `C¹` with

`D_p′ = (1/τ)·L_p - 2·Z_p`

(`hasDerivAt_laplaceDisplacementIntegral`), where `L_p` is the companion
normalizer and `Z_p` the Laplace normalizer.  Since `L_p′ = (1/τ)·D_p`
(Stage 1), this is the pointwise second-order ODE `τ²L_p″ = L_p - 2τZ_p`:
the plan's distributional identity `(τ²∂²-1)Z_p = -2τ·p` is bypassed
entirely, and the "two solutions of one operator" Wronskian structure of the
1-d program lives at the level of classical derivatives of smoothings.

**The alignment reduction.**  Define the companion alignment
`K(x) = L_p(x)·Z_q(x) - L_q(x)·Z_p(x)`.  The headline
`laplaceZeroDrift_imp_eq_of_companionAligned` proves: zero raw drift together
with `K ≡ 0` forces `p = q`.  Chain: zero drift gives the cross-displacement
identity (Stage 1); with alignment and `Z_p > 0` it kills the Wronskian
`L_p′L_q - L_pL_q′`, so `L_p/L_q` has zero derivative and `L_p = c·L_q`;
alignment converts this to `Z_p = c·Z_q`; smoothing injectivity (Stage 2)
applied to `p` and `c•q` gives `p = c•q`, and total mass forces `c = 1`.
**The open 1-d converse is thereby reduced to the single scalar identity
`K ≡ 0`.**
-/

open MeasureTheory Set Filter Topology
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

/-! ## Scalar helpers -/

private lemma mul_exp_neg_le' {τ : ℝ} (hτ : 0 < τ) {s : ℝ} (hs : 0 ≤ s) :
    s * Real.exp (-(1 / τ) * s) ≤ τ := by
  have h1 : s / τ + 1 ≤ Real.exp (s / τ) := Real.add_one_le_exp (s / τ)
  have h2 := mul_le_mul_of_nonneg_left h1 hτ.le
  have h3 : τ * (s / τ + 1) = s + τ := by field_simp
  have hexp : Real.exp (-(1 / τ) * s) = (Real.exp (s / τ))⁻¹ := by
    rw [← Real.exp_neg]
    congr 1
    field_simp
  rw [hexp, mul_inv_le_iff₀ (Real.exp_pos _)]
  nlinarith [Real.exp_pos (s / τ), hs]

private lemma laplaceKernel_le_one' {τ : ℝ} (hτ : 0 < τ) (x y : ℝ) :
    laplaceKernel τ x y ≤ 1 := by
  unfold laplaceKernel
  rw [Real.exp_le_one_iff]
  have : (0 : ℝ) ≤ 1 / τ := by positivity
  nlinarith [norm_nonneg (x - y)]

private lemma one_sub_exp_neg_le (u : ℝ) :
    1 - Real.exp (-u) ≤ u := by
  have := Real.add_one_le_exp (-u)
  linarith

/-! ## Positivity and integrability of the companion normalizer -/

lemma laplaceCompanionKernel_integrable'
    (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    Integrable (fun y => laplaceCompanionKernel τ x y) p := by
  refine Integrable.of_bound ?_ (2 * τ) ?_
  · apply Continuous.aestronglyMeasurable
    unfold laplaceCompanionKernel laplaceKernel
    fun_prop
  · filter_upwards with y
    have hk0 : 0 < laplaceKernel τ x y := Real.exp_pos _
    have hk1 : laplaceKernel τ x y ≤ 1 := laplaceKernel_le_one' hτ x y
    have hsk : ‖x - y‖ * laplaceKernel τ x y ≤ τ := by
      unfold laplaceKernel
      exact mul_exp_neg_le' hτ (norm_nonneg (x - y))
    rw [Real.norm_eq_abs]
    unfold laplaceCompanionKernel
    rw [abs_of_nonneg (by positivity)]
    nlinarith [norm_nonneg (x - y)]

/-- The companion normalizer is strictly positive: `L_p ≥ τ·Z_p > 0`. -/
theorem laplaceCompanionNormalizer_pos
    (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsProbabilityMeasure p] (x : ℝ) :
    0 < kernelNormalizer (laplaceCompanionKernel τ) p x := by
  have hZ := laplaceKernelNormalizer_pos p τ hτ x
  have hτZ : 0 < τ * kernelNormalizer (laplaceKernel τ) p x := mul_pos hτ hZ
  refine lt_of_lt_of_le hτZ ?_
  unfold kernelNormalizer
  rw [← integral_const_mul]
  apply integral_mono ((laplaceKernel_integrable p τ hτ x).const_mul τ)
    (laplaceCompanionKernel_integrable' τ hτ p x)
  intro y
  unfold laplaceCompanionKernel
  have hk0 : 0 ≤ laplaceKernel τ x y := (Real.exp_pos _).le
  nlinarith [norm_nonneg (x - y)]

/-! ## The 1-d derivative of the companion normalizer -/

/-- One-dimensional form of the Stage-1 companion score identity:
`L_p′(x) = (1/τ)·D_p(x)`. -/
theorem hasDerivAt_laplaceCompanionNormalizer
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ) :
    HasDerivAt (kernelNormalizer (laplaceCompanionKernel τ) p)
      ((1 / τ) * ∫ y, laplaceWeightedDisplacement τ x y ∂p) x := by
  have h := (hasFDerivAt_laplaceCompanionNormalizer τ hτ p x).hasDerivAt
  simpa [innerSL_apply_apply, RCLike.inner_apply] using h

/-! ## The displacement integrand is `C¹` everywhere -/

private lemma hasDerivAt_laplaceDisplacementKernel
    (τ : ℝ) (hτ : 0 < τ) (y x : ℝ) :
    HasDerivAt (fun z : ℝ => laplaceWeightedDisplacement τ z y)
      ((‖x - y‖ / τ - 1) * laplaceKernel τ x y) x := by
  rcases lt_trichotomy x y with hxy | rfl | hxy
  · -- below the sample: `‖z - y‖ = y - z` near `x`
    have hev : (fun z : ℝ => laplaceWeightedDisplacement τ z y) =ᶠ[𝓝 x]
        fun z : ℝ => Real.exp ((1 / τ) * (z - y)) * (y - z) := by
      filter_upwards [Iio_mem_nhds hxy] with z hz
      unfold laplaceWeightedDisplacement laplaceKernel
      rw [smul_eq_mul, Real.norm_eq_abs, abs_of_neg (sub_neg.mpr hz)]
      congr 1
      congr 1
      ring
    have h1 : HasDerivAt (fun z : ℝ => (1 / τ) * (z - y)) (1 / τ) x := by
      simpa using ((hasDerivAt_id x).sub_const y).const_mul (1 / τ)
    have h2 := h1.exp
    have h3 : HasDerivAt (fun z : ℝ => y - z) (-1 : ℝ) x := by
      simpa using (hasDerivAt_id x).const_sub y
    have hmul : HasDerivAt
        (fun z : ℝ => Real.exp ((1 / τ) * (z - y)) * (y - z))
        (Real.exp ((1 / τ) * (x - y)) * (1 / τ) * (y - x) +
          Real.exp ((1 / τ) * (x - y)) * (-1 : ℝ)) x := h2.mul h3
    have hf := hmul.congr_of_eventuallyEq hev
    have hval : Real.exp ((1 / τ) * (x - y)) * (1 / τ) * (y - x) +
        Real.exp ((1 / τ) * (x - y)) * (-1 : ℝ) =
        (‖x - y‖ / τ - 1) * laplaceKernel τ x y := by
      unfold laplaceKernel
      rw [Real.norm_eq_abs, abs_of_neg (sub_neg.mpr hxy),
        show -(1 / τ) * -(x - y) = (1 / τ) * (x - y) by ring]
      ring
    exact hval ▸ hf
  · -- at the sample: quadratic remainder, derivative `-1`
    rw [show (‖x - x‖ / τ - 1) * laplaceKernel τ x x = (-1 : ℝ) by
      unfold laplaceKernel
      simp]
    rw [hasDerivAt_iff_isLittleO_nhds_zero]
    rw [Asymptotics.isLittleO_iff]
    intro c hc
    filter_upwards [Metric.ball_mem_nhds (0 : ℝ) (mul_pos hc hτ)] with h hh
    rw [mem_ball_zero_iff] at hh
    have hf0 : laplaceWeightedDisplacement τ x x = 0 := by
      unfold laplaceWeightedDisplacement
      simp
    have hfh : laplaceWeightedDisplacement τ (x + h) x =
        Real.exp (-(1 / τ) * ‖h‖) * (-h) := by
      unfold laplaceWeightedDisplacement laplaceKernel
      rw [smul_eq_mul, add_sub_cancel_left]
      ring_nf
    rw [hf0, hfh]
    have hexp0 : Real.exp (-(1 / τ) * ‖h‖) ≤ 1 := by
      rw [Real.exp_le_one_iff]
      have : (0 : ℝ) ≤ 1 / τ := by positivity
      nlinarith [norm_nonneg h]
    have hexp1 : 1 - Real.exp (-(1 / τ) * ‖h‖) ≤ ‖h‖ / τ := by
      have := one_sub_exp_neg_le ((1 / τ) * ‖h‖)
      calc 1 - Real.exp (-(1 / τ) * ‖h‖)
          = 1 - Real.exp (-((1 / τ) * ‖h‖)) := by ring_nf
        _ ≤ (1 / τ) * ‖h‖ := this
        _ = ‖h‖ / τ := by ring
    have hrem : Real.exp (-(1 / τ) * ‖h‖) * (-h) - 0 - h • (-1 : ℝ) =
        h * (1 - Real.exp (-(1 / τ) * ‖h‖)) := by
      rw [smul_eq_mul]
      ring
    rw [hrem, Real.norm_eq_abs, abs_mul]
    have habs : |1 - Real.exp (-(1 / τ) * ‖h‖)| =
        1 - Real.exp (-(1 / τ) * ‖h‖) := abs_of_nonneg (by linarith)
    rw [habs]
    have hh' : |h| < c * τ := by rwa [Real.norm_eq_abs] at hh
    have habs_norm : |h| = ‖h‖ := (Real.norm_eq_abs h).symm
    calc |h| * (1 - Real.exp (-(1 / τ) * ‖h‖))
        ≤ |h| * (‖h‖ / τ) := by
          apply mul_le_mul_of_nonneg_left hexp1 (abs_nonneg h)
      _ ≤ |h| * c := by
          apply mul_le_mul_of_nonneg_left _ (abs_nonneg h)
          rw [div_le_iff₀ hτ, ← habs_norm]
          nlinarith [abs_nonneg h]
      _ = c * ‖h‖ := by rw [habs_norm]; ring
  · -- above the sample: `‖z - y‖ = z - y` near `x`
    have hev : (fun z : ℝ => laplaceWeightedDisplacement τ z y) =ᶠ[𝓝 x]
        fun z : ℝ => Real.exp (-(1 / τ) * (z - y)) * (y - z) := by
      filter_upwards [Ioi_mem_nhds hxy] with z hz
      unfold laplaceWeightedDisplacement laplaceKernel
      rw [smul_eq_mul, Real.norm_eq_abs, abs_of_pos (sub_pos.mpr hz)]
    have h1 : HasDerivAt (fun z : ℝ => -(1 / τ) * (z - y)) (-(1 / τ)) x := by
      simpa using ((hasDerivAt_id x).sub_const y).const_mul (-(1 / τ))
    have h2 := h1.exp
    have h3 : HasDerivAt (fun z : ℝ => y - z) (-1 : ℝ) x := by
      simpa using (hasDerivAt_id x).const_sub y
    have hmul : HasDerivAt
        (fun z : ℝ => Real.exp (-(1 / τ) * (z - y)) * (y - z))
        (Real.exp (-(1 / τ) * (x - y)) * (-(1 / τ)) * (y - x) +
          Real.exp (-(1 / τ) * (x - y)) * (-1 : ℝ)) x := h2.mul h3
    have hf := hmul.congr_of_eventuallyEq hev
    have hval : Real.exp (-(1 / τ) * (x - y)) * (-(1 / τ)) * (y - x) +
        Real.exp (-(1 / τ) * (x - y)) * (-1 : ℝ) =
        (‖x - y‖ / τ - 1) * laplaceKernel τ x y := by
      unfold laplaceKernel
      rw [Real.norm_eq_abs, abs_of_pos (sub_pos.mpr hxy)]
      ring
    exact hval ▸ hf

/-! ## The classical ODE: `D_p′ = (1/τ)·L_p - 2·Z_p` -/

/-- **The 1-d elliptic structure, made classical.**  The mean-shift numerator
`D_p(x) = ∫ kτ(x,y)(y-x) dp` is differentiable everywhere with

`D_p′ = (1/τ)·L_p - 2·Z_p`.

Combined with the Stage-1 identity `L_p′ = (1/τ)·D_p`, this is the pointwise
second-order ODE `τ²·L_p″ = L_p - 2τ·Z_p`: no distribution theory is needed
anywhere in the 1-d Wronskian program.  Holds for EVERY probability measure —
the integrand's derivative is uniformly bounded by `2`. -/
theorem hasDerivAt_laplaceDisplacementIntegral
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ) :
    HasDerivAt (fun z : ℝ => ∫ y, laplaceWeightedDisplacement τ z y ∂p)
      ((1 / τ) * kernelNormalizer (laplaceCompanionKernel τ) p x -
        2 * kernelNormalizer (laplaceKernel τ) p x) x := by
  have hτ0 : 0 < τ := hτ
  have key := (hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (F := fun z y => laplaceWeightedDisplacement τ z y)
    (F' := fun z y => (‖z - y‖ / τ - 1) * laplaceKernel τ z y)
    (x₀ := x) (bound := fun _ => (2 : ℝ)) (μ := p)
    (s := Set.univ) Filter.univ_mem
    ?_ ?_ ?_ ?_ ?_ ?_).2
  · have hval : (∫ y, (‖x - y‖ / τ - 1) * laplaceKernel τ x y ∂p) =
        (1 / τ) * kernelNormalizer (laplaceCompanionKernel τ) p x -
          2 * kernelNormalizer (laplaceKernel τ) p x := by
      have hptwise : ∀ y : ℝ, (‖x - y‖ / τ - 1) * laplaceKernel τ x y =
          (1 / τ) * laplaceCompanionKernel τ x y - 2 * laplaceKernel τ x y := by
        intro y
        unfold laplaceCompanionKernel
        field_simp
        ring
      simp_rw [hptwise]
      rw [integral_sub ((laplaceCompanionKernel_integrable' τ hτ0 p x).const_mul _)
        ((laplaceKernel_integrable p τ hτ0 x).const_mul _),
        integral_const_mul, integral_const_mul]
      rfl
    rw [hval] at key
    exact key
  · filter_upwards with z
    apply Continuous.aestronglyMeasurable
    unfold laplaceWeightedDisplacement laplaceKernel
    fun_prop
  · exact laplaceWeightedDisplacement_integrable τ hτ0 p x
  · apply Continuous.aestronglyMeasurable
    unfold laplaceKernel
    fun_prop
  · refine ae_of_all _ fun y z _ => ?_
    have hk0 : 0 < laplaceKernel τ z y := Real.exp_pos _
    have hk1 : laplaceKernel τ z y ≤ 1 := laplaceKernel_le_one' hτ0 z y
    have hsk : ‖z - y‖ * laplaceKernel τ z y ≤ τ := by
      unfold laplaceKernel
      exact mul_exp_neg_le' hτ0 (norm_nonneg (z - y))
    rw [Real.norm_eq_abs, abs_mul]
    have h1 : |‖z - y‖ / τ - 1| ≤ ‖z - y‖ / τ + 1 := by
      have := abs_sub (‖z - y‖ / τ) (1 : ℝ)
      calc |‖z - y‖ / τ - 1| ≤ |‖z - y‖ / τ| + |(1 : ℝ)| := abs_sub _ _
        _ = ‖z - y‖ / τ + 1 := by
            rw [abs_of_nonneg (by positivity), abs_one]
    have h2 : |laplaceKernel τ z y| = laplaceKernel τ z y :=
      abs_of_pos hk0
    rw [h2]
    have h3 : (‖z - y‖ / τ) * laplaceKernel τ z y ≤ 1 := by
      rw [div_mul_eq_mul_div, div_le_one hτ0]
      exact hsk
    nlinarith [mul_le_mul_of_nonneg_right h1 hk0.le]
  · exact integrable_const _
  · exact ae_of_all _ fun y z _ =>
      hasDerivAt_laplaceDisplacementKernel τ hτ0 y z

/-! ## The alignment reduction of the open converse -/

/-- **Stage-3 headline: the alignment reduction.**  On the line, zero raw
Laplace drift between arbitrary probability measures together with the
companion-alignment identity `L_p·Z_q = L_q·Z_p` forces `p = q`.  The open
1-d arbitrary-target converse is thereby reduced to the single scalar
identity `K := L_p·Z_q - L_q·Z_p ≡ 0`.  Axiom-free; no moment hypotheses. -/
theorem laplaceZeroDrift_imp_eq_of_companionAligned
    (τ : ℝ) (hτ : ValidBandwidth τ)
    (p q : Measure ℝ) [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (halign : ∀ x : ℝ,
      kernelNormalizer (laplaceCompanionKernel τ) p x *
          kernelNormalizer (laplaceKernel τ) q x =
        kernelNormalizer (laplaceCompanionKernel τ) q x *
          kernelNormalizer (laplaceKernel τ) p x) :
    p = q := by
  have hτ0 : 0 < τ := hτ
  have hZp : ∀ x, 0 < kernelNormalizer (laplaceKernel τ) p x :=
    laplaceKernelNormalizer_pos p τ hτ0
  have hZq : ∀ x, 0 < kernelNormalizer (laplaceKernel τ) q x :=
    laplaceKernelNormalizer_pos q τ hτ0
  have hLp : ∀ x, 0 < kernelNormalizer (laplaceCompanionKernel τ) p x :=
    laplaceCompanionNormalizer_pos τ hτ0 p
  have hLq : ∀ x, 0 < kernelNormalizer (laplaceCompanionKernel τ) q x :=
    laplaceCompanionNormalizer_pos τ hτ0 q
  -- Stage-1 cross-displacement identity from zero drift
  have hcross := (laplaceZeroDrift_iff_crossDisplacement τ hτ p q).mp hzero
  simp only [smul_eq_mul] at hcross
  -- the Wronskian numerator vanishes: D_p·L_q = L_p·D_q
  have hDL : ∀ x : ℝ,
      (∫ y, laplaceWeightedDisplacement τ x y ∂p) *
          kernelNormalizer (laplaceCompanionKernel τ) q x =
        kernelNormalizer (laplaceCompanionKernel τ) p x *
          ∫ y, laplaceWeightedDisplacement τ x y ∂q := by
    intro x
    -- Z_p x * (D_p x * L_q x) = Z_p x * (L_p x * D_q x), then cancel Z_p
    apply mul_left_cancel₀ (ne_of_gt (hZp x))
    calc kernelNormalizer (laplaceKernel τ) p x *
          ((∫ y, laplaceWeightedDisplacement τ x y ∂p) *
            kernelNormalizer (laplaceCompanionKernel τ) q x)
        = (kernelNormalizer (laplaceCompanionKernel τ) q x *
            kernelNormalizer (laplaceKernel τ) p x) *
          ∫ y, laplaceWeightedDisplacement τ x y ∂p := by ring
      _ = (kernelNormalizer (laplaceCompanionKernel τ) p x *
            kernelNormalizer (laplaceKernel τ) q x) *
          ∫ y, laplaceWeightedDisplacement τ x y ∂p := by rw [halign x]
      _ = kernelNormalizer (laplaceCompanionKernel τ) p x *
          (kernelNormalizer (laplaceKernel τ) q x *
            ∫ y, laplaceWeightedDisplacement τ x y ∂p) := by ring
      _ = kernelNormalizer (laplaceCompanionKernel τ) p x *
          (kernelNormalizer (laplaceKernel τ) p x *
            ∫ y, laplaceWeightedDisplacement τ x y ∂q) := by rw [hcross x]
      _ = kernelNormalizer (laplaceKernel τ) p x *
          (kernelNormalizer (laplaceCompanionKernel τ) p x *
            ∫ y, laplaceWeightedDisplacement τ x y ∂q) := by ring
  -- the ratio L_p/L_q has zero derivative everywhere
  have hratio : ∀ x : ℝ, HasDerivAt
      (fun z => kernelNormalizer (laplaceCompanionKernel τ) p z /
        kernelNormalizer (laplaceCompanionKernel τ) q z) 0 x := by
    intro x
    have hu := hasDerivAt_laplaceCompanionNormalizer τ hτ p x
    have hv := hasDerivAt_laplaceCompanionNormalizer τ hτ q x
    have hdiv := hu.div hv (ne_of_gt (hLq x))
    have hzero' : ((1 / τ) * (∫ y, laplaceWeightedDisplacement τ x y ∂p) *
          kernelNormalizer (laplaceCompanionKernel τ) q x -
        kernelNormalizer (laplaceCompanionKernel τ) p x *
          ((1 / τ) * ∫ y, laplaceWeightedDisplacement τ x y ∂q)) /
        kernelNormalizer (laplaceCompanionKernel τ) q x ^ 2 = 0 := by
      rw [div_eq_zero_iff]
      left
      linear_combination (1 / τ) * hDL x
    rwa [hzero'] at hdiv
  -- hence the ratio is a positive constant `c`
  have hconst : ∀ x : ℝ,
      kernelNormalizer (laplaceCompanionKernel τ) p x /
        kernelNormalizer (laplaceCompanionKernel τ) q x =
      kernelNormalizer (laplaceCompanionKernel τ) p 0 /
        kernelNormalizer (laplaceCompanionKernel τ) q 0 := by
    intro x
    exact is_const_of_deriv_eq_zero
      (fun t => (hratio t).differentiableAt)
      (fun t => (hratio t).deriv) x 0
  set c : ℝ := kernelNormalizer (laplaceCompanionKernel τ) p 0 /
    kernelNormalizer (laplaceCompanionKernel τ) q 0 with hc
  have hcpos : 0 < c := div_pos (hLp 0) (hLq 0)
  have hLc : ∀ x, kernelNormalizer (laplaceCompanionKernel τ) p x =
      c * kernelNormalizer (laplaceCompanionKernel τ) q x := by
    intro x
    have h := hconst x
    rw [div_eq_iff (ne_of_gt (hLq x))] at h
    exact h
  -- alignment converts the companion ratio into the Laplace-normalizer ratio
  have hZc : ∀ x, kernelNormalizer (laplaceKernel τ) p x =
      c * kernelNormalizer (laplaceKernel τ) q x := by
    intro x
    have h := halign x
    rw [hLc x] at h
    -- h : (c·L_q)·Z_q = L_q·Z_p; cancel the positive factor L_q
    have h' : kernelNormalizer (laplaceCompanionKernel τ) q x *
        (c * kernelNormalizer (laplaceKernel τ) q x) =
        kernelNormalizer (laplaceCompanionKernel τ) q x *
          kernelNormalizer (laplaceKernel τ) p x := by
      linear_combination h
    exact (mul_left_cancel₀ (ne_of_gt (hLq x)) h').symm
  -- smoothing injectivity against the scaled measure `c • q`
  have hq' : IsFiniteMeasure ((ENNReal.ofReal c) • q) := by
    constructor
    rw [Measure.smul_apply, smul_eq_mul]
    exact ENNReal.mul_lt_top ENNReal.ofReal_lt_top (measure_lt_top q _)
  have hZq' : ∀ x, kernelNormalizer (laplaceKernel τ) p x =
      kernelNormalizer (laplaceKernel τ) ((ENNReal.ofReal c) • q) x := by
    intro x
    unfold kernelNormalizer
    rw [integral_smul_measure, ENNReal.toReal_ofReal hcpos.le, smul_eq_mul]
    exact hZc x
  have hpq' : p = (ENNReal.ofReal c) • q := by
    haveI := hq'
    exact laplaceKernelNormalizer_injective τ hτ p _ hZq'
  -- total mass pins the constant to one
  have hmass : ENNReal.ofReal c = 1 := by
    have h1 : p Set.univ = 1 := measure_univ
    rw [hpq', Measure.smul_apply, smul_eq_mul, measure_univ, mul_one] at h1
    exact h1
  rw [hpq', hmass, one_smul]

end DriftingIdentifiability
