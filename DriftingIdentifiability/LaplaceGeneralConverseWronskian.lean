import DriftingIdentifiability.LaplaceGeneralConverseBalance
import DriftingIdentifiability.LaplaceInjectivity
import Mathlib.Analysis.Calculus.MeanValue

/-!
# Milestone 5 gate: the normalizer-Wronskian reduction

A new, clean reduction of the open core.  Let `Z_p := kernelNormalizer
(laplaceKernel τ) p` and let `Z_p'` be its (certified, Milestone-4) right
derivative `laplaceKernelNormalizerRightDerivCoeff τ p`.  Define the
**normalizer Wronskian** `W(x) := Z_p'(x) Z_q(x) - Z_p(x) Z_q'(x)`.

`laplaceKernelNormalizer_wronskian_zero_imp_eq`: **`W ≡ 0 ⟹ p = q`**
(for probability measures; zero drift is not needed).  Proof: `W ≡ 0` makes
the ratio `Z_p / Z_q` have right-derivative `0`, hence constant `= c > 0`, so
`Z_p = c·Z_q`; Laplace smoothing injectivity against `(ofReal c)·q` gives
`p = (ofReal c)·q`, and total mass forces `c = 1`.

Why this matters for Milestone 5.  Under zero drift the normalizer Wronskian
is *globally regular* — unconditionally `W' = -(2/τ)(Z_q dp - Z_p dq)` (a
finite signed measure), `W` is constant on every gap of `supp(p+q)`, and
`W → 0` at both `±∞` (outside compact support `Z_p, Z_q` are proportional
exponentials, so `W` vanishes there).  In contrast the truncated pairing `𝔞`
and the companion alignment `K` have singular/degenerate coordinates.  So `W`
is the recommended coordinate in which to prove `≡ 0`.  The remaining open
step is exactly `zero drift ⟹ W ≡ 0`; the obstruction is at the zeros of the
mean shift `m = D_p/Z_p`, where the second-order ODE that `Z_p, Z_q` jointly
solve degenerates (see the roadmap's Milestone-5 analysis).
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-- The raw Laplace normalizer `Z_p(x) = ∫ kτ(x,y) dp` is continuous. -/
lemma continuous_laplaceKernelNormalizer (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsFiniteMeasure p] : Continuous (fun x => kernelNormalizer (laplaceKernel τ) p x) := by
  have hτ0 : 0 < τ := hτ
  have hcy : ∀ x : ℝ, Continuous fun y : ℝ => laplaceKernel τ x y := by
    intro x; unfold laplaceKernel; fun_prop
  have hcx : ∀ y : ℝ, Continuous fun x : ℝ => laplaceKernel τ x y := by
    intro y; unfold laplaceKernel; fun_prop
  unfold kernelNormalizer
  rw [continuous_iff_continuousAt]
  intro x₀
  refine continuousAt_of_dominated
    (Eventually.of_forall fun x => (hcy x).aestronglyMeasurable)
    (Eventually.of_forall fun x => ae_of_all p fun y => ?_)
    (integrable_const (1 : ℝ))
    (ae_of_all p fun y => (hcx y).continuousAt)
  rw [Real.norm_eq_abs, abs_of_nonneg (by unfold laplaceKernel; exact (Real.exp_pos _).le)]
  unfold laplaceKernel
  rw [Real.exp_le_one_iff]
  nlinarith [norm_nonneg (x - y), (by positivity : (0 : ℝ) ≤ 1 / τ)]

/-- **Milestone-5 gate.**  If the normalizer Wronskian
`Z_p'·Z_q - Z_p·Z_q'` vanishes identically, then `p = q`.  (`Z_p'` is the
Milestone-4 right derivative `laplaceKernelNormalizerRightDerivCoeff`.)  Zero
drift is not required. -/
theorem laplaceKernelNormalizer_wronskian_zero_imp_eq
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hW : ∀ x, laplaceKernelNormalizerRightDerivCoeff τ p x
          * kernelNormalizer (laplaceKernel τ) q x
        = kernelNormalizer (laplaceKernel τ) p x
          * laplaceKernelNormalizerRightDerivCoeff τ q x) : p = q := by
  have hτ0 : 0 < τ := hτ
  have hZppos : ∀ x, 0 < kernelNormalizer (laplaceKernel τ) p x :=
    fun x => laplaceKernelNormalizer_pos p τ hτ0 x
  have hZqpos : ∀ x, 0 < kernelNormalizer (laplaceKernel τ) q x :=
    fun x => laplaceKernelNormalizer_pos q τ hτ0 x
  have hcontZp := continuous_laplaceKernelNormalizer τ hτ p
  have hcontZq := continuous_laplaceKernelNormalizer τ hτ q
  -- the ratio `Z_p / Z_q` is constant
  have hconst : ∀ a b : ℝ, a ≤ b →
      kernelNormalizer (laplaceKernel τ) p b / kernelNormalizer (laplaceKernel τ) q b
        = kernelNormalizer (laplaceKernel τ) p a / kernelNormalizer (laplaceKernel τ) q a := by
    intro a b hab
    have hcont : ContinuousOn
        (fun t => kernelNormalizer (laplaceKernel τ) p t / kernelNormalizer (laplaceKernel τ) q t)
        (Set.Icc a b) :=
      hcontZp.continuousOn.div hcontZq.continuousOn (fun t _ => (hZqpos t).ne')
    have hderiv : ∀ t ∈ Set.Ico a b, HasDerivWithinAt
        (fun t => kernelNormalizer (laplaceKernel τ) p t / kernelNormalizer (laplaceKernel τ) q t)
        0 (Set.Ici t) t := by
      intro t _
      have hdiv := (hasDerivWithinAt_Ici_laplaceKernelNormalizer τ hτ p t).div
        (hasDerivWithinAt_Ici_laplaceKernelNormalizer τ hτ q t) (hZqpos t).ne'
      have hz : (laplaceKernelNormalizerRightDerivCoeff τ p t
            * kernelNormalizer (laplaceKernel τ) q t
          - kernelNormalizer (laplaceKernel τ) p t
            * laplaceKernelNormalizerRightDerivCoeff τ q t)
            / kernelNormalizer (laplaceKernel τ) q t ^ 2 = 0 := by
        rw [hW t, sub_self, zero_div]
      rw [← hz]; exact hdiv
    exact constant_of_has_deriv_right_zero hcont hderiv b ⟨hab, le_refl b⟩
  -- extract the constant `c` and show `Z_p = c·Z_q`
  set c := kernelNormalizer (laplaceKernel τ) p 0
    / kernelNormalizer (laplaceKernel τ) q 0 with hc_def
  have hc_pos : 0 < c := div_pos (hZppos 0) (hZqpos 0)
  have hZeq : ∀ x, kernelNormalizer (laplaceKernel τ) p x
      = c * kernelNormalizer (laplaceKernel τ) q x := by
    intro x
    have hr : kernelNormalizer (laplaceKernel τ) p x
        / kernelNormalizer (laplaceKernel τ) q x = c := by
      rcases le_total 0 x with hx | hx
      · rw [hc_def]; exact hconst 0 x hx
      · rw [hc_def]; exact (hconst x 0 hx).symm
    field_simp [ (hZqpos x).ne' ] at hr
    linarith [hr]
  -- smoothing injectivity against the scaled measure `c • q`
  have hq' : IsFiniteMeasure ((ENNReal.ofReal c) • q) := by
    constructor
    rw [Measure.smul_apply, smul_eq_mul]
    exact ENNReal.mul_lt_top ENNReal.ofReal_lt_top (measure_lt_top q _)
  have hZq' : ∀ x, kernelNormalizer (laplaceKernel τ) p x
      = kernelNormalizer (laplaceKernel τ) ((ENNReal.ofReal c) • q) x := by
    intro x
    unfold kernelNormalizer
    rw [integral_smul_measure, ENNReal.toReal_ofReal hc_pos.le, smul_eq_mul]
    exact hZeq x
  have hpq' : p = (ENNReal.ofReal c) • q := by
    haveI := hq'
    exact laplaceKernelNormalizer_injective τ hτ p _ hZq'
  have hmass : ENNReal.ofReal c = 1 := by
    have h1 : p Set.univ = 1 := measure_univ
    rw [hpq', Measure.smul_apply, smul_eq_mul, measure_univ, mul_one] at h1
    exact h1
  rw [hpq', hmass, one_smul]

end DriftingIdentifiability
