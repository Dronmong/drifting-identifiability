import DriftingIdentifiability.LaplaceWronskian

/-!
# Milestone 5: the companion Wronskian is a primitive of the alignment defect

Let `L_p := kernelNormalizer (laplaceCompanionKernel τ) p` (the companion
normalizer, `C²` classically) and `Z_p := kernelNormalizer (laplaceKernel τ) p`.
The certified derivatives are `L_p′ = (1/τ)D_p` and `D_p′ = (1/τ)L_p − 2Z_p`
(`hasDerivAt_laplaceCompanionNormalizer`, `hasDerivAt_laplaceDisplacementIntegral`).

Define the **companion Wronskian** `V := L_p·L_q′ − L_p′·L_q` and the
**alignment defect** `K := L_p·Z_q − L_q·Z_p` (whose vanishing drives
`laplaceZeroDrift_imp_eq_of_companionAligned`).  Then, UNCONDITIONALLY and
classically,

`hasDerivAt_laplaceCompanionWronskian`:  `V′(x) = −(2/τ)·K(x)`.

So `K` is a perfect derivative — the derivative of the companion Wronskian.
Consequently `V ≡ const ⟹ K ≡ 0 ⟹ p = q` under zero drift
(`laplaceZeroDrift_imp_eq_of_companionWronskian_const`), i.e. "the companion
normalizers are aligned (their Wronskian is constant)".  This is the classical
(C²) companion-side counterpart of the normalizer-Wronskian gate
`laplaceKernelNormalizer_wronskian_zero_imp_eq`; it also yields the necessary
condition `∫ K = 0` (since `V → 0` at both `±∞`).

Milestone 5 remains open: forcing `V ≡ const` (equivalently `K ≡ 0`, or `W ≡ 0`
for the raw normalizer Wronskian) under zero drift is the analytic core; the
obstruction sits at the zeros of the mean shift and is resolved only globally
(see the roadmap).
-/

open MeasureTheory

namespace DriftingIdentifiability

open Paper

/-- The companion Wronskian `V = L_p·L_q′ − L_p′·L_q`, using the certified
companion derivative `L_p′ = (1/τ)·D_p`. -/
noncomputable def laplaceCompanionWronskian (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  kernelNormalizer (laplaceCompanionKernel τ) p x
      * ((1 / τ) * ∫ y, laplaceWeightedDisplacement τ x y ∂q)
    - ((1 / τ) * ∫ y, laplaceWeightedDisplacement τ x y ∂p)
      * kernelNormalizer (laplaceCompanionKernel τ) q x

/-- The companion alignment defect `K = L_p·Z_q − L_q·Z_p`. -/
noncomputable def laplaceCompanionAlignmentDefect (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  kernelNormalizer (laplaceCompanionKernel τ) p x * kernelNormalizer (laplaceKernel τ) q x
    - kernelNormalizer (laplaceCompanionKernel τ) q x * kernelNormalizer (laplaceKernel τ) p x

/-- **The companion Wronskian is a primitive of the alignment defect:**
`V′(x) = −(2/τ)·K(x)`.  Unconditional and classical. -/
theorem hasDerivAt_laplaceCompanionWronskian (τ : ℝ) (hτ : ValidBandwidth τ)
    (p q : Measure ℝ) [IsProbabilityMeasure p] [IsProbabilityMeasure q] (x : ℝ) :
    HasDerivAt (fun t => laplaceCompanionWronskian τ p q t)
      (-(2 / τ) * laplaceCompanionAlignmentDefect τ p q x) x := by
  have hLp := hasDerivAt_laplaceCompanionNormalizer τ hτ p x
  have hLq := hasDerivAt_laplaceCompanionNormalizer τ hτ q x
  have hDp := hasDerivAt_laplaceDisplacementIntegral τ hτ p x
  have hDq := hasDerivAt_laplaceDisplacementIntegral τ hτ q x
  have hLq' := hDq.const_mul (1 / τ)
  have hLp' := hDp.const_mul (1 / τ)
  have hprod := (hLp.mul hLq').sub (hLp'.mul hLq)
  refine hprod.congr_deriv ?_
  unfold laplaceCompanionAlignmentDefect
  ring

/-- **Companion-Wronskian gate.**  If the companion Wronskian is constant, then
zero drift forces `p = q`.  (Equivalent to companion alignment `K ≡ 0` since
`V′ = −(2/τ)K`.) -/
theorem laplaceZeroDrift_imp_eq_of_companionWronskian_const
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hV : ∀ x, laplaceCompanionWronskian τ p q x = laplaceCompanionWronskian τ p q 0) :
    p = q := by
  have hτ0 : 0 < τ := hτ
  have hK : ∀ x, laplaceCompanionAlignmentDefect τ p q x = 0 := by
    intro x
    have hd := hasDerivAt_laplaceCompanionWronskian τ hτ p q x
    have hc : HasDerivAt (fun t => laplaceCompanionWronskian τ p q t) 0 x := by
      have hfun : (fun t => laplaceCompanionWronskian τ p q t)
          = fun _ => laplaceCompanionWronskian τ p q 0 := funext hV
      rw [hfun]; exact hasDerivAt_const x _
    have hval := hd.unique hc
    have hne : -(2 / τ) ≠ 0 := neg_ne_zero.mpr (ne_of_gt (by positivity))
    exact (mul_eq_zero.mp hval).resolve_left hne
  refine laplaceZeroDrift_imp_eq_of_companionAligned τ hτ p q hzero (fun x => ?_)
  have hkx := hK x
  unfold laplaceCompanionAlignmentDefect at hkx
  linarith [hkx]

end DriftingIdentifiability
