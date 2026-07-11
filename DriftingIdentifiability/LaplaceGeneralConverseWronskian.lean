import DriftingIdentifiability.LaplaceGeneralConverseReduction
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

private lemma massDet_eq_zero_of_pairings_eq_zero_of_p_sides
    {pm qm pp qp pc qc ph qh : ℝ}
    (hpc : pc ≠ 0) (hph : ph ≠ 0)
    (hfactor : pm * ph + pp * pc ≠ 0)
    (hA : qc * pm - pc * qm = 0)
    (hB : qm * ph - qp * pc - pm * qh + pp * qc = 0)
    (hC : ph * qp - qh * pp = 0) :
    pp * qm - pm * qp = 0 := by
  let l : ℝ := qc / pc
  let m : ℝ := qh / ph
  have hqc : qc = l * pc := by
    dsimp [l]
    field_simp [hpc]
  have hqh : qh = m * ph := by
    dsimp [m]
    field_simp [hph]
  have hqm : qm = l * pm := by
    have hA' : l * pc * pm - pc * qm = 0 := by
      simpa [hqc] using hA
    have hA'' : pc * (l * pm - qm) = 0 := by
      nlinarith only [hA']
    have hdiff : l * pm - qm = 0 :=
      (mul_eq_zero.mp hA'').resolve_left hpc
    linarith
  have hqp : qp = m * pp := by
    have hC' : ph * qp - m * ph * pp = 0 := by
      simpa [hqh] using hC
    have hC'' : ph * (qp - m * pp) = 0 := by
      nlinarith only [hC']
    have hdiff : qp - m * pp = 0 :=
      (mul_eq_zero.mp hC'').resolve_left hph
    linarith
  have hB' : (l - m) * (pm * ph + pp * pc) = 0 := by
    rw [hqm, hqp, hqh, hqc] at hB
    ring_nf at hB
    ring_nf
    nlinarith only [hB]
  have hlm : l = m := by
    have hdiff : l - m = 0 :=
      (mul_eq_zero.mp hB').resolve_right hfactor
    linarith
  rw [hqm, hqp, hlm]
  ring

/-- The raw-normalizer Wronskian
`W = Z_p⁺·Z_q - Z_p·Z_q⁺`, where `Z⁺` denotes the Milestone-4 right derivative
of the Laplace normalizer. -/
noncomputable def laplaceKernelNormalizerWronskian
    (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  laplaceKernelNormalizerRightDerivCoeff τ p x *
      kernelNormalizer (laplaceKernel τ) q x -
    kernelNormalizer (laplaceKernel τ) p x *
      laplaceKernelNormalizerRightDerivCoeff τ q x

/-- The raw-normalizer Wronskian is just the determinant of the lower/upper
exponential mass split.  This is the concrete Milestone-5 coordinate:
`W = (2/τ)(P⁺Q⁻ - P⁻Q⁺)`. -/
theorem laplaceKernelNormalizerWronskian_eq_massDet
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    laplaceKernelNormalizerWronskian τ p q x =
      (2 / τ) *
        (upperExpMass τ p x * lowerExpMass τ q x -
          lowerExpMass τ p x * upperExpMass τ q x) := by
  unfold laplaceKernelNormalizerWronskian laplaceKernelNormalizerRightDerivCoeff
  rw [laplaceKernelNormalizer_eq_lower_upper τ hτ p x,
    laplaceKernelNormalizer_eq_lower_upper τ hτ q x]
  have hcancel : Real.exp (-x / τ) * Real.exp (x / τ) = 1 := by
    rw [← Real.exp_add]
    rw [show -x / τ + x / τ = 0 by ring, Real.exp_zero]
  have hcancel' : Real.exp (-(x * τ⁻¹)) * Real.exp (x * τ⁻¹) = 1 := by
    simpa [div_eq_mul_inv] using hcancel
  field_simp [hτ.ne']
  ring_nf
  have hterm :
      Real.exp (-(x * τ⁻¹)) * lowerExpMass τ p x *
            Real.exp (x * τ⁻¹) * upperExpMass τ q x * 2 =
        lowerExpMass τ p x * upperExpMass τ q x * 2 := by
    rw [show Real.exp (-(x * τ⁻¹)) * lowerExpMass τ p x *
          Real.exp (x * τ⁻¹) * upperExpMass τ q x * 2 =
        (Real.exp (-(x * τ⁻¹)) * Real.exp (x * τ⁻¹)) *
          (lowerExpMass τ p x * upperExpMass τ q x * 2) by ring,
      hcancel', one_mul]
  rw [hterm, hcancel', one_mul]

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

/-- Same Wronskian gate, stated using the named Wronskian function. -/
theorem laplaceKernelNormalizer_wronskian_eq_zero_imp_eq
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hW : ∀ x, laplaceKernelNormalizerWronskian τ p q x = 0) : p = q :=
  laplaceKernelNormalizer_wronskian_zero_imp_eq τ hτ p q (fun x => by
    have hx := hW x
    unfold laplaceKernelNormalizerWronskian at hx
    exact sub_eq_zero.mp hx)

/-- On a zero-mass gap of `p+q`, the raw-normalizer Wronskian is constant.
In mass-determinant form this is immediate from constancy of the four
one-sided exponential masses. -/
theorem laplaceKernelNormalizerWronskian_eq_of_sum_gap_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q]
    {u v x : ℝ} (hgap : (p + q) (Set.Ioo u v) = 0)
    (hux : u < x) (hxv : x < v) :
    laplaceKernelNormalizerWronskian τ p q x =
      laplaceKernelNormalizerWronskian τ p q u := by
  have hpGap : p (Set.Ioo u v) = 0 :=
    left_gap_measure_zero_of_sum_gap_zero hgap
  have hqGap : q (Set.Ioo u v) = 0 :=
    right_gap_measure_zero_of_sum_gap_zero hgap
  rw [laplaceKernelNormalizerWronskian_eq_massDet τ hτ p q x,
    laplaceKernelNormalizerWronskian_eq_massDet τ hτ p q u]
  have hPm : lowerExpMass τ p x = lowerExpMass τ p u :=
    lowerExpMass_eq_of_gap_zero τ hτ p hpGap hux hxv
  have hQm : lowerExpMass τ q x = lowerExpMass τ q u :=
    lowerExpMass_eq_of_gap_zero τ hτ q hqGap hux hxv
  have hPp : upperExpMass τ p x = upperExpMass τ p u :=
    upperExpMass_eq_of_gap_zero τ hτ p hpGap hux hxv
  have hQp : upperExpMass τ q x = upperExpMass τ q u :=
    upperExpMass_eq_of_gap_zero τ hτ q hqGap hux hxv
  rw [hPm, hQm, hPp, hQp]

/-- **Local Milestone-5 propagation brick.**  On a zero-mass gap of `p+q`,
zero raw Laplace drift forces the raw-normalizer Wronskian to vanish at any
cut where the `p`-side lower/upper compensated coordinates are genuinely
connected.

The proof is the algebraic core of the gap/shooting picture: on a gap the
three zero-drift coefficients `𝔞, 𝔟, 𝔠` are all zero.  The equations
`𝔞 = 0` and `𝔠 = 0` identify the lower and upper `q/p` ratios; `𝔟 = 0` forces
those two ratios to agree whenever the displayed connection factor is nonzero.
That agreement is exactly the mass determinant defining `W`. -/
theorem laplaceKernelNormalizerWronskian_eq_zero_on_gap_of_p_connection
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {u v x : ℝ} (hgap : (p + q) (Set.Ioo u v) = 0)
    (hux : u < x) (hxv : x < v)
    (hpc : lowerCompensatedMoment τ p x ≠ 0)
    (hph : upperCompensatedMoment τ p x ≠ 0)
    (hfactor :
      lowerExpMass τ p x * upperCompensatedMoment τ p x +
        upperExpMass τ p x * lowerCompensatedMoment τ p x ≠ 0) :
    laplaceKernelNormalizerWronskian τ p q x = 0 := by
  have hA : truncatedPairing τ p q x = 0 :=
    truncatedPairing_eq_zero_on_gap τ hτ p q hzero hgap hux hxv
  have hB : middlePairing τ p q x = 0 :=
    (zeroDrift_truncatedPairing_eq_zero_iff_middlePairing τ hτ p q hzero x).mp hA
  have hC : upperPairing τ p q x = 0 :=
    (zeroDrift_truncatedPairing_eq_zero_iff_upperPairing τ hτ p q hzero x).mp hA
  have hdet :
      upperExpMass τ p x * lowerExpMass τ q x -
          lowerExpMass τ p x * upperExpMass τ q x = 0 := by
    exact massDet_eq_zero_of_pairings_eq_zero_of_p_sides
      (pm := lowerExpMass τ p x)
      (qm := lowerExpMass τ q x)
      (pp := upperExpMass τ p x)
      (qp := upperExpMass τ q x)
      (pc := lowerCompensatedMoment τ p x)
      (qc := lowerCompensatedMoment τ q x)
      (ph := upperCompensatedMoment τ p x)
      (qh := upperCompensatedMoment τ q x)
      hpc hph hfactor
      (by simpa [truncatedPairing] using hA)
      (by simpa [middlePairing] using hB)
      (by simpa [upperPairing] using hC)
  rw [laplaceKernelNormalizerWronskian_eq_massDet τ hτ p q x, hdet, mul_zero]

/-- A more ergonomic version of
`laplaceKernelNormalizerWronskian_eq_zero_on_gap_of_p_connection`: if the
`p`-law has strictly positive lower and upper exponential masses and
compensated moments at the cut, the connection factor is automatically
nonzero. -/
theorem laplaceKernelNormalizerWronskian_eq_zero_on_gap_of_p_twoSided
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {u v x : ℝ} (hgap : (p + q) (Set.Ioo u v) = 0)
    (hux : u < x) (hxv : x < v)
    (hPm : 0 < lowerExpMass τ p x)
    (hPp : 0 < upperExpMass τ p x)
    (hPc : 0 < lowerCompensatedMoment τ p x)
    (hPh : 0 < upperCompensatedMoment τ p x) :
    laplaceKernelNormalizerWronskian τ p q x = 0 := by
  refine laplaceKernelNormalizerWronskian_eq_zero_on_gap_of_p_connection
    τ hτ p q hzero hgap hux hxv hPc.ne' hPh.ne' ?_
  have hfactor_pos :
      0 <
        lowerExpMass τ p x * upperCompensatedMoment τ p x +
          upperExpMass τ p x * lowerCompensatedMoment τ p x := by
    nlinarith [mul_pos hPm hPh, mul_pos hPp hPc]
  exact hfactor_pos.ne'

private lemma upperExpMass_eq_zero_of_right_tail
    (τ : ℝ) (p : Measure ℝ) [IsFiniteMeasure p] {M x : ℝ}
    (hp : p (Set.Ioi M) = 0) (hMx : M ≤ x) :
    upperExpMass τ p x = 0 := by
  unfold upperExpMass
  exact setIntegral_eq_zero_of_measure_zero (fun y => Real.exp (-y / τ))
    (measure_mono_null (fun y hy => lt_of_le_of_lt hMx hy) hp)

private lemma lowerExpMass_eq_zero_of_left_tail
    (τ : ℝ) (p : Measure ℝ) [IsFiniteMeasure p] {M x : ℝ}
    (hp : p (Set.Iic M) = 0) (hxM : x ≤ M) :
    lowerExpMass τ p x = 0 := by
  unfold lowerExpMass
  exact setIntegral_eq_zero_of_measure_zero (fun y => Real.exp (y / τ))
    (measure_mono_null (fun y hy => hy.trans hxM) hp)

/-- If both measures have no mass to the right of `M`, then the raw-normalizer
Wronskian vanishes throughout the right tail. -/
theorem laplaceKernelNormalizerWronskian_eq_zero_of_right_tail
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] {M x : ℝ}
    (hp : p (Set.Ioi M) = 0) (hq : q (Set.Ioi M) = 0) (hMx : M ≤ x) :
    laplaceKernelNormalizerWronskian τ p q x = 0 := by
  rw [laplaceKernelNormalizerWronskian_eq_massDet τ hτ p q x,
    upperExpMass_eq_zero_of_right_tail τ p hp hMx,
    upperExpMass_eq_zero_of_right_tail τ q hq hMx]
  ring

/-- If both measures have no mass to the left of or at `M`, then the
raw-normalizer Wronskian vanishes throughout the left tail. -/
theorem laplaceKernelNormalizerWronskian_eq_zero_of_left_tail
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] {M x : ℝ}
    (hp : p (Set.Iic M) = 0) (hq : q (Set.Iic M) = 0) (hxM : x ≤ M) :
    laplaceKernelNormalizerWronskian τ p q x = 0 := by
  rw [laplaceKernelNormalizerWronskian_eq_massDet τ hτ p q x,
    lowerExpMass_eq_zero_of_left_tail τ p hp hxM,
    lowerExpMass_eq_zero_of_left_tail τ q hq hxM]
  ring

end DriftingIdentifiability
