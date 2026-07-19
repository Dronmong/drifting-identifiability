import DriftingIdentifiability.TrustedBoundary
import Mathlib.MeasureTheory.Measure.Dirac
import Mathlib.MeasureTheory.Integral.Bochner.Basic

/-!
# Finite-range kernels are mass-blind (dynamics roadmap, A2 / frontier D1)

The general Euclidean converse `laplaceZeroDrift_identifies_euclidean` used the
*exponential tail* of the Laplace kernel in an essential way: it made the two
clusters of a two-mode target interact, however faintly.  This file proves the
sharp converse — that the tail is **load-bearing** — by exhibiting a clean
counterexample for every *finite-range* kernel.

For a kernel that vanishes once the argument separation reaches `ρ`, take two
point masses `a`, `b` separated by at least `2ρ`.  The two-atom mixtures

```
p = w·δ_a + (1−w)·δ_b,     q = w'·δ_a + (1−w')·δ_b
```

have **identical normalized mean-shift fields everywhere**: near either atom
the other is out of range, so the local weight cancels in the ratio `D/Z`, and
in the dead zone between/around the clusters both normalizers vanish (the Lean
`(0)⁻¹ • D = 0` convention makes the field `0` there).  Hence the drift
`V(p,q) ≡ 0` while `p ≠ q` whenever `w ≠ w'`.

Consequence: the zero-drift converse **fails for every finite-range kernel**,
in every dimension, with no regularity beyond finite range.  Strict positivity
of the kernel — not RKHS-characteristicness — is the relevant dividing line for
*normalized displacement drift*: a compactly supported kernel can still be
characteristic in the ordinary kernel-mean-embedding sense yet be blind to
cluster mass here.
-/

open MeasureTheory Set
open scoped ENNReal

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [MeasurableSpace E] [MeasurableSingletonClass E] [CompleteSpace E]

/-- A kernel has **finite range** `ρ` if it vanishes once its arguments are at
least `ρ` apart. -/
def FiniteRangeKernel (k : E → E → ℝ) (ρ : ℝ) : Prop :=
  ∀ x y : E, ρ ≤ ‖x - y‖ → k x y = 0

/-- The two-atom mixture `w·δ_a + (1−w)·δ_b`. -/
noncomputable def twoAtomMix (w : ℝ) (a b : E) : Measure E :=
  ENNReal.ofReal w • Measure.dirac a + ENNReal.ofReal (1 - w) • Measure.dirac b

/-! ## Integration against the two-atom mixture -/

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- Every integral against the two-atom mixture is the corresponding
weighted average of atom values. -/
lemma integral_twoAtomMix {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    [CompleteSpace F]
    (w : ℝ) (hw0 : 0 ≤ w) (hw1 : w ≤ 1) (a b : E) (f : E → F) :
    ∫ y, f y ∂(twoAtomMix w a b) = w • f a + (1 - w) • f b := by
  rw [twoAtomMix,
    integral_add_measure
      ((integrable_dirac (by finiteness)).smul_measure ENNReal.ofReal_ne_top)
      ((integrable_dirac (by finiteness)).smul_measure ENNReal.ofReal_ne_top),
    integral_smul_measure, integral_smul_measure, integral_dirac, integral_dirac,
    ENNReal.toReal_ofReal hw0, ENNReal.toReal_ofReal (by linarith)]

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [MeasurableSingletonClass E]
  [CompleteSpace E] in
lemma twoAtomMix_isProbabilityMeasure {w : ℝ} (hw0 : 0 ≤ w) (hw1 : w ≤ 1)
    (a b : E) : IsProbabilityMeasure (twoAtomMix w a b) := by
  constructor
  rw [twoAtomMix, Measure.add_apply, Measure.smul_apply, Measure.smul_apply,
    measure_univ, measure_univ, smul_eq_mul, smul_eq_mul, mul_one, mul_one,
    ← ENNReal.ofReal_add hw0 (by linarith),
    show w + (1 - w) = 1 by ring, ENNReal.ofReal_one]

/-- Closed form of the mean-shift field for a two-atom mixture. -/
lemma meanShift_twoAtomMix (k : E → E → ℝ) (w : ℝ) (hw0 : 0 ≤ w) (hw1 : w ≤ 1)
    (a b x : E) :
    meanShift k (twoAtomMix w a b) x =
      (w * k x a + (1 - w) * k x b)⁻¹ •
        ((w * k x a) • (a - x) + ((1 - w) * k x b) • (b - x)) := by
  unfold meanShift kernelNormalizer
  rw [integral_twoAtomMix w hw0 hw1 a b (fun y => k x y),
    integral_twoAtomMix w hw0 hw1 a b (fun y => k x y • (y - x))]
  simp only [smul_eq_mul, smul_smul]

/-! ## The weight cancellation -/

omit [MeasurableSpace E] [MeasurableSingletonClass E] [CompleteSpace E] in
/-- The single-active-atom collapse: `γ⁻¹ • (γ • v)` is either `v` (if
`γ ≠ 0`) or `0`, so it agrees for two scalars that vanish together. -/
lemma inv_smul_smul_eq_of_iff {γ γ' : ℝ} (v : E) (h : γ = 0 ↔ γ' = 0) :
    γ⁻¹ • (γ • v) = γ'⁻¹ • (γ' • v) := by
  by_cases hγ : γ = 0
  · rw [hγ, h.mp hγ]
  · have hγ' : γ' ≠ 0 := fun hh => hγ (h.mpr hh)
    rw [inv_smul_smul₀ hγ, inv_smul_smul₀ hγ']

/-- Away from an atom carrying all the weight, `w·k = 0 ⟺ k = 0`. -/
private lemma mul_kernel_eq_zero_iff {w α : ℝ} (hw : w ≠ 0) :
    w * α = 0 ↔ α = 0 :=
  ⟨fun h => (mul_eq_zero.mp h).resolve_left hw, fun h => by rw [h, mul_zero]⟩

/-! ## Mass blindness -/

/-- **The mean-shift field cannot see cluster mass.**  For a finite-range
kernel and two well-separated atoms, the normalized mean shift is the same for
every interior mixing weight in `(0,1)` — the weight cancels near each atom and
both normalizers vanish elsewhere. -/
theorem meanShift_twoAtomMix_eq_of_separated {k : E → E → ℝ} {ρ : ℝ}
    (hk : FiniteRangeKernel k ρ) {a b : E} (hab : 2 * ρ ≤ ‖a - b‖)
    {w w' : ℝ} (hw : w ∈ Ioo (0 : ℝ) 1) (hw' : w' ∈ Ioo (0 : ℝ) 1) (x : E) :
    meanShift k (twoAtomMix w a b) x = meanShift k (twoAtomMix w' a b) x := by
  obtain ⟨hw0, hw1⟩ := hw
  obtain ⟨hw'0, hw'1⟩ := hw'
  -- either atom `a` or atom `b` is out of the kernel's range at `x`
  have hcase : k x a = 0 ∨ k x b = 0 := by
    by_contra hcon
    rw [not_or] at hcon
    obtain ⟨ha0, hb0⟩ := hcon
    have hxa : ‖x - a‖ < ρ := lt_of_not_ge (fun hle => ha0 (hk x a hle))
    have hxb : ‖x - b‖ < ρ := lt_of_not_ge (fun hle => hb0 (hk x b hle))
    have hsum : ‖a - b‖ < 2 * ρ := by
      calc ‖a - b‖ = ‖(x - b) - (x - a)‖ := by rw [sub_sub_sub_cancel_left]
        _ ≤ ‖x - b‖ + ‖x - a‖ := norm_sub_le _ _
        _ < ρ + ρ := add_lt_add hxb hxa
        _ = 2 * ρ := by ring
    linarith
  rw [meanShift_twoAtomMix k w hw0.le hw1.le,
    meanShift_twoAtomMix k w' hw'0.le hw'1.le]
  rcases hcase with hka | hkb
  · -- atom `a` inactive: only the `b` term survives
    rw [hka]
    simp only [mul_zero, zero_smul, zero_add]
    have h1 := mul_kernel_eq_zero_iff (α := k x b)
      (show (1 - w) ≠ 0 from (show (0:ℝ) < 1 - w by linarith).ne')
    have h2 := mul_kernel_eq_zero_iff (α := k x b)
      (show (1 - w') ≠ 0 from (show (0:ℝ) < 1 - w' by linarith).ne')
    exact inv_smul_smul_eq_of_iff (b - x) (h1.trans h2.symm)
  · -- atom `b` inactive: only the `a` term survives
    rw [hkb]
    simp only [mul_zero, zero_smul, add_zero]
    have h1 := mul_kernel_eq_zero_iff (α := k x a) hw0.ne'
    have h2 := mul_kernel_eq_zero_iff (α := k x a) hw'0.ne'
    exact inv_smul_smul_eq_of_iff (a - x) (h1.trans h2.symm)

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] in
/-- The two mixtures are genuinely different laws: they disagree on `{a}`. -/
theorem twoAtomMix_ne {a b : E} (hab : a ≠ b) {w w' : ℝ}
    (hw0 : 0 ≤ w) (hw1 : w ≤ 1) (hw'0 : 0 ≤ w') (hw'1 : w' ≤ 1)
    (hne : w ≠ w') :
    twoAtomMix w a b ≠ twoAtomMix w' a b := by
  have hmass : ∀ v : ℝ, 0 ≤ v → v ≤ 1 →
      (twoAtomMix v a b) {a} = ENNReal.ofReal v := by
    intro v hv0 hv1
    rw [twoAtomMix, Measure.add_apply, Measure.smul_apply, Measure.smul_apply,
      smul_eq_mul, smul_eq_mul,
      Measure.dirac_apply_of_mem (mem_singleton a),
      Measure.dirac_apply' _ (measurableSet_singleton a),
      Set.indicator_of_notMem (by simpa [eq_comm] using hab) _]
    simp
  intro heq
  have hcontra : (twoAtomMix w a b) {a} = (twoAtomMix w' a b) {a} := by rw [heq]
  rw [hmass w hw0 hw1, hmass w' hw'0 hw'1] at hcontra
  have := congrArg ENNReal.toReal hcontra
  rw [ENNReal.toReal_ofReal hw0, ENNReal.toReal_ofReal hw'0] at this
  exact hne this

/-- **A2 / D1 — finite-range kernels do not identify.**  For any finite-range
kernel, there exist two distinct probability measures with identically zero
mean-shift drift: the zero-drift converse fails, in every dimension, without
any regularity beyond finite range.  The tail of the Laplace kernel is
load-bearing. -/
theorem finiteRangeKernel_zeroDrift_not_identifies {k : E → E → ℝ} {ρ : ℝ}
    (hρ : 0 < ρ) (hk : FiniteRangeKernel k ρ) {a b : E} (hab : 2 * ρ ≤ ‖a - b‖)
    {w w' : ℝ} (hw : w ∈ Ioo (0 : ℝ) 1) (hw' : w' ∈ Ioo (0 : ℝ) 1)
    (hne : w ≠ w') :
    IsProbabilityMeasure (twoAtomMix w a b) ∧
      IsProbabilityMeasure (twoAtomMix w' a b) ∧
      ZeroDrift (meanShiftDrift k) (twoAtomMix w a b) (twoAtomMix w' a b) ∧
      twoAtomMix w a b ≠ twoAtomMix w' a b := by
  obtain ⟨hw0, hw1⟩ := hw
  obtain ⟨hw'0, hw'1⟩ := hw'
  have hab_ne : a ≠ b := by
    intro h
    rw [h, sub_self, norm_zero] at hab
    linarith
  refine ⟨twoAtomMix_isProbabilityMeasure hw0.le hw1.le a b,
    twoAtomMix_isProbabilityMeasure hw'0.le hw'1.le a b, ?_,
    twoAtomMix_ne hab_ne hw0.le hw1.le hw'0.le hw'1.le hne⟩
  intro x
  rw [meanShiftDrift, meanShift_twoAtomMix_eq_of_separated hk hab
    ⟨hw0, hw1⟩ ⟨hw'0, hw'1⟩ x, sub_self]

end DriftingIdentifiability
