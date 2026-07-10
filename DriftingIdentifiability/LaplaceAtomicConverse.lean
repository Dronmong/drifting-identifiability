import DriftingIdentifiability.LaplaceInjectivity
import Mathlib.Algebra.Polynomial.Roots

/-!
# The atomic Laplace converse: zero drift identifies finite mixtures

Stage 3b of `LaplaceArbitraryConverse.md` — the open arbitrary-target
converse for the paper's practical Laplace kernel, RESOLVED on the class of
finitely-supported probability measures on the line:

> for every positive bandwidth, pointwise zero raw mean-shift drift between
> any two finitely-supported probability measures forces the measures to be
> equal — arbitrary atoms, arbitrary support size, no frame conditions,
> no probe choices, no moment or separation hypotheses.

This is the first arbitrary-PAIR converse content for the practical kernel
(the Stage-2 Dirac rigidity had one degenerate side), on exactly the finite
representation class of the paper's own Appendix C.

Mechanism (the **moment-parallelism argument**): zero drift is the bilinear
identity `Σᵢⱼ aᵢbⱼ(zᵢ-zⱼ)kτ(x,zᵢ)kτ(x,zⱼ) = 0`.  Between consecutive atoms
each kernel factor is a one-sided exponential, so the identity becomes a
quadratic in `u = exp(x/τ)²` with constant coefficients; a real quadratic
vanishing on an infinite set of `u`-values is the zero polynomial, so its
constant coefficient

`𝔞ₖ = Σ_{i,j≤k} (aᵢ e^{zᵢ/τ})(bⱼ e^{zⱼ/τ})(zᵢ-zⱼ)`

vanishes at EVERY truncation `k`.  A strictly-signed-sum argument matches
the bottom atoms, and a telescoping induction — `𝔞_{m+1}` collapses to
`(β_{m+1} - λα_{m+1})` times a strictly negative factor — forces
`b = λ·a`, with normalization pinning `λ = 1`.  Axiom-free.
-/

open MeasureTheory Set Filter
open scoped BigOperators

namespace DriftingIdentifiability

open Paper

/-! ## Atomic measures and their integrals -/

/-- The finitely-supported measure `Σᵢ aᵢ·δ_{zᵢ}`. -/
noncomputable def atomicMeasure {n : ℕ} (z : Fin (n + 1) → ℝ)
    (a : Fin (n + 1) → ℝ) : Measure ℝ :=
  ∑ i, ENNReal.ofReal (a i) • Measure.dirac (z i)

private lemma integrable_continuous_dirac {f : ℝ → ℝ} (hf : Continuous f)
    (c : ℝ) : Integrable f (Measure.dirac c) := by
  refine ⟨hf.aestronglyMeasurable, ?_⟩
  rw [MeasureTheory.hasFiniteIntegral_iff_enorm, lintegral_dirac]
  exact enorm_lt_top

/-- Integrals of continuous functions against an atomic measure are weighted
sums of point values. -/
private lemma integral_atomicMeasure {n : ℕ} (z a : Fin (n + 1) → ℝ)
    (ha : ∀ i, 0 ≤ a i) {f : ℝ → ℝ} (hf : Continuous f) :
    ∫ y, f y ∂(atomicMeasure z a) = ∑ i, a i * f (z i) := by
  unfold atomicMeasure
  rw [integral_finsetSum_measure fun i _ =>
    (integrable_continuous_dirac hf (z i)).smul_measure ENNReal.ofReal_ne_top]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [integral_smul_measure, integral_dirac, ENNReal.toReal_ofReal (ha i),
    smul_eq_mul]

private lemma isProbability_atomicMeasure {n : ℕ} (z a : Fin (n + 1) → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hsa : ∑ i, a i = 1) :
    IsProbabilityMeasure (atomicMeasure z a) := by
  constructor
  unfold atomicMeasure
  rw [Measure.coe_finsetSum, Finset.sum_apply]
  have h1 : ∀ i : Fin (n + 1),
      (ENNReal.ofReal (a i) • Measure.dirac (z i)) Set.univ =
        ENNReal.ofReal (a i) := by
    intro i
    rw [Measure.smul_apply, smul_eq_mul, measure_univ, mul_one]
  rw [Finset.sum_congr rfl fun i _ => h1 i,
    ← ENNReal.ofReal_sum_of_nonneg fun i _ => ha i, hsa, ENNReal.ofReal_one]

/-! ## The bilinear zero-drift identity for atomic measures -/

/-- Zero raw Laplace drift between two atomic measures gives the pointwise
bilinear identity `Σᵢⱼ aᵢbⱼ(zᵢ-zⱼ)k(x,zᵢ)k(x,zⱼ) = 0`. -/
private lemma atomic_bilinear_identity
    (τ : ℝ) (hτ : ValidBandwidth τ) {n : ℕ}
    (z a b : Fin (n + 1) → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 ≤ b i)
    (hsa : ∑ i, a i = 1) (hsb : ∑ i, b i = 1)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (atomicMeasure z a) (atomicMeasure z b)) (x : ℝ) :
    ∑ i, ∑ j, a i * b j * (z i - z j) *
      (laplaceKernel τ x (z i) * laplaceKernel τ x (z j)) = 0 := by
  haveI := isProbability_atomicMeasure z a ha hsa
  haveI := isProbability_atomicMeasure z b hb hsb
  have hcross := (laplaceZeroDrift_iff_crossDisplacement τ hτ
    (atomicMeasure z a) (atomicMeasure z b)).mp hzero x
  simp only [smul_eq_mul] at hcross
  have hkcont : Continuous fun y : ℝ => laplaceKernel τ x y := by
    unfold laplaceKernel
    fun_prop
  have hdcont : Continuous fun y : ℝ => laplaceWeightedDisplacement τ x y := by
    unfold laplaceWeightedDisplacement laplaceKernel
    fun_prop
  unfold kernelNormalizer at hcross
  rw [integral_atomicMeasure z a ha hkcont,
    integral_atomicMeasure z b hb hkcont,
    integral_atomicMeasure z a ha hdcont,
    integral_atomicMeasure z b hb hdcont] at hcross
  simp only [laplaceWeightedDisplacement, smul_eq_mul] at hcross
  have hgoal : (∑ i, ∑ j, a i * b j * (z i - z j) *
      (laplaceKernel τ x (z i) * laplaceKernel τ x (z j))) =
      (∑ i, a i * (laplaceKernel τ x (z i) * (z i - x))) *
          (∑ j, b j * laplaceKernel τ x (z j)) -
        (∑ i, a i * laplaceKernel τ x (z i)) *
          (∑ j, b j * (laplaceKernel τ x (z j) * (z j - x))) := by
    rw [Finset.sum_mul_sum, Finset.sum_mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    ring
  rw [hgoal, mul_comm (∑ i, a i * (laplaceKernel τ x (z i) * (z i - x)))
    (∑ j, b j * laplaceKernel τ x (z j)), hcross, sub_self]

/-! ## Interval decomposition and coefficient extraction -/

/-- A real quadratic vanishing on an infinite set is the zero polynomial. -/
private lemma quadratic_vanish {s : Set ℝ} (hs : s.Infinite)
    {c₀ c₁ c₂ : ℝ} (h : ∀ u ∈ s, c₀ + c₁ * u + c₂ * u ^ 2 = 0) :
    c₀ = 0 ∧ c₁ = 0 ∧ c₂ = 0 := by
  set P : Polynomial ℝ :=
    Polynomial.C c₀ + Polynomial.C c₁ * Polynomial.X +
      Polynomial.C c₂ * Polynomial.X ^ 2 with hP
  have hPzero : P = 0 := by
    apply Polynomial.eq_zero_of_infinite_isRoot
    apply hs.mono
    intro u hu
    have h0 := h u hu
    simp only [Set.mem_setOf_eq, Polynomial.IsRoot, hP, Polynomial.eval_add,
      Polynomial.eval_mul, Polynomial.eval_pow, Polynomial.eval_C,
      Polynomial.eval_X]
    linarith
  refine ⟨?_, ?_, ?_⟩
  · have := congrArg (fun Q : Polynomial ℝ => Q.coeff 0) hPzero
    simpa [hP] using this
  · have := congrArg (fun Q : Polynomial ℝ => Q.coeff 1) hPzero
    simpa [hP] using this
  · have := congrArg (fun Q : Polynomial ℝ => Q.coeff 2) hPzero
    simpa [hP] using this

/-- The truncated antisymmetric tilted-moment pairing `𝔞ₖ`. -/
private noncomputable def frakA (τ : ℝ) {n : ℕ} (z a b : Fin (n + 1) → ℝ)
    (k : Fin (n + 1)) : ℝ :=
  ∑ i ∈ Finset.univ.filter (· ≤ k), ∑ j ∈ Finset.univ.filter (· ≤ k),
    (a i * Real.exp (z i / τ)) * (b j * Real.exp (z j / τ)) * (z i - z j)

/-- The `u`-coefficient of the interval decomposition (its value is never
used, only its constancy in `x`). -/
private noncomputable def frakB (τ : ℝ) {n : ℕ} (z a b : Fin (n + 1) → ℝ)
    (k : Fin (n + 1)) : ℝ :=
  ∑ i, ∑ j,
    (if i ≤ k then (if j ≤ k then 0 else
        a i * b j * (z i - z j) *
          (Real.exp (z i / τ) / Real.exp (z j / τ)))
      else (if j ≤ k then
        a i * b j * (z i - z j) *
          (Real.exp (z j / τ) / Real.exp (z i / τ)) else 0))

/-- The `u²`-coefficient of the interval decomposition. -/
private noncomputable def frakC (τ : ℝ) {n : ℕ} (z a b : Fin (n + 1) → ℝ)
    (k : Fin (n + 1)) : ℝ :=
  ∑ i, ∑ j,
    (if i ≤ k then 0 else (if j ≤ k then 0 else
      a i * b j * (z i - z j) *
        (Real.exp (z i / τ) * Real.exp (z j / τ))⁻¹))

/-- `𝔞ₖ` as a full double sum with an indicator. -/
private lemma frakA_eq_ite (τ : ℝ) {n : ℕ} (z a b : Fin (n + 1) → ℝ)
    (k : Fin (n + 1)) :
    frakA τ z a b k = ∑ i, ∑ j,
      (if i ≤ k then (if j ≤ k then
          (a i * Real.exp (z i / τ)) * (b j * Real.exp (z j / τ)) *
            (z i - z j) else 0) else 0) := by
  unfold frakA
  rw [Finset.sum_filter]
  refine Finset.sum_congr rfl fun i _ => ?_
  by_cases hi : i ≤ k
  · simp only [hi, if_true]
    rw [Finset.sum_filter]
  · simp only [hi, if_false]
    rw [Finset.sum_const_zero]

/-- The open interval strictly between atom `k` and the next atom (or one
unit above the last atom). -/
private def intervalTop {n : ℕ} (z : Fin (n + 1) → ℝ) (k : Fin (n + 1)) : ℝ :=
  if h : (k : ℕ) + 1 < n + 1 then z ⟨(k : ℕ) + 1, h⟩ else z k + 1

private lemma lt_intervalTop {n : ℕ} {z : Fin (n + 1) → ℝ}
    (hz : StrictMono z) (k : Fin (n + 1)) : z k < intervalTop z k := by
  unfold intervalTop
  split
  · apply hz
    rw [Fin.lt_def]
    simp
  · linarith

private lemma intervalTop_le {n : ℕ} {z : Fin (n + 1) → ℝ}
    (hz : StrictMono z) {k i : Fin (n + 1)} (hik : k < i) :
    intervalTop z k ≤ z i := by
  unfold intervalTop
  split
  case isTrue h =>
    apply hz.monotone
    rw [Fin.le_def]
    have : (k : ℕ) < (i : ℕ) := hik
    simpa using this
  case isFalse h =>
    exfalso
    have h1 := i.isLt
    have h2 : (k : ℕ) < (i : ℕ) := hik
    omega

private lemma kernel_left {τ : ℝ} {x c : ℝ} (hcx : c < x) :
    laplaceKernel τ x c = Real.exp (c / τ) / Real.exp (x / τ) := by
  unfold laplaceKernel
  rw [Real.norm_eq_abs, abs_of_pos (sub_pos.mpr hcx),
    show -(1 / τ) * (x - c) = c / τ - x / τ by ring, Real.exp_sub]

private lemma kernel_right {τ : ℝ} {x c : ℝ} (hxc : x < c) :
    laplaceKernel τ x c = Real.exp (x / τ) / Real.exp (c / τ) := by
  unfold laplaceKernel
  rw [Real.norm_eq_abs, abs_of_neg (sub_neg.mpr hxc),
    show -(1 / τ) * -(x - c) = x / τ - c / τ by ring, Real.exp_sub]

/-- Pointwise polynomial form of one bilinear term on the `k`-th interval. -/
private lemma term_poly_form (τ : ℝ) {n : ℕ}
    (z a b : Fin (n + 1) → ℝ) (k : Fin (n + 1)) {x : ℝ}
    (hleft : ∀ i : Fin (n + 1), i ≤ k → z i < x)
    (hright : ∀ i : Fin (n + 1), ¬ i ≤ k → x < z i)
    (i j : Fin (n + 1)) :
    a i * b j * (z i - z j) *
        (laplaceKernel τ x (z i) * laplaceKernel τ x (z j)) *
      Real.exp (x / τ) ^ 2 =
    (if i ≤ k then (if j ≤ k then
        (a i * Real.exp (z i / τ)) * (b j * Real.exp (z j / τ)) *
          (z i - z j) else 0) else 0) +
    (if i ≤ k then (if j ≤ k then 0 else
        a i * b j * (z i - z j) *
          (Real.exp (z i / τ) / Real.exp (z j / τ)))
      else (if j ≤ k then
        a i * b j * (z i - z j) *
          (Real.exp (z j / τ) / Real.exp (z i / τ)) else 0)) *
      Real.exp (x / τ) ^ 2 +
    (if i ≤ k then 0 else (if j ≤ k then 0 else
        a i * b j * (z i - z j) *
          (Real.exp (z i / τ) * Real.exp (z j / τ))⁻¹)) *
      (Real.exp (x / τ) ^ 2) ^ 2 := by
  have hx : Real.exp (x / τ) ≠ 0 := Real.exp_ne_zero _
  have hzi : Real.exp (z i / τ) ≠ 0 := Real.exp_ne_zero _
  have hzj : Real.exp (z j / τ) ≠ 0 := Real.exp_ne_zero _
  by_cases hi : i ≤ k <;> by_cases hj : j ≤ k
  · rw [kernel_left (hleft i hi), kernel_left (hleft j hj)]
    simp only [hi, hj, if_true]
    field_simp
    ring
  · rw [kernel_left (hleft i hi), kernel_right (hright j hj)]
    simp only [hi, hj, if_true, if_false]
    field_simp
    ring
  · rw [kernel_right (hright i hi), kernel_left (hleft j hj)]
    simp only [hi, hj, if_true, if_false]
    field_simp
    ring
  · rw [kernel_right (hright i hi), kernel_right (hright j hj)]
    simp only [hi, hj, if_false]
    field_simp
    ring

/-- **Coefficient extraction:** the truncated pairing `𝔞ₖ` vanishes at every
truncation `k`. -/
private lemma frakA_eq_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) {n : ℕ}
    (z a b : Fin (n + 1) → ℝ) (hz : StrictMono z)
    (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 ≤ b i)
    (hsa : ∑ i, a i = 1) (hsb : ∑ i, b i = 1)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (atomicMeasure z a) (atomicMeasure z b))
    (k : Fin (n + 1)) : frakA τ z a b k = 0 := by
  have hτ0 : 0 < τ := hτ
  -- the quadratic identity on the k-th interval, in `u = exp(x/τ)²`
  have hquad : ∀ x ∈ Ioo (z k) (intervalTop z k),
      frakA τ z a b k + frakB τ z a b k * Real.exp (x / τ) ^ 2 +
        frakC τ z a b k * (Real.exp (x / τ) ^ 2) ^ 2 = 0 := by
    intro x hx
    have hleft : ∀ i : Fin (n + 1), i ≤ k → z i < x :=
      fun i hik => lt_of_le_of_lt (hz.monotone hik) hx.1
    have hright : ∀ i : Fin (n + 1), ¬ i ≤ k → x < z i :=
      fun i hik => lt_of_lt_of_le hx.2 (intervalTop_le hz (not_le.mp hik))
    have hbil := atomic_bilinear_identity τ hτ z a b ha hb hsa hsb hzero x
    have hmul := congrArg (fun r => r * Real.exp (x / τ) ^ 2) hbil
    simp only [zero_mul] at hmul
    rw [Finset.sum_mul] at hmul
    calc frakA τ z a b k + frakB τ z a b k * Real.exp (x / τ) ^ 2 +
          frakC τ z a b k * (Real.exp (x / τ) ^ 2) ^ 2
        = ∑ i, ∑ j, (a i * b j * (z i - z j) *
            (laplaceKernel τ x (z i) * laplaceKernel τ x (z j)) *
              Real.exp (x / τ) ^ 2) := by
          rw [frakA_eq_ite]
          unfold frakB frakC
          rw [Finset.sum_mul, Finset.sum_mul,
            ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.sum_mul, Finset.sum_mul,
            ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
          refine Finset.sum_congr rfl fun j _ => ?_
          exact (term_poly_form τ z a b k hleft hright i j).symm
      _ = 0 := by
          rw [← hmul]
          refine Finset.sum_congr rfl fun i _ => ?_
          rw [Finset.sum_mul]
  -- a quadratic vanishing on the image of a nondegenerate interval is zero
  have hmono : StrictMonoOn (fun t : ℝ => Real.exp (t / τ) ^ 2)
      (Ioo (z k) (intervalTop z k)) := by
    intro s _ t _ hst
    have h1 : Real.exp (s / τ) < Real.exp (t / τ) := by
      apply Real.exp_lt_exp.mpr
      gcongr
    have h2 := Real.exp_pos (s / τ)
    have h3 := Real.exp_pos (t / τ)
    simp only
    nlinarith
  have hinf : ((fun x : ℝ => Real.exp (x / τ) ^ 2) ''
      Ioo (z k) (intervalTop z k)).Infinite :=
    Set.Infinite.image hmono.injOn
      (Set.infinite_coe_iff.mp (Set.Ioo.infinite (lt_intervalTop hz k)))
  have hpoly := quadratic_vanish hinf (c₀ := frakA τ z a b k)
    (c₁ := frakB τ z a b k) (c₂ := frakC τ z a b k) ?_
  · exact hpoly.1
  · rintro u ⟨x, hx, rfl⟩
    exact hquad x hx

/-! ## From vanishing pairings to equal weights -/

private lemma antisym_double_sum {n : ℕ} (s : Finset (Fin (n + 1)))
    (c : Fin (n + 1) → ℝ) (z : Fin (n + 1) → ℝ) :
    ∑ i ∈ s, ∑ j ∈ s, c i * c j * (z i - z j) = 0 := by
  have h : (∑ i ∈ s, ∑ j ∈ s, c i * c j * (z i - z j)) =
      (∑ i ∈ s, c i * z i) * (∑ j ∈ s, c j) -
        (∑ i ∈ s, c i) * (∑ j ∈ s, c j * z j) := by
    rw [Finset.sum_mul_sum, Finset.sum_mul_sum, ← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [← Finset.sum_sub_distrib]
    refine Finset.sum_congr rfl fun j _ => ?_
    ring
  rw [h]
  ring

private lemma frakA_swap (τ : ℝ) {n : ℕ} (z a b : Fin (n + 1) → ℝ)
    (k : Fin (n + 1)) : frakA τ z b a k = -frakA τ z a b k := by
  unfold frakA
  rw [Finset.sum_comm, ← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun i _ => ?_
  rw [← Finset.sum_neg_distrib]
  refine Finset.sum_congr rfl fun j _ => ?_
  ring

/-- If the truncated pairings all vanish and `q` has mass at the bottom atom,
so does `p`: otherwise the pairing at the least `p`-atom is a strictly
positive sum. -/
private lemma bottom_pos_aux (τ : ℝ) {n : ℕ}
    (z a b : Fin (n + 1) → ℝ) (hz : StrictMono z)
    (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 ≤ b i) (hsa : ∑ i, a i = 1)
    (hA : ∀ k, frakA τ z a b k = 0) (hb0 : 0 < b 0) : 0 < a 0 := by
  by_contra ha0
  have ha0' : a 0 = 0 := le_antisymm (not_lt.mp ha0) (ha 0)
  have hne : (Finset.univ.filter fun i => 0 < a i).Nonempty := by
    by_contra hemp
    rw [Finset.not_nonempty_iff_eq_empty] at hemp
    have hzero : ∑ i, a i = 0 := by
      apply Finset.sum_eq_zero
      intro i _
      by_contra hai
      have hpos : 0 < a i := lt_of_le_of_ne (ha i) (Ne.symm hai)
      have hmem : i ∈ Finset.univ.filter fun i => 0 < a i := by
        simp [hpos]
      rw [hemp] at hmem
      exact absurd hmem (Finset.notMem_empty i)
    rw [hsa] at hzero
    exact one_ne_zero hzero
  set K := (Finset.univ.filter fun i => 0 < a i).min' hne with hK
  have hKpos : 0 < a K := by
    have hmem := (Finset.univ.filter fun i => 0 < a i).min'_mem hne
    rw [← hK] at hmem
    exact (Finset.mem_filter.mp hmem).2
  have hK0 : K ≠ 0 := by
    intro h
    rw [h] at hKpos
    rw [ha0'] at hKpos
    exact lt_irrefl 0 hKpos
  have hKzero : ∀ j, j < K → a j = 0 := by
    intro j hj
    by_contra haj
    have hpos : 0 < a j := lt_of_le_of_ne (ha j) (Ne.symm haj)
    have hle : K ≤ j := Finset.min'_le _ _ (by simp [hpos])
    exact absurd hle (not_le.mpr hj)
  have hcollapse : frakA τ z a b K =
      ∑ j ∈ Finset.univ.filter (· ≤ K),
        (a K * Real.exp (z K / τ)) *
          ((b j * Real.exp (z j / τ)) * (z K - z j)) := by
    unfold frakA
    rw [Finset.sum_eq_single_of_mem K (by simp)]
    · refine Finset.sum_congr rfl fun j _ => ?_
      ring
    · intro i hi hiK
      have hile : i ≤ K := (Finset.mem_filter.mp hi).2
      have hilt : i < K := lt_of_le_of_ne hile hiK
      rw [hKzero i hilt]
      simp
  have hpos : 0 < ∑ j ∈ Finset.univ.filter (· ≤ K),
      (b j * Real.exp (z j / τ)) * (z K - z j) := by
    apply Finset.sum_pos'
    · intro j hj
      have hjK : j ≤ K := (Finset.mem_filter.mp hj).2
      have hzj : z j ≤ z K := hz.monotone hjK
      have h1 : 0 ≤ b j * Real.exp (z j / τ) :=
        mul_nonneg (hb j) (Real.exp_pos _).le
      exact mul_nonneg h1 (sub_nonneg.mpr hzj)
    · refine ⟨0, by simp, ?_⟩
      have h0K : (0 : Fin (n + 1)) < K := Fin.pos_of_ne_zero hK0
      have hz0 : z 0 < z K := hz h0K
      have h1 : 0 < b 0 * Real.exp (z 0 / τ) :=
        mul_pos hb0 (Real.exp_pos _)
      exact mul_pos h1 (sub_pos.mpr hz0)
  have hA' := hA K
  rw [hcollapse, ← Finset.mul_sum] at hA'
  have haKe : 0 < a K * Real.exp (z K / τ) := mul_pos hKpos (Real.exp_pos _)
  exact absurd hA' (ne_of_gt (mul_pos haKe hpos))

/-- **Weight rigidity.**  The vanishing of every truncated pairing forces the
weight vectors to be proportional, and normalization makes them equal. -/
private lemma weights_eq_of_frakA
    (τ : ℝ) {n : ℕ}
    (z a b : Fin (n + 1) → ℝ) (hz : StrictMono z)
    (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 ≤ b i)
    (hsa : ∑ i, a i = 1) (hsb : ∑ i, b i = 1)
    (hsupp : ∀ i, 0 < a i ∨ 0 < b i)
    (hA : ∀ k, frakA τ z a b k = 0) : a = b := by
  have hA' : ∀ k, frakA τ z b a k = 0 := by
    intro k
    rw [frakA_swap, hA k, neg_zero]
  -- both bottom atoms carry mass
  obtain ⟨ha0, hb0⟩ : 0 < a 0 ∧ 0 < b 0 := by
    rcases hsupp 0 with h | h
    · exact ⟨h, bottom_pos_aux τ z b a hz hb ha hsb hA' h⟩
    · exact ⟨bottom_pos_aux τ z a b hz ha hb hsa hA h, h⟩
  set lam := b 0 / a 0 with hlam
  -- proportionality by strong induction on the index
  have hprop : ∀ N : ℕ, ∀ m : Fin (n + 1), (m : ℕ) ≤ N → b m = lam * a m := by
    intro N
    induction N with
    | zero =>
      intro m hm
      have hm0 : m = 0 := Fin.ext (Nat.le_zero.mp hm)
      rw [hm0, hlam]
      field_simp
    | succ N ih =>
      intro m hm
      by_cases hmN : (m : ℕ) ≤ N
      · exact ih m hmN
      · have hm0 : m ≠ 0 := by
          intro h
          rw [h] at hmN
          simp at hmN
        -- substitute the inductive hypothesis into the pairing at `m`
        have hsubst : ∀ j : Fin (n + 1), j ≤ m →
            b j = lam * a j + (if j = m then b m - lam * a m else 0) := by
          intro j hjm
          by_cases hjm' : j = m
          · subst hjm'
            rw [if_pos rfl]
            ring
          · rw [if_neg hjm']
            have hjlt : (j : ℕ) < (m : ℕ) :=
              Fin.lt_def.mp (lt_of_le_of_ne hjm hjm')
            rw [ih j (by omega)]
            ring
        have hexpand : frakA τ z a b m =
            (b m - lam * a m) * Real.exp (z m / τ) *
              ∑ i ∈ Finset.univ.filter (· ≤ m),
                (a i * Real.exp (z i / τ)) * (z i - z m) := by
          unfold frakA
          calc ∑ i ∈ Finset.univ.filter (· ≤ m),
                ∑ j ∈ Finset.univ.filter (· ≤ m),
                  (a i * Real.exp (z i / τ)) * (b j * Real.exp (z j / τ)) *
                    (z i - z j)
              = ∑ i ∈ Finset.univ.filter (· ≤ m),
                ∑ j ∈ Finset.univ.filter (· ≤ m),
                  (lam * ((a i * Real.exp (z i / τ)) *
                      (a j * Real.exp (z j / τ)) * (z i - z j)) +
                    (if j = m then (a i * Real.exp (z i / τ)) *
                      ((b m - lam * a m) * Real.exp (z m / τ)) *
                        (z i - z m) else 0)) := by
                refine Finset.sum_congr rfl fun i _ => ?_
                refine Finset.sum_congr rfl fun j hj => ?_
                have hjm : j ≤ m := (Finset.mem_filter.mp hj).2
                rw [hsubst j hjm]
                by_cases hjm' : j = m
                · rw [if_pos hjm', if_pos hjm', hjm']
                  ring
                · rw [if_neg hjm', if_neg hjm']
                  ring
            _ = lam * (∑ i ∈ Finset.univ.filter (· ≤ m),
                  ∑ j ∈ Finset.univ.filter (· ≤ m),
                    (a i * Real.exp (z i / τ)) *
                      (a j * Real.exp (z j / τ)) * (z i - z j)) +
                ∑ i ∈ Finset.univ.filter (· ≤ m),
                  (a i * Real.exp (z i / τ)) *
                    ((b m - lam * a m) * Real.exp (z m / τ)) * (z i - z m) := by
                rw [Finset.mul_sum, ← Finset.sum_add_distrib]
                refine Finset.sum_congr rfl fun i _ => ?_
                rw [Finset.mul_sum, Finset.sum_add_distrib,
                  Finset.sum_ite_eq' (Finset.univ.filter (· ≤ m)) m]
                simp
            _ = (b m - lam * a m) * Real.exp (z m / τ) *
                ∑ i ∈ Finset.univ.filter (· ≤ m),
                  (a i * Real.exp (z i / τ)) * (z i - z m) := by
                rw [antisym_double_sum, mul_zero, zero_add, Finset.mul_sum]
                refine Finset.sum_congr rfl fun i _ => ?_
                ring
        -- the coefficient sum is strictly negative
        have hneg : ∑ i ∈ Finset.univ.filter (· ≤ m),
            (a i * Real.exp (z i / τ)) * (z i - z m) < 0 := by
          have hposneg : 0 < ∑ i ∈ Finset.univ.filter (· ≤ m),
              (a i * Real.exp (z i / τ)) * (z m - z i) := by
            apply Finset.sum_pos'
            · intro i hi
              have him : i ≤ m := (Finset.mem_filter.mp hi).2
              exact mul_nonneg (mul_nonneg (ha i) (Real.exp_pos _).le)
                (sub_nonneg.mpr (hz.monotone him))
            · refine ⟨0, by simp, ?_⟩
              have h0m : (0 : Fin (n + 1)) < m := Fin.pos_of_ne_zero hm0
              exact mul_pos (mul_pos ha0 (Real.exp_pos _))
                (sub_pos.mpr (hz h0m))
          have hflip : ∑ i ∈ Finset.univ.filter (· ≤ m),
              (a i * Real.exp (z i / τ)) * (z i - z m) =
              -∑ i ∈ Finset.univ.filter (· ≤ m),
                (a i * Real.exp (z i / τ)) * (z m - z i) := by
            rw [← Finset.sum_neg_distrib]
            refine Finset.sum_congr rfl fun i _ => ?_
            ring
          rw [hflip]
          linarith
        have hzero' := hA m
        rw [hexpand] at hzero'
        have hfactor : b m - lam * a m = 0 := by
          rcases mul_eq_zero.mp hzero' with h | h
          · rcases mul_eq_zero.mp h with h' | h'
            · exact h'
            · exact absurd h' (Real.exp_ne_zero _)
          · exact absurd h (ne_of_lt hneg)
        linarith
  have hall : ∀ i : Fin (n + 1), b i = lam * a i :=
    fun i => hprop (i : ℕ) i le_rfl
  have hlam1 : lam = 1 := by
    have h1 : ∑ i, b i = lam * ∑ i, a i := by
      rw [Finset.mul_sum]
      exact Finset.sum_congr rfl fun i _ => hall i
    rw [hsa, hsb, mul_one] at h1
    exact h1.symm
  funext i
  rw [hall i, hlam1, one_mul]

/-! ## The headline theorems -/

/-- **The atomic Laplace converse (weights form).**  Zero raw Laplace
mean-shift drift between two finitely-supported probability measures on a
common refined support forces the weight vectors to coincide.  Arbitrary
atoms, arbitrary support size, no moment or separation hypotheses;
axiom-free. -/
theorem laplaceZeroDrift_atomic_weights_eq
    (τ : ℝ) (hτ : ValidBandwidth τ) {n : ℕ}
    (z : Fin (n + 1) → ℝ) (hz : StrictMono z)
    (a b : Fin (n + 1) → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 ≤ b i)
    (hsa : ∑ i, a i = 1) (hsb : ∑ i, b i = 1)
    (hsupp : ∀ i, 0 < a i ∨ 0 < b i)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (atomicMeasure z a) (atomicMeasure z b)) :
    a = b :=
  weights_eq_of_frakA τ z a b hz ha hb hsa hsb hsupp
    (frakA_eq_zero τ hτ z a b hz ha hb hsa hsb hzero)

/-- **The atomic Laplace converse.**  For the paper's practical Laplace
kernel, pointwise zero raw mean-shift drift identifies finitely-supported
probability measures on the line: the open arbitrary-target converse holds
on the entire finite-mixture class — the representation class of the paper's
own Appendix C — with no frame conditions, probe choices, moments, or
bandwidth restrictions.  Axiom-free. -/
theorem laplaceZeroDrift_atomic_identifies
    (τ : ℝ) (hτ : ValidBandwidth τ) {n : ℕ}
    (z : Fin (n + 1) → ℝ) (hz : StrictMono z)
    (a b : Fin (n + 1) → ℝ)
    (ha : ∀ i, 0 ≤ a i) (hb : ∀ i, 0 ≤ b i)
    (hsa : ∑ i, a i = 1) (hsb : ∑ i, b i = 1)
    (hsupp : ∀ i, 0 < a i ∨ 0 < b i)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (atomicMeasure z a) (atomicMeasure z b)) :
    atomicMeasure z a = atomicMeasure z b := by
  rw [laplaceZeroDrift_atomic_weights_eq τ hτ z hz a b ha hb hsa hsb
    hsupp hzero]

end DriftingIdentifiability
