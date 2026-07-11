import DriftingIdentifiability.LaplaceGeneralConverseEndgame
import Mathlib.Algebra.Polynomial.Roots

/-!
# Right-dense-gap converse for the 1-d Laplace field

This module starts Milestone 3 of `LaplaceGeneralConverseRoadmap.md`.

The first layer is purely topological: a right-continuous real-valued function
with zeros arbitrarily close from the right of every point is identically zero.
Applied to the lower truncated pairing `truncatedPairing`, this composes with
the Milestone-2 endgame
`laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero`.

The second layer, added below this bridge, turns zero-mass gaps of `p+q` into
right-dense zeros of the truncated pairing under zero drift.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-- A real quadratic vanishing on an infinite set is the zero polynomial.
Copied from the atomic converse file because that helper is intentionally
private there. -/
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

/-- A real-valued function has zeros arbitrarily close from the right of every
point.  This is the topological core used by the concrete right-dense-gap
hypothesis below. -/
def RightDenseZeros (f : ℝ → ℝ) : Prop :=
  ∀ x ε : ℝ, 0 < ε → ∃ y : ℝ, x < y ∧ y < x + ε ∧ f y = 0

/-- Concrete right-dense zero-mass gaps for the combined measure `p+q`.

This is the Lean-facing version of "the combined support is nowhere dense":
after every point and at every scale there is a nonempty open interval carrying
no `p`- or `q`-mass. -/
def RightDenseZeroMassGaps (p q : Measure ℝ) : Prop :=
  ∀ x ε : ℝ, 0 < ε →
    ∃ u v : ℝ, x < u ∧ u < v ∧ v < x + ε ∧ (p + q) (Set.Ioo u v) = 0

/-- If a function is right-continuous at every point and has zeros arbitrarily
close from the right of every point, then it vanishes everywhere. -/
lemma eq_zero_of_rightDenseZeros_of_continuousWithinAt_Ici
    {f : ℝ → ℝ}
    (hcont : ∀ x : ℝ, ContinuousWithinAt f (Set.Ici x) x)
    (hzero : RightDenseZeros f) :
    ∀ x : ℝ, f x = 0 := by
  intro x
  have hx_closure : x ∈ closure (Set.Ici x ∩ {y : ℝ | f y = 0}) := by
    rw [Metric.mem_closure_iff]
    intro ε hε
    obtain ⟨y, hxy, hyε, hy0⟩ := hzero x ε hε
    refine ⟨y, ⟨⟨le_of_lt hxy, hy0⟩, ?_⟩⟩
    rw [Real.dist_eq]
    have habs : |x - y| = y - x := by
      rw [abs_of_nonpos (sub_nonpos.mpr (le_of_lt hxy))]
      ring
    rw [habs]
    linarith
  have hmap : MapsTo f (Set.Ici x ∩ {y : ℝ | f y = 0}) ({0} : Set ℝ) := by
    intro y hy
    exact hy.2
  have hx_image : f x ∈ closure ({0} : Set ℝ) :=
    ((hcont x).mono (by intro y hy; exact hy.1)).mem_closure hx_closure hmap
  simpa [closure_singleton] using hx_image

/-- Right-dense zeros of the truncated pairing are enough to identify the two
probability measures.  This is the Milestone-3 topological bridge into the
already-certified Milestone-2 endgame. -/
theorem laplaceZeroDrift_identifies_of_rightDense_truncatedPairing_zeros
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : RightDenseZeros (fun x => truncatedPairing τ p q x)) :
    p = q := by
  refine laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero τ hτ p q ?_
  exact eq_zero_of_rightDenseZeros_of_continuousWithinAt_Ici
    (fun x => truncatedPairing_continuousWithinAt_Ici τ hτ p q x) hzero

/-! ## Zero-mass gap bookkeeping -/

/-- Integrals over a set of measure zero vanish.  Kept local to this module
because the gap argument uses it repeatedly. -/
lemma setIntegral_eq_zero_of_measure_zero
    {μ : Measure ℝ} {s : Set ℝ} (f : ℝ → ℝ) (hμ : μ s = 0) :
    ∫ y in s, f y ∂μ = 0 := by
  have hrestrict : μ.restrict s = 0 := Measure.restrict_eq_zero.mpr hμ
  change ∫ y, f y ∂(μ.restrict s) = 0
  rw [hrestrict]
  exact integral_zero_measure f

lemma left_gap_measure_zero_of_sum_gap_zero
    {p q : Measure ℝ} {u v : ℝ} (hgap : (p + q) (Set.Ioo u v) = 0) :
    p (Set.Ioo u v) = 0 := by
  have hsum : p (Set.Ioo u v) + q (Set.Ioo u v) = 0 := by
    simpa [Measure.add_apply] using hgap
  exact (add_eq_zero.mp hsum).1

lemma right_gap_measure_zero_of_sum_gap_zero
    {p q : Measure ℝ} {u v : ℝ} (hgap : (p + q) (Set.Ioo u v) = 0) :
    q (Set.Ioo u v) = 0 := by
  have hsum : p (Set.Ioo u v) + q (Set.Ioo u v) = 0 := by
    simpa [Measure.add_apply] using hgap
  exact (add_eq_zero.mp hsum).2

lemma measure_Ioc_zero_of_gap_zero
    {μ : Measure ℝ} {u v x : ℝ} (hgap : μ (Set.Ioo u v) = 0)
    (_hux : u < x) (hxv : x < v) :
    μ (Set.Ioc u x) = 0 := by
  refine measure_mono_null ?_ hgap
  intro y hy
  exact ⟨hy.1, lt_of_le_of_lt hy.2 hxv⟩

/-- On a zero-mass gap, the lower exponential mass is constant. -/
lemma lowerExpMass_eq_of_gap_zero
    (τ : ℝ) (hτ : 0 < τ) (μ : Measure ℝ) [IsFiniteMeasure μ]
    {u v x : ℝ} (hgap : μ (Set.Ioo u v) = 0) (hux : u < x) (hxv : x < v) :
    lowerExpMass τ μ x = lowerExpMass τ μ u := by
  have hsub := lowerExpMass_sub τ hτ μ (le_of_lt hux)
  have hIoc : μ (Set.Ioc u x) = 0 :=
    measure_Ioc_zero_of_gap_zero hgap hux hxv
  have hzero :
      ∫ y in Set.Ioc u x, Real.exp (y / τ) ∂μ = 0 :=
    setIntegral_eq_zero_of_measure_zero (fun y => Real.exp (y / τ)) hIoc
  rw [hzero] at hsub
  linarith

/-- On a zero-mass gap, the lower compensated moment evolves affinely. -/
lemma lowerCompensatedMoment_eq_of_gap_zero
    (τ : ℝ) (hτ : 0 < τ) (μ : Measure ℝ) [IsFiniteMeasure μ]
    {u v x : ℝ} (hgap : μ (Set.Ioo u v) = 0) (hux : u < x) (hxv : x < v) :
    lowerCompensatedMoment τ μ x =
      lowerCompensatedMoment τ μ u + (x - u) * lowerExpMass τ μ u := by
  have hsub := lowerCompensatedMoment_sub τ hτ μ (le_of_lt hux)
  have hIoc : μ (Set.Ioc u x) = 0 :=
    measure_Ioc_zero_of_gap_zero hgap hux hxv
  have hzero :
      ∫ y in Set.Ioc u x, (x - y) * Real.exp (y / τ) ∂μ = 0 :=
    setIntegral_eq_zero_of_measure_zero
      (fun y => (x - y) * Real.exp (y / τ)) hIoc
  rw [hzero] at hsub
  linarith

private lemma Ioi_eq_Ioc_union_Ioi {u x : ℝ} (hux : u < x) :
    Set.Ioi u = Set.Ioc u x ∪ Set.Ioi x := by
  ext y
  constructor
  · intro hy
    by_cases hyx : y ≤ x
    · exact Or.inl ⟨hy, hyx⟩
    · exact Or.inr (lt_of_not_ge hyx)
  · rintro (hy | hy)
    · exact hy.1
    · exact lt_trans hux hy

private lemma disjoint_Ioc_Ioi_same_right (u x : ℝ) :
    Disjoint (Set.Ioc u x) (Set.Ioi x) := by
  rw [Set.disjoint_left]
  intro y hyIoc hyIoi
  exact not_lt_of_ge hyIoc.2 hyIoi

/-- On a zero-mass gap, the upper exponential mass is constant. -/
lemma upperExpMass_eq_of_gap_zero
    (τ : ℝ) (hτ : 0 < τ) (μ : Measure ℝ) [IsFiniteMeasure μ]
    {u v x : ℝ} (hgap : μ (Set.Ioo u v) = 0) (hux : u < x) (hxv : x < v) :
    upperExpMass τ μ x = upperExpMass τ μ u := by
  have hIoc : μ (Set.Ioc u x) = 0 :=
    measure_Ioc_zero_of_gap_zero hgap hux hxv
  have hzero :
      ∫ y in Set.Ioc u x, Real.exp (-y / τ) ∂μ = 0 :=
    setIntegral_eq_zero_of_measure_zero (fun y => Real.exp (-y / τ)) hIoc
  have hInt : IntegrableOn (fun y : ℝ => Real.exp (-y / τ)) (Set.Ioi u) μ :=
    integrable_upperExpKernel τ hτ μ u
  have hIocInt : IntegrableOn (fun y : ℝ => Real.exp (-y / τ)) (Set.Ioc u x) μ :=
    hInt.mono_set (by intro y hy; exact hy.1)
  have hIoiInt : IntegrableOn (fun y : ℝ => Real.exp (-y / τ)) (Set.Ioi x) μ :=
    hInt.mono_set (by intro y hy; exact lt_trans hux hy)
  have hsplit :
      upperExpMass τ μ u =
        ∫ y in Set.Ioc u x, Real.exp (-y / τ) ∂μ +
          ∫ y in Set.Ioi x, Real.exp (-y / τ) ∂μ := by
    unfold upperExpMass
    rw [Ioi_eq_Ioc_union_Ioi hux,
      setIntegral_union (disjoint_Ioc_Ioi_same_right u x) measurableSet_Ioi hIocInt hIoiInt]
  unfold upperExpMass at hsplit ⊢
  rw [hzero, zero_add] at hsplit
  exact hsplit.symm

/-- On a zero-mass gap, the upper compensated moment evolves affinely with the
opposite sign from the lower compensated moment. -/
lemma upperCompensatedMoment_eq_of_gap_zero
    (τ : ℝ) (hτ : 0 < τ) (μ : Measure ℝ) [IsFiniteMeasure μ]
    {u v x : ℝ} (hgap : μ (Set.Ioo u v) = 0) (hux : u < x) (hxv : x < v) :
    upperCompensatedMoment τ μ x =
      upperCompensatedMoment τ μ u - (x - u) * upperExpMass τ μ u := by
  have hIoc : μ (Set.Ioc u x) = 0 :=
    measure_Ioc_zero_of_gap_zero hgap hux hxv
  have hzero :
      ∫ y in Set.Ioc u x, (y - u) * Real.exp (-y / τ) ∂μ = 0 :=
    setIntegral_eq_zero_of_measure_zero
      (fun y => (y - u) * Real.exp (-y / τ)) hIoc
  have hInt : IntegrableOn (fun y : ℝ => (y - u) * Real.exp (-y / τ)) (Set.Ioi u) μ :=
    integrable_upperCompKernel τ hτ μ u
  have hIocInt :
      IntegrableOn (fun y : ℝ => (y - u) * Real.exp (-y / τ)) (Set.Ioc u x) μ :=
    hInt.mono_set (by intro y hy; exact hy.1)
  have hIoiInt :
      IntegrableOn (fun y : ℝ => (y - u) * Real.exp (-y / τ)) (Set.Ioi x) μ :=
    hInt.mono_set (by intro y hy; exact lt_trans hux hy)
  have hsplit :
      upperCompensatedMoment τ μ u =
        ∫ y in Set.Ioc u x, (y - u) * Real.exp (-y / τ) ∂μ +
          ∫ y in Set.Ioi x, (y - u) * Real.exp (-y / τ) ∂μ := by
    unfold upperCompensatedMoment
    rw [Ioi_eq_Ioc_union_Ioi hux,
      setIntegral_union (disjoint_Ioc_Ioi_same_right u x) measurableSet_Ioi hIocInt hIoiInt]
  have hblock :
      (∫ y in Set.Ioi x, (y - u) * Real.exp (-y / τ) ∂μ)
        = upperCompensatedMoment τ μ x + (x - u) * upperExpMass τ μ x := by
    unfold upperCompensatedMoment upperExpMass
    have hpt : (fun y : ℝ => (y - u) * Real.exp (-y / τ))
        = fun y => (y - x) * Real.exp (-y / τ) + (x - u) * Real.exp (-y / τ) := by
      funext y
      ring
    rw [hpt,
      integral_add (integrable_upperCompKernel τ hτ μ x)
        (Integrable.const_mul (integrable_upperExpKernel τ hτ μ x) (x - u)),
      integral_const_mul]
  have hupperExp : upperExpMass τ μ x = upperExpMass τ μ u :=
    upperExpMass_eq_of_gap_zero τ hτ μ hgap hux hxv
  rw [hzero, zero_add, hblock, hupperExp] at hsplit
  linarith

/-- On a zero-mass gap of `p+q`, all three one-sided coefficients in the
Laplace decomposition are constant. -/
lemma pairings_eq_of_sum_gap_zero
    (τ : ℝ) (hτ : 0 < τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q]
    {u v x : ℝ} (hgap : (p + q) (Set.Ioo u v) = 0)
    (hux : u < x) (hxv : x < v) :
    truncatedPairing τ p q x = truncatedPairing τ p q u ∧
      middlePairing τ p q x = middlePairing τ p q u ∧
        upperPairing τ p q x = upperPairing τ p q u := by
  have hpGap : p (Set.Ioo u v) = 0 :=
    left_gap_measure_zero_of_sum_gap_zero hgap
  have hqGap : q (Set.Ioo u v) = 0 :=
    right_gap_measure_zero_of_sum_gap_zero hgap
  have hPm : lowerExpMass τ p x = lowerExpMass τ p u :=
    lowerExpMass_eq_of_gap_zero τ hτ p hpGap hux hxv
  have hQm : lowerExpMass τ q x = lowerExpMass τ q u :=
    lowerExpMass_eq_of_gap_zero τ hτ q hqGap hux hxv
  have hP : lowerCompensatedMoment τ p x =
      lowerCompensatedMoment τ p u + (x - u) * lowerExpMass τ p u :=
    lowerCompensatedMoment_eq_of_gap_zero τ hτ p hpGap hux hxv
  have hQ : lowerCompensatedMoment τ q x =
      lowerCompensatedMoment τ q u + (x - u) * lowerExpMass τ q u :=
    lowerCompensatedMoment_eq_of_gap_zero τ hτ q hqGap hux hxv
  have hPp : upperExpMass τ p x = upperExpMass τ p u :=
    upperExpMass_eq_of_gap_zero τ hτ p hpGap hux hxv
  have hQp : upperExpMass τ q x = upperExpMass τ q u :=
    upperExpMass_eq_of_gap_zero τ hτ q hqGap hux hxv
  have hPh : upperCompensatedMoment τ p x =
      upperCompensatedMoment τ p u - (x - u) * upperExpMass τ p u :=
    upperCompensatedMoment_eq_of_gap_zero τ hτ p hpGap hux hxv
  have hQh : upperCompensatedMoment τ q x =
      upperCompensatedMoment τ q u - (x - u) * upperExpMass τ q u :=
    upperCompensatedMoment_eq_of_gap_zero τ hτ q hqGap hux hxv
  constructor
  · unfold truncatedPairing
    rw [hQ, hPm, hP, hQm]
    ring
  constructor
  · unfold middlePairing
    rw [hQm, hPh, hQp, hP, hPm, hQh, hPp, hQ]
    ring
  · unfold upperPairing
    rw [hPh, hQp, hQh, hPp]
    ring

/-- **Gap extraction.**  On any open zero-mass gap of `p+q`, the zero-drift
decomposition is a quadratic in `exp (x/τ)^2` with constant coefficients;
therefore the lower coefficient `truncatedPairing` vanishes throughout the
gap. -/
theorem truncatedPairing_eq_zero_on_gap
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {u v x : ℝ} (hgap : (p + q) (Set.Ioo u v) = 0)
    (hux : u < x) (hxv : x < v) :
    truncatedPairing τ p q x = 0 := by
  have hτ0 : 0 < τ := hτ
  have hquad : ∀ y ∈ Set.Ioo u v,
      truncatedPairing τ p q u +
        middlePairing τ p q u * (Real.exp (y / τ) ^ 2) +
          upperPairing τ p q u * (Real.exp (y / τ) ^ 2) ^ 2 = 0 := by
    intro y hy
    have hpair := pairings_eq_of_sum_gap_zero τ hτ0 p q hgap hy.1 hy.2
    have hdec := laplaceZeroDrift_decomposition τ hτ p q hzero y
    set em : ℝ := Real.exp (-y / τ)
    set ep : ℝ := Real.exp (y / τ)
    set A : ℝ := truncatedPairing τ p q y
    set B : ℝ := middlePairing τ p q y
    set C : ℝ := upperPairing τ p q y
    have hdec' : em ^ 2 * A + B + ep ^ 2 * C = 0 := by
      simpa [em, ep, A, B, C] using hdec
    have hcancel : em * ep = 1 := by
      change Real.exp (-y / τ) * Real.exp (y / τ) = 1
      rw [← Real.exp_add]
      rw [show -y / τ + y / τ = 0 by ring, Real.exp_zero]
    have hmul : (em ^ 2 * A + B + ep ^ 2 * C) * ep ^ 2 = 0 := by
      rw [hdec', zero_mul]
    have hrewrite :
        (em ^ 2 * A + B + ep ^ 2 * C) * ep ^ 2 =
          A + B * ep ^ 2 + C * (ep ^ 2) ^ 2 := by
      calc
        (em ^ 2 * A + B + ep ^ 2 * C) * ep ^ 2
            = (em * ep) ^ 2 * A + B * ep ^ 2 + C * (ep ^ 2) ^ 2 := by
                ring
        _ = A + B * ep ^ 2 + C * (ep ^ 2) ^ 2 := by
                rw [hcancel, one_pow, one_mul]
    rw [hrewrite] at hmul
    rw [← hpair.1, ← hpair.2.1, ← hpair.2.2]
    simpa [ep, A, B, C] using hmul
  have hmono : StrictMonoOn (fun t : ℝ => Real.exp (t / τ) ^ 2) (Set.Ioo u v) := by
    intro s _ t _ hst
    have h1 : Real.exp (s / τ) < Real.exp (t / τ) := by
      apply Real.exp_lt_exp.mpr
      exact div_lt_div_of_pos_right hst hτ0
    have h2 := Real.exp_pos (s / τ)
    have h3 := Real.exp_pos (t / τ)
    simp only
    nlinarith
  have hinf : ((fun y : ℝ => Real.exp (y / τ) ^ 2) '' Set.Ioo u v).Infinite :=
    Set.Infinite.image hmono.injOn
      (Set.infinite_coe_iff.mp (Set.Ioo.infinite (lt_trans hux hxv)))
  have hpoly := quadratic_vanish hinf
    (c₀ := truncatedPairing τ p q u)
    (c₁ := middlePairing τ p q u)
    (c₂ := upperPairing τ p q u) ?_
  · have hpairx := pairings_eq_of_sum_gap_zero τ hτ0 p q hgap hux hxv
    rw [hpairx.1]
    exact hpoly.1
  · rintro w ⟨y, hy, rfl⟩
    exact hquad y hy

/-- Right-dense zero-mass gaps turn zero drift into right-dense zeros of the
truncated pairing. -/
theorem rightDenseZeros_truncatedPairing_of_zeroDrift_rightDenseZeroMassGaps
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hgaps : RightDenseZeroMassGaps p q) :
    RightDenseZeros (fun x => truncatedPairing τ p q x) := by
  intro x ε hε
  obtain ⟨u, v, hxu, huv, hvε, hgap⟩ := hgaps x ε hε
  let y : ℝ := (u + v) / 2
  have huy : u < y := by
    dsimp [y]
    linarith
  have hyv : y < v := by
    dsimp [y]
    linarith
  refine ⟨y, ?_, ?_, ?_⟩
  · exact lt_trans hxu huy
  · exact lt_trans hyv hvε
  · exact truncatedPairing_eq_zero_on_gap τ hτ p q hzero hgap huy hyv

/-- **Milestone 3 headline.**  For the one-dimensional Laplace kernel, zero
drift identifies arbitrary probability measures whose combined support has
right-dense zero-mass gaps.  The hypothesis is the concrete Lean-facing form
of the roadmap's nowhere-dense-support condition. -/
theorem laplaceZeroDrift_identifies_of_rightDense_zeroMassGaps
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hgaps : RightDenseZeroMassGaps p q) :
    p = q := by
  exact laplaceZeroDrift_identifies_of_rightDense_truncatedPairing_zeros τ hτ p q
    (rightDenseZeros_truncatedPairing_of_zeroDrift_rightDenseZeroMassGaps
      τ hτ p q hzero hgaps)

/-- Alias using the roadmap terminology.  The actual formal assumption remains
the explicit right-dense zero-mass gap condition. -/
theorem laplaceZeroDrift_identifies_of_nowhereDense_support
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hgaps : RightDenseZeroMassGaps p q) :
    p = q :=
  laplaceZeroDrift_identifies_of_rightDense_zeroMassGaps τ hτ p q hzero hgaps

end DriftingIdentifiability
