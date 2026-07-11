import DriftingIdentifiability.LaplaceGeneralConverse
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.MeasureTheory.Constructions.BorelSpace.Order

/-!
# The general 1-d Laplace converse: the endgame `𝔞 ≡ 0 ⟹ p = q`

This module implements Milestone 2 of `LaplaceGeneralConverseRoadmap.md`.

Given the *truncated pairing* `𝔞(x) = Q(x) P⁻(x) - P(x) Q⁻(x)` from
`LaplaceGeneralConverse.lean` (`truncatedPairing`), we prove that

`(∀ x, truncatedPairing τ p q x = 0) → p = q`

for arbitrary probability measures on `ℝ`, with no moment hypotheses.  Zero
drift is *not* needed as a hypothesis — the vanishing of the truncated pairing
alone forces equality.  (Whether zero drift forces `𝔞 ≡ 0` in general is
Milestone 5, still open; the finite/atomic case is
`LaplaceAtomicConverse.lean`.)

The proof:

* The compensated lower moment `P(x) = ∫_{y≤x}(x-y)e^{y/τ}dp` is `C⁰` with
  right-derivative `P⁻(x) = ∫_{y≤x}e^{y/τ}dp` (a squeeze that reuses the
  Milestone-1 right-continuity of `P⁻`).
* `𝔞` is exactly the Wronskian bracket `Q·P⁻ - P·Q⁻` of `(P, Q)`, so on the
  positivity ray `{P > 0}` the ratio `Q/P` has right-derivative `0`; by
  `constant_of_has_deriv_right_zero` it is constant `= λ`, giving `Q = λ·P`.
* An open-up-set / support argument extends `Q = λ·P` to all of `ℝ`;
  differentiating gives `Q⁻ = λ·P⁻` everywhere, i.e. equal `e^{y/τ}`-weighted
  CDFs.
* Undoing the density (`withDensity`) and using `ext_of_Iic` on truncations
  gives `q = (ofReal λ) • p`; total mass forces `ofReal λ = 1`, hence `p = q`.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-! ## Positivity, monotonicity, and continuity of the lower transforms -/

lemma lowerExpMass_nonneg (τ : ℝ) (p : Measure ℝ) (x : ℝ) :
    0 ≤ lowerExpMass τ p x :=
  setIntegral_nonneg measurableSet_Iic (fun _ _ => (Real.exp_pos _).le)

lemma lowerCompensatedMoment_nonneg (τ : ℝ) (p : Measure ℝ) (x : ℝ) :
    0 ≤ lowerCompensatedMoment τ p x :=
  setIntegral_nonneg measurableSet_Iic
    (fun y hy => mul_nonneg (by have := Set.mem_Iic.mp hy; linarith) (Real.exp_pos _).le)

/-- The compensated lower moment is strictly positive exactly when there is
`p`-mass strictly below `x`. -/
lemma lowerCompensatedMoment_pos_iff (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ)
    [IsFiniteMeasure p] (x : ℝ) :
    0 < lowerCompensatedMoment τ p x ↔ 0 < p (Set.Iio x) := by
  have hset :
      Function.support (fun y => (x - y) * Real.exp (y / τ)) ∩ Set.Iic x = Set.Iio x := by
    ext y
    constructor
    · rintro ⟨hy, hle⟩
      rw [Function.mem_support] at hy
      have hne : x - y ≠ 0 := fun hh => hy (by rw [hh, zero_mul])
      exact lt_of_le_of_ne (Set.mem_Iic.mp hle) (sub_ne_zero.mp hne).symm
    · intro hy
      rw [Set.mem_Iio] at hy
      refine ⟨?_, Set.mem_Iic.mpr hy.le⟩
      rw [Function.mem_support]
      exact mul_ne_zero (sub_ne_zero.mpr (ne_of_lt hy).symm) (Real.exp_pos _).ne'
  unfold lowerCompensatedMoment
  rw [setIntegral_pos_iff_support_of_nonneg_ae
      (by filter_upwards [ae_restrict_mem measurableSet_Iic] with y hy
          exact mul_nonneg (by have := Set.mem_Iic.mp hy; linarith) (Real.exp_pos _).le)
      (integrable_lowerCompKernel τ hτ p x), hset]

/-- There is a probe where the compensated lower moment is positive. -/
lemma exists_lowerCompensatedMoment_pos (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] : ∃ x, 0 < lowerCompensatedMoment τ p x := by
  by_contra hcon
  have hcon' : ∀ x, lowerCompensatedMoment τ p x ≤ 0 :=
    fun x => not_lt.mp (fun h => hcon ⟨x, h⟩)
  have hz : ∀ x : ℝ, p (Set.Iio x) = 0 := by
    intro x
    by_contra hh
    have hpos : 0 < p (Set.Iio x) := pos_iff_ne_zero.mpr hh
    have := (lowerCompensatedMoment_pos_iff τ hτ p x).mpr hpos
    exact absurd this (not_lt.mpr (hcon' x))
  have huniv : (Set.univ : Set ℝ) = ⋃ n : ℕ, Set.Iio (n : ℝ) := by
    ext y
    simp only [Set.mem_univ, Set.mem_iUnion, Set.mem_Iio, true_iff]
    obtain ⟨n, hn⟩ := exists_nat_gt y
    exact ⟨n, hn⟩
  have hzero : p Set.univ = 0 := by
    rw [huniv]; exact measure_iUnion_null (fun n => hz _)
  rw [measure_univ] at hzero
  exact one_ne_zero hzero

/-- Elementary pointwise bound `max (x-y) 0 · e^{y/τ} ≤ τ · e^{x/τ}`. -/
lemma max_mul_exp_le (τ : ℝ) (hτ : 0 < τ) (x y : ℝ) :
    max (x - y) 0 * Real.exp (y / τ) ≤ τ * Real.exp (x / τ) := by
  by_cases hh : x - y ≤ 0
  · rw [max_eq_right hh, zero_mul]; positivity
  · have hs : 0 ≤ x - y := (not_le.mp hh).le
    rw [max_eq_left hs]
    have hexp : Real.exp (y / τ) =
        Real.exp (x / τ) * Real.exp (-(1 / τ) * (x - y)) := by
      rw [← Real.exp_add]; congr 1; field_simp [hτ.ne']; ring
    calc
      (x - y) * Real.exp (y / τ)
          = Real.exp (x / τ) * ((x - y) * Real.exp (-(1 / τ) * (x - y))) := by
            rw [hexp]; ring
      _ ≤ Real.exp (x / τ) * τ :=
            mul_le_mul_of_nonneg_left (mul_exp_neg_le_general hτ hs) (Real.exp_pos _).le
      _ = τ * Real.exp (x / τ) := by ring

/-- The compensated lower moment as a genuine `ℝ`-integral of a globally
continuous (in `x`) integrand `max (x-y) 0 · e^{y/τ}`. -/
lemma lowerCompensatedMoment_eq_integral_max (τ : ℝ) (p : Measure ℝ) (x : ℝ) :
    lowerCompensatedMoment τ p x = ∫ y, max (x - y) 0 * Real.exp (y / τ) ∂p := by
  unfold lowerCompensatedMoment
  rw [← integral_indicator measurableSet_Iic]
  refine integral_congr_ae (Eventually.of_forall (fun y => ?_))
  by_cases hy : y ≤ x
  · rw [Set.indicator_of_mem (Set.mem_Iic.mpr hy)]
    change (x - y) * Real.exp (y / τ) = max (x - y) 0 * Real.exp (y / τ)
    rw [max_eq_left (by linarith : (0 : ℝ) ≤ x - y)]
  · rw [Set.indicator_of_notMem (by simpa using hy)]
    change (0 : ℝ) = max (x - y) 0 * Real.exp (y / τ)
    rw [max_eq_right (by simp only [not_le] at hy; linarith : x - y ≤ 0), zero_mul]

/-- The compensated lower moment is continuous. -/
lemma continuous_lowerCompensatedMoment (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ)
    [IsFiniteMeasure p] : Continuous (fun x => lowerCompensatedMoment τ p x) := by
  simp_rw [lowerCompensatedMoment_eq_integral_max]
  rw [continuous_iff_continuousAt]
  intro x₀
  refine continuousAt_of_dominated (bound := fun y => τ * Real.exp ((x₀ + 1) / τ))
    (Eventually.of_forall (fun x => by fun_prop)) ?_ (integrable_const _)
    (Eventually.of_forall (fun y => by fun_prop))
  filter_upwards [isOpen_Iio.mem_nhds (show x₀ < x₀ + 1 by linarith)] with x hx
  refine Eventually.of_forall (fun y => ?_)
  rw [Real.norm_eq_abs,
    abs_of_nonneg (mul_nonneg (le_max_right _ _) (Real.exp_pos _).le)]
  calc
    max (x - y) 0 * Real.exp (y / τ) ≤ τ * Real.exp (x / τ) := max_mul_exp_le τ hτ x y
    _ ≤ τ * Real.exp ((x₀ + 1) / τ) := by
        have : x ≤ x₀ + 1 := (Set.mem_Iio.mp hx).le
        exact mul_le_mul_of_nonneg_left
          (Real.exp_le_exp.mpr (div_le_div_of_nonneg_right this hτ.le)) hτ.le

/-! ## The right-derivative of the compensated lower moment -/

/-- Increment of `P⁻` across `(x₀, x']`. -/
lemma lowerExpMass_sub (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p]
    {x₀ x' : ℝ} (hx : x₀ ≤ x') :
    lowerExpMass τ p x' - lowerExpMass τ p x₀
      = ∫ y in Set.Ioc x₀ x', Real.exp (y / τ) ∂p := by
  have hInt : IntegrableOn (fun y => Real.exp (y / τ)) (Set.Iic x') p :=
    integrable_lowerExpKernel τ hτ p x'
  have h1 : IntegrableOn (fun y => Real.exp (y / τ)) (Set.Iic x₀) p :=
    hInt.mono_set (Set.Iic_subset_Iic.mpr hx)
  have h2 : IntegrableOn (fun y => Real.exp (y / τ)) (Set.Ioc x₀ x') p :=
    hInt.mono_set Set.Ioc_subset_Iic_self
  have hdisj : Disjoint (Set.Iic x₀) (Set.Ioc x₀ x') :=
    (Set.Iic_disjoint_Ioi le_rfl).mono_right Set.Ioc_subset_Ioi_self
  unfold lowerExpMass
  rw [show Set.Iic x' = Set.Iic x₀ ∪ Set.Ioc x₀ x' from
      (Set.Iic_union_Ioc_eq_Iic hx).symm,
    setIntegral_union hdisj measurableSet_Ioc h1 h2]
  ring

/-- Increment of `P` across `(x₀, x']`: the affine part `(x'-x₀)·P⁻(x₀)` plus a
nonnegative remainder supported on `(x₀, x']`. -/
lemma lowerCompensatedMoment_sub (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p]
    {x₀ x' : ℝ} (hx : x₀ ≤ x') :
    lowerCompensatedMoment τ p x' - lowerCompensatedMoment τ p x₀
      = (x' - x₀) * lowerExpMass τ p x₀
        + ∫ y in Set.Ioc x₀ x', (x' - y) * Real.exp (y / τ) ∂p := by
  have hIntC : IntegrableOn (fun y => (x' - y) * Real.exp (y / τ)) (Set.Iic x') p :=
    integrable_lowerCompKernel τ hτ p x'
  have hC0 : IntegrableOn (fun y => (x' - y) * Real.exp (y / τ)) (Set.Iic x₀) p :=
    hIntC.mono_set (Set.Iic_subset_Iic.mpr hx)
  have hCoc : IntegrableOn (fun y => (x' - y) * Real.exp (y / τ)) (Set.Ioc x₀ x') p :=
    hIntC.mono_set Set.Ioc_subset_Iic_self
  have hdisj : Disjoint (Set.Iic x₀) (Set.Ioc x₀ x') :=
    (Set.Iic_disjoint_Ioi le_rfl).mono_right Set.Ioc_subset_Ioi_self
  have hbaseExp : IntegrableOn (fun y => Real.exp (y / τ)) (Set.Iic x') p :=
    integrable_lowerExpKernel τ hτ p x'
  have hExpMass : IntegrableOn (fun y => Real.exp (y / τ)) (Set.Iic x₀) p :=
    hbaseExp.mono_set (Set.Iic_subset_Iic.mpr hx)
  have hComp0 : IntegrableOn (fun y => (x₀ - y) * Real.exp (y / τ)) (Set.Iic x₀) p :=
    integrable_lowerCompKernel τ hτ p x₀
  -- split `P x'` over `Iic x₀ ∪ Ioc x₀ x'`
  have hsplit : lowerCompensatedMoment τ p x'
      = (∫ y in Set.Iic x₀, (x' - y) * Real.exp (y / τ) ∂p)
        + ∫ y in Set.Ioc x₀ x', (x' - y) * Real.exp (y / τ) ∂p := by
    unfold lowerCompensatedMoment
    rw [show Set.Iic x' = Set.Iic x₀ ∪ Set.Ioc x₀ x' from
        (Set.Iic_union_Ioc_eq_Iic hx).symm,
      setIntegral_union hdisj measurableSet_Ioc hC0 hCoc]
  -- expand the `Iic x₀` block: `(x' - y) = (x' - x₀) + (x₀ - y)`
  have hblock :
      (∫ y in Set.Iic x₀, (x' - y) * Real.exp (y / τ) ∂p)
        = (x' - x₀) * lowerExpMass τ p x₀ + lowerCompensatedMoment τ p x₀ := by
    unfold lowerExpMass lowerCompensatedMoment
    have hpt : (fun y : ℝ => (x' - y) * Real.exp (y / τ))
        = fun y => (x' - x₀) * Real.exp (y / τ) + (x₀ - y) * Real.exp (y / τ) := by
      funext y; ring
    rw [hpt, integral_add (Integrable.const_mul hExpMass _) hComp0, integral_const_mul]
  rw [hsplit, hblock]; ring

/-- **Milestone-2, E1.**  The compensated lower moment `P` is right-differentiable
with right-derivative `P⁻`. -/
theorem hasDerivWithinAt_lowerCompensatedMoment (τ : ℝ) (hτ : ValidBandwidth τ)
    (p : Measure ℝ) [IsFiniteMeasure p] (x₀ : ℝ) :
    HasDerivWithinAt (lowerCompensatedMoment τ p) (lowerExpMass τ p x₀)
      (Set.Ici x₀) x₀ := by
  have hτ0 : 0 < τ := hτ
  rw [hasDerivWithinAt_iff_tendsto_slope]
  have hdiff : Set.Ici x₀ \ {x₀} = Set.Ioi x₀ := by
    ext y; simp only [Set.mem_sdiff, Set.mem_Ici, Set.mem_singleton_iff, Set.mem_Ioi]
    constructor
    · rintro ⟨hle, hne⟩; exact lt_of_le_of_ne hle (Ne.symm hne)
    · intro hlt; exact ⟨hlt.le, ne_of_gt hlt⟩
  rw [hdiff]
  have hupper : Tendsto (fun x' => lowerExpMass τ p x') (𝓝[Set.Ioi x₀] x₀)
      (𝓝 (lowerExpMass τ p x₀)) :=
    (lowerExpMass_continuousWithinAt_Ici τ hτ p x₀).tendsto.mono_left
      (nhdsWithin_mono x₀ Set.Ioi_subset_Ici_self)
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hupper ?_ ?_
  · refine eventually_of_mem self_mem_nhdsWithin (fun x' hx' => ?_)
    have hlt : x₀ < x' := hx'
    rw [slope_def_field, le_div_iff₀ (by linarith)]
    have hsub := lowerCompensatedMoment_sub τ hτ0 p hlt.le
    have hR : 0 ≤ ∫ y in Set.Ioc x₀ x', (x' - y) * Real.exp (y / τ) ∂p :=
      setIntegral_nonneg measurableSet_Ioc
        (fun y hy => mul_nonneg (by have := (Set.mem_Ioc.mp hy).2; linarith) (Real.exp_pos _).le)
    nlinarith [hsub, hR]
  · refine eventually_of_mem self_mem_nhdsWithin (fun x' hx' => ?_)
    have hlt : x₀ < x' := hx'
    rw [slope_def_field, div_le_iff₀ (by linarith)]
    have hsub := lowerCompensatedMoment_sub τ hτ0 p hlt.le
    have hexpsub := lowerExpMass_sub τ hτ0 p hlt.le
    have hRle : (∫ y in Set.Ioc x₀ x', (x' - y) * Real.exp (y / τ) ∂p)
        ≤ (x' - x₀) * ∫ y in Set.Ioc x₀ x', Real.exp (y / τ) ∂p := by
      rw [← integral_const_mul]
      have hbaseC : IntegrableOn (fun y => (x' - y) * Real.exp (y / τ)) (Set.Iic x') p :=
        integrable_lowerCompKernel τ hτ0 p x'
      have hbaseE : IntegrableOn (fun y => Real.exp (y / τ)) (Set.Iic x') p :=
        integrable_lowerExpKernel τ hτ0 p x'
      have hi1 : IntegrableOn (fun y => (x' - y) * Real.exp (y / τ)) (Set.Ioc x₀ x') p :=
        hbaseC.mono_set Set.Ioc_subset_Iic_self
      have hi2 : IntegrableOn (fun y => (x' - x₀) * Real.exp (y / τ)) (Set.Ioc x₀ x') p :=
        Integrable.const_mul (hbaseE.mono_set Set.Ioc_subset_Iic_self) _
      refine setIntegral_mono_on hi1 hi2 measurableSet_Ioc (fun y hy => ?_)
      have hy2 := (Set.mem_Ioc.mp hy).1
      exact mul_le_mul_of_nonneg_right (by linarith) (Real.exp_pos _).le
    rw [← hexpsub] at hRle
    nlinarith [hsub, hRle]

/-! ## Ratio constancy and extension of the proportionality -/

/-- `∫⁻_{Iic c} e^{y/τ} = ofReal (P⁻ c)` (bridge from `withDensity` to the
Bochner transform). -/
lemma lintegral_expDensity_Iic (τ : ℝ) (hτ : 0 < τ) (μ : Measure ℝ) [IsFiniteMeasure μ]
    (c : ℝ) :
    ∫⁻ y in Set.Iic c, ENNReal.ofReal (Real.exp (y / τ)) ∂μ
      = ENNReal.ofReal (lowerExpMass τ μ c) := by
  unfold lowerExpMass
  rw [ofReal_integral_eq_lintegral_ofReal (integrable_lowerExpKernel τ hτ μ c)
    (Filter.Eventually.of_forall (fun y => (Real.exp_pos (y / τ)).le))]

/-- **Milestone-2, support step.**  If `Q = L·P` on the positivity ray, then
`Q = 0` wherever `P = 0`: the ratio extends to `0` at the left edge of the
support and `q` carries no mass below it. -/
lemma lowerCompensatedMoment_eq_zero_of_proportional
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (L : ℝ)
    (hray : ∀ x, 0 < lowerCompensatedMoment τ p x →
      lowerCompensatedMoment τ q x = L * lowerCompensatedMoment τ p x)
    {x : ℝ} (hPx : lowerCompensatedMoment τ p x = 0) :
    lowerCompensatedMoment τ q x = 0 := by
  have hτ0 : 0 < τ := hτ
  obtain ⟨x₀, hx₀⟩ := exists_lowerCompensatedMoment_pos τ hτ0 p
  have hAup : ∀ s t : ℝ, s ≤ t → 0 < lowerCompensatedMoment τ p s →
      0 < lowerCompensatedMoment τ p t := by
    intro s t hst hs
    rw [lowerCompensatedMoment_pos_iff τ hτ0 p] at hs ⊢
    exact lt_of_lt_of_le hs (measure_mono (Set.Iio_subset_Iio hst))
  have hxlb : ∀ t, 0 < lowerCompensatedMoment τ p t → x ≤ t := by
    intro t ht
    by_contra hcon
    have hpos : 0 < lowerCompensatedMoment τ p x := hAup t x (not_le.mp hcon).le ht
    rw [hPx] at hpos; exact lt_irrefl 0 hpos
  have hne : {t : ℝ | 0 < lowerCompensatedMoment τ p t}.Nonempty := ⟨x₀, hx₀⟩
  have hbddb : BddBelow {t : ℝ | 0 < lowerCompensatedMoment τ p t} :=
    ⟨x, fun t ht => hxlb t ht⟩
  have hxs : x ≤ sInf {t : ℝ | 0 < lowerCompensatedMoment τ p t} :=
    le_csInf hne (fun t ht => hxlb t ht)
  -- P vanishes at the infimum of the positivity set
  have hPs : lowerCompensatedMoment τ p (sInf {t : ℝ | 0 < lowerCompensatedMoment τ p t}) = 0 := by
    set s := sInf {t : ℝ | 0 < lowerCompensatedMoment τ p t} with hs_def
    by_contra hsne
    have hspos : 0 < p (Set.Iio s) := by
      rw [← lowerCompensatedMoment_pos_iff τ hτ0 p]
      exact lt_of_le_of_ne (lowerCompensatedMoment_nonneg τ p s) (Ne.symm hsne)
    have hex : ∃ y, y < s ∧ 0 < p (Set.Iio y) := by
      by_contra hcon
      have hnull : ∀ m : ℕ, p (Set.Iio (s - 1 / ((m : ℝ) + 1))) = 0 := by
        intro m
        have hlt : s - 1 / ((m : ℝ) + 1) < s := by
          have : 0 < 1 / ((m : ℝ) + 1) := by positivity
          linarith
        rw [← le_zero_iff]
        exact not_lt.mp (fun hp => hcon ⟨_, hlt, hp⟩)
      have hunion : Set.Iio s = ⋃ m : ℕ, Set.Iio (s - 1 / ((m : ℝ) + 1)) := by
        ext y
        simp only [Set.mem_Iio, Set.mem_iUnion]
        constructor
        · intro hy
          obtain ⟨m, hm⟩ := exists_nat_one_div_lt (show (0 : ℝ) < s - y by linarith)
          exact ⟨m, by linarith⟩
        · rintro ⟨m, hm⟩
          have : 0 < 1 / ((m : ℝ) + 1) := by positivity
          linarith
      rw [hunion, measure_iUnion_null hnull] at hspos
      exact lt_irrefl 0 hspos
    obtain ⟨y, hylt, hypos⟩ := hex
    have hymem : 0 < lowerCompensatedMoment τ p y :=
      (lowerCompensatedMoment_pos_iff τ hτ0 p y).mpr hypos
    have hle := csInf_le hbddb hymem
    exact absurd hle (not_le.mpr hylt)
  -- ratio extends by continuity: Q = L·P at the infimum, hence `= 0`
  have hcontQ : Continuous (fun t => lowerCompensatedMoment τ q t) :=
    continuous_lowerCompensatedMoment τ hτ0 q
  have hcontLP : Continuous (fun t => L * lowerCompensatedMoment τ p t) :=
    continuous_const.mul (continuous_lowerCompensatedMoment τ hτ0 p)
  have hEqOn : Set.EqOn (fun t => lowerCompensatedMoment τ q t)
      (fun t => L * lowerCompensatedMoment τ p t)
      (Set.Ioi (sInf {t : ℝ | 0 < lowerCompensatedMoment τ p t})) := by
    intro t ht
    have htpos : 0 < lowerCompensatedMoment τ p t := by
      obtain ⟨a, haA, hat⟩ := exists_lt_of_csInf_lt hne (Set.mem_Ioi.mp ht)
      exact hAup a t hat.le haA
    exact hray t htpos
  have hQs : lowerCompensatedMoment τ q (sInf {t : ℝ | 0 < lowerCompensatedMoment τ p t})
      = L * lowerCompensatedMoment τ p (sInf {t : ℝ | 0 < lowerCompensatedMoment τ p t}) := by
    have hcl := hEqOn.closure hcontQ hcontLP
    rw [closure_Ioi] at hcl
    exact hcl Set.self_mem_Ici
  rw [hPs, mul_zero] at hQs
  have hqIio : q (Set.Iio (sInf {t : ℝ | 0 < lowerCompensatedMoment τ p t})) = 0 := by
    by_contra hqne
    have hpos : 0 < lowerCompensatedMoment τ q _ :=
      (lowerCompensatedMoment_pos_iff τ hτ0 q _).mpr (pos_iff_ne_zero.mpr hqne)
    rw [hQs] at hpos; exact lt_irrefl 0 hpos
  have hqx : q (Set.Iio x) = 0 :=
    le_zero_iff.mp (le_trans (measure_mono (Set.Iio_subset_Iio hxs)) (le_of_eq hqIio))
  have hnotpos : ¬ 0 < lowerCompensatedMoment τ q x := by
    rw [lowerCompensatedMoment_pos_iff τ hτ0 q, hqx]; exact lt_irrefl 0
  exact le_antisymm (not_lt.mp hnotpos) (lowerCompensatedMoment_nonneg τ q x)

/-- **Milestone-2, proportionality.**  Vanishing truncated pairing forces the
compensated lower moments to be proportional: `∃ L ≥ 0, Q = L·P` everywhere. -/
lemma lowerCompensatedMoment_proportional
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (h : ∀ x, truncatedPairing τ p q x = 0) :
    ∃ L : ℝ, 0 ≤ L ∧ ∀ x, lowerCompensatedMoment τ q x = L * lowerCompensatedMoment τ p x := by
  have hτ0 : 0 < τ := hτ
  obtain ⟨x₀, hx₀⟩ := exists_lowerCompensatedMoment_pos τ hτ0 p
  refine ⟨lowerCompensatedMoment τ q x₀ / lowerCompensatedMoment τ p x₀,
    div_nonneg (lowerCompensatedMoment_nonneg τ q x₀) hx₀.le, ?_⟩
  have hAup : ∀ s t : ℝ, s ≤ t → 0 < lowerCompensatedMoment τ p s →
      0 < lowerCompensatedMoment τ p t := by
    intro s t hst hs
    rw [lowerCompensatedMoment_pos_iff τ hτ0 p] at hs ⊢
    exact lt_of_lt_of_le hs (measure_mono (Set.Iio_subset_Iio hst))
  have hconst : ∀ a b : ℝ, a ≤ b → (∀ t ∈ Set.Icc a b, 0 < lowerCompensatedMoment τ p t) →
      lowerCompensatedMoment τ q b / lowerCompensatedMoment τ p b
        = lowerCompensatedMoment τ q a / lowerCompensatedMoment τ p a := by
    intro a b hab hpos
    have hcont : ContinuousOn
        (fun t => lowerCompensatedMoment τ q t / lowerCompensatedMoment τ p t) (Set.Icc a b) := by
      apply ContinuousOn.div
      · exact (continuous_lowerCompensatedMoment τ hτ0 q).continuousOn
      · exact (continuous_lowerCompensatedMoment τ hτ0 p).continuousOn
      · exact fun t ht => ne_of_gt (hpos t ht)
    have hderiv : ∀ t ∈ Set.Ico a b, HasDerivWithinAt
        (fun t => lowerCompensatedMoment τ q t / lowerCompensatedMoment τ p t) 0 (Set.Ici t) t := by
      intro t ht
      have htmem : t ∈ Set.Icc a b := ⟨ht.1, ht.2.le⟩
      have hPt : lowerCompensatedMoment τ p t ≠ 0 := ne_of_gt (hpos t htmem)
      have hdiv := (hasDerivWithinAt_lowerCompensatedMoment τ hτ q t).div
        (hasDerivWithinAt_lowerCompensatedMoment τ hτ p t) hPt
      have hnum : lowerExpMass τ q t * lowerCompensatedMoment τ p t
          - lowerCompensatedMoment τ q t * lowerExpMass τ p t = 0 := by
        have hpair := h t
        unfold truncatedPairing at hpair
        linear_combination -hpair
      have hz : (lowerExpMass τ q t * lowerCompensatedMoment τ p t
          - lowerCompensatedMoment τ q t * lowerExpMass τ p t)
            / lowerCompensatedMoment τ p t ^ 2 = 0 := by
        rw [hnum, zero_div]
      rw [← hz]; exact hdiv
    exact constant_of_has_deriv_right_zero hcont hderiv b ⟨hab, le_refl b⟩
  have hray : ∀ x, 0 < lowerCompensatedMoment τ p x →
      lowerCompensatedMoment τ q x
        = (lowerCompensatedMoment τ q x₀ / lowerCompensatedMoment τ p x₀)
          * lowerCompensatedMoment τ p x := by
    intro x hxpos
    rcases le_total x x₀ with hle | hle
    · have hpos : ∀ t ∈ Set.Icc x x₀, 0 < lowerCompensatedMoment τ p t :=
        fun t ht => hAup x t ht.1 hxpos
      exact (div_eq_iff (ne_of_gt hxpos)).mp (hconst x x₀ hle hpos).symm
    · have hpos : ∀ t ∈ Set.Icc x₀ x, 0 < lowerCompensatedMoment τ p t :=
        fun t ht => hAup x₀ t ht.1 hx₀
      exact (div_eq_iff (ne_of_gt hxpos)).mp (hconst x₀ x hle hpos)
  intro x
  by_cases hxpos : 0 < lowerCompensatedMoment τ p x
  · exact hray x hxpos
  · have hPx0 : lowerCompensatedMoment τ p x = 0 :=
      le_antisymm (not_lt.mp hxpos) (lowerCompensatedMoment_nonneg τ p x)
    rw [hPx0, mul_zero]
    exact lowerCompensatedMoment_eq_zero_of_proportional τ hτ p q _ hray hPx0

/-! ## Measure recovery -/

/-- **Milestone-2, E6 (truncated).**  Equal `e^{y/τ}`-weighted CDFs up to a
constant `L` force `q = (ofReal L) • p` on every `Iic n`. -/
lemma restrict_eq_of_lowerExpMass_prop (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (L : ℝ) (hL0 : 0 ≤ L)
    (hexp : ∀ x, lowerExpMass τ q x = L * lowerExpMass τ p x) (n : ℕ) :
    q.restrict (Set.Iic (n : ℝ)) = ENNReal.ofReal L • p.restrict (Set.Iic (n : ℝ)) := by
  have hτ0 : 0 < τ := hτ
  have hwmeas : Measurable (fun y : ℝ => ENNReal.ofReal (Real.exp (y / τ))) := by fun_prop
  have hw'meas : Measurable (fun y : ℝ => ENNReal.ofReal (Real.exp (-y / τ))) := by fun_prop
  haveI hfinq : IsFiniteMeasure ((q.restrict (Set.Iic (n : ℝ))).withDensity
      (fun y => ENNReal.ofReal (Real.exp (y / τ)))) := by
    refine isFiniteMeasure_withDensity ?_
    refine ne_top_of_le_ne_top
      (lintegral_const_lt_top (μ := q.restrict (Set.Iic (n : ℝ)))
        (c := ENNReal.ofReal (Real.exp ((n : ℝ) / τ))) ENNReal.ofReal_ne_top).ne
      (lintegral_mono_ae ?_)
    filter_upwards [ae_restrict_mem measurableSet_Iic] with y hy
    exact ENNReal.ofReal_le_ofReal
      (Real.exp_le_exp.mpr (div_le_div_of_nonneg_right (Set.mem_Iic.mp hy) hτ0.le))
  -- transform of a truncated restriction at `Iic a`
  have htrunc : ∀ (μ : Measure ℝ) [IsFiniteMeasure μ] (a : ℝ),
      ((μ.restrict (Set.Iic (n : ℝ))).withDensity
          (fun y => ENNReal.ofReal (Real.exp (y / τ)))) (Set.Iic a)
        = ENNReal.ofReal (lowerExpMass τ μ (a ⊓ (n : ℝ))) := by
    intro μ _ a
    rw [withDensity_apply _ measurableSet_Iic,
      show (∫⁻ y in Set.Iic a, ENNReal.ofReal (Real.exp (y / τ)) ∂(μ.restrict (Set.Iic (n : ℝ))))
        = ∫⁻ y, ENNReal.ofReal (Real.exp (y / τ))
            ∂((μ.restrict (Set.Iic (n : ℝ))).restrict (Set.Iic a)) from rfl,
      Measure.restrict_restrict measurableSet_Iic, Set.Iic_inter_Iic]
    exact lintegral_expDensity_Iic τ hτ0 μ (a ⊓ (n : ℝ))
  have hagree : ∀ a : ℝ,
      ((q.restrict (Set.Iic (n : ℝ))).withDensity
          (fun y => ENNReal.ofReal (Real.exp (y / τ)))) (Set.Iic a)
        = (ENNReal.ofReal L • (p.restrict (Set.Iic (n : ℝ))).withDensity
            (fun y => ENNReal.ofReal (Real.exp (y / τ)))) (Set.Iic a) := by
    intro a
    rw [Measure.smul_apply, smul_eq_mul, htrunc q a, htrunc p a, hexp (a ⊓ (n : ℝ)),
      ENNReal.ofReal_mul hL0]
  have hνeq := Measure.ext_of_Iic _ _ hagree
  -- undo the density with the reciprocal `e^{-y/τ}`
  have hww1 : (fun y : ℝ => ENNReal.ofReal (Real.exp (y / τ)))
      * (fun y : ℝ => ENNReal.ofReal (Real.exp (-y / τ))) = 1 := by
    funext y
    simp only [Pi.mul_apply, Pi.one_apply]
    rw [← ENNReal.ofReal_mul (Real.exp_pos _).le, ← Real.exp_add,
      show y / τ + -y / τ = 0 by ring, Real.exp_zero, ENNReal.ofReal_one]
  have hundo : ∀ μ : Measure ℝ,
      (μ.withDensity (fun y => ENNReal.ofReal (Real.exp (y / τ)))).withDensity
        (fun y => ENNReal.ofReal (Real.exp (-y / τ))) = μ := by
    intro μ
    rw [← withDensity_mul _ hwmeas hw'meas, hww1, withDensity_one]
  have key := congrArg
    (fun ν => ν.withDensity (fun y => ENNReal.ofReal (Real.exp (-y / τ)))) hνeq
  simp only [withDensity_smul_measure] at key
  rw [hundo (q.restrict (Set.Iic (n : ℝ))), hundo (p.restrict (Set.Iic (n : ℝ)))] at key
  exact key

/-- **Milestone 2 (headline).**  On `ℝ`, vanishing of the truncated pairing
`𝔞 ≡ 0` alone forces `p = q`, for arbitrary probability measures and with no
moment hypotheses.  (Zero drift is not needed as a hypothesis.) -/
theorem laplaceZeroDrift_identifies_of_truncatedPairing_eq_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (h : ∀ x, truncatedPairing τ p q x = 0) : p = q := by
  have hτ0 : 0 < τ := hτ
  obtain ⟨L, hL0, hprop⟩ := lowerCompensatedMoment_proportional τ hτ p q h
  -- differentiate `Q = L·P` to get `Q⁻ = L·P⁻` everywhere
  have hexp : ∀ x, lowerExpMass τ q x = L * lowerExpMass τ p x := by
    intro x
    have hfun : lowerCompensatedMoment τ q = fun t => L * lowerCompensatedMoment τ p t :=
      funext hprop
    have hdQ' : HasDerivWithinAt (fun t => L * lowerCompensatedMoment τ p t)
        (lowerExpMass τ q x) (Set.Ici x) x := by
      rw [← hfun]; exact hasDerivWithinAt_lowerCompensatedMoment τ hτ q x
    have hdP : HasDerivWithinAt (fun t => L * lowerCompensatedMoment τ p t)
        (L * lowerExpMass τ p x) (Set.Ici x) x :=
      HasDerivWithinAt.const_mul L (hasDerivWithinAt_lowerCompensatedMoment τ hτ p x)
    have e1 := hdQ'.derivWithin (uniqueDiffWithinAt_Ici x)
    have e2 := hdP.derivWithin (uniqueDiffWithinAt_Ici x)
    rw [← e1, e2]
  -- transfer to all `Iic n`, then to `ℝ`
  have hrestrict : ∀ n : ℕ,
      q.restrict (Set.Iic (n : ℝ)) = ENNReal.ofReal L • p.restrict (Set.Iic (n : ℝ)) :=
    fun n => restrict_eq_of_lowerExpMass_prop τ hτ p q L hL0 hexp n
  have hfull : q = ENNReal.ofReal L • p := by
    ext s hs
    have hmono : Monotone (fun n : ℕ => s ∩ Set.Iic (n : ℝ)) := by
      intro a b hab
      exact Set.inter_subset_inter_right _ (Set.Iic_subset_Iic.mpr (by exact_mod_cast hab))
    have huniv : ⋃ n : ℕ, s ∩ Set.Iic (n : ℝ) = s := by
      rw [← Set.inter_iUnion]
      have huu : ⋃ n : ℕ, Set.Iic (n : ℝ) = Set.univ := by
        ext y
        simp only [Set.mem_iUnion, Set.mem_Iic, Set.mem_univ, iff_true]
        obtain ⟨n, hn⟩ := exists_nat_ge y
        exact ⟨n, hn⟩
      rw [huu, Set.inter_univ]
    have hq := tendsto_measure_iUnion_atTop (μ := q) hmono
    rw [huniv] at hq
    have hp := tendsto_measure_iUnion_atTop (μ := p) hmono
    rw [huniv] at hp
    simp only [Function.comp_def] at hq hp
    have hstep : ∀ n : ℕ,
        q (s ∩ Set.Iic (n : ℝ)) = ENNReal.ofReal L * p (s ∩ Set.Iic (n : ℝ)) := by
      intro n
      have hr := congrArg (fun μ => μ s) (hrestrict n)
      simp only [Measure.restrict_apply hs, Measure.smul_apply, smul_eq_mul] at hr
      exact hr
    have hlim : Tendsto (fun n : ℕ => ENNReal.ofReal L * p (s ∩ Set.Iic (n : ℝ))) atTop
        (𝓝 (ENNReal.ofReal L * p s)) :=
      ENNReal.Tendsto.const_mul hp (Or.inr ENNReal.ofReal_ne_top)
    rw [Measure.smul_apply, smul_eq_mul]
    refine tendsto_nhds_unique hq ?_
    simp_rw [hstep]
    exact hlim
  have hmass : ENNReal.ofReal L = 1 := by
    have h1 := congrArg (fun μ => μ Set.univ) hfull
    simp only [Measure.smul_apply, smul_eq_mul, measure_univ, mul_one] at h1
    exact h1.symm
  rw [hfull, hmass, one_smul]

end DriftingIdentifiability
