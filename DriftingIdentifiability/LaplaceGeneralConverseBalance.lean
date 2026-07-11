import DriftingIdentifiability.LaplaceGeneralConverseNowhereDense

/-!
# Balance-identity scaffold for the general 1-d Laplace converse

This module starts Milestone 4 of `LaplaceGeneralConverseRoadmap.md`.

The target balance identity under zero drift is

`exp (-(2*x)/τ) * 𝔞(x) = exp ((2*x)/τ) * 𝔠(x)`,

where `𝔞 = truncatedPairing` and `𝔠 = upperPairing`.

The full theorem still requires the unconditional derivative/weak-Fubini
identity for the cross-displacement scalar.  This file formalizes the
non-controversial infrastructure around that target:

* scaled lower/upper pairings and the balance defect;
* right-continuity of the upper pairing and of the balance defect;
* conditional bridges showing that the derivative identity immediately gives
  the pointwise balance law under zero drift.

Important: for arbitrary measures the two-sided classical derivative of the
cross-displacement scalar can fail at atoms.  The mathematically correct target
is the right-derivative/weak identity; the older `HasDerivAt` bridge is kept
only as a convenient stronger conditional.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

private lemma one_sub_exp_neg_le' (u : ℝ) :
    1 - Real.exp (-u) ≤ u := by
  have := Real.add_one_le_exp (-u)
  linarith

private lemma abs_exp_neg_sub_exp_le (a B : ℝ) (ha : 0 ≤ a) (hB : a ≤ B) :
    |Real.exp (-a) - Real.exp a| ≤ 2 * Real.exp B * a := by
  have hle : Real.exp (-a) ≤ Real.exp a := by
    exact Real.exp_le_exp.mpr (by linarith)
  rw [abs_of_nonpos (sub_nonpos.mpr hle)]
  have hrewrite : Real.exp a - Real.exp (-a) =
      Real.exp a * (1 - Real.exp (-(2 * a))) := by
    rw [mul_sub, mul_one, ← Real.exp_add]
    ring_nf
  rw [show -(Real.exp (-a) - Real.exp a) = Real.exp a - Real.exp (-a) by ring,
    hrewrite]
  have htail : 1 - Real.exp (-(2 * a)) ≤ 2 * a :=
    one_sub_exp_neg_le' (2 * a)
  have htail_nonneg : 0 ≤ 1 - Real.exp (-(2 * a)) := by
    rw [sub_nonneg]
    exact Real.exp_le_one_iff.mpr (by nlinarith)
  calc
    Real.exp a * (1 - Real.exp (-(2 * a)))
        ≤ Real.exp a * (2 * a) := by
          exact mul_le_mul_of_nonneg_left htail (Real.exp_pos _).le
    _ ≤ Real.exp B * (2 * a) := by
          exact mul_le_mul_of_nonneg_right
            (Real.exp_le_exp.mpr hB) (by nlinarith)
    _ = 2 * Real.exp B * a := by ring

/-- The lower coefficient in the zero-drift decomposition, scaled as it occurs
in the cross-displacement scalar. -/
noncomputable def scaledLowerPairing (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  Real.exp (-(2 * x) / τ) * truncatedPairing τ p q x

/-- The upper coefficient in the zero-drift decomposition, scaled as it occurs
in the cross-displacement scalar. -/
noncomputable def scaledUpperPairing (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  Real.exp ((2 * x) / τ) * upperPairing τ p q x

/-- The Milestone-4 balance defect.  The desired balance identity is exactly
`laplaceBalanceDefect τ p q x = 0`. -/
noncomputable def laplaceBalanceDefect (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  scaledUpperPairing τ p q x - scaledLowerPairing τ p q x

/-- The expected right derivative of the nonsmooth Laplace normalizer.  This
is the single remaining analytic derivative formula needed for Milestone 4. -/
noncomputable def laplaceKernelNormalizerRightDerivCoeff
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (1 / τ) *
    (-(Real.exp (-x / τ)) * lowerExpMass τ p x +
      Real.exp (x / τ) * upperExpMass τ p x)

/-- The derivative coefficient of the smooth displacement numerator, expressed
in one-sided coordinates. -/
noncomputable def laplaceDisplacementIntegralDerivCoeff
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (1 / τ) *
      (Real.exp (-x / τ) * lowerCompensatedMoment τ p x +
        Real.exp (x / τ) * upperCompensatedMoment τ p x) -
    (Real.exp (-x / τ) * lowerExpMass τ p x +
      Real.exp (x / τ) * upperExpMass τ p x)

/-- The already-certified classical derivative of the displacement numerator,
rewritten in the one-sided coordinates used by the balance identity. -/
theorem laplaceDisplacementIntegral_derivCoeff_eq
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ) :
    (1 / τ) * kernelNormalizer (laplaceCompanionKernel τ) p x -
        2 * kernelNormalizer (laplaceKernel τ) p x =
      laplaceDisplacementIntegralDerivCoeff τ p x := by
  rw [laplaceCompanionNormalizer_eq_lower_upper τ hτ p x,
    laplaceKernelNormalizer_eq_lower_upper τ hτ p x]
  unfold laplaceDisplacementIntegralDerivCoeff
  field_simp [hτ.ne']
  ring

private lemma Ioi_eq_Ioc_union_Ioi_of_le {x x' : ℝ} (hx : x ≤ x') :
    Set.Ioi x = Set.Ioc x x' ∪ Set.Ioi x' := by
  ext y
  constructor
  · intro hy
    by_cases hyx : y ≤ x'
    · exact Or.inl ⟨hy, hyx⟩
    · exact Or.inr (lt_of_not_ge hyx)
  · rintro (hy | hy)
    · exact hy.1
    · exact lt_of_le_of_lt hx hy

private lemma disjoint_Ioc_Ioi_same_right (x x' : ℝ) :
    Disjoint (Set.Ioc x x') (Set.Ioi x') := by
  rw [Set.disjoint_left]
  intro y hyIoc hyIoi
  exact not_lt_of_ge hyIoc.2 hyIoi

lemma upperExpMass_sub (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p]
    {x x' : ℝ} (hx : x ≤ x') :
    upperExpMass τ p x' - upperExpMass τ p x
      = -∫ y in Set.Ioc x x', Real.exp (-y / τ) ∂p := by
  have hInt : IntegrableOn (fun y : ℝ => Real.exp (-y / τ)) (Set.Ioi x) p :=
    integrable_upperExpKernel τ hτ p x
  have hIocInt : IntegrableOn (fun y : ℝ => Real.exp (-y / τ)) (Set.Ioc x x') p :=
    hInt.mono_set (by intro y hy; exact hy.1)
  have hIoiInt : IntegrableOn (fun y : ℝ => Real.exp (-y / τ)) (Set.Ioi x') p :=
    hInt.mono_set (by intro y hy; exact lt_of_le_of_lt hx hy)
  have hsplit :
      upperExpMass τ p x =
        ∫ y in Set.Ioc x x', Real.exp (-y / τ) ∂p + upperExpMass τ p x' := by
    unfold upperExpMass
    rw [Ioi_eq_Ioc_union_Ioi_of_le hx,
      setIntegral_union (disjoint_Ioc_Ioi_same_right x x') measurableSet_Ioi
        hIocInt hIoiInt]
  linarith

/-- The shrinking-strip remainder in the right-increment formula for the raw
Laplace normalizer.  It is zero to first order as `t ↓ x` from the right. -/
noncomputable def laplaceNormalizerRightRemainder
    (τ : ℝ) (p : Measure ℝ) (x t : ℝ) : ℝ :=
  ∫ y in Set.Ioc x t,
    (Real.exp (-(t - y) / τ) - Real.exp ((t - y) / τ)) ∂p

private lemma laplaceNormalizerRightRemainder_slope_bound
    (τ : ℝ) (hτ : 0 < τ) {x t y : ℝ}
    (htx : x < t) (ht1 : t < x + 1) (hy : y ∈ Set.Ioc x t) :
    |(t - x)⁻¹ *
        (Real.exp (-(t - y) / τ) - Real.exp ((t - y) / τ))|
      ≤ 2 * Real.exp (1 / τ) * (1 / τ) := by
  have hden : 0 < t - x := sub_pos.mpr htx
  have hy_nonneg : 0 ≤ t - y := sub_nonneg.mpr hy.2
  have hy_le : t - y ≤ t - x := by linarith [hy.1]
  have ht_le_one : t - x ≤ 1 := by linarith
  have ha : 0 ≤ (t - y) / τ := div_nonneg hy_nonneg hτ.le
  have hB : (t - y) / τ ≤ 1 / τ := by
    exact div_le_div_of_nonneg_right (by linarith) hτ.le
  have hscalar := abs_exp_neg_sub_exp_le ((t - y) / τ) (1 / τ) ha hB
  have hscalar' :
      |Real.exp (-(t - y) / τ) - Real.exp ((t - y) / τ)|
        ≤ 2 * Real.exp (1 / τ) * ((t - y) / τ) := by
    convert hscalar using 2
    ring_nf
  rw [abs_mul, abs_inv, abs_of_pos hden]
  have hstep :
      (t - x)⁻¹ *
          |Real.exp (-(t - y) / τ) - Real.exp ((t - y) / τ)|
        ≤ (t - x)⁻¹ * (2 * Real.exp (1 / τ) * ((t - y) / τ)) := by
    exact mul_le_mul_of_nonneg_left hscalar' (inv_nonneg.mpr hden.le)
  have hratio : ((t - y) / τ) / (t - x) ≤ 1 / τ := by
    rw [div_le_iff₀ hden]
    calc
      (t - y) / τ ≤ (t - x) / τ := div_le_div_of_nonneg_right hy_le hτ.le
      _ = 1 / τ * (t - x) := by ring
  calc
    (t - x)⁻¹ *
        |Real.exp (-(t - y) / τ) - Real.exp ((t - y) / τ)|
        ≤ (t - x)⁻¹ * (2 * Real.exp (1 / τ) * ((t - y) / τ)) := hstep
    _ = 2 * Real.exp (1 / τ) * (((t - y) / τ) / (t - x)) := by
        field_simp [hden.ne']
    _ ≤ 2 * Real.exp (1 / τ) * (1 / τ) := by
        exact mul_le_mul_of_nonneg_left hratio
          (mul_nonneg (by positivity) (Real.exp_pos _).le)

theorem hasDerivWithinAt_laplaceNormalizerRightRemainder
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    HasDerivWithinAt (fun t => laplaceNormalizerRightRemainder τ p x t) 0
      (Set.Ici x) x := by
  rw [hasDerivWithinAt_iff_tendsto_slope]
  have hdiff : Set.Ici x \ {x} = Set.Ioi x := by
    ext t
    simp only [Set.mem_sdiff, Set.mem_Ici, Set.mem_singleton_iff, Set.mem_Ioi]
    constructor
    · rintro ⟨hle, hne⟩
      exact lt_of_le_of_ne hle (Ne.symm hne)
    · intro hlt
      exact ⟨hlt.le, ne_of_gt hlt⟩
  rw [hdiff]
  let C : ℝ := 2 * Real.exp (1 / τ) * (1 / τ)
  have hC_nonneg : 0 ≤ C := by
    unfold C
    exact mul_nonneg (mul_nonneg (by positivity) (Real.exp_pos _).le)
      (one_div_pos.mpr hτ).le
  have hDCT :
      Tendsto
        (fun t => ∫ y,
          (Set.Ioc x t).indicator
            (fun y : ℝ =>
              (t - x)⁻¹ *
                (Real.exp (-(t - y) / τ) - Real.exp ((t - y) / τ))) y ∂p)
        (𝓝[Set.Ioi x] x) (𝓝 (∫ _ : ℝ, (0 : ℝ) ∂p)) := by
    refine tendsto_integral_filter_of_norm_le_const (μ := p)
      (l := 𝓝[Set.Ioi x] x) (G := ℝ) (f := fun _ : ℝ => (0 : ℝ))
      ?h_meas ?h_bound ?h_lim
    · refine Eventually.of_forall fun t => ?_
      exact ((by fun_prop :
        AEStronglyMeasurable
          (fun y : ℝ =>
            (t - x)⁻¹ *
              (Real.exp (-(t - y) / τ) - Real.exp ((t - y) / τ))) p).indicator
        measurableSet_Ioc)
    · refine ⟨C, ?_⟩
      filter_upwards [self_mem_nhdsWithin,
        nhdsWithin_le_nhds (Iio_mem_nhds (show x < x + 1 by linarith))]
        with t htx ht1
      exact ae_of_all p fun y => by
        by_cases hy : y ∈ Set.Ioc x t
        · rw [Set.indicator_of_mem hy, Real.norm_eq_abs]
          exact laplaceNormalizerRightRemainder_slope_bound τ hτ htx ht1 hy
        · rw [Set.indicator_of_notMem hy, norm_zero]
          exact hC_nonneg
    · exact ae_of_all p fun y => by
        by_cases hyx : x < y
        · refine tendsto_const_nhds.congr' ?_
          filter_upwards [nhdsWithin_le_nhds (Iio_mem_nhds hyx)] with t ht
          have hynot : y ∉ Set.Ioc x t := by
            intro hy
            exact not_le_of_gt ht hy.2
          rw [Set.indicator_of_notMem hynot]
        · have hyxle : y ≤ x := le_of_not_gt hyx
          refine tendsto_const_nhds.congr' ?_
          filter_upwards with t
          have hynot : y ∉ Set.Ioc x t := by
            intro hy
            exact not_lt_of_ge hyxle hy.1
          rw [Set.indicator_of_notMem hynot]
  have hDCT0 :
      Tendsto
        (fun t => ∫ y,
          (Set.Ioc x t).indicator
            (fun y : ℝ =>
              (t - x)⁻¹ *
                (Real.exp (-(t - y) / τ) - Real.exp ((t - y) / τ))) y ∂p)
        (𝓝[Set.Ioi x] x) (𝓝 0) := by
    simpa using hDCT
  refine hDCT0.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with t htx
  rw [slope_def_field]
  unfold laplaceNormalizerRightRemainder
  have hzero :
      (∫ y in Set.Ioc x x,
        (Real.exp (-(x - y) / τ) - Real.exp ((x - y) / τ)) ∂p) = 0 := by
    simp
  rw [hzero, sub_zero, div_eq_inv_mul, ← integral_const_mul,
    ← integral_indicator measurableSet_Ioc]

private lemma laplaceKernelNormalizer_eq_rightMain_add_remainder
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p]
    {x t : ℝ} (hxt : x ≤ t) :
    kernelNormalizer (laplaceKernel τ) p t =
      Real.exp (-t / τ) * lowerExpMass τ p x +
        Real.exp (t / τ) * upperExpMass τ p x +
          laplaceNormalizerRightRemainder τ p x t := by
  rw [laplaceKernelNormalizer_eq_lower_upper τ hτ p t]
  have hLower := lowerExpMass_sub τ hτ p hxt
  have hUpper := upperExpMass_sub τ hτ p hxt
  have hLm :
      lowerExpMass τ p t =
        lowerExpMass τ p x + ∫ y in Set.Ioc x t, Real.exp (y / τ) ∂p := by
    linarith
  have hUp :
      upperExpMass τ p t =
        upperExpMass τ p x - ∫ y in Set.Ioc x t, Real.exp (-y / τ) ∂p := by
    linarith
  rw [hLm, hUp]
  have hAbase : IntegrableOn (fun y : ℝ => Real.exp (y / τ)) (Set.Iic t) p :=
    integrable_lowerExpKernel τ hτ p t
  have hBbase : IntegrableOn (fun y : ℝ => Real.exp (-y / τ)) (Set.Ioi x) p :=
    integrable_upperExpKernel τ hτ p x
  have hAint : IntegrableOn (fun y : ℝ => Real.exp (y / τ)) (Set.Ioc x t) p :=
    hAbase.mono_set Set.Ioc_subset_Iic_self
  have hBint : IntegrableOn (fun y : ℝ => Real.exp (-y / τ)) (Set.Ioc x t) p :=
    hBbase.mono_set (by intro y hy; exact hy.1)
  have hleft :
      (∫ y in Set.Ioc x t, Real.exp (-(t - y) / τ) ∂p) =
        Real.exp (-t / τ) * ∫ y in Set.Ioc x t, Real.exp (y / τ) ∂p := by
    calc
      (∫ y in Set.Ioc x t, Real.exp (-(t - y) / τ) ∂p)
          = ∫ y in Set.Ioc x t, Real.exp (-t / τ) * Real.exp (y / τ) ∂p := by
              apply setIntegral_congr_fun measurableSet_Ioc
              intro y hy
              change Real.exp (-(t - y) / τ) =
                Real.exp (-t / τ) * Real.exp (y / τ)
              rw [← Real.exp_add]
              congr 1
              ring
      _ = Real.exp (-t / τ) * ∫ y in Set.Ioc x t, Real.exp (y / τ) ∂p := by
              rw [integral_const_mul]
  have hright :
      (∫ y in Set.Ioc x t, Real.exp ((t - y) / τ) ∂p) =
        Real.exp (t / τ) * ∫ y in Set.Ioc x t, Real.exp (-y / τ) ∂p := by
    calc
      (∫ y in Set.Ioc x t, Real.exp ((t - y) / τ) ∂p)
          = ∫ y in Set.Ioc x t, Real.exp (t / τ) * Real.exp (-y / τ) ∂p := by
              apply setIntegral_congr_fun measurableSet_Ioc
              intro y hy
              change Real.exp ((t - y) / τ) =
                Real.exp (t / τ) * Real.exp (-y / τ)
              rw [← Real.exp_add]
              congr 1
              ring
      _ = Real.exp (t / τ) * ∫ y in Set.Ioc x t, Real.exp (-y / τ) ∂p := by
              rw [integral_const_mul]
  have hR :
      laplaceNormalizerRightRemainder τ p x t =
        Real.exp (-t / τ) * ∫ y in Set.Ioc x t, Real.exp (y / τ) ∂p -
          Real.exp (t / τ) * ∫ y in Set.Ioc x t, Real.exp (-y / τ) ∂p := by
    have hLeftBase :
        IntegrableOn (fun y : ℝ => Real.exp (-t / τ) * Real.exp (y / τ))
          (Set.Ioc x t) p :=
      hAint.const_mul (Real.exp (-t / τ))
    have hLeftInt : IntegrableOn (fun y : ℝ => Real.exp (-(t - y) / τ)) (Set.Ioc x t) p :=
      hLeftBase.congr_fun (by
        intro y hy
        change Real.exp (-t / τ) * Real.exp (y / τ) =
          Real.exp (-(t - y) / τ)
        rw [← Real.exp_add]
        congr 1
        ring) measurableSet_Ioc
    have hRightBase :
        IntegrableOn (fun y : ℝ => Real.exp (t / τ) * Real.exp (-y / τ))
          (Set.Ioc x t) p :=
      hBint.const_mul (Real.exp (t / τ))
    have hRightInt : IntegrableOn (fun y : ℝ => Real.exp ((t - y) / τ)) (Set.Ioc x t) p :=
      hRightBase.congr_fun (by
        intro y hy
        change Real.exp (t / τ) * Real.exp (-y / τ) =
          Real.exp ((t - y) / τ)
        rw [← Real.exp_add]
        congr 1
        ring) measurableSet_Ioc
    unfold laplaceNormalizerRightRemainder
    rw [integral_sub hLeftInt hRightInt, hleft, hright]
  rw [hR]
  ring

/-- The raw 1-d Laplace normalizer has the expected right derivative for
arbitrary finite measures.  This is the analytic socket that closes
Milestone 4: atoms are handled by taking the derivative within `Ici x`, and
the shrinking strip contributes zero to first order. -/
theorem hasDerivWithinAt_Ici_laplaceKernelNormalizer
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    HasDerivWithinAt (fun t => kernelNormalizer (laplaceKernel τ) p t)
      (laplaceKernelNormalizerRightDerivCoeff τ p x) (Set.Ici x) x := by
  have hExpNeg : HasDerivAt (fun t : ℝ => Real.exp (-t / τ))
      (-(1 / τ) * Real.exp (-x / τ)) x := by
    have hlin : HasDerivAt (fun t : ℝ => -t / τ) (-(1 / τ)) x := by
      simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
        ((hasDerivAt_id x).const_mul (-(1 / τ)))
    simpa [mul_comm, mul_left_comm, mul_assoc] using hlin.exp
  have hExpPos : HasDerivAt (fun t : ℝ => Real.exp (t / τ))
      ((1 / τ) * Real.exp (x / τ)) x := by
    have hlin : HasDerivAt (fun t : ℝ => t / τ) (1 / τ) x := by
      simpa [div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using
        ((hasDerivAt_id x).const_mul (1 / τ))
    simpa [mul_comm, mul_left_comm, mul_assoc] using hlin.exp
  have hLower :=
    hExpNeg.hasDerivWithinAt.mul_const (lowerExpMass τ p x) (s := Set.Ici x)
  have hUpper :=
    hExpPos.hasDerivWithinAt.mul_const (upperExpMass τ p x) (s := Set.Ici x)
  have hRem := hasDerivWithinAt_laplaceNormalizerRightRemainder τ hτ p x
  have hsum := (hLower.add hUpper).add hRem
  have hvalue :
      (-(1 / τ) * Real.exp (-x / τ) * lowerExpMass τ p x +
            (1 / τ) * Real.exp (x / τ) * upperExpMass τ p x) +
          0 =
        laplaceKernelNormalizerRightDerivCoeff τ p x := by
    unfold laplaceKernelNormalizerRightDerivCoeff
    ring
  have hsum' : HasDerivWithinAt
      (fun t : ℝ =>
        Real.exp (-t / τ) * lowerExpMass τ p x +
          Real.exp (t / τ) * upperExpMass τ p x +
            laplaceNormalizerRightRemainder τ p x t)
      (laplaceKernelNormalizerRightDerivCoeff τ p x) (Set.Ici x) x := by
    rw [← hvalue]
    exact hsum
  refine hsum'.congr_of_eventuallyEq_of_mem ?_ (by simp : x ∈ Set.Ici x)
  filter_upwards [self_mem_nhdsWithin] with t ht
  exact laplaceKernelNormalizer_eq_rightMain_add_remainder τ hτ p ht

/-- Right-continuity of the upper pairing follows from right-continuity of the
upper one-sided transforms. -/
theorem upperPairing_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    ContinuousWithinAt (fun t => upperPairing τ p q t) (Set.Ici x) x := by
  unfold upperPairing
  exact ((upperCompensatedMoment_continuousWithinAt_Ici τ hτ p x).mul
      (upperExpMass_continuousWithinAt_Ici τ hτ q x)).sub
    ((upperCompensatedMoment_continuousWithinAt_Ici τ hτ q x).mul
      (upperExpMass_continuousWithinAt_Ici τ hτ p x))

theorem scaledLowerPairing_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    ContinuousWithinAt (fun t => scaledLowerPairing τ p q t) (Set.Ici x) x := by
  unfold scaledLowerPairing
  exact (by fun_prop : ContinuousWithinAt (fun t : ℝ => Real.exp (-(2 * t) / τ))
      (Set.Ici x) x).mul
    (truncatedPairing_continuousWithinAt_Ici τ hτ p q x)

theorem scaledUpperPairing_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    ContinuousWithinAt (fun t => scaledUpperPairing τ p q t) (Set.Ici x) x := by
  unfold scaledUpperPairing
  exact (by fun_prop : ContinuousWithinAt (fun t : ℝ => Real.exp ((2 * t) / τ))
      (Set.Ici x) x).mul
    (upperPairing_continuousWithinAt_Ici τ hτ p q x)

/-- Right-continuity of the balance defect.  This is the regularity needed to
upgrade a weak/a.e. balance identity to a pointwise one. -/
theorem laplaceBalanceDefect_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    ContinuousWithinAt (fun t => laplaceBalanceDefect τ p q t) (Set.Ici x) x := by
  unfold laplaceBalanceDefect
  exact (scaledUpperPairing_continuousWithinAt_Ici τ hτ p q x).sub
    (scaledLowerPairing_continuousWithinAt_Ici τ hτ p q x)

/-- Algebraic reduction of the Milestone-4 right-derivative identity to the
right derivative of the Laplace normalizer.

The displacement numerator is already classically differentiable.  Therefore,
once the nonsmooth normalizer has the expected right derivative, the
cross-displacement scalar has right derivative `(2/τ) * balanceDefect`. -/
theorem hasDerivWithinAt_Ici_crossDisplacement_of_kernelNormalizerRightDeriv
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hZ : ∀ r : Measure ℝ, IsProbabilityMeasure r →
      ∀ x : ℝ,
        HasDerivWithinAt (fun t => kernelNormalizer (laplaceKernel τ) r t)
          (laplaceKernelNormalizerRightDerivCoeff τ r x) (Set.Ici x) x) :
    ∀ x : ℝ,
      HasDerivWithinAt (fun t => laplaceCrossDisplacementScalar τ p q t)
        ((2 / τ) * laplaceBalanceDefect τ p q x) (Set.Ici x) x := by
  intro x
  letI hp : IsProbabilityMeasure p := ‹IsProbabilityMeasure p›
  letI hq : IsProbabilityMeasure q := ‹IsProbabilityMeasure q›
  have hZp := hZ p hp x
  have hZq := hZ q hq x
  have hDp₀ := (hasDerivAt_laplaceDisplacementIntegral τ hτ p x).hasDerivWithinAt
    (s := Set.Ici x)
  have hDq₀ := (hasDerivAt_laplaceDisplacementIntegral τ hτ q x).hasDerivWithinAt
    (s := Set.Ici x)
  have hDp : HasDerivWithinAt
      (fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂p)
      (laplaceDisplacementIntegralDerivCoeff τ p x) (Set.Ici x) x := by
    rw [← laplaceDisplacementIntegral_derivCoeff_eq τ hτ p x]
    exact hDp₀
  have hDq : HasDerivWithinAt
      (fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂q)
      (laplaceDisplacementIntegralDerivCoeff τ q x) (Set.Ici x) x := by
    rw [← laplaceDisplacementIntegral_derivCoeff_eq τ hτ q x]
    exact hDq₀
  have hprod₁ := hZq.mul hDp
  have hprod₂ := hZp.mul hDq
  have hsub := hprod₁.sub hprod₂
  unfold laplaceCrossDisplacementScalar
  change HasDerivWithinAt
    (((fun t : ℝ => kernelNormalizer (laplaceKernel τ) q t) *
        fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂p) -
      ((fun t : ℝ => kernelNormalizer (laplaceKernel τ) p t) *
        fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂q))
    ((2 / τ) * laplaceBalanceDefect τ p q x) (Set.Ici x) x
  have hvalue :
      laplaceKernelNormalizerRightDerivCoeff τ q x *
            (∫ y, laplaceWeightedDisplacement τ x y ∂p) +
          kernelNormalizer (laplaceKernel τ) q x *
            laplaceDisplacementIntegralDerivCoeff τ p x -
        (laplaceKernelNormalizerRightDerivCoeff τ p x *
            (∫ y, laplaceWeightedDisplacement τ x y ∂q) +
          kernelNormalizer (laplaceKernel τ) p x *
            laplaceDisplacementIntegralDerivCoeff τ q x)
        = (2 / τ) * laplaceBalanceDefect τ p q x := by
    rw [laplaceKernelNormalizer_eq_lower_upper τ hτ p x,
      laplaceKernelNormalizer_eq_lower_upper τ hτ q x,
      laplaceDisplacementIntegral_eq_lower_upper τ hτ p x,
      laplaceDisplacementIntegral_eq_lower_upper τ hτ q x]
    unfold laplaceKernelNormalizerRightDerivCoeff laplaceDisplacementIntegralDerivCoeff
      laplaceBalanceDefect scaledUpperPairing scaledLowerPairing
      truncatedPairing upperPairing
    have hneg : (Real.exp (-(x * τ⁻¹))) ^ 2 = Real.exp (-(x * τ⁻¹ * 2)) := by
      rw [sq, ← Real.exp_add]
      congr 1
      ring
    have hpos : (Real.exp (x * τ⁻¹)) ^ 2 = Real.exp (x * τ⁻¹ * 2) := by
      rw [sq, ← Real.exp_add]
      congr 1
      ring
    field_simp [hτ.ne']
    ring_nf
    rw [hneg, hpos]
    ring
  rw [← hvalue]
  exact hsub

/-- The unconditional Milestone-4 right-derivative identity for the
cross-displacement scalar. -/
theorem hasDerivWithinAt_Ici_laplaceCrossDisplacement
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] :
    ∀ x : ℝ,
      HasDerivWithinAt (fun t => laplaceCrossDisplacementScalar τ p q t)
        ((2 / τ) * laplaceBalanceDefect τ p q x) (Set.Ici x) x :=
  hasDerivWithinAt_Ici_crossDisplacement_of_kernelNormalizerRightDeriv τ hτ p q
    (fun r hr x => by
      letI : IsProbabilityMeasure r := hr
      exact hasDerivWithinAt_Ici_laplaceKernelNormalizer τ hτ r x)

/-- Correct one-sided Milestone-4 bridge.

If the unconditional **right-derivative** identity

`D⁺(laplaceCrossDisplacementScalar τ p q)(x) =
  (2/τ) * laplaceBalanceDefect τ p q x`

is available pointwise as a `HasDerivWithinAt` statement on `Ici x`, then zero
drift forces the balance identity pointwise.  This is the socket the remaining
weak/Stieltjes proof should target. -/
theorem laplaceBalance_identity_of_hasDerivWithinAt_Ici_crossDisplacement
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hderiv : ∀ x : ℝ,
      HasDerivWithinAt (fun t => laplaceCrossDisplacementScalar τ p q t)
        ((2 / τ) * laplaceBalanceDefect τ p q x) (Set.Ici x) x) :
    ∀ x : ℝ, scaledLowerPairing τ p q x = scaledUpperPairing τ p q x := by
  intro x
  have hscalar_zero : ∀ t : ℝ, laplaceCrossDisplacementScalar τ p q t = 0 := by
    intro t
    have hcross := (laplaceZeroDrift_iff_crossDisplacement τ hτ p q).mp hzero t
    simp only [smul_eq_mul] at hcross
    unfold laplaceCrossDisplacementScalar
    exact sub_eq_zero.mpr hcross
  have hconst :
      (fun t => laplaceCrossDisplacementScalar τ p q t) = fun _ : ℝ => (0 : ℝ) := by
    funext t
    exact hscalar_zero t
  have hzeroDeriv :
      HasDerivWithinAt (fun t => laplaceCrossDisplacementScalar τ p q t) 0
        (Set.Ici x) x := by
    rw [hconst]
    exact hasDerivWithinAt_const x (Set.Ici x) (0 : ℝ)
  have h₁ := (hderiv x).derivWithin (uniqueDiffWithinAt_Ici x)
  have h₂ := hzeroDeriv.derivWithin (uniqueDiffWithinAt_Ici x)
  have huniq : (2 / τ) * laplaceBalanceDefect τ p q x = 0 := by
    rw [← h₁, h₂]
  have hfactor : (2 : ℝ) / τ ≠ 0 := div_ne_zero two_ne_zero hτ.ne'
  have hdefect : laplaceBalanceDefect τ p q x = 0 := by
    rcases mul_eq_zero.mp huniq with hbad | hgood
    · exact absurd hbad hfactor
    · exact hgood
  unfold laplaceBalanceDefect at hdefect
  exact (sub_eq_zero.mp hdefect).symm

/-- **Milestone 4 headline.**  Zero raw 1-d Laplace drift forces the
pointwise balance identity

`exp (-(2*x)/τ) * 𝔞(x) = exp ((2*x)/τ) * 𝔠(x)`.

The proof is now unconditional for arbitrary probability measures: the raw
Laplace normalizer has the needed right derivative, the smooth displacement
numerator was already differentiable, and zero drift makes the
cross-displacement scalar identically zero. -/
theorem laplaceBalance_identity_of_zeroDrift
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    ∀ x : ℝ, scaledLowerPairing τ p q x = scaledUpperPairing τ p q x :=
  laplaceBalance_identity_of_hasDerivWithinAt_Ici_crossDisplacement τ hτ p q hzero
    (hasDerivWithinAt_Ici_laplaceCrossDisplacement τ hτ p q)

/-- Milestone-4 reduction to the single nonsmooth normalizer derivative.

The displacement numerator side is already differentiable.  Therefore, to get
the pointwise balance identity from zero drift, it is enough to prove the
right-derivative formula for the raw Laplace normalizer for arbitrary
probability measures. -/
theorem laplaceBalance_identity_of_kernelNormalizerRightDeriv
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hZ : ∀ r : Measure ℝ, IsProbabilityMeasure r →
      ∀ x : ℝ,
        HasDerivWithinAt (fun t => kernelNormalizer (laplaceKernel τ) r t)
          (laplaceKernelNormalizerRightDerivCoeff τ r x) (Set.Ici x) x) :
    ∀ x : ℝ, scaledLowerPairing τ p q x = scaledUpperPairing τ p q x :=
  laplaceBalance_identity_of_hasDerivWithinAt_Ici_crossDisplacement τ hτ p q hzero
    (hasDerivWithinAt_Ici_crossDisplacement_of_kernelNormalizerRightDeriv τ hτ p q hZ)

/-- Conditional Milestone-4 bridge.

If the unconditional derivative identity

`(laplaceCrossDisplacementScalar τ p q)' =
  (2/τ) * laplaceBalanceDefect τ p q`

is available pointwise, then zero drift forces the balance identity
`scaledLowerPairing = scaledUpperPairing` pointwise.  The remaining Milestone-4
work is to prove the derivative/weak-Fubini identity itself for arbitrary
probability measures. -/
theorem laplaceBalance_identity_of_hasDerivAt_crossDisplacement
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hderiv : ∀ x : ℝ,
      HasDerivAt (fun t => laplaceCrossDisplacementScalar τ p q t)
        ((2 / τ) * laplaceBalanceDefect τ p q x) x) :
    ∀ x : ℝ, scaledLowerPairing τ p q x = scaledUpperPairing τ p q x := by
  intro x
  have hscalar_zero : ∀ t : ℝ, laplaceCrossDisplacementScalar τ p q t = 0 := by
    intro t
    have hcross := (laplaceZeroDrift_iff_crossDisplacement τ hτ p q).mp hzero t
    simp only [smul_eq_mul] at hcross
    unfold laplaceCrossDisplacementScalar
    exact sub_eq_zero.mpr hcross
  have hconst :
      (fun t => laplaceCrossDisplacementScalar τ p q t) = fun _ : ℝ => (0 : ℝ) := by
    funext t
    exact hscalar_zero t
  have hzeroDeriv :
      HasDerivAt (fun t => laplaceCrossDisplacementScalar τ p q t) 0 x := by
    rw [hconst]
    exact hasDerivAt_const x (0 : ℝ)
  have huniq := (hderiv x).unique hzeroDeriv
  have hfactor : (2 : ℝ) / τ ≠ 0 := div_ne_zero two_ne_zero hτ.ne'
  have hdefect : laplaceBalanceDefect τ p q x = 0 := by
    rcases mul_eq_zero.mp huniq with hbad | hgood
    · exact absurd hbad hfactor
    · exact hgood
  unfold laplaceBalanceDefect at hdefect
  exact (sub_eq_zero.mp hdefect).symm

end DriftingIdentifiability
