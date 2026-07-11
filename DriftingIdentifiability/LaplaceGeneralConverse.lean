import DriftingIdentifiability.LaplaceWronskian
import Mathlib.MeasureTheory.Integral.Prod

/-!
# The general 1-d Laplace converse: one-sided transforms

This module starts Milestone 1 of `LaplaceGeneralConverseRoadmap.md`.

The atomic converse extracts a family of left-truncated bilinear pairings.
For arbitrary measures the analogous quantities are expressed through
one-sided transforms

* `P⁻(x) = ∫_{y≤x} exp (y/τ) dp(y)`,
* `P(x)  = ∫_{y≤x} (x-y) exp (y/τ) dp(y)`,
* `P⁺(x) = ∫_{y>x} exp (-y/τ) dp(y)`,
* `P̂(x)  = ∫_{y>x} (y-x) exp (-y/τ) dp(y)`.

The first formalized identities are the algebraic bracket formulas

`𝔞(x) = Q(x) P⁻(x) - P(x) Q⁻(x)`

and its upper-tail mirror.  The file now also contains the full four-region
decomposition of the cross-displacement field and the no-moment
right-continuity/tail-limit regularity needed for the lower truncated
pairing.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-! ## Scalar bounds for the one-sided kernels -/

/-- The elementary bound `s * exp (-s/τ) ≤ τ` for `s ≥ 0`, used to make the
compensated one-sided moments finite without moment assumptions. -/
lemma mul_exp_neg_le_general {τ : ℝ} (hτ : 0 < τ) {s : ℝ} (hs : 0 ≤ s) :
    s * Real.exp (-(1 / τ) * s) ≤ τ := by
  have h1 : s / τ + 1 ≤ Real.exp (s / τ) := Real.add_one_le_exp (s / τ)
  have h2 := mul_le_mul_of_nonneg_left h1 hτ.le
  have h3 : τ * (s / τ + 1) = s + τ := by field_simp [hτ.ne']
  have hexp : Real.exp (-(1 / τ) * s) = (Real.exp (s / τ))⁻¹ := by
    rw [← Real.exp_neg]
    congr 1
    field_simp [hτ.ne']
  rw [hexp, mul_inv_le_iff₀ (Real.exp_pos _)]
  nlinarith [Real.exp_pos (s / τ), hs]

/-! ## One-sided transforms -/

/-- Lower one-sided exponential mass `P⁻(x) = ∫_{y≤x} exp (y/τ) dp(y)`. -/
noncomputable def lowerExpMass (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iic x, Real.exp (y / τ) ∂p

/-- Lower compensated moment `P(x) = ∫_{y≤x} (x-y) exp (y/τ) dp(y)`. -/
noncomputable def lowerCompensatedMoment (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Iic x, (x - y) * Real.exp (y / τ) ∂p

/-- Upper one-sided exponential mass `P⁺(x) = ∫_{y>x} exp (-y/τ) dp(y)`. -/
noncomputable def upperExpMass (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Ioi x, Real.exp (-y / τ) ∂p

/-- Upper compensated moment `P̂(x) = ∫_{y>x} (y-x) exp (-y/τ) dp(y)`. -/
noncomputable def upperCompensatedMoment (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y in Set.Ioi x, (y - x) * Real.exp (-y / τ) ∂p

lemma integrable_lowerExpKernel
    (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    Integrable (fun y : ℝ => Real.exp (y / τ)) (p.restrict (Set.Iic x)) := by
  refine Integrable.of_bound (by fun_prop) (Real.exp (x / τ)) ?_
  filter_upwards [ae_restrict_mem measurableSet_Iic] with y hy
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  have hy' : y ≤ x := by simpa using hy
  exact Real.exp_le_exp.mpr (by gcongr)

lemma integrable_lowerCompKernel
    (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    Integrable (fun y : ℝ => (x - y) * Real.exp (y / τ))
      (p.restrict (Set.Iic x)) := by
  refine Integrable.of_bound (by fun_prop) (τ * Real.exp (x / τ)) ?_
  filter_upwards [ae_restrict_mem measurableSet_Iic] with y hy
  have hy' : y ≤ x := by simpa using hy
  have hs : 0 ≤ x - y := sub_nonneg.mpr hy'
  have hnonneg : 0 ≤ (x - y) * Real.exp (y / τ) :=
    mul_nonneg hs (Real.exp_pos _).le
  rw [Real.norm_eq_abs, abs_of_nonneg hnonneg]
  have hexp : Real.exp (y / τ) =
      Real.exp (x / τ) * Real.exp (-(1 / τ) * (x - y)) := by
    rw [← Real.exp_add]
    congr 1
    field_simp [hτ.ne']
    ring
  calc
    (x - y) * Real.exp (y / τ)
        = Real.exp (x / τ) * ((x - y) * Real.exp (-(1 / τ) * (x - y))) := by
            rw [hexp]
            ring
    _ ≤ Real.exp (x / τ) * τ := by
            exact mul_le_mul_of_nonneg_left
              (mul_exp_neg_le_general hτ hs) (Real.exp_pos _).le
    _ = τ * Real.exp (x / τ) := by ring

private lemma integrable_upperExpKernel
    (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    Integrable (fun y : ℝ => Real.exp (-y / τ)) (p.restrict (Set.Ioi x)) := by
  refine Integrable.of_bound (by fun_prop) (Real.exp (-x / τ)) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with y hy
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  have hy' : x < y := by simpa using hy
  have hyle : x ≤ y := le_of_lt hy'
  exact Real.exp_le_exp.mpr (by gcongr)

private lemma integrable_upperCompKernel
    (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    Integrable (fun y : ℝ => (y - x) * Real.exp (-y / τ))
      (p.restrict (Set.Ioi x)) := by
  refine Integrable.of_bound (by fun_prop) (τ * Real.exp (-x / τ)) ?_
  filter_upwards [ae_restrict_mem measurableSet_Ioi] with y hy
  have hy' : x < y := by simpa using hy
  have hs : 0 ≤ y - x := sub_nonneg.mpr (le_of_lt hy')
  have hnonneg : 0 ≤ (y - x) * Real.exp (-y / τ) :=
    mul_nonneg hs (Real.exp_pos _).le
  rw [Real.norm_eq_abs, abs_of_nonneg hnonneg]
  have hexp : Real.exp (-y / τ) =
      Real.exp (-x / τ) * Real.exp (-(1 / τ) * (y - x)) := by
    rw [← Real.exp_add]
    congr 1
    field_simp [hτ.ne']
    ring
  calc
    (y - x) * Real.exp (-y / τ)
        = Real.exp (-x / τ) * ((y - x) * Real.exp (-(1 / τ) * (y - x))) := by
            rw [hexp]
            ring
    _ ≤ Real.exp (-x / τ) * τ := by
            exact mul_le_mul_of_nonneg_left
              (mul_exp_neg_le_general hτ hs) (Real.exp_pos _).le
    _ = τ * Real.exp (-x / τ) := by ring

private lemma tendsto_measure_Iic_atBot_zero
    (p : Measure ℝ) [IsFiniteMeasure p] :
    Tendsto (fun x : ℝ => p (Set.Iic x)) atBot (𝓝 (0 : ENNReal)) := by
  have hfin : ∃ i : ℝ, p (Set.Iic i) ≠ ⊤ := by
    exact ⟨0, measure_ne_top p (Set.Iic 0)⟩
  have hmain :
      Tendsto (p ∘ fun x : ℝ => Set.Iic x) atBot
        (𝓝 (p (⋂ x : ℝ, Set.Iic x))) :=
    tendsto_measure_iInter_atBot (μ := p) (s := fun x : ℝ => Set.Iic x)
      (fun _ => measurableSet_Iic.nullMeasurableSet) monotone_Iic hfin
  have hInter : (⋂ x : ℝ, Set.Iic x) = ∅ := by
    rw [iInter_Iic_eq_empty_iff]
    exact not_bddBelow_iff.mpr (fun a => ⟨a - 1, by simp, by linarith⟩)
  simpa [Function.comp_def, hInter] using hmain

/-! ## No-moment regularity of the lower transforms -/

theorem lowerExpMass_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    ContinuousWithinAt (fun t => lowerExpMass τ p t) (Set.Ici x) x := by
  unfold ContinuousWithinAt lowerExpMass
  simp_rw [← integral_indicator measurableSet_Iic]
  refine tendsto_integral_filter_of_norm_le_const (μ := p) ?h_meas ?h_bound ?h_lim
  · exact Eventually.of_forall fun t =>
      ((by fun_prop :
        AEStronglyMeasurable (fun y : ℝ => Real.exp (y / τ)) p).indicator measurableSet_Iic)
  · refine ⟨Real.exp ((x + 1) / τ), ?_⟩
    filter_upwards [nhdsWithin_le_nhds (Iio_mem_nhds (show x < x + 1 by linarith))]
      with t ht
    exact ae_of_all p fun y => by
      by_cases hy : y ≤ t
      · rw [Set.indicator_of_mem (show y ∈ Set.Iic t by simpa using hy)
            (fun y : ℝ => Real.exp (y / τ)),
          Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
        exact Real.exp_le_exp.mpr
          (div_le_div_of_nonneg_right (hy.trans (le_of_lt ht)) hτ.le)
      · rw [Set.indicator_of_notMem (show y ∉ Set.Iic t by simpa using hy)
            (fun y : ℝ => Real.exp (y / τ)),
          norm_zero]
        exact Real.exp_pos _ |>.le
  · exact ae_of_all p fun y => by
      by_cases hy : y ≤ x
      · have htarget :
            (Set.Iic x).indicator (fun y : ℝ => Real.exp (y / τ)) y =
              Real.exp (y / τ) := by
          rw [Set.indicator_of_mem (show y ∈ Set.Iic x by simpa using hy)
            (fun y : ℝ => Real.exp (y / τ))]
        rw [htarget]
        refine tendsto_const_nhds.congr' ?_
        filter_upwards [self_mem_nhdsWithin] with t ht
        rw [Set.indicator_of_mem (show y ∈ Set.Iic t by simpa using hy.trans ht)
          (fun y : ℝ => Real.exp (y / τ))]
      · have hxy : x < y := lt_of_not_ge hy
        have htarget :
            (Set.Iic x).indicator (fun y : ℝ => Real.exp (y / τ)) y = 0 := by
          rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hy)
            (fun y : ℝ => Real.exp (y / τ))]
        rw [htarget]
        refine tendsto_const_nhds.congr' ?_
        filter_upwards [nhdsWithin_le_nhds (Iio_mem_nhds hxy)] with t ht
        have hyt : ¬ y ≤ t := not_le.mpr ht
        rw [Set.indicator_of_notMem (show y ∉ Set.Iic t by simpa using hyt)
          (fun y : ℝ => Real.exp (y / τ))]

theorem lowerCompensatedMoment_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    ContinuousWithinAt (fun t => lowerCompensatedMoment τ p t) (Set.Ici x) x := by
  unfold ContinuousWithinAt lowerCompensatedMoment
  simp_rw [← integral_indicator measurableSet_Iic]
  refine tendsto_integral_filter_of_norm_le_const (μ := p) ?h_meas ?h_bound ?h_lim
  · refine Eventually.of_forall fun t => ?_
    exact ((by fun_prop :
      AEStronglyMeasurable (fun y : ℝ => (t - y) * Real.exp (y / τ)) p).indicator
        measurableSet_Iic)
  · refine ⟨τ * Real.exp ((x + 1) / τ), ?_⟩
    filter_upwards [nhdsWithin_le_nhds (Iio_mem_nhds (show x < x + 1 by linarith))]
      with t ht
    exact ae_of_all p fun y => by
      by_cases hy : y ≤ t
      · have hs : 0 ≤ t - y := sub_nonneg.mpr hy
        have hnonneg : 0 ≤ (t - y) * Real.exp (y / τ) :=
          mul_nonneg hs (Real.exp_pos _).le
        rw [Set.indicator_of_mem (show y ∈ Set.Iic t by simpa using hy)
              (fun y : ℝ => (t - y) * Real.exp (y / τ)),
            Real.norm_eq_abs, abs_of_nonneg hnonneg]
        have hbasic :
            (t - y) * Real.exp (y / τ) ≤ τ * Real.exp (t / τ) := by
          have hexp : Real.exp (y / τ) =
              Real.exp (t / τ) * Real.exp (-(1 / τ) * (t - y)) := by
            rw [← Real.exp_add]
            congr 1
            field_simp [hτ.ne']
            ring
          calc
            (t - y) * Real.exp (y / τ)
                = Real.exp (t / τ) *
                    ((t - y) * Real.exp (-(1 / τ) * (t - y))) := by
                    rw [hexp]
                    ring
            _ ≤ Real.exp (t / τ) * τ := by
                    exact mul_le_mul_of_nonneg_left
                      (mul_exp_neg_le_general hτ hs) (Real.exp_pos _).le
            _ = τ * Real.exp (t / τ) := by ring
        exact hbasic.trans (by
          have ht' : t ≤ x + 1 := le_of_lt ht
          exact mul_le_mul_of_nonneg_left
            (Real.exp_le_exp.mpr (div_le_div_of_nonneg_right ht' hτ.le)) hτ.le)
      · rw [Set.indicator_of_notMem (show y ∉ Set.Iic t by simpa using hy)
              (fun y : ℝ => (t - y) * Real.exp (y / τ)),
            norm_zero]
        exact mul_nonneg hτ.le (Real.exp_pos _).le
  · exact ae_of_all p fun y => by
      by_cases hy : y ≤ x
      · have hcont :
            Tendsto (fun t : ℝ => (t - y) * Real.exp (y / τ))
              (nhdsWithin x (Set.Ici x))
              (𝓝 ((x - y) * Real.exp (y / τ))) := by
          exact ((tendsto_id'.2 nhdsWithin_le_nhds).sub tendsto_const_nhds).mul
            tendsto_const_nhds
        simpa [Set.indicator_of_mem (show y ∈ Set.Iic x by simpa using hy)
              (fun y : ℝ => (x - y) * Real.exp (y / τ))] using
          hcont.congr' (by
            filter_upwards [self_mem_nhdsWithin] with t ht
            rw [Set.indicator_of_mem (show y ∈ Set.Iic t by simpa using hy.trans ht)
                  (fun y : ℝ => (t - y) * Real.exp (y / τ))])
      · have hxy : x < y := lt_of_not_ge hy
        have htarget :
            (Set.Iic x).indicator (fun y : ℝ => (x - y) * Real.exp (y / τ)) y = 0 := by
          rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hy)
            (fun y : ℝ => (x - y) * Real.exp (y / τ))]
        rw [htarget]
        refine tendsto_const_nhds.congr' ?_
        filter_upwards [nhdsWithin_le_nhds (Iio_mem_nhds hxy)] with t ht
        have hyt : ¬ y ≤ t := not_le.mpr ht
        rw [Set.indicator_of_notMem (show y ∉ Set.Iic t by simpa using hyt)
          (fun y : ℝ => (t - y) * Real.exp (y / τ))]

theorem lowerExpMass_tendsto_atBot_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] :
    Tendsto (fun x => lowerExpMass τ p x) atBot (𝓝 0) := by
  unfold lowerExpMass
  simp_rw [← integral_indicator measurableSet_Iic]
  have h_event : ∀ᶠ x : ℝ in atBot, x ≤ 0 := Iic_mem_atBot 0
  have hDCT :
      Tendsto
        (fun x => ∫ y, (Set.Iic x).indicator (fun y : ℝ => Real.exp (y / τ)) y ∂p)
        atBot (𝓝 (∫ _ : ℝ, (0 : ℝ) ∂p)) := by
    refine tendsto_integral_filter_of_norm_le_const (μ := p) (l := (atBot : Filter ℝ))
      (G := ℝ) (f := fun _ : ℝ => (0 : ℝ)) ?h_meas ?h_bound ?h_lim
    · exact Eventually.of_forall fun x =>
        ((by fun_prop :
          AEStronglyMeasurable (fun y : ℝ => Real.exp (y / τ)) p).indicator measurableSet_Iic)
    · refine ⟨1, ?_⟩
      filter_upwards [h_event] with x hx
      exact ae_of_all p fun y => by
        by_cases hy : y ≤ x
        · have hy0 : y ≤ 0 := hy.trans hx
          rw [Set.indicator_of_mem (show y ∈ Set.Iic x by simpa using hy)
                (fun y : ℝ => Real.exp (y / τ)),
              Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
          calc
            Real.exp (y / τ) ≤ Real.exp (0 / τ) := by
              exact Real.exp_le_exp.mpr (div_le_div_of_nonneg_right hy0 hτ.le)
            _ = 1 := by simp
        · rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hy)
                (fun y : ℝ => Real.exp (y / τ)),
              norm_zero]
          exact zero_le_one
    · exact ae_of_all p fun y => by
        refine tendsto_nhds_of_eventually_eq ?_
        filter_upwards [(Iio_mem_atBot y : Set.Iio y ∈ (atBot : Filter ℝ))] with x hx
        have hyx : ¬ y ≤ x := not_le.mpr hx
        rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hyx)
          (fun y : ℝ => Real.exp (y / τ))]
  simpa using hDCT

theorem lowerCompensatedMoment_tendsto_atBot_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] :
    Tendsto (fun x => lowerCompensatedMoment τ p x) atBot (𝓝 0) := by
  unfold lowerCompensatedMoment
  simp_rw [← integral_indicator measurableSet_Iic]
  have h_event : ∀ᶠ x : ℝ in atBot, x ≤ 0 := Iic_mem_atBot 0
  have hDCT :
      Tendsto
        (fun x => ∫ y,
          (Set.Iic x).indicator (fun y : ℝ => (x - y) * Real.exp (y / τ)) y ∂p)
        atBot (𝓝 (∫ _ : ℝ, (0 : ℝ) ∂p)) := by
    refine tendsto_integral_filter_of_norm_le_const (μ := p) (l := (atBot : Filter ℝ))
      (G := ℝ) (f := fun _ : ℝ => (0 : ℝ)) ?h_meas ?h_bound ?h_lim
    · refine Eventually.of_forall fun x => ?_
      exact ((by fun_prop :
        AEStronglyMeasurable (fun y : ℝ => (x - y) * Real.exp (y / τ)) p).indicator
          measurableSet_Iic)
    · refine ⟨τ, ?_⟩
      filter_upwards [h_event] with x hx
      exact ae_of_all p fun y => by
        by_cases hy : y ≤ x
        · have hs : 0 ≤ x - y := sub_nonneg.mpr hy
          have hnonneg : 0 ≤ (x - y) * Real.exp (y / τ) :=
            mul_nonneg hs (Real.exp_pos _).le
          rw [Set.indicator_of_mem (show y ∈ Set.Iic x by simpa using hy)
                (fun y : ℝ => (x - y) * Real.exp (y / τ)),
              Real.norm_eq_abs, abs_of_nonneg hnonneg]
          have hbasic :
              (x - y) * Real.exp (y / τ) ≤ τ * Real.exp (x / τ) := by
            have hexp : Real.exp (y / τ) =
                Real.exp (x / τ) * Real.exp (-(1 / τ) * (x - y)) := by
              rw [← Real.exp_add]
              congr 1
              field_simp [hτ.ne']
              ring
            calc
              (x - y) * Real.exp (y / τ)
                  = Real.exp (x / τ) *
                      ((x - y) * Real.exp (-(1 / τ) * (x - y))) := by
                      rw [hexp]
                      ring
              _ ≤ Real.exp (x / τ) * τ := by
                      exact mul_le_mul_of_nonneg_left
                        (mul_exp_neg_le_general hτ hs) (Real.exp_pos _).le
              _ = τ * Real.exp (x / τ) := by ring
          exact hbasic.trans (by
            calc
              τ * Real.exp (x / τ) ≤ τ * Real.exp (0 / τ) := by
                exact mul_le_mul_of_nonneg_left
                  (Real.exp_le_exp.mpr (div_le_div_of_nonneg_right hx hτ.le)) hτ.le
              _ = τ := by simp)
        · rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hy)
                (fun y : ℝ => (x - y) * Real.exp (y / τ)),
              norm_zero]
          exact hτ.le
    · exact ae_of_all p fun y => by
        refine tendsto_nhds_of_eventually_eq ?_
        filter_upwards [(Iio_mem_atBot y : Set.Iio y ∈ (atBot : Filter ℝ))] with x hx
        have hyx : ¬ y ≤ x := not_le.mpr hx
        rw [Set.indicator_of_notMem (show y ∉ Set.Iic x by simpa using hyx)
          (fun y : ℝ => (x - y) * Real.exp (y / τ))]
  simpa using hDCT

/-! ## One-sided formulas for the Laplace kernel and numerator -/

private lemma laplaceKernel_left_of_le (τ x y : ℝ) (hy : y ≤ x) :
    laplaceKernel τ x y = Real.exp (-x / τ) * Real.exp (y / τ) := by
  unfold laplaceKernel
  rw [Real.norm_eq_abs, abs_of_nonneg (sub_nonneg.mpr hy)]
  rw [show -(1 / τ) * (x - y) = -x / τ + y / τ by ring, Real.exp_add]

private lemma laplaceKernel_right_of_lt (τ x y : ℝ) (hy : x < y) :
    laplaceKernel τ x y = Real.exp (x / τ) * Real.exp (-y / τ) := by
  unfold laplaceKernel
  rw [Real.norm_eq_abs, abs_of_neg (sub_neg.mpr hy)]
  rw [show -(1 / τ) * -(x - y) = x / τ + -y / τ by ring, Real.exp_add]

private lemma laplaceWeightedDisplacement_left_of_le (τ x y : ℝ) (hy : y ≤ x) :
    laplaceWeightedDisplacement τ x y =
      -(Real.exp (-x / τ)) * ((x - y) * Real.exp (y / τ)) := by
  unfold laplaceWeightedDisplacement
  rw [laplaceKernel_left_of_le τ x y hy, smul_eq_mul]
  ring

private lemma laplaceWeightedDisplacement_right_of_lt (τ x y : ℝ) (hy : x < y) :
    laplaceWeightedDisplacement τ x y =
      Real.exp (x / τ) * ((y - x) * Real.exp (-y / τ)) := by
  unfold laplaceWeightedDisplacement
  rw [laplaceKernel_right_of_lt τ x y hy, smul_eq_mul]
  ring

theorem laplaceKernelNormalizer_eq_lower_upper
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    kernelNormalizer (laplaceKernel τ) p x =
      Real.exp (-x / τ) * lowerExpMass τ p x +
        Real.exp (x / τ) * upperExpMass τ p x := by
  have hInt : Integrable (fun y => laplaceKernel τ x y) p :=
    laplaceKernel_integrable p τ hτ x
  unfold kernelNormalizer lowerExpMass upperExpMass
  rw [← setIntegral_univ,
    ← Iic_union_Ioi (a := x),
    setIntegral_union (Iic_disjoint_Ioi le_rfl) measurableSet_Ioi
      (hInt.integrableOn) (hInt.integrableOn)]
  congr 1
  · rw [← integral_const_mul]
    apply setIntegral_congr_fun measurableSet_Iic
    intro y hy
    have hy' : y ≤ x := by simpa using hy
    exact laplaceKernel_left_of_le τ x y hy'
  · rw [← integral_const_mul]
    apply setIntegral_congr_fun measurableSet_Ioi
    intro y hy
    have hy' : x < y := by simpa using hy
    exact laplaceKernel_right_of_lt τ x y hy'

theorem laplaceDisplacementIntegral_eq_lower_upper
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    (∫ y, laplaceWeightedDisplacement τ x y ∂p) =
      -(Real.exp (-x / τ)) * lowerCompensatedMoment τ p x +
        Real.exp (x / τ) * upperCompensatedMoment τ p x := by
  have hInt : Integrable (fun y => laplaceWeightedDisplacement τ x y) p :=
    laplaceWeightedDisplacement_integrable τ hτ p x
  unfold lowerCompensatedMoment upperCompensatedMoment
  rw [← setIntegral_univ,
    ← Iic_union_Ioi (a := x),
    setIntegral_union (Iic_disjoint_Ioi le_rfl) measurableSet_Ioi
      (hInt.integrableOn) (hInt.integrableOn)]
  congr 1
  · rw [← integral_const_mul]
    apply setIntegral_congr_fun measurableSet_Iic
    intro y hy
    have hy' : y ≤ x := by simpa using hy
    exact laplaceWeightedDisplacement_left_of_le τ x y hy'
  · rw [← integral_const_mul]
    apply setIntegral_congr_fun measurableSet_Ioi
    intro y hy
    have hy' : x < y := by simpa using hy
    exact laplaceWeightedDisplacement_right_of_lt τ x y hy'

/-! ## Bracket pairings -/

/-- The left-truncated pairing
`𝔞(x) = Q(x) P⁻(x) - P(x) Q⁻(x)`.

This is the continuous analogue of the atomic `frakA` coefficient family. -/
noncomputable def truncatedPairing (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  lowerCompensatedMoment τ q x * lowerExpMass τ p x -
    lowerCompensatedMoment τ p x * lowerExpMass τ q x

/-- The upper-tail mirror
`𝔠(x) = P̂(x) Q⁺(x) - Q̂(x) P⁺(x)`. -/
noncomputable def upperPairing (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  upperCompensatedMoment τ p x * upperExpMass τ q x -
    upperCompensatedMoment τ q x * upperExpMass τ p x

/-- The mixed-region coefficient in the four-region decomposition. -/
noncomputable def middlePairing (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  lowerExpMass τ q x * upperCompensatedMoment τ p x -
    upperExpMass τ q x * lowerCompensatedMoment τ p x -
      lowerExpMass τ p x * upperCompensatedMoment τ q x +
        upperExpMass τ p x * lowerCompensatedMoment τ q x

/-- Cross-displacement scalar
`Z_q D_p - Z_p D_q`, the scalar version of the Stage-1 zero-drift
cross-displacement equation on the real line. -/
noncomputable def laplaceCrossDisplacementScalar
    (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  kernelNormalizer (laplaceKernel τ) q x *
      (∫ y, laplaceWeightedDisplacement τ x y ∂p) -
    kernelNormalizer (laplaceKernel τ) p x *
      (∫ y, laplaceWeightedDisplacement τ x y ∂q)

/-- **Milestone-1 four-region decomposition.**  The cross-displacement scalar
splits into the lower, mixed, and upper one-sided coefficients. -/
theorem laplaceCrossDisplacementScalar_decomposition
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    laplaceCrossDisplacementScalar τ p q x =
      (Real.exp (-x / τ)) ^ 2 * truncatedPairing τ p q x +
        middlePairing τ p q x +
          (Real.exp (x / τ)) ^ 2 * upperPairing τ p q x := by
  unfold laplaceCrossDisplacementScalar
  rw [laplaceKernelNormalizer_eq_lower_upper τ hτ q x,
    laplaceKernelNormalizer_eq_lower_upper τ hτ p x,
    laplaceDisplacementIntegral_eq_lower_upper τ hτ p x,
    laplaceDisplacementIntegral_eq_lower_upper τ hτ q x]
  unfold truncatedPairing upperPairing middlePairing
  set em : ℝ := Real.exp (-x / τ)
  set ep : ℝ := Real.exp (x / τ)
  set Pm : ℝ := lowerExpMass τ p x
  set Qm : ℝ := lowerExpMass τ q x
  set P : ℝ := lowerCompensatedMoment τ p x
  set Q : ℝ := lowerCompensatedMoment τ q x
  set Pp : ℝ := upperExpMass τ p x
  set Qp : ℝ := upperExpMass τ q x
  set Ph : ℝ := upperCompensatedMoment τ p x
  set Qh : ℝ := upperCompensatedMoment τ q x
  have hcancel : em * ep = 1 := by
    change Real.exp (-x / τ) * Real.exp (x / τ) = 1
    rw [← Real.exp_add]
    rw [show -x / τ + x / τ = 0 by ring, Real.exp_zero]
  change (em * Qm + ep * Qp) * (-em * P + ep * Ph) -
      (em * Pm + ep * Pp) * (-em * Q + ep * Qh) =
    em ^ 2 * (Q * Pm - P * Qm) +
      (Qm * Ph - Qp * P - Pm * Qh + Pp * Q) +
        ep ^ 2 * (Ph * Qp - Qh * Pp)
  calc
    (em * Qm + ep * Qp) * (-em * P + ep * Ph) -
        (em * Pm + ep * Pp) * (-em * Q + ep * Qh)
        = em ^ 2 * (Q * Pm - P * Qm) +
            (em * ep) * (Qm * Ph - Qp * P - Pm * Qh + Pp * Q) +
              ep ^ 2 * (Ph * Qp - Qh * Pp) := by
            ring
    _ = em ^ 2 * (Q * Pm - P * Qm) +
          (Qm * Ph - Qp * P - Pm * Qh + Pp * Q) +
            ep ^ 2 * (Ph * Qp - Qh * Pp) := by
          rw [hcancel, one_mul]

/-- Same four-region decomposition, written in the roadmap's exponential
notation. -/
theorem laplaceCrossDisplacementScalar_decomposition_exp
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    laplaceCrossDisplacementScalar τ p q x =
      Real.exp (-(2 * x) / τ) * truncatedPairing τ p q x +
        middlePairing τ p q x +
          Real.exp ((2 * x) / τ) * upperPairing τ p q x := by
  rw [laplaceCrossDisplacementScalar_decomposition τ hτ p q x]
  have hneg : (Real.exp (-x / τ)) ^ 2 = Real.exp (-(2 * x) / τ) := by
    rw [sq, ← Real.exp_add]
    congr 1
    ring
  have hpos : (Real.exp (x / τ)) ^ 2 = Real.exp ((2 * x) / τ) := by
    rw [sq, ← Real.exp_add]
    congr 1
    ring
  rw [hneg, hpos]

/-- Zero drift implies the decomposed lower/mixed/upper coefficient identity. -/
theorem laplaceZeroDrift_decomposition
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ) :
      (Real.exp (-x / τ)) ^ 2 * truncatedPairing τ p q x +
        middlePairing τ p q x +
          (Real.exp (x / τ)) ^ 2 * upperPairing τ p q x = 0 := by
  have hcross := (laplaceZeroDrift_iff_crossDisplacement τ hτ p q).mp hzero x
  simp only [smul_eq_mul] at hcross
  have hscalar : laplaceCrossDisplacementScalar τ p q x = 0 := by
    unfold laplaceCrossDisplacementScalar
    exact sub_eq_zero.mpr hcross
  rw [laplaceCrossDisplacementScalar_decomposition τ hτ p q x] at hscalar
  exact hscalar

/-- Zero-drift coefficient identity in the roadmap's exponential notation. -/
theorem laplaceZeroDrift_decomposition_exp
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ) :
      Real.exp (-(2 * x) / τ) * truncatedPairing τ p q x +
        middlePairing τ p q x +
          Real.exp ((2 * x) / τ) * upperPairing τ p q x = 0 := by
  have hcross := (laplaceZeroDrift_iff_crossDisplacement τ hτ p q).mp hzero x
  simp only [smul_eq_mul] at hcross
  have hscalar : laplaceCrossDisplacementScalar τ p q x = 0 := by
    unfold laplaceCrossDisplacementScalar
    exact sub_eq_zero.mpr hcross
  rw [laplaceCrossDisplacementScalar_decomposition_exp τ hτ p q x] at hscalar
  exact hscalar

/-! ## Regularity assembly lemmas -/

/-- Right-continuity of the four lower one-sided transforms implies
right-continuity of the truncated pairing.  The remaining Milestone-1
analytic work is to prove the four transform hypotheses without moment
assumptions by interval-shrinking. -/
theorem truncatedPairing_continuousWithinAt_of_lowerTransforms
    (τ : ℝ) (p q : Measure ℝ) (x : ℝ)
    (hPm : ContinuousWithinAt (fun t => lowerExpMass τ p t) (Set.Ici x) x)
    (hQm : ContinuousWithinAt (fun t => lowerExpMass τ q t) (Set.Ici x) x)
    (hP : ContinuousWithinAt (fun t => lowerCompensatedMoment τ p t) (Set.Ici x) x)
    (hQ : ContinuousWithinAt (fun t => lowerCompensatedMoment τ q t) (Set.Ici x) x) :
    ContinuousWithinAt (fun t => truncatedPairing τ p q t) (Set.Ici x) x := by
  unfold truncatedPairing
  exact (hQ.mul hPm).sub (hP.mul hQm)

/-- If the lower one-sided transforms vanish at `-∞`, so does the truncated
pairing. -/
theorem truncatedPairing_tendsto_atBot_zero_of_lowerTransforms
    (τ : ℝ) (p q : Measure ℝ)
    (hPm : Tendsto (fun x => lowerExpMass τ p x) atBot (𝓝 0))
    (hQm : Tendsto (fun x => lowerExpMass τ q x) atBot (𝓝 0))
    (hP : Tendsto (fun x => lowerCompensatedMoment τ p x) atBot (𝓝 0))
    (hQ : Tendsto (fun x => lowerCompensatedMoment τ q x) atBot (𝓝 0)) :
    Tendsto (fun x => truncatedPairing τ p q x) atBot (𝓝 0) := by
  unfold truncatedPairing
  simpa using (hQ.mul hPm).sub (hP.mul hQm)

theorem truncatedPairing_continuousWithinAt_Ici
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    ContinuousWithinAt (fun t => truncatedPairing τ p q t) (Set.Ici x) x :=
  truncatedPairing_continuousWithinAt_of_lowerTransforms τ p q x
    (lowerExpMass_continuousWithinAt_Ici τ hτ p x)
    (lowerExpMass_continuousWithinAt_Ici τ hτ q x)
    (lowerCompensatedMoment_continuousWithinAt_Ici τ hτ p x)
    (lowerCompensatedMoment_continuousWithinAt_Ici τ hτ q x)

theorem truncatedPairing_tendsto_atBot_zero
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] :
    Tendsto (fun x => truncatedPairing τ p q x) atBot (𝓝 0) :=
  truncatedPairing_tendsto_atBot_zero_of_lowerTransforms τ p q
    (lowerExpMass_tendsto_atBot_zero τ hτ p)
    (lowerExpMass_tendsto_atBot_zero τ hτ q)
    (lowerCompensatedMoment_tendsto_atBot_zero τ hτ p)
    (lowerCompensatedMoment_tendsto_atBot_zero τ hτ q)

/-- Product-integral form of the lower bracket after splitting the kernel
algebra into separated products.  This is intentionally separated from the
direct double-integral form; the latter requires the bounded-integrability
bookkeeping for the conceptual integrand. -/
noncomputable def lowerBracketProductIntegral
    (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  (∫ z in Set.Iic x ×ˢ Set.Iic x,
      Real.exp (z.1 / τ) * ((x - z.2) * Real.exp (z.2 / τ)) ∂(p.prod q)) -
    ∫ z in Set.Iic x ×ˢ Set.Iic x,
      ((x - z.1) * Real.exp (z.1 / τ)) * Real.exp (z.2 / τ) ∂(p.prod q)

/-- Product-integral form of the upper bracket after splitting into separated
products. -/
noncomputable def upperBracketProductIntegral
    (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  (∫ z in Set.Ioi x ×ˢ Set.Ioi x,
      ((z.1 - x) * Real.exp (-z.1 / τ)) * Real.exp (-z.2 / τ) ∂(p.prod q)) -
    ∫ z in Set.Ioi x ×ˢ Set.Ioi x,
      Real.exp (-z.1 / τ) * ((z.2 - x) * Real.exp (-z.2 / τ)) ∂(p.prod q)

/-- Direct restricted double-integral form of the lower truncated pairing. -/
noncomputable def lowerTruncatedPairingIntegral
    (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ z, (z.1 - z.2) * Real.exp ((z.1 + z.2) / τ)
    ∂((p.restrict (Set.Iic x)).prod (q.restrict (Set.Iic x)))

/-- Direct restricted double-integral form of the upper truncated pairing. -/
noncomputable def upperTruncatedPairingIntegral
    (τ : ℝ) (p q : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ z, (z.1 - z.2) * Real.exp (-(z.1 + z.2) / τ)
    ∂((p.restrict (Set.Ioi x)).prod (q.restrict (Set.Ioi x)))

@[simp] theorem lowerBracketProductIntegral_eq_truncatedPairing
    (τ : ℝ) (p q : Measure ℝ) [SFinite p] [SFinite q] (x : ℝ) :
    lowerBracketProductIntegral τ p q x = truncatedPairing τ p q x := by
  unfold lowerBracketProductIntegral truncatedPairing lowerCompensatedMoment
    lowerExpMass
  have h₁ :
      (∫ z in Set.Iic x ×ˢ Set.Iic x,
        Real.exp (z.1 / τ) * ((x - z.2) * Real.exp (z.2 / τ)) ∂(p.prod q)) =
        (∫ y in Set.Iic x, Real.exp (y / τ) ∂p) *
          ∫ y in Set.Iic x, (x - y) * Real.exp (y / τ) ∂q := by
    simpa using
      (setIntegral_prod_mul
        (μ := p) (ν := q)
        (f := fun y : ℝ => Real.exp (y / τ))
        (g := fun y : ℝ => (x - y) * Real.exp (y / τ))
        (s := Set.Iic x) (t := Set.Iic x))
  have h₂ :
      (∫ z in Set.Iic x ×ˢ Set.Iic x,
        ((x - z.1) * Real.exp (z.1 / τ)) * Real.exp (z.2 / τ) ∂(p.prod q)) =
        (∫ y in Set.Iic x, (x - y) * Real.exp (y / τ) ∂p) *
          ∫ y in Set.Iic x, Real.exp (y / τ) ∂q := by
    simpa using
      (setIntegral_prod_mul
        (μ := p) (ν := q)
        (f := fun y : ℝ => (x - y) * Real.exp (y / τ))
        (g := fun y : ℝ => Real.exp (y / τ))
        (s := Set.Iic x) (t := Set.Iic x))
  rw [h₁, h₂]
  ring

@[simp] theorem upperBracketProductIntegral_eq_upperPairing
    (τ : ℝ) (p q : Measure ℝ) [SFinite p] [SFinite q] (x : ℝ) :
    upperBracketProductIntegral τ p q x = upperPairing τ p q x := by
  unfold upperBracketProductIntegral upperPairing upperCompensatedMoment
    upperExpMass
  have h₁ :
      (∫ z in Set.Ioi x ×ˢ Set.Ioi x,
        ((z.1 - x) * Real.exp (-z.1 / τ)) * Real.exp (-z.2 / τ) ∂(p.prod q)) =
        (∫ y in Set.Ioi x, (y - x) * Real.exp (-y / τ) ∂p) *
          ∫ y in Set.Ioi x, Real.exp (-y / τ) ∂q := by
    simpa using
      (setIntegral_prod_mul
        (μ := p) (ν := q)
        (f := fun y : ℝ => (y - x) * Real.exp (-y / τ))
        (g := fun y : ℝ => Real.exp (-y / τ))
        (s := Set.Ioi x) (t := Set.Ioi x))
  have h₂ :
      (∫ z in Set.Ioi x ×ˢ Set.Ioi x,
        Real.exp (-z.1 / τ) * ((z.2 - x) * Real.exp (-z.2 / τ)) ∂(p.prod q)) =
        (∫ y in Set.Ioi x, Real.exp (-y / τ) ∂p) *
          ∫ y in Set.Ioi x, (y - x) * Real.exp (-y / τ) ∂q := by
    simpa using
      (setIntegral_prod_mul
        (μ := p) (ν := q)
        (f := fun y : ℝ => Real.exp (-y / τ))
        (g := fun y : ℝ => (y - x) * Real.exp (-y / τ))
        (s := Set.Ioi x) (t := Set.Ioi x))
  rw [h₁, h₂]
  ring

/-! ## Pointwise algebra behind the direct double-integral form -/

/-- Pointwise algebra for the lower-left region:
`(y-z)e^{(y+z)/τ}` is the difference of the two separated products used in
`truncatedPairing`. -/
theorem lowerBracket_integrand_eq
    (τ x y z : ℝ) :
    (y - z) * Real.exp ((y + z) / τ) =
      Real.exp (y / τ) * ((x - z) * Real.exp (z / τ)) -
        ((x - y) * Real.exp (y / τ)) * Real.exp (z / τ) := by
  rw [show (y + z) / τ = y / τ + z / τ by ring, Real.exp_add]
  ring

/-- Pointwise algebra for the upper-right region, the mirror of
`lowerBracket_integrand_eq`. -/
theorem upperBracket_integrand_eq
    (τ x y z : ℝ) :
    (y - z) * Real.exp (-(y + z) / τ) =
      ((y - x) * Real.exp (-y / τ)) * Real.exp (-z / τ) -
        Real.exp (-y / τ) * ((z - x) * Real.exp (-z / τ)) := by
  rw [show -(y + z) / τ = -y / τ + -z / τ by ring, Real.exp_add]
  ring

/-- Milestone-1 bracket identity, lower side: the conceptual restricted
double integral equals `Q·P⁻ - P·Q⁻`. -/
theorem lowerTruncatedPairingIntegral_eq_truncatedPairing
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    lowerTruncatedPairingIntegral τ p q x = truncatedPairing τ p q x := by
  have hpE := integrable_lowerExpKernel τ hτ p x
  have hqE := integrable_lowerExpKernel τ hτ q x
  have hpC := integrable_lowerCompKernel τ hτ p x
  have hqC := integrable_lowerCompKernel τ hτ q x
  unfold lowerTruncatedPairingIntegral truncatedPairing lowerCompensatedMoment
    lowerExpMass
  calc
    ∫ z, (z.1 - z.2) * Real.exp ((z.1 + z.2) / τ)
        ∂((p.restrict (Set.Iic x)).prod (q.restrict (Set.Iic x)))
        =
      ∫ z,
        Real.exp (z.1 / τ) * ((x - z.2) * Real.exp (z.2 / τ)) -
          ((x - z.1) * Real.exp (z.1 / τ)) * Real.exp (z.2 / τ)
        ∂((p.restrict (Set.Iic x)).prod (q.restrict (Set.Iic x))) := by
          apply integral_congr_ae
          filter_upwards with z
          exact lowerBracket_integrand_eq τ x z.1 z.2
    _ =
      (∫ y, Real.exp (y / τ) ∂p.restrict (Set.Iic x)) *
          ∫ y, (x - y) * Real.exp (y / τ) ∂q.restrict (Set.Iic x) -
        (∫ y, (x - y) * Real.exp (y / τ) ∂p.restrict (Set.Iic x)) *
          ∫ y, Real.exp (y / τ) ∂q.restrict (Set.Iic x) := by
          rw [integral_sub (hpE.mul_prod hqC) (hpC.mul_prod hqE)]
          have h₁ :
              (∫ a : ℝ × ℝ,
                Real.exp (a.1 / τ) * ((x - a.2) * Real.exp (a.2 / τ))
                  ∂(p.restrict (Set.Iic x)).prod (q.restrict (Set.Iic x))) =
                (∫ y, Real.exp (y / τ) ∂p.restrict (Set.Iic x)) *
                  ∫ y, (x - y) * Real.exp (y / τ)
                    ∂q.restrict (Set.Iic x) := by
            simpa using
              (integral_prod_mul
                (μ := p.restrict (Set.Iic x)) (ν := q.restrict (Set.Iic x))
                (f := fun y : ℝ => Real.exp (y / τ))
                (g := fun y : ℝ => (x - y) * Real.exp (y / τ)))
          have h₂ :
              (∫ a : ℝ × ℝ,
                ((x - a.1) * Real.exp (a.1 / τ)) * Real.exp (a.2 / τ)
                  ∂(p.restrict (Set.Iic x)).prod (q.restrict (Set.Iic x))) =
                (∫ y, (x - y) * Real.exp (y / τ)
                    ∂p.restrict (Set.Iic x)) *
                  ∫ y, Real.exp (y / τ) ∂q.restrict (Set.Iic x) := by
            simpa using
              (integral_prod_mul
                (μ := p.restrict (Set.Iic x)) (ν := q.restrict (Set.Iic x))
                (f := fun y : ℝ => (x - y) * Real.exp (y / τ))
                (g := fun y : ℝ => Real.exp (y / τ)))
          rw [h₁, h₂]
    _ =
      (∫ y in Set.Iic x, (x - y) * Real.exp (y / τ) ∂q) *
          ∫ y in Set.Iic x, Real.exp (y / τ) ∂p -
        (∫ y in Set.Iic x, (x - y) * Real.exp (y / τ) ∂p) *
          ∫ y in Set.Iic x, Real.exp (y / τ) ∂q := by
          ring

/-- Milestone-1 bracket identity, upper side: the conceptual restricted
double integral equals `P̂·Q⁺ - Q̂·P⁺`. -/
theorem upperTruncatedPairingIntegral_eq_upperPairing
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsFiniteMeasure p] [IsFiniteMeasure q] (x : ℝ) :
    upperTruncatedPairingIntegral τ p q x = upperPairing τ p q x := by
  have hpE := integrable_upperExpKernel τ hτ p x
  have hqE := integrable_upperExpKernel τ hτ q x
  have hpC := integrable_upperCompKernel τ hτ p x
  have hqC := integrable_upperCompKernel τ hτ q x
  unfold upperTruncatedPairingIntegral upperPairing upperCompensatedMoment
    upperExpMass
  calc
    ∫ z, (z.1 - z.2) * Real.exp (-(z.1 + z.2) / τ)
        ∂((p.restrict (Set.Ioi x)).prod (q.restrict (Set.Ioi x)))
        =
      ∫ z,
        ((z.1 - x) * Real.exp (-z.1 / τ)) * Real.exp (-z.2 / τ) -
          Real.exp (-z.1 / τ) * ((z.2 - x) * Real.exp (-z.2 / τ))
        ∂((p.restrict (Set.Ioi x)).prod (q.restrict (Set.Ioi x))) := by
          apply integral_congr_ae
          filter_upwards with z
          exact upperBracket_integrand_eq τ x z.1 z.2
    _ =
      (∫ y, (y - x) * Real.exp (-y / τ) ∂p.restrict (Set.Ioi x)) *
          ∫ y, Real.exp (-y / τ) ∂q.restrict (Set.Ioi x) -
        (∫ y, Real.exp (-y / τ) ∂p.restrict (Set.Ioi x)) *
          ∫ y, (y - x) * Real.exp (-y / τ) ∂q.restrict (Set.Ioi x) := by
          rw [integral_sub (hpC.mul_prod hqE) (hpE.mul_prod hqC)]
          have h₁ :
              (∫ a : ℝ × ℝ,
                ((a.1 - x) * Real.exp (-a.1 / τ)) * Real.exp (-a.2 / τ)
                  ∂(p.restrict (Set.Ioi x)).prod (q.restrict (Set.Ioi x))) =
                (∫ y, (y - x) * Real.exp (-y / τ)
                    ∂p.restrict (Set.Ioi x)) *
                  ∫ y, Real.exp (-y / τ) ∂q.restrict (Set.Ioi x) := by
            simpa using
              (integral_prod_mul
                (μ := p.restrict (Set.Ioi x)) (ν := q.restrict (Set.Ioi x))
                (f := fun y : ℝ => (y - x) * Real.exp (-y / τ))
                (g := fun y : ℝ => Real.exp (-y / τ)))
          have h₂ :
              (∫ a : ℝ × ℝ,
                Real.exp (-a.1 / τ) * ((a.2 - x) * Real.exp (-a.2 / τ))
                  ∂(p.restrict (Set.Ioi x)).prod (q.restrict (Set.Ioi x))) =
                (∫ y, Real.exp (-y / τ) ∂p.restrict (Set.Ioi x)) *
                  ∫ y, (y - x) * Real.exp (-y / τ)
                    ∂q.restrict (Set.Ioi x) := by
            simpa using
              (integral_prod_mul
                (μ := p.restrict (Set.Ioi x)) (ν := q.restrict (Set.Ioi x))
                (f := fun y : ℝ => Real.exp (-y / τ))
                (g := fun y : ℝ => (y - x) * Real.exp (-y / τ)))
          rw [h₁, h₂]
    _ =
      (∫ y in Set.Ioi x, (y - x) * Real.exp (-y / τ) ∂p) *
          ∫ y in Set.Ioi x, Real.exp (-y / τ) ∂q -
        (∫ y in Set.Ioi x, (y - x) * Real.exp (-y / τ) ∂q) *
          ∫ y in Set.Ioi x, Real.exp (-y / τ) ∂p := by
          ring

end DriftingIdentifiability
