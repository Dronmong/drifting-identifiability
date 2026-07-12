import DriftingIdentifiability.LaplaceACAsymptotics
import DriftingIdentifiability.LaplaceACPropagation

/-!
# Final assembly for the absolutely-continuous Laplace converse

This file contains the final, axiom-free socket for the a.c. Laplace route
documented in `LaplaceACDerivation.md`.

The upstream files have already proved the hard analytic packages:

* L6 kills the two outer rays of the Wronskian;
* L8 kills the two flanks of each upward crossing;
* L9 turns a finite alternating down/up cover into global Wronskian vanishing;
* the certified Wronskian gate turns global Wronskian vanishing into `p = q`.

The theorem here intentionally does not hide the remaining modelling input:
callers must provide the finite alternating cover for the actual Wronskian.
That cover is exactly the output expected from applying the L6 and L8
certificates to the chosen finite breakpoint list.
-/

open MeasureTheory Set Filter Topology
open scoped intervalIntegral

namespace DriftingIdentifiability

open Paper

/-! ## Small FTC/local-boundedness helpers for certificate constructors -/

private lemma uIcc_subset_Ioo_of_endpoints_mem_Ioo
    {a b x y : ℝ} (hx : x ∈ Ioo a b) (hy : y ∈ Ioo a b) :
    uIcc x y ⊆ Ioo a b := by
  intro z hz
  rcases le_total x y with hxy | hyx
  · rw [uIcc_of_le hxy] at hz
    exact ⟨lt_of_lt_of_le hx.1 hz.1, lt_of_le_of_lt hz.2 hy.2⟩
  · rw [uIcc_of_ge hyx] at hz
    exact ⟨lt_of_lt_of_le hy.1 hz.1, lt_of_le_of_lt hz.2 hx.2⟩

/-- Primitive-by-FTC helper used by upward-crossing certificates.  If a scalar
coefficient is continuous on an open gap, then `z ↦ ∫ s in base..z, c s` has
the expected right derivative at every point of that gap. -/
private theorem intervalPrimitive_hasDerivWithinAt_Ici_of_continuousOn_Ioo
    {c : ℝ → ℝ} {a b base t : ℝ}
    (hcont : ContinuousOn c (Ioo a b))
    (hbase : base ∈ Ioo a b) (ht : t ∈ Ioo a b) :
    HasDerivWithinAt (fun z : ℝ => ∫ s in base..z, c s) (c t) (Ici t) t := by
  have hint : IntervalIntegrable c volume base t := by
    exact (hcont.mono (uIcc_subset_Ioo_of_endpoints_mem_Ioo hbase ht)).intervalIntegrable
  let mid : ℝ := (t + b) / 2
  have ht_mid : t < mid := by
    dsimp [mid]
    linarith [ht.2]
  have hmid_b : mid < b := by
    dsimp [mid]
    linarith [ht.2]
  let J : Set ℝ := Icc t mid
  have hJsubset : J ⊆ Ioo a b := by
    intro z hz
    exact ⟨lt_of_lt_of_le ht.1 hz.1, lt_of_le_of_lt hz.2 hmid_b⟩
  have hcontJ : ContinuousOn c J := hcont.mono hJsubset
  have hmeasJ : StronglyMeasurableAtFilter c (𝓝[J] t) volume :=
    hcontJ.stronglyMeasurableAtFilter_nhdsWithin measurableSet_Icc t
  have htJ : t ∈ J := ⟨le_rfl, ht_mid.le⟩
  haveI : Fact (t ∈ J) := ⟨htJ⟩
  have hderivJ : HasDerivWithinAt (fun z : ℝ => ∫ s in base..z, c s)
      (c t) J t :=
    intervalIntegral.integral_hasDerivWithinAt_right
      hint hmeasJ (hcontJ t htJ)
  exact hderivJ.mono_of_mem_nhdsWithin (Icc_mem_nhdsGE ht_mid)

/-- Continuity of the same FTC primitive on compact subintervals of the gap. -/
private theorem intervalPrimitive_continuousOn_Icc_of_continuousOn_Ioo
    {c : ℝ → ℝ} {a b base x y : ℝ}
    (hcont : ContinuousOn c (Ioo a b))
    (hbase : base ∈ Ioo a b) (hx : x ∈ Ioo a b) (hy : y ∈ Ioo a b) :
    ContinuousOn (fun z : ℝ => ∫ s in base..z, c s) (Icc x y) := by
  refine continuousOn_of_forall_continuousAt ?_
  intro z hz
  have hzGap : z ∈ Ioo a b :=
    ⟨lt_of_lt_of_le hx.1 hz.1, lt_of_le_of_lt hz.2 hy.2⟩
  have hcz : ContinuousAt c z :=
    hcont.continuousAt (Ioo_mem_nhds hzGap.1 hzGap.2)
  have hint : IntervalIntegrable c volume base z := by
    exact (hcont.mono (uIcc_subset_Ioo_of_endpoints_mem_Ioo hbase hzGap)).intervalIntegrable
  exact (intervalIntegral.integral_hasDerivAt_right
    hint (ContinuousOn.stronglyMeasurableAtFilter isOpen_Ioo hcont z hzGap) hcz).continuousAt

/-- The normalizer-Wronskian is continuous once both normalizers have the C²
certificate used by the Abel bridge.  The certificate gives continuity of the
right-derivative coefficient; the normalizers themselves are already continuous
for finite measures. -/
theorem continuous_laplaceKernelNormalizerWronskian_of_regular
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q) :
    Continuous (fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x) := by
  rw [continuous_iff_continuousAt]
  intro x
  unfold laplaceKernelNormalizerWronskian
  exact
    ((hp.hasDerivAt_rightDerivCoeff x).continuousAt.mul
        ((continuous_laplaceKernelNormalizer τ hτ q).continuousAt)).sub
      (((continuous_laplaceKernelNormalizer τ hτ p).continuousAt).mul
        (hq.hasDerivAt_rightDerivCoeff x).continuousAt)

private theorem laplaceWronskian_isBoundedUnder_nhdsLT_of_regular
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q) (x : ℝ) :
    IsBoundedUnder (· ≤ ·) (𝓝[<] x)
      (norm ∘ fun y : ℝ => laplaceKernelNormalizerWronskian τ p q y) := by
  have hWcont : Continuous (fun y : ℝ => laplaceKernelNormalizerWronskian τ p q y) :=
    continuous_laplaceKernelNormalizerWronskian_of_regular τ hτ p q hp hq
  exact (hWcont.continuousAt.norm.isBoundedUnder_le).mono nhdsWithin_le_nhds

private theorem laplaceWronskian_isBoundedUnder_nhdsGT_of_regular
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q) (x : ℝ) :
    IsBoundedUnder (· ≤ ·) (𝓝[>] x)
      (norm ∘ fun y : ℝ => laplaceKernelNormalizerWronskian τ p q y) := by
  have hWcont : Continuous (fun y : ℝ => laplaceKernelNormalizerWronskian τ p q y) :=
    continuous_laplaceKernelNormalizerWronskian_of_regular τ hτ p q hp hq
  exact (hWcont.continuousAt.norm.isBoundedUnder_le).mono nhdsWithin_le_nhds

/-- Final deterministic certificate for the a.c. Laplace Wronskian endgame.

`breaks` is the finite alternating sign-change list, intended to have shape
`[down₁, up₂, down₃, ..., down_M]`.  The `alternating_cover` field says that
the actual normalizer Wronskian vanishes on the left ray, on both flanks of
each upward crossing, and on the right ray.  L9 then fills in the breakpoint
values by continuity. -/
structure LaplaceACFinalAssembly (τ : ℝ) (p q : Measure ℝ) where
  breaks : List ℝ
  wronskian_continuous :
    Continuous (fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x)
  alternating_cover :
    VanishesOnAlternatingUpwardPairs
      (fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x) breaks

/-- The final assembly certificate forces the actual normalizer Wronskian to
vanish everywhere. -/
theorem laplaceAC_wronskian_eq_zero_of_finalAssembly
    (τ : ℝ) (p q : Measure ℝ)
    (h : LaplaceACFinalAssembly τ p q) :
    ∀ x : ℝ, laplaceKernelNormalizerWronskian τ p q x = 0 := by
  exact continuous_eq_zero_of_alternatingUpwardPairs h.breaks
    h.wronskian_continuous h.alternating_cover

/-- **Final a.c. Laplace assembly gate.**  Once the L6/L8/L9 deterministic
certificate has been built for the actual Wronskian, the certified Wronskian
injectivity theorem gives `p = q`.

Zero drift is not repeated here because it is used upstream to produce the
assembly certificate; after that certificate is available, the Wronskian gate
itself needs only probability measures and a valid bandwidth. -/
theorem laplaceAC_identifies_of_finalAssembly
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (h : LaplaceACFinalAssembly τ p q) :
    p = q :=
  laplaceKernelNormalizer_wronskian_eq_zero_imp_eq τ hτ p q
    (laplaceAC_wronskian_eq_zero_of_finalAssembly τ p q h)

/-- Convenience form for the single-crossing/3A case.  If one downward
breakpoint has zero Wronskian on both outer rays, it gives the alternating cover
expected by the final assembly theorem. -/
theorem VanishesOnAlternatingUpwardPairs.singleDown
    {W : ℝ → ℝ} {a : ℝ}
    (hleft : ∀ x : ℝ, x < a → W x = 0)
    (hright : ∀ x : ℝ, a < x → W x = 0) :
    VanishesOnAlternatingUpwardPairs W [a] := by
  exact ⟨hleft, hright⟩

/-- Convenience form for the three-crossing/first nontrivial 3B case.  A
down/up/down list is covered by the left outer ray, the two flanks of the upward
crossing, and the right outer ray. -/
theorem VanishesOnAlternatingUpwardPairs.downUpDown
    {W : ℝ → ℝ} {a b c : ℝ}
    (hleft : ∀ x : ℝ, x < a → W x = 0)
    (hleftFlank : ∀ x : ℝ, a < x → x < b → W x = 0)
    (hrightFlank : ∀ x : ℝ, b < x → x < c → W x = 0)
    (hright : ∀ x : ℝ, c < x → W x = 0) :
    VanishesOnAlternatingUpwardPairs W [a, b, c] := by
  exact ⟨hleft, hleftFlank, hrightFlank, hright⟩

/-! ## Concrete single-crossing / 3A assembly -/

/-- **3A upstream assembly, Wronskian form.**

This theorem constructs the final assembly certificate in the single downward
crossing case.  The hypotheses are intentionally explicit:

* C²-normalizer regularity gives Wronskian continuity and the Abel equation;
* two-sided exponential moments give the Wronskian tail limits;
* integrability of `y` under `p` lets L3 supply `0 ≤ μ'`;
* `m > 0` to the left and `m < 0` to the right encode the single downward
  crossing sign geometry.

The proof applies the L6 outer-ray package with a moving endpoint `x`, so the
sign assumptions are needed only on the open rays around the crossing point. -/
noncomputable def laplaceACFinalAssembly_singleDown_of_outerSigns
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {a : ℝ}
    (hpExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) p)
    (hqExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) q)
    (hpExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) p)
    (hqExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) q)
    (hpMoment : Integrable (fun y : ℝ => y) p)
    (hm_left : ∀ t : ℝ, t < a → 0 < laplaceMeanShiftRatio τ p t)
    (hm_right : ∀ t : ℝ, a < t → laplaceMeanShiftRatio τ p t < 0) :
    LaplaceACFinalAssembly τ p q := by
  let W : ℝ → ℝ := fun x => laplaceKernelNormalizerWronskian τ p q x
  let μDeriv : ℝ → ℝ := fun x => laplaceMeanShiftRatioDeriv τ p x + 1
  let m : ℝ → ℝ := fun x => laplaceMeanShiftRatio τ p x
  have hμ_nonneg : ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1 :=
    laplaceMeanShiftRatioDeriv_add_one_nonneg_of_regular τ hτ p hp hpMoment
  have hWcont : Continuous W := by
    simpa [W] using
      continuous_laplaceKernelNormalizerWronskian_of_regular τ hτ p q hp hq
  have hWtop : Tendsto W atTop (𝓝 0) := by
    simpa [W] using
      laplaceKernelNormalizerWronskian_tendsto_atTop_zero τ hτ p q
        hpExpPos hqExpPos
  have hWbot : Tendsto W atBot (𝓝 0) := by
    simpa [W] using
      laplaceKernelNormalizerWronskian_tendsto_atBot_zero τ hτ p q
        hpExpNeg hqExpNeg
  have hleft : ∀ x : ℝ, x < a → W x = 0 := by
    intro x hxa
    refine abel_left_outer_zero_of_muDeriv_nonneg_of_m_pos
      (W := W) (muDeriv := μDeriv) (m := m) (a := x)
      ?_ ?_ ?_ ?_ hWbot x le_rfl
    · intro b y hby hyx
      exact hWcont.continuousOn
    · intro b y hby hyx t ht
      have hta : t < a := lt_of_lt_of_le ht.2 (le_trans hyx hxa.le)
      have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
        (hm_left t hta).ne'
      have hderiv :=
        hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
          τ hτ p q hp hq hzero t hmt
      simpa [W, μDeriv, m] using hderiv.hasDerivWithinAt
    · intro t _ht
      exact hμ_nonneg t
    · intro t htx
      exact hm_left t (lt_of_le_of_lt htx hxa)
  have hright : ∀ x : ℝ, a < x → W x = 0 := by
    intro x hax
    refine abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg
      (W := W) (muDeriv := μDeriv) (m := m) (a := x)
      ?_ ?_ ?_ ?_ hWtop x le_rfl
    · intro y b hyx hyb
      exact hWcont.continuousOn
    · intro y b hyx hyb t ht
      have hat : a < t := lt_of_lt_of_le hax (le_trans hyx ht.1)
      have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
        (hm_right t hat).ne
      have hderiv :=
        hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
          τ hτ p q hp hq hzero t hmt
      simpa [W, μDeriv, m] using hderiv.hasDerivWithinAt
    · intro t _ht
      exact hμ_nonneg t
    · intro t hxt
      exact hm_right t (lt_of_lt_of_le hax hxt)
  exact
    { breaks := [a]
      wronskian_continuous := hWcont
      alternating_cover :=
        VanishesOnAlternatingUpwardPairs.singleDown hleft hright }

/-- **3A single-crossing identification theorem.**  This is the previous
assembly theorem composed with the certified Wronskian gate. -/
theorem laplaceAC_identifies_singleDown_of_outerSigns
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {a : ℝ}
    (hpExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) p)
    (hqExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) q)
    (hpExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) p)
    (hqExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) q)
    (hpMoment : Integrable (fun y : ℝ => y) p)
    (hm_left : ∀ t : ℝ, t < a → 0 < laplaceMeanShiftRatio τ p t)
    (hm_right : ∀ t : ℝ, a < t → laplaceMeanShiftRatio τ p t < 0) :
    p = q :=
  laplaceAC_identifies_of_finalAssembly τ hτ p q
    (laplaceACFinalAssembly_singleDown_of_outerSigns τ hτ p q hp hq hzero
      hpExpPos hqExpPos hpExpNeg hqExpNeg hpMoment hm_left hm_right)

/-! ## Concrete three-crossing / first 3B assembly -/

/-- **First nontrivial 3B upstream assembly, Wronskian form.**

This packages the breakpoint pattern `[down, up, down]`.  The two outer open
rays are killed by L6, and the two interior gaps are killed by the L8
upward-crossing logarithmic-singularity wrappers at the middle breakpoint.

The local `δ,L` and primitive hypotheses are still explicit: this theorem is
the deterministic assembly layer, not the separate smoothness lemma that
produces those local witnesses. -/
noncomputable def laplaceACFinalAssembly_downUpDown_of_outerSigns_of_upwardCrossing
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {a b c l r δL δR LL LR : ℝ} {AL AR : ℝ → ℝ}
    (hab : a < b) (hbc : b < c)
    (hal : a < l) (hlb : l < b)
    (hbr : b < r) (hrc : r < c)
    (hδL : 0 < δL) (hδR : 0 < δR)
    (hLL : 0 < LL) (hLR : 0 < LR)
    (hpExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) p)
    (hqExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) q)
    (hpExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) p)
    (hqExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) q)
    (hpMoment : Integrable (fun y : ℝ => y) p)
    (hm_left : ∀ t : ℝ, t < a → 0 < laplaceMeanShiftRatio τ p t)
    (hm_ab_neg : ∀ t : ℝ, a < t → t < b → laplaceMeanShiftRatio τ p t < 0)
    (hm_bc_pos : ∀ t : ℝ, b < t → t < c → 0 < laplaceMeanShiftRatio τ p t)
    (hm_right : ∀ t : ℝ, c < t → laplaceMeanShiftRatio τ p t < 0)
    (hAL : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y,
        HasDerivWithinAt AL
          (((2 : ℝ) * (laplaceMeanShiftRatioDeriv τ p t + 1)) /
            laplaceMeanShiftRatio τ p t) (Ici t) t)
    (hALcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn AL (Icc x y))
    (hAR : ∀ x y : ℝ, b < x → x ≤ y → y < c →
      ∀ t ∈ Ico x y,
        HasDerivWithinAt AR
          (((2 : ℝ) * (laplaceMeanShiftRatioDeriv τ p t + 1)) /
            laplaceMeanShiftRatio τ p t) (Ici t) t)
    (hARcont : ∀ x y : ℝ, b < x → x ≤ y → y < c →
      ContinuousOn AR (Icc x y))
    (hWboundedLeft : IsBoundedUnder (· ≤ ·) (𝓝[<] b)
      (norm ∘ fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x))
    (hWboundedRight : IsBoundedUnder (· ≤ ·) (𝓝[>] b)
      (norm ∘ fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x))
    (hμ_left : ∀ t : ℝ, l ≤ t → t < b →
      δL ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    (hm_left_lower : ∀ t : ℝ, l ≤ t → t < b →
      -(LL * (b - t)) ≤ laplaceMeanShiftRatio τ p t)
    (hμ_right : ∀ t : ℝ, b < t → t < r →
      δR ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    (hm_right_upper : ∀ t : ℝ, b < t → t < r →
      laplaceMeanShiftRatio τ p t ≤ LR * (t - b)) :
    LaplaceACFinalAssembly τ p q := by
  let W : ℝ → ℝ := fun x => laplaceKernelNormalizerWronskian τ p q x
  let μDeriv : ℝ → ℝ := fun x => laplaceMeanShiftRatioDeriv τ p x + 1
  let m : ℝ → ℝ := fun x => laplaceMeanShiftRatio τ p x
  have hμ_nonneg : ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1 :=
    laplaceMeanShiftRatioDeriv_add_one_nonneg_of_regular τ hτ p hp hpMoment
  have hWcont : Continuous W := by
    simpa [W] using
      continuous_laplaceKernelNormalizerWronskian_of_regular τ hτ p q hp hq
  have hWtop : Tendsto W atTop (𝓝 0) := by
    simpa [W] using
      laplaceKernelNormalizerWronskian_tendsto_atTop_zero τ hτ p q
        hpExpPos hqExpPos
  have hWbot : Tendsto W atBot (𝓝 0) := by
    simpa [W] using
      laplaceKernelNormalizerWronskian_tendsto_atBot_zero τ hτ p q
        hpExpNeg hqExpNeg
  have hleft : ∀ x : ℝ, x < a → W x = 0 := by
    intro x hxa
    refine abel_left_outer_zero_of_muDeriv_nonneg_of_m_pos
      (W := W) (muDeriv := μDeriv) (m := m) (a := x)
      ?_ ?_ ?_ ?_ hWbot x le_rfl
    · intro u v huv hvx
      exact hWcont.continuousOn
    · intro u v huv hvx t ht
      have hta : t < a := lt_of_lt_of_le ht.2 (le_trans hvx hxa.le)
      have hmt : laplaceMeanShiftRatio τ p t ≠ 0 := (hm_left t hta).ne'
      have hderiv :=
        hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
          τ hτ p q hp hq hzero t hmt
      simpa [W, μDeriv, m] using hderiv.hasDerivWithinAt
    · intro t _ht
      exact hμ_nonneg t
    · intro t htx
      exact hm_left t (lt_of_le_of_lt htx hxa)
  have hright : ∀ x : ℝ, c < x → W x = 0 := by
    intro x hcx
    refine abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg
      (W := W) (muDeriv := μDeriv) (m := m) (a := x)
      ?_ ?_ ?_ ?_ hWtop x le_rfl
    · intro u v huv huv'
      exact hWcont.continuousOn
    · intro u v huv huv' t ht
      have hct : c < t := lt_of_lt_of_le hcx (le_trans huv ht.1)
      have hmt : laplaceMeanShiftRatio τ p t ≠ 0 := (hm_right t hct).ne
      have hderiv :=
        hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
          τ hτ p q hp hq hzero t hmt
      simpa [W, μDeriv, m] using hderiv.hasDerivWithinAt
    · intro t _ht
      exact hμ_nonneg t
    · intro t hxt
      exact hm_right t (lt_of_lt_of_le hcx hxt)
  have hleftFlank : ∀ x : ℝ, a < x → x < b → W x = 0 := by
    refine abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower
      (W := W) (A := AL) (μDeriv := μDeriv) (m := m)
      (a := a) (b := b) (l := l) (δ := δL) (L := LL)
      hab hal hlb hδL hLL ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
    · intro x y hax hxy hyb
      exact hWcont.continuousOn.mul
        (Real.continuous_exp.comp_continuousOn (hALcont x y hax hxy hyb))
    · intro x y hax hxy hyb t ht
      have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
        (hm_ab_neg t (lt_of_lt_of_le hax ht.1) (lt_of_lt_of_le ht.2 hyb.le)).ne
      have hderiv :=
        hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
          τ hτ p q hp hq hzero t hmt
      simpa [W, μDeriv, m] using hderiv.hasDerivWithinAt
    · intro x y hax hxy hyb
      simpa [μDeriv, m] using hAL x y hax hxy hyb
    · intro x y hax hxy hyb
      exact hALcont x y hax hxy hyb
    · simpa [W] using hWboundedLeft
    · intro t hlt htb
      exact hμ_left t hlt htb
    · intro t hlt htb
      exact hm_ab_neg t (lt_of_lt_of_le hal hlt) htb
    · intro t hlt htb
      exact hm_left_lower t hlt htb
  have hrightFlank : ∀ x : ℝ, b < x → x < c → W x = 0 := by
    refine abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper
      (W := W) (A := AR) (μDeriv := μDeriv) (m := m)
      (a := b) (b := c) (r := r) (δ := δR) (L := LR)
      hbc hbr hrc hδR hLR ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
    · intro x y hbx hxy hyc
      exact hWcont.continuousOn.mul
        (Real.continuous_exp.comp_continuousOn (hARcont x y hbx hxy hyc))
    · intro x y hbx hxy hyc t ht
      have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
        (hm_bc_pos t (lt_of_lt_of_le hbx ht.1) (lt_of_lt_of_le ht.2 hyc.le)).ne'
      have hderiv :=
        hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
          τ hτ p q hp hq hzero t hmt
      simpa [W, μDeriv, m] using hderiv.hasDerivWithinAt
    · intro x y hbx hxy hyc
      simpa [μDeriv, m] using hAR x y hbx hxy hyc
    · intro x y hbx hxy hyc
      exact hARcont x y hbx hxy hyc
    · simpa [W] using hWboundedRight
    · intro t hbt htr
      exact hμ_right t hbt htr
    · intro t hbt htr
      exact hm_bc_pos t hbt (lt_trans htr hrc)
    · intro t hbt htr
      exact hm_right_upper t hbt htr
  exact
    { breaks := [a, b, c]
      wronskian_continuous := hWcont
      alternating_cover :=
        VanishesOnAlternatingUpwardPairs.downUpDown
          hleft hleftFlank hrightFlank hright }

/-- Identification theorem for the first nontrivial `[down, up, down]`
pattern, composed with the certified Wronskian gate. -/
theorem laplaceAC_identifies_downUpDown_of_outerSigns_of_upwardCrossing
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {a b c l r δL δR LL LR : ℝ} {AL AR : ℝ → ℝ}
    (hab : a < b) (hbc : b < c)
    (hal : a < l) (hlb : l < b)
    (hbr : b < r) (hrc : r < c)
    (hδL : 0 < δL) (hδR : 0 < δR)
    (hLL : 0 < LL) (hLR : 0 < LR)
    (hpExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) p)
    (hqExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) q)
    (hpExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) p)
    (hqExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) q)
    (hpMoment : Integrable (fun y : ℝ => y) p)
    (hm_left : ∀ t : ℝ, t < a → 0 < laplaceMeanShiftRatio τ p t)
    (hm_ab_neg : ∀ t : ℝ, a < t → t < b → laplaceMeanShiftRatio τ p t < 0)
    (hm_bc_pos : ∀ t : ℝ, b < t → t < c → 0 < laplaceMeanShiftRatio τ p t)
    (hm_right : ∀ t : ℝ, c < t → laplaceMeanShiftRatio τ p t < 0)
    (hAL : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y,
        HasDerivWithinAt AL
          (((2 : ℝ) * (laplaceMeanShiftRatioDeriv τ p t + 1)) /
            laplaceMeanShiftRatio τ p t) (Ici t) t)
    (hALcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn AL (Icc x y))
    (hAR : ∀ x y : ℝ, b < x → x ≤ y → y < c →
      ∀ t ∈ Ico x y,
        HasDerivWithinAt AR
          (((2 : ℝ) * (laplaceMeanShiftRatioDeriv τ p t + 1)) /
            laplaceMeanShiftRatio τ p t) (Ici t) t)
    (hARcont : ∀ x y : ℝ, b < x → x ≤ y → y < c →
      ContinuousOn AR (Icc x y))
    (hWboundedLeft : IsBoundedUnder (· ≤ ·) (𝓝[<] b)
      (norm ∘ fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x))
    (hWboundedRight : IsBoundedUnder (· ≤ ·) (𝓝[>] b)
      (norm ∘ fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x))
    (hμ_left : ∀ t : ℝ, l ≤ t → t < b →
      δL ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    (hm_left_lower : ∀ t : ℝ, l ≤ t → t < b →
      -(LL * (b - t)) ≤ laplaceMeanShiftRatio τ p t)
    (hμ_right : ∀ t : ℝ, b < t → t < r →
      δR ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    (hm_right_upper : ∀ t : ℝ, b < t → t < r →
      laplaceMeanShiftRatio τ p t ≤ LR * (t - b)) :
    p = q :=
  laplaceAC_identifies_of_finalAssembly τ hτ p q
    (laplaceACFinalAssembly_downUpDown_of_outerSigns_of_upwardCrossing
      τ hτ p q hp hq hzero hab hbc hal hlb hbr hrc hδL hδR hLL hLR
      hpExpPos hqExpPos hpExpNeg hqExpNeg hpMoment
      hm_left hm_ab_neg hm_bc_pos hm_right hAL hALcont hAR hARcont
      hWboundedLeft hWboundedRight hμ_left hm_left_lower hμ_right hm_right_upper)

/-! ## Arbitrary finite alternating assembly -/

/-- Local certificate for one upward crossing inside an alternating
down/up/down breakpoint list.

For breakpoints `leftDown < up < rightDown`, this records exactly the data
needed by the L8 logarithmic-singularity wrappers to kill both adjacent open
gaps `(leftDown, up)` and `(up, rightDown)` for the actual Laplace Wronskian.
The fields are deliberately local; outer rays and tail limits are handled by
the recursive chain below. -/
structure LaplaceACUpwardCrossingCertificate
    (τ : ℝ) (p q : Measure ℝ) (leftDown up rightDown : ℝ) where
  leftNear : ℝ
  rightNear : ℝ
  δL : ℝ
  δR : ℝ
  LL : ℝ
  LR : ℝ
  AL : ℝ → ℝ
  AR : ℝ → ℝ
  h_left_up : leftDown < up
  h_up_right : up < rightDown
  h_left_l : leftDown < leftNear
  h_l_up : leftNear < up
  h_up_r : up < rightNear
  h_r_right : rightNear < rightDown
  hδL : 0 < δL
  hδR : 0 < δR
  hLL : 0 < LL
  hLR : 0 < LR
  hm_left_neg : ∀ t : ℝ, leftDown < t → t < up →
    laplaceMeanShiftRatio τ p t < 0
  hm_right_pos : ∀ t : ℝ, up < t → t < rightDown →
    0 < laplaceMeanShiftRatio τ p t
  hAL : ∀ x y : ℝ, leftDown < x → x ≤ y → y < up →
    ∀ t ∈ Ico x y,
      HasDerivWithinAt AL
        (((2 : ℝ) * (laplaceMeanShiftRatioDeriv τ p t + 1)) /
          laplaceMeanShiftRatio τ p t) (Ici t) t
  hALcont : ∀ x y : ℝ, leftDown < x → x ≤ y → y < up →
    ContinuousOn AL (Icc x y)
  hAR : ∀ x y : ℝ, up < x → x ≤ y → y < rightDown →
    ∀ t ∈ Ico x y,
      HasDerivWithinAt AR
        (((2 : ℝ) * (laplaceMeanShiftRatioDeriv τ p t + 1)) /
          laplaceMeanShiftRatio τ p t) (Ici t) t
  hARcont : ∀ x y : ℝ, up < x → x ≤ y → y < rightDown →
    ContinuousOn AR (Icc x y)
  hWboundedLeft : IsBoundedUnder (· ≤ ·) (𝓝[<] up)
    (norm ∘ fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x)
  hWboundedRight : IsBoundedUnder (· ≤ ·) (𝓝[>] up)
    (norm ∘ fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x)
  hμ_left : ∀ t : ℝ, leftNear ≤ t → t < up →
    δL ≤ laplaceMeanShiftRatioDeriv τ p t + 1
  hm_left_lower : ∀ t : ℝ, leftNear ≤ t → t < up →
    -(LL * (up - t)) ≤ laplaceMeanShiftRatio τ p t
  hμ_right : ∀ t : ℝ, up < t → t < rightNear →
    δR ≤ laplaceMeanShiftRatioDeriv τ p t + 1
  hm_right_upper : ∀ t : ℝ, up < t → t < rightNear →
    laplaceMeanShiftRatio τ p t ≤ LR * (t - up)

/-- Constructor for an upward-crossing certificate from the regularity layer
plus local crossing inequalities.

This is the main upstream convenience constructor for L8.  Compared with the
raw certificate, callers no longer provide:

* the primitive functions `AL`, `AR`;
* FTC proofs that those primitives differentiate to `2 μ' / m`;
* continuity of the primitives on compact subintervals;
* boundedness of the Wronskian near the crossing.

Those are derived here from `LaplaceC2NormalizerRegular`, continuity of the
coefficient away from the zero, and the interval-integral FTC.  The remaining
inputs are the genuinely local crossing facts: sign on the two adjacent gaps,
positive lower bounds for `μ'`, and one-sided linear bounds for `m`. -/
noncomputable def LaplaceACUpwardCrossingCertificate.of_regular
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    {leftDown up rightDown leftNear rightNear δL δR LL LR : ℝ}
    (h_left_up : leftDown < up)
    (h_up_right : up < rightDown)
    (h_left_l : leftDown < leftNear)
    (h_l_up : leftNear < up)
    (h_up_r : up < rightNear)
    (h_r_right : rightNear < rightDown)
    (hδL : 0 < δL) (hδR : 0 < δR)
    (hLL : 0 < LL) (hLR : 0 < LR)
    (hm_left_neg : ∀ t : ℝ, leftDown < t → t < up →
      laplaceMeanShiftRatio τ p t < 0)
    (hm_right_pos : ∀ t : ℝ, up < t → t < rightDown →
      0 < laplaceMeanShiftRatio τ p t)
    (hμ_left : ∀ t : ℝ, leftNear ≤ t → t < up →
      δL ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    (hm_left_lower : ∀ t : ℝ, leftNear ≤ t → t < up →
      -(LL * (up - t)) ≤ laplaceMeanShiftRatio τ p t)
    (hμ_right : ∀ t : ℝ, up < t → t < rightNear →
      δR ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    (hm_right_upper : ∀ t : ℝ, up < t → t < rightNear →
      laplaceMeanShiftRatio τ p t ≤ LR * (t - up)) :
    LaplaceACUpwardCrossingCertificate τ p q leftDown up rightDown := by
  let μDeriv : ℝ → ℝ := fun x => laplaceMeanShiftRatioDeriv τ p x + 1
  let m : ℝ → ℝ := fun x => laplaceMeanShiftRatio τ p x
  let c : ℝ → ℝ := fun x => ((2 : ℝ) * μDeriv x) / m x
  have hm_cont : Continuous m := by
    rw [continuous_iff_continuousAt]
    intro x
    simpa [m] using
      (hasDerivAt_laplaceMeanShiftRatio_regular τ hτ p hp x).continuousAt
  have hμ_cont : Continuous μDeriv := by
    rw [continuous_iff_continuousAt]
    intro x
    change ContinuousAt (laplaceMeanShiftRatioDeriv τ p + fun _ : ℝ => (1 : ℝ)) x
    exact
      (hasDerivAt_laplaceMeanShiftRatioDeriv_regular τ hτ p hp x).continuousAt.add
        continuousAt_const
  have hc_left : ContinuousOn c (Ioo leftDown up) := by
    dsimp [c]
    exact ((continuous_const.mul hμ_cont).continuousOn).div hm_cont.continuousOn
      (fun t ht => (hm_left_neg t ht.1 ht.2).ne)
  have hc_right : ContinuousOn c (Ioo up rightDown) := by
    dsimp [c]
    exact ((continuous_const.mul hμ_cont).continuousOn).div hm_cont.continuousOn
      (fun t ht => (hm_right_pos t ht.1 ht.2).ne')
  refine
    { leftNear := leftNear
      rightNear := rightNear
      δL := δL
      δR := δR
      LL := LL
      LR := LR
      AL := fun z : ℝ => ∫ s in leftNear..z, c s
      AR := fun z : ℝ => ∫ s in rightNear..z, c s
      h_left_up := h_left_up
      h_up_right := h_up_right
      h_left_l := h_left_l
      h_l_up := h_l_up
      h_up_r := h_up_r
      h_r_right := h_r_right
      hδL := hδL
      hδR := hδR
      hLL := hLL
      hLR := hLR
      hm_left_neg := hm_left_neg
      hm_right_pos := hm_right_pos
      hAL := ?_
      hALcont := ?_
      hAR := ?_
      hARcont := ?_
      hWboundedLeft := ?_
      hWboundedRight := ?_
      hμ_left := hμ_left
      hm_left_lower := hm_left_lower
      hμ_right := hμ_right
      hm_right_upper := hm_right_upper }
  · intro x y hleftx hxy hyup t ht
    have htGap : t ∈ Ioo leftDown up :=
      ⟨lt_of_lt_of_le hleftx ht.1, lt_trans ht.2 hyup⟩
    simpa [c, μDeriv, m] using
      intervalPrimitive_hasDerivWithinAt_Ici_of_continuousOn_Ioo
        (c := c) (a := leftDown) (b := up) (base := leftNear) (t := t)
        hc_left ⟨h_left_l, h_l_up⟩ htGap
  · intro x y hleftx hxy hyup
    have hxGap : x ∈ Ioo leftDown up :=
      ⟨hleftx, lt_of_le_of_lt hxy hyup⟩
    have hyGap : y ∈ Ioo leftDown up :=
      ⟨lt_of_lt_of_le hleftx hxy, hyup⟩
    simpa [c, μDeriv, m] using
      intervalPrimitive_continuousOn_Icc_of_continuousOn_Ioo
        (c := c) (a := leftDown) (b := up) (base := leftNear) (x := x) (y := y)
        hc_left ⟨h_left_l, h_l_up⟩ hxGap hyGap
  · intro x y hupx hxy hyright t ht
    have htGap : t ∈ Ioo up rightDown :=
      ⟨lt_of_lt_of_le hupx ht.1, lt_trans ht.2 hyright⟩
    simpa [c, μDeriv, m] using
      intervalPrimitive_hasDerivWithinAt_Ici_of_continuousOn_Ioo
        (c := c) (a := up) (b := rightDown) (base := rightNear) (t := t)
        hc_right ⟨h_up_r, h_r_right⟩ htGap
  · intro x y hupx hxy hyright
    have hxGap : x ∈ Ioo up rightDown :=
      ⟨hupx, lt_of_le_of_lt hxy hyright⟩
    have hyGap : y ∈ Ioo up rightDown :=
      ⟨lt_of_lt_of_le hupx hxy, hyright⟩
    simpa [c, μDeriv, m] using
      intervalPrimitive_continuousOn_Icc_of_continuousOn_Ioo
        (c := c) (a := up) (b := rightDown) (base := rightNear) (x := x) (y := y)
        hc_right ⟨h_up_r, h_r_right⟩ hxGap hyGap
  · exact laplaceWronskian_isBoundedUnder_nhdsLT_of_regular τ hτ p q hp hq up
  · exact laplaceWronskian_isBoundedUnder_nhdsGT_of_regular τ hτ p q hp hq up

/-- One local upward-crossing certificate kills both adjacent gaps. -/
theorem LaplaceACUpwardCrossingCertificate.vanishes
    {τ : ℝ} (hτ : ValidBandwidth τ) {p q : Measure ℝ}
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {leftDown up rightDown : ℝ}
    (cert : LaplaceACUpwardCrossingCertificate τ p q leftDown up rightDown) :
    (∀ x : ℝ, leftDown < x → x < up →
        laplaceKernelNormalizerWronskian τ p q x = 0) ∧
      (∀ x : ℝ, up < x → x < rightDown →
        laplaceKernelNormalizerWronskian τ p q x = 0) := by
  let W : ℝ → ℝ := fun x => laplaceKernelNormalizerWronskian τ p q x
  let μDeriv : ℝ → ℝ := fun x => laplaceMeanShiftRatioDeriv τ p x + 1
  let m : ℝ → ℝ := fun x => laplaceMeanShiftRatio τ p x
  have hWcont : Continuous W := by
    simpa [W] using
      continuous_laplaceKernelNormalizerWronskian_of_regular τ hτ p q hp hq
  have hleft : ∀ x : ℝ, leftDown < x → x < up → W x = 0 := by
    refine abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower
      (W := W) (A := cert.AL) (μDeriv := μDeriv) (m := m)
      (a := leftDown) (b := up) (l := cert.leftNear) (δ := cert.δL) (L := cert.LL)
      cert.h_left_up cert.h_left_l cert.h_l_up cert.hδL cert.hLL
      ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
    · intro x y hx hxy hy
      exact hWcont.continuousOn.mul
        (Real.continuous_exp.comp_continuousOn (cert.hALcont x y hx hxy hy))
    · intro x y hx hxy hy t ht
      have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
        (cert.hm_left_neg t (lt_of_lt_of_le hx ht.1) (lt_of_lt_of_le ht.2 hy.le)).ne
      have hderiv :=
        hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
          τ hτ p q hp hq hzero t hmt
      simpa [W, μDeriv, m] using hderiv.hasDerivWithinAt
    · intro x y hx hxy hy
      simpa [μDeriv, m] using cert.hAL x y hx hxy hy
    · intro x y hx hxy hy
      exact cert.hALcont x y hx hxy hy
    · simpa [W] using cert.hWboundedLeft
    · intro t hlt htu
      exact cert.hμ_left t hlt htu
    · intro t hlt htu
      exact cert.hm_left_neg t (lt_of_lt_of_le cert.h_left_l hlt) htu
    · intro t hlt htu
      exact cert.hm_left_lower t hlt htu
  have hright : ∀ x : ℝ, up < x → x < rightDown → W x = 0 := by
    refine abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper
      (W := W) (A := cert.AR) (μDeriv := μDeriv) (m := m)
      (a := up) (b := rightDown) (r := cert.rightNear) (δ := cert.δR) (L := cert.LR)
      cert.h_up_right cert.h_up_r cert.h_r_right cert.hδR cert.hLR
      ?_ ?_ ?_ ?_ ?_ ?_ ?_ ?_
    · intro x y hx hxy hy
      exact hWcont.continuousOn.mul
        (Real.continuous_exp.comp_continuousOn (cert.hARcont x y hx hxy hy))
    · intro x y hx hxy hy t ht
      have hmt : laplaceMeanShiftRatio τ p t ≠ 0 :=
        (cert.hm_right_pos t (lt_of_lt_of_le hx ht.1) (lt_of_lt_of_le ht.2 hy.le)).ne'
      have hderiv :=
        hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
          τ hτ p q hp hq hzero t hmt
      simpa [W, μDeriv, m] using hderiv.hasDerivWithinAt
    · intro x y hx hxy hy
      simpa [μDeriv, m] using cert.hAR x y hx hxy hy
    · intro x y hx hxy hy
      exact cert.hARcont x y hx hxy hy
    · simpa [W] using cert.hWboundedRight
    · intro t hut htr
      exact cert.hμ_right t hut htr
    · intro t hut htr
      exact cert.hm_right_pos t hut (lt_trans htr cert.h_r_right)
    · intro t hut htr
      exact cert.hm_right_upper t hut htr
  exact ⟨hleft, hright⟩

/-- Recursive arbitrary-length alternating chain.

Starting from a downward crossing `leftDown`, the empty tail is the final right
outer ray sign condition.  A nonempty valid tail must begin with an upward
crossing followed by the next downward crossing and then recurse.  Singleton
tails are intentionally impossible: an alternating list cannot end at an
upward crossing. -/
def LaplaceACAlternatingChain
    (τ : ℝ) (p q : Measure ℝ) (leftDown : ℝ) : List ℝ → Type
  | [] =>
      ULift (PLift (∀ t : ℝ, leftDown < t → laplaceMeanShiftRatio τ p t < 0))
  | [_up] => ULift (PLift False)
  | up :: rightDown :: rest =>
      LaplaceACUpwardCrossingCertificate τ p q leftDown up rightDown ×
        LaplaceACAlternatingChain τ p q rightDown rest

/-- Constructor for the final right-outer base case of an alternating chain. -/
def LaplaceACAlternatingChain.rightOuter
    {τ : ℝ} {p q : Measure ℝ} {lastDown : ℝ}
    (hright : ∀ t : ℝ, lastDown < t → laplaceMeanShiftRatio τ p t < 0) :
    LaplaceACAlternatingChain τ p q lastDown [] :=
  ULift.up (PLift.up hright)

/-- Constructor for adding one upward crossing to an alternating chain. -/
def LaplaceACAlternatingChain.cons
    {τ : ℝ} {p q : Measure ℝ} {leftDown up rightDown : ℝ} {rest : List ℝ}
    (cert : LaplaceACUpwardCrossingCertificate τ p q leftDown up rightDown)
    (tail : LaplaceACAlternatingChain τ p q rightDown rest) :
    LaplaceACAlternatingChain τ p q leftDown (up :: rightDown :: rest) :=
  (cert, tail)

/-- Convenience constructor adding one upward crossing using the regularity
constructor for the local L8 certificate.

This removes the primitive/FTC/W-boundedness obligations from recursive chain
construction.  The remaining arguments are the genuine local crossing
inequalities: signs on the two adjacent gaps, positive lower bounds for `μ'`,
and one-sided linear control of `m` near the upward crossing. -/
noncomputable def LaplaceACAlternatingChain.cons_regular
    {τ : ℝ} (hτ : ValidBandwidth τ) {p q : Measure ℝ}
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    {leftDown up rightDown leftNear rightNear δL δR LL LR : ℝ} {rest : List ℝ}
    (h_left_up : leftDown < up)
    (h_up_right : up < rightDown)
    (h_left_l : leftDown < leftNear)
    (h_l_up : leftNear < up)
    (h_up_r : up < rightNear)
    (h_r_right : rightNear < rightDown)
    (hδL : 0 < δL) (hδR : 0 < δR)
    (hLL : 0 < LL) (hLR : 0 < LR)
    (hm_left_neg : ∀ t : ℝ, leftDown < t → t < up →
      laplaceMeanShiftRatio τ p t < 0)
    (hm_right_pos : ∀ t : ℝ, up < t → t < rightDown →
      0 < laplaceMeanShiftRatio τ p t)
    (hμ_left : ∀ t : ℝ, leftNear ≤ t → t < up →
      δL ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    (hm_left_lower : ∀ t : ℝ, leftNear ≤ t → t < up →
      -(LL * (up - t)) ≤ laplaceMeanShiftRatio τ p t)
    (hμ_right : ∀ t : ℝ, up < t → t < rightNear →
      δR ≤ laplaceMeanShiftRatioDeriv τ p t + 1)
    (hm_right_upper : ∀ t : ℝ, up < t → t < rightNear →
      laplaceMeanShiftRatio τ p t ≤ LR * (t - up))
    (tail : LaplaceACAlternatingChain τ p q rightDown rest) :
    LaplaceACAlternatingChain τ p q leftDown (up :: rightDown :: rest) :=
  LaplaceACAlternatingChain.cons
    (LaplaceACUpwardCrossingCertificate.of_regular τ hτ p q hp hq
      h_left_up h_up_right h_left_l h_l_up h_up_r h_r_right
      hδL hδR hLL hLR hm_left_neg hm_right_pos
      hμ_left hm_left_lower hμ_right hm_right_upper)
    tail

/-- The recursive chain produces the L9 alternating-cover tail. -/
theorem LaplaceACAlternatingChain.to_vanishesFrom
    {τ : ℝ} (hτ : ValidBandwidth τ) {p q : Measure ℝ}
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hpExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) p)
    (hqExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) q)
    (hpMoment : Integrable (fun y : ℝ => y) p) :
    ∀ {leftDown : ℝ} {rest : List ℝ},
      LaplaceACAlternatingChain τ p q leftDown rest →
        VanishesOnAlternatingUpwardPairsFrom
          (fun x : ℝ => laplaceKernelNormalizerWronskian τ p q x) leftDown rest
  | leftDown, [], hrightSign => by
      let W : ℝ → ℝ := fun x => laplaceKernelNormalizerWronskian τ p q x
      let μDeriv : ℝ → ℝ := fun x => laplaceMeanShiftRatioDeriv τ p x + 1
      let m : ℝ → ℝ := fun x => laplaceMeanShiftRatio τ p x
      have hμ_nonneg : ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1 :=
        laplaceMeanShiftRatioDeriv_add_one_nonneg_of_regular τ hτ p hp hpMoment
      have hWcont : Continuous W := by
        simpa [W] using
          continuous_laplaceKernelNormalizerWronskian_of_regular τ hτ p q hp hq
      have hWtop : Tendsto W atTop (𝓝 0) := by
        simpa [W] using
          laplaceKernelNormalizerWronskian_tendsto_atTop_zero τ hτ p q
            hpExpPos hqExpPos
      intro x hx
      exact abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg
        (W := W) (muDeriv := μDeriv) (m := m) (a := x)
        (fun u v huv huv' => hWcont.continuousOn)
        (fun u v huv huv' t ht => by
          have hlt : leftDown < t := lt_of_lt_of_le hx (le_trans huv ht.1)
          have hmt : laplaceMeanShiftRatio τ p t ≠ 0 := (hrightSign.down.down t hlt).ne
          have hderiv :=
            hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
              τ hτ p q hp hq hzero t hmt
          simpa [W, μDeriv, m] using hderiv.hasDerivWithinAt)
        (fun t _ => hμ_nonneg t)
        (fun t hxt => hrightSign.down.down t (lt_of_lt_of_le hx hxt))
        hWtop x le_rfl
  | _leftDown, [_up], hbad => False.elim hbad.down.down
  | leftDown, up :: rightDown :: rest, hchain => by
      rcases hchain with ⟨cert, htail⟩
      rcases cert.vanishes hτ hp hq hzero with ⟨hleftGap, hrightGap⟩
      exact ⟨hleftGap, hrightGap,
        LaplaceACAlternatingChain.to_vanishesFrom
          hτ hp hq hzero hpExpPos hqExpPos hpMoment htail⟩

/-- Arbitrary finite alternating final assembly.

This is the general version of the single-down and `[down, up, down]` wrappers:
the first downward crossing supplies the left outer sign, and
`LaplaceACAlternatingChain` supplies every upward-crossing flank certificate
and the final right outer sign. -/
noncomputable def laplaceACFinalAssembly_of_alternatingChain
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {firstDown : ℝ} {rest : List ℝ}
    (hpExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) p)
    (hqExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) q)
    (hpExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) p)
    (hqExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) q)
    (hpMoment : Integrable (fun y : ℝ => y) p)
    (hm_left : ∀ t : ℝ, t < firstDown → 0 < laplaceMeanShiftRatio τ p t)
    (hchain : LaplaceACAlternatingChain τ p q firstDown rest) :
    LaplaceACFinalAssembly τ p q := by
  let W : ℝ → ℝ := fun x => laplaceKernelNormalizerWronskian τ p q x
  let μDeriv : ℝ → ℝ := fun x => laplaceMeanShiftRatioDeriv τ p x + 1
  let m : ℝ → ℝ := fun x => laplaceMeanShiftRatio τ p x
  have hμ_nonneg : ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1 :=
    laplaceMeanShiftRatioDeriv_add_one_nonneg_of_regular τ hτ p hp hpMoment
  have hWcont : Continuous W := by
    simpa [W] using
      continuous_laplaceKernelNormalizerWronskian_of_regular τ hτ p q hp hq
  have hWbot : Tendsto W atBot (𝓝 0) := by
    simpa [W] using
      laplaceKernelNormalizerWronskian_tendsto_atBot_zero τ hτ p q
        hpExpNeg hqExpNeg
  have hleft : ∀ x : ℝ, x < firstDown → W x = 0 := by
    intro x hx
    exact abel_left_outer_zero_of_muDeriv_nonneg_of_m_pos
      (W := W) (muDeriv := μDeriv) (m := m) (a := x)
      (fun u v huv hvx => hWcont.continuousOn)
      (fun u v huv hvx t ht => by
        have htf : t < firstDown := lt_of_lt_of_le ht.2 (le_trans hvx hx.le)
        have hmt : laplaceMeanShiftRatio τ p t ≠ 0 := (hm_left t htf).ne'
        have hderiv :=
          hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
            τ hτ p q hp hq hzero t hmt
        simpa [W, μDeriv, m] using hderiv.hasDerivWithinAt)
      (fun t _ => hμ_nonneg t)
      (fun t htx => hm_left t (lt_of_le_of_lt htx hx))
      hWbot x le_rfl
  exact
    { breaks := firstDown :: rest
      wronskian_continuous := hWcont
      alternating_cover :=
        ⟨hleft,
          LaplaceACAlternatingChain.to_vanishesFrom
            hτ hp hq hzero hpExpPos hqExpPos hpMoment hchain⟩ }

/-- Identification theorem for an arbitrary finite alternating chain. -/
theorem laplaceAC_identifies_of_alternatingChain
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {firstDown : ℝ} {rest : List ℝ}
    (hpExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) p)
    (hqExpPos : Integrable (fun y : ℝ => Real.exp (y / τ)) q)
    (hpExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) p)
    (hqExpNeg : Integrable (fun y : ℝ => Real.exp (-y / τ)) q)
    (hpMoment : Integrable (fun y : ℝ => y) p)
    (hm_left : ∀ t : ℝ, t < firstDown → 0 < laplaceMeanShiftRatio τ p t)
    (hchain : LaplaceACAlternatingChain τ p q firstDown rest) :
    p = q :=
  laplaceAC_identifies_of_finalAssembly τ hτ p q
    (laplaceACFinalAssembly_of_alternatingChain τ hτ p q hp hq hzero
      hpExpPos hqExpPos hpExpNeg hqExpNeg hpMoment hm_left hchain)

/-! ## Continuous-density end-to-end wrapper -/

/-- User-facing finite-alternating a.c. certificate.

This is the honest end-to-end hypothesis package for the current one-dimensional
Laplace theorem.  It says:

* both laws have continuous nonnegative Lebesgue densities, so the C²-normalizer
  regularity used by Abel is derived internally;
* both laws have the two-sided exponential denominator moments used by the tail
  Wronskian limits;
* `p` has the ordinary first moment used by the tilted-mean monotonicity theorem;
* the mean-shift ratio of `p` has a finite alternating sign-change cover,
  represented by a first downward crossing, a left outer sign, and a recursive
  chain of regular upward-crossing certificates ending in the right outer sign.

The last bullet is the formal version of "finitely many sign-changing zeros" at
the current level of automation: future zero-enumeration lemmas can construct
this certificate from more syntactic assumptions such as analytic density plus
simple zeros. -/
structure LaplaceACContinuousDensityFiniteAlternating
    (τ : ℝ) (p q : Measure ℝ) where
  ρp : ℝ → ℝ
  ρq : ℝ → ℝ
  hpρ : ContinuousDensityMeasure p ρp
  hqρ : ContinuousDensityMeasure q ρq
  hpMoment : LaplaceTwoSidedExpFirstMoment τ p
  hqMoment : LaplaceTwoSidedExpFirstMoment τ q
  hpFirstMoment : Integrable (fun y : ℝ => y) p
  firstDown : ℝ
  rest : List ℝ
  hm_left : ∀ t : ℝ, t < firstDown → 0 < laplaceMeanShiftRatio τ p t
  chain : LaplaceACAlternatingChain τ p q firstDown rest

/-- The regularity certificate for `p` extracted from a continuous-density
finite-alternating package. -/
noncomputable def LaplaceACContinuousDensityFiniteAlternating.regularP
    {τ : ℝ} (hτ : ValidBandwidth τ) {p q : Measure ℝ}
    [IsFiniteMeasure p]
    (h : LaplaceACContinuousDensityFiniteAlternating τ p q) :
    LaplaceC2NormalizerRegular τ p :=
  laplaceC2NormalizerRegular_of_continuousDensity τ hτ p h.ρp h.hpρ

/-- The regularity certificate for `q` extracted from a continuous-density
finite-alternating package. -/
noncomputable def LaplaceACContinuousDensityFiniteAlternating.regularQ
    {τ : ℝ} (hτ : ValidBandwidth τ) {p q : Measure ℝ}
    [IsFiniteMeasure q]
    (h : LaplaceACContinuousDensityFiniteAlternating τ p q) :
    LaplaceC2NormalizerRegular τ q :=
  laplaceC2NormalizerRegular_of_continuousDensity τ hτ q h.ρq h.hqρ

/-- **End-to-end one-dimensional a.c. Laplace theorem, finite-alternating
continuous-density form.**

Under zero raw Laplace drift, continuous density regularity, exponential tails,
and a finite alternating sign-change certificate for the mean-shift ratio, the
target and training laws are equal.  No C²-normalizer or `HasDerivAt` hypotheses
are exposed: they are generated from the continuous-density fields. -/
theorem laplaceAC_identifies_of_continuousDensity_finiteAlternating
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (h : LaplaceACContinuousDensityFiniteAlternating τ p q) :
    p = q := by
  have hpReg : LaplaceC2NormalizerRegular τ p :=
    h.regularP hτ
  have hqReg : LaplaceC2NormalizerRegular τ q :=
    h.regularQ hτ
  exact
    laplaceAC_identifies_of_alternatingChain τ hτ p q hpReg hqReg hzero
      h.hpMoment.exp_pos h.hqMoment.exp_pos
      h.hpMoment.exp_neg h.hqMoment.exp_neg
      h.hpFirstMoment h.hm_left h.chain

/-- Single-downward-crossing constructor for the continuous-density
finite-alternating package.  This is the `M = 1` / 3A instance of the general
finite-alternating theorem. -/
def LaplaceACContinuousDensityFiniteAlternating.singleDown
    {τ : ℝ} {p q : Measure ℝ}
    (ρp ρq : ℝ → ℝ)
    (hpρ : ContinuousDensityMeasure p ρp)
    (hqρ : ContinuousDensityMeasure q ρq)
    (hpMoment : LaplaceTwoSidedExpFirstMoment τ p)
    (hqMoment : LaplaceTwoSidedExpFirstMoment τ q)
    (hpFirstMoment : Integrable (fun y : ℝ => y) p)
    {a : ℝ}
    (hm_left : ∀ t : ℝ, t < a → 0 < laplaceMeanShiftRatio τ p t)
    (hm_right : ∀ t : ℝ, a < t → laplaceMeanShiftRatio τ p t < 0) :
    LaplaceACContinuousDensityFiniteAlternating τ p q :=
  { ρp := ρp
    ρq := ρq
    hpρ := hpρ
    hqρ := hqρ
    hpMoment := hpMoment
    hqMoment := hqMoment
    hpFirstMoment := hpFirstMoment
    firstDown := a
    rest := []
    hm_left := hm_left
    chain := LaplaceACAlternatingChain.rightOuter hm_right }

end DriftingIdentifiability
