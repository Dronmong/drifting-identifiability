import DriftingIdentifiability.LaplaceAtomAlignment

/-!
# Higher-dimensional Laplace converse, milestone L3: ball-average cone extraction

This file develops the analytic content that discharges the
`LaplaceAtomConeProductData` hypothesis of `LaplaceAtomAlignment.lean`, following
the `w(ε)`-normalizer route recorded in `LaplaceHigherDim.md` §4.8 (L3).

Foundational geometric facts first: the ball is symmetric about its centre, so
the centred coordinate `x - a` integrates to zero over `B(a, ε)`, and hence the
ball-average defect of a function differentiable at `a` is `o(ε)`.
-/

open MeasureTheory Filter Topology Metric
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [MeasureSpace E] [BorelSpace E] [(volume : Measure E).IsAddHaarMeasure]
  [(volume : Measure E).IsNegInvariant]

/-! ## Reflection symmetry of the ball -/

set_option linter.unusedSectionVars false in
/-- The reflection `x ↦ 2a - x` maps `ball a ε` onto itself. -/
lemma image_reflection_ball (a : E) (ε : ℝ) :
    (fun x : E => 2 • a - x) '' Metric.ball a ε = Metric.ball a ε := by
  ext y
  simp only [Set.mem_image, Metric.mem_ball]
  constructor
  · rintro ⟨x, hx, rfl⟩
    rw [dist_eq_norm]
    have : 2 • a - x - a = -(x - a) := by rw [two_smul]; abel
    rw [this, norm_neg, ← dist_eq_norm]
    exact hx
  · intro hy
    refine ⟨2 • a - y, ?_, by abel⟩
    rw [dist_eq_norm]
    have : 2 • a - y - a = -(y - a) := by rw [two_smul]; abel
    rw [this, norm_neg, ← dist_eq_norm]
    exact hy

set_option linter.unusedSectionVars false in
/-- The reflection `x ↦ 2a - x` is measure-preserving for the volume. -/
lemma measurePreserving_reflection (a : E) :
    MeasurePreserving (fun x : E => 2 • a - x) (volume : Measure E) volume := by
  have h1 : MeasurePreserving (fun x : E => -x) (volume : Measure E) volume :=
    Measure.measurePreserving_neg volume
  have h2 : MeasurePreserving (fun x : E => 2 • a + x) (volume : Measure E) volume :=
    measurePreserving_add_left volume (2 • a)
  have heq : (fun x : E => 2 • a - x) = (fun x : E => 2 • a + x) ∘ (fun x : E => -x) := by
    funext x; simp [sub_eq_add_neg]
  rw [heq]
  exact h2.comp h1

set_option linter.unusedSectionVars false in
/-- The reflection is a measurable embedding (it is an involutive homeomorphism). -/
lemma measurableEmbedding_reflection (a : E) :
    MeasurableEmbedding (fun x : E => 2 • a - x) := by
  have heq : (fun x : E => 2 • a - x) =
      ((Homeomorph.neg E).trans (Homeomorph.addLeft (2 • a))) := by
    funext x
    simp [sub_eq_add_neg]
  rw [heq]
  exact ((Homeomorph.neg E).trans (Homeomorph.addLeft (2 • a))).measurableEmbedding

set_option linter.unusedSectionVars false in
/-- **Reflection symmetry.**  The centred coordinate integrates to zero over a
ball: `∫_{B(a,ε)} (x - a) = 0`.  This is the vanishing of the first moment of the
(symmetric) ball, and it makes the linear part of any Taylor expansion average
out. -/
lemma setIntegral_sub_center_ball_eq_zero (a : E) (ε : ℝ) :
    ∫ x in Metric.ball a ε, (x - a) ∂(volume : Measure E) = 0 := by
  have hkey :
      ∫ x in Metric.ball a ε, (x - a) ∂(volume : Measure E) =
        ∫ x in Metric.ball a ε, ((2 • a - x) - a) ∂(volume : Measure E) := by
    conv_lhs => rw [← image_reflection_ball a ε]
    rw [(measurePreserving_reflection a).setIntegral_image_emb
      (measurableEmbedding_reflection a) (fun x => x - a) (Metric.ball a ε)]
  have hneg : ∫ x in Metric.ball a ε, ((2 • a - x) - a) ∂(volume : Measure E) =
      -∫ x in Metric.ball a ε, (x - a) ∂(volume : Measure E) := by
    rw [← integral_neg]
    congr 1
    funext x
    rw [two_smul]; abel
  rw [hneg] at hkey
  have hII : (2 : ℝ) • (∫ x in Metric.ball a ε, (x - a) ∂(volume : Measure E)) = 0 := by
    rw [two_smul]
    exact add_eq_zero_iff_eq_neg.mpr hkey
  exact (smul_eq_zero.mp hII).resolve_left (by norm_num)

set_option linter.unusedSectionVars false in
/-- **Ball average of the centred coordinate vanishes**: `⨍_{B(a,ε)} (x - a) = 0`.
This is the form used in the cone-extraction argument — the linear part of a
Taylor expansion averages out over the (symmetric) ball. -/
lemma setAverage_sub_center_ball_eq_zero (a : E) (ε : ℝ) :
    ⨍ x in Metric.ball a ε, (x - a) ∂(volume : Measure E) = 0 := by
  rw [setAverage_eq, setIntegral_sub_center_ball_eq_zero, smul_zero]

/-! ## Ball-average defect of a differentiable function -/

section Analytic

variable [FiniteDimensional ℝ E]

set_option linter.unusedSectionVars false in
/-- A continuous function is integrable on any ball (finite Haar measure of the
ball, boundedness on the compact closed ball). -/
lemma integrableOn_ball_of_continuous {F : Type*} [NormedAddCommGroup F]
    {φ : E → F} (hφ : Continuous φ) (a : E) (ε : ℝ) :
    IntegrableOn φ (Metric.ball a ε) (volume : Measure E) :=
  (hφ.locallyIntegrable.integrableOn_isCompact (isCompact_closedBall a ε)).mono_set
    Metric.ball_subset_closedBall

set_option linter.unusedSectionVars false in
/-- **Ball-average defect of a function differentiable at `a` is `o(ε)`.**  For
`φ` continuous and differentiable at `a`, `ε⁻¹ • (φ a − ⨍_{B(a,ε)} φ) → 0` as
`ε → 0⁺`: the linear Taylor term averages out by reflection symmetry
(`setAverage_sub_center_ball_eq_zero`), and the remainder is `o(‖x−a‖) ≤ o(ε)`.
This is the mechanism sending the cone coefficient of the atomless part of `Z`
(and of the whole displacement field `D`) to zero. -/
lemma tendsto_setAverage_defect_of_differentiableAt
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F] [CompleteSpace F]
    {φ : E → F} {a : E} (hφc : Continuous φ) (hφ : DifferentiableAt ℝ φ a) :
    Tendsto
      (fun ε : ℝ => ε⁻¹ • (φ a - ⨍ x in Metric.ball a ε, φ x ∂(volume : Measure E)))
      (𝓝[>] (0 : ℝ)) (𝓝 0) := by
  have hfd : HasFDerivAt φ (fderiv ℝ φ a) a := hφ.hasFDerivAt
  set L := fderiv ℝ φ a with hLdef
  set r : E → F := fun x => φ x - φ a - L (x - a) with hrdef
  have hlo : r =o[𝓝 a] fun x => x - a := hfd.isLittleO
  rw [Metric.tendsto_nhdsWithin_nhds]
  intro δ hδ
  obtain ⟨ρ, hρpos, hbnd⟩ :
      ∃ ρ > 0, ∀ x ∈ Metric.ball a ρ, ‖r x‖ ≤ (δ / 2) * ‖x - a‖ := by
    have h := Asymptotics.isLittleO_iff.mp hlo (show (0 : ℝ) < δ / 2 by positivity)
    rw [Metric.eventually_nhds_iff] at h
    obtain ⟨ρ, hρpos, hb⟩ := h
    exact ⟨ρ, hρpos, fun x hx => hb (Metric.mem_ball.mp hx)⟩
  refine ⟨ρ, hρpos, fun ε hεmem hεdist => ?_⟩
  rw [Set.mem_Ioi] at hεmem
  rw [Real.dist_eq, sub_zero, abs_of_pos hεmem] at hεdist
  rw [dist_zero_right]
  -- finiteness / positivity of the ball volume
  have hVne : (volume : Measure E) (Metric.ball a ε) ≠ 0 := (measure_ball_pos _ a hεmem).ne'
  have hVlt : (volume : Measure E) (Metric.ball a ε) ≠ ⊤ := measure_ball_lt_top.ne
  set V : ℝ := (volume : Measure E).real (Metric.ball a ε) with hVdef
  have hVpos : 0 < V := ENNReal.toReal_pos hVne hVlt
  -- integrability of the pieces on the ball
  have hint_phi : IntegrableOn φ (Metric.ball a ε) volume :=
    integrableOn_ball_of_continuous hφc a ε
  have hint_lin : IntegrableOn (fun x => L (x - a)) (Metric.ball a ε) volume :=
    integrableOn_ball_of_continuous (L.continuous.comp (continuous_id.sub continuous_const)) a ε
  have hint_const : IntegrableOn (fun _ : E => φ a) (Metric.ball a ε) volume :=
    integrableOn_const hVlt
  -- ∫ L(x-a) = 0
  have hlin0 : ∫ x in Metric.ball a ε, L (x - a) ∂volume = 0 := by
    rw [show (∫ x in Metric.ball a ε, L (x - a) ∂(volume : Measure E)) =
          L (∫ x in Metric.ball a ε, (x - a) ∂volume) from
        ContinuousLinearMap.integral_comp_comm L
          (integrableOn_ball_of_continuous (continuous_id.sub continuous_const) a ε),
      setIntegral_sub_center_ball_eq_zero, map_zero]
  -- ∫ r = ∫ φ - V • φ a
  have hrint : ∫ x in Metric.ball a ε, r x ∂volume =
      (∫ x in Metric.ball a ε, φ x ∂volume) - V • φ a := by
    calc ∫ x in Metric.ball a ε, r x ∂volume
        = ∫ x in Metric.ball a ε, ((φ x - φ a) - L (x - a)) ∂volume := by
          simp only [hrdef]
      _ = (∫ x in Metric.ball a ε, (φ x - φ a) ∂volume) -
            ∫ x in Metric.ball a ε, L (x - a) ∂volume :=
          integral_sub (hint_phi.sub hint_const) hint_lin
      _ = ∫ x in Metric.ball a ε, (φ x - φ a) ∂volume := by rw [hlin0, sub_zero]
      _ = (∫ x in Metric.ball a ε, φ x ∂volume) - ∫ _ in Metric.ball a ε, φ a ∂volume :=
          integral_sub hint_phi hint_const
      _ = (∫ x in Metric.ball a ε, φ x ∂volume) - V • φ a := by rw [setIntegral_const]
  -- key identity: φ a - ⨍ φ = -(V⁻¹ • ∫ r)
  have hkey : φ a - ⨍ x in Metric.ball a ε, φ x ∂(volume : Measure E) =
      -(V⁻¹ • ∫ x in Metric.ball a ε, r x ∂volume) := by
    rw [setAverage_eq, hrint, smul_sub, smul_smul, inv_mul_cancel₀ hVpos.ne', one_smul]
    abel
  rw [hkey, smul_neg, norm_neg, smul_smul]
  -- ‖(ε⁻¹ * V⁻¹) • ∫ r‖ ≤ δ/2 < δ
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos (by positivity)]
  have hnorm_int : ‖∫ x in Metric.ball a ε, r x ∂volume‖ ≤ (δ / 2) * ε * V := by
    calc ‖∫ x in Metric.ball a ε, r x ∂volume‖
        ≤ ∫ x in Metric.ball a ε, ‖r x‖ ∂volume := norm_integral_le_integral_norm _
      _ ≤ ∫ _ in Metric.ball a ε, (δ / 2) * ε ∂volume := by
          apply setIntegral_mono_on (hint_phi.sub hint_const |>.sub hint_lin).norm
            (integrableOn_const hVlt) measurableSet_ball
          intro x hx
          calc ‖r x‖ ≤ (δ / 2) * ‖x - a‖ :=
                hbnd x (Metric.ball_subset_ball hεdist.le hx)
            _ ≤ (δ / 2) * ε := by
                have : ‖x - a‖ ≤ ε := by
                  rw [← dist_eq_norm]; exact le_of_lt (Metric.mem_ball.mp hx)
                nlinarith [this, hδ.le]
      _ = (δ / 2) * ε * V := by
          rw [setIntegral_const, smul_eq_mul]; ring
  calc ε⁻¹ * V⁻¹ * ‖∫ x in Metric.ball a ε, r x ∂volume‖
      ≤ ε⁻¹ * V⁻¹ * ((δ / 2) * ε * V) := by
        apply mul_le_mul_of_nonneg_left hnorm_int (by positivity)
    _ = δ / 2 := by field_simp
    _ < δ := by linarith

end Analytic

/-! ## The kernel's own average defect `w(a,ε)` and its lower bound -/

section KernelDefect

variable [FiniteDimensional ℝ E] [Nontrivial E]

/-- `w(a,ε) = 1 − ⨍_{B(a,ε)} e^{−‖x−a‖/τ}`, the ball-average defect of the Laplace
kernel centred at `a`.  This is the universal (measure-independent) normalizer of
the cone coefficient: for an atom `δ_a` the raw defect of `Z` is *exactly*
`w(a,ε)`, so dividing by it makes the extracted atom mass come out to `1`. -/
noncomputable def kernelAverageDefect (τ : ℝ) (a : E) (ε : ℝ) : ℝ :=
  1 - ⨍ x in Metric.ball a ε, Real.exp (-‖x - a‖ / τ) ∂(volume : Measure E)

set_option linter.unusedSectionVars false in
/-- Real Haar volume of a ball scales as `r^n`: `vol.real(B(a,r)) = r^n · κ`. -/
lemma measureReal_ball_eq (a : E) {r : ℝ} (hr : 0 ≤ r) :
    (volume : Measure E).real (Metric.ball a r) =
      r ^ Module.finrank ℝ E * (volume : Measure E).real (Metric.ball 0 1) := by
  have hfin : (volume : Measure E) (Metric.ball 0 1) ≠ ⊤ := measure_ball_lt_top.ne
  rw [Measure.real, Measure.real, Measure.addHaar_ball _ a hr, ENNReal.toReal_mul,
    ENNReal.toReal_ofReal (by positivity)]

/-- **Lower bound `w(a,ε) ≥ c·ε`** for small `ε`, with no exact constant: the
inner ball contributes at most its volume fraction `2^{-n}`, the annulus carries
`e^{−ε/(2τ)}`, and `1 − e^{−ε/(2τ)} ≥ (ε/2τ)e^{-1}`.  This is all that is needed
to make the atomless/`o(ε)` remainders vanish against `w`. -/
lemma kernelAverageDefect_ge {τ : ℝ} (hτ : 0 < τ) (a : E) {ε : ℝ}
    (hε : 0 < ε) (hετ : ε ≤ 2 * τ) :
    ε / (4 * Real.exp 1 * τ) ≤ kernelAverageDefect τ a ε := by
  set n := Module.finrank ℝ E with hn
  set κ := (volume : Measure E).real (Metric.ball 0 1) with hκ
  have hκpos : 0 < κ := by
    rw [hκ, Measure.real]
    exact ENNReal.toReal_pos (measure_ball_pos _ 0 one_pos).ne' measure_ball_lt_top.ne
  set V := (volume : Measure E).real (Metric.ball a ε) with hVdef
  have hVval : V = ε ^ n * κ := measureReal_ball_eq a hε.le
  have hVpos : 0 < V := by rw [hVval]; positivity
  have hVne : (volume : Measure E) (Metric.ball a ε) ≠ 0 := (measure_ball_pos _ a hε).ne'
  have hVlt : (volume : Measure E) (Metric.ball a ε) ≠ ⊤ := measure_ball_lt_top.ne
  -- inner ball and annulus volumes
  have hinnerval : (volume : Measure E).real (Metric.ball a (ε / 2)) = (ε / 2) ^ n * κ :=
    measureReal_ball_eq a (by positivity)
  have hsub : Metric.ball a (ε / 2) ⊆ Metric.ball a ε :=
    Metric.ball_subset_ball (by linarith)
  have hinnerlt : (volume : Measure E) (Metric.ball a (ε / 2)) ≠ ⊤ := measure_ball_lt_top.ne
  -- split the kernel integral over inner ball ∪ annulus
  have hkcont : Continuous (fun x : E => Real.exp (-‖x - a‖ / τ)) := by fun_prop
  have hkint : IntegrableOn (fun x : E => Real.exp (-‖x - a‖ / τ)) (Metric.ball a ε) volume :=
    integrableOn_ball_of_continuous hkcont a ε
  have hdisj : Disjoint (Metric.ball a (ε / 2)) (Metric.ball a ε \ Metric.ball a (ε / 2)) :=
    Set.disjoint_sdiff_right
  have huniont : Metric.ball a (ε / 2) ∪ (Metric.ball a ε \ Metric.ball a (ε / 2)) =
      Metric.ball a ε := Set.union_sdiff_cancel hsub
  have hsplit : ∫ x in Metric.ball a ε, Real.exp (-‖x - a‖ / τ) ∂volume =
      (∫ x in Metric.ball a (ε / 2), Real.exp (-‖x - a‖ / τ) ∂volume) +
        ∫ x in Metric.ball a ε \ Metric.ball a (ε / 2), Real.exp (-‖x - a‖ / τ) ∂volume := by
    have h := setIntegral_union hdisj (measurableSet_ball.diff measurableSet_ball)
      (hkint.mono_set hsub) (hkint.mono_set Set.sdiff_subset)
    rw [huniont] at h
    exact h
  -- annulus real volume
  have hannval : (volume : Measure E).real (Metric.ball a ε \ Metric.ball a (ε / 2)) =
      V - (volume : Measure E).real (Metric.ball a (ε / 2)) := by
    rw [Measure.real, Measure.real, hVdef, Measure.real,
      measure_sdiff hsub measurableSet_ball.nullMeasurableSet hinnerlt, ENNReal.toReal_sub_of_le
        (measure_mono hsub) hVlt]
  -- bound each piece: inner ≤ vol, annulus ≤ e^{-ε/2τ}·vol
  have hbound_inner : ∫ x in Metric.ball a (ε / 2), Real.exp (-‖x - a‖ / τ) ∂volume ≤
      (volume : Measure E).real (Metric.ball a (ε / 2)) := by
    calc ∫ x in Metric.ball a (ε / 2), Real.exp (-‖x - a‖ / τ) ∂volume
        ≤ ∫ _ in Metric.ball a (ε / 2), (1 : ℝ) ∂volume := by
          apply setIntegral_mono_on (hkint.mono_set hsub) (integrableOn_const hinnerlt)
            measurableSet_ball
          intro x _
          rw [Real.exp_le_one_iff]
          apply div_nonpos_of_nonpos_of_nonneg (by simp [norm_nonneg]) hτ.le
      _ = (volume : Measure E).real (Metric.ball a (ε / 2)) := by
          rw [setIntegral_const, smul_eq_mul, mul_one]
  have hbound_ann : ∫ x in Metric.ball a ε \ Metric.ball a (ε / 2),
        Real.exp (-‖x - a‖ / τ) ∂volume ≤
      Real.exp (-(ε / 2) / τ) *
        (volume : Measure E).real (Metric.ball a ε \ Metric.ball a (ε / 2)) := by
    calc ∫ x in Metric.ball a ε \ Metric.ball a (ε / 2), Real.exp (-‖x - a‖ / τ) ∂volume
        ≤ ∫ _ in Metric.ball a ε \ Metric.ball a (ε / 2), Real.exp (-(ε / 2) / τ) ∂volume := by
          apply setIntegral_mono_on (hkint.mono_set Set.sdiff_subset)
            (integrableOn_const ((measure_mono Set.sdiff_subset).trans_lt (Ne.lt_top hVlt)).ne)
            (measurableSet_ball.diff measurableSet_ball)
          intro x hx
          apply Real.exp_le_exp.mpr
          apply div_le_div_of_nonneg_right _ hτ.le
          simp only [Set.mem_sdiff, Metric.mem_ball, not_lt] at hx
          rw [neg_le_neg_iff, ← dist_eq_norm]
          exact hx.2
      _ = Real.exp (-(ε / 2) / τ) *
            (volume : Measure E).real (Metric.ball a ε \ Metric.ball a (ε / 2)) := by
          rw [setIntegral_const, smul_eq_mul, mul_comm]
  -- ∫ exp ≤ (ε/2)^n κ + e0·(V - (ε/2)^n κ)
  set e0 := Real.exp (-(ε / 2) / τ) with he0def
  have he0pos : 0 < e0 := Real.exp_pos _
  have hintbound : ∫ x in Metric.ball a ε, Real.exp (-‖x - a‖ / τ) ∂volume ≤
      (ε / 2) ^ n * κ + e0 * (V - (ε / 2) ^ n * κ) := by
    have hi := hbound_inner
    rw [hinnerval] at hi
    have hann := hbound_ann
    rw [hannval, hinnerval] at hann
    rw [hsplit]
    linarith [hi, hann]
  -- ⨍ exp ≤ t + e0·(1-t) with t = (ε/2)^n/ε^n
  have hIVeq : (ε / 2) ^ n * κ / V = (1 / 2) ^ n := by
    have hpow : (ε / 2 : ℝ) ^ n = (1 / 2) ^ n * ε ^ n := by
      rw [← mul_pow]; congr 1; ring
    rw [hVval, hpow]
    field_simp
  have havg : ⨍ x in Metric.ball a ε, Real.exp (-‖x - a‖ / τ) ∂(volume : Measure E) ≤
      (1 / 2) ^ n + e0 * (1 - (1 / 2) ^ n) := by
    rw [setAverage_eq, smul_eq_mul, ← hVdef]
    calc V⁻¹ * ∫ x in Metric.ball a ε, Real.exp (-‖x - a‖ / τ) ∂volume
        ≤ V⁻¹ * ((ε / 2) ^ n * κ + e0 * (V - (ε / 2) ^ n * κ)) :=
          mul_le_mul_of_nonneg_left hintbound (by positivity)
      _ = (ε / 2) ^ n * κ / V + e0 * (1 - (ε / 2) ^ n * κ / V) := by
          field_simp
      _ = (1 / 2) ^ n + e0 * (1 - (1 / 2) ^ n) := by rw [hIVeq]
  -- (1/2)^n ≤ 1/2  (n ≥ 1)
  have hn1 : 1 ≤ n := Module.finrank_pos
  have ht_le : ((1 : ℝ) / 2) ^ n ≤ 1 / 2 := by
    calc ((1:ℝ)/2)^n ≤ (1/2)^1 :=
          pow_le_pow_of_le_one (by norm_num) (by norm_num) hn1
      _ = 1/2 := by norm_num
  have ht_nonneg : (0:ℝ) ≤ (1/2)^n := by positivity
  -- 1 - e0 ≥ (ε/(2τ))·e^{-1}
  have hs_nonneg : 0 ≤ ε / (2 * τ) := by positivity
  have hs_le : ε / (2 * τ) ≤ 1 := by rw [div_le_one (by positivity)]; linarith
  have he0_eq : e0 = Real.exp (-(ε / (2 * τ))) := by rw [he0def]; congr 1; field_simp
  have he0_lb : (ε / (2 * τ)) * Real.exp (-1) ≤ 1 - e0 := by
    rw [he0_eq]
    set s := ε / (2 * τ) with hsdef
    have hstep1 : s * Real.exp (-s) ≤ 1 - Real.exp (-s) := by
      have h1 : s + 1 ≤ Real.exp s := Real.add_one_le_exp s
      have h2 : (s + 1) * Real.exp (-s) ≤ 1 := by
        calc (s + 1) * Real.exp (-s) ≤ Real.exp s * Real.exp (-s) :=
              mul_le_mul_of_nonneg_right h1 (Real.exp_pos _).le
          _ = 1 := by rw [← Real.exp_add, add_neg_cancel, Real.exp_zero]
      nlinarith [h2]
    have hstep2 : s * Real.exp (-1) ≤ s * Real.exp (-s) :=
      mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (by linarith)) hs_nonneg
    linarith [hstep1, hstep2]
  -- combine
  have hw_ge : (1 / 2) * ((ε / (2 * τ)) * Real.exp (-1)) ≤ kernelAverageDefect τ a ε := by
    have hexp_le : ⨍ x in Metric.ball a ε, Real.exp (-‖x - a‖ / τ) ∂(volume : Measure E) ≤
        1 - (1 / 2) * ((ε / (2 * τ)) * Real.exp (-1)) := by
      have hfac : (1 : ℝ) - (1/2)^n ≥ 1/2 := by linarith [ht_le]
      nlinarith [havg, he0_lb, ht_nonneg, he0pos.le, hfac,
        mul_nonneg hs_nonneg (Real.exp_pos (-1)).le]
    unfold kernelAverageDefect
    linarith [hexp_le]
  calc ε / (4 * Real.exp 1 * τ)
      = (1 / 2) * ((ε / (2 * τ)) * Real.exp (-1)) := by
        rw [Real.exp_neg]; field_simp; ring
    _ ≤ kernelAverageDefect τ a ε := hw_ge

end KernelDefect

end DriftingIdentifiability
