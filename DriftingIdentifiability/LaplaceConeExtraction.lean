import DriftingIdentifiability.LaplaceAtomAlignment

/-!
# Higher-dimensional Laplace converse, milestone L3: ball-average cone extraction

This file develops the analytic content that discharges the atom-alignment
conclusion of `LaplaceAtomAlignment.lean`, following the `w(ε)`-normalizer route
recorded in `LaplaceHigherDim.md` §4.8 (L3).  The older
`LaplaceAtomConeProductData` gate remains as a fixed-scale legacy interface;
the final theorem here bypasses that interface with the certified
`w`-normalized cone coefficient.

Foundational geometric facts first: the ball is symmetric about its centre, so
the centred coordinate `x - a` integrates to zero over `B(a, ε)`, and hence the
ball-average defect of a function differentiable at `a` is `o(ε)`.
-/

open MeasureTheory Filter Topology Metric
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
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

/-! ## Single-kernel cone facts -/

section SingleKernel

/-- The Laplace kernel as a function of the probe with fixed source `y`. -/
noncomputable def laplaceKernelPt (τ : ℝ) (y : E) (x : E) : ℝ :=
  Real.exp (-‖x - y‖ / τ)

set_option linter.unusedSectionVars false in
lemma laplaceKernelPt_continuous (τ : ℝ) (y : E) : Continuous (laplaceKernelPt τ y) := by
  unfold laplaceKernelPt; fun_prop

set_option linter.unusedSectionVars false in
lemma laplaceKernelPt_le_one {τ : ℝ} (hτ : 0 < τ) (y x : E) : laplaceKernelPt τ y x ≤ 1 := by
  rw [laplaceKernelPt, Real.exp_le_one_iff]
  exact div_nonpos_of_nonpos_of_nonneg (by simp [norm_nonneg]) hτ.le

set_option linter.unusedSectionVars false in
lemma laplaceKernelPt_pos {τ : ℝ} (y x : E) : 0 < laplaceKernelPt τ y x := Real.exp_pos _

set_option linter.unusedSectionVars false in
/-- `e^s − e^t ≤ s − t` for `t ≤ s ≤ 0` (`exp` is `1`-Lipschitz on `Iic 0`). -/
lemma exp_sub_exp_le_of_nonpos {s t : ℝ} (hs : s ≤ 0) (hst : t ≤ s) :
    Real.exp s - Real.exp t ≤ s - t := by
  have h1 : Real.exp s - Real.exp t = Real.exp t * (Real.exp (s - t) - 1) := by
    rw [mul_sub, mul_one, ← Real.exp_add, show t + (s - t) = s from by ring]
  have h2 : Real.exp (s - t) - 1 ≤ (s - t) * Real.exp (s - t) := by
    have hb := Real.add_one_le_exp (-(s - t))
    have he : Real.exp (-(s - t)) * Real.exp (s - t) = 1 := by
      rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
    nlinarith [hb, Real.exp_pos (s - t), he]
  calc Real.exp s - Real.exp t = Real.exp t * (Real.exp (s - t) - 1) := h1
    _ ≤ Real.exp t * ((s - t) * Real.exp (s - t)) :=
        mul_le_mul_of_nonneg_left h2 (Real.exp_pos t).le
    _ = (s - t) * Real.exp s := by
        rw [show Real.exp t * ((s - t) * Real.exp (s - t))
              = (s - t) * (Real.exp t * Real.exp (s - t)) from by ring,
          ← Real.exp_add, show t + (s - t) = s from by ring]
    _ ≤ (s - t) * 1 :=
        mul_le_mul_of_nonneg_left (Real.exp_le_one_iff.mpr hs) (by linarith)
    _ = s - t := mul_one _

set_option linter.unusedSectionVars false in
/-- `|e^u − e^v| ≤ |u − v|` for `u, v ≤ 0`. -/
lemma abs_exp_sub_exp_le_of_nonpos {u v : ℝ} (hu : u ≤ 0) (hv : v ≤ 0) :
    |Real.exp u - Real.exp v| ≤ |u - v| := by
  rcases le_total v u with h | h
  · rw [abs_of_nonneg (by linarith [Real.exp_le_exp.mpr h]),
      abs_of_nonneg (by linarith)]
    exact exp_sub_exp_le_of_nonpos hu h
  · rw [abs_of_nonpos (by linarith [Real.exp_le_exp.mpr h]),
      abs_of_nonpos (by linarith), neg_sub, neg_sub]
    exact exp_sub_exp_le_of_nonpos hv h

set_option linter.unusedSectionVars false in
/-- The kernel is `1/τ`-Lipschitz in the probe: `|κ_y(x) − κ_y(a)| ≤ (1/τ)‖x−a‖`. -/
lemma abs_laplaceKernelPt_sub_le {τ : ℝ} (hτ : 0 < τ) (y a x : E) :
    |laplaceKernelPt τ y x - laplaceKernelPt τ y a| ≤ (1 / τ) * ‖x - a‖ := by
  have harg : |(-‖x - y‖ / τ) - (-‖a - y‖ / τ)| ≤ (1 / τ) * ‖x - a‖ := by
    rw [div_sub_div_same, abs_div, abs_of_pos hτ, one_div, ← div_eq_inv_mul,
      div_le_div_iff_of_pos_right hτ]
    calc |(-‖x - y‖) - (-‖a - y‖)| = |‖a - y‖ - ‖x - y‖| := by congr 1; ring
      _ ≤ ‖(a - y) - (x - y)‖ := abs_norm_sub_norm_le _ _
      _ = ‖x - a‖ := by rw [sub_sub_sub_cancel_right, norm_sub_rev]
  calc |laplaceKernelPt τ y x - laplaceKernelPt τ y a|
      ≤ |(-‖x - y‖ / τ) - (-‖a - y‖ / τ)| :=
        abs_exp_sub_exp_le_of_nonpos
          (div_nonpos_of_nonpos_of_nonneg (by simp [norm_nonneg]) hτ.le)
          (div_nonpos_of_nonpos_of_nonneg (by simp [norm_nonneg]) hτ.le)
      _ ≤ (1 / τ) * ‖x - a‖ := harg

end SingleKernel

/-! ## The `w`-normalized cone coefficient: single-kernel limits -/

section WNormalized

variable [FiniteDimensional ℝ E] [Nontrivial E]

/-- The `w(ε)`-normalized scalar cone coefficient.  This is the variant used in
the L3 discharge route because the atom-centered kernel has coefficient exactly
`1`, avoiding the exact layer-cake constant for `⨍ ‖x-a‖`. -/
noncomputable def kernelAverageConeCoeffW (τ : ℝ) (a : E) (φ : E → ℝ) (ε : ℝ) : ℝ :=
  (kernelAverageDefect τ a ε)⁻¹ *
    (φ a - ⨍ x in Metric.ball a ε, φ x ∂(volume : Measure E))

set_option linter.unusedSectionVars false in
lemma kernelAverageDefect_pos {τ : ℝ} (hτ : 0 < τ) (a : E) {ε : ℝ}
    (hε : 0 < ε) (hετ : ε ≤ 2 * τ) :
    0 < kernelAverageDefect τ a ε := by
  have h := kernelAverageDefect_ge hτ a hε hετ
  have hpos : 0 < ε / (4 * Real.exp 1 * τ) := by positivity
  exact lt_of_lt_of_le hpos h

set_option linter.unusedSectionVars false in
/-- For the atom-centered kernel, the `w`-normalized coefficient is exactly `1`
whenever `w ≠ 0`. -/
lemma kernelAverageConeCoeffW_laplaceKernelPt_self_eq_one
    {τ : ℝ} (a : E) {ε : ℝ} (hw : kernelAverageDefect τ a ε ≠ 0) :
    kernelAverageConeCoeffW τ a (laplaceKernelPt τ a) ε = 1 := by
  unfold kernelAverageConeCoeffW kernelAverageDefect laplaceKernelPt
  have hcenter : Real.exp (-‖a - a‖ / τ) = 1 := by simp
  rw [hcenter]
  exact inv_mul_cancel₀ hw

set_option linter.unusedSectionVars false in
/-- The self-kernel coefficient tends to `1` along positive radii. -/
lemma tendsto_kernelAverageConeCoeffW_laplaceKernelPt_self
    {τ : ℝ} (hτ : 0 < τ) (a : E) :
    Tendsto (fun ε : ℝ => kernelAverageConeCoeffW τ a (laplaceKernelPt τ a) ε)
      (𝓝[>] (0 : ℝ)) (𝓝 1) := by
  rw [Metric.tendsto_nhdsWithin_nhds]
  intro δ hδ
  refine ⟨2 * τ, mul_pos (by norm_num) hτ, fun ε hεpos hεdist => ?_⟩
  have hεpos' : 0 < ε := hεpos
  have hεle : ε ≤ 2 * τ := by
    have habs : |ε| < 2 * τ := by simpa [Real.dist_eq] using hεdist
    have hlt : ε < 2 * τ := by rwa [abs_of_pos hεpos'] at habs
    exact le_of_lt hlt
  have heq := kernelAverageConeCoeffW_laplaceKernelPt_self_eq_one (τ := τ) a
    (kernelAverageDefect_pos hτ a hεpos' hεle).ne'
  simpa [heq] using hδ

set_option linter.unusedSectionVars false in
/-- The lower bound on `w(a,ε)` gives a uniform upper bound on `w(a,ε)⁻¹ * ε`.
This is the numerical heart of the dominated-convergence bound for the
`w`-normalized cone coefficient. -/
lemma inv_kernelAverageDefect_mul_radius_le
    {τ : ℝ} (hτ : 0 < τ) (a : E) {ε : ℝ}
    (hε : 0 < ε) (hετ : ε ≤ 2 * τ) :
    (kernelAverageDefect τ a ε)⁻¹ * ε ≤ 4 * Real.exp 1 * τ := by
  set C : ℝ := 4 * Real.exp 1 * τ with hC
  have hCpos : 0 < C := by positivity
  have hwpos : 0 < kernelAverageDefect τ a ε :=
    kernelAverageDefect_pos hτ a hε hετ
  have hlow := kernelAverageDefect_ge hτ a hε hετ
  have hlowC : ε / C ≤ kernelAverageDefect τ a ε := by
    simpa [C, hC] using hlow
  have hεCpos : 0 < ε / C := div_pos hε hCpos
  have hinv : (kernelAverageDefect τ a ε)⁻¹ ≤ (ε / C)⁻¹ := by
    simpa [one_div] using one_div_le_one_div_of_le hεCpos hlowC
  calc (kernelAverageDefect τ a ε)⁻¹ * ε
      ≤ (ε / C)⁻¹ * ε := mul_le_mul_of_nonneg_right hinv hε.le
    _ = C := by field_simp [hε.ne', hCpos.ne']
    _ = 4 * Real.exp 1 * τ := hC

set_option linter.unusedSectionVars false in
/-- Away from the source point, the fixed-source Laplace kernel is differentiable
at the averaging centre. -/
lemma laplaceKernelPt_differentiableAt_of_ne {τ : ℝ} {a y : E} (hy : y ≠ a) :
    DifferentiableAt ℝ (laplaceKernelPt τ y) a := by
  unfold laplaceKernelPt
  have hsub : DifferentiableAt ℝ (fun x : E => x - y) a :=
    differentiableAt_id.sub_const y
  have hsub_ne : a - y ≠ 0 := sub_ne_zero.mpr (Ne.symm hy)
  have hnorm : DifferentiableAt ℝ (fun x : E => ‖x - y‖) a :=
    hsub.norm ℝ hsub_ne
  have hscaled : DifferentiableAt ℝ (fun x : E => (-τ⁻¹) * ‖x - y‖) a :=
    hnorm.const_mul (-τ⁻¹)
  simpa [laplaceKernelPt, div_eq_mul_inv, neg_mul, mul_comm, mul_left_comm, mul_assoc]
    using hscaled.exp

set_option linter.unusedSectionVars false in
/-- Off the atom, the `w`-normalized single-kernel cone coefficient vanishes.
This is the pointwise input for the dominated-convergence atom extraction. -/
lemma tendsto_kernelAverageConeCoeffW_laplaceKernelPt_of_ne
    {τ : ℝ} (hτ : 0 < τ) {a y : E} (hy : y ≠ a) :
    Tendsto (fun ε : ℝ => kernelAverageConeCoeffW τ a (laplaceKernelPt τ y) ε)
      (𝓝[>] (0 : ℝ)) (𝓝 0) := by
  have hdef := tendsto_setAverage_defect_of_differentiableAt
    (laplaceKernelPt_continuous τ y) (laplaceKernelPt_differentiableAt_of_ne (τ := τ) hy)
  rw [Metric.tendsto_nhdsWithin_nhds]
  intro δ hδ
  set C : ℝ := 4 * Real.exp 1 * τ with hC
  have hCpos : 0 < C := by positivity
  have hη : 0 < δ / C := by positivity
  rw [Metric.tendsto_nhdsWithin_nhds] at hdef
  obtain ⟨ρ₁, hρ₁pos, hρ₁⟩ := hdef (δ / C) hη
  refine ⟨min ρ₁ (2 * τ), lt_min hρ₁pos (mul_pos (by norm_num) hτ),
    fun ε hεpos hεdist => ?_⟩
  have hερ₁ : |ε - 0| < ρ₁ := lt_of_lt_of_le hεdist (min_le_left _ _)
  have hετ_abs : |ε - 0| < 2 * τ := lt_of_lt_of_le hεdist (min_le_right _ _)
  have hεpos' : 0 < ε := hεpos
  have hερ₁dist : dist ε 0 < ρ₁ := by simpa [Real.dist_eq] using hερ₁
  have hετ : ε ≤ 2 * τ := by
    have habs : |ε| < 2 * τ := by simpa [Real.dist_eq] using hετ_abs
    have hlt : ε < 2 * τ := by rwa [abs_of_pos hεpos'] at habs
    exact le_of_lt hlt
  have hsmall := hρ₁ hεpos hερ₁dist
  rw [dist_zero_right] at hsmall ⊢
  unfold kernelAverageConeCoeffW
  set d : ℝ := laplaceKernelPt τ y a -
    ⨍ x in Metric.ball a ε, laplaceKernelPt τ y x ∂(volume : Measure E)
  have hwpos : 0 < kernelAverageDefect τ a ε :=
    kernelAverageDefect_pos hτ a hεpos' hετ
  have hbound_ratio : |(kernelAverageDefect τ a ε)⁻¹ * ε| ≤ C := by
    rw [abs_of_pos (mul_pos (inv_pos.mpr hwpos) hεpos')]
    simpa [C, hC] using inv_kernelAverageDefect_mul_radius_le hτ a hεpos' hετ
  have hdecomp :
      (kernelAverageDefect τ a ε)⁻¹ * d =
        ((kernelAverageDefect τ a ε)⁻¹ * ε) * (ε⁻¹ * d) := by
    field_simp [hεpos'.ne']
  rw [Real.norm_eq_abs, hdecomp]
  calc |((kernelAverageDefect τ a ε)⁻¹ * ε) * (ε⁻¹ * d)|
      = |(kernelAverageDefect τ a ε)⁻¹ * ε| * |ε⁻¹ * d| := abs_mul _ _
    _ ≤ C * |ε⁻¹ * d| := mul_le_mul_of_nonneg_right hbound_ratio (abs_nonneg _)
    _ < δ := by
        have hs : |ε⁻¹ * d| < δ / C := by
          simpa [d, smul_eq_mul, Real.norm_eq_abs] using hsmall
        have hmul : C * |ε⁻¹ * d| < C * (δ / C) :=
          mul_lt_mul_of_pos_left hs hCpos
        have hCmul : C * (δ / C) = δ := by field_simp [hCpos.ne']
        linarith

set_option linter.unusedSectionVars false in
/-- If a continuous scalar function differs from its centre value by at most `M`
on a ball, then the centre-minus-average defect is bounded by `M`. -/
lemma abs_sub_setAverage_le_of_forall_abs_sub_le
    {φ : E → ℝ} {a : E} {ε M : ℝ}
    (hφc : Continuous φ) (hε : 0 < ε)
    (hbound : ∀ x ∈ Metric.ball a ε, |φ a - φ x| ≤ M) :
    |φ a - ⨍ x in Metric.ball a ε, φ x ∂(volume : Measure E)| ≤ M := by
  set s := Metric.ball a ε with hs
  have hVne : (volume : Measure E) s ≠ 0 := by
    rw [hs]
    exact (measure_ball_pos _ a hε).ne'
  have hVlt : (volume : Measure E) s ≠ ⊤ := by
    have hVltTop : (volume : Measure E) s < ⊤ := by
      rw [hs]
      exact measure_ball_lt_top
    exact hVltTop.ne
  have hVpos : 0 < (volume : Measure E).real s :=
    ENNReal.toReal_pos hVne hVlt
  set V : ℝ := (volume : Measure E).real s with hVdef
  have hVpos : 0 < V := ENNReal.toReal_pos hVne hVlt
  have hint_phi : IntegrableOn φ s volume := by
    rw [hs]
    exact integrableOn_ball_of_continuous hφc a ε
  have hint_const : IntegrableOn (fun _ : E => φ a) s volume :=
    integrableOn_const hVlt
  have hint_diff : IntegrableOn (fun x : E => φ a - φ x) s volume :=
    hint_const.sub hint_phi
  have hkey :
      φ a - ⨍ x in s, φ x ∂(volume : Measure E) =
        V⁻¹ * ∫ x in s, (φ a - φ x) ∂(volume : Measure E) := by
    rw [setAverage_eq, smul_eq_mul, ← hVdef]
    rw [integral_sub hint_const hint_phi, setIntegral_const, smul_eq_mul]
    field_simp [hVpos.ne']
    ring
  have habs_int :
      |∫ x in s, (φ a - φ x) ∂(volume : Measure E)| ≤
        ∫ x in s, |φ a - φ x| ∂(volume : Measure E) := by
    simpa [Real.norm_eq_abs] using
      (norm_integral_le_integral_norm (fun x : E => φ a - φ x)
        (μ := (volume : Measure E).restrict s))
  have hint_abs : IntegrableOn (fun x : E => |φ a - φ x|) s volume := by
    exact hint_diff.norm
  have hconst_int : IntegrableOn (fun _ : E => M) s volume :=
    integrableOn_const hVlt
  have h_int_le :
      ∫ x in s, |φ a - φ x| ∂(volume : Measure E) ≤ M * V := by
    calc ∫ x in s, |φ a - φ x| ∂(volume : Measure E)
        ≤ ∫ _x in s, M ∂(volume : Measure E) := by
          apply setIntegral_mono_on hint_abs hconst_int
            (by rw [hs]; exact measurableSet_ball)
          intro x hx
          exact hbound x (by simpa [hs] using hx)
      _ = M * V := by
        rw [setIntegral_const, smul_eq_mul, hVdef]
        ring
  rw [hs] at hkey
  rw [hkey]
  calc |V⁻¹ * ∫ x in Metric.ball a ε, (φ a - φ x) ∂(volume : Measure E)|
      = V⁻¹ * |∫ x in Metric.ball a ε, (φ a - φ x) ∂(volume : Measure E)| := by
        rw [abs_mul, abs_of_pos (inv_pos.mpr hVpos)]
    _ ≤ V⁻¹ * (M * V) := by
        apply mul_le_mul_of_nonneg_left
        · exact le_trans habs_int (by simpa [hs] using h_int_le)
        · positivity
    _ = M := by field_simp [hVpos.ne']

set_option linter.unusedSectionVars false in
/-- Uniform domination for single-kernel `w`-coefficients.  This is the bounded
integrand needed in the Fubini/DCT atom-extraction step: after normalization by
the self-kernel defect, every fixed-source kernel contributes at most `4e`. -/
lemma abs_kernelAverageConeCoeffW_laplaceKernelPt_le
    {τ : ℝ} (hτ : 0 < τ) (a y : E) {ε : ℝ}
    (hε : 0 < ε) (hετ : ε ≤ 2 * τ) :
    |kernelAverageConeCoeffW τ a (laplaceKernelPt τ y) ε| ≤ 4 * Real.exp 1 := by
  have hwpos : 0 < kernelAverageDefect τ a ε :=
    kernelAverageDefect_pos hτ a hε hετ
  have hnum :
    |laplaceKernelPt τ y a -
        ⨍ x in Metric.ball a ε, laplaceKernelPt τ y x ∂(volume : Measure E)| ≤
        (1 / τ) * ε := by
    apply abs_sub_setAverage_le_of_forall_abs_sub_le
      (laplaceKernelPt_continuous τ y) hε
    intro x hx
    have hxle : ‖x - a‖ ≤ ε := by
      rw [← dist_eq_norm]
      exact le_of_lt (Metric.mem_ball.mp hx)
    calc |laplaceKernelPt τ y a - laplaceKernelPt τ y x|
        ≤ (1 / τ) * ‖a - x‖ := abs_laplaceKernelPt_sub_le hτ y x a
      _ = (1 / τ) * ‖x - a‖ := by rw [norm_sub_rev]
      _ ≤ (1 / τ) * ε := by gcongr
  unfold kernelAverageConeCoeffW
  set d : ℝ := laplaceKernelPt τ y a -
    ⨍ x in Metric.ball a ε, laplaceKernelPt τ y x ∂(volume : Measure E)
  have hratio := inv_kernelAverageDefect_mul_radius_le hτ a hε hετ
  calc |(kernelAverageDefect τ a ε)⁻¹ * d|
      = (kernelAverageDefect τ a ε)⁻¹ * |d| := by
        rw [abs_mul, abs_of_pos (inv_pos.mpr hwpos)]
    _ ≤ (kernelAverageDefect τ a ε)⁻¹ * ((1 / τ) * ε) := by
        apply mul_le_mul_of_nonneg_left
        · simpa [d] using hnum
        · positivity
    _ = ((kernelAverageDefect τ a ε)⁻¹ * ε) * (1 / τ) := by ring
    _ ≤ (4 * Real.exp 1 * τ) * (1 / τ) := by
        apply mul_le_mul_of_nonneg_right hratio
        positivity
    _ = 4 * Real.exp 1 := by field_simp [hτ.ne']

set_option linter.unusedSectionVars false in
/-- Measurability, in the source variable `y`, of the `w`-normalized coefficient
of the fixed-source kernel `x ↦ k(x,y)`.  This supplies the measurability input
for dominated convergence. -/
lemma measurable_kernelAverageConeCoeffW_laplaceKernelPt
    (τ : ℝ) (a : E) (ε : ℝ) :
    Measurable (fun y : E => kernelAverageConeCoeffW τ a (laplaceKernelPt τ y) ε) := by
  have hcenter : Measurable (fun y : E => laplaceKernelPt τ y a) := by
    have hc : Continuous (fun y : E => laplaceKernelPt τ y a) := by
      unfold laplaceKernelPt
      fun_prop
    exact hc.measurable
  have hjoint : StronglyMeasurable
      (Function.uncurry (fun x : E => fun y : E => laplaceKernelPt τ y x)) := by
    have hc : Continuous (fun z : E × E => laplaceKernelPt τ z.2 z.1) := by
      unfold laplaceKernelPt
      fun_prop
    change StronglyMeasurable (fun z : E × E => laplaceKernelPt τ z.2 z.1)
    exact hc.stronglyMeasurable
  have hIntSM : StronglyMeasurable
      (fun y : E => ∫ x in Metric.ball a ε,
        laplaceKernelPt τ y x ∂(volume : Measure E)) := by
    simpa [Measure.restrict_apply_univ] using
      (hjoint.integral_prod_left
        (μ := (volume : Measure E).restrict (Metric.ball a ε)))
  have havg : Measurable
      (fun y : E => ⨍ x in Metric.ball a ε,
        laplaceKernelPt τ y x ∂(volume : Measure E)) := by
    simp_rw [setAverage_eq, smul_eq_mul]
    exact measurable_const.mul hIntSM.measurable
  unfold kernelAverageConeCoeffW
  exact measurable_const.mul (hcenter.sub havg)

set_option linter.unusedSectionVars false in
/-- The singleton indicator integrates to the real atom mass. -/
lemma integral_indicator_singleton_one_eq_atomMassReal
    (μ : Measure E) [IsFiniteMeasure μ] (a : E) :
    ∫ y, Set.indicator ({a} : Set E) (fun _ : E => (1 : ℝ)) y ∂μ =
      atomMassReal μ a := by
  rw [integral_indicator (measurableSet_singleton a)]
  unfold atomMassReal
  rw [setIntegral_const, smul_eq_mul, mul_one, Measure.real]

set_option linter.unusedSectionVars false in
/-- **DCT atom extraction, integral form.**  Integrating the pointwise
`w`-normalized single-kernel coefficient against a finite measure extracts the
atom at the centre.  The remaining L3 Fubini socket is to identify this integral
with the actual `w`-normalized ball-average defect of `Z_μ`. -/
lemma tendsto_integral_kernelAverageConeCoeffW_laplaceKernelPt
    {τ : ℝ} (hτ : 0 < τ) (a : E) (μ : Measure E) [IsFiniteMeasure μ] :
    Tendsto
      (fun ε : ℝ =>
        ∫ y, kernelAverageConeCoeffW τ a (laplaceKernelPt τ y) ε ∂μ)
      (𝓝[>] (0 : ℝ)) (𝓝 (atomMassReal μ a)) := by
  have hmeas :
      ∀ᶠ ε in 𝓝[>] (0 : ℝ),
        AEStronglyMeasurable
          (fun y : E => kernelAverageConeCoeffW τ a (laplaceKernelPt τ y) ε) μ :=
    Eventually.of_forall fun ε =>
      (measurable_kernelAverageConeCoeffW_laplaceKernelPt τ a ε).aestronglyMeasurable
  have hsmall : ∀ᶠ ε in 𝓝[>] (0 : ℝ), ε ≤ 2 * τ :=
    eventually_nhdsWithin_of_eventually_nhds <| by
      filter_upwards [Iio_mem_nhds (show (0 : ℝ) < 2 * τ by positivity)] with ε hεlt
      exact le_of_lt hεlt
  have hbound :
      ∀ᶠ ε in 𝓝[>] (0 : ℝ),
        ∀ᵐ y ∂μ,
          ‖kernelAverageConeCoeffW τ a (laplaceKernelPt τ y) ε‖ ≤
            (fun _ : E => 4 * Real.exp 1) y := by
    filter_upwards [self_mem_nhdsWithin, hsmall] with ε hεpos hεle
    exact ae_of_all μ fun y => by
      simpa [Real.norm_eq_abs] using
        abs_kernelAverageConeCoeffW_laplaceKernelPt_le hτ a y hεpos hεle
  have hlim :
      ∀ᵐ y ∂μ,
        Tendsto
          (fun ε : ℝ => kernelAverageConeCoeffW τ a (laplaceKernelPt τ y) ε)
          (𝓝[>] (0 : ℝ))
          (𝓝 (Set.indicator ({a} : Set E) (fun _ : E => (1 : ℝ)) y)) := by
    refine ae_of_all μ fun y => ?_
    by_cases hy : y = a
    · subst y
      simpa using tendsto_kernelAverageConeCoeffW_laplaceKernelPt_self hτ a
    · have hnot : y ∉ ({a} : Set E) := by simpa using hy
      rw [Set.indicator_of_notMem hnot]
      exact tendsto_kernelAverageConeCoeffW_laplaceKernelPt_of_ne hτ hy
  have hDCT :
      Tendsto
        (fun ε : ℝ =>
          ∫ y, kernelAverageConeCoeffW τ a (laplaceKernelPt τ y) ε ∂μ)
        (𝓝[>] (0 : ℝ))
        (𝓝 (∫ y, Set.indicator ({a} : Set E) (fun _ : E => (1 : ℝ)) y ∂μ)) :=
    tendsto_integral_filter_of_dominated_convergence
      (fun _ : E => 4 * Real.exp 1) hmeas hbound (integrable_const _) hlim
  simpa [integral_indicator_singleton_one_eq_atomMassReal μ a] using hDCT

set_option linter.unusedSectionVars false in
/-- A fixed-source Laplace kernel is integrable against any finite source
measure, uniformly bounded by `1`. -/
lemma integrable_laplaceKernelPt_fixed
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    Integrable (fun y : E => laplaceKernelPt τ y x) μ := by
  refine Integrable.of_bound ?_ 1 ?_
  · apply Continuous.aestronglyMeasurable
    unfold laplaceKernelPt
    fun_prop
  · filter_upwards with y
    rw [Real.norm_eq_abs, abs_of_pos (laplaceKernelPt_pos (τ := τ) y x)]
    exact laplaceKernelPt_le_one (τ := τ) hτ y x

set_option linter.unusedSectionVars false in
/-- Product integrability of the bounded single-kernel integrand on
`B(a,ε) × μ`.  This is the Fubini input for reconciling the DCT integral form
with the actual ball-average defect of the normalizer. -/
lemma integrable_laplaceKernelPt_prod_restrict_ball
    {τ : ℝ} (hτ : 0 < τ) (a : E) (ε : ℝ)
    (μ : Measure E) [IsFiniteMeasure μ] :
    Integrable (fun z : E × E => laplaceKernelPt τ z.2 z.1)
      (((volume : Measure E).restrict (Metric.ball a ε)).prod μ) := by
  haveI : Fact ((volume : Measure E) (Metric.ball a ε) < ⊤) := ⟨measure_ball_lt_top⟩
  refine Integrable.of_bound ?_ 1 ?_
  · have hc : Continuous (fun z : E × E => laplaceKernelPt τ z.2 z.1) := by
      unfold laplaceKernelPt
      fun_prop
    exact hc.aestronglyMeasurable
  · filter_upwards with z
    rw [Real.norm_eq_abs, abs_of_pos (laplaceKernelPt_pos (τ := τ) z.2 z.1)]
    exact laplaceKernelPt_le_one (τ := τ) hτ z.2 z.1

set_option linter.unusedSectionVars false in
/-- The paper's Laplace normalizer is the source integral of the single-kernel
function used by the cone-extraction DCT layer. -/
lemma kernelNormalizer_laplace_eq_integral_laplaceKernelPt
    (τ : ℝ) (μ : Measure E) (x : E) :
    kernelNormalizer (laplaceKernel τ) μ x =
      ∫ y, laplaceKernelPt τ y x ∂μ := by
  unfold kernelNormalizer
  apply integral_congr_ae
  exact ae_of_all μ fun y => by
    rw [laplaceKernel_eq_exp]
    rfl

set_option linter.unusedSectionVars false in
/-- Fubini reconciliation for the ball integral of the Laplace normalizer. -/
lemma setIntegral_kernelNormalizer_laplace_eq_integral_setIntegral_laplaceKernelPt
    {τ : ℝ} (hτ : 0 < τ) (a : E) (ε : ℝ)
    (μ : Measure E) [IsFiniteMeasure μ] :
    ∫ x in Metric.ball a ε, kernelNormalizer (laplaceKernel τ) μ x ∂(volume : Measure E) =
      ∫ y, ∫ x in Metric.ball a ε, laplaceKernelPt τ y x ∂(volume : Measure E) ∂μ := by
  have hprod := integrable_laplaceKernelPt_prod_restrict_ball hτ a ε μ
  calc ∫ x in Metric.ball a ε, kernelNormalizer (laplaceKernel τ) μ x ∂(volume : Measure E)
      = ∫ x in Metric.ball a ε, ∫ y, laplaceKernelPt τ y x ∂μ ∂(volume : Measure E) := by
        apply integral_congr_ae
        exact ae_of_all _ fun x => kernelNormalizer_laplace_eq_integral_laplaceKernelPt τ μ x
    _ = ∫ y, ∫ x in Metric.ball a ε, laplaceKernelPt τ y x ∂(volume : Measure E) ∂μ := by
        exact integral_integral_swap
          (μ := (volume : Measure E).restrict (Metric.ball a ε)) (ν := μ)
          (f := fun x y => laplaceKernelPt τ y x) hprod

set_option linter.unusedSectionVars false in
/-- Fubini reconciliation for the averaged normalizer. -/
lemma setAverage_kernelNormalizer_laplace_eq_integral_setAverage_laplaceKernelPt
    {τ : ℝ} (hτ : 0 < τ) (a : E) (ε : ℝ)
    (μ : Measure E) [IsFiniteMeasure μ] :
    ⨍ x in Metric.ball a ε, kernelNormalizer (laplaceKernel τ) μ x ∂(volume : Measure E) =
      ∫ y, ⨍ x in Metric.ball a ε, laplaceKernelPt τ y x ∂(volume : Measure E) ∂μ := by
  have hprod := integrable_laplaceKernelPt_prod_restrict_ball hτ a ε μ
  have hinner_int : Integrable
      (fun y : E => ∫ x in Metric.ball a ε, laplaceKernelPt τ y x ∂(volume : Measure E)) μ :=
    hprod.integral_prod_right
  rw [setAverage_eq, smul_eq_mul,
    setIntegral_kernelNormalizer_laplace_eq_integral_setIntegral_laplaceKernelPt hτ a ε μ]
  simp_rw [setAverage_eq, smul_eq_mul]
  rw [integral_const_mul]

set_option linter.unusedSectionVars false in
/-- **Fubini bridge for `Z_μ`.**  The integral-form atom extraction theorem is
exactly the `w`-normalized ball-average cone coefficient of the actual
normalizer. -/
lemma integral_kernelAverageConeCoeffW_laplaceKernelPt_eq_kernelNormalizer
    {τ : ℝ} (hτ : 0 < τ) (a : E) (ε : ℝ)
    (μ : Measure E) [IsFiniteMeasure μ] :
    ∫ y, kernelAverageConeCoeffW τ a (laplaceKernelPt τ y) ε ∂μ =
      kernelAverageConeCoeffW τ a
        (fun x => kernelNormalizer (laplaceKernel τ) μ x) ε := by
  have hka : kernelNormalizer (laplaceKernel τ) μ a =
      ∫ y, laplaceKernelPt τ y a ∂μ :=
    kernelNormalizer_laplace_eq_integral_laplaceKernelPt τ μ a
  have havg : ⨍ x in Metric.ball a ε,
      kernelNormalizer (laplaceKernel τ) μ x ∂(volume : Measure E) =
      ∫ y, ⨍ x in Metric.ball a ε,
        laplaceKernelPt τ y x ∂(volume : Measure E) ∂μ :=
    setAverage_kernelNormalizer_laplace_eq_integral_setAverage_laplaceKernelPt hτ a ε μ
  have hprod := integrable_laplaceKernelPt_prod_restrict_ball hτ a ε μ
  have hint_center : Integrable (fun y : E => laplaceKernelPt τ y a) μ :=
    integrable_laplaceKernelPt_fixed hτ μ a
  have hint_avg : Integrable
      (fun y : E => ⨍ x in Metric.ball a ε,
        laplaceKernelPt τ y x ∂(volume : Measure E)) μ := by
    have hint_inner : Integrable
        (fun y : E => ∫ x in Metric.ball a ε,
          laplaceKernelPt τ y x ∂(volume : Measure E)) μ :=
      hprod.integral_prod_right
    simp_rw [setAverage_eq, smul_eq_mul]
    exact hint_inner.const_mul _
  unfold kernelAverageConeCoeffW
  change ∫ y,
      (kernelAverageDefect τ a ε)⁻¹ *
        (laplaceKernelPt τ y a -
          ⨍ x in Metric.ball a ε, laplaceKernelPt τ y x ∂(volume : Measure E)) ∂μ =
    (kernelAverageDefect τ a ε)⁻¹ *
      (kernelNormalizer (laplaceKernel τ) μ a -
        ⨍ x in Metric.ball a ε,
          kernelNormalizer (laplaceKernel τ) μ x ∂(volume : Measure E))
  rw [hka, havg]
  rw [← integral_sub hint_center hint_avg, integral_const_mul]

set_option linter.unusedSectionVars false in
/-- **Normalizer atom extraction.**  The actual `w`-normalized ball-average
defect of the Laplace normalizer detects precisely the atom mass at the centre. -/
lemma tendsto_kernelAverageConeCoeffW_kernelNormalizer_laplace
    {τ : ℝ} (hτ : 0 < τ) (a : E) (μ : Measure E) [IsFiniteMeasure μ] :
    Tendsto
      (fun ε : ℝ =>
        kernelAverageConeCoeffW τ a
          (fun x => kernelNormalizer (laplaceKernel τ) μ x) ε)
      (𝓝[>] (0 : ℝ)) (𝓝 (atomMassReal μ a)) := by
  simpa [integral_kernelAverageConeCoeffW_laplaceKernelPt_eq_kernelNormalizer hτ a]
    using tendsto_integral_kernelAverageConeCoeffW_laplaceKernelPt hτ a μ

/-! ### Vector-valued displacement-kernel cone coefficients -/

/-- The `w(ε)`-normalized vector cone coefficient. -/
noncomputable def kernelAverageConeCoeffWVec
    (τ : ℝ) (a : E) (φ : E → E) (ε : ℝ) : E :=
  (kernelAverageDefect τ a ε)⁻¹ •
    (φ a - ⨍ x in Metric.ball a ε, φ x ∂(volume : Measure E))

set_option linter.unusedSectionVars false in
/-- The atom-centered displacement kernel is odd around the centre, so its
ball integral vanishes exactly. -/
lemma setIntegral_laplaceKernelPt_self_smul_sub_eq_zero
    (τ : ℝ) (a : E) (ε : ℝ) :
    ∫ x in Metric.ball a ε,
      laplaceKernelPt τ a x • (a - x) ∂(volume : Measure E) = 0 := by
  set φ : E → E := fun x => laplaceKernelPt τ a x • (a - x) with hφ
  have hkey :
      ∫ x in Metric.ball a ε, φ x ∂(volume : Measure E) =
        ∫ x in Metric.ball a ε, φ (2 • a - x) ∂(volume : Measure E) := by
    conv_lhs => rw [← image_reflection_ball a ε]
    rw [(measurePreserving_reflection a).setIntegral_image_emb
      (measurableEmbedding_reflection a) φ (Metric.ball a ε)]
  have hneg : ∫ x in Metric.ball a ε, φ (2 • a - x) ∂(volume : Measure E) =
      -∫ x in Metric.ball a ε, φ x ∂(volume : Measure E) := by
    rw [← integral_neg]
    congr 1
    funext x
    have hk : laplaceKernelPt τ a (2 • a - x) = laplaceKernelPt τ a x := by
      unfold laplaceKernelPt
      congr 1
      congr 1
      have h₁ : 2 • a - x - a = -(x - a) := by rw [two_smul]; abel
      rw [h₁, norm_neg]
    simp only [hφ]
    calc laplaceKernelPt τ a (2 • a - x) • (a - (2 • a - x))
        = laplaceKernelPt τ a x • (x - a) := by
          rw [hk]
          congr 1
          rw [two_smul]
          abel
      _ = -(laplaceKernelPt τ a x • (a - x)) := by
          rw [show x - a = -(a - x) by abel, smul_neg]
  rw [hneg] at hkey
  have hII : (2 : ℝ) • (∫ x in Metric.ball a ε, φ x ∂(volume : Measure E)) = 0 := by
    rw [two_smul]
    exact add_eq_zero_iff_eq_neg.mpr hkey
  have hzero : ∫ x in Metric.ball a ε, φ x ∂(volume : Measure E) = 0 :=
    (smul_eq_zero.mp hII).resolve_left (by norm_num)
  simpa [φ, hφ] using hzero

set_option linter.unusedSectionVars false in
/-- The atom-centered displacement kernel has zero ball average. -/
lemma setAverage_laplaceKernelPt_self_smul_sub_eq_zero
    (τ : ℝ) (a : E) (ε : ℝ) :
    ⨍ x in Metric.ball a ε,
      laplaceKernelPt τ a x • (a - x) ∂(volume : Measure E) = 0 := by
  rw [setAverage_eq, setIntegral_laplaceKernelPt_self_smul_sub_eq_zero, smul_zero]

set_option linter.unusedSectionVars false in
/-- The self-source displacement kernel has exactly zero `w`-cone coefficient. -/
lemma kernelAverageConeCoeffWVec_laplaceDisplacementKernel_self_eq_zero
    (τ : ℝ) (a : E) (ε : ℝ) :
    kernelAverageConeCoeffWVec τ a
      (fun x => laplaceKernelPt τ a x • (a - x)) ε = 0 := by
  unfold kernelAverageConeCoeffWVec
  rw [setAverage_laplaceKernelPt_self_smul_sub_eq_zero]
  simp

set_option linter.unusedSectionVars false in
/-- The fixed-source displacement-kernel integrand is continuous in the probe. -/
lemma laplaceDisplacementKernelPt_continuous (τ : ℝ) (y : E) :
    Continuous (fun x : E => laplaceKernelPt τ y x • (y - x)) := by
  exact (laplaceKernelPt_continuous τ y).smul (continuous_const.sub continuous_id)

set_option linter.unusedSectionVars false in
/-- Away from the source point, the fixed-source displacement-kernel integrand
is differentiable at the averaging centre. -/
lemma laplaceDisplacementKernelPt_differentiableAt_of_ne
    {τ : ℝ} {a y : E} (hy : y ≠ a) :
    DifferentiableAt ℝ (fun x : E => laplaceKernelPt τ y x • (y - x)) a := by
  exact (laplaceKernelPt_differentiableAt_of_ne (τ := τ) hy).smul
    ((differentiableAt_const y).sub differentiableAt_id)

set_option linter.unusedSectionVars false in
/-- Off the source point, the vector displacement-kernel `w`-coefficient
vanishes.  Together with the exact self-source zero lemma, this is the pointwise
core for the future integrated displacement extraction. -/
lemma tendsto_kernelAverageConeCoeffWVec_laplaceDisplacementKernel_of_ne
    {τ : ℝ} (hτ : 0 < τ) {a y : E} (hy : y ≠ a) :
    Tendsto
      (fun ε : ℝ =>
        kernelAverageConeCoeffWVec τ a
          (fun x => laplaceKernelPt τ y x • (y - x)) ε)
      (𝓝[>] (0 : ℝ)) (𝓝 0) := by
  have hdef := tendsto_setAverage_defect_of_differentiableAt
    (laplaceDisplacementKernelPt_continuous τ y)
    (laplaceDisplacementKernelPt_differentiableAt_of_ne (τ := τ) hy)
  rw [Metric.tendsto_nhdsWithin_nhds]
  intro δ hδ
  set C : ℝ := 4 * Real.exp 1 * τ with hC
  have hCpos : 0 < C := by positivity
  have hη : 0 < δ / C := by positivity
  rw [Metric.tendsto_nhdsWithin_nhds] at hdef
  obtain ⟨ρ₁, hρ₁pos, hρ₁⟩ := hdef (δ / C) hη
  refine ⟨min ρ₁ (2 * τ), lt_min hρ₁pos (mul_pos (by norm_num) hτ),
    fun ε hεpos hεdist => ?_⟩
  have hερ₁ : |ε - 0| < ρ₁ := lt_of_lt_of_le hεdist (min_le_left _ _)
  have hετ_abs : |ε - 0| < 2 * τ := lt_of_lt_of_le hεdist (min_le_right _ _)
  have hεpos' : 0 < ε := hεpos
  have hερ₁dist : dist ε 0 < ρ₁ := by simpa [Real.dist_eq] using hερ₁
  have hετ : ε ≤ 2 * τ := by
    have habs : |ε| < 2 * τ := by simpa [Real.dist_eq] using hετ_abs
    have hlt : ε < 2 * τ := by rwa [abs_of_pos hεpos'] at habs
    exact le_of_lt hlt
  have hsmall := hρ₁ hεpos hερ₁dist
  rw [dist_zero_right] at hsmall ⊢
  unfold kernelAverageConeCoeffWVec
  set d : E :=
    (laplaceKernelPt τ y a • (y - a)) -
      ⨍ x in Metric.ball a ε,
        laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E)
  have hwpos : 0 < kernelAverageDefect τ a ε :=
    kernelAverageDefect_pos hτ a hεpos' hετ
  have hbound_ratio : |(kernelAverageDefect τ a ε)⁻¹ * ε| ≤ C := by
    rw [abs_of_pos (mul_pos (inv_pos.mpr hwpos) hεpos')]
    simpa [C, hC] using inv_kernelAverageDefect_mul_radius_le hτ a hεpos' hετ
  have hdecomp :
      (kernelAverageDefect τ a ε)⁻¹ • d =
        ((kernelAverageDefect τ a ε)⁻¹ * ε) • (ε⁻¹ • d) := by
    rw [smul_smul]
    congr 1
    field_simp [hεpos'.ne']
  rw [hdecomp]
  calc ‖((kernelAverageDefect τ a ε)⁻¹ * ε) • (ε⁻¹ • d)‖
      = |(kernelAverageDefect τ a ε)⁻¹ * ε| * ‖ε⁻¹ • d‖ := norm_smul _ _
    _ ≤ C * ‖ε⁻¹ • d‖ := mul_le_mul_of_nonneg_right hbound_ratio (norm_nonneg _)
    _ < δ := by
        have hs : ‖ε⁻¹ • d‖ < δ / C := by
          simpa [d] using hsmall
        have hmul : C * ‖ε⁻¹ • d‖ < C * (δ / C) :=
          mul_lt_mul_of_pos_left hs hCpos
        have hCmul : C * (δ / C) = δ := by field_simp [hCpos.ne']
        linarith

set_option linter.unusedSectionVars false in
/-- Pointwise displacement-kernel extraction: every fixed source has zero
`w`-cone coefficient in the displacement numerator.  At `y=a` this is exact odd
symmetry; away from `a` it is differentiability plus the `w ≥ cε` lower bound. -/
lemma tendsto_kernelAverageConeCoeffWVec_laplaceDisplacementKernel
    {τ : ℝ} (hτ : 0 < τ) (a y : E) :
    Tendsto
      (fun ε : ℝ =>
        kernelAverageConeCoeffWVec τ a
          (fun x => laplaceKernelPt τ y x • (y - x)) ε)
      (𝓝[>] (0 : ℝ)) (𝓝 0) := by
  by_cases hy : y = a
  · subst y
    refine tendsto_const_nhds.congr' ?_
    exact Eventually.of_forall fun ε =>
      (kernelAverageConeCoeffWVec_laplaceDisplacementKernel_self_eq_zero τ a ε).symm
  · exact tendsto_kernelAverageConeCoeffWVec_laplaceDisplacementKernel_of_ne hτ hy

set_option linter.unusedSectionVars false in
/-- Scalar derivative bound for `u ↦ u exp(-u)`: its derivative has absolute
value at most `1` on `[0,∞)`. -/
private lemma abs_one_sub_mul_exp_neg_le_one {u : ℝ} (hu : 0 ≤ u) :
    |(1 - u) * Real.exp (-u)| ≤ 1 := by
  by_cases hle : u ≤ 1
  · have hnonneg : 0 ≤ 1 - u := by linarith
    have hsuble : 1 - u ≤ 1 := by linarith
    have hexple : Real.exp (-u) ≤ 1 := Real.exp_le_one_iff.mpr (by linarith)
    have hexpnonneg : 0 ≤ Real.exp (-u) := (Real.exp_pos _).le
    rw [abs_mul, abs_of_nonneg hnonneg, abs_of_nonneg hexpnonneg]
    nlinarith
  · have hge : 1 ≤ u := le_of_not_ge hle
    have hnonpos : 1 - u ≤ 0 := by linarith
    have hexpnonneg : 0 ≤ Real.exp (-u) := (Real.exp_pos _).le
    rw [abs_mul, abs_of_nonpos hnonpos, abs_of_nonneg hexpnonneg]
    have hpart : (u - 1) * Real.exp (-u) ≤ u * Real.exp (-u) := by
      have : u - 1 ≤ u := by linarith
      exact mul_le_mul_of_nonneg_right this hexpnonneg
    have hpeak : u * Real.exp (-u) ≤ Real.exp (-1) := by
      simpa using (mul_exp_neg_div_le (τ := (1 : ℝ)) zero_lt_one hu)
    have hexp1 : Real.exp (-1 : ℝ) ≤ 1 := Real.exp_le_one_iff.mpr (by norm_num)
    nlinarith

set_option linter.unusedSectionVars false in
/-- The radial scalar function `r ↦ r exp(-r/τ)` is 1-Lipschitz on
`[0,∞)`. -/
lemma radial_mul_exp_neg_div_lipschitz
    {τ r s : ℝ} (hτ : 0 < τ) (hr : 0 ≤ r) (hs : 0 ≤ s) :
    |r * Real.exp (-r / τ) - s * Real.exp (-s / τ)| ≤ |r - s| := by
  set f : ℝ → ℝ := fun t => t * Real.exp (-t / τ) with hfdef
  have hf : ∀ t ∈ Set.Ici (0 : ℝ), DifferentiableAt ℝ f t := by
    intro t ht
    unfold f
    have hlinAt : HasDerivAt (fun u : ℝ => -u / τ) (-(1 / τ)) t := by
      simpa [one_div, neg_div] using (hasDerivAt_id t).neg.div_const τ
    exact differentiableAt_id.mul hlinAt.exp.differentiableAt
  have hderiv : ∀ t ∈ Set.Ici (0 : ℝ), ‖deriv f t‖ ≤ (1 : ℝ) := by
    intro t ht
    have ht0 : 0 ≤ t := ht
    have hdiff1 : DifferentiableAt ℝ (fun t : ℝ => t) t := differentiableAt_id
    have hlinAt : HasDerivAt (fun u : ℝ => -u / τ) (-(1 / τ)) t := by
      simpa [one_div, neg_div] using (hasDerivAt_id t).neg.div_const τ
    have hdiff2 : DifferentiableAt ℝ (fun t : ℝ => Real.exp (-t / τ)) t :=
      hlinAt.exp.differentiableAt
    have hder :
        deriv f t = (1 - t / τ) * Real.exp (-t / τ) := by
      unfold f
      change deriv ((fun t : ℝ => t) * fun t : ℝ => Real.exp (-t / τ)) t =
        (1 - t / τ) * Real.exp (-t / τ)
      rw [deriv_mul hdiff1 hdiff2, deriv_id'']
      have hexpder :
          deriv (fun x : ℝ => Real.exp (-x / τ)) t =
            Real.exp (-t / τ) * (-(1 / τ)) := hlinAt.exp.deriv
      rw [hexpder]
      field_simp [hτ.ne']
      ring
    rw [hder, Real.norm_eq_abs]
    have hnonneg : 0 ≤ t / τ := div_nonneg ht0 hτ.le
    simpa [sub_eq_add_neg, neg_div] using abs_one_sub_mul_exp_neg_le_one hnonneg
  have hmv := (convex_Ici (0 : ℝ)).norm_image_sub_le_of_norm_deriv_le
    (f := f) (C := (1 : ℝ)) hf hderiv hs hr
  simpa [f, hfdef, Real.norm_eq_abs, abs_sub_comm, one_mul] using hmv

set_option linter.unusedSectionVars false in
/-- Uniform Lipschitz bound, in the probe variable, for the fixed-source
Laplace displacement kernel.  Crucially the constant is independent of the
source point `y`, so no moment assumption on the source measure is needed. -/
lemma norm_laplaceDisplacementKernelPt_sub_le
    {τ : ℝ} (hτ : 0 < τ) (a x y : E) :
    ‖laplaceKernelPt τ y a • (y - a) -
        laplaceKernelPt τ y x • (y - x)‖ ≤
      3 * ‖x - a‖ := by
  set ka : ℝ := laplaceKernelPt τ y a with hka
  set kx : ℝ := laplaceKernelPt τ y x with hkx
  set r : ℝ := ‖y - a‖ with hr
  set s : ℝ := ‖y - x‖ with hs
  have hka_pos : 0 < ka := by simpa [ka, hka] using laplaceKernelPt_pos (τ := τ) y a
  have hkx_pos : 0 < kx := by simpa [kx, hkx] using laplaceKernelPt_pos (τ := τ) y x
  have hka_le : ka ≤ 1 := by simpa [ka, hka] using laplaceKernelPt_le_one (τ := τ) hτ y a
  have hr_nonneg : 0 ≤ r := by simp [hr]
  have hs_nonneg : 0 ≤ s := by simp [hs]
  have hrs : |r - s| ≤ ‖x - a‖ := by
    rw [hr, hs]
    calc |‖y - a‖ - ‖y - x‖|
        ≤ ‖(y - a) - (y - x)‖ := abs_norm_sub_norm_le _ _
      _ = ‖x - a‖ := by rw [sub_sub_sub_cancel_left]
  have hrad :
      |r * ka - s * kx| ≤ |r - s| := by
    have hrad' := radial_mul_exp_neg_div_lipschitz hτ hr_nonneg hs_nonneg
    have hka_eq : ka = Real.exp (-r / τ) := by
      rw [hka, hr]
      unfold laplaceKernelPt
      rw [norm_sub_rev]
    have hkx_eq : kx = Real.exp (-s / τ) := by
      rw [hkx, hs]
      unfold laplaceKernelPt
      rw [norm_sub_rev]
    simpa [hka_eq, hkx_eq, mul_comm, mul_left_comm, mul_assoc] using hrad'
  have hsecond : |ka - kx| * s ≤ 2 * ‖x - a‖ := by
    have htri : |s * ka - s * kx| ≤ |s * ka - r * ka| + |r * ka - s * kx| :=
      abs_sub_le _ _ _
    have hleft : |s * ka - r * ka| ≤ |r - s| := by
      calc |s * ka - r * ka|
          = |(s - r) * ka| := by ring_nf
        _ = |s - r| * |ka| := abs_mul _ _
        _ = |r - s| * ka := by rw [abs_sub_comm, abs_of_pos hka_pos]
        _ ≤ |r - s| * 1 := mul_le_mul_of_nonneg_left hka_le (abs_nonneg _)
        _ = |r - s| := by ring
    have hsk : |ka - kx| * s = |s * ka - s * kx| := by
      calc |ka - kx| * s
          = s * |ka - kx| := by ring
        _ = |s| * |ka - kx| := by rw [abs_of_nonneg hs_nonneg]
        _ = |s * (ka - kx)| := by rw [abs_mul]
        _ = |s * ka - s * kx| := by ring_nf
    calc |ka - kx| * s
        = |s * ka - s * kx| := hsk
      _ ≤ |s * ka - r * ka| + |r * ka - s * kx| := htri
      _ ≤ |r - s| + |r - s| := add_le_add hleft hrad
      _ ≤ ‖x - a‖ + ‖x - a‖ := add_le_add hrs hrs
      _ = 2 * ‖x - a‖ := by ring
  calc ‖laplaceKernelPt τ y a • (y - a) -
        laplaceKernelPt τ y x • (y - x)‖
      = ‖ka • (y - a) - kx • (y - x)‖ := by rw [hka, hkx]
    _ = ‖ka • ((y - a) - (y - x)) + (ka - kx) • (y - x)‖ := by
        congr 1
        module
    _ ≤ ‖ka • ((y - a) - (y - x))‖ + ‖(ka - kx) • (y - x)‖ := norm_add_le _ _
    _ = |ka| * ‖(y - a) - (y - x)‖ + |ka - kx| * ‖y - x‖ := by
        rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs]
    _ = ka * ‖x - a‖ + |ka - kx| * s := by
        rw [abs_of_pos hka_pos, hs, sub_sub_sub_cancel_left]
    _ ≤ 1 * ‖x - a‖ + 2 * ‖x - a‖ := by
        apply add_le_add
        · exact mul_le_mul_of_nonneg_right hka_le (norm_nonneg _)
        · simpa [s, hs] using hsecond
    _ = 3 * ‖x - a‖ := by ring

set_option linter.unusedSectionVars false in
/-- Vector-valued analogue of `abs_sub_setAverage_le_of_forall_abs_sub_le`. -/
lemma norm_sub_setAverage_le_of_forall_norm_sub_le
    {φ : E → E} {a : E} {ε M : ℝ}
    (hφc : Continuous φ) (hε : 0 < ε)
    (hbound : ∀ x ∈ Metric.ball a ε, ‖φ a - φ x‖ ≤ M) :
    ‖φ a - ⨍ x in Metric.ball a ε, φ x ∂(volume : Measure E)‖ ≤ M := by
  set s := Metric.ball a ε with hs
  have hVne : (volume : Measure E) s ≠ 0 := by
    rw [hs]
    exact (measure_ball_pos _ a hε).ne'
  have hVlt : (volume : Measure E) s ≠ ⊤ := by
    have hVltTop : (volume : Measure E) s < ⊤ := by
      rw [hs]
      exact measure_ball_lt_top
    exact hVltTop.ne
  set V : ℝ := (volume : Measure E).real s with hVdef
  have hVpos : 0 < V := ENNReal.toReal_pos hVne hVlt
  have hint_phi : IntegrableOn φ s volume := by
    rw [hs]
    exact integrableOn_ball_of_continuous hφc a ε
  have hint_const : IntegrableOn (fun _ : E => φ a) s volume :=
    integrableOn_const hVlt
  have hint_diff : IntegrableOn (fun x : E => φ a - φ x) s volume :=
    hint_const.sub hint_phi
  have hkey : φ a - ⨍ x in s, φ x ∂(volume : Measure E) =
      -(V⁻¹ • ∫ x in s, (φ x - φ a) ∂(volume : Measure E)) := by
    have hrint : ∫ x in s, (φ x - φ a) ∂(volume : Measure E) =
        (∫ x in s, φ x ∂(volume : Measure E)) - V • φ a := by
      rw [integral_sub hint_phi hint_const, setIntegral_const, hVdef]
    rw [setAverage_eq, ← hVdef, hrint, smul_sub, smul_smul,
      inv_mul_cancel₀ hVpos.ne', one_smul]
    abel
  have habs_int :
      ‖∫ x in s, (φ x - φ a) ∂(volume : Measure E)‖ ≤
        ∫ x in s, ‖φ x - φ a‖ ∂(volume : Measure E) :=
    norm_integral_le_integral_norm _
  have hint_norm : IntegrableOn (fun x : E => ‖φ x - φ a‖) s volume :=
    (hint_phi.sub hint_const).norm
  have hconst_int : IntegrableOn (fun _ : E => M) s volume :=
    integrableOn_const hVlt
  have h_int_le :
      ∫ x in s, ‖φ x - φ a‖ ∂(volume : Measure E) ≤ M * V := by
    calc ∫ x in s, ‖φ x - φ a‖ ∂(volume : Measure E)
        ≤ ∫ _x in s, M ∂(volume : Measure E) := by
          apply setIntegral_mono_on hint_norm hconst_int
            (by rw [hs]; exact measurableSet_ball)
          intro x hx
          calc ‖φ x - φ a‖ = ‖φ a - φ x‖ := norm_sub_rev _ _
            _ ≤ M := hbound x (by simpa [hs] using hx)
      _ = M * V := by
        rw [setIntegral_const, smul_eq_mul, hVdef]
        ring
  rw [hs] at hkey
  rw [hkey, norm_neg, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hVpos)]
  calc V⁻¹ * ‖∫ x in Metric.ball a ε, (φ x - φ a) ∂(volume : Measure E)‖
      ≤ V⁻¹ * (M * V) := by
        apply mul_le_mul_of_nonneg_left
        · exact le_trans habs_int (by simpa [hs] using h_int_le)
        · positivity
    _ = M := by field_simp [hVpos.ne']

set_option linter.unusedSectionVars false in
/-- Uniform domination for vector displacement-kernel `w`-coefficients. -/
lemma norm_kernelAverageConeCoeffWVec_laplaceDisplacementKernel_le
    {τ : ℝ} (hτ : 0 < τ) (a y : E) {ε : ℝ}
    (hε : 0 < ε) (hετ : ε ≤ 2 * τ) :
    ‖kernelAverageConeCoeffWVec τ a
        (fun x => laplaceKernelPt τ y x • (y - x)) ε‖ ≤
      12 * Real.exp 1 * τ := by
  have hwpos : 0 < kernelAverageDefect τ a ε :=
    kernelAverageDefect_pos hτ a hε hετ
  have hnum :
      ‖(laplaceKernelPt τ y a • (y - a)) -
        ⨍ x in Metric.ball a ε,
          laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E)‖ ≤
        3 * ε := by
    apply norm_sub_setAverage_le_of_forall_norm_sub_le
      (laplaceDisplacementKernelPt_continuous τ y) hε
    intro x hx
    have hxle : ‖x - a‖ ≤ ε := by
      rw [← dist_eq_norm]
      exact le_of_lt (Metric.mem_ball.mp hx)
    calc ‖laplaceKernelPt τ y a • (y - a) -
          laplaceKernelPt τ y x • (y - x)‖
        ≤ 3 * ‖x - a‖ := norm_laplaceDisplacementKernelPt_sub_le hτ a x y
      _ ≤ 3 * ε := by gcongr
  unfold kernelAverageConeCoeffWVec
  set d : E := (laplaceKernelPt τ y a • (y - a)) -
    ⨍ x in Metric.ball a ε,
      laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E)
  have hratio := inv_kernelAverageDefect_mul_radius_le hτ a hε hετ
  calc ‖(kernelAverageDefect τ a ε)⁻¹ • d‖
      = |(kernelAverageDefect τ a ε)⁻¹| * ‖d‖ := norm_smul _ _
    _ = (kernelAverageDefect τ a ε)⁻¹ * ‖d‖ := by
        rw [abs_of_pos (inv_pos.mpr hwpos)]
    _ ≤ (kernelAverageDefect τ a ε)⁻¹ * (3 * ε) := by
        apply mul_le_mul_of_nonneg_left
        · simpa [d] using hnum
        · positivity
    _ = 3 * ((kernelAverageDefect τ a ε)⁻¹ * ε) := by ring
    _ ≤ 3 * (4 * Real.exp 1 * τ) := by gcongr
    _ = 12 * Real.exp 1 * τ := by ring

set_option linter.unusedSectionVars false in
/-- Measurability, in the source variable, of the vector displacement-kernel
`w`-coefficient. -/
lemma stronglyMeasurable_kernelAverageConeCoeffWVec_laplaceDisplacementKernel
    (τ : ℝ) (a : E) (ε : ℝ) :
    StronglyMeasurable
      (fun y : E =>
        kernelAverageConeCoeffWVec τ a
          (fun x => laplaceKernelPt τ y x • (y - x)) ε) := by
  have hcenter : StronglyMeasurable
      (fun y : E => laplaceKernelPt τ y a • (y - a)) := by
    apply Continuous.stronglyMeasurable
    exact (by
      unfold laplaceKernelPt
      fun_prop : Continuous (fun y : E => laplaceKernelPt τ y a • (y - a)))
  have hjoint : StronglyMeasurable
      (Function.uncurry
        (fun x : E => fun y : E => laplaceKernelPt τ y x • (y - x))) := by
    have hc : Continuous
        (fun z : E × E => laplaceKernelPt τ z.2 z.1 • (z.2 - z.1)) := by
      unfold laplaceKernelPt
      fun_prop
    change StronglyMeasurable
      (fun z : E × E => laplaceKernelPt τ z.2 z.1 • (z.2 - z.1))
    exact hc.stronglyMeasurable
  have hIntSM : StronglyMeasurable
      (fun y : E => ∫ x in Metric.ball a ε,
        laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E)) := by
    simpa [Measure.restrict_apply_univ] using
      (hjoint.integral_prod_left
        (μ := (volume : Measure E).restrict (Metric.ball a ε)))
  have havg : StronglyMeasurable
      (fun y : E => ⨍ x in Metric.ball a ε,
        laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E)) := by
    simp_rw [setAverage_eq]
    exact hIntSM.const_smul _
  unfold kernelAverageConeCoeffWVec
  exact (hcenter.sub havg).const_smul _

set_option linter.unusedSectionVars false in
/-- **Integrated displacement extraction, DCT form.**  Integrating the
`w`-normalized fixed-source displacement-kernel coefficient against any finite
source measure gives zero in the small-ball limit. -/
lemma tendsto_integral_kernelAverageConeCoeffWVec_laplaceDisplacementKernel
    {τ : ℝ} (hτ : 0 < τ) (a : E) (μ : Measure E) [IsFiniteMeasure μ] :
    Tendsto
      (fun ε : ℝ =>
        ∫ y,
          kernelAverageConeCoeffWVec τ a
            (fun x => laplaceKernelPt τ y x • (y - x)) ε ∂μ)
      (𝓝[>] (0 : ℝ)) (𝓝 0) := by
  have hmeas :
      ∀ᶠ ε in 𝓝[>] (0 : ℝ),
        AEStronglyMeasurable
          (fun y : E =>
            kernelAverageConeCoeffWVec τ a
              (fun x => laplaceKernelPt τ y x • (y - x)) ε) μ :=
    Eventually.of_forall fun ε =>
      (stronglyMeasurable_kernelAverageConeCoeffWVec_laplaceDisplacementKernel
        τ a ε).aestronglyMeasurable
  have hsmall : ∀ᶠ ε in 𝓝[>] (0 : ℝ), ε ≤ 2 * τ :=
    eventually_nhdsWithin_of_eventually_nhds <| by
      filter_upwards [Iio_mem_nhds (show (0 : ℝ) < 2 * τ by positivity)] with ε hεlt
      exact le_of_lt hεlt
  have hbound :
      ∀ᶠ ε in 𝓝[>] (0 : ℝ),
        ∀ᵐ y ∂μ,
          ‖kernelAverageConeCoeffWVec τ a
              (fun x => laplaceKernelPt τ y x • (y - x)) ε‖ ≤
            (fun _ : E => 12 * Real.exp 1 * τ) y := by
    filter_upwards [self_mem_nhdsWithin, hsmall] with ε hεpos hεle
    exact ae_of_all μ fun y =>
      norm_kernelAverageConeCoeffWVec_laplaceDisplacementKernel_le hτ a y hεpos hεle
  have hlim :
      ∀ᵐ y ∂μ,
        Tendsto
          (fun ε : ℝ =>
            kernelAverageConeCoeffWVec τ a
              (fun x => laplaceKernelPt τ y x • (y - x)) ε)
          (𝓝[>] (0 : ℝ)) (𝓝 (0 : E)) := by
    exact ae_of_all μ fun y =>
      tendsto_kernelAverageConeCoeffWVec_laplaceDisplacementKernel hτ a y
  simpa using tendsto_integral_filter_of_dominated_convergence
    (fun _ : E => 12 * Real.exp 1 * τ) hmeas hbound (integrable_const _) hlim

set_option linter.unusedSectionVars false in
/-- Product integrability of the bounded vector displacement-kernel integrand on
`B(a,ε) × μ`. -/
lemma integrable_laplaceDisplacementKernelPt_prod_restrict_ball
    {τ : ℝ} (hτ : 0 < τ) (a : E) (ε : ℝ)
    (μ : Measure E) [IsFiniteMeasure μ] :
    Integrable
      (fun z : E × E => laplaceKernelPt τ z.2 z.1 • (z.2 - z.1))
      (((volume : Measure E).restrict (Metric.ball a ε)).prod μ) := by
  haveI : Fact ((volume : Measure E) (Metric.ball a ε) < ⊤) := ⟨measure_ball_lt_top⟩
  refine Integrable.of_bound ?_ (τ * Real.exp (-1)) ?_
  · have hc : Continuous
        (fun z : E × E => laplaceKernelPt τ z.2 z.1 • (z.2 - z.1)) := by
      unfold laplaceKernelPt
      fun_prop
    exact hc.aestronglyMeasurable
  · filter_upwards with z
    rw [norm_smul, Real.norm_eq_abs, abs_of_pos (laplaceKernelPt_pos (τ := τ) z.2 z.1)]
    unfold laplaceKernelPt
    rw [norm_sub_rev]
    simpa [mul_comm] using mul_exp_neg_div_le hτ (norm_nonneg (z.2 - z.1))

set_option linter.unusedSectionVars false in
/-- The Laplace displacement field is the source integral of the vector
fixed-source kernel used in the cone-extraction DCT layer. -/
lemma laplaceDisplacementField_eq_integral_laplaceKernelPt
    (τ : ℝ) (μ : Measure E) (x : E) :
    laplaceDisplacementField τ μ x =
      ∫ y, laplaceKernelPt τ y x • (y - x) ∂μ := by
  unfold laplaceDisplacementField
  apply integral_congr_ae
  exact ae_of_all μ fun y => by
    dsimp
    rw [laplaceKernel_eq_exp]
    rfl

set_option linter.unusedSectionVars false in
/-- Fubini reconciliation for the ball integral of the displacement field. -/
lemma setIntegral_laplaceDisplacementField_eq_integral_setIntegral_laplaceKernelPt
    {τ : ℝ} (hτ : 0 < τ) (a : E) (ε : ℝ)
    (μ : Measure E) [IsFiniteMeasure μ] :
    ∫ x in Metric.ball a ε, laplaceDisplacementField τ μ x ∂(volume : Measure E) =
      ∫ y, ∫ x in Metric.ball a ε,
        laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E) ∂μ := by
  have hprod := integrable_laplaceDisplacementKernelPt_prod_restrict_ball hτ a ε μ
  calc ∫ x in Metric.ball a ε, laplaceDisplacementField τ μ x ∂(volume : Measure E)
      = ∫ x in Metric.ball a ε,
          ∫ y, laplaceKernelPt τ y x • (y - x) ∂μ ∂(volume : Measure E) := by
        apply integral_congr_ae
        exact ae_of_all _ fun x => laplaceDisplacementField_eq_integral_laplaceKernelPt τ μ x
    _ = ∫ y, ∫ x in Metric.ball a ε,
          laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E) ∂μ := by
        exact integral_integral_swap
          (μ := (volume : Measure E).restrict (Metric.ball a ε)) (ν := μ)
          (f := fun x y => laplaceKernelPt τ y x • (y - x)) hprod

set_option linter.unusedSectionVars false in
/-- Fubini reconciliation for the averaged displacement field. -/
lemma setAverage_laplaceDisplacementField_eq_integral_setAverage_laplaceKernelPt
    {τ : ℝ} (hτ : 0 < τ) (a : E) (ε : ℝ)
    (μ : Measure E) [IsFiniteMeasure μ] :
    ⨍ x in Metric.ball a ε, laplaceDisplacementField τ μ x ∂(volume : Measure E) =
      ∫ y, ⨍ x in Metric.ball a ε,
        laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E) ∂μ := by
  have hprod := integrable_laplaceDisplacementKernelPt_prod_restrict_ball hτ a ε μ
  have hinner_int : Integrable
      (fun y : E => ∫ x in Metric.ball a ε,
        laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E)) μ :=
    hprod.integral_prod_right
  rw [setAverage_eq,
    setIntegral_laplaceDisplacementField_eq_integral_setIntegral_laplaceKernelPt hτ a ε μ]
  simp_rw [setAverage_eq]
  rw [integral_smul]

set_option linter.unusedSectionVars false in
/-- Fubini bridge for the vector displacement numerator. -/
lemma integral_kernelAverageConeCoeffWVec_laplaceDisplacementKernel_eq_displacementField
    {τ : ℝ} (hτ : 0 < τ) (a : E) (ε : ℝ)
    (μ : Measure E) [IsFiniteMeasure μ] :
    ∫ y,
        kernelAverageConeCoeffWVec τ a
          (fun x => laplaceKernelPt τ y x • (y - x)) ε ∂μ =
      kernelAverageConeCoeffWVec τ a
        (fun x => laplaceDisplacementField τ μ x) ε := by
  have hDa : laplaceDisplacementField τ μ a =
      ∫ y, laplaceKernelPt τ y a • (y - a) ∂μ :=
    laplaceDisplacementField_eq_integral_laplaceKernelPt τ μ a
  have havg : ⨍ x in Metric.ball a ε,
      laplaceDisplacementField τ μ x ∂(volume : Measure E) =
      ∫ y, ⨍ x in Metric.ball a ε,
        laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E) ∂μ :=
    setAverage_laplaceDisplacementField_eq_integral_setAverage_laplaceKernelPt hτ a ε μ
  have hprod := integrable_laplaceDisplacementKernelPt_prod_restrict_ball hτ a ε μ
  have hint_center : Integrable (fun y : E => laplaceKernelPt τ y a • (y - a)) μ :=
    (integrable_laplaceDisplacementField_integrand hτ μ a).congr
      (ae_of_all μ fun y => by
        dsimp
        rw [laplaceKernel_eq_exp]
        rfl)
  have hint_avg : Integrable
      (fun y : E => ⨍ x in Metric.ball a ε,
        laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E)) μ := by
    have hint_inner : Integrable
        (fun y : E => ∫ x in Metric.ball a ε,
          laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E)) μ :=
      hprod.integral_prod_right
    simp_rw [setAverage_eq]
    exact hint_inner.smul _
  unfold kernelAverageConeCoeffWVec
  change ∫ y,
      (kernelAverageDefect τ a ε)⁻¹ •
        ((laplaceKernelPt τ y a • (y - a)) -
          ⨍ x in Metric.ball a ε,
            laplaceKernelPt τ y x • (y - x) ∂(volume : Measure E)) ∂μ =
    (kernelAverageDefect τ a ε)⁻¹ •
      (laplaceDisplacementField τ μ a -
        ⨍ x in Metric.ball a ε,
          laplaceDisplacementField τ μ x ∂(volume : Measure E))
  rw [hDa, havg]
  rw [← integral_sub hint_center hint_avg, integral_smul]

set_option linter.unusedSectionVars false in
/-- **Integrated displacement-numerator extraction.**  The actual
`w`-normalized ball-average defect of the Laplace displacement field tends to
zero for every finite source measure. -/
lemma tendsto_kernelAverageConeCoeffWVec_laplaceDisplacementField
    {τ : ℝ} (hτ : 0 < τ) (a : E) (μ : Measure E) [IsFiniteMeasure μ] :
    Tendsto
      (fun ε : ℝ =>
        kernelAverageConeCoeffWVec τ a
          (fun x => laplaceDisplacementField τ μ x) ε)
      (𝓝[>] (0 : ℝ)) (𝓝 (0 : E)) := by
  simpa [integral_kernelAverageConeCoeffWVec_laplaceDisplacementKernel_eq_displacementField hτ a]
    using tendsto_integral_kernelAverageConeCoeffWVec_laplaceDisplacementKernel hτ a μ

/-! ### Product-rule estimates for `Z_ν • D_μ` -/

set_option linter.unusedSectionVars false in
/-- The Laplace normalizer is globally Lipschitz, with the explicit finite-mass
constant obtained by integrating the single-kernel Lipschitz bound. -/
lemma abs_kernelNormalizer_laplace_sub_le
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] (x a : E) :
    |kernelNormalizer (laplaceKernel τ) μ x -
        kernelNormalizer (laplaceKernel τ) μ a| ≤
      ((μ Set.univ).toReal / τ) * ‖x - a‖ := by
  have hx : Integrable (fun y : E => laplaceKernelPt τ y x) μ :=
    integrable_laplaceKernelPt_fixed hτ μ x
  have ha : Integrable (fun y : E => laplaceKernelPt τ y a) μ :=
    integrable_laplaceKernelPt_fixed hτ μ a
  rw [kernelNormalizer_laplace_eq_integral_laplaceKernelPt τ μ x,
    kernelNormalizer_laplace_eq_integral_laplaceKernelPt τ μ a,
    ← integral_sub hx ha]
  have hnorm :
      |∫ y, laplaceKernelPt τ y x - laplaceKernelPt τ y a ∂μ| ≤
        ∫ y, |laplaceKernelPt τ y x - laplaceKernelPt τ y a| ∂μ := by
    simpa [Real.norm_eq_abs] using
      (norm_integral_le_integral_norm
        (fun y : E => laplaceKernelPt τ y x - laplaceKernelPt τ y a)
        (μ := μ))
  have hmono :
      ∫ y, |laplaceKernelPt τ y x - laplaceKernelPt τ y a| ∂μ ≤
        ∫ _y, (1 / τ) * ‖x - a‖ ∂μ := by
    apply integral_mono
    · exact (hx.sub ha).norm
    · exact integrable_const _
    · intro y
      simpa [abs_sub_comm, norm_sub_rev] using
        abs_laplaceKernelPt_sub_le hτ y x a
  calc |∫ y, laplaceKernelPt τ y x - laplaceKernelPt τ y a ∂μ|
      ≤ ∫ y, |laplaceKernelPt τ y x - laplaceKernelPt τ y a| ∂μ := hnorm
    _ ≤ ∫ _y, (1 / τ) * ‖x - a‖ ∂μ := hmono
    _ = ((μ Set.univ).toReal / τ) * ‖x - a‖ := by
        rw [integral_const]
        simp [Measure.real, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc]

set_option linter.unusedSectionVars false in
/-- The Laplace displacement numerator is globally Lipschitz, with a coarse
constant sufficient for the product-rule cross term. -/
lemma norm_laplaceDisplacementField_sub_le
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] (x a : E) :
    ‖laplaceDisplacementField τ μ x - laplaceDisplacementField τ μ a‖ ≤
      (3 * (μ Set.univ).toReal) * ‖x - a‖ := by
  have hx : Integrable (fun y : E => laplaceKernelPt τ y x • (y - x)) μ :=
    (integrable_laplaceDisplacementField_integrand hτ μ x).congr
      (ae_of_all μ fun y => by
        dsimp
        rw [laplaceKernel_eq_exp]
        rfl)
  have ha : Integrable (fun y : E => laplaceKernelPt τ y a • (y - a)) μ :=
    (integrable_laplaceDisplacementField_integrand hτ μ a).congr
      (ae_of_all μ fun y => by
        dsimp
        rw [laplaceKernel_eq_exp]
        rfl)
  rw [laplaceDisplacementField_eq_integral_laplaceKernelPt τ μ x,
    laplaceDisplacementField_eq_integral_laplaceKernelPt τ μ a,
    ← integral_sub hx ha]
  have hnorm :
      ‖∫ y,
          laplaceKernelPt τ y x • (y - x) -
            laplaceKernelPt τ y a • (y - a) ∂μ‖ ≤
        ∫ y,
          ‖laplaceKernelPt τ y x • (y - x) -
            laplaceKernelPt τ y a • (y - a)‖ ∂μ := by
    exact norm_integral_le_integral_norm _
  have hmono :
      ∫ y,
          ‖laplaceKernelPt τ y x • (y - x) -
            laplaceKernelPt τ y a • (y - a)‖ ∂μ ≤
        ∫ _y, 3 * ‖x - a‖ ∂μ := by
    apply integral_mono
    · exact (hx.sub ha).norm
    · exact integrable_const _
    · intro y
      simpa [norm_sub_rev]
        using norm_laplaceDisplacementKernelPt_sub_le hτ a x y
  calc ‖∫ y,
          laplaceKernelPt τ y x • (y - x) -
            laplaceKernelPt τ y a • (y - a) ∂μ‖
      ≤ ∫ y,
          ‖laplaceKernelPt τ y x • (y - x) -
            laplaceKernelPt τ y a • (y - a)‖ ∂μ := hnorm
    _ ≤ ∫ _y, 3 * ‖x - a‖ ∂μ := hmono
    _ = (3 * (μ Set.univ).toReal) * ‖x - a‖ := by
        rw [integral_const]
        simp [Measure.real, mul_comm, mul_assoc]

set_option linter.unusedSectionVars false in
/-- Product expansion for the `w`-normalized vector cone coefficient.  The last
term is quadratic in the oscillations of `f` and `g`; the subsequent estimate
shows it vanishes for `f = Z_ν`, `g = D_μ`. -/
lemma kernelAverageConeCoeffWVec_smul_product_eq
    {τ : ℝ} (a : E) {ε : ℝ} (hε : 0 < ε)
    {f : E → ℝ} {g : E → E} (hf : Continuous f) (hg : Continuous g) :
    kernelAverageConeCoeffWVec τ a (fun x => f x • g x) ε =
      (kernelAverageConeCoeffW τ a f ε) • g a +
        f a • kernelAverageConeCoeffWVec τ a g ε -
          (kernelAverageDefect τ a ε)⁻¹ •
            (⨍ x in Metric.ball a ε,
              (f x - f a) • (g x - g a) ∂(volume : Measure E)) := by
  set s := Metric.ball a ε with hs
  have hs_meas : MeasurableSet s := by rw [hs]; exact measurableSet_ball
  have hVne : (volume : Measure E) s ≠ 0 := by
    rw [hs]
    exact (measure_ball_pos _ a hε).ne'
  have hVlt : (volume : Measure E) s ≠ ⊤ := by
    have hVltTop : (volume : Measure E) s < ⊤ := by
      rw [hs]
      exact measure_ball_lt_top
    exact hVltTop.ne
  have hVpos : 0 < (volume : Measure E).real s :=
    ENNReal.toReal_pos hVne hVlt
  have hf_int : IntegrableOn f s volume := by
    rw [hs]
    exact integrableOn_ball_of_continuous hf a ε
  have hg_int : IntegrableOn g s volume := by
    rw [hs]
    exact integrableOn_ball_of_continuous hg a ε
  have hfg_int : IntegrableOn (fun x => f x • g x) s volume := by
    rw [hs]
    exact integrableOn_ball_of_continuous (hf.smul hg) a ε
  have hcross_int : IntegrableOn
      (fun x => (f x - f a) • (g x - g a)) s volume := by
    rw [hs]
    exact integrableOn_ball_of_continuous
      ((hf.sub continuous_const).smul (hg.sub continuous_const)) a ε
  have hconst_int : IntegrableOn (fun _ : E => f a • g a) s volume :=
    integrableOn_const hVlt
  have hdfga_int : IntegrableOn (fun x => (f a - f x) • g a) s volume :=
    by
      rw [hs]
      exact integrableOn_ball_of_continuous
        ((continuous_const.sub hf).smul continuous_const) a ε
  have hfadg_int : IntegrableOn (fun x => f a • (g a - g x)) s volume :=
    by
      rw [hs]
      exact integrableOn_ball_of_continuous
        (continuous_const.smul (continuous_const.sub hg)) a ε
  have hdecomp :
      (fun x : E => f a • g a - f x • g x) =ᵐ[(volume : Measure E).restrict s]
        fun x => (f a - f x) • g a +
          f a • (g a - g x) -
            (f x - f a) • (g x - g a) := by
    exact ae_of_all _ fun x => by
      module
  have hint_sub : IntegrableOn (fun x => f a • g a - f x • g x) s volume :=
    hconst_int.sub hfg_int
  have hmain_int :
      ∫ x in s, f a • g a - f x • g x ∂(volume : Measure E) =
        ∫ x in s, (f a - f x) • g a ∂(volume : Measure E) +
          ∫ x in s, f a • (g a - g x) ∂(volume : Measure E) -
            ∫ x in s, (f x - f a) • (g x - g a) ∂(volume : Measure E) := by
    calc ∫ x in s, f a • g a - f x • g x ∂(volume : Measure E)
        = ∫ x in s, (f a - f x) • g a +
            f a • (g a - g x) -
              (f x - f a) • (g x - g a) ∂(volume : Measure E) := by
          exact integral_congr_ae hdecomp
      _ = ∫ x in s, ((f a - f x) • g a +
            f a • (g a - g x)) ∂(volume : Measure E) -
            ∫ x in s, (f x - f a) • (g x - g a) ∂(volume : Measure E) := by
          rw [integral_sub]
          · exact (hdfga_int.add hfadg_int)
          · exact hcross_int
      _ = ∫ x in s, (f a - f x) • g a ∂(volume : Measure E) +
            ∫ x in s, f a • (g a - g x) ∂(volume : Measure E) -
            ∫ x in s, (f x - f a) • (g x - g a) ∂(volume : Measure E) := by
          rw [integral_add hdfga_int hfadg_int]
  have hdfga :
      ∫ x in s, (f a - f x) • g a ∂(volume : Measure E) =
        (∫ x in s, f a - f x ∂(volume : Measure E)) • g a := by
    simpa using
      (integral_smul_const
        (μ := (volume : Measure E).restrict s)
        (fun x : E => f a - f x) (g a))
  have hfadg :
      ∫ x in s, f a • (g a - g x) ∂(volume : Measure E) =
        f a • (∫ x in s, g a - g x ∂(volume : Measure E)) := by
    simpa using
      (integral_smul
        (μ := (volume : Measure E).restrict s)
        (f a) (fun x : E => g a - g x))
  have hf_sub :
      ∫ x in s, f a - f x ∂(volume : Measure E) =
        (volume : Measure E).real s * f a -
          ∫ x in s, f x ∂(volume : Measure E) := by
    rw [integral_sub
      (μ := (volume : Measure E).restrict s)
      (f := fun _ : E => f a) (g := f)
      (integrableOn_const hVlt) hf_int, setIntegral_const]
    simp [smul_eq_mul, mul_comm]
  have hg_sub :
      ∫ x in s, g a - g x ∂(volume : Measure E) =
        (volume : Measure E).real s • g a -
          ∫ x in s, g x ∂(volume : Measure E) := by
    rw [integral_sub
      (μ := (volume : Measure E).restrict s)
      (f := fun _ : E => g a) (g := g)
      (integrableOn_const hVlt) hg_int, setIntegral_const]
  rw [hs] at *
  have hprod_diff :
      f a • g a -
          ((volume : Measure E).real (Metric.ball a ε))⁻¹ •
            ∫ x in Metric.ball a ε, f x • g x ∂(volume : Measure E) =
        ((volume : Measure E).real (Metric.ball a ε))⁻¹ •
          ∫ x in Metric.ball a ε,
            f a • g a - f x • g x ∂(volume : Measure E) := by
    rw [integral_sub
      (μ := (volume : Measure E).restrict (Metric.ball a ε))
      (f := fun _ : E => f a • g a) (g := fun x : E => f x • g x)
      hconst_int hfg_int, setIntegral_const]
    rw [smul_sub, smul_smul, inv_mul_cancel₀ (ne_of_gt hVpos), one_smul]
  have hf_diff :
      f a -
          ((volume : Measure E).real (Metric.ball a ε))⁻¹ *
            ∫ x in Metric.ball a ε, f x ∂(volume : Measure E) =
        ((volume : Measure E).real (Metric.ball a ε))⁻¹ *
          ∫ x in Metric.ball a ε, f a - f x ∂(volume : Measure E) := by
    rw [hf_sub]
    field_simp [ne_of_gt hVpos]
  have hg_diff :
      g a -
          ((volume : Measure E).real (Metric.ball a ε))⁻¹ •
            ∫ x in Metric.ball a ε, g x ∂(volume : Measure E) =
        ((volume : Measure E).real (Metric.ball a ε))⁻¹ •
          ∫ x in Metric.ball a ε, g a - g x ∂(volume : Measure E) := by
    rw [hg_sub]
    rw [smul_sub, smul_smul, inv_mul_cancel₀ (ne_of_gt hVpos), one_smul]
  by_cases hw : kernelAverageDefect τ a ε = 0
  · unfold kernelAverageConeCoeffWVec kernelAverageConeCoeffW
    simp [hw]
  unfold kernelAverageConeCoeffWVec kernelAverageConeCoeffW
  simp_rw [setAverage_eq, smul_eq_mul]
  rw [hprod_diff, hf_diff, hg_diff]
  rw [hmain_int]
  rw [hdfga, hfadg]
  module

set_option linter.unusedSectionVars false in
/-- Continuity of the Laplace normalizer, obtained from the explicit Lipschitz
bound above. -/
lemma continuous_kernelNormalizer_laplace_of_finite
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] :
    Continuous (fun x : E => kernelNormalizer (laplaceKernel τ) μ x) := by
  have hlip : LipschitzWith
      (Real.toNNReal (((μ Set.univ).toReal / τ)))
      (fun x : E => kernelNormalizer (laplaceKernel τ) μ x) := by
    refine LipschitzWith.of_dist_le' ?_
    intro x y
    rw [Real.dist_eq, dist_eq_norm]
    simpa [Real.norm_eq_abs] using
      abs_kernelNormalizer_laplace_sub_le hτ μ x y
  exact hlip.continuous

set_option linter.unusedSectionVars false in
/-- Continuity of the Laplace displacement numerator, obtained from the explicit
Lipschitz bound above. -/
lemma continuous_laplaceDisplacementField_of_finite
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] :
    Continuous (fun x : E => laplaceDisplacementField τ μ x) := by
  have hlip : LipschitzWith
      (Real.toNNReal (3 * (μ Set.univ).toReal))
      (fun x : E => laplaceDisplacementField τ μ x) := by
    refine LipschitzWith.of_dist_le' ?_
    intro x y
    simpa [dist_eq_norm] using norm_laplaceDisplacementField_sub_le hτ μ x y
  exact hlip.continuous

set_option linter.unusedSectionVars false in
/-- The quadratic product-rule remainder is `O(ε²/w(ε))`, hence `O(ε)`, for
the Laplace normalizer/displacement pair. -/
lemma norm_kernelAverageProductRemainder_laplace_le
    {τ : ℝ} (hτ : 0 < τ) (a : E)
    (ν μ : Measure E) [IsFiniteMeasure ν] [IsFiniteMeasure μ]
    {ε : ℝ} (hε : 0 < ε) (hετ : ε ≤ 2 * τ) :
    ‖(kernelAverageDefect τ a ε)⁻¹ •
        (⨍ x in Metric.ball a ε,
          (kernelNormalizer (laplaceKernel τ) ν x -
              kernelNormalizer (laplaceKernel τ) ν a) •
            (laplaceDisplacementField τ μ x -
              laplaceDisplacementField τ μ a) ∂(volume : Measure E))‖ ≤
      (((ν Set.univ).toReal / τ) * (3 * (μ Set.univ).toReal)) *
        (4 * Real.exp 1 * τ) * ε := by
  set LZ : ℝ := (ν Set.univ).toReal / τ with hLZ
  set LD : ℝ := 3 * (μ Set.univ).toReal with hLD
  set M : ℝ := LZ * LD * ε ^ 2 with hM
  have hLZ_nonneg : 0 ≤ LZ := by
    rw [hLZ]
    positivity
  have hLD_nonneg : 0 ≤ LD := by
    rw [hLD]
    positivity
  have hM_nonneg : 0 ≤ M := by
    rw [hM]
    positivity
  have hcross_cont : Continuous
      (fun x : E =>
        (kernelNormalizer (laplaceKernel τ) ν x -
            kernelNormalizer (laplaceKernel τ) ν a) •
          (laplaceDisplacementField τ μ x -
            laplaceDisplacementField τ μ a)) := by
    exact
      (((continuous_kernelNormalizer_laplace_of_finite hτ ν).sub continuous_const).smul
        ((continuous_laplaceDisplacementField_of_finite hτ μ).sub continuous_const))
  have havg_bound :
      ‖⨍ x in Metric.ball a ε,
          (kernelNormalizer (laplaceKernel τ) ν x -
              kernelNormalizer (laplaceKernel τ) ν a) •
            (laplaceDisplacementField τ μ x -
              laplaceDisplacementField τ μ a) ∂(volume : Measure E)‖ ≤ M := by
    have h :=
      norm_sub_setAverage_le_of_forall_norm_sub_le
        (φ := fun x : E =>
          (kernelNormalizer (laplaceKernel τ) ν x -
              kernelNormalizer (laplaceKernel τ) ν a) •
            (laplaceDisplacementField τ μ x -
              laplaceDisplacementField τ μ a))
        (a := a) hcross_cont hε
        (M := M) (by
          intro x hx
          have hxle : ‖x - a‖ ≤ ε := by
            rw [← dist_eq_norm]
            exact le_of_lt (Metric.mem_ball.mp hx)
          have hz :
              |kernelNormalizer (laplaceKernel τ) ν x -
                  kernelNormalizer (laplaceKernel τ) ν a| ≤ LZ * ‖x - a‖ := by
            simpa [LZ, hLZ] using abs_kernelNormalizer_laplace_sub_le hτ ν x a
          have hd :
              ‖laplaceDisplacementField τ μ x -
                  laplaceDisplacementField τ μ a‖ ≤ LD * ‖x - a‖ := by
            simpa [LD, hLD] using norm_laplaceDisplacementField_sub_le hτ μ x a
          simp only [sub_self, zero_smul, zero_sub, norm_neg]
          rw [norm_smul, Real.norm_eq_abs]
          have hnorm_nonneg : 0 ≤ ‖x - a‖ := norm_nonneg _
          calc |kernelNormalizer (laplaceKernel τ) ν x -
                  kernelNormalizer (laplaceKernel τ) ν a| *
                  ‖laplaceDisplacementField τ μ x -
                    laplaceDisplacementField τ μ a‖
              ≤ (LZ * ‖x - a‖) * (LD * ‖x - a‖) := by
                exact mul_le_mul hz hd (norm_nonneg _)
                  (mul_nonneg hLZ_nonneg hnorm_nonneg)
            _ ≤ (LZ * ε) * (LD * ε) := by
                gcongr
            _ = M := by
                rw [hM]
                ring)
    simpa using h
  have hwpos : 0 < kernelAverageDefect τ a ε :=
    kernelAverageDefect_pos hτ a hε hετ
  have hratio := inv_kernelAverageDefect_mul_radius_le hτ a hε hετ
  calc ‖(kernelAverageDefect τ a ε)⁻¹ •
        (⨍ x in Metric.ball a ε,
          (kernelNormalizer (laplaceKernel τ) ν x -
              kernelNormalizer (laplaceKernel τ) ν a) •
            (laplaceDisplacementField τ μ x -
              laplaceDisplacementField τ μ a) ∂(volume : Measure E))‖
      = (kernelAverageDefect τ a ε)⁻¹ *
          ‖⨍ x in Metric.ball a ε,
            (kernelNormalizer (laplaceKernel τ) ν x -
                kernelNormalizer (laplaceKernel τ) ν a) •
              (laplaceDisplacementField τ μ x -
                laplaceDisplacementField τ μ a) ∂(volume : Measure E)‖ := by
        rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hwpos)]
    _ ≤ (kernelAverageDefect τ a ε)⁻¹ * M := by
        exact mul_le_mul_of_nonneg_left havg_bound (inv_nonneg.mpr hwpos.le)
    _ = (LZ * LD) * ((kernelAverageDefect τ a ε)⁻¹ * ε) * ε := by
        rw [hM]
        ring
    _ ≤ (LZ * LD) * (4 * Real.exp 1 * τ) * ε := by
        gcongr
    _ = (((ν Set.univ).toReal / τ) * (3 * (μ Set.univ).toReal)) *
        (4 * Real.exp 1 * τ) * ε := by
        rw [hLZ, hLD]

set_option linter.unusedSectionVars false in
/-- The product-rule cross term tends to zero for the Laplace normalizer and
displacement numerator. -/
lemma tendsto_kernelAverageProductRemainder_laplace
    {τ : ℝ} (hτ : 0 < τ) (a : E)
    (ν μ : Measure E) [IsFiniteMeasure ν] [IsFiniteMeasure μ] :
    Tendsto
      (fun ε : ℝ =>
        (kernelAverageDefect τ a ε)⁻¹ •
          (⨍ x in Metric.ball a ε,
            (kernelNormalizer (laplaceKernel τ) ν x -
                kernelNormalizer (laplaceKernel τ) ν a) •
              (laplaceDisplacementField τ μ x -
                laplaceDisplacementField τ μ a) ∂(volume : Measure E)))
      (𝓝[>] (0 : ℝ)) (𝓝 (0 : E)) := by
  set K : ℝ :=
    (((ν Set.univ).toReal / τ) * (3 * (μ Set.univ).toReal)) *
      (4 * Real.exp 1 * τ) with hK
  have hK_nonneg : 0 ≤ K := by
    rw [hK]
    positivity
  rw [tendsto_zero_iff_norm_tendsto_zero]
  refine squeeze_zero'
    (f := fun ε : ℝ =>
      ‖(kernelAverageDefect τ a ε)⁻¹ •
          (⨍ x in Metric.ball a ε,
            (kernelNormalizer (laplaceKernel τ) ν x -
                kernelNormalizer (laplaceKernel τ) ν a) •
              (laplaceDisplacementField τ μ x -
                laplaceDisplacementField τ μ a) ∂(volume : Measure E))‖)
    (g := fun ε : ℝ => K * ε)
    (Eventually.of_forall fun ε => norm_nonneg _) ?_ ?_
  · have hsmall : ∀ᶠ ε in 𝓝[>] (0 : ℝ), ε ≤ 2 * τ :=
      eventually_nhdsWithin_of_eventually_nhds <| by
        filter_upwards [Iio_mem_nhds (show (0 : ℝ) < 2 * τ by positivity)] with ε hεlt
        exact le_of_lt hεlt
    filter_upwards [self_mem_nhdsWithin, hsmall] with ε hεpos hεle
    simpa [K, hK] using
      norm_kernelAverageProductRemainder_laplace_le hτ a ν μ hεpos hεle
  · have hε0 : Tendsto (fun ε : ℝ => ε) (𝓝[>] (0 : ℝ)) (𝓝 0) :=
      tendsto_nhdsWithin_of_tendsto_nhds tendsto_id
    simpa using (tendsto_const_nhds.mul hε0 : Tendsto (fun ε : ℝ => K * ε)
      (𝓝[>] (0 : ℝ)) (𝓝 (K * 0)))

set_option linter.unusedSectionVars false in
/-- **Product rule for the Laplace atom cone coefficient.**  The actual
`w`-normalized cone coefficient of `Z_ν • D_μ` extracts exactly the atom mass
of `ν` times the displacement numerator of `μ` at the centre. -/
lemma tendsto_kernelAverageConeCoeffWVec_laplaceNormalizerDisplacementProduct
    {τ : ℝ} (hτ : 0 < τ) (a : E)
    (ν μ : Measure E) [IsFiniteMeasure ν] [IsFiniteMeasure μ] :
    Tendsto
      (fun ε : ℝ =>
        kernelAverageConeCoeffWVec τ a
          (laplaceNormalizerDisplacementProduct τ ν μ) ε)
      (𝓝[>] (0 : ℝ))
      (𝓝 (atomMassReal ν a • laplaceDisplacementField τ μ a)) := by
  have hZ :=
    tendsto_kernelAverageConeCoeffW_kernelNormalizer_laplace hτ a ν
  have hD :=
    tendsto_kernelAverageConeCoeffWVec_laplaceDisplacementField hτ a μ
  have hR :=
    tendsto_kernelAverageProductRemainder_laplace hτ a ν μ
  have hprod :
      Tendsto
        (fun ε : ℝ =>
          (kernelAverageConeCoeffW τ a
            (fun x => kernelNormalizer (laplaceKernel τ) ν x) ε) •
              laplaceDisplacementField τ μ a +
            kernelNormalizer (laplaceKernel τ) ν a •
              kernelAverageConeCoeffWVec τ a
                (fun x => laplaceDisplacementField τ μ x) ε -
            (kernelAverageDefect τ a ε)⁻¹ •
              (⨍ x in Metric.ball a ε,
                (kernelNormalizer (laplaceKernel τ) ν x -
                    kernelNormalizer (laplaceKernel τ) ν a) •
                  (laplaceDisplacementField τ μ x -
                    laplaceDisplacementField τ μ a) ∂(volume : Measure E)))
        (𝓝[>] (0 : ℝ))
        (𝓝 (atomMassReal ν a • laplaceDisplacementField τ μ a)) := by
    have hfirst :
        Tendsto
          (fun ε : ℝ =>
            (kernelAverageConeCoeffW τ a
              (fun x => kernelNormalizer (laplaceKernel τ) ν x) ε) •
                laplaceDisplacementField τ μ a)
          (𝓝[>] (0 : ℝ))
          (𝓝 (atomMassReal ν a • laplaceDisplacementField τ μ a)) :=
      hZ.smul tendsto_const_nhds
    have hsecond :
        Tendsto
          (fun ε : ℝ =>
            kernelNormalizer (laplaceKernel τ) ν a •
              kernelAverageConeCoeffWVec τ a
                (fun x => laplaceDisplacementField τ μ x) ε)
          (𝓝[>] (0 : ℝ)) (𝓝 (0 : E)) := by
      have hconst :
          Tendsto (fun _ : ℝ => kernelNormalizer (laplaceKernel τ) ν a)
            (𝓝[>] (0 : ℝ)) (𝓝 (kernelNormalizer (laplaceKernel τ) ν a)) :=
        tendsto_const_nhds
      simpa using hconst.smul hD
    simpa [sub_eq_add_neg] using (hfirst.add hsecond).sub hR
  refine hprod.congr' ?_
  filter_upwards [self_mem_nhdsWithin] with ε hεpos
  symm
  unfold laplaceNormalizerDisplacementProduct
  exact kernelAverageConeCoeffWVec_smul_product_eq
    (τ := τ) a hεpos
    (continuous_kernelNormalizer_laplace_of_finite hτ ν)
    (continuous_laplaceDisplacementField_of_finite hτ μ)

/-! ### Unconditional `w`-normalized L3 discharge -/

set_option linter.unusedSectionVars false in
/-- **L3 atom alignment, discharged.**  The `w(ε)`-normalized cone-extraction
theorem supplies the analytic input directly: under zero raw Laplace drift, the
atom masses and displacement numerators align at every point.

This is the unconditional replacement for the earlier legacy
`LaplaceAtomConeProductData` gate, whose statement used the old fixed
`((n+1)τ)/(nε)` scale. -/
theorem laplaceZeroDrift_atomAlignment_of_coneExtraction
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (a : E)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    atomMassReal q a • laplaceDisplacementField τ p a =
      atomMassReal p a • laplaceDisplacementField τ q a := by
  have hfun := laplaceNormalizerDisplacementProduct_eq_of_zeroDrift hτ p q hzero
  have hleft :
      Tendsto
        (fun ε : ℝ =>
          kernelAverageConeCoeffWVec τ a
            (laplaceNormalizerDisplacementProduct τ q p) ε)
        (𝓝[>] (0 : ℝ))
        (𝓝 (atomMassReal q a • laplaceDisplacementField τ p a)) :=
    tendsto_kernelAverageConeCoeffWVec_laplaceNormalizerDisplacementProduct hτ a q p
  have hright :
      Tendsto
        (fun ε : ℝ =>
          kernelAverageConeCoeffWVec τ a
            (laplaceNormalizerDisplacementProduct τ q p) ε)
        (𝓝[>] (0 : ℝ))
        (𝓝 (atomMassReal p a • laplaceDisplacementField τ q a)) := by
    refine
      (tendsto_kernelAverageConeCoeffWVec_laplaceNormalizerDisplacementProduct
        hτ a p q).congr' ?_
    exact Eventually.of_forall fun ε => by
      rw [hfun]
  exact tendsto_nhds_unique hleft hright

set_option linter.unusedSectionVars false in
/-- Scalar mass-ratio consequence of the discharged L3 atom alignment. -/
theorem laplaceZeroDrift_atomMassRatio_of_coneExtraction
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (a : E)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hDp_ne : laplaceDisplacementField τ p a ≠ 0) :
    atomMassReal q a * kernelNormalizer (laplaceKernel τ) p a =
      atomMassReal p a * kernelNormalizer (laplaceKernel τ) q a := by
  set Zp := kernelNormalizer (laplaceKernel τ) p a with hZp
  set Zq := kernelNormalizer (laplaceKernel τ) q a with hZq
  set Dp := laplaceDisplacementField τ p a with hDp_def
  set Dq := laplaceDisplacementField τ q a with hDq_def
  have hDp_ne' : Dp ≠ 0 := by
    rw [hDp_def]
    exact hDp_ne
  have hZp_pos : 0 < Zp := by
    rw [hZp]
    exact laplaceKernelNormalizer_pos p τ hτ a
  have hdisp : Zq • Dp = Zp • Dq := by
    rw [hZp, hZq, hDp_def, hDq_def]
    exact zeroDrift_displacementAligned hτ p q hzero a
  have hDq : Dq = (Zp⁻¹ * Zq) • Dp := by
    calc Dq = Zp⁻¹ • (Zp • Dq) := by
          rw [inv_smul_smul₀ hZp_pos.ne']
      _ = Zp⁻¹ • (Zq • Dp) := by rw [hdisp]
      _ = (Zp⁻¹ * Zq) • Dp := by rw [smul_smul]
  have halign : atomMassReal q a • Dp = atomMassReal p a • Dq := by
    rw [hDp_def, hDq_def]
    exact laplaceZeroDrift_atomAlignment_of_coneExtraction hτ p q a hzero
  rw [hDq] at halign
  rw [smul_smul] at halign
  have hscalar :
      atomMassReal q a = atomMassReal p a * (Zp⁻¹ * Zq) :=
    smul_left_injective ℝ hDp_ne' halign
  calc atomMassReal q a * Zp
      = (atomMassReal p a * (Zp⁻¹ * Zq)) * Zp := by rw [hscalar]
    _ = atomMassReal p a * Zq := by
        field_simp [hZp_pos.ne']

set_option linter.unusedSectionVars false in
/-- Nonzero-displacement atom-rigidity consequence of the discharged L3 atom
alignment. -/
theorem laplaceZeroDrift_atomMass_zero_iff_of_coneExtraction
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (a : E)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hDp_ne : laplaceDisplacementField τ p a ≠ 0) :
    atomMassReal p a = 0 ↔ atomMassReal q a = 0 := by
  have hratio := laplaceZeroDrift_atomMassRatio_of_coneExtraction
    hτ p q a hzero hDp_ne
  have hZp_pos : 0 < kernelNormalizer (laplaceKernel τ) p a :=
    laplaceKernelNormalizer_pos p τ hτ a
  have hZq_pos : 0 < kernelNormalizer (laplaceKernel τ) q a :=
    laplaceKernelNormalizer_pos q τ hτ a
  constructor
  · intro hp
    have h : atomMassReal q a * kernelNormalizer (laplaceKernel τ) p a = 0 := by
      rw [hratio, hp, zero_mul]
    exact (mul_eq_zero.mp h).resolve_right hZp_pos.ne'
  · intro hq
    have h : atomMassReal p a * kernelNormalizer (laplaceKernel τ) q a = 0 := by
      rw [← hratio, hq, zero_mul]
    exact (mul_eq_zero.mp h).resolve_right hZq_pos.ne'

end WNormalized

end DriftingIdentifiability
