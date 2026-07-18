import DriftingIdentifiability.LaplaceFoliationFlow

/-!
# G4: the global maximum principle for the foliation defect

This file removes the "seedless fully degenerate component" gap of G4.  The
observation is that the scalar defect `H = ψ_p − (Z_p/Z_q)·ψ_q` of a
zero-drift pair satisfies a genuine global maximum principle:

* `H ≤ ψ_p − τ²Z_p = ∫ τ‖x−y‖e^{−‖x−y‖/τ} dp`, an unconditional bound that
  **vanishes at infinity** — so any positive supremum of `H` is attained on a
  compact sublevel set;
* along the `q`-gradient flow the certified Abel equation
  `H' = −(ψ_q/τ²)·H` with `ψ_q > 0` makes a positive `H` grow strictly
  backward in time, at a rate bounded below on any ball;
* the field speed is at most `τ/e`, so a short backward orbit started near a
  maximum point cannot leave a fixed ball, forcing `H` to exceed its own
  supremum — unless the maximum point lies in `interior {D_q = 0}`, where the
  pointwise elliptic identity forces `H = 0`.

Hence `sup H ≤ 0`; swapping the roles of `p` and `q` flips the sign and gives
`H ≡ 0` outright.  No seed, no leaf factorization, no tube rigidity, and no
transnormal classification is needed.
-/

open MeasureTheory Filter Set Topology Metric
open scoped NNReal RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
  [SecondCountableTopology E] [FiniteDimensional ℝ E]

/-! ## The unconditional first-moment bound -/

/-- The kernel-weighted first moment `∫ τ‖x−y‖e^{−‖x−y‖/τ} dμ`, the exact gap
between the companion potential and `τ²` times the normalizer. -/
noncomputable def laplaceFirstMoment (τ : ℝ) (μ : Measure E) (x : E) : ℝ :=
  ∫ y, τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) ∂μ

omit [InnerProductSpace ℝ E] [CompleteSpace E] [SecondCountableTopology E]
    [FiniteDimensional ℝ E] in
lemma integrable_laplaceFirstMoment_integrand {τ : ℝ} (hτ : 0 < τ)
    (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    Integrable (fun y => τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ)) μ := by
  refine Integrable.of_bound ?_ (τ * (τ * Real.exp (-1))) ?_
  · apply Continuous.aestronglyMeasurable
    fun_prop
  · filter_upwards with y
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity), mul_assoc]
    exact mul_le_mul_of_nonneg_left
      (mul_exp_neg_div_le hτ (norm_nonneg _)) hτ.le

omit [InnerProductSpace ℝ E] [BorelSpace E] [CompleteSpace E]
    [SecondCountableTopology E] [FiniteDimensional ℝ E] in
lemma laplaceFirstMoment_nonneg {τ : ℝ} (hτ : 0 < τ)
    (μ : Measure E) (x : E) : 0 ≤ laplaceFirstMoment τ μ x :=
  integral_nonneg fun y => by positivity

omit [InnerProductSpace ℝ E] [CompleteSpace E] [SecondCountableTopology E]
    [FiniteDimensional ℝ E] in
/-- The exact potential split `ψ = τ²Z + (first moment)`. -/
theorem laplaceDisplacementPotential_eq_normalizer_add_firstMoment
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    laplaceDisplacementPotential τ μ x =
      τ ^ 2 * kernelNormalizer (laplaceKernel τ) μ x +
        laplaceFirstMoment τ μ x := by
  have hker : Integrable (fun y => τ ^ 2 * laplaceKernel τ x y) μ :=
    (laplaceKernel_integrable μ τ hτ x).const_mul _
  have hmom := integrable_laplaceFirstMoment_integrand hτ μ x
  have hsum :
      (∫ y, (τ ^ 2 * laplaceKernel τ x y +
          τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ)) ∂μ) =
        (∫ y, τ ^ 2 * laplaceKernel τ x y ∂μ) +
          ∫ y, τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) ∂μ :=
    integral_add hker hmom
  have hpt : ∀ y : E,
      matern32Profile τ ‖x - y‖ =
        τ ^ 2 * laplaceKernel τ x y +
          τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) := by
    intro y
    rw [matern32Profile, laplaceKernel_eq_exp]
    ring
  rw [laplaceDisplacementPotential]
  rw [show (fun y => matern32Profile τ ‖x - y‖) =
      fun y => τ ^ 2 * laplaceKernel τ x y +
        τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) from funext hpt]
  rw [hsum, kernelNormalizer, integral_const_mul]
  rfl

omit [InnerProductSpace ℝ E] [CompleteSpace E] [SecondCountableTopology E]
    [FiniteDimensional ℝ E] in
/-- `ψ ≥ τ²Z`; in particular the companion potential is strictly positive. -/
theorem laplaceDisplacementPotential_ge_normalizer
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] (x : E) :
    τ ^ 2 * kernelNormalizer (laplaceKernel τ) μ x ≤
      laplaceDisplacementPotential τ μ x := by
  rw [laplaceDisplacementPotential_eq_normalizer_add_firstMoment hτ μ x]
  linarith [laplaceFirstMoment_nonneg hτ μ x]

omit [InnerProductSpace ℝ E] [CompleteSpace E] [SecondCountableTopology E]
    [FiniteDimensional ℝ E] in
theorem laplaceDisplacementPotential_pos
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ] (x : E) :
    0 < laplaceDisplacementPotential τ μ x :=
  lt_of_lt_of_le
    (by
      have hZ := laplaceKernelNormalizer_pos μ τ hτ x
      positivity)
    (laplaceDisplacementPotential_ge_normalizer hτ μ x)

/-- **The unconditional defect bound.**  No zero-drift hypothesis is needed:
`R·ψ_q ≥ R·τ²Z_q = τ²Z_p` pointwise, so the defect is at most the
`p` first moment. -/
theorem laplaceFoliationDefect_le_firstMoment
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (x : E) :
    laplaceFoliationDefect τ p q x ≤ laplaceFirstMoment τ p x := by
  have hR : 0 < laplaceNormalizerRatio τ p q x :=
    laplaceNormalizerRatio_pos hτ p q x
  have hψq := laplaceDisplacementPotential_ge_normalizer hτ q x
  have hZq : 0 < kernelNormalizer (laplaceKernel τ) q x :=
    laplaceKernelNormalizer_pos q τ hτ x
  have hkey : τ ^ 2 * kernelNormalizer (laplaceKernel τ) p x ≤
      laplaceNormalizerRatio τ p q x * laplaceDisplacementPotential τ q x := by
    have h1 : laplaceNormalizerRatio τ p q x *
        (τ ^ 2 * kernelNormalizer (laplaceKernel τ) q x) ≤
        laplaceNormalizerRatio τ p q x * laplaceDisplacementPotential τ q x :=
      mul_le_mul_of_nonneg_left hψq hR.le
    have h2 : laplaceNormalizerRatio τ p q x *
        (τ ^ 2 * kernelNormalizer (laplaceKernel τ) q x) =
        τ ^ 2 * kernelNormalizer (laplaceKernel τ) p x := by
      unfold laplaceNormalizerRatio
      field_simp
    linarith
  have hsplit :=
    laplaceDisplacementPotential_eq_normalizer_add_firstMoment hτ p x
  unfold laplaceFoliationDefect
  linarith

/-! ## Vanishing of the first moment at infinity -/

omit [InnerProductSpace ℝ E] [CompleteSpace E] [SecondCountableTopology E]
    [FiniteDimensional ℝ E] in
/-- The real measure of complements of balls tends to zero. -/
lemma tendsto_measureReal_compl_closedBall
    (μ : Measure E) [IsProbabilityMeasure μ] :
    Tendsto (fun k : ℕ => μ.real (Metric.closedBall (0 : E) k)ᶜ)
      atTop (nhds 0) := by
  have hanti : Antitone (fun k : ℕ => (Metric.closedBall (0 : E) k)ᶜ) := by
    intro a b hab
    exact compl_subset_compl.mpr
      (Metric.closedBall_subset_closedBall (by exact_mod_cast hab))
  have hinter : (⋂ k : ℕ, (Metric.closedBall (0 : E) k)ᶜ) = ∅ := by
    ext y
    simp only [mem_iInter, mem_compl_iff, Metric.mem_closedBall,
      dist_zero_right, mem_empty_iff_false, iff_false, not_forall, not_not]
    exact ⟨⌈‖y‖⌉₊, Nat.le_ceil _⟩
  have hlim := tendsto_measure_iInter_atTop (μ := μ)
    (fun k => (measurableSet_closedBall.compl).nullMeasurableSet)
    hanti ⟨0, measure_ne_top μ _⟩
  rw [hinter, measure_empty] at hlim
  have hreal := (ENNReal.tendsto_toReal (by simp)).comp hlim
  simpa [measureReal_def, Function.comp_def] using hreal

omit [NormedAddCommGroup E] [InnerProductSpace ℝ E] [BorelSpace E]
    [CompleteSpace E] [SecondCountableTopology E] [FiniteDimensional ℝ E] in
private lemma prob_measureReal_le_one_max
    {μ : Measure E} [IsProbabilityMeasure μ] (s : Set E) : μ.real s ≤ 1 :=
  ENNReal.toReal_mono ENNReal.one_ne_top prob_le_one

omit [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]
    [FiniteDimensional ℝ E] in
/-- Half-bandwidth pointwise bound: `τ·r·e^{−r/τ} ≤ A·e^{−r/(2τ)}` with
`A = 2τ²/e`. -/
private lemma firstMoment_integrand_le_halfExp {τ : ℝ} (hτ : 0 < τ)
    {r : ℝ} (hr : 0 ≤ r) :
    τ * r * Real.exp (-r / τ) ≤
      τ * (2 * τ * Real.exp (-1)) * Real.exp (-r / (2 * τ)) := by
  have hsplit : Real.exp (-r / τ) =
      Real.exp (-r / (2 * τ)) * Real.exp (-r / (2 * τ)) := by
    rw [← Real.exp_add]
    congr 1
    field_simp
    ring
  have hhalf : r * Real.exp (-r / (2 * τ)) ≤ 2 * τ * Real.exp (-1) :=
    mul_exp_neg_div_le (by positivity) hr
  have hexp2 : (0 : ℝ) < Real.exp (-r / (2 * τ)) := Real.exp_pos _
  calc τ * r * Real.exp (-r / τ)
      = τ * ((r * Real.exp (-r / (2 * τ))) * Real.exp (-r / (2 * τ))) := by
        rw [hsplit]; ring
    _ ≤ τ * ((2 * τ * Real.exp (-1)) * Real.exp (-r / (2 * τ))) := by
        have := mul_le_mul_of_nonneg_right hhalf hexp2.le
        exact mul_le_mul_of_nonneg_left this hτ.le
    _ = τ * (2 * τ * Real.exp (-1)) * Real.exp (-r / (2 * τ)) := by ring

omit [InnerProductSpace ℝ E] [CompleteSpace E] [SecondCountableTopology E]
    [FiniteDimensional ℝ E] in
/-- **First-moment decay.**  Beyond a sufficiently large radius the
kernel-weighted first moment is uniformly small. -/
theorem laplaceFirstMoment_small_far
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ]
    {ε : ℝ} (hε : 0 < ε) :
    ∃ Rad : ℝ, ∀ x : E, Rad ≤ ‖x‖ → laplaceFirstMoment τ μ x < ε := by
  set C : ℝ := τ * (τ * Real.exp (-1)) with hCdef
  have hCpos : 0 < C := by rw [hCdef]; positivity
  set A : ℝ := τ * (2 * τ * Real.exp (-1)) with hAdef
  have hApos : 0 < A := by rw [hAdef]; positivity
  -- tail choice: a core radius k capturing all but ε/(2(C+1)) of the mass
  have hεtail : 0 < ε / (2 * (C + 1)) := by positivity
  obtain ⟨k, hk⟩ :=
    ((tendsto_measureReal_compl_closedBall μ).eventually
      (gt_mem_nhds hεtail)).exists
  -- core-decay choice: beyond Rad the core contribution is below ε/2
  have hcore : Tendsto
      (fun R : ℝ => A * Real.exp (-((R - k) / (2 * τ)))) atTop (nhds 0) := by
    have h1 : Tendsto (fun R : ℝ => (R - k) / (2 * τ)) atTop atTop := by
      apply Tendsto.atTop_div_const (by positivity : (0:ℝ) < 2 * τ)
      simpa [sub_eq_add_neg] using
        tendsto_atTop_add_const_right atTop (-(k : ℝ)) tendsto_id
    have h2 : Tendsto (fun R : ℝ => Real.exp (-((R - k) / (2 * τ))))
        atTop (nhds 0) := by
      have := Real.tendsto_exp_atBot.comp (tendsto_neg_atTop_atBot.comp h1)
      simpa [Function.comp_def] using this
    simpa using h2.const_mul A
  obtain ⟨Rad, hRad⟩ := Filter.eventually_atTop.mp
    (hcore.eventually (gt_mem_nhds (half_pos hε)))
  refine ⟨Rad, fun x hx => ?_⟩
  have hint := integrable_laplaceFirstMoment_integrand hτ μ x
  -- split the integral at the core ball
  have hsplit :
      laplaceFirstMoment τ μ x =
        (∫ y in Metric.closedBall (0 : E) k,
            τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) ∂μ) +
          ∫ y in (Metric.closedBall (0 : E) k)ᶜ,
            τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) ∂μ := by
    rw [laplaceFirstMoment,
      ← integral_add_compl measurableSet_closedBall hint]
  -- core estimate
  have hcoreBound :
      (∫ y in Metric.closedBall (0 : E) k,
          τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) ∂μ) ≤
        A * Real.exp (-((‖x‖ - k) / (2 * τ))) := by
    have hptcore : ∀ y ∈ Metric.closedBall (0 : E) k,
        τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) ≤
          A * Real.exp (-((‖x‖ - k) / (2 * τ))) := by
      intro y hy
      have hyk : ‖y‖ ≤ k := by
        simpa [dist_zero_right] using Metric.mem_closedBall.mp hy
      have hdist : ‖x‖ - k ≤ ‖x - y‖ := by
        have := norm_sub_norm_le x y
        linarith
      have hmono : Real.exp (-‖x - y‖ / (2 * τ)) ≤
          Real.exp (-((‖x‖ - k) / (2 * τ))) := by
        apply Real.exp_le_exp.mpr
        rw [neg_div]
        apply neg_le_neg
        gcongr
      calc τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ)
          ≤ A * Real.exp (-‖x - y‖ / (2 * τ)) := by
            rw [hAdef]
            exact firstMoment_integrand_le_halfExp hτ (norm_nonneg _)
        _ ≤ A * Real.exp (-((‖x‖ - k) / (2 * τ))) :=
            mul_le_mul_of_nonneg_left hmono hApos.le
    calc (∫ y in Metric.closedBall (0 : E) k,
            τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) ∂μ)
        ≤ ∫ _y in Metric.closedBall (0 : E) k,
            A * Real.exp (-((‖x‖ - k) / (2 * τ))) ∂μ :=
          setIntegral_mono_on hint.integrableOn
            ((integrable_const _).integrableOn)
            measurableSet_closedBall hptcore
      _ = μ.real (Metric.closedBall (0 : E) k) *
            (A * Real.exp (-((‖x‖ - k) / (2 * τ)))) := by
          rw [setIntegral_const, smul_eq_mul]
      _ ≤ 1 * (A * Real.exp (-((‖x‖ - k) / (2 * τ)))) :=
          mul_le_mul_of_nonneg_right (prob_measureReal_le_one_max _)
            (by positivity)
      _ = A * Real.exp (-((‖x‖ - k) / (2 * τ))) := one_mul _
  -- tail estimate
  have htailBound :
      (∫ y in (Metric.closedBall (0 : E) k)ᶜ,
          τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) ∂μ) ≤
        μ.real (Metric.closedBall (0 : E) k)ᶜ * C := by
    have hpt : ∀ y ∈ (Metric.closedBall (0 : E) k)ᶜ,
        τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) ≤ C := by
      intro y _
      rw [hCdef, mul_assoc]
      exact mul_le_mul_of_nonneg_left
        (mul_exp_neg_div_le hτ (norm_nonneg _)) hτ.le
    calc (∫ y in (Metric.closedBall (0 : E) k)ᶜ,
            τ * ‖x - y‖ * Real.exp (-‖x - y‖ / τ) ∂μ)
        ≤ ∫ _y in (Metric.closedBall (0 : E) k)ᶜ, C ∂μ :=
          setIntegral_mono_on hint.integrableOn
            ((integrable_const _).integrableOn)
            measurableSet_closedBall.compl hpt
      _ = μ.real (Metric.closedBall (0 : E) k)ᶜ * C := by
          rw [setIntegral_const, smul_eq_mul]
  -- combine
  have hcoreSmall : A * Real.exp (-((‖x‖ - k) / (2 * τ))) < ε / 2 :=
    hRad ‖x‖ hx
  have htailSmall : μ.real (Metric.closedBall (0 : E) k)ᶜ * C < ε / 2 := by
    have h1 : μ.real (Metric.closedBall (0 : E) k)ᶜ * C <
        (ε / (2 * (C + 1))) * C :=
      mul_lt_mul_of_pos_right hk hCpos
    have h2 : (ε / (2 * (C + 1))) * C ≤ ε / 2 := by
      rw [div_mul_eq_mul_div, div_le_div_iff₀ (by positivity) (by norm_num)]
      nlinarith
    linarith
  rw [hsplit]
  linarith

/-! ## Attained positive maximum -/

/-- If the defect is positive somewhere, it attains a positive global
maximum: positive sublevel sets are compact by the first-moment decay. -/
theorem laplaceFoliationDefect_exists_max_of_pos
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    {x₀ : E} (h₀ : 0 < laplaceFoliationDefect τ p q x₀) :
    ∃ z : E, 0 < laplaceFoliationDefect τ p q z ∧
      ∀ y : E, laplaceFoliationDefect τ p q y ≤
        laplaceFoliationDefect τ p q z := by
  obtain ⟨Rad, hRad⟩ := laplaceFirstMoment_small_far hτ p h₀
  set K : Set E := {y : E | laplaceFoliationDefect τ p q x₀ ≤
    laplaceFoliationDefect τ p q y} with hKdef
  have hHcont := continuous_laplaceFoliationDefect hτ p q
  have hKclosed : IsClosed K := isClosed_le continuous_const hHcont
  have hKsub : K ⊆ Metric.ball (0 : E) (max Rad 1) := by
    intro y hy
    rw [Metric.mem_ball, dist_zero_right]
    by_contra hcon
    rw [not_lt] at hcon
    have hyRad : Rad ≤ ‖y‖ := le_trans (le_max_left _ _) hcon
    have hlt := hRad y hyRad
    have hle := laplaceFoliationDefect_le_firstMoment hτ p q y
    have hyK : laplaceFoliationDefect τ p q x₀ ≤
        laplaceFoliationDefect τ p q y := hy
    linarith
  have hKcompact : IsCompact K :=
    Metric.isCompact_of_isClosed_isBounded hKclosed
      (Metric.isBounded_ball.subset hKsub)
  have hx₀K : x₀ ∈ K := by
    rw [hKdef]
    simp only [Set.mem_setOf_eq]
    exact le_rfl
  have hKne : K.Nonempty := ⟨x₀, hx₀K⟩
  obtain ⟨z, hzK, hzmax⟩ :=
    hKcompact.exists_isMaxOn hKne hHcont.continuousOn
  have hzK' : laplaceFoliationDefect τ p q x₀ ≤
      laplaceFoliationDefect τ p q z := hzK
  refine ⟨z, lt_of_lt_of_le h₀ hzK', fun y => ?_⟩
  by_cases hy : laplaceFoliationDefect τ p q x₀ ≤ laplaceFoliationDefect τ p q y
  · have hyK : y ∈ K := hy
    exact hzmax hyK
  · exact le_trans (not_le.mp hy).le hzK'

/-! ## Backward gradient curves -/

/-- The uniform Picard–Lindelöf time step: it depends only on the bandwidth. -/
noncomputable def laplaceCurveTime (τ : ℝ) : ℝ :=
  1 / (2 * (τ * Real.exp (-1) + 1))

lemma laplaceCurveTime_pos {τ : ℝ} (hτ : 0 < τ) : 0 < laplaceCurveTime τ := by
  unfold laplaceCurveTime
  positivity

/-- Uniform-time local integral curves through every point. -/
theorem exists_gradientCurve_uniform
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ] (x : E) :
    ∃ γ : ℝ → E, γ 0 = x ∧
      ∀ t ∈ Ioo (-laplaceCurveTime τ) (laplaceCurveTime τ),
        HasDerivAt γ (laplaceDisplacementField τ μ (γ t)) t := by
  let L : ℝ≥0 := ⟨τ * Real.exp (-1),
    mul_nonneg (le_of_lt hτ) (le_of_lt (Real.exp_pos _))⟩
  have hε : 0 < laplaceCurveTime τ := laplaceCurveTime_pos hτ
  have hεeq : laplaceCurveTime τ = 1 / (2 * ((L : ℝ) + 1)) := rfl
  let t₀ : Icc (-laplaceCurveTime τ) (laplaceCurveTime τ) :=
    ⟨0, by constructor <;> linarith⟩
  have hPL : IsPicardLindelof
      (fun _ : ℝ => laplaceDisplacementField τ μ) t₀ x 1 0 L 2 := by
    refine {
      lipschitzOnWith := ?_
      continuousOn := ?_
      norm_le := ?_
      mul_max_le := ?_ }
    · intro t _
      exact (lipschitzWith_laplaceDisplacementField hτ μ).lipschitzOnWith
    · intro y _
      exact continuous_const.continuousOn
    · intro t _ y _
      exact norm_laplaceDisplacementField_le hτ μ y
    · change (L : ℝ) * max (laplaceCurveTime τ - 0)
        (0 - -laplaceCurveTime τ) ≤ (1 : ℝ) - 0
      rw [sub_zero, zero_sub, neg_neg, max_self, sub_zero, hεeq]
      have hden : 0 < 2 * ((L : ℝ) + 1) := by positivity
      rw [one_div, ← div_eq_mul_inv, div_le_iff₀ hden]
      nlinarith [L.coe_nonneg]
  rcases hPL.exists_eq_forall_mem_Icc_hasDerivWithinAt₀ with ⟨γ, hγ₀, hγ⟩
  refine ⟨γ, ?_, ?_⟩
  · simpa [t₀] using hγ₀
  · intro t ht
    exact (hγ t (Ioo_subset_Icc_self ht)).hasDerivAt
      (Icc_mem_nhds ht.1 ht.2)

/-- A gradient curve that starts regular stays regular backward in time:
Grönwall applied to `D∘γ` with the uniform Hessian bound `2`. -/
theorem laplaceDisplacementField_ne_zero_on_backward_curve
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ]
    {γ : ℝ → E} {T : ℝ} (_hT : 0 ≤ T)
    (hγ : ∀ u ∈ Icc (-T) 0, HasDerivAt γ (laplaceDisplacementField τ μ (γ u)) u)
    (h0 : laplaceDisplacementField τ μ (γ 0) ≠ 0) :
    ∀ u ∈ Icc (-T) 0, laplaceDisplacementField τ μ (γ u) ≠ 0 := by
  intro t ht hcon
  set w : ℝ → E := fun u => laplaceDisplacementField τ μ (γ u) with hw
  have hsub : Icc t 0 ⊆ Icc (-T) 0 := Icc_subset_Icc ht.1 le_rfl
  have hwderiv : ∀ u ∈ Ico t 0, HasDerivWithinAt w
      (laplaceDisplacementHessian τ μ (γ u)
        (laplaceDisplacementField τ μ (γ u))) (Ici u) u := by
    intro u hu
    exact ((hasFDerivAt_laplaceDisplacementField hτ μ (γ u)).comp_hasDerivAt u
      (hγ u (hsub (Ico_subset_Icc_self hu)))).hasDerivWithinAt
  have hwcont : ContinuousOn w (Icc t 0) := by
    intro u hu
    exact (((hasFDerivAt_laplaceDisplacementField hτ μ
      (γ u)).continuousAt).comp
        (hγ u (hsub hu)).continuousAt).continuousWithinAt
  have hbound : ∀ u ∈ Ico t 0,
      ‖laplaceDisplacementHessian τ μ (γ u)
        (laplaceDisplacementField τ μ (γ u))‖ ≤ 2 * ‖w u‖ + 0 := by
    intro u _
    rw [add_zero]
    calc ‖laplaceDisplacementHessian τ μ (γ u)
          (laplaceDisplacementField τ μ (γ u))‖
        ≤ ‖laplaceDisplacementHessian τ μ (γ u)‖ *
            ‖laplaceDisplacementField τ μ (γ u)‖ :=
          ContinuousLinearMap.le_opNorm _ _
      _ ≤ 2 * ‖w u‖ :=
          mul_le_mul_of_nonneg_right
            (norm_laplaceDisplacementHessian_le hτ μ (γ u)) (norm_nonneg _)
  have hstart : ‖w t‖ ≤ 0 := by
    rw [hw]
    simp [hcon]
  have hgron := norm_le_gronwallBound_of_norm_deriv_right_le
    hwcont hwderiv hstart hbound 0 ⟨ht.2, le_rfl⟩
  rw [gronwallBound_ε0] at hgron
  have hzero0 : w 0 = 0 := by
    have : ‖w 0‖ ≤ 0 := by
      calc ‖w 0‖ ≤ 0 * Real.exp (2 * (0 - t)) := hgron
        _ = 0 := zero_mul _
    exact norm_le_zero_iff.mp this
  exact h0 (by simpa [hw] using hzero0)

omit [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]
    [FiniteDimensional ℝ E] in
/-- Backward speed bound: the curve moves at most `τ/e` per unit time. -/
theorem norm_gradientCurve_sub_start_le
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsProbabilityMeasure μ]
    {γ : ℝ → E} {T : ℝ} (hT : 0 ≤ T)
    (hγ : ∀ u ∈ Icc (-T) 0, HasDerivAt γ (laplaceDisplacementField τ μ (γ u)) u) :
    ∀ u ∈ Icc (-T) 0, ‖γ u - γ 0‖ ≤ τ * Real.exp (-1) * T := by
  intro u hu
  have hmvt := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (f := γ) (f' := fun z => laplaceDisplacementField τ μ (γ z))
    (C := τ * Real.exp (-1))
    (fun z hz => (hγ z hz).hasDerivWithinAt)
    (fun z _ => norm_laplaceDisplacementField_le hτ μ (γ z))
    (convex_Icc _ _) (right_mem_Icc.mpr (by linarith)) hu
  calc ‖γ u - γ 0‖ ≤ τ * Real.exp (-1) * ‖u - 0‖ := hmvt
    _ ≤ τ * Real.exp (-1) * T := by
        have habs : ‖u - 0‖ ≤ T := by
          rw [sub_zero, Real.norm_eq_abs, abs_le]
          exact ⟨by linarith [hu.1], by linarith [hu.2]⟩
        exact mul_le_mul_of_nonneg_left habs (by positivity)

omit [FiniteDimensional ℝ E] in
/-- The companion potential is continuous. -/
theorem continuous_laplaceDisplacementPotential
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] :
    Continuous (laplaceDisplacementPotential τ μ) := by
  rw [continuous_iff_continuousAt]
  exact fun x => (hasFDerivAt_laplaceDisplacementPotential hτ μ x).continuousAt

/-- **Backward exponential growth of a nonnegative defect.**  Along a regular
backward orbit segment on which `ψ_q ≥ ρτ²`, the defect grows by at least
`e^{ρT}`. -/
theorem laplaceFoliationDefect_backward_growth
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {γ : ℝ → E} {T S : ℝ} (hT : 0 ≤ T) (hTS : T < S)
    (hγ : ∀ u ∈ Ioo (-S) S, HasDerivAt γ
      (laplaceDisplacementField τ q (γ u)) u)
    (hreg : ∀ u ∈ Icc (-T) 0, laplaceDisplacementField τ q (γ u) ≠ 0)
    {ρ : ℝ}
    (hρ : ∀ u ∈ Icc (-T) 0,
      ρ * τ ^ 2 ≤ laplaceDisplacementPotential τ q (γ u))
    (hH0 : 0 ≤ laplaceFoliationDefect τ p q (γ 0)) :
    laplaceFoliationDefect τ p q (γ 0) * Real.exp (ρ * T) ≤
      laplaceFoliationDefect τ p q (γ (-T)) := by
  have hS0 : 0 < S := lt_of_le_of_lt hT hTS
  have hsubIoo : Icc (-T) 0 ⊆ Ioo (-S) S := by
    intro u hu
    exact ⟨by linarith [hu.1], by linarith [hu.2]⟩
  set W : ℝ → ℝ := fun u => laplaceFoliationDefect τ p q (γ u) with hWdef
  set c : ℝ → ℝ := fun u =>
    laplaceDisplacementPotential τ q (γ u) / τ ^ 2 with hcdef
  have hγcont : ContinuousOn γ (Ioo (-S) S) := fun u hu =>
    (hγ u hu).continuousAt.continuousWithinAt
  have hccont : ContinuousOn c (Ioo (-S) S) :=
    (((continuous_laplaceDisplacementPotential hτ q).comp_continuousOn
      hγcont).div_const _)
  set A : ℝ → ℝ := fun z => ∫ u in (-T)..z, c u with hAdef
  have hAfull : ∀ z ∈ Ioo (-S) S, HasDerivAt A (c z) z := by
    intro z hz
    have hmemT : (-T : ℝ) ∈ Ioo (-S) S := ⟨by linarith, by linarith⟩
    have hcint : IntervalIntegrable c volume (-T) z :=
      ContinuousOn.intervalIntegrable
        (hccont.mono (OrdConnected.uIcc_subset inferInstance hmemT hz))
    exact intervalIntegral.integral_hasDerivAt_right hcint
      (hccont.stronglyMeasurableAtFilter isOpen_Ioo z hz)
      (hccont.continuousAt (isOpen_Ioo.mem_nhds hz))
  have hWcont : ContinuousOn W (Icc (-T) 0) := fun u hu =>
    (((continuous_laplaceFoliationDefect hτ p q).continuousAt).comp
      (hγ u (hsubIoo hu)).continuousAt).continuousWithinAt
  have hAcont : ContinuousOn A (Icc (-T) 0) := fun u hu =>
    (hAfull u (hsubIoo hu)).continuousAt.continuousWithinAt
  have hcont : ContinuousOn (fun x => W x * Real.exp (A x)) (Icc (-T) 0) :=
    hWcont.mul (Real.continuous_exp.comp_continuousOn hAcont)
  have hW : ∀ x ∈ Ico (-T) 0,
      HasDerivWithinAt W (-(c x) * W x) (Ici x) x := by
    intro x hx
    have habel := hasDerivAt_laplaceFoliationDefect_comp_gradientCurve_abel
      hτ p q hzero (hγ x (hsubIoo (Ico_subset_Icc_self hx)))
      (hreg x (Ico_subset_Icc_self hx))
    exact habel.hasDerivWithinAt
  have hA : ∀ x ∈ Ico (-T) 0, HasDerivWithinAt A (c x) (Ici x) x :=
    fun x hx =>
      (hAfull x (hsubIoo (Ico_subset_Icc_self hx))).hasDerivWithinAt
  have hconst := abel_integratingFactor_const_Icc hcont hW hA
  have hend := hconst 0 ⟨by linarith, le_rfl⟩
  have hAT : A (-T) = 0 := intervalIntegral.integral_same
  rw [hAT, Real.exp_zero, mul_one] at hend
  have hA0 : ρ * T ≤ A 0 := by
    have hρ' : ∀ u ∈ Icc (-T) 0, ρ ≤ c u := by
      intro u hu
      rw [hcdef]
      rw [le_div_iff₀ (by positivity : (0:ℝ) < τ ^ 2)]
      exact hρ u hu
    have hcint : IntervalIntegrable c volume (-T) 0 := by
      have hmemT : (-T : ℝ) ∈ Ioo (-S) S := ⟨by linarith, by linarith⟩
      have hmem0 : (0 : ℝ) ∈ Ioo (-S) S := ⟨by linarith, hS0⟩
      exact ContinuousOn.intervalIntegrable
        (hccont.mono (OrdConnected.uIcc_subset inferInstance hmemT hmem0))
    have hmono := intervalIntegral.integral_mono_on (by linarith : (-T:ℝ) ≤ 0)
      (intervalIntegrable_const) hcint hρ'
    rw [intervalIntegral.integral_const, smul_eq_mul, sub_neg_eq_add,
      zero_add, mul_comm] at hmono
    exact hmono
  calc laplaceFoliationDefect τ p q (γ 0) * Real.exp (ρ * T)
      ≤ laplaceFoliationDefect τ p q (γ 0) * Real.exp (A 0) :=
        mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hA0) hH0
    _ = laplaceFoliationDefect τ p q (γ (-T)) := hend

/-! ## The maximum principle -/

/-- On the interior of the critical set the pointwise elliptic identity forces
the defect to vanish. -/
theorem laplaceFoliationDefect_eq_zero_of_mem_interior_qField_eq_zero
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    {x : E} (hx : x ∈ interior {y | laplaceDisplacementField τ q y = 0}) :
    laplaceFoliationDefect τ p q x = 0 := by
  have hUopen : IsOpen (interior {y | laplaceDisplacementField τ q y = 0}) :=
    isOpen_interior
  have hqcrit : ∀ y ∈ interior {y | laplaceDisplacementField τ q y = 0},
      laplaceDisplacementField τ q y = 0 := by
    intro y hy
    have hy' : y ∈ {y : E | laplaceDisplacementField τ q y = 0} :=
      interior_subset hy
    exact hy'
  have hpcrit : ∀ y ∈ interior {y | laplaceDisplacementField τ q y = 0},
      laplaceDisplacementField τ p y = 0 := fun y hy =>
    laplaceDisplacementField_eq_zero_of_zeroDrift_of_eq_zero
      hτ p q hzero (hqcrit y hy)
  have hpLap := laplaceDisplacementLaplacian_eq_zero_of_eq_zero_on_open
    hτ p hUopen hpcrit hx
  have hqLap := laplaceDisplacementLaplacian_eq_zero_of_eq_zero_on_open
    hτ q hUopen hqcrit hx
  have hpde := laplaceDisplacementPotential_elliptic hτ p x
  have hqde := laplaceDisplacementPotential_elliptic hτ q x
  rw [hpLap, mul_zero, sub_zero] at hpde
  rw [hqLap, mul_zero, sub_zero] at hqde
  have hZq : kernelNormalizer (laplaceKernel τ) q x ≠ 0 :=
    (laplaceKernelNormalizer_pos q τ hτ x).ne'
  unfold laplaceFoliationDefect laplaceNormalizerRatio
  rw [hpde, hqde]
  field_simp
  ring

/-- **The global maximum principle.**  Under zero drift the foliation defect
is nonpositive everywhere: a positive maximum could neither sit at a regular
point nor at a critical boundary point (backward flow inflates it beyond its
own supremum without leaving a good ball), nor inside the critical set
(elliptic identity). -/
theorem laplaceFoliationDefect_nonpos
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : E) :
    laplaceFoliationDefect τ p q x ≤ 0 := by
  by_contra hcon
  rw [not_le] at hcon
  obtain ⟨z, hz, hmax⟩ := laplaceFoliationDefect_exists_max_of_pos hτ p q hcon
  by_cases hint : z ∈ interior {y | laplaceDisplacementField τ q y = 0}
  · exact absurd
      (laplaceFoliationDefect_eq_zero_of_mem_interior_qField_eq_zero
        hτ p q hzero hint) hz.ne'
  -- z is a closure point of the regular set
  have hclos : ∀ η : ℝ, 0 < η →
      ∃ y : E, dist y z < η ∧ laplaceDisplacementField τ q y ≠ 0 := by
    intro η hη
    by_contra hno
    push Not at hno
    apply hint
    rw [mem_interior_iff_mem_nhds]
    filter_upwards [Metric.ball_mem_nhds z hη] with y hy
    exact hno y (Metric.mem_ball.mp hy)
  -- the good ball: `ψ_q ≥ δτ²` with `δτ² = ψ_q(z)/2`
  have hψz : 0 < laplaceDisplacementPotential τ q z :=
    laplaceDisplacementPotential_pos hτ q z
  have hδpos : 0 < laplaceDisplacementPotential τ q z / (2 * τ ^ 2) := by
    positivity
  set δ : ℝ := laplaceDisplacementPotential τ q z / (2 * τ ^ 2) with hδdef
  have hδτ : δ * τ ^ 2 = laplaceDisplacementPotential τ q z / 2 := by
    rw [hδdef]
    field_simp
  obtain ⟨R₀, hR₀pos, hR₀⟩ : ∃ R₀ : ℝ, 0 < R₀ ∧ ∀ y : E, dist y z < R₀ →
      δ * τ ^ 2 ≤ laplaceDisplacementPotential τ q y := by
    have hhalf : laplaceDisplacementPotential τ q z / 2 <
        laplaceDisplacementPotential τ q z := half_lt_self hψz
    have hev := ((continuous_laplaceDisplacementPotential hτ
      q).continuousAt (x := z)).eventually (eventually_gt_nhds hhalf)
    rw [Metric.eventually_nhds_iff] at hev
    obtain ⟨R₀, hR₀pos, hball⟩ := hev
    refine ⟨R₀, hR₀pos, fun y hy => ?_⟩
    rw [hδτ]
    exact (hball hy).le
  -- the backward time: short enough for the uniform curve and to stay in the ball
  have hLpos : 0 < τ * Real.exp (-1) + 1 := by positivity
  set T : ℝ := min (laplaceCurveTime τ / 2)
    (R₀ / (2 * (τ * Real.exp (-1) + 1))) with hTdef
  have hTpos : 0 < T := lt_min (half_pos (laplaceCurveTime_pos hτ))
    (by positivity)
  have hTS : T < laplaceCurveTime τ :=
    lt_of_le_of_lt (min_le_left _ _) (half_lt_self (laplaceCurveTime_pos hτ))
  have hTspeed : τ * Real.exp (-1) * T ≤ R₀ / 2 := by
    have h1 : T ≤ R₀ / (2 * (τ * Real.exp (-1) + 1)) := min_le_right _ _
    have h2 : τ * Real.exp (-1) * T ≤
        τ * Real.exp (-1) * (R₀ / (2 * (τ * Real.exp (-1) + 1))) :=
      mul_le_mul_of_nonneg_left h1 (by positivity)
    have h3 : τ * Real.exp (-1) * (R₀ / (2 * (τ * Real.exp (-1) + 1))) ≤
        R₀ / 2 := by
      rw [← mul_div_assoc, div_le_div_iff₀ (by positivity) (by norm_num)]
      nlinarith [hR₀pos.le, mul_nonneg hτ.le (Real.exp_pos (-1)).le]
    linarith
  -- the seed threshold
  have hexpT : (1 : ℝ) < Real.exp (δ * T) := by
    rw [← Real.exp_zero]
    exact Real.exp_lt_exp.mpr (by positivity)
  have hthr : laplaceFoliationDefect τ p q z / Real.exp (δ * T) <
      laplaceFoliationDefect τ p q z := by
    rw [div_lt_iff₀ (Real.exp_pos _)]
    nlinarith
  obtain ⟨η, hηpos, hη⟩ : ∃ η : ℝ, 0 < η ∧ ∀ y : E, dist y z < η →
      laplaceFoliationDefect τ p q z / Real.exp (δ * T) <
        laplaceFoliationDefect τ p q y := by
    have hev := ((continuous_laplaceFoliationDefect hτ
      p q).continuousAt (x := z)).eventually (eventually_gt_nhds hthr)
    rw [Metric.eventually_nhds_iff] at hev
    obtain ⟨η, hηpos, hball⟩ := hev
    exact ⟨η, hηpos, fun y hy => hball hy⟩
  -- pick a regular seed close to the maximum point
  obtain ⟨x', hx'dist, hx'reg⟩ := hclos (min η (R₀ / 2))
    (lt_min hηpos (by positivity))
  have hx'η : dist x' z < η := lt_of_lt_of_le hx'dist (min_le_left _ _)
  have hx'R : dist x' z < R₀ / 2 := lt_of_lt_of_le hx'dist (min_le_right _ _)
  -- the backward curve from the seed
  obtain ⟨γ, hγ0, hγ⟩ := exists_gradientCurve_uniform hτ q x'
  have hsubT : Icc (-T) 0 ⊆ Ioo (-laplaceCurveTime τ) (laplaceCurveTime τ) := by
    intro u hu
    constructor
    · linarith [hu.1]
    · exact lt_of_le_of_lt hu.2 (laplaceCurveTime_pos hτ)
  have hγIcc : ∀ u ∈ Icc (-T) 0, HasDerivAt γ
      (laplaceDisplacementField τ q (γ u)) u :=
    fun u hu => hγ u (hsubT hu)
  have hreg := laplaceDisplacementField_ne_zero_on_backward_curve hτ q
    hTpos.le hγIcc (by rw [hγ0]; exact hx'reg)
  -- the curve stays in the good ball
  have hstay : ∀ u ∈ Icc (-T) 0, dist (γ u) z < R₀ := by
    intro u hu
    have hspeed := norm_gradientCurve_sub_start_le hτ q hTpos.le hγIcc u hu
    calc dist (γ u) z ≤ dist (γ u) (γ 0) + dist (γ 0) z := dist_triangle _ _ _
      _ = ‖γ u - γ 0‖ + dist x' z := by rw [dist_eq_norm, hγ0]
      _ < τ * Real.exp (-1) * T + R₀ / 2 := by
          have := hspeed
          linarith
      _ ≤ R₀ / 2 + R₀ / 2 := by linarith
      _ = R₀ := by ring
  have hρ : ∀ u ∈ Icc (-T) 0,
      δ * τ ^ 2 ≤ laplaceDisplacementPotential τ q (γ u) :=
    fun u hu => hR₀ _ (hstay u hu)
  -- the seed value exceeds the threshold
  have hHx' : laplaceFoliationDefect τ p q z / Real.exp (δ * T) <
      laplaceFoliationDefect τ p q (γ 0) := by
    rw [hγ0]
    exact hη x' hx'η
  have hthrpos : 0 < laplaceFoliationDefect τ p q z / Real.exp (δ * T) := by
    positivity
  -- backward growth pushes the defect above its supremum
  have hgrowth := laplaceFoliationDefect_backward_growth hτ p q hzero
    hTpos.le hTS hγ hreg hρ (le_of_lt (lt_trans hthrpos hHx'))
  have hover : laplaceFoliationDefect τ p q z <
      laplaceFoliationDefect τ p q (γ (-T)) := by
    calc laplaceFoliationDefect τ p q z
        = (laplaceFoliationDefect τ p q z / Real.exp (δ * T)) *
            Real.exp (δ * T) := by
          field_simp
      _ < laplaceFoliationDefect τ p q (γ 0) * Real.exp (δ * T) :=
          mul_lt_mul_of_pos_right hHx' (Real.exp_pos _)
      _ ≤ laplaceFoliationDefect τ p q (γ (-T)) := hgrowth
  linarith [hmax (γ (-T))]

end DriftingIdentifiability
