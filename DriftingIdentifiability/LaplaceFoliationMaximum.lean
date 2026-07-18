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

end DriftingIdentifiability
