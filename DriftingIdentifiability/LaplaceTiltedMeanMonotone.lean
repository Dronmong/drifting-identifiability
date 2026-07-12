import DriftingIdentifiability.LaplaceGeneralConverse

/-!
# Monotonicity of the Laplace tilted mean (linchpin lemma L3/L7)

This module tests the **no-axiom route** for the a.c. Laplace converse
(`LaplaceACDerivation.md`).  The decisive lemma is that the tilted mean

`μ_p(x) = (∫ y·k(x,y) dp) / (∫ k(x,y) dp)`,  `k(x,y) = exp(-|x-y|/τ)`,

is monotone in `x`.  The earlier worry was that this "correlation" fact would
need an external Chebyshev/FKG correlation-inequality axiom.  It does not: the
proof is elementary — the **Monge / total-positivity (TP2)** property of the
Laplace kernel (a pure inequality about `|·|`) plus a symmetrization of the
double integral.  Everything here is axiom-free.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-- **Monge / quadrangle inequality** for `|·|` on `ℝ`.  For `x₁ ≤ x₂` and
`y₁ ≤ y₂` the "aligned" pairing has the smaller total distance:
`|x₁-y₁| + |x₂-y₂| ≤ |x₁-y₂| + |x₂-y₁|`. -/
lemma abs_monge {x₁ x₂ y₁ y₂ : ℝ} (hx : x₁ ≤ x₂) (hy : y₁ ≤ y₂) :
    |x₁ - y₁| + |x₂ - y₂| ≤ |x₁ - y₂| + |x₂ - y₁| := by
  rcases abs_cases (x₁ - y₁) with ⟨h1, s1⟩ | ⟨h1, s1⟩ <;>
    rcases abs_cases (x₂ - y₂) with ⟨h2, s2⟩ | ⟨h2, s2⟩ <;>
    rcases abs_cases (x₁ - y₂) with ⟨h3, s3⟩ | ⟨h3, s3⟩ <;>
    rcases abs_cases (x₂ - y₁) with ⟨h4, s4⟩ | ⟨h4, s4⟩ <;>
    rw [h1, h2, h3, h4] <;> linarith

/-- The Laplace kernel on `ℝ` as an explicit exponential of `|·|`. -/
lemma laplaceKernel_real (τ x y : ℝ) :
    laplaceKernel τ x y = Real.exp (-(1 / τ) * |x - y|) := by
  unfold laplaceKernel; rw [Real.norm_eq_abs]

/-- **Total positivity (TP2)** of the Laplace kernel: for `x₁ ≤ x₂` and `v ≤ u`,
`k(x₁,u)·k(x₂,v) ≤ k(x₁,v)·k(x₂,u)`. -/
lemma laplaceKernel_tp2 {τ : ℝ} (hτ : 0 < τ) {x₁ x₂ v u : ℝ}
    (hx : x₁ ≤ x₂) (hvu : v ≤ u) :
    laplaceKernel τ x₁ u * laplaceKernel τ x₂ v
      ≤ laplaceKernel τ x₁ v * laplaceKernel τ x₂ u := by
  rw [laplaceKernel_real, laplaceKernel_real, laplaceKernel_real, laplaceKernel_real,
    ← Real.exp_add, ← Real.exp_add, Real.exp_le_exp]
  have key : |x₁ - v| + |x₂ - u| ≤ |x₁ - u| + |x₂ - v| := abs_monge hx hvu
  have h1τ : (0 : ℝ) ≤ 1 / τ := by positivity
  nlinarith [mul_le_mul_of_nonneg_left key h1τ]

/-- The **symmetrized integrand is pointwise nonnegative**: for `x₁ ≤ x₂`,
`(u - v)·(k(x₁,v)k(x₂,u) - k(x₁,u)k(x₂,v)) ≥ 0`.  This is the pointwise engine of
tilted-mean monotonicity, and it is exactly the TP2 property above. -/
lemma laplace_symmetrized_nonneg {τ : ℝ} (hτ : 0 < τ) {x₁ x₂ : ℝ}
    (hx : x₁ ≤ x₂) (u v : ℝ) :
    0 ≤ (u - v) *
      (laplaceKernel τ x₁ v * laplaceKernel τ x₂ u
        - laplaceKernel τ x₁ u * laplaceKernel τ x₂ v) := by
  rcases le_total v u with h | h
  · exact mul_nonneg (by linarith) (by linarith [laplaceKernel_tp2 hτ hx h])
  · have htp := laplaceKernel_tp2 hτ hx h
    have hEq : (u - v) *
        (laplaceKernel τ x₁ v * laplaceKernel τ x₂ u
          - laplaceKernel τ x₁ u * laplaceKernel τ x₂ v)
        = (v - u) *
        (laplaceKernel τ x₁ u * laplaceKernel τ x₂ v
          - laplaceKernel τ x₁ v * laplaceKernel τ x₂ u) := by ring
    rw [hEq]
    exact mul_nonneg (by linarith) (by linarith)

/-- The Laplace **tilted mean** `μ_p(x) = (∫ y·k(x,y) dp)/(∫ k(x,y) dp)`. -/
noncomputable def laplaceTiltedMean (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (∫ y, laplaceKernel τ x y * y ∂p) / kernelNormalizer (laplaceKernel τ) p x

/-- **The linchpin (L3): the Laplace tilted mean is monotone.**  Proved from the
Monge/TP2 property of the kernel plus a symmetrization of the double integral —
NO correlation-inequality axiom, no Levinson, no Frobenius.  Axiom-free.  This is
the decisive test that the a.c.-converse derivation can avoid all project-own
axioms. -/
theorem laplaceTiltedMean_monotone {τ : ℝ} (hτ : 0 < τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hint : Integrable (fun y : ℝ => y) p) :
    Monotone (laplaceTiltedMean τ p) := by
  intro x₁ x₂ hx
  have hZ₁ : 0 < kernelNormalizer (laplaceKernel τ) p x₁ := laplaceKernelNormalizer_pos p τ hτ x₁
  have hZ₂ : 0 < kernelNormalizer (laplaceKernel τ) p x₂ := laplaceKernelNormalizer_pos p τ hτ x₂
  -- kernel sections are bounded, strongly measurable, hence integrable; y·k integrable
  have hkbound : ∀ x : ℝ, ∀ᵐ y ∂p, ‖laplaceKernel τ x y‖ ≤ (1 : ℝ) := by
    intro x; refine ae_of_all p (fun y => ?_)
    rw [laplaceKernel_real, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    refine Real.exp_le_one_iff.mpr ?_
    have h1 : (0 : ℝ) ≤ 1 / τ := by positivity
    nlinarith [mul_nonneg h1 (abs_nonneg (x - y))]
  have hkaesm : ∀ x : ℝ, AEStronglyMeasurable (fun y => laplaceKernel τ x y) p := by
    intro x
    have hc : Continuous fun y => laplaceKernel τ x y := by unfold laplaceKernel; fun_prop
    exact hc.aestronglyMeasurable
  have hZint : ∀ x : ℝ, Integrable (fun y => laplaceKernel τ x y) p := fun x =>
    (integrable_const (1 : ℝ)).mono' (hkaesm x) (hkbound x)
  have hAint : ∀ x : ℝ, Integrable (fun y => laplaceKernel τ x y * y) p := fun x =>
    hint.bdd_mul (hkaesm x) (hkbound x)
  -- reduce to the cross inequality on double integrals
  rw [laplaceTiltedMean, laplaceTiltedMean, div_le_div_iff₀ hZ₁ hZ₂]
  unfold kernelNormalizer
  have e1 : (∫ y, laplaceKernel τ x₁ y * y ∂p) * (∫ y, laplaceKernel τ x₂ y ∂p)
      = ∫ z, laplaceKernel τ x₁ z.1 * z.1 * laplaceKernel τ x₂ z.2 ∂(p.prod p) :=
    (integral_prod_mul (fun u => laplaceKernel τ x₁ u * u) (fun v => laplaceKernel τ x₂ v)).symm
  have e2 : (∫ y, laplaceKernel τ x₂ y * y ∂p) * (∫ y, laplaceKernel τ x₁ y ∂p)
      = ∫ z, laplaceKernel τ x₂ z.1 * z.1 * laplaceKernel τ x₁ z.2 ∂(p.prod p) :=
    (integral_prod_mul (fun u => laplaceKernel τ x₂ u * u) (fun v => laplaceKernel τ x₁ v)).symm
  rw [e1, e2]
  have hbig1 : Integrable
      (fun z : ℝ × ℝ => laplaceKernel τ x₁ z.1 * z.1 * laplaceKernel τ x₂ z.2) (p.prod p) :=
    (hAint x₁).mul_prod (hZint x₂)
  have hbig2 : Integrable
      (fun z : ℝ × ℝ => laplaceKernel τ x₂ z.1 * z.1 * laplaceKernel τ x₁ z.2) (p.prod p) :=
    (hAint x₂).mul_prod (hZint x₁)
  rw [← sub_nonneg, ← integral_sub hbig2 hbig1]
  -- symmetrize: ∫ F = ∫ F∘swap, so 2∫F = ∫(F + F∘swap), and F + F∘swap ≥ 0 pointwise
  have hFint : Integrable
      (fun z : ℝ × ℝ => laplaceKernel τ x₂ z.1 * z.1 * laplaceKernel τ x₁ z.2
        - laplaceKernel τ x₁ z.1 * z.1 * laplaceKernel τ x₂ z.2) (p.prod p) := hbig2.sub hbig1
  have hFswapint : Integrable
      (fun z : ℝ × ℝ => laplaceKernel τ x₂ z.2 * z.2 * laplaceKernel τ x₁ z.1
        - laplaceKernel τ x₁ z.2 * z.2 * laplaceKernel τ x₂ z.1) (p.prod p) :=
    hFint.swap
  have hswap := (integral_prod_swap (μ := p) (ν := p)
    (fun z : ℝ × ℝ => laplaceKernel τ x₂ z.1 * z.1 * laplaceKernel τ x₁ z.2
      - laplaceKernel τ x₁ z.1 * z.1 * laplaceKernel τ x₂ z.2)).symm
  have hsum :
      (∫ z, (laplaceKernel τ x₂ z.1 * z.1 * laplaceKernel τ x₁ z.2
          - laplaceKernel τ x₁ z.1 * z.1 * laplaceKernel τ x₂ z.2) ∂(p.prod p))
        + (∫ z, (laplaceKernel τ x₂ z.1 * z.1 * laplaceKernel τ x₁ z.2
          - laplaceKernel τ x₁ z.1 * z.1 * laplaceKernel τ x₂ z.2) ∂(p.prod p))
      = ∫ z, ((laplaceKernel τ x₂ z.1 * z.1 * laplaceKernel τ x₁ z.2
            - laplaceKernel τ x₁ z.1 * z.1 * laplaceKernel τ x₂ z.2)
          + (laplaceKernel τ x₂ z.2 * z.2 * laplaceKernel τ x₁ z.1
            - laplaceKernel τ x₁ z.2 * z.2 * laplaceKernel τ x₂ z.1)) ∂(p.prod p) := by
    rw [integral_add hFint hFswapint]
    rw [show (∫ z, (laplaceKernel τ x₂ z.2 * z.2 * laplaceKernel τ x₁ z.1
          - laplaceKernel τ x₁ z.2 * z.2 * laplaceKernel τ x₂ z.1) ∂(p.prod p))
        = ∫ z, (laplaceKernel τ x₂ z.1 * z.1 * laplaceKernel τ x₁ z.2
          - laplaceKernel τ x₁ z.1 * z.1 * laplaceKernel τ x₂ z.2) ∂(p.prod p) from hswap.symm]
  have hpos : 0 ≤ ∫ z, ((laplaceKernel τ x₂ z.1 * z.1 * laplaceKernel τ x₁ z.2
            - laplaceKernel τ x₁ z.1 * z.1 * laplaceKernel τ x₂ z.2)
          + (laplaceKernel τ x₂ z.2 * z.2 * laplaceKernel τ x₁ z.1
            - laplaceKernel τ x₁ z.2 * z.2 * laplaceKernel τ x₂ z.1)) ∂(p.prod p) := by
    refine integral_nonneg (fun z => ?_)
    simp only [Pi.zero_apply]
    have hnn := laplace_symmetrized_nonneg hτ hx z.1 z.2
    have hEq : laplaceKernel τ x₂ z.1 * z.1 * laplaceKernel τ x₁ z.2
          - laplaceKernel τ x₁ z.1 * z.1 * laplaceKernel τ x₂ z.2
        + (laplaceKernel τ x₂ z.2 * z.2 * laplaceKernel τ x₁ z.1
          - laplaceKernel τ x₁ z.2 * z.2 * laplaceKernel τ x₂ z.1)
        = (z.1 - z.2) * (laplaceKernel τ x₁ z.2 * laplaceKernel τ x₂ z.1
          - laplaceKernel τ x₁ z.1 * laplaceKernel τ x₂ z.2) := by ring
    rw [hEq]; exact hnn
  linarith [hsum, hpos]

end DriftingIdentifiability
