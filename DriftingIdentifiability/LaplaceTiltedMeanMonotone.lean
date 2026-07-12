import DriftingIdentifiability.LaplaceGeneralConverseBalance

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

/-- Strict form of `abs_monge` on the overlap region.  For two ordered
intervals `[x₁,x₂]` and `[v,u]`, the Monge inequality is strict exactly when the
intervals overlap with positive length.  The hypotheses `v < x₂` and `x₁ < u`
record that overlap. -/
lemma abs_monge_strict_of_overlap {x₁ x₂ v u : ℝ} (hx : x₁ < x₂) (hvu : v < u)
    (hvx : v < x₂) (hxu : x₁ < u) :
    |x₁ - v| + |x₂ - u| < |x₁ - u| + |x₂ - v| := by
  rcases abs_cases (x₁ - v) with ⟨h1, s1⟩ | ⟨h1, s1⟩ <;>
    rcases abs_cases (x₂ - u) with ⟨h2, s2⟩ | ⟨h2, s2⟩ <;>
    rcases abs_cases (x₁ - u) with ⟨h3, s3⟩ | ⟨h3, s3⟩ <;>
    rcases abs_cases (x₂ - v) with ⟨h4, s4⟩ | ⟨h4, s4⟩ <;>
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

/-- Strict TP2 on the overlap region.  This is the strict pointwise ingredient
needed for L7: once the tilted law puts positive mass on both sides of a
candidate zero, the symmetrized covariance integrand is positive on a product
set of positive mass. -/
lemma laplaceKernel_tp2_strict_of_overlap {τ : ℝ} (hτ : 0 < τ) {x₁ x₂ v u : ℝ}
    (hx : x₁ < x₂) (hvu : v < u) (hvx : v < x₂) (hxu : x₁ < u) :
    laplaceKernel τ x₁ u * laplaceKernel τ x₂ v
      < laplaceKernel τ x₁ v * laplaceKernel τ x₂ u := by
  rw [laplaceKernel_real, laplaceKernel_real, laplaceKernel_real, laplaceKernel_real,
    ← Real.exp_add, ← Real.exp_add, Real.exp_lt_exp]
  have key : |x₁ - v| + |x₂ - u| < |x₁ - u| + |x₂ - v| :=
    abs_monge_strict_of_overlap hx hvu hvx hxu
  have h1τ : (0 : ℝ) < 1 / τ := by positivity
  nlinarith [mul_lt_mul_of_pos_left key h1τ]

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

/-- Strict version of `laplace_symmetrized_nonneg` on the overlap region.  This
is the local L7 brick: strict positivity is available whenever the two probe
locations are ordered, the two sample locations are ordered, and the intervals
overlap.  The later measure-level L7 lemma should combine this pointwise
strictness with positive tilted mass on both sides of a mean-shift zero. -/
lemma laplace_symmetrized_pos_of_overlap {τ : ℝ} (hτ : 0 < τ) {x₁ x₂ v u : ℝ}
    (hx : x₁ < x₂) (hvu : v < u) (hvx : v < x₂) (hxu : x₁ < u) :
    0 < (u - v) *
      (laplaceKernel τ x₁ v * laplaceKernel τ x₂ u
        - laplaceKernel τ x₁ u * laplaceKernel τ x₂ v) := by
  have htp := laplaceKernel_tp2_strict_of_overlap hτ hx hvu hvx hxu
  exact mul_pos (sub_pos.mpr hvu) (sub_pos.mpr htp)

/-- Straddling form of `laplace_symmetrized_pos_of_overlap`.  This is the
version intended for the strict-at-zero L7 argument: if a candidate zero `x` is
surrounded by probe points `x₁ < x < x₂` and the tilted law has one sample below
`x` and one sample above `x`, then the pointwise symmetrized integrand is
strictly positive on that pair. -/
lemma laplace_symmetrized_pos_of_straddles {τ : ℝ} (hτ : 0 < τ)
    {x₁ x₂ x v u : ℝ} (hx₁ : x₁ < x) (hx₂ : x < x₂) (hv : v < x) (hu : x < u) :
    0 < (u - v) *
      (laplaceKernel τ x₁ v * laplaceKernel τ x₂ u
        - laplaceKernel τ x₁ u * laplaceKernel τ x₂ v) := by
  exact laplace_symmetrized_pos_of_overlap hτ (lt_trans hx₁ hx₂) (lt_trans hv hu)
    (lt_trans hv hx₂) (lt_trans hx₁ hu)

/-- The Laplace **tilted mean** `μ_p(x) = (∫ y·k(x,y) dp)/(∫ k(x,y) dp)`. -/
noncomputable def laplaceTiltedMean (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (∫ y, laplaceKernel τ x y * y ∂p) / kernelNormalizer (laplaceKernel τ) p x

/-- The same tilted mean, written in the displacement form `x + D/Z` used by
the mean-shift field.  This avoids differentiating `∫ y·k(x,y)` directly and
connects L7 to the certified first-order identities for `D` and `Z`. -/
noncomputable def laplaceTiltedMeanFromDisplacement
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  x + (∫ y, laplaceWeightedDisplacement τ x y ∂p) /
    kernelNormalizer (laplaceKernel τ) p x

/-- The integral definition of the tilted mean agrees with the displacement
form `x + D/Z`.  This bridge lets later L7/L8 work use the derivative theorem
for `laplaceTiltedMeanFromDisplacement` as a theorem about the usual tilted
mean. -/
theorem laplaceTiltedMean_eq_fromDisplacement
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ) :
    laplaceTiltedMean τ p x = laplaceTiltedMeanFromDisplacement τ p x := by
  have hZpos : 0 < kernelNormalizer (laplaceKernel τ) p x :=
    laplaceKernelNormalizer_pos p τ hτ x
  have hDint : Integrable (fun y => laplaceWeightedDisplacement τ x y) p :=
    laplaceWeightedDisplacement_integrable τ hτ p x
  have hKint : Integrable (fun y => laplaceKernel τ x y) p :=
    laplaceKernel_integrable p τ hτ x
  have hXKint : Integrable (fun y => x * laplaceKernel τ x y) p :=
    hKint.const_mul x
  have hpoint :
      (fun y : ℝ => laplaceKernel τ x y * y) =
        fun y : ℝ => laplaceWeightedDisplacement τ x y + x * laplaceKernel τ x y := by
    funext y
    unfold laplaceWeightedDisplacement
    rw [smul_eq_mul]
    ring
  unfold laplaceTiltedMean laplaceTiltedMeanFromDisplacement
  rw [hpoint, integral_add hDint hXKint, integral_const_mul]
  field_simp [hZpos.ne']
  unfold kernelNormalizer
  ring

/-- Upper one-sided exponential mass is nonnegative. -/
lemma upperExpMass_nonneg (τ : ℝ) (p : Measure ℝ) (x : ℝ) :
    0 ≤ upperExpMass τ p x :=
  setIntegral_nonneg measurableSet_Ioi (fun _ _ => (Real.exp_pos _).le)

/-- Upper compensated moment is nonnegative. -/
lemma upperCompensatedMoment_nonneg (τ : ℝ) (p : Measure ℝ) (x : ℝ) :
    0 ≤ upperCompensatedMoment τ p x :=
  setIntegral_nonneg measurableSet_Ioi
    (fun _ hy => mul_nonneg (sub_nonneg.mpr (le_of_lt (Set.mem_Ioi.mp hy)))
      (Real.exp_pos _).le)

/-- Strict positivity of the upper exponential mass from positive right-tail
mass. -/
lemma upperExpMass_pos_of_measure_Ioi_pos
    (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ)
    (hp : 0 < p (Set.Ioi x)) :
    0 < upperExpMass τ p x := by
  have hset :
      Function.support (fun y : ℝ => Real.exp (-y / τ)) ∩ Set.Ioi x = Set.Ioi x := by
    ext y
    constructor
    · intro hy
      exact hy.2
    · intro hy
      exact ⟨(Real.exp_pos _).ne', hy⟩
  unfold upperExpMass
  rw [setIntegral_pos_iff_support_of_nonneg_ae
      (by filter_upwards [ae_restrict_mem measurableSet_Ioi] with _ hy
          exact (Real.exp_pos _).le)
      (integrable_upperExpKernel τ hτ p x), hset]
  exact hp

/-- The right-derivative coefficient for the displacement-form tilted mean.

The algebraic content is

`mu'_+ * Z^2 =
  (2/tau) * (e^- lowerComp * e^+ upperMass
    + e^+ upperComp * e^- lowerMass)`,

so positivity follows immediately when the tilted law has mass on both sides of
`x`. -/
noncomputable def laplaceTiltedMeanRightDerivCoeff
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ((2 / τ) *
      (lowerCompensatedMoment τ p x * upperExpMass τ p x +
        upperCompensatedMoment τ p x * lowerExpMass τ p x)) /
    (kernelNormalizer (laplaceKernel τ) p x) ^ 2

private lemma laplaceTiltedMean_rightDeriv_algebra
    {tau A B lm um lc uc Z : ℝ} (htau : tau ≠ 0) (hZne : Z ≠ 0)
    (hZ : Z = A * lm + B * um) (hAB : A * B = 1) :
    (1 : ℝ) +
        ((((1 / tau) * (A * lc + B * uc) - (A * lm + B * um)) * Z -
            (-(A) * lc + B * uc) * ((1 / tau) * (-(A) * lm + B * um))) /
          Z ^ 2) =
      ((2 / tau) * (lc * um + uc * lm)) / Z ^ 2 := by
  subst Z
  have hZsq : (A * lm + B * um) ^ 2 ≠ 0 := pow_ne_zero 2 hZne
  field_simp [htau, hZne, hZsq]
  ring_nf
  have h1 : A * lm * B * uc * 2 = lm * uc * 2 := by
    calc
      A * lm * B * uc * 2 = (A * B) * (lm * uc * 2) := by ring
      _ = lm * uc * 2 := by rw [hAB]; ring
  have h2 : A * B * um * lc * 2 = um * lc * 2 := by
    calc
      A * B * um * lc * 2 = (A * B) * (um * lc * 2) := by ring
      _ = um * lc * 2 := by rw [hAB]; ring
  nlinarith

/-- **L7 derivative formula, right-derivative form.**  The displacement-form
tilted mean has a right derivative whose coefficient is the positive one-sided
mass expression above.  This uses only certified first-order data:
`D' = L/tau - 2Z`, the one-sided formula for `D`, and the certified right
derivative of `Z`. -/
theorem hasDerivWithinAt_Ici_laplaceTiltedMeanFromDisplacement
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ) :
    HasDerivWithinAt (fun t => laplaceTiltedMeanFromDisplacement τ p t)
      (laplaceTiltedMeanRightDerivCoeff τ p x) (Set.Ici x) x := by
  have hZpos : 0 < kernelNormalizer (laplaceKernel τ) p x :=
    laplaceKernelNormalizer_pos p τ hτ x
  have hD₀ :=
    (hasDerivAt_laplaceDisplacementIntegral τ hτ p x).hasDerivWithinAt
      (s := Set.Ici x)
  have hD : HasDerivWithinAt
      (fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂p)
      (laplaceDisplacementIntegralDerivCoeff τ p x) (Set.Ici x) x := by
    rw [← laplaceDisplacementIntegral_derivCoeff_eq τ hτ p x]
    exact hD₀
  have hZ := hasDerivWithinAt_Ici_laplaceKernelNormalizer τ hτ p x
  have hquot := hD.div hZ (ne_of_gt hZpos)
  have hid : HasDerivWithinAt (fun t : ℝ => t) 1 (Set.Ici x) x :=
    (hasDerivAt_id x).hasDerivWithinAt
  have hsum := hid.add hquot
  have hcoeff :
      (1 : ℝ) +
          (laplaceDisplacementIntegralDerivCoeff τ p x *
                kernelNormalizer (laplaceKernel τ) p x -
              (∫ y, laplaceWeightedDisplacement τ x y ∂p) *
                laplaceKernelNormalizerRightDerivCoeff τ p x) /
            (kernelNormalizer (laplaceKernel τ) p x) ^ 2 =
        laplaceTiltedMeanRightDerivCoeff τ p x := by
    let A : ℝ := Real.exp (-x / τ)
    let B : ℝ := Real.exp (x / τ)
    let lm : ℝ := lowerExpMass τ p x
    let um : ℝ := upperExpMass τ p x
    let lc : ℝ := lowerCompensatedMoment τ p x
    let uc : ℝ := upperCompensatedMoment τ p x
    have hZval : kernelNormalizer (laplaceKernel τ) p x = A * lm + B * um := by
      dsimp [A, B, lm, um]
      exact laplaceKernelNormalizer_eq_lower_upper τ hτ p x
    have hAB : A * B = 1 := by
      dsimp [A, B]
      rw [← Real.exp_add]
      have harg : -x / τ + x / τ = 0 := by
        field_simp [hτ.ne']
        ring
      rw [harg, Real.exp_zero]
    have halg := laplaceTiltedMean_rightDeriv_algebra
      (tau := τ) (A := A) (B := B) (lm := lm) (um := um) (lc := lc) (uc := uc)
      (Z := kernelNormalizer (laplaceKernel τ) p x) hτ.ne' (ne_of_gt hZpos)
      hZval hAB
    simpa [laplaceTiltedMeanRightDerivCoeff, laplaceKernelNormalizerRightDerivCoeff,
      laplaceDisplacementIntegralDerivCoeff, A, B, lm, um, lc, uc,
      laplaceDisplacementIntegral_eq_lower_upper τ hτ p x] using halg
  change HasDerivWithinAt
    ((fun t : ℝ => t) +
      ((fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂p) /
        fun t : ℝ => kernelNormalizer (laplaceKernel τ) p t))
    (laplaceTiltedMeanRightDerivCoeff τ p x) (Set.Ici x) x
  simpa [hcoeff] using hsum

/-- **L7 strictness.**  If the law has positive mass strictly below and strictly
above `x`, then the right-derivative coefficient of the shifted tilted mean is
strictly positive at `x`.  This is the formal straddling condition used at
mean-shift zeros in the a.c. converse plan. -/
theorem laplaceTiltedMeanRightDerivCoeff_pos_of_twoSidedMass
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ)
    (hleft : 0 < p (Set.Iio x)) (hright : 0 < p (Set.Ioi x)) :
    0 < laplaceTiltedMeanRightDerivCoeff τ p x := by
  have hZpos : 0 < kernelNormalizer (laplaceKernel τ) p x :=
    laplaceKernelNormalizer_pos p τ hτ x
  have hLowerCompPos : 0 < lowerCompensatedMoment τ p x :=
    (lowerCompensatedMoment_pos_iff τ hτ p x).mpr hleft
  have hUpperExpPos : 0 < upperExpMass τ p x :=
    upperExpMass_pos_of_measure_Ioi_pos τ hτ p x hright
  have hUpperCompNonneg : 0 ≤ upperCompensatedMoment τ p x :=
    upperCompensatedMoment_nonneg τ p x
  have hLowerExpNonneg : 0 ≤ lowerExpMass τ p x :=
    lowerExpMass_nonneg τ p x
  have hterm₁ :
      0 < lowerCompensatedMoment τ p x * upperExpMass τ p x := by
    exact mul_pos hLowerCompPos hUpperExpPos
  have hterm₂ :
      0 ≤ upperCompensatedMoment τ p x * lowerExpMass τ p x := by
    exact mul_nonneg hUpperCompNonneg hLowerExpNonneg
  have hsum :
      0 < lowerCompensatedMoment τ p x * upperExpMass τ p x +
          upperCompensatedMoment τ p x * lowerExpMass τ p x :=
    add_pos_of_pos_of_nonneg hterm₁ hterm₂
  have hscale : 0 < 2 / τ := div_pos (by norm_num) hτ
  have hnum :
      0 < (2 / τ) *
        (lowerCompensatedMoment τ p x * upperExpMass τ p x +
          upperCompensatedMoment τ p x * lowerExpMass τ p x) :=
    mul_pos hscale hsum
  have hden : 0 < (kernelNormalizer (laplaceKernel τ) p x) ^ 2 :=
    sq_pos_of_pos hZpos
  unfold laplaceTiltedMeanRightDerivCoeff
  exact div_pos hnum hden

/-- Combined L7 package: under two-sided mass, the shifted tilted mean has a
strictly positive right derivative at `x`. -/
theorem hasStrictDerivWithinAt_Ici_laplaceTiltedMeanFromDisplacement_of_twoSidedMass
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ)
    (hleft : 0 < p (Set.Iio x)) (hright : 0 < p (Set.Ioi x)) :
    ∃ c : ℝ, 0 < c ∧
      HasDerivWithinAt (fun t => laplaceTiltedMeanFromDisplacement τ p t)
        c (Set.Ici x) x := by
  refine ⟨laplaceTiltedMeanRightDerivCoeff τ p x,
    laplaceTiltedMeanRightDerivCoeff_pos_of_twoSidedMass τ hτ p x hleft hright,
    hasDerivWithinAt_Ici_laplaceTiltedMeanFromDisplacement τ hτ p x⟩

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
