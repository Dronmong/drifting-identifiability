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

and its upper-tail mirror.  The direct double-integral and four-region
decomposition of the cross-displacement field are the next Milestone-1
subgoals; this file deliberately does not pretend those analytic bookkeeping
steps are already done.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-! ## Scalar bounds for the one-sided kernels -/

/-- The elementary bound `s * exp (-s/τ) ≤ τ` for `s ≥ 0`, used to make the
compensated one-sided moments finite without moment assumptions. -/
private lemma mul_exp_neg_le_general {τ : ℝ} (hτ : 0 < τ) {s : ℝ} (hs : 0 ≤ s) :
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

private lemma integrable_lowerExpKernel
    (τ : ℝ) (hτ : 0 < τ) (p : Measure ℝ) [IsFiniteMeasure p] (x : ℝ) :
    Integrable (fun y : ℝ => Real.exp (y / τ)) (p.restrict (Set.Iic x)) := by
  refine Integrable.of_bound (by fun_prop) (Real.exp (x / τ)) ?_
  filter_upwards [ae_restrict_mem measurableSet_Iic] with y hy
  rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  have hy' : y ≤ x := by simpa using hy
  exact Real.exp_le_exp.mpr (by gcongr)

private lemma integrable_lowerCompKernel
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
