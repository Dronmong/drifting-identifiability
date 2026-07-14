import DriftingIdentifiability.LaplaceRadialFoundations

/-!
# Higher-dimensional Laplace converse, milestone L3: atom-alignment gate

This file implements the axiom-free algebraic core of milestone `L3` from
`LaplaceHigherDim.md`.

The intended analytic proof extracts atoms from the conical defect of the
Laplace normalizer by averaging over small balls.  The hard part is the
measure-theoretic/cone-extraction estimate showing that the product

`x ↦ Z_ν(x) • D_μ(x)`

has small-ball defect converging to `ν({a}) • D_μ(a)`.  Rather than axiomatize
that estimate, this file records it as an explicit hypothesis structure
`LaplaceAtomConeProductData` and proves the exact consequence needed downstream:
zero drift forces the atom-alignment identity

`q({a}) • D_p(a) = p({a}) • D_q(a)`.

So this is not a new trusted assumption; it is the verified algebraic gate that
the future ball-average asymptotics must feed.
-/

open MeasureTheory Filter Topology
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasureSpace E] [BorelSpace E] [CompleteSpace E] [SecondCountableTopology E]

/-! ## Ball-average cone coefficients -/

/-- Average of a Bochner-valued function over the ambient ball `B(a, ε)`.

The definition deliberately uses `toReal` so it remains a total function even
outside the usual `0 < volume (ball a ε) < ∞` regime.  The cone-asymptotic
hypotheses below are only intended near positive radii, where the standard
Euclidean finite-dimensional proof will supply the nonzero finite-volume facts. -/
noncomputable def ballAverage (f : E → E) (a : E) (ε : ℝ) : E :=
  ((volume (Metric.ball a ε)).toReal)⁻¹ • ∫ x in Metric.ball a ε, f x

/-- The scale predicted by the `n`-dimensional cone computation:
`((n+1) τ)/(n ε)`.  The intended use is `n = finrank ℝ E` with `2 ≤ n`. -/
noncomputable def laplaceAtomConeScale (τ : ℝ) (n : ℕ) (ε : ℝ) : ℝ :=
  (((n : ℝ) + 1) * τ) / ((n : ℝ) * ε)

/-- Small-ball cone coefficient of a vector-valued function at `a`. -/
noncomputable def laplaceAtomConeCoeff (τ : ℝ) (n : ℕ)
    (f : E → E) (a : E) (ε : ℝ) : E :=
  laplaceAtomConeScale τ n ε • (f a - ballAverage f a ε)

/-! ## Atom masses and the product field `Z_ν • D_μ` -/

/-- The real mass of the atom `{a}`.  For finite measures this is finite and
nonnegative. -/
noncomputable def atomMassReal (μ : Measure E) (a : E) : ℝ :=
  (μ {a}).toReal

omit [NormedAddCommGroup E] [InnerProductSpace ℝ E] [BorelSpace E]
  [CompleteSpace E] [SecondCountableTopology E] in
lemma atomMassReal_nonneg (μ : Measure E) (a : E) :
    0 ≤ atomMassReal μ a := by
  unfold atomMassReal
  exact ENNReal.toReal_nonneg

/-- The product field whose cone coefficient detects an atom of `ν` against
the displacement field of `μ`. -/
noncomputable def laplaceNormalizerDisplacementProduct
    (τ : ℝ) (ν μ : Measure E) (x : E) : E :=
  kernelNormalizer (laplaceKernel τ) ν x • laplaceDisplacementField τ μ x

/-- Explicit cone-product data at `a`.

Analytically, these two fields are meant to be proved from the ball-average
asymptotics in `LaplaceHigherDim.md`:

* the normalizer has a first-order conical defect equal to the atom mass;
* the displacement numerator has zero conical defect;
* the product rule therefore sends `Z_ν • D_μ` to `ν({a}) • D_μ(a)`.

Keeping this as a structure makes the trusted boundary sharp: downstream
atom-alignment theorems depend only on visible, checkable limit hypotheses, not
on an axiom or an opaque theorem. -/
structure LaplaceAtomConeProductData
    (τ : ℝ) (n : ℕ) (p q : Measure E) (a : E) : Prop where
  qZ_pD :
    Tendsto
      (fun ε : ℝ =>
        laplaceAtomConeCoeff τ n
          (laplaceNormalizerDisplacementProduct τ q p) a ε)
      (𝓝[>] (0 : ℝ))
      (𝓝 (atomMassReal q a • laplaceDisplacementField τ p a))
  pZ_qD :
    Tendsto
      (fun ε : ℝ =>
        laplaceAtomConeCoeff τ n
          (laplaceNormalizerDisplacementProduct τ p q) a ε)
      (𝓝[>] (0 : ℝ))
      (𝓝 (atomMassReal p a • laplaceDisplacementField τ q a))

/-! ## Zero drift forces atom alignment once the cone data is available -/

set_option linter.unusedSectionVars false in
/-- Zero drift makes the two product fields `Z_q • D_p` and `Z_p • D_q`
pointwise equal. -/
lemma laplaceNormalizerDisplacementProduct_eq_of_zeroDrift
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) :
    laplaceNormalizerDisplacementProduct τ q p =
      laplaceNormalizerDisplacementProduct τ p q := by
  funext x
  exact zeroDrift_displacementAligned hτ p q hzero x

/-- **L3 atom-alignment gate.**  Under the explicit cone-product data at `a`,
zero raw Laplace drift forces the atom-alignment identity

`q({a}) • D_p(a) = p({a}) • D_q(a)`.

This is the formal target that the future ball-average cone-extraction lemmas
must feed. -/
theorem laplaceZeroDrift_atomAlignment_of_coneProductData
    {τ : ℝ} (hτ : 0 < τ) {n : ℕ} (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (a : E)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hcone : LaplaceAtomConeProductData τ n p q a) :
    atomMassReal q a • laplaceDisplacementField τ p a =
      atomMassReal p a • laplaceDisplacementField τ q a := by
  have hfun := laplaceNormalizerDisplacementProduct_eq_of_zeroDrift hτ p q hzero
  have hseq :
      (fun ε : ℝ =>
        laplaceAtomConeCoeff τ n
          (laplaceNormalizerDisplacementProduct τ q p) a ε) =
      fun ε : ℝ =>
        laplaceAtomConeCoeff τ n
          (laplaceNormalizerDisplacementProduct τ p q) a ε := by
    funext ε
    rw [hfun]
  have hright :
      Tendsto
        (fun ε : ℝ =>
          laplaceAtomConeCoeff τ n
            (laplaceNormalizerDisplacementProduct τ q p) a ε)
        (𝓝[>] (0 : ℝ))
        (𝓝 (atomMassReal p a • laplaceDisplacementField τ q a)) := by
    rw [hseq]
    exact hcone.pZ_qD
  exact tendsto_nhds_unique hcone.qZ_pD hright

/-- At points where the displacement numerator of `p` is nonzero, atom
alignment plus zero-drift displacement alignment gives the scalar mass-ratio
identity

`q({a}) * Z_p(a) = p({a}) * Z_q(a)`.
-/
theorem laplaceZeroDrift_atomMassRatio_of_coneProductData
    {τ : ℝ} (hτ : 0 < τ) {n : ℕ} (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (a : E)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hcone : LaplaceAtomConeProductData τ n p q a)
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
    exact laplaceZeroDrift_atomAlignment_of_coneProductData hτ p q a hzero hcone
  rw [hDq] at halign
  rw [smul_smul] at halign
  have hscalar :
      atomMassReal q a = atomMassReal p a * (Zp⁻¹ * Zq) :=
    smul_left_injective ℝ hDp_ne' halign
  calc atomMassReal q a * Zp
      = (atomMassReal p a * (Zp⁻¹ * Zq)) * Zp := by rw [hscalar]
    _ = atomMassReal p a * Zq := by
        field_simp [hZp_pos.ne']

/-- Nonzero-displacement atom-rigidity form: under the cone-product data, a
point is an atom of `p` iff it is an atom of `q`, provided the common drift
numerator is nonzero there. -/
theorem laplaceZeroDrift_atomMass_zero_iff_of_coneProductData
    {τ : ℝ} (hτ : 0 < τ) {n : ℕ} (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q] (a : E)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (hcone : LaplaceAtomConeProductData τ n p q a)
    (hDp_ne : laplaceDisplacementField τ p a ≠ 0) :
    atomMassReal p a = 0 ↔ atomMassReal q a = 0 := by
  have hratio := laplaceZeroDrift_atomMassRatio_of_coneProductData
    hτ p q a hzero hcone hDp_ne
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

end DriftingIdentifiability
