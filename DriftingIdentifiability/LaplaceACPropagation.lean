import DriftingIdentifiability.LaplaceACAbel
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.Normed.Ring.Lemmas

/-!
# Propagation lemmas for the absolutely-continuous Laplace converse

This file contains the deterministic ODE/Abel propagation layer for the
`LaplaceACDerivation.md` endgame.  The lemmas here are deliberately abstract:
they do not mention `p` or `q`; instead they consume the Abel equation supplied
by `LaplaceACAbel.lean`.

The intended dictionary is:

* `W` is the normalizer Wronskian;
* `c = 2 μ' / m` on a sign interval of the mean shift `m`;
* `A` is a finite-interval primitive of `c`;
* Abel says `W' = -c W`;
* therefore `W * exp A` is constant.

L6 uses the tail lemma when the integrating factor has a finite tail limit and
`W → 0`.  L8 uses the bounded-squeeze lemma when the integrating factor tends
to zero at an upward crossing.  L9 then becomes a covering/gluing statement.
-/

open Filter Set Topology

namespace DriftingIdentifiability

/-- A finite-interval Grönwall propagation lemma for Abel-type equations.

If `W` is continuous on `[a,b]`, has right derivative `c x * W x` on `[a,b)`,
and the coefficient is bounded on the interval, then one zero endpoint forces
`W` to vanish on the whole interval.

This is the local continuation/gluing tool used after an outer or crossing
argument has produced one zero value. -/
theorem abel_zero_propagates_Icc
    {W c : ℝ → ℝ} {a b : ℝ}
    (hcont : ContinuousOn W (Icc a b))
    (hderiv : ∀ x ∈ Ico a b, HasDerivWithinAt W (c x * W x) (Ici x) x)
    (hbound : ∃ K : ℝ, 0 ≤ K ∧ ∀ x ∈ Ico a b, ‖c x‖ ≤ K)
    (ha : W a = 0) :
    ∀ x ∈ Icc a b, W x = 0 := by
  rcases hbound with ⟨K, hKnonneg, hK⟩
  refine eq_zero_of_abs_deriv_le_mul_abs_self_of_eq_zero_right
    (f := W) (f' := fun x => c x * W x) (K := K)
    hcont hderiv ha ?_
  intro x hx
  calc
    ‖c x * W x‖ = ‖c x‖ * ‖W x‖ := norm_mul _ _
    _ ≤ K * ‖W x‖ := mul_le_mul_of_nonneg_right (hK x hx) (norm_nonneg _)

/-- Integrating-factor constancy on a finite interval.

If `W' = -cW` and `A' = c` on `[a,b)`, then `W * exp A` is constant on
`[a,b]`.  This is the exact Abel identity used by both the outer BV argument
and the interior blow-up argument. -/
theorem abel_integratingFactor_const_Icc
    {W A c : ℝ → ℝ} {a b : ℝ}
    (hcont : ContinuousOn (fun x => W x * Real.exp (A x)) (Icc a b))
    (hW : ∀ x ∈ Ico a b, HasDerivWithinAt W (-(c x) * W x) (Ici x) x)
    (hA : ∀ x ∈ Ico a b, HasDerivWithinAt A (c x) (Ici x) x) :
    ∀ x ∈ Icc a b, W x * Real.exp (A x) = W a * Real.exp (A a) := by
  refine constant_of_has_deriv_right_zero hcont ?_
  intro x hx
  have hExp : HasDerivWithinAt (Real.exp ∘ A)
      (Real.exp (A x) * c x) (Ici x) x := by
    exact (Real.hasDerivAt_exp (A x)).comp_hasDerivWithinAt x (hA x hx)
  have hProd := (hW x hx).mul hExp
  change HasDerivWithinAt (W * (Real.exp ∘ A)) 0 (Ici x) x
  simpa [mul_assoc, mul_left_comm, mul_comm] using hProd

/-- If a function is eventually equal to a constant and also tends to zero, then
the constant is zero.  This tiny lemma is the reusable endpoint of the tail and
boundedness squeezes below. -/
theorem eventually_const_eq_zero_of_tendsto_zero
    {α : Type*} {l : Filter α} [NeBot l] {F : α → ℝ} {C : ℝ}
    (hconst : ∀ᶠ x in l, F x = C)
    (hzero : Tendsto F l (𝓝 0)) :
    C = 0 := by
  have hconst_tendsto : Tendsto (fun _ : α => C) l (𝓝 C) := tendsto_const_nhds
  have hconst_zero : Tendsto (fun _ : α => C) l (𝓝 0) :=
    hzero.congr' hconst
  exact tendsto_nhds_unique hconst_tendsto hconst_zero

/-- L6-style right-tail squeeze from an already established integrating-factor
constant.  If `W x * exp(A x)` is constant for all sufficiently far-right
points, `W → 0`, and the integrating factor has a finite limit, then the
constant at the base point is zero; since `exp(A a) ≠ 0`, `W a = 0`. -/
theorem abel_right_tail_zero_of_integratingFactor_const
    {W A : ℝ → ℝ} {a L : ℝ}
    (hconst : ∀ b : ℝ, a ≤ b →
      W b * Real.exp (A b) = W a * Real.exp (A a))
    (hWtail : Tendsto W atTop (𝓝 0))
    (hAtail : Tendsto (fun x => Real.exp (A x)) atTop (𝓝 L)) :
    W a = 0 := by
  have hprod_zero : Tendsto (fun x => W x * Real.exp (A x)) atTop (𝓝 0) := by
    simpa using (hWtail.mul hAtail)
  have hevent : ∀ᶠ x in atTop,
      W x * Real.exp (A x) = W a * Real.exp (A a) :=
    eventually_atTop.2 ⟨a, fun x hx => hconst x hx⟩
  have hbase : W a * Real.exp (A a) = 0 :=
    eventually_const_eq_zero_of_tendsto_zero hevent hprod_zero
  exact (mul_eq_zero.mp hbase).resolve_right (Real.exp_ne_zero _)

/-- L6-style left-tail squeeze from an already established integrating-factor
constant. -/
theorem abel_left_tail_zero_of_integratingFactor_const
    {W A : ℝ → ℝ} {a L : ℝ}
    (hconst : ∀ b : ℝ, b ≤ a →
      W b * Real.exp (A b) = W a * Real.exp (A a))
    (hWtail : Tendsto W atBot (𝓝 0))
    (hAtail : Tendsto (fun x => Real.exp (A x)) atBot (𝓝 L)) :
    W a = 0 := by
  have hprod_zero : Tendsto (fun x => W x * Real.exp (A x)) atBot (𝓝 0) := by
    simpa using (hWtail.mul hAtail)
  have hevent : ∀ᶠ x in atBot,
      W x * Real.exp (A x) = W a * Real.exp (A a) :=
    eventually_atBot.2 ⟨a, fun x hx => hconst x hx⟩
  have hbase : W a * Real.exp (A a) = 0 :=
    eventually_const_eq_zero_of_tendsto_zero hevent hprod_zero
  exact (mul_eq_zero.mp hbase).resolve_right (Real.exp_ne_zero _)

/-- Right-tail L6 wrapper: derive the constant relation on every `[a,b]` from
the Abel equation and an integrating-factor primitive, then squeeze with
`W → 0`. -/
theorem abel_right_tail_zero_of_integratingFactor
    {W A c : ℝ → ℝ} {a L : ℝ}
    (hcont : ∀ b : ℝ, a ≤ b →
      ContinuousOn (fun x => W x * Real.exp (A x)) (Icc a b))
    (hW : ∀ b : ℝ, a ≤ b →
      ∀ x ∈ Ico a b, HasDerivWithinAt W (-(c x) * W x) (Ici x) x)
    (hA : ∀ b : ℝ, a ≤ b →
      ∀ x ∈ Ico a b, HasDerivWithinAt A (c x) (Ici x) x)
    (hWtail : Tendsto W atTop (𝓝 0))
    (hAtail : Tendsto (fun x => Real.exp (A x)) atTop (𝓝 L)) :
    W a = 0 := by
  refine abel_right_tail_zero_of_integratingFactor_const
    (W := W) (A := A) (a := a) (L := L) ?_ hWtail hAtail
  intro b hab
  exact abel_integratingFactor_const_Icc (hcont b hab) (hW b hab) (hA b hab)
    b ⟨hab, le_rfl⟩

/-- Left-tail L6 wrapper: the symmetric version of
`abel_right_tail_zero_of_integratingFactor`. -/
theorem abel_left_tail_zero_of_integratingFactor
    {W A c : ℝ → ℝ} {a L : ℝ}
    (hcont : ∀ b : ℝ, b ≤ a →
      ContinuousOn (fun x => W x * Real.exp (A x)) (Icc b a))
    (hW : ∀ b : ℝ, b ≤ a →
      ∀ x ∈ Ico b a, HasDerivWithinAt W (-(c x) * W x) (Ici x) x)
    (hA : ∀ b : ℝ, b ≤ a →
      ∀ x ∈ Ico b a, HasDerivWithinAt A (c x) (Ici x) x)
    (hWtail : Tendsto W atBot (𝓝 0))
    (hAtail : Tendsto (fun x => Real.exp (A x)) atBot (𝓝 L)) :
    W a = 0 := by
  refine abel_left_tail_zero_of_integratingFactor_const
    (W := W) (A := A) (a := a) (L := L) ?_ hWtail hAtail
  intro b hba
  exact (abel_integratingFactor_const_Icc (hcont b hba) (hW b hba) (hA b hba)
    a ⟨hba, le_rfl⟩).symm

/-- L8-style boundedness squeeze.  If `W * E` is eventually constant along a
one-sided approach to a crossing, `W` is bounded there, and the integrating
factor `E` tends to zero, then the constant must be zero.

For the AC converse, `E = exp A`.  The divergence statement at an upward
crossing is precisely the input `E → 0`; this lemma performs the final
boundedness contradiction without encoding any crossing geometry. -/
theorem bounded_factor_const_zero_of_factor_tendsto_zero
    {α : Type*} {l : Filter α} [NeBot l] {W E : α → ℝ} {C : ℝ}
    (hconst : ∀ᶠ x in l, W x * E x = C)
    (hWbounded : IsBoundedUnder (· ≤ ·) l (norm ∘ W))
    (hEzero : Tendsto E l (𝓝 0)) :
    C = 0 := by
  have hprod_zero : Tendsto (fun x => W x * E x) l (𝓝 0) :=
    Filter.isBoundedUnder_le_mul_tendsto_zero hWbounded hEzero
  exact eventually_const_eq_zero_of_tendsto_zero hconst hprod_zero

/-- L8 in the concrete integrating-factor notation. -/
theorem bounded_integratingFactor_const_zero_of_factor_tendsto_zero
    {α : Type*} {l : Filter α} [NeBot l] {W A : α → ℝ} {C : ℝ}
    (hconst : ∀ᶠ x in l, W x * Real.exp (A x) = C)
    (hWbounded : IsBoundedUnder (· ≤ ·) l (norm ∘ W))
    (hEzero : Tendsto (fun x => Real.exp (A x)) l (𝓝 0)) :
    C = 0 :=
  bounded_factor_const_zero_of_factor_tendsto_zero hconst hWbounded hEzero

end DriftingIdentifiability
