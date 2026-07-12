import DriftingIdentifiability.LaplaceACAbel
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.Normed.Ring.Lemmas
import Mathlib.Topology.Algebra.Module.Cardinality
import Mathlib.Topology.Order.OrderClosed

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

/-- Continuity of `exp` packaged in the form used by the integrating-factor
lemmas. -/
theorem exp_comp_tendsto_nhds
    {α : Type*} {l : Filter α} {A : α → ℝ} {L : ℝ}
    (hA : Tendsto A l (𝓝 L)) :
    Tendsto (fun x => Real.exp (A x)) l (𝓝 (Real.exp L)) :=
  (Real.continuous_exp.tendsto L).comp hA

/-- If the primitive tends to `-∞`, its exponential integrating factor tends
to zero.  This is the analytic shape of the L8 upward-crossing divergence. -/
theorem exp_comp_tendsto_zero_of_tendsto_atBot
    {α : Type*} {l : Filter α} {A : α → ℝ}
    (hA : Tendsto A l atBot) :
    Tendsto (fun x => Real.exp (A x)) l (𝓝 0) :=
  Real.tendsto_exp_atBot.comp hA

/-! ## L6: outer intervals -/

/-- **L6 right outer interval.**  If every finite interval `[x,b]` in the
right outer region satisfies the Abel/integrating-factor hypotheses, `W → 0`
at `+∞`, and the integrating factor has a finite tail limit, then `W` vanishes
throughout the right outer ray.

The analytic BV work in the a.c. converse is precisely to construct such an
`A` with `A' = c = 2μ'/m` and finite `exp A` limit on the outer sign interval. -/
theorem abel_right_outer_zero_of_integratingFactor
    {W A c : ℝ → ℝ} {a L : ℝ}
    (hcont : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x b))
    (hW : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ∀ t ∈ Ico x b, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ∀ t ∈ Ico x b, HasDerivWithinAt A (c t) (Ici t) t)
    (hWtail : Tendsto W atTop (𝓝 0))
    (hAtail : Tendsto (fun x => Real.exp (A x)) atTop (𝓝 L)) :
    ∀ x : ℝ, a ≤ x → W x = 0 := by
  intro x hax
  exact abel_right_tail_zero_of_integratingFactor
    (W := W) (A := A) (c := c) (a := x) (L := L)
    (fun b hxb => hcont x b hax hxb)
    (fun b hxb => hW x b hax hxb)
    (fun b hxb => hA x b hax hxb)
    hWtail hAtail

/-- L6 right outer interval, stated using convergence of the primitive `A`
itself rather than convergence of `exp A`. -/
theorem abel_right_outer_zero_of_integratingFactor_of_tendsto_primitive
    {W A c : ℝ → ℝ} {a L : ℝ}
    (hcont : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x b))
    (hW : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ∀ t ∈ Ico x b, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ∀ t ∈ Ico x b, HasDerivWithinAt A (c t) (Ici t) t)
    (hWtail : Tendsto W atTop (𝓝 0))
    (hAtail : Tendsto A atTop (𝓝 L)) :
    ∀ x : ℝ, a ≤ x → W x = 0 :=
  abel_right_outer_zero_of_integratingFactor hcont hW hA hWtail
    (exp_comp_tendsto_nhds hAtail)

/-- **L6 left outer interval.**  Symmetric version of
`abel_right_outer_zero_of_integratingFactor`. -/
theorem abel_left_outer_zero_of_integratingFactor
    {W A c : ℝ → ℝ} {a L : ℝ}
    (hcont : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc b x))
    (hW : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ∀ t ∈ Ico b x, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ∀ t ∈ Ico b x, HasDerivWithinAt A (c t) (Ici t) t)
    (hWtail : Tendsto W atBot (𝓝 0))
    (hAtail : Tendsto (fun x => Real.exp (A x)) atBot (𝓝 L)) :
    ∀ x : ℝ, x ≤ a → W x = 0 := by
  intro x hxa
  exact abel_left_tail_zero_of_integratingFactor
    (W := W) (A := A) (c := c) (a := x) (L := L)
    (fun b hbx => hcont b x hbx hxa)
    (fun b hbx => hW b x hbx hxa)
    (fun b hbx => hA b x hbx hxa)
    hWtail hAtail

/-- L6 left outer interval, stated using convergence of the primitive `A`
itself rather than convergence of `exp A`. -/
theorem abel_left_outer_zero_of_integratingFactor_of_tendsto_primitive
    {W A c : ℝ → ℝ} {a L : ℝ}
    (hcont : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc b x))
    (hW : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ∀ t ∈ Ico b x, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ∀ t ∈ Ico b x, HasDerivWithinAt A (c t) (Ici t) t)
    (hWtail : Tendsto W atBot (𝓝 0))
    (hAtail : Tendsto A atBot (𝓝 L)) :
    ∀ x : ℝ, x ≤ a → W x = 0 :=
  abel_left_outer_zero_of_integratingFactor hcont hW hA hWtail
    (exp_comp_tendsto_nhds hAtail)

/-! ## L8: upward-crossing flanks -/

/-- **L8 left flank.**  Suppose `x < a` lies to the left of an upward crossing.
If Abel/integrating-factor constancy holds on every `[x,b]` with `b < a`,
`W` is bounded as one approaches `a` from the left, and `exp A → 0` from the
left, then `W x = 0`.

This packages the boundedness-vs-blow-up contradiction: if the constant
`W x * exp(A x)` were nonzero, then `W` would have to blow up as `exp(A)` tends
to zero. -/
theorem abel_left_flank_zero_of_integratingFactor
    {W A c : ℝ → ℝ} {x a : ℝ}
    (hxa : x < a)
    (hcont : ∀ b : ℝ, x ≤ b → b < a →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x b))
    (hW : ∀ b : ℝ, x ≤ b → b < a →
      ∀ t ∈ Ico x b, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ b : ℝ, x ≤ b → b < a →
      ∀ t ∈ Ico x b, HasDerivWithinAt A (c t) (Ici t) t)
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[<] a) (norm ∘ W))
    (hEzero : Tendsto (fun t => Real.exp (A t)) (𝓝[<] a) (𝓝 0)) :
    W x = 0 := by
  have hconst : ∀ᶠ b in 𝓝[<] a,
      W b * Real.exp (A b) = W x * Real.exp (A x) := by
    filter_upwards [Ioo_mem_nhdsLT hxa] with b hb
    exact abel_integratingFactor_const_Icc
      (W := W) (A := A) (c := c)
      (hcont b hb.1.le hb.2) (hW b hb.1.le hb.2) (hA b hb.1.le hb.2)
      b ⟨hb.1.le, le_rfl⟩
  have hbase : W x * Real.exp (A x) = 0 :=
    bounded_integratingFactor_const_zero_of_factor_tendsto_zero
      hconst hWbounded hEzero
  exact (mul_eq_zero.mp hbase).resolve_right (Real.exp_ne_zero _)

/-- **L8 right flank.**  Symmetric crossing-flank theorem approaching an upward
crossing from the right. -/
theorem abel_right_flank_zero_of_integratingFactor
    {W A c : ℝ → ℝ} {a x : ℝ}
    (hax : a < x)
    (hcont : ∀ b : ℝ, a < b → b ≤ x →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc b x))
    (hW : ∀ b : ℝ, a < b → b ≤ x →
      ∀ t ∈ Ico b x, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ b : ℝ, a < b → b ≤ x →
      ∀ t ∈ Ico b x, HasDerivWithinAt A (c t) (Ici t) t)
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[>] a) (norm ∘ W))
    (hEzero : Tendsto (fun t => Real.exp (A t)) (𝓝[>] a) (𝓝 0)) :
    W x = 0 := by
  have hconst : ∀ᶠ b in 𝓝[>] a,
      W b * Real.exp (A b) = W x * Real.exp (A x) := by
    filter_upwards [Ioo_mem_nhdsGT hax] with b hb
    exact (abel_integratingFactor_const_Icc
      (W := W) (A := A) (c := c)
      (hcont b hb.1 hb.2.le) (hW b hb.1 hb.2.le) (hA b hb.1 hb.2.le)
      x ⟨hb.2.le, le_rfl⟩).symm
  have hbase : W x * Real.exp (A x) = 0 :=
    bounded_integratingFactor_const_zero_of_factor_tendsto_zero
      hconst hWbounded hEzero
  exact (mul_eq_zero.mp hbase).resolve_right (Real.exp_ne_zero _)

/-- If the right side of an upward crossing has the L8 integrating-factor
divergence data, then `W` vanishes on the whole right flank up to the next
breakpoint. -/
theorem abel_right_interval_zero_of_upwardCrossing
    {W A c : ℝ → ℝ} {a b : ℝ}
    (_hab : a < b)
    (hcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x y))
    (hW : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt A (c t) (Ici t) t)
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[>] a) (norm ∘ W))
    (hEzero : Tendsto (fun t => Real.exp (A t)) (𝓝[>] a) (𝓝 0)) :
    ∀ x : ℝ, a < x → x < b → W x = 0 := by
  intro x hax hxb
  exact abel_right_flank_zero_of_integratingFactor
    (W := W) (A := A) (c := c) hax
    (fun y hay hyx => hcont y x hay hyx hxb)
    (fun y hay hyx => hW y x hay hyx hxb)
    (fun y hay hyx => hA y x hay hyx hxb)
    hWbounded hEzero

/-- L8 right flank from primitive divergence `A → -∞` at the upward crossing. -/
theorem abel_right_interval_zero_of_upwardCrossing_of_tendsto_atBot
    {W A c : ℝ → ℝ} {a b : ℝ}
    (hab : a < b)
    (hcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x y))
    (hW : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt A (c t) (Ici t) t)
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[>] a) (norm ∘ W))
    (hAdiv : Tendsto A (𝓝[>] a) atBot) :
    ∀ x : ℝ, a < x → x < b → W x = 0 :=
  abel_right_interval_zero_of_upwardCrossing hab hcont hW hA hWbounded
    (exp_comp_tendsto_zero_of_tendsto_atBot hAdiv)

/-- If the left side of an upward crossing has the L8 integrating-factor
divergence data, then `W` vanishes on the whole left flank down to the previous
breakpoint. -/
theorem abel_left_interval_zero_of_upwardCrossing
    {W A c : ℝ → ℝ} {a b : ℝ}
    (_hab : a < b)
    (hcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x y))
    (hW : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt A (c t) (Ici t) t)
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[<] b) (norm ∘ W))
    (hEzero : Tendsto (fun t => Real.exp (A t)) (𝓝[<] b) (𝓝 0)) :
    ∀ x : ℝ, a < x → x < b → W x = 0 := by
  intro x hax hxb
  exact abel_left_flank_zero_of_integratingFactor
    (W := W) (A := A) (c := c) hxb
    (fun y hxy hyb => hcont x y hax hxy hyb)
    (fun y hxy hyb => hW x y hax hxy hyb)
    (fun y hxy hyb => hA x y hax hxy hyb)
    hWbounded hEzero

/-- L8 left flank from primitive divergence `A → -∞` at the upward crossing. -/
theorem abel_left_interval_zero_of_upwardCrossing_of_tendsto_atBot
    {W A c : ℝ → ℝ} {a b : ℝ}
    (hab : a < b)
    (hcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x y))
    (hW : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt A (c t) (Ici t) t)
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[<] b) (norm ∘ W))
    (hAdiv : Tendsto A (𝓝[<] b) atBot) :
    ∀ x : ℝ, a < x → x < b → W x = 0 :=
  abel_left_interval_zero_of_upwardCrossing hab hcont hW hA hWbounded
    (exp_comp_tendsto_zero_of_tendsto_atBot hAdiv)

/-! ## L9: finite breakpoint gluing -/

/-- If a continuous real-valued function vanishes on a dense set, it vanishes
everywhere. -/
theorem continuous_eq_zero_of_dense_zeroSet
    {W : ℝ → ℝ}
    (hcont : Continuous W)
    (hdense : Dense {x : ℝ | W x = 0}) :
    ∀ x : ℝ, W x = 0 := by
  have hclosed : IsClosed {x : ℝ | W x = 0} :=
    isClosed_eq hcont continuous_const
  intro x
  have hx : x ∈ closure {x : ℝ | W x = 0} := by
    rw [hdense.closure_eq]
    trivial
  exact hclosed.closure_subset hx

/-- **L9 continuity gluing.**  If a continuous `W` vanishes away from a finite
set of breakpoints, then it vanishes everywhere.  This is the formal gluing
step after the outer and crossing-flank arguments have killed every open sign
interval. -/
theorem continuous_eq_zero_of_zero_off_finset
    {W : ℝ → ℝ} (breaks : Finset ℝ)
    (hcont : Continuous W)
    (hzero : ∀ x : ℝ, x ∉ (breaks : Set ℝ) → W x = 0) :
    ∀ x : ℝ, W x = 0 := by
  have hfinite : (breaks : Set ℝ).Finite := breaks.finite_toSet
  have hdense_compl : Dense ((breaks : Set ℝ)ᶜ) :=
    hfinite.countable.dense_compl ℝ
  have hdense_zero : Dense {x : ℝ | W x = 0} :=
    hdense_compl.mono fun x hx => hzero x hx
  exact continuous_eq_zero_of_dense_zeroSet hcont hdense_zero

end DriftingIdentifiability
