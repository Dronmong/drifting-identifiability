import DriftingIdentifiability.LaplaceACAbel
import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
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

/-! ## Primitive-free L6 sign packages -/

/-- On a finite interval, an Abel solution with nonpositive coefficient has
nondecreasing square.  This is the primitive-free form of the right outer L6
argument: if `W' = -cW` and `c ≤ 0`, then `(W^2)' ≥ 0`. -/
theorem abel_square_left_le_right_of_nonpos_coeff
    {W c : ℝ → ℝ} {x y : ℝ}
    (hxy : x ≤ y)
    (hcont : ContinuousOn W (Icc x y))
    (hW : ∀ t ∈ Ico x y, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hc : ∀ t ∈ Ico x y, c t ≤ 0) :
    W x ^ 2 ≤ W y ^ 2 := by
  let F : ℝ → ℝ := W * W
  have hFcont : ContinuousOn F (Icc x y) := by
    simpa [F] using hcont.mul hcont
  have hFderiv : ∀ t ∈ Ico x y,
      HasDerivWithinAt F
        ((-(c t) * W t) * W t + W t * (-(c t) * W t)) (Ici t) t := by
    intro t ht
    simpa [F] using (hW t ht).mul (hW t ht)
  have hconst_deriv : ∀ t ∈ Ico x y,
      HasDerivWithinAt (fun _ : ℝ => W x ^ 2) 0 (Ici t) t := by
    intro t _; exact (hasDerivAt_const t (W x ^ 2)).hasDerivWithinAt
  have hconst_cont : ContinuousOn (fun _ : ℝ => W x ^ 2) (Icc x y) :=
    continuousOn_const
  have hbase : (fun _ : ℝ => W x ^ 2) x ≤ F x := by
    simp [F, pow_two]
  have hbound : ∀ t ∈ Ico x y,
      (fun _ : ℝ => 0) t ≤
        (fun t : ℝ => (-(c t) * W t) * W t + W t * (-(c t) * W t)) t := by
    intro t ht
    have hcneg : 0 ≤ -c t := neg_nonneg.mpr (hc t ht)
    have hs : 0 ≤ W t * W t := mul_self_nonneg (W t)
    have hnonneg : 0 ≤ 2 * (-c t) * (W t * W t) :=
      mul_nonneg (mul_nonneg (by norm_num) hcneg) hs
    calc
      0 ≤ 2 * (-c t) * (W t * W t) := hnonneg
      _ = (-(c t) * W t) * W t + W t * (-(c t) * W t) := by ring
  have hmono : ∀ z, z ∈ Icc x y → (fun _ : ℝ => W x ^ 2) z ≤ F z :=
    image_le_of_deriv_right_le_deriv_boundary
      (a := x) (b := y)
      (f := fun _ : ℝ => W x ^ 2) (f' := fun _ : ℝ => 0)
      (B := F)
      (B' := fun t : ℝ => (-(c t) * W t) * W t + W t * (-(c t) * W t))
      hconst_cont hconst_deriv hbase hFcont hFderiv hbound
  simpa [F, pow_two] using hmono y ⟨hxy, le_rfl⟩

/-- On a finite interval, an Abel solution with nonnegative coefficient has
nonincreasing square.  This is the primitive-free form of the left outer L6
argument. -/
theorem abel_square_right_le_left_of_nonneg_coeff
    {W c : ℝ → ℝ} {x y : ℝ}
    (hxy : x ≤ y)
    (hcont : ContinuousOn W (Icc x y))
    (hW : ∀ t ∈ Ico x y, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hc : ∀ t ∈ Ico x y, 0 ≤ c t) :
    W y ^ 2 ≤ W x ^ 2 := by
  let F : ℝ → ℝ := W * W
  have hFcont : ContinuousOn F (Icc x y) := by
    simpa [F] using hcont.mul hcont
  have hFderiv : ∀ t ∈ Ico x y,
      HasDerivWithinAt F
        ((-(c t) * W t) * W t + W t * (-(c t) * W t)) (Ici t) t := by
    intro t ht
    simpa [F] using (hW t ht).mul (hW t ht)
  have hconst_deriv : ∀ t ∈ Ico x y,
      HasDerivWithinAt (fun _ : ℝ => W x ^ 2) 0 (Ici t) t := by
    intro t _; exact (hasDerivAt_const t (W x ^ 2)).hasDerivWithinAt
  have hconst_cont : ContinuousOn (fun _ : ℝ => W x ^ 2) (Icc x y) :=
    continuousOn_const
  have hbase : F x ≤ (fun _ : ℝ => W x ^ 2) x := by
    simp [F, pow_two]
  have hbound : ∀ t ∈ Ico x y,
      (fun t : ℝ => (-(c t) * W t) * W t + W t * (-(c t) * W t)) t ≤
        (fun _ : ℝ => 0) t := by
    intro t ht
    have hcpos : 0 ≤ c t := hc t ht
    have hs : 0 ≤ W t * W t := mul_self_nonneg (W t)
    have hnonneg : 0 ≤ 2 * c t * (W t * W t) :=
      mul_nonneg (mul_nonneg (by norm_num) hcpos) hs
    calc
      (-(c t) * W t) * W t + W t * (-(c t) * W t)
          = -(2 * c t * (W t * W t)) := by ring
      _ ≤ 0 := neg_nonpos.mpr hnonneg
  have hmono : ∀ z, z ∈ Icc x y → F z ≤ (fun _ : ℝ => W x ^ 2) z :=
    image_le_of_deriv_right_le_deriv_boundary
      (a := x) (b := y)
      (f := F)
      (f' := fun t : ℝ => (-(c t) * W t) * W t + W t * (-(c t) * W t))
      (B := fun _ : ℝ => W x ^ 2) (B' := fun _ : ℝ => 0)
      hFcont hFderiv hbase hconst_cont hconst_deriv hbound
  simpa [F, pow_two] using hmono y ⟨hxy, le_rfl⟩

theorem sq_eq_zero_of_nonneg_of_forall_le_of_tendsto_zero_atTop
    {W : ℝ → ℝ} {x : ℝ}
    (hmono : ∀ y : ℝ, x ≤ y → W x ^ 2 ≤ W y ^ 2)
    (hWtail : Tendsto W atTop (𝓝 0)) :
    W x = 0 := by
  by_contra hx
  have hxpos : 0 < W x ^ 2 := sq_pos_of_ne_zero hx
  have hFtail : Tendsto (fun y : ℝ => W y ^ 2) atTop (𝓝 0) := by
    simpa [pow_two] using hWtail.mul hWtail
  have hsmall : ∀ᶠ y in atTop, W y ^ 2 < W x ^ 2 :=
    hFtail.eventually (Iio_mem_nhds hxpos)
  have hlarge : ∀ᶠ y in atTop, x ≤ y := eventually_ge_atTop x
  rcases (hsmall.and hlarge).exists with ⟨y, hy_small, hy_large⟩
  exact not_lt_of_ge (hmono y hy_large) hy_small

theorem sq_eq_zero_of_nonneg_of_forall_le_of_tendsto_zero_atBot
    {W : ℝ → ℝ} {x : ℝ}
    (hmono : ∀ y : ℝ, y ≤ x → W x ^ 2 ≤ W y ^ 2)
    (hWtail : Tendsto W atBot (𝓝 0)) :
    W x = 0 := by
  by_contra hx
  have hxpos : 0 < W x ^ 2 := sq_pos_of_ne_zero hx
  have hFtail : Tendsto (fun y : ℝ => W y ^ 2) atBot (𝓝 0) := by
    simpa [pow_two] using hWtail.mul hWtail
  have hsmall : ∀ᶠ y in atBot, W y ^ 2 < W x ^ 2 :=
    hFtail.eventually (Iio_mem_nhds hxpos)
  have hlarge : ∀ᶠ y in atBot, y ≤ x := eventually_le_atBot x
  rcases (hsmall.and hlarge).exists with ⟨y, hy_small, hy_large⟩
  exact not_lt_of_ge (hmono y hy_large) hy_small

/-- **L6 right outer ray, primitive-free sign form.**

If `W' = -cW`, `c ≤ 0` on the right outer ray, and `W → 0` at `+∞`, then
`W` vanishes throughout the ray.  This is the concrete outer-ray mechanism for
the Laplace AC route: on the right of the last zero, `m = μ - x < 0` while
`μ' ≥ 0`, hence `c = 2μ'/m ≤ 0`. -/
theorem abel_right_outer_zero_of_nonpos_coeff
    {W c : ℝ → ℝ} {a : ℝ}
    (hcont : ∀ x b : ℝ, a ≤ x → x ≤ b → ContinuousOn W (Icc x b))
    (hW : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ∀ t ∈ Ico x b, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hc : ∀ t : ℝ, a ≤ t → c t ≤ 0)
    (hWtail : Tendsto W atTop (𝓝 0)) :
    ∀ x : ℝ, a ≤ x → W x = 0 := by
  intro x hax
  refine sq_eq_zero_of_nonneg_of_forall_le_of_tendsto_zero_atTop ?_ hWtail
  intro y hxy
  exact abel_square_left_le_right_of_nonpos_coeff hxy
    (hcont x y hax hxy) (hW x y hax hxy)
    (fun t ht => hc t (le_trans hax ht.1))

/-- **L6 left outer ray, primitive-free sign form.**

If `W' = -cW`, `0 ≤ c` on the left outer ray, and `W → 0` at `-∞`, then
`W` vanishes throughout the ray.  In the Laplace AC route this corresponds to
the left of the first zero, where `m = μ - x > 0` and `μ' ≥ 0`. -/
theorem abel_left_outer_zero_of_nonneg_coeff
    {W c : ℝ → ℝ} {a : ℝ}
    (hcont : ∀ b x : ℝ, b ≤ x → x ≤ a → ContinuousOn W (Icc b x))
    (hW : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ∀ t ∈ Ico b x, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hc : ∀ t : ℝ, t ≤ a → 0 ≤ c t)
    (hWtail : Tendsto W atBot (𝓝 0)) :
    ∀ x : ℝ, x ≤ a → W x = 0 := by
  intro x hxa
  refine sq_eq_zero_of_nonneg_of_forall_le_of_tendsto_zero_atBot ?_ hWtail
  intro y hyx
  exact abel_square_right_le_left_of_nonneg_coeff hyx
    (hcont y x hyx hxa) (hW y x hyx hxa)
    (fun t ht => hc t (le_trans ht.2.le hxa))

/-- **L6 right outer ray for the actual Laplace Abel coefficient.**

This is the concrete sign-specialized wrapper used by the a.c. converse.  On a
right outer ray, monotonicity gives `0 ≤ μ'` and the outer sign geometry gives
`m < 0`; hence the Abel coefficient `(2 * μ') / m` is nonpositive, so the
primitive-free right outer package kills `W` from the tail limit `W → 0`. -/
theorem abel_right_outer_zero_of_muDeriv_nonneg_of_m_neg
    {W muDeriv m : ℝ → ℝ} {a : ℝ}
    (hcont : ∀ x b : ℝ, a ≤ x → x ≤ b → ContinuousOn W (Icc x b))
    (hW : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ∀ t ∈ Ico x b,
        HasDerivWithinAt W (-(((2 : ℝ) * muDeriv t) / m t) * W t) (Ici t) t)
    (hmu : ∀ t : ℝ, a ≤ t → 0 ≤ muDeriv t)
    (hm : ∀ t : ℝ, a ≤ t → m t < 0)
    (hWtail : Tendsto W atTop (𝓝 0)) :
    ∀ x : ℝ, a ≤ x → W x = 0 := by
  refine abel_right_outer_zero_of_nonpos_coeff
    (W := W) (c := fun t : ℝ => ((2 : ℝ) * muDeriv t) / m t) (a := a)
    hcont ?_ ?_ hWtail
  · simpa using hW
  · intro t ht
    exact div_nonpos_of_nonneg_of_nonpos
      (mul_nonneg (by norm_num) (hmu t ht)) (le_of_lt (hm t ht))

/-- **L6 left outer ray for the actual Laplace Abel coefficient.**

On a left outer ray, monotonicity gives `0 ≤ μ'` and the outer sign geometry
gives `0 < m`; hence `(2 * μ') / m` is nonnegative, so the primitive-free left
outer package kills `W` from the tail limit `W → 0` at `-∞`. -/
theorem abel_left_outer_zero_of_muDeriv_nonneg_of_m_pos
    {W muDeriv m : ℝ → ℝ} {a : ℝ}
    (hcont : ∀ b x : ℝ, b ≤ x → x ≤ a → ContinuousOn W (Icc b x))
    (hW : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ∀ t ∈ Ico b x,
        HasDerivWithinAt W (-(((2 : ℝ) * muDeriv t) / m t) * W t) (Ici t) t)
    (hmu : ∀ t : ℝ, t ≤ a → 0 ≤ muDeriv t)
    (hm : ∀ t : ℝ, t ≤ a → 0 < m t)
    (hWtail : Tendsto W atBot (𝓝 0)) :
    ∀ x : ℝ, x ≤ a → W x = 0 := by
  refine abel_left_outer_zero_of_nonneg_coeff
    (W := W) (c := fun t : ℝ => ((2 : ℝ) * muDeriv t) / m t) (a := a)
    hcont ?_ ?_ hWtail
  · simpa using hW
  · intro t ht
    exact div_nonneg
      (mul_nonneg (by norm_num) (hmu t ht)) (le_of_lt (hm t ht))

/-! ## L8 divergence packages -/

/-- Order-squeeze into `atBot`.

This is the tiny topological endpoint behind the L8 upstream packages: if an
upper barrier `B` tends to `-∞` and `A ≤ B` eventually, then `A` also tends to
`-∞`. -/
theorem tendsto_atBot_of_eventually_le_of_tendsto_atBot
    {α : Type*} {l : Filter α} {A B : α → ℝ}
    (hB : Tendsto B l atBot)
    (hAB : ∀ᶠ x in l, A x ≤ B x) :
    Tendsto A l atBot := by
  rw [tendsto_atBot]
  intro b
  filter_upwards [hB.eventually_le_atBot b, hAB] with x hBx hAx
  exact le_trans hAx hBx

/-! ## Tail-BV packages for L6 -/

/-- A right-tail Cauchy criterion packaged for L6.

If the oscillation of a primitive `A` on every tail interval `[x,y]` is bounded
by a tail gauge `G x` and `G x → 0`, then `A` has a finite limit at `+∞`.
This is the deterministic endpoint needed after the AC/BV estimates construct
the integrating-factor primitive. -/
theorem cauchy_map_atTop_of_tail_norm_sub_le
    {A G : ℝ → ℝ} {a : ℝ}
    (hG : Tendsto G atTop (𝓝 0))
    (hbound : ∀ x y : ℝ, a ≤ x → x ≤ y → ‖A y - A x‖ ≤ G x) :
    Cauchy (map A atTop) := by
  rw [Metric.cauchy_iff]
  constructor
  · infer_instance
  · intro ε hε
    have hsmallDist : ∀ᶠ x in atTop, dist (G x) 0 < ε :=
      (Metric.tendsto_nhds.mp hG) ε hε
    have hsmall : ∀ᶠ x in atTop, G x < ε := by
      filter_upwards [hsmallDist] with x hx
      exact lt_of_le_of_lt (le_abs_self (G x)) (by simpa [Real.dist_eq] using hx)
    rcases eventually_atTop.1 hsmall with ⟨M0, hM0⟩
    let M : ℝ := max a M0
    refine ⟨A '' Ici M, ?_, ?_⟩
    · change {x | A x ∈ A '' Ici M} ∈ atTop
      exact mem_of_superset (Ici_mem_atTop M) fun x hx => ⟨x, hx, rfl⟩
    · intro u hu v hv
      rcases hu with ⟨x, hxM, rfl⟩
      rcases hv with ⟨y, hyM, rfl⟩
      by_cases hxy : x ≤ y
      · have hax : a ≤ x := le_trans (le_max_left a M0) hxM
        have hGx : G x < ε := hM0 x (le_trans (le_max_right a M0) hxM)
        calc
          dist (A x) (A y) = ‖A y - A x‖ := by
            rw [dist_eq_norm, norm_sub_rev]
          _ ≤ G x := hbound x y hax hxy
          _ < ε := hGx
      · have hyx : y ≤ x := le_of_not_ge hxy
        have hay : a ≤ y := le_trans (le_max_left a M0) hyM
        have hGy : G y < ε := hM0 y (le_trans (le_max_right a M0) hyM)
        calc
          dist (A x) (A y) = ‖A x - A y‖ := by
            rw [dist_eq_norm]
          _ ≤ G y := hbound y x hay hyx
          _ < ε := hGy

/-- Right-tail finite-limit constructor from the L6 tail-BV estimate. -/
theorem exists_tendsto_atTop_of_tail_norm_sub_le
    {A G : ℝ → ℝ} {a : ℝ}
    (hG : Tendsto G atTop (𝓝 0))
    (hbound : ∀ x y : ℝ, a ≤ x → x ≤ y → ‖A y - A x‖ ≤ G x) :
    ∃ L : ℝ, Tendsto A atTop (𝓝 L) :=
  cauchy_map_iff_exists_tendsto.mp
    (cauchy_map_atTop_of_tail_norm_sub_le hG hbound)

/-- Left-tail finite-limit constructor from the symmetric L6 tail-BV estimate. -/
theorem exists_tendsto_atBot_of_tail_norm_sub_le
    {A G : ℝ → ℝ} {a : ℝ}
    (hG : Tendsto G atBot (𝓝 0))
    (hbound : ∀ x y : ℝ, x ≤ y → y ≤ a → ‖A x - A y‖ ≤ G y) :
    ∃ L : ℝ, Tendsto A atBot (𝓝 L) := by
  have hGtop : Tendsto (fun x : ℝ => G (-x)) atTop (𝓝 0) :=
    hG.comp tendsto_neg_atTop_atBot
  have htop :
      ∃ L : ℝ, Tendsto (fun x : ℝ => A (-x)) atTop (𝓝 L) := by
    refine exists_tendsto_atTop_of_tail_norm_sub_le
      (A := fun x : ℝ => A (-x)) (G := fun x : ℝ => G (-x)) (a := -a)
      hGtop ?_
    intro x y hx hxy
    have hyx : -y ≤ -x := neg_le_neg hxy
    have hxa : -x ≤ a := by
      exact neg_le.mp hx
    simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using
      hbound (-y) (-x) hyx hxa
  rcases htop with ⟨L, hL⟩
  refine ⟨L, ?_⟩
  simpa [Function.comp_def] using hL.comp tendsto_neg_atBot_atTop

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

/-- L6 right outer interval from a tail-BV estimate for the primitive.

This is the upstream packaging used by the AC converse: once the analytic work
has produced a primitive `A' = c` whose tail oscillation is bounded by a gauge
`G x → 0`, the finite primitive limit required by
`abel_right_outer_zero_of_integratingFactor_of_tendsto_primitive` is constructed
internally. -/
theorem abel_right_outer_zero_of_tail_bvPrimitive
    {W A c G : ℝ → ℝ} {a : ℝ}
    (hcont : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x b))
    (hW : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ∀ t ∈ Ico x b, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x b : ℝ, a ≤ x → x ≤ b →
      ∀ t ∈ Ico x b, HasDerivWithinAt A (c t) (Ici t) t)
    (hWtail : Tendsto W atTop (𝓝 0))
    (hG : Tendsto G atTop (𝓝 0))
    (hBV : ∀ x y : ℝ, a ≤ x → x ≤ y → ‖A y - A x‖ ≤ G x) :
    ∀ x : ℝ, a ≤ x → W x = 0 := by
  rcases exists_tendsto_atTop_of_tail_norm_sub_le hG hBV with ⟨L, hAtail⟩
  exact abel_right_outer_zero_of_integratingFactor_of_tendsto_primitive
    (W := W) (A := A) (c := c) (a := a) (L := L)
    hcont hW hA hWtail hAtail

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

/-- L6 left outer interval from a symmetric tail-BV estimate for the primitive. -/
theorem abel_left_outer_zero_of_tail_bvPrimitive
    {W A c G : ℝ → ℝ} {a : ℝ}
    (hcont : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc b x))
    (hW : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ∀ t ∈ Ico b x, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ b x : ℝ, b ≤ x → x ≤ a →
      ∀ t ∈ Ico b x, HasDerivWithinAt A (c t) (Ici t) t)
    (hWtail : Tendsto W atBot (𝓝 0))
    (hG : Tendsto G atBot (𝓝 0))
    (hBV : ∀ x y : ℝ, x ≤ y → y ≤ a → ‖A x - A y‖ ≤ G y) :
    ∀ x : ℝ, x ≤ a → W x = 0 := by
  rcases exists_tendsto_atBot_of_tail_norm_sub_le hG hBV with ⟨L, hAtail⟩
  exact abel_left_outer_zero_of_integratingFactor_of_tendsto_primitive
    (W := W) (A := A) (c := c) (a := a) (L := L)
    hcont hW hA hWtail hAtail

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

/-- L8 left flank from an upper barrier for the primitive.

This is the upstream divergence package: instead of requiring callers to prove
`A → -∞` directly, it is enough to give an eventually larger barrier `B` with
`B → -∞`.  The crossing-specific analysis can supply such a barrier later
(for instance a logarithmic one). -/
theorem abel_left_flank_zero_of_integratingFactor_of_atBotBarrier
    {W A B c : ℝ → ℝ} {x a : ℝ}
    (hxa : x < a)
    (hcont : ∀ b : ℝ, x ≤ b → b < a →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x b))
    (hW : ∀ b : ℝ, x ≤ b → b < a →
      ∀ t ∈ Ico x b, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ b : ℝ, x ≤ b → b < a →
      ∀ t ∈ Ico x b, HasDerivWithinAt A (c t) (Ici t) t)
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[<] a) (norm ∘ W))
    (hBdiv : Tendsto B (𝓝[<] a) atBot)
    (hAupper : ∀ᶠ t in 𝓝[<] a, A t ≤ B t) :
    W x = 0 :=
  abel_left_flank_zero_of_integratingFactor hxa hcont hW hA hWbounded
    (exp_comp_tendsto_zero_of_tendsto_atBot
      (tendsto_atBot_of_eventually_le_of_tendsto_atBot hBdiv hAupper))

/-- L8 right flank from an upper barrier for the primitive. -/
theorem abel_right_flank_zero_of_integratingFactor_of_atBotBarrier
    {W A B c : ℝ → ℝ} {a x : ℝ}
    (hax : a < x)
    (hcont : ∀ b : ℝ, a < b → b ≤ x →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc b x))
    (hW : ∀ b : ℝ, a < b → b ≤ x →
      ∀ t ∈ Ico b x, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ b : ℝ, a < b → b ≤ x →
      ∀ t ∈ Ico b x, HasDerivWithinAt A (c t) (Ici t) t)
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[>] a) (norm ∘ W))
    (hBdiv : Tendsto B (𝓝[>] a) atBot)
    (hAupper : ∀ᶠ t in 𝓝[>] a, A t ≤ B t) :
    W x = 0 :=
  abel_right_flank_zero_of_integratingFactor hax hcont hW hA hWbounded
    (exp_comp_tendsto_zero_of_tendsto_atBot
      (tendsto_atBot_of_eventually_le_of_tendsto_atBot hBdiv hAupper))

/-! ### Concrete logarithmic barriers for L8 -/

/-- The distance to a right-hand endpoint tends to `0` from above. -/
theorem tendsto_sub_const_nhdsGT_zero {a : ℝ} :
    Tendsto (fun x : ℝ => x - a) (𝓝[>] a) (𝓝[>] 0) := by
  refine tendsto_nhdsWithin_iff.mpr ⟨?_, ?_⟩
  · have hnhds : Tendsto (fun x : ℝ => x - a) (𝓝[>] a) (𝓝 (a - a)) :=
      (tendsto_id.sub tendsto_const_nhds).mono_left nhdsWithin_le_nhds
    simpa using hnhds
  · filter_upwards [self_mem_nhdsWithin] with x hx
    exact sub_pos.mpr (show a < x from hx)

/-- The distance to a left-hand endpoint tends to `0` from above. -/
theorem tendsto_const_sub_nhdsLT_zero {a : ℝ} :
    Tendsto (fun x : ℝ => a - x) (𝓝[<] a) (𝓝[>] 0) := by
  refine tendsto_nhdsWithin_iff.mpr ⟨?_, ?_⟩
  · have hnhds : Tendsto (fun x : ℝ => a - x) (𝓝[<] a) (𝓝 (a - a)) :=
      (tendsto_const_nhds.sub tendsto_id).mono_left nhdsWithin_le_nhds
    simpa using hnhds
  · filter_upwards [self_mem_nhdsWithin] with x hx
    exact sub_pos.mpr (show x < a from hx)

/-- Logarithmic right-hand barrier tends to `-∞` at the crossing. -/
theorem tendsto_log_sub_const_nhdsGT_atBot {a C γ : ℝ} (hγ : 0 < γ) :
    Tendsto (fun x : ℝ => C + γ * Real.log (x - a)) (𝓝[>] a) atBot := by
  exact tendsto_atBot_add_const_left _ _
    ((Real.tendsto_log_nhdsGT_zero.comp tendsto_sub_const_nhdsGT_zero).const_mul_atBot hγ)

/-- Logarithmic left-hand barrier tends to `-∞` at the crossing. -/
theorem tendsto_log_const_sub_nhdsLT_atBot {a C γ : ℝ} (hγ : 0 < γ) :
    Tendsto (fun x : ℝ => C + γ * Real.log (a - x)) (𝓝[<] a) atBot := by
  exact tendsto_atBot_add_const_left _ _
    ((Real.tendsto_log_nhdsGT_zero.comp tendsto_const_sub_nhdsLT_zero).const_mul_atBot hγ)

theorem hasDerivWithinAt_log_sub_const
    {a x γ : ℝ} (hax : a < x) :
    HasDerivWithinAt (fun u : ℝ => γ * Real.log (u - a))
      (γ * (1 / (x - a))) (Ici x) x := by
  have hsub : HasDerivAt (fun u : ℝ => u - a) 1 x := by
    simpa using (hasDerivAt_id x).sub_const a
  have hlog : HasDerivAt (fun u : ℝ => Real.log (u - a)) ((1 / (x - a)) * 1) x := by
    simpa [one_div, Function.comp_def] using
      (Real.hasDerivAt_log (sub_ne_zero.mpr (ne_of_gt hax))).comp x hsub
  have hlog' : HasDerivAt (fun u : ℝ => Real.log (u - a)) (1 / (x - a)) x := by
    simpa using hlog
  simpa [mul_comm, mul_left_comm, mul_assoc] using
    (hlog'.const_mul γ).hasDerivWithinAt

theorem hasDerivWithinAt_log_const_sub
    {a x γ : ℝ} (hxa : x < a) :
    HasDerivWithinAt (fun u : ℝ => γ * Real.log (a - u))
      (-(γ * (1 / (a - x)))) (Ici x) x := by
  have hsub : HasDerivAt (fun u : ℝ => a - u) (-1) x := by
    change HasDerivAt ((fun _ : ℝ => a) - id) (-1) x
    simpa using (hasDerivAt_const x a).sub (hasDerivAt_id x)
  have hlog : HasDerivAt (fun u : ℝ => Real.log (a - u)) ((1 / (a - x)) * (-1)) x := by
    simpa [one_div, Function.comp_def] using
      (Real.hasDerivAt_log (sub_ne_zero.mpr (ne_of_gt hxa))).comp x hsub
  have hlog' : HasDerivAt (fun u : ℝ => Real.log (a - u)) (-(1 / (a - x))) x := by
    simpa using hlog
  simpa [mul_comm, mul_left_comm, mul_assoc] using
    (hlog'.const_mul γ).hasDerivWithinAt

/-- Right-side logarithmic primitive bound.

If `A' = c` and `c` dominates `γ/(t-a)` on `(a,r)`, then near `a` from the
right, `A` is bounded above by a logarithmic barrier tending to `-∞`. -/
theorem eventually_primitive_le_log_barrier_right
    {A c : ℝ → ℝ} {a r γ : ℝ}
    (har : a < r) (_hγ : 0 < γ)
    (hAcont : ∀ x : ℝ, a < x → x ≤ r → ContinuousOn A (Icc x r))
    (hA : ∀ x : ℝ, a < x → x ≤ r →
      ∀ t ∈ Ico x r, HasDerivWithinAt A (c t) (Ici t) t)
    (hc : ∀ t : ℝ, a < t → t < r → γ / (t - a) ≤ c t) :
    ∀ᶠ x in 𝓝[>] a,
      A x ≤ A r + γ * (Real.log (x - a) - Real.log (r - a)) := by
  filter_upwards [Ioo_mem_nhdsGT har] with x hx
  have hxr : x ≤ r := hx.2.le
  let B : ℝ → ℝ := fun s => A x + γ * Real.log (s - a) - γ * Real.log (x - a)
  have hBcont : ContinuousOn B (Icc x r) := by
    have hsubcont : ContinuousOn (fun s : ℝ => s - a) (Icc x r) := by
      exact continuousOn_id.sub continuousOn_const
    have hne : ∀ s ∈ Icc x r, s - a ≠ 0 := by
      intro s hs
      exact sub_ne_zero.mpr (ne_of_gt (lt_of_lt_of_le hx.1 hs.1))
    have hlog : ContinuousOn (fun s : ℝ => Real.log (s - a)) (Icc x r) :=
      hsubcont.log hne
    exact (continuousOn_const.add (continuousOn_const.mul hlog)).sub continuousOn_const
  have hBderiv : ∀ t ∈ Ico x r,
      HasDerivWithinAt B (γ * (1 / (t - a))) (Ici t) t := by
    intro t ht
    have hat : a < t := lt_of_lt_of_le hx.1 ht.1
    have hderiv := hasDerivWithinAt_log_sub_const (a := a) (x := t) (γ := γ) hat
    dsimp only [B]
    change HasDerivWithinAt
      (fun u : ℝ => A x + γ * Real.log (u - a) - γ * Real.log (x - a))
      (γ * (1 / (t - a))) (Ici t) t
    exact (hderiv.const_add (A x)).sub_const (γ * Real.log (x - a))
  have hbase : B x ≤ A x := by simp [B]
  have hcomp : ∀ z, z ∈ Icc x r → B z ≤ A z :=
    image_le_of_deriv_right_le_deriv_boundary
      (a := x) (b := r)
      (f := B) (f' := fun t : ℝ => γ * (1 / (t - a)))
      (B := A) (B' := c)
      hBcont hBderiv hbase (hAcont x hx.1 hxr) (hA x hx.1 hxr)
      (fun t ht => by
        have h := hc t (lt_of_lt_of_le hx.1 ht.1) ht.2
        rw [div_eq_mul_one_div] at h
        exact h)
  have hrineq : B r ≤ A r := hcomp r ⟨hxr, le_rfl⟩
  have hrineq' : A x + γ * Real.log (r - a) ≤ A r + γ * Real.log (x - a) := by
    simpa [B] using hrineq
  linarith

/-- Left-side logarithmic primitive bound.

If `A' = c` and `c ≤ -γ/(a-t)` on `(l,a)`, then near `a` from the left, `A`
is bounded above by a logarithmic barrier tending to `-∞`. -/
theorem eventually_primitive_le_log_barrier_left
    {A c : ℝ → ℝ} {l a γ : ℝ}
    (hla : l < a) (_hγ : 0 < γ)
    (hAcont : ∀ x : ℝ, l ≤ x → x < a → ContinuousOn A (Icc l x))
    (hA : ∀ x : ℝ, l ≤ x → x < a →
      ∀ t ∈ Ico l x, HasDerivWithinAt A (c t) (Ici t) t)
    (hc : ∀ t : ℝ, l ≤ t → t < a → c t ≤ -(γ / (a - t))) :
    ∀ᶠ x in 𝓝[<] a,
      A x ≤ A l + γ * (Real.log (a - x) - Real.log (a - l)) := by
  filter_upwards [Ioo_mem_nhdsLT hla] with x hx
  have hlx : l ≤ x := hx.1.le
  let B : ℝ → ℝ := fun s => A l + γ * Real.log (a - s) - γ * Real.log (a - l)
  have hBcont : ContinuousOn B (Icc l x) := by
    have hsubcont : ContinuousOn (fun s : ℝ => a - s) (Icc l x) := by
      exact continuousOn_const.sub continuousOn_id
    have hne : ∀ s ∈ Icc l x, a - s ≠ 0 := by
      intro s hs
      exact sub_ne_zero.mpr (ne_of_gt (lt_of_le_of_lt hs.2 hx.2))
    have hlog : ContinuousOn (fun s : ℝ => Real.log (a - s)) (Icc l x) :=
      hsubcont.log hne
    exact (continuousOn_const.add (continuousOn_const.mul hlog)).sub continuousOn_const
  have hBderiv : ∀ t ∈ Ico l x,
      HasDerivWithinAt B (-(γ * (1 / (a - t)))) (Ici t) t := by
    intro t ht
    have hta : t < a := lt_trans ht.2 hx.2
    have hderiv := hasDerivWithinAt_log_const_sub (a := a) (x := t) (γ := γ) hta
    dsimp only [B]
    change HasDerivWithinAt
      (fun u : ℝ => A l + γ * Real.log (a - u) - γ * Real.log (a - l))
      (-(γ * (1 / (a - t)))) (Ici t) t
    exact (hderiv.const_add (A l)).sub_const (γ * Real.log (a - l))
  have hbase : A l ≤ B l := by simp [B]
  have hcomp : ∀ z, z ∈ Icc l x → A z ≤ B z :=
    image_le_of_deriv_right_le_deriv_boundary
      (a := l) (b := x)
      (f := A) (f' := c)
      (B := B) (B' := fun t : ℝ => -(γ * (1 / (a - t))))
      (hAcont x hlx hx.2) (hA x hlx hx.2) hbase hBcont hBderiv
      (fun t ht => by
        have h := hc t ht.1 (lt_trans ht.2 hx.2)
        rw [div_eq_mul_one_div] at h
        exact h)
  have hxineq : A x ≤ A l + γ * Real.log (a - x) - γ * Real.log (a - l) := by
    simpa [B] using hcomp x ⟨hlx, le_rfl⟩
  linarith

/-- Right-side logarithmic barrier tends to `-∞`, packaged with the primitive
comparison above. -/
theorem primitive_tendsto_atBot_of_right_log_singularity
    {A c : ℝ → ℝ} {a r γ : ℝ}
    (har : a < r) (hγ : 0 < γ)
    (hAcont : ∀ x : ℝ, a < x → x ≤ r → ContinuousOn A (Icc x r))
    (hA : ∀ x : ℝ, a < x → x ≤ r →
      ∀ t ∈ Ico x r, HasDerivWithinAt A (c t) (Ici t) t)
    (hc : ∀ t : ℝ, a < t → t < r → γ / (t - a) ≤ c t) :
    Tendsto A (𝓝[>] a) atBot := by
  refine tendsto_atBot_of_eventually_le_of_tendsto_atBot
    (A := A) (B := fun x : ℝ => A r + γ * (Real.log (x - a) - Real.log (r - a))) ?_ ?_
  · have h :=
      tendsto_log_sub_const_nhdsGT_atBot
        (a := a) (C := A r - γ * Real.log (r - a)) (γ := γ) hγ
    convert h using 1
    ext x
    ring
  · exact eventually_primitive_le_log_barrier_right har hγ hAcont hA hc

/-- Left-side logarithmic barrier tends to `-∞`, packaged with the primitive
comparison above. -/
theorem primitive_tendsto_atBot_of_left_log_singularity
    {A c : ℝ → ℝ} {l a γ : ℝ}
    (hla : l < a) (hγ : 0 < γ)
    (hAcont : ∀ x : ℝ, l ≤ x → x < a → ContinuousOn A (Icc l x))
    (hA : ∀ x : ℝ, l ≤ x → x < a →
      ∀ t ∈ Ico l x, HasDerivWithinAt A (c t) (Ici t) t)
    (hc : ∀ t : ℝ, l ≤ t → t < a → c t ≤ -(γ / (a - t))) :
    Tendsto A (𝓝[<] a) atBot := by
  refine tendsto_atBot_of_eventually_le_of_tendsto_atBot
    (A := A) (B := fun x : ℝ => A l + γ * (Real.log (a - x) - Real.log (a - l))) ?_ ?_
  · have h :=
      tendsto_log_const_sub_nhdsLT_atBot
        (a := a) (C := A l - γ * Real.log (a - l)) (γ := γ) hγ
    convert h using 1
    ext x
    ring
  · exact eventually_primitive_le_log_barrier_left hla hγ hAcont hA hc

/-- Algebraic source of the right-side logarithmic singularity for the actual
Laplace Abel coefficient.

If `μ' ≥ δ > 0` and `0 < m ≤ L(t-a)` on the right of an upward crossing, then
`2μ'/m` dominates `(2δ/L)/(t-a)`. -/
theorem right_laplaceCoeff_logSingularity_of_muDeriv_lower_m_upper
    {δ L μ m h : ℝ} (hδ : 0 < δ) (hL : 0 < L) (hh : 0 < h)
    (hmpos : 0 < m) (hμ : δ ≤ μ) (hmupper : m ≤ L * h) :
    (((2 : ℝ) * δ) / L) / h ≤ ((2 : ℝ) * μ) / m := by
  field_simp [hL.ne', hh.ne', hmpos.ne']
  nlinarith [mul_nonneg (le_of_lt hL) (le_of_lt hh),
    mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) (le_of_lt hδ),
    mul_le_mul_of_nonneg_left hμ (by norm_num : (0 : ℝ) ≤ 2), hmupper]

/-- Algebraic source of the left-side logarithmic singularity for the actual
Laplace Abel coefficient.

If `μ' ≥ δ > 0` and `-L(a-t) ≤ m < 0` on the left of an upward crossing, then
`2μ'/m ≤ -(2δ/L)/(a-t)`. -/
theorem left_laplaceCoeff_logSingularity_of_muDeriv_lower_m_lower
    {δ L μ m h : ℝ} (hδ : 0 < δ) (hL : 0 < L) (hh : 0 < h)
    (hmneg : m < 0) (hμ : δ ≤ μ) (hmlower : -(L * h) ≤ m) :
    ((2 : ℝ) * μ) / m ≤ -((((2 : ℝ) * δ) / L) / h) := by
  let n : ℝ := -m
  have hnpos : 0 < n := by
    dsimp [n]
    exact neg_pos.mpr hmneg
  have hnupper : n ≤ L * h := by
    dsimp [n]
    linarith
  have hright : (((2 : ℝ) * δ) / L) / h ≤ ((2 : ℝ) * μ) / n :=
    right_laplaceCoeff_logSingularity_of_muDeriv_lower_m_upper
      hδ hL hh hnpos hμ hnupper
  have hneg := neg_le_neg hright
  have hquot : ((2 : ℝ) * μ) / m = -(((2 : ℝ) * μ) / n) := by
    dsimp [n]
    field_simp [hmneg.ne]
  rw [hquot]
  exact hneg

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

/-- L8 right interval from a concrete logarithmic singularity.

This is the upstream form used at the right side of an upward crossing: if on
some reference subinterval `(a,r)` the primitive coefficient satisfies
`γ/(t-a) ≤ c t` with `γ > 0`, then any primitive `A' = c` tends to `-∞` as
`t → a+`; boundedness of `W` then forces vanishing on the whole right interval. -/
theorem abel_right_interval_zero_of_upwardCrossing_of_logSingularity
    {W A c : ℝ → ℝ} {a b r γ : ℝ}
    (hab : a < b) (har : a < r) (hrb : r < b) (hγ : 0 < γ)
    (hcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x y))
    (hW : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt A (c t) (Ici t) t)
    (hAcont : ∀ x y : ℝ, a < x → x ≤ y → y < b → ContinuousOn A (Icc x y))
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[>] a) (norm ∘ W))
    (hc : ∀ t : ℝ, a < t → t < r → γ / (t - a) ≤ c t) :
    ∀ x : ℝ, a < x → x < b → W x = 0 := by
  have hAdiv : Tendsto A (𝓝[>] a) atBot :=
    primitive_tendsto_atBot_of_right_log_singularity
      (A := A) (c := c) (a := a) (r := r) (γ := γ)
      har hγ
      (fun x hax hxr => hAcont x r hax hxr hrb)
      (fun x hax hxr => hA x r hax hxr hrb)
      hc
  exact abel_right_interval_zero_of_upwardCrossing_of_tendsto_atBot
    (W := W) (A := A) (c := c) (a := a) (b := b)
    hab hcont hW hA hWbounded hAdiv

/-- L8 right interval for the actual Laplace coefficient `c = 2 μ'/m`.

This closes the right-side upstream barrier construction from the natural local
inputs supplied by strict tilted-mean growth and one-sided sign-change geometry:
`μ' ≥ δ > 0` and `0 < m ≤ L(t-a)` near the upward crossing. -/
theorem abel_right_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_upper
    {W A μDeriv m : ℝ → ℝ} {a b r δ L : ℝ}
    (hab : a < b) (har : a < r) (hrb : r < b) (hδ : 0 < δ) (hL : 0 < L)
    (hcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x y))
    (hW : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y,
        HasDerivWithinAt W (-((((2 : ℝ) * μDeriv t) / m t)) * W t) (Ici t) t)
    (hA : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y,
        HasDerivWithinAt A (((2 : ℝ) * μDeriv t) / m t) (Ici t) t)
    (hAcont : ∀ x y : ℝ, a < x → x ≤ y → y < b → ContinuousOn A (Icc x y))
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[>] a) (norm ∘ W))
    (hμ : ∀ t : ℝ, a < t → t < r → δ ≤ μDeriv t)
    (hmpos : ∀ t : ℝ, a < t → t < r → 0 < m t)
    (hmupper : ∀ t : ℝ, a < t → t < r → m t ≤ L * (t - a)) :
    ∀ x : ℝ, a < x → x < b → W x = 0 := by
  have hγ : 0 < ((2 : ℝ) * δ) / L := by
    exact div_pos (mul_pos (by norm_num) hδ) hL
  exact abel_right_interval_zero_of_upwardCrossing_of_logSingularity
    (W := W) (A := A)
    (c := fun t : ℝ => ((2 : ℝ) * μDeriv t) / m t)
    (a := a) (b := b) (r := r) (γ := ((2 : ℝ) * δ) / L)
    hab har hrb hγ hcont hW hA hAcont hWbounded
    (fun t hat htr =>
      right_laplaceCoeff_logSingularity_of_muDeriv_lower_m_upper
        hδ hL (sub_pos.mpr hat) (hmpos t hat htr) (hμ t hat htr)
        (hmupper t hat htr))

/-- L8 right interval from an upper barrier for the primitive near the upward
crossing. -/
theorem abel_right_interval_zero_of_upwardCrossing_of_atBotBarrier
    {W A B c : ℝ → ℝ} {a b : ℝ}
    (hab : a < b)
    (hcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x y))
    (hW : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt A (c t) (Ici t) t)
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[>] a) (norm ∘ W))
    (hBdiv : Tendsto B (𝓝[>] a) atBot)
    (hAupper : ∀ᶠ t in 𝓝[>] a, A t ≤ B t) :
    ∀ x : ℝ, a < x → x < b → W x = 0 :=
  abel_right_interval_zero_of_upwardCrossing_of_tendsto_atBot
    hab hcont hW hA hWbounded
    (tendsto_atBot_of_eventually_le_of_tendsto_atBot hBdiv hAupper)

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

/-- L8 left interval from a concrete logarithmic singularity.

This is the upstream form used at the left side of an upward crossing at `b`.
If on `(l,b)` the coefficient satisfies `c t ≤ -γ/(b-t)` with `γ > 0`, then a
primitive tends to `-∞` as `t → b-`; boundedness of `W` then forces vanishing on
the whole left interval. -/
theorem abel_left_interval_zero_of_upwardCrossing_of_logSingularity
    {W A c : ℝ → ℝ} {a b l γ : ℝ}
    (hab : a < b) (hal : a < l) (hlb : l < b) (hγ : 0 < γ)
    (hcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x y))
    (hW : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt A (c t) (Ici t) t)
    (hAcont : ∀ x y : ℝ, a < x → x ≤ y → y < b → ContinuousOn A (Icc x y))
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[<] b) (norm ∘ W))
    (hc : ∀ t : ℝ, l ≤ t → t < b → c t ≤ -(γ / (b - t))) :
    ∀ x : ℝ, a < x → x < b → W x = 0 := by
  have hAdiv : Tendsto A (𝓝[<] b) atBot :=
    primitive_tendsto_atBot_of_left_log_singularity
      (A := A) (c := c) (l := l) (a := b) (γ := γ)
      hlb hγ
      (fun x hlx hxb => hAcont l x hal hlx hxb)
      (fun x hlx hxb => hA l x hal hlx hxb)
      hc
  exact abel_left_interval_zero_of_upwardCrossing_of_tendsto_atBot
    (W := W) (A := A) (c := c) (a := a) (b := b)
    hab hcont hW hA hWbounded hAdiv

/-- L8 left interval for the actual Laplace coefficient `c = 2 μ'/m`.

This closes the left-side upstream barrier construction from the natural local
inputs supplied by strict tilted-mean growth and one-sided sign-change geometry:
`μ' ≥ δ > 0` and `-L(b-t) ≤ m < 0` near the upward crossing. -/
theorem abel_left_interval_zero_of_upwardCrossing_of_muDeriv_lower_m_lower
    {W A μDeriv m : ℝ → ℝ} {a b l δ L : ℝ}
    (hab : a < b) (hal : a < l) (hlb : l < b) (hδ : 0 < δ) (hL : 0 < L)
    (hcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x y))
    (hW : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y,
        HasDerivWithinAt W (-((((2 : ℝ) * μDeriv t) / m t)) * W t) (Ici t) t)
    (hA : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y,
        HasDerivWithinAt A (((2 : ℝ) * μDeriv t) / m t) (Ici t) t)
    (hAcont : ∀ x y : ℝ, a < x → x ≤ y → y < b → ContinuousOn A (Icc x y))
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[<] b) (norm ∘ W))
    (hμ : ∀ t : ℝ, l ≤ t → t < b → δ ≤ μDeriv t)
    (hmneg : ∀ t : ℝ, l ≤ t → t < b → m t < 0)
    (hmlower : ∀ t : ℝ, l ≤ t → t < b → -(L * (b - t)) ≤ m t) :
    ∀ x : ℝ, a < x → x < b → W x = 0 := by
  have hγ : 0 < ((2 : ℝ) * δ) / L := by
    exact div_pos (mul_pos (by norm_num) hδ) hL
  exact abel_left_interval_zero_of_upwardCrossing_of_logSingularity
    (W := W) (A := A)
    (c := fun t : ℝ => ((2 : ℝ) * μDeriv t) / m t)
    (a := a) (b := b) (l := l) (γ := ((2 : ℝ) * δ) / L)
    hab hal hlb hγ hcont hW hA hAcont hWbounded
    (fun t hlt htb =>
      left_laplaceCoeff_logSingularity_of_muDeriv_lower_m_lower
        hδ hL (sub_pos.mpr htb) (hmneg t hlt htb) (hμ t hlt htb)
        (hmlower t hlt htb))

/-- L8 left interval from an upper barrier for the primitive near the upward
crossing. -/
theorem abel_left_interval_zero_of_upwardCrossing_of_atBotBarrier
    {W A B c : ℝ → ℝ} {a b : ℝ}
    (hab : a < b)
    (hcont : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ContinuousOn (fun t => W t * Real.exp (A t)) (Icc x y))
    (hW : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt W (-(c t) * W t) (Ici t) t)
    (hA : ∀ x y : ℝ, a < x → x ≤ y → y < b →
      ∀ t ∈ Ico x y, HasDerivWithinAt A (c t) (Ici t) t)
    (hWbounded : IsBoundedUnder (· ≤ ·) (𝓝[<] b) (norm ∘ W))
    (hBdiv : Tendsto B (𝓝[<] b) atBot)
    (hAupper : ∀ᶠ t in 𝓝[<] b, A t ≤ B t) :
    ∀ x : ℝ, a < x → x < b → W x = 0 :=
  abel_left_interval_zero_of_upwardCrossing_of_tendsto_atBot
    hab hcont hW hA hWbounded
    (tendsto_atBot_of_eventually_le_of_tendsto_atBot hBdiv hAupper)

/-! ## L9: finite breakpoint gluing -/

/-- Vanishing data on all open gaps to the right of a fixed left breakpoint.

For a list `[b₁, b₂, ...]`, this records vanishing on `(left,b₁)`,
`(b₁,b₂)`, ..., and finally on the right ray after the last breakpoint.  The
list is intended to be ordered in applications, but the lemma below only needs
the data itself. -/
def VanishesOnOrderedGapsFrom (W : ℝ → ℝ) (left : ℝ) : List ℝ → Prop
  | [] => ∀ x : ℝ, left < x → W x = 0
  | right :: rest =>
      (∀ x : ℝ, left < x → x < right → W x = 0) ∧
        VanishesOnOrderedGapsFrom W right rest

/-- Vanishing data on the left ray, all adjacent open gaps, and the right ray
associated to a finite breakpoint list. -/
def VanishesOnOrderedGaps (W : ℝ → ℝ) : List ℝ → Prop
  | [] => ∀ x : ℝ, W x = 0
  | first :: rest =>
      (∀ x : ℝ, x < first → W x = 0) ∧
        VanishesOnOrderedGapsFrom W first rest

/-- Alternating upward-crossing cover data to the right of a downward
breakpoint.

The intended shape is `[up₁, down₂, up₃, down₄, ...]` after an initial downward
breakpoint.  Each upward crossing kills the two adjacent open gaps:
`(leftDown, up)` and `(up, rightDown)`.  The recursion then continues from the
new downward breakpoint.  The empty tail is the final right ray after the last
downward breakpoint.  A singleton tail is deliberately impossible: it would
mean the breakpoint list ends at an upward crossing, which cannot be the
odd-parity Laplace mean-shift pattern. -/
def VanishesOnAlternatingUpwardPairsFrom (W : ℝ → ℝ) (leftDown : ℝ) : List ℝ → Prop
  | [] => ∀ x : ℝ, leftDown < x → W x = 0
  | [_up] => False
  | up :: rightDown :: rest =>
      (∀ x : ℝ, leftDown < x → x < up → W x = 0) ∧
      (∀ x : ℝ, up < x → x < rightDown → W x = 0) ∧
        VanishesOnAlternatingUpwardPairsFrom W rightDown rest

/-- Top-level alternating upward-crossing cover data.

The intended breakpoint list has the form
`[down₁, up₂, down₃, up₄, ..., down_M]`: L6 kills the left ray before `down₁`,
each upward crossing kills its two adjacent interior gaps, and the last
downward crossing supplies the right ray. -/
def VanishesOnAlternatingUpwardPairs (W : ℝ → ℝ) : List ℝ → Prop
  | [] => ∀ x : ℝ, W x = 0
  | firstDown :: rest =>
      (∀ x : ℝ, x < firstDown → W x = 0) ∧
        VanishesOnAlternatingUpwardPairsFrom W firstDown rest

/-- The alternating upward-crossing cover produces the ordered-gap witness used
by the L9 gluing theorem. -/
theorem VanishesOnOrderedGapsFrom.of_alternatingUpwardPairs
    {W : ℝ → ℝ} {leftDown : ℝ} :
    ∀ {breaks : List ℝ},
      VanishesOnAlternatingUpwardPairsFrom W leftDown breaks →
        VanishesOnOrderedGapsFrom W leftDown breaks
  | [], hvan => hvan
  | [_up], hvan => False.elim hvan
  | up :: rightDown :: rest, hvan => by
      rcases hvan with ⟨hleftGap, hrightGap, htail⟩
      exact ⟨hleftGap, ⟨hrightGap,
        VanishesOnOrderedGapsFrom.of_alternatingUpwardPairs htail⟩⟩

/-- Top-level alternating upward-crossing cover produces the ordered-gap data. -/
theorem VanishesOnOrderedGaps.of_alternatingUpwardPairs
    {W : ℝ → ℝ} :
    ∀ {breaks : List ℝ},
      VanishesOnAlternatingUpwardPairs W breaks → VanishesOnOrderedGaps W breaks
  | [], hvan => hvan
  | firstDown :: rest, hvan => by
      rcases hvan with ⟨hleftRay, htail⟩
      exact ⟨hleftRay, VanishesOnOrderedGapsFrom.of_alternatingUpwardPairs htail⟩

/-- The gap data to the right of a fixed breakpoint implies vanishing away from
the remaining breakpoints. -/
theorem VanishesOnOrderedGapsFrom.eq_zero_of_not_mem
    {W : ℝ → ℝ} {left : ℝ} :
    ∀ {breaks : List ℝ}, VanishesOnOrderedGapsFrom W left breaks →
      ∀ {x : ℝ}, left < x → x ∉ breaks → W x = 0
  | [], hvan, x, hxleft, _ => hvan x hxleft
  | right :: rest, hvan, x, hxleft, hxnot => by
      rcases hvan with ⟨hgap, htail⟩
      have hxne : x ≠ right := by
        intro hx
        exact hxnot (by simp [hx])
      rcases lt_trichotomy x right with hlt | heq | hgt
      · exact hgap x hxleft hlt
      · exact (hxne heq).elim
      · exact VanishesOnOrderedGapsFrom.eq_zero_of_not_mem htail hgt (by
          intro hxmem
          exact hxnot (by simp [hxmem]))

/-- Finite ordered-gap data implies vanishing away from the listed breakpoints. -/
theorem VanishesOnOrderedGaps.eq_zero_of_not_mem
    {W : ℝ → ℝ} :
    ∀ {breaks : List ℝ}, VanishesOnOrderedGaps W breaks →
      ∀ {x : ℝ}, x ∉ breaks → W x = 0
  | [], hvan, x, _ => hvan x
  | first :: rest, hvan, x, hxnot => by
      rcases hvan with ⟨hleft, htail⟩
      have hxne : x ≠ first := by
        intro hx
        exact hxnot (by simp [hx])
      rcases lt_trichotomy x first with hlt | heq | hgt
      · exact hleft x hlt
      · exact (hxne heq).elim
      · exact VanishesOnOrderedGapsFrom.eq_zero_of_not_mem htail hgt (by
          intro hxmem
          exact hxnot (by simp [hxmem]))

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

/-- **L9 ordered-breakpoint cover.**  If `W` is continuous and vanishes on the
left ray, every adjacent open gap, and the right ray determined by a finite
breakpoint list, then `W` vanishes everywhere.

This is the upstream combinatorial package used after L6 kills the outer gaps
and L8 kills the interior flanks.  Continuity handles the breakpoints
themselves. -/
theorem continuous_eq_zero_of_vanishesOnOrderedGaps
    {W : ℝ → ℝ} (breaks : List ℝ)
    (hcont : Continuous W)
    (hvan : VanishesOnOrderedGaps W breaks) :
    ∀ x : ℝ, W x = 0 := by
  refine continuous_eq_zero_of_zero_off_finset (breaks.toFinset) hcont ?_
  intro x hx
  exact VanishesOnOrderedGaps.eq_zero_of_not_mem hvan (by
    intro hxmem
    exact hx (by simpa using hxmem))

/-- **L9 alternating parity cover.**  If `W` is continuous and the finite
breakpoint list carries the alternating downward/upward cover data generated
by the Laplace mean-shift sign pattern, then `W` vanishes everywhere.

This is the final upstream L9 package: it converts the parity cover into the
ordered-gap witness and then applies the continuity gluing theorem. -/
theorem continuous_eq_zero_of_alternatingUpwardPairs
    {W : ℝ → ℝ} (breaks : List ℝ)
    (hcont : Continuous W)
    (hvan : VanishesOnAlternatingUpwardPairs W breaks) :
    ∀ x : ℝ, W x = 0 := by
  exact continuous_eq_zero_of_vanishesOnOrderedGaps breaks hcont
    (VanishesOnOrderedGaps.of_alternatingUpwardPairs hvan)

end DriftingIdentifiability
