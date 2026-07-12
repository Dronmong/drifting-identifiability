import DriftingIdentifiability.LaplaceTiltedMeanMonotone

/-!
# L5: common-ODE Abel calculus for the a.c. Laplace converse

This module contains the axiom-free calculus core for L5 in
`LaplaceACDerivation.md`.

The analytic plan says that, under zero drift, the two Laplace normalizers
`Z_p` and `Z_q` solve the same second-order scalar ODE on each region where the
common mean-shift `m` is nonzero:

```text
m Z'' + a Z' + b Z = 0
```

with `a = 2 mu'` and `b = mu'' - m/tau^2`.  Abel's computation then gives

```text
W' = -(a/m) W
```

for the Wronskian `W = Z_p' Z_q - Z_p Z_q'`.

The theorem below proves exactly that pointwise calculus statement from
ordinary `HasDerivAt` hypotheses.  It deliberately does **not** axiomatize any
measure-theoretic a.c. regularity or any final identifiability conclusion.
Later L5 work should instantiate this lemma a.e. using the certified first-order
Laplace identities and the a.c. regularity of the one-sided masses.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-- Scalar Wronskian from a pair of functions and named derivative functions.

The order matches the project's raw-normalizer Wronskian convention:
`f' * g - f * g'`. -/
noncomputable def scalarWronskian
    (f f' g g' : ℝ → ℝ) (x : ℝ) : ℝ :=
  f' x * g x - f x * g' x

/-- Derivative of a scalar Wronskian, before using any ODE. -/
theorem hasDerivAt_scalarWronskian
    (f f' g g' : ℝ → ℝ) (x : ℝ)
    {f'' g'' : ℝ}
    (hf : HasDerivAt f (f' x) x)
    (hf' : HasDerivAt f' f'' x)
    (hg : HasDerivAt g (g' x) x)
    (hg' : HasDerivAt g' g'' x) :
    HasDerivAt (fun t => scalarWronskian f f' g g' t)
      (f'' * g x - f x * g'') x := by
  have hprod₁ := hf'.mul hg
  have hprod₂ := hf.mul hg'
  have hsub := hprod₁.sub hprod₂
  change HasDerivAt (f' * g - f * g') (f'' * g x - f x * g'') x
  have hderiv :
      (f'' * g x + f' x * g' x) - (f' x * g' x + f x * g'') =
        f'' * g x - f x * g'' := by
    ring
  simpa [hderiv] using hsub

/-! ## Algebra from the certified first-order Laplace identities -/

/-- L5 algebra, first step.  If the common mean-shift relation `D = m Z` has
pointwise derivative `m' Z + m Z'`, and the certified first-order identity gives
`D' = L/τ - 2Z`, then the companion value is
`L = τ ((m' + 2) Z + m Z')`.

This theorem is deliberately scalar: later AC work can feed it the appropriate
a.e. derivative values. -/
theorem laplaceCommonODE_companion_value_of_firstOrder
    {τ m m' Z Z' L : ℝ} (hτ : τ ≠ 0)
    (hD' : m' * Z + m * Z' = L / τ - 2 * Z) :
    L = τ * ((m' + 2) * Z + m * Z') := by
  have hLdiv : L / τ = m' * Z + m * Z' + 2 * Z := by
    linarith
  calc
    L = τ * (L / τ) := by
      rw [mul_comm, div_mul_cancel₀ L hτ]
    _ = τ * (m' * Z + m * Z' + 2 * Z) := by rw [hLdiv]
    _ = τ * ((m' + 2) * Z + m * Z') := by ring

/-- L5 algebra, second step.  If the differentiated companion formula has value
`L' = τ (m'' Z + 2(m' + 1)Z' + m Z'')`, while the certified first-order identity
gives `L' = m Z / τ`, then `Z` satisfies the common second-order ODE

`m Z'' + 2(m' + 1) Z' + (m'' - m/τ²) Z = 0`.

With `μ = m + x`, this is exactly
`m Z'' + 2 μ' Z' + (μ'' - m/τ²) Z = 0`. -/
theorem laplaceCommonODE_value_of_companion_derivative
    {τ m m' m'' Z Z' Z'' L' : ℝ} (hτ : τ ≠ 0)
    (hL'diff : L' = τ * (m'' * Z + 2 * (m' + 1) * Z' + m * Z''))
    (hL'cert : L' = m * Z / τ) :
    m * Z'' + 2 * (m' + 1) * Z' + (m'' - m / τ ^ 2) * Z = 0 := by
  have hA :
      m'' * Z + 2 * (m' + 1) * Z' + m * Z'' = m * Z / τ ^ 2 := by
    have hτA :
        τ * (m'' * Z + 2 * (m' + 1) * Z' + m * Z'') = m * Z / τ := by
      rw [← hL'diff, hL'cert]
    have hτ2 : τ ^ 2 ≠ 0 := pow_ne_zero 2 hτ
    calc
      m'' * Z + 2 * (m' + 1) * Z' + m * Z''
          = (τ * (m'' * Z + 2 * (m' + 1) * Z' + m * Z'')) / τ := by
            rw [mul_div_cancel_left₀ _ hτ]
      _ = (m * Z / τ) / τ := by rw [hτA]
      _ = m * Z / τ ^ 2 := by
        field_simp [hτ, hτ2]
  calc
    m * Z'' + 2 * (m' + 1) * Z' + (m'' - m / τ ^ 2) * Z
        = (m'' * Z + 2 * (m' + 1) * Z' + m * Z'') - m * Z / τ ^ 2 := by ring
    _ = 0 := by rw [hA]; ring

private lemma commonODE_wronskian_deriv_value
    {m a b f f' f'' g g' g'' : ℝ}
    (hm : m ≠ 0)
    (hf : m * f'' + a * f' + b * f = 0)
    (hg : m * g'' + a * g' + b * g = 0) :
    f'' * g - f * g'' = -(a / m) * (f' * g - f * g') := by
  have hfsol : m * f'' = -(a * f' + b * f) := by
    linarith
  have hgsol : m * g'' = -(a * g' + b * g) := by
    linarith
  have hmain : m * (f'' * g - f * g'') = -a * (f' * g - f * g') := by
    calc
      m * (f'' * g - f * g'') = (m * f'') * g - f * (m * g'') := by ring
      _ = (-(a * f' + b * f)) * g - f * (-(a * g' + b * g)) := by
        rw [hfsol, hgsol]
      _ = -a * (f' * g - f * g') := by ring
  field_simp [hm]
  calc
    (f'' * g - f * g'') * m = m * (f'' * g - f * g'') := by ring
    _ = -a * (f' * g - f * g') := hmain
    _ = -(a * (g * f' - f * g')) := by ring

/-- **L5 Abel core.**  If `f` and `g` solve the same scalar second-order ODE
`m Z'' + a Z' + b Z = 0` at a point where `m ≠ 0`, then their Wronskian
`W = f' g - f g'` satisfies `W' = -(a/m) W` at that point.

For the Laplace a.c. converse, the intended specialization is
`f = Z_p`, `g = Z_q`, `a = 2 * mu'`, and `b = mu'' - m/tau^2`. -/
theorem hasDerivAt_wronskian_of_common_secondOrder
    (m a b f f' g g' : ℝ → ℝ) (x : ℝ)
    {f'' g'' : ℝ}
    (hf : HasDerivAt f (f' x) x)
    (hf' : HasDerivAt f' f'' x)
    (hg : HasDerivAt g (g' x) x)
    (hg' : HasDerivAt g' g'' x)
    (hm : m x ≠ 0)
    (hodef : m x * f'' + a x * f' x + b x * f x = 0)
    (hodeg : m x * g'' + a x * g' x + b x * g x = 0) :
    HasDerivAt (fun t => scalarWronskian f f' g g' t)
      (-(a x / m x) * scalarWronskian f f' g g' x) x := by
  have hW := hasDerivAt_scalarWronskian f f' g g' x hf hf' hg hg'
  have hval :
      f'' * g x - f x * g'' =
        -(a x / m x) * scalarWronskian f f' g g' x := by
    exact commonODE_wronskian_deriv_value hm hodef hodeg
  simpa [hval] using hW

/-- Same Abel core, with the Laplace coefficient notation `a = 2 * mu'`.

This is just a convenience wrapper for the exact formula used in
`LaplaceACDerivation.md`: `W' = -(2*mu'/m) W`. -/
theorem hasDerivAt_wronskian_of_laplace_commonODE
    (m mu' b f f' g g' : ℝ → ℝ) (x : ℝ)
    {f'' g'' : ℝ}
    (hf : HasDerivAt f (f' x) x)
    (hf' : HasDerivAt f' f'' x)
    (hg : HasDerivAt g (g' x) x)
    (hg' : HasDerivAt g' g'' x)
    (hm : m x ≠ 0)
    (hodef : m x * f'' + (2 * mu' x) * f' x + b x * f x = 0)
    (hodeg : m x * g'' + (2 * mu' x) * g' x + b x * g x = 0) :
    HasDerivAt (fun t => scalarWronskian f f' g g' t)
      (-(2 * mu' x / m x) * scalarWronskian f f' g g' x) x := by
  simpa [mul_div_assoc] using
    hasDerivAt_wronskian_of_common_secondOrder
      m (fun t => 2 * mu' t) b f f' g g' x
      hf hf' hg hg' hm hodef hodeg

end DriftingIdentifiability
