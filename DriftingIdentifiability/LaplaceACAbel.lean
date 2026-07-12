import DriftingIdentifiability.LaplaceTiltedMeanMonotone
import DriftingIdentifiability.LaplaceGeneralConverseWronskian

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

/-! ## Project-native L5 wrapper for the Laplace normalizer Wronskian -/

/-- The project-native common mean-shift scalar in the 1-d Laplace converse:
`m_p = D_p/Z_p`, where `D_p` is the mean-shift numerator and `Z_p` the raw
Laplace normalizer.  Under zero drift, this same scalar also satisfies
`D_q = m_p Z_q`. -/
noncomputable def laplaceMeanShiftRatio
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (∫ y, laplaceWeightedDisplacement τ x y ∂p) /
    kernelNormalizer (laplaceKernel τ) p x

theorem laplaceMeanShiftRatio_common_self
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ) :
    (∫ y, laplaceWeightedDisplacement τ x y ∂p) =
      laplaceMeanShiftRatio τ p x *
        kernelNormalizer (laplaceKernel τ) p x := by
  have hZ : kernelNormalizer (laplaceKernel τ) p x ≠ 0 :=
    (laplaceKernelNormalizer_pos p τ hτ x).ne'
  unfold laplaceMeanShiftRatio
  field_simp [hZ]

theorem laplaceMeanShiftRatio_common_of_zeroDrift
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q) (x : ℝ) :
    (∫ y, laplaceWeightedDisplacement τ x y ∂q) =
      laplaceMeanShiftRatio τ p x *
        kernelNormalizer (laplaceKernel τ) q x := by
  have hZp : kernelNormalizer (laplaceKernel τ) p x ≠ 0 :=
    (laplaceKernelNormalizer_pos p τ hτ x).ne'
  have hcross := (laplaceZeroDrift_iff_crossDisplacement τ hτ p q).mp hzero x
  simp only [smul_eq_mul] at hcross
  unfold laplaceMeanShiftRatio
  calc
    (∫ y, laplaceWeightedDisplacement τ x y ∂q)
        = (kernelNormalizer (laplaceKernel τ) p x *
            (∫ y, laplaceWeightedDisplacement τ x y ∂q)) /
            kernelNormalizer (laplaceKernel τ) p x := by
          field_simp [hZp]
    _ = (kernelNormalizer (laplaceKernel τ) q x *
            (∫ y, laplaceWeightedDisplacement τ x y ∂p)) /
            kernelNormalizer (laplaceKernel τ) p x := by
          rw [← hcross]
    _ = ((∫ y, laplaceWeightedDisplacement τ x y ∂p) /
            kernelNormalizer (laplaceKernel τ) p x) *
          kernelNormalizer (laplaceKernel τ) q x := by
          ring

private theorem laplaceCompanion_value_of_commonMeanShift_at
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (m m' : ℝ → ℝ) (x : ℝ)
    (hm : HasDerivAt m (m' x) x)
    (hZ : HasDerivAt (fun t => kernelNormalizer (laplaceKernel τ) p t)
      (laplaceKernelNormalizerRightDerivCoeff τ p x) x)
    (hcommon :
      ∀ t : ℝ, (∫ y, laplaceWeightedDisplacement τ t y ∂p) =
        m t * kernelNormalizer (laplaceKernel τ) p t) :
    kernelNormalizer (laplaceCompanionKernel τ) p x =
      τ * ((m' x + 2) * kernelNormalizer (laplaceKernel τ) p x +
        m x * laplaceKernelNormalizerRightDerivCoeff τ p x) := by
  have hprod := hm.mul hZ
  have hDcommon :
      HasDerivAt (fun t : ℝ => ∫ y, laplaceWeightedDisplacement τ t y ∂p)
        (m' x * kernelNormalizer (laplaceKernel τ) p x +
          m x * laplaceKernelNormalizerRightDerivCoeff τ p x) x := by
    refine hprod.congr_of_eventuallyEq ?_
    exact Eventually.of_forall fun t => by
      simpa [Pi.mul_apply] using hcommon t
  have hDcert := hasDerivAt_laplaceDisplacementIntegral τ hτ p x
  have hDvalue :
      m' x * kernelNormalizer (laplaceKernel τ) p x +
          m x * laplaceKernelNormalizerRightDerivCoeff τ p x =
        (1 / τ) * kernelNormalizer (laplaceCompanionKernel τ) p x -
          2 * kernelNormalizer (laplaceKernel τ) p x :=
    hDcommon.unique hDcert
  exact laplaceCommonODE_companion_value_of_firstOrder hτ.ne' (by
    rw [hDvalue]
    ring)

private theorem hasDerivAt_laplaceCompanion_formula
    (τ : ℝ) (p : Measure ℝ) (m m' m'' : ℝ → ℝ) (Z'' : ℝ → ℝ) (x : ℝ)
    (hm : HasDerivAt m (m' x) x)
    (hm' : HasDerivAt m' (m'' x) x)
    (hZ : HasDerivAt (fun t => kernelNormalizer (laplaceKernel τ) p t)
      (laplaceKernelNormalizerRightDerivCoeff τ p x) x)
    (hZ' : HasDerivAt (laplaceKernelNormalizerRightDerivCoeff τ p) (Z'' x) x) :
    HasDerivAt
      (fun t : ℝ =>
        τ * ((m' t + 2) * kernelNormalizer (laplaceKernel τ) p t +
          m t * laplaceKernelNormalizerRightDerivCoeff τ p t))
      (τ * (m'' x * kernelNormalizer (laplaceKernel τ) p x +
        2 * (m' x + 1) * laplaceKernelNormalizerRightDerivCoeff τ p x +
          m x * Z'' x)) x := by
  have hmd_plus : HasDerivAt (fun t : ℝ => m' t + 2) (m'' x) x := by
    simpa using hm'.add_const 2
  have hterm₁ := hmd_plus.mul hZ
  have hterm₂ := hm.mul hZ'
  have hsum := hterm₁.add hterm₂
  have hmul := hsum.const_mul τ
  have hderiv :
      τ * ((m'' x * kernelNormalizer (laplaceKernel τ) p x +
              (m' x + 2) * laplaceKernelNormalizerRightDerivCoeff τ p x) +
            (m' x * laplaceKernelNormalizerRightDerivCoeff τ p x +
              m x * Z'' x)) =
        τ * (m'' x * kernelNormalizer (laplaceKernel τ) p x +
          2 * (m' x + 1) * laplaceKernelNormalizerRightDerivCoeff τ p x +
            m x * Z'' x) := by
    ring
  simpa [Pi.mul_apply, Pi.add_apply, hderiv] using hmul

private theorem laplaceCommonODE_of_commonMeanShift_at
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (m m' m'' : ℝ → ℝ) (Z'' : ℝ → ℝ) (x : ℝ)
    (hm_all : ∀ t : ℝ, HasDerivAt m (m' t) t)
    (hm' : HasDerivAt m' (m'' x) x)
    (hZ_all : ∀ t : ℝ,
      HasDerivAt (fun s => kernelNormalizer (laplaceKernel τ) p s)
        (laplaceKernelNormalizerRightDerivCoeff τ p t) t)
    (hZ' : HasDerivAt (laplaceKernelNormalizerRightDerivCoeff τ p) (Z'' x) x)
    (hcommon :
      ∀ t : ℝ, (∫ y, laplaceWeightedDisplacement τ t y ∂p) =
        m t * kernelNormalizer (laplaceKernel τ) p t) :
    m x * Z'' x +
        2 * (m' x + 1) * laplaceKernelNormalizerRightDerivCoeff τ p x +
      (m'' x - m x / τ ^ 2) * kernelNormalizer (laplaceKernel τ) p x = 0 := by
  have hLformula : ∀ t : ℝ,
      kernelNormalizer (laplaceCompanionKernel τ) p t =
        τ * ((m' t + 2) * kernelNormalizer (laplaceKernel τ) p t +
          m t * laplaceKernelNormalizerRightDerivCoeff τ p t) := by
    intro t
    exact laplaceCompanion_value_of_commonMeanShift_at τ hτ p m m' t
      (hm_all t) (hZ_all t) hcommon
  have hFormulaDeriv :=
    hasDerivAt_laplaceCompanion_formula τ p m m' m'' Z'' x
      (hm_all x) hm' (hZ_all x) hZ'
  have hLderiv_formula :
      HasDerivAt (fun t => kernelNormalizer (laplaceCompanionKernel τ) p t)
        (τ * (m'' x * kernelNormalizer (laplaceKernel τ) p x +
          2 * (m' x + 1) * laplaceKernelNormalizerRightDerivCoeff τ p x +
            m x * Z'' x)) x := by
    refine hFormulaDeriv.congr_of_eventuallyEq ?_
    exact Eventually.of_forall fun t => by
      simpa using hLformula t
  have hLcert₀ := hasDerivAt_laplaceCompanionNormalizer τ hτ p x
  have hLcert :
      HasDerivAt (fun t => kernelNormalizer (laplaceCompanionKernel τ) p t)
        (m x * kernelNormalizer (laplaceKernel τ) p x / τ) x := by
    convert hLcert₀ using 1
    rw [hcommon x]
    ring
  have hLvalue :
      τ * (m'' x * kernelNormalizer (laplaceKernel τ) p x +
          2 * (m' x + 1) * laplaceKernelNormalizerRightDerivCoeff τ p x +
            m x * Z'' x) =
        m x * kernelNormalizer (laplaceKernel τ) p x / τ :=
    hLderiv_formula.unique hLcert
  exact laplaceCommonODE_value_of_companion_derivative
    (τ := τ) (m := m x) (m' := m' x) (m'' := m'' x)
    (Z := kernelNormalizer (laplaceKernel τ) p x)
    (Z' := laplaceKernelNormalizerRightDerivCoeff τ p x)
    (Z'' := Z'' x)
    (L' := τ * (m'' x * kernelNormalizer (laplaceKernel τ) p x +
          2 * (m' x + 1) * laplaceKernelNormalizerRightDerivCoeff τ p x +
            m x * Z'' x))
    hτ.ne' rfl hLvalue

/-- **L5 project-native wrapper.**  Suppose the common mean-shift relation
`D_p = m Z_p`, `D_q = m Z_q` holds and the raw Laplace normalizers have the
classical first/second derivative data needed at `x`.  Then the named
normalizer Wronskian satisfies the Abel equation

`W' = -(2 (m' + 1) / m) W`

at every point where `m x ≠ 0`.

This closes the L5 *ODE/Abel derivation* from explicit regularity data.  The
separate analytic task for the a.c. theorem is to prove that the hypotheses
below hold a.e. for absolutely-continuous laws with the required exponential
moments. -/
theorem hasDerivAt_laplaceKernelNormalizerWronskian_of_commonMeanShift
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (m m' m'' Zp'' Zq'' : ℝ → ℝ) (x : ℝ)
    (hm_all : ∀ t : ℝ, HasDerivAt m (m' t) t)
    (hm' : HasDerivAt m' (m'' x) x)
    (hZp_all : ∀ t : ℝ,
      HasDerivAt (fun s => kernelNormalizer (laplaceKernel τ) p s)
        (laplaceKernelNormalizerRightDerivCoeff τ p t) t)
    (hZq_all : ∀ t : ℝ,
      HasDerivAt (fun s => kernelNormalizer (laplaceKernel τ) q s)
        (laplaceKernelNormalizerRightDerivCoeff τ q t) t)
    (hZp' : HasDerivAt (laplaceKernelNormalizerRightDerivCoeff τ p) (Zp'' x) x)
    (hZq' : HasDerivAt (laplaceKernelNormalizerRightDerivCoeff τ q) (Zq'' x) x)
    (hpcommon :
      ∀ t : ℝ, (∫ y, laplaceWeightedDisplacement τ t y ∂p) =
        m t * kernelNormalizer (laplaceKernel τ) p t)
    (hqcommon :
      ∀ t : ℝ, (∫ y, laplaceWeightedDisplacement τ t y ∂q) =
        m t * kernelNormalizer (laplaceKernel τ) q t)
    (hmx : m x ≠ 0) :
    HasDerivAt (fun t => laplaceKernelNormalizerWronskian τ p q t)
      (-(2 * (m' x + 1) / m x) *
        laplaceKernelNormalizerWronskian τ p q x) x := by
  have hODEp := laplaceCommonODE_of_commonMeanShift_at τ hτ p m m' m'' Zp'' x
    hm_all hm' hZp_all hZp' hpcommon
  have hODEq := laplaceCommonODE_of_commonMeanShift_at τ hτ q m m' m'' Zq'' x
    hm_all hm' hZq_all hZq' hqcommon
  have hW := hasDerivAt_wronskian_of_laplace_commonODE
    (m := m) (mu' := fun t => m' t + 1)
    (b := fun t => m'' t - m t / τ ^ 2)
    (f := fun t => kernelNormalizer (laplaceKernel τ) p t)
    (f' := laplaceKernelNormalizerRightDerivCoeff τ p)
    (g := fun t => kernelNormalizer (laplaceKernel τ) q t)
    (g' := laplaceKernelNormalizerRightDerivCoeff τ q)
    (x := x)
    (f'' := Zp'' x) (g'' := Zq'' x)
    (hZp_all x) hZp' (hZq_all x) hZq' hmx
    (by simpa [mul_assoc] using hODEp)
    (by simpa [mul_assoc] using hODEq)
  simpa [scalarWronskian, laplaceKernelNormalizerWronskian, mul_div_assoc] using hW

/-- **L5 zero-drift form.**  Under zero raw Laplace drift, use the natural common
mean-shift ratio `m = D_p/Z_p`.  If this ratio and the raw normalizers have the
classical derivative data required at `x`, then the named normalizer Wronskian
obeys Abel's equation there:

`W' = -(2 (m' + 1) / m) W`.

This is the form meant to be fed by the later a.c. regularity wrapper. -/
theorem hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (m' m'' Zp'' Zq'' : ℝ → ℝ) (x : ℝ)
    (hm_all : ∀ t : ℝ, HasDerivAt (laplaceMeanShiftRatio τ p) (m' t) t)
    (hm' : HasDerivAt m' (m'' x) x)
    (hZp_all : ∀ t : ℝ,
      HasDerivAt (fun s => kernelNormalizer (laplaceKernel τ) p s)
        (laplaceKernelNormalizerRightDerivCoeff τ p t) t)
    (hZq_all : ∀ t : ℝ,
      HasDerivAt (fun s => kernelNormalizer (laplaceKernel τ) q s)
        (laplaceKernelNormalizerRightDerivCoeff τ q t) t)
    (hZp' : HasDerivAt (laplaceKernelNormalizerRightDerivCoeff τ p) (Zp'' x) x)
    (hZq' : HasDerivAt (laplaceKernelNormalizerRightDerivCoeff τ q) (Zq'' x) x)
    (hmx : laplaceMeanShiftRatio τ p x ≠ 0) :
    HasDerivAt (fun t => laplaceKernelNormalizerWronskian τ p q t)
      (-(2 * (m' x + 1) / laplaceMeanShiftRatio τ p x) *
        laplaceKernelNormalizerWronskian τ p q x) x := by
  refine hasDerivAt_laplaceKernelNormalizerWronskian_of_commonMeanShift
    τ hτ p q (laplaceMeanShiftRatio τ p) m' m'' Zp'' Zq'' x
    hm_all hm' hZp_all hZq_all hZp' hZq' ?_ ?_ hmx
  · intro t
    exact laplaceMeanShiftRatio_common_self τ hτ p t
  · intro t
    exact laplaceMeanShiftRatio_common_of_zeroDrift τ hτ p q hzero t

/-- A.e. L5 wrapper.  Once the remaining a.c. regularity work supplies the
second-derivative data almost everywhere, the zero-drift Abel identity follows
almost everywhere on the nonzero-mean-shift locus. -/
theorem ae_hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (μ : Measure ℝ) (m' m'' Zp'' Zq'' : ℝ → ℝ)
    (hm_all : ∀ t : ℝ, HasDerivAt (laplaceMeanShiftRatio τ p) (m' t) t)
    (hZp_all : ∀ t : ℝ,
      HasDerivAt (fun s => kernelNormalizer (laplaceKernel τ) p s)
        (laplaceKernelNormalizerRightDerivCoeff τ p t) t)
    (hZq_all : ∀ t : ℝ,
      HasDerivAt (fun s => kernelNormalizer (laplaceKernel τ) q s)
        (laplaceKernelNormalizerRightDerivCoeff τ q t) t)
    (hsecond : ∀ᵐ x ∂μ,
      HasDerivAt m' (m'' x) x ∧
        HasDerivAt (laplaceKernelNormalizerRightDerivCoeff τ p) (Zp'' x) x ∧
          HasDerivAt (laplaceKernelNormalizerRightDerivCoeff τ q) (Zq'' x) x) :
    ∀ᵐ x ∂μ,
      laplaceMeanShiftRatio τ p x ≠ 0 →
        HasDerivAt (fun t => laplaceKernelNormalizerWronskian τ p q t)
          (-(2 * (m' x + 1) / laplaceMeanShiftRatio τ p x) *
            laplaceKernelNormalizerWronskian τ p q x) x := by
  filter_upwards [hsecond] with x hx hmx
  rcases hx with ⟨hm', hZp', hZq'⟩
  exact hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift
    τ hτ p q hzero m' m'' Zp'' Zq'' x
    hm_all hm' hZp_all hZq_all hZp' hZq' hmx

end DriftingIdentifiability
