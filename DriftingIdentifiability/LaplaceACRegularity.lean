import DriftingIdentifiability.LaplaceACAbel

/-!
# L5 regularity discharge for the a.c. Laplace converse

`LaplaceACAbel.lean` proves the Abel/Wronskian algebra from explicit derivative
facts.  This file packages the smallest honest regularity certificate needed to
construct those facts for the actual Laplace normalizers.

The certificate is intentionally analytic rather than axiomatic: it records that
the raw Laplace normalizer has the classical first derivative named by the
project (`laplaceKernelNormalizerRightDerivCoeff`) and that this derivative has a
second derivative.  From this, together with the already-certified first-order
Laplace identities

* `L_p' = D_p / τ`, and
* `D_p' = L_p / τ - 2 Z_p`,

we derive the `m'`/`m''` data for the common ratio `m = D_p/Z_p`, and then feed
the existing Abel bridge.  No new axiom, `sorry`, or opaque constant is used.

Analytically, continuous-density hypotheses are the concrete sufficient
condition for this certificate via the one-dimensional Green identity
`Z'' = (Z - 2τ f)/τ²`; `LaplaceACDensityRegularity.lean` formalizes that
density-to-certificate theorem.
-/

open MeasureTheory Set Filter Topology

namespace DriftingIdentifiability

open Paper

/-! ## Explicit C²-normalizer regularity certificate -/

/-- A classical C² certificate for the one-dimensional Laplace normalizer of a
measure.

The first field says the two-sided derivative of `Z_p(x) = ∫ kτ(x,y) dp(y)` is
the project-native coefficient `laplaceKernelNormalizerRightDerivCoeff`.  The
second field says that named coefficient is itself differentiable, with value
`secondDeriv`.

This is a *regularity class*, not an axiom.  The separate module
`LaplaceACDensityRegularity.lean` proves it from continuous nonnegative density
representations; exponential moments are used elsewhere for tail/compactness
arguments rather than for this local C² certificate. -/
structure LaplaceC2NormalizerRegular (τ : ℝ) (p : Measure ℝ) where
  secondDeriv : ℝ → ℝ
  hasDerivAt_normalizer :
    ∀ x : ℝ,
      HasDerivAt (fun s => kernelNormalizer (laplaceKernel τ) p s)
        (laplaceKernelNormalizerRightDerivCoeff τ p x) x
  hasDerivAt_rightDerivCoeff :
    ∀ x : ℝ,
      HasDerivAt (laplaceKernelNormalizerRightDerivCoeff τ p) (secondDeriv x) x

/-! ## Derived derivative data for the mean-shift ratio -/

/-- Project-native notation for the Laplace mean-shift numerator
`D_p(x) = ∫ kτ(x,y)(y-x) dp(y)`. -/
noncomputable def laplaceMeanShiftNumerator
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  ∫ y, laplaceWeightedDisplacement τ x y ∂p

/-- The certified first derivative of the Laplace mean-shift numerator. -/
noncomputable def laplaceMeanShiftNumeratorDeriv
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (1 / τ) * kernelNormalizer (laplaceCompanionKernel τ) p x -
    2 * kernelNormalizer (laplaceKernel τ) p x

/-- The derivative of the certified numerator derivative, once the normalizer
has the classical first derivative supplied by `LaplaceC2NormalizerRegular`. -/
noncomputable def laplaceMeanShiftNumeratorSecondDeriv
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (1 / τ) * ((1 / τ) * laplaceMeanShiftNumerator τ p x) -
    2 * laplaceKernelNormalizerRightDerivCoeff τ p x

/-- Numerator of the quotient-rule formula for `(D/Z)'`. -/
noncomputable def laplaceMeanShiftRatioDerivNumerator
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  laplaceMeanShiftNumeratorDeriv τ p x *
      kernelNormalizer (laplaceKernel τ) p x -
    laplaceMeanShiftNumerator τ p x *
      laplaceKernelNormalizerRightDerivCoeff τ p x

/-- Denominator of the quotient-rule formula for `(D/Z)'`. -/
noncomputable def laplaceMeanShiftRatioDerivDenominator
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (kernelNormalizer (laplaceKernel τ) p x) ^ 2

/-- The derivative value for `laplaceMeanShiftRatio τ p = D_p/Z_p`. -/
noncomputable def laplaceMeanShiftRatioDeriv
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  laplaceMeanShiftRatioDerivNumerator τ p x /
    laplaceMeanShiftRatioDerivDenominator τ p x

/-- Derivative of the quotient numerator used in `laplaceMeanShiftRatioDeriv`. -/
noncomputable def laplaceMeanShiftRatioDerivNumeratorDeriv
    (τ : ℝ) (p : Measure ℝ) (Z'' : ℝ → ℝ) (x : ℝ) : ℝ :=
  (laplaceMeanShiftNumeratorSecondDeriv τ p x *
      kernelNormalizer (laplaceKernel τ) p x +
    laplaceMeanShiftNumeratorDeriv τ p x *
      laplaceKernelNormalizerRightDerivCoeff τ p x) -
    (laplaceMeanShiftNumeratorDeriv τ p x *
      laplaceKernelNormalizerRightDerivCoeff τ p x +
    laplaceMeanShiftNumerator τ p x * Z'' x)

/-- Derivative of the quotient denominator `Z^2`. -/
noncomputable def laplaceMeanShiftRatioDerivDenominatorDeriv
    (τ : ℝ) (p : Measure ℝ) (x : ℝ) : ℝ :=
  (2 : ℝ) * (kernelNormalizer (laplaceKernel τ) p x) ^ (2 - 1) *
    laplaceKernelNormalizerRightDerivCoeff τ p x

/-- The second derivative value for the common mean-shift ratio, expressed only
in terms of certified first-order identities and the normalizer C² certificate. -/
noncomputable def laplaceMeanShiftRatioSecondDeriv
    (τ : ℝ) (p : Measure ℝ) (Z'' : ℝ → ℝ) (x : ℝ) : ℝ :=
  (laplaceMeanShiftRatioDerivNumeratorDeriv τ p Z'' x *
      laplaceMeanShiftRatioDerivDenominator τ p x -
    laplaceMeanShiftRatioDerivNumerator τ p x *
      laplaceMeanShiftRatioDerivDenominatorDeriv τ p x) /
    (laplaceMeanShiftRatioDerivDenominator τ p x) ^ 2

theorem hasDerivAt_laplaceMeanShiftNumerator
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (x : ℝ) :
    HasDerivAt (laplaceMeanShiftNumerator τ p)
      (laplaceMeanShiftNumeratorDeriv τ p x) x := by
  change HasDerivAt (fun z : ℝ => ∫ y, laplaceWeightedDisplacement τ z y ∂p)
    ((1 / τ) * kernelNormalizer (laplaceCompanionKernel τ) p x -
      2 * kernelNormalizer (laplaceKernel τ) p x) x
  exact hasDerivAt_laplaceDisplacementIntegral τ hτ p x

theorem hasDerivAt_laplaceMeanShiftNumeratorDeriv
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hp : LaplaceC2NormalizerRegular τ p) (x : ℝ) :
    HasDerivAt (laplaceMeanShiftNumeratorDeriv τ p)
      (laplaceMeanShiftNumeratorSecondDeriv τ p x) x := by
  have hL := hasDerivAt_laplaceCompanionNormalizer τ hτ p x
  have hZ := hp.hasDerivAt_normalizer x
  have hLscaled : HasDerivAt
      (fun t : ℝ => (1 / τ) * kernelNormalizer (laplaceCompanionKernel τ) p t)
      ((1 / τ) * ((1 / τ) * laplaceMeanShiftNumerator τ p x)) x := by
    simpa [laplaceMeanShiftNumerator, mul_assoc] using
      hL.const_mul (1 / τ)
  have hZscaled : HasDerivAt
      (fun t : ℝ => 2 * kernelNormalizer (laplaceKernel τ) p t)
      (2 * laplaceKernelNormalizerRightDerivCoeff τ p x) x :=
    hZ.const_mul 2
  change HasDerivAt
    (fun t : ℝ =>
      (1 / τ) * kernelNormalizer (laplaceCompanionKernel τ) p t -
        2 * kernelNormalizer (laplaceKernel τ) p t)
    ((1 / τ) * ((1 / τ) * laplaceMeanShiftNumerator τ p x) -
      2 * laplaceKernelNormalizerRightDerivCoeff τ p x) x
  exact hLscaled.sub hZscaled

theorem hasDerivAt_laplaceMeanShiftRatio_regular
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hp : LaplaceC2NormalizerRegular τ p) (x : ℝ) :
    HasDerivAt (laplaceMeanShiftRatio τ p)
      (laplaceMeanShiftRatioDeriv τ p x) x := by
  have hD := hasDerivAt_laplaceMeanShiftNumerator τ hτ p x
  have hZ := hp.hasDerivAt_normalizer x
  have hZne : kernelNormalizer (laplaceKernel τ) p x ≠ 0 :=
    (laplaceKernelNormalizer_pos p τ hτ x).ne'
  have hdiv := hD.div hZ hZne
  change HasDerivAt
    (laplaceMeanShiftNumerator τ p /
      fun s : ℝ => kernelNormalizer (laplaceKernel τ) p s)
    (laplaceMeanShiftRatioDeriv τ p x) x
  simpa [laplaceMeanShiftRatioDeriv, laplaceMeanShiftRatioDerivNumerator,
    laplaceMeanShiftRatioDerivDenominator, laplaceMeanShiftNumerator] using hdiv

/-- Derivative formula for the usual tilted mean, derived from the C²
normalizer certificate and the identity
`laplaceTiltedMean = x + laplaceMeanShiftRatio`.

This is the small L3 wiring lemma needed by the final a.c. assemblies: the
derivative of the tilted mean is exactly `m' + 1`, where
`m = laplaceMeanShiftRatio`. -/
theorem hasDerivAt_laplaceTiltedMean_regular
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hp : LaplaceC2NormalizerRegular τ p) (x : ℝ) :
    HasDerivAt (laplaceTiltedMean τ p)
      (laplaceMeanShiftRatioDeriv τ p x + 1) x := by
  have hratio := hasDerivAt_laplaceMeanShiftRatio_regular τ hτ p hp x
  have hdisp : HasDerivAt (laplaceTiltedMeanFromDisplacement τ p)
      (1 + laplaceMeanShiftRatioDeriv τ p x) x := by
    have hsum : HasDerivAt (fun t : ℝ => t + laplaceMeanShiftRatio τ p t)
        (1 + laplaceMeanShiftRatioDeriv τ p x) x :=
      (hasDerivAt_id x).add hratio
    change HasDerivAt
      (fun t : ℝ => t + (∫ y, laplaceWeightedDisplacement τ t y ∂p) /
        kernelNormalizer (laplaceKernel τ) p t)
      (1 + laplaceMeanShiftRatioDeriv τ p x) x
    simpa [laplaceMeanShiftRatio] using hsum
  have heq : laplaceTiltedMean τ p = laplaceTiltedMeanFromDisplacement τ p := by
    funext y
    exact laplaceTiltedMean_eq_fromDisplacement τ hτ p y
  simpa [heq, add_comm] using hdisp

/-- **L3 bridge for final assembly.**  The already-certified monotonicity of the
Laplace tilted mean implies the nonnegativity hypothesis consumed by the L6/L8
assembly theorems.

The extra integrability assumption is exactly the one used by
`laplaceTiltedMean_monotone`; smooth density / exponential-moment callers can
discharge it upstream. -/
theorem laplaceMeanShiftRatioDeriv_add_one_nonneg_of_regular
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hp : LaplaceC2NormalizerRegular τ p)
    (hint : Integrable (fun y : ℝ => y) p) :
    ∀ t : ℝ, 0 ≤ laplaceMeanShiftRatioDeriv τ p t + 1 := by
  intro t
  exact (hasDerivAt_laplaceTiltedMean_regular τ hτ p hp t).nonneg_of_monotone
    (laplaceTiltedMean_monotone hτ p hint)

/-- **Strict L7 bridge.**  At a point where the law has positive mass on both
sides, the C² regularity derivative of the usual tilted mean,
`laplaceMeanShiftRatioDeriv + 1`, is strictly positive.

This converts the already-proved right-derivative/two-sided-mass L7 theorem into
the exact coefficient used by the Abel propagation layer. -/
theorem laplaceMeanShiftRatioDeriv_add_one_pos_of_twoSidedMass_regular
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hp : LaplaceC2NormalizerRegular τ p) (x : ℝ)
    (hleft : 0 < p (Set.Iio x)) (hright : 0 < p (Set.Ioi x)) :
    0 < laplaceMeanShiftRatioDeriv τ p x + 1 := by
  have hcoeff_pos :
      0 < laplaceTiltedMeanRightDerivCoeff τ p x :=
    laplaceTiltedMeanRightDerivCoeff_pos_of_twoSidedMass τ hτ p x hleft hright
  have hregular :
      HasDerivWithinAt (laplaceTiltedMean τ p)
        (laplaceMeanShiftRatioDeriv τ p x + 1) (Set.Ici x) x :=
    (hasDerivAt_laplaceTiltedMean_regular τ hτ p hp x).hasDerivWithinAt
  have hrightDeriv :
      HasDerivWithinAt (laplaceTiltedMean τ p)
        (laplaceTiltedMeanRightDerivCoeff τ p x) (Set.Ici x) x := by
    have hdisp :=
      hasDerivWithinAt_Ici_laplaceTiltedMeanFromDisplacement τ hτ p x
    have heq : laplaceTiltedMean τ p = laplaceTiltedMeanFromDisplacement τ p := by
      funext y
      exact laplaceTiltedMean_eq_fromDisplacement τ hτ p y
    simpa [heq] using hdisp
  have hEq :
      laplaceMeanShiftRatioDeriv τ p x + 1 =
        laplaceTiltedMeanRightDerivCoeff τ p x :=
    by
      have h₁ := hregular.derivWithin (uniqueDiffWithinAt_Ici x)
      have h₂ := hrightDeriv.derivWithin (uniqueDiffWithinAt_Ici x)
      exact h₁.symm.trans h₂
  rwa [hEq]

theorem hasDerivAt_laplaceMeanShiftRatioDeriv_regular
    (τ : ℝ) (hτ : ValidBandwidth τ) (p : Measure ℝ)
    [IsProbabilityMeasure p] (hp : LaplaceC2NormalizerRegular τ p) (x : ℝ) :
    HasDerivAt (laplaceMeanShiftRatioDeriv τ p)
      (laplaceMeanShiftRatioSecondDeriv τ p hp.secondDeriv x) x := by
  have hD := hasDerivAt_laplaceMeanShiftNumerator τ hτ p x
  have hD' := hasDerivAt_laplaceMeanShiftNumeratorDeriv τ hτ p hp x
  have hZ := hp.hasDerivAt_normalizer x
  have hZ' := hp.hasDerivAt_rightDerivCoeff x
  have hN₁ := hD'.mul hZ
  have hN₂ := hD.mul hZ'
  have hN : HasDerivAt (laplaceMeanShiftRatioDerivNumerator τ p)
      (laplaceMeanShiftRatioDerivNumeratorDeriv τ p hp.secondDeriv x) x := by
    change HasDerivAt
      ((laplaceMeanShiftNumeratorDeriv τ p *
          fun s : ℝ => kernelNormalizer (laplaceKernel τ) p s) -
        laplaceMeanShiftNumerator τ p *
          laplaceKernelNormalizerRightDerivCoeff τ p)
      (laplaceMeanShiftRatioDerivNumeratorDeriv τ p hp.secondDeriv x) x
    simpa [laplaceMeanShiftRatioDerivNumeratorDeriv] using hN₁.sub hN₂
  have hDen : HasDerivAt (laplaceMeanShiftRatioDerivDenominator τ p)
      (laplaceMeanShiftRatioDerivDenominatorDeriv τ p x) x := by
    change HasDerivAt
      ((fun s : ℝ => kernelNormalizer (laplaceKernel τ) p s) ^ 2)
      (laplaceMeanShiftRatioDerivDenominatorDeriv τ p x) x
    simpa [laplaceMeanShiftRatioDerivDenominatorDeriv] using hZ.pow 2
  have hDen_ne : laplaceMeanShiftRatioDerivDenominator τ p x ≠ 0 := by
    unfold laplaceMeanShiftRatioDerivDenominator
    exact pow_ne_zero 2 (laplaceKernelNormalizer_pos p τ hτ x).ne'
  have hdiv := hN.div hDen hDen_ne
  change HasDerivAt
    (laplaceMeanShiftRatioDerivNumerator τ p /
      laplaceMeanShiftRatioDerivDenominator τ p)
    (laplaceMeanShiftRatioSecondDeriv τ p hp.secondDeriv x) x
  simpa [laplaceMeanShiftRatioSecondDeriv] using hdiv

/-! ## No-leftover-hypothesis L5 Abel theorem -/

/-- **L5 regularity-discharge theorem.**  Under zero raw Laplace drift and C²
normalizer regularity for `p` and `q`, the actual normalizer Wronskian satisfies
Abel's equation at every point where the common mean-shift ratio is nonzero.

Unlike the lower-level theorem in `LaplaceACAbel.lean`, this statement contains
no exposed `HasDerivAt` hypotheses for `m`, `Z_p`, or `Z_q`; they are constructed
from `LaplaceC2NormalizerRegular` plus the certified first-order Laplace
identities. -/
theorem hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift_regular
    (τ : ℝ) (hτ : ValidBandwidth τ) (p q : Measure ℝ)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hp : LaplaceC2NormalizerRegular τ p)
    (hq : LaplaceC2NormalizerRegular τ q)
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (x : ℝ)
    (hmx : laplaceMeanShiftRatio τ p x ≠ 0) :
    HasDerivAt (fun t => laplaceKernelNormalizerWronskian τ p q t)
      (-(2 * (laplaceMeanShiftRatioDeriv τ p x + 1) /
          laplaceMeanShiftRatio τ p x) *
        laplaceKernelNormalizerWronskian τ p q x) x := by
  exact hasDerivAt_laplaceKernelNormalizerWronskian_of_zeroDrift
    τ hτ p q hzero
    (laplaceMeanShiftRatioDeriv τ p)
    (laplaceMeanShiftRatioSecondDeriv τ p hp.secondDeriv)
    hp.secondDeriv hq.secondDeriv x
    (fun t => hasDerivAt_laplaceMeanShiftRatio_regular τ hτ p hp t)
    (hasDerivAt_laplaceMeanShiftRatioDeriv_regular τ hτ p hp x)
    (fun t => hp.hasDerivAt_normalizer t)
    (fun t => hq.hasDerivAt_normalizer t)
    (hp.hasDerivAt_rightDerivCoeff x)
    (hq.hasDerivAt_rightDerivCoeff x)
    hmx

end DriftingIdentifiability
