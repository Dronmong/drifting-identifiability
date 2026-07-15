import DriftingIdentifiability.LaplaceEuclideanInjectivity
import Mathlib.Analysis.SpecialFunctions.Gaussian.GaussianIntegral
import Mathlib.Analysis.Calculus.ParametricIntegral

/-!
# The radial Laplace profile has a positive Fourier transform (L4 crux)

The higher-dimensional Laplace paper kernel on `EuclideanSpace ℝ ι` is the radial
profile `x ↦ e^{-‖x‖/τ}`.  `LaplaceEuclideanInjectivity.lean` reduces the
Euclidean smoothing-injectivity theorem to the single analytic fact that this
profile has a nowhere-zero Fourier transform (`hbase_ne`).

This file proves that fact by **Gaussian subordination**.  The engine is the
Glasser-type integral

`∫_{(0,∞)} e^{-x² - k²/x²} dx = (√π/2) e^{-2k}`   (`glasser_integral`)

proved via the ODE `F'(k) = -2 F(k)` (differentiation under the integral plus the
self-reciprocal substitution `x ↦ k/x`).  It yields the subordination identity

`e^{-a} = (2/√π) ∫_{(0,∞)} e^{-s²} e^{-a²/(4s²)} ds`   (`exp_neg_eq_subordination`)

writing the exponential as a *positive* mixture of Gaussians; Tonelli against the
positive Gaussian Fourier transform then makes the profile's transform a positive
integral.
-/

open MeasureTheory Real Set Filter Topology intervalIntegral
open scoped FourierTransform RealInnerProductSpace

namespace DriftingIdentifiability

/-! ## Reciprocal-Gaussian integrability, via the substitution `x ↦ 1/x` -/

/-- `∫_{(0,∞)} x⁻² e^{-c/x²} dx = √(π/c)/2` for `c > 0`, by the substitution
`x ↦ 1/x` reducing it to a Gaussian. -/
lemma integral_inv_sq_exp_neg_div_sq {c : ℝ} (hc : 0 < c) :
    ∫ x in Ioi (0 : ℝ), x ^ (-2 : ℤ) * Real.exp (-c / x ^ 2) = Real.sqrt (π / c) / 2 := by
  have h := integral_comp_rpow_Ioi (fun y : ℝ => Real.exp (-c * y ^ 2)) (p := -1)
    (by norm_num)
  rw [← integral_gaussian_Ioi c]
  rw [← h]
  apply setIntegral_congr_fun measurableSet_Ioi
  intro x hx
  have hx0 : (0 : ℝ) < x := hx
  have hxne : x ≠ 0 := hx0.ne'
  simp only [smul_eq_mul, abs_neg, abs_one, one_mul]
  rw [Real.rpow_neg_one]
  have hxpow : x ^ ((-1 : ℝ) - 1) = x ^ (-2 : ℤ) := by
    rw [show ((-1 : ℝ) - 1) = ((-2 : ℤ) : ℝ) by norm_num, Real.rpow_intCast]
  rw [hxpow]
  have hinv : Real.exp (-c * (x⁻¹) ^ 2) = Real.exp (-c / x ^ 2) := by
    congr 1
    rw [inv_pow]
    field_simp
  rw [hinv]

/-- Integrability counterpart of `integral_inv_sq_exp_neg_div_sq`; this is the
reciprocal-Gaussian bound used for differentiating the Glasser integral under
the integral sign. -/
lemma integrableOn_inv_sq_exp_neg_div_sq {c : ℝ} (hc : 0 < c) :
    IntegrableOn (fun x : ℝ => x ^ (-2 : ℤ) * Real.exp (-c / x ^ 2))
      (Ioi (0 : ℝ)) := by
  have hiff :
      IntegrableOn
          (fun x : ℝ =>
            (|(-1 : ℝ)| * x ^ ((-1 : ℝ) - 1)) •
              Real.exp (-c * (x ^ (-1 : ℝ)) ^ 2))
          (Ioi (0 : ℝ)) ↔
        IntegrableOn (fun y : ℝ => Real.exp (-c * y ^ 2)) (Ioi (0 : ℝ)) :=
    MeasureTheory.integrableOn_Ioi_comp_rpow_iff
      (E := ℝ) (fun y : ℝ => Real.exp (-c * y ^ 2)) (p := -1) (by norm_num)
  have hg : IntegrableOn (fun y : ℝ => Real.exp (-c * y ^ 2)) (Ioi (0 : ℝ)) :=
    integrableOn_Ioi_exp_neg_mul_sq_iff.mpr hc
  rw [← hiff] at hg
  refine hg.congr_fun ?_ measurableSet_Ioi
  intro x hx
  have hx0 : (0 : ℝ) < x := hx
  dsimp
  rw [Real.rpow_neg_one]
  have hxpow : x ^ ((-1 : ℝ) - 1) = x ^ (-2 : ℤ) := by
    rw [show ((-1 : ℝ) - 1) = ((-2 : ℤ) : ℝ) by norm_num, Real.rpow_intCast]
  rw [hxpow]
  simp only [abs_neg, abs_one, one_mul]
  congr 2
  rw [inv_pow]
  field_simp

/-! ## The Glasser integral and the self-reciprocal substitution -/

/-- The Glasser integrand `x ↦ e^{-x² - k²/x²}`. -/
noncomputable def glasserKernel (k x : ℝ) : ℝ := Real.exp (-x ^ 2 - k ^ 2 / x ^ 2)

/-- The Glasser integral over `(0,∞)`. -/
noncomputable def glasserIntegral (k : ℝ) : ℝ :=
  ∫ x in Ioi (0 : ℝ), glasserKernel k x

/-- Measurability of the Glasser kernel in the integration variable. -/
lemma measurable_glasserKernel (k : ℝ) : Measurable (fun x : ℝ => glasserKernel k x) := by
  unfold glasserKernel
  fun_prop

/-- Measurability of the parameter-derivative integrand. -/
lemma measurable_glasserKernel_paramDeriv (k : ℝ) :
    Measurable (fun x : ℝ => -2 * k * x ^ (-2 : ℤ) * glasserKernel k x) := by
  unfold glasserKernel
  fun_prop

/-- `x ↦ e^{-x²-k²/x²}` is integrable on `(0,∞)` for every `k`, dominated by
`e^{-x²}`. -/
lemma glasserKernel_integrableOn (k : ℝ) :
    IntegrableOn (glasserKernel k) (Ioi (0 : ℝ)) := by
  refine Integrable.mono' (g := fun x : ℝ => Real.exp (-1 * x ^ 2))
    (integrableOn_Ioi_exp_neg_mul_sq_iff.mpr one_pos) ?_ ?_
  · refine (Real.measurable_exp.comp ?_).aestronglyMeasurable
    exact (measurable_id.pow_const 2).neg.sub (measurable_const.div (measurable_id.pow_const 2))
  · filter_upwards [ae_restrict_mem measurableSet_Ioi] with x hx
    simp only [glasserKernel, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
    apply Real.exp_le_exp.mpr
    have hx0 : (0 : ℝ) < x := hx
    have : (0:ℝ) ≤ k ^ 2 / x ^ 2 := by positivity
    nlinarith [this]

/-- Pointwise derivative of the Glasser kernel with respect to the parameter
`k`, away from the endpoint `x = 0`. -/
lemma hasDerivAt_glasserKernel_param {k x : ℝ} (hx : x ≠ 0) :
    HasDerivAt (fun u : ℝ => glasserKernel u x)
      (-2 * k * x ^ (-2 : ℤ) * glasserKernel k x) k := by
  unfold glasserKernel
  have hpow : HasDerivAt (fun u : ℝ => u ^ 2) (2 * k) k := by
    simpa using (hasDerivAt_pow 2 k)
  have hdiv : HasDerivAt (fun u : ℝ => u ^ 2 / x ^ 2) ((2 * k) / x ^ 2) k :=
    hpow.div_const (x ^ 2)
  convert! (((hasDerivAt_const k (-x ^ 2)).sub hdiv).exp) using 1
  simp only [Pi.sub_apply]
  field_simp [hx]
  ring_nf

/-- Local reciprocal-Gaussian domination for the parameter derivative of the
Glasser kernel.  If the parameter `z` stays in `(k/2, 2k)`, then the derivative
is bounded by an integrable reciprocal Gaussian depending only on `k`. -/
lemma norm_glasserKernel_paramDeriv_le
    {k z x : ℝ} (hk : 0 < k) (hx : 0 < x) (hz : z ∈ Ioo (k / 2) (2 * k)) :
    ‖-2 * z * x ^ (-2 : ℤ) * glasserKernel z x‖ ≤
      4 * k * (x ^ (-2 : ℤ) * Real.exp (-((k / 2) ^ 2) / x ^ 2)) := by
  have hzpos : 0 < z := lt_trans (by linarith : 0 < k / 2) hz.1
  have hzle : z ≤ 2 * k := hz.2.le
  have hxne : x ≠ 0 := hx.ne'
  have hx2pos : 0 < x ^ 2 := sq_pos_of_ne_zero hxne
  have hxzpow_nonneg : 0 ≤ x ^ (-2 : ℤ) := by positivity
  have hker_pos : 0 < glasserKernel z x := by
    unfold glasserKernel
    positivity
  have hker_nonneg : 0 ≤ glasserKernel z x := hker_pos.le
  rw [Real.norm_eq_abs, abs_of_nonpos]
  · have hker_le :
        glasserKernel z x ≤ Real.exp (-((k / 2) ^ 2) / x ^ 2) := by
      unfold glasserKernel
      apply Real.exp_le_exp.mpr
      have hleft_nonneg : 0 ≤ k / 2 := by positivity
      have hzsq : (k / 2) ^ 2 ≤ z ^ 2 := by
        exact sq_le_sq.mpr (by
          rw [abs_of_nonneg hleft_nonneg, abs_of_nonneg hzpos.le]
          exact hz.1.le)
      have hdiv : (k / 2) ^ 2 / x ^ 2 ≤ z ^ 2 / x ^ 2 :=
        div_le_div_of_nonneg_right hzsq hx2pos.le
      calc
        -x ^ 2 - z ^ 2 / x ^ 2 ≤ 0 - z ^ 2 / x ^ 2 := by
          nlinarith [sq_nonneg x]
        _ ≤ 0 - (k / 2) ^ 2 / x ^ 2 := by
          linarith
        _ = -((k / 2) ^ 2) / x ^ 2 := by ring
    calc
      -(-2 * z * x ^ (-2 : ℤ) * glasserKernel z x)
          = 2 * z * x ^ (-2 : ℤ) * glasserKernel z x := by ring
      _ ≤ 2 * z * x ^ (-2 : ℤ) *
            Real.exp (-((k / 2) ^ 2) / x ^ 2) := by
            gcongr
      _ ≤ 4 * k * (x ^ (-2 : ℤ) *
            Real.exp (-((k / 2) ^ 2) / x ^ 2)) := by
            have hfactor :
                0 ≤ x ^ (-2 : ℤ) * Real.exp (-((k / 2) ^ 2) / x ^ 2) := by
              positivity
            nlinarith
  · have hnonneg : 0 ≤ 2 * z * x ^ (-2 : ℤ) * glasserKernel z x := by positivity
    nlinarith

private lemma glasserKernel_mul_inv_eq_self {k x : ℝ} (hk : k ≠ 0) (hx : x ≠ 0) :
    glasserKernel k (k * x⁻¹) = glasserKernel k x := by
  unfold glasserKernel
  congr 1
  field_simp [hk, hx]
  ring

/-- The self-reciprocal substitution behind the Glasser integral:

`∫ x⁻² e^{-x²-k²/x²} dx = k⁻¹ ∫ e^{-x²-k²/x²} dx`.

Equivalently, after multiplying by `k`, the Glasser integral equals its
reciprocal-weighted version.  This is the first planned L4 step in
`LaplaceHigherDim.md`: substitute `y = k/x` using `integral_comp_rpow_Ioi`
at exponent `-1`, then scale by `k`. -/
lemma integral_inv_sq_mul_glasserKernel_eq_inv_mul_integral {k : ℝ} (hk : 0 < k) :
    ∫ x in Ioi (0 : ℝ), x ^ (-2 : ℤ) * glasserKernel k x =
      k⁻¹ * ∫ x in Ioi (0 : ℝ), glasserKernel k x := by
  have hkne : k ≠ 0 := hk.ne'
  have hsub :=
    integral_comp_rpow_Ioi (fun y : ℝ => glasserKernel k (k * y)) (p := -1)
      (by norm_num)
  calc
    ∫ x in Ioi (0 : ℝ), x ^ (-2 : ℤ) * glasserKernel k x
        = ∫ x in Ioi (0 : ℝ),
            (|(-1 : ℝ)| * x ^ ((-1 : ℝ) - 1)) •
              glasserKernel k (k * x ^ (-1 : ℝ)) := by
          apply setIntegral_congr_fun measurableSet_Ioi
          intro x hx
          have hx0 : (0 : ℝ) < x := hx
          have hxne : x ≠ 0 := hx0.ne'
          dsimp
          have hxpow : x ^ ((-1 : ℝ) - 1) = x ^ (-2 : ℤ) := by
            rw [show ((-1 : ℝ) - 1) = ((-2 : ℤ) : ℝ) by norm_num,
              Real.rpow_intCast]
          rw [Real.rpow_neg_one, hxpow]
          simp only [abs_neg, abs_one, one_mul]
          rw [glasserKernel_mul_inv_eq_self (k := k) (x := x) hkne hxne]
    _ = ∫ y in Ioi (0 : ℝ), glasserKernel k (k * y) := by
          simpa only [abs_neg, abs_one, one_mul] using hsub
    _ = k⁻¹ * ∫ x in Ioi (0 : ℝ), glasserKernel k x := by
          simpa [smul_eq_mul, mul_zero] using
            (integral_comp_mul_left_Ioi (glasserKernel k) (0 : ℝ) hk)

/-- Multiplicative form of the self-reciprocal substitution:
`∫ e^{-x²-k²/x²} dx = k ∫ x⁻² e^{-x²-k²/x²} dx`. -/
lemma glasserIntegral_eq_mul_integral_inv_sq_mul {k : ℝ} (hk : 0 < k) :
    ∫ x in Ioi (0 : ℝ), glasserKernel k x =
      k * ∫ x in Ioi (0 : ℝ), x ^ (-2 : ℤ) * glasserKernel k x := by
  rw [integral_inv_sq_mul_glasserKernel_eq_inv_mul_integral hk]
  field_simp [hk.ne']

/-- The Glasser integral satisfies the ODE `F'(k) = -2 F(k)` for `k > 0`.

This is the second planned L4 step: the derivative under the integral is
dominated on a local parameter window by the reciprocal-Gaussian integrable
bound, and the self-reciprocal substitution collapses
`∫ x⁻² e^{-x²-k²/x²}` back to `k⁻¹ F(k)`. -/
lemma hasDerivAt_glasserIntegral {k : ℝ} (hk : 0 < k) :
    HasDerivAt glasserIntegral (-2 * glasserIntegral k) k := by
  have hs : Ioo (k / 2) (2 * k) ∈ 𝓝 k := by
    exact Ioo_mem_nhds (by linarith) (by linarith)
  have hc : 0 < (k / 2) ^ 2 := by positivity
  have hbound_int :
      Integrable
        (fun x : ℝ =>
          4 * k * (x ^ (-2 : ℤ) * Real.exp (-((k / 2) ^ 2) / x ^ 2)))
        (volume.restrict (Ioi (0 : ℝ))) := by
    simpa [mul_assoc] using
      (integrableOn_inv_sq_exp_neg_div_sq hc).const_mul (4 * k)
  have key := (hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (F := fun z x => glasserKernel z x)
    (F' := fun z x => -2 * z * x ^ (-2 : ℤ) * glasserKernel z x)
    (x₀ := k)
    (bound := fun x : ℝ =>
      4 * k * (x ^ (-2 : ℤ) * Real.exp (-((k / 2) ^ 2) / x ^ 2)))
    (μ := volume.restrict (Ioi (0 : ℝ)))
    (s := Ioo (k / 2) (2 * k)) hs
    ?_ ?_ ?_ ?_ hbound_int ?_).2
  · have hvalue :
        (∫ x in Ioi (0 : ℝ), -2 * k * x ^ (-2 : ℤ) * glasserKernel k x)
          = -2 * glasserIntegral k := by
      unfold glasserIntegral
      calc
        (∫ x in Ioi (0 : ℝ), -2 * k * x ^ (-2 : ℤ) * glasserKernel k x)
            = (-2 * k) *
                ∫ x in Ioi (0 : ℝ), x ^ (-2 : ℤ) * glasserKernel k x := by
              rw [← MeasureTheory.integral_const_mul]
              congr 1
              funext x
              ring
        _ = (-2 * k) * (k⁻¹ * ∫ x in Ioi (0 : ℝ), glasserKernel k x) := by
              rw [integral_inv_sq_mul_glasserKernel_eq_inv_mul_integral hk]
        _ = -2 * ∫ x in Ioi (0 : ℝ), glasserKernel k x := by
              field_simp [hk.ne']
    change HasDerivAt glasserIntegral
      (∫ x in Ioi (0 : ℝ), -2 * k * x ^ (-2 : ℤ) * glasserKernel k x) k at key
    rw [hvalue] at key
    exact key
  · filter_upwards with z
    exact (measurable_glasserKernel z).aestronglyMeasurable
  · simpa [IntegrableOn] using (glasserKernel_integrableOn k)
  · exact (measurable_glasserKernel_paramDeriv k).aestronglyMeasurable
  · filter_upwards [ae_restrict_mem measurableSet_Ioi] with x hx z hz
    exact norm_glasserKernel_paramDeriv_le hk hx hz
  · filter_upwards [ae_restrict_mem measurableSet_Ioi] with x hx z hz
    exact hasDerivAt_glasserKernel_param hx.ne'

/-- Endpoint value of the Glasser integral. -/
lemma glasserIntegral_zero :
    glasserIntegral 0 = Real.sqrt π / 2 := by
  unfold glasserIntegral glasserKernel
  have hfun :
      (fun x : ℝ => Real.exp (-x ^ 2 - 0 ^ 2 / x ^ 2)) =
        fun x : ℝ => Real.exp (-1 * x ^ 2) := by
    funext x
    congr 1
    ring
  rw [hfun]
  rw [integral_gaussian_Ioi 1]
  simp

/-- The ODE-normalized Glasser integral. -/
noncomputable def glasserScaled (k : ℝ) : ℝ :=
  glasserIntegral k * Real.exp (2 * k)

/-- The scaled Glasser integral has zero derivative on `(0,∞)`. -/
lemma hasDerivAt_glasserScaled {k : ℝ} (hk : 0 < k) :
    HasDerivAt glasserScaled 0 k := by
  unfold glasserScaled
  have hF := hasDerivAt_glasserIntegral hk
  have hE : HasDerivAt (fun z : ℝ => Real.exp (2 * z)) (2 * Real.exp (2 * k)) k := by
    simpa [mul_comm] using ((hasDerivAt_id k).mul_const 2).exp
  have h := hF.mul hE
  change HasDerivAt (glasserIntegral * fun z : ℝ => Real.exp (2 * z)) 0 k
  have hzero :
      -(2 * glasserIntegral k * Real.exp (2 * k)) +
          glasserIntegral k * (2 * Real.exp (2 * k)) = 0 := by
    ring
  simpa [hzero] using h

/-- `glasserIntegral k * exp (2k)` is constant on the positive half-line. -/
lemma glasserScaled_eq_of_pos {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
    glasserScaled a = glasserScaled b := by
  refine isOpen_Ioi.is_const_of_deriv_eq_zero isPreconnected_Ioi ?_ ?_ ha hb
  · intro x hx
    exact (hasDerivAt_glasserScaled hx).differentiableAt.differentiableWithinAt
  · intro x hx
    exact (hasDerivAt_glasserScaled hx).deriv

/-- Right-continuity of the Glasser integral at `0`, by dominated convergence
with dominating function `x ↦ e^{-x²}`. -/
lemma tendsto_glasserIntegral_nhdsWithin_zero :
    Tendsto glasserIntegral (𝓝[>] (0 : ℝ)) (𝓝 (glasserIntegral 0)) := by
  unfold glasserIntegral
  refine tendsto_integral_filter_of_dominated_convergence
    (μ := volume.restrict (Ioi (0 : ℝ)))
    (l := 𝓝[>] (0 : ℝ))
    (bound := fun x : ℝ => Real.exp (-1 * x ^ 2))
    ?h_meas ?h_bound ?h_int ?h_lim
  · exact Eventually.of_forall fun k =>
      (measurable_glasserKernel k).aestronglyMeasurable
  · exact Eventually.of_forall fun k =>
      (ae_restrict_iff' measurableSet_Ioi).2 (ae_of_all _ fun x hx => by
        unfold glasserKernel
        rw [Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
        apply Real.exp_le_exp.mpr
        have hnonneg : 0 ≤ k ^ 2 / x ^ 2 := by positivity
        nlinarith [hnonneg])
  · simpa [IntegrableOn] using
      (integrableOn_Ioi_exp_neg_mul_sq_iff.mpr one_pos :
        IntegrableOn (fun x : ℝ => Real.exp (-1 * x ^ 2)) (Ioi (0 : ℝ)))
  · refine (ae_restrict_iff' measurableSet_Ioi).2 (ae_of_all _ fun x hx => ?_)
    unfold glasserKernel
    have hid : Tendsto (fun k : ℝ => k) (𝓝[>] (0 : ℝ)) (𝓝 (0 : ℝ)) :=
      tendsto_nhdsWithin_of_tendsto_nhds tendsto_id
    have hkpow : Tendsto (fun k : ℝ => k ^ 2) (𝓝[>] (0 : ℝ)) (𝓝 ((0 : ℝ) ^ 2)) :=
      hid.pow 2
    have harg :
        Tendsto (fun k : ℝ => -x ^ 2 - k ^ 2 / x ^ 2)
          (𝓝[>] (0 : ℝ)) (𝓝 (-x ^ 2 - 0 ^ 2 / x ^ 2)) := by
      exact tendsto_const_nhds.sub (hkpow.div_const (x ^ 2))
    exact Real.continuous_exp.continuousAt.tendsto.comp harg

/-- The scaled Glasser integral tends to its endpoint value at `0+`. -/
lemma tendsto_glasserScaled_nhdsWithin_zero :
    Tendsto glasserScaled (𝓝[>] (0 : ℝ)) (𝓝 (Real.sqrt π / 2)) := by
  unfold glasserScaled
  have hF := tendsto_glasserIntegral_nhdsWithin_zero
  have hid : Tendsto (fun k : ℝ => k) (𝓝[>] (0 : ℝ)) (𝓝 (0 : ℝ)) :=
    tendsto_nhdsWithin_of_tendsto_nhds tendsto_id
  have hE : Tendsto (fun k : ℝ => Real.exp (2 * k)) (𝓝[>] (0 : ℝ)) (𝓝 1) := by
    have harg : Tendsto (fun k : ℝ => 2 * k) (𝓝[>] (0 : ℝ)) (𝓝 (2 * 0)) :=
      tendsto_const_nhds.mul hid
    simpa [Function.comp_def] using Real.continuous_exp.continuousAt.tendsto.comp harg
  simpa [glasserIntegral_zero] using hF.mul hE

/-- The Glasser integral closed form on the positive half-line. -/
lemma glasserIntegral_eq_closed_of_pos {k : ℝ} (hk : 0 < k) :
    glasserIntegral k = (Real.sqrt π / 2) * Real.exp (-2 * k) := by
  have heq_event :
      (fun ε : ℝ => glasserScaled ε) =ᶠ[𝓝[>] (0 : ℝ)]
        (fun _ : ℝ => glasserScaled k) := by
    filter_upwards [self_mem_nhdsWithin] with ε hε
    exact glasserScaled_eq_of_pos hε hk
  have hlim_const :
      Tendsto (fun _ : ℝ => glasserScaled k) (𝓝[>] (0 : ℝ))
        (𝓝 (Real.sqrt π / 2)) :=
    tendsto_glasserScaled_nhdsWithin_zero.congr' heq_event
  have hscaled : glasserScaled k = Real.sqrt π / 2 :=
    tendsto_nhds_unique tendsto_const_nhds hlim_const
  unfold glasserScaled at hscaled
  have hexp_ne : Real.exp (2 * k) ≠ 0 := (Real.exp_pos _).ne'
  calc
    glasserIntegral k =
        (glasserIntegral k * Real.exp (2 * k)) * (Real.exp (2 * k))⁻¹ := by
          field_simp [hexp_ne]
    _ = (Real.sqrt π / 2) * Real.exp (-2 * k) := by
          rw [hscaled]
          rw [← Real.exp_neg (2 * k)]
          ring_nf

/-! ## Scalar subordination identity -/

/-- Integral form of the Gaussian subordination identity for the scalar
exponential. -/
lemma subordination_integral_eq_of_pos {a : ℝ} (ha : 0 < a) :
    ∫ s in Ioi (0 : ℝ),
        Real.exp (-s ^ 2) * Real.exp (-(a ^ 2) / (4 * s ^ 2)) =
      (Real.sqrt π / 2) * Real.exp (-a) := by
  have hk : 0 < a / 2 := by positivity
  have h := glasserIntegral_eq_closed_of_pos (k := a / 2) hk
  unfold glasserIntegral glasserKernel at h
  convert h using 1
  · apply setIntegral_congr_fun measurableSet_Ioi
    intro s hs
    dsimp
    rw [← Real.exp_add]
    congr 1
    have hs0 : s ≠ 0 := (show (0 : ℝ) < s from hs).ne'
    field_simp [hs0]
    ring
  · ring_nf

/-- Normalized scalar subordination identity. -/
lemma exp_neg_eq_subordination_of_pos {a : ℝ} (ha : 0 < a) :
    Real.exp (-a) =
      (2 / Real.sqrt π) *
        ∫ s in Ioi (0 : ℝ),
          Real.exp (-s ^ 2) * Real.exp (-(a ^ 2) / (4 * s ^ 2)) := by
  have hsqrt_ne : Real.sqrt π ≠ 0 := by positivity
  rw [subordination_integral_eq_of_pos ha]
  field_simp [hsqrt_ne]

/-- Normalized scalar subordination identity, including the endpoint `a = 0`.

The positive-half-line theorem is the genuine Glasser computation.  The endpoint
uses the elementary half-Gaussian integral. -/
lemma exp_neg_eq_subordination_of_nonneg {a : ℝ} (ha : 0 ≤ a) :
    Real.exp (-a) =
      (2 / Real.sqrt π) *
        ∫ s in Ioi (0 : ℝ),
          Real.exp (-s ^ 2) * Real.exp (-(a ^ 2) / (4 * s ^ 2)) := by
  rcases ha.eq_or_lt with rfl | hpos
  · have hsqrt_ne : Real.sqrt π ≠ 0 := by positivity
    have hgauss :
        ∫ s in Ioi (0 : ℝ), Real.exp (-s ^ 2) = Real.sqrt π / 2 := by
      simpa [one_mul] using (integral_gaussian_Ioi 1)
    simp only [neg_zero, Real.exp_zero, zero_pow (by norm_num : (2 : ℕ) ≠ 0),
      zero_div, mul_one]
    rw [hgauss]
    field_simp [hsqrt_ne]
  · exact exp_neg_eq_subordination_of_pos hpos

/-! ## Gaussian Fourier transform facts used by the radial bridge -/

section EuclideanFourier

variable {ι : Type*} [Fintype ι]

/-- Real-valued closed form for the Fourier transform of a centered Gaussian on
`EuclideanSpace`.  This is just Mathlib's Gaussian Fourier transform theorem,
rewritten through `ofReal_cpow` so the sign/positivity facts are transparent. -/
lemma fourier_gaussian_sq_norm_eq_real
    {b : ℝ} (hb : 0 < b) (w : EuclideanSpace ℝ ι) :
    𝓕 (fun (v : EuclideanSpace ℝ ι) =>
        Complex.exp (-(b : ℂ) * (‖v‖ : ℂ) ^ 2)) w =
      (Real.rpow (π / b)
          (((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2) *
        Real.exp (-(π ^ 2) * ‖w‖ ^ 2 / b) : ℂ) := by
  rw [fourier_gaussian_innerProductSpace (V := EuclideanSpace ℝ ι) (b := (b : ℂ))]
  · have hdiv : ((π : ℂ) / (b : ℂ)) = ((π / b : ℝ) : ℂ) := by
      rw [← Complex.ofReal_div]
    have hpow :
        ((π : ℂ) / (b : ℂ)) ^
            ((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℂ) / 2) =
          ((Real.rpow (π / b)
              (((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2) : ℝ) :
            ℂ) := by
      rw [hdiv]
      have hexp :
          ((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℂ) / 2) =
            ((((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2 : ℝ) : ℂ) := by
        norm_num
      rw [hexp, ← Complex.ofReal_cpow (by positivity)]
      rfl
    rw [hpow]
    have harg :
        (-↑π ^ 2 * ↑‖w‖ ^ 2 / (b : ℂ)) =
          ((-(π ^ 2) * ‖w‖ ^ 2 / b : ℝ) : ℂ) := by
      norm_num [Complex.ofReal_div]
    rw [harg, ← Complex.ofReal_exp]
  · simpa using hb

/-- The centered Gaussian Fourier transform is a strictly positive real number. -/
lemma fourier_gaussian_sq_norm_re_pos
    {b : ℝ} (hb : 0 < b) (w : EuclideanSpace ℝ ι) :
    0 <
      (𝓕 (fun (v : EuclideanSpace ℝ ι) =>
          Complex.exp (-(b : ℂ) * (‖v‖ : ℂ) ^ 2)) w).re := by
  rw [fourier_gaussian_sq_norm_eq_real (ι := ι) hb w]
  rw [← Complex.ofReal_mul]
  simp only [Complex.ofReal_re]
  exact mul_pos (Real.rpow_pos_of_pos (div_pos Real.pi_pos hb) _)
    (Real.exp_pos _)

/-- In particular, the centered Gaussian Fourier transform never vanishes. -/
lemma fourier_gaussian_sq_norm_ne_zero
    {b : ℝ} (hb : 0 < b) (w : EuclideanSpace ℝ ι) :
    𝓕 (fun (v : EuclideanSpace ℝ ι) =>
        Complex.exp (-(b : ℂ) * (‖v‖ : ℂ) ^ 2)) w ≠ 0 := by
  intro hzero
  have hpos := fourier_gaussian_sq_norm_re_pos (ι := ι) hb w
  rw [hzero] at hpos
  exact (lt_irrefl (0 : ℝ)) hpos

end EuclideanFourier

/-! ## Pointwise Euclidean subordination -/

section EuclideanSubordination

variable {ι : Type*} [Fintype ι]

/-- The real scalar density appearing in the Gaussian subordination of the
radial Laplace profile. -/
noncomputable def laplaceRadialSubordinationScalar
    (τ s : ℝ) (x : EuclideanSpace ℝ ι) : ℝ :=
  (2 / Real.sqrt π) * (Real.exp (-s ^ 2) *
    Real.exp (-((‖x‖ / τ) ^ 2) / (4 * s ^ 2)))

/-- Pointwise Gaussian-mixture representation of the Euclidean radial Laplace
profile.  This is the scalar subordination identity applied to
`a = ‖x‖ / τ`, with the real integral transported into `ℂ`. -/
lemma laplaceEuclideanFourierBase_eq_subordination_integral
    {τ : ℝ} (hτ : 0 < τ) (x : EuclideanSpace ℝ ι) :
    laplaceEuclideanFourierBase (E := EuclideanSpace ℝ ι) τ x =
      ∫ s in Ioi (0 : ℝ),
        ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ) := by
  unfold laplaceEuclideanFourierBase laplaceRadialSubordinationScalar
  rw [integral_complex_ofReal]
  rw [MeasureTheory.integral_const_mul]
  have ha : 0 ≤ ‖x‖ / τ := div_nonneg (norm_nonneg _) hτ.le
  have h := exp_neg_eq_subordination_of_nonneg (a := ‖x‖ / τ) ha
  rw [show -‖x‖ / τ = -(‖x‖ / τ) by ring]
  exact_mod_cast h

/-- On the positive half-line in the subordination parameter, the scalar
subordination density is a positive Gaussian in the spatial variable. -/
lemma laplaceRadialSubordinationScalar_eq_gaussian
    {τ s : ℝ} (hτ : τ ≠ 0) (hs : s ≠ 0) (x : EuclideanSpace ℝ ι) :
    laplaceRadialSubordinationScalar (ι := ι) τ s x =
      (2 / Real.sqrt π) * Real.exp (-s ^ 2) *
        Real.exp (-(1 / (4 * τ ^ 2 * s ^ 2)) * ‖x‖ ^ 2) := by
  unfold laplaceRadialSubordinationScalar
  calc
    (2 / Real.sqrt π) *
        (Real.exp (-s ^ 2) *
          Real.exp (-((‖x‖ / τ) ^ 2) / (4 * s ^ 2)))
        =
      (2 / Real.sqrt π) * Real.exp (-s ^ 2) *
        Real.exp (-((‖x‖ / τ) ^ 2) / (4 * s ^ 2)) := by ring
    _ =
      (2 / Real.sqrt π) * Real.exp (-s ^ 2) *
        Real.exp (-(1 / (4 * τ ^ 2 * s ^ 2)) * ‖x‖ ^ 2) := by
        congr 1
        congr 1
        field_simp [hτ, hs]

/-- Positivity of the scalar subordination density for positive bandwidth and
positive subordination parameter. -/
lemma laplaceRadialSubordinationScalar_pos
    (τ s : ℝ) (x : EuclideanSpace ℝ ι) :
    0 < laplaceRadialSubordinationScalar (ι := ι) τ s x := by
  unfold laplaceRadialSubordinationScalar
  positivity

/-- The Gaussian coefficient generated by subordination is positive. -/
lemma laplaceRadialSubordinationGaussianCoeff_pos
    {τ s : ℝ} (hτ : 0 < τ) (hs : 0 < s) :
    0 < 1 / (4 * τ ^ 2 * s ^ 2) := by
  positivity

omit [Fintype ι] in
/-- The Gaussian mass factor produced by subordination is a constant multiple
of `s^n`, where `n` is the Euclidean dimension. -/
lemma laplaceRadialSubordinationGaussianMass_factor
    {τ s : ℝ} (hτ : 0 < τ) (hs : 0 < s) :
    (π / (1 / (4 * τ ^ 2 * s ^ 2))) ^
        (((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2) =
      (4 * π * τ ^ 2) ^
        (((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2) *
        s ^ (Module.finrank ℝ (EuclideanSpace ℝ ι)) := by
  let d : ℝ := (((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2)
  have hbase :
      π / (1 / (4 * τ ^ 2 * s ^ 2)) = (4 * π * τ ^ 2) * s ^ 2 := by
    field_simp [hτ.ne', hs.ne']
  rw [hbase]
  change ((4 * π * τ ^ 2) * s ^ 2) ^ d =
    (4 * π * τ ^ 2) ^ d * s ^ Module.finrank ℝ (EuclideanSpace ℝ ι)
  rw [Real.mul_rpow (by positivity : 0 ≤ 4 * π * τ ^ 2) (sq_nonneg s)]
  have hs2 :
      (s ^ 2) ^ d =
        s ^ ((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) := by
    rw [← Real.rpow_two s]
    dsimp [d]
    rw [← Real.rpow_mul hs.le]
    ring_nf
  rw [hs2, Real.rpow_natCast]

omit [Fintype ι] in
/-- Integrability in the subordination parameter of the total Gaussian mass.
This is the Tonelli estimate: after integrating the Gaussian in space, the
remaining `s`-dependence is `s^n e^{-s²}` up to a positive constant. -/
lemma integrableOn_laplaceRadialSubordinationGaussianMass
    {τ : ℝ} (hτ : 0 < τ) :
    IntegrableOn (fun s : ℝ =>
      (2 / Real.sqrt π) * Real.exp (-s ^ 2) *
        ((π / (1 / (4 * τ ^ 2 * s ^ 2))) ^
          (((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2))) (Ioi 0) := by
  let n : ℝ := (Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ)
  let K : ℝ := (2 / Real.sqrt π) * (4 * π * τ ^ 2) ^
    (((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2)
  have hn : -1 < n := by
    have hn0 : 0 ≤ n := by
      dsimp [n]
      positivity
    linarith
  have hbaseInt :
      IntegrableOn (fun s : ℝ => s ^ n * Real.exp (-1 * s ^ 2)) (Ioi 0) := by
    exact integrableOn_rpow_mul_exp_neg_mul_sq (b := 1) one_pos (s := n) hn
  have hK :
      IntegrableOn (fun s : ℝ => K * (s ^ n * Real.exp (-1 * s ^ 2))) (Ioi 0) :=
    hbaseInt.const_mul K
  refine hK.congr_fun ?_ measurableSet_Ioi
  intro s hs
  have hspos : 0 < s := hs
  dsimp [K, n]
  rw [laplaceRadialSubordinationGaussianMass_factor (ι := ι) hτ hspos]
  rw [Real.rpow_natCast]
  ring_nf

/-- Exact spatial integral of the norm of a subordinated Fourier slice.  The
phase has norm one, so the integral is just the Gaussian mass from the
subordination estimate. -/
lemma integral_norm_laplaceRadialSubordination_fourier_slice
    {τ s : ℝ} (hτ : 0 < τ) (hs : 0 < s) (w : EuclideanSpace ℝ ι) :
    (∫ x : EuclideanSpace ℝ ι,
      ‖Real.fourierChar (-(inner ℝ x w)) •
        ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ)‖) =
      (2 / Real.sqrt π) * Real.exp (-s ^ 2) *
        ((π / (1 / (4 * τ ^ 2 * s ^ 2))) ^
          (((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2)) := by
  have hpoint : ∀ x : EuclideanSpace ℝ ι,
      ‖Real.fourierChar (-(inner ℝ x w)) •
        ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ)‖ =
        (2 / Real.sqrt π) * Real.exp (-s ^ 2) *
          Real.exp (-(1 / (4 * τ ^ 2 * s ^ 2)) * ‖x‖ ^ 2) := by
    intro x
    rw [Circle.norm_smul]
    rw [Complex.norm_real]
    rw [Real.norm_eq_abs]
    rw [abs_of_pos (laplaceRadialSubordinationScalar_pos (ι := ι) τ s x)]
    rw [laplaceRadialSubordinationScalar_eq_gaussian (ι := ι) hτ.ne' hs.ne' x]
  calc
    (∫ x : EuclideanSpace ℝ ι,
      ‖Real.fourierChar (-(inner ℝ x w)) •
        ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ)‖)
      = ∫ x : EuclideanSpace ℝ ι,
          (2 / Real.sqrt π) * Real.exp (-s ^ 2) *
          Real.exp (-(1 / (4 * τ ^ 2 * s ^ 2)) * ‖x‖ ^ 2) := by
          apply MeasureTheory.integral_congr_ae
          filter_upwards with x
          exact hpoint x
    _ = (2 / Real.sqrt π) * Real.exp (-s ^ 2) *
        ∫ x : EuclideanSpace ℝ ι,
          Real.exp (-(1 / (4 * τ ^ 2 * s ^ 2)) * ‖x‖ ^ 2) := by
          rw [← MeasureTheory.integral_const_mul]
    _ = (2 / Real.sqrt π) * Real.exp (-s ^ 2) *
        ((π / (1 / (4 * τ ^ 2 * s ^ 2))) ^
          (((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2)) := by
          rw [GaussianFourier.integral_rexp_neg_mul_sq_norm (V := EuclideanSpace ℝ ι)
            (laplaceRadialSubordinationGaussianCoeff_pos hτ hs)]

/-- For each fixed positive subordination parameter, the phase-weighted
Gaussian slice is integrable in the spatial variable. -/
lemma integrable_laplaceRadialSubordination_fourier_slice
    {τ s : ℝ} (hτ : 0 < τ) (hs : 0 < s) (w : EuclideanSpace ℝ ι) :
    Integrable (fun x : EuclideanSpace ℝ ι =>
      Real.fourierChar (-(inner ℝ x w)) •
        ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ)) := by
  let b : ℝ := 1 / (4 * τ ^ 2 * s ^ 2)
  let C : ℂ := (((2 / Real.sqrt π) * Real.exp (-s ^ 2) : ℝ) : ℂ)
  let g : EuclideanSpace ℝ ι → ℂ := fun x =>
    Complex.exp (-(b : ℂ) * (‖x‖ : ℂ) ^ 2)
  have hb : 0 < b := by
    dsimp [b]
    positivity
  have hg : Integrable g := by
    have h := GaussianFourier.integrable_cexp_neg_mul_sq_norm_add
      (V := EuclideanSpace ℝ ι) (b := (b : ℂ))
      (show 0 < (b : ℂ).re by simpa using hb) 0 (0 : EuclideanSpace ℝ ι)
    simpa [g] using h
  have hCg : Integrable (fun x : EuclideanSpace ℝ ι => C * g x) := hg.const_mul C
  have hphase :
      AEStronglyMeasurable
        (fun x : EuclideanSpace ℝ ι => (Real.fourierChar (-(inner ℝ x w)) : ℂ)) := by
    apply Continuous.aestronglyMeasurable
    fun_prop
  have hbounded :
      ∀ᵐ x : EuclideanSpace ℝ ι,
        ‖(Real.fourierChar (-(inner ℝ x w)) : ℂ)‖ ≤ 1 := by
    filter_upwards with x
    simp
  have hphaseCg :
      Integrable
        (fun x : EuclideanSpace ℝ ι =>
          (Real.fourierChar (-(inner ℝ x w)) : ℂ) * (C * g x)) :=
    hCg.bdd_mul hphase hbounded
  have hpoint : ∀ x : EuclideanSpace ℝ ι,
      ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ) = C * g x := by
    intro x
    rw [laplaceRadialSubordinationScalar_eq_gaussian (ι := ι) hτ.ne' hs.ne' x]
    dsimp [C, g, b]
    have harg :
        (-((1 / (4 * τ ^ 2 * s ^ 2) : ℝ) : ℂ) * (‖x‖ : ℂ) ^ 2) =
          ((-(1 / (4 * τ ^ 2 * s ^ 2)) * ‖x‖ ^ 2 : ℝ) : ℂ) := by
      norm_num [Complex.ofReal_mul]
    rw [harg, ← Complex.ofReal_exp]
    norm_num [Complex.ofReal_mul]
  refine hphaseCg.congr ?_
  filter_upwards with x
  rw [hpoint x]
  simp [Circle.smul_def]

/-- Product integrability of the phase-weighted subordination integrand.
This is the global Tonelli/Fubini estimate needed to swap the spatial Fourier
integral with the positive Gaussian-mixture parameter. -/
lemma integrable_laplaceRadialSubordination_fourier_product
    {τ : ℝ} (hτ : 0 < τ) (w : EuclideanSpace ℝ ι) :
    Integrable
      (fun z : ℝ × EuclideanSpace ℝ ι =>
        Real.fourierChar (-(inner ℝ z.2 w)) •
          ((laplaceRadialSubordinationScalar (ι := ι) τ z.1 z.2 : ℝ) : ℂ))
      ((volume.restrict (Ioi (0 : ℝ))).prod
        (volume : Measure (EuclideanSpace ℝ ι))) := by
  have hmeas : AEStronglyMeasurable
      (fun z : ℝ × EuclideanSpace ℝ ι =>
        Real.fourierChar (-(inner ℝ z.2 w)) •
          ((laplaceRadialSubordinationScalar (ι := ι) τ z.1 z.2 : ℝ) : ℂ))
      ((volume.restrict (Ioi (0 : ℝ))).prod
        (volume : Measure (EuclideanSpace ℝ ι))) := by
    apply Measurable.aestronglyMeasurable
    simp only [Circle.smul_def]
    unfold laplaceRadialSubordinationScalar
    fun_prop
  refine (integrable_prod_iff hmeas).2 ⟨?_, ?_⟩
  · filter_upwards [ae_restrict_mem measurableSet_Ioi] with s hs
    exact integrable_laplaceRadialSubordination_fourier_slice (ι := ι) hτ hs w
  · have hmass := integrableOn_laplaceRadialSubordinationGaussianMass (ι := ι) hτ
    refine hmass.congr ?_
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with s hs
    exact (integral_norm_laplaceRadialSubordination_fourier_slice (ι := ι) hτ hs w).symm

/-- For each positive subordination parameter `s`, the inner Fourier integral
of the subordinated Gaussian profile is a positive scalar multiple of
Mathlib's centered Gaussian Fourier transform. -/
lemma laplaceRadialSubordination_inner_fourier_eq
    {τ s : ℝ} (hτ : 0 < τ) (hs : 0 < s)
    (w : EuclideanSpace ℝ ι) :
    (∫ x : EuclideanSpace ℝ ι,
      Real.fourierChar (-(inner ℝ x w)) •
        ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ)) =
      (((2 / Real.sqrt π) * Real.exp (-s ^ 2) : ℝ) : ℂ) *
        𝓕 (fun (x : EuclideanSpace ℝ ι) =>
          Complex.exp (-(1 / (4 * τ ^ 2 * s ^ 2) : ℂ) *
            (‖x‖ : ℂ) ^ 2)) w := by
  let C : ℂ := (((2 / Real.sqrt π) * Real.exp (-s ^ 2) : ℝ) : ℂ)
  let g : EuclideanSpace ℝ ι → ℂ := fun x =>
    Complex.exp (-(1 / (4 * τ ^ 2 * s ^ 2) : ℂ) * (‖x‖ : ℂ) ^ 2)
  have hpoint : ∀ x : EuclideanSpace ℝ ι,
      ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ) = C * g x := by
    intro x
    rw [laplaceRadialSubordinationScalar_eq_gaussian (ι := ι) hτ.ne' hs.ne' x]
    dsimp [C, g]
    have harg :
        (-(1 / (4 * (τ : ℂ) ^ 2 * (s : ℂ) ^ 2)) * (‖x‖ : ℂ) ^ 2) =
          ((-(1 / (4 * τ ^ 2 * s ^ 2)) * ‖x‖ ^ 2 : ℝ) : ℂ) := by
      norm_num [Complex.ofReal_div]
    rw [harg, ← Complex.ofReal_exp]
    norm_num [Complex.ofReal_mul]
  calc
    (∫ x : EuclideanSpace ℝ ι,
      Real.fourierChar (-(inner ℝ x w)) •
        ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ))
        = ∫ x : EuclideanSpace ℝ ι,
            Real.fourierChar (-(inner ℝ x w)) • (C * g x) := by
            apply MeasureTheory.integral_congr_ae
            filter_upwards with x
            rw [hpoint x]
    _ = C * ∫ x : EuclideanSpace ℝ ι,
            Real.fourierChar (-(inner ℝ x w)) • g x := by
            rw [← MeasureTheory.integral_const_mul]
            congr 1
            funext x
            simp [Circle.smul_def]
            ring
    _ = (((2 / Real.sqrt π) * Real.exp (-s ^ 2) : ℝ) : ℂ) *
        𝓕 (fun (x : EuclideanSpace ℝ ι) =>
          Complex.exp (-(1 / (4 * τ ^ 2 * s ^ 2) : ℂ) *
            (‖x‖ : ℂ) ^ 2)) w := by
          rfl

/-- The inner Fourier integral appearing in the subordination bridge has
strictly positive real part. -/
lemma laplaceRadialSubordination_inner_fourier_re_pos
    {τ s : ℝ} (hτ : 0 < τ) (hs : 0 < s)
    (w : EuclideanSpace ℝ ι) :
    0 <
      ((∫ x : EuclideanSpace ℝ ι,
        Real.fourierChar (-(inner ℝ x w)) •
          ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ))).re := by
  rw [laplaceRadialSubordination_inner_fourier_eq (ι := ι) hτ hs w]
  have hb : 0 < 1 / (4 * τ ^ 2 * s ^ 2) :=
    laplaceRadialSubordinationGaussianCoeff_pos hτ hs
  have hC : 0 < (2 / Real.sqrt π) * Real.exp (-s ^ 2) := by positivity
  have hfun :
      (fun x : EuclideanSpace ℝ ι =>
          Complex.exp (-(1 / (4 * (τ : ℂ) ^ 2 * (s : ℂ) ^ 2)) *
            (‖x‖ : ℂ) ^ 2)) =
        (fun x : EuclideanSpace ℝ ι =>
          Complex.exp (-((1 / (4 * τ ^ 2 * s ^ 2) : ℝ) : ℂ) *
            (‖x‖ : ℂ) ^ 2)) := by
    funext x
    congr 1
    norm_num [Complex.ofReal_div]
  rw [hfun]
  rw [fourier_gaussian_sq_norm_eq_real (ι := ι) hb w]
  rw [← Complex.ofReal_mul]
  let R : ℝ :=
    Real.rpow (π / (1 / (4 * τ ^ 2 * s ^ 2)))
        (((Module.finrank ℝ (EuclideanSpace ℝ ι) : ℕ) : ℝ) / 2) *
      Real.exp (-(π ^ 2) * ‖w‖ ^ 2 / (1 / (4 * τ ^ 2 * s ^ 2)))
  change 0 < ((((2 / Real.sqrt π) * Real.exp (-s ^ 2) : ℝ) : ℂ) * (R : ℂ)).re
  have hcast :
      (((((2 / Real.sqrt π) * Real.exp (-s ^ 2) : ℝ) : ℂ) * (R : ℂ)).re) =
        ((2 / Real.sqrt π) * Real.exp (-s ^ 2)) * R := by
    rw [← Complex.ofReal_mul]
    rfl
  rw [hcast]
  exact mul_pos hC
    (mul_pos (Real.rpow_pos_of_pos (div_pos Real.pi_pos hb) _)
      (Real.exp_pos _))

/-- Fourier transform of the Euclidean radial Laplace profile, written as the
integral over the positive Gaussian subordination parameter of the inner
Gaussian Fourier transforms. -/
lemma fourier_laplaceEuclideanFourierBase_eq_subordination_integral
    {τ : ℝ} (hτ : 0 < τ) (w : EuclideanSpace ℝ ι) :
    𝓕 (laplaceEuclideanFourierBase (E := EuclideanSpace ℝ ι) τ) w =
      ∫ s in Ioi (0 : ℝ),
        ∫ x : EuclideanSpace ℝ ι,
          Real.fourierChar (-(inner ℝ x w)) •
            ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ) := by
  have hprod :=
    integrable_laplaceRadialSubordination_fourier_product (ι := ι) hτ w
  rw [Real.fourier_eq]
  calc
    (∫ x : EuclideanSpace ℝ ι,
        Real.fourierChar (-(inner ℝ x w)) •
          laplaceEuclideanFourierBase (E := EuclideanSpace ℝ ι) τ x)
        =
      ∫ x : EuclideanSpace ℝ ι,
        Real.fourierChar (-(inner ℝ x w)) •
          (∫ s in Ioi (0 : ℝ),
            ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ)) := by
          apply MeasureTheory.integral_congr_ae
          filter_upwards with x
          rw [laplaceEuclideanFourierBase_eq_subordination_integral (ι := ι) hτ x]
    _ =
      ∫ x : EuclideanSpace ℝ ι,
        ∫ s in Ioi (0 : ℝ),
          Real.fourierChar (-(inner ℝ x w)) •
            ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ) := by
          congr 1
          funext x
          change
            ((Real.fourierChar (-(inner ℝ x w)) : ℂ) *
              (∫ s in Ioi (0 : ℝ),
                ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ))) =
              ∫ s in Ioi (0 : ℝ),
                (Real.fourierChar (-(inner ℝ x w)) : ℂ) *
                  ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ)
          rw [← MeasureTheory.integral_const_mul]
    _ =
      ∫ s in Ioi (0 : ℝ),
        ∫ x : EuclideanSpace ℝ ι,
          Real.fourierChar (-(inner ℝ x w)) •
            ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ) := by
          exact (MeasureTheory.integral_integral_swap
            (f := fun s x =>
              Real.fourierChar (-(inner ℝ x w)) •
                ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ))
            hprod).symm

/-- The inner subordinated Fourier integral is integrable in the
subordination parameter. -/
lemma integrable_laplaceRadialSubordination_inner_fourier
    {τ : ℝ} (hτ : 0 < τ) (w : EuclideanSpace ℝ ι) :
    Integrable
      (fun s : ℝ =>
        ∫ x : EuclideanSpace ℝ ι,
          Real.fourierChar (-(inner ℝ x w)) •
            ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ))
      (volume.restrict (Ioi (0 : ℝ))) := by
  exact (integrable_laplaceRadialSubordination_fourier_product (ι := ι) hτ w).integral_prod_left

/-- The Fourier transform of the Euclidean radial Laplace profile has
strictly positive real part.  The proof is the promised Tonelli argument:
after subordination, it is an integral of positive Gaussian Fourier
transforms over the positive parameter half-line. -/
theorem fourier_laplaceEuclideanFourierBase_re_pos
    {τ : ℝ} (hτ : 0 < τ) (w : EuclideanSpace ℝ ι) :
    0 <
      (𝓕 (laplaceEuclideanFourierBase (E := EuclideanSpace ℝ ι) τ) w).re := by
  let F : ℝ → ℝ := fun s =>
    ((∫ x : EuclideanSpace ℝ ι,
      Real.fourierChar (-(inner ℝ x w)) •
        ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ))).re
  have hinner := integrable_laplaceRadialSubordination_inner_fourier (ι := ι) hτ w
  rw [fourier_laplaceEuclideanFourierBase_eq_subordination_integral (ι := ι) hτ w]
  have hre :
      ((∫ s in Ioi (0 : ℝ),
        ∫ x : EuclideanSpace ℝ ι,
          Real.fourierChar (-(inner ℝ x w)) •
            ((laplaceRadialSubordinationScalar (ι := ι) τ s x : ℝ) : ℂ))).re =
        ∫ s in Ioi (0 : ℝ), F s := by
    exact (integral_re hinner).symm
  rw [hre]
  have hFint : Integrable F (volume.restrict (Ioi (0 : ℝ))) := hinner.re
  have hFnonneg_ae : 0 ≤ᵐ[volume.restrict (Ioi (0 : ℝ))] F := by
    filter_upwards [ae_restrict_mem measurableSet_Ioi] with s hs
    exact (laplaceRadialSubordination_inner_fourier_re_pos (ι := ι) hτ hs w).le
  rw [integral_pos_iff_support_of_nonneg_ae hFnonneg_ae hFint]
  have hsubset : Ioo (1 : ℝ) 2 ⊆ Function.support F := by
    intro s hs
    change F s ≠ 0
    exact ne_of_gt
      (laplaceRadialSubordination_inner_fourier_re_pos (ι := ι) hτ
        (lt_trans zero_lt_one hs.1) w)
  have hmeasure_mono :
      (volume.restrict (Ioi (0 : ℝ))) (Ioo (1 : ℝ) 2) ≤
        (volume.restrict (Ioi (0 : ℝ))) (Function.support F) :=
    measure_mono hsubset
  have hinterval_pos :
      0 < (volume.restrict (Ioi (0 : ℝ))) (Ioo (1 : ℝ) 2) := by
    have hinter : Ioo (1 : ℝ) 2 ∩ Ioi (0 : ℝ) = Ioo (1 : ℝ) 2 := by
      ext s
      constructor
      · intro hs
        exact hs.1
      · intro hs
        exact ⟨hs, lt_trans zero_lt_one hs.1⟩
    rw [Measure.restrict_apply measurableSet_Ioo]
    rw [hinter]
    rw [Real.volume_Ioo]
    norm_num
  exact lt_of_lt_of_le hinterval_pos hmeasure_mono

/-- The Fourier transform of the Euclidean radial Laplace profile never
vanishes. -/
theorem fourier_laplaceEuclideanFourierBase_ne_zero
    {τ : ℝ} (hτ : 0 < τ) (w : EuclideanSpace ℝ ι) :
    𝓕 (laplaceEuclideanFourierBase (E := EuclideanSpace ℝ ι) τ) w ≠ 0 := by
  intro hzero
  have hpos := fourier_laplaceEuclideanFourierBase_re_pos (ι := ι) hτ w
  rw [hzero] at hpos
  exact (lt_irrefl (0 : ℝ)) hpos

/-- Fully discharged Euclidean-space L4 smoothing injectivity for the paper's
Laplace kernel.  The radial Fourier nonvanishing hypothesis required by
`LaplaceEuclideanInjectivity` is supplied by Gaussian subordination above. -/
theorem laplaceSmoothingInjective_euclideanSpace
    (τ : ℝ) (hτ : Paper.ValidBandwidth τ) :
    LaplaceSmoothingInjective (EuclideanSpace ℝ ι) τ :=
  laplaceSmoothingInjective_euclideanSpace_of_fourier_ne_zero
    (ι := ι) τ hτ
    (fun w => fourier_laplaceEuclideanFourierBase_ne_zero (ι := ι) hτ w)

end EuclideanSubordination

end DriftingIdentifiability
