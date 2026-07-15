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

/-! ## The Glasser integral and the self-reciprocal substitution -/

/-- The Glasser integrand `x ↦ e^{-x² - k²/x²}`. -/
noncomputable def glasserKernel (k x : ℝ) : ℝ := Real.exp (-x ^ 2 - k ^ 2 / x ^ 2)

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

end DriftingIdentifiability
