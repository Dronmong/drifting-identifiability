import DriftingIdentifiability.LaplaceRadialConverse3

/-!
# Radial Laplace converse, milestone L5: rotation invariance (`n = 3`)

The one remaining geometric fact: the ray normalizer profile determines the
normalizer **everywhere**, `Z_{radialMixture₃ ν}(x) = Z̃_ν(‖x‖)`.

The proof avoids polar coordinates and 3-d changes of variables entirely
(`LaplaceHigherDim.md §4.10`, 2026-07-16 discovery).  In the polar-angle
parametrisation `u = cos t` the tilted zonal average

`J(θ) = (1/4π)·∫₀^π ∫_{−π}^{π} g(cosθ·cos t + sinθ·sin t·cos φ) · sin t dφ dt`

is **constant in θ**: its derivative splits, after the pointwise generator
identity

`∂θ w = −(cos φ·∂t w − cot t·sin φ·∂φ w · tan-free form)`,

into two elementary integrations by parts whose boundary terms vanish
(`sin 0 = sin π = 0`, `sin(±π) = 0`) and whose bulk terms cancel exactly
(both equal `∫∫ cos t·cos φ·g'(w)·…`).  The azimuth is handled by
φ-periodicity, and the collision shell `s = ‖x‖` by continuity in `s` of both
sides (`continuous_shellZ`).
-/

open MeasureTheory Filter Topology Set intervalIntegral
open scoped RealInnerProductSpace

namespace DriftingIdentifiability
open Paper

/-! ## The tilted zonal kernel and its parameters -/

/-- The tilted inner product `w(θ,t,φ) = cosθ·cos t + sinθ·sin t·cos φ` (the
inner product of the chart point at polar angle `t`, azimuth `φ` with the unit
vector at polar angle `θ`, azimuth `0`). -/
noncomputable def tiltW (θ t φ : ℝ) : ℝ :=
  Real.cos θ * Real.cos t + Real.sin θ * Real.sin t * Real.cos φ

lemma abs_tiltW_le_one (θ t φ : ℝ) : |tiltW θ t φ| ≤ 1 := by
  unfold tiltW
  have h1 := Real.abs_cos_le_one θ
  have h2 := Real.abs_cos_le_one t
  have h3 := Real.abs_sin_le_one θ
  have h4 := Real.abs_sin_le_one t
  have h5 := Real.abs_cos_le_one φ
  have key : |Real.cos θ * Real.cos t + Real.sin θ * Real.sin t * Real.cos φ|
      ≤ |Real.cos θ| * |Real.cos t| + |Real.sin θ| * |Real.sin t| * |Real.cos φ| := by
    calc |Real.cos θ * Real.cos t + Real.sin θ * Real.sin t * Real.cos φ|
        ≤ |Real.cos θ * Real.cos t| + |Real.sin θ * Real.sin t * Real.cos φ| :=
          abs_add_le _ _
      _ = |Real.cos θ| * |Real.cos t| + |Real.sin θ| * |Real.sin t| * |Real.cos φ| := by
          rw [abs_mul, abs_mul, abs_mul]
  refine key.trans ?_
  have hcs : |Real.cos θ| * |Real.cos t| + |Real.sin θ| * |Real.sin t| * |Real.cos φ|
      ≤ |Real.cos θ| * |Real.cos t| + |Real.sin θ| * |Real.sin t| := by
    have := mul_le_mul_of_nonneg_left h5
      (mul_nonneg (abs_nonneg (Real.sin θ)) (abs_nonneg (Real.sin t)))
    nlinarith [abs_nonneg (Real.sin θ), abs_nonneg (Real.sin t)]
  refine hcs.trans ?_
  nlinarith [Real.sin_sq_add_cos_sq θ, Real.sin_sq_add_cos_sq t,
    sq_abs (Real.sin θ), sq_abs (Real.cos θ), sq_abs (Real.sin t), sq_abs (Real.cos t),
    sq_nonneg (|Real.cos θ| * |Real.sin t| - |Real.sin θ| * |Real.cos t|),
    abs_nonneg (Real.cos θ), abs_nonneg (Real.sin t),
    abs_nonneg (Real.sin θ), abs_nonneg (Real.cos t)]

/-- The tilted kernel value: `g(R,s,w) = exp(−(1/τ)·√(R²+s²−2Rs·w))`. -/
noncomputable def tiltKernel (τ R s w : ℝ) : ℝ :=
  Real.exp (-(1 / τ) * Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w))

/-- Away from the collision (`R ≠ s`, both nonnegative), the radicand is
uniformly positive on `|w| ≤ 1`. -/
lemma radicand_pos {R s : ℝ} (hR : 0 ≤ R) (hs : 0 ≤ s) (hRs : R ≠ s)
    {w : ℝ} (hw : |w| ≤ 1) :
    0 < R ^ 2 + s ^ 2 - 2 * R * s * w := by
  have h1 : R ^ 2 + s ^ 2 - 2 * R * s * w ≥ R ^ 2 + s ^ 2 - 2 * R * s := by
    have hRs' : 0 ≤ R * s := mul_nonneg hR hs
    nlinarith [abs_le.mp hw]
  have h2 : (R - s) ^ 2 > 0 := by
    have : R - s ≠ 0 := sub_ne_zero.mpr hRs
    positivity
  nlinarith

/-! ## Derivatives of the tilted data -/

/-- The `w`-derivative of the tilted kernel. -/
noncomputable def tiltKernel' (τ R s w : ℝ) : ℝ :=
  R * s / τ * Real.exp (-(1 / τ) * Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w))
    / Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w)

lemma hasDerivAt_tiltKernel (τ R s : ℝ) {w : ℝ}
    (hpos : 0 < R ^ 2 + s ^ 2 - 2 * R * s * w) :
    HasDerivAt (tiltKernel τ R s) (tiltKernel' τ R s w) w := by
  have hrad : HasDerivAt (fun w : ℝ => R ^ 2 + s ^ 2 - 2 * R * s * w)
      (-(2 * R * s)) w := by
    simpa using ((hasDerivAt_id w).const_mul (2 * R * s)).const_sub (R ^ 2 + s ^ 2)
  have hsqrt := hrad.sqrt hpos.ne'
  have hmul := hsqrt.const_mul (-(1 / τ))
  have hexp := hmul.exp
  have hval : Real.exp (-(1 / τ) * Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w))
        * (-(1 / τ) * (-(2 * R * s) / (2 * Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w))))
      = tiltKernel' τ R s w := by
    rw [tiltKernel']
    have hsne : Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w) ≠ 0 :=
      (Real.sqrt_pos.mpr hpos).ne'
    field_simp
  rw [hval] at hexp
  exact hexp

lemma tiltKernel_pos (τ R s w : ℝ) : 0 < tiltKernel τ R s w := Real.exp_pos _

lemma tiltKernel_le_one (τ : ℝ) (hτ : 0 < τ) (R s w : ℝ) :
    tiltKernel τ R s w ≤ 1 := by
  rw [tiltKernel, Real.exp_le_one_iff]
  have h1 : 0 ≤ Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w) := Real.sqrt_nonneg _
  have h2 : (0 : ℝ) ≤ 1 / τ := by positivity
  nlinarith

/-- Uniform bound for `|K'|` away from the collision. -/
lemma abs_tiltKernel'_le (τ : ℝ) (hτ : 0 < τ) {R s : ℝ}
    (hR : 0 ≤ R) (hs : 0 ≤ s) (hRs : R ≠ s) {w : ℝ} (hw : |w| ≤ 1) :
    |tiltKernel' τ R s w| ≤ R * s / τ / |R - s| := by
  have hpos := radicand_pos hR hs hRs hw
  have hsqrt_pos : 0 < Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w) :=
    Real.sqrt_pos.mpr hpos
  have hsqrt_ge : |R - s| ≤ Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w) := by
    rw [← Real.sqrt_sq_eq_abs]
    apply Real.sqrt_le_sqrt
    have := abs_le.mp hw
    nlinarith [mul_nonneg hR hs]
  have hRspos : (0 : ℝ) ≤ R * s / τ := by positivity
  have hexp1 : Real.exp (-(1 / τ) * Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w)) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    have h2 : (0 : ℝ) ≤ 1 / τ := by positivity
    nlinarith [Real.sqrt_nonneg (R ^ 2 + s ^ 2 - 2 * R * s * w)]
  have habs0 : 0 < |R - s| := abs_pos.mpr (sub_ne_zero.mpr hRs)
  rw [tiltKernel', abs_div, abs_mul,
    abs_of_nonneg hRspos, abs_of_pos (Real.exp_pos _), abs_of_pos hsqrt_pos]
  calc R * s / τ * Real.exp (-(1 / τ) * Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w))
        / Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w)
      ≤ R * s / τ * 1 / Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w) := by
        gcongr
    _ = R * s / τ / Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * w) := by ring
    _ ≤ R * s / τ / |R - s| := by
        gcongr

/-- θ-derivative of the tilted inner product. -/
lemma hasDerivAt_tiltW_theta (θ t φ : ℝ) :
    HasDerivAt (fun θ => tiltW θ t φ)
      (-Real.sin θ * Real.cos t + Real.cos θ * Real.sin t * Real.cos φ) θ := by
  unfold tiltW
  have h1 : HasDerivAt (fun θ : ℝ => Real.cos θ * Real.cos t)
      (-Real.sin θ * Real.cos t) θ := (Real.hasDerivAt_cos θ).mul_const _
  have h2 := ((Real.hasDerivAt_sin θ).mul_const (Real.sin t)).mul_const (Real.cos φ)
  have h3 := h1.add h2
  have hkey : (-Real.sin θ * Real.cos t) + (Real.cos θ * Real.sin t * Real.cos φ)
      = -Real.sin θ * Real.cos t + Real.cos θ * Real.sin t * Real.cos φ := by ring
  exact hkey ▸ h3

/-- t-derivative of the tilted inner product. -/
lemma hasDerivAt_tiltW_t (θ t φ : ℝ) :
    HasDerivAt (fun t => tiltW θ t φ)
      (-(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ) t := by
  unfold tiltW
  have h1 : HasDerivAt (fun t : ℝ => Real.cos θ * Real.cos t)
      (Real.cos θ * -Real.sin t) t := (Real.hasDerivAt_cos t).const_mul _
  have h4 : HasDerivAt (fun t : ℝ => Real.sin θ * Real.sin t)
      (Real.sin θ * Real.cos t) t := (Real.hasDerivAt_sin t).const_mul _
  have h2 := h4.mul_const (Real.cos φ)
  have h3 := h1.add h2
  have hkey : Real.cos θ * -Real.sin t + Real.sin θ * Real.cos t * Real.cos φ
      = -(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ := by ring
  exact hkey ▸ h3

/-- φ-derivative of the tilted inner product. -/
lemma hasDerivAt_tiltW_phi (θ t φ : ℝ) :
    HasDerivAt (fun φ => tiltW θ t φ)
      (Real.sin θ * Real.sin t * -Real.sin φ) φ := by
  unfold tiltW
  have h2 : HasDerivAt (fun φ : ℝ => Real.sin θ * Real.sin t * Real.cos φ)
      (Real.sin θ * Real.sin t * -Real.sin φ) φ :=
    (Real.hasDerivAt_cos φ).const_mul _
  have h3 := (hasDerivAt_const φ (Real.cos θ * Real.cos t)).add h2
  have hkey : 0 + Real.sin θ * Real.sin t * -Real.sin φ
      = Real.sin θ * Real.sin t * -Real.sin φ := by ring
  exact hkey ▸ h3

/-- **The generator identity**: `sin t · ∂θw = −sin t·cos φ·∂tw + cos t·sin φ·∂φw`
— the divergence-free rotation flow in polar-angle coordinates. -/
lemma tiltW_generator (θ t φ : ℝ) :
    Real.sin t * (-Real.sin θ * Real.cos t + Real.cos θ * Real.sin t * Real.cos φ)
      = -(Real.sin t * Real.cos φ)
          * (-(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ)
        + (Real.cos t * Real.sin φ) * (Real.sin θ * Real.sin t * -Real.sin φ) := by
  have hcs := Real.sin_sq_add_cos_sq φ
  linear_combination (Real.sin θ * Real.sin t * Real.cos t) * hcs

/-! ## The tilted zonal average is constant in the tilt -/

/-- The (un-normalised) polar-angle box measure `[0,π] × [−π,π]`. -/
noncomputable def tiltBase : Measure (ℝ × ℝ) :=
  (volume.restrict (Ioc (0 : ℝ) Real.pi)).prod
    (volume.restrict (Ioc (-Real.pi) Real.pi))

instance : IsFiniteMeasure tiltBase := by
  haveI h1 : IsFiniteMeasure (volume.restrict (Ioc (0 : ℝ) Real.pi)) :=
    ⟨by rw [Measure.restrict_apply_univ, Real.volume_Ioc]; exact ENNReal.ofReal_lt_top⟩
  haveI h2 : IsFiniteMeasure (volume.restrict (Ioc (-Real.pi) Real.pi)) :=
    ⟨by rw [Measure.restrict_apply_univ, Real.volume_Ioc]; exact ENNReal.ofReal_lt_top⟩
  unfold tiltBase
  infer_instance

/-- The tilted zonal average `J(θ)` in polar-angle coordinates. -/
noncomputable def tiltJ (τ R s θ : ℝ) : ℝ :=
  ∫ z : ℝ × ℝ, Real.sin z.1 * tiltKernel τ R s (tiltW θ z.1 z.2) ∂tiltBase

/-- Its formal θ-derivative. -/
noncomputable def tiltJderiv (τ R s θ : ℝ) : ℝ :=
  ∫ z : ℝ × ℝ, Real.sin z.1 * (tiltKernel' τ R s (tiltW θ z.1 z.2)
    * (-Real.sin θ * Real.cos z.1 + Real.cos θ * Real.sin z.1 * Real.cos z.2)) ∂tiltBase

lemma continuous_tiltW₂ (θ : ℝ) :
    Continuous fun z : ℝ × ℝ => tiltW θ z.1 z.2 := by
  unfold tiltW
  fun_prop

lemma continuous_tiltKernel_comp (τ R s θ : ℝ) :
    Continuous fun z : ℝ × ℝ => tiltKernel τ R s (tiltW θ z.1 z.2) := by
  unfold tiltKernel tiltW
  fun_prop

lemma continuous_tiltKernel'_comp (τ R s : ℝ) (hR : 0 ≤ R) (hs : 0 ≤ s)
    (hRs : R ≠ s) (θ : ℝ) :
    Continuous fun z : ℝ × ℝ => tiltKernel' τ R s (tiltW θ z.1 z.2) := by
  have hne : ∀ z : ℝ × ℝ,
      Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * tiltW θ z.1 z.2) ≠ 0 := by
    intro z
    exact (Real.sqrt_pos.mpr (radicand_pos hR hs hRs (abs_tiltW_le_one θ z.1 z.2))).ne'
  unfold tiltKernel'
  refine Continuous.div ?_ ?_ hne
  · have h1 : Continuous fun z : ℝ × ℝ =>
        Real.sqrt (R ^ 2 + s ^ 2 - 2 * R * s * tiltW θ z.1 z.2) := by
      unfold tiltW
      fun_prop
    exact continuous_const.mul (Real.continuous_exp.comp (h1.const_mul _))
  · unfold tiltW
    fun_prop

lemma abs_tiltWtheta_le_two (θ t φ : ℝ) :
    |-Real.sin θ * Real.cos t + Real.cos θ * Real.sin t * Real.cos φ| ≤ 2 := by
  have h1 := Real.abs_sin_le_one θ
  have h2 := Real.abs_cos_le_one t
  have h3 := Real.abs_cos_le_one θ
  have h4 := Real.abs_sin_le_one t
  have h5 := Real.abs_cos_le_one φ
  calc |-Real.sin θ * Real.cos t + Real.cos θ * Real.sin t * Real.cos φ|
      ≤ |-Real.sin θ * Real.cos t| + |Real.cos θ * Real.sin t * Real.cos φ| :=
        abs_add_le _ _
    _ = |Real.sin θ| * |Real.cos t| + |Real.cos θ| * |Real.sin t| * |Real.cos φ| := by
        rw [abs_mul, abs_mul, abs_mul, abs_neg]
    _ ≤ 1 + 1 := by
        refine add_le_add ?_ ?_
        · nlinarith [abs_nonneg (Real.sin θ), abs_nonneg (Real.cos t)]
        · nlinarith [abs_nonneg (Real.cos θ), abs_nonneg (Real.sin t),
            abs_nonneg (Real.cos φ), mul_nonneg (abs_nonneg (Real.cos θ)) (abs_nonneg (Real.sin t))]
    _ = 2 := by norm_num

lemma continuous_sin_fst_mul_kernel (τ R s θ : ℝ) :
    Continuous fun z : ℝ × ℝ => Real.sin z.1 * tiltKernel τ R s (tiltW θ z.1 z.2) := by
  have h := continuous_tiltKernel_comp τ R s θ
  fun_prop

/-- `J` is differentiable with the formal derivative (dominated
differentiation; the radicand is uniformly positive off the collision). -/
lemma hasDerivAt_tiltJ (τ : ℝ) (hτ : 0 < τ) {R s : ℝ}
    (hR : 0 ≤ R) (hs : 0 ≤ s) (hRs : R ≠ s) (θ : ℝ) :
    HasDerivAt (tiltJ τ R s) (tiltJderiv τ R s θ) θ := by
  have hfe : tiltJ τ R s = fun θ =>
      ∫ z : ℝ × ℝ, Real.sin z.1 * tiltKernel τ R s (tiltW θ z.1 z.2) ∂tiltBase := rfl
  have hCK : (0 : ℝ) ≤ R * s / τ / |R - s| := by positivity
  have key := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (μ := tiltBase) (x₀ := θ)
    (F := fun θ (z : ℝ × ℝ) => Real.sin z.1 * tiltKernel τ R s (tiltW θ z.1 z.2))
    (F' := fun θ (z : ℝ × ℝ) => Real.sin z.1 * (tiltKernel' τ R s (tiltW θ z.1 z.2)
      * (-Real.sin θ * Real.cos z.1 + Real.cos θ * Real.sin z.1 * Real.cos z.2)))
    (bound := fun _ => (R * s / τ / |R - s|) * 2)
    (s := univ) univ_mem
    (Filter.Eventually.of_forall fun θ' =>
      (continuous_sin_fst_mul_kernel τ R s θ').aestronglyMeasurable)
    ?_ ?_ ?_ ?_ ?_
  · rw [hfe]
    exact key.2
  · -- integrability of F θ
    refine ⟨(continuous_sin_fst_mul_kernel τ R s θ).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := 1) (ae_of_all _ fun z => ?_)⟩
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos (tiltKernel_pos τ R s _)]
    calc |Real.sin z.1| * tiltKernel τ R s (tiltW θ z.1 z.2)
        ≤ 1 * 1 := mul_le_mul (Real.abs_sin_le_one _)
          (tiltKernel_le_one τ hτ R s _) (tiltKernel_pos τ R s _).le zero_le_one
      _ = 1 := by ring
  · -- measurability of F' θ
    have hc : Continuous fun z : ℝ × ℝ =>
        Real.sin z.1 * (tiltKernel' τ R s (tiltW θ z.1 z.2)
          * (-Real.sin θ * Real.cos z.1 + Real.cos θ * Real.sin z.1 * Real.cos z.2)) := by
      have h1 := continuous_tiltKernel'_comp τ R s hR hs hRs θ
      fun_prop
    exact hc.aestronglyMeasurable
  · -- uniform bound
    refine ae_of_all _ fun z => ?_
    intro θ' _
    rw [Real.norm_eq_abs, abs_mul, abs_mul]
    have hK' := abs_tiltKernel'_le τ hτ hR hs hRs (abs_tiltW_le_one θ' z.1 z.2)
    have hWθ := abs_tiltWtheta_le_two θ' z.1 z.2
    calc |Real.sin z.1| * (|tiltKernel' τ R s (tiltW θ' z.1 z.2)|
          * |-Real.sin θ' * Real.cos z.1 + Real.cos θ' * Real.sin z.1 * Real.cos z.2|)
        ≤ 1 * ((R * s / τ / |R - s|) * 2) := by
          refine mul_le_mul (Real.abs_sin_le_one _) ?_ (by positivity) zero_le_one
          exact mul_le_mul hK' hWθ (abs_nonneg _) hCK
      _ = (R * s / τ / |R - s|) * 2 := by ring
  · exact integrable_const _
  · -- pointwise differentiability
    refine ae_of_all _ fun z => ?_
    intro θ' _
    have hpos := radicand_pos hR hs hRs (abs_tiltW_le_one θ' z.1 z.2)
    have hK := hasDerivAt_tiltKernel τ R s hpos
    have hW := hasDerivAt_tiltW_theta θ' z.1 z.2
    exact (hK.comp θ' hW).const_mul (Real.sin z.1)

/-- The polar-angle integration by parts: `∫₀^π sin t·∂ₜG dt = −∫₀^π cos t·G dt`
(boundary terms vanish at `sin 0 = sin π = 0`). -/
lemma tilt_IBP_t (τ : ℝ) {R s : ℝ} (hR : 0 ≤ R) (hs : 0 ≤ s) (hRs : R ≠ s)
    (θ φ : ℝ) :
    (∫ t in Ioc (0 : ℝ) Real.pi, Real.sin t * (tiltKernel' τ R s (tiltW θ t φ)
      * (-(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ)))
    = -∫ t in Ioc (0 : ℝ) Real.pi, Real.cos t * tiltKernel τ R s (tiltW θ t φ) := by
  have hle : (0 : ℝ) ≤ Real.pi := Real.pi_pos.le
  rw [← intervalIntegral.integral_of_le hle, ← intervalIntegral.integral_of_le hle]
  have hIBP := intervalIntegral.integral_mul_deriv_eq_deriv_mul
    (u := Real.sin) (u' := Real.cos)
    (v := fun t => tiltKernel τ R s (tiltW θ t φ))
    (v' := fun t => tiltKernel' τ R s (tiltW θ t φ)
      * (-(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ))
    (a := 0) (b := Real.pi)
    (fun x _ => Real.hasDerivAt_sin x)
    (fun x _ => by
      have hpos := radicand_pos hR hs hRs (abs_tiltW_le_one θ x φ)
      have h := (hasDerivAt_tiltKernel τ R s hpos).comp x (hasDerivAt_tiltW_t θ x φ)
      exact h)
    (Real.continuous_cos.intervalIntegrable 0 Real.pi)
    (by
      have hc : Continuous fun t : ℝ => tiltKernel' τ R s (tiltW θ t φ)
          * (-(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ) := by
        have h1 : Continuous fun t : ℝ => tiltKernel' τ R s (tiltW θ t φ) := by
          have h2 := continuous_tiltKernel'_comp τ R s hR hs hRs θ
          exact h2.comp (f := fun t : ℝ => ((t, φ) : ℝ × ℝ)) (by fun_prop)
        fun_prop
      exact hc.intervalIntegrable 0 Real.pi)
  rw [hIBP, Real.sin_pi, Real.sin_zero]
  ring

/-- The azimuthal integration by parts: `∫_{−π}^π sin φ·∂φG dφ = −∫ cos φ·G dφ`
(boundary terms vanish at `sin(±π) = 0`). -/
lemma tilt_IBP_phi (τ : ℝ) {R s : ℝ} (hR : 0 ≤ R) (hs : 0 ≤ s) (hRs : R ≠ s)
    (θ t : ℝ) :
    (∫ φ in Ioc (-Real.pi) Real.pi, Real.sin φ * (tiltKernel' τ R s (tiltW θ t φ)
      * (Real.sin θ * Real.sin t * -Real.sin φ)))
    = -∫ φ in Ioc (-Real.pi) Real.pi, Real.cos φ * tiltKernel τ R s (tiltW θ t φ) := by
  have hle : -Real.pi ≤ Real.pi := by linarith [Real.pi_pos]
  rw [← intervalIntegral.integral_of_le hle, ← intervalIntegral.integral_of_le hle]
  have hIBP := intervalIntegral.integral_mul_deriv_eq_deriv_mul
    (u := Real.sin) (u' := Real.cos)
    (v := fun φ => tiltKernel τ R s (tiltW θ t φ))
    (v' := fun φ => tiltKernel' τ R s (tiltW θ t φ)
      * (Real.sin θ * Real.sin t * -Real.sin φ))
    (a := -Real.pi) (b := Real.pi)
    (fun x _ => Real.hasDerivAt_sin x)
    (fun x _ => by
      have hpos := radicand_pos hR hs hRs (abs_tiltW_le_one θ t x)
      have h := (hasDerivAt_tiltKernel τ R s hpos).comp x (hasDerivAt_tiltW_phi θ t x)
      exact h)
    (Real.continuous_cos.intervalIntegrable _ _)
    (by
      have hc : Continuous fun φ : ℝ => tiltKernel' τ R s (tiltW θ t φ)
          * (Real.sin θ * Real.sin t * -Real.sin φ) := by
        have h1 : Continuous fun φ : ℝ => tiltKernel' τ R s (tiltW θ t φ) := by
          have h2 := continuous_tiltKernel'_comp τ R s hR hs hRs θ
          exact h2.comp (f := fun φ : ℝ => ((t, φ) : ℝ × ℝ)) (by fun_prop)
        fun_prop
      exact hc.intervalIntegrable _ _)
  rw [hIBP, Real.sin_pi, Real.sin_neg, Real.sin_pi]
  ring

lemma abs_tiltWt_le_two (θ t φ : ℝ) :
    |-(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ| ≤ 2 := by
  have h1 := Real.abs_cos_le_one θ
  have h2 := Real.abs_sin_le_one t
  have h3 := Real.abs_sin_le_one θ
  have h4 := Real.abs_cos_le_one t
  have h5 := Real.abs_cos_le_one φ
  calc |-(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ|
      ≤ |-(Real.cos θ * Real.sin t)| + |Real.sin θ * Real.cos t * Real.cos φ| :=
        abs_add_le _ _
    _ = |Real.cos θ| * |Real.sin t| + |Real.sin θ| * |Real.cos t| * |Real.cos φ| := by
        rw [abs_neg, abs_mul, abs_mul, abs_mul]
    _ ≤ 1 + 1 := by
        refine add_le_add ?_ ?_
        · nlinarith [abs_nonneg (Real.cos θ), abs_nonneg (Real.sin t)]
        · nlinarith [abs_nonneg (Real.sin θ), abs_nonneg (Real.cos t),
            abs_nonneg (Real.cos φ),
            mul_nonneg (abs_nonneg (Real.sin θ)) (abs_nonneg (Real.cos t))]
    _ = 2 := by norm_num

lemma abs_tiltWphi_le_two (θ t φ : ℝ) :
    |Real.sin θ * Real.sin t * -Real.sin φ| ≤ 2 := by
  have h1 := Real.abs_sin_le_one θ
  have h2 := Real.abs_sin_le_one t
  have h3 := Real.abs_sin_le_one φ
  have habs : |Real.sin θ * Real.sin t * -Real.sin φ|
      = |Real.sin θ| * |Real.sin t| * |Real.sin φ| := by
    rw [abs_mul, abs_mul, abs_neg]
  rw [habs]
  nlinarith [abs_nonneg (Real.sin θ), abs_nonneg (Real.sin t), abs_nonneg (Real.sin φ),
    mul_nonneg (abs_nonneg (Real.sin θ)) (abs_nonneg (Real.sin t))]

/-- **The flow derivative vanishes**: the two integrations by parts produce the
same bulk term `∫∫ cos t·cos φ·G` with opposite signs. -/
lemma tiltJderiv_eq_zero (τ : ℝ) (hτ : 0 < τ) {R s : ℝ}
    (hR : 0 ≤ R) (hs : 0 ≤ s) (hRs : R ≠ s) (θ : ℝ) :
    tiltJderiv τ R s θ = 0 := by
  have hCK : (0 : ℝ) ≤ R * s / τ / |R - s| := by positivity
  -- the three product-level integrands
  have hcontK' := continuous_tiltKernel'_comp τ R s hR hs hRs θ
  have hcontG := continuous_tiltKernel_comp τ R s θ
  have hfA : Integrable (fun z : ℝ × ℝ => Real.sin z.1 * Real.cos z.2
      * (tiltKernel' τ R s (tiltW θ z.1 z.2)
        * (-(Real.cos θ * Real.sin z.1) + Real.sin θ * Real.cos z.1 * Real.cos z.2)))
      tiltBase := by
    refine ⟨?_, HasFiniteIntegral.of_bounded
      (C := (R * s / τ / |R - s|) * 2) (ae_of_all _ fun z => ?_)⟩
    · have hc : Continuous fun z : ℝ × ℝ => Real.sin z.1 * Real.cos z.2
          * (tiltKernel' τ R s (tiltW θ z.1 z.2)
            * (-(Real.cos θ * Real.sin z.1) + Real.sin θ * Real.cos z.1 * Real.cos z.2)) := by
        fun_prop
      exact hc.aestronglyMeasurable
    · rw [Real.norm_eq_abs, abs_mul, abs_mul, abs_mul]
      have hK' := abs_tiltKernel'_le τ hτ hR hs hRs (abs_tiltW_le_one θ z.1 z.2)
      have hWt := abs_tiltWt_le_two θ z.1 z.2
      calc |Real.sin z.1| * |Real.cos z.2|
            * (|tiltKernel' τ R s (tiltW θ z.1 z.2)|
              * |-(Real.cos θ * Real.sin z.1) + Real.sin θ * Real.cos z.1 * Real.cos z.2|)
          ≤ 1 * 1 * ((R * s / τ / |R - s|) * 2) := by
            refine mul_le_mul ?_ ?_ (by positivity) (by norm_num)
            · exact mul_le_mul (Real.abs_sin_le_one _) (Real.abs_cos_le_one _)
                (abs_nonneg _) zero_le_one
            · exact mul_le_mul hK' hWt (abs_nonneg _) hCK
        _ = (R * s / τ / |R - s|) * 2 := by ring
  have hfB : Integrable (fun z : ℝ × ℝ => Real.cos z.1 * Real.sin z.2
      * (tiltKernel' τ R s (tiltW θ z.1 z.2)
        * (Real.sin θ * Real.sin z.1 * -Real.sin z.2))) tiltBase := by
    refine ⟨?_, HasFiniteIntegral.of_bounded
      (C := (R * s / τ / |R - s|) * 2) (ae_of_all _ fun z => ?_)⟩
    · have hc : Continuous fun z : ℝ × ℝ => Real.cos z.1 * Real.sin z.2
          * (tiltKernel' τ R s (tiltW θ z.1 z.2)
            * (Real.sin θ * Real.sin z.1 * -Real.sin z.2)) := by
        fun_prop
      exact hc.aestronglyMeasurable
    · rw [Real.norm_eq_abs, abs_mul, abs_mul, abs_mul]
      have hK' := abs_tiltKernel'_le τ hτ hR hs hRs (abs_tiltW_le_one θ z.1 z.2)
      have hWφ := abs_tiltWphi_le_two θ z.1 z.2
      calc |Real.cos z.1| * |Real.sin z.2|
            * (|tiltKernel' τ R s (tiltW θ z.1 z.2)|
              * |Real.sin θ * Real.sin z.1 * -Real.sin z.2|)
          ≤ 1 * 1 * ((R * s / τ / |R - s|) * 2) := by
            refine mul_le_mul ?_ ?_ (by positivity) (by norm_num)
            · exact mul_le_mul (Real.abs_cos_le_one _) (Real.abs_sin_le_one _)
                (abs_nonneg _) zero_le_one
            · exact mul_le_mul hK' hWφ (abs_nonneg _) hCK
        _ = (R * s / τ / |R - s|) * 2 := by ring
  have hfX : Integrable (fun z : ℝ × ℝ => Real.cos z.1 * Real.cos z.2
      * tiltKernel τ R s (tiltW θ z.1 z.2)) tiltBase := by
    refine ⟨?_, HasFiniteIntegral.of_bounded (C := 1) (ae_of_all _ fun z => ?_)⟩
    · have hc : Continuous fun z : ℝ × ℝ => Real.cos z.1 * Real.cos z.2
          * tiltKernel τ R s (tiltW θ z.1 z.2) := by
        fun_prop
      exact hc.aestronglyMeasurable
    · rw [Real.norm_eq_abs, abs_mul, abs_mul,
        abs_of_pos (tiltKernel_pos τ R s _)]
      calc |Real.cos z.1| * |Real.cos z.2| * tiltKernel τ R s (tiltW θ z.1 z.2)
          ≤ 1 * 1 * 1 := by
            refine mul_le_mul ?_ (tiltKernel_le_one τ hτ R s _)
              (tiltKernel_pos τ R s _).le (by norm_num)
            exact mul_le_mul (Real.abs_cos_le_one _) (Real.abs_cos_le_one _)
              (abs_nonneg _) zero_le_one
        _ = 1 := by norm_num
  -- split the derivative integrand along the generator identity
  have hsplit : tiltJderiv τ R s θ
      = -(∫ z : ℝ × ℝ, Real.sin z.1 * Real.cos z.2
          * (tiltKernel' τ R s (tiltW θ z.1 z.2)
            * (-(Real.cos θ * Real.sin z.1) + Real.sin θ * Real.cos z.1 * Real.cos z.2))
          ∂tiltBase)
        + ∫ z : ℝ × ℝ, Real.cos z.1 * Real.sin z.2
          * (tiltKernel' τ R s (tiltW θ z.1 z.2)
            * (Real.sin θ * Real.sin z.1 * -Real.sin z.2)) ∂tiltBase := by
    calc tiltJderiv τ R s θ
        = ∫ z : ℝ × ℝ, (-(Real.sin z.1 * Real.cos z.2
            * (tiltKernel' τ R s (tiltW θ z.1 z.2)
              * (-(Real.cos θ * Real.sin z.1)
                + Real.sin θ * Real.cos z.1 * Real.cos z.2)))
          + Real.cos z.1 * Real.sin z.2
            * (tiltKernel' τ R s (tiltW θ z.1 z.2)
              * (Real.sin θ * Real.sin z.1 * -Real.sin z.2))) ∂tiltBase := by
          rw [tiltJderiv]
          refine integral_congr_ae (Filter.Eventually.of_forall fun z => ?_)
          have hgen := tiltW_generator θ z.1 z.2
          linear_combination tiltKernel' τ R s (tiltW θ z.1 z.2) * hgen
      _ = (∫ z : ℝ × ℝ, -(Real.sin z.1 * Real.cos z.2
            * (tiltKernel' τ R s (tiltW θ z.1 z.2)
              * (-(Real.cos θ * Real.sin z.1)
                + Real.sin θ * Real.cos z.1 * Real.cos z.2))) ∂tiltBase)
          + ∫ z : ℝ × ℝ, Real.cos z.1 * Real.sin z.2
            * (tiltKernel' τ R s (tiltW θ z.1 z.2)
              * (Real.sin θ * Real.sin z.1 * -Real.sin z.2)) ∂tiltBase :=
          MeasureTheory.integral_add hfA.neg hfB
      _ = -(∫ z : ℝ × ℝ, Real.sin z.1 * Real.cos z.2
            * (tiltKernel' τ R s (tiltW θ z.1 z.2)
              * (-(Real.cos θ * Real.sin z.1)
                + Real.sin θ * Real.cos z.1 * Real.cos z.2)) ∂tiltBase)
          + ∫ z : ℝ × ℝ, Real.cos z.1 * Real.sin z.2
            * (tiltKernel' τ R s (tiltW θ z.1 z.2)
              * (Real.sin θ * Real.sin z.1 * -Real.sin z.2)) ∂tiltBase := by
          rw [MeasureTheory.integral_neg]
  -- both pieces equal −∫∫ cos t · cos φ · G
  have hA : (∫ z : ℝ × ℝ, Real.sin z.1 * Real.cos z.2
      * (tiltKernel' τ R s (tiltW θ z.1 z.2)
        * (-(Real.cos θ * Real.sin z.1) + Real.sin θ * Real.cos z.1 * Real.cos z.2))
      ∂tiltBase)
      = -∫ z : ℝ × ℝ, Real.cos z.1 * Real.cos z.2
          * tiltKernel τ R s (tiltW θ z.1 z.2) ∂tiltBase := by
    rw [tiltBase] at hfA hfX ⊢
    rw [integral_prod_symm _ hfA, integral_prod_symm _ hfX,
      ← MeasureTheory.integral_neg]
    refine integral_congr_ae (Filter.Eventually.of_forall fun φ => ?_)
    change (∫ t in Ioc (0 : ℝ) Real.pi, Real.sin t * Real.cos φ
        * (tiltKernel' τ R s (tiltW θ t φ)
          * (-(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ)))
      = -(∫ t in Ioc (0 : ℝ) Real.pi, Real.cos t * Real.cos φ
          * tiltKernel τ R s (tiltW θ t φ))
    have hstep1 : (∫ t in Ioc (0 : ℝ) Real.pi, Real.sin t * Real.cos φ
        * (tiltKernel' τ R s (tiltW θ t φ)
          * (-(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ)))
        = Real.cos φ * ∫ t in Ioc (0 : ℝ) Real.pi, Real.sin t
          * (tiltKernel' τ R s (tiltW θ t φ)
            * (-(Real.cos θ * Real.sin t) + Real.sin θ * Real.cos t * Real.cos φ)) := by
      rw [← MeasureTheory.integral_const_mul]
      refine setIntegral_congr_fun measurableSet_Ioc fun t _ => ?_
      ring
    have hstep3 : (∫ t in Ioc (0 : ℝ) Real.pi, Real.cos t * Real.cos φ
        * tiltKernel τ R s (tiltW θ t φ))
        = Real.cos φ * ∫ t in Ioc (0 : ℝ) Real.pi, Real.cos t
          * tiltKernel τ R s (tiltW θ t φ) := by
      rw [← MeasureTheory.integral_const_mul]
      refine setIntegral_congr_fun measurableSet_Ioc fun t _ => ?_
      ring
    rw [hstep1, tilt_IBP_t τ hR hs hRs θ φ, hstep3]
    ring
  have hB : (∫ z : ℝ × ℝ, Real.cos z.1 * Real.sin z.2
      * (tiltKernel' τ R s (tiltW θ z.1 z.2)
        * (Real.sin θ * Real.sin z.1 * -Real.sin z.2)) ∂tiltBase)
      = -∫ z : ℝ × ℝ, Real.cos z.1 * Real.cos z.2
          * tiltKernel τ R s (tiltW θ z.1 z.2) ∂tiltBase := by
    rw [tiltBase] at hfB hfX ⊢
    rw [integral_prod _ hfB, integral_prod _ hfX,
      ← MeasureTheory.integral_neg]
    refine integral_congr_ae (Filter.Eventually.of_forall fun t => ?_)
    change (∫ φ in Ioc (-Real.pi) Real.pi, Real.cos t * Real.sin φ
        * (tiltKernel' τ R s (tiltW θ t φ)
          * (Real.sin θ * Real.sin t * -Real.sin φ)))
      = -(∫ φ in Ioc (-Real.pi) Real.pi, Real.cos t * Real.cos φ
          * tiltKernel τ R s (tiltW θ t φ))
    have hstep1 : (∫ φ in Ioc (-Real.pi) Real.pi, Real.cos t * Real.sin φ
        * (tiltKernel' τ R s (tiltW θ t φ)
          * (Real.sin θ * Real.sin t * -Real.sin φ)))
        = Real.cos t * ∫ φ in Ioc (-Real.pi) Real.pi, Real.sin φ
          * (tiltKernel' τ R s (tiltW θ t φ)
            * (Real.sin θ * Real.sin t * -Real.sin φ)) := by
      rw [← MeasureTheory.integral_const_mul]
      refine setIntegral_congr_fun measurableSet_Ioc fun φ _ => ?_
      ring
    have hstep3 : (∫ φ in Ioc (-Real.pi) Real.pi, Real.cos t * Real.cos φ
        * tiltKernel τ R s (tiltW θ t φ))
        = Real.cos t * ∫ φ in Ioc (-Real.pi) Real.pi, Real.cos φ
          * tiltKernel τ R s (tiltW θ t φ) := by
      rw [← MeasureTheory.integral_const_mul]
      refine setIntegral_congr_fun measurableSet_Ioc fun φ _ => ?_
      ring
    rw [hstep1, tilt_IBP_phi τ hR hs hRs θ t, hstep3]
    ring
  rw [hsplit, hA, hB]
  ring

/-- **The tilted zonal average is constant**: `J(θ) = J(0)` for every tilt. -/
theorem tiltJ_const (τ : ℝ) (hτ : 0 < τ) {R s : ℝ}
    (hR : 0 ≤ R) (hs : 0 ≤ s) (hRs : R ≠ s) (θ : ℝ) :
    tiltJ τ R s θ = tiltJ τ R s 0 := by
  have hderiv : ∀ θ' : ℝ, HasDerivAt (tiltJ τ R s) 0 θ' := by
    intro θ'
    have h := hasDerivAt_tiltJ τ hτ hR hs hRs θ'
    rwa [tiltJderiv_eq_zero τ hτ hR hs hRs θ'] at h
  have hMVT := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (f := tiltJ τ R s) (f' := fun _ => (0 : ℝ))
    (s := Icc (min θ 0) (max θ 0)) (C := 0)
    (fun x _ => (hderiv x).hasDerivWithinAt)
    (fun x _ => by simp) (convex_Icc _ _)
    (Set.mem_Icc.mpr ⟨min_le_right θ 0, le_max_right θ 0⟩)
    (Set.mem_Icc.mpr ⟨min_le_left θ 0, le_max_left θ 0⟩)
  rw [zero_mul] at hMVT
  have h0 := norm_le_zero_iff.mp hMVT
  have := sub_eq_zero.mp h0
  linarith [this]

/-! ## Evaluation of `J` at `θ = 0` and the azimuth shift -/

/-- The polar-angle pullback `∫₀^π sin t·h(cos t) dt = ∫_{−1}^1 h`. -/
lemma integral_sin_mul_comp_cos (h : ℝ → ℝ) (hh : Continuous h) :
    (∫ t in (0 : ℝ)..Real.pi, Real.sin t * h (Real.cos t))
      = ∫ u in (-1 : ℝ)..1, h u := by
  have hCoV := intervalIntegral.integral_comp_mul_deriv
    (f := Real.cos) (f' := fun t => -Real.sin t) (g := h)
    (a := 0) (b := Real.pi)
    (fun x _ => Real.hasDerivAt_cos x)
    (Real.continuous_sin.neg.continuousOn) hh
  rw [Real.cos_zero, Real.cos_pi] at hCoV
  have hL : (∫ t in (0 : ℝ)..Real.pi, (h ∘ Real.cos) t * -Real.sin t)
      = -∫ t in (0 : ℝ)..Real.pi, Real.sin t * h (Real.cos t) := by
    rw [← intervalIntegral.integral_neg]
    refine intervalIntegral.integral_congr fun t _ => ?_
    simp only [Function.comp_apply]
    ring
  have hR : (∫ u in (1 : ℝ)..(-1), h u) = -∫ u in (-1 : ℝ)..1, h u :=
    intervalIntegral.integral_symm _ _
  rw [hL, hR] at hCoV
  linarith [hCoV]

/-- The azimuth-shift invariance of the circle integral. -/
lemma integral_Ioc_cos_shift (F : ℝ → ℝ) (ψ : ℝ) :
    (∫ φ in Ioc (-Real.pi) Real.pi, F (Real.cos (φ - ψ)))
      = ∫ φ in Ioc (-Real.pi) Real.pi, F (Real.cos φ) := by
  have hle : -Real.pi ≤ Real.pi := by linarith [Real.pi_pos]
  rw [← intervalIntegral.integral_of_le hle, ← intervalIntegral.integral_of_le hle]
  have h1 : (∫ φ in (-Real.pi)..Real.pi, F (Real.cos (φ - ψ)))
      = ∫ φ in (-Real.pi - ψ)..(Real.pi - ψ), F (Real.cos φ) :=
    intervalIntegral.integral_comp_sub_right (fun φ => F (Real.cos φ)) ψ
  have hper : Function.Periodic (fun φ => F (Real.cos φ)) (2 * Real.pi) :=
    Real.cos_periodic.comp F
  have h2 := hper.intervalIntegral_add_eq (-Real.pi - ψ) (-Real.pi)
  rw [show -Real.pi - ψ + 2 * Real.pi = Real.pi - ψ by ring,
    show -Real.pi + 2 * Real.pi = Real.pi by ring] at h2
  rw [h1]
  exact h2

lemma continuous_tiltKernel (τ R s : ℝ) : Continuous (tiltKernel τ R s) := by
  unfold tiltKernel
  fun_prop

lemma tiltKernel_eq_shellDist_exp (τ R s u : ℝ) :
    tiltKernel τ R s u = Real.exp (-(1 / τ) * shellDist R s u) := rfl

/-- `J(0) = 4π·shellZ`: the polar zonal average is the shell kernel average. -/
lemma tiltJ_zero_eq (τ : ℝ) (hτ : 0 < τ) (R s : ℝ) :
    tiltJ τ R s 0 = 4 * Real.pi * shellZ τ R s := by
  have hπ : (0 : ℝ) ≤ Real.pi := Real.pi_pos.le
  have hW0 : ∀ t φ : ℝ, tiltW 0 t φ = Real.cos t := by
    intro t φ
    rw [tiltW, Real.cos_zero, Real.sin_zero]
    ring
  have hint : Integrable (fun z : ℝ × ℝ =>
      Real.sin z.1 * tiltKernel τ R s (tiltW 0 z.1 z.2))
      ((volume.restrict (Ioc (0 : ℝ) Real.pi)).prod
        (volume.restrict (Ioc (-Real.pi) Real.pi))) := by
    refine ⟨(continuous_sin_fst_mul_kernel τ R s 0).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := 1) (ae_of_all _ fun z => ?_)⟩
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos (tiltKernel_pos τ R s _)]
    calc |Real.sin z.1| * tiltKernel τ R s (tiltW 0 z.1 z.2)
        ≤ 1 * 1 := mul_le_mul (Real.abs_sin_le_one _)
          (tiltKernel_le_one τ hτ R s _) (tiltKernel_pos τ R s _).le zero_le_one
      _ = 1 := by ring
  have hfe : tiltJ τ R s 0 = ∫ z : ℝ × ℝ,
      Real.sin z.1 * tiltKernel τ R s (tiltW 0 z.1 z.2)
      ∂((volume.restrict (Ioc (0 : ℝ) Real.pi)).prod
        (volume.restrict (Ioc (-Real.pi) Real.pi))) := rfl
  rw [hfe, integral_prod _ hint]
  have hinner : ∀ t : ℝ, (∫ φ in Ioc (-Real.pi) Real.pi,
        Real.sin t * tiltKernel τ R s (tiltW 0 t φ))
      = (2 * Real.pi) * (Real.sin t * tiltKernel τ R s (Real.cos t)) := by
    intro t
    have hc : ∀ φ ∈ Ioc (-Real.pi) Real.pi,
        Real.sin t * tiltKernel τ R s (tiltW 0 t φ)
          = Real.sin t * tiltKernel τ R s (Real.cos t) := fun φ _ => by rw [hW0]
    rw [setIntegral_congr_fun measurableSet_Ioc hc, setIntegral_const,
      Real.volume_real_Ioc_of_le (by linarith [Real.pi_pos]), smul_eq_mul]
    ring
  rw [setIntegral_congr_fun measurableSet_Ioc (fun t _ => hinner t),
    MeasureTheory.integral_const_mul, ← intervalIntegral.integral_of_le hπ,
    integral_sin_mul_comp_cos (tiltKernel τ R s) (continuous_tiltKernel τ R s),
    intervalIntegral.integral_of_le (by norm_num : (-1 : ℝ) ≤ 1)]
  have hcong : (∫ u in Ioc (-1 : ℝ) 1, tiltKernel τ R s u)
      = ∫ u in Ioc (-1 : ℝ) 1, Real.exp (-(1 / τ) * shellDist R s u) := by
    refine setIntegral_congr_fun measurableSet_Ioc fun u _ => ?_
    exact tiltKernel_eq_shellDist_exp τ R s u
  rw [hcong, shellZ]
  ring

/-! ## The tilted per-shell identity -/

lemma continuous_phi_slice (τ : ℝ) (hτ : 0 < τ) (R s θ₀ : ℝ) :
    Continuous fun u : ℝ => ∫ φ in Ioc (-Real.pi) Real.pi,
      tiltKernel τ R s (Real.cos θ₀ * u
        + Real.sin θ₀ * Real.sqrt (1 - u ^ 2) * Real.cos φ) := by
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-Real.pi) Real.pi)) :=
    ⟨by rw [Measure.restrict_apply_univ, Real.volume_Ioc]; exact ENNReal.ofReal_lt_top⟩
  refine continuous_of_dominated (bound := fun _ => (1 : ℝ)) (fun u => ?_) (fun u => ?_) ?_ ?_
  · have hc : Continuous fun φ : ℝ => tiltKernel τ R s (Real.cos θ₀ * u
        + Real.sin θ₀ * Real.sqrt (1 - u ^ 2) * Real.cos φ) := by
      unfold tiltKernel
      fun_prop
    exact hc.aestronglyMeasurable
  · refine ae_of_all _ fun φ => ?_
    rw [Real.norm_eq_abs, abs_of_pos (tiltKernel_pos τ R s _)]
    exact tiltKernel_le_one τ hτ R s _
  · exact integrable_const 1
  · refine ae_of_all _ fun φ => ?_
    have hc : Continuous fun u : ℝ => tiltKernel τ R s (Real.cos θ₀ * u
        + Real.sin θ₀ * Real.sqrt (1 - u ^ 2) * Real.cos φ) := by
      unfold tiltKernel
      fun_prop
    exact hc

/-- **The tilted per-shell identity** (non-collision): the chart average of the
tilted kernel about ANY axis equals the polar shell average. -/
lemma chartBase_tilted_eq_shellZ (τ : ℝ) (hτ : 0 < τ) {R s : ℝ}
    (hR : 0 ≤ R) (hs : 0 ≤ s) (hRs : R ≠ s) (θ₀ ψ : ℝ) :
    (∫ w : ℝ × ℝ, tiltKernel τ R s (Real.cos θ₀ * w.1
      + Real.sin θ₀ * Real.sqrt (1 - w.1 ^ 2) * Real.cos (w.2 - ψ)) ∂chartBase)
      = shellZ τ R s := by
  haveI hfin1 : IsFiniteMeasure (volume.restrict (Ioc (-1 : ℝ) 1)) :=
    ⟨by rw [Measure.restrict_apply_univ, Real.volume_Ioc]; exact ENNReal.ofReal_lt_top⟩
  haveI hfin2 : IsFiniteMeasure (volume.restrict (Ioc (-Real.pi) Real.pi)) :=
    ⟨by rw [Measure.restrict_apply_univ, Real.volume_Ioc]; exact ENNReal.ofReal_lt_top⟩
  have hint : Integrable (fun w : ℝ × ℝ => tiltKernel τ R s (Real.cos θ₀ * w.1
      + Real.sin θ₀ * Real.sqrt (1 - w.1 ^ 2) * Real.cos (w.2 - ψ)))
      ((volume.restrict (Ioc (-1 : ℝ) 1)).prod
        (volume.restrict (Ioc (-Real.pi) Real.pi))) := by
    refine ⟨?_, HasFiniteIntegral.of_bounded (C := 1) (ae_of_all _ fun w => ?_)⟩
    · have hc : Continuous fun w : ℝ × ℝ => tiltKernel τ R s (Real.cos θ₀ * w.1
          + Real.sin θ₀ * Real.sqrt (1 - w.1 ^ 2) * Real.cos (w.2 - ψ)) := by
        unfold tiltKernel
        fun_prop
      exact hc.aestronglyMeasurable
    · rw [Real.norm_eq_abs, abs_of_pos (tiltKernel_pos τ R s _)]
      exact tiltKernel_le_one τ hτ R s _
  rw [chartBase, MeasureTheory.integral_smul_measure, integral_prod _ hint]
  have hshift : ∀ u : ℝ, (∫ φ in Ioc (-Real.pi) Real.pi,
      tiltKernel τ R s (Real.cos θ₀ * u
        + Real.sin θ₀ * Real.sqrt (1 - u ^ 2) * Real.cos (φ - ψ)))
      = ∫ φ in Ioc (-Real.pi) Real.pi, tiltKernel τ R s (Real.cos θ₀ * u
        + Real.sin θ₀ * Real.sqrt (1 - u ^ 2) * Real.cos φ) := fun u =>
    integral_Ioc_cos_shift (fun c => tiltKernel τ R s (Real.cos θ₀ * u
      + Real.sin θ₀ * Real.sqrt (1 - u ^ 2) * c)) ψ
  rw [setIntegral_congr_fun measurableSet_Ioc (fun u _ => hshift u)]
  have hintJ : Integrable (fun z : ℝ × ℝ =>
      Real.sin z.1 * tiltKernel τ R s (tiltW θ₀ z.1 z.2))
      ((volume.restrict (Ioc (0 : ℝ) Real.pi)).prod
        (volume.restrict (Ioc (-Real.pi) Real.pi))) := by
    refine ⟨(continuous_sin_fst_mul_kernel τ R s θ₀).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := 1) (ae_of_all _ fun z => ?_)⟩
    rw [Real.norm_eq_abs, abs_mul, abs_of_pos (tiltKernel_pos τ R s _)]
    calc |Real.sin z.1| * tiltKernel τ R s (tiltW θ₀ z.1 z.2)
        ≤ 1 * 1 := mul_le_mul (Real.abs_sin_le_one _)
          (tiltKernel_le_one τ hτ R s _) (tiltKernel_pos τ R s _).le zero_le_one
      _ = 1 := by ring
  have htJ : tiltJ τ R s θ₀ = ∫ u in Ioc (-1 : ℝ) 1,
      (∫ φ in Ioc (-Real.pi) Real.pi, tiltKernel τ R s (Real.cos θ₀ * u
        + Real.sin θ₀ * Real.sqrt (1 - u ^ 2) * Real.cos φ)) := by
    have hfe : tiltJ τ R s θ₀ = ∫ z : ℝ × ℝ, Real.sin z.1
        * tiltKernel τ R s (tiltW θ₀ z.1 z.2)
        ∂((volume.restrict (Ioc (0 : ℝ) Real.pi)).prod
          (volume.restrict (Ioc (-Real.pi) Real.pi))) := rfl
    rw [hfe, integral_prod _ hintJ]
    have hinner : ∀ t ∈ Ioc (0 : ℝ) Real.pi,
        (∫ φ in Ioc (-Real.pi) Real.pi,
          Real.sin t * tiltKernel τ R s (tiltW θ₀ t φ))
        = Real.sin t * ∫ φ in Ioc (-Real.pi) Real.pi, tiltKernel τ R s
            (Real.cos θ₀ * Real.cos t
              + Real.sin θ₀ * Real.sqrt (1 - Real.cos t ^ 2) * Real.cos φ) := by
      intro t ht
      have hsin : Real.sqrt (1 - Real.cos t ^ 2) = Real.sin t := by
        rw [show (1 : ℝ) - Real.cos t ^ 2 = Real.sin t ^ 2 by
          linear_combination -(Real.sin_sq_add_cos_sq t)]
        exact Real.sqrt_sq (Real.sin_nonneg_of_nonneg_of_le_pi ht.1.le ht.2)
      rw [← MeasureTheory.integral_const_mul]
      refine setIntegral_congr_fun measurableSet_Ioc fun φ _ => ?_
      rw [hsin, tiltW]
    rw [setIntegral_congr_fun measurableSet_Ioc hinner,
      ← intervalIntegral.integral_of_le Real.pi_pos.le,
      integral_sin_mul_comp_cos _ (continuous_phi_slice τ hτ R s θ₀),
      intervalIntegral.integral_of_le (by norm_num : (-1 : ℝ) ≤ 1)]
  rw [← htJ, tiltJ_const τ hτ hR hs hRs θ₀, tiltJ_zero_eq τ hτ R s,
    ENNReal.toReal_inv, ENNReal.toReal_ofReal (by positivity), smul_eq_mul]
  have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp

/-- Collision-extended tilted per-shell identity (for a positive probe radius,
by continuity in the shell radius). -/
lemma chartBase_tilted_eq_shellZ' (τ : ℝ) (hτ : 0 < τ) {R : ℝ} (hR : 0 < R)
    {s : ℝ} (hs : 0 ≤ s) (θ₀ ψ : ℝ) :
    (∫ w : ℝ × ℝ, tiltKernel τ R s (Real.cos θ₀ * w.1
      + Real.sin θ₀ * Real.sqrt (1 - w.1 ^ 2) * Real.cos (w.2 - ψ)) ∂chartBase)
      = shellZ τ R s := by
  rcases ne_or_eq R s with hne | heq
  · exact chartBase_tilted_eq_shellZ τ hτ hR.le hs hne θ₀ ψ
  · subst heq
    have hcontL : ContinuousAt (fun s' : ℝ =>
        ∫ w : ℝ × ℝ, tiltKernel τ R s' (Real.cos θ₀ * w.1
          + Real.sin θ₀ * Real.sqrt (1 - w.1 ^ 2) * Real.cos (w.2 - ψ))
          ∂chartBase) R := by
      refine continuousAt_of_dominated (bound := fun _ => (1 : ℝ))
        (Filter.Eventually.of_forall fun s' => ?_)
        (Filter.Eventually.of_forall fun s' => ae_of_all _ fun w => ?_)
        (integrable_const 1) (ae_of_all _ fun w => ?_)
      · have hc : Continuous fun w : ℝ × ℝ => tiltKernel τ R s' (Real.cos θ₀ * w.1
            + Real.sin θ₀ * Real.sqrt (1 - w.1 ^ 2) * Real.cos (w.2 - ψ)) := by
          unfold tiltKernel
          fun_prop
        exact hc.aestronglyMeasurable
      · rw [Real.norm_eq_abs, abs_of_pos (tiltKernel_pos τ R s' _)]
        exact tiltKernel_le_one τ hτ R s' _
      · have hc : Continuous fun s' : ℝ => tiltKernel τ R s' (Real.cos θ₀ * w.1
            + Real.sin θ₀ * Real.sqrt (1 - w.1 ^ 2) * Real.cos (w.2 - ψ)) := by
          unfold tiltKernel
          fun_prop
        exact hc.continuousAt
    have hcontR : ContinuousAt (shellZ τ R) R :=
      (continuous_shellZ τ hτ R).continuousAt
    have heq' : (fun s' : ℝ => ∫ w : ℝ × ℝ, tiltKernel τ R s' (Real.cos θ₀ * w.1
        + Real.sin θ₀ * Real.sqrt (1 - w.1 ^ 2) * Real.cos (w.2 - ψ)) ∂chartBase)
        =ᶠ[𝓝[≠] R] shellZ τ R := by
      have hpos0 : ∀ᶠ s' in 𝓝 R, 0 < s' := Ioi_mem_nhds hR
      have hpos : ∀ᶠ s' in 𝓝[≠] R, 0 < s' := hpos0.filter_mono nhdsWithin_le_nhds
      filter_upwards [hpos, self_mem_nhdsWithin] with s' hs' hne'
      exact chartBase_tilted_eq_shellZ τ hτ hR.le hs'.le
        (Ne.symm hne') θ₀ ψ
    exact tendsto_nhds_unique_of_eventuallyEq
      (hcontL.tendsto.mono_left nhdsWithin_le_nhds)
      (hcontR.tendsto.mono_left nhdsWithin_le_nhds) heq'

/-! ## The master radiality theorem -/

lemma integral_chartBase_congr {G H : ℝ × ℝ → ℝ}
    (h : ∀ u φ : ℝ, u ∈ Ioc (-1 : ℝ) 1 → G (u, φ) = H (u, φ)) :
    ∫ w : ℝ × ℝ, G w ∂chartBase = ∫ w : ℝ × ℝ, H w ∂chartBase := by
  rw [chartBase]
  refine integral_congr_ae ?_
  have hnull : ((volume.restrict (Ioc (-1 : ℝ) 1)).prod
      (volume.restrict (Ioc (-Real.pi) Real.pi)))
      {w : ℝ × ℝ | w.1 ∉ Ioc (-1 : ℝ) 1} = 0 := by
    have hsub : {w : ℝ × ℝ | w.1 ∉ Ioc (-1 : ℝ) 1}
        ⊆ (Ioc (-1 : ℝ) 1)ᶜ ×ˢ (univ : Set ℝ) := by
      rintro ⟨u, φ⟩ hu
      exact ⟨hu, mem_univ _⟩
    refine measure_mono_null hsub ?_
    rw [Measure.prod_prod, Measure.restrict_apply (measurableSet_Ioc.compl)]
    rw [compl_inter_self, measure_empty, zero_mul]
  have hae : ∀ᵐ w : ℝ × ℝ ∂((volume.restrict (Ioc (-1 : ℝ) 1)).prod
      (volume.restrict (Ioc (-Real.pi) Real.pi))), w.1 ∈ Ioc (-1 : ℝ) 1 := by
    rw [ae_iff]
    exact hnull
  exact Measure.ae_smul_measure (hae.mono fun w hw => h w.1 w.2 hw) _

/-- Coordinates are dominated by the norm. -/
lemma abs_coord_le_norm (x : EuclideanSpace ℝ (Fin 3)) (i : Fin 3) :
    |x i| ≤ ‖x‖ := by
  have := PiLp.norm_apply_le x i
  simpa [Real.norm_eq_abs] using this

/-- The general probe-to-chart distance expansion. -/
lemma dist_sq_smul_sphereChart {u : ℝ} (hu : u ^ 2 ≤ 1)
    (x : EuclideanSpace ℝ (Fin 3)) (s φ : ℝ) :
    ‖x - s • sphereChart u φ‖ ^ 2
      = ‖x‖ ^ 2 + s ^ 2 - 2 * s * (x 0 * u
        + Real.sqrt (1 - u ^ 2) * (x 1 * Real.cos φ + x 2 * Real.sin φ)) := by
  rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq (x := x),
    Fin.sum_univ_three, Fin.sum_univ_three]
  simp only [PiLp.sub_apply, PiLp.smul_apply, sphereChart_apply_zero,
    sphereChart_apply_one, sphereChart_apply_two, smul_eq_mul]
  have hsq : Real.sqrt (1 - u ^ 2) ^ 2 = 1 - u ^ 2 := Real.sq_sqrt (by linarith)
  have hcs := Real.cos_sq_add_sin_sq φ
  linear_combination (s ^ 2 * (Real.cos φ ^ 2 + Real.sin φ ^ 2)) * hsq
    + (s ^ 2 * (1 - u ^ 2)) * hcs

lemma rayProbe_zero_eq : rayProbe 0 = (0 : EuclideanSpace ℝ (Fin 3)) := by
  ext i
  fin_cases i <;> rfl

end DriftingIdentifiability
