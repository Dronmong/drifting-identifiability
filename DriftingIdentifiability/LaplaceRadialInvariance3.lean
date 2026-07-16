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

end DriftingIdentifiability
