import DriftingIdentifiability.LaplaceRadialFoundations
import DriftingIdentifiability.LaplaceCompanion

/-!
# Radial Laplace converse, milestone L5 (v1, `n = 3`): the shell layer

This file is the geometric/measure-theoretic foundation (`S-layer`) of the
paper-faithful ℓ²/radial higher-dimensional Laplace converse, specialised to
`E = EuclideanSpace ℝ (Fin 3)` — the canonical case in which the zonal density
of the sphere is *uniform* (Archimedes' hat-box theorem).

The design record is `LaplaceHigherDim.md` §4.10 (discoveries F1–F11).  The key
constructions here:

* `sphereChart u φ` — an explicit chart of the unit sphere `S² ⊂ ℝ³` by
  `(u, φ) ∈ [-1,1] × [-π,π]`, `u = cos(polar angle)`.  Archimedes: the `u`-
  marginal of the uniform sphere measure is *uniform* at `n = 3`.
* `radialMixture₃ ν` — the rotation-invariant probability measure on `ℝ³` whose
  radial profile is `ν` (a probability measure on `ℝ` supported in `[0,∞)`),
  built as a single pushforward of `ν ⊗ chartBase`.  No `Measure.toSphere`,
  no `Measure.bind`, no zonal-density construction.
* the **integral-collapse** identity reducing every integral against
  `radialMixture₃ ν` to `∫ s, (⨍_{u∈[-1,1]} f(s • Φ(u,φ))) dν` — the φ-integral
  is trivial because the probe distance `‖r•e₁ - s•Φ(u,φ)‖² = r² + s² - 2rsu`
  is φ-independent.

Downstream files (`LaplaceRadialRay3`, `…System3`, `…Invariance3`,
`…Converse3`) build the ray objects, the Abel system, and the endgame on top of
this layer.
-/

open MeasureTheory Filter Topology Set
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

/-! ## The unit-sphere chart `Φ(u,φ)` and the ray probe -/

/-- The explicit chart of the unit sphere `S² ⊂ ℝ³`:
`Φ(u,φ) = (u, √(1-u²)·cos φ, √(1-u²)·sin φ)`, where `u = cos(polar angle)` is
the height and `φ` the azimuth.  On `u ∈ [-1,1]` this is a genuine unit
vector. -/
noncomputable def sphereChart (u φ : ℝ) : EuclideanSpace ℝ (Fin 3) :=
  !₂[u, Real.sqrt (1 - u ^ 2) * Real.cos φ, Real.sqrt (1 - u ^ 2) * Real.sin φ]

/-- The ray probe `r • e₁ = (r, 0, 0)`. -/
noncomputable def rayProbe (r : ℝ) : EuclideanSpace ℝ (Fin 3) := !₂[r, 0, 0]

@[simp] lemma sphereChart_apply_zero (u φ : ℝ) : sphereChart u φ 0 = u := rfl

@[simp] lemma sphereChart_apply_one (u φ : ℝ) :
    sphereChart u φ 1 = Real.sqrt (1 - u ^ 2) * Real.cos φ := rfl

@[simp] lemma sphereChart_apply_two (u φ : ℝ) :
    sphereChart u φ 2 = Real.sqrt (1 - u ^ 2) * Real.sin φ := rfl

@[simp] lemma rayProbe_apply_zero (r : ℝ) : rayProbe r 0 = r := rfl
@[simp] lemma rayProbe_apply_one (r : ℝ) : rayProbe r 1 = 0 := rfl
@[simp] lemma rayProbe_apply_two (r : ℝ) : rayProbe r 2 = 0 := rfl

/-- The chart is jointly continuous in `(u, φ)`. -/
lemma continuous_sphereChart : Continuous fun p : ℝ × ℝ => sphereChart p.1 p.2 := by
  have htoLp : Continuous
      (fun v : Fin 3 → ℝ => (WithLp.toLp 2 v : EuclideanSpace ℝ (Fin 3))) := by
    fun_prop
  have h : Continuous fun p : ℝ × ℝ =>
      (![p.1, Real.sqrt (1 - p.1 ^ 2) * Real.cos p.2,
             Real.sqrt (1 - p.1 ^ 2) * Real.sin p.2] : Fin 3 → ℝ) := by
    refine continuous_pi fun i => ?_
    fin_cases i
    · change Continuous fun p : ℝ × ℝ => p.1
      exact continuous_fst
    · change Continuous fun p : ℝ × ℝ => Real.sqrt (1 - p.1 ^ 2) * Real.cos p.2
      fun_prop
    · change Continuous fun p : ℝ × ℝ => Real.sqrt (1 - p.1 ^ 2) * Real.sin p.2
      fun_prop
  exact htoLp.comp h

/-- `‖Φ(u,φ)‖² = 1` for `u ∈ [-1,1]` (the chart lands on the unit sphere). -/
lemma sphereChart_normSq {u : ℝ} (hu : u ^ 2 ≤ 1) (φ : ℝ) :
    ‖sphereChart u φ‖ ^ 2 = 1 := by
  rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
  simp only [sphereChart_apply_zero, sphereChart_apply_one, sphereChart_apply_two]
  have hsq : Real.sqrt (1 - u ^ 2) ^ 2 = 1 - u ^ 2 :=
    Real.sq_sqrt (by linarith)
  have : (Real.sqrt (1 - u ^ 2) * Real.cos φ) ^ 2
      + (Real.sqrt (1 - u ^ 2) * Real.sin φ) ^ 2
      = (1 - u ^ 2) * (Real.cos φ ^ 2 + Real.sin φ ^ 2) := by
    rw [mul_pow, mul_pow, hsq]; ring
  rw [show u ^ 2 + (Real.sqrt (1 - u ^ 2) * Real.cos φ) ^ 2
        + (Real.sqrt (1 - u ^ 2) * Real.sin φ) ^ 2
      = u ^ 2 + ((Real.sqrt (1 - u ^ 2) * Real.cos φ) ^ 2
        + (Real.sqrt (1 - u ^ 2) * Real.sin φ) ^ 2) by ring, this,
    Real.cos_sq_add_sin_sq]
  ring

/-- **The probe-distance identity.**  `‖r•e₁ - s•Φ(u,φ)‖² = r² + s² - 2rsu`,
independent of the azimuth `φ` — the geometric fact that collapses every ray
integral to a 1-d `u`-integral. -/
lemma dist_sq_rayProbe_smul_sphereChart {u : ℝ} (hu : u ^ 2 ≤ 1) (r s φ : ℝ) :
    ‖rayProbe r - s • sphereChart u φ‖ ^ 2 = r ^ 2 + s ^ 2 - 2 * r * s * u := by
  rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
  simp only [PiLp.sub_apply, PiLp.smul_apply, rayProbe_apply_zero, rayProbe_apply_one,
    rayProbe_apply_two, sphereChart_apply_zero, sphereChart_apply_one, sphereChart_apply_two,
    smul_eq_mul]
  have hsq : Real.sqrt (1 - u ^ 2) ^ 2 = 1 - u ^ 2 :=
    Real.sq_sqrt (by linarith)
  have hcs : Real.cos φ ^ 2 + Real.sin φ ^ 2 = 1 := Real.cos_sq_add_sin_sq φ
  have hexpand :
      (r - s * u) ^ 2
        + (0 - s * (Real.sqrt (1 - u ^ 2) * Real.cos φ)) ^ 2
        + (0 - s * (Real.sqrt (1 - u ^ 2) * Real.sin φ)) ^ 2
      = (r - s * u) ^ 2
        + s ^ 2 * (Real.sqrt (1 - u ^ 2) ^ 2) * (Real.cos φ ^ 2 + Real.sin φ ^ 2) := by
    ring
  rw [hexpand, hsq, hcs]
  ring

/-! ## The radial-mixture measure and its integral collapse -/

/-- The base measure on the chart domain `[-1,1] × [-π,π]`, normalised to a
probability measure (total mass `4π · (4π)⁻¹ = 1`). -/
noncomputable def chartBase : Measure (ℝ × ℝ) :=
  (ENNReal.ofReal (4 * Real.pi))⁻¹ •
    ((volume.restrict (Ioc (-1 : ℝ) 1)).prod (volume.restrict (Ioc (-Real.pi) Real.pi)))

instance : IsProbabilityMeasure chartBase := by
  constructor
  have hπ : (0 : ℝ) < 4 * Real.pi := by positivity
  rw [chartBase, Measure.smul_apply, smul_eq_mul, ← Set.univ_prod_univ, Measure.prod_prod,
    Measure.restrict_apply_univ, Measure.restrict_apply_univ, Real.volume_Ioc, Real.volume_Ioc,
    ← ENNReal.ofReal_mul (by norm_num)]
  rw [show (1 - (-1 : ℝ)) * (Real.pi - -Real.pi) = 4 * Real.pi by ring]
  exact ENNReal.inv_mul_cancel (ENNReal.ofReal_pos.mpr hπ).ne' ENNReal.ofReal_ne_top

/-- The chart pushforward map `(s, u, φ) ↦ s • Φ(u,φ)`. -/
noncomputable def chartMap (z : ℝ × ℝ × ℝ) : EuclideanSpace ℝ (Fin 3) :=
  z.1 • sphereChart z.2.1 z.2.2

@[simp] lemma chartMap_mk (s u φ : ℝ) : chartMap (s, u, φ) = s • sphereChart u φ := rfl

lemma chartMap_mk_pair (s : ℝ) (w : ℝ × ℝ) :
    chartMap (s, w) = s • sphereChart w.1 w.2 := rfl

lemma continuous_chartMap : Continuous chartMap := by
  unfold chartMap
  exact continuous_fst.smul
    (continuous_sphereChart.comp (f := fun z : ℝ × ℝ × ℝ => (z.2.1, z.2.2)) (by fun_prop))

/-- **The radial-mixture measure**: the rotation-invariant probability measure on
`ℝ³` with radial profile `ν`, as a single pushforward of `ν ⊗ chartBase`.
No `Measure.toSphere`, no `Measure.bind`, no zonal-density construction. -/
noncomputable def radialMixture₃ (ν : Measure ℝ) : Measure (EuclideanSpace ℝ (Fin 3)) :=
  (ν.prod chartBase).map chartMap

instance radialMixture₃_isProbabilityMeasure
    (ν : Measure ℝ) [IsProbabilityMeasure ν] : IsProbabilityMeasure (radialMixture₃ ν) :=
  Measure.isProbabilityMeasure_map continuous_chartMap.aemeasurable

/-- **Integral collapse.**  Every integral against `radialMixture₃ ν` reduces to a
`ν`-integral of a chart integral over `[-1,1] × [-π,π]`. -/
lemma integral_radialMixture₃ (ν : Measure ℝ) [IsProbabilityMeasure ν]
    {f : EuclideanSpace ℝ (Fin 3) → ℝ} (hf : Integrable f (radialMixture₃ ν)) :
    ∫ y, f y ∂(radialMixture₃ ν)
      = ∫ s, ∫ w : ℝ × ℝ, f (chartMap (s, w)) ∂chartBase ∂ν := by
  have hmap : AEMeasurable chartMap (ν.prod chartBase) := continuous_chartMap.aemeasurable
  have hint : Integrable (fun z => f (chartMap z)) (ν.prod chartBase) :=
    (integrable_map_measure hf.aestronglyMeasurable hmap).mp hf
  rw [radialMixture₃, integral_map hmap hf.aestronglyMeasurable]
  exact integral_prod _ hint

/-- **The zonal (φ-) collapse.**  If `G` is continuous, bounded, and depends only
on `w.1 = u` (through `g`) on the chart support `u ∈ (-1,1]`, then its
`chartBase`-integral is the uniform `u`-average `(1/2)∫_{-1}^{1} g`.  The `4π`
normalisation and the trivial `2π` azimuth integral cancel. -/
lemma integral_chartBase_zonal {G : ℝ × ℝ → ℝ} {g : ℝ → ℝ} {C : ℝ}
    (hG : Continuous G) (hC : ∀ w, |G w| ≤ C)
    (hEq : ∀ u φ, u ∈ Ioc (-1 : ℝ) 1 → G (u, φ) = g u) :
    ∫ w : ℝ × ℝ, G w ∂chartBase = (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1, g u := by
  have hAvol : volume (Ioc (-1 : ℝ) 1) = ENNReal.ofReal 2 := by
    rw [Real.volume_Ioc]; norm_num
  have hBvol : volume (Ioc (-Real.pi) Real.pi) = ENNReal.ofReal (2 * Real.pi) := by
    rw [Real.volume_Ioc]; congr 1; ring
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-1 : ℝ) 1)) :=
    ⟨by rw [Measure.restrict_apply_univ, hAvol]; exact ENNReal.ofReal_lt_top⟩
  haveI : IsFiniteMeasure (volume.restrict (Ioc (-Real.pi) Real.pi)) :=
    ⟨by rw [Measure.restrict_apply_univ, hBvol]; exact ENNReal.ofReal_lt_top⟩
  have hint : Integrable G
      ((volume.restrict (Ioc (-1 : ℝ) 1)).prod (volume.restrict (Ioc (-Real.pi) Real.pi))) :=
    ⟨hG.aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := C)
        (ae_of_all _ (fun w => by rw [Real.norm_eq_abs]; exact hC w))⟩
  rw [chartBase, integral_smul_measure, integral_prod _ hint]
  have hcong : ∀ u ∈ Ioc (-1 : ℝ) 1,
      ∫ φ in Ioc (-Real.pi) Real.pi, G (u, φ) ∂volume = 2 * Real.pi * g u := by
    intro u hu
    rw [setIntegral_congr_fun measurableSet_Ioc (fun φ _ => hEq u φ hu), setIntegral_const,
      Real.volume_real_Ioc_of_le (by linarith [Real.pi_pos]), smul_eq_mul]
    ring
  rw [setIntegral_congr_fun measurableSet_Ioc hcong, integral_const_mul]
  rw [ENNReal.toReal_inv, ENNReal.toReal_ofReal (by positivity), smul_eq_mul]
  have hπ : Real.pi ≠ 0 := Real.pi_ne_zero
  field_simp
  ring

/-! ## Ray objects: the normalizer profile -/

/-- The probe-to-shell distance `d(r,s,u) = √(r² + s² - 2rsu)`. -/
noncomputable def shellDist (r s u : ℝ) : ℝ := Real.sqrt (r ^ 2 + s ^ 2 - 2 * r * s * u)

/-- Per-shell zonal average of the Laplace kernel:
`Z̄(r,s) = (1/2)∫_{-1}^{1} e^{-d(r,s,u)/τ} du`. -/
noncomputable def shellZ (τ r s : ℝ) : ℝ :=
  (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1, Real.exp (-(1 / τ) * shellDist r s u)

/-- The probe-to-chart norm collapses to the shell distance on the chart
support `u² ≤ 1`. -/
lemma norm_rayProbe_sub_smul_sphereChart {u : ℝ} (hu : u ^ 2 ≤ 1) (r s φ : ℝ) :
    ‖rayProbe r - s • sphereChart u φ‖ = shellDist r s u := by
  rw [shellDist, ← dist_sq_rayProbe_smul_sphereChart hu r s φ]
  exact (Real.sqrt_sq (norm_nonneg _)).symm

lemma laplaceKernel_rayProbe_nonneg (τ r : ℝ) (y : EuclideanSpace ℝ (Fin 3)) :
    0 ≤ laplaceKernel τ (rayProbe r) y := (Real.exp_pos _).le

lemma laplaceKernel_rayProbe_le_one (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin 3)) : laplaceKernel τ (rayProbe r) y ≤ 1 := by
  have h0 : -(1 / τ) * ‖rayProbe r - y‖ ≤ 0 := by
    have hnn : 0 ≤ (1 / τ) * ‖rayProbe r - y‖ :=
      mul_nonneg (one_div_pos.mpr hτ).le (norm_nonneg _)
    linarith [neg_mul (1 / τ) (‖rayProbe r - y‖)]
  simp only [laplaceKernel]
  have hle := Real.exp_le_exp.mpr h0
  rwa [Real.exp_zero] at hle

lemma continuous_laplaceKernel_rayProbe (τ r : ℝ) :
    Continuous (fun y : EuclideanSpace ℝ (Fin 3) => laplaceKernel τ (rayProbe r) y) := by
  simp only [laplaceKernel]
  exact Real.continuous_exp.comp (((continuous_const.sub continuous_id).norm).const_mul (-(1 / τ)))

/-- The Laplace kernel from the ray probe to a chart point collapses to a function
of the distance alone (φ-independent) on the chart support. -/
lemma laplaceKernel_rayProbe_chart {u : ℝ} (hu : u ^ 2 ≤ 1) (τ r s φ : ℝ) :
    laplaceKernel τ (rayProbe r) (s • sphereChart u φ)
      = Real.exp (-(1 / τ) * shellDist r s u) := by
  simp only [laplaceKernel]
  rw [norm_rayProbe_sub_smul_sphereChart hu]

/-- **The ray normalizer is a `ν`-mixture of per-shell zonal kernel averages.**
`Z̃_ν(r) = Z_{radialMixture₃ ν}(r•e₁) = ∫ shellZ τ r s dν`. -/
lemma kernelNormalizer_radialMixture₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    kernelNormalizer (laplaceKernel τ) (radialMixture₃ ν) (rayProbe r)
      = ∫ s, shellZ τ r s ∂ν := by
  have hf : Integrable (fun y => laplaceKernel τ (rayProbe r) y) (radialMixture₃ ν) :=
    ⟨(continuous_laplaceKernel_rayProbe τ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := 1)
        (ae_of_all _ (fun y => by
          rw [Real.norm_eq_abs, abs_of_nonneg (laplaceKernel_rayProbe_nonneg τ r y)]
          exact laplaceKernel_rayProbe_le_one τ hτ r y))⟩
  rw [kernelNormalizer, integral_radialMixture₃ ν hf]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))
  refine integral_chartBase_zonal
    (G := fun w : ℝ × ℝ => laplaceKernel τ (rayProbe r) (chartMap (s, w)))
    (g := fun u => Real.exp (-(1 / τ) * shellDist r s u)) (C := 1) ?_ ?_ ?_
  · exact (continuous_laplaceKernel_rayProbe τ r).comp
      (continuous_chartMap.comp (by fun_prop))
  · intro w
    rw [abs_of_nonneg (laplaceKernel_rayProbe_nonneg τ r _)]
    exact laplaceKernel_rayProbe_le_one τ hτ r _
  · intro u φ hu
    have hu2 : u ^ 2 ≤ 1 := by nlinarith [hu.1, hu.2]
    rw [chartMap_mk_pair]
    exact laplaceKernel_rayProbe_chart hu2 τ r s φ

/-! ## Ray objects: the companion normalizer profile -/

/-- Per-shell zonal average of the companion kernel:
`C̄(r,s) = (1/2)∫_{-1}^{1} (τ + d)·e^{-d/τ} du` (Matérn-3/2 profile). -/
noncomputable def shellC (τ r s : ℝ) : ℝ :=
  (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1,
    (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u)

lemma laplaceCompanionKernel_rayProbe_nonneg (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin 3)) : 0 ≤ laplaceCompanionKernel τ (rayProbe r) y := by
  simp only [laplaceCompanionKernel]
  exact mul_nonneg (add_nonneg hτ.le (norm_nonneg _)) (laplaceKernel_rayProbe_nonneg τ r y)

/-- The companion kernel `(τ+d)e^{-d/τ}` is bounded by `τ` (max at `d = 0`), via
`1 + x ≤ eˣ`. -/
lemma laplaceCompanionKernel_rayProbe_le (τ : ℝ) (hτ : 0 < τ) (r : ℝ)
    (y : EuclideanSpace ℝ (Fin 3)) : laplaceCompanionKernel τ (rayProbe r) y ≤ τ := by
  simp only [laplaceCompanionKernel, laplaceKernel]
  set d := ‖rayProbe r - y‖ with hd
  have h1 : τ + d ≤ τ * Real.exp (d / τ) := by
    have hle : d / τ + 1 ≤ Real.exp (d / τ) := Real.add_one_le_exp (d / τ)
    have hcancel : τ * (d / τ + 1) = τ + d := by field_simp [hτ.ne']; ring
    nlinarith [mul_le_mul_of_nonneg_left hle hτ.le, hcancel]
  have hcancel2 : Real.exp (d / τ) * Real.exp (-(1 / τ) * d) = 1 := by
    rw [← Real.exp_add, show d / τ + -(1 / τ) * d = 0 by ring, Real.exp_zero]
  have h3 := mul_le_mul_of_nonneg_right h1 (Real.exp_pos (-(1 / τ) * d)).le
  rw [mul_assoc, hcancel2, mul_one] at h3
  exact h3

lemma continuous_laplaceCompanionKernel_rayProbe (τ r : ℝ) :
    Continuous (fun y : EuclideanSpace ℝ (Fin 3) => laplaceCompanionKernel τ (rayProbe r) y) := by
  simp only [laplaceCompanionKernel]
  exact (continuous_const.add ((continuous_const.sub continuous_id).norm)).mul
    (continuous_laplaceKernel_rayProbe τ r)

lemma laplaceCompanionKernel_rayProbe_chart {u : ℝ} (hu : u ^ 2 ≤ 1) (τ r s φ : ℝ) :
    laplaceCompanionKernel τ (rayProbe r) (s • sphereChart u φ)
      = (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u) := by
  simp only [laplaceCompanionKernel]
  rw [norm_rayProbe_sub_smul_sphereChart hu, laplaceKernel_rayProbe_chart hu]

/-- **The companion normalizer is a `ν`-mixture of per-shell zonal companion
averages.**  `C̃_ν(r) = ∫ shellC τ r s dν`. -/
lemma kernelNormalizer_companion_radialMixture₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    kernelNormalizer (laplaceCompanionKernel τ) (radialMixture₃ ν) (rayProbe r)
      = ∫ s, shellC τ r s ∂ν := by
  have hf : Integrable (fun y => laplaceCompanionKernel τ (rayProbe r) y) (radialMixture₃ ν) :=
    ⟨(continuous_laplaceCompanionKernel_rayProbe τ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := τ)
        (ae_of_all _ (fun y => by
          rw [Real.norm_eq_abs, abs_of_nonneg (laplaceCompanionKernel_rayProbe_nonneg τ hτ r y)]
          exact laplaceCompanionKernel_rayProbe_le τ hτ r y))⟩
  rw [kernelNormalizer, integral_radialMixture₃ ν hf]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))
  refine integral_chartBase_zonal
    (G := fun w : ℝ × ℝ => laplaceCompanionKernel τ (rayProbe r) (chartMap (s, w)))
    (g := fun u => (τ + shellDist r s u) * Real.exp (-(1 / τ) * shellDist r s u)) (C := τ) ?_ ?_ ?_
  · exact (continuous_laplaceCompanionKernel_rayProbe τ r).comp
      (continuous_chartMap.comp (by fun_prop))
  · intro w
    rw [abs_of_nonneg (laplaceCompanionKernel_rayProbe_nonneg τ hτ r _)]
    exact laplaceCompanionKernel_rayProbe_le τ hτ r _
  · intro u φ hu
    have hu2 : u ^ 2 ≤ 1 := by nlinarith [hu.1, hu.2]
    rw [chartMap_mk_pair]
    exact laplaceCompanionKernel_rayProbe_chart hu2 τ r s φ

/-! ## Ray objects: the drift numerator profile -/

/-- Per-shell zonal average of the first coordinate of the Laplace-weighted
displacement:
`D̄(r,s) = (1/2)∫_{-1}^{1} e^{-d(r,s,u)/τ}(s u - r) du`. -/
noncomputable def shellD (τ r s : ℝ) : ℝ :=
  (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1,
    Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r)

/-- The axial displacement coordinate `X = t-r = s u-r` along the probe ray. -/
noncomputable def shellAxial (r s u : ℝ) : ℝ := s * u - r

/-- The squared tangential radius `ρ² = s²(1-u²)` of a shell point relative to
the probe axis. -/
noncomputable def shellRhoSq (s u : ℝ) : ℝ := s ^ 2 * (1 - u ^ 2)

/-- Per-shell zonal average of the axial coordinate `t = s u`, weighted by the
Laplace kernel.  This is the right-hand side kernel in the `T₃` identity. -/
noncomputable def shellT (τ r s : ℝ) : ℝ :=
  (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1,
    Real.exp (-(1 / τ) * shellDist r s u) * (s * u)

/-- Per-shell zonal average of `ρ²/d`, weighted by the Laplace kernel.  This is
the left-hand side kernel in the `T₃` identity.  Division is Lean's total real
division; downstream `r,s>0` identities handle the collision point separately
through the polynomial `d`-substitution. -/
noncomputable def shellRhoSqOverDist (τ r s : ℝ) : ℝ :=
  (1 / 2) * ∫ u in Ioc (-1 : ℝ) 1,
    Real.exp (-(1 / τ) * shellDist r s u) * (shellRhoSq s u / shellDist r s u)

/-- The polynomial numerator for `ρ²` after the reverse distance substitution
`u = (r²+s²-z²)/(2rs)`:
`ρ² = ((z²-(r-s)²)((r+s)²-z²))/(4r²)`. -/
noncomputable def shellRhoPoly (r s z : ℝ) : ℝ :=
  (z ^ 2 - (r - s) ^ 2) * ((r + s) ^ 2 - z ^ 2)

@[simp] lemma shellRhoPoly_left_endpoint (r s : ℝ) :
    shellRhoPoly r s (r - s) = 0 := by
  unfold shellRhoPoly
  ring

@[simp] lemma shellRhoPoly_neg_left_endpoint (r s : ℝ) :
    shellRhoPoly r s (s - r) = 0 := by
  unfold shellRhoPoly
  ring

@[simp] lemma shellRhoPoly_right_endpoint (r s : ℝ) :
    shellRhoPoly r s (r + s) = 0 := by
  unfold shellRhoPoly
  ring

@[simp] lemma shellRhoPoly_abs_left_endpoint (r s : ℝ) :
    shellRhoPoly r s |r - s| = 0 := by
  unfold shellRhoPoly
  rw [sq_abs]
  ring

lemma hasDerivAt_shellRhoPoly (r s z : ℝ) :
    HasDerivAt (fun x : ℝ => shellRhoPoly r s x)
      (4 * z * (r ^ 2 + s ^ 2 - z ^ 2)) z := by
  unfold shellRhoPoly
  have h₁ : HasDerivAt (fun x : ℝ => x ^ 2 - (r - s) ^ 2) (2 * z) z := by
    simpa using (hasDerivAt_pow 2 z).sub_const ((r - s) ^ 2)
  have h₂ : HasDerivAt (fun x : ℝ => (r + s) ^ 2 - x ^ 2) (-(2 * z)) z := by
    simpa using (hasDerivAt_pow 2 z).const_sub ((r + s) ^ 2)
  have h :
      HasDerivAt
        (fun x : ℝ => (x ^ 2 - (r - s) ^ 2) * ((r + s) ^ 2 - x ^ 2))
        ((2 * z) * ((r + s) ^ 2 - z ^ 2) +
          (z ^ 2 - (r - s) ^ 2) * (-(2 * z))) z := h₁.mul h₂
  convert h using 1
  ring

lemma hasDerivAt_exp_neg_inv_mul (τ z : ℝ) :
    HasDerivAt (fun x : ℝ => Real.exp (-(1 / τ) * x))
      (-(1 / τ) * Real.exp (-(1 / τ) * z)) z := by
  have h := ((hasDerivAt_id z).const_mul (-(1 / τ))).exp
  convert h using 1
  · ext x
    rfl
  · simp [id, mul_comm]

lemma hasDerivAt_shellRhoPoly_mul_exp (τ r s z : ℝ) :
    HasDerivAt
      (fun x : ℝ => shellRhoPoly r s x * Real.exp (-(1 / τ) * x))
      ((4 * z * (r ^ 2 + s ^ 2 - z ^ 2)) * Real.exp (-(1 / τ) * z)
        + shellRhoPoly r s z * (-(1 / τ) * Real.exp (-(1 / τ) * z))) z := by
  exact (hasDerivAt_shellRhoPoly r s z).mul (hasDerivAt_exp_neg_inv_mul τ z)

/-- The endpoint-zero FTC identity for the polynomial-exponential primitive
used in the reverse-distance proof of the `T₃` shell identity. -/
lemma integral_deriv_shellRhoPoly_mul_exp_eq_zero (τ r s : ℝ) :
    (∫ z in |r - s|..(r + s),
      deriv (fun x : ℝ => shellRhoPoly r s x * Real.exp (-(1 / τ) * x)) z) = 0 := by
  let a : ℝ := |r - s|
  let b : ℝ := r + s
  let F : ℝ → ℝ := fun x => shellRhoPoly r s x * Real.exp (-(1 / τ) * x)
  let F' : ℝ → ℝ := fun z =>
    (4 * z * (r ^ 2 + s ^ 2 - z ^ 2)) * Real.exp (-(1 / τ) * z)
      + shellRhoPoly r s z * (-(1 / τ) * Real.exp (-(1 / τ) * z))
  have hdiff : ∀ x ∈ Set.uIcc a b, DifferentiableAt ℝ F x := by
    intro x _
    exact (hasDerivAt_shellRhoPoly_mul_exp τ r s x).differentiableAt
  have hderiv : deriv F = F' := by
    funext x
    exact (hasDerivAt_shellRhoPoly_mul_exp τ r s x).deriv
  have hF'int : IntervalIntegrable F' volume a b := by
    apply Continuous.intervalIntegrable
    dsimp [F', shellRhoPoly]
    fun_prop
  have hFint : IntervalIntegrable (deriv F) volume a b := by
    simpa [hderiv] using hF'int
  have hftc := intervalIntegral.integral_deriv_eq_sub
    (a := a) (b := b) (f := F) hdiff hFint
  dsimp [F, a, b] at hftc ⊢
  rw [hftc]
  simp

/-- The explicit polynomial-exponential derivative integrates to zero over the
distance endpoints.  This is the raw integration-by-parts identity used by the
reverse-distance `T₃` proof before rearranging constants. -/
lemma integral_shellRhoPoly_exp_deriv_formula_eq_zero (τ r s : ℝ) :
    (∫ z in |r - s|..(r + s),
      (4 * z * (r ^ 2 + s ^ 2 - z ^ 2)) * Real.exp (-(1 / τ) * z)
        + shellRhoPoly r s z * (-(1 / τ) * Real.exp (-(1 / τ) * z))) = 0 := by
  let F : ℝ → ℝ := fun x => shellRhoPoly r s x * Real.exp (-(1 / τ) * x)
  have hderiv : ∀ z : ℝ,
      deriv F z =
        (4 * z * (r ^ 2 + s ^ 2 - z ^ 2)) * Real.exp (-(1 / τ) * z)
          + shellRhoPoly r s z * (-(1 / τ) * Real.exp (-(1 / τ) * z)) := by
    intro z
    exact (hasDerivAt_shellRhoPoly_mul_exp τ r s z).deriv
  calc
    (∫ z in |r - s|..(r + s),
      (4 * z * (r ^ 2 + s ^ 2 - z ^ 2)) * Real.exp (-(1 / τ) * z)
        + shellRhoPoly r s z * (-(1 / τ) * Real.exp (-(1 / τ) * z)))
        = ∫ z in |r - s|..(r + s), deriv F z := by
          apply intervalIntegral.integral_congr
          intro z _
          exact (hderiv z).symm
    _ = 0 := by
          simpa [F] using integral_deriv_shellRhoPoly_mul_exp_eq_zero τ r s

/-- Rearranged polynomial-exponential identity behind the per-shell `T₃`
formula.  The proof uses the endpoint-zero FTC identity above and only then
splits constants out of the interval integral. -/
lemma shellRhoPoly_integral_identity
    {τ : ℝ} (hτ : 0 < τ) (r s : ℝ) :
    (∫ z in |r - s|..(r + s),
      shellRhoPoly r s z * Real.exp (-(1 / τ) * z)) =
      4 * τ *
        ∫ z in |r - s|..(r + s),
          z * (r ^ 2 + s ^ 2 - z ^ 2) * Real.exp (-(1 / τ) * z) := by
  let F : ℝ → ℝ := fun z =>
    z * (r ^ 2 + s ^ 2 - z ^ 2) * Real.exp (-(1 / τ) * z)
  let G : ℝ → ℝ := fun z =>
    shellRhoPoly r s z * Real.exp (-(1 / τ) * z)
  have hFint : IntervalIntegrable F volume |r - s| (r + s) := by
    apply Continuous.intervalIntegrable
    dsimp [F]
    fun_prop
  have hGint : IntervalIntegrable G volume |r - s| (r + s) := by
    apply Continuous.intervalIntegrable
    dsimp [G, shellRhoPoly]
    fun_prop
  have hraw := integral_shellRhoPoly_exp_deriv_formula_eq_zero τ r s
  have hsplit :
      (∫ z in |r - s|..(r + s),
        (4 * z * (r ^ 2 + s ^ 2 - z ^ 2)) * Real.exp (-(1 / τ) * z)
          + shellRhoPoly r s z * (-(1 / τ) * Real.exp (-(1 / τ) * z))) =
        4 * (∫ z in |r - s|..(r + s), F z)
          - (1 / τ) * (∫ z in |r - s|..(r + s), G z) := by
    calc
      (∫ z in |r - s|..(r + s),
        (4 * z * (r ^ 2 + s ^ 2 - z ^ 2)) * Real.exp (-(1 / τ) * z)
          + shellRhoPoly r s z * (-(1 / τ) * Real.exp (-(1 / τ) * z)))
          = ∫ z in |r - s|..(r + s), 4 * F z + (-(1 / τ)) * G z := by
              apply intervalIntegral.integral_congr
              intro z _
              dsimp [F, G]
              ring
      _ = (∫ z in |r - s|..(r + s), 4 * F z)
            + (∫ z in |r - s|..(r + s), (-(1 / τ)) * G z) := by
              rw [intervalIntegral.integral_add (hFint.const_mul 4)
                (hGint.const_mul (-(1 / τ)))]
      _ = 4 * (∫ z in |r - s|..(r + s), F z)
            + (-(1 / τ)) * (∫ z in |r - s|..(r + s), G z) := by
              rw [intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul]
      _ = 4 * (∫ z in |r - s|..(r + s), F z)
          - (1 / τ) * (∫ z in |r - s|..(r + s), G z) := by
              ring
  rw [hsplit] at hraw
  have hτne : τ ≠ 0 := hτ.ne'
  let I : ℝ := ∫ z in |r - s|..(r + s), F z
  let J : ℝ := ∫ z in |r - s|..(r + s), G z
  have hsumIJ : 4 * I - (1 / τ) * J = 0 := by
    simpa [I, J] using hraw
  have hmain : J = 4 * τ * I := by
    have hτinv : τ * (1 / τ) = 1 := by
      field_simp [hτne]
    have hEq : (1 / τ) * J = 4 * I := by
      linarith
    calc
      J = (τ * (1 / τ)) * J := by rw [hτinv, one_mul]
      _ = τ * ((1 / τ) * J) := by ring
      _ = τ * (4 * I) := by rw [hEq]
      _ = 4 * τ * I := by ring
  simpa [I, J, F, G] using hmain

@[simp] lemma chartMap_mk_pair_apply_zero (s u φ : ℝ) :
    chartMap (s, (u, φ)) 0 = s * u := by
  simp [chartMap_mk_pair]

lemma shellRhoSq_nonneg {u : ℝ} (hu : u ^ 2 ≤ 1) (s : ℝ) :
    0 ≤ shellRhoSq s u := by
  unfold shellRhoSq
  positivity

lemma shellDist_sq_eq {u : ℝ} (hu : u ^ 2 ≤ 1) (r s : ℝ) :
    shellDist r s u ^ 2 = r ^ 2 + s ^ 2 - 2 * r * s * u := by
  unfold shellDist
  have hnonneg : 0 ≤ r ^ 2 + s ^ 2 - 2 * r * s * u := by
    rw [← dist_sq_rayProbe_smul_sphereChart hu r s 0]
    positivity
  exact Real.sq_sqrt hnonneg

/-- The elementary cylindrical decomposition `d² = X² + ρ²` on the chart
support. -/
lemma shellDist_sq_eq_axial_add_rho {u : ℝ} (hu : u ^ 2 ≤ 1) (r s : ℝ) :
    shellDist r s u ^ 2 = shellAxial r s u ^ 2 + shellRhoSq s u := by
  rw [shellDist_sq_eq hu r s]
  unfold shellAxial shellRhoSq
  ring

lemma shellD_eq_shellT_sub_r_mul_shellZ (τ r s : ℝ) :
    shellD τ r s = shellT τ r s - r * shellZ τ r s := by
  have hIntT :
      IntegrableOn
        (fun u : ℝ => Real.exp (-(1 / τ) * shellDist r s u) * (s * u))
        (Ioc (-1 : ℝ) 1) := by
    have hc : Continuous
        (fun u : ℝ => Real.exp (-(1 / τ) * shellDist r s u) * (s * u)) := by
      unfold shellDist
      fun_prop
    exact hc.integrableOn_Icc.mono_set Ioc_subset_Icc_self
  have hIntZ :
      IntegrableOn
        (fun u : ℝ => Real.exp (-(1 / τ) * shellDist r s u))
        (Ioc (-1 : ℝ) 1) := by
    have hc : Continuous
        (fun u : ℝ => Real.exp (-(1 / τ) * shellDist r s u)) := by
      unfold shellDist
      fun_prop
    exact hc.integrableOn_Icc.mono_set Ioc_subset_Icc_self
  unfold shellD shellT shellZ
  calc
    (1 / 2) *
        ∫ u in Ioc (-1 : ℝ) 1,
          Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r)
        =
      (1 / 2) *
        ∫ u in Ioc (-1 : ℝ) 1,
          (Real.exp (-(1 / τ) * shellDist r s u) * (s * u)
            - r * Real.exp (-(1 / τ) * shellDist r s u)) := by
          congr 1
          apply setIntegral_congr_fun measurableSet_Ioc
          intro u _
          ring
    _ =
      (1 / 2) *
        ((∫ u in Ioc (-1 : ℝ) 1,
          Real.exp (-(1 / τ) * shellDist r s u) * (s * u))
          - ∫ u in Ioc (-1 : ℝ) 1,
            r * Real.exp (-(1 / τ) * shellDist r s u)) := by
          rw [integral_sub hIntT (hIntZ.const_mul r)]
    _ =
      (1 / 2) *
        ((∫ u in Ioc (-1 : ℝ) 1,
          Real.exp (-(1 / τ) * shellDist r s u) * (s * u))
          - r * ∫ u in Ioc (-1 : ℝ) 1,
            Real.exp (-(1 / τ) * shellDist r s u)) := by
          rw [integral_const_mul]
    _ =
      (1 / 2) *
        (∫ u in Ioc (-1 : ℝ) 1,
          Real.exp (-(1 / τ) * shellDist r s u) * (s * u))
        - r *
          ((1 / 2) * ∫ u in Ioc (-1 : ℝ) 1,
            Real.exp (-(1 / τ) * shellDist r s u)) := by ring

lemma continuous_laplaceWeightedDisplacement_rayProbe_coord (τ r : ℝ) :
    Continuous fun y : EuclideanSpace ℝ (Fin 3) =>
      (laplaceWeightedDisplacement τ (rayProbe r) y) 0 := by
  unfold laplaceWeightedDisplacement laplaceKernel
  fun_prop

/-- Public local replacement for the private bound in `LaplaceCompanion.lean`:
the first coordinate of the weighted displacement is uniformly bounded by
`τ`. -/
lemma abs_laplaceWeightedDisplacement_rayProbe_coord_le
    (τ : ℝ) (hτ : 0 < τ) (r : ℝ) (y : EuclideanSpace ℝ (Fin 3)) :
    |(laplaceWeightedDisplacement τ (rayProbe r) y) 0| ≤ τ := by
  have hcoord :
      |(laplaceWeightedDisplacement τ (rayProbe r) y) 0| ≤
        ‖laplaceWeightedDisplacement τ (rayProbe r) y‖ := by
    simpa [Real.norm_eq_abs] using
      (PiLp.norm_apply_le (laplaceWeightedDisplacement τ (rayProbe r) y) (0 : Fin 3))
  refine hcoord.trans ?_
  unfold laplaceWeightedDisplacement laplaceKernel
  rw [norm_smul, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
  rw [norm_sub_rev y (rayProbe r)]
  set d := ‖rayProbe r - y‖
  have hmul : d * Real.exp (-d / τ) ≤ τ * Real.exp (-1) :=
    mul_exp_neg_div_le hτ (norm_nonneg _)
  have he : Real.exp (-1 : ℝ) ≤ 1 := by
    rw [Real.exp_le_one_iff]
    norm_num
  have hτe : τ * Real.exp (-1 : ℝ) ≤ τ := by
    nlinarith [mul_le_mul_of_nonneg_left he hτ.le]
  have harg : -(1 / τ) * d = -d / τ := by ring
  rw [harg]
  calc
    Real.exp (-d / τ) * d = d * Real.exp (-d / τ) := by ring
    _ ≤ τ * Real.exp (-1) := hmul
    _ ≤ τ := hτe

lemma laplaceWeightedDisplacement_rayProbe_chart_coord {u : ℝ} (hu : u ^ 2 ≤ 1)
    (τ r s φ : ℝ) :
    (laplaceWeightedDisplacement τ (rayProbe r) (s • sphereChart u φ)) 0 =
      Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r) := by
  unfold laplaceWeightedDisplacement
  rw [PiLp.smul_apply, PiLp.sub_apply, laplaceKernel_rayProbe_chart hu τ r s φ]
  simp only [PiLp.smul_apply, sphereChart_apply_zero, rayProbe_apply_zero, smul_eq_mul]

/-- **The first coordinate of the drift numerator is a `ν`-mixture of per-shell
zonal displacement averages.** -/
lemma laplaceWeightedDisplacement_coord_radialMixture₃ (τ : ℝ) (hτ : 0 < τ)
    (ν : Measure ℝ) [IsProbabilityMeasure ν] (r : ℝ) :
    (∫ y, laplaceWeightedDisplacement τ (rayProbe r) y ∂(radialMixture₃ ν)) 0
      = ∫ s, shellD τ r s ∂ν := by
  let F : EuclideanSpace ℝ (Fin 3) → EuclideanSpace ℝ (Fin 3) :=
    fun y => laplaceWeightedDisplacement τ (rayProbe r) y
  have hFint : Integrable F (radialMixture₃ ν) :=
    laplaceWeightedDisplacement_integrable τ hτ (radialMixture₃ ν) (rayProbe r)
  have hcoord :
      (∫ y, F y ∂(radialMixture₃ ν)) 0 =
        ∫ y, (F y) 0 ∂(radialMixture₃ ν) := by
    have hproj :=
      (EuclideanSpace.proj (0 : Fin 3)).integral_comp_comm (μ := radialMixture₃ ν) hFint
    simpa [F, EuclideanSpace.coe_proj] using hproj.symm
  change (∫ y, F y ∂(radialMixture₃ ν)) 0 = ∫ s, shellD τ r s ∂ν
  rw [hcoord]
  have hf : Integrable (fun y : EuclideanSpace ℝ (Fin 3) => (F y) 0) (radialMixture₃ ν) :=
    ⟨(continuous_laplaceWeightedDisplacement_rayProbe_coord τ r).aestronglyMeasurable,
      HasFiniteIntegral.of_bounded (C := τ)
        (ae_of_all _ (fun y => by
          rw [Real.norm_eq_abs]
          exact abs_laplaceWeightedDisplacement_rayProbe_coord_le τ hτ r y))⟩
  rw [integral_radialMixture₃ ν hf]
  refine integral_congr_ae (Filter.Eventually.of_forall (fun s => ?_))
  change (∫ w : ℝ × ℝ, (F (chartMap (s, w))) 0 ∂chartBase) = shellD τ r s
  unfold shellD
  refine integral_chartBase_zonal
    (G := fun w : ℝ × ℝ => (F (chartMap (s, w))) 0)
    (g := fun u => Real.exp (-(1 / τ) * shellDist r s u) * (s * u - r))
    (C := τ) ?_ ?_ ?_
  · exact (continuous_laplaceWeightedDisplacement_rayProbe_coord τ r).comp
      (continuous_chartMap.comp (by fun_prop))
  · intro w
    exact abs_laplaceWeightedDisplacement_rayProbe_coord_le τ hτ r _
  · intro u φ hu
    have hu2 : u ^ 2 ≤ 1 := by nlinarith [hu.1, hu.2]
    rw [chartMap_mk_pair]
    exact laplaceWeightedDisplacement_rayProbe_chart_coord hu2 τ r s φ

end DriftingIdentifiability
