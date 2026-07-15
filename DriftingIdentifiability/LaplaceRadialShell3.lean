import DriftingIdentifiability.LaplaceRadialFoundations

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

end DriftingIdentifiability
