import DriftingIdentifiability.LaplaceAtomAlignment

/-!
# Higher-dimensional Laplace converse, milestone L3: ball-average cone extraction

This file develops the analytic content that discharges the
`LaplaceAtomConeProductData` hypothesis of `LaplaceAtomAlignment.lean`, following
the `w(ε)`-normalizer route recorded in `LaplaceHigherDim.md` §4.8 (L3).

Foundational geometric facts first: the ball is symmetric about its centre, so
the centred coordinate `x - a` integrates to zero over `B(a, ε)`, and hence the
ball-average defect of a function differentiable at `a` is `o(ε)`.
-/

open MeasureTheory Filter Topology Metric
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [MeasureSpace E] [BorelSpace E] [(volume : Measure E).IsAddHaarMeasure]
  [(volume : Measure E).IsNegInvariant]

/-! ## Reflection symmetry of the ball -/

set_option linter.unusedSectionVars false in
/-- The reflection `x ↦ 2a - x` maps `ball a ε` onto itself. -/
lemma image_reflection_ball (a : E) (ε : ℝ) :
    (fun x : E => 2 • a - x) '' Metric.ball a ε = Metric.ball a ε := by
  ext y
  simp only [Set.mem_image, Metric.mem_ball]
  constructor
  · rintro ⟨x, hx, rfl⟩
    rw [dist_eq_norm]
    have : 2 • a - x - a = -(x - a) := by rw [two_smul]; abel
    rw [this, norm_neg, ← dist_eq_norm]
    exact hx
  · intro hy
    refine ⟨2 • a - y, ?_, by abel⟩
    rw [dist_eq_norm]
    have : 2 • a - y - a = -(y - a) := by rw [two_smul]; abel
    rw [this, norm_neg, ← dist_eq_norm]
    exact hy

set_option linter.unusedSectionVars false in
/-- The reflection `x ↦ 2a - x` is measure-preserving for the volume. -/
lemma measurePreserving_reflection (a : E) :
    MeasurePreserving (fun x : E => 2 • a - x) (volume : Measure E) volume := by
  have h1 : MeasurePreserving (fun x : E => -x) (volume : Measure E) volume :=
    Measure.measurePreserving_neg volume
  have h2 : MeasurePreserving (fun x : E => 2 • a + x) (volume : Measure E) volume :=
    measurePreserving_add_left volume (2 • a)
  have heq : (fun x : E => 2 • a - x) = (fun x : E => 2 • a + x) ∘ (fun x : E => -x) := by
    funext x; simp [sub_eq_add_neg]
  rw [heq]
  exact h2.comp h1

set_option linter.unusedSectionVars false in
/-- The reflection is a measurable embedding (it is an involutive homeomorphism). -/
lemma measurableEmbedding_reflection (a : E) :
    MeasurableEmbedding (fun x : E => 2 • a - x) := by
  have heq : (fun x : E => 2 • a - x) =
      ((Homeomorph.neg E).trans (Homeomorph.addLeft (2 • a))) := by
    funext x
    simp [sub_eq_add_neg]
  rw [heq]
  exact ((Homeomorph.neg E).trans (Homeomorph.addLeft (2 • a))).measurableEmbedding

set_option linter.unusedSectionVars false in
/-- **Reflection symmetry.**  The centred coordinate integrates to zero over a
ball: `∫_{B(a,ε)} (x - a) = 0`.  This is the vanishing of the first moment of the
(symmetric) ball, and it makes the linear part of any Taylor expansion average
out. -/
lemma setIntegral_sub_center_ball_eq_zero (a : E) (ε : ℝ) :
    ∫ x in Metric.ball a ε, (x - a) ∂(volume : Measure E) = 0 := by
  have hkey :
      ∫ x in Metric.ball a ε, (x - a) ∂(volume : Measure E) =
        ∫ x in Metric.ball a ε, ((2 • a - x) - a) ∂(volume : Measure E) := by
    conv_lhs => rw [← image_reflection_ball a ε]
    rw [(measurePreserving_reflection a).setIntegral_image_emb
      (measurableEmbedding_reflection a) (fun x => x - a) (Metric.ball a ε)]
  have hneg : ∫ x in Metric.ball a ε, ((2 • a - x) - a) ∂(volume : Measure E) =
      -∫ x in Metric.ball a ε, (x - a) ∂(volume : Measure E) := by
    rw [← integral_neg]
    congr 1
    funext x
    rw [two_smul]; abel
  rw [hneg] at hkey
  have hII : (2 : ℝ) • (∫ x in Metric.ball a ε, (x - a) ∂(volume : Measure E)) = 0 := by
    rw [two_smul]
    exact add_eq_zero_iff_eq_neg.mpr hkey
  exact (smul_eq_zero.mp hII).resolve_left (by norm_num)

set_option linter.unusedSectionVars false in
/-- **Ball average of the centred coordinate vanishes**: `⨍_{B(a,ε)} (x - a) = 0`.
This is the form used in the cone-extraction argument — the linear part of a
Taylor expansion averages out over the (symmetric) ball. -/
lemma setAverage_sub_center_ball_eq_zero (a : E) (ε : ℝ) :
    ⨍ x in Metric.ball a ε, (x - a) ∂(volume : Measure E) = 0 := by
  rw [setAverage_eq, setIntegral_sub_center_ball_eq_zero, smul_zero]

end DriftingIdentifiability
