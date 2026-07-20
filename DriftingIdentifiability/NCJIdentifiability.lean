import DriftingIdentifiability.Algorithm2SNIS
import DriftingIdentifiability.FiniteStability
import DriftingIdentifiability.LaplaceEuclideanConverse
import DriftingIdentifiability.GaussianConvolutionInjectivity

/-!
# NCJ drifting: the certified identifiability layer (T1, T2)

Formal obligations T1 and T2 of `numerics/IdentifiabilityDrivenImprovementPlan.md`
for the normalized, cross-fitted, jittered (NCJ) drifting candidate.  No new
axiom is introduced; every declaration composes existing certified results.

* **T1 — positive-gain zero-set preservation.**  The audited factorization
  `algorithm2Drift i = (P_i · Q_i) • Δ_i` (`Algorithm2Estimator.lean`) shows
  the paper's affinity-mass product is a strictly positive per-query gain.
  Replacing it by ANY strictly positive gain `g i` (NCJ version 1 uses
  `g = 1`) preserves the pointwise zero set of the unmasked estimator:
  `positiveGainField g … i = 0 ↔ algorithm2Drift … i = 0`, hence the
  all-anchor zero sets agree, and any two positive gains have identical zero
  sets.  For the quantitative frame transfer the plan's
  `interactionFrameBound_of_positiveGain` specializes the certified
  strict-pair scaling lemma (`FiniteStability.lean`) to gains bounded below.
  Nothing here assumes injectivity or the desired identifiability conclusion.

* **T2 — jittered identifiability composition.**  Symmetric Gaussian jitter
  compares the smoothed laws `p ∗ N(0, σ²I)` and `q ∗ N(0, σ²I)` at the
  population level.  Composing the general Euclidean Laplace converse
  (`LaplaceEuclideanConverse.lean`) applied to the smoothed laws with the
  certified Gaussian-convolution injectivity
  (`GaussianConvolutionInjectivity.lean`) yields: zero Laplace drift between
  the jittered laws already identifies the ORIGINAL laws, `p = q`.  The
  bandwidth hypothesis is `0 < τ`; the jitter scale `σ` is arbitrary
  (at `σ = 0` the smoothing is a Dirac mass and the statement degenerates to
  the unjittered converse), and `p, q` are arbitrary Borel probability
  measures on any finite-dimensional Euclidean space.
-/

open scoped BigOperators
open MeasureTheory ProbabilityTheory

namespace DriftingIdentifiability

universe u

/-! ## T1 frame transfer: positive gains preserve the interaction frame bound -/

namespace PaperFiniteIdentifiability

open Paper

variable {V : Type u} [NormedAddCommGroup V] [NormedSpace ℝ V] {m : ℕ}

/-- **Positive-gain frame transfer** (the plan's named finite frame statement).
If the bare interaction vectors satisfy a frame bound `c > 0` and the modified
vectors are per-pair positive rescalings with gains bounded below by
`gmin > 0`, the modified interaction satisfies the explicit frame bound
`gmin · c`.  Specializes `interactionFrameBound_of_strictPairScaling`. -/
theorem interactionFrameBound_of_uniformPositiveGain
    (U U' : Fin m → Fin m → V) (g : StrictPair m → ℝ) {c gmin : ℝ}
    (hframe : InteractionFrameBound U c) (hgmin : 0 < gmin)
    (hg : ∀ p : StrictPair m, gmin ≤ g p)
    (hU' : ∀ p : StrictPair m, U' p.1.1 p.1.2 = g p • U p.1.1 p.1.2) :
    InteractionFrameBound U' (gmin * c) :=
  interactionFrameBound_of_strictPairScaling U U' g hframe hgmin
    (fun p => (hg p).trans (le_abs_self _)) hU'

end PaperFiniteIdentifiability

/-! ## T1: positive-gain zero-set preservation for Algorithm 2 -/

namespace Algorithm2

open Paper

variable {E : Type u} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable {Nx Npos Nneg : ℕ}

/-- The normalized centroid-difference field `Δ_i = Cpos_i - Cneg_i` of the
unmasked Algorithm-2 estimator: the distribution-matching signal left after
removing the affinity-mass gain.  This is the population target of NCJ's
constant-gain field. -/
noncomputable def normalizedCentroidField
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) : E :=
  algorithm2PositiveCentroid x yPos yNeg temperature (fun _ _ => false) i -
    algorithm2NegativeCentroid x yPos yNeg temperature (fun _ _ => false) i

/-- The positive-gain drifting field `g_i • Δ_i`.  NCJ version 1 is the
constant gain `g = 1`; the paper's own field corresponds to the (query-
dependent) gain `P_i · Q_i`. -/
noncomputable def positiveGainField
    (g : Fin Nx → ℝ) (x : Fin Nx → E) (yPos : Fin Npos → E)
    (yNeg : Fin Nneg → E) (temperature : ℝ) (i : Fin Nx) : E :=
  g i • normalizedCentroidField x yPos yNeg temperature i

/-- **T1, pointwise form.**  For any strictly positive gain, the gain field
vanishes at a query exactly when the exact unmasked Algorithm-2 drift does.
Composes the certified factorization with mass positivity; assumes only
nonempty sample sides and `0 < g i`. -/
theorem positiveGainField_eq_zero_iff_algorithm2Drift_eq_zero
    [Nonempty (Fin Npos)] [Nonempty (Fin Nneg)]
    (g : Fin Nx → ℝ) (hg : ∀ i, 0 < g i)
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) :
    positiveGainField g x yPos yNeg temperature i = 0 ↔
      algorithm2Drift x yPos yNeg temperature (fun _ _ => false) i = 0 := by
  rw [algorithm2Drift_false_eq_zero_iff_centroidDiff_eq_zero]
  unfold positiveGainField normalizedCentroidField
  constructor
  · intro h
    exact (smul_eq_zero.mp h).resolve_left (ne_of_gt (hg i))
  · intro h
    rw [h, smul_zero]

/-- **T1, zero-set form.**  A strictly positive gain preserves the zero set of
the exact unmasked Algorithm-2 drift over all queries simultaneously. -/
theorem positiveGain_zeroSet_preservation
    [Nonempty (Fin Npos)] [Nonempty (Fin Nneg)]
    (g : Fin Nx → ℝ) (hg : ∀ i, 0 < g i)
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) :
    (∀ i, positiveGainField g x yPos yNeg temperature i = 0) ↔
      (∀ i, algorithm2Drift x yPos yNeg temperature (fun _ _ => false) i = 0) :=
  forall_congr' fun i =>
    positiveGainField_eq_zero_iff_algorithm2Drift_eq_zero g hg x yPos yNeg
      temperature i

/-- **T1, gain-invariance form.**  Any two strictly positive gains — in
particular NCJ's constant gain `1` and the paper's mass-product gain — induce
fields with identical zero sets. -/
theorem positiveGainField_zeroSet_gain_invariant
    [Nonempty (Fin Npos)] [Nonempty (Fin Nneg)]
    (g₁ g₂ : Fin Nx → ℝ) (hg₁ : ∀ i, 0 < g₁ i) (hg₂ : ∀ i, 0 < g₂ i)
    (x : Fin Nx → E) (yPos : Fin Npos → E) (yNeg : Fin Nneg → E)
    (temperature : ℝ) (i : Fin Nx) :
    positiveGainField g₁ x yPos yNeg temperature i = 0 ↔
      positiveGainField g₂ x yPos yNeg temperature i = 0 :=
  (positiveGainField_eq_zero_iff_algorithm2Drift_eq_zero g₁ hg₁ x yPos yNeg
      temperature i).trans
    (positiveGainField_eq_zero_iff_algorithm2Drift_eq_zero g₂ hg₂ x yPos yNeg
      temperature i).symm

end Algorithm2

/-! ## T2: jittered identifiability composition -/

open Paper

/-- **T2: the jittered Laplace converse.**  If the ℓ²-Laplace mean-shift drift
between the symmetrically jittered laws `p ∗ N(0, σ²I)` and `q ∗ N(0, σ²I)`
vanishes, then the original laws are equal: the smoothed laws are identified
by the general Euclidean Laplace converse, and Gaussian convolution is
injective.  `p, q` are arbitrary Borel probability measures on a
finite-dimensional Euclidean space; `σ` is unrestricted (σ = 0 degenerates to
the unjittered converse). -/
theorem laplaceZeroDrift_jittered_identifies_euclidean
    {ι : Type*} [Fintype ι] (τ σ : ℝ) (hτ : 0 < τ)
    (p q : Measure (EuclideanSpace ℝ ι))
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (p ∗ scaledStdGaussian σ) (q ∗ scaledStdGaussian σ)) :
    p = q :=
  Measure.eq_of_conv_scaledStdGaussian_eq σ p q
    (laplaceZeroDrift_identifies_euclidean τ hτ
      (p ∗ scaledStdGaussian σ) (q ∗ scaledStdGaussian σ) hzero)

/-- **T2 on `ℝⁿ`.**  The roadmap-form statement for every finite `n`. -/
theorem laplaceZeroDrift_jittered_identifies_rn
    (τ σ : ℝ) (hτ : 0 < τ) (n : ℕ)
    (p q : Measure (EuclideanSpace ℝ (Fin n)))
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ))
      (p ∗ scaledStdGaussian σ) (q ∗ scaledStdGaussian σ)) :
    p = q :=
  laplaceZeroDrift_jittered_identifies_euclidean τ σ hτ p q hzero

/-! ## T3: cross-fitted estimator consistency -/

namespace Algorithm2

section CentroidRadius

variable {F : Type u} [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- A self-normalized centroid with a positive weight floor is a convex
combination of the samples, so it stays within the sample radius of the
target. -/
theorem selfNormalizedCentroid_dist_le
    {N : ℕ} [Nonempty (Fin N)] (w : F → ℝ) (Z : Fin N → F) (c : F)
    {wmin R : ℝ} (hwmin : 0 < wmin)
    (hlb : ∀ j, wmin ≤ w (Z j)) (hbd : ∀ j, ‖Z j - c‖ ≤ R) :
    ‖(∑ j, w (Z j))⁻¹ • (∑ j, w (Z j) • Z j) - c‖ ≤ R := by
  have hpos : ∀ j, 0 < w (Z j) := fun j => lt_of_lt_of_le hwmin (hlb j)
  have hD : 0 < ∑ j, w (Z j) :=
    Finset.sum_pos (fun j _ => hpos j) Finset.univ_nonempty
  have hexp : (∑ j, w (Z j) • (Z j - c))
      = (∑ j, w (Z j) • Z j) - (∑ j, w (Z j)) • c := by
    simp_rw [smul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_smul]
  have hsplit : (∑ j, w (Z j))⁻¹ • (∑ j, w (Z j) • Z j) - c
      = (∑ j, w (Z j))⁻¹ • (∑ j, w (Z j) • (Z j - c)) := by
    rw [hexp, smul_sub, smul_smul, inv_mul_cancel₀ hD.ne', one_smul]
  rw [hsplit, norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.mpr hD)]
  have hnorm : ‖∑ j, w (Z j) • (Z j - c)‖ ≤ (∑ j, w (Z j)) * R := by
    calc ‖∑ j, w (Z j) • (Z j - c)‖
        ≤ ∑ j, ‖w (Z j) • (Z j - c)‖ := norm_sum_le _ _
      _ ≤ ∑ j, w (Z j) * R := by
          apply Finset.sum_le_sum
          intro j _
          rw [norm_smul, Real.norm_eq_abs, abs_of_pos (hpos j)]
          exact mul_le_mul_of_nonneg_left (hbd j) (hpos j).le
      _ = (∑ j, w (Z j)) * R := by rw [← Finset.sum_mul]
  calc (∑ j, w (Z j))⁻¹ * ‖∑ j, w (Z j) • (Z j - c)‖
      ≤ (∑ j, w (Z j))⁻¹ * ((∑ j, w (Z j)) * R) :=
        mul_le_mul_of_nonneg_left hnorm (inv_pos.mpr hD).le
    _ = R := by rw [← mul_assoc, inv_mul_cancel₀ hD.ne', one_mul]

end CentroidRadius

section CrossFitted

open Paper

variable {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
variable {F : Type u} [NormedAddCommGroup F] [InnerProductSpace ℝ F]
  [MeasurableSpace F] [BorelSpace F] [CompleteSpace F]
  [SecondCountableTopology F]
variable {Nx Npos Nneg : ℕ} [Nonempty (Fin Nx)]

/-- **T3: cross-fitted centroid-difference consistency.**  With fixed anchors
(the cross-fitted setting: the query batch is drawn independently of both
sample batches, so one conditions on it), no eye mask, within-batch
independence, a weight floor/ceiling `0 < wmin ≤ w ≤ wmax` for the induced
column-reweighted kernel, and sample-radius bounds, the centroid-difference
estimator `Δ̂_i = Ĉpos_i − Ĉneg_i` — by T1 exactly the constant-gain NCJ
field — has mean-squared error about its population target `cpos − cneg` at
most `2σp²/(wmin²·Npos) + 2σn²/(wmin²·Nneg)`.

The targets are characterized by `E[w·(Y − c)] = 0`, i.e. the self-normalized
population ratios `c = E[wY]/E[w]`.  Independence *between* the positive and
negative batches is not needed for this bound: after row-scale cancellation
each centroid ignores the other batch.  No exact-unbiasedness claim is made —
finite-sample ratio bias is absorbed into the definition of `c`.  Inherits
only the allowlisted sample-mean variance axiom through
`selfNormalized_meanSquare_le`. -/
theorem crossFitted_centroidDiff_meanSquare_le
    (hNpos : 0 < Npos) (hNneg : 0 < Nneg)
    (anchors : Fin Nx → F) (temperature : ℝ) (i : Fin Nx)
    (Ypos : Fin Npos → Ω → F) (Yneg : Fin Nneg → Ω → F) (cpos cneg : F)
    {wmin wmax Rp Rn σp σn : ℝ} (hwmin : 0 < wmin) (hwmax : 0 ≤ wmax)
    (hRp : 0 ≤ Rp) (hRn : 0 ≤ Rn)
    (hYposmeas : ∀ j, Measurable (Ypos j))
    (hYnegmeas : ∀ j, Measurable (Yneg j))
    (hw : Measurable (algorithm2ColumnReweightedWeight anchors temperature i))
    (hposindep : ∀ j k, j ≠ k → IndepFun (Ypos j) (Ypos k) P)
    (hnegindep : ∀ j k, j ≠ k → IndepFun (Yneg j) (Yneg k) P)
    (hposwlb : ∀ j ω, wmin ≤
      algorithm2ColumnReweightedWeight anchors temperature i (Ypos j ω))
    (hposwub : ∀ j ω,
      algorithm2ColumnReweightedWeight anchors temperature i (Ypos j ω) ≤ wmax)
    (hnegwlb : ∀ j ω, wmin ≤
      algorithm2ColumnReweightedWeight anchors temperature i (Yneg j ω))
    (hnegwub : ∀ j ω,
      algorithm2ColumnReweightedWeight anchors temperature i (Yneg j ω) ≤ wmax)
    (hposbd : ∀ j ω, ‖Ypos j ω - cpos‖ ≤ Rp)
    (hnegbd : ∀ j ω, ‖Yneg j ω - cneg‖ ≤ Rn)
    (hposmean : ∀ j,
      ∫ ω, algorithm2ColumnReweightedWeight anchors temperature i (Ypos j ω) •
        (Ypos j ω - cpos) ∂P = 0)
    (hnegmean : ∀ j,
      ∫ ω, algorithm2ColumnReweightedWeight anchors temperature i (Yneg j ω) •
        (Yneg j ω - cneg) ∂P = 0)
    (hposσ : ∀ j,
      ∫ ω, ‖algorithm2ColumnReweightedWeight anchors temperature i (Ypos j ω) •
        (Ypos j ω - cpos)‖ ^ 2 ∂P ≤ σp ^ 2)
    (hnegσ : ∀ j,
      ∫ ω, ‖algorithm2ColumnReweightedWeight anchors temperature i (Yneg j ω) •
        (Yneg j ω - cneg)‖ ^ 2 ∂P ≤ σn ^ 2) :
    ∫ ω,
        ‖(algorithm2PositiveCentroid anchors (fun j => Ypos j ω)
            (fun l => Yneg l ω) temperature (fun _ _ => false) i -
          algorithm2NegativeCentroid anchors (fun j => Ypos j ω)
            (fun l => Yneg l ω) temperature (fun _ _ => false) i) -
          (cpos - cneg)‖ ^ 2 ∂P
      ≤ 2 * (σp ^ 2 / (wmin ^ 2 * Npos)) +
          2 * (σn ^ 2 / (wmin ^ 2 * Nneg)) := by
  haveI : Nonempty (Fin Npos) := ⟨⟨0, hNpos⟩⟩
  haveI : Nonempty (Fin Nneg) := ⟨⟨0, hNneg⟩⟩
  -- measurability of the two realized centroids
  have hXm : Measurable (fun ω => algorithm2PositiveCentroid anchors
      (fun j => Ypos j ω) (fun l => Yneg l ω) temperature
      (fun _ _ => false) i) := by
    have hrep : (fun ω => algorithm2PositiveCentroid anchors
        (fun j => Ypos j ω) (fun l => Yneg l ω) temperature
        (fun _ _ => false) i)
        = fun ω => (∑ j : Fin Npos, algorithm2ColumnReweightedWeight anchors
            temperature i (Ypos j ω))⁻¹ •
          (∑ j : Fin Npos, algorithm2ColumnReweightedWeight anchors
            temperature i (Ypos j ω) • Ypos j ω) := by
      funext ω
      exact algorithm2PositiveCentroid_false_eq_columnReweighted anchors
        (fun j => Ypos j ω) (fun l => Yneg l ω) temperature i
    rw [hrep]
    exact (Finset.measurable_sum Finset.univ fun j _ =>
        hw.comp (hYposmeas j)).inv.smul
      (Finset.measurable_sum Finset.univ fun j _ =>
        (hw.comp (hYposmeas j)).smul (hYposmeas j))
  have hYm : Measurable (fun ω => algorithm2NegativeCentroid anchors
      (fun j => Ypos j ω) (fun l => Yneg l ω) temperature
      (fun _ _ => false) i) := by
    have hrep : (fun ω => algorithm2NegativeCentroid anchors
        (fun j => Ypos j ω) (fun l => Yneg l ω) temperature
        (fun _ _ => false) i)
        = fun ω => (∑ j : Fin Nneg, algorithm2ColumnReweightedWeight anchors
            temperature i (Yneg j ω))⁻¹ •
          (∑ j : Fin Nneg, algorithm2ColumnReweightedWeight anchors
            temperature i (Yneg j ω) • Yneg j ω) := by
      funext ω
      exact algorithm2NegativeCentroid_false_eq_columnReweighted anchors
        (fun j => Ypos j ω) (fun l => Yneg l ω) temperature i
    rw [hrep]
    exact (Finset.measurable_sum Finset.univ fun j _ =>
        hw.comp (hYnegmeas j)).inv.smul
      (Finset.measurable_sum Finset.univ fun j _ =>
        (hw.comp (hYnegmeas j)).smul (hYnegmeas j))
  -- pointwise radius bounds
  have hXbd : ∀ ω, ‖algorithm2PositiveCentroid anchors (fun j => Ypos j ω)
      (fun l => Yneg l ω) temperature (fun _ _ => false) i - cpos‖ ≤ Rp := by
    intro ω
    rw [algorithm2PositiveCentroid_false_eq_columnReweighted anchors
      (fun j => Ypos j ω) (fun l => Yneg l ω) temperature i]
    exact selfNormalizedCentroid_dist_le
      (algorithm2ColumnReweightedWeight anchors temperature i)
      (fun j => Ypos j ω) cpos hwmin
      (fun j => hposwlb j ω) (fun j => hposbd j ω)
  have hYbd : ∀ ω, ‖algorithm2NegativeCentroid anchors (fun j => Ypos j ω)
      (fun l => Yneg l ω) temperature (fun _ _ => false) i - cneg‖ ≤ Rn := by
    intro ω
    rw [algorithm2NegativeCentroid_false_eq_columnReweighted anchors
      (fun j => Ypos j ω) (fun l => Yneg l ω) temperature i]
    exact selfNormalizedCentroid_dist_le
      (algorithm2ColumnReweightedWeight anchors temperature i)
      (fun j => Yneg j ω) cneg hwmin
      (fun j => hnegwlb j ω) (fun j => hnegbd j ω)
  -- integrability from boundedness
  have hXint : Integrable (fun ω => ‖algorithm2PositiveCentroid anchors
      (fun j => Ypos j ω) (fun l => Yneg l ω) temperature
      (fun _ _ => false) i - cpos‖ ^ 2) P := by
    refine (integrable_const (Rp ^ 2)).mono'
      ((hXm.sub measurable_const).norm.pow_const 2).aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    nlinarith [hXbd ω, norm_nonneg (algorithm2PositiveCentroid anchors
      (fun j => Ypos j ω) (fun l => Yneg l ω) temperature
      (fun _ _ => false) i - cpos)]
  have hYint : Integrable (fun ω => ‖algorithm2NegativeCentroid anchors
      (fun j => Ypos j ω) (fun l => Yneg l ω) temperature
      (fun _ _ => false) i - cneg‖ ^ 2) P := by
    refine (integrable_const (Rn ^ 2)).mono'
      ((hYm.sub measurable_const).norm.pow_const 2).aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    nlinarith [hYbd ω, norm_nonneg (algorithm2NegativeCentroid anchors
      (fun j => Ypos j ω) (fun l => Yneg l ω) temperature
      (fun _ _ => false) i - cneg)]
  have hZbd : ∀ ω, ‖(algorithm2PositiveCentroid anchors (fun j => Ypos j ω)
      (fun l => Yneg l ω) temperature (fun _ _ => false) i -
      algorithm2NegativeCentroid anchors (fun j => Ypos j ω)
        (fun l => Yneg l ω) temperature (fun _ _ => false) i) -
      (cpos - cneg)‖ ≤ Rp + Rn := by
    intro ω
    have hrw : (algorithm2PositiveCentroid anchors (fun j => Ypos j ω)
        (fun l => Yneg l ω) temperature (fun _ _ => false) i -
        algorithm2NegativeCentroid anchors (fun j => Ypos j ω)
          (fun l => Yneg l ω) temperature (fun _ _ => false) i) -
        (cpos - cneg)
        = (algorithm2PositiveCentroid anchors (fun j => Ypos j ω)
            (fun l => Yneg l ω) temperature (fun _ _ => false) i - cpos) -
          (algorithm2NegativeCentroid anchors (fun j => Ypos j ω)
            (fun l => Yneg l ω) temperature (fun _ _ => false) i - cneg) := by
      abel
    rw [hrw]
    exact (norm_sub_le _ _).trans (add_le_add (hXbd ω) (hYbd ω))
  have hZint : Integrable (fun ω =>
      ‖(algorithm2PositiveCentroid anchors (fun j => Ypos j ω)
          (fun l => Yneg l ω) temperature (fun _ _ => false) i -
        algorithm2NegativeCentroid anchors (fun j => Ypos j ω)
          (fun l => Yneg l ω) temperature (fun _ _ => false) i) -
        (cpos - cneg)‖ ^ 2) P := by
    have hm : Measurable (fun ω =>
        ‖(algorithm2PositiveCentroid anchors (fun j => Ypos j ω)
            (fun l => Yneg l ω) temperature (fun _ _ => false) i -
          algorithm2NegativeCentroid anchors (fun j => Ypos j ω)
            (fun l => Yneg l ω) temperature (fun _ _ => false) i) -
          (cpos - cneg)‖ ^ 2) :=
      ((hXm.sub hYm).sub measurable_const).norm.pow_const 2
    refine (integrable_const ((Rp + Rn) ^ 2)).mono'
      hm.aestronglyMeasurable ?_
    filter_upwards with ω
    rw [Real.norm_eq_abs, abs_of_nonneg (sq_nonneg _)]
    nlinarith [hZbd ω, norm_nonneg ((algorithm2PositiveCentroid anchors
      (fun j => Ypos j ω) (fun l => Yneg l ω) temperature
      (fun _ _ => false) i -
      algorithm2NegativeCentroid anchors (fun j => Ypos j ω)
        (fun l => Yneg l ω) temperature (fun _ _ => false) i) -
      (cpos - cneg))]
  -- the two SNIS mean-square bounds and the deterministic glue
  have hXmse := algorithm2PositiveCentroid_false_meanSquare_le P hNpos anchors
    temperature i Ypos Yneg cpos hwmin hwmax hRp hYposmeas hw hposindep
    hposwlb hposwub hposbd hposmean hposσ
  have hYmse := algorithm2NegativeCentroid_false_meanSquare_le P hNneg anchors
    temperature i Ypos Yneg cneg hwmin hwmax hRn hYnegmeas hw hnegindep
    hnegwlb hnegwub hnegbd hnegmean hnegσ
  exact meanSquare_sub_sub_le_two_add
    (fun ω => algorithm2PositiveCentroid anchors (fun j => Ypos j ω)
      (fun l => Yneg l ω) temperature (fun _ _ => false) i)
    (fun ω => algorithm2NegativeCentroid anchors (fun j => Ypos j ω)
      (fun l => Yneg l ω) temperature (fun _ _ => false) i)
    cpos cneg hXint hYint hZint hXmse hYmse

end CrossFitted

end Algorithm2

end DriftingIdentifiability
