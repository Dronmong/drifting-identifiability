import DriftingIdentifiability.TrustedBoundary

/-!
# Objective 5: feature-space conclusions

The population theorem is a data-space theorem.  When training is performed
after a feature map `φ : X → F`, the safe conclusion is equality of feature
laws `φ♯p = φ♯q`.  Equality of the original source laws requires an additional,
independently checkable condition on the feature map.

This module formalizes that boundary:

* a `FeatureModel` is a measurable feature map;
* feature-space identifiability theorems lift automatically to equality of
  feature laws;
* equality of source laws is recovered only when the feature is a
  `MeasurableEmbedding`;
* for multiple features, one embedded feature is enough to lift equality of all
  feature laws to source-law equality;
* heterogeneous feature families may use different feature spaces per index;
* approximate lifting must pass through a `FeatureStabilityCertificate`, which
  controls a stated source discrepancy by stated feature-law discrepancies;
* a non-injective feature collision gives a formal Dirac counterexample.

No theorem here assumes the desired drift identifiability statement.  The only
lifting condition used for source measures is the ordinary Mathlib theorem that
`Measure.map` is injective along a measurable embedding.
-/

open MeasureTheory

namespace DriftingIdentifiability

open Paper

universe u v

/-- A practical feature model: a measurable map from source/data space `X` into
a feature space `F`. -/
structure FeatureModel (X : Type u) (F : Type v)
    [MeasurableSpace X] [MeasurableSpace F] where
  toFun : X → F
  measurable_toFun : Measurable toFun

namespace FeatureModel

variable {X : Type u} {F : Type v}
variable [MeasurableSpace X] [MeasurableSpace F]

instance : CoeFun (FeatureModel X F) (fun _ => X → F) where
  coe Φ := Φ.toFun

/-- The law seen by a feature model.  This is the paper's `featureLaw`. -/
noncomputable def law (Φ : FeatureModel X F) (p : Distribution X) :
    Distribution F :=
  featureLaw Φ.toFun p Φ.measurable_toFun.aemeasurable

theorem law_eq_map (Φ : FeatureModel X F) (p : Distribution X) :
    Φ.law p = Measure.map Φ.toFun p := by
  rfl

/-- Source equality always implies feature-law equality. -/
theorem law_eq_of_source_eq (Φ : FeatureModel X F) {p q : Distribution X}
    (h : p = q) : Φ.law p = Φ.law q := by
  subst q
  rfl

/-- The safe output of a feature-space identifiability theorem is equality of
the feature laws, not automatically equality of the source laws. -/
theorem law_eq_of_feature_identifiesAtZero [Zero F]
    (Φ : FeatureModel X F)
    (condition : Distribution F → Distribution F → Prop)
    (V : DriftingField F)
    (hidentify : IdentifiesAtZero condition V)
    (p q : Distribution X)
    (hcondition : condition (Φ.law p) (Φ.law q))
    (hzero : ZeroDrift V (Φ.law p) (Φ.law q)) :
    Φ.law p = Φ.law q :=
  hidentify (Φ.law p) (Φ.law q) hcondition hzero

/-- Source-law equality can be recovered from feature-law equality under an
independently checkable measurable-embedding hypothesis. -/
theorem source_eq_of_law_eq (Φ : FeatureModel X F)
    (hemb : MeasurableEmbedding Φ.toFun) {p q : Distribution X}
    (h : Φ.law p = Φ.law q) : p = q := by
  apply hemb.map_injective
  simpa [law, featureLaw, pushforward] using h

/-- A feature-space zero-drift theorem lifts all the way back to source-law
equality only when the feature map is a measurable embedding. -/
theorem source_eq_of_feature_identifiesAtZero [Zero F]
    (Φ : FeatureModel X F) (hemb : MeasurableEmbedding Φ.toFun)
    (condition : Distribution F → Distribution F → Prop)
    (V : DriftingField F)
    (hidentify : IdentifiesAtZero condition V)
    (p q : Distribution X)
    (hcondition : condition (Φ.law p) (Φ.law q))
    (hzero : ZeroDrift V (Φ.law p) (Φ.law q)) :
    p = q :=
  Φ.source_eq_of_law_eq hemb
    (Φ.law_eq_of_feature_identifiesAtZero condition V hidentify p q hcondition hzero)

end FeatureModel

/-! ## Multiple feature maps -/

section MultiFeature

variable {X : Type u} {F : Type v}
variable [MeasurableSpace X] [MeasurableSpace F]
variable {n : ℕ}

/-- Equality of all feature laws in a finite feature family. -/
def AllFeatureLawsEqual (Φ : Fin n → FeatureModel X F)
    (p q : Distribution X) : Prop :=
  ∀ j : Fin n, (Φ j).law p = (Φ j).law q

theorem allFeatureLawsEqual_of_source_eq (Φ : Fin n → FeatureModel X F)
    {p q : Distribution X} (h : p = q) : AllFeatureLawsEqual Φ p q := by
  intro j
  exact (Φ j).law_eq_of_source_eq h

/-- If all feature laws agree and one selected feature is a measurable
embedding, then the source laws agree. -/
theorem source_eq_of_allFeatureLawsEqual_of_embedding
    (Φ : Fin n → FeatureModel X F) {p q : Distribution X}
    (hfeatures : AllFeatureLawsEqual Φ p q)
    (j : Fin n) (hemb : MeasurableEmbedding (Φ j).toFun) :
    p = q :=
  (Φ j).source_eq_of_law_eq hemb (hfeatures j)

/-- Applying a feature-space identifiability theorem independently to every
feature gives equality of every feature law. -/
theorem allFeatureLawsEqual_of_each_feature_identifiesAtZero [Zero F]
    (Φ : Fin n → FeatureModel X F)
    (condition : Fin n → Distribution F → Distribution F → Prop)
    (V : Fin n → DriftingField F)
    (hidentify : ∀ j, IdentifiesAtZero (condition j) (V j))
    (p q : Distribution X)
    (hcondition : ∀ j, condition j ((Φ j).law p) ((Φ j).law q))
    (hzero : ∀ j, ZeroDrift (V j) ((Φ j).law p) ((Φ j).law q)) :
    AllFeatureLawsEqual Φ p q := by
  intro j
  exact (Φ j).law_eq_of_feature_identifiesAtZero
    (condition j) (V j) (hidentify j) p q (hcondition j) (hzero j)

/-- Multi-feature source-law lift: feature-space zero-drift theorems for all
features plus one embedded feature imply equality of the source laws. -/
theorem source_eq_of_each_feature_identifiesAtZero_of_embedding [Zero F]
    (Φ : Fin n → FeatureModel X F)
    (condition : Fin n → Distribution F → Distribution F → Prop)
    (V : Fin n → DriftingField F)
    (hidentify : ∀ j, IdentifiesAtZero (condition j) (V j))
    (p q : Distribution X)
    (hcondition : ∀ j, condition j ((Φ j).law p) ((Φ j).law q))
    (hzero : ∀ j, ZeroDrift (V j) ((Φ j).law p) ((Φ j).law q))
    (j : Fin n) (hemb : MeasurableEmbedding (Φ j).toFun) :
    p = q :=
  source_eq_of_allFeatureLawsEqual_of_embedding Φ
    (allFeatureLawsEqual_of_each_feature_identifiesAtZero
      Φ condition V hidentify p q hcondition hzero)
    j hemb

end MultiFeature

/-! ## Heterogeneous feature families -/

section Heterogeneous

variable {X : Type u} [MeasurableSpace X]
variable {n : ℕ} {F : Fin n → Type v} [∀ j, MeasurableSpace (F j)]

/-- A finite feature family whose feature spaces may differ by index.  This
matches the practical situation where one may combine, for example, pixel-space,
embedding-space, and classifier-logit features. -/
structure HeterogeneousFeatureFamily
    (X : Type u) [MeasurableSpace X]
    {n : ℕ} (F : Fin n → Type v) [∀ j, MeasurableSpace (F j)] where
  toFun : ∀ j : Fin n, X → F j
  measurable_toFun : ∀ j, Measurable (toFun j)

namespace HeterogeneousFeatureFamily

/-- The law seen by the `j`th heterogeneous feature. -/
noncomputable def law (Φ : HeterogeneousFeatureFamily X F)
    (j : Fin n) (p : Distribution X) : Distribution (F j) :=
  featureLaw (Φ.toFun j) p (Φ.measurable_toFun j).aemeasurable

theorem law_eq_map (Φ : HeterogeneousFeatureFamily X F)
    (j : Fin n) (p : Distribution X) :
    Φ.law j p = Measure.map (Φ.toFun j) p := by
  rfl

/-- Equality of all laws in a heterogeneous feature family. -/
def AllLawsEqual (Φ : HeterogeneousFeatureFamily X F)
    (p q : Distribution X) : Prop :=
  ∀ j : Fin n, Φ.law j p = Φ.law j q

theorem allLawsEqual_of_source_eq (Φ : HeterogeneousFeatureFamily X F)
    {p q : Distribution X} (h : p = q) : Φ.AllLawsEqual p q := by
  intro j
  subst q
  rfl

/-- One embedded heterogeneous feature is enough to lift equality of all
feature laws to equality of source laws. -/
theorem source_eq_of_allLawsEqual_of_embedding
    (Φ : HeterogeneousFeatureFamily X F) {p q : Distribution X}
    (hfeatures : Φ.AllLawsEqual p q)
    (j : Fin n) (hemb : MeasurableEmbedding (Φ.toFun j)) :
    p = q := by
  apply hemb.map_injective
  simpa [law, featureLaw, pushforward] using hfeatures j

/-- A heterogeneous feature family is measure-determining when equality of all
its pushforward laws forces equality of source laws.  This is a condition, not
an automatic theorem; agents must prove it from concrete structure, e.g. from
an embedded feature or another standard measure-determining result. -/
def MeasureDetermining (Φ : HeterogeneousFeatureFamily X F) : Prop :=
  ∀ ⦃p q : Distribution X⦄, Φ.AllLawsEqual p q → p = q

theorem measureDetermining_of_embedding
    (Φ : HeterogeneousFeatureFamily X F)
    (j : Fin n) (hemb : MeasurableEmbedding (Φ.toFun j)) :
    Φ.MeasureDetermining := by
  intro p q hfeatures
  exact Φ.source_eq_of_allLawsEqual_of_embedding hfeatures j hemb

/-- Heterogeneous feature-space identifiability gives equality of all
feature laws, one feature at a time. -/
theorem allLawsEqual_of_each_feature_identifiesAtZero
    [∀ j, Zero (F j)]
    (Φ : HeterogeneousFeatureFamily X F)
    (condition : ∀ j : Fin n, Distribution (F j) → Distribution (F j) → Prop)
    (V : ∀ j : Fin n, DriftingField (F j))
    (hidentify : ∀ j, IdentifiesAtZero (condition j) (V j))
    (p q : Distribution X)
    (hcondition : ∀ j, condition j (Φ.law j p) (Φ.law j q))
    (hzero : ∀ j, ZeroDrift (V j) (Φ.law j p) (Φ.law j q)) :
    Φ.AllLawsEqual p q := by
  intro j
  exact hidentify j (Φ.law j p) (Φ.law j q) (hcondition j) (hzero j)

/-- Heterogeneous feature-space source lift via an independently proved
measure-determining condition. -/
theorem source_eq_of_each_feature_identifiesAtZero_of_measureDetermining
    [∀ j, Zero (F j)]
    (Φ : HeterogeneousFeatureFamily X F)
    (hdet : Φ.MeasureDetermining)
    (condition : ∀ j : Fin n, Distribution (F j) → Distribution (F j) → Prop)
    (V : ∀ j : Fin n, DriftingField (F j))
    (hidentify : ∀ j, IdentifiesAtZero (condition j) (V j))
    (p q : Distribution X)
    (hcondition : ∀ j, condition j (Φ.law j p) (Φ.law j q))
    (hzero : ∀ j, ZeroDrift (V j) (Φ.law j p) (Φ.law j q)) :
    p = q :=
  hdet (Φ.allLawsEqual_of_each_feature_identifiesAtZero
    condition V hidentify p q hcondition hzero)

/-- Heterogeneous feature-space source lift via one embedded feature. -/
theorem source_eq_of_each_feature_identifiesAtZero_of_embedding
    [∀ j, Zero (F j)]
    (Φ : HeterogeneousFeatureFamily X F)
    (condition : ∀ j : Fin n, Distribution (F j) → Distribution (F j) → Prop)
    (V : ∀ j : Fin n, DriftingField (F j))
    (hidentify : ∀ j, IdentifiesAtZero (condition j) (V j))
    (p q : Distribution X)
    (hcondition : ∀ j, condition j (Φ.law j p) (Φ.law j q))
    (hzero : ∀ j, ZeroDrift (V j) (Φ.law j p) (Φ.law j q))
    (j : Fin n) (hemb : MeasurableEmbedding (Φ.toFun j)) :
    p = q :=
  Φ.source_eq_of_each_feature_identifiesAtZero_of_measureDetermining
    (Φ.measureDetermining_of_embedding j hemb)
    condition V hidentify p q hcondition hzero

/-! ### Quantitative feature-stability certificates -/

/-- A quantitative, independently checkable replacement for exact
measure-determining.  It certifies that a chosen source discrepancy is bounded
by a finite weighted sum of feature-law discrepancies.  The project may later
instantiate this with total variation, Wasserstein, MMD, or a task-specific
metric; this definition itself assumes none of those facts. -/
structure FeatureStabilityCertificate
    (Φ : HeterogeneousFeatureFamily X F)
    (sourceDist : Distribution X → Distribution X → ℝ)
    (featureDist : ∀ j : Fin n, Distribution (F j) → Distribution (F j) → ℝ)
    (C : ℝ) : Prop where
  nonneg_C : 0 ≤ C
  control : ∀ p q,
    sourceDist p q ≤
      C * ∑ j : Fin n, featureDist j (Φ.law j p) (Φ.law j q)

/-- Approximate feature matching controls the certified source discrepancy. -/
theorem sourceDist_le_of_featureDist_le
    (Φ : HeterogeneousFeatureFamily X F)
    {sourceDist : Distribution X → Distribution X → ℝ}
    {featureDist : ∀ j : Fin n, Distribution (F j) → Distribution (F j) → ℝ}
    {C : ℝ} (cert : FeatureStabilityCertificate Φ sourceDist featureDist C)
    (p q : Distribution X) (ε : Fin n → ℝ)
    (hε : ∀ j, featureDist j (Φ.law j p) (Φ.law j q) ≤ ε j) :
    sourceDist p q ≤ C * ∑ j : Fin n, ε j := by
  calc sourceDist p q
      ≤ C * ∑ j : Fin n, featureDist j (Φ.law j p) (Φ.law j q) :=
        cert.control p q
    _ ≤ C * ∑ j : Fin n, ε j := by
        exact mul_le_mul_of_nonneg_left
          (Finset.sum_le_sum fun j _ => hε j) cert.nonneg_C

/-- Exact zero feature discrepancies force the certified source discrepancy to
be nonpositive.  A separate separating/nonnegativity fact for `sourceDist` is
needed to turn this into source-law equality. -/
theorem sourceDist_le_zero_of_featureDist_zero
    (Φ : HeterogeneousFeatureFamily X F)
    {sourceDist : Distribution X → Distribution X → ℝ}
    {featureDist : ∀ j : Fin n, Distribution (F j) → Distribution (F j) → ℝ}
    {C : ℝ} (cert : FeatureStabilityCertificate Φ sourceDist featureDist C)
    (p q : Distribution X)
    (hzero : ∀ j, featureDist j (Φ.law j p) (Φ.law j q) = 0) :
    sourceDist p q ≤ 0 := by
  simpa using
    (sourceDist_le_of_featureDist_le Φ cert p q (fun _ => 0)
      (fun j => by simp [hzero j]))

/-- A source discrepancy separates measures when being at most zero forces
source-law equality.  This packages the final step for exact recovery from a
quantitative certificate without assuming anything about feature maps. -/
def SourceDistanceSeparates
    (sourceDist : Distribution X → Distribution X → ℝ) : Prop :=
  ∀ p q, sourceDist p q ≤ 0 → p = q

/-- If a quantitative certificate controls a separating source discrepancy,
then zero feature discrepancies imply source-law equality. -/
theorem source_eq_of_featureDist_zero_of_stability
    (Φ : HeterogeneousFeatureFamily X F)
    {sourceDist : Distribution X → Distribution X → ℝ}
    {featureDist : ∀ j : Fin n, Distribution (F j) → Distribution (F j) → ℝ}
    {C : ℝ} (cert : FeatureStabilityCertificate Φ sourceDist featureDist C)
    (hsep : SourceDistanceSeparates sourceDist)
    (p q : Distribution X)
    (hzero : ∀ j, featureDist j (Φ.law j p) (Φ.law j q) = 0) :
    p = q :=
  hsep p q (sourceDist_le_zero_of_featureDist_zero Φ cert p q hzero)

end HeterogeneousFeatureFamily

end Heterogeneous

/-! ## Formal warning: non-injective features collapse source laws -/

/-- A non-injective feature collision makes the two feature Dirac laws equal.
This is the default obstruction: feature matching alone sees only `φ♯p`. -/
theorem featureLaw_dirac_eq_of_collision
    {X : Type u} {F : Type v} [MeasurableSpace X] [MeasurableSpace F]
    (φ : X → F) (hφ : Measurable φ) {a b : X}
    (hcollapse : φ a = φ b) :
    featureLaw φ (Measure.dirac a) hφ.aemeasurable =
      featureLaw φ (Measure.dirac b) hφ.aemeasurable := by
  simp [featureLaw, pushforward, Measure.map_dirac' hφ, hcollapse]

/-- In a measurable space that separates points, a non-injective feature
collision gives genuinely distinct source Dirac laws with equal feature laws. -/
theorem featureLaw_collision_distinct_source_diracs
    {X : Type u} {F : Type v} [MeasurableSpace X] [MeasurableSpace F]
    [MeasurableSpace.SeparatesPoints X]
    (φ : X → F) (hφ : Measurable φ) {a b : X}
    (hab : a ≠ b) (hcollapse : φ a = φ b) :
    featureLaw φ (Measure.dirac a) hφ.aemeasurable =
        featureLaw φ (Measure.dirac b) hφ.aemeasurable ∧
      (Measure.dirac a : Distribution X) ≠ Measure.dirac b :=
  ⟨featureLaw_dirac_eq_of_collision φ hφ hcollapse, MeasureTheory.dirac_ne_dirac hab⟩

end DriftingIdentifiability
