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
