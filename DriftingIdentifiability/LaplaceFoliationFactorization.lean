import DriftingIdentifiability.LaplaceFoliationChart
import Mathlib.Analysis.Calculus.ContDiff.Operations

/-!
# G4 regular foliation: local scalar factorization

This file packages the implicit chart from `LaplaceFoliationChart` into the
local functional-dependence statement needed by the non-degenerate-leaf
argument.  The analytic input is deliberately generic: in a `C¹` chart whose
first coordinate is a scalar function `ψ`, a differentiable function whose
derivative annihilates `ker dψ` is constant on every connected vertical
slice.  The final section specializes this to the actual Laplace normalizer
ratio under zero drift.
-/

open MeasureTheory Filter Topology Set
open scoped RealInnerProductSpace

namespace DriftingIdentifiability

open Paper

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
  [SecondCountableTopology E] [FiniteDimensional ℝ E]

/-- The integrated Laplace displacement potential is genuinely `C¹`: its
Fréchet derivative is the inner product with the continuous displacement
field. -/
theorem contDiff_one_laplaceDisplacementPotential
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] :
    ContDiff ℝ 1 (laplaceDisplacementPotential τ μ) := by
  rw [contDiff_one_iff_fderiv]
  refine ⟨?_, ?_⟩
  · exact fun x =>
      (hasFDerivAt_laplaceDisplacementPotential hτ μ x).differentiableAt
  · have hfield : Continuous (laplaceDisplacementField τ μ) :=
      continuous_laplaceDisplacementField hτ μ
    have hinner : Continuous
        (fun x => innerSL ℝ (laplaceDisplacementField τ μ x)) :=
      (innerSL ℝ).continuous.comp hfield
    convert hinner using 1
    funext x
    exact (hasFDerivAt_laplaceDisplacementPotential hτ μ x).fderiv

/-- The regular set of the integrated displacement field is open. -/
theorem isOpen_laplaceDisplacementField_ne_zero
    {τ : ℝ} (hτ : 0 < τ) (μ : Measure E) [IsFiniteMeasure μ] :
    IsOpen {x | laplaceDisplacementField τ μ x ≠ 0} := by
  change IsOpen ((laplaceDisplacementField τ μ ⁻¹' ({0} : Set E))ᶜ)
  exact (isClosed_singleton.preimage
    (continuous_laplaceDisplacementField hτ μ)).isOpen_compl

/-- Restrict the regular leaf chart to the open regular set and then to its
maximal `C¹` source/target.  This makes differentiability of the inverse chart
available at every point used below, rather than only at the base point. -/
noncomputable def laplaceRegularLeafC1Chart
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    OpenPartialHomeomorph E
      (ℝ × (innerSL ℝ (laplaceDisplacementField τ q x)).ker) :=
  (((laplaceRegularLeafChart hτ q x hreg).restrOpen
      {z | laplaceDisplacementField τ q z ≠ 0}
      (isOpen_laplaceDisplacementField_ne_zero hτ q)).restrContDiff
        ℝ 1 (by simp))

private theorem contDiffAt_laplaceRegularLeafChart
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    ContDiffAt ℝ 1 (laplaceRegularLeafChart hτ q x hreg) x := by
  let hf := hasStrictFDerivAt_laplaceDisplacementPotential hτ q x
  let f' := innerSL ℝ (laplaceDisplacementField τ q x)
  let hrange : f'.range = ⊤ := innerSL_range_eq_top hreg
  let hker : f'.ker.ClosedComplemented :=
    Submodule.ClosedComplemented.of_finiteDimensional f'.ker
  change ContDiffAt ℝ 1
    (fun z =>
      (laplaceDisplacementPotential τ q z,
        Classical.choose hker (z - x))) x
  exact (contDiff_one_laplaceDisplacementPotential hτ q).contDiffAt.prodMk
    ((Classical.choose hker).contDiff.comp
      (contDiff_id.sub contDiff_const)).contDiffAt

/-- The base point belongs to the source of the `C¹` regular chart. -/
theorem laplaceRegularLeafC1Chart_mem_source
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    x ∈ (laplaceRegularLeafC1Chart hτ q x hreg).source := by
  let e := laplaceRegularLeafChart hτ q x hreg
  let er := e.restrOpen {z | laplaceDisplacementField τ q z ≠ 0}
    (isOpen_laplaceDisplacementField_ne_zero hτ q)
  have hx_e : x ∈ e.source := laplaceRegularLeafChart_mem_source hτ q x hreg
  have hx_er : x ∈ er.source := by
    exact ⟨hx_e, hreg⟩
  have heC1 : ContDiffAt ℝ 1 er x := by
    change ContDiffAt ℝ 1 e x
    exact contDiffAt_laplaceRegularLeafChart hτ q x hreg
  have hmap : er x ∈ er.target := er.map_source hx_er
  let φ := HasStrictFDerivAt.implicitFunctionDataOfComplemented
    (laplaceDisplacementPotential τ q)
    (innerSL ℝ (laplaceDisplacementField τ q x))
    (hasStrictFDerivAt_laplaceDisplacementPotential hτ q x)
    (innerSL_range_eq_top hreg)
    (Submodule.ClosedComplemented.of_finiteDimensional
      (innerSL ℝ (laplaceDisplacementField τ q x)).ker)
  let L : E ≃L[ℝ]
      ℝ × (innerSL ℝ (laplaceDisplacementField τ q x)).ker :=
    φ.leftDeriv.equivProdOfSurjectiveOfIsCompl φ.rightDeriv
      φ.range_leftDeriv φ.range_rightDeriv φ.isCompl_ker
  have hstrict : HasStrictFDerivAt er
      (L : E →L[ℝ]
        ℝ × (innerSL ℝ (laplaceDisplacementField τ q x)).ker) x := by
    change HasStrictFDerivAt e
      (L : E →L[ℝ]
        ℝ × (innerSL ℝ (laplaceDisplacementField τ q x)).ker) x
    change HasStrictFDerivAt φ.prodFun
      (L : E →L[ℝ]
        ℝ × (innerSL ℝ (laplaceDisplacementField τ q x)).ker) x
    simpa [φ, L] using φ.hasStrictFDerivAt
  have hinv : er.symm (er x) = x := er.left_inv hx_er
  have einvC1 : ContDiffAt ℝ 1 er.symm (er x) := by
    apply er.contDiffAt_symm (f₀' := L) hmap
    · rw [hinv]
      exact hstrict.hasFDerivAt
    · rw [hinv]
      exact heC1
  have hx_restr : x ∈ (er.restrContDiff ℝ 1 (by simp)).source :=
    ⟨hx_er, heC1, by simpa [er.left_inv hx_er] using einvC1⟩
  simpa [laplaceRegularLeafC1Chart, er, e] using hx_restr

/-- The first coordinate of the restricted `C¹` chart is still the actual
Laplace displacement potential. -/
@[simp] theorem laplaceRegularLeafC1Chart_fst
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) (z : E) :
    (laplaceRegularLeafC1Chart hτ q x hreg z).1 =
      laplaceDisplacementPotential τ q z := by
  exact laplaceRegularLeafChart_fst hτ q x hreg z

/-! ## Generic vertical-slice theorem -/

section VerticalSlice

variable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
  [CompleteSpace F]

omit [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
    [SecondCountableTopology E] [FiniteDimensional ℝ E] [CompleteSpace F] in
/-- In a `C¹` scalar-first-coordinate chart, tangent annihilation makes a
function constant on every connected open vertical slice contained in the
chart target.  This is the precise mean-value-theorem bridge between the
pointwise differential statement and local scalar factorization. -/
theorem openPartialHomeomorph_constOn_verticalSlice
    (e : OpenPartialHomeomorph E (ℝ × F)) (ψ R : E → ℝ)
    (heC1 : ContDiffOn ℝ 1 e.symm e.target)
    (hfst : ∀ z, (e z).1 = ψ z)
    (hψ : Differentiable ℝ ψ)
    (hR : ∀ z ∈ e.source, DifferentiableAt ℝ R z)
    (htangent : ∀ z ∈ e.source, ∀ v,
      fderiv ℝ ψ z v = 0 → fderiv ℝ R z v = 0)
    {s : ℝ} {V : Set F} (hVopen : IsOpen V) (hVconn : IsPreconnected V)
    (hVtarget : ∀ y ∈ V, (s, y) ∈ e.target)
    {y₀ y₁ : F} (hy₀ : y₀ ∈ V) (hy₁ : y₁ ∈ V) :
    R (e.symm (s, y₀)) = R (e.symm (s, y₁)) := by
  let g : F → ℝ := fun y => R (e.symm (s, y))
  have hgdiff : DifferentiableOn ℝ g V := by
    intro y hy
    have hwy : (s, y) ∈ e.target := hVtarget y hy
    have hinv : DifferentiableAt ℝ e.symm (s, y) :=
      ((heC1 (s, y) hwy).contDiffAt
        (e.open_target.mem_nhds hwy)).differentiableAt (by norm_num)
    have hprod : DifferentiableAt ℝ (fun u : F => (s, u)) y :=
      (differentiableAt_const s).prodMk differentiableAt_id
    exact (hR (e.symm (s, y)) (e.map_target hwy)).comp y
      (hinv.comp y hprod) |>
        DifferentiableAt.differentiableWithinAt
  have hgzero : V.EqOn (fderiv ℝ g) 0 := by
    intro y hy
    have hwy : (s, y) ∈ e.target := hVtarget y hy
    have hzsource : e.symm (s, y) ∈ e.source := e.map_target hwy
    have hinv : DifferentiableAt ℝ e.symm (s, y) :=
      ((heC1 (s, y) hwy).contDiffAt
        (e.open_target.mem_nhds hwy)).differentiableAt (by norm_num)
    have hprod : DifferentiableAt ℝ (fun u : F => (s, u)) y :=
      (differentiableAt_const s).prodMk differentiableAt_id
    have hparam : DifferentiableAt ℝ (fun u : F => e.symm (s, u)) y :=
      hinv.comp y hprod
    have hto : Tendsto (fun u : F => (s, u)) (𝓝 y) (𝓝 (s, y)) :=
      tendsto_const_nhds.prodMk_nhds tendsto_id
    have hstay : ∀ᶠ u in 𝓝 y, (s, u) ∈ e.target :=
      hto (e.open_target.mem_nhds hwy)
    have hlevel : (fun u : F => ψ (e.symm (s, u))) =ᶠ[𝓝 y]
        (fun _ => s) := by
      filter_upwards [hstay] with u hu
      calc
        ψ (e.symm (s, u)) = (e (e.symm (s, u))).1 := (hfst _).symm
        _ = s := congrArg Prod.fst (e.right_inv hu)
    have hlevelDeriv : fderiv ℝ (fun u : F => ψ (e.symm (s, u))) y = 0 := by
      rw [hlevel.fderiv_eq]
      simp
    have hψchain : HasFDerivAt (ψ ∘ fun u : F => e.symm (s, u))
        ((fderiv ℝ ψ (e.symm (s, y))).comp
          (fderiv ℝ (fun u : F => e.symm (s, u)) y)) y :=
      (hψ (e.symm (s, y))).hasFDerivAt.comp y hparam.hasFDerivAt
    have hRchain : HasFDerivAt (R ∘ fun u : F => e.symm (s, u))
        ((fderiv ℝ R (e.symm (s, y))).comp
          (fderiv ℝ (fun u : F => e.symm (s, u)) y)) y :=
      (hR (e.symm (s, y)) hzsource).hasFDerivAt.comp y hparam.hasFDerivAt
    apply ContinuousLinearMap.ext
    intro v
    have htan : fderiv ℝ ψ (e.symm (s, y))
        (fderiv ℝ (fun u : F => e.symm (s, u)) y v) = 0 := by
      calc
        fderiv ℝ ψ (e.symm (s, y))
            (fderiv ℝ (fun u : F => e.symm (s, u)) y v) =
            fderiv ℝ (fun u : F => ψ (e.symm (s, u))) y v := by
              have hc := congrArg (fun L : F →L[ℝ] ℝ => L v) hψchain.fderiv
              simpa [Function.comp_def] using hc.symm
        _ = 0 := by rw [hlevelDeriv]; rfl
    calc
      fderiv ℝ g y v =
          fderiv ℝ R (e.symm (s, y))
            (fderiv ℝ (fun u : F => e.symm (s, u)) y v) := by
              have hc := congrArg (fun L : F →L[ℝ] ℝ => L v) hRchain.fderiv
              simpa [g, Function.comp_def] using hc
      _ = 0 := htangent (e.symm (s, y)) hzsource _ htan
      _ = (0 : F →L[ℝ] ℝ) v := rfl
  exact hVopen.is_const_of_fderiv_eq_zero hVconn hgdiff hgzero hy₀ hy₁

omit [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
    [SecondCountableTopology E] [FiniteDimensional ℝ E] [CompleteSpace F] in
/-- Local scalar factorization obtained from the vertical-slice theorem.  On
a product ball in chart coordinates, `R` depends only on the first coordinate
`ψ`.  No constant-rank or functional-dependence theorem is assumed. -/
theorem openPartialHomeomorph_exists_localDifferentiableFactorization
    (e : OpenPartialHomeomorph E (ℝ × F)) (ψ R : E → ℝ)
    (heC1 : ContDiffOn ℝ 1 e.symm e.target)
    (hfst : ∀ z, (e z).1 = ψ z)
    (hψ : Differentiable ℝ ψ)
    (hR : ∀ z ∈ e.source, DifferentiableAt ℝ R z)
    (htangent : ∀ z ∈ e.source, ∀ v,
      fderiv ℝ ψ z v = 0 → fderiv ℝ R z v = 0)
    {x : E} (hx : x ∈ e.source) :
    ∃ ε > 0, ∃ h : ℝ → ℝ,
      DifferentiableOn ℝ h (Metric.ball (ψ x) ε) ∧
        ∀ z ∈ e.source, dist (ψ z) (ψ x) < ε →
          dist (e z).2 (e x).2 < ε → R z = h (ψ z) := by
  have hxmap : e x ∈ e.target := e.map_source hx
  obtain ⟨ε, hε, hball⟩ := Metric.isOpen_iff.mp e.open_target (e x) hxmap
  let h : ℝ → ℝ := fun s => R (e.symm (s, (e x).2))
  have hhdiff : DifferentiableOn ℝ h (Metric.ball (ψ x) ε) := by
    intro s hs
    have htarget : (s, (e x).2) ∈ e.target := by
      apply hball
      rw [Metric.mem_ball, Prod.dist_eq]
      change max (dist s (e x).1) (dist (e x).2 (e x).2) < ε
      rw [hfst x, dist_self, max_eq_left dist_nonneg]
      exact hs
    have hinv : DifferentiableAt ℝ e.symm (s, (e x).2) :=
      ((heC1 (s, (e x).2) htarget).contDiffAt
        (e.open_target.mem_nhds htarget)).differentiableAt (by norm_num)
    have hline : DifferentiableAt ℝ (fun u : ℝ => (u, (e x).2)) s :=
      differentiableAt_id.prodMk (differentiableAt_const (e x).2)
    exact (hR (e.symm (s, (e x).2)) (e.map_target htarget)).comp s
      (hinv.comp s hline) |>.differentiableWithinAt
  refine ⟨ε, hε, h, hhdiff, ?_⟩
  intro z hz hfirst hsecond
  let s : ℝ := ψ z
  let y : F := (e z).2
  let V : Set F := Metric.ball (e x).2 ε
  have hVopen : IsOpen V := Metric.isOpen_ball
  have hVconn : IsPreconnected V := (convex_ball (e x).2 ε).isPreconnected
  have hVtarget : ∀ u ∈ V, (s, u) ∈ e.target := by
    intro u hu
    apply hball
    rw [Metric.mem_ball, Prod.dist_eq]
    change max (dist s (e x).1) (dist u (e x).2) < ε
    rw [hfst x]
    exact max_lt hfirst hu
  have hy : y ∈ V := hsecond
  have hybase : (e x).2 ∈ V := Metric.mem_ball_self hε
  have hc := openPartialHomeomorph_constOn_verticalSlice e ψ R heC1 hfst hψ hR
    htangent hVopen hVconn hVtarget hy hybase
  have hecoords : (s, y) = e z := by
    apply Prod.ext
    · simpa [s] using (hfst z).symm
    · rfl
  rw [hecoords, e.left_inv hz] at hc
  exact hc

omit [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
    [SecondCountableTopology E] [FiniteDimensional ℝ E] [CompleteSpace F] in
/-- Version of the local factorization theorem that forgets the additionally
proved differentiability of the one-variable factor. -/
theorem openPartialHomeomorph_exists_localFactorization
    (e : OpenPartialHomeomorph E (ℝ × F)) (ψ R : E → ℝ)
    (heC1 : ContDiffOn ℝ 1 e.symm e.target)
    (hfst : ∀ z, (e z).1 = ψ z)
    (hψ : Differentiable ℝ ψ)
    (hR : ∀ z ∈ e.source, DifferentiableAt ℝ R z)
    (htangent : ∀ z ∈ e.source, ∀ v,
      fderiv ℝ ψ z v = 0 → fderiv ℝ R z v = 0)
    {x : E} (hx : x ∈ e.source) :
    ∃ ε > 0, ∃ h : ℝ → ℝ, ∀ z ∈ e.source,
      dist (ψ z) (ψ x) < ε →
      dist (e z).2 (e x).2 < ε → R z = h (ψ z) := by
  obtain ⟨ε, hε, h, _hdiff, hfactor⟩ :=
    openPartialHomeomorph_exists_localDifferentiableFactorization e ψ R heC1
      hfst hψ hR htangent hx
  exact ⟨ε, hε, h, hfactor⟩

end VerticalSlice

omit [MeasurableSpace E] [BorelSpace E] [CompleteSpace E]
    [SecondCountableTopology E] [FiniteDimensional ℝ E] in
/-- Differentiating a genuine local scalar factorization along the gradient
gives the one-variable derivative times the squared gradient norm. -/
theorem fderiv_gradient_eq_deriv_mul_norm_sq_of_eventuallyEq
    {ψ R : E → ℝ} {g : ℝ → ℝ} {z D : E}
    (hψ : HasFDerivAt ψ (innerSL ℝ D) z)
    (hg : DifferentiableAt ℝ g (ψ z))
    (heq : R =ᶠ[𝓝 z] fun w => g (ψ w)) :
    fderiv ℝ R z D = deriv g (ψ z) * ‖D‖ ^ 2 := by
  have hchain : HasFDerivAt (g ∘ ψ)
      ((ContinuousLinearMap.toSpanSingleton ℝ (deriv g (ψ z))).comp
        (innerSL ℝ D)) z :=
    by simpa only [toSpanSingleton_deriv] using
      hg.hasFDerivAt.comp z hψ
  have hfd : fderiv ℝ R z = fderiv ℝ (g ∘ ψ) z := by
    simpa only [Function.comp_def] using heq.fderiv_eq
  rw [hfd, hchain.fderiv]
  simp [innerSL_apply_apply, mul_comm]

omit [InnerProductSpace ℝ E] [MeasurableSpace E] [BorelSpace E]
    [CompleteSpace E] [SecondCountableTopology E] [FiniteDimensional ℝ E] in
/-- A product-ball factorization in an open chart is a genuine germ equality
at every point satisfying the strict product-ball inequalities. -/
theorem eventuallyEq_of_openPartialHomeomorph_localFactorization
    {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    (e : OpenPartialHomeomorph E (ℝ × F)) (ψ R : E → ℝ)
    (hψcont : Continuous ψ) {x z : E} {ε : ℝ} {h : ℝ → ℝ}
    (hfactor : ∀ w ∈ e.source,
      dist (ψ w) (ψ x) < ε →
      dist (e w).2 (e x).2 < ε → R w = h (ψ w))
    (hz : z ∈ e.source)
    (hfirst : dist (ψ z) (ψ x) < ε)
    (hsecond : dist (e z).2 (e x).2 < ε) :
    R =ᶠ[𝓝 z] fun w => h (ψ w) := by
  have hsrc : ∀ᶠ w in 𝓝 z, w ∈ e.source := e.open_source.mem_nhds hz
  have hfirst_ev : ∀ᶠ w in 𝓝 z, dist (ψ w) (ψ x) < ε :=
    ((hψcont.continuousAt.dist continuousAt_const).eventually_lt
      continuousAt_const hfirst)
  have hsecond_ev : ∀ᶠ w in 𝓝 z, dist (e w).2 (e x).2 < ε :=
    ((((e.continuousAt hz).snd.dist continuousAt_const).eventually_lt
      continuousAt_const hsecond))
  filter_upwards [hsrc, hfirst_ev, hsecond_ev] with w hw hwfirst hwsecond
  exact hfactor w hw hwfirst hwsecond

/-! ## Actual Laplace local factorizations -/

/-- Every source point of the regular `C¹` chart is genuinely regular. -/
theorem laplaceRegularLeafC1Chart_regular
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0)
    {z : E} (hz : z ∈ (laplaceRegularLeafC1Chart hτ q x hreg).source) :
    laplaceDisplacementField τ q z ≠ 0 := by
  exact hz.1.2

/-- The restricted chart still sends its base point to potential value and
zero tangent coordinate. -/
@[simp] theorem laplaceRegularLeafC1Chart_apply_self
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    laplaceRegularLeafC1Chart hτ q x hreg x =
      (laplaceDisplacementPotential τ q x, 0) := by
  exact laplaceRegularLeafChart_apply_self hτ q x hreg

/-- The inverse of the audited regular chart is `C¹` throughout its target. -/
theorem contDiffOn_laplaceRegularLeafC1Chart_symm
    {τ : ℝ} (hτ : 0 < τ) (q : Measure E) [IsFiniteMeasure q]
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    ContDiffOn ℝ 1 (laplaceRegularLeafC1Chart hτ q x hreg).symm
      (laplaceRegularLeafC1Chart hτ q x hreg).target := by
  unfold laplaceRegularLeafC1Chart
  exact OpenPartialHomeomorph.contDiffOn_restrContDiff_target ℝ _ (by simp)

/-- **Actual ratio factorization.**  Under zero drift, near every regular
point the normalizer ratio is a scalar function of the `q` displacement
potential.  The neighborhood is an explicit product ball in the certified
implicit chart. -/
theorem laplaceNormalizerRatio_exists_localFactorization_of_zeroDrift
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    ∃ ε > 0, ∃ h : ℝ → ℝ,
      ∀ z ∈ (laplaceRegularLeafC1Chart hτ q x hreg).source,
        dist (laplaceDisplacementPotential τ q z)
            (laplaceDisplacementPotential τ q x) < ε →
        dist (laplaceRegularLeafC1Chart hτ q x hreg z).2 0 < ε →
        laplaceNormalizerRatio τ p q z =
          h (laplaceDisplacementPotential τ q z) := by
  let e := laplaceRegularLeafC1Chart hτ q x hreg
  have hfactor := openPartialHomeomorph_exists_localFactorization e
    (laplaceDisplacementPotential τ q) (laplaceNormalizerRatio τ p q)
    (contDiffOn_laplaceRegularLeafC1Chart_symm hτ q x hreg)
    (laplaceRegularLeafC1Chart_fst hτ q x hreg)
    (fun z => (hasFDerivAt_laplaceDisplacementPotential hτ q z).differentiableAt)
    (fun z hz => differentiableAt_laplaceNormalizerRatio_of_zeroDrift hτ p q hzero
      (laplaceRegularLeafC1Chart_regular hτ q x hreg hz))
    (fun z hz v hv => by
      have hv' : ⟪laplaceDisplacementField τ q z, v⟫ = 0 := by
        rw [(hasFDerivAt_laplaceDisplacementPotential hτ q z).fderiv] at hv
        exact hv
      exact laplaceNormalizerRatio_fderiv_tangent_eq_zero hτ p q hzero
        (laplaceRegularLeafC1Chart_regular hτ q x hreg hz) hv')
    (laplaceRegularLeafC1Chart_mem_source hτ q x hreg)
  simpa only [e, laplaceRegularLeafC1Chart_apply_self, Prod.snd] using hfactor

/-- Strengthened ratio factorization exposing that the one-variable factor is
differentiable throughout its leaf-value interval. -/
theorem laplaceNormalizerRatio_exists_localDifferentiableFactorization_of_zeroDrift
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    ∃ ε > 0, ∃ h : ℝ → ℝ,
      DifferentiableOn ℝ h
        (Metric.ball (laplaceDisplacementPotential τ q x) ε) ∧
      ∀ z ∈ (laplaceRegularLeafC1Chart hτ q x hreg).source,
        dist (laplaceDisplacementPotential τ q z)
            (laplaceDisplacementPotential τ q x) < ε →
        dist (laplaceRegularLeafC1Chart hτ q x hreg z).2 0 < ε →
        laplaceNormalizerRatio τ p q z =
          h (laplaceDisplacementPotential τ q z) := by
  let e := laplaceRegularLeafC1Chart hτ q x hreg
  have hfactor := openPartialHomeomorph_exists_localDifferentiableFactorization e
    (laplaceDisplacementPotential τ q) (laplaceNormalizerRatio τ p q)
    (contDiffOn_laplaceRegularLeafC1Chart_symm hτ q x hreg)
    (laplaceRegularLeafC1Chart_fst hτ q x hreg)
    (fun z => (hasFDerivAt_laplaceDisplacementPotential hτ q z).differentiableAt)
    (fun z hz => differentiableAt_laplaceNormalizerRatio_of_zeroDrift hτ p q hzero
      (laplaceRegularLeafC1Chart_regular hτ q x hreg hz))
    (fun z hz v hv => by
      have hv' : ⟪laplaceDisplacementField τ q z, v⟫ = 0 := by
        rw [(hasFDerivAt_laplaceDisplacementPotential hτ q z).fderiv] at hv
        exact hv
      exact laplaceNormalizerRatio_fderiv_tangent_eq_zero hτ p q hzero
        (laplaceRegularLeafC1Chart_regular hτ q x hreg hz) hv')
    (laplaceRegularLeafC1Chart_mem_source hτ q x hreg)
  simpa only [e, laplaceRegularLeafC1Chart_apply_self, Prod.snd] using hfactor

/-- **Actual potential factorization.**  On the same type of regular chart,
zero drift also makes the `p` potential a scalar function of the `q`
potential. -/
theorem laplaceDisplacementPotential_exists_localFactorization_of_zeroDrift
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    ∃ ε > 0, ∃ G : ℝ → ℝ,
      ∀ z ∈ (laplaceRegularLeafC1Chart hτ q x hreg).source,
        dist (laplaceDisplacementPotential τ q z)
            (laplaceDisplacementPotential τ q x) < ε →
        dist (laplaceRegularLeafC1Chart hτ q x hreg z).2 0 < ε →
        laplaceDisplacementPotential τ p z =
          G (laplaceDisplacementPotential τ q z) := by
  let e := laplaceRegularLeafC1Chart hτ q x hreg
  have hfactor := openPartialHomeomorph_exists_localFactorization e
    (laplaceDisplacementPotential τ q) (laplaceDisplacementPotential τ p)
    (contDiffOn_laplaceRegularLeafC1Chart_symm hτ q x hreg)
    (laplaceRegularLeafC1Chart_fst hτ q x hreg)
    (fun z => (hasFDerivAt_laplaceDisplacementPotential hτ q z).differentiableAt)
    (fun z _ => (hasFDerivAt_laplaceDisplacementPotential hτ p z).differentiableAt)
    (fun z _ v hv => by
      rw [(hasFDerivAt_laplaceDisplacementPotential hτ q z).fderiv] at hv
      rw [innerSL_apply_apply] at hv
      rw [(hasFDerivAt_laplaceDisplacementPotential hτ p z).fderiv,
        laplaceDisplacementField_eq_ratio_smul_of_zeroDrift hτ p q hzero z]
      simp only [innerSL_apply_apply, real_inner_smul_left, hv, mul_zero])
    (laplaceRegularLeafC1Chart_mem_source hτ q x hreg)
  simpa only [e, laplaceRegularLeafC1Chart_apply_self, Prod.snd] using hfactor

/-- **Non-degenerate leaves manufacture Abel seeds.**  Around every regular
base point there is a chart radius such that two points on the same local
`q`-potential leaf with different squared gradient norms force the actual
defect `ψ_p - (Z_p/Z_q) ψ_q` to vanish at both points. -/
theorem laplaceFoliation_exists_nondegenerateLeaf_seed
    {τ : ℝ} (hτ : 0 < τ) (p q : Measure E)
    [IsProbabilityMeasure p] [IsProbabilityMeasure q]
    (hzero : ZeroDrift (meanShiftDrift (laplaceKernel τ)) p q)
    (x : E) (hreg : laplaceDisplacementField τ q x ≠ 0) :
    ∃ ε > 0, ∀ z₁ z₂,
      z₁ ∈ (laplaceRegularLeafC1Chart hτ q x hreg).source →
      z₂ ∈ (laplaceRegularLeafC1Chart hτ q x hreg).source →
      dist (laplaceDisplacementPotential τ q z₁)
          (laplaceDisplacementPotential τ q x) < ε →
      dist (laplaceRegularLeafC1Chart hτ q x hreg z₁).2 0 < ε →
      dist (laplaceDisplacementPotential τ q z₂)
          (laplaceDisplacementPotential τ q x) < ε →
      dist (laplaceRegularLeafC1Chart hτ q x hreg z₂).2 0 < ε →
      laplaceDisplacementPotential τ q z₁ =
          laplaceDisplacementPotential τ q z₂ →
      ‖laplaceDisplacementField τ q z₁‖ ^ 2 ≠
          ‖laplaceDisplacementField τ q z₂‖ ^ 2 →
      (laplaceDisplacementPotential τ p z₁ -
          laplaceNormalizerRatio τ p q z₁ *
            laplaceDisplacementPotential τ q z₁ = 0) ∧
        (laplaceDisplacementPotential τ p z₂ -
          laplaceNormalizerRatio τ p q z₂ *
            laplaceDisplacementPotential τ q z₂ = 0) := by
  obtain ⟨εR, hεR, h, hhdiff, hRfactor⟩ :=
    laplaceNormalizerRatio_exists_localDifferentiableFactorization_of_zeroDrift
      hτ p q hzero x hreg
  obtain ⟨εG, hεG, G, hGfactor⟩ :=
    laplaceDisplacementPotential_exists_localFactorization_of_zeroDrift
      hτ p q hzero x hreg
  let ε := min εR εG
  have hε : 0 < ε := lt_min hεR hεG
  refine ⟨ε, hε, ?_⟩
  intro z₁ z₂ hz₁ hz₂ hfirst₁ hsecond₁ hfirst₂ hsecond₂ hs hnorm
  have hfirstR₁ : dist (laplaceDisplacementPotential τ q z₁)
      (laplaceDisplacementPotential τ q x) < εR :=
    hfirst₁.trans_le (min_le_left _ _)
  have hfirstR₂ : dist (laplaceDisplacementPotential τ q z₂)
      (laplaceDisplacementPotential τ q x) < εR :=
    hfirst₂.trans_le (min_le_left _ _)
  have hsecondR₁ : dist (laplaceRegularLeafC1Chart hτ q x hreg z₁).2 0 < εR :=
    hsecond₁.trans_le (min_le_left _ _)
  have hsecondR₂ : dist (laplaceRegularLeafC1Chart hτ q x hreg z₂).2 0 < εR :=
    hsecond₂.trans_le (min_le_left _ _)
  have hfirstG₁ : dist (laplaceDisplacementPotential τ q z₁)
      (laplaceDisplacementPotential τ q x) < εG :=
    hfirst₁.trans_le (min_le_right _ _)
  have hfirstG₂ : dist (laplaceDisplacementPotential τ q z₂)
      (laplaceDisplacementPotential τ q x) < εG :=
    hfirst₂.trans_le (min_le_right _ _)
  have hsecondG₁ : dist (laplaceRegularLeafC1Chart hτ q x hreg z₁).2 0 < εG :=
    hsecond₁.trans_le (min_le_right _ _)
  have hsecondG₂ : dist (laplaceRegularLeafC1Chart hτ q x hreg z₂).2 0 < εG :=
    hsecond₂.trans_le (min_le_right _ _)
  have hRf₁ := hRfactor z₁ hz₁ hfirstR₁ hsecondR₁
  have hRf₂ := hRfactor z₂ hz₂ hfirstR₂ hsecondR₂
  have hGf₁ := hGfactor z₁ hz₁ hfirstG₁ hsecondG₁
  have hGf₂ := hGfactor z₂ hz₂ hfirstG₂ hsecondG₂
  have hhAt : DifferentiableAt ℝ h (laplaceDisplacementPotential τ q z₁) :=
    (hhdiff _ hfirstR₁).differentiableAt
      (Metric.isOpen_ball.mem_nhds hfirstR₁)
  have hψcont : Continuous (laplaceDisplacementPotential τ q) :=
    (contDiff_one_laplaceDisplacementPotential hτ q).continuous
  have hRfactor' : ∀ w ∈ (laplaceRegularLeafC1Chart hτ q x hreg).source,
      dist (laplaceDisplacementPotential τ q w)
          (laplaceDisplacementPotential τ q x) < εR →
      dist (laplaceRegularLeafC1Chart hτ q x hreg w).2
          (laplaceRegularLeafC1Chart hτ q x hreg x).2 < εR →
      laplaceNormalizerRatio τ p q w =
        h (laplaceDisplacementPotential τ q w) := by
    simpa only [laplaceRegularLeafC1Chart_apply_self, Prod.snd] using hRfactor
  have hsecondR₁' : dist (laplaceRegularLeafC1Chart hτ q x hreg z₁).2
      (laplaceRegularLeafC1Chart hτ q x hreg x).2 < εR := by
    simpa only [laplaceRegularLeafC1Chart_apply_self, Prod.snd] using hsecondR₁
  have hsecondR₂' : dist (laplaceRegularLeafC1Chart hτ q x hreg z₂).2
      (laplaceRegularLeafC1Chart hτ q x hreg x).2 < εR := by
    simpa only [laplaceRegularLeafC1Chart_apply_self, Prod.snd] using hsecondR₂
  have hlocal₁ : laplaceNormalizerRatio τ p q =ᶠ[𝓝 z₁]
      fun w => h (laplaceDisplacementPotential τ q w) :=
    eventuallyEq_of_openPartialHomeomorph_localFactorization
      (laplaceRegularLeafC1Chart hτ q x hreg)
      (laplaceDisplacementPotential τ q) (laplaceNormalizerRatio τ p q)
      hψcont hRfactor' hz₁ hfirstR₁ hsecondR₁'
  have hlocal₂ : laplaceNormalizerRatio τ p q =ᶠ[𝓝 z₂]
      fun w => h (laplaceDisplacementPotential τ q w) :=
    eventuallyEq_of_openPartialHomeomorph_localFactorization
      (laplaceRegularLeafC1Chart hτ q x hreg)
      (laplaceDisplacementPotential τ q) (laplaceNormalizerRatio τ p q)
      hψcont hRfactor' hz₂ hfirstR₂ hsecondR₂'
  have hd₁ := fderiv_gradient_eq_deriv_mul_norm_sq_of_eventuallyEq
    (hasFDerivAt_laplaceDisplacementPotential hτ q z₁) hhAt hlocal₁
  have hhAt₂ : DifferentiableAt ℝ h (laplaceDisplacementPotential τ q z₂) := by
    simpa [hs] using hhAt
  have hd₂ := fderiv_gradient_eq_deriv_mul_norm_sq_of_eventuallyEq
    (hasFDerivAt_laplaceDisplacementPotential hτ q z₂) hhAt₂ hlocal₂
  have hc₁ := laplaceFoliation_measureCancellation hτ p q hzero
    (laplaceRegularLeafC1Chart_regular hτ q x hreg hz₁)
  have hc₂ := laplaceFoliation_measureCancellation hτ p q hzero
    (laplaceRegularLeafC1Chart_regular hτ q x hreg hz₂)
  rw [hd₁] at hc₁
  rw [hd₂] at hc₂
  have hdefect :
      laplaceDisplacementPotential τ p z₁ -
          laplaceNormalizerRatio τ p q z₁ *
            laplaceDisplacementPotential τ q z₁ =
        laplaceDisplacementPotential τ p z₂ -
          laplaceNormalizerRatio τ p q z₂ *
            laplaceDisplacementPotential τ q z₂ := by
    rw [hGf₁, hGf₂, hRf₁, hRf₂, hs]
  let a := deriv h (laplaceDisplacementPotential τ q z₁)
  let n₁ := ‖laplaceDisplacementField τ q z₁‖ ^ 2
  let n₂ := ‖laplaceDisplacementField τ q z₂‖ ^ 2
  have hsame : τ ^ 2 * (a * n₁) = τ ^ 2 * (a * n₂) := by
    dsimp [a, n₁, n₂]
    calc
      τ ^ 2 * (deriv h (laplaceDisplacementPotential τ q z₁) *
          ‖laplaceDisplacementField τ q z₁‖ ^ 2) =
          laplaceDisplacementPotential τ p z₁ -
            laplaceNormalizerRatio τ p q z₁ *
              laplaceDisplacementPotential τ q z₁ := hc₁
      _ = laplaceDisplacementPotential τ p z₂ -
            laplaceNormalizerRatio τ p q z₂ *
              laplaceDisplacementPotential τ q z₂ := hdefect
      _ = τ ^ 2 * (deriv h (laplaceDisplacementPotential τ q z₂) *
          ‖laplaceDisplacementField τ q z₂‖ ^ 2) := hc₂.symm
      _ = τ ^ 2 * (deriv h (laplaceDisplacementPotential τ q z₁) *
          ‖laplaceDisplacementField τ q z₂‖ ^ 2) := by rw [hs]
  have hprod : τ ^ 2 * a * (n₁ - n₂) = 0 := by
    nlinarith [hsame]
  have hτsq : τ ^ 2 ≠ 0 := pow_ne_zero 2 hτ.ne'
  have hndiff : n₁ - n₂ ≠ 0 := sub_ne_zero.mpr hnorm
  have ha : a = 0 := by
    rcases mul_eq_zero.mp hprod with hleft | hright
    · exact (mul_eq_zero.mp hleft).resolve_left hτsq
    · exact (hndiff hright).elim
  have hz₁zero : laplaceDisplacementPotential τ p z₁ -
      laplaceNormalizerRatio τ p q z₁ *
        laplaceDisplacementPotential τ q z₁ = 0 := by
    rw [← hc₁]
    simp [a, ha]
  exact ⟨hz₁zero, hdefect ▸ hz₁zero⟩

end DriftingIdentifiability
