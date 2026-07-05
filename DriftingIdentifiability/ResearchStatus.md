# Research status

## Verified paper-native result

- [x] The Appendix C.1 grouping and coefficient-minor argument is proved
  without project axioms.
- [x] A finite basis is now required to consist of measurable, nonnegative,
  integrable unit-mass densities.
- [x] Finite mixtures are genuine `Measure`s constructed with `withDensity`,
  and Lean proves they are probability measures.
- [x] The measure interaction integral is proved equal to the paper's
  density-weighted interaction integral.
- [x] Nonzero normalizers and equation (11) connect zero normalized population
  mean-shift drift to zero interaction numerator at the probes.
- [x] `finitePopulationMeanShift_identifies` proves equality of the represented
  probability measures.
- [x] Zero ideal population energy also suffices under integrability,
  continuity, and full topological support.
- [x] `InteractionFrameBound` supplies a quantitative, independently testable
  conditioning constant and implies the paper's qualitative nondegeneracy.
- [x] Coefficient error is bounded by normalized probe-drift error with the
  explicit factor `2B/c`; vanishing probe drift therefore gives coefficient
  convergence under uniform bounds.
- [x] A necessary probe-dimension bound is formalized.
- [x] The finite family is registered as a `CandidateSpec` with a formal
  distinct-pair legitimacy proof.

## Honest scope

The promoted theorem concerns the ideal normalized population mean-shift field
in data space, without CFG. It does not claim that a finite minibatch
bi-softmax estimator is equal to the population field. Feature-space matching
proves only pushforward equality unless the feature is a measurable embedding.

## Conditional research modules

`CharacteristicIdentifiability.lean` and `GaussianNondegeneracy.lean` are
available only through `DriftingIdentifiability.Conditional`. They depend on
external Gaussian/RKHS axioms or a synthetic interaction construction and are
not accepted project solutions. Their assumptions are restricted to suitable
inner-product/Borel/finite-dimensional settings.

## Exact remaining gap

The finite population theorem itself is complete. Its one unresolved
mathematical hypothesis is the positive interaction frame bound for the
paper's **actual integral-induced interaction vectors**:

```text
c · ∑_{i<j} |zᵢⱼ| ≤ ‖∑_{i<j} zᵢⱼ Uᵢⱼ‖    for some c > 0,
```

where

```text
Uᵢⱼ(n) = ∬ φᵢ(y⁺) φⱼ(y⁻)
              K(xₙ,y⁺,y⁻) dμ(y⁻) dμ(y⁺).
```

The project proves everything that follows from this bound: qualitative
nondegeneracy, exact measure identifiability, the stability estimate
`‖a-b‖₁ ≤ (2B/c)‖V_probes‖`, and coefficient convergence. What is not yet
proved is that a practical choice of the paper's kernel, probability-density
basis, and probes produces such a `c`, or that the resulting `c` is not so
small that the guarantee becomes numerically useless.

The synthetic Gaussian Gram construction does not close this gap: it proves
nondegeneracy for an artificially constructed anti-symmetric interaction
family, but that family has not been shown equal to the integral-induced
vectors above. The characteristic-kernel route also does not close it because
its embedding-injectivity premise is conditional and substantively equivalent
to the desired conclusion for that field.

### Closed for `m = 2` with the actual vectors (`EmpiricalFrameBound.lean`)

For a **two-point empirical reference** `μ = ½(δ_{z₀}+δ_{z₁})` and the mean-shift
interaction kernel, the *actual* double integral above is computed in closed form
(`basisInteraction_empirical2`):

```text
Uᵢⱼ(n) = ¼ (φᵢ(z₀)φⱼ(z₁) − φᵢ(z₁)φⱼ(z₀)) · k(xₙ,z₀) k(xₙ,z₁) · (z₀ − z₁).
```

With a Gaussian kernel (`k > 0` by `Real.exp_pos`), distinct points, and a basis
with nonzero value minor, the single strict-pair vector `U₀₁` is nonzero, so
`empiricalInteractionFrameBound` establishes a **positive frame bound with an
explicit constant** `c = ‖U₀₁‖` for the actual induced vectors — no synthetic
substitute and no external axiom (`#print axioms` = Mathlib foundations only).
This carries out steps 1–3 below for `m = 2`.

### What remains

1. ✅/`m=2`: choose an implementable density basis, kernel, and probe scheme —
   done (two-point empirical reference, Gaussian kernel).
2. ✅/`m=2`: compute the integral-induced interaction matrix — done in closed
   form.
3. ✅/`m=2`: prove the smallest frame value is positive — done, `c = ‖U₀₁‖`.
4. **Open:** general `m ≥ 3` (where `StrictPair` has more than one element and
   genuine linear independence of the `Uᵢⱼ` is required, not just one nonzero
   vector), and whether `c` remains acceptably bounded as model size and
   dimension grow.

### The general-`m` gap is now exactly *linear independence*

`interactionFrameBound_of_linearIndependent` (axiom-free, general `m`) proves the
converse of `interactionFrameBound_linearIndependent`: a linearly independent
induced family always admits *some* positive frame bound (via
`LinearMap.exists_antilipschitzWith` on the synthesis map). Hence for the actual
induced vectors the **frame-bound hypothesis is equivalent to plain qualitative
nondegeneracy**, and the remaining general-`m` obligation is precisely:

```text
LinearIndependent ℝ (fun (i<j) ↦ Uᵢⱼ),   Uᵢⱼ(n) = k(xₙ,zᵢ)k(xₙ,zⱼ)(zᵢ−zⱼ).
```

For `E = ℝ` this reduces (the `zᵢ−zⱼ` are nonzero scalars) to independence of the
product-kernel probe profiles `{(k(xₙ,zᵢ)k(xₙ,zⱼ))ₙ}_{i<j}` — a product-kernel
strict-positive-definiteness / distinct-pairwise-sum statement. Proving it stays
axiom-free/promoted requires genuine analysis (Gaussian product independence);
importing it would place the general-`m` result in the conditional layer.

The `m = 2` closure uses the actual `basisInteraction`. Separate later
extensions—not gaps in the present population theorem—are finite-sample bounds
for Algorithm 2, learned-feature/source-law identifiability, CFG via signed
measures, and generalization beyond finite density families.
