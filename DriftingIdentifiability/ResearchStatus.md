# Research status

## Executive answer

The original mathematical existence goal has been achieved: the project gives
explicit, nonvacuous conditions under which zero ideal population mean-shift
drift forces `p = q`, and the result is machine checked without new or
conditional Gaussian/RKHS axioms.

The practical modeling goal is not yet complete. The strongest concrete result
currently concerns finite atomic distributions on `ℝ` sharing a known support.
The main remaining mathematical problem is a useful quantitative lower bound
for its frame constant `c`; the main applied problem is connecting the ideal
population/probe statement to minibatch neural-network training on realistic
smooth, high-dimensional distributions.

## What has been accomplished

### 1. The finite-dimensional heart of Appendix C.1 is formalized

The project now rigorously implements the paper's heuristic chain:

1. Equation (29): represent `p` and `q` using finite coefficients `a` and `b`.
2. Equation (30): define the integral-induced interaction vectors `Uᵢⱼ`.
3. Equation (31): expand the population drift as `∑ᵢ∑ⱼ aᵢbⱼUᵢⱼ`.
4. Antisymmetry: prove `Uᵢⱼ = -Uⱼᵢ` and group the expansion into
   `∑_{i<j}(aᵢbⱼ-aⱼbᵢ)Uᵢⱼ`.
5. Nondegeneracy: a linearly independent strict-pair family forces every
   coefficient minor `aᵢbⱼ-aⱼbᵢ` to vanish.
6. Probability normalization: vanishing minors imply `a=b`.
7. Equal coefficients produce equal represented probability measures.

The grouped-minor algebra and the normalized-coefficient conclusion are proved
in Lean rather than axiomatized. No step assumes zero-drift uniqueness or an
equivalent injectivity conclusion.

### 2. The density heuristic was upgraded to genuine probability measures

A `ProbabilityDensityBasis` requires every basis function to be measurable,
nonnegative, integrable, and unit mass relative to a reference probability
measure. Mixtures are constructed using `Measure.withDensity`, and Lean proves
that they are probability measures.

The project also proves that the measure-level product integral against the two
mixture laws agrees with the density-weighted interaction integral used in the
Appendix C.1 calculation.

### 3. The normalized paper field is connected to the bilinear numerator

Under explicit integrability and nonzero-normalizer assumptions, equation (11)
turns zero normalized population mean-shift drift into zero unnormalized
interaction drift at every required probe. Equation (31) then connects this to
the finite coefficient-minor argument.

The promoted generic result is
`finitePopulationMeanShift_identifies`. It proves equality of the represented
probability measures for any finite probability-density basis whose actual
integral-induced interaction vectors satisfy the frame condition.

### 4. Exact, approximate, and energy statements are separated

`InteractionFrameBound U c` strengthens qualitative independence to

```text
c ∑_{i<j}|zᵢⱼ| ≤ ‖∑_{i<j} zᵢⱼ Uᵢⱼ‖,    c>0.
```

The following consequences are proved:

- a positive frame bound implies qualitative nondegeneracy;
- in finite dimension, qualitative nondegeneracy implies that some positive
  frame constant exists;
- coefficient error satisfies
  `‖a-b‖₁ ≤ (2B/c)‖V_probes‖`;
- vanishing probe drift gives coefficient convergence under uniform bounds;
- `card(StrictPair m) ≤ N · dim(E)` is necessary for nondegeneracy.

Separately, zero integrable population drift energy implies almost-everywhere
zero drift. Continuity and full topological support upgrade this to pointwise
zero and allow the exact theorem to be applied. A formal counterexample records
why full support cannot silently be omitted.

### 5. The frame hypothesis is discharged for an explicit general-`m` family

`EmpiricalFrameBound.lean` constructs, for every `m ≥ 2`:

- an injective support `z : Fin m → ℝ`;
- the uniform empirical probability reference on those points;
- point-mass functions proved to form a genuine probability-density basis;
- the unit-bandwidth Gaussian mean-shift kernel;
- `N = card(StrictPair m) = m(m-1)/2` integer probes.

For this basis, the actual equation (30) integral is computed exactly:

```text
Uᵢⱼ(n) = k(xₙ,zᵢ) k(xₙ,zⱼ) (zᵢ-zⱼ).
```

At the structured probes, the Gaussian interaction factors as

```text
Uᵢⱼ(n)
 = exp(-n²)
   [exp(-(zᵢ²+zⱼ²)/2)(zᵢ-zⱼ)]
   [exp(zᵢ+zⱼ)]ⁿ.
```

If the support points are distinct and all strict-pair sums `zᵢ+zⱼ` are
distinct, the geometric bases are distinct. An axiom-free Vandermonde argument
therefore proves that the actual integral-induced vectors are linearly
independent. This is not the synthetic Gaussian construction from the
conditional layer.

The resulting end-to-end theorem is `gaussianEmpiricalPoint_identifies`. Its
caller supplies the geometric support conditions, coefficients, and zero
population drift; it does not supply a frame bound or hidden injectivity
assumption. An explicit three-point support verifies that the geometric
condition is realizable, while the equally spaced four-point regression shows
that distinct points alone do not prevent pair-sum collisions.

The earlier `m=2` route is also fully packaged by
`empirical01Gaussian_identifies` for arbitrary positive Gaussian bandwidth.

### 6. One part of the stability constant is explicit

For the unit Gaussian, `0 < k(x,y) ≤ 1`. The concrete normalizers therefore lie
in `(0,1]`, and the product bound is `B=1`. The concrete stability theorem is

```text
‖a-b‖₁ ≤ (2/c) ‖V_probes‖.
```

The remaining unknown is the size of `c`, not the normalizer factor.

### 7. Trust and failure boundaries are enforced

- The promoted results use only the reviewed paper facts for equations (11)
  and (31) and basis-interaction antisymmetry.
- Conditional characteristic-kernel and synthetic Gaussian modules are
  excluded from the default proof route.
- Counterexamples cover collapsed bases, duplicated interactions,
  insufficient probe dimension, zero normalizers, missing support, and
  non-injective feature maps.
- The trust audit, warning-as-error build, conditional-module build, and
  promoted-theorem axiom audit pass.

## Strongest result and its exact scope

The strongest completely instantiated result currently says:

> Let `p` and `q` be two simplex-weighted probability measures on the same
> finite injective support in `ℝ`. Assume the strict-pair support sums are
> distinct. Use the unit Gaussian mean-shift kernel and one structured probe
> per strict pair. If the ideal normalized population drift is pointwise zero,
> then `p=q`.

This is an exact population theorem in data space. It does not presently claim:

- equality for arbitrary continuous distributions;
- equality of source laws from a non-injective feature representation;
- equivalence between Algorithm 2's finite bi-softmax estimator and the
  population field;
- a result for CFG or signed target laws;
- that gradient-based optimization reaches the zero-drift hypothesis;
- that the concrete conditions preserve practical model expressivity or
  generation quality.

The concrete atomic measures do not have full topological support. Therefore
the generic zero-energy-to-pointwise-zero corollary cannot be applied directly
to this concrete family. The exact theorem is currently driven by pointwise
`ZeroDrift`, even though its proof only consumes drift values at the selected
probes.

## Remaining objectives, in priority order

### Objective 1: obtain a useful explicit lower bound for `c`

This is the single remaining quantitative gap inside the concrete exact route.
Finite-dimensional compactness proves `c>0`, but the selected value is
noncomputable and may be extremely small.

Desired result:

- bound `c` using minimum support separation, minimum pair-sum separation,
  model size, bandwidth, and probe placement;
- determine how `c` scales as `m` grows;
- identify configurations where the theorem is exact but numerically useless;
- replace the existential constant with a computable singular-value or
  certified Vandermonde bound.

### Objective 2: align the hypothesis with the observable training loss

The current final theorem assumes pointwise zero drift, while the proof uses
only finitely many probes and practical training observes minibatch samples.

Desired results:

- state and prove the exact theorem directly from zero drift at the required
  probes;
- determine when sampled or expected drift loss controls those probe values;
- handle the fact that the atomic concrete law lacks full support;
- distinguish exact probe zero, almost-everywhere zero, small expected loss,
  and high-probability minibatch error.

### Objective 3: move from a proof-of-concept family to practical model classes

The present construction is atomic, one-dimensional, shares a fixed support,
and uses quadratically many probes. These conditions are mathematically clean
but not yet a realistic image-generator architecture.

Desired extensions:

- smooth or continuous density bases;
- higher-dimensional Euclidean/data-space constructions;
- fewer or adaptively selected probes;
- variable Gaussian bandwidth with quantitative bounds;
- the paper's practical Laplace-style kernel;
- conditions that retain adequate model expressivity.

### Objective 4: prove finite-sample guarantees for Algorithm 2

Desired results:

- concentration bounds between the minibatch bi-softmax field and the ideal
  population field;
- treatment of dependence caused by reusing generated samples as negatives;
- propagation of estimator error through the `2/c` stability estimate;
- sample complexity as a function of dimension, batch size, and conditioning.

### Objective 5: handle feature-space training correctly

The current data-space theorem gives equality of data-space measures. In
feature space, zero drift can generally identify only the pushforward laws.

Desired results:

- retain pushforward equality as the default conclusion;
- lift it to source-law equality only under an independently stated
  measure-determining or measurable-embedding condition;
- study approximate lifting when the learned feature map is not injective.

### Objective 6: treat CFG separately

CFG equation (16) can define a signed affine target rather than a probability
measure. It is outside the current probability-mixture theorem.

Desired result: develop a signed-measure or affine-density formulation with its
own normalization, identifiability, and stability analysis.

### Objective 7: evaluate whether the conditions are useful in practice

After obtaining computable conditioning criteria:

- measure the interaction matrix singular values on realistic features;
- test sensitivity to support geometry, basis size, bandwidth, and probes;
- compare population and minibatch drift numerically;
- evaluate whether enforcing the conditions harms expressivity, optimization,
  FID/IS, or compute requirements.

## What would complete the practical phase

The practical objective should be considered complete only when the project has
an implementable model/kernel/probe design, a useful certified or empirically
validated conditioning constant, a finite-sample bridge to the training
estimator, and evidence that the restrictions do not destroy model quality.

## Conditional research modules

`CharacteristicIdentifiability.lean` and `GaussianNondegeneracy.lean` remain
available only through `DriftingIdentifiability.Conditional`. They depend on
external Gaussian/RKHS axioms or a synthetic interaction construction and are
not accepted project solutions.
