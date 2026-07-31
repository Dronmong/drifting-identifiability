# Neural amortization of pooled-rank PSQT

**Status:** mathematics rechecked; Phases 0 and 1 completed; the original
large-RSR gate failed, and the conditioned transport successor has now passed
its consumed-registry development gate  
**Date checked:** 2026-07-22  
**Local empirical predecessor:** `PSQTAccumulatorConfirmatoryResults.md`  
**Phase-0 checkpoint:** `NeuralPooledRankPhase0Results.md`  
**Phase-1 checkpoint:** `NeuralPooledRankPhase1Results.md`  
**Successor:** `ConditionedTransportAmortizationResearch.md` and
`ConditionedTransportAmortizationResults.md`  
**Intended next boundary:** fresh preregistered synthetic confirmation before
any frozen-feature claim

## 1. Executive judgment

The proposed neural extension is mathematically coherent and worth testing.
The clean version is:

1. build a fixed target quantile atlas from pooled target projections;
2. draw a large effective population from a neural generator;
3. globally rank the generated projections without retaining an autograd graph;
4. rerun the same latent variables in microbatches and backpropagate toward the
   atlas-assigned targets; and
5. optionally combine this global rank loss with the paper's local drifting
   update.

This is not “putting a neural network inside KLL.” KLL remains a target-side
streaming summary. The neural network amortizes the transport: after training,
one forward pass replaces the repeated free-particle reconstruction used by
PSQT.

The strongest reason to try this route is the repository's sealed 2D result.
Pooled projected ranks, rather than averaged minibatch quantiles, accounted for
a large and robust improvement. A neural generator can potentially preserve
that global statistic while providing an out-of-sample sampler. The main risk
is no longer target estimation. It is the **amortization gap**: a shared neural
map may be unable or unwilling to realize the correction that free particles
can take independently.

The recommendation survives the mathematical recheck, with four corrections
to the informal proposal:

- the exact neural loss and the existing PSQT particle correction differ by a
  factor of the effective population size;
- KLL guarantees rank error, not value-space or transport error;
- for a fixed finite dataset and fixed directions, a finalized exact quantile
  table may be preferable to KLL; and
- finitely many feature projections identify only the tested projected
  marginals, not the source distribution in general.

## 2. What the repository has actually established

The sealed confirmatory experiment in
`psqt_accumulator_confirmatory_runs/20260722-005506-confirmatory/` used 64
fresh 2D targets from eight families. Relative to historical online PSQT, the
Apache KLL arm obtained:

- geometric-mean endpoint ED2 ratio `0.3370`, with target-bootstrap 95% CI
  `[0.3111, 0.3669]`;
- held-out SW1 ratio `0.5187`;
- improvement in all eight prespecified families; and
- zero divergence and complete recovery of the evaluated 5% and 10% rare
  modes.

The corresponding ED2 and held-out SW1 ratios against the registered
`tau = 1` paper arm were `0.0762` and `0.2891`. These are results for a
nonparametric 2D particle testbed under repository metrics. They are not FID,
IS, ImageNet, encoder-feature, or neural-generator results.

The experiment isolates a useful mechanism: pooling ranks across the whole
target stream avoids the bias introduced by averaging independently estimated
minibatch quantile functions. It does not establish that KLL itself is the
unique or optimal way to preserve those pooled ranks.

## 3. Exact mathematical object

Let:

- `G_theta(z, c)` be a conditional or unconditional generator;
- `E_a` be a frozen feature map for layer `a`;
- `h_j = E_a(G_theta(z_j, c))` lie in `R^(d_a)`;
- `u_(a,l)` be unit projection directions, `l = 1,...,L_a`;
- `s_(a,l,j) = <u_(a,l), h_j>`;
- `pi_(a,l)` sort the generated scalar projections increasingly;
- `r_i = (i - 1/2) / B` for an effective generated population of size `B`;
- `Qhat_(a,l,c)(r_i)` be the fixed target quantile supplied by an exact table,
  a KLL sketch, or a reservoir; and
- `w_a >= 0` be a feature-layer weight.

The proposed loss is the scaled empirical sliced 2-Wasserstein objective

```text
J_B(theta)
  = sum_a w_a * d_a / (2 L_a B)
      * sum_l sum_i
          (s_(a,l,pi_(a,l)(i)) - Qhat_(a,l,c)(r_i))^2.
```

For one layer, omitting `d/2` gives the ordinary Monte Carlo empirical
`SW_2^2`. The factor `d` is the repository's tight-frame scaling. It changes
gradient magnitude, not minimizers, and can be absorbed into the learning rate.

### 3.1 Gradient away from ties

Let `rho_(a,l)(j)` be the rank assigned to particle `j`. On any neighborhood
where projected values do not cross, the sorting permutations are constant.
Ordinary differentiation gives

```text
grad_(h_j) J_B
  = sum_a w_a * d_a / (L_a B)
      * sum_l
          (s_(a,l,j) - Qhat_(a,l,c)(r_(rho_(a,l)(j)))) u_(a,l).
```

Thus the negative feature gradient points from the current projection toward
the target projection at the same generated rank. At projected ties the loss
is nonsmooth; stable tie-breaking selects a valid piecewise-smooth branch.
This is the same practical situation as standard empirical sliced-Wasserstein
training and should be tested explicitly with duplicate samples.

### 3.2 Exact relation to the current PSQT correction

The current implementation uses

```text
c_j = d / L * sum_l (target_(l,rho_l(j)) - s_(l,j)) u_l
```

in `persistent_sliced_quantile_transport.py` and
`projected_quantile_accumulators.py`. For the averaged loss above,

```text
c_j = -B * grad_(h_j) J_B.
```

Equivalently, `c_j` is the exact negative gradient of the unaveraged
particle objective `B * J_B`. Therefore the particle and neural constructions
have the same directions, but their effective step sizes agree only after the
population-size normalization is handled. A neural implementation should use
the averaged loss and tune an ordinary optimizer learning rate; it should not
copy the particle step size without this conversion.

This identity was checked two ways:

- symbolically by differentiating each fixed-rank quadratic term; and
- numerically on a random `B=7`, `d=3`, `L=11` instance. Central finite
  differences agreed with the analytic gradient to `7.365e-11`, and the
  identity `c = -B grad J_B` agreed to `1.110e-16`.

### 3.3 Why the `d/L` backprojection is sensible

For directions uniformly distributed on the unit sphere,

```text
E[d * u u^T] = I.
```

For a finite unit-norm tight frame,

```text
(d/L) * sum_l u_l u_l^T = I.
```

Consequently, if all projected residuals arise from one common displacement
`v`, the repository backprojection reconstructs `v` exactly. This explains
the scaling. It does not imply that arbitrary independently rank-matched
residuals are jointly consistent with a single displacement or a single
multivariate target map.

If learned or reused directions cease to be a tight frame, there are two honest
choices:

1. retain the sliced loss and let autograd use its ordinary gradient; or
2. precondition the particle field with a regularized inverse frame operator.

The second changes the optimizer geometry and requires its own stability test.

## 4. Why Run-Sort-ReRun gives the right neural gradient

A large effective population is essential. A perfect target atlas does not
help a rare mode if an ordinary generated minibatch contains no sample at the
corresponding rank.

The exact two-pass procedure is:

1. **Run:** hold `theta` fixed, draw and save `B` latent variables, and compute
   generated features/projections without retaining activations;
2. **Sort:** globally rank every generated projection and assign each latent
   its target-atlas values;
3. **ReRun:** regenerate the same latent variables in microbatches with
   autograd enabled, accumulate losses normalized by the full `B`, and make
   exactly one optimizer update after every microbatch has contributed.

This produces the same mathematical gradient as a full-activation batch if:

- parameters and optimizer state are unchanged between the two passes;
- the same latent values and conditional labels are replayed;
- dropout, augmentation, and other stochastic state are exactly replayed or
  disabled;
- normalization layers do not change batch-dependent outputs between passes;
- rank assignments are treated as fixed targets during the rerun; and
- loss normalization uses the effective population `B`, not the microbatch
  size.

The repository already regression-tests the analogous one-dimensional
mechanism in `lbqcd.py`: full-batch and chunked RSR parameter updates agree to
very tight floating-point tolerances.

This is not ordinary gradient accumulation. Accumulating gradients from many
separately ranked small batches does not reproduce the ranks of their pooled
population. The global Sort step is the essential operation.

For exact replay, prefer GroupNorm or LayerNorm over training-mode BatchNorm.
If BatchNorm is retained, its statistics and update behavior must be frozen or
replayed consistently. Store explicit RNG states for dropout and stochastic
augmentation.

## 5. What KLL does and does not guarantee

The target side needs one scalar quantile summary for every fixed
`(class, feature layer, direction)` tuple. Apache KLL is a sensible bounded
streaming implementation, and the repository independently audited its
serialization, merge, monotonicity, support, stream-count, and empirical-rank
behavior.

For `k=128`, the pinned Apache implementation reports normalized rank error
approximately `0.02052`. The sealed stress audit observed maximum error
`0.01672`, p95 maximum error `0.01152`, and median maximum error `0.00651`.

The important qualification is:

> KLL controls error in rank space, not distance between returned values.

On the rank-error event, a schematic interior statement is

```text
Q(r - epsilon) <= Qhat(r) <= Q(r + epsilon),
```

with endpoints clamped and conventions adjusted for atoms. If the true
quantile function is `M`-Lipschitz on this interval, this implies

```text
abs(Qhat(r) - Q(r)) <= M * epsilon.
```

But quantile Lipschitzness requires genuine distributional regularity, such as
a density bounded away from zero on the region of interest. A continuous
density alone is insufficient. Across a support gap, a small rank error can
produce a large value error. This is precisely why a nominal 2% rank scale does
not certify faithful handling of a 1% rare component, even though individual
empirical tests may succeed.

If a target quantile error at a matched rank is `delta_(l,j)`, the induced
one-layer feature-gradient error satisfies

```text
norm(Delta grad_(h_j) J_B)
  <= d / (L B) * sum_l abs(delta_(l,j)).
```

Under a uniform value-error bound `delta`, this is at most `d*delta/B`; the
corresponding PSQT particle-field error is at most `d*delta`. A parameter-space
gradient bound additionally needs control of the generator/encoder Jacobian.
No such bound follows from KLL rank accuracy alone.

### 5.1 Exact table versus KLL

For a static finite training dataset, frozen encoder, and fixed directions, an
offline exact target table can be computed once and the raw projected samples
discarded. Its deployed state is only the table, not all observations. The
repository's `309,248`-byte exact-pooled diagnostic retained its stream history
during the online experiment; that number is not a lower bound on a finalized
exact atlas.

Therefore the neural experiment must include both:

- an exact finalized target atlas, which is the quality ceiling for fixed data;
  and
- a KLL atlas, which is relevant when targets arrive as a stream, sketches are
  merged across workers, augmentations define an ongoing target distribution,
  or rescanning all target features is undesirable.

The scientific mechanism should be called **persistent pooled-rank
supervision**. KLL is one useful accumulator, not the mechanism itself.

## 6. Identifiability and zero-set limits

The exact population sliced-Wasserstein distance over all directions is a
distributional metric under the usual moment assumptions. The implemented
atlas, however, has finitely many fixed directions. Zero finite-atlas loss says
only that those projected marginals match.

It does not generally imply:

- equality of feature distributions;
- equality of image/source distributions;
- equality under directions not in the atlas; or
- equality after a non-injective encoder.

Even an exact finite target table may be insufficient to determine a general
high-dimensional law. Independently approximated KLL rows can also be slightly
incompatible with any one joint distribution, so exact zero training loss may
be unattainable. Neither issue invalidates the objective, but both require
held-out projection and non-sliced evaluation.

This project’s formal identifiability results remain useful guard rails:

- finite probes require an independently certified measure-determining or
  finite-model condition;
- feature equality lifts to source equality only under an explicit embedding
  or measure-determining condition; and
- small finite-probe loss needs a quantitative frame/stability certificate to
  imply a stated distributional error.

Neural amortization does not inherit the repository's Laplace population
converse automatically. It is a new finite-projection feature-space training
procedure.

## 7. Proposed architecture

### 7.1 Target preprocessing

For each class and selected frozen feature layer:

1. fix normalization and preprocessing;
2. fix a registered direction bank before evaluation;
3. stream target features through exact, KLL, or reservoir accumulators;
4. query midpoint quantiles on the registered grid;
5. serialize the accumulator and finalized table with hashes; and
6. reserve a disjoint set of held-out directions for evaluation only.

Directions determine the sketch. A KLL sketch of one projection cannot answer
a newly invented direction without rescanning raw features. Adaptive-direction
methods therefore require either a large precomputed direction pool, retained
raw/reservoir features, or a fresh target pass.

### 7.2 Training step

For each optimizer update:

1. choose one class initially, so all `B` ranks are class-conditional;
2. sample a registered subset or orthogonal block of atlas directions;
3. perform the no-grad Run over `B_eff` latent variables;
4. sort generated projections and attach atlas targets;
5. ReRun the same latents in gradient microbatches;
6. accumulate the globally normalized rank loss;
7. optionally add a local paper-drift or reconstruction/perceptual term; and
8. update the generator once.

Do **not** persist a KLL sketch of generated projections across optimizer
updates. The generator distribution changes after every update, so such a
sketch becomes stale. Generated projections must be reranked from a current
effective population. An EMA or replay approximation is a separate biased
method and should not be the first implementation.

### 7.3 Hybrid with the paper field

The pooled-rank term is global: it protects mass, coverage, and tails. The
paper field is local and may improve geometry or finish details. The safest
initial hybrid is a registered weighted sum or alternating schedule:

```text
loss_total = loss_local + lambda(t) * loss_pooled_rank.
```

A plausible schedule emphasizes pooled ranks early, when missing modes and
global occupancy are the main danger, then increases the relative local term
later. This is a hypothesis, not a result. It must be compared with both pure
arms under matched generator evaluations and target observations.

## 8. Scaling analysis

Let `N` be target observations, `B` the effective generated population, `d`
the feature dimension, `L` the active directions, and `K` atlas knots.

Target preprocessing approximately costs:

```text
projection: O(N L d)
exact sorting: O(L N log N)
KLL updates: implementation-dependent near-streaming cost per scalar
deployed table: O(L K)
sketch state: O(L * sketch_size(k, N))
```

One RSR optimizer update approximately costs:

```text
generator + encoder evaluations: 2 B examples
projection/backprojection: O(B L d)
sorting: O(L B log B)
saved latents: O(B * latent_dimension)
saved projected/rank data: O(B L), unless direction-blocked
activation memory: O(microbatch), not O(B)
```

RSR removes the activation-memory barrier; it does not remove the compute or
rank-storage cost of a large population. Direction blocking can reduce peak
`B*L` storage, but rerunning once per direction block may increase generator
cost unless feature activations or outputs are cached. This tradeoff must be
measured, not assumed away.

High dimension creates a second problem: most random projections can be weakly
informative, while a small fixed bank can be gamed. The first experiments
should use orthogonal direction blocks and compare them with iid directions.
Adaptive or data-dependent directions are a later phase because they complicate
the immutable target atlas.

## 9. Literature cross-check and novelty boundary

The components have strong precedents:

- [Run-Sort-ReRun](https://proceedings.mlr.press/v139/lezama21a.html)
  already shows that global empirical sliced-Wasserstein ranks can be computed
  over large populations and backpropagated by replaying latent variables in
  microbatches. It reports effective populations of 16,000 and 4,000
  projections in 2,048-dimensional Inception features for some experiments.
- [Streaming Sliced Optimal Transport](https://arxiv.org/abs/2505.06835)
  already combines quantile approximation with sliced-Wasserstein computation
  and studies streaming gradient flows with approximation guarantees.
- [Sliced-Wasserstein Flows](https://proceedings.mlr.press/v97/liutkus19a.html)
  develops nonparametric projected-CDF transport and uses representation-space
  dimension reduction for image data.
- [Conditional Sliced-Wasserstein Flows](https://proceedings.mlr.press/v202/du23c.html)
  extends that flow viewpoint to conditional generation and introduces
  image-appropriate inductive biases.
- [Orthogonal Estimation of Wasserstein Distances](https://proceedings.mlr.press/v89/rowland19a.html)
  supports testing orthogonally coupled projection directions to reduce Monte
  Carlo variability.
- [Sliced Wasserstein Generative Models](https://arxiv.org/abs/1904.05408)
  and [Run-Sort-ReRun](https://proceedings.mlr.press/v139/lezama21a.html)
  establish direct neural training with sliced objectives.
- [Amortized Projection Optimization](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f02f1185b97518ab5bd7ebde466992d3-Abstract-Conference.html)
  uses “amortized” in a different sense: a network predicts informative
  projection directions. It should not be confused with amortizing our
  particle transport into a generator.
- [One-Step Generative Modeling via Wasserstein Gradient Flows](https://arxiv.org/abs/2605.11755)
  is a recent preprint-level example of constructing a particle flow and then
  compressing it into a one-step neural generator. It is conceptual support,
  not evidence for this particular KLL/PSQT design.
- [Apache DataSketches KLL documentation](https://datasketches.apache.org/docs/KLL/KLLAccuracyAndSize.html)
  confirms that KLL accuracy is specified in normalized-rank space and warns,
  through its bounds documentation, that value errors can be large across
  discontinuities or gaps.

The literature search did not locate the exact combination of:

```text
immutable target-only KLL projection atlas
+ current-generator global RSR ranks
+ PSQT tight-frame correction
+ optional local Drifting field
```

That may be an original integration and experimental contribution. It should
not yet be called a new mathematical transport principle, and novelty should
be rechecked in a formal paper search before publication.

## 10. Failure modes that must be attacked directly

1. **Amortization gap.** Free particles improve, but the neural generator
   cannot realize their independent motions.
2. **Generated-population under-resolution.** Small `B_eff` misses rare ranks
   even with a perfect target atlas.
3. **Sketch gap crossing.** KLL rank error maps across a low-density or empty
   target interval and produces a large value error.
4. **Finite-direction overfitting.** Training directions improve while held-out
   SW, ED, precision/recall, or FID does not.
5. **Encoder collision.** Feature laws match but perceptually important image
   distinctions disappear.
6. **Atlas incompatibility.** Approximate one-dimensional tables cannot all be
   jointly realized, causing oscillation or a nonzero loss floor.
7. **Replay mismatch.** Dropout, augmentation, BatchNorm, or a premature
   optimizer step changes rerun outputs.
8. **Bad scaling.** Copying the free-particle step size into an averaged neural
   loss makes updates off by a factor involving `B`, `d`, or `L`.
9. **Direction rigidity.** Fixed sketches prevent cheap adoption of new
   informative directions.
10. **Misleading efficiency claim.** RSR saves activation memory but performs
    an extra forward pass and large projection/sort work.

## 11. Implementation and experimental order

### Phase 0: algebraic PyTorch harness

Implement only a small loss module and tests:

1. exact target-table construction for fixed directions;
2. KLL target-table construction behind the same interface;
3. full-batch rank-matched sliced loss;
4. two-pass microbatch RSR loss; and
5. optional frame diagnostics.

Required tests:

- autograd versus central finite differences away from ties;
- full-activation versus RSR parameter gradients;
- invariance to microbatch partition;
- identity/free-particle model versus existing PSQT correction after applying
  the proved factor `c = -B grad J_B`;
- exact table versus no-compaction empirical quantiles;
- tie, atom, support-gap, 1% rare-mode, and 5% rare-mode cases; and
- deterministic replay with all stochastic layers used by the test model.

Do not start image generation until these checks pass.

### Phase 1: fresh neural synthetic registry

Use new, unconsumed target instances in dimensions `2, 4, 8, 16`. Begin with
small MLP generators and compare:

1. neural port of the paper field;
2. ordinary small-minibatch sliced-Wasserstein training;
3. exact-atlas RSR;
4. KLL-atlas RSR;
5. KLL atlas with a small generated population; and
6. a registered paper-plus-KLL hybrid.

The exact-atlas arm separates neural amortization from sketch error. The
small-generated-population arm tests whether large-population ranking is the
actual causal mechanism. Include a free-particle exact-atlas ceiling to
measure the amortization gap.

Primary metrics should include held-out-direction SW, energy distance, mode
coverage, rare-mass error, and target-balanced occupancy. Record wall time,
peak memory, generator example evaluations, projection dot products, sorting
work, and target observations. Select algorithms at the target-instance level,
not by treating optimizer seeds as independent targets.

### Phase 2: frozen real-feature distributions

Before pixels, train a generator directly in frozen real feature space. This
isolates whether the high-dimensional atlas and RSR mechanism work without
decoder/image confounding. Test multiple direction counts, orthogonal blocks,
`k`, `B_eff`, and exact versus KLL atlases. Evaluate with held-out directions
and non-sliced feature metrics.

### Phase 3: small image benchmark

Only after Phase 2 succeeds, connect `G_theta` to images and backpropagate
through a frozen feature encoder. Start with MNIST/Fashion-MNIST or CIFAR-10,
then use:

- a separate evaluation encoder;
- FID or an appropriate small-data correction;
- precision/recall or density/coverage metrics;
- visual quality and memorization checks; and
- matched training and inference cost ledgers.

The exact target atlas should remain an arm. KLL earns its place only if the
streaming/distributed/augmentation benefit outweighs its approximation error.

### Phase 4: integration with the paper model

Compare pure paper, pure pooled-rank, simultaneous hybrid, and alternating
hybrid schedules. Freeze the schedule before confirmation. A valid statement
of improvement requires the paper's own output metrics and comparable model,
data, and compute conditions—not only repository ED2/SW1.

## 12. Go/no-go gates

Proceed from Phase 0 only if:

- finite-difference and RSR gradient relative errors are below a registered
  tolerance such as `1e-5`;
- full and microbatched updates agree up to expected floating-point order;
- the identity model reproduces the PSQT correction after normalization; and
- duplicate/tie cases remain finite and deterministic.

Proceed from synthetic neural tests only if:

- exact-atlas RSR improves over small-batch SW and the paper neural baseline on
  held-out metrics;
- KLL retains a useful fraction of the exact-atlas gain;
- gains persist beyond 2D and across target families;
- rare-mode improvement is not bought by severe bridge mass or tail artifacts;
  and
- the amortization gap to free particles is understood quantitatively.

Proceed to images only if frozen-feature generation succeeds and finite-bank
overfitting is controlled. A failure of KLL with success of the exact atlas is
a sketch-resolution result, not a failure of pooled-rank neural training. A
failure of both neural arms with success of free particles is an amortization
or architecture failure.

## 13. Bottom line

The best first neural experiment is not “KLL everywhere.” It is a controlled
factorization:

```text
pooled target ranks       exact table versus KLL
large generated ranks     ordinary minibatch versus RSR
transport realization     free particles versus neural generator
local geometry            pooled-rank alone versus paper-field hybrid
```

That design directly tests which part of the successful 2D PSQT result
transfers. The math supports the loss and the RSR gradient. The literature
supports every main component. What remains genuinely unknown—and therefore
scientifically useful—is whether the repository's persistent pooled-rank
advantage survives neural amortization, dimension growth, and a fair comparison
with the original Drifting model.
