# Phase-1 protocol: neural pooled-rank development

**Status:** amended after the registered optimization-repair screen and frozen
before the standard development run  
**Date:** 2026-07-22  
**Predecessor:** `NeuralPooledRankPhase0Results.md`

## Question

Does persistent pooled-rank supervision survive neural amortization as the
feature dimension grows, and which part of the mechanism matters?

This is a development study. It may select architecture and training settings
for a later confirmation, but it cannot itself support a confirmatory claim.
The existing sealed 2D PSQT registry is not reused.

## Required decomposition

The comparison separates four possible bottlenecks:

```text
target statistic       exact pooled table versus KLL
generated ranks        large RSR population versus small population
transport realization  neural generator versus free particles
local geometry         pooled ranks alone versus paper-field hybrid
```

## Fresh registry

The registry contains four target families in every dimension `2, 4, 8, 16`:

1. balanced separated Gaussian mixture;
2. Gaussian mixture with a 5% rare component;
3. correlated heavy-tailed elliptical distribution; and
4. nonlinear curved/dependent distribution.

Every center, mixing matrix, rotation, sampler seed, training direction, and
held-out direction is generated and serialized before running any algorithm.
No generated target is rejected after seeing an endpoint.

Training and held-out directions are independent orthogonal blocks. Complete
blocks make the registered training frame tight while retaining held-out
projection tests that cannot influence optimization.

## Common preprocessing and generator

Each trial constructs one target pool, centers it by its empirical mean, and
divides all coordinates by one pooled RMS coordinate scale. The same
normalization is used by every arm. Mixture centers are transformed only for
post-training oracle diagnostics.

The neural generator is:

```text
latent dimension   max(4, target dimension)
architecture       Linear-64-Tanh-64-Tanh-Linear
optimizer          Adam
base learning rate 1e-3
normalization      none
dropout            none
```

The two standard initialization conditions are `concentrated` and `broad`.
All arms in a paired cell receive bitwise-identical initial parameters, latent
streams, target pool, evaluation latents, and evaluation reference.

### Registered optimizer repair

The first engineering smoke run exposed a gross update-frequency mismatch:
under the generator-example budget, ordinary arms received 64 Adam steps,
large RSR received 8, and small RSR received 32. The base `1e-3` rate left the
large arms near initialization. A paired development-only screen was therefore
run on the same four already-consumed smoke targets; its complete grid and
outputs are preserved under
`neural_pooled_rank_optimization_runs/20260722-154549-lr-screen/`.

Every arm was screened, so pooled-rank methods were not given an optimizer
advantage over the baselines. The following fixed multipliers are used for all
dimensions, target families, replications, and initializations in the standard
development run:

| Arm | Multiplier | Adam learning rate |
|---|---:|---:|
| paper neural | 16 | 0.016 |
| minibatch SW | 16 | 0.016 |
| exact-atlas large RSR | 32 | 0.032 |
| KLL-atlas large RSR | 32 | 0.032 |
| KLL-atlas small RSR | 16 | 0.016 |
| KLL-plus-paper hybrid | 32 | 0.032 |

Selection used the joint ED2/held-out-SW1 plateau rather than an independently
best setting for each target. Each selected point is below a tested higher
rate, except the hybrid whose screened response was nonmonotone and is retained
only as a diagnostic. This amendment is development tuning, not confirmation;
the four smoke targets cannot subsequently support an independent performance
claim. The runner records both multiplier and realized rate in every row.

## Development profiles

| Quantity | Smoke | Standard development |
|---|---:|---:|
| targets | 4, one per dimension | all 16 |
| initializations | concentrated | concentrated, broad |
| replications | 1 | 3 |
| unique target pool | 2,048 | 20,480 |
| generator-example budget | 2,048 | 20,480 |
| ordinary batch | 32 | 64 |
| large RSR population | 128 | 512 |
| RSR gradient microbatch | 32 | 64 |
| training directions | 64 | 64 |
| held-out directions | 128 | 128 |
| atlas knots | 64 | 128 |
| evaluation samples | 512 | 2,048 |
| free-particle sweeps | 30 | 100 |

The smoke target selection is fixed by registry order: balanced GMM in 2D,
rare GMM in 4D, correlated heavy-tail in 8D, and nonlinear dependence in 16D.

## Registered arms

### P0: neural paper baseline

- exact repository Algorithm-2 field;
- `tau = 1`, self-mask enabled;
- fixed stop-gradient field gain `0.15`;
- ordinary generated and target batch;
- stop-gradient regression field;
- one generator forward per update.

The update count is `generator_budget / ordinary_batch`.

### P1: ordinary minibatch sliced-Wasserstein

- the same ordinary target batch as P0;
- exact within-minibatch ranks;
- all 64 training directions;
- one generator forward per update.

The update count is the same as P0.

### P2: exact-atlas large-population RSR

- exact inverted-ECDF atlas from the full target pool;
- large current generated population;
- two generator evaluations per latent through Run and ReRun;
- no local paper field.

The update count is
`generator_budget / (2 * large_population)`.

### P3: KLL-atlas large-population RSR

Identical to P2 except that every target projection is summarized by Apache
DataSketches KLL 5.2.0 with `k = 128`. Every realized serialized KLL state is
saved in the run artifact.

### P4: KLL-atlas small-population RSR

Identical target atlas to P3, but the current generated population equals the
ordinary batch. Its update count is
`generator_budget / (2 * ordinary_batch)`. This isolates generated-rank
resolution from target-atlas quality.

### P5: scheduled KLL-plus-paper hybrid

- the same large-population KLL RSR step as P3;
- an Algorithm-2 field computed on the detached Run population and a paired
  target-pool slice;
- fixed-rank loss weight decreasing linearly from `1.0` to `0.25`;
- local-field surrogate weight increasing linearly from `0.25` to `1.0`;
- `tau = 1`, self-mask enabled, with the same base field gain `0.15`.

This is a development schedule, not a claim of optimal weighting. It has more
target accesses and kernel work than pure P3; those costs are reported.

### D0: exact free-particle ceiling

This diagnostic starts from the generator's large-population output and runs
the existing tight-frame reconstruction against the exact atlas. It is not a
neural generator, is not generator-compute matched, and is ineligible to win a
primary comparison. Its role is to measure the amortization gap.

## Budget accounting

The primary neural arms have equal generator-example evaluations. They do not
have equal primitive operation counts:

- P0 pays quadratic kernel work;
- P1 pays ordinary-batch projection and sorting work;
- P2-P5 pay a second generator pass and large-population sorting;
- P5 additionally pays local kernel work; and
- atlas construction is an offline target-side cost.

Record separately:

- unique target observations;
- target example accesses;
- unique latent samples;
- generator example evaluations;
- generator forward calls;
- projection dot products;
- sorting proxy `L * B * log2(B)`;
- paper kernel pairs;
- atlas bytes and KLL serialized bytes;
- wall time; and
- final model parameter count.

No arm may be described as cheaper from wall time alone.

## Metrics

Primary development metrics are:

- squared energy distance (ED2); and
- held-out-direction sliced Wasserstein-1.

Supporting diagnostics are:

- training-direction quantile RMSE;
- mixture mode coverage and mass error where oracle labels exist;
- output RMS spread;
- empirical covariance effective rank;
- divergence/nonfinite count; and
- free-particle-to-neural endpoint gap.

All projection quality claims must use held-out directions. Training-direction
loss may diagnose optimization but cannot establish generalization.

## Phase-1 invariants

Before a smoke run:

1. rerun the complete Phase-0 suite;
2. verify Torch and NumPy paper fields agree on random inputs;
3. verify matched positive/negative paper drift cancels without a mask;
4. verify every registered direction bank's frame diagnostics;
5. verify paired initial models and latent streams are identical;
6. verify KLL tables are monotone and serialized states replay exactly; and
7. verify every arm consumes exactly the registered generator-example budget.

## Interpretation gates

The smoke run is successful when it finishes without divergence and the
accounting/invariant layer passes. Endpoint quality from smoke may reveal a
gross implementation failure but must not select a winner.

The standard development run is promising only if:

- exact-atlas RSR improves over ordinary minibatch SW on held-out metrics;
- KLL retains a substantial fraction of the exact-atlas improvement;
- the large generated population improves over P4;
- gains are not confined to 2D;
- the free-particle ceiling shows finite headroom rather than an impossible
  atlas; and
- any hybrid gain survives its additional cost ledger.

A fresh protocol and fresh target registry are required after development for
confirmation.
