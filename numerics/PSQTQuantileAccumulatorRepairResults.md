# PSQT accumulator repair: development result

**Status:** exploratory development; not frozen confirmation  
**Analysis:** `PSQTQuantileAccumulatorDefectAnalysis.md`  
**Protocol:** `PSQTQuantileAccumulatorRepairPlan.md`  
**Primary artifact:** `psqt_accumulator_runs/20260722-000222-screen`

## What was implemented

The experiment preserves the existing paper estimator and historical online
PSQT as controls.  It adds a separate accumulator module with:

- the historical mean of minibatch empirical quantiles, factorized from
  reconstruction;
- the coordinate-wise median of all minibatch quantile tables;
- Algorithm-R reservoirs of 512, 1,024, and 2,048 raw 2D observations;
- fixed-capacity KLL-style weighted random-compactor sketches, independently
  maintained on every projection, with capacities 64 and 128;
- an exact finite-stream pooled ceiling;
- fixed-table tight-frame particle reconstruction with statistical sample
  mass separated from reconstruction iterations.

The KLL-style implementation preserves stream weight exactly, returns
observed projected values, and passed deterministic order/support and
no-compaction exactness invariants.  It uses a simple fixed capacity at every
level; it does **not** claim the optimal KLL level-capacity theorem.

## Screen design

The primary screen used:

```text
original targets                 9
rare diagonal stress targets     minority weights 1%, 5%, 10%
initializations                  concentrated, far
paired seeds                     5
particles / quantile knots       64
fixed directions                 32
held-out directions              64, phase shifted
target observations              300 * 64 = 19,200 per arm/trial
fixed-table reconstruction       100 sweeps, step size 0.5
historical reconstruction        3 sweeps after each of 300 batches
paper baseline                   exact repository estimator, tau = 1
```

Every paired arm received the same initial particles and target observations.
The bounded arm was selected using only the original nine targets; the added
rare-mode cases could not dominate selection.

## Main result

The development-selected bounded arm was `kll-style-k128`.

| Comparison on the original nine targets | ratio; lower is better |
|---|---:|
| KLL-style k128 ED2 / historical online PSQT | **0.4897** |
| KLL-style k128 held-out SW1 / historical PSQT | **0.6171** |
| KLL-style k128 ED2 / selected paper | **0.1850** |
| exact-pooled ED2 / historical PSQT | **0.4910** |
| exact-pooled ED2 / selected paper | **0.1854** |

The sketch and exact-pooled results are statistically indistinguishable at
this development resolution; the sketch being numerically 0.25% better is not
evidence that approximation improves the population method.  It is ordinary
finite-seed/evaluation variation.

The exact-pooled ratio `0.4910` independently reproduces the earlier in-memory
diagnosis (`about 0.492`).  The factorized batch-mean table achieved ED2 ratio
`1.0198` against historical online PSQT and had exactly the same median target
table RMSE (`0.03817`).  Therefore changing the reconstruction schedule did
not create the improvement: changing the target statistic did.

## Family robustness

| Family | KLL k128 / historical PSQT ED2 | KLL k128 / paper ED2 |
|---|---:|---:|
| Gaussian mixtures | **0.4823** | **0.1236** |
| ring | **0.6877** | **0.2091** |
| circles | **0.6579** | **0.2616** |
| moons | **0.6039** | **0.1654** |
| skew/heavy-tail | **0.8558** | **0.3025** |
| correlated Gaussian | **0.7762** | **0.4397** |
| diagonal dependence | **0.0796** | **0.1116** |
| rare diagonal stress | **0.1853** | **0.0691** |

All families improved over historical PSQT.  Most importantly, the only
original regression was reversed: diagonal-dependence ED2 fell by about 92%
relative to historical PSQT and by about 89% relative to the paper baseline.

The selected arm recovered the 5% and 10% minority component in every paired
trial.  It also assigned one particle to the 1% mode, but this should be read
against the representation floor: one of 64 particles is 1.5625%, so the
model cannot express 1% mass exactly.

## Bridge diagnosis after repair

For the original equal diagonal mixture, median excess bridge occupancy was:

| Arm | median excess bridge mass |
|---|---:|
| historical online PSQT | 0.1250 (8 particles) |
| paper `tau=1` | 0.0469 (3 particles) |
| KLL-style k128 | 0.0156 (1 particle) |
| exact pooled | 0.0156 (1 particle) |
| reservoir 1,024 | 0.0000 median |

Individual KLL/exact visualizations can still contain one or two off-mode
particles.  That is now a secondary finite-direction/reconstruction residual,
not the eight-particle statistical bridge.  The saved diagonal visualization
shows the distinction directly.

## Accuracy and cost tradeoff

Medians below are implementation-specific wall times on this machine.  Sort
and storage ledgers are the portable comparisons.

| Arm | ED2 / historical | ED2 / paper | table RMSE | wall s | persistent scalars |
|---|---:|---:|---:|---:|---:|
| historical online PSQT | 1.0000 | 0.3777 | 0.03817 | 0.106 | 2,305 |
| batch mean, fixed table | 1.0198 | 0.3852 | 0.03817 | 0.055 | 2,305 |
| batch median | 0.5516 | 0.2083 | 0.02080 | 0.067 | 614,656 |
| reservoir 512 | 0.8394 | 0.3171 | 0.03969 | **0.028** | **1,282** |
| reservoir 1,024 | 0.6768 | 0.2556 | 0.02538 | **0.028** | 2,306 |
| reservoir 2,048 | 0.5536 | 0.2091 | 0.01693 | **0.029** | 4,354 |
| KLL-style k64 | 0.5075 | 0.1917 | 0.02944 | 0.197 | 8,512 |
| KLL-style k128 | **0.4897** | **0.1850** | **0.01503** | 0.186 | 9,536 |
| exact pooled ceiling | 0.4910 | 0.1854 | 0 | **0.027** | 38,656 and growing |

Two Pareto choices emerge:

1. **Quality arm:** KLL-style k128 reaches the exact-pooled quality ceiling
   with bounded state, but is about 1.75 times slower and 4.1 times larger than
   historical PSQT in this unoptimized Python implementation.
2. **Efficiency arm:** reservoir 1,024 uses essentially the same persistent
   scalar count as historical PSQT, is about 3.8 times faster in this runner,
   and still reduces ED2 by about 32%.  Reservoir 2,048 closes more of the
   quality gap at modest additional storage.

The exact-pooled arm appears fast only because NumPy performs one compiled
sort after all samples have been retained.  Its state grows linearly with the
stream, so it is a ceiling rather than a scalable winner.  Batch median is
also disqualified by its full history storage.

## Registered gates

All development gates passed for KLL-style k128:

- finite output and no divergence;
- diagonal ED2 below historical PSQT and paper;
- aggregate ED2 and held-out SW1 below historical PSQT;
- no family regression larger than 5%;
- median excess bridge occupancy at most one particle;
- majority recovery of 5% and 10% modes;
- bounded state below exact pooling.

## Honest conclusion and next decision

The experiment validates the causal diagnosis and produces the strongest 2D
PSQT development result so far.  The improvement is not from extra geometric
optimization: it comes from retaining the target distribution's pooled rank
structure instead of learning the Wasserstein barycenter of sparse
minibatches.

It does not yet establish a new general result because the target families,
accumulator capacities, and gates were all examined during development.  The
next legitimate step is to freeze either:

- KLL-style k128 as the quality candidate; or
- reservoir 1,024/2,048 as the lower-cost candidate,

then evaluate both on a fresh, sealed 2D target registry with target-level
paired uncertainty.  If the KLL quality arm is promoted, its fixed-capacity
Python compactor should first be replaced or cross-checked against a standard
KLL implementation and optimized so the measured overhead is not mistaken
for an inherent algorithmic cost.

