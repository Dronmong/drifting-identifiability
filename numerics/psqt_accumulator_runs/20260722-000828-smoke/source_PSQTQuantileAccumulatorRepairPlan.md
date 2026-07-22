# PSQT target-quantile accumulator repair experiment

**Status:** implemented and executed as a development screen; not frozen
confirmation  
**Analysis:** `PSQTQuantileAccumulatorDefectAnalysis.md`  
**Results:** `PSQTQuantileAccumulatorRepairResults.md`  
**Goal:** replace the biased mean-of-minibatch-quantiles statistic while
preserving online operation, bounded state, and the successful PSQT geometry

## 1. Questions and claim boundary

The experiment asks four ordered questions:

1. Does changing only the accumulator recover most of the exact-pooled gain?
2. Can a bounded-memory online accumulator eliminate the diagonal bridge?
3. Does that repair avoid regressions on connected and skewed targets?
4. What accuracy is purchased with each additional stored scalar, sort, and
   reconstruction sweep?

This is a reused-family development experiment.  It may select an
accumulator and operating point but cannot establish a fresh confirmatory
performance claim.  `exact-pooled` is an oracle-like finite-stream ceiling,
not a deployable bounded-memory arm.

## 2. Implementation separation

The historical `PersistentSlicedQuantileTransport` remains unchanged.  New
accumulators live in `projected_quantile_accumulators.py`; the paired runner
lives in `run_psqt_accumulator_repair.py`.

Each repaired trial has two explicit phases:

```text
target-statistic phase:
    observe exactly T * B target points and construct a projected table

geometric phase:
    reconstruct the same initial N-particle cloud against that fixed table
    for a registered number of tight-frame sweeps
```

This prevents the old accidental coupling between batch size, initialization
weight, target sample mass, and reconstruction work.  All arms use identical
target observations, directions, knots, initial particles, and geometric
iterations.

## 3. Accumulator arms

### A0. Historical online PSQT

Run the existing implementation with one initialization-sized batch prior and
three reconstruction steps after every minibatch.  This anchors the new runner
to the existing result.

### A1. Batch-mean table

Construct the same mean minibatch-quantile table, including the historical
one-batch initialization prior, but perform the registered geometric phase
only after accumulation.  Comparing A0 and A1 isolates scheduling from the
target statistic.

### A2. Batch median

Take the coordinate-wise median of completed minibatch quantile tables.  This
is a diagnostic robust aggregate, not the preferred scalable solution.  The
runner must charge the full stored table history.

### A3. Raw-point reservoirs

Use unbiased Algorithm-R reservoirs of 512, 1,024, and 2,048 raw 2D target
points.  Project and sort only when querying the final table.  Reservoirs
support arbitrary later directions, but their component-mass error must be
measured across seeds.

### A4. Per-direction randomized compaction sketches

For each fixed direction, insert every projected observation into an
independent weighted hierarchy of random compactors.  Query quantiles from the
retained weighted order statistics.  The development implementation uses
fixed-capacity KLL-style compactors and labels them as such; it does not claim
the optimal KLL space theorem for this simplified capacity schedule.

Test capacities 64 and 128.  The sketch must verify total retained weight,
monotone queries, deterministic replay under a fixed RNG seed, and exactness
when no compaction occurs.

### A5. Exact pooled ceiling

Store all target observations, then query their projected empirical
quantiles.  This supplies the finite-stream ceiling and quantifies approximation
loss.  Its linearly growing storage disqualifies it from deployment.

### B. Selected paper baseline

Run the previously selected exact paper estimator at `tau = 1` on the same
initial cloud and minibatches.  It is a contextual baseline, not part of the
accumulator factorization.

## 4. Quantile convention

Accumulator queries use inverted empirical-CDF/order-statistic quantiles.
This returns observed projected values and does not linearly interpolate
across an empty support gap.  The historical control retains its original
NumPy linear-quantile convention.  The convention difference is reported and
can be ablated separately if it becomes material.

## 5. Registered development profiles

### Smoke

```text
directions                 16
knots / particles          32
updates                    40
batch                      32
target observations        1,280
reconstruction sweeps      60
seeds                       1
targets                     first four plus rare-mode stress targets
```

### Screen

```text
directions                 32
knots / particles          64
updates                    300
batch                      64
target observations        19,200
reconstruction sweeps      100
reconstruction step        0.5
seeds                       5
initializations             concentrated, far
held-out directions         64 phase-shifted uniform lines
```

The screen contains the original nine targets plus diagonal two-mode mixtures
with minority weights 1%, 5%, and 10%.  The 1% target is explicitly a particle
resolution stress test: a 64-particle cloud has mass quantum `1/64`, so exact
1% mass matching is impossible.

## 6. Metrics

Primary quality metrics:

- squared energy distance (`ED2`);
- held-out sliced Wasserstein-1;
- held-out projected-quantile RMSE.

Failure-specific diagnostics:

- bridge occupancy between two known separated modes;
- excess bridge occupancy over an independent target reference;
- mode coverage and mode-mass L1 error;
- target-table RMSE against the exact pooled table;
- split-stream exact-table disagreement as a target-statistic noise floor;
- for sketches, retained weight and retained item count.

Cost ledger:

- target observations;
- projection dot products;
- target and reconstruction sort-work proxies;
- reconstruction sweeps;
- persistent state scalars;
- peak working scalars;
- wall time.

## 7. Development decision gates

A bounded-memory repair is promising only if it satisfies all of:

1. finite metrics and zero numerical divergence in every cell;
2. diagonal-family ED2 below both historical PSQT and selected paper;
3. aggregate ED2 and held-out SW1 below historical PSQT;
4. no family geometric-mean ED2 regression larger than 5% versus historical
   PSQT;
5. median excess bridge occupancy no larger than one particle;
6. 5% and 10% minority modes retained in a majority of paired trials;
7. explicit memory and wall-time costs substantially below exact pooling.

These are selection gates on reused targets, not hypothesis-test thresholds.

## 8. Stability and follow-up

The runner computes an even/odd-batch split-table disagreement.  A promoted
online implementation should eventually maintain two independent sketches or
confidence summaries and decline aggressive reconstruction when their
quantiles disagree beyond the observed sampling floor.

If a sketch passes the accumulator gates but two bridge particles remain,
the next experiment may add one geometric intervention at a time:

1. increase the fixed direction bank from 32 to 64 or 128;
2. add discrepancy-focused directions using a small raw reservoir;
3. apply sequential replayable ridge layers;
4. only then test a sparse Sinkhorn/barycentric refresh.

No geometric intervention should be selected on a corrupted batch-mean table.

## 9. Reproducibility requirements

Every run artifact must contain:

- command and master seed;
- Git commit and dirty status;
- exact profile and arm configuration;
- hashes and snapshots of the runner, accumulator, PSQT, target, analysis, and
  plan sources;
- raw row CSV, aggregate JSON, result Markdown, and representative visuals;
- a statement that the screen is development-only.

## 10. Execution record

The smoke and screen profiles were executed successfully.  The primary screen
artifact is:

```text
psqt_accumulator_runs/20260722-000222-screen
```

All executable invariants and all registered development gates passed.  See
`PSQTQuantileAccumulatorRepairResults.md` for the measured tradeoffs and the
remaining confirmation boundary.

