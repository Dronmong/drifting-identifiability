# Frozen protocol: PSQT accumulator confirmation

**Protocol status:** freeze-ready; the sealed registry and source-hash manifest
are created only after the disposable smoke run passes  
**Roadmap:** `PSQTAccumulatorConfirmatoryRoadmap.md`  
**KLL audit:** `kll_audit_runs/20260722-004356-k128`

## Primary question

On fresh prespecified two-dimensional target instances, do pooled-rank target
statistics improve historical minibatch-quantile PSQT and outperform the exact
repository implementation of the paper estimator?

## Environment

```text
Python             3.12
NumPy              resolved and recorded by uv
SciPy              resolved and recorded by uv
Matplotlib         resolved and recorded by uv
Apache DataSketches 5.2.0
```

Python 3.12 is required because DataSketches 5.2.0 has no Windows wheel for
the repository's default Python 3.14.  The exact invocation is:

```text
uv run --python 3.12 --with numpy --with scipy --with matplotlib \
  --with datasketches==5.2.0 \
  python numerics/run_psqt_accumulator_confirmatory.py ...
```

Apache's Python KLL binding does not expose its internal compaction RNG seed.
This is a known reproducibility limitation discovered before registry
generation.  Every trained projection sketch is serialized in the run
artifact, which makes its table and reconstructed output exactly replayable.
All target, initialization, reservoir, bootstrap, and visualization randomness
is explicitly seeded and recorded.

## Fixed algorithms

### Paper baseline

- exact repository `drift_paper` implementation;
- `tau = 1`;
- `eta = 0.15`;
- self mask enabled;
- 300 updates.

### Historical PSQT

- 32 fixed uniform unoriented directions;
- 64 midpoint quantile knots and 64 particles;
- one initialization-sized batch prior;
- linear empirical minibatch quantiles;
- three reconstruction sweeps per minibatch;
- reconstruction step size 0.5;
- 300 updates.

### Quality arm

- Apache DataSketches `kll_floats_sketch` version 5.2.0;
- independent sketch on each of the 32 fixed directions;
- `k = 128`;
- 64 midpoint quantile queries with `inclusive=True`, matching an inverted
  empirical CDF when no compaction occurs;
- all 19,200 target observations inserted;
- 100 fixed-table reconstruction sweeps at step size 0.5.

### Efficiency arm

- Algorithm-R reservoir of 1,024 raw 2D target observations;
- 32 fixed directions and 64 inverted-ECDF quantiles;
- all 19,200 observations eligible for the reservoir;
- 100 fixed-table reconstruction sweeps at step size 0.5.

### Diagnostic ceiling

`exact-pooled-ceiling` stores all 19,200 target observations.  It is excluded
from primary inference and promotion.

## Sealed registry

The sealed registry is generated once by
`generate_psqt_accumulator_registry.py --kind sealed`.  It contains eight
independent target instances in each of eight families:

1. randomly rotated/translated unequal Gaussian mixtures;
2. disconnected non-Gaussian ellipse mixtures;
3. rotated 2%, 5%, and 10% rare-mode mixtures;
4. correlated Gaussian, Student-t, and Laplace unimodal laws;
5. perturbed rings, spirals, and arcs;
6. unequal multiple-curve targets;
7. rotated skewed heavy-tailed targets;
8. off-axis binary, checkerboard, and nonlinear dependence traps.

The registry generator imports no candidate algorithm, refuses overwrite, and
writes a SHA-256 sidecar.  The freeze manifest binds that hash to all source
files before execution.

## Paired trial design

```text
target instances                 64
initializations                  concentrated, far
independent target streams       5 per initialization
target observations              19,200 per arm/trial
target minibatch                 64
particles                        64
evaluation reference             4,096 per target, shared across paired arms
held-out directions              64 phase-shifted uniform lines
```

The same target stream is reused across the two initializations for a given
target/stream index.  Initial particles differ by initialization but are
paired across algorithms.  Evaluation references are generated once per
target and are never training observations.

## Metrics

Primary:

- squared energy distance (ED2).

Secondary quality:

- held-out sliced Wasserstein-1;
- held-out projected-quantile RMSE;
- mode coverage and mass L1 on synthetic mixtures;
- minority particle mass and recovery;
- excess bridge occupancy on prespecified two-mode separated targets;
- divergence count.

Costs:

- wall time after import/warm-up;
- kernel pairs;
- projection/backprojection dot products;
- sort-work proxy where observable;
- reconstruction sweeps;
- persistent-state and peak-working bytes;
- KLL retained items and serialized bytes.

## Statistical unit and analysis

The independent unit is a target instance.  For every target and arm, take the
median metric over the ten initialization/stream trials.  Primary effects are
geometric means of target-level ratios.  A 95% paired bootstrap interval is
computed by resampling target instances within each family, using a fixed
recorded bootstrap seed and 5,000 draws.

The ordered primary comparisons are:

1. KLL ED2 versus historical PSQT;
2. KLL ED2 versus paper;
3. Reservoir ED2 versus historical PSQT;
4. Reservoir ED2 versus paper.

Held-out SW1 cannot replace a failed ED2 comparison.

Divergent trials are retained and assigned a registered ED2/SW1 penalty of
`100 * target.scale`; they also count as automatic losses.  Raw divergence
counts are reported.

## Promotion gates

### KLL quality arm

All must hold:

1. ED2 ratio versus historical PSQT below 0.80 and 95% upper bound below 1;
2. ED2 ratio versus paper below 0.80 and upper bound below 1;
3. held-out SW1 ratio below 1 versus both baselines;
4. at least 70% of target instances won against each baseline;
5. no family ED2 ratio above 1.10 versus historical PSQT;
6. zero divergence;
7. at least 90% trial-level recovery of 5% and 10% rare modes;
8. median excess bridge occupancy at most one particle (`1/64`).

### Reservoir efficiency arm

All must hold:

1. ED2 ratio versus historical PSQT below 0.90 and upper bound below 1;
2. ED2 ratio versus paper below 0.90 and upper bound below 1;
3. held-out SW1 no more than 1.05 versus historical PSQT;
4. at least 60% target wins against each baseline;
5. persistent state no more than 1.10 times historical PSQT;
6. median wall time below historical PSQT;
7. zero divergence;
8. rare-mode recovery no worse than historical PSQT.

## Stop rules

After `psqt_accumulator_confirmatory_freeze.json` is written:

- any source or registry hash mismatch aborts execution;
- no algorithm, target, metric, threshold, or analysis may change;
- a correctness bug contaminates the registry and requires a newly seeded
  registry after the repair is logged;
- partial result inspection cannot alter or stop the run;
- no per-target bandwidth, sketch capacity, direction, or reconstruction
  selection is allowed;
- the final result is written whether gates pass or fail.

