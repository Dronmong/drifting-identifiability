# PSQT accumulator repair: confirmatory promotion roadmap

**Status:** proposed next phase; implementation and confirmation not yet run  
**Predecessor analysis:** `PSQTQuantileAccumulatorDefectAnalysis.md`  
**Development protocol:** `PSQTQuantileAccumulatorRepairPlan.md`  
**Development result:** `PSQTQuantileAccumulatorRepairResults.md`

## 1. Objective

Determine whether repairing PSQT's target-quantile accumulator produces a
general, reproducible improvement over:

1. historical online PSQT, which averages minibatch quantile functions; and
2. the repository's exact implementation of the paper estimator at the
   previously selected `tau = 1`.

The development screen found that a bounded KLL-style accumulator nearly
matched exact pooling, while a raw reservoir supplied a cheaper Pareto point.
Those target families, capacities, metrics, and visualizations have already
influenced algorithm selection.  They cannot be reused as confirmatory
evidence.

The next phase must therefore separate:

```text
engineering validation -> protocol freeze -> sealed target generation
-> one confirmatory execution -> fixed statistical analysis
```

## 2. Claim boundary

If the protocol succeeds, the supported claim is limited to:

> On a prespecified fresh collection of low-dimensional particle-generation
> problems, persistent pooled-rank target statistics improve PSQT relative to
> minibatch-quantile averaging and outperform the tested paper estimator under
> the declared sample and particle budgets.

The experiment will not establish:

- superiority on ImageNet, encoder features, or neural generators;
- superiority over every bandwidth or per-target paper oracle;
- an asymptotic convergence theorem;
- optimality of KLL, reservoirs, sliced transport, or the direction bank;
- performance in dimensions beyond those explicitly tested;
- lower compute cost for the KLL quality arm unless the cost gate passes.

## 3. Phase A: harden the quality accumulator

This phase is engineering validation, not another performance search.  It must
finish before generating the confirmatory registry.

### A1. Cross-check against a standard KLL implementation

The development class `KLLStyleProjectedAccumulator` uses the KLL randomized
compaction mechanism with one fixed capacity at every level.  It deliberately
does not implement or claim the optimal level-dependent KLL schedule.

Add a cross-check harness that compares it with a maintained standard KLL
implementation, preferably Apache DataSketches or another independently
reviewed implementation.  For each stream, compare:

- quantiles at all 64 midpoint probabilities;
- empirical rank error of every returned quantile;
- retained item count and serialized state size;
- deterministic replay under fixed randomness, where supported;
- merge of stream partitions versus a single sequential stream;
- results under different minibatch partitions of the same ordered stream.

Test streams must include:

- standard Gaussian and Student-t;
- skewed/log-normal;
- two narrow separated modes;
- unequal mixtures with 1%, 2%, 5%, and 10% minority mass;
- repeated/discrete values and exact support gaps;
- monotone, shuffled, and adversarially ordered observations.

### A2. Required sketch invariants

The promoted implementation must verify:

1. all input observations have positive represented weight;
2. total retained weight equals total observations per projection;
3. returned quantiles are nondecreasing;
4. every returned value belongs to the observed projected support;
5. no-compaction output equals the selected empirical quantile convention;
6. state size remains bounded as stream length grows;
7. fixed-seed replay is deterministic;
8. merging disjoint summaries stays within the registered rank-error budget.

### A3. Decide the promoted sketch

Choose exactly one of:

- a standard compiled KLL implementation wrapped behind the repository
  accumulator interface; or
- the local implementation upgraded to the standard KLL capacity schedule and
  independently cross-checked.

Do not choose between implementations using endpoint ED2 on the development
or confirmatory target registries.  Choose using correctness, rank error,
bounded state, portability, and implementation cost.

### A4. Freeze capacity and direction behavior

The development quality point was:

```text
directions             32 fixed uniform unoriented lines
quantile knots         64 midpoint probabilities
sketch capacity        128
particles              64
reconstruction         100 fixed-table sweeps
step size              0.5
```

Preserve these values unless the standard implementation uses a materially
different capacity definition.  Any capacity translation must be justified by
retained-state or rank-error equivalence, not endpoint performance.

Adaptive directions are out of scope for this confirmation.  Each sketch is
tied to one prespecified fixed direction.

### A5. Optimize without changing mathematics

Allowed optimizations:

- vectorize projection updates;
- batch inserts into compiled sketches;
- avoid repeated Python allocation;
- query tables only at the registered refresh/finalization points;
- serialize compact state efficiently;
- share immutable direction and probability arrays.

Forbidden optimizations:

- changing capacity based on observed target difficulty;
- increasing directions for failing targets;
- choosing reconstruction steps using endpoint quality;
- changing the target statistic or interpolation convention;
- adding a paper-drift, Sinkhorn, or ridge-map finisher.

### A6. Engineering acceptance

Phase A is complete only when:

- the new accumulator and cross-check invariants pass;
- rank error is recorded for every stress stream;
- no silent interpolation across empty gaps occurs;
- source and dependency versions are pinned;
- its persistent-state and working-memory ledgers are implemented;
- the existing accumulator smoke test remains green.

If the promoted sketch cannot meet these requirements, the quality arm should
be labeled `bounded-rank-compactor` rather than KLL.  Do not preserve the KLL
name by weakening the audit.

## 4. Phase B: freeze the candidate arms

Exactly four primary algorithms enter confirmation.

### B0. Paper baseline

```text
name                    paper-tau1
estimator               exact repository Algorithm-2 port
temperature             1.0
particle step size      0.15
self mask               true
updates                 300
```

This is the bandwidth selected before the confirmatory registry exists.  Do
not add per-target bandwidth selection to the primary comparison.

### B1. Historical PSQT control

```text
name                    historical-online-psqt
directions              32
knots / particles       64
prior                   one initialization-sized batch
target statistic        running mean of batch quantiles
reconstruction          3 sweeps per minibatch
step size               0.5
updates                 300
```

This establishes that the accumulator repair, rather than a changed registry,
explains any improvement.

### B2. Quality candidate

```text
name                    pooled-rank-kll128
directions              32
knots / particles       64
target statistic        promoted per-direction KLL summaries
capacity                frozen Phase-A translation of development k128
target observations     19,200
reconstruction          100 fixed-table sweeps
step size               0.5
```

### B3. Efficiency candidate

```text
name                    pooled-rank-reservoir1024
directions              32
knots / particles       64
target statistic        Algorithm-R raw-point reservoir
reservoir capacity      1,024 2D points
target observations     19,200
reconstruction          100 fixed-table sweeps
step size               0.5
```

Reservoir 1,024 is preferred over 2,048 for the primary efficiency claim
because its development persistent-state count was essentially equal to
historical PSQT.  Reservoir 2,048 may be included as a labeled secondary
Pareto diagnostic if it is registered before target generation, but it cannot
replace Reservoir 1,024 after results are seen.

### B4. Non-primary ceiling

An `exact-pooled-ceiling` may be run as a diagnostic.  It stores every target
observation and is ineligible for selection, primary inference, or a
bounded-memory claim.

## 5. Phase C: construct a fresh sealed target registry

### C1. Deterministic generation before evaluation

Implement a registry generator with its own master seed.  It must write every
target parameter to a JSON file before any algorithm runs.  Hash the registry
and copy it into every artifact.

The generator may use prespecified random parameter draws, but no draw may be
rejected after inspecting algorithm output.  Reject only parameters violating
explicit construction constraints, and log all rejections and reasons.

### C2. Unit of inference

The independent unit is the **target instance**, not the optimizer seed.
Generate multiple target instances per family.  Initialization and stream
seeds are paired repeated measurements within a target.

Recommended minimum:

```text
families                         8
fresh target instances/family   8
initializations                 concentrated, far
paired streams/initialization   5
total target instances          64
paired trials/arm               640
```

If runtime requires a smaller registry, reduce paired streams before reducing
the number of independent target instances.

### C3. Fresh target families

All numeric ranges must be fixed in the registry generator.

#### F1. Random Gaussian mixtures

- components: uniformly selected from 2 through 7;
- random rotation and translation;
- unequal Dirichlet-distributed weights with a registered concentration;
- component scales spanning narrow to moderately overlapping;
- accepted minimum and maximum pair separation recorded.

#### F2. Disconnected non-Gaussian mixtures

- combinations of compact uniform ellipses, short arcs, or Laplace clusters;
- random component orientation;
- no exact reuse of the development diagonal construction.

#### F3. Rare-mode mixtures

- minority mass selected from `{0.02, 0.05, 0.10}`;
- random mode direction and translation;
- separation and noise drawn from registered ranges;
- 1% excluded from the primary family because 64 particles cannot represent
  1% mass exactly; it may remain a labeled resolution stress test.

#### F4. Correlated unimodal targets

- eigenvalue ratio and rotation drawn randomly;
- Gaussian and non-Gaussian radial laws;
- covariance parameters not reused from development.

#### F5. Curved connected supports

- arcs, spirals, warped moons, and perturbed rings;
- randomized curvature, width, phase, and rotation;
- at least one topology-preserving connected family.

#### F6. Multiple connected components

- nested or separated curves with unequal component mass;
- randomized gap widths;
- not restricted to circular geometry.

#### F7. Skewed and heavy-tailed targets

- skew and tail parameters from bounded registered ranges;
- robust reference sample sizes large enough for stable ED2/SW1 evaluation.

#### F8. Dependence traps

- coordinate marginals deliberately uninformative;
- random off-axis orientation;
- checkerboard, diagonal/anti-diagonal, or nonlinear dependence;
- parameters constructed without evaluating the candidate algorithms.

### C4. Registry validation independent of candidates

Before opening algorithm results, verify:

- all samples are finite;
- target scale and separation constraints hold;
- reference ED2 and held-out SW1 are stable when reference size doubles;
- rare components appear with expected binomial frequency;
- no target duplicates a development instance;
- fixed training directions and held-out directions do not overlap;
- target plots and parameters are saved without candidate outputs.

## 6. Phase D: freeze the evaluation protocol

### D1. Paired data

Within every target/initialization/stream cell, all arms receive:

- identical initial particles;
- identical ordered target minibatches;
- identical evaluation and cross-reference samples;
- identical fixed training and held-out direction banks;
- identical target-observation and particle budgets.

Algorithm-specific internal randomness must use separately derived, recorded
seeds.  Changing an arm's RNG must not change the target stream.

### D2. Primary metric

The primary endpoint is squared energy distance:

```text
ED2(output particles, independent target reference).
```

Aggregate per-stream repetitions to one target-level value before statistical
inference.  Ratios use a small prespecified numerical floor only to avoid
`log(0)`; record the floor.

### D3. Secondary quality metrics

- held-out sliced Wasserstein-1 on 64 phase-shifted directions;
- held-out projected-quantile RMSE;
- oracle mode-mass L1 for synthetic mixtures, explicitly labeled diagnostic;
- mode coverage;
- rare-mode recovered mass;
- bridge/excess low-density occupancy for separated targets;
- numerical divergence count.

### D4. Cost metrics

- training wall time after one untimed warm-up;
- target observations;
- kernel-pair count for paper;
- projection and backprojection dot-product count;
- sort-work proxy;
- reconstruction sweeps;
- persistent-state scalars and bytes;
- peak working scalars and bytes;
- serialized accumulator size.

Wall time is machine- and implementation-specific.  The operation and memory
ledgers remain the portable comparison.

### D5. Reference sample stability

Use at least 4,096 independent reference observations per target if runtime
permits.  On a prespecified subset, repeat evaluation with 8,192 observations.
Report the induced metric change; do not change reference size after comparing
arms.

## 7. Phase E: prespecified statistical analysis

### E1. Target-level paired effects

For each target, compute the median across paired stream/initialization
repetitions for every arm.  Form log ratios:

```text
log(ED2_candidate / ED2_baseline)
log(SW1_candidate / SW1_baseline).
```

Report:

- geometric mean ratio across targets;
- median target ratio;
- fraction of targets won;
- worst-family geometric mean ratio;
- 95% target-level paired bootstrap interval;
- family-stratified ratios and intervals.

Bootstrap target instances, stratified by family.  Do not bootstrap individual
seeds as if they were independent targets.

### E2. Primary comparisons

The primary family contains exactly four comparisons:

1. KLL quality arm ED2 versus historical PSQT;
2. KLL quality arm ED2 versus paper;
3. Reservoir efficiency arm ED2 versus historical PSQT;
4. Reservoir efficiency arm ED2 versus paper.

Use a prespecified Holm correction or declare one ordered hierarchy:

```text
KLL vs historical -> KLL vs paper -> reservoir vs historical
-> reservoir vs paper.
```

Held-out SW1 and cost comparisons are secondary and should not silently
replace a failed ED2 endpoint.

### E3. Missing or divergent trials

- Never discard a divergent trial.
- Assign it the registered finite penalty or classify it as an automatic loss;
  choose and record this rule before execution.
- Preserve censored timing and failure diagnostics.
- Report the raw divergence count separately.

## 8. Acceptance gates

### G1. Quality-arm promotion

Promote the KLL quality arm only if all hold:

1. ED2 geometric-mean ratio versus historical PSQT is below `0.80` and its
   95% target-bootstrap upper bound is below `1.0`;
2. ED2 ratio versus paper is below `0.80` with upper bound below `1.0`;
3. held-out SW1 ratio versus both baselines is below `1.0`;
4. it wins at least 70% of target instances against each baseline;
5. no family geometric-mean ED2 ratio exceeds `1.10` versus historical PSQT;
6. no numerical divergence occurs;
7. 5% and 10% rare modes are recovered in at least 90% of relevant trials;
8. median excess bridge occupancy is at most one particle.

### G2. Efficiency-arm promotion

Promote Reservoir 1,024 only if all hold:

1. ED2 ratio versus historical PSQT is below `0.90`, upper bound below `1.0`;
2. ED2 ratio versus paper is below `0.90`, upper bound below `1.0`;
3. held-out SW1 does not regress by more than 5% versus historical PSQT;
4. it wins at least 60% of target instances against each baseline;
5. persistent state is no larger than 1.10 times historical PSQT;
6. median wall time is below historical PSQT on the benchmark machine;
7. no numerical divergence occurs;
8. rare-mode recovery is not worse than historical PSQT.

### G3. General-improvement claim

Use the phrase “general low-dimensional improvement” only if:

- the quality arm passes G1;
- at least one bounded arm beats the paper and historical PSQT on both primary
  quality metrics;
- no family shows a material unexplained regression;
- target-level uncertainty, not seed-level uncertainty, supports the result;
- all artifact and protocol audits pass.

Otherwise report a family- or metric-specific result.

## 9. Artifact and file architecture

Recommended new files:

```text
numerics/
  standard_projected_kll.py
  audit_projected_kll.py
  PSQTAccumulatorConfirmatoryProtocol.md
  psqt_accumulator_confirmatory_registry.json
  generate_psqt_accumulator_registry.py
  run_psqt_accumulator_confirmatory.py
  PSQTAccumulatorConfirmatoryResults.md
  psqt_accumulator_confirmatory_runs/<timestamp>/
```

Every final artifact must contain:

- immutable registry JSON and SHA-256;
- protocol and source snapshots;
- dependency names and versions;
- Git commit and dirty-worktree status;
- command, machine, Python, NumPy, and sketch-library versions;
- raw per-trial CSV;
- target-level aggregate CSV;
- bootstrap draws or their deterministic seeds;
- summary JSON and result Markdown;
- representative plots chosen by a prespecified rule, not by appearance;
- operation, state, and peak-memory ledgers;
- invariant and audit outputs.

## 10. Stop and contamination rules

After the registry hash and protocol are frozen:

- do not change algorithm code, capacities, directions, metrics, thresholds,
  or statistical analysis;
- if a correctness bug is found, stop the run, record the failure, repair it,
  generate a new registry seed, and label the previous registry contaminated;
- do not inspect partial family results to decide whether to continue;
- do not replace a failed arm with Reservoir 2,048 or exact pooling;
- do not tune paper bandwidth on the confirmatory targets;
- do not reuse confirmatory targets for a second “confirmation.”

## 11. Concrete execution order

1. Implement the standard KLL wrapper and independent cross-check harness.
2. Run sketch stress tests and record rank/memory behavior.
3. Freeze the four primary arms and optional exact-pooled diagnostic.
4. Write `PSQTAccumulatorConfirmatoryProtocol.md` with exact code hashes and
   gates.
5. Implement and test the registry generator without importing candidate
   algorithms.
6. Generate the fresh registry once; save and hash it.
7. Run a structural smoke test on a separate disposable registry.
8. Revalidate source hashes and execute the sealed registry once.
9. Aggregate by target, run the prespecified stratified paired bootstrap, and
   evaluate gates automatically.
10. Write the result whether gates pass or fail; preserve all raw artifacts.
11. Only after publication of the result decide whether to pursue geometric
    residual repairs or higher-dimensional scaling.

## 12. Recommended decision

The immediate next implementation task should be **Phase A only**: standardize
and cross-check the KLL quality accumulator without another endpoint search.
Once that implementation is trustworthy, freeze KLL-k128 and Reservoir-R1024
and move directly to the fresh sealed registry.

