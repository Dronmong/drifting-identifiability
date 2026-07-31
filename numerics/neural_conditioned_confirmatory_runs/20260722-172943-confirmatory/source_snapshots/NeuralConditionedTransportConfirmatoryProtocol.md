# Frozen protocol: neural conditioned-transport confirmation

**Protocol status:** freeze-ready; no confirmatory endpoint has been observed  
**Development predecessor:** `ConditionedTransportAmortizationResults.md`  
**Date frozen:** 2026-07-22  
**Execution note:** v1 and v2 were invalidated by pre-reporting infrastructure
bugs recorded in `NeuralConditionedTransportFailures.md`; this protocol now
governs the fresh v3 registry.

## Primary question

On newly generated multidimensional synthetic targets, does deterministic
conditioned transport-then-amortize improve both energy distance and held-out
sliced Wasserstein distance over the repository's faithful neural port of the
paper field and over the strongest competing neural baselines?

This protocol tests the repository algorithms. It does not claim superiority
on the paper's image models, datasets, encoders, FID, or Inception Score.

## Fixed environment

```text
Python                 3.12
PyTorch                2.7.1 CPU
Apache DataSketches    5.2.0
NumPy/SciPy             resolved by uv and recorded
deterministic Torch algorithms enabled
Torch threads           1
```

KLL's Python binding does not expose its internal compaction seed. Therefore
all KLL sketches are constructed, serialized, hashed, and frozen before any
model is trained. Confirmation reuses those exact payload-derived tables.

## Fresh target registry

`generate_neural_conditioned_registry.py` is algorithm-independent and refuses
overwrite. V3 master seed `20260803` creates 64 independent target instances:

```text
dimensions         2, 4, 8, 16
families           balanced GMM, rare GMM, correlated t, nonlinear
instances          4 per dimension/family cell
training pool      20,480 observations
evaluation sample   2,048 observations, disjoint seed
registered probes      64 training, 128 held-out
initializations    concentrated, broad
```

Parameters vary within frozen ranges:

- balanced GMM: 3--6 equal-mass components, radius 2.3--3.2, component scales
  0.28--0.58;
- rare GMM: minority mass in `{0.02, 0.05, 0.10}`, separation 3.2--4.5, main
  scale 0.45--0.65, rare scale 0.22--0.35;
- correlated t: degrees of freedom 3.5--7, axis ratio varying by target, and
  skew strength 0.10--0.35;
- nonlinear: curve strength 0.55--1.0 and noise 0.15--0.30.

No rejection sampling or target replacement is permitted.

## Frozen algorithms

Every neural arm uses the same MLP architecture and paired initial state.
Every arm receives exactly 20,480 generator-example evaluations.

### Baselines

1. `paper-neural`: exact repository Algorithm-2 field port, `tau=1`, gain
   `0.15`, batch 64, 320 Adam updates, learning rate `0.016`.
2. `minibatch-sw`: fresh 64-sample target ranks, 64 registered directions,
   batch 64, 320 Adam updates, learning rate `0.016`.
3. `kll-small-rsr`: frozen KLL target atlas, generated population 64,
   Run-Sort-ReRun, 160 Adam updates, learning rate `0.016`.

### Deterministic primary

`cta-exact-hybrid`:

- retain all 64 registered training directions;
- append deterministic orthogonal blocks until the symmetric quadratic sensing
  matrix has full rank and condition number at most 25;
- exact 128-knot inverted-ECDF atlas from all 20,480 target observations;
- generated particle population 512;
- coherent PSQT particle step `eta=0.5`;
- paper local field RMS-normalized to the PSQT correction, scale cap 256, fixed
  weight 0.25;
- eight shuffled 64-sample frozen-teacher Adam updates per macro-step;
- 20 macro-steps, 160 Adam updates, learning rate `0.016`.

### KLL retention arm

`cta-kll-hybrid` is identical to the primary except that it uses the frozen
Apache KLL-k128 quantile table. Its payloads are fixed before training.

The conservative guard is excluded: it was active but reduced development
quality. Exact-global and KLL-global are excluded to avoid expanding the
confirmatory family after their role was already resolved in development.

## Pairing and work accounting

Within each target and initialization, all arms share:

- the initial neural parameters;
- training and evaluation latent banks;
- target pool and evaluation reference;
- base 64 training and 128 held-out directions; and
- target-batch and student-permutation schedules where applicable.

Candidate arms use each 512-latent macro population twice (teacher and student)
and therefore see 10,240 unique training latents. Ordinary baselines see
20,480; RSR sees 10,240. This is the already-selected generator-example budget
policy. Kernel pairs, projection products, sort work, target accesses, forward
calls, Adam updates, atlas bytes, serialized KLL bytes, and wall time are all
reported. No efficiency superiority gate is defined.

## Metrics and statistical unit

Primary metrics:

- squared energy distance (`ED2`);
- held-out-direction sliced Wasserstein-1 (`SW1`).

Secondary diagnostics:

- training-direction quantile RMSE;
- covariance effective rank;
- output RMS;
- mixture mode coverage, L1 mass error, and rare-mode mass error;
- divergence and all work ledgers.

The independent unit is a target instance, not an initialization. For each
target/arm/metric, reduce the concentrated and broad initializations by their
median. Ratios use an additive numerical floor `1e-12`. Geometric means are
computed over the 64 target ratios.

Uncertainty uses 5,000 paired bootstrap draws. Each draw resamples the four
instances independently within every dimension/family cell, preserving all 16
cells. The two-sided percentile interval is reported. The bootstrap seed is
stored in the registry.

The `baseline-envelope` comparator is defined target-by-target and
metric-by-metric as the minimum of paper, minibatch SW, and small KLL-RSR.
This definition is frozen before outcomes.

## Promotion gates

### Exact primary: all gates must pass

1. ED2 geometric-mean ratio versus paper is below `0.75` and its 95% upper
   bound is below `1`.
2. SW1 geometric-mean ratio versus paper is below `0.80` and its upper bound is
   below `1`.
3. ED2 and SW1 ratios versus the baseline envelope are each below `0.90`, with
   both 95% upper bounds below `1`.
4. The primary beats paper on both metrics in at least `48/64` targets (75%).
5. It beats the baseline envelope on both metrics in at least `40/64` targets
   (62.5%).
6. No dimension/family cell has an ED2 geometric-mean ratio above `1.10`
   versus paper.
7. Zero divergence.
8. Across the 16 rare-GMM targets, median mode coverage is at least paper's
   and median rare-mass error is at most `0.80` times paper's (when paper's
   median error is positive).

### KLL retention: all gates must pass

1. KLL/exact ED2 and SW1 geometric-mean ratios are each at most `1.15`, with
   95% upper bounds at most `1.25`.
2. KLL beats paper on both metrics in at least `45/64` targets.
3. No dimension/family cell has an ED2 ratio above `1.25` versus exact.
4. Zero divergence and every frozen KLL payload passes hash/offset audit.

## Stop and contamination rules

After the registry, frozen atlases, and source-hash manifest are written:

- any hash mismatch aborts the run;
- no target, seed, direction, arm, optimizer setting, metric, reduction,
  bootstrap procedure, or gate may change;
- partial outputs may not alter execution;
- the complete result is reported whether it passes or fails;
- a correctness bug requires logging the failure and generating a new registry
  with a new master seed; and
- performance disappointment is not a correctness bug.
