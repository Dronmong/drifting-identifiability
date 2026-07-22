# Persistent Sliced Quantile Transport: audited 2D implementation plan

**Status:** development plan; not a frozen confirmatory protocol  
**Primary question:** does the transport-aligned persistence responsible for
the one-dimensional PQT result survive genuine two-dimensional dependence?

This document deliberately separates a mechanism experiment from a claim about
images or a conventional neural generator.  The first deliverable is a
nonparametric 2D particle generator with a complete sample/work ledger and
visual comparisons against the paper estimator.  A replayable compositional
map and learned compression are subsequent promotion stages.

## 1. Mathematical audit

### 1.1 Why coordinate-wise PQT is not the answer

Applying scalar PQT independently to `x` and `y` fixes only the two coordinate
marginals.  It cannot distinguish, for example, mass concentrated near
`y = x` from mass concentrated near `y = -x` when their coordinate marginals
agree.  Coordinate PQT is therefore included only as a required negative
control.

For a unit direction `theta`, write

```text
a_i(theta) = <theta, x_i>
b_j(theta) = <theta, y_j>.
```

All one-dimensional projections determine a Borel probability law on
Euclidean space, but a finite direction bank is only an approximation.  The
implementation must consequently evaluate on held-out directions and on a
non-sliced metric such as energy distance.

### 1.2 Persistent projected-quantile statistics

Fix probability knots `u_k = (k + 1/2) / K` and directions
`theta_1, ..., theta_L`.  For every direction, store a nondecreasing projected
target quantile vector `Qbar[l, k]`.  Given target minibatch `Y_t`, update

```text
Qbar_t[l, :] =
  (w_{t-1} * Qbar_{t-1}[l, :] + Q(Y_t theta_l; u)) / (w_{t-1} + 1),
w_t = w_{t-1} + 1.
```

This is exactly the running-average rule used by scalar PQT.  For finite batch
size it estimates the expectation of the empirical batch quantile, not
literally the population quantile.  That distinction must remain explicit.

The initial projected quantiles receive `prior_batches` units of weight.  A
constant target distribution and fixed batch size make this a standard
running average; no unreported learning-rate schedule is introduced.

### 1.3 Geometric reconstruction

Let `pi_l` sort the particles by projection on `theta_l`.  At the particle
midpoint probabilities, interpolate the stored target vector and define

```text
r[i,l] = desired_projection[i,l] - <theta_l, x_i>.
```

One simultaneous reconstruction step is

```text
x_i <- x_i + eta * (d / L) * sum_l r[i,l] theta_l.
```

Away from projection ties, this is a scaled negative gradient step for the
finite sliced quantile least-squares objective

```text
J(X) = (1 / (2L)) * sum_l sum_i
       (sorted(<theta_l, X>)_i - target_l(u_i))^2.
```

Stable sorting supplies a deterministic subgradient selection at ties.  We do
not claim global convexity: the sorting permutations change with `X`, and
finite projected quantile tables need not be exactly realizable by one common
particle cloud.

For the 2D bank

```text
theta_l = (cos(pi*l/L), sin(pi*l/L)),  l = 0,...,L-1,
```

the tight-frame identity is

```text
(2/L) * sum_l theta_l theta_l^T = I_2.
```

Thus the `2/L` factor gives the correct scale for a pure translation rather
than shrinking it by one half.  Inner reconstruction steps solve the
finite-direction consistency problem without changing the statistical weight
assigned to a target minibatch.

### 1.4 Exact 1D reduction

Take `d = L = 1`, `theta = 1`, `N = K`, and one reconstruction step with
`eta = 1`.  Particle rank `k` is sent to the newly accumulated value
`Qbar[k]`.  Consequently the sorted particle vector is exactly the scalar PQT
knot vector after every update.  This is an executable invariant, not merely
an analogy.

### 1.5 Replayable ridge maps

A scalar projected correction can also be stored as

```text
R(x) = x + eta * (T(<theta,x>) - <theta,x>) theta,
```

where `T = Q_target o F_source` is represented by monotone quantile knots.
Composing such ridge maps gives a replayable nonparametric generator for new
latent samples.  The first benchmark uses particles because it isolates the
transport mechanism.  Ridge-layer primitives and deterministic replay tests
are implemented now; map-depth control and compression are later stages.

## 2. Development implementation

Create `persistent_sliced_quantile_transport.py` with:

1. validated midpoint grids and direction banks;
2. `PersistentSlicedQuantileTransport`, containing particles, projected target
   quantiles, prior mass, and an auditable work ledger;
3. simultaneous tight-frame reconstruction;
4. projected training loss and held-out-direction diagnostics;
5. `RidgeQuantileLayer` and `CompositionalSlicedQuantileMap` for replay;
6. deterministic invariants for:
   - exact reduction to scalar PQT;
   - the 2D tight-frame identity;
   - exact translation scaling;
   - rotation equivariance when the direction bank is rotated with the data;
   - monotone projected target tables;
   - distinction of equal-marginal/different-dependence examples;
   - ridge-layer replay.

The implementation must not modify the frozen PQT, QLD, or LB-QCD code.

## 3. First paired 2D experiment

Create a separate development runner.  It reuses target definitions, the exact
paper bi-softmax estimator, energy distance, sliced-W1, and work accounting
from `lowdim_drift.py`.

### Arms

1. paper estimator at a small predeclared bandwidth grid;
2. the best fixed paper arm reported descriptively and a per-cell paper oracle
   reported separately;
3. coordinate-only PQT (`theta = e_1, e_2`) as a negative control;
4. uniform-direction PSQT;
5. optional online/nonpersistent sliced transport as an attribution arm after
   the core system is stable.

All paired arms receive the same initial particle cloud, target minibatch
stream, update count, particle count, and target-sample count.  An oracle paper
bandwidth is not a deployable algorithm and may not replace the selected paper
arm in a later gate.

### Development targets

- separated equal and unequal Gaussian mixtures;
- rotated correlated and anti-correlated targets;
- heteroscedastic mixtures;
- ring and concentric-circle supports;
- two moons;
- a skewed/heavy-tailed connected law;
- an equal-coordinate-marginal dependence trap.

Use concentrated/missing and far initializations.  Do not tune on the later
fresh confirmatory registry.

### Metrics

- squared energy distance (primary, non-sliced);
- sliced-W1 on a fixed held-out direction bank disjoint from training angles;
- mode coverage and mode-mass L1 when oracle mixture labels exist;
- projected quantile RMSE on training and held-out directions;
- event time and divergence;
- target samples, projection sorts, projection dot products, kernel pairs,
  stored scalars, and wall time.

Visuals use identical axes and paired seeds and show target, initialization,
paper, coordinate-PQT, and PSQT.  A visual is supporting evidence, never the
gate.

## 4. Development order

1. Run invariant and smoke tests.
2. Search only the small grid
   `L in {8,16,32}`, `K in {64,128}`, reconstruction steps in `{1,3,6}`, and
   reconstruction step size in `{0.25,0.5,1}` on development targets.
3. Select using target-balanced ED2, with held-out SW1 and worst-family
   regressions as safeguards.
4. Test whether uniform directions fail specifically on rare modes or curved
   support.
5. Only if needed, add a predeclared mixture of uniform and discrepancy-focused
   directions.  Adaptive directions must be charged to the work ledger.
6. Freeze a fresh 2D registry and protocol before running confirmatory seeds.

## 5. Promotion gates

The exact thresholds will be frozen only after development, but a credible 2D
claim should require all of the following:

- target-balanced ED2 and held-out SW1 improvement over selected paper;
- bootstrap upper endpoints below no improvement;
- improvement over the per-cell paper oracle as a stronger secondary test;
- no material regression in any predefined target family or initialization;
- success on the equal-marginal dependence trap;
- no excess divergence;
- both equal-target-sample and wall-time/work ledgers;
- complete raw rows, source snapshots, hashes, and paired seeds.

## 6. Scope and next stages

A successful particle result would establish that the PQT mechanism survives
genuine 2D dependence.  It would not yet establish:

- superiority of a conventional neural generator;
- scaling beyond low dimension;
- image or feature-space performance;
- that a finite direction bank identifies arbitrary distributions exactly.

After a successful frozen 2D particle benchmark, proceed in this order:

1. promote the replayable ridge-map version;
2. periodically distill the ridge composition into a compact network;
3. compare uniform, discrepancy-maximizing, and random-path projections;
4. add occasional Sinkhorn barycentric refreshes only if slicing leaves a
   measured joint-geometry residual;
5. test 4D and 8D before any image-scale claim.

## 7. Literature boundary

Projected quantile transport is not itself novel.  The implementation is
informed by sliced-Wasserstein flows and sliced iterative normalizing flows.
The repository-specific research question is whether PQT's persistent
statistics, resolution accounting, and explicit cost discipline produce a
better low-dimensional drifting procedure.  Any later originality claim must
compare exact algorithms and assumptions rather than rename sliced transport.

