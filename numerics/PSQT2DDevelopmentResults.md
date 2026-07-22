# Persistent Sliced Quantile Transport: first 2D development result

**Status:** exploratory development, not a frozen confirmation  
**Implementation plan:** `PSQTHigherDimImplementationPlan.md`  
**Primary artifact:** `psqt_2d_runs/20260721-232453-screen`

## What was implemented

The screen tests a particle version of Persistent Sliced Quantile Transport
(PSQT).  It maintains running empirical target-quantile averages on a fixed
bank of 2D projections, then reconstructs one persistent particle cloud using
tight-frame backprojection.  The implementation also includes replayable
monotone ridge-quantile layers, although the first benchmark deliberately uses
particles to isolate the transport mechanism.

The executable invariant suite verifies:

- exact reduction to the repository's scalar PQT update when `d = L = 1`;
- the 2D uniform-direction tight-frame identity;
- correct translation scaling;
- equivariance when particles, targets, and directions rotate together;
- preservation of projected-quantile monotonicity;
- failure of coordinate marginals and success of off-axis projections on an
  equal-marginal/different-dependence construction;
- deterministic replay and line monotonicity of ridge-map layers.

All invariants and the independent exact-paper-estimator invariants passed.

## Screen design

The development screen used:

```text
targets                         9
initializations                 concentrated, far
paired seeds per cell           5
updates                         300
particles                       64
target minibatch                64
training target observations    19,200 per arm/trial
evaluation reference            1,024
held-out projection directions  64, phase-shifted from training
```

The targets cover equal, unequal, and heteroscedastic Gaussian mixtures; a
two-mode diagonal dependence trap; a correlated Gaussian; a ring; concentric
circles; two moons; and a skewed heavy-tailed connected law.

Every paired arm received the same initial cloud, target minibatches, update
count, and target observations.  Paper arms used the exact repository
bi-softmax estimator with `tau in {0.2, 0.5, 1, 2}` and `eta = 0.15 tau`.
Both the paper and PSQT configurations were selected descriptively on this
development screen, so no uncertainty interval from these data would be
confirmatory.

## Main result

The development-selected arms were:

```text
paper: paper-tau1
PSQT:  psqt-L32-K64-R3-e0.5
```

Target/initialization cells were reduced to seed medians before ratios were
geometrically aggregated.

| Comparison | ED2 ratio | held-out SW1 ratio |
|---|---:|---:|
| PSQT / selected paper | **0.3684** | **0.6908** |
| PSQT / coordinate-only PQT | **0.1373** | -- |
| PSQT / per-cell paper bandwidth oracle | **0.4635** | -- |

PSQT beat selected paper in **16 of 18** target/initialization cells.  There
were no divergences and all primary arms consumed exactly 19,200 training
target observations per trial.

### Family ED2 ratios versus selected paper

| Family | PSQT / paper |
|---|---:|
| Gaussian mixtures | **0.2694** |
| ring | **0.1998** |
| circles | **0.3409** |
| moons | **0.3650** |
| skewed/heavy-tail | **0.3470** |
| correlated Gaussian | **0.5475** |
| diagonal dependence trap | **1.3540** |

The aggregate improvement therefore survives genuine 2D curved support,
multimodality, unequal mass, heteroscedasticity, correlation, heavy tails, and
far initialization.  It is not uniform: selected paper remains better on both
diagonal-dependence cells.

## The important failure

Coordinate-only PQT scatters mass among combinations permitted by the two
correct marginals and performs very poorly on the diagonal target.  Off-axis
PSQT correctly recovers the diagonal orientation and two endpoints, proving
that the additional projections contain useful joint information.  However,
it leaves too many particles along the low-density bridge between the modes.
Paper drifting forms tighter endpoint clouds and wins ED2 on this family.

Increasing from 8 to 32 directions, doubling reconstruction iterations, or
changing reconstruction step size did not remove this bridge:

| PSQT arm | aggregate ED2 / paper | diagonal concentrated ED2 | median wall s |
|---|---:|---:|---:|
| `L8-R3-e.5` | 0.4128 | 0.01568 | 0.050 |
| `L16-R3-e.5` | 0.3793 | **0.01568** | 0.070 |
| `L16-R3-e1` | 0.3779 | 0.01571 | 0.070 |
| `L16-R6-e.5` | 0.3799 | 0.01572 | 0.086 |
| `L32-R3-e.5` | **0.3684** | 0.01573 | 0.110 |

This makes a simple resolution explanation unlikely.  The present
simultaneous least-squares reconstruction uses one rank assignment per
projection but no persistent joint assignment between projections.  On a
disconnected diagonal law, mutually consistent marginal orderings can be
approximated by a thin bridge.  The next mechanism experiment should compare:

1. sequential replayable ridge-map sweeps;
2. discrepancy-weighted directions concentrated near the unresolved bridge;
3. a sparse Sinkhorn/barycentric joint-assignment refresh;
4. a bridge-occupancy diagnostic that is independent of oracle mode labels.

That experiment must retain uniform directions and selected paper as controls.

## Cost audit

| Arm | median training wall s | kernel pairs | sort-work proxy | stored scalars |
|---|---:|---:|---:|---:|
| selected paper `tau=1` | 0.086 | 2,457,600 | 0 | **128** |
| coordinate PQT | **0.029** | 0 | 460,800 | 325 |
| PSQT `L16-R3-e1` | **0.070** | 0 | 7,372,800 | 1,249 |
| selected PSQT `L32-R3-e.5` | 0.110 | 0 | 14,745,600 | 2,305 |

PSQT eliminates kernels but increases storage.  After vectorizing the
direction-wise quantiles and rank assignments, the selected 32-direction arm
is approximately 28% slower than paper on this machine, while `L16-R3-e1` is
approximately 19% faster and retains nearly all of the quality gain.  It is
therefore the present Pareto candidate.  Wall time remains implementation- and
machine-specific; kernel and sort ledgers are the portable comparison.

## Visual finding

The saved figures use identical plot bounds and paired seed-zero concentrated
initializations.  Representative artifacts are:

- `visual_PS2-diagonal-dependence.png`;
- `visual_PS2-ring.png`;
- `visual_PS2-moons.png`;
- `visual_PS2-GMM5-unequal.png`.

PSQT visibly recovers the ring and moon topology much more cleanly than the
selected paper arm at this finite horizon.  The diagonal figure makes the
remaining bridge defect equally visible; it should not be hidden by aggregate
tables.

## Honest conclusion

This screen is strong evidence that the central PQT mechanism transfers beyond
one dimension: the advantage is not merely coordinate-marginal matching, and
it survives multiple genuinely 2D geometries under an equal target-sample
budget.  It is not yet a general 2D result because:

- algorithms and paper bandwidth were selected on these same development
  targets;
- only five seeds were used;
- the disconnected diagonal family regresses;
- current PSQT is slower and larger than the paper particle arm;
- the result is for a nonparametric particle model, not a neural generator.

The next valid milestone is to repair or explicitly bound the bridge failure,
select one Pareto configuration, freeze a fresh 2D registry and protocol, and
then run target-level paired uncertainty without changing the algorithm.
