# Projection and kernel cost optimization results

## Status

This file records development evidence for the implementation plan in
`ProjectionKernelCostOptimizationPlan.md`.  It is not a fresh confirmation and
does not replace `NeuralConditionedTransportConfirmatoryResults.md`.

The completed stages are:

- E0 exact projection reuse and normalized-kernel refactor; and
- P1 balanced registered orthogonal-block direction sharding; and
- K1 midpoint-stratified local-field cadence; and
- K2 weighted projection-tree representatives with exact multiplicity and
  deleted-self-mask semantics.

K3 positive kernel features have not been implemented.

## Audited artifacts

All three 20-step runs used the same consumed 16-target development registry,
the same target/model/latent seeds, and five conditioned-transport arms.  Each
artifact passed its internal audit.

| Setting | Artifact |
|---|---|
| active 32 | `conditioned_transport_runs/20260722-181755-consumed` |
| active 64 | `conditioned_transport_runs/20260722-181914-consumed` |
| full registered bank | `conditioned_transport_runs/20260722-182034-consumed` |

The full-bank artifact was rerun after E0 so that field-refactor effects are
not conflated with direction-sharding effects.

## Aggregate endpoints

Raw medians over the 16 target cells were:

| Active directions | Exact-hybrid ED2 | Exact-hybrid SW1 | KLL-hybrid ED2 | KLL-hybrid SW1 |
|---:|---:|---:|---:|---:|
| 32 | **0.013972** | **0.077041** | **0.014173** | **0.077995** |
| 64 | 0.017261 | 0.080715 | 0.015105 | 0.080089 |
| full | 0.017261 | 0.082573 | 0.017241 | 0.083764 |

Because ratios of separate medians can be misleading, the primary comparison
uses within-target ratios against the fresh full-bank row:

| Arm, active 32 | Paired median ED2 ratio | ED2 wins | Paired median SW1 ratio | SW1 wins | Median wall ratio |
|---|---:|---:|---:|---:|---:|
| exact hybrid | **0.9486** | 12/16 | **0.9553** | 11/16 | **0.8382** |
| KLL hybrid | **0.9371** | 11/16 | **0.9588** | 12/16 | **0.8531** |
| KLL guarded | **0.9487** | 10/16 | 0.9925 | 9/16 | **0.7778** |

Thus active-32 did not merely preserve the full-bank endpoint in this
development run.  It modestly improved the typical ED2 and SW1 endpoint while
reducing measured CPU training time.  The likely mechanism is useful
stochastic/alternating projection regularization; this is an empirical
interpretation, not a theorem.

Active-64 equals the full bank in 2D, 4D, and 8D because those registered banks
already have 64 directions.  Its only intervention is therefore in 16D.  Its
overall paired medians are exactly one for this reason, although its four 16D
cells had paired median ED2 and SW1 ratios of `0.9461` and `0.9614`.

## Cost result

For the exact hybrid, active-32 versus full produced:

| Scope | Training projection ratio | Total projection ratio | Wall-time ratio |
|---|---:|---:|---:|
| all 16 cells, median | **0.5000** | **0.8333** | **0.8382** |
| four 16D cells, median | **0.1742** | **0.7247** | **0.6288** |

`Training projection ratio` measures repeated generated direction/sample dot
products.  `Total projection ratio` also includes constructing the complete
persistent target atlas, which is intentionally unchanged and can be
amortized over multiple models/runs.  The local field still evaluates
10,485,760 kernel pairs, so P1 does not address the remaining quadratic kernel
work.

At 16D the registered banks contain 176 or 192 directions.  Active-32 uses two
complete 16-direction orthogonal blocks per macro-step.  Across 20 macro-steps
every block is visited at least three times and block exposure differs by at
most one.

## Interpretation

P1 is successful as a development result:

1. full-active equivalence is bitwise exact;
2. every subset is a registered immutable atlas slice;
3. only complete certified orthogonal blocks are selected;
4. exposure is balanced and audited;
5. active-32 reduced repeated projection work and wall time; and
6. no typical endpoint-quality penalty appeared on the consumed registry.

This does **not** yet establish a new confirmed algorithm.  Active-32 was
selected after viewing the development results.  Before a final efficiency
claim it must be frozen together with the kernel-cost changes and evaluated on
fresh paired targets/seeds.

## Decision and next step

Carry active-32 forward as the primary projection configuration.  Retain full
and active-64 as comparators; active-16 is an aggressive boundary rather than
the expected winner.

K1 then tested local-field cadence with active-32 fixed:

- full local field on 20/20 macro-steps;
- evenly spaced local field on 10/20 macro-steps; and
- evenly spaced local field on 5/20 macro-steps.

The additional audited artifacts are:

| Local calls | Artifact |
|---:|---|
| 10 | `conditioned_transport_runs/20260722-182640-consumed` |
| 5 | `conditioned_transport_runs/20260722-182750-consumed` |

Against active-32 with all 20 calls, the exact hybrid produced:

| Local calls | Kernel-pair ratio | Paired median ED2 ratio | ED2 wins | Paired median SW1 ratio | SW1 wins | Wall ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | **0.50** | 1.1496 | 4/16 | 1.0851 | 4/16 | 0.8634 |
| 5 | **0.25** | 1.2452 | 3/16 | 1.1408 | 4/16 | 0.7876 |

Raw exact-hybrid medians changed from `0.013972 / 0.077041` at 20 calls to
`0.016851 / 0.085783` at 10 calls and `0.018684 / 0.092587` at 5 calls.
Degradation is monotone in this screen.  The local correction is therefore
doing useful temporal work rather than serving as a rarely needed repair.

K1 remains a valid user-selectable quality/cost control, and the 10-call arm
may still outperform the historical paper port.  It is not the preferred
best-quality successor: halving pair count bought only a 13.7% median CPU
training-time reduction after E0 while losing about 15.0% ED2 and 8.5% SW1
relative to the active-32/full-local arm.

## K2 weighted representatives

### Implementation and hard gates

K2 keeps every one of the 20 useful local updates and compresses each positive
and generated 512-point support to `M` balanced projection-tree leaves.  A
leaf center carries its integer multiplicity.  Every original column remains
conceptually present, so the generated query's own negative entry is deleted
before its per-column normalizer is aggregated.  This is not the incorrect
shortcut of applying one mask to an entire representative.

The implementation has three fail-closed checks:

1. the `M=512` weighted field matches the dense masked and unmasked fields to
   `3.33e-16` in float64;
2. its full-support row and column masses match exactly in the regression
   case; and
3. representative multiplicities must equal assignment counts and sum to the
   original support size.

The tree was vectorized level-by-level.  A 512-by-16 CPU microbenchmark gave
about `2.22 ms` for the complete `M=128` build-plus-field path, `1.90 ms` for
`M=64`, and `2.69 ms` for the dense field.  The first recursive Python
implementation was slower and is not the retained code.

### Valid artifacts

All valid K2 comparisons pass `PAPER_TAU=1.0` explicitly, use active-32, and
retain 20/20 local calls:

| Purpose | M | Artifact |
|---|---:|---|
| direct field/mass audit smoke | 128 | `conditioned_transport_runs/20260722-185903-smoke` |
| direct field/mass audit smoke | 64 | `conditioned_transport_runs/20260722-190351-smoke` |
| consumed development screen | 128 | `conditioned_transport_runs/20260722-185925-consumed` |
| consumed development screen | 64 | `conditioned_transport_runs/20260722-190035-consumed` |

Artifacts `20260722-184706-consumed-INVALID-TAU`,
`20260722-184837-consumed-INVALID-TAU`,
`20260722-185259-consumed-INVALID-TAU`, and
`20260722-185436-consumed-INVALID-TAU` are **invalid K2 comparisons**: an audit
exposed that they inherited the generic primitive's
`tau=0.5` default while the registered paper wrapper uses `PAPER_TAU=1.0`.
They must not be cited as compression results.

### Field approximation diagnostics

The audit compares the compressed and dense fields on the same features and
also materializes the dense conceptual row/column masses.  Values below are
from the deterministic exact-hybrid smoke cells; two early macro-steps were
audited per cell.  Cosine is the minimum of the two calls, while other errors
are means.

| M | d | field relative L2 | field cosine | row-mass relative L2 | column-mass relative L2 |
|---:|---:|---:|---:|---:|---:|
| 128 | 2 | 0.0209 | 0.9998 | 0.0020 | 0.0456 |
| 128 | 4 | 0.0606 | 0.9969 | 0.0167 | 0.1123 |
| 128 | 8 | 0.2230 | 0.9666 | 0.0638 | 0.1691 |
| 128 | 16 | 0.3411 | 0.9502 | 0.2939 | 0.2982 |
| 64 | 2 | 0.0396 | 0.9989 | 0.0049 | 0.0625 |
| 64 | 4 | 0.0882 | 0.9947 | 0.0272 | 0.1416 |
| 64 | 8 | 0.3463 | 0.9453 | 0.1006 | 0.2019 |
| 64 | 16 | 0.4676 | 0.9007 | 0.4137 | 0.4037 |

The degradation with dimension is real.  RMS normalization of the local field
removes most magnitude bias, but it cannot repair directional error.  This is
why `M=128` is the quality-oriented setting and `M=64` is only an aggressive
efficiency setting.

### Endpoint and cost comparison against dense active-32

The deterministic exact-hybrid arm is the primary comparison because Apache
KLL compaction is not seedable through the current Python API.  Ratios are
paired within the same 16 consumed targets; below one is better.

| M | kernel-pair ratio | paired median ED2 ratio | ED2 wins | paired median SW1 ratio | SW1 wins | median wall ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | **0.25** | 1.0378 | 6/16 | 1.0224 | 4/16 | **0.9179** |
| 64 | **0.125** | 1.0818 | 4/16 | 1.0448 | 3/16 | **0.9115** |

K2 is therefore a genuine Pareto tradeoff, not a free quality improvement.
`M=128` removes 75% of local kernel pairs and reduces measured end-to-end CPU
time by about 8.2%, at a typical cost of 3.8% ED2 and 2.2% SW1.  `M=64` saves
another factor of two in pairs but gives up materially more endpoint quality.

The tree itself adds projection and sorting work.  Relative to dense
active-32, the `M=128` ledger reports 1.0875x projection scalar products and
1.0396x sort-work proxy; `M=64` reports 1.0750x and 1.0368x.  The measured wall
gain comes from replacing pairwise distance/kernel work despite this overhead.

### Historical comparison with the repository paper-neural port

The table in this subsection is retained as development provenance, but its
wall ratio is superseded. The cited paper row came from an earlier runner with
a different field implementation and timing context; it is not a matched
wall-clock comparator for K2. The audit-repaired, same-run comparison and
fresh confirmation are reported in
`ProjectionKernelOptimizationConfirmationResults.md`.

The relevant paper baseline is replication zero with concentrated
initialization in `neural_pooled_rank_runs/20260722-160125-standard-merged`.
Generator-example evaluation counts are equal.  On the same 16 development
targets:

| Setting | geometric ED2 ratio vs paper | geometric SW1 ratio vs paper | paired quality wins | kernel-pair ratio | median wall ratio |
|---|---:|---:|---:|---:|---:|
| active-32, M=128 | **0.3445** | **0.5649** | **16/16 on both** | **1.0** | **0.3107** |
| active-32, M=64 | **0.3632** | **0.5808** | **16/16 on both** | **0.5** | **0.3076** |

The pair-count observation remains valid, but the `0.31x` historical wall
ratio must not be cited as a matched speed result. The repaired fresh
comparison found a median online-training ratio of `0.6150` and a
setup-plus-training ratio of `0.7636`.

### Decision

Active-32 plus `M=128` was frozen as the primary efficiency candidate. It keeps
the local cadence that K1 showed was valuable, matches the paper port's kernel
pair count, and preserves most of the dense hybrid's endpoint gain. The fresh
paired confirmation is now complete and passed its predeclared ED2/SW1 gates;
see `ProjectionKernelOptimizationConfirmationResults.md`. `M=64` remains an
aggressive ablation rather than the promoted configuration. K3 positive
features remains a later scalability route.
