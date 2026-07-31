# Coherent-route neural amortization development results

Date: 2026-07-24

Artifacts:

- `route_amortization_development.json`
- `route_amortization_development.json.sha256`
- `hybrid_route_development.json`
- `hybrid_route_development.json.sha256`

Scope:

- development, not confirmation;
- seven structured two-dimensional targets;
- three seeds;
- \(N=256\) persistent training latents;
- \(L=32\) sliced directions;
- 20 macro-steps x 20 Adam updates;
- the repository's 2x64 Tanh neural-PSQT generator;
- fresh latent evaluation;
- disjoint target planning, calibration, and evaluation pools;
- no official-paper comparison.

## Main verdict

Pure maximum-consensus EST is not a good neural teacher in this experiment.

The strongest route is the minimum-Euclidean-cost balanced assignment. A
nearly equivalent sparse alternative is:

> use the EST union only as a candidate graph, then choose the
> minimum-Euclidean-cost perfect matching on that graph.

Increasing the sliced-consensus weight above zero generally makes the neural
endpoint worse. The persistent sliced information is therefore useful here as
**edge discovery/sparsification**, not as the dominant route objective.

## 1. Pure EST result

Compared with fixed minimum-Euclidean assignment, fixed
maximum-consensus EST had median paired ratios/differences:

| endpoint | pure EST / Euclidean or difference |
|---|---:|
| ED2 ratio | `2.921` |
| SW1 ratio | `1.844` |
| precision difference | `-0.0596` |
| coverage difference | `-0.1777` |
| precision x coverage ratio | `0.688` |

Pure EST obtains high sliced vote agreement but accepts routes that are
materially longer and harder for the continuous generator to learn.

## 2. Replanning instability

When the same latent cohort was reassigned by balanced EST at every
macro-step, the median route-switch fraction by target was:

```text
checkerboard  0.885
Swiss roll    0.896
moons         0.832
rings         0.888
pinwheel      0.873
separated     0.922
rare modes    0.920
```

Thus 83--92% of persistent latent identities changed target per macro-step.
This directly validates the route-instability concern from the research plan.
Recomputing a mathematically balanced plan does not make the neural regression
target temporally coherent.

## 3. Fixed Euclidean route

The fixed minimum-Euclidean assignment was the strongest broadly reliable
teacher:

- it had the lowest or near-lowest neural ED2/SW1 on every target;
- it gave much better disconnected-mode support than pure EST;
- it was substantially easier to fit on the persistent training cohort;
- it generalized from the fixed training cohort to fresh latent samples.

This is not a novel result by itself. It is evidence that route smoothness and
persistence matter more to neural amortization than maximizing agreement with
independent projected ranks.

## 4. Sparse hybrid family

The hybrid assignment minimizes, on EST-supported edges,

\[
\frac{\|x_i-y_j\|^2}{s^2}
-\lambda\log\left(\frac{n_{ij}}{L}\right),
\]

where \(n_{ij}\) is the number of sliced plans proposing the edge.

Paired medians relative to dense Euclidean assignment:

| arm | ED2 ratio | SW1 ratio | precision diff | coverage diff | precision x coverage ratio |
|---|---:|---:|---:|---:|---:|
| `hybrid_0` | `1.0395` | `1.0200` | `+0.0020` | `-0.0049` | `0.9971` |
| `hybrid_0.05` | `1.0552` | `1.0395` | `+0.0039` | `-0.0234` | `0.9843` |
| `hybrid_0.1` | `1.1534` | `1.0699` | `+0.0020` | `-0.0137` | `0.9772` |

Weights `0.25`, `0.5`, and `1.0` deteriorated further on separated modes and
curved supports.

The selected development candidate is therefore `hybrid_0`:

- exact source/target empirical marginals;
- one target identity per source;
- EST-support restriction;
- no additional preference for repeated sliced votes;
- approximately dense-Euclidean neural quality in this small development run.

No nonzero consensus weight is promoted.

Teacher-route and route-construction diagnostics relative to dense Euclidean
assignment:

| arm | squared route-cost ratio | sliced-agreement ratio | end-to-end assignment-time ratio |
|---|---:|---:|---:|
| `hybrid_0` | `1.0036` | `2.354` | `1.827` |
| `hybrid_0.05` | `1.0119` | `3.610` | `0.612` |
| pure EST consensus | `1.1562` | `4.251` | `0.177` |

All discrete teacher arms had maximum target-marginal L1 error `0`.

At this small CPU size, `hybrid_0` is not faster end to end than dense
Euclidean assignment: building projections/ranks plus the weighted sparse
matching costs more than the dense solver. The previously reported sparse
speedup applied to the matching solver after the EST graph had already been
constructed. It must not be presented as an end-to-end speedup.

`hybrid_0.05` is an interesting cost/consensus compromise, but it loses more
neural endpoint quality. It remains an efficiency ablation, not the selected
quality arm.

## 5. Independent sliced-loss control

Relative to fixed Euclidean assignment, the old independently sliced neural
loss had paired medians:

| endpoint | result |
|---|---:|
| ED2 ratio | `1.117` |
| SW1 ratio | `1.062` |
| precision difference | `-0.0459` |
| coverage difference | `-0.0117` |
| precision x coverage ratio | `0.848` |

It remained competitive on global ED2 for checkerboard and rings, but paid a
support-precision cost. This reproduces the earlier global-mass versus local
geometry tradeoff under the corrected structured suite.

## 6. What is and is not established

Established in development:

1. fresh EST replanning is highly unstable;
2. fixed routes are easier to amortize;
3. Euclidean route smoothness is more valuable than pure sliced consensus;
4. EST support can sparsify the route search with only a small endpoint loss;
5. the old sliced loss still sacrifices support precision.

Not established:

1. a win over the official paper model;
2. a win over dense Euclidean assignment;
3. an end-to-end speed advantage--the plan-building projection/sort cost must
   be included, not only sparse solver time;
4. neural support quality sufficient for presentation;
5. route-aware/discrete-head gains;
6. higher-dimensional behavior;
7. encoder independence.

## 7. Decision

Do not confirm pure EST and do not return to the rejected nearest-neighbor
partial controller.

Proceed to a route-aware architecture screen using:

- dense Euclidean fixed assignment as the teacher-quality ceiling;
- `hybrid_0` as the sparse PSQT-derived candidate graph;
- independent sliced loss as the historical neural control.

The next architectural question is whether a route-conditioned or
mixture-of-experts generator can improve disconnected-support precision and
coverage while preserving fresh-latent generalization.

Before any efficiency claim, extend the ledger to include:

- direction projections;
- sorting;
- EST edge construction;
- sparse matching;
- dense cost construction;
- target accesses;
- model training;
- peak memory.
