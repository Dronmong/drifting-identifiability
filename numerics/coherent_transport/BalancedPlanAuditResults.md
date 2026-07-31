# Corrected balanced-plan development audit

Date: 2026-07-24

Artifact:

- `balanced_plan_audit.json`
- `balanced_plan_audit.json.sha256`

Command:

```text
uv run --with numpy --with scipy python \
  numerics/coherent_transport/run_balanced_plan_audit.py \
  --seeds 10 \
  --out numerics/coherent_transport/balanced_plan_audit.json
```

Scope:

- development only;
- seven structured two-dimensional targets;
- ten target/initialization seeds per target;
- \(N=256\);
- \(L=32\) sliced directions;
- disjoint planning, calibration-A, calibration-B, and evaluation target pools;
- no neural network;
- no official-paper performance claim.

## Verdict

The repaired sparse balanced EST route is a valid and potentially useful
teacher primitive.

It provides:

- exact empirical source and target marginals;
- one discrete target identity per source;
- support restricted to edges proposed by at least one sliced rank plan;
- the same maximum-consensus objective as the dense solver in all 70 cells;
- a sparse graph with median density `0.1006`; and
- median assignment-solver speedup `4.61x` over dense Euclidean Hungarian
  assignment at this \(N\) on this CPU.

It does **not** yet establish a better generative model. A full bijection sends
the source empirical measure to the planning-target empirical measure
regardless of whether the permutation came from EST, Euclidean assignment, or
a random permutation. Full endpoint metrics are therefore not a valid
mechanism discriminator.

## Aggregate mechanism findings

Across all 70 target/seed cells:

| comparison | median result |
|---|---:|
| sparse EST graph density | `0.1006` |
| dense/sparse EST consensus objective agreement | `70 / 70` |
| sparse EST target-marginal L1 | `0` |
| sparse EST unique-target fraction | `1` |
| Euclidean solve time / sparse EST solve time | `4.61x` |
| sparse EST / Euclidean mean squared route cost | `1.378x` |
| range of squared-route-cost ratio | `[1.076, 2.006]` |
| sparse EST / Euclidean sliced-agreement ratio | `3.674x` |

Interpretation:

- EST obtains much stronger agreement with the collection of sliced plans;
- this costs a moderate increase over the globally shortest Euclidean
  assignment;
- sparse matching obtains the same EST objective as dense matching without
  materializing the full graph;
- the graph is already about ten percent dense at \(L=32,N=256\), so its
  scaling with \(L,N,d\) still requires measurement.

## Why the earlier particle endpoint result is not promoted

Every balanced assignment returns a permutation of the planning targets:

\[
\{Y_{\pi(i)}:i=1,\ldots,N\}=\{Y_j:j=1,\ldots,N\}.
\]

Consequently its exact endpoint empirical distribution is independent of the
permutation. A random bijection, the Euclidean-optimal bijection, and balanced
EST all have the same endpoint sample multiset.

The assignment matters for:

- path length;
- path crossings;
- sliced consistency;
- temporal route stability;
- the difficulty of neural amortization;
- the behavior of bounded/intermediate updates.

It does not matter for the final free-particle empirical measure if particles
are simply moved all the way to their assigned planning targets.

The corrected one-step screen did not show a decisive universal geometry win:

- median sparse-EST precision difference versus Euclidean assignment:
  `+0.0059`;
- median coverage difference: `+0.0098`;
- median ED2 difference: `-0.0093`;
- random-bijection one-step precision was often higher because broad initial
  particles underwent a larger contraction toward the target cloud.

Those numbers reinforce that one-step support from a broad initialization is
not the right selection criterion for a route.

## Correct program state

Verified:

1. averaged EST is a valid coupling;
2. barycentric target averaging can leave structured support;
3. modal routing removes target averaging but loses target marginals;
4. sparse balanced EST preserves exact marginals on the EST support;
5. sparse and dense EST solvers maximize the same consensus objective;
6. sparse EST is faster than dense Euclidean Hungarian assignment at the
   tested size.

Not verified:

1. better neural amortization;
2. lower inference cost;
3. better official-style paper results;
4. persistence under fresh latent samples;
5. higher-dimensional scaling;
6. an encoder-robust model;
7. the partial-transport hypothesis.

## Next gate

The next discriminating development experiment is neural route retention.

Compare:

1. fresh latent cohorts with freshly recomputed balanced EST routes;
2. a persistent latent cohort with persistent routes;
3. minimum-Euclidean assignment as a low-route-cost teacher;
4. random bijection as an exact-marginal but incoherent control.

Measure:

- held-out ED2 and SW1;
- support precision/coverage/leakage;
- route switching;
- train-cohort endpoint retention;
- fresh-latent generalization;
- teacher route cost;
- wall time and memory.

Only if a coherent teacher survives fresh-latent neural amortization should the
program return to adaptive partial mass, semantic features, or image-scale
work.
