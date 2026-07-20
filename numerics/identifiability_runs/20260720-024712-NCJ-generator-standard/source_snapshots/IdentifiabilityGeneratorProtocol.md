# Frozen learned-generator transfer protocol

Status: **frozen before the standard generator run**.

Parent particle result:
`identifiability_runs/20260720-011000-NCJ-test-standard/e4_gate.json` records
`PASS=true`.  Validation selected `eta=0.0525`, clip `2.0`, and
`sigma/tau=0`.  Consequently the transferred method is normalized,
cross-fitted drifting (NCF); symmetric jitter was tested and rejected rather
than silently retained in the name.

This protocol governs only low-dimensional synthetic learned generators.  It
does not support an ImageNet, feature-encoder, or high-dimensional image claim.

## 1. Fixed model and optimization

Each target uses a two-hidden-layer MLP

```text
z -> tanh(32) -> tanh(32) -> x in R^d
```

with latent dimension `max(2,d)`.  Parameters use paired Xavier-normal
initialization.  Every arm uses Adam with

```text
learning rate = 0.0025
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
```

for 800 standard-profile updates.  Query and target minibatches both have
size 64.  The fixed drifting bandwidth is `tau=0.35`.

The training loss has the paper's stop-gradient semantics:

```text
L = mean ||x - stop_gradient(x + V)||^2.
```

The implemented parameter gradient is therefore proportional to
`-J_G(z)^T V`; the field is never differentiated through.  The irrelevant
common factor two is absorbed into the fixed learning rate.

The standard profile uses 12 paired seeds per target/initialization cell.
The smoke profile (`50` steps, batches `32`, one seed) is software validation
only and cannot support a scientific conclusion.

## 2. Initialization regimes

Four paired parameter initializations are fixed:

* `broad`: centered Xavier generator at target scale;
* `missing`: low-spread generator centered on the first component when a
  component representation exists, otherwise centered at zero;
* `far`: low-spread generator translated `3.25` target scales along the all-
  ones direction;
* `concentrated`: nearly collapsed centered generator.

Target structure is used only to define the declared stress initialization
and oracle evaluation metrics.  It never enters a field, gain, update, or
stopping decision.

## 3. Frozen target registry

The executed registry is `identifiability_generator_registry.json`.  It uses
parameters absent from both particle registries and contains eight targets:
unimodal Gaussian, 1-D and 2-D unequal Gaussian mixtures, ring, moons,
banana, 5-D Student, and 5-D spherical shell.

The registry and this protocol are hashed and source-snapshotted by the
runner.  The registry hash is inserted into the runner before its first
standard execution.  No target is a tuning target: the particle-frozen policy
is evaluated directly.

## 4. Arms and compute matching

Seven paired arms are run:

| Arm | Gain | Negative reference | Mask | Extra generator forward |
|---|---|---|---|---|
| `paper` | paper `P*Q` | reused queries | on | no |
| `paper-matched` | paper `P*Q` | reused queries | on | independent unused reference |
| `normalized-only` | constant | reused queries | on | no |
| `crossfit-only` | paper `P*Q` | independent generated batch | off | yes |
| `jitter-only` | paper `P*Q` | reused queries | on | no; sigma is frozen at zero |
| `normalized-crossfit` | constant | independent generated batch | off | yes |
| `ncj` | constant | independent generated batch | off | yes; sigma is frozen at zero |

`paper` and `paper-matched` have identical mathematical updates.  The latter
performs and discards the independent reference forward used by NCF, so it is
the primary compute-matched control.  All arms use the same affinity
kernel-pair budget.  `jitter-only = paper` and
`ncj = normalized-crossfit` are required exact regression checks after the
validation choice `sigma=0`.

## 5. Pairing and measurements

For each `(target, initialization, seed)`, all arms receive identical initial
parameters, target minibatches, query latents, reference latents, final
latents, evaluation references, and sliced-Wasserstein projections.  An arm
that does not need a reference forward does not consume or mutate its stream.

Primary metric: final energy distance squared (ED2).  Required secondary
outputs are sliced Wasserstein-1, mixture coverage/mass error where defined,
nonparametric support coverage, on-support field residual, first-passage time
with censoring, divergence, particle spread, mechanism diagnostics, generator
forward count, kernel-pair count, and setup-inclusive wall time.  The event
threshold remains `ED2 <= 0.05 * target.scale`, evaluated on the current
query batch against a fixed reference.

Each standard artifact contains the same manifest/rows/summary/trajectory/
final-particle/source-snapshot bundle as the particle study.

## 6. Frozen generator gate

Scientific units are target/initialization cells; seeds are repeated paired
runs within cells.  Confidence intervals use the same hierarchical bootstrap
as the particle gate.  NCF passes learned-generator transfer only if all hold:

1. target-balanced geometric-mean `ED2_NCF / ED2_paper <= 0.80`;
2. its hierarchical 95% CI upper endpoint is `< 1`;
3. the hierarchical CI upper endpoint versus `paper-matched` is `< 1`;
4. at least 60% of cell-median ratios versus paper are `< 1`;
5. both Gaussian-mixture and non-Gaussian subgroup CI upper endpoints are
   `< 1`;
6. mixture mode coverage is no lower by more than 0.05 in paired mean, and
   divergence is no more than two percentage points above paper;
7. NCF and paper-matched have identical generator-forward counts, and every
   arm has the same kernel-pair count;
8. `paper = paper-matched = jitter-only` and
   `ncj = normalized-crossfit` in endpoint ED2 to numerical tolerance.

No gate criterion may be changed after the standard run starts.  If this gate
fails, the defensible result remains the already-passing empirical-particle
improvement; no general learned-generator claim is made.

