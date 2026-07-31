# Conditioned-transport repair: deeper failure diagnosis

**Status:** post-hoc diagnosis complete; artifacts consumed  
**Date:** 2026-07-23  
**Parent result:** `ConditionedTransportLimitationRepairResults.md`  
**Machine-readable audit:** `conditioned_transport_repair_failure_diagnosis.json`

## 1. Bottom line

The proposed representative repair did not fail because its tree construction
was wrong. It did exactly what it was designed to do:

- reduced representative radius;
- reduced row- and column-mass error;
- made the compressed local field closer to the dense local field; and
- materially improved the 16-dimensional endpoint.

The failed promotion arose from combining three different effects:

1. **There was little approximation headroom in 2D.** Both compressed fields
   were already almost identical to the dense field, so changing the tree
   mostly perturbed a nonlinear training trajectory.
2. **There was real approximation headroom in 16D.** The variance tree and
   larger capacity repaired it, and all 16D target-reduced ED2 and SW1
   comparisons favored the repair.
3. **The rare-mode metric overstated support recovery in both models.** It
   counted bridge/tail points as rare-mode samples. Neither conditioned model
   actually formed the isolated 16D rare cluster.

The most important correction to the previous interpretation is therefore:

> The representative repair improved the high-dimensional bulk field, but
> representative accuracy was not the main rare-mode bottleneck. The
> one-step teacher itself almost never reached the genuine rare component.

Further tree tuning alone is unlikely to fix this. The next useful experiment
should change the teacher rollout and tail amortization, while retaining the
high-dimensional representative improvement.

## 2. Evidence boundary

This document is an explanatory analysis of the already-consumed frozen
registry and saved outputs:

- concentrated:
  `conditioned_transport_runs/20260723-144129-consumed/`;
- broad:
  `conditioned_transport_runs/20260723-144337-consumed/`.

The subgroup analyses, trajectory reconstructions, and proposed switch rule
are post-hoc design evidence. They are not a new confirmation and cannot be
reported as a promoted performance result.

The reproducible saved-output analysis is:

```powershell
uv run --with numpy --with matplotlib python `
  numerics/analyze_conditioned_transport_repair_failure.py
```

It emits a hashed JSON report and two figures under
`conditioned_transport_repair_failure_figures/`.

## 3. What the repair changed

The current control uses:

- a fixed-level tree;
- `M=128` representatives;
- seven predetermined split directions; and
- fixed local weight `0.25`.

The repaired candidate uses:

- a per-node maximum-variance registered direction;
- `M=128` in 2D and 4D;
- `M=256` in 8D and 16D; and
- the same fixed local weight `0.25`.

Thus the comparison has two regimes:

- in 2D/4D it isolates **partition strategy**;
- in 8D/16D it changes both **partition strategy and capacity**.

The local field is RMS-normalized to the global correction before multiplying
by `0.25`. Better approximation therefore does not produce a larger local
step. It mainly rotates the local correction toward the dense field.

## 4. The dimension split is the central result

Fresh candidate/control ratios, treating each initialization row separately,
were:

| Dimension | ED2 ratio | ED2 wins | SW1 ratio | SW1 wins | Field-error ratio |
|---:|---:|---:|---:|---:|---:|
| 2 | `1.0646` | `7/16` | `1.0238` | `7/16` | `0.6524` |
| 4 | `0.9990` | `10/16` | `0.9947` | `10/16` | `0.7725` |
| 8 | `0.9687` | `10/16` | `0.9893` | `7/16` | `0.4884` |
| 16 | `0.9491` | `16/16` | `0.9766` | `14/16` | `0.4937` |

After reducing concentrated and broad initialization within target, all eight
16D targets favored the repair on both ED2 and SW1.

This is not a random global null result. It is a crossover:

```text
low dimension: dense-field approximation is already easy
high dimension: historical compression is a real bottleneck
```

The consumed strategy-development artifacts reinforce this. In 16D, endpoint
quality generally improved as the local field moved toward dense:

| Local field | ED2/fixed M128 | SW1/fixed M128 |
|---|---:|---:|
| variance `M128` | `0.9737` | `0.9874` |
| variance `M256` | `0.9559` | `0.9744` |
| dense `M512` | `0.9416` | `0.9681` |

The same monotone pattern did not hold in 2D, where every field was already
very close to dense.

## 5. Why aggregate field accuracy did not predict endpoints

Across the 64 initialization/target rows, the Spearman correlations were:

| Diagnostic versus candidate/control endpoint ratio | ED2 | SW1 |
|---|---:|---:|
| field-error ratio | `+0.117` | `+0.173` |
| field-cosine gain | `-0.384` | `-0.267` |
| row-mass-error ratio | `+0.299` | `+0.220` |
| column-mass-error ratio | `+0.403` | `+0.330` |

Lower endpoint ratios are better. Directional cosine improvement was more
informative than aggregate relative-L2 error, but none of these diagnostics
was a reliable endpoint predictor.

There are four reasons.

### 5.1 The diagnostic was bulk weighted

Field relative-L2 error flattens all 512 particles and all coordinates. A 5%
tail can be poor while the 95% bulk dominates the reported improvement.

### 5.2 Only the first two field calls were audited

The frozen protocol used two dense diagnostic calls per target, while the
local field was used for 20 macro-steps. A targeted rerun found that the
variance field remained more accurate later, but the original gate did not
measure trajectory-wide or tail-stratified error.

### 5.3 RMS normalization removes magnitude information

For both trees, the local field is rescaled to the global correction's RMS.
The mean teacher-displacement ratio stayed near one. The repair changed the
direction of a fixed-strength auxiliary term; it did not simply apply a
stronger version of a known descent step.

### 5.4 The amortized optimization is nonsmooth and path dependent

Every macro-step:

1. ranks a fresh 512-particle population;
2. constructs a frozen teacher;
3. performs eight sequential 64-example Adam updates; and
4. reranks a new latent cohort at the next macro-step.

Small changes can swap projected ranks at the next step. Later student
microsteps fit a teacher computed before earlier optimizer updates. Tiny
teacher rotations can therefore produce different nonlinear network
trajectories even when both one-step teachers are excellent.

This also explains the failed cross-fitted scalar controller. Its estimated
one-step improvement had essentially no rank correlation with the final
endpoint. On nonlinear targets, a small reduction from weight `0.25` was
sometimes amplified into a large final regression.

## 6. What happened in the two worst 2D cases

The two target-reduced 2D rare-GMM ratios were:

| Target | ED2 ratio | SW1 ratio |
|---|---:|---:|
| `NPR-d2-rare-gmm-v00` | `1.4319` | `1.1492` |
| `NPR-d2-rare-gmm-v01` | `1.1552` | `1.0869` |

For broad-init `v00`, this was not loss of rare nearest-center count:

| Output | Common count | Rare-nearest count |
|---|---:|---:|
| target reference | `1943` | `105` |
| repaired | `1934` | `114` |
| current control | `1932` | `116` |
| paper port | `2048` | `0` |

The main bulk difference was common-component calibration:

- repaired common-component mean error: `0.0694`;
- current common-component mean error: `0.0341`;
- repaired relative common covariance error: `0.2152`;
- current relative common covariance error: `0.1876`.

Both conditioned models generated a diffuse path toward the rare center
rather than a clean compact rare component. Their rare-assigned covariance
errors were over `5.7x` the reference rare covariance.

The large ED2 ratio also exaggerates the absolute change because the control
endpoint was already very small:

- repaired ED2: `0.00520`;
- current ED2: `0.00307`.

This is a real regression, but it is a low-dimensional bulk-shape regression,
not evidence that the variance tree deleted rare target observations.

## 7. The representative tree did not mix away the rare target

Across the 20 actual active-direction batches of the broad 16D rare target:

| Positive representative tree | Leaves touching rare points | Pure rare leaves | Mixed leaves | Mean RMS radius |
|---|---:|---:|---:|---:|
| fixed `M128` | `11.10` | `2.90` | `8.20` | `2.82` |
| variance `M256` | `14.05` | `13.10` | `0.95` | `2.01` |

The variance tree isolated the rare target observations much more cleanly.
The rare failure therefore did not arise because the new positive tree
averaged rare samples into common representatives.

The remaining approximation error comes from the full normalized
interaction:

- positive representatives;
- generated/negative representatives;
- row normalizers;
- column normalizers; and
- global RMS rescaling.

Correct positive mass alone does not certify a correct per-particle tail
vector.

## 8. The frozen mode-coverage metric was misleading

The existing metric labels a generated point by whichever component center is
closer, then declares a mode covered if assigned mass is at least half the
true component weight.

On broad-init 16D `v00`:

- target rare-nearest count: `108`;
- repaired rare-nearest count: `49`;
- current rare-nearest count: `81`;
- threshold: at least `52` points.

The repaired model failed the binary threshold by only three points. That
explains the abrupt reported coverage change from `1.0` to `0.5`.

But this apparent near miss hides the more important defect. Define the rare
core using the 95th percentile of target-reference distance to the rare
center:

| Output | Rare-nearest count | Rare-core count |
|---|---:|---:|
| target reference | `108` | `102` |
| repaired | `49` | `0` |
| current control | `81` | `1` |
| paper port | `0` | `0` |

Both conditioned models missed the actual isolated 16D rare cluster. The
current control merely placed more bridge/tail points beyond the midpoint
between centers.

The same issue exists in 2D:

| Output | Rare-nearest count | Rare-core count |
|---|---:|---:|
| target reference | `105` | `99` |
| repaired | `114` | `46` |
| current control | `116` | `47` |
| paper port | `0` | `0` |

Thus nearest-center coverage overstated rare-mode quality even when its binary
gate passed.

The global support precision/recall metric was also too coarse for this
question. In 16D its target-calibrated radius was large enough for bridge
points to provide high apparent recall.

## 9. What the full dense field reveals

A targeted deterministic reconstruction used the exact dense 512-point local
field on broad-init 16D rare `v00`.

Final diagnostics were:

- ED2: `0.02779`;
- SW1: `0.08730`;
- rare-nearest count: `58`;
- rare-core count: `0`.

For comparison:

| Local field | ED2 | SW1 | Rare-nearest | Rare-core |
|---|---:|---:|---:|---:|
| fixed `M128` | `0.03121` | `0.09201` | `81` | `1` |
| variance `M256` | `0.02850` | `0.08862` | `49` | `0` |
| dense `M512` | `0.02779` | `0.08730` | `58` | `0` |

This separates two facts:

1. approaching the dense local field improves the bulk ED2/SW1 endpoint in
   this high-dimensional case;
2. even the exact dense local field does not create the genuine rare mode.

The fixed tree's larger nearest-center count was therefore an accidental bias,
not evidence that its local field was more correct.

## 10. The teacher, not only the neural student, missed the rare core

A trajectory reconstruction counted rare-core particles in every
512-particle run population and its one-step teacher.

For fixed, repaired, and dense local fields:

- 19 of 20 teacher batches contained zero rare-core particles;
- the remaining batch contained only one;
- the final teacher batch contained zero rare-core particles for all three.

At macro-step 20:

| Local field | Run rare-nearest | Teacher rare-nearest | Teacher rare-core |
|---|---:|---:|---:|
| fixed `M128` | `14` | `21` | `0` |
| variance `M256` | `11` | `14` | `0` |
| dense `M512` | `11` | `16` | `0` |

The neural student was not being handed a well-formed rare cluster and then
forgetting it. The one-step teacher itself supplied a bridge.

The architecture explains why:

- each macro-step uses a fresh latent cohort;
- a cohort receives only one half-step (`particle_step = 0.5`);
- the 5% isolated mode is far from the common component;
- the global projected correction is averaged across directions;
- the local density-weighted field is dominated by the 95% common mass; and
- uniform distillation gives rare-seeking particles little gradient weight.

The generator is also a continuous map from a connected Gaussian latent
space. It can approximate disconnected modes only through a thin bridge and
must learn a sharp low-probability routing region for a 5% distant component.
That is a difficult finite-budget optimization problem.

## 11. Revised interpretation of the failed promotion

The honest conclusion is more positive and more specific than “the repair did
not work”:

- **Representation compression:** repaired successfully.
- **High-dimensional bulk distribution quality:** consistently improved.
- **Low-dimensional replacement:** unnecessary and sometimes harmful.
- **Rare nearest-center gate:** brittle and scientifically inadequate.
- **Actual rare-core recovery:** still unsolved by current, repaired, and
  dense local-field variants.
- **Cross-fitted scalar control:** too short-horizon for this path-dependent
  amortized process.

The post-hoc rule “use fixed below 8D and repaired at 8D/16D” would reconstruct
ratios of approximately:

- ED2: `0.9792`;
- SW1: `0.9914`.

This is design evidence only. It must be tested on a new registry before any
claim.

## 12. Recommended next experiment

Do not spend the next experiment on another representative tree. Preserve:

- fixed `M128` in low dimension;
- variance `M256` when the fixed tree's geometry is demonstrably difficult;
- the exact persistent target atlas; and
- the current audited cost ledger.

Then test a **persistent free-particle teacher rollout**.

### 12.1 Teacher rollout

For each 512-particle latent cohort:

1. compute the current generator features;
2. rerank against the target atlas;
3. take `K` free-particle transport substeps on the same features, with
   `K in {1, 2, 4}`;
4. recompute ranks after every substep;
5. either disable the local term on globally extreme particles when it
   opposes the global correction, or project its anti-aligned component out;
6. amortize only the final rolled-out teacher.

This directly tests whether repeated coherent transport can reach the rare
core before neural fitting.

### 12.2 Tail-balanced amortization

During student fitting:

1. rank teacher particles by displacement magnitude and projected
   extremeness;
2. stratify every microbatch so high-displacement/tail teacher particles are
   represented;
3. use inverse-stratum weights so the global objective remains calibrated;
4. report both unweighted and weighted student residuals; and
5. check whether rare-core particles present in the teacher survive in the
   generator output.

This tests a different failure boundary:

```text
teacher reaches rare core, student loses it
```

versus:

```text
teacher never reaches rare core
```

### 12.3 Architecture escalation

If a multi-step teacher forms a rare core but a continuous single-head
generator cannot retain it, then test:

- a discrete latent code;
- a small mixture-of-experts generator; or
- target-only cluster-conditioned latent routing.

Synthetic component labels may be used for evaluation, but not for training
or route selection.

## 13. Required metric repair

The next protocol should replace binary nearest-center coverage with:

1. target-calibrated component-core mass;
2. component-conditioned mean and covariance error;
3. bridge occupancy in target-low-density regions;
4. nearest-center mass only as a secondary diagnostic;
5. ED2 and held-out SW1 for bulk quality; and
6. per-dimension reporting rather than only an aggregate ratio.

For unlabeled targets, fit any clustering or density threshold on an
independent target-only calibration split and freeze it before model outputs
are evaluated.

## 14. Decision

The high-dimensional representative repair is worth retaining as a component,
but it should not be promoted as a universal replacement. The next bottleneck
is not local-kernel approximation. It is long-range, low-mass occupancy in the
teacher dynamics and its neural amortization.

The most informative next test is therefore:

> Can a repeated same-particle PSQT teacher reach a genuine rare core, and can
> tail-balanced amortization preserve it?

That experiment has a clear causal interpretation whichever way it fails.

## 15. Follow-up outcome

The experiment above has now been implemented and freshly confirmed; see
`AdaptiveRolloutConfirmationProtocol.md` and
`AdaptiveRolloutConfirmationResults.md`.

The decisive repair was a dimension-adaptive persistent teacher: retain one
step in 2D/4D, rerank the same free particles twice in 8D, and four times in
16D. Against the prior fixed control, the unseen 32-target confirmation
produced target-level geometric ED2/SW1 ratios `0.8570 / 0.9243`, with both
95% bootstrap upper bounds below one. Genuine rare-core mass increased at 8D
and 16D, while the low-dimensional endpoints remained exactly unchanged.

Tail-balanced amortization and per-particle local safety were implemented and
tested but were not promoted: they added cost without a consistent endpoint
gain. The post-hoc component-conditioned moment audit is recorded in
`adaptive_rollout_component_moments.json`.
