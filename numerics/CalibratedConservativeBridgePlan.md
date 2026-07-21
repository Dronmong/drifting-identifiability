# Calibrated conservative bridge: implementation and experiment plan

**Date:** 2026-07-21

**Status:** implementation plan only; no performance claim

**Predecessor:**
[`QLDSharpConservativeImplementationPlan.md`](QLDSharpConservativeImplementationPlan.md)

**Evidence base:** the audited S2 run
`conservative_runs/20260721-153656-S2-screen`, summarized in
[`ConservativeFinisherResults.md`](ConservativeFinisherResults.md)

## Executive decision

Do not promote any field from the first conservative-finisher screen. Do not
open its reserved Registry B or C. The registered S2 gate failed because every
candidate violated at least one connected-control guard.

The next experiment will test a narrower and materially different algorithm:

```text
QLD global allocation
        -> short, calibrated sharp-deleted bridge
        -> paper Algorithm-2 stabilization
```

The conservative field is no longer asked to replace the whole final 30% of
training. It is a short local correction inserted between the globally useful
QLD phase and the empirically useful paper phase. It receives a separate,
calibrated optimizer state; the paper phase resumes from the optimizer state
that existed before the bridge.

The primary hypothesis is that the preceding experiment confounded field
quality with an optimizer reset shock. The secondary hypotheses are that exact
deletion is preferable to crossfit at this scale, a fixed absolute bandwidth
is not portable across target geometries, and numerical positivity is too weak
as a finite-particle quality diagnostic.

## 1. Frozen diagnosis from the failed screen

The next implementation must preserve the following facts rather than
reinterpreting the failed run.

### 1.1 Optimizer reset changed the effective algorithm

The carried paper finisher and reset paper finisher used the same field. Their
median effective parameter-step norms during the suffix were approximately:

```text
paper, Adam carried:  0.0064
paper, Adam reset:    0.0255
```

Resetting Adam therefore enlarged the typical parameter displacement by about
four times. The reset paper control then lost to QLD-v1:

```text
ED2 ratio: 1.0443
SW1 ratio: 1.0833
```

All new conservative arms used reset Adam. Their field RMS values differed by
more than an order of magnitude, but reset Adam mapped them to nearly the same
`0.025--0.027` median parameter-step norm. This erased much of the meaningful
scale information carried by the bandwidth and field.

### 1.2 Crossfit was costly and conflicted with QLD

At `tau=0.5`, the independent QLD-gradient diagnostic was negative in about:

```text
sharp crossfit: 26% of recorded comparisons
sharp deleted:  0.6% of recorded comparisons
```

Crossfit also required about 33,280 training generator-example evaluations
per arm versus 25,600 for deleted/paper/QLD arms. It did not produce a
corresponding endpoint gain. Crossfit remains an ablation, not the primary
reference convention.

### 1.3 The conservative fields had useful but transient behavior

Sharp-deleted `tau=0.5` reached simultaneous same-step ratios versus QLD of
approximately:

```text
step 300: ED2 0.931, SW1 0.968
step 400: ED2 1.061, SW1 1.095
```

This does not license per-trial early stopping: the saved evaluation probe was
used to discover the pattern. It does justify a newly registered, globally
fixed short bridge followed by an independent stabilization phase.

### 1.4 One absolute bandwidth could not cover the registry

For sharp deletion, `tau=0.2` helped separated rare mixtures, equal high-mode
mixtures, and heteroscedastic mixtures, but failed on overlapping and
heavy-tailed controls. Larger bandwidth improved the overlap target but harmed
the skew and separated targets. The target scales and local spacings varied
substantially, while the implementation used the same absolute `tau` in raw
coordinates.

### 1.5 A nonzero denominator is not an occupancy certificate

No denominator floor activated, but small-bandwidth denominator minima reached
roughly `1e-5` in recorded trajectories. This can be finite and still have
poor effective sample size. The finite-particle analysis in
[Finite-Particle Convergence Rates](https://arxiv.org/abs/2605.22795) identifies
reciprocal-KDE self-interaction and local occupancy as the relevant controls.

## 2. Literature-aligned corrections

The implementation should reproduce the important stabilization choices used
by the relevant primary sources before concluding that the fields themselves
are ineffective.

- [Drifting Fields are not Conservative](https://arxiv.org/abs/2604.06333)
  uses a sharp/log-KDE objective with detached negative references and exact
  diagonal exclusion. Its reported configurations also use scale-normalized
  kernel widths, optimizer ramp-up, and gradient clipping; its larger setup
  uses a reference memory bank.
- [Kernel-Gradient Drifting Models](https://arxiv.org/abs/2605.10727) uses
  substantially larger synthetic batches and an explicit step cap or gradient
  clipping in its reported controlled experiments.
- [Finite-Particle Convergence Rates](https://arxiv.org/abs/2605.22795)
  explains why reciprocal KDE mass and local occupancy, not mere positivity,
  govern the finite-particle correction.
- [Gradient Flow Drifting](https://arxiv.org/abs/2603.10592) motivates mixed
  divergence objectives, but this remains a later branch. It must not be
  introduced until the simpler optimizer and scale confounds are resolved.

These papers motivate the changes; their reported results do not imply that
the changes will transfer to this repository's one-dimensional `TanhMLP`.

## 3. Primary algorithm: QLD calibrated sharp bridge

Let the full horizon be `T`, initially `T=400` on the existing mechanism
registry. Define:

```text
QLD prefix:       steps 1 ... round(0.70 T)
sharp bridge:     next 20 updates
paper stabilizer: all remaining updates
```

For `T=400`, this is `280` QLD updates, `20` sharp updates, and `100` paper
updates. Twenty is fixed because it corresponds to the pre-observed step-300
checkpoint. It is not selected separately for each target or seed.

The primary bridge field is:

```text
field:          sharp-normalized Laplace
reference:      reused_deleted
raw bandwidth:  tau = 0.5 for the optimizer-calibration stage
optimizer:      separate sharp Adam state
sharp LR:       0.25 * repository Adam LR
warm-up:        linear over all 20 sharp updates
paper state:    restore the preserved pre-bridge QLD Adam state
```

The paper stabilizer begins from the generator parameters produced by the
sharp bridge, but its Adam moments and step index are restored from the saved
QLD/paper state. This is a genuine dual-state switch:

```text
parameters       shared and continuously updated
QLD/paper state  saved at bridge entry, restored at bridge exit
sharp state      independent; discarded or archived at bridge exit
```

The sharp optimizer learning rate on bridge update `j in {1,...,20}` is:

```text
lr_sharp(j) = base_lr * 0.25 * j / 20.
```

No per-target adjustment is allowed in this first causal experiment.

## 4. Parameter-step trust control

The runner must support an optional global-norm trust cap on the actual Adam
parameter displacement, applied after moment/bias correction but before the
parameters are committed.

At the QLD/sharp handoff:

1. clone the saved prefix;
2. run five paper updates on a dedicated calibration stream;
3. record the median flattened parameter-step norm `s_paper`;
4. discard the calibration clone;
5. set the sharp bridge cap to `1.25 * s_paper`.

The primary calibrated arm uses both LR warm-up and this cap. The calibration
stream is independent of training, event, selection, and final evaluation
streams. Its samples and generator evaluations are charged to the work ledger.

This rule is target-adaptive but label-free and predeclared. It calibrates the
optimizer geometry, not the observed performance metric. No ED2, SW1,
component label, or target family may enter the cap.

Record for every sharp update:

```text
uncapped parameter-step norm
capped parameter-step norm
cap value
whether the cap activated
raw gradient norm
field RMS
cosine(-raw gradient, committed parameter step)
```

Frequent cap activation is diagnostic evidence that the nominal sharp
optimizer remains mismatched. It is not silently acceptable.

## 5. Exact optimizer implementation

Do not modify `TanhMLP.stopgrad_step` or the historical QLD/paper baselines.
Add a separate configurable Adam update used only by experimental arms.

Suggested API:

```python
adam_step_configurable(
    model,
    grads,
    *,
    learning_rate,
    maximum_parameter_step=None,
) -> AdamStepDiagnostics
```

`AdamStepDiagnostics` should contain the pre-cap and post-cap flattened step
norms, activation flag, and per-parameter deltas. Its computation must agree
bitwise with the repository Adam update when the base learning rate is used
and no cap is supplied.

Required invariants:

1. no-cap/base-LR is bitwise equal to `_adam_step`;
2. multiplying LR by `c` multiplies the pre-cap first update by `c`;
3. a triggered cap produces the requested global norm;
4. an inactive cap changes no value;
5. the parameter delta has positive cosine with the declared descent
   direction unless the direction is zero;
6. saving/restoring the paper state is lossless;
7. sharp updates do not mutate the saved paper state;
8. restoring paper state after the bridge gives the same buffers and step
   index as the pre-bridge snapshot;
9. all work associated with the five-update calibration clone is counted.

## 6. Causal arm set for the optimizer bridge screen

Use the existing exposed Registry A only for this mechanism diagnosis. Do not
generate or inspect a new registry until one bridge passes the guards below.

The first screen contains exactly these arms:

| arm | schedule | sharp optimizer | purpose |
|---|---|---|---|
| `qld-v1` | 70% QLD, 30% paper | none | registered baseline |
| `qld-full` | 100% QLD | none | global-only baseline |
| `bridge-reset-full-lr` | QLD, 20 sharp, paper | reset, base LR, no warm-up | reproduce optimizer shock |
| `bridge-reset-quarter` | QLD, 20 sharp, paper | reset, 0.25 LR, no warm-up | isolate LR reduction |
| `bridge-warm-quarter` | QLD, 20 sharp, paper | reset, 0.25 LR, linear warm-up | isolate warm-up |
| `bridge-calibrated` | QLD, 20 sharp, paper | warm-quarter + trust cap | primary candidate |
| `bridge-carry-copy` | QLD, 20 sharp, paper | copy of QLD state; original preserved | test whether reset is necessary |
| `qld-sharp20-only` | QLD, 20 sharp, then stop | calibrated | transient mechanism diagnostic only |

All bridge arms use sharp deletion and raw `tau=0.5`. Do not add crossfit,
other bandwidths, multiscale kernels, phase-duration sweeps, or mixed
divergences to this causal screen.

The `qld-sharp20-only` arm has fewer updates and is not eligible for the main
endpoint selection. It exists only to verify that the bridge itself reproduces
the previously observed transient.

## 7. Scale-normalized sharp field

Only after one calibrated raw-bandwidth bridge passes the optimizer mechanism
guards should scale normalization be introduced.

### 7.1 Frozen target-only scale

Draw a dedicated target-only calibration sample of size at least 4,096. In one
dimension:

1. sort the sample;
2. compute each point's distance to its fifth nearest neighbor;
3. winsorize these distances at the 5th and 95th percentiles;
4. take their mean as `s_knn`;
5. require `s_knn` finite and strictly positive;
6. freeze it for the entire trial.

Evaluate kernels using normalized positions `x/s_knn` and `y/s_knn` with a
globally fixed dimensionless bandwidth. The same `s_knn` is used for target
and generated samples. Generated samples must not influence it. Record the
sample, seed, estimate, and hash in the artifact.

This follows the scale-invariance motivation of the top-k EMA rule in the
sharp-normalization paper while avoiding component labels and per-step target
leakage. An EMA is unnecessary for the stationary synthetic targets.

### 7.2 Scale experiment

Compare only:

```text
calibrated raw tau=0.5 bridge
calibrated scale-normalized bridge at one frozen dimensionless tau
qld-v1
```

Select the dimensionless bandwidth on Registry A using a small declared grid
of at most three values. Freeze one global rule before any fresh development
registry is run. A per-target oracle is diagnostic only and cannot be the
candidate.

## 8. Conservative multiscale sharp kernel

Introduce this only if calibrated scale normalization is stable but still
shows the observed narrow-versus-broad bandwidth tradeoff.

For positive weights `a_l` and positive scales `tau_l`, define the normalized
sharp mixture:

```text
H(r) = sum_l a_l * (1 + r/tau_l) * exp(-r/tau_l)
```

Its gradient with respect to the anchor is:

```text
grad_x H(|x-y|)
  = sum_l a_l * exp(-|x-y|/tau_l) * (y-x) / tau_l^2.
```

The score for a cloud `Y` is:

```text
S_multi,Y(x)
  = mean_y grad_x H(|x-y|) / mean_y H(|x-y|),
```

and the field is `S_multi,pos - S_multi,neg`. Use exact deletion for the
negative cloud. Positive global weights preserve a genuine scalar KDE score;
do not use position-dependent mixture weights.

The initial fixed scale set should be small, for example three dimensionless
scales around the selected single-scale value. Normalize every component as
shown above so the largest `tau` does not dominate merely because the
unnormalized companion kernel carries a `tau^2` factor.

Required invariants:

1. vectorized/slow equality;
2. translation and positive-global-scale invariance;
3. swap antisymmetry and identical-cloud cancellation;
4. exact diagonal deletion;
5. finite-difference agreement with `grad H`;
6. field agreement with the gradient of the multiscale log-KDE loss;
7. reduction to the existing sharp field when one component is used;
8. nonnegative finite mixture denominator on ordinary batches.

Do not formalize a multiscale identifiability theorem until the empirical
candidate is fixed. A later proof must separately establish positivity,
smoothness, and injectivity/characteristicness of the chosen mixture rather
than assuming that a score representation alone proves equality.

## 9. Local occupancy and ESS diagnostics

For every positive and negative KDE row and every kernel component, record:

```text
mass       = sum_j w_j
ESS        = (sum_j w_j)^2 / sum_j w_j^2
max share  = max_j w_j / sum_j w_j
neighbor count with w_j >= exp(-3) * max_j w_j
```

For mixtures, also record these quantities for the combined denominator.
Store row minima and the 1st, 10th, 50th, and 90th percentiles at every
trajectory checkpoint.

The first calibrated bridge screen uses ESS only as a diagnostic. Do not gate
or rescale the field from ESS until a predeclared analysis shows that low ESS
predicts harm. A gate that can turn the field off away from the target may
create new stationary points and must be reviewed against the identifiability
guardrails.

If low ESS is predictive, the preferred repair is a larger exact-deleted
reference bank, not crossfit:

- keep the backward/query batch fixed;
- evaluate additional detached generated references;
- charge every forward example and kernel pair;
- compare fresh-bank and short FIFO/EMA-bank variants;
- record reference staleness explicitly;
- never describe a stale memory-bank estimator as unbiased.

## 10. Trajectory and attribution requirements

Retain all trajectory fields from the preceding runner and add:

- phase label in `{QLD, sharp_bridge, paper_stabilizer}`;
- optimizer-state identifier;
- current objective-specific learning rate;
- trust-cap value and activation;
- pre-cap and committed step norms;
- paper-state step index before save and after restore;
- ESS/local-occupancy summaries;
- ED2/SW1 change across the bridge alone;
- ED2/SW1 change across the paper stabilizer;
- whether the paper phase preserved, improved, or erased the bridge gain;
- finisher/paper and finisher/QLD parameter-gradient cosines on independent
  diagnostic batches.

Save trajectories at:

```text
step 0
every 25 QLD updates
the QLD/sharp handoff
every one of the 20 sharp updates
the sharp/paper handoff
the first five paper updates
every 25 later updates
the endpoint
```

Final metrics and any online selection probe must use different streams.

## 11. Advancement gates

The optimizer-calibration mechanism advances from Registry A only if the
primary `bridge-calibrated` arm satisfies all of:

| gate | threshold |
|---|---:|
| ED2 / QLD-v1 | at most `0.98` |
| SW1 / QLD-v1 | at most `0.99` |
| worst connected-control ED2 / QLD-v1 | at most `1.05` |
| each primary initialization ED2 / QLD-v1 | at most `1.02` |
| divergence | none |
| denominator floor activation | none |
| median committed sharp step / carried-paper step | between `0.5` and `1.5` |
| 90th-percentile committed sharp step / carried-paper step | at most `2.0` |
| paper-state restoration invariant | exact pass |

Use target-balanced cell medians and a paired hierarchical bootstrap for the
aggregate ratios. Registry A remains a development registry; passing these
gates only permits creation of a fresh bridge-development registry.

If no optimizer-calibrated arm passes, stop. Do not add bandwidth, multiscale,
memory-bank, duration, and mixed-divergence changes simultaneously. The result
would then support the conclusion that the transient cannot be converted into
a stable bridge by optimizer calibration alone.

## 12. Fresh-registry sequence

### Bridge Registry A

Use the existing `conservative_registry_a.json` only for optimizer and scale
mechanism work. Its outcomes are already exposed.

### Bridge Registry B

Generate at least 16 fresh targets only after one complete bridge candidate is
frozen. Match the same predefined families with different locations, weights,
scales, tail parameters, and overlap. Hash the resolved JSON. Reject canonical
duplicates from every prior registry.

Minimum profile:

```text
updates          = 1,200
ordinary batch   = 128
paired seeds     = 8
endpoint samples >= 4,096
trajectory every = 25, plus every bridge update
```

Required baselines:

- paper Algorithm 2 at frozen bandwidth;
- QLD-v1;
- full QLD;
- the frozen calibrated bridge;
- the same bridge without the sharp phase, with compute matched where useful.

### Bridge Registry C

Do not generate or inspect confirmation targets until the frozen candidate
passes every Registry-B gate. After C is opened, no architecture, scale rule,
optimizer rule, bridge length, threshold, or baseline may be repaired.

## 13. Work and fairness accounting

Continue reporting standalone deployable cost and physical common-prefix
research cost separately. Add:

- optimizer-calibration clone updates;
- scale-calibration target samples and sort/kNN work;
- sharp and paper optimizer-state storage/restoration;
- extra reference-bank forwards;
- per-component multiscale kernel pairs;
- ESS diagnostic reductions;
- every metric, selection, and final-evaluation probe.

The calibrated bridge should remain below the existing advancement ceiling of
two times the paper generator evaluations. A candidate that wins only by using
an unreported large reference bank fails the audit.

## 14. Explicitly prohibited shortcuts

Do not:

- select the saved best step separately for each trial;
- use endpoint ED2 or SW1 to set the trust cap;
- tune bridge duration, LR, bandwidth, and scale rule in one combinatorial
  sweep;
- re-promote crossfit without new evidence;
- use component labels for scale, bandwidth, ESS, optimizer, or stopping;
- alter the historical QLD-v1 or paper implementations;
- call a positive denominator a finite-particle stability certificate;
- claim that a conservative or identifying field is automatically a descent
  direction for ED2, SW1, or neural-generator parameters;
- claim that a mixture of individually identifying fields is identifying
  without a proof excluding cancellation;
- open Registry B after a failed Registry-A gate;
- formalize empirical optimizer behavior as a theorem.

## 15. Proposed files and artifacts

```text
numerics/CalibratedConservativeBridgePlan.md       this plan
numerics/calibrated_bridge.py                      optimizer/phase helpers
numerics/run_calibrated_bridge_development.py      staged runner
numerics/CalibratedBridgeProtocol.md               frozen protocol
numerics/CalibratedBridgeResults.md                cumulative result
numerics/bridge_candidate.json                     created only after a pass
numerics/bridge_registry_b.json                    created only after freeze
numerics/bridge_runs/<timestamp-stage>/            immutable artifacts
```

Each run directory should retain:

```text
manifest.json
source_hashes.json
source_snapshots/
invariants.json
rows.csv
trajectories.csv
work_ledger.json
optimizer_diagnostics.json
occupancy_diagnostics.json
summary.json
RESULTS.md
```

## 16. Concrete implementation order

1. Commit this plan before implementing outcome-sensitive choices.
2. Refactor no historical baseline; add a separate configurable Adam step.
3. Implement optimizer save/restore, LR warm-up, trust cap, and all invariants.
4. Add the three-phase runner using the existing Registry-A streams and
   common-prefix infrastructure.
5. Add phase-specific trajectories and ESS/local-occupancy diagnostics.
6. Freeze `CalibratedBridgeProtocol.md` with the causal arm set above.
7. Run a smoke profile and audit exact baseline compatibility.
8. Run the Registry-A optimizer-calibration screen once.
9. Apply the advancement gates without selecting a new arm by hindsight.
10. If optimizer calibration passes, implement and separately screen target
    scale normalization.
11. If a bandwidth tradeoff remains, implement the conservative multiscale
    kernel and its invariant suite.
12. Freeze exactly one candidate, then generate and hash a fresh Registry B.
13. Run Registry B once under the standard profile.
14. Only after a full Registry-B pass, freeze confirmation and generate
    Registry C.
15. Begin Lean formalization only for the empirically frozen field, while
    keeping stochastic optimizer claims outside the theorem.

## Bottom line

The first conservative-finisher experiment did not show a general endpoint
improvement, but it did isolate a plausible integration failure: reset Adam
made the new field's effective parameter step roughly four times the carried
paper step, while an overly long sharp-only suffix erased useful intermediate
behavior. The next experiment should test that diagnosis directly with a
short, optimizer-calibrated, exact-deleted bridge and a restored paper
stabilizer. Scale normalization, multiscale smoothing, and larger reference
banks are ordered follow-ups, not simultaneous rescue knobs.
