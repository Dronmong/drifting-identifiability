# QLD sharp-conservative successor: implementation and evaluation plan

**Date:** 2026-07-21

**Status:** implementation plan only; no result or performance claim

**Input evidence:** completed `QLD-confirmatory-v1`, `LBQCD-confirmatory-v1`,
and failed `OASQD-development-v1` campaigns

**Primary objective:** produce a learned-generator method that improves
reliably over both the paper Algorithm-2 implementation and the repository's
QLD-v1 baseline on fresh, low-dimensional synthetic targets, while retaining
an auditable population target and honest compute accounting.

## Executive decision

Do not continue tuning occupancy thresholds or virtual rank-batch sizes. The
next campaign will test whether the **local finisher**, rather than the global
rank phase, is now the limiting component.

The primary proposed method is **QLD with a sharp-normalized conservative
Laplace finisher**, abbreviated **QLD-SC**:

1. use ordinary one-dimensional QLD for global mass allocation;
2. checkpoint the generator at the phase boundary;
3. replace the paper's bi-softmax Algorithm-2 finisher by a
   sharp-normalized Laplace score field;
4. isolate the finisher from stale QLD Adam moments;
5. compare this against corrected mean-shift, kernel-gradient, full QLD, and
   the unchanged paper finisher from the exact same prefix.

This is a field-design experiment, not another estimator-resolution
experiment. The intended division of labor is:

- QLD supplies long-range, scale-free monotone transport;
- the sharp-normalized field supplies a local conservative score correction;
- independent or leave-one-out negative references prevent a trivial self
  term from controlling the normalizer;
- objective-specific optimizer state prevents the QLD gradient history from
  distorting a qualitatively different finishing field.

Standalone sharp normalization and kernel-gradient drifting are not novelty
claims. They are contemporary methods and must be cited as such. A potentially
new contribution would be the specific QLD/sharp hybrid, the common-prefix
optimizer diagnosis, and a formally guarded zero-set-preserving design. A
dedicated novelty review is still required before making that claim.

## 1. Frozen evidence and forensic diagnosis

Do not change, overwrite, or retune any sealed QLD, LB-QCD, or OA-SQD result.
The authoritative OA-SQD account remains
[`OASQDDevelopmentResults.md`](OASQDDevelopmentResults.md).

The frozen O5 numbers are:

| comparison | result |
|---|---:|
| OA-SQD / paper `.5` ED2 | `.8448` |
| OA-SQD / QLD-v1 ED2 | `1.0031` |
| bootstrap interval vs QLD | `[.9904, 1.0198]` |
| OA-SQD / paper SW1 | `.8669` |
| generator evaluations / paper | `1.4435` |
| total global updates | `1,200` |

A post-run, read-only trial-level audit adds the following mechanism facts:

- only `24/256` candidate trials routed to a global update;
- all `232` non-routed trials were exactly equal to QLD in ED2, SW1, and
  mass-L1;
- every routed trial used exactly `50` global updates;
- among routed trials there were `11` wins and `13` losses against QLD;
- the routed-trial geometric-mean ED2 ratio was `1.0481`.

The aggregate tie therefore arose mainly because the candidate was QLD in
`90.625%` of trials. When the new mechanism acted, it did not show a stable
benefit.

### 1.1 Why the occupancy gate is not the next repair

The final edge gate clears every active deficit when the generated median is
inside a central target-quantile interval. A generator can completely miss a
small tail mode while retaining a central median, so this gate suppresses the
defect it was intended to detect. With two-check activation, two-check
clearing, a 25-update check period, and `one_shot=True`, the nominally adaptive
procedure also collapses to a single 50-update intervention.

This can be repaired mechanically, but it is not the main performance path.
Ungated full RSR, fixed RSR, periodic corrections, restored gradient noise,
and adaptive virtual resolution were already tied with or worse than QLD at
standard scale. They estimate the same rank-transport direction more
accurately; they do not introduce a better descent field.

Retain these OA-SQD products as infrastructure:

- the independent, fair event-probe stream;
- the complete work ledger;
- the sample-only target atlas as a diagnostic;
- the conditionally unbiased stratified backward estimator;
- RNG stream separation and source snapshots.

Do not promote the current median edge gate or one-shot controller into the
new algorithm.

### 1.2 Current learned-generator bottlenecks

The next implementation must explicitly investigate four bottlenecks:

1. **Field bottleneck.** The final 30% always returns to the implemented paper
   Algorithm-2 field, even though recent work distinguishes that bi-softmax
   field from corrected population mean shift and from conservative score
   fields.
2. **Parameterization bottleneck.** A useful particle displacement becomes
   `J_theta^T v` in a shared generator. The earlier NCJ particle gain failed
   after this mapping.
3. **Optimizer-state bottleneck.** QLD and the paper field currently share one
   Adam history across a hard objective switch. No reset or objective-specific
   state has been tested.
4. **Observability bottleneck.** Existing runners store endpoints but not
   enough phase-boundary trajectories to determine whether the paper phase
   repairs or erases the QLD state.

## 2. Literature-backed design boundary

The following sources motivate the candidate set, but none substitutes for a
matched repository experiment.

1. [Drifting Fields are not Conservative](https://arxiv.org/abs/2604.06333)
   identifies the position-dependent normalization as the source of
   non-conservatism for non-Gaussian radial kernels, introduces sharp
   normalization and a log-KDE loss, and reports that the corrected/log-KDE
   variants match or beat the implemented drifting objective in its tests.
2. [Kernel-Gradient Drifting Models](https://arxiv.org/abs/2605.10727)
   replaces Euclidean displacement by kernel gradients, obtains a smoothed
   score field, and reports improvements in most controlled non-Gaussian
   comparisons.
3. [Finite-Particle Convergence Rates for Conservative and Non-Conservative
   Drifting Models](https://arxiv.org/abs/2605.22795) isolates a local
   scale-mismatch residual for the non-conservative Laplace displacement
   field. Increasing the particle count alone does not remove that residual.
4. [Gradient Flow Drifting](https://arxiv.org/abs/2603.10592) develops
   divergence-level conservative fields and a reverse-KL/chi-square mixture.
   This is a secondary branch because density-ratio weighting brings greater
   numerical risk.
5. [Expected Batch Optimal Transport Plans](https://arxiv.org/abs/2605.12174)
   formalizes larger batch OT as a progressively better approximation to the
   OT plan, not as a universally better downstream optimizer.

These are recent preprints or contemporary papers. Reproduce their formulas
from the primary source, test them independently, and do not infer that their
reported performance automatically transfers to this small NumPy generator.

## 3. Mathematical definitions for the field library

All fields below act on generated anchors `x_i` using a positive target batch
`y_pos` and a detached negative/reference batch `y_neg`. Use the
one-dimensional Laplace kernel

```text
k_tau(x,y) = exp(-abs(x-y)/tau),    tau > 0.
```

Every implementation must have a vectorized version and a deliberately slow
reference version. Cross-check them before training.

### 3.1 Existing exact paper Algorithm-2 field

Reuse the audited implementation in `driftlab.compute_v_paper` /
`Algorithm2Estimator.compute_field`. It uses:

- row softmax over positive and negative samples together;
- column softmax over anchors;
- geometric-mean affinities;
- positive/negative mass products;
- the diagonal self mask when anchors are reused as negatives.

Do not silently replace this baseline by a theoretical population formula.
It is the comparison target because it is the paper pseudo-code already used
by all low-dimensional runs.

### 3.2 Corrected Laplace mean-shift field

For a finite reference cloud `Y`, define

```text
Z_Y(x) = mean_y k_tau(x,y)
N_Y(x) = mean_y k_tau(x,y) * (y-x)
M_Y(x) = N_Y(x) / Z_Y(x).
```

The corrected displacement field is

```text
V_mean(x) = M_pos(x) - M_neg(x).
```

This is the direct finite-sample counterpart of normalized population
mean-shift drift. It is not the same as the bi-softmax Algorithm-2 estimator.
Use stable denominator floors only as declared numerical safeguards and log
every floor activation.

### 3.3 Sharp-normalized Laplace field

For the Laplace displacement numerator, use the sharp companion kernel

```text
h_tau(x,y) = tau * (abs(x-y) + tau) * exp(-abs(x-y)/tau).
```

Up to an irrelevant positive global scale, this is
`(1 + abs(x-y)/tau) * exp(-abs(x-y)/tau)`. It satisfies

```text
grad_x h_tau(x,y) = k_tau(x,y) * (y-x).
```

Define the sharp KDE and its score by

```text
Zsharp_Y(x) = mean_y h_tau(x,y)
Ssharp_Y(x) = N_Y(x) / Zsharp_Y(x).
```

The proposed conservative field is

```text
V_sharp(x) = Ssharp_pos(x) - Ssharp_neg(x).
```

Equivalently, its stop-gradient scalar loss at the current negative law is

```text
L_sharp = mean_i [log Zsharp_neg(x_i) - log Zsharp_pos(x_i)].
```

Gradient descent on this loss moves along `V_sharp` up to the declared scalar
step convention. Implement both the field and scalar-loss forms and verify
their parameter gradients agree on random nondegenerate batches.

For stable scalar evaluation use `logmeanexp` with

```text
log h_tau = log(tau) + log(abs(x-y)+tau) - abs(x-y)/tau.
```

The common `log(tau)` term may cancel between positive and negative KDEs, but
retain it in the reference implementation so the formula remains explicit.

### 3.4 Ordinary Laplace kernel-gradient field

The ordinary KDE score uses the derivative of `k_tau` rather than the
displacement numerator:

```text
grad_x k_tau(x,y)
  = -(1/tau) * sign(x-y) * k_tau(x,y).

Sgrad_Y(x)
  = mean_y grad_x k_tau(x,y) / mean_y k_tau(x,y).

V_kgrad(x) = Sgrad_pos(x) - Sgrad_neg(x).
```

Choose the value `0` for the derivative contribution at exact coincidence and
test that convention. This field and `V_sharp` are both conservative score
fields, but they smooth with different kernels and are not interchangeable for
Laplace.

### 3.5 Negative/reference variants

Every new field must support exactly two declared reference modes:

1. `reused_deleted`: reuse the generated batch, remove the diagonal term from
   both numerator and denominator;
2. `crossfit`: draw an independent latent batch, evaluate the generator, and
   detach that cloud as the negative reference.

Do not call a `1e6` logit penalty an exact deletion in the new field library.
Implement deletion directly. Keep the paper arm's penalty unchanged because
that arm must remain bitwise compatible.

The crossfit arm costs an additional generator forward and must be charged in
the ledger. It is an ablation, not an assumption that the earlier NCJ particle
gain will transfer.

## 4. Optimizer-state experiment

The QLD prefix and every finisher share generator parameters but need not share
Adam statistics. Implement these handoff modes:

```text
carry       keep m, v, and Adam step index from QLD (current behavior)
reset       set m = 0, v = 0, and Adam step index = 0 at the handoff
dual        preserve a QLD Adam state and create a distinct finisher state
```

`dual` matters only for later alternation experiments. During the first
common-prefix finisher screen, compare `carry` and `reset`. Do not add a
first-moment-only reset unless both fail and diagnostics implicate momentum
rather than scale adaptation.

For every recorded update, optionally reconstruct Adam's effective parameter
step and record:

```text
raw parameter-gradient norm
effective Adam-step norm
cosine(raw gradient, effective step)
cosine(finisher gradient, independent QLD gradient)
```

Flatten parameter dictionaries in one stable, declared name order. Unit-test
that flatten/unflatten is lossless.

## 5. Common-prefix branching architecture

This is the central efficiency and attribution device.

For each `(target, initialization, seed)`:

1. initialize one `TanhMLP` and its RNG streams;
2. train ordinary QLD through the declared prefix, initially `70%` of the
   horizon;
3. record endpoint metrics at the phase boundary;
4. deep-copy parameters, Adam buffers, step index, and all required immutable
   configuration into each branch;
5. reset optimizer buffers only in branches whose handoff mode says `reset`;
6. run every finisher with paired latent, positive-data, negative-reference,
   and evaluation streams;
7. ensure one branch reproduces the historical QLD-v1 continuation exactly.

The branches must not share mutable arrays after cloning. Add a mutation test:
updating one branch must leave every other branch and the saved prefix
bitwise unchanged.

### 5.1 Primary branch set

The first serious screen should contain only:

| arm | prefix | finisher | reference | Adam handoff |
|---|---|---|---|---|
| `paper-0.5` | none | paper Algorithm 2 | reused/masked | continuous |
| `qld-v1` | QLD 70% | paper Algorithm 2 | reused/masked | carry |
| `qld-full` | QLD 70% | QLD | ordinary batch | carry |
| `qld-paper-reset` | QLD 70% | paper Algorithm 2 | reused/masked | reset |
| `qld-mean-crossfit` | QLD 70% | corrected mean shift | crossfit | reset |
| `qld-sharp-deleted` | QLD 70% | sharp Laplace | reused/deleted | reset |
| `qld-sharp-crossfit` | QLD 70% | sharp Laplace | crossfit | reset |
| `qld-kgrad-crossfit` | QLD 70% | kernel-gradient Laplace | crossfit | reset |

The predeclared primary research contrast is
`qld-sharp-crossfit / qld-v1`. The other branches explain where any gain comes
from. Do not select the best label after the standard development run without
having declared a selection rule in the protocol.

### 5.2 Bandwidth development

Use a small global development grid only:

```text
tau in {0.2, 0.5, 1.0}
```

If all three are clearly under-scaled for the sharp kernel, one documented
diagnostic extension to `2.0` is allowed before the standard development
protocol is frozen. The chosen `tau` must be global across targets. A per-cell
hindsight oracle may be reported as a diagnostic but can never be the
candidate.

Do not reuse the failed one-step bandwidth selector. The repository already
showed that one-step output alignment did not reliably predict the complete
Adam trajectory.

## 6. Required invariants before training

Create a fast invariant suite and fail the runner before any target sweep if
one check fails.

### 6.1 Algebraic and numerical invariants

1. Identical positive and negative point clouds, stored in distinct arrays,
   give zero in the slow reference implementation to numerical tolerance.
   Separate iid clouds drawn from the same law are a consistency test and are
   **not** required to cancel exactly at finite batch size.
2. Swapping positive and negative batches negates every new field.
3. Translating anchors and both reference clouds by the same constant leaves
   every field unchanged.
4. Positive rescaling of the sharp kernel leaves its score unchanged.
5. Vectorized and slow loop implementations agree.
6. Direct `V_sharp` parameter gradients agree with gradients of `L_sharp`.
7. The analytical derivative of `h_tau` agrees with central finite
   differences away from coincidences.
8. Reused/deleted mode contains exactly `N-1` negative terms per anchor.
9. Crossfit output is unchanged if the positive and negative input arrays are
   copied or made non-contiguous.
10. Denominator floors are never hit on ordinary smoke-test batches; a
    deliberately extreme test must hit and report them.

### 6.2 Regression invariants

1. The `paper-0.5` branch remains bitwise equal to the current paper runner.
2. The `qld-v1` common-prefix branch remains bitwise equal to the current
   QLD-v1 runner under the same profile and streams.
3. `carry` cloning gives the same next Adam step as uninterrupted training.
4. `reset` gives zero moment buffers and a step index of zero before its first
   update.
5. Two branches from the same prefix see identical paired positive and latent
   samples unless their declared reference mode requires an extra stream.
6. Metric and event probes are independent of all training streams.

## 7. Trajectory and mechanism instrumentation

Endpoints are insufficient for this campaign. Save a row at:

```text
step 0
every 25 updates
the exact QLD/finisher handoff
the first five finisher updates individually
the final update
```

Each trajectory row must include:

- ED2 and SW1 on fixed independent evaluation samples;
- target-balanced quantile-bin CDF error;
- synthetic component mass-L1 and minimum component occupancy, evaluation
  only;
- generator output mean, standard deviation, and selected quantiles;
- raw field RMS and maximum norm;
- parameter-gradient norm;
- effective Adam-step norm;
- gradient cosines described in Section 4;
- denominator minima and floor activations;
- generator evaluations, unique latents, target samples, kernel pairs, sort
  work, and wall time accumulated so far.

At the handoff, compute a short-horizon diagnostic from the same saved prefix:

1. clone the branch;
2. apply exactly five candidate updates on paired training streams;
3. evaluate on an independent fixed probe;
4. compare predicted first-order gradient alignment with realized five-step
   ED2/SW1 change.

This diagnostic is explanatory only. Do not use it as an online selector in
the first campaign.

## 8. Redesigned event diagnostics

Retire `weighted_reach >= .90` as the sole event. It permits any collection of
modes with total target mass below 10% to remain absent.

Predeclare two events:

1. **Label-free distribution event:** both ED2 and SW1 on the independent
   event probe fall below fixed fractions of their step-zero values, with the
   fractions chosen in the protocol before results are observed.
2. **Synthetic occupancy event:** every target component above the registry's
   declared minimum reportable mass has nonzero observed occupancy and total
   component mass-L1 is below a fixed threshold.

The second event uses component metadata for evaluation only. No training arm,
bandwidth, field, or controller may see those labels. Report censoring and use
the same probe size and schedule for every arm.

Also report time to best-so-far ED2 and the best ED2 attained before and after
the handoff. This directly tests whether a finisher improves or erases its
prefix.

## 9. Fresh registry design

Do not tune on any existing QLD, LB-QCD, or OA-SQD target registry.

Create two disjoint development registries and reserve a third confirmation
registry:

### Registry A: mechanism screen

Use at least eight targets spanning:

- separated unequal mixtures with minimum weights near `.005`, `.01`, and
  `.03`;
- equal high-mode-count mixtures;
- heteroscedastic mixtures;
- overlapping mixtures;
- one connected skew target;
- one connected heavy-tail target.

Use `missing` and `concentrated` as primary initializations. Keep `far` as a
separately reported stress diagnostic, not part of the primary aggregate.

### Registry B: standard development tournament

Use at least sixteen new target specifications with the same families but new
parameters, component locations, scales, and weights. No exact target
specification may occur in Registry A or any old sealed registry.

### Registry C: untouched confirmation

Do not generate or inspect Registry C until one fully specified candidate
passes every Registry-B gate. Freeze its generator architecture, optimizer,
field, bandwidth, reference mode, phase schedule, and thresholds before
opening Registry C.

Store canonical JSON and SHA-256 hashes. The loader must reject an exact target
specification duplicated from any prior registry. Registry source code may
generate specifications, but the resolved immutable JSON is authoritative.

## 10. Experimental stages

### S0 -- seal the plan and build reproducibility boundaries

Deliverables:

- this plan committed before outcome-driven changes;
- Registry A generated and hash guarded;
- independent RNG streams declared;
- run manifest and source snapshot support;
- a results template stating that no result yet exists.

### S1 -- implement and audit the field library

Suggested file:

```text
numerics/conservative_finishers.py
```

Implement Sections 3 and 6 without running a target sweep. Store the invariant
report in the first smoke artifact.

Acceptance:

- all invariants pass;
- paper and QLD compatibility guards pass;
- no denominator floor is silently used;
- slow/vectorized and field/loss gradients agree.

### S2 -- common-prefix and optimizer audit

Suggested runner:

```text
numerics/run_conservative_finisher_development.py S2 --profile smoke
numerics/run_conservative_finisher_development.py S2 --profile screen
```

Use Registry A, three paired seeds, and the primary branch set. The goal is
mechanism elimination, not candidate promotion.

Questions:

- Does the existing paper phase improve or worsen its QLD prefix?
- Does resetting Adam improve the unchanged paper finisher?
- Does the sharp field produce a better five-step and final trajectory?
- Is crossfit materially better than exact deletion?
- Are candidate parameter gradients aligned with independent QLD descent?

Eliminate an arm if it is numerically unstable, triggers denominator floors
frequently, or loses to QLD by more than 10% on either connected control.

### S3 -- freeze one finisher candidate

Select using a predeclared lexicographic rule:

1. must satisfy every safety and control guard;
2. lowest target-balanced ED2 ratio vs QLD;
3. if within one bootstrap standard error, prefer lower generator-evaluation
   cost;
4. if still tied, prefer `deleted` over `crossfit`, then smaller bandwidth.

Freeze one arm, one bandwidth, one reference mode, and one Adam handoff. Write
them to an immutable candidate JSON before Registry B is run.

### S4 -- Registry-B standard development tournament

Minimum profile:

```text
updates          = 1,200
ordinary batch   = 128
paired seeds     = 8
endpoint samples = 4,096 or more
trajectory every = 25
```

Required baselines:

- paper Algorithm 2 at the globally frozen paper bandwidth;
- the same paper bandwidth oracle as a diagnostic only;
- historical QLD-v1;
- full QLD;
- QLD plus paper with reset Adam;
- the frozen conservative candidate.

Apply the gates to the frozen candidate, never to the best observed arm after
the run.

### S5 -- optional alternation only after a finisher win

If a conservative finisher beats QLD but loses some mass allocation during
the final phase, test a small, predeclared alternation using objective-specific
Adam buffers:

```text
K conservative updates, then 1 QLD correction
K in {4, 8}
final 10% conservative only
```

Do not add alternation if the finisher does not first beat QLD in S4. Do not
infer convergence merely because each individual field identifies its target.

### S6 -- untouched confirmation

Only after all S4 gates pass:

- commit candidate and protocol;
- generate/hash Registry C;
- run once;
- do not repair or replace the candidate after seeing confirmation;
- report a failed conjunction as failure.

### S7 -- formal follow-up

Formalization starts only after empirical selection. Candidate theorems may
include:

1. positivity and differentiability of the explicit Laplace sharp kernel;
2. `grad h_tau = k_tau * (y-x)`;
3. sharp-score representation of `V_sharp`;
4. equality of sharp KDE scores implies equality of the sharp KDEs on a
   connected domain;
5. characteristic/injective companion smoothing implies `p=q`;
6. positive scalar gain and phase scheduling do not change the intended zero
   of an individual field.

Do **not** claim that alternating identifying fields automatically converges,
that a finite minibatch shares the exact population zero set, or that a neural
stationary point implies equality of output laws. Follow `AGENTS.md`, add no
axiom or `sorry`, and run the full trust audit after every Lean change.

## 11. Metrics, aggregation, and uncertainty

### 11.1 Primary effectiveness

Use the target-balanced geometric mean of cell-median ED2 ratios against
QLD-v1. A cell is `(target, primary initialization)`. Medians are taken over
paired seeds before target aggregation so a target with more samples does not
receive extra weight.

Report a hierarchical paired bootstrap interval that resamples targets and
paired seeds. Store the bootstrap seed and complete replicate distribution.

### 11.2 Secondary effectiveness

- SW1 ratio;
- component mass-L1;
- label-free quantile-bin CDF error;
- family and initialization ratios;
- win fraction across primary cells;
- worst predefined family;
- best-before-handoff, handoff, and endpoint metrics;
- redesigned event times and censoring.

No candidate passes by ED2 alone. A field that directly optimizes an ED-like
quantity must still improve SW1 and mass calibration to prevent metric gaming.

### 11.3 Compute accounting

Report at least:

- optimizer updates;
- generator forward calls;
- generator example evaluations;
- unique latent samples;
- positive target samples;
- negative/reference samples;
- kernel pairs;
- sort work;
- backward examples;
- metric and trajectory probes;
- wall time.

Common-prefix sharing reduces research compute but does not reduce the cost of
a deployable arm. Report both the physical campaign cost and the standalone
per-arm training cost.

## 12. Development advancement gates

The Registry-B candidate advances only if all conditions hold:

| gate | threshold |
|---|---:|
| candidate / QLD-v1 ED2 | at most `.95` |
| candidate / QLD-v1 SW1 | at most `.97` |
| candidate / selected-paper ED2 | at most `.78` |
| each primary initialization / QLD-v1 ED2 | at most `.98` |
| worst predefined family / QLD-v1 ED2 | at most `1.05` |
| primary-cell win fraction | at least `.60` |
| hierarchical ED2 interval upper endpoint | below `1.0` |
| generator evaluations / paper | at most `2.0` |
| divergence | no worse than QLD and paper |
| frequent denominator-floor activation | none |

These thresholds may be adjusted only before Registry B is run and with a
written reason unrelated to observed Registry-B outcomes. A passed development
gate is not confirmation.

## 13. Secondary branches and stopping rules

### 13.1 Mixed reverse-KL/chi-square field

Try this only if the sharp and kernel-gradient finishers are stable but tie
QLD. Reproduce the exact density-ratio weights from the primary Gradient Flow
Drifting source. Use Gaussian KDE first because its differentiability and
score identity are clean. Track ratio clipping explicitly. This branch has
greater mode-collapse and numerical-overflow risk and is not the first
candidate.

### 13.2 Quantile plus energy-distance objective

An unbiased energy-distance U-statistic can supply explicit attraction and
generated-sample repulsion. It is attractive as a scalar complement, but in
one dimension it may duplicate much of QLD's distributional information and
directly favors the primary ED2 metric. Test it only after the conservative
finisher question is settled, and require simultaneous SW1 and mass gains.

### 13.3 Gradient conflict surgery

Measure gradient conflict before changing it. If QLD and the selected
conservative field frequently have negative parameter-gradient cosine, a
later experiment may project the auxiliary gradient away from the QLD
gradient. Do not introduce PCGrad or a similar rule without first showing
that conflict is common and predictive of harm in the saved trajectories.

### 13.4 Stop the branch when evidence says to stop

If corrected mean shift, sharp normalization, kernel-gradient, and optimizer
reset all tie QLD on Registry B, record that the local field is not the
remaining bottleneck in this architecture. The next move would then be
generator capacity/parameterization or a genuinely different scalar
distribution objective, not another finisher schedule sweep.

## 14. Explicitly forbidden repetitions and interpretations

Do not spend the next campaign on:

- more OA-SQD edge-quantile sweeps;
- larger virtual rank tables as a primary method;
- pulse-length or check-period searches;
- scalar gain/friction schedules alone;
- the failed one-step bandwidth selector;
- jitter or injected particle noise;
- naive random sliced extensions;
- selecting an arm by a per-cell hindsight oracle;
- reusing a sealed target registry for tuning.

Do not claim:

- ImageNet, real-feature, or high-dimensional superiority;
- novelty of sharp normalization or kernel-gradient drifting;
- that conservativity alone guarantees neural optimization success;
- that the full converse theorem proves minibatch or Adam convergence;
- that an unbiased pre-Adam gradient yields an unbiased Adam step;
- that a switched or alternating process has a common Lyapunov function;
- that passing Registry B is confirmation.

## 15. Proposed files and artifacts

```text
numerics/QLDSharpConservativeImplementationPlan.md   this document
numerics/conservative_finishers.py                   field/loss library
numerics/run_conservative_finisher_development.py    staged runner
numerics/conservative_registry_a.json                mechanism registry
numerics/conservative_registry_b.json                development registry
numerics/ConservativeFinisherProtocol.md              frozen protocol
numerics/ConservativeFinisherResults.md               outcome report
numerics/conservative_runs/<timestamp-stage>/         immutable artifacts
```

Each run directory should contain:

```text
manifest.json
source_hashes.json
source_snapshots/
invariants.json
rows.csv
trajectories.csv
work_ledger.json
summary.json
RESULTS.md
```

## 16. Recommended concrete order

1. Commit this plan before implementing outcome-sensitive choices.
2. Implement `h_tau`, corrected mean shift, sharp score, and kernel-gradient
   fields with slow references.
3. Complete every algebraic, gradient, deletion, and compatibility invariant.
4. Implement branch-safe QLD checkpointing and Adam `carry/reset` modes.
5. Add phase-boundary trajectories, gradient diagnostics, and redesigned
   events.
6. Generate and seal Registry A.
7. Run S2 and eliminate unsafe mechanisms.
8. Freeze one bandwidth/reference/optimizer candidate by the declared rule.
9. Generate and seal Registry B, freeze the S4 protocol, and commit.
10. Run the standard development tournament once.
11. If and only if all gates pass, consider alternation and then freeze an
    untouched confirmation.
12. Formalize only the empirically selected conservative mechanism, without
    importing switched-optimizer conclusions that were not proved.

## Bottom line

The repository already has a broad low-dimensional improvement over the paper
through QLD, and it already has efficient large-rank estimator infrastructure.
OA-SQD showed that better routing of the same global rank direction is not the
next reliable gain. The sharpest remaining question is whether the final
paper bi-softmax phase is an inferior local corrector and whether stale Adam
state hides the benefit of a conservative alternative.

The common-prefix QLD-SC campaign answers that question with minimal wasted
compute, strong pairing, and clear failure semantics. If it succeeds, it gives
an algorithmic improvement tied directly to the project's kernel and
identifiability analysis. If it fails, it rules out an entire family of local
field repairs and tells the project to move to generator parameterization or a
new distributional objective.
