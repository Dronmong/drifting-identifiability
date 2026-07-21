# QLD next-generation research: amplify the win, repair the failure

**Date:** 2026-07-20  
**Input:** sealed `QLD-confirmatory-v1` run  
**Purpose:** determine what genuinely worked, what did not, and the most
efficient path to a stronger algorithm without tuning on the sealed test.

## Executive conclusion

QLD v1 found a real but modest mechanism:

- global rank transport is scale-free, mass-aware, cheap, and usually improves
  final distributional error;
- the paper's Laplace field remains useful as a local finishing operator;
- the combination is approximately twice as fast as full-time paper drift and
  statistically favorable against one globally selected paper bandwidth;
- it is not uniformly better than target-specific paper tuning, does not extend
  through naive random slicing, is poor from far initialization, and loses on
  unequal-weight mixtures.

The recommended next algorithm is **Large-Batch Quantile-Calibrated Drifting
(LB-QCD)**:

1. replace minibatch rank matching by memory-efficient virtual-large-batch
   global rank matching;
2. use the quantile displacement as a held-out teacher to select the Laplace
   refinement bandwidth online;
3. only if the unequal-weight deficit remains, add a bounded positive
   low-density quantile weight;
4. for dimensions above one, replace random projections by informative,
   diverse projections--not by more random slices.

This ordering attacks measured defects one at a time and preserves QLD's main
advantages.

## 1. What the sealed run says

### 1.1 Strengths that survived confirmation

| property | evidence | interpretation |
|---|---:|---|
| Final ED2 | ratio `.9105`, CI `[.8270,.9873]` | modest target-balanced improvement against validation-selected paper |
| Final SW1 | ratio `.8949`, CI `[.8383,.9535]` | improvement is not an ED2-only artifact |
| Breadth | wins 23/32 cells | effect is distributed, not carried by one target |
| Missing init | ratio `.8918`, 11/16 wins | global allocation helps the intended regime |
| Concentrated init | ratio `.9295`, 12/16 wins | benefit is not confined to one initialization |
| Overlap | family ratio `.7469` | rank transport works well where a local scale is ambiguous |
| Contamination | family ratio `.8113` | legitimate remote mass is retained rather than ignored |
| Heteroscedasticity | family ratio `.8687` | scale-free ranks tolerate varying component widths |
| Calibration | mass-L1 ratio `.9385` | final mass allocation improves modestly |
| Stability | 0 divergences | no stability price was observed |
| Kernel work | ratio `.300` | 70% of training avoids quadratic kernel affinities |
| Wall time | ratio `.498` | measured training was about twice as fast |

These are unusually coherent relative to the earlier repository attempts. The
benefit transferred to the learned generator, survived a sealed target suite,
appeared in two distribution metrics, and came with lower computation.

### 1.2 Weaknesses that survived confirmation

| weakness | evidence | likely cause |
|---|---:|---|
| Effect too small for the gate | `.9105 > .80` | QLD improves the average case but does not dominate |
| Oracle nearly catches it | oracle ratio `.9774` | much of the gain is robustness to one global bandwidth |
| Unequal weights | family ratio `1.0726`; oracle ratio `1.2727` | rare quantile intervals are poorly resolved by batches of 128; fixed `.5` refinement is too coarse for all three targets |
| Cell inconsistency at high K | e.g. K22 concentrated `1.294`, K32 missing `1.225` | rank supervision and generator optimization remain batch/init sensitive |
| Slower coverage event | time ratio `1.357`; 47 vs 38 censored | rank transport improves final mass/error but not the chosen three-sigma time-to-cover event |
| Far-start failure | earlier ratio `2.494` | Adam-limited global translation cannot finish before the local-field handoff |
| Naive 2-D failure | random sliced warm start lost | random directions dilute or cancel the informative fission direction |
| Fixed phase schedule | 70/30 chosen once | cannot adapt refinement scale or handoff to target geometry |

The method should no longer be described simply as a faster mode-recovery
algorithm. Its confirmed strength is **cheap final distribution matching under
scale uncertainty**. Its time-to-coverage behavior is worse.

## 2. The unequal-weight diagnosis

Unequal mass does not by itself invalidate rank transport. In one dimension,
the monotone coupling between empirical measures automatically allocates each
target component according to the number of its samples. Replacing it by
unbalanced OT would allow the optimizer to discard real rare mass, which is the
opposite of the desired distribution-matching property.

The finite batch is the more concrete problem. For the rarest components in
the sealed unequal targets:

| target | minimum weight | expected count at B=128 | probability absent at B=128 | probability absent at M=1024 |
|---|---:|---:|---:|---:|
| T09 | .0300 | 3.84 | .0203 | `2.8e-14` |
| T10 | .0156 | 2.00 | .1332 | `9.9e-8` |
| T11 | .0119 | 1.52 | .2159 | `4.7e-6` |

When the rare interval is absent, the empirical quantile map cannot send any
generated sample there. When it appears with one or two samples, its rank
boundary jumps substantially between updates. This is a bias/resolution issue
for the stochastic training objective, not evidence that the population
quantile map is wrong.

This matches existing theory: minibatch Wasserstein optimization is an
implicit regularized objective and can lose the distance identity property;
other work explicitly identifies misspecified minibatch transport maps and
relationships ignored across minibatches. See
[Fatras et al. (2020)](https://proceedings.mlr.press/v108/fatras20a.html),
[mini-batch partial OT](https://proceedings.mlr.press/v162/nguyen22e.html), and
[Batch-of-Minibatches OT](https://proceedings.mlr.press/v162/nguyen22d.html).

## 3. Primary repair: virtual-large-batch quantile fission

### 3.1 Algorithm

Use a Run-Sort-ReRun implementation during the quantile phase:

1. Draw a virtual batch of `M` latent samples and `M` target samples.
2. Run the generator in microbatches of 128 without retaining activations;
   store latent vectors and scalar outputs.
3. Sort all `M` generated and target outputs globally and record the matching.
4. Re-run each latent microbatch with activations, attach its globally matched
   stop-gradient targets, and accumulate parameter gradients.
5. Apply one Adam update after all microbatch gradients have been accumulated.

This is a direct adaptation of
[Run-Sort-ReRun](https://proceedings.mlr.press/v139/lezama21a.html), which was
designed to escape sliced-Wasserstein batch-size limits while keeping memory
bounded. It is not a novelty claim by itself.

### 3.2 Why it amplifies QLD

- Preserves exact global rank allocation, QLD's strongest mechanism.
- Resolves rare quantile intervals without mode labels or target weights.
- Reduces discontinuous rank-boundary noise.
- Remains sorting-based rather than kernel-quadratic.
- Does not alter the population equilibrium.
- Can be implemented exactly in the current NumPy generator by adding gradient
  accumulation, so it is an unusually clean mechanism experiment.

### 3.3 Budgets that must both be reported

Larger virtual batches consume more generator examples. Report two comparisons:

- **update matched:** same 1,200 optimizer updates, measuring the quality
  ceiling and total wall time;
- **example/wall matched:** reduce the number of virtual-batch updates so the
  number of generator evaluations or measured wall time matches QLD v1.

Without both, a gain could merely be more compute.

### 3.4 Initial values

Development grid only:

```text
M in {256, 512, 1024}
microbatch = 128
quantile fraction in {0.60, 0.70, 0.80}
```

Start with `M=512` and `M=1024`. They reduce the T11 rare-component absence
probability from `.216` to `.00217` and `4.7e-6`, respectively.

**Expected effectiveness: 9/10.**  
**Originality: 3/10 as a component; 6/10 in the drifting hybrid.**  
**Risk: low mathematical risk, moderate compute-accounting risk.**

## 4. Second repair: quantile-calibrated Laplace refinement

The oracle diagnostic says target-specific bandwidth tuning erases most of the
average advantage. Instead of hiding that, use QLD's scale-free field to select
the local kernel scale without labels.

### 4.1 Held-out alignment selector

At the start of refinement and every `H` updates thereafter:

1. draw an independent probe batch;
2. compute its large-batch or ordinary rank displacement `R`;
3. compute paper fields `V_tau` for a small candidate set;
4. normalize candidate fields to equal RMS step size;
5. choose the `tau` with the best first-order quantile descent score

```text
score(tau) = <R, V_tau> / (||V_tau|| + epsilon).
```

Equivalently, use a small standardized Euler lookahead and select the field
that most decreases held-out empirical W2. Freeze the chosen bandwidth for the
next block. The probe batch must be independent of the update batch.

This uses global quantile transport as a teacher for the paper field. It could
recover the oracle's scale advantage without target names, mixture parameters,
or repeated full training.

### 4.2 Why this is preferable to old bandwidth heuristics

The repository's geometry heuristics tried to infer one bandwidth from cluster
statistics and repeatedly misfired. Field alignment asks a narrower operational
question: *which paper field currently points most nearly along a known global
distribution-descent direction?* It can change as the generator approaches the
target.

Candidate set for development:

```text
tau in {0.1, 0.2, 0.5, 1.0, 2.0}
H in {25, 50, 100}
```

Evaluate selector accuracy against the hindsight best `tau`, but never give
the selector oracle labels.

**Expected effectiveness: 8/10.**  
**Originality: 7/10; no exact QLD/Laplace alignment scheme was found in the
review, though adaptive kernel selection broadly is not new.**  
**Risk: moderate; field alignment may not predict an Adam parameter update.**

## 5. Third repair only if needed: bounded rare-quantile emphasis

If large-batch matching still loses on unequal targets, modify the quantile
energy rather than deleting mass. For quantile functions `Qp,Qq`, use

```text
D_w(p,q) = integral_0^1 w(u) |Qp(u)-Qq(u)|^2 du,
```

with `0 < w_min <= w(u) <= w_max`. Estimate `w` from a smoothed target-quantile
spacing profile and clip it, for example to `[0.5,2]`. Low-density/rare
quantile regions receive more gradient; the weight is annealed back toward one
before Laplace refinement.

The bounded-positivity condition is important: it preserves
`D_w(p,q)=0 iff p=q` under the usual finite-moment assumptions. Setting weights
to zero or using unbalanced mass deletion would lose this clean property.

This is an inference from the project mathematics and must be proved and
stress-tested. It is not presented as an established literature theorem in
this exact algorithmic form.

**Expected effectiveness: 6/10 after large-batch repair.**  
**Originality: 7/10.**  
**Risk: medium-high; it may overemphasize contaminants or create bridges.**

## 6. Additional ideas and their priority

### A. Predictor-corrector alternation

Interleave one cheap quantile correction every several Laplace updates rather
than making a hard phase switch. This may stop local refinement from undoing
global mass allocation. End with a paper-only block so the final field remains
easy to interpret.

**Priority: medium.** Test only after large-batch and bandwidth calibration,
because it complicates attribution.

### B. Data-adaptive latent quantile prior

Learn a low-cost quantile warp of the latent prior so the generator begins
closer to the target's tail geometry. Current 2026 work reports that learned
1-D quantile priors can shorten transport paths with little overhead
([Chemseddine et al., 2026](https://openreview.net/forum?id=vLQO6nrpYq)). This
could address the far-start weakness, but it changes the model family and is
less directly tied to the drifting field.

**Priority: medium-low for QLD v2; high only if far-start robustness becomes a
primary objective.**

### C. Partial or unbalanced OT

The literature shows partial/unbalanced minibatch OT can mitigate misspecified
maps. It is valuable when contamination is unwanted or source/target mass is
genuinely unmatched. Here the remote and rare components are part of the true
target. Permitting mass deletion can improve an average metric by abandoning
exactly the modes we intend to learn.

**Priority: low and diagnostic only.** Do not make it the default unequal-mass
repair.

### D. More paper/Sinkhorn balancing

[Sinkhorn-Drifting](https://arxiv.org/abs/2603.12366) already shows two-sided
balancing improves temperature robustness at additional training cost. This
repository's finite balancing did not solve homogeneous fission. It remains a
strong external baseline, but it does not replace the global quantile phase.

**Priority: baseline, not the main invention.**

## 7. Route to dimensions above one

The random-slice failure is consistent with the projection-complexity
literature: many random directions can be uninformative, while one max direction
can ignore other useful structure. Relevant alternatives include
[Max-Sliced Wasserstein](https://arxiv.org/abs/1904.05877),
[Distributional Sliced-Wasserstein](https://arxiv.org/abs/2002.07367), and
[random-path projection directions](https://proceedings.mlr.press/v235/nguyen24l.html).

The recommended multidimensional fission field is therefore:

1. generate candidate directions from normalized target/generated point
   differences (random-path directions);
2. retain several high-discrepancy directions subject to an orthogonality or
   diversity constraint;
3. run virtual-large-batch rank transport along those directions;
4. average with discrepancy-based weights;
5. finish with the certified data-space Laplace field.

This directly attacks why random slicing failed: it increases information per
projection without collapsing to a single adversarial direction.

Do not start here. First establish that large-batch QLD v2 decisively improves
the one-dimensional weak families.

## 8. Formalization opportunities

The empirical program can again use the proof infrastructure rather than only
borrowing terminology.

1. **Weighted quantile identifiability:** prove `D_w=0 -> p=q` for measurable
   `w` bounded below by a positive constant.
2. **Quantile descent:** formalize monotone rank displacement as descent of
   empirical one-dimensional `W2^2`.
3. **Large-batch consistency:** state convergence of the empirical quantile
   field to `Qp(Fq(x))-x` under atomless/finite-second-moment assumptions.
4. **Adaptive-bandwidth endpoint:** if the selector always returns a positive
   valid bandwidth, zero of the selected Laplace field at a fixed state invokes
   the existing converse for that bandwidth.
5. **Do not overclaim switching convergence:** alternating two identifying
   fields does not automatically give a Lyapunov or finite-time convergence
   theorem.

The weighted result is especially attractive: its positivity guardrail makes
rare-region amplification compatible with exact identification.

## 9. Concrete experimental order

### Stage N0: new development registry

Create targets disjoint from all QLD v1 validation/test data:

- six unequal mixtures with minimum weights from `.005` to `.05`;
- three equal/heteroscedastic controls;
- two legitimate remote-contamination controls;
- one connected heavy-tail control;
- missing and concentrated initialization, plus a small far-start diagnostic.

Use eight paired development seeds. These targets are for mechanism selection,
not claims.

### Stage N1: large-batch ablation

Compare:

```text
paper selected globally
paper oracle diagnostic
QLD v1
QLD-RSR M=256
QLD-RSR M=512
QLD-RSR M=1024
```

Report update-matched and wall/example-matched results. Advance only if RSR
improves unequal-family ED2 without losing more than 5% on controls.

### Stage N2: bandwidth calibration

On the same development split, compare fixed refinement with held-out
alignment selection. Measure:

- final ED2/SW1;
- agreement between selected and hindsight-best bandwidth;
- added kernel work and wall time;
- stability of selections across seeds and training blocks.

Advance only if the combined method reaches ED2 ratio at most `.82` against
the globally selected paper baseline and at most `.95` against the oracle
diagnostic.

### Stage N3: rare weighting, conditionally

Run only if N1 leaves a clear unequal-family gap. Tune the smallest bounded
weight range that repairs it. Reject any setting that worsens contaminated or
heavy-tail controls by more than 10%.

### Stage N4: freeze LB-QCD

Freeze one candidate and create a completely new registry. Use at least 20
paired seeds and the same target-level hierarchical bootstrap. Do not reuse the
QLD v1 sealed targets for selection or a second claim.

### Stage N5: multidimensional probe

Only after N4 passes, test informative diverse projections in 2-D. The first
gate is simple: beat the tuned paper baseline and the already-rejected random
slice field on ED2 while preserving coverage and compute accounting.

## 10. Decision summary

The best immediate bet is not a more exotic divergence. It is to make QLD's
successful global operation statistically faithful at rare quantiles, then use
that same operation to calibrate the local paper field.

In priority order:

1. **Virtual-large-batch Run-Sort-ReRun quantile fission.**
2. **Held-out quantile-aligned Laplace bandwidth selection.**
3. **Bounded positive rare-quantile weights, only if still needed.**
4. **Informative diverse projections for dimensions above one.**

This path is effective, honest about existing literature, computationally
plausible, and unusually compatible with the project's identifiability results.

## 11. Implementation outcome (2026-07-20)

N0--N2 have now been executed. Unconditional virtual batching verified the
rare-quantile diagnosis but failed the control-family guardrail. Phase-fraction,
periodic-pulse, and mean-zero noise-restoration repairs did not remove the
overlap failure. A target-only quantile-resolution router did: it invokes
`M=1024` RSR only when a compact separated target region has fewer than eight
expected representatives in the ordinary batch.

On the 1,200-update, eight-seed development run, resolution-gated LB-QCD reached
ED2 ratios `.7948` against the globally selected paper baseline and `.8685`
against the per-cell hindsight paper oracle. Its target-bootstrap interval
against the selected paper was `[.7063,.8910]`, and every family was favorable.
It cost `1.2217x` measured worker wall time and `7.8906x` generator-example
evaluations while retaining the `.30` kernel-pair ratio.

Both bandwidth selectors were rejected: cosine alignment produced `.9779` and
an exact cloned-Adam lookahead produced `.9350` at the screen profile, versus
`.9059` for fixed `tau=.5`. N3 weighting was therefore skipped. A far-start
diagnostic remains unfavorable (`3.3089x` paper ED2).

Full details and the frozen-candidate recommendation are in
[LBQCDDevelopmentResults.md](LBQCDDevelopmentResults.md).

The subsequent frozen N4 test produced ED2 `.8218` with interval
`[.7649,.8915]`, SW1 `.8679`, 30/32 cell wins, all families favorable, and
`.9437` versus a per-cell hindsight paper oracle. Its formal gate nevertheless
failed because the predeclared ED2 effect threshold was `.80`. See
[LBQCDConfirmatoryResults.md](LBQCDConfirmatoryResults.md).
