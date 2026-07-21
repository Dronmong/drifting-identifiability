# OA-SQD development results

**Date:** 2026-07-21

**Program:** `OASQD-development-v1`

**Status:** development gate failed; do not advance to frozen confirmation

This document records the implementation and staged evaluation of
Occupancy-Adaptive Stratified Quantile Drifting (OA-SQD), specified in
[`OccupancyAdaptiveQuantileResearch.md`](OccupancyAdaptiveQuantileResearch.md)
and governed by
[`OASQDDevelopmentProtocol.md`](OASQDDevelopmentProtocol.md). These are
one-dimensional synthetic learned-generator experiments, not an ImageNet,
real-feature, multidimensional, or confirmatory performance claim.

## Verdict

The implementation succeeded technically and produced three reusable pieces:

1. a tail-aware empirical quantile atlas that passed its declared controls;
2. a conditionally unbiased rank-stratified backward estimator that cut the
   full-RSR backward workload without a visible endpoint regression;
3. a sample-only occupancy controller and complete computation ledger.

The proposed successor did **not** produce a robust improvement over QLD-v1.
The O4 three-seed screen selected an edge-gated candidate with OA-SQD/QLD ED2
`0.9896`, but the predeclared candidate obtained `1.0031` in the eight-seed O5
standard tournament. Its bootstrap interval against QLD was
`[0.9904, 1.0198]`. The screen advantage therefore did not replicate.

OA-SQD still improved over the paper implementation at `tau=.5`: its O5 ED2
ratio was `0.8448` and its SW1 ratio was `0.8669`. That improvement is not a
new algorithmic gain over the repository's simpler QLD-v1 baseline, which
itself had paper ED2 ratio `0.8422` on the same run.

The predeclared conjunction failed, so O6 was not run and no confirmation
registry was created.

## Audit correction to the motivating evidence

The earlier LB-QCD coverage-time comparison was confounded: the event
diagnostic received the arm's training batch, so full-RSR arms were evaluated
with `N=1024` while ordinary QLD was evaluated with `N=128`. The endpoint ED2,
SW1, and mass results remain valid, but the old “earlier in 13 and tied in 5”
coverage statement is not a fair cross-arm timing result.

The OA-SQD runner fixes this by drawing an independent event probe of the same
size for every arm. Under that fair diagnostic, state-aware full RSR was
approximately tied with fixed LB-QCD in O2 (event-time ratio `0.9963`), while
the frozen O5 OA-SQD candidate was worse (`1.0480`).

## What was implemented

### Target atlas and occupancy controller

[`oasqd.py`](oasqd.py) implements:

- full-sample, split-half, and bootstrap-persistent gap evidence;
- tail-aware compactness checks and connected Gaussian/Student controls;
- Wilson intervals for empirical target and generated region masses;
- an ordinary under-resolution test plus a one-sided occupancy-deficit test;
- two-check activation, two-check clearing, cooldown, bounded pulses, and a
  one-shot mode;
- a conservative virtual-batch choice from
  `{128, 256, 512, 1024, 2048}`, with cap hits reported;
- an optional generated-median edge gate used only in the final O4 selection
  pass.

The atlas and controller use samples only. Synthetic component metadata is not
available to the algorithm.

### Stratified Run-Sort-ReRun

Every global update first constructs the full rank table. The backward pass
then includes every nonempty rank/target-region stratum and allocates remaining
slots proportionally. A sample from stratum `j` receives the exact
importance factor `N_j / (M b_j)`, so the estimator is conditionally unbiased
for the finite full-table gradient. The implementation uses `M+B` generator
example evaluations rather than the full method's `2M`.

The fixed-table Monte Carlo audit used 400 repetitions:

| quantity | result |
|---|---:|
| gradient dimension | `1,185` |
| mean-error norm | `.1797` |
| Monte Carlo SE norm | `.0814` |
| error / SE | `2.207` |
| stratified gradient MSE | `2.6774` |
| uniform-subset gradient MSE | `5.0058` |
| stratified / uniform MSE | `.5349` |
| declared bias audit | pass |

At screen scale, the stratified stop arm had QLD ED2 ratio `.9994` versus
`.9986` for full RSR, while total generator-example evaluations fell from
`26.37M` to `17.53M` and backward-example evaluations fell from `13.75M` to
`4.92M`.

### Reproducibility boundary

The development registry contains 16 fresh targets and is hash guarded by

```text
111FA056B30931F2BEBC6C95D7DFBB4CA0C810910ECAC2A8BF693782C6E858B8
```

The loader rejects an exact target specification duplicated from either
sealed LB-QCD registry. Atlas, controller, target table, latent, backward
selection, endpoint, metric, and event random streams are separated. Every
run stores its manifest, source hashes, source snapshots, rows, summary, and
work ledger under [`oasqd_runs/`](oasqd_runs/).

The O5 runner also requires `--primary-candidate`. It applies the gates to
that predeclared arm rather than selecting the best arm after observing O5.

## Staged findings

### O1 — atlas: pass

Authoritative artifact:
[`20260721-140511-o1-screen`](oasqd_runs/20260721-140511-o1-screen/RESULTS.md).

The atlas had zero false positives on the declared connected controls,
detected every synthetic separated rare component down through mass `.0025`,
and exactly recovered the declared separated regions. This establishes that
the empirical partition behaves as designed on these controls; it is not a
general component-identification theorem.

### O2 — state-aware full RSR: narrow pass

Authoritative artifact:
[`20260721-140550-o2-screen`](oasqd_runs/20260721-140550-o2-screen/RESULTS.md).

The stop controller improved over frozen LB-QCD (`.9633` ED2 ratio), was
essentially tied with QLD (`.9986`), and used `9,860` global updates rather
than `14,560`. Its fairly measured event-time ratio was `.9963`. However, it
cost `5.11x` the paper arm in generator-example evaluations and split by
initialization as `1.0062` for concentrated versus `.9910` for missing.

This supports “stopping is safer than a fixed long global phase,” but not
“global RSR improves the endpoint beyond QLD.”

### O3 — stratified estimator: technical success

Authoritative artifact:
[`20260721-140854-o3-screen`](oasqd_runs/20260721-140854-o3-screen/RESULTS.md).

The stratified arm retained the full arm's endpoint to within `.09%` relative
to QLD while materially reducing work. The fixed-table audit found no declared
bias failure and roughly halved gradient MSE relative to uniform subsampling.
This is the clearest positive output of the project.

### O4 — adaptive resolution and target table: screen-only signal

Attribution artifact:
[`20260721-141101-o4-screen`](oasqd_runs/20260721-141101-o4-screen/RESULTS.md).

The systematic target table supplied most of the small O4 screen benefit:

| arm | ED2 / QLD |
|---|---:|
| fixed-`M`, systematic | `.9916` |
| adaptive-`M`, iid target | `1.0029` |
| adaptive-`M`, systematic | `.9908` |

The ungated adaptive arm helped missing starts (`.9540`) but harmed
concentrated starts (`1.0290`). A sample-only edge gate was added to suppress
that unnecessary routing. The preselection sweep chose `Q=.08`, with screen
ED2/QLD `.9896`, concentrated ratio exactly `1.0`, missing ratio `.9794`, and
generator-evaluation/paper ratio `1.516`.

That aggregate signal came from only five routed target/start cells and was
therefore treated as provisional rather than as a result.

### O5 — standard development tournament: fail

Authoritative artifact:
[`20260721-142019-o5-standard`](oasqd_runs/20260721-142019-o5-standard/RESULTS.md).

The frozen candidate was
`oasqd-edge-stop-Q0.08-H25-P50-N2048-K8`. Results used 16 targets, both primary
initializations, eight paired seeds, and 1,200 updates.

| comparison or diagnostic | result |
|---|---:|
| candidate / paper `.5` ED2 | `.8448` |
| candidate / QLD ED2 | `1.0031` |
| candidate / fixed LB-QCD ED2 | `.9966` |
| candidate / per-cell paper oracle ED2 | `.9370` |
| candidate / paper SW1 | `.8669` |
| missing / QLD ED2 | `1.0062` |
| concentrated / QLD ED2 | `1.0000` |
| event time / fixed LB-QCD | `1.0480` |
| generator evaluations / paper | `1.4435` |
| global updates over all 256 trials | `1,200` |
| divergences | `0` |

Family ratios versus QLD ranged from `.9964` to `1.0120`; connected,
contaminated, heavy-tail, heteroscedastic, and overlap controls were exactly
`1.0` because the candidate did not route them. This is a safe but mostly
inactive controller, not a generally superior optimizer.

The conjunction result was:

| gate | pass? |
|---|---:|
| ED2 / QLD at most `.95` | no |
| ED2 / paper at most `.78` | no |
| each initialization / QLD at most `.98` | no |
| worst family / QLD at most `1.05` | yes |
| event time / fixed at most `1.00` | no |
| generator evaluations / paper at most `3.0` | yes |
| divergence no worse | yes |

## Interpretation and next decision

The occupancy diagnosis was directionally sensible—broad global routing did
hurt concentrated starts—but the repair became too selective to improve the
aggregate endpoint. More importantly, even ungated full RSR was only tied
with QLD at standard scale. The bottleneck is therefore not merely controller
precision: on this benchmark, global quantile updates do not add a stable
optimization advantage after the direct QLD hybrid is already present.

Per the original decision rule, OA-SQD should not proceed to O6 or O7 now.
The atlas, fair event protocol, and unbiased stratified estimator should be
retained as infrastructure. A future performance branch should change the
training signal itself—such as the predeclared mixed-divergence or
optimizer-dynamics route—rather than continue tuning occupancy thresholds on
this development registry.
