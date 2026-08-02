# Stage B2 post-confirmation analysis

**Date:** 2026-07-31  
**Status:** B2 passed its frozen mechanism gate. This document is a post-hoc
analysis and prospective planning record; it does not modify or retroactively
re-adjudicate the frozen B2 result.

Primary artifacts:

- `numerics/EncoderIndependentB2Protocol.md`
- `numerics/encoder_independent_drifting/stage_b2/b2_preflight.json`
- `numerics/encoder_independent_drifting/stage_b2/b2_baseline.json`
- `numerics/encoder_independent_drifting/stage_b2/b2_freeze.json`
- `numerics/encoder_independent_drifting/stage_b2/b2_confirmatory.json`

## 1. Executive assessment

B2 legitimately passed the gate frozen before candidate training. The result is
best described as a **mechanism success**, not yet an overall generator
promotion:

> A modest, encoder-free normalized-Laplace correction caused a consistent
> 23--27% reduction in paired held-out raw drift energy on a fresh,
> class-aligned distribution shift without measurably reducing the frozen
> recall statistic.

This conclusion is stronger than a bare `2/3 units pass` headline because every
one of the 18 paired audit comparisons favored B2. It is weaker than a general
model-improvement claim because global effective rank fell substantially,
FID/KID were mixed against B0, and B1 remained stronger on the standard quality
metrics.

The immediate procedural priority is to commit or tag the exact B2 source,
protocol, artifacts, and postmortem before making prospective changes. The
hash chain is valid, but the B2 work currently lives in a dirty worktree and is
not recoverable from the recorded base commit alone.

## 2. Integrity and methodology audit

The audited chain passed:

- protocol and executable-source manifests match the freeze;
- B0, B1, preflight, baseline, and freeze hashes match;
- the confirmatory sidecar matches the result artifact;
- all three saved B2 checkpoint hashes match;
- the fresh CINIC-10 ImageNet-derived pool and its disjoint allocation remain
  bound by hash;
- the matched-real controls are valid;
- the confirmation used the frozen `tau`, `lambda_event`, units, seeds, and
  thresholds.

The principal methodological strengths were:

1. outcome-blind bandwidth and event-weight calibration;
2. exact row-normalized Laplace weights evaluated by a stable softmax;
3. independent target positives, full-support probes, and differentiable model
   negatives;
4. gradients only through a separate generated negative trajectory;
5. exact pairing of B0/B2 initialization, flow batches, endpoint noise, bridge
   time, augmentation, and evaluation priors;
6. a fresh external pool with no decoded-pixel CIFAR-10 overlap;
7. frozen matched-real controls, collapse/memorization vetoes, and artifact
   boundaries;
8. a real-real floor shared within each candidate/B0 audit pair.

The finite loss remains only a stochastic surrogate for the population energy.
The experiment does not establish exact zero drift, pointwise zero drift, or
the formal converse's hypotheses for the learned law.

## 3. Confirmatory result

| Unit | B0 excess -> B2 excess | excess reduction | raw-energy reduction | paired wins | recall change | effective-rank change |
|---|---:|---:|---:|---:|---:|---:|
| 300 | 13.28 -> 5.92 | 55.4% | 25.1% | 6/6 | +0.0049 | -40.1% |
| 301 | 9.12 -> 3.12 | 65.7% | 27.2% | 6/6 | -0.0039 | -40.5% |
| 302 | 13.91 -> 8.13 | 41.5% | 22.9% | 6/6 | -0.0005 | -37.7% |

Unit 302 failed only the frozen requirement that median floor-subtracted
excess be at most one half of B0's excess. It nevertheless improved all six
paired audits.

The mean paired reductions and approximate 95% t intervals were:

| Unit | mean raw-energy difference | approximate 95% interval |
|---|---:|---:|
| 300 | -7.28 | [-8.44, -6.12] |
| 301 | -5.86 | [-6.77, -4.96] |
| 302 | -5.98 | [-7.13, -4.83] |

The baseline/candidate raw audit energies had correlations above `0.995`.
Their individual standard deviations were about `9--11`, while paired
difference standard deviations were only `0.86--1.11`. This is strong evidence
that common-random-number pairing removed most instrument noise and should be a
standard feature of future gates.

## 4. The real-real floor and the brittle 50% rule

The real-real floor's replicate coefficient of variation was approximately
21%, and individual excess values could be negative. That is not evidence that
the paired improvement is spurious. Within a replicate,

```text
(E_B2 - E_floor) - (E_B0 - E_floor) = E_B2 - E_B0.
```

The floor cancels exactly from the paired difference. It does, however, affect
the ratio used by the binary 50%-of-excess rule. Three similar raw reductions
were converted into two formal passes and one failure depending on how far B0
sat above the noisy floor.

Future gates should therefore make paired raw-energy reduction the primary
effect and retain floor-relative gap closure as a secondary/resolvability
quantity. If gap closure remains a primary magnitude target, estimate it using
larger or cross-fitted real-real batches and attach uncertainty rather than
thresholding one ratio of medians.

## 5. Bandwidth interpretation

The frozen calibration gave

```text
tau                         = 7.0854
median target distance      = 36.3399
tau / median distance       = 0.1950
off-diagonal ESS target     = 0.60
achieved ESS fraction       = 0.59999984
```

At the median distance the unnormalized kernel is only about `0.0059`, but
this is not a useful measure of locality. Row normalization cancels a common
small factor. Effective sample size is the relevant diagnostic:

- calibration used roughly 76 effective neighbors out of 127;
- training positives used roughly 36 effective neighbors out of 64 at the
  median event;
- training negatives used roughly 43 effective neighbors out of 64.

Operationally, this is a moderately broad kernel, not a nearest-neighbor
kernel. The ESS calibration is a strong design choice and should be retained.

There is still a lower-tail warning. Some recorded rows reached ESS fractions
near `0.02--0.08`, meaning that rare probes were dominated by approximately one
to five samples. Future health gates should record and threshold predeclared
lower ESS quantiles and maximum-weight quantiles, not only the median.

With 3,072 pixel coordinates and per-coordinate probe noise `0.05`, the typical
noise norm is

```text
0.05 * sqrt(3072) = 2.77 = 0.39 tau.
```

Thus the population probe law has full support, but finite probes remain close
to target samples. Future experiments may test a predeclared noise mixture, but
must not conflate finite near-data probing with global pointwise verification.

Bandwidth selection in high-dimensional kernel methods is itself a substantive
model-selection problem. Relevant primary references include:

- Garreau, Jitkrittum, and Kanagawa,
  [Large sample analysis of the median heuristic](https://arxiv.org/abs/1707.07269).
- Gao and Shao,
  [Kernel two-sample tests in high dimensions](https://academic.oup.com/biomet/article/110/2/411/6670793).

## 6. Quality tradeoff exposed by B2

Recall was retained, local nearest-neighbor diversity rose, duplicate rate
remained zero, and memorization vetoes passed. However, raw-pixel effective rank
fell by `38--40%` in every unit:

| Unit | B0 rank | B2 rank | B1 rank |
|---|---:|---:|---:|
| 300 | 13.86 | 8.30 | 12.46 |
| 301 | 14.00 | 8.33 | 12.13 |
| 302 | 15.98 | 9.96 | 14.09 |

The inherited absolute collapse floor was only `4.49`, so B2 remained well
above a catastrophic-collapse threshold. That veto was working as designed,
but it was not a relative geometry-retention gate.

Against B0:

- FID/KID improved in unit 300 and worsened in units 301 and 302;
- precision declined slightly in all three units;
- recall changes were small;
- PR-F1 was essentially neutral.

Against the report-only B1 incumbent, B2 achieved slightly lower held-out drift
excess in all three units but B1 retained substantially better FID, KID, and
effective rank. B2 has therefore isolated a useful correction mechanism but has
not displaced B1 as the strongest overall model.

Finite-sample FID is biased, including model-dependent bias, so the 512-sample
FID remains indicative rather than adjudicative:

- Chong and Forsyth,
  [Effectively Unbiased FID and Inception Score](https://openaccess.thecvf.com/content_CVPR_2020/html/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.html).
- Binkowski et al.,
  [Demystifying MMD GANs](https://openreview.net/pdf?id=r1lUOzWCW), for the
  unbiased KID/MMD construction.
- Naeem et al.,
  [Reliable Fidelity and Diversity Metrics](https://proceedings.mlr.press/v119/naeem20a.html).

## 7. Cost and optimizer behavior

B2 incurred substantial compute:

- `1.71--1.73x` B0 wall time;
- approximately `3.87` total GPU hours for the three units;
- approximately `5.49 GB` peak allocated memory on the 6 GB device;
- 3,000 correction events and 24,000 correction-path model evaluations per
  unit;
- 8-step differentiable correction trajectories versus 32-step final
  evaluation trajectories.

The numerical event coefficient `lambda_event = 1.929e-4` is not evidence of a
weak correction. The raw loss sums 3,072 pixel coordinates and has very large
parameter gradients. At calibration events, the weighted correction gradient
was about `20--36%` of the flow-gradient norm; cadence reduces the nominal
average influence to roughly `2--4%`.

The weighted correction loss was about `2.6--2.8%` of flow loss at a median
logged event, yet held-out drift improved materially. The correction loss also
fell from first-quarter medians near `33` to last-quarter medians near
`23--24`, supporting a learned rather than purely accidental effect.

Approximately `19--23%` of logged correction events exceeded the gradient-clip
threshold. The calibration currently records separate pre-clip gradient norms
but not their cosine or the correction's post-clip update share. Future runs
should record:

- flow/correction gradient cosine;
- combined and component norms before clipping;
- post-clip component/update contributions where feasible;
- clipping frequency and magnitude;
- dimension-normalized field energy and probe-noise scale.

## 8. Prospective gate redesign

Do not use these revisions to retroactively change B2. Apply them only to new
experiments.

### 8.1 Mechanism gate

Require:

1. paired raw-energy reduction with a predeclared confidence interval below
   zero;
2. a minimum raw relative reduction;
3. floor-relative gap closure as a secondary, uncertainty-qualified quantity;
4. kernel-health quantiles over training and evaluation;
5. sufficient independent training units and paired audit replicates.

By this reading B2 improved the mechanism in all three units.

### 8.2 Promotion gate

Require separately:

1. paired precision and recall non-inferiority with uncertainty;
2. relative effective-rank retention, not only an absolute collapse floor;
3. KID plus density/coverage or other multiscale fidelity/diversity metrics;
4. direct comparison with the strongest incumbent, currently B1;
5. wall-time and memory Pareto reporting;
6. one in-domain and one shifted untouched evaluation source before a general
   image-generation claim.

The current absolute recall margin `0.025` permits a relatively large loss when
baseline recall is only `0.11--0.19`. A paired bootstrap or a predeclared
relative margin would be more informative. Sampling uncertainty for PR metrics
should be reported explicitly; see Urlus et al.,
[Pointwise sampling uncertainties on the Precision-Recall curve](https://proceedings.mlr.press/v206/urlus23a.html).

## 9. Recommended B2.5 factorial before B3

Yes: run a small prospective controlled factorial before committing to B3.
The purpose is not to reconfirm B2 on consumed data. It is to determine whether
B1 and B2 solve complementary problems:

| Arm | Purpose |
|---|---|
| B0 | paired flow-only control |
| B1 | spectral/global-geometry anchor |
| B2 | normalized-Laplace drift correction |
| B1+B2 | test whether B1 preserves rank while B2 reduces drift |

### 9.1 Guardrails

1. Use new development units and new stochastic streams.
2. Do not use the consumed B2 confirmation allocation for selection or
   threshold choice.
3. Retrain all four arms under the same new paired units; do not compare a new
   combined arm only to historical point estimates.
4. Initially keep B1 and B2 coefficients, schedules, `ESS=0.60`, correction
   NFE, and total training budget fixed. This isolates the interaction.
5. Predeclare checkpoint evaluations, for example 10k, 20k, and 30k, to test
   whether effective-rank loss accumulates late. Do not select a checkpoint on
   confirmation data.
6. Measure paired raw drift, effective-rank ratio, precision, recall, KID,
   density/coverage, clipping, ESS quantiles, wall time, and memory.
7. Treat this as development. Freeze a new confirmation only if one arm gives a
   clear Pareto improvement.

### 9.2 Decision rule

The combined arm is promising only if it:

- preserves B2's paired drift reduction;
- materially restores rank relative to B2;
- does not lose recall or precision relative to the paired B0/B1 controls;
- has a defensible cost-quality tradeoff.

If B1+B2 succeeds, it supplies a much better-defined B3 starting point. If it
fails, the factorial still identifies whether the failure is objective
conflict, excess correction duration, or a bandwidth/cadence issue. Moving
straight to B3 would otherwise compound mechanisms without resolving the most
important tradeoff exposed by B2.

### 9.3 Deferred bandwidth experiment

Do not combine the first factorial with a large bandwidth search. First test the
four arms at the already justified `ESS=0.60`. If B1+B2 is promising but still
compresses rank, run a separate development-only ESS ladder such as
`{0.40, 0.60, 0.80}`, or a pre-normalized two-scale mixture, while holding the
average gradient budget constant. This separation preserves causal
interpretability and avoids an unnecessarily large combinatorial experiment.

## 10. Bottom line

B2 provides strong evidence that the theory-aligned normalized-Laplace
correction is useful and trainable without an encoder. It also exposed a likely
global-geometry cost and a weakness in the current binary excess-ratio gate.
The correct next move is a small, newly paired B0/B1/B2/B1+B2 development
factorial before B3, followed by prospective gate redesign and a fresh freeze
only if the combined arm achieves a genuine Pareto improvement.
