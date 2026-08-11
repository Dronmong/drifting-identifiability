# S3 Failure Diagnosis and Repaired One-Step Research Plan

**Status:** post-run diagnosis and implementation specification; the one-unit
S3R screen has now completed and is interpreted in
[`EncoderIndependentS3RResultsAndBudgetDecision.md`](EncoderIndependentS3RResultsAndBudgetDecision.md)  
**Scope:** raw-pixel, encoder-free, exactly one-network-call generation  
**Artifact policy:** the completed S3 package and its hashed run artifacts are
frozen; all revisions live in the separate `stage_pmf_r` package.

## 1. What S3 actually established

The two S3 units were mechanically valid and scientifically unsuccessful.
Each unit trained for 60,000 optimizer updates / 960,000 examples on the 5,000
CIFAR-10 automobile training images.  The sampler used one neural-network
evaluation and no VAE, learned feature encoder, teacher, reference bank, or
iterative solver.  Both independent units converged to the same failure mode:
blurry automobile-like blobs with essentially zero coverage and an effective
output rank near 2.5.

This is useful negative evidence.  It rejects the tested objective/architecture
combination; it does not reject raw-pixel one-step generation in general.

## 2. Evidence from the saved run

### Endpoint dynamics

The EMA checkpoint sequence shows two distinct phases:

| checkpoint | effective rank | second-moment ratio | interpretation |
|---|---:|---:|---|
| 2k | about 93--95 | 0.006 | diverse only because outputs are almost zero |
| 10k | about 52 | 0.11--0.14 | amplitude begins to form |
| 30k | about 2.4--2.6 | 0.34--0.36 | geometric collapse |
| 60k | about 2.5--2.6 | 0.35--0.41 | collapse persists |

Rank must therefore be interpreted jointly with amplitude.  A high rank at a
second-moment ratio of 0.006 is not useful diversity.

An additional raw-versus-EMA check on unit 700 showed raw/EMA effective ranks
of 5.52/50.93 at 10k, 4.64/2.45 at 30k, and 3.78/2.48 at 60k.  EMA delayed the
visible collapse but did not cause it.

### Optimization conflict and instability

The recorded JVP magnitude and raw MeanFlow error are strongly associated
(log-scale correlations 0.802 and 0.835 in the two units), and both develop
large late tails.  Every recorded gradient norm exceeded the old clip value
of 1.0, so the optimizer operated in the clipped regime at all recorded
points.

Direct diagnostic gradients also reproduce AlphaFlow's central observation.
For unit 700, the trajectory-flow-matching versus trajectory-consistency
gradient cosine was negative on 5/6 batches at 10k, 5/6 at 30k, and 3/6 at
60k.  The median values were -0.287, -0.353, and -0.008.  This is a more
specific diagnosis than “the JVP is noisy”: two legitimate parts of the
continuous MeanFlow objective often ask the parameters to move in opposing
directions.

The main and auxiliary losses themselves had positive gradient cosine in a
small spot check.  The auxiliary branch was therefore not simply fighting the
main head.  Its deeper problem was representational: it shared the complete
backbone and owned only a 18,480-parameter output projection.

## 3. Architecture fidelity gap

The local S3 model was a useful resource-scaled test, not a faithful small
version of the strongest published pixel MeanFlow:

- local: 12 transformer blocks, width 384, 64 image tokens, batch 16,
  AdamW at `1e-4`, clip 1, and an output-only auxiliary branch;
- released pMF: a substantially larger trunk and genuinely deep unshared
  velocity branch, 256 image tokens, much larger batches, and a tuned Muon
  setup;
- published pMF quality also depends strongly on a perceptual loss.  The
  reported MSE-only result is much worse than the perceptually supervised
  result.

The perceptual dependency is intentionally not imported here: doing so would
abandon the encoder-independent question.  The repaired experiment instead
targets the optimization and branch-capacity defects that can be fixed without
a learned representation.

## 4. Literature-supported repair mechanisms

### 4.1 Discrete AlphaFlow curriculum

AlphaFlow decomposes MeanFlow into trajectory flow matching (TFM) and
trajectory consistency (TC), measures their negative gradient correlation,
and replaces immediate joint optimization with a curriculum.  For local time
convention `z_t=(1-t)x+t*noise`, it sets

\[
s=\alpha r+(1-\alpha)t,\qquad
z_s=z_t-(t-s)(noise-x),
\]

and trains

\[
u_\theta(z_t,r,t)\approx
\alpha(noise-x)+(1-\alpha)
\operatorname{sg}(u_\theta(z_s,r,s)).
\]

`alpha=1` is TFM.  Lower alpha progressively introduces trajectory
consistency.  The paper reports `0.005` as its best small discrete ratio.  Our
screen deliberately floors alpha at `0.005` rather than switching to the
continuous alpha=0/JVP objective; this keeps the arm JVP-free and tests whether
the observed conflict can be avoided before another expensive continuous run.

### 4.2 x1-prediction Euler Mean Flow

Euler Mean Flow replaces the JVP by a stopped local finite difference.  Its
x1-prediction form is especially relevant because the paper reports that
raw-pixel velocity prediction fails while direct endpoint prediction succeeds.
After reversing the paper's time orientation to this repository's
data-at-zero/noise-at-one convention, a local step `delta` uses

\[
z_{t-\delta}'=z_t+\frac{\delta}{t}
(\hat x_\theta(z_t,t,0)-z_t)
\]

and, when `t-r>delta`, the detached target

\[
x + (t-r-\delta)\frac{t}{r}
\frac{\hat x_\theta(z'_{t-\delta},t-\delta,r)
-\hat x_\theta(z_t,t,r)}{\delta}.
\]

Short/diagonal intervals reduce to direct endpoint regression.  The code must
be checked in float64 against the corresponding local JVP before training;
the arXiv HTML pseudocode contains obvious transcription errors, so the
numbered equations, not the rendered pseudocode, are the source of truth.

### 4.3 Deep auxiliary separation

The repaired continuous-pMF control receives genuinely unshared transformer
blocks before its auxiliary prediction head.  The auxiliary branch remains
training-only and cannot add an inference call.  This tests the capacity gap
without conflating it with a learned encoder.

## 5. Repaired experiment: S3R

### R0: diagnostics and mathematical preflight

Before any long run, require:

1. alpha=1 reduces exactly to TFM, diagonal rows remain TFM throughout the
   curriculum, the released per-pixel adaptive epsilon is `0.001`, and the
   alpha schedule is monotone with a `0.005` lower bound;
2. the EMF finite difference agrees with the corresponding directional JVP on
   a float64 test model;
3. the auxiliary branch has nonzero gradients, contains real unshared blocks,
   and is absent from one-call inference;
4. raw and EMA endpoint health are evaluated on the same sealed train-only
   noise;
5. endpoint second-moment ratio and effective rank are reported together;
6. diagonal/interior error, per-sample JVP p50/p90/max, TFM/TC gradient cosine,
   pre-clip norm, clipping fraction, and Haar-band energy/rank are available.

### R1: matched developmental screen

Use three arms with the same data, initialization family, inference backbone,
optimizer, effective batch, example budget, and sealed train-only diagnostics:

| arm | repair being isolated |
|---|---|
| `pmf` | continuous pMF plus a genuinely deep auxiliary branch |
| `alpha` | discrete AlphaFlow curriculum, held at alpha >= 0.005 |
| `emf` | direct-x1 Euler Mean Flow with a stopped local difference, the paper's reversed-clock `1/t^2` time weight, and `0.02` denominator floor |

The initial screen budget is effective batch 64 and 800,000 examples, i.e.
12,500 optimizer updates with four microbatches of 16.  Use AdamW `1e-4` for
the matched mechanism screen.  Raise the clip ceiling from 1 to 10 and log the
actual clipping fraction; do not silently disable or continuously trigger it.
Muon is a later optimizer ablation, not part of mechanism attribution.

No CIFAR test image, Inception feature, FID, KID, or human grid is used to pick
the developmental arm.  A candidate may advance only if train-only fixed
diagnostics show:

- second-moment and centered-variance ratios at least 0.5;
- raw and EMA effective rank at least 60% of the fixed target rank once the
  moment ratio exceeds 0.15;
- no persistent midpoint-to-end rank collapse;
- clipping on fewer than 5% of updates;
- stable JVP/error tails for the continuous arm;
- improving diagonal and interior objectives.

These are development gates, not final evidence of image quality.

### R2: sealed confirmation

Take only the R1 winner into two fixed independent units.  Freeze all choices
before accessing the official test split.  Then run the same evaluation and
memorization suite as S3 and require both units to be recognizable,
noncollapsed, and nonmemorizing.  A favorable FID alone cannot promote it.

### R3: spatial fidelity, only after objective stability

If R1 fixes rank/amplitude but images remain blurry, then test a finer token
grid or a lightweight local decoder and fixed, invertible multiscale losses.
A full orthonormal Haar transform with equal weights is exactly an L2 control;
only unequal positive subband weights change the objective.  No learned
feature encoder is introduced.

### R4: drifting correction

Only a successful one-step foundation proceeds to the already designed S4
Sinkhorn/drifting fork.  Attaching a distribution correction to a collapsed
foundation would not identify whether the correction or generator failed.

## 6. Claim boundary

This plan does not claim that AlphaFlow or EMF will solve the task.  It turns
the observed S3 failure into three falsifiable mechanism tests, preserves
strict one-call inference and encoder independence, and adds early diagnostics
capable of stopping another 60k-update collapse before test evaluation.

## 7. Implementation checkpoint (2026-08-03)

The R0/R1 foundation is implemented in
`numerics/encoder_independent_drifting/stage_pmf_r/` without modifying the
frozen S3 source or result artifacts:

- `model.py` adds four real unshared auxiliary transformer blocks while
  keeping the auxiliary path out of the one-call sampler;
- `objectives.py` implements the three arms, the floored AlphaFlow schedule,
  and the time-reversed direct-x1 EMF Equation 18, including its explicit
  denominator clamp and x1/u time weight;
- `diagnostics.py` implements amplitude-aware raw/EMA endpoint health,
  TFM/TC gradient cosine, JVP/error quantiles, and orthonormal Haar-band
  energy/rank reports;
- `training.py` implements the matched 800k-example screen, clip-frequency
  accounting, fixed train-only health samples, checkpointing, and exact
  resume;
- `preflight.py` checks the objective identities, EMF finite-difference/JVP
  agreement, Haar energy conservation, auxiliary/inference separation, all
  three backward passes, full-shape GPU memory, and projected runtime;
- `run_screen.py` is blocked unless a source-matched **developmental**
  preflight exists and the operator explicitly opts into the run.  A smoke
  preflight cannot unlock it.

## 8. Final pre-launch audit (2026-08-03)

The objective equations were checked again against the primary papers and,
where available, released source.  This found and repaired three issues before
the developmental GPU run:

1. The released AlphaFlow loss uses per-sample mean squared error with
   `adaptive_loss_weight_eps = 0.001`.  Its `r=t` rows retain numerator one and
   ordinary flow-matching targets throughout the curriculum; only interior
   discrete rows receive numerator alpha.  The first S3R draft incorrectly
   applied alpha to every row and shared pMF's sum-based `0.01` weighting.
2. Direct-x1 EMF Equation 18 is correct after the clock substitution
   `t_paper = 1 - t_local`, `r_paper = 1 - r_local`.  The same substitution
   turns the paper's `1/(1-t_paper)^2` x1/u loss weight into `1/t_local^2`, and
   its `0.02` clamps on `1-t_paper` and `1-r_paper` into clamps on local `t`
   and `r`.  These practical factors were absent from the first draft.
3. A nonzero constant image can clear a second-moment threshold while having
   no sample diversity.  Rank interpretability and the developmental gate now
   depend on centered variance as well as second moment.

The audit also confirmed that all arms use the same CIFAR-10 automobile train
pool, initialization seed family, stochastic data/noise/time streams,
optimizer, effective batch, and 800,000-example budget.  The S3R loader opens
only the official training archive; the test split is not even instantiated by
the developmental process.  All inference paths make exactly one model call.
Absolute-time conditioning is enabled only for the two methods whose
fields require it; the continuous pMF control preserves the previously audited
interval-only derivative.

One matched unit across the three arms is enough to answer the narrow
development question, "did any repair avoid the known amplitude/rank
collapse?"  It is not a statistical confirmation of image quality or method
superiority.  Any passing arm still needs the two-unit sealed confirmation in
R2.

Primary references used in this audit:

- AlphaFlow paper: <https://arxiv.org/html/2510.20771>
- AlphaFlow released loss and configuration:
  <https://github.com/snap-research/alphaflow>
- Euler Mean Flow paper, especially Equations 15--18 and Appendix C.1:
  <https://arxiv.org/html/2602.02571>

On the local RTX 4050 6 GB preflight, all three production-shaped microbatches
and their AdamW optimizer steps fit.  Across repeated short preflights, the
timing projection was about 0.8--0.9 h for AlphaFlow, 1.1--1.3 h for EMF, and
3.1--3.5 h for continuous pMF at the declared 12,500-update budget.  These are
short-kernel estimates, not guaranteed wall-clock run times.  The continuous
arm used about 2.45 decimal GB peak allocated memory; the two JVP-free arms
used about 0.84 GB.  This confirms that the full factorial is locally
practical while quantifying the large cost of retaining the continuous JVP.

No developmental or confirmatory training arm has been run.  The next action
is to review the green full-shape preflight, then run R1 one arm at a time.

## Primary sources

- Zhang et al., [AlphaFlow: Understanding and Improving MeanFlow
  Models](https://arxiv.org/html/2510.20771), especially Sections 3--4 and
  Algorithms 1--2.
- [Official AlphaFlow implementation](https://github.com/snap-research/alphaflow).
- Liu et al., [Euler Mean Flow](https://arxiv.org/html/2602.02571), especially
  Equations 15--19 and the pixel-space JiT experiment.
- Lu et al., [One-step Latent-free Image Generation with Pixel Mean
  Flows](https://arxiv.org/html/2601.22158v3) and the
  [official pMF code](https://github.com/Lyy-iiis/pMF).
- Li and He, [PixelDiT](https://arxiv.org/html/2511.20645), for the direct-pixel
  prediction and data-manifold motivation.
